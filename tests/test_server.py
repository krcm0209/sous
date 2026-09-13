import asyncio
import io
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import SousService
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine


@pytest.fixture(autouse=True)
def _no_leaked_warning_capture():
    """server.main() turns logging.captureWarnings on for the process. Left on,
    a later captureWarnings(True) anywhere in the suite is a no-op once pytest
    has restored showwarning — and that test passes or fails by file order."""
    yield
    logging.captureWarnings(False)


@pytest.fixture()
def svc(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    cfg.config_path.write_text('[commands]\nallowlist = ["pytest"]\n')
    store = TaskStore(tmp_path / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    return SousService(store, engines, cfg), store, root


def test_delegate_returns_id_and_position(svc):
    service, store, root = svc
    out = service.delegate_task("t", "do it", str(root))
    assert "task_id" in out and out["queue_position"] == 1
    assert store.get(out["task_id"]).state == "queued"


def test_delegate_rejects_relative_root(svc):
    service, _, _ = svc
    assert "error" in service.delegate_task("t", "x", "relative/path")


def test_delegate_rejects_missing_root(svc):
    service, _, _ = svc
    assert "error" in service.delegate_task("t", "x", "/nope/not/here")


def test_delegate_rejects_root_containing_data_dir(svc, tmp_path: Path):
    """C1 (MCP boundary): a project_root that is an ancestor of the sous
    data dir would put config.toml/tasks.db inside the sandbox."""
    service, _, _ = svc
    out = service.delegate_task("t", "x", str(tmp_path))  # tmp_path contains data_dir
    assert "error" in out and "data dir" in out["error"]


def test_delegate_rejects_root_equal_to_data_dir(svc, tmp_path: Path):
    service, _, _ = svc
    (tmp_path / "data").mkdir()
    out = service.delegate_task("t", "x", str(tmp_path / "data"))
    assert "error" in out and "data dir" in out["error"]


def test_delegate_accepts_sibling_of_data_dir(svc):
    service, _, root = svc
    assert "task_id" in service.delegate_task("t", "x", str(root))


def _fs_is_case_insensitive(tmp_path: Path) -> bool:
    probe = tmp_path / "sous_case_probe_x"
    probe.write_text("x")
    hit = (tmp_path / "sous_case_probe_X").exists()
    probe.unlink()
    return hit


def test_delegate_rejects_case_variant_root_containing_data_dir(tmp_path: Path):
    """C1 (case bypass): a case-variant ancestor of the data dir must be
    rejected just like the exact-case ancestor, or the boundary check is
    defeated on a case-insensitive FS."""
    if not _fs_is_case_insensitive(tmp_path):
        pytest.skip("requires a case-insensitive filesystem (APFS/HFS+)")
    home = tmp_path / "home"
    home.mkdir()
    data = home / "data"
    data.mkdir()
    cfg = SousConfig(data_dir=data, config_path=home / "config.toml")
    store = TaskStore(tmp_path / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    service = SousService(store, engines, cfg)
    # exact-case ancestor is rejected (sanity)
    assert "error" in service.delegate_task("t", "x", str(home))
    # case-variant ancestor must be rejected the same way
    out = service.delegate_task("t", "x", str(tmp_path / "HOME"))
    assert "error" in out and "data dir" in out["error"]


def test_delegate_rejects_non_allowlisted_verify(svc):
    service, _, root = svc
    out = service.delegate_task("t", "x", str(root), verify_commands=["rm -rf /"])
    assert "error" in out and "rm -rf /" in out["error"]


def test_delegate_accepts_allowlisted_verify(svc):
    service, _, root = svc
    out = service.delegate_task("t", "x", str(root), verify_commands=["pytest -q"])
    assert "task_id" in out


def test_delegate_unparseable_verify_command_returns_error(svc):
    """C4: an unmatched quote in a client-supplied verify_command must come
    back as a structured error, not raise ValueError out of the service."""
    service, _, root = svc
    out = service.delegate_task("t", "x", str(root), verify_commands=['echo "unclosed'])
    assert "error" in out and 'echo "unclosed' in out["error"]


def test_status_single_and_all(svc):
    service, store, root = svc
    a = service.delegate_task("a", "x", str(root))["task_id"]
    service.delegate_task("b", "x", str(root))
    one = service.task_status(a)
    assert one["state"] == "queued" and one["queue_position"] == 1
    both = service.task_status()
    assert len(both["tasks"]) == 2


def test_status_unknown_id(svc):
    service, _, _ = svc
    assert "error" in service.task_status("nope")


def test_result_not_ready_then_ready(svc):
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert "error" in service.task_result(tid)
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": []})
    out = service.task_result(tid)
    assert out["outcome"] == "completed" and out["report"]["summary"] == "s"


def test_result_include_diff_from_git(svc, tmp_path: Path):
    service, store, root = svc
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "f.txt").write_text("one\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    (root / "f.txt").write_text("two\n")
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": [{"path": "f.txt"}]})
    out = service.task_result(tid, include_diff=True)
    assert "-one" in out["diff"] and "+two" in out["diff"]


def test_result_include_diff_from_git_worktree(svc, tmp_path: Path):
    """A git *worktree* has a `.git` FILE (gitdir pointer), not a directory —
    the diff must still be produced there (and in submodules / subdirectories
    of a repo), which is why the repo check must be `git rev-parse
    --is-inside-work-tree` rather than `.git`-is-a-directory."""
    service, store, root = svc
    main_repo = tmp_path / "main_repo"
    main_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main_repo, check=True)
    (main_repo / "f.txt").write_text("one\n")
    subprocess.run(["git", "add", "."], cwd=main_repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=main_repo,
        check=True,
    )
    worktree = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", str(worktree)], cwd=main_repo, check=True)
    assert not (worktree / ".git").is_dir()  # sanity: it's a file, not a dir
    (worktree / "f.txt").write_text("two\n")
    tid = service.delegate_task("t", "x", str(worktree))["task_id"]
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": [{"path": "f.txt"}]})
    out = service.task_result(tid, include_diff=True)
    assert "-one" in out["diff"] and "+two" in out["diff"]


def _git_repo_with_commit(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "f.txt").write_text("one\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )


def test_result_include_diff_shows_created_untracked_file(svc):
    """D1: `git diff -- <paths>` omits untracked files, and a worker CREATING
    a file is the most common action — the diff must include the new file's
    contents, or 'Claude always reviews the diff' silently fails for exactly
    the case the review workflow exists to cover."""
    service, store, root = svc
    _git_repo_with_commit(root)
    (root / "made.txt").write_text("brand new line\n")  # untracked
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": [{"path": "made.txt"}]})
    out = service.task_result(tid, include_diff=True)
    assert out["diff"] is not None
    assert "made.txt" in out["diff"]
    assert "+brand new line" in out["diff"]


def test_result_include_diff_shows_modified_and_created_together(svc):
    service, store, root = svc
    _git_repo_with_commit(root)
    (root / "f.txt").write_text("two\n")  # modified, tracked
    (root / "made.txt").write_text("brand new line\n")  # created, untracked
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.finish(
        tid,
        "completed",
        {"summary": "s", "files_changed": [{"path": "f.txt"}, {"path": "made.txt"}]},
    )
    out = service.task_result(tid, include_diff=True)
    assert "-one" in out["diff"] and "+two" in out["diff"]
    assert "+brand new line" in out["diff"]


def test_result_include_diff_non_repo_returns_none(svc):
    service, store, root = svc  # root is a plain directory, no git repo
    (root / "made.txt").write_text("brand new line\n")
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": [{"path": "made.txt"}]})
    out = service.task_result(tid, include_diff=True)
    assert out["diff"] is None


def test_cancel(svc):
    service, _, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert service.cancel_task(tid)["cancelled"] is True
    assert "error" in service.cancel_task("nope")


def test_respond_requires_awaiting(svc):
    service, _, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert "error" in service.respond_to_command_request(tid, approve=True)


def test_respond_approves_and_persists(svc):
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.request_approval(tid, "go vet ./...")
    out = service.respond_to_command_request(tid, approve=True, persist_to_allowlist=True)
    assert out["ok"] is True
    from sous.config import current_allowlist

    assert ["go", "vet", "./..."] in current_allowlist(service.config.config_path)
    assert store.poll_approval(tid) == "approved"


def test_persisting_a_cd_prefixed_command_matches_the_next_request(svc):
    """Approve-and-persist on `cd <dir> && <cmd>` must persist the canonical
    form the allowlist actually matches (the stripped remainder), so the same
    cd-prefixed request runs next time WITHOUT prompting. Persisting the raw
    `cd ... && ...` would write an entry that can never match the stripped
    argv, and the 'remembered' approval would keep asking."""
    from sous.config import current_allowlist
    from sous.toolexec import ToolExecutor, command_allowed

    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.request_approval(tid, "cd sub && go vet ./...")
    out = service.respond_to_command_request(tid, approve=True, persist_to_allowlist=True)
    assert out["ok"] is True
    # persisted the remainder, not the cd-prefixed string
    allow = current_allowlist(service.config.config_path)
    assert ["go", "vet", "./..."] in allow
    assert not any(entry and entry[0] == "cd" for entry in allow)
    # and the same request now passes the allowlist that run_command applies
    ex = ToolExecutor(root, service.config.config_path)
    (ex.project_root / "sub").mkdir()
    from sous.toolexec import normalize_cd_prefix

    cmd = "cd sub && go vet ./..."
    argv = normalize_cd_prefix(cmd, ["cd", "sub", "&&", "go", "vet", "./..."], ex.project_root)
    assert command_allowed(argv, current_allowlist(service.config.config_path))


def test_respond_race_does_not_persist_allowlist(svc, monkeypatch):
    """M2: if the approval races a timeout-deny (respond_approval returns
    False), the command must NOT already be persisted to the allowlist."""
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.request_approval(tid, "go vet ./...")
    before = service.config.config_path.read_text()
    monkeypatch.setattr(store, "respond_approval", lambda task_id, approve: False)
    out = service.respond_to_command_request(tid, approve=True, persist_to_allowlist=True)
    assert out["ok"] is False
    assert service.config.config_path.read_text() == before
    from sous.config import current_allowlist

    assert ["go", "vet", "./..."] not in current_allowlist(service.config.config_path)


def test_second_respond_cannot_reverse_persisted_approval(svc):
    """A3 (service boundary): approve-with-persist writes the allowlist entry;
    a later deny must NOT win the approval state, or the command would be left
    permanently allowlisted while reported as denied."""
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.request_approval(tid, "go vet ./...")
    first = service.respond_to_command_request(tid, approve=True, persist_to_allowlist=True)
    assert first["ok"] is True
    second = service.respond_to_command_request(tid, approve=False)
    assert second["ok"] is False  # first response already landed
    assert store.poll_approval(tid) == "approved"


def test_server_status(svc):
    service, _, root = svc
    service.delegate_task("t", "x", str(root))
    s = service.server_status()
    assert s["queue"]["queued"] == 1
    assert s["model"]["loaded"] is False
    assert s["config"]["model_id"]
    assert ["pytest"] in s["config"]["allowlist"]


def test_server_status_counts_past_200_tasks(svc):
    """E2: queue depth came from list_recent(limit=200), so more than 200
    active tasks under-reported — the count must cover every row."""
    service, store, root = svc
    for i in range(205):
        store.enqueue(
            title=f"t{i}",
            instructions="x",
            project_root=str(root),
            context_files=[],
            verify_commands=[],
        )
    store.claim_next()
    s = service.server_status()
    assert s["queue"]["queued"] == 204
    assert s["queue"]["running"] == 1


def test_create_server_registers_six_tools(svc):
    service, store, root = svc
    from sous.server import create_server

    mcp = create_server(store, service.engines, service.config)
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "delegate_to_local_model",
        "task_status",
        "task_result",
        "cancel_task",
        "respond_to_command_request",
        "server_status",
    }


def test_create_server_sets_discovery_instructions(svc):
    """Server-level instructions are the one part of the MCP surface clients
    put in front of the model unconditionally — tool descriptions can be
    deferred out of context, leaving only names. Discovery therefore lives or
    dies on instructions being present, naming the delegate tool, and fitting
    under Claude Code's 2KB truncation limit (an overlong block silently loses
    its own ending)."""
    service, store, root = svc
    from sous.server import create_server

    mcp = create_server(store, service.engines, service.config)
    assert mcp.instructions
    assert "delegate_to_local_model" in mcp.instructions
    assert len(mcp.instructions.encode()) < 2048


def test_delegate_tool_description_carries_the_motive(svc):
    """The description lists what QUALIFIES for delegation; it must also say
    why delegation beats doing the work inline (which the model always can) —
    the plan-economics rationale is the tie-breaker at tool-selection time."""
    service, store, root = svc
    from sous.server import create_server

    mcp = create_server(store, service.engines, service.config)
    tools = asyncio.run(mcp.list_tools())
    delegate = next(t for t in tools if t.name == "delegate_to_local_model")
    assert "plan" in (delegate.description or "").lower()


# --- only one daemon may run: a second would fail the first's in-flight task
# --- via recover_interrupted() and load a second copy of the model.


def test_second_daemon_exits_without_touching_the_queue(tmp_path: Path, monkeypatch):
    """The lock must be taken before ANY queue access.

    server.main() opens the TaskStore and calls recover_interrupted() — which
    marks the first daemon's RUNNING task failed — then starts a worker that
    can load a second model. All of that happens before mcp.run() binds the
    port, so the bind is far too late to be the guard. If the lock is acquired
    first, tasks.db is never even created.
    """
    from sous.server import _acquire_singleton_lock
    from sous.server import main as serve_main

    data = tmp_path / "data"
    data.mkdir()

    # Hold the port for the whole test: without the lock, main() would reach
    # mcp.run() and serve forever instead of failing, hanging the suite. An
    # occupied port makes the unlocked path terminate so the assertion below
    # is what distinguishes pass from fail.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / "config.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")

        holder = _acquire_singleton_lock(data)  # stand-in for the live daemon
        try:
            monkeypatch.setattr("sous.server.load_config", lambda: cfg)
            with pytest.raises(SystemExit):
                serve_main()
            assert not (data / "tasks.db").exists(), "second daemon opened the shared queue"
        finally:
            holder.close()


def test_lock_excludes_a_separate_process(tmp_path: Path):
    """flock, not a pidfile: the kernel drops it when the holder dies, so a
    SIGKILLed daemon can never wedge the next start out of the lock."""
    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()

    probe = (
        "import sys; from pathlib import Path;"
        "from sous.server import _acquire_singleton_lock;"
        f"_acquire_singleton_lock(Path({str(data)!r}));"
        "print('ACQUIRED')"
    )
    holder = _acquire_singleton_lock(data)
    try:
        blocked = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert blocked.returncode != 0, "second process acquired a held lock"
        assert "ACQUIRED" not in blocked.stdout
    finally:
        holder.close()

    # holder released -> the next process gets it
    freed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert "ACQUIRED" in freed.stdout, freed.stderr


def test_non_contention_lock_failure_is_not_reported_as_another_daemon(tmp_path: Path, monkeypatch):
    """ENOLCK, or a filesystem without flock support, is a real startup failure.

    Contention is EAGAIN (EWOULDBLOCK). Translating every OSError into "another
    daemon already holds ..." would send the operator hunting for a process that
    does not exist, hiding the actual cause.
    """
    import errno
    import fcntl

    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()

    def no_locks_available(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks_available)
    with pytest.raises(OSError) as exc:
        _acquire_singleton_lock(data)
    assert exc.value.errno == errno.ENOLCK


def test_lock_file_records_the_holder_pid(tmp_path: Path):
    """The lock file names its holder, so a client can tell a restarted daemon
    from the one it connected to — a port check alone cannot."""
    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()
    holder = _acquire_singleton_lock(data)
    try:
        assert (data / "daemon.lock").read_text().strip() == str(os.getpid())
    finally:
        holder.close()


def test_main_installs_the_shutdown_handler_before_serving(tmp_path: Path, monkeypatch):
    """The handler is useless if main() never installs it.

    Everything else about shutdown is covered in tests/test_commands.py; this
    pins the one-line link between them, which no other test would catch.
    """
    import sous.server as server

    data = tmp_path / "data"
    data.mkdir()
    installed = []
    with socket.socket() as occupied:  # make mcp.run() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / "config.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        monkeypatch.setattr(
            server, "_install_shutdown_handler", lambda stop: installed.append(stop)
        )
        with pytest.raises(SystemExit):
            server.main()
    assert installed, "main() served without installing the shutdown handler"


def test_main_routes_config_load_warnings_through_the_daemon_shape(
    tmp_path: Path, monkeypatch, caplog
):
    """load_config() runs inside main(), which must already have installed the
    warnings-to-logging redirect by the time it does — otherwise a config typo
    reaches stderr as a raw `UserWarning`, unshaped and unleveled, while every
    other line the daemon writes carries a timestamp and a level.
    """
    import dataclasses
    import logging

    import sous.server as server
    from sous.config import load_config as real_load_config

    data = tmp_path / "data"
    data.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text("[nonsense]\n")  # unknown section: load_config warns

    with socket.socket() as occupied:  # make mcp.run() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)

        def fake_load_config():
            # The real parser, so the warning is genuine — only the paths
            # (never the real ~/.sous) and port are stood in for.
            cfg = real_load_config(config_path)
            return dataclasses.replace(cfg, data_dir=data, server_port=occupied.getsockname()[1])

        monkeypatch.setattr(server, "load_config", fake_load_config)
        # logging.captureWarnings(True) only rebinds warnings.showwarning the
        # first time it is called in a process — a no-op on every later call
        # while it thinks capture is already on. An earlier test in this same
        # file already left it "on" from a stale rebinding, so reset it here
        # (and restore that reset afterward) rather than depend on suite order.
        logging.captureWarnings(False)
        try:
            with caplog.at_level("WARNING"), pytest.raises(SystemExit):
                server.main()
        finally:
            logging.captureWarnings(False)

    warning_records = [r for r in caplog.records if r.name == "py.warnings"]
    assert warning_records, "config warning never reached logging as py.warnings"
    assert "unknown section" in warning_records[0].getMessage()


# --- login-shell PATH resolution ------------------------------------------------


def test_login_shell_path_extracted_despite_noisy_init_files(tmp_path: Path, monkeypatch):
    """Login shells are entitled to print banners from init files; the probe's
    markers must keep that noise out of the captured PATH."""
    from sous.server import _login_shell_path

    fake = tmp_path / "shell"
    fake.write_text(
        '#!/bin/sh\necho "welcome banner"\nPATH="/fake/tools:/usr/bin"\nexport PATH\neval "$4"\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("SHELL", str(fake))
    assert _login_shell_path() == "/fake/tools:/usr/bin"


def test_login_shell_path_none_when_the_shell_is_broken(monkeypatch):
    """An exotic or missing shell must mean fallback, never a crashed daemon."""
    from sous.server import _login_shell_path

    monkeypatch.setenv("SHELL", "/nonexistent/shell")
    assert _login_shell_path() is None


def test_main_adopts_the_login_shell_path_before_serving(tmp_path: Path, monkeypatch):
    """Under launchd the daemon inherits the bare system PATH, so allowlisted
    commands like `uv run pytest` stop resolving even though they work in the
    user's terminal. main() must adopt the login shell's PATH before the worker
    exists — scrubbed_env() passes PATH through, so this is what sandboxed
    commands resolve against."""
    import sous.server as server

    data = tmp_path / "data"
    data.mkdir()
    with socket.socket() as occupied:  # make mcp.run() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / "config.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        monkeypatch.setattr(server, "_login_shell_path", lambda: "/resolved/bin:/usr/bin")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")  # launchd-bare stand-in
        with pytest.raises(SystemExit):
            server.main()
    assert os.environ["PATH"] == "/resolved/bin:/usr/bin"


class _FakeStderr(io.StringIO):
    """A stream main() can log through (write/flush both work) whose isatty()
    is controlled by the test, unlike a real TextIOWrapper's."""

    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _run_main_against_an_occupied_port(tmp_path: Path, monkeypatch, *, subdir: str) -> None:
    """make serve() fail fast instead of actually serving, so main() runs its
    startup steps and exits via the bind failure."""
    import sous.server as server

    data = tmp_path / subdir
    data.mkdir()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / f"{subdir}.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        with pytest.raises(SystemExit):
            server.main()


def test_main_disables_progress_bars_only_when_stderr_is_not_a_tty(tmp_path: Path, monkeypatch):
    """Under launchd stderr is a pipe with nobody to animate a bar for; a
    `sous serve` run by hand for a first multi-GB model download has a real
    terminal, and the download's own progress should still show there."""
    calls: list[bool] = []
    monkeypatch.setattr("huggingface_hub.utils.disable_progress_bars", lambda: calls.append(True))

    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty=True))
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="tty")
    assert calls == []  # interactive: progress bars stay on

    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty=False))
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="no-tty")
    assert calls == [True]  # non-interactive: progress bars are silenced

    # fd 2 closed at exec (`sous serve 2>&-`): Python sets sys.stderr to None,
    # and the daemon must still start rather than die silently on isatty().
    monkeypatch.setattr(sys, "stderr", None)
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="no-stderr")
    assert calls == [True, True]


def test_server_status_reports_context_policy(svc):
    """Users need to see which sizing policy the daemon is running without
    reading config files — this is the MCP-visible surface for it."""
    service, _, _ = svc
    ctx = service.server_status()["config"]["context"]
    # max_context_tokens included so fixed mode reports its operative value,
    # not just the fact that the policy is fixed.
    assert ctx == {
        "mode": "fixed",
        "fraction": 0.8,
        "min_tokens": 8192,
        "max_context_tokens": 32768,
    }


def test_status_memory_probe_releases_mlx_thread_state(svc, monkeypatch):
    """server_status runs in whatever short-lived worker thread the MCP layer
    hands it; any mlx state it created must be released before that thread can
    exit (ml-explore/mlx#4327)."""
    import sous.server as server

    service, _, _ = svc
    released = []
    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: released.append(True))
    service.server_status()
    assert released


def test_server_status_reports_gateway_config(svc):
    """The gateway is off by default and experimental; the MCP-visible status
    is how a user confirms which model ids the daemon would serve locally."""
    service, _, _ = svc
    gw = service.server_status()["config"]["gateway"]
    assert gw == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }


def test_uvicorn_config_bounds_graceful_shutdown():
    """uvicorn holds SIGTERM while serving and, unbounded, waits for open
    connections — a long non-streaming gateway turn would defer sous's own
    shutdown handler by minutes. The bound is what keeps `sous stop` prompt."""
    from sous.server import GRACEFUL_SHUTDOWN_SECONDS, uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_SECONDS == 5
    assert cfg.host == "127.0.0.1"
    # uvicorn's access log prints the raw request target, query string
    # included, at INFO — the daemon's own level.
    assert cfg.access_log is False


def test_create_server_installs_the_daemon_log_handler_and_drops_the_rich_one(svc):
    import logging

    from sous.logs import SOUS_HANDLER_NAME
    from sous.server import create_server

    class RichHandler(logging.Handler):  # the SDK's, matched by class name
        def emit(self, record: logging.LogRecord) -> None:
            pass

    service, store, _ = svc
    root = logging.getLogger()
    root.addHandler(RichHandler())
    try:
        create_server(store, service.engines, service.config)
        names = [h.get_name() for h in root.handlers]
        assert names.count(SOUS_HANDLER_NAME) == 1
        assert not [h for h in root.handlers if type(h).__name__ == "RichHandler"]
    finally:
        for h in list(root.handlers):
            if h.get_name() == SOUS_HANDLER_NAME or type(h).__name__ == "RichHandler":
                root.removeHandler(h)


def test_uvicorn_config_leaves_logging_to_the_daemon():
    """uvicorn's default dictConfig gives its loggers their own handlers and
    stops propagation, so its lines would keep the `INFO:     …` shape
    instead of the daemon's. log_config=None lets them reach the root."""
    from sous.server import uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.log_config is None
    assert cfg.access_log is False
