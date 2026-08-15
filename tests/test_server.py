import asyncio
import subprocess
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import SousService
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine


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
        "delegate_task",
        "task_status",
        "task_result",
        "cancel_task",
        "respond_to_command_request",
        "server_status",
    }
