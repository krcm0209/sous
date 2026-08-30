import os
import resource
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import sous.toolexec as toolexec
from sous.toolexec import (
    MAX_TOOL_OUTPUT,
    ToolExecutor,
    _kill_process_group,
    command_allowed,
    scrubbed_env,
)


@pytest.fixture(autouse=True)
def _reset_shutdown_latch(monkeypatch):
    """terminate_active_commands() latches shutdown for the whole module.

    Without this, any test that calls it silently turns every later
    run_command in the session into "aborted: the daemon is shutting down" —
    a failure that would look like a bug in the command under test.
    """
    monkeypatch.setattr(toolexec, "_registration_closed", False)


def test_allowlist_token_matching():
    allow = [["npm", "test"], ["pytest"]]
    assert command_allowed(["npm", "test"], allow)
    assert command_allowed(["npm", "test", "--workspaces"], allow)
    assert command_allowed(["pytest", "tests/x.py", "-v"], allow)
    assert not command_allowed(["npm", "testx"], allow)
    assert not command_allowed(["npm", "install"], allow)
    assert not command_allowed(["npm"], allow)  # shorter than entry
    assert not command_allowed([], allow)


def test_scrubbed_env(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", "x")
    monkeypatch.setenv("AWS_SECRET", "x")
    monkeypatch.setenv("DB_PASSWORD", "x")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = scrubbed_env()
    assert env["PATH"] == "/usr/bin"
    assert "MY_API_TOKEN" not in env
    assert "AWS_SECRET" not in env
    assert "DB_PASSWORD" not in env


@pytest.fixture()
def ex(tmp_path: Path) -> ToolExecutor:
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[commands]\nallowlist = ["/bin/echo", "true"]\n')
    return ToolExecutor(root, cfg)


def test_allowlisted_command_runs(ex: ToolExecutor):
    out = ex.run_command("/bin/echo hello")
    assert "exit code 0" in out and "hello" in out


def test_shell_metacharacters_inert(ex: ToolExecutor, tmp_path: Path):
    out = ex.run_command("/bin/echo hi; touch pwned")
    assert "exit code 0" in out
    assert "hi; touch pwned" in out  # echoed literally
    assert not (tmp_path / "proj" / "pwned").exists()


def test_denied_without_hook(ex: ToolExecutor):
    out = ex.run_command("rm -rf /")
    assert out.startswith("command denied")


def test_hook_approves(ex: ToolExecutor):
    out = ex.run_command("/usr/bin/printf ok", approval=lambda cmd: True)
    assert "exit code 0" in out and "ok" in out


def test_hook_denies(ex: ToolExecutor):
    out = ex.run_command("/usr/bin/printf no", approval=lambda cmd: False)
    assert out.startswith("command denied")


def test_allowlist_hot_reload(ex: ToolExecutor):
    assert ex.run_command("/usr/bin/true").startswith("command denied")
    ex.config_path.write_text('[commands]\nallowlist = ["/usr/bin/true"]\n')
    assert "exit code 0" in ex.run_command("/usr/bin/true")


def test_timeout(ex: ToolExecutor):
    ex.config_path.write_text('[commands]\nallowlist = ["/bin/sleep"]\n')
    out = ex.run_command("/bin/sleep 5", timeout=1)
    assert "timed out" in out


def test_empty_allowlist_entry_does_not_match(ex: ToolExecutor):
    """Empty allowlist entry must not match any command (prevent allow-all gate)."""
    assert not command_allowed(["rm", "-rf", "/"], [[], ["npm", "test"]])


def test_empty_allowlist_entry_in_config(ex: ToolExecutor):
    """End-to-end: empty entry in config cannot bypass security gate."""
    ex.config_path.write_text('[commands]\nallowlist = ["", "/bin/echo"]\n')
    # Empty entry should not allow /usr/bin/true
    assert ex.run_command("/usr/bin/true").startswith("command denied")
    # But /bin/echo should still be allowed (non-empty entry)
    out = ex.run_command("/bin/echo hi")
    assert "exit code 0" in out


# --- B3: command-induced file changes must be recorded ---


@pytest.fixture()
def pyex(tmp_path: Path) -> ToolExecutor:
    """Executor whose allowlist contains the current Python interpreter, the
    stand-in for the default-allowlisted formatters (black, prettier, ruff)
    whose whole job is modifying files."""
    root = tmp_path / "proj"
    root.mkdir()
    # config in its own dir: data_dir defaults to config_path.parent and is
    # write-protected, so it must not be an ancestor of the project root.
    cfg = tmp_path / "sous-home" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    return ToolExecutor(root, cfg)


def test_command_created_file_recorded(pyex: ToolExecutor):
    out = pyex.run_command(f"{sys.executable} -c \"open('gen.txt', 'w').write('made-by-command')\"")
    assert "exit code 0" in out
    changes = {c.path: c for c in pyex.changed_files()}
    assert "gen.txt" in changes
    assert changes["gen.txt"].kind == "created"
    assert changes["gen.txt"].before_sha is None
    assert changes["gen.txt"].after_sha


def test_command_modified_file_recorded(pyex: ToolExecutor):
    (pyex.project_root / "fmt.py").write_text("x=1\n")
    out = pyex.run_command(
        f"{sys.executable} -c \"open('fmt.py', 'w').write('x = 1  # formatted')\""
    )
    assert "exit code 0" in out
    changes = {c.path: c for c in pyex.changed_files()}
    assert "fmt.py" in changes
    assert changes["fmt.py"].kind == "modified"
    # the pre-command snapshot is stat-only: prior content hash is unknown
    assert changes["fmt.py"].before_sha is None


def test_command_edit_refreshes_stale_hash_of_tracked_file(pyex: ToolExecutor):
    """A file the worker wrote and a formatter then rewrote must carry the
    formatter's content hash, not the stale pre-command one."""
    pyex.write_file("t.txt", "one")
    [before] = pyex.changed_files()
    pyex.run_command(f"{sys.executable} -c \"open('t.txt', 'w').write('two-formatted')\"")
    [after] = [c for c in pyex.changed_files() if c.path == "t.txt"]
    assert after.kind == "created"  # original kind survives
    assert after.after_sha != before.after_sha  # hash refreshed


# --- forged stat metadata must not hide a modification from the audit ---


def test_forged_mtime_and_size_rewrite_still_recorded(pyex: ToolExecutor):
    """Allowlisted test runners execute code the worker just wrote (e.g. a
    conftest.py pytest imports), so executed code can rewrite a file with
    EQUAL-LENGTH content and restore the original mtime via os.utime, making
    a (mtime_ns, size) pair byte-identical before and after. The audit must
    still record the file: ctime bumps on every inode change — including the
    os.utime call itself — and cannot be set back from userspace."""
    (pyex.project_root / "secret.py").write_text("SECRET = 'aaa'\n")
    (pyex.project_root / "forge.py").write_text(
        "import os\n"
        "st = os.stat('secret.py')\n"
        "open('secret.py', 'w').write(\"SECRET = 'bbb'\\n\")\n"  # same length
        "os.utime('secret.py', ns=(st.st_atime_ns, st.st_mtime_ns))\n"
    )
    out = pyex.run_command(f"{sys.executable} forge.py")
    assert "exit code 0" in out
    changes = {c.path: c for c in pyex.changed_files()}
    assert "secret.py" in changes, "equal-length rewrite with restored mtime evaded the audit"
    assert changes["secret.py"].kind == "modified"


def test_snapshot_tuple_includes_ctime(pyex: ToolExecutor):
    """Unit-level: two stats of an unchanged file compare equal, while a
    same-length rewrite with restored mtime compares unequal — only possible
    if the snapshot signature carries ctime alongside (mtime_ns, size)."""
    f = pyex.project_root / "s.txt"
    f.write_text("aaa")
    first = pyex._tree_snapshot()["s.txt"]
    assert pyex._tree_snapshot()["s.txt"] == first  # unchanged file: equal
    assert len(first) == 3  # (mtime_ns, size, ctime_ns)
    st = os.stat(f)
    f.write_text("bbb")  # same length
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    forged = pyex._tree_snapshot()["s.txt"]
    assert forged[:2] == first[:2]  # the forgery really did fool mtime+size
    assert forged != first  # ...but ctime still betrays it


def test_unchanged_files_not_reported_after_command(pyex: ToolExecutor):
    """Merely reading a file (atime-only traffic) must not put it in the
    audit: over-reporting every untouched file would bury the real diff."""
    (pyex.project_root / "ro.txt").write_text("stable")
    out = pyex.run_command(f"{sys.executable} -c \"print(open('ro.txt').read())\"")
    assert "exit code 0" in out
    assert pyex.changed_files() == []


# --- timeout must kill the whole process group, not just the direct child ---


@pytest.fixture()
def shex(tmp_path: Path) -> ToolExecutor:
    """Executor whose allowlist contains /bin/sh, the stand-in for allowlisted
    test runners (pytest-xdist, npm test, make) that spawn descendants."""
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "sous-home" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('[commands]\nallowlist = ["/bin/sh"]\n')
    return ToolExecutor(root, cfg)


@pytest.mark.slow
def test_timeout_kills_descendants_before_recording(shex: ToolExecutor):
    """A descendant that would outlive the timeout must not be able to write
    files after run_command has returned and recorded changes. Real processes
    on purpose: the bug is OS process-tree behavior, unmockable."""
    out = shex.run_command('/bin/sh -c "(sleep 3; echo pwned > late.txt) & sleep 10"', timeout=1)
    assert "timed out" in out
    # Sleep past when the orphaned descendant would have fired (t=3s from
    # command start; we are at ~t=1s). If the group kill worked, nothing
    # is left alive to write late.txt.
    time.sleep(4)
    assert not (shex.project_root / "late.txt").exists()
    assert all(c.path != "late.txt" for c in shex.changed_files())


@pytest.mark.slow
def test_timeout_group_kill_leaves_no_survivors(shex: ToolExecutor):
    """A descendant the group kill can reach must actually be dead after the
    call returns (not merely orphaned and still running)."""
    out = shex.run_command('/bin/sh -c "sleep 30 & echo $! > child.pid; sleep 10"', timeout=1)
    assert "timed out" in out
    pid = int((shex.project_root / "child.pid").read_text())
    # The backgrounded sleep was in the child's process group; after the
    # escalated group kill it must no longer exist. Allow a moment for
    # launchd to reap the reparented orphan.
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"descendant {pid} survived the process-group kill")
    # child.pid was written before the timeout: the audit must still see it.
    assert any(c.path == "child.pid" for c in shex.changed_files())


def test_unexpected_exception_kills_group_and_reraises(
    shex: ToolExecutor, monkeypatch: pytest.MonkeyPatch
):
    """A non-timeout exception escaping wait() (KeyboardInterrupt /
    SystemExit during daemon shutdown) must kill the whole process group and
    re-raise — parity with subprocess.run's internal kill-on-any-exception.
    The exception is injected on the FIRST wait() call only, so the
    group-kill helper's own grace-period and reap waits still work."""
    import sous.toolexec as toolexec

    captured: dict = {}
    real_popen = toolexec.subprocess.Popen

    class ExplodingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["proc"] = self
            self._exploded = False

        def wait(self, *args, **kwargs):
            if not self._exploded:
                self._exploded = True
                raise KeyboardInterrupt
            return super().wait(*args, **kwargs)

    monkeypatch.setattr(toolexec.subprocess, "Popen", ExplodingPopen)
    with pytest.raises(KeyboardInterrupt):
        shex.run_command('/bin/sh -c "sleep 30 & sleep 30"', timeout=5)
    proc = captured["proc"]
    # (b) the child's whole process group must be gone afterwards
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("process group survived the exception backstop")


def test_command_changes_in_git_dir_not_recorded(pyex: ToolExecutor):
    (pyex.project_root / ".git").mkdir()
    pyex.run_command(f"{sys.executable} -c \"open('.git/junk', 'w').write('x')\"")
    assert all(".git" not in c.path for c in pyex.changed_files())


# --- output capping: bounded memory, head+tail retention ---


def _write_emit_script(root: Path, mb: int) -> None:
    """A script emitting a distinctive first line, `mb` MB of filler, and a
    distinctive last line — the shape of a noisy test run whose verdict is
    at the END."""
    (root / "emit.py").write_text(
        "import sys\n"
        "sys.stdout.write('HEAD-MARKER-FIRST-LINE\\n')\n"
        "chunk = 'x' * 65536\n"
        f"for _ in range({mb} * 16):\n"
        "    sys.stdout.write(chunk)\n"
        "sys.stdout.write('\\nTAIL-MARKER-LAST-LINE\\n')\n"
    )


def test_huge_output_capped_without_buffering_in_memory(pyex: ToolExecutor):
    """The 16 KB cap must be enforced without ever holding the command's
    full output in RAM: the daemon shares 64 GB with a ~28.7 GB resident
    model, and running noisy test suites is the worker's primary job.
    ~50 MB of output must not grow peak RSS by anything near 50 MB."""
    _write_emit_script(pyex.project_root, 50)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = pyex.run_command(f"{sys.executable} emit.py")
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert "exit code 0" in out
    assert len(out) <= MAX_TOOL_OUTPUT + 100  # cap holds (+ marker slack)
    assert "HEAD-MARKER-FIRST-LINE" in out  # the 50 MB really flowed
    assert "TAIL-MARKER-LAST-LINE" in out
    # ru_maxrss is bytes on macOS (the product's target; KB on Linux, where
    # this margin is even more generous). Allow ample noise, but nothing
    # near the ~50 MB the command printed.
    growth = after - before
    assert growth < 100 * 1024 * 1024, f"peak RSS grew {growth} bytes"


def test_capped_output_retains_head_and_tail(pyex: ToolExecutor):
    """For a verify command the verdict (`3 failed`, a traceback, a summary
    line) is at the END — head-only truncation hides exactly what the
    worker needs. Both the beginning and the end must survive the cap,
    with an elision marker between them."""
    _write_emit_script(pyex.project_root, 1)
    out = pyex.run_command(f"{sys.executable} emit.py")
    assert "exit code 0" in out
    assert "HEAD-MARKER-FIRST-LINE" in out
    assert "TAIL-MARKER-LAST-LINE" in out
    assert "elided" in out
    assert len(out) <= MAX_TOOL_OUTPUT + 100


def test_capped_output_retains_stderr_tail(pyex: ToolExecutor):
    """When the noise is on stderr (compilers, linters), the tail of stderr
    is the verdict and must survive the cap."""
    (pyex.project_root / "emit_err.py").write_text(
        "import sys\n"
        "sys.stdout.write('OUT-HEAD\\n')\n"
        "chunk = 'e' * 65536\n"
        "for _ in range(16):\n"
        "    sys.stderr.write(chunk)\n"
        "sys.stderr.write('\\nERR-TAIL-VERDICT\\n')\n"
    )
    out = pyex.run_command(f"{sys.executable} emit_err.py")
    assert "exit code 0" in out
    assert "OUT-HEAD" in out
    assert "ERR-TAIL-VERDICT" in out
    assert len(out) <= MAX_TOOL_OUTPUT + 100


def test_small_output_unchanged_no_elision(pyex: ToolExecutor):
    """Small outputs keep the exact existing contract: exit-code line, then
    stdout, then stderr, no elision marker."""
    out = pyex.run_command(
        f"{sys.executable} -c \"import sys; print('out-marker'); "
        f"print('err-marker', file=sys.stderr)\""
    )
    assert out == "exit code 0\nout-marker\n\nerr-marker\n"
    assert "elided" not in out


def test_group_kill_survives_eperm_from_killpg(shex: ToolExecutor, monkeypatch):
    """macOS returns EPERM, not ESRCH, from killpg when no member of the group
    can be signalled — the state the group reaches once its remaining members
    are un-reaped zombies. That is "nothing left to kill", not a failure, and
    it must not escape run_command.

    Observed as an intermittent CI failure: the group emptied to zombies
    between the SIGTERM and the SIGKILL, and only ProcessLookupError was
    suppressed."""
    real_killpg = os.killpg

    def eperm_on_hard_kill(pgid: int, sig: int):
        if sig == signal.SIGKILL:
            raise PermissionError(1, "Operation not permitted")
        return real_killpg(pgid, sig)

    monkeypatch.setattr(toolexec.os, "killpg", eperm_on_hard_kill)
    out = shex.run_command('/bin/sh -c "sleep 10"', timeout=1)
    assert "timed out" in out


def test_hard_kill_precedes_reaping_the_group_leader(monkeypatch):
    """start_new_session makes the child its own group leader, so the pgid IS
    its pid. Reaping it releases that pid, after which the pgid may be
    recycled and a later SIGKILL could land on an unrelated group. The
    escalation must happen while the leader is still un-reaped."""
    proc = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 30"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pgid = os.getpgid(proc.pid)
    reaped_when_sent: dict[int, bool] = {}
    real_killpg = os.killpg

    def spy(pg: int, sig: int):
        reaped_when_sent.setdefault(sig, proc.returncode is not None)
        return real_killpg(pg, sig)

    monkeypatch.setattr(toolexec.os, "killpg", spy)
    try:
        _kill_process_group(pgid, proc)
    finally:
        if proc.returncode is None:  # pragma: no cover - only on failure paths
            proc.kill()
            proc.wait()

    assert signal.SIGKILL in reaped_when_sent, "expected a SIGKILL escalation"
    assert reaped_when_sent[signal.SIGKILL] is False, (
        "the group leader was already reaped when SIGKILL was sent, so its pid "
        "— and therefore the process-group id — could have been recycled"
    )


def test_fast_command_that_exits_before_pgid_capture(ex: ToolExecutor, monkeypatch):
    """A child that exits before the parent captures its group id must not fail.

    macOS getpgid() returns ESRCH for a zombie, unlike Linux, so reading the
    pgid off an already-exited child raises ProcessLookupError. Sleeping right
    after Popen forces the ordering that a fast command hits by chance.
    """
    real_popen = subprocess.Popen

    def popen_then_let_child_exit(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        time.sleep(0.3)  # child exits and becomes an un-reaped zombie
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen_then_let_child_exit)

    out = ex.run_command("/bin/echo hello")
    assert "exit code 0" in out and "hello" in out


# --- shutdown must not orphan a running command ------------------------------


def test_terminate_active_commands_kills_an_in_flight_child(tmp_path: Path):
    """A command running when the daemon stops must not outlive it.

    run_command puts its child in a new session, so the child survives its
    parent and keeps writing files unless the group is killed explicitly.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    ex = ToolExecutor(root, cfg)

    marker = tmp_path / "ticks.txt"
    marker.touch()
    script = (
        f"import time\nwhile True:\n    open({str(marker)!r},'a').write('t')\n    time.sleep(0.1)"
    )
    done = threading.Event()

    def run():
        ex.run_command(f"{sys.executable} -c {shlex.quote(script)}", timeout=60)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    for _ in range(100):  # wait until the child is actually writing
        if marker.stat().st_size > 0:
            break
        time.sleep(0.1)
    assert marker.stat().st_size > 0, "child never started"

    killed = toolexec.terminate_active_commands()
    assert killed == 1
    assert done.wait(timeout=15), "run_command never returned after the kill"

    before = marker.stat().st_size
    time.sleep(1.0)
    assert marker.stat().st_size == before, "child survived and kept writing"


def test_sigterm_to_the_daemon_kills_a_running_command(tmp_path: Path):
    """End to end, in a real process taking a real SIGTERM.

    Default SIGTERM handling terminates the daemon outright — main()'s finally
    never runs — so without an installed handler the sandboxed child is
    orphaned and goes on mutating the project after the daemon is gone.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    marker = tmp_path / "ticks.txt"
    marker.touch()

    child = (
        f"import time\nwhile True:\n    open({str(marker)!r},'a').write('t')\n    time.sleep(0.1)"
    )
    command = f"{sys.executable} -c {shlex.quote(child)}"
    daemon_src = (
        "import threading, time\n"
        "from pathlib import Path\n"
        "from sous.server import _install_shutdown_handler\n"
        "from sous.toolexec import ToolExecutor\n"
        f"ex = ToolExecutor(Path({str(root)!r}), Path({str(cfg)!r}))\n"
        "_install_shutdown_handler(threading.Event())\n"
        f"run = lambda: ex.run_command({command!r}, timeout=120)\n"
        "threading.Thread(target=run, daemon=True).start()\n"
        "time.sleep(120)\n"
    )
    daemon = subprocess.Popen([sys.executable, "-c", daemon_src])
    try:
        for _ in range(150):
            if marker.stat().st_size > 0:
                break
            time.sleep(0.1)
        assert marker.stat().st_size > 0, "command never started in the daemon"

        daemon.send_signal(signal.SIGTERM)
        daemon.wait(timeout=20)

        time.sleep(0.5)
        before = marker.stat().st_size
        time.sleep(1.5)
        assert marker.stat().st_size == before, (
            "sandboxed child outlived the daemon and kept writing to the project"
        )
    finally:
        daemon.kill()


def test_command_started_during_shutdown_does_not_slip_through(tmp_path: Path):
    """Shutdown must close registration, not just snapshot it.

    stop.set() does not interrupt the task in flight: after its command is
    killed, the agent loop can issue another one. A snapshot taken before that
    never sees the new group, and the process exits leaving it orphaned —
    exactly the leak this file is meant to prevent.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    ex = ToolExecutor(root, cfg)

    assert toolexec.terminate_active_commands() == 0  # nothing running; latches shutdown

    marker = tmp_path / "late.txt"
    marker.touch()
    script = (
        f"import time\nwhile True:\n    open({str(marker)!r},'a').write('t')\n    time.sleep(0.1)"
    )
    out = ex.run_command(f"{sys.executable} -c {shlex.quote(script)}", timeout=30)

    assert "shutting down" in out, f"late command ran to completion: {out!r}"
    before = marker.stat().st_size
    time.sleep(1.0)
    assert marker.stat().st_size == before, "late command's child survived shutdown"


def test_terminate_does_not_linger_between_snapshot_and_kill(tmp_path: Path):
    """No grace sleep on the shutdown path.

    The worker is blocked in proc.wait() and reaps the group leader as soon as
    it dies, so a delay between signalling and SIGKILL is a window where the
    pgid can be recycled onto an unrelated group. The daemon exits immediately
    afterwards, so nothing would observe a graceful teardown anyway.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    ex = ToolExecutor(root, cfg)

    marker = tmp_path / "ticks.txt"
    marker.touch()
    script = (
        f"import time\nwhile True:\n    open({str(marker)!r},'a').write('t')\n    time.sleep(0.1)"
    )
    done = threading.Event()

    def run():
        ex.run_command(f"{sys.executable} -c {shlex.quote(script)}", timeout=60)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    for _ in range(100):
        if marker.stat().st_size > 0:
            break
        time.sleep(0.1)
    assert marker.stat().st_size > 0, "child never started"

    started = time.monotonic()
    assert toolexec.terminate_active_commands() == 1
    elapsed = time.monotonic() - started
    assert elapsed < toolexec._KILL_GRACE_SECONDS, (
        f"terminate slept {elapsed:.2f}s between snapshot and kill"
    )
    # Wait for run_command to unwind before leaving: it releases the group id on
    # the way out, and a half-finished command would leak a live entry in the
    # module-level registry into whatever test runs next.
    assert done.wait(timeout=15), "run_command never returned after the kill"


def test_group_is_unregistered_before_the_audit_runs(tmp_path: Path, monkeypatch):
    """Drop the pgid the moment it is reaped, not after auditing.

    proc.wait() reaps the leader, so from that instant the number belongs to
    the OS again. _record_command_changes walks the whole project tree and the
    output read-back follows it — leaving the id registered across both means a
    shutdown in that window SIGKILLs whatever inherited it.
    """
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[commands]\nallowlist = ["/bin/echo"]\n')
    ex = ToolExecutor(root, cfg)

    seen: dict[str, set[int]] = {}
    real = ToolExecutor._record_command_changes

    def spy(self, before_snap):
        with toolexec._active_groups_lock:
            seen["registered"] = set(toolexec._active_groups)
        return real(self, before_snap)

    monkeypatch.setattr(ToolExecutor, "_record_command_changes", spy)
    out = ex.run_command("/bin/echo hi")

    assert "exit code 0" in out
    assert seen["registered"] == set(), "reaped pgid was still registered during the audit"


# ---- `cd <dir> && <command>` normalization -------------------------------


@pytest.fixture()
def ex_pwd(tmp_path: Path) -> ToolExecutor:
    """Executor whose allowlist includes /bin/pwd so tests can observe cwd."""
    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[commands]\nallowlist = ["/bin/echo", "/bin/pwd"]\n')
    return ToolExecutor(root, cfg)


def test_cd_prefix_runs_allowlisted_command_in_that_dir(ex_pwd: ToolExecutor):
    """The worker's habitual `cd <dir> && cmd` must behave as a shell user
    expects: cmd runs with cwd=<dir> — not "command not found: cd"."""
    sub = ex_pwd.project_root / "sub"
    out = ex_pwd.run_command(f"cd {sub} && /bin/pwd")
    assert "exit code 0" in out
    assert str(sub.resolve()) in out


def test_cd_prefix_relative_dir(ex_pwd: ToolExecutor):
    out = ex_pwd.run_command("cd sub && /bin/pwd")
    assert "exit code 0" in out
    assert str((ex_pwd.project_root / "sub").resolve()) in out


def test_cd_outside_root_denied_and_never_offered_for_approval(ex_pwd: ToolExecutor):
    """Path confinement is not approvable anywhere else in the sandbox; the
    cd target must be checked BEFORE the approval hook can bless it."""
    calls: list[str] = []

    def hook(cmd: str) -> bool:
        calls.append(cmd)
        return True

    out = ex_pwd.run_command("cd /etc && /bin/echo hi", approval=hook)
    assert out.startswith("command rejected")
    assert "escapes" in out or "root" in out
    assert calls == []


def test_cd_dotdot_escape_denied(ex_pwd: ToolExecutor):
    out = ex_pwd.run_command(f"cd {ex_pwd.project_root}/.. && /bin/echo hi")
    assert out.startswith("command rejected")


def test_cd_symlink_escape_denied(ex_pwd: ToolExecutor, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (ex_pwd.project_root / "link").symlink_to(outside)
    out = ex_pwd.run_command("cd link && /bin/echo hi")
    assert out.startswith("command rejected")


def test_cd_chained_commands_rejected_with_guidance(ex_pwd: ToolExecutor):
    calls: list[str] = []
    out = ex_pwd.run_command(
        "cd sub && /bin/echo a && /bin/echo b",
        approval=lambda c: calls.append(c) or True,
    )
    assert out.startswith("command rejected")
    assert "one command" in out
    assert calls == []


def test_bare_cd_rejected_with_guidance(ex_pwd: ToolExecutor):
    out = ex_pwd.run_command("cd sub")
    assert out.startswith("command rejected")
    assert "project root" in out


def test_cd_to_missing_dir_rejected(ex_pwd: ToolExecutor):
    out = ex_pwd.run_command("cd nope && /bin/echo hi")
    assert out.startswith("command rejected")
    assert "not a directory" in out


def test_cd_prefix_rest_still_goes_through_allowlist_and_approval(ex_pwd: ToolExecutor):
    """Normalization must not widen the allowlist: the remainder is matched
    exactly as if typed alone, and the approval hook sees the ORIGINAL
    command string so the human reviews what the model actually asked for."""
    seen: list[str] = []

    def approve(cmd: str) -> bool:
        seen.append(cmd)
        return True

    original = "cd sub && /usr/bin/printf ok"
    out = ex_pwd.run_command(original, approval=approve)
    assert "exit code 0" in out and "ok" in out
    assert seen == [original]

    out = ex_pwd.run_command("cd sub && /usr/bin/printf no", approval=lambda c: False)
    assert out.startswith("command denied")


def test_literal_amp_amp_argument_stays_inert_without_cd(ex_pwd: ToolExecutor):
    """Only the `cd` idiom is normalized: && anywhere else stays a literal
    argument, exactly like the existing metacharacter guarantee."""
    out = ex_pwd.run_command("/bin/echo a && b")
    assert "exit code 0" in out
    assert "a && b" in out


def test_cd_target_swap_during_approval_cannot_redirect_cwd(tmp_path: Path):
    """TOCTOU guard: the approval hook can block for minutes, and a background
    process from an earlier command can swap the cd target for a symlink in
    that window. The launch must use the directory handle pinned at check
    time, not re-walk the path — a swapped-in symlink to the outside must
    never become cwd. Observed via marker files: the pinned inode still shows
    its own file after a rename; the escape target's file must never appear."""
    import os

    root = tmp_path / "proj"
    sub = root / "sub"
    sub.mkdir(parents=True)
    (sub / "canary_inside").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary_outside").write_text("x")
    cfg = tmp_path / "config.toml"
    cfg.write_text('[commands]\nallowlist = ["/bin/ls"]\n')
    ex2 = ToolExecutor(root, cfg)

    def swap_then_approve(cmd: str) -> bool:
        os.rename(sub, root / "sub_moved")
        os.symlink(outside, sub)
        return True

    out = ex2.run_command("cd sub && /bin/ls", approval=swap_then_approve)
    assert "canary_outside" not in out, out
    # If it ran at all, it ran in the pinned inode (now named sub_moved),
    # which still carries its own marker — never the escape target.
    if "exit code 0" in out:
        assert "canary_inside" in out, out


def test_open_confined_dir_rejects_fd_that_escaped_the_root(tmp_path: Path, monkeypatch):
    """Layered defense for the resolve->open race: even when the path-string
    check passes, the OPENED directory is re-verified by its own fd. An fd
    that really points outside the root must be rejected."""
    import sous.toolexec as tx

    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(tx, "resolve_confined", lambda *a, **k: outside.resolve())
    with pytest.raises(tx.PathViolation):
        tx.open_confined_dir(root, "sub")


def test_open_confined_dir_returns_usable_fd_for_valid_dir(tmp_path: Path):
    import os

    import sous.toolexec as tx

    root = tmp_path / "proj"
    (root / "sub").mkdir(parents=True)
    fd = tx.open_confined_dir(root, "sub")
    try:
        assert os.path.samestat(os.fstat(fd), os.stat(root / "sub"))
    finally:
        os.close(fd)
