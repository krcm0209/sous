import errno
import json
import os
import plistlib
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sous.cli import _BOOTOUT_NOT_LOADED, LABEL, launchd_plist


def _free_cli_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plist_is_valid_and_correct():
    xml = launchd_plist("/usr/local/bin/sous", Path("/Users/x/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == ["/usr/local/bin/sous", "serve"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["StandardOutPath"] == "/Users/x/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/.sous/daemon.log"


def test_plist_sets_no_environment_variables():
    """PATH is deliberately NOT baked into the plist: the daemon adopts the
    user's login-shell PATH at startup (server._login_shell_path), which stays
    current and works however the daemon was launched. An install-time PATH
    snapshot here would be a second, staler mechanism that shadows it."""
    xml = launchd_plist("/Users/x/.local/bin/sous", Path("/Users/x/.sous"))
    data = plistlib.loads(xml.encode())
    assert "EnvironmentVariables" not in data


def test_plist_escapes_xml_special_characters():
    """C5: a valid macOS path containing & or < must yield a plist that
    launchctl can actually parse — not silently-invalid XML."""
    xml = launchd_plist("/opt/a&b/sous", Path("/Users/x/My & <Special> Docs/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["ProgramArguments"] == ["/opt/a&b/sous", "serve"]
    assert data["StandardOutPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.log"


def test_status_reports_not_running(tmp_path, capsys, monkeypatch):
    from sous import cli
    from sous.config import SousConfig

    cfg = SousConfig(
        server_port=1,
        data_dir=tmp_path,  # port 1: never listening
        config_path=tmp_path / "c.toml",
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "not running" in out


def test_mcp_subcommand_dispatches_to_the_proxy(monkeypatch):
    """`sous mcp` must reach the proxy and propagate its exit code, so a failed
    cold start surfaces to the launching client instead of exiting 0."""
    import sous.proxy
    from sous.cli import main

    called = {}

    def fake_run():
        called["ran"] = True
        return 1

    monkeypatch.setattr(sous.proxy, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        main(["mcp"])
    assert called.get("ran") is True
    assert exc.value.code == 1


# --- stop / uninstall-launchd -------------------------------------------------


def _cfg(tmp_path: Path, port: int):
    from sous.config import SousConfig

    return SousConfig(server_port=port, data_dir=tmp_path, config_path=tmp_path / "c.toml")


def _fake_daemon(tmp_path: Path, port: int, hold_lock: bool = True):
    """A stand-in that holds the port, the flock, and its pid, like the real one.

    Holding the port matters: `stop` waits for it to close, so a victim that
    does not own it would let the test pass against code that never signalled.
    Holding the flock matters too — that is how `stop` tells a live daemon from
    a leftover pid file. `hold_lock=False` fakes the stale-pid case.
    """
    # It must ACCEPT, not just listen: `stop` probes the port repeatedly while
    # waiting for it to close, and an unaccepted backlog fills up and starts
    # refusing — which reads as "daemon not running" and hides the real result.
    lock_line = (
        f"import fcntl;lk=open({str(tmp_path / 'daemon.lock')!r},'r+b');"
        f"fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        if hold_lock
        else ""
    )
    if hold_lock:
        (tmp_path / "daemon.lock").touch()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            lock_line + f"import socket;s=socket.socket();"
            f"s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(64)\n"
            f"while True:\n    c,_=s.accept()\n    c.close()",
        ]
    )
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.1)
    else:  # never connected — fail here, not later with a confusing symptom
        proc.kill()
        raise AssertionError(f"fake daemon never listened on {port}")
    (tmp_path / "daemon.lock").write_text(f"{proc.pid}\n")
    return proc


def test_stop_reports_when_not_running(tmp_path, capsys, monkeypatch):
    from sous import cli

    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))  # port 1: never up
    # Never consult the host's real launchd: this machine may well have the
    # agent loaded, which would make the result depend on the developer's box.
    monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
    cli.main(["stop"])
    assert "not running" in capsys.readouterr().out


def test_stop_refuses_when_launchd_manages_the_daemon(tmp_path, capsys, monkeypatch):
    """Under KeepAlive a signal is undone in a second, so refuse and say why."""
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: True)
        with pytest.raises(SystemExit) as exc:
            cli.main(["stop"])
        assert exc.value.code != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "uninstall-launchd" in out
        assert daemon.poll() is None, "signalled a launchd-managed daemon anyway"
    finally:
        daemon.kill()


def test_stop_signals_the_recorded_pid(tmp_path, capsys, monkeypatch):
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        cli.main(["stop"])
        daemon.wait(timeout=10)
        assert daemon.returncode is not None
    finally:
        daemon.kill()


def test_uninstall_launchd_boots_out_and_removes_the_plist(tmp_path, capsys, monkeypatch):
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")
    booted = []
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: (booted.append(label), 0)[1])
    cli.main(["uninstall-launchd"])
    assert booted == [cli.LABEL]
    assert not plist.exists()


def test_uninstall_launchd_is_idempotent(tmp_path, capsys, monkeypatch):
    """Running it twice, or with nothing installed, must not blow up."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"  # never created
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: cli._BOOTOUT_NOT_LOADED)
    cli.main(["uninstall-launchd"])
    assert "not installed" in capsys.readouterr().out.lower()


def test_stop_checks_launchd_before_the_port(tmp_path, capsys, monkeypatch):
    """A KeepAlive job can be loaded while its daemon is mid-restart.

    Checking the port first reports "not running" and exits 0 in that window,
    even though launchd is about to bring it back — which breaks the
    unmanaged-only contract exactly when it matters.
    """
    from sous import cli

    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))  # port down
    monkeypatch.setattr(cli, "_launchd_loaded", lambda label: True)
    with pytest.raises(SystemExit) as exc:
        cli.main(["stop"])
    assert exc.value.code != 0
    assert "launchd" in capsys.readouterr().out


def test_stop_refuses_a_stale_pid_file(tmp_path, capsys, monkeypatch):
    """daemon.lock outlives the daemon, so the pid in it may be recycled.

    Something is listening and a pid is recorded, but nobody holds the lock —
    signalling that pid could hit an unrelated process.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port, hold_lock=False)
    try:
        (tmp_path / "daemon.lock").write_text("999999\n")  # not the listener
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        with pytest.raises(SystemExit) as exc:
            cli.main(["stop"])
        assert exc.value.code != 0
        assert daemon.poll() is None, "signalled despite a stale lock"
    finally:
        daemon.kill()


def test_stop_warns_about_all_active_tasks(tmp_path, capsys, monkeypatch):
    """The warning must cover every state recover_interrupted() will fail.

    list_recent() caps at 20 rows, so a long-running task drops off the list
    once 20 newer ones are queued, and AWAITING_APPROVAL was missed entirely.
    """
    from sous import cli
    from sous.tasks import TaskStore

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        store = TaskStore(tmp_path / "tasks.db")
        old = store.enqueue("long-runner", "x", "/tmp", [], [])
        store.claim_next()
        approving = store.enqueue("needs-approval", "x", "/tmp", [], [])
        store.claim_next()
        store.request_approval(approving.id, "pytest")
        for i in range(25):  # push the running task off list_recent()
            store.enqueue(f"filler{i}", "x", "/tmp", [], [])
        still_running = store.get(old.id)
        assert still_running is not None and still_running.state == "running"

        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        cli.main(["stop"])
        out = capsys.readouterr().out
        assert "running" in out and "awaiting_approval" in out
    finally:
        daemon.kill()


def test_uninstall_keeps_the_plist_when_bootout_really_fails(tmp_path, capsys, monkeypatch):
    """Only "not loaded" is benign. A permissions error must not be reported as
    success with the plist deleted and the KeepAlive job still running."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: 5)  # I/O error, not "absent"
    with pytest.raises(SystemExit) as exc:
        cli.main(["uninstall-launchd"])
    assert exc.value.code != 0
    assert plist.exists(), "deleted the plist despite a failed bootout"


def test_uninstall_boots_out_even_when_the_plist_is_gone(tmp_path, capsys, monkeypatch):
    """A hand-deleted plist does not unload the job; the label can still be
    bootstrapped, so uninstall must target the label regardless."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"  # never created
    booted = []
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: (booted.append(label), 0)[1])
    cli.main(["uninstall-launchd"])
    assert booted == [cli.LABEL], "did not attempt bootout without a plist"


def test_uninstall_does_not_advise_stop_for_a_daemon_it_just_unloaded(
    tmp_path, capsys, monkeypatch
):
    """bootout terminates the job, so an immediate port check races its exit.

    Advising `sous stop` then names a daemon that is already going away, and
    running it reports "not running" — which reads as though the advice was
    wrong rather than merely early.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    def bootout_then_it_dies(label):
        threading.Timer(0.8, daemon.kill).start()  # launchd tears it down shortly after
        return 0

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", bootout_then_it_dies)
        cli.main(["uninstall-launchd"])
        out = capsys.readouterr().out
        assert "unloaded the launchd agent" in out
        assert "sous stop" not in out, "advised stopping a daemon that was already exiting"
    finally:
        daemon.kill()


def test_uninstall_still_advises_stop_when_a_daemon_outlives_the_unload(
    tmp_path, capsys, monkeypatch
):
    """The wait must not swallow the case the advice exists for.

    Goes through the successful-unload branch on purpose: that is the one the
    wait was added to, so a regression suppressing the advice whenever bootout
    succeeds has to fail here. The grace is shortened rather than waited out,
    and the elapsed time is asserted so the timeout path is genuinely taken
    instead of the daemon merely looking absent.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)  # nothing kills this one
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", lambda label: 0)  # unload succeeded
        monkeypatch.setattr(cli, "_UNLOAD_GRACE_SECONDS", 0.5)
        started = time.monotonic()
        cli.main(["uninstall-launchd"])
        elapsed = time.monotonic() - started

        out = capsys.readouterr().out
        assert "unloaded the launchd agent" in out
        assert "sous stop" in out, "suppressed the advice for a daemon that survived"
        assert elapsed >= 0.5, f"returned in {elapsed:.2f}s — the wait never ran"
    finally:
        daemon.kill()


def test_uninstall_advises_stop_when_there_was_nothing_to_unload(tmp_path, capsys, monkeypatch):
    """No unload happened, so nothing was going to terminate the daemon and the
    advice applies immediately — the branch the wait must not reach."""
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", lambda label: cli._BOOTOUT_NOT_LOADED)
        cli.main(["uninstall-launchd"])
        assert "sous stop" in capsys.readouterr().out
    finally:
        daemon.kill()


def _install_env(
    tmp_path,
    monkeypatch,
    *,
    bootout_code: int,
    lock_held: bool,
    loaded: bool = True,
    lock_frees: bool = True,
):
    """install-launchd with launchctl, the lock probe and the plist path faked."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    calls: dict = {"bootout": [], "run": [], "await_lock_free": []}
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/sous")
    monkeypatch.setattr(
        cli, "_bootout", lambda label: (calls["bootout"].append(label), bootout_code)[1]
    )
    monkeypatch.setattr(cli, "_launchd_loaded", lambda label: loaded)
    monkeypatch.setattr(
        cli,
        "_await_lock_free",
        lambda data_dir, seconds: (
            calls["await_lock_free"].append((data_dir, seconds)),
            lock_frees,
        )[1],
    )
    monkeypatch.setattr(cli, "_lock_is_held", lambda data_dir: lock_held)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check: calls["run"].append(cmd))
    return cli, plist, calls


def test_install_launchd_boots_out_then_writes_and_bootstraps(tmp_path, capsys, monkeypatch):
    """Re-running must work: bootstrap fails on an already-loaded label, and
    a plist that changed (both streams now go to daemon.log) reloads only if
    the old job is out first."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    cli.main(["install-launchd"])
    assert calls["bootout"] == [cli.LABEL]
    data = plistlib.loads(plist.read_bytes())
    assert data["StandardOutPath"] == data["StandardErrorPath"] == str(tmp_path / "daemon.log")
    assert calls["run"] and calls["run"][0][:2] == ["launchctl", "bootstrap"]
    assert calls["run"][0][-1] == str(plist)


def test_install_launchd_when_nothing_was_loaded(tmp_path, capsys, monkeypatch):
    """A first install: bootout reports "not loaded" and that is fine."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=_BOOTOUT_NOT_LOADED, lock_held=False
    )
    cli.main(["install-launchd"])
    assert calls["bootout"] == [cli.LABEL]  # tried, and tolerated "not loaded"
    assert plist.exists() and calls["run"]


def test_install_launchd_refuses_while_a_daemon_holds_the_lock(tmp_path, capsys, monkeypatch):
    """A hand-started daemon still writing daemon.err.log must not have the
    file appended and unlinked under it; the user stops it first."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=_BOOTOUT_NOT_LOADED, lock_held=True
    )
    (tmp_path / "daemon.err.log").write_text("old\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    assert "sous stop" in capsys.readouterr().out
    assert not plist.exists() and (tmp_path / "daemon.err.log").exists() and not calls["run"]


def test_install_launchd_folds_the_legacy_stderr_log_into_daemon_log(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    (tmp_path / "daemon.log").write_bytes(b"out-1\n")
    (tmp_path / "daemon.err.log").write_bytes(b"err-1\nerr-2\n")
    cli.main(["install-launchd"])
    assert (tmp_path / "daemon.log").read_bytes() == b"out-1\nerr-1\nerr-2\n"
    assert not (tmp_path / "daemon.err.log").exists()
    assert "folded daemon.err.log into daemon.log" in capsys.readouterr().out


def test_install_launchd_without_a_legacy_log_touches_nothing(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    cli.main(["install-launchd"])
    assert not (tmp_path / "daemon.log").exists() or (tmp_path / "daemon.log").read_bytes() == b""
    assert "folded" not in capsys.readouterr().out


def test_install_launchd_keeps_the_plist_when_bootout_really_fails(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=5, lock_held=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1 and not plist.exists() and not calls["run"]


def test_install_launchd_installs_when_bootout_errors_on_a_job_launchd_does_not_list(
    tmp_path, capsys, monkeypatch
):
    """A first install over SSH with no GUI login: bootout of gui/<uid>/… fails
    with a domain error, not "not loaded" — and there is no job to protect, so
    the plist still gets written (main's behaviour before bootout was added)."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=125, lock_held=False, loaded=False
    )
    cli.main(["install-launchd"])
    assert plist.exists() and calls["run"]
    assert calls["await_lock_free"] == []  # nothing was unloaded, so nothing to wait for


def test_install_launchd_says_the_job_is_unloaded_when_its_daemon_outlives_the_wait(
    tmp_path, capsys, monkeypatch
):
    """The bootout already took the job down; a daemon still shutting down past
    the wait must not leave the user with a hedge and no daemon."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=0, lock_held=True, lock_frees=False
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "unloaded" in out and "retry" in out
    assert not plist.exists() and not calls["run"]


def test_install_launchd_reports_an_unwritable_plist_without_a_traceback(
    tmp_path, capsys, monkeypatch
):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    plist.mkdir()  # writing a file over a directory raises IsADirectoryError
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert str(plist) in out and "unloaded" in out
    assert not calls["run"]


def test_install_launchd_waits_for_the_lock_only_after_a_real_bootout(tmp_path, monkeypatch):
    """A resident model keeps the daemon in lifespan shutdown, and the flock
    with it, for seconds after the port stops accepting — so the wait must
    poll the lock, not the port, and only when a job was actually unloaded. A
    regression that waited unconditionally would still pass a port-based
    check here; it would not pass this one."""
    cli, _, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=_BOOTOUT_NOT_LOADED, lock_held=False
    )
    cli.main(["install-launchd"])
    assert calls["await_lock_free"] == []

    cli, _, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    cli.main(["install-launchd"])
    assert calls["await_lock_free"] == [(tmp_path, cli._DAEMON_EXIT_SECONDS)]
    from sous.server import GRACEFUL_SHUTDOWN_SECONDS

    # Longer than the daemon's own graceful-shutdown bound plus the runner's
    # close, or a busy daemon still exiting reads as a held lock.
    assert cli._DAEMON_EXIT_SECONDS > GRACEFUL_SHUTDOWN_SECONDS + 2


def test_install_launchd_folds_into_a_freshly_created_daemon_log(tmp_path, capsys, monkeypatch):
    """`"ab"` creates the target when it doesn't already exist — the shape a
    user sees whose stdout log was rotated away but never had a stderr file."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    (tmp_path / "daemon.err.log").write_bytes(b"err-1\nerr-2\n")
    cli.main(["install-launchd"])
    assert (tmp_path / "daemon.log").read_bytes() == b"err-1\nerr-2\n"
    assert not (tmp_path / "daemon.err.log").exists()


def test_install_launchd_reports_a_failed_fold_and_stops(tmp_path, capsys, monkeypatch):
    """A copy failure must not crash out with a raw traceback, must not lose
    or duplicate the legacy file, and must not proceed to write a plist over
    a daemon that was just unloaded and now has nowhere to log."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    (tmp_path / "daemon.log").mkdir()  # opening a directory for append raises OSError
    (tmp_path / "daemon.err.log").write_bytes(b"err-1\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "daemon.err.log" in out and "daemon.log" in out
    assert "unloaded" in out  # the bootout already happened; say so
    assert (tmp_path / "daemon.err.log").read_bytes() == b"err-1\n"
    assert not plist.exists() and not calls["run"]


def test_install_launchd_reports_a_failed_unlink_and_stops(tmp_path, capsys, monkeypatch):
    """Unlike the copy failure above, the append has already landed and been
    fsynced by the time unlink can fail — rolling it back would trade a
    duplicate for a loss, so this must fail safe the other way: say the fold
    already happened, name both files, and tell the operator to delete the
    old one by hand rather than let a raw OSError escape as a traceback."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    (tmp_path / "daemon.log").write_bytes(b"out-1\n")
    (tmp_path / "daemon.err.log").write_bytes(b"err-1\n")

    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.name == "daemon.err.log":
            raise OSError(errno.EACCES, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "daemon.err.log" in out and "daemon.log" in out
    assert "delete it by hand" in out
    # the append happened and was not rolled back
    assert (tmp_path / "daemon.log").read_bytes() == b"out-1\nerr-1\n"
    assert (tmp_path / "daemon.err.log").exists()  # unlink failed: still there
    assert not plist.exists() and not calls["run"]


def test_fold_legacy_stderr_log_fsyncs_the_target_before_unlinking_the_source(
    tmp_path, monkeypatch
):
    """The appended bytes sit in the page cache while the unlink is journaled
    metadata; without an fsync in between, a panic in that window loses the
    folded history even though the source is already gone."""
    from sous import cli

    (tmp_path / "daemon.err.log").write_bytes(b"err-1\nerr-2\n")
    calls: list[int] = []
    real_fsync = os.fsync

    def spy(fd):
        calls.append(fd)
        assert (tmp_path / "daemon.err.log").exists()  # not unlinked yet
        real_fsync(fd)

    monkeypatch.setattr(cli.os, "fsync", spy)
    assert cli._fold_legacy_stderr_log(tmp_path) is True
    assert calls, "os.fsync was never called before folding the legacy log"
    assert not (tmp_path / "daemon.err.log").exists()


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_fold_drops_a_legacy_name_that_links_to_daemon_log(tmp_path, capsys, link):
    """Appending a file onto itself never reaches EOF past the write buffer —
    daemon.log would grow until the disk filled. A link is already one log."""
    from sous import cli

    log = tmp_path / "daemon.log"
    log.write_bytes(b"x" * (1 << 18))  # past io.DEFAULT_BUFFER_SIZE
    legacy = tmp_path / "daemon.err.log"
    if link == "symlink":
        legacy.symlink_to(log)
    else:
        legacy.hardlink_to(log)
    assert cli._fold_legacy_stderr_log(tmp_path) is True
    assert log.read_bytes() == b"x" * (1 << 18)
    assert not legacy.exists() and not legacy.is_symlink()


def test_fold_leaves_daemon_log_linked_to_the_legacy_file_alone(tmp_path, capsys):
    """The reverse link: daemon.err.log is the only real copy, so removing it
    would take all the history with it."""
    from sous import cli

    legacy = tmp_path / "daemon.err.log"
    legacy.write_bytes(b"history\n")
    (tmp_path / "daemon.log").symlink_to(legacy)
    assert cli._fold_legacy_stderr_log(tmp_path) is False
    assert legacy.read_bytes() == b"history\n"
    assert (tmp_path / "daemon.log").read_bytes() == b"history\n"


def test_install_launchd_exits_nonzero_when_bootstrap_fails(tmp_path, capsys, monkeypatch):
    """A failed bootstrap now follows an unconditional bootout, so exiting 0
    here would report success with the daemon unloaded and nothing in its
    place — worse than the pre-existing job it just tore down."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)

    def _fail(cmd, check):
        raise FileNotFoundError("launchctl")

    monkeypatch.setattr(cli.subprocess, "run", _fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    assert "run manually" in capsys.readouterr().out
    assert plist.exists()  # the plist itself is still written; only the load failed


# --- wait ----------------------------------------------------------------------


def _wait_store(tmp_path, monkeypatch):
    from sous import cli
    from sous.config import SousConfig
    from sous.tasks import TaskStore

    cfg = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    return TaskStore(tmp_path / "tasks.db")


def _enqueue(store):
    return store.enqueue(
        title="t", instructions="x", project_root="/", context_files=[], verify_commands=[]
    )


def test_wait_returns_immediately_for_a_finished_task(tmp_path, capsys, monkeypatch):
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {"summary": "s"})
    cli.main(["wait", t.id])
    out = capsys.readouterr().out
    assert "done" in out and "completed" in out


def test_wait_blocks_until_approval_is_requested(tmp_path, capsys, monkeypatch):
    """The point of `wait`: agents park it in a background shell instead of
    tight-polling task_status or reading tasks.db by hand — so it must wake on
    awaiting_approval (a human is needed NOW), not only on terminal states."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()

    def approve_later():
        time.sleep(0.3)
        store.request_approval(t.id, "git diff")

    flipper = threading.Thread(target=approve_later)
    started = time.monotonic()
    flipper.start()
    try:
        cli.main(["wait", t.id, "--interval", "0.05"])
    finally:
        flipper.join()
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    assert "awaiting_approval" in out
    assert "git diff" in out, "the pending command is the thing the human must see"
    assert elapsed >= 0.3, f"returned in {elapsed:.2f}s — never actually waited"


def test_wait_unknown_task_exits_2(tmp_path, capsys, monkeypatch):
    from sous import cli

    _wait_store(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", "nope"])
    assert exc.value.code == 2


def test_wait_timeout_exits_1(tmp_path, capsys, monkeypatch):
    """A queued task that never advances must not hang the caller forever when
    a timeout was asked for — and the timeout must be exit 1, distinct from
    unknown-task (2), so scripts can tell 'still running' from 'gone'."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)  # stays queued: nothing ever claims it
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", t.id, "--timeout", "0.3", "--interval", "0.05"])
    assert exc.value.code == 1
    assert "queued" in capsys.readouterr().out


def test_wait_timeout_is_not_quantized_by_interval(tmp_path, capsys, monkeypatch):
    """A 0.3s timeout must expire near 0.3s even with the default 2s interval —
    each sleep has to be capped to the remaining budget, or the deadline check
    only runs on interval boundaries (and a task finishing inside the overrun
    would be reported as success AFTER the caller's deadline)."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)  # stays queued
    started = time.monotonic()
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", t.id, "--timeout", "0.3"])  # interval left at default
    elapsed = time.monotonic() - started
    assert exc.value.code == 1
    assert elapsed < 1.0, f"timed out after {elapsed:.2f}s — quantized to the interval"


def test_wait_rejects_degenerate_intervals_and_timeouts(tmp_path, capsys, monkeypatch):
    """--interval 0 recreates the tight-polling this command exists to prevent,
    a negative interval raises out of time.sleep, and a NaN timeout never
    expires — all three must be argparse usage errors (exit 2), not runtime
    misbehavior."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {"summary": "s"})  # even a done task: reject first
    for argv in (
        ["wait", t.id, "--interval", "0", "--timeout", "0.2"],
        ["wait", t.id, "--interval", "-1", "--timeout", "0.2"],
        ["wait", t.id, "--timeout", "nan"],
        ["wait", t.id, "--interval", "nan", "--timeout", "0.2"],
        ["wait", t.id, "--timeout", "-5"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2, argv


def test_wait_timeout_zero_is_an_immediate_probe(tmp_path, capsys, monkeypatch):
    """--timeout 0 is the defined non-blocking form: one state check, then
    report — exit 0 if the task already needs attention, exit 1 otherwise."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    pending = _enqueue(store)
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", pending.id, "--timeout", "0"])
    assert exc.value.code == 1
    done = _enqueue(store)
    store.claim_next()  # claims `pending`... order: claim_next takes oldest queued
    store.claim_next()
    store.finish(done.id, "completed", {"summary": "s"})
    cli.main(["wait", done.id, "--timeout", "0"])
    assert "done" in capsys.readouterr().out


# --- sous claude ---------------------------------------------------------------------


_DEFAULT_STATUS = object()


def _status(**gateway) -> dict:
    """The keys of `GET /sous/status` — `SousService.status_document()`,
    less the recent turns and tasks — of which the launcher reads
    `config.gateway` for its checks and `engine.model_id` for its one line."""
    return {
        "engine": {
            "model_id": "mlx-community/Qwen3.8-27B-4bit",
            "loaded": True,
            "loading": False,
            "holders": 0,
        },
        "config": {
            "gateway": {
                "enabled": True,
                "local_models": ["sous-local"],
                "max_context_tokens": 131072,
                "upstream_url": "https://api.anthropic.com",
                **gateway,
            }
        },
    }


def _claude_setup(tmp_path, monkeypatch, *, status=_DEFAULT_STATUS, **overrides):
    """A gateway-enabled config file, a `claude` on PATH, a daemon that answers
    `GET /sous/status`, and an execve that records instead of replacing the
    process."""
    import os

    from sous import cli
    from sous.config import SousConfig

    # Popped rather than passed alongside **overrides: a test overriding
    # gateway_enabled would otherwise collide with a literal keyword of the
    # same name below and raise "got multiple values for keyword argument".
    gateway_enabled = overrides.pop("gateway_enabled", True)
    cfg = SousConfig(
        server_port=8383,
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        gateway_enabled=gateway_enabled,
        **overrides,
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/opt/bin/claude" if name == "claude" else None
    )
    answer = _status() if status is _DEFAULT_STATUS else status
    monkeypatch.setattr(cli, "_daemon_status", lambda port, data_dir: answer)
    holds: list[int] = []
    monkeypatch.setattr(
        cli,
        "_hold",
        lambda port: (holds.append(port), {"loaded": True, "loading": False, "holders": 1})[1],
    )
    for var in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "API_TIMEOUT_MS",
        *cli._TIER_VARS,
    ):
        monkeypatch.delenv(var, raising=False)
    calls: list[tuple] = []
    monkeypatch.setattr(os, "execve", lambda exe, argv, env: calls.append((exe, argv, env)))
    return cfg, calls, holds


def test_claude_argv_appends_the_lsp_opt_out_unless_the_user_chose_their_own():
    """Claude Code's --disallowedTools is variadic, so it goes last (prepended,
    it would swallow a positional prompt) — and before a `--` if there is one."""
    from sous.cli import claude_argv

    assert claude_argv([]) == ["--disallowedTools", "LSP"]
    assert claude_argv(["-p", "hi"]) == ["-p", "hi", "--disallowedTools", "LSP"]
    assert claude_argv(["--", "fix it"]) == ["--disallowedTools", "LSP", "--", "fix it"]
    for own in (
        ["--disallowedTools", "Foo"],
        ["--disallowedTools=Foo"],
        ["--disallowed-tools", "Foo", "Bar"],
        ["--disallowed-tools=Foo"],
    ):
        assert claude_argv(["-p", "x", *own]) == ["-p", "x", *own]
    # After a `--` the same text is a prompt, not an option Claude Code will
    # parse, so it cannot be the user choosing their own opt-out.
    assert claude_argv(["--", "--disallowedTools"]) == [
        "--disallowedTools",
        "LSP",
        "--",
        "--disallowedTools",
    ]
    assert claude_argv(["--disallowedTools", "X", "--", "y"]) == [
        "--disallowedTools",
        "X",
        "--",
        "y",
    ]


def test_claude_env_sets_the_gateway_variables_and_nothing_credential_shaped():
    from sous.cli import claude_env

    env = claude_env(
        9999,
        ["sous-fast", "sous-local"],
        131072,
        {"PATH": "/usr/bin", "API_TIMEOUT_MS": "5", "HOME": "/Users/x"},
    )
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sous-fast"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/Users/x"
    # Global, not per-model: pinning it would make the frontier main loop
    # compact at the local window (decision 10).
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    for forbidden in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        assert forbidden not in env


def test_claude_execs_claude_with_the_gateway_environment(tmp_path, capsys, monkeypatch):
    from sous import cli

    cfg, calls, _ = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", "-p", "hi"])
    [(exe, argv, env)] = calls
    assert exe == "/opt/bin/claude"
    assert argv == ["/opt/bin/claude", "-p", "hi", "--disallowedTools", "LSP"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8383"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sous-local"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    err = capsys.readouterr().err
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8383" in err
    assert "CLAUDE_CODE_SUBAGENT_MODEL=sous-local" in err
    assert "CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1" in err
    assert "warning" not in err


def test_claude_passes_every_argument_through_including_help(tmp_path, monkeypatch):
    """Dispatch happens before argparse: `sous claude --help` is Claude Code's
    help, and a leading option is not sous's to reject."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", "--help"])
    cli.main(["claude", "--model", "opus", "--verbose"])
    assert calls[0][1] == ["/opt/bin/claude", "--help", "--disallowedTools", "LSP"]
    assert calls[1][1] == [
        "/opt/bin/claude",
        "--model",
        "opus",
        "--verbose",
        "--disallowedTools",
        "LSP",
    ]


def test_claude_warns_about_a_credential_variable_but_still_launches(tmp_path, capsys, monkeypatch):
    """Warn, don't refuse — either variable moves the main loop from the
    subscription to API-credit billing, and that is the user's call to make."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-canary")
    cli.main(["claude"])
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "warning" in err and "ANTHROPIC_API_KEY" in err and "API credits" in err
    assert "sk-ant-api03-canary" not in err
    assert calls[0][2]["ANTHROPIC_API_KEY"] == "sk-ant-api03-canary"  # untouched, not unset


def test_claude_warns_about_an_inherited_tier_variable_but_still_launches(
    tmp_path, capsys, monkeypatch
):
    """Same ruling as the credential variables: the user's environment is the
    user's. A leftover export from the whole-session recipe in CONTRIBUTING is
    exactly this case, so name the variable and its value (a model id, not a
    secret) and say what it costs — then launch."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "sous-local")
    cli.main(["claude"])
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "warning" in err
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in err and "sous-local" in err
    assert "hybrid" in err
    # Warned about, never stripped: the export is the user's to keep.
    assert calls[0][2]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "sous-local"


def test_claude_refuses_when_the_running_daemon_has_the_gateway_off(tmp_path, capsys, monkeypatch):
    """The file says enabled; the daemon says off. The daemon's startup
    snapshot is the truth — a file edited since it started changes nothing
    until it restarts."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch, status=_status(enabled=False))
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    err = capsys.readouterr().err
    assert "[gateway]" in err and "restart" in err


def test_claude_refuses_when_no_daemon_answers(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch, status=None)
    monkeypatch.setattr(cli, "_port_open", lambda port: False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    err = capsys.readouterr().err
    assert "no daemon" in err and "sous serve" in err


def test_claude_uses_the_daemons_values_and_says_so_when_the_file_disagrees(
    tmp_path, capsys, monkeypatch
):
    """Pinning subagents to an id the running daemon would forward upstream is
    the whole failure this replaces: the env comes from the daemon, and the
    drift from the file is reported rather than silently applied."""
    from sous import cli

    _, calls, _ = _claude_setup(
        tmp_path,
        monkeypatch,
        status=_status(local_models=["sous-fast"], max_context_tokens=131072),
        gateway_local_models=("sous-local",),
        gateway_max_context_tokens=65536,
    )
    cli.main(["claude"])
    [(_exe, _argv, env)] = calls
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sous-fast"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    err = capsys.readouterr().err
    assert "local_models" in err and "max_context_tokens" in err
    assert "restart" in err


def test_claude_refuses_without_the_claude_binary(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    assert "PATH" in capsys.readouterr().err


class _FakeSousHTTP:
    """A daemon that speaks only the two /sous/ routes, over a real socket:
    the launcher's HTTP client is under test, so no transport is faked."""

    def __init__(
        self,
        status: dict | None,
        status_code: int = 200,
        hold_code: int = 200,
        raw: bytes | None = None,
    ):
        """`raw`, when given, is sent verbatim (with the status code) for both
        routes: what a wrong daemon on the port might answer."""
        import http.server

        self.holds: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # quiet
                pass

            def _send(self, code: int, payload: bytes) -> None:
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _reply(self, code: int, body: dict) -> None:
                self._send(code, raw if raw is not None else json.dumps(body).encode())

            def do_GET(self) -> None:
                if self.path == "/sous/status" and status_code == 200:
                    self._reply(200, status or {})
                else:
                    error = {"type": "error", "error": {"type": "not_found_error", "message": ""}}
                    self._reply(status_code, error)

            def do_POST(self) -> None:
                if self.path != "/sous/hold":
                    error = {"type": "error", "error": {"type": "not_found_error", "message": ""}}
                    self._reply(404, error)
                    return
                length = int(self.headers.get("content-length", "0"))
                outer.holds.append(json.loads(self.rfile.read(length)))
                self._reply(hold_code, {"loaded": False, "loading": True, "holders": 1})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        # serve_forever notices shutdown() once per poll; the default half
        # second would be most of each test's time.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_daemon_status_reads_sous_status_over_http(tmp_path):
    from sous import cli

    fake = _FakeSousHTTP(_status())
    try:
        assert cli._daemon_status(fake.port, tmp_path) == _status()
    finally:
        fake.close()


def test_loopback_calls_ignore_a_configured_proxy(tmp_path, monkeypatch):
    """The daemon is loopback; a stray HTTP_PROXY/ALL_PROXY (no matching
    no_proxy) must never route _daemon_status or _hold through it, or see
    the hold body — same rule as gateway/upstream.py's trust_env=False."""
    from sous import cli

    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://10.255.255.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://10.255.255.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    fake = _FakeSousHTTP(_status())
    try:
        assert cli._daemon_status(fake.port, tmp_path) == _status()
        assert cli._hold(fake.port) == {"loaded": False, "loading": True, "holders": 1}
    finally:
        fake.close()


def _held_daemon_lock(data_dir: Path):
    """The flock a live daemon holds, taken in-process: flock is per open
    file description, so _lock_is_held's own open of the same file sees it."""
    import fcntl

    handle = (data_dir / "daemon.lock").open("wb")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_daemon_status_treats_a_404_from_the_lock_holder_as_a_daemon_that_predates_this_cli(
    tmp_path, capsys
):
    """An older daemon has no /sous/status: with the gateway on it forwards
    the path upstream and relays the API's 404, with it off Starlette 404s.
    Either way the answer is restart, not a fallback to MCP — and it is a
    daemon that answered, because one holds the lock."""
    from sous import cli

    fake = _FakeSousHTTP(None, status_code=404)
    lock = _held_daemon_lock(tmp_path)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._daemon_status(fake.port, tmp_path)
    finally:
        lock.close()
        fake.close()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "predates this CLI" in err and "sous stop" in err and "sous serve" in err


def test_daemon_status_names_a_404_from_something_that_is_not_sous(tmp_path, capsys):
    """Any HTTP server 404s an unknown path. With no daemon holding the lock,
    'restart the daemon' would send the user to stop a daemon that is not
    there; the port is what has to change."""
    from sous import cli

    fake = _FakeSousHTTP(None, status_code=404)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._daemon_status(fake.port, tmp_path)
    finally:
        fake.close()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not a sous daemon" in err and "[server].port" in err
    assert "predates" not in err


def test_daemon_status_reports_a_listener_that_does_not_answer(tmp_path, capsys):
    from sous import cli

    fake = _FakeSousHTTP(None, status_code=500)
    try:
        assert cli._daemon_status(fake.port, tmp_path) is None
    finally:
        fake.close()
    err = capsys.readouterr().err
    assert "did not answer /sous/status" in err and "500" in err


def test_daemon_status_is_none_when_nothing_listens(tmp_path):
    from sous import cli

    assert cli._daemon_status(1, tmp_path) is None  # port 1: never up


def test_a_status_that_is_not_json_is_a_daemon_that_does_not_answer(tmp_path, capsys):
    """Something else on the port — a 200 that is not the document."""
    from sous import cli

    fake = _FakeSousHTTP(None, raw=b"<html>nope</html>")
    try:
        assert cli._daemon_status(fake.port, tmp_path) is None
    finally:
        fake.close()
    assert "did not answer /sous/status" in capsys.readouterr().err


def test_a_reply_past_the_size_cap_is_a_daemon_that_does_not_answer(tmp_path, capsys):
    """httpx would buffer whatever the listener sends; the launcher reads
    under a cap, since what holds the port need not be the daemon."""
    from sous import cli

    fake = _FakeSousHTTP(None, raw=b"[" + b"1," * cli._REPLY_LIMIT + b"1]")
    try:
        assert cli._daemon_status(fake.port, tmp_path) is None
        assert cli._hold(fake.port) is None
    finally:
        fake.close()
    err = capsys.readouterr().err
    assert "did not answer /sous/status (ReadError)" in err
    assert "could not hold the model (ReadError)" in err


def test_a_hold_that_cannot_read_its_own_process_warns_and_returns_none(monkeypatch, capsys):
    """Everything a hold needs is inside its warning path, psutil included:
    the session works without the hold, so nothing here may stop the launch."""
    import psutil

    from sous import cli

    def denied(pid=None):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    assert cli._hold(1) is None
    assert "could not hold the model (AccessDenied); launching anyway" in capsys.readouterr().err


def test_hold_posts_this_process_and_returns_the_daemons_answer():
    import psutil

    from sous import cli

    fake = _FakeSousHTTP(_status())
    try:
        assert cli._hold(fake.port) == {"loaded": False, "loading": True, "holders": 1}
    finally:
        fake.close()
    [body] = fake.holds
    assert body["pid"] == os.getpid()
    assert abs(body["create_time"] - psutil.Process().create_time()) < 1.0
    assert set(body) == {"pid", "create_time"}


def test_a_refused_hold_warns_and_returns_none(capsys):
    from sous import cli

    fake = _FakeSousHTTP(_status(), hold_code=500)
    try:
        assert cli._hold(fake.port) is None
    finally:
        fake.close()
    err = capsys.readouterr().err
    assert "warning" in err and "hold" in err and "launching anyway" in err


def test_a_hold_answered_with_something_that_is_not_an_object_warns(capsys):
    from sous import cli

    fake = _FakeSousHTTP(_status(), raw=b"[1, 2]")
    try:
        assert cli._hold(fake.port) is None
    finally:
        fake.close()
    assert "unexpected answer" in capsys.readouterr().err


def test_claude_holds_the_model_after_the_checks_and_says_so(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude"])
    assert holds == [8383] and len(calls) == 1
    err = capsys.readouterr().err
    assert (
        "sous claude: mlx-community/Qwen3.8-27B-4bit already loaded; held while this session runs"
        in err
    )


def test_claude_reports_a_preload(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_hold", lambda port: {"loaded": False, "loading": True, "holders": 1})
    cli.main(["claude"])
    assert len(calls) == 1
    assert (
        "sous claude: preloading mlx-community/Qwen3.8-27B-4bit; held while this session runs"
        in (capsys.readouterr().err)
    )


def test_claude_launches_even_when_the_hold_is_refused(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_hold", lambda port: None)
    cli.main(["claude", "-p", "hi"])
    [(exe, argv, env)] = calls
    assert argv == ["/opt/bin/claude", "-p", "hi", "--disallowedTools", "LSP"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8383"
    assert "held while this session runs" not in capsys.readouterr().err


def test_claude_does_not_hold_when_a_check_fails(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch, status=_status(enabled=False))
    with pytest.raises(SystemExit):
        cli.main(["claude"])
    assert holds == [] and calls == []


@pytest.mark.parametrize("flag", ["--help", "-h", "--version", "-v"])
def test_claude_does_not_hold_for_an_invocation_that_exits_at_once(
    tmp_path, capsys, monkeypatch, flag
):
    """A hold for `claude --version` would start a multi-minute load nobody
    waits for and, once the process is gone, restart the idle clock under
    weights nobody is using. The preflight still runs; only the hold is skipped."""
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", flag])
    assert holds == [] and len(calls) == 1
    assert calls[0][1] == ["/opt/bin/claude", flag, "--disallowedTools", "LSP"]
    assert "held while this session runs" not in capsys.readouterr().err


@pytest.mark.slow
def test_daemon_status_reads_the_real_daemons_gateway_config(tmp_path):
    """The launcher's one source of truth, against the real server: a plain
    GET /sous/status on the daemon's port, and None when nothing is
    listening."""
    import contextlib
    from collections.abc import Iterator

    import uvicorn

    from sous.cli import _daemon_status
    from sous.config import SousConfig
    from sous.engine.base import EngineManager
    from sous.server import create_server, uvicorn_config
    from sous.tasks import TaskStore
    from tests.fake_engine import FakeEngine

    @contextlib.contextmanager
    def serve(app, port: int) -> Iterator[None]:
        server = uvicorn.Server(uvicorn_config(app, "127.0.0.1", port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started, "uvicorn never started"
        try:
            yield
        finally:
            server.should_exit = True
            thread.join(20)

    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "c.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    app = create_server(TaskStore(tmp_path / "tasks.db"), engines, cfg).streamable_http_app()
    port = _free_cli_port()
    with serve(app, port):
        status = _daemon_status(port, tmp_path)
    assert status is not None
    assert status["config"]["gateway"] == {
        "enabled": True,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }
    # Nothing listening on that port any more.
    assert _daemon_status(port, tmp_path) is None


def test_claude_skips_the_hold_for_an_exit_at_once_flag_anywhere_before_a_double_dash(
    tmp_path, capsys, monkeypatch
):
    """Claude Code honours --version and --help after other options too, so
    `--model opus --version` exits at once; after a `--` the same string is
    prompt text, and that session is held like any other."""
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", "--model", "opus", "--version"])
    assert holds == [] and len(calls) == 1
    cli.main(["claude", "-p", "--", "--help"])
    assert holds == [8383] and len(calls) == 2


def test_claude_refuses_a_status_document_without_the_gateway_values(tmp_path, capsys, monkeypatch):
    """A 200 that is an object but not the document: a KeyError traceback
    would name nothing the user can act on."""
    from sous import cli

    status = _status()
    del status["config"]["gateway"]["local_models"]
    _, calls, holds = _claude_setup(tmp_path, monkeypatch, status=status)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == [] and holds == []
    err = capsys.readouterr().err
    assert "unexpected document" in err and "sous serve" in err


def test_claude_tells_a_listener_that_did_not_answer_from_no_daemon(tmp_path, capsys, monkeypatch):
    """`sous serve` against a bound port fails on the daemon lock, so the
    advice for a listener that did not answer is a restart, not a start."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch, status=None)
    monkeypatch.setattr(cli, "_port_open", lambda port: True)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    err = capsys.readouterr().err
    assert "did not answer as a sous daemon" in err and "sous stop" in err
    assert "no daemon" not in err


def test_claude_reports_a_hold_that_is_neither_loaded_nor_loading(tmp_path, capsys, monkeypatch):
    """The daemon could not start its preload thread: the hold stands, the
    first turn loads, and the line says so rather than claiming a load."""
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli, "_hold", lambda port: {"loaded": False, "loading": False, "holders": 1}
    )
    cli.main(["claude"])
    assert len(calls) == 1
    assert (
        "sous claude: mlx-community/Qwen3.8-27B-4bit not loaded yet; held while this session runs"
        in capsys.readouterr().err
    )


# --- sous statusline --------------------------------------------------------------


def _document(**overrides) -> dict:
    doc = {
        "engine": {"loaded": True, "loading": False, "holders": 0, "prompt_cache": {"slots": 5}},
        "inflight": [],
        "queue": {"queued": 0, "running": 0},
        "config": {},
        "recent_turns": [],
        "recent_tasks": [],
    }
    doc.update(overrides)
    return doc


def _turn(**overrides) -> dict:
    turn = {
        "id": "msg_1",
        "phase": "decode",
        "started_at": 1_000.0,
        "phase_since": 1_010.0,
        "generated_tokens": 612,
        "to_prefill": None,
        "decode_tps": 14.7,
        "eta_seconds": 18.4,
    }
    turn.update(overrides)
    return turn


def test_statusline_text_for_each_state():
    from sous.cli import statusline_text

    now = 1_042.0
    assert statusline_text(_document(inflight=[_turn()]), now) == (
        "sous: decode 612 tok · 14.7 tok/s · eta 18s"
    )
    assert statusline_text(_document(inflight=[_turn(decode_tps=None, eta_seconds=None)]), now) == (
        "sous: decode 612 tok · 00:42"
    )
    assert (
        statusline_text(
            _document(inflight=[_turn(phase="prefill", to_prefill=3120, eta_seconds=4.0)]), now
        )
        == "sous: prefill 3,120 tok · eta 4s"
    )
    assert statusline_text(_document(inflight=[_turn(phase="prefill")]), now) == (
        "sous: prefill · 00:42"
    )
    assert statusline_text(_document(inflight=[_turn(phase="loading")]), now) == (
        "sous: loading · 00:42"
    )
    queued = _document(inflight=[_turn(), _turn(id="msg_2", phase="queued")])
    assert statusline_text(queued, now) == (
        "sous: decode 612 tok · 14.7 tok/s · eta 18s · +1 queued"
    )
    assert statusline_text(_document(), now) == "sous: idle · 5 slots"
    assert statusline_text(_document(engine={"loaded": True, "holders": 2}), now) == (
        "sous: idle · held"
    )
    assert statusline_text(_document(engine={"loaded": False, "loading": True}), now) == (
        "sous: loading"
    )
    assert statusline_text(_document(engine={"loaded": False, "loading": False}), now) == (
        "sous: idle · model unloaded"
    )
    assert statusline_text({}, now) == "sous: idle · model unloaded"


def _statusline_config(tmp_path, monkeypatch, port: int):
    from sous import cli
    from sous.config import SousConfig

    cfg = SousConfig(server_port=port, data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)


def test_statusline_reads_the_daemon_and_prints_one_line(tmp_path, capsys, monkeypatch):
    import io

    from sous import cli

    fake = _FakeSousHTTP(_document(inflight=[_turn()]))
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        # Claude Code pipes its own JSON in; it is read and ignored.
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"model": {"id": "claude-opus-5"}}'))
        cli.main(["statusline"])
    finally:
        fake.close()
    out = capsys.readouterr().out
    assert out == "sous: decode 612 tok · 14.7 tok/s · eta 18s\n"


def test_statusline_says_daemon_down_and_exits_zero(tmp_path, capsys, monkeypatch):
    from sous import cli

    _statusline_config(tmp_path, monkeypatch, _free_cli_port())
    cli.main(["statusline"])
    assert capsys.readouterr().out == "sous: daemon down\n"


def test_statusline_treats_a_wrong_answer_as_daemon_down(tmp_path, capsys, monkeypatch):
    from sous import cli

    fake = _FakeSousHTTP(None, raw=b"<html>not sous</html>")
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        cli.main(["statusline"])
    finally:
        fake.close()
    assert capsys.readouterr().out == "sous: daemon down\n"


class _RawListener:
    """A socket on 127.0.0.1 that answers its first connection with
    `script` — (seconds to wait, bytes to send) in order — then hangs up:
    what a port held by something other than an HTTP server, or by a server
    that dribbles, looks like to the status line."""

    def __init__(self, script: list[tuple[float, bytes]]):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.script = script
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
            with conn:
                for delay, chunk in self.script:
                    time.sleep(delay)
                    conn.sendall(chunk)
        except OSError:
            pass  # the test closed the listener, or the client is gone

    def close(self) -> None:
        self.sock.close()


def test_statusline_treats_a_listener_that_is_not_http_as_daemon_down(
    tmp_path, capsys, monkeypatch
):
    """urllib's URLError is an OSError; http.client's BadStatusLine — a port
    held by something that does not speak HTTP — is not, and once a second
    a traceback is no status line."""
    from sous import cli

    listener = _RawListener([(0.0, b"i am not http\r\n\r\n")])
    try:
        _statusline_config(tmp_path, monkeypatch, listener.port)
        cli.main(["statusline"])
    finally:
        listener.close()
    assert capsys.readouterr().out == "sous: daemon down\n"


def test_statusline_gives_up_on_a_listener_that_dribbles(tmp_path, capsys, monkeypatch):
    """urllib's timeout is per socket read and re-arms on every byte, so a
    listener that dribbles one byte at a time would hold the line for as
    long as it liked. The budget is wall-clock, on the whole call."""
    from sous import cli

    headers = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
    listener = _RawListener([(0.0, headers), *([(0.3, b"{")] * 10)])
    started = time.monotonic()
    try:
        _statusline_config(tmp_path, monkeypatch, listener.port)
        cli.main(["statusline"])
    finally:
        listener.close()
    assert capsys.readouterr().out == "sous: daemon down\n"
    assert time.monotonic() - started < 1.5


def test_statusline_treats_someone_elses_json_as_daemon_down(tmp_path, capsys, monkeypatch):
    """A JSON object from whatever else holds the port is not a status
    document: without the engine block it would read as a loaded, idle
    daemon and say so."""
    from sous import cli

    fake = _FakeSousHTTP({"ok": True})
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        cli.main(["statusline"])
    finally:
        fake.close()
    assert capsys.readouterr().out == "sous: daemon down\n"


def test_statusline_ignores_a_configured_proxy(tmp_path, capsys, monkeypatch):
    from sous import cli

    fake = _FakeSousHTTP(_document())
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        cli.main(["statusline"])
    finally:
        fake.close()
    assert capsys.readouterr().out == "sous: idle · 5 slots\n"


def test_statusline_imports_nothing_heavy(tmp_path):
    """It runs once a second under Claude Code: the stdlib only. Checked in
    a fresh interpreter, since this test process has long since imported
    httpx for the launcher's tests."""
    fake = _FakeSousHTTP(_document())
    try:
        probe = (
            "import sys\n"
            "from pathlib import Path\n"
            "from sous import cli\n"
            "from sous.config import SousConfig\n"
            f"cfg = SousConfig(server_port={fake.port}, data_dir=Path(sys.argv[1]), "
            "config_path=Path(sys.argv[1]) / 'c.toml')\n"
            "cli.load_config = lambda: cfg\n"
            "cli.main(['statusline'])\n"
            "print(sorted(m for m in ('textual', 'httpx', 'mlx', 'psutil') if m in sys.modules))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    finally:
        fake.close()
    assert done.returncode == 0, done.stderr
    assert done.stdout == "sous: idle · 5 slots\n[]\n"


# --- sous top ---------------------------------------------------------------------------


def test_top_and_status_watch_run_the_terminal(tmp_path, monkeypatch):
    from sous import cli, tui
    from sous.config import SousConfig

    cfg = SousConfig(server_port=8391, data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_has_terminal", lambda: True)
    ports: list[int] = []
    monkeypatch.setattr(tui, "run_top", lambda port: (ports.append(port), 3)[1])
    with pytest.raises(SystemExit) as exc:
        cli.main(["top"])
    assert exc.value.code == 3
    with pytest.raises(SystemExit) as exc:
        cli.main(["status", "--watch"])
    assert exc.value.code == 3
    assert ports == [8391, 8391]


def test_top_without_a_terminal_says_so_instead_of_drawing_into_the_pipe(
    tmp_path, capsys, monkeypatch
):
    """An agent's shell, or a redirect: Textual would paint the pass into
    the pipe and wait for a key that never comes."""
    from sous import cli, tui
    from sous.config import SousConfig

    cfg = SousConfig(server_port=8391, data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_has_terminal", lambda: False)
    monkeypatch.setattr(tui, "run_top", lambda port: pytest.fail("the terminal was started"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["top"])
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == "" and err == "sous top: needs a terminal (try sous status, or sous statusline)\n"


def test_only_sous_top_imports_textual(tmp_path):
    """The daemon, the launcher's module and the status line load without
    the UI framework; a fresh interpreter proves it, since this process has
    imported it for the terminal's own tests."""
    probe = (
        "import sys\n"
        "import sous.cli, sous.server, sous.monitor, sous.inflight\n"
        "print('textual' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "False\n"


def test_tune_subcommand_dispatches_with_its_flags(monkeypatch):
    import sous.tune
    from sous import cli

    seen = {}
    monkeypatch.setattr(sous.tune, "main", lambda args, **k: seen.update(vars(args)) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(
            ["tune", "--quick", "--repeat", "3", "--runs", "3", "--models", "a/b", "c/d", "--yes"]
        )
    assert e.value.code == 0
    assert seen["quick"] is True and seen["repeat"] == 3
    assert seen["runs"] == 3
    assert seen["models"] == ["a/b", "c/d"] and seen["yes"] is True and seen["apply"] is False


def test_restart_hint_names_launchds_kickstart_or_the_two_commands():
    from sous.cli import restart_hint

    assert restart_hint(managed=False) == "sous stop, then sous serve"
    assert restart_hint(managed=True) == f"launchctl kickstart -k gui/{os.getuid()}/{LABEL}"
