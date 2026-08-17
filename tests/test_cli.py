import plistlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sous.cli import LABEL, launchd_plist


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
    assert data["StandardErrorPath"] == "/Users/x/.sous/daemon.err.log"


def test_plist_escapes_xml_special_characters():
    """C5: a valid macOS path containing & or < must yield a plist that
    launchctl can actually parse — not silently-invalid XML."""
    xml = launchd_plist("/opt/a&b/sous", Path("/Users/x/My & <Special> Docs/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["ProgramArguments"] == ["/opt/a&b/sous", "serve"]
    assert data["StandardOutPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.err.log"


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
