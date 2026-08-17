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


def _fake_daemon(tmp_path: Path, port: int):
    """A stand-in that holds the port and records its pid, like the real one.

    Holding the port matters: `stop` waits for it to close, so a victim that
    does not own it would let the test pass against code that never signalled.
    """
    # It must ACCEPT, not just listen: `stop` probes the port repeatedly while
    # waiting for it to close, and an unaccepted backlog fills up and starts
    # refusing — which reads as "daemon not running" and hides the real result.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import socket;s=socket.socket();"
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
    monkeypatch.setattr(cli, "_bootout", lambda p: booted.append(p))
    cli.main(["uninstall-launchd"])
    assert booted == [plist]
    assert not plist.exists()


def test_uninstall_launchd_is_idempotent(tmp_path, capsys, monkeypatch):
    """Running it twice, or with nothing installed, must not blow up."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"  # never created
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda p: None)
    cli.main(["uninstall-launchd"])
    assert "not installed" in capsys.readouterr().out.lower()
