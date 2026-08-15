import plistlib
from pathlib import Path

from sous.cli import LABEL, launchd_plist


def test_plist_is_valid_and_correct():
    xml = launchd_plist("/usr/local/bin/sous", Path("/Users/x/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == ["/usr/local/bin/sous", "serve"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["StandardOutPath"] == "/Users/x/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/.sous/daemon.err.log"


def test_status_reports_not_running(tmp_path, capsys, monkeypatch):
    from sous import cli
    from sous.config import SousConfig
    cfg = SousConfig(server_port=1, data_dir=tmp_path,  # port 1: never listening
                     config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "not running" in out
