"""sous CLI: serve / status / install-launchd."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from sous.config import load_config

LABEL = "com.sous.daemon"

_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log_dir}/daemon.log</string>
    <key>StandardErrorPath</key><string>{log_dir}/daemon.err.log</string>
</dict>
</plist>
"""


def launchd_plist(sous_executable: str, log_dir: Path) -> str:
    return _PLIST_TEMPLATE.format(label=LABEL, exe=sous_executable, log_dir=log_dir)


def _cmd_status() -> None:
    config = load_config()
    try:
        with socket.create_connection(("127.0.0.1", config.server_port), timeout=1):
            print(f"sous daemon: listening on 127.0.0.1:{config.server_port}")
    except OSError:
        print(f"sous daemon: not running (port {config.server_port})")
        print("start it with: sous serve   (or: sous install-launchd)")
        return
    from sous.tasks import TaskStore
    store = TaskStore(config.data_dir / "tasks.db")
    for t in store.list_recent(limit=10):
        print(f"  {t.id}  {t.state:<18} {t.title}")


def _cmd_install_launchd() -> None:
    config = load_config()
    exe = shutil.which("sous") or sys.argv[0]
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(launchd_plist(exe, config.data_dir))
    print(f"wrote {plist_path}")
    cmd = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)]
    try:
        subprocess.run(cmd, check=True)
        print("daemon loaded; it will start at login and stay alive")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("could not load automatically; run manually:")
        print("  " + " ".join(cmd))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sous",
                                     description="local MLX sous-chef for Claude")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the daemon (MCP over HTTP on 127.0.0.1)")
    sub.add_parser("status", help="check the daemon and recent tasks")
    sub.add_parser("install-launchd", help="install start-at-login LaunchAgent")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from sous.server import main as serve_main
        serve_main()
    elif args.command == "status":
        _cmd_status()
    elif args.command == "install-launchd":
        _cmd_install_launchd()
