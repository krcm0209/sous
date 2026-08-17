"""sous CLI: serve / status / stop / mcp / install- and uninstall-launchd."""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from sous.config import load_config

LABEL = "com.sous.daemon"


def launchd_plist(sous_executable: str, log_dir: Path) -> str:
    # plistlib handles XML escaping — a path containing & or < must still
    # produce a plist launchctl can parse (string formatting silently
    # produced invalid XML while install-launchd reported success).
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [sous_executable, "serve"],
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": f"{log_dir}/daemon.log",
            "StandardErrorPath": f"{log_dir}/daemon.err.log",
        },
        sort_keys=False,
    ).decode()


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


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _launchd_loaded(label: str) -> bool:
    """Whether launchd is currently managing the daemon.

    Absent or unusable launchctl reads as "not managed": the caller then falls
    back to signalling directly, which is the right answer on a machine where
    launchd is not in the picture.
    """
    try:
        listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10)
    except OSError, subprocess.SubprocessError:
        return False
    return any(line.split("\t")[-1] == label for line in listed.stdout.splitlines())


def _bootout(plist_path: Path) -> None:
    # Not check=True: "not loaded" is a fine state to be in when uninstalling.
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(plist_path)],
        capture_output=True,
        timeout=10,
    )


def _daemon_pid(data_dir: Path) -> int | None:
    """The pid the running daemon recorded when it took its lock."""
    try:
        return int((data_dir / "daemon.lock").read_text().strip())
    except OSError, ValueError:
        return None


def _cmd_stop() -> None:
    config = load_config()
    if not _port_open(config.server_port):
        print(f"sous daemon: not running (port {config.server_port})")
        return
    if _launchd_loaded(LABEL):
        # KeepAlive would restart it within a second, so a signal here would
        # look like a no-op rather than a refusal.
        print("sous daemon: managed by launchd, which would restart it immediately")
        print("  remove it:  sous uninstall-launchd")
        print(f"  restart it: launchctl kickstart -k gui/{os.getuid()}/{LABEL}")
        raise SystemExit(1)
    pid = _daemon_pid(config.data_dir)
    if pid is None:
        print(f"sous daemon: listening on {config.server_port}, but no pid in daemon.lock")
        print("  stop it by hand, or upgrade the running daemon to one that records it")
        raise SystemExit(1)

    from sous.tasks import TaskState, TaskStore

    running = [
        t
        for t in TaskStore(config.data_dir / "tasks.db").list_recent()
        if t.state == TaskState.RUNNING
    ]
    for t in running:
        print(f"sous: task {t.id} ({t.title}) is running; it will be reported failed on restart")

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _port_open(config.server_port):
        time.sleep(0.2)
    if _port_open(config.server_port):
        print(f"sous daemon: sent SIGTERM to {pid} but port {config.server_port} is still open")
        raise SystemExit(1)
    print(f"sous daemon: stopped (pid {pid})")
    print("any running `sous mcp` bridges will exit on their own")


def _cmd_uninstall_launchd() -> None:
    plist_path = _plist_path()
    if not plist_path.exists():
        print(f"sous: launchd agent not installed ({plist_path})")
        return
    _bootout(plist_path)
    plist_path.unlink(missing_ok=True)
    print(f"removed {plist_path}")
    print("the daemon will no longer start at login")
    config = load_config()
    if _port_open(config.server_port):
        print("the running daemon is now unmanaged; stop it with: sous stop")


def _cmd_install_launchd() -> None:
    config = load_config()
    exe = shutil.which("sous") or sys.argv[0]
    plist_path = _plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(launchd_plist(exe, config.data_dir))
    print(f"wrote {plist_path}")
    cmd = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)]
    try:
        subprocess.run(cmd, check=True)
        print("daemon loaded; it will start at login and stay alive")
    except subprocess.CalledProcessError, FileNotFoundError:
        print("could not load automatically; run manually:")
        print("  " + " ".join(cmd))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sous", description="local MLX sous-chef for Claude")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the daemon (MCP over HTTP on 127.0.0.1)")
    sub.add_parser("status", help="check the daemon and recent tasks")
    sub.add_parser("stop", help="stop the daemon (unmanaged daemons only)")
    sub.add_parser("mcp", help="bridge stdio to the daemon (for stdio-only MCP clients)")
    sub.add_parser("install-launchd", help="install start-at-login LaunchAgent")
    sub.add_parser("uninstall-launchd", help="remove the start-at-login LaunchAgent")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from sous.server import main as serve_main

        serve_main()
    elif args.command == "mcp":
        # Attribute lookup, not `from ... import run`: the exit code has to
        # reach the launching client, and this stays patchable for tests.
        import sous.proxy

        raise SystemExit(sous.proxy.run())
    elif args.command == "status":
        _cmd_status()
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "install-launchd":
        _cmd_install_launchd()
    elif args.command == "uninstall-launchd":
        _cmd_uninstall_launchd()
