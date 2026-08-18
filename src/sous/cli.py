"""sous CLI: serve / status / wait / stop / mcp / install- and uninstall-launchd."""

from __future__ import annotations

import argparse
import fcntl
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
            # launchd starts agents with the bare system PATH, and the sandbox
            # passes PATH through to worker commands — so without this neither
            # `uv` nor any user-installed tool resolves, and every allowlisted
            # verify command silently degrades into a human approval. The
            # executable's own directory is where uv-tool shims (sous, uv) live.
            "EnvironmentVariables": {
                "PATH": f"{Path(sous_executable).parent}:"
                "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            },
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


def _cmd_wait(task_id: str, timeout: float | None, interval: float) -> None:
    """Block until the task needs attention, so agents can park this in a
    background shell instead of tight-polling task_status — or, worse, reading
    tasks.db by hand (observed in the wild; the schema is not a contract).

    Wakes on awaiting_approval as well as the terminal states: an approval
    request needs a human NOW, and a wait that slept through it would let the
    request time out into an auto-deny.
    """
    config = load_config()
    from sous.tasks import FINISHED_STATES, TaskState, TaskStore

    store = TaskStore(config.data_dir / "tasks.db")
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    while True:
        t = store.get(task_id)
        if t is None:
            print(f"sous: unknown task {task_id}")
            raise SystemExit(2)
        if t.state in FINISHED_STATES or t.state == TaskState.AWAITING_APPROVAL:
            line = f"state={t.state}"
            if t.outcome:
                line += f" outcome={t.outcome}"
            if t.state == TaskState.AWAITING_APPROVAL and t.pending_command:
                line += f" pending_command={t.pending_command}"
            print(line)
            return
        if deadline is not None and time.monotonic() >= deadline:
            print(f"state={t.state} (timeout)")
            raise SystemExit(1)
        time.sleep(interval)


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


_UNLOAD_GRACE_SECONDS = 5.0


def _await_port_closed(port: int, seconds: float) -> bool:
    """Wait up to `seconds` for the port to stop accepting."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _port_open(port):
            return True
        time.sleep(0.2)
    return not _port_open(port)


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


# launchctl's exit code for a service target that is not loaded. Every other
# nonzero code is a real failure and must not be mistaken for "already gone".
_BOOTOUT_NOT_LOADED = 3


def _bootout(label: str) -> int:
    """Unload the job by service target, returning launchctl's exit code.

    By label rather than by plist path: the plist may have been deleted by hand
    while the label is still bootstrapped, and that is precisely the state
    where uninstalling has to do something.
    """
    try:
        done = subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return _BOOTOUT_NOT_LOADED  # no launchctl here: nothing to unload
    return done.returncode


def _daemon_pid(data_dir: Path) -> int | None:
    """The pid the running daemon recorded when it took its lock."""
    try:
        return int((data_dir / "daemon.lock").read_text().strip())
    except OSError, ValueError:
        return None


def _lock_is_held(data_dir: Path) -> bool:
    """Whether a live daemon currently holds the lock.

    daemon.lock outlives the daemon that wrote it, so the pid inside proves
    nothing on its own — the OS may have recycled it onto an unrelated process,
    and a port probe only shows that *something* is listening. The flock is the
    authoritative signal: if we can take it, nobody is holding it.
    """
    try:
        handle = (data_dir / "daemon.lock").open("r+b")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        handle.close()  # releases anything we just took
    return False


def _cmd_stop() -> None:
    config = load_config()
    # launchd first, deliberately. A KeepAlive job can be loaded while its
    # daemon is mid-restart; checking the port first would report "not running"
    # and exit 0 in exactly that window, while launchd brings it straight back.
    if _launchd_loaded(LABEL):
        print("sous daemon: managed by launchd, which would restart it immediately")
        print("  remove it:  sous uninstall-launchd")
        print(f"  restart it: launchctl kickstart -k gui/{os.getuid()}/{LABEL}")
        raise SystemExit(1)
    if not _port_open(config.server_port):
        print(f"sous daemon: not running (port {config.server_port})")
        return
    pid = _daemon_pid(config.data_dir)
    if pid is None:
        print(f"sous daemon: listening on {config.server_port}, but no pid in daemon.lock")
        print("  stop it by hand, or upgrade the running daemon to one that records it")
        raise SystemExit(1)
    if not _lock_is_held(config.data_dir):
        print(f"sous: daemon.lock names pid {pid} but nothing holds the lock")
        print(f"  something else is on port {config.server_port}; not signalling a stale pid")
        raise SystemExit(1)

    from sous.tasks import TaskState, TaskStore

    # count_by_state aggregates every row; list_recent() caps at 20 and would
    # miss a long-running task once newer ones are queued past it.
    counts = TaskStore(config.data_dir / "tasks.db").count_by_state()
    interrupted = {
        state: n
        for state, n in counts.items()
        if state in (TaskState.RUNNING, TaskState.AWAITING_APPROVAL) and n
    }
    if interrupted:
        summary = ", ".join(f"{n} {state}" for state, n in sorted(interrupted.items()))
        print(f"sous: {summary}; these will be reported failed when the daemon restarts")

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"sous: pid {pid} is already gone")
        return
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
    # Unload first and unconditionally: a plist deleted by hand leaves the label
    # bootstrapped and the daemon running, which is exactly the state where
    # returning early would claim to have uninstalled something and not have.
    code = _bootout(LABEL)
    if code not in (0, _BOOTOUT_NOT_LOADED):
        # Anything else (permissions, launchctl error) leaves the KeepAlive job
        # alive; deleting the plist here would report success over a live daemon.
        print(f"sous: launchctl bootout failed (exit {code}); leaving {plist_path} in place")
        raise SystemExit(1)

    had_plist = plist_path.exists()
    plist_path.unlink(missing_ok=True)
    if had_plist:
        print(f"removed {plist_path}")
    if code == 0:
        print("unloaded the launchd agent; it will no longer start at login")
    elif not had_plist:
        print(f"sous: launchd agent not installed ({plist_path})")
        return

    config = load_config()
    if code == 0:
        # bootout terminates the job it just unloaded, so checking the port
        # straight away races that teardown: the daemon is still accepting for
        # a moment, and advising `sous stop` sends the user after a process
        # that is already exiting — they run it and are told "not running",
        # which reads as bad advice rather than early advice. Only a daemon
        # that outlives the unload was never launchd's to begin with.
        _await_port_closed(config.server_port, _UNLOAD_GRACE_SECONDS)
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
    wait = sub.add_parser("wait", help="block until a task finishes or requests a command approval")
    wait.add_argument("task_id")
    wait.add_argument(
        "--timeout", type=float, default=None, help="give up after N seconds (exit 1)"
    )
    wait.add_argument("--interval", type=float, default=2.0, help="poll every N seconds")
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
    elif args.command == "wait":
        _cmd_wait(args.task_id, args.timeout, args.interval)
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "install-launchd":
        _cmd_install_launchd()
    elif args.command == "uninstall-launchd":
        _cmd_uninstall_launchd()
