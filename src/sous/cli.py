"""sous CLI: serve / status / wait / stop / mcp / claude / install- and uninstall-launchd."""

from __future__ import annotations

import argparse
import fcntl
import http.client
import math
import os
import plistlib
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from sous.config import SousConfig, load_config

LABEL = "com.sous.daemon"

# What `sous claude` sets, and — as important — what it never sets. Either
# credential variable switches Claude Code from the subscription login to
# API-credit billing (the load-bearing #41 fact); the tier variables would pull
# the main loop onto the local model. oMLX's launcher sets all of them because
# it never forwards anything; sous forwards the main loop, so it must not.
_CREDENTIAL_VARS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
_DISALLOWED_FLAGS = ("--disallowedTools", "--disallowed-tools")
_LSP_OFF = ["--disallowedTools", "LSP"]
# Model load plus a long prefill: minutes, not the SDK's default.
_API_TIMEOUT_MS = "3000000"
_PROBE_TIMEOUT_SECONDS = 15.0


def claude_argv(user_args: list[str]) -> list[str]:
    """The user's arguments plus `--disallowedTools LSP` (a language server
    connecting mid-session appends its schema to every request and re-prefills
    the conversation) unless they chose their own. Appended, never prepended:
    the option is variadic, so ahead of a positional prompt it would swallow
    the prompt as a tool name. Ahead of a `--`, which ends Claude Code's
    option parsing, if there is one — and only the arguments before that `--`
    can be the user's own choice, since anything after it is literal text
    (`sous claude -- --disallowedTools` asks for that prompt, not that flag)."""
    end = user_args.index("--") if "--" in user_args else len(user_args)
    for arg in user_args[:end]:
        if arg in _DISALLOWED_FLAGS or arg.startswith(tuple(f"{f}=" for f in _DISALLOWED_FLAGS)):
            return list(user_args)
    return [*user_args[:end], *_LSP_OFF, *user_args[end:]]


def claude_env(config: SousConfig, base: Mapping[str, str]) -> dict[str, str]:
    """The inherited environment plus the four variables the gateway needs.

    CLAUDE_CODE_MAX_CONTEXT_TOKENS is honoured only for non-claude-* ids, so
    it sizes the local subagent's window and leaves the main loop alone.
    CLAUDE_CODE_AUTO_COMPACT_WINDOW is deliberately NOT set: it is global
    ("the minimum of this setting and your model's maximum context window",
    per Claude Code's own /config copy), so pinning it to the local window
    would make the frontier main loop compact far too early. The subagent's
    threshold is already bounded by its window.
    """
    env = dict(base)
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{config.server_port}"
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = config.gateway_local_models[0]
    env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(config.gateway_max_context_tokens)
    env["API_TIMEOUT_MS"] = _API_TIMEOUT_MS
    return env


def _probe_gateway(port: int) -> tuple[int, bool] | None:
    """(status, did the gateway forward it?) for HEAD /api/hello, or None when
    nothing is listening. A forwarded answer — or the forwarder's own error —
    carries `Via: 1.1 sous`; the daemon's own "no such route" 404 does not.
    stdlib http.client on purpose: the CLI should not pay for an async HTTP
    client to run `sous status`."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=_PROBE_TIMEOUT_SECONDS)
    try:
        conn.request("HEAD", "/api/hello")
        response = conn.getresponse()
        return response.status, "sous" in (response.getheader("via") or "")
    except OSError, http.client.HTTPException:
        return None
    finally:
        conn.close()


def _cmd_claude(user_args: list[str]) -> None:
    """Replace this process with Claude Code pointed at the gateway."""
    config = load_config()
    if not config.gateway_enabled:
        print(
            f"sous claude: the gateway is off; set [gateway].enabled = true in "
            f"{config.config_path} and restart the daemon",
            file=sys.stderr,
        )
        raise SystemExit(1)
    exe = shutil.which("claude")
    if exe is None:
        print("sous claude: `claude` is not on PATH; install Claude Code first", file=sys.stderr)
        raise SystemExit(1)
    probe = _probe_gateway(config.server_port)
    if probe is None:
        print(
            f"sous claude: no daemon on 127.0.0.1:{config.server_port}; start it with: "
            "sous serve   (or: sous install-launchd)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    status, forwarded = probe
    if status == 404 and not forwarded:
        print(
            "sous claude: the daemon is running without the gateway (started before "
            "[gateway].enabled was set?); restart it and try again",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if status >= 500:
        print(
            f"sous claude: warning: the gateway could not reach its upstream just now "
            f"(HTTP {status}); launching anyway — Claude Code will say if it persists",
            file=sys.stderr,
        )
    for var in _CREDENTIAL_VARS:
        if os.environ.get(var):
            print(
                f"sous claude: warning: {var} is set, so Claude Code will bill the main loop "
                "to API credits instead of your subscription; unset it to use the login",
                file=sys.stderr,
            )
    env = claude_env(config, os.environ)
    print(
        f"sous claude: ANTHROPIC_BASE_URL={env['ANTHROPIC_BASE_URL']} "
        f"CLAUDE_CODE_SUBAGENT_MODEL={env['CLAUDE_CODE_SUBAGENT_MODEL']} "
        f"CLAUDE_CODE_MAX_CONTEXT_TOKENS={env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']} "
        f"API_TIMEOUT_MS={env['API_TIMEOUT_MS']}",
        file=sys.stderr,
    )
    # exec, not a subprocess: the TTY, the signals and the exit code are
    # Claude Code's own from here on.
    os.execve(exe, [exe, *claude_argv(user_args)], env)


def launchd_plist(sous_executable: str, log_dir: Path) -> str:
    # plistlib handles XML escaping — a path containing & or < must still
    # produce a plist launchctl can parse (string formatting silently
    # produced invalid XML while install-launchd reported success).
    # No EnvironmentVariables.PATH here on purpose: the daemon adopts the
    # user's login-shell PATH itself at startup (server._login_shell_path),
    # which works identically however it was launched — an install-time
    # snapshot in the plist would just be a second, staler mechanism.
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
        # Cap each sleep to the remaining budget: sleeping a full interval and
        # only then checking would quantize the deadline to interval boundaries
        # — and a task finishing inside that overrun would be reported as a
        # success AFTER the caller's timeout. (--timeout 0 thereby becomes the
        # non-blocking probe: one state check, then report.)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            print(f"state={t.state} (timeout)")
            raise SystemExit(1)
        time.sleep(interval if remaining is None else min(interval, remaining))


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


def _arg_interval(text: str) -> float:
    """A zero interval recreates the tight-polling `wait` exists to prevent, a
    negative one raises out of time.sleep, and NaN poisons the sleep math —
    reject all three as usage errors instead of misbehaving at runtime."""
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("interval must be a positive finite number of seconds")
    return value


def _arg_timeout(text: str) -> float:
    """NaN never compares past the deadline (the wait would ignore an explicit
    timeout and block forever); negatives are nonsense. Zero is allowed and
    defined: an immediate, non-blocking probe."""
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("timeout must be a non-negative finite number of seconds")
    return value


def main(argv: list[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    if raw[:1] == ["claude"]:
        # Before argparse: every argument after `claude` is Claude Code's,
        # including a leading `-p` or `--help`, which argparse.REMAINDER would
        # reject as an unknown option of ours.
        _cmd_claude(raw[1:])
        return
    parser = argparse.ArgumentParser(prog="sous", description="local MLX sous-chef for Claude")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the daemon (MCP over HTTP on 127.0.0.1)")
    sub.add_parser("status", help="check the daemon and recent tasks")
    wait = sub.add_parser("wait", help="block until a task finishes or requests a command approval")
    wait.add_argument("task_id")
    wait.add_argument(
        "--timeout",
        type=_arg_timeout,
        default=None,
        help="give up after N seconds, exit 1 (0 = non-blocking probe)",
    )
    wait.add_argument("--interval", type=_arg_interval, default=2.0, help="poll every N seconds")
    sub.add_parser("stop", help="stop the daemon (unmanaged daemons only)")
    sub.add_parser("mcp", help="bridge stdio to the daemon (for stdio-only MCP clients)")
    sub.add_parser("install-launchd", help="install start-at-login LaunchAgent")
    sub.add_parser("uninstall-launchd", help="remove the start-at-login LaunchAgent")
    # Registered for `sous --help` only: the verb is dispatched at the top of
    # main(), before argparse ever sees it, so there is no `claude` branch below.
    sub.add_parser(
        "claude",
        help="launch Claude Code against the gateway: subagents local, main loop upstream "
        "(every following argument passes through to claude)",
    )
    args = parser.parse_args(raw)
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
