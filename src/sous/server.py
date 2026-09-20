"""SousService and the daemon's MCP server: no tools of its own, just the
monitor and gateway routes mounted on it, plus daemon main()."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import pwd
import re
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO

import anyio
import uvicorn
from mcp.server import MCPServer

from sous.config import SousConfig, load_config
from sous.engine.base import EngineManager, release_mlx_thread_state
from sous.gateway.routes import Gateway, mount_gateway
from sous.gateway.upstream import Upstream
from sous.inflight import Inflight
from sous.logs import configure_daemon_logging, enable_warning_capture, format_line
from sous.monitor import mount_monitor
from sous.toolexec import terminate_active_commands

_logger = logging.getLogger("sous.server")


def _mlx_memory_gb() -> float | None:
    try:
        import mlx.core as mx

        return round(mx.get_active_memory() / 1e9, 2)
    except Exception:  # noqa: BLE001 — mlx absent or API moved
        return None
    finally:
        # Runs in whatever short-lived MCP worker thread served the request;
        # mlx state left behind segfaults that thread's eventual exit
        # (ml-explore/mlx#4327).
        release_mlx_thread_state()


class SousService:
    def __init__(
        self,
        engines: EngineManager,
        config: SousConfig,
        inflight: Inflight | None = None,
    ):
        self.engines = engines
        self.config = config
        # The gateway's turns report here; create_server hands the same
        # registry to both, so the status document sees what the gateway does.
        self.inflight = inflight or Inflight()

    def _config_stamp(self) -> tuple[int, int]:
        """The config file's mtime_ns and size, (-1, -1) when there is no
        file: what the status version keys on. The size is there for a
        write that keeps the mtime and changes the length — a `cp -p`, a
        backup restore — which the stamp would otherwise never see; one
        that keeps both is still invisible."""
        try:
            st = self.config.config_path.stat()
        except OSError:
            return (-1, -1)
        return (st.st_mtime_ns, st.st_size)

    def status_version(self) -> tuple[int, int, tuple[int, int]]:
        """What the event stream polls between documents: the in-flight
        registry's version, the engine manager's, and the config file's
        stamp. Three cheap reads and no lock; a change to any of them is a
        document worth sending, and nothing else is."""
        return (self.inflight.version, self.engines.version, self._config_stamp())

    def status_document(self, *, recent: bool) -> dict:
        """The one document every status surface serves. `recent=True` adds
        the last fifty turns, for the HTTP routes and the terminal;
        `recent=False` is the narrower shape, for a caller where every key
        read has a cost. Runs on a worker thread: the engine's status takes
        its lock, and the registry's snapshot may call into the prompt
        cache."""
        try:
            engine = self.engines.status()
            live = self.inflight.snapshot()
            # Last of all: the read releases this thread's mlx state, and the
            # snapshot's probe reaches the prompt cache, which can free mlx
            # arrays. Freeing after the release is what segfaults the thread
            # on its way out (ml-explore/mlx#4327).
            engine["memory_gb"] = _mlx_memory_gb()
        finally:
            # The status and the snapshot free arrays the same way before the
            # memory read, and the routes that call this catch whatever they
            # raise: a build that fails between the two would otherwise hand
            # a pooled thread back with its mlx state live.
            release_mlx_thread_state()
        document = {
            "engine": engine,
            "inflight": live["inflight"],
            "config": {
                "model_id": self.config.model_id,
                "idle_unload_minutes": self.config.idle_unload_minutes,
                "port": self.config.server_port,
                "gateway": {
                    "enabled": self.config.gateway_enabled,
                    "local_models": list(self.config.gateway_local_models),
                    "max_context_tokens": self.config.gateway_max_context_tokens,
                    "upstream_url": self.config.gateway_upstream_url,
                },
            },
        }
        if recent:
            document["recent_turns"] = live["recent_turns"]
        return document


def create_server(
    engines: EngineManager,
    config: SousConfig,
    *,
    upstream: Upstream | None = None,
    inflight: Inflight | None = None,
) -> MCPServer:
    # One registry: the gateway's turns write it, and the status routes
    # read it. A caller may pass its own (a test that writes to it while
    # driving the app).
    inflight = inflight or Inflight()
    svc = SousService(engines, config, inflight)
    # mount_gateway (below) runs after MCPServer(...) is constructed, so the
    # lifespan closure below needs a late-bound holder for whatever it
    # mounts — empty when the gateway is disabled, since there is then
    # nothing for close() to mean.
    mounted_gateway: list[Gateway] = []

    @contextlib.asynccontextmanager
    async def _lifespan(_: MCPServer) -> AsyncIterator[None]:
        # Nothing else loops unload_if_idle() any more: the lifespan is the
        # one span that matches the served app's lifetime exactly.
        engines.start_idle_sweep()
        try:
            yield None
        finally:
            engines.stop_idle_sweep()
            # Fires on every path that drives the app's ASGI lifespan:
            # uvicorn's graceful shutdown and its SIGTERM path alike
            # (capture_signals re-raises SIGTERM only after Server.shutdown()
            # awaits this shutdown), plus any embedded lifecycle that drives
            # lifespan. Without it the gateway's session thread stays parked
            # in _requests.get() and never reaches release_mlx_thread_state()
            # (ml-explore/mlx#4327) on a non-signal exit — and the upstream
            # forwarder's connection pool is closed with it.
            for gateway in mounted_gateway:
                await gateway.aclose()

    mcp = MCPServer("sous", lifespan=_lifespan)

    # MCPServer.__init__ just ran the SDK's configure_logging("INFO"), which
    # basicConfig's a RichHandler onto the root logger — every record wrapped
    # at 80 columns. Replace it with the daemon's one-line shape now, before
    # the first line anyone cares about is written.
    configure_daemon_logging()

    # The daemon's own routes go on before the gateway's: the gateway ends
    # with a catch-all that forwards upstream, and a /sous/ path must never
    # get there. Mounted whatever the gateway flag says — `sous claude`
    # asks /sous/status whether the gateway is on.
    mount_monitor(mcp, engines, lambda: svc.status_document(recent=True), svc.status_version)
    if config.gateway_enabled:
        mounted_gateway.append(
            mount_gateway(mcp, engines, config, upstream=upstream, inflight=inflight)
        )

    return mcp


def _login_shell_path() -> str | None:
    """PATH as the user's login shell sees it, or None if that can't be learned.

    Under launchd the daemon inherits the bare system PATH, so allowlisted
    commands (`uv run pytest`, anything user-installed) stop resolving even
    though they work in the user's terminal — each one silently degrades into
    a human approval request. Guessing at install time (shim directories,
    `which uv`, hardcoded Homebrew paths) bakes one machine's setup into a
    plist that rots; the user's own shell is the only authority on where
    their tools live, so ask it once at startup. Markers bracket the answer
    because login shells are entitled to print banners from init files, and
    `-i` is included because plenty of real PATH setup lives in rc files that
    non-interactive shells skip.
    """
    shell = os.environ.get("SHELL") or pwd.getpwuid(os.getuid()).pw_shell
    if not shell:
        return None
    probe = 'printf "%s" "<<sous-path>>$PATH<<sous-path-end>>"'
    try:
        out = subprocess.run(
            [shell, "-l", "-i", "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return None
    found = re.findall(r"<<sous-path>>(.*?)<<sous-path-end>>", out.stdout, re.DOTALL)
    return found[-1] if found and found[-1] else None


def _acquire_singleton_lock(data_dir: Path) -> IO[bytes]:
    """Take the daemon's exclusive lock, or exit if another daemon holds it.

    flock rather than a pidfile: the kernel releases it when the holder dies,
    including SIGKILL and the crashes launchd restarts from, so a stale lock
    can never wedge the daemon out of ever starting again.

    The returned handle must stay open for the process's lifetime — closing it
    (or letting it be collected) drops the lock.
    """
    lock_path = data_dir / "daemon.lock"
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        handle.close()
        if e.errno in (errno.EAGAIN, errno.EACCES):  # EWOULDBLOCK is EAGAIN
            raise SystemExit(f"sous: another daemon already holds {lock_path}") from None
        # ENOLCK, or a filesystem that cannot flock: a real startup failure.
        # Calling it contention would send the operator hunting for a second
        # daemon that does not exist.
        raise
    # Record the holder now that the lock is ours (so this cannot clobber a live
    # daemon's entry). `sous mcp` reads it to tell a restarted daemon from the
    # one it connected to: the port alone cannot, because launchd puts a new
    # daemon on the same port within a second and every old session is dead.
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n".encode())
    handle.flush()
    return handle


def _install_shutdown_handler(stop: threading.Event) -> None:
    """Kill in-flight commands on SIGTERM/SIGINT before the process goes away.

    Default handling terminates the daemon outright — measured: exit -15 with
    main()'s finally never reached — so cleanup in a finally would be dead
    code. Meanwhile a command's child is in its own session (start_new_session
    in toolexec), so it survives the daemon and goes on writing to the user's
    project. launchd restarts and `sous stop` both send SIGTERM, so this is the
    ordinary path out, not an edge case.
    """

    def handle(signum, frame) -> None:  # noqa: ARG001 — signal handler signature
        stop.set()
        killed = terminate_active_commands()
        if killed:
            print(
                format_line("WARNING", "sous.server", f"killed {killed} running command group(s)"),
                file=sys.stderr,
            )
        sys.stderr.flush()
        os._exit(0)

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, handle)


# uvicorn owns SIGTERM/SIGINT while it serves: capture_signals swaps sous's
# handler out and re-raises the signal only after its own shutdown returns,
# and by default that shutdown waits for every open connection with no bound.
# A non-streaming gateway turn (Claude Code's retry shape) can hold one for
# the whole generation timeout, which would defer the command-group kill in
# _install_shutdown_handler by the same amount. Bound it: streams already
# cancel themselves on the exit signal (sse-starlette), and a cancelled
# non-streaming handler leaves its turn draining on the executor thread.
# This is the whole of the shutdown story in the daemon: capture_signals
# re-raises SIGTERM inside the serve() coroutine, so _install_shutdown_handler
# os._exit()s from there — anyio.run's teardown, and any thread pool it would
# join on the way out, is never reached.
GRACEFUL_SHUTDOWN_SECONDS = 5


def uvicorn_config(app, host: str, port: int, log_level: str = "info") -> uvicorn.Config:
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        # uvicorn's access log prints the raw request target for every
        # response — query string included — at INFO, which is the level the
        # daemon runs at. The gateway's catch-all forwards whatever Claude Code
        # puts in one, and nothing of value is lost by silencing it: the
        # gateway writes its own bounded metadata line per forwarded request,
        # and the MCP transport never had a query string worth logging.
        access_log=False,
        # uvicorn's default dictConfig gives `uvicorn` and `uvicorn.error`
        # their own stderr handlers and stops propagation, so its lines would
        # keep the `INFO:     Started server process` shape beside the
        # daemon's timestamped ones. None leaves logging to sous.logs.
        log_config=None,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


def serve(mcp: MCPServer, host: str, port: int) -> None:
    """What `mcp.run(transport="streamable-http")` does, minus the unbounded
    shutdown wait: the SDK builds this same app and uvicorn config but exposes
    no graceful-shutdown timeout."""
    app = mcp.streamable_http_app(host=host)
    anyio.run(uvicorn.Server(uvicorn_config(app, host, port)).serve)


def main() -> None:
    # The line shape from the first line on — load_config() below is about to
    # warn on whatever is wrong with the user's config file, and that must
    # already come out shaped and leveled, not as a raw UserWarning on stderr.
    # Install the handler (create_server re-runs the idempotent installer
    # after MCPServer.__init__ puts the SDK's RichHandler back) and route
    # warnings.warn through logging (pytest owns showwarning in tests) before
    # anything else runs. Neither depends on the config or the singleton lock
    # below, so there is no reason for either to wait.
    configure_daemon_logging()
    enable_warning_capture()
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # Adopt the login shell's PATH, so the daemon behaves the same however
    # it was launched (launchd, `sous mcp`, or a terminal). Fallback on
    # failure: whatever PATH we inherited.
    if login_path := _login_shell_path():
        os.environ["PATH"] = login_path
    # Before anything else claims the port or the data dir: the port bind at
    # the end of this function is far too late to be the guard against a
    # second daemon starting. `_lock` is unused by design — it must stay
    # open to hold it.
    _lock = _acquire_singleton_lock(config.data_dir)
    # Only the daemon entry point silences huggingface_hub's tqdm bars —
    # terminal animation that lands in a log file as garbage (`Fetching 13
    # files: 100%|██████████|`).
    from huggingface_hub.utils import disable_progress_bars

    # launchd's stderr is a pipe with nobody to animate a bar for; a `sous
    # serve` run by hand for a first multi-GB model download has a real
    # terminal, and the download's own progress should still show there.
    # stderr is None when fd 2 was closed at exec (`sous serve 2>&-`).
    if sys.stderr is None or not sys.stderr.isatty():
        disable_progress_bars()
    engines = EngineManager(config)
    stop = threading.Event()
    _install_shutdown_handler(stop)
    mcp = create_server(engines, config)
    if config.gateway_enabled:
        _logger.info(
            f"gateway (experimental) serving {', '.join(config.gateway_local_models)} "
            f"at http://127.0.0.1:{config.server_port}/v1/messages; "
            f"everything else is forwarded to {config.gateway_upstream_url}"
        )
    try:
        serve(mcp, "127.0.0.1", config.server_port)
    finally:
        stop.set()
