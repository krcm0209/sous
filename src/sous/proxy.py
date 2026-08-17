"""stdio<->HTTP bridge: `sous mcp`, for clients that only launch subprocesses.

Claude Desktop starts MCP servers as stdio subprocesses, but sous is a
long-lived HTTP daemon — one process, one worker, one resident model. Running
the server itself over stdio would give every client its own worker and its own
copy of the model, so this forwards messages to the daemon instead and keeps no
state of its own. N clients share one daemon.

It is deliberately a message pump, not a semantic proxy: payloads are passed
through untouched, so tools and notifications need no support here.

That costs one thing, knowingly. stdio carries no HTTP headers, so there is no
`Mcp-Method`/`Mcp-Name`/protocol-version metadata to forward and the daemon
treats the session as legacy — the SDK's documented fallback, which a
`mode='auto'` client takes without complaint. The bridge does not synthesize
those headers: the daemon validates them only when present, so omitting them
skips that rung entirely while a wrong guess would be a HEADER_MISMATCH
rejection. Passing through less is the safe direction here.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time

import anyio
from anyio.abc import TaskGroup
from anyio.to_thread import run_sync
from mcp import stdio_server
from mcp.client.streamable_http import streamable_http_client

from sous.config import SousConfig, load_config

DEFAULT_WAIT_SECONDS = 20.0
_POLL_SECONDS = 0.2
_LIVENESS_POLL_SECONDS = 1.0


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_daemon(
    port: int, start_command: list[str], wait_seconds: float = DEFAULT_WAIT_SECONDS
) -> bool:
    """Return True once a daemon answers on `port`, starting one if needed.

    Starting is safe to do from several proxies at once: the daemon takes an
    exclusive flock, so a loser of the race exits instead of running a second
    worker and loading a second copy of the model.

    The child is detached (it must outlive this proxy) and its output is
    discarded — this runs before stdio_server() diverts fd 1, so an inherited
    stdout would put the daemon's startup line into the client's protocol
    stream and corrupt the session.
    """
    if _port_open(port):
        return True
    subprocess.Popen(
        start_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(_POLL_SECONDS)
    return _port_open(port)


async def _pump(source, dest, task_group: TaskGroup) -> None:
    """Forward one direction until it closes, then tear the other one down."""
    async for message in source:
        if isinstance(message, Exception):
            break
        await dest.send(message)
    task_group.cancel_scope.cancel()


async def _watch_daemon(port: int) -> None:
    """End the bridge when the daemon goes away.

    Two things make this necessary, and neither is obvious:

    A dead daemon does not close the client's SSE read stream — it yields
    nothing, raises nothing, and never ends — so the pumps alone wait forever
    while the client keeps talking to a bridge with nothing behind it. The
    daemon is on loopback, so the port is an unambiguous liveness signal.
    _port_open blocks, hence the worker thread.

    Cancelling the task group is then not enough to exit: stdio_server() is
    blocked reading stdin, which the client still holds open, and that read
    stalls the transports' __aexit__ indefinitely. So end the process outright.
    A bridge holds no state and buffers nothing, and dying is precisely the
    signal the client needs — EOF on stdout is how it learns the server is
    gone. Client-initiated shutdown still unwinds normally: stdin reaches EOF,
    the pump ends, and the task group closes on its own.
    """
    while await run_sync(_port_open, port):
        await anyio.sleep(_LIVENESS_POLL_SECONDS)
    print(f"sous: daemon on 127.0.0.1:{port} went away; closing the bridge", file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)


async def _bridge(url: str, port: int) -> None:
    async with (
        stdio_server() as (client_read, client_write),
        streamable_http_client(url) as (daemon_read, daemon_write),
        anyio.create_task_group() as tg,
    ):
        tg.start_soon(_pump, client_read, daemon_write, tg)
        tg.start_soon(_pump, daemon_read, client_write, tg)
        tg.start_soon(_watch_daemon, port)


def run(config: SousConfig | None = None, start_command: list[str] | None = None) -> int:
    config = config or load_config()
    command = start_command or [shutil.which("sous") or sys.argv[0], "serve"]
    if not ensure_daemon(config.server_port, command):
        # stdout belongs to the protocol; everything the operator reads is stderr.
        print(f"sous: no daemon on 127.0.0.1:{config.server_port}", file=sys.stderr)
        print("start it with: sous serve   (or: sous install-launchd)", file=sys.stderr)
        return 1
    anyio.run(_bridge, f"http://127.0.0.1:{config.server_port}/mcp", config.server_port)
    return 0
