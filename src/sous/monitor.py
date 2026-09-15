"""The daemon's own loopback routes under /sous/: what `sous claude`, `sous
top` and `sous statusline` speak to the daemon. Mounted before the gateway's
routes and whatever the gateway flag says, so a path under /sous — the bare
/sous included — is answered here or 404s here for every method the routes
register (a verb none of them lists gets Starlette's own 405 first), and is
never forwarded to the upstream."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Callable

from anyio.to_thread import run_sync
from mcp.server import MCPServer
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError, _invalid
from sous.loopback import ALL_METHODS, check_loopback
from sous.sse import PING as _PING
from sous.sse import PING_INTERVAL_SECONDS as EVENT_PING_SECONDS
from sous.sse import SEP as _SEP

_logger = logging.getLogger("sous.monitor")

# A hold body is two numbers; anything larger is not one.
HOLD_BODY_LIMIT = 1024
# The event stream polls the composite version the service hands it — the
# in-flight registry, the engine manager and the task store — this often: a
# burst of changes inside one tick is one event, and a change is on the wire
# within a tick. Ten a second is what a terminal can show and what a
# token-per-delta feed produces at the default model's decode speed.
EVENT_TICK_SECONDS = 0.1
# While a turn is in flight, a delegated task is running or a load is
# under way the document also goes out this often unchanged: the terminal
# reads a silent stream during a turn as a stalled daemon, a prefill runs
# for a minute without a registry change, and a task's clock and the cache
# counters it moves are in the document while the worker writes the store
# once per tool call. Idle there is no heartbeat at all — a load, a hold
# and a task change bump a version of their own, and the idle clock is the
# terminal's to run.
EVENT_HEARTBEAT_SECONDS = 1.0
# Idle — no turn in flight, no load under way — nothing on the registry
# moves without a turn starting, so the poll slows to this: a new order is
# on screen within half a second, and the loop costs a fifth of the 10 Hz
# one. Busy, the fast tick is what puts a token delta on the wire in time.
EVENT_IDLE_TICK_SECONDS = 0.5


def _busy(document: dict) -> bool:
    # A delegated task counts: the worker never writes the registry, and
    # `running` includes a task awaiting approval, whose clock runs too.
    engine = document.get("engine") or {}
    queue = document.get("queue") or {}
    return bool(document.get("inflight") or queue.get("running") or engine.get("loading"))


async def _status_events(
    status: Callable[[], dict], version: Callable[[], object]
) -> AsyncIterator[ServerSentEvent]:
    """The document now, then again whenever `version()` moved since the
    last one went out — checked every tick — and, only while the last
    document showed a turn, a running task or a load, at least once a
    heartbeat regardless.
    Built on a worker thread each time. Nothing is buffered for a client
    that is gone: the response cancels this generator on disconnect, and
    the tick's sleep is where that lands. A failure building or encoding
    the document ends the stream rather than raising through uvicorn: the
    client sees the connection close and redials, the same as any other
    daemon failure logs its type and never its message."""
    seen: object = None  # nothing sent yet, whatever the clock says
    sent = time.monotonic()
    busy = True  # until the first document says otherwise
    while True:
        # Read before the build, not after: a change landing during it is
        # then a newer value on the next check, never one masked.
        current = version()
        now = time.monotonic()
        if current != seen or (busy and now - sent >= EVENT_HEARTBEAT_SECONDS):
            try:
                document = await run_sync(status)
                # Encoded under the same guard, and as strictly as /sous/status
                # (Starlette's JSONResponse refuses non-finite numbers): a value
                # json cannot serialize would otherwise leave as a traceback
                # through uvicorn, and a NaN as a token no other parser reads.
                data = json.dumps(document, separators=(",", ":"), allow_nan=False)
            except Exception as e:  # noqa: BLE001 — logged and the stream ends, never raised
                _logger.error(f"GET /sous/events failed ({type(e).__name__})")
                return
            yield ServerSentEvent(event="status", data=data, sep=_SEP)
            seen, sent = current, now
            busy = _busy(document)
        await asyncio.sleep(EVENT_TICK_SECONDS if busy else EVENT_IDLE_TICK_SECONDS)


def _refused(e: RequestError) -> Response:
    return JSONResponse(e.body(), status_code=e.status)


async def _hold_body(request: Request) -> tuple[int, float]:
    """`{"pid": int, "create_time": number}` and nothing else. Read in chunks
    under the cap rather than whole: the route is loopback-only, but a body
    that is not a hold is refused before it is held in memory."""
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > HOLD_BODY_LIMIT:
                raise _invalid(f"hold body exceeds {HOLD_BODY_LIMIT} bytes")
            chunks.append(chunk)
    except ClientDisconnect:
        # Nobody will read this response; it exists so a disconnect is not a
        # 500 with a traceback in the daemon log.
        raise _invalid("client disconnected mid-body") from None
    try:
        body = json.loads(b"".join(chunks))
    except ValueError:
        raise _invalid("hold body is not JSON") from None
    if not isinstance(body, dict) or set(body) != {"pid", "create_time"}:
        raise _invalid("hold body must be exactly {pid, create_time}")
    pid, create_time = body["pid"], body["create_time"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise _invalid("pid must be a positive integer")
    if isinstance(create_time, bool) or not isinstance(create_time, int | float):
        raise _invalid("create_time must be a finite number")
    try:
        if not math.isfinite(create_time):
            raise _invalid("create_time must be a finite number")
    except OverflowError:
        # math.isfinite converts its argument to a C double; an int too
        # large for one raises OverflowError rather than answering False.
        raise _invalid("create_time must be a finite number") from None
    # The same conversion isfinite just survived, so this one cannot raise.
    return pid, float(create_time)


async def _served(route: str, fn: Callable[..., dict], *args: object) -> Response:
    """`fn` on a worker thread, its dict as the reply. A failure is a 500 in
    the error vocabulary every other route speaks, not Starlette's bare
    text one — which `sous claude` reads as no daemon at all."""
    try:
        return JSONResponse(await run_sync(fn, *args))
    except Exception as e:  # noqa: BLE001 — every failure becomes an error body
        # The type only: an error's message can name a path.
        _logger.error(f"{route} failed ({type(e).__name__})")
        return _refused(RequestError(500, "api_error", type(e).__name__))


def mount_monitor(
    mcp: MCPServer,
    engines: EngineManager,
    status: Callable[[], dict],
    version: Callable[[], object],
) -> None:
    """Register GET /sous/status, GET /sous/events, POST /sous/hold and a
    404 for every other /sous/ path. `status` builds the full status
    document (recent turns and tasks included); `version` is what the event
    stream polls — a value that differs from the last one whenever the
    document would. The status and hold handlers hand their work to a
    thread — status() reads the task store and hold() takes the engine
    manager's lock, and neither belongs on the event loop — and the event
    stream builds each of its documents the same way."""
    # sse-starlette logs every frame it sends at DEBUG — the status document,
    # verbatim, up to ten times a second — and the gateway's own pin of this
    # logger only runs when the gateway is mounted. /sous/events is mounted
    # unconditionally, so the no-bodies-in-logs rule cannot depend on that.
    logging.getLogger("sse_starlette").setLevel(logging.INFO)

    async def sous_status(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return await _served("GET /sous/status", status)

    async def sous_events(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return EventSourceResponse(
            _status_events(status, version),
            ping=EVENT_PING_SECONDS,
            ping_message_factory=lambda: _PING,
            sep=_SEP,
        )

    async def sous_hold(request: Request) -> Response:
        try:
            check_loopback(request)
            pid, create_time = await _hold_body(request)
        except RequestError as e:
            return _refused(e)
        return await _served("POST /sous/hold", engines.hold, pid, create_time)

    async def sous_unknown(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return _refused(RequestError(404, "not_found_error", "no such route"))

    mcp.custom_route("/sous/status", methods=["GET"])(sous_status)
    mcp.custom_route("/sous/events", methods=["GET"])(sous_events)
    mcp.custom_route("/sous/hold", methods=["POST"])(sous_hold)
    # Registered after the real routes and before the gateway mounts its
    # catch-all (the SDK keeps registration order), so no path under /sous
    # can reach the upstream with the gateway on. The bare /sous needs its
    # own entry: `/sous/{path:path}` does not match it, and the gateway's
    # `/{path:path}` would.
    mcp.custom_route("/sous", methods=list(ALL_METHODS))(sous_unknown)
    mcp.custom_route("/sous/{path:path}", methods=list(ALL_METHODS))(sous_unknown)
