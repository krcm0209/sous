"""The daemon's own loopback routes under /sous/: what `sous claude` speaks
to the daemon. Mounted before the gateway's routes and whatever the gateway
flag says, so a path under /sous — the bare /sous included — is answered
here or 404s here, and is never forwarded to the upstream."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from anyio.to_thread import run_sync
from mcp.server import MCPServer
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError
from sous.loopback import check_loopback

# A hold body is two numbers; anything larger is not one.
HOLD_BODY_LIMIT = 1024
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def _refused(e: RequestError) -> Response:
    return JSONResponse(e.body(), status_code=e.status)


def _invalid(message: str) -> RequestError:
    return RequestError(400, "invalid_request_error", message)


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
    if (
        isinstance(create_time, bool)
        or not isinstance(create_time, int | float)
        or not math.isfinite(create_time)
    ):
        raise _invalid("create_time must be a finite number")
    return pid, float(create_time)


def mount_monitor(mcp: MCPServer, engines: EngineManager, status: Callable[[], dict]) -> None:
    """Register GET /sous/status, POST /sous/hold and a 404 for every other
    /sous/ path. `status` is the daemon's server_status document. Both real
    handlers hand their work to a thread: status() reads the task store and
    hold() takes the engine manager's lock, and neither belongs on the event
    loop."""

    async def sous_status(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return JSONResponse(await run_sync(status))

    async def sous_hold(request: Request) -> Response:
        try:
            check_loopback(request)
            pid, create_time = await _hold_body(request)
        except RequestError as e:
            return _refused(e)
        return JSONResponse(await run_sync(engines.hold, pid, create_time))

    async def sous_unknown(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return _refused(RequestError(404, "not_found_error", "no such route"))

    mcp.custom_route("/sous/status", methods=["GET"])(sous_status)
    mcp.custom_route("/sous/hold", methods=["POST"])(sous_hold)
    # Registered after the two real routes and before the gateway mounts its
    # catch-all (the SDK keeps registration order), so no path under /sous
    # can reach the upstream with the gateway on. The bare /sous needs its
    # own entry: `/sous/{path:path}` does not match it, and the gateway's
    # `/{path:path}` would.
    mcp.custom_route("/sous", methods=_METHODS)(sous_unknown)
    mcp.custom_route("/sous/{path:path}", methods=_METHODS)(sous_unknown)
