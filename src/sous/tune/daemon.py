"""Whether a tune may start: the daemon must hold no model, no session and
no work, and it is asked to release the weights rather than stopped."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str


def _body(raw: bytes) -> dict:
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def ready_for_tune(
    port: int,
    *,
    request: Callable[..., tuple[int, bytes]] | None = None,
    port_open: Callable[[int], bool] | None = None,
) -> Readiness:
    from sous.cli import _port_open, _sous_request

    call = request or _sous_request
    if not (port_open or _port_open)(port):
        return Readiness(True, f"no daemon on port {port}")
    status, raw = call(port, "GET", "/sous/status")
    if status != 200:
        return Readiness(False, f"port {port} answered {status} to /sous/status; is it sous?")
    doc = _body(raw)
    engine = doc.get("engine") or {}
    if engine.get("holders"):
        return Readiness(False, "a sous claude session holds the model; finish it first")
    running = (doc.get("queue") or {}).get("running") or 0
    inflight = len(doc.get("inflight") or [])
    if running or inflight:
        return Readiness(
            False, f"the daemon is busy: {running} task(s) running, {inflight} turn(s) in flight"
        )
    if not (engine.get("loaded") or engine.get("loading")):
        return Readiness(True, "the daemon holds no model")
    status, raw = call(port, "POST", "/sous/unload")
    if status == 200:
        return Readiness(True, "the daemon released the model")
    if status == 409:
        message = ((_body(raw).get("error") or {}).get("message")) or "the daemon refused"
        return Readiness(False, str(message))
    return Readiness(False, f"unload answered {status}")
