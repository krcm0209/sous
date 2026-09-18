"""Whether a tune may start: the daemon must hold no model, no session and
no work, and it is asked to release the weights rather than stopped."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str


def _error_message(body: dict | None) -> str | None:
    error = (body or {}).get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    return str(message) if message else None


def ready_for_tune(
    port: int,
    *,
    request: Callable[..., tuple[int, bytes]] | None = None,
    port_open: Callable[[int], bool] | None = None,
) -> Readiness:
    """One reading of the daemon before the run: a point in time, not a
    reservation — nothing stops a task or a session arriving afterwards, so
    `main` asks again before the first bench load and before every suite
    arm."""
    import httpx

    from sous.cli import _json_object, _port_open, _sous_request, restart_hint

    call = request or _sous_request
    if not (port_open or _port_open)(port):
        return Readiness(True, f"no daemon on port {port}")
    try:
        status, raw = call(port, "GET", "/sous/status")
    except httpx.HTTPError as exc:
        # The port answered a connect and then nothing usable: the daemon
        # went away in between, or what holds the port is not sous.
        return Readiness(
            False, f"port {port} did not answer /sous/status ({type(exc).__name__}); is it sous?"
        )
    if status != 200:
        return Readiness(False, f"port {port} answered {status} to /sous/status; is it sous?")
    doc = _json_object(raw) or {}
    engine = doc.get("engine") or {}
    if engine.get("holders"):
        return Readiness(False, "a sous claude session holds the model; finish it first")
    queue = doc.get("queue") or {}
    running = queue.get("running") or 0
    # A queued task is claimed on the worker's next poll and loads the model
    # at once: as much a use of the machine as a running one.
    queued = queue.get("queued") or 0
    inflight = len(doc.get("inflight") or [])
    if running or queued or inflight:
        return Readiness(
            False,
            f"the daemon is busy: {running} task(s) running, {queued} queued, "
            f"{inflight} turn(s) in flight",
        )
    if engine.get("loading"):
        # unload_now() would refuse this itself; asking first saves the POST
        # and names the wait.
        return Readiness(
            False,
            "the daemon is loading a model; wait for the load to finish "
            "(or stop the daemon) and retry",
        )
    if engine.get("unloading"):
        # `loaded` is already false while the weights come off the GPU; a
        # tune that started now would load beside them.
        return Readiness(False, "the daemon is unloading its model; retry in a moment")
    if not engine.get("loaded"):
        return Readiness(True, "the daemon holds no model")
    try:
        status, raw = call(port, "POST", "/sous/unload")
    except httpx.HTTPError as exc:
        return Readiness(False, f"the daemon did not answer /sous/unload ({type(exc).__name__})")
    if status == 200:
        return Readiness(True, "the daemon released the model")
    if status == 404:
        return Readiness(
            False,
            f"the daemon predates /sous/unload; restart it ({restart_hint(managed=False)}; "
            f"{restart_hint(managed=True)} when managed)",
        )
    if status == 409:
        message = _error_message(_json_object(raw)) or "the daemon refused"
        if message == "nothing loaded":
            # The idle sweep freed the model between the two calls: the
            # state the tune was asking for.
            return Readiness(True, "the daemon holds no model")
        return Readiness(False, message)
    return Readiness(False, f"unload answered {status}")
