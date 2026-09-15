"""The daemon's own loopback routes under /sous/: reachable with the gateway
off, never forwarded with it on, guarded like the gateway's routes."""

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import httpx
import psutil
import pytest
from starlette.requests import Request

from sous import monitor
from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError
from sous.inflight import Inflight
from sous.monitor import HOLD_BODY_LIMIT
from sous.monitor import _hold_body as _parse_hold_body
from sous.server import create_server
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine
from tests.fake_upstream import FakeUpstream


def _app(
    tmp_path: Path, *, gateway_enabled: bool = False, upstream=None, alive=None, inflight=None
):
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        gateway_enabled=gateway_enabled,
    )
    engines = EngineManager(
        cfg, engine_factory=lambda mid: FakeEngine([]), holder_alive=alive or (lambda pid, ct: True)
    )
    store = TaskStore(tmp_path / "tasks.db")
    upstream = upstream or FakeUpstream().upstream()
    server = create_server(store, engines, cfg, upstream=upstream, inflight=inflight)
    return server.streamable_http_app(), engines


def _request(app, method: str, path: str, body=None, headers=None) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8383"
        ) as client:
            # A dict is JSON-encoded; bytes, str, None and an async iterator
            # (a chunked body) go to httpx as they are.
            content = json.dumps(body) if isinstance(body, dict) else body
            return await client.request(
                method,
                path,
                content=content,
                headers={"content-type": "application/json", **(headers or {})},
            )

    return asyncio.run(go())


def _hold_body(**overrides) -> dict:
    return {"pid": os.getpid(), "create_time": psutil.Process().create_time(), **overrides}


def _await_loaded(app, timeout: float = 5.0) -> dict:
    """The preload runs on its own thread; poll the route the launcher would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        engine = _request(app, "GET", "/sous/status").json()["engine"]
        if engine["loaded"] and not engine["loading"]:
            return engine
        time.sleep(0.01)
    raise AssertionError("the preload never finished")


def test_status_is_the_full_status_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "GET", "/sous/status")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    doc = r.json()
    assert doc["engine"]["loaded"] is False
    assert doc["engine"]["loading"] is False
    assert doc["engine"]["holders"] == 0
    assert "memory_gb" in doc["engine"]
    assert doc["inflight"] == [] and doc["recent_turns"] == [] and doc["recent_tasks"] == []
    assert doc["config"]["gateway"] == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }
    assert set(doc) == {"engine", "inflight", "queue", "config", "recent_turns", "recent_tasks"}


def test_hold_registers_the_caller_and_preloads(tmp_path: Path):
    app, engines = _app(tmp_path)
    r = _request(app, "POST", "/sous/hold", _hold_body())
    assert r.status_code == 200
    assert r.json() == {"loaded": False, "loading": True, "holders": 1}
    engine = _await_loaded(app)
    assert engine["holders"] == 1
    s = engines.status()
    assert s["loaded"] is True and s["loading"] is False and s["holders"] == 1
    r = _request(app, "POST", "/sous/hold", _hold_body(pid=os.getpid() + 1))
    assert r.json() == {"loaded": True, "loading": False, "holders": 2}


def test_hold_refuses_every_other_shape_with_an_anthropic_shaped_400(tmp_path: Path):
    app, engines = _app(tmp_path)
    bad = [
        b"not json",
        b"[]",
        json.dumps({"pid": os.getpid()}),
        json.dumps(_hold_body(extra=1)),
        json.dumps(_hold_body(pid=True)),
        json.dumps(_hold_body(pid=0)),
        json.dumps(_hold_body(pid="12")),
        json.dumps(_hold_body(create_time="now")),
        json.dumps(_hold_body(create_time=1e999)),
        b'{"pid": 1, "create_time": NaN}',
        b'{"pid": 1, "create_time": ' + b"9" * 400 + b"}",
        json.dumps(_hold_body(create_time=1.0)).encode() + b" " * HOLD_BODY_LIMIT,
    ]
    for body in bad:
        r = _request(app, "POST", "/sous/hold", body)
        assert r.status_code == 400, body[:40]
        answer = r.json()
        assert answer["type"] == "error" and answer["error"]["type"] == "invalid_request_error"
        assert isinstance(answer["error"]["message"], str)

    async def chunks():
        # httpx sends an async byte iterator as several ASGI messages, so the
        # cap has to be enforced as the body accumulates, not on one read.
        for _ in range(4):
            yield b"x" * 400

    r = _request(app, "POST", "/sous/hold", chunks())
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    s = engines.status()
    assert s["holders"] == 0 and s["loading"] is False and s["loaded"] is False


def test_a_client_that_drops_mid_body_is_a_400_not_a_traceback():
    """httpx's in-process transport cannot hang up mid-body, so the handler's
    body reader is driven directly with a receive that disconnects."""

    async def receive():
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/sous/hold",
            "headers": [],
            "query_string": b"",
        },
        receive,
    )
    with pytest.raises(RequestError) as exc:
        asyncio.run(_parse_hold_body(request))
    assert exc.value.status == 400 and exc.value.error_type == "invalid_request_error"


def test_routes_are_loopback_guarded(tmp_path: Path):
    app, engines = _app(tmp_path)
    for method, path, body in (("GET", "/sous/status", None), ("POST", "/sous/hold", _hold_body())):
        r = _request(app, method, path, body, headers={"host": "sous.example:8383"})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        r = _request(app, method, path, body, headers={"origin": "https://evil.example"})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        r = _request(app, method, path, body, headers={"sec-fetch-site": "cross-site"})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
    assert engines.status()["holders"] == 0


def test_anything_else_under_sous_is_404_and_never_forwarded(tmp_path: Path):
    fake = FakeUpstream()
    app, _ = _app(tmp_path, gateway_enabled=True, upstream=fake.upstream())
    # `/sous` without the slash would full-match the gateway's catch-all
    # and be forwarded with the client's credentials; a method the real
    # routes do not take falls through to the 404 too.
    for method, path in (
        ("GET", "/sous"),
        ("GET", "/sous/"),
        ("POST", "/sous/events"),
        ("DELETE", "/sous/hold/1"),
        ("POST", "/sous/status"),
        ("GET", "/sous/hold"),
    ):
        r = _request(app, method, path)
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error", path
    assert _request(app, "GET", "/sous/status").status_code == 200
    assert _request(app, "HEAD", "/sous/status").status_code == 200
    assert _request(app, "POST", "/sous/hold", _hold_body()).status_code == 200
    assert fake.requests == []
    # The gateway's own catch-all still forwards what is not ours.
    assert _request(app, "GET", "/api/hello").status_code == 200
    assert [s["path"] for s in fake.requests] == ["/api/hello"]


def test_routes_are_up_with_the_gateway_off(tmp_path: Path):
    app, _ = _app(tmp_path, gateway_enabled=False)
    assert _request(app, "GET", "/sous/status").status_code == 200
    assert _request(app, "POST", "/sous/hold", _hold_body()).status_code == 200
    assert _request(app, "GET", "/sous/nope").status_code == 404
    assert _request(app, "GET", "/sous").status_code == 404
    assert _request(app, "GET", "/api/hello").status_code == 404  # no gateway, no forwarding


def test_a_failing_status_or_hold_is_an_error_body_not_a_bare_500(
    tmp_path: Path, monkeypatch, caplog
):
    """Starlette's own 500 is a line of text the launcher reads as no daemon
    at all; a failure inside the daemon answers in the vocabulary every other
    route speaks, and names its type in the log."""
    import logging

    from sous.server import SousService

    def failing_status(self, *, recent):
        raise RuntimeError("tasks.db is locked")

    def failing_hold(self, pid, create_time):
        raise RuntimeError("no thread")

    monkeypatch.setattr(SousService, "status_document", failing_status)
    monkeypatch.setattr(EngineManager, "hold", failing_hold)
    app, _ = _app(tmp_path)
    with caplog.at_level(logging.ERROR, logger="sous.monitor"):
        r = _request(app, "GET", "/sous/status")
        assert r.status_code == 500
        assert r.json() == {
            "type": "error",
            "error": {"type": "api_error", "message": "RuntimeError"},
        }
        r = _request(app, "POST", "/sous/hold", _hold_body())
        assert r.status_code == 500 and r.json()["error"]["type"] == "api_error"
    messages = [r.getMessage() for r in caplog.records if r.name == "sous.monitor"]
    assert messages == [
        "GET /sous/status failed (RuntimeError)",
        "POST /sous/hold failed (RuntimeError)",
    ]


def test_status_carries_the_recent_ring_the_gateway_writes(tmp_path: Path):
    """create_server builds one registry for the gateway and the status
    routes: a turn the gateway served shows up in /sous/status."""
    from tests.fake_engine import FakeEngine as _Fake

    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: _Fake(["reply"]))
    app = create_server(
        TaskStore(tmp_path / "tasks.db"), engines, cfg, upstream=FakeUpstream().upstream()
    ).streamable_http_app()
    body = {
        "model": "sous-local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _request(app, "POST", "/v1/messages", body).status_code == 200
    doc = _request(app, "GET", "/sous/status").json()
    (turn,) = doc["recent_turns"]
    assert turn["status"] == 200 and turn["model"] == "sous-local"


# --- /sous/events -------------------------------------------------------------------


def _frames(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data))
    return out


def _collect_events(app, *, want: int, seconds: float = 3.0, headers=(), during=None) -> list:
    """GET /sous/events straight through the ASGI app — httpx's transport
    buffers a response to its end, and this one has none — until `want`
    non-ping frames have arrived, then disconnect and wait for the handler
    to finish. `during(loop)` runs once the first frame is in."""

    async def go():
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/sous/events",
            "raw_path": b"/sous/events",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1:8383"), *headers],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8383),
        }
        disconnect = asyncio.Event()
        got = asyncio.Event()
        chunks: list[bytes] = []
        status: list[int] = []

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                status.append(message["status"])
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                frames = _frames(b"".join(chunks).decode())
                if len(frames) == 1 and during is not None:
                    during()
                if len([f for f in frames if f[0] != "ping"]) >= want:
                    got.set()

        task = asyncio.create_task(app(scope, receive, send))
        try:
            if want:
                await asyncio.wait_for(got.wait(), seconds)
            else:
                # A refused request completes on its own; nothing to wait for.
                try:
                    await asyncio.wait_for(task, seconds)
                except TimeoutError:
                    pytest.fail(f"the request was served, not refused: status={status}")
        finally:
            disconnect.set()
            # The disconnect must end the stream: a generator that kept
            # running would be a client's whole session of work for nobody.
            # A timed-out wait above has already cancelled the task; a task
            # that failed on its own still re-raises here.
            if not task.cancelled():
                await asyncio.wait_for(task, 5)
        return status, _frames(b"".join(chunks).decode())

    return asyncio.run(go())


def test_events_start_with_the_full_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    status, frames = _collect_events(app, want=1)
    assert status == [200]
    event, doc = frames[0]
    assert event == "status"
    assert set(doc) == {"engine", "inflight", "queue", "config", "recent_turns", "recent_tasks"}
    assert doc["engine"]["loaded"] is False


def test_events_follow_registry_changes_coalesced(tmp_path: Path, monkeypatch):
    """A hundred changes inside one tick are one event, not a hundred; and
    the heartbeat must not add to the count within the window."""
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 60.0)
    registry = Inflight()
    app, _ = _app(tmp_path, inflight=registry)

    def burst() -> None:
        for n in range(100):
            registry.begin(f"msg_{n}", model="sous-local", stream=True, max_tokens=1)
            registry.end(f"msg_{n}")
        registry.begin("msg_last", model="sous-local", stream=True, max_tokens=1)

    _, frames = _collect_events(app, want=2, during=burst)
    statuses = [d for e, d in frames if e == "status"]
    assert 2 <= len(statuses) <= 3
    assert [t["id"] for t in statuses[-1]["inflight"]] == ["msg_last"]


def test_events_beat_once_a_second_when_nothing_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.2)
    app, _ = _app(tmp_path)
    started = time.monotonic()
    _, frames = _collect_events(app, want=3)
    assert len([e for e, _ in frames if e == "status"]) >= 3
    assert time.monotonic() - started < 2.0


def test_events_ping_on_the_configured_cadence(tmp_path: Path, monkeypatch):
    """The heartbeat keeps status frames flowing so the collector stays
    connected past the first ping; the ping itself is sse-starlette's."""
    monkeypatch.setattr(monitor, "EVENT_PING_SECONDS", 0.2)
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.15)
    app, _ = _app(tmp_path)
    _, frames = _collect_events(app, want=3, seconds=3.0)
    assert ("ping", {"type": "ping"}) in frames


def test_events_are_loopback_guarded_and_get_only(tmp_path: Path):
    app, _ = _app(tmp_path)
    status, frames = _collect_events(
        app, want=0, headers=[(b"origin", b"https://evil.example")], seconds=1.0
    )
    assert status == [403] and frames == []
    r = _request(app, "POST", "/sous/events")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_events_refuse_a_cross_site_fetch_and_serve_a_typed_url(tmp_path: Path):
    """A page's <iframe src> or no-cors GET carries no Origin, so the
    Origin rule cannot see it, but it does carry Sec-Fetch-Site — and one
    such request would buy a perpetual status stream. The URL the user types
    carries `none` and streams."""
    app, _ = _app(tmp_path)
    for site in (b"cross-site", b"same-site"):
        status, frames = _collect_events(
            app, want=0, headers=[(b"sec-fetch-site", site)], seconds=1.0
        )
        assert status == [403] and frames == [], site
    for site in (b"none", b"same-origin"):
        status, frames = _collect_events(app, want=1, headers=[(b"sec-fetch-site", site)])
        assert status == [200] and frames[0][0] == "status", site


def test_events_document_is_built_off_the_event_loop(tmp_path: Path, monkeypatch):
    from sous.server import SousService

    threads: set[str] = set()
    original = SousService.status_document

    def recording(self, *, recent):
        threads.add(threading.current_thread().name)
        return original(self, recent=recent)

    monkeypatch.setattr(SousService, "status_document", recording)
    app, _ = _app(tmp_path)
    _collect_events(app, want=1)
    assert threads and "MainThread" not in threads


def test_a_failing_status_build_ends_the_events_stream_not_a_traceback(
    tmp_path: Path, monkeypatch, caplog
):
    """The same failure discipline as /sous/status: a broken document build
    ends the stream and logs its type, never an uncaught traceback through
    uvicorn and never a message that could name a path."""
    import logging

    from sous.server import SousService

    def failing_status(self, *, recent):
        raise RuntimeError("tasks.db is locked")

    monkeypatch.setattr(SousService, "status_document", failing_status)
    app, _ = _app(tmp_path)
    with caplog.at_level(logging.ERROR, logger="sous.monitor"):
        status, frames = _collect_events(app, want=0)
    assert status == [200]
    assert [f for f in frames if f[0] == "status"] == []
    messages = [r.getMessage() for r in caplog.records if r.name == "sous.monitor"]
    assert messages == ["GET /sous/events failed (RuntimeError)"]
    assert "locked" not in messages[0] and "tasks.db" not in messages[0]


def test_a_document_json_cannot_encode_ends_the_stream_too(tmp_path: Path, monkeypatch, caplog):
    """The encode is under the same guard as the build: a value json cannot
    serialize ends the stream and logs its type, rather than raising through
    uvicorn from inside the generator."""
    import logging

    from sous.server import SousService

    monkeypatch.setattr(
        SousService, "status_document", lambda self, *, recent: {"engine": object()}
    )
    app, _ = _app(tmp_path)
    with caplog.at_level(logging.ERROR, logger="sous.monitor"):
        status, frames = _collect_events(app, want=0)
    assert status == [200]
    assert [f for f in frames if f[0] == "status"] == []
    assert [r.getMessage() for r in caplog.records if r.name == "sous.monitor"] == [
        "GET /sous/events failed (TypeError)"
    ]


def test_a_non_finite_number_ends_the_stream_as_status_refuses_it(
    tmp_path: Path, monkeypatch, caplog
):
    """`/sous/status` answers through Starlette's JSONResponse, which refuses
    NaN; the stream's own encoder must too, or the one document is a 500 on
    one route and a `NaN` token no other parser reads on the other."""
    import logging

    from sous.server import SousService

    monkeypatch.setattr(
        SousService,
        "status_document",
        lambda self, *, recent: {"engine": {"memory_gb": float("nan")}},
    )
    app, _ = _app(tmp_path)
    with caplog.at_level(logging.ERROR, logger="sous.monitor"):
        status, frames = _collect_events(app, want=0)
        assert _request(app, "GET", "/sous/status").status_code == 500
    assert status == [200]
    assert [f for f in frames if f[0] == "status"] == []
    assert [r.getMessage() for r in caplog.records if r.name == "sous.monitor"] == [
        "GET /sous/events failed (ValueError)",
        "GET /sous/status failed (ValueError)",
    ]


def test_mounting_the_monitor_pins_sse_starlette_above_debug(tmp_path: Path, monkeypatch):
    """/sous/events is mounted whether or not the gateway is, and sse-starlette
    logs every frame it sends at DEBUG — the pin must not depend on the
    gateway's own copy of it, which only runs when the gateway is mounted."""
    import logging

    logger = logging.getLogger("sse_starlette")
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    _app(tmp_path, gateway_enabled=False)
    assert logger.level == logging.INFO
