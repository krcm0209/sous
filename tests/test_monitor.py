"""The daemon's own loopback routes under /sous/: reachable with the gateway
off, never forwarded with it on, guarded like the gateway's routes."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import psutil
import pytest
from starlette.requests import Request

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError
from sous.monitor import HOLD_BODY_LIMIT
from sous.monitor import _hold_body as _parse_hold_body
from sous.server import create_server
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine
from tests.fake_upstream import FakeUpstream


def _app(tmp_path: Path, *, gateway_enabled: bool = False, upstream=None, alive=None):
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
    return create_server(store, engines, cfg, upstream=upstream).streamable_http_app(), engines


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
        model = _request(app, "GET", "/sous/status").json()["model"]
        if model["loaded"] and not model["loading"]:
            return model
        time.sleep(0.01)
    raise AssertionError("the preload never finished")


def test_status_is_the_server_status_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "GET", "/sous/status")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    doc = r.json()
    assert doc["model"]["loaded"] is False
    assert doc["model"]["loading"] is False
    assert doc["model"]["holders"] == 0
    assert doc["config"]["gateway"] == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }
    assert set(doc) == {"model", "memory_gb", "queue", "config"}


def test_hold_registers_the_caller_and_preloads(tmp_path: Path):
    app, engines = _app(tmp_path)
    r = _request(app, "POST", "/sous/hold", _hold_body())
    assert r.status_code == 200
    assert r.json() == {"loaded": False, "loading": True, "holders": 1}
    model = _await_loaded(app)
    assert model["holders"] == 1
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
