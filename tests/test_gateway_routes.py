"""The gateway's HTTP surface, in-process through the ASGI app create_server
builds. httpx's ASGITransport buffers a response, so streaming tests read the
whole SSE body after the fact; timing-sensitive behaviour lives in
test_gateway_http.py against a real server."""

import asyncio
import concurrent.futures
import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest
from mcp.server import MCPServer
from sse_starlette import EventSourceResponse
from starlette.requests import Request

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.routes import MAX_BODY_DEPTH, Gateway, mount_gateway
from sous.server import create_server
from sous.tasks import TaskStore
from tests.fake_engine import ChunkedFakeEngine, FakeEngine

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}
XML_CALL = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _app(tmp_path: Path, engine, **overrides):
    overrides.setdefault("gateway_enabled", True)
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        **overrides,
    )
    store = TaskStore(tmp_path / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    return create_server(store, engines, cfg).streamable_http_app()


def _gateway_app(tmp_path: Path, engine, **overrides) -> tuple[Gateway, object]:
    """Like _app, but hands back the Gateway too — create_server drops
    mount_gateway's return value, and Gateway.close() tests need it."""
    overrides.setdefault("gateway_enabled", True)
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        **overrides,
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    mcp = MCPServer("test")
    gateway = mount_gateway(mcp, engines, cfg)
    return gateway, mcp.streamable_http_app()


def _request(app, method: str, path: str, body=None, headers=None) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8383"
        ) as client:
            content = body if isinstance(body, bytes | str | type(None)) else json.dumps(body)
            return await client.request(
                method,
                path,
                content=content,
                headers={"content-type": "application/json", **(headers or {})},
            )

    return asyncio.run(go())


def _post(app, body, path="/v1/messages", headers=None) -> httpx.Response:
    return _request(app, "POST", path, body, headers)


def _body(**overrides) -> dict:
    body = {
        "model": "sous-local",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event: str | None = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data))
    return out


# --- probes and routing -------------------------------------------------------------


def test_hello_answers_head_and_get(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]))
    assert _request(app, "HEAD", "/api/hello").status_code == 200
    assert _request(app, "GET", "/api/hello").status_code == 200


def test_routes_are_absent_when_the_gateway_is_disabled(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]), gateway_enabled=False)
    assert _request(app, "HEAD", "/api/hello").status_code == 404
    assert _post(app, _body()).status_code == 404


def test_unknown_model_is_an_anthropic_shaped_404(tmp_path: Path):
    """Phase 2 forwards these upstream; until then the client hears exactly
    what the real API says for an unknown model, and gate 1 showed the main
    loop recovers from that."""
    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, _body(model="claude-opus-5"))
    assert r.status_code == 404
    assert r.json() == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "model: claude-opus-5"},
    }


def test_configured_local_models_are_all_served(tmp_path: Path):
    app = _app(tmp_path, FakeEngine(["ok"]), gateway_local_models=("sous-local", "sous-fast"))
    assert _post(app, _body(model="sous-fast")).status_code == 200


def test_host_header_must_be_loopback(tmp_path: Path):
    """Custom routes skip the /mcp transport's Host check; a page whose hostname
    re-resolves to 127.0.0.1 must not get to drive the local model."""
    import sous.gateway.routes as routes

    app = _app(tmp_path, FakeEngine([]))
    for host in ("evil.example:8383", "evil.example"):
        r = _post(app, _body(), headers={"host": host})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 403
    for host in ("127.0.0.1:8383", "localhost", "[::1]:8383", "[::1]", "LOCALHOST:8383"):
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 200
    assert set(routes._ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}


def test_a_foreign_origin_is_rejected_before_the_model_is_touched(tmp_path: Path):
    """A cross-origin fetch with Content-Type: text/plain is a CORS simple
    request: no preflight, and a perfectly legitimate loopback Host. The page
    never reads the reply, but the turn it starts would hold the gateway lock,
    the engine lock and the cache slot for a whole generation."""
    inner = FakeEngine(["never", "never"])
    app = _app(tmp_path, inner)
    for body in (_body(), _body(stream=True)):
        r = _post(app, body, headers={"origin": "https://evil.example"})
        assert r.status_code == 403
        assert r.json()["error"]["type"] == "permission_error"
        assert "origins" in r.json()["error"]["message"]
    for origin in ("http://evil.example:8383", "null", "http://127.0.0.1.evil.example"):
        r = _request(app, "HEAD", "/api/hello", headers={"origin": origin})
        assert r.status_code == 403, origin
    assert inner.calls == []  # rejected before any generation, streamed or not


def test_a_malformed_origin_is_rejected_not_raised(tmp_path: Path):
    """urlsplit raises ValueError on an unbalanced IPv6 bracket; that must be
    caught and treated as a foreign origin, not escape as a 500."""
    inner = FakeEngine(["never"])
    app = _app(tmp_path, inner)
    r = _post(app, _body(), headers={"origin": "http://[::1"})
    assert r.status_code == 403
    assert r.json()["error"]["type"] == "permission_error"
    r = _request(app, "HEAD", "/api/hello", headers={"origin": "http://[::1"})
    assert r.status_code == 403  # HEAD carries no body to assert error.type on
    assert inner.calls == []  # rejected before any generation


def test_loopback_and_absent_origins_pass(tmp_path: Path):
    """Absent Origin is the normal case — Claude Code, httpx and curl send
    none — and a page served from the daemon's own loopback origin is the
    gateway's own client."""
    app = _app(tmp_path, FakeEngine([]))
    for origin in (
        "http://127.0.0.1",
        "http://127.0.0.1:8383",
        "http://localhost",
        "http://localhost:8383",
        "http://[::1]",
        "http://[::1]:8383",
        "http://LOCALHOST:8383",
    ):
        r = _request(app, "HEAD", "/api/hello", headers={"origin": origin})
        assert r.status_code == 200, origin
    assert _request(app, "HEAD", "/api/hello").status_code == 200


# --- request validation ---------------------------------------------------------------


def test_invalid_json_and_bad_shapes_are_400s(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, b"{not json")
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = _post(app, _body(max_tokens=0))
    assert r.status_code == 400 and "max_tokens" in r.json()["error"]["message"]
    r = _post(app, [1, 2, 3])
    assert r.status_code == 400


def test_a_body_past_the_depth_cap_is_a_400_not_a_recursion_error(tmp_path: Path):
    # RecursionError is stack-size dependent, not a reliable "too deep"
    # signal (a body that decodes here on this machine's stack would still
    # blow up the chat template's tojson encoder on a smaller one) —
    # MAX_BODY_DEPTH is the real contract _read_json enforces.
    inner = FakeEngine(["unused"])
    n = MAX_BODY_DEPTH + 1
    body = ("[" * n + "]" * n).encode()
    r = _request(_app(tmp_path, inner), "POST", "/v1/messages", body=body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert inner.calls == []


def test_a_tool_schema_nested_about_ten_levels_still_converts(tmp_path: Path):
    # Positive control: a schema far deeper than any real client sends still
    # converts and reaches the engine — the depth cap only bites bodies well
    # past what chat_tools would ever be handed legitimately.
    schema: dict = {"type": "string"}
    for _ in range(10):
        schema = {"type": "object", "properties": {"nested": schema}}
    tools = [{"name": "Nested", "input_schema": schema}]
    r = _post(_app(tmp_path, FakeEngine(["ok"])), _body(tools=tools))
    assert r.status_code == 200


def test_non_json_constants_in_the_body_are_a_400(tmp_path: Path):
    # json.loads accepts NaN/Infinity (and 1e999 → inf) by default; the real
    # API rejects them, and so must the boundary that decides 400-or-turn.
    inner = FakeEngine(["unused"])
    app = _app(tmp_path, inner)
    for raw in (
        b'{"model":"sous-local","max_tokens":10,"messages":[{"role":"user","content":"hi"}],'
        b'"temperature": NaN}',
        b'{"model":"sous-local","max_tokens":10,"messages":[{"role":"user","content":"hi"}],'
        b'"top_p": -Infinity}',
        b'{"model":"sous-local","max_tokens":10,"messages":[{"role":"user","content":"hi"}],'
        b'"temperature": 1e999}',
    ):
        r = _request(app, "POST", "/v1/messages", body=raw)
        assert r.status_code == 400, raw
        assert r.json()["error"]["type"] == "invalid_request_error"
    assert inner.calls == []


def test_malformed_tool_properties_is_a_400_and_never_reaches_the_engine(tmp_path: Path):
    """Pins the shaped-status property end to end: chat_tools rejects a
    non-object `properties` before ToolSet.from_tools (called after the
    RequestError handling in routes.py) can turn it into a bare 500."""
    inner = FakeEngine(["unused"])
    app = _app(tmp_path, inner)
    tools = [{"name": "Broken", "input_schema": {"type": "object", "properties": ["a"]}}]
    r = _post(app, _body(tools=tools))
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert inner.calls == []


def test_unconvertible_content_length_values_never_raise(tmp_path: Path):
    """int() rejects a decimal string past its digit limit, and "²" is a
    digit to str.isdigit but not to int(); either used to escape the header
    check as a bare ValueError before the shaped-error handling."""
    inner = FakeEngine(["ok"])
    gateway, app = _gateway_app(tmp_path, inner)
    r = _post(app, _body(), headers={"content-length": "9" * 5000})
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"

    # "²" cannot go through httpx (non-ASCII header), so hand-build the ASGI
    # request the way Starlette would deliver it (latin-1 header bytes).
    body = json.dumps(_body()).encode()

    async def go():
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [
                (b"host", b"127.0.0.1:8383"),
                (b"content-type", b"application/json"),
                (b"content-length", "²".encode("latin-1")),
            ],
            "query_string": b"",
        }
        return await gateway.messages(Request(scope, receive))

    response = asyncio.run(go())
    assert response.status_code == 200  # the header is ignored, not a 500
    assert len(inner.calls) == 1


def test_oversized_bodies_are_413_by_header_and_by_actual_size(tmp_path: Path, monkeypatch):
    import sous.gateway.routes as routes

    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, _body(), headers={"content-length": str(routes.MAX_REQUEST_BYTES + 1)})
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"
    monkeypatch.setattr(routes, "MAX_REQUEST_BYTES", 64)
    r = _post(app, _body(messages=[{"role": "user", "content": "x" * 200}]))
    assert r.status_code == 413


# --- streaming --------------------------------------------------------------------------


def test_streamed_text_turn_has_the_anthropic_event_sequence(tmp_path: Path):
    inner = FakeEngine(["Hello there"])
    app = _app(tmp_path, inner)
    r = _post(app, _body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert [e for e, _ in events] == [
        "ping",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[1][1]["message"]
    assert start["model"] == "sous-local" and start["usage"]["input_tokens"] == inner.count_tokens(
        [{"role": "user", "content": "hi"}], []
    )
    assert events[3][1]["delta"] == {"type": "text_delta", "text": "Hello there"}
    assert events[5][1] == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    }


def test_streamed_tool_call_turn(tmp_path: Path):
    inner = ChunkedFakeEngine(
        [
            "I will read a.py.\n\n|<tool_call>\n<function=Read>\n|<parameter=file_path>\na.py\n"
            "</parameter>\n</function>\n</tool_call>"
        ]
    )
    app = _app(tmp_path, inner)
    r = _post(app, _body(stream=True, tools=[READ_TOOL]))
    events = _events(r.text)
    types = [e for e, _ in events]
    assert types == [
        "ping",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_start = events[5][1]
    assert tool_start["index"] == 1
    assert tool_start["content_block"]["type"] == "tool_use"
    assert tool_start["content_block"]["name"] == "Read"
    assert tool_start["content_block"]["id"].startswith("toolu_")
    assert json.loads(events[6][1]["delta"]["partial_json"]) == {"file_path": "a.py"}
    assert events[8][1]["delta"]["stop_reason"] == "tool_use"
    # The template got the converted tool, not the Anthropic shape.
    assert inner.tools_seen[0][0] == {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": READ_TOOL["input_schema"],
        },
    }


def test_non_streaming_turn_matches_the_streamed_content(tmp_path: Path):
    script = "Reading.\n\n" + XML_CALL
    streamed = _post(
        _app(tmp_path / "a", ChunkedFakeEngine([script])), _body(stream=True, tools=[READ_TOOL])
    )
    plain = _post(_app(tmp_path / "b", FakeEngine([script])), _body(tools=[READ_TOOL]))
    assert plain.status_code == 200
    message = plain.json()
    assert set(message) == {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "usage",
    }
    assert message["id"].startswith("msg_") and message["stop_reason"] == "tool_use"
    assert message["content"][0] == {"type": "text", "text": "Reading."}
    assert message["content"][1]["name"] == "Read" and message["content"][1]["input"] == {
        "file_path": "a.py"
    }
    assert message["usage"]["output_tokens"] > 0  # FakeEngine counts whitespace-separated words
    streamed_types = [
        b["content_block"]["type"] for e, b in _events(streamed.text) if e == "content_block_start"
    ]
    assert streamed_types == ["text", "tool_use"]


def test_prompt_conversion_reaches_the_engine(tmp_path: Path):
    """Inline system, billing header and the volatile marker: the engine sees
    one stable system message, so the prefix cache can hold across turns."""
    inner = FakeEngine(["ok"])
    app = _app(tmp_path, inner)
    body = _body(
        system=[
            {"type": "text", "text": "x-anthropic-billing-header: cc_is_subagent=true"},
            {"type": "text", "text": "Canonical.\n\n<total_tokens>99 tokens left</total_tokens>"},
        ],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "system", "content": "Inline."},
        ],
        tools=[READ_TOOL, {"type": "web_search_20250305", "name": "web_search"}],
    )
    assert _post(app, body).status_code == 200
    assert inner.calls[0] == [
        {"role": "system", "content": "Canonical.\n\nInline."},
        {"role": "user", "content": "task"},
    ]
    assert [t["function"]["name"] for t in inner.tools_seen[0]] == ["Read"]


def test_max_tokens_is_clamped_to_the_window(tmp_path: Path):
    inner = FakeEngine(["ok", "ok"])
    app = _app(tmp_path, inner)
    _post(app, _body(max_tokens=100_000))  # above the 65536 window, so the clamp bites
    assert inner.max_tokens_seen == [
        65536 - inner.count_tokens([{"role": "user", "content": "hi"}], [])
    ]
    _post(app, _body(max_tokens=10))
    assert inner.max_tokens_seen[-1] == 10


def test_prompt_too_long_is_400_plain_and_an_error_event_streamed(tmp_path: Path):
    inner = FakeEngine(["never", "never"])
    app = _app(tmp_path, inner)
    huge = _body(messages=[{"role": "user", "content": "x" * 300_000}])
    r = _post(app, huge)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "prompt is too long" in r.json()["error"]["message"]
    r = _post(app, {**huge, "stream": True})
    assert r.status_code == 200  # headers are already out; the error is in-band
    events = _events(r.text)
    assert events[0][0] == "ping"
    event, payload = events[-1]
    assert event == "error"
    tokens = inner.count_tokens(huge["messages"], [])
    assert payload == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"prompt is too long: {tokens} tokens > 65536 maximum",
        },
    }
    assert inner.calls == []


def test_engine_failure_is_500_plain_and_an_api_error_event_streamed(tmp_path: Path):
    class Exploding(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            raise ValueError("boom")

    app = _app(tmp_path, Exploding([]))
    r = _post(app, _body())
    assert r.status_code == 500 and r.json()["error"]["type"] == "api_error"
    r = _post(app, _body(stream=True))
    events = _events(r.text)
    assert events[-1][0] == "error" and events[-1][1]["error"]["type"] == "api_error"
    assert "boom" in events[-1][1]["error"]["message"]


def test_malformed_tool_call_comes_back_as_text(tmp_path: Path):
    app = _app(
        tmp_path, FakeEngine(["<tool_call>\n<function=Read>\n<parameter=file_path>\nunterminated"])
    )
    message = _post(app, _body(tools=[READ_TOOL])).json()
    assert message["content"] == [
        {
            "type": "text",
            "text": "<tool_call>\n<function=Read>\n<parameter=file_path>\nunterminated",
        }
    ]
    assert message["stop_reason"] == "end_turn"


def test_count_tokens(tmp_path: Path):
    inner = FakeEngine([])
    app = _app(tmp_path, inner)
    body = {
        "model": "sous-local",
        "messages": [{"role": "user", "content": "hello world"}],
        "tools": [READ_TOOL],
    }
    r = _post(app, body, path="/v1/messages/count_tokens")
    assert r.status_code == 200
    assert r.json() == {
        "input_tokens": inner.count_tokens([{"role": "user", "content": "hello world"}], [])
    }
    assert (
        _post(app, {**body, "model": "claude-x"}, path="/v1/messages/count_tokens").status_code
        == 404
    )


# --- turn admission (MAX_PENDING_TURNS) --------------------------------------------


def test_a_full_queue_answers_529_with_an_anthropic_shaped_body(tmp_path: Path, monkeypatch):
    """Drives the 529 path directly, both streaming and non-streaming:
    pre-acquiring gateway._pending simulates a turn pool already at
    MAX_PENDING_TURNS without needing real concurrency — that timing (an
    immediate 529, then a released slot once the holding turn completes)
    lives in test_gateway_http.py against a real server."""
    import sous.gateway.routes as routes

    monkeypatch.setattr(routes, "MAX_PENDING_TURNS", 1)
    gateway, app = _gateway_app(tmp_path, FakeEngine([]))
    assert gateway._pending.acquire(blocking=False)  # occupy the one slot this config allows
    r = _post(app, _body(stream=False))
    assert r.status_code == 529
    assert r.json() == {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "too many turns queued"},
    }
    r = _post(app, _body(stream=True))
    assert r.status_code == 529
    assert r.json()["error"]["type"] == "overloaded_error"


# --- logging discipline --------------------------------------------------------------------


def test_log_lines_carry_metadata_only(tmp_path: Path, capsys):
    """Spec security posture: bodies and header values never reach a log."""
    secret_text = "SECRET-PROMPT-TEXT-7f3a"
    secret_token = "sk-ant-oat01-SECRET-TOKEN-9c1d"
    app = _app(tmp_path, FakeEngine(["a reply"]))
    r = _post(
        app,
        _body(messages=[{"role": "user", "content": secret_text}], stream=True),
        headers={"authorization": f"Bearer {secret_token}", "anthropic-beta": "oauth-2025-04-20"},
    )
    assert r.status_code == 200
    err = capsys.readouterr().err
    assert "sous gateway: POST /v1/messages" in err
    assert "model=sous-local" in err and "stream=1" in err and "input_tokens=" in err
    assert secret_text not in err
    assert secret_token not in err and "oauth-2025-04-20" not in err
    assert "a reply" not in err


def test_mounting_pins_the_sse_logger_above_debug(tmp_path: Path, monkeypatch):
    """sse-starlette logs every frame it sends at DEBUG — the model's reply,
    verbatim. Mounting the gateway pins that logger, so the no-bodies rule does
    not depend on the daemon happening to run at INFO."""
    logger = logging.getLogger("sse_starlette")
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    _app(tmp_path, FakeEngine([]))
    assert logger.level == logging.INFO


def test_a_dropped_tools_name_never_reaches_the_log_and_its_type_is_bounded(tmp_path: Path, capsys):
    """`name` is free-form client text out of a request body, which the log
    never carries; `type` is one of Anthropic's fixed identifiers and worth
    keeping, but a client can still send any string, so it is truncated."""
    app = _app(tmp_path, FakeEngine(["ok"]))
    name = "n" * 10_000
    kind = "t" * 500
    assert _post(app, _body(tools=[{"type": kind, "name": name}])).status_code == 200
    err = capsys.readouterr().err
    assert "dropped 1 tool" in err
    assert "n" * 20 not in err  # no part of the name is logged
    assert "t" * 60 in err and "t" * 100 not in err  # the type, truncated


def test_a_long_dropped_tool_list_is_capped_in_the_log(tmp_path: Path, capsys):
    app = _app(tmp_path, FakeEngine(["ok"]))
    tools = [{"type": f"tool_{n}_20250101", "name": "x"} for n in range(11)]
    assert _post(app, _body(tools=tools)).status_code == 200
    err = capsys.readouterr().err
    assert "dropped 11 tool(s)" in err
    assert "tool_7_20250101" in err and "tool_8_20250101" not in err
    assert "… (+3 more)" in err


def test_a_dropped_tool_type_cannot_forge_a_log_line(tmp_path: Path, capsys):
    """`type` is a client-controlled string. Logged raw, a newline in one would
    write whatever line the client chose into the daemon log. Short enough here
    to survive the length cap, so the escaping is what has to stop it."""
    app = _app(tmp_path, FakeEngine(["ok"]))
    forged = "x\nsous gateway: POST /v1/messages status=200"
    body = _body(tools=[{"type": forged, "name": "web_search"}])
    assert _post(app, body).status_code == 200
    err = capsys.readouterr().err
    assert "dropped 1 tool(s)" in err
    assert forged not in err  # the newline was escaped, so the forged line never lands
    assert "\\n" in err


def test_a_non_string_tool_type_is_a_400_and_never_reaches_the_log(tmp_path: Path, capsys):
    """A non-string `type` must be rejected before conversion, not str()-ed
    into the dropped-tool-types log line — that would put body content
    (an arbitrary JSON value) into the daemon log."""
    engine = FakeEngine(["ok"])
    app = _app(tmp_path, engine)
    secret = "s3cr3t-value"
    body = _body(
        tools=[{"type": {"token": secret}, "name": "X", "input_schema": {"type": "object"}}]
    )
    r = _post(app, body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    err = capsys.readouterr().err
    assert secret not in err
    assert engine.calls == []


# --- app-shutdown cleanup (session thread + executors) -----------------------------


def test_close_after_a_completed_turn_stops_the_session_and_the_pools(tmp_path: Path):
    gateway, app = _gateway_app(tmp_path, FakeEngine(["ok"]))
    assert _post(app, _body(stream=False)).status_code == 200
    session = gateway._runner._session
    assert session is not None and session._thread.is_alive()
    t0 = time.monotonic()
    gateway.close()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.5  # promptly: nothing here is a running generation to wait out
    assert not session._thread.is_alive()
    assert gateway._turns._shutdown and gateway._counts._shutdown


def test_close_does_not_touch_a_running_turns_session(tmp_path: Path):
    """A turn in flight holds TurnRunner._lock; close(timeout=0.5) must give up
    on it quickly rather than wait out the generation, and must leave the
    session alone rather than close it out from under the running turn."""
    entered = threading.Event()

    class Announcing(ChunkedFakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            entered.set()
            return super().generate(messages, tools, max_tokens, on_delta)

    inner = Announcing(["slow|slow|slow"], delay=0.3)
    gateway, app = _gateway_app(tmp_path, inner)
    turn = threading.Thread(target=lambda: _post(app, _body(stream=False)), daemon=True)
    turn.start()
    assert entered.wait(5)  # the turn holds TurnRunner._lock and is generating
    session_before = gateway._runner._session
    assert session_before is not None
    t0 = time.monotonic()
    gateway.close(timeout=0.5)  # must not raise
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert gateway._runner._session is session_before  # untouched: the lock was busy
    turn.join(10)
    assert inner.finished.wait(5)


def test_a_stream_never_iterated_still_drains_and_releases_its_slot(tmp_path: Path):
    """messages() must submit the turn — and tie the pending slot's release
    to that future — before it ever returns the EventSourceResponse. Calling
    self._stream(...) only creates the generator; if nothing ever iterates it
    (client gone before sse-starlette's first __anext__, or its task group
    cancelled first), a submission living inside that generator's body would
    never run and the slot would leak forever."""
    import sous.gateway.routes as routes

    inner = FakeEngine(["ok"])
    gateway, _app = _gateway_app(tmp_path, inner)
    body = json.dumps(_body(stream=True)).encode()

    async def go():
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(b"host", b"127.0.0.1:8383"), (b"content-type", b"application/json")],
            "query_string": b"",
        }
        request = Request(scope, receive)
        response = await gateway.messages(request)
        assert isinstance(response, EventSourceResponse)
        # Never iterate/call response: this is the never-iterated case.
        deadline = time.monotonic() + 5
        while gateway._pending._value != routes.MAX_PENDING_TURNS and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert gateway._pending._value == routes.MAX_PENDING_TURNS

    asyncio.run(go())
    assert len(inner.calls) == 1  # the turn drained even though nobody read it


def _hand_built_request(body: dict) -> Request:
    """A Request that never goes through ASGITransport, so the caller can
    hold the coroutine it drives in an asyncio.Task and cancel that task
    directly -- httpx's ASGITransport buffers the whole response and gives
    no hook to cancel mid-flight."""
    encoded = json.dumps(body).encode()
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": [(b"host", b"127.0.0.1:8383"), (b"content-type", b"application/json")],
        "query_string": b"",
    }
    return Request(scope, receive)


def test_cancelling_a_non_streaming_request_keeps_its_slot_until_the_turn_ends(tmp_path: Path):
    """A client disconnect cancels the request TASK awaiting messages(), not
    the turn running on its own executor thread -- the pending slot must stay
    held until that turn actually finishes draining, not until the cancelled
    task unwinds. RED at 3d068ea: the slot returned to MAX_PENDING_TURNS right
    after task.cancel(), because the release was tied to the asyncio wrapper
    future (which a cancelled task completes immediately) instead of to the
    concurrent.futures.Future backing the still-running generation."""
    import sous.gateway.routes as routes

    inner = ChunkedFakeEngine(["a|b|c"], delay=0.5)  # ~1.5s to drain, 3 pieces
    gateway, _app = _gateway_app(tmp_path, inner)

    async def go():
        request = _hand_built_request(_body(stream=False))
        task = asyncio.create_task(gateway.messages(request))

        deadline = time.monotonic() + 5
        while not inner.generate_threads and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert inner.generate_threads  # generation has actually started

        assert gateway._pending._value == routes.MAX_PENDING_TURNS - 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Still held: the turn is still draining on the executor thread.
        assert gateway._pending._value == routes.MAX_PENDING_TURNS - 1

        await asyncio.to_thread(inner.finished.wait, 10)
        deadline = time.monotonic() + 5
        while gateway._pending._value != routes.MAX_PENDING_TURNS and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert gateway._pending._value == routes.MAX_PENDING_TURNS
        assert len(inner.calls) == 1  # the turn ran to completion, draining

    asyncio.run(go())


def test_cancelling_a_queued_non_streaming_request_cancels_the_turn(tmp_path: Path):
    """A turn cancelled while still queued in the pool (never started) must
    be cancelled outright rather than left to run for a client that is
    already gone -- asyncio.wrap_future chains the request task's
    cancellation down to the concurrent.futures.Future's own .cancel(), which
    succeeds only while the job has not started. This behaviour already held
    before the slot fix (run_in_executor wraps its future the same way), so
    this test does not discriminate old code from new; it pins the contract
    so the queued case cannot regress while the running case above (the one
    that was broken) is the discriminating test."""
    import sous.gateway.routes as routes

    inner = FakeEngine(["never"])
    gateway, _app = _gateway_app(tmp_path, inner)

    # Saturate the turn pool with a single worker occupied by a blocking
    # dummy job, so the request's own turn sits queued -- not started -- in
    # the pool, which is the only state cf.cancel() can actually cancel.
    gateway._turns.shutdown(wait=False, cancel_futures=True)
    gateway._turns = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    dummy_release = threading.Event()
    gateway._turns.submit(dummy_release.wait, 5)

    async def go():
        request = _hand_built_request(_body(stream=False))
        task = asyncio.create_task(gateway.messages(request))
        await asyncio.sleep(0.05)  # give the loop a tick to submit and queue

        assert gateway._pending._value == routes.MAX_PENDING_TURNS - 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        dummy_release.set()  # let the pool's occupying job finish

        deadline = time.monotonic() + 5
        while gateway._pending._value != routes.MAX_PENDING_TURNS and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert gateway._pending._value == routes.MAX_PENDING_TURNS
        assert inner.calls == []  # the turn was cancelled before it ever ran

    asyncio.run(go())
