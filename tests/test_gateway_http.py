"""Gateway behaviour only a real server shows: keepalive pings while the
model is silent, and a client that hangs up mid-stream. httpx's in-process
transport buffers whole responses, so these run uvicorn on a loopback port in
a thread — same stack as the daemon, no subprocess."""

import asyncio
import concurrent.futures
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from mcp.server import MCPServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.routes import Gateway, mount_gateway
from sous.gateway.upstream import Upstream
from sous.server import GRACEFUL_SHUTDOWN_SECONDS, create_server, uvicorn_config
from sous.tasks import TaskStore
from tests.fake_engine import ChunkedFakeEngine
from tests.fake_upstream import FakeUpstream

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _app(
    tmp_path: Path,
    engine,
    upstream_url: str | None = None,
    upstream: Upstream | None = None,
):
    """`upstream_url` points [gateway].upstream_url at a fake served by _serve
    (a loopback http origin the config accepts); tests that forward nothing
    get an in-process fake so no Gateway in this file can ever reach the
    network."""
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        gateway_enabled=True,
        gateway_upstream_url=upstream_url or "https://api.anthropic.com",
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    if upstream_url is None and upstream is None:
        upstream = FakeUpstream().upstream()
    return create_server(
        TaskStore(tmp_path / "tasks.db"), engines, cfg, upstream=upstream
    ).streamable_http_app()


def _gateway_app(tmp_path: Path, engine) -> tuple[Gateway, object]:
    """Like _app, but hands back the Gateway too — create_server drops
    mount_gateway's return value, and reaching gateway._turns needs it."""
    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    mcp = MCPServer("test")
    gateway = mount_gateway(mcp, engines, cfg)
    return gateway, mcp.streamable_http_app()


@contextlib.contextmanager
def _serve(app) -> Iterator[tuple[str, uvicorn.Server, threading.Thread]]:
    """Run the ASGI app under uvicorn in a thread, with the daemon's own
    uvicorn configuration so the graceful-shutdown bound under test is the
    real one. Yields the base URL, the server (off the main thread its
    should_exit flag is the only stop switch) and the serving thread."""
    port = _free_port()
    server = uvicorn.Server(uvicorn_config(app, "127.0.0.1", port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn never started"
    try:
        yield f"http://127.0.0.1:{port}", server, thread
    finally:
        server.should_exit = True
        thread.join(GRACEFUL_SHUTDOWN_SECONDS + 10)


def _wait_for_generation(inner: ChunkedFakeEngine) -> None:
    deadline = time.monotonic() + 5
    while not inner.generate_threads and time.monotonic() < deadline:
        time.sleep(0.05)
    assert inner.generate_threads, "the turn never started"


def _body(**overrides) -> dict:
    body = {
        "model": "sous-local",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _frames(lines: Iterator[str]) -> Iterator[tuple[str, dict | None]]:
    event = data = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: ") :])
        elif line == "" and event is not None:
            yield event, data
            event = data = None


def _fake_upstream_app(record: dict) -> Starlette:
    """A stand-in for api.anthropic.com over a real socket: /v1/messages
    streams two SSE frames a second apart (or, with record["hold"], frames
    every 0.2 s until the connection drops) and notes when its generator is
    closed; /api/hello answers 200. Records the headers it saw."""
    record["closed"] = threading.Event()

    async def messages(request: Request) -> Response:
        record["headers"] = dict(request.headers)
        record["body"] = await request.body()

        async def frames():
            try:
                yield b"event: message_start\ndata: {}\n\n"
                if record.get("hold"):
                    while True:
                        await asyncio.sleep(0.2)
                        yield b"event: ping\ndata: {}\n\n"
                await asyncio.sleep(1.0)
                yield b"event: message_stop\ndata: {}\n\n"
            finally:
                record["closed"].set()

        return StreamingResponse(frames(), media_type="text/event-stream")

    async def hello(request: Request) -> Response:
        return Response(status_code=200)

    return Starlette(
        routes=[
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/api/hello", hello, methods=["GET", "HEAD"]),
        ]
    )


def _upstream_body() -> dict:
    return {"model": "claude-opus-5", "max_tokens": 64, "stream": True, "messages": []}


def test_pings_keep_flowing_while_the_model_is_silent(tmp_path: Path, monkeypatch):
    """Checklist item 2: Claude Code disconnects on a silent stream. The first
    ping is the first byte; sse-starlette repeats it on the interval while the
    fake engine sleeps between its two pieces."""
    import sous.gateway.routes as routes

    monkeypatch.setattr(routes, "PING_INTERVAL_SECONDS", 1)
    inner = ChunkedFakeEngine(["slow|reply"], delay=1.3)
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
        client.stream("POST", f"{base}/v1/messages", json=_body()) as r,
    ):
        assert r.status_code == 200
        events = list(_frames(r.iter_lines()))
    kinds = [e for e, _ in events]
    assert kinds[0] == "ping"
    assert kinds.count("ping") >= 2, kinds
    # A ping may land between message_stop and the stream closing, so the last
    # non-ping frame is what pins the sequence.
    assert [k for k in kinds if k != "ping"][-1] == "message_stop"
    text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta" and d)
    assert text == "slowreply"


def test_client_disconnect_drains_the_turn_and_never_wedges_the_next(tmp_path: Path):
    """Spec Phase 1 requirement: an undrained producer holding the engine lock
    would block every later generation. The thread finishes the turn after the
    client is gone, and the next request runs on the same session."""
    inner = ChunkedFakeEngine(["a|b|c|d|e", "second"], delay=0.3)
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
    ):
        with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
            for line in r.iter_lines():
                if "text_delta" in line:
                    break  # hang up after the first piece
        assert inner.finished.wait(10), "the abandoned turn never completed"
        t0 = time.monotonic()
        second = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert second.status_code == 200
        assert second.json()["content"] == [{"type": "text", "text": "second"}]
        assert time.monotonic() - t0 < 5
    assert inner.generate_threads[0] is inner.generate_threads[1]


def test_non_streaming_over_a_real_socket(tmp_path: Path):
    inner = ChunkedFakeEngine(["hello| world"])
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
    ):
        r = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert r.status_code == 200
        assert r.json()["content"] == [{"type": "text", "text": "hello world"}]
        assert r.json()["usage"]["output_tokens"] == 2


def test_a_request_abandoned_while_queued_never_generates(tmp_path: Path):
    """Drain-to-completion covers a generation that started. One still waiting
    for the lock when its client leaves must not start: it would only delay the
    live requests queued behind it by a whole turn."""
    inner = ChunkedFakeEngine(["a|b|c|d|e|f", "third"], delay=0.4)
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
    ):

        def first_turn() -> None:
            with contextlib.suppress(Exception):
                client.post(f"{base}/v1/messages", json=_body(stream=False))

        first = threading.Thread(target=first_turn, daemon=True)
        first.start()
        _wait_for_generation(inner)  # the first turn holds the gateway lock
        with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
            assert r.status_code == 200  # headers and the first ping arrive while queued
        # Leaving the block closed the second request while it was still queued.
        first.join(10)
        time.sleep(1.0)  # every chance for an abandoned turn to (wrongly) start
        assert len(inner.calls) == 1
        third = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert third.json()["content"] == [{"type": "text", "text": "third"}]


def test_shutdown_is_bounded_while_a_non_streaming_turn_runs(tmp_path: Path):
    """uvicorn owns SIGTERM while it serves and, unbounded, waits for every open
    connection before sous's own handler runs; a non-streaming gateway turn
    (Claude Code's retry shape) can hold one for the whole generation timeout.
    What bounds that wait is the daemon's `timeout_graceful_shutdown`, and this
    test measures exactly that bound. It can only see it because turns run on
    the gateway's private pool: this harness pokes `should_exit` on a server
    off the main thread, where `capture_signals` is a no-op, so `Server.run`'s
    `asyncio.run` teardown runs and joins the DEFAULT executor for up to 300s —
    a turn draining there would hold the serving thread for the whole
    generation and the assertion below would be timing the drain instead. The
    generation runs far longer than the bound to tell the cases apart."""
    inner = ChunkedFakeEngine(["|".join(["slow"] * 12)], delay=1.0)  # ~12s, vs a 5s bound
    outcome: list[object] = []
    client = httpx.Client(timeout=60)
    with _serve(_app(tmp_path, inner)) as (base, server, thread):

        def request() -> None:
            try:
                outcome.append(
                    client.post(f"{base}/v1/messages", json=_body(stream=False)).status_code
                )
            except Exception as e:  # noqa: BLE001 — any failure is a non-200 outcome
                outcome.append(type(e).__name__)

        threading.Thread(target=request, daemon=True).start()
        _wait_for_generation(inner)
        t0 = time.monotonic()
        server.should_exit = True
        thread.join(GRACEFUL_SHUTDOWN_SECONDS + 4)
        elapsed = time.monotonic() - t0
        assert not thread.is_alive(), (
            f"uvicorn waited {elapsed:.1f}s for the turn instead of bounding the shutdown"
        )
    assert inner.finished.wait(20)  # the turn still drained to completion, off the loop
    # The deadline cancels the handler mid-await, which uvicorn reports as a 500;
    # what matters is that the client is not left holding a completed turn.
    assert outcome and outcome[0] != 200, outcome
    client.close()


def test_count_tokens_is_not_blocked_by_a_saturated_turn_pool(tmp_path: Path):
    """count_tokens never takes TurnRunner._lock, so it must not queue behind
    a turn parked in the turns pool waiting on that lock — that separation is
    exactly what count_tokens's own dedicated pool buys. Shrinking _turns to
    one worker and filling it with a slow generation makes the failure mode
    reproducible: put count_tokens back on _turns and this times out instead
    of returning quickly."""
    inner = ChunkedFakeEngine(["a slow reply"], delay=3.0)  # one piece, ~3s generation
    gateway, app = _gateway_app(tmp_path, inner)
    gateway._turns = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    count_body = {"model": "sous-local", "messages": [{"role": "user", "content": "hi"}]}
    with (
        _serve(app) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
    ):
        with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
            assert r.status_code == 200
            _wait_for_generation(inner)  # the pool's one worker is now busy generating
            t0 = time.monotonic()
            count = client.post(f"{base}/v1/messages/count_tokens", json=count_body)
            elapsed = time.monotonic() - t0
        assert count.status_code == 200, count.text
        assert elapsed < 2.0, f"count_tokens waited {elapsed:.1f}s behind the saturated turn pool"
    assert inner.finished.wait(10)


def test_app_shutdown_closes_the_gateway_session_thread(tmp_path: Path, monkeypatch):
    """End-to-end proof the SDK's lifespan hook actually fires: create_server
    wires a lifespan that closes the mounted Gateway on the app's ASGI
    shutdown. Without it the gateway's session thread stays parked in
    _requests.get() forever and never reaches release_mlx_thread_state()
    (ml-explore/mlx#4327) on a non-signal exit — this drives the real
    lifespan (via streamable_http_app(), same as create_server's caller)
    rather than calling Gateway.close() directly."""
    import sous.server as server_mod

    captured: list[Gateway] = []
    real_mount_gateway = server_mod.mount_gateway

    def spy(mcp, engines, cfg, **kw):
        gateway = real_mount_gateway(mcp, engines, cfg, **kw)
        captured.append(gateway)
        return gateway

    monkeypatch.setattr(server_mod, "mount_gateway", spy)
    inner = ChunkedFakeEngine(["ok"])
    with _serve(_app(tmp_path, inner)) as (base, server, thread):
        r = httpx.post(f"{base}/v1/messages", json=_body(stream=False), timeout=30)
        assert r.status_code == 200
        [gateway] = captured
        session = gateway._runner._session
        assert session is not None and session._thread.is_alive()
        server.should_exit = True
    # _serve's own finally already set should_exit and joined the uvicorn
    # thread with a generous bound; by the time that join returns, uvicorn's
    # Server.shutdown() has already awaited the app's ASGI lifespan shutdown
    # (Server.shutdown -> self.lifespan.shutdown()) — so create_server's
    # lifespan, and therefore Gateway.close(), has already run.
    assert not thread.is_alive()
    assert not session._thread.is_alive()


def test_a_full_queue_answers_529_immediately_and_releases_on_completion(
    tmp_path: Path, monkeypatch
):
    """MAX_PENDING_TURNS bounds admission to the turn pool: a burst beyond it
    gets a real 529 before any bytes, not an untimed wait behind Gateway._turns'
    executor queue — and the slot it holds is released only when the turn
    that holds it actually finishes, not when a later request merely asks."""
    import sous.gateway.routes as routes

    monkeypatch.setattr(routes, "MAX_PENDING_TURNS", 1)
    inner = ChunkedFakeEngine(["a|b|c", "third"], delay=1.0)  # first turn ~3s, second scripted
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
    ):

        def first_turn() -> None:
            with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
                list(r.iter_lines())  # drain to completion; never disconnect early

        first = threading.Thread(target=first_turn, daemon=True)
        first.start()
        _wait_for_generation(inner)  # the first turn now holds the one pending slot

        t0 = time.monotonic()
        second = client.post(f"{base}/v1/messages", json=_body(stream=False))
        elapsed = time.monotonic() - t0
        assert second.status_code == 529
        assert second.json()["error"]["type"] == "overloaded_error"
        assert elapsed < 1.0, f"529 took {elapsed:.2f}s instead of returning immediately"

        assert inner.finished.wait(10), "the first turn never completed"
        first.join(10)

        third = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert third.status_code == 200
        assert third.json()["content"] == [{"type": "text", "text": "third"}]

    assert len(inner.calls) == 2  # the 529'd request never reached the engine


# --- forwarding (Phase 2) ---------------------------------------------------------------


def test_a_forwarded_stream_is_relayed_as_it_arrives(tmp_path: Path):
    """The relay must not buffer: the first upstream frame reaches the client
    before the upstream's one-second pause ends. Headers cross intact and the
    Host is the upstream's own."""
    record: dict = {}
    with (
        _serve(_fake_upstream_app(record)) as (upstream_base, _us, _ut),
        _serve(_app(tmp_path, ChunkedFakeEngine([]), upstream_url=upstream_base)) as (base, _s, _t),
        httpx.Client(timeout=30) as client,
    ):
        t0 = time.monotonic()
        with client.stream(
            "POST",
            f"{base}/v1/messages",
            json=_upstream_body(),
            headers={
                "authorization": "Bearer sk-ant-oat01-canary",
                "anthropic-beta": "oauth-2025-04-20",
            },
        ) as r:
            assert r.status_code == 200
            assert r.headers["via"] == "1.1 sous"
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = r.iter_lines()
            first = next(line for line in lines if line.startswith("event: "))
            assert first == "event: message_start"
            assert time.monotonic() - t0 < 0.9, "the first frame was held back"
            rest = list(lines)
        assert "event: message_stop" in rest
        assert client.head(f"{base}/api/hello").status_code == 200
    assert record["headers"]["authorization"] == "Bearer sk-ant-oat01-canary"
    assert record["headers"]["anthropic-beta"] == "oauth-2025-04-20"
    assert record["headers"]["host"] == upstream_base.removeprefix("http://")
    assert json.loads(record["body"]) == _upstream_body()


def test_a_client_that_hangs_up_closes_the_upstream_stream(tmp_path: Path):
    """Without the shielded close in the relay, an abandoned forwarded stream
    keeps the upstream generating (and billing) until its read timeout."""
    record: dict = {"hold": True}
    with (
        _serve(_fake_upstream_app(record)) as (upstream_base, _us, _ut),
        _serve(_app(tmp_path, ChunkedFakeEngine([]), upstream_url=upstream_base)) as (base, _s, _t),
        httpx.Client(timeout=30) as client,
    ):
        with client.stream("POST", f"{base}/v1/messages", json=_upstream_body()) as r:
            for line in r.iter_lines():
                if line == "event: ping":
                    break  # hang up mid-stream
        assert record["closed"].wait(10), "the upstream stream was never closed"


def test_an_unreachable_upstream_is_a_prompt_502_and_the_local_model_still_serves(tmp_path: Path):
    dead = f"http://127.0.0.1:{_free_port()}"
    with (
        _serve(_app(tmp_path, ChunkedFakeEngine(["local"]), upstream_url=dead)) as (base, _s, _t),
        httpx.Client(timeout=30) as client,
    ):
        t0 = time.monotonic()
        r = client.head(f"{base}/api/hello")
        assert r.status_code == 502
        assert r.headers["via"] == "1.1 sous"
        assert time.monotonic() - t0 < 5
        r = client.post(f"{base}/v1/messages", json=_upstream_body())
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "api_error"
        r = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert r.status_code == 200
        assert r.json()["content"] == [{"type": "text", "text": "local"}]


def test_app_shutdown_closes_the_upstream_client(tmp_path: Path):
    """The lifespan hook that closes the gateway (Phase 1) now also closes the
    forwarder's connection pool — through the real ASGI lifespan, not by
    calling aclose() directly."""
    upstream = Upstream("https://api.anthropic.com")
    with _serve(_app(tmp_path, ChunkedFakeEngine(["ok"]), upstream=upstream)) as (base, server, _t):
        assert (
            httpx.post(f"{base}/v1/messages", json=_body(stream=False), timeout=30).status_code
            == 200
        )
        assert not upstream._client.is_closed
        server.should_exit = True
    assert upstream._client.is_closed
