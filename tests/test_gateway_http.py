"""Gateway behaviour only a real server shows: keepalive pings while the
model is silent, and a client that hangs up mid-stream. httpx's in-process
transport buffers whole responses, so these run uvicorn on a loopback port in
a thread — same stack as the daemon, no subprocess."""

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

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.routes import Gateway, mount_gateway
from sous.server import GRACEFUL_SHUTDOWN_SECONDS, create_server, uvicorn_config
from sous.tasks import TaskStore
from tests.fake_engine import ChunkedFakeEngine

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _app(tmp_path: Path, engine):
    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    return create_server(TaskStore(tmp_path / "tasks.db"), engines, cfg).streamable_http_app()


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
        assert client.head(f"{base}/api/hello").status_code == 200


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
