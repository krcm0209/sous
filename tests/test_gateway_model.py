"""The gateway route in front of a real (tiny) model: the event stream is
well-formed end to end, counts are real, and a second, longer request reuses
the prompt cache through the gateway-owned session."""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from mcp.server import MCPServer

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.routes import mount_gateway

pytestmark = pytest.mark.model

TINY = "mlx-community/Qwen3-0.6B-4bit"  # ~350 MB; text-only, so the LM backend


def _post(app, body: dict) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8383", timeout=600
        ) as client:
            return await client.post("/v1/messages", json=body)

    return asyncio.run(go())


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event: str | None = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event is not None:
            out.append((event, data))
    return out


def test_real_model_streams_a_well_formed_turn_and_reuses_the_cache(tmp_path: Path):
    from sous.engine.lm import LMEngine

    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: LMEngine(TINY, prompt_cache=True))
    # Same construction as tests/test_gateway_routes.py::_gateway_app: it hands
    # back the mounted Gateway so its real MLX GenerationSession can be closed
    # below. ASGITransport drives no lifespan, so create_server(...).app()
    # alone never runs Gateway.close() — the session's thread would stay
    # parked and never reach release_mlx_thread_state() (see CLAUDE.md).
    mcp = MCPServer("test")
    gateway = mount_gateway(mcp, engines, cfg)
    app = mcp.streamable_http_app()
    tool = {
        "name": "echo",
        "description": "Echo a word back",
        "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}},
    }
    first = {
        "model": "sous-local",
        "max_tokens": 48,
        "stream": True,
        "tools": [tool],
        "messages": [{"role": "user", "content": "Say the word banana and nothing else."}],
    }
    try:
        r = _post(app, first)
        assert r.status_code == 200
        events = _events(r.text)
        kinds = [e for e, _ in events]
        assert kinds[0] == "ping" and kinds[1] == "message_start"
        assert [k for k in kinds if k != "ping"][-1] == "message_stop"  # a late ping may trail
        start = events[1][1]["message"]
        assert start["usage"]["input_tokens"] > 0
        delta = next(d for e, d in events if e == "message_delta")
        assert delta["usage"]["output_tokens"] > 0
        assert delta["delta"]["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
        indices = [d["index"] for e, d in events if e == "content_block_start"]
        assert indices == list(range(len(indices)))

        # Turn 2 extends turn 1's conversation: the gateway's long-lived
        # session keeps the KV cache, so this must be a prefix-cache hit.
        reply_text = "".join(
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d["delta"]["type"] == "text_delta"
        )
        second = {
            **first,
            "stream": False,
            "messages": first["messages"]
            + [
                {"role": "assistant", "content": reply_text or "banana"},
                {"role": "user", "content": "Now say kiwi and nothing else."},
            ],
        }
        r2 = _post(app, second)
        assert r2.status_code == 200
        assert r2.json()["usage"]["input_tokens"] > start["usage"]["input_tokens"]
        stats = engines.get().prompt_cache_stats()
        assert stats["hits"] >= 1, stats
    finally:
        t0 = time.monotonic()
        gateway.close()
        assert time.monotonic() - t0 < 2.5  # bounded: no turn is in flight here
        assert gateway._runner._session is None
    engines.get().unload()
