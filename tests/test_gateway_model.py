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

    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
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
        # Disjoint usage: the reused prefix is a cache read and input_tokens is
        # the rest, so the two together are the longer prompt and the read is
        # the hit made visible to the client.
        usage = r2.json()["usage"]
        assert usage["cache_read_input_tokens"] > 0, usage
        assert (
            usage["input_tokens"] + usage["cache_read_input_tokens"]
            > start["usage"]["input_tokens"]
        )
        stats = engines.get().prompt_cache_stats()
        assert stats["hits"] >= 1, stats
    finally:
        t0 = time.monotonic()
        gateway.close()
        assert time.monotonic() - t0 < 2.5  # bounded: no turn is in flight here
        assert gateway._runner._session is None
    engines.get().unload()


def _usage(r: httpx.Response) -> dict:
    return r.json()["usage"]


MODELS = [
    pytest.param(TINY, "lm", id="tiny"),
    # The hybrid model the copy-then-extend shape exists for: its recurrent
    # layers cannot rewind, so a retained slot is a real second cache.
    pytest.param("mlx-community/Qwen3.8-27B-4bit", "vlm", id="27b"),
]


def _engine(model_id: str, backend: str):
    # An explicit budget: the factory bypasses [model].prompt_cache_gb, and a
    # budget of 0 would move T1's slot instead of retaining it. temperature=0.0
    # makes decoding greedy (make_sampler(temp=0) is argmax): the test below
    # compares warm and cold generations token-for-token, and only greedy
    # decoding makes that comparison meaningful — sampling at any temperature
    # above zero is not pinned by seeding the global mlx RNG.
    if backend == "vlm":
        from sous.engine.vlm import VLMEngine

        return VLMEngine(model_id, temperature=0.0, prompt_cache=True, cache_budget=8 << 30)
    from sous.engine.lm import LMEngine

    return LMEngine(model_id, temperature=0.0, prompt_cache=True, cache_budget=8 << 30)


@pytest.mark.parametrize(("model_id", "backend"), MODELS)
def test_an_attachment_keeps_the_conversation_warm_and_bit_exact(
    tmp_path: Path, model_id: str, backend: str
):
    """T1 a brief; T3 the same conversation plus a tool exchange and a
    role:system attachment after the tool result. T3 must be served from
    T1's slot (cache_read_input_tokens covers T1's whole render) and produce
    exactly what a cold run of the same prompt produces under greedy decoding
    — the render is the same text in the same place, only warm. Then a branch
    of T1 (a summary-shaped last turn) is served from the retained slot and
    is bit-exact against its own cold run too."""
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    engines = EngineManager(cfg, engine_factory=lambda mid: _engine(model_id, backend))
    mcp = MCPServer("test")
    gateway = mount_gateway(mcp, engines, cfg)
    app = mcp.streamable_http_app()
    read_tool = {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    }
    system = [{"type": "text", "text": "You are a terse assistant. " * 40}]
    t1 = [{"role": "user", "content": "Read the file notes.txt and tell me its first word."}]
    t3 = [
        *t1,
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "notes.txt"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "banana bread recipe"},
                {"type": "text", "text": "<total_tokens>900 tokens left</total_tokens>\n"},
            ],
        },
        {
            "role": "system",
            "content": "Available agent types for the Agent tool:\n- Explore: read-only",
        },
    ]
    branch = [
        *t1,
        {"role": "assistant", "content": "Reading."},
        {"role": "user", "content": "Describe your most recent action in 3-5 words."},
    ]

    def body(messages: list[dict]) -> dict:
        return {
            "model": "sous-local",
            "max_tokens": 24,
            "tools": [read_tool],
            "system": system,
            "messages": messages,
        }

    try:
        first = _post(app, body(t1))
        assert first.status_code == 200
        warm = _post(app, body(t3))
        assert warm.status_code == 200
        assert _usage(warm)["cache_read_input_tokens"] > 0
        branch_warm = _post(app, body(branch))
        assert (
            _usage(branch_warm)["cache_read_input_tokens"]
            == _usage(warm)["cache_read_input_tokens"]
        )

        engines.get().reset_prompt_cache()
        cold = _post(app, body(t3))
        assert _usage(cold)["cache_read_input_tokens"] == 0
        assert cold.json()["content"] == warm.json()["content"]
        engines.get().reset_prompt_cache()
        branch_cold = _post(app, body(branch))
        assert branch_cold.json()["content"] == branch_warm.json()["content"]
    finally:
        gateway.close()
        engines.get().unload()
    assert gateway._runner._session is None
