import pytest

pytestmark = pytest.mark.model  # needs a real model download; run locally only

TINY = "mlx-community/Qwen3-0.6B-4bit"  # ~350 MB


def test_lm_engine_generates_and_counts():
    from sous.engine.lm import LMEngine
    from sous.protocol import WORKER_TOOLS

    e = LMEngine(TINY)
    msgs = [{"role": "user", "content": "Say the word banana and nothing else."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=64)
    assert isinstance(out, str) and len(out) > 0
    assert e.count_tokens(msgs, WORKER_TOOLS) > 0
    e.unload()
