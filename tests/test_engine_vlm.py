import pytest

pytestmark = pytest.mark.model

TINY_VLM = "mlx-community/Qwen2-VL-2B-Instruct-4bit"  # ~1 GB, exercises vlm path


def test_vlm_engine_text_only_generation():
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    e = VLMEngine(TINY_VLM)
    msgs = [{"role": "user", "content": "Say the word kiwi and nothing else."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=64)
    assert isinstance(out, str) and len(out) > 0
    assert e.count_tokens(msgs, WORKER_TOOLS) > 0
    e.unload()
