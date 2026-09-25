"""The real default model's verify forward: grouped attention against mlx-vlm's
per-row loop.

Acceptance bars:
- The engine reports verify attention active with the default drafter.
- At prefixes that straddle the M5 Pro's plan transitions (1021, 8190, 32766)
  and at a subagent-sized 57,000, for T = 3..8, the logits and every DFlash2
  capture hidden state are bit-equal with the hook on and off.
- The grouped path runs on all 16 full-attention layers.

Takes about 6 minutes on the M5 Pro: one 27B load and a 57K prefill.
"""

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import verifyattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
CAPTURE = [5, 19, 33, 47, 61]  # DFlash2's target_layer_ids on the 27B
PREFIXES = [1021, 8190, 32766, 57000]
ROOT = Path(__file__).resolve().parents[1]


def _corpus(engine, n):
    files = sorted((ROOT / "src" / "sous").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ids = engine._encode("".join(f.read_text() for f in files))
    assert len(ids) >= n, len(ids)
    return ids[:n]


def _offset(cache):
    return max((int(getattr(c, "offset", 0) or 0) for c in cache), default=0)


def _verify(lm, cache, tokens):
    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=CAPTURE,
        speculative_verify=True,
    )
    arrays = [out.logits, *out.hidden_states]
    mx.eval(arrays)
    out.gdn_states.abort()
    return arrays


def test_grouped_verify_forward_is_bit_equal_to_mlx_vlms_loop():
    engine = VLMEngine(DEFAULT_27B, cache_budget=0, draft_id=DRAFTER)
    try:
        status = engine.verify_attention_status
        assert status["state"] == "active", status
        modules = verifyattn._attention_modules(engine._model)
        assert len(modules) == 16
        gqa = getattr(modules[0], verifyattn._TAG)
        assert engine._model is not None
        lm = engine._model.language_model
        ids = _corpus(engine, max(PREFIXES) + 2056)
        cache = engine.new_cache()
        for target in PREFIXES:
            while _offset(cache) < target:
                start = _offset(cache)
                engine.prefill(cache, ids[start : start + min(2048, target - start)])
                mx.eval([c.state for c in cache if hasattr(c, "state")])
                assert _offset(cache) > start, "prefill did not advance"
            prefix = _offset(cache)
            for t in range(3, 9):
                tokens = ids[prefix : prefix + t]
                for module in modules:
                    object.__setattr__(module, verifyattn._TAG, 0)
                stock = _verify(lm, cache, tokens)
                for module in modules:
                    object.__setattr__(module, verifyattn._TAG, gqa)
                before = verifyattn.calls["grouped"]
                ours = _verify(lm, cache, tokens)
                assert verifyattn.calls["grouped"] - before == len(modules), (prefix, t)
                assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True)), (
                    prefix,
                    t,
                )
    finally:
        engine.unload()
