"""Runtime gate for int8 prefill: real weights, real tensor units, local only.

Acceptance from the spec: routed count 336 on the default model, >= 1.4x prefill
at 4,096 tokens and >= 1.3x at 32,768, KL(stock || int8) <= 0.06 nats on the
standard prompt, and a generation session that exits cleanly with the kernels on.
Takes ~6 minutes on the M5 Pro (two 27B loads, two 32K prefills).
"""

import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill as i8  # noqa: E402

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not i8.availability().available, reason=f"no tensor units: {i8.availability().reason}"
    ),
]

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
HYBRID_VLM = "mlx-community/Qwen3.5-9B-MLX-4bit"
ROOT = Path(__file__).resolve().parents[1]


def _standard_text() -> str:
    return (ROOT / "src/sous/engine/promptcache.py").read_text() + (
        ROOT / "src/sous/gateway/turn.py"
    ).read_text()


def _repeat_to(ids: list[int], n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


def _prefill_seconds(engine, ids: list[int]) -> float:
    best = float("inf")
    for _ in range(2):
        cache = engine.new_cache()
        mx.synchronize()
        t = time.perf_counter()
        engine.prefill(cache, ids)
        mx.synchronize()
        best = min(best, time.perf_counter() - t)
        del cache
        mx.clear_cache()
    return best


def _last_logits(engine, ids: list[int], positions: int = 256):
    model, _ = engine._loaded()
    cache = engine.new_cache()
    out = model.language_model(mx.array(ids)[None], cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    logits = logits[0, -positions:].astype(mx.float32)
    mx.eval(logits)
    return logits


def _kl(p_logits, q_logits) -> float:
    lp = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    lq = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
    return mx.mean(mx.sum(mx.exp(lp) * (lp - lq), axis=-1)).item()


def test_default_model_meets_the_acceptance_bars():
    from sous.engine.vlm import VLMEngine

    stock = VLMEngine(DEFAULT_27B, cache_budget=0)
    assert stock.int8_prefill_status["state"] == "off"
    base = stock._encode(_standard_text())
    ids_4k, ids_32k, ids_2k = _repeat_to(base, 4096), _repeat_to(base, 32768), base[:2048]
    stock_4k = _prefill_seconds(stock, ids_4k)
    stock_32k = _prefill_seconds(stock, ids_32k)
    stock_logits = _last_logits(stock, ids_2k)
    stock.unload()
    del stock
    mx.clear_cache()

    fast = VLMEngine(DEFAULT_27B, cache_budget=0, int8_prefill=True)
    assert fast.int8_prefill_status == {"state": "active", "reason": None, "routed": 336}
    fast_4k = _prefill_seconds(fast, ids_4k)
    fast_32k = _prefill_seconds(fast, ids_32k)
    fast_logits = _last_logits(fast, ids_2k)
    kl = _kl(stock_logits, fast_logits)
    fast.unload()

    print(
        f"\nprefill 4096: {4096 / stock_4k:.0f} -> {4096 / fast_4k:.0f} tok/s "
        f"({stock_4k / fast_4k:.2f}x)"
        f"\nprefill 32768: {32768 / stock_32k:.0f} -> {32768 / fast_32k:.0f} tok/s "
        f"({stock_32k / fast_32k:.2f}x)"
        f"\nKL(stock || int8) over 256 positions: {kl:.4f} nats"
    )
    assert stock_4k / fast_4k >= 1.4
    assert stock_32k / fast_32k >= 1.3
    assert kl <= 0.06


def test_hybrid_9b_routes_its_real_gdn_and_mlp_classes_and_still_generates():
    """The CI routing tests use a stand-in GDN; this is the real
    Qwen3_5GatedDeltaNet / Qwen3_5MLP pair from mlx-vlm on a smaller affine-Q4
    checkpoint, through a ManagedEngine session so the generation thread exits
    with the kernels loaded (the mlx thread-teardown rule)."""
    from sous.engine.base import ManagedEngine
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    engine = VLMEngine(HYBRID_VLM, cache_budget=0, int8_prefill=True)
    assert engine.int8_prefill_status["state"] == "active"
    assert engine.int8_prefill_status["routed"] > 0
    managed = ManagedEngine(engine)
    prompt = "Summarize this in one sentence:\n" + _standard_text()[:6000]
    msgs = [{"role": "user", "content": prompt}]
    session = managed.session()
    try:
        out = session.generate(msgs, WORKER_TOOLS, 32, 600.0)
    finally:
        session.close()
        assert session.join(timeout=30)
    assert isinstance(out, str) and out.strip()
    engine.unload()
