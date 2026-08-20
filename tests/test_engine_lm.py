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


def test_lm_snapshot_restore_is_bit_exact():
    """Restoring a contaminated cache must equal a cold prefill of the same
    prefix, exactly. Anything less and reuse silently changes the context."""
    import mlx.core as mx  # ty: ignore[unresolved-import]
    from mlx_lm.models.cache import make_prompt_cache

    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import restore, snapshot

    e = LMEngine(TINY)
    ids = list(e._tokenizer.encode("def f(x):\n    return x + 1\n" * 40))  # ty: ignore[unresolved-attribute]
    prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(e._model)  # ty: ignore[invalid-argument-type]
    e.prefill(ref, prefix)
    work = make_prompt_cache(e._model)  # ty: ignore[invalid-argument-type]
    e.prefill(work, prefix)

    snap, nbytes = snapshot(work, e.copy_array)
    assert nbytes == 0, "a pure-attention cache needs no state copy"
    e.prefill(work, suffix)
    restore(work, snap, e.copy_array)

    for a, b in zip(work, ref, strict=True):
        off = int(a.offset)
        assert off == int(b.offset)
        for xa, xb in zip(a.state, b.state, strict=True):
            d = mx.max(
                mx.abs(xa[..., :off, :].astype(mx.float32) - xb[..., :off, :].astype(mx.float32))
            )
            mx.eval(d)
            assert float(d.item()) == 0.0
    e.unload()


def test_lm_engine_reuses_across_turns():
    from sous.engine.lm import LMEngine
    from sous.protocol import WORKER_TOOLS

    e = LMEngine(TINY)
    msgs = [{"role": "user", "content": "Say the word banana and nothing else."}]
    first = e.generate(msgs, WORKER_TOOLS, max_tokens=32)
    msgs = msgs + [
        {"role": "assistant", "content": first},
        {"role": "user", "content": "Now say kiwi and nothing else."},
    ]
    e.generate(msgs, WORKER_TOOLS, max_tokens=32)
    stats = e.prompt_cache_stats()
    assert stats["hits"] == 1, stats
    assert stats["reused_tokens"] > 0
    e.unload()
