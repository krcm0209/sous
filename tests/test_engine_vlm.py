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


def test_vlm_snapshot_restore_is_bit_exact():
    """Same guarantee, driven through the VLM engine. TINY_VLM (Qwen2-VL-2B)
    is pure attention, so this exercises the *trim* path through the VLM
    engine, not the hybrid state-copy path — which is exactly the case that
    proves the snapshot/trim split is architectural, not per-backend. The
    state-copy path needs a linear-attention hybrid and is verified by hand
    against mlx-community/Qwen3.5-9B-MLX-4bit before the PR, not here."""
    import mlx.core as mx  # ty: ignore[unresolved-import]
    from mlx_vlm.models.cache import make_prompt_cache

    from sous.engine.promptcache import restore, snapshot
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(TINY_VLM)
    # _loaded() rather than e._model directly: it is annotated `-> tuple`, so
    # the unpacked local is untyped and needs no suppression for the None arm
    # that direct attribute access would otherwise hit.
    model, _ = e._loaded()
    ids = e._encode("def f(x):\n    return x + 1\n" * 40)
    prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(model.language_model)
    e.prefill(ref, prefix)
    work = make_prompt_cache(model.language_model)
    e.prefill(work, prefix)

    snap, _ = snapshot(work, e.copy_array)
    e.prefill(work, suffix)
    restore(work, snap, e.copy_array)

    for a, b in zip(work, ref, strict=True):
        sa, sb = a.state, b.state
        off = int(getattr(a, "offset", 0) or 0)
        for xa, xb in zip(sa, sb, strict=True):
            if xa is None or xb is None:
                continue
            # Redundant on this model: KVCache.state already returns only the
            # live region, so the slice below never trims anything further.
            # Kept because a hybrid cache's state can differ; harmless here.
            if hasattr(a, "trim") and xa.ndim >= 3 and off:
                xa, xb = xa[..., :off, :], xb[..., :off, :]
            d = mx.max(mx.abs(xa.astype(mx.float32) - xb.astype(mx.float32)))
            mx.eval(d)
            assert float(d.item()) == 0.0
    e.unload()


def test_vlm_engine_reuses_across_turns():
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    e = VLMEngine(TINY_VLM)
    msgs = [{"role": "user", "content": "Say the word kiwi and nothing else."}]
    first = e.generate(msgs, WORKER_TOOLS, max_tokens=32)
    msgs = msgs + [
        {"role": "assistant", "content": first},
        {"role": "user", "content": "Now say plum and nothing else."},
    ]
    e.generate(msgs, WORKER_TOOLS, max_tokens=32)
    stats = e.prompt_cache_stats()
    assert stats["hits"] == 1, stats
    e.unload()
