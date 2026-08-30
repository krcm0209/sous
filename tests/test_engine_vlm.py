import pytest

pytestmark = pytest.mark.model

TINY_VLM = "mlx-community/Qwen2-VL-2B-Instruct-4bit"  # ~1 GB, exercises vlm path
HYBRID_VLM = "mlx-community/Qwen3.5-9B-MLX-4bit"  # ~5.6 GB, linear-attention hybrid


def test_vlm_engine_text_only_generation():
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    e = VLMEngine(TINY_VLM)
    msgs = [{"role": "user", "content": "Say the word kiwi and nothing else."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=64)
    assert isinstance(out, str) and len(out) > 0
    assert e.count_tokens(msgs, WORKER_TOOLS) > 0
    e.unload()


@pytest.mark.parametrize(
    ("model_id", "is_hybrid"),
    [
        pytest.param(TINY_VLM, False, id="pure-attention"),
        pytest.param(HYBRID_VLM, True, id="linear-attention-hybrid"),
    ],
)
def test_vlm_snapshot_restore_is_bit_exact(model_id, is_hybrid):
    """Restoring a contaminated cache must equal a cold prefill of the same
    prefix, exactly — on both cache shapes the orchestrator can build.

    TINY_VLM (Qwen2-VL-2B) is pure attention, so it takes the *trim* path —
    the same one the LM engine's bit-exactness test takes — which is exactly
    what proves the snapshot/trim-versus-state-copy split is architectural,
    not per-backend. HYBRID_VLM (Qwen3.5-9B) is a linear-attention hybrid:
    its make_cache() (mlx_vlm's qwen3_5 LanguageModel) pairs an ArraysCache
    per linear-attention layer with a KVCache per full-attention layer, so it
    takes the *state-copy* branch instead — the one every shipped user
    actually hits on the default 27B model, and the only branch these two
    bit-exactness tests previously left to manual verification."""
    import mlx.core as mx
    from mlx_vlm.models.cache import make_prompt_cache

    from sous.engine.promptcache import restore, snapshot
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(model_id)
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

    snap, nbytes = snapshot(work, e.copy_array)
    # The distinguishing property each model was chosen for: fail loudly if a
    # model ever stops having the cache shape it was picked to exercise.
    if is_hybrid:
        assert nbytes > 0, "a hybrid cache's recurrent layers must be copied"
    else:
        assert nbytes == 0, "a pure-attention cache needs no state copy"
    e.prefill(work, suffix)
    restore(work, snap, e.copy_array)

    for a, b in zip(work, ref, strict=True):
        sa, sb = a.state, b.state
        off = int(getattr(a, "offset", 0) or 0)
        for xa, xb in zip(sa, sb, strict=True):
            if xa is None or xb is None:
                # Only a *pair* of Nones is bit-exact equality; one side None
                # and the other populated is a dropped/invented state entry,
                # exactly the restore bug this test exists to catch.
                assert xa is None and xb is None, "one side is None, the other is not"
                continue
            # Redundant on the pure-attention model: KVCache.state already
            # returns only the live region, so the slice below never trims
            # anything further there. Load-bearing on the hybrid model's own
            # KVCache layers for the same reason; its ArraysCache layers have
            # no offset/trim at all, so hasattr(a, "trim") skips them here.
            if hasattr(a, "trim") and xa.ndim >= 3 and off:
                xa, xb = xa[..., :off, :], xb[..., :off, :]
            d = mx.max(mx.abs(xa.astype(mx.float32) - xb.astype(mx.float32)))
            mx.eval(d)
            assert d.item() == 0.0
    e.unload()


def test_vlm_engine_reuses_across_turns():
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    # Explicitly on: the shipped default is off until the worker stops running
    # each generation on its own thread, and this test is about reuse itself.
    e = VLMEngine(TINY_VLM, prompt_cache=True)
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


DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"


@pytest.mark.parametrize("prompt_cache", [False, True], ids=["cold", "warm-cache"])
def test_vlm_drafter_speeds_up_generation_on_default_model(prompt_cache):
    """The configured drafter must actually engage: after a generation, the
    drafter has recorded speculative rounds, and output is non-empty. Uses the
    shipped default model + drafter pair (heavy — local only, like every test
    in this file).

    Parametrized over prompt_cache because the #55 gate demands both: the
    warm path decodes a SUFFIX onto a cache prefilled by a separate drafter-
    less generate() call, and that split — not the cold path — is where a
    drafter with no captured hidden state would break. Two generations in the
    warm case so the second one actually rides a reused prefix."""
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    e = VLMEngine(DEFAULT_27B, draft_id=DRAFTER, prompt_cache=prompt_cache)
    assert e._draft is not None, "drafter should load and validate on the default model"
    msgs = [{"role": "user", "content": "Write a haiku about rain."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=64)
    assert isinstance(out, str) and len(out) > 0
    assert len(getattr(e._draft, "accept_lens", [])) > 0, "no speculative rounds ran"
    if prompt_cache:
        msgs = [
            *msgs,
            {"role": "assistant", "content": out},
            {"role": "user", "content": "Now one about snow."},
        ]
        e._draft.accept_lens.clear()
        out2 = e.generate(msgs, WORKER_TOOLS, max_tokens=64)
        assert isinstance(out2, str) and len(out2) > 0
        stats = e.prompt_cache_stats()
        assert stats.get("hits", 0) >= 1, f"warm path never engaged: {stats}"
        assert len(e._draft.accept_lens) > 0, "no speculative rounds on the warm path"
    e.unload()
    assert e._draft is None


def test_vlm_incompatible_drafter_degrades_gracefully():
    """A drafter that cannot serve the target (wrong architecture) must not
    take the engine down: warn, run without it, and still generate."""
    import warnings as w

    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        e = VLMEngine(TINY_VLM, draft_id=DRAFTER)
    assert e._draft is None
    assert any("drafter" in str(x.message).lower() for x in caught)
    msgs = [{"role": "user", "content": "Say the word kiwi and nothing else."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=32)
    assert isinstance(out, str) and len(out) > 0
    e.unload()
