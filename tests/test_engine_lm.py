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
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import restore, snapshot

    e = LMEngine(TINY)
    # _loaded() rather than e._model / e._tokenizer directly: it is annotated
    # `-> tuple`, so the unpacked locals are untyped and need no suppression
    # for the None arm that direct attribute access would otherwise hit.
    model, tokenizer = e._loaded()
    ids = list(tokenizer.encode("def f(x):\n    return x + 1\n" * 40))
    prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(model)
    e.prefill(ref, prefix)
    work = make_prompt_cache(model)
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
            assert d.item() == 0.0
    e.unload()


def test_lm_engine_reuses_across_turns():
    from sous.engine.lm import LMEngine
    from sous.protocol import WORKER_TOOLS

    # Explicitly on: the shipped default is off until the worker stops running
    # each generation on its own thread, and this test is about reuse itself.
    e = LMEngine(TINY, prompt_cache=True)
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


def test_lm_engine_streams_deltas_that_reassemble_the_reply():
    from sous.engine.base import Delta
    from sous.engine.lm import LMEngine
    from sous.protocol import WORKER_TOOLS

    e = LMEngine(TINY)
    seen: list[Delta] = []
    msgs = [{"role": "user", "content": "Count from one to five."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=32, on_delta=seen.append)
    assert "".join(d.text for d in seen) == out
    assert [d.output_tokens for d in seen] == sorted(d.output_tokens for d in seen)
    assert seen[-1].finish_reason in ("stop", "length")
    assert all(d.finish_reason is None for d in seen[:-1])
    e.unload()


def test_lm_fork_copy_matches_a_cold_prefill_bit_for_bit():
    """A cache continued from a fork copy must equal a cache prefilled cold
    over the same tokens, exactly — the fork is a second cache, and any
    drift here would silently change the context of every subagent that
    starts from it.

    The reference is the same two prefill calls without a fork, NOT one call
    over header+tail: a quantized matmul's rounding depends on how many rows
    it is given, so mlx's own 440-token prefill and its 220+220 differ by one
    bf16 ULP (measured: a key of -316 against -314) from the first layer's
    keys onward, at position 0. That is a property of the prefill split every
    warm turn already makes, not of the copy — so comparing against it would
    test kernel determinism rather than the fork, and the copy is required to
    be exact here.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import fork_copy, slot_bytes

    def assert_identical(x, y):
        for a, b in zip(x, y, strict=True):
            assert int(a.offset) == int(b.offset)
            for xa, xb in zip(a.state, b.state, strict=True):
                d = mx.max(mx.abs(xa.astype(mx.float32) - xb.astype(mx.float32)))
                mx.eval(d)
                assert d.item() == 0.0

    e = LMEngine(TINY, cache_budget=0)
    model, tokenizer = e._loaded()
    ids = list(tokenizer.encode("def f(x):\n    return x + 1\n" * 40))
    header, tail = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(model)
    e.prefill(ref, header)
    e.prefill(ref, tail)

    src = make_prompt_cache(model)
    e.prefill(src, header)
    # What a source that was never forked from looks like — the same single
    # prefill call, so the comparison below is exact rather than a kernel
    # determinism test.
    untouched = make_prompt_cache(model)
    e.prefill(untouched, header)

    fork = make_prompt_cache(model)
    fork_copy(src, fork, e.copy_array)
    assert slot_bytes(fork) > 0
    assert slot_bytes(fork) <= slot_bytes(src)  # the copy carries no step padding
    assert_identical(fork, src)  # the copy itself, before anything continues it
    e.prefill(fork, tail)

    assert_identical(fork, ref)
    # And the source is still the header and nothing else: the slot stays in
    # the map for the next conversation to fork again, so a copy that aliased
    # it would corrupt every later fork — and would pass every check above.
    assert [int(c.offset) for c in src] == [len(header)] * len(src)
    assert_identical(src, untouched)
    e.unload()


def test_lm_engine_serves_a_second_conversation_from_the_header_fork():
    """End to end on a real tokenizer and template: two conversations with the
    same (long) system turn; the second's first turn is a fork hit."""
    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import FORK_MIN_TOKENS

    e = LMEngine(TINY, prompt_cache=True, cache_budget=1 << 34)
    system = {"role": "system", "content": "You are terse. " * (FORK_MIN_TOKENS // 3)}
    e.generate([system, {"role": "user", "content": "Say A."}], [], 4)
    e.generate([system, {"role": "user", "content": "Say B, please."}], [], 4)
    s = e.prompt_cache_stats()
    assert s["forks"] == 1
    assert s["fork_hits"] == 1
    assert s["reused_tokens"] >= FORK_MIN_TOKENS
    e.unload()
