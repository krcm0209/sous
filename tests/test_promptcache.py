"""The rules that decide prompt-cache reuse, exercised without mlx.

The engine tests are model-marked and never run in CI, so every decision that
affects correctness is made here, against fakes.
"""

from __future__ import annotations

import threading
import weakref
from typing import cast

import pytest

from sous.engine.base import Delta
from sous.engine.promptcache import (
    PrefixCache,
    PromptCacheStats,
    PromptMemo,
    all_trimmable,
    restore,
    reuse_length,
    snapshot,
    trim_to,
)


class FakeArray:
    """Stands in for an mx.array: only identity and nbytes matter here."""

    def __init__(self, value: int, nbytes: int = 8):
        self.value = value
        self.nbytes = nbytes

    def __eq__(self, other):
        return isinstance(other, FakeArray) and other.value == self.value


def copy_array(a: FakeArray) -> FakeArray:
    return FakeArray(a.value, a.nbytes)


class FakeTrimmable:
    """A KVCache stand-in: O(n) state, rewound by moving an integer offset."""

    offset: int

    def __init__(self, offset: int = 0):
        self.offset = offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


class FakeRecurrent:
    """An ArraysCache stand-in: O(1) state, no trim at all."""

    cache: list

    def __init__(self, state=None):
        self.cache = list(state) if state else [None, None]

    def is_trimmable(self) -> bool:
        return False

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = list(v)


# ---- reuse_length ----------------------------------------------------------


def test_reuse_length_empty_cache_is_a_miss():
    assert reuse_length([], [1, 2, 3]) == 0


def test_reuse_length_strict_extension_reuses_whole_cache():
    assert reuse_length([1, 2, 3], [1, 2, 3, 4]) == 3


def test_reuse_length_exact_match_is_a_miss():
    # mlx rejects an empty prompt, so a full match must leave nothing to feed.
    assert reuse_length([1, 2, 3], [1, 2, 3]) == 0


def test_reuse_length_divergence_at_position_zero():
    assert reuse_length([1, 2, 3], [9, 2, 3, 4]) == 0


def test_reuse_length_divergence_in_the_middle():
    # A partial common prefix is still a miss: sous never rewinds a cache.
    assert reuse_length([1, 2, 3], [1, 9, 3, 4]) == 0


def test_reuse_length_cache_longer_than_new_prompt():
    assert reuse_length([1, 2, 3, 4], [1, 2]) == 0


# ---- all_trimmable ---------------------------------------------------------


def test_all_trimmable_true_for_pure_attention():
    assert all_trimmable([FakeTrimmable(), FakeTrimmable()]) is True


def test_all_trimmable_false_when_any_layer_is_recurrent():
    assert all_trimmable([FakeTrimmable(), FakeRecurrent()]) is False


def test_all_trimmable_false_for_empty_cache():
    assert all_trimmable([]) is False


# ---- snapshot / restore ----------------------------------------------------


def test_snapshot_records_offsets_and_copies_only_recurrent_state():
    cache = [FakeTrimmable(offset=100), FakeRecurrent([FakeArray(1), FakeArray(2)])]
    snap, nbytes = snapshot(cache, copy_array)
    assert snap[0] == ("trim", 100)
    assert snap[1][0] == "state"
    assert nbytes == 16  # only the recurrent layer's two arrays


def test_snapshot_of_pure_attention_costs_nothing():
    _, nbytes = snapshot([FakeTrimmable(offset=5)] * 3, copy_array)
    assert nbytes == 0


def test_snapshot_detaches_the_recurrent_state():
    arr = FakeArray(1)
    cache = [FakeRecurrent([arr, None])]
    snap, _ = snapshot(cache, copy_array)
    assert snap[0][1][0] is not arr  # a copy, not the live reference


def test_restore_rewinds_offsets_and_writes_state_back():
    cache = [FakeTrimmable(offset=100), FakeRecurrent([FakeArray(1), None])]
    snap, _ = snapshot(cache, copy_array)
    cast(FakeTrimmable, cache[0]).offset = 163  # contaminate
    cast(FakeRecurrent, cache[1]).state = [FakeArray(99), FakeArray(98)]
    restore(cache, snap, copy_array)
    assert cast(FakeTrimmable, cache[0]).offset == 100
    assert cast(FakeRecurrent, cache[1]).state == [FakeArray(1), None]


def test_restore_never_grows_a_trimmable_layer():
    cache = [FakeTrimmable(offset=100)]
    snap, _ = snapshot(cache, copy_array)
    cache[0].offset = 50  # already below the snapshot
    restore(cache, snap, copy_array)
    assert cache[0].offset == 50


# ---- trim_to ---------------------------------------------------------------


def test_trim_to_rewinds_every_layer_to_the_same_length():
    cache = [FakeTrimmable(offset=163), FakeTrimmable(offset=163)]
    trim_to(cache, 100)
    assert [c.offset for c in cache] == [100, 100]


def test_trim_to_is_a_no_op_when_already_short_enough():
    cache = [FakeTrimmable(offset=80)]
    trim_to(cache, 100)
    assert cache[0].offset == 80


# ---- stats ----------------------------------------------------------------


def test_stats_as_dict_reports_every_counter():
    s = PromptCacheStats(hits=2, misses=1, reused_tokens=900, snapshot_bytes=16, cold_retries=1)
    assert s.as_dict() == {
        "hits": 2,
        "misses": 1,
        "reused_tokens": 900,
        "snapshot_bytes": 16,
        "cold_retries": 1,
    }


# ---- memo -----------------------------------------------------------------


def test_memo_returns_ids_for_the_same_text():
    m = PromptMemo()
    m.put("stable", "abc", [1, 2, 3])
    assert m.get("stable", "abc") == [1, 2, 3]


def test_memo_misses_on_different_text():
    m = PromptMemo()
    m.put("stable", "abc", [1, 2, 3])
    assert m.get("stable", "abd") is None


def test_memo_keeps_both_slots_independently():
    m = PromptMemo()
    m.put("stable", "abc", [1])
    m.put("full", "abcd", [1, 2])
    assert m.get("stable", "abc") == [1]
    assert m.get("full", "abcd") == [1, 2]


def test_memo_clear_drops_everything():
    m = PromptMemo()
    m.put("stable", "abc", [1])
    m.clear()
    assert m.get("stable", "abc") is None


def test_memo_rejects_an_unknown_slot():
    with pytest.raises(KeyError):
        PromptMemo().put("bogus", "abc", [1])


class FakeHooks:
    """A whole engine, minus mlx. Records what the orchestrator asked for."""

    def __init__(
        self,
        trimmable: bool,
        layers: int = 2,
        fail_once: bool = False,
        decode_impl=None,
    ):
        self.trimmable = trimmable
        self.layers = layers
        self.fail_once = fail_once
        # When a failure is scripted, whether one delta escapes first — the
        # streaming no-retry rule is decided on exactly that difference.
        self.stream_before_fail = False
        self.on_deltas: list = []
        # Injected rather than monkeypatched: assigning over a bound method makes
        # ty report invalid-assignment, and ty checks the tests too.
        self.decode_impl = decode_impl
        self.caches: list[list] = []
        self.prefilled: list[list[int]] = []
        self.decoded: list[list[int]] = []
        self.generated = [7, 8, 9]

    def new_cache(self) -> list:
        if self.trimmable:
            cache = [FakeTrimmable() for _ in range(self.layers)]
        else:
            cache = [FakeTrimmable(), FakeRecurrent([FakeArray(0), None])]
        self.caches.append(cache)
        return cache

    def _advance(self, cache, n):
        for c in cache:
            if isinstance(c, FakeTrimmable):
                c.offset += n

    def prefill(self, cache, token_ids):
        self.prefilled.append(list(token_ids))
        self._advance(cache, len(token_ids))

    def decode(self, cache, token_ids, max_tokens, on_delta=None):
        self.on_deltas.append(on_delta)
        if self.fail_once:
            self.fail_once = False
            if self.stream_before_fail and on_delta is not None:
                on_delta(Delta("partial", 1, None))
            raise RuntimeError("boom")
        if self.decode_impl is not None:
            return self.decode_impl(self, cache, token_ids, max_tokens)
        self.decoded.append(list(token_ids))
        self._advance(cache, len(token_ids) + len(self.generated))
        if on_delta is not None:
            on_delta(Delta("text", 1, "stop"))
        return "text"

    def copy_array(self, a):
        return copy_array(a)


STABLE_1, FULL_1 = [1, 2, 3, 4], [1, 2, 3, 4, 90, 91]
STABLE_2, FULL_2 = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 90, 91]
DIVERGED, DIVERGED_FULL = [1, 2, 99, 4, 5, 6], [1, 2, 99, 4, 5, 6, 90, 91]


def test_trimmable_first_turn_is_a_miss_and_decodes_the_whole_prompt():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    assert pc.generate(STABLE_1, FULL_1, 16) == "text"
    assert h.decoded == [FULL_1]
    assert h.prefilled == []  # the trimmable path never prefills separately
    assert pc.stats()["misses"] == 1


def test_trimmable_leaves_the_cache_at_the_stable_boundary():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    assert [c.offset for c in h.caches[0]] == [len(STABLE_1)] * 2


def test_trimmable_second_turn_reuses_and_decodes_only_the_new_tokens():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    assert h.decoded[1] == FULL_2[len(STABLE_1) :]
    assert len(h.caches) == 1  # the same cache object, not a rebuild
    assert pc.stats()["hits"] == 1
    assert pc.stats()["reused_tokens"] == len(STABLE_1)


def test_non_trimmable_second_turn_prefills_the_delta_then_the_suffix():
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    assert h.prefilled[1] == STABLE_2[len(STABLE_1) :]
    assert h.decoded[1] == FULL_2[len(STABLE_2) :]
    assert pc.stats()["hits"] == 1


def test_non_trimmable_restores_the_cache_to_the_stable_boundary():
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    assert h.caches[0][0].offset == len(STABLE_1)


def test_non_trimmable_reports_snapshot_bytes():
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    assert pc.stats()["snapshot_bytes"] == 8


def test_divergence_rebuilds_the_cache():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(DIVERGED, DIVERGED_FULL, 16)
    assert len(h.caches) == 2  # an elision rewrote history; start over
    assert pc.stats()["misses"] == 2
    assert pc.stats()["hits"] == 0


# ---- same-turn stable/full mismatch (finding 1) ----------------------------

# A template that rewrites rather than appends for add_generation_prompt=True:
# full_ids shares no prefix relation with stable_ids at all, unlike every
# other fixture in this file where full = stable + a generation suffix.
REWRITTEN_STABLE, REWRITTEN_FULL = [1, 2, 3, 4], [9, 9, 9, 9, 90, 91]


@pytest.mark.parametrize("trimmable", [True, False])
def test_same_turn_mismatch_decodes_whole_prompt_cold_and_warns(trimmable):
    h = FakeHooks(trimmable=trimmable)
    pc = PrefixCache(h)
    with pytest.warns(UserWarning, match="prompt cache"):
        assert pc.generate(REWRITTEN_STABLE, REWRITTEN_FULL, 16) == "text"
    assert h.decoded == [REWRITTEN_FULL]  # the whole prompt, not a reuse-sliced suffix
    assert h.prefilled == []  # the guard sits above the trim/state-copy split
    assert pc.stats() == {
        "hits": 0,
        "misses": 1,
        "reused_tokens": 0,
        "snapshot_bytes": 0,
        "cold_retries": 0,
    }


@pytest.mark.parametrize("trimmable", [True, False])
def test_same_turn_mismatch_retains_nothing_so_the_next_turn_is_also_cold(trimmable):
    h = FakeHooks(trimmable=trimmable)
    pc = PrefixCache(h)
    with pytest.warns(UserWarning, match="prompt cache"):
        pc.generate(REWRITTEN_STABLE, REWRITTEN_FULL, 16)
    # A well-formed turn follows; nothing from the bad turn was carried into it.
    pc.generate(STABLE_1, FULL_1, 16)
    assert len(h.caches) == 2  # a fresh cache was built, not a reuse of anything
    assert h.prefilled == ([] if trimmable else [STABLE_1])  # the ordinary miss path
    assert pc.stats()["hits"] == 0
    assert pc.stats()["misses"] == 2


def test_disabled_never_reuses_and_never_counts():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, enabled=False)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == 2
    assert pc.stats() == PromptCacheStats().as_dict()


def test_disabled_never_reads_stable_ids():
    """LMEngine/VLMEngine.generate skip computing the stable render whenever
    the cache is disabled, and pass `[]` in its place — safe only because the
    disabled branch here, `hooks.decode(hooks.new_cache(), list(full_ids), ...)`,
    never looks at `stable_ids` at all. Prove that directly rather than assuming
    it: plant a stale cache plus a `_held` prefix that `stable_ids` genuinely
    extends, so a real engine's `[]` would look nothing like it, and if the
    disabled branch ever started consulting `stable_ids` (or `self._cache` /
    `self._held`) for a reuse decision, this plant would register as a bona
    fide hit — reusing the planted cache and decoding only the unreused suffix
    of full_ids. It must instead build a brand new cache and decode the whole
    prompt, exactly as if `stable_ids` had never been passed.
    """
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, enabled=False)
    # STABLE_1 is a genuine strict prefix of STABLE_2 (see reuse_length), so if
    # consulted this is indistinguishable from a legitimate warm cache left by
    # an earlier, enabled turn.
    planted_cache = h.new_cache()
    pc._cache, pc._held = planted_cache, STABLE_1
    pc.generate(STABLE_2, FULL_2, 16)
    assert h.decoded == [FULL_2]  # the whole prompt, not a reuse-sliced suffix
    assert len(h.caches) == 2  # a fresh cache was built; the planted one was ignored
    assert pc.stats() == PromptCacheStats().as_dict()  # no hit/miss ever recorded


def test_reset_drops_the_cache_so_the_next_turn_is_cold():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.reset()
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == 2


def test_reset_clears_the_counters():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.reset()
    assert pc.stats() == PromptCacheStats().as_dict()


def test_a_warm_failure_retries_cold_once_and_counts_it():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True
    with pytest.warns(UserWarning, match="prompt cache"):
        assert pc.generate(STABLE_2, FULL_2, 16) == "text"
    assert pc.stats()["cold_retries"] == 1
    assert h.decoded[-1] == FULL_2  # the retry fed the whole prompt


def test_a_cold_failure_is_not_retried():
    h = FakeHooks(trimmable=True, fail_once=True)
    pc = PrefixCache(h)
    with pytest.raises(RuntimeError, match="boom"):
        pc.generate(STABLE_1, FULL_1, 16)
    assert pc.stats()["cold_retries"] == 0


def test_the_cold_retrys_cache_is_adopted_for_the_next_turn():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True
    with pytest.warns(UserWarning):
        pc.generate(STABLE_2, FULL_2, 16)
    built = len(h.caches)
    # full = stable + the fixed generation suffix, same shape as every other
    # fixture here (finding 1 now rejects a full_ids that isn't that).
    stable_3 = STABLE_2 + [7]
    pc.generate(stable_3, stable_3 + [90, 91], 16)
    assert len(h.caches) == built  # the retry's cache carried forward


def test_a_propagating_failure_leaves_no_cache_behind():
    """A raise mid-stream leaves the cache holding tokens the token record does
    not describe, so the carrier must already be invalid when it propagates."""

    def always_fail(hooks, cache, token_ids, max_tokens):
        raise RuntimeError("boom")

    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)  # a warm cache is now held
    h.decode_impl = always_fail
    with pytest.warns(UserWarning), pytest.raises(RuntimeError):
        pc.generate(STABLE_2, FULL_2, 16)  # warm fails, then the cold retry does
    h.decode_impl = None
    built = len(h.caches)
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == built + 1  # cold again: nothing was carried


def test_a_late_write_back_from_an_abandoned_generation_is_dropped():
    """reset() bumps an epoch, so a stalled thread landing afterwards drops
    itself instead of resurrecting a finished task's cache under the next one."""
    seen = {}

    def resets_midway(hooks, cache, token_ids, max_tokens):
        pc.reset()  # the task ends while this "thread" runs
        seen["reset"] = True
        hooks.decoded.append(list(token_ids))
        return "text"

    h = FakeHooks(trimmable=True, decode_impl=resets_midway)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    assert seen["reset"]
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == 2  # nothing was carried across the reset


def test_a_miss_releases_the_previous_cache_before_the_replacement_is_prefilled():
    """A miss can only happen mid-task once elision has already fired, which
    means the prompt has already exceeded the window — so the previous cache
    is at its largest exactly when this happens (finding 1). If PrefixCache
    kept it referenced while the replacement is prefilled to full size, both
    would be pinned in memory at once, which is exactly the doubling the spec
    promises never happens.

    Proven with a weakref, not gc internals: these fakes have no reference
    cycle, so CPython's refcounting collects an object the instant its last
    reference is dropped — `weak() is None` is a direct, deterministic read
    of "PrefixCache itself holds nothing more", not a hint that needs
    `gc.collect()` to come true.
    """
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)

    # Every other test wants h.caches to keep a full history; this one must
    # not, or that history alone would keep the old cache reachable forever
    # and the weakref would never go dead regardless of the fix under test.
    old_cache = h.caches.pop(0)
    weak = weakref.ref(old_cache[0])
    del old_cache

    seen: dict[str, bool] = {}

    def checks_the_old_cache_is_already_gone(hooks, cache, token_ids, max_tokens):
        seen["released"] = weak() is None
        return "text"

    h.decode_impl = checks_the_old_cache_is_already_gone
    pc.generate(DIVERGED, DIVERGED_FULL, 16)  # no shared prefix with held: a miss
    assert seen["released"] is True


def test_a_late_cold_retry_write_after_reset_does_not_land_on_fresh_counters():
    """The same trick as the cache case above, but for stats (finding 4): a
    stalled "thread" that resets mid-decode and only afterwards raises still
    gets its retry counted somewhere — but `generate` binds `stats =
    self._stats` once at the top and writes through that local for the rest
    of the call, so the write lands on the object this call started with.
    reset()'s fresh replacement, the one the next task's counters actually
    read via pc.stats(), is left untouched."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)  # an ordinary warm-up turn first

    calls = {"n": 0}

    def resets_then_fails_once(hooks, cache, token_ids, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            pc.reset()  # the task ends while this "thread" is still decoding
            raise RuntimeError("stalled")
        hooks.decoded.append(list(token_ids))
        return "text"

    h.decode_impl = resets_then_fails_once
    with pytest.warns(UserWarning, match="prompt cache"):
        pc.generate(STABLE_2, FULL_2, 16)

    assert calls["n"] == 2  # the failure and its retry both really ran
    assert pc.stats() == PromptCacheStats().as_dict()  # fresh counters, untouched


# ---- streaming deltas ----------------------------------------------------------


def test_on_delta_reaches_decode_on_every_path():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16, seen.append)  # cold
    pc.generate(STABLE_2, FULL_2, 16, seen.append)  # warm
    pc.generate(STABLE_2 + [7], FULL_2[:-2] + [7, 90, 91], 16)  # no consumer
    assert seen == [Delta("text", 1, "stop"), Delta("text", 1, "stop")]
    # The cache wraps the callback to count emissions, so decode sees a
    # callable (not the very object) when one was given, and None otherwise.
    assert [cb is not None for cb in h.on_deltas] == [True, True, False]
    disabled = PrefixCache(FakeHooks(trimmable=True), enabled=False)
    disabled.generate(STABLE_1, FULL_1, 16, seen.append)
    assert len(seen) == 3


def test_a_warm_failure_after_streamed_deltas_is_not_retried():
    """A cold retry would replay text the consumer already forwarded; the
    failure is surfaced instead, and the counters say no retry happened."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True
    h.stream_before_fail = True
    with pytest.raises(RuntimeError, match="not retrying cold"):
        pc.generate(STABLE_2, FULL_2, 16, lambda d: None)
    assert pc.stats()["cold_retries"] == 0
    # The warm attempt raised before FakeHooks recorded it and no cold retry
    # ran: the only decode on record is the first turn's, decode was entered
    # exactly twice, and no replacement cache was ever built.
    assert h.decoded == [FULL_1]
    assert len(h.on_deltas) == 2
    assert len(h.caches) == 1


def test_a_replay_safe_warm_failure_after_streamed_deltas_still_retries_cold():
    """A ReplaySafe on_delta's output never left the process (it is a
    non-streaming turn's accounting-only callback), so — unlike a bare
    callback — its emitted deltas do not forbid the cold retry."""
    from sous.engine.base import ReplaySafe

    seen: list[Delta] = []
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True
    h.stream_before_fail = True
    with pytest.warns(UserWarning, match="retrying cold"):
        result = pc.generate(STABLE_2, FULL_2, 16, ReplaySafe(seen.append))
    assert result == "text"
    assert pc.stats()["cold_retries"] == 1
    # Both the failed warm attempt's partial delta and the retry's final one
    # reached the callback: nothing about ReplaySafe suppresses delivery, it
    # only tells the cache the deliveries were never forwarded to a client.
    assert seen == [Delta("partial", 1, None), Delta("text", 1, "stop")]


def test_a_warm_failure_before_any_delta_still_retries_cold():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True  # raises before emitting anything (stream_before_fail stays False)
    with pytest.warns(UserWarning, match="retrying cold"):
        assert pc.generate(STABLE_2, FULL_2, 16, seen.append) == "text"
    assert pc.stats()["cold_retries"] == 1
    assert seen == [Delta("text", 1, "stop")]  # only the retry's delta got out


# ---- thread ownership (issue #34) ------------------------------------------


def test_a_cache_built_on_another_thread_is_a_cold_miss():
    """mlx KV-cache arrays are usable only from the thread whose streams
    created them (issue #34). A cache slot built on thread A must refuse a
    consumer on any other thread rather than let it touch those arrays."""
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)

    def on_thread_a() -> None:
        pc.generate(STABLE_1, FULL_1, 16)
        pc.generate(STABLE_2, FULL_2, 16)  # strictly extends STABLE_1: reuses

    thread_a = threading.Thread(target=on_thread_a)
    thread_a.start()
    thread_a.join()

    # The slot is populated and owned by thread A: the second call above,
    # still running on thread A, reused the first call's cache.
    assert pc.stats()["hits"] == 1
    assert pc.stats()["reused_tokens"] == len(STABLE_1)

    # From the test's own thread — a different thread than the one that built
    # the cache — call with stable ids that strictly extend the held prefix:
    # the shape that WOULD reuse if thread ownership weren't checked.
    stable_3 = [*STABLE_2, 7, 8]
    full_3 = [*stable_3, 90, 91]
    pc.generate(stable_3, full_3, 16)

    assert pc.stats()["misses"] == 2  # counted as a miss, not a cross-thread hit
    assert pc.stats()["reused_tokens"] == len(STABLE_1)  # unchanged
    assert pc.stats()["cold_retries"] == 0  # cold directly, not via a failed warm attempt
    assert h.prefilled[-1] == stable_3  # the full stable ids, no reuse offset

    # The slot is now owned by the main thread: a further call from here with
    # an extending prefix reuses normally.
    stable_4 = [*stable_3, 9, 10]
    full_4 = [*stable_4, 90, 91]
    pc.generate(stable_4, full_4, 16)
    assert pc.stats()["hits"] == 2
