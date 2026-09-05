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
    FORK_MIN_TOKENS,
    PrefixCache,
    PromptCacheStats,
    PromptMemo,
    all_trimmable,
    auto_cache_budget,
    fork_copy,
    fork_point,
    restore,
    reuse_length,
    slot_bytes,
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
    """A KVCache stand-in: O(n) state, rewound by moving an integer offset.
    `state` carries the offset as one FakeArray so fork_copy can rebuild it
    the way KVCache's setter derives offset from the copied shape; nbytes
    grows with the offset like a real KV buffer."""

    offset: int

    def __init__(self, offset: int = 0):
        self.offset = offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        return [FakeArray(self.offset, nbytes=self.offset * 8)]

    @state.setter
    def state(self, v):
        self.offset = v[0].value

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        assert not v

    @property
    def nbytes(self) -> int:
        return self.offset * 8


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

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        assert not v

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.cache if a is not None)


def _empty_stats() -> dict:
    return {**PromptCacheStats().as_dict(), "slots": 0, "resident_bytes": 0}


# Roomy enough that nothing is ever evicted for size in tests that want to see
# several slots coexist; the default of 0 keeps a single slot, as before.
ROOMY = 1 << 40


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
        self.headroom_value: int | None = None

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

    def headroom(self):
        return self.headroom_value


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
    assert pc.stats() == {**_empty_stats(), "misses": 1}


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
    assert pc.stats() == _empty_stats()


def test_disabled_never_reads_stable_ids():
    """LMEngine/VLMEngine.generate skip computing the stable render whenever
    the cache is disabled, and pass `[]` in its place — safe only because the
    disabled branch never looks at `stable_ids` at all. Plant a slot whose
    held prefix `stable_ids` genuinely extends; if the disabled branch ever
    consulted the slots for a reuse decision, this plant would register as a
    bona fide hit. It must instead build a brand new cache and decode the
    whole prompt."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, enabled=False, max_bytes=ROOMY)
    planted = h.new_cache()
    pc._plant(planted, STABLE_1)  # test seam: a turn slot owned by this thread
    pc.generate(STABLE_2, FULL_2, 16)
    assert h.decoded == [FULL_2]
    assert len(h.caches) == 2
    assert pc.stats() == {**_empty_stats(), "slots": 1, "resident_bytes": slot_bytes(planted)}


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
    assert pc.stats() == _empty_stats()


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
    assert pc.stats() == _empty_stats()  # fresh counters, untouched


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


# ---- fork_point ------------------------------------------------------------

HEADER = list(range(FORK_MIN_TOKENS))
BODY = [90_000, 90_001, 90_002]


def test_fork_point_accepts_a_long_header_that_is_a_strict_prefix():
    assert fork_point(HEADER, HEADER + BODY) == len(HEADER)


def test_fork_point_rejects_a_header_that_is_not_a_prefix():
    assert fork_point(HEADER, [1, *HEADER[1:], *BODY]) == 0


def test_fork_point_rejects_a_header_equal_to_the_whole_render():
    # Nothing would be left to prefill after the fork; also an exact match is
    # a miss under reuse_length, so the two rules agree.
    assert fork_point(HEADER, HEADER) == 0


def test_fork_point_rejects_a_short_header():
    short = HEADER[: FORK_MIN_TOKENS - 1]
    assert fork_point(short, short + BODY) == 0


# ---- fork_copy / slot_bytes --------------------------------------------------


def test_fork_copy_gives_the_destination_the_same_offsets_and_state():
    src = [FakeTrimmable(offset=100), FakeRecurrent([FakeArray(1), FakeArray(2)])]
    dst = [FakeTrimmable(), FakeRecurrent()]
    fork_copy(src, dst, copy_array)
    assert cast(FakeTrimmable, dst[0]).offset == 100
    assert cast(FakeRecurrent, dst[1]).state == [FakeArray(1), FakeArray(2)]


def test_fork_copy_detaches_the_recurrent_state():
    arr = FakeArray(1)
    src = [FakeRecurrent([arr, None])]
    dst = [FakeRecurrent()]
    fork_copy(src, dst, copy_array)
    assert cast(FakeRecurrent, dst[0]).state[0] is not arr
    assert cast(FakeRecurrent, dst[0]).state[1] is None


def test_fork_copy_leaves_the_source_untouched():
    src = [FakeTrimmable(offset=100)]
    fork_copy(src, [FakeTrimmable()], copy_array)
    cast(FakeTrimmable, src[0]).offset += 1  # the copy must not alias the source
    assert cast(FakeTrimmable, src[0]).offset == 101


def test_slot_bytes_sums_every_layer():
    cache = [FakeTrimmable(offset=10), FakeRecurrent([FakeArray(1, nbytes=8), None])]
    assert slot_bytes(cache) == 80 + 8


def test_slot_bytes_tolerates_a_layer_without_nbytes():
    class Bare:
        pass

    assert slot_bytes([Bare(), FakeTrimmable(offset=1)]) == 8


# ---- auto_cache_budget -------------------------------------------------------


def test_auto_cache_budget_is_working_set_minus_weights_reserve_and_slack():
    gib = 1 << 30
    got = auto_cache_budget(working_set=52 * gib, active=18 * gib, reserve_bytes=8 * gib)
    assert got == (52 - 18 - 8 - 2) * gib


def test_auto_cache_budget_never_goes_negative():
    gib = 1 << 30
    assert auto_cache_budget(working_set=24 * gib, active=18 * gib, reserve_bytes=8 * gib) == 0


# ---- stats -----------------------------------------------------------------


def test_stats_as_dict_reports_every_counter():
    s = PromptCacheStats(
        hits=2,
        misses=1,
        reused_tokens=900,
        snapshot_bytes=16,
        cold_retries=1,
        fork_hits=1,
        forks=1,
        evictions=3,
    )
    assert s.as_dict() == {
        "hits": 2,
        "misses": 1,
        "reused_tokens": 900,
        "snapshot_bytes": 16,
        "cold_retries": 1,
        "fork_hits": 1,
        "forks": 1,
        "evictions": 3,
    }


def test_stats_add_sums_the_counters_and_maxes_the_snapshot_gauge():
    a = PromptCacheStats(hits=1, forks=1, snapshot_bytes=64)
    a.add(PromptCacheStats(hits=2, evictions=4, snapshot_bytes=16))
    assert (a.hits, a.forks, a.evictions) == (3, 1, 4)
    # snapshot_bytes is assigned per turn, not accumulated: folding owners
    # together must report the largest one's copy cost, not 80.
    assert a.snapshot_bytes == 64
    a.add(PromptCacheStats(snapshot_bytes=100))
    assert a.snapshot_bytes == 100


def test_memo_accepts_the_header_slot():
    m = PromptMemo()
    m.put("header", "sys", [1, 2])
    assert m.get("header", "sys") == [1, 2]


# ---- keyed slots -------------------------------------------------------------

A1, A1_FULL = [1, 2, 3, 4], [1, 2, 3, 4, 90, 91]
A2, A2_FULL = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 90, 91]
B1, B1_FULL = [7, 8, 9, 10], [7, 8, 9, 10, 90, 91]
B2, B2_FULL = [7, 8, 9, 10, 11], [7, 8, 9, 10, 11, 90, 91]


def test_two_interleaved_conversations_both_reuse():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)  # A's slot survived B's turn
    pc.generate(B2, B2_FULL, 16)
    assert pc.stats()["hits"] == 2
    assert pc.stats()["misses"] == 2
    assert h.decoded[2] == A2_FULL[len(A1) :]
    assert h.decoded[3] == B2_FULL[len(B1) :]
    assert len(h.caches) == 2  # one cache per conversation, each extended in place


def test_a_turn_slot_is_consumed_by_the_turn_that_extends_it():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)
    held = [s.held for s in pc.slots()]
    assert held == [A2]  # not [A1, A2]: the old key is gone with the cache it named


def test_the_longest_matching_slot_wins():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    short = h.new_cache()
    long = h.new_cache()
    pc._plant(short, A1)
    pc._plant(long, A2)
    a3 = [*A2, 7]
    pc.generate(a3, [*a3, 90, 91], 16)
    assert pc.stats()["reused_tokens"] == len(A2)
    assert h.decoded[-1] == [7, 90, 91]


def test_default_budget_keeps_exactly_one_slot():
    """max_bytes=0 is the constructor default and today's behaviour: at most
    one slot is ever resident, so a second conversation replaces the first."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    assert [s.held for s in pc.slots()] == [B1]
    assert pc.stats()["evictions"] == 1
    pc.generate(A2, A2_FULL, 16)
    assert pc.stats()["misses"] == 3  # A's slot was evicted by B's publish


def test_a_fork_the_turn_selected_survives_the_pre_turn_eviction_pass():
    """The pre-turn cap pass runs after `_take`, and a fork slot stays in the
    map when it is taken — so the pass must protect it. Otherwise the turn
    that decided to reuse the shared slot is the very turn that drops it, and
    no later conversation finds it."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=1)  # tighter than any populated slot
    planted = h.new_cache()
    h._advance(planted, len(A1))  # a fork with real bytes, so it is chargeable
    pc._plant(planted, A1, kind="fork")
    seen: dict = {}

    def look_while_the_turn_runs(hooks, cache, token_ids, max_tokens):
        seen["kinds"] = [s.kind for s in pc.slots()]
        hooks.decoded.append(list(token_ids))
        return "text"

    h.decode_impl = look_while_the_turn_runs
    pc.generate(A2, A2_FULL, 16)
    assert seen["kinds"] == ["fork"]
    assert pc.stats()["fork_hits"] == 1
    # Charged like any slot: the publish pass, which protects only the new
    # turn slot, is what finally evicts it — once, not once before and once
    # after.
    assert pc.stats()["evictions"] == 1
    assert [s.kind for s in pc.slots()] == ["turn"]


def test_pressure_eviction_never_holds_the_lock_across_a_headroom_call():
    """`headroom()` is an engine call (an mlx memory query on a real engine).
    reset() must never queue behind one: its caller can be the worker's
    `finally` while a stalled generation still runs, and blocking there wedges
    the next task."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY, reserve_bytes=1000)
    reading = threading.Event()
    release = threading.Event()

    def blocking_headroom() -> int | None:
        reading.set()
        release.wait(10)
        return 5000  # above the reserve, so nothing is evicted either way

    h.headroom = blocking_headroom  # ty: ignore[invalid-assignment]
    turn = threading.Thread(target=lambda: pc.generate(A1, A1_FULL, 16))
    turn.start()
    assert reading.wait(5)

    done = threading.Event()

    def reset_and_report():
        pc.reset()
        done.set()

    threading.Thread(target=reset_and_report).start()
    assert done.wait(2), "reset() blocked behind the engine's headroom() call"
    release.set()
    turn.join(5)
    assert not turn.is_alive()


def test_slot_count_is_capped():
    from sous.engine.promptcache import MAX_SLOTS

    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    for i in range(MAX_SLOTS + 3):
        stable = [1000 + i, 1]
        pc.generate(stable, [*stable, 90, 91], 16)
    assert len(pc.slots()) == MAX_SLOTS
    assert pc.stats()["evictions"] == 3
    # LRU: the three oldest are the ones gone.
    assert [1000, 1] not in [s.held for s in pc.slots()]
    assert [1000 + MAX_SLOTS + 2, 1] in [s.held for s in pc.slots()]


def test_resident_bytes_track_the_slots():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    expected = sum(slot_bytes(c) for c in h.caches)
    assert pc.stats()["resident_bytes"] == expected
    assert pc.stats()["slots"] == 2


# ---- owners ------------------------------------------------------------------


def _run_on_thread(fn) -> threading.Thread:
    t = threading.Thread(target=fn)
    t.start()
    t.join()
    return t


def test_stats_are_scoped_to_the_owner_thread():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)  # main thread: one miss

    def other():
        pc.generate(B1, B1_FULL, 16)
        pc.generate(B2, B2_FULL, 16)

    t = _run_on_thread(other)
    mine = pc.stats(owner=threading.current_thread())
    assert (mine["hits"], mine["misses"]) == (0, 1)
    # The other thread is dead: its slots are swept and its counters folded
    # into the daemon-wide history rather than lost.
    theirs = pc.stats(owner=t)
    assert (theirs["hits"], theirs["misses"], theirs["slots"]) == (0, 0, 0)
    total = pc.stats()
    assert (total["hits"], total["misses"], total["slots"]) == (1, 2, 1)


def test_reset_with_an_owner_drops_only_that_owners_slots():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    stop = threading.Event()
    started = threading.Event()

    def other():
        pc.generate(B1, B1_FULL, 16)
        started.set()
        stop.wait(5)

    t = threading.Thread(target=other)
    t.start()
    started.wait(5)
    assert pc.stats()["slots"] == 2
    pc.reset(owner=t)
    assert [s.held for s in pc.slots()] == [A1]
    assert pc.stats(owner=t)["misses"] == 0  # folded into history...
    assert pc.stats()["misses"] == 2  # ...not lost
    stop.set()
    t.join(5)
    pc.generate(A2, A2_FULL, 16)  # the surviving owner's slot still reuses
    assert pc.stats(owner=threading.current_thread())["hits"] == 1


def test_a_retired_owners_late_publish_is_refused():
    """The worker retires its session's thread when the task ends; a stalled
    generation on that thread that finishes later must not resurrect a slot."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    me = threading.current_thread()

    def retires_midway(hooks, cache, token_ids, max_tokens):
        pc.reset(owner=me)  # the task ends while this generation runs
        hooks.decoded.append(list(token_ids))
        return "text"

    h.decode_impl = retires_midway
    pc.generate(A1, A1_FULL, 16)
    assert pc.slots() == []
    h.decode_impl = None
    pc.generate(A2, A2_FULL, 16)  # the retired thread is us: still refused
    assert pc.slots() == []
    assert pc.stats()["misses"] == 2


def test_a_dead_owners_slots_are_swept_on_the_next_call():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    _run_on_thread(lambda: pc.generate(B1, B1_FULL, 16))
    # No retire happened; the thread simply exited. Anything it left is
    # unusable (its mlx streams are gone) and must not stay resident.
    assert pc.stats()["slots"] == 0
    assert pc.stats()["misses"] == 1  # history keeps the count


def test_reset_without_an_owner_still_drops_everything_and_bumps_the_epoch():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    _run_on_thread(lambda: pc.generate(B1, B1_FULL, 16))
    pc.reset()
    assert pc.slots() == []
    assert pc.stats() == _empty_stats()


# ---- forks -------------------------------------------------------------------

# Two conversations that share a header long enough to fork, then diverge.
H = list(range(1, FORK_MIN_TOKENS + 1))
FORK = len(H)
C1 = [*H, 501, 502]
C1_FULL = [*C1, 90, 91]
C1_NEXT = [*C1, 503]
C1_NEXT_FULL = [*C1_NEXT, 90, 91]
C2 = [*H, 601, 602, 603]
C2_FULL = [*C2, 90, 91]
C3 = [*H, 701]
C3_FULL = [*C3, 90, 91]


@pytest.mark.parametrize("trimmable", [True, False])
def test_a_cold_turn_forks_at_the_header(trimmable):
    h = FakeHooks(trimmable=trimmable)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    kinds = sorted((s.kind, s.held) for s in pc.slots())
    assert kinds == [("fork", H), ("turn", C1)]
    assert pc.stats()["forks"] == 1
    # The header was prefilled as its own segment, whatever the layer kinds,
    # because the copy has to be taken exactly at the boundary.
    assert h.prefilled[0] == H
    if trimmable:
        assert h.decoded[0] == C1_FULL[FORK:]  # the rest still fuses into decode
    else:
        assert h.prefilled[1] == C1[FORK:]


def test_the_fork_is_an_independent_copy():
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    fork = next(s for s in pc.slots() if s.kind == "fork")
    turn = next(s for s in pc.slots() if s.kind == "turn")
    assert fork.cache is not turn.cache
    assert cast(FakeTrimmable, fork.cache[0]).offset == FORK
    assert cast(FakeTrimmable, turn.cache[0]).offset == len(C1)


def test_a_new_conversation_sharing_the_header_starts_from_the_fork():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    pc.generate(C2, C2_FULL, 16, fork_at=FORK)
    s = pc.stats()
    assert (s["hits"], s["fork_hits"], s["reused_tokens"]) == (1, 1, FORK)
    assert h.decoded[-1] == C2_FULL[FORK:]  # only the tail was fed
    # The fork stayed (copied, not consumed), so a third conversation hits too.
    assert [x.held for x in pc.slots() if x.kind == "fork"] == [H]
    pc.generate(C3, C3_FULL, 16, fork_at=FORK)
    assert pc.stats()["fork_hits"] == 2
    assert sorted(x.held for x in pc.slots() if x.kind == "turn") == sorted([C1, C2, C3])


def test_a_fork_hit_does_not_fork_again_and_neither_does_a_turn_hit():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    pc.generate(C2, C2_FULL, 16, fork_at=FORK)  # fork hit: reuse == fork_at
    pc.generate(C1_NEXT, C1_NEXT_FULL, 16, fork_at=FORK)  # turn hit: reuse > fork_at
    assert pc.stats()["forks"] == 1
    assert sum(1 for s in pc.slots() if s.kind == "fork") == 1


def test_a_second_conversations_turn_slot_does_not_disturb_the_firsts():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    pc.generate(C2, C2_FULL, 16, fork_at=FORK)
    pc.generate(C1_NEXT, C1_NEXT_FULL, 16, fork_at=FORK)
    assert pc.stats()["hits"] == 2
    assert h.decoded[-1] == C1_NEXT_FULL[len(C1) :]  # C1's own slot, not the fork


def test_fork_at_zero_never_forks():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16)  # the default
    assert [s.kind for s in pc.slots()] == ["turn"]
    assert h.prefilled == []  # the trimmable path fused as before


def test_fork_at_past_the_render_is_ignored():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=len(C1) + 5)
    assert [s.kind for s in pc.slots()] == ["turn"]


def test_a_cold_retry_still_forks():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    # Consume the fork so the next turn's warm attempt has reuse == fork_at
    # (no fork wanted), then fail it: the cold retry prefills from 0 and the
    # header is on its way past the boundary again — but a fork with those
    # ids already exists, so none is added.
    h.fail_once = True
    with pytest.warns(UserWarning, match="retrying cold"):
        pc.generate(C2, C2_FULL, 16, fork_at=FORK)
    assert pc.stats()["forks"] == 1
    assert pc.stats()["cold_retries"] == 1


def test_a_fork_is_not_taken_when_the_fork_exists_but_this_turn_missed_it():
    """Owner filter: a fork owned by another thread is invisible, so this
    thread forks its own. The two coexist under different owners."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    _run_on_thread(lambda: pc.generate(C1, C1_FULL, 16, fork_at=FORK))
    # that thread is dead: swept. Plant instead, on a live helper thread.
    stop = threading.Event()
    ready = threading.Event()

    def holder():
        pc.generate(C1, C1_FULL, 16, fork_at=FORK)
        ready.set()
        stop.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    ready.wait(5)
    pc.generate(C2, C2_FULL, 16, fork_at=FORK)  # main thread: a miss, forks its own
    assert pc.stats(owner=threading.current_thread())["forks"] == 1
    assert pc.stats(owner=t)["forks"] == 1
    stop.set()
    t.join(5)


# ---- budget ------------------------------------------------------------------


def _bytes_of(pc: PrefixCache, held) -> int:
    return next(s.nbytes for s in pc.slots() if s.held == held)


def test_the_byte_budget_evicts_least_recently_used_first():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    one = _bytes_of(pc, A1)
    # Room for exactly two turn slots of this size, then a third arrives.
    pc.max_bytes = 2 * one + 1
    pc.generate([20, 21, 22, 23], [20, 21, 22, 23, 90, 91], 16)
    assert [s.held for s in pc.slots()] == [B1, [20, 21, 22, 23]]
    assert pc.stats()["evictions"] == 1


def test_a_hit_refreshes_the_slots_recency():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)  # A is now the most recent
    pc.max_bytes = _bytes_of(pc, A2) + _bytes_of(pc, B1) - 1  # room for one
    pc.generate([20, 21], [20, 21, 90, 91], 16)
    assert B1 not in [s.held for s in pc.slots()]
    assert A2 in [s.held for s in pc.slots()]


def test_the_slot_just_published_is_never_evicted_by_its_own_publish():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=1)  # nothing fits
    pc.generate(A1, A1_FULL, 16)
    assert [s.held for s in pc.slots()] == [A1]
    pc.generate(A2, A2_FULL, 16)
    assert pc.stats()["hits"] == 1  # the protected slot was there to be reused


def test_a_fork_is_charged_and_can_be_evicted_like_any_slot():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=FORK)
    fork_bytes = _bytes_of(pc, H)
    assert fork_bytes == FORK * 8 * 2  # two trimmable layers at the boundary
    assert pc.stats()["resident_bytes"] == fork_bytes + _bytes_of(pc, C1)
    pc.max_bytes = 0
    pc.generate(C2, C2_FULL, 16, fork_at=FORK)  # hits the fork, then evicts it
    assert [s.kind for s in pc.slots()] == ["turn"]


def test_pressure_evicts_until_headroom_covers_the_reserve():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY, reserve_bytes=1000)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    readings = iter([100, 100, 5000])  # below reserve until two slots go

    h.headroom = lambda: next(readings)  # ty: ignore[invalid-assignment]
    pc.generate([20, 21], [20, 21, 90, 91], 16)
    assert [s.held for s in pc.slots()] == [[20, 21]]  # both older slots gone
    assert pc.stats()["evictions"] == 2


def test_pressure_never_evicts_the_slot_just_published():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY, reserve_bytes=1000)
    h.headroom_value = 0  # permanently short
    pc.generate(A1, A1_FULL, 16)
    assert [s.held for s in pc.slots()] == [A1]


def test_unknown_headroom_skips_the_pressure_check():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY, reserve_bytes=1 << 60)
    h.headroom_value = None
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    assert pc.stats()["slots"] == 2
