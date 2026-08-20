"""The rules that decide prompt-cache reuse, exercised without mlx.

The engine tests are model-marked and never run in CI, so every decision that
affects correctness is made here, against fakes.
"""

from __future__ import annotations

from typing import cast

import pytest

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

    def decode(self, cache, token_ids, max_tokens):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("boom")
        if self.decode_impl is not None:
            return self.decode_impl(self, cache, token_ids, max_tokens)
        self.decoded.append(list(token_ids))
        self._advance(cache, len(token_ids) + len(self.generated))
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


def test_disabled_never_reuses_and_never_counts():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, enabled=False)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == 2
    assert pc.stats() == PromptCacheStats().as_dict()


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
    pc.generate(STABLE_2 + [7], FULL_2 + [7], 16)
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
