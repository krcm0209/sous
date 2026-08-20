# Prompt Cache Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry one KV cache across the turns of a single sous task, anchored to a prompt boundary that survives from turn to turn, so each turn prefills only the tokens the conversation gained.

**Architecture:** A new mlx-free module holds the rules and the per-turn orchestrator. The orchestrator drives four hooks that each engine implements, so `LMEngine` and `VLMEngine` share one algorithm. The cache is always left at the *stable render* boundary (`add_generation_prompt=False`), which is a proven strict prefix of the next turn's stable render. Restoring to that boundary uses `trim` where a layer can rewind, and a state copy where it cannot — the layers that cannot rewind are the recurrent ones, whose state does not grow with sequence length.

**Tech Stack:** Python 3.14, mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.13, pytest, ruff, ty, uv.

**Spec:** `docs/superpowers/specs/2026-08-19-prompt-cache-reuse-design.md`

## Global Constraints

- Python `>=3.14`. `except A, B:` without parentheses is valid 3.14 syntax (PEP 758), not a bug.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- `mlx`, `mlx_lm`, and `mlx_vlm` imports stay **function-local**. The lint CI job runs on ubuntu where they are absent. Never hoist them to module level.
- `src/sous/engine/promptcache.py` must contain **no mlx import at all**, not even function-local. Array operations arrive through an injected `copy_array` hook.
- Any thread that touches mlx MUST call `engine.base.release_mlx_thread_state()` before it exits (ml-explore/mlx#4327). This plan adds no new threads.
- Tests must never touch the real `~/.sous`. Always pass `tmp_path`-based `config_path` / `data_dir`.
- `docs/superpowers/**` are point-in-time records. Do not edit the spec while executing this plan.
- Conventional Commits, imperative lowercase subject, *why* in the body.
- `main` is protected. Work on branch `feat/prompt-cache-reuse`; open a PR.
- All four CI jobs must be green: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`.
- Comments explain non-obvious *why*. Never restate what the code does.

## File structure

| File | Responsibility |
|---|---|
| **Create** `src/sous/engine/promptcache.py` | The rules (`reuse_length`, `all_trimmable`, `snapshot`, `restore`, `trim_to`), the counters (`PromptCacheStats`), the two-slot `PromptMemo`, and the `PrefixCache` orchestrator with its `CacheHooks` protocol. No mlx. |
| **Create** `tests/test_promptcache.py` | Every rule and the whole orchestrator, against fake cache layers and fake hooks. Runs in CI. |
| **Modify** `src/sous/engine/base.py` | Two protocol methods; `ManagedEngine` forwarding; `prompt_cache` through `_default_factory` and `EngineManager`. |
| **Modify** `src/sous/engine/lm.py` | Implement the four hooks over mlx-lm; hold a `PrefixCache`. |
| **Modify** `src/sous/engine/vlm.py` | Implement the four hooks over mlx-vlm; hold a `PrefixCache`. |
| **Modify** `src/sous/config.py` | `[model] prompt_cache`. |
| **Modify** `src/sous/worker.py` | Reset at task start and in a `finally`; count elisions; put the stats block in the report and in `_failure_extra`. |
| **Modify** `src/sous/context.py` | Module docstring: the KV cache is no longer transient per generation. |
| **Modify** `tests/fake_engine.py` | Two new methods, so `FakeEngine` still satisfies `Engine`. |
| **Modify** `tests/test_engine_base.py`, `tests/test_worker.py`, `tests/test_config.py` | Cover the new behaviour. |
| **Modify** `tests/test_engine_lm.py`, `tests/test_engine_vlm.py` | `model`-marked bit-exactness tests. |
| **Modify** `scripts/e2e_smoke.py` | A second turn, and print the stats block. |
| **Modify** `README.md` | `[model] prompt_cache`, and that reuse pays most in `auto` context mode. |

---

### Task 1: The rules and the counters

**Files:**
- Create: `src/sous/engine/promptcache.py`
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `reuse_length(cached_ids: Sequence[int], new_ids: Sequence[int]) -> int`; `all_trimmable(cache: Sequence[object]) -> bool`; `snapshot(cache, copy_array) -> tuple[list, int]`; `restore(cache, snap, copy_array) -> None`; `trim_to(cache, n_tokens: int) -> None`; `PromptCacheStats` dataclass with fields `hits, misses, reused_tokens, snapshot_bytes, cold_retries` and method `as_dict() -> dict`; `PromptMemo` with `get(key: str, text: str) -> list[int] | None`, `put(key: str, text: str, ids: list[int]) -> None`, `clear() -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_promptcache.py
"""The rules that decide prompt-cache reuse, exercised without mlx.

The engine tests are model-marked and never run in CI, so every decision that
affects correctness is made here, against fakes.
"""

import pytest

from sous.engine.promptcache import (
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
    cache[0].offset = 163                          # contaminate
    cache[1].state = [FakeArray(99), FakeArray(98)]
    restore(cache, snap, copy_array)
    assert cache[0].offset == 100
    assert cache[1].state == [FakeArray(1), None]


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sous.engine.promptcache'`

- [ ] **Step 3: Write the module**

```python
# src/sous/engine/promptcache.py
"""Cross-turn prefix cache: the rules, the counters, and the orchestrator.

This module imports no mlx, deliberately. The engine tests are model-marked and
never run in CI, so every decision that affects correctness is made here, where
a fake cache layer can exercise it. Array copies arrive through an injected
`copy_array` callable rather than an mx import.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


def reuse_length(cached_ids: Sequence[int], new_ids: Sequence[int]) -> int:
    """How many leading tokens of `new_ids` the cache already holds.

    Strict prefix only: the whole cache, or nothing. sous never rewinds a cache
    to a shorter prefix, because a hybrid model's linear-attention layers hold a
    recurrent state with no inverse. `len(new_ids) > len(cached_ids)` is
    required rather than incidental: mlx rejects an empty prompt, so an exact
    match must count as a miss.
    """
    if not cached_ids or len(new_ids) <= len(cached_ids):
        return 0
    for a, b in zip(cached_ids, new_ids):
        if a != b:
            return 0
    return len(cached_ids)


def all_trimmable(cache: Sequence[object]) -> bool:
    """Whether every layer can rewind, so the cache needs no state copy.

    Asks `is_trimmable()` rather than testing for a `trim` attribute:
    RotatingKVCache owns a `trim` but reports False once its window has
    wrapped, and trimming it then would desync its ring index. Re-asked every
    turn for the same reason — the answer can change mid-task.
    """
    if not cache:
        return False
    return all(getattr(c, "is_trimmable", lambda: False)() for c in cache)


def snapshot(cache: Sequence[object], copy_array: Callable) -> tuple[list, int]:
    """Record what it takes to put `cache` back exactly as it is now.

    A layer that can rewind needs only its offset — an integer. A layer that
    cannot needs its state copied, and those are precisely the recurrent layers
    whose state does not grow with sequence length. That asymmetry is why a
    hybrid cache forks for a fixed cost instead of a copy of the whole KV.

    Returns the snapshot and the bytes copied, for the task report.
    """
    out: list = []
    nbytes = 0
    for c in cache:
        if getattr(c, "is_trimmable", lambda: False)():
            out.append(("trim", int(getattr(c, "offset", 0) or 0)))
            continue
        copies = [None if a is None else copy_array(a) for a in c.state]
        nbytes += sum(getattr(a, "nbytes", 0) for a in copies if a is not None)
        out.append(("state", copies))
    return out, nbytes


def restore(cache: Sequence[object], snap: Sequence[tuple], copy_array: Callable) -> None:
    """Put `cache` back to where `snapshot` was taken."""
    for c, (kind, value) in zip(cache, snap):
        if kind == "trim":
            current = int(getattr(c, "offset", 0) or 0)
            if current > value:
                c.trim(current - value)
            continue
        c.state = [None if a is None else copy_array(a) for a in value]


def trim_to(cache: Sequence[object], n_tokens: int) -> None:
    """Rewind every layer to hold exactly `n_tokens`. All layers must rewind."""
    for c in cache:
        current = int(getattr(c, "offset", 0) or 0)
        if current > n_tokens:
            c.trim(current - n_tokens)


@dataclass
class PromptCacheStats:
    hits: int = 0
    misses: int = 0
    reused_tokens: int = 0
    snapshot_bytes: int = 0
    cold_retries: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "snapshot_bytes": self.snapshot_bytes,
            "cold_retries": self.cold_retries,
        }


_MEMO_SLOTS = ("stable", "full")


class PromptMemo:
    """One slot per render, keyed by the exact prompt text.

    Each turn needs both the stable render and the full prompt, and
    `count_tokens` asks for one of them before `generate` asks again. Keying on
    text equality rather than on call order means a stale slot can never yield
    the wrong ids, even if an abandoned generation thread overwrites it.
    """

    def __init__(self) -> None:
        self._slots: dict[str, tuple[str, list[int]] | None] = dict.fromkeys(_MEMO_SLOTS)

    def get(self, slot: str, text: str) -> list[int] | None:
        entry = self._slots[slot]
        if entry is not None and entry[0] == text:
            return entry[1]
        return None

    def put(self, slot: str, text: str, ids: list[int]) -> None:
        if slot not in self._slots:
            raise KeyError(slot)
        self._slots[slot] = (text, list(ids))

    def clear(self) -> None:
        self._slots = dict.fromkeys(_MEMO_SLOTS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py -v`
Expected: PASS.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat: prompt-cache rules, counters, and prompt memo

The rule that decides reuse and the rule that restores a cache both live
in an mlx-free module. The engine tests are model-marked and never run in
CI, so putting them here is what makes them testable at all.

snapshot() branches on is_trimmable() rather than on a trim attribute:
RotatingKVCache has trim but reports False once its window wraps, and
trimming it then desyncs its ring index."
```

---

### Task 2: The per-turn orchestrator

**Files:**
- Modify: `src/sous/engine/promptcache.py` (append)
- Test: `tests/test_promptcache.py` (append)

**Interfaces:**
- Consumes: `reuse_length`, `all_trimmable`, `snapshot`, `restore`, `trim_to`, `PromptCacheStats` from Task 1.
- Produces: `CacheHooks` protocol with `new_cache() -> list`, `prefill(cache: list, token_ids: list[int]) -> None`, `decode(cache: list, token_ids: list[int], max_tokens: int) -> str`, `copy_array(a: object) -> object`. And `PrefixCache(hooks: CacheHooks, enabled: bool = True)` with `generate(stable_ids: list[int], full_ids: list[int], max_tokens: int) -> str`, `reset() -> None`, `stats() -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_promptcache.py  (append)

from sous.engine.promptcache import PrefixCache


class FakeHooks:
    """A whole engine, minus mlx. Records what the orchestrator asked for."""

    def __init__(self, trimmable: bool, layers: int = 2, fail_once: bool = False):
        self.trimmable = trimmable
        self.layers = layers
        self.fail_once = fail_once
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
    assert h.prefilled == []          # the trimmable path never prefills separately
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
    assert h.decoded[1] == FULL_2[len(STABLE_1):]
    assert len(h.caches) == 1          # the same cache object, not a rebuild
    assert pc.stats()["hits"] == 1
    assert pc.stats()["reused_tokens"] == len(STABLE_1)


def test_non_trimmable_second_turn_prefills_the_delta_then_the_suffix():
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    assert h.prefilled[1] == STABLE_2[len(STABLE_1):]
    assert h.decoded[1] == FULL_2[len(STABLE_2):]
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
    assert len(h.caches) == 2          # an elision rewrote history; start over
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
    assert h.decoded[-1] == FULL_2      # the retry fed the whole prompt


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
    assert len(h.caches) == built       # the retry's cache carried forward


def test_a_propagating_failure_leaves_no_cache_behind():
    """A raise mid-stream leaves the cache holding tokens the token record does
    not describe, so the carrier must already be invalid when it propagates."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)          # a warm cache is now held
    ok_decode = h.decode

    def always_fail(cache, token_ids, max_tokens):
        raise RuntimeError("boom")

    h.decode = always_fail
    with pytest.warns(UserWarning), pytest.raises(RuntimeError):
        pc.generate(STABLE_2, FULL_2, 16)      # warm fails, then the cold retry does
    h.decode = ok_decode
    built = len(h.caches)
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == built + 1          # cold again: nothing was carried


def test_a_late_write_back_from_an_abandoned_generation_is_dropped():
    """reset() bumps an epoch, so a stalled thread landing afterwards drops
    itself instead of resurrecting a finished task's cache under the next one."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    seen = {}

    def slow_decode(cache, token_ids, max_tokens):
        pc.reset()                      # the task ends while this "thread" runs
        seen["reset"] = True
        h.decoded.append(list(token_ids))
        return "text"

    h.decode = slow_decode
    pc.generate(STABLE_1, FULL_1, 16)
    assert seen["reset"]
    pc.generate(STABLE_2, FULL_2, 16)
    assert len(h.caches) == 2           # nothing was carried across the reset
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k PrefixCache or trimmable or divergence -v`
Expected: FAIL — `ImportError: cannot import name 'PrefixCache'`

- [ ] **Step 3: Append the orchestrator**

```python
# src/sous/engine/promptcache.py  (append)

import warnings
from typing import Protocol


class CacheHooks(Protocol):
    """The four things only an engine can do. Everything else is shared."""

    def new_cache(self) -> list: ...
    def prefill(self, cache: list, token_ids: list[int]) -> None: ...
    def decode(self, cache: list, token_ids: list[int], max_tokens: int) -> str: ...
    def copy_array(self, a: object) -> object: ...


class PrefixCache:
    """One KV cache per task, always left at the stable-render boundary.

    The boundary matters: the stable render (`add_generation_prompt=False`) is a
    strict prefix of the next turn's stable render, while the full prompt never
    is — the chat template appends a generation-only block that it strips when
    it later re-renders that same assistant turn as history. Anchoring here is
    what makes reuse possible at all.
    """

    def __init__(self, hooks: CacheHooks, enabled: bool = True):
        self._hooks = hooks
        self.enabled = enabled
        self._cache: list | None = None
        self._held: list[int] = []
        self._epoch = 0
        self._stats = PromptCacheStats()

    def reset(self) -> None:
        """Drop the cache, the token record, and the counters.

        Deliberately takes no lock. ManagedEngine's generation lock is still
        held by an abandoned stalled generation, so a reset that waited for it
        would wedge the next task. The epoch bump is what makes that safe.
        """
        self._epoch += 1
        self._cache = None
        self._held = []
        self._stats = PromptCacheStats()

    def stats(self) -> dict:
        return self._stats.as_dict()

    def generate(self, stable_ids: list[int], full_ids: list[int], max_tokens: int) -> str:
        hooks = self._hooks
        if not self.enabled:
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens)

        epoch = self._epoch
        cache, held = self._cache, self._held
        # Invalid until a generation completes. The cache is mutated in place,
        # so a raise mid-stream leaves it holding tokens `held` does not
        # describe, and reusing it then would duplicate them.
        self._cache, self._held = None, []

        reuse = reuse_length(held, stable_ids) if cache is not None else 0
        if reuse:
            self._stats.hits += 1
            self._stats.reused_tokens += reuse
        else:
            self._stats.misses += 1
            cache, reuse = hooks.new_cache(), 0

        try:
            text = self._run(cache, stable_ids, full_ids, reuse, max_tokens)
        except Exception as e:
            if reuse == 0:
                raise
            # An optimization bug must never fail a task; decide_context sets
            # the same rule for auto sizing. Only a warm attempt is retried, so
            # a genuine engine error still surfaces at once.
            self._stats.cold_retries += 1
            warnings.warn(
                f"sous prompt cache: warm generation failed ({e}); retrying cold",
                stacklevel=2,
            )
            cache = hooks.new_cache()
            text = self._run(cache, stable_ids, full_ids, 0, max_tokens)

        if epoch == self._epoch:
            self._cache, self._held = cache, list(stable_ids)
        return text

    def _run(
        self,
        cache: list,
        stable_ids: list[int],
        full_ids: list[int],
        reuse: int,
        max_tokens: int,
    ) -> str:
        hooks = self._hooks
        anchor = len(stable_ids)
        if all_trimmable(cache):
            # Everything rewinds, so prefill and decode fuse into one pass and
            # the generation block plus the generated tokens are simply trimmed
            # back off afterwards.
            text = hooks.decode(cache, list(full_ids[reuse:]), max_tokens)
            trim_to(cache, anchor)
            return text
        # A recurrent layer cannot rewind, so stop at the anchor, record it,
        # and put the cache back there once the generation is done.
        hooks.prefill(cache, list(stable_ids[reuse:]))
        snap, nbytes = snapshot(cache, hooks.copy_array)
        self._stats.snapshot_bytes = nbytes
        text = hooks.decode(cache, list(full_ids[anchor:]), max_tokens)
        restore(cache, snap, hooks.copy_array)
        return text
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py -v`
Expected: PASS — all tests.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat: per-turn prefix-cache orchestrator behind four engine hooks

Both engines run the same algorithm; only new_cache/prefill/decode/
copy_array differ. The path splits on all_trimmable() rather than by
backend, because the property is architectural: a text-only model loaded
through mlx-lm rewinds, and a hybrid loaded through either does not.

The cache is always left at the stable-render anchor, so no snapshot is
held between turns."
```

---

### Task 3: Engine protocol, ManagedEngine, and the test fake

**Files:**
- Modify: `src/sous/engine/base.py:14-23` (protocol), `:86-105` (ManagedEngine)
- Modify: `tests/fake_engine.py`
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `Engine.reset_prompt_cache() -> None` and `Engine.prompt_cache_stats() -> dict`, forwarded by `ManagedEngine` and implemented by `FakeEngine`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_base.py  (append)

def test_managed_engine_forwards_reset_prompt_cache():
    inner = FakeEngine([])
    managed = ManagedEngine(inner)
    managed.reset_prompt_cache()
    assert inner.resets == 1


def test_managed_engine_forwards_prompt_cache_stats():
    inner = FakeEngine([])
    inner.stats = {"hits": 3}
    assert ManagedEngine(inner).prompt_cache_stats() == {"hits": 3}


def test_reset_prompt_cache_does_not_wait_for_the_generation_lock():
    """A stalled generation is abandoned while still holding _gen_lock. A reset
    that waited for it would wedge the next task, so it must be lock-free."""
    import threading
    import time

    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            started.set()
            release.wait(5)
            return "done"

    managed = ManagedEngine(BlockingEngine(["x"]))
    t = threading.Thread(target=lambda: managed.generate([], [], 8), daemon=True)
    t.start()
    assert started.wait(5)
    assert managed.generation_in_flight()
    t0 = time.monotonic()
    managed.reset_prompt_cache()          # must not block behind the lock
    assert time.monotonic() - t0 < 1.0
    release.set()
    t.join(5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k prompt_cache -v`
Expected: FAIL — `AttributeError: 'ManagedEngine' object has no attribute 'reset_prompt_cache'`

- [ ] **Step 3: Extend the protocol, the wrapper, and the fake**

In `src/sous/engine/base.py`, add to the `Engine` protocol after `count_tokens`:

```python
    def reset_prompt_cache(self) -> None: ...
    def prompt_cache_stats(self) -> dict: ...
```

Add to `ManagedEngine`, after `count_tokens`:

```python
    def reset_prompt_cache(self) -> None:
        # No _gen_lock, on purpose. An abandoned stalled generation still holds
        # it, and run_task calls this in a finally — waiting there would wedge
        # the next task. PrefixCache's epoch guard is what makes a lock-free
        # reset safe against that thread's late write-back.
        self._inner.reset_prompt_cache()

    def prompt_cache_stats(self) -> dict:
        return self._inner.prompt_cache_stats()
```

In `tests/fake_engine.py`, add to `__init__`: `self.resets = 0` and `self.stats: dict = {}`. Then add:

```python
    def reset_prompt_cache(self) -> None:
        self.resets += 1

    def prompt_cache_stats(self) -> dict:
        return dict(self.stats)
```

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `uv run pytest -m "not model" -q`
Expected: PASS.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass. `ty` checks the tests, so `FakeEngine` must satisfy `Engine` structurally.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/base.py tests/fake_engine.py tests/test_engine_base.py
git commit -m "feat: reset_prompt_cache and prompt_cache_stats on the Engine protocol

reset_prompt_cache deliberately skips ManagedEngine's generation lock. An
abandoned stalled generation holds that lock for as long as it runs, and
run_task resets in a finally, so taking it there would wedge the next
task behind a wedged one."
```

---

### Task 4: The `[model] prompt_cache` flag

**Files:**
- Modify: `src/sous/config.py:44` (`_KNOWN`), `:64-65` (fields), `:170-171` (`from_toml`)
- Modify: `src/sous/engine/base.py:63-73` (`_default_factory`), `:109-118` (`EngineManager.__init__`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SousConfig.prompt_cache: bool`; `_default_factory(model_id, temperature, top_p, top_k, prompt_cache)`; both engine constructors accept `prompt_cache: bool = True`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py` already imports `load_config` from `sous.config`; add
`SousConfig` to that same import list, since these tests need the dataclass
default.

```python
# tests/test_config.py  (append)

def test_prompt_cache_defaults_to_true():
    assert SousConfig().prompt_cache is True


def test_prompt_cache_can_be_disabled(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nprompt_cache = false\n")
    assert load_config(path).prompt_cache is False


def test_prompt_cache_is_a_known_model_key(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nprompt_cache = true\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_config(path)
    assert not [w for w in caught if "unknown" in str(w.message).lower()]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k prompt_cache -v`
Expected: FAIL — `AttributeError: 'SousConfig' object has no attribute 'prompt_cache'`

- [ ] **Step 3: Thread the flag through**

`src/sous/config.py:44` — add the key:

```python
    "model": {
        "id",
        "idle_unload_minutes",
        "max_context_tokens",
        "temperature",
        "top_p",
        "top_k",
        "prompt_cache",
    },
```

`src/sous/config.py` — after `top_k: int = 20`:

```python
    # Reuse one KV cache across the turns of a task, prefilling only what the
    # conversation gained. false restores per-turn prefill without a downgrade.
    prompt_cache: bool = True
```

`src/sous/config.py` — after `top_k=model.get("top_k", 20),`:

```python
        prompt_cache=model.get("prompt_cache", True),
```

`src/sous/engine/base.py` — `_default_factory`:

```python
def _default_factory(
    model_id: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    prompt_cache: bool = True,
) -> Engine:
    backend = select_backend(fetch_model_config(model_id))
    if backend == "vlm":
        from sous.engine.vlm import VLMEngine

        return VLMEngine(
            model_id, temperature=temperature, top_p=top_p, top_k=top_k,
            prompt_cache=prompt_cache,
        )
    from sous.engine.lm import LMEngine

    return LMEngine(
        model_id, temperature=temperature, top_p=top_p, top_k=top_k,
        prompt_cache=prompt_cache,
    )
```

`src/sous/engine/base.py` — the factory lambda in `EngineManager.__init__`:

```python
            lambda model_id: _default_factory(
                model_id,
                config.temperature,
                config.top_p,
                config.top_k,
                config.prompt_cache,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not model" -q`
Expected: PASS. Tasks 5 and 6 add the constructor parameter; until then `_default_factory` would fail at runtime, which no CI test exercises because it needs a real model.

- [ ] **Step 5: Commit**

```bash
git add src/sous/config.py src/sous/engine/base.py tests/test_config.py
git commit -m "feat: add [model] prompt_cache config flag

A kill switch matching the degrade-safely habit of context_mode: a
suspected cache bug is a config edit, not a downgrade."
```

---

### Task 5: LMEngine — the mlx-lm hooks

**Files:**
- Modify: `src/sous/engine/lm.py` (whole file)
- Test: `tests/test_engine_unloaded.py` (the unloaded guards must still hold)

**Interfaces:**
- Consumes: `PrefixCache`, `PromptMemo` from Tasks 1-2; `prompt_cache: bool` from Task 4.
- Produces: `LMEngine.generate/count_tokens/unload/reset_prompt_cache/prompt_cache_stats`, satisfying `Engine`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_unloaded.py  (append)

def test_lm_reset_prompt_cache_works_after_unload():
    """run_task resets in a finally, which can land after an idle unload."""
    engine = _unloaded_lm()
    engine._cache = None
    engine.reset_prompt_cache()          # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0
```

The `_unloaded_lm()` helper builds the engine with `object.__new__`, so it must
set every attribute the new methods touch. Extend the helper:

```python
def _unloaded_lm() -> LMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(LMEngine)
    engine.model_id = "test/model"
    engine._model = None
    engine._tokenizer = None
    engine._memo = PromptMemo()
    engine._cache = PrefixCache(engine, enabled=True)
    return engine
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_engine_unloaded.py -k reset_prompt_cache -v`
Expected: FAIL — `AttributeError: 'LMEngine' object has no attribute 'reset_prompt_cache'`

- [ ] **Step 3: Rewrite `src/sous/engine/lm.py`**

```python
"""Text-only backend via mlx-lm."""

from __future__ import annotations

from sous.engine.promptcache import PrefixCache, PromptMemo


class LMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = True,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = model_id
        # mlx-lm ships no type stubs, so the (model, tokenizer) arity of load()
        # is not visible to the type checker.
        self._model, self._tokenizer = load(model_id)  # ty: ignore[invalid-assignment]
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._cache = PrefixCache(self, enabled=prompt_cache)

    def _loaded(self) -> tuple:
        """The (model, tokenizer) pair, or a clear error if already unloaded.

        EngineManager drops its reference right after unload(), so a call here
        on an unloaded engine should be unreachable — but raising beats the
        bare AttributeError on None that the attributes would otherwise give."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"engine for {self.model_id} has been unloaded")
        return self._model, self._tokenizer

    def _prompt(self, messages: list[dict], tools: list[dict], generation: bool = True) -> str:
        # enable_thinking=False: sous delegates mechanical prep, not reasoning —
        # a "thinking" model must not spend its turn budget on <think> chain-of-
        # thought instead of emitting the tool call. Inert on templates that
        # don't define the variable (e.g. plain non-thinking models).
        _, tokenizer = self._loaded()
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=generation,
            tokenize=False,
            enable_thinking=False,
        )

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> tuple[str, list[int]]:
        """Tokenize a render once per turn. The two slots are the stable render
        and the full prompt; count_tokens and generate both read them."""
        _, tokenizer = self._loaded()
        text = self._prompt(messages, tools, generation=slot == "full")
        cached = self._memo.get(slot, text)
        if cached is not None:
            return text, cached
        ids = list(tokenizer.encode(text))
        self._memo.put(slot, text, ids)
        return text, ids

    # ---- CacheHooks ------------------------------------------------------

    def new_cache(self) -> list:
        from mlx_lm.models.cache import make_prompt_cache

        model, _ = self._loaded()
        return make_prompt_cache(model)

    def prefill(self, cache: list, token_ids: list[int]) -> None:
        # mlx-lm has no prefill-only entry point: stream_generate raises on
        # max_tokens=0 because its `token` local is unbound when the loop never
        # runs. Calling the model directly is what generate_step does anyway,
        # and RoPE offsets come from the cache, so a warm suffix needs nothing
        # extra. Only the non-trimmable path reaches here.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        model, _ = self._loaded()
        if not token_ids:
            return
        model(mx.array(token_ids)[None], cache=cache)
        mx.eval([c.state for c in cache])

    def decode(self, cache: list, token_ids: list[int], max_tokens: int) -> str:
        from mlx_lm import stream_generate

        model, tokenizer = self._loaded()
        chunks: list[str] = []
        for r in stream_generate(
            model,
            tokenizer,
            token_ids,
            max_tokens=max_tokens,
            sampler=self._sampler,
            prompt_cache=cache,
        ):
            chunks.append(r.text)
        return "".join(chunks)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return mx.array(a)

    # ---- Engine ----------------------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        _, stable_ids = self._ids("stable", messages, tools)
        _, full_ids = self._ids("full", messages, tools)
        return self._cache.generate(stable_ids, full_ids, max_tokens)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._ids("full", messages, tools)[1])

    def reset_prompt_cache(self) -> None:
        self._cache.reset()
        self._memo.clear()

    def prompt_cache_stats(self) -> dict:
        return self._cache.stats()

    def unload(self) -> None:
        import gc

        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        self.reset_prompt_cache()
        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not model" -q`
Expected: PASS.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass. `promptcache` at module level is fine — it holds no mlx import.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/lm.py tests/test_engine_unloaded.py
git commit -m "feat: cross-turn prompt cache on the mlx-lm backend

stream_generate replaces generate because only the streaming form takes
pre-tokenized ids, which is what lets the caller feed just the suffix.
prefill calls the model directly: mlx-lm has no prefill-only entry point
and stream_generate raises on max_tokens=0."
```

---

### Task 6: VLMEngine — the mlx-vlm hooks

**Files:**
- Modify: `src/sous/engine/vlm.py` (whole file)
- Test: `tests/test_engine_unloaded.py` (extend the vlm helper the same way)

**Interfaces:**
- Consumes: `PrefixCache`, `PromptMemo`; `prompt_cache: bool`.
- Produces: the same five `Engine` methods as Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_unloaded.py  (append)

def test_vlm_reset_prompt_cache_works_after_unload():
    engine = _unloaded_vlm()
    engine.reset_prompt_cache()          # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0
```

Extend the helper the same way as the LM one:

```python
def _unloaded_vlm() -> VLMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = None
    engine._processor = None
    engine._memo = PromptMemo()
    engine._cache = PrefixCache(engine, enabled=True)
    return engine
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_engine_unloaded.py -k vlm_reset -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Rewrite `src/sous/engine/vlm.py`**

```python
"""Multimodal backend via mlx-vlm, used text-only in v1."""

from __future__ import annotations

from sous.engine.promptcache import PrefixCache, PromptMemo


class VLMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = True,
    ):
        from mlx_vlm import load
        from mlx_vlm.sample_utils import make_sampler

        self.model_id = model_id
        self._model, self._processor = load(model_id)
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._cache = PrefixCache(self, enabled=prompt_cache)

    def _loaded(self) -> tuple:
        """The (model, processor) pair, or a clear error if already unloaded.
        Same rationale as LMEngine._loaded."""
        if self._model is None or self._processor is None:
            raise RuntimeError(f"engine for {self.model_id} has been unloaded")
        return self._model, self._processor

    @property
    def _tokenizer(self):
        # Via _loaded() so the _prompt/count_tokens paths raise the same named
        # RuntimeError after unload() — reading _processor directly would hand
        # back None and fail later with a bare AttributeError.
        _, processor = self._loaded()
        return getattr(processor, "tokenizer", processor)

    def _prompt(self, messages: list[dict], tools: list[dict], generation: bool = True) -> str:
        # enable_thinking=False: same rationale as LMEngine. Confirmed inert
        # (no-op) for templates such as Qwen2-VL's that don't define the
        # variable — verified empirically, not assumed.
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=generation,
            tokenize=False,
            enable_thinking=False,
        )

    def _encode(self, text: str) -> list[int]:
        from mlx_vlm.utils import should_add_special_tokens

        model, processor = self._loaded()
        # Parity with prepare_inputs' text-only path, which is what mlx-vlm
        # itself would tokenize this prompt with. A mismatch would not fail —
        # it would silently miss on every turn.
        add_special = should_add_special_tokens(model.config.model_type, processor)
        return list(self._tokenizer.encode(text, add_special_tokens=add_special))

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> tuple[str, list[int]]:
        text = self._prompt(messages, tools, generation=slot == "full")
        cached = self._memo.get(slot, text)
        if cached is not None:
            return text, cached
        ids = self._encode(text)
        self._memo.put(slot, text, ids)
        return text, ids

    # ---- CacheHooks ------------------------------------------------------

    def new_cache(self) -> list:
        from mlx_vlm.models.cache import make_prompt_cache

        model, _ = self._loaded()
        # The nested language model, never the wrapper: many wrappers expose a
        # `layers` property but no `make_cache`, so passing the wrapper builds
        # all-plain KVCache and loses the model's real cache layout.
        return make_prompt_cache(model.language_model)

    def prefill(self, cache: list, token_ids: list[int]) -> None:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from mlx_vlm import generate

        model, processor = self._loaded()
        if not token_ids:
            return
        # max_tokens=0 is prefill-only: dispatch has an explicit
        # `if not generated_tokens:` branch that yields a result and returns
        # without touching any cache state.
        generate(
            model,
            processor,
            "",
            max_tokens=0,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
        )

    def decode(self, cache: list, token_ids: list[int], max_tokens: int) -> str:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from mlx_vlm import generate

        model, processor = self._loaded()
        # prompt_cache plus input_ids, not prompt_cache_state. mlx-vlm primes
        # Qwen mRoPE state before feeding a suffix, and that priming turns out
        # to be bit-identical to no priming for text-only prompts — so sous
        # owns the cache outright rather than driving mlx-vlm's reuse path.
        result = generate(
            model,
            processor,
            "",
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
        )
        # mlx-vlm returns GenerationResult in recent versions; older return str
        return result.text if hasattr(result, "text") else str(result)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return mx.array(a)

    # ---- Engine ----------------------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        _, stable_ids = self._ids("stable", messages, tools)
        _, full_ids = self._ids("full", messages, tools)
        return self._cache.generate(stable_ids, full_ids, max_tokens)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._ids("full", messages, tools)[1])

    def reset_prompt_cache(self) -> None:
        self._cache.reset()
        self._memo.clear()

    def prompt_cache_stats(self) -> dict:
        return self._cache.stats()

    def unload(self) -> None:
        import gc

        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        self.reset_prompt_cache()
        self._model = None
        self._processor = None
        gc.collect()
        mx.clear_cache()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not model" -q`
Expected: PASS.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/vlm.py tests/test_engine_unloaded.py
git commit -m "feat: cross-turn prompt cache on the mlx-vlm backend

Passes prompt_cache plus input_ids rather than prompt_cache_state.
mlx-vlm primes Qwen mRoPE state before feeding a suffix, and measuring
primed against unprimed on Qwen3.5-9B gave bit-identical caches for
text-only prompts (64/64 tensors, two split points). So sous owns the
cache directly and depends on none of mlx-vlm's reuse bookkeeping.

_encode reproduces prepare_inputs' text-only tokenization via the public
should_add_special_tokens, because a mismatch there would not fail — it
would silently miss on every turn."
```

---

### Task 7: worker.py — lifetime, elision count, and the report

**Files:**
- Modify: `src/sous/worker.py:121-137` (`_elide_if_needed`), `:179-186` (`_failure_extra`), `:197-392` (`run_task`)
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `Engine.reset_prompt_cache()` and `Engine.prompt_cache_stats()` from Task 3.
- Produces: `_elide_if_needed(...) -> tuple[int, int]` returning `(token_count, elisions)`; a `prompt_cache` block in the task report and in `_failure_extra`.

- [ ] **Step 1: Write the failing tests**

`tests/test_worker.py` already provides an `env` fixture returning
`(root, cfg, store)`, a `_start(store, root, verify=(), context=())` helper that
enqueues and claims a task, and the `CALL` / `FINISH` constants. Use those; do
not add a second set.

```python
# tests/test_worker.py  (append)

def test_run_task_resets_the_prompt_cache_at_start_and_end(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([FINISH])
    run_task(task, store, engine, cfg)
    assert engine.resets == 2                    # once at entry, once in the finally


def test_run_task_resets_even_when_the_task_fails(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(["<tool_call>{bad json}</tool_call>"] * 3)
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.FAILED
    assert engine.resets == 2


def test_report_carries_the_prompt_cache_block(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([FINISH])
    engine.stats = {"hits": 4, "misses": 1, "reused_tokens": 900,
                    "snapshot_bytes": 0, "cold_retries": 0}
    run_task(task, store, engine, cfg)
    block = store.get(task.id).report["prompt_cache"]
    assert block["hits"] == 4
    assert block["elisions"] == 0


def test_failure_extra_carries_the_prompt_cache_block(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(["<tool_call>{bad json}</tool_call>"] * 3)
    engine.stats = {"hits": 1, "misses": 1, "reused_tokens": 10,
                    "snapshot_bytes": 0, "cold_retries": 0}
    run_task(task, store, engine, cfg)
    assert store.get(task.id).report["prompt_cache"]["hits"] == 1


def test_elisions_are_counted_in_the_report(env):
    """A small window forces _elide_if_needed to rewrite old tool results,
    which is the only thing that can break the prefix the cache is anchored to.
    The count is the report's answer to why reuse missed."""
    root, cfg, store = env
    (root / "big.txt").write_text("word " * 4000)
    cfg = dataclasses.replace(cfg, max_context_tokens=1500)
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="read_file", args='{"path": "big.txt"}'),
            CALL.format(name="read_file", args='{"path": "hello.py"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    assert store.get(task.id).report["prompt_cache"]["elisions"] >= 1


def test_elide_if_needed_reports_how_many_messages_it_rewrote():
    from sous.worker import _elide_if_needed

    engine = FakeEngine([])
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "<tool_result>" + "x" * 400 + "</tool_result>"},
        {"role": "user", "content": "<tool_result>" + "y" * 400 + "</tool_result>"},
    ]
    count, elisions = _elide_if_needed(messages, engine, 40)
    assert elisions >= 1
    assert isinstance(count, int)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_worker.py -k "prompt_cache or elisions or elide_if_needed" -v`
Expected: FAIL — `assert engine.resets == 2` gets 0; `_elide_if_needed` returns an int, not a tuple.

- [ ] **Step 3: Change `worker.py`**

Replace `_elide_if_needed` so it reports how many messages it rewrote:

```python
def _elide_if_needed(
    messages: list[dict], engine: Engine, max_context_tokens: int
) -> tuple[int, int]:
    """Elide old tool results until under the context cap. Returns the final
    token count and how many messages were rewritten — the caller must check
    the count against the cap, because when nothing elidable remains it can
    still be over, and an oversized prompt must never be sent.

    The rewrite count is the report's answer to "why did prompt-cache reuse
    miss": an in-place edit to old history is the only thing that breaks the
    prefix the cache is anchored to."""
    elisions = 0
    while (count := engine.count_tokens(messages, WORKER_TOOLS)) > max_context_tokens:
        for m in messages:
            if (
                m["role"] == "user"
                and m["content"].startswith("<tool_result")
                and "[elided" not in m["content"]
            ):
                m["content"] = "<tool_result>[elided: re-read the file if needed]</tool_result>"
                elisions += 1
                break
        else:
            return count, elisions  # nothing left to elide; still over the cap
    return count, elisions
```

Change `_failure_extra` to take the engine and the running elision count:

```python
def _failure_extra(
    ex: ToolExecutor, transcript: _Transcript, engine: Engine, elisions: int
) -> dict:
    """Attached to every terminal store.fail() in run_task so a failed task
    still tells Claude what changed on disk, where to audit, and what the
    prompt cache did — silent, unreviewed file modifications are the worst
    failure shape here."""
    return {
        "files_changed": [vars(c) for c in ex.changed_files()],
        "transcript_path": str(transcript.path),
        "prompt_cache": {**engine.prompt_cache_stats(), "elisions": elisions},
    }
```

In `run_task`:

1. Introduce `elisions = 0` beside `turns = 0` and `malformed = 0`.
2. Wrap everything after `ex`/`transcript` are built in `try:` / `finally: engine.reset_prompt_cache()`, and call `engine.reset_prompt_cache()` once **before** the `try` so a stale prefix from a previous task can never be reused.
3. Replace the elision call site:

```python
        token_count, elided = _elide_if_needed(messages, engine, context.tokens)
        elisions += elided
```

4. Replace every `_failure_extra(ex, transcript)` call with
   `_failure_extra(ex, transcript, engine, elisions)`. There are **eight** call
   sites, at `worker.py:238, 248, 260, 279, 286, 301, 312, 327` — the loop-top
   cancel check, the two context-overflow exits, the stall path, the
   engine-error path, the two model-confused exits, and the per-tool-call cancel
   check. Grep for `_failure_extra(` and change them all; missing one leaves a
   `TypeError` on a path only a failing task reaches.
5. Add the block to the report, after `"budget"`:

```python
        "prompt_cache": {**engine.prompt_cache_stats(), "elisions": elisions},
```

The report must be built **before** the `finally` runs, which it is: the
`finally` only wraps the loop and the verify phase, and `store.finish` is the
last statement inside the `try`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -m "not model" -q`
Expected: PASS.

- [ ] **Step 5: Check lint and types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/sous/worker.py tests/test_worker.py
git commit -m "feat: task-scoped prompt cache lifetime and reuse reporting

run_task resets at entry so no prefix crosses tasks, and in a finally so
a finished task leaves nothing resident.

The report carries hits, misses, reused tokens and elisions together,
because hits and misses alone say reuse failed while the elision count
says why: an in-place rewrite of old history is the only thing that
breaks the prefix the cache is anchored to."
```

---

### Task 8: Documentation that the change makes wrong

**Files:**
- Modify: `src/sous/context.py:1-8` (module docstring)
- Modify: `README.md` (config table and a note on context mode)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. Prose only.

- [ ] **Step 1: Fix the `context.py` docstring**

The current text says the KV cache "is transient per generation, so the window
is a cap on what a task MAY use, not memory the daemon holds." That is now
wrong. Replace the final sentence of the module docstring with:

```python
"""Sizing the worker's context window from live memory headroom.

Fixed mode serves the configured max_context_tokens unchanged. Auto mode
exists because the right window is a function of the machine's moment: the
same 64 GB Mac supports a 262k window when idle and far less with a browser
and an IDE resident.

The window is a cap on what a task MAY use, not memory the daemon holds. But
since prompt-cache reuse, the KV cache lives for the whole task rather than
for one generation, so `fraction` now bounds sustained residency instead of a
peak — a run_command subprocess competes with a live cache that used to be
freed between turns. Residency still tracks the tokens a task actually uses,
not the window.
"""
```

- [ ] **Step 2: Update the README**

Add `prompt_cache` to the `[model]` section of the config reference, with
`true` as the default and one line: reuse one KV cache across the turns of a
task; `false` restores per-turn prefill.

Add, next to the `[context]` documentation: reuse pays most in `auto` mode.
Elision is the only thing that discards the cache, and elision fires only when
the prompt exceeds the window, so a window the task never reaches means the
cache survives the whole task. The shipped default is `fixed` at 32768 tokens.

- [ ] **Step 3: Verify**

Run: `uv run pytest -m "not model" -q && uv run ruff check . && uv run ruff format --check .`
Expected: PASS. (`docs/` is ruff-excluded; `README.md` and docstrings are not code.)

- [ ] **Step 4: Commit**

```bash
git add src/sous/context.py README.md
git commit -m "docs: the KV cache is no longer transient per generation

context.py's docstring claimed the window was a cap and not held memory.
With cross-turn reuse the cache lives for the whole task, so [context]
fraction now bounds sustained residency rather than a peak.

Also records that reuse pays most in auto mode: elision is the only thing
that discards the cache, and it fires only when the prompt exceeds the
window."
```

---

### Task 9: e2e_smoke — a second turn and the numbers

**Files:**
- Modify: `scripts/e2e_smoke.py`

**Interfaces:**
- Consumes: the `prompt_cache` report block from Task 7.
- Produces: nothing.

- [ ] **Step 1: Make the task multi-turn and print the block**

The current smoke task ("create hello.txt with one line") can finish in one
turn, which measures nothing. Change the instructions so the worker must read
before it writes, guaranteeing at least two turns:

```python
        proj_file = proj / "notes.md"
        proj_file.write_text("# Notes\n\nalpha\nbeta\ngamma\n")
        task = store.enqueue(
            title="smoke",
            instructions=(
                "Read notes.md, then create hello.txt containing exactly this "
                "one line: hello sous"
            ),
            project_root=str(proj),
            context_files=[],
            verify_commands=[],
        )
```

After the existing report print, add:

```python
        cache = (current.report or {}).get("prompt_cache")
        print(f"prompt_cache: {cache}")
```

- [ ] **Step 2: Update the module docstring**

Add: the task is deliberately multi-turn so the `prompt_cache` block is
non-trivial. Run it once with `[model] prompt_cache = true` and once with
`false`, and compare `budget.seconds` — that is the before-and-after issue #27
asks for. Note that the 0.6B model is text-only and therefore fully trimmable,
so it exercises the one-call path, not the snapshot path.

- [ ] **Step 3: Verify**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: PASS. Do not run the script in CI; it downloads a model.

- [ ] **Step 4: Commit**

```bash
git add scripts/e2e_smoke.py
git commit -m "test: make the smoke task multi-turn and print prompt-cache stats

A single-turn task measures nothing about cross-turn reuse. Reading a
file before writing one guarantees at least two turns, so running the
script with the flag on and off is the reproducible before-and-after."
```

---

### Task 10: model-marked bit-exactness tests

**Files:**
- Modify: `tests/test_engine_lm.py`, `tests/test_engine_vlm.py`

**Interfaces:**
- Consumes: `snapshot`, `restore` from Task 1; both engines from Tasks 5-6.
- Produces: nothing.

These are the tests that would have caught the earlier draft's error. They are
`model`-marked, so CI never runs them; run them locally before opening the PR.

- [ ] **Step 1: Write the LM test**

```python
# tests/test_engine_lm.py  (append)

def test_lm_snapshot_restore_is_bit_exact():
    """Restoring a contaminated cache must equal a cold prefill of the same
    prefix, exactly. Anything less and reuse silently changes the context."""
    import mlx.core as mx

    from mlx_lm.models.cache import make_prompt_cache
    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import restore, snapshot

    e = LMEngine(TINY)
    ids = list(e._tokenizer.encode("def f(x):\n    return x + 1\n" * 40))
    prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(e._model)
    e.prefill(ref, prefix)
    work = make_prompt_cache(e._model)
    e.prefill(work, prefix)

    snap, nbytes = snapshot(work, e.copy_array)
    assert nbytes == 0, "a pure-attention cache needs no state copy"
    e.prefill(work, suffix)
    restore(work, snap, e.copy_array)

    for a, b in zip(work, ref):
        off = int(a.offset)
        assert off == int(b.offset)
        for xa, xb in zip(a.state, b.state):
            d = mx.max(mx.abs(xa[..., :off, :].astype(mx.float32)
                              - xb[..., :off, :].astype(mx.float32)))
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
```

- [ ] **Step 2: Write the VLM test**

```python
# tests/test_engine_vlm.py  (append)

def test_vlm_snapshot_restore_is_bit_exact():
    """Same guarantee on the hybrid path, where the snapshot copies recurrent
    state instead of recording offsets."""
    import mlx.core as mx

    from mlx_vlm.models.cache import make_prompt_cache
    from sous.engine.promptcache import restore, snapshot
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(TINY_VLM)
    ids = e._encode("def f(x):\n    return x + 1\n" * 40)
    prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(e._model.language_model)
    e.prefill(ref, prefix)
    work = make_prompt_cache(e._model.language_model)
    e.prefill(work, prefix)

    snap, _ = snapshot(work, e.copy_array)
    e.prefill(work, suffix)
    restore(work, snap, e.copy_array)

    for a, b in zip(work, ref):
        sa, sb = a.state, b.state
        off = int(getattr(a, "offset", 0) or 0)
        for xa, xb in zip(sa, sb):
            if xa is None or xb is None:
                continue
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
```

`TINY_VLM` is `mlx-community/Qwen2-VL-2B-Instruct-4bit`, which is pure
attention — so it exercises the trimmable path through the VLM engine, which is
exactly the case that proves the split is architectural and not per-backend.
The snapshot path needs a hybrid, so also run the same reuse test by hand
against `mlx-community/Qwen3.5-9B-MLX-4bit` before opening the PR.

- [ ] **Step 3: Run them locally**

Run: `uv run pytest -m model -v`
Expected: PASS. Downloads roughly 1.4 GB on first run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_lm.py tests/test_engine_vlm.py
git commit -m "test: bit-exactness of snapshot/restore on both engine paths

These are model-marked and never run in CI, but they are the tests that
would have caught the design error an earlier draft of the spec shipped
with: they assert the restored cache equals a cold prefill of the same
prefix exactly, rather than merely producing plausible output."
```

---

## Verification before the PR

Not tasks, because none of them can run in CI. All three are required by the
spec and must be done on this machine before the PR is opened.

- [ ] **The 27B.** Load `mlx-community/Qwen3.8-27B-mxfp8`, run a two-turn
  delegation, and record `prompt_cache.snapshot_bytes`. The spec projects
  ~147 MiB (48 linear layers x 3.00 MiB). Note the measured value in the PR.
- [ ] **Characterise the incremental-prefill difference.** On the 9B, compare a
  two-call prefill against a one-call prefill of the same tokens, reporting the
  distribution (RMS and percentiles) rather than maxima, and compare the
  next-token distribution on a prompt that is *not* an already-complete render.
  The spec deliberately states no magnitude for this; the PR should.
- [ ] **One real delegated task on the default model.** Confirms the task still
  completes, and that no thread leaks mlx state — CLAUDE.md requires this after
  any change touching mlx, and CI cannot do it.
- [ ] **Suspected-flaky tests get run in a loop, not judged on one pass.**
  `uv run pytest tests/test_promptcache.py -q --count=50` if any test in this
  plan looks non-deterministic (requires `pytest-repeat`; otherwise loop in the
  shell).

## Risks the executor should know

- **The uncoupled VLM decode path is measured for prefill, not for multi-token
  generation.** Priming was proven unnecessary by feeding a 1067-token suffix
  and comparing caches bit-for-bit, and a single-token generation through the
  same path worked. A long generation through `prompt_cache` plus `input_ids`
  has not been run. Task 10's `test_vlm_engine_reuses_across_turns` and the
  real-model verification are what catch it. If output turns to garbage on the
  VLM path, this is the first thing to suspect.
- **`RotatingKVCache` never appears today** because sous never passes
  `max_kv_size`. `all_trimmable` still asks `is_trimmable()` every turn rather
  than caching the answer, because a rotating cache flips to False once its
  window wraps.
- **Two `generate` calls per turn on the hybrid path** both sit inside
  `Engine.generate`, so `_generate_with_timeout` still sees one generation and
  one deadline. A cold retry can push a turn past its deadline, which the worker
  reports as a stall or budget exhaustion — the existing shape for any slow
  generation.
