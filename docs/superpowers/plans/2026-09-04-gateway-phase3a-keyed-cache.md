# Gateway Phase 3a (Keyed Prefix Cache) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prompt cache's single `(cache, held)` slot with a small LRU map of slots, keyed by the token prefix each holds and owned by the thread that built it, so that (a) two conversations interleaving on the local model no longer evict each other, (b) a *new* conversation that shares a long prefix with an earlier one (every Claude Code subagent of one type shares ~50K tokens of system prompt and tool schemas) starts warm from a **fork slot** copied at that boundary instead of prefilling for ~150 s, and (c) resident slot memory is bounded by a byte budget derived from the machine, shrunk under memory pressure, and never charged against the in-flight turn.

**Architecture:** All decisions stay in `src/sous/engine/promptcache.py`, which stays mlx-free and fully unit-tested against fakes. A `Slot` is a cache plus the exact ids it holds, its owner thread, its kind (`turn` — published when a turn ends, *consumed* by the turn that extends it, exactly today's semantics; `fork` — a copy taken at the system-header boundary during a cold prefill, *copied* on every hit and left in place), and its byte size. Lookup is "the longest slot owned by this thread whose ids are a strict prefix of the new stable render"; the existing strict-prefix/epoch/thread-ownership correctness argument applies per slot unchanged. The engines gain one job: rendering the leading system message alone (with tools) to find the header boundary. Stats and reset become owner-scoped (`prompt_cache_stats(owner=…)`, `reset_prompt_cache(owner=…)`) so the worker retires only its own task's slots and the gateway's per-turn hit/miss is exact; a bare `reset()` keeps today's drop-everything meaning for unload and stalls. `[model].prompt_cache_gb` (default `"auto"`) sizes the resident budget: working set minus weights minus one full window's KV minus slack.

**Tech Stack:** Python 3.14, uv, mlx 0.32.2 / mlx-lm / mlx-vlm (cache classes expose `state`, `meta_state`, `nbytes`, `is_trimmable` — probed, see decisions), psutil (already a dependency via `sous.context`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md` — Phase 3a paragraph ("The single `(cache, held)` slot becomes an LRU map keyed by conversation prefix, with a byte budget reserved out of the generation budget (oMLX's `_hot_cache_reserved_bytes` pattern: charge `min(cap, used + slack)`, shrink under pressure, protect the in-flight conversation). Each slot keeps the existing epoch/strict-prefix correctness argument."), the core-decisions row "Prompt cache", and the Phase 3b paragraph's "Ship 3a first; measure serialized latency at N=2–3 before paying for any of this". Evidence motivating the fork slots is the Phase 2 exit report on issue #41 (comment 5547146397, "Cache (Phase 3a evidence)"): run 2's second Explore subagent shared the entire ~57K-token system+tools prefix with the first and still missed; within one conversation the slot already works (50,726 tokens reused, 5.9 s vs 152 s). The Phase 1 plan (`docs/superpowers/plans/2026-09-02-gateway-phase1-endpoint.md`) records why the cache lives on one long-lived gateway session thread; the Phase 2 plan (`docs/superpowers/plans/2026-09-04-gateway-phase2-routing.md`) is the record of the routing this plan does not touch.

## Global Constraints

- Python >= 3.14; everything runs via `uv run`. Never pip. `except A, B:` without parentheses is valid 3.14 syntax (PEP 758) — don't "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`. `uv run ty check` covers tests too.
- **mlx / mlx_lm / mlx_vlm imports stay function-local** (absent on non-macOS; the lint job runs on ubuntu). `promptcache.py` imports no mlx at all — every decision there is exercised by fakes in CI.
- **Any thread that touches mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits** (ml-explore/mlx#4327). Corollary that matters here: **never call it from a thread that is still using its caches** — it destroys that thread's streams, and the KV arrays live on them (#34). The new memory readers in `base.py` therefore do *not* release (unlike `sous.context._live_memory`, which does and must not be reused on a session thread).
- **Cache arrays are usable only from the thread whose streams built them** (#34). A slot is owned by a thread; lookup never crosses owners; a fork copy is made on the hitting thread and belongs to it.
- **sous never rewinds a cache to a shorter prefix** (a hybrid model's linear-attention layers hold a recurrent state with no inverse). Every reuse is "the whole slot, or nothing"; a fork is a copy taken *while* prefilling past the boundary, never a trim afterwards.
- **No doubling:** a consumed turn slot is the only reference to its cache when the turn starts prefilling (today's rule, kept). A fork slot is a deliberate second copy, charged to the budget. The in-flight turn's own cache is never in the map and never counted against the resident budget; the budget is derived with one full window's KV already reserved for it.
- **`PrefixCache` refuses its cold retry once any non-`ReplaySafe` delta has reached the client** — unchanged.
- **Engine `on_delta` callbacks fire on the generation thread inside the decode loop** — never block or raise in one.
- Budget exhaustion is `done` with outcome `budget-exhausted`, never `failed` (worker, untouched here).
- Tests never touch the real `~/.sous`: every `SousConfig` in tests is built with tmp_path-based `data_dir`/`config_path`. `model`-marked tests download weights and are local/manual only; `slow`-marked tests spawn real threads/processes and are run, not skipped.
- `docs/superpowers/**` are point-in-time records: never edit existing spec/plan files (this plan is a new file).
- The gateway's logging rules are unchanged: the new `cache=fork` log token and the new status fields are counts and byte totals only — never a token id, never a prompt.
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>` (model-less, per the user's global CLAUDE.md). Work on a branch (`feat/gateway-phase3a`); `main` is protected. CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. Every commit step runs `uv run ruff format .` *before* `uv run ruff check .`.

---

## Decisions this plan locks (beyond the spec)

Settled by probing the installed libraries (mlx 0.32.2, mlx-lm, mlx-vlm) and the default model's config; implementers should not reopen them.

1. **Two slot kinds, two hit behaviours.** A `turn` slot is what today's single slot is: published when a turn ends, holding the stable render (`stable_ids`), and *consumed* (removed from the map, its cache extended in place) by the turn that strictly extends it — no copy, no doubling. A `fork` slot is a *copy* of the cache taken at a prefix boundary during a cold prefill; a hit *copies it again* into the turn's working cache and leaves the slot in place for the next conversation. A conversation slot could in principle be copied too, but no caller branches a conversation — YAGNI.
2. **The fork boundary is the header: the stable render of the leading system message alone, with tools.** Claude Code's `convert.chat_messages` always emits the merged system prompt as `messages[0]` (`role == "system"`), and Qwen's template renders system text and the `<tools>` block inside that one system turn, so rendering `messages[:1]` with `tools` yields the exact text prefix shared by every subagent of one type. The engine tokenizes it (a third `PromptMemo` slot, `"header"`) and `promptcache.fork_point(header_ids, stable_ids)` accepts it only when it is a strict token prefix of the turn's stable render (`reuse_length`, the same rule as everything else — chat-template agnostic, verified per turn, never assumed) and at least `FORK_MIN_TOKENS = 4096` long: a shorter header prefills in under a second and is not worth a copy of its KV; the worker's ~2K-token system prompt never qualifies, a Claude Code subagent's ~50K one always does. Alternative rejected: learning shared prefixes from traffic (longest common prefix with existing slots) — template-agnostic too, but the *second* subagent would still run cold; the header fork makes it warm.
3. **A fork is taken only on a turn that is prefilling past the boundary itself** (`reuse < fork_at < len(stable_ids)`), and only when no fork slot with exactly those ids already exists for this owner. A turn that hit a fork slot (`reuse == fork_at`) or a turn slot (`reuse > fork_at`) cannot fork (the header is inside a cache that cannot rewind) and does not need to. Consequence: forks are created by cold turns — the first subagent's first turn — and by nothing else.
4. **`fork_copy(src, dst, copy_array)` copies every layer's `state` array-by-array and carries `meta_state` over.** That pair is what mlx's own `_BaseCache.from_state` rebuilds a layer from, so it is complete for every cache class mlx ships (`KVCache`, `ArraysCache`, `RotatingKVCache`, … — all expose `state`/`meta_state`/`nbytes` in both mlx-lm and mlx-vlm, probed). For a `KVCache`, the `state` getter slices `keys`/`values` to the current offset and `mx.array(slice)` materialises exactly that slice (probed: a 57-token slice of a 300-token buffer copies to 57 × row bytes, no 256-step padding), and the setter derives `offset` from the copied shape. Unlike `snapshot()` (which records only an offset for trimmable layers because it will *restore the same object*), a fork is an independent second cache and must copy the attention layers too. Nested-tuple states (`QuantizedKVCache`) are not supported and not used by sous. Fork copies happen only after a prefill of ≥ `FORK_MIN_TOKENS`, so every layer is populated (a `KVCache` with `keys is None` would raise in its `state` getter).
5. **Slot bytes are the sum of each layer's `nbytes`.** For a `KVCache` that is the whole allocated buffer (padding to the 256 step and the trimmed-off generation region included) — the truth about residency, and an upper bound. Fakes in tests expose `nbytes` too.
6. **Budget = `[model].prompt_cache_gb`, default `"auto"` = `max(0, working_set − active_after_load − reserve − 2 GiB)`**, computed once when the engine loads (so `active` is the weights, drafter included) by `promptcache.auto_cache_budget()` from numbers `base.measure_cache_budget()` reads. `reserve = reserve_tokens × kv_bytes_per_token(model_config)` (the existing `sous.context` formula: 65,536 B/token for the default model) with `reserve_tokens = max([model].max_context_tokens, [gateway].max_context_tokens if the gateway is enabled else 0)` — the largest cache one turn can build, kept free for the in-flight turn. That is the spec's "reserved out of the generation budget". On the reference machine (M5 Pro, 64 GB: working set 55.66 GB, weights ≈ 18 GB with the drafter, gateway window 131072 → reserve 8 GiB) the automatic budget is ≈ 27 GiB — about seven 57K-token fork slots or three full-window conversations. When the KV cost is unknown (`kv_bytes_per_token` returns `None`), the budget is 0 with a warning. **A budget of 0 is exactly today's behaviour**: after every publish everything but the just-published slot is evicted, so one slot remains — and it is the `PrefixCache` constructor default, so every existing test keeps its meaning.
7. **Eviction is LRU by last use, at publish time, protecting the slot just published** — that is "protect the in-flight conversation": a turn's own slot is never evicted by its own publish even when it alone exceeds the budget (a 32 GB machine with an 8 GiB window has no room for a second slot, and still keeps one). Two caps: `max_bytes` (the budget) and `MAX_SLOTS = 16` (a sanity bound; dead subagent conversations are what fills the map, and LRU discards them first).
8. **Pressure: after eviction to the static caps, while `hooks.headroom()` is below `reserve_bytes` and an evictable slot remains, evict one more and re-read.** `headroom = max(0, min(working_set − active, available_RAM + mlx_cache))` — `sous.context.auto_context_tokens`'s definition, read live without releasing thread state (`base.live_headroom()`). Dropping the last Python reference to a slot frees its mlx buffers into mlx's allocator cache immediately (`active` falls, `cache` rises), so the re-read sees the eviction without `mx.clear_cache()`. This is the spec's "shrink under pressure": the budget cap is static, the pressure check is what tracks a browser and an IDE arriving. oMLX's `min(cap, used + slack)` is the *enforcer-side* view of the same idea (how much to reserve from a process ceiling for the hot cache); sous has no process enforcer, so the reservation is the fixed `reserve` above and the shrink is this loop.
9. **Owner-scoped stats and reset.** `PrefixCache.stats(owner=None)` returns the daemon-wide view (all owners plus a `history` of owners already swept) or one owner's counters; both views carry `slots` and `resident_bytes`. `PrefixCache.reset(owner=None)`: with no owner, today's drop-everything + epoch bump (unload; a stalled gateway session); with an owner, drop that owner's slots, fold its counters into history, and **retire** it (a `weakref.WeakSet` of threads) so a late publish from an abandoned generation on that thread is refused. The `Engine` protocol, `ManagedEngine`, `FakeEngine` and every caller gain the `owner` keyword; `GenerationSession` exposes `.thread` so callers can name their session's owner. Counters: `hits` (every warm run, fork or turn), `fork_hits` (the subset via a fork copy), `misses`, `reused_tokens`, `snapshot_bytes`, `cold_retries`, `forks` (fork slots created), `evictions`.
10. **The worker no longer resets at task start and retires only its own session at the end.** Its session thread is new, so no resident slot can be adopted (owner filter), and a strict-prefix match would be correct anyway; the start reset existed only to be safe and today it is what wipes the gateway's slot whenever a delegated task runs between two subagent turns (README: "a delegated task running at the same time … evicts it"). At the end, `reset_prompt_cache(owner=session.thread)` frees the task's slots promptly and refuses a late publish, on the worker thread as before. The task report's `prompt_cache` block is that owner's counters — per task, exact, no zeroing race.
11. **The gateway's per-turn hit/miss is exact** (`stats(owner=session.thread)` before and after; only that thread moves those counters and the turn holds the gateway lock). The stall path retires the dropped session's thread instead of resetting everything. `TurnResult` gains `forked: bool`; the log line says `cache=fork` for a hit served from a fork copy, `cache=hit`/`cache=miss` otherwise.
12. **Dead owners are swept** (slots dropped, counters folded into history) at every `generate`, `stats` and `reset` — a worker session thread that exited without a retire (a stall whose late publish landed) leaves nothing resident past the next call.
13. **A small lock guards the slot list and the counters, held only across dict/list operations, never across prefill or decode.** `reset()` still never waits on the generation lock (its docstring rationale stands); this lock is a different, microsecond-scale one that nothing holds for long.
14. **`reuse_length` becomes a C-speed slice comparison** (`new_ids[:n] == cached_ids`), same semantics, same tests: with up to 16 candidate slots of ~50K tokens each, the Python loop would cost tens of milliseconds per lookup.
15. **Status surfaces the map:** `EngineManager.status()` gains `"prompt_cache": <daemon-wide stats dict>` when the engine is loaded (`server_status` already returns `engines.status()`), so `slots`/`resident_bytes`/`forks`/`evictions` are one MCP call away — counts and bytes only.

---

## File Structure

Modified:

| File | Change |
|---|---|
| `src/sous/engine/promptcache.py` | `FORK_MIN_TOKENS`, `MAX_SLOTS`, `CACHE_BUDGET_SLACK`; `fork_point()`, `slot_bytes()`, `fork_copy()`, `auto_cache_budget()`; `reuse_length` slice compare; `PromptCacheStats` new counters + `add()`; `PromptMemo` `"header"` slot; `CacheHooks.headroom()`; `Slot` dataclass; `PrefixCache` rewritten around a slot list with owner scoping, forks, budget and pressure eviction. |
| `src/sous/engine/base.py` | `Engine` protocol: `reset_prompt_cache(owner=None)`, `prompt_cache_stats(owner=None)`; `ManagedEngine` forwards both; `GenerationSession.thread`; `measure_cache_budget()`, `live_headroom()`; `_default_factory` computes `reserve_bytes` and threads `cache_budget`; `EngineManager` passes config; `EngineManager.status()` `prompt_cache`. |
| `src/sous/engine/lm.py`, `src/sous/engine/vlm.py` | Constructor params `cache_budget`, `reserve_bytes`; `headroom()` hook; header render → `fork_at`; owner-scoped stats/reset passthrough. |
| `src/sous/config.py` | `[model].prompt_cache_gb` (`"auto"` or a non-negative number), field `prompt_cache_gb: float \| None`. |
| `src/sous/worker.py` | No start reset; `_failure_extra(..., owner)`; owner-scoped report; end-of-task `reset_prompt_cache(owner=session.thread)`. |
| `src/sous/gateway/turn.py`, `src/sous/gateway/routes.py` | Owner-scoped before/after; stall retires the dropped session; `TurnResult.forked`; `cache=fork` log token. |
| `tests/test_promptcache.py`, `tests/fake_engine.py`, `tests/test_engine_base.py`, `tests/test_engine_lm.py` (model), `tests/test_engine_unloaded.py`, `tests/test_config.py`, `tests/test_worker.py`, `tests/test_gateway_turn.py`, `tests/test_gateway_routes.py` | Tests per task below. |
| `README.md`, `CONTRIBUTING.md`, `CLAUDE.md` | Cache section, config block, whole-session recipe expectations, gotchas. |

No new source files: the spec sizes this at ~300 lines and every piece belongs to a module that already owns the concern.

---

### Task 1: promptcache helpers — `fork_point`, `fork_copy`, `slot_bytes`, `auto_cache_budget`, stats counters, header memo slot, fast `reuse_length`

**Files:**
- Modify: `src/sous/engine/promptcache.py` (top of module through `PromptMemo`; `CacheHooks`)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 2–5):
  - `FORK_MIN_TOKENS: int = 4096`, `MAX_SLOTS: int = 16`, `CACHE_BUDGET_SLACK: int = 2 << 30`
  - `fork_point(header_ids: Sequence[int], stable_ids: Sequence[int]) -> int`
  - `slot_bytes(cache: Sequence[Any]) -> int`
  - `fork_copy(src: Sequence[Any], dst: Sequence[Any], copy_array: Callable) -> None`
  - `auto_cache_budget(*, working_set: int, active: int, reserve_bytes: int) -> int`
  - `PromptCacheStats(hits, misses, reused_tokens, snapshot_bytes, cold_retries, fork_hits, forks, evictions)` with `as_dict()` and `add(other)`
  - `PromptMemo` accepts slot `"header"`
  - `CacheHooks.headroom() -> int | None`

- [ ] **Step 1: Extend the test fakes so they can be forked and sized**

In `tests/test_promptcache.py`, replace `FakeTrimmable` and `FakeRecurrent` with versions that expose `state`/`meta_state`/`nbytes` the way mlx's classes do (a fork copies `state` and `meta_state`; the budget reads `nbytes`):

```python
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
```

Add to `FakeHooks.__init__` a `self.headroom_value: int | None = None` and the method:

```python
    def headroom(self):
        return self.headroom_value
```

- [ ] **Step 2: Write the failing tests for the new helpers**

Append to `tests/test_promptcache.py` (add `FORK_MIN_TOKENS`, `auto_cache_budget`, `fork_copy`, `fork_point`, `slot_bytes` to the import list):

```python
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


def test_stats_add_sums_every_counter():
    a = PromptCacheStats(hits=1, forks=1)
    a.add(PromptCacheStats(hits=2, evictions=4))
    assert (a.hits, a.forks, a.evictions) == (3, 1, 4)


def test_memo_accepts_the_header_slot():
    m = PromptMemo()
    m.put("header", "sys", [1, 2])
    assert m.get("header", "sys") == [1, 2]
```

Delete the old `test_stats_as_dict_reports_every_counter` (it is replaced above).

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -q -k "fork_point or fork_copy or slot_bytes or auto_cache_budget or stats or header" 2>&1 | tail -5`
Expected: FAIL with `ImportError: cannot import name 'fork_point'`.

- [ ] **Step 4: Implement the helpers**

In `src/sous/engine/promptcache.py`:

Add imports: `import dataclasses`, `import time`, `import weakref`, and `from dataclasses import dataclass, field` (the `field` import is used in Task 2; add it now so the module compiles in one shape).

Replace `reuse_length`'s body:

```python
def reuse_length(cached_ids: Sequence[int], new_ids: Sequence[int]) -> int:
    """How many leading tokens of `new_ids` the cache already holds.

    Strict prefix only: the whole cache, or nothing. sous never rewinds a cache
    to a shorter prefix, because a hybrid model's linear-attention layers hold a
    recurrent state with no inverse. `len(new_ids) > len(cached_ids)` is
    required rather than incidental: mlx rejects an empty prompt, so an exact
    match must count as a miss.
    """
    n = len(cached_ids)
    if not n or len(new_ids) <= n:
        return 0
    # A slice comparison runs in C. With up to MAX_SLOTS candidate slots of
    # ~50K tokens each to test per turn, a Python loop here would cost tens
    # of milliseconds on every lookup.
    return n if list(new_ids[:n]) == list(cached_ids) else 0
```

After `trim_to`, add:

```python
# A header shorter than this is not worth a fork slot: prefilling it costs
# under a second on the default model, and a fork is a full second copy of its
# KV. The worker's ~2K-token system prompt never qualifies; a Claude Code
# subagent's ~50K one always does.
FORK_MIN_TOKENS = 4096
# A sanity bound on the slot count. Finished subagent conversations are what
# fills the map, and LRU eviction discards those first; the byte budget is the
# real limit.
MAX_SLOTS = 16
# Kept free beyond weights, resident slots and the in-flight turn's own cache
# when the budget is derived automatically: drafter activations, mlx's
# allocator slack, the tokenizer.
CACHE_BUDGET_SLACK = 2 << 30


def fork_point(header_ids: Sequence[int], stable_ids: Sequence[int]) -> int:
    """Where a turn may fork a shared-prefix slot: the header's length when the
    header is a strict token prefix of the stable render and long enough to be
    worth a copy, else 0.

    `reuse_length` is the test on purpose: [model].id accepts any chat
    template, and whether "the system turn rendered alone" is a token prefix
    of "the whole conversation rendered" is a property of that template and
    tokenizer. It is verified here every turn, never assumed.
    """
    n = len(header_ids)
    if n < FORK_MIN_TOKENS or reuse_length(header_ids, stable_ids) != n:
        return 0
    return n


def slot_bytes(cache: Sequence[Any]) -> int:
    """Bytes a cache holds resident: every layer's `nbytes`. For a KVCache that
    is the whole allocated buffer, 256-step padding and the trimmed-off
    generation region included — the truth about residency, and an upper
    bound."""
    return sum(int(getattr(c, "nbytes", 0) or 0) for c in cache)


def fork_copy(src: Sequence[Any], dst: Sequence[Any], copy_array: Callable) -> None:
    """Make `dst` — a fresh cache of the same layout — hold exactly what `src`
    holds now, as an independent copy.

    Every layer's `state` is copied array by array and `meta_state` carried
    over: that pair is what mlx's own `_BaseCache.from_state` rebuilds a layer
    from, so it is complete for every cache class mlx ships. A KVCache's
    `state` getter slices keys/values to the current offset, so the copy
    materialises just that many tokens (no step padding) and the setter
    derives the offset from the copied shape. Unlike `snapshot`, which records
    only an offset for a layer it will later restore in place, a fork must
    copy the attention layers too — it is a second cache, not a bookmark.

    Callers fork only after a prefill of at least FORK_MIN_TOKENS tokens, so
    every layer is populated (an empty KVCache's `state` getter would raise).
    Nested-tuple states (QuantizedKVCache) are not handled; sous never
    quantizes its KV.
    """
    for s, d in zip(src, dst, strict=True):
        d.state = [None if a is None else copy_array(a) for a in s.state]
        d.meta_state = s.meta_state


def auto_cache_budget(*, working_set: int, active: int, reserve_bytes: int) -> int:
    """Bytes resident slots may hold beyond the in-flight turn: what Metal
    serves without paging, minus what mlx already holds (the weights, when
    read at load), minus the largest cache one turn can build (`reserve_bytes`
    — the window times the KV cost of a token), minus a fixed slack. This is
    the "reserved out of the generation budget" of the spec: the turn's own
    cache is paid for first, slots get what is left."""
    return max(0, working_set - active - reserve_bytes - CACHE_BUDGET_SLACK)
```

Replace `PromptCacheStats`:

```python
@dataclass
class PromptCacheStats:
    hits: int = 0  # every warm run, from a turn slot or a fork copy
    misses: int = 0
    reused_tokens: int = 0
    snapshot_bytes: int = 0
    cold_retries: int = 0
    fork_hits: int = 0  # the subset of hits served by copying a fork slot
    forks: int = 0  # fork slots created
    evictions: int = 0  # slots dropped for budget, count or pressure

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def add(self, other: PromptCacheStats) -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
```

Change `_MEMO_SLOTS = ("stable", "full", "header")` and the `PromptMemo` docstring's first line to "One slot per render, keyed by the exact prompt text." plus a sentence: "The header slot holds the system turn rendered alone — the fork boundary."

Extend `CacheHooks`:

```python
class CacheHooks(Protocol):
    """The five things only an engine can do. Everything else is shared."""

    def new_cache(self) -> list: ...
    def prefill(self, cache: list, token_ids: list[int]) -> None: ...
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None
    ) -> str: ...
    def copy_array(self, a: object) -> object: ...
    # Bytes the machine can still give a cache right now, or None when it
    # cannot tell (no mlx). Read on the owner thread; must not release thread
    # state.
    def headroom(self) -> int | None: ...
```

- [ ] **Step 5: Run the whole prompt-cache file**

Run: `uv run pytest tests/test_promptcache.py -q 2>&1 | tail -5`
Expected: all PASS (the `PrefixCache` tests still run against the old single-slot implementation; the fakes' new properties do not change their behaviour).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): prompt-cache helpers for forking, sizing and budgeting slots

fork_copy makes an independent second cache from a live one (state plus
meta_state per layer, the pair mlx's own from_state rebuilds from);
fork_point accepts a header as a fork boundary only when it is a strict
token prefix of the turn's render and long enough to be worth a copy;
slot_bytes and auto_cache_budget size what a slot costs and what the
machine can spare once the weights and one full window are paid for.
reuse_length becomes a C-speed slice compare: keyed slots will run it up
to MAX_SLOTS times per turn over ~50K-token prefixes.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `PrefixCache` keyed slots with owner scoping (turn slots only, budget 0)

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`PrefixCache`)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: Task 1's `PromptCacheStats.add`, `slot_bytes`, `reuse_length`, `CacheHooks.headroom`.
- Produces (used by Tasks 3–6):
  - `Slot(cache: list, held: list[int], owner: threading.Thread, kind: str, nbytes: int, last_used: float)` — `kind` is `"turn"` or `"fork"`.
  - `PrefixCache(hooks, enabled=True, *, max_bytes: int = 0, reserve_bytes: int = 0)`
  - `PrefixCache.generate(stable_ids, full_ids, max_tokens, on_delta=None, fork_at: int = 0) -> str` — `fork_at` is accepted and ignored in this task; Task 3 gives it meaning.
  - `PrefixCache.stats(owner: threading.Thread | None = None) -> dict` — counters plus `slots` and `resident_bytes`.
  - `PrefixCache.reset(owner: threading.Thread | None = None) -> None`
  - `PrefixCache.slots() -> list[Slot]` — a snapshot for tests and status; never mutated by callers.

This task keeps *behaviour* identical at the default budget (one slot survives each publish) while changing the *structure*: a slot list, owner filtering, owner-scoped counters, retirement and sweeping. Existing tests must pass unchanged except where they asserted the exact `stats()` dict (two new keys) or reached into `pc._cache`/`pc._held` (one test, rewritten below).

- [ ] **Step 1: Update the existing tests that touch the old internals**

In `tests/test_promptcache.py`, add near the top (after the fakes):

```python
def _empty_stats() -> dict:
    return {**PromptCacheStats().as_dict(), "slots": 0, "resident_bytes": 0}


# Roomy enough that nothing is ever evicted for size in tests that want to see
# several slots coexist; the default of 0 keeps a single slot, as before.
ROOMY = 1 << 40
```

Replace every `PromptCacheStats().as_dict()` comparison against `pc.stats()` with `_empty_stats()` (`test_same_turn_mismatch_decodes_whole_prompt_cold_and_warns` compares a literal dict — extend it with `"fork_hits": 0, "forks": 0, "evictions": 0, "slots": 0, "resident_bytes": 0`; `test_disabled_never_reuses_and_never_counts`, `test_disabled_never_reads_stable_ids`, `test_reset_clears_the_counters`, `test_a_late_cold_retry_write_after_reset_does_not_land_on_fresh_counters`).

Rewrite the planted-slot test, which reached into `pc._cache`/`pc._held`:

```python
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
```

- [ ] **Step 2: Write the failing tests for keyed slots and owner scoping**

Append:

```python
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
    """max_bytes=0 is the constructor default and today's behaviour: every
    publish evicts everything but the slot just published."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    assert [s.held for s in pc.slots()] == [B1]
    assert pc.stats()["evictions"] == 1
    pc.generate(A2, A2_FULL, 16)
    assert pc.stats()["misses"] == 3  # A's slot was evicted by B's publish


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
```

Keep every existing `PrefixCache` test (thread ownership, retries, replay-safety, late write-back, memory release) — they are the correctness argument the spec says each slot must keep.

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -q -k "interleaved or consumed or longest or default_budget or capped or resident or owner or retired or swept or epoch or disabled_never_reads" 2>&1 | tail -5`
Expected: FAIL with `TypeError: PrefixCache.__init__() got an unexpected keyword argument 'max_bytes'` / `AttributeError: ... has no attribute 'slots'`.

- [ ] **Step 4: Rewrite `PrefixCache`**

Replace the class in `src/sous/engine/promptcache.py` (the `_run` method's body stays as it is today except for the new `owner`/`epoch`/`fork_at` parameters, which Task 3 uses):

```python
@dataclass
class Slot:
    """One resident cache and the exact ids it holds, at the stable boundary.

    `owner` is the thread whose mlx streams built the arrays (issue #34): the
    Thread object, not its ident — idents are recycled after a thread exits,
    a strongly held Thread object cannot falsely match a later thread. `kind`
    is "turn" (published when a turn ends; consumed by the turn that extends
    it) or "fork" (a copy taken at a shared-prefix boundary; copied on every
    hit, left in place)."""

    cache: list
    held: list[int]
    owner: threading.Thread
    kind: str
    nbytes: int
    last_used: float = field(default_factory=time.monotonic)


class PrefixCache:
    """KV caches at the stable-render boundary, one slot per resident prefix.

    The boundary matters: the stable render (`add_generation_prompt=False`) is a
    strict prefix of the next turn's stable render, while the full prompt never
    is — the chat template appends a generation-only block that it strips when
    it later re-renders that same assistant turn as history. Anchoring here is
    what makes reuse possible at all.

    A turn looks for the longest slot its own thread owns whose ids are a
    strict prefix of its stable render. A "turn" slot is taken over and
    extended in place, so a conversation never holds two copies of itself; a
    "fork" slot is copied, so the next conversation sharing that prefix finds
    it too. Everything published is charged to `max_bytes`; the in-flight
    turn's own cache never is — `reserve_bytes` (one full window of KV) was
    subtracted from the machine's headroom before `max_bytes` was derived.
    """

    def __init__(
        self,
        hooks: CacheHooks,
        enabled: bool = True,
        *,
        max_bytes: int = 0,
        reserve_bytes: int = 0,
    ):
        # These defaults are the orchestrator's own semantics, not the
        # user-facing ones — both engines always pass them explicitly. The
        # shipped defaults live in SousConfig (prompt_cache, prompt_cache_gb);
        # don't couple these to them. max_bytes=0 keeps exactly one slot: the
        # behaviour every pre-keyed test in the suite was written against.
        self._hooks = hooks
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes
        # Held only across list/dict operations, never across prefill or
        # decode. reset() must never wait on a generation (its caller may be
        # the worker's finally while a stalled generation still runs), and
        # with this discipline it never does.
        self._lock = threading.Lock()
        self._slots: list[Slot] = []
        self._owner_stats: dict[threading.Thread, PromptCacheStats] = {}
        # Counters of owners already retired or swept, so the daemon-wide view
        # keeps their history without keeping their Thread objects.
        self._history = PromptCacheStats()
        # Owners whose late publishes are refused: a task's session thread
        # after the task ended, a gateway session after a stall. Weak, so a
        # thread that is truly gone costs nothing to remember.
        self._retired: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
        self._epoch = 0

    # ---- bookkeeping (call with self._lock held) ----------------------------

    def _sweep(self) -> None:
        """Drop what dead threads left: their arrays lived on streams that no
        longer exist, and nothing can ever adopt them."""
        dead = {s.owner for s in self._slots if not s.owner.is_alive()}
        dead |= {o for o in self._owner_stats if not o.is_alive()}
        if not dead:
            return
        self._slots = [s for s in self._slots if s.owner not in dead]
        for owner in dead:
            self._history.add(self._owner_stats.pop(owner, PromptCacheStats()))

    def _stats_for(self, owner: threading.Thread) -> PromptCacheStats:
        return self._owner_stats.setdefault(owner, PromptCacheStats())

    def _resident(self) -> int:
        return sum(s.nbytes for s in self._slots)

    def _evictable(self, protect: Slot) -> list[Slot]:
        return [s for s in self._slots if s is not protect]

    def _drop_lru(self, protect: Slot) -> None:
        victim = min(self._evictable(protect), key=lambda s: s.last_used)
        self._slots.remove(victim)
        self._stats_for(victim.owner).evictions += 1
        # `victim` dies with this frame: the slot's cache is freed the moment
        # nothing else references it, which is what the pressure re-read in
        # _evict relies on.

    def _evict(self, protect: Slot) -> None:
        """Bring the map under its caps, then under memory pressure, never
        touching `protect` — the slot just published. A turn's own slot is
        never evicted by its own publish: on a machine with room for exactly
        one slot, that one still survives."""
        while self._evictable(protect) and (
            len(self._slots) > MAX_SLOTS or self._resident() > self.max_bytes
        ):
            self._drop_lru(protect)
        headroom = self._hooks.headroom()
        while headroom is not None and headroom < self.reserve_bytes and self._evictable(protect):
            self._drop_lru(protect)
            headroom = self._hooks.headroom()

    def _take(self, owner: threading.Thread, stable_ids: list[int]) -> Slot | None:
        """The longest slot `owner` holds that `stable_ids` strictly extends.
        A turn slot leaves the map with the caller (its cache is about to be
        mutated); a fork slot stays."""
        best: Slot | None = None
        for s in self._slots:
            if s.owner is not owner or not reuse_length(s.held, stable_ids):
                continue
            if best is None or len(s.held) > len(best.held):
                best = s
        if best is None:
            return None
        best.last_used = time.monotonic()
        if best.kind == "turn":
            self._slots.remove(best)
        return best

    def _publish(self, slot: Slot, epoch: int) -> bool:
        """Add `slot` unless the world moved on: a full reset since the turn
        began, or the owner retired (its task ended) while it generated."""
        with self._lock:
            if epoch != self._epoch or slot.owner in self._retired:
                return False
            self._slots.append(slot)
            self._evict(protect=slot)
            return True

    def _plant(self, cache: list, held: list[int]) -> None:
        """Test seam: publish a turn slot for the calling thread directly."""
        with self._lock:
            self._slots.append(
                Slot(cache, list(held), threading.current_thread(), "turn", slot_bytes(cache))
            )

    # ---- public --------------------------------------------------------------

    def slots(self) -> list[Slot]:
        with self._lock:
            return list(self._slots)

    def reset(self, owner: threading.Thread | None = None) -> None:
        """Drop resident caches.

        With an owner: that thread's slots and counters (the counters fold
        into history), and the owner is retired so a generation still running
        on it cannot publish afterwards. Without: everything, and the epoch
        bump makes every in-flight publish from any thread drop itself —
        unload and a stalled gateway session use this form.

        Takes only the bookkeeping lock, never the generation lock: the caller
        may be the worker's finally while a stalled generation still holds
        that one, and waiting there would wedge the next task.
        """
        with self._lock:
            if owner is None:
                self._epoch += 1
                self._slots = []
                self._owner_stats = {}
                self._history = PromptCacheStats()
                return
            self._slots = [s for s in self._slots if s.owner is not owner]
            self._history.add(self._owner_stats.pop(owner, PromptCacheStats()))
            self._retired.add(owner)

    def stats(self, owner: threading.Thread | None = None) -> dict:
        """Counters plus `slots` and `resident_bytes`: for one owner thread, or
        daemon-wide (every live owner plus the history of retired ones)."""
        with self._lock:
            self._sweep()
            if owner is not None:
                total = self._owner_stats.get(owner, PromptCacheStats())
                mine = [s for s in self._slots if s.owner is owner]
            else:
                total = PromptCacheStats()
                total.add(self._history)
                for each in self._owner_stats.values():
                    total.add(each)
                mine = self._slots
            return {
                **total.as_dict(),
                "slots": len(mine),
                "resident_bytes": sum(s.nbytes for s in mine),
            }

    def generate(
        self,
        stable_ids: list[int],
        full_ids: list[int],
        max_tokens: int,
        on_delta: OnDelta | None = None,
        fork_at: int = 0,
    ) -> str:
        hooks = self._hooks
        if not self.enabled:
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)

        owner = threading.current_thread()
        with self._lock:
            self._sweep()
            # Bind the owner's stats object itself and write through this
            # local for the whole call (passed into _run too). reset(owner)
            # swaps in nothing for this owner; if an abandoned generation
            # thread reaches a late write after that, it lands on this
            # orphaned object instead of on counters a later reader sees.
            stats = self._stats_for(owner)
            epoch = self._epoch

        # `_run`'s anchor (`len(stable_ids)`) only means what it assumes: that
        # `full_ids` is the stable render plus a generation-only suffix, true
        # for chat templates that append a generation block rather than
        # rewrite one. [model].id accepts any MLX model and therefore any
        # chat template, so sous cannot assume that in general. reuse_length
        # already tests exactly this — strict prefix, with a suffix left to
        # decode — so ask it instead of duplicating the rule. Checked before
        # any slot is taken: nothing from this turn is retained, and every
        # slot an earlier, well-behaved turn published survives for a later
        # one to reuse.
        if reuse_length(stable_ids, full_ids) == 0:
            stats.misses += 1
            warnings.warn(
                "sous prompt cache: full prompt is not the stable render "
                "plus a generation suffix; decoding cold this turn",
                stacklevel=2,
            )
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)

        with self._lock:
            # Owner-filtered: the arrays live only on the publishing thread's
            # mlx streams (issue #34), so a different session thread (a worker
            # task vs the gateway's long-lived one) gets a cold miss rather
            # than a warm run doomed to the cross-thread mlx failure.
            slot = self._take(owner, stable_ids)
        if slot is not None:
            reuse = len(slot.held)
            if slot.kind == "fork":
                # The slot stays for the next conversation; this turn works
                # on its own copy.
                warm: list = hooks.new_cache()
                fork_copy(slot.cache, warm, hooks.copy_array)
                stats.fork_hits += 1
            else:
                warm = slot.cache
            stats.hits += 1
            stats.reused_tokens += reuse
        else:
            reuse = 0
            stats.misses += 1
            warm = hooks.new_cache()
        # Drop the only remaining reference to a consumed turn slot's cache
        # other than `warm` before anything else is allocated: on a miss with
        # a full window there is no slot to speak of, but on a consumed slot
        # this is what keeps one conversation from ever holding two copies of
        # itself. (A fork slot is referenced by the map by design.)
        slot = None

        # A warm attempt that already streamed text cannot be retried: the
        # consumer has forwarded those deltas, and a cold re-run would deliver
        # the turn a second time. An on_delta wrapped in ReplaySafe forwards
        # nothing outside the process (accounting only, e.g. a non-streaming
        # turn's delta count) — checked on the ORIGINAL callback, before
        # _counting wraps it in its own closure.
        replay_safe = isinstance(on_delta, ReplaySafe)
        emitted = 0

        def _counting(sink: OnDelta) -> OnDelta:
            def relay(delta: Delta) -> None:
                nonlocal emitted
                emitted += 1
                sink(delta)

            return relay

        relay = _counting(on_delta) if on_delta is not None else None

        text = ""
        retry_reason: str | None = None
        try:
            text = self._run(
                stats, warm, stable_ids, full_ids, reuse, max_tokens, relay, fork_at, owner, epoch
            )
        except Exception as e:
            if reuse == 0:
                raise
            # Capture only the message here; the retry itself runs after this
            # suite exits. Python clears the `as e` binding and its traceback
            # at the end of the except clause (PEP 3110) — retrying inside it
            # would keep the failed full-size cache pinned by that traceback
            # for the whole retry, on top of the cold replacement being
            # prefilled: exactly the doubling the spec promises never happens.
            retry_reason = str(e)

        if retry_reason is not None:
            if emitted and not replay_safe:
                raise RuntimeError(
                    f"warm generation failed after streaming {emitted} delta(s) "
                    f"({retry_reason}); not retrying cold, which would replay the turn"
                )
            # An optimization bug must never fail a task; decide_context sets
            # the same rule for auto sizing. Only a warm attempt is retried, so
            # a genuine engine error still surfaces at once.
            stats.cold_retries += 1
            warnings.warn(
                f"sous prompt cache: warm generation failed ({retry_reason}); retrying cold",
                stacklevel=2,
            )
            warm = hooks.new_cache()
            text = self._run(
                stats, warm, stable_ids, full_ids, 0, max_tokens, relay, fork_at, owner, epoch
            )

        self._publish(Slot(warm, list(stable_ids), owner, "turn", slot_bytes(warm)), epoch)
        return text

    def _run(
        self,
        stats: PromptCacheStats,
        cache: list,
        stable_ids: list[int],
        full_ids: list[int],
        reuse: int,
        max_tokens: int,
        on_delta: OnDelta | None,
        fork_at: int,
        owner: threading.Thread,
        epoch: int,
    ) -> str:
        hooks = self._hooks
        anchor = len(stable_ids)
        if all_trimmable(cache):
            # Everything rewinds, so prefill and decode fuse into one pass and
            # the generation block plus the generated tokens are simply trimmed
            # back off afterwards.
            text = hooks.decode(cache, list(full_ids[reuse:]), max_tokens, on_delta)
            trim_to(cache, anchor)
            return text
        # A recurrent layer cannot rewind, so stop at the anchor, record it,
        # and put the cache back there once the generation is done.
        hooks.prefill(cache, list(stable_ids[reuse:]))
        snap, nbytes = snapshot(cache, hooks.copy_array)
        stats.snapshot_bytes = nbytes
        text = hooks.decode(cache, list(full_ids[anchor:]), max_tokens, on_delta)
        restore(cache, snap, hooks.copy_array)
        return text
```

Note the "held" record and the reset docstring: the old `test_reset_drops_the_cache_so_the_next_turn_is_cold`, `test_a_late_write_back_from_an_abandoned_generation_is_dropped` and `test_a_cache_built_on_another_thread_is_a_cold_miss` pass unchanged (`reset()` with no owner; the epoch; the owner filter in `_take`).

- [ ] **Step 5: Run the whole prompt-cache file**

Run: `uv run pytest tests/test_promptcache.py -q 2>&1 | tail -5`
Expected: all PASS. If `test_a_miss_releases_the_previous_cache_before_the_replacement_is_prefilled` fails, the culprit is a lingering reference to the evicted/consumed slot — check `_drop_lru` and the `slot = None` line; do not weaken the test.

- [ ] **Step 6: Run the rest of the suite (engines and gateway tests use fakes, but `FakeEngine`'s signatures have not changed yet, so everything should still pass)**

Run: `uv run pytest -m "not model" -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): keyed prompt-cache slots owned by the thread that built them

One slot per resident prefix instead of one slot per engine: a turn takes
over the longest slot its own thread holds that its render strictly
extends, and publishes back under the longer key. Two conversations on the
local model no longer evict each other. Counters and reset are scoped to
an owner thread, so a task can retire only its own session's slots (the
worker's reset used to wipe the gateway's cache every time a delegated
task ran) and the gateway's per-turn hit/miss is exact. The constructor
default budget of 0 keeps exactly one slot — every pre-existing test keeps
its meaning.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Fork slots at the header boundary, byte budget and pressure eviction

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`PrefixCache._run`, new `_fork_wanted`)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: Task 1's `fork_copy`, `slot_bytes`, `FORK_MIN_TOKENS`; Task 2's `Slot`, `_publish`, `_evict`.
- Produces: `PrefixCache.generate(..., fork_at=N)` creates a `"fork"` slot holding `stable_ids[:N]` on a turn that prefills past `N`; a later turn whose render strictly extends those ids hits it by copy. `max_bytes` and `reserve_bytes` now have observable effect (LRU eviction under the byte cap; pressure eviction when `hooks.headroom()` is below `reserve_bytes`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py`:

```python
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
```

The `h.headroom = lambda ...` assignment over a method needs the `# ty: ignore[invalid-assignment]` pragma shown (the same reason `FakeHooks.decode_impl` is injected rather than monkeypatched); if ty accepts it without the pragma, drop the pragma — ruff's `PGH004`/unused-ignore rules are not enabled, but an unused suppression is still noise.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -q -k "fork or budget or evict or recency or pressure or headroom" 2>&1 | tail -5`
Expected: FAIL — `test_a_cold_turn_forks_at_the_header` asserts a `"fork"` slot that is never created; the budget tests fail on `evictions`/slot lists (eviction under the byte cap already works from Task 2's `_evict`, so some of the budget tests may already pass — that is fine).

- [ ] **Step 3: Implement forking in `_run`**

Add to `PrefixCache` (bookkeeping section):

```python
    def _fork_wanted(
        self, owner: threading.Thread, stable_ids: list[int], fork_at: int, reuse: int
    ) -> int:
        """Whether this turn should leave a fork slot at `fork_at`: only a turn
        that is itself prefilling past the boundary can (a hit that started at
        or beyond it holds the header inside a cache that cannot rewind), and
        only when no fork with exactly those ids exists for this owner."""
        if not (reuse < fork_at < len(stable_ids)):
            return 0
        held = stable_ids[:fork_at]
        with self._lock:
            for s in self._slots:
                if s.owner is owner and s.kind == "fork" and s.held == held:
                    return 0
        return fork_at
```

Replace `_run`:

```python
    def _run(
        self,
        stats: PromptCacheStats,
        cache: list,
        stable_ids: list[int],
        full_ids: list[int],
        reuse: int,
        max_tokens: int,
        on_delta: OnDelta | None,
        fork_at: int,
        owner: threading.Thread,
        epoch: int,
    ) -> str:
        hooks = self._hooks
        anchor = len(stable_ids)
        fork_at = self._fork_wanted(owner, stable_ids, fork_at, reuse)
        if fork_at:
            # Stop at the boundary, copy, and continue from the copy's twin:
            # a fork is taken while prefilling past the header because no
            # layer can be rewound to it afterwards. The copy is charged to
            # the budget like any slot and may be evicted by the very publish
            # that adds it — then this turn simply left nothing behind.
            hooks.prefill(cache, list(stable_ids[reuse:fork_at]))
            copy = hooks.new_cache()
            fork_copy(cache, copy, hooks.copy_array)
            if self._publish(
                Slot(copy, list(stable_ids[:fork_at]), owner, "fork", slot_bytes(copy)), epoch
            ):
                stats.forks += 1
            del copy
            reuse = fork_at
        if all_trimmable(cache):
            # Everything rewinds, so (the rest of) prefill and decode fuse into
            # one pass and the generation block plus the generated tokens are
            # simply trimmed back off afterwards.
            text = hooks.decode(cache, list(full_ids[reuse:]), max_tokens, on_delta)
            trim_to(cache, anchor)
            return text
        # A recurrent layer cannot rewind, so stop at the anchor, record it,
        # and put the cache back there once the generation is done.
        hooks.prefill(cache, list(stable_ids[reuse:]))
        snap, nbytes = snapshot(cache, hooks.copy_array)
        stats.snapshot_bytes = nbytes
        text = hooks.decode(cache, list(full_ids[anchor:]), max_tokens, on_delta)
        restore(cache, snap, hooks.copy_array)
        return text
```

The `del copy` matters: a fork the publish evicted (the budget was full) must not stay pinned by this frame through the whole decode.

- [ ] **Step 4: Run the whole prompt-cache file, twice (thread-using tests)**

Run: `uv run pytest tests/test_promptcache.py -q 2>&1 | tail -3 && uv run pytest tests/test_promptcache.py -q -p no:randomly 2>&1 | tail -3`
Expected: all PASS both times.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): fork a shared-prefix slot at the header, under a byte budget

A cold turn that prefills past the header boundary leaves a copy of the
cache there; a later conversation whose render extends that header starts
from the copy instead of from nothing. That is the Phase 2 exit evidence
turned into a mechanism: every Claude Code subagent of one type shares
~50K tokens of system prompt and tool schemas, and the second one used to
miss because the single slot held the first one's whole conversation.
Resident slots are bounded by max_bytes (LRU, the slot just published
protected) and shrunk further while live headroom is below one window's
worth of KV — the spec's 'reserved out of the generation budget'.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Engine protocol, `ManagedEngine`, `GenerationSession.thread`, memory readers, factory plumbing, status, config

**Files:**
- Modify: `src/sous/engine/base.py`, `src/sous/config.py`
- Modify: `tests/fake_engine.py`
- Test: `tests/test_engine_base.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: Task 1's `auto_cache_budget`; `sous.context.kv_bytes_per_token`.
- Produces (used by Tasks 5–6):
  - `Engine.reset_prompt_cache(self, owner: threading.Thread | None = None) -> None`
  - `Engine.prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict`
  - `ManagedEngine` forwards both keywords.
  - `GenerationSession.thread -> threading.Thread` (property)
  - `measure_cache_budget(reserve_bytes: int) -> int`
  - `live_headroom() -> int | None`
  - `_default_factory(..., cache_budget: int | None = None, reserve_tokens: int = 0)`
  - `SousConfig.prompt_cache_gb: float | None = None` (None = auto); TOML `[model].prompt_cache_gb = "auto" | <number ≥ 0>`
  - `EngineManager.status()["prompt_cache"]` — the daemon-wide stats dict when loaded, absent otherwise.
  - `FakeEngine.reset_owners: list[threading.Thread | None]`, `FakeEngine.stats_owners: list[threading.Thread | None]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py` — add (follow the file's existing pattern of writing a TOML to `tmp_path` and calling `load_config(path)`):

```python
def test_prompt_cache_gb_defaults_to_auto(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "[model]\nid = 'x'\n"))
    assert cfg.prompt_cache_gb is None


def test_prompt_cache_gb_accepts_a_number(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "[model]\nprompt_cache_gb = 12.5\n"))
    assert cfg.prompt_cache_gb == 12.5


def test_prompt_cache_gb_zero_means_a_single_slot(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "[model]\nprompt_cache_gb = 0\n"))
    assert cfg.prompt_cache_gb == 0.0


@pytest.mark.parametrize("bad", ["-1", "true", "'lots'", "nan"])
def test_prompt_cache_gb_rejects_garbage_with_a_warning(tmp_path: Path, bad: str):
    with pytest.warns(UserWarning, match=r"\[model\]\.prompt_cache_gb"):
        cfg = load_config(_write(tmp_path, f"[model]\nprompt_cache_gb = {bad}\n"))
    assert cfg.prompt_cache_gb is None
```

(`_write` is whatever helper `tests/test_config.py` already uses to write a config file; if it is named differently, use that name.)

`tests/test_engine_base.py` — add:

```python
def test_managed_engine_forwards_owner_scoped_reset_and_stats():
    inner = FakeEngine([])
    managed = ManagedEngine(inner)
    me = threading.current_thread()
    managed.reset_prompt_cache(owner=me)
    managed.prompt_cache_stats(owner=me)
    assert inner.reset_owners == [me]
    assert inner.stats_owners == [me]


def test_session_exposes_its_thread():
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    session.generate([], [], 8, timeout=5)
    assert inner.generate_threads == [session.thread]
    session.close()
    session.join(5)


def test_status_carries_the_prompt_cache_view_once_loaded(tmp_path):
    inner = FakeEngine([])
    inner.stats = {"hits": 1, "slots": 2, "resident_bytes": 3}
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    assert "prompt_cache" not in manager.status()
    manager.get()
    assert manager.status()["prompt_cache"] == {"hits": 1, "slots": 2, "resident_bytes": 3}


def test_default_factory_threads_the_cache_budget_and_reserve(monkeypatch):
    """The reserve is one full window of KV at the model's per-token cost, for
    the larger of the worker's and the gateway's windows."""
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(
        base,
        "fetch_model_config",
        lambda mid: {
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    )
    base._default_factory("m", prompt_cache=True, cache_budget=7, reserve_tokens=1000)
    # 2 (K,V) x 2 layers x 1 head x 8 dim x 2 bytes = 64 B/token
    assert seen["reserve_bytes"] == 64 * 1000
    assert seen["cache_budget"] == 7


def test_default_factory_falls_back_to_a_single_slot_when_the_kv_cost_is_unknown(monkeypatch):
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "mystery"})
    with pytest.warns(UserWarning, match="KV cost"):
        base._default_factory("m", prompt_cache=True, cache_budget=None, reserve_tokens=1000)
    assert (seen["cache_budget"], seen["reserve_bytes"]) == (0, 0)


def test_engine_manager_passes_the_configured_budget_and_the_larger_window(tmp_path, monkeypatch):
    from sous.engine import base

    seen = {}

    def factory(model_id, *args, **kw):
        seen.update(kw)
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", factory)
    cfg = _cfg(
        tmp_path,
        prompt_cache_gb=1.5,
        max_context_tokens=32768,
        gateway_enabled=True,
        gateway_max_context_tokens=131072,
    )
    EngineManager(cfg).get()
    assert seen["cache_budget"] == int(1.5 * (1 << 30))
    assert seen["reserve_tokens"] == 131072
    seen.clear()
    EngineManager(_cfg(tmp_path, gateway_enabled=False, max_context_tokens=32768)).get()
    assert seen["cache_budget"] is None  # auto
    assert seen["reserve_tokens"] == 32768
```

`_cfg` is the file's existing tmp_path config helper (the file has one for `test_get_is_lazy_and_cached`; extend it to accept `**overrides` if it does not already). The last test relies on `EngineManager` calling `_default_factory` with the budget values as *keywords* — write it that way.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py -q -k "prompt_cache_gb or owner_scoped or exposes_its_thread or prompt_cache_view or cache_budget or single_slot or larger_window" 2>&1 | tail -5`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'owner'`, `AttributeError: 'GenerationSession' object has no attribute 'thread'`, config attribute missing).

- [ ] **Step 3: Config**

In `src/sous/config.py`: add `"prompt_cache_gb"` to `_KNOWN["model"]`; add the field after `prompt_cache`:

```python
    # Memory resident prompt-cache slots may hold beyond the in-flight turn's
    # own cache, in GiB. None means automatic: what Metal's working set has
    # left once the weights, one full window of KV (the larger of the worker's
    # and the gateway's) and 2 GiB of slack are paid for. 0 keeps a single
    # slot. Slots are what let two conversations interleave on the local model
    # without evicting each other, and what lets a new subagent start from a
    # copy of the ~50K-token header its predecessor already prefilled.
    prompt_cache_gb: float | None = None
```

Add the validator next to `_speculative_block_size`:

```python
def _prompt_cache_gb(model: dict) -> float | None:
    """[model].prompt_cache_gb: "auto" (None) or a non-negative number of GiB.
    Anything else warns and means auto."""
    value = model.get("prompt_cache_gb", "auto")
    if value == "auto":
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value != value  # NaN
        or value < 0
    ):
        warnings.warn(
            f'sous config: [model].prompt_cache_gb {value!r} must be "auto" or a '
            'non-negative number of GiB; using "auto"',
            stacklevel=3,
        )
        return None
    return float(value)
```

and in `load_config`: `prompt_cache_gb=_prompt_cache_gb(model),`.

- [ ] **Step 4: `base.py`**

Protocol:

```python
    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None: ...
    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict: ...
```

`ManagedEngine`:

```python
    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        # No _gen_lock, on purpose. An abandoned stalled generation still holds
        # it, and run_task calls this in a finally — waiting there would wedge
        # the next task. What actually makes a lock-free reset safe against
        # that thread's late write-back is not the epoch guard by itself: a
        # slot is always published together with the exact token ids it
        # contains, and reuse_length demands a full strict-prefix match, so a
        # stale slot adopted by a later task is either rejected outright or
        # genuinely correct for it. The epoch (and, per owner, retirement) is
        # only a cheap early-out on top of that — it skips the adoption, it
        # doesn't guarantee it.
        self._inner.reset_prompt_cache(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._inner.prompt_cache_stats(owner)
```

`GenerationSession`:

```python
    @property
    def thread(self) -> threading.Thread:
        """The thread every generation of this session runs on — the owner
        of every prompt-cache slot those generations publish."""
        return self._thread
```

Memory readers (after `release_mlx_thread_state`):

```python
def measure_cache_budget(reserve_bytes: int) -> int:
    """The automatic resident-slot budget, read once the weights are loaded so
    `active` is the weights (drafter included). Deliberately no
    release_mlx_thread_state() here: this runs on whichever thread loaded the
    engine, and that thread releases on its own schedule — a release from
    inside would destroy streams the caller still uses (#34)."""
    import mlx.core as mx

    from sous.engine.promptcache import auto_cache_budget

    info = mx.device_info()
    return auto_cache_budget(
        working_set=int(info["max_recommended_working_set_size"]),
        active=mx.get_active_memory(),
        reserve_bytes=reserve_bytes,
    )


def live_headroom() -> int | None:
    """Bytes a cache could still take without paging: the tighter of Metal's
    working set minus what mlx holds and available RAM plus mlx's reclaimable
    buffer cache (sous.context.auto_context_tokens' definition). Read on the
    owner thread mid-turn; same no-release rule as measure_cache_budget. None
    when the numbers are unavailable — the caller then skips its check."""
    try:
        import mlx.core as mx
        import psutil

        info = mx.device_info()
        metal = int(info["max_recommended_working_set_size"]) - mx.get_active_memory()
        system = psutil.virtual_memory().available + mx.get_cache_memory()
        return max(0, min(metal, system))
    except Exception:  # noqa: BLE001 — mlx absent or API moved; a check we skip, not a failure
        return None
```

`_default_factory`:

```python
def _default_factory(
    model_id: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    prompt_cache: bool = True,
    draft_id: str = "",
    draft_block_size: int = 0,
    cache_budget: int | None = None,
    reserve_tokens: int = 0,
) -> Engine:
    from sous.context import kv_bytes_per_token

    model_config = fetch_model_config(model_id)
    backend = select_backend(model_config)
    # One full window of KV, kept free for the in-flight turn before slots get
    # anything (the spec's "reserved out of the generation budget"). Unknown
    # per-token cost means the reserve cannot be sized, so the automatic
    # budget degrades to a single slot — never to an unbounded one.
    bytes_per_token = kv_bytes_per_token(model_config)
    reserve_bytes = reserve_tokens * bytes_per_token if bytes_per_token else 0
    if bytes_per_token is None and cache_budget is None:
        warnings.warn(
            f"sous: KV cost per token unknown for {model_id}; keeping a single "
            "prompt-cache slot (set [model].prompt_cache_gb to override)",
            stacklevel=2,
        )
        cache_budget = 0
    if backend == "vlm":
        from sous.engine import vlm

        return vlm.VLMEngine(
            model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt_cache=prompt_cache,
            draft_id=draft_id,
            draft_block_size=draft_block_size,
            cache_budget=cache_budget,
            reserve_bytes=reserve_bytes,
        )
    from sous.engine import lm

    return lm.LMEngine(
        model_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prompt_cache=prompt_cache,
        cache_budget=cache_budget,
        reserve_bytes=reserve_bytes,
    )
```

(`import warnings` at the top of `base.py`.) `EngineManager.__init__`'s lambda:

```python
            lambda model_id: _default_factory(
                model_id,
                config.temperature,
                config.top_p,
                config.top_k,
                config.prompt_cache,
                draft_id=config.speculative_draft_id,
                draft_block_size=config.speculative_block_size,
                cache_budget=(
                    None if config.prompt_cache_gb is None else int(config.prompt_cache_gb * (1 << 30))
                ),
                # The largest cache one turn can build on this daemon: the
                # gateway's window when it is on, else the worker's.
                reserve_tokens=max(
                    config.max_context_tokens,
                    config.gateway_max_context_tokens if config.gateway_enabled else 0,
                ),
            )
```

`EngineManager.status()`:

```python
    def status(self) -> dict:
        with self._lock:
            idle = (time.monotonic() - self._last_used) if self._last_used else None
            out = {
                "loaded": self._engine is not None,
                "model_id": self._config.model_id,
                "idle_seconds": idle,
            }
            if self._engine is not None:
                # Counts and byte totals only; never a token id.
                out["prompt_cache"] = self._engine.prompt_cache_stats()
            return out
```

- [ ] **Step 5: `tests/fake_engine.py`**

```python
        self.reset_owners: list[threading.Thread | None] = []
        self.stats_owners: list[threading.Thread | None] = []
    ...
    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self.resets += 1
        self.reset_idents.append(threading.get_ident())
        self.reset_owners.append(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        self.stats_owners.append(owner)
        return dict(self.stats)
```

- [ ] **Step 6: Run the two test files, then the suite**

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py -q 2>&1 | tail -3 && uv run pytest -m "not model" -q 2>&1 | tail -3`
Expected: all PASS. (`test_engine_unloaded.py` constructs `LMEngine`/`VLMEngine` through monkeypatched loaders — if it fails on the new constructor keywords, that is Task 5's job; note it and continue.)

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/base.py src/sous/config.py tests/fake_engine.py tests/test_engine_base.py tests/test_config.py
git commit -m "feat(engine): owner-scoped cache API, resident budget config and memory readers

The Engine protocol's reset and stats take an owner thread, and a session
names its own, so callers can retire and account for exactly the slots
their generations built. [model].prompt_cache_gb sizes the resident
budget (\"auto\": working set minus weights minus one full window of KV
minus slack, measured once the weights are loaded); the factory computes
the reserve from the model's per-token KV cost for the larger of the two
windows. The memory readers never release thread state: they run on
threads whose caches are still live.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Engines — header render, fork point, budget at load, headroom hook

**Files:**
- Modify: `src/sous/engine/lm.py`, `src/sous/engine/vlm.py`
- Test: `tests/test_engine_base.py` (mlx-free, via `object.__new__` engines the way `tests/test_engine_unloaded.py` builds them), `tests/test_engine_unloaded.py`, `tests/test_engine_lm.py` (model-marked, local only)

**Interfaces:**
- Consumes: Task 1's `fork_point`, `PromptMemo` `"header"` slot; Task 4's `measure_cache_budget`, `live_headroom`, constructor keywords `cache_budget: int | None`, `reserve_bytes: int`.
- Produces: `LMEngine(..., cache_budget=None, reserve_bytes=0)`, `VLMEngine(..., cache_budget=None, reserve_bytes=0)`; both implement `headroom()`; `generate` passes `fork_at` computed from the header render; `reset_prompt_cache(owner)` / `prompt_cache_stats(owner)` pass through.

Both engines get identical edits; the code below is written once with `LMEngine` names — apply the same to `VLMEngine` (`self._processor` in place of `self._tokenizer` where the class already differs).

- [ ] **Step 1: Write the failing mlx-free tests**

In `tests/test_engine_base.py` (or a new `tests/test_engine_forkpoint.py` if the file is getting long — it is already ~550 lines, so prefer the new file):

```python
"""The engines' fork boundary: the system turn rendered alone, verified as a
token prefix of the whole render — exercised without mlx by faking the
tokenizer, the way tests/test_engine_unloaded.py builds engines."""

import threading

from sous.engine.lm import LMEngine
from sous.engine.promptcache import FORK_MIN_TOKENS, PrefixCache, PromptMemo


class FakeTokenizer:
    """Renders messages as a fixed-width token per character so a header that
    is a text prefix is also a token prefix, and vice versa."""

    bos_token = None

    def __init__(self, header_chars: int = FORK_MIN_TOKENS):
        self.header_chars = header_chars

    def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize, enable_thinking):
        out = []
        for m in messages:
            if m["role"] == "system":
                out.append("S" * self.header_chars)
            else:
                out.append(m["content"])
        if add_generation_prompt:
            out.append("G")
        return "|".join(out)

    def encode(self, text, add_special_tokens):
        return [ord(c) for c in text]


class Recording:
    """A PrefixCache stand-in recording the fork_at it was handed."""

    enabled = True

    def __init__(self):
        self.fork_ats: list[int] = []

    def generate(self, stable_ids, full_ids, max_tokens, on_delta=None, fork_at=0):
        self.fork_ats.append(fork_at)
        return "text"

    def stats(self, owner=None):
        return {}

    def reset(self, owner=None):
        pass


def _engine(tokenizer) -> tuple[LMEngine, Recording]:
    engine = object.__new__(LMEngine)
    engine.model_id = "test/model"
    engine._model = object()
    engine._tokenizer = tokenizer
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    rec = Recording()
    engine._cache = rec  # ty: ignore[invalid-assignment]
    return engine, rec


SYSTEM = {"role": "system", "content": "ignored by the fake"}
USER = {"role": "user", "content": "hello"}


def test_a_leading_system_turn_long_enough_is_the_fork_point():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [FORK_MIN_TOKENS]  # the header render's length


def test_a_short_system_turn_does_not_fork():
    engine, rec = _engine(FakeTokenizer(header_chars=10))
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_no_system_turn_means_no_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([USER], [], 8)
    assert rec.fork_ats == [0]


def test_a_system_only_prompt_does_not_fork():
    # Nothing would follow the header; fork_point's strict-prefix rule says 0.
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [0]


def test_a_template_whose_header_is_not_a_token_prefix_does_not_fork():
    class Rewriting(FakeTokenizer):
        def apply_chat_template(self, messages, **kw):
            text = super().apply_chat_template(messages, **kw)
            # A template that changes the system turn once a user turn follows.
            return text.replace("S", "s", 1) if len(messages) > 1 else text

    engine, rec = _engine(Rewriting())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_the_header_render_is_skipped_when_the_cache_is_disabled():
    engine, rec = _engine(FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]
    assert engine._memo.get("header", "S" * FORK_MIN_TOKENS) is None


def test_owner_scoped_stats_and_reset_pass_through():
    class Spy(Recording):
        def __init__(self):
            super().__init__()
            self.seen = []

        def stats(self, owner=None):
            self.seen.append(("stats", owner))
            return {}

        def reset(self, owner=None):
            self.seen.append(("reset", owner))

    engine, _ = _engine(FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]
```

Add the same six fork-point tests for `VLMEngine` in the same file with a `_vlm_engine` builder (`engine._processor = tokenizer`; `should_add_special_tokens` is imported inside `VLMEngine._encode` from `mlx_vlm.utils`, which is absent on the lint runner but present wherever the suite runs — the existing unloaded tests already import `sous.engine.vlm`; to keep this file mlx-free, monkeypatch `VLMEngine._encode` to `lambda self, text: [ord(c) for c in text]` with `monkeypatch.setattr(VLMEngine, "_encode", ...)`).

Add to `tests/test_engine_unloaded.py`'s two builders: `engine._cache = PrefixCache(engine, enabled=True)` stays valid (the constructor keywords have defaults). Add one test:

```python
def test_lm_headroom_never_raises_without_mlx(monkeypatch):
    import sous.engine.base as base

    monkeypatch.setattr(base, "live_headroom", lambda: None)
    assert _unloaded_lm().headroom() is None
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_engine_forkpoint.py tests/test_engine_unloaded.py -q 2>&1 | tail -5`
Expected: FAIL — `Recording.generate() got an unexpected keyword` is *not* the failure (the fake accepts it); the failure is `rec.fork_ats == [0]` where `[FORK_MIN_TOKENS]` is expected, and `AttributeError: 'LMEngine' object has no attribute 'headroom'`.

- [ ] **Step 3: Implement in `lm.py`**

Constructor:

```python
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = False,
        cache_budget: int | None = None,
        reserve_bytes: int = 0,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        from sous.engine.base import measure_cache_budget

        self.model_id = model_id
        self._model, self._tokenizer = load(model_id)  # ty: ignore[invalid-assignment]
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._tokenize_lock = threading.Lock()
        # Measured after load, so the weights are inside `active`.
        if cache_budget is None:
            cache_budget = measure_cache_budget(reserve_bytes)
        self._cache = PrefixCache(
            self, enabled=prompt_cache, max_bytes=cache_budget, reserve_bytes=reserve_bytes
        )
```

Hooks — add after `copy_array`:

```python
    def headroom(self) -> int | None:
        from sous.engine.base import live_headroom

        return live_headroom()
```

(Imported inside the method so `tests/test_engine_unloaded.py` can monkeypatch `sous.engine.base.live_headroom` and see it take effect.)

`prefill`'s comment "Only the non-trimmable path reaches here." → "The non-trimmable path and any turn that forks at the header reach here; the trimmable path otherwise fuses prefill into decode."

`generate`:

```python
    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        full_ids = self._ids("full", messages, tools)
        if not self._cache.enabled:
            # The stable render is only an anchor for reuse, and PrefixCache
            # discards it when disabled — so computing it (and the header)
            # would cost extra tokenizations per turn for nothing.
            return self._cache.generate([], full_ids, max_tokens, on_delta)
        stable_ids = self._ids("stable", messages, tools)
        return self._cache.generate(
            stable_ids, full_ids, max_tokens, on_delta, fork_at=self._fork_at(messages, tools, stable_ids)
        )

    def _fork_at(self, messages: list[dict], tools: list[dict], stable_ids: list[int]) -> int:
        """The header boundary: the leading system turn rendered alone, with
        the same tools, when it is a strict token prefix of the whole stable
        render and long enough to be worth a fork slot (promptcache.fork_point
        decides both). Every Claude Code subagent of one type shares exactly
        this prefix; the worker's short system prompt never qualifies."""
        if len(messages) < 2 or messages[0].get("role") != "system":
            return 0
        header_ids = self._ids("header", messages[:1], tools)
        return fork_point(header_ids, stable_ids)
```

(`from sous.engine.promptcache import PrefixCache, PromptMemo, fork_point`.) The `_ids` slot `"header"` renders with `generation=False` already (`generation=slot == "full"`).

Passthroughs:

```python
    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self._cache.reset(owner)
        if owner is None:
            self._memo.clear()

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._cache.stats(owner)
```

The memo is text-keyed and shared by every caller, so an owner-scoped reset leaves it alone; only the drop-everything form (unload) clears it, as before.

- [ ] **Step 4: Implement the same in `vlm.py`**

Same constructor tail (after the drafter block — `measure_cache_budget` must run after the drafter is loaded and quantized, so its ~1–2 GB is inside `active`), same `headroom`, same `generate`/`_fork_at`, same passthroughs. `VLMEngine`'s `_ids` already handles the slot name generically.

- [ ] **Step 5: Model-marked fork test (local only)**

Append to `tests/test_engine_lm.py`:

```python
def test_lm_fork_copy_matches_a_cold_prefill_bit_for_bit():
    """A cache continued from a fork copy must equal a cache prefilled cold
    over the same tokens, exactly — the fork is a second cache, and any
    drift here would silently change the context of every subagent that
    starts from it."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import fork_copy, slot_bytes

    e = LMEngine(TINY, cache_budget=0)
    model, tokenizer = e._loaded()
    ids = list(tokenizer.encode("def f(x):\n    return x + 1\n" * 40))
    header, tail = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    ref = make_prompt_cache(model)
    e.prefill(ref, header + tail)

    src = make_prompt_cache(model)
    e.prefill(src, header)
    fork = make_prompt_cache(model)
    fork_copy(src, fork, e.copy_array)
    assert slot_bytes(fork) > 0
    assert slot_bytes(fork) <= slot_bytes(src)  # the copy carries no step padding
    e.prefill(fork, tail)

    for a, b in zip(fork, ref, strict=True):
        assert int(a.offset) == int(b.offset)
        for xa, xb in zip(a.state, b.state, strict=True):
            d = mx.max(mx.abs(xa.astype(mx.float32) - xb.astype(mx.float32)))
            mx.eval(d)
            assert d.item() == 0.0
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
```

Run locally: `uv run pytest tests/test_engine_lm.py -q -m model -k "fork" 2>&1 | tail -3` — expected PASS (downloads ~350 MB once). Record the result in the commit body. The default (hybrid) model's `ArraysCache` path is covered by the recurrent-layer fakes here and by the existing `snapshot`/`restore` state-copy machinery; Task 8 exercises it for real.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -m "not model" -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/lm.py src/sous/engine/vlm.py tests/test_engine_forkpoint.py tests/test_engine_unloaded.py tests/test_engine_lm.py
git commit -m "feat(engine): fork the prompt cache at the system header

Both engines render the leading system turn alone, with the same tools,
and hand its token length to the cache as the fork boundary when it is a
strict prefix of the whole render (verified every turn, so any chat
template is safe) and long enough to be worth a copy. The resident budget
is measured once the weights (and the drafter) are loaded, so it is what
the machine actually has left.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Callers — worker retires its own session, gateway reports exact per-turn reuse and `cache=fork`

**Files:**
- Modify: `src/sous/worker.py`, `src/sous/gateway/turn.py`, `src/sous/gateway/routes.py`
- Test: `tests/test_worker.py`, `tests/test_gateway_turn.py`, `tests/test_gateway_routes.py`

**Interfaces:**
- Consumes: Task 4's `GenerationSession.thread`, owner keywords on `ManagedEngine`; `FakeEngine.reset_owners`/`stats_owners`.
- Produces: `TurnResult.forked: bool = False`; log token `cache=fork|hit|miss`.

- [ ] **Step 1: Update the worker tests**

In `tests/test_worker.py`:

- Every `assert inner.resets == 2` / `engine.resets == 2` becomes `== 1` (lines ~415, ~467, ~625, and the two tests below).
- Replace `test_run_task_resets_the_prompt_cache_at_start_and_end` with:

```python
def test_run_task_retires_only_its_own_session_at_the_end(env):
    """No reset at entry — the task's session thread is new, so nothing
    resident can be adopted, and a reset there would wipe the gateway's
    slots every time a delegated task ran. At the end the task retires
    exactly its session's thread, from the worker thread as before."""
    root, cfg, store = env
    task = _start(store, root)
    inner = FakeEngine([FINISH])
    engine = SessionCapturingEngine(inner)
    run_task(task, store, engine, cfg)
    assert inner.resets == 1
    assert inner.reset_owners == [engine.sessions[0].thread]
    _join_sessions(engine)
    assert inner.reset_idents == [threading.get_ident()]
```

- `test_run_task_resets_even_when_the_task_fails`: `assert engine.resets == 1`.
- `test_report_carries_the_prompt_cache_block` and `test_failure_extra_carries_the_prompt_cache_block`: wrap the engine in `SessionCapturingEngine` and add `assert engine_inner.stats_owners and all(o is engine.sessions[0].thread for o in engine_inner.stats_owners)` — the report is that owner's counters.

- [ ] **Step 2: Update the gateway tests**

`tests/test_gateway_turn.py`:

```python
def test_cache_hit_is_reported_from_the_sessions_own_counters(tmp_path: Path):
    inner = FakeEngine(["a", "b", "c"])
    inner.stats = {"hits": 0, "fork_hits": 0, "reused_tokens": 0}
    runner, _ = _runner(tmp_path, inner)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {"hits": 1, "fork_hits": 0, "reused_tokens": 900}
        if len(inner.calls) == 3:
            inner.stats = {"hits": 2, "fork_hits": 1, "reused_tokens": 900 + 4000}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    first = runner.run(MSGS, [], 100, RecordingSink())
    second = runner.run(MSGS, [], 100, RecordingSink())
    third = runner.run(MSGS, [], 100, RecordingSink())
    assert (first.cache_hit, first.forked, first.reused_tokens) == (False, False, 0)
    assert (second.cache_hit, second.forked, second.reused_tokens) == (True, False, 900)
    assert (third.cache_hit, third.forked, third.reused_tokens) == (True, True, 4000)
    # Every read named the gateway session's thread: exact per-turn deltas,
    # unaffected by a worker task's counters or resets.
    session_thread = inner.generate_threads[0]
    assert inner.stats_owners and all(o is session_thread for o in inner.stats_owners)


def test_a_stall_retires_the_dropped_sessions_thread_not_everything(tmp_path: Path):
    # Reuse the body of test_a_stall_drops_the_session_and_the_next_turn_gets_a_new_one
    # up to the GenerationStalled assertion, then:
    ...
    assert inner.reset_owners == [stalled_thread]  # the first session's thread, not None
```

(Adapt the existing stall test: capture `inner.generate_threads[0]` as `stalled_thread` after the stalled call.) Replace the existing `test_cache_hit_is_reported_from_the_engines_counters` with the first test above.

`tests/test_gateway_routes.py`: find the test that asserts the `cache=hit`/`cache=miss` log line (grep `cache=`); add a case where the engine's stats move `fork_hits` and assert the line carries `cache=fork`. If no existing test pins the log token, add one next to the existing `_log_turn` coverage using the same `caplog`/stderr capture that file already uses for `sous gateway:` lines.

- [ ] **Step 3: Run the updated tests to verify they fail**

Run: `uv run pytest tests/test_worker.py tests/test_gateway_turn.py tests/test_gateway_routes.py -q 2>&1 | tail -5`
Expected: FAIL (`resets == 2`, `reset_owners == [None]`, `TurnResult` has no `forked`).

- [ ] **Step 4: Worker**

In `src/sous/worker.py`:

`_failure_extra` gains `owner: threading.Thread` and reads `engine.prompt_cache_stats(owner=owner)`; every call site passes `session.thread` (`_failure_extra(ex, transcript, engine, elisions, session.thread)`). `import threading` at the top if not present.

Replace the entry reset and its comment:

```python
    # No reset here. A slot is usable only from the thread that built it, and
    # this task's session thread is brand new, so nothing resident can be
    # adopted — and a strict-prefix match would be correct if it could. A
    # reset here used to wipe the gateway's slot every time a delegated task
    # ran between two subagent turns.
    # One generation thread for the whole task: the prompt cache lives on
    # that thread's mlx streams, so this is what lets turn N+1 reuse it (#34).
    session = engine.session()
```

The report: `"prompt_cache": {**engine.prompt_cache_stats(owner=session.thread), "elisions": elisions},`.

The finally:

```python
    finally:
        # Runs on every exit from the try — normal finish, any early return
        # in the loop, or an exception escaping it. The session thread ends
        # here; the retire stays on the worker thread, the single owner of
        # it on every path, and names the session's thread so only this
        # task's slots go (the gateway's stay) and a late publish from a
        # stalled generation on that thread is refused.
        session.close()
        engine.reset_prompt_cache(owner=session.thread)
```

- [ ] **Step 5: Gateway turn and log**

`src/sous/gateway/turn.py`:

```python
@dataclass(frozen=True)
class TurnResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    cache_hit: bool
    reused_tokens: int
    seconds: float
    forked: bool = False  # the hit was served by copying a fork slot
```

In `run()`: replace the "Hit/miss is for the log only…" comment and the two reads:

```python
                # Owner-scoped, so the before/after delta is exact: only this
                # session's thread moves these counters, and the turn holds
                # the gateway lock, so nothing else's hit or reset can land
                # between the two reads.
                before = engine.prompt_cache_stats(owner=session.thread)
                ...
                after = engine.prompt_cache_stats(owner=session.thread)
                return TurnResult(
                    ...,
                    cache_hit=after.get("hits", 0) > before.get("hits", 0),
                    forked=after.get("fork_hits", 0) > before.get("fork_hits", 0),
                    reused_tokens=max(0, after.get("reused_tokens", 0) - before.get("reused_tokens", 0)),
                    seconds=time.monotonic() - started,
                )
```

The stall path:

```python
                except GenerationStalled:
                    # ... (existing comment) ...
                    stalled = session.thread
                    self._drop_session()
                    with contextlib.suppress(Exception):
                        engine.reset_prompt_cache(owner=stalled)
                    raise
```

and amend the existing comment's last sentences: "Retire the stalled session's thread too: when the abandoned thread finishes it would publish the KV cache it built on ITS streams, and a cache is usable only from the thread that built it (#34). Retirement makes that late publish drop itself, and leaves the worker's slots alone."

`src/sous/gateway/routes.py` `_log_turn`:

```python
        cache = "fork" if result.forked else "hit" if result.cache_hit else "miss"
        _log(
            f"POST /v1/messages model={_model_label(chat)} stream={int(stream)} status=200 "
            f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
            f"stop={assembler.stop_reason} cache={cache} "
            f"reused_tokens={result.reused_tokens} seconds={result.seconds:.1f}"
        )
```

- [ ] **Step 6: Run the suite, the threaded files twice**

Run: `uv run pytest -m "not model" -q 2>&1 | tail -3 && uv run pytest tests/test_worker.py tests/test_gateway_turn.py -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/worker.py src/sous/gateway/turn.py src/sous/gateway/routes.py tests/test_worker.py tests/test_gateway_turn.py tests/test_gateway_routes.py
git commit -m "feat: the worker retires only its own cache slots; gateway turns report exact reuse

run_task no longer resets the whole prompt cache at entry and exit — that
wiped the gateway's slot every time a delegated task ran between two
subagent turns. It retires its session's thread at the end instead, and
its report is that thread's counters. The gateway reads its own session's
counters before and after a turn, so hit/miss and reused_tokens are exact,
and the log says cache=fork when the turn started from a copied header.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Documentation — README, CONTRIBUTING.md, CLAUDE.md

**Files:**
- Modify: `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`

**Interfaces:** none (docs). Everything stated here must match Tasks 1–6 exactly: `[model].prompt_cache_gb`, `"auto"`, the 4096-token fork floor, `MAX_SLOTS` 16, 2 GiB slack, the `cache=fork` log token, the status fields.

- [ ] **Step 1: README — gateway limitations bullet**

Replace the bullet beginning `- **One turn at a time, one cache slot.**` with:

```markdown
- **One turn at a time; keyed prompt-cache slots.** Local turns are serialized
  behind the same lock as delegated tasks. The prompt cache keeps one slot per
  resident conversation (bounded by `[model].prompt_cache_gb`), so a subagent's
  consecutive turns reuse their own slot, two subagents interleaving reuse
  theirs, and a delegated task running in between no longer evicts anything.
  A new subagent of a type seen before starts from a *fork*: a copy of the
  cache taken where its predecessor's system prompt and tool schemas end
  (~50K tokens for a Claude Code subagent), so its first turn prefills only
  its own brief — seconds instead of minutes. Two subagents still run one at
  a time; batching is a later phase.
```

- [ ] **Step 2: README — log line**

In the paragraph beginning "Each `/v1/messages` turn served locally logs one metadata-only line", change "cache hit/miss" to "cache `hit`/`fork`/`miss` (`fork`: the turn started from a copied header slot)".

- [ ] **Step 3: README — config block and prompt-cache paragraph**

In the `[model]` block of the configuration example, after `prompt_cache = true` add:

```toml
prompt_cache_gb = "auto"  # resident cache slots beyond the running turn; a number of GiB, or 0 for one slot
```

Replace the paragraph beginning "`[model].prompt_cache` (default `true`) reuses one KV cache across the turns of a task" with:

```markdown
`[model].prompt_cache` (default `true`) reuses a KV cache across the turns of a
conversation, prefilling only what the conversation gained instead of the
whole thing every turn. All of a task's generations run on one worker-owned
thread so the cache survives between turns; measured on the default model in
one process, six growing turns took 29.5s warm against 77s cold, with per-turn
time flat instead of growing. Set it to `false` to prefill every turn from
scratch.

`[model].prompt_cache_gb` (default `"auto"`) bounds the caches kept resident
*beyond* the turn that is running: one slot per conversation, plus one *fork*
slot per distinct system prompt long enough to be worth copying (4096 tokens
or more) — the ~50K-token header every Claude Code subagent of one type
shares. `"auto"` is what Metal's recommended working set has left once the
weights, one full context window of KV (the larger of `[model]`'s and
`[gateway]`'s) and 2 GiB of slack are paid for — about 27 GiB on a 64 GB
machine with the default model and gateway window, room for several
conversations. Slots are evicted least-recently-used first when the budget,
a count of 16, or live memory pressure says so; the conversation that just
ran is never evicted by its own turn, so `0` means exactly one slot (the
pre-3a behaviour) and a 32 GB machine degrades to that on its own.
`server_status` reports `prompt_cache` — slots, resident bytes, hits, fork
hits, evictions — counts only.
```

- [ ] **Step 4: README — validation status bullet**

In the "Gateway mode (experimental) serves one local turn at a time on a single-slot prompt cache" bullet, replace "on a single-slot prompt cache" with "on a keyed prompt cache (one slot per conversation plus header forks, budgeted by `[model].prompt_cache_gb`)".

- [ ] **Step 5: CONTRIBUTING.md — whole-session recipe expectations**

Replace the sentences from "Expect every turn here to report `cache=miss`" through "keyed slots that would fix the whole-session case are a later phase." with:

```markdown
Expect the main loop's turns to report `cache=hit` after the first: Claude
Code's small background queries (titles, suggestions) get slots of their own
instead of evicting the main loop's, and each `~80K`-token main turn prefills
only what the conversation gained. A subagent spawned mid-session reports
`cache=fork` on its first turn when an earlier subagent of the same type
already prefilled that header, `cache=miss` otherwise.
```

- [ ] **Step 6: CLAUDE.md — gotchas**

After the `PrefixCache` cold-retry gotcha, add:

```markdown
- Prompt-cache slots (`engine/promptcache.py`) are owned by the thread that
  built them (#34) and looked up only by that thread. `reset_prompt_cache`
  and `prompt_cache_stats` take an `owner`: the worker retires its session's
  thread (`session.thread`) at task end and never calls the bare `reset()`,
  which drops the gateway's slots too. A `fork` slot is a *copy* taken while
  a cold turn prefills past the system header (`fork_point`, 4096-token
  floor); a `turn` slot is *moved* into the turn that extends it. Never
  rewind a cache to make a slot — a hybrid model's recurrent layers cannot.
  `base.measure_cache_budget`/`live_headroom` deliberately do not call
  `release_mlx_thread_state()`: they run on threads whose caches are live.
```

- [ ] **Step 7: Check the docs render and nothing else changed**

Run: `uv run ruff format --check . && git diff --stat`
Expected: only the three docs files changed.

- [ ] **Step 8: Commit**

```bash
git add README.md CONTRIBUTING.md CLAUDE.md
git commit -m "docs: describe keyed prompt-cache slots, header forks and the resident budget

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Phase 3a exit — a hybrid session with two subagents of one type (manual, after merge)

**Files:** none in the repo. Output: a comment on issue #41; a memory note.

**Interfaces:** the merged `main`, reinstalled (`uv tool install --reinstall .`), the unmanaged daemon restarted so it runs the new build. Kyle's `~/.sous/config.toml` already carries `[gateway] enabled = true` and `max_context_tokens = 131072`; `[model].prompt_cache_gb` stays unset (`"auto"`).

The spec gives Phase 3a no exit sentence of its own; this task uses the evidence that motivated it: *a second subagent of the same type starts warm.* Everything is read from `~/.sous/daemon.log` (`sous gateway:` lines) and `server_status`; nothing here needs `/status`.

- [ ] **Step 1: Restart on the new build and confirm the budget**

```bash
uv tool install --reinstall . && sous status
```

Restart the daemon the way it was last started (`exec sous serve >> ~/.sous/daemon.log 2>> ~/.sous/daemon.err.log`, after stopping the old pid from `~/.sous/daemon.lock`). Then run one tiny delegated task (any `delegate_to_local_model` with a one-line instruction) so the model loads, and read `server_status`: `model.prompt_cache` must be present with `slots`, `resident_bytes`, `forks`, `evictions` all `0` after the task (the worker retired its slots). Record `memory_gb` — the automatic budget is `55.66 GB − that − 8 GiB − 2 GiB`; note it.

- [ ] **Step 2: Run the hybrid session (headless, as in the Phase 2 exit)**

In a scratch repo (a two-line README is enough), with no `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY`/tier variables in the environment:

```bash
sous claude -p 'Use an Explore subagent to find where this repo'"'"'s README says "demo" (file and line). When it returns, use a SECOND Explore subagent to count the lines in README.md. Report both results verbatim.' --output-format json --allowedTools "Task,Agent,Read,Glob,Grep,Bash(wc *)"
```

- [ ] **Step 3: Read the gateway log**

Expected shape in `~/.sous/daemon.log`, in order:

| local turn | expected `cache=` | expected `reused_tokens` | expected `seconds` |
|---|---|---|---|
| Explore #1 turn 1 | `miss` | 0 | ~150 (cold; this turn creates the fork) |
| Explore #1 turn 2 | `hit` | ≈ turn 1's `input_tokens` | < 10 |
| Explore #2 turn 1 | **`fork`** | ≈ 50,000 (the header) | **< 20** |
| Explore #2 turn 2 | `hit` | ≈ its turn 1's `input_tokens` | < 10 |

If Explore #2 turn 1 says `miss`, first check `server_status`'s `forks` after Explore #1: `0` means the header was not a token prefix of the render (or the fork was evicted at publish — check `evictions` and `resident_bytes` against the budget); `1` with a miss means the two subagents' system prompts differ — diff the two `input_tokens` and look at Claude Code's agent prompt for a per-spawn element. Report what was found either way; do not paper over it.

- [ ] **Step 4: Interleave a delegated task and confirm nothing was lost**

While the session is idle after Step 2, run one delegated task, then spawn a third Explore in a new `sous claude -p` session with the same shape. Expected: Explore #3 turn 1 is `cache=fork` (the worker's retire left the gateway's fork alone). `server_status` after: `slots` counts the gateway's fork plus its turn slots (up to the budget), none from the worker.

- [ ] **Step 5: Report**

Post a comment on #41 titled "Phase 3a exit: keyed slots, header forks" with: the budget derivation from Step 1's numbers; the table from Step 3 with real values; Step 4's result; `server_status.prompt_cache` at the end; and one sentence on what this leaves for 3b (batching) — the spec asks for serialized latency at N=2–3 to be measured before paying for batching, and this session's Step 3 numbers are the N=2 data point. Update the memory note (`gateway-phase3a-executed`) with the same facts and anything that surprised.

---

## Self-review

**Spec coverage.**
- "The single `(cache, held)` slot becomes an LRU map keyed by conversation prefix" → Task 2 (`Slot.held`, `_take`, LRU `last_used`).
- "with a byte budget reserved out of the generation budget" → Task 4 (`reserve_tokens × kv_bytes_per_token`, subtracted before the budget in `auto_cache_budget`), Task 1 (`auto_cache_budget`), Task 5 (measured after load).
- "oMLX's `_hot_cache_reserved_bytes` pattern: charge `min(cap, used + slack)`" → decision 8 explains the translation: sous has no process enforcer, so the reservation is the fixed `reserve_bytes` and the cap is `max_bytes`; the spec's *effect* (the hot cache never crowds out a generation) is what Tasks 3–4 deliver. Stated as an interpretation, not hidden.
- "shrink under pressure" → Task 3 (`hooks.headroom() < reserve_bytes` loop), Task 4 (`live_headroom`).
- "protect the in-flight conversation" → Task 3 (`_evict(protect=…)` never evicts the slot just published; the in-flight turn's cache is never in the map).
- "Each slot keeps the existing epoch/strict-prefix correctness argument" → Task 2 keeps every existing `PrefixCache` test (epoch, late write-back, thread ownership, replay safety, no-doubling) and adds owner retirement on top.
- The Phase 2 evidence (a second subagent misses on a shared ~57K header) → Tasks 3, 5; verified in Task 8.
- The README promise "Keyed cache slots come later" → Task 7.
- Not in scope, deliberately: batching, mid-generation abort, count-aware admission (all Phase 3b); worker-side forks (the worker's header is below the floor by design).

**Placeholder scan.** Task 6 Step 2's second gateway test is written as "adapt the existing stall test" with the one new assertion spelled out — the existing test's body is in the repo at `tests/test_gateway_turn.py::test_a_stall_drops_the_session_and_the_next_turn_gets_a_new_one`, which the implementer reads; the assertion to add is given. Task 6 Step 2's routes test says "if no existing test pins the log token, add one" — acceptable because it names the file, the token and the capture mechanism; the implementer greps `cache=` first. Task 5 Step 1's VLM variants are described relative to the LM tests with the one differing line given. No "TBD", no "handle edge cases".

**Type consistency.** `PrefixCache.generate(stable_ids, full_ids, max_tokens, on_delta=None, fork_at=0)` — Tasks 2, 3, 5 (`Recording.generate` mirrors it). `stats(owner=None)` / `reset(owner=None)` — Tasks 2, 4 (`Engine`, `ManagedEngine`, `FakeEngine`), 5 (engines), 6 (callers). `Slot(cache, held, owner, kind, nbytes, last_used=…)` — Tasks 2, 3 (`_publish(Slot(...), epoch)`). `measure_cache_budget(reserve_bytes) -> int`, `live_headroom() -> int | None` — Tasks 4, 5. Constructor keywords `cache_budget: int | None`, `reserve_bytes: int` — Tasks 4 (factory, tests), 5 (engines). `TurnResult.forked` — Task 6 (turn, routes, tests). `FakeEngine.reset_owners`/`stats_owners` — Tasks 4, 6. `GenerationSession.thread` — Tasks 4, 6. Stats keys `fork_hits`, `forks`, `evictions`, `slots`, `resident_bytes` — Tasks 1, 2, 3, 4 (status test), 6 (turn), 7 (docs), 8 (report).

**Known judgement calls for the executor to keep, not reopen:** the fork floor (4096), `MAX_SLOTS` (16), the slack (2 GiB), header = `messages[:1]` when `role == "system"`, `max_bytes=0` as the `PrefixCache` default. Each is recorded in the decisions section with its reason.
