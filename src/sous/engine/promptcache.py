"""Cross-turn prefix cache: the rules, the counters, and the orchestrator.

This module imports no mlx, deliberately. The engine tests are model-marked and
never run in CI, so every decision that affects correctness is made here, where
a fake cache layer can exercise it. Array copies arrive through an injected
`copy_array` callable rather than an mx import.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sous.engine.base import Delta, OnDelta, ReplaySafe


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
    # of milliseconds on every lookup. Both operands are lists in practice
    # (that is what the memo and every Slot hold), and re-wrapping them would
    # add two ~50K-element copies per candidate to save a type check.
    head = new_ids[:n] if isinstance(new_ids, list) else list(new_ids[:n])
    return n if head == (cached_ids if isinstance(cached_ids, list) else list(cached_ids)) else 0


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """How many leading tokens `a` and `b` share. The miss diagnostic: with
    the fork boundaries known, where a new render stopped agreeing with the
    closest resident slot says whether the tool array or the system text
    differed — without keeping a token of either.

    A plain scan, on purpose. A binary search over slice compares looks
    smarter but each slice copies and INCREFs every element it covers, and on
    two ~57K-token lists it measured 2.5 ms against 0.6 ms for this loop
    (Python 3.14, M5 Pro); the C-level `map(ne)`/`compress` form was no faster.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def all_trimmable(cache: Sequence[Any]) -> bool:
    # Sequence[Any] rather than a Protocol: mlx-lm and mlx-vlm ship their own
    # cache classes with no shared base and no type stubs, and trim/offset exist
    # only on the layers that rewind. A precise type here would be a fiction no
    # real class satisfies, and ty would reject the attribute access.
    """Whether every layer can rewind, so the cache needs no state copy.

    Asks `is_trimmable()` rather than testing for a `trim` attribute:
    RotatingKVCache owns a `trim` but reports False once its window has
    wrapped, and trimming it then would desync its ring index. Re-asked every
    turn for the same reason — the answer can change mid-conversation.
    """
    if not cache:
        return False
    return all(getattr(c, "is_trimmable", lambda: False)() for c in cache)


def snapshot(cache: Sequence[Any], copy_array: Callable) -> tuple[list, int]:
    """Record what it takes to put `cache` back exactly as it is now.

    A layer that can rewind needs only its offset — an integer. A layer that
    cannot needs its state copied, and those are precisely the recurrent layers
    whose state does not grow with sequence length. That asymmetry is why a
    hybrid cache forks for a fixed cost instead of a copy of the whole KV.

    Returns the snapshot and the bytes copied, for the turn's accounting.
    """
    out: list = []
    nbytes = 0
    for c in cache:
        if getattr(c, "is_trimmable", lambda: False)():
            out.append(("trim", int(getattr(c, "offset", 0) or 0)))
            continue
        # mlx cache classes carry bookkeeping in meta_state as well as state.
        # ArraysCache — today's only non-trimmable class — inherits "" from
        # _BaseCache, so copying state alone is a complete snapshot. Assert
        # that stays true: a future non-trimmable cache with real meta_state
        # would otherwise restore wrongly and silently, and the warm-retry
        # path even absorbs this raise into a cold run rather than failing
        # the turn, so this check is cheap insurance, not a new failure mode.
        meta_state = getattr(c, "meta_state", "")
        assert not meta_state, (
            f"{type(c).__name__}.meta_state is non-empty; snapshot()/restore() "
            "must be extended to copy it, not just state"
        )
        copies = [None if a is None else copy_array(a) for a in c.state]
        nbytes += sum(getattr(a, "nbytes", 0) for a in copies if a is not None)
        out.append(("state", copies))
    return out, nbytes


def restore(cache: Sequence[Any], snap: Sequence[tuple], copy_array: Callable) -> None:
    """Put `cache` back to where `snapshot` was taken."""
    # strict=True: the snapshot was built from this cache layer by layer, so a
    # length mismatch is a bug rather than a case to tolerate.
    for c, (kind, value) in zip(cache, snap, strict=True):
        if kind == "trim":
            current = int(getattr(c, "offset", 0) or 0)
            if current > value:
                c.trim(current - value)
            continue
        c.state = [None if a is None else copy_array(a) for a in value]


def trim_to(cache: Sequence[Any], n_tokens: int) -> None:
    """Rewind every layer to hold exactly `n_tokens`. All layers must rewind."""
    for c in cache:
        current = int(getattr(c, "offset", 0) or 0)
        if current > n_tokens:
            c.trim(current - n_tokens)


# A header shorter than this is not worth a fork slot: prefilling it costs
# under a second on the default model, and a fork is a full second copy of its
# KV. A ~2K-token system prompt never qualifies; a Claude Code
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


# Probe conversations, each rendered with the turn's tools. The header pair
# keeps the real system turn and differs only in the first user turn's
# content, so their renders share exactly the system block — the header. The
# tools pair keeps the user turn fixed and differs at the first character of
# the system text, so their renders share whatever the template emits BEFORE
# the client's text: on a template that renders tools first (Qwen3.5/3.8) the
# whole tool block — the ~45–56K tokens every Claude Code session presenting
# the same tool set shares, across sessions and projects — and on one that
# renders the system text first (Qwen3) a few tokens, which fork_point drops.
_HEADER_PROBES = ({"role": "user", "content": "0"}, {"role": "user", "content": "1"})
_TOOLS_PROBES = (
    ({"role": "system", "content": "0"}, {"role": "user", "content": "0"}),
    ({"role": "system", "content": "1"}, {"role": "user", "content": "0"}),
)


def probe_boundaries(
    render: Callable[[list[dict]], str],
    encode: Callable[[str], list[int]],
    memo: PromptMemo,
    system: dict,
    stable_ids: Sequence[int],
) -> list[int]:
    """Every boundary a turn may fork at, ascending: the tools boundary and the
    header boundary, each the common prefix of a probe pair's renders, encoded
    once (memoized by text, one slot each) and verified by `fork_point`
    against this turn's own render. A candidate that fails the check is
    dropped and equal candidates merge. A request with no system text renders
    no separator before it on the default template, so the tools text is then
    not a prefix and drops out on its own.

    Found from probe pairs rather than from partial renders: [model].id
    accepts any chat template, and a template may refuse a message list with
    no user turn outright (the default model's does:
    `raise_exception("No user query found in messages.")`). The caller holds
    its tokenize lock — `render` and `encode` both run here.
    """
    found: set[int] = set()
    pairs = (
        ("tools", [list(conversation) for conversation in _TOOLS_PROBES]),
        ("header", [[system, user] for user in _HEADER_PROBES]),
    )
    for slot, conversations in pairs:
        text = os.path.commonprefix([render(conversation) for conversation in conversations])
        ids = memo.get(slot, text)
        if ids is None:
            ids = encode(text)
            memo.put(slot, text, ids)
        found.add(fork_point(ids, stable_ids))
    found.discard(0)
    return sorted(found)


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

    Callers copy only a cache that has been prefilled — a fork after at least
    FORK_MIN_TOKENS tokens, a taken slot holding at least one — so every
    layer is populated (an empty KVCache's `state` getter would raise).
    Nested-tuple states (QuantizedKVCache) are not handled; sous never
    quantizes its KV.
    """
    for s, d in zip(src, dst, strict=True):
        d.state = [None if a is None else copy_array(a) for a in s.state]
        d.meta_state = s.meta_state


def metal_headroom(*, working_set: int, active: int) -> int:
    """Bytes this process can still put on the GPU without Metal paging its own
    buffers: the device's recommended working set minus what mlx holds. The
    question the pressure valve's first half asks — room for one more window.
    Process-local on purpose: whether the rest of the machine is squeezed is
    the kernel's call (see `CacheHooks.pressure`), not an arithmetic one; the
    psutil `available` figure that used to stand in for it counts a freshly
    loaded model's file-backed pages as used and evicted the forks a cold turn
    had just made."""
    return max(0, working_set - active)


# macOS's own memory-pressure classifier (`kern.memorystatus_vm_pressure_level`):
# 1 normal, 2 warn, 4 critical. Read after each publish; the kernel updates it on
# its own cadence, not in step with our frees, so the valve acts once per
# publish rather than looping until it clears.
KERNEL_PRESSURE_WARN = 2
KERNEL_PRESSURE_CRITICAL = 4


def auto_cache_budget(*, working_set: int, active: int, reserve_bytes: int) -> int:
    """Bytes resident slots may hold beyond the in-flight turn: what Metal
    serves without paging, minus what mlx already holds (the weights, when
    read at load), minus the largest cache one turn can build (`reserve_bytes`
    — the window times the KV cost of a token), minus a fixed slack. This is
    the reserve taken out of the generation budget: the turn's own cache is
    paid for first, slots get what is left."""
    return max(0, working_set - active - reserve_bytes - CACHE_BUDGET_SLACK)


# Per-turn readings for the endpoint's turn line, reset by begin_turn() at the
# top of every generate() and again when a failed warm attempt retries cold.
TURN_GAUGES = frozenset(
    {
        "prefilled_tokens",
        "took_len",
        "took_kind",
        "bound_lo",
        "bound_hi",
        "probe_seconds",
        "prefill_seconds",
        "decode_seconds",
        "persist_seconds",
        "restore_seconds",
    }
)
_GAUGES = frozenset({"snapshot_bytes", "miss_lcp"}) | TURN_GAUGES


def without_turn_gauges(stats: dict) -> dict:
    """The `prompt_cache` block every status document carries, with the
    per-turn gauges dropped. They mean something only on the turn line, where
    they describe that turn: daemon-wide they are a max over every owner ever
    seen, beside counters that span every turn — numbers no reader can use,
    paid for in the frontier model's tokens."""
    return {k: v for k, v in stats.items() if k not in TURN_GAUGES}


# The status document's `prompt_cache.disk` block when no store was built:
# off by config, or an engine constructed without a directory (every test).
DISK_OFF: dict = {
    "state": "off",
    "reason": None,
    "forks": 0,
    "bytes": 0,
    "budget_bytes": 0,
    "evictions": 0,
}


# The clock the per-turn timers read. A module attribute rather than a bare
# time.monotonic so a test can swap in a hand-advanced one and assert exact
# phase durations; Slot.last_used keeps the real clock.
_clock = time.monotonic


@dataclass
class PromptCacheStats:
    hits: int = 0  # every warm run, from a turn slot or a fork copy
    misses: int = 0
    reused_tokens: int = 0
    snapshot_bytes: int = 0
    cold_retries: int = 0
    fork_hits: int = 0  # the subset of hits served by copying a fork slot
    forks: int = 0  # fork slots created
    evictions: int = 0  # every slot dropped, for budget, count or pressure
    # Gauge, assigned per miss: how many leading tokens the missed render
    # shared with the closest slot this owner held (0 when it held none).
    miss_lcp: int = 0
    # The subset of `evictions` the pressure valve took (Metal headroom short
    # of a window, or the kernel at warn/critical) — what says, after a
    # restart, whether pressure or the byte cap emptied the map.
    pressure_evictions: int = 0
    # Turn-slot takes that copied the slot and left it in the map, and those
    # that removed it and adopted its arrays because the budget could not
    # hold a copy beside the live cache. Both are hits; the split says
    # whether the take left the predecessor in place for a branch of the
    # conversation to start from — a later eviction (`evictions`,
    # `pressure_evictions`) can still take it before one arrives.
    retained: int = 0
    moved: int = 0
    # Disk forks: files written, files read back, turns served from one,
    # and restores the memory readings refused (the file stays).
    persists: int = 0
    restores: int = 0
    disk_hits: int = 0
    restore_skips: int = 0
    # Per-turn gauges, zeroed by begin_turn() at the top of every generate()
    # and assigned as the turn runs, so a turn that bypasses _run (cache off,
    # render not a strict prefix) reports zeros rather than the last turn's.
    prefilled_tokens: int = 0  # stable tokens this turn had to prefill
    took_len: int = 0  # length of the slot _take returned; 0 on a miss
    # What kind of slot that was, as the turn line names it: "turn" (copied
    # and left in place), "turn-moved" (removed, its arrays adopted), "fork";
    # "" on a miss, and after a cold retry, which took no slot in the end.
    took_kind: str = ""
    bound_lo: int = 0  # the probe's lower boundary (tools), 0 when absent
    bound_hi: int = 0  # the probe's upper boundary (header), 0 when absent
    probe_seconds: float = 0.0
    prefill_seconds: float = 0.0  # every hooks.prefill, plus fork copies and the snapshot
    decode_seconds: float = 0.0  # hooks.decode, plus the restore
    persist_seconds: float = 0.0  # writing the live cache at a boundary; never prefill time
    restore_seconds: float = 0.0  # reading a fork off disk; never prefill time

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def begin_turn(self) -> None:
        for f in dataclasses.fields(self):
            if f.name in TURN_GAUGES:
                setattr(self, f.name, f.default)

    def add(self, other: PromptCacheStats) -> None:
        """Fold `other`'s counters into these.

        Every field is a running count and sums — except the gauges
        (`_GAUGES`), which are assigned per turn (`snapshot_bytes`, the last
        non-trimmable turn's copy cost; `miss_lcp`, the last miss's shared
        prefix; and the per-turn phase timings and slot geometry below).
        Summing a gauge would make the daemon-wide view grow with the number
        of owners rather than report a reading, so the largest wins.
        """
        for f in dataclasses.fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            setattr(self, f.name, max(mine, theirs) if f.name in _GAUGES else mine + theirs)


_MEMO_SLOTS = ("stable", "full", "header", "tools")


class PromptMemo:
    """One slot per render, keyed by the exact prompt text.

    The header and tools slots hold the two fork boundaries' texts: what the
    header probe pair's renders have in common (the whole system block) and
    what the tools pair's have (the tool block, on a template that renders
    tools first) — see `probe_boundaries`. Two slots, because the two texts
    would evict each other from one on every cold turn.

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


class CacheHooks(Protocol):
    """The things only an engine can do. Everything else is shared.

    `eval_cache` materialises a cache's arrays on the calling thread (a fork
    copy is lazy, and the pressure valve reads active memory). `persist`
    writes the live working cache at a boundary to `path` — called on the
    owner thread, holding no lock, never with a Slot; `restore` fills a fresh
    cache from `path` on the calling thread and evaluates it, raising the
    store's verification error on any mismatch."""

    def new_cache(self) -> list: ...
    def prefill(self, cache: list, token_ids: list[int]) -> None: ...
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None
    ) -> str: ...
    def copy_array(self, a: object) -> object: ...
    # Bytes the machine can still give a cache right now, or None when it
    # cannot tell (no mlx). Read on the owner thread; must not release thread
    # state. Called with no cache lock held and must take none of its own: the
    # cache's promise that reset() never waits on a generation rests on its
    # bookkeeping lock covering list and dict work plus the deallocation a
    # drop triggers — all bounded — and this is the one engine call the
    # eviction path makes.
    def headroom(self) -> int | None: ...
    # The kernel's memory-pressure level (KERNEL_PRESSURE_WARN / _CRITICAL, or
    # 1 for normal), or None where it cannot be read. Same rules as headroom():
    # owner thread, no cache lock held, no lock of its own, no thread-state
    # release.
    def pressure(self) -> int | None: ...
    def eval_cache(self, cache: list) -> None: ...
    def persist(
        self, cache: list, path: Path, ids: list[int], metadata: dict[str, str]
    ) -> None: ...
    def restore(self, path: Path, header: Any, cache: list, ids: list[int]) -> None: ...


class StoreHooks(Protocol):
    """The disk store as the orchestrator sees it (engine/forkstore.ForkStore
    in production, a fake in tests). Files are shared across owner threads;
    the arrays a restore allocates belong to the restoring thread, so
    residency stays owner-scoped while the file is not."""

    state: str

    def has(self, ids: Sequence[int]) -> bool: ...
    def longest_prefix(self, stable_ids: Sequence[int]) -> Any: ...
    def persist(
        self,
        ids: Sequence[int],
        boundary: str,
        write: Callable[[Path, dict[str, str]], None],
        *,
        expected_bytes: int,
    ) -> bool: ...
    def restore(self, entry: Any, load: Callable[[Path, Any], None]) -> bool: ...
    def touch(self, ids: Sequence[int]) -> None: ...
    def status(self) -> dict: ...


@dataclass
class Slot:
    """One resident cache and the exact ids it holds, at the stable boundary.

    `owner` is the thread whose mlx streams built the arrays (issue #34): the
    Thread object, not its ident — idents are recycled after a thread exits,
    a strongly held Thread object cannot falsely match a later thread. `kind`
    is "turn" (published when a turn ends; copied and left in place by the
    turn that extends it when the budget holds the copy, otherwise removed and
    its arrays adopted) or "fork" (a copy taken at a shared-prefix boundary;
    copied on every hit, left in place). Which turn slots are one
    conversation at different lengths is read off `held` (`_ancestors`),
    never recorded. A fork's file on disk, when the store keeps one, is
    shared across owners; only the arrays are owner-scoped."""

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
    strict prefix of its stable render. A "turn" slot is copied and left in
    place when the budget can hold the copy — so a branch of the same
    conversation (a progress-summary call) still finds it — and moved, its
    arrays extended in place, when it cannot; a "fork" slot is always copied,
    so the next conversation sharing that prefix finds it too.
    Everything published is charged to `max_bytes`; the in-flight
    turn's own cache never is — `reserve_bytes` (one full window of KV) was
    subtracted from the machine's headroom before `max_bytes` was derived.

    Disk forks (`store`) are the one thing here shared across owners: a file
    has no thread affinity, and every restore allocates arrays of its own on
    the restoring thread. The owner rule is about arrays, not content.
    """

    def __init__(
        self,
        hooks: CacheHooks,
        enabled: bool = True,
        *,
        max_bytes: int = 0,
        reserve_bytes: int = 0,
        store: StoreHooks | None = None,
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
        self._store = store
        # Held only across list and dict work, plus the deallocation a drop
        # triggers — never across prefill or decode. reset() must never wait
        # on a generation (its caller may be the turn runner's stall path
        # while that generation still runs), and freeing a cache is bounded, so
        # with this discipline it never does.
        self._lock = threading.Lock()
        self._slots: list[Slot] = []
        self._owner_stats: dict[threading.Thread, PromptCacheStats] = {}
        # Counters of owners already retired or swept, so the daemon-wide view
        # keeps their history without keeping their Thread objects.
        self._history = PromptCacheStats()
        # Owners whose late publishes are refused: a session thread dropped
        # after a stall. Weak, so a
        # thread that is truly gone costs nothing to remember.
        self._retired: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
        self._epoch = 0

    # ---- bookkeeping (the helpers up to _take want self._lock held; each of
    # the entry points after it says in its docstring what it does with the
    # lock, and _evict_pressure must NOT be called holding it) ---------------

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

    def _evictable(self, protect: Sequence[Slot]) -> list[Slot]:
        # Identity, not equality: Slot is a dataclass, and `==` would compare
        # two ~50K-token id lists (and two caches) per candidate.
        return [s for s in self._slots if not any(s is p for p in protect)]

    def _holds(self, slot: Slot) -> bool:
        return any(s is slot for s in self._slots)

    def _remove(self, slot: Slot) -> None:
        self._slots = [s for s in self._slots if s is not slot]

    def _drop(self, victim: Slot, *, pressure: bool = False) -> None:
        """Take `victim` out of the map and charge its owner. The caller's
        reference is the last one: the slot's cache is freed the moment that
        frame lets go, which is what the pressure re-read in _evict_pressure
        relies on."""
        self._remove(victim)
        stats = self._stats_for(victim.owner)
        stats.evictions += 1
        if pressure:
            stats.pressure_evictions += 1

    def _drop_lru(self, protect: Sequence[Slot], *, pressure: bool = False) -> None:
        self._drop(min(self._evictable(protect), key=lambda s: s.last_used), pressure=pressure)

    def _evict_caps(self, protect: Sequence[Slot]) -> None:
        """Bring the map under its count and byte caps, never touching a slot
        in `protect`. On its own before a turn starts, `protect` is empty:
        nothing here is spoken for."""
        while self._evictable(protect) and (
            len(self._slots) > MAX_SLOTS or self._resident() > self.max_bytes
        ):
            self._drop_lru(protect)

    def _make_room(
        self, nbytes: int, protect: Sequence[Slot] = (), *, require: Slot | None = None
    ) -> bool:
        """Evict LRU until a slot of `nbytes` would fit under the byte cap, and
        say whether it now does.

        For the fork copy, which is charged before it is allocated rather than
        after: publishing first and evicting afterwards would put the copy and
        whatever it displaces on the machine at the same time — the transient
        doubling that must never happen. `protect` is what this turn must
        keep — the forks it already published, or the turn slot it is about
        to copy: a copy is never made room for by evicting what it copies or
        sits beside, and when the budget cannot hold both the answer is False
        and the caller skips the copy rather than evicting into it. The
        turn's own cache is not in the map and needs no protecting.
        A copy that cannot fit beside what `protect` alone holds is refused
        up front, before any eviction runs — otherwise a copy that can never
        fit would empty the map of everything evictable for nothing.

        `require`, when given, must still be in the map, else False before any
        eviction: the lock was released after the take, and the endpoint can
        retire the owner meanwhile (`reset(owner)` for a generation it
        abandoned as stalled), emptying its slots. That turn's publish will
        be refused, and a copy for it is not worth making room for.

        Takes the lock itself.
        """
        with self._lock:
            if require is not None and not self._holds(require):
                return False
            protected_bytes = sum(s.nbytes for s in protect)
            if protected_bytes + nbytes > self.max_bytes:
                return False
            while self._evictable(protect) and self._resident() + nbytes > self.max_bytes:
                self._drop_lru(protect)
            return self._resident() + nbytes <= self.max_bytes

    def _take(self, owner: threading.Thread, stable_ids: list[int]) -> Slot | None:
        """The longest slot `owner` holds that `stable_ids` strictly extends.
        Nothing leaves the map here: a fork slot is copied by the caller, and
        whether a turn slot is copied or moved is decided in `generate`, where
        the budget is visible."""
        best: Slot | None = None
        for s in self._slots:
            if s.owner is not owner or not reuse_length(s.held, stable_ids):
                continue
            if best is None or len(s.held) > len(best.held):
                best = s
        if best is None:
            return None
        best.last_used = time.monotonic()
        return best

    def _fork_boundaries(
        self,
        stats: PromptCacheStats,
        owner: threading.Thread,
        stable_ids: list[int],
        fork_at: Sequence[int] | Callable[[], Sequence[int]],
        reuse: int,
    ) -> list[tuple[int, bool, bool]]:
        """The boundaries this turn will stop at, ascending, each with what
        it wants there: a resident copy (`need_slot` — this owner holds no
        fork with exactly those ids, and there is a budget to charge one to)
        and/or a file (`need_file` — the store has none for those ids; only
        the lowest boundary is ever written, the one the next session shares).
        A boundary is returned when either is true. The candidates are those
        this turn prefills to (`reuse <= b`: a turn that starts exactly at a
        boundary — a disk restore, or a fork the valve took — republishes
        there through the same path, with an empty prefill), that leave
        something to prefill after them (`b < len(stable_ids)`) and that
        clear the fork floor. Nothing is resolved when there is neither a
        budget nor a store: at `max_bytes == 0` without a store nothing about
        the turn changes.

        The probe is resolved on warm turns too. With one boundary that was
        waste — a turn that started at or past it could not fork there — but
        with two it is the point: a session's first turn starts warm at
        another session's tools fork (`reuse` is the tools boundary) and must
        still publish its own header fork, or that session's next subagent
        reuses ~45K tokens instead of ~57K. The memo makes a warm probe four
        renders and a string compare; the encode it saves is the cost. A warm
        attempt that fails and is retried cold resolves the probe twice — two
        extra renders under the tokenize lock, memoized encode — so the cost
        is on record.

        Resolved before the lock is taken: the probe tokenizes, and nothing
        that slow may run under the bookkeeping lock that reset() promises
        never to wait behind. Takes the lock itself, for the scan.
        """
        if self.max_bytes <= 0 and self._store is None:
            return []
        if isinstance(fork_at, Sequence):
            candidates = list(fork_at)
        else:
            started = _clock()
            try:
                candidates = list(fork_at())
            except Exception as e:
                # An optimization must never fail a turn — the same rule the
                # warm-retry path follows. A chat template is free to refuse
                # a probe conversation; the turn just does not fork.
                warnings.warn(
                    f"sous prompt cache: fork probe failed ({type(e).__name__}); "
                    "not forking this turn",
                    stacklevel=2,
                )
                return []
            finally:
                stats.probe_seconds += _clock() - started
        inside = sorted({b for b in candidates if b >= FORK_MIN_TOKENS and b < len(stable_ids)})
        if len(inside) >= 2:
            stats.bound_lo, stats.bound_hi = inside[0], inside[-1]
        elif inside:
            stats.bound_lo, stats.bound_hi = 0, inside[0]
        wanted = sorted({b for b in inside if reuse <= b})
        if not inside:
            return []
        with self._lock:
            forks = [s.held for s in self._slots if s.owner is owner and s.kind == "fork"]
        # The lowest boundary's file, when it exists, is touched on every
        # turn that passes or starts at or above it: a resident fork's file
        # is otherwise the oldest in the store and the first the LRU takes.
        store = self._store if self._store is not None and self._store.state == "active" else None
        lowest = inside[0]
        on_disk = store is not None and store.has(stable_ids[:lowest])
        if on_disk:
            store.touch(stable_ids[:lowest])
        out: list[tuple[int, bool, bool]] = []
        for b in wanted:
            need_slot = self.max_bytes > 0 and stable_ids[:b] not in forks
            need_file = store is not None and b == lowest and not on_disk
            if need_slot or need_file:
                out.append((b, need_slot, need_file))
        return out

    def _publish(self, slot: Slot, epoch: int, protect: Sequence[Slot] = ()) -> bool:
        """Add `slot` unless the world moved on: a full reset since the turn
        began, or the owner retired (its session was dropped) while it generated.

        The caps are applied under the same lock as the append — nothing must
        ever observe the map over budget — but the pressure pass runs after
        the lock is released, because it asks the engine for a reading. Both
        protect `slot` and `protect` (the forks this turn published before
        it): a slot is never evicted by its own publish, so on a machine with
        room for exactly one slot that one survives — and a cold turn's second
        fork never evicts its first.

        The ancestor drop runs under the same lock, before the caps, so the
        caps see the map the conversation actually needs.
        """
        keep = (slot, *protect)
        with self._lock:
            if epoch != self._epoch or slot.owner in self._retired:
                return False
            self._slots.append(slot)
            if slot.kind == "turn":
                self._retire_ancestors(slot, keep_longest=True)
            self._evict_caps(protect=keep)
        self._evict_pressure(protect=keep)
        return True

    def _ancestors(self, slot: Slot) -> list[Slot]:
        """This owner's turn slots whose ids `slot`'s strictly extend: the
        same conversation at shorter lengths. Read off the map at every use
        rather than recorded at the take, so a chain broken by a cold retry
        or a moved slot needs no repair. A fork is a prefix of every turn
        slot of its session and never counts; a branch (a progress-summary
        call takes the same prefix down a different last turn) extends
        nothing the real conversation's next turn publishes, and vice versa,
        so neither is the other's ancestor. Lock held by the caller."""
        return [
            s
            for s in self._slots
            if s.kind == "turn" and s.owner is slot.owner and reuse_length(s.held, slot.held)
        ]

    def _retire_ancestors(self, slot: Slot, *, keep_longest: bool) -> None:
        """Drop `slot`'s ancestors, charged to `evictions` (neither the budget
        nor pressure asked for it), keeping the longest when asked. Every
        retained take leaves the predecessor in place, so without this a
        ten-turn subagent would hold ten copies of itself: a turn retires
        the lengths below the slot it takes as it takes it — their bytes
        fund its copy, rather than another conversation's slot — and its
        publish keeps the slot it took beside the one it adds, so a linear
        conversation holds at most its current and previous lengths. Lock
        held by the caller."""
        ancestors = self._ancestors(slot)
        longest = max(ancestors, key=lambda s: len(s.held)) if keep_longest and ancestors else None
        for s in ancestors:
            if s is not longest:
                self._drop(s)

    def _evict_pressure(self, protect: Sequence[Slot]) -> None:
        """Shrink the map when the machine says so, never touching a slot in
        `protect` — the slot just published and this turn's earlier forks.

        Two readings, two questions. Metal's headroom asks whether this
        process can still hold one more window: while it cannot, drop LRU one
        at a time and re-read. The kernel's pressure level asks whether the
        rest of the machine is being squeezed: at warn, drop one slot per
        publish, so the map stops growing and shrinks by one slot for each
        slot a turn adds (up to three on a cold turn — two forks and the
        turn slot — one on a warm turn); at critical, drop everything
        unprotected. Once per publish, not until
        the level clears — the kernel re-evaluates on its own cadence, and a
        loop on it would empty the map before it could answer.

        Takes the lock one drop at a time instead of holding it across the
        loop, because `headroom()` and `pressure()` are engine calls (an mlx
        memory query and a sysctl, on a real engine). The bookkeeping lock
        covers list and dict work plus the deallocation a drop triggers, all
        bounded; that is the whole reason reset() can promise never to wait
        on a generation, and it would stop being true the moment an engine
        call ran underneath it. Dropping the lock between readings is safe:
        every iteration re-reads both the reading and the map, so a
        concurrent publish or reset just changes what the next pass sees.
        """
        while (headroom := self._hooks.headroom()) is not None and headroom < self.reserve_bytes:
            with self._lock:
                if not self._evictable(protect):
                    return
                self._drop_lru(protect, pressure=True)
        level = self._hooks.pressure()
        if level is None or level < KERNEL_PRESSURE_WARN:
            return
        with self._lock:
            victims = len(self._evictable(protect)) if level >= KERNEL_PRESSURE_CRITICAL else 1
            for _ in range(victims):
                if not self._evictable(protect):
                    return
                self._drop_lru(protect, pressure=True)

    def _plant(self, cache: list, held: list[int], kind: str = "turn") -> None:
        """Test seam: publish a slot for the calling thread directly."""
        with self._lock:
            self._slots.append(
                Slot(cache, list(held), threading.current_thread(), kind, slot_bytes(cache))
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
        unload uses this form; the endpoint's stall path retires
        an owner instead.

        Takes only the bookkeeping lock, never the generation lock: the caller
        may be the turn runner recovering from a stall while that generation
        still holds the other one, and waiting there would wedge the next turn.
        """
        with self._lock:
            if owner is None:
                # `_retired` is deliberately kept: a retired owner stays
                # retired for the rest of its life. Every owner is a session
                # thread the endpoint has dropped — never reused,
                # so there is nothing to un-retire and much to get wrong.
                self._epoch += 1
                self._slots = []
                self._owner_stats = {}
                self._history = PromptCacheStats()
                return
            self._slots = [s for s in self._slots if s.owner is not owner]
            self._history.add(self._owner_stats.pop(owner, PromptCacheStats()))
            self._retired.add(owner)

    def stats(self, owner: threading.Thread | None = None) -> dict:
        """Counters plus `slots`, `resident_bytes` and `disk`: for one owner
        thread, or daemon-wide (every live owner plus the history of retired
        ones)."""
        disk = self._store.status() if self._store is not None else dict(DISK_OFF)
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
                "disk": disk,
            }

    def _copy_of(
        self, stats: PromptCacheStats, cache: list, what: str, fallback: str
    ) -> list | None:
        """A fresh cache holding what `cache` holds, or None when the copy
        failed — `what` names the copy and `fallback` what the caller does
        without it, for the warning. An optimization must never fail a turn,
        the rule the warm-retry and probe paths follow too; a half-built copy
        dies here, before the caller allocates anything in its place. Prefill
        time, like the fork copies in `_run`: the copy is part of building
        the cache the turn runs on."""
        hooks = self._hooks
        started = _clock()
        try:
            copy = hooks.new_cache()
            fork_copy(cache, copy, hooks.copy_array)
            hooks.eval_cache(copy)
        except Exception as e:
            warnings.warn(
                f"sous prompt cache: {what} ({type(e).__name__}); {fallback}", stacklevel=3
            )
            return None
        finally:
            stats.prefill_seconds += _clock() - started
        return copy

    def _persist(self, stats: PromptCacheStats, cache: list, ids: list[int], boundary: str) -> None:
        """Write the live cache, which stands exactly at `ids`' end, to the
        store — before any resident copy, so the machine holds the live cache
        alone across the write, and outside the prefill timer, so the turn
        line's prefill rate stays true. Charged to nothing: the live cache
        is the turn's own. Its own handler, because a cold turn's generate
        re-raises rather than retrying and a disk error must never fail a
        viable prefill."""
        if self._store is None:
            return
        hooks = self._hooks
        started = _clock()
        try:
            # The padded buffers ARE what gets written (persist_cache saves
            # the whole capacity, not the trimmed offset slice): slot_bytes
            # plus the ids tensor (int32) plus a fixed allowance for the
            # safetensors header.
            expected_bytes = slot_bytes(cache) + 4 * len(ids) + (64 << 10)
            if self._store.persist(
                ids,
                boundary,
                lambda path, metadata: hooks.persist(cache, path, ids, metadata),
                expected_bytes=expected_bytes,
            ):
                stats.persists += 1
        except Exception as e:  # noqa: BLE001 — an optimization must never fail a turn
            warnings.warn(
                f"sous prompt cache: fork persist failed ({type(e).__name__}); "
                "continuing without one",
                stacklevel=2,
            )
        finally:
            stats.persist_seconds += _clock() - started

    def _restore_from_disk(
        self, stats: PromptCacheStats, stable_ids: list[int]
    ) -> tuple[list, int] | None:
        """The longest stored prefix of `stable_ids`, read into a fresh cache
        on this thread — the turn's own working cache, uncharged, covered by
        the reserve — or None for a plain miss.

        The one place a whole fork is allocated in one step, so the two
        readings the valve uses are consulted first: while Metal's headroom
        is short of the file, drop least-recently-used resident slots (a miss
        took nothing, so nothing is protected); if it is still short, or the
        kernel is at critical, skip — the file stays, the turn prefills
        cold. Residency is not made here: the boundary loop republishes a
        resident fork at the restored length through its charged path.
        Runs before the probe is resolved, so a turn warm at a restored
        tools fork still publishes its own header fork."""
        store = self._store
        if store is None or store.state != "active":
            return None
        entry = store.longest_prefix(stable_ids)
        if entry is None:
            return None
        hooks = self._hooks
        started = _clock()
        try:
            while (headroom := hooks.headroom()) is not None and headroom < entry.size:
                with self._lock:
                    if not self._evictable(()):
                        break
                    self._drop_lru((), pressure=True)
            headroom = hooks.headroom()
            level = hooks.pressure()
            if (headroom is not None and headroom < entry.size) or (
                level is not None and level >= KERNEL_PRESSURE_CRITICAL
            ):
                stats.restore_skips += 1
                return None
            cache = hooks.new_cache()
            ids = list(stable_ids[: entry.n])
            if not store.restore(
                entry, lambda path, header: hooks.restore(path, header, cache, ids)
            ):
                return None
            stats.restores += 1
            return cache, entry.n
        except Exception as e:  # noqa: BLE001 — an optimization must never fail a turn
            warnings.warn(
                f"sous prompt cache: fork restore failed ({type(e).__name__}); prefilling cold",
                stacklevel=2,
            )
            return None
        finally:
            stats.restore_seconds += _clock() - started

    def generate(
        self,
        stable_ids: list[int],
        full_ids: list[int],
        max_tokens: int,
        on_delta: OnDelta | None = None,
        fork_at: Sequence[int] | Callable[[], Sequence[int]] = (),
    ) -> str:
        """`fork_at` names the boundaries this turn may leave a fork slot at —
        token counts into `stable_ids`, in any order — or a callable that
        computes them. The callable form is the engines' probe: it renders and
        tokenizes, so it is deferred until `_fork_boundaries` knows there is a
        budget to fork into or a store to write to."""
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
            stats.begin_turn()
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
            # Nothing was looked up, so the gauge must not keep an earlier
            # miss's reading for the endpoint to log as this turn's.
            stats.miss_lcp = 0
            warnings.warn(
                "sous prompt cache: full prompt is not the stable render "
                "plus a generation suffix; decoding cold this turn",
                stacklevel=2,
            )
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)

        with self._lock:
            # Owner-filtered: the arrays live only on the publishing thread's
            # mlx streams (issue #34), so a different session thread (one the
            # endpoint replaced after a stall, say) gets a cold miss rather
            # than a warm run doomed to the cross-thread mlx failure.
            slot = self._take(owner, stable_ids)
            # The miss diagnostic is measured against the same slots lookup
            # just refused: a slot another thread owns could never have
            # served this turn, so it says nothing about what was available.
            # Only the references are taken here — the scan itself runs after
            # the lock is released (below): ~0.6 ms per 57K-token slot is not
            # the list-and-dict work this lock is scoped to, and a slot's
            # `held` is never mutated after publish, so reading it unlocked
            # is safe even if the map changes meanwhile. A fork take collects
            # them too: its copy can still fail and turn this into a miss.
            candidates = (
                [s.held for s in self._slots if s.owner is owner]
                if slot is None or slot.kind == "fork"
                else []
            )
            if slot is not None and slot.kind == "turn":
                # The lengths below the taken slot are what this turn's
                # publish would retire anyway (it keeps the slot it took
                # beside the one it adds): gone now, their bytes fund the
                # copy below instead of another conversation's slot.
                self._retire_ancestors(slot, keep_longest=False)
            # Whatever is still in the map is not going to serve this turn,
            # and this turn's own cache is about to be prefilled to full size
            # beside it. Bring the map under its caps here rather than only at
            # publish, so the two are never outstanding at once: at the
            # default budget of 0 this is what releases the previous turn's
            # cache before a miss rebuilds one. Only the caps, not the
            # pressure reading — headroom is read once the turn's own cache is
            # real and its cost visible, which is at publish. A taken fork
            # slot is still in the map and must survive this pass: dropping
            # the very slot this turn just chose to share would defeat the
            # point of having forked it. A taken turn slot is in the map too,
            # but whether it stays is decided below, and its pass runs after
            # that decision — bytes about to leave the map are never charged
            # against slots that are staying.
            if slot is None or slot.kind == "fork":
                self._evict_caps(protect=(slot,) if slot is not None else ())
        warm: list | None = None
        kind = ""
        clone_failed = False
        if slot is not None and slot.kind == "fork":
            # The slot stays for the next conversation; this turn works on its
            # own copy. Taking that copy is part of the warm optimization, so
            # a failure here degrades to a cold prefill rather than failing
            # the turn — the same rule the warm-retry and header-probe paths
            # follow. Nothing has been emitted at this point, so unlike a
            # mid-generation failure there is no replay to weigh. The slot
            # itself is untouched (a fork is never removed by `_take`), so the
            # next conversation still finds it.
            warm = self._copy_of(stats, slot.cache, "fork clone failed", "prefilling cold")
            if warm is None:
                # Take the miss branch below: this is not a hit, and
                # `cold_retries` is not the counter for it either — that one
                # means a warm run that failed after it had started.
                slot = None
                clone_failed = True
            else:
                stats.fork_hits += 1
                kind = "fork"
        elif slot is not None:
            # A turn slot is copied and left in place when the budget can hold
            # it beside the slot this turn will publish — the copy grown by
            # this turn's tokens, forecast from the taken slot's bytes per
            # token: the parent's own size would pass a budget the publish
            # then cannot honour, copy and all — evicting least-recently-used
            # slots now to make that room. So a branch of the conversation
            # (Claude Code's progress-summary call takes the same prefix down
            # a different last turn) does not consume the slot the real
            # conversation extends next. When it cannot — always at a budget
            # of 0, where the feasibility rule refuses a copy that would not
            # fit even on an empty map — or when the owner was retired
            # meanwhile, the slot is moved: removed from the map and its
            # arrays adopted, the pre-retention behaviour byte for byte. A
            # copy failure is an optimization failure and falls back to the
            # move; the turn still runs warm on the original arrays.
            projected = -(-slot.nbytes * len(stable_ids) // len(slot.held))
            if self._make_room(projected, protect=(slot,), require=slot):
                warm = self._copy_of(
                    stats, slot.cache, "turn-slot copy failed", "extending the slot in place"
                )
            retained = warm is not None
            with self._lock:
                if not retained:
                    self._remove(slot)
                # The cap pass for a turn take runs here, after the decision,
                # so bytes about to leave the map are never charged against
                # slots that stay; a retained slot must survive it, and
                # protecting one that just left is a no-op.
                self._evict_caps(protect=(slot,))
            if retained:
                stats.retained += 1
                kind = "turn"
            else:
                stats.moved += 1
                kind = "turn-moved"
        if slot is not None:
            reuse = len(slot.held)
            stats.took_len = reuse  # the slot this turn took; stays 0 on a miss
            stats.took_kind = kind
            if warm is None:
                warm = slot.cache
            stats.hits += 1
            stats.reused_tokens += reuse
        else:
            # A failed clone was an allocation failure of the size a restore
            # would ask for again; go straight to the cold path then.
            restored = None if clone_failed else self._restore_from_disk(stats, stable_ids)
            if restored is not None:
                warm, reuse = restored
                stats.took_len = reuse
                stats.took_kind = "disk"
                stats.hits += 1
                stats.disk_hits += 1
                stats.reused_tokens += reuse
            else:
                reuse = 0
                stats.misses += 1
                # Scanned here, after the clone, so a fork whose copy failed is
                # measured like any other miss instead of keeping an earlier one's.
                stats.miss_lcp = max(
                    (common_prefix_length(held, stable_ids) for held in candidates), default=0
                )
                warm = hooks.new_cache()
        # Drop this frame's reference to the slot before anything else is
        # allocated: on a moved slot this is what keeps one conversation from
        # ever holding two copies of itself, and on a retained one the map's
        # reference is the only one that should outlive this point. (A fork
        # slot is referenced by the map by design.)
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

        # `text` gets a real value on every reachable path below, but not one
        # a flow analysis can prove without correlating `retry_reason` back to
        # which branch of the try/except ran — so it starts bound here rather
        # than relying on that proof.
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
            # prefilled: exactly the doubling that must never happen.
            retry_reason = str(e)

        if retry_reason is not None:
            if emitted and not replay_safe:
                raise RuntimeError(
                    f"warm generation failed after streaming {emitted} delta(s) "
                    f"({retry_reason}); not retrying cold, which would replay the turn"
                )
            # An optimization bug must never fail a turn. Only a warm attempt
            # is retried, so a genuine engine error still surfaces at once.
            stats.cold_retries += 1
            # The per-turn gauges describe the attempt that produced the text:
            # left alone they would add the failed warm attempt to the retry
            # and keep naming the slot it abandoned.
            stats.begin_turn()
            warnings.warn(
                f"sous prompt cache: warm generation failed ({retry_reason}); retrying cold",
                stacklevel=2,
            )
            warm = hooks.new_cache()
            text = self._run(
                stats, warm, stable_ids, full_ids, 0, max_tokens, relay, fork_at, owner, epoch
            )

        # Protects only itself (decision 10): this turn's own forks are newer
        # than anything else in the map, so LRU already spares them unless
        # the budget holds nothing else — and then it cannot hold a fork
        # beside this turn slot either, so dropping the fork now is exactly
        # what the next turn's pre-turn cap pass would do anyway. Protecting
        # it here would only leave the map over budget across a turn
        # boundary instead of at one.
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
        fork_at: Sequence[int] | Callable[[], Sequence[int]],
        owner: threading.Thread,
        epoch: int,
    ) -> str:
        hooks = self._hooks
        anchor = len(stable_ids)
        entry_reuse = reuse
        stats.prefilled_tokens += anchor - entry_reuse
        published: list[Slot] = []
        for boundary, need_slot, need_file in self._fork_boundaries(
            stats, owner, stable_ids, fork_at, reuse
        ):
            # Stop at the boundary and continue from there: a fork is taken
            # while prefilling past it because no layer can be rewound to it
            # afterwards. Ascending order means no copy is ever wanted behind
            # a prefix already prefilled. An empty segment (a turn that
            # started exactly here) is a no-op on both backends.
            started = _clock()
            hooks.prefill(cache, list(stable_ids[reuse:boundary]))
            reuse = boundary
            stats.prefill_seconds += _clock() - started
            if need_file:
                # The lowest boundary is the tools boundary when the probe
                # found two, and the header when it found one.
                kind = "tools" if stats.bound_lo else "header"
                self._persist(stats, cache, list(stable_ids[:boundary]), kind)
            if not need_slot:
                continue
            started = _clock()
            # What the cache holds at the boundary is what the copy will hold,
            # so it is also the price — re-read at every stop, since the cache
            # has grown.
            price = slot_bytes(cache)
            copy: list | None = None
            slot: Slot | None = None
            try:
                # Charged before it is allocated, with this turn's earlier
                # copies protected: a budget with room for one copy skips the
                # second rather than evicting the first into it, and a copy
                # that cannot fit even on an empty map is not taken at all —
                # better one prefill than a second cache the machine has no
                # room for.
                if self._make_room(price, protect=published):
                    copy = hooks.new_cache()
                    fork_copy(cache, copy, hooks.copy_array)
                    hooks.eval_cache(copy)
                    slot = Slot(copy, list(stable_ids[:boundary]), owner, "fork", slot_bytes(copy))
                    if self._publish(slot, epoch, protect=published):
                        stats.forks += 1
                        published.append(slot)
            except Exception as e:
                # The fork is an optimization on a turn that is already
                # prefilled to the boundary and needs nothing more from it.
                # Saying so explicitly matters here: a cold attempt forks with
                # `reuse` 0, and `generate`'s handler re-raises instead of
                # retrying there — an exception escaping this block would fail
                # a viable generation.
                warnings.warn(
                    f"sous prompt cache: fork copy failed ({type(e).__name__}); "
                    "continuing without a fork",
                    stacklevel=2,
                )
            finally:
                # A publish can still refuse the slot — a reset since the turn
                # began, or the owner retired mid-turn — and neither a refused
                # nor a half-built copy may stay pinned by this frame for the
                # whole decode that follows. (An accepted one is never evicted
                # by its own publish: _publish protects the slot it adds — and
                # `published.append(slot)` above already took its own
                # reference, so clearing `slot` here does not lose it.) `slot`
                # is cleared too, not just `copy`: `Slot.cache` is the copy, so
                # a Slot left referenced by this frame keeps the whole copy
                # alive across the decode below just as surely as `copy` would.
                # The copy is part of building the cache, so it is prefill time.
                stats.prefill_seconds += _clock() - started
                copy = slot = None
        # The list existed to protect the copies from each other while they
        # were charged. Dropping it here means a fork that pressure or a reset
        # evicts during the decode below is freed then, not pinned by this
        # frame until the turn ends.
        published = []
        if all_trimmable(cache):
            # Everything rewinds, so (the rest of) prefill and decode fuse into
            # one pass and the generation block plus the generated tokens are
            # simply trimmed back off afterwards.
            started = _clock()
            text = hooks.decode(cache, list(full_ids[reuse:]), max_tokens, on_delta)
            trim_to(cache, anchor)
            stats.decode_seconds += _clock() - started
            return text
        # A recurrent layer cannot rewind, so stop at the anchor, record it,
        # and put the cache back there once the generation is done.
        started = _clock()
        hooks.prefill(cache, list(stable_ids[reuse:]))
        snap, nbytes = snapshot(cache, hooks.copy_array)
        stats.snapshot_bytes = nbytes
        stats.prefill_seconds += _clock() - started
        started = _clock()
        text = hooks.decode(cache, list(full_ids[anchor:]), max_tokens, on_delta)
        restore(cache, snap, hooks.copy_array)
        stats.decode_seconds += _clock() - started
        return text
