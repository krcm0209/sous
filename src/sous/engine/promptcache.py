"""Cross-turn prefix cache: the rules, the counters, and the orchestrator.

This module imports no mlx, deliberately. The engine tests are model-marked and
never run in CI, so every decision that affects correctness is made here, where
a fake cache layer can exercise it. Array copies arrive through an injected
`copy_array` callable rather than an mx import.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    # of milliseconds on every lookup.
    return n if list(new_ids[:n]) == list(cached_ids) else 0


def all_trimmable(cache: Sequence[Any]) -> bool:
    # Sequence[Any] rather than a Protocol: mlx-lm and mlx-vlm ship their own
    # cache classes with no shared base and no type stubs, and trim/offset exist
    # only on the layers that rewind. A precise type here would be a fiction no
    # real class satisfies, and ty would reject the attribute access.
    """Whether every layer can rewind, so the cache needs no state copy.

    Asks `is_trimmable()` rather than testing for a `trim` attribute:
    RotatingKVCache owns a `trim` but reports False once its window has
    wrapped, and trimming it then would desync its ring index. Re-asked every
    turn for the same reason — the answer can change mid-task.
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

    Returns the snapshot and the bytes copied, for the task report.
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
        # the task, so this check is cheap insurance, not a new failure mode.
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


_MEMO_SLOTS = ("stable", "full", "header")


class PromptMemo:
    """One slot per render, keyed by the exact prompt text.

    The header slot holds the system turn rendered alone — the fork boundary.

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

    def _evictable(self, protect: Slot | None) -> list[Slot]:
        return [s for s in self._slots if s is not protect]

    def _drop_lru(self, protect: Slot | None) -> None:
        victim = min(self._evictable(protect), key=lambda s: s.last_used)
        self._slots.remove(victim)
        self._stats_for(victim.owner).evictions += 1
        # `victim` dies with this frame: the slot's cache is freed the moment
        # nothing else references it, which is what the pressure re-read in
        # _evict relies on.

    def _evict_caps(self, protect: Slot | None) -> None:
        """Bring the map under its count and byte caps, never touching
        `protect`. On its own before a turn starts, `protect` is None: nothing
        here is spoken for."""
        while self._evictable(protect) and (
            len(self._slots) > MAX_SLOTS or self._resident() > self.max_bytes
        ):
            self._drop_lru(protect)

    def _evict(self, protect: Slot) -> None:
        """Bring the map under its caps, then under memory pressure, never
        touching `protect` — the slot just published. A turn's own slot is
        never evicted by its own publish: on a machine with room for exactly
        one slot, that one still survives."""
        self._evict_caps(protect)
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
            # Whatever is still in the map is not going to serve this turn,
            # and this turn's own cache is about to be prefilled to full size
            # beside it. Bring the map under its caps here rather than only at
            # publish, so the two are never outstanding at once: at the
            # default budget of 0 this is what releases the previous turn's
            # cache before a miss rebuilds one. Only the caps, not the
            # pressure reading — headroom is read once the turn's own cache is
            # real and its cost visible, which is at publish.
            self._evict_caps(protect=None)
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
