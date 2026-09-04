"""Cross-turn prefix cache: the rules, the counters, and the orchestrator.

This module imports no mlx, deliberately. The engine tests are model-marked and
never run in CI, so every decision that affects correctness is made here, where
a fake cache layer can exercise it. Array copies arrive through an injected
`copy_array` callable rather than an mx import.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    if not cached_ids or len(new_ids) <= len(cached_ids):
        return 0
    # strict=False is deliberate: the lengths differ by design. ruff selects B,
    # so B905 requires the flag to be explicit either way.
    for a, b in zip(cached_ids, new_ids, strict=False):
        if a != b:
            return 0
    return len(cached_ids)


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


class CacheHooks(Protocol):
    """The four things only an engine can do. Everything else is shared."""

    def new_cache(self) -> list: ...
    def prefill(self, cache: list, token_ids: list[int]) -> None: ...
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None
    ) -> str: ...
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
        # This default is the orchestrator's own semantics, not the user-facing
        # one — both engines always pass `enabled` explicitly. The shipped
        # default lives in SousConfig.prompt_cache; don't couple this one
        # to it.
        self._hooks = hooks
        self.enabled = enabled
        self._cache: list | None = None
        self._held: list[int] = []
        # The thread whose mlx streams built self._cache (issue #34). A Thread
        # object, not its ident: idents are recycled after a thread exits, so
        # an ident could falsely match a later, unrelated thread — a strongly
        # held Thread object cannot.
        self._owner: threading.Thread | None = None
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
        self._owner = None
        self._stats = PromptCacheStats()

    def stats(self) -> dict:
        return self._stats.as_dict()

    def generate(
        self,
        stable_ids: list[int],
        full_ids: list[int],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        hooks = self._hooks
        if not self.enabled:
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)

        # `_run`'s anchor (`len(stable_ids)`) only means what it assumes: that
        # `full_ids` is the stable render plus a generation-only suffix, true
        # for chat templates that append a generation block rather than
        # rewrite one. [model].id accepts any MLX model and therefore any
        # chat template, so sous cannot assume that in general. reuse_length
        # already tests exactly this — strict prefix, with a suffix left to
        # decode — so ask it instead of duplicating the rule. Checked before
        # any cache is chosen or published: self._cache/self._held are left
        # exactly as they were, so nothing from this turn is retained and a
        # cache still held from an earlier, well-behaved turn survives for a
        # later one to reuse.
        if reuse_length(stable_ids, full_ids) == 0:
            self._stats.misses += 1
            warnings.warn(
                "sous prompt cache: full prompt is not the stable render "
                "plus a generation suffix; decoding cold this turn",
                stacklevel=2,
            )
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)

        epoch = self._epoch
        # Bind the stats object itself, not self._stats, and write through
        # this local for the whole call (passed into _run too). reset() swaps
        # in a fresh PromptCacheStats; if an abandoned generation thread
        # reaches a late write (snapshot_bytes after prefill, cold_retries in
        # the retry path) after that swap, it lands on this orphaned object
        # instead of the next task's counters.
        stats = self._stats
        cache, held = self._cache, self._held
        # Invalid until a generation completes. The cache is mutated in place,
        # so a raise mid-stream leaves it holding tokens `held` does not
        # describe, and reusing it then would duplicate them.
        self._cache, self._held = None, []

        reuse = reuse_length(held, stable_ids) if cache is not None else 0
        if reuse and self._owner is not threading.current_thread():
            # The cached arrays live only on the publishing thread's mlx
            # streams (issue #34); a different session thread (a worker task
            # vs the gateway's long-lived one) must prefill cold rather than
            # touch them. Force the same miss path a prefix mismatch takes,
            # rather than attempt a warm run doomed to the cross-thread mlx
            # failure that _run's except clause would otherwise have to catch.
            reuse = 0
        # `cache is not None and reuse` rather than a bare `if reuse`: it is what
        # lets the type checker see `warm` as a plain list in both branches, with
        # no assert and no ignore pragma.
        if cache is not None and reuse:
            stats.hits += 1
            stats.reused_tokens += reuse
            warm: list = cache
        else:
            reuse = 0
            stats.misses += 1
            warm = hooks.new_cache()
        # Drop the only remaining reference to the previous turn's cache
        # before a full-size replacement is prefilled below. On a miss,
        # `cache` above still points at the prior turn's cache, and a miss
        # only ever happens mid-task after elision — i.e. precisely when that
        # cache is at its largest. Without this line both caches are pinned
        # in memory at once while the new one is prefilled to full size,
        # which is exactly the doubling the spec promises never happens.
        cache = None

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
            text = self._run(stats, warm, stable_ids, full_ids, reuse, max_tokens, relay)
        except Exception as e:
            if reuse == 0:
                raise
            # Capture only the message here; the retry itself runs after this
            # suite exits. Python clears the `as e` binding and its traceback
            # at the end of the except clause (PEP 3110) — retrying inside it
            # would keep the failed full-size cache pinned by that traceback
            # for the whole retry, on top of the cold replacement being
            # prefilled, which is the same doubling as the miss path above.
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
            text = self._run(stats, warm, stable_ids, full_ids, 0, max_tokens, relay)

        if epoch == self._epoch:
            self._cache, self._held = warm, list(stable_ids)
            self._owner = threading.current_thread()
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
