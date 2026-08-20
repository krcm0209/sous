"""Cross-turn prefix cache: the rules, the counters, and the orchestrator.

This module imports no mlx, deliberately. The engine tests are model-marked and
never run in CI, so every decision that affects correctness is made here, where
a fake cache layer can exercise it. Array copies arrive through an injected
`copy_array` callable rather than an mx import.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


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
