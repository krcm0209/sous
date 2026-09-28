"""The DFlash drafter's context on the prompt-cache path.

mlx-vlm's DFlash round loop (`speculative/dflash.py:_dflash_rounds`) hands the
drafter, on its first draft call after `reset`, the target-layer hidden
states of every token the decode call's own prefill processed, and the
drafter's per-layer rotating cache (`sliding_window - 1` positions) attends
over them on every later round. On a hybrid model sous prefills the stable
prompt in a drafter-less call and decodes only the generation prompt — 7
tokens on Qwen3.8 — so the drafter drafted every subagent turn from those 7
tokens. On 64 real subagent turns at block 3 (#131's findings, #142) that
was 2.27 tokens per round against 2.58 with the turn's own prefill in front
of it (~+10% decode) and 2.62 with the last 2,047 tokens of the prompt.

This module captures the newest `window` positions of the hidden states of
everything a turn prefills before its decode, and prepends them to the
decode call's own on that first draft call. Two hooks, both inert until
armed:

- a class-level post-hook on the language model's forward that copies
  `output.hidden_states` into the sink armed on that instance (`_SINK`, set
  with `object.__setattr__` so it stays out of mlx's parameter tree). The sink
  keeps only the newest `window` positions: ~105 MB per 2048-token chunk on
  the default model (five layers x 5120, bf16), ~210 MB while a second chunk
  is being taken, and ~105 MB for the assembled tail across the decode,
  whatever the prompt's length;
- instance-level wraps of the drafter's `draft_block`, `draft_block_greedy`
  and `reset` that prepend the pending context once, on the first draft call
  after a reset, and record how many positions that call saw.

Neither changes a token: the verify is exact, so a missing or wrong context
costs acceptance, never output. That is also why the gates here are
structural (drafter kind, the hooks the round loop looks for, the layers the
drafter names, a language model that runs single tokens through its forward)
rather than source hashes: a drift in mlx-vlm shows on the turn line as
`draft_context=7` (the capture lost) or as `draft_context=0` beside
`draft_context=unavailable` on the load line (a gate tripped), not as a
different reply.
"""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Sequence
from typing import Any

from sous.engine.int8prefill import _root

logger = logging.getLogger("sous.engine.draftctx")

# The context a drafter with an unbounded (plain KVCache) context gets: the
# window the shipped DFlash2 checkpoints train with, so the first round's
# projection over the context stays bounded whatever the prompt's length.
DEFAULT_WINDOW = 2048
# Set with object.__setattr__ so they stay out of mlx's parameter tree.
_SINK = "_sous_draft_sink"  # on the language model: the armed Sink, or None
_PENDING = "_sous_draft_pending"  # on the drafter: the context to prepend, or None
_FIRST = "_sous_draft_first"  # on the drafter: no draft call since the last reset
_FED = "_sous_draft_fed"  # on the drafter: positions the first draft call saw
_WRAPPED_DRAFTER = "_sous_draft_wrapped"  # on the drafter: its calls are shadowed
_WRAPPED: set[type] = set()


class Sink:
    """The newest `keep` positions of the hidden states a turn's prefills
    produced, per forward as the layers' arrays concatenated along the last
    axis — the shape the round loop hands the drafter — and evaluated as they
    arrive: a lazy hidden state would pin that forward's whole graph. One
    sink spans every segment a turn prefills before its decode, so a cold
    turn's tail ends where the generation prompt begins whatever fork stops
    split the prefill."""

    def __init__(self, layer_ids: list[int], keep: int):
        self.layer_ids = list(layer_ids)
        self.keep = keep
        self._chunks: list[Any] = []
        self._held = 0
        self._failed = False

    def fail(self) -> None:
        """Void the capture: a tail missing a forward's rows would end short
        of the generation prompt and seed the drafter with the wrong
        positions, which costs more acceptance than no context at all."""
        self._failed = True
        self._chunks = []
        self._held = 0

    def take(self, hidden_states: Any) -> None:
        import mlx.core as mx

        # A list of per-layer arrays is what the forward returns; anything
        # else (None from a model that does not capture) is nothing to keep.
        if self._failed or self.keep <= 0:
            return
        if not isinstance(hidden_states, list | tuple) or not hidden_states:
            return
        h = mx.concatenate(list(hidden_states), axis=-1)
        mx.async_eval(h)
        self._chunks.append(h)
        self._held += int(h.shape[1])
        # Drop whole chunks the tail can no longer reach; the slice at the
        # end trims the rest.
        while self._chunks and self._held - int(self._chunks[0].shape[1]) >= self.keep:
            self._held -= int(self._chunks[0].shape[1])
            self._chunks.pop(0)

    def finish(self) -> Any | None:
        """The tail as one evaluated array of its own, or None when nothing
        was captured (a model whose forward returns no hidden states)."""
        import mlx.core as mx

        if not self._chunks:
            return None
        parts, excess = list(self._chunks), self._held - self.keep
        if excess > 0:
            # Cut the excess off the oldest chunk before assembling: a slice
            # of the assembled array would be a view keeping every row of it
            # alive for the whole decode, twice the tail at a chunk boundary.
            parts[0] = parts[0][:, excess:, :]
        if len(parts) > 1:
            tail = mx.concatenate(parts, axis=1)
        elif excess > 0:
            tail = mx.array(parts[0])  # a copy: the slice alone is a view
        else:
            tail = parts[0]
        mx.eval(tail)
        self._chunks = []
        self._held = 0
        return tail


def _forward_hook(cls: type) -> None:
    orig = cls.__call__

    @functools.wraps(orig)
    def __call__(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        sink = getattr(self, _SINK, None)
        if sink is not None:
            try:
                sink.take(getattr(out, "hidden_states", None))
            except Exception as e:  # noqa: BLE001 — a capture must never fail a turn
                object.__setattr__(self, _SINK, None)
                sink.fail()
                warnings.warn(
                    f"sous: drafter context capture failed ({type(e).__name__}: {e}); "
                    "this turn's drafter starts from the generation prompt",
                    stacklevel=2,
                )
        return out

    cls.__call__ = __call__  # ty: ignore[invalid-assignment]


def install_forward_hook(cls: type) -> None:
    """Hook one language-model class's forward, once per process. The hook
    reads only an instance that has a sink armed, so installing it changes
    nothing for any other model or turn."""
    if cls not in _WRAPPED:
        _forward_hook(cls)
        _WRAPPED.add(cls)


def _wrap_drafter(drafter: Any) -> None:
    """Shadow the drafter's draft calls and reset per instance, the way
    mlx-vlm binds the target's argmax onto the drafter (`argmax_from_hidden`).
    Once: a second engine over the same drafter would otherwise nest them."""
    import mlx.core as mx

    if getattr(drafter, _WRAPPED_DRAFTER, False):
        return
    object.__setattr__(drafter, _WRAPPED_DRAFTER, True)
    reset = drafter.reset

    @functools.wraps(reset)
    def _reset(*args, **kwargs):
        object.__setattr__(drafter, _FIRST, True)
        object.__setattr__(drafter, _FED, 0)
        return reset(*args, **kwargs)

    object.__setattr__(drafter, "reset", _reset)

    for name in ("draft_block", "draft_block_greedy"):
        orig = getattr(drafter, name, None)
        if orig is None:
            continue

        @functools.wraps(orig)
        def _draft(last_bonus, hidden, cache, block_size, sampler, *args, _orig=orig, **kwargs):
            if getattr(drafter, _FIRST, False):
                object.__setattr__(drafter, _FIRST, False)
                pending = getattr(drafter, _PENDING, None)
                object.__setattr__(drafter, _PENDING, None)
                if (
                    pending is not None
                    and pending.ndim == hidden.ndim == 3
                    and pending.shape[0] == hidden.shape[0]
                    and pending.shape[-1] == hidden.shape[-1]
                ):
                    hidden = mx.concatenate([pending, hidden], axis=1)
                object.__setattr__(drafter, _FED, int(hidden.shape[1]))
            return _orig(last_bonus, hidden, cache, block_size, sampler, *args, **kwargs)

        object.__setattr__(drafter, name, _draft)


def _unwrap_drafter(drafter: Any) -> None:
    """Undo `_wrap_drafter`: the wraps close over the drafter, a cycle that
    only a gc pass collects, and the drafter holds the target's embedding
    and head through `bind` — so an unload that left them would keep the
    weights resident until some later collection."""
    for name in ("reset", "draft_block", "draft_block_greedy", _WRAPPED_DRAFTER):
        if name in getattr(drafter, "__dict__", {}):
            object.__delattr__(drafter, name)


def window(drafter: Any) -> int:
    """How many context positions the drafter keeps, read off its config the
    way its `make_cache` does: a buffered window, else a sliding layer's
    window less one, else DEFAULT_WINDOW for an unbounded context."""
    config = getattr(drafter, "config", None)
    buffered = getattr(config, "draft_window_size", None)
    if buffered is not None and int(buffered) > 0:
        return int(buffered)
    sliding = getattr(config, "sliding_window", None)
    if sliding is not None and int(sliding) > 1:
        layer_types = getattr(config, "layer_types", None) or []
        if any(kind == "sliding_attention" for kind in layer_types):
            return int(sliding) - 1
    return DEFAULT_WINDOW


class DraftContext:
    """What one engine's prefills capture and its decodes seed. `off` when the
    engine has no drafter, `unavailable` when it has one this module cannot
    serve; either way every method is a no-op and `status` says why."""

    def __init__(self, status: dict[str, Any], drafter: Any = None, layer_ids: Sequence[int] = ()):
        self.status = status
        self._drafter = drafter
        self.layer_ids = list(layer_ids)
        self.keep = int(status.get("window") or 0)

    @property
    def active(self) -> bool:
        return self.status["state"] == "active"

    def sink(self, capture: Any) -> Sink | None:
        """The sink a prefill captures into: `capture` continued when it is
        one this engine handed out, a fresh one when the turn asks to start
        capturing, None when it asks for nothing or there is nothing to seed."""
        if isinstance(capture, Sink):
            return capture
        return Sink(self.layer_ids, self.keep) if capture and self.active else None

    @staticmethod
    def tail(capture: Any) -> Any | None:
        """What a turn's capture holds, assembled — None for no capture."""
        return capture.finish() if isinstance(capture, Sink) else None

    def close(self) -> None:
        """Let go of the drafter: undo its wraps and drop the reference, so an
        unload frees the drafter and, through it, the target's modules."""
        if self._drafter is not None:
            _unwrap_drafter(self._drafter)
            self._drafter = None

    @staticmethod
    def capture_kwargs(sink: Sink | None) -> dict[str, Any]:
        """The kwarg that makes every prefill forward return the target-layer
        hidden states — what mlx-vlm's own speculative prefill asks for."""
        return {} if sink is None else {"capture_layer_ids": list(sink.layer_ids)}

    def arm(self, model: Any, sink: Sink | None) -> _Armed:
        """Arm `sink` on `model`'s language model for the duration of a `with`
        block; a None sink arms nothing."""
        return _Armed(_root(model) if sink is not None else None, sink)

    def seed(self, context: Any) -> _Seeded:
        """Hand `context` (or nothing) to the drafter's next first draft call,
        for the duration of a `with` block: a decode that never drafts — one
        token asked for, or a stop on the first — leaves nothing pending for
        a later turn and reads as no context at all."""
        return _Seeded(self._drafter if self.active else None, context)

    def fed(self) -> int:
        """The context the most recent decode drafted from: the positions its
        first draft call was handed, capped at the window — the drafter's
        sliding attention drops the oldest rows past it, so a full tail plus
        the generation prompt is still one window. 0 when it never drafted."""
        if not self.active:
            return 0
        return min(int(getattr(self._drafter, _FED, 0) or 0), self.keep)


class _Armed:
    def __init__(self, lm: Any, sink: Sink | None):
        self._lm, self._sink = lm, sink

    def __enter__(self) -> Sink | None:
        if self._lm is not None:
            object.__setattr__(self._lm, _SINK, self._sink)
        return self._sink

    def __exit__(self, *exc: object) -> None:
        if self._lm is not None:
            object.__setattr__(self._lm, _SINK, None)


class _Seeded:
    def __init__(self, drafter: Any, context: Any):
        self._drafter, self._context = drafter, context

    def __enter__(self) -> None:
        if self._drafter is not None:
            object.__setattr__(self._drafter, _PENDING, self._context)
            # The round loop's reset zeroes this too, but a decode that stops
            # before the loop runs never resets, and must not read the
            # previous decode's context as its own.
            object.__setattr__(self._drafter, _FED, 0)

    def __exit__(self, *exc: object) -> None:
        if self._drafter is not None:
            object.__setattr__(self._drafter, _PENDING, None)


OFF = DraftContext({"state": "off", "reason": None, "window": 0, "layers": 0})


def rounds_snapshot(drafter: Any) -> Any:
    """The drafter's lifetime round counters before a decode, or None
    without a drafter — `rounds_since` reads the decode's own share off it."""
    if drafter is None:
        return None
    from mlx_vlm.speculative.utils import speculative_stats_snapshot

    return speculative_stats_snapshot(drafter)


def rounds_since(drafter: Any, snapshot: Any) -> tuple[int, int, int]:
    """(rounds, accepted, drafted) the drafter recorded since `snapshot`:
    mlx-vlm's lifetime counters survive the reset every decode starts with,
    unlike its per-call `accept_lens`, and a call that never drafted (one
    token asked for) reads as zeros."""
    if drafter is None or snapshot is None:
        return (0, 0, 0)
    from mlx_vlm.speculative.utils import speculative_stats_since

    rounds, accepted, drafted = speculative_stats_since(drafter, snapshot)
    return (int(rounds or 0), int(accepted or 0), int(drafted or 0))


def _refuse(reason: str) -> DraftContext:
    warnings.warn(
        f"sous: drafter context unavailable ({reason}); the drafter starts every turn "
        "from the generation prompt",
        stacklevel=3,
    )
    return DraftContext({"state": "unavailable", "reason": reason, "window": 0, "layers": 0})


def enable(model: Any, drafter: Any, kind: str) -> DraftContext:
    """Arm one loaded model and its drafter. Never raises: a model load must not
    fail because of an optimisation, and without it the drafter simply keeps
    starting from the generation prompt."""
    if drafter is None:
        return OFF
    try:
        if kind != "dflash":
            return _refuse(f"drafter kind {kind!r} (dflash only)")
        # A drafter the round loop prepares hidden states for consumes them
        # under a different contract than the plain draft call this seeds;
        # one that sizes its block from the prompt length the loop shows it
        # would size it for the 7 tokens the decode call prefilled while
        # drafting from a full window, off whatever that policy assumes.
        for hook in ("prepare_target_hidden", "choose_block_ceiling", "choose_initial_block_size"):
            if callable(getattr(drafter, hook, None)):
                return _refuse(f"drafter defines {hook}")
        config = getattr(drafter, "config", None)
        layer_ids = [int(i) for i in getattr(config, "target_layer_ids", None) or ()]
        if not layer_ids:
            return _refuse("drafter names no target layers")
        for name in ("reset", "draft_block"):
            if not callable(getattr(drafter, name, None)):
                return _refuse(f"drafter has no {name}")
        lm = _root(model)
        if lm is model:
            return _refuse("model has no language_model")
        # A language model that routes one-token forwards around its forward
        # (qwen4_exp does) would capture nothing for the prefill's last token.
        if callable(getattr(lm, "_batch_invariant_decode", None)):
            return _refuse("language model decodes single tokens outside its forward")
        keep = window(drafter)
        install_forward_hook(type(lm))
        _wrap_drafter(drafter)
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        return _refuse(str(e) or type(e).__name__)
    logger.info("drafter context: %d target layers, window %d", len(layer_ids), keep)
    return DraftContext(
        {"state": "active", "reason": None, "window": keep, "layers": len(layer_ids)},
        drafter,
        layer_ids,
    )
