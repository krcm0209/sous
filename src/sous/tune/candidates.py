"""The curated candidate table, what a checkpoint's config says about
itself, and whether it fits this machine."""

from __future__ import annotations

import datetime
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from sous.context import TOKEN_STEP, kv_bytes_per_token, native_max_tokens
from sous.engine.base import fetch_model_config, select_backend

# A drafter ships bf16 and sous quantizes it to 4-bit at load: 4 bits of
# code plus group scales against 16, so ~3.5x smaller resident.
DRAFTER_QUANT_DIVISOR = 3.5
# Working memory a turn needs beyond weights and KV: activations, the
# drafter's captured hidden states, the allocator's slack.
SLACK_BYTES = 2 * 2**30
# A window is a multiple of the step mlx grows KV buffers in.
WINDOW_STEP = TOKEN_STEP
_GIB = 1 << 30
_GB = 10**9
_FAST_VERIFIER_BITS = (4, 5, 8)


@dataclass(frozen=True)
class Candidate:
    id: str
    tier: str
    drafters: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Table:
    checked: datetime.date
    validated_with: str
    trusted_orgs: tuple[str, ...]
    publishers: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def load_table(path: Path | None = None) -> Table:
    text = path.read_text() if path else files("sous.tune").joinpath("candidates.toml").read_text()
    raw = tomllib.loads(text)
    return Table(
        checked=raw["checked"],
        validated_with=str(raw["validated_with"]),
        trusted_orgs=tuple(raw["trusted_orgs"]),
        publishers=tuple(raw["publishers"]),
        candidates=tuple(
            Candidate(
                id=c["id"],
                tier=c["tier"],
                drafters=tuple(c.get("drafters", ())),
                note=c.get("note", ""),
            )
            for c in raw["candidate"]
        ),
    )


@dataclass(frozen=True)
class QuantSummary:
    mode: str | None
    bits: int | None
    group_size: int | None
    overrides: int
    verifier_fast: bool
    verifier_slow_layers: int
    int8_routable: bool
    int8_excluded_layers: int


def _fast(mode: str | None, bits: int | None) -> bool:
    # mlx-vlm's exact verifier runs its fused kernels only for affine 4/5/8-bit
    # layers; every other layout falls to one matmul per verify position.
    return mode == "affine" and bits in _FAST_VERIFIER_BITS


def _routable(mode: str | None, bits: int | None, group: int | None) -> bool:
    return mode == "affine" and bits == 4 and group == 64


def summarize_quantization(config: dict) -> QuantSummary:
    q = config.get("quantization") or config.get("quantization_config") or {}
    mode = q.get("mode", "affine") if q else None
    bits = q.get("bits") if q else None
    group = q.get("group_size") if q else None
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    slow = sum(1 for v in overrides.values() if not _fast(v.get("mode", mode), v.get("bits", bits)))
    excluded = sum(
        1
        for v in overrides.values()
        if not _routable(v.get("mode", mode), v.get("bits", bits), v.get("group_size", group))
    )
    return QuantSummary(
        mode=mode,
        bits=bits,
        group_size=group,
        overrides=len(overrides),
        verifier_fast=_fast(mode, bits),
        verifier_slow_layers=slow,
        int8_routable=_routable(mode, bits, group),
        int8_excluded_layers=excluded,
    )


@dataclass(frozen=True)
class Checkpoint:
    id: str
    model_type: str
    backend: str
    bytes: int | None
    kv_bytes_per_token: int | None
    native_max_tokens: int | None
    hidden_size: int | None
    num_layers: int | None
    num_target_layers: int | None
    quant: QuantSummary
    vocab_size: int | None = None


def _text(config: dict) -> dict:
    return config.get("text_config", config)


def describe(
    model_id: str,
    *,
    config_fn: Callable[[str], dict] | None = None,
    size_fn: Callable[[str], int | None] | None = None,
) -> Checkpoint:
    """Everything the tune needs from a checkpoint without loading it. The
    size callable is `hub.snapshot_bytes` in production and may answer None
    offline; the config comes through `fetch_model_config` (a cached
    kilobyte, the one network read that needs no consent)."""
    if size_fn is None:
        from sous.tune import hub

        size_fn = hub.snapshot_bytes

    config = (config_fn or fetch_model_config)(model_id)
    text = _text(config)
    return Checkpoint(
        id=model_id,
        model_type=str(config.get("model_type", "")),
        backend=select_backend(config),
        bytes=size_fn(model_id),
        kv_bytes_per_token=kv_bytes_per_token(config),
        native_max_tokens=native_max_tokens(config),
        hidden_size=text.get("hidden_size"),
        num_layers=text.get("num_hidden_layers"),
        num_target_layers=config.get("num_target_layers"),
        quant=summarize_quantization(config),
        vocab_size=text.get("vocab_size"),
    )


def drafter_compatible(target: Checkpoint, drafter: Checkpoint) -> tuple[bool, str]:
    """The shape checks mlx-vlm's drafter validation makes, made here before
    any download: a drafter reads the target's hidden states and emits its
    tokens, so the widths and vocabularies must agree and its declared
    target depth must be the target's. The engine survives a drafter that
    fails the real check by running without it, which a measurement cannot
    afford (see bench), so the refusal has to come first."""
    if drafter.hidden_size != target.hidden_size:
        return False, (
            f"hidden size {drafter.hidden_size} does not match the target's {target.hidden_size}"
        )
    if drafter.num_target_layers is not None and drafter.num_target_layers != target.num_layers:
        return False, (
            f"declares {drafter.num_target_layers} target layers; the target has "
            f"{target.num_layers}"
        )
    if (
        drafter.vocab_size is not None
        and target.vocab_size is not None
        and drafter.vocab_size != target.vocab_size
    ):
        return False, (
            f"vocabulary of {drafter.vocab_size} does not match the target's {target.vocab_size}"
        )
    return True, ""


@dataclass(frozen=True)
class Fit:
    fits: bool
    window: int
    needed_bytes: int
    working_set_bytes: int
    detail: str


def _needed(target_bytes: int, drafter_bytes: int, kv: int, window: int) -> int:
    return int(target_bytes + drafter_bytes / DRAFTER_QUANT_DIVISOR + kv * window + SLACK_BYTES)


def fit(
    target: Checkpoint,
    drafter: Checkpoint | None,
    *,
    window: int,
    floor: int,
    working_set_bytes: int,
) -> Fit:
    """Weights, the quantized drafter, one window of KV and the slack against
    Metal's recommended working set. A window that does not fit is lowered
    to the largest step that does; below `floor` the candidate is refused,
    and the arithmetic goes into `detail` either way so a refusal explains
    itself."""
    kv = target.kv_bytes_per_token
    dbytes = 0 if drafter is None else drafter.bytes
    if target.bytes is None or kv is None or dbytes is None:
        what = (
            "weights size" if target.bytes is None else "KV cost" if kv is None else "drafter size"
        )
        return Fit(False, window, 0, working_set_bytes, f"{what} unknown (offline?)")
    needed = _needed(target.bytes, dbytes, kv, window)
    chosen = window
    if needed > working_set_bytes:
        room = working_set_bytes - _needed(target.bytes, dbytes, kv, 0)
        chosen = max(0, room // kv) // WINDOW_STEP * WINDOW_STEP
        chosen = min(chosen, window)
        if chosen < floor:
            chosen = floor
        needed = _needed(target.bytes, dbytes, kv, chosen)
    fits = needed <= working_set_bytes
    drafter_note = f" + drafter {dbytes / _GB:.1f} GB/{DRAFTER_QUANT_DIVISOR}" if drafter else ""
    detail = (
        f"weights {target.bytes / _GB:.1f} GB ({target.bytes / _GIB:.1f} GiB){drafter_note}"
        f" + KV {kv} B/token x {chosen} + {SLACK_BYTES / _GIB:.0f} GiB slack"
        f" = {needed / _GIB:.1f} GiB vs working set {working_set_bytes / _GIB:.1f} GiB"
        # A lowered window sits within one step of the working set, and the
        # daemon's own prompt-cache budget is the same arithmetic — what is
        # left at that window is not enough for a fork slot.
        + (
            ""
            if chosen == window
            else f" (window lowered from {window}; little or nothing left for prompt-cache forks)"
        )
        + ("" if fits else f"; below the {floor}-token floor")
    )
    return Fit(fits, chosen, needed, working_set_bytes, detail)
