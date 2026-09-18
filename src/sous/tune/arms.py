"""One arm is one configuration the bench measures: a model, a drafter or
none, a block size, and the windows it fits at."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from sous.config import GATEWAY_MIN_CONTEXT_TOKENS, SousConfig
from sous.tune.candidates import Candidate, Checkpoint, Fit, drafter_compatible, fit

# The verify depths worth measuring: 2 and 3 pay on this family's GQA ratio,
# 5 is the most rows mlx's fused attention kernel takes.
BLOCK_SIZES = (2, 3, 5)


@dataclass(frozen=True)
class Arm:
    label: str
    config: SousConfig
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    gateway_window: int | None
    tier: str
    current: bool
    unvetted: bool = False


@dataclass(frozen=True)
class Refusal:
    model_id: str
    drafter_id: str
    reason: str


def _short(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def label_for(model_id: str, drafter_id: str, block_size: int) -> str:
    if not drafter_id:
        return _short(model_id)
    depth = "auto" if block_size == 0 else str(block_size)
    return f"{_short(model_id)} + {_short(drafter_id)} @{depth}"


def _with_current(user: SousConfig, candidates: list[Candidate]) -> list[Candidate]:
    if any(c.id == user.model_id for c in candidates):
        return list(candidates)
    drafters = (user.speculative_draft_id,) if user.speculative_draft_id else ()
    return [Candidate(user.model_id, "current", drafters, "the configured model"), *candidates]


def _arm(
    user: SousConfig,
    cand: Candidate,
    drafter_id: str,
    block: int,
    f: Fit,
    gateway_window: int | None,
) -> Arm:
    config = dataclasses.replace(
        user,
        model_id=cand.id,
        speculative_draft_id=drafter_id,
        speculative_block_size=block if drafter_id else user.speculative_block_size,
        max_context_tokens=min(user.max_context_tokens, f.window),
        gateway_max_context_tokens=(
            gateway_window if gateway_window is not None else user.gateway_max_context_tokens
        ),
    )
    current = (
        cand.id == user.model_id
        and drafter_id == user.speculative_draft_id
        and (not drafter_id or block == user.speculative_block_size)
    )
    return Arm(
        label=label_for(cand.id, drafter_id, block),
        config=config,
        model_id=cand.id,
        drafter_id=drafter_id,
        block_size=block if drafter_id else 0,
        window=min(user.max_context_tokens, f.window),
        gateway_window=gateway_window,
        tier=cand.tier,
        current=current,
    )


def quick_arms(
    user: SousConfig,
    candidates: list[Candidate],
    checkpoints: dict[str, Checkpoint],
    *,
    working_set_bytes: int,
) -> tuple[list[Arm], list[Refusal]]:
    """Every quality-neutral arm of every candidate that fits, the user's own
    configuration among them. All arms of a model share one window: the one
    the fit yields with the model's heaviest drafter, so a block size is never
    measured at a window another block size could not run at."""
    window = max(
        user.max_context_tokens,
        user.gateway_max_context_tokens if user.gateway_enabled else 0,
    )
    floor = GATEWAY_MIN_CONTEXT_TOKENS if user.gateway_enabled else user.context_min_tokens
    arms: list[Arm] = []
    refusals: list[Refusal] = []
    for cand in _with_current(user, candidates):
        target = checkpoints.get(cand.id)
        if target is None:
            refusals.append(Refusal(cand.id, "", "checkpoint unknown (offline?)"))
            continue
        drafters: list[Checkpoint] = []
        for drafter_id in cand.drafters:
            drafter = checkpoints.get(drafter_id)
            if drafter is None:
                refusals.append(Refusal(cand.id, drafter_id, "drafter unknown (offline?)"))
                continue
            if drafter.bytes is None:
                # Unresolvable cost, not incompatibility: reaching fit() with
                # it would refuse the whole model, drafterless arm included.
                refusals.append(Refusal(cand.id, drafter_id, "drafter size unknown (offline?)"))
                continue
            ok, why = drafter_compatible(target, drafter)
            if ok:
                drafters.append(drafter)
            else:
                refusals.append(Refusal(cand.id, drafter_id, why))
        heaviest = max(drafters, key=lambda d: d.bytes or 0, default=None)
        f = fit(target, heaviest, window=window, floor=floor, working_set_bytes=working_set_bytes)
        if not f.fits:
            refusals.append(Refusal(cand.id, "", f.detail))
            continue
        gateway_window = None
        if user.gateway_enabled:
            gateway_window = min(user.gateway_max_context_tokens, f.window)
        arms.append(_arm(user, cand, "", 0, f, gateway_window))
        for drafter in drafters:
            is_users_own_pair = cand.id == user.model_id and drafter.id == user.speculative_draft_id
            # The user's own block size may sit outside the curated set (0 is
            # the drafter's adaptive policy; anything else they set by hand),
            # and it must still be measurable as `current` — every other pair
            # sticks to the curated set.
            blocks = (
                sorted({*BLOCK_SIZES, user.speculative_block_size})
                if is_users_own_pair
                else BLOCK_SIZES
            )
            for block in blocks:
                arms.append(_arm(user, cand, drafter.id, block, f, gateway_window))
    return arms, refusals
