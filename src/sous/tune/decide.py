"""The quick rule: among the current model's quality-neutral arms, the
fastest decode wins, and only the keys that differ from the user's config
become changes. The model itself is never this rule's to change."""

from __future__ import annotations

from dataclasses import dataclass, field

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow


@dataclass(frozen=True)
class QuickChoice:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    gateway_window: int | None
    changes: dict[str, dict[str, object]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def score(row: BenchRow, long: bool | None = None) -> float | None:
    """Decode at the long context when `long` — the gateway's regime and
    where the drafter's gain is smallest — else at the short one. With
    `long` unset, whichever the row has, longest first: a reading of one
    row. A ranking across rows must score every row in one unit (see
    quick_decision); a 1K number beside a 16K one says nothing."""
    if not row.ok:
        return None
    if long is None:
        return row.decode_tps_16k if row.decode_tps_16k is not None else row.decode_tps_1k
    return row.decode_tps_16k if long else row.decode_tps_1k


def _changes(user: SousConfig, arm: Arm) -> dict[str, dict[str, object]]:
    model: dict[str, object] = {}
    if arm.drafter_id != user.speculative_draft_id:
        model["speculative_draft_id"] = arm.drafter_id
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
    # The arm's own fit, not the model's shared measurement window: that one
    # reserves memory for the heaviest drafter, which this arm may not load.
    window = arm.fit_window if arm.fit_window is not None else arm.window
    if window < user.max_context_tokens:
        model["max_context_tokens"] = window
    changes: dict[str, dict[str, object]] = {}
    if model:
        changes["model"] = model
    gateway = arm.fit_gateway_window if arm.fit_gateway_window is not None else arm.gateway_window
    if user.gateway_enabled and gateway is not None and gateway < user.gateway_max_context_tokens:
        changes["gateway"] = {"max_context_tokens": gateway}
    return changes


def _no_baseline_reason(current_arm: Arm | None, rows: list[BenchRow]) -> str:
    if current_arm is None:
        return "the configured arm was not measured; no change proposed"
    failed = next((r for r in rows if r.key == current_arm.key), None)
    detail = failed.error if failed is not None and failed.error else "not attempted"
    return f"the configured arm ({current_arm.label}) did not measure: {detail}; no change proposed"


def quick_decision(user: SousConfig, arms: list[Arm], rows: list[BenchRow]) -> QuickChoice | None:
    by_key = {a.key: a for a in arms}
    # Only rows of this run's arms, measured at the window those arms run
    # at: a resumed run whose config changed in between has rows of both.
    mine = [
        r
        for r in rows
        if r.model_id == user.model_id
        and r.ok
        and r.key in by_key
        and r.window == by_key[r.key].window
    ]
    # One unit for every row compared: the long context when every row has
    # it, the short one otherwise — never each row's own best.
    long = all(r.decode_tps_16k is not None for r in mine)
    scored = [(r, s) for r in mine if (s := score(r, long)) is not None]
    if not scored:
        return None
    current_arm = next((a for a in arms if a.current), None)
    current = next(((r, s) for r, s in scored if by_key[r.key].current), None)
    # A strict improvement over the current arm is required: a tie changes
    # nothing, and a re-run on a noisy afternoon must not flip a setting.
    winner = current
    for row, s in scored:
        if winner is None or s > winner[1]:
            winner = (row, s)
    if winner is None:
        return None
    arm = by_key[winner[0].key]
    # Without a scored measurement of the user's own configuration there is
    # nothing to compare the winner against: diffing it against the config
    # directly (_changes below) would propose a change on the strength of
    # one arm's number alone, never knowing whether the arm actually beats
    # what the user already has.
    if current is None:
        return QuickChoice(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            gateway_window=arm.gateway_window,
            changes={},
            reasons=[_no_baseline_reason(current_arm, rows)],
        )
    which = "decode at 16K context" if long else "decode at 1K context"
    best = winner[1]
    reasons: list[str] = []
    if winner[0].key != current[0].key:
        reasons.append(
            f"{which}: {best:.1f} tok/s for {winner[0].label}, "
            f"{current[1]:.1f} tok/s for the current arm ({current[0].label})"
        )
    else:
        reasons.append(f"{which}: {best:.1f} tok/s for {winner[0].label}")
        reasons.append("the current arm is already the fastest measured")
    return QuickChoice(
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        window=arm.window,
        gateway_window=arm.gateway_window,
        changes=_changes(user, arm),
        reasons=reasons,
    )
