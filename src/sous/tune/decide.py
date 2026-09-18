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


def score(row: BenchRow) -> float | None:
    """Decode at the long context when it was measured — the gateway's regime
    and where the drafter's gain is smallest — else at the short one."""
    if not row.ok:
        return None
    return row.decode_tps_16k if row.decode_tps_16k is not None else row.decode_tps_1k


def best_per_model(rows: list[BenchRow]) -> dict[str, BenchRow]:
    best: dict[str, BenchRow] = {}
    for row in rows:
        s = score(row)
        if s is None:
            continue
        held = best.get(row.model_id)
        if held is None or s > (score(held) or 0.0):
            best[row.model_id] = row
    return best


def _changes(user: SousConfig, arm: Arm) -> dict[str, dict[str, object]]:
    model: dict[str, object] = {}
    if arm.drafter_id != user.speculative_draft_id:
        model["speculative_draft_id"] = arm.drafter_id
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
    if arm.window < user.max_context_tokens:
        model["max_context_tokens"] = arm.window
    changes: dict[str, dict[str, object]] = {}
    if model:
        changes["model"] = model
    if (
        user.gateway_enabled
        and arm.gateway_window is not None
        and arm.gateway_window < user.gateway_max_context_tokens
    ):
        changes["gateway"] = {"max_context_tokens": arm.gateway_window}
    return changes


def _no_baseline_reason(current_arm: Arm | None, rows: list[BenchRow]) -> str:
    if current_arm is None:
        return "the configured arm was not measured; no change proposed"
    failed = next((r for r in rows if r.label == current_arm.label), None)
    detail = failed.error if failed is not None and failed.error else "not attempted"
    return f"the configured arm ({current_arm.label}) did not measure: {detail}; no change proposed"


def quick_decision(user: SousConfig, arms: list[Arm], rows: list[BenchRow]) -> QuickChoice | None:
    by_label = {a.label: a for a in arms}
    mine = [
        r
        for r in rows
        if r.model_id == user.model_id and score(r) is not None and r.label in by_label
    ]
    if not mine:
        return None
    current_arm = next((a for a in arms if a.current), None)
    current = next((r for r in mine if by_label[r.label].current), None)
    # A strict improvement over the current arm is required: a tie changes
    # nothing, and a re-run on a noisy afternoon must not flip a setting.
    winner = current
    for row in mine:
        if winner is None or (score(row) or 0.0) > (score(winner) or 0.0):
            winner = row
    if winner is None:
        return None
    arm = by_label[winner.label]
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
    which = "decode at 16K context" if winner.decode_tps_16k is not None else "decode at 1K context"
    best = score(winner)
    reasons: list[str] = []
    if current.label != winner.label:
        current_best = score(current)
        reasons.append(
            f"{which}: {best:.1f} tok/s for {winner.label}, "
            f"{current_best:.1f} tok/s for the current arm ({current.label})"
        )
    else:
        reasons.append(f"{which}: {best:.1f} tok/s for {winner.label}")
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
