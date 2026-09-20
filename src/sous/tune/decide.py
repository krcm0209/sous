"""The rules. Quick: among the current model's quality-neutral arms, the
fastest decode wins, and the model itself is never this rule's to change.
Full: the fastest arm whose suite result is within a fixed margin of the
reference, printed with every number so the choice explains itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.hardware import gib
from sous.tune.suite.runner import SuiteRun

MARGIN = 0.05
# The floor and every mean grade are float arithmetic over pass fractions
# (thirds, sevenths, ...): compared raw, an arm sitting exactly at the
# margin is rejected on rounding noise the printed reason (rounded to two
# decimals) can't show, so the line reads as a self-contradiction. The
# boundary decided here must be the one the reasons print.
_GRADE_EPSILON = 1e-9


@dataclass(frozen=True)
class QuickChoice:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
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


def _changes(user: SousConfig, arm: Arm, *, full: bool = False) -> dict[str, dict[str, object]]:
    model: dict[str, object] = {}
    # Only the full run may write the model and the quality-affecting keys:
    # a quick run measured throughput and nothing about what the model says.
    if full and arm.model_id != user.model_id:
        model["id"] = arm.model_id
    if arm.drafter_id != user.speculative_draft_id:
        model["speculative_draft_id"] = arm.drafter_id
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
    # The arm's own fit, not the model's shared measurement window: that one
    # reserves memory for the heaviest drafter, which this arm may not load.
    window = arm.fit_window if arm.fit_window is not None else arm.window
    if window < user.max_context_tokens:
        model["max_context_tokens"] = window
    if full and arm.int8_prefill != user.int8_prefill:
        model["int8_prefill"] = arm.int8_prefill
    if full and arm.greedy and user.temperature != 0:
        model["temperature"] = 0.0
    changes: dict[str, dict[str, object]] = {}
    if model:
        changes["model"] = model
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
        changes=_changes(user, arm),
        reasons=reasons,
    )


@dataclass(frozen=True)
class ArmSummary:
    label: str
    key: tuple[str, str, int, bool, bool]
    model_id: str
    runs: int
    completed: int
    mean_grade: float
    wall_seconds: float
    repetitions: int
    malformed: int
    approvals_denied: int
    output_tokens: int
    peak_memory_bytes: int | None


def summarize(arm: Arm, runs: list[SuiteRun], rows: list[BenchRow]) -> ArmSummary | None:
    """The arm's suite result in the rule's terms; None when it has no run.
    The peak comes from the bench row of the same model, drafter and block:
    the suite measures time and quality, the bench measured memory."""
    # A resumed run whose window changed carries suite rows of two windows
    # (see _suite_stage's own done-set, keyed the same way): only rows at
    # this arm's window belong to it, or wall time and completed counts mix
    # two different measurements.
    mine = [r for r in runs if r.key == arm.suite_key and r.window == arm.window]
    if not mine:
        return None
    # A retried (task, index) — _suite_stage no longer treats a runner-only
    # "error" row as done, so it runs again — leaves the stale row in
    # results.jsonl beside the fresh one; only the later row (last write
    # wins) should count, or a successful retry gets summed with the
    # failure it replaced.
    latest: dict[tuple[str, int], SuiteRun] = {}
    for r in mine:
        latest[(r.task, r.index)] = r
    mine = list(latest.values())
    peak = next(
        (r.peak_memory_bytes for r in rows if r.ok and r.key == arm.key and r.window == arm.window),
        None,
    )
    return ArmSummary(
        label=arm.label,
        key=arm.suite_key,
        model_id=arm.model_id,
        runs=len(mine),
        completed=sum(1 for r in mine if r.completed),
        mean_grade=sum(r.grade for r in mine) / len(mine),
        wall_seconds=sum(r.seconds for r in mine),
        repetitions=sum(r.repetitions for r in mine),
        malformed=sum(r.malformed for r in mine),
        approvals_denied=sum(r.approvals_denied for r in mine),
        output_tokens=sum(r.output_tokens for r in mine),
        peak_memory_bytes=peak,
    )


def model_stage(arms: list[Arm], rows: list[BenchRow]) -> list[Arm]:
    """What the suite runs in the model stage: each model's fastest
    quality-neutral arm by the bench (one scoring unit per model, as the
    quick rule scores; a tie keeps the current arm, then the drafterless
    one), and the current arm besides, so the reference is always measured.
    A model none of whose arms measured is not a candidate for the suite."""
    by_key = {a.key: a for a in arms}
    ok = [r for r in rows if r.ok and r.key in by_key and r.window == by_key[r.key].window]
    chosen: list[Arm] = []
    seen: set[str] = set()
    for arm in arms:
        if arm.model_id in seen:
            continue
        seen.add(arm.model_id)
        mine = [r for r in ok if r.model_id == arm.model_id]
        long = all(r.decode_tps_16k is not None for r in mine)
        scored = [(r, s) for r in mine if (s := score(r, long)) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda rs: (rs[1], by_key[rs[0].key].current, not rs[0].drafter_id))
        chosen.append(by_key[best[0].key])
    current = next((a for a in arms if a.current), None)
    if current is not None and current not in chosen and any(r.key == current.key for r in ok):
        chosen.append(current)
    return chosen


@dataclass(frozen=True)
class FullChoice:
    label: str
    arm: Arm
    reference_label: str
    changes: dict[str, dict[str, object]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _peak_text(s: ArmSummary) -> str:
    return "-" if s.peak_memory_bytes is None else gib(s.peak_memory_bytes)


def full_decision(
    user: SousConfig,
    arms: list[Arm],
    runs: list[SuiteRun],
    rows: list[BenchRow],
    *,
    runs_per_task: int,
    reference: Arm | None = None,
) -> FullChoice | None:
    """The eligible arm with the lowest suite wall time. Eligible: mean
    grade within MARGIN of the reference's, completed runs within one
    task's worth of the reference's, no more repetition incidents. The
    reference is the current arm, or the first measured one when the
    configured model did not fit (the largest fitting tier: the table is
    ordered largest first), or the arm given — the winner stage's winner.
    Every number goes into `reasons`, so the report is the rule."""
    summaries = [(a, s) for a in arms if (s := summarize(a, runs, rows)) is not None]
    if not summaries:
        return None
    if reference is not None:
        ref = next(((a, s) for a, s in summaries if a.suite_key == reference.suite_key), None)
        why = "the model stage's winner"
    else:
        ref = next(((a, s) for a, s in summaries if a.current), None)
        why = "the configured arm"
        if ref is None:
            ref = summaries[0]
            why = (
                "the first measured arm in the table's order (largest tier first); "
                "the configured arm has no runs"
            )
    if ref is None:
        return None
    ref_arm, ref_sum = ref
    grade_floor = ref_sum.mean_grade - MARGIN
    completed_floor = ref_sum.completed - runs_per_task
    reasons = [
        f"reference: {ref_arm.label} ({why})",
        f"eligible: mean grade >= {grade_floor:.2f} ({ref_sum.mean_grade:.2f} - {MARGIN:.2f}), "
        f"completed runs >= {completed_floor} ({ref_sum.completed} - {runs_per_task} per task), "
        f"repetition incidents <= {ref_sum.repetitions}",
    ]
    if ref_sum.completed == 0:
        reasons.append(
            "note: the reference completed no run, so the completed-runs floor "
            "constrains nothing; its grade and repetition terms still apply"
        )
    reasons.append(
        "winner: the eligible arm with the lowest suite wall time; "
        "a tie goes to the smaller peak memory"
    )
    eligible: list[tuple[Arm, ArmSummary]] = []
    for arm, s in summaries:
        problems = []
        if s.mean_grade + _GRADE_EPSILON < grade_floor:
            problems.append(f"grade {s.mean_grade:.2f} < {grade_floor:.2f}")
        if s.completed < completed_floor:
            problems.append(f"completed {s.completed} < {completed_floor}")
        if s.repetitions > ref_sum.repetitions:
            problems.append(f"repetitions {s.repetitions} > {ref_sum.repetitions}")
        line = (
            f"{arm.label}: grade {s.mean_grade:.2f}, completed {s.completed}/{s.runs}, "
            f"wall {s.wall_seconds:.0f} s, repetitions {s.repetitions}, peak {_peak_text(s)}"
        )
        if problems:
            reasons.append(f"{line} -> not eligible: {'; '.join(problems)}")
        else:
            reasons.append(f"{line} -> eligible")
            eligible.append((arm, s))
    # The reference is eligible against itself by construction. It goes
    # first so an exact tie keeps it: a re-run must not flip the model on
    # equal numbers.
    eligible.sort(key=lambda e: e[0] is not ref_arm)
    winner, w = min(
        eligible,
        key=lambda e: (
            e[1].wall_seconds,
            e[1].peak_memory_bytes if e[1].peak_memory_bytes is not None else float("inf"),
        ),
    )
    if winner is ref_arm:
        reasons.append(f"winner: {ref_arm.label} (the reference; nothing eligible is faster)")
    else:
        reasons.append(
            f"winner: {winner.label} ({w.wall_seconds:.0f} s vs {ref_sum.wall_seconds:.0f} s "
            f"for the reference)"
        )
    return FullChoice(
        label=winner.label,
        arm=winner,
        reference_label=ref_arm.label,
        changes=_changes(user, winner, full=True),
        reasons=reasons,
    )
