"""`sous tune`: benchmark this machine, grade the candidates, pick the
settings, show the diff. `--quick` stops after the throughput stage."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date

from sous.config import SousConfig, load_config
from sous.tune import arms as arms_mod
from sous.tune import bench as bench_mod
from sous.tune import candidates as cand_mod
from sous.tune import daemon as daemon_mod
from sous.tune import hardware as hw_mod
from sous.tune import hub as hub_mod
from sous.tune import report as report_mod
from sous.tune import suite as suite_mod
from sous.tune.decide import (
    ArmSummary,
    FullChoice,
    QuickChoice,
    full_decision,
    model_stage,
    quick_decision,
    summarize,
)
from sous.tune.rundir import RunDir
from sous.tune.suite import runner as runner_mod

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
_GIB = 1 << 30


def _managed() -> bool:
    from sous.cli import LABEL, _launchd_loaded

    return _launchd_loaded(LABEL)


def _describe_all(ids: set[str], describe, out) -> dict[str, cand_mod.Checkpoint]:
    found: dict[str, cand_mod.Checkpoint] = {}
    for model_id in sorted(ids):
        try:
            found[model_id] = describe(model_id)
        except Exception as e:  # noqa: BLE001 — an unreadable checkpoint is skipped, named
            out(f"  {model_id}: cannot read its config ({type(e).__name__}); skipped")
    return found


def _consented(arms, plan, approved) -> list:
    """Arms whose snapshots are all on disk or approved: a refused model loses
    every arm, a refused drafter leaves the model's no-drafter arm."""
    refused = {d.repo_id for d in plan} - approved
    return [a for a in arms if a.model_id not in refused and a.drafter_id not in refused]


def _snapshots(arms) -> set[str]:
    return {a.model_id for a in arms} | {a.drafter_id for a in arms if a.drafter_id}


def _manual_candidate(model_id: str, user: SousConfig) -> cand_mod.Candidate:
    """An id named by --models but absent from the table gets no curated
    drafter — unless it is the user's own configured model, where dropping
    the drafter would measure one undrafted arm and then propose removing it."""
    drafters = ()
    if model_id == user.model_id and user.speculative_draft_id:
        drafters = (user.speculative_draft_id,)
    return cand_mod.Candidate(model_id, "manual", drafters)


def _suite_stage(
    stage: list[arms_mod.Arm],
    tasks: list[suite_mod.SuiteTask],
    runs_per_task: int,
    run: RunDir,
    suite_runs: list[runner_mod.SuiteRun],
    *,
    suite: Callable[..., runner_mod.SuiteOutcome],
    ready: Callable[[int], daemon_mod.Readiness],
    user: SousConfig,
    out: Callable[..., None],
) -> str | None:
    """Every arm's (task, run) not already in the run's rows, one arm at a
    time. Returns the text of what must stop the run — a daemon that is no
    longer free, or weights left resident — else None."""
    for arm in stage:
        done = {
            (r.task, r.index)
            for r in suite_runs
            if r.key == arm.suite_key and r.window == arm.window
        }
        if len(done) >= len(tasks) * runs_per_task:
            continue
        # Minutes to hours have passed since the last check: a session or a
        # task that arrived since would load beside the suite's engine.
        readiness = ready(user.server_port)
        if not readiness.ready:
            return f"daemon: {readiness.reason}; re-run with --resume {run.run_id} once it is free"
        out(f"  {arm.label}: suite ...")

        def record(r: runner_mod.SuiteRun) -> None:
            run.append("suite", r.as_dict())
            suite_runs.append(r)

        outcome = suite(
            arm,
            tasks,
            runs=runs_per_task,
            done=done,
            record=record,
            scratch=run.path / "suite",
            out=out,
        )
        if outcome.error:
            out(f"  {arm.label}: {outcome.error}")
        if not outcome.released:
            return (
                f"the model could not be released ({outcome.error}); restart the daemon or "
                f"this process and re-run with --resume {run.run_id}"
            )
    return None


def _summaries(
    arms: list[arms_mod.Arm],
    suite_runs: list[runner_mod.SuiteRun],
    rows: list[bench_mod.BenchRow],
) -> list[ArmSummary]:
    """Every arm with a suite run to summarize — including a stage a stop cut
    short, so a run's report always matches what results.jsonl holds."""
    return [s for a in arms if (s := summarize(a, suite_runs, rows)) is not None]


def _full_stages(
    user: SousConfig,
    hardware: hw_mod.Hardware,
    checkpoints: dict[str, cand_mod.Checkpoint],
    arms: list[arms_mod.Arm],
    rows: list[bench_mod.BenchRow],
    tasks: list[suite_mod.SuiteTask],
    runs_per_task: int,
    run: RunDir,
    *,
    suite: Callable[..., runner_mod.SuiteOutcome],
    ready: Callable[[int], daemon_mod.Readiness],
    out: Callable[..., None],
) -> tuple[FullChoice | None, list[ArmSummary], str | None]:
    """The model stage, then the winner stage on its choice. Returns the
    choice, the summaries the report shows, and the text of a failure that
    stops the run."""
    stage = model_stage(arms, rows)
    eta = runner_mod.estimate_seconds(stage, rows, tasks=len(tasks), runs=runs_per_task)
    line = f"suite: {len(tasks)} tasks x {runs_per_task} runs on {len(stage)} arm(s)"
    if eta is not None:
        line += f"; roughly {eta / 60:.0f} min from the measured speeds"
    out(line)
    suite_runs = [runner_mod.SuiteRun.from_dict(r) for r in run.rows("suite")]
    stop = _suite_stage(
        stage, tasks, runs_per_task, run, suite_runs, suite=suite, ready=ready, user=user, out=out
    )
    if stop is not None:
        return None, _summaries(stage, suite_runs, rows), stop
    choice = full_decision(user, stage, suite_runs, rows, runs_per_task=runs_per_task)
    extra: list[arms_mod.Arm] = []
    if choice is not None:
        winner = choice.arm
        extra = arms_mod.winner_stage_arms(
            winner, nax=hardware.nax, checkpoint=checkpoints[winner.model_id]
        )
        if extra:
            out(f"winner stage on {winner.label}: " + ", ".join(a.label for a in extra))
            stop = _suite_stage(
                extra,
                tasks,
                runs_per_task,
                run,
                suite_runs,
                suite=suite,
                ready=ready,
                user=user,
                out=out,
            )
            if stop is not None:
                return None, _summaries([*stage, *extra], suite_runs, rows), stop
            choice = full_decision(
                user,
                [winner, *extra],
                suite_runs,
                rows,
                runs_per_task=runs_per_task,
                reference=winner,
            )
    return choice, _summaries([*stage, *extra], suite_runs, rows), None


def main(
    args: argparse.Namespace,
    *,
    config: SousConfig | None = None,
    out: Callable[..., None] = print,
    detect: Callable[[], hw_mod.Hardware] = hw_mod.detect,
    load_table: Callable[[], cand_mod.Table] = cand_mod.load_table,
    ready: Callable[[int], daemon_mod.Readiness] = daemon_mod.ready_for_tune,
    describe: Callable[[str], cand_mod.Checkpoint] = cand_mod.describe,
    is_cached: Callable[[str], bool] = hub_mod.is_cached,
    size: Callable[[str], int | None] | None = None,
    download: Callable[[str], object] | None = None,
    ask: Callable[[str], str] = input,
    bench: Callable[..., bench_mod.BenchRow] = bench_mod.bench_arm,
    suite: Callable[..., runner_mod.SuiteOutcome] = runner_mod.run_suite,
    load_tasks: Callable[[], list[suite_mod.SuiteTask]] = suite_mod.load_tasks,
    isatty: Callable[[], bool] | None = None,
    managed: Callable[[], bool] = _managed,
) -> int:
    """Every collaborator is a keyword parameter with the real one as its
    default, so a test's injection is checked at the call and a misspelt
    one is an error rather than a run against the live daemon or the Hub."""
    quick = bool(getattr(args, "quick", False))
    user = config or load_config()
    hardware = detect()
    table = load_table()
    readiness = ready(user.server_port)
    out(f"daemon: {readiness.reason}")
    if not readiness.ready:
        return EXIT_REFUSED

    candidates = list(table.candidates)
    if getattr(args, "models", None):
        candidates = [
            next((c for c in table.candidates if c.id == m), _manual_candidate(m, user))
            for m in args.models
        ]
    ids = {c.id for c in candidates} | {d for c in candidates for d in c.drafters}
    ids.add(user.model_id)
    if user.speculative_draft_id:
        ids.add(user.speculative_draft_id)
    checkpoints = _describe_all(ids, describe, out)
    arms, refusals = arms_mod.quick_arms(
        user, candidates, checkpoints, working_set_bytes=hardware.working_set_bytes
    )
    if not arms:
        out("no candidate fits this machine at the configured window:")
        for r in refusals:
            out(f"  {r.model_id}: {r.reason}")
        return EXIT_REFUSED

    items = []
    for a in arms:
        items.append((a.model_id, f"candidate ({a.tier} tier)", "its arms"))
        if a.drafter_id:
            items.append(
                (a.drafter_id, f"drafter for {a.model_id}", f"{a.model_id} runs without a drafter")
            )
    if size is None:
        # describe() already asked the Hub for every checkpoint's size.
        def size(repo_id: str) -> int | None:
            cp = checkpoints.get(repo_id)
            return cp.bytes if cp is not None else hub_mod.snapshot_bytes(repo_id)

    plan = hub_mod.plan_downloads(items, cached=is_cached, size=size)
    if getattr(args, "yes", False):
        approved = {d.repo_id for d in plan}
    else:
        approved = hub_mod.ask_consent(
            plan, hardware.disk_free_bytes, hub_cache=hardware.hub_cache, ask=ask, out=out
        )
    arms = _consented(arms, plan, approved)
    if not arms:
        out("nothing left to measure after the downloads you declined")
        return EXIT_REFUSED
    # Only what a surviving arm names: a drafter approved for a model that
    # was refused has no arm left to serve.
    fetching = sorted(approved & _snapshots(arms))
    wanted = sum(d.bytes or 0 for d in plan if d.repo_id in fetching)
    if wanted > hardware.disk_free_bytes:
        out(
            f"sous tune: the approved downloads need {wanted / _GIB:.1f} GiB and "
            f"{hardware.disk_free_bytes / _GIB:.1f} GiB is free at {hardware.hub_cache}"
        )
        return EXIT_REFUSED
    try:
        hub_mod.fetch(fetching, download=download, out=out)
    except Exception as e:  # noqa: BLE001 — a failed download ends the run, named
        out(f"sous tune: download failed: {type(e).__name__}: {e}")
        return EXIT_FAILED

    base = user.data_dir / "tune"
    resume = getattr(args, "resume", None)
    if resume:
        try:
            run = RunDir.existing(base, resume)
        except FileNotFoundError as e:
            out(f"sous tune: {e}")
            return EXIT_REFUSED
    else:
        run = RunDir.new(base)
        run.write_json("hardware.json", hardware.as_dict())
    rows = [bench_mod.BenchRow.from_dict(r) for r in run.rows("bench")]
    # An arm is done when a row of it was measured at this run's window: a
    # resume under a changed config measures it again rather than compare
    # rows of two windows.
    done = {(r.key, r.window) for r in rows}
    todo = [a for a in arms if (a.key, a.window) not in done]
    out(f"measuring {len(todo)} arm(s); results in {run.path}")
    released_failure = False
    if todo:
        # The first reading was a point in time, and consent and downloads
        # may have taken minutes: a session or a task that arrived since
        # would load beside the bench.
        readiness = ready(user.server_port)
        if not readiness.ready:
            out(f"daemon: {readiness.reason}")
            return EXIT_REFUSED
    try:
        for i, arm in enumerate(todo):
            out(f"  {arm.label} ...")
            row = bench(arm, repeat=getattr(args, "repeat", 2), out=out)
            run.append("bench", row.as_dict())
            rows.append(row)
            if not row.released:
                # Two models resident at once would corrupt every later
                # peak-memory reading in this run: stop rather than measure
                # the remaining arms against a machine that isn't clean.
                remaining = len(todo) - i - 1
                out(
                    f"sous tune: the model could not be released ({row.error}); "
                    f"{remaining} arm(s) skipped — restart the daemon or this process "
                    f"and re-run with --resume {run.run_id}"
                )
                released_failure = True
                break
    except Exception as e:  # noqa: BLE001 — rows already appended stay on disk for --resume
        out(f"sous tune: failed while measuring: {type(e).__name__}: {e}")
        return EXIT_FAILED

    choice: QuickChoice | FullChoice | None
    summaries: list[ArmSummary] = []
    tasks: list[suite_mod.SuiteTask] = []
    runs_per_task = getattr(args, "runs", 2)
    if quick or released_failure:
        choice = quick_decision(user, arms, rows) if quick else None
    else:
        tasks = load_tasks()
        choice, summaries, stop = _full_stages(
            user,
            hardware,
            checkpoints,
            arms,
            rows,
            tasks,
            runs_per_task,
            run,
            suite=suite,
            ready=ready,
            out=out,
        )
        if stop is not None:
            out(f"sous tune: {stop}")
            released_failure = True
    text = report_mod.render_report(
        hardware=hardware,
        table_age_days=(date.today() - table.checked).days,
        checkpoints=checkpoints,
        refusals=refusals,
        rows=rows,
        choice=choice,
        current_model=user.model_id,
        quick=quick,
        suite=summaries,
        tasks=len(tasks),
        runs=runs_per_task,
    )
    run.write_text("report.md", text)
    out(text)
    if released_failure:
        return EXIT_FAILED
    if choice is None or not choice.changes:
        return EXIT_OK
    new_text, diff = report_mod.config_diff(user.config_path, choice.changes)
    out(diff)
    apply = getattr(args, "apply", False) or getattr(args, "yes", False)
    if not apply:
        if not (isatty or sys.stdin.isatty)():
            out("not a terminal: nothing applied (pass --apply to write the changes)")
            return EXIT_OK
        apply = hub_mod.confirm(f"Apply these changes to {user.config_path}? [y/N] ", ask)
    if not apply:
        out("nothing applied")
        return EXIT_OK
    backup = report_mod.apply_changes(user.config_path, new_text, run.run_id)
    out(f"applied; backup at {backup}" if backup else "applied")
    note = report_mod.restart_note(choice.changes, managed=managed())
    if note:
        out(note)
    return EXIT_OK
