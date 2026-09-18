"""`sous tune`: benchmark this machine, pick the settings, show the diff."""

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
from sous.tune.decide import quick_decision
from sous.tune.rundir import RunDir

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _managed() -> bool:
    from sous.cli import LABEL, _launchd_loaded

    return _launchd_loaded(LABEL)


def _label() -> str:
    from sous.cli import LABEL

    return LABEL


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


def _manual_candidate(model_id: str, user: SousConfig) -> cand_mod.Candidate:
    """An id named by --models but absent from the table gets no curated
    drafter — unless it is the user's own configured model, where dropping
    the drafter would measure one undrafted arm and then propose removing it."""
    drafters = ()
    if model_id == user.model_id and user.speculative_draft_id:
        drafters = (user.speculative_draft_id,)
    return cand_mod.Candidate(model_id, "manual", drafters)


def main(args: argparse.Namespace, *, config: SousConfig | None = None, **deps) -> int:
    out: Callable[..., None] = deps.get("out", print)
    if not getattr(args, "quick", False):
        out("sous tune: only the quick stage exists in this version; run `sous tune --quick`")
        return EXIT_REFUSED
    user = config or load_config()
    hardware = deps.get("detect", hw_mod.detect)()
    table = deps.get("load_table", cand_mod.load_table)()
    ready = deps.get("ready", daemon_mod.ready_for_tune)(user.server_port)
    out(f"daemon: {ready.reason}")
    if not ready.ready:
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
    describe = deps.get("describe", cand_mod.describe)
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
    plan = hub_mod.plan_downloads(
        items,
        cached=deps.get("is_cached", hub_mod.is_cached),
        size=deps.get("size", hub_mod.snapshot_bytes),
    )
    if getattr(args, "yes", False):
        approved = {d.repo_id for d in plan}
    else:
        approved = hub_mod.ask_consent(
            plan,
            hardware.disk_free_bytes,
            hub_cache=hardware.hub_cache,
            ask=deps.get("ask", input),
            out=out,
        )
    arms = _consented(arms, plan, approved)
    if not arms:
        out("nothing left to measure after the downloads you declined")
        return EXIT_REFUSED
    hub_mod.fetch(sorted(approved), download=deps.get("download"), out=out)

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
    done = {r["label"] for r in run.rows("bench")}
    bench = deps.get("bench", bench_mod.bench_arm)
    out(f"measuring {len(arms)} arm(s); results in {run.path}")
    released_failure = False
    try:
        for i, arm in enumerate(arms):
            if arm.label in done:
                continue
            out(f"  {arm.label} ...")
            row = bench(arm, repeat=getattr(args, "repeat", 2), out=out)
            run.append("bench", row.as_dict())
            if not row.released:
                # Two models resident at once would corrupt every later
                # peak-memory reading in this run: stop rather than measure
                # the remaining arms against a machine that isn't clean.
                remaining = len(arms) - i - 1
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
    rows = [bench_mod.BenchRow.from_dict(r) for r in run.rows("bench")]

    choice = quick_decision(user, arms, rows)
    text = report_mod.render_report(
        hardware=hardware,
        table_age_days=(date.today() - table.checked).days,
        checkpoints=checkpoints,
        refusals=refusals,
        rows=rows,
        choice=choice,
        current_model=user.model_id,
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
        if not deps.get("isatty", sys.stdin.isatty)():
            out("not a terminal: nothing applied (pass --apply to write the changes)")
            return EXIT_OK
        try:
            answer = deps.get("ask", input)(f"Apply these changes to {user.config_path}? [y/N] ")
        except EOFError:
            answer = ""
        apply = answer.strip().lower() in ("y", "yes")
    if not apply:
        out("nothing applied")
        return EXIT_OK
    backup = report_mod.apply_changes(user.config_path, new_text, run.run_id)
    out(f"applied; backup at {backup}" if backup else "applied")
    note = report_mod.restart_note(
        choice.changes, managed=deps.get("managed", _managed)(), label=_label()
    )
    if note:
        out(note)
    return EXIT_OK
