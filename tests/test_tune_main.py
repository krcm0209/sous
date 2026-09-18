import argparse
import dataclasses
import json
from datetime import date

import pytest

from sous.config import SousConfig, load_config
from sous.tune import main
from sous.tune.bench import BenchRow
from sous.tune.candidates import Candidate, Table
from sous.tune.candidates import describe as real_describe
from sous.tune.daemon import Readiness
from sous.tune.hardware import Hardware
from sous.tune.rundir import RunDir
from sous.tune.suite import SuiteTask
from sous.tune.suite.runner import SuiteOutcome, SuiteRun
from tests import tune_fixtures as fx

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _args(**over):
    base = dict(quick=True, models=None, repeat=1, runs=2, resume=None, yes=False, apply=False)
    base.update(over)
    return argparse.Namespace(**base)


def _hardware(tmp_path, working_set=fx.M5_PRO_WORKING_SET):
    return Hardware(
        chip="M5 Pro",
        memory_bytes=64 * 2**30,
        working_set_bytes=working_set,
        architecture="applegpu_g17s",
        nax=True,
        nax_reason=None,
        macos="26.6",
        versions={},
        hub_cache=str(tmp_path / "hub"),
        disk_free_bytes=10**12,
    )


def _table():
    return Table(
        checked=date.today(),
        validated_with="x",
        trusted_orgs=("mlx-community",),
        publishers=("Qwen",),
        candidates=(
            Candidate(M, "27b-dense", (D,)),
            Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ()),
        ),
    )


def _configs():
    return {
        M: fx.qwen_27b(),
        D: fx.dflash2_27b(),
        "mlx-community/Qwen3.5-9B-MLX-4bit": fx.qwen_9b(),
        "z-lab/Qwen3.5-9B-DFlash": fx.dflash_9b(),
    }


def _sizes():
    return {
        M: 16_100_000_000,
        D: 3_850_000_000,
        "mlx-community/Qwen3.5-9B-MLX-4bit": 6 * 10**9,
        "z-lab/Qwen3.5-9B-DFlash": 2_600_000_000,
    }


def _bench(scores):
    def bench(arm, **kw):
        s = scores.get(arm.label, 10.0)
        return BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=True,
            error=None,
            load_seconds=1.0,
            prefill_tps_2k=400.0,
            prefill_tps_16k=350.0,
            decode_tps_1k=s + 5,
            decode_tps_16k=s,
            ttft_seconds=1.0,
            peak_memory_bytes=1,
            spread=0.0,
        )

    return bench


def _tasks(tmp_path):
    return [
        SuiteTask(
            name=n,
            path=tmp_path,
            title=n,
            category="bug-fix",
            instructions="x",
            context_files=(),
            verify_commands=(),
            max_turns=1,
            max_minutes=1,
        )
        for n in ("a", "b")
    ]


def _suite(grades, seconds, seen):
    def suite(arm, tasks, *, runs, done, record, scratch, out, **kw):
        seen.append((arm.label, sorted(done)))
        results = []
        for task in tasks:
            for i in range(runs):
                if (task.name, i) in done:
                    continue
                r = SuiteRun(
                    task=task.name,
                    index=i,
                    label=arm.label,
                    model_id=arm.model_id,
                    drafter_id=arm.drafter_id,
                    block_size=arm.block_size,
                    int8_prefill=arm.int8_prefill,
                    greedy=arm.greedy,
                    window=arm.window,
                    state="done",
                    outcome="completed",
                    turns=3,
                    seconds=(seconds or {}).get(arm.label, 10.0),
                    output_tokens=100,
                    malformed=0,
                    repetitions=0,
                    approvals_denied=0,
                    grade=(grades or {}).get(arm.label, 1.0),
                    grade_detail="",
                    error=None,
                    transcript_path=None,
                )
                record(r)
                results.append(r)
        return SuiteOutcome(results, True, None)

    return suite


def _deps(
    tmp_path,
    *,
    scores,
    cached=(),
    answers=(),
    ready=None,
    isatty=True,
    grades=None,
    seconds=None,
    seen=None,
):
    ready = ready if ready is not None else Readiness(True, "no daemon")
    configs, sizes = _configs(), _sizes()
    it = iter(answers)
    downloaded = []
    return dict(
        detect=lambda: _hardware(tmp_path),
        load_table=lambda: _table(),
        describe=lambda mid, config_fn=None, size_fn=None: real_describe(
            mid, config_fn=lambda m: configs[m], size_fn=lambda m: sizes[m]
        ),
        ready=lambda port: ready,
        bench=_bench(scores),
        is_cached=lambda rid: rid in cached,
        size=lambda rid: sizes[rid],
        download=downloaded.append,
        ask=lambda prompt: (print(prompt, end=""), next(it, "n"))[1],
        isatty=lambda: isatty,
        managed=lambda: False,
        suite=_suite(grades, seconds, seen if seen is not None else []),
        load_tasks=lambda: _tasks(tmp_path),
        out=print,
    ), downloaded


def _cfg(tmp_path, **over):
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\nspeculative_block_size = 3\n")
    return SousConfig(data_dir=tmp_path, config_path=p, **over)


def test_a_quick_run_measures_the_fitting_arms_and_applies_on_yes(tmp_path, capsys):
    deps, downloaded = _deps(
        tmp_path,
        scores={
            "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0,
            "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3": 18.0,
        },
        cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"),
        answers=("y",),
    )
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "## Throughput" in out and "quality untested" in out
    assert "+speculative_block_size = 2" in out
    assert "Apply these changes" in out
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2
    assert "# mine" in (tmp_path / "config.toml").read_text()
    assert list((tmp_path / "tune").iterdir())  # the run dir with results and the report
    assert downloaded == []


def test_downloads_are_asked_one_by_one_and_a_refused_model_loses_its_arms(tmp_path, capsys):
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("n",))
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "Download #1" in out and "Qwen3.5-9B-MLX-4bit" in out
    assert downloaded == [] and "Qwen3.5-9B-MLX-4bit" not in out.split("## Throughput")[1]


def test_an_approved_download_is_fetched_before_any_measurement(tmp_path, capsys):
    order = []
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("y",))
    real_bench = deps["bench"]
    deps["bench"] = lambda arm, **kw: (order.append("bench"), real_bench(arm, **kw))[1]
    deps["download"] = lambda rid: order.append(f"download {rid}")
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert order[0] == "download mlx-community/Qwen3.5-9B-MLX-4bit"


def test_a_busy_daemon_refuses_before_anything_is_measured(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, ready=Readiness(False, "the daemon is busy: 1 task(s)"))
    called = []
    deps["bench"] = lambda arm, **kw: called.append(arm)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 2
    assert "busy" in capsys.readouterr().out and called == []


def test_a_model_that_does_not_fit_gets_guidance_and_no_diff(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    deps["detect"] = lambda: _hardware(tmp_path, working_set=fx.M2_AIR_WORKING_SET)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "refused" in out and "cannot recommend" in out and "Apply these changes" not in out


def test_without_a_tty_nothing_is_applied_unless_apply_is_passed(tmp_path, capsys):
    scores = {"Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0}
    deps, _ = _deps(
        tmp_path, scores=scores, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"), isatty=False
    )
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 3
    assert main(_args(apply=True), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2


def test_resume_skips_arms_already_in_the_run(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    seen = []
    real = deps["bench"]
    deps["bench"] = lambda arm, **kw: (seen.append(arm.label), real(arm, **kw))[1]
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    assert main(_args(resume=run_id), config=_cfg(tmp_path), **deps) == 0
    assert seen == [] and first


def test_models_naming_the_configured_model_outside_the_table_keeps_its_drafter(tmp_path, capsys):
    model_id = "mlx-community/Qwen3.5-9B-MLX-4bit"
    drafter_id = "z-lab/Qwen3.5-9B-DFlash"
    deps, _ = _deps(tmp_path, scores={}, cached=(model_id, drafter_id))
    # This id is deliberately absent here (unlike the shared _table(), where
    # it already sits as a drafterless "9b" candidate): --models must fall to
    # the manual-candidate default the fix carries the user's drafter onto.
    deps["load_table"] = lambda: Table(
        checked=date.today(),
        validated_with="x",
        trusted_orgs=("mlx-community",),
        publishers=("Qwen",),
        candidates=(Candidate(M, "27b-dense", (D,)),),
    )
    cfg = _cfg(tmp_path, model_id=model_id, speculative_draft_id=drafter_id)
    code = main(_args(models=[model_id]), config=cfg, **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "Qwen3.5-9B-MLX-4bit + Qwen3.5-9B-DFlash @3" in out.split("## Throughput")[1]
    assert "Apply these changes" not in out and "speculative_draft_id" not in out


def test_a_mistyped_resume_is_refused_not_a_traceback(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    assert main(_args(resume="nope"), config=_cfg(tmp_path), **deps) == 2
    assert "nope" in capsys.readouterr().out


def test_a_refused_release_stops_measuring_and_reports_what_was_done(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    real = deps["bench"]
    seen = []

    def bench(arm, **kw):
        seen.append(arm)
        row = real(arm, **kw)
        if len(seen) == 1:
            row = dataclasses.replace(
                row, released=False, error="unload refused: a generation is in flight"
            )
        return row

    deps["bench"] = bench
    assert main(_args(), config=_cfg(tmp_path), **deps) == 1
    out = capsys.readouterr().out
    assert "could not be released" in out and "a generation is in flight" in out
    assert "arm(s) skipped" in out and "--resume" in out
    assert "Apply these changes" not in out
    run_id = next(p.name for p in (tmp_path / "tune").iterdir())
    assert len(RunDir.existing(tmp_path / "tune", run_id).rows("bench")) == 1


def test_a_bench_that_raises_is_exit_1_and_keeps_finished_rows(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    real = deps["bench"]
    seen = []

    def bench(arm, **kw):
        seen.append(arm)
        if len(seen) == 2:
            raise RuntimeError("boom")
        return real(arm, **kw)

    deps["bench"] = bench
    assert main(_args(), config=_cfg(tmp_path), **deps) == 1
    assert "boom" in capsys.readouterr().out
    run_id = next(p.name for p in (tmp_path / "tune").iterdir())
    assert len(RunDir.existing(tmp_path / "tune", run_id).rows("bench")) == 1


def test_a_failed_download_ends_the_run_with_a_message_not_a_traceback(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D), answers=("y",))

    def failing(rid):
        raise OSError(28, "No space left on device")

    deps["download"] = failing
    assert main(_args(), config=_cfg(tmp_path), **deps) == 1
    out = capsys.readouterr().out
    assert "download failed" in out and "No space left" in out


def test_a_drafter_approved_for_a_refused_model_is_not_downloaded(tmp_path, capsys):
    """The prompts are independent, so a user may decline the 27B and accept
    its drafter; nothing left to measure names the drafter then."""
    deps, downloaded = _deps(tmp_path, scores={}, cached=(), answers=("n", "y", "y"))
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "Download #1" in out and "Download #2" in out and "Download #3" in out
    assert downloaded == ["mlx-community/Qwen3.5-9B-MLX-4bit"]


def test_downloads_that_exceed_the_free_disk_are_refused_before_fetching(tmp_path, capsys):
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("y",))
    hw = _hardware(tmp_path)
    deps["detect"] = lambda: dataclasses.replace(hw, disk_free_bytes=10**9)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 2
    assert "GiB is free" in capsys.readouterr().out and downloaded == []


def test_the_daemon_is_asked_again_right_before_the_first_load(tmp_path, capsys):
    """Consent and downloads take minutes; a session or a task that arrived
    since would load beside the bench."""
    answers = iter([Readiness(True, "no daemon"), Readiness(False, "a sous claude session holds")])
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    deps["ready"] = lambda port: next(answers)
    called = []
    deps["bench"] = lambda arm, **kw: called.append(arm)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 2
    assert "sous claude" in capsys.readouterr().out and called == []


def test_resume_measures_an_arm_again_when_its_window_changed(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    seen = []
    real = deps["bench"]
    deps["bench"] = lambda arm, **kw: (seen.append(arm.label), real(arm, **kw))[1]
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    smaller = _cfg(tmp_path, max_context_tokens=16384)
    assert main(_args(resume=run_id), config=smaller, **deps) == 0
    assert seen == first


def test_resume_survives_a_torn_results_line(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    results = tmp_path / "tune" / run_id / "results.jsonl"
    with results.open("a") as f:
        f.write('{"kind": "bench", "label": "torn')
    assert main(_args(resume=run_id), config=_cfg(tmp_path), **deps) == 0


def test_a_misspelt_collaborator_is_an_error_not_a_run_against_the_real_daemon(tmp_path):
    deps, _ = _deps(tmp_path, scores={})
    deps["raedy"] = deps.pop("ready")
    with pytest.raises(TypeError):
        main(_args(), config=_cfg(tmp_path), **deps)


N = "mlx-community/Qwen3.5-9B-MLX-4bit"
NINE = "Qwen3.5-9B-MLX-4bit"
CUR = "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3"


def test_a_full_run_grades_each_models_fastest_arm_and_can_change_the_model(tmp_path, capsys):
    seen = []
    deps, _ = _deps(
        tmp_path,
        scores={CUR: 30.0, "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @5": 28.0, NINE: 45.0},
        cached=(M, D, N),
        grades={NINE: 0.9, CUR: 0.92},
        seconds={NINE: 5.0, CUR: 20.0},
        seen=seen,
    )
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    # The model stage: each model's fastest arm, the current one among them.
    assert [label for label, _ in seen[:2]] == [CUR, NINE]
    # The winner stage on the 9B: nax is on and the fixture is affine 4-bit gs64.
    assert [label for label, _ in seen[2:]] == [f"{NINE} + int8 prefill", f"{NINE} greedy"]
    assert "## Suite" in out and "suite: 2 tasks x 2 runs on 2 arm(s)" in out
    assert "roughly" in out and "min from the measured speeds" in out
    assert f'+id = "{N}"' in out
    assert "restart the daemon to apply id, speculative_draft_id" in out
    assert f'id = "{N}"' in (tmp_path / "config.toml").read_text()


def test_the_winner_stage_offers_only_what_the_hardware_allows(tmp_path, capsys):
    seen = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N), seen=seen)
    no_nax = dataclasses.replace(_hardware(tmp_path), nax=False, nax_reason="pre-M5")
    deps["detect"] = lambda: no_nax
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    assert [label for label, _ in seen if "int8" in label] == []
    assert [label for label, _ in seen if label.endswith(" greedy")]


def test_a_greedy_arm_that_wins_writes_temperature_zero(tmp_path, capsys):
    deps, _ = _deps(
        tmp_path,
        scores={CUR: 30.0, NINE: 20.0},
        cached=(M, D, N),
        seconds={f"{CUR} greedy": 5.0},
    )
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "+temperature = 0.0" in out and "+id" not in out


def test_resume_skips_suite_runs_already_recorded(tmp_path, capsys):
    seen = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N), seen=seen)
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    assert main(_args(quick=False, resume=run_id), config=_cfg(tmp_path), **deps) == 0
    assert first and seen == []  # every (task, run) of every arm is already on disk
    results = (tmp_path / "tune" / run_id / "results.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in results]
    assert {r["kind"] for r in rows} == {"bench", "suite"}


def test_a_suite_arm_whose_weights_stay_resident_stops_the_run(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))
    stuck = "unload refused: held by 1 session(s)"
    deps["suite"] = lambda arm, tasks, **kw: SuiteOutcome([], False, stuck)
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 1
    out = capsys.readouterr().out
    assert "could not be released (unload refused: held by 1 session(s))" in out
    assert "--resume" in out


def test_the_daemon_is_asked_again_before_every_suite_arm(tmp_path, capsys):
    calls = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))
    real = deps["ready"]
    deps["ready"] = lambda port: (calls.append(port), real(port))[1]
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 0
    # once up front, once before the first bench load, once per suite arm
    # (two models, then two winner-stage arms)
    assert len(calls) == 2 + 4


def test_a_quick_run_never_touches_the_suite(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))

    def never(*a, **k):
        raise AssertionError("the suite ran in a quick run")

    deps["suite"] = never
    deps["load_tasks"] = never
    assert main(_args(quick=True), config=_cfg(tmp_path), **deps) == 0
    assert "## Suite" not in capsys.readouterr().out
