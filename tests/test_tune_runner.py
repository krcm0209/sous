"""The suite runner drives the suite's own loop over a scratch copy of the
project: the tools are real, a disallowed command is denied and counted, and
every number in a run is read back from the transcript and the engine's
deltas."""

import dataclasses
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import ManagedEngine, ReplaySafe
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import (
    CountingEngine,
    SuiteRun,
    arm_runs,
    estimate_seconds,
    interpreter_first,
    loaded_tile,
    metrics_from_transcript,
    run_one,
    run_suite,
)
from tests.fake_engine import FakeEngine

PYTHON = Path(sys.executable)
CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "done", "concerns": ""}')


def _task(name="implement_rpn"):
    return next(t for t in load_tasks() if t.name == name)


def _arm(tmp_path, **over):
    cfg = SousConfig(
        data_dir=tmp_path / "never-used",
        config_path=tmp_path / "never-used.toml",
        model_id="org/m",
        speculative_draft_id="",
        max_context_tokens=8192,
        **over,
    )
    return Arm(
        label="m",
        config=cfg,
        model_id="org/m",
        drafter_id="",
        block_size=0,
        window=8192,
        tier="t",
        current=True,
    )


def _solution_write(task):
    content = (task.solution / "rpn.py").read_text()
    return CALL.format(name="write_file", args=json.dumps({"path": "rpn.py", "content": content}))


def test_metrics_from_a_transcript_count_malformed_calls_and_repetition_streaks(tmp_path):
    p = tmp_path / "transcript.jsonl"
    same = {"event": "tool", "name": "read_file", "arguments": {"path": "a.py"}, "result": "x"}
    other = {"event": "tool", "name": "read_file", "arguments": {"path": "b.py"}, "result": "y"}
    events = [
        {"event": "generation", "turn": 1, "text": "..."},
        {"event": "malformed", "error": "no call"},
        same,
        same,
        same,
        same,  # one streak of four is one incident
        other,
        {"event": "malformed", "error": "bad json"},
        same,
        same,
        same,  # a second streak is a second incident
        {"event": "finished", "outcome": "completed"},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    assert metrics_from_transcript(p) == (2, 2)
    assert metrics_from_transcript(tmp_path / "missing.jsonl") == (0, 0)


def test_interpreter_first_prepends_the_interpreters_directory_and_restores(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    with interpreter_first(Path("/venv/bin/python3.14")):
        assert os.environ["PATH"].split(os.pathsep) == ["/venv/bin", "/usr/bin"]
    assert os.environ["PATH"] == "/usr/bin"


def test_a_counting_engine_reads_the_deltas_through_a_replay_safe_callback():
    inner = FakeEngine(["one two three", "four"])
    counting = CountingEngine(inner)
    counting.generate([], [], 8)
    counting.generate([], [], 8)
    assert counting.output_tokens == 4
    assert all(isinstance(cb, ReplaySafe) for cb in inner.on_deltas_seen)
    assert counting.model_id == "fake/model"
    assert ManagedEngine(counting).drafter is None


def test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run(tmp_path):
    task = _task()
    inner = FakeEngine([_solution_write(task), FINISH])
    counting = CountingEngine(inner)
    engine = ManagedEngine(counting)
    run = run_one(task, 0, _arm(tmp_path), engine, counting, tmp_path / "s", python=PYTHON)
    assert run.task == "implement_rpn" and run.index == 0 and run.label == "m"
    assert run.state == "done" and run.outcome == "completed" and run.completed
    assert run.turns == 2 and run.grade == 1.0, run.grade_detail
    assert run.output_tokens == counting.output_tokens > 0
    assert run.malformed == 0 and run.repetitions == 0 and run.approvals_denied == 0
    assert run.error is None and run.seconds >= 0
    assert run.transcript_path == str(tmp_path / "s" / "transcript.jsonl")
    assert Path(run.transcript_path).is_file()
    written = (tmp_path / "s" / "project" / "rpn.py").read_text()
    assert written == (task.solution / "rpn.py").read_text()
    # The fake engine reports no projection tile: under true, an engine without it.
    assert run.key == (*_arm(tmp_path).key, False, False, True, True, True, "absent")


def test_a_command_outside_the_allowlist_is_denied_and_counted(tmp_path):
    task = _task()
    denied = CALL.format(name="run_command", args='{"command": "rm -rf /"}')
    allowed = CALL.format(name="run_command", args='{"command": "python -m unittest discover"}')
    inner = FakeEngine([denied, allowed, FINISH])
    counting = CountingEngine(inner)
    run = run_one(
        task, 1, _arm(tmp_path), ManagedEngine(counting), counting, tmp_path / "s", python=PYTHON
    )
    assert run.approvals_denied == 1 and run.state == "done" and run.index == 1
    transcript_path = Path(run.transcript_path)  # ty: ignore[invalid-argument-type]
    lines = [json.loads(l) for l in transcript_path.read_text().splitlines()]  # noqa: E741
    tools = [e for e in lines if e.get("event") == "tool"]
    assert tools[0]["result"] == "command denied (not allowlisted): rm -rf /"
    assert tools[1]["result"].startswith("exit code")
    assert run.grade == 0.0  # nothing was implemented


def test_an_engine_failure_is_a_failed_run_with_a_zero_grade(tmp_path):
    task = _task()
    counting = CountingEngine(FakeEngine([]))  # the script is exhausted at once
    run = run_one(
        task, 0, _arm(tmp_path), ManagedEngine(counting), counting, tmp_path / "s", python=PYTHON
    )
    assert run.state == "failed" and not run.completed
    assert run.error and "engine error" in run.error
    assert run.grade == 0.0 and run.turns == 0


def test_run_suite_loads_once_runs_what_is_not_done_records_in_order_and_releases(tmp_path):
    task = _task()
    scripts = [_solution_write(task), FINISH] * 3
    inner = FakeEngine(scripts)
    loads = []

    def factory(model_id):
        loads.append(model_id)
        return inner

    recorded = []
    outcome = run_suite(
        _arm(tmp_path),
        [task],
        runs=3,
        done={("implement_rpn", 1)},
        record=recorded.append,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert loads == ["org/m"]
    assert [(r.task, r.index) for r in outcome.runs] == [("implement_rpn", 0), ("implement_rpn", 2)]
    assert recorded == outcome.runs
    assert all(r.grade == 1.0 for r in outcome.runs)
    assert outcome.released and outcome.error is None
    assert inner.unloaded
    assert (tmp_path / "scratch" / "m" / "implement_rpn-3" / "project" / "rpn.py").is_file()


def test_every_arm_and_run_sees_one_project_root_and_run_i_samples_seed_i(tmp_path, monkeypatch):
    """An arm comparison must compare arms, not prompts: the system prompt
    names one project path for every arm and run index, and run i of every
    arm samples from seed i, so runs differ on purpose and pair across arms.
    Each run's project still ends up archived beside its transcript."""
    task = _task()
    seeds: list[int | None] = []
    real = ManagedEngine.session

    def session(self, seed=None):
        seeds.append(seed)
        return real(self, seed=seed)

    monkeypatch.setattr(ManagedEngine, "session", session)
    work = tmp_path / "run" / "project"
    # What an interrupted run left behind must not leak into the next one.
    work.mkdir(parents=True)
    (work / "stale.txt").write_text("from a run that died")
    prompts: list[str] = []
    for label in ("a", "b"):
        inner = FakeEngine([_solution_write(task), FINISH] * 2)
        outcome = run_suite(
            dataclasses.replace(_arm(tmp_path), label=label),
            [task],
            runs=2,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / "run" / "suite",
            out=lambda *a: None,
            factory=lambda mid, inner=inner: inner,
            python=PYTHON,
            active_memory=lambda: 0,
        )
        # Graded where the run left its project, not after it moved away.
        assert [(r.state, r.grade) for r in outcome.runs] == [("done", 1.0), ("done", 1.0)]
        prompts += [call[0]["content"] for call in inner.calls]
    assert len(prompts) == 8 and len(set(prompts)) == 1
    assert f"Project root: {work}" in prompts[0]
    assert seeds == [0, 1, 0, 1]
    assert not work.exists()
    for label in ("a", "b"):
        for n in (1, 2):
            archived = tmp_path / "run" / "suite" / label / f"implement_rpn-{n}" / "project"
            assert (archived / "rpn.py").is_file() and not (archived / "stale.txt").exists()


def test_a_run_that_raises_midway_still_archives_its_project(tmp_path, monkeypatch):
    from sous.tune.suite import runner

    def broken(**kwargs):
        (kwargs["root"] / "half.txt").write_text("written before the failure")
        raise OSError("disk full")

    monkeypatch.setattr(runner, "run_loop", broken)
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert [(r.state, r.error) for r in outcome.runs] == [("error", "OSError: disk full")]
    archived = tmp_path / "scratch" / "m" / "implement_rpn-1" / "project"
    assert (archived / "half.txt").is_file() and (archived / "rpn.py").is_file()
    assert not (tmp_path / "project").exists()


def test_the_next_run_waits_for_an_abandoned_generation(tmp_path, monkeypatch):
    """run_loop can give up on a stalled generation (budget-exhausted) at
    its own wall-clock deadline while the engine's session thread keeps
    decoding under ManagedEngine._gen_lock until it hits its token cap. The
    next run's own budget clock must not start against a locked engine."""
    from sous.tune.suite import runner

    task = _task()
    inner = FakeEngine([FINISH, FINISH])
    real_run_one = runner.run_one

    def run_one_then_hold(task, index, arm, engine, counting, scratch, **kwargs):
        result = real_run_one(task, index, arm, engine, counting, scratch, **kwargs)
        if index == 0:
            # run_one has already returned (as run_loop does on budget
            # exhaustion), but the engine's session thread is simulated as
            # still decoding underneath, holding the lock a while longer.
            engine._gen_lock.acquire()
            threading.Timer(0.1, engine._gen_lock.release).start()
        return result

    monkeypatch.setattr(runner, "run_one", run_one_then_hold)
    monkeypatch.setattr(runner, "_DRAIN_POLL_SECONDS", 0.02)
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [task],
        runs=2,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lines.append,
        factory=lambda mid: inner,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert [r.state for r in outcome.runs] == ["done", "done"]
    assert outcome.released and outcome.error is None
    waiting = [line for line in lines if "waiting for an abandoned generation" in line]
    assert waiting == ["  m: waiting for an abandoned generation to end before the next run"]


def test_run_suite_runs_on_a_thread_of_its_own(tmp_path):
    task = _task()
    names = []
    run_suite(
        _arm(tmp_path),
        [task],
        runs=1,
        done=set(),
        record=lambda r: names.append(threading.current_thread().name),
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    # Each run is recorded from the thread that owns the arm's engine — the
    # one that releases its mlx state on the way out — never the caller's.
    assert names == ["sous-tune-suite-m"]


def test_a_load_failure_is_an_outcome_with_no_runs(tmp_path):
    def factory(model_id):
        raise RuntimeError("no weights")

    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == [] and outcome.released
    assert outcome.error == "load failed: RuntimeError: no weights"


def test_a_load_that_failed_with_weights_left_resident_is_not_released(tmp_path, monkeypatch):
    # A drafter that would not fit beside a target already mapped: the
    # factory raises with the target's weights still on the GPU, and the
    # next arm must not load beside them.
    from sous.tune import bench

    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 0.0)
    readings = [0]  # the baseline; every later reading is the resident 3 GiB

    def factory(model_id):
        raise RuntimeError("drafter would not fit")

    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lines.append,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: readings.pop(0) if readings else 3 * 2**30,
    )
    assert outcome.runs == [] and not outcome.released
    assert outcome.error == (
        "load failed: RuntimeError: drafter would not fit; "
        "memory not released: 3.0 GiB still resident"
    )
    assert "  m: 3.0 GiB still resident after the failed load" in lines


def test_a_failed_load_is_judged_only_after_its_exception_is_dropped(tmp_path, monkeypatch):
    # The exception's traceback holds the loader thread's frames and, with
    # them, whatever the factory had built before it raised: a reading taken
    # while the exception is bound sees those arrays as resident. The
    # judging read must come after they are gone.
    from sous.tune import bench

    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 0.0)
    events: list[str] = []

    class HalfBuilt:
        def __del__(self):
            events.append("freed")

    def factory(model_id):
        half_built = HalfBuilt()  # noqa: F841 — alive through the traceback until the exception is
        raise RuntimeError("drafter would not fit")

    def active_memory():
        events.append("read")
        return 0

    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=active_memory,
    )
    assert outcome.released and outcome.error == "load failed: RuntimeError: drafter would not fit"
    assert events[0] == "read" and "freed" in events
    assert events.index("freed") < len(events) - 1 and events[-1] == "read"


def test_an_arm_whose_drafter_or_int8_did_not_load_runs_nothing(tmp_path):
    class Engine(FakeEngine):
        drafter = ""
        int8_prefill_status = {"state": "unavailable", "reason": "no tensor units", "routed": 0}

    arm = _arm(tmp_path)
    drafted = Arm(
        **{**vars(arm), "drafter_id": "z/d", "block_size": 3, "label": "m + d @3"}  # ty: ignore[invalid-argument-type]
    )
    lines = []
    outcome = run_suite(
        drafted,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "a",
        out=lines.append,
        factory=lambda mid: Engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == [] and outcome.released
    assert outcome.error == "drafter z/d requested, engine runs with none"
    # int8_under_test=True: the winner stage's own int8 arm, which exists to
    # measure int8 prefill — the engine refusing it must fail the arm.
    int8 = Arm(
        **{
            **vars(arm),
            "int8_prefill": True,
            "int8_under_test": True,
            "label": "m + int8 prefill",
        }  # ty: ignore[invalid-argument-type]
    )
    outcome = run_suite(
        int8,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "b",
        out=lines.append,
        factory=lambda mid: Engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == []
    assert outcome.error == "int8 prefill requested, engine reports unavailable: no tensor units"


def test_an_arm_that_merely_inherits_int8_runs_when_the_engine_cannot_route_it(tmp_path):
    """int8_prefill=True but int8_under_test=False: the arm only inherited
    the setting from the user's config (quick_arms mirrors it onto every
    arm), so an engine that cannot route int8 here must not refuse the
    arm — the daemon would run it on the stock path with a warning, and the
    suite must do the same rather than treat an inherited setting as a hard
    requirement."""

    class Engine(FakeEngine):
        drafter = ""
        int8_prefill_status = {"state": "unavailable", "reason": "no tensor units", "routed": 0}

    arm = _arm(tmp_path)
    inherited = Arm(
        **{
            **vars(arm),
            "int8_prefill": True,
            "int8_under_test": False,
            "label": "m + int8 (inherited)",
        }  # ty: ignore[invalid-argument-type]
    )
    lines = []
    outcome = run_suite(
        inherited,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "c",
        out=lines.append,
        factory=lambda mid: Engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1
    assert outcome.error is None
    # The inherited attention tile's notice says "runs stock" too: match int8's own line.
    assert any("int8 prefill unavailable here (no tensor units)" in line for line in lines)


def test_memory_is_judged_against_the_baseline_taken_before_the_load(tmp_path, monkeypatch):
    from sous.tune import bench

    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 0.0)
    readings = iter([5 * 2**30, 5 * 2**30, 9 * 2**30])
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: next(readings),
    )
    # Baseline 5 GiB before the load, 5 GiB after the unload's wait, then 9
    # GiB on the final reading: 4 GiB more than the baseline stayed resident.
    assert not outcome.released
    assert outcome.error == "memory not released: 4.0 GiB still resident"
    assert [r.state for r in outcome.runs] == ["done"]


def test_a_run_that_raises_outside_the_worker_is_recorded_as_an_error_and_the_suite_goes_on(
    tmp_path, monkeypatch
):
    from sous.tune.suite import runner

    calls = []
    real = runner.run_one

    def flaky(task, index, *args, **kwargs):
        calls.append(index)
        if index == 0:
            raise OSError("disk full")
        return real(task, index, *args, **kwargs)

    monkeypatch.setattr(runner, "run_one", flaky)
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=2,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert calls == [0, 1]
    assert [r.state for r in outcome.runs] == ["error", "done"]
    assert outcome.runs[0].error == "OSError: disk full" and outcome.runs[0].grade == 0.0


def test_a_record_callback_that_raises_still_releases_the_weights_and_keeps_the_runs(tmp_path):
    task = _task()
    inner = FakeEngine([FINISH])

    def factory(model_id):
        return inner

    def record(run):
        raise RuntimeError("disk full")

    outcome = run_suite(
        _arm(tmp_path),
        [task],
        runs=1,
        done=set(),
        record=record,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert inner.unloaded
    assert outcome.released
    assert outcome.error == "RuntimeError: disk full"
    assert len(outcome.runs) == 1


def test_suite_runs_round_trip_through_dicts(tmp_path):
    run = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=True,
        attention_tile=False,
        projection_kernel=False,
        kv_int8=False,
        projection_tile="mlp",
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=1,
        grade=0.5,
        grade_detail="1/2",
        error=None,
        transcript_path=None,
    )
    assert SuiteRun.from_dict(json.loads(json.dumps(run.as_dict()))) == run
    assert run.key == ("org/m", "", 0, False, True, False, False, False, "mlp")


def test_estimate_seconds_scales_with_the_measured_speeds(tmp_path):
    arm = _arm(tmp_path)
    row = BenchRow(
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        window=8192,
        ok=True,
        error=None,
        load_seconds=1.0,
        prefill_tps_2k=300.0,
        prefill_tps_16k=None,
        decode_tps_1k=30.0,
        decode_tps_16k=None,
        ttft_seconds=1.0,
        peak_memory_bytes=1,
        spread=None,
    )
    per_run = 8 * (3000 / 300 + 300 / 30)
    assert estimate_seconds([arm], [row], tasks=8, runs=2) == per_run * 16
    assert estimate_seconds([arm], [], tasks=8, runs=2) is None


def test_a_suite_arm_gets_no_fork_store(monkeypatch, tmp_path):
    import sous.tune.suite.runner as runner_mod

    seen = {}

    def fake_default_engine_factory(config, **kw):
        seen.update(kw)
        return lambda mid: FakeEngine([FINISH])

    monkeypatch.setattr(runner_mod, "default_engine_factory", fake_default_engine_factory)
    run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=None,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert seen == {"forks": False}


def test_a_row_written_before_the_attention_tile_existed_reads_as_stock(tmp_path):
    arm = _arm(tmp_path)
    row = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=False,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        projection_tile=True,
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="1/1",
        error=None,
        transcript_path=None,
    ).as_dict()
    del row["attention_tile"]
    old = SuiteRun.from_dict(row)
    assert old.attention_tile is False
    # The arm inherits the tile's default, so a resume runs the task again
    # rather than counting the stock row toward the tile arm.
    assert arm.attention_tile is True
    assert old.key == (*arm.key, False, False, False, True, True, True)
    assert old.key != arm.suite_key


class _TileUnavailable(FakeEngine):
    drafter = ""
    attention_tile_status = {
        "state": "unavailable",
        "reason": "macOS 15.5 < 26.2",
        "splits": None,
        "probe_seconds": None,
    }


def test_an_arm_that_inherits_the_tile_runs_stock_where_the_engine_cannot_serve_it(tmp_path):
    """Every arm inherits attention_tile from the user's config, and none is
    built to measure it: an engine that reports the tile unavailable must
    not refuse the arm, since the daemon would serve that checkpoint with
    stock attention."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _TileUnavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert outcome.runs[0].attention_tile is True
    assert (
        "  m: attention tile unavailable here (macOS 15.5 < 26.2); "
        "the engine runs stock, as the daemon would"
    ) in lines


def test_no_tile_notice_when_the_tile_is_active_or_the_arm_runs_stock(tmp_path):
    class Active(FakeEngine):
        drafter = ""
        attention_tile_status = {
            "state": "active",
            "reason": None,
            "splits": 20,
            "probe_seconds": 0.4,
        }

    arm = _arm(tmp_path)
    stock = dataclasses.replace(
        arm, attention_tile=False, config=dataclasses.replace(arm.config, attention_tile=False)
    )
    for case, engine in ((arm, Active), (stock, _TileUnavailable)):
        lines = []
        outcome = run_suite(
            case,
            [_task()],
            runs=1,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / f"s-{case.attention_tile}",
            out=lines.append,
            factory=lambda mid, engine=engine: engine([FINISH]),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert len(outcome.runs) == 1 and outcome.error is None
        assert outcome.runs[0].attention_tile is case.attention_tile
        assert not any("attention tile" in line for line in lines), lines


def test_the_tile_notice_is_only_for_an_engine_that_reports_it_unavailable(tmp_path):
    """`off` is an engine with no tile at all (the mlx-lm backend), and an
    engine that reports nothing has no tile either: neither is a machine
    where the tile cannot run, so neither gets the notice."""

    class Off(FakeEngine):
        drafter = ""
        attention_tile_status = {"state": "off", "reason": None}

    class Silent(FakeEngine):
        drafter = ""

    class Unexplained(FakeEngine):
        drafter = ""
        attention_tile_status = {"state": "unavailable", "reason": None}

    notice = "  m: attention tile unavailable here (no reason given); "
    notice += "the engine runs stock, as the daemon would"
    for name, engine, expected in (
        ("off", Off, []),
        ("silent", Silent, []),
        ("unexplained", Unexplained, [notice]),
    ):
        lines = []
        outcome = run_suite(
            _arm(tmp_path),
            [_task()],
            runs=1,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / f"s-{name}",
            out=lines.append,
            factory=lambda mid, engine=engine: engine([FINISH]),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert len(outcome.runs) == 1 and outcome.error is None, name
        assert [line for line in lines if "attention tile" in line] == expected, name


def test_a_row_written_before_the_projection_kernel_existed_reads_as_stock(tmp_path):
    arm = _arm(tmp_path)
    row = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=False,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        projection_tile=True,
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="1/1",
        error=None,
        transcript_path=None,
    ).as_dict()
    del row["projection_kernel"]
    old = SuiteRun.from_dict(row)
    # No run from before the setting existed went through the kernel, on any
    # machine: the row is a stock-projection row.
    assert old.projection_kernel is False and old.attention_tile is True
    # The arm inherits the kernel's default, so a resume runs the task again
    # rather than counting the stock row toward the kernel arm.
    assert arm.projection_kernel is True
    assert old.key == (*arm.key, False, False, True, False, True, True)
    assert old.key != arm.suite_key
    del row["attention_tile"]
    older = SuiteRun.from_dict(row)
    assert (older.attention_tile, older.projection_kernel) == (False, False)


class _ProjUnavailable(FakeEngine):
    drafter = ""
    projection_kernel_status = {
        "state": "unavailable",
        "reason": "attention tile off",
        "probe_seconds": None,
    }


def test_an_arm_that_inherits_the_kernel_runs_stock_where_the_engine_cannot_serve_it(tmp_path):
    """Like the tile, the projection kernel is inherited from the user's
    config and no arm measures it: an engine that reports it unavailable
    runs the arm stock after a notice, as the daemon would."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _ProjUnavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert outcome.runs[0].projection_kernel is True
    assert (
        "  m: projection kernel unavailable here (attention tile off); "
        "the engine runs stock, as the daemon would"
    ) in lines


class _DrafterRouteUnavailable(FakeEngine):
    drafter = ""
    projection_kernel_status = {"state": "active", "reason": None, "probe_seconds": 0.3}
    projection_kernel_drafter_status = {
        "state": "unavailable",
        "reason": "probe: program_source:3:1: error: boom",
        "linears": 47,
        "probe_seconds": 0.4,
    }


def test_an_arm_whose_drafter_route_is_unavailable_gets_a_notice_of_its_own(tmp_path):
    """The drafter's own linears ride the kernel under a status of their own: an
    arm whose kernel serves the target but not the drafter measures stock
    drafter projections under a kernel-on key, and says so."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _DrafterRouteUnavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert [line for line in lines if "projection kernel" in line] == [
        "  m: projection kernel drafter unavailable here "
        "(probe: program_source:3:1: error: boom); "
        "the engine runs the drafter's own projections stock, as the daemon would"
    ]


def test_no_kernel_notice_when_it_is_active_off_unreported_or_not_inherited(tmp_path):
    class Active(FakeEngine):
        drafter = ""
        projection_kernel_status = {"state": "active", "reason": None, "probe_seconds": 0.3}
        projection_kernel_drafter_status = {
            "state": "active",
            "reason": None,
            "linears": 47,
            "probe_seconds": 0.2,
        }

    class Off(FakeEngine):
        drafter = ""
        projection_kernel_status = {"state": "off", "reason": None}
        projection_kernel_drafter_status = {
            "state": "off",
            "reason": "projection kernel off",
            "linears": 0,
            "probe_seconds": None,
        }

    class Silent(FakeEngine):
        drafter = ""

    arm = _arm(tmp_path)
    stock = dataclasses.replace(
        arm,
        projection_kernel=False,
        config=dataclasses.replace(arm.config, projection_kernel=False),
    )
    for name, case, engine in (
        ("active", arm, Active),
        ("off", arm, Off),
        ("silent", arm, Silent),
        ("stock", stock, _ProjUnavailable),
        ("drafter stock", stock, _DrafterRouteUnavailable),
    ):
        lines = []
        outcome = run_suite(
            case,
            [_task()],
            runs=1,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / f"s-{name}",
            out=lines.append,
            factory=lambda mid, engine=engine: engine([FINISH]),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert len(outcome.runs) == 1 and outcome.error is None, name
        assert outcome.runs[0].projection_kernel is case.projection_kernel, name
        assert not any("projection kernel" in line for line in lines), (name, lines)


def test_a_row_written_before_kv_int8_existed_reads_as_bf16(tmp_path):
    arm = _arm(tmp_path)
    row = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=False,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        projection_tile=True,
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="1/1",
        error=None,
        transcript_path=None,
    ).as_dict()
    del row["kv_int8"]
    old = SuiteRun.from_dict(row)
    # No run from before the setting existed read int8 K/V, on any machine:
    # the row is a bf16 row, and the settings it did record stay as written.
    assert old.kv_int8 is False
    assert (old.attention_tile, old.projection_kernel) == (True, True)
    # The arm inherits the setting's default, so a resume runs the task again
    # rather than counting the bf16 row toward the int8 arm.
    assert arm.kv_int8 is True
    assert old.key == (*arm.key, False, False, True, True, False, True)
    assert old.key != arm.suite_key


class _Kv8Unavailable(FakeEngine):
    drafter = ""
    kv_int8_status = {
        "state": "unavailable",
        "reason": "attention tile off",
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }


def test_an_arm_that_inherits_kv_int8_runs_bf16_where_the_engine_cannot_serve_it(tmp_path):
    """kv_int8 is inherited from the user's config and no arm measures it:
    an engine that reports it unavailable runs the arm on bf16 K/V after a
    notice, as the daemon would."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _Kv8Unavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert outcome.runs[0].kv_int8 is True
    assert (
        "  m: int8 K/V unavailable here (attention tile off); "
        "the engine runs bf16 K/V, as the daemon would"
    ) in lines


def test_no_kv8_notice_when_it_is_active_off_unreported_or_not_inherited(tmp_path):
    """`off` is the setting switched off or an engine without kv8 (the
    mlx-lm backend), and an engine that reports nothing has none either:
    neither is a machine where kv8 cannot run, so neither gets the notice."""

    class Active(FakeEngine):
        drafter = ""
        kv_int8_status = {
            "state": "active",
            "reason": None,
            "probe_seconds": 0.5,
            "floor": 1536,
            "ceiling": 84481,
        }

    class Off(FakeEngine):
        drafter = ""
        kv_int8_status = {
            "state": "off",
            "reason": None,
            "probe_seconds": None,
            "floor": None,
            "ceiling": None,
        }

    class Silent(FakeEngine):
        drafter = ""

    arm = _arm(tmp_path)
    bf16 = dataclasses.replace(
        arm, kv_int8=False, config=dataclasses.replace(arm.config, kv_int8=False)
    )
    for name, case, engine in (
        ("active", arm, Active),
        ("off", arm, Off),
        ("silent", arm, Silent),
        ("bf16", bf16, _Kv8Unavailable),
    ):
        lines = []
        outcome = run_suite(
            case,
            [_task()],
            runs=1,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / f"s-{name}",
            out=lines.append,
            factory=lambda mid, engine=engine: engine([FINISH]),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert len(outcome.runs) == 1 and outcome.error is None, name
        assert outcome.runs[0].kv_int8 is case.kv_int8, name
        assert not any("int8 K/V" in line for line in lines), (name, lines)


def test_a_row_written_before_the_projection_tile_existed_reads_as_off(tmp_path):
    arm = _arm(tmp_path)
    row = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=False,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        projection_tile="all",
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="1/1",
        error=None,
        transcript_path=None,
    ).as_dict()
    del row["projection_tile"]
    old = SuiteRun.from_dict(row)
    # No run from before the setting existed went through the tile, on any
    # machine; the settings the row did record stay as written.
    assert old.projection_tile is False
    assert (old.attention_tile, old.projection_kernel, old.kv_int8) == (True, True, True)
    # The arm inherits the setting's default, so a resume runs the task again
    # rather than counting the row toward the tile arm.
    assert arm.projection_tile is True
    assert old.key == (*arm.key, False, False, True, True, True, False)
    assert old.key != arm.suite_key


def _ptile_engine(reason, state="unavailable"):
    class Engine(FakeEngine):
        drafter = ""
        projection_tile_status = {
            "state": state,
            "scope": None,
            "reason": reason,
            "copy_gb": None,
            "probe_seconds": None,
        }

    return Engine


def _suite_lines(tmp_path, name, arm, engine):
    lines = []
    outcome = run_suite(
        arm,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / f"s-{name}",
        out=lines.append,
        factory=lambda mid: engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None, name
    # The row records what the load reported, never the setting.
    want = loaded_tile(getattr(engine, "projection_tile_status", None), arm.projection_tile)
    assert outcome.runs[0].projection_tile == want, name
    return [line for line in lines if "projection tile" in line]


def test_loaded_tile_reads_the_scope_while_active_and_the_state_otherwise():
    def status(state, scope=None):
        return {"state": state, "scope": scope, "reason": None}

    for setting in (True, "mlp", "all"):
        assert loaded_tile(status("active", "all"), setting) == "all"
        assert loaded_tile(status("active", "mlp"), setting) == "mlp"
        assert loaded_tile(status("unavailable"), setting) == "unavailable"
        # A VLM load reports off only under false: here it is an engine without
        # the tile (the mlx-lm backend's), or one with no status at all.
        assert loaded_tile(status("off"), setting) == "absent"
        assert loaded_tile(None, setting) == "absent"
    assert loaded_tile(status("off"), False) == "off"
    assert loaded_tile(None, False) == "off"


# setting -> the recorded states a resume under it counts, each mapped to the group
# it joins; every other state reruns the arm. Rows written before rows recorded the
# load's state hold the setting: false joins "off", true keeps a group of its own.
_TILE_CELLS = {
    False: {"off": "off", False: "off"},
    True: {
        "all": "all",
        "mlp": "mlp",
        "unavailable": "unavailable",
        "absent": "absent",
        True: True,
    },
    "mlp": {"mlp": "mlp", "unavailable": "unavailable", "absent": "absent"},
    "all": {"all": "all", "mlp": "mlp", "unavailable": "unavailable", "absent": "absent"},
}
_RECORDED = ("all", "mlp", "unavailable", "off", "absent", True, False)
_NOT_COUNTED = object()


@pytest.mark.parametrize("recorded", _RECORDED, ids=repr)
@pytest.mark.parametrize("setting", list(_TILE_CELLS), ids=repr)
def test_a_resume_counts_a_row_only_under_a_setting_that_can_load_its_state(
    tmp_path, setting, recorded
):
    """ "off" is a tile switched off, which no tile-capable setting counts; "mlp"
    counts under "all" too, which reports it when no stack had a plan; "absent",
    an engine without the tile, counts under every setting but false."""
    arm = dataclasses.replace(_arm(tmp_path), projection_tile=setting)
    row = dataclasses.replace(_finished_row(arm), projection_tile=recorded)
    groups, state = arm_runs(arm, [row])
    want = _TILE_CELLS[setting].get(recorded, _NOT_COUNTED)
    if want is _NOT_COUNTED:
        assert (groups, state) == ({}, None)
    else:
        assert state == want and type(state) is type(want) and groups == {want: [row]}


def _finished_row(arm):
    return SuiteRun(
        task="t",
        index=0,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        attention_tile=arm.attention_tile,
        projection_kernel=arm.projection_kernel,
        kv_int8=arm.kv_int8,
        projection_tile=arm.projection_tile,
        window=arm.window,
        state="done",
        outcome="completed",
        turns=1,
        seconds=1.0,
        output_tokens=1,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="",
        error=None,
        transcript_path=None,
    )


def test_rows_record_the_scope_the_load_reported_and_resume_on_its_done_set(tmp_path):
    """Under true the load picks the scope: rows record it, and the done set the
    runner keeps is the one of the scope this load reported, so runs finished
    under another scope go again rather than pool with these."""
    arm = _arm(tmp_path)
    assert arm.projection_tile is True
    task = _task()

    class Mlp(FakeEngine):
        drafter = ""
        projection_tile_status = {
            "state": "active",
            "scope": "mlp",
            "reason": "copy would leave 13.00 GiB of prompt-cache budget (< 14)",
            "copy_gb": None,
            "probe_seconds": 0.5,
            "budget_left_gib": 17.0,
        }

    def run(done, done_by_tile, name):
        recorded = []
        outcome = run_suite(
            arm,
            [task],
            runs=2,
            done=done,
            done_by_tile=done_by_tile,
            record=recorded.append,
            scratch=tmp_path / name,
            out=lambda *a: None,
            factory=lambda mid: Mlp([FINISH] * 2),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert outcome.error is None and outcome.runs == recorded
        return [(r.index, r.projection_tile) for r in recorded]

    first = {(task.name, 0), (task.name, 1)}
    # Both runs finished under scope all: this load reports mlp, and runs both.
    assert run(first, {"all": first}, "a") == [(0, "mlp"), (1, "mlp")]
    # One finished under mlp already: only the other runs.
    assert run(first, {"all": first, "mlp": {(task.name, 0)}}, "b") == [(1, "mlp")]
    # Without the groups the done set handed in is the one kept.
    assert run({(task.name, 0)}, None, "c") == [(1, "mlp")]


def test_an_arm_that_inherits_the_projection_tile_runs_without_it_after_a_notice(tmp_path):
    """The tile is inherited and unmeasured like the kernel: an engine that
    reports it unavailable runs the arm without it, as the daemon would."""
    for reason in ("int8 prefill off", "GPU core count 10 not measured (measured: 20)"):
        lines = _suite_lines(tmp_path, reason[:4], _arm(tmp_path), _ptile_engine(reason))
        assert lines == [
            f"  m: projection tile unavailable here ({reason}); "
            "the engine runs without it, as the daemon would"
        ], reason


def test_no_projection_tile_notice_for_a_drafterless_arm_or_a_tile_not_in_play(tmp_path):
    """Every drafterless arm on a machine the tile serves gets `no drafter`, the
    refusal the tile is built to make: a notice there would repeat on every
    such arm and say nothing. `active`, `off`, no status or a flag the arm does
    not carry are no machine the tile cannot run on either."""

    class Silent(FakeEngine):
        drafter = ""

    arm = _arm(tmp_path)
    off = dataclasses.replace(
        arm, projection_tile=False, config=dataclasses.replace(arm.config, projection_tile=False)
    )
    for name, case, engine in (
        ("no-drafter", arm, _ptile_engine("no drafter")),
        ("active", arm, _ptile_engine(None, state="active")),
        ("off", arm, _ptile_engine(None, state="off")),
        ("silent", arm, Silent),
        ("not-inherited", off, _ptile_engine("int8 prefill off")),
    ):
        assert _suite_lines(tmp_path, name, case, engine) == [], name
