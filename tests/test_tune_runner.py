"""The suite runner drives sous's own worker loop: the sandbox and the
allowlist are real, approvals are denied, and every number in a run is read
back from the store, the transcript and the engine's deltas."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import cast

from sous.config import SousConfig
from sous.engine.base import ManagedEngine, ReplaySafe
from sous.tasks import TaskStore
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import (
    SUITE_ALLOWLIST,
    CountingEngine,
    SuiteRun,
    estimate_seconds,
    interpreter_first,
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
        approval_timeout_minutes=1,
        **over,
    )
    return Arm(
        label="m",
        config=cfg,
        model_id="org/m",
        drafter_id="",
        block_size=0,
        window=8192,
        gateway_window=None,
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


def test_the_suite_allowlist_is_the_shipped_one_plus_unittest():
    from sous.config import DEFAULT_ALLOWLIST

    assert SUITE_ALLOWLIST[: len(DEFAULT_ALLOWLIST)] == tuple(DEFAULT_ALLOWLIST)
    assert SUITE_ALLOWLIST[len(DEFAULT_ALLOWLIST) :] == (
        "python -m unittest",
        "python3 -m unittest",
    )


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
    assert run.transcript_path and Path(run.transcript_path).is_file()
    written = (tmp_path / "s" / "project" / "rpn.py").read_text()
    assert written == (task.solution / "rpn.py").read_text()
    assert run.key == (*_arm(tmp_path).key, False, False)


def test_a_command_outside_the_allowlist_is_denied_and_counted(tmp_path):
    task = _task()
    denied = CALL.format(name="run_command", args='{"command": "echo hi"}')
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
    assert "command denied" in tools[0]["result"]
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


def test_the_denier_survives_a_store_error_and_records_it():
    from sous.tune.suite import runner

    class BustedStore:
        def get(self, task_id):
            raise RuntimeError("busy")

    with runner._Denier(cast(TaskStore, BustedStore()), "t", 0.01) as d:
        time.sleep(0.05)
        assert d._thread.is_alive()
    assert d.error == "RuntimeError: busy"
    assert d.denied == 0


def test_a_denier_failure_marks_the_run_as_an_error(tmp_path, monkeypatch):
    from sous.tune.suite import runner

    class FailingDenier(runner._Denier):
        def __enter__(self):
            super().__enter__()
            self.error = "RuntimeError: busy"
            return self

    monkeypatch.setattr(runner, "_Denier", FailingDenier)
    task = _task()
    counting = CountingEngine(FakeEngine([FINISH]))
    run = run_one(
        task, 0, _arm(tmp_path), ManagedEngine(counting), counting, tmp_path / "s", python=PYTHON
    )
    assert run.state == "error"
    assert run.error == "approval denier failed: RuntimeError: busy"


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


def test_the_next_run_waits_for_an_abandoned_generation(tmp_path, monkeypatch):
    """run_task can give up on a stalled generation (budget-exhausted) at
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
            # run_one has already returned (as run_task does on budget
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
    assert any("runs stock" in line for line in lines)


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
    assert run.key == ("org/m", "", 0, False, True)


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
