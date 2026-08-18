import dataclasses
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.tasks import TaskState, TaskStore
from sous.worker import run_task, run_worker_loop
from tests.fake_engine import FakeEngine


@pytest.fixture()
def env(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "hello.py").write_text("print('hi')\n")
    cfg = SousConfig(
        data_dir=tmp_path / "sous-data",
        config_path=tmp_path / "config.toml",
        max_turns=10,
        max_minutes=1,
        approval_timeout_minutes=1,
    )
    store = TaskStore(tmp_path / "tasks.db")
    return root, cfg, store


def _start(store: TaskStore, root: Path, verify=(), context=()):
    store.enqueue(
        title="t",
        instructions="do the thing",
        project_root=str(root),
        context_files=list(context),
        verify_commands=list(verify),
    )
    return store.claim_next()


CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "did it", "concerns": ""}')


def test_happy_path_write_then_finish(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="write_file", args='{"path": "out.txt", "content": "hello"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE and got.outcome == "completed"
    assert (root / "out.txt").read_text() == "hello"
    assert got.report["summary"] == "did it"
    assert got.report["files_changed"][0]["path"] == "out.txt"
    assert Path(got.report["transcript_path"]).exists()


def test_transcript_is_jsonl(env):
    root, cfg, store = env
    task = _start(store, root)
    run_task(task, store, FakeEngine([FINISH]), cfg)
    lines = Path(store.get(task.id).report["transcript_path"]).read_text().splitlines()
    assert all(json.loads(line) for line in lines)
    assert len(lines) >= 2  # at least one generation + terminal event


def test_context_files_preloaded(env):
    root, cfg, store = env
    task = _start(store, root, context=["hello.py"])
    engine = FakeEngine([FINISH])
    run_task(task, store, engine, cfg)
    first_messages = engine.calls[0]
    assert any("print('hi')" in str(m.get("content", "")) for m in first_messages)


def test_budget_exhaustion_by_turns(env):
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_turns=2)
    task = _start(store, root)
    engine = FakeEngine([CALL.format(name="list_dir", args="{}")] * 2)
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE and got.outcome == "budget-exhausted"


def test_three_consecutive_malformed_fails(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(["<tool_call>{bad json}</tool_call>"] * 3)
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "model-confused" in got.report["error"]


def test_file_written_then_model_confused_still_reports_files_changed(env):
    """A task that writes a file and then goes model-confused must still
    tell Claude what changed and where to audit — never a silent,
    unreviewed file modification hidden behind a bare {"error": ...}."""
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hello sous"}'),
            "let me think about this some more with no tool call",
            "still thinking, no tool call here either",
            "and a third turn with no tool call at all",
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "model-confused" in got.report["error"]
    assert (root / "hello.txt").read_text() == "hello sous"
    assert got.report["files_changed"][0]["path"] == "hello.txt"
    assert Path(got.report["transcript_path"]).exists()


def test_malformed_then_recovery_resets_counter(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            "<tool_call>{bad}</tool_call>",
            CALL.format(name="list_dir", args="{}"),
            "<tool_call>{bad}</tool_call>",
            "<tool_call>{bad}</tool_call>",
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE  # never hit 3 in a row


def test_prose_only_gets_nudge(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(["let me think about this...", FINISH])
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    nudge_turn = engine.calls[1]
    assert any("must call a tool" in str(m.get("content", "")) for m in nudge_turn)


def test_cancel_mid_loop(env):
    root, cfg, store = env
    task = _start(store, root)
    store.cancel(task.id)  # flag set while "running"
    run_task(task, store, FakeEngine([FINISH]), cfg)
    assert store.get(task.id).state == TaskState.CANCELLED


def test_cancel_after_write_still_reports_files_changed(env):
    """B1: a task cancelled after editing files must report files_changed and
    the transcript path exactly like the failed path does — cancel is the one
    terminal state that was still hiding what the worker touched."""
    root, cfg, store = env
    task = _start(store, root)

    class CancelAfterGen(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            out = super().generate(messages, tools, max_tokens)
            if not self.script:
                # Flag lands after the final generation, i.e. after turn 1's
                # write already ran. (Cancelling right after turn 1's generate
                # would now — correctly — stop the task BEFORE the write:
                # cancellation is checked at every tool boundary.)
                store.cancel(task.id)
            return out

    engine = CancelAfterGen(
        [
            CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hello sous"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.CANCELLED
    assert (root / "hello.txt").read_text() == "hello sous"
    assert got.report["files_changed"][0]["path"] == "hello.txt"
    assert Path(got.report["transcript_path"]).exists()


def test_cancel_between_calls_in_same_turn_stops_before_finish(env):
    """B1: cancellation is honored at EVERY tool boundary, not once per model
    turn — a single response carrying [write_file, finish] with cancel landing
    while write_file executes must end cancelled, never done, and the report
    must still name the written file."""
    root, cfg, store = env
    task = _start(store, root)

    # Cancel lands right after the first tool executes (set_activity runs
    # after each _execute), i.e. between write_file and finish.
    orig_set_activity = store.set_activity

    def cancel_on_activity(task_id, text, turns_used):
        orig_set_activity(task_id, text, turns_used)
        store.cancel(task_id)

    store.set_activity = cancel_on_activity
    engine = FakeEngine(
        [
            CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hello sous"}')
            + FINISH,  # both calls in ONE response
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.CANCELLED
    assert got.outcome != "completed"
    assert (root / "hello.txt").read_text() == "hello sous"
    assert got.report["files_changed"][0]["path"] == "hello.txt"
    assert Path(got.report["transcript_path"]).exists()


def test_restart_recovery_reports_files_changed_and_transcript(env):
    """B2: files the worker already wrote must be visible after a daemon
    restart — the worker persists changed_files as it goes, and
    recover_interrupted folds them (plus the deterministic transcript path)
    into the failure report."""
    root, cfg, store = env
    task = _start(store, root)

    class DyingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            if not self.script:
                raise KeyboardInterrupt  # simulates the daemon dying mid-task
            return super().generate(messages, tools, max_tokens)

    engine = DyingEngine(
        [
            CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hello sous"}'),
        ]
    )
    with pytest.raises(KeyboardInterrupt):
        run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.RUNNING  # left mid-flight
    assert store.recover_interrupted(cfg.data_dir) == 1
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "restart" in got.report["error"]
    assert got.report["files_changed"][0]["path"] == "hello.txt"
    assert got.report["transcript_path"] == str(
        cfg.data_dir / "tasks" / task.id / "transcript.jsonl"
    )


def test_finish_without_summary_is_retriable_not_completion(env):
    """C1: summary is declared required in WORKER_TOOLS — a finish missing it
    (or carrying only whitespace) must come back as a recoverable tool error
    the model can retry, never a false 'completed' with an empty report."""
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="finish", args='{"concerns": "none"}'),  # no summary
            CALL.format(name="finish", args='{"summary": "   "}'),  # whitespace
            FINISH,  # proper retry
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE and got.outcome == "completed"
    assert got.report["summary"] == "did it"  # from the proper finish only
    assert len(engine.calls) == 3  # both bad finishes were retried, not fatal
    for turn in (engine.calls[1], engine.calls[2]):
        assert any(
            "error" in str(m.get("content", "")).lower() and "summary" in str(m.get("content", ""))
            for m in turn
        )


def test_verify_commands_run_and_reported(env):
    root, cfg, store = env
    cfg.config_path.write_text('[commands]\nallowlist = ["/bin/echo"]\n')
    task = _start(store, root, verify=["/bin/echo verified-ok"])
    run_task(task, store, FakeEngine([FINISH]), cfg)
    [v] = store.get(task.id).report["verify"]
    assert v["command"] == "/bin/echo verified-ok"
    assert "verified-ok" in v["output"]


def test_path_violation_reported_not_fatal(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="read_file", args='{"path": "../../etc/passwd"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    result_turn = engine.calls[1]
    assert any("escapes project root" in str(m.get("content", "")) for m in result_turn)


def test_approval_flow_approved(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="run_command", args='{"command": "/bin/echo custom"}'),
            FINISH,
        ]
    )

    def approver():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            t = store.get(task.id)
            if t.state == TaskState.AWAITING_APPROVAL:
                assert t.pending_command == "/bin/echo custom"
                store.respond_approval(task.id, approve=True)
                return
            time.sleep(0.05)

    thread = threading.Thread(target=approver)
    thread.start()
    run_task(task, store, engine, cfg)
    thread.join()
    got = store.get(task.id)
    assert got.state == TaskState.DONE
    result_turn = engine.calls[1]
    assert any("custom" in str(m.get("content", "")) for m in result_turn)


def test_approval_flow_denied(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="run_command", args='{"command": "/bin/echo custom"}'),
            FINISH,
        ]
    )

    def denier():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if store.get(task.id).state == TaskState.AWAITING_APPROVAL:
                store.respond_approval(task.id, approve=False)
                return
            time.sleep(0.05)

    thread = threading.Thread(target=denier)
    thread.start()
    run_task(task, store, engine, cfg)
    thread.join()
    result_turn = engine.calls[1]
    assert any("denied" in str(m.get("content", "")) for m in result_turn)


def test_generation_timeout_at_wall_budget_is_budget_exhausted(env):
    """C1: the generation timeout IS the remaining wall-clock budget, so a
    timeout with the deadline passed means the budget ran out — per spec that
    ends the task as done/budget-exhausted with a partial report, never
    failed. (Previously test_generation_stall_fails_task, which asserted
    FAILED/'stalled' for exactly this case.)"""
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_minutes=0.005)  # 0.3 s wall budget

    class StallingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            time.sleep(2)
            return FINISH

    task = _start(store, root)
    run_task(task, store, StallingEngine([FINISH]), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE
    assert got.outcome == "budget-exhausted"
    assert "files_changed" in got.report  # partial report still assembled
    assert Path(got.report["transcript_path"]).exists()


def test_genuine_stall_with_budget_remaining_fails(env, monkeypatch):
    """C1 (the other side): a stall while wall-clock budget genuinely remains
    is still a failure, not budget exhaustion."""
    import sous.worker as worker_mod

    root, cfg, store = env  # max_minutes=1: plenty of budget remains

    def stall_immediately(engine, messages, max_tokens, timeout_seconds):
        raise worker_mod.GenerationStalled("generation stalled (> 5s)")

    monkeypatch.setattr(worker_mod, "_generate_with_timeout", stall_immediately)
    task = _start(store, root)
    run_task(task, store, FakeEngine([FINISH]), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "stalled" in got.report["error"]
    assert "files_changed" in got.report


def test_context_over_cap_with_nothing_to_elide_fails_cleanly(env):
    """C2: when no elidable tool_result remains and the count is still above
    max_context_tokens, the task must fail with a clear reason naming the
    measured count and the cap — never send an oversized prompt."""
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_context_tokens=10)
    task = _start(store, root)
    engine = FakeEngine([FINISH])
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    err = got.report["error"]
    assert "context" in err.lower()
    assert "10" in err  # the cap, named
    assert engine.calls == []  # the oversized prompt was never sent
    assert "files_changed" in got.report
    assert Path(got.report["transcript_path"]).exists()


def test_context_elision_replaces_old_tool_results(env):
    root, cfg, store = env
    (root / "big.txt").write_text("word " * 4000)
    cfg = dataclasses.replace(cfg, max_context_tokens=1500)
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="read_file", args='{"path": "big.txt"}'),
            CALL.format(name="read_file", args='{"path": "hello.py"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    final_turn = engine.calls[-1]
    assert any("[elided" in str(m.get("content", "")) for m in final_turn)


def test_tool_error_from_bad_argument_value_does_not_crash_task(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="grep", args='{"pattern": "("}'),  # invalid regex
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE and got.outcome == "completed"
    result_turn = engine.calls[1]
    assert any("error" in str(m.get("content", "")).lower() for m in result_turn)


def test_verify_command_that_raises_is_reported_not_fatal(env):
    root, cfg, store = env
    script = root / "verify.sh"
    script.write_text("#!/bin/sh\necho nope\n")
    script.chmod(0o644)  # no execute bit -> subprocess.run raises PermissionError
    cfg.config_path.write_text(f'[commands]\nallowlist = ["{script}"]\n')
    task = _start(store, root, verify=[str(script)])
    run_task(task, store, FakeEngine([FINISH]), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE
    [v] = got.report["verify"]
    assert v["command"] == str(script)
    assert "error" in v["output"].lower() or "permission" in v["output"].lower()


def test_verify_commands_skipped_when_budget_exhausted(env):
    """A3: verify commands must not run past the task's wall-clock budget —
    each gets a visible 'skipped' entry in the report instead of silently
    overshooting max_minutes by a fresh command timeout apiece."""
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_minutes=0)  # deadline already passed
    cfg.config_path.write_text('[commands]\nallowlist = ["/usr/bin/touch"]\n')
    task = _start(store, root, verify=["/usr/bin/touch verify-ran.txt"])
    run_task(task, store, FakeEngine([]), cfg)  # engine never consulted
    got = store.get(task.id)
    assert got.state == TaskState.DONE and got.outcome == "budget-exhausted"
    [v] = got.report["verify"]  # the skip keeps the {command, output} shape
    assert v["command"] == "/usr/bin/touch verify-ran.txt"
    assert "skipped" in v["output"] and "budget" in v["output"]
    assert not (root / "verify-ran.txt").exists()  # it truly did not run


def test_run_command_timeout_clamped_to_remaining_budget(env):
    """A2: a run_command issued with seconds of task budget left must get a
    timeout clamped to that remainder — and never a non-positive one."""
    from sous.protocol import ToolCall
    from sous.toolexec import ToolExecutor
    from sous.worker import _execute

    root, cfg, store = env
    cfg = dataclasses.replace(cfg, command_timeout_seconds=120)

    # Deliberately partial: this test drives only the run_command branch, so
    # the double implements only that. cast, not a Protocol — it does not
    # implement the rest of the surface _execute can dispatch to.
    class RecordingEx:
        def __init__(self):
            self.timeout = None

        def run_command(self, command, approval=None, timeout=None):
            self.timeout = timeout
            return "exit code 0\nok"

    call = ToolCall(name="run_command", arguments={"command": "/bin/echo hi"})
    almost_out = RecordingEx()
    _execute(call, cast(ToolExecutor, almost_out), cfg, None, deadline=time.monotonic() + 2)
    assert almost_out.timeout is not None
    assert 0 < almost_out.timeout <= 2  # clamped well below the 120s default

    exhausted = RecordingEx()
    _execute(call, cast(ToolExecutor, exhausted), cfg, None, deadline=time.monotonic() - 5)
    assert exhausted.timeout is not None
    assert exhausted.timeout > 0  # never a non-positive timeout


def test_approval_wait_capped_by_task_deadline(env):
    """A1: an approval requested near the task deadline must be denied when
    the wall-clock budget runs out, not held for approval_timeout_minutes —
    and the deny must restore the running state like the timeout-deny does."""
    from sous.worker import _make_approval_hook

    root, cfg, store = env
    cfg = dataclasses.replace(cfg, approval_timeout_minutes=1)  # 60s on its own
    task = _start(store, root)
    hook = _make_approval_hook(task, store, cfg, time.monotonic() + 0.3)
    t0 = time.monotonic()
    assert hook("/bin/echo custom") is False
    assert time.monotonic() - t0 < 5  # denied at the ~0.3s budget, not 60s
    got = store.get(task.id)
    assert got.state == TaskState.RUNNING  # running state restored
    assert got.pending_command is None


def test_worker_loop_survives_bookkeeping_exception(env, capsys):
    """M1: a transient failure in claim_next (outside the per-task try) must
    not kill the worker thread — the loop should log and keep polling."""
    root, cfg, store = env
    stop = threading.Event()
    calls = {"n": 0}

    # Partial double for the same reason as RecordingEx: the loop never gets
    # past claim_next here, so that is all it implements.
    class FlakyStore:
        def claim_next(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            stop.set()
            return None

    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    run_worker_loop(cast(TaskStore, FlakyStore()), engines, cfg, stop, poll_interval=0.01)
    assert calls["n"] >= 2  # survived the first failure and polled again


def test_engine_exception_fails_task_cleanly(env):
    root, cfg, store = env
    task = _start(store, root)

    class ExplodingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            raise ValueError("boom")

    run_task(task, store, ExplodingEngine([FINISH]), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "engine error" in got.report["error"]
    assert "boom" in got.report["error"]


def test_run_task_honors_context_decision_over_config(env):
    """The per-task ContextDecision, not the static config value, must govern
    the window: ten tokens cannot fit even the system prompt, so the task must
    fail on context overflow despite config allowing 32768."""
    from sous.context import ContextDecision

    root, cfg, store = env
    task = _start(store, root)
    run_task(task, store, FakeEngine([FINISH]), cfg, context=ContextDecision(10, "test"))
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "context" in got.report["error"].lower()


def test_report_records_the_context_window_used(env):
    """Verifiability by construction: every report says what window the task
    actually ran with and why, so auto sizing can be audited after the fact."""
    from sous.context import ContextDecision

    root, cfg, store = env
    task = _start(store, root)
    run_task(
        task, store, FakeEngine([FINISH]), cfg, context=ContextDecision(5000, "auto: test-run")
    )
    got = store.get(task.id)
    assert got.report["budget"]["context_tokens"] == 5000
    assert got.report["budget"]["context_reason"] == "auto: test-run"


def test_generation_is_bounded_by_the_remaining_window(env):
    """The window bounds prompt PLUS output. With auto sizing the window can
    BE the model's native maximum, where an unbounded 4096-token generation
    would run past the positional limit — not just the memory estimate."""
    from sous.context import ContextDecision
    from sous.protocol import WORKER_TOOLS

    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([FINISH])
    run_task(task, store, engine, cfg, context=ContextDecision(3000, "test"))
    prompt = engine.count_tokens(engine.calls[0], WORKER_TOOLS)
    assert engine.max_tokens_seen[0] == 3000 - prompt

    task2 = _start(store, root)
    roomy = FakeEngine([FINISH])
    run_task(task2, store, roomy, cfg, context=ContextDecision(100_000, "test"))
    assert roomy.max_tokens_seen[0] == cfg.max_tokens_per_generation


def test_prompt_leaving_no_output_room_is_overflow(env):
    """A prompt that exactly fills the window passes the elision check but
    cannot generate a single token — that must be the same clean overflow
    failure, not a zero-token generation handed to the engine."""
    from sous.context import ContextDecision
    from sous.protocol import WORKER_TOOLS

    root, cfg, store = env
    probe_task = _start(store, root)
    probe = FakeEngine([FINISH])
    run_task(probe_task, store, probe, cfg)
    prompt = probe.count_tokens(probe.calls[0], WORKER_TOOLS)

    task = _start(store, root)
    starved = FakeEngine([FINISH])
    run_task(task, store, starved, cfg, context=ContextDecision(prompt, "test"))
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "no room" in got.report["error"]
    assert starved.max_tokens_seen == []  # never reached the engine


def test_generation_thread_releases_mlx_state_before_exit(env, monkeypatch):
    """mlx >= 0.32.1 requires mx.clear_streams() at the end of every thread
    that touched mlx (ml-explore/mlx#4327): without it, the exiting generation
    thread's TLS teardown segfaults the WHOLE daemon mid-task. The release must
    happen once per generation, in the generation thread itself."""
    import sous.worker as worker

    released_in = []
    monkeypatch.setattr(
        worker, "release_mlx_thread_state", lambda: released_in.append(threading.get_ident())
    )
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine(
        [
            CALL.format(name="write_file", args='{"path": "out.txt", "content": "x"}'),
            FINISH,
        ]
    )
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    assert len(released_in) == 2, "one release per generation turn"
    here = threading.get_ident()
    assert all(t != here for t in released_in), "must run in the generation thread"
