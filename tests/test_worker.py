import dataclasses
import json
import sqlite3
import threading
import time
from pathlib import Path

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
        max_turns=10, max_minutes=1, approval_timeout_minutes=1,
    )
    store = TaskStore(tmp_path / "tasks.db")
    return root, cfg, store


def _start(store: TaskStore, root: Path, verify=(), context=()):
    t = store.enqueue(
        title="t", instructions="do the thing", project_root=str(root),
        context_files=list(context), verify_commands=list(verify),
    )
    return store.claim_next()


CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "did it", "concerns": ""}')


def test_happy_path_write_then_finish(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([
        CALL.format(name="write_file",
                    args='{"path": "out.txt", "content": "hello"}'),
        FINISH,
    ])
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
    assert all(json.loads(l) for l in lines)
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
    engine = FakeEngine(['<tool_call>{bad json}</tool_call>'] * 3)
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
    engine = FakeEngine([
        CALL.format(name="write_file",
                    args='{"path": "hello.txt", "content": "hello sous"}'),
        "let me think about this some more with no tool call",
        "still thinking, no tool call here either",
        "and a third turn with no tool call at all",
    ])
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
    engine = FakeEngine([
        '<tool_call>{bad}</tool_call>',
        CALL.format(name="list_dir", args="{}"),
        '<tool_call>{bad}</tool_call>',
        '<tool_call>{bad}</tool_call>',
        FINISH,
    ])
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
    engine = FakeEngine([
        CALL.format(name="read_file", args='{"path": "../../etc/passwd"}'),
        FINISH,
    ])
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    result_turn = engine.calls[1]
    assert any("escapes project root" in str(m.get("content", "")) for m in result_turn)


def test_approval_flow_approved(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([
        CALL.format(name="run_command", args='{"command": "/bin/echo custom"}'),
        FINISH,
    ])

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
    engine = FakeEngine([
        CALL.format(name="run_command", args='{"command": "/bin/echo custom"}'),
        FINISH,
    ])

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


def test_generation_stall_fails_task(env):
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_minutes=0.005)  # 0.3 s wall budget

    class StallingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            time.sleep(2)
            return FINISH

    task = _start(store, root)
    run_task(task, store, StallingEngine([FINISH]), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "stalled" in got.report["error"]


def test_context_elision_replaces_old_tool_results(env):
    root, cfg, store = env
    (root / "big.txt").write_text("word " * 4000)
    cfg = dataclasses.replace(cfg, max_context_tokens=1500)
    task = _start(store, root)
    engine = FakeEngine([
        CALL.format(name="read_file", args='{"path": "big.txt"}'),
        CALL.format(name="read_file", args='{"path": "hello.py"}'),
        FINISH,
    ])
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    final_turn = engine.calls[-1]
    assert any("[elided" in str(m.get("content", "")) for m in final_turn)


def test_tool_error_from_bad_argument_value_does_not_crash_task(env):
    root, cfg, store = env
    task = _start(store, root)
    engine = FakeEngine([
        CALL.format(name="grep", args='{"pattern": "("}'),  # invalid regex
        FINISH,
    ])
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


def test_worker_loop_survives_bookkeeping_exception(env, capsys):
    """M1: a transient failure in claim_next (outside the per-task try) must
    not kill the worker thread — the loop should log and keep polling."""
    root, cfg, store = env
    stop = threading.Event()
    calls = {"n": 0}

    class FlakyStore:
        def claim_next(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            stop.set()
            return None

    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    run_worker_loop(FlakyStore(), engines, cfg, stop, poll_interval=0.01)
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
