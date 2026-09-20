import json
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.tune.suite.loop import Budget, Transcript, run_loop
from sous.tune.suite.tools import ScratchTools
from tests.fake_engine import FakeEngine

CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "done", "concerns": ""}')
WRITE = CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hi"}')


def _run(
    tmp_path: Path,
    script: list[str],
    budget: Budget = Budget(turns=8, minutes=5),  # noqa: B008 — frozen; sharing it is harmless
):
    root = tmp_path / "project"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    mgr = EngineManager(cfg, engine_factory=lambda _mid: FakeEngine(script))
    transcript = Transcript(tmp_path / "transcript.jsonl")
    try:
        result = run_loop(
            instructions="write hello.txt",
            context_files=(),
            root=root,
            engine=mgr.get(),
            window=8192,
            budget=budget,
            tools=ScratchTools(root),
            transcript=transcript,
        )
    finally:
        mgr.unload_now()
    events = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    return result, events, root


def test_a_finished_task_is_completed_with_its_summary(tmp_path: Path):
    result, events, root = _run(tmp_path, [WRITE, FINISH])
    assert result.outcome == "completed"
    assert result.error is None
    assert result.turns == 2
    assert result.summary == "done"
    assert (root / "hello.txt").read_text() == "hi"
    assert [e["event"] for e in events] == ["generation", "tool", "generation", "finished"]
    assert events[1]["name"] == "write_file"
    assert events[-1]["outcome"] == "completed"


def test_running_out_of_turns_is_budget_exhausted_not_failed(tmp_path: Path):
    result, events, _ = _run(tmp_path, [WRITE, WRITE], Budget(turns=1, minutes=5))
    assert result.outcome == "budget-exhausted"
    assert result.error is None
    assert result.turns == 1
    assert events[-1] == {"event": "finished", "outcome": "budget-exhausted"}


def test_three_malformed_turns_fail_the_task_and_are_counted(tmp_path: Path):
    bad = "<tool_call>{not json}</tool_call>"
    result, events, _ = _run(tmp_path, [bad, bad, bad])
    assert result.outcome == "failed"
    assert result.error == "model-confused: 3 consecutive malformed tool calls"
    assert sum(e["event"] == "malformed" for e in events) == 3


def test_three_turns_without_a_tool_call_fail_the_task(tmp_path: Path):
    result, _, _ = _run(tmp_path, ["thinking...", "still thinking", "hmm"])
    assert result.outcome == "failed"
    assert result.error == "model-confused: 3 consecutive turns without a tool call"


def test_an_engine_error_is_a_failed_run(tmp_path: Path):
    result, events, _ = _run(tmp_path, [])  # FakeEngine raises once its script is exhausted
    assert result.outcome == "failed"
    assert result.error is not None and result.error.startswith("engine error: ")
    assert events[0]["event"] == "engine_error"


def test_finish_without_a_summary_is_a_tool_error_and_the_loop_goes_on(tmp_path: Path):
    empty = CALL.format(name="finish", args='{"summary": ""}')
    result, events, _ = _run(tmp_path, [empty, FINISH])
    assert result.outcome == "completed"
    assert events[1] == {
        "event": "tool",
        "name": "finish",
        "arguments": {"summary": ""},
        "result": "error: finish requires a non-empty summary",
    }
