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
WINDOW = 8192
ELIDED = "<tool_result>[elided: re-read the file if needed]</tool_result>"


def _run(
    tmp_path: Path,
    script: list[str],
    budget: Budget = Budget(turns=8, minutes=5),  # noqa: B008 — frozen; sharing it is harmless
    engine: FakeEngine | None = None,
):
    """`engine` replaces the default scripted one: a test that scripts token
    counts needs to hold on to the instance the loop ran against."""
    root = tmp_path / "project"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    fake = engine if engine is not None else FakeEngine(script)
    mgr = EngineManager(cfg, engine_factory=lambda _mid: fake)
    transcript = Transcript(tmp_path / "transcript.jsonl")
    try:
        result = run_loop(
            instructions="write hello.txt",
            context_files=(),
            root=root,
            engine=mgr.get(),
            window=WINDOW,
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


class _ElidingEngine(FakeEngine):
    """Over the window while any tool result is still verbatim in the
    messages, under it once they have all been elided."""

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        verbatim = any(
            m["role"] == "user" and m["content"].startswith("<tool_result name=") for m in messages
        )
        return 10_000 if verbatim else 100


class _FullEngine(FakeEngine):
    """No elision brings this prompt under the window."""

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return WINDOW + 1


def test_old_tool_results_are_elided_so_a_long_task_keeps_going(tmp_path: Path):
    script: list[str] = [WRITE, WRITE, FINISH]
    engine = _ElidingEngine(script)
    result, events, _ = _run(tmp_path, script, engine=engine)
    assert result.outcome == "completed"
    assert result.error is None
    assert result.elisions >= 1
    assert not any(e["event"] == "context_overflow" for e in events)
    results_sent = [m for m in engine.calls[-1] if m["content"].startswith("<tool_result")]
    assert results_sent and all(m["content"] == ELIDED for m in results_sent)


def test_a_prompt_with_nothing_left_to_elide_fails_the_run(tmp_path: Path):
    script: list[str] = [WRITE, FINISH]
    result, events, _ = _run(tmp_path, script, engine=_FullEngine(script))
    assert result.outcome == "failed"
    assert result.turns == 0
    assert result.elisions == 0
    assert result.error is not None and result.error.startswith("context overflow:")
    assert events[0] == {"event": "context_overflow", "error": result.error}
