"""One suite task through the local model: the shape of the loop the worker
ran, minus the queue, the approvals and the audit a throwaway scratch project
does not need. Writes the transcript events the runner's metrics read."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sous.engine.base import GenerationStalled, ManagedEngine
from sous.protocol import ParseError, ToolCall, parse_tool_calls
from sous.tune.payload import TOOLS, TOOLSET, build_system_prompt
from sous.tune.suite.tools import ScratchTools, ToolError

MAX_CONSECUTIVE_MALFORMED = 3
FORMAT_REMINDER = (
    "Your tool call could not be parsed ({error}). Re-emit it using exactly "
    "the tool-call format specified in your instructions."
)
NUDGE = "You must call a tool to make progress. Call finish when done."
ELIDED = "<tool_result>[elided: re-read the file if needed]</tool_result>"


@dataclass(frozen=True)
class Budget:
    turns: int
    minutes: int
    tokens_per_generation: int = 4096


@dataclass(frozen=True)
class LoopResult:
    outcome: str
    turns: int
    seconds: float
    error: str | None
    summary: str
    concerns: str
    elisions: int


class Transcript:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, **event) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


def _tool_result_message(name: str, result: str) -> dict:
    return {"role": "user", "content": f'<tool_result name="{name}">\n{result}\n</tool_result>'}


def _execute(call: ToolCall, tools: ScratchTools, deadline: float, command_timeout: float) -> str:
    try:
        a = call.arguments
        match call.name:
            case "read_file":
                return tools.read_file(a["path"], a.get("offset", 0), a.get("limit", 2000))
            case "write_file":
                return tools.write_file(a["path"], a["content"])
            case "edit_file":
                return tools.edit_file(a["path"], a["old"], a["new"])
            case "list_dir":
                return tools.list_dir(a.get("path", "."))
            case "glob":
                return tools.glob(a["pattern"])
            case "grep":
                return tools.grep(a["pattern"], a.get("glob_pattern", "**/*"))
            case "run_command":
                # One command may not run past the task's own deadline.
                remaining = deadline - time.monotonic()
                return tools.run_command(
                    a["command"], timeout=max(1, min(command_timeout, remaining))
                )
            case _:
                return f"error: unhandled tool {call.name}"
    except ToolError as e:
        return f"error: {e}"
    except KeyError as e:
        return f"error: missing required argument {e}"
    except OSError as e:
        return f"error: {e}"
    except Exception as e:  # noqa: BLE001 — a tool's failure is the model's to read, never the loop's to raise
        return f"error: {e}"


def _elide_if_needed(messages: list[dict], engine: ManagedEngine, window: int) -> tuple[int, int]:
    """Rewrite the oldest verbatim tool results until the prompt is back under
    the window, returning the final count and how many were rewritten. The
    caller must still check the count: with nothing elidable left it can be
    over, and an oversized prompt must never be sent.

    A task whose early reads have scrolled out of the window is not a failed
    task — the model can read that file again, and dropping the text is far
    cheaper than dropping the run a dozen turns in."""
    elisions = 0
    while (count := engine.count_tokens(messages, TOOLS)) > window:
        for m in messages:
            if (
                m["role"] == "user"
                and m["content"].startswith("<tool_result")
                and "[elided" not in m["content"]
            ):
                m["content"] = ELIDED
                elisions += 1
                break
        else:
            return count, elisions  # nothing left to elide; still over the window
    return count, elisions


def run_loop(
    *,
    instructions: str,
    context_files: Sequence[str],
    root: Path,
    engine: ManagedEngine,
    window: int,
    budget: Budget,
    tools: ScratchTools,
    transcript: Transcript,
    command_timeout: float = 120.0,
) -> LoopResult:
    session = engine.session()
    try:
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(root)},
            {"role": "user", "content": instructions},
        ]
        for cf in context_files:
            try:
                content = tools.read_file(cf)
            except (ToolError, OSError) as e:
                content = f"error: {e}"
            messages.append({"role": "user", "content": f"Contents of {cf}:\n{content}"})

        started = time.monotonic()
        deadline = started + budget.minutes * 60
        turns = malformed = elisions = 0
        summary = concerns = ""
        outcome, error = "budget-exhausted", None

        def fail(reason: str) -> None:
            nonlocal outcome, error
            outcome, error = "failed", reason

        while turns < budget.turns and time.monotonic() < deadline:
            token_count, elided = _elide_if_needed(messages, engine, window)
            elisions += elided
            if token_count > window:
                reason = (
                    f"context overflow: {token_count} tokens exceeds the "
                    f"{window}-token window with nothing left to elide"
                )
                transcript.log(event="context_overflow", error=reason)
                fail(reason)
                break
            # The window bounds prompt PLUS output, so a prompt that lands
            # exactly on it has nowhere to generate.
            output_room = window - token_count
            if output_room <= 0:
                reason = (
                    f"context overflow: prompt fills the {window}-token window; no room to generate"
                )
                transcript.log(event="context_overflow", error=reason)
                fail(reason)
                break
            remaining = max(0.1, deadline - time.monotonic())
            try:
                text = session.generate(
                    messages,
                    TOOLS,
                    min(budget.tokens_per_generation, output_room),
                    timeout=remaining,
                )
            except GenerationStalled as e:
                if time.monotonic() >= deadline:
                    transcript.log(event="budget-exhausted", error=str(e))
                    break
                transcript.log(event="stalled", error=str(e))
                fail(str(e))
                break
            except Exception as e:  # noqa: BLE001 — the engine's failure is the run's outcome, not a crash of the suite
                transcript.log(event="engine_error", error=str(e))
                fail(f"engine error: {e}")
                break
            turns += 1
            transcript.log(event="generation", turn=turns, text=text)
            messages.append({"role": "assistant", "content": text})

            try:
                calls = parse_tool_calls(text, TOOLSET)
            except ParseError as e:
                malformed += 1
                transcript.log(event="malformed", error=str(e))
                if malformed >= MAX_CONSECUTIVE_MALFORMED:
                    fail("model-confused: 3 consecutive malformed tool calls")
                    break
                messages.append({"role": "user", "content": FORMAT_REMINDER.format(error=e)})
                continue
            if not calls:
                malformed += 1
                if malformed >= MAX_CONSECUTIVE_MALFORMED:
                    fail("model-confused: 3 consecutive turns without a tool call")
                    break
                messages.append({"role": "user", "content": NUDGE})
                continue
            malformed = 0

            finished = False
            for call in calls:
                if call.name == "finish":
                    raw_summary = call.arguments.get("summary")
                    if raw_summary is None or not str(raw_summary).strip():
                        result = "error: finish requires a non-empty summary"
                        transcript.log(
                            event="tool",
                            name=call.name,
                            arguments=call.arguments,
                            result=result[:2000],
                        )
                        messages.append(_tool_result_message(call.name, result))
                        continue
                    summary = str(raw_summary)
                    concerns = str(call.arguments.get("concerns", ""))
                    outcome, finished = "completed", True
                    break
                result = _execute(call, tools, deadline, command_timeout)
                transcript.log(
                    event="tool", name=call.name, arguments=call.arguments, result=result[:2000]
                )
                messages.append(_tool_result_message(call.name, result))
            if finished:
                break

        transcript.log(event="finished", outcome=outcome)
        return LoopResult(
            outcome, turns, time.monotonic() - started, error, summary, concerns, elisions
        )
    finally:
        session.close()
        engine.reset_prompt_cache(owner=session.thread)
