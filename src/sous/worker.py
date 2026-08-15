"""The agent loop: generate → parse → execute → append, under budgets."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import Engine, EngineManager
from sous.protocol import WORKER_TOOLS, ParseError, ToolCall, parse_tool_calls
from sous.tasks import Task, TaskStore
from sous.toolexec import PathViolation, ToolExecutor

MAX_CONSECUTIVE_MALFORMED = 3

SYSTEM_TEMPLATE = """You are sous, a focused coding subcontractor working alone \
inside one project. Complete the task below exactly as instructed.

Rules:
- Make minimal changes. Follow the existing code style.
- Read a file before editing it. Never invent file contents.
- Use run_command only for tests, linters, and formatters.
- When the task is complete (or you cannot proceed), call finish with an honest summary.

Project root: {root}
Top-level entries:
{listing}
"""

FORMAT_REMINDER = (
    "Your tool call could not be parsed ({error}). Re-emit it using exactly "
    "the tool-call format specified in your instructions."
)

NUDGE = "You must call a tool to make progress. Call finish when done."


def build_system_prompt(project_root: Path) -> str:
    entries = sorted(
        e.name + ("/" if e.is_dir() else "") for e in project_root.iterdir()
    )[:50]
    return SYSTEM_TEMPLATE.format(root=project_root, listing="\n".join(entries))


class _Transcript:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, **event) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


def _make_approval_hook(task: Task, store: TaskStore, config: SousConfig):
    def hook(command: str) -> bool:
        store.request_approval(task.id, command)
        deadline = time.monotonic() + config.approval_timeout_minutes * 60
        while time.monotonic() < deadline:
            if store.is_cancel_requested(task.id):
                return False
            response = store.poll_approval(task.id)
            if response is not None:
                return response == "approved"
            time.sleep(0.05)
        store.respond_approval(task.id, approve=False)  # timeout → deny
        store.poll_approval(task.id)                    # restore running state
        return False
    return hook


def _execute(call: ToolCall, ex: ToolExecutor, config: SousConfig, approval) -> str:
    try:
        a = call.arguments
        match call.name:
            case "read_file":
                return ex.read_file(a["path"], a.get("offset", 0), a.get("limit", 2000))
            case "write_file":
                return ex.write_file(a["path"], a["content"])
            case "edit_file":
                return ex.edit_file(a["path"], a["old"], a["new"])
            case "list_dir":
                return ex.list_dir(a.get("path", "."))
            case "glob":
                return ex.glob(a["pattern"])
            case "grep":
                return ex.grep(a["pattern"], a.get("glob_pattern", "**/*"))
            case "run_command":
                return ex.run_command(a["command"], approval=approval,
                                      timeout=config.command_timeout_seconds)
            case _:
                return f"error: unhandled tool {call.name}"
    except PathViolation as e:
        return f"error: {e}"
    except KeyError as e:
        return f"error: missing required argument {e}"
    except OSError as e:
        return f"error: {e}"
    except Exception as e:
        # A small local model routinely emits a bad argument VALUE (an
        # unparseable regex, a wrong-typed offset, an absolute glob
        # pattern, ...). That must come back as a tool result the model
        # can recover from, never crash the whole task.
        return f"error: {e}"


def _elide_if_needed(messages: list[dict], engine: Engine, config: SousConfig) -> int:
    """Elide old tool results until under the context cap. Returns the final
    token count — the caller must check it against the cap: when nothing
    elidable remains it can still be over, and an oversized prompt must never
    be sent (engine error or memory exhaustion is what the cap prevents)."""
    while (count := engine.count_tokens(messages, WORKER_TOOLS)) > config.max_context_tokens:
        for m in messages:
            if (m["role"] == "user" and m["content"].startswith("<tool_result")
                    and "[elided" not in m["content"]):
                m["content"] = "<tool_result>[elided: re-read the file if needed]</tool_result>"
                break
        else:
            return count  # nothing left to elide; still over the cap
    return count


class GenerationStalled(Exception):
    pass


def _generate_with_timeout(engine: Engine, messages: list[dict], max_tokens: int,
                           timeout_seconds: float) -> str:
    """Run a (synchronous, uninterruptible) MLX generation with a deadline.

    MLX generation can't be aborted mid-stream. The generation runs on a
    daemon thread so a truly wedged model can never block interpreter (or
    daemon-shutdown) exit — only the queue hand-off is waited on, with a
    bounded timeout. On timeout the thread is abandoned to finish (or hang)
    in the background; its eventual result or exception is simply dropped.
    """
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_q.put(("ok", engine.generate(messages, WORKER_TOOLS, max_tokens)))
        except BaseException as e:  # noqa: BLE001 — relayed to the caller
            result_q.put(("err", e))

    threading.Thread(target=_run, daemon=True).start()
    try:
        kind, value = result_q.get(timeout=timeout_seconds)
    except queue.Empty:
        raise GenerationStalled(
            f"generation stalled (> {round(timeout_seconds, 1)}s)"
        ) from None
    if kind == "err":
        raise value
    return value


def _failure_extra(ex: ToolExecutor, transcript: _Transcript) -> dict:
    """Attached to every terminal store.fail() in run_task so a failed task
    still tells Claude what changed on disk and where to audit — silent,
    unreviewed file modifications are the worst failure shape here."""
    return {
        "files_changed": [vars(c) for c in ex.changed_files()],
        "transcript_path": str(transcript.path),
    }


def run_task(task: Task, store: TaskStore, engine: Engine, config: SousConfig) -> None:
    root = Path(task.project_root)
    ex = ToolExecutor(root, config.config_path, data_dir=config.data_dir)
    transcript = _Transcript(config.data_dir / "tasks" / task.id / "transcript.jsonl")
    approval = _make_approval_hook(task, store, config)

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(root)},
        {"role": "user", "content": task.instructions},
    ]
    for cf in task.context_files:
        try:
            content = ex.read_file(cf)
        except (PathViolation, OSError) as e:
            content = f"error: {e}"
        messages.append({"role": "user",
                         "content": f"Contents of {cf}:\n{content}"})

    started = time.monotonic()
    turns = 0
    malformed = 0
    summary, concerns = "", ""
    outcome = "budget-exhausted"

    deadline = started + config.max_minutes * 60
    while turns < config.max_turns and time.monotonic() < deadline:
        if store.is_cancel_requested(task.id):
            transcript.log(event="cancelled")
            store.mark_cancelled(task.id, extra=_failure_extra(ex, transcript))
            return
        token_count = _elide_if_needed(messages, engine, config)
        if token_count > config.max_context_tokens:
            reason = (f"context overflow: {token_count} tokens exceeds "
                      f"max_context_tokens={config.max_context_tokens} with "
                      f"nothing left to elide")
            transcript.log(event="context_overflow", error=reason)
            store.fail(task.id, reason, extra=_failure_extra(ex, transcript))
            return
        remaining = max(0.1, deadline - time.monotonic())
        try:
            text = _generate_with_timeout(
                engine, messages, config.max_tokens_per_generation, remaining,
            )
        except GenerationStalled as e:
            if time.monotonic() >= deadline:
                # The generation timeout IS the remaining wall-clock budget,
                # so a timeout here means the budget ran out, not that the
                # engine wedged. Per spec, hitting any budget ends the task
                # as done/budget-exhausted with a partial report.
                transcript.log(event="budget-exhausted", error=str(e))
                break
            transcript.log(event="stalled", error=str(e))
            store.fail(task.id, str(e), extra=_failure_extra(ex, transcript))
            return
        except Exception as e:
            # The engine raised something other than a stall (a real
            # generation failure). Fail the task cleanly rather than let it
            # escape run_task's "never raises" contract.
            transcript.log(event="engine_error", error=str(e))
            store.fail(task.id, f"engine error: {e}", extra=_failure_extra(ex, transcript))
            return
        turns += 1
        transcript.log(event="generation", turn=turns, text=text)
        messages.append({"role": "assistant", "content": text})

        try:
            calls = parse_tool_calls(text)
        except ParseError as e:
            malformed += 1
            transcript.log(event="malformed", error=str(e))
            if malformed >= MAX_CONSECUTIVE_MALFORMED:
                store.fail(task.id, "model-confused: 3 consecutive malformed tool calls",
                          extra=_failure_extra(ex, transcript))
                return
            messages.append({"role": "user",
                             "content": FORMAT_REMINDER.format(error=e)})
            continue
        if not calls:
            malformed += 1
            if malformed >= MAX_CONSECUTIVE_MALFORMED:
                store.fail(task.id, "model-confused: 3 consecutive turns without a tool call",
                          extra=_failure_extra(ex, transcript))
                return
            messages.append({"role": "user", "content": NUDGE})
            continue
        malformed = 0

        finished = False
        for call in calls:
            if call.name == "finish":
                summary = str(call.arguments.get("summary", ""))
                concerns = str(call.arguments.get("concerns", ""))
                outcome = "completed"
                finished = True
                break
            result = _execute(call, ex, config, approval)
            arg_hint = next(iter(call.arguments.values()), "")
            store.set_activity(task.id, f"{call.name}: {str(arg_hint)[:80]}", turns)
            # Persist the changed-file list as we go: if the daemon dies here,
            # recover_interrupted can still report what was already touched.
            store.update_changed_files(task.id,
                                       [vars(c) for c in ex.changed_files()])
            transcript.log(event="tool", name=call.name,
                           arguments=call.arguments, result=result[:2000])
            messages.append({"role": "user",
                             "content": f'<tool_result name="{call.name}">\n{result}\n</tool_result>'})
        if finished:
            break

    def _run_verify(cmd: str) -> str:
        try:
            return ex.run_command(cmd, timeout=config.command_timeout_seconds)
        except Exception as e:
            # run_command already turns most failures (timeout, missing
            # binary, denied) into strings, but not all of them (e.g. a
            # verify script without the execute bit raises PermissionError).
            # Budget exhaustion is not a failure — the report must still
            # assemble and the task must still reach `done`.
            return f"error: {e}"

    verify = [{"command": cmd, "output": _run_verify(cmd)} for cmd in task.verify_commands]
    report = {
        "summary": summary,
        "concerns": concerns,
        "files_changed": [vars(c) for c in ex.changed_files()],
        "verify": verify,
        "budget": {"turns": turns, "seconds": round(time.monotonic() - started)},
        "transcript_path": str(transcript.path),
    }
    transcript.log(event="finished", outcome=outcome)
    store.finish(task.id, outcome, report)


def run_worker_loop(store: TaskStore, engines: EngineManager, config: SousConfig,
                    stop: threading.Event, poll_interval: float = 0.5) -> None:
    while not stop.is_set():
        try:
            task = store.claim_next()
            if task is None:
                engines.unload_if_idle()
                stop.wait(poll_interval)
                continue
            try:
                engine = engines.get()
                run_task(task, store, engine, config)
            except Exception as e:  # noqa: BLE001 — task-scoped failure
                store.fail(task.id, f"worker error: {e}")
            finally:
                engines.touch()
                store.prune(config.task_retention)
        except Exception as e:  # noqa: BLE001 — worker thread must never die
            # Bookkeeping failure (claim/prune/unload, or even fail() itself):
            # the MCP main thread keeps serving, so a dead worker thread would
            # wedge the queue forever with no launchd self-heal. Log, back
            # off one poll interval, keep looping.
            print(f"sous: worker loop error (continuing): {e}", file=sys.stderr)
            stop.wait(poll_interval)
