"""The agent loop: generate → parse → execute → append, under budgets."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

from sous.config import SousConfig
from sous.context import ContextDecision, decide_context
from sous.engine.base import Engine, EngineManager, release_mlx_thread_state
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
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in project_root.iterdir())[:50]
    return SYSTEM_TEMPLATE.format(root=project_root, listing="\n".join(entries))


class _Transcript:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, **event) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


def _make_approval_hook(task: Task, store: TaskStore, config: SousConfig, deadline: float):
    """`deadline` is the task's wall-clock deadline (time.monotonic() terms):
    the approval wait ends at the earlier of the approval timeout and the
    task deadline — max_minutes bounds the whole task, approvals included."""

    def hook(command: str) -> bool:
        store.request_approval(task.id, command)
        wait_until = min(time.monotonic() + config.approval_timeout_minutes * 60, deadline)
        while time.monotonic() < wait_until:
            if store.is_cancel_requested(task.id):
                return False
            response = store.poll_approval(task.id)
            if response is not None:
                return response == "approved"
            time.sleep(0.05)
        store.respond_approval(task.id, approve=False)  # timeout/budget → deny
        store.poll_approval(task.id)  # restore running state
        return False

    return hook


def _execute(
    call: ToolCall, ex: ToolExecutor, config: SousConfig, approval, deadline: float
) -> str:
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
                # Clamp to the remaining task budget so one command cannot
                # run past the task deadline — but never pass a non-positive
                # timeout.
                remaining = deadline - time.monotonic()
                timeout = max(1, min(config.command_timeout_seconds, remaining))
                return ex.run_command(a["command"], approval=approval, timeout=timeout)
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


def _elide_if_needed(messages: list[dict], engine: Engine, max_context_tokens: int) -> int:
    """Elide old tool results until under the context cap. Returns the final
    token count — the caller must check it against the cap: when nothing
    elidable remains it can still be over, and an oversized prompt must never
    be sent (engine error or memory exhaustion is what the cap prevents)."""
    while (count := engine.count_tokens(messages, WORKER_TOOLS)) > max_context_tokens:
        for m in messages:
            if (
                m["role"] == "user"
                and m["content"].startswith("<tool_result")
                and "[elided" not in m["content"]
            ):
                m["content"] = "<tool_result>[elided: re-read the file if needed]</tool_result>"
                break
        else:
            return count  # nothing left to elide; still over the cap
    return count


class GenerationStalled(Exception):
    pass


def _generate_with_timeout(
    engine: Engine, messages: list[dict], max_tokens: int, timeout_seconds: float
) -> str:
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
        finally:
            # This thread dies after one generation, and an exiting thread
            # that touched mlx without releasing its streams segfaults the
            # whole daemon (ml-explore/mlx#4327). After the queue put, so the
            # waiting caller is never delayed by cleanup.
            release_mlx_thread_state()

    threading.Thread(target=_run, daemon=True).start()
    try:
        kind, value = result_q.get(timeout=timeout_seconds)
    except queue.Empty:
        raise GenerationStalled(f"generation stalled (> {round(timeout_seconds, 1)}s)") from None
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


def _tool_result_message(name: str, result: str) -> dict:
    """The user-role turn that feeds a tool's result back to the model."""
    return {
        "role": "user",
        "content": f'<tool_result name="{name}">\n{result}\n</tool_result>',
    }


def run_task(
    task: Task,
    store: TaskStore,
    engine: Engine,
    config: SousConfig,
    context: ContextDecision | None = None,
) -> None:
    # The caller (run_worker_loop) decides the window per task — memory
    # headroom moves with whatever else the machine is doing. No decision
    # means the fixed configured cap.
    if context is None:
        context = ContextDecision(config.max_context_tokens, "fixed")
    root = Path(task.project_root)
    ex = ToolExecutor(root, config.config_path, data_dir=config.data_dir)
    transcript = _Transcript(config.data_dir / "tasks" / task.id / "transcript.jsonl")

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(root)},
        {"role": "user", "content": task.instructions},
    ]
    for cf in task.context_files:
        try:
            content = ex.read_file(cf)
        except (PathViolation, OSError) as e:
            content = f"error: {e}"
        messages.append({"role": "user", "content": f"Contents of {cf}:\n{content}"})

    started = time.monotonic()
    # The one wall-clock authority for the whole task: the generation loop,
    # the approval wait, run_command timeouts, and the verify loop are all
    # bounded by this same deadline.
    deadline = started + config.max_minutes * 60
    approval = _make_approval_hook(task, store, config, deadline)
    turns = 0
    malformed = 0
    summary, concerns = "", ""
    outcome = "budget-exhausted"

    while turns < config.max_turns and time.monotonic() < deadline:
        if store.is_cancel_requested(task.id):
            transcript.log(event="cancelled")
            store.mark_cancelled(task.id, extra=_failure_extra(ex, transcript))
            return
        token_count = _elide_if_needed(messages, engine, context.tokens)
        if token_count > context.tokens:
            reason = (
                f"context overflow: {token_count} tokens exceeds the "
                f"{context.tokens}-token window ({context.reason}) with "
                f"nothing left to elide"
            )
            transcript.log(event="context_overflow", error=reason)
            store.fail(task.id, reason, extra=_failure_extra(ex, transcript))
            return
        # The window bounds prompt PLUS output: with auto sizing the window
        # can BE the model's native maximum, where an unbounded generation
        # would run past the positional limit, not just the memory estimate.
        output_room = context.tokens - token_count
        if output_room <= 0:
            reason = (
                f"context overflow: prompt fills the {context.tokens}-token "
                f"window ({context.reason}); no room to generate"
            )
            transcript.log(event="context_overflow", error=reason)
            store.fail(task.id, reason, extra=_failure_extra(ex, transcript))
            return
        remaining = max(0.1, deadline - time.monotonic())
        try:
            text = _generate_with_timeout(
                engine,
                messages,
                min(config.max_tokens_per_generation, output_room),
                remaining,
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
                store.fail(
                    task.id,
                    "model-confused: 3 consecutive malformed tool calls",
                    extra=_failure_extra(ex, transcript),
                )
                return
            messages.append({"role": "user", "content": FORMAT_REMINDER.format(error=e)})
            continue
        if not calls:
            malformed += 1
            if malformed >= MAX_CONSECUTIVE_MALFORMED:
                store.fail(
                    task.id,
                    "model-confused: 3 consecutive turns without a tool call",
                    extra=_failure_extra(ex, transcript),
                )
                return
            messages.append({"role": "user", "content": NUDGE})
            continue
        malformed = 0

        finished = False
        for call in calls:
            # Cancellation is honored at EVERY tool boundary: a response can
            # carry several calls, and a cancel that lands while one executes
            # must stop the rest — including a trailing finish, which would
            # otherwise turn a cancelled task into done.
            if store.is_cancel_requested(task.id):
                transcript.log(event="cancelled")
                store.mark_cancelled(task.id, extra=_failure_extra(ex, transcript))
                return
            if call.name == "finish":
                raw_summary = call.arguments.get("summary")
                if raw_summary is None or not str(raw_summary).strip():
                    # summary is required by the finish schema: a finish
                    # without one must not become a false completion with an
                    # empty report. Same recoverable tool-error shape as every
                    # other tool; the turn budget bounds retries.
                    result = "error: finish requires a non-empty summary"
                    transcript.log(
                        event="tool", name=call.name, arguments=call.arguments, result=result
                    )
                    messages.append(_tool_result_message(call.name, result))
                    continue
                summary = str(raw_summary)
                concerns = str(call.arguments.get("concerns", ""))
                outcome = "completed"
                finished = True
                break
            result = _execute(call, ex, config, approval, deadline)
            arg_hint = next(iter(call.arguments.values()), "")
            store.set_activity(task.id, f"{call.name}: {str(arg_hint)[:80]}", turns)
            # Persist the changed-file list as we go: if the daemon dies here,
            # recover_interrupted can still report what was already touched.
            store.update_changed_files(task.id, [vars(c) for c in ex.changed_files()])
            transcript.log(
                event="tool", name=call.name, arguments=call.arguments, result=result[:2000]
            )
            messages.append(_tool_result_message(call.name, result))
        if finished:
            break

    def _run_verify(cmd: str) -> str:
        # max_minutes bounds verify commands too: past the deadline each
        # remaining command is recorded as skipped — visible in the report,
        # never a silent multi-minute overshoot.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "skipped: task wall-clock budget exhausted"
        try:
            return ex.run_command(cmd, timeout=min(config.command_timeout_seconds, remaining))
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
        "budget": {
            "turns": turns,
            "seconds": round(time.monotonic() - started),
            "context_tokens": context.tokens,
            "context_reason": context.reason,
        },
        "transcript_path": str(transcript.path),
    }
    transcript.log(event="finished", outcome=outcome)
    store.finish(task.id, outcome, report)


def run_worker_loop(
    store: TaskStore,
    engines: EngineManager,
    config: SousConfig,
    stop: threading.Event,
    poll_interval: float = 0.5,
) -> None:
    while not stop.is_set():
        try:
            task = store.claim_next()
            if task is None:
                engines.unload_if_idle()
                stop.wait(poll_interval)
                continue
            try:
                engine = engines.get()
                # After get(): the weights must be loaded (and counted in
                # active memory) before headroom means anything.
                run_task(task, store, engine, config, context=decide_context(config))
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
