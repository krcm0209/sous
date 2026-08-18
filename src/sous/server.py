"""MCP layer: six tools wrapping SousService, plus daemon main()."""

from __future__ import annotations

import errno
import fcntl
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from mcp.server import MCPServer

from sous.config import SousConfig, current_allowlist, load_config, persist_allowlist_entry
from sous.engine.base import EngineManager, release_mlx_thread_state
from sous.tasks import FINISHED_STATES, Task, TaskState, TaskStore
from sous.toolexec import _is_within, command_allowed, terminate_active_commands
from sous.worker import run_worker_loop


def _mlx_memory_gb() -> float | None:
    try:
        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return round(mx.get_active_memory() / 1e9, 2)
    except Exception:  # noqa: BLE001 — mlx absent or API moved
        return None
    finally:
        # Runs in whatever short-lived MCP worker thread served the request;
        # mlx state left behind segfaults that thread's eventual exit
        # (ml-explore/mlx#4327).
        release_mlx_thread_state()


class SousService:
    def __init__(self, store: TaskStore, engines: EngineManager, config: SousConfig):
        self.store = store
        self.engines = engines
        self.config = config

    def delegate_task(
        self,
        title: str,
        instructions: str,
        project_root: str,
        context_files: list[str] | None = None,
        verify_commands: list[str] | None = None,
    ) -> dict:
        root = Path(project_root)
        if not root.is_absolute():
            return {"error": f"project_root must be an absolute path: {project_root}"}
        if not root.is_dir():
            return {"error": f"project_root does not exist: {project_root}"}
        if _is_within(self.config.data_dir.resolve(), root.resolve()):
            return {
                "error": f"project_root contains the sous data dir "
                f"({self.config.data_dir}); a task rooted there "
                f"could rewrite sous's own allowlist, task db, "
                f"and audit transcripts"
            }
        allowlist = current_allowlist(self.config.config_path)
        bad = []
        for c in verify_commands or []:
            try:
                argv = shlex.split(c)
            except ValueError:
                # Client-supplied string with e.g. an unmatched quote — a
                # structured error, never a ValueError out of the service.
                bad.append(c)
                continue
            if not command_allowed(argv, allowlist):
                bad.append(c)
        if bad:
            return {"error": "verify_commands not allowlisted (or unparseable): " + ", ".join(bad)}
        task = self.store.enqueue(
            title=title,
            instructions=instructions,
            project_root=str(root),
            context_files=context_files or [],
            verify_commands=verify_commands or [],
        )
        return {"task_id": task.id, "queue_position": self.store.queue_position(task.id) or 0}

    def _status_entry(self, t: Task) -> dict:
        end = t.finished_at or time.time()
        elapsed = (end - t.started_at) if t.started_at else None
        return {
            "id": t.id,
            "title": t.title,
            "state": t.state,
            "outcome": t.outcome,
            "queue_position": self.store.queue_position(t.id),
            "turns_used": t.turns_used,
            "elapsed_seconds": round(elapsed) if elapsed else None,
            "last_activity": t.last_activity,
            "pending_command": t.pending_command,
        }

    def task_status(self, task_id: str | None = None) -> dict:
        if task_id is not None:
            t = self.store.get(task_id)
            if t is None:
                return {"error": f"unknown task: {task_id}"}
            return self._status_entry(t)
        return {"tasks": [self._status_entry(t) for t in self.store.list_recent()]}

    def task_result(self, task_id: str, include_diff: bool = False) -> dict:
        t = self.store.get(task_id)
        if t is None:
            return {"error": f"unknown task: {task_id}"}
        if t.state not in FINISHED_STATES:
            return {"error": f"task is {t.state}; result not ready"}
        out = {"task_id": t.id, "state": t.state, "outcome": t.outcome, "report": t.report}
        if include_diff:
            out["diff"] = self._diff(t)
        return out

    def _diff(self, t: Task) -> str | None:
        files = [f["path"] for f in (t.report or {}).get("files_changed", [])]
        if not files or not self._is_git_repo(t.project_root):
            return None
        # `git diff` cannot show untracked files, and file CREATION is the
        # single most common worker action — so split the reported paths into
        # tracked (regular diff) and untracked (an add-style --no-index diff
        # against /dev/null each). Strictly read-only: no `git add`, no index
        # writes — this is a reporting path over the user's repository.
        ls = subprocess.run(
            ["git", "ls-files", "-z", "--", *files],
            cwd=t.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        tracked = set(ls.stdout.split("\0")) - {""}
        parts: list[str] = []
        if tracked_files := [f for f in files if f in tracked]:
            proc = subprocess.run(
                ["git", "diff", "--", *tracked_files],
                cwd=t.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            parts.append(proc.stdout)
        for f in files:
            if f in tracked:
                continue
            # --no-index exits 1 when the files differ — for a newly created
            # file that IS the expected outcome, not an error.
            proc = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", f],
                cwd=t.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            parts.append(proc.stdout)
        combined = "".join(parts)
        return combined[-30_000:] if combined else None

    @staticmethod
    def _is_git_repo(project_root: str) -> bool:
        """Authoritative repo check: a `.git`-is-a-directory probe misses git
        worktrees and submodules, where `.git` is a FILE pointing at the real
        gitdir, and it also misses project_root being a subdirectory of a
        repo. `git rev-parse --is-inside-work-tree` handles all three."""
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return check.returncode == 0 and check.stdout.strip() == "true"

    def cancel_task(self, task_id: str) -> dict:
        if self.store.get(task_id) is None:
            return {"error": f"unknown task: {task_id}"}
        return {"cancelled": self.store.cancel(task_id)}

    def respond_to_command_request(
        self, task_id: str, approve: bool, persist_to_allowlist: bool = False
    ) -> dict:
        t = self.store.get(task_id)
        if t is None:
            return {"error": f"unknown task: {task_id}"}
        if t.state != TaskState.AWAITING_APPROVAL:
            return {"error": f"task is {t.state}, not awaiting approval"}
        ok = self.store.respond_approval(task_id, approve)
        # Persist only after the approval actually landed — a timeout-deny
        # racing this call must not leave the command allowlisted forever.
        if ok and approve and persist_to_allowlist and t.pending_command:
            persist_allowlist_entry(t.pending_command, self.config.config_path)
        return {"ok": ok}

    def server_status(self) -> dict:
        counts = self.store.count_by_state()
        return {
            "model": self.engines.status(),
            "memory_gb": _mlx_memory_gb(),
            "queue": {
                "queued": counts.get(TaskState.QUEUED, 0),
                "running": counts.get(TaskState.RUNNING, 0)
                + counts.get(TaskState.AWAITING_APPROVAL, 0),
            },
            "config": {
                "model_id": self.config.model_id,
                "port": self.config.server_port,
                "max_turns": self.config.max_turns,
                "max_minutes": self.config.max_minutes,
                "allowlist": current_allowlist(self.config.config_path),
                "context": {
                    "mode": self.config.context_mode,
                    "fraction": self.config.context_fraction,
                    "min_tokens": self.config.context_min_tokens,
                    # Fixed mode's operative value — without it a client sees
                    # THAT the policy is fixed but not what it's fixed to.
                    "max_context_tokens": self.config.max_context_tokens,
                },
            },
        }


# Clients surface server instructions to the model unconditionally, even on
# surfaces that defer tool schemas out of context (Claude Code shows only tool
# NAMES until an explicit fetch) — so this text is what makes sous discoverable
# at decision time, and it must stand alone: when to delegate, why it beats
# working inline, then mechanics. Claude Code truncates instructions at 2KB
# with no marker, so the ending is the first thing an overrun loses; a test
# pins the size.
_INSTRUCTIONS = """\
sous runs a local MLX model on this Mac that executes self-contained coding
tasks in a sandboxed, queued tool loop. Every line the local worker generates
is output the user's Claude plan did not pay for — so when a task qualifies,
delegate it instead of generating the output inline.

Delegate mechanical, repetitive, low-risk work: boilerplate, test
scaffolding, bulk renames or migrations, docstring/comment sweeps, lint-fix
sweeps, fixture generation. Do NOT delegate architecture, subtle debugging,
security-sensitive code, API design, or anything needing this conversation's
context or taste. Delegation pays only when a short spec yields a large diff;
if you must author the content in the spec, work inline instead.

To delegate, call delegate_to_local_model with self-contained instructions —
the worker sees nothing of this chat — stating the goal, scope limits, and
acceptance criteria, plus project_root (absolute path). Trust the worker
with the how: point context_files at convention docs instead of restating
them, let allowlisted verify_commands catch what a linter would — worker
attempts are free, your prompt is not. It returns a task_id at once. If
your next step needs the result, block on `sous wait <task_id>`; otherwise
keep working and check task_status between steps.

If task_status shows awaiting_approval, the worker wants to run the command
in pending_command: relay it to the human verbatim (approve once / allowlist
/ deny), answer via respond_to_command_request — unanswered requests
auto-deny. Don't edit files the running task is touching.

Collect with task_result (include_diff=true) and review the diff like a PR
from an eager junior. A clean diff earns a brief acceptance — don't
re-narrate good work. On a miss, re-delegate a narrower self-contained task
scoped to just the flaws (the worker keeps nothing between tasks), and say
only that you're re-instructing it. budget-exhausted is partial
work: review, then finish or re-delegate narrower. The worker's output is a
draft, never a merge.
"""


def create_server(store: TaskStore, engines: EngineManager, config: SousConfig) -> MCPServer:
    svc = SousService(store, engines, config)
    mcp = MCPServer("sous", instructions=_INSTRUCTIONS)

    # The MCP-facing name deliberately differs from SousService.delegate_task:
    # on surfaces that defer tool schemas, the name is the only signal the
    # model has at decision time, so it must carry its own trigger (mechanical
    # work -> a local model). Clients already namespace by server — Claude
    # sees mcp__sous__delegate_to_local_model — so no sous_ prefix.
    @mcp.tool()
    def delegate_to_local_model(
        title: str,
        instructions: str,
        project_root: str,
        context_files: list[str] | None = None,
        verify_commands: list[str] | None = None,
    ) -> dict:
        """Delegate a mechanical, self-contained coding task to the local model.

        Every line the local worker generates is output the user's Claude plan
        did not pay for — prefer delegating qualifying work over generating it
        inline. It pays when a short spec yields a large diff; if you would be
        authoring the content in the spec, work inline instead. Use for
        volume-heavy, low-risk work (boilerplate, test scaffolding, bulk
        renames, docstrings, lint fixes) — NOT for architecture, tricky
        debugging, or security-sensitive code. The worker has NO conversation
        context: instructions must be self-contained (goal, scope limits,
        acceptance criteria) but lean — trust the worker with the how and
        re-delegate narrower on a miss. Returns immediately with a task_id;
        poll with task_status and ALWAYS review the result diff before
        accepting.
        """
        return svc.delegate_task(title, instructions, project_root, context_files, verify_commands)

    @mcp.tool()
    def task_status(task_id: str | None = None) -> dict:
        """Check delegated task progress. Omit task_id to list all recent tasks.

        A task in state awaiting_approval wants to run the command shown in
        pending_command — ask the human, then call respond_to_command_request.
        """
        return svc.task_status(task_id)

    @mcp.tool()
    def task_result(task_id: str, include_diff: bool = False) -> dict:
        """Fetch a finished task's report (summary, files changed, verify output,
        transcript path). Set include_diff=true for a unified diff (git repos).
        Treat the output as a draft: review it before accepting."""
        return svc.task_result(task_id, include_diff)

    @mcp.tool()
    def cancel_task(task_id: str) -> dict:
        """Cancel a queued task immediately, or stop a running task at its next
        tool boundary."""
        return svc.cancel_task(task_id)

    @mcp.tool()
    def respond_to_command_request(
        task_id: str, approve: bool, persist_to_allowlist: bool = False
    ) -> dict:
        """Resolve an awaiting_approval task. Only call after asking the human.
        approve=true runs the pending command once; persist_to_allowlist=true
        additionally adds it to the config allowlist for all future tasks."""
        return svc.respond_to_command_request(task_id, approve, persist_to_allowlist)

    @mcp.tool()
    def server_status() -> dict:
        """Daemon health: model load state, memory, queue depth, active config."""
        return svc.server_status()

    return mcp


def _login_shell_path() -> str | None:
    """PATH as the user's login shell sees it, or None if that can't be learned.

    Under launchd the daemon inherits the bare system PATH, so allowlisted
    commands (`uv run pytest`, anything user-installed) stop resolving even
    though they work in the user's terminal — each one silently degrades into
    a human approval request. Guessing at install time (shim directories,
    `which uv`, hardcoded Homebrew paths) bakes one machine's setup into a
    plist that rots; the user's own shell is the only authority on where
    their tools live, so ask it once at startup. Markers bracket the answer
    because login shells are entitled to print banners from init files, and
    `-i` is included because plenty of real PATH setup lives in rc files that
    non-interactive shells skip.
    """
    shell = os.environ.get("SHELL") or pwd.getpwuid(os.getuid()).pw_shell
    if not shell:
        return None
    probe = 'printf "%s" "<<sous-path>>$PATH<<sous-path-end>>"'
    try:
        out = subprocess.run(
            [shell, "-l", "-i", "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return None
    found = re.findall(r"<<sous-path>>(.*?)<<sous-path-end>>", out.stdout, re.DOTALL)
    return found[-1] if found and found[-1] else None


def _acquire_singleton_lock(data_dir: Path) -> IO[bytes]:
    """Take the daemon's exclusive lock, or exit if another daemon holds it.

    flock rather than a pidfile: the kernel releases it when the holder dies,
    including SIGKILL and the crashes launchd restarts from, so a stale lock
    can never wedge the daemon out of ever starting again.

    The returned handle must stay open for the process's lifetime — closing it
    (or letting it be collected) drops the lock.
    """
    lock_path = data_dir / "daemon.lock"
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        handle.close()
        if e.errno in (errno.EAGAIN, errno.EACCES):  # EWOULDBLOCK is EAGAIN
            raise SystemExit(f"sous: another daemon already holds {lock_path}") from None
        # ENOLCK, or a filesystem that cannot flock: a real startup failure.
        # Calling it contention would send the operator hunting for a second
        # daemon that does not exist.
        raise
    # Record the holder now that the lock is ours (so this cannot clobber a live
    # daemon's entry). `sous mcp` reads it to tell a restarted daemon from the
    # one it connected to: the port alone cannot, because launchd puts a new
    # daemon on the same port within a second and every old session is dead.
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n".encode())
    handle.flush()
    return handle


def _install_shutdown_handler(stop: threading.Event) -> None:
    """Kill in-flight commands on SIGTERM/SIGINT before the process goes away.

    Default handling terminates the daemon outright — measured: exit -15 with
    main()'s finally never reached — so cleanup in a finally would be dead
    code. Meanwhile a command's child is in its own session (start_new_session
    in toolexec), so it survives the daemon and goes on writing to the user's
    project. launchd restarts and `sous stop` both send SIGTERM, so this is the
    ordinary path out, not an edge case.

    The worker thread is deliberately not joined: a task can run for minutes,
    and killing the command groups is what protects the user's files. The task
    itself is reported failed by recover_interrupted() on the next start.
    """

    def handle(signum, frame) -> None:  # noqa: ARG001 — signal handler signature
        stop.set()
        killed = terminate_active_commands()
        if killed:
            print(f"sous: killed {killed} running command group(s)", file=sys.stderr)
        sys.stderr.flush()
        os._exit(0)

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, handle)


def main() -> None:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # Adopt the login shell's PATH before the worker exists: scrubbed_env()
    # passes PATH through, so this is what sandboxed commands resolve against,
    # identically however the daemon was launched (launchd, `sous mcp`, or a
    # terminal). Fallback on failure: whatever PATH we inherited.
    if login_path := _login_shell_path():
        os.environ["PATH"] = login_path
    # Before ANY queue access: recover_interrupted() below would fail a running
    # daemon's in-flight task, and the worker would load a second copy of the
    # model. The port bind at the end of this function is far too late to be
    # the guard. `_lock` is unused by design — it must stay open to hold it.
    _lock = _acquire_singleton_lock(config.data_dir)
    store = TaskStore(config.data_dir / "tasks.db")
    interrupted = store.recover_interrupted(config.data_dir)
    if interrupted:
        print(f"sous: marked {interrupted} interrupted task(s) as failed")
    engines = EngineManager(config)
    stop = threading.Event()
    _install_shutdown_handler(stop)
    worker = threading.Thread(
        target=run_worker_loop,
        args=(store, engines, config, stop),
        daemon=True,
    )
    worker.start()
    mcp = create_server(store, engines, config)
    try:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=config.server_port)
    finally:
        stop.set()
