"""MCP layer: six tools wrapping SousService, plus daemon main()."""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from pathlib import Path

from mcp.server import MCPServer

from sous.config import (SousConfig, current_allowlist, load_config,
                         persist_allowlist_entry)
from sous.engine.base import EngineManager
from sous.tasks import FINISHED_STATES, Task, TaskState, TaskStore
from sous.toolexec import command_allowed
from sous.worker import run_worker_loop


def _mlx_memory_gb() -> float | None:
    try:
        import mlx.core as mx
        return round(mx.get_active_memory() / 1e9, 2)
    except Exception:  # noqa: BLE001 — mlx absent or API moved
        return None


class SousService:
    def __init__(self, store: TaskStore, engines: EngineManager, config: SousConfig):
        self.store = store
        self.engines = engines
        self.config = config

    def delegate_task(self, title: str, instructions: str, project_root: str,
                      context_files: list[str] | None = None,
                      verify_commands: list[str] | None = None) -> dict:
        root = Path(project_root)
        if not root.is_absolute():
            return {"error": f"project_root must be an absolute path: {project_root}"}
        if not root.is_dir():
            return {"error": f"project_root does not exist: {project_root}"}
        if self.config.data_dir.resolve().is_relative_to(root.resolve()):
            return {"error": f"project_root contains the sous data dir "
                             f"({self.config.data_dir}); a task rooted there "
                             f"could rewrite sous's own allowlist, task db, "
                             f"and audit transcripts"}
        allowlist = current_allowlist(self.config.config_path)
        bad = [c for c in (verify_commands or [])
               if not command_allowed(shlex.split(c), allowlist)]
        if bad:
            return {"error": "verify_commands not allowlisted: " + ", ".join(bad)}
        task = self.store.enqueue(
            title=title, instructions=instructions, project_root=str(root),
            context_files=context_files or [], verify_commands=verify_commands or [],
        )
        return {"task_id": task.id,
                "queue_position": self.store.queue_position(task.id) or 0}

    def _status_entry(self, t: Task) -> dict:
        end = t.finished_at or time.time()
        elapsed = (end - t.started_at) if t.started_at else None
        return {
            "id": t.id, "title": t.title, "state": t.state, "outcome": t.outcome,
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
        out = {"task_id": t.id, "state": t.state, "outcome": t.outcome,
               "report": t.report}
        if include_diff:
            out["diff"] = self._diff(t)
        return out

    def _diff(self, t: Task) -> str | None:
        files = [f["path"] for f in (t.report or {}).get("files_changed", [])]
        if not files or not self._is_git_repo(t.project_root):
            return None
        proc = subprocess.run(
            ["git", "diff", "--", *files], cwd=t.project_root,
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout[-30_000:]

    @staticmethod
    def _is_git_repo(project_root: str) -> bool:
        """Authoritative repo check: a `.git`-is-a-directory probe misses git
        worktrees and submodules, where `.git` is a FILE pointing at the real
        gitdir, and it also misses project_root being a subdirectory of a
        repo. `git rev-parse --is-inside-work-tree` handles all three."""
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=project_root,
            capture_output=True, text=True, timeout=10,
        )
        return check.returncode == 0 and check.stdout.strip() == "true"

    def cancel_task(self, task_id: str) -> dict:
        if self.store.get(task_id) is None:
            return {"error": f"unknown task: {task_id}"}
        return {"cancelled": self.store.cancel(task_id)}

    def respond_to_command_request(self, task_id: str, approve: bool,
                                   persist_to_allowlist: bool = False) -> dict:
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
        recent = self.store.list_recent(limit=200)
        return {
            "model": self.engines.status(),
            "memory_gb": _mlx_memory_gb(),
            "queue": {
                "queued": sum(t.state == TaskState.QUEUED for t in recent),
                "running": sum(t.state in (TaskState.RUNNING,
                                           TaskState.AWAITING_APPROVAL)
                               for t in recent),
            },
            "config": {
                "model_id": self.config.model_id,
                "port": self.config.server_port,
                "max_turns": self.config.max_turns,
                "max_minutes": self.config.max_minutes,
                "allowlist": current_allowlist(self.config.config_path),
            },
        }


def create_server(store: TaskStore, engines: EngineManager,
                  config: SousConfig) -> MCPServer:
    svc = SousService(store, engines, config)
    mcp = MCPServer("sous")

    @mcp.tool()
    def delegate_task(title: str, instructions: str, project_root: str,
                      context_files: list[str] | None = None,
                      verify_commands: list[str] | None = None) -> dict:
        """Delegate a mechanical, self-contained coding task to the local model.

        Use for volume-heavy, low-risk work (boilerplate, test scaffolding, bulk
        renames, docstrings, lint fixes) — NOT for architecture, tricky debugging,
        or security-sensitive code. The worker has NO conversation context:
        instructions must be fully self-contained (goal, constraints, acceptance
        criteria). Returns immediately with a task_id; poll with task_status and
        ALWAYS review the result diff before accepting.
        """
        return svc.delegate_task(title, instructions, project_root,
                                 context_files, verify_commands)

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
    def respond_to_command_request(task_id: str, approve: bool,
                                   persist_to_allowlist: bool = False) -> dict:
        """Resolve an awaiting_approval task. Only call after asking the human.
        approve=true runs the pending command once; persist_to_allowlist=true
        additionally adds it to the config allowlist for all future tasks."""
        return svc.respond_to_command_request(task_id, approve, persist_to_allowlist)

    @mcp.tool()
    def server_status() -> dict:
        """Daemon health: model load state, memory, queue depth, active config."""
        return svc.server_status()

    return mcp


def main() -> None:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = TaskStore(config.data_dir / "tasks.db")
    interrupted = store.recover_interrupted()
    if interrupted:
        print(f"sous: marked {interrupted} interrupted task(s) as failed")
    engines = EngineManager(config)
    stop = threading.Event()
    worker = threading.Thread(
        target=run_worker_loop, args=(store, engines, config, stop), daemon=True,
    )
    worker.start()
    mcp = create_server(store, engines, config)
    try:
        mcp.run(transport="streamable-http", host="127.0.0.1",
                port=config.server_port)
    finally:
        stop.set()
