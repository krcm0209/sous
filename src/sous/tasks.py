"""SQLite-backed task queue. Every method opens its own connection (WAL)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class TaskState:
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINISHED_STATES = (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    project_root TEXT NOT NULL,
    context_files TEXT NOT NULL,
    verify_commands TEXT NOT NULL,
    state TEXT NOT NULL,
    outcome TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    last_activity TEXT NOT NULL DEFAULT '',
    turns_used INTEGER NOT NULL DEFAULT 0,
    report TEXT,
    pending_command TEXT,
    approval_response TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class Task:
    id: str
    title: str
    instructions: str
    project_root: str
    context_files: list[str]
    verify_commands: list[str]
    state: str
    outcome: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    last_activity: str
    turns_used: int
    report: dict | None
    pending_command: str | None
    cancel_requested: bool


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"], title=row["title"], instructions=row["instructions"],
        project_root=row["project_root"],
        context_files=json.loads(row["context_files"]),
        verify_commands=json.loads(row["verify_commands"]),
        state=row["state"], outcome=row["outcome"],
        created_at=row["created_at"], started_at=row["started_at"],
        finished_at=row["finished_at"], last_activity=row["last_activity"],
        turns_used=row["turns_used"],
        report=json.loads(row["report"]) if row["report"] else None,
        pending_command=row["pending_command"],
        cancel_requested=bool(row["cancel_requested"]),
    )


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def enqueue(self, title: str, instructions: str, project_root: str,
                context_files: list[str], verify_commands: list[str]) -> Task:
        task_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO tasks (id, title, instructions, project_root,"
                " context_files, verify_commands, state, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (task_id, title, instructions, project_root,
                 json.dumps(context_files), json.dumps(verify_commands),
                 TaskState.QUEUED, time.time()),
            )
        return self.get(task_id)

    def get(self, task_id: str) -> Task | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def list_recent(self, limit: int = 20) -> list[Task]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def queue_position(self, task_id: str) -> int | None:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM tasks WHERE state=? ORDER BY created_at",
                (TaskState.QUEUED,),
            ).fetchall()
        waiting = [r["id"] for r in rows]
        return waiting.index(task_id) + 1 if task_id in waiting else None

    def claim_next(self) -> Task | None:
        with self._conn() as c:
            row = c.execute(
                "UPDATE tasks SET state=?, started_at=? WHERE id = ("
                " SELECT id FROM tasks WHERE state=? ORDER BY created_at LIMIT 1"
                ") RETURNING id",
                (TaskState.RUNNING, time.time(), TaskState.QUEUED),
            ).fetchone()
            if row is None:
                return None
        return self.get(row["id"])

    def set_activity(self, task_id: str, text: str, turns_used: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET last_activity=?, turns_used=? WHERE id=?",
                (text, turns_used, task_id),
            )

    def request_approval(self, task_id: str, command: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET state=?, pending_command=?, approval_response=NULL"
                " WHERE id=?",
                (TaskState.AWAITING_APPROVAL, command, task_id),
            )

    def respond_approval(self, task_id: str, approve: bool) -> bool:
        with self._conn() as c:
            # approval_response IS NULL makes the first response atomic and
            # final: a retried/duplicate response returns False instead of
            # reversing the first (which could leave a command persisted to
            # the allowlist by an approve, then reported as denied).
            cur = c.execute(
                "UPDATE tasks SET approval_response=? WHERE id=? AND state=?"
                " AND approval_response IS NULL",
                ("approved" if approve else "denied", task_id,
                 TaskState.AWAITING_APPROVAL),
            )
            return cur.rowcount == 1

    def poll_approval(self, task_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT approval_response FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None or row["approval_response"] is None:
                return None
            c.execute(
                "UPDATE tasks SET state=?, pending_command=NULL,"
                " approval_response=NULL WHERE id=?",
                (TaskState.RUNNING, task_id),
            )
            return row["approval_response"]

    def finish(self, task_id: str, outcome: str, report: dict) -> None:
        self._end(task_id, TaskState.DONE, outcome, report)

    def fail(self, task_id: str, reason: str, extra: dict | None = None) -> None:
        self._end(task_id, TaskState.FAILED, None, {"error": reason, **(extra or {})})

    def mark_cancelled(self, task_id: str) -> None:
        self._end(task_id, TaskState.CANCELLED, None, {})

    def _end(self, task_id: str, state: str, outcome: str | None, report: dict) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET state=?, outcome=?, report=?, finished_at=?"
                " WHERE id=?",
                (state, outcome, json.dumps(report), time.time(), task_id),
            )

    def cancel(self, task_id: str) -> bool:
        with self._conn() as c:
            # Try to cancel if queued (atomic transition with state guard)
            cur = c.execute(
                "UPDATE tasks SET state=?, report=?, finished_at=? WHERE id=? AND state=?",
                (TaskState.CANCELLED, json.dumps({}), time.time(), task_id,
                 TaskState.QUEUED),
            )
            if cur.rowcount == 1:
                return True

            # If not queued, check if it's still active and set cancel_requested
            row = c.execute(
                "SELECT state FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None or row["state"] in FINISHED_STATES:
                return False
            # Must be running or awaiting_approval
            c.execute("UPDATE tasks SET cancel_requested=1 WHERE id=?", (task_id,))
        return True

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT cancel_requested FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def recover_interrupted(self) -> int:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE tasks SET state=?, report=?, finished_at=?"
                " WHERE state IN (?, ?)",
                (TaskState.FAILED,
                 json.dumps({"error": "interrupted by daemon restart"}),
                 time.time(), TaskState.RUNNING, TaskState.AWAITING_APPROVAL),
            )
            return cur.rowcount

    def prune(self, retention: int) -> int:
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM tasks WHERE state IN (?,?,?) AND id NOT IN ("
                " SELECT id FROM tasks WHERE state IN (?,?,?)"
                " ORDER BY finished_at DESC LIMIT ?)",
                (*FINISHED_STATES, *FINISHED_STATES, retention),
            )
            return cur.rowcount
