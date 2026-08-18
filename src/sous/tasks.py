"""SQLite-backed task queue. Every method opens and closes its own connection (WAL)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    changed_files TEXT
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
        id=row["id"],
        title=row["title"],
        instructions=row["instructions"],
        project_root=row["project_root"],
        context_files=json.loads(row["context_files"]),
        verify_commands=json.loads(row["verify_commands"]),
        state=row["state"],
        outcome=row["outcome"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        last_activity=row["last_activity"],
        turns_used=row["turns_used"],
        report=json.loads(row["report"]) if row["report"] else None,
        pending_command=row["pending_command"],
        cancel_requested=bool(row["cancel_requested"]),
    )


class TaskStore:
    def __init__(self, db_path: Path):
        """Open (or create) the task database at db_path, creating its
        parent directory and migrating pre-existing DBs to the current
        schema."""
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            # CREATE TABLE IF NOT EXISTS won't add new columns to an existing
            # database — migrate changed_files in for pre-existing DBs.
            cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
            if "changed_files" not in cols:
                c.execute("ALTER TABLE tasks ADD COLUMN changed_files TEXT")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # `with sqlite3.connect(...)` only ends the transaction — it leaves the
        # descriptor open. Returning the bare connection leaked db + WAL per
        # call, crash-looping the daemon at launchd's 256-descriptor soft limit.
        conn = sqlite3.connect(self.db_path)
        try:
            # Setup belongs inside the try: the WAL pragma takes a lock and can
            # raise SQLITE_BUSY under contention, which would otherwise escape
            # the finally and leak the descriptor we just opened.
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def enqueue(
        self,
        title: str,
        instructions: str,
        project_root: str,
        context_files: list[str],
        verify_commands: list[str],
    ) -> Task:
        """Insert a new task in the queued state and return it."""
        task_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO tasks (id, title, instructions, project_root,"
                " context_files, verify_commands, state, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    title,
                    instructions,
                    project_root,
                    json.dumps(context_files),
                    json.dumps(verify_commands),
                    TaskState.QUEUED,
                    time.time(),
                ),
            )
        task = self.get(task_id)
        assert task is not None  # just inserted in the transaction above
        return task

    def get(self, task_id: str) -> Task | None:
        """Return the task with the given id, or None if it does not exist."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def count_by_state(self) -> dict[str, int]:
        """Aggregate task counts per state over ALL rows — queue depth must
        never be derived from a LIMITed listing."""
        with self._conn() as c:
            rows = c.execute("SELECT state, COUNT(*) AS n FROM tasks GROUP BY state").fetchall()
        return {r["state"]: r["n"] for r in rows}

    def list_recent(self, limit: int = 20) -> list[Task]:
        """Return up to limit tasks in any state, newest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def queue_position(self, task_id: str) -> int | None:
        """Return the task's 1-based position among queued tasks, or None if
        it is not queued."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM tasks WHERE state=? ORDER BY created_at",
                (TaskState.QUEUED,),
            ).fetchall()
        waiting = [r["id"] for r in rows]
        return waiting.index(task_id) + 1 if task_id in waiting else None

    def claim_next(self) -> Task | None:
        """Atomically mark the oldest queued task running (stamping
        started_at) and return it, or None if the queue is empty."""
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
        """Record the task's latest activity text and turn count."""
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET last_activity=?, turns_used=? WHERE id=?",
                (text, turns_used, task_id),
            )

    def update_changed_files(self, task_id: str, files: list[dict]) -> None:
        """Persist the changed-file list as the worker goes, so a daemon crash
        cannot hide which files the task already touched (recover_interrupted
        reads this back into the failure report)."""
        with self._conn() as c:
            c.execute("UPDATE tasks SET changed_files=? WHERE id=?", (json.dumps(files), task_id))

    def request_approval(self, task_id: str, command: str) -> None:
        """Move the task to awaiting_approval, storing the command that
        needs a human decision."""
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET state=?, pending_command=?, approval_response=NULL WHERE id=?",
                (TaskState.AWAITING_APPROVAL, command, task_id),
            )

    def respond_approval(self, task_id: str, approve: bool) -> bool:
        """Record the first approval decision for an awaiting task. Return True
        only for the decision that took effect; a duplicate or late response
        (task not awaiting, or already answered) returns False."""
        with self._conn() as c:
            # approval_response IS NULL makes the first response atomic and
            # final: a retried/duplicate response returns False instead of
            # reversing the first (which could leave a command persisted to
            # the allowlist by an approve, then reported as denied).
            cur = c.execute(
                "UPDATE tasks SET approval_response=? WHERE id=? AND state=?"
                " AND approval_response IS NULL",
                ("approved" if approve else "denied", task_id, TaskState.AWAITING_APPROVAL),
            )
            return cur.rowcount == 1

    def poll_approval(self, task_id: str) -> str | None:
        """Return "approved" or "denied" once a decision is recorded,
        atomically clearing the pending command and resuming the task;
        return None while no decision is recorded yet."""
        with self._conn() as c:
            row = c.execute("SELECT approval_response FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None or row["approval_response"] is None:
                return None
            c.execute(
                "UPDATE tasks SET state=?, pending_command=NULL, approval_response=NULL WHERE id=?",
                (TaskState.RUNNING, task_id),
            )
            return row["approval_response"]

    def finish(self, task_id: str, outcome: str, report: dict) -> None:
        """End the task as done, storing the outcome and report."""
        self._end(task_id, TaskState.DONE, outcome, report)

    def fail(self, task_id: str, reason: str, extra: dict | None = None) -> None:
        """End the task as failed, with the reason under "error" in the report
        and any extra fields merged alongside it."""
        self._end(task_id, TaskState.FAILED, None, {"error": reason, **(extra or {})})

    def mark_cancelled(self, task_id: str, extra: dict | None = None) -> None:
        """End the task as cancelled, storing any extra report fields (e.g.
        files changed before the cancel)."""
        # Same optional extra merge as fail(): a task cancelled after editing
        # files must still report files_changed and the transcript path.
        self._end(task_id, TaskState.CANCELLED, None, dict(extra or {}))

    def _end(self, task_id: str, state: str, outcome: str | None, report: dict) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET state=?, outcome=?, report=?, finished_at=? WHERE id=?",
                (state, outcome, json.dumps(report), time.time(), task_id),
            )

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task (atomically, with a state guard), or flag an
        active task for cooperative cancellation. Return True if either
        happened; False for an unknown or already finished task."""
        with self._conn() as c:
            # Try to cancel if queued (atomic transition with state guard)
            cur = c.execute(
                "UPDATE tasks SET state=?, report=?, finished_at=? WHERE id=? AND state=?",
                (TaskState.CANCELLED, json.dumps({}), time.time(), task_id, TaskState.QUEUED),
            )
            if cur.rowcount == 1:
                return True

            # Not queued: flag the task for cancellation only while it is
            # still active, as ONE guarded UPDATE. A SELECT-then-blind-UPDATE
            # would let a worker finishing between the two leave a terminal
            # task flagged and cancel() returning True, contrary to the
            # finished-task contract (mirrors the atomic queued path above).
            cur = c.execute(
                "UPDATE tasks SET cancel_requested=1 WHERE id=? AND state NOT IN (?,?,?)",
                (task_id, *FINISHED_STATES),
            )
            return cur.rowcount == 1

    def is_cancel_requested(self, task_id: str) -> bool:
        """Return True if a cooperative cancellation has been requested for
        the task (False for an unknown task)."""
        with self._conn() as c:
            row = c.execute("SELECT cancel_requested FROM tasks WHERE id=?", (task_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def recover_interrupted(self, data_dir: Path | None = None) -> int:
        """Fail tasks left running/awaiting_approval by a daemon crash. The
        report carries the changed_files the worker persisted while running
        and the deterministic transcript path — a restart must never hide
        which files the worker already touched."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, changed_files FROM tasks WHERE state IN (?, ?)",
                (TaskState.RUNNING, TaskState.AWAITING_APPROVAL),
            ).fetchall()
            for row in rows:
                report: dict = {
                    "error": "interrupted by daemon restart",
                    "files_changed": (
                        json.loads(row["changed_files"]) if row["changed_files"] else []
                    ),
                }
                if data_dir is not None:
                    report["transcript_path"] = str(
                        Path(data_dir) / "tasks" / row["id"] / "transcript.jsonl"
                    )
                c.execute(
                    "UPDATE tasks SET state=?, report=?, finished_at=? WHERE id=?",
                    (TaskState.FAILED, json.dumps(report), time.time(), row["id"]),
                )
            return len(rows)

    def prune(self, retention: int) -> int:
        """Delete finished tasks beyond the newest retention by finished_at,
        returning how many rows were deleted."""
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM tasks WHERE state IN (?,?,?) AND id NOT IN ("
                " SELECT id FROM tasks WHERE state IN (?,?,?)"
                " ORDER BY finished_at DESC LIMIT ?)",
                (*FINISHED_STATES, *FINISHED_STATES, retention),
            )
            return cur.rowcount
