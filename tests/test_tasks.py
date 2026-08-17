import os
import sqlite3
from pathlib import Path

import pytest

from sous.tasks import Task, TaskState, TaskStore


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.db")


def _enqueue(store: TaskStore, title: str = "t") -> Task:
    return store.enqueue(
        title=title,
        instructions="do it",
        project_root="/tmp/p",
        context_files=["a.py"],
        verify_commands=["pytest"],
    )


# get()/claim_next() return Task | None, and report is dict | None, because
# production callers genuinely handle absence. Tests that go straight to an
# attribute are asserting the row exists; these say so once, and fail with
# that message instead of an AttributeError on None.


def _get(store: TaskStore, task_id: str) -> Task:
    task = store.get(task_id)
    assert task is not None, f"expected task {task_id} to exist"
    return task


def _claim(store: TaskStore) -> Task:
    task = store.claim_next()
    assert task is not None, "expected a queued task to claim"
    return task


def _report(store: TaskStore, task_id: str) -> dict:
    report = _get(store, task_id).report
    assert report is not None, f"expected task {task_id} to have a report"
    return report


def test_enqueue_and_get_roundtrip(store: TaskStore):
    t = _enqueue(store)
    got = _get(store, t.id)
    assert got.state == TaskState.QUEUED
    assert got.instructions == "do it"
    assert got.context_files == ["a.py"]
    assert got.verify_commands == ["pytest"]


def test_claim_next_is_fifo_and_marks_running(store: TaskStore):
    t1, t2 = _enqueue(store, "one"), _enqueue(store, "two")
    c = _claim(store)
    assert c.id == t1.id and c.state == TaskState.RUNNING
    assert _get(store, t1.id).started_at is not None
    assert _claim(store).id == t2.id
    assert store.claim_next() is None


def test_queue_position(store: TaskStore):
    t1, t2, t3 = _enqueue(store), _enqueue(store), _enqueue(store)
    store.claim_next()
    assert store.queue_position(t2.id) == 1
    assert store.queue_position(t3.id) == 2
    assert store.queue_position(t1.id) is None  # running, not waiting


def test_finish_stores_report(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {"summary": "did the thing"})
    got = _get(store, t.id)
    assert got.state == TaskState.DONE
    assert got.outcome == "completed"
    assert got.report == {"summary": "did the thing"}
    assert got.finished_at is not None


def test_approval_flow(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.request_approval(t.id, "go vet ./...")
    assert _get(store, t.id).state == TaskState.AWAITING_APPROVAL
    assert _get(store, t.id).pending_command == "go vet ./..."
    assert store.poll_approval(t.id) is None  # still pending
    assert store.respond_approval(t.id, approve=True) is True
    assert store.poll_approval(t.id) == "approved"
    got = _get(store, t.id)
    assert got.state == TaskState.RUNNING and got.pending_command is None


def test_respond_approval_when_not_awaiting_returns_false(store: TaskStore):
    t = _enqueue(store)
    assert store.respond_approval(t.id, approve=True) is False


def test_second_approval_response_does_not_overwrite_first(store: TaskStore):
    """A3: only the first response wins — a retried/duplicate response must
    not reverse it (deny-after-approve or approve-after-deny)."""
    t = _enqueue(store)
    store.claim_next()
    store.request_approval(t.id, "go vet ./...")
    assert store.respond_approval(t.id, approve=False) is True
    assert store.respond_approval(t.id, approve=True) is False  # loser
    assert store.poll_approval(t.id) == "denied"  # first response stands


def test_cancel_queued_is_immediate(store: TaskStore):
    t = _enqueue(store)
    assert store.cancel(t.id) is True
    assert _get(store, t.id).state == TaskState.CANCELLED


def test_cancel_running_sets_flag(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    assert store.cancel(t.id) is True
    assert _get(store, t.id).state == TaskState.RUNNING
    assert store.is_cancel_requested(t.id) is True
    store.mark_cancelled(t.id)
    assert _get(store, t.id).state == TaskState.CANCELLED


def test_cancel_finished_returns_false(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {})
    assert store.cancel(t.id) is False


def test_cancel_finished_does_not_set_flag(store: TaskStore):
    """B2: the active-task cancel path must be one atomic guarded UPDATE — a
    worker finishing between a read and a blind write would otherwise leave a
    terminal task flagged with cancel() lying True."""
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {})
    assert store.cancel(t.id) is False
    assert store.is_cancel_requested(t.id) is False
    assert _get(store, t.id).state == TaskState.DONE


def test_fail_without_extra_stores_only_error(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.fail(t.id, "boom")
    got = _get(store, t.id)
    assert got.state == TaskState.FAILED
    assert got.report == {"error": "boom"}


def test_fail_with_extra_merges_keys_into_report(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.fail(
        t.id, "boom", extra={"files_changed": [{"path": "a.py"}], "transcript_path": "/tmp/x.jsonl"}
    )
    got = _get(store, t.id)
    assert got.state == TaskState.FAILED
    assert got.report == {
        "error": "boom",
        "files_changed": [{"path": "a.py"}],
        "transcript_path": "/tmp/x.jsonl",
    }


def test_recover_interrupted(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    n = store.recover_interrupted()
    assert n == 1
    got = _get(store, t.id)
    assert got.state == TaskState.FAILED
    assert "restart" in _report(store, t.id)["error"]


def test_recover_interrupted_includes_persisted_changed_files(store: TaskStore, tmp_path: Path):
    """B2: recovery must surface the changed_files the worker persisted while
    running, plus the deterministic transcript path."""
    t = _enqueue(store)
    store.claim_next()
    store.update_changed_files(
        t.id, [{"path": "a.py", "kind": "modified", "before_sha": "aa", "after_sha": "bb"}]
    )
    assert store.recover_interrupted(tmp_path / "data") == 1
    got = _get(store, t.id)
    assert got.state == TaskState.FAILED
    report = _report(store, t.id)
    assert report["files_changed"] == [
        {"path": "a.py", "kind": "modified", "before_sha": "aa", "after_sha": "bb"}
    ]
    assert report["transcript_path"] == str(tmp_path / "data" / "tasks" / t.id / "transcript.jsonl")


_PRE_MIGRATION_SCHEMA = """
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


def test_changed_files_column_migrates_existing_db(tmp_path: Path):
    """B2: CREATE TABLE IF NOT EXISTS won't add changed_files to an existing
    database — TaskStore.__init__ must ALTER TABLE it in."""
    db = tmp_path / "tasks.db"
    conn = sqlite3.connect(db)
    conn.executescript(_PRE_MIGRATION_SCHEMA)
    conn.close()
    store = TaskStore(db)  # must migrate, not raise
    t = _enqueue(store)
    store.claim_next()
    store.update_changed_files(t.id, [{"path": "x.py"}])
    assert store.recover_interrupted(tmp_path) == 1
    assert _report(store, t.id)["files_changed"] == [{"path": "x.py"}]


def test_count_by_state_aggregates_all_rows(store: TaskStore):
    """E2: queue depth must come from an aggregate over ALL rows, not a
    LIMITed listing."""
    for i in range(5):
        _enqueue(store, f"t{i}")
    store.claim_next()
    counts = store.count_by_state()
    assert counts[TaskState.QUEUED] == 4
    assert counts[TaskState.RUNNING] == 1
    assert counts.get(TaskState.DONE, 0) == 0


def test_persistence_across_instances(tmp_path: Path):
    db = tmp_path / "tasks.db"
    t = _enqueue(TaskStore(db))
    assert _get(TaskStore(db), t.id).title == "t"


def test_prune_keeps_recent(store: TaskStore):
    ids = []
    for i in range(5):
        t = _enqueue(store, f"t{i}")
        store.claim_next()
        store.finish(t.id, "completed", {})
        ids.append(t.id)
    active = _enqueue(store, "active")
    deleted = store.prune(retention=2)
    assert deleted == 3
    assert store.get(ids[0]) is None
    assert store.get(ids[-1]) is not None
    assert store.get(active.id) is not None  # never prunes non-finished


def test_cancel_running_cannot_clobber_state(store: TaskStore):
    """Verifies cancel() on a running task does not clobber state atomically."""
    t = _enqueue(store)
    store.claim_next()  # Now running
    # Cancel the running task - should set flag, not transition state
    assert store.cancel(t.id) is True
    got = _get(store, t.id)
    assert got.state == TaskState.RUNNING  # Still running, not cancelled
    assert got.cancel_requested is True  # Flag is set
    assert got.finished_at is None  # Not finished


def _open_fd_count() -> int:
    """Descriptors held by this process. /dev/fd works on macOS and Linux."""
    return len(os.listdir("/dev/fd"))


def test_repeated_operations_do_not_leak_file_descriptors(store: TaskStore):
    """Every TaskStore method opens its own connection; each must close it.

    `with sqlite3.connect(...)` only commits the transaction, it does not
    close. Leaking two descriptors (db + WAL) per call crash-loops the daemon
    under launchd's 256-descriptor soft limit after ~128 operations.
    """
    _enqueue(store, "warmup")  # let one-time schema/WAL setup settle
    baseline = _open_fd_count()

    for i in range(50):
        task = _enqueue(store, f"t{i}")
        store.get(task.id)

    assert _open_fd_count() <= baseline + 2


def test_failed_statement_rolls_back(store: TaskStore):
    """_conn must keep commit-on-success / rollback-on-error while closing."""
    task = _enqueue(store, "keep")
    with pytest.raises(sqlite3.IntegrityError), store._conn() as c:
        c.execute("UPDATE tasks SET title = 'clobbered'")
        c.execute("INSERT INTO tasks (id) VALUES (?)", (task.id,))  # dup PK

    assert _get(store, task.id).title == "keep"


class _WalPragmaFails(sqlite3.Connection):
    """A connection whose WAL pragma fails, as SQLITE_BUSY does under contention."""

    def execute(self, sql: str, *args):
        if "journal_mode" in sql:
            raise sqlite3.OperationalError("database is locked")
        return super().execute(sql, *args)


def test_connection_closed_when_setup_fails(store: TaskStore, monkeypatch: pytest.MonkeyPatch):
    """Post-connect() setup must sit inside the try, or a raising PRAGMA leaks.

    `_conn` opens the descriptor before the try block; anything that raises
    between connect() and try: escapes the finally and leaks exactly the way
    the bare-connection version did.
    """
    _enqueue(store, "warmup")
    baseline = _open_fd_count()

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *a, **kw: real_connect(*a, **kw, factory=_WalPragmaFails),
    )

    for _ in range(50):
        with pytest.raises(sqlite3.OperationalError), store._conn():
            pass  # pragma: no cover — _conn raises on __enter__

    assert _open_fd_count() <= baseline + 2
