from pathlib import Path

import pytest

from sous.tasks import Task, TaskState, TaskStore


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / "tasks.db")


def _enqueue(store: TaskStore, title: str = "t") -> Task:
    return store.enqueue(
        title=title, instructions="do it", project_root="/tmp/p",
        context_files=["a.py"], verify_commands=["pytest"],
    )


def test_enqueue_and_get_roundtrip(store: TaskStore):
    t = _enqueue(store)
    got = store.get(t.id)
    assert got is not None
    assert got.state == TaskState.QUEUED
    assert got.instructions == "do it"
    assert got.context_files == ["a.py"]
    assert got.verify_commands == ["pytest"]


def test_claim_next_is_fifo_and_marks_running(store: TaskStore):
    t1, t2 = _enqueue(store, "one"), _enqueue(store, "two")
    c = store.claim_next()
    assert c.id == t1.id and c.state == TaskState.RUNNING
    assert store.get(t1.id).started_at is not None
    assert store.claim_next().id == t2.id
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
    got = store.get(t.id)
    assert got.state == TaskState.DONE
    assert got.outcome == "completed"
    assert got.report == {"summary": "did the thing"}
    assert got.finished_at is not None


def test_approval_flow(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.request_approval(t.id, "go vet ./...")
    assert store.get(t.id).state == TaskState.AWAITING_APPROVAL
    assert store.get(t.id).pending_command == "go vet ./..."
    assert store.poll_approval(t.id) is None  # still pending
    assert store.respond_approval(t.id, approve=True) is True
    assert store.poll_approval(t.id) == "approved"
    got = store.get(t.id)
    assert got.state == TaskState.RUNNING and got.pending_command is None


def test_respond_approval_when_not_awaiting_returns_false(store: TaskStore):
    t = _enqueue(store)
    assert store.respond_approval(t.id, approve=True) is False


def test_cancel_queued_is_immediate(store: TaskStore):
    t = _enqueue(store)
    assert store.cancel(t.id) is True
    assert store.get(t.id).state == TaskState.CANCELLED


def test_cancel_running_sets_flag(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    assert store.cancel(t.id) is True
    assert store.get(t.id).state == TaskState.RUNNING
    assert store.is_cancel_requested(t.id) is True
    store.mark_cancelled(t.id)
    assert store.get(t.id).state == TaskState.CANCELLED


def test_cancel_finished_returns_false(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {})
    assert store.cancel(t.id) is False


def test_fail_without_extra_stores_only_error(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.fail(t.id, "boom")
    got = store.get(t.id)
    assert got.state == TaskState.FAILED
    assert got.report == {"error": "boom"}


def test_fail_with_extra_merges_keys_into_report(store: TaskStore):
    t = _enqueue(store)
    store.claim_next()
    store.fail(t.id, "boom", extra={"files_changed": [{"path": "a.py"}],
                                    "transcript_path": "/tmp/x.jsonl"})
    got = store.get(t.id)
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
    got = store.get(t.id)
    assert got.state == TaskState.FAILED
    assert "restart" in got.report["error"]


def test_persistence_across_instances(tmp_path: Path):
    db = tmp_path / "tasks.db"
    t = _enqueue(TaskStore(db))
    assert TaskStore(db).get(t.id).title == "t"


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
    got = store.get(t.id)
    assert got.state == TaskState.RUNNING  # Still running, not cancelled
    assert got.cancel_requested is True  # Flag is set
    assert got.finished_at is None  # Not finished
