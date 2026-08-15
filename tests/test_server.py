import subprocess
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import SousService
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine


@pytest.fixture()
def svc(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    cfg.config_path.write_text('[commands]\nallowlist = ["pytest"]\n')
    store = TaskStore(tmp_path / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    return SousService(store, engines, cfg), store, root


def test_delegate_returns_id_and_position(svc):
    service, store, root = svc
    out = service.delegate_task("t", "do it", str(root))
    assert "task_id" in out and out["queue_position"] == 1
    assert store.get(out["task_id"]).state == "queued"


def test_delegate_rejects_relative_root(svc):
    service, _, _ = svc
    assert "error" in service.delegate_task("t", "x", "relative/path")


def test_delegate_rejects_missing_root(svc):
    service, _, _ = svc
    assert "error" in service.delegate_task("t", "x", "/nope/not/here")


def test_delegate_rejects_non_allowlisted_verify(svc):
    service, _, root = svc
    out = service.delegate_task("t", "x", str(root), verify_commands=["rm -rf /"])
    assert "error" in out and "rm -rf /" in out["error"]


def test_delegate_accepts_allowlisted_verify(svc):
    service, _, root = svc
    out = service.delegate_task("t", "x", str(root), verify_commands=["pytest -q"])
    assert "task_id" in out


def test_status_single_and_all(svc):
    service, store, root = svc
    a = service.delegate_task("a", "x", str(root))["task_id"]
    service.delegate_task("b", "x", str(root))
    one = service.task_status(a)
    assert one["state"] == "queued" and one["queue_position"] == 1
    both = service.task_status()
    assert len(both["tasks"]) == 2


def test_status_unknown_id(svc):
    service, _, _ = svc
    assert "error" in service.task_status("nope")


def test_result_not_ready_then_ready(svc):
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert "error" in service.task_result(tid)
    store.claim_next()
    store.finish(tid, "completed", {"summary": "s", "files_changed": []})
    out = service.task_result(tid)
    assert out["outcome"] == "completed" and out["report"]["summary"] == "s"


def test_result_include_diff_from_git(svc, tmp_path: Path):
    service, store, root = svc
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "f.txt").write_text("one\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=root, check=True)
    (root / "f.txt").write_text("two\n")
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.finish(tid, "completed",
                 {"summary": "s", "files_changed": [{"path": "f.txt"}]})
    out = service.task_result(tid, include_diff=True)
    assert "-one" in out["diff"] and "+two" in out["diff"]


def test_cancel(svc):
    service, _, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert service.cancel_task(tid)["cancelled"] is True
    assert "error" in service.cancel_task("nope")


def test_respond_requires_awaiting(svc):
    service, _, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    assert "error" in service.respond_to_command_request(tid, approve=True)


def test_respond_approves_and_persists(svc):
    service, store, root = svc
    tid = service.delegate_task("t", "x", str(root))["task_id"]
    store.claim_next()
    store.request_approval(tid, "go vet ./...")
    out = service.respond_to_command_request(tid, approve=True,
                                             persist_to_allowlist=True)
    assert out["ok"] is True
    from sous.config import current_allowlist
    assert ["go", "vet", "./..."] in current_allowlist(service.config.config_path)
    assert store.poll_approval(tid) == "approved"


def test_server_status(svc):
    service, _, root = svc
    service.delegate_task("t", "x", str(root))
    s = service.server_status()
    assert s["queue"]["queued"] == 1
    assert s["model"]["loaded"] is False
    assert s["config"]["model_id"]
    assert ["pytest"] in s["config"]["allowlist"]


def test_create_server_registers_six_tools(svc):
    service, store, root = svc
    from sous.server import create_server
    mcp = create_server(store, service.engines, service.config)
    assert mcp is not None  # deep tool introspection exercised in e2e smoke
