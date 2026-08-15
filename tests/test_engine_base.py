import time

from sous.config import SousConfig
from sous.engine.base import EngineManager, select_backend
from tests.fake_engine import FakeEngine


def test_select_backend_vision_config():
    assert select_backend({"vision_config": {}, "model_type": "qwen3_vl"}) == "vlm"

def test_select_backend_vl_model_type():
    assert select_backend({"model_type": "qwen2_5_vl"}) == "vlm"

def test_select_backend_text_only():
    assert select_backend({"model_type": "qwen3_moe"}) == "lm"


def _manager(idle_minutes: int = 30) -> tuple[EngineManager, list]:
    created: list[FakeEngine] = []

    def factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    cfg = SousConfig(idle_unload_minutes=idle_minutes)
    return EngineManager(cfg, engine_factory=factory), created


def test_get_is_lazy_and_cached():
    mgr, created = _manager()
    assert created == []          # nothing loaded yet
    e1 = mgr.get()
    e2 = mgr.get()
    assert e1 is e2 and len(created) == 1


def test_unload_if_idle():
    mgr, created = _manager(idle_minutes=0)
    mgr.get()
    mgr.touch()
    time.sleep(0.01)
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True
    assert mgr.status()["loaded"] is False


def test_no_unload_when_fresh():
    mgr, created = _manager(idle_minutes=30)
    mgr.get()
    mgr.touch()
    assert mgr.unload_if_idle() is False
    assert mgr.status()["loaded"] is True


def test_status_when_never_loaded():
    mgr, _ = _manager()
    s = mgr.status()
    assert s["loaded"] is False and s["model_id"]
