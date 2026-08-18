import threading
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
    assert created == []  # nothing loaded yet
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


class _BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__([])
        self.entered = threading.Event()
        self.release = threading.Event()
        self.in_flight = 0
        self.overlap = False

    def generate(self, messages, tools, max_tokens):
        self.in_flight += 1
        if self.in_flight > 1:
            self.overlap = True
        self.entered.set()
        self.release.wait(5)
        self.in_flight -= 1
        return "ok"


def test_generations_never_overlap_on_one_engine():
    """C3: on a stall the daemon thread is abandoned while still using the
    engine; the next task must WAIT rather than start a second concurrent
    generation on the same MLX model instance."""
    inner = _BlockingEngine()
    cfg = SousConfig(idle_unload_minutes=30)
    mgr = EngineManager(cfg, engine_factory=lambda mid: inner)
    engine = mgr.get()
    t1 = threading.Thread(target=engine.generate, args=([], [], 8), daemon=True)
    t1.start()
    assert inner.entered.wait(5)
    t2 = threading.Thread(target=engine.generate, args=([], [], 8), daemon=True)
    t2.start()
    time.sleep(0.2)
    assert inner.in_flight == 1  # the second generation is waiting, not running
    inner.release.set()
    t1.join(5)
    t2.join(5)
    assert not inner.overlap


def test_unload_refused_while_generation_in_flight():
    """C3: idle-unload racing an abandoned generation would free the model
    weights under it — unload_if_idle must refuse (False) while a generation
    is in flight, then proceed normally once it finishes."""
    inner = _BlockingEngine()
    cfg = SousConfig(idle_unload_minutes=0)
    mgr = EngineManager(cfg, engine_factory=lambda mid: inner)
    engine = mgr.get()
    t = threading.Thread(target=engine.generate, args=([], [], 8), daemon=True)
    t.start()
    assert inner.entered.wait(5)
    time.sleep(0.01)  # let the 0-minute idle threshold elapse
    assert mgr.unload_if_idle() is False
    assert inner.unloaded is False
    inner.release.set()
    t.join(5)
    time.sleep(0.01)
    assert mgr.unload_if_idle() is True  # generation done → unload proceeds


def test_release_mlx_thread_state_calls_clear_streams(monkeypatch):
    """The call-site tests monkeypatch the helper away, so only this pins that
    it really reaches mx.clear_streams — an API typo inside would otherwise be
    swallowed by its own except and every test would stay green while the
    native-crash safeguard is hollow."""
    import sys
    import types

    from sous.engine.base import release_mlx_thread_state

    calls = []
    fake_core = types.SimpleNamespace(clear_streams=lambda: calls.append(True))
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    release_mlx_thread_state()
    assert calls == [True]


def test_release_mlx_thread_state_never_raises(monkeypatch):
    """Cleanup runs in dying threads; an mlx quirk must never raise out."""
    import sys
    import types

    from sous.engine.base import release_mlx_thread_state

    def boom():
        raise RuntimeError("mlx teardown quirk")

    fake_core = types.SimpleNamespace(clear_streams=boom)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    release_mlx_thread_state()  # must not raise
