import sys
import threading
import time
import types

import pytest

from sous.config import SousConfig
from sous.engine.base import (
    EngineManager,
    GenerationStalled,
    ManagedEngine,
    select_backend,
)
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

    def generate(self, messages, tools, max_tokens, on_delta=None):
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


def test_managed_engine_forwards_reset_prompt_cache():
    inner = FakeEngine([])
    managed = ManagedEngine(inner)
    managed.reset_prompt_cache()
    assert inner.resets == 1


def test_managed_engine_forwards_prompt_cache_stats():
    inner = FakeEngine([])
    inner.stats = {"hits": 3}
    assert ManagedEngine(inner).prompt_cache_stats() == {"hits": 3}


def test_reset_prompt_cache_does_not_wait_for_the_generation_lock():
    """A stalled generation is abandoned while still holding _gen_lock. A reset
    that waited for it would wedge the next task, so it must be lock-free."""
    import threading
    import time

    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            started.set()
            release.wait(5)
            return "done"

    managed = ManagedEngine(BlockingEngine(["x"]))
    t = threading.Thread(target=lambda: managed.generate([], [], 8), daemon=True)
    t.start()
    assert started.wait(5)
    assert managed.generation_in_flight()
    t0 = time.monotonic()
    managed.reset_prompt_cache()  # must not block behind the lock
    assert time.monotonic() - t0 < 1.0
    release.set()
    t.join(5)


# ---- GenerationSession (issue #34) ----------------------------------------


def _msgs() -> list[dict]:
    return [{"role": "user", "content": "x"}]


def test_session_runs_all_generations_on_one_fresh_thread():
    inner = FakeEngine(["a", "b"])
    session = ManagedEngine(inner).session()
    assert session.generate(_msgs(), [], 8, timeout=5) == "a"
    assert session.generate(_msgs(), [], 8, timeout=5) == "b"
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()
    assert inner.generate_threads[0] is inner.generate_threads[1] is session._thread
    assert session._thread is not threading.current_thread()


def test_session_releases_mlx_state_once_on_its_own_thread(monkeypatch):
    import sous.engine.base as base

    released_in: list[int] = []
    monkeypatch.setattr(
        base, "release_mlx_thread_state", lambda: released_in.append(threading.get_ident())
    )
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    assert session.generate(_msgs(), [], 8, timeout=5) == "a"
    session.close()
    session._thread.join(5)
    assert released_in == [session._thread.ident]


def test_session_relays_exceptions_and_survives_them():
    class Flaky(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            out = super().generate(messages, tools, max_tokens)
            if out == "boom":
                raise ValueError("boom")
            return out

    inner = Flaky(["boom", "ok"])
    session = ManagedEngine(inner).session()
    with pytest.raises(ValueError, match="boom"):
        session.generate(_msgs(), [], 8, timeout=5)
    # The same session, the same thread: an engine error must not kill the loop.
    assert session.generate(_msgs(), [], 8, timeout=5) == "ok"
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()


def test_session_close_without_any_generation():
    session = ManagedEngine(FakeEngine([])).session()
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()


def test_stalled_generation_is_abandoned_and_its_late_result_dropped():
    gate = threading.Event()

    class Gated(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            gate.wait(10)
            return super().generate(messages, tools, max_tokens)

    inner = Gated(["late"])
    session = ManagedEngine(inner).session()
    with pytest.raises(GenerationStalled):
        session.generate(_msgs(), [], 8, timeout=0.05)
    assert session._abandoned.is_set()
    gate.set()  # ordering pin: the generation completes only after abandonment
    session._thread.join(5)
    assert not session._thread.is_alive()
    assert session._replies.empty()  # the late result was dropped, not queued


def test_abandoned_waiter_on_the_lock_never_generates():
    """Issue #34, consideration 7: a generation abandoned while QUEUED on
    _gen_lock must exit when the lock frees, never run under the next task's
    identity. Fails if the session checks _abandoned before taking the lock
    instead of after."""
    entered = threading.Event()
    release = threading.Event()

    class Wedged(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            entered.set()
            release.wait(10)
            return super().generate(messages, tools, max_tokens)

    inner = Wedged(["a"])
    managed = ManagedEngine(inner)
    session_a = managed.session()
    session_b = managed.session()
    a_result: list[str] = []
    threading.Thread(
        target=lambda: a_result.append(session_a.generate(_msgs(), [], 8, timeout=10)),
        daemon=True,
    ).start()
    assert entered.wait(5)  # A is wedged inside generate, holding _gen_lock
    with pytest.raises(GenerationStalled):
        session_b.generate(_msgs(), [], 8, timeout=0.05)  # B abandoned on the lock
    release.set()
    session_b._thread.join(5)
    assert not session_b._thread.is_alive()
    session_a.close()
    session_a._thread.join(5)
    assert a_result == ["a"]
    assert len(inner.calls) == 1  # B's request never reached the engine


def test_close_tolerates_an_undequeued_stalled_request():
    """A starved session thread may never dequeue a timed-out request, so the
    request still fills the maxsize-1 queue when run_task's finally calls
    close(). close() must not raise queue.Full there — the thread dequeues
    that request eventually, sees _abandoned under the lock, and exits."""
    gate = threading.Event()

    class Gated(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            gate.wait(10)
            return super().generate(messages, tools, max_tokens)

    inner = Gated(["a"])
    session = ManagedEngine(inner).session()
    # Occupy the thread inside generate, then fill the queue behind its back —
    # the exact state a stalled, never-dequeued request leaves behind.
    session._requests.put_nowait((_msgs(), [], 8, None))
    for _ in range(1000):
        if session._requests.empty():
            break
        time.sleep(0.005)
    else:
        pytest.fail("session thread never dequeued the first request")
    session._requests.put_nowait((_msgs(), [], 8, None))  # the undequeued stalled request
    session._abandoned.set()  # what generate() does when it times out
    session.close()  # must not raise queue.Full
    gate.set()
    session._thread.join(5)
    assert not session._thread.is_alive()
    assert len(inner.calls) == 1  # the undequeued request never generated


def test_close_unleaks_an_idle_thread_holding_an_unconsumed_reply():
    """The reply-vs-timeout race can abandon a session whose thread already
    queued its reply and parked. CLOSE must wake it so it exits and releases —
    otherwise an ("err", e) reply would pin the KV cache through its traceback
    for the daemon's lifetime."""
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    # Drive the loop directly: a reply lands, but no caller consumes it.
    session._requests.put_nowait((_msgs(), [], 8, None))
    for _ in range(1000):
        if not session._replies.empty():
            break
        time.sleep(0.005)
    else:
        pytest.fail("session thread never produced the reply")
    session._abandoned.set()  # what generate() does when it times out
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()


def test_default_factory_passes_drafter_settings_to_vlm_only(monkeypatch):
    """The drafter is an mlx-vlm feature: the VLM backend must receive the
    configured drafter id and block size, and the LM backend must not be
    handed arguments it has no parameter for."""
    import sous.engine.base as base
    import sous.engine.lm as lm_mod
    import sous.engine.vlm as vlm_mod

    captured: dict[str, dict] = {}

    class FakeVLM:
        def __init__(self, model_id, **kwargs):
            captured["vlm"] = kwargs

    class FakeLM:
        def __init__(self, model_id, **kwargs):
            captured["lm"] = kwargs

    monkeypatch.setattr(vlm_mod, "VLMEngine", FakeVLM)
    monkeypatch.setattr(lm_mod, "LMEngine", FakeLM)

    monkeypatch.setattr(base, "fetch_model_config", lambda _id: {"vision_config": {}})
    base._default_factory("m", 0.7, 0.8, 20, True, draft_id="z-lab/drafter", draft_block_size=3)
    assert captured["vlm"]["draft_id"] == "z-lab/drafter"
    assert captured["vlm"]["draft_block_size"] == 3

    monkeypatch.setattr(base, "fetch_model_config", lambda _id: {"model_type": "qwen3"})
    base._default_factory("m", 0.7, 0.8, 20, True, draft_id="z-lab/drafter", draft_block_size=3)
    assert "draft_id" not in captured["lm"]
    assert "draft_block_size" not in captured["lm"]


def test_engine_manager_threads_drafter_config_into_default_factory(monkeypatch):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    cfg = SousConfig(speculative_draft_id="z-lab/drafter", speculative_block_size=5)
    EngineManager(cfg).get()
    assert seen["kwargs"]["draft_id"] == "z-lab/drafter"
    assert seen["kwargs"]["draft_block_size"] == 5


# ---- streaming deltas (gateway) ---------------------------------------------


def test_session_relays_deltas_on_the_session_thread():
    """Deltas are emitted from inside the engine's decode loop, i.e. on the
    session thread — the consumer bridges them to wherever it lives."""
    from sous.engine.base import Delta

    seen: list[tuple[Delta, threading.Thread]] = []
    inner = FakeEngine(["hello world"])
    session = ManagedEngine(inner).session()
    text = session.generate(
        _msgs(), [], 8, timeout=5, on_delta=lambda d: seen.append((d, threading.current_thread()))
    )
    session.close()
    session._thread.join(5)
    assert text == "hello world"
    assert [d for d, _ in seen] == [Delta("hello world", 2, "stop")]
    assert seen[0][1] is session._thread


def test_managed_engine_forwards_on_delta():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    managed = ManagedEngine(FakeEngine(["x y z"]))
    assert managed.generate(_msgs(), [], 8, on_delta=seen.append) == "x y z"
    assert seen == [Delta("x y z", 3, "stop")]


def test_chunked_fake_engine_streams_pieces_with_cumulative_counts():
    from sous.engine.base import Delta
    from tests.fake_engine import ChunkedFakeEngine

    seen: list[Delta] = []
    e = ChunkedFakeEngine(["a|b|c"])
    assert e.generate(_msgs(), [], 8, on_delta=seen.append) == "abc"
    assert seen == [Delta("a", 1, None), Delta("b", 2, None), Delta("c", 3, "stop")]
    assert e.finished.is_set()


# --- tokenization is serialized ---------------------------------------------


class _RecordingTokenizer:
    """Notes every time two threads are inside a tokenizer call at once."""

    bos_token = None
    chat_template = None

    def __init__(self) -> None:
        self._inside = 0
        self.overlaps: list[str] = []

    def _busy(self, what: str) -> None:
        self._inside += 1
        if self._inside > 1:
            self.overlaps.append(what)
        time.sleep(0.005)  # wide enough for every other thread to pile in
        self._inside -= 1

    def apply_chat_template(self, messages, **kwargs) -> str:
        self._busy("apply_chat_template")
        return str(messages)

    def encode(self, text, add_special_tokens=True) -> list[int]:
        self._busy("encode")
        return [len(text)]


def _stub(monkeypatch, name: str, **attrs) -> None:
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(__name__=name, **attrs))


def _hammer_ids(engine, tokenizer: _RecordingTokenizer) -> None:
    """Tokenize from several threads at once, each with its own text so the
    PromptMemo never short-circuits the encode."""

    def one(n: int) -> None:
        engine._ids("full", [{"role": "user", "content": f"message {n}"}], [])

    threads = [threading.Thread(target=one, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads)
    assert tokenizer.overlaps == [], tokenizer.overlaps


def test_lm_tokenization_is_serialized(monkeypatch):
    """ManagedEngine.count_tokens deliberately skips _gen_lock, and the gateway
    made that a second caller: a turn tokenizes on a pool thread while Claude
    Code's count_tokens arrives mid-turn. HF's fast tokenizer mutates shared
    Rust state on every encode, so the two must not overlap."""
    from sous.engine.lm import LMEngine

    tokenizer = _RecordingTokenizer()
    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (object(), tokenizer))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    _hammer_ids(LMEngine("test/model"), tokenizer)


def test_vlm_tokenization_is_serialized(monkeypatch):
    """Same contract on the backend the gateway actually runs."""
    from sous.engine.vlm import VLMEngine

    tokenizer = _RecordingTokenizer()
    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="fake"))
    processor = types.SimpleNamespace(tokenizer=tokenizer)
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    _stub(monkeypatch, "mlx_vlm.utils", should_add_special_tokens=lambda model_type, proc: True)
    _hammer_ids(VLMEngine("test/model"), tokenizer)
