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


def _cfg(tmp_path, **overrides) -> SousConfig:
    return SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", **overrides)


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


def test_a_lease_holds_off_the_idle_unload():
    """A gateway turn holds the engine across count_tokens and generate, and
    only the latter takes _gen_lock. Without a lease the unload sweep would
    free the weights under the tokenizer pass."""
    mgr, created = _manager(idle_minutes=0)
    mgr.get()
    time.sleep(0.01)
    assert mgr.unload_if_idle() is True  # baseline: idle 0 unloads at once
    mgr.get()
    time.sleep(0.01)
    with mgr.lease():
        assert mgr.unload_if_idle() is False
        assert mgr.status()["loaded"] is True
        assert created[1].unloaded is False
    assert mgr.unload_if_idle() is True  # lease gone → the sweep proceeds
    assert created[1].unloaded is True


def test_status_when_never_loaded():
    mgr, _ = _manager()
    s = mgr.status()
    assert s["loaded"] is False and s["model_id"]


def test_get_logs_the_load_once_with_its_duration(caplog):
    import logging

    mgr, created = _manager()
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and len(created) == 1
    assert lines[0].startswith("model_load seconds=") and lines[0].endswith(" model=fake/model")


def _positional_factory(model_id: str) -> FakeEngine:
    """An engine that reports which side owns the rotary positions, the way
    the VLM backend does."""
    engine = FakeEngine([])
    engine.positions = "engine"  # ty: ignore[unresolved-attribute]
    return engine


def test_get_logs_the_positions_the_engine_reports(caplog):
    import logging

    mgr = EngineManager(SousConfig(), engine_factory=_positional_factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and lines[0].endswith(" model=fake/model positions=engine")


def test_get_logs_no_positions_for_an_engine_without_them(caplog):
    """The LM backend (and a plain FakeEngine, which stands in for it here)
    has no such attribute, so the load line must not grow one."""
    import logging

    mgr, _ = _manager()
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and "positions=" not in lines[0]


def test_get_loads_on_a_thread_of_its_own_that_releases_its_mlx_state(monkeypatch):
    """The caller's thread never touches mlx: the factory runs on a loader
    thread that releases its streams before it exits. A gateway pool thread
    outlives its turn and releases unconditionally, and a load on it left it
    unable to touch mlx again — every cold start after the first on that
    thread failed."""
    from sous.engine import base

    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        base,
        "release_mlx_thread_state",
        lambda: seen.append(("release", threading.get_ident())),
    )

    def factory(model_id: str) -> FakeEngine:
        seen.append((threading.current_thread().name, threading.get_ident()))
        return FakeEngine([])

    EngineManager(SousConfig(), engine_factory=factory).get()
    assert [name for name, _ in seen] == ["sous-model-load", "release"]
    idents = {ident for _, ident in seen}
    assert len(idents) == 1 and threading.get_ident() not in idents


def test_a_thread_that_released_between_two_cold_loads_can_load_again():
    """The gateway pool thread's life, on real mlx: load, release, idle
    unload, load again. Before the load had a thread of its own the second
    load raised "There is no Stream(gpu, 0) in current thread"."""
    mx = pytest.importorskip("mlx.core")
    from sous.engine.base import release_mlx_thread_state

    def factory(model_id: str) -> FakeEngine:
        mx.eval(mx.arange(4))  # the least a real load does on its thread
        return FakeEngine([])

    mgr = EngineManager(SousConfig(idle_unload_minutes=0), engine_factory=factory)
    outcome: list = []

    def pool_thread() -> None:
        try:
            for _ in range(2):
                mgr.get()
                release_mlx_thread_state()  # what TurnRunner.run's finally does
                time.sleep(0.01)
                assert mgr.unload_if_idle() is True
            outcome.append("ok")
        except Exception as exc:  # noqa: BLE001 — the failure is the assertion
            outcome.append(exc)

    thread = threading.Thread(target=pool_thread, name="sous-gateway-turn_0")
    thread.start()
    thread.join(10)
    assert outcome == ["ok"]


class _GatedFactory:
    """A model factory that blocks until released, so a test can look at the
    manager mid-load. `started` is set once the factory has been entered."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.created: list[FakeEngine] = []

    def __call__(self, model_id: str) -> FakeEngine:
        self.started.set()
        assert self.release.wait(10)
        engine = FakeEngine([])
        self.created.append(engine)
        return engine


def _gated_manager(idle_minutes: int = 30) -> tuple[EngineManager, _GatedFactory]:
    """Holders here are made-up pids that only the load tests use, so every
    one counts as alive: the real psutil check would prune them on the next
    hold() or status()."""
    factory = _GatedFactory()
    return EngineManager(
        SousConfig(idle_unload_minutes=idle_minutes),
        engine_factory=factory,
        holder_alive=lambda pid, create_time: True,
    ), factory


def test_status_answers_while_the_model_loads():
    """The worker's get() used to hold the manager lock for the whole load —
    minutes for a 27B — so status() (and everything else on the lock) waited
    it out. A load in progress must be reportable, not a blocking call."""
    mgr, factory = _gated_manager()
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        t0 = time.monotonic()
        s = mgr.status()
        assert time.monotonic() - t0 < 1.0
        assert s["loaded"] is False and s["loading"] is True
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()
    s = mgr.status()
    assert s["loaded"] is True and s["loading"] is False


def test_concurrent_gets_share_one_load():
    mgr, factory = _gated_manager()
    engines: list = []
    threads = [
        threading.Thread(target=lambda: engines.append(mgr.get()), daemon=True) for _ in range(3)
    ]
    for t in threads:
        t.start()
    try:
        assert factory.started.wait(5)
    finally:
        factory.release.set()
    for t in threads:
        t.join(5)
        assert not t.is_alive()
    assert len(factory.created) == 1
    assert all(e is engines[0] for e in engines)


def test_a_failed_load_leaves_the_manager_empty_and_loadable():
    calls: list[str] = []

    def factory(model_id: str):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("weights missing")
        return FakeEngine([])

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    with pytest.raises(RuntimeError):
        mgr.get()
    s = mgr.status()
    assert s["loaded"] is False and s["loading"] is False
    assert mgr.get() is mgr.get() and len(calls) == 2


def test_a_failed_load_releases_the_threads_waiting_on_it():
    """A waiter parks until the load in progress ends; a load that ends by
    raising must wake it too, or the daemon is wedged behind one bad load.
    Each waiter then finds nothing loaded and loads on its own."""
    entered = threading.Event()
    proceed = threading.Event()
    calls: list[str] = []

    def factory(model_id: str):
        calls.append(model_id)
        if len(calls) == 1:
            entered.set()
            assert proceed.wait(10)
            raise RuntimeError("weights missing")
        return FakeEngine([])

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    failed = threading.Event()

    def load_and_fail():
        try:
            mgr.get()
        except RuntimeError:
            failed.set()

    loader = threading.Thread(target=load_and_fail, daemon=True)
    loader.start()
    assert entered.wait(5)
    got: list = []
    waiter = threading.Thread(target=lambda: got.append(mgr.get()), daemon=True)
    waiter.start()
    proceed.set()
    loader.join(5)
    assert failed.is_set()
    waiter.join(5)
    assert not waiter.is_alive(), "a failed load left its waiter parked"
    assert len(calls) == 2 and got[0] is mgr.get()


def test_unload_if_idle_is_refused_at_once_during_a_load():
    """Refused, and refused without waiting: the sweep runs on the worker's
    thread every poll, and parking it behind a load for minutes is the
    stall the lock change removes."""
    mgr, factory = _gated_manager(idle_minutes=0)
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        t0 = time.monotonic()
        assert mgr.unload_if_idle() is False
        assert time.monotonic() - t0 < 1.0
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()


class _SlowUnloadEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__([])
        self.unloading = threading.Event()
        self.release = threading.Event()

    def unload(self) -> None:
        self.unloading.set()
        assert self.release.wait(10)
        super().unload()


def test_get_during_an_unload_waits_for_it_and_loads_fresh():
    """Freeing a 27B takes seconds and get() must never hand out an engine
    that is being torn down, nor load a second copy beside it: it waits, then
    loads. status() meanwhile reports neither loaded nor loading."""
    slow = _SlowUnloadEngine()
    second_started = threading.Event()
    made: list[FakeEngine] = []

    def factory(model_id: str):
        if made:
            second_started.set()
        made.append(slow if not made else FakeEngine([]))
        return made[-1]

    mgr = EngineManager(SousConfig(idle_unload_minutes=0), engine_factory=factory)
    mgr.get()
    time.sleep(0.01)
    sweeper = threading.Thread(target=mgr.unload_if_idle, daemon=True)
    sweeper.start()
    assert slow.unloading.wait(5)
    got: list = []
    getter = threading.Thread(target=lambda: got.append(mgr.get()), daemon=True)
    getter.start()
    # Parked behind the unload, not loading a second copy beside it.
    assert not second_started.wait(0.2)
    assert got == []
    s = mgr.status()
    assert s["loading"] is False and s["loaded"] is False
    slow.release.set()
    sweeper.join(5)
    getter.join(5)
    assert not getter.is_alive()
    assert len(made) == 2 and got[0] is not slow and slow.unloaded is True


def test_a_failed_unload_does_not_park_the_next_load():
    """unload() runs outside the lock behind the _unloading flag; if it
    raises, the flag must still clear, or every later get() waits forever
    with nothing in the log but the worker's "continuing" line."""
    made: list[FakeEngine] = []

    class _RaisingUnload(FakeEngine):
        def unload(self) -> None:
            raise RuntimeError("teardown failed")

    def factory(model_id: str):
        made.append(_RaisingUnload([]) if not made else FakeEngine([]))
        return made[-1]

    mgr = EngineManager(SousConfig(idle_unload_minutes=0), engine_factory=factory)
    mgr.get()
    time.sleep(0.01)
    with pytest.raises(RuntimeError):
        mgr.unload_if_idle()
    s = mgr.status()
    assert s["loaded"] is False and s["loading"] is False
    got: list = []
    getter = threading.Thread(target=lambda: got.append(mgr.get()), daemon=True)
    getter.start()
    getter.join(5)
    assert not getter.is_alive(), "a failed unload left the next load parked"
    # `made` records raw engines and a load wraps its engine once, so the
    # identity check goes through the wrapper the second load made.
    assert len(made) == 2 and got[0]._inner is made[1]


class _Liveness:
    """A stand-in for the psutil check: which (pid, create_time) pairs are
    live processes right now."""

    def __init__(self) -> None:
        self.live: set[tuple[int, float]] = set()

    def __call__(self, pid: int, create_time: float) -> bool:
        return (pid, create_time) in self.live


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _held_manager(idle_minutes: int = 30, factory=None):
    created: list[FakeEngine] = []

    def default_factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    live, clock = _Liveness(), _Clock()
    mgr = EngineManager(
        SousConfig(idle_unload_minutes=idle_minutes),
        engine_factory=factory or default_factory,
        holder_alive=live,
        clock=clock,
    )
    return mgr, created, live, clock


def _join_preload(mgr: EngineManager) -> None:
    """Wait for a preload the test just triggered. The manager forgets the
    thread as its last act, so a missing thread means a finished preload —
    which the status must then agree with."""
    thread = mgr._preload
    if thread is None:
        assert mgr.status()["loading"] is False
        return
    thread.join(5)
    assert not thread.is_alive()


def test_hold_starts_exactly_one_preload_whose_load_releases_its_mlx_state(monkeypatch):
    """The release is monkeypatched, so this pins that it happens once, on
    the loader thread get() spawns — the preload thread itself touches no
    mlx — not what the real call does to a loaded model; the real-daemon run
    is what exercises that."""
    from sous.engine import base

    released_in: list[str] = []
    monkeypatch.setattr(
        base,
        "release_mlx_thread_state",
        lambda: released_in.append(threading.current_thread().name),
    )
    mgr, factory = _gated_manager()
    first = mgr.hold(101, 5.0)
    assert first == {"loaded": False, "loading": True, "holders": 1}
    assert factory.started.wait(5)
    preload = mgr._preload
    assert preload is not None and preload.daemon and preload.name == "sous-preload"
    second = mgr.hold(102, 6.0)
    assert second == {"loaded": False, "loading": True, "holders": 2}
    assert mgr._preload is preload  # no second thread while the first loads
    s = mgr.status()
    assert s["loaded"] is False and s["loading"] is True and s["holders"] == 2
    factory.release.set()
    preload.join(5)
    assert not preload.is_alive()
    assert len(factory.created) == 1
    assert released_in == ["sous-model-load"]
    s = mgr.status()
    assert s["loaded"] is True and s["loading"] is False and s["holders"] == 2
    assert mgr.hold(103, 7.0) == {"loaded": True, "loading": False, "holders": 3}


def test_a_hold_during_another_threads_load_starts_no_preload():
    """The worker or a gateway turn may already be loading; a hold then only
    registers itself and reports the load in progress."""
    mgr, factory = _gated_manager()
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        assert mgr.hold(51, 1.0) == {"loaded": False, "loading": True, "holders": 1}
        assert mgr._preload is None
        assert mgr.status()["loading"] is True
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()
    assert len(factory.created) == 1  # one load, not two


def test_a_hold_on_an_unloaded_model_reports_the_load_it_starts():
    """The launcher prints one of two lines from these flags; a hold on an
    unloaded model starts or joins a load, so short of the OS refusing a
    thread there is no third state."""
    mgr, created, live, _ = _held_manager()
    answer = mgr.hold(81, 1.0)
    _join_preload(mgr)
    assert answer == {"loaded": False, "loading": True, "holders": 1}


def test_a_hold_whose_preload_thread_cannot_start_reports_neither_and_never_raises(
    monkeypatch, caplog
):
    """The one third state: the OS refused a thread. hold() keeps its
    contract (a warning, never a raise), forgets the thread it could not
    start so the next hold tries again, and status() does not report a load
    that is not happening."""
    import logging

    mgr, created, live, _ = _held_manager()
    live.live.update({(91, 1.0), (92, 2.0)})
    refusals = 0

    def refuse(self):
        nonlocal refusals
        refusals += 1
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with caplog.at_level(logging.WARNING, logger="sous.engine"):
        answer = mgr.hold(91, 1.0)
    assert answer == {"loaded": False, "loading": False, "holders": 1}
    assert mgr._preload is None and mgr.status()["loading"] is False
    assert "preload thread could not start (RuntimeError)" in [
        r.getMessage() for r in caplog.records
    ]
    monkeypatch.undo()
    assert mgr.hold(92, 2.0) == {"loaded": False, "loading": True, "holders": 2}
    _join_preload(mgr)
    assert refusals == 1 and len(created) == 1 and mgr.status()["loaded"] is True


def test_loading_never_reads_true_beside_a_loaded_model():
    """The preload thread forgets itself only after releasing its mlx state,
    which is after get() has published the engine: in that window the thread
    alone must not say a load is in progress."""
    mgr, created, live, _ = _held_manager()
    mgr.get()
    mgr._preload = threading.current_thread()  # the window, held open
    try:
        assert mgr.status()["loading"] is False
        assert mgr.hold(93, 1.0)["loading"] is False
    finally:
        mgr._preload = None


def test_a_failed_preload_is_logged_and_never_raised(caplog):
    import logging

    def factory(model_id: str):
        raise RuntimeError("weights missing")

    mgr, _, live, _ = _held_manager(factory=factory)
    live.live.add((7, 1.0))
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        assert mgr.hold(7, 1.0)["loading"] is True
        _join_preload(mgr)
    messages = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert "hold pid=7 (holders=1)" in messages
    assert f"preloading {mgr._config.model_id}" in messages
    assert "preload failed (RuntimeError)" in messages
    assert "weights missing" not in "".join(messages)
    assert mgr.status()["loading"] is False and mgr.status()["holders"] == 1


def test_the_sweep_keeps_the_model_while_any_holder_lives(caplog):
    import logging

    mgr, created, live, clock = _held_manager(idle_minutes=30)
    live.live.update({(11, 1.0), (12, 2.0)})
    mgr.hold(11, 1.0)
    mgr.hold(12, 2.0)
    _join_preload(mgr)
    clock.now += 3600  # an hour idle: past the threshold, but held
    assert mgr.unload_if_idle() is False
    live.live.discard((11, 1.0))  # one holder exits
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        assert mgr.unload_if_idle() is False
    assert "hold released pid=11 (holders=1)" in [r.getMessage() for r in caplog.records]
    assert mgr.status()["holders"] == 1 and created[0].unloaded is False


def test_a_reused_pid_is_a_different_holder():
    """PID reuse: the same number with a different start time is not the
    process that asked for the hold."""
    mgr, created, live, clock = _held_manager(idle_minutes=0)
    live.live.add((21, 100.0))
    mgr.hold(21, 100.0)
    _join_preload(mgr)
    clock.now += 1
    assert mgr.unload_if_idle() is False  # held
    live.live = {(21, 100.0 + 5.0)}  # a new process wearing the old pid
    assert mgr.unload_if_idle() is False  # pruned: the idle clock restarts on this sweep
    assert mgr.status()["holders"] == 0
    clock.now += 1
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True


def test_the_idle_clock_restarts_when_the_last_holder_leaves():
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    live.live.add((31, 1.0))
    mgr.hold(31, 1.0)
    _join_preload(mgr)
    clock.now += 7200  # two hours held
    live.live.clear()
    assert mgr.unload_if_idle() is False  # released now: idle clock restarts here
    assert mgr.status()["holders"] == 0
    clock.now += 30 * 60  # exactly the threshold
    assert mgr.unload_if_idle() is False
    clock.now += 1
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True


def test_a_hold_during_an_unload_reloads_after_it():
    slow = _SlowUnloadEngine()
    made: list[FakeEngine] = []

    def factory(model_id: str):
        made.append(slow if not made else FakeEngine([]))
        return made[-1]

    live, clock = _Liveness(), _Clock()
    mgr = EngineManager(
        SousConfig(idle_unload_minutes=0), engine_factory=factory, holder_alive=live, clock=clock
    )
    mgr.get()
    clock.now += 1
    sweeper = threading.Thread(target=mgr.unload_if_idle, daemon=True)
    sweeper.start()
    assert slow.unloading.wait(5)
    live.live.add((41, 1.0))
    assert mgr.hold(41, 1.0) == {"loaded": False, "loading": True, "holders": 1}
    slow.release.set()
    sweeper.join(5)
    _join_preload(mgr)
    assert len(made) == 2 and mgr.status()["loaded"] is True


def test_holder_alive_is_this_process_under_its_real_start_time_only():
    import os

    import psutil

    from sous.engine.base import _holder_alive

    start = psutil.Process().create_time()
    assert _holder_alive(os.getpid(), start) is True
    assert _holder_alive(os.getpid(), start - 3600.0) is False
    assert _holder_alive(2**30, start) is False  # beyond every platform's pid range
    assert _holder_alive(-1, start) is False  # psutil.Process(-1) raises ValueError
    assert _holder_alive("abc", start) is False  # ty: ignore[invalid-argument-type]


def test_a_holder_we_cannot_read_counts_as_gone(monkeypatch):
    """psutil raises AccessDenied for another user's process; the daemon and
    its holders share a user, so that is never ours and must not pin the
    model. Any other psutil error is the same answer."""
    import psutil

    from sous.engine.base import _holder_alive

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    assert _holder_alive(1234, 1.0) is False


def test_holders_are_pruned_even_with_nothing_loaded():
    """A load that failed leaves no engine; the registry must still forget a
    holder whose process is gone, or /sous/status reports sessions that
    ended hours ago."""

    def factory(model_id: str):
        raise RuntimeError("weights missing")

    mgr, _, live, _ = _held_manager(factory=factory)
    live.live.add((71, 1.0))
    mgr.hold(71, 1.0)
    _join_preload(mgr)
    assert mgr.status()["holders"] == 1
    live.live.clear()
    assert mgr.unload_if_idle() is False  # nothing loaded, but the sweep still prunes
    assert mgr.status()["holders"] == 0


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


def test_managed_engine_forwards_owner_scoped_reset_and_stats():
    inner = FakeEngine([])
    managed = ManagedEngine(inner)
    me = threading.current_thread()
    managed.reset_prompt_cache(owner=me)
    managed.prompt_cache_stats(owner=me)
    assert inner.reset_owners == [me]
    assert inner.stats_owners == [me]


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


def test_session_exposes_its_thread():
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    session.generate([], [], 8, timeout=5)
    assert inner.generate_threads == [session.thread]
    session.close()
    session.join(5)


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

    # Neither fake model config carries KV-cost fields, so cache_budget's
    # unspecified-auto default falls back to a single slot with a warning —
    # unrelated to what this test checks, but real _default_factory behavior.
    monkeypatch.setattr(base, "fetch_model_config", lambda _id: {"vision_config": {}})
    with pytest.warns(UserWarning, match="KV cost"):
        base._default_factory("m", 0.7, 0.8, 20, True, draft_id="z-lab/drafter", draft_block_size=3)
    assert captured["vlm"]["draft_id"] == "z-lab/drafter"
    assert captured["vlm"]["draft_block_size"] == 3

    monkeypatch.setattr(base, "fetch_model_config", lambda _id: {"model_type": "qwen3"})
    with pytest.warns(UserWarning, match="KV cost"):
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


# ---- int8 prefill plumbing --------------------------------------------------------


def test_engine_manager_threads_int8_prefill_into_default_factory(monkeypatch):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(SousConfig(int8_prefill=True)).get()
    assert seen["kwargs"]["int8_prefill"] is True


def test_default_factory_passes_int8_prefill_to_both_engines(monkeypatch):
    from sous.engine import base, lm, vlm

    seen: dict[str, dict] = {}

    class RecordingVLM:
        def __init__(self, model_id, **kwargs):
            seen["vlm"] = kwargs

    class RecordingLM:
        def __init__(self, model_id, **kwargs):
            seen["lm"] = kwargs

    monkeypatch.setattr(vlm, "VLMEngine", RecordingVLM)
    monkeypatch.setattr(lm, "LMEngine", RecordingLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"vision_config": {}})
    monkeypatch.setattr("sous.context.kv_bytes_per_token", lambda cfg: 1024)
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, int8_prefill=True)
    assert seen["vlm"]["int8_prefill"] is True
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3_5"})
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, int8_prefill=True)
    assert seen["lm"]["int8_prefill"] is True


def test_status_carries_the_int8_prefill_view_when_the_engine_reports_one(tmp_path):
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "int8_prefill" not in manager.status(), "fakes without the attribute stay silent"
    inner.int8_prefill_status = {  # ty: ignore[unresolved-attribute]
        "state": "active",
        "reason": None,
        "routed": 336,
    }
    assert manager.status()["int8_prefill"] == {"state": "active", "reason": None, "routed": 336}


def test_status_carries_the_positions_view_when_the_engine_reports_one(tmp_path):
    """The load line says which side owns the rotary positions once; the
    status document says it for as long as the model is resident, so an
    mlx-vlm bump that flips the probe is readable without a reload."""
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "positions" not in manager.status(), "fakes without the attribute stay silent"
    inner.positions = "engine"  # ty: ignore[unresolved-attribute]
    assert manager.status()["positions"] == "engine"


def _positionless_model() -> types.SimpleNamespace:
    """A stub model whose text-only embedding helper returns no positions of
    its own, in the shape every mlx-vlm helper returns."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="fake"),
        get_input_embeddings=lambda *a, **kw: types.SimpleNamespace(position_ids=None),
    )


def test_vlm_engine_finds_the_positions_its_helper_returns(monkeypatch):
    """A Qwen-lineage model's text-only embedding helper returns rotary
    positions of its own; the probe in __init__ must land on `engine`, once,
    asking the helper the way generate_step does (ids, no pixels, mask=None)."""
    from sous.engine.vlm import VLMEngine

    calls: list[tuple[tuple, dict]] = []

    def helper(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(position_ids=object())

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="fake"), get_input_embeddings=helper
    )
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    engine = VLMEngine("test/model", cache_budget=0)
    assert engine.positions == "engine" and engine._positional is True
    assert len(calls) == 1
    (ids, pixels), kwargs = calls[0]
    assert ids.shape == (1, 1) and pixels is None and kwargs == {"mask": None}


def test_vlm_engine_leaves_positions_to_a_model_whose_helper_returns_none(monkeypatch):
    from sous.engine.vlm import VLMEngine

    _stub(
        monkeypatch,
        "mlx_vlm",
        load=lambda model_id: (_positionless_model(), _RecordingTokenizer()),
    )
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    engine = VLMEngine("test/model", cache_budget=0)
    assert engine.positions == "model" and engine._positional is False


def test_vlm_engine_enables_int8_prefill_on_the_loaded_model(monkeypatch):
    from sous.engine import int8prefill
    from sous.engine.vlm import VLMEngine

    model = _positionless_model()
    processor = types.SimpleNamespace(tokenizer=_RecordingTokenizer())
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    seen: dict[str, object] = {}

    def fake_enable(m, *, enabled):
        seen["model"], seen["enabled"] = m, enabled
        return {"state": "off", "reason": None, "routed": 0}

    monkeypatch.setattr(int8prefill, "enable", fake_enable)
    engine = VLMEngine("test/model", cache_budget=0, int8_prefill=True)
    assert seen == {"model": model, "enabled": True}
    assert engine.int8_prefill_status == {"state": "off", "reason": None, "routed": 0}


def test_lm_engine_enables_int8_prefill_on_the_loaded_model(monkeypatch):
    from sous.engine import int8prefill
    from sous.engine.lm import LMEngine

    model = object()
    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (model, _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    seen: dict[str, object] = {}

    def fake_enable(m, *, enabled):
        seen["model"], seen["enabled"] = m, enabled
        return {"state": "unavailable", "reason": "test", "routed": 0}

    monkeypatch.setattr(int8prefill, "enable", fake_enable)
    engine = LMEngine("test/model", cache_budget=0, int8_prefill=False)
    assert seen == {"model": model, "enabled": False}
    assert engine.int8_prefill_status["state"] == "unavailable"


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
    model = _positionless_model()
    processor = types.SimpleNamespace(tokenizer=tokenizer)
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    _stub(monkeypatch, "mlx_vlm.utils", should_add_special_tokens=lambda model_type, proc: True)
    _hammer_ids(VLMEngine("test/model"), tokenizer)


# ---- prompt-cache budget plumbing (Phase 3a) --------------------------------


def test_status_carries_the_prompt_cache_view_once_loaded(tmp_path):
    inner = FakeEngine([])
    inner.stats = {"hits": 1, "slots": 2, "resident_bytes": 3}
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    assert "prompt_cache" not in manager.status()
    manager.get()
    assert manager.status()["prompt_cache"] == {"hits": 1, "slots": 2, "resident_bytes": 3}


def test_status_leaves_out_the_per_turn_gauges(tmp_path):
    """server_status hands this block to the frontier model; daemon-wide, a
    per-turn gauge is a max over every owner ever seen — tokens for nothing."""
    inner = FakeEngine([])
    inner.stats = {"hits": 1, "prefill_seconds": 1.4974267615067218, "took_len": 900}
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert manager.status()["prompt_cache"] == {"hits": 1}


def test_default_factory_threads_the_cache_budget_and_reserve(monkeypatch):
    """The reserve is one full window of KV at the model's per-token cost, for
    the larger of the worker's and the gateway's windows."""
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(
        base,
        "fetch_model_config",
        lambda mid: {
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    )
    base._default_factory("m", prompt_cache=True, cache_budget=7, reserve_tokens=1000)
    # 2 (K,V) x 2 layers x 1 head x 8 dim x 2 bytes = 64 B/token
    assert seen["reserve_bytes"] == 64 * 1000
    assert seen["cache_budget"] == 7


def _mystery_lm(monkeypatch) -> dict:
    """_default_factory against a model whose KV cost per token is unknown."""
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "mystery"})
    return seen


def test_default_factory_falls_back_to_a_single_slot_when_the_kv_cost_is_unknown(monkeypatch):
    from sous.engine import base

    seen = _mystery_lm(monkeypatch)
    with pytest.warns(UserWarning, match="single prompt-cache slot"):
        base._default_factory("m", prompt_cache=True, cache_budget=None, reserve_tokens=1000)
    assert (seen["cache_budget"], seen["reserve_bytes"]) == (0, 0)


def test_an_explicit_budget_with_an_unknown_kv_cost_still_warns(monkeypatch):
    """Without a per-token cost there is no reserve, so the pressure check can
    never fire: the budget cap is the only thing bounding the map. Silence
    would make that look like a working configuration."""
    from sous.engine import base

    seen = _mystery_lm(monkeypatch)
    with pytest.warns(UserWarning, match="memory-pressure check is disabled"):
        base._default_factory("m", prompt_cache=True, cache_budget=1 << 30, reserve_tokens=1000)
    assert (seen["cache_budget"], seen["reserve_bytes"]) == (1 << 30, 0)


def test_measure_cache_budget_reads_mlxs_numbers(monkeypatch):
    from sous.engine import base
    from sous.engine.promptcache import CACHE_BUDGET_SLACK

    _fake_mlx(monkeypatch, info={"max_recommended_working_set_size": 100 + CACHE_BUDGET_SLACK})
    assert base.measure_cache_budget(30) == 100 - 40 - 30


@pytest.mark.parametrize("info", [{}, "raise"])
def test_measure_cache_budget_degrades_to_a_single_slot_when_mlx_cannot_answer(monkeypatch, info):
    """An mlx API change must not brick delegation and the gateway: every other
    reader of these numbers degrades, and so does this one."""
    from sous.engine import base

    _fake_mlx(monkeypatch, info=info)
    with pytest.warns(UserWarning, match="could not measure the prompt-cache budget"):
        assert base.measure_cache_budget(30) == 0


def _fake_mlx(monkeypatch, *, info) -> None:
    """A stand-in mlx.core, for machines with mlx and for CI without it."""
    import sys
    import types

    def device_info():
        if info == "raise":
            raise RuntimeError("mlx moved")
        return info

    # SimpleNamespace, not ModuleType: the import machinery takes whatever
    # sys.modules holds, and attributes set in a constructor keep ty happy.
    core = types.SimpleNamespace(device_info=device_info, get_active_memory=lambda: 40)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_engine_manager_passes_the_configured_budget_and_the_larger_window(tmp_path, monkeypatch):
    from sous.engine import base

    seen = {}

    def factory(model_id, *args, **kw):
        seen.update(kw)
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", factory)
    cfg = _cfg(
        tmp_path,
        prompt_cache_gb=1.5,
        max_context_tokens=32768,
        gateway_enabled=True,
        gateway_max_context_tokens=131072,
    )
    EngineManager(cfg).get()
    assert seen["cache_budget"] == int(1.5 * (1 << 30))
    assert seen["reserve_tokens"] == 131072
    seen.clear()
    EngineManager(_cfg(tmp_path, gateway_enabled=False, max_context_tokens=32768)).get()
    assert seen["cache_budget"] is None  # auto
    assert seen["reserve_tokens"] == 32768


def test_kernel_memory_pressure_reads_the_kernels_level_or_none():
    """macOS reports 1 (normal), 2 (warn) or 4 (critical); anywhere the sysctl
    is missing the reader says None and the valve stays out of the way."""
    import sys

    from sous.engine.base import kernel_memory_pressure

    level = kernel_memory_pressure()
    assert level in (None, 1, 2, 4)
    if sys.platform == "darwin":
        assert isinstance(level, int)


def test_a_zombie_holder_counts_as_gone():
    """A holder that exited but has not been reaped still has a start time
    psutil can read; only its status says it is gone. Without the status
    check the model stayed pinned until the parent got round to reaping."""
    import subprocess
    import sys

    import psutil

    from sous.engine.base import _holder_alive

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        start = psutil.Process(proc.pid).create_time()
        deadline = time.monotonic() + 10
        while psutil.Process(proc.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "the child never exited"
            time.sleep(0.01)
        assert _holder_alive(proc.pid, start) is False
    finally:
        proc.wait()
    assert _holder_alive(proc.pid, start) is False


def test_status_and_hold_never_report_a_session_that_has_ended():
    """Only the idle sweep pruned, and the worker sweeps only between tasks:
    for a whole delegated task /sous/status counted sessions that had ended,
    and a new hold's reply counted them too."""
    mgr, created, live, clock = _held_manager()
    live.live.add((11, 1.0))
    mgr.hold(11, 1.0)
    _join_preload(mgr)
    live.live.clear()  # the session exits; no sweep runs
    assert mgr.status()["holders"] == 0
    live.live.add((12, 2.0))
    assert mgr.hold(12, 2.0)["holders"] == 1


def test_the_idle_clock_restarts_whichever_caller_sees_the_last_holder_leave():
    """A /sous/status poll can be the first to notice a departure; the sweep
    after it must still find the idle clock restarted, or the model unloads
    the moment a long session ends."""
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    live.live.add((31, 1.0))
    mgr.hold(31, 1.0)
    _join_preload(mgr)
    clock.now += 7200
    live.live.clear()
    assert mgr.status()["holders"] == 0  # status() pruned first
    assert mgr.unload_if_idle() is False  # and the sweep sees a fresh clock
    clock.now += 30 * 60 + 1
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True


def test_version_moves_on_load_unload_hold_release_and_idle_resets_and_nothing_else():
    """What the event stream polls between documents. A read moves nothing;
    every reset of the idle clock does, because the terminal runs that
    clock itself between documents and must hear about a restart."""
    created: list[FakeEngine] = []

    def factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    alive = {"ok": True}
    cfg = SousConfig(idle_unload_minutes=0)
    m = EngineManager(cfg, engine_factory=factory, holder_alive=lambda pid, started: alive["ok"])
    v = m.version
    m.get()
    assert m.version > v, "a load moved nothing"
    v = m.version
    m.status()
    assert m.version == v, "a read moved the version"
    m.get()
    assert m.version == v + 1, "a hit resets the idle clock and moved nothing"
    m.touch()
    assert m.version == v + 2, "a touch resets the idle clock and moved nothing"
    v = m.version
    m.hold(4242, 1.0)
    assert m.version == v + 1, "a hold moved nothing"
    m.status()
    assert m.version == v + 1, "a status read with a live holder moved the version"
    alive["ok"] = False
    m.status()
    assert m.version == v + 2, "a pruned holder moved nothing"
    v = m.version
    time.sleep(0.01)
    assert m.unload_if_idle() is True
    assert m.version == v + 2, "an unload is two changes: it started, it finished"
    assert m.unload_if_idle() is False
    assert m.version == v + 2, "a refused unload moved the version"


def test_status_reports_an_unload_in_progress():
    """The weights come off the GPU over seconds with nothing else in the
    document to show for it, and the event stream keeps its heartbeat for
    the span only if the document says the span is on."""
    slow = _SlowUnloadEngine()
    mgr = EngineManager(SousConfig(idle_unload_minutes=0), engine_factory=lambda mid: slow)
    mgr.get()
    assert mgr.status()["unloading"] is False
    time.sleep(0.01)
    sweeper = threading.Thread(target=mgr.unload_if_idle, daemon=True)
    sweeper.start()
    assert slow.unloading.wait(5)
    s = mgr.status()
    assert s["unloading"] is True and s["loaded"] is False and s["loading"] is False
    slow.release.set()
    sweeper.join(5)
    assert mgr.status()["unloading"] is False


def test_a_preload_that_succeeds_costs_no_document_after_the_load():
    """get() published the engine and bumped; the thread forgetting itself
    afterwards changes nothing a document shows, so it must not bump."""
    cfg = SousConfig(idle_unload_minutes=30)
    m = EngineManager(
        cfg, engine_factory=lambda mid: FakeEngine([]), holder_alive=lambda pid, started: True
    )
    v = m.version
    m.hold(4242, 1.0)
    deadline = time.monotonic() + 5.0
    while m._preload is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert m._preload is None and m.status()["loaded"] is True
    # The hold, the load starting, the load ending — nothing for the thread's exit.
    assert m.version == v + 3


def test_a_preload_that_fails_announces_its_end():
    """The version must move when a failed preload's thread forgets itself:
    `loading` is true until then, and a client that never hears the end
    keeps painting a load that is over."""

    def failing(model_id: str):
        raise RuntimeError("no weights")

    cfg = SousConfig(idle_unload_minutes=30)
    m = EngineManager(cfg, engine_factory=failing, holder_alive=lambda pid, started: True)
    v = m.version
    m.hold(4242, 1.0)
    deadline = time.monotonic() + 5.0
    while m.status()["loading"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert m.status()["loading"] is False
    # The hold, the load starting, its failure, the thread forgetting itself.
    assert m.version == v + 4


def test_unload_now_frees_a_fresh_engine_and_reports_it():
    """The tune asks the daemon to release the weights whatever the idle
    clock says: the refusals are the sweep's, the clock is not."""
    mgr, created = _manager(idle_minutes=30)
    mgr.get()
    mgr.touch()
    assert mgr.unload_now() == {"unloaded": True, "reason": None}
    assert created[0].unloaded is True
    assert mgr.status()["loaded"] is False


def test_unload_now_refuses_with_a_reason_when_nothing_is_loaded():
    mgr, _ = _manager()
    assert mgr.unload_now() == {"unloaded": False, "reason": "nothing loaded"}


def test_unload_now_refuses_under_a_lease_a_holder_and_a_generation():
    created: list[FakeEngine] = []

    def factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    # The default holder check asks psutil about the pid; a fake pid would be
    # pruned before the refusal is reached.
    mgr = EngineManager(SousConfig(), engine_factory=factory, holder_alive=lambda pid, ct: True)
    engine = mgr.get()
    with mgr.lease():
        assert mgr.unload_now() == {"unloaded": False, "reason": "the engine is leased by a turn"}
    mgr.hold(4242, 1.0)
    assert mgr.unload_now() == {"unloaded": False, "reason": "held by 1 session(s)"}
    mgr._holders.clear()
    with engine._gen_lock:
        assert mgr.unload_now() == {"unloaded": False, "reason": "a generation is in flight"}
    assert created[0].unloaded is False
    assert mgr.unload_now()["unloaded"] is True


def test_unload_now_bumps_the_version_and_clears_the_idle_clock():
    mgr, _ = _manager()
    mgr.get()
    before = mgr.version
    mgr.unload_now()
    assert mgr.version > before
    assert mgr.status()["idle_seconds"] is None


def test_unload_now_names_a_load_or_an_unload_in_progress():
    """`_engine` is None during both, and "nothing loaded" would send the
    caller waiting for the memory back the wrong way."""
    started = threading.Event()
    release = threading.Event()

    class SlowUnload(FakeEngine):
        def unload(self) -> None:
            started.set()
            release.wait(5.0)
            super().unload()

    def factory(model_id: str):
        started.set()
        release.wait(5.0)
        return SlowUnload([])

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    assert started.wait(5.0)
    assert mgr.unload_now() == {"unloaded": False, "reason": "a model load is in progress"}
    release.set()
    loader.join(5.0)
    started.clear()
    release.clear()
    unloader = threading.Thread(target=mgr.unload_now, daemon=True)
    unloader.start()
    assert started.wait(5.0)
    assert mgr.unload_now() == {"unloaded": False, "reason": "an unload is in progress"}
    release.set()
    unloader.join(5.0)
    assert mgr.unload_now() == {"unloaded": False, "reason": "nothing loaded"}


def test_default_engine_factory_maps_every_model_value_onto_the_backend(monkeypatch, tmp_path):
    from sous.engine import base

    seen = {}

    def fake(model_id, *args, **kwargs):
        seen["model_id"] = model_id
        seen["args"] = args
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(base, "_default_factory", fake)
    cfg = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="org/m",
        temperature=0.0,
        top_p=0.9,
        top_k=5,
        prompt_cache=False,
        speculative_draft_id="z/d",
        speculative_block_size=2,
        prompt_cache_gb=1.5,
        max_context_tokens=4096,
        gateway_enabled=True,
        gateway_max_context_tokens=65536,
        int8_prefill=True,
    )
    base.default_engine_factory(cfg)("org/m")
    assert seen["model_id"] == "org/m"
    assert seen["args"] == (0.0, 0.9, 5, False)
    assert seen["kwargs"] == {
        "draft_id": "z/d",
        "draft_block_size": 2,
        "cache_budget": int(1.5 * (1 << 30)),
        "reserve_tokens": 65536,
        "int8_prefill": True,
    }


def test_engine_manager_without_a_factory_uses_the_default_one(monkeypatch, tmp_path):
    from sous.engine import base

    calls = []
    monkeypatch.setattr(
        base, "default_engine_factory", lambda cfg: lambda mid: calls.append((cfg.model_id, mid))
    )
    cfg = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", model_id="org/m")
    base.EngineManager(cfg)._factory("org/m")


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_idle_sweep_unloads_an_idle_model_on_its_own_thread():
    mgr, created, live, clock = _held_manager(idle_minutes=1)
    mgr.get()
    assert mgr.status()["loaded"]
    clock.now += 61
    mgr.start_idle_sweep(interval_s=0.01)
    try:
        assert _wait_until(lambda: not mgr.status()["loaded"])
        assert any(t.name == "sous-idle-sweep" for t in threading.enumerate())
    finally:
        mgr.stop_idle_sweep()
    assert not any(t.name == "sous-idle-sweep" for t in threading.enumerate())


def test_the_idle_sweep_keeps_a_fresh_model_and_starts_once():
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    mgr.get()
    mgr.start_idle_sweep(interval_s=0.01)
    mgr.start_idle_sweep(interval_s=0.01)
    try:
        time.sleep(0.05)
        assert mgr.status()["loaded"]
        assert sum(t.name == "sous-idle-sweep" for t in threading.enumerate()) == 1
    finally:
        mgr.stop_idle_sweep()
    mgr.stop_idle_sweep()  # idempotent when nothing runs


def test_the_idle_sweep_releases_its_mlx_state_once_on_exit(monkeypatch):
    import sous.engine.base as base

    released: list[str] = []
    monkeypatch.setattr(base, "release_mlx_thread_state", lambda: released.append("x"))
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    mgr.start_idle_sweep(interval_s=0.01)
    time.sleep(0.03)
    mgr.stop_idle_sweep()
    assert released == ["x"]
