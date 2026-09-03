"""One gateway turn on the shared engine: serialized, drained, session reuse."""

import threading
import time
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import Delta, EngineManager, GenerationStalled
from sous.gateway.turn import (
    GatewayBusy,
    PromptTooLong,
    TurnAbandoned,
    TurnResult,
    TurnRunner,
)
from tests.fake_engine import ChunkedFakeEngine, FakeEngine


class RecordingSink:
    def __init__(self):
        self.started_with: list[int] = []
        self.deltas: list[Delta] = []
        self.threads: set[threading.Thread] = set()

    def started(self, input_tokens: int) -> None:
        self.started_with.append(input_tokens)
        self.threads.add(threading.current_thread())

    def delta(self, delta: Delta) -> None:
        self.deltas.append(delta)
        self.threads.add(threading.current_thread())


def _cfg(tmp_path: Path, **overrides) -> SousConfig:
    overrides.setdefault("gateway_enabled", True)
    return SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        **overrides,
    )


def _runner(tmp_path: Path, inner, **overrides) -> tuple[TurnRunner, EngineManager]:
    engines = EngineManager(_cfg(tmp_path, **overrides), engine_factory=lambda mid: inner)
    return TurnRunner(engines, _cfg(tmp_path, **overrides)), engines


MSGS = [{"role": "user", "content": "hello"}]


def test_run_streams_deltas_and_reports_real_counts(tmp_path: Path):
    inner = ChunkedFakeEngine(["Hel|lo"])
    runner, _ = _runner(tmp_path, inner)
    sink = RecordingSink()
    result = runner.run(MSGS, [], 4096, sink)
    assert isinstance(result, TurnResult)
    assert result.text == "Hello"
    assert result.input_tokens == inner.count_tokens(MSGS, [])
    assert result.output_tokens == 2 and result.finish_reason == "stop"
    assert sink.started_with == [result.input_tokens]
    assert [d.text for d in sink.deltas] == ["Hel", "lo"]
    assert result.seconds >= 0 and result.cache_hit is False and result.reused_tokens == 0


def test_max_tokens_is_clamped_to_the_room_left_in_the_window(tmp_path: Path):
    inner = FakeEngine(["ok"])
    runner, _ = _runner(tmp_path, inner)
    # 100_000 > the 65536 window: with a request that already fits, the test
    # would be vacuous (Claude Code's 32000 fits comfortably).
    runner.run(MSGS, [], 100_000, RecordingSink())
    room = runner._window - inner.count_tokens(MSGS, [])
    assert inner.max_tokens_seen == [room]
    inner.script.append("ok")
    runner.run(MSGS, [], 10, RecordingSink())
    assert inner.max_tokens_seen[-1] == 10


def test_prompt_that_fills_the_window_is_rejected_before_generating(tmp_path: Path):
    inner = FakeEngine(["never"])
    runner, _ = _runner(tmp_path, inner)  # window 65536; FakeEngine counts len/4
    sink = RecordingSink()
    with pytest.raises(PromptTooLong) as exc:
        runner.run([{"role": "user", "content": "x" * 300_000}], [], 100, sink)
    assert exc.value.window == 65536 and exc.value.tokens >= 65536
    assert "prompt is too long" in str(exc.value)
    assert sink.started_with == [] and inner.calls == []


def test_all_turns_share_one_session_thread_so_the_prompt_cache_can_survive(tmp_path: Path):
    inner = FakeEngine(["a", "b"])
    runner, _ = _runner(tmp_path, inner)
    runner.run(MSGS, [], 100, RecordingSink())
    runner.run(MSGS, [], 100, RecordingSink())
    assert inner.generate_threads[0] is inner.generate_threads[1]
    assert inner.resets == 0  # unlike run_task, a turn never resets the cache
    runner.close()


def test_a_reloaded_engine_gets_a_fresh_session(tmp_path: Path):
    """After an idle unload the next get() builds a new ManagedEngine; the old
    session's thread would call into weights that are gone."""
    made: list[FakeEngine] = []

    def factory(mid):
        e = FakeEngine(["a", "b"])
        made.append(e)
        return e

    cfg = _cfg(tmp_path, idle_unload_minutes=0)
    engines = EngineManager(cfg, engine_factory=factory)
    runner = TurnRunner(engines, cfg)
    runner.run(MSGS, [], 100, RecordingSink())
    first_session = runner._session
    assert first_session is not None
    time.sleep(0.01)
    assert engines.unload_if_idle() is True
    runner.run(MSGS, [], 100, RecordingSink())
    assert len(made) == 2 and made[1].calls  # second turn ran on the new engine
    assert runner._session is not first_session
    first_session._thread.join(5)
    assert not first_session._thread.is_alive()


def test_turns_are_serialized(tmp_path: Path):
    inner = ChunkedFakeEngine(["one|two", "three"], delay=0.2)
    runner, _ = _runner(tmp_path, inner)
    order: list[str] = []

    def go(label):
        runner.run(MSGS, [], 100, RecordingSink())
        order.append(label)

    a = threading.Thread(target=go, args=("a",))
    b = threading.Thread(target=go, args=("b",))
    a.start()
    time.sleep(0.05)
    b.start()
    a.join(5)
    b.join(5)
    assert order == ["a", "b"]
    assert len(inner.generate_threads) == 2


def test_busy_gateway_gives_up_after_the_timeout(tmp_path: Path):
    entered = threading.Event()

    class Announcing(ChunkedFakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            entered.set()  # t is past session.generate(timeout=...) by now
            return super().generate(messages, tools, max_tokens, on_delta)

    inner = Announcing(["slow|slow|slow"], delay=0.3)
    runner, _ = _runner(tmp_path, inner)
    t = threading.Thread(target=runner.run, args=(MSGS, [], 100, RecordingSink()))
    t.start()
    assert entered.wait(5)  # t holds the gateway lock and is generating
    # Config is minutes-granular; the waiter needs a sub-second bound. Set it
    # only now, after t captured the long timeout for its own generation.
    runner._timeout = 0.2
    with pytest.raises(GatewayBusy):
        runner.run(MSGS, [], 100, RecordingSink())
    t.join(5)
    assert inner.finished.wait(5)


def test_a_stall_drops_the_session_and_the_next_turn_gets_a_new_one(tmp_path: Path):
    gate = threading.Event()

    class Gated(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            gate.wait(10)
            return super().generate(messages, tools, max_tokens, on_delta)

    inner = Gated(["late", "fresh"])
    runner, _ = _runner(tmp_path, inner)
    runner._timeout = 0.1
    with pytest.raises(GenerationStalled):
        runner.run(MSGS, [], 100, RecordingSink())
    assert runner._session is None
    assert inner.resets == 1  # the abandoned thread's cache must never be adopted
    gate.set()  # the stalled thread finishes and releases the engine lock
    time.sleep(0.2)
    runner._timeout = 5
    assert runner.run(MSGS, [], 100, RecordingSink()).text == "fresh"
    assert inner.generate_threads[0] is not inner.generate_threads[1]


def test_a_turn_abandoned_while_queued_never_generates(tmp_path: Path):
    """Drain-to-completion covers a generation that started. One whose client
    left while it was still waiting for the lock must not start."""
    inner = FakeEngine(["never"])
    runner, _ = _runner(tmp_path, inner)
    gone = threading.Event()
    gone.set()
    with pytest.raises(TurnAbandoned):
        runner.run(MSGS, [], 100, RecordingSink(), abandoned=gone)
    assert inner.calls == []
    assert not runner._lock.locked()


def test_run_releases_mlx_thread_state_and_touches_the_engine(tmp_path: Path, monkeypatch):
    import sous.gateway.turn as turn

    released: list[bool] = []
    monkeypatch.setattr(turn, "release_mlx_thread_state", lambda: released.append(True))
    inner = FakeEngine(["ok"])
    runner, engines = _runner(tmp_path, inner)
    runner.run(MSGS, [], 100, RecordingSink())
    assert released == [True]
    idle = engines.status()["idle_seconds"]
    assert idle is not None and idle < 1.0


def test_count_tokens_uses_the_engine_and_releases(tmp_path: Path, monkeypatch):
    import sous.gateway.turn as turn

    released: list[bool] = []
    monkeypatch.setattr(turn, "release_mlx_thread_state", lambda: released.append(True))
    inner = FakeEngine([])
    runner, _ = _runner(tmp_path, inner)
    assert runner.count_tokens(MSGS, []) == inner.count_tokens(MSGS, [])
    assert released == [True]


def test_cache_hit_is_reported_from_the_engines_counters(tmp_path: Path):
    inner = FakeEngine(["a", "b"])
    inner.stats = {"hits": 0, "reused_tokens": 0}
    runner, _ = _runner(tmp_path, inner)

    # Simulate the engine's stats moving during the second turn.
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {"hits": 1, "reused_tokens": 900}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    first = runner.run(MSGS, [], 100, RecordingSink())
    second = runner.run(MSGS, [], 100, RecordingSink())
    assert (first.cache_hit, first.reused_tokens) == (False, 0)
    assert (second.cache_hit, second.reused_tokens) == (True, 900)
