import dataclasses

import pytest

from sous.config import SousConfig
from sous.protocol import WORKER_TOOLS
from sous.tune import bench
from sous.tune.arms import Arm
from sous.tune.bench import (
    DECODE_TOKENS,
    LONG_CONTEXT,
    PREFILL_CONTEXT,
    BenchRow,
    bench_arm,
    build_prompt,
)
from tests.fake_engine import ChunkedFakeEngine, FakeEngine


def _count(messages, tools):
    return sum(len(str(m)) for m in messages) // 4


def test_build_prompt_reaches_the_target_deterministically():
    a = build_prompt(_count, 2048)
    b = build_prompt(_count, 2048)
    assert a == b
    assert _count(a, WORKER_TOOLS) >= 2048
    assert a[0]["role"] == "system" and a[1]["role"] == "user"
    assert "def " in a[1]["content"]


@pytest.mark.parametrize("target", [1024, 2048, 16384])
def test_build_prompt_overshoots_by_less_than_ten_percent(target):
    counted = _count(build_prompt(_count, target), WORKER_TOOLS)
    assert counted >= target
    assert counted < target * 1.1


def _arm(tmp_path, window=131072):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        max_context_tokens=window,
        speculative_draft_id="z/draft",
        speculative_block_size=3,
    )
    return Arm(
        label="m + draft @3",
        config=cfg,
        model_id="org/m",
        drafter_id="z/draft",
        block_size=3,
        window=window,
        gateway_window=None,
        tier="t",
        current=True,
    )


def _fake(script_len=12):
    engines = []

    def factory(model_id):
        e = FakeEngine(["w " * DECODE_TOKENS] * script_len)
        e.stats = {
            "prefill_seconds": 2.0,
            "prefilled_tokens": PREFILL_CONTEXT,
            "decode_seconds": 4.0,
        }
        engines.append(e)
        return e

    return factory, engines


def test_bench_arm_reads_throughput_from_the_engines_gauges(tmp_path):
    factory, engines = _fake()
    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=2,
        peak_memory=lambda: 4096,
        reset_peak=lambda: None,
        active_memory=lambda: 100,
        out=lambda *a, **k: None,
    )
    assert row.ok and row.error is None
    assert row.model_id == "org/m" and row.drafter_id == "z/draft" and row.block_size == 3
    assert row.prefill_tps_2k == PREFILL_CONTEXT / 2.0
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0
    assert row.decode_tps_16k == DECODE_TOKENS / 4.0
    assert row.prefill_tps_16k == PREFILL_CONTEXT / 2.0  # the fake's gauge, whatever the prompt
    assert row.ttft_seconds is not None and row.ttft_seconds >= 0
    assert row.peak_memory_bytes == 4096 and row.spread == 0.0
    assert row.load_seconds is not None
    assert engines[0].unloaded is True
    # Every turn ran on the one session thread the prompt cache is scoped to.
    assert len({t for t in engines[0].generate_threads}) == 1


def test_a_small_window_skips_the_long_context_measurements(tmp_path):
    factory, engines = _fake()
    row = bench_arm(
        _arm(tmp_path, window=8192),
        factory=factory,
        repeat=1,
        peak_memory=lambda: 1,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok and row.decode_tps_16k is None and row.prefill_tps_16k is None
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0
    assert LONG_CONTEXT > 8192


def test_a_failing_load_is_a_row_not_an_exception(tmp_path):
    def factory(model_id):
        raise RuntimeError("no weights")

    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok is False and row.error == "RuntimeError: no weights"
    assert row.decode_tps_1k is None


def test_rows_round_trip_through_dicts():
    row = BenchRow(
        label="a",
        model_id="m",
        drafter_id="",
        block_size=0,
        window=1,
        ok=True,
        error=None,
        load_seconds=1.0,
        prefill_tps_2k=2.0,
        prefill_tps_16k=None,
        decode_tps_1k=3.0,
        decode_tps_16k=None,
        ttft_seconds=0.5,
        peak_memory_bytes=7,
        spread=0.1,
    )
    assert BenchRow.from_dict(row.as_dict()) == row
    assert dataclasses.asdict(row)["label"] == "a"


def test_a_refused_unload_is_recorded_and_skips_the_settle_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench.EngineManager,
        "unload_now",
        lambda self: {"unloaded": False, "reason": "a generation is in flight"},
    )
    factory, _ = _fake()
    memory_calls = []

    def active_memory():
        memory_calls.append(None)
        return 0

    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=1,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=active_memory,
        out=lambda *a, **k: None,
    )
    assert row.ok is True
    assert row.error == "unload refused: a generation is in flight"
    assert row.released is False
    # Just the pre-load baseline: a refused unload skips the settle-wait
    # entirely rather than spinning on memory that cannot come back.
    assert len(memory_calls) == 1


def test_a_teardown_exception_keeps_the_finished_row(tmp_path):
    class RaisingUnloadEngine(FakeEngine):
        def unload(self) -> None:
            raise RuntimeError("weights pinned")

    def factory(model_id):
        e = RaisingUnloadEngine(["w " * DECODE_TOKENS] * 12)
        e.stats = {
            "prefill_seconds": 2.0,
            "prefilled_tokens": PREFILL_CONTEXT,
            "decode_seconds": 4.0,
        }
        return e

    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=1,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok is True  # the measurement itself finished
    assert row.error is not None
    assert "teardown failed" in row.error and "weights pinned" in row.error
    assert row.released is False


def test_spread_is_none_with_a_single_repeat(tmp_path):
    factory, _ = _fake()
    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=1,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok is True
    assert row.spread is None
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0


def test_the_best_repeat_wins_and_the_gauges_are_read_per_owner(tmp_path):
    class VariableDecodeEngine(FakeEngine):
        """decode_seconds alternates across the calls that measure a full
        decode (max_tokens == DECODE_TOKENS, as the short-context and the
        16K decode turns do); prefill and ttft-probe turns see a constant
        rate, the way the fixed-canned-gauge fake elsewhere does."""

        def __init__(self, script):
            super().__init__(script)
            self.owners_seen = []
            self._decode_calls = 0

        def prompt_cache_stats(self, owner=None):
            self.owners_seen.append(owner)
            decode_seconds = 4.0
            if self.max_tokens_seen and self.max_tokens_seen[-1] == DECODE_TOKENS:
                decode_seconds = (4.0, 8.0)[self._decode_calls % 2]
                self._decode_calls += 1
            return {
                "prefill_seconds": 2.0,
                "prefilled_tokens": PREFILL_CONTEXT,
                "decode_seconds": decode_seconds,
            }

    engines = []

    def factory(model_id):
        e = VariableDecodeEngine(["w " * DECODE_TOKENS] * 12)
        engines.append(e)
        return e

    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=2,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok is True
    # The first repeat's 4.0 s (64 tok/s) beats the second's 8.0 s (32 tok/s).
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0
    assert row.spread == 0.5
    owners = engines[0].owners_seen
    assert len(owners) >= 2
    assert len(set(owners)) == 1
    assert owners[0] is not None


def test_without_gauges_the_rates_come_from_the_deltas(tmp_path):
    # Simulates the text-only mlx-lm backend once every layer's cache is
    # trimmable: prefill and decode fuse into one hooks.decode() call, so
    # prefill_seconds/prefilled_tokens never move off 0 and decode_seconds
    # covers the whole turn, not just decode.
    script: list[str] = ["|".join(["w " * 64] * 4)] * 8

    def factory(model_id):
        e = ChunkedFakeEngine(script, delay=0.01)
        e.stats = {"prefill_seconds": 0.0, "prefilled_tokens": 0, "decode_seconds": 0.5}
        return e

    row = bench_arm(
        _arm(tmp_path),
        factory=factory,
        repeat=1,
        peak_memory=lambda: 0,
        reset_peak=lambda: None,
        active_memory=lambda: 0,
        out=lambda *a, **k: None,
    )
    assert row.ok, row.error
    assert row.prefill_tps_2k is not None and row.prefill_tps_2k > 0
    assert row.decode_tps_1k is not None and row.decode_tps_1k > 0
    # The fused gauge (decode_seconds alone, covering prefill too) must not
    # be the source of a decode-only rate.
    assert row.decode_tps_1k != DECODE_TOKENS / 0.5
