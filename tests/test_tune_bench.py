import dataclasses

from sous.config import SousConfig
from sous.protocol import WORKER_TOOLS
from sous.tune.arms import Arm
from sous.tune.bench import (
    DECODE_TOKENS,
    LONG_CONTEXT,
    PREFILL_CONTEXT,
    BenchRow,
    bench_arm,
    build_prompt,
)
from tests.fake_engine import FakeEngine


def _count(messages, tools):
    return sum(len(str(m)) for m in messages) // 4


def test_build_prompt_reaches_the_target_deterministically():
    a = build_prompt(_count, 2048)
    b = build_prompt(_count, 2048)
    assert a == b
    assert _count(a, WORKER_TOOLS) >= 2048
    assert a[0]["role"] == "system" and a[1]["role"] == "user"
    assert "def " in a[1]["content"]


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
