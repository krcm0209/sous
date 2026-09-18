"""Local only: one quick arm through the real engine on the smallest model,
so the bench's gauge reads, the session thread and the unload are exercised
against mlx itself, not a fake."""

import pytest

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import bench_arm

pytestmark = pytest.mark.model

TINY = "mlx-community/Qwen3-0.6B-4bit"


def test_bench_arm_measures_the_tiny_model_and_releases_it(tmp_path):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        model_id=TINY,
        speculative_draft_id="",
        max_context_tokens=8192,
    )
    arm = Arm(
        label="tiny",
        config=cfg,
        model_id=TINY,
        drafter_id="",
        block_size=0,
        window=8192,
        gateway_window=None,
        tier="test",
        current=True,
    )
    row = bench_arm(arm, repeat=1, out=lambda *a, **k: None)
    assert row.ok, row.error
    assert row.prefill_tps_2k and row.prefill_tps_2k > 0
    assert row.decode_tps_1k and row.decode_tps_1k > 0
    assert row.decode_tps_16k is None  # the window is too small for the long context
    assert row.ttft_seconds is not None and row.load_seconds is not None
    assert row.peak_memory_bytes and row.peak_memory_bytes > 0
