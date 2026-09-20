"""Local only: one quick arm through the real engine on the smallest model,
so the bench's gauge reads, the session thread and the unload are exercised
against mlx itself, not a fake."""

from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import bench_arm
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import run_suite

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


def test_the_smallest_suite_task_runs_through_the_runner_on_the_tiny_model(tmp_path):
    """Outcome recorded, grade computed, nothing asserted about the grade —
    the 0.6B cannot pass a task; what this proves is the real engine, the
    real sandbox and the grader wired together, and the weights released."""
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
        tier="test",
        current=True,
    )
    task = next(t for t in load_tasks() if t.name == "docstring_sweep")
    recorded = []
    outcome = run_suite(
        arm,
        [task],
        runs=1,
        done=set(),
        record=recorded.append,
        scratch=tmp_path / "scratch",
        out=lambda *a, **k: None,
    )
    assert outcome.error is None and outcome.released, outcome.error
    assert len(outcome.runs) == 1 and recorded == outcome.runs
    run = outcome.runs[0]
    assert run.state in ("done", "failed")
    assert 0.0 <= run.grade <= 1.0 and run.grade_detail
    assert run.turns >= 1 and run.seconds > 0 and run.output_tokens > 0
    assert run.transcript_path and Path(run.transcript_path).is_file()
