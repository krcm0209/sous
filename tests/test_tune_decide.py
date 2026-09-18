from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.decide import best_per_model, quick_decision, score

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _row(label, model=M, drafter=D, block=3, d1k=25.0, d16k=18.0, ok=True):
    return BenchRow(
        label=label,
        model_id=model,
        drafter_id=drafter,
        block_size=block,
        window=131072,
        ok=ok,
        error=None if ok else "x",
        load_seconds=5.0,
        prefill_tps_2k=480.0,
        prefill_tps_16k=400.0,
        decode_tps_1k=d1k,
        decode_tps_16k=d16k,
        ttft_seconds=1.0,
        peak_memory_bytes=1,
        spread=0.02,
    )


def _arm(
    user, label, drafter=D, block=3, window=131072, gateway_window=None, current=False, model=M
):
    return Arm(
        label=label,
        config=user,
        model_id=model,
        drafter_id=drafter,
        block_size=block,
        window=window,
        gateway_window=gateway_window,
        tier="27b-dense",
        current=current,
    )


def test_score_prefers_the_long_context_measurement():
    assert score(_row("a", d16k=18.0, d1k=25.0)) == 18.0
    assert score(_row("a", d16k=None, d1k=25.0)) == 25.0
    assert score(_row("a", ok=False)) is None


def test_best_per_model_picks_the_top_score_of_each_model():
    rows = [
        _row("a", d16k=18.0),
        _row("b", block=2, d16k=19.5),
        _row("c", model="o/9b", drafter="", block=0, d16k=40.0),
    ]
    best = best_per_model(rows)
    assert best[M].label == "b" and best["o/9b"].label == "c"


def test_a_faster_block_size_becomes_the_only_change(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d16k=18.0), _row("b2", block=2, d16k=19.5)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "b2"
    assert choice.changes == {"model": {"speculative_block_size": 2}}
    assert any("19.5" in r and "18.0" in r for r in choice.reasons)


def test_a_tie_keeps_the_current_arm_and_yields_no_changes(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d16k=18.0), _row("b2", block=2, d16k=18.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "cur" and choice.changes == {}


def test_no_drafter_winning_clears_the_drafter_key(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "plain", drafter="", block=0)]
    rows = [_row("cur", d16k=10.0), _row("plain", drafter="", block=0, d16k=12.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    assert choice.changes == {"model": {"speculative_draft_id": ""}}


def test_a_lowered_window_is_written_for_the_worker_and_the_gateway(tmp_path):
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        max_context_tokens=131072,
        gateway_enabled=True,
        gateway_max_context_tokens=262144,
    )
    arms = [_arm(user, "cur", current=True, window=65536, gateway_window=65536)]
    rows = [_row("cur")]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    assert choice.changes == {
        "model": {"max_context_tokens": 65536},
        "gateway": {"max_context_tokens": 65536},
    }


def test_other_models_never_win_and_a_missing_current_model_is_none(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "moe", model="o/moe", drafter="", block=0)]
    rows = [_row("cur", d16k=18.0), _row("moe", model="o/moe", drafter="", block=0, d16k=60.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.model_id == M
    assert quick_decision(user, arms, [rows[1]]) is None
    assert quick_decision(user, arms, [_row("cur", ok=False)]) is None
