from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.decide import quick_decision, score

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _row(label, model=M, drafter=D, block=3, d1k=25.0, d16k=18.0, ok=True, window=131072):
    return BenchRow(
        label=label,
        model_id=model,
        drafter_id=drafter,
        block_size=block,
        window=window,
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
    user,
    label,
    drafter=D,
    block=3,
    window=131072,
    gateway_window=None,
    current=False,
    model=M,
    fit_window=None,
    fit_gateway_window=None,
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
        fit_window=fit_window,
        fit_gateway_window=fit_gateway_window,
    )


def test_score_prefers_the_long_context_measurement():
    assert score(_row("a", d16k=18.0, d1k=25.0)) == 18.0
    assert score(_row("a", d16k=None, d1k=25.0)) == 25.0
    assert score(_row("a", ok=False)) is None


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
    rows = [_row("cur", window=65536)]
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


def test_a_row_whose_arm_is_gone_is_ignored(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True)]
    rows = [_row("cur", d16k=18.0), _row("ghost", block=2, d16k=99.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "cur" and choice.changes == {}
    assert quick_decision(user, arms, [_row("ghost", block=2, d16k=99.0)]) is None


def test_without_a_current_arm_the_best_row_still_labels_the_choice_but_proposes_nothing(tmp_path):
    """No arm is marked current — the user's own pair never measured at
    all — so there is no baseline to diff the winner against; proposing
    `changes` from the winner's numbers alone would trust one arm's speed
    without ever knowing whether it beats what the user already has."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "b2", block=2), _arm(user, "b5", block=5)]
    rows = [_row("b2", block=2, d16k=20.0), _row("b5", block=5, d16k=22.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "b5"
    assert choice.changes == {}
    assert len(choice.reasons) == 1
    assert "not measured" in choice.reasons[0]


def test_a_failed_current_arm_proposes_no_change_but_names_the_failure(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", ok=False), _row("b2", block=2, d16k=19.5)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "b2"
    assert choice.changes == {}
    assert "cur" in choice.reasons[0]
    assert "did not measure" in choice.reasons[0] and "x" in choice.reasons[0]


def test_every_compared_row_is_scored_in_one_unit(tmp_path):
    """A row without a 16K number must not be ranked on its 1K one against
    siblings ranked at 16K: the short context flatters a drafter, and the
    winner's settings get written to the config."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d1k=20.0, d16k=18.0), _row("b2", block=2, d1k=24.0, d16k=None)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    # 1K for both: 24.0 beats 20.0, and the reason says which context.
    assert choice.label == "b2" and "1K" in choice.reasons[0]
    assert "24.0" in choice.reasons[0] and "20.0" in choice.reasons[0]


def test_a_row_measured_at_another_window_is_not_compared(tmp_path):
    """A resumed run whose config changed in between carries rows of two
    windows; only this run's window counts, the rest is measured again."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d16k=18.0), _row("b2", block=2, d16k=30.0, window=16384)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "cur" and choice.changes == {}


def test_arms_are_told_apart_by_model_drafter_and_block_not_by_label(tmp_path):
    """Two orgs' checkpoints with one repo basename share a label; the
    winner must resolve to the arm its row measured."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_draft_id="")
    other = "some-org/Qwen3.8-27B-4bit"
    arms = [
        _arm(user, "Qwen3.8-27B-4bit", drafter="", block=0, current=True),
        _arm(user, "Qwen3.8-27B-4bit", drafter="", block=0, model=other, fit_window=23040),
    ]
    rows = [_row("Qwen3.8-27B-4bit", drafter="", block=0, d16k=18.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.model_id == M
    assert choice.changes == {} and "already the fastest" in choice.reasons[-1]


def test_a_proposal_writes_the_arms_own_fit_window_not_the_shared_one(tmp_path):
    """The measurement window is lowered for the model's heaviest drafter;
    a drafterless winner must not keep that reservation."""
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        max_context_tokens=131072,
        gateway_enabled=True,
        gateway_max_context_tokens=131072,
    )
    arms = [
        _arm(user, "cur", current=True, window=78080, gateway_window=78080, fit_window=78080),
        _arm(
            user,
            "plain",
            drafter="",
            block=0,
            window=78080,
            gateway_window=78080,
            fit_window=100864,
            fit_gateway_window=100864,
        ),
    ]
    rows = [
        _row("cur", d16k=10.0, window=78080),
        _row("plain", drafter="", block=0, d16k=12.0, window=78080),
    ]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    assert choice.changes == {
        "model": {"speculative_draft_id": "", "max_context_tokens": 100864},
        "gateway": {"max_context_tokens": 100864},
    }
