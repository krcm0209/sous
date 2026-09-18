from sous.config import SousConfig
from sous.tune.arms import BLOCK_SIZES, quick_arms
from sous.tune.candidates import Candidate, describe
from tests import tune_fixtures as fx


def _checkpoints():
    return {
        "mlx-community/Qwen3.8-27B-4bit": describe(
            "mlx-community/Qwen3.8-27B-4bit",
            config_fn=lambda m: fx.qwen_27b(),
            size_fn=lambda m: 16_100_000_000,
        ),
        "z-lab/Qwen3.8-27B-DFlash2": describe(
            "z-lab/Qwen3.8-27B-DFlash2",
            config_fn=lambda m: fx.dflash2_27b(),
            size_fn=lambda m: 3_850_000_000,
        ),
        "mlx-community/Qwen3.5-9B-MLX-4bit": describe(
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            config_fn=lambda m: fx.qwen_9b(),
            size_fn=lambda m: 6_000_000_000,
        ),
        "z-lab/Qwen3.5-9B-DFlash": describe(
            "z-lab/Qwen3.5-9B-DFlash",
            config_fn=lambda m: fx.dflash_9b(),
            size_fn=lambda m: 2_600_000_000,
        ),
    }


def _candidates():
    return [
        Candidate("mlx-community/Qwen3.8-27B-4bit", "27b-dense", ("z-lab/Qwen3.8-27B-DFlash2",)),
        Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ("z-lab/Qwen3.5-9B-DFlash",)),
    ]


def test_on_the_m5_pro_every_model_gets_a_no_drafter_arm_and_one_per_block_size(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms, refusals = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    assert refusals == []
    by_model = {}
    for a in arms:
        by_model.setdefault(a.model_id, []).append(a)
    assert len(by_model["mlx-community/Qwen3.8-27B-4bit"]) == 1 + len(BLOCK_SIZES)
    assert len(by_model["mlx-community/Qwen3.5-9B-MLX-4bit"]) == 1 + len(BLOCK_SIZES)
    current = [a for a in arms if a.current]
    assert len(current) == 1
    assert (current[0].drafter_id, current[0].block_size) == ("z-lab/Qwen3.8-27B-DFlash2", 3)
    assert current[0].config.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    assert current[0].config.speculative_block_size == 3
    assert current[0].window == user.max_context_tokens
    plain = [a for a in by_model["mlx-community/Qwen3.8-27B-4bit"] if not a.drafter_id][0]
    assert plain.block_size == 0 and plain.config.speculative_draft_id == ""
    assert plain.label == "Qwen3.8-27B-4bit"
    assert current[0].label == "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3"


def test_on_the_m2_air_the_27b_is_refused_and_the_9b_runs_at_a_lowered_window(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", max_context_tokens=131072)
    arms, refusals = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M2_AIR_WORKING_SET
    )
    assert [r.model_id for r in refusals] == ["mlx-community/Qwen3.8-27B-4bit"]
    assert "GiB" in refusals[0].reason
    assert {a.model_id for a in arms} == {"mlx-community/Qwen3.5-9B-MLX-4bit"}
    assert all(a.window < 131072 and a.window % 256 == 0 for a in arms)
    assert all(a.config.max_context_tokens == a.window for a in arms)
    assert not any(a.current for a in arms)


def test_the_gateway_window_is_lowered_too_and_floors_at_claude_codes_minimum(tmp_path):
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        gateway_enabled=True,
        gateway_max_context_tokens=262144,
    )
    arms, _ = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M2_AIR_WORKING_SET
    )
    nine = [a for a in arms if a.model_id == "mlx-community/Qwen3.5-9B-MLX-4bit"]
    assert nine and all(
        a.gateway_window is not None and a.gateway_window >= 48 * 1024 for a in nine
    )
    assert all(a.config.gateway_max_context_tokens == a.gateway_window for a in nine)


def test_a_current_model_outside_the_table_is_still_an_arm(tmp_path):
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        speculative_draft_id="",
    )
    arms, _ = quick_arms(
        user, _candidates()[:1], _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    nine = [a for a in arms if a.model_id == "mlx-community/Qwen3.5-9B-MLX-4bit"]
    assert nine and nine[0].tier == "current"
    assert [a for a in nine if a.current][0].drafter_id == ""


def test_an_incompatible_drafter_is_dropped_before_any_arm_exists(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    cps = _checkpoints()
    cands = [Candidate("mlx-community/Qwen3.8-27B-4bit", "27b-dense", ("z-lab/Qwen3.5-9B-DFlash",))]
    arms, refusals = quick_arms(user, cands, cps, working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert [a.drafter_id for a in arms if a.model_id == "mlx-community/Qwen3.8-27B-4bit"] == [""]
    assert any("hidden size" in r.reason for r in refusals)
