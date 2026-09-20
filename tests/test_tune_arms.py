import dataclasses

from sous.config import SousConfig
from sous.tune.arms import BLOCK_SIZES, Arm, quick_arms, winner_stage_arms
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
    # The table's incompatible drafter never becomes an arm; the user's own
    # configured one (absent from this table row) still does.
    drafters = {a.drafter_id for a in arms if a.model_id == "mlx-community/Qwen3.8-27B-4bit"}
    assert drafters == {"", "z-lab/Qwen3.8-27B-DFlash2"}
    assert any("hidden size" in r.reason for r in refusals)


def test_a_drafter_with_unknown_bytes_is_refused_but_the_model_keeps_its_plain_arm(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    cps = _checkpoints()
    cps["z-lab/Qwen3.8-27B-DFlash2"] = dataclasses.replace(
        cps["z-lab/Qwen3.8-27B-DFlash2"], bytes=None
    )
    arms, refusals = quick_arms(user, _candidates(), cps, working_set_bytes=fx.M5_PRO_WORKING_SET)
    mine = [a for a in arms if a.model_id == "mlx-community/Qwen3.8-27B-4bit"]
    assert [a.drafter_id for a in mine] == [""]  # the no-drafter arm survives
    assert any(
        r.model_id == "mlx-community/Qwen3.8-27B-4bit"
        and r.drafter_id == "z-lab/Qwen3.8-27B-DFlash2"
        and "size unknown" in r.reason
        for r in refusals
    )


def test_a_configured_block_size_of_zero_for_auto_is_still_the_current_arm(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_block_size=0)
    arms, _ = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    current = [a for a in arms if a.current]
    assert len(current) == 1
    assert current[0].block_size == 0
    assert current[0].label.endswith("@auto")
    assert current[0].config.speculative_block_size == 0


def test_a_configured_block_size_outside_the_curated_set_is_added(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_block_size=4)
    arms, _ = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    current = [a for a in arms if a.current]
    assert len(current) == 1
    assert current[0].block_size == 4
    assert current[0].label.endswith("@4")
    drafter_blocks = sorted(
        a.block_size
        for a in arms
        if a.model_id == "mlx-community/Qwen3.8-27B-4bit" and a.drafter_id
    )
    assert drafter_blocks == [2, 3, 4, 5]


def test_a_table_model_gains_the_users_own_drafter(tmp_path):
    """The table pairs the 9B with its DFlash; a user who configured another
    drafter for it must still see their own pair measured, or there is no
    baseline and nothing is ever proposed."""
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        speculative_draft_id="z-lab/Qwen3.5-9B-DFlash",
        speculative_block_size=4,
    )
    cands = [Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ("z-lab/Qwen3.5-9B-DFlash",))]
    arms, _ = quick_arms(user, cands, _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert [a.block_size for a in arms if a.current] == [4]
    other = Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ())
    arms, _ = quick_arms(user, [other], _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert [(a.drafter_id, a.block_size) for a in arms if a.current] == [
        ("z-lab/Qwen3.5-9B-DFlash", 4)
    ]


def test_a_configured_drafter_the_model_cannot_use_makes_the_plain_arm_current(tmp_path):
    """The README's smaller-machine recipe changes only [model].id, leaving
    the 27B drafter configured for a 9B: the daemon runs it undrafted, so
    the no-drafter arm is the configuration in use — the baseline a better
    drafter is measured against."""
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        speculative_draft_id="z-lab/Qwen3.8-27B-DFlash2",
    )
    cands = [Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ("z-lab/Qwen3.5-9B-DFlash",))]
    arms, refusals = quick_arms(
        user, cands, _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    assert any("hidden size" in r.reason for r in refusals)
    current = [a for a in arms if a.current]
    assert len(current) == 1 and current[0].drafter_id == ""
    assert {a.drafter_id for a in arms} == {"", "z-lab/Qwen3.5-9B-DFlash"}


def test_each_arm_carries_the_window_it_alone_fits_at(tmp_path):
    user = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        speculative_draft_id="z-lab/Qwen3.5-9B-DFlash",
        max_context_tokens=131072,
    )
    arms, _ = quick_arms(
        user, _candidates()[1:], _checkpoints(), working_set_bytes=fx.M2_AIR_WORKING_SET
    )
    plain = next(a for a in arms if not a.drafter_id)
    drafted = next(a for a in arms if a.drafter_id)
    # One measurement window for the model, lowered for its drafter...
    assert plain.window == drafted.window < 131072
    # ...but the drafterless arm proposes the window it fits at by itself.
    assert plain.fit_window is not None and drafted.fit_window is not None
    assert plain.fit_window > drafted.fit_window == drafted.window
    assert plain.key == (plain.model_id, "", 0) and plain.serving_window == plain.window


def test_a_text_only_model_gets_no_drafter_arms(tmp_path):
    """The factory drops the drafter for the mlx-lm backend without a word;
    an arm that asked for one would measure undrafted numbers under its
    label."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_draft_id="")
    cps = _checkpoints()
    cps["o/text"] = describe(
        "o/text",
        config_fn=lambda m: {
            **fx.qwen_27b()["text_config"],
            "model_type": "qwen3",
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        },
        size_fn=lambda m: 16_100_000_000,
    )
    assert cps["o/text"].backend == "lm"
    cands = [Candidate("o/text", "manual", ("z-lab/Qwen3.8-27B-DFlash2",))]
    arms, refusals = quick_arms(user, cands, cps, working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert [a.drafter_id for a in arms if a.model_id == "o/text"] == [""]
    assert any("text-only" in r.reason for r in refusals)


def _winner(tmp_path, temperature=0.7, int8=False):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
    )
    return Arm(
        label="27B + DFlash2 @3",
        config=cfg,
        model_id="mlx-community/Qwen3.8-27B-4bit",
        drafter_id="z-lab/Qwen3.8-27B-DFlash2",
        block_size=3,
        window=131072,
        gateway_window=None,
        tier="27b-dense",
        current=True,
        fit_window=131072,
        int8_prefill=int8,
        greedy=temperature == 0,
    )


def test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions(tmp_path):
    arm = _winner(tmp_path)
    assert arm.key == ("mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2", 3)
    assert arm.suite_key == (*arm.key, False, False)


def test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_a_sampled_winner_a_greedy_arm(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)
    assert [a.label for a in arms] == ["27B + DFlash2 @3 + int8 prefill", "27B + DFlash2 @3 greedy"]
    int8, greedy = arms
    assert int8.int8_prefill and int8.config.int8_prefill and not int8.greedy
    assert greedy.greedy and greedy.config.temperature == 0 and not greedy.int8_prefill
    assert int8.suite_key == (*int8.key, True, False)
    assert greedy.suite_key == (*greedy.key, False, True)
    assert all(not a.current and a.fit_window == 131072 for a in arms)
    # Only the winner stage's own int8 arm is under test: the engine
    # refusing it must fail the arm, unlike an arm that merely inherited
    # int8_prefill from the user's config.
    assert int8.int8_under_test and not greedy.int8_under_test


def test_without_nax_only_the_greedy_arm_is_offered(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    assert [a.label for a in winner_stage_arms(_winner(tmp_path), nax=False, checkpoint=cp)] == [
        "27B + DFlash2 @3 greedy"
    ]


def test_a_checkpoint_int8_cannot_route_gets_no_int8_arm(tmp_path):
    cp = describe(
        "mlx-community/Qwen3.8-27B-mxfp4",
        config_fn=lambda m: fx.qwen_27b({"group_size": 32, "bits": 4, "mode": "mxfp4"}),
        size_fn=lambda m: 15_000_000_000,
    )
    arms = winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)
    assert [a.greedy for a in arms] == [True]


def test_a_winner_already_greedy_and_int8_gets_no_extra_arms(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path, temperature=0, int8=True), nax=True, checkpoint=cp)
    assert arms == []


def test_a_moe_checkpoint_gets_no_int8_arm(tmp_path):
    cp = describe(
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        config_fn=lambda m: fx.qwen_27b(model_type="qwen3_5_moe"),
        size_fn=lambda m: 20_000_000_000,
    )
    arms = winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)
    assert [a.greedy for a in arms] == [True]


def test_quick_arms_mirror_the_users_int8_and_greedy_settings(tmp_path):
    user = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        int8_prefill=True,
        temperature=0.0,
    )
    arms, refusals = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    assert refusals == []
    for arm in arms:
        assert arm.int8_prefill is True
        assert arm.greedy is True
        # Inherited, not proposed: quick_arms never puts an arm under test.
        assert arm.int8_under_test is False
        assert arm.suite_key == (*arm.key, True, True)
        cp = _checkpoints()[arm.model_id]
        extra_arms = winner_stage_arms(arm, nax=True, checkpoint=cp)
        assert extra_arms == []
