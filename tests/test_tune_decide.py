import dataclasses

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.decide import (
    ArmSummary,
    full_decision,
    model_stage,
    quick_decision,
    score,
    summarize,
)
from sous.tune.suite.runner import SuiteRun

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


M9 = "mlx-community/Qwen3.5-9B-MLX-4bit"


def _run(
    arm, task="t", index=0, *, grade=1.0, seconds=60.0, state="done", repetitions=0, tokens=100
):
    return SuiteRun(
        task=task,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        window=arm.window,
        state=state,
        outcome="completed" if state == "done" else None,
        turns=4,
        seconds=seconds,
        output_tokens=tokens,
        malformed=0,
        repetitions=repetitions,
        approvals_denied=0,
        grade=grade,
        grade_detail="",
        error=None if state == "done" else "x",
        transcript_path=None,
    )


def _runs(arm, grades, **kw):
    return [_run(arm, task=f"t{i}", grade=g, **kw) for i, g in enumerate(grades)]


def _user(tmp_path, **over):
    return SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", **over)


def _peak(label, model, drafter, block, gib):
    row = _row(label, model=model, drafter=drafter, block=block)
    return dataclasses.replace(row, peak_memory_bytes=int(gib * 2**30))


def test_model_stage_picks_each_models_fastest_arm_and_keeps_the_current_one(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", block=3, current=True)
    fast = _arm(user, "27 @5", block=5)
    plain = _arm(user, "27", drafter="", block=0)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    nine_d = _arm(user, "9 + d @3", drafter="z/9d", block=3, model=M9)
    rows = [
        _row("27 @3", block=3, d16k=18.0),
        _row("27 @5", block=5, d16k=20.0),
        _row("27", drafter="", block=0, d16k=15.0),
        _row("9", model=M9, drafter="", block=0, d16k=40.0),
        _row("9 + d @3", model=M9, drafter="z/9d", block=3, d16k=40.0),
    ]
    stage = model_stage([cur, fast, plain, nine, nine_d], rows)
    assert [a.label for a in stage] == ["27 @5", "9", "27 @3"]


def test_model_stage_drops_a_model_without_a_successful_row(tmp_path):
    user = _user(tmp_path)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    assert model_stage([nine], [_row("9", model=M9, drafter="", block=0, ok=False)]) == []


def test_summarize_reads_the_runs_and_the_bench_peak(tmp_path):
    user = _user(tmp_path)
    arm = _arm(user, "27 @3", current=True)
    runs = _runs(arm, [1.0, 0.5], seconds=30.0) + [_run(arm, "t2", state="failed", grade=0.0)]
    s = summarize(arm, runs, [_peak("27 @3", M, D, 3, 20.0)])
    assert s == ArmSummary(
        label="27 @3",
        key=arm.suite_key,
        model_id=M,
        runs=3,
        completed=2,
        mean_grade=0.5,
        wall_seconds=120.0,
        repetitions=0,
        malformed=0,
        approvals_denied=0,
        output_tokens=300,
        peak_memory_bytes=20 * 2**30,
    )
    assert summarize(_arm(user, "27 @5", block=5), runs, []) is None


def test_summarize_ignores_runs_measured_at_another_window(tmp_path):
    """A resumed run whose max_context_tokens changed in between carries
    suite rows of two windows, exactly like _suite_stage's own done-set:
    only the rows measured at this arm's own window belong to it."""
    user = _user(tmp_path)
    arm = _arm(user, "27 @3", current=True)
    mine = _run(arm, "t", grade=1.0, seconds=30.0)
    other = dataclasses.replace(mine, window=32768)
    s = summarize(arm, [mine, other], [])
    assert s is not None
    assert s.runs == 1
    assert s.wall_seconds == 30.0
    assert s.mean_grade == 1.0


def test_a_faster_arm_within_the_margin_wins_and_changes_the_model(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True, fit_window=131072)
    nine = _arm(user, "9", drafter="", block=0, model=M9, fit_window=131072)
    runs = _runs(cur, [1.0, 0.9, 0.9, 1.0], seconds=100.0) + _runs(nine, [0.9] * 4, seconds=40.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "9" and choice.reference_label == "27 @3"
    assert choice.changes == {"model": {"id": M9, "speculative_draft_id": ""}}
    text = "\n".join(choice.reasons)
    assert "reference: 27 @3 (the configured arm)" in text
    assert "mean grade >= 0.90 (0.95 - 0.05)" in text
    assert "9: grade 0.90, completed 4/4, wall 160 s" in text and "-> eligible" in text
    assert "winner: 9 (160 s vs 400 s for the reference)" in text


def test_a_faster_arm_below_the_margin_loses(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0, 1.0], seconds=100.0) + _runs(nine, [0.9, 0.98], seconds=10.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3" and choice.changes == {}
    assert any("9: " in r and "not eligible: grade 0.94 < 0.95" in r for r in choice.reasons)
    assert "winner: 27 @3 (the reference; nothing eligible is faster)" in choice.reasons[-1]


def test_too_few_completed_runs_or_more_repetitions_make_an_arm_ineligible(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0] * 4, seconds=100.0) + [
        _run(nine, "t0", grade=1.0, seconds=10.0),
        _run(nine, "t1", grade=1.0, seconds=10.0, state="failed"),
        _run(nine, "t2", grade=1.0, seconds=10.0, state="failed"),
        _run(nine, "t3", grade=1.0, seconds=10.0, state="failed"),
    ]
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3"
    assert any("not eligible: completed 1 < 2" in r for r in choice.reasons)
    runs = _runs(cur, [1.0] * 2, seconds=100.0) + _runs(
        nine, [1.0] * 2, seconds=10.0, repetitions=1
    )
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3"
    assert any("not eligible: repetitions 2 > 0" in r for r in choice.reasons)


def test_a_wall_time_tie_goes_to_the_smaller_peak_memory(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0], seconds=50.0) + _runs(nine, [1.0], seconds=50.0)
    rows = [_peak("27 @3", M, D, 3, 20.0), _peak("9", M9, "", 0, 8.0)]
    choice = full_decision(user, [cur, nine], runs, rows, runs_per_task=1)
    assert choice is not None and choice.label == "9"
    assert "a tie goes to the smaller peak memory" in "\n".join(choice.reasons)
    # An exact tie in both keeps the reference.
    rows = [_peak("27 @3", M, D, 3, 8.0), _peak("9", M9, "", 0, 8.0)]
    tie = full_decision(user, [cur, nine], runs, rows, runs_per_task=1)
    assert tie is not None and tie.label == "27 @3"


def test_without_runs_of_the_current_arm_the_first_measured_arm_is_the_reference(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    four = _arm(user, "4", drafter="", block=0, model="mlx-community/Qwen3.5-4B-MLX-4bit")
    runs = _runs(nine, [0.8], seconds=50.0) + _runs(four, [0.8], seconds=20.0)
    choice = full_decision(user, [cur, nine, four], runs, [], runs_per_task=1)
    assert choice is not None and choice.reference_label == "9" and choice.label == "4"
    assert "reference: 9 (the fastest arm of the largest fitting tier" in choice.reasons[0]
    assert full_decision(user, [cur], [], [], runs_per_task=1) is None


def test_the_winner_stage_judges_each_extra_arm_against_the_winner(tmp_path):
    user = _user(tmp_path)
    winner = _arm(user, "27 @3", current=True)
    int8 = dataclasses.replace(
        winner,
        label="27 @3 + int8 prefill",
        config=dataclasses.replace(user, int8_prefill=True),
        current=False,
        int8_prefill=True,
    )
    greedy = dataclasses.replace(
        winner,
        label="27 @3 greedy",
        config=dataclasses.replace(user, temperature=0.0),
        current=False,
        greedy=True,
    )
    runs = (
        _runs(winner, [1.0, 1.0], seconds=100.0)
        + _runs(int8, [1.0, 0.96], seconds=80.0)
        + _runs(greedy, [1.0, 1.0], seconds=70.0)
    )
    choice = full_decision(
        user, [winner, int8, greedy], runs, [], runs_per_task=2, reference=winner
    )
    assert choice is not None and choice.label == "27 @3 greedy"
    assert choice.changes == {"model": {"temperature": 0.0}}
    assert "reference: 27 @3 (the model stage's winner)" in choice.reasons[0]
    runs = _runs(winner, [1.0, 1.0], seconds=100.0) + _runs(int8, [1.0, 0.96], seconds=80.0)
    choice = full_decision(user, [winner, int8], runs, [], runs_per_task=2, reference=winner)
    assert choice is not None and choice.changes == {"model": {"int8_prefill": True}}
    runs = _runs(winner, [1.0, 1.0], seconds=100.0) + _runs(int8, [1.0, 0.88], seconds=80.0)
    choice = full_decision(user, [winner, int8], runs, [], runs_per_task=2, reference=winner)
    assert choice is not None and choice.changes == {}
    assert full_decision(user, [winner, int8], [], [], runs_per_task=2, reference=winner) is None


def test_a_full_choice_over_another_model_writes_its_drafter_block_and_fit_window(tmp_path):
    user = _user(tmp_path, gateway_enabled=True)
    cur = _arm(user, "27 @3", current=True, fit_window=131072, fit_gateway_window=131072)
    nine = _arm(
        user,
        "9 + d @2",
        drafter="z/9d",
        block=2,
        model=M9,
        window=65536,
        gateway_window=65536,
        fit_window=65536,
        fit_gateway_window=65536,
    )
    runs = _runs(cur, [1.0], seconds=100.0) + _runs(nine, [1.0], seconds=50.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None
    # The worker's window is not written: the arm's fit (65536) is not below
    # the configured 32768, and a window is only ever lowered. The gateway's
    # is, from 131072.
    assert choice.changes == {
        "model": {"id": M9, "speculative_draft_id": "z/9d", "speculative_block_size": 2},
        "gateway": {"max_context_tokens": 65536},
    }


def test_an_arm_exactly_at_the_margin_is_eligible(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [0.75] * 3, seconds=100.0) + _runs(nine, [0.70] * 3, seconds=50.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None and choice.label == "9"
    assert not any("not eligible" in r for r in choice.reasons)


def test_completed_runs_at_the_floor_and_equal_repetitions_are_eligible(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0] * 4, seconds=100.0, repetitions=1) + [
        _run(nine, "t0", grade=1.0, seconds=10.0, repetitions=1),
        _run(nine, "t1", grade=1.0, seconds=10.0, repetitions=1),
        _run(nine, "t2", grade=1.0, seconds=10.0, repetitions=1, state="failed"),
        _run(nine, "t3", grade=1.0, seconds=10.0, repetitions=1, state="failed"),
    ]
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "9"
    line = next(r for r in choice.reasons if r.startswith("9:"))
    assert line.endswith("-> eligible")


def test_a_reference_with_no_completed_run_is_said_so(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [0.6, 0.6], seconds=100.0, state="failed") + _runs(nine, [1.0], seconds=50.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None and choice.label == "9"
    assert any("note: the reference completed no run" in r for r in choice.reasons)


def test_a_graded_but_failed_reference_still_constrains_the_grade(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [0.6, 0.6], seconds=100.0, state="failed") + _runs(nine, [0.5], seconds=10.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None and choice.label == "27 @3"
    assert any("note: the reference completed no run" in r for r in choice.reasons)
    line = next(r for r in choice.reasons if r.startswith("9:"))
    assert line.endswith("not eligible: grade 0.50 < 0.55")
