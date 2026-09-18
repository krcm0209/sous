import argparse
from datetime import date

from sous.config import SousConfig, load_config
from sous.tune import main
from sous.tune.bench import BenchRow
from sous.tune.candidates import Candidate, Table
from sous.tune.candidates import describe as real_describe
from sous.tune.daemon import Readiness
from sous.tune.hardware import Hardware
from tests import tune_fixtures as fx

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _args(**over):
    base = dict(quick=True, models=None, repeat=1, resume=None, yes=False, apply=False)
    base.update(over)
    return argparse.Namespace(**base)


def _hardware(tmp_path, working_set=fx.M5_PRO_WORKING_SET):
    return Hardware(
        chip="M5 Pro",
        memory_bytes=64 * 2**30,
        working_set_bytes=working_set,
        architecture="applegpu_g17s",
        nax=True,
        nax_reason=None,
        macos="26.6",
        versions={},
        hub_cache=str(tmp_path / "hub"),
        disk_free_bytes=10**12,
    )


def _table():
    return Table(
        checked=date.today(),
        validated_with="x",
        trusted_orgs=("mlx-community",),
        publishers=("Qwen",),
        candidates=(
            Candidate(M, "27b-dense", (D,)),
            Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ()),
        ),
    )


def _configs():
    return {
        M: fx.qwen_27b(),
        D: fx.dflash2_27b(),
        "mlx-community/Qwen3.5-9B-MLX-4bit": fx.qwen_9b(),
    }


def _sizes():
    return {M: 16_100_000_000, D: 3_850_000_000, "mlx-community/Qwen3.5-9B-MLX-4bit": 6 * 10**9}


def _bench(scores):
    def bench(arm, **kw):
        s = scores.get(arm.label, 10.0)
        return BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=True,
            error=None,
            load_seconds=1.0,
            prefill_tps_2k=400.0,
            prefill_tps_16k=350.0,
            decode_tps_1k=s + 5,
            decode_tps_16k=s,
            ttft_seconds=1.0,
            peak_memory_bytes=1,
            spread=0.0,
        )

    return bench


def _deps(tmp_path, *, scores, cached=(), answers=(), ready=None, isatty=True):
    ready = ready if ready is not None else Readiness(True, "no daemon")
    configs, sizes = _configs(), _sizes()
    it = iter(answers)
    downloaded = []
    return dict(
        detect=lambda: _hardware(tmp_path),
        load_table=lambda: _table(),
        describe=lambda mid, config_fn=None, size_fn=None: real_describe(
            mid, config_fn=lambda m: configs[m], size_fn=lambda m: sizes[m]
        ),
        ready=lambda port: ready,
        bench=_bench(scores),
        is_cached=lambda rid: rid in cached,
        size=lambda rid: sizes[rid],
        download=downloaded.append,
        ask=lambda prompt: next(it, "n"),
        isatty=lambda: isatty,
        managed=lambda: False,
        out=print,
    ), downloaded


def _cfg(tmp_path, **over):
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\nspeculative_block_size = 3\n")
    return SousConfig(data_dir=tmp_path, config_path=p, **over)


def test_a_quick_run_measures_the_fitting_arms_and_applies_on_yes(tmp_path, capsys):
    deps, downloaded = _deps(
        tmp_path,
        scores={
            "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0,
            "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3": 18.0,
        },
        cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"),
        answers=("y",),
    )
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "## Throughput" in out and "quality untested" in out
    assert "+speculative_block_size = 2" in out
    assert "Apply these changes" in out
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2
    assert "# mine" in (tmp_path / "config.toml").read_text()
    assert list((tmp_path / "tune").iterdir())  # the run dir with results and the report
    assert downloaded == []


def test_downloads_are_asked_one_by_one_and_a_refused_model_loses_its_arms(tmp_path, capsys):
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("n",))
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "Download #1" in out and "Qwen3.5-9B-MLX-4bit" in out
    assert downloaded == [] and "Qwen3.5-9B-MLX-4bit" not in out.split("## Throughput")[1]


def test_an_approved_download_is_fetched_before_any_measurement(tmp_path, capsys):
    order = []
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("y",))
    real_bench = deps["bench"]
    deps["bench"] = lambda arm, **kw: (order.append("bench"), real_bench(arm, **kw))[1]
    deps["download"] = lambda rid: order.append(f"download {rid}")
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert order[0] == "download mlx-community/Qwen3.5-9B-MLX-4bit"


def test_a_busy_daemon_refuses_before_anything_is_measured(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, ready=Readiness(False, "the daemon is busy: 1 task(s)"))
    called = []
    deps["bench"] = lambda arm, **kw: called.append(arm)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 2
    assert "busy" in capsys.readouterr().out and called == []


def test_a_model_that_does_not_fit_gets_guidance_and_no_diff(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    deps["detect"] = lambda: _hardware(tmp_path, working_set=fx.M2_AIR_WORKING_SET)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "refused" in out and "cannot recommend" in out and "Apply these changes" not in out


def test_without_a_tty_nothing_is_applied_unless_apply_is_passed(tmp_path, capsys):
    scores = {"Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0}
    deps, _ = _deps(
        tmp_path, scores=scores, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"), isatty=False
    )
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 3
    assert main(_args(apply=True), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2


def test_resume_skips_arms_already_in_the_run(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    seen = []
    real = deps["bench"]
    deps["bench"] = lambda arm, **kw: (seen.append(arm.label), real(arm, **kw))[1]
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    assert main(_args(resume=run_id), config=_cfg(tmp_path), **deps) == 0
    assert seen == [] and first


def test_a_full_run_is_refused_in_this_version(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={})
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 2
    assert "--quick" in capsys.readouterr().out
