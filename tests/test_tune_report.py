import os
from pathlib import Path

import pytest

from sous.config import load_config
from sous.engine.int8prefill import Availability
from sous.tune.arms import Refusal
from sous.tune.bench import BenchRow
from sous.tune.candidates import describe
from sous.tune.decide import QuickChoice
from sous.tune.hardware import detect
from sous.tune.report import apply_changes, config_diff, render_report, restart_note
from tests import tune_fixtures as fx

M = "mlx-community/Qwen3.8-27B-4bit"


def _hardware(tmp_path):
    return detect(
        device_info=lambda: {
            "device_name": "Apple M5 Pro",
            "memory_size": 64 * 2**30,
            "max_recommended_working_set_size": fx.M5_PRO_WORKING_SET,
            "architecture": "applegpu_g17s",
        },
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda p: (1, 2, 400 * 2**30),
        availability=lambda: Availability(True),
        version=lambda n: "1.0",
        hub_cache=str(tmp_path),
    )


def _row(label, model=M, ok=True, error=None, released=True):
    return BenchRow(
        label=label,
        model_id=model,
        drafter_id="d",
        block_size=3,
        window=131072,
        ok=ok,
        error=error if ok else "RuntimeError: boom",
        load_seconds=5.2,
        prefill_tps_2k=479.4,
        prefill_tps_16k=401.0,
        decode_tps_1k=25.6,
        decode_tps_16k=18.3,
        ttft_seconds=1.1,
        peak_memory_bytes=20 * 2**30,
        spread=0.03,
        released=released,
    )


def test_the_report_has_every_section_and_labels_other_models_quality_untested(tmp_path):
    cps = {
        M: describe(M, config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 16_100_000_000),
        "o/9b": describe("o/9b", config_fn=lambda m: fx.qwen_9b(), size_fn=lambda m: 6 * 10**9),
    }
    choice = QuickChoice(
        label="cur",
        model_id=M,
        drafter_id="d",
        block_size=3,
        window=131072,
        gateway_window=None,
        changes={},
        reasons=["the current arm is already the fastest measured"],
    )
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=3,
        checkpoints=cps,
        refusals=[Refusal("o/big", "", "weights 30.0 GB ... below the floor")],
        rows=[_row("cur"), _row("nine", model="o/9b"), _row("bad", ok=False)],
        choice=choice,
        current_model=M,
    )
    for heading in ("Hardware", "Candidates", "Throughput", "Choice"):
        assert heading in text
    assert "Apple M5 Pro" in text and "51.8 GiB" in text
    assert "16.1 GB" in text and "64 KiB/token" in text
    assert "o/big" in text and "below the floor" in text
    assert "479" in text and "18.3" in text and "RuntimeError: boom" in text
    assert "quality untested" in text.split("nine")[1].splitlines()[0]
    assert "already the fastest" in text
    assert "stale" not in text


def test_nax_reason_none_renders_without_a_parenthetical(tmp_path):
    hw = detect(
        device_info=lambda: {
            "device_name": "Apple M5 Pro",
            "memory_size": 64 * 2**30,
            "max_recommended_working_set_size": fx.M5_PRO_WORKING_SET,
            "architecture": "applegpu_g17s",
        },
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda p: (1, 2, 400 * 2**30),
        availability=lambda: Availability(False),  # reason omitted, defaults to None
        version=lambda n: "1.0",
        hub_cache=str(tmp_path),
    )
    text = render_report(
        hardware=hw,
        table_age_days=1,
        checkpoints={},
        refusals=[],
        rows=[],
        choice=None,
        current_model=M,
    )
    assert "NAX tensor units: no" in text
    assert "no (None)" not in text


def test_the_context_legend_lives_under_throughput_not_candidates(tmp_path):
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints={},
        refusals=[],
        rows=[],
        choice=None,
        current_model=M,
    )
    legend = "(prefill/decode at 2K/16K and 1K/16K tokens of context)"
    candidates_idx = text.index("## Candidates")
    throughput_idx = text.index("## Throughput")
    legend_idx = text.index(legend)
    assert candidates_idx < throughput_idx < legend_idx


def test_int8_excluded_layers_are_named_in_the_candidate_line(tmp_path):
    cps = {
        M: describe(
            M,
            config_fn=lambda m: fx.qwen_27b(fx.oq4_quantization()),
            size_fn=lambda m: 16_100_000_000,
        ),
    }
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints=cps,
        refusals=[],
        rows=[],
        choice=None,
        current_model=M,
    )
    assert "161 layers excluded" in text


def test_a_drafter_checkpoint_renders_its_quantized_resident_size(tmp_path):
    cps = {
        "z-lab/Qwen3.8-27B-DFlash2": describe(
            "z-lab/Qwen3.8-27B-DFlash2",
            config_fn=lambda m: fx.dflash2_27b(),
            size_fn=lambda m: 3_850_000_000,
        ),
    }
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints=cps,
        refusals=[],
        rows=[],
        choice=None,
        current_model=M,
    )
    assert "z-lab/Qwen3.8-27B-DFlash2" in text and "drafter" in text
    assert "1.1 GB resident after 4-bit quantization" in text
    assert "KiB/token" not in text.split("## Throughput")[0]  # no KV line for a drafter


def test_a_stale_table_and_a_missing_choice_are_said_plainly(tmp_path):
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=120,
        checkpoints={},
        refusals=[],
        rows=[],
        choice=None,
        current_model=M,
    )
    assert "stale" in text and "--models" in text and "--discover" not in text
    assert "cannot recommend" in text and M in text


def test_config_diff_preserves_comments_and_untouched_keys(tmp_path):
    p = tmp_path / "config.toml"
    original = (
        '# mine\n[model]\nid = "x/y"   # keep\nspeculative_block_size = 3\n'
        "\n[gateway]\nenabled = true\n"
    )
    p.write_text(original)
    new, diff = config_diff(
        p, {"model": {"speculative_block_size": 2}, "gateway": {"max_context_tokens": 65536}}
    )
    assert "# mine" in new and 'id = "x/y"   # keep' in new
    assert "speculative_block_size = 2" in new and "max_context_tokens = 65536" in new
    assert "-speculative_block_size = 3" in diff and "+speculative_block_size = 2" in diff
    assert "+max_context_tokens = 65536" in diff
    assert load_config(p).speculative_block_size == 3  # nothing written yet


def test_config_diff_starts_from_an_empty_document_when_the_file_is_absent(tmp_path):
    p = tmp_path / "config.toml"
    new, diff = config_diff(p, {"model": {"speculative_draft_id": ""}})
    assert 'speculative_draft_id = ""' in new and diff.startswith("---")


def test_apply_writes_a_backup_then_the_new_text(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nspeculative_block_size = 3\n")
    new, _ = config_diff(p, {"model": {"speculative_block_size": 2}})
    backup = apply_changes(p, new, "20260917-101500")
    assert backup is not None
    assert backup == tmp_path / "config.toml.bak-tune-20260917-101500"
    assert backup.read_text() == "[model]\nspeculative_block_size = 3\n"
    assert load_config(p).speculative_block_size == 2
    assert apply_changes(tmp_path / "fresh.toml", new, "x") is None


def test_apply_leaves_the_original_untouched_if_the_write_crashes_mid_way(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    original = "[model]\nspeculative_block_size = 3\n"
    p.write_text(original)
    new, _ = config_diff(p, {"model": {"speculative_block_size": 2}})
    real_write_text = Path.write_text

    def crashing_write_text(self, *args, **kwargs):
        if self.name.startswith(f"{p.name}.tmp-"):
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", crashing_write_text)
    with pytest.raises(OSError):
        apply_changes(p, new, "20260918-x")
    assert p.read_text() == original  # the swap never ran; nothing was truncated


def test_restart_note_fires_for_any_change_and_names_the_keys():
    assert restart_note({}, managed=False) is None
    unmanaged = restart_note(
        {"model": {"speculative_block_size": 2, "max_context_tokens": 65536}}, managed=False
    )
    assert unmanaged is not None
    assert "speculative_block_size" in unmanaged and "max_context_tokens" in unmanaged
    assert "sous stop" in unmanaged and "sous serve" in unmanaged
    managed = restart_note({"model": {"speculative_draft_id": ""}}, managed=True)
    assert managed is not None
    assert "speculative_draft_id" in managed
    assert f"launchctl kickstart -k gui/{os.getuid()}/com.sous.daemon" in managed


def test_a_finished_row_whose_weights_stayed_resident_says_so_in_the_report(tmp_path):
    """The bench leaves `ok` True on a measurement that finished and only
    the teardown of which failed; the report must still carry the reason,
    or a resumed run reads the row as clean."""
    text = render_report(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints={},
        refusals=[],
        rows=[_row("cur", error="unload refused: a generation is in flight", released=False)],
        choice=None,
        current_model=M,
    )
    line = text.split("## Throughput")[1]
    assert "18.3" in line and "unload refused: a generation is in flight" in line


def test_apply_keeps_the_configs_mode_on_the_file_and_its_backup(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nspeculative_block_size = 3\n")
    p.chmod(0o600)
    new, _ = config_diff(p, {"model": {"speculative_block_size": 2}})
    backup = apply_changes(p, new, "20260918-x")
    assert backup is not None
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert oct(backup.stat().st_mode & 0o777) == "0o600"
    assert load_config(p).speculative_block_size == 2
