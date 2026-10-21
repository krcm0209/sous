import dataclasses
from datetime import date

from sous.config import MIN_CONTEXT_TOKENS
from sous.tune.candidates import (
    Checkpoint,
    describe,
    drafter_compatible,
    fit,
    load_table,
    summarize_quantization,
)
from tests import tune_fixtures as fx


def test_the_shipped_table_loads_and_names_the_default_model_first():
    table = load_table()
    assert isinstance(table.checked, date)
    assert table.candidates[0].id == "mlx-community/Qwen3.8-27B-4bit"
    assert table.candidates[0].drafters == ("z-lab/Qwen3.8-27B-DFlash2",)
    assert "mlx-community" in table.trusted_orgs and "Qwen" in table.publishers
    tiers = {c.tier for c in table.candidates}
    assert {"27b-dense", "35b-moe", "9b", "4b", "2b"} <= tiers


def test_quantization_summary_of_a_plain_4bit_checkpoint():
    q = summarize_quantization(fx.qwen_27b())
    assert (q.mode, q.bits, q.group_size, q.overrides) == ("affine", 4, 64, 0)
    assert q.verifier_fast and q.verifier_slow_layers == 0
    assert q.int8_routable and q.int8_excluded_layers == 0


def test_quantization_summary_of_oq4_keeps_the_fast_verifier_but_counts_the_6bit_layer():
    q = summarize_quantization(fx.qwen_27b(fx.oq4_quantization()))
    assert q.overrides == 161
    assert q.verifier_fast and q.verifier_slow_layers == 1
    assert q.int8_routable and q.int8_excluded_layers == 161


def test_quantization_summary_of_mxfp4_and_6bit_and_unquantized():
    mx4 = summarize_quantization(fx.qwen_27b({"group_size": 32, "bits": 4, "mode": "mxfp4"}))
    assert mx4.verifier_fast is False and mx4.int8_routable is False
    six = summarize_quantization(fx.qwen_27b({"group_size": 64, "bits": 6, "mode": "affine"}))
    assert six.verifier_fast is False
    none = summarize_quantization({"model_type": "qwen3_5"})
    assert none.mode is None and none.verifier_fast is False


def test_describe_reads_bytes_shape_and_kv_cost_through_its_injection_points():
    cp = describe(
        "mlx-community/Qwen3.8-27B-4bit",
        config_fn=lambda mid: fx.qwen_27b(),
        size_fn=lambda mid: 16_100_000_000,
    )
    assert cp.backend == "vlm" and cp.model_type == "qwen3_5"
    assert cp.bytes == 16_100_000_000
    assert cp.kv_bytes_per_token == 2 * 16 * 4 * 256 * 2  # 16 attention layers, 64 KiB
    assert cp.native_max_tokens == 262144
    assert cp.hidden_size == 5120 and cp.num_layers == 64 and cp.num_target_layers is None


def test_describe_of_a_drafter_records_its_target_layer_count():
    cp = describe(
        "z-lab/Qwen3.8-27B-DFlash2",
        config_fn=lambda m: fx.dflash2_27b(),
        size_fn=lambda m: 3_850_000_000,
    )
    assert cp.num_target_layers == 64 and cp.hidden_size == 5120
    assert cp.kv_bytes_per_token is not None  # a drafter's own KV is real but small


def test_drafter_compatibility_matches_hidden_size_and_layer_count():
    target = describe("t", config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 1)
    good = describe("d", config_fn=lambda m: fx.dflash2_27b(), size_fn=lambda m: 1)
    wrong = describe("w", config_fn=lambda m: fx.dflash_9b(), size_fn=lambda m: 1)
    assert drafter_compatible(target, good) == (True, "")
    ok, why = drafter_compatible(target, wrong)
    assert ok is False and "hidden size" in why


def _cp(bytes_, kv=65536, native=262144):
    return Checkpoint(
        id="m",
        model_type="qwen3_5",
        backend="vlm",
        bytes=bytes_,
        kv_bytes_per_token=kv,
        native_max_tokens=native,
        hidden_size=1,
        num_layers=1,
        num_target_layers=None,
        quant=summarize_quantization(fx.qwen_27b()),
    )


def test_fit_keeps_the_configured_window_when_everything_fits():
    drafter = _cp(3_850_000_000, kv=2 * 5 * 8 * 128 * 2)
    f = fit(
        _cp(16_100_000_000),
        drafter,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M5_PRO_WORKING_SET,
    )
    assert f.fits and f.window == 131072
    assert "131072" in f.detail and "GiB" in f.detail


def test_fit_lowers_the_window_to_a_256_multiple_before_giving_up():
    f = fit(
        _cp(6_000_000_000, kv=32768),
        None,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M2_AIR_WORKING_SET,
    )
    assert f.fits and f.window % 256 == 0 and 8192 <= f.window < 131072
    assert f.needed_bytes <= fx.M2_AIR_WORKING_SET


def test_fit_refuses_a_model_that_cannot_reach_the_floor_and_prints_the_arithmetic():
    f = fit(
        _cp(16_100_000_000),
        None,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M2_AIR_WORKING_SET,
    )
    assert f.fits is False and f.window == 8192
    assert "16.1 GB" in f.detail or "15.0 GiB" in f.detail
    assert "10.7 GiB" in f.detail


def test_fit_refuses_when_a_size_is_unknown():
    f = fit(_cp(None), None, window=8192, floor=8192, working_set_bytes=10**12)
    assert f.fits is False and "unknown" in f.detail


def test_fit_refuses_when_the_drafters_size_is_unknown():
    drafter = _cp(None, kv=2 * 5 * 8 * 128 * 2)
    f = fit(_cp(16_100_000_000), drafter, window=131072, floor=8192, working_set_bytes=10**12)
    assert f.fits is False and "drafter size unknown" in f.detail


def test_the_fit_floor_is_claude_codes_minimum():
    assert MIN_CONTEXT_TOKENS == 48 * 1024


def test_drafter_compatibility_refuses_a_vocabulary_mismatch():
    """mlx-vlm's own validation compares vocabularies too, and the engine
    survives a drafter that fails it by running undrafted — a bench row
    would then carry a drafter it never had."""
    target = describe(
        "t",
        config_fn=lambda m: {
            **fx.qwen_27b(),
            "text_config": {**fx.qwen_27b()["text_config"], "vocab_size": 248320},
        },
        size_fn=lambda m: 1,
    )
    good = describe(
        "d", config_fn=lambda m: {**fx.dflash2_27b(), "vocab_size": 248320}, size_fn=lambda m: 1
    )
    wrong = describe(
        "w", config_fn=lambda m: {**fx.dflash2_27b(), "vocab_size": 151936}, size_fn=lambda m: 1
    )
    assert target.vocab_size == 248320 and good.vocab_size == 248320
    assert drafter_compatible(target, good) == (True, "")
    ok, why = drafter_compatible(target, wrong)
    assert ok is False and "vocabulary" in why
    # Unknown on either side is not a mismatch.
    assert drafter_compatible(
        target, describe("u", config_fn=lambda m: fx.dflash2_27b(), size_fn=lambda m: 1)
    ) == (True, "")


def test_a_lowered_window_says_what_it_costs_the_prompt_cache():
    f = fit(
        _cp(6_000_000_000, kv=32768),
        None,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M2_AIR_WORKING_SET,
    )
    assert f.fits and "prompt-cache" in f.detail


def test_describe_keeps_the_config_it_read_for_checks_that_read_more_of_it():
    cp = describe("t", config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 1)
    assert cp.config == fx.qwen_27b()
    # Equality and the hash stay about the summarised fields.
    assert cp == dataclasses.replace(cp, config={})
    assert hash(cp) == hash(dataclasses.replace(cp, config={}))


# The default 27B with its DFlash2 drafter at 131072 on the M5 Pro: headroom
# 51.8 GiB - 16.1 GB - 3.85 GB / 3.5, the window's KV reserve 8 GiB, and the
# daemon's 2 GiB slack leave 22.0 GiB of prompt-cache budget beside a 4.09 GB
# copy and 11.9 GiB beside a 15 GB one.
_SMALL_COPY = 4_090_000_000
_LARGE_COPY = 15_000_000_000


def _fit_27b(**tile):
    return fit(
        _cp(16_100_000_000),
        _cp(3_850_000_000, kv=2 * 5 * 8 * 128 * 2),
        window=131072,
        floor=8192,
        working_set_bytes=fx.M5_PRO_WORKING_SET,
        **tile,
    )


def test_fit_reports_the_projection_tiles_scope_without_counting_the_copy():
    base = _fit_27b()
    assert "projection tile" not in base.detail
    small = _fit_27b(tile_copy_bytes=_SMALL_COPY, tile_config=True)
    large = _fit_27b(tile_copy_bytes=_LARGE_COPY, tile_config=True)
    # The copy is elective: the guard skips it rather than shrink the window.
    for f in (small, large):
        assert (f.fits, f.window, f.needed_bytes) == (base.fits, base.window, base.needed_bytes)
    assert "projection tile scope=all copy=4.1 GB (budget left 22.01 GiB)" in small.detail
    assert "projection tile scope=mlp (copy would leave 11.85 GiB" in large.detail


def test_fit_reports_the_scope_the_config_and_an_explicit_budget_decide():
    off = _fit_27b(tile_copy_bytes=_SMALL_COPY)  # the tile switched off
    assert "projection tile" not in off.detail
    assert (
        "projection tile scope=mlp"
        in _fit_27b(tile_copy_bytes=_SMALL_COPY, tile_config="mlp").detail
    )
    assert (
        "projection tile scope=all"
        in _fit_27b(tile_copy_bytes=_LARGE_COPY, tile_config="all").detail
    )
    fits = _fit_27b(tile_copy_bytes=_SMALL_COPY, tile_config=True, cache_budget=20 * 2**30)
    tight = _fit_27b(tile_copy_bytes=_SMALL_COPY, tile_config=True, cache_budget=24 * 2**30)
    assert "projection tile scope=all" in fits.detail
    assert "projection tile scope=mlp" in tight.detail
    assert "configured budget 24.00 GiB does not fit beside the copy" in tight.detail


def test_fit_reports_the_copy_the_prompt_cache_off_allows():
    """The copy that would leave the automatic budget under the rule is made with
    the prompt cache off: nothing would hold that budget."""
    on = _fit_27b(tile_copy_bytes=_LARGE_COPY, tile_config=True)
    off = _fit_27b(tile_copy_bytes=_LARGE_COPY, tile_config=True, cache_enabled=False)
    assert "projection tile scope=mlp (copy would leave 11.85 GiB" in on.detail
    assert (
        "projection tile scope=all copy=15.0 GB (prompt cache off; the copy fits beside "
        "the reserve; budget left 11.85 GiB)" in off.detail
    )
    assert (off.fits, off.window, off.needed_bytes) == (on.fits, on.window, on.needed_bytes)


def test_fit_reports_no_scope_for_a_model_it_refuses():
    f = fit(
        _cp(16_100_000_000),
        None,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M2_AIR_WORKING_SET,
        tile_copy_bytes=_SMALL_COPY,
        tile_config=True,
    )
    assert f.fits is False
    assert "projection tile" not in f.detail


def _fit_lowered(**tile):
    return fit(
        _cp(6_000_000_000, kv=32768),
        None,
        window=131072,
        floor=8192,
        working_set_bytes=fx.M2_AIR_WORKING_SET,
        **tile,
    )


def test_a_lowered_window_stays_where_the_fit_put_it_beside_the_copy():
    base = _fit_lowered()
    tiled = _fit_lowered(tile_copy_bytes=_SMALL_COPY, tile_config=True)
    assert base.window < 131072
    assert (tiled.window, tiled.needed_bytes) == (base.window, base.needed_bytes)
    assert "projection tile scope=mlp" in tiled.detail
