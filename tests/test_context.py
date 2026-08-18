"""Auto context sizing: KV math, headroom policy, and the per-task decision.

Everything here runs with fakes — no mlx, no psutil, no model downloads."""

import warnings
from pathlib import Path

from sous.config import SousConfig
from sous.context import (
    ContextDecision,
    MemorySnapshot,
    auto_context_tokens,
    decide_context,
    kv_bytes_per_token,
    native_max_tokens,
)

GIB = 1 << 30

# Shaped like Qwen3.8-27B's real config.json: a VLM (language shape nested
# under text_config) with hybrid attention — only the full_attention layers
# accumulate KV; the linear_attention layers hold constant-size state.
HYBRID = {
    "text_config": {
        "num_hidden_layers": 8,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "num_attention_heads": 24,
        "hidden_size": 5120,
        "max_position_embeddings": 262144,
    }
}


def test_kv_bytes_counts_only_full_attention_layers():
    """Treating all layers as full attention would overestimate the hybrid
    default model's KV cost 4x and shrink the window for nothing."""
    # 2 (K+V) x 2 full-attn layers x 4 kv heads x 256 head_dim x 2 bytes
    assert kv_bytes_per_token(HYBRID) == 2 * 2 * 4 * 256 * 2


def test_kv_bytes_plain_gqa_uses_all_layers():
    cfg = {"num_hidden_layers": 28, "num_key_value_heads": 4, "head_dim": 128}
    assert kv_bytes_per_token(cfg) == 2 * 28 * 4 * 128 * 2


def test_kv_bytes_derives_head_dim_when_absent():
    cfg = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "num_attention_heads": 8,
    }
    assert kv_bytes_per_token(cfg) == 2 * 2 * 2 * (64 // 8) * 2


def test_kv_bytes_none_when_shape_unknown():
    assert kv_bytes_per_token({}) is None
    assert kv_bytes_per_token({"text_config": {"num_hidden_layers": 4}}) is None
    # All-linear: no per-token KV growth we know how to size — refuse rather
    # than divide by zero or claim infinite context.
    assert (
        kv_bytes_per_token(
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "layer_types": ["linear_attention"] * 2,
            }
        )
        is None
    )


def test_native_max_nested_and_flat():
    assert native_max_tokens(HYBRID) == 262144
    assert native_max_tokens({"max_position_embeddings": 4096}) == 4096
    assert native_max_tokens({}) is None


def _auto(
    fraction: float = 0.8,
    min_tokens: int = 8192,
    bytes_per_token: int = 65536,
    native_max: int = 262144,
    working_set: int = 52 * GIB,
    active: int = 29 * GIB,
    cache: int = 0,
    available: int = 30 * GIB,
) -> ContextDecision:
    return auto_context_tokens(
        fraction=fraction,
        min_tokens=min_tokens,
        bytes_per_token=bytes_per_token,
        native_max=native_max,
        working_set=working_set,
        active=active,
        cache=cache,
        available=available,
    )


def test_auto_uses_min_of_metal_and_system_headroom():
    # Metal headroom 4 GiB beats system 8 GiB; 0.5 x 4 GiB / 64 KiB = 32768.
    d = _auto(working_set=33 * GIB, active=29 * GIB, available=8 * GIB, fraction=0.5)
    assert d.tokens == 32768


def test_auto_counts_mlx_cache_as_reclaimable_system_memory():
    """mx.get_cache_memory() is freed-but-retained memory owned by this
    process — psutil sees it as consumed, but clear_cache() gets it back."""
    a = _auto(working_set=100 * GIB, active=0, available=2 * GIB, cache=2 * GIB, fraction=0.5)
    b = _auto(working_set=100 * GIB, active=0, available=4 * GIB, cache=0, fraction=0.5)
    assert a.tokens == b.tokens == 32768


def test_auto_clamps_to_native_max():
    d = _auto(working_set=500 * GIB, active=0, available=500 * GIB, fraction=1.0)
    assert d.tokens == 262144


def test_auto_clamps_to_floor_when_memory_is_tight():
    # 0.1 x 1 GiB / 64 KiB = 1638 tokens — below any usable window.
    d = _auto(working_set=30 * GIB, active=29 * GIB, available=100 * GIB, fraction=0.1)
    assert d.tokens == 8192
    assert "floor" in d.reason


def test_auto_negative_headroom_still_returns_floor():
    d = _auto(working_set=20 * GIB, active=29 * GIB)
    assert d.tokens == 8192


def test_auto_rounds_down_to_kv_cache_step():
    # mlx-lm's KVCache grows in 256-token steps; an unaligned window wastes
    # a partial step. Odd headroom must still yield an aligned window.
    d = _auto(available=(10 * GIB) + 12345678, working_set=100 * GIB, active=0)
    assert d.tokens % 256 == 0
    assert d.tokens > 8192


def _cfg(tmp_path: Path, **over) -> SousConfig:
    return SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "c.toml", **over)


def test_decide_fixed_mode_never_probes_anything(tmp_path: Path):
    def boom(*a):
        raise AssertionError("probed model/memory in fixed mode")

    d = decide_context(_cfg(tmp_path), model_config_fn=boom, memory_fn=boom)
    assert d == ContextDecision(32768, "fixed")


def test_decide_auto_happy_path(tmp_path: Path):
    cfg = _cfg(tmp_path, context_mode="auto", context_fraction=0.5, context_min_tokens=1024)
    mem = MemorySnapshot(
        working_set=33 * GIB, active=29 * GIB, cache=0, available=8 * GIB
    )  # 4 GiB metal headroom
    d = decide_context(cfg, model_config_fn=lambda mid: HYBRID, memory_fn=lambda: mem)
    # 0.5 x 4 GiB / 8 KiB-per-token (tiny HYBRID) = 262144, clamped to native max
    assert d.tokens == 262144
    assert "auto" in d.reason


def test_decide_auto_falls_back_to_fixed_on_any_failure(tmp_path: Path):
    """A sizing bug must degrade to the configured fixed window with a
    warning — never break delegation."""
    cfg = _cfg(tmp_path, context_mode="auto")

    def boom(mid):
        raise RuntimeError("hf is down")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = decide_context(cfg, model_config_fn=boom, memory_fn=lambda: None)
    assert d.tokens == cfg.max_context_tokens
    assert "fixed" in d.reason
    assert any("auto" in str(w.message) for w in caught)


def test_decide_auto_unknown_model_shape_falls_back(tmp_path: Path):
    cfg = _cfg(tmp_path, context_mode="auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = decide_context(cfg, model_config_fn=lambda mid: {}, memory_fn=lambda: None)
    assert d.tokens == cfg.max_context_tokens
    assert caught
