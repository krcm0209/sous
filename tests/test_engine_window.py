"""Context-window arithmetic: KV bytes per token and native max tokens from
a model's config.json. Runs with fakes only — no mlx, no model downloads."""

from sous.engine.window import kv_bytes_per_token, native_max_tokens

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
