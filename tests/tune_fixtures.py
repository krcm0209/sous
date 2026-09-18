"""config.json shapes of the checkpoints the tune reasons about, reduced to
the keys sous reads. Hybrid layer lists follow Qwen3.5: every fourth layer
is full attention."""


def _layer_types(n: int) -> list[str]:
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]


def qwen_27b(quantization: dict | None = None, model_type: str = "qwen3_5") -> dict:
    return {
        "model_type": model_type,
        "vision_config": {"model_type": "qwen3_5"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "num_attention_heads": 24,
            "head_dim": 256,
            "hidden_size": 5120,
            "max_position_embeddings": 262144,
            "layer_types": _layer_types(64),
        },
        "quantization": quantization or {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def qwen_9b() -> dict:
    return {
        "model_type": "qwen3_5",
        "vision_config": {"model_type": "qwen3_5"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "num_attention_heads": 16,
            "head_dim": 256,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "layer_types": _layer_types(32),
        },
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def oq4_quantization() -> dict:
    """A 4-bit base with sensitivity boosts: 160 layers at 5-bit and one at
    6-bit, the shape mlx-community/Qwen3.8-27B-oQ4 ships."""
    q: dict = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(160):
        q[f"language_model.model.layers.{i % 64}.linear_attn.out_proj.{i}"] = {
            "group_size": 64,
            "bits": 5,
            "mode": "affine",
        }
    q["language_model.model.layers.3.self_attn.k_proj"] = {
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    }
    return q


def dflash2_27b() -> dict:
    return {
        "model_type": "qwen3",
        "num_hidden_layers": 5,
        "hidden_size": 5120,
        "num_target_layers": 64,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 262144,
    }


def dflash_9b() -> dict:
    return {
        "model_type": "qwen3",
        "num_hidden_layers": 6,
        "hidden_size": 4096,
        "num_target_layers": 32,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 262144,
    }


M5_PRO_WORKING_SET = 55662788608  # 51.8 GiB
M2_AIR_WORKING_SET = 11453251584  # 10.7 GiB
GB = 10**9
