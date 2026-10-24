"""Context-window arithmetic from a model's config.json: what one token of
KV cache costs, and the length the model was trained to attend to."""

from __future__ import annotations

# mlx-lm's KVCache grows its buffers in 256-token steps; an unaligned window
# ends in a partially-usable step.
TOKEN_STEP = 256
# The rows a VLM decode's full-attention KV caches grow by; mlx-vlm's KVCache
# grows by 256. Each growth copies every row of every full-attention layer's K
# and V: on the M5 Pro with the default model's 16 layers, 40-77 ms at 57K
# keys and 265-305 ms at 200K, once per ~249 committed rows at 256, and each
# growth frees a buffer of a size never asked for again, which piled up in
# mlx's buffer cache (39 GB at 200K) until the machine hit memory pressure and
# rounds stalled for seconds. At 4096 the growth cost per generated token is
# 93-94 % lower at both lengths.
DECODE_KV_STEP = 4096


def _text_config(model_config: dict) -> dict:
    # VLMs (the default model included) nest the language model's shape under
    # text_config; text-only models keep it at the top level.
    return model_config.get("text_config", model_config)


def kv_bytes_per_token(model_config: dict) -> int | None:
    """Bytes of KV cache one token costs, or None when the shape is unknown.

    2 (K and V) x attention layers x kv heads (GQA) x head_dim x 2 bytes
    (mlx caches in the compute dtype, fp16/bf16, even for quantized weights).
    Hybrid architectures (the default Qwen3.8 runs 3 Gated DeltaNet layers
    per full-attention layer) only accumulate KV in the full-attention
    layers — the linear ones hold constant-size state — so count only those:
    charging all 64 layers would overestimate 4x and shrink the window for
    nothing.
    """
    cfg = _text_config(model_config)
    try:
        layers = cfg["num_hidden_layers"]
        kv_heads = cfg["num_key_value_heads"]
    except KeyError:
        return None
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        try:
            head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        except KeyError:
            return None
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list):
        layers = sum(1 for t in layer_types if t == "full_attention")
    if not layers:
        # All-linear: nothing grows per token that this formula can size.
        return None
    return 2 * layers * kv_heads * head_dim * 2


def native_max_tokens(model_config: dict) -> int | None:
    return _text_config(model_config).get("max_position_embeddings")
