"""Sizing the worker's context window from live memory headroom.

Fixed mode serves the configured max_context_tokens unchanged. Auto mode
exists because the right window is a function of the machine's moment: the
same 64 GB Mac supports a 262k window when idle and far less with a browser
and an IDE resident — and the KV cache is transient per generation, so the
window is a cap on what a task MAY use, not memory the daemon holds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

from sous.config import SousConfig

# mlx-lm's KVCache grows its buffers in 256-token steps; an unaligned window
# ends in a partially-usable step.
_TOKEN_STEP = 256
_GIB = 1 << 30


@dataclass(frozen=True)
class MemorySnapshot:
    working_set: int  # Metal max_recommended_working_set_size, bytes
    active: int  # mx.get_active_memory(): weights + live buffers
    cache: int  # mx.get_cache_memory(): freed-but-retained, reclaimable
    available: int  # system-wide available RAM (psutil semantics)


@dataclass(frozen=True)
class ContextDecision:
    tokens: int
    reason: str


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


def auto_context_tokens(
    *,
    fraction: float,
    min_tokens: int,
    bytes_per_token: int,
    native_max: int,
    working_set: int,
    active: int,
    cache: int,
    available: int,
) -> ContextDecision:
    """Turn memory headroom into a token window.

    Headroom is the tighter of two ceilings: what Metal will serve without
    paging (working set minus what mlx already holds) and what the rest of
    the machine can spare (available RAM, plus mlx's own buffer cache, which
    psutil counts as consumed but clear_cache() reclaims). Exceeding either
    doesn't error — it thrashes, which is worse.
    """
    metal = working_set - active
    system = available + cache
    headroom = max(0, min(metal, system))
    raw = int(headroom * fraction) // bytes_per_token
    aligned_native = native_max // _TOKEN_STEP * _TOKEN_STEP
    tokens = min(raw // _TOKEN_STEP * _TOKEN_STEP, aligned_native)
    # The floor itself is bounded by the native maximum: a small model must
    # never be handed a window past its positional embeddings just because
    # the configured floor assumed a bigger one. Aligned like everything else.
    floor = min(min_tokens // _TOKEN_STEP * _TOKEN_STEP, aligned_native)
    if tokens < floor:
        return ContextDecision(
            floor,
            f"auto: clamped to floor {floor} "
            f"(headroom {headroom / _GIB:.1f} GiB supports only {tokens})",
        )
    return ContextDecision(
        tokens,
        f"auto: min(metal {metal / _GIB:.1f} GiB, system {system / _GIB:.1f} GiB) "
        f"x {fraction} / {bytes_per_token} B/token -> {tokens}",
    )


@lru_cache(maxsize=4)
def _model_config(model_id: str) -> dict:
    from sous.engine.base import fetch_model_config

    return fetch_model_config(model_id)


def _live_memory() -> MemorySnapshot:
    # mlx.core is a compiled extension absent on non-macOS; psutil is only
    # needed on this path — both stay function-local like every mlx import.
    import mlx.core as mx  # ty: ignore[unresolved-import]
    import psutil

    from sous.engine.base import release_mlx_thread_state

    try:
        info = mx.device_info()
        return MemorySnapshot(
            working_set=int(info["max_recommended_working_set_size"]),
            active=mx.get_active_memory(),
            cache=mx.get_cache_memory(),
            available=psutil.virtual_memory().available,
        )
    finally:
        # mlx state must not outlive this call in whatever thread ran it
        # (ml-explore/mlx#4327).
        release_mlx_thread_state()


def decide_context(
    config: SousConfig,
    *,
    model_config_fn=None,
    memory_fn=None,
) -> ContextDecision:
    """The per-task window. Call with the model already loaded, so the weights
    are inside `active` — and call it per task, not per daemon: available RAM
    swings by tens of GB as the user's other apps come and go.

    Any failure degrades to the fixed configured window with a warning: a
    sizing bug must never break delegation.
    """
    if config.context_mode != "auto":
        return ContextDecision(config.max_context_tokens, "fixed")
    try:
        model_config = (model_config_fn or _model_config)(config.model_id)
        bytes_per_token = kv_bytes_per_token(model_config)
        native_max = native_max_tokens(model_config)
        if bytes_per_token is None or native_max is None:
            raise ValueError(f"unknown model shape for {config.model_id}")
        memory = (memory_fn or _live_memory)()
        return auto_context_tokens(
            fraction=config.context_fraction,
            min_tokens=config.context_min_tokens,
            bytes_per_token=bytes_per_token,
            native_max=native_max,
            working_set=memory.working_set,
            active=memory.active,
            cache=memory.cache,
            available=memory.available,
        )
    except Exception as e:  # noqa: BLE001 — degrade to fixed, never break tasks
        warnings.warn(
            f"sous context: auto sizing failed ({e}); using fixed {config.max_context_tokens}",
            stacklevel=2,
        )
        return ContextDecision(config.max_context_tokens, f"fixed (auto sizing failed: {e})")
