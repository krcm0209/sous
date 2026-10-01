"""Shared helpers for the attention tile's tests: a plain-mlx stand-in for the
Metal kernel, serving-layout inputs, the `nax` skip marker, a tiny qwen3_5 model
at the 27B's attention shape, and the installers for both hooks.

mlx is imported inside each helper, so importing this module needs only what
`tileattn.availability()` itself imports."""

import importlib
from typing import Any

import pytest

from sous.engine import tileattn, verifyattn

VOCAB = 512

# Read once: a failed compile is not remembered, so each call would compile again.
TILE = tileattn.availability()
nax = pytest.mark.skipif(not TILE.available, reason=f"attention tile unavailable: {TILE.reason}")


def _tree_sum(x: Any, axis: int) -> Any:
    """Sum over `axis` by pairwise halving of a zero-padded power-of-two length: every
    step is an elementwise add, so the pairing depends only on the axis length."""
    import mlx.core as mx

    axis = axis % x.ndim
    size = x.shape[axis]
    width = 1 << max(size - 1, 0).bit_length()
    if width != size:
        pad = list(x.shape)
        pad[axis] = width - size
        x = mx.concatenate([x, mx.zeros(pad, dtype=x.dtype)], axis=axis)
    while width > 1:
        width //= 2
        lo = [slice(None)] * x.ndim
        hi = [slice(None)] * x.ndim
        lo[axis] = slice(0, width)
        hi[axis] = slice(width, 2 * width)
        x = x[tuple(lo)] + x[tuple(hi)]
    return mx.squeeze(x, axis=axis)


def stand_in_tile(queries: Any, keys: Any, values: Any, n: int, chunk: int) -> Any:
    """A plain-mlx model of the kernel's partition, with tileattn.tile's contract.
    Row t of a T-row call sees keys [0, n - T + t]; keys are cut into splits at
    multiples of `chunk`, reads are clamped to n - 1 and masked past the row's
    limit; each row and split keeps an fp32 max, sum and partial output, and the
    splits are combined in split order. Rows never share an op that mixes them (no
    matmul across packed rows) and every reduction's order depends on the chunk
    alone, so a row's output depends on the chunk and nothing else about the call.
    Returns the kernel's contiguous [1, T, 24, 256] bf16."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    nsplit = -(-n // chunk)
    prefix = n - t
    neg = mx.array(-float("inf"), dtype=mx.float32)
    rows = []
    for r in range(t):
        limit = prefix + r + 1
        # [1, 4, 6, 1, 256]: the row's six query heads per KV head
        q = queries[:, :, r, :].astype(mx.float32).reshape(1, hkv, gqa, 1, d)
        maxes, sums, partials = [], [], []
        for s in range(nsplit):
            pos = s * chunk + mx.arange(chunk)
            idx = mx.minimum(pos, n - 1)
            k = mx.take(keys, idx, axis=2).astype(mx.float32)[:, :, None]  # [1, 4, 1, c, 256]
            v = mx.take(values, idx, axis=2).astype(mx.float32)[:, :, None]
            score = _tree_sum(q * k, axis=-1) * tileattn.SCALE  # [1, 4, 6, c]
            score = mx.where(pos < limit, score, neg)
            m = mx.max(score, axis=-1, keepdims=True)  # a max is order-free
            ref = mx.where(m == neg, mx.zeros_like(m), m)
            p = mx.exp(score - ref)
            maxes.append(m)
            sums.append(_tree_sum(p, axis=-1)[..., None])
            # Rounded through bf16: with fp32 partials a partition change flips about
            # one element in 6,144 of the bf16 output, too few for a sensitivity test.
            partial = _tree_sum(p[..., None] * v, axis=-2)  # [1, 4, 6, 256]
            partials.append(partial.astype(mx.bfloat16).astype(mx.float32))
        gmax = maxes[0]
        for m in maxes[1:]:
            gmax = mx.maximum(gmax, m)
        num = mx.zeros_like(partials[0])
        den = mx.zeros_like(sums[0])
        for split_max, split_sum, split_out in zip(maxes, sums, partials, strict=True):
            w = mx.exp(split_max - gmax)
            num = num + w * split_out
            den = den + w * split_sum
        rows.append((num / den).reshape(1, 1, tileattn.HQ, d))
    return mx.concatenate(rows, axis=1).astype(mx.bfloat16)


def serving_qkv(n: int, t: int, *, seed: int = 0) -> tuple[Any, Any, Any]:
    """Random bf16 inputs laid out as serving hands them over: queries a transposed
    [1, 24, T, 256] slice of a [1, T, 24, 256] pool (q_norm and rope output), keys
    and values [..., :n, :] slices of a KVCache-style buffer grown in 256-token steps
    with at least one spare row. Built with GPU ops and evaluated."""
    import mlx.core as mx

    cap = -(-(n + 1) // 256) * 256
    q_key, k_key, v_key = mx.random.split(mx.random.key(seed), 3)
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=q_key).astype(mx.bfloat16)
    keys = mx.random.normal((1, tileattn.HKV, cap, tileattn.D), key=k_key).astype(mx.bfloat16)
    values = mx.random.normal((1, tileattn.HKV, cap, tileattn.D), key=v_key).astype(mx.bfloat16)
    mx.eval(pool, keys, values)
    return pool.transpose(0, 2, 1, 3), keys[:, :, :n, :], values[:, :, :n, :]


def tiny_language_model(*, seed: int = 0, heads: int = 24, kv_heads: int = 4) -> Any:
    """A random qwen3_5 language model with two full-attention layers (1 and 3) at
    the 27B's attention shape by default, bf16, unquantized."""
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    args = TextConfig(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=heads,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=kv_heads,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,
    )
    mx.random.seed(seed)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    return lm


def prefill_cache(lm: Any, ids: list[int]) -> list:
    """A fresh cache holding `ids`, prefilled in one call. Explicit positions and a
    zero delta, as the VLM engine hands them: the bare language model cannot derive
    rope positions without a vision config."""
    import mlx.core as mx

    cache = lm.make_cache()
    lm(
        mx.array([ids], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(len(ids), dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    )
    mx.eval([c.state for c in cache])
    return cache


def verify_forward(lm: Any, cache: list, tokens: list[int]) -> list:
    """One verify forward as DFlash runs it, then the speculative round aborted so
    the cache is back where it was: the logits and three layers' hidden states."""
    import mlx.core as mx

    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=[0, 1, 2],
        speculative_verify=True,
    )
    arrays = [out.logits, *out.hidden_states]
    mx.eval(arrays)
    out.gdn_states.abort()
    return arrays


def hooked(monkeypatch: pytest.MonkeyPatch, *, splits: int = 20, stand_in: bool = True) -> None:
    """The decode hook installed and the flag set for this test only; monkeypatch
    restores the language module's global and the flag afterwards."""
    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    current = language.scaled_dot_product_attention
    if not getattr(current, tileattn._HOOK_MARK, False):
        monkeypatch.setattr(
            language, "scaled_dot_product_attention", tileattn._decode_wrapper(current)
        )
    monkeypatch.setattr(tileattn, "_active_splits", splits)
    if stand_in:
        monkeypatch.setattr(tileattn, "tile", stand_in_tile)


def verify_ready(monkeypatch: pytest.MonkeyPatch, lm: Any) -> list:
    """verifyattn's wrapper installed and `lm`'s full-attention modules tagged, as
    its enable() leaves them. Tags stay on the modules; a test that needs them off
    sets them to 0."""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    verifyattn.install_wrapper()
    monkeypatch.setattr(verifyattn, "_ARCH", mx.device_info()["architecture"])
    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)
    modules = verifyattn._attention_modules(lm)
    for module in modules:
        object.__setattr__(module, verifyattn._TAG, tileattn.GQA)
    return modules


def guard_tile_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers restores for what enable() and the probe leave behind for the
    rest of the process: mlx-vlm's decode global (the hook), the flag and the
    remembered compile verdict. A test that activates the tile, or runs the
    stand-in through the compile check, then cannot leak either into the next."""
    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    monkeypatch.setattr(tileattn, "_active_splits", tileattn._active_splits)
    monkeypatch.setattr(tileattn, "_compile_ok", tileattn._compile_ok)


def tile_ready(monkeypatch: pytest.MonkeyPatch, *, cores: int = 20) -> None:
    """Every enable() guard that reads the machine passes on any GPU: the
    platform rule, and a core count of `cores` under this GPU's own name. The
    stand-in is the kernel, and the probe's far mark is one the stand-in
    reaches in seconds."""
    import mlx.core as mx

    from sous.engine import gpucores

    device = str(mx.device_info()["device_name"])
    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(
        tileattn.gpucores, "read", lambda: gpucores.GPUCores(cores, device, None, "iokit")
    )
    monkeypatch.setattr(tileattn, "tile", stand_in_tile)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    guard_tile_state(monkeypatch)
