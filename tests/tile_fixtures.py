"""Shared helpers for the attention tile's tests: a plain-mlx stand-in for the
Metal kernel, serving-layout inputs, and the `nax` skip marker.

mlx is imported inside each helper, so importing this module needs only what
`tileattn.availability()` itself imports."""

from typing import Any

import pytest

from sous.engine import tileattn

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
