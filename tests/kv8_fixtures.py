"""Shared helpers for kv8's tests: a plain-mlx stand-in for the int8 tile kernel, the
reference quantisers and attention, tie rows, serving-layout inputs and the `nax`
skip marker.

mlx is imported inside each helper, so importing this module needs only what
`kv8.availability()` itself imports."""

import contextlib
from collections.abc import Generator, Sequence
from typing import Any

import pytest

from sous.engine import kv8, tileattn
from tests import tile_fixtures as tfx

# Read once: a failed compile is not remembered, so each call would compile again.
KV8 = kv8.availability()
nax = pytest.mark.skipif(not KV8.available, reason=f"kv8 kernels unavailable: {KV8.reason}")

# The kernel's fp32 SCALE_LOG2: 1/16 is a power of two, so the product rounds to fp32
# exactly as log2(e) alone does, whichever side rounds first.
SCALE_LOG2 = 0.0625 * 1.4426950408889634


def _quantize_rows(x: Any, *, fp16_scale: bool) -> tuple[Any, Any]:
    """Per-row symmetric int8 over the last axis: (codes int8, scale). The scale is
    amax / 127, rounded to fp16 and saturated at its largest finite value for K and
    V, kept in fp32 for Q; a zero scale divides by 1."""
    import mlx.core as mx

    x32 = x.astype(mx.float32)
    amax = mx.max(mx.abs(x32), axis=-1)
    scale = amax / 127.0
    if fp16_scale:
        scale = mx.minimum(scale, 65504.0).astype(mx.float16)
    f = mx.where(scale > 0, scale.astype(mx.float32), 1.0)
    codes = mx.clip(mx.round(x32 / f[..., None]), -127, 127).astype(mx.int8)
    return codes, scale


def reference_quantize(k: Any, v: Any) -> tuple[Any, Any]:
    """kv8.quantize's contract in mlx ops on the GPU: (codes int8 [1, 4, R, 512], sc
    fp16 [1, 4, R, 2]). mx.round is round-half-to-even, Metal's rint."""
    import mlx.core as mx

    k_codes, k_scale = _quantize_rows(k, fp16_scale=True)
    v_codes, v_scale = _quantize_rows(v, fp16_scale=True)
    return mx.concatenate([k_codes, v_codes], axis=-1), mx.stack([k_scale, v_scale], axis=-1)


def reference_quantize_q(queries: Any) -> tuple[Any, Any]:
    """The kernel's in-register Q quantisation in mlx ops: (codes int8 [1, 24, T, 256],
    sq fp32 [1, 24, T])."""
    return _quantize_rows(queries, fp16_scale=False)


def reference_attention(queries: Any, codes: Any, sc: Any, n: int) -> Any:
    """fp32 causal attention over what the codes mean, on the GPU: the queries as the
    kernel quantises them (codes * sq), K and V dequantised from the mirror's first n
    rows, row t of T over keys [0, n - T + t]. [1, 24, T, 256] float32."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    q_codes, sq = reference_quantize_q(queries)
    qd = (q_codes.astype(mx.float32) * sq[..., None]).reshape(1, hkv, gqa, t, d)
    k, v = kv8.dequant(codes[:, :, :n], sc[:, :, :n])
    scores = (qd @ k[:, :, None].swapaxes(-1, -2)) * tileattn.SCALE  # [1, 4, 6, T, n]
    limit = (n - t + mx.arange(t) + 1)[:, None]
    scores = mx.where(mx.arange(n)[None, :] < limit, scores, -float("inf"))
    p = mx.softmax(scores, axis=-1, precise=True)
    return (p @ v[:, :, None]).reshape(1, tileattn.HQ, t, d)


def stand_in_tile8(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> Any:
    """A plain-mlx model of kv8.tile8's contract, tile_fixtures.stand_in_tile's
    partition over the codes. The queries are quantised as the kernel quantises them;
    a (row, key) score is the integer dot of the codes, exact in fp32 in any order
    since every partial sum stays below 2**24, times sq * (SCALE_LOG2 * sK) in that
    order; probabilities are base 2, and PV takes p * sV over the V codes while the
    row sum takes p. Keys are cut into splits at multiples of `chunk`, reads clamped
    to n - 1 and masked past the row's limit, and the splits combined in split order,
    so a row's output depends on the chunk and nothing else about the call. Returns
    [1, T, 24, 256] bf16."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    nsplit = -(-n // chunk)
    prefix = n - t
    q_codes, sq = reference_quantize_q(queries)
    live = codes[:, :, :n].astype(mx.float32)
    k_codes, v_codes = live[..., :d], live[..., d:]
    scales = sc[:, :, :n].astype(mx.float32)
    neg = mx.array(-float("inf"), dtype=mx.float32)
    rows = []
    for r in range(t):
        limit = prefix + r + 1
        # [1, 4, 6, 1, 256] and [1, 4, 6, 1]: the row's six query heads per KV head
        q = q_codes[:, :, r, :].astype(mx.float32).reshape(1, hkv, gqa, 1, d)
        s_q = sq[:, :, r].reshape(1, hkv, gqa, 1)
        maxes, sums, partials = [], [], []
        for s in range(nsplit):
            pos = s * chunk + mx.arange(chunk)
            idx = mx.minimum(pos, n - 1)
            k = mx.take(k_codes, idx, axis=2)[:, :, None]  # [1, 4, 1, c, 256]
            v = mx.take(v_codes, idx, axis=2)[:, :, None]
            s_k = mx.take(scales[..., 0], idx, axis=2)[:, :, None]  # [1, 4, 1, c]
            s_v = mx.take(scales[..., 1], idx, axis=2)[:, :, None]
            score = mx.sum(q * k, axis=-1) * (s_q * (SCALE_LOG2 * s_k))  # [1, 4, 6, c]
            score = mx.where(pos < limit, score, neg)
            m = mx.max(score, axis=-1, keepdims=True)  # a max is order-free
            ref = mx.where(m == neg, mx.zeros_like(m), m)
            p = mx.power(2.0, score - ref)
            maxes.append(m)
            sums.append(tfx._tree_sum(p, axis=-1)[..., None])
            # Rounded through bf16, as tile_fixtures' stand-in rounds its partials: in
            # fp32 a partition change flips too few output bits for a sensitivity test.
            partial = tfx._tree_sum((p * s_v)[..., None] * v, axis=-2)  # [1, 4, 6, 256]
            partials.append(partial.astype(mx.bfloat16).astype(mx.float32))
        gmax = maxes[0]
        for m in maxes[1:]:
            gmax = mx.maximum(gmax, m)
        num = mx.zeros_like(partials[0])
        den = mx.zeros_like(sums[0])
        for split_max, split_sum, split_out in zip(maxes, sums, partials, strict=True):
            w = mx.power(2.0, split_max - gmax)
            num = num + w * split_out
            den = den + w * split_sum
        rows.append((num / den).reshape(1, 1, tileattn.HQ, d))
    return mx.concatenate(rows, axis=1).astype(mx.bfloat16)


def tie_rows(rows: int, *, heads: int = tileattn.HKV, seed: int = 0) -> Any:
    """bf16 [1, heads, rows, 256] whose every element but dim 0 divides by its row's
    scale onto a tie. Dim 0 is +-127 * 2**e, so amax / 127 is the power of two 2**e
    (exact in fp16 and fp32); every other dim is (k + 1/2) * 2**e for an integer k in
    [-127, 126], so x / scale is k + 1/2 exactly and only the rounding rule decides
    the code. e cycles through -8, -4, 0 and 3 by row. Evaluated."""
    import mlx.core as mx

    k_key, s_key = mx.random.split(mx.random.key(seed))
    shape = (1, heads, rows, tileattn.D)
    k = mx.random.randint(-127, 127, shape, key=k_key).astype(mx.float32)
    sign = mx.where(mx.random.bernoulli(0.5, (1, heads, rows, 1), key=s_key), 1.0, -1.0)
    scale = mx.array([2.0 ** (-8, -4, 0, 3)[r % 4] for r in range(rows)]).reshape(1, 1, rows, 1)
    x = mx.concatenate([127 * sign * scale, (k[..., 1:] + 0.5) * scale], axis=-1)
    out = x.astype(mx.bfloat16)
    mx.eval(out)
    return out


def serving_codes(n: int, t: int, *, seed: int = 0) -> tuple[Any, Any, Any, Any, Any]:
    """(queries, codes, sc, keys, values) as kv8 serves them: tile_fixtures.serving_qkv's
    transposed query rows and bf16 [..., :n, :] K/V views of a 256-step padded buffer,
    and their mirror as a first build leaves it, kv8.quantize over the n live rows
    into a capacity of the next 256-row step past n, the rows past n unwritten.
    Evaluated."""
    import mlx.core as mx

    queries, keys, values = tfx.serving_qkv(n, t, seed=seed)
    cap = -(-(n + 1) // 256) * 256
    codes, sc = kv8.quantize(keys, values, cap)
    mx.eval(codes, sc)
    return queries, codes, sc, keys, values


@contextlib.contextmanager
def window(caches: Sequence[Any] = (), *, force: bool = True) -> Generator[kv8.Window]:
    """kv8 active and a serving window open for the with-block: forced (every
    row from N8 served, whatever the ceiling) unless `force` is False, for a
    cap test with ceiling() set. On the way out the window is closed, which
    drops the mirrors and trim wraps on `caches`, unless the block closed it
    itself, and the flag is put back."""
    active = kv8._active
    opened = kv8.open_window(force=force)
    # After open_window: a window leaked by an earlier test makes it assert, and
    # the flag must not stay set for the rest of the session then.
    kv8._active = 1
    try:
        yield opened
    finally:
        if kv8._window is opened:
            kv8.close_window(caches)
        kv8._active = active


def ceiling(monkeypatch: pytest.MonkeyPatch, L: int) -> None:
    """kv8's ceiling at L keys for this test only, as enable() sets it from the
    window: where an unforced window caps."""
    monkeypatch.setattr(kv8, "_ceiling", L)


def small_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """For this test only, kv8 serves from N0 and the probe's far mark is one the
    stand-in reaches in seconds, whatever N8 and far mark the modules carry."""
    monkeypatch.setattr(kv8, "N8", tileattn.N0)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
