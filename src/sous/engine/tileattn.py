"""A tensor-unit attention tile for qwen3_5's one-row decode and exact verifier.

The six query heads that share a KV head are packed with a call's rows into MPP
``matmul2d`` fragments on the M5's tensor units: a split kernel per (key split,
KV head) with an fp32 online softmax in registers, then a reduce kernel that
combines the splits in a fixed order. The kernels (``kernels/attention_tile.metal``)
are specialised to bf16, 24 query / 4 KV heads, head dim 256 and scale 1/16.

Exactness rests on one property: a row's output depends on its key partition,
the chunk each split covers, and on nothing else about the call. Its 32-key
blocks align the same way in every call, keys past its limit get probability
exactly 0, and a (row, key) product does not depend on the fragment row it sits
in. So the chunk is fixed within each step of 32 * S keys (S the GPU's core
count, one wave of threadgroups); a verify call is cut into runs that never
cross N0, a step boundary or 8 rows; and below N0 decode and verify both take
stock paths that are exact against each other. Every verify row then equals a
one-row decode at its own length, so greedy output with a drafter equals greedy
output without one, whatever the block size.

Two hooks reach it. One-row decode goes through a replacement of the
``scaled_dot_product_attention`` global in mlx-vlm's qwen3_5 language module,
the call ``Qwen3_5Attention.__call__`` makes after its projections; verify rows
come from verifyattn's wrapper of the exact verifier. Both decide after the
projections and fall back to the very call the model would have made, and both
act only while ``_active_splits`` is set. Only an ``enable()`` that passed every
gate and the load-time probe sets it; every ``enable()`` and every unload clears
it. Nothing here raises into a model load.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from sous.engine import nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.tileattn")

HQ = 24
HKV = 4
GQA = 6
D = 256
# D ** -0.5, compiled into the kernel as SCALE_LOG2.
SCALE = 0.0625
# The split kernel is instantiated for T = 1..MAX_RUN rows; longer verify calls are cut.
MAX_RUN = 8
# The first key count the tile serves. Below it stock one-row SDPA is as fast (T = 1
# wall time against stock: 0.97x at 1024 keys, 1.02-1.04x at 1101-1281, 1.08x at
# 1536, 1.11x at 2048).
N0 = 1536
# The kernel's key block: a chunk is a multiple of it, and a step is GRAN * S keys.
GRAN = 32
# Split targets measured for speed: 20, the M5 Pro's GPU core count.
MEASURED_SPLITS = frozenset({20})
# attention_tile.metal holds the split body, this line, then the reduce body.
_REDUCE_MARK = "// REDUCE\n"
# mlx's default, stated rather than inherited: parity between the row variants needs
# each to round exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}


def chunk_for(n: int, splits: int) -> int:
    """The split size for a row that sees n keys; 0 means the stock path. Step k
    covers n in (GRAN*S*(k-1), GRAN*S*k] and uses chunk GRAN*k, so no row gets more
    than S splits: S splits x HKV heads is one wave of threadgroups on S cores."""
    if n < N0:
        return 0
    return GRAN * -(-n // (GRAN * splits))


def step_of(n: int, splits: int) -> tuple[int, int, int]:
    """(first n, last n, chunk) of the step that holds key count n."""
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return (1, N0 - 1, 0)
    k = chunk // GRAN
    return (max(N0, GRAN * splits * (k - 1) + 1), GRAN * splits * k, chunk)


def boundaries(n_max: int, splits: int) -> list[int]:
    """Every key count B <= n_max whose step differs from B - 1's, N0 first."""
    out = [N0]
    n = N0
    while True:
        n = step_of(n, splits)[1] + 1
        if n > n_max:
            return out
        out.append(n)


def verify_plan(prefix: int, t: int, splits: int) -> list[tuple[int, int, int]]:
    """A T-row verify behind `prefix` keys as runs (j, k, chunk) of rows [j, k);
    row r sees prefix + r + 1 keys and chunk 0 means stock. A run never crosses N0
    or a step boundary and never holds more than MAX_RUN rows, so any T is served."""
    runs: list[tuple[int, int, int]] = []
    j = 0
    while j < t:
        chunk = chunk_for(prefix + j + 1, splits)
        k = j + 1
        while k < t and k - j < MAX_RUN and chunk_for(prefix + k + 1, splits) == chunk:
            k += 1
        runs.append((j, k, chunk))
        j = k
    return runs


def _kernel_pair() -> tuple[Callable[..., list[Any]], Callable[..., list[Any]]]:
    """The split and reduce kernels, built on first use. The split kernel reads K/V
    in place through their strides: a KVCache hands back a slice of its padded
    buffer, which ensure_row_contiguous would copy whole on every call. Only the
    split kernel uses the tensor ops, so only its header carries nax.h."""
    split_source, reduce_source = _kernel_text("attention_tile.metal").split(_REDUCE_MARK)
    common = _kernel_text("common.h")
    split = _kernel(
        "sous_attention_tile_split",
        ["q", "k", "v", "params"],
        ["part", "stats"],
        split_source,
        common + _kernel_text("nax.h"),
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    reduce = _kernel(
        "sous_attention_tile_reduce",
        ["part", "stats", "params"],
        ["out"],
        reduce_source,
        common,
        compile_options=_SAFE_MATH,
    )
    return split, reduce


def tile(queries: Any, keys: Any, values: Any, n: int, chunk: int) -> Any:
    """One kernel call over T <= MAX_RUN rows: row t sees keys [0, n - T + t], every
    K/V read is clamped to n - 1, and keys are split at multiples of `chunk`.
    Queries are [1, 24, T, 256], K/V [1, 4, >= n, 256], all bf16 with innermost
    stride 1. Returns the contiguous [1, T, 24, 256] bf16 output."""
    import mlx.core as mx

    split, reduce = _kernel_pair()
    t = queries.shape[2]
    mr = GQA * t
    rgt = -(-mr // 16)
    nsplit = -(-n // chunk)
    params = mx.array([n, chunk, nsplit], dtype=mx.int32)
    part, stats = split(
        inputs=[queries, keys, values, params],
        template=[("TR", t), ("MR", mr), ("RGT", rgt)],
        grid=(64 * rgt, nsplit, HKV),
        threadgroup=(64 * rgt, 1, 1),
        output_shapes=[(HKV * nsplit * mr * D,), (HKV * nsplit * mr * 2,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = reduce(
        inputs=[part, stats, params],
        template=[("MR", mr)],
        grid=(D, mr, HKV),
        threadgroup=(D, 1, 1),
        output_shapes=[(1, t, HQ, D)],
        output_dtypes=[mx.bfloat16],
    )
    return out


@functools.cache
def _arch() -> str:
    """This process's GPU architecture, for grouped_attention's plan. Read here, never
    from verifyattn._ARCH, which is set only when verifyattn's own probe passed."""
    import mlx.core as mx

    return str(mx.device_info().get("architecture", ""))


def in_scope(queries: Any, keys: Any, values: Any, scale: float) -> bool:
    """What decode() and verify() assume of the projected tensors; both hooks check
    it after the projections."""
    import mlx.core as mx

    return (
        isinstance(queries, mx.array)
        and isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[0] == keys.shape[0] == values.shape[0] == 1
        and queries.shape[1] == HQ
        and keys.shape[1] == values.shape[1] == HKV
        and queries.shape[3] == keys.shape[3] == values.shape[3] == D
        and queries.dtype == keys.dtype == values.dtype == mx.bfloat16
        and queries.shape[2] >= 1
        and keys.shape[2] == values.shape[2] >= queries.shape[2]
        and scale == SCALE
    )


def decode(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The one-row call over all n = keys.shape[2] keys, [1, 24, 1, 256]. Stock one-row
    SDPA below N0, where the decode hook never calls this; the probe and the tests
    compare against it there."""
    import mlx.core as mx

    n = keys.shape[2]
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=SCALE, mask=None)
    return tile(queries, keys, values, n, chunk).transpose(0, 2, 1, 3)


def verify(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The verifier's T-row causal call, row r over keys [0, n - T + r], as
    [1, 24, T, 256]: one tile call per run of verify_plan(), grouped stock calls for
    the runs below N0, concatenated in row order."""
    import mlx.core as mx

    from sous.engine.verifyattn import grouped_attention

    t = queries.shape[2]
    n = keys.shape[2]
    prefix = n - t
    runs = verify_plan(prefix, t, splits)
    if len(runs) == 1 and runs[0][2]:
        return tile(queries, keys, values, n, runs[0][2]).transpose(0, 2, 1, 3)
    parts = []
    for j, k, chunk in runs:
        q = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
        kk = keys if k == t else keys[:, :, : prefix + k, :]
        vv = values if k == t else values[:, :, : prefix + k, :]
        if chunk:
            parts.append(tile(q, kk, vv, prefix + k, chunk))
        else:
            stock = grouped_attention(q, kk, vv, SCALE, prefix + j, _arch())
            parts.append(stock.transpose(0, 2, 1, 3))
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.transpose(0, 2, 1, 3)


_compile_ok = False


def _compile_probe() -> str | None:
    """Compile and run every row variant, T = 1..MAX_RUN, on serving-layout inputs;
    None when they all work, else the compiler's error line. The kernel is compiled
    at runtime against the OS's own Metal toolchain, so a new macOS can reject a
    source that compiled before. The inputs are built with GPU ops and evaluated
    first: a compile failure with a CPU-stream op in flight deadlocks inside mlx's
    exception path (0.32.2) instead of raising. Only success is remembered, so one
    bad moment cannot switch the tile off for a long-lived daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        q_key, k_key, v_key = mx.random.split(mx.random.key(0), 3)
        rows = mx.random.normal((1, MAX_RUN, HQ, D), key=q_key).astype(mx.bfloat16)
        # One 256-token step past N0: the K/V are slices of a padded buffer, as served.
        keys = mx.random.normal((1, HKV, N0 + 256, D), key=k_key).astype(mx.bfloat16)
        values = mx.random.normal((1, HKV, N0 + 256, D), key=v_key).astype(mx.bfloat16)
        mx.eval(rows, keys, values)
        queries = rows.transpose(0, 2, 1, 3)
        chunk = chunk_for(N0, min(MEASURED_SPLITS))
        for t in range(1, MAX_RUN + 1):
            mx.eval(tile(queries[:, :, :t], keys[:, :, :N0], values[:, :, :N0], N0, chunk))
    except Exception as e:  # noqa: BLE001 — any failure means the kernel cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("attention tile: the kernel failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares with int8 prefill, then a compile and run
    of every row variant. The `nax` tests skip on this, never on int8's: the two
    kernels can fail to compile on different toolchains."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the attention tile kernel failed: {failed}")
    return nax.Availability(True)
