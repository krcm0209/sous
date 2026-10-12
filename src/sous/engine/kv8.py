"""int8 K/V and Q through the attention tile, inside a generation.

The attention tile (``tileattn``) reads bf16 K/V at about 280 GB/s: bandwidth-bound
at one row and issue-bound from three, where a verify call sits. kv8 halves the
bytes and moves QK onto the tensor units' int8 rows. K and V become per-token
symmetric int8 codes with one fp16 scale per (token, KV head); Q is quantised the
same way inside the kernel with one fp32 scale per (query head, row); QK runs as
int8 x int8 -> int32, the K and Q scales multiply the finished score and the V
scale folds into the probability (``kernels/attention_tile_kv8.metal``). The split
kernel keeps the tile's partition, chunk schedule and partial layout, so the
tile's own reduce kernel combines its splits, and every verify row equals the
one-row call at its own length exactly as the tile's rows do.

The codes live in a mirror beside the bf16 cache that exists only inside
``VLMEngine.decode()``: every prefill, fork, slot and stored prompt stays bf16.
The fused row quantiser (``kernels/kv8_quant.metal``) fills it, and a row's codes
and scales depend on that row alone, so quantising a prefix at once or in pieces
gives the same bits.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from sous.engine import nax, tileattn
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.kv8")

# The first key count kv8 serves, a tile step boundary, so no verify run straddles
# it; below it the tile's and the stock paths run unchanged. It sits at the tile's
# N0 unless the int8 saving, net of the per-round quantisation, is measured below
# zero somewhere above it.
N8: int = 1536
# Rows a mirror is built or grown with past the call's own keys: at most one
# growth per 4,096 generated tokens, 135 MB over 16 layers.
GROWTH = 4096
# The kernel against an fp32 reference over the same codes (K, V and Q), as a
# relative RMS; its own error there is about 2e-3.
REL_RMS_BOUND = 1e-2
# kv8 against stock bf16 attention over the bf16 rows the codes came from; the
# format's own error is about 0.7 % on Gaussian K/V.
STOCK_RMS_BOUND = 3e-2
# The attribute a full-attention layer's KVCache carries its mirror under.
MIRROR_ATTR = "_sous_kv8"
# Mirror bytes per token per KV head: 256 K codes, 256 V codes and two fp16 scales,
# against the bf16 cache's 1,024.
BYTES_PER_ROW_PER_HEAD = 516
# mlx's default, stated rather than inherited: verify rows equal decode rows only if
# each rounds exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}
# Appended to the debug split kernel's header only: the body then also writes the Q
# codes it staged, to an output only that kernel object declares.
_QDBG = "#define QDBG 1\n"


# Once per process, as the tile's: per-launch text work slows the decode hot path.
@functools.cache
def _kernels() -> tuple[
    Callable[..., list[Any]], Callable[..., list[Any]], Callable[..., list[Any]]
]:
    """The quantiser, kv8's split kernel and the tile's reduce kernel, built on first
    use. Both of kv8's kernels read their inputs in place through their strides: the
    quantiser takes slices of a KVCache's padded buffers and the split kernel the
    mirror's capacity-sized ones, which ensure_row_contiguous would copy whole on
    every call. Only the split kernel uses the tensor ops, so only its header carries
    nax.h."""
    common = _kernel_text("common.h")
    quant = _kernel(
        "sous_kv8_quant",
        ["k", "v", "params"],
        ["codes", "sc"],
        _kernel_text("kv8_quant.metal"),
        common,
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    split = _kernel(
        "sous_kv8_tile_split",
        ["q", "codes", "sc", "params"],
        ["part", "stats"],
        _kernel_text("attention_tile_kv8.metal"),
        common + _kernel_text("nax.h"),
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    return quant, split, tileattn._kernel_pair()[1]


@functools.cache
def _qdbg_split() -> Callable[..., list[Any]]:
    """The split kernel with QDBG defined: the same body, which then also writes the Q
    codes it staged to a third output, qcodes. mlx names every output of a kernel
    object in the kernel's signature, and a store to an undeclared name does not
    compile even in a branch a template argument switches off, so the debug store
    sits behind the preprocessor and in a kernel object of its own. Tests only."""
    return _kernel(
        "sous_kv8_tile_split_qdbg",
        ["q", "codes", "sc", "params"],
        ["part", "stats", "qcodes"],
        _kernel_text("attention_tile_kv8.metal"),
        _kernel_text("common.h") + _kernel_text("nax.h") + _QDBG,
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )


def quantize(k: Any, v: Any, out_rows: int) -> tuple[Any, Any]:
    """Mirror rows of bf16 K and V views [1, 4, R, 256] (innermost stride 1, read in
    place): codes int8 [1, 4, out_rows, 512], each row's K codes then its V codes, and
    sc fp16 [1, 4, out_rows, 2] = {sK, sV}. Per (row, head): scale = amax / 127 in
    fp16, saturated at fp16's largest finite value; code = clamp(rint(x / f), -127,
    127) with f the scale, or 1 where the scale is 0. Rows from R up are left
    unwritten, so a first build sizes the outputs to the mirror's capacity in the
    same dispatch."""
    import mlx.core as mx

    rows = k.shape[2]
    if not 1 <= rows <= out_rows:
        raise ValueError(f"kv8 quantises 1..{out_rows} rows into {out_rows}, got {rows}")
    hkv, d = tileattn.HKV, tileattn.D
    codes, sc = _kernels()[0](
        inputs=[k, v, mx.array([rows, out_rows], dtype=mx.int32)],
        grid=(d, rows, hkv),
        threadgroup=(d, 1, 1),
        output_shapes=[(1, hkv, out_rows, 2 * d), (1, hkv, out_rows, 2)],
        output_dtypes=[mx.int8, mx.float16],
    )
    return codes, sc


def dequant(codes: Any, sc: Any) -> tuple[Any, Any]:
    """What mirror rows mean: fp32 K and V [1, 4, R, 256], each code times its row's
    scale. The tests' and the probe's reference; the kernel never dequantises."""
    import mlx.core as mx

    values = codes.astype(mx.float32)
    scales = sc.astype(mx.float32)
    d = tileattn.D
    return values[..., :d] * scales[..., 0:1], values[..., d:] * scales[..., 1:2]


def _launch(
    split: Callable[..., list[Any]],
    queries: Any,
    codes: Any,
    sc: Any,
    n: int,
    chunk: int,
    *,
    qcodes: bool = False,
) -> list[Any]:
    """One split dispatch and the tile's reduce: [out], plus the staged Q codes when
    `split` is the debug kernel and `qcodes` asks for them."""
    import mlx.core as mx

    hq, hkv, d = tileattn.HQ, tileattn.HKV, tileattn.D
    t = queries.shape[2]
    mr = tileattn.GQA * t
    rgt = -(-mr // 16)
    nsplit = -(-n // chunk)
    params = mx.array([n, chunk, nsplit], dtype=mx.int32)
    shapes: list[tuple[int, ...]] = [(hkv * nsplit * mr * d,), (hkv * nsplit * mr * 2,)]
    dtypes = [mx.float32, mx.float32]
    if qcodes:
        shapes.append((1, hq, t, d))
        dtypes.append(mx.int8)
    outs = split(
        inputs=[queries, codes, sc, params],
        template=[("TR", t), ("MR", mr), ("RGT", rgt)],
        grid=(64 * rgt, nsplit, hkv),
        threadgroup=(64 * rgt, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    (out,) = _kernels()[2](
        inputs=[outs[0], outs[1], params],
        template=[("MR", mr)],
        grid=(d, mr, hkv),
        threadgroup=(d, 1, 1),
        output_shapes=[(1, t, hq, d)],
        output_dtypes=[mx.bfloat16],
    )
    return [out, *outs[2:]]


def tile8(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> Any:
    """One kernel call over T <= MAX_RUN rows: row t sees keys [0, n - T + t], every
    code and scale read is clamped to n - 1, and keys are split at multiples of
    `chunk`. Queries are bf16 [1, 24, T, 256] with innermost stride 1; codes and sc
    are mirror buffers of at least n rows whose scale plane's head stride is a
    multiple of 4 rows. Returns the contiguous [1, T, 24, 256] bf16 output."""
    return _launch(_kernels()[1], queries, codes, sc, n, chunk)[0]


def _tile8_qcodes(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> tuple[Any, Any]:
    """tile8() through the debug kernel: its output and the Q codes the kernel staged,
    int8 [1, 24, T, 256] in q's dim order."""
    out, qcodes = _launch(_qdbg_split(), queries, codes, sc, n, chunk, qcodes=True)
    return out, qcodes


def decode_codes(queries: Any, codes: Any, sc: Any, n: int, splits: int) -> Any:
    """The one-row call over n keys of a mirror, [1, 24, 1, 256], at the tile's chunk
    for n. kv8 has no stock path of its own: below N0 the caller runs the tile's."""
    chunk = tileattn.chunk_for(n, splits)
    if chunk == 0:
        raise ValueError(f"kv8 serves from N0 = {tileattn.N0} keys, got {n}")
    return tile8(queries, codes, sc, n, chunk).transpose(0, 2, 1, 3)


def verify_codes(
    queries: Any,
    codes: Any,
    sc: Any,
    n: int,
    splits: int,
    runs: list[tuple[int, int, int]] | None = None,
) -> Any:
    """The verifier's T-row causal call over n keys of a mirror, row r over keys
    [0, n - T + r], as [1, 24, T, 256]: one kernel call per run of
    tileattn.verify_plan(), concatenated in row order. `runs` is that plan when the
    caller already has it. Every run must be at or past N0: kv8 has no stock path of
    its own."""
    import mlx.core as mx

    t = queries.shape[2]
    prefix = n - t
    if runs is None:
        runs = tileattn.verify_plan(prefix, t, splits)
    if any(chunk == 0 for _, _, chunk in runs):
        raise ValueError(f"kv8 serves from N0 = {tileattn.N0} keys, got a row at {prefix + 1}")
    if len(runs) == 1:
        return tile8(queries, codes, sc, n, runs[0][2]).transpose(0, 2, 1, 3)
    parts = [tile8(queries[:, :, j:k], codes, sc, prefix + k, chunk) for j, k, chunk in runs]
    return mx.concatenate(parts, axis=1).transpose(0, 2, 1, 3)


_compile_ok = False


def _compile_probe() -> str | None:
    """Compile and run the quantiser and every row variant of the split kernel,
    T = 1..MAX_RUN, on serving-layout inputs; None when they all work, else the
    compiler's error line. Both are compiled at runtime against the OS's own Metal
    toolchain, so a new macOS can reject a source that compiled before. The inputs
    are built with GPU ops and evaluated first: a compile failure with a CPU-stream
    op in flight deadlocks inside mlx's exception path instead of raising. Only
    success is remembered, so one bad moment cannot switch kv8 off for a long-lived
    daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        hq, hkv, d = tileattn.HQ, tileattn.HKV, tileattn.D
        n = tileattn.N0
        # The mirror's capacity runs past its live rows, as a first build leaves it.
        cap = n + 256
        q_key, k_key, v_key = mx.random.split(mx.random.key(0), 3)
        rows = mx.random.normal((1, tileattn.MAX_RUN, hq, d), key=q_key).astype(mx.bfloat16)
        keys = mx.random.normal((1, hkv, cap, d), key=k_key).astype(mx.bfloat16)
        values = mx.random.normal((1, hkv, cap, d), key=v_key).astype(mx.bfloat16)
        mx.eval(rows, keys, values)
        codes, sc = quantize(keys[:, :, :n], values[:, :, :n], cap)
        mx.eval(codes, sc)
        queries = rows.transpose(0, 2, 1, 3)
        chunk = tileattn.chunk_for(n, min(tileattn.MEASURED_SPLITS))
        for t in range(1, tileattn.MAX_RUN + 1):
            mx.eval(tile8(queries[:, :, :t], codes, sc, n, chunk))
    except Exception as e:  # noqa: BLE001 — any failure means the kernels cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("int8 K/V mirror: the kernels failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares, then a compile and run of the quantiser
    and every row variant. kv8's `nax` tests skip on this, never on the tile's: the
    two split kernels can fail to compile on different toolchains."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the int8 K/V kernels failed: {failed}")
    return nax.Availability(True)
