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

import dataclasses
import functools
import importlib
import logging
import time
import traceback
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sous.engine import nax, projkernel, tileattn, verifyattn
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.kv8")

# The first key count kv8 serves, a tile step boundary, so no verify run straddles
# it; below it the tile's and the stock paths run unchanged. The int8 saving net of
# the per-round quantiser measured below zero at the tile's N0 on the M5 Pro (-0.05 ms
# a 16-layer round at 1,536 keys) and above it from 4,096 keys up (+0.20 ms there,
# +5.3 ms at 57K), so the floor is the step boundary at or past 4,096.
N8: int = 4481
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


# --------------------------------------------------------------------------------
# The serving window, the mirror and its trim wrap, the ceiling and the floor
# --------------------------------------------------------------------------------

# A mirror's capacity is a multiple of this many rows: the kernel's 16-byte
# scale loads need each head plane of `sc` (cap x 4 bytes) to start 16-byte
# aligned, and 256 is KVCache's own step.
_CAP_STEP = 256
# Set on the trim wrap, so a drop puts back only a trim this module installed.
_TRIM_MARK = "_sous_kv8_trim"


@dataclasses.dataclass
class Window:
    """One decode's serving window. `state` is "idle" until a call is served,
    then "served", and "capped" once a call's first row is at or past the
    ceiling, whether or not anything was served before it (`rows` may then be
    0). `rows` counts the rows quantised in this window, `served` the calls
    served from a mirror, `built` whether any mirror was built. `force` lifts
    the ceiling: for the load-time probe and the tests only. `layers` is the
    number of mirrors built in this window: every full-attention layer builds
    once, so `rows // layers` is the rows one layer quantised."""

    serial: int
    state: str = "idle"
    rows: int = 0
    served: int = 0
    built: bool = False
    force: bool = False
    layers: int = 0


@dataclasses.dataclass
class Mirror:
    """The int8 copy of one full-attention layer's committed K/V, kept on its
    KVCache as MIRROR_ATTR. `codes` int8 [1, 4, cap, 512] holds each token's 256
    K codes then its 256 V codes, `sc` fp16 [1, 4, cap, 2] its {sK, sV}; rows
    [0, n) are valid and no read goes past n. `serial` is the window it was built
    in. `original_trim` is the cache's own `trim` attribute from before the wrap,
    or None when the class's method was in effect: dropping the mirror puts it
    back."""

    codes: Any
    sc: Any
    cap: int
    n: int
    serial: int
    original_trim: Any


# 1 only after enable() passed every gate and the probe: every enable() clears
# it first, clear() clears it, and the probe raises it only while it runs. A
# plain int, so nothing it holds outlives an unload.
_active: int = 0
# The ceiling L: kv8 serves only rows below it. enable() sets it from ceiling();
# 0 serves nothing outside a forced window.
_ceiling: int = 0
# The open serving window, or None. One module global is enough: the engine's
# generation lock serialises every forward, and every decode() runs on the
# caller's session thread.
_window: Window | None = None
# The newest window's serial. Never reset, so a mirror an earlier window left on
# a cache, an earlier load's included, is stale in every later window.
_serial: int = 0
# Per attention call or per window, not per turn: the tests read deltas, and so
# can a harness that wraps the daemon. Nothing is logged per turn.
calls: dict[str, int] = {
    "decode_kv8": 0,  # one-row calls served from the mirror
    "verify_kv8": 0,  # verify calls with at least one run served (the verify seam counts)
    "verify_rows_kv8": 0,  # verify rows served from the mirror (the verify seam counts)
    "verify_launches_kv8": 0,  # kv8 kernel runs of those calls (the verify seam counts)
    "mirror_builds": 0,  # mirrors quantised from row 0
    "mirror_rows": 0,  # rows quantised into a mirror, builds included
    "mirror_trims": 0,  # KVCache.trim calls through the wrap
    "mirror_grown": 0,  # capacity growths
    "windows": 0,  # windows opened
    "windows_capped": 0,  # windows that met a call at or past the ceiling
}


def _round_up(rows: int) -> int:
    return -(-rows // _CAP_STEP) * _CAP_STEP


def ceiling(window: int, splits: int) -> int:
    """L for a `window`-token reserve at `splits` splits: the largest tile step
    boundary b whose mirror fits beside the turn's bf16 KV at every instant,
    else 0. With C = b rounded up to the capacity step, the rule is
    1.504 C + 0.504 C / 16 + 272 <= window: the bf16 KV up to C and the mirror
    at C (516 bytes per token per head against bf16's 1,024), one of the 16
    full-attention layers' old mirror while a growth copies it, and the 16 rows
    a straddling verify holds past L plus one capacity step. A decode grows its
    bf16 KV by window.DECODE_KV_STEP = 4096 rows, so its buffers can overshoot
    the window-sized reserve by up to 4,095 rows (about 256 MiB on the default
    model), mirror or none. Under the automatic prompt-cache budget
    promptcache's 2 GiB CACHE_BUDGET_SLACK absorbs that; a configured
    `[model].prompt_cache_gb` holds nothing back for it, and a decode near L
    can then run about 256 MiB past the recommended working set, which Metal
    pages rather than kv8 mis-serves. Either way the overhang term stays
    16 + 256 rows and the rows kv8 serves do not depend on the decode's step.
    Being a step boundary, L is a verify_plan cut, so no run straddles it."""
    best = 0
    for b in tileattn.boundaries(window, splits):
        c = _round_up(b)
        # The rule in thousandths, so a boundary on the margin is decided in exact
        # integers (C // 16 is exact: C is a multiple of 256).
        if 1504 * c + 504 * (c // 16) + 272_000 <= 1000 * window:
            best = b
    return best


def _floor_reason(splits: int) -> str | None:
    """Why N8 cannot be the floor at `splits` splits, or None. verify_plan cuts
    at every step boundary, so a floor on one keeps every verify run wholly on
    one side of it; a boundary at one S need not be one at another."""
    if N8 in tileattn.boundaries(N8, splits):
        return None
    return f"floor {N8} keys is not a tile step boundary at {splits} splits"


def open_window(*, force: bool = False) -> Window:
    """Open one decode's serving window: until close_window(), serves() may
    answer True and served calls build and read mirrors. `force` serves every
    row from N8 whatever the ceiling."""
    global _window, _serial
    assert _window is None, "a kv8 serving window is already open"
    _serial += 1
    _window = Window(serial=_serial, force=force)
    calls["windows"] += 1
    return _window


def close_window(cache: Sequence[Any]) -> tuple[str, int]:
    """Close the window: drop every mirror and trim wrap in `cache` (the
    decode's per-layer caches; layers without a mirror are passed over) and
    return the window's state and the rows it quantised per full-attention
    layer (every layer quantises the same rows). When a mirror was
    built, mlx's buffer cache is emptied: the mirrors' freed buffers would sit
    there through the publish that follows the decode. With no window open
    (clear() ran) the mirrors still go and the answer is ("off", 0): this runs
    in a finally, where raising would mask the error in flight."""
    global _window
    window = _window
    _window = None
    for layer in cache:
        _drop(layer)
    if window is None:
        return "off", 0
    if window.built:
        import mlx.core as mx

        mx.clear_cache()
    # Rows add up over the full-attention layers, which all quantise the same
    # ones; the gauge is one layer's count. With nothing built (a window whose
    # rows a test set directly) the raw count stands.
    return window.state, window.rows // window.layers if window.layers else window.rows


def serving_now() -> bool:
    """kv8 is active and a window is open that no call has capped."""
    window = _window
    return bool(_active) and window is not None and window.state != "capped"


def serves(n_row: int) -> bool:
    """Whether the row that sees n_row keys reads the mirror: N8 <= n_row < L in
    an open, uncapped window (a forced one serves from N8 whatever L). A row's
    path depends on its own key count alone, so a drafter-on and a drafter-off
    decode of one request serve the same rows, which drafter-on/off parity
    needs while the projection tile is not active; while it is, the same
    property keeps a row's path independent of the block size, which block
    invariance on the drafter's path needs. The seams look this up through the
    module on every call, so a harness may wrap it."""
    window = _window
    if not _active or window is None or window.state == "capped":
        return False
    return n_row >= N8 and (n_row < _ceiling or window.force)


def observe_first_row(cache: Any, n_first: int) -> None:
    """Both seams report an in-scope call's first row's key count here: the
    verify seam for every call, before its runs, the decode seam for each call
    it does not serve. At or past the ceiling the window caps, once, and this
    layer's mirror goes (each layer's own call drops its own): the committed
    prefix never shrinks inside a decode, so no later call of it could be
    served. A forced window never caps."""
    window = _window
    if window is None or window.force or n_first < _ceiling:
        return
    if window.state != "capped":
        window.state = "capped"
        calls["windows_capped"] += 1
    _drop(cache)


def prepare(cache: Any, t: int) -> Mirror:
    """`cache`'s mirror brought up to date for a served call that has just
    appended t rows; once per served call, before any of its runs. Rows below
    offset - t are the committed prefix, so n is clamped there, and then rows
    [n, min(offset, L)) are quantised: no row at or past L is ever written. The
    first served call of a window on this layer, or one meeting a mirror from
    another window, builds it from row 0 instead; rows past the capacity grow
    it first. `cache` is one layer's KVCache."""
    window = _window
    assert window is not None, "kv8.prepare outside a serving window"
    offset = cache.offset
    top = offset if window.force else min(offset, _ceiling)
    window.served += 1
    if window.state == "idle":
        window.state = "served"
    mirror = getattr(cache, MIRROR_ATTR, None)
    if mirror is None or mirror.serial != window.serial:
        return _build(cache, window, top)
    # A defence: inside a window the trim wrap has already lowered n for every
    # rollback.
    mirror.n = min(mirror.n, offset - t)
    if top > mirror.cap:
        _grow(mirror, _capacity(offset, window))
    start = mirror.n
    if top > start:
        codes, sc = quantize(
            cache.keys[:, :, start:top, :], cache.values[:, :, start:top, :], top - start
        )
        # The Mirror holds the only reference to its buffers, so mlx donates them
        # and both updates write in place.
        mirror.codes[:, :, start:top, :] = codes
        mirror.sc[:, :, start:top, :] = sc
        mirror.n = top
        _count_rows(window, top - start)
    return mirror


def _build(cache: Any, window: Window, top: int) -> Mirror:
    """Rows [0, top) quantised in one dispatch straight into a fresh mirror at
    the window's capacity (rows past top are left unwritten: no read reaches
    them), and the cache's trim wrapped. A mirror from another window goes
    first, but its wrap stays and serves the new one."""
    attrs = vars(cache)
    stale = attrs.pop(MIRROR_ATTR, None)
    if stale is None:
        original = attrs.get("trim")
        object.__setattr__(cache, "trim", _wrap_trim(cache, original))
    else:
        original = stale.original_trim
    # This `del` frees nothing itself: prepare's `mirror` local still holds the
    # stale Mirror until prepare returns. The stale buffers go before the new
    # ones exist because `quantize` is lazy and allocates at eval time, after
    # prepare has returned.
    del stale
    cap = _capacity(cache.offset, window)
    codes, sc = quantize(cache.keys[:, :, :top, :], cache.values[:, :, :top, :], cap)
    mirror = Mirror(
        codes=codes, sc=sc, cap=cap, n=top, serial=window.serial, original_trim=original
    )
    object.__setattr__(cache, MIRROR_ATTR, mirror)
    calls["mirror_builds"] += 1
    window.built = True
    window.layers += 1
    _count_rows(window, top)
    return mirror


def _capacity(offset: int, window: Window) -> int:
    """GROWTH rows of headroom past the offset, so a decode grows at most once
    per GROWTH generated tokens, and never past L rounded up, the size
    ceiling() priced (a forced window has no L)."""
    cap = _round_up(offset + GROWTH)
    return cap if window.force else min(cap, _round_up(_ceiling))


def _grow(mirror: Mirror, cap: int) -> None:
    """One layer's mirror widened to `cap` rows. The new buffers are evaluated
    before the old ones are dropped, so the transient is this layer's old
    buffer: left lazy, every layer's old buffer would live until the forward
    is evaluated."""
    import mlx.core as mx

    extra = cap - mirror.cap
    codes_tail = mx.zeros((1, tileattn.HKV, extra, 2 * tileattn.D), dtype=mx.int8)
    sc_tail = mx.zeros((1, tileattn.HKV, extra, 2), dtype=mx.float16)
    codes = mx.concatenate([mirror.codes, codes_tail], axis=2)
    sc = mx.concatenate([mirror.sc, sc_tail], axis=2)
    mx.eval(codes, sc)
    mirror.codes, mirror.sc, mirror.cap = codes, sc, cap
    calls["mirror_grown"] += 1


def _count_rows(window: Window, rows: int) -> None:
    calls["mirror_rows"] += rows
    window.rows += rows


def _wrap_trim(cache: Any, original: Any) -> Callable[[int], Any]:
    """`cache`'s trim, then its mirror's n lowered to the new offset. Inside a
    decode only KVCache.trim lowers an offset, and DFlash's commit and abort
    both reach it as an instance call, so this wrap is what keeps a mirror
    valid across rollbacks. `original` is the instance's own trim from before,
    or None for the class's. The cache is held weakly: held strongly from its
    own attribute, it would be a reference cycle whose K/V waits for the cycle
    collector."""
    ref = weakref.ref(cache)
    method = type(cache).trim

    def trim(n: int) -> Any:
        target: Any = ref()
        trimmed = method(target, n) if original is None else original(n)
        mirror = getattr(target, MIRROR_ATTR, None)
        if mirror is not None:
            mirror.n = min(mirror.n, target.offset)
        calls["mirror_trims"] += 1
        return trimmed

    setattr(trim, _TRIM_MARK, True)
    return trim


def _drop(cache: Any) -> None:
    """`cache`'s mirror removed and the trim it wrapped put back. A cache
    without one (a linear-attention layer's, or one never served) is passed
    over, and a trim someone set over the wrap since is left alone."""
    attrs = getattr(cache, "__dict__", None)
    if attrs is None:
        return
    mirror = attrs.pop(MIRROR_ATTR, None)
    if mirror is None or not getattr(attrs.get("trim"), _TRIM_MARK, False):
        return
    if mirror.original_trim is None:
        del attrs["trim"]
    else:
        attrs["trim"] = mirror.original_trim


def decode(queries: Any, cache: Any, splits: int) -> Any:
    """A served one-row call: the mirror brought up to date with the call's own
    row, then the kernel over all n = cache.offset keys, [1, 24, 1, 256] bf16."""
    mirror = prepare(cache, 1)
    calls["decode_kv8"] += 1
    return decode_codes(queries, mirror.codes, mirror.sc, cache.offset, splits)


def verify_run(queries: Any, mirror: Mirror, prefix: int, j: int, k: int, chunk: int) -> Any:
    """One served run of a T-row verify call behind `prefix` keys: rows [j, k)
    of `queries` [1, 24, T, 256], row r over keys [0, prefix + r], read from the
    mirror prepare() returned for the call. Returns the contiguous
    [1, k - j, 24, 256] a tile run returns, for the caller to concatenate along
    axis 1. A launch only: the verify seam counts the call, its rows and its
    launches."""
    t = queries.shape[2]
    rows = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
    return tile8(rows, mirror.codes, mirror.sc, prefix + k, chunk)


def clear() -> None:
    """Stop serving (VLMEngine.unload()): the flag and any open window go.
    Mirrors live on caches that go with the engine, so nothing else is held."""
    global _active, _window
    _active = 0
    _window = None


# ---- enable(): the gates, the pins, availability, the ceiling and the probe ---------

# What keeps the mirror's rows the cache's rows inside a decode: verify rows are
# appended through update_and_fetch, which truncates to the offset before it
# grows, so rows below the offset never move; nothing but KVCache.trim lowers an
# offset (prefix_cache_reserve pads and keeps rows and offset); and DFlash's
# commit and abort roll rows back only through the instance trim the mirror
# wraps, never by replaying them. mlx-vlm's rotating-cache path already replays
# accepted rows, and a KVCache moved onto such a path would leave the mirror
# stale with every other check green. sha256 of inspect.getsource, keyed
# (module, qualname), as verifyattn pins its own sources; getsource follows
# __wrapped__, so a class-level wrapper never reads as drift. Validated on
# mlx-vlm 0.7.2; add a digest only after re-reading that function against the
# trim wrap.
VALIDATED_KV8_SOURCES: dict[tuple[str, str], str] = {
    ("mlx_vlm.models.cache", "KVCache.update_and_fetch"): (
        "9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b"
    ),
    ("mlx_vlm.models.cache", "KVCache.trim"): (
        "8e020b3f950f96a9c9ec66502e8bb744764de418d527a7c8616cff80433dcb79"
    ),
    ("mlx_vlm.models.cache", "KVCache.prefix_cache_reserve"): (
        "4233bfad20c4845ab27ddb87d23e7a3827f1d73556100d1a759563ee58aa4109"
    ),
    ("mlx_vlm.speculative.cache_state", "start_speculative_cache"): (
        "5a462e518be86200b48291b75876dcedd2fe3cdab7192ac9c3062b4088273585"
    ),
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.commit"): (
        "d4c6ae13b43a0054aeea966bd865156bca534504aa568ee670e11b869d9b52a3"
    ),
    # Calls cache.trim on the instance, so an aborted round reaches the wrap.
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.abort"): (
        "f279604e5e1bc5c050cb1fe3bbf1a3d8de13ccf998b184ece45025615f55add2"
    ),
    # Resolves getattr(cache, "trim"), so a partial commit reaches the wrap.
    ("mlx_vlm.speculative.cache_state", "_trim_append_caches"): (
        "03056d9dae1435ff39b01f90d9244a5563d6c70d15d3c44ab9b9ad8fd176677d"
    ),
    ("mlx_vlm.speculative.cache_state", "_cache_position_reader"): (
        "349251fa2854f1021103a62f5502ab79285aedec63c12ece8c060f29ba89e4b6"
    ),
    ("mlx_vlm.speculative.cache_state", "commit_speculative_round"): (
        "58c3e8cf462dea7a576f56b42d0ff38d45daea051b95bc9e9bd8d1ef3698997c"
    ),
    ("mlx_vlm.speculative.cache_state", "abort_speculative_round"): (
        "642f9cc3e46734e260f3cb776398ed4a18ff361f88086cfb01fe2d704d76d69c"
    ),
    ("mlx_vlm.speculative.cache_state", "rollback_speculative_cache"): (
        "67de5bcb251299a27202019a8547635fe74b40d009764586dc3c53f04d27c053"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_rounds"): (
        "eee8dc9ab95a74ed87c74064607736003b7d0dafbdafc32c9fba91dc190d2b32"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_verify"): (
        "39e4a6cd343ae7e054ae3584ff6879fc895494e413041f099ce08da14e3fc097"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_verify_greedy"): (
        "1b472dc300554ab6017495bdeb18480be431578aa6958ff7b35c0145e8412971"
    ),
    ("mlx_vlm.speculative.dflash", "_reserve_dflash_target_cache"): (
        "a122a70af6663e7a233554b47997e2fb177f8e971c14b6b9aaf928b7bd5e789d"
    ),
    ("mlx_vlm.models.qwen3_5.language", "LanguageModel.rollback_speculative_cache"): (
        "134fe07e0a9fb905a56d4397f8b7bba86729330cac8c8d9a8d16becadb316058"
    ),
    ("mlx_vlm.models.qwen3_5.speculative_verifier", "Qwen3_5BatchInvariantForward.verify"): (
        "65153e7d0b6099869666eb4d10703056b0432c57025db76d808ff46e0c231ae9"
    ),
}
# The verify depth the probe's verifier step runs; the exactness steps prove
# every T from 1 to 8 on their own.
_VERIFY_ROWS = 5
# The row counts the exactness step runs at each of its two marks.
_EXACT_ROWS = (1, 5, 8)
# The quantiser step's random and tie rows; an all-zero row follows them.
_QUANT_RANDOM = 37
_QUANT_TIES = 8
# fp16's largest finite value: the quantiser's scale saturates there.
_FP16_MAX = 65504.0


def _reference_quantize(k: Any, v: Any) -> tuple[Any, Any]:
    """The quantiser's formula in mlx ops, for the probe's bitwise check:
    per (head, row) s = fp16(min(amax / 127, fp16 max)), codes rint(x / s)
    clamped to +-127, divided by 1 instead where s rounded to 0. mx.round rounds
    half to even, as Metal's rint does. Returns (codes int8 [1, 4, R, 512], K
    then V; scales fp16 [1, 4, R, 2], sK then sV)."""
    import mlx.core as mx

    def one(x: Any) -> tuple[Any, Any]:
        x32 = x.astype(mx.float32)
        amax = mx.max(mx.abs(x32), axis=-1, keepdims=True)
        s = mx.minimum(amax / 127, _FP16_MAX).astype(mx.float16)
        f = mx.where(s > 0, s.astype(mx.float32), 1.0)
        return mx.clip(mx.round(x32 / f), -127, 127).astype(mx.int8), s

    k_codes, k_scales = one(k)
    v_codes, v_scales = one(v)
    return (
        mx.concatenate([k_codes, v_codes], axis=-1),
        mx.concatenate([k_scales, v_scales], axis=-1),
    )


def _tie_rows(rows: int) -> Any:
    """[1, 4, rows, 256] bf16 rows whose amax is 127 * 2**e, so the scale is
    exactly 2**e and x / s lands on a half for every element but the amax: the
    rows where the rounding rule shows. The scales come from Python floats, so
    every product is exact."""
    import mlx.core as mx

    lane = mx.arange(tileattn.D)
    halves = (lane % 254).astype(mx.float32) - 126.5
    base = mx.where(lane == 0, 127.0, halves)
    scales = mx.array([(-1) ** r * 2.0 ** (r % 8 - 3) for r in range(rows)], dtype=mx.float32)
    tied = scales[:, None] * base[None, :]
    shape = (1, tileattn.HKV, rows, tileattn.D)
    return mx.broadcast_to(tied[None, None], shape).astype(mx.bfloat16)


def _dequant_q(queries: Any) -> Any:
    """What the kernel's in-kernel Q quantisation means, in fp32: per (query
    head, row) codes rint(q / sq) clamped to +-127, times sq = amax / 127; a
    zero row stays zero."""
    import mlx.core as mx

    q32 = queries.astype(mx.float32)
    sq = mx.max(mx.abs(q32), axis=-1, keepdims=True) / 127
    f = mx.where(sq > 0, sq, 1.0)
    return mx.clip(mx.round(q32 / f), -127, 127) * sq


def _reference_attention(q32: Any, k32: Any, v32: Any) -> Any:
    """fp32 attention on the GPU (fp64 is CPU-only in mlx): T query rows over n
    keys, row r seeing keys [0, n - T + r], each KV head shared by the six
    query heads after it. Returns [1, 24, T, 256] fp32."""
    import mlx.core as mx

    t, n = q32.shape[2], k32.shape[2]
    q = q32.reshape(1, tileattn.HKV, tileattn.GQA, t, tileattn.D)
    scores = (q @ k32[:, :, None].swapaxes(-1, -2)) * tileattn.SCALE
    keep = mx.arange(n)[None, :] <= (n - t + mx.arange(t))[:, None]
    weights = mx.softmax(mx.where(keep, scores, float("-inf")), axis=-1)
    return (weights @ v32[:, :, None]).reshape(1, tileattn.HQ, t, tileattn.D)


def _distance(got: Any, want: Any) -> float:
    """Relative RMS of `got` from `want` in fp32; infinite when `got` is not
    finite everywhere, so a NaN fails every bound."""
    import mlx.core as mx

    if not mx.all(mx.isfinite(got)).item():
        return float("inf")
    return tileattn._rel_rms(got, want)


def _rows(qpool: Any, start: int, t: int) -> Any:
    """T query rows of the [1, 20, 24, 256] pool as serving hands them over:
    a transposed [1, 24, T, 256] view."""
    return qpool[:, start : start + t].transpose(0, 2, 1, 3)


def _run(queries: Any, codes: Any, sc: Any, n: int, splits: int) -> Any:
    """[1, 24, T, 256]: the one-row call at T = 1, the verify call above it."""
    if queries.shape[2] == 1:
        return decode_codes(queries, codes, sc, n, splits)
    return verify_codes(queries, codes, sc, n, splits)


def _restored(offset: int, key: Any) -> Any:
    """A KVCache as a fork restore leaves it: keys, values, then offset
    assigned (the state setter would derive the offset from the padded shape),
    the 256-step padded buffer NaN past the live rows. The same key gives the
    same cache, so a served and an unserved call can start from one state."""
    import mlx.core as mx

    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    capacity = -(-(offset + tileattn.MAX_RUN + 1) // 256) * 256
    k_key, v_key = mx.random.split(key)
    live = (1, tileattn.HKV, offset, tileattn.D)
    tail = mx.full(
        (1, tileattn.HKV, capacity - offset, tileattn.D), float("nan"), dtype=mx.bfloat16
    )
    cache: Any = kv_cache()
    keys = mx.random.normal(live, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(live, key=v_key).astype(mx.bfloat16)
    cache.keys = mx.concatenate([keys, tail], axis=2)
    cache.values = mx.concatenate([values, tail], axis=2)
    cache.offset = offset
    mx.eval(cache.keys, cache.values)
    return cache


def _probe_quantiser(key: Any) -> str | None:
    """The quantiser bitwise against _reference_quantize on random rows, tie
    rows and an all-zero row, read through strided views of a padded buffer
    into a capacity-sized output, as a first build writes it; and rows added
    in two steps bitwise equal to one build."""
    import mlx.core as mx

    k_key, v_key = mx.random.split(key)
    random_shape = (1, tileattn.HKV, _QUANT_RANDOM, tileattn.D)
    ties = _tie_rows(_QUANT_TIES)
    zero = mx.zeros((1, tileattn.HKV, 1, tileattn.D), dtype=mx.bfloat16)
    pad = mx.zeros((1, tileattn.HKV, 256, tileattn.D), dtype=mx.bfloat16)
    kbuf = mx.concatenate(
        [mx.random.normal(random_shape, key=k_key).astype(mx.bfloat16), ties, zero, pad], axis=2
    )
    vbuf = mx.concatenate(
        [mx.random.normal(random_shape, key=v_key).astype(mx.bfloat16), -ties, zero, pad], axis=2
    )
    mx.eval(kbuf, vbuf)
    rows = _QUANT_RANDOM + _QUANT_TIES + 1
    k, v = kbuf[:, :, :rows], vbuf[:, :, :rows]
    codes, sc = quantize(k, v, -(-rows // 256) * 256)
    want_codes, want_sc = _reference_quantize(k, v)
    mx.eval(codes, sc, want_codes, want_sc)
    segments = (
        ("random", 0, _QUANT_RANDOM),
        ("tie", _QUANT_RANDOM, _QUANT_RANDOM + _QUANT_TIES),
        ("all-zero", rows - 1, rows),
    )
    for label, a, b in segments:
        if not (
            mx.array_equal(codes[:, :, a:b], want_codes[:, :, a:b]).item()
            and mx.array_equal(sc[:, :, a:b], want_sc[:, :, a:b]).item()
        ):
            return f"quantiser: the {label} rows differ from the reference"
    # A cut inside the random rows, at neither end and not on a power of two.
    cut = _QUANT_RANDOM // 2 + 1
    first_codes, first_sc = quantize(kbuf[:, :, :cut], vbuf[:, :, :cut], cut)
    rest_codes, rest_sc = quantize(kbuf[:, :, cut:rows], vbuf[:, :, cut:rows], rows - cut)
    steps_codes = mx.concatenate([first_codes, rest_codes], axis=2)
    steps_sc = mx.concatenate([first_sc, rest_sc], axis=2)
    if not (
        mx.array_equal(steps_codes, codes[:, :, :rows]).item()
        and mx.array_equal(steps_sc, sc[:, :, :rows]).item()
    ):
        return "quantiser: rows added in two steps differ from one build"
    return None


def _probe_exactness(
    kb: Any, vb: Any, qpool: Any, firsts: tuple[int, ...], splits: int
) -> str | None:
    """What parity cannot see in a kernel wrong the same way on both paths:
    the kernel within REL_RMS_BOUND of fp32 attention over the very codes it
    reads (K, V and Q dequantised), for calls whose first row sees each of
    `firsts` keys, at T = 1, 5 and 8, with x8 queries, a dominant key at
    n - 1 and a decoy at n, just past the live range, that no read may
    reach."""
    import mlx.core as mx

    ones = mx.ones((1, tileattn.HKV, 1, tileattn.D), dtype=mx.bfloat16)
    for first in firsts:
        for t in _EXACT_ROWS:
            n = first + t - 1
            q8 = (_rows(qpool, 0, t).astype(mx.float32) * 8).astype(mx.bfloat16)
            # The last row's own query per KV head: only that row sees key n - 1.
            heads = q8[:, :, t - 1].reshape(1, tileattn.HKV, tileattn.GQA, tileattn.D)
            dominant = heads[:, :, :1]
            kp = mx.concatenate([kb[:, :, : n - 1], dominant, dominant, kb[:, :, n + 1 :]], axis=2)
            vp = mx.concatenate([vb[:, :, : n - 1], ones, -ones, vb[:, :, n + 1 :]], axis=2)
            mx.eval(q8, kp, vp)
            codes, sc = quantize(kp, vp, kp.shape[2])
            mx.eval(codes, sc)
            got = _run(q8, codes, sc, n, splits)
            k32, v32 = dequant(codes[:, :, :n], sc[:, :, :n])
            want = _reference_attention(_dequant_q(q8), k32, v32)
            distance = _distance(got, want)
            if not distance <= REL_RMS_BOUND:
                return (
                    f"exactness: T={t} at n={n} is {distance:.3g} from the fp32 reference"
                    f" (bound {REL_RMS_BOUND})"
                )
    return None


def _probe_parity(codes: Any, sc: Any, qpool: Any, marks: list[int], splits: int) -> str | None:
    """What greedy parity with the drafter on or off rests on while the
    projection tile is not active, and block invariance on the drafter's path
    while it is: every verify row array_equal to the one-row call at its own
    key count, at each mark for T = 1..MAX_RUN, with the rows below the mark,
    straddling it and from it (the tile's three placements). Query row i of
    the pool sits at position mark - 10 + i."""
    import mlx.core as mx

    for mark in marks:
        base = mark - 10
        dec = {
            pos + 1: decode_codes(_rows(qpool, pos - base, 1), codes, sc, pos + 1, splits)
            for pos in range(base, base + 20)
            if pos + 1 >= N8
        }
        mx.eval(list(dec.values()))
        for t in range(1, tileattn.MAX_RUN + 1):
            for prefix in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}):
                if prefix + 1 < N8:
                    continue  # rows below the floor are the tile's, not kv8's
                out = verify_codes(_rows(qpool, prefix - base, t), codes, sc, prefix + t, splits)
                same = mx.stack(
                    [mx.array_equal(out[:, :, r], dec[prefix + r + 1][:, :, 0]) for r in range(t)]
                )
                if not mx.all(same).item():
                    return f"parity: verify rows at prefix {prefix}, T={t} differ from decode"
    return None


def _probe_fence(
    codes: Any, sc: Any, qpool: Any, points: tuple[tuple[int, int], ...], splits: int
) -> str | None:
    """Reads stay in [0, n). Rows past n set to -128 codes (never a
    code the quantiser writes) and NaN scales, read whole and through a view
    ending exactly at n over the capacity-sized buffer (whose scale plane keeps
    its 16-byte head stride), give the clean result bitwise, and finite."""
    import mlx.core as mx

    cap = codes.shape[2]
    for n, t in points:
        queries = _rows(qpool, 0, t)
        want = _run(queries, codes, sc, n, splits)
        tail = (1, tileattn.HKV, cap - n)
        bad_codes = mx.concatenate(
            [codes[:, :, :n], mx.full((*tail, 2 * tileattn.D), -128, dtype=mx.int8)], axis=2
        )
        bad_sc = mx.concatenate(
            [sc[:, :, :n], mx.full((*tail, 2), float("nan"), dtype=mx.float16)], axis=2
        )
        mx.eval(want, bad_codes, bad_sc)
        for c, s in ((bad_codes, bad_sc), (bad_codes[:, :, :n], bad_sc[:, :, :n])):
            got = _run(queries, c, s, n, splits)
            if not (mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()):
                return f"reads past n: T={t} at n={n} differs from the clean buffer"
    return None


def _probe_mirror(key: Any, held: list[Any]) -> str | None:
    """The mirror rule on a real KVCache under a forced window: after
    every served call the mirror equals a bulk re-quantisation of the cache's
    rows. In order: a first build over a cache whose capacity is not a multiple
    of 4; a trim through the wrap, an unserved append over the trimmed rows,
    then a served call; the shrinking block across the floor; a mirror left
    from another window, which the next window rebuilds; a growth."""
    import mlx.core as mx

    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    # An odd-sized first segment: the truncating growth that follows it leaves
    # the cache a capacity that is not a multiple of 4.
    first = N8 + 1 if (N8 + 1) % 4 else N8 + 2
    k_key, v_key = mx.random.split(key)
    pool = (1, tileattn.HKV, first + GROWTH + 1024, tileattn.D)
    kpool = mx.random.normal(pool, key=k_key).astype(mx.bfloat16)
    vpool = mx.random.normal(pool, key=v_key).astype(mx.bfloat16)
    mx.eval(kpool, vpool)
    cache: Any = kv_cache()
    held.append(cache)
    drawn = 0

    def append(rows: int) -> None:
        # Fresh rows on every call, so an append over trimmed rows changes them.
        nonlocal drawn
        cache.update_and_fetch(kpool[:, :, drawn : drawn + rows], vpool[:, :, drawn : drawn + rows])
        mx.eval(cache.keys, cache.values)
        drawn += rows

    def requantised(mirror: Mirror) -> bool:
        off = cache.offset
        codes, sc = quantize(cache.keys[:, :, :off], cache.values[:, :, :off], off)
        return bool(
            mirror.n == off
            and mirror.cap >= off
            and mirror.cap % 256 == 0
            and mx.array_equal(mirror.codes[:, :, :off], codes).item()
            and mx.array_equal(mirror.sc[:, :, :off], sc).item()
        )

    open_window(force=True)
    append(first)
    append(256)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: the first build differs from a bulk re-quantisation"
    cache.trim(3)
    if getattr(cache, MIRROR_ATTR).n != cache.offset:
        return "mirror: a trim did not reach the mirror"
    append(3)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a trim and an unserved append differ from a bulk re-quantisation"
    # 16 rows served from just below the floor, rolled back below it, 8 rows
    # appended below it without a served call, then a served call.
    cache.trim(cache.offset - (N8 - 4))
    append(16)
    prepare(cache, 16)
    cache.trim(24)
    append(8)
    append(8)
    if not requantised(prepare(cache, 8)):
        return "mirror: the shrinking block across the floor differs from a bulk re-quantisation"
    # The cache moves under a mirror from another window with no wrap in the
    # way (the class's own trim), as a restore between decodes would.
    close_window([])
    type(cache).trim(cache, 8)
    append(8)
    open_window(force=True)
    builds = calls["mirror_builds"]
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a mirror from another window differs from a bulk re-quantisation"
    if calls["mirror_builds"] != builds + 1:
        return "mirror: a mirror from another window was not rebuilt"
    grown = calls["mirror_grown"]
    append(GROWTH + 256)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a growth differs from a bulk re-quantisation"
    if calls["mirror_grown"] != grown + 1:
        return "mirror: a growth was not counted"
    close_window([cache])
    return None


def _probe_stock(
    kb: Any, vb: Any, codes: Any, sc: Any, qpool: Any, ns: tuple[int, ...], splits: int
) -> str | None:
    """The format's own error: one-row calls over Gaussian K/V within
    STOCK_RMS_BOUND of stock bf16 attention over the same rows unquantised."""
    import mlx.core as mx

    queries = _rows(qpool, 0, 1)
    for n in ns:
        got = decode_codes(queries, codes, sc, n, splits)
        want = mx.fast.scaled_dot_product_attention(
            queries, kb[:, :, :n], vb[:, :, :n], scale=tileattn.SCALE, mask=None
        )
        distance = _distance(got, want)
        if not distance <= STOCK_RMS_BOUND:
            return (
                f"quantisation: n={n} is {distance:.3g} from stock attention"
                f" (bound {STOCK_RMS_BOUND})"
            )
    return None


def _probe_module(
    modules: list[Any], splits: int, verify_active: bool, n: int, key: Any, held: list[Any]
) -> str | None:
    """The loaded model's own attention module, projections and rope
    included, one row through mlx-vlm's decode global with a forced window open
    against the same call on the bf16 path, both over a restored-style KVCache;
    then, with exact verify active, the same module through the exact
    verifier's wrapper over _VERIFY_ROWS rows. Each served call must move its
    counter by one. Without a drafter kv8 serves decode only, so the verifier
    is never built."""
    import mlx.core as mx

    tileattn.install_decode_hook()
    tileattn._active_splits = splits
    module = modules[0]
    hidden = module.o_proj.weight.shape[0]
    x_key, c_key, v_key = mx.random.split(key, 3)
    x = mx.random.normal((1, 1, hidden), key=x_key).astype(mx.bfloat16)
    xs = mx.random.normal((1, _VERIFY_ROWS, hidden), key=v_key).astype(mx.bfloat16)
    mx.eval(x, xs)

    def restored(offset: int) -> Any:
        cache = _restored(offset, c_key)
        held.append(cache)
        return cache

    def served(call: Any, offset: int) -> Any:
        open_window(force=True)
        cache = restored(offset)
        out = call(cache)
        mx.eval(out)
        close_window([cache])
        return out

    want = module(x, mask=None, cache=restored(n - 1))
    mx.eval(want)
    before = calls["decode_kv8"]
    got = served(lambda cache: module(x, mask=None, cache=cache), n - 1)
    if calls["decode_kv8"] != before + 1:
        return "the model's attention module did not reach kv8"
    distance = _distance(got, want)
    if not distance <= STOCK_RMS_BOUND:
        return (
            f"correctness: the model's module is {distance:.3g} from the bf16 path"
            f" (bound {STOCK_RMS_BOUND})"
        )
    if not verify_active:
        return None
    verifier = projkernel._verifier_class()()

    def verify(cache: Any) -> Any:
        return verifier._attention(module, xs, "causal", cache, None, None)

    want = verify(restored(n))
    mx.eval(want)
    before = calls["verify_kv8"]
    got = served(verify, n)
    if calls["verify_kv8"] != before + 1:
        return "the exact verifier's attention did not reach kv8"
    distance = _distance(got, want)
    if not distance <= STOCK_RMS_BOUND:
        return (
            f"correctness: the verifier's attention is {distance:.3g} from the bf16 path"
            f" (bound {STOCK_RMS_BOUND})"
        )
    return None


def _probe_steps(
    modules: list[Any], splits: int, verify_active: bool, held: list[Any]
) -> str | None:
    """Every input is made with GPU ops and evaluated before the launches that
    read it, for the deadlock _compile_probe() describes. The marks: the floor
    N8, the next two step boundaries and the far mark, the first boundary at or
    past tileattn.PROBE_FAR_KEYS."""
    import mlx.core as mx

    global _active
    _active = 1
    marks = [b for b in tileattn.boundaries(N8 + 3 * tileattn.GRAN * splits, splits) if b >= N8]
    marks = marks[:3]
    far = max(tileattn.far_mark(splits), marks[-1])
    mid_step = (marks[1] + marks[2]) // 2
    q_key, k_key, v_key, quant_key, mirror_key, module_key = mx.random.split(mx.random.key(0), 6)

    reason = _probe_quantiser(quant_key)
    if reason is not None:
        return reason
    # Room past the far mark for the parity rows, the decoy and the fence.
    cap = -(-(far + 2 * tileattn.MAX_RUN + 1) // 256) * 256
    kv_shape = (1, tileattn.HKV, cap, tileattn.D)
    kb = mx.random.normal(kv_shape, key=k_key).astype(mx.bfloat16)
    vb = mx.random.normal(kv_shape, key=v_key).astype(mx.bfloat16)
    qpool = mx.random.normal((1, 20, tileattn.HQ, tileattn.D), key=q_key).astype(mx.bfloat16)
    mx.eval(kb, vb, qpool)
    codes, sc = quantize(kb, vb, cap)
    mx.eval(codes, sc)
    fence = ((N8, 1), (mid_step, 3), (far + tileattn.MAX_RUN - 1, tileattn.MAX_RUN))
    reason = (
        _probe_exactness(kb, vb, qpool, (N8, far), splits)
        or _probe_parity(codes, sc, qpool, sorted({*marks, far}), splits)
        or _probe_fence(codes, sc, qpool, fence, splits)
        or _probe_mirror(mirror_key, held)
        or _probe_stock(kb, vb, codes, sc, qpool, (N8, far), splits)
    )
    # Hundreds of MB at the far mark: gone before the module step allocates.
    del kb, vb, qpool, codes, sc
    if reason is not None:
        return reason
    return _probe_module(modules, splits, verify_active, mid_step, module_key, held)


def probe(modules: list[Any], splits: int, arch: str, verify_active: bool) -> str | None:
    """None when kv8 may serve `modules` on this GPU at `splits` splits, else
    why not. Runs on the loading thread after availability(); `arch` is the
    tile's GPU architecture, taken for the tile probe's signature, and every
    row probed here is at or past N8 >= N0, where no stock plan runs, so no
    step reads it. Raises the flag and opens forced windows only while it
    runs; in `finally` it closes the window and drops every mirror it built
    (with their trim wraps), clears the flag, puts back the tile's flag and
    mlx-vlm's decode global, restores kv8's, the tile's and the projection
    kernel's counters (a quantized model's projections reach the projection
    kernel's hooks in the module step) and clears mlx's cache. Its arrays
    are gone before the cache is cleared, so the budget measured next reads
    true."""
    import mlx.core as mx

    global _active, _window
    language: Any = importlib.import_module(tileattn._DECODE_MODULE)
    installed = language.scaled_dot_product_attention
    tile_splits = tileattn._active_splits
    counted = (dict(calls), dict(tileattn.calls), dict(projkernel.calls))
    held: list[Any] = []
    try:
        return _probe_steps(modules, splits, verify_active, held)
    except BaseException as e:
        # The traceback holds the steps' frames, and every probe array in them,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        try:
            if held:
                if _window is None:
                    open_window(force=True)
                close_window(held)
        finally:
            held.clear()
            _active, _window = 0, None
            language.scaled_dot_product_attention = installed
            tileattn._active_splits = tile_splits
            calls.update(counted[0])
            tileattn.calls.update(counted[1])
            projkernel.calls.update(counted[2])
            mx.clear_cache()


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use kv8: report why. An `expected` refusal says only
    that kv8 does not apply here (the tile is not active, the window fits no
    ceiling above the floor), which is how most Macs load a default-on kv8:
    one INFO line, and the reason in the status. Any other refusal means it
    should have run and did not (a pinned source that drifted, kernels this OS
    rejects, a floor that is no cut point, a failed probe): one warning, since
    a switch that is silently inert there hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("int8 K/V mirror unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: int8 K/V mirror unavailable ({reason}); "
            "decode and verify attention read the bf16 cache",
            stacklevel=3,
        )
    return {
        "state": "unavailable",
        "reason": reason,
        "probe_seconds": probe_seconds,
        "floor": None,
        "ceiling": None,
    }


def _tile_reason(tile_status: Mapping[str, Any] | None) -> str | None:
    """kv8 rides on the attention tile: its platform gate, S, the chunk
    schedule, verify_plan, MAX_RUN, the reduce kernel and, with a drafter,
    exact verify attention are all the tile's."""
    state = (tile_status or {}).get("state")
    if state == "active":
        return None
    if state == "off":
        return "attention tile off"
    return (tile_status or {}).get("reason") or f"attention tile {state or 'unknown'}"


def _pins_reason() -> str | None:
    """Which pinned mlx-vlm function changed, or None. An unreadable source (a
    function mlx-vlm removed) counts as changed."""
    for (module, qualname), digest in VALIDATED_KV8_SOURCES.items():
        if verifyattn._source_digest(module, qualname) != digest:
            return f"mlx-vlm {qualname} changed"
    return None


def enable(
    model: Any,
    *,
    enabled: bool,
    tile_status: Mapping[str, Any] | None,
    verify_status: Mapping[str, Any] | None,
    window: int,
) -> dict[str, Any]:
    """Decide whether this load's decode and verify attention read int8 K/V
    from N8 keys up to the ceiling L. `window` is [model].max_context_tokens,
    which L is computed from. Returns the status the engine exposes and never
    raises: a model load must not fail because an accelerator is missing.
    probe_seconds covers the kernels' compile and the probe, kv8's whole cost
    to a load."""
    global _active, _window, _ceiling
    # First, on every call: a flag, a window or a ceiling left from an earlier
    # load must never serve this one, whatever happens below.
    _active, _window, _ceiling = 0, None, 0
    if not enabled:
        return {
            "state": "off",
            "reason": None,
            "probe_seconds": None,
            "floor": None,
            "ceiling": None,
        }
    seconds: float | None = None
    try:
        reason = _tile_reason(tile_status)
        if reason is not None:
            return _refuse(reason, expected=True)
        splits = int((tile_status or {}).get("splits") or 0)
        reason = _pins_reason()
        if reason is not None:
            return _refuse(reason)
        modules = verifyattn._attention_modules(model)
        # The exact verifier runs only beside a drafter, so only then does the
        # probe take its half.
        verify_active = (verify_status or {}).get("state") == "active"
        limit = 0
        expected = False
        start = time.perf_counter()
        try:
            available = availability()
            if not available.available:
                reason = available.reason or "the int8 K/V kernels did not run"
            elif (reason := _floor_reason(splits)) is not None:
                pass
            else:
                limit = ceiling(window, splits)
                if limit <= N8:
                    expected = True
                    reason = (
                        f"window below the kv8 floor (window {window}, ceiling {limit}, floor {N8})"
                    )
                else:
                    reason = probe(modules, splits, tileattn._arch(), verify_active)
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds, expected=expected)
        _active, _ceiling = 1, limit
        logger.info(
            "int8 K/V mirror: %d layers, %d <= keys < %d, probe %.2fs",
            len(modules),
            N8,
            limit,
            seconds,
        )
        return {
            "state": "active",
            "reason": None,
            "probe_seconds": seconds,
            "floor": N8,
            "ceiling": limit,
        }
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active, _window, _ceiling = 0, None, 0
        # The warning carries one line; the traceback goes to the log.
        logger.warning("int8 K/V mirror: enabling failed", exc_info=True)
        return _refuse(str(e) or type(e).__name__, seconds)
