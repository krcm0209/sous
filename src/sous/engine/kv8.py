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
import logging
import weakref
from collections.abc import Callable, Sequence
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
    the ceiling: for the load-time probe and the tests only."""

    serial: int
    state: str = "idle"
    rows: int = 0
    served: int = 0
    built: bool = False
    force: bool = False


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
    a straddling verify holds past L plus one capacity step. Being a step
    boundary, L is a verify_plan cut, so no run straddles it."""
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
    return the window's state and the rows it quantised. When a mirror was
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
    return window.state, window.rows


def serving_now() -> bool:
    """kv8 is active and a window is open that no call has capped."""
    window = _window
    return bool(_active) and window is not None and window.state != "capped"


def serves(n_row: int) -> bool:
    """Whether the row that sees n_row keys reads the mirror: N8 <= n_row < L in
    an open, uncapped window (a forced one serves from N8 whatever L). A row's
    path depends on its own key count alone, so a drafter-on and a drafter-off
    decode of one request serve the same rows. The seams look this up through
    the module on every call, so a harness may wrap it."""
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
    # A stale mirror's buffers are freed before the new ones are allocated.
    del stale
    cap = _capacity(cache.offset, window)
    codes, sc = quantize(cache.keys[:, :, :top, :], cache.values[:, :, :top, :], cap)
    mirror = Mirror(
        codes=codes, sc=sc, cap=cap, n=top, serial=window.serial, original_trim=original
    )
    object.__setattr__(cache, MIRROR_ATTR, mirror)
    calls["mirror_builds"] += 1
    window.built = True
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
