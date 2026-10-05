"""A simdgroup-MMA kernel for qwen3_5's small-row quantized projections.

Affine-Q4 (group 64) projections of 1..8 bf16 activation rows, 1..4 linears that
share the rows in one dispatch, run as 8x8x8 ``simdgroup_matrix`` MMAs
(``kernels/projection_mma.metal``): exact half weights times fp32 activations,
accumulated in fp32, with a fixed-order K split and per-group epilogue. A row's
bits depend on nothing else about the call: not on the row count, not on which
linears share the dispatch, not on which of the three variants served it. The
variants (plain; staged through threadgroup memory; staged with activation sums
presummed by a small prepare kernel) differ only in how they load, so the choice
between them is made by shape for speed and never changes an output. That
independence is what lets verify rows and one-row decode agree bit for bit.

A call of one row, which the MMA kernel serves at a fraction of its lanes, goes
instead to a one-row kernel (``kernels/projection_row.metal``) that computes the
same per-row arithmetic in the same order with one simdgroup per column, so its
output equals the MMA kernel's at one row bitwise; ``kernel_available()`` and the
load-time probe check that it does.

The kernel needs no tensor units and no MPP, so it compiles on any Metal GPU with
half x float simdgroup MMA; ``kernel_available()`` is what its tests skip on. Its
lane map, W x^T orientation and epilogue follow Splash (incoai/splash), and its
exact half-weight decode Splash's Apple7/8 port (paperniuk/splash), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import logging
import math
import time
import traceback
import warnings
from collections.abc import Callable
from typing import Any

from sous.engine import nax, tileattn, verifyattn
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text, _model_type, _root

logger = logging.getLogger("sous.engine.projkernel")

# Rows one kernel call serves: the MMA's eight columns.
MAX_ROWS = 8
# The kernel against a plain-mlx fp32 reference, in relative RMS. Its own error is
# bf16's rounding floor (~1e-3); a wrong lane map or K order lands near 1.
REL_RMS_BOUND = 1e-2
_MAX_LINEARS = 4
_GROUP = 64
# mlx binds a kernel input of fewer elements than this in the constant address
# space, and the bodies read every input through device pointers, so such an input
# fails to compile. Of a projection's arrays only the scales and biases of a
# one-column linear at K < 512 fall under it.
_MIN_DEVICE_SIZE = 8
# The Metal bodies' frozen F = 4, NSG = 1, KS = 4 and GB = 4, as the launch needs
# them: 32 columns per threadgroup, 128 threads, K / 64 >= KS for the plain body,
# K % (64 * KS * GB) == 0 for the staged one.
_COLS = 32
_THREADS = 128
_KS = 4
_STAGED_K = 1024
# Shape rules for speed, measured on the M5 Pro's 27B projections; the variants
# agree bitwise, so these never change an output. Presummed activations pay from
# lm_head's width; staging pays once the weights reach about 20 MB.
_PRESUM_N = 100_000
_STAGED_WORK = 40_000_000

VARIANTS = ("plain", "staged", "staged_ps")
# projection_mma.metal holds the plain body, this line, the staged body, the next
# line, then the group-sums body.
_STAGED_MARK = "// STAGED\n"
_SUMS_MARK = "// GROUP_SUMS\n"
# Where each body takes the code that picks its linear.
_SELECT = "__SELECT__"
# mlx's default, stated rather than inherited: the variants agree bitwise only if
# each rounds exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}
# The one-row body's frozen RC = 1 and NSG = 4: columns per threadgroup, threads.
_ROW_COLS = 4
_ROW_THREADS = 128
# The widest K the one-row kernel takes: its group dots and sums, NSG * G * 2
# fp32 values, must fit the 32 KB of threadgroup memory every Apple GPU has. A
# one-row call past it goes to the MMA kernel, which computes the same bits.
_ROW_MAX_K = 32 * 1024 // (4 * 2 * 4) * _GROUP
_ROW_KERNEL = "sous_proj_row"


@functools.cache
def _bodies() -> tuple[str, str, str]:
    """The plain, staged and group-sums bodies."""
    plain, rest = _kernel_text("projection_mma.metal").split(_STAGED_MARK)
    staged, sums = rest.split(_SUMS_MARK)
    return plain, staged, sums


def _select(nl: int) -> str:
    """The code that points W_, S_, B_, Y_ and NL_ at this threadgroup's linear:
    linear l owns tiles [CT{l}, CT{l+1}), and nested ternaries keep the pointers in
    registers. One linear needs no choice."""
    if nl == 1:
        return "#define W_ w0\n#define S_ s0\n#define B_ b0\n#define Y_ y0\n  const uint NL_ = N0;"

    def pick(prefix: str) -> str:
        choice = f"{prefix}{nl - 1}"
        for i in range(nl - 2, -1, -1):
            choice = f"(L == {i} ? {prefix}{i} : {choice})"
        return choice

    lines = [
        "for (uint l = 1; l < NLIN; ++l) if (tgy >= (l == 1 ? CT1 : l == 2 ? CT2 : CT3)) L = l;",
        "tgy -= (L == 0 ? 0 : L == 1 ? CT1 : L == 2 ? CT2 : CT3);",
        f"const device uint32_t *W_ = {pick('w')};",
        f"const device bfloat *S_ = {pick('s')};",
        f"const device bfloat *B_ = {pick('b')};",
        f"device bfloat *Y_ = {pick('y')};",
        f"const uint NL_ = {pick('N')};",
    ]
    return "\n  ".join(lines)


@functools.cache
def _source(kind: str, nl: int) -> str:
    plain, staged, _ = _bodies()
    return (plain if kind == "plain" else staged).replace(_SELECT, _select(nl))


@functools.cache
def _header(presum: bool) -> str:
    switch = "#define PRESUM 1\n" if presum else ""
    return _kernel_text("common.h") + switch + _kernel_text("projection_mma.h")


@functools.cache
def _kernel_spec(kind: str, nl: int) -> tuple[str, list[str], list[str], str, str]:
    """A variant's name, input names, output names, source and header at one
    linear count: text only, so a launch does not build it again."""
    inputs = ["x", "sums"] if kind == "staged_ps" else ["x"]
    for i in range(nl):
        inputs += [f"w{i}", f"s{i}", f"b{i}"]
    return (
        f"sous_proj_{kind}_n{nl}",
        inputs,
        [f"y{i}" for i in range(nl)],
        _source(kind, nl),
        _header(kind == "staged_ps"),
    )


def _projection_kernel(kind: str, nl: int) -> Callable[..., list[Any]]:
    """One kernel per variant and linear count: int8prefill's builder caches by name,
    and each count generates a different selection."""
    return _kernel(*_kernel_spec(kind, nl), compile_options=_SAFE_MATH)


def _group_sums_kernel() -> Callable[..., list[Any]]:
    return _kernel(
        "sous_proj_group_sums",
        ["x"],
        ["sums"],
        _bodies()[2],
        _header(False),
        compile_options=_SAFE_MATH,
    )


def _row_source() -> str:
    return _kernel_text("projection_row.metal")


def _row_kernel() -> Callable[..., list[Any]]:
    """The one-row kernel: one body, its K and N template arguments."""
    return _kernel(
        _ROW_KERNEL,
        ["x", "w", "scales", "biases"],
        ["y"],
        _row_source(),
        _kernel_text("common.h"),
        compile_options=_SAFE_MATH,
    )


def variant(k: int, n_sum: int) -> str:
    """The variant for K and the summed N of a call's linears: never the row count,
    so every row count of a projection takes the same one."""
    if k % _STAGED_K == 0:
        if n_sum >= _PRESUM_N:
            return "staged_ps"
        if n_sum * k >= _STAGED_WORK:
            return "staged"
    return "plain"


def eligible(module: Any) -> bool:
    """An nn.QuantizedLinear the kernel decodes: affine, 4-bit, group 64, uint32
    weights, bf16 scales and biases, no bias term, K / 64 at least the kernel's
    four-way K split, and scales of at least _MIN_DEVICE_SIZE values. Any N past
    that: partial tiles are clamped on read and guarded on write."""
    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(module, nn.QuantizedLinear) or "bias" in module:
        return False
    if getattr(module, "mode", None) != "affine":
        return False
    if getattr(module, "bits", None) != 4 or getattr(module, "group_size", None) != _GROUP:
        return False
    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    if weight is None or scales is None or biases is None:
        return False
    if weight.dtype != mx.uint32 or weight.ndim != 2 or scales.ndim != 2:
        return False
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        return False
    if scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        return False
    if scales.size < _MIN_DEVICE_SIZE:
        return False
    k = scales.shape[1] * _GROUP
    return weight.shape[1] * 8 == k and k // _GROUP >= _KS


# --------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------


def _group_sums(x2: Any) -> Any:
    """Each quant group's activation sum per row of a contiguous [T, K] bf16 x, as
    [K / 64, 8] fp32 with rows past T zero: the staged body's own sums, computed
    once for every tile of a wide projection."""
    import mlx.core as mx

    rows, k = x2.shape
    (sums,) = _group_sums_kernel()(
        inputs=[x2],
        template=[("TR", rows), ("KD", k)],
        grid=(32 * (k // _GROUP), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k // _GROUP, 8)],
        output_dtypes=[mx.float32],
    )
    return sums


@functools.cache
def _dispatch(rows: int, k: int, ns: tuple[int, ...]) -> dict[str, Any]:
    """The launch arguments that T, K and each linear's N fix, built once per
    shape: a decoded token launches every projection of the model, and
    rebuilding these cost more Python than the launch itself. Tiles are laid
    out linear by linear, a partial last tile per linear."""
    import mlx.core as mx

    nl = len(ns)
    ct = [0]
    for n in ns:
        ct.append(ct[-1] + -(-n // _COLS))
    template: list[tuple[str, Any]] = [("TR", rows), ("KD", k), ("NLIN", nl)]
    template += [(f"N{i}", ns[i] if i < nl else 0) for i in range(_MAX_LINEARS)]
    template += [(f"CT{i}", ct[i] if i < nl else 1 << 30) for i in (1, 2, 3)]
    return {
        "template": template,
        "grid": (_THREADS, ct[-1], 1),
        "threadgroup": (_THREADS, 1, 1),
        "output_shapes": [(rows, n) for n in ns],
        "output_dtypes": [mx.bfloat16] * nl,
    }


def _launch(kind: str, x2: Any, weights: list[tuple[Any, Any, Any]]) -> list[Any]:
    """One dispatch of one variant over [T, K] rows and 1..4 linears; [T, N_i] bf16
    each."""
    rows, k = x2.shape
    inputs = [x2, _group_sums(x2)] if kind == "staged_ps" else [x2]
    for w, scales, biases in weights:
        inputs += [w, scales, biases]
    dispatch = _dispatch(rows, k, tuple(w.shape[0] for w, _, _ in weights))
    return _projection_kernel(kind, len(weights))(inputs=inputs, **dispatch)


@functools.cache
def _row_dispatch(k: int, n: int) -> dict[str, Any]:
    """The one-row launch arguments for K and N, built once per shape for the
    reason _dispatch is."""
    import mlx.core as mx

    return {
        "template": [("KD", k), ("ND", n)],
        "grid": (_ROW_THREADS * -(-n // _ROW_COLS), 1, 1),
        "threadgroup": (_ROW_THREADS, 1, 1),
        "output_shapes": [(1, n)],
        "output_dtypes": [mx.bfloat16],
    }


def _launch_row(x2: Any, w: Any, scales: Any, biases: Any) -> Any:
    """One [1, K] bf16 row through one linear on the one-row kernel; [1, N] bf16."""
    (y,) = _row_kernel()(inputs=[x2, w, scales, biases], **_row_dispatch(x2.shape[-1], w.shape[0]))
    return y


def linears(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """x [..., K] bf16 with 1..MAX_ROWS rows (the product of its leading dimensions)
    through 1..4 affine-Q4 gs64 linears that share it, given as (weight uint32
    [N, K/8], scales bf16 [N, K/64], biases bf16 [N, K/64]); one [..., N_i] bf16 per
    linear. One row goes to the one-row kernel, one dispatch per linear (a row's
    bits do not depend on the linears beside it), unless K is past _ROW_MAX_K.
    Raises ValueError for what the kernel cannot take; the hooks check their
    calls before reaching here, so a raise is a caller's bug, never a fallback."""
    import mlx.core as mx

    k = x.shape[-1]
    rows = x.size // k if k else 0
    if x.dtype != mx.bfloat16:
        raise ValueError(f"projection kernel needs bf16 activations, got {x.dtype}")
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"projection kernel takes 1..{MAX_ROWS} rows, got {rows}")
    if not 1 <= len(weights) <= _MAX_LINEARS:
        raise ValueError(f"projection kernel takes 1..{_MAX_LINEARS} linears, got {len(weights)}")
    for w, scales, _ in weights:
        if w.shape[-1] * 8 != k or scales.shape[-1] * _GROUP != k:
            raise ValueError(f"projection kernel: a linear's K is not the activations' K={k}")
        if scales.size < _MIN_DEVICE_SIZE:
            raise ValueError(
                f"projection kernel: a linear's {scales.size} scales are fewer than "
                f"{_MIN_DEVICE_SIZE}"
            )
    kind = variant(k, sum(int(w.shape[0]) for w, _, _ in weights))
    if k // _GROUP < _KS or (kind != "plain" and k % _STAGED_K):
        raise ValueError(f"projection kernel: K={k} does not fit the {kind} variant")
    x2 = x if x.ndim == 2 else x.reshape(rows, k)
    if rows == 1 and k <= _ROW_MAX_K:
        calls["one_row"] += len(weights)
        # `one_row` is read at call time, the seam tests swap a stand-in into.
        outs = [one_row(x2, w, scales, biases) for w, scales, biases in weights]
    else:
        outs = _launch(kind, x2, weights)
    if x.ndim == 2:
        return tuple(outs)
    return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outs)


def linear(x: Any, w: Any, scales: Any, biases: Any) -> Any:
    """One linear: linears() with a single (w, scales, biases)."""
    return linears(x, [(w, scales, biases)])[0]


# The seams the hooks and linears() call through, looked up at call time so
# tests can swap in plain-mlx stand-ins with the same contracts.
mma = linears
one_row = _launch_row


# --------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------

# The probe's shape: K = 1024 lets every variant run; the four widths cover partial
# tiles, N = 48 (in_proj_b/a) and a single fragment.
_PROBE_K = _STAGED_K
_PROBE_NS = (40, 72, 48, 8)
# The row the probe also computes alone, against its row in the eight-row call.
_PROBE_ROW = 3

_kernel_ok = False

# The cancelling layouts the one-row checks add to random rows. Random rows leave
# a reordered fp32 sum on the same bf16 almost everywhere at the probe's widths;
# in these the large terms cancel exactly, so the small result shows how it was
# summed, and each layout shows the reorders of one step:
# - "quarter" cancels the first K quarter against the last and "middle" the
#   second against the third: between them, any other order or pairing of the
#   four quarter partials (where a3 = -a0, (a3 + a2) + (a1 + a0) rounds a1 and
#   a2 as the kernel does, so "quarter" alone misses it);
# - "group" cancels a quarter of every quant group against another: the order
#   of a group's 64 products and the tree over its eight lane sums (where
#   another GPU's MMA summed in another order, the MMA kernel's rows would move
#   here);
# - "lane" cancels two values of each lane's eight against two of its others:
#   the tree inside a lane sum;
# - "pair" cancels whole quant groups against their neighbours, so a quarter's
#   running sum swells and returns: the epilogue's order and its two fmas
#   (separately rounded products move it).
CANCELLING = ("quarter", "middle", "group", "lane", "pair")
# How far the cancelling layouts' large values sit above the rest.
_CANCEL = 256.0


def _mirror(k: int, layout: str) -> tuple[Any, Any, int]:
    """(source columns, mirror columns, the offset from one to the other) of a
    cancelling layout, as [K] masks. Quarters are split by whole quant groups,
    as both kernels split K, and the narrowest quarter sets the width. Mirror
    columns lie in their source's quant group or a whole group of their own,
    so copied weight columns quantize to the same codes."""
    import mlx.core as mx

    col = mx.arange(k)
    groups = k // _GROUP
    if layout in ("quarter", "middle"):
        first, second = (0, _KS - 1) if layout == "quarter" else (1, 2)
        width = _GROUP * (groups // _KS)
        start = _GROUP * (first * groups // _KS)
        offset = _GROUP * (second * groups // _KS) - start
        source = (col >= start) & (col < start + width)
        return source, mx.roll(source, offset), offset
    if layout == "group":
        within = col % _GROUP
        return within < 16, within >= 48, 48
    if layout == "lane":
        # a lane's eight values are 16-column blocks' [o, o + 4) and [o + 8, o + 12)
        within = col % 16
        source = (within < 8) & (within % 4 < 2)
        return source, mx.roll(source, 8), 8
    if layout == "pair":
        source = (col // _GROUP % 4 == 0) & (col + _GROUP < k)
        return source, mx.roll(source, _GROUP), _GROUP
    raise ValueError(f"no cancelling layout {layout!r}")


def _cancelling_rows(rows: int, k: int, layout: str, key: Any) -> Any:
    """[rows, K] bf16: large values over a layout's source columns, exactly
    their negation over its mirror, small values elsewhere."""
    import mlx.core as mx

    source, mirror, offset = _mirror(k, layout)
    keys = mx.random.split(key, 2)
    small = (mx.random.normal((rows, k), key=keys[0]) * 1e-2).astype(mx.bfloat16)
    big = (mx.random.normal((rows, k), key=keys[1]) * _CANCEL).astype(mx.bfloat16)
    return mx.where(source, big, mx.where(mirror, -mx.roll(big, offset, 1), small))


def _cancelling_linear(n: int, k: int, layout: str, key: Any) -> tuple[Any, Any, Any]:
    """An affine-Q4 gs64 linear whose mirror columns copy its source columns:
    whole quant groups, or columns of one group, so the copies quantize to the
    same codes, and against _cancelling_rows the two spans' products cancel
    term by term."""
    import mlx.core as mx

    _, mirror, offset = _mirror(k, layout)
    dense = (mx.random.normal((n, k), key=key) * 0.02).astype(mx.bfloat16)
    dense = mx.where(mirror, mx.roll(dense, offset, 1), dense)
    w, scales, biases = mx.quantize(dense, group_size=_GROUP, bits=4)
    return w, scales, biases


def _kernel_probe() -> str | None:
    """Compile and run every variant and linear count once on eight rows, plus each
    variant on one row alone; None when every output is right, else why not. Every
    call must equal the plain kernel's four-linear call bitwise (the variants and
    the linear counts share one arithmetic), the lone row must equal its row of the
    eight, and the plain call must sit within REL_RMS_BOUND of an fp32 dequantized
    reference: an Apple GPU whose MMA lane map differed from the probed one fails
    here instead of serving wrong numbers. Last, every row through the one-row
    kernel must equal the MMA kernel's one-row call bitwise, on random rows and in
    each CANCELLING layout, where a kernel that sums in another order shows. The
    inputs are built with GPU ops and evaluated first: a compile failure with a
    CPU-stream op in flight deadlocks inside mlx's exception path (0.32.2)
    instead of raising."""
    import mlx.core as mx

    try:
        keys = mx.random.split(mx.random.key(0), 1 + len(_PROBE_NS))
        x = mx.random.normal((MAX_ROWS, _PROBE_K), key=keys[0]).astype(mx.bfloat16)
        weights = []
        for i, n in enumerate(_PROBE_NS):
            dense = mx.random.normal((n, _PROBE_K), key=keys[1 + i]) * 0.02
            w, scales, biases = mx.quantize(dense.astype(mx.bfloat16), group_size=_GROUP, bits=4)
            weights.append((w, scales, biases))
        refs = [
            x.astype(mx.float32)
            @ mx.dequantize(
                w, s.astype(mx.float32), b.astype(mx.float32), group_size=_GROUP, bits=4
            ).T
            for w, s, b in weights
        ]
        mx.eval(x, weights, refs)
        base = _launch("plain", x, weights)
        mx.eval(base)
        for i, (got, ref) in enumerate(zip(base, refs, strict=True)):
            error = tileattn._rel_rms(got, ref)
            if not error <= REL_RMS_BOUND:
                return f"linear {i} is off its reference by relative RMS {error:.3g}"
        for kind in VARIANTS:
            for nl in range(1, _MAX_LINEARS + 1):
                outs = _launch(kind, x, weights[:nl])
                mx.eval(outs)
                for i, out in enumerate(outs):
                    if not mx.array_equal(out, base[i]).item():
                        return f"{kind} differs from plain in linear {i} of a {nl}-linear call"
            row = _PROBE_ROW
            (alone,) = _launch(kind, x[row : row + 1], weights[:1])
            if not mx.array_equal(alone, base[0][row : row + 1]).item():
                return f"{kind}: a row alone differs from the same row among eight"
        cases = [("random", x, weights)]
        for j, layout in enumerate(CANCELLING):
            keys = mx.random.split(mx.random.key(1 + j), 1 + len(_PROBE_NS))
            rows = _cancelling_rows(MAX_ROWS, _PROBE_K, layout, keys[0])
            linears_ = [
                _cancelling_linear(n, _PROBE_K, layout, keys[1 + i])
                for i, n in enumerate(_PROBE_NS)
            ]
            mx.eval(rows, linears_)
            cases.append((f"{layout}-cancelling", rows, linears_))
        for label, rows, linears_ in cases:
            for i, linear_ in enumerate(linears_):
                got = [_launch_row(rows[r : r + 1], *linear_) for r in range(MAX_ROWS)]
                want = [_launch("plain", rows[r : r + 1], [linear_])[0] for r in range(MAX_ROWS)]
                if not mx.array_equal(mx.concatenate(got), mx.concatenate(want)).item():
                    return (
                        f"one-row: linear {i} differs from the MMA kernel at one row "
                        f"({label} activations)"
                    )
    except Exception as e:  # noqa: BLE001 — any failure means the kernel cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("projection kernel: the kernel failed to build or run:\n%s", e)
        return _error_line(str(e))
    return None


def kernel_available() -> nax.Availability:
    """Whether the kernel compiles and computes right on this GPU. Only success is
    remembered: a failure is probed again on the next call, so one bad moment
    cannot switch the kernel off for a long-lived daemon."""
    global _kernel_ok
    if _kernel_ok:
        return nax.Availability(True)
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return nax.Availability(False, f"mlx unavailable: {e}")
    if not mx.metal.is_available():
        return nax.Availability(False, "Metal unavailable")
    failed = _kernel_probe()
    if failed is not None:
        return nax.Availability(False, f"the projection kernel failed: {failed}")
    _kernel_ok = True
    return nax.Availability(True)


# --------------------------------------------------------------------------------
# Routing: the tags, the flag and the two hooks
# --------------------------------------------------------------------------------

# Set with object.__setattr__ so it lives in the module's __dict__: nn.Module's own
# __setattr__ would put a bool INTO the parameter tree.
_TAG = "_sous_projection_kernel"
# Set on all three wrappers so a second install never wraps again. functools.wraps
# copies the wrapped function's __dict__, so the mark is still found when int8's
# wrapper of nn.QuantizedLinear.__call__ sits on top of ours.
_HOOK_MARK = "_sous_projection_kernel_hook"
# 1 while the kernel serves, else 0: set only by an enable() that reached active,
# cleared first by every enable(), by clear() and by the probe on its way out. A
# plain int, so nothing it holds outlives an unload.
_active = 0
# Per projection call, not per turn: the tests read deltas, and so can a harness
# that wraps the daemon. Nothing is logged per turn.
calls: dict[str, int] = {
    "verify_kernel": 0,  # verifier _linear/_linears calls the kernel served
    "verify_split": 0,  # of those, calls over MAX_ROWS rows cut into runs
    "plain_kernel": 0,  # QuantizedLinear calls the kernel served
    "declined": 0,  # flag set and every linear tagged, refused for dtype or shape
    "one_row": 0,  # one-row kernel dispatches linears() served, one per linear
}
_VERIFIER_MODULE = "mlx_vlm.models.qwen3_5.speculative_verifier"


def _verifier_class() -> Any:
    return importlib.import_module(_VERIFIER_MODULE).Qwen3_5BatchInvariantForward


def _tagged(module: Any) -> bool:
    return bool(getattr(module, _TAG, False))


def tag(model: Any) -> int:
    """Tag every QuantizedLinear of the loaded target's language model, lm_head
    included, and return how many; enable() has checked each one eligible. The
    drafter's own linears live in its own tree and are never reached. The
    lm_head DFlash binds into the drafter is the target's own object, so the
    drafter's sampled head goes through the kernel too, which changes proposals
    only."""
    import mlx.nn as nn

    seen: set[int] = set()
    for _, module in _root(model).named_modules():
        if isinstance(module, nn.QuantizedLinear) and id(module) not in seen:
            object.__setattr__(module, _TAG, True)
            seen.add(id(module))
    return len(seen)


def untag(model: Any) -> None:
    """Take the tags off `model`; both hooks then hand its calls on."""
    for _, module in _root(model).named_modules():
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, False)


def clear() -> None:
    """Stop serving: both hooks hand every call on from here on. Tags left on an
    unloaded model are inert while the flag is 0, and untagging it would need a
    reference that keeps its weights alive."""
    global _active
    _active = 0


def _weights(module: Any) -> tuple[Any, Any, Any]:
    return module.weight, module.scales, module.biases


def _run_mma(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """mma over the call's linears, at most _MAX_LINEARS per launch: a fused
    group's rows equal its singles', so where a longer list is cut changes no
    bit. `mma` is read from the module at call time, the seam tests swap the
    plain-mlx stand-in into."""
    out: list[Any] = []
    for i in range(0, len(weights), _MAX_LINEARS):
        out.extend(mma(x, weights[i : i + _MAX_LINEARS]))
    return tuple(out)


def _serve_verify(linears: Any, x: Any) -> tuple[Any, ...] | None:
    """The kernel's outputs for one verifier call while the flag is set, else
    None, and the wrapper calls mlx-vlm's method with the call's own arguments.
    A call of more than MAX_ROWS rows (block 0 lets the drafter's own policy
    verify 16) is cut along T into runs of at most MAX_ROWS: a row's bits do not
    depend on the rows beside it, so every run equals its rows computed one at a
    time, as decode computes them."""
    if not linears or not all(_tagged(m) for m in linears):
        return None
    import mlx.core as mx

    if getattr(x, "ndim", 0) != 3 or x.dtype != mx.bfloat16:
        calls["declined"] += 1
        return None
    batch, rows = x.shape[0], x.shape[1]
    step = MAX_ROWS // batch if batch else 0
    if not step or not rows:
        calls["declined"] += 1
        return None
    weights = [_weights(m) for m in linears]
    calls["verify_kernel"] += 1
    if rows <= step:
        return _run_mma(x, weights)
    calls["verify_split"] += 1
    runs = [_run_mma(x[:, j : j + step], weights) for j in range(0, rows, step)]
    return tuple(mx.concatenate(list(parts), axis=1) for parts in zip(*runs, strict=True))


def _serve_plain(module: Any, x: Any) -> Any | None:
    """The kernel's output for one call of a tagged module while the flag is set,
    else None, and the wrapper calls the method it wrapped: mlx's, or int8's,
    which takes its own calls of 128 rows and more. Rows are the product of the
    leading dimensions, so this serves one-row decode, a last prefill chunk of
    at most MAX_ROWS rows and the one-row forward that ends every prefill
    segment, and nothing longer."""
    if not _tagged(module):
        return None
    import mlx.core as mx

    shape = x.shape
    if not 1 <= math.prod(shape[:-1]) <= MAX_ROWS:
        return None
    if len(shape) < 2 or x.dtype != mx.bfloat16:
        calls["declined"] += 1
        return None
    calls["plain_kernel"] += 1
    return _run_mma(x, [_weights(module)])[0]


def _wrap_linear(original: Callable[..., Any]) -> Callable[..., Any]:
    """The verifier hook for its single projections: o_proj, down_proj, out_proj
    and the verify head."""

    # functools.wraps: the source pins follow __wrapped__ back to mlx-vlm's body.
    @functools.wraps(original)
    def _linear(self: Any, linear: Any, x: Any) -> Any:
        if _active:
            out = _serve_verify((linear,), x)
            if out is not None:
                return out[0]
        return original(self, linear, x)

    setattr(_linear, _HOOK_MARK, True)
    return _linear


def _wrap_linears(original: Callable[..., Any]) -> Callable[..., Any]:
    """The verifier hook for its fused groups: q+k+v, gate+up and qkv+z+b+a."""

    @functools.wraps(original)
    def _linears(self: Any, linears: Any, x: Any) -> Any:
        if _active:
            out = _serve_verify(linears, x)
            if out is not None:
                return out
        return original(self, linears, x)

    setattr(_linears, _HOOK_MARK, True)
    return _linears


def _wrap_call(original: Callable[..., Any]) -> Callable[..., Any]:
    """The module hook: nn.QuantizedLinear.__call__, for every call on the target
    outside the verifier."""

    @functools.wraps(original)
    def call(self: Any, x: Any) -> Any:
        if _active:
            out = _serve_plain(self, x)
            if out is not None:
                return out
        return original(self, x)

    setattr(call, _HOOK_MARK, True)
    return call


def install_hooks() -> None:
    """Wrap the exact verifier's _linear and _linears and nn.QuantizedLinear's
    __call__, once per process: a method already carrying the mark is left alone.
    enable() calls this on the first activation only, and the wrappers are never
    removed; while the flag is 0 each costs one int read per call.

    The verifier is wrapped on the class because its projections reach mlx-vlm's
    ops through names imported into its own module, and verifyattn's _attention
    wrapper calls self._linears and self._linear, so it goes through the same
    hook. gemma4's, nemotron_h's and qwen4_exp's verifiers inherit the wrap: only
    the tags keep them stock. Each wrapper falls through to what it wrapped, so
    int8's wrapper of nn.QuantizedLinear.__call__ may sit above or below this
    one: int8 takes 128 rows and more, this one 8 and fewer."""
    import mlx.nn as nn

    verifier = _verifier_class()
    if not getattr(verifier._linear, _HOOK_MARK, False):
        verifier._linear = _wrap_linear(verifier._linear)
    if not getattr(verifier._linears, _HOOK_MARK, False):
        verifier._linears = _wrap_linears(verifier._linears)
    # Any: ty would reject a wrapper assigned over the method's own signature.
    quantized: Any = nn.QuantizedLinear
    if not getattr(quantized.__call__, _HOOK_MARK, False):
        quantized.__call__ = _wrap_call(quantized.__call__)


# ---- the gates, the pins, the warm-up, the probe and enable() ---------------------

# The dense Qwen3.5-family text model. The MoE variant reuses the exact verifier's
# class and the dense MLP for its shared expert, beside routed experts the kernel
# never serves, so it is refused by model type rather than by class.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
# DFlash drafts with modules of its own, which are never tagged, and verifies
# through the exact verifier's _linear and _linears. Other drafters are unmeasured.
SUPPORTED_DRAFTER_KINDS = frozenset({"dflash"})
# What decides that every target projection of at most MAX_ROWS rows reaches one
# of the two hooks, and that greedy verify takes its tokens from the hooked head:
# sha256 of inspect.getsource, keyed "module:qualname". getsource follows
# __wrapped__ through both hooks and int8's wrappers, so these are mlx's and
# mlx-vlm's own sources whatever is installed over them. Validated on mlx 0.32.2
# and mlx-vlm 0.7.2; add a digest only after re-reading that function against
# both hooks.
VALIDATED_PROJ_SOURCES: dict[str, str] = {
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linear": (
        "668b9a7064f30c63348028fb72f22119b73e2897e2d246024c80fbe54e0404dd"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears": (
        "5cdaf151d00f3ab314f4722b475eec88a5228060be36c321c1363314ab249e0a"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._feed_forward": (
        "df7c1083833e60422732d7b1b5dfabfb6a45f5ef073c8d1416810588dbf95434"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._gated_delta": (
        "3df1521798b530554798c23f5b2d386de0e88e2ac50fb9f391c5035d4cf8f30e"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._model": (
        "cec9745522cd79893849777daa1a22cfd7736f10309ecd480db7e4f7d4778fbb"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward.__call__": (
        "969ac680c045372ddc2dfc4abfac22503ad04f30b8102528f9d3a6c499e6309c"
    ),
    # Dispatches each verify layer to _gated_delta, _attention and _feed_forward.
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._layer": (
        "3c097499afff39b8bcf44c56d70b7a1f2803cf4d67ad905119fafca1d167891a"
    ),
    # Reaches linear_attn, self_attn and mlp as module calls in decode and prefill.
    "mlx_vlm.models.qwen3_5.language:Qwen3_5DecoderLayer.__call__": (
        "9feda6407e1129df917a478e5687d2bc7f53ea21c7986f641468b5749a90e10a"
    ),
    "mlx_vlm.models.qwen3_5.language:Qwen3_5MLP.__call__": (
        "dc08d40bfdf18cf60619caad43c76a1094f3de1b14d2c53b9e2228fa698bc40a"
    ),
    "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet.__call__": (
        "0ab0be71156568f50a248f898d431b657b36b399166e29e25c16c7a45f6cc81b"
    ),
    # Defined on Qwen3_5GatedDeltaNet itself in 0.7.2: decode's b and a projections.
    "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet._project_gates": (
        "3d1f08bfeaa29f50a1d95aa6d5dda3766a93bc7068e81f9ce99b75b304bdbee6"
    ),
    "mlx_vlm.generate.ar:generate_step": (
        "ace632f74028d998d34c8b867b44632c13816e0db4567d37401a0bc657dc7a6a"
    ),
    "mlx.nn:QuantizedLinear.__call__": (
        "f43bed1629a938c0f00e37b4a9c85db84be59dc93b49a27de97c44228e264b8b"
    ),
    "mlx_vlm.speculative.dflash:_dflash_verify": (
        "39e4a6cd343ae7e054ae3584ff6879fc895494e413041f099ce08da14e3fc097"
    ),
    "mlx_vlm.speculative.dflash:_dflash_verify_greedy": (
        "1b472dc300554ab6017495bdeb18480be431578aa6958ff7b35c0145e8412971"
    ),
    # Picks the verify call each round makes and takes its tokens from its logits.
    "mlx_vlm.speculative.dflash:_dflash_rounds": (
        "eee8dc9ab95a74ed87c74064607736003b7d0dafbdafc32c9fba91dc190d2b32"
    ),
}
# The probe's through-the-hooks rows: a verify depth above today's block of 4.
_PROBE_ROWS = 5
# The row the probe plants a NaN in: inside every T = MAX_ROWS call, at neither end.
_NAN_ROW = 3


def _in_features(module: Any) -> int:
    """K of an eligible projection, read off its per-group scales."""
    return module.scales.shape[1] * _GROUP


def _dispatch_key(group: tuple[Any, ...]) -> tuple[int, tuple[int, ...]]:
    """What picks a pipeline: K and each linear's N, in order. The variant follows
    from K and their sum, and T is the template's own argument."""
    return _in_features(group[0]), tuple(m.weight.shape[0] for m in group)


def _dispatch_groups(root: Any) -> list[tuple[Any, ...]]:
    """One linear group per dispatch key the model's projections can reach: the
    exact verifier's three fused groups, and every projection alone, the way
    decode, a short prefill tail and the verifier's singles call them."""
    import mlx.nn as nn

    groups: list[tuple[Any, ...]] = []
    for layer in root.model.layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            groups.append((gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a))
        else:
            attention = layer.self_attn
            groups.append((attention.q_proj, attention.k_proj, attention.v_proj))
        groups.append((layer.mlp.gate_proj, layer.mlp.up_proj))
    groups += [(m,) for _, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]
    unique: dict[tuple[int, tuple[int, ...]], tuple[Any, ...]] = {}
    for group in groups:
        unique.setdefault(_dispatch_key(group), group)
    return list(unique.values())


def warm_up(model: Any) -> None:
    """Compile and run every pipeline the model's projections can reach, each
    dispatch key at T = 1..MAX_ROWS and every linear's (K, N) on the one-row
    kernel: no compile first happens mid-turn, and a Metal toolchain that
    rejects either kernel fails here, inside enable()'s refusal path, rather
    than inside a user's turn. The inputs are zeros built on the GPU and
    evaluated before the first launch: a compile failure with a CPU-stream op
    in flight deadlocks mlx's exception path instead of raising. The counters
    are left as found, since a load serves no call."""
    import mlx.core as mx

    groups = _dispatch_groups(_root(model))
    inputs = {
        k: mx.zeros((1, MAX_ROWS, k), dtype=mx.bfloat16)
        for k in sorted({_in_features(group[0]) for group in groups})
    }
    mx.eval(list(inputs.values()))
    counted = dict(calls)
    try:
        singles: dict[tuple[int, int], Any] = {}
        for group in groups:
            weights = [_weights(m) for m in group]
            x = inputs[_in_features(group[0])]
            mx.eval([mma(x[:, :t], weights) for t in range(1, MAX_ROWS + 1)])
            for m in group:
                singles.setdefault((_in_features(m), m.weight.shape[0]), m)
        # The real `mma` already sends each T = 1 call above to the one-row
        # kernel; this states the one-row pipelines in their own right, so the
        # warm-up still covers them when `mma` is a stand-in that does not.
        for (k, _), m in singles.items():
            if k <= _ROW_MAX_K:
                mx.eval(one_row(inputs[k][0, :1], *_weights(m)))
    finally:
        calls.update(counted)


def _stock_distance(got: Any, want: Any) -> float:
    """The relative RMS distance of `got` from `want` in fp32; infinite when
    `got` is not finite everywhere. A NaN `want` gives NaN, which fails any
    `<=` bound, as it must."""
    import mlx.core as mx

    if not mx.all(mx.isfinite(got)).item():
        return float("inf")
    return tileattn._rel_rms(got, want)


def _check_group(group: tuple[Any, ...], names: list[str], key: Any) -> str | None:
    """One dispatch on its real weights: every row at T = 1..MAX_ROWS equals the
    row alone, each linear fused equals it alone, the result sits within
    REL_RMS_BOUND of stock quantized_matmul (parity cannot see a kernel wrong
    the same way on every call), and a NaN row leaves the others untouched."""
    import mlx.core as mx

    weights = [_weights(m) for m in group]
    x = mx.random.normal((1, MAX_ROWS, _in_features(group[0])), key=key).astype(mx.bfloat16)
    nan = mx.array(float("nan"), dtype=mx.bfloat16)
    x_nan = mx.where(mx.arange(MAX_ROWS)[None, :, None] == _NAN_ROW, nan, x)
    mx.eval(x, x_nan)
    alone = [mma(x[:, r : r + 1], weights) for r in range(MAX_ROWS)]
    refs = [mx.concatenate([rows[i] for rows in alone], axis=1) for i in range(len(group))]
    mx.eval(refs)
    del alone
    label = "+".join(names)
    full: tuple[Any, ...] = ()
    for t in range(1, MAX_ROWS + 1):
        full = mma(x[:, :t], weights)
        same = mx.stack([mx.array_equal(o, ref[:, :t]) for o, ref in zip(full, refs, strict=True)])
        if not mx.all(same).item():
            return f"T-invariance: {label} at T={t} differs from its rows alone"
    if len(group) > 1:
        for i, name in enumerate(names):
            (single,) = mma(x, [weights[i]])
            if not mx.array_equal(single, full[i]).item():
                return f"grouping: {name} fused in {label} differs from {name} alone"
    for (w, scales, biases), got, name in zip(weights, full, names, strict=True):
        want = mx.quantized_matmul(
            x, w, scales=scales, biases=biases, transpose=True, group_size=_GROUP, bits=4
        )
        distance = _stock_distance(got, want)
        if not distance <= REL_RMS_BOUND:
            return f"correctness: {name} is {distance:.3g} from stock (bound {REL_RMS_BOUND})"
    keep = mx.array([r for r in range(MAX_ROWS) if r != _NAN_ROW])
    for got, clean, name in zip(mma(x_nan, weights), full, names, strict=True):
        if not mx.array_equal(mx.take(got, keep, axis=1), mx.take(clean, keep, axis=1)).item():
            return f"isolation: a NaN in row {_NAN_ROW} changed the other rows of {name}"
    return None


def _probe_steps(root: Any) -> str | None:
    """Every input is made with GPU ops and evaluated before the launches that
    read it, for the deadlock warm_up() describes."""
    import mlx.core as mx

    global _active
    layers = list(root.model.layers)
    full = next((layer for layer in layers if not layer.is_linear), None)
    linear = next((layer for layer in layers if layer.is_linear), None)
    if full is None or linear is None:
        return "the model lacks a full-attention or a linear-attention layer"
    names = {id(module): path for path, module in root.named_modules()}

    # Through both hooks, on the first full-attention layer (layer 3 of the
    # 27B): the exact verifier's feed-forward over _PROBE_ROWS rows (gate+up
    # fused, then down) against the plain MLP one row at a time, as decode
    # calls it. Its rows must be identical, which is what greedy parity rests on.
    mlp = full.mlp
    verifier = _verifier_class()()
    x = mx.random.normal((1, _PROBE_ROWS, _in_features(mlp.gate_proj)), key=mx.random.key(0))
    x = x.astype(mx.bfloat16)
    mx.eval(x)
    before = dict(calls)
    _active = 1
    verified = verifier._feed_forward(mlp, x)
    mx.eval(verified)
    middle = dict(calls)
    plain = [mlp(x[:, r : r + 1]) for r in range(_PROBE_ROWS)]
    mx.eval(plain)
    _active = 0
    if middle["verify_kernel"] - before["verify_kernel"] != 2:
        return "the exact verifier's projections did not reach the kernel"
    if calls["plain_kernel"] - middle["plain_kernel"] != 3 * _PROBE_ROWS:
        return "the model's own projections did not reach the kernel"
    for r in range(_PROBE_ROWS):
        if not mx.array_equal(verified[:, r : r + 1], plain[r]).item():
            return f"parity: verify row {r} of {_PROBE_ROWS} differs from the one-row forward"
    del x, verified, plain

    # Directly, on real weights: the full-attention layer's q+k+v, o, gate+up and
    # down, the first linear-attention layer's qkv+z+b+a and out_proj (layer 0
    # of the 27B), and the head, which covers the staged variants.
    attention, gdn = full.self_attn, linear.linear_attn
    groups: list[tuple[Any, ...]] = [
        (attention.q_proj, attention.k_proj, attention.v_proj),
        (attention.o_proj,),
        (mlp.gate_proj, mlp.up_proj),
        (mlp.down_proj,),
        (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
        (gdn.out_proj,),
    ]
    head = getattr(root, "lm_head", None)
    if head is not None:
        groups.append((head,))
    for seed, group in enumerate(groups, start=1):
        reason = _check_group(group, [names.get(id(m), "?") for m in group], mx.random.key(seed))
        if reason is not None:
            return reason
    return _check_rows(root, names)


def _check_rows(root: Any, names: dict[int, str]) -> str | None:
    """The one-row kernel against the MMA kernel on every dispatch key the model
    reaches, on its real weights, and at every K on a synthetic linear per
    CANCELLING layout: a random row and a row in each layout, each through
    `one_row`, must equal its row of one `mma` call over all of them bitwise.
    Several rows, because at one row `mma` is the one-row kernel itself; the MMA
    kernel's rows do not depend on the row count, which _check_group and
    kernel_available() both hold it to. For the same reason the steps before
    this one, which compare rows of a several-row call with `mma` at one row,
    already compare the two kernels: on the real kernel a one-row kernel off on
    random rows fails there first, under their reasons, and this step adds the
    cancelling layouts and the linear shapes they do not reach."""
    import mlx.core as mx

    groups = [g for g in _dispatch_groups(root) if _in_features(g[0]) <= _ROW_MAX_K]
    cases = [([_weights(m) for m in g], [names.get(id(m), "?") for m in g]) for g in groups]
    kinds = ("random", *(f"{layout}-cancelling" for layout in CANCELLING))
    rows: dict[int, Any] = {}
    for i, k in enumerate(sorted({_in_features(g[0]) for g in groups})):
        keys = mx.random.split(mx.random.key(100 + i), 1 + 2 * len(CANCELLING))
        built = [mx.random.normal((1, k), key=keys[0]).astype(mx.bfloat16)]
        for j, layout in enumerate(CANCELLING):
            built.append(_cancelling_rows(1, k, layout, keys[1 + j]))
            linear_ = _cancelling_linear(_PROBE_NS[0], k, layout, keys[1 + len(CANCELLING) + j])
            cases.append(([linear_], [f"a {layout}-cancelling K={k} linear"]))
        rows[k] = mx.concatenate(built)
    mx.eval(list(rows.values()), [weights for weights, _ in cases])
    for weights, labels in cases:
        x = rows[weights[0][0].shape[1] * 8]
        refs = mma(x, weights)
        checks = [
            (label, kind, mx.array_equal(one_row(x[r : r + 1], *linear_), ref[r : r + 1]))
            for linear_, ref, label in zip(weights, refs, labels, strict=True)
            for r, kind in enumerate(kinds)
        ]
        same = mx.stack([check for *_, check in checks])
        mx.eval(same)
        for i, (label, kind, _) in enumerate(checks):
            if not same[i].item():
                return f"one-row: {label} differs from the MMA kernel ({kind} activations)"
    return None


def probe(model: Any) -> str | None:
    """None when the kernel may serve `model`'s projections on this GPU, else
    why not. Runs on the loading thread after enable() has tagged the model and
    installed the hooks; raises the flag only around the calls that must reach
    them, and leaves the flag clear and the counters as it found them. Its
    arrays are gone before the cache is cleared, so the budget measured next
    reads true."""
    import mlx.core as mx

    global _active
    counted = dict(calls)
    try:
        return _probe_steps(_root(model))
    except BaseException as e:
        # The traceback holds the steps' frames, and every probe array in them,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _active = 0
        calls.update(counted)
        mx.clear_cache()


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use the kernel: report why. An `expected` refusal says
    only that the kernel does not apply here (the tile, the model, the
    drafter), which is how most Macs load a default-on kernel: one INFO line,
    and the reason in the status. Any other refusal means it should have run
    and did not (a pinned dependency that drifted, a kernel this OS rejects, a
    failed probe): one warning, since a switch that is silently inert there
    hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("projection kernel unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: projection kernel unavailable ({reason}); "
            "verify and decode run the stock projections",
            stacklevel=3,
        )
    return {"state": "unavailable", "reason": reason, "probe_seconds": probe_seconds}


def _tile_reason(tile_status: dict[str, Any] | None) -> str | None:
    """The kernel rides on the attention tile: the platform, the core count and,
    with a drafter, exact verify attention are all the tile's gates, and block
    5 was measured only beside it."""
    state = (tile_status or {}).get("state")
    if state == "active":
        return None
    if state == "off":
        return "attention tile off"
    return (tile_status or {}).get("reason") or f"attention tile {state or 'unknown'}"


def _model_type_reason(model_type: Any) -> str | None:
    """Shared by enable(), which reads the loaded model's type, and static_reason(),
    which reads config.json's: one text for both."""
    if model_type in SUPPORTED_MODEL_TYPES:
        return None
    return f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)"


def _drafter_kind_reason(drafter_kind: str | None) -> str | None:
    """Shared by enable() and static_reason(), as _model_type_reason is."""
    if drafter_kind is None or drafter_kind in SUPPORTED_DRAFTER_KINDS:
        return None
    return f"drafter kind {drafter_kind!r} (dflash only)"


def _model_reason(model: Any) -> str | None:
    """Why this model's projections are not the ones the kernel takes, or None:
    every quantized projection of the language model must be eligible, since
    decode and verify route them all, and the activations must be bf16."""
    import mlx.core as mx
    import mlx.nn as nn

    reason = _model_type_reason(_model_type(model))
    if reason is not None:
        return reason
    root = _root(model)
    projections = [(p, m) for p, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]
    if not projections:
        return "no quantized projections"
    for path, module in projections:
        if not eligible(module):
            return f"projection {path} is not affine 4-bit gs64 with bf16 scales and no bias"
    dtype = root.model.norm.weight.dtype
    if dtype != mx.bfloat16:
        return f"activation dtype {dtype} is not bfloat16"
    return None


def _greedy_verify_reason(model: Any) -> str | None:
    """Why greedy verify would leave the hooks, or None. A target that defines
    speculative_verify_dflash_hidden would have it take its tokens from
    quantized_argmax, which neither hook sees. No qwen3_5 language model
    defines it in the validated mlx-vlm, so meeting it means mlx-vlm moved."""
    if hasattr(_root(model), "speculative_verify_dflash_hidden"):
        return (
            "the language model defines speculative_verify_dflash_hidden: "
            "greedy verify would take its tokens outside the kernel"
        )
    return None


def _pins_reason() -> str | None:
    """Which pinned dependency drifted, or None."""
    import mlx.core as mx

    version = mx.__version__  # ty: ignore[unresolved-attribute]
    if version not in verifyattn.VALIDATED_MLX:
        return f"mlx {version} not validated"
    for key, digest in VALIDATED_PROJ_SOURCES.items():
        module, qualname = key.split(":")
        if verifyattn._source_digest(module, qualname) != digest:
            owner = "mlx" if module.split(".")[0] == "mlx" else "mlx-vlm"
            return f"{owner} {qualname} changed"
    return None


def enable(
    model: Any,
    *,
    enabled: bool,
    drafter_kind: str | None,
    tile_status: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether this load's small-row target projections run on the
    kernel. Returns the status the engine exposes and never raises: a model
    load must not fail because an accelerator is missing. probe_seconds covers
    the warm-up and the probe, the kernel's whole cost to a load."""
    global _active
    # First, on every call: a flag left from an earlier load must never serve
    # this one, whatever happens below. Tags left on an earlier model are inert
    # while it is clear, and a reference kept to untag them would keep that
    # model's weights alive past its unload, so they stay.
    _active = 0
    if not enabled:
        return {"state": "off", "reason": None, "probe_seconds": None}
    seconds: float | None = None
    try:
        reason = _tile_reason(tile_status)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = _model_reason(model) or _drafter_kind_reason(drafter_kind)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = _greedy_verify_reason(model) or _pins_reason()
        if reason is not None:
            return _refuse(reason)
        tagged = 0
        start = time.perf_counter()
        try:
            try:
                warm_up(model)
            except Exception as e:  # noqa: BLE001 — a kernel this OS rejects refuses the load
                # The one-line reason cannot carry a compiler's full report; the log can.
                logger.warning("projection kernel: the warm-up failed", exc_info=True)
                reason = f"warm-up: {_error_line(str(e) or type(e).__name__)}"
            else:
                tagged = tag(model)
                install_hooks()
                reason = probe(model)
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            untag(model)
            return _refuse(reason, seconds)
        _active = 1
        logger.info("projection kernel: %d projections, probe %.2fs", tagged, seconds)
        return {"state": "active", "reason": None, "probe_seconds": seconds}
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active = 0
        with contextlib.suppress(Exception):
            untag(model)
        # The warning carries one line; the traceback goes to the log.
        logger.warning("projection kernel: enabling failed", exc_info=True)
        return _refuse(str(e) or type(e).__name__, seconds)


def static_reason(config: Any, model_config: dict, drafter_kind: str | None) -> str | None:
    """Why the kernel would not be active for this config, model and drafter,
    or None when it would be, decided without loading anything: sous tune
    resolves the block an unset speculative_block_size runs at by it. It
    mirrors enable()'s order (the switch, then the tile's machine and model
    gates, then the kernel's own) and reads the model from its config.json. A
    refusal only a load can see, a probe failure, pin drift or exact verify
    attention missing beside the drafter, is beyond it."""
    if not config.projection_kernel:
        return "projection kernel off"
    # No drafter kind: the kernel's own checks come first, and its drafter
    # rule (the same text) follows them.
    reason = tileattn.static_reason(config, model_config, None)
    if reason is not None:
        return reason
    # The tile's check answered for the tile's model types; enable() also
    # asks the kernel's own (_model_reason), which is a separate set.
    reason = _model_type_reason(model_config.get("model_type"))
    if reason is not None:
        return reason
    reason = _quantization_reason(model_config.get("quantization"))
    if reason is not None:
        return reason
    return _drafter_kind_reason(drafter_kind)


def _quantization_reason(quantization: Any) -> str | None:
    """The config.json side of _model_reason's eligibility: the checkpoint's
    default and every per-projection override under the language model must be
    affine 4-bit gs64. Embeddings never reach the kernel, so their overrides
    do not count; nor does the vision tower's."""
    if not isinstance(quantization, dict):
        return "unquantized weights (affine 4-bit gs64 only)"

    def spec(entry: dict) -> tuple[Any, Any, Any]:
        return entry.get("mode", "affine"), entry.get("bits"), entry.get("group_size")

    mode, bits, group = spec(quantization)
    if (mode, bits, group) != ("affine", 4, _GROUP):
        return f"quantization {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"
    for path, entry in quantization.items():
        if not (isinstance(entry, dict) and path.startswith("language_model.")):
            continue
        if path.endswith("embed_tokens"):
            continue
        mode, bits, group = spec(entry)
        if (mode, bits, group) != ("affine", 4, _GROUP):
            return f"projection {path} is {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"
    return None
