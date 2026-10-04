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
independence is what lets verify rows and one-row decode go through the same
kernel and agree bit for bit.

The kernel needs no tensor units and no MPP, so it compiles on any Metal GPU with
half x float simdgroup MMA; ``kernel_available()`` is what its tests skip on. Its
lane map, W x^T orientation and epilogue follow Splash (incoai/splash), and its
exact half-weight decode Splash's Apple7/8 port (paperniuk/splash), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from sous.engine import nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.projkernel")

# Rows one kernel call serves: the MMA's eight columns.
MAX_ROWS = 8
# The kernel against a plain-mlx fp32 reference, in relative RMS. Its own error is
# bf16's rounding floor (~1e-3); a wrong lane map or K order lands near 1.
REL_RMS_BOUND = 1e-2
_MAX_LINEARS = 4
_GROUP = 64
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


def _projection_kernel(kind: str, nl: int) -> Callable[..., list[Any]]:
    """One kernel per variant and linear count: int8prefill's builder caches by name,
    and each count generates a different selection."""
    inputs = ["x", "sums"] if kind == "staged_ps" else ["x"]
    for i in range(nl):
        inputs += [f"w{i}", f"s{i}", f"b{i}"]
    return _kernel(
        f"sous_proj_{kind}_n{nl}",
        inputs,
        [f"y{i}" for i in range(nl)],
        _source(kind, nl),
        _header(kind == "staged_ps"),
        compile_options=_SAFE_MATH,
    )


def _group_sums_kernel() -> Callable[..., list[Any]]:
    return _kernel(
        "sous_proj_group_sums",
        ["x"],
        ["sums"],
        _bodies()[2],
        _header(False),
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
    weights, bf16 scales and biases, no bias term, and K / 64 at least the kernel's
    four-way K split. Any N: partial tiles are clamped on read and guarded on write."""
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


def _launch(kind: str, x2: Any, weights: list[tuple[Any, Any, Any]]) -> list[Any]:
    """One dispatch of one variant over [T, K] rows and 1..4 linears; [T, N_i] bf16
    each. Tiles are laid out linear by linear, a partial last tile per linear."""
    import mlx.core as mx

    rows, k = x2.shape
    nl = len(weights)
    ns = [int(w.shape[0]) for w, _, _ in weights]
    ct = [0]
    for n in ns:
        ct.append(ct[-1] + -(-n // _COLS))
    inputs = [x2, _group_sums(x2)] if kind == "staged_ps" else [x2]
    for w, scales, biases in weights:
        inputs += [w, scales, biases]
    template: list[tuple[str, Any]] = [("TR", rows), ("KD", k), ("NLIN", nl)]
    template += [(f"N{i}", ns[i] if i < nl else 0) for i in range(_MAX_LINEARS)]
    template += [(f"CT{i}", ct[i] if i < nl else 1 << 30) for i in (1, 2, 3)]
    return _projection_kernel(kind, nl)(
        inputs=inputs,
        template=template,
        grid=(_THREADS, ct[-1], 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, n) for n in ns],
        output_dtypes=[mx.bfloat16] * nl,
    )


def linears(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """x [..., K] bf16 with 1..MAX_ROWS rows (the product of its leading dimensions)
    through 1..4 affine-Q4 gs64 linears that share it, given as (weight uint32
    [N, K/8], scales bf16 [N, K/64], biases bf16 [N, K/64]); one [..., N_i] bf16 per
    linear. Raises ValueError for what the kernel cannot take; the hooks check their
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
    kind = variant(k, sum(int(w.shape[0]) for w, _, _ in weights))
    if k // _GROUP < _KS or (kind != "plain" and k % _STAGED_K):
        raise ValueError(f"projection kernel: K={k} does not fit the {kind} variant")
    outs = _launch(kind, x if x.ndim == 2 else x.reshape(rows, k), weights)
    if x.ndim == 2:
        return tuple(outs)
    return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outs)


def linear(x: Any, w: Any, scales: Any, biases: Any) -> Any:
    """One linear: linears() with a single (w, scales, biases)."""
    return linears(x, [(w, scales, biases)])[0]


# The seam the hooks call through, looked up at call time so tests can swap in a
# plain-mlx stand-in with the same contract.
mma = linears


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


def _relative_rms(got: Any, want: Any) -> float:
    import mlx.core as mx

    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _kernel_probe() -> str | None:
    """Compile and run every variant and linear count once on eight rows, plus each
    variant on one row alone; None when every output is right, else why not. Every
    call must equal the plain kernel's four-linear call bitwise (the variants and
    the linear counts share one arithmetic), the lone row must equal its row of the
    eight, and the plain call must sit within REL_RMS_BOUND of an fp32 dequantized
    reference: an Apple GPU whose MMA lane map differed from the probed one fails
    here instead of serving wrong numbers. The inputs are built with GPU ops and
    evaluated first: a compile failure with a CPU-stream op in flight deadlocks
    inside mlx's exception path (0.32.2) instead of raising."""
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
            error = _relative_rms(got, ref)
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
