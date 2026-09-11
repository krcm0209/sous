"""INT8-activation prefill for affine-Q4/gs64 projections on the M5 tensor units.

Prefill GEMMs (>= MIN_ROWS rows) of tagged ``nn.QuantizedLinear`` projections run as
int8 activations x the checkpoint's own packed Q4 weights -> int32 on the neural
accelerators, with the affine correction applied per 64-code group. Two Metal
kernels live here as source strings and are compiled by mlx at model load
(``mx.fast.metal_kernel``): no native build, no ABI coupling to an mlx version.

Derived from oMLX (jundot/omlx#3548, Apache License 2.0): qwen35_oq_a8.metal,
qwen35_oq_a8_nax.metal and oq_a8_decode.h. See THIRD_PARTY_NOTICES.md.

Spec: docs/superpowers/specs/2026-09-11-int8-activation-prefill-design.md.
"""

# The GEMM kernel source (_GEMM_SOURCE below) is foreign C++/Metal text embedded
# verbatim as a raw string, ported line-for-line from oMLX for a fragment layout
# and K-order that must match Task 1's reorder_k exactly: ruff's Python line-length
# convention has no bearing on it, and wrapping it to fit would just be cosmetic
# risk against a correctness-sensitive transcription.
# ruff: noqa: E501

from __future__ import annotations

import logging
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger("sous.engine.int8prefill")

GROUP = 64
# Below this many rows the Stage-A pass costs more than the faster GEMM saves
# (crossover measured between 4 and 8 rows; 1.6x by 128). Set well above the
# crossover so decode (1 row) and speculative verify (<= 6 rows) never route.
MIN_ROWS = 128
# The PR's best Q4 tile: BM = 2 * 16 * WM rows, BN = 32 * WN columns, 32 * WM * WN
# threads per threadgroup. The seven tile variants are within ~3% of each other.
TILE_WM = 1
TILE_WN = 2
BM = 2 * 16 * TILE_WM
BN = 32 * TILE_WN

_COMMON_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
"""

# Stage A: one 256-thread threadgroup (eight simdgroups) per row. Each simdgroup owns
# whole 64-code groups, so the group sums never straddle simdgroups and reduce with
# one simd_sum. Ra is built from the ROUNDED codes, not from x: the affine bias
# correction in the GEMM is exact only against the codes actually multiplied.
_STAGE_A_SOURCE = r"""
  constexpr int SG = 8;
  const int row = int(threadgroup_position_in_grid.x);
  constexpr int groups = K / 64;
  const int simd_gid = int(simdgroup_index_in_threadgroup);
  const int simd_lid = int(thread_index_in_simdgroup);
  const device T* xr = x + size_t(row) * size_t(K);
  device int8_t* qr = qa + size_t(row) * size_t(K);
  device short* rr = ra + size_t(row) * size_t(groups);
  threadgroup float partial_amax[SG];
  float amax = 0.0f;
  for (int g = simd_gid; g < groups; g += SG) {
    const int base = g * 64 + simd_lid;
    amax = max(amax, abs(float(xr[base])));
    amax = max(amax, abs(float(xr[base + 32])));
  }
  amax = simd_max(amax);
  if (simd_lid == 0) partial_amax[simd_gid] = amax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_amax = 0.0f;
  UNROLL for (int i = 0; i < SG; ++i) row_amax = max(row_amax, partial_amax[i]);
  // An all-zero row has no representable scale: emit zeros rather than NaN and
  // let the affine bias term carry that row's output.
  const float inv = row_amax > 0.0f ? (127.0f / row_amax) : 0.0f;
  if (simd_gid == 0 && simd_lid == 0) sa[row] = row_amax > 0.0f ? (row_amax / 127.0f) : 0.0f;
  for (int g = simd_gid; g < groups; g += SG) {
    const int base = g * 64 + simd_lid;
    const int q0 = int(clamp(rint(float(xr[base]) * inv), -127.0f, 127.0f));
    const int q1 = int(clamp(rint(float(xr[base + 32]) * inv), -127.0f, 127.0f));
    qr[base] = int8_t(q0);
    qr[base + 32] = int8_t(q1);
    // |Ra| <= 64 * 127 = 8128, so int16 holds it.
    const int gsum = simd_sum(q0 + q1);
    if (simd_lid == 0) rr[g] = short(gsum);
  }
"""

_kernels: dict[str, Callable[..., list[Any]]] = {}


def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str,
) -> Callable[..., list[Any]]:
    """One metal_kernel object per name, built on first use. mlx compiles each
    (source, template) instantiation once per process behind it."""
    kernel = _kernels.get(name)
    if kernel is None:
        import mlx.core as mx

        kernel = cast(
            "Callable[..., list[Any]]",
            mx.fast.metal_kernel(
                name=name,
                input_names=input_names,
                output_names=output_names,
                source=source,
                header=header,
            ),
        )
        _kernels[name] = kernel
    return kernel


def reorder_k(qa: Any) -> Any:
    """Put each 64-code group into the GEMM's fragment K-order.

    Slot ``16c + 4t + j`` of a group holds code ``k = 16c + 8*(t>>1) + 2j + (t&1)``:
    the order in which the GEMM's nibble decode hands weights to the tensor-op
    fragment. Legal because a dot product is order-independent and the affine
    group is the same set of 64 either way, so scales, biases and Ra are untouched.
    Writing ``t = 2a + b`` the map is the transpose of the last two axes of
    ``(c, a, j, b)``; one contiguous copy of an int8 [M, K].
    """
    import mlx.core as mx

    m, k = qa.shape
    return mx.contiguous(
        qa.reshape(m, k // GROUP, 4, 2, 4, 2).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)
    )


def stage_a(x: Any) -> tuple[Any, Any, Any]:
    """bf16/fp16 ``[..., K]`` -> ``(qa int8 [M, K] reordered, sa f32 [M], ra int16 [K/64, M])``.

    Ra comes back group-major so a group's row sums are contiguous across the lanes
    that need them in the GEMM.
    """
    import mlx.core as mx

    k = x.shape[-1]
    m = x.size // k
    x2 = x.reshape(m, k)
    qa, sa, ra = _kernel(
        "sous_int8_stage_a", ["x"], ["qa", "sa", "ra"], _STAGE_A_SOURCE, _COMMON_HEADER
    )(
        inputs=[x2],
        template=[("T", x.dtype), ("K", k)],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,), (m, k // GROUP)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    )
    return reorder_k(qa), sa, mx.contiguous(ra.T)


_NAX_HEADER = (
    _COMMON_HEADER
    + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Mirror of mlx::steel::BaseNAXFrag::get_coord() (mlx 0.32.2, steel/gemm/nax.h): the
// lane's (column x, row y) inside a 16x16 fragment. A lane's slots r*4 + j are
// (row y + 8r, col x + j) of its fragment; a 16x32 destination is two such
// fragments along n.
inline short2 nax_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}
"""
)

# oq_a8_qmm_t_nax_v8<T, BITS=4, ACT_MODE=0, WM, WN> from oMLX, with K/N as template
# constants and M as a one-element input.
#
#   out[m, n] = sa[m] * sum_g( Sw[n,g] * sum_{k in g} qa[m,k]*qw[n,k] + Bw[n,g] * Ra[g,m] )
#
# The inner sum is four 16x32x16 int8 tensor-op steps per group into int32; the
# fp32 correction lands once per group as two fmas. Weights are the packed uint32
# words unmodified: one uint2 per weight row per group holds the lane's 16-code run
# and the nibbles are decoded straight into the right-hand fragment. Rows past M
# read row M-1 and are never stored, so the K loop has no bounds branch.
#
# scales/biases are read transposed [K/64, N] (four contiguous lanes' worth per
# load): the checkpoint's own stored [N, K/64] layout was also tried, reading
# straight off the projection with no per-call copy, but measured 20-25% slower
# here — over the plan's 3% budget — so qmm transposes once per call instead.
# Layout benchmark 2026-09-11, M=2048 (M5 Pro, applegpu_g17s), TOP/s stored vs
# transposed: gate_up 39.2/50.9, down 34.8/43.3, gdn_qkv 38.2/49.9, gdn_out
# 36.3/45.2 -> transposed kept.
_GEMM_SOURCE = r"""
  constexpr int kFragM = 16, kFragN = 32, kFragK = 16, kDestElems = 16, kSteps = 4;
  constexpr int TM = 2;
  constexpr int TBM = TM * kFragM * WM;
  constexpr int TBN = kFragN * WN;
  constexpr int words = 8;
  constexpr int groups = K / 64;
  const int M = Mp[0];

  const int sg = int(simdgroup_index_in_threadgroup);
  const int sg_m = sg % WM;
  const int sg_n = sg / WM;
  const int row_base = int(threadgroup_position_in_grid.y) * TBM + sg_m * (TM * kFragM);
  const int col_base = int(threadgroup_position_in_grid.x) * TBN + sg_n * kFragN;
  const short2 coord = nax_coord(thread_index_in_simdgroup);

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM, kFragN, kFragK, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kFragM, kFragN, kFragK, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  const int n_run0 = col_base + int(coord.x);
  const int m_base = row_base + int(coord.y);

  float Cf[TM][kDestElems];
  UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int e = 0; e < kDestElems; ++e) Cf[i][e] = 0.0f; }

  const device uint32_t* wbase = w + size_t(col_base + int(coord.y)) * size_t(groups) * words;
  const int w_stride8 = 8 * groups * words;
  const int w_stride16 = kFragM * groups * words;
  const int cx = int(coord.x) >> 2;

  for (int g = 0; g < groups; ++g) {
    uint2 wg[4];
    UNROLL for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 + (q >> 1) * w_stride16 + size_t(g) * words;
      wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
    }
    UNROLL for (int t = 0; t < kSteps; ++t) {
      UNROLL for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        const uint32_t word = wg[q][t >> 1];
        const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
      UNROLL for (int hf = 0; hf < 2; ++hf) {
        UNROLL for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + hf * kFragM + r * 8, M - 1);
          const char4 quad = as_type<char4>(*reinterpret_cast<const device uint32_t*>(
              qa + size_t(m) * size_t(K) + size_t(g) * 64 + size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else        { if (hf == 0) op.run(ct_a, ct_b, acc0);     else op.run(ct_a, ct_b, acc1); }
      }
    }
    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    const device T* srow = scales + size_t(g) * size_t(N);
    const device T* brow = biases + size_t(g) * size_t(N);
    UNROLL for (int h = 0; h < 2; ++h) {
      const int n0 = n_run0 + h * kFragM;
      sv[h] = *reinterpret_cast<const device vec<T, 4>*>(srow + n0);
      bv[h] = *reinterpret_cast<const device vec<T, 4>*>(brow + n0);
    }
    const device short* rrow = ra + size_t(g) * size_t(M);
    float r_g[TM][2];
    UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int r = 0; r < 2; ++r) {
      const int m = min(m_base + i * kFragM + r * 8, M - 1);
      r_g[i][r] = float(rrow[m]);
    } }
    UNROLL for (int e = 0; e < kDestElems; ++e) {
      const int r = ((e & 7) >> 2);
      const float swc = float(sv[e >> 3][e & 3]);
      const float bwc = float(bv[e >> 3][e & 3]);
      Cf[0][e] = metal::fma(swc, float(acc0[e]), metal::fma(bwc, r_g[0][r], Cf[0][e]));
      Cf[1][e] = metal::fma(swc, float(acc1[e]), metal::fma(bwc, r_g[1][r], Cf[1][e]));
    }
  }
  UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int e = 0; e < kDestElems; ++e) {
    const int ee = e & 7;
    const int r = ee >> 2;
    const int m = row_base + i * kFragM + int(coord.y) + r * 8;
    if (m < M) {
      const int n = col_base + (e >> 3) * kFragM + int(coord.x) + (ee & 3);
      out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(sa[m] * Cf[i][e]);
    }
  } }
"""


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


def availability() -> Availability:
    """Mirror of mlx's own is_nax_available() (device.cpp, 0.32.2): macOS >= 26.2
    and an Apple GPU of generation >= 17 (>= 18 for 'p'-suffix parts) — the M5
    family and later. Older GPUs run the Metal-4 tensor ops on the plain shader
    path, where int8 buys nothing, so they are `unavailable`, not merely slow."""
    if platform.system() != "Darwin":
        return Availability(False, "not macOS")
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return Availability(False, f"mlx unavailable: {e}")
    if not mx.metal.is_available():
        return Availability(False, "Metal unavailable")
    release = platform.mac_ver()[0]
    parts = tuple(int(p) for p in release.split(".") if p.isdigit())
    if parts[:2] < (26, 2):
        return Availability(False, f"macOS {release} < 26.2")
    arch = str(mx.device_info().get("architecture", ""))
    match = _ARCH.match(arch)
    if match is None:
        return Availability(False, f"unrecognized GPU architecture {arch!r}")
    gen = int(match.group(1))
    need = 18 if match.group(2) == "p" else 17
    if gen < need:
        return Availability(
            False, f"GPU {arch} predates the neural accelerators (gen {gen} < {need})"
        )
    return Availability(True)


def qmm(qa: Any, sa: Any, ra: Any, weight: Any, scales: Any, biases: Any) -> Any:
    """``[M, N]`` in ``scales.dtype`` from Stage-A outputs and a projection's packed
    affine-Q4 weight with its scales/biases in their stored ``[N, K/64]`` layout."""
    import mlx.core as mx

    m, k = qa.shape
    n = weight.shape[0]
    if n % BN or k % GROUP:
        raise ValueError(
            f"int8 prefill GEMM needs N % {BN} == 0 and K % {GROUP} == 0; got N={n} K={k}"
        )
    (out,) = _kernel(
        "sous_int8_q4_gemm",
        ["qa", "sa", "ra", "w", "scales", "biases", "Mp"],
        ["out"],
        _GEMM_SOURCE,
        _NAX_HEADER,
    )(
        inputs=[
            qa,
            sa,
            ra,
            weight,
            mx.contiguous(scales.T),
            mx.contiguous(biases.T),
            mx.array([m], dtype=mx.int32),
        ],
        template=[
            ("T", scales.dtype),
            ("WM", TILE_WM),
            ("WN", TILE_WN),
            ("K", k),
            ("N", n),
        ],
        grid=(32 * (n // BN), TILE_WM * ((m + BM - 1) // BM), TILE_WN),
        threadgroup=(32, TILE_WM, TILE_WN),
        output_shapes=[(m, n)],
        output_dtypes=[scales.dtype],
    )
    return out


def linear_int8(linear: Any, x: Any, stage: tuple[Any, Any, Any] | None = None) -> Any:
    """One tagged projection on the INT8 path; ``stage`` is a Stage A already taken
    for this exact ``x`` (shared across gate/up or qkv/z), else it is taken here."""
    qa, sa, ra = stage if stage is not None else stage_a(x)
    out = qmm(qa, sa, ra, linear.weight, linear.scales, linear.biases)
    return out.reshape(*x.shape[:-1], linear.weight.shape[0])
