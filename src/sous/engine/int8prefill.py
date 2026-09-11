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

from __future__ import annotations

import logging
from collections.abc import Callable
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
