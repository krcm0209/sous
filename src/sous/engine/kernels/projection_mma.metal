// Small-row affine-Q4 (group 64) projections, Y = X W^T for 1-8 bf16 activation
// rows and 1-4 linears that share them, computed as W X^T in 8x8x8 simdgroup MMAs:
// the weight's output columns in the MMA's rows, the activation rows in its
// columns. Each weight is the exact half (nibble | 0x6400) - 1024 and each
// activation bf16 widened to fp32, so every product is exact; per quant group
// acc = fma(dot, scale, acc), then acc = fma(sum(x), bias, acc); four simdgroups
// split K and their partials are added in a fixed order. Rows past TR are zeroed
// and their reads clamped, and a tile serves one linear, so a row's bits depend on
// neither the row count nor the linears beside it, and the three bodies below
// agree bitwise. The lane map, the register-operand MMA, the W X^T orientation
// with its nibble-pair reads and one-group prefetch, and the factored epilogue
// follow Splash (incoai/splash at f43509a); the exact half weights times fp32
// activations follow its Apple7/8 port (paperniuk/splash, d81795d and d2f902e).
// Both are under the Apache License 2.0; see THIRD_PARTY_NOTICES.md. The K split,
// the staging, the group sums and the multi-linear dispatch are sous's own.
// Prepended by projkernel.py with common.h + projection_mma.h; the Python side
// splits this file at the STAGED and GROUP_SUMS marker lines and fills in the
// linear-selection code.
//
// Plain body. Template: TR (rows, 1..8), KD (K), NLIN, N0..N3 (each linear's N),
// CT1..CT3 (each later linear's first tile). Grid (128, tiles, 1), threadgroup
// (128, 1, 1): one threadgroup per 32-column tile of one linear. A lane (fm, fn)
// holds weight column fm and activation rows fn, fn + 1; within a quant group, MMA
// k index 2c + e at step (h, s) is physical k 16c + 8h + s + 4e.
  constexpr int F = 4;    // 8-column fragments per simdgroup
  constexpr int NSG = 1;  // simdgroups along N
  constexpr int KS = 4;   // simdgroups along K: part of the summation order
  constexpr uint G = KD / 64;
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint sn = sgi % NSG, sk = sgi / NSG;
  uint tgy = threadgroup_position_in_grid.y;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  constexpr uint COLS = 8 * F * NSG;
  threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NSG * F * 32];

  // which linear this threadgroup's tile belongs to
  uint L = 0;
  __SELECT__

  const uint base = tgy * COLS + sn * (8 * F);
  const device uint8_t *wcol[F];
  const device bfloat *scol[F];
  const device bfloat *bcol[F];
  for (uint f = 0; f < F; ++f) {
    const uint n = min(base + f * 8 + fm, NL_ - 1);
    wcol[f] = (const device uint8_t *)W_ + ulong(n) * (KD / 2) + c * 8;
    scol[f] = (const device bfloat *)S_ + ulong(n) * G;
    bcol[f] = (const device bfloat *)B_ + ulong(n) * G;
  }
  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;

  float2 acc[F];
  for (uint f = 0; f < F; ++f) acc[f] = float2(0);

  // this simdgroup's quant groups
  const uint g0 = (sk * G) / KS, g1 = ((sk + 1) * G) / KS;

  // weights one group ahead
  uint2 wq[F];
  for (uint f = 0; f < F; ++f)
    wq[f] = (g0 < g1) ? *(const device uint2 *)(wcol[f] + ulong(g0) * 32) : uint2(0);
  for (uint g = g0; g < g1; ++g) {
    uint2 wv[F];
    for (uint f = 0; f < F; ++f) wv[f] = wq[f];
    if (g + 1 < g1)
      for (uint f = 0; f < F; ++f) wq[f] = *(const device uint2 *)(wcol[f] + ulong(g + 1) * 32);

    vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
    vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
    vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
    vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
    if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
    if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
    const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);

    // the group's activation sum per row, in a fixed tree across the fm lanes
    float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                       (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
    sm += simd_shuffle_xor(sm, 2);
    sm += simd_shuffle_xor(sm, 4);
    sm += simd_shuffle_xor(sm, 16);

    float2 dot[F];
    for (uint f = 0; f < F; ++f) dot[f] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
#pragma unroll
        for (uint f = 0; f < F; ++f) {
          const uint nib = ((h ? wv[f].y : wv[f].x) >> (4 * s)) & 0x000F000Fu;
          const float2 bf = h ? float2(xb0[s], xb1[s]) : float2(xa0[s], xa1[s]);
          const half2 a = as_type<half2>(nib | 0x64006400u) - half2(1024.0h);
          mma8<half, float>(dot[f], a, bf);
        }
      }
    }
    for (uint f = 0; f < F; ++f) {
      const float scale = float(scol[f][g]);
      const float bias = float(bcol[f][g]);
      acc[f] = fma(dot[f], scale, acc[f]);
      acc[f] = fma(sm, bias, acc[f]);
    }
  }

  // the K partials, added in simdgroup order
  if (KS > 1) {
    if (sk > 0)
      for (uint f = 0; f < F; ++f) red[(((sk - 1) * NSG + sn) * F + f) * 32 + lane] = acc[f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk > 0) return;
    for (uint k2 = 1; k2 < KS; ++k2)
      for (uint f = 0; f < F; ++f) acc[f] += red[(((k2 - 1) * NSG + sn) * F + f) * 32 + lane];
  }
  for (uint f = 0; f < F; ++f) {
    const uint n = base + f * 8 + fm;
    if (n >= NL_) continue;
    if (v0) Y_[ulong(r0) * NL_ + n] = static_cast<bfloat>(acc[f].x);
    if (v1) Y_[ulong(r1) * NL_ + n] = static_cast<bfloat>(acc[f].y);
  }

// STAGED
// Staged body: the plain body's arithmetic in the same order, with the weights
// brought in through threadgroup memory. Each simdgroup stages GB quant groups of its
// 32 columns with 16-byte loads, one stage ahead, and reads them back in the MMA
// lane map. Needs G % (KS * GB) == 0, so K % 1024 == 0. Template and launch as the
// plain body's; with PRESUM 1 it reads each group's activation sums from "sums"
// (the group-sums body below) instead of computing them.
  constexpr int F = 4;    // 8-column fragments per simdgroup
  constexpr int NSG = 1;  // simdgroups along N
  constexpr int KS = 4;   // simdgroups along K: part of the summation order
  constexpr int GB = 4;   // quant groups per staging stage
  constexpr uint G = KD / 64;
  constexpr uint NSGT = NSG * KS;
  constexpr uint CHUNKS = 8 * F * GB * 2;  // 16-byte chunks per stage per simdgroup
  constexpr uint NLD = CHUNKS / 32;        // uint4 loads per lane per stage
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint sn = sgi % NSG, sk = sgi / NSG;
  uint tgy = threadgroup_position_in_grid.y;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  constexpr uint COLS = 8 * F * NSG;
  threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NSG * F * 32];
  threadgroup uint2 stg_all[NSGT * GB * F * 32];
  threadgroup uint2 *stg = stg_all + sgi * (GB * F * 32);

  // which linear this threadgroup's tile belongs to
  uint L = 0;
  __SELECT__

  const uint base = tgy * COLS + sn * (8 * F);
  const device bfloat *scol[F];
  const device bfloat *bcol[F];
  for (uint f = 0; f < F; ++f) {
    const uint n = min(base + f * 8 + fm, NL_ - 1);
    scol[f] = (const device bfloat *)S_ + ulong(n) * G;
    bcol[f] = (const device bfloat *)B_ + ulong(n) * G;
  }
  // loader map: chunk q = i * 32 + lane -> column q / (2 GB), 16-byte chunk q % (2 GB)
  const device uint8_t *lsrc[NLD];
  uint ldst[NLD];
  for (uint i = 0; i < NLD; ++i) {
    const uint q = i * 32 + lane;
    const uint col = q / (2 * GB), ch = q % (2 * GB);
    const uint n = min(base + col, NL_ - 1);
    lsrc[i] = (const device uint8_t *)W_ + ulong(n) * (KD / 2) + ch * 16;
    // chunk ch = group gi (ch / 2), words 4 (ch & 1) .. +3 = uint2 slots c = 2 (ch & 1), +1
    const uint gi = ch >> 1, f = col >> 3, cm = col & 7;
    ldst[i] = ((gi * F + f) * 8 + cm) * 4 + 2 * (ch & 1);
  }
  const uint rd = fm * 4 + (fn >> 1);  // lane's uint2 slot within a (gi, f) block of 32

  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;

  float2 acc[F];
  for (uint f = 0; f < F; ++f) acc[f] = float2(0);

  const uint g0 = (sk * G) / KS, g1 = ((sk + 1) * G) / KS;  // multiples of GB
  uint4 pre[NLD];
  for (uint i = 0; i < NLD; ++i) pre[i] = *(const device uint4 *)(lsrc[i] + ulong(g0) * 32);
  for (uint gs = g0; gs < g1; gs += GB) {
    simdgroup_barrier(mem_flags::mem_threadgroup);  // previous stage fully read
    for (uint i = 0; i < NLD; ++i) {
      stg[ldst[i]] = pre[i].xy;
      stg[ldst[i] + 1] = pre[i].zw;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (gs + GB < g1)
      for (uint i = 0; i < NLD; ++i) pre[i] = *(const device uint4 *)(lsrc[i] + ulong(gs + GB) * 32);
    for (uint gi = 0; gi < GB; ++gi) {
      const uint g = gs + gi;
      uint2 wv[F];
      for (uint f = 0; f < F; ++f) wv[f] = stg[(gi * F + f) * 32 + rd];

      vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
      vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
      vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
      vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
      if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
      if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
      const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);
#if PRESUM
      const float2 sm = *(const device float2 *)(sums + g * 8 + fn);
#else
      float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                         (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
      sm += simd_shuffle_xor(sm, 2);
      sm += simd_shuffle_xor(sm, 4);
      sm += simd_shuffle_xor(sm, 16);
#endif

      float2 dot[F];
      for (uint f = 0; f < F; ++f) dot[f] = float2(0);
#pragma unroll
      for (uint h = 0; h < 2; ++h) {
#pragma unroll
        for (uint s = 0; s < 4; ++s) {
#pragma unroll
          for (uint f = 0; f < F; ++f) {
            const uint nib = ((h ? wv[f].y : wv[f].x) >> (4 * s)) & 0x000F000Fu;
            const float2 bf = h ? float2(xb0[s], xb1[s]) : float2(xa0[s], xa1[s]);
            const half2 a = as_type<half2>(nib | 0x64006400u) - half2(1024.0h);
            mma8<half, float>(dot[f], a, bf);
          }
        }
      }
      for (uint f = 0; f < F; ++f) {
        const float scale = float(scol[f][g]);
        const float bias = float(bcol[f][g]);
        acc[f] = fma(dot[f], scale, acc[f]);
        acc[f] = fma(sm, bias, acc[f]);
      }
    }
  }

  // the K partials, added in simdgroup order
  if (KS > 1) {
    if (sk > 0)
      for (uint f = 0; f < F; ++f) red[(((sk - 1) * NSG + sn) * F + f) * 32 + lane] = acc[f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk > 0) return;
    for (uint k2 = 1; k2 < KS; ++k2)
      for (uint f = 0; f < F; ++f) acc[f] += red[(((k2 - 1) * NSG + sn) * F + f) * 32 + lane];
  }
  for (uint f = 0; f < F; ++f) {
    const uint n = base + f * 8 + fm;
    if (n >= NL_) continue;
    if (v0) Y_[ulong(r0) * NL_ + n] = static_cast<bfloat>(acc[f].x);
    if (v1) Y_[ulong(r1) * NL_ + n] = static_cast<bfloat>(acc[f].y);
  }

// GROUP_SUMS
// Group-sums body: each quant group's activation sum per row, [K / 64, 8] fp32 with
// rows past TR zero, in exactly the plain body's arithmetic, so the staged body that
// reads it (PRESUM 1) agrees bitwise with the other two. Template: TR, KD. Grid
// (32 * K / 64, 1, 1), threadgroup (32, 1, 1): one simdgroup per group.
  const uint lane = thread_index_in_simdgroup;
  const uint g = threadgroup_position_in_grid.x;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;
  vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
  vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
  vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
  vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
  if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
  if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
  const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);
  float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                     (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
  sm += simd_shuffle_xor(sm, 2);
  sm += simd_shuffle_xor(sm, 4);
  sm += simd_shuffle_xor(sm, 16);
  if (fm == 0) { sums[g * 8 + fn] = sm.x; sums[g * 8 + fn + 1] = sm.y; }
