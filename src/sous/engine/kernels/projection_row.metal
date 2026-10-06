// One-row affine-Q4 (group 64) projection, y = x W^T for a single bf16
// activation row and one linear, in exactly the summation order the bodies of
// projection_mma.metal give a row, so its output equals theirs bit for bit:
// - the four K quarters own quant groups [q*G/4, (q+1)*G/4), as the MMA
//   kernel's four K simdgroups split them;
// - a group's dot starts from +0 and adds its 64 exact products in the MMA's
//   order (h, s outer, then MMA k = 2c + e, i.e. physical k 16c + 8h + s + 4e),
//   which is sequential C + p0 + ... + p7 inside every 8x8x8
//   simdgroup_multiply_accumulate on the Apple GPUs probed;
// - a group's activation sum is the MMA bodies' fixed tree;
// - per group acc = fma(dot, scale, acc), then acc = fma(sum_x, bias, acc),
//   and the quarters are added ((a0 + a1) + a2) + a3 before one bf16 rounding.
// Every product is exact, so only the dot chains, the epilogue and the reduce
// round, and each is written out as one explicit fma or add: under
// math_mode safe the compiler still contracts a plain acc + d * s into an fma,
// so an expression that looks unfused is not.
// Unlike the MMA kernel, whose 32-column tile spends most of its lanes on rows
// a one-row call does not have, each lane here takes whole quant groups of one
// column (lanes l, l + 32, ...), so a column's weights stream coalesced; the
// dots and sums meet in threadgroup memory, where four lanes fold one quarter's
// epilogue each in group order and shuffles add the quarters. The tile (RC
// columns per simdgroup, NSG simdgroups) changes no output, only the speed; its
// threadgroup memory, NSG * G * (RC + 1) fp32 values, bounds K, which
// projkernel.py routes past _ROW_MAX_K to the MMA kernel instead.
// sous's own, except the nibble decode and the two-fma epilogue it shares with
// projection_mma.metal, which follow Splash and its Apple7/8 port (Apache-2.0;
// see THIRD_PARTY_NOTICES.md). Prepended by projkernel.py with common.h.
//
// Template: KD (K), ND (N). Grid (32 * NSG * ceil(N / (RC * NSG)), 1, 1),
// threadgroup (32 * NSG, 1, 1).
  constexpr uint RC = 1;   // output columns per simdgroup
  constexpr uint NSG = 4;  // simdgroups per threadgroup
  constexpr uint KS = 4;   // K quarters: part of the summation order
  constexpr uint G = KD / 64;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint col0 = (threadgroup_position_in_grid.x * NSG + sg) * RC;
  // [sg][g][r]: column r's dot for group g; [sg][g][RC]: the group's sum of x
  threadgroup float dots[NSG][G][RC + 1];
  const device bfloat *xb = (const device bfloat *)x;
  const device uint8_t *wb = (const device uint8_t *)w;

  for (uint g = lane; g < G; g += 32) {
    const device vec<bfloat, 8> *xg = (const device vec<bfloat, 8> *)(xb + g * 64);
    float xf[64];
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const vec<bfloat, 8> v = xg[j];
#pragma unroll
      for (uint i = 0; i < 8; ++i) xf[8 * j + i] = float(v[i]);
    }
    // the MMA bodies' tree: each fm lane's eight values, then the butterfly
    float L[8];
#pragma unroll
    for (uint fm = 0; fm < 8; ++fm) {
      const uint o = 16 * (fm >> 1) + 4 * (fm & 1);
      L[fm] = ((xf[o] + xf[o + 1]) + (xf[o + 2] + xf[o + 3])) +
              ((xf[o + 8] + xf[o + 9]) + (xf[o + 10] + xf[o + 11]));
    }
    dots[sg][g][RC] = ((L[0] + L[1]) + (L[2] + L[3])) + ((L[4] + L[5]) + (L[6] + L[7]));

    for (uint r = 0; r < RC; ++r) {
      // a partial tile's spare columns read the last column and write nothing
      const uint n = min(col0 + r, uint(ND - 1));
      const device uint4 *wr = (const device uint4 *)(wb + ulong(n) * (KD / 2) + g * 32);
      const uint4 lo = wr[0], hi = wr[1];
      const uint wd[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float dot = 0.0f;
#pragma unroll
      for (uint h = 0; h < 2; ++h)
#pragma unroll
        for (uint s = 0; s < 4; ++s)
#pragma unroll
          for (uint c = 0; c < 4; ++c) {
            const uint j = 2 * c + h;
            const half2 a = as_type<half2>(((wd[j] >> (4 * s)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
            dot = fma(float(a.x), xf[8 * j + s], dot);
            dot = fma(float(a.y), xf[8 * j + s + 4], dot);
          }
      dots[sg][g][r] = dot;
    }
  }
  // each simdgroup reads back only its own dots
  simdgroup_barrier(mem_flags::mem_threadgroup);

  // lane 4r + q folds column r's quarter q, eight columns per pass
#pragma unroll
  for (uint pass = 0; pass < (RC + 7) / 8; ++pass) {
    const uint r = 8 * pass + (lane >> 2), q = lane & 3;
    const uint n = col0 + r;
    float acc = 0.0f;
    if (r < RC) {
      const uint nn = min(n, uint(ND - 1));
      const device bfloat *sc = (const device bfloat *)scales + ulong(nn) * G;
      const device bfloat *bi = (const device bfloat *)biases + ulong(nn) * G;
      const uint g0 = (q * G) / KS, g1 = ((q + 1) * G) / KS;
      for (uint g = g0; g < g1; ++g) {
        acc = fma(dots[sg][g][r], float(sc[g]), acc);
        acc = fma(dots[sg][g][RC], float(bi[g]), acc);
      }
    }
    const float a1 = simd_shuffle_down(acc, 1);
    const float a2 = simd_shuffle_down(acc, 2);
    const float a3 = simd_shuffle_down(acc, 3);
    if (r < RC && q == 0 && n < ND) y[n] = static_cast<bfloat>(((acc + a1) + a2) + a3);
  }
