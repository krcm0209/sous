// Stage A: one 256-thread threadgroup (eight simdgroups) per row. Each simdgroup owns
// whole 64-code groups, so the group sums never straddle simdgroups and reduce with
// one simd_sum. Ra is built from the ROUNDED codes, not from x: the affine bias
// correction in the GEMM is exact only against the codes actually multiplied.
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
  // An all-zero row has no representable scale: emit zero codes and sa = 0 rather than
  // NaN — the row's true product is zero and sa = 0 makes the GEMM store exactly that.
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
