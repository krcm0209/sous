// oq_a8_qmm_t_nax_v8<T, BITS=4, ACT_MODE=0, WM, WN> from oMLX, with K/N as template
// constants and M as a one-element input.
//
//   out[m, n] = sa[m] * sum_g( Sw[n,g] * sum_{k in g} qa[m,k]*qw[n,k] + Bw[n,g] * Ra[g,m] )
//
// The inner sum is four 16x32x16 int8 tensor-op steps per group into int32; the
// fp32 correction lands once per group as two fmas. Weights are the packed uint32
// words unmodified: one uint2 per weight row per group holds the lane's 16-code run
// and the nibbles are decoded straight into the right-hand fragment. Rows past M
// read row M-1 and are never stored, so the K loop has no bounds branch.
//
// scales/biases are read transposed [K/64, N] (four contiguous lanes' worth per
// load): the checkpoint's own stored [N, K/64] layout was also tried, reading
// straight off the projection with no per-call copy, but measured 20-25% slower
// here — more than the 3% a stored-layout read was allowed to cost — so qmm transposes once per call instead.
// Layout benchmark 2026-09-11, M=2048 (M5 Pro, applegpu_g17s), TOP/s stored vs
// transposed: gate_up 39.2/50.9, down 34.8/43.3, gdn_qkv 38.2/49.9, gdn_out
// 36.3/45.2 -> transposed kept.
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
