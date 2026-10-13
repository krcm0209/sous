// kv8's split kernel (kv8.py): the attention tile's relay kernel over int8 K, V and Q,
// for qwen3_5's decode and exact verifier: 24 query / 4 KV heads, head dim 256, scale
// 1/16, fp32 probabilities. Prepended by kv8.py with common.h + nax.h. The partials keep
// the tile's layout, so attention_tile.metal's reduce body combines them unchanged.
//
// Template: TR (rows T, 1..8), MR (= 6*TR packed rows per KV head, m = t*6 + h), RGT
// (ceil(MR/16) 16-row fragment groups). Grid (64*RGT, nsplit, 4), threadgroup
// (64*RGT, 1, 1): one threadgroup per (key split, KV head), two simdgroups per 16-row
// group, each running 8 of the 16 QK k-steps and half of the 256 output dims, as the
// tile's split body does. Inputs: q bf16 [1, 24, TR, 256]; codes int8 [1, 4, cap, 512],
// each token's 256 K codes then its 256 V codes, read as two strided views of the one
// buffer (V at +256); sc fp16 [1, 4, cap, 2] = {sK, sV} per (token, KV head), its last
// two axes contiguous and its head stride a multiple of 8 halves, so a lane's four
// consecutive keys' scales are one aligned 16-byte load; params = {n, chunk, nsplit}.
// Only the head and token strides are read. Row t sees keys [0, n - TR + t].
//
// Q is quantised here, per (query head, row): sq = amax / 127 in fp32, f = sq when
// sq > 0 and 1 otherwise (a zero row gets codes 0), code = clamp(rint(q / f), -127,
// 127). QK runs on int8 x int8 -> int32 rows and the relay adds the halves' int32
// partials, which is exact (|S| <= 256 * 127^2 < 2^24). The score is then
// float(Si) * (sq * (SCALE_LOG2 * sK)), each product rounded once in that order; sV
// folds into the probability fed to PV (int8 V on <float, int8_t, float> rows) and the
// row sum takes the unscaled probability. Nothing is dequantised in the loop. Every K,
// V and scale read is clamped to n - 1: that clamp, not the mask, keeps a NaN or a
// stale row past n out of the output (0 x NaN is NaN), and keys past a row's limit get
// probability exactly 0, so a row's output depends on `chunk` and nothing else about
// the call, as in the tile.
//
// With QDBG defined (kv8's debug kernel object only, which alone declares the qcodes
// output) the staged Q codes are also written out, [1, 24, TR, 256] int8 in q's dim
// order, for the tests to hold against the host's quantisation bit for bit.
  constexpr int GQ = 6;
  constexpr int DD = 256;
  constexpr float SCALE_LOG2 = 0.0625f * 1.4426950408889634f;
  constexpr float NEG = -1.0e30f;
  typedef float PE;  // fp32 probabilities into the PV matmul

  const ushort lane = ushort(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  // A division and a multiply-subtract, as in the tile: written as sg >> 1 and sg & 1
  // the tile ran 1.4x slower at T >= 3 (57K, M5 Pro, macOS 27.0), same output.
  const int rg = sg / 2;
  const int half_ = sg - rg * 2;
  const int nn0 = half_ * 4;
  const int split = int(threadgroup_position_in_grid.y);
  const int kvh = int(threadgroup_position_in_grid.z);
  const int n = params[0];
  const int chunk = params[1];
  const int nsplit = params[2];
  const int k_begin = split * chunk;
  const int k_end = min(n, k_begin + chunk);
  const int prefix = n - TR;

  const short2 cc = nax_coord(lane);
  const int fn = cc.x;
  const int fm = cc.y;

  const device int8_t* kbase = codes + kvh * codes_strides[1] + fn * 16;
  const device int8_t* vbase = kbase + DD + nn0 * 8;
  const int cst = int(codes_strides[2]);
  const device half* sbase = sc + kvh * sc_strides[1];

  int lim[2];
  const device bfloat* qp[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = min(rg * 16 + fm + 8 * i, MR - 1);
    const int t = m / GQ;
    const int h = m - t * GQ;
    lim[i] = prefix + t + 1;
    qp[i] = q + (GQ * kvh + h) * q_strides[1] + t * q_strides[2] + fn * 16;
  }

  // Q's scale per row: a lane holds 32 of each of its two rows' 256 dims (its half's 8
  // k-steps of 4), the four lanes that share a row (xor 1 and 8) this half's 128, and
  // the two halves trade their maxima through threadgroup memory (one barrier).
  bfloat4 qv[2][8];
  float qmax[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    float a = 0.0f;
    UNROLL for (int kq = 0; kq < 8; ++kq) {
      qv[i][kq] = *reinterpret_cast<const device bfloat4*>(qp[i] + (half_ * 8 + kq) * 4);
      const float4 x = fabs(float4(qv[i][kq]));
      a = max(a, max(max(x.x, x.y), max(x.z, x.w)));
    }
    a = max(a, simd_shuffle_xor(a, ushort(1)));
    a = max(a, simd_shuffle_xor(a, ushort(8)));
    qmax[i] = a;
  }
  threadgroup float qmx[RGT * 32];
  if (fn == 0) {
    UNROLL for (int i = 0; i < 2; ++i) qmx[(rg * 2 + half_) * 16 + fm + 8 * i] = qmax[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sqr[2];
  float qf[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const float a = max(qmax[i], qmx[(rg * 2 + 1 - half_) * 16 + fm + 8 * i]);
    sqr[i] = a / 127.0f;
    qf[i] = (sqr[i] > 0.0f) ? sqr[i] : 1.0f;
  }
  // Q codes staged once in threadgroup memory as packed char4, one uint per lane per row
  // per k-step: each half stages, and later reads, its own 8 k-steps, and each lane only
  // what it wrote itself, so no barrier is needed before the first read.
  threadgroup uint qtg[RGT * 1024];
  UNROLL for (int kq = 0; kq < 8; ++kq) {
    const int kk = half_ * 8 + kq;
    UNROLL for (int i = 0; i < 2; ++i) {
      const float4 c = clamp(rint(float4(qv[i][kq]) / qf[i]), -127.0f, 127.0f);
      const uint w = as_type<uint>(char4(c));
      qtg[((rg * 16 + kk) * 2 + i) * 32 + lane] = w;
#ifdef QDBG
      const int m = rg * 16 + fm + 8 * i;
      if (split == 0 && m < MR) {
        const int t = m / GQ;
        const int h = m - t * GQ;
        *reinterpret_cast<device uint*>(qcodes + (size_t(GQ * kvh + h) * TR + t) * DD + fn * 16 + kk * 4) = w;
      }
#endif
    }
  }
  // The relay: a lane's 16 int32 partial scores (four int4) per row group.
  threadgroup int4 xs[RGT * 4 * 32];
  threadgroup int4* xl = xs + (rg * 32 + lane) * 4;

  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
  auto ct_q = qk_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_k = qk_op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  using ct_q_t = metal::remove_addrspace_t<decltype(ct_q)>;
  using ct_k_t = metal::remove_addrspace_t<decltype(ct_k)>;
  auto ct_s = qk_op.template get_destination_cooperative_tensor<ct_q_t, ct_k_t, int32_t>();

  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
  auto ct_p = pv_op.template get_left_input_cooperative_tensor<PE, int8_t, float>();
  auto ct_v = pv_op.template get_right_input_cooperative_tensor<PE, int8_t, float>();
  using ct_p_t = metal::remove_addrspace_t<decltype(ct_p)>;
  using ct_v_t = metal::remove_addrspace_t<decltype(ct_v)>;
  // The output accumulators stay in the cooperative tensors for the whole split and
  // are rescaled in place: one per output tile of this half.
  auto ct_o0 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o1 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o2 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o3 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  UNROLL for (int e = 0; e < 16; ++e) { ct_o0[e] = 0.0f; ct_o1[e] = 0.0f; ct_o2[e] = 0.0f; ct_o3[e] = 0.0f; }
  float rmax[2] = {NEG, NEG};
  float rsum[2] = {0.0f, 0.0f};
  const int l_base = half_ * 8;

  for (int kb0 = k_begin; kb0 < k_end; kb0 += 32) {
    const device int8_t* kr[4];
    const device int8_t* vr[4];
    UNROLL for (int r = 0; r < 4; ++r) {
      const int key = min(kb0 + fm + 8 * r, n - 1);
      kr[r] = kbase + key * cst;
      vr[r] = vbase + key * cst;
    }
    // {sK, sV} of this lane's 8 score keys kb0 + hh*16 + fn + j: one 16-byte load per
    // 16-key half while the whole block lies below n, clamped half2 loads otherwise.
    half2 sv[2][4];
    if (kb0 + 32 <= n) {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        const uint4 w = *reinterpret_cast<const device uint4*>(sbase + (kb0 + hh * 16 + fn) * 2);
        sv[hh][0] = as_type<half2>(w.x); sv[hh][1] = as_type<half2>(w.y);
        sv[hh][2] = as_type<half2>(w.z); sv[hh][3] = as_type<half2>(w.w);
      }
    } else {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int j = 0; j < 4; ++j) {
          sv[hh][j] = *reinterpret_cast<const device half2*>(sbase + min(kb0 + hh * 16 + fn + j, n - 1) * 2);
        }
      }
    }
    UNROLL for (int e = 0; e < 16; ++e) ct_s[e] = 0;

    // Si_half = Q K^T over this half's 8 k-steps of 16 (permuted) dims, four k-steps per
    // 16-byte K load, accumulated in int32 in the cooperative tensor
    UNROLL for (int u = 0; u < 2; ++u) {
      uint4 kw[4];
      UNROLL for (int r = 0; r < 4; ++r) kw[r] = *reinterpret_cast<const device uint4*>(kr[r] + l_base * 4 + u * 16);
      UNROLL for (int s = 0; s < 4; ++s) {
        const int l = l_base + u * 4 + s;
        UNROLL for (int i = 0; i < 2; ++i) {
          const char4 qc = as_type<char4>(qtg[((rg * 16 + l) * 2 + i) * 32 + lane]);
          UNROLL for (int j = 0; j < 4; ++j) ct_q[i * 4 + j] = qc[j];
        }
        UNROLL for (int r = 0; r < 4; ++r) {
          const char4 b = as_type<char4>(s == 0 ? kw[r].x : s == 1 ? kw[r].y : s == 2 ? kw[r].z : kw[r].w);
          UNROLL for (int j = 0; j < 4; ++j) ct_k[r * 4 + j] = b[j];
        }
        qk_op.run(ct_q, ct_k, ct_s);
      }
    }
    int Si[16];
    UNROLL for (int e = 0; e < 16; ++e) Si[e] = ct_s[e];

    // relay: Si = Si_half0 + Si_half1 in int32, the same bits in both halves
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) xl[c4] = int4(Si[c4 * 4], Si[c4 * 4 + 1], Si[c4 * 4 + 2], Si[c4 * 4 + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 0) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const int4 o = xl[c4];
        Si[c4 * 4] += o.x; Si[c4 * 4 + 1] += o.y; Si[c4 * 4 + 2] += o.z; Si[c4 * 4 + 3] += o.w;
        xl[c4] = int4(Si[c4 * 4], Si[c4 * 4 + 1], Si[c4 * 4 + 2], Si[c4 * 4 + 3]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const int4 o = xl[c4];
        Si[c4 * 4] = o.x; Si[c4 * 4 + 1] = o.y; Si[c4 * 4 + 2] = o.z; Si[c4 * 4 + 3] = o.w;
      }
    }

    // scores in log2 units, Si * (sq * (SCALE_LOG2 * sK)); then the causal mask (keys past
    // n are clamped copies and always masked) and the online softmax (exp2)
    float S[16];
    UNROLL for (int hh = 0; hh < 2; ++hh) {
      UNROLL for (int j = 0; j < 4; ++j) {
        const float fk = SCALE_LOG2 * float(sv[hh][j].x);
        UNROLL for (int i = 0; i < 2; ++i) S[hh * 8 + i * 4 + j] = float(Si[hh * 8 + i * 4 + j]) * (sqr[i] * fk);
      }
    }
    if (kb0 + 32 > prefix + 1) {
      UNROLL for (int i = 0; i < 2; ++i) {
        UNROLL for (int hh = 0; hh < 2; ++hh) {
          UNROLL for (int j = 0; j < 4; ++j) {
            const int key = kb0 + hh * 16 + fn + j;
            if (key >= lim[i]) S[hh * 8 + i * 4 + j] = NEG;
          }
        }
      }
    }
    float fac[2];
    UNROLL for (int i = 0; i < 2; ++i) {
      float mx = NEG;
      UNROLL for (int j = 0; j < 4; ++j) mx = max(mx, max(S[i * 4 + j], S[8 + i * 4 + j]));
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float nm = max(rmax[i], mx);
      const float ref = (nm < -1.0e29f) ? 0.0f : nm;
      fac[i] = (nm == rmax[i]) ? 1.0f : fast::exp2(rmax[i] - ref);
      rmax[i] = nm;
      float ls = 0.0f;
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int j = 0; j < 4; ++j) {
          const float p = fast::exp2(S[hh * 8 + i * 4 + j] - ref);
          S[hh * 8 + i * 4 + j] = p;
          ls += p;
        }
      }
      rsum[i] = rsum[i] * fac[i] + ls;
    }
    if (simd_any(fac[0] != 1.0f || fac[1] != 1.0f)) {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int i = 0; i < 2; ++i) {
          UNROLL for (int j = 0; j < 4; ++j) {
            const int e = hh * 8 + i * 4 + j;
            ct_o0[e] *= fac[i]; ct_o1[e] *= fac[i]; ct_o2[e] *= fac[i]; ct_o3[e] *= fac[i];
          }
        }
      }
    }

    // O += (P sV) V over the block's two 16-key halves, this simdgroup's 4 output tiles;
    // one 16-byte V load covers two tiles
    UNROLL for (int ks = 0; ks < 2; ++ks) {
      UNROLL for (int e = 0; e < 8; ++e) ct_p[e] = PE(S[ks * 8 + e] * float(sv[ks][e & 3].y));
      UNROLL for (int np = 0; np < 2; ++np) {
        uint4 vw[2];
        UNROLL for (int i = 0; i < 2; ++i) vw[i] = *reinterpret_cast<const device uint4*>(vr[ks * 2 + i] + np * 16);
        UNROLL for (int u = 0; u < 2; ++u) {
          UNROLL for (int i = 0; i < 2; ++i) {
            const char4 lo = as_type<char4>(u ? vw[i].z : vw[i].x);
            const char4 hi = as_type<char4>(u ? vw[i].w : vw[i].y);
            UNROLL for (int j = 0; j < 4; ++j) { ct_v[i * 4 + j] = lo[j]; ct_v[8 + i * 4 + j] = hi[j]; }
          }
          const int nn = np * 2 + u;
          if (nn == 0) pv_op.run(ct_p, ct_v, ct_o0); else if (nn == 1) pv_op.run(ct_p, ct_v, ct_o1);
          else if (nn == 2) pv_op.run(ct_p, ct_v, ct_o2); else pv_op.run(ct_p, ct_v, ct_o3);
        }
      }
    }
  }

  // partials [kvh][split][m][256] (physical dim order) and {max (log2 units), sum} per row
  float O[4][16];
  UNROLL for (int e = 0; e < 16; ++e) { O[0][e] = ct_o0[e]; O[1][e] = ct_o1[e]; O[2][e] = ct_o2[e]; O[3][e] = ct_o3[e]; }
  UNROLL for (int i = 0; i < 2; ++i) {
    float s = rsum[i];
    s += simd_shuffle_xor(s, ushort(1));
    s += simd_shuffle_xor(s, ushort(8));
    rsum[i] = s;
  }
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = rg * 16 + fm + 8 * i;
    if (m < MR) {
      const size_t row = (size_t(kvh) * size_t(nsplit) + size_t(split)) * MR + m;
      device float* pp = part + row * DD + fn * 16 + nn0 * 8;
      UNROLL for (int nn = 0; nn < 4; ++nn) {
        *reinterpret_cast<device float4*>(pp + nn * 8) =
            float4(O[nn][i * 4 + 0], O[nn][i * 4 + 1], O[nn][i * 4 + 2], O[nn][i * 4 + 3]);
        *reinterpret_cast<device float4*>(pp + nn * 8 + 4) =
            float4(O[nn][8 + i * 4 + 0], O[nn][8 + i * 4 + 1], O[nn][8 + i * 4 + 2], O[nn][8 + i * 4 + 3]);
      }
      if (fn == 0 && half_ == 0) {
        stats[row * 2] = rmax[i];
        stats[row * 2 + 1] = rsum[i];
      }
    }
  }
