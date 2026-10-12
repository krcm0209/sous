// Exact GQA-packed attention for decode and verify on the tensor units: bf16,
// 24 query / 4 KV heads, head dim 256, scale 1/16, fp32 probabilities.
// Prepended by tileattn.py with common.h + nax.h (the split body) or common.h (the
// reduce body); the Python side splits this file at the REDUCE marker line.
//
// Split body. Template: TR (rows T, 1..8), MR (= 6*TR packed rows per KV head,
// m = t*6 + h), RGT (ceil(MR/16) 16-row fragment groups). Grid (64*RGT, nsplit, 4),
// threadgroup (64*RGT, 1, 1): one threadgroup per (key split, KV head), two
// simdgroups per 16-row group. The two halves of a group split the work twice over:
// each runs 8 of the 16 QK k-steps and the partial score tiles are summed once per
// 32-key block, in the fixed order half 0 + half 1, through threadgroup memory (two
// barriers per block); then each accumulates half of the 256 output dims. Before the
// relay both halves ran all 16 k-steps and kept only their own columns, so a third
// fragment group (T = 6..8) cost +50 % per chain at 57K; with it T = 6 costs +7 % over
// T = 5. The split only changes the order of one fp32 sum per score, and every row
// still depends on its chunk alone (the relay is per block and per row group).
// Threadgroup memory: 8 KB of Q and 2 KB of relay per row group, 30 KB at RGT = 3.
// params = {n, chunk, nsplit}. Row t sees keys [0, n - TR + t]; every K/V read is
// clamped to n - 1 and keys past a row's limit get probability exactly 0, so a row's
// output depends on `chunk` and nothing else about the call.
//
// Head-dim permutation: the QK reduction over D is order-free, so k-step kk /
// fragment column c reads physical dim (c/4)*64 + kk*4 + c%4 for both Q and K (each
// lane owns 64 contiguous dims of every row it touches: 16-byte loads). PV's output
// columns use the same trick: tile nn, half hf, column fn+j is physical dim
// fn*16 + nn*8 + hf*4 + j, so V loads are 16-byte runs and O partials are stored in
// order. q, k and v need innermost stride 1 and 16-byte-aligned rows; only the head
// and token strides are read.
  constexpr int GQ = 6;
  constexpr int DD = 256;
  constexpr float SCALE_LOG2 = 0.0625f * 1.4426950408889634f;
  constexpr float NEG = -1.0e30f;
  typedef float PE;  // fp32 probabilities into the PV matmul

  const ushort lane = ushort(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  // Keep these as a division and a multiply-subtract: written as sg >> 1 and sg & 1 (same values) the
  // compiler emits code that runs 1.4x slower at T >= 3 (57K, M5 Pro, macOS 27.0), same output.
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

  const device bfloat* kbase = k + kvh * k_strides[1] + fn * 16;
  const device bfloat* vbase = v + kvh * v_strides[1] + fn * 16 + nn0 * 8;
  const int kst = int(k_strides[2]);
  const int vst = int(v_strides[2]);

  int lim[2];
  const device bfloat* qp[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = min(rg * 16 + fm + 8 * i, MR - 1);
    const int t = m / GQ;
    const int h = m - t * GQ;
    lim[i] = prefix + t + 1;
    qp[i] = q + (GQ * kvh + h) * q_strides[1] + t * q_strides[2] + fn * 16;
  }
  // Q staged once in threadgroup memory in lane order; each half stages, and later
  // reads, its own 8 of the group's 16 k-steps.
  threadgroup bfloat4 qtg[RGT * 1024];
  UNROLL for (int kq = 0; kq < 8; ++kq) {
    const int kk = half_ * 8 + kq;
    UNROLL for (int i = 0; i < 2; ++i) {
      qtg[((rg * 16 + kk) * 2 + i) * 32 + lane] = *reinterpret_cast<const device bfloat4*>(qp[i] + kk * 4);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The relay: a lane's 16 partial scores (four float4) per row group.
  threadgroup float4 xs[RGT * 4 * 32];
  threadgroup float4* xl = xs + (rg * 32 + lane) * 4;

  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
  auto ct_q = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_k = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using ct_q_t = metal::remove_addrspace_t<decltype(ct_q)>;
  using ct_k_t = metal::remove_addrspace_t<decltype(ct_k)>;
  auto ct_s = qk_op.template get_destination_cooperative_tensor<ct_q_t, ct_k_t, float>();

  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
  auto ct_p = pv_op.template get_left_input_cooperative_tensor<PE, bfloat, float>();
  auto ct_v = pv_op.template get_right_input_cooperative_tensor<PE, bfloat, float>();
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
    const device bfloat* kr[4];
    const device bfloat* vr[4];
    UNROLL for (int r = 0; r < 4; ++r) {
      const int key = min(kb0 + fm + 8 * r, n - 1);
      kr[r] = kbase + key * kst;
      vr[r] = vbase + key * vst;
    }
    float S[16];
    UNROLL for (int e = 0; e < 16; ++e) ct_s[e] = 0.0f;

    // S_half = Q K^T over this half's 8 k-steps of 16 (permuted) dims, two k-steps per
    // 16-byte K load, accumulated in the cooperative tensor
    UNROLL for (int l2 = 0; l2 < 4; ++l2) {
      uint4 kw[4];
      UNROLL for (int r = 0; r < 4; ++r) kw[r] = *reinterpret_cast<const device uint4*>(kr[r] + l_base * 4 + l2 * 8);
      UNROLL for (int s = 0; s < 2; ++s) {
        const int l = l_base + l2 * 2 + s;
        UNROLL for (int i = 0; i < 2; ++i) {
          const bfloat4 qv = bfloat4(qtg[((rg * 16 + l) * 2 + i) * 32 + lane]);
          UNROLL for (int j = 0; j < 4; ++j) ct_q[i * 4 + j] = qv[j];
        }
        UNROLL for (int r = 0; r < 4; ++r) {
          const bfloat4 b = as_type<bfloat4>(s == 0 ? kw[r].xy : kw[r].zw);
          UNROLL for (int j = 0; j < 4; ++j) ct_k[r * 4 + j] = b[j];
        }
        qk_op.run(ct_q, ct_k, ct_s);
      }
    }
    UNROLL for (int e = 0; e < 16; ++e) S[e] = ct_s[e];

    // relay: S = S_half0 + S_half1, the same bits in both halves
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) xl[c4] = float4(S[c4 * 4], S[c4 * 4 + 1], S[c4 * 4 + 2], S[c4 * 4 + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 0) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const float4 o = xl[c4];
        S[c4 * 4] += o.x; S[c4 * 4 + 1] += o.y; S[c4 * 4 + 2] += o.z; S[c4 * 4 + 3] += o.w;
        xl[c4] = float4(S[c4 * 4], S[c4 * 4 + 1], S[c4 * 4 + 2], S[c4 * 4 + 3]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const float4 o = xl[c4];
        S[c4 * 4] = o.x; S[c4 * 4 + 1] = o.y; S[c4 * 4 + 2] = o.z; S[c4 * 4 + 3] = o.w;
      }
    }

    // scale, causal mask (keys past n are clamped copies and always masked), online softmax (exp2)
    UNROLL for (int e = 0; e < 16; ++e) S[e] *= SCALE_LOG2;
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

    // O += P V over the block's two 16-key halves, this simdgroup's 4 output tiles
    UNROLL for (int ks = 0; ks < 2; ++ks) {
      UNROLL for (int e = 0; e < 8; ++e) ct_p[e] = PE(S[ks * 8 + e]);
      UNROLL for (int nn = 0; nn < 4; ++nn) {
        uint4 vw[2];
        UNROLL for (int i = 0; i < 2; ++i) vw[i] = *reinterpret_cast<const device uint4*>(vr[ks * 2 + i] + nn * 8);
        UNROLL for (int i = 0; i < 2; ++i) {
          const bfloat4 lo = as_type<bfloat4>(vw[i].xy);
          const bfloat4 hi = as_type<bfloat4>(vw[i].zw);
          UNROLL for (int j = 0; j < 4; ++j) { ct_v[i * 4 + j] = lo[j]; ct_v[8 + i * 4 + j] = hi[j]; }
        }
        if (nn == 0) pv_op.run(ct_p, ct_v, ct_o0); else if (nn == 1) pv_op.run(ct_p, ct_v, ct_o1);
        else if (nn == 2) pv_op.run(ct_p, ct_v, ct_o2); else pv_op.run(ct_p, ct_v, ct_o3);
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

// REDUCE
// Reduce body. Template: MR. Grid (256, MR, 4), threadgroup (256, 1, 1). Combines a
// row's splits in split order (fixed order) and writes out [1, T, 24, 256], the
// model's [B, L, H, D] layout.
  constexpr int DD = 256;
  const int d = int(thread_position_in_threadgroup.x);
  const int m = int(threadgroup_position_in_grid.y);
  const int kvh = int(threadgroup_position_in_grid.z);
  const int nsplit = params[2];
  const device float* st = stats + (size_t(kvh) * size_t(nsplit) * MR + m) * 2;
  float gmax = -1.0e30f;
  for (int s = 0; s < nsplit; ++s) gmax = max(gmax, st[size_t(s) * MR * 2]);
  float num = 0.0f;
  float den = 0.0f;
  const device float* pp = part + (size_t(kvh) * size_t(nsplit) * MR + m) * DD + d;
  for (int s = 0; s < nsplit; ++s) {
    const float w = fast::exp2(st[size_t(s) * MR * 2] - gmax);
    num += w * pp[size_t(s) * MR * DD];
    den += w * st[size_t(s) * MR * 2 + 1];
  }
  const int t = m / 6;
  const int h = m - t * 6;
  out[(size_t(t) * 24 + 6 * kvh + h) * DD + d] = static_cast<bfloat>(num / den);
