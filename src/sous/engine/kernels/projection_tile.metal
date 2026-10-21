// The projection tile: Y = X W^T for 8 bf16 activation rows through one
// slab-ordered affine-Q4 (group 64) projection or stack of projections, on the
// tensor units. Rows past a call's T are zero rows of the padded [8, K] input, so T
// is never an argument and every row's bits are those of the same row alone. The
// weights are slab.py's planes: codes [tiles][G][256][8 words], scales and biases
// [tiles][G][256], N zero-padded to whole 256-column tiles; one group of a tile's TN
// columns is one contiguous run of 32-byte units, which is what lets the tensor op
// stream it. Prepended by projtile.py with common.h, the DUAL and SPLITOUT defines,
// nax.h and projection_tile.h; the Python side splits this file at its REDUCE marker
// line.
//
// Derived from incoai/splash, runtime/metal/kernels/common/q4_mpp_tiles.h (the tile
// and its sums) and runtime/metal/kernels/decode/linear_q4.metal (the persistent
// loop) at 0d5a035, under the Apache License 2.0; see THIRD_PARTY_NOTICES.md.
//
// Tile body. Template: N (the padded width written), K, TN (columns per tile, 128 or
// 256), S (simdgroups, 4 or 8), GROUPS (persistent threadgroups), PIPE (two groups'
// matmuls issued before their epilogues; the epilogues stay in group order), SPLITS
// (grid y; split s covers the quant groups [s*B/SPLITS*4, (s+1)*B/SPLITS*4) of the
// B = K/256 blocks). Grid (GROUPS*S*32, SPLITS, 1), threadgroup (S*32, 1, 1).
// Inputs x [8, K] bf16, w0/sc0/bi0 the planes (w1/sc1/bi1 under DUAL); outputs
// y0 (y1) [8, N] bf16, or under SPLITOUT [SPLITS, 8, N] fp32. Per element, from
// +0.0f and per quant group in ascending order: P = the group's 64 exact products
// (one matmul2d into a fresh fp32 cooperative tensor), then
// acc = fma(P, scale, acc); acc = fma(rowsum, bias, acc); one bf16 rounding.
//
// The row sums live in a 64-float threadgroup array, two halves of a block's four
// groups by eight rows, behind three barriers: after the prologue fills the first
// block's sums, after each refill of the next block's (in FINISH_GROUP), and at the
// end of every tile, since the next tile's prologue rewrites the first half while a
// straggling simdgroup may still be reading it. No output test can see a missing
// prologue or closing barrier, so tests/test_projtile.py pins all three by count.
  constexpr uint G = K / 64;
  constexpr uint BLOCKS = K / 256;
  // Nibbles between adjacent columns of a group's B tensor (one 32-byte unit per
  // column), and bytes between a tile's consecutive groups (256 units).
  constexpr int CSTRIDE = 64;
  constexpr ulong GSTRIDE = 256ul * 32ul;
  threadgroup float input_sums[64];
  const uint group = threadgroup_position_in_grid.x;
  const uint split = threadgroup_position_in_grid.y;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  // Splash's Split128 ranges (q4_mpp_tiles.h:327-328); SPLITS = 1 gives [0, G).
  const uint first_group = split * BLOCKS / SPLITS * 4;
  const uint end_group = (split + 1) * BLOCKS / SPLITS * 4;
  // mlx hands every input as const device; the tensor types take plain pointers.
  const device bfloat *in = (const device bfloat *)x;
  device bfloat *inm = (device bfloat *)x;
  device uchar *wb0 = (device uchar *)w0;
  const device bfloat *sc0p = (const device bfloat *)sc0;
  const device bfloat *bi0p = (const device bfloat *)bi0;
#if DUAL
  device uchar *wb1 = (device uchar *)w1;
  const device bfloat *sc1p = (const device bfloat *)sc1;
  const device bfloat *bi1p = (const device bfloat *)bi1;
#endif
  auto a = tensor(inm, dextents<int, 2>{int(K), 8}, array<int, 2>{1, int(K)});
  // mode::multiply stated: every group writes a fresh partial, never accumulates.
  constexpr auto descriptor = matmul2d_descriptor(
      8, TN, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<S>> op;
  for (uint tile_index = group; tile_index < N / TN; tile_index += GROUPS) {
    const uint origin = tile_index * TN;
    const uint tile = origin / 256u;
    const uint tile_offset = origin % 256u;
    // The slab tile's base plus tile_offset columns of 32 bytes
    // (q4_mpp_tiles.h:156-161, 199-201).
    device uchar *tw0 = wb0 + ulong(tile) * G * 256ul * 32ul + ulong(tile_offset) * 32ul;
#if DUAL
    device uchar *tw1 = wb1 + ulong(tile) * G * 256ul * 32ul + ulong(tile_offset) * 32ul;
#endif
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
        tw0, dextents<int, 2>{64, int(TN)}, array<int, 2>{1, CSTRIDE});
    auto a0 = a.slice<64, 8>(0, 0);
    auto b00 = first_b0.slice<64, TN>(0, 0);
    // MSL 4.1 types a local cooperative tensor with its address space; strip it.
    using A0 = metal::remove_addrspace_t<decltype(a0)>;
    using B0 = metal::remove_addrspace_t<decltype(b00)>;
    auto acc0 = op.template get_destination_cooperative_tensor<A0, B0, float>();
    using Acc = metal::remove_addrspace_t<decltype(acc0)>;
    const ushort CAP = acc0.get_capacity();
    // Splash q4_mpp_tiles.h:176-180: all elements only when the capacity covers
    // exactly 8 x TN.
    const bool fully_ = uint(CAP) * (uint(S) * 32u) == 8u * uint(TN);
    const ushort trav = fully_ ? ushort(0) : q4_traversal(acc0);
#if DUAL
    Acc acc1;
#endif
    VISIT(
      acc0[i] = 0.0f;
      acc1_zero(i);
    )
    q4_store_input_sums<8, S>(in, K, first_group * 64, input_sums,
                              ((first_group >> 2) & 1u) * 32u, lane, sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint qg = first_group;
    if constexpr (PIPE) {
      for (; qg + 1 < end_group; qg += 2) {
        Acc f0, s0;
#if DUAL
        Acc f1, s1;
#endif
        RUN_GROUP(qg, f0, f1);
        RUN_GROUP(qg + 1, s0, s1);
        FINISH_GROUP(qg, f0, f1);
        FINISH_GROUP(qg + 1, s0, s1);
      }
    }
    for (; qg < end_group; ++qg) {
      Acc p0;
#if DUAL
      Acc p1;
#endif
      RUN_GROUP(qg, p0, p1);
      FINISH_GROUP(qg, p0, p1);
    }
    VISIT(
      auto idx = acc0.get_multidimensional_index(i);
      const ulong row = ulong(idx[1]);
      const ulong col = ulong(origin) + ulong(idx[0]);
      STORE_ELEM
    )
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
// REDUCE
#pragma clang fp reassociate(off)
  // The split's second pass. Template: TOTAL (8 * N), SPLITS. Grid
  // (ceil(TOTAL / 256) * 256, 1, 1), threadgroup (256, 1, 1). Each element's fp32
  // partials added in split order from 0.0f, rounded to bf16 once; never mx.sum,
  // whose order is unspecified. Splash adds them in its last-arriving threadgroup
  // (split_reduce.h:32-39), which needs coherent device memory metal_kernel cannot
  // declare.
  const ulong e = thread_position_in_grid.x;
  if (e >= TOTAL) return;
  float total = 0.0f;
  for (uint s = 0; s < SPLITS; ++s) total += part[ulong(s) * TOTAL + e];
  y[e] = bfloat(total);
