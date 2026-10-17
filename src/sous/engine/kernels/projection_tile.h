// The projection tile's helpers: per quant group, one MPP matmul2d of 8 bf16
// activation rows against one group's 4-bit codes of a slab-ordered affine-Q4
// projection (slab.py), then an fp32 epilogue with the group's row sums. Prepended
// by projtile.py after common.h, the DUAL and SPLITOUT defines (one kernel object
// per pair, sous_proj_tile_d{DUAL}_s{SPLITOUT}) and nax.h.
//
// Derived from incoai/splash, runtime/metal/kernels/common/q4_mpp_tiles.h at
// 0d5a035 (q4_store_input_sums, q4_traversal, q4_visit, and the tile's run_group
// and finish_group), under the Apache License 2.0; see THIRD_PARTY_NOTICES.md.
// sous's changes: run_group, finish_group and q4_visit's lambdas are hand-inlined
// as the macros below; the epilogue is sous's two fmas in place of Splash's
// acc += P * scale + rowsum * bias; the store is a plain bf16 store, two under
// DUAL (gate and up, sharing every activation slice and row sum, stored apart
// because the model applies its own SiLU), or under SPLITOUT the unrounded fp32
// accumulators the split's reduce adds.
using namespace mpp::tensor_ops;

// Row r's sums of the four quant groups of the 256-input block at input_origin,
// into sums[sum_origin + q * Rows + r]: per lane the pair x[l] + x[l + 32] in
// fp32, then simd_sum. q4_mpp_tiles.h:70-91, but for const pointers.
template <ushort Rows, ushort Simdgroups>
inline void q4_store_input_sums(const device bfloat *input, uint input_size,
                                uint input_origin, threadgroup float *sums,
                                uint sum_origin, uint simd_lane, uint simd_group) {
  for (uint row = simd_group; row < Rows; row += Simdgroups) {
    uint origin = row * input_size + input_origin + simd_lane;
    float first = simd_sum(float(input[origin]) + float(input[origin + 32]));
    float second = simd_sum(float(input[origin + 64]) + float(input[origin + 96]));
    float third = simd_sum(float(input[origin + 128]) + float(input[origin + 160]));
    float fourth = simd_sum(float(input[origin + 192]) + float(input[origin + 224]));
    if (simd_lane == 0) {
      sums[sum_origin + row] = first;
      sums[sum_origin + Rows + row] = second;
      sums[sum_origin + 2 * Rows + row] = third;
      sums[sum_origin + 3 * Rows + row] = fourth;
    }
  }
}

// Which of a destination's elements are valid, as a traversal: 0 all, 1 four of
// each eight, 2 the first half then four of each eight, 3 the first half, 4 test
// each element. q4_mpp_tiles.h:14-41, the enum as a ushort. Load-bearing on the M5
// Pro: an 8-row destination carries 16-row fragments whose rows 8..15 are invalid
// elements, and a loop over the whole capacity writes them past the [8, N] output
// and into the next split's partials.
template <class Tensor>
inline ushort q4_traversal(const thread Tensor &values) {
  const ushort capacity = values.get_capacity();
  bool all = true;
  bool halfPrefix = capacity != 0 && (capacity % 2) == 0;
  bool striped = capacity != 0 && (capacity % 8) == 0;
  bool prefixed = capacity != 0 && (capacity % 16) == 0;
  for (ushort i = 0; i < capacity; ++i) {
    const bool valid = values.is_valid_element(i);
    all &= valid;
    halfPrefix &= valid == (i < capacity / 2);
    striped &= valid == ((i & 7) < 4);
    prefixed &= valid == (i < capacity / 2 || ((i & 7) < 4));
  }
  return all ? 0 : striped ? 1 : prefixed ? 2 : halfPrefix ? 3 : 4;
}

// q4_visit (q4_mpp_tiles.h:43-68): runs a statement body using `i` over the valid
// elements of acc0 in the traversal `trav` picked. Every element loop of the tile,
// the zeroing, the epilogue and the store, goes through it.
#define VISIT(...)                                                                   \
  if (trav == 0) {                                                                   \
    for (ushort i = 0; i < CAP; ++i) { __VA_ARGS__ }                                 \
  } else if (trav == 1) {                                                            \
    for (ushort j_ = 0; j_ < CAP / 2; ++j_) {                                        \
      const ushort i = ushort((j_ / 4) * 8 + j_ % 4); __VA_ARGS__ }                  \
  } else if (trav == 2) {                                                            \
    for (ushort j_ = 0; j_ < CAP / 2; ++j_) { const ushort i = j_; __VA_ARGS__ }     \
    for (ushort j_ = 0; j_ < CAP / 4; ++j_) {                                        \
      const ushort i = ushort(CAP / 2 + (j_ / 4) * 8 + j_ % 4); __VA_ARGS__ }        \
  } else if (trav == 3) {                                                            \
    for (ushort j_ = 0; j_ < CAP / 2; ++j_) { const ushort i = j_; __VA_ARGS__ }     \
  } else {                                                                           \
    for (ushort i = 0; i < CAP; ++i) if (acc0.is_valid_element(i)) { __VA_ARGS__ }   \
  }

#if DUAL
#define acc1_zero(I) (acc1[I] = 0.0f)
#else
#define acc1_zero(I) ((void)0)
#endif
#if SPLITOUT && DUAL
#define STORE_ELEM { const ulong o_ = (ulong(split) * 8ul + row) * N + col;          \
                     y0[o_] = acc0[i]; y1[o_] = acc1[i]; }
#elif SPLITOUT
#define STORE_ELEM { y0[(ulong(split) * 8ul + row) * N + col] = acc0[i]; }
#elif DUAL
#define STORE_ELEM { y0[row * N + col] = bfloat(float(acc0[i]));                     \
                     y1[row * N + col] = bfloat(float(acc1[i])); }
#else
#define STORE_ELEM { y0[row * N + col] = bfloat(float(acc0[i])); }
#endif

// run_group (q4_mpp_tiles.h:193-214): quant group QG's activation slice and B
// tensor(s), one matmul2d per weight stream into the fresh partial(s) P0 (P1).
// A group's codes for this tile's TN columns are one contiguous run of 32-byte
// units, CSTRIDE nibbles apart, GSTRIDE bytes after the previous group's.
#if DUAL
#define RUN_GROUP(QG, P0, P1) {                                                      \
    auto a_slice = a.slice<64, 8>(int((QG) * 64), 0);                                \
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt0(               \
        tw0 + ulong(QG) * GSTRIDE, dextents<int, 2>{64, int(TN)},                    \
        array<int, 2>{1, CSTRIDE});                                                  \
    auto bs0 = bt0.slice<64, TN>(0, 0);                                              \
    op.run(a_slice, bs0, P0);                                                        \
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt1(               \
        tw1 + ulong(QG) * GSTRIDE, dextents<int, 2>{64, int(TN)},                    \
        array<int, 2>{1, CSTRIDE});                                                  \
    auto bs1 = bt1.slice<64, TN>(0, 0);                                              \
    op.run(a_slice, bs1, P1); }
#else
#define RUN_GROUP(QG, P0, P1) {                                                      \
    auto a_slice = a.slice<64, 8>(int((QG) * 64), 0);                                \
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt0(               \
        tw0 + ulong(QG) * GSTRIDE, dextents<int, 2>{64, int(TN)},                    \
        array<int, 2>{1, CSTRIDE});                                                  \
    auto bs0 = bt0.slice<64, TN>(0, 0);                                              \
    op.run(a_slice, bs0, P0); }
#endif

// finish_group (q4_mpp_tiles.h:215-244) with sous's epilogue, per valid element:
// acc = fma(P, scale, acc); acc = fma(rowsum, bias, acc), in fp32, groups in
// ascending order. When QG ends a 256-input block that is not the range's last,
// the next block's row sums go into the other half of input_sums, and every
// simdgroup waits for them before the next group's epilogue reads them.
#if DUAL
#define EPI_DUAL(P1)                                                                 \
      { float v1 = acc1[i];                                                          \
        v1 = fma(float(P1[i]), float(sc1p[p]), v1);                                  \
        v1 = fma(rs, float(bi1p[p]), v1);                                            \
        acc1[i] = v1; }
#else
#define EPI_DUAL(P1)
#endif
#define FINISH_GROUP(QG, P0, P1) {                                                   \
    const uint qg_ = (QG);                                                           \
    const uint so_ = ((qg_ >> 2) & 1u) * 32u + (qg_ & 3u) * 8u;                       \
    const ulong pb_ = (ulong(tile) * G + qg_) * 256ul + tile_offset;                 \
    VISIT(                                                                           \
      auto idx = acc0.get_multidimensional_index(i);                                 \
      const uint row = uint(idx[1]);                                                 \
      const ulong p = pb_ + ulong(idx[0]);                                           \
      const float rs = input_sums[so_ + row];                                        \
      float v0 = acc0[i];                                                            \
      v0 = fma(float(P0[i]), float(sc0p[p]), v0);                                    \
      v0 = fma(rs, float(bi0p[p]), v0);                                              \
      acc0[i] = v0;                                                                  \
      EPI_DUAL(P1)                                                                   \
    )                                                                                \
    if ((qg_ & 3u) == 3u && qg_ + 1 < end_group) {                                   \
      q4_store_input_sums<8, S>(in, K, qg_ * 64 + 64, input_sums,                    \
                                (((qg_ + 1) >> 2) & 1u) * 32u, lane, sg);            \
      threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    } }
