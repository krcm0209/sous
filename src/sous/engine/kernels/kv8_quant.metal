// kv8's row quantiser (kv8.py): per-token symmetric int8 K and V for the int8 mirror the
// attention tile reads. MPP-free, prepended with common.h only, so it compiles on any
// Metal GPU and CI tests it.
//
// Grid (256, rows, 4), threadgroup (256, 1, 1): thread d of (KV head h, row r). Inputs
// k and v are bf16 [1, 4, rows, 256] views with innermost stride 1, read in place
// through their head and token strides; params = {rows, out_rows}. For K and for V
// apiece, per (row, head): amax over the 256 dims (simd_max, then the eight
// simdgroups' maxima: a max is exact in any order), the scale s = half(amax / 127)
// saturated at half's largest finite value, f = s when s > 0 and 1 otherwise (an
// all-zero or underflowing row gets codes 0 rather than 0 / 0), and
// code = clamp(rint(x / f), -127, 127), rint rounding half to even. Outputs: codes int8
// [1, 4, out_rows, 512], each row's 256 K codes then its 256 V codes, and sc fp16
// [1, 4, out_rows, 2] = {sK, sV}. Rows from `rows` up are left unwritten: a first build
// sizes both outputs to the mirror's capacity, and nothing reads past the live rows.
// A row's codes and scales depend on that row alone, so quantising a prefix in one
// call and in pieces gives the same bits.
  const int d = int(thread_position_in_threadgroup.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int h = int(threadgroup_position_in_grid.z);
  const int out_rows = params[1];
  const float kx = float(k[h * k_strides[1] + r * k_strides[2] + d]);
  const float vx = float(v[h * v_strides[1] + r * v_strides[2] + d]);

  threadgroup float red[2][8];
  const int sg = int(simdgroup_index_in_threadgroup);
  float mk = simd_max(fabs(kx));
  float mv = simd_max(fabs(vx));
  if (thread_index_in_simdgroup == 0) {
    red[0][sg] = mk;
    red[1][sg] = mv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mk = red[0][0];
  mv = red[1][0];
  UNROLL for (int i = 1; i < 8; ++i) {
    mk = max(mk, red[0][i]);
    mv = max(mv, red[1][i]);
  }

  const half sk = half(min(mk / 127.0f, 65504.0f));
  const half sv = half(min(mv / 127.0f, 65504.0f));
  const float fk = (sk > 0.0h) ? float(sk) : 1.0f;
  const float fv = (sv > 0.0h) ? float(sv) : 1.0f;
  const size_t row = size_t(h) * size_t(out_rows) + size_t(r);
  codes[row * 512 + d] = int8_t(clamp(rint(kx / fk), -127.0f, 127.0f));
  codes[row * 512 + 256 + d] = int8_t(clamp(rint(vx / fv), -127.0f, 127.0f));
  if (d == 0) {
    sc[row * 2] = sk;
    sc[row * 2 + 1] = sv;
  }
