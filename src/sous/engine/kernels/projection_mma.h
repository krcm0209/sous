// The projection kernel's header, prepended by projkernel.py after common.h. The
// staged kernel that reads presummed activations has PRESUM set to 1 ahead of it.
#include <metal_simdgroup_matrix>

#ifndef PRESUM
#define PRESUM 0
#endif

// One 8x8x8 simdgroup_multiply_accumulate on a lane's two elements of each operand,
// in the probed lane map: A[fm][fn..fn+1] (weight column fm, k fn..fn+1),
// B[fm][fn..fn+1] (k fm, activation rows fn and fn+1), C and D[fm][fn..fn+1]. An
// output column reads only its own activation row, and the hardware sums its eight
// products onto C in a fixed sequential order.
template <typename TA, typename TB>
inline void mma8(thread float2 &c, vec<TA, 2> a, vec<TB, 2> b) {
  simdgroup_matrix<TA, 8, 8> A;
  simdgroup_matrix<TB, 8, 8> B;
  simdgroup_float8x8 C, D;
  reinterpret_cast<thread vec<TA, 2> &>(A.thread_elements()) = a;
  reinterpret_cast<thread vec<TB, 2> &>(B.thread_elements()) = b;
  reinterpret_cast<thread float2 &>(C.thread_elements()) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = reinterpret_cast<thread float2 &>(D.thread_elements());
}
