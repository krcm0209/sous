// Prepended after common.h for the tensor-op kernels only, the int8 GEMM and the
// attention tile's split kernel: the Metal-4 tensor-op header needs macOS 26.2+, and
// keeping it out of Stage A and the tile's reduce kernel lets those compile (and CI
// test them) on any Metal GPU.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Mirror of mlx::steel::BaseNAXFrag::get_coord() (mlx 0.32.2, steel/gemm/nax.h): the
// lane's (column x, row y) inside a 16x16 fragment. A lane's slots r*4 + j are
// (row y + 8r, col x + j) of its fragment; a 16x32 destination is two such
// fragments along n.
inline short2 nax_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}
