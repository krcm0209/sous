// Shared by every kernel; prepended by int8prefill.py, tileattn.py and
// projkernel.py as the metal_kernel header.
#include <metal_stdlib>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
