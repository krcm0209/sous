// Shared by both kernels; prepended by int8prefill.py as the metal_kernel header.
#include <metal_stdlib>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
