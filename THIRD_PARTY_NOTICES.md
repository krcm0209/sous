# Third-party notices

sous is MIT-licensed (see `LICENSE`). The following components are derived from
work under other licenses and keep their original terms.

## oMLX INT8-activation prefill kernels

`src/sous/engine/kernels/` holds Metal kernel sources derived from oMLX
(https://github.com/jundot/omlx), files `omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8.metal`,
`qwen35_oq_a8_nax.metal` and `oq_a8_decode.h` as merged in jundot/omlx#3548
(author: PowerSpy), ported from an AOT-compiled extension to runtime-compiled
`mx.fast.metal_kernel` sources with the Q5 and per-group-scaling variants removed.
`attention_tile.metal` in the same directory is sous's own work and is not derived
from oMLX or Splash; `projection_row.metal` is sous's own apart from the two
expressions it shares with `projection_mma.metal` (next section); and neither
`projection_mma.h` nor `projection_mma.metal` is derived from oMLX.

Attribution: oMLX contributors (the upstream files carry no copyright header).
Licensed under the Apache License, Version 2.0 (a copy is in
LICENSES/Apache-2.0.txt); you may not use those portions except in compliance
with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.

## Splash simdgroup-matrix Q4 projection kernel

`src/sous/engine/kernels/projection_mma.h` and `projection_mma.metal` are
derived in part from Splash (https://github.com/incoai/splash; files
`runtime/metal/kernels/common/sgmatrix.h` and
`runtime/metal/kernels/decode/linear_q4_sgmatrix.metal` as of f43509a) and from
its Apple7/8 port (https://github.com/paperniuk/splash, branch
`apple7-m1-kernels`, author: paperniuk, proposed upstream as incoai/splash#131;
`runtime/metal/kernels/prefill/linear_q4_mma.metal` as of d81795d and
`runtime/metal/kernels/decode/linear_q4_mma.metal` as of d2f902e). Derived from
them: Splash's probed simdgroup-matrix lane map, whose index expressions are
reproduced verbatim; its MMA on operands held in plain registers, here with a
weight type and an activation type of their own; the W·Xᵀ orientation (eight
output columns of 4-bit weights against eight activation rows) with its
weight-word addressing, its shift-and-mask nibble-pair extraction and its
one-group weight prefetch; the port's exact half-precision weights,
`as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h)` as written there, times
fp32 activations; and the factored per-group epilogue
`acc = fma(dot, scale, acc); acc = fma(sum, bias, acc)`. Those two expressions
are the only part of `projection_row.metal`, the one-row kernel, taken from
Splash; its lane layout, loads and summation are sous's own. The rest of the MMA
kernel is sous's own: reading mlx's own [N, K/8] weights and [T, K] activations in place, where
Splash repacks weights into 256-column tiles and reads a prepared Xᵀ table; the
in-kernel per-group row sums and the group-sums kernel that reproduces them;
the fixed-order split of K across the simdgroups of one threadgroup, where
Splash splits K across threadgroups through coherent partials and an atomic
counter; the threadgroup-memory weight staging; the zeroed rows past T; the
shape-based choice between the variants; and the code generation that serves
one to four linears per dispatch.

Attribution: Splash contributors (the upstream files carry no copyright header).
Licensed under the Apache License, Version 2.0 (a copy is in
LICENSES/Apache-2.0.txt); you may not use those portions except in compliance
with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
