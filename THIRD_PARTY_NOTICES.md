# Third-party notices

sous is MIT-licensed (see `LICENSE`). The following components are derived from
work under other licenses and keep their original terms.

## oMLX INT8-activation prefill kernels

`src/sous/engine/kernels/` holds Metal kernel sources derived from oMLX
(https://github.com/jundot/omlx), files `omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8.metal`,
`qwen35_oq_a8_nax.metal` and `oq_a8_decode.h` as merged in jundot/omlx#3548
(author: PowerSpy), ported from an AOT-compiled extension to runtime-compiled
`mx.fast.metal_kernel` sources with the Q5 and per-group-scaling variants removed.
`q4_int8_gemm.metal`'s `SLAB` read addresses Splash's slab weight layout (last
section); its arithmetic is the oMLX-derived GEMM's.
`attention_tile.metal` in the same directory is sous's own work and is not derived
from oMLX or Splash; `projection_row.metal` is sous's own apart from the two
expressions it shares with `projection_mma.metal` (next section) and its slab
body's weight layout (last section); and none of `projection_mma.h`,
`projection_mma.metal`, `projection_tile.h` and `projection_tile.metal` is
derived from oMLX.

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
`acc = fma(dot, scale, acc); acc = fma(sum, bias, acc)`. Those two expressions,
and the weight layout its slab body reads (last section), are the only parts of
`projection_row.metal`, the one-row kernel, taken from Splash; its lane layout,
loads and summation are sous's own. The rest of the MMA kernel is sous's own:
reading [T, K] activations in place, where Splash reads a prepared Xᵀ table, and
weights either in mlx's own [N, K/8] layout or, in the `SLAB` bodies, in
Splash's 256-column slab layout (last section), with the same arithmetic; the
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

## Splash MPP projection tile and slab weight layout

`src/sous/engine/kernels/projection_tile.h` and `projection_tile.metal` are
derived from Splash (https://github.com/incoai/splash, as of 0d5a035; files
`runtime/metal/kernels/common/q4_mpp_tiles.h` — `q4_store_input_sums`,
`q4_traversal`, `q4_visit` and the tile's `run_group` and `finish_group` — and
the persistent threadgroup loop of `runtime/metal/kernels/decode/linear_q4.metal`).
Derived from them: the affine 4-bit tile that runs, per quant group, one MPP
`matmul2d` of eight bf16 activation rows against the group's 4-bit codes on the
tensor units; its per-256-input-block activation row sums in a double-buffered
threadgroup array behind its barriers; its traversal of a cooperative tensor's
valid elements; its persistent threadgroups; and its split of K into ranges of
whole blocks. sous's changes: `run_group`, `finish_group` and `q4_visit`'s
lambdas inlined by hand as macros; the epilogue
`acc = fma(P, scale, acc); acc = fma(rowsum, bias, acc)` in place of Splash's
`acc += P * scale + rowsum * bias`; a plain bf16 store, a dual store that runs
gate and up from one set of activation slices and row sums, and a split variant
that stores unrounded fp32 partials for a fixed-order reduce kernel of sous's
own, where Splash adds its splits in its last-arriving threadgroup.

The slab weight layout is Splash's: each projection's 4-bit codes held as
32-byte units, one column's one group of 64 codes each, ordered
[256-column tile][group][column], with the bf16 scales and biases in the same
order and the width zero-padded to whole tiles. sous adopts it for the repack
in `src/sous/engine/slab.py` and for the reads of `projection_mma.metal`'s
`SLAB` bodies, `projection_row.metal`'s slab body and `q4_int8_gemm.metal`'s
`SLAB` read; those three kernels' arithmetic is unchanged by it (their own
attributions above).

Attribution: Splash contributors (the upstream files carry no copyright header).
Licensed under the Apache License, Version 2.0 (a copy is in
LICENSES/Apache-2.0.txt); you may not use those portions except in compliance
with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
