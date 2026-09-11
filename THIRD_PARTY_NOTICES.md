# Third-party notices

sous is MIT-licensed (see `LICENSE`). The following components are derived from
work under other licenses and keep their original terms.

## oMLX INT8-activation prefill kernels

`src/sous/engine/int8prefill.py` contains Metal kernel sources derived from oMLX
(https://github.com/jundot/omlx), files `omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8.metal`,
`qwen35_oq_a8_nax.metal` and `oq_a8_decode.h` as merged in jundot/omlx#3548
(author: PowerSpy), ported from an AOT-compiled extension to runtime-compiled
`mx.fast.metal_kernel` sources with the Q5 and per-group-scaling variants removed.

Copyright (c) oMLX contributors. Licensed under the Apache License, Version 2.0;
you may not use those portions except in compliance with the License. You may
obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
