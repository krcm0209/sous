# sous — INT8-Activation Prefill on the M5 Tensor Units

*2026-09-11. Brings the prefill kernel of jundot/omlx#3548 (merged 2026-09-11)
to sous as an opt-in, hosted as a runtime-compiled Metal kernel rather than a
C++ extension. Grounded in a same-day spike on the M5 Pro this daemon runs on:
a throwaway port of the kernel, sous's own prefill path before and after, and a
numerics yardstick against the 8-bit checkpoint. Attention routing is tracked
separately in #76; the tool-loop A/B harness that gates any default-on gets its
own spec.*

## Why prefill, and what the PR does

Prompt-cache reuse (Phase 3a, 3c) removed most *repeated* prefill from the
gateway; what remains is the raw prefill rate on everything that is not a
strict cache hit — a new session's first turn (~57K tokens), a worker task's
context, the boundary probes, every miss. Measured through sous's own path
(`mlx_vlm.generate(max_tokens=0)`) on `mlx-community/Qwen3.8-27B-4bit`, stock
mlx 0.32.2 prefills 4,096 tokens at 484 tok/s and 32,768 at 386 tok/s (85 s).

Stock `mx.quantized_matmul` already runs on the M5 neural accelerators (NAX)
for these shapes, but with bf16 activations against dequantized weights. The
PR keeps the checkpoint's packed affine Q4 words, decodes them into tensor-op
fragment registers, quantizes the *activations* to INT8 once per row, and runs
int8 × int8 → int32 with the affine correction applied at every 64-code group
boundary. The integer datapath is roughly twice the bf16 tensor rate, and the
weights are still read once, packed. It routes only prefill shapes (rows ≥ 128)
in the SwiGLU MLP and the linear-attention (GDN) projections; decode and
speculative verify never see it. oMLX ships it off by default because INT8
activations change numerics, and the PR states accuracy was not measured.

## What the spike established

- **No native build is needed.** `mx.fast.metal_kernel` compiles the Metal-4
  `MetalPerformancePrimitives` tensor ops (`mpp::tensor_ops::matmul2d`, int8)
  at runtime on macOS 26.6 / mlx 0.32.2; a 16×16×32 probe matched numpy
  bit-for-bit. This Mac has Command Line Tools only (`xcrun metal` is absent),
  so oMLX's cmake/nanobind extension is not even buildable here; the Python
  route needs nothing beyond the mlx wheel.
- **Kernel throughput** (M = 2048, bf16, this 20-core M5 Pro): stock
  `quantized_matmul` 26–28 TOP/s; the port 43–52 TOP/s across the model's MLP
  and GDN shapes (1.65–1.85x; the PR quotes 42 vs 23 on a 16-core part). The
  seven tile variants are within ~3% of each other. Stage A costs 0.4–0.95 ms
  against 2.7–8.4 ms of GEMM.
- **End to end** through sous's prefill path, MLP + GDN + attention routed:

  | prompt | stock | INT8 prefill | gain |
  |---|---|---|---|
  | 4,096 tokens | 483.7 tok/s | 753.9 tok/s | 1.56x |
  | 16,384 tokens | 427.3 tok/s | 691.1 tok/s | 1.62x |
  | 32,768 tokens | 385.6 tok/s (85.0 s) | 580.2 tok/s (56.5 s) | 1.50x |

  Larger than the PR's 1.32x because sous's stock path is slower than oMLX's
  tuned W4A16 kernel, not because the kernel is faster.
- **Numerics** (last 256 of 2,048 code-prompt positions, greedy):

  | comparison | KL mean (nats) | greedy agreement |
  |---|---|---|
  | 8-bit → 4-bit stock (the drift 4-bit weights already add) | 0.052 | 88.7% |
  | 4-bit stock → +INT8 prefill, MLP + GDN | 0.033 | 91.4% |
  | 4-bit stock → +INT8 prefill, also attention q/k/v/o | 0.053 | 89.5% |

  Every position where the stock model was confident (p ≥ 0.9) agreed in every
  configuration; the flips are at genuinely uncertain positions. Routing the
  attention projections costs ~1.6x the drift for a speed contribution that
  was not isolated — hence #76 and the MLP + GDN set here.
- **Memory.** The spike cached a transposed copy of every projection's scales
  and biases (~24 MB per layer, ~1.5 GiB resident). That is avoidable; see
  *GEMM*.

## Goal

Ship the INT8-activation prefill path behind `[model] int8_prefill`
(default `false`) for affine-Q4/gs64 Qwen3.5-family checkpoints on M5-class
GPUs. Acceptance, measured on this M5 Pro through `VLMEngine.prefill`:

- ≥ 1.4x prefill throughput at 4,096 tokens and ≥ 1.3x at 32,768 versus stock
  (the spike, with attention also routed, measured 1.56x and 1.50x);
- KL(stock ‖ INT8) ≤ 0.06 nats on the spike's standard prompt;
- byte-identical behaviour with the key off or the hardware unavailable;
- a model load never fails because of this feature.

## Non-goals

- **Default-on.** Gated on a reproducible tool-loop A/B (separate spec: the
  harness from the mxfp8-vs-4bit decision exists only as scratchpad scripts).
- **Attention q/k/v/o routing** (#76).
- **Per-group (64) activation scaling** — the PR's `ACT_MODE 1`. The lever if
  row scaling proves too lossy on the A/B; the kernel is written so it can be
  added without changing the routing layer.
- **Q5, mixed-bit (oQ), mxfp4/mxfp8, 8-bit, gs128 checkpoints.** Only affine
  Q4 gs64 routes; everything else falls through to stock, silently per
  projection.
- **MoE variants** (`qwen3_5_moe`). Dense MLP classes only.
- **Decode or verify acceleration.** The 128-row floor is a constant.
- **Upstreaming to MLX.** Worth doing later; not this change.

## Design

### Module: `src/sous/engine/int8prefill.py`

Pure Python, one module. mlx imports stay function-local (the lint job runs on
ubuntu). Public surface:

- `availability() -> Availability` — `state` in `{available, unavailable}`
  with a human `reason`.
- `enable(model, *, enabled: bool) -> Status` — tags the model, installs the
  class wrappers, warms the kernels, returns
  `{"state": "off" | "active" | "unavailable", "reason": str | None, "routed": int}`.
  Never raises.
- `stage_a(x)`, `qmm(...)`, `eligible_linear(linear)` for tests and the
  wrappers. Constants `MIN_ROWS = 128`, `GROUP = 64`, `TILE = (1, 2)`.

Two Metal sources live as module strings, compiled through
`mx.fast.metal_kernel`. Only the GEMM's header includes
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, so the Stage-A
kernel compiles on any Metal GPU and CI can exercise it without tensor units.

### Stage A: activation quantization

One 256-thread threadgroup per row (eight simdgroups, each owning whole
64-code groups):

1. row `amax = max|x|`; `sa[row] = amax / 127`; an all-zero row emits zeros
   rather than NaN and lets the affine bias term carry the output;
2. `q = clamp(rint(x · 127/amax), −127, 127)` as int8;
3. `ra[row, g] = Σ q` over each group, from the *rounded* codes (int16;
   |Ra| ≤ 8128), so the bias correction below is exact.

Outputs, as the GEMM reads them: `qa` int8 `[M, K]` in the fragment K-order,
`sa` f32 `[M]`, `ra` int16 `[K/64, M]` group-major and contiguous. The
K-order is applied in Python by `reshape(M, K/64, 4, 2, 4, 2)`,
`transpose(0, 1, 2, 3, 5, 4)`, `reshape(M, K)`, `mx.contiguous`: slot
`16c + 4t + j` of a group holds code `k = 16c + 8·(t>>1) + 2j + (t&1)`. It is
legal because a dot product is order-independent, the affine group is the
same set of 64 either way, and `Ra` is a sum over it — scales, biases and `Ra`
are untouched by the reorder. Measured cost including the kernel: 0.4–0.95 ms
per call.

Stage A is shared where the PR shares it: gate and up consume one; GDN
`in_proj_qkv` and `in_proj_z` consume one; `down_proj` and `out_proj` each
quantize their own input.

### GEMM

Semantics, for row `m`, column `n`, groups `g` of 64:

```
out[m, n] = sa[m] · Σ_g ( Sw[n, g] · Σ_{k∈g} qa[m, k]·qw[n, k]  +  Bw[n, g] · Ra[g, m] )
```

The inner sum runs on the tensor units as four 16(m)×32(n)×16(k) int8 steps
per group into int32 accumulators (mode `multiply` on the first step of a
group, `multiply_accumulate` after), and the fp32 correction lands once per
group as two fused multiply-adds. Weights are the checkpoint's packed uint32
words, unmodified: each lane reads its 16-code run as one `uint2` per weight
row and decodes nibbles with a mask and a shift into the right-hand fragment.
Lane coordinates follow mlx's `BaseNAXFrag::get_coord()` (steel/gemm/nax.h);
the left fragment element `r·4 + j` is `(m = y + 8r, k = x + j)`, the right
element `h·8 + r·4 + j` is `(n = 16h + y + 8r, k = x + j)`, and destination
element `e` is row `y + ((e&7)>>2)·8`, column `16·(e>>3) + x + (e&3)`. The
spike verified this layout bit-exactly against an integer reference.

Fixed tile `BM = 32, BN = 64` (`WM = 1, WN = 2`, 64 threads per threadgroup),
grid `(N/64, ⌈M/32⌉)`; the PR's best Q4 variant, and the seven variants are
within 3%. Rows past `M` read row `M−1` and are never stored, so a partial
last tile needs no branch in the K loop. `N % 64 == 0` is therefore an
eligibility rule; every projection of the default model satisfies it except
the GDN `in_proj_a`/`in_proj_b` heads (N = 48), which stay stock and are
negligible FLOPs.

`K` and `N` are template constants (one compiled instantiation per distinct
projection shape — five for the default model); `M` is a one-element int32
input so the last chunk of a prompt does not recompile. `T` is the checkpoint's
scale dtype (bf16 or fp16).

**Metadata layout.** The kernel reads `scales`/`biases` in their stored
`[N, K/64]` layout — sixteen scalar loads per group per lane (eight scales,
eight biases) at a stride of `K/64`, cache-friendly because a lane's next 32
groups sit in the same line —
so no per-projection copy exists at all. The plan benchmarks this against the
spike's transposed `[K/64, N]` layout at M = 2048; if the direct read costs
more than 3%, the fallback is a per-call `mx.contiguous(scales.T)` inside the
routed call (≈ 24 MB of transient copies per layer per chunk, measured under
1% at M = 2048), never a resident cache.

### Eligibility and tagging

A projection is eligible when all hold: `nn.QuantizedLinear`; `mode ==
"affine"`; `bits == 4`; `group_size == 64`; no `bias`; `weight` uint32 2-D
with `weight.shape[1] · 32 == K · 4`; `scales`/`biases` fp16 or bf16 of shape
`[N, K/64]`; `N % 64 == 0`; `K % 64 == 0`.

`enable()` walks the *target* model's decoder layers only
(`model.language_model.model.layers` for mlx-vlm, `model.model.layers` for
mlx-lm) and sets `_sous_int8_prefill = True` on eligible projections. Nothing
else is tagged: not `lm_head`, not embeddings, not the DFlash drafter (a
separate module tree that `_load_quantized_drafter` also quantizes to affine
Q4 gs64 — tagging by model, as the PR does, is what keeps it stock).

Class wrappers are installed once per process, idempotently, and act only on
tagged modules:

- MLP `__call__` on `mlx_vlm.models.qwen3_5.language.Qwen3_5MLP` and
  `mlx_lm.models.qwen3_5.MLP` (which is `qwen3_next.Qwen3NextMLP`, shared with
  Qwen3-Next models — harmless because untagged modules fall through): when
  `x.shape[-2] ≥ MIN_ROWS` and gate, up and down are all tagged, one Stage A
  feeds gate and up, `swiglu` combines them, and down gets its own Stage A;
  otherwise the original call.
- GDN `__call__` on `Qwen3_5GatedDeltaNet` (mlx-vlm) and `GatedDeltaNet`
  (mlx-lm): when the row floor is met and `in_proj_qkv`/`in_proj_z` are
  tagged, compute one Stage A and hand it to those two projections for the
  duration of the original call (identity of the input array object, cleared
  in `finally`); the recurrent implementation itself is untouched.
- `nn.QuantizedLinear.__call__`: a tagged module with a handed-down Stage A
  for this exact input uses it; a tagged module at or above the row floor
  without one (`out_proj`) quantizes and routes; everything else is the
  original call.

Rows are `x.shape[-2]`; sous never batches, and a batch would flatten to
`M = B·S` as in the PR. Speculative verify calls the linears with at most
`SPECULATIVE_BLOCK_MAX + 1` rows — six against a floor of 128 — so
the verifier's exactness path is untouched by construction.

### Availability, enablement, degradation

`availability()` mirrors mlx's own `is_nax_available()` (device.cpp, 0.32.2):
Metal available; macOS ≥ 26.2 (`platform.mac_ver()`); `mx.device_info()`
`architecture` matches `applegpu_g<gen><suffix>` with `gen ≥ 17`, or `≥ 18`
when the suffix is `p`. Anything else is `unavailable` with the failing check
as the reason.

`enable(model, enabled=True)`:

1. `availability()`; unavailable → `{"state": "unavailable", ...}`, nothing
   tagged.
2. Tag eligible projections and install the wrappers.
3. Warm up: for every distinct `(K, N, T)` the tagged projections need, run
   Stage A and the GEMM on a 128-row zero input and `mx.eval` the result. This
   compiles every kernel the model will use (first-turn latency stays flat) and
   is the runtime probe: a missing header, a compiler rejection, or a
   `TypeError` from a changed `metal_kernel` signature surfaces here.
4. Any exception in 2–3: untag everything (the wrappers are inert on untagged
   modules), log one WARNING with the exception text, return
   `{"state": "unavailable", "reason": ...}`.
5. Success: one INFO line — `int8 prefill: routed 336 projections` — counts
   only, never a token, and `{"state": "active", "routed": 336}`.

With the key off, `enable()` returns `{"state": "off"}` immediately: nothing
is imported from the Metal side, nothing compiled, no wrapper installed.

The engine stores the returned status on `int8_prefill_status`;
`EngineManager.status()` copies it into an `int8_prefill` key when present, so
`sous status` and `server_status` show the state and the routed count. Fake
engines need no change (the attribute is optional).

### Engine integration

`VLMEngine.__init__` and `LMEngine.__init__` call
`int8prefill.enable(self._model, enabled=config.int8_prefill)` immediately
after `load()`, before the drafter loads and before `measure_cache_budget`, so
warm-up temporaries are released before the budget is measured and the
drafter cannot be tagged. The flag reaches the engines the way
`speculative_draft_id` does today.

`unload()` needs nothing new: there are no resident copies, and compiled
kernels stay in mlx's in-process cache (kilobytes). The thread rule is
unchanged — the kernels run on whatever generation thread calls the model,
and `release_mlx_thread_state()` still applies to it.

`PrefixCache` is unaffected. A conversation's KV cache may hold tokens
prefilled by the INT8 path and tokens prefilled stock (a warm extension under
128 rows), plus decoded tokens; both paths approximate the same function and
the mix is by design.

### Configuration

`[model] int8_prefill = false` (bool). A non-bool value logs the same style of
warning the other `[model]` keys use and falls back to `false`. `SousConfig`
gains `int8_prefill: bool = False`.

## Invariants preserved

- Decode, speculative verify and the drafter run exactly as before; the row
  floor is a constant, not a knob.
- With the key off, or on hardware without tensor units, code paths are
  byte-identical to today; a model load never fails because of this feature.
- No request content is logged: counts, shapes and exception text only.
- The sandbox (`toolexec.py`) and the gateway are untouched.
- mlx imports stay function-local; the lint job on ubuntu still imports the
  module.
- The kernel reads the checkpoint's weights in place; the module's arrays stay
  usable by decode.

## Testing

CI-runnable (`macos-15`: Metal, no tensor units), in `tests/test_int8prefill.py`:

- **Stage A vs numpy**: codes equal except for rounding-tie differences of ±1
  at ≤ 0.1% of positions (fp32 vs fp64 product rounding; the spike saw 0.03%);
  `ra` equals the sum of the kernel's *own* codes exactly; `sa` within 1e-6
  relative; the all-zero row emits zeros.
- **K-order**: the reorder round-trips through its inverse, and the explicit
  slot formula holds for every `(c, t, j)`.
- **Eligibility table**: affine Q4 gs64 accepted; mxfp4, 8-bit, gs128, a
  `bias`, `N = 48`, fp32 scales rejected.
- **Routing**, with tiny quantized modules and `qmm` monkeypatched to a
  reference implementation so the tests run without tensor units: rows below
  128 take the stock path; gate/up receive the same Stage A object; the shared
  stage never leaks past the GDN call; one ineligible member sends the whole
  MLP call stock; untagged modules (a second model in the same process) are
  never routed; wrappers install idempotently.
- **Enablement**: `availability()` on the runner reports `unavailable` with a
  reason; `enable()` then tags nothing and raises nothing; a warm-up failure
  untags and degrades; the key off returns `off` without importing kernels.
- **Config and status**: parsing incl. the non-bool warning; the
  `int8_prefill` key in `EngineManager.status()`; existing fake-engine tests
  unchanged.

Local only (`model`-marked, or `skipif(not availability().available)`):

- **GEMM bit-exactness** against an integer reference with power-of-two
  scales and biases (every fp32 partial exact, so the only rounding is the
  final bf16 cast): random packed words, `K ∈ {64, 128, 1024}`, `N ∈ {64,
  128}`, `M ∈ {32, 37}` including a partial last tile. This is the test that
  fails if the fragment layout or the K-order is wrong.
- **GEMM vs stock `quantized_matmul`** at model shapes: relative RMS ≤ 1.5e-2
  (the INT8 activation noise itself; the spike measured 9.2e-3).
- **Real model** (`mlx-community/Qwen3.8-27B-4bit`, through `VLMEngine`):
  routed count 336; ≥ 1.4x at 4,096 tokens and ≥ 1.3x at 32,768; KL(stock ‖
  INT8) ≤ 0.06 on the
  standard prompt (the concatenation of `src/sous/engine/promptcache.py` and
  `src/sous/gateway/turn.py`, first 2,048 tokens, last 256 positions).
- **Soak**: one real delegated task with the key on, per the thread-teardown
  rule in CLAUDE.md.

## Documentation

- README, `[model]` section: the key, the M5 / macOS 26.2 requirement, the
  measured speedups, and the caveat that INT8 activations change numerics
  (why it ships off).
- CLAUDE.md gotchas: the K-order invariant (never "simplify" the reorder or
  the nibble decode independently), the 128-row floor, the 26.2 header floor
  and that the GEMM header is the only Metal-4 include.
- `THIRD_PARTY_NOTICES.md`: the kernel is a derivative of oMLX's Apache-2.0
  `qwen35_oq_a8.metal`, `qwen35_oq_a8_nax.metal` and `oq_a8_decode.h`
  (jundot/omlx#3548, PowerSpy); the module header carries the same notice.
  sous stays MIT.

## Risks and open questions

- **Runtime Metal compiler drift.** A macOS update could change what the
  runtime compiler accepts or how long it takes; the warm-up probe turns that
  into a degradation, never a failure. Warm-up time is measured in the plan
  (expected under two seconds for five shapes).
- **`metal_kernel` API changes** in a future mlx: caught by the warm-up's
  `TypeError` path; the module degrades rather than the daemon failing.
- **Numerics on real work.** The KL yardstick says the drift is inside the
  4-bit envelope, but only the tool-loop A/B decides the default. Per-row
  scaling is sensitive to activation outliers; per-group scaling is the
  documented next lever.
- **Metadata layout.** The direct `[N, K/64]` read is unmeasured; the spec
  fixes the decision rule (≤ 3% or fall back to per-call transposes), not the
  outcome.
- **Upstream MLX** may ship an INT8-activation quantized matmul; if it does,
  this module becomes removable and the config key a no-op.
