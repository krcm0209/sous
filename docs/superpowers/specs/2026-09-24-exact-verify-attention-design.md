# sous — Exact grouped attention for the speculative verifier

*2026-09-24. The first change to come out of the Splash-engine assessment (see #126 and #127 for the levers it deferred). Grounded in:*
- *measurements on the maintainer's M5 Pro (applegpu_g17s, macOS 27.0, mlx 0.32.2, mlx-vlm 0.7.1, the default Qwen3.8-27B-4bit + DFlash2), independently audited: every re-run reproduced within 10%, most within 4%;*
- *measurements on an M2 Air (applegpu_g14g, macOS 15.5);*
- *a read of mlx 0.32.2's `scaled_dot_product_attention.cpp` and mlx-vlm 0.7.1's `Qwen3_5BatchInvariantForward`.*

*The measurements ran in an out-of-tree spike, not in this repo.*

## What the measurements established

**The verifier re-reads the KV once per row.** mlx-vlm's exact verifier (`models/qwen3_5/speculative_verifier.py`, `_attention`) computes the attention of a T-row verify forward in one of three ways:
- T = 2: one bool-masked SDPA call.
- T > 2 with no left padding (every sous turn): a Python loop of T single-query `scaled_dot_product_attention` calls. Row *i* reads keys `[0, prefix + i + 1)`, so the whole KV prefix is streamed once per row.
- T = 1: the plain decode call.

At the ~57K-token context sous's subagent turns run at, verify attention is a large share of the round:
- The whole verify forward at T = 3 is 113 ms, against 75 ms for an M=1 decode.
- The per-row loop alone is 47.8 ms per 16 full-attention layers.

**Grouping rows into one stock call is bit-exact when their plans match.** For query lengths ≤ 8, mlx 0.32.2 runs a query through `sdpa_vector` (one pass) or `sdpa_vector_2pass` (two passes with a block count), chosen from four things:
- the architecture suffix;
- the key count;
- `n_simds = gqa × q_len`;
- the vector-path limits: `q_len ≤ 8`, `q_len × gqa ≤ 32`, and head dim in {64, 96, 128, 192, 256}.

Inside one plan:
- key *i* goes to simdgroup `i mod 32` (one pass) or block `i mod blocks` (two passes), independent of the key count;
- the causal mask *skips* excluded keys rather than folding them in as −∞.

So row *r* of a causal call over rows `j..k−1` runs exactly the arithmetic of a single-query call on its own `prefix + r + 1` keys, provided the group's plan equals the row's singleton plan.

For the 27B (gqa 6, head dim 256) on an 's' part, groups break only in two places:
- where rows straddle the 1024, 8192, 32768 or 65536 key boundaries;
- past 5 rows, where the vector limit sends a call to the unfused fallback.

**Measured exactness.** On the M5 Pro the case grid was 12 prefixes from 100 to 131072 (every boundary straddle included) × T = 1..8 × 3 seeds, 96 prefix×T cases:
- Grouped calls were `array_equal` to the per-row loop in 96/96 cases.
- Ungrouped single calls differed exactly where the plan mirror predicted and nowhere else.
- Hooked into the real 27B verifier at 8K and 32K for T = 2..8, the logits and all five DFlash2 capture hidden states were bit-equal to the stock verifier.

**Measured cost.** Attention for 16 layers of 27B shapes on the M5 Pro, ms:

| keys | T=2 | T=3 | T=4 | T=5 |
|---|---|---|---|---|
| 57K, per-row loop | 32.0 | 47.8 | 62.2 | 76.4 |
| 57K, grouped stock calls | 24.6 | 37.5 | 45.6 | 55.5 |
| 57K, custom multi-row kernel (#127) | 21.7 | 36.5 | 42.6 | 57.2 |
| 2K, per-row loop | 2.00 | 2.66 | 3.35 | 4.06 |
| 2K, grouped stock calls | 1.60 | 2.09 | 2.45 | 2.90 |

**Measured end to end** with the custom kernel, which grouping approaches. Real 27B + DFlash2, block 3, round loop only, pooled over 4 code-heavy prompts:
- at ~57K: greedy 16.9 → 19.0 tok/s (+12%), sampled 16.3 → 18.0 (+11%);
- at 2K: +0.5–0.7%.

Greedy output stayed token-identical to plain decode in every case. Grouping gets about 90% of the kernel's attention saving at T = 3, so it should land near **+8–11% decode at ~57K and about 0% at 2K**. The gate below measures it.

## Decisions

- **Grouped stock SDPA, not a custom kernel.** Grouping is about 40 lines of plain mlx with no Metal source. That means no macOS/MSL compile risk (#123) and no fork-store epoch churn. At production block 3 it gets about 90% of the custom kernel's saving; the kernel is deferred to #127.
- **Lives in sous.** It is a sous-side hook with a contract test that pins the mlx-vlm internals it reads. No upstream PR.
- **No config switch.** Output cannot differ from the stock verifier: exactness is proven at load, and anything unproven falls back to the stock loop. So there is nothing for a user to choose.

## Design

### Mechanism

`src/sous/engine/verifyattn.py`, new. The same shape as `int8prefill.py`:
- mlx and mlx_vlm imports are function-local;
- mlx-vlm internals are resolved with importlib;
- nothing here ever raises into a model load.

- `plan(arch, n_keys, gqa, q_len) -> ("fallback" | "1pass" | "2pass", blocks)` mirrors mlx 0.32.2's routing for head dim 256:
  - the vector-path limits (`q_len ≤ 8`, `q_len × gqa ≤ 32`);
  - the one-pass/two-pass switch: two-pass when the suffix is 'd' or 's' and keys ≥ 1024, or when gqa > 1 and keys ≥ 4096;
  - the block count per suffix: 's' 64/128/256/512/1024 by key count once `n_simds > 4`; 'd' 128/256/512/1024; otherwise 32/64 by `n_simds`.
  - It is pure Python and takes the architecture string as an argument.
- `groups(arch, prefix, T, gqa) -> [(j, k)]`: maximal contiguous runs of rows where one causal call's plan equals every member row's singleton plan. A single row is always its own exact group.
- `grouped_attention(queries, keys, values, scale, prefix, arch)` issues one `mx.fast.scaled_dot_product_attention` per group:
  - queries are the group's rows; K/V are sliced to `prefix + k`;
  - the mask is `"causal"` for groups of two or more rows and `None` for a single row, which is exactly the M=1 decode call;
  - results are concatenated along the row axis.

### The hook

A class-level wrap of `Qwen3_5BatchInvariantForward._attention` is installed once per process. It is inert unless the attention module it receives carries a tag. `enable()` sets the tag with `object.__setattr__` on the loaded target's full-attention modules, which keeps it out of mlx's parameter tree. Any other model in the process takes the original path.

**Scope is decided before the projections run.** `_prepare_projected_qkv` appends the verify rows to the KV cache. Once it has run, the original method cannot be called without appending them twice. So the wrapper hands the call to the original method *before* touching anything unless all of these hold:
- the module is tagged;
- batch 1 and 2 ≤ T ≤ 8;
- the incoming mask is `None` or `"causal"`;
- the cache is a plain unquantized `KVCache`: no `bits`, and no left-padding metadata (`_qwen3_5_decode_left_padding`, or left-padding info reported by the module's helper);
- the module's `head_dim` is 256 and its query/KV head ratio matches what was probed.

In scope, the wrapper runs the original body with only the loop replaced:
1. q/k/v through `self._linears`;
2. `attention._prepare_projected_qkv`;
3. `_qwen3_5_left_padded_attention`, whose output is used if it returns one (as the original does);
4. otherwise `grouped_attention`;
5. then the gated `o_proj` through `self._linear`.

If a post-projection invariant still fails (keys not an array, dtypes unequal), the wrapper runs the stock per-row loop itself. That keeps the original semantics and never re-enters the original method.

### The exactness guarantee

Three guards stack. If any one fails, the state is `unavailable` with a reason, and the original loop runs.

1. **Version gate.** `VALIDATED_MLX = {"0.32.2"}`. On any other mlx the path is unavailable ("mlx X not validated"). A CI test asserts that the locked mlx is in the set, so a dependency bump fails CI until someone re-reads the dispatch, updates `plan()` if needed and extends the set. `MLX_SDPA_BLOCKS` set in the environment also makes the path unavailable: mlx honours that override, and `plan()` does not.
2. **Load-time probe**, in `enable()`, on the `sous-model-load` thread:
   - **Inputs:** random q/k/v with a fixed seed, in the loaded model's own q heads, KV heads, head dim and dtype.
   - **Lengths:** every plan boundary for this device's architecture up to the configured context window, straddled on both sides, plus interior points. T runs from 2 to 8.
   - **Check:** `grouped_attention` must be `mx.array_equal` to the per-row loop. The first mismatch disables the path and names the case.
   - **Cost:** tens of ms and about 270 MB of transient K/V at the largest length, released before the cache budget is measured.
3. **Per-call scope check**, as described above.

**Failure policy (int8prefill's rule).** Nothing here raises or fails a model load. Every failure is one `warnings.warn` plus `state: unavailable`.

### Wiring and visibility

- **`VLMEngine.__init__` calls `verifyattn.enable(model, ...)` only when a drafter loaded.** Without speculation the verifier never runs, and the state is `off`. The call comes after `_load_quantized_drafter` and before the cache budget is measured, on the loading thread. The result is kept as `verify_attention_status`. The LM backend is untouched.
- **Model-load line** gains `verify_attention=active|unavailable|off`.
- **Status document** (`EngineManager.status`) gains `verify_attention: {state, reason}` beside `int8_prefill`. Nothing is logged per turn.
- **Unchanged:** the fork key, `forkstore._EPOCH_FILES` / `engine_epoch`, `engine/kernels/`, the config and the TUI. Prefill never goes through the verifier, and verify output is bit-identical, so what is on disk cannot depend on this.

## Testing

**Pure Python** (no GPU; runs in every job that can import the module):
- `plan()` table tests for the 'g', 's' and 'd' suffixes. The expected values are derived from `scaled_dot_product_attention.cpp` at v0.32.2 and cover every threshold on both sides.
- `groups()` at each boundary straddle and past the vector limit.
- The scope predicate, case by case: batch, T, mask, left padding, quantized cache, head dim, gqa.

**GPU, in CI** (macOS runner, whatever architecture it reports):
- Grouped output is `array_equal` to the per-row loop at boundary-straddling prefixes, for gqa 4 and 6 at head dim 256, bf16 and fp16.
- `probe()` returns active.
- `probe()` returns unavailable when `plan()` is deliberately wrong, which proves the probe can catch a mismatch.
- `MLX_SDPA_BLOCKS` set → unavailable.
- The version gate: the locked mlx is in `VALIDATED_MLX`.

**mlx-vlm contract test (CI).** It builds a tiny random qwen3_5 language model through mlx-vlm: a few layers including full attention, head dim 256, gqa 6.
- It prefills past a plan boundary for the runner's architecture, then runs `speculative_verify=True` forwards at T = 2..5 with and without the hook.
- Logits must be bit-equal, and a counter must show that the grouped path ran.
- It also asserts that the private names the hook reads exist: `Qwen3_5BatchInvariantForward._attention`, `_linears`, `_linear`, `_helpers`, `_qwen3_5_left_padded_attention`, `_prepare_projected_qkv` and `kv_sequence_length`. An mlx-vlm bump that moves them fails CI.

**Model test (`-m model`, manual, M5 Pro).** The real 27B verify forward at 8K and 32K, T = 2..5, hooked vs stock:
- the logits and every DFlash2 capture hidden state must be bit-equal;
- the hook must have engaged on all 16 full-attention layers.

**Gate before merge (M5 Pro, daemon idle, one GPU job at a time).** End-to-end DFlash2 decode at block 3, greedy and sampled, on ~57K and ~2K contexts, main vs branch, round loop only, pooled over at least 4 prompts. Pass requires all three:
- greedy output token-identical to plain decode on every prompt;
- ≥ +5% decode at ~57K;
- no regression at 2K beyond run-to-run noise (±3–5%).

## Documentation

A CLAUDE.md gotcha records four things:
- the plan mirror and why grouping is exact;
- the version gate and "re-prove `plan()` on every mlx bump";
- that scope is decided before the projections;
- that the module sits outside the fork epoch on purpose.

The README's speculative-decoding lines change only if the gate's numbers justify it.

## Out of scope

- **The custom multi-row kernel:** #127.
- **Cheaper verify projections and blocks above 5:** #126. `SPECULATIVE_BLOCK_MAX` is unchanged.
- **Non-exact attention.** A GQA-packed matmul is flat in T but loses bit parity, and at T = 3 it gains nothing.
- **Rejection sampling and the positioned sampler:** #88.
- **Other model families and head dims, and the LM backend.**
