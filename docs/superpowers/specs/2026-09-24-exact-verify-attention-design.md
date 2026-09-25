# sous — Exact grouped attention for the speculative verifier

*2026-09-24. The first change to come out of the Splash-engine assessment; #126 and #127 hold the levers it deferred. Grounded in measurements on the maintainer's M5 Pro (applegpu_g17s, macOS 27.0, mlx 0.32.2, mlx-vlm 0.7.1, the default Qwen3.8-27B-4bit with DFlash2), independently audited: every re-run reproduced within 10%, most within 4%. Also grounded in measurements on an M2 Air (applegpu_g14g, macOS 15.5), a read of mlx 0.32.2's `scaled_dot_product_attention.cpp` and `sdpa_vector.h` and of mlx-vlm 0.7.1's `Qwen3_5BatchInvariantForward`, and a three-part adversarial review of an earlier draft of this spec. The measurements ran in an out-of-tree spike.*

## What the measurements established

**The verifier re-reads the KV once per row.** mlx-vlm's exact verifier computes a T-row verify forward's attention in `models/qwen3_5/speculative_verifier.py`, `_attention`, in one of three ways:
- T = 1: the plain decode call.
- T = 2: one bool-masked SDPA call.
- T > 2 with no left padding, which is every sous turn: a Python loop of T single-query `scaled_dot_product_attention` calls. Row *i* reads keys `[0, prefix + i + 1)`, so the whole KV prefix is streamed once per row.

Every sous verify forward reaches `_attention` with mask `"causal"`: `KVCache.make_mask` returns it for any N > 1. At the ~57K-token context sous's subagent turns run at, the verify forward at T = 3 is 127 ms, against 75 ms for an M=1 decode. The per-row loop alone takes 47.8 ms over the 16 full-attention layers.

**Grouping rows into one stock call is bit-exact when their plans match.** For query lengths ≤ 8, mlx 0.32.2 runs a query through `sdpa_vector` (one pass) or `sdpa_vector_2pass` (two passes, with a block count). The choice depends on:
- the architecture suffix;
- the key count;
- `n_simds = gqa × q_len`;
- the vector-path limits: `q_len ≤ 8`, `q_len ≤ n_keys`, `q_len × gqa ≤ 32`, and head dim in {64, 96, 128, 192, 256}.

Inside one plan, key *i* goes to simdgroup `i mod 32` (one pass) or block `i mod blocks` (two passes), whatever the key count. The causal mask *skips* excluded keys rather than folding them in as −∞. So row *r* of a causal call over rows `j..k−1` runs exactly the arithmetic of a single-query call on its own `prefix + r + 1` keys, provided the group's plan equals the row's singleton plan.

Three smaller facts matter for exactness:
- **Query layout.** A multi-row query slice is not row-contiguous, so it runs with the `query_transposed` function constant set, while a one-row slice runs without it. By the source this changes only the query address, and every measurement below used the serving layout.
- **K/V slicing.** Slicing K/V of a batch-1 cache never triggers a copy.
- **The gqa-8 special kernel.** mlx's `sdpa_vector_2pass_1_gqa` fast path, which no grouped call could match, cannot run at head dim 256.

For the 27B (gqa 6, head dim 256) on an 's' part, groups break in two places:
- where rows cross the plan transitions at 1024 (one pass to 64 blocks), 1025 (to 128), 8193, 32769 and 65537 keys;
- past 5 rows, where the vector limit sends a call to the unfused fallback.

**T = 2 needs no change.** The stock verifier already makes one call at T = 2. That call's plan is the plan of its last row with `n_simds = 2·gqa`, which does not always equal row 0's singleton plan. Grouping it would therefore *change* output at the T = 2 straddles, and it saves nothing: the stock single call at 57K takes 24.6 ms, and grouped takes 25.1 ms. The change leaves T = 2 alone.

**Measured exactness, synthetic.** On the M5 Pro, 27B shapes: 12 prefixes from 100 to 131072, including boundary straddles, × T = 1..8 × 3 seeds. Grouped calls were `array_equal` to the per-row loop in 96 of 96 prefix×T cases. Ungrouped single calls differed exactly where the plan mirror predicted and nowhere else.

**Measured exactness and cost, in the real model.** Grouped stock calls were patched into the real 27B verifier through sous's `VLMEngine`, with real sous source prefilled. At every one of the 24 points below, the logits and all five DFlash2 capture hidden states were bit-equal to the stock verifier (max abs 0), and the grouped path ran on all 16 full-attention layers. The prefixes 1021, 8190 and 32766 put verify rows across plan transitions: the groups split there, as `[(0, 2), (2, 3)]` and similar.

Whole verify forward, ms (median of 3):

| prefix | T=3 | T=4 | T=5 | T=6 | T=8 |
|---|---|---|---|---|---|
| 1021, stock → grouped | 67.5 → 67.2 | 76.0 → 76.1 | 98.2 → 98.0 | 131.3 → 130.3 | 184.2 → 182.8 |
| 8190 | 75.8 → 77.7 | 85.0 → 82.9 | 107.8 → 103.9 | 144.2 → 140.1 | 204.2 → 199.2 |
| 32766 | 94.8 → 88.7 | 114.2 → 114.4 | 163.1 → 130.2 | 201.7 → 190.5 | 271.9 → 253.1 |
| 57000 | **127.4 → 111.6** | 148.1 → 139.2 | 195.8 → 167.1 | 251.6 → 215.6 | 321.7 → 286.3 |

Run-to-run noise on the shared M5 is about ±5%, so single cells such as 32766 at T = 4 and T = 5 are noisy.

**End to end.** A custom multi-row kernel (#127) gave the same forward saving at 57K, T = 3: 13–15 ms. End to end at block 3, round loop only, pooled over 4 code-heavy prompts, it gave:
- greedy 16.9 → 19.0 tok/s (+12%) at ~57K;
- sampled 16.3 → 18.0 tok/s (+11%) at ~57K;
- +0.5–0.7% at 2K.

Greedy output stayed token-identical to plain decode throughout. Grouping should land at about **+10–12% decode at ~57K and about 0% at 2K**. The gate below measures it.

## Decisions

- **Grouped stock SDPA, not a custom kernel.** About 40 lines of plain mlx with no Metal source. That means no macOS/MSL compile risk (#123) and no fork-store epoch churn. It matches the kernel's saving at production block 3; the kernel is deferred to #127.
- **Scope 3 ≤ T ≤ 8.** T = 2 is already one stock call (see above). Above 8, mlx leaves the vector path.
- **Lives in sous.** A sous-side hook, with runtime gates on the mlx version and on the mlx-vlm source it depends on. No upstream PR.
- **No config switch.** The output cannot differ from the stock verifier. Exactness is proven at load, and anything unproven runs the stock loop, so there is nothing for a user to choose.

## Design

### Mechanism

`src/sous/engine/verifyattn.py` is new and has the same shape as `int8prefill.py`: mlx and mlx_vlm imports are function-local, mlx-vlm internals are resolved with importlib, and nothing here ever raises into a model load.

- **`plan(arch, n_keys, gqa, q_len)`** returns `("fallback" | "1pass" | "2pass", blocks)` and mirrors mlx 0.32.2's routing.
  - It returns fallback for `q_len > 8`, `q_len > n_keys` or `q_len × gqa > 32`.
  - The two-pass switch is `(suffix in "ds" and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096)`.
  - Block counts by suffix:
    - 's': 64, or once `n_keys > 1024` and `n_simds > 4`: 128 / 256 / 512 / 1024 at ≤ 8192 / ≤ 32768 / ≤ 65536 / above.
    - 'd': 128; 256 when `n_simds ≤ 2` and `n_keys > 8192`; 512 / 1024 when `n_simds ≥ 6` at `n_keys ≥ 16384` / `≥ 65536`.
    - Other suffixes: 64 when `n_simds ≥ 4`, else 32.
  - `arch` is `mx.device_info()["architecture"]`, the same string mlx's dispatch reads.
  - It supports head dim 256 only. Any other head dim is "unsupported", so a future scope change cannot silently pick up the gqa-8 special kernel.
- **`groups(arch, prefix, T, gqa)`** returns `[(j, k)]`: maximal contiguous runs of rows where one causal call's plan equals every member row's singleton plan. A single row is always its own exact group.
- **`grouped_attention(queries, keys, values, scale, prefix, arch)`** issues one `mx.fast.scaled_dot_product_attention` per group.
  - Queries are the group's rows, and K/V are sliced to `prefix + k`.
  - The mask is `"causal"` for groups of two or more rows, and `None` for one row, which is the M=1 decode call.
  - Results are concatenated along the row axis.

### The hook

A class-level wrap of `Qwen3_5BatchInvariantForward._attention` is installed once per process. It is inert unless the attention module it receives carries a tag. `enable()` sets the tag with `object.__setattr__` on the loaded target's full-attention modules, which keeps it out of mlx's parameter tree. Other models take the original path: Gemma4 and NemotronH override `_attention`, and Qwen4Exp inherits the wrapper but is never tagged. Drafters never call the target's attention.

**Scope is decided before the projections run.** `_prepare_projected_qkv` appends the verify rows to the KV cache, so once it has run the original method cannot be called without appending them twice. The wrapper therefore hands the call to the original method *before touching anything* unless all of these hold:
- the module is tagged;
- batch 1, and 3 ≤ T ≤ 8;
- the incoming mask is the string `"causal"`;
- `type(cache) is KVCache`, resolved from `mlx_vlm.models.cache` in `enable()`. The exact type, not `isinstance`: QSAKVCache, HyV4KVCache and RingSlidingKVCache subclass it with other layouts;
- the cache has no `bits`;
- no left padding: neither `_qwen3_5_decode_left_padding` on the cache nor the module helper `_qwen3_5_left_padding_info(cache)` reports any;
- the module's `head_dim` is 256, and its query/KV head ratio is the one the probe proved.

In scope, the wrapper runs the original body with only the per-row loop replaced:
1. q/k/v through `self._linears`;
2. `attention._prepare_projected_qkv`;
3. `self._helpers()._qwen3_5_left_padded_attention`, whose output is used if it returns one, as the original does;
4. otherwise `grouped_attention`;
5. then the gated `o_proj` through `self._linear`.

If a post-projection invariant still fails (keys not an array, dtypes unequal), the wrapper runs the stock per-row loop itself. That is the original's semantics for T ≥ 3, and the original method is never re-entered.

`update_and_fetch` still runs exactly once per forward, so commit, rollback and abort of a speculative round are unaffected.

### The exactness guarantee

Four guards stack. If any fails, the state is `unavailable` with a reason, and the stock loop runs.

1. **mlx version gate.** `VALIDATED_MLX = {"0.32.2"}`. `plan()` mirrors compiled C++ that sous cannot inspect at runtime, so any other mlx is "mlx X not validated". `MLX_SDPA_BLOCKS` in the environment is also unavailable: mlx honours that override, and `plan()` does not.
2. **mlx-vlm source gate.** The daemon's mlx-vlm can differ from the lock: on the M5 the tool environment ran 0.7.2 while the lock pinned 0.7.1. So the gate is on source, checked at runtime. `VALIDATED_MLX_VLM_SOURCES` maps each function the hook relies on to the SHA-256s of its `inspect.getsource` that someone has validated:
   - `Qwen3_5BatchInvariantForward._attention`
   - `Qwen3_5Attention._prepare_projected_qkv`
   - `_create_qwen3_5_attention_mask`
   - `_qwen3_5_left_padded_attention`
   - `_qwen3_5_left_padding_info`
   - `KVCache.update_and_fetch`
   - `KVCache.make_mask`
   - `mlx_vlm.models.base.scaled_dot_product_attention`
   - `mlx_vlm.models.base.slice_kv_sequence`
   - `mlx_vlm.models.base.kv_sequence_length`

   The last three are what the M=1 decode call and `_row_loop`'s reference
   both go through to reach `mx.fast.scaled_dot_product_attention`; the
   first seven alone pin the loop's control flow but not the call it makes.

   Any other hash, or source that cannot be read, is "mlx-vlm `<name>` changed". A body change of any kind disables the path until someone re-reads it and adds the new hash.
3. **Load-time probe**, in `enable()`, on the `sous-model-load` thread.
   - **Inputs** mirror serving:
     - q/k/v use the loaded model's q heads, KV heads, head dim and dtype, with a fixed seed.
     - q is built as `_prepare_projected_qkv` builds it: `[1, T, Hq, D]`, then transposed.
     - K/V are slices of a buffer padded to 256-token steps, as `KVCache.update_and_fetch` returns them.
   - **Lengths** are generated, not listed:
     - every N at which `plan(arch, N, gqa, q)` differs from `plan(arch, N − 1, gqa, q)` for any q in 1..min(8, 32 // gqa);
     - for each such N and each T in 3..8, the prefixes `N − T`, `N − T // 2 − 1` and `N − 1`, so rows sit on both sides of the transition;
     - plus one interior prefix between consecutive transitions;
     - capped at the last transition + 7 keys: 65544 on 's', 65543 on 'd' and 4103 on other suffixes.
   - **Check:** `grouped_attention` must be `mx.array_equal` to the per-row loop. The first mismatch disables the path and names the case.
   - **Memory and time.** Measured on an M2. On its own 'g' table the probe takes 0.9 s cold (it is the process's first attention call, so the SDPA kernels compile inside it) and 0.1 s warm, with a 51 MB peak. Under the 's' and 'd' tables it takes 2.5–2.9 s cold and 1.4–2.2 s warm, with about 0.8 GB peak. The M5 Pro's figure is recorded on its load line (`verify_attention_probe_s`).
4. **Per-call scope check**, as above.

**Failure policy (int8prefill's rule).** Nothing here raises or fails a model load. Every failure is one `warnings.warn` plus `state: unavailable`.

### Wiring and visibility

- **`src/sous/engine/vlm.py`.** `VLMEngine.__init__` calls `verifyattn.enable(model)` only when a drafter loaded, because without speculation the verifier never runs; otherwise the state is `off`. The call comes after `_load_quantized_drafter` and before the cache budget is measured, on the loading thread. The result is kept as `verify_attention_status`.
- **`src/sous/engine/base.py`.**
  - `ManagedEngine` gains a `verify_attention_status` property, mirroring `int8_prefill_status`.
  - The model-load line in `EngineManager.get()` appends `verify_attention=active|unavailable|off`, as it appends `positions=` today.
  - `EngineManager.status()` gains `verify_attention: {state, reason}` beside `int8_prefill`.
  - Nothing is logged per turn.
- **Unchanged:** the LM backend, the fork key, `forkstore._EPOCH_FILES` / `engine_epoch`, `engine/kernels/`, the config, `sous tune` and the TUI. Prefill never goes through the verifier, and verify output is bit-identical, so nothing on disk can depend on this. `vlm.py` itself is in `_EPOCH_FILES`, so this change invalidates the on-disk forks once after it lands.

## Testing

Every test below runs in CI's single `test` job (macos-15, `pytest -m "not model"`), except the model test. The lint job never installs the project.

- **Plan and grouping (no GPU needed):**
  - `plan()` table tests for the 's', 'd' and 'g' suffixes against mlx 0.32.2's `scaled_dot_product_attention.cpp`, on both sides of every threshold, including N = 1024 vs 1025;
  - `groups()` at each transition and past the vector limit;
  - the probe's length generator;
  - the scope predicate case by case: batch, T = 2 → original, mask forms, the KVCache subclasses → original, bits, left padding, head dim, gqa.
- **Exactness and probe on the GPU.** These run in subprocesses with `MLX_METAL_GPU_ARCH` set to `applegpu_g13s`, `applegpu_g13d` and the runner's own value. mlx reads that variable once, when it constructs the device, and uses it for SDPA routing and `device_info`, so every suffix's plan table is proven bit-exact on whatever GPU CI has. Each run checks:
  - grouped output is `array_equal` to the per-row loop at the generated prefixes, for gqa 4 and 6 at head dim 256, bf16 and fp16;
  - `probe()` returns active;
  - `probe()` returns unavailable when `plan()` is deliberately wrong, which proves the probe can catch a mismatch;
  - `MLX_SDPA_BLOCKS` set → unavailable.
- **Gates.**
  - The locked mlx is in `VALIDATED_MLX`.
  - The locked mlx-vlm's source hashes are in `VALIDATED_MLX_VLM_SOURCES`.
  - A dependency bump that changes either fails CI until someone re-validates it.
- **mlx-vlm contract test.**
  - A tiny random qwen3_5 language model is built through mlx-vlm, with a few layers including full attention, head dim 256 and gqa 6.
  - It is prefilled to a prefix that straddles a transition for the runner's architecture.
  - `speculative_verify=True` forwards run at T = 3..5 with and without the hook.
  - Logits must be bit-equal, a counter must show the grouped path ran, and T = 2 must take the original path.
- **Model test (`-m model`, manual, M5 Pro).** The real 27B verify forward at prefixes 1021, 8190, 32766 and 57000, T = 3..8, hooked against stock. The logits and every DFlash2 capture hidden state must be bit-equal, and the hook must have engaged on all 16 full-attention layers.
- **Gate before merge (M5 Pro, daemon idle, one GPU job at a time).**
  - **Run:** end-to-end DFlash2 decode at block 3, greedy and sampled, on ~57K and ~2K contexts. Compare main against the branch, round loop only, pooled over at least 4 prompts. Record the probe's load-time cost.
  - **Pass:**
    - greedy output token-identical to plain decode on every prompt;
    - ≥ +5% decode at ~57K;
    - no regression at 2K beyond run-to-run noise (±3–5%).

## Documentation

A CLAUDE.md gotcha covers:
- the plan mirror and why grouping is exact;
- why T = 2 is left alone;
- that scope is decided before the projections;
- both version gates and "re-validate on every mlx or mlx-vlm bump";
- why the module sits outside the fork epoch.

The README's speculative-decoding lines change only if the gate's numbers justify it.

## Out of scope

- The custom multi-row kernel (#127).
- Cheaper verify projections and blocks above 5 (#126). `SPECULATIVE_BLOCK_MAX` is unchanged.
- Non-exact attention: a GQA-packed matmul is flat in T but loses bit parity, and gains nothing at T = 3.
- Rejection sampling and the positioned sampler (#88).
- Other model families, other head dims, and the LM backend.
