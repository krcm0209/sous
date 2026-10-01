# sous — Tensor-unit attention tile for decode and verify

*2026-09-30. The in-tree design for #132's verify tile. It is grounded in an out-of-tree spike on the maintainer's M5 Pro (applegpu_g17s, 20 GPU cores, macOS 27.0, mlx 0.32.2, mlx-vlm 0.7.2, the default Qwen3.8-27B-4bit with DFlash2 at block 3). Every stage of the spike was independently re-run and checked. The findings are on #132, in the microbench comment of 2026-09-29 and the in-model comment of 2026-09-30. Before review, three adversarial reviews checked an earlier draft of this spec against the code and mlx-vlm 0.7.2's source.*

## What the measurements established

**Verify attention is most of decode's context-dependent cost, and the tensor units cut it by 2–2.6x.** At block 3 the verify forward's attention runs three rows. Today it goes through #128's grouped stock SDPA calls. The tile runs it on the M5's tensor units instead, as MPP `matmul2d(16, 32, 16)` fragments.
- **GQA packing.** The six query heads that share a KV head are packed with the rows, giving M = 6·T rows per KV head in 16-row fragments.
- **Simdgroups.** Two simdgroups serve each fragment group, and each accumulates half of the 256 output dims in registers.
- **Softmax.** The online softmax is fp32 in registers, with no threadgroup barriers in the key loop.
- **Splits.** Keys are split across threadgroups and combined by a second kernel, a fixed-order reduce.
- **Reads.** Every read is clamped to n.

At 57K keys and T = 3 the tile takes 13.95 ms per 16-layer forward, against 32.53 ms for #128's grouping. It reads the KV at 250–264 GB/s, which is 81–86% of the chip's DRAM figure.

**Greedy parity survives if decode uses the same kernel with a partition that steps with context.** A row's output depends on the key partition, meaning the chunk size per split, and on nothing else about the call:
- its 32-key blocks are aligned the same way in every call;
- keys past its limit get probability exactly 0;
- the MMA result of a (row, key) pair does not depend on which fragment row it occupies.

So a verify row is bit-identical to a one-row call at its own key count whenever both use the same chunk.

**The schedule.** From N0 = 1,536 keys up, the chunk is `32 · ceil(n / 640)`. The partition is therefore constant within 640-key steps, and it gives 16–20 splits per KV head, exactly 20 from n = 12,161.
- **Why 20 splits.** 20 splits × 4 KV heads is 80 threadgroups on the M5 Pro's 20 cores. At 21 splits, T = 3 drops from 2.26–2.63x to 1.75–1.90x against grouped, which is consistent with a second wave.
- **Below N0** the tile does not beat stock one-row decode, so decode and verify there use stock paths that are exact against each other.
- **Exactness.** 261,648 of 261,648 verify rows were bit-identical to decode at their own length.
  - The rows covered all 408 step boundaries up to 262K, T = 1..8 with every straddle placement, and normal and ×8 queries.
  - They also covered the stock branch below N0, T = 2 included, at mlx's plan transitions 1,024 and 1,025.
- **A real KVCache.** Separately, 25 of 25 rows matched through a real mlx-vlm `KVCache` whose buffer ends at n.

Microbench, ms per 16-layer dependent chain, with the tile's schedule path against today's paths:

| keys | T=1: stock → tile | T=3: #128 → tile | T=8: #128 → tile |
|---|---|---|---|
| 2K | 1.10 → 0.99 | 2.11 → 1.11 | 4.89 → 1.66 |
| 8K | 2.82 → 2.38 | 5.77 → 2.62 | 14.42 → 3.62 |
| 57K | 15.63 → 13.48 | 32.53 → 13.95 | 82.07 → 20.63 |
| 131K | 35.25 → 30.61 | 74.39 → 30.85 | 192.12 → 46.97 |
| 262K | 70.45 → 59.41 | 160.99 → 62.87 | 417.16 → 98.06 |

**Error.** Against a float64 reference, the tile's error is at or below stock's at 2K, 57K and 196K. fp32 probabilities in the PV matmul give 27% lower relative RMS on normal queries, at no measurable cost.

**In the model.** An out-of-tree wrapper routed the target's 16 full-attention layers through the tile inside a real `sous serve`: the regular forward's one-row decode, and the exact verifier's attention.
- **Parity.** Greedy output with the drafter matched output without it on 23 of 23 real subagent turns (44–229K), with the same prompt-cache histories. An independent check matched 11 of 11 more.
- **Near-ties.** The tile diverged from main at near-ties on 4 of 23 turns, and on 12 of 64 in the greedy gate pair. The last-bit attention differences flip the argmax there.
- **Decode.** On #131's 64 real turns (44–77K), sampled, same-evening ABBA: 19.84 → 24.17 tok/s (1.22x, turn-bootstrap CI 1.20–1.24).
  - The gain was 1.20x at 44–64K and 1.26x at 64–77K. One long conversation measured 1.45x at ≥ 128K.
  - Greedy on the turns with identical output: 1.21x.
- **Unchanged:** acceptance (2.541 against 2.528 tokens per round), prefill rate and peak memory.
- **Tool-loop suite** (8 tasks × 6 runs per arm): tile 0.956, stock 0.949. The same two failures appeared in both arms.
- **Where the evidence stops.** Only T = 3 ran in the model, with T = 2 on 16 last-round calls; T = 1 and T = 4..8 never ran in a turn. No real turn was below 44K. Drafter-off decode was 4–6% faster, but in a pair not controlled for drift.
- **One anomaly.** One turn looped to max_tokens under greedy with the tile. Main and the tile both ran away on the same turn in earlier greedy runs from another cache state, so the turn is loop-prone under greedy whichever path runs.

**Codegen is fragile.** Rewriting two index expressions equivalently (`sg >> 1` for `sg / 2`) kept the output bit-identical, but made the kernel 1.4x slower at T = 3 and 8.

**The prefill tile failed its gate** (≤ 1.04x stock at 2,048 rows × 128K), and it is not part of this design.

## Decisions

- **On by default, with an opt-out.** `[model].attention_tile = true`.
  - The tile's output is as accurate as stock's and keeps greedy parity with the drafter on or off.
  - It is not bit-identical to today's stock kernels, so unlike #128 it gets a switch. `false` runs today's paths.
- **Existing hook points, not new wrappers.**
  - `tileattn.py` holds the kernel, the schedule, the gate and the probe.
  - #128's verifier hook sends rows to it.
  - Decode swaps the one function that `Qwen3_5Attention.__call__` calls after its projections.
  - Beyond what verifyattn already copies, only the verifier's T = 1 and T = 2 attention calls are reproduced, and `_attention`'s source hash covers them.
- **One kernel, with a stepped partition.** The fragment tile serves decode (T = 1) and verify at any T.
  - Verify runs are cut at 8 rows.
  - The Splash-style tile, which read past n and was faster only at T = 8, is dropped.
- **The split target S is the GPU's core count, and only measured values are on.**
  - S comes from IOKit. The tile is `active` only where S is in `MEASURED_SPLITS` = {20}, the M5 Pro that the merge gate measures.
  - Any other core count is `unavailable` with that reason, because the guards prove exactness, not speed.
  - A chip joins by running the merge gate's kernel-timing check on it and adding its S.
- **Drafters.** The tile runs with no drafter or with a DFlash-family drafter.
  - Other drafter kinds are refused. mlx-vlm's qwen3_5 MTP drafter builds its layers from the target's attention class and plain `KVCache`s, so the decode hook would serve its proposals, and that path is unmeasured.
- **Lives in sous.** It has runtime gates on mlx, mlx-vlm and the platform, and there is no upstream PR.

## Design

### Files

- **`src/sous/engine/kernels/attention_tile.metal`** (new) holds the split kernel and the reduce kernel from the spike.
  - They are specialised to bf16, 24 query / 4 KV heads, head dim 256, scale 1/16 and fp32 probabilities.
  - They include the MetalPerformancePrimitives header, which puts them in the macOS 26.2+ / MSL 4.1 compile-risk class (#123).
  - Local cooperative-tensor types are wrapped in `metal::remove_addrspace_t`.
  - The fragment lane layout comes from the existing `kernels/nax.h`, sous's mirror of mlx's `BaseNAXFrag::get_coord`, which the int8 GEMM already includes. So there is no new third-party code.
  - Both kernels are built with `compile_options={"math_mode": "safe"}` stated explicitly, not inherited as a default.
- **`src/sous/engine/tileattn.py`** (new) has the same shape as `int8prefill.py` and `verifyattn.py`:
  - mlx imports are function-local;
  - mlx-vlm internals are resolved with importlib;
  - nothing raises into a model load.
- **`src/sous/engine/verifyattn.py`** keeps its single verifier wrapper and gains a tile branch.
- **The NAX platform rule** (macOS ≥ 26.2, GPU generation ≥ 17, or ≥ 18 for 'p' parts) moves out of `int8prefill.availability()` into a shared helper that both modules call.

### Schedule

- **`chunk_for(n)`** returns 0, meaning the stock path, below N0 = 1,536. From N0 up it returns `32 · ceil(n / (32 · S))`, so each step is 32·S keys wide and holds at most S splits.
- **`verify_plan(prefix, T)`** cuts a verify call's rows into contiguous runs.
  - A run never crosses a step boundary and never exceeds 8 rows.
  - Rows below N0 form their own runs, which go to the stock path.
  - A row's output depends only on its chunk, so any T, including T > 8 from a drafter with a larger block, stays exact.
- **`decode(q, k, v)`** runs one tile call at `chunk_for(n)`.
- **`verify(q, k, v)`** takes the prefix as `keys.shape[-2] - T` and runs one tile call per run.
  - Runs below N0 go through `verifyattn.grouped_attention`, T = 1 and 2 included. Its groups share each member's singleton plan: a one-row group is the M=1 decode call, and a multi-row group is exact by #128's plan-sharing argument.
  - At T = 2 that is usually one two-row causal call, not the stock verifier's single masked call.
  - Output is concatenated in row order.
- **The architecture string.** `tileattn` reads `mx.device_info()["architecture"]` itself for `grouped_attention` and never reads `verifyattn._ARCH`.
- **The post-projection check.** `in_scope(q, k, v, scale)` is one predicate that decode and verify both use after the projections:
  - batch 1;
  - 24/4/256 heads;
  - q, k and v all bf16;
  - `scale == 1/16`;
  - keys and values the same length, ≥ T.

**Layout invariant.** The kernel assumes q, k and v have innermost stride 1 and 16-byte-aligned rows, since it makes `uint4`/`bfloat4` loads and reads only the head and token strides. mlx exposes no strides to Python, so the invariant rests on the producers:
- `KVCache.update_and_fetch` returns `[..., :offset, :]` of its `mx.zeros`/concatenate buffer, and it is hashed;
- `forkio` restores assign `mx.load` arrays;
- the queries are the transposed `q_norm` and rope output.

The probe drives a real attention module of the loaded model (see Guards), so the real producers' layouts are exercised on the device at load.

### Decode hook

The first `tileattn.enable()` that reaches `active` replaces the module global `scaled_dot_product_attention` in `mlx_vlm.models.qwen3_5.language`, which is what `Qwen3_5Attention.__call__` calls after `_prepare_projected_qkv`. A process that never activates the tile runs mlx-vlm's function untouched. The wrapper:
- uses `functools.wraps` and forwards `*args, **kwargs` unchanged;
- serves a call only when all of these hold:
  - the module flag is set;
  - `mask is None` and `sinks is None`;
  - the query is one row;
  - `in_scope` holds, and n ≥ N0;
  - `type(cache) is KVCache`, with no `bits` and no left padding.

Every other call goes to the original function with the same arguments.

**Deciding after the projections is safe.** The fallback is the very call the method would have made, so the cache is never appended twice.

**The global cannot tell which module calls it.** Scope therefore rests on the checks above plus the flag:
- The flag is set only for a validated qwen3_5 target, and it is cleared by every `enable()` and every unload.
- The families that reuse the qwen3_5 language model (qwen3_5_moe, qwen4_exp, indic_ocr, minicpmv4_6, prism_hadamard_qwen35) are refused at `enable()` by model type.
- The left-padded helpers pass `cache=None` or a batch cache, so they fail the cache check.
- The DFlash and DFlash2 drafters call `mx.fast.scaled_dot_product_attention` directly, so they never reach the global.
- MTP drafters are refused, as described under Decisions.

**Which calls reach the hook.** Every one-row forward of the target reaches it: plain decode through `generate_step`, and the final one-row forward that ends every prefill segment. The DFlash round loop makes no target call except the verify.

### Verify hook

verifyattn's `_in_scope` still decides before the projections. While the tile is active for the loaded model:
- **Rows:** the range widens from 3..8 to T ≥ 1, with runs cut at 8.
- **Mask:** None at T = 1, `"causal"` above.
- **dtype:** bf16 is required.

The DFlash round loop issues only T ≥ 2, and T = 2 only on a response's last round. T = 1 is covered for completeness.

After the projections, rows go to `tileattn.verify()` when both `_post_ok` and `in_scope` hold. Otherwise the wrapper computes today's path for that T, and never re-enters the original method:
- T ≥ 3: `grouped_attention`, or `_row_loop` when `_post_ok` fails;
- T = 2: one bool-masked call;
- T = 1: one unmasked call.

When the tile is not active, verifyattn behaves exactly as it does today.

### Guards

`tileattn.enable(model, *, enabled, drafter_kind, verify_status)` returns `{state, reason, splits, probe_seconds}`, where state is `active`, `unavailable` or `off`.

**The flag.** Its first statement clears the module flag, on every call. The flag is a plain value, the active S or 0, and never a reference to the model or its modules. Only an `active` result sets it again. `VLMEngine.unload()` clears it too.

The guards run in this order, and any failure is `unavailable` with its reason:
1. **Config.** `enabled` false → `off`.
2. **Platform.** The shared NAX rule. It runs early so that an M2 reports the NAX reason, whatever its model.
3. **Model.** Type `qwen3_5`, and every full-attention module at 24 / 4 / 256, scale 1/16 and bf16.
4. **Drafter.** None, or a DFlash-family kind. When a drafter loaded, `verify_status` must be `active`. Otherwise decode would run the tile while the verifier runs stock SDPA, and drafter-on output would no longer equal drafter-off output.
5. **mlx.** verifyattn's whole `_gate()` passes: the version is in `VALIDATED_MLX`, `MLX_SDPA_BLOCKS` is unset, and every verifyattn source hash matches.
6. **mlx-vlm decode sources.** The source hashes of `Qwen3_5Attention.__call__`, `Qwen3_5Model.__call__` and `LanguageModel.__call__` (all in `mlx_vlm.models.qwen3_5.language`) are validated. Decode parity depends on a one-row forward reaching the replaced global with mask None.
7. **Split target.** The GPU core count is read once per process.
   - The read uses IOKit through ctypes: `IOServiceMatching("AGXAccelerator")`, `IOServiceGetMatchingServices` and `IORegistryEntryCreateCFProperty("gpu-core-count")`.
   - The fallback is `/usr/sbin/ioreg -a -r -c AGXAccelerator -d 1`, parsed with plistlib, with an absolute path and a timeout.
   - Exactly one matching service with an integer > 0 is required, and its `model` must equal `mx.device_info()["device_name"]`.
   - S must be in `MEASURED_SPLITS`.
8. **Probe**, on the `sous-model-load` thread.
   - **Inputs** are laid out as serving lays them out:
     - transposed `[1, T, 24, 256]` query slices;
     - K/V slices of 256-step padded buffers, plus buffers that end exactly at n with NaN planted past n in every KV head;
     - everything built with GPU ops only, with no float64 (which is CPU-only in mlx), and evaluated before the first launch, which avoids the #123 deadlock.
   - **Compile** every T = 1..8 variant, so no compile can first happen mid-turn.
   - **Parity.** Verify rows must be `mx.array_equal` to `decode()`:
     - at N0, at the first two step boundaries and at the boundary nearest 57K, with rows on both sides of each;
     - for T = 1 and 2 at every verifyattn plan transition below N0, where they go through `grouped_attention`, compared with stock M=1 SDPA.
   - **Correctness.** Parity alone cannot catch a kernel that is wrong the same way on both paths, so the probe also compares the tile with stock one-row SDPA.
     - Lengths: N0, one mid-step length, and the ~57K boundary.
     - Inputs: ×8 queries, a dominant key planted at n − 1 and another in the buffer at n.
     - Pass: the output is finite and its relative RMS difference is ≤ 1e-2, written so that NaN fails.
   - **The real module.** One of the loaded model's `Qwen3_5Attention` modules runs a one-row forward, with its own projections and rope, through a real `KVCache` holding a restored-style full buffer at n ≥ N0. The tile must match stock within the same bound.
   - **Cleanup.** The probe deletes its inputs and results and calls `mx.clear_cache()` before returning, as verifyattn's probe does, so the cache budget measured next reads true.
   - **A compile failure** logs the compiler's full report and keeps its first `error:` line as the reason.

**Failure policy (int8prefill's rule).** Nothing raises into a model load. Every failure is one `warnings.warn` plus `unavailable`, and the stock paths serve.

### Wiring and visibility

- **`vlm.py`.**
  - `VLMEngine.__init__` calls `tileattn.enable(...)` right after `verifyattn.enable` and before the fork store and cache budget are set up, on the loading thread. It passes the drafter's kind (None when there is none) and `verify_attention_status`.
  - The tile runs with or without a drafter.
  - Without one, decode is expected to gain about 4%. That figure is unresolved, and the gate below requires ≥ 1.00.
  - `unload()` clears the flag.
- **`lm.py`.** `LMEngine` sets `attention_tile_status = {"state": "off", "reason": None}` and passes it to `fork_key_fields`.
- **`config.py`.**
  - New field `attention_tile: bool = True`, read from `[model]`, and it joins `_KNOWN["model"]`.
  - A non-bool value warns and means false, i.e. today's paths, following `_int8_prefill`'s rule that a typo must not select changed numerics.
  - The daemon reads it at startup, so a change needs a restart.
- **`sous tune`.**
  - Arms inherit the key from the user's config.
  - It joins `Arm.suite_key` and the suite runner's row key, so a resumed run cannot mix tile and stock rows.
  - Like `_check_int8`, an inherited `true` that the engine reports `unavailable` prints a notice and runs on.
- **`base.py`:**
  - `ManagedEngine.attention_tile_status`, mirroring `verify_attention_status`;
  - the model-load line appends `attention_tile=active|unavailable|off`, plus `attention_tile_splits=` and `attention_tile_probe_s=` when active;
  - `EngineManager.status()` gains `attention_tile: {state, reason}`.
- **Logging.** Nothing is logged per turn. A module-level `calls` dict, as in verifyattn, counts decode and verify calls:
  - on the tile;
  - on the stock branch below N0;
  - declined after the projections;
  - out of scope;
  - straddles.

  Tests read it, and the merge gate exports it.

### Fork store

The tile changes what a token produces, in the last bits. The one-row forward that ends each prefill segment runs on it; the fork floor of 4,096 keys is above N0. That changes:
- that token's KV rows;
- through its hidden state, every later layer's KV and GDN state.

The kernel is also compiled at load by the OS's Metal toolchain, against the OS's MetalPerformancePrimitives, so its last bits can change with macOS. Therefore:
- **Epoch files.** `tileattn.py` joins `forkstore._EPOCH_FILES`. `attention_tile.metal` is already covered by the `kernels/*.metal` glob.
- **Tile field.** `fork_key_fields` takes a required `tile_status`, and records `tile` as `active:S` when the tile served this load, `off` otherwise. `unavailable` has the same numerics as `off`, because the flag is clear.
- **OS field.** `fork_key_fields` gains an `os` field, the macOS product version plus build, whenever a runtime-compiled sous kernel is active: the tile, or int8 prefill, which had the same gap.
- **What that guarantees.** A store written with the tile is never restored without it, with another S, or on another macOS build, and the reverse holds too.
- **Cold start.** Landing the change cold-starts every fork store once, including stores with the tile `off`, because the key and the epoch change. While the tile is active, a macOS update cold-starts the store again.

## Testing

CI's `test` job runs on macos-15, whose GPU cannot compile the kernel. The lint job never runs pytest. New test modules use `pytest.importorskip("mlx.core")`. S is monkeypatched in every CI test and never read from IOKit.

- **Pure logic:**
  - `chunk_for`, the step boundaries and `verify_plan` for S ∈ {10, 16, 20, 40}: at N0, at step edges, straddles, and runs cut at 8 rows for T = 9..16;
  - every gate predicate, in order;
  - IOKit parsing: one service, none, several, a mismatched model, and the unreadable case;
  - the config default, a non-bool value, and `_KNOWN`;
  - tune's suite and row keys;
  - the load-line and status fields for both backends;
  - `fork_key_fields` changing with the tile state, S and the OS;
  - `tileattn.py` in `_EPOCH_FILES`.
- **A stand-in kernel.** A plain-mlx implementation of the same partitioned attention replaces the Metal call, as the int8 tests replace `qmm`.
  - It computes each row independently, with per-row products and fixed-order reductions, never a matmul across packed rows. Its output then depends on the chunk and on nothing else about the call.
  - **Decode scope:** mask, sinks, shapes, dtype, scale, cache type, left padding, the flag, and n below N0.
  - **Verify scope:** T = 1..16, below and above N0, straddles, verifyattn inactive, and `_post_ok` or `in_scope` failing.
  - **Output:** assembly and row order, and verify rows bitwise equal to decode rows.
  - **Sensitivity:** a straddling run executed as one call at a single chunk, and decode at a neighbouring step's chunk, must both differ bitwise from the correct plan, so the equality tests are not vacuous.
  - **Probe decisions:**
    - it passes → `active`;
    - a chunk-ignoring stand-in → `unavailable` with the parity reason;
    - a stand-in scaled by 1.1 on both paths → `unavailable` with the correctness reason;
    - a raising stand-in → `unavailable`, no exception, and the flag cleared.
  - **Lifecycle in one process:**
    - active, then `enabled=False`;
    - active, then a failed gate;
    - active, then another model type;
    - active, then an unload.

    After each, decode and verify are bit-equal to the original functions, and the counters show no tile calls.
  - **Fixtures.** Tests install the decode hook through monkeypatch, so the module global is restored after each test.
- **mlx-vlm contract test.** A tiny random qwen3_5 language model with 24 / 4 / 256 attention and about 2K tokens of prefill runs through the real forward and the exact verifier, with the stand-in kernel.
  - **Precondition, asserted first.** With the tile inactive, verify rows equal step-by-step decode bitwise, below N0 and at T = 3 above it. That is main's own exactness on this GPU. If it fails on CI, quantize the tiny model, as the 27B is, rather than weaken the tile assertion.
  - **With the stand-in tile:**
    - verify-row logits equal step-by-step decode logits;
    - the counters show decode and verify on the tile, T = 2 included;
    - a straddle splits.
  - **Tile off:** logits are bit-equal to the precondition's reference.
- **`nax`-marked tests** skip on the tile's own availability, read once at collection: the platform rule, plus a compile-and-run of every T variant. They do not use int8's availability.
  - the real kernel, with the probe returning `active`;
  - exactness at a spread of step boundaries;
  - error against an fp32 reference no worse than stock's;
  - NaN planted past n in every KV head, with the buffer ending at n;
  - the contract test again, with the real kernel;
  - a deliberately broken kernel source, run in a subprocess with a 120 s timeout. It must report `unavailable` with its `error:` line and exit cleanly.
- **Model test** (`-m model`, manual, M5 Pro). The real 27B, greedy, prompt cache off, drafter on against drafter off.
  - It must be identical at blocks 3 and 8 (the engine's block size is set directly, since config caps it at 5), on a ~1,450-token prompt whose generation crosses N0 and the first step, and on one ending just below the 57,600 step.
  - The counters must show verify runs below N0, T = 2, straddles and, at block 8, T = 8 on the tile.
  - `attention_tile = false` must be identical to main.

## Merge gate

**Setup.** The M5 Pro, with the maintainer's daemon idle and one GPU job at a time.
- **Harness.** The spike's replay harness replays #131's real subagent turns through an isolated `sous serve`.
  - It runs the branch under a GATE_WRAP shim that writes `tileattn.calls` and `verifyattn.calls` deltas after every `generate()`.
  - Every run uses `max_context_tokens = 262144` and `prompt_cache_gb = 14`, with its own HOME, and deletes its forks afterwards.
  - A run counts only if the lock reports the daemon stayed idle and the harness confirmed the child daemon's process group gone and its port free.
- **Load.** The M5 Pro's load line shows `attention_tile=active attention_tile_splits=20`, and the probe's cost is recorded.

**Pass criteria:**
- **Path.** In every branch run, tile decode and verify counts are > 0, with 0 declined and 0 out of scope. The off run makes 0 tile calls.
- **Speed.** Sampled, in ABBA order (main, branch, branch, main) on the 64 turns:
  - pooled `out_tokens / decode_s` is ≥ 1.15x branch/main, and the turn-bootstrap 95% CI excludes 1.10;
  - the difference in tokens per round has a CI that includes 0;
  - prefill tok/s is within ±3% of main in the same ABBA, and peak `memory_gb` within ±0.2 GB.

  Greedy is reported on identical-output turns only. Any max_tokens stop or repetition loop is listed and compared with main's run of the same turn.
- **Drafter off.** With `speculative_draft_id = ""`, sampled, ABBA on a 16-turn subset: branch/main ≥ 1.00 pooled.
- **Parity.** The branch with the drafter against the branch without it, greedy, max_tokens 256.
  - At least 16 turns, including the a6443762 continuation at ≥ 128K.
  - Every reply and stop reason must be identical.
  - The prompt-cache history (hit or miss, slot taken, reused and prefilled tokens) must be equal on every compared pair.
  - One divergence fails the gate.
- **Off equals main.** `attention_tile = false`, greedy, with the same settings: reply-identical to main on the 64 turns, with equal cache histories.
- **Suite.** 8 tasks × 6 runs per arm, in alternating rounds.
  - The branch's mean grade is at least stock's − 0.02.
  - No task that is 6/6 on stock drops below 5/6 on the branch.
  - No new failure mode appears; the failing diffs are compared, as the spike did.
- **Kernel timings.** In one process, interleaved, the in-tree entries run against stock M=1 / #128 grouped and against the spike's `tileattn.py` kernel, at T = 1, 3 and 8 from 2K to 262K, with an A/A pair to show the noise floor.
  - In-tree / spike median ratio within 0.97–1.03 in every cell.
  - Speedup over stock or grouped within 5% of the spike's table.
- **M2 Air.** Running the 9B with its DFlash drafter:
  - the load line reads `attention_tile=unavailable` with the NAX reason;
  - greedy replies on the #143 M2 task set are identical to main's.
- **`scripts/api_smoke.py`** passes on both Macs.

## Documentation

- **A CLAUDE.md gotcha:**
  - **Why parity holds:** a fixed partition per step, runs split at steps and at 8 rows, and stock below N0.
  - **The decode hook:** a module-global replacement installed on the first activation. Its scope rests on the flag and the checks, and it is decided after the projections.
  - **The flag's lifecycle.**
  - **The gates and the probe.**
  - **The fork-key fields.**
  - **Timings on kernel edits:** every kernel edit, and every extension of `VALIDATED_MLX`, needs the tile's exactness tests and its T = 1, 3 and 8 timings re-run.
- **Amend verifyattn's "T = 2 is left alone" gotcha:** it holds only while the tile is inactive.
- **`docs/configuration.md`** documents `attention_tile`, including that a change takes effect on the daemon's next start.
- **The README's decode figures** change only if the merge gate's numbers justify it.

## Out of scope

- The prefill attention tile, which failed its gate.
- int8 KV.
- Block 8 and cheaper verify projections (#126, #133).
- Other model families, head dims and dtypes, and the LM backend.
- Split targets other than the measured ones, and per-chip tuning.
