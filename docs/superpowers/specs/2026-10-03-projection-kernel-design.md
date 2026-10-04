# sous — Small-row projection kernel for verify and decode

*2026-10-03. The in-tree design for #148. It is grounded in out-of-tree work on the maintainer's M5 Pro (applegpu_g17s, 20 GPU cores, macOS 27.0, mlx 0.32.2, mlx-vlm 0.7.2, the default Qwen3.8-27B-4bit with DFlash2) in two stages: #133's microbench (results on #133, 2026-10-01) and #148's in-model spike (results on #148, 2026-10-02). The maintainer ruled design (b) for greedy parity, chose the engine-resolved block default and approved the design in four parts before this spec was written. Before commit, three adversarial reviews checked an earlier draft against the code, mlx-vlm 0.7.2's source and the spike's data: one for code correctness, one for measurement claims and the merge gate, one for plan-ability. Their findings are folded in.*

## What the measurements established

**mlx-vlm's exact verifier makes deeper blocks expensive in their projections.** With #132's tile, verify attention is cheap, and the verify round's remaining cost is mostly its quantized projections.
- **The verifier's cost.** It keeps each row bitwise equal to one-row decode (qmv), but its cost climbs with rows.
- **The spike's kernel.** Its simdgroup MMA kernel stays nearly flat. It multiplies exact half-precision nibbles by fp32 activations in 8×8×8 simdgroup MMAs with fp32 accumulation, and reduces its K split in a fixed order.
- **NAX is not the answer.** dflash-mlx's tensor-unit kernel is slower than this one on mlx's layout (#133).

Projections per 27B verify forward, summed from #133's dependent-chain microbench (ms; in_proj_a/b excluded):

| rows | exact verifier | simdgroup kernel | qmv (one row) |
|---|---|---|---|
| 1 | 62.6 | 75.0 | 58.7 |
| 4 | 76.4 | 78.1 | |
| 5 | 98.5 | 78.2 | |
| 6 | 129.0 | 77.2 | |
| 8 | 178.6 | 78.9 | |

**In the model, the kernel makes block 5 pay.** #148 replayed #131's 64 real subagent turns (44–77K tokens) through a real `sous serve`, drawing the arm at random each round.
- **The bar** (pre-registered: 5k/4v ≥ 1.05 with the CI above 1.00) passed.
- **Where the gain comes from:** a block-5 kernel round costs 5 ms more than today's block-4 round. Without the kernel the projections alone add 22 ms from 4 to 5 rows (#133's 98.5 − 76.4).

| arm | ms per round | tokens per round | sampled tok/s vs 4v (95% CI) | greedy tok/s vs 4v |
|---|---|---|---|---|
| 4v (today) | 110.2 | 3.00 | 1.000 | 1.000 |
| 4k | 112.8 | 3.03 | 0.987 [0.965, 1.011] | 0.980 |
| 5k | 115.2 | 3.46 | **1.103 [1.072, 1.141]** | 1.124 |
| 6k | 127.4 | 3.96 | 1.142 [1.085, 1.215] | 1.131 |

**The kernel's rows are bitwise independent of the row count and of which linears share a dispatch.**
- **The premise check.** On the 27B's real weights, every fused group's T-row output (q+k+v, gate+up, qkv+z+b+a) and every single (o, down, out_proj, lm_head) equals the same rows computed one at a time.
- **Why it holds:**
  - the products are exact;
  - the MMA sums its eight terms in a fixed sequential order;
  - the per-group scale and bias epilogue and the K-split reduction run in a fixed order;
  - rows past T are zeroed;
  - nothing in the dispatch depends on T.
- **Not bitwise qmv, but more accurate.** Against a float64 reference its error is 1.5–1.7e-3, at bf16's rounding floor. qmv's is 6–10e-3, and on lm_head the kernel's KL is 10x lower.

**Greedy parity needs the kernel on decode too** (design (b)). Greedy, fixed block 5, 64 turns:

| design | drafter on vs off identical | drafter-off decode |
|---|---|---|
| stock (control) | 64/64 | 1.00x |
| (a) kernel for verify rows ≥ 5 only | 57/64 | 1.00x |
| **(b) kernel for every verify and for decode** | **64/64** | **0.784x** (10.68 vs 13.62 tok/s) |

- **Greedy speed.** Greedy speculative decode at block 5 runs 2.31x stock plain greedy decode under (b), against 2.0x at today's block 4. Against (b)'s own drafter-off decode it is 2.94x. The spike's greedy runs also sent the greedy drafter's head argmax through the kernel, which this design leaves stock, so the in-tree greedy figures may differ slightly.
- **What (b) changes without a drafter.** Its greedy output differs from today's on 14 of 64 turns, as any numeric change would.

**No quality harm on the tool-loop suite.** Measured with every arm given one shared project path per prompt; the per-arm paths in `sous tune`'s suite are a confound (#149).
- **Full suite, 8 tasks × 6 prompts:**
  - main default 1.000, 48/48 perfect;
  - (b) at block 5 0.986, 46/48 perfect;
  - 2 discordant pairs, p = 0.25.
- **The two misses** are the same migrate_config test that stock block 5 also misses.
- **dataclass_from_schema** passed 40/40 prompts in each of stock block 5, design (a) and design (b). Main's default, block 4, was not one of those arms.

## Decisions

- **Design (b), the maintainer's ruling.** Every verify and every target call of ≤ 8 rows outside the verifier goes through the kernel, decode included. A verify of more than 8 rows (block 0, the drafter's own policy, can reach 16) is split into runs of 8 or fewer, which is exact because the kernel is T-invariant.
  - That keeps greedy output with the drafter on equal to output with it off.
  - The cost: drafter-off decode runs at about 0.78x where the kernel is active.
- **On by default, with an opt-out.** `[model].projection_kernel = true`. `false` runs today's paths, and the block default then resolves to 4.
- **Rides on the tile.** The kernel is `active` only where #132's attention tile is `active`. That inherits the tile's M5 Pro gate, and with a drafter it inherits exact verify attention too. Block 5 was measured only with both.
- **Engine-resolved block default, the maintainer's choice.** An unset `speculative_block_size` means 5 where the kernel is `active` and 4 otherwise. An explicit value always wins, and 0 keeps its meaning: the drafter's own policy.
- **The measured kernel family, whole.** mode h only, with the spike's three configurations:
  - the plain variant for small shapes;
  - the staged variant for large ones;
  - the staged variant with presummed activations for lm_head.

  Everything else the spike explored is dropped.
- **Two tag-gated hooks.**
  - The verifier's `_linear` and `_linears` cover every verify projection and the verify head.
  - `nn.QuantizedLinear.__call__` covers decode and short prefill tails, only on modules the kernel tagged on the loaded target.
- **Left stock:**
  - the greedy drafter's head argmax (`quantized_argmax`), which affects proposals only;
  - every non-target module, the drafter's own included.
- **Lives in sous.** Runtime gates on mlx, mlx-vlm and the platform; no upstream PR.

## Design

### Files

- **`src/sous/engine/kernels/projection_mma.metal`** (new). The spike's mode-h kernel bodies (`kern.py`, `kern_st.py`), ported:
  - the plain kernel;
  - the staged kernel;
  - the group-sums prepare kernel.

  It uses only `metal_simdgroup_matrix` and `bfloat`. There is no MPP and no `nax.h`, so it compiles on any Metal GPU that supports half×float simdgroup MMA (MSL 3.2 on the M2, 4.1 on the M5) and can be tested on CI.
  - **Layout.** The `mma8` helper and the `PRESUM` switch sit in a new `kernels/projection_mma.h`. The bodies are split by markers, as `attention_tile.metal`'s are.
  - **Codegen.** Python fills in the per-linear-count select (1–4 linears) and the template constants.
  - **Names.** Every variant, linear count and presum combination gets its own kernel name (`sous_proj_{plain,staged,staged_ps}_n{1..4}`), because `int8prefill._kernel` caches by name.
  - **Compile options.** Built with `compile_options={"math_mode": "safe"}` stated explicitly, as the tile's are.
- **`src/sous/engine/projkernel.py`** (new), in the shape of `tileattn.py`:
  - mlx imports are function-local;
  - mlx-vlm internals are resolved with importlib;
  - nothing raises into a model load;
  - it reuses `int8prefill._kernel`/`_kernel_text`;
  - it exports `linear(s)`, `enable`, `clear`, `kernel_available`, `static_reason`, `calls` and the tag;
  - the hooks reach the kernel through a module attribute (`projkernel.mma`), the seam tests swap a plain-mlx stand-in into, as `tileattn.tile` is.
- **`src/sous/engine/int8prefill.py`.** All three of its wrappers (`nn.QuantizedLinear.__call__`, `Qwen3_5MLP.__call__` and `Qwen3_5GatedDeltaNet.__call__`) gain `functools.wraps`. Without it, a source pin hashes int8's wrapper, and `int8_prefill = true` would make the kernel read as pin drift. The stale "verify uses ≤ 6 rows" comment is corrected.
- **`src/sous/engine/vlm.py`, `lm.py`, `base.py`, `forkstore.py`, `config.py`, `tune/arms.py`, `tune/decide.py`, `tune/suite/runner.py`, `kernels/__init__.py`, `THIRD_PARTY_NOTICES.md`.** The wiring, visibility, block default, fork key and notices below.

### Kernel

- **Arithmetic contract.** The frozen constants are F = 4, NSG = 1, KS = 4, GB = 4 (staged), CH = 1, SPLITS = 1, PF = 1 and SBV = 1. CH or SPLITS above 1 would change the summation order and break plain = staged.
  - Weights are dequantized exactly: `as_type<half2>(nibble | 0x64006400) − 1024`. Activations are bf16 widened to fp32. Each product is therefore exact.
  - One 8×8×8 `simdgroup_multiply_accumulate` per fragment, accumulating in fp32 in its sequential order.
  - Per quant group, in order: `acc = fma(dot, scale, acc); acc = fma(sum_x, bias, acc)`.
  - The four in-threadgroup K partials are reduced in a fixed order.
  - Rows ≥ T are zeroed, and x pointers are clamped. The output is bf16.
  - KS is part of the arithmetic: changing it changes the bits, so it is a constant, not a knob.
- **Variants agree bitwise.** That is what makes choosing by shape safe. The choice depends only on K and on the summed N of the linears in the call, never on T:
  - summed N ≥ 100,000: staged, presummed x via the group-sums kernel (lm_head);
  - summed N · K ≥ 40e6: staged;
  - otherwise: plain.

  The staged variant applies only when K % 1024 = 0, and falls back to plain otherwise. The plain one needs K/64 ≥ 4. The 27B's K (5120, 6144, 17408) satisfies both.
- **Multi-linear dispatch.**
  - One call serves 1–4 linears that share x, with tiles laid out linear by linear.
  - Partial tiles are clamped on read and guarded on write, so N = 48 (in_proj_b/a) works.
  - Each tile serves exactly one linear, which is why a fused group equals its singles.
- **Inputs.** `nn.QuantizedLinear`, affine, 4-bit, group size 64, bf16 scales and biases, no `bias`. x is bf16 with 1 ≤ B·T ≤ 8 rows, made contiguous.
- **Compilation.** The row count and the shape constants are template arguments, so each (dispatch key, T) is its own pipeline.
  - **The 27B's 12 dispatch keys.** Verify uses q+k+v, gate+up, qkv+z+b+a, o/out_proj, down and lm_head. Decode and prefill tails add the singles q, k/v, gate/up, in_proj_qkv, z and b/a.
  - **Count:** 12 keys × T = 1..8, plus 8 group-sums pipelines, is about 104.
  - **Cost:** on the M2 a cold first compile cost about 71 ms per pipeline (the reviewer's measurement on the spike's kernels); a second process found them in Metal's cache at about 1 ms each.
  - `enable()` compiles every one the model uses at load.
  - The first plan task measures the cold and warm compile time on the M5 Pro. If the cold load exceeds the gate's bound, compiling each T on first use is the fallback.

### Routing

- **Tags.** On success, `enable()` tags every `QuantizedLinear` of the loaded target's language model with `object.__setattr__(m, "_sous_projection_kernel", True)`, which keeps the tag out of mlx's parameter tree. On the 27B that is 497 modules, lm_head included. The drafter's own linears are separate objects and are never tagged.
- **Flag.** A plain int, `_active` (1 or 0), never a model reference.
  - Set: only by an `active` result.
  - Cleared: by the first statement of every `enable()` and by `clear()` (called from `VLMEngine.unload()` beside `tileattn.clear()`, before anything there can raise). The probe raises it temporarily and always leaves it at 0.
  - Both hooks read it on every call.
- **Installed once per process.** Both hooks are `functools.wraps` wrappers carrying a mark attribute (as tileattn's `_HOOK_MARK`), installed on the first activation and never removed. `wraps` copies `__dict__`, so the mark survives an int8 wrap on top. With `wraps`, the next load's source pins follow the wrappers to the originals.
- **Hook 1, verify.** A class-level wrap of `Qwen3_5BatchInvariantForward._linear` and `_linears` (`models/qwen3_5/speculative_verifier.py`).
  - **Routes when:** `_active`, every linear in the call is tagged, and x is bf16 with ndim 3.
  - **More than 8 rows:** the call is split along T into runs of at most 8 and the outputs concatenated. This covers block 0, where the drafter's policy can verify up to 16 rows.
  - **Otherwise** the original method runs.
  - **Coverage:** every verify projection, through the fused groups q+k+v, gate+up and qkv+z+b+a and the singles o_proj, down_proj and out_proj, plus the verify head `_linear(lm_head)`, which feeds both the sampled walk and greedy's `sampler(logits)`.
  - **verifyattn:** its `_attention` wrapper calls `self._linears`/`self._linear`, so it is covered.
  - **Other families.** Hooking the class leaves `speculative/ops/linear.py`'s globals alone. But the Gemma4, NemotronH and Qwen4Exp verifiers subclass `Qwen3_5BatchInvariantForward` and inherit the wrap, so only the tag keeps them on stock. A test pins that.
- **Hook 2, everything else on the target.** A wrap of `nn.QuantizedLinear.__call__`.
  - **Routes when:** `_active`, the module is tagged, x is bf16, and 1 ≤ rows ≤ 8, where rows is the product of the leading dimensions.
  - **Otherwise** it falls through to the method it wrapped.
  - **Coverage:**
    - one-row decode (`generate_step`'s `_step`, every projection a plain module call);
    - the last prefill chunk when it is ≤ 8 rows, and the one-row forward that ends every prefill segment;
    - the sampled DFlash drafter's call of the target's shared `lm_head` (`DFlash2DraftModel._logits`), which changes proposals only.
- **Coexistence with int8.** int8 wraps the same method for its own tags at ≥ 128 rows, measured on `shape[-2]`. The row ranges are disjoint, and each wrapper falls through to whatever it wrapped, so the install order does not matter.
- **Why the two hooks agree.** Decode calls each linear alone; verify fuses groups. The kernel's rows are T-invariant and group-invariant, so a verify row and the decode of the same token produce identical bits. The probe proves that through both hooks on the loaded model.
- **The residual parity risk.** This one is not the kernel's. Drafter-off greedy takes its argmax over bf16 `logits − logsumexp(logits)` (`generate/ar.py`), while greedy verify takes it over raw logits (`dflash.py`, `sampler(logits)`).
  - **When it matters:** rounding is monotone, so the two can disagree only if the subtraction rounds the top two logits into a tie, in a flat row whose top logit sits below half the logsumexp.
  - **Not new:** stock and the tile already depend on the same thing.
  - **Coverage:** the probe cannot cover it; the gate's parity runs are its witness.

### Guards

`projkernel.enable(model, *, enabled, drafter_kind, tile_status)` returns `{state, reason, probe_seconds}` and never raises. In order:

1. `_active = 0`. Tags left on a previous, unloaded model are inert while `_active` is 0, and holding a reference to untag it would keep its weights alive. So the previous model is not touched.
2. `enabled=False` returns `off`.
3. The tile: `tile_status["state"] != "active"` refuses with the tile's reason, or `attention tile off` when the tile was switched off.
4. The model:
   - `model_type` `qwen3_5` (the MoE variant is refused);
   - every `QuantizedLinear` of the language model eligible by the input rules above;
   - bf16 activations.
5. The drafter: none or `dflash`. The target's language model must not define `speculative_verify_dflash_hidden`: if it did, greedy verify would take its tokens from `quantized_argmax`, which bypasses both hooks.
6. The pins, in `VALIDATED_PROJ_SOURCES`:
   - sha256 of `inspect.getsource`, following `__wrapped__`, as `verifyattn._source_digest` does;
   - covers `Qwen3_5BatchInvariantForward._linear`, `_linears`, `_feed_forward`, `_gated_delta`, `_model` and `__call__`;
   - covers `Qwen3_5MLP.__call__`, `Qwen3_5GatedDeltaNet.__call__`, `_project_gates`, `generate_step` and mlx's `nn.QuantizedLinear.__call__`;
   - covers `_dflash_verify` and `_dflash_verify_greedy`, which decide that greedy verify reaches `_linear(lm_head)`;
   - plus the mlx version set in `verifyattn.VALIDATED_MLX`.
7. Warm-up: compile every pipeline the model's dispatch keys need, T = 1..8. Inputs are built with GPU ops and evaluated first, because of the CPU-stream deadlock.
8. Tag the target, install the hooks if needed, and run the probe on the `sous-model-load` thread with `_active` raised temporarily:
   - **Through the hooks:** the verifier's layer-3 `_feed_forward` at T = 5 and the same layer's plain modules at T = 1, with `calls` showing both hooks reached. The verify rows must be bitwise equal to the plain rows.
   - **Directly, on real linears:** layer 3's q+k+v, o and gate+up+down, layer 0's qkv+z+b+a and out_proj, and lm_head.
     - At T = 1..8 each row is bitwise equal to the one-row single, and fused groups equal their singles.
     - Relative RMS against stock `quantized_matmul` is ≤ 1e-2 in fp32, which catches a kernel that is wrong the same way on both paths.
     - A NaN planted in one row leaves the others unchanged.
   - **Cleanup:** it saves and restores `calls`, leaves `_active` at 0, and calls `mx.clear_cache()` in `finally`. On failure it untags the model it was given.
9. Set `_active = 1` and return `active`.

**Refusal kinds.**
- **Expected** (steps 3–5), which says the kernel does not apply to this load: one INFO line, `projection kernel unavailable: <reason>`, plus `unavailable`.
- **Unexpected**, which says it should have run and did not (pin drift, a compile or probe failure, any exception, logged with its traceback): one `warnings.warn` plus `unavailable`.
- `off` is never a refusal.

### Block default

- **`config.py`:**
  - `speculative_block_size: int = 4` keeps its value.
  - A new `speculative_block_explicit: bool = False` is set True by `load_config` when `[model].speculative_block_size` is present and valid. A clamped value counts as explicit; an invalid one falls back with its warning and is not explicit.
  - A new `SPECULATIVE_BLOCK_KERNEL = 5` sits beside `SPECULATIVE_BLOCK_DEFAULT = 4` and `SPECULATIVE_BLOCK_MAX = 5`.
- **The invalid-value warning** says "using the default" instead of naming 4.
- **`default_engine_factory`** passes `draft_block_size` as today plus a new `draft_block_explicit` (`config.speculative_block_explicit`). `_default_factory` and `VLMEngine` take `draft_block_explicit: bool = True`, so engines built directly (tests, scripts) are unaffected, and the block stays an `int` throughout.
- **`VLMEngine.__init__`.**
  - `projkernel.enable` runs right after `tileattn.enable`, because it needs the tile's status. That is before `draftctx.enable`, the ForkStore key and `measure_cache_budget`.
  - `_pin_block_size` moves after it. That move is safe: verifyattn never sees the drafter, tileattn reads only its kind, and draftctx never reads `prefer_requested_block_size`.
  - When the block is not explicit and the kernel is `active`, the block becomes `SPECULATIVE_BLOCK_KERNEL`, and `self._draft_block_size` is reassigned before the pin.
- **`sous tune`:**
  - `_arm` sets `speculative_block_explicit=True` whenever it pins a drafter's block. `dataclasses.replace` would otherwise copy the user's `False`, and every arm on a kernel Mac would run block 5 whatever its label says.
  - **The static check.** To mark the user's current arm, and for `decide` to compare against, tune resolves the user's effective block by the same rule. It uses `projkernel.static_reason(config, model_config, drafter_kind)`, one function tests can monkeypatch. It checks, without loading a model:
    - the NAX platform rule and the tile's core-count match;
    - `attention_tile` and `projection_kernel` both true;
    - from `fetch_model_config`: model type `qwen3_5` and the tile's attention shape 24/4/256;
    - the drafter kind.
  - **Known gap:** a load-time refusal the static check cannot see (a probe failure, pin drift) can leave the current-arm mark one block off. That is documented, not hidden.
  - **Suite keys.** `projection_kernel` changes output, so it joins `Arm.suite_key` and the suite row key (`SuiteRun.key`, with a `from_dict` default for older rows), as `attention_tile` did. `tune/suite/runner.py` gains its notice for `unavailable`.

### Wiring and visibility

- **Constructors.** `VLMEngine(projection_kernel=False)` is the constructor default, as for the tile and int8. Only `default_engine_factory` passes the config's value. The LM backend reports `{"state": "off"}`.
- **Config key.** `[model].projection_kernel` joins `_KNOWN["model"]`. A validator in the `_attention_tile` style makes a non-bool warn and mean false. A change takes effect on the daemon's next start.
- **Load line.** It gains `projection_kernel=<state>`, plus `projection_kernel_probe_s=` when active, and `draft_block=<n>` whenever a drafter loads.
- **Status document.** The `engine` block gains `projection_kernel: {state, reason}` and `draft_block`.
- **Per-turn logging.** None. `projkernel.calls` counts each path for tests and harnesses, with three keys:
  - `verify_kernel` and `plain_kernel` count calls served by hook 1 and hook 2;
  - `declined` counts calls that met the tag and the flag but were turned away (wrong dtype, too few dimensions).
- **`sous tune`** prints a notice only for `unavailable`, never for `off`.

### Fork store

- **Why it changes.** Under (b) the one-row forward that ends every prefill segment, and any prefill tail of ≤ 8 rows, runs on the kernel. Stored KV and GDN state therefore depend on it.
- **The changes:**
  - `projkernel.py` joins `forkstore._EPOCH_FILES`; the `.metal` file is hashed by the existing directory scan.
  - `fork_key_fields` gains `proj: active|off`, with `unavailable` counting as off.
  - Callers pass the status, and `lm.py` passes `off`.
- **The drafter stays out of the key.** The kernel's routing never depends on whether a drafter loaded, which preserves the store's existing premise.
- **On deploy** every store goes cold once.

### Notices

- **The new section.** THIRD_PARTY_NOTICES.md gains a Splash section (incoai/splash, Apache-2.0, in the existing form). It names what the kernel derives from Splash's M1/M2 port and its Apple9 path:
  - the `nibble | 0x6400 − 1024` half weights times fp32 activations;
  - the factored scale/bias epilogue;
  - the W·Xᵀ orientation and the probed simdgroup lane map.

  It states that the in-threadgroup split-K, the staging, the group sums and the multi-linear codegen are sous's own.
- **Verbatim check.** A plan task compares the port line by line with the upstream sources before the notice's wording is final.
- **`kernels/__init__.py`'s docstring** lists the new kernel. `pyproject.toml`'s license files already include Apache-2.0.

## Testing

**CI** (macos-15 has a Metal GPU but no M5). The kernel needs no MPP. Its tests run wherever `projkernel.kernel_available()` succeeds, and skip otherwise.
- **What `kernel_available()` does.** It compiles every variant and linear count, runs each once on GPU-built inputs, checks T-invariance and agreement with a plain-mlx dequantized reference, and remembers only success.
- **Lane-map risk.** The lane map was probed on Apple GPUs. A GPU where it differs then skips rather than failing, and the first CI run must confirm the arithmetic tests ran rather than skipped.

The tests:
- **Arithmetic:**
  - T-invariance at T = 1..8, with each row equal to the same row computed alone and to the matching rows of T = 8;
  - group invariance for 2, 3 and 4 fused linears, including N = 48 and an N that is not a multiple of 32;
  - the plain, staged and presummed variants agree bitwise;
  - a NaN in one row leaves the others unchanged;
  - accuracy against a float64 dequantized reference;
  - unsupported inputs (fp16 x, more than 8 rows, other bits or group sizes, a bias, another mode) fall through to stock.
- **Fixtures.** `guard_proj_state` restores `nn.QuantizedLinear.__call__`, the verifier's `_linear`/`_linears`, `_active`, int8's install flag and the compile verdict. `proj_ready` fakes the tile status, the platform and the pins, and swaps the stand-in into `projkernel.mma`.
- **Gate:**
  - `enable()` order and INFO-versus-warn per refusal, with the platform, the tile status and the pins faked as `tile_fixtures` does;
  - enable, clear, then enable again on a fresh model in one process stays `active`, which is the pins following `__wrapped__`;
  - with int8 active first, projkernel's pins still pass;
  - the flag's lifecycle across `enable`, `clear` and a failed probe;
  - pin drift;
  - tag scoping: the target's linears only. The drafter's own modules are never tagged; the `lm_head` it shares with the target is excluded from that check by identity;
  - other families' verifiers that inherit hook 1 stay on stock;
  - a verify of more than 8 rows is split into runs and equals the same rows computed alone;
  - both hooks fall through when the flag is clear, the module is untagged or rows exceed 8;
  - int8 coexistence in either install order, with int8's ≥ 128-row path unchanged.
- **Contract.**
  - The real mlx-vlm verifier forward and the real plain forward on a tiny qwen3_5-shaped quantized model, with a plain-mlx stand-in for the kernel: verify rows must equal decode rows, and both hooks must be reached.
  - The real `generate_step` over a stub model must reach hook 2 on the prefill tail and on decode.
- **Config and block:**
  - `speculative_block_explicit` set by `load_config` for present, clamped and invalid values;
  - the engine resolving 5 or 4;
  - tune's current-arm and `decide` resolution;
  - the existing default-block tests updated.
- **Fork store and visibility:** the epoch file list and the `proj` key field; the load-line and status pins.

**Model tests** (`-m model`, the M5 Pro): greedy output identical with the drafter on and off at block 5, at block 8 set directly on the engine, and at block 0 (the drafter's policy, verifies above 8 rows), with the kernel active, as `test_tileattn_model` does for the tile.

## Merge gate

**Setup.** The M5 Pro, with the maintainer's daemon idle and one GPU job at a time.
- **Harness.** #148's replay harness: #131's 64 real subagent turns through an isolated `sous serve` on each tree.
  - A GATE_WRAP shim writes `projkernel.calls` deltas and per-round records after every `generate()`.
  - Every run uses `max_context_tokens = 262144` and `prompt_cache_gb = 14`, with its own HOME, and deletes its forks afterwards.
  - A run counts only if the lock reports the daemon stayed idle.
- **Load.** The branch's load line shows `projection_kernel=active draft_block=5`.
  - **Warm load:** warm-up plus probe adds at most 1 s.
  - **Cold load:** the first load after install (empty Metal cache) is recorded and must add at most 10 s. Otherwise per-T lazy compilation replaces the load-time warm-up before merge.

**Pass criteria:**
- **Port equals spike.** On the 27B's real weights, for every dispatch key at T = 1..8, the in-tree kernel's output is `array_equal` to the spike's `verify_proj`. That is what carries the spike's parity and quality evidence over to the port.
- **Path.**
  - `declined` = 0 in every branch run.
  - `verify_kernel` > 0 in every branch run with a drafter, and `plain_kernel` > 0 in every branch run.
  - A `projection_kernel = false` run makes 0 kernel calls and shows `draft_block=4`.
- **Speed, primary.** The spike's method on the branch.
  - **The arms:** the GATE_WRAP shim draws the arm per round inside one branch process: 4v (kernel flag cleared, block 4) or 5k (kernel active, block 5). Configured block 5, two sampled seeds.
  - **The metric:** #148's stratified dt-attr metric with a 95% CI over turns.
  - **Pass:** 5k/4v ≥ 1.05 with the CI lower bound > 1.00. A point in [1.00, 1.05) with the CI above 1.00 is not a pass; it goes back to the maintainer with the numbers.
- **Speed, secondary.** Sampled, in ABBA order (main, branch, branch, main) on the 64 turns, each tree at its default.
  - Pooled `out_tokens / decode_s` branch/main is reported, and must not be below 1.00.
  - Fixed-config ABBA is weak for a ~10% effect: #118's read +2.8% [−1.0, +6.5] on a 4.9% effect. And a tree's two runs may replay identical outputs (#149's per-thread sampling). So this is a sanity check, not the bar.
  - Prefill tok/s is within ±3% of main in the same ABBA, and peak `memory_gb` within ±0.2 GB.
  - Greedy is reported on identical-output turns only.
- **Drafter off.** `speculative_draft_id = ""`, sampled, ABBA on the first 16 of the 64 turns in gate132's order: branch/main ≥ 0.75 pooled.
  - This restates the tile's ≥ 1.00 criterion, which design (b) cannot meet by construction. The spike measured 0.784x, with hook overhead in both of its arms; main has no hook, so the margin is thinner than it looks.
- **Parity.** The branch with the drafter against the branch without it, greedy, max_tokens 256, on all 64 turns plus the a6443762 continuation at ≥ 128K.
  - Every reply and stop reason must be identical, and the prompt-cache history must be equal on every pair.
  - One divergence fails the gate. Before ruling, the first differing token's top-2 logit gap is read, to tell a kernel fault from the argmax residual above.
- **Off equals main.** `projection_kernel = false`, greedy, same settings: reply-identical to main on the 64 turns, with equal cache histories. The block's resolution to 4 is checked separately: `draft_block=4` on the load line, and at most 3 drafted tokens per round on every turn line. Reply identity cannot show the block, since stock greedy output does not depend on it.
- **Suite.** #148's path-neutral paired design (suite148g): 8 tasks × 6 prompts, branch default against main default.
  - **Runs:** each (prompt, tree) runs on a fresh load in a process of that tree, with the prompt's project root identical in both trees and the same run order.
  - **Pass:** no harm at a one-sided exact McNemar p < 0.10 on perfect runs. Failing diffs are compared, as the spike did.
  - **Detection limit:** with 48 pairs this flags harm only at 4–0 discordance or worse (3–0 gives p = 0.125).
  - **Dropped rule:** the tile gate's "no 6/6 task below 5/6" is dropped on purpose. The spike's own stock block-5 arm took migrate_config from 6/6 to 4/6 with the same failure as the kernel arm, so the rule would fail block 5 itself.
- **Kernel timings.** The in-tree kernel against the spike's `verify_proj` in one process, interleaved, for each of the 12 dispatch keys at T = 1..8, with an A/A pair to show the noise floor.
  - Each cell's band is the larger of ±3% and the A/A spread.
  - The per-forward sum (#133: 75–79 ms) must land within ±2%.
- **M2 Air.** Running the 9B with its DFlash drafter:
  - the load line reads `projection_kernel=unavailable` with the tile's reason and `draft_block=4`;
  - greedy replies on the #143 M2 task set are identical to main's;
  - drafted tokens per round are at most 3.
- **`scripts/api_smoke.py`** passes on both Macs.

## Documentation

- **A CLAUDE.md gotcha** for `projkernel`:
  - the arithmetic contract and why fused groups equal singles;
  - the split of verify calls over 8 rows;
  - the argmax residual that parity still rests on;
  - the two hooks and their tag-and-rows scope;
  - the flag's lifecycle;
  - int8 coexistence;
  - the guards and the probe;
  - the `proj` fork-key field and the epoch file;
  - the rule that every kernel edit re-runs the exactness tests and the timing check.
- **`docs/configuration.md`:**
  - documents `projection_kernel`;
  - restates the block default as "5 where the projection kernel is active, else 4";
  - updates greedy speculative decode to about 2.3x stock plain greedy decode, naming that baseline;
  - gives drafter-off decode's cost.
- **`docs/observability.md`** covers `projection_kernel=` and `draft_block=`. **`docs/tuning.md`** covers the current-arm resolution.
- **The README's decode figures** change only if the merge gate's numbers justify it.

## Out of scope

- Block 6; the cap stays 5.
- Other GPUs; the tile's gate decides where the kernel runs.
- The greedy drafter's head argmax.
- Other quantizations, model families and the LM backend.
- #149's tune fix.
- Upstreaming.
