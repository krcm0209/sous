# INT8 K/V Mirror Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve decode's one-row attention and the exact verifier's attention, from N8 keys up to a ceiling L, from a per-token int8 copy of the context's K/V with int8 queries quantised in the kernel, on the M5 Pro's tensor units, behind the attention tile; on by default through `[model].kv_int8`, keeping greedy output with the drafter equal to greedy output without it, and shipped only if the pre-registered merge gate passes.

**Architecture:** A new `src/sous/engine/kv8.py` holds the array-level entries (the quantiser and kernel dispatch), the per-decode serving window, the mirror attached to each full-attention layer's `KVCache` with its instance `trim` wrap, the ceiling and floor, `availability()`, `enable()`, `clear()`, the probe, `calls` and the source pins. It runs over two new Metal sources: `kernels/kv8_quant.metal` (the fused row quantiser, MPP-free) and `kernels/attention_tile_kv8.metal` (the relay tile's split body over int8 operands; the tile's reduce kernel is reused unchanged). Two seams in `tileattn` route rows: `_serve_decode` and `serve_verify`, which gains the `cache` argument `verifyattn._wrap` already holds. `VLMEngine.decode()` opens the window immediately before `stream_generate` and closes it in a `finally`; config, status, load line, turn-line gauges, the fork-epoch deny-list and `sous tune` keys make it visible, switchable and correctly keyed. The mirror is transient: nothing kv8 serves reaches a fork, a slot or the store.

**Tech Stack:** Python 3.14, mlx 0.32.3, mlx-vlm 0.7.2, mlx-lm (the lock), Metal (`mx.fast.metal_kernel`; MPP `matmul2d` tiles through `nax.h` for the split kernel, `common.h` only for the quantiser), pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-10-12-int8-kv-mirror-design.md`. Read it before starting any task. The stage-1 spike it argues from is out of tree at `~/code/sous-spikes/l4-int8kv/` (`kern_kv8q.metal`, `dq8b.py`, `results.md`).

**Branch:** `feat/int8-kv-mirror`, stacked on `feat/attention-tile-split-qk` (the relay tile, PR #181, commit 860ba80). The spec and this plan are the branch's first commits; the merge gate (Task 10) starts only once #181 is merged and the branch is rebased onto main.

**Task order:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, each on the previous one's commits. Task 6 (the contract tests) needs Task 5's engine wiring; Task 7 edits files Task 5 edited first and finds its blocks by quoted text.

## Global Constraints

**Files and names**
- Create `src/sous/engine/kv8.py`, `src/sous/engine/kernels/kv8_quant.metal`, `src/sous/engine/kernels/attention_tile_kv8.metal`, `tests/kv8_fixtures.py`, `tests/test_kv8.py`, `tests/test_kv8_contract.py`, `tests/test_kv8_model.py`.
- Kernel names: `sous_kv8_quant` (inputs `k`, `v`, `params` int32 `{rows, out_rows}`; outputs `codes` int8 `[1, 4, out_rows, 512]`, K then V, and `sc` fp16 `[1, 4, out_rows, 2]`, sK then sV; no template; grid `(256, rows, 4)`, threadgroup `(256, 1, 1)`); `sous_kv8_tile_split` (inputs `q`, `codes`, `sc`, `params`; outputs `part`, `stats` in the tile's layout; templates `TR`, `MR`, `RGT`); the debug object `sous_kv8_tile_split_qdbg` (the same body with `#define QDBG 1` in its header and a third output `qcodes`; QDBG is never a template argument, since mlx puts every output a kernel object names into its signature). Every kernel goes through `int8prefill._kernel(...)` with `compile_options={"math_mode": "safe"}` stated explicitly; the split kernel takes `nax.h` after `common.h`, the quantiser `common.h` only.
- `kv8` imports `tileattn` at module level; `tileattn` imports `kv8` function-locally inside the two seams only. mlx, mlx_lm and mlx_vlm imports are function-local everywhere in `src/`; mlx-vlm internals are resolved with `importlib`.

**Frozen constants and arithmetic** (they change bits or the serving rule)
- `N8 = 1536` (= `tileattn.N0`), `GROWTH = 4096`, `REL_RMS_BOUND = 1e-2`, `STOCK_RMS_BOUND = 3e-2`, `MIRROR_ATTR = "_sous_kv8"`, `BYTES_PER_ROW_PER_HEAD = 516`.
- Quantiser: per (head, row) `s = fp16(min(amax / 127, 65504))`, codes `rint(x / s)` clamped to ±127 with round half to even (Metal `rint`, `mx.round`), a scale that rounds to 0 divides by 1 instead, so an all-zero or underflowing row gives codes 0; K, V and Q alike (Q with one fp32 scale per (query head, row), quantised inside the kernel; `qf = sq > 0 ? sq : 1` while the score still uses `sq`).
- Score association `float(Si) * (sqr[i] * (SCALE_LOG2 * sK))`; `ct_p = p * sV` with the row sum on the unscaled p; every read clamped to n − 1; the 16-byte scale load only when `kb0 + 32 <= n`; three barriers per block (the Q row maxima, then the relay's two); partials in the tile's layout.
- Ceiling: `ceiling(window, splits)` is the largest `b` in `tileattn.boundaries(window, splits)` with `C = round_up(b, 256)` satisfying `1.504·C + 0.504·C/16 + 272 ≤ window`, computed in integers as `1504*C + 504*(C // 16) + 272_000 <= 1000*window`, else 0. At S = 20: 84,481 at 131072, 170,241 at 262144 (the spec's exact figures; never "172K").
- Mirror capacity `cap = min(round_up(offset + GROWTH, 256), round_up(L, 256))`, without the second term in a forced window; a mirror never holds a row at or past L; growth layer by layer with each new buffer evaluated before the old is dropped; slice updates in place.

**Module state and the serving rule**
- `_active: int` (1 only after every gate and the probe; the first statement of every `enable()` clears it; `clear()` clears it; the probe raises it only while it runs), `_ceiling: int`, `_window: Window | None`, `_serial: int` (never reset).
- `Window(serial, state, rows, served, built, force)` with states `idle`, `served`, `capped`; `Mirror(codes, sc, cap, n, serial, original_trim)`.
- `serving_now()` is `_active and _window is not None and _window.state != "capped"`; `serves(n_row)` is `serving_now() and N8 <= n_row and (n_row < _ceiling or _window.force)`. Both seams look `serves`, `decode` and `observe_first_row` up through the module on every call. A decode call's `n_row` is `n_keys`; a verify run's is its first row's key count, `prefix + j + 1`. A call whose first row is at or past L caps the window (sticky for the decode), drops the mirrors and runs on the tile; a straddling call keeps its mirror and serves only its rows below L.
- `open_window(*, force=False)` asserts no window is open; `close_window(cache) -> (state, rows)` drops every mirror and wrap, tolerates caches without one and a window that was never opened (it runs in a `finally`), and calls `mx.clear_cache()` only when a mirror was built.
- The trim wrap is an instance attribute `trim` carrying `_TRIM_MARK`, holding the cache weakly, lowering the mirror's n to the cache's offset after the call; only a marked wrap is ever restored; a mirror whose serial is stale is rebuilt from row 0.
- `calls` keys exactly: `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_rows`, `mirror_trims`, `mirror_grown`, `windows`, `windows_capped`.

**Gates, refusals, pins** (`enable(model, *, enabled, tile_status, verify_status, window) -> dict` with keys `state`, `reason`, `probe_seconds`, `floor`, `ceiling`; never raises)
- Order: clear the state; `enabled` false gives `off`; the tile not `active` is an expected refusal (one INFO line `int8 K/V mirror unavailable: <reason>`); pin drift warns; then, timed into `probe_seconds`: `availability()` (a compile failure warns with the error line), the floor as a step boundary at the active S (`_floor_reason`, warns), the ceiling (`L <= N8` expected: `window below the kv8 floor (...)`), the probe (warns). Unexpected refusals are one `warnings.warn` starting `sous: int8 K/V mirror unavailable (` and ending `); decode and verify attention read the bf16 cache`. An exception is logged as `int8 K/V mirror: enabling failed` with its traceback.
- `_compile_ok` remembers success only. `VALIDATED_KV8_SOURCES` holds 17 sha256 digests keyed `(module, qualname)` computed on mlx-vlm 0.7.2 with `verifyattn._source_digest` (following `__wrapped__`).
- The probe runs on the `sous-model-load` thread, builds its inputs with GPU ops evaluated before the first launch, restores `calls`, the tile's flag and decode global, leaves `_active` 0 and `_window` None, and calls `mx.clear_cache()` in `finally`.

**Engine, config, visibility**
- `SousConfig.kv_int8: bool = True`, in `_KNOWN["model"]`; a non-bool warns and means false. `VLMEngine(..., kv_int8: bool = False, kv_int8_window: int = 0)`; only `default_engine_factory` passes the config's value, and `_default_factory` passes `kv_int8_window=reserve_tokens` only when a reserve was measured. `enable()` runs after the projection kernel and before draftctx, the fork store key and `measure_cache_budget`.
- `decode()` opens the window inside the draft-context `seed` block only when `kv8._active`, closes it in a `finally` with `close_window(cache)`, and passes the pair into the `Speculation` it builds after a loop that did not raise. `unload()` calls `kv8.clear()` after `projkernel.clear()`. `LMEngine.kv_int8_status` is the five-key `off` dict; `ManagedEngine.kv_int8_status` is `getattr(self._inner, "kv_int8_status", None)`.
- Load line: ` kv_int8=<state>` after `draft_block=`, plus ` kv_int8_probe_s=`, ` kv_int8_floor=`, ` kv_int8_ceiling=` when active. Status document: `engine.kv_int8 = {state, reason}`. Turn line: ` kv8=<off|idle|served|capped> kv8_rows=<n>` after `draft_context=`, from `Speculation.kv8`/`kv8_rows` through `TURN_GAUGES`, `PromptCacheStats`, `_note_speculation` and `TurnResult`, read per owner, never as a daemon-wide max.
- Fork store: `forkstore._EPOCH_EXCLUDED_KERNELS = ("attention_tile_kv8.metal", "kv8_quant.metal")` checked inside `engine_epoch()`'s kernels loop; `kv8.py` stays out of `_EPOCH_FILES`; the key gains no field. Deploying cold-starts the store once (vlm.py and tileattn.py change); say so in the PR.
- `sous tune`: `Arm.kv_int8: bool = True` mirrored from the config, last in `Arm.suite_key`; `SuiteRun.kv_int8` required, in `SuiteRun.key`, `from_dict` defaulting to False; `runner._check_kv8(arm, engine, out)` prints a notice for `unavailable` only, wording `the engine runs bf16 K/V`.

**Shared rules**
- Threads: no new threads. Every thread that touches mlx calls `release_mlx_thread_state()` before it exits, as the tree already does; the probe and warm-ups run on the load thread.
- Test hygiene: new test modules start with `mx = pytest.importorskip("mlx.core")`; shared helpers live in `tests/kv8_fixtures.py`, imported as `from tests import kv8_fixtures as kfx`; the `nax` marker skips on `kv8.availability()`, never the tile's; every engine-level test sets `kv8._ceiling` through `kfx.ceiling`, and `force` is for the probe and array-level tests only; tests never touch `~/.sous` (tmp_path-based paths everywhere); the engine constructors default every kernel off.
- Comments explain why, never what, and never cite the spec, the plan, a decision number or a task number; `# ty: ignore[rule]`, never `# type: ignore`; ruff line length 100.
- Commits: Conventional Commits, imperative lowercase subject, the why in the body, signed (`git log -1 --format='%G? %an <%ae>%n%B'` shows `G` and the maintainer's noreply identity), no attribution lines of any kind, never `--no-gpg-sign` or `--no-verify`. Nothing in a commit, a file, the PR or an issue names the maintainer or a home path: say "the maintainer".
- The M5 Pro: reached as `ssh m5pro` or through `~/code/sous-spikes/p148/gate/m5.sh m5pro-lan '<command>'`; the branch runs from its own detached worktree `~/code/personal/sous-remote-kv8` of `~/code/personal/sous-remote`, never from `~/code/personal/sous`, `~/.sous` or the production daemon on port 8383; harness daemons use port 8384 and isolated HOMEs; every GPU job runs under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py` and counts only with `daemon stayed idle during the job`. Harnesses stay out of tree (`~/code/sous-spikes/kv8-gate/`, rsynced to `~/code/personal/sous-spikes/kv8-gate/`) and are never committed.
- `docs/superpowers/**` records are never edited after the fact; the pre-registered gate rules and bars are never moved after a number is seen (a harness defect is logged with a timestamp and the step re-run).

**Files more than one task edits, in order** (every replaced block is unique in its file; later tasks find theirs by quoted text)

| File | Tasks, in order | Notes |
|---|---|---|
| `src/sous/engine/kv8.py` | 1 → 2 → 4 | Import blocks merged; bodies appended; `availability()`, `_compile_probe()`, `_compile_ok` and `ceiling()` are defined once, in Tasks 1–2. |
| `tests/kv8_fixtures.py` | 1 → 2 | Append only. |
| `tests/test_kv8.py` | 1 → 2 → 3 → 4 → 5 | Header imports merged (`contextlib` once). |
| `src/sous/engine/base.py` | 5 → 7 | No overlap. |
| `src/sous/engine/promptcache.py` | 5 → 7 | No overlap. |
| `tests/test_engine_base.py` | 5 → 7 | Task 7 appends after the last test. |
| `tileattn.py`, `verifyattn.py` | 3 only | |
| `vlm.py`, `lm.py`, `config.py` | 5 only | |
| `forkstore.py`, `docs/observability.md` | 7 only | |
| `CLAUDE.md`, `docs/configuration.md`, `docs/tuning.md`, `tune/*` | 8 only | Task 1 already updates the comments in `common.h`, `nax.h`, `kernels/__init__.py` and `nax.py`. |

## Review Focus

Three input classes the spec implies and no draft task exercised; each line's test is in the task that owns the code.

1. **A fork taken from a served cache.** `fork_copy` copies `state` and `meta_state` into a fresh `KVCache`, so a slot or store copy must carry neither the mirror nor the trim wrap, and trimming the fork must leave the served cache's mirror valid — Task 2, `test_mirror_never_travels_with_a_fork`.
2. **A model whose KV cost per token cannot be measured.** The factory then reserves nothing; kv8 must get a window of 0 (and refuse with `window below the kv8 floor`) rather than a ceiling sized against a reserve that does not exist — Task 5, `test_default_factory_passes_no_window_when_the_reserve_cannot_be_measured`.
3. **A rollback larger than the cache, or of zero rows.** DFlash's abort can trim more than was appended; `KVCache.trim` clamps and the wrap reads the offset afterwards, so n must land on the new offset and never below it — Task 2, `test_mirror_trim_past_the_offset_lowers_n_to_the_new_offset`.

---

### Task 1: The kv8 kernels and their array-level entries

**Files:**
- Create: `src/sous/engine/kernels/kv8_quant.metal`
- Create: `src/sous/engine/kernels/attention_tile_kv8.metal`
- Create: `src/sous/engine/kv8.py` (its first part; later tasks append below it and leave its docstring as written here)
- Modify: `src/sous/engine/kernels/common.h:1-2` (the first comment: a fourth module prepends it)
- Modify: `src/sous/engine/kernels/nax.h:1-4` (the first comment: a third tensor-op kernel prepends it)
- Modify: `src/sous/engine/kernels/__init__.py:1-24` (the package docstring names the new files)
- Modify: `src/sous/engine/nax.py:3-6` (the module docstring names the kernels the rule gates)
- Create: `tests/kv8_fixtures.py` (Task 2 adds the `window()`, `small_floor()` and `ceiling()` fixtures)
- Test: `tests/test_kv8.py` (create; later tasks append)

**Interfaces:**
- Consumes (all already in the tree):
  - int8prefill: `_kernel(name, input_names, output_names, source, header, *, ensure_row_contiguous=True, compile_options=None)` (cached by name in `int8prefill._kernels`), `_kernel_text(name)`, `_error_line(text)`.
  - nax: `Availability(available: bool, reason: str | None = None)`, `platform_reason() -> str | None`.
  - tileattn: `HQ = 24`, `HKV = 4`, `GQA = 6`, `D = 256`, `SCALE = 0.0625`, `N0 = 1536`, `GRAN = 32`, `MAX_RUN = 8`, `MEASURED_SPLITS`, `chunk_for(n, splits)`, `verify_plan(prefix, t, splits)`, `boundaries(n_max, splits)`, `far_mark(splits)`, `_kernel_pair()` (`functools.cache`d; `[1]` is the reduce kernel `sous_attention_tile_reduce`, inputs `part`, `stats`, `params`, template `MR`).
  - tests/tile_fixtures.py: `serving_qkv(n, t, *, seed=0)` and `_tree_sum(x, axis)`.
  - The stage-1 spike, read only: `~/code/sous-spikes/l4-int8kv/kern_kv8q.metal` (the measured `i8q` kernel, host-quantised Q, separate K and V buffers) and `dq8b.py`'s `SRC` (the fused quantiser).
- Produces:
  - `kernels/kv8_quant.metal`, kernel `sous_kv8_quant`: inputs `k`, `v` (bf16 `[1, 4, rows, 256]` views read through their strides), `params` int32 `{rows, out_rows}`; outputs `codes` int8 `[1, 4, out_rows, 512]` (K codes at +0, V codes at +256 per row) and `sc` fp16 `[1, 4, out_rows, 2]` = `{sK, sV}`; grid `(256, rows, 4)`, threadgroup `(256, 1, 1)`; header `common.h` only; `ensure_row_contiguous=False`, `math_mode: safe`.
  - `kernels/attention_tile_kv8.metal`, kernel `sous_kv8_tile_split`: inputs `q`, `codes`, `sc`, `params` int32 `{n, chunk, nsplit}`; outputs `part`, `stats` in the tile's layout; templates `TR`, `MR`, `RGT`; grid `(64 * RGT, nsplit, 4)`, threadgroup `(64 * RGT, 1, 1)`; header `common.h` + `nax.h`; `ensure_row_contiguous=False`, `math_mode: safe`. The debug kernel object `sous_kv8_tile_split_qdbg` compiles the same body with `#define QDBG 1` appended to the header and a third output `qcodes` int8 `[1, 24, TR, 256]`.
  - In `kv8`:
    - Constants: `N8: int = 1536`, `GROWTH = 4096`, `REL_RMS_BOUND = 1e-2`, `STOCK_RMS_BOUND = 3e-2`, `MIRROR_ATTR = "_sous_kv8"`, `BYTES_PER_ROW_PER_HEAD = 516`; `logger = logging.getLogger("sous.engine.kv8")`.
    - Array-level entries: `quantize(k, v, out_rows) -> (codes, sc)` (raises `ValueError` unless `1 <= k.shape[2] <= out_rows`); `dequant(codes, sc) -> (K, V)` fp32 `[1, 4, R, 256]`; `tile8(queries, codes, sc, n, chunk) -> [1, T, 24, 256]` bf16, one split + reduce over `T <= MAX_RUN` rows, the stand-in's swap target, looked up through the module by every caller; `decode_codes(queries, codes, sc, n, splits) -> [1, 24, 1, 256]` (raises `ValueError` below N0); `verify_codes(queries, codes, sc, n, splits, runs=None) -> [1, 24, T, 256]` (raises `ValueError` if any run has chunk 0).
    - Availability: `availability() -> nax.Availability` (the platform rule, then `_compile_probe()`; reason `the int8 K/V kernels failed: <line>`), `_compile_probe() -> str | None`, `_compile_ok: bool`.
    - Private, for later tasks and tests: `_SAFE_MATH`, `_QDBG = "#define QDBG 1\n"`, `_kernels() -> (quant, split, reduce)` (`functools.cache`d; `reduce is tileattn._kernel_pair()[1]`), `_qdbg_split()` (`functools.cache`d), `_launch(split, queries, codes, sc, n, chunk, *, qcodes=False) -> list`, `_tile8_qcodes(queries, codes, sc, n, chunk) -> (out, qcodes)`.
    - Not here (later tasks add them): `_active`, `_ceiling`, `_window`, `_serial`, `calls`, `Window`, `Mirror`, every cache-level function, `enable()`, `clear()`, `probe()` and `VALIDATED_KV8_SOURCES`.
  - In `tests/kv8_fixtures.py` (imported as `from tests import kv8_fixtures as kfx`): `KV8 = kv8.availability()` (read once at import), `nax` (the skip marker), `SCALE_LOG2`, `reference_quantize(k, v) -> (codes, sc)`, `reference_quantize_q(queries) -> (codes, sq)`, `reference_attention(queries, codes, sc, n) -> [1, 24, T, 256]` fp32, `stand_in_tile8(queries, codes, sc, n, chunk)` (`kv8.tile8`'s contract), `tie_rows(rows, *, heads=4, seed=0)`, `serving_codes(n, t, *, seed=0) -> (queries, codes, sc, keys, values)`.
  - In `tests/test_kv8.py`, helpers later tasks reuse: `_bf16(shape, seed, scale=1.0)`, `_relative_rms(got, want)`, `_poisoned_past(codes, sc, n)`, `_assert_rows_equal_decode(splits, mark, ts)`, the `fresh_kernels` and `kernel8` fixtures.

The split kernel is the in-tree relay tile's split body with int8 operands, as the stage-1 spike measured it (`kern_kv8q.metal`, variant `i8q`: 1.69x the relay tile at T = 5 and 57K keys), with the two changes the spike did not time: Q is quantised in the kernel's prologue rather than by about nine mlx ops on the host, and K and V are two strided views of one packed `codes` buffer rather than two buffers. Everything else is the tile's: the grid and threadgroup, the 32·S chunk schedule, the online softmax, the fixed-order relay with its two barriers per block (here over exact int32 partials), resident accumulators, the clamp of every read to n − 1, the zero-probability mask and the partial/stats layout, so `tileattn._kernel_pair()[1]` reduces kv8's partials unchanged and a row's output depends on its chunk and nothing else about the call. The quantiser is the spike's fused kernel reading strided views, with the scale saturated at fp16's largest finite value.

The spec calls the debug instantiation a template flag. mlx declares every output a kernel object names in the kernel's signature, and a store to an undeclared name does not compile even in a branch a template argument switches off (checked on mlx 0.32.3: `if (FLAG) dbg[i] = x[i];` with no `dbg` output fails to build at `FLAG = 0`). So the debug store sits behind `#ifdef QDBG`, and the debug instantiation is a kernel object of its own whose header appends `#define QDBG 1` and whose outputs add `qcodes`; the production object is byte-for-byte the body without it.

When this plan was written, two parts were run on the maintainer's M2 (macOS 15.5, mlx 0.32.3), which has no tensor units: the quantiser, bitwise equal to `reference_quantize` on random rows, strided views, tie rows and zero, underflowing, subnormal-scale and saturating rows; and the Q prologue cut out of the split body with a local `nax_coord` (it uses no tensor ops), its staged codes bitwise equal to `reference_quantize_q` at T = 1..8 with a zero row. The MPP half compiles only on the M5 Pro: Step 11 is where the whole split kernel first runs.

- [ ] **Step 1: Create the shared test helpers**

Create `tests/kv8_fixtures.py`. `stand_in_tile8` is the plain-mlx model of `kv8.tile8`'s contract that this task's entry tests and the later dispatch and contract tests swap in: `tile_fixtures.stand_in_tile`'s partition over the codes, the kernel's score association, base-2 probabilities. Its scores are exact up to that association because an int8 dot product of 256 terms stays below 2**24 in fp32 whatever the order.

```python
"""Shared helpers for kv8's tests: a plain-mlx stand-in for the int8 tile kernel, the
reference quantisers and attention, tie rows, serving-layout inputs and the `nax`
skip marker.

mlx is imported inside each helper, so importing this module needs only what
`kv8.availability()` itself imports."""

from typing import Any

import pytest

from sous.engine import kv8, tileattn
from tests import tile_fixtures as tfx

# Read once: a failed compile is not remembered, so each call would compile again.
KV8 = kv8.availability()
nax = pytest.mark.skipif(not KV8.available, reason=f"kv8 kernels unavailable: {KV8.reason}")

# The kernel's fp32 SCALE_LOG2: 1/16 is a power of two, so the product rounds to fp32
# exactly as log2(e) alone does, whichever side rounds first.
SCALE_LOG2 = 0.0625 * 1.4426950408889634


def _quantize_rows(x: Any, *, fp16_scale: bool) -> tuple[Any, Any]:
    """Per-row symmetric int8 over the last axis: (codes int8, scale). The scale is
    amax / 127, rounded to fp16 and saturated at its largest finite value for K and
    V, kept in fp32 for Q; a zero scale divides by 1."""
    import mlx.core as mx

    x32 = x.astype(mx.float32)
    amax = mx.max(mx.abs(x32), axis=-1)
    scale = amax / 127.0
    if fp16_scale:
        scale = mx.minimum(scale, 65504.0).astype(mx.float16)
    f = mx.where(scale > 0, scale.astype(mx.float32), 1.0)
    codes = mx.clip(mx.round(x32 / f[..., None]), -127, 127).astype(mx.int8)
    return codes, scale


def reference_quantize(k: Any, v: Any) -> tuple[Any, Any]:
    """kv8.quantize's contract in mlx ops on the GPU: (codes int8 [1, 4, R, 512], sc
    fp16 [1, 4, R, 2]). mx.round is round-half-to-even, Metal's rint."""
    import mlx.core as mx

    k_codes, k_scale = _quantize_rows(k, fp16_scale=True)
    v_codes, v_scale = _quantize_rows(v, fp16_scale=True)
    return mx.concatenate([k_codes, v_codes], axis=-1), mx.stack([k_scale, v_scale], axis=-1)


def reference_quantize_q(queries: Any) -> tuple[Any, Any]:
    """The kernel's in-register Q quantisation in mlx ops: (codes int8 [1, 24, T, 256],
    sq fp32 [1, 24, T])."""
    return _quantize_rows(queries, fp16_scale=False)


def reference_attention(queries: Any, codes: Any, sc: Any, n: int) -> Any:
    """fp32 causal attention over what the codes mean, on the GPU: the queries as the
    kernel quantises them (codes * sq), K and V dequantised from the mirror's first n
    rows, row t of T over keys [0, n - T + t]. [1, 24, T, 256] float32."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    q_codes, sq = reference_quantize_q(queries)
    qd = (q_codes.astype(mx.float32) * sq[..., None]).reshape(1, hkv, gqa, t, d)
    k, v = kv8.dequant(codes[:, :, :n], sc[:, :, :n])
    scores = (qd @ k[:, :, None].swapaxes(-1, -2)) * tileattn.SCALE  # [1, 4, 6, T, n]
    limit = (n - t + mx.arange(t) + 1)[:, None]
    scores = mx.where(mx.arange(n)[None, :] < limit, scores, -float("inf"))
    p = mx.softmax(scores, axis=-1, precise=True)
    return (p @ v[:, :, None]).reshape(1, tileattn.HQ, t, d)


def stand_in_tile8(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> Any:
    """A plain-mlx model of kv8.tile8's contract, tile_fixtures.stand_in_tile's
    partition over the codes. The queries are quantised as the kernel quantises them;
    a (row, key) score is the integer dot of the codes, exact in fp32 in any order
    since every partial sum stays below 2**24, times sq * (SCALE_LOG2 * sK) in that
    order; probabilities are base 2, and PV takes p * sV over the V codes while the
    row sum takes p. Keys are cut into splits at multiples of `chunk`, reads clamped
    to n - 1 and masked past the row's limit, and the splits combined in split order,
    so a row's output depends on the chunk and nothing else about the call. Returns
    [1, T, 24, 256] bf16."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    nsplit = -(-n // chunk)
    prefix = n - t
    q_codes, sq = reference_quantize_q(queries)
    live = codes[:, :, :n].astype(mx.float32)
    k_codes, v_codes = live[..., :d], live[..., d:]
    scales = sc[:, :, :n].astype(mx.float32)
    neg = mx.array(-float("inf"), dtype=mx.float32)
    rows = []
    for r in range(t):
        limit = prefix + r + 1
        # [1, 4, 6, 1, 256] and [1, 4, 6, 1]: the row's six query heads per KV head
        q = q_codes[:, :, r, :].astype(mx.float32).reshape(1, hkv, gqa, 1, d)
        s_q = sq[:, :, r].reshape(1, hkv, gqa, 1)
        maxes, sums, partials = [], [], []
        for s in range(nsplit):
            pos = s * chunk + mx.arange(chunk)
            idx = mx.minimum(pos, n - 1)
            k = mx.take(k_codes, idx, axis=2)[:, :, None]  # [1, 4, 1, c, 256]
            v = mx.take(v_codes, idx, axis=2)[:, :, None]
            s_k = mx.take(scales[..., 0], idx, axis=2)[:, :, None]  # [1, 4, 1, c]
            s_v = mx.take(scales[..., 1], idx, axis=2)[:, :, None]
            score = mx.sum(q * k, axis=-1) * (s_q * (SCALE_LOG2 * s_k))  # [1, 4, 6, c]
            score = mx.where(pos < limit, score, neg)
            m = mx.max(score, axis=-1, keepdims=True)  # a max is order-free
            ref = mx.where(m == neg, mx.zeros_like(m), m)
            p = mx.power(2.0, score - ref)
            maxes.append(m)
            sums.append(tfx._tree_sum(p, axis=-1)[..., None])
            # Rounded through bf16, as tile_fixtures' stand-in rounds its partials: in
            # fp32 a partition change flips too few output bits for a sensitivity test.
            partial = tfx._tree_sum((p * s_v)[..., None] * v, axis=-2)  # [1, 4, 6, 256]
            partials.append(partial.astype(mx.bfloat16).astype(mx.float32))
        gmax = maxes[0]
        for m in maxes[1:]:
            gmax = mx.maximum(gmax, m)
        num = mx.zeros_like(partials[0])
        den = mx.zeros_like(sums[0])
        for split_max, split_sum, split_out in zip(maxes, sums, partials, strict=True):
            w = mx.power(2.0, split_max - gmax)
            num = num + w * split_out
            den = den + w * split_sum
        rows.append((num / den).reshape(1, 1, tileattn.HQ, d))
    return mx.concatenate(rows, axis=1).astype(mx.bfloat16)


def tie_rows(rows: int, *, heads: int = tileattn.HKV, seed: int = 0) -> Any:
    """bf16 [1, heads, rows, 256] whose every element but dim 0 divides by its row's
    scale onto a tie. Dim 0 is +-127 * 2**e, so amax / 127 is the power of two 2**e
    (exact in fp16 and fp32); every other dim is (k + 1/2) * 2**e for an integer k in
    [-127, 126], so x / scale is k + 1/2 exactly and only the rounding rule decides
    the code. e cycles through -8, -4, 0 and 3 by row. Evaluated."""
    import mlx.core as mx

    k_key, s_key = mx.random.split(mx.random.key(seed))
    shape = (1, heads, rows, tileattn.D)
    k = mx.random.randint(-127, 127, shape, key=k_key).astype(mx.float32)
    sign = mx.where(mx.random.bernoulli(0.5, (1, heads, rows, 1), key=s_key), 1.0, -1.0)
    scale = mx.array([2.0 ** (-8, -4, 0, 3)[r % 4] for r in range(rows)]).reshape(1, 1, rows, 1)
    x = mx.concatenate([127 * sign * scale, (k[..., 1:] + 0.5) * scale], axis=-1)
    out = x.astype(mx.bfloat16)
    mx.eval(out)
    return out


def serving_codes(n: int, t: int, *, seed: int = 0) -> tuple[Any, Any, Any, Any, Any]:
    """(queries, codes, sc, keys, values) as kv8 serves them: tile_fixtures.serving_qkv's
    transposed query rows and bf16 [..., :n, :] K/V views of a 256-step padded buffer,
    and their mirror as a first build leaves it, kv8.quantize over the n live rows
    into a capacity of the next 256-row step past n, the rows past n unwritten.
    Evaluated."""
    import mlx.core as mx

    queries, keys, values = tfx.serving_qkv(n, t, seed=seed)
    cap = -(-(n + 1) // 256) * 256
    codes, sc = kv8.quantize(keys, values, cap)
    mx.eval(codes, sc)
    return queries, codes, sc, keys, values
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_kv8.py`. The quantiser, kernel-source, kernel-object, stand-in and availability tests run on any Metal GPU; the `kfx.nax` tests and the `real` parameter of the `kernel8` fixture run the Metal split kernel and skip wherever `kv8.availability()` fails.

```python
"""kv8 (sous.engine.kv8): int8 K/V and Q through the attention tile.

The quantiser needs no tensor units, so its tests run on any Metal GPU, CI's
macos-15 included. The int8 split kernel does: its entries run everywhere with a
plain-mlx stand-in, and the `nax`-marked tests run the real kernel where kv8's own
availability() passes.
"""

from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, kv8, nax, tileattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402

N0 = tileattn.N0
HQ, HKV, D = tileattn.HQ, tileattn.HKV, tileattn.D


def _bf16(shape, seed, scale=1.0):
    x = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _assert_quantizes_like_the_reference(k, v, out_rows=None):
    rows = k.shape[2]
    codes, sc = kv8.quantize(k, v, out_rows or rows)
    want_codes, want_sc = kfx.reference_quantize(k, v)
    mx.eval(codes, sc, want_codes, want_sc)
    assert codes.dtype == mx.int8 and sc.dtype == mx.float16
    assert mx.array_equal(codes[:, :, :rows], want_codes).item()
    assert mx.array_equal(sc[:, :, :rows], want_sc).item()
    return codes[:, :, :rows], sc[:, :, :rows]


# ---- the quantiser ----------------------------------------------------------------


@pytest.mark.parametrize(("rows", "scale"), [(1, 1.0), (5, 0.03), (37, 4.0), (300, 1.0)])
def test_the_quantiser_matches_the_reference_on_random_rows(rows, scale):
    _assert_quantizes_like_the_reference(
        _bf16((1, HKV, rows, D), rows, scale), _bf16((1, HKV, rows, D), 100 + rows, 2 * scale)
    )


def test_the_quantiser_reads_views_of_a_padded_buffer_in_place():
    """The cache hands over [..., a:b, :] slices of a buffer padded in 256-row steps."""
    keys = _bf16((1, HKV, 768, D), 1)
    values = _bf16((1, HKV, 768, D), 2)
    view_k, view_v = keys[:, :, 300:437], values[:, :, 300:437]
    _assert_quantizes_like_the_reference(view_k, view_v, out_rows=512)


def test_tie_rows_round_half_to_even():
    """Every element but each row's first divides by the scale onto k + 1/2 exactly,
    so the codes are decided by the rounding rule alone."""
    x = kfx.tie_rows(8)
    scale = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True) / 127
    frac = (x.astype(mx.float32) / scale)[..., 1:] % 1
    assert mx.all(frac == 0.5).item()
    codes, _ = _assert_quantizes_like_the_reference(x, kfx.tie_rows(8, seed=1))
    ties = codes[..., 1:D].astype(mx.float32)
    want = mx.round((x.astype(mx.float32) / scale)[..., 1:])
    assert mx.array_equal(ties, want).item() and mx.all(ties % 2 == 0).item()


def test_mx_round_is_round_half_to_even_on_both_streams():
    """The reference quantiser's rounding is Metal's rint only while mx.round rounds
    half to even: a change in mlx fails here, not in a tie-row comparison."""
    ties = mx.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 126.5])
    want = [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 126.0]
    assert mx.round(ties, stream=mx.gpu).tolist() == want
    assert mx.round(ties, stream=mx.cpu).tolist() == want


_SPECIAL_ROWS = {
    # amax / 127 is 0: the row divides by 1 and every code is 0
    "zero": lambda shape, seed: mx.zeros(shape, dtype=mx.bfloat16),
    # amax / 127 rounds to 0 in fp16 (below 2**-25)
    "underflow": lambda shape, seed: _bf16(shape, seed, 1e-7),
    # amax / 127 is an fp16 subnormal
    "subnormal scale": lambda shape, seed: _bf16(shape, seed, 1e-4),
    # amax / 127 passes fp16's largest finite value: the scale saturates at 65504
    "saturating": lambda shape, seed: _bf16(shape, seed, 1e8),
}


@pytest.mark.parametrize("kind", list(_SPECIAL_ROWS))
def test_the_quantiser_matches_the_reference_on_special_rows(kind):
    shape = (1, HKV, 3, D)
    k = _SPECIAL_ROWS[kind](shape, 7)
    v = _bf16(shape, 8)
    codes, sc = _assert_quantizes_like_the_reference(k, v)
    if kind in ("zero", "underflow"):
        assert not mx.any(codes[..., :D]).item() and not mx.any(sc[..., 0]).item()
    if kind == "saturating":
        assert mx.all(sc[..., 0] == 65504).item()
        assert mx.all(mx.isfinite(sc)).item()


def test_bulk_equals_incremental():
    """A first build over a whole prefix and the same rows quantised in two later
    pieces give the same bits: each row depends on itself alone."""
    keys, values = _bf16((1, HKV, 256, D), 3), _bf16((1, HKV, 256, D), 4)
    bulk_codes, bulk_sc = kv8.quantize(keys[:, :, :50], values[:, :, :50], 256)
    first = kv8.quantize(keys[:, :, :17], values[:, :, :17], 17)
    second = kv8.quantize(keys[:, :, 17:50], values[:, :, 17:50], 33)
    mx.eval(bulk_codes, bulk_sc, first, second)
    pieces = mx.concatenate([first[0], second[0]], axis=2)
    assert mx.array_equal(bulk_codes[:, :, :50], pieces).item()
    assert mx.array_equal(bulk_sc[:, :, :50], mx.concatenate([first[1], second[1]], axis=2)).item()


@pytest.mark.parametrize(("rows", "out_rows"), [(5, 4), (0, 4)])
def test_quantize_refuses_rows_outside_one_to_out_rows(rows, out_rows):
    k = mx.zeros((1, HKV, rows, D), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="kv8 quantises"):
        kv8.quantize(k, k, out_rows)


def test_dequant_is_each_code_times_its_rows_scale():
    k, v = _bf16((1, HKV, 6, D), 5), _bf16((1, HKV, 6, D), 6, 3.0)
    codes, sc = kv8.quantize(k, v, 6)
    dk, dv = kv8.dequant(codes, sc)
    assert dk.shape == dv.shape == (1, HKV, 6, D) and dk.dtype == mx.float32
    want_k = codes[..., :D].astype(mx.float32) * sc[..., :1].astype(mx.float32)
    assert mx.array_equal(dk, want_k).item()
    # the format's own error on Gaussian rows: about 1 % relative RMS
    error = mx.sqrt(mx.mean((dv - v.astype(mx.float32)) ** 2) / mx.mean(v.astype(mx.float32) ** 2))
    assert error.item() < 0.02


# ---- kernel source and the kernel objects ----------------------------------------


def _text(name: str) -> str:
    return files("sous.engine.kernels").joinpath(name).read_text()


def test_the_quantiser_source_is_mpp_free():
    text = _text("kv8_quant.metal")
    assert "#include" not in text and "mpp::" not in text
    assert "rint(" in text and "65504.0f" in text and "k_strides" in text


def test_the_split_kernel_source_is_the_relay_over_int8_operands():
    text = _text("attention_tile_kv8.metal")
    assert "#include" not in text and "// REDUCE" not in text
    assert "nax_coord(lane)" in text and "codes_strides" in text and "sc_strides" in text
    # MSL 4.1 gives locals a generic address space: every local cooperative-tensor type
    # is stripped of it, or macOS 27's compiler rejects the operands.
    assert text.count("metal::remove_addrspace_t<decltype(") == 4
    assert text.count("<int8_t, int8_t, int32_t>()") == 2
    assert text.count("<PE, int8_t, float>()") == 2
    # One barrier for Q's row maxima, then the relay's two per block.
    assert text.count("threadgroup_barrier(") == 3
    # qcodes exists only in the debug kernel object: no code outside the QDBG block
    # may name it, or the production object would not compile.
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    before, rest = code.split("#ifdef QDBG")
    inside, after = rest.split("#endif", 1)
    assert "qcodes" in inside and "qcodes" not in before + after


@pytest.fixture
def fresh_kernels():
    """kv8's and the tile's kernel objects built afresh inside the test, and forgotten
    after it, so the recording kernels a test builds never serve a later one."""
    caches = (kv8._kernels, kv8._qdbg_split, tileattn._kernel_pair)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def _recording(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs

        def launch(**call):
            return [
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(call["output_shapes"], call["output_dtypes"], strict=True)
            ]

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    return made


def test_the_kernels_are_built_with_safe_math_and_read_strides(monkeypatch, fresh_kernels):
    made = _recording(monkeypatch)
    quant, split, reduce = kv8._kernels()
    assert reduce is tileattn._kernel_pair()[1]
    q, s = made["sous_kv8_quant"], made["sous_kv8_tile_split"]
    assert (q["input_names"], q["output_names"]) == (["k", "v", "params"], ["codes", "sc"])
    assert (s["input_names"], s["output_names"]) == (
        ["q", "codes", "sc", "params"],
        ["part", "stats"],
    )
    for kwargs in (q, s):
        assert kwargs["ensure_row_contiguous"] is False
        assert kwargs["compile_options"] == {"math_mode": "safe"}
    assert q["header"] == _text("common.h")
    assert s["header"] == _text("common.h") + _text("nax.h")
    assert "#define QDBG" not in s["header"]


def test_the_debug_kernel_only_adds_the_qcodes_output(monkeypatch, fresh_kernels):
    made = _recording(monkeypatch)
    kv8._kernels()
    kv8._qdbg_split()
    split, debug = made["sous_kv8_tile_split"], made["sous_kv8_tile_split_qdbg"]
    assert debug["output_names"] == [*split["output_names"], "qcodes"]
    assert debug["input_names"] == split["input_names"] and debug["source"] == split["source"]
    assert debug["header"] == split["header"] + "#define QDBG 1\n"
    assert debug["compile_options"] == split["compile_options"]


def test_launches_after_the_first_do_no_kernel_text_work(monkeypatch, fresh_kernels):
    """tile8() runs once per full-attention layer per forward: the kernel sources are
    read once per process and the kernels built once, never per launch."""
    made = _recording(monkeypatch)
    read = []
    source = kv8._kernel_text

    def counted(name):
        read.append(name)
        return source(name)

    monkeypatch.setattr(kv8, "_kernel_text", counted)
    queries, codes, sc, _, _ = kfx.serving_codes(2000, 3)
    for _ in range(3):
        assert kv8.tile8(queries, codes, sc, 2000, 96).shape == (1, 3, HQ, D)
    assert set(made) == {
        "sous_kv8_quant",
        "sous_kv8_tile_split",
        "sous_attention_tile_split",
        "sous_attention_tile_reduce",
    }
    assert read == ["common.h", "kv8_quant.metal", "attention_tile_kv8.metal", "nax.h"]


# ---- the entries, with the stand-in (and the real kernel where it compiles) ------


@pytest.fixture(params=[pytest.param("stand-in"), pytest.param("real", marks=kfx.nax)])
def kernel8(request, monkeypatch):
    """The plain-mlx stand-in everywhere; the Metal kernel where kv8 is available."""
    if request.param == "stand-in":
        monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    return request.param


def test_the_entries_return_the_models_layouts(kernel8):
    q, codes, sc, _, _ = kfx.serving_codes(2000, 1)
    assert kv8.tile8(q, codes, sc, 2000, 96).shape == (1, 1, HQ, D)
    assert kv8.decode_codes(q, codes, sc, 2000, 20).shape == (1, HQ, 1, D)
    for n, t in ((2000, 5), (N0 + 7, 5), (2000, 12)):
        q, codes, sc, _, _ = kfx.serving_codes(n, t)
        out = kv8.verify_codes(q, codes, sc, n, 20)
        assert out.shape == (1, HQ, t, D) and out.dtype == mx.bfloat16


def test_decode_codes_refuses_fewer_than_n0_keys():
    q, codes, sc, _, _ = kfx.serving_codes(N0 - 1, 1)
    with pytest.raises(ValueError, match="kv8 serves from N0"):
        kv8.decode_codes(q, codes, sc, N0 - 1, 20)


def test_verify_codes_refuses_a_run_below_n0():
    """N0 + 2 keys over 5 rows: the first two rows see 1534 and 1535 keys."""
    q, codes, sc, _, _ = kfx.serving_codes(N0 + 2, 5)
    with pytest.raises(ValueError, match="got a row at 1534"):
        kv8.verify_codes(q, codes, sc, N0 + 2, 20)


def test_verify_codes_takes_the_callers_plan(monkeypatch):
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    n, t = 1930, 12
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    runs = tileattn.verify_plan(n - t, t, 20)
    assert len(runs) == 3  # cut at 8 rows and at the 1921 step boundary
    planned = kv8.verify_codes(q, codes, sc, n, 20, runs)
    assert mx.array_equal(planned, kv8.verify_codes(q, codes, sc, n, 20)).item()


@pytest.mark.parametrize(
    ("n", "t", "chunk"), [(2000, 3, 96), (2000, 8, 96), (4096, 8, 224), (1543, 5, 64)]
)
def test_a_stand_in_row_is_the_one_row_call_at_its_own_length(n, t, chunk):
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    out = kfx.stand_in_tile8(q, codes, sc, n, chunk)
    assert out.shape == (1, t, HQ, D) and out.dtype == mx.bfloat16
    for r in range(t):
        m = n - t + r + 1
        one = kfx.stand_in_tile8(q[:, :, r : r + 1], codes, sc, m, chunk)
        assert mx.array_equal(out[:, r], one[:, 0]).item(), r


@pytest.mark.parametrize(("n", "chunk"), [(2000, 96), (4096, 224), (1537, 64)])
def test_a_stand_in_row_moves_with_its_chunk(n, chunk):
    """So an equality between two paths proves they used the same chunk."""
    q, codes, sc, _, _ = kfx.serving_codes(n, 1)
    same_step = kfx.stand_in_tile8(q, codes, sc, n, chunk)
    next_step = kfx.stand_in_tile8(q, codes, sc, n, chunk + tileattn.GRAN)
    assert not mx.array_equal(same_step, next_step).item()


@pytest.mark.parametrize(("n", "t"), [(2000, 1), (2000, 5), (4096, 8)])
def test_the_stand_in_is_the_kernels_arithmetic(n, t):
    """The stand-in sits within the kernel's bound of the fp32 reference over the
    same codes, so the contract tests that run on it run on the kernel's numbers."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    got = kfx.stand_in_tile8(q, codes, sc, n, tileattn.chunk_for(n, 20))
    want = kfx.reference_attention(q, codes, sc, n)
    assert _relative_rms(got.transpose(0, 2, 1, 3), want) <= kv8.REL_RMS_BOUND


def _relative_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2))).item()


def _poisoned_past(codes, sc, n):
    """The mirror with -128 codes and NaN scales from row n to its capacity."""
    cap = codes.shape[2]
    bad_codes = mx.full((1, HKV, cap - n, 2 * D), -128, dtype=mx.int8)
    bad_sc = mx.full((1, HKV, cap - n, 2), float("nan"), dtype=mx.float16)
    poisoned = (
        mx.concatenate([codes[:, :, :n], bad_codes], axis=2),
        mx.concatenate([sc[:, :, :n], bad_sc], axis=2),
    )
    mx.eval(poisoned)
    return poisoned


def _assert_fenced(n, t):
    """-128 codes and NaN scales planted from n to the capacity, read through
    [:, :, :n] views of the capacity-sized buffers as the mirror hands them over (the
    scale plane's head stride stays a multiple of 4 rows), give the clean output."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    want = kv8.verify_codes(q, codes, sc, n, 20)
    bad_codes, bad_sc = _poisoned_past(codes, sc, n)
    assert bad_codes.shape[2] % 256 == 0 and mx.all(mx.isnan(bad_sc[:, :, n:])).item()
    got = kv8.verify_codes(q, bad_codes[:, :, :n], bad_sc[:, :, :n], n, 20)
    assert mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()


@pytest.mark.parametrize(("n", "t"), [(N0, 1), (2000, 3), (1600, 8), (1548, 12)])
def test_codes_and_scales_past_n_change_nothing(kernel8, n, t):
    _assert_fenced(n, t)


@pytest.mark.parametrize(("n", "t"), [(N0, 1), (2000, 3), (1600, 8), (1548, 12)])
def test_reads_stay_below_n(kernel8, n, t):
    """Buffers that end exactly at n (n a multiple of 4, so the scale plane's head
    stride stays 16-byte aligned) give the output of the capacity-sized ones: no read
    reaches past the last live row."""
    assert n % 4 == 0
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    want = kv8.verify_codes(q, codes, sc, n, 20)
    end_codes, end_sc = mx.contiguous(codes[:, :, :n]), mx.contiguous(sc[:, :, :n])
    mx.eval(end_codes, end_sc)
    got = kv8.verify_codes(q, end_codes, end_sc, n, 20)
    assert mx.array_equal(got, want).item()


def _parity_prefixes(mark, t):
    """The tile's three placements around a mark (rows below, straddling and from it),
    less those with a row below N0, which kv8 never serves."""
    return [p for p in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}) if p + 1 >= N0]


def _assert_rows_equal_decode(splits, mark, ts):
    """Every verify row around `mark` equals decode_codes() at its own key count. Pool
    row i is the query at position base + i, which sees base + i + 1 keys."""
    base = mark - 20
    q, codes, sc, _, _ = kfx.serving_codes(mark + 20, 40, seed=mark)
    decoded = {}

    def decode_at(pos):
        if pos not in decoded:
            i = pos - base
            decoded[pos] = kv8.decode_codes(q[:, :, i : i + 1], codes, sc, pos + 1, splits)
        return decoded[pos]

    for t in ts:
        for prefix in _parity_prefixes(mark, t):
            i = prefix - base
            out = kv8.verify_codes(q[:, :, i : i + t], codes, sc, prefix + t, splits)
            assert out.shape == (1, HQ, t, D) and out.dtype == mx.bfloat16
            for r in range(t):
                same = mx.array_equal(out[:, :, r : r + 1], decode_at(prefix + r)).item()
                assert same, (splits, mark, prefix, t, r)


@pytest.mark.slow
@pytest.mark.parametrize("splits", [10, 20])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_stand_in_verify_rows_equal_decode_rows(monkeypatch, splits, step):
    """At N0 and the first two step boundaries, T = 1..16: straddles and runs cut at
    8 rows."""
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    mark = tileattn.boundaries(4096, splits)[step]
    _assert_rows_equal_decode(splits, mark, range(1, 17))


# ---- availability and the compile probe ------------------------------------------


def test_availability_refuses_the_platform_before_compiling(monkeypatch):
    def compiled():
        raise AssertionError("the kernels compiled on a refused platform")

    monkeypatch.setattr(kv8.nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(kv8, "_compile_probe", compiled)
    assert kv8.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_availability_reports_the_compilers_error_line(monkeypatch):
    line = "program_source:12:3: error: use of undeclared identifier 'x'"
    monkeypatch.setattr(kv8.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(kv8, "_compile_probe", lambda: line)
    assert kv8.availability() == nax.Availability(False, f"the int8 K/V kernels failed: {line}")


_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
    "  x = sg / 2;\n"
)


def test_compile_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(kv8, "_compile_ok", False)
    monkeypatch.setattr(kv8, "tile8", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.kv8"):
        reason = kv8._compile_probe()
    assert reason == "program_source:12:3: error: use of undeclared identifier 'x'"
    assert "x = sg / 2;" in caplog.text


def test_compile_probe_runs_the_quantiser_and_every_row_variant(monkeypatch):
    """Only success is remembered: a failure is probed again on the next call."""
    rows, quantised = [], []
    quantize = kv8.quantize

    def counted(k, v, out_rows):
        quantised.append((k.shape[2], out_rows))
        return quantize(k, v, out_rows)

    def flaky(queries, codes, sc, n, chunk):
        rows.append(queries.shape[2])
        if len(rows) == 1:
            raise RuntimeError("transient")
        return mx.zeros((1, queries.shape[2], HQ, D), dtype=mx.bfloat16)

    monkeypatch.setattr(kv8, "_compile_ok", False)
    monkeypatch.setattr(kv8, "quantize", counted)
    monkeypatch.setattr(kv8, "tile8", flaky)
    assert kv8._compile_probe() == "transient"
    assert kv8._compile_probe() is None
    assert kv8._compile_probe() is None
    assert rows == [1, *range(1, tileattn.MAX_RUN + 1)]
    assert quantised == [(N0, N0 + 256)] * 2


# ---- the real kernel (needs the tensor units; skipped where they are absent) -----


@kfx.nax
def test_every_row_variant_compiles_and_runs(monkeypatch):
    monkeypatch.setattr(kv8, "_compile_ok", False)
    assert kv8._compile_probe() is None


@kfx.nax
@pytest.mark.parametrize("n", [N0 + 7, 2000, 57_601])
@pytest.mark.parametrize("t", [1, 5, 8])
@pytest.mark.parametrize("gain", [1, 8])
def test_real_kernel_is_within_its_bound_of_the_fp32_reference(n, t, gain):
    """The fp32 reference reads the same codes (K, V and the host's Q codes), so this
    judges the kernel, not the format; x8 queries sharpen the softmax."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t, seed=n + t)
    q = q * gain  # exact in bf16
    got = kv8.verify_codes(q, codes, sc, n, 20)
    want = kfx.reference_attention(q, codes, sc, n)
    assert mx.all(mx.isfinite(got)).item()
    assert _relative_rms(got, want) <= kv8.REL_RMS_BOUND


def _debug_queries(t):
    """[1, 24, T, 256] bf16 laid out as serving lays queries out (a transposed view of
    [1, T, 24, 256]): a random row at a scale that alternates with T, tie rows after
    it, and one all-zero query head in the last row."""
    rows = [_bf16((1, HQ, 1, D), t, 8.0 if t % 2 else 0.01)]
    if t > 1:
        rows.append(kfx.tie_rows(t - 1, heads=HQ, seed=t))
    q = mx.concatenate(rows, axis=2)
    zero = (mx.arange(HQ)[:, None] == 5) & (mx.arange(t)[None, :] == t - 1)
    q = mx.where(zero[None, :, :, None], mx.zeros_like(q), q)
    q = mx.contiguous(q.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    mx.eval(q)
    return q


@kfx.nax
@pytest.mark.parametrize("t", range(1, tileattn.MAX_RUN + 1))
def test_real_kernel_quantises_q_as_the_host_does_bit_for_bit(t):
    """The debug kernel writes out the Q codes it staged: equal to the host's, ties
    and the zero row included, and the debug store changes no output bit."""
    q = _debug_queries(t)
    _, codes, sc, _, _ = kfx.serving_codes(N0 + 8, t, seed=t)
    n = N0 + 8
    chunk = tileattn.chunk_for(n, 20)
    out, staged = kv8._tile8_qcodes(q, codes, sc, n, chunk)
    want, _ = kfx.reference_quantize_q(q)
    assert staged.shape == (1, HQ, t, D) and staged.dtype == mx.int8
    assert mx.array_equal(staged, want).item()
    assert mx.array_equal(out, kv8.tile8(q, codes, sc, n, chunk)).item()


@kfx.nax
@pytest.mark.parametrize("mark", [1536, 1921, 2561, 12161, 57601, 131201])
def test_real_verify_rows_equal_decode_rows_at_step_boundaries(mark):
    """N0, the next two step boundaries, the far mark (the first boundary at or past
    57,000 keys) and two more, T = 1..8 at the tile's three placements."""
    assert mark in tileattn.boundaries(131_201, 20)
    assert tileattn.far_mark(20) == 57_601
    _assert_rows_equal_decode(20, mark, range(1, tileattn.MAX_RUN + 1))


@kfx.nax
@pytest.mark.parametrize("t", [1, 5, 8])
def test_real_kernel_reads_nothing_past_n_at_the_far_mark(t):
    _assert_fenced(tileattn.far_mark(20), t)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_kv8.py`
Expected: FAIL at collection, importing `tests/kv8_fixtures.py`: `ImportError: cannot import name 'kv8' from 'sous.engine'`.

- [ ] **Step 4: Add the quantiser kernel**

Create `src/sous/engine/kernels/kv8_quant.metal` exactly as below. There is no `#include`: `kv8.py` prepends `common.h`.

```metal
// kv8's row quantiser (kv8.py): per-token symmetric int8 K and V for the int8 mirror the
// attention tile reads. MPP-free, prepended with common.h only, so it compiles on any
// Metal GPU and CI tests it.
//
// Grid (256, rows, 4), threadgroup (256, 1, 1): thread d of (KV head h, row r). Inputs
// k and v are bf16 [1, 4, rows, 256] views with innermost stride 1, read in place
// through their head and token strides; params = {rows, out_rows}. For K and for V
// apiece, per (row, head): amax over the 256 dims (simd_max, then the eight
// simdgroups' maxima: a max is exact in any order), the scale s = half(amax / 127)
// saturated at half's largest finite value, f = s when s > 0 and 1 otherwise (an
// all-zero or underflowing row gets codes 0 rather than 0 / 0), and
// code = clamp(rint(x / f), -127, 127), rint rounding half to even. Outputs: codes int8
// [1, 4, out_rows, 512], each row's 256 K codes then its 256 V codes, and sc fp16
// [1, 4, out_rows, 2] = {sK, sV}. Rows from `rows` up are left unwritten: a first build
// sizes both outputs to the mirror's capacity, and nothing reads past the live rows.
// A row's codes and scales depend on that row alone, so quantising a prefix in one
// call and in pieces gives the same bits.
  const int d = int(thread_position_in_threadgroup.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int h = int(threadgroup_position_in_grid.z);
  const int out_rows = params[1];
  const float kx = float(k[h * k_strides[1] + r * k_strides[2] + d]);
  const float vx = float(v[h * v_strides[1] + r * v_strides[2] + d]);

  threadgroup float red[2][8];
  const int sg = int(simdgroup_index_in_threadgroup);
  float mk = simd_max(fabs(kx));
  float mv = simd_max(fabs(vx));
  if (thread_index_in_simdgroup == 0) {
    red[0][sg] = mk;
    red[1][sg] = mv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mk = red[0][0];
  mv = red[1][0];
  UNROLL for (int i = 1; i < 8; ++i) {
    mk = max(mk, red[0][i]);
    mv = max(mv, red[1][i]);
  }

  const half sk = half(min(mk / 127.0f, 65504.0f));
  const half sv = half(min(mv / 127.0f, 65504.0f));
  const float fk = (sk > 0.0h) ? float(sk) : 1.0f;
  const float fv = (sv > 0.0h) ? float(sv) : 1.0f;
  const size_t row = size_t(h) * size_t(out_rows) + size_t(r);
  codes[row * 512 + d] = int8_t(clamp(rint(kx / fk), -127.0f, 127.0f));
  codes[row * 512 + 256 + d] = int8_t(clamp(rint(vx / fv), -127.0f, 127.0f));
  if (d == 0) {
    sc[row * 2] = sk;
    sc[row * 2 + 1] = sv;
  }
```

- [ ] **Step 5: Add the split kernel**

Create `src/sous/engine/kernels/attention_tile_kv8.metal` exactly as below. There is no `#include`: `kv8.py` prepends `common.h` and `nax.h`. Keep `sg / 2` and `sg - rg * 2` as written (the comment above them says why), and never name `qcodes` outside the `#ifdef QDBG` block, comments aside: only the debug kernel object declares it, and the source test checks this.

```metal
// kv8's split kernel (kv8.py): the attention tile's relay kernel over int8 K, V and Q,
// for qwen3_5's decode and exact verifier: 24 query / 4 KV heads, head dim 256, scale
// 1/16, fp32 probabilities. Prepended by kv8.py with common.h + nax.h. The partials keep
// the tile's layout, so attention_tile.metal's reduce body combines them unchanged.
//
// Template: TR (rows T, 1..8), MR (= 6*TR packed rows per KV head, m = t*6 + h), RGT
// (ceil(MR/16) 16-row fragment groups). Grid (64*RGT, nsplit, 4), threadgroup
// (64*RGT, 1, 1): one threadgroup per (key split, KV head), two simdgroups per 16-row
// group, each running 8 of the 16 QK k-steps and half of the 256 output dims, as the
// tile's split body does. Inputs: q bf16 [1, 24, TR, 256]; codes int8 [1, 4, cap, 512],
// each token's 256 K codes then its 256 V codes, read as two strided views of the one
// buffer (V at +256); sc fp16 [1, 4, cap, 2] = {sK, sV} per (token, KV head), its last
// two axes contiguous and its head stride a multiple of 8 halves, so a lane's four
// consecutive keys' scales are one aligned 16-byte load; params = {n, chunk, nsplit}.
// Only the head and token strides are read. Row t sees keys [0, n - TR + t].
//
// Q is quantised here, per (query head, row): sq = amax / 127 in fp32, f = sq when
// sq > 0 and 1 otherwise (a zero row gets codes 0), code = clamp(rint(q / f), -127,
// 127). QK runs on int8 x int8 -> int32 rows and the relay adds the halves' int32
// partials, which is exact (|S| <= 256 * 127^2 < 2^24). The score is then
// float(Si) * (sq * (SCALE_LOG2 * sK)), each product rounded once in that order; sV
// folds into the probability fed to PV (int8 V on <float, int8_t, float> rows) and the
// row sum takes the unscaled probability. Nothing is dequantised in the loop. Every K,
// V and scale read is clamped to n - 1: that clamp, not the mask, keeps a NaN or a
// stale row past n out of the output (0 x NaN is NaN), and keys past a row's limit get
// probability exactly 0, so a row's output depends on `chunk` and nothing else about
// the call, as in the tile.
//
// With QDBG defined (kv8's debug kernel object only, which alone declares the qcodes
// output) the staged Q codes are also written out, [1, 24, TR, 256] int8 in q's dim
// order, for the tests to hold against the host's quantisation bit for bit.
  constexpr int GQ = 6;
  constexpr int DD = 256;
  constexpr float SCALE_LOG2 = 0.0625f * 1.4426950408889634f;
  constexpr float NEG = -1.0e30f;
  typedef float PE;  // fp32 probabilities into the PV matmul

  const ushort lane = ushort(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  // A division and a multiply-subtract, as in the tile: written as sg >> 1 and sg & 1
  // the tile ran 1.4x slower at T >= 3 (57K, M5 Pro, macOS 27.0), same output.
  const int rg = sg / 2;
  const int half_ = sg - rg * 2;
  const int nn0 = half_ * 4;
  const int split = int(threadgroup_position_in_grid.y);
  const int kvh = int(threadgroup_position_in_grid.z);
  const int n = params[0];
  const int chunk = params[1];
  const int nsplit = params[2];
  const int k_begin = split * chunk;
  const int k_end = min(n, k_begin + chunk);
  const int prefix = n - TR;

  const short2 cc = nax_coord(lane);
  const int fn = cc.x;
  const int fm = cc.y;

  const device int8_t* kbase = codes + kvh * codes_strides[1] + fn * 16;
  const device int8_t* vbase = kbase + DD + nn0 * 8;
  const int cst = int(codes_strides[2]);
  const device half* sbase = sc + kvh * sc_strides[1];

  int lim[2];
  const device bfloat* qp[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = min(rg * 16 + fm + 8 * i, MR - 1);
    const int t = m / GQ;
    const int h = m - t * GQ;
    lim[i] = prefix + t + 1;
    qp[i] = q + (GQ * kvh + h) * q_strides[1] + t * q_strides[2] + fn * 16;
  }

  // Q's scale per row: a lane holds 32 of each of its two rows' 256 dims (its half's 8
  // k-steps of 4), the four lanes that share a row (xor 1 and 8) this half's 128, and
  // the two halves trade their maxima through threadgroup memory (one barrier).
  bfloat4 qv[2][8];
  float qmax[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    float a = 0.0f;
    UNROLL for (int kq = 0; kq < 8; ++kq) {
      qv[i][kq] = *reinterpret_cast<const device bfloat4*>(qp[i] + (half_ * 8 + kq) * 4);
      const float4 x = fabs(float4(qv[i][kq]));
      a = max(a, max(max(x.x, x.y), max(x.z, x.w)));
    }
    a = max(a, simd_shuffle_xor(a, ushort(1)));
    a = max(a, simd_shuffle_xor(a, ushort(8)));
    qmax[i] = a;
  }
  threadgroup float qmx[RGT * 32];
  if (fn == 0) {
    UNROLL for (int i = 0; i < 2; ++i) qmx[(rg * 2 + half_) * 16 + fm + 8 * i] = qmax[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sqr[2];
  float qf[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const float a = max(qmax[i], qmx[(rg * 2 + 1 - half_) * 16 + fm + 8 * i]);
    sqr[i] = a / 127.0f;
    qf[i] = (sqr[i] > 0.0f) ? sqr[i] : 1.0f;
  }
  // Q codes staged once in threadgroup memory as packed char4, one uint per lane per row
  // per k-step: each half stages, and later reads, its own 8 k-steps, and each lane only
  // what it wrote itself, so no barrier is needed before the first read.
  threadgroup uint qtg[RGT * 1024];
  UNROLL for (int kq = 0; kq < 8; ++kq) {
    const int kk = half_ * 8 + kq;
    UNROLL for (int i = 0; i < 2; ++i) {
      const float4 c = clamp(rint(float4(qv[i][kq]) / qf[i]), -127.0f, 127.0f);
      const uint w = as_type<uint>(char4(c));
      qtg[((rg * 16 + kk) * 2 + i) * 32 + lane] = w;
#ifdef QDBG
      const int m = rg * 16 + fm + 8 * i;
      if (split == 0 && m < MR) {
        const int t = m / GQ;
        const int h = m - t * GQ;
        *reinterpret_cast<device uint*>(qcodes + (size_t(GQ * kvh + h) * TR + t) * DD + fn * 16 + kk * 4) = w;
      }
#endif
    }
  }
  // The relay: a lane's 16 int32 partial scores (four int4) per row group.
  threadgroup int4 xs[RGT * 4 * 32];
  threadgroup int4* xl = xs + (rg * 32 + lane) * 4;

  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
  auto ct_q = qk_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_k = qk_op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  using ct_q_t = metal::remove_addrspace_t<decltype(ct_q)>;
  using ct_k_t = metal::remove_addrspace_t<decltype(ct_k)>;
  auto ct_s = qk_op.template get_destination_cooperative_tensor<ct_q_t, ct_k_t, int32_t>();

  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
  auto ct_p = pv_op.template get_left_input_cooperative_tensor<PE, int8_t, float>();
  auto ct_v = pv_op.template get_right_input_cooperative_tensor<PE, int8_t, float>();
  using ct_p_t = metal::remove_addrspace_t<decltype(ct_p)>;
  using ct_v_t = metal::remove_addrspace_t<decltype(ct_v)>;
  // The output accumulators stay in the cooperative tensors for the whole split and
  // are rescaled in place: one per output tile of this half.
  auto ct_o0 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o1 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o2 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  auto ct_o3 = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();
  UNROLL for (int e = 0; e < 16; ++e) { ct_o0[e] = 0.0f; ct_o1[e] = 0.0f; ct_o2[e] = 0.0f; ct_o3[e] = 0.0f; }
  float rmax[2] = {NEG, NEG};
  float rsum[2] = {0.0f, 0.0f};
  const int l_base = half_ * 8;

  for (int kb0 = k_begin; kb0 < k_end; kb0 += 32) {
    const device int8_t* kr[4];
    const device int8_t* vr[4];
    UNROLL for (int r = 0; r < 4; ++r) {
      const int key = min(kb0 + fm + 8 * r, n - 1);
      kr[r] = kbase + key * cst;
      vr[r] = vbase + key * cst;
    }
    // {sK, sV} of this lane's 8 score keys kb0 + hh*16 + fn + j: one 16-byte load per
    // 16-key half while the whole block lies below n, clamped half2 loads otherwise.
    half2 sv[2][4];
    if (kb0 + 32 <= n) {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        const uint4 w = *reinterpret_cast<const device uint4*>(sbase + (kb0 + hh * 16 + fn) * 2);
        sv[hh][0] = as_type<half2>(w.x); sv[hh][1] = as_type<half2>(w.y);
        sv[hh][2] = as_type<half2>(w.z); sv[hh][3] = as_type<half2>(w.w);
      }
    } else {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int j = 0; j < 4; ++j) {
          sv[hh][j] = *reinterpret_cast<const device half2*>(sbase + min(kb0 + hh * 16 + fn + j, n - 1) * 2);
        }
      }
    }
    UNROLL for (int e = 0; e < 16; ++e) ct_s[e] = 0;

    // Si_half = Q K^T over this half's 8 k-steps of 16 (permuted) dims, four k-steps per
    // 16-byte K load, accumulated in int32 in the cooperative tensor
    UNROLL for (int u = 0; u < 2; ++u) {
      uint4 kw[4];
      UNROLL for (int r = 0; r < 4; ++r) kw[r] = *reinterpret_cast<const device uint4*>(kr[r] + l_base * 4 + u * 16);
      UNROLL for (int s = 0; s < 4; ++s) {
        const int l = l_base + u * 4 + s;
        UNROLL for (int i = 0; i < 2; ++i) {
          const char4 qc = as_type<char4>(qtg[((rg * 16 + l) * 2 + i) * 32 + lane]);
          UNROLL for (int j = 0; j < 4; ++j) ct_q[i * 4 + j] = qc[j];
        }
        UNROLL for (int r = 0; r < 4; ++r) {
          const char4 b = as_type<char4>(s == 0 ? kw[r].x : s == 1 ? kw[r].y : s == 2 ? kw[r].z : kw[r].w);
          UNROLL for (int j = 0; j < 4; ++j) ct_k[r * 4 + j] = b[j];
        }
        qk_op.run(ct_q, ct_k, ct_s);
      }
    }
    int Si[16];
    UNROLL for (int e = 0; e < 16; ++e) Si[e] = ct_s[e];

    // relay: Si = Si_half0 + Si_half1 in int32, the same bits in both halves
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) xl[c4] = int4(Si[c4 * 4], Si[c4 * 4 + 1], Si[c4 * 4 + 2], Si[c4 * 4 + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 0) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const int4 o = xl[c4];
        Si[c4 * 4] += o.x; Si[c4 * 4 + 1] += o.y; Si[c4 * 4 + 2] += o.z; Si[c4 * 4 + 3] += o.w;
        xl[c4] = int4(Si[c4 * 4], Si[c4 * 4 + 1], Si[c4 * 4 + 2], Si[c4 * 4 + 3]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (half_ == 1) {
      UNROLL for (int c4 = 0; c4 < 4; ++c4) {
        const int4 o = xl[c4];
        Si[c4 * 4] = o.x; Si[c4 * 4 + 1] = o.y; Si[c4 * 4 + 2] = o.z; Si[c4 * 4 + 3] = o.w;
      }
    }

    // scores in log2 units, Si * (sq * (SCALE_LOG2 * sK)); then the causal mask (keys past
    // n are clamped copies and always masked) and the online softmax (exp2)
    float S[16];
    UNROLL for (int hh = 0; hh < 2; ++hh) {
      UNROLL for (int j = 0; j < 4; ++j) {
        const float fk = SCALE_LOG2 * float(sv[hh][j].x);
        UNROLL for (int i = 0; i < 2; ++i) S[hh * 8 + i * 4 + j] = float(Si[hh * 8 + i * 4 + j]) * (sqr[i] * fk);
      }
    }
    if (kb0 + 32 > prefix + 1) {
      UNROLL for (int i = 0; i < 2; ++i) {
        UNROLL for (int hh = 0; hh < 2; ++hh) {
          UNROLL for (int j = 0; j < 4; ++j) {
            const int key = kb0 + hh * 16 + fn + j;
            if (key >= lim[i]) S[hh * 8 + i * 4 + j] = NEG;
          }
        }
      }
    }
    float fac[2];
    UNROLL for (int i = 0; i < 2; ++i) {
      float mx = NEG;
      UNROLL for (int j = 0; j < 4; ++j) mx = max(mx, max(S[i * 4 + j], S[8 + i * 4 + j]));
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float nm = max(rmax[i], mx);
      const float ref = (nm < -1.0e29f) ? 0.0f : nm;
      fac[i] = (nm == rmax[i]) ? 1.0f : fast::exp2(rmax[i] - ref);
      rmax[i] = nm;
      float ls = 0.0f;
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int j = 0; j < 4; ++j) {
          const float p = fast::exp2(S[hh * 8 + i * 4 + j] - ref);
          S[hh * 8 + i * 4 + j] = p;
          ls += p;
        }
      }
      rsum[i] = rsum[i] * fac[i] + ls;
    }
    if (simd_any(fac[0] != 1.0f || fac[1] != 1.0f)) {
      UNROLL for (int hh = 0; hh < 2; ++hh) {
        UNROLL for (int i = 0; i < 2; ++i) {
          UNROLL for (int j = 0; j < 4; ++j) {
            const int e = hh * 8 + i * 4 + j;
            ct_o0[e] *= fac[i]; ct_o1[e] *= fac[i]; ct_o2[e] *= fac[i]; ct_o3[e] *= fac[i];
          }
        }
      }
    }

    // O += (P sV) V over the block's two 16-key halves, this simdgroup's 4 output tiles;
    // one 16-byte V load covers two tiles
    UNROLL for (int ks = 0; ks < 2; ++ks) {
      UNROLL for (int e = 0; e < 8; ++e) ct_p[e] = PE(S[ks * 8 + e] * float(sv[ks][e & 3].y));
      UNROLL for (int np = 0; np < 2; ++np) {
        uint4 vw[2];
        UNROLL for (int i = 0; i < 2; ++i) vw[i] = *reinterpret_cast<const device uint4*>(vr[ks * 2 + i] + np * 16);
        UNROLL for (int u = 0; u < 2; ++u) {
          UNROLL for (int i = 0; i < 2; ++i) {
            const char4 lo = as_type<char4>(u ? vw[i].z : vw[i].x);
            const char4 hi = as_type<char4>(u ? vw[i].w : vw[i].y);
            UNROLL for (int j = 0; j < 4; ++j) { ct_v[i * 4 + j] = lo[j]; ct_v[8 + i * 4 + j] = hi[j]; }
          }
          const int nn = np * 2 + u;
          if (nn == 0) pv_op.run(ct_p, ct_v, ct_o0); else if (nn == 1) pv_op.run(ct_p, ct_v, ct_o1);
          else if (nn == 2) pv_op.run(ct_p, ct_v, ct_o2); else pv_op.run(ct_p, ct_v, ct_o3);
        }
      }
    }
  }

  // partials [kvh][split][m][256] (physical dim order) and {max (log2 units), sum} per row
  float O[4][16];
  UNROLL for (int e = 0; e < 16; ++e) { O[0][e] = ct_o0[e]; O[1][e] = ct_o1[e]; O[2][e] = ct_o2[e]; O[3][e] = ct_o3[e]; }
  UNROLL for (int i = 0; i < 2; ++i) {
    float s = rsum[i];
    s += simd_shuffle_xor(s, ushort(1));
    s += simd_shuffle_xor(s, ushort(8));
    rsum[i] = s;
  }
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = rg * 16 + fm + 8 * i;
    if (m < MR) {
      const size_t row = (size_t(kvh) * size_t(nsplit) + size_t(split)) * MR + m;
      device float* pp = part + row * DD + fn * 16 + nn0 * 8;
      UNROLL for (int nn = 0; nn < 4; ++nn) {
        *reinterpret_cast<device float4*>(pp + nn * 8) =
            float4(O[nn][i * 4 + 0], O[nn][i * 4 + 1], O[nn][i * 4 + 2], O[nn][i * 4 + 3]);
        *reinterpret_cast<device float4*>(pp + nn * 8 + 4) =
            float4(O[nn][8 + i * 4 + 0], O[nn][8 + i * 4 + 1], O[nn][8 + i * 4 + 2], O[nn][8 + i * 4 + 3]);
      }
      if (fn == 0 && half_ == 0) {
        stats[row * 2] = rmax[i];
        stats[row * 2 + 1] = rsum[i];
      }
    }
  }
```

- [ ] **Step 6: Name kv8 where the shared headers and the kernels package describe their users**

In `src/sous/engine/kernels/common.h`, replace:

```c
// Shared by every kernel; prepended by int8prefill.py, tileattn.py and
// projkernel.py as the metal_kernel header.
```

with:

```c
// Shared by every kernel; prepended by int8prefill.py, tileattn.py, projkernel.py
// and kv8.py as the metal_kernel header.
```

In `src/sous/engine/kernels/nax.h`, replace:

```c
// Prepended after common.h for the tensor-op kernels only, the int8 GEMM and the
// attention tile's split kernel: the Metal-4 tensor-op header needs macOS 26.2+, and
// keeping it out of Stage A and the tile's reduce kernel lets those compile (and CI
// test them) on any Metal GPU.
```

with:

```c
// Prepended after common.h for the tensor-op kernels only, the int8 GEMM and the
// attention tile's and kv8's split kernels: the Metal-4 tensor-op header needs macOS
// 26.2+, and keeping it out of Stage A, the tile's reduce kernel and kv8's quantiser
// lets those compile (and CI test them) on any Metal GPU.
```

In `src/sous/engine/nax.py`, replace:

```python
Both runtime-compiled kernels that use the Metal-4 tensor ops, int8 prefill's GEMM
and the attention tile's split kernel, serve only where mlx itself would use the
neural accelerators. On an older GPU the tensor ops run on the plain shader path,
where neither kernel buys anything, so it is refused rather than merely slow.
```

with:

```python
The runtime-compiled kernels that use the Metal-4 tensor ops, int8 prefill's GEMM
and the attention tile's and kv8's split kernels, serve only where mlx itself would
use the neural accelerators. On an older GPU the tensor ops run on the plain shader
path, where none of them buys anything, so it is refused rather than merely slow.
```

In `src/sous/engine/kernels/__init__.py`, replace the whole docstring (the file's only content):

```python
"""Metal sources for sous's runtime-compiled kernels: int8 prefill, the
attention tile and the small-row projection kernel.

Read by ``sous.engine.int8prefill``, ``sous.engine.tileattn`` and
``sous.engine.projkernel`` through ``importlib.resources`` and handed to
``mx.fast.metal_kernel``, which compiles them at model load. The ``.metal`` files
are kernel *bodies* (mlx generates the signature), so they do not compile
standalone. ``common.h`` is prepended to every body; ``nax.h``, which includes the
Metal-4 tensor-op header, only to the tensor-op kernels: the int8 GEMM and the
tile's split kernel. ``attention_tile.metal`` holds two bodies, the split kernel
and the reduce kernel, divided at its ``// REDUCE`` line. ``projection_mma.metal``
holds three, the plain kernel, the staged kernel and the group-sums kernel,
divided at its ``// STAGED`` and ``// GROUP_SUMS`` lines; ``projection_mma.h``
holds the 8x8x8 simdgroup multiply-accumulate helper and the presum switch.
``projection_row.metal`` is the one-row kernel, which serves a single row in the
same summation order with no MMA. None of these uses tensor ops, so they compile
on any Metal GPU and CI runs their tests.

The int8 kernels are derived from oMLX (jundot/omlx#3548) and the projection
kernel in part from Splash (incoai/splash and its Apple7/8 port), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md. ``attention_tile.metal`` is sous's
own, and so is ``projection_row.metal`` apart from the nibble decode and epilogue it
shares with the projection kernel.
"""
```

with:

```python
"""Metal sources for sous's runtime-compiled kernels: int8 prefill, the
attention tile, kv8's int8 attention and the small-row projection kernel.

Read by ``sous.engine.int8prefill``, ``sous.engine.tileattn``,
``sous.engine.kv8`` and ``sous.engine.projkernel`` through
``importlib.resources`` and handed to ``mx.fast.metal_kernel``, which compiles
them at model load. The ``.metal`` files are kernel *bodies* (mlx generates the
signature), so they do not compile standalone. ``common.h`` is prepended to every
body; ``nax.h``, which includes the Metal-4 tensor-op header, only to the
tensor-op kernels: the int8 GEMM and the tile's and kv8's split kernels.
``attention_tile.metal`` holds two bodies, the split kernel and the reduce kernel,
divided at its ``// REDUCE`` line. ``attention_tile_kv8.metal`` is the tile's
split kernel over int8 K, V and Q, whose partials the tile's reduce kernel
combines, and ``kv8_quant.metal`` the row quantiser that fills its int8 mirror;
the quantiser uses no tensor ops, so CI runs its tests. ``projection_mma.metal``
holds three, the plain kernel, the staged kernel and the group-sums kernel,
divided at its ``// STAGED`` and ``// GROUP_SUMS`` lines; ``projection_mma.h``
holds the 8x8x8 simdgroup multiply-accumulate helper and the presum switch.
``projection_row.metal`` is the one-row kernel, which serves a single row in the
same summation order with no MMA. None of the projection kernels uses tensor ops,
so they compile on any Metal GPU and CI runs their tests.

The int8 prefill kernels are derived from oMLX (jundot/omlx#3548) and the
projection kernel in part from Splash (incoai/splash and its Apple7/8 port), both
under the Apache License 2.0; see THIRD_PARTY_NOTICES.md. ``attention_tile.metal``,
``attention_tile_kv8.metal`` and ``kv8_quant.metal`` are sous's own, and so is
``projection_row.metal`` apart from the nibble decode and epilogue it shares with
the projection kernel.
"""
```

`common.h` and `nax.h` are hashed into the fork store's epoch (`forkstore.engine_epoch` reads every `kernels/*.metal` and `*.h`), and so, until `forkstore` gains its deny-list for kv8's two kernel files in a later task, are those two files: the store's key moves with this commit. The branch already changes the epoch through its `vlm.py` and `tileattn.py` seams, so deploying it pays one cold store either way.

- [ ] **Step 7: Create the module's array-level core**

Create `src/sous/engine/kv8.py`. mlx is imported inside each function: the lint job runs on ubuntu without it. `decode_codes`, `verify_codes` and `_compile_probe` call `tile8` and `quantize` through the module globals at call time, which is how the tests swap in the stand-in and count the probe's launches.

```python
"""int8 K/V and Q through the attention tile, inside a generation.

The attention tile (``tileattn``) reads bf16 K/V at about 280 GB/s: bandwidth-bound
at one row and issue-bound from three, where a verify call sits. kv8 halves the
bytes and moves QK onto the tensor units' int8 rows. K and V become per-token
symmetric int8 codes with one fp16 scale per (token, KV head); Q is quantised the
same way inside the kernel with one fp32 scale per (query head, row); QK runs as
int8 x int8 -> int32, the K and Q scales multiply the finished score and the V
scale folds into the probability (``kernels/attention_tile_kv8.metal``). The split
kernel keeps the tile's partition, chunk schedule and partial layout, so the
tile's own reduce kernel combines its splits, and every verify row equals the
one-row call at its own length exactly as the tile's rows do.

The codes live in a mirror beside the bf16 cache that exists only inside
``VLMEngine.decode()``: every prefill, fork, slot and stored prompt stays bf16.
The fused row quantiser (``kernels/kv8_quant.metal``) fills it, and a row's codes
and scales depend on that row alone, so quantising a prefix at once or in pieces
gives the same bits.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from sous.engine import nax, tileattn
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.kv8")

# The first key count kv8 serves, a tile step boundary, so no verify run straddles
# it; below it the tile's and the stock paths run unchanged. It sits at the tile's
# N0 unless the int8 saving, net of the per-round quantisation, is measured below
# zero somewhere above it.
N8: int = 1536
# Rows a mirror is built or grown with past the call's own keys: at most one
# growth per 4,096 generated tokens, 135 MB over 16 layers.
GROWTH = 4096
# The kernel against an fp32 reference over the same codes (K, V and Q), as a
# relative RMS; its own error there is about 2e-3.
REL_RMS_BOUND = 1e-2
# kv8 against stock bf16 attention over the bf16 rows the codes came from; the
# format's own error is about 0.7 % on Gaussian K/V.
STOCK_RMS_BOUND = 3e-2
# The attribute a full-attention layer's KVCache carries its mirror under.
MIRROR_ATTR = "_sous_kv8"
# Mirror bytes per token per KV head: 256 K codes, 256 V codes and two fp16 scales,
# against the bf16 cache's 1,024.
BYTES_PER_ROW_PER_HEAD = 516
# mlx's default, stated rather than inherited: verify rows equal decode rows only if
# each rounds exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}
# Appended to the debug split kernel's header only: the body then also writes the Q
# codes it staged, to an output only that kernel object declares.
_QDBG = "#define QDBG 1\n"


# Once per process, as the tile's: per-launch text work slows the decode hot path.
@functools.cache
def _kernels() -> tuple[
    Callable[..., list[Any]], Callable[..., list[Any]], Callable[..., list[Any]]
]:
    """The quantiser, kv8's split kernel and the tile's reduce kernel, built on first
    use. Both of kv8's kernels read their inputs in place through their strides: the
    quantiser takes slices of a KVCache's padded buffers and the split kernel the
    mirror's capacity-sized ones, which ensure_row_contiguous would copy whole on
    every call. Only the split kernel uses the tensor ops, so only its header carries
    nax.h."""
    common = _kernel_text("common.h")
    quant = _kernel(
        "sous_kv8_quant",
        ["k", "v", "params"],
        ["codes", "sc"],
        _kernel_text("kv8_quant.metal"),
        common,
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    split = _kernel(
        "sous_kv8_tile_split",
        ["q", "codes", "sc", "params"],
        ["part", "stats"],
        _kernel_text("attention_tile_kv8.metal"),
        common + _kernel_text("nax.h"),
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    return quant, split, tileattn._kernel_pair()[1]


@functools.cache
def _qdbg_split() -> Callable[..., list[Any]]:
    """The split kernel with QDBG defined: the same body, which then also writes the Q
    codes it staged to a third output, qcodes. mlx names every output of a kernel
    object in the kernel's signature, and a store to an undeclared name does not
    compile even in a branch a template argument switches off, so the debug store
    sits behind the preprocessor and in a kernel object of its own. Tests only."""
    return _kernel(
        "sous_kv8_tile_split_qdbg",
        ["q", "codes", "sc", "params"],
        ["part", "stats", "qcodes"],
        _kernel_text("attention_tile_kv8.metal"),
        _kernel_text("common.h") + _kernel_text("nax.h") + _QDBG,
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )


def quantize(k: Any, v: Any, out_rows: int) -> tuple[Any, Any]:
    """Mirror rows of bf16 K and V views [1, 4, R, 256] (innermost stride 1, read in
    place): codes int8 [1, 4, out_rows, 512], each row's K codes then its V codes, and
    sc fp16 [1, 4, out_rows, 2] = {sK, sV}. Per (row, head): scale = amax / 127 in
    fp16, saturated at fp16's largest finite value; code = clamp(rint(x / f), -127,
    127) with f the scale, or 1 where the scale is 0. Rows from R up are left
    unwritten, so a first build sizes the outputs to the mirror's capacity in the
    same dispatch."""
    import mlx.core as mx

    rows = k.shape[2]
    if not 1 <= rows <= out_rows:
        raise ValueError(f"kv8 quantises 1..{out_rows} rows into {out_rows}, got {rows}")
    hkv, d = tileattn.HKV, tileattn.D
    codes, sc = _kernels()[0](
        inputs=[k, v, mx.array([rows, out_rows], dtype=mx.int32)],
        grid=(d, rows, hkv),
        threadgroup=(d, 1, 1),
        output_shapes=[(1, hkv, out_rows, 2 * d), (1, hkv, out_rows, 2)],
        output_dtypes=[mx.int8, mx.float16],
    )
    return codes, sc


def dequant(codes: Any, sc: Any) -> tuple[Any, Any]:
    """What mirror rows mean: fp32 K and V [1, 4, R, 256], each code times its row's
    scale. The tests' and the probe's reference; the kernel never dequantises."""
    import mlx.core as mx

    values = codes.astype(mx.float32)
    scales = sc.astype(mx.float32)
    d = tileattn.D
    return values[..., :d] * scales[..., 0:1], values[..., d:] * scales[..., 1:2]


def _launch(
    split: Callable[..., list[Any]],
    queries: Any,
    codes: Any,
    sc: Any,
    n: int,
    chunk: int,
    *,
    qcodes: bool = False,
) -> list[Any]:
    """One split dispatch and the tile's reduce: [out], plus the staged Q codes when
    `split` is the debug kernel and `qcodes` asks for them."""
    import mlx.core as mx

    hq, hkv, d = tileattn.HQ, tileattn.HKV, tileattn.D
    t = queries.shape[2]
    mr = tileattn.GQA * t
    rgt = -(-mr // 16)
    nsplit = -(-n // chunk)
    params = mx.array([n, chunk, nsplit], dtype=mx.int32)
    shapes: list[tuple[int, ...]] = [(hkv * nsplit * mr * d,), (hkv * nsplit * mr * 2,)]
    dtypes = [mx.float32, mx.float32]
    if qcodes:
        shapes.append((1, hq, t, d))
        dtypes.append(mx.int8)
    outs = split(
        inputs=[queries, codes, sc, params],
        template=[("TR", t), ("MR", mr), ("RGT", rgt)],
        grid=(64 * rgt, nsplit, hkv),
        threadgroup=(64 * rgt, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    (out,) = _kernels()[2](
        inputs=[outs[0], outs[1], params],
        template=[("MR", mr)],
        grid=(d, mr, hkv),
        threadgroup=(d, 1, 1),
        output_shapes=[(1, t, hq, d)],
        output_dtypes=[mx.bfloat16],
    )
    return [out, *outs[2:]]


def tile8(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> Any:
    """One kernel call over T <= MAX_RUN rows: row t sees keys [0, n - T + t], every
    code and scale read is clamped to n - 1, and keys are split at multiples of
    `chunk`. Queries are bf16 [1, 24, T, 256] with innermost stride 1; codes and sc
    are mirror buffers of at least n rows whose scale plane's head stride is a
    multiple of 4 rows. Returns the contiguous [1, T, 24, 256] bf16 output."""
    return _launch(_kernels()[1], queries, codes, sc, n, chunk)[0]


def _tile8_qcodes(queries: Any, codes: Any, sc: Any, n: int, chunk: int) -> tuple[Any, Any]:
    """tile8() through the debug kernel: its output and the Q codes the kernel staged,
    int8 [1, 24, T, 256] in q's dim order."""
    out, qcodes = _launch(_qdbg_split(), queries, codes, sc, n, chunk, qcodes=True)
    return out, qcodes


def decode_codes(queries: Any, codes: Any, sc: Any, n: int, splits: int) -> Any:
    """The one-row call over n keys of a mirror, [1, 24, 1, 256], at the tile's chunk
    for n. kv8 has no stock path of its own: below N0 the caller runs the tile's."""
    chunk = tileattn.chunk_for(n, splits)
    if chunk == 0:
        raise ValueError(f"kv8 serves from N0 = {tileattn.N0} keys, got {n}")
    return tile8(queries, codes, sc, n, chunk).transpose(0, 2, 1, 3)


def verify_codes(
    queries: Any,
    codes: Any,
    sc: Any,
    n: int,
    splits: int,
    runs: list[tuple[int, int, int]] | None = None,
) -> Any:
    """The verifier's T-row causal call over n keys of a mirror, row r over keys
    [0, n - T + r], as [1, 24, T, 256]: one kernel call per run of
    tileattn.verify_plan(), concatenated in row order. `runs` is that plan when the
    caller already has it. Every run must be at or past N0: kv8 has no stock path of
    its own."""
    import mlx.core as mx

    t = queries.shape[2]
    prefix = n - t
    if runs is None:
        runs = tileattn.verify_plan(prefix, t, splits)
    if any(chunk == 0 for _, _, chunk in runs):
        raise ValueError(f"kv8 serves from N0 = {tileattn.N0} keys, got a row at {prefix + 1}")
    if len(runs) == 1:
        return tile8(queries, codes, sc, n, runs[0][2]).transpose(0, 2, 1, 3)
    parts = [tile8(queries[:, :, j:k], codes, sc, prefix + k, chunk) for j, k, chunk in runs]
    return mx.concatenate(parts, axis=1).transpose(0, 2, 1, 3)


_compile_ok = False


def _compile_probe() -> str | None:
    """Compile and run the quantiser and every row variant of the split kernel,
    T = 1..MAX_RUN, on serving-layout inputs; None when they all work, else the
    compiler's error line. Both are compiled at runtime against the OS's own Metal
    toolchain, so a new macOS can reject a source that compiled before. The inputs
    are built with GPU ops and evaluated first: a compile failure with a CPU-stream
    op in flight deadlocks inside mlx's exception path instead of raising. Only
    success is remembered, so one bad moment cannot switch kv8 off for a long-lived
    daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        hq, hkv, d = tileattn.HQ, tileattn.HKV, tileattn.D
        n = tileattn.N0
        # The mirror's capacity runs past its live rows, as a first build leaves it.
        cap = n + 256
        q_key, k_key, v_key = mx.random.split(mx.random.key(0), 3)
        rows = mx.random.normal((1, tileattn.MAX_RUN, hq, d), key=q_key).astype(mx.bfloat16)
        keys = mx.random.normal((1, hkv, cap, d), key=k_key).astype(mx.bfloat16)
        values = mx.random.normal((1, hkv, cap, d), key=v_key).astype(mx.bfloat16)
        mx.eval(rows, keys, values)
        codes, sc = quantize(keys[:, :, :n], values[:, :, :n], cap)
        mx.eval(codes, sc)
        queries = rows.transpose(0, 2, 1, 3)
        chunk = tileattn.chunk_for(n, min(tileattn.MEASURED_SPLITS))
        for t in range(1, tileattn.MAX_RUN + 1):
            mx.eval(tile8(queries[:, :, :t], codes, sc, n, chunk))
    except Exception as e:  # noqa: BLE001 — any failure means the kernels cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("int8 K/V mirror: the kernels failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares, then a compile and run of the quantiser
    and every row variant. kv8's `nax` tests skip on this, never on the tile's: the
    two split kernels can fail to compile on different toolchains."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the int8 K/V kernels failed: {failed}")
    return nax.Availability(True)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kv8.py -rs`
Expected: PASS, 52 passed and 45 skipped on the maintainer's M2 (macOS 15.5), about 30 s: every quantiser, source, kernel-object, stand-in and availability test runs, and the 45 skips (the 36 `kfx.nax` tests and the 9 `real` parameters of `kernel8`) all read `kv8 kernels unavailable: macOS 15.5 < 26.2`. CI's macos-15 runner skips the same 45 with its own macOS version.

- [ ] **Step 9: Run the neighbouring tests**

Run: `uv run pytest tests/test_tileattn.py tests/test_int8prefill.py tests/test_forkstore.py tests/test_nax.py`
Expected: PASS, 293 passed and 30 skipped on the maintainer's M2 (about 90 s): the comment edits change no behaviour, and `test_kernel_sources_ship_as_package_files` still finds no MPP include in `common.h`.

- [ ] **Step 10: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/kernels/kv8_quant.metal src/sous/engine/kernels/attention_tile_kv8.metal src/sous/engine/kernels/common.h src/sous/engine/kernels/nax.h src/sous/engine/kernels/__init__.py src/sous/engine/nax.py src/sous/engine/kv8.py tests/kv8_fixtures.py tests/test_kv8.py
git commit -m "feat(engine): add kv8's int8 attention kernels and their entries" -m "The attention tile reads bf16 K/V at 270-283 GB/s and is issue-bound
from three rows, where verify calls sit. Its relay kernel over int8 K, V
and Q, with Q quantised on the host, ran 1.69x the bf16 relay at T = 5
and 57K keys and 1.81x at T = 1. This adds that kernel with Q quantised
in its own prologue and K and V read as two strided views of one packed
buffer, the fused row quantiser that fills the mirror it reads, and the
array-level entries the serving seams will call; nothing serves through
them yet. QK sums int8 products in int32, so the relay between the
simdgroup halves is exact and every verify row still equals the one-row
call at its own length. A debug kernel object writes out the Q codes the
kernel staged, so the tests hold them to the host's quantisation bit for
bit."
```

Expected: `ruff format` leaves every file unchanged, `ruff check` and `ty check` print `All checks passed!`, and `git log -1 --format='%G? %an <%ae>%n%B'` shows `G`, the maintainer's noreply identity and no attribution line.

- [ ] **Step 11: Run the real-kernel tests on the M5 Pro**

The split kernel needs the tensor units, so this is where its `kfx.nax` tests first run. The M5 is reached through `~/code/sous-spikes/p148/gate/m5.sh` (ssh to the LAN alias `m5pro-lan`, checked against the host key recorded for the tailnet name). The branch runs from its own detached worktree, `~/code/personal/sous-remote-kv8`, of `~/code/personal/sous-remote`, where nothing but `git fetch` and `git worktree add` ever runs; never touch `~/code/personal/sous`, `~/.sous` or the daemon. The job runs under `gpu_run_m5.py`, which waits for the maintainer's daemon to be idle.

```bash
git push -u origin feat/int8-kv-mirror
~/code/sous-spikes/p148/gate/m5.sh m5pro-lan 'R=~/code/personal/sous-remote W=~/code/personal/sous-remote-kv8 && git -C $R fetch -q origin && { [ -d $W ] || git -C $R worktree add -q --detach $W origin/feat/int8-kv-mirror; } && git -C $W checkout -q --detach origin/feat/int8-kv-mirror && cd $W && uv sync -q && env GPU_LABEL=kv8-kernels python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py uv run pytest tests/test_kv8.py -rs 2>&1 | tail -15'
```

Expected: `97 passed` with nothing skipped, and the runner's `daemon stayed idle during the job` (any other verdict: run it again). About two minutes, most of it the 57,601- and 131,201-key cases. A skip that reads `kv8 kernels unavailable: the int8 K/V kernels failed: <line>` is the M5's toolchain rejecting the split kernel: the full compiler report is in the warning `int8 K/V mirror: the kernels failed to build or run` that `_compile_probe` logs; fix the Metal (every local cooperative-tensor `decltype` keeps its `metal::remove_addrspace_t`), commit the fix as `fix(engine): ...`, push and rerun. A failing Q-codes, parity, fence or bound assertion is a kernel bug: fix the Metal, never the expectation.


### Task 2: The serving window, the mirror and its trim wrap, the ceiling and the floor

**Files:**
- Modify: `src/sous/engine/kv8.py` (the import block; the serving section appended after Task 1's code)
- Modify: `tests/kv8_fixtures.py` (the import block; `window`, `ceiling` and `small_floor` appended)
- Test: `tests/test_kv8.py` (the header; the serving tests appended after Task 1's tests)

All three files are Task 1's. This task replaces no existing line in them or anywhere else in the tree: it adds imports and appends sections.

**Interfaces:**
- Consumes:
  - Task 1, in `kv8`:
    - `N8 = 1536`, a plain module int that both seams and this task read at call time (tests monkeypatch it); `GROWTH = 4096`; `MIRROR_ATTR = "_sous_kv8"`; `BYTES_PER_ROW_PER_HEAD = 516`.
    - `quantize(k, v, out_rows) -> (codes, sc)`: bf16 views `[1, 4, R, 256]` read through their strides; returns fresh int8 `[1, 4, out_rows, 512]` and fp16 `[1, 4, out_rows, 2]` with rows `[0, R)` written and rows `[R, out_rows)` left unwritten. It is a pure per-row function, so bulk and incremental quantisation agree bit for bit.
    - `tile8(queries, codes, sc, n, chunk) -> [1, T, 24, 256]` bf16: row t of a T-row call over keys `[0, n - T + t]`, every read clamped to `n - 1`, `codes`/`sc` holding at least n rows. Looked up through the module at call time, so the stand-in can be swapped in.
    - `decode_codes(queries, codes, sc, n, splits) -> [1, 24, 1, 256]`: `tile8` at `tileattn.chunk_for(n, splits)`, transposed; n ≥ N0; buffers holding at least n rows.
    - The module's `from sous.engine import tileattn` (module level).
  - Task 1, in `tests/kv8_fixtures.py`: `stand_in_tile8(queries, codes, sc, n, chunk)`, with `tile8`'s contract and each row's output independent of the rows beside it.
  - Task 1, in `tests/test_kv8.py`: the header with `mx = pytest.importorskip("mlx.core")` and the `kv8`, `tileattn` and `kfx` imports.
  - tileattn (existing): `N0`, `HQ`, `HKV`, `D`, `PROBE_FAR_KEYS`, `boundaries(n_max, splits)` (N0 first, always), `chunk_for(n, splits)`, `verify_plan(prefix, t, splits) -> [(j, k, chunk)]`.
  - mlx-vlm 0.7.2 `mlx_vlm.models.cache`:
    - `KVCache.update_and_fetch`, which truncates the buffer to the offset before it grows;
    - `KVCache.trim(n)`, which only lowers `offset`;
    - the `KVCache.state` setter;
    - `ArraysCache(size=2)`, the linear-attention layers' cache.
- Produces, in `kv8`:
  - `Window`, a mutable dataclass `(serial: int, state: str = "idle", rows: int = 0, served: int = 0, built: bool = False, force: bool = False)`.
    - `state` is `idle`, then `served` from the first served call. It becomes `capped` at the first call whose first row is at or past L, and stays there.
    - `rows` counts rows quantised in the window, `served` the calls `prepare` served, and `built` says a mirror was built.
  - `Mirror`, a dataclass `(codes, sc, cap: int, n: int, serial: int, original_trim)`, set on a KVCache as `MIRROR_ATTR` with `object.__setattr__`. `original_trim` is the instance's own `trim` from before the wrap, or None when the class's method was in effect.
  - Module state:
    - `_active: int = 0`, `_ceiling: int = 0` and `_window: Window | None = None`;
    - `_serial: int = 0`, raised by every `open_window` and never reset;
    - `calls` with exactly the keys `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_rows`, `mirror_trims`, `mirror_grown`, `windows` and `windows_capped`, all starting at 0.
    - This task moves every key except the three `verify_*`. Those belong to the verify seam, which counts the call, its rows and its launches; `verify_run` counts nothing.
  - `ceiling(window: int, splits: int) -> int`: the largest b in `tileattn.boundaries(window, splits)` with `1.504*C + 0.504*C/16 + 272 <= window`, where C = b rounded up to 256; else 0. At S = 20: 0 below a 2,631-token window, 1,536 from it, 84,481 at 131072 and 170,241 at 262144.
  - `_floor_reason(splits: int) -> str | None`: None when `N8 in tileattn.boundaries(N8, splits)`, else `f"floor {N8} keys is not a tile step boundary at {splits} splits"`. This is the N8-in-boundaries check `enable()` runs.
  - `open_window(*, force: bool = False) -> Window`: raises `AssertionError` if a window is open; raises `_serial` and `calls["windows"]`.
  - `close_window(cache: Sequence[Any]) -> tuple[str, int]`:
    - drops every mirror and trim wrap in `cache`, passing over layers without one;
    - clears `_window`;
    - calls `mx.clear_cache()` when the window built a mirror;
    - returns `(state, rows)`, or `("off", 0)` when no window was open.
  - `serving_now() -> bool`: `_active` is set and a window is open whose state is not `capped`.
  - `serves(n_row: int) -> bool`: `serving_now()`, `n_row >= N8`, and `n_row < _ceiling` unless the window is forced. It reads `N8`, `_ceiling` and `_window` at call time.
  - `observe_first_row(cache: Any, n_first: int) -> None`: when a window is open, not forced, and `n_first >= _ceiling`, sets `state = "capped"`, counts `windows_capped` once per window, and drops this cache's mirror and wrap.
  - `prepare(cache: Any, t: int) -> Mirror`, the one call that builds, grows, clamps and quantises:
    - raises `AssertionError` with no window open;
    - counts the call in `Window.served`;
    - sets `n = min(n, offset - t)`;
    - quantises rows `[n, min(offset, L))`, or `[n, offset)` in a forced window, and sets `n` to that top.
    - The first served call of a window on a layer, or a mirror from another window, builds from row 0 in one `quantize` dispatch of `out_rows = cap`. Later calls make one row-sized dispatch and two slice assignments.
    - `cap = min(round_up(offset + GROWTH, 256), round_up(L, 256))`, without the second term in a forced window.
  - `decode(queries: Any, cache: Any, splits: int) -> Any`: `prepare(cache, 1)`, `calls["decode_kv8"] += 1`, then `decode_codes` over the mirror at `n = cache.offset`.
  - `verify_run(queries: Any, mirror: Mirror, prefix: int, j: int, k: int, chunk: int) -> Any`: `tile8` over rows `[j, k)` at `n = prefix + k`, `[1, k - j, 24, 256]`.
  - `clear() -> None`: `_active = 0` and `_window = None`, nothing else.
  - Private, for later tasks and tests:
    - `_CAP_STEP = 256` and `_TRIM_MARK = "_sous_kv8_trim"`;
    - `_round_up(rows) -> int`, to a multiple of 256;
    - `_build(cache, window, top)`, `_capacity(offset, window)`, `_grow(mirror, cap)` and `_count_rows(window, rows)`;
    - `_wrap_trim(cache, original)`, which calls the trim it wrapped, then sets `n = min(n, offset)` and counts `mirror_trims`;
    - `_drop(cache)`, which removes the mirror and puts back the trim only if the instance's `trim` still carries `_TRIM_MARK`.
- Produces, in `tests/kv8_fixtures.py`:
  - `window(caches: Sequence[Any] = (), *, force: bool = True)`: a context manager yielding the open `kv8.Window`. It sets `kv8._active = 1` for the block. On the way out it closes the window with `caches` unless the block closed it, then puts the flag back.
  - `ceiling(monkeypatch, L)`: `kv8._ceiling = L` for the test.
  - `small_floor(monkeypatch)`: `kv8.N8 = tileattn.N0` and `tileattn.PROBE_FAR_KEYS = 3_000` for the test.
- Produces, in `tests/test_kv8.py`: the module-level helpers `_MIRROR_S`, `_mirror_counts_since`, `_mirror_append`, `_mirror_cache`, `_mirror_of`, `_mirror_bulk`, `_mirror_matches`, `_mirror_queries`, `_mirror_recording_quantize` and `_mirror_fits`. Their names keep them clear of Task 1's helpers in the same file.

How the seams (Task 3) are meant to call this:
- **Decode seam:** `kv8.serves(n_keys)` → `kv8.decode(queries, cache, splits)`, else `kv8.observe_first_row(cache, n_keys)`.
- **Verify seam:**
  - `kv8.observe_first_row(cache, prefix + 1)` first;
  - each run is served iff `kv8.serves(prefix + j + 1)`;
  - if any run is served, `mirror = kv8.prepare(cache, t)` once, before the loop;
  - `kv8.verify_run(queries, mirror, prefix, j, k, chunk)` per served run.

**Validity, in one place.** A mirror's `n` moves down in three ways:
- through the trim wrap. This is the mechanism: inside a decode only `KVCache.trim` lowers an offset, and DFlash's commit (`_trim_append_caches`, `getattr(cache, "trim")`) and abort (`cache.trim`) both reach the instance attribute.
- through the clamp `prepare` runs first. This is a defence: rows below `offset - t` are the committed prefix.
- through the serial. A mirror built in another window is rebuilt from row 0; a fork restore assigns `state` without a trim.

It moves up only in `prepare`. Rows at or past L are never written, and the capacity never exceeds `round_up(L, 256)`.

**Notes on the code:**
- **The wrap holds the cache weakly.** A wrap set as an instance attribute and holding the cache strongly would make a reference cycle, `cache.__dict__ → wrap → cache`. The cache's K/V, hundreds of megabytes per layer at long contexts, would then wait for the cycle collector whenever a wrap outlived its window.
- **One wrap per cache.** A mirror from another window keeps the wrap that window installed (`_build` carries its `original_trim` over), so a rebuild never wraps twice. `_drop` puts back only a `trim` carrying `_TRIM_MARK`, so a trim someone set over the wrap since is left alone.
- **The capacity is the mirror's own.** `KVCache.update_and_fetch` cuts its buffer to the offset before it grows, so `keys.shape[2]` is routinely not a multiple of 4 (1,537 + 512 = 2,049 after a second prefill segment). The kernel's 16-byte scale loads need the `sc` head plane's stride, `cap × 4` bytes, to be a multiple of 16, so `cap` is a multiple of 256.
- **Growth evaluates before it drops.** On the M2, with a plain-mlx quantiser in Task 1's place, six layers grown from 5,888 to 10,240 rows in one pass peaked 1.00 grown layer buffers above the grown total (one old buffer plus its zero tail). Left lazy, they peaked 6.06 above it: every layer's old buffer was alive at the forward's evaluation.
- **Updates are in place.** The `Mirror` is the only holder of its buffers, so mlx donates them to the slice updates. Measured the same way, a 5-row update peaked at about 60 KB against a 12 MB mirror. With one extra reference to `codes` held, it became a 12 MB copy.
- **ceiling() in integers.** `1504*C + 504*(C // 16) + 272_000 <= 1000*window` is that rule multiplied by 1,000. It is exact because C is a multiple of 256, so a boundary on the margin cannot flip on float rounding. At 262144 the rule gives 170,241 keys, the figure the design states.
- **ty.** `_active`, `_ceiling` and `_serial` are annotated `int`. Unannotated, ty infers `Literal[0]` and rejects the fixture's `kv8._active = 1` (`invalid-assignment`).
- **close_window() runs in a finally.** With no window open, because `clear()` ran, it still drops the mirrors and returns `("off", 0)` rather than raising over the error in flight.
- **Call-time lookups.** `N8`, `GROWTH`, `quantize`, `tile8` and `decode_codes` are read as module globals on every call, which is what lets the tests swap them.

- [ ] **Step 1: Add the fixtures**

In `tests/kv8_fixtures.py`, add each of these imports that Task 1's import block does not already have. `uv run ruff check --fix tests/kv8_fixtures.py` restores isort order.

```python
import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from sous.engine import kv8, tileattn
```

Append to the end of the file. `window()` sets the flag directly rather than through monkeypatch, so it can be used without one, and puts back the value it found.

```python
@contextlib.contextmanager
def window(caches: Sequence[Any] = (), *, force: bool = True) -> Iterator[kv8.Window]:
    """kv8 active and a serving window open for the with-block: forced (every
    row from N8 served, whatever the ceiling) unless `force` is False, for a
    cap test with ceiling() set. On the way out the window is closed, which
    drops the mirrors and trim wraps on `caches`, unless the block closed it
    itself, and the flag is put back."""
    active = kv8._active
    kv8._active = 1
    opened = kv8.open_window(force=force)
    try:
        yield opened
    finally:
        if kv8._window is opened:
            kv8.close_window(caches)
        kv8._active = active


def ceiling(monkeypatch: pytest.MonkeyPatch, L: int) -> None:
    """kv8's ceiling at L keys for this test only, as enable() sets it from the
    window: where an unforced window caps."""
    monkeypatch.setattr(kv8, "_ceiling", L)


def small_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """For this test only, kv8 serves from N0 and the probe's far mark is one the
    stand-in reaches in seconds, whatever N8 and far mark the modules carry."""
    monkeypatch.setattr(kv8, "N8", tileattn.N0)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_kv8.py`, the header must hold these lines; add any that Task 1's header lacks.

```python
import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import kv8, tileattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402
```

Append to the end of the file. How the tests are built:
- **Real caches.** Every cache is a real mlx-vlm `KVCache`, filled through `update_and_fetch` as an attention layer fills it, and every quantisation is Task 1's real quantiser. It is MPP-free, so CI's GPU runs it. Only where a kernel would run is `kv8.tile8` replaced by `kfx.stand_in_tile8`.
- **The reference.** Every mirror is checked against `kv8.quantize` over contiguous copies of rows `[0, n)` in one call (`_mirror_bulk`). Quantisation is a pure per-row function, so an incrementally kept mirror must equal it bit for bit. The copies also show that the mirror's strided views were read correctly.
- **Sizes.** `small_floor` keeps N8 at N0 wherever the floor matters, so the tests stay at 1.5K–10K keys whatever floor the gate settles on.
- **Windows.** `kfx.window()` is forced, except where a test needs the ceiling: those set it with `kfx.ceiling` and open an unforced window.
- **Growth.** `GROWTH` is cut to 256 where a test must cross a capacity in a few calls.
- **Peak memory.** Each memory test evaluates and synchronises before it reads the base and resets the peak. The growth test compares the peak with the grown total computed from the shapes, not with active memory read afterwards: mlx frees a finished command buffer's inputs asynchronously.
- **The negative control.** `test_mirror_trim_that_bypasses_the_wrap_leaves_stale_rows` shows the wrap, not the clamp, is what keeps replaced rows from going stale. It also pins the clamp.

When this plan was written, these tests were run against this task's code over a plain-mlx stand-in for Task 1's quantiser: 32 passed in 1.6 s on the M2. They were also run against eleven deliberate breakages, and each made at least one test fail:
- the wrap not lowering n;
- no serial check;
- lazy growth;
- a copied update;
- decode reading the capacity;
- rows past L written;
- no capacity bound at L;
- the cap counted per layer;
- no clamp;
- no drop on a cap;
- the floor ignored.

```python
# ---- the serving window, the mirror and its trim wrap, the ceiling and the floor ----

# The M5 Pro's split target. The stand-in serves any S; the schedule is S's.
_MIRROR_S = 20


def _mirror_counts_since(before):
    """kv8's counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in kv8.calls.items() if n != before[k]}


def _mirror_append(cache, rows, *, seed):
    """`rows` random bf16 K/V rows appended through KVCache.update_and_fetch, as an
    attention layer appends them, and the cache evaluated."""
    k_key, v_key = mx.random.split(mx.random.key(seed))
    shape = (1, tileattn.HKV, rows, tileattn.D)
    keys = mx.random.normal(shape, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(shape, key=v_key).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)


def _mirror_cache(*segments, seed=0):
    """A real KVCache prefilled by one update_and_fetch per segment of rows."""
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    for i, rows in enumerate(segments):
        _mirror_append(cache, rows, seed=1000 * seed + i)
    return cache


def _mirror_of(cache):
    return getattr(cache, kv8.MIRROR_ATTR, None)


def _mirror_bulk(cache, n):
    """Rows [0, n) quantised in one call from contiguous copies: what every mirror
    must hold, since quantisation is a pure per-row function. The copies make a
    match also show the mirror's views were read through their strides."""
    codes, sc = kv8.quantize(
        mx.contiguous(cache.keys[:, :, :n]), mx.contiguous(cache.values[:, :, :n]), n
    )
    mx.eval(codes, sc)
    return codes, sc


def _mirror_matches(cache, n):
    """The cache's mirror is valid for exactly rows [0, n), and they equal a bulk
    quantisation of the cache's rows."""
    mirror = _mirror_of(cache)
    if mirror is None or mirror.n != n:
        return False
    codes, sc = _mirror_bulk(cache, n)
    return (
        mx.array_equal(mirror.codes[:, :, :n], codes).item()
        and mx.array_equal(mirror.sc[:, :, :n], sc).item()
    )


def _mirror_queries(t, *, seed):
    """[1, 24, T, 256] bf16 queries as served: a transposed [1, T, 24, 256] pool."""
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=mx.random.key(seed))
    pool = pool.astype(mx.bfloat16)
    mx.eval(pool)
    return pool.transpose(0, 2, 1, 3)


def _mirror_recording_quantize(monkeypatch):
    """kv8.quantize, recorded as (rows, out_rows) per dispatch."""
    seen = []
    real = kv8.quantize

    def recording(k, v, out_rows):
        seen.append((k.shape[2], out_rows))
        return real(k, v, out_rows)

    monkeypatch.setattr(kv8, "quantize", recording)
    return seen


def _mirror_fits(window, b):
    c = -(-b // 256) * 256
    return 1.504 * c + 0.504 * c / 16 + 272 <= window


@pytest.mark.parametrize(("window", "limit"), [(131072, 84481), (262144, 170241)])
def test_ceiling_is_the_last_step_boundary_whose_mirror_fits_beside_the_kv(window, limit):
    assert kv8.ceiling(window, _MIRROR_S) == limit
    bounds = tileattn.boundaries(window, _MIRROR_S)
    assert _mirror_fits(window, limit)
    assert not _mirror_fits(window, bounds[bounds.index(limit) + 1])


def test_ceiling_is_zero_below_the_first_fitting_window():
    """N0 rounds up to a capacity of 1,536 rows, which needs a 2,631-token window."""
    assert kv8.ceiling(0, _MIRROR_S) == 0
    assert kv8.ceiling(2630, _MIRROR_S) == 0
    assert kv8.ceiling(2631, _MIRROR_S) == tileattn.N0


def test_floor_must_be_a_step_boundary(monkeypatch):
    for floor in (tileattn.N0, 1921, 2561):
        monkeypatch.setattr(kv8, "N8", floor)
        assert kv8._floor_reason(_MIRROR_S) is None
    monkeypatch.setattr(kv8, "N8", 2000)
    reason = kv8._floor_reason(_MIRROR_S)
    assert reason == "floor 2000 keys is not a tile step boundary at 20 splits"
    # 1,921 is a boundary at 20 splits and not at 16, where steps are 512 keys.
    monkeypatch.setattr(kv8, "N8", 1921)
    assert kv8._floor_reason(16) is not None


def test_window_serials_increase_and_a_second_window_is_refused():
    before = dict(kv8.calls)
    with kfx.window() as first, pytest.raises(AssertionError):
        kv8.open_window()
    with kfx.window() as second:
        assert kv8._window is second
    assert kv8._window is None
    assert second.serial == first.serial + 1 == kv8._serial
    assert (first.state, first.rows, first.served, first.built) == ("idle", 0, 0, False)
    assert _mirror_counts_since(before) == {"windows": 2}


def test_window_absent_or_flag_clear_serves_and_builds_nothing(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1 << 20)
    monkeypatch.setattr(kv8, "_active", 1)
    cache = _mirror_cache(kv8.N8 + 8)
    assert kv8._window is None
    assert not kv8.serving_now() and not kv8.serves(kv8.N8)
    kv8.observe_first_row(cache, 1 << 21)  # no window: nothing to cap
    with pytest.raises(AssertionError):
        kv8.prepare(cache, 1)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)
    with kfx.window(force=False) as opened:
        monkeypatch.setattr(kv8, "_active", 0)
        assert not kv8.serving_now() and not kv8.serves(kv8.N8)
        assert opened.state == "idle"


def test_serves_rows_from_the_floor_to_below_the_ceiling(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 2561)
    with kfx.window(force=False):
        assert kv8.serving_now()
        assert [kv8.serves(n) for n in (kv8.N8 - 1, kv8.N8, 2560, 2561, 1 << 20)] == [
            False,
            True,
            True,
            False,
            False,
        ]


def test_window_of_zero_serves_nothing_unless_forced(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, kv8.ceiling(0, _MIRROR_S))
    rows = (kv8.N8 - 1, kv8.N8, kv8.N8 + 1, 1 << 20)
    with kfx.window(force=False):
        assert not any(kv8.serves(n) for n in rows)
    cache = _mirror_cache(kv8.N8 + 64)
    with kfx.window([cache]) as forced:
        assert [kv8.serves(n) for n in rows] == [False, True, True, True]
        kv8.observe_first_row(cache, 1 << 20)
        assert forced.state == "idle"
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        # No ceiling bounds a forced window's rows or capacity.
        assert (mirror.n, mirror.cap) == (cache.offset, 5888)
        assert forced.state == "served"


def test_window_close_drops_every_mirror_and_wrap_and_reports_the_window(monkeypatch):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    kfx.small_floor(monkeypatch)
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    kv = [_mirror_cache(kv8.N8 + 8, seed=1), _mirror_cache(kv8.N8 + 8, seed=2)]
    arrays = [ArraysCache(size=2), ArraysCache(size=2)]
    layers = [arrays[0], kv[0], arrays[1], kv[1]]  # qwen3_5's layout: linear, full, ...
    with kfx.window() as opened:
        for cache in kv:
            _mirror_append(cache, 1, seed=3)
            kv8.prepare(cache, 1)
            assert getattr(vars(cache)["trim"], kv8._TRIM_MARK)
        assert (opened.state, opened.served, opened.built) == ("served", 2, True)
        assert kv8.close_window(layers) == ("served", 2 * (kv8.N8 + 9))
    assert kv8._window is None and cleared == [True]
    before = dict(kv8.calls)
    for cache in kv:
        assert _mirror_of(cache) is None and "trim" not in vars(cache)
        assert cache.trim.__func__ is KVCache.trim
        cache.trim(1)
    assert _mirror_counts_since(before) == {}
    assert not any(kv8.MIRROR_ATTR in vars(c) for c in arrays)


def test_window_close_with_nothing_built_leaves_mlx_cache_alone(monkeypatch):
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    with kfx.window():
        assert kv8.close_window([]) == ("idle", 0)
    assert cleared == []


def test_mirror_wrap_calls_and_restores_a_trim_of_the_caches_own(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    seen = []

    def own_trim(n):
        seen.append(n)
        cache.offset -= n
        return n

    object.__setattr__(cache, "trim", own_trim)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert mirror.original_trim is own_trim
        assert cache.trim(2) == 2
        assert seen == [2] and mirror.n == cache.offset == kv8.N8 + 7
    assert vars(cache)["trim"] is own_trim


def test_window_close_after_clear_still_drops_the_mirrors(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        kv8.clear()
        assert kv8.close_window([cache]) == ("off", 0)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)


def test_window_clear_drops_the_flag_and_the_window_only(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        serial = kv8._serial
        kv8.clear()
        assert (kv8._active, kv8._window, kv8._serial) == (0, None, serial)
        assert not kv8.serving_now()
        # Mirrors live on caches, which go with the engine.
        assert _mirror_of(cache) is not None
    kv8.close_window([cache])


@pytest.mark.parametrize(("limit", "cap"), [(84481, 5888), (2561, 2816)])
def test_mirror_first_served_call_builds_the_prefix_in_one_dispatch(monkeypatch, limit, cap):
    """At min(round_up(n + GROWTH, 256), round_up(L, 256)) rows: GROWTH rows of
    headroom, never past what the ceiling priced."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, limit)
    cache = _mirror_cache(kv8.N8 + 63)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache], force=False) as opened:
        _mirror_append(cache, 1, seed=1)
        n = cache.offset
        mirror = kv8.prepare(cache, 1)
        assert seen == [(n, cap)]
        assert (mirror.cap, mirror.n, mirror.serial) == (cap, n, opened.serial)
        assert (mirror.codes.shape, mirror.sc.shape) == ((1, 4, cap, 512), (1, 4, cap, 2))
        assert (mirror.codes.dtype, mirror.sc.dtype) == (mx.int8, mx.float16)
        assert (opened.state, opened.rows, opened.served, opened.built) == ("served", n, 1, True)
        assert _mirror_counts_since(before) == {"windows": 1, "mirror_builds": 1, "mirror_rows": n}
        assert _mirror_matches(cache, n)


def test_mirror_each_served_call_quantises_its_own_rows_once(monkeypatch):
    """A DFlash-shaped sequence: verify 5, reject 3 (a trim through the wrap),
    verify 5, decode 1, verify 8. Each update is one dispatch of exactly the
    call's rows."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 16)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache]) as opened:
        _mirror_append(cache, 5, seed=1)
        kv8.prepare(cache, 5)
        built = cache.offset
        assert seen == [(built, kv8._round_up(built + kv8.GROWTH))]
        cache.trim(3)
        assert _mirror_of(cache).n == cache.offset
        for t, seed in ((5, 2), (1, 3), (8, 4)):
            _mirror_append(cache, t, seed=seed)
            seen.clear()
            kv8.prepare(cache, t)
            assert seen == [(t, t)]
            assert _mirror_matches(cache, cache.offset)
        assert opened.rows == built + 5 + 1 + 8 and opened.served == 4
    assert _mirror_counts_since(before) == {
        "windows": 1,
        "mirror_builds": 1,
        "mirror_rows": opened.rows,
        "mirror_trims": 1,
    }


def test_mirror_trim_then_unserved_append_requantises_exactly_the_replaced_rows(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32)
    seen = _mirror_recording_quantize(monkeypatch)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        cache.trim(6)
        assert _mirror_of(cache).n == cache.offset == kv8.N8 + 27
        _mirror_append(cache, 4, seed=2)  # rows N8 + 27 .. N8 + 30 replaced, not served
        _mirror_append(cache, 1, seed=3)  # the served call's own row
        seen.clear()
        kv8.prepare(cache, 1)
        assert seen == [(5, 5)]
        assert _mirror_matches(cache, kv8.N8 + 32)


def test_mirror_trim_that_bypasses_the_wrap_leaves_stale_rows(monkeypatch):
    """The negative control for the test above: the served call's clamp alone
    re-quantises only from offset - t, so the replaced rows stay stale. The wrap
    is the mechanism, the clamp a defence."""
    from mlx_vlm.models.cache import KVCache

    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        KVCache.trim(cache, 6)
        _mirror_append(cache, 4, seed=2)
        _mirror_append(cache, 1, seed=3)
        kv8.prepare(cache, 1)
        assert _mirror_of(cache).n == cache.offset
        assert not _mirror_matches(cache, cache.offset)


def test_mirror_shrinking_block_across_the_floor_ends_equal_to_a_bulk_quantisation(
    monkeypatch,
):
    """16 rows served from N8, a rollback to below N8, 8 rows below N8 that no
    call serves, then a served call: it re-quantises the 8 replaced rows with
    its own."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8)
    seen = _mirror_recording_quantize(monkeypatch)
    with kfx.window([cache]):
        _mirror_append(cache, 16, seed=1)
        assert kv8.serves(kv8.N8 + 1)
        kv8.prepare(cache, 16)
        cache.trim(24)
        assert _mirror_of(cache).n == cache.offset == kv8.N8 - 8
        assert not kv8.serves(kv8.N8 - 7)
        _mirror_append(cache, 8, seed=2)
        _mirror_append(cache, 5, seed=3)
        seen.clear()
        kv8.prepare(cache, 5)
        assert seen == [(13, 13)]
        assert _mirror_matches(cache, kv8.N8 + 5)


def test_mirror_from_another_window_is_rebuilt_from_row_zero(monkeypatch):
    """A window closed without this cache leaves its mirror behind; a fork
    restore then replaces every row without a trim, and a prefill appends. The
    next window's first served call sees the old serial and rebuilds from row 0,
    reusing the wrap already in place."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32, seed=1)
    with kfx.window() as first:
        _mirror_append(cache, 1, seed=2)
        kv8.prepare(cache, 1)
    assert _mirror_of(cache).serial == first.serial
    wrap = vars(cache)["trim"]
    restored = _mirror_cache(kv8.N8 + 40, seed=3)
    cache.state = (restored.keys[:, :, : restored.offset], restored.values[:, :, : restored.offset])
    del restored
    _mirror_append(cache, 8, seed=4)
    before = dict(kv8.calls)
    with kfx.window([cache]) as second:
        _mirror_append(cache, 1, seed=5)
        mirror = kv8.prepare(cache, 1)
        assert second.serial > first.serial and mirror.serial == second.serial
        assert vars(cache)["trim"] is wrap and mirror.original_trim is None
        assert _mirror_counts_since(before) == {
            "windows": 1,
            "mirror_builds": 1,
            "mirror_rows": cache.offset,
        }
        assert _mirror_matches(cache, cache.offset)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)


def test_mirror_of_a_cache_whose_capacity_is_not_a_multiple_of_four(monkeypatch):
    """KVCache cuts its buffer to the offset before it grows, so a second
    prefill segment leaves 1,537 + 512 = 2,049 rows of capacity; the mirror's
    capacity is its own."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 1, 300)
    assert cache.keys.shape[2] == 2049
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert cache.keys.shape[2] == 2049 and mirror.cap % 256 == 0
        assert _mirror_matches(cache, cache.offset)
        _mirror_append(cache, 5, seed=2)
        kv8.prepare(cache, 5)
        assert _mirror_matches(cache, cache.offset)


def test_mirror_served_decode_reads_no_row_past_its_keys(monkeypatch):
    """Rows past n hold -128 codes and NaN scales; a second call at the same
    offset re-quantises only its own row and must not see them."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 40)
    queries = _mirror_queries(1, seed=1)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=2)
        clean = kv8.decode(queries, cache, _MIRROR_S)
        mx.eval(clean)
        mirror = _mirror_of(cache)
        n = cache.offset
        mirror.codes[:, :, n:] = -128
        mirror.sc[:, :, n:] = float("nan")
        mx.eval(mirror.codes, mirror.sc)
        again = kv8.decode(queries, cache, _MIRROR_S)
        mx.eval(again)
    assert mx.all(mx.isfinite(again)).item()
    assert mx.array_equal(again, clean).item()


def test_mirror_served_calls_read_it_as_a_bulk_quantisation_reads(monkeypatch):
    """decode() equals decode_codes() over a bulk quantisation, and each row of a
    verify run equals the one-row call at its own length."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 24)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        n = cache.offset
        q1 = _mirror_queries(1, seed=2)
        before = dict(kv8.calls)
        got = kv8.decode(q1, cache, _MIRROR_S)
        assert _mirror_counts_since(before) == {
            "decode_kv8": 1,
            "mirror_builds": 1,
            "mirror_rows": n,
        }
        codes, sc = _mirror_bulk(cache, n)
        assert got.shape == (1, tileattn.HQ, 1, tileattn.D)
        assert mx.array_equal(got, kv8.decode_codes(q1, codes, sc, n, _MIRROR_S)).item()

        _mirror_append(cache, 5, seed=3)
        prefix = cache.offset - 5
        ((j, k, chunk),) = tileattn.verify_plan(prefix, 5, _MIRROR_S)
        q5 = _mirror_queries(5, seed=4)
        before = dict(kv8.calls)
        mirror = kv8.prepare(cache, 5)
        part = kv8.verify_run(q5, mirror, prefix, j, k, chunk)
        assert _mirror_counts_since(before) == {"mirror_rows": 5}
        assert part.shape == (1, 5, tileattn.HQ, tileattn.D)
        codes, sc = _mirror_bulk(cache, cache.offset)
        for r in range(5):
            alone = kv8.decode_codes(q5[:, :, r : r + 1], codes, sc, prefix + r + 1, _MIRROR_S)
            assert mx.array_equal(part[:, r : r + 1].transpose(0, 2, 1, 3), alone).item(), r


def test_mirror_grows_exactly_once_when_a_decode_passes_its_capacity(monkeypatch):
    """GROWTH cut to 256 rows so 8-row calls cross the capacity in seconds:
    built at 2,048 rows, grown once to 2,560 at offset 2,053."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "GROWTH", 256)
    cache = _mirror_cache(kv8.N8 + 60)
    before = dict(kv8.calls)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        assert kv8.prepare(cache, 1).cap == 2048
        seed = 2
        while cache.offset < 2100:
            _mirror_append(cache, 8, seed=seed)
            mirror = kv8.prepare(cache, 8)
            mx.eval(mirror.codes, mirror.sc)
            seed += 1
        assert mirror.cap == 2560
        assert _mirror_counts_since(before) == {
            "windows": 1,
            "mirror_builds": 1,
            "mirror_rows": cache.offset,
            "mirror_grown": 1,
        }
        assert _mirror_matches(cache, cache.offset)


def test_mirror_capacity_stops_at_the_ceiling_and_no_row_past_it_is_written(monkeypatch):
    """L = 2,561 and a capacity of round_up(L, 256) = 2,816: a straddling verify
    grows the mirror to that and quantises rows up to L, not past."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 2561)
    monkeypatch.setattr(kv8, "GROWTH", 256)
    cache = _mirror_cache(kv8.N8 + 64)
    with kfx.window([cache], force=False):
        _mirror_append(cache, 1, seed=1)
        assert kv8.prepare(cache, 1).cap == 2048
        _mirror_append(cache, 700, seed=2)
        _mirror_append(cache, 1, seed=3)
        assert kv8.prepare(cache, 1).cap == 2560
        _mirror_append(cache, 2556 - cache.offset, seed=4)
        _mirror_append(cache, 8, seed=5)
        assert kv8.serves(2557) and not kv8.serves(2561)
        mirror = kv8.prepare(cache, 8)
        assert (mirror.cap, mirror.n, cache.offset) == (2816, 2561, 2564)
        assert _mirror_matches(cache, 2561)


def test_mirror_update_writes_in_place(monkeypatch):
    """The mirror's buffers are referenced from the Mirror alone, so mlx donates
    them to the two slice updates: an update's peak is its own rows, never a
    copy of the 12 MB mirror."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 63)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        mx.eval(mirror.codes, mirror.sc)
        size = mirror.codes.nbytes + mirror.sc.nbytes
        _mirror_append(cache, 5, seed=2)
        mx.synchronize()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mirror = kv8.prepare(cache, 5)
        mx.eval(mirror.codes, mirror.sc)
        peak = mx.get_peak_memory() - base
        assert _mirror_matches(cache, cache.offset)
    assert size > 12_000_000
    assert peak < size // 8, (peak, size)


def test_mirror_growth_holds_one_layers_old_buffer_at_a_time(monkeypatch):
    """Six layers grow in one forward, each from 5,888 to 10,240 rows. Each
    layer's new buffer is evaluated before its old one is dropped, so the peak
    passes the grown total by about one layer's buffer; grown lazily, all six
    old buffers would be alive at once."""
    kfx.small_floor(monkeypatch)
    caches = [_mirror_cache(kv8.N8 + 64, seed=s) for s in range(6)]
    row = kv8.BYTES_PER_ROW_PER_HEAD  # 512 codes and two fp16 scales
    before = dict(kv8.calls)
    with kfx.window(caches):
        for cache in caches:
            _mirror_append(cache, 1, seed=10)
            kv8.prepare(cache, 1)
            _mirror_append(cache, 4280, seed=11)
            mirror = kv8.prepare(cache, 4280)
            mx.eval(mirror.codes, mirror.sc)
            assert (mirror.cap, mirror.n) == (5888, cache.offset)
        for cache in caches:
            _mirror_append(cache, 16, seed=12)
        mx.synchronize()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        for cache in caches:
            kv8.prepare(cache, 16)
        mx.eval([(_mirror_of(c).codes, _mirror_of(c).sc) for c in caches])
        peak = mx.get_peak_memory() - base
        assert all(_mirror_of(c).cap == 10240 for c in caches)
    grown = 6 * (10240 - 5888) * tileattn.HKV * row
    layer = 10240 * tileattn.HKV * row
    assert _mirror_counts_since(before)["mirror_grown"] == 6
    assert peak - grown < 2 * layer, (peak, grown, layer)


def test_cap_a_call_at_the_ceiling_caps_once_and_drops_each_layers_mirror(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    first, second = _mirror_cache(kv8.N8 + 64, seed=1), _mirror_cache(kv8.N8 + 64, seed=2)
    before = dict(kv8.calls)
    with kfx.window([first, second], force=False) as opened:
        for cache in (first, second):
            _mirror_append(cache, 1, seed=3)
            kv8.prepare(cache, 1)
        kv8.observe_first_row(first, 1920)  # below the ceiling: nothing happens
        assert opened.state == "served" and _mirror_of(first) is not None
        kv8.observe_first_row(first, 1921)
        assert opened.state == "capped"
        assert _mirror_of(first) is None and "trim" not in vars(first)
        assert _mirror_of(second) is not None
        assert not kv8.serving_now() and not kv8.serves(kv8.N8 + 100)
        kv8.observe_first_row(second, 1925)
        assert _mirror_of(second) is None and "trim" not in vars(second)
        assert kv8.close_window([first, second]) == ("capped", 2 * (kv8.N8 + 65))
    counts = _mirror_counts_since(before)
    assert counts["windows_capped"] == 1 and counts["mirror_builds"] == 2


def test_cap_a_window_whose_first_call_is_past_the_ceiling_has_no_rows(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    cache = _mirror_cache(1930)
    with kfx.window([cache], force=False):
        _mirror_append(cache, 1, seed=1)
        assert not kv8.serves(cache.offset)
        kv8.observe_first_row(cache, cache.offset)
        assert kv8.close_window([cache]) == ("capped", 0)


def test_cap_a_straddling_verify_keeps_its_mirror_and_serves_only_rows_below_it(monkeypatch):
    """L = 1,921, a step boundary at 20 splits. An 8-row verify behind 1,915 keys
    is cut there: its first run (1,916 to 1,920 keys) reads the mirror, its
    second is the tile's, and the mirror holds no row at or past L. The commit
    trims through the wrap; the decode whose row is at L caps the window."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(1915)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache], force=False) as opened:
        _mirror_append(cache, 8, seed=1)
        prefix = 1915
        kv8.observe_first_row(cache, prefix + 1)
        runs = tileattn.verify_plan(prefix, 8, _MIRROR_S)
        assert runs == [(0, 5, 96), (5, 8, 128)]
        assert [kv8.serves(prefix + j + 1) for j, _, _ in runs] == [True, False]
        mirror = kv8.prepare(cache, 8)
        assert seen == [(1921, 2048)]
        assert (mirror.n, mirror.cap) == (1921, 2048)
        queries = _mirror_queries(8, seed=2)
        part = kv8.verify_run(queries, mirror, prefix, 0, 5, 96)
        codes, sc = _mirror_bulk(cache, 1921)
        for r in range(5):
            n = prefix + r + 1
            alone = kv8.decode_codes(queries[:, :, r : r + 1], codes, sc, n, _MIRROR_S)
            assert mx.array_equal(part[:, r : r + 1].transpose(0, 2, 1, 3), alone).item(), r
        assert opened.state == "served" and _mirror_of(cache) is mirror
        cache.trim(3)  # 4 accepted and the bonus row: 1,920 keys committed
        assert mirror.n == cache.offset == 1920
        _mirror_append(cache, 1, seed=3)
        assert not kv8.serves(cache.offset)
        kv8.observe_first_row(cache, cache.offset)
        assert opened.state == "capped" and _mirror_of(cache) is None
        assert kv8.close_window([cache]) == ("capped", 1921)
    assert _mirror_counts_since(before) == {
        "windows": 1,
        "windows_capped": 1,
        "mirror_builds": 1,
        "mirror_rows": 1921,
        "mirror_trims": 1,
    }


def test_mirror_never_travels_with_a_fork(monkeypatch):
    """`fork_copy` builds a fresh cache from `state` and `meta_state`, so the
    mirror and its trim wrap stay with the cache that was served: a fork
    published to a slot or the store holds bf16 K/V and the class's `trim`,
    and trimming the fork leaves the served cache's mirror alone."""
    from mlx_vlm.models.cache import KVCache

    from sous.engine.promptcache import fork_copy

    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 32)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mx.eval(kv8.decode(_mirror_queries(1, seed=2), cache, _MIRROR_S))
        assert _mirror_matches(cache, kv8.N8 + 33)
        fork = KVCache()
        fork_copy([cache], [fork], mx.array)
        mx.eval(fork.keys, fork.values)
        assert fork.offset == cache.offset
        assert _mirror_of(fork) is None
        assert "trim" not in vars(fork), "the fork's trim is the class's"
        assert fork.trim(8) == 8
        assert _mirror_matches(cache, kv8.N8 + 33), "the served cache's mirror is untouched"
    assert _mirror_of(cache) is None


def test_mirror_trim_past_the_offset_lowers_n_to_the_new_offset(monkeypatch):
    """`KVCache.trim` clamps the count to the offset and returns what it
    trimmed, and the wrap reads the offset afterwards: a rollback asked for
    more rows than the cache holds (an abort after nothing was appended) and
    a trim of zero both leave n equal to the offset, never below it."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    before = dict(kv8.calls)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert mirror.n == kv8.N8 + 9
        assert cache.trim(0) == 0
        assert mirror.n == cache.offset == kv8.N8 + 9
        assert cache.trim(kv8.N8 + 100) == kv8.N8 + 9
        assert mirror.n == cache.offset == 0
    assert _mirror_counts_since(before)["mirror_trims"] == 2
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_kv8.py -v -k "test_window_ or test_mirror_ or test_ceiling_ or test_floor_ or test_serves_ or test_cap_"`
Expected: FAIL, 32 failed. Every failure is an `AttributeError` naming an attribute this task adds to `sous.engine.kv8`: `_active`, `_ceiling`, `calls`, `ceiling` or `_floor_reason`.

- [ ] **Step 4: Add the window, the mirror, the ceiling and the floor**

In `src/sous/engine/kv8.py`, add each of these imports that Task 1's import block does not already have. `uv run ruff check --fix src/sous/engine/kv8.py` restores isort order. None of them imports mlx at module load.

```python
import dataclasses
import weakref
from collections.abc import Callable, Sequence
from typing import Any

from sous.engine import tileattn
```

Append to the end of the file, after Task 1's code:

```python
# --------------------------------------------------------------------------------
# The serving window, the mirror and its trim wrap, the ceiling and the floor
# --------------------------------------------------------------------------------

# A mirror's capacity is a multiple of this many rows: the kernel's 16-byte
# scale loads need each head plane of `sc` (cap x 4 bytes) to start 16-byte
# aligned, and 256 is KVCache's own step.
_CAP_STEP = 256
# Set on the trim wrap, so a drop puts back only a trim this module installed.
_TRIM_MARK = "_sous_kv8_trim"


@dataclasses.dataclass
class Window:
    """One decode's serving window. `state` is "idle" until a call is served,
    then "served", and "capped" once a call's first row is at or past the
    ceiling, whether or not anything was served before it (`rows` may then be
    0). `rows` counts the rows quantised in this window, `served` the calls
    served from a mirror, `built` whether any mirror was built. `force` lifts
    the ceiling: for the load-time probe and the tests only."""

    serial: int
    state: str = "idle"
    rows: int = 0
    served: int = 0
    built: bool = False
    force: bool = False


@dataclasses.dataclass
class Mirror:
    """The int8 copy of one full-attention layer's committed K/V, kept on its
    KVCache as MIRROR_ATTR. `codes` int8 [1, 4, cap, 512] holds each token's 256
    K codes then its 256 V codes, `sc` fp16 [1, 4, cap, 2] its {sK, sV}; rows
    [0, n) are valid and no read goes past n. `serial` is the window it was built
    in. `original_trim` is the cache's own `trim` attribute from before the wrap,
    or None when the class's method was in effect: dropping the mirror puts it
    back."""

    codes: Any
    sc: Any
    cap: int
    n: int
    serial: int
    original_trim: Any


# 1 only after enable() passed every gate and the probe: every enable() clears
# it first, clear() clears it, and the probe raises it only while it runs. A
# plain int, so nothing it holds outlives an unload.
_active: int = 0
# The ceiling L: kv8 serves only rows below it. enable() sets it from ceiling();
# 0 serves nothing outside a forced window.
_ceiling: int = 0
# The open serving window, or None. One module global is enough: the engine's
# generation lock serialises every forward, and every decode() runs on the
# caller's session thread.
_window: Window | None = None
# The newest window's serial. Never reset, so a mirror an earlier window left on
# a cache, an earlier load's included, is stale in every later window.
_serial: int = 0
# Per attention call or per window, not per turn: the tests read deltas, and so
# can a harness that wraps the daemon. Nothing is logged per turn.
calls: dict[str, int] = {
    "decode_kv8": 0,  # one-row calls served from the mirror
    "verify_kv8": 0,  # verify calls with at least one run served (the verify seam counts)
    "verify_rows_kv8": 0,  # verify rows served from the mirror (the verify seam counts)
    "verify_launches_kv8": 0,  # kv8 kernel runs of those calls (the verify seam counts)
    "mirror_builds": 0,  # mirrors quantised from row 0
    "mirror_rows": 0,  # rows quantised into a mirror, builds included
    "mirror_trims": 0,  # KVCache.trim calls through the wrap
    "mirror_grown": 0,  # capacity growths
    "windows": 0,  # windows opened
    "windows_capped": 0,  # windows that met a call at or past the ceiling
}


def _round_up(rows: int) -> int:
    return -(-rows // _CAP_STEP) * _CAP_STEP


def ceiling(window: int, splits: int) -> int:
    """L for a `window`-token reserve at `splits` splits: the largest tile step
    boundary b whose mirror fits beside the turn's bf16 KV at every instant,
    else 0. With C = b rounded up to the capacity step, the rule is
    1.504 C + 0.504 C / 16 + 272 <= window: the bf16 KV up to C and the mirror
    at C (516 bytes per token per head against bf16's 1,024), one of the 16
    full-attention layers' old mirror while a growth copies it, and the 16 rows
    a straddling verify holds past L plus one capacity step. Being a step
    boundary, L is a verify_plan cut, so no run straddles it."""
    best = 0
    for b in tileattn.boundaries(window, splits):
        c = _round_up(b)
        # The rule in thousandths, so a boundary on the margin is decided in exact
        # integers (C // 16 is exact: C is a multiple of 256).
        if 1504 * c + 504 * (c // 16) + 272_000 <= 1000 * window:
            best = b
    return best


def _floor_reason(splits: int) -> str | None:
    """Why N8 cannot be the floor at `splits` splits, or None. verify_plan cuts
    at every step boundary, so a floor on one keeps every verify run wholly on
    one side of it; a boundary at one S need not be one at another."""
    if N8 in tileattn.boundaries(N8, splits):
        return None
    return f"floor {N8} keys is not a tile step boundary at {splits} splits"


def open_window(*, force: bool = False) -> Window:
    """Open one decode's serving window: until close_window(), serves() may
    answer True and served calls build and read mirrors. `force` serves every
    row from N8 whatever the ceiling."""
    global _window, _serial
    assert _window is None, "a kv8 serving window is already open"
    _serial += 1
    _window = Window(serial=_serial, force=force)
    calls["windows"] += 1
    return _window


def close_window(cache: Sequence[Any]) -> tuple[str, int]:
    """Close the window: drop every mirror and trim wrap in `cache` (the
    decode's per-layer caches; layers without a mirror are passed over) and
    return the window's state and the rows it quantised. When a mirror was
    built, mlx's buffer cache is emptied: the mirrors' freed buffers would sit
    there through the publish that follows the decode. With no window open
    (clear() ran) the mirrors still go and the answer is ("off", 0): this runs
    in a finally, where raising would mask the error in flight."""
    global _window
    window = _window
    _window = None
    for layer in cache:
        _drop(layer)
    if window is None:
        return "off", 0
    if window.built:
        import mlx.core as mx

        mx.clear_cache()
    return window.state, window.rows


def serving_now() -> bool:
    """kv8 is active and a window is open that no call has capped."""
    window = _window
    return bool(_active) and window is not None and window.state != "capped"


def serves(n_row: int) -> bool:
    """Whether the row that sees n_row keys reads the mirror: N8 <= n_row < L in
    an open, uncapped window (a forced one serves from N8 whatever L). A row's
    path depends on its own key count alone, so a drafter-on and a drafter-off
    decode of one request serve the same rows. The seams look this up through
    the module on every call, so a harness may wrap it."""
    window = _window
    if not _active or window is None or window.state == "capped":
        return False
    return n_row >= N8 and (n_row < _ceiling or window.force)


def observe_first_row(cache: Any, n_first: int) -> None:
    """Both seams report an in-scope call's first row's key count here: the
    verify seam for every call, before its runs, the decode seam for each call
    it does not serve. At or past the ceiling the window caps, once, and this
    layer's mirror goes (each layer's own call drops its own): the committed
    prefix never shrinks inside a decode, so no later call of it could be
    served. A forced window never caps."""
    window = _window
    if window is None or window.force or n_first < _ceiling:
        return
    if window.state != "capped":
        window.state = "capped"
        calls["windows_capped"] += 1
    _drop(cache)


def prepare(cache: Any, t: int) -> Mirror:
    """`cache`'s mirror brought up to date for a served call that has just
    appended t rows; once per served call, before any of its runs. Rows below
    offset - t are the committed prefix, so n is clamped there, and then rows
    [n, min(offset, L)) are quantised: no row at or past L is ever written. The
    first served call of a window on this layer, or one meeting a mirror from
    another window, builds it from row 0 instead; rows past the capacity grow
    it first. `cache` is one layer's KVCache."""
    window = _window
    assert window is not None, "kv8.prepare outside a serving window"
    offset = cache.offset
    top = offset if window.force else min(offset, _ceiling)
    window.served += 1
    if window.state == "idle":
        window.state = "served"
    mirror = getattr(cache, MIRROR_ATTR, None)
    if mirror is None or mirror.serial != window.serial:
        return _build(cache, window, top)
    # A defence: inside a window the trim wrap has already lowered n for every
    # rollback.
    mirror.n = min(mirror.n, offset - t)
    if top > mirror.cap:
        _grow(mirror, _capacity(offset, window))
    start = mirror.n
    if top > start:
        codes, sc = quantize(
            cache.keys[:, :, start:top, :], cache.values[:, :, start:top, :], top - start
        )
        # The Mirror holds the only reference to its buffers, so mlx donates them
        # and both updates write in place.
        mirror.codes[:, :, start:top, :] = codes
        mirror.sc[:, :, start:top, :] = sc
        mirror.n = top
        _count_rows(window, top - start)
    return mirror


def _build(cache: Any, window: Window, top: int) -> Mirror:
    """Rows [0, top) quantised in one dispatch straight into a fresh mirror at
    the window's capacity (rows past top are left unwritten: no read reaches
    them), and the cache's trim wrapped. A mirror from another window goes
    first, but its wrap stays and serves the new one."""
    attrs = vars(cache)
    stale = attrs.pop(MIRROR_ATTR, None)
    if stale is None:
        original = attrs.get("trim")
        object.__setattr__(cache, "trim", _wrap_trim(cache, original))
    else:
        original = stale.original_trim
    # A stale mirror's buffers are freed before the new ones are allocated.
    del stale
    cap = _capacity(cache.offset, window)
    codes, sc = quantize(cache.keys[:, :, :top, :], cache.values[:, :, :top, :], cap)
    mirror = Mirror(
        codes=codes, sc=sc, cap=cap, n=top, serial=window.serial, original_trim=original
    )
    object.__setattr__(cache, MIRROR_ATTR, mirror)
    calls["mirror_builds"] += 1
    window.built = True
    _count_rows(window, top)
    return mirror


def _capacity(offset: int, window: Window) -> int:
    """GROWTH rows of headroom past the offset, so a decode grows at most once
    per GROWTH generated tokens, and never past L rounded up, the size
    ceiling() priced (a forced window has no L)."""
    cap = _round_up(offset + GROWTH)
    return cap if window.force else min(cap, _round_up(_ceiling))


def _grow(mirror: Mirror, cap: int) -> None:
    """One layer's mirror widened to `cap` rows. The new buffers are evaluated
    before the old ones are dropped, so the transient is this layer's old
    buffer: left lazy, every layer's old buffer would live until the forward
    is evaluated."""
    import mlx.core as mx

    extra = cap - mirror.cap
    codes_tail = mx.zeros((1, tileattn.HKV, extra, 2 * tileattn.D), dtype=mx.int8)
    sc_tail = mx.zeros((1, tileattn.HKV, extra, 2), dtype=mx.float16)
    codes = mx.concatenate([mirror.codes, codes_tail], axis=2)
    sc = mx.concatenate([mirror.sc, sc_tail], axis=2)
    mx.eval(codes, sc)
    mirror.codes, mirror.sc, mirror.cap = codes, sc, cap
    calls["mirror_grown"] += 1


def _count_rows(window: Window, rows: int) -> None:
    calls["mirror_rows"] += rows
    window.rows += rows


def _wrap_trim(cache: Any, original: Any) -> Callable[[int], Any]:
    """`cache`'s trim, then its mirror's n lowered to the new offset. Inside a
    decode only KVCache.trim lowers an offset, and DFlash's commit and abort
    both reach it as an instance call, so this wrap is what keeps a mirror
    valid across rollbacks. `original` is the instance's own trim from before,
    or None for the class's. The cache is held weakly: held strongly from its
    own attribute, it would be a reference cycle whose K/V waits for the cycle
    collector."""
    ref = weakref.ref(cache)
    method = type(cache).trim

    def trim(n: int) -> Any:
        target: Any = ref()
        trimmed = method(target, n) if original is None else original(n)
        mirror = getattr(target, MIRROR_ATTR, None)
        if mirror is not None:
            mirror.n = min(mirror.n, target.offset)
        calls["mirror_trims"] += 1
        return trimmed

    setattr(trim, _TRIM_MARK, True)
    return trim


def _drop(cache: Any) -> None:
    """`cache`'s mirror removed and the trim it wrapped put back. A cache
    without one (a linear-attention layer's, or one never served) is passed
    over, and a trim someone set over the wrap since is left alone."""
    attrs = getattr(cache, "__dict__", None)
    if attrs is None:
        return
    mirror = attrs.pop(MIRROR_ATTR, None)
    if mirror is None or not getattr(attrs.get("trim"), _TRIM_MARK, False):
        return
    if mirror.original_trim is None:
        del attrs["trim"]
    else:
        attrs["trim"] = mirror.original_trim


def decode(queries: Any, cache: Any, splits: int) -> Any:
    """A served one-row call: the mirror brought up to date with the call's own
    row, then the kernel over all n = cache.offset keys, [1, 24, 1, 256] bf16."""
    mirror = prepare(cache, 1)
    calls["decode_kv8"] += 1
    return decode_codes(queries, mirror.codes, mirror.sc, cache.offset, splits)


def verify_run(queries: Any, mirror: Mirror, prefix: int, j: int, k: int, chunk: int) -> Any:
    """One served run of a T-row verify call behind `prefix` keys: rows [j, k)
    of `queries` [1, 24, T, 256], row r over keys [0, prefix + r], read from the
    mirror prepare() returned for the call. Returns the contiguous
    [1, k - j, 24, 256] a tile run returns, for the caller to concatenate along
    axis 1. A launch only: the verify seam counts the call, its rows and its
    launches."""
    t = queries.shape[2]
    rows = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
    return tile8(rows, mirror.codes, mirror.sc, prefix + k, chunk)


def clear() -> None:
    """Stop serving (VLMEngine.unload()): the flag and any open window go.
    Mirrors live on caches that go with the engine, so nothing else is held."""
    global _active, _window
    _active = 0
    _window = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kv8.py -v -k "test_window_ or test_mirror_ or test_ceiling_ or test_floor_ or test_serves_ or test_cap_"`
Expected: PASS, 32 passed, in about 2 s on the M2.

- [ ] **Step 6: Run the neighbouring suites, then everything**

Nothing outside `kv8` calls the new functions yet. These runs show that the new module state and the fixtures leak nothing into Task 1's tests or the tile's.

Run: `uv run pytest tests/test_kv8.py tests/test_tileattn.py tests/test_tileattn_contract.py -v`
Expected: PASS. The only skips are the `nax`-marked tests, Task 1's and the tile's, where the tensor units are absent.

Run: `uv run pytest -m "not model"`
Expected: PASS.

Run both under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py` when another session may be using the GPU.

- [ ] **Step 7: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/kv8.py tests/kv8_fixtures.py tests/test_kv8.py
git commit -m "feat(engine): keep a per-decode int8 mirror of each layer's KV, valid across rollbacks" -m "kv8 reads int8 K/V from a mirror that lives on each full-attention layer's
KVCache for one decode only, so prefill, the forks, the fork store and the
slots keep seeing bf16. Inside a decode only KVCache.trim lowers an offset,
and DFlash's commit and abort both reach it as an instance call. The mirror
therefore follows the offset through an instance wrap of trim, so rows a
rejected draft left behind are quantised again once they are replaced. A
served call also clamps to its committed prefix, and a mirror from another
window is rebuilt from row 0.

A row is served only from the floor N8 to below the ceiling L. L is the
last tile step boundary whose mirror fits beside the turn's bf16 KV in the
window-sized reserve, so the slot budget does not change. A row's path
depends on its own key count alone, so drafter-on and drafter-off decodes
serve the same rows, and a call at or past L caps the window for the rest
of the decode.

A growth evaluates each layer's new buffer before it drops the old one, and
updates are slice assignments that mlx performs in place. The tests bound
both by peak memory."
```


### Task 3: The seams: the decode hook and `serve_verify` route rows to kv8

**Files:**
- Modify: `src/sous/engine/tileattn.py` (`verify`, lines 248–281, becomes `_verify_runs` plus a thin `verify`; `_serve_decode`, lines 362–388; `serve_verify`, lines 416–431)
- Modify: `src/sous/engine/verifyattn.py` (line 438, inside `_wrap`: the `serve_verify` call passes `cache`)
- Test: `tests/test_kv8.py` (imports; the dispatch section appended after the earlier tasks' sections)

**Interfaces:**
- Consumes (built by Tasks 1–2, names and signatures as the interface sheet fixes them):
  - `kv8.N8` (module int, read at call time; tests monkeypatch it), `kv8._active`, `kv8._window`, `kv8._ceiling`, `kv8.MIRROR_ATTR = "_sous_kv8"`.
  - `kv8.calls`, with the keys `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8` and `windows_capped` among them.
  - `kv8.serves(n_row: int) -> bool`: `serving_now() and n_row >= N8 and (n_row < _ceiling or _window.force)`.
  - `kv8.observe_first_row(cache, n_first: int) -> None`: with a window open and not forced and `n_first >= _ceiling`, sets the window's state to `"capped"`, adds 1 to `calls["windows_capped"]` the first time in that window, and drops this cache's mirror, after which `getattr(cache, kv8.MIRROR_ATTR, None) is None`. Without a window it does nothing.
  - `kv8.decode(queries, cache, splits) -> [1, 24, 1, 256]` bf16: `prepare(cache, 1)`, then the kernel over the mirror; adds 1 to `calls["decode_kv8"]`. After it, `getattr(cache, kv8.MIRROR_ATTR).n == cache.offset`.
  - `kv8.prepare(cache, t) -> Mirror`: clamps the mirror to the committed prefix, then quantises rows `[n, min(cache.offset, _ceiling))` (from row 0 on the window's first build), reaching the quantiser only through the module global `quantize`, looked up at call time.
  - `kv8.verify_run(queries, mirror, prefix, j, k, chunk) -> [1, k - j, 24, 256]` bf16, contiguous, the tile's layout: launch only, it counts nothing. `queries` is the call's whole `[1, 24, T, 256]`; it takes rows `[j, k)` itself, over `n = prefix + k` keys of the mirror.
  - `kv8.quantize(k, v, out_rows) -> (codes, sc)`, `kv8.tile8(queries, codes, sc, n, chunk)`, `kv8.open_window(*, force=False) -> Window` (asserts no window is open), `kv8.close_window(cache: Sequence) -> tuple[str, int]` (afterwards `getattr(c, kv8.MIRROR_ATTR, None) is None` for every cache it was handed), `Window.state`.
  - `tests/kv8_fixtures.py`: `stand_in_tile8(queries, codes, sc, n, chunk)` with `tile8`'s contract (row t of a T-row call sees keys `[0, n - T + t]`, and a row's output depends on its chunk and nothing else about the call); `ceiling(monkeypatch, L)`, which sets `kv8._ceiling` to L for the test; `small_floor(monkeypatch)`, which sets `kv8.N8` to `tileattn.N0`.
  - `tests/tile_fixtures.py` (existing): `hooked`, `tiny_language_model`, `prefill_cache`, `verify_ready`, `verify_forward`, `VOCAB`.
- Produces:
  - `tileattn._verify_runs(queries, keys, values, runs, served=None) -> [1, 24, T, 256]`: the one run loop. `served`, when given, is `(j, k, chunk) -> part | None`, asked only for runs with chunk > 0; a None answer means the tile serves that run.
  - `tileattn.verify(queries, keys, values, splits, runs=None)`: unchanged signature and output, now `_verify_runs` without `served`.
  - `tileattn.serve_verify(queries, keys, values, splits, cache)`: `cache` is the layer's `KVCache`, the one the verify rows were just appended to.
  - Counter ownership. `decode_tile` moves only when the tile itself serves a decode. `serve_verify` adds 1 to `kv8.calls["verify_kv8"]` per call with any kv8 run, plus `verify_rows_kv8` (rows) and `verify_launches_kv8` (runs) for those runs. It still counts `verify_calls`, `verify_t*` and `verify_straddles` for every call, as before, and counts `verify_rows_tile` and `verify_launches` only for the tile's own runs.
  - `verifyattn._wrap` hands its `cache` argument to `serve_verify`.

The routing rule each seam applies, in this order:
- **Decode** (one row, n = `keys.shape[2]` keys): below N0 the stock path, as today. Otherwise, if `kv8.serves(n)`, kv8. Otherwise `kv8.observe_first_row(cache, n)`, then the tile.
- **Verify** (T rows behind `prefix = keys.shape[2] - T`): `kv8.observe_first_row(cache, prefix + 1)` comes first, so a call whose first row is at or past the ceiling caps the window before any run asks `serves`. Each run of `verify_plan` is then:
  - stock, when its chunk is 0;
  - kv8, when `kv8.serves(prefix + j + 1)`, its first row's key count;
  - the tile, otherwise.

  Because the floor and the ceiling are step boundaries and `verify_plan` cuts at every step boundary, every row of a run gets its first row's answer. A call with any kv8 run calls `kv8.prepare(cache, T)` once, before the first launch. A call that straddles the ceiling keeps its mirror: only a call whose first row is at or past the ceiling drops it.

The tests put the floor and the ceiling on the second and third tile step boundaries at S = 20: `N8 = 1921` and `L = 2561`. With those lines, each case can put a call below N0 (stock), on the tile below the floor, on kv8, and past the ceiling. The kernels are the stand-ins (`tfx.stand_in_tile` for the tile, `kfx.stand_in_tile8` for kv8). The quantiser is the real one: it is MPP-free and runs on CI's GPU. Every window is opened unforced, the way `VLMEngine.decode()` opens it.

How each path is checked:
- **A kv8 row** must equal the stand-in's one-row call at its own key count, run over the real quantiser's codes for that row's keys. That holds bit for bit because the mirror is a pure per-row function of the bf16 rows.
- **A tile row or a stock row** must equal `tileattn.decode` at its own key count. That is the tile's existing contract.

- [ ] **Step 1: Write the failing tests**

The section needs the lines below in the header of `tests/test_kv8.py`. Add each one the header does not already have, then run `uv run ruff check --fix tests/test_kv8.py` to put the block back in isort order.

```python
import contextlib

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import kv8, tileattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402
```

Append to the end of `tests/test_kv8.py`. Every name in the section carries a `dispatch` prefix so it cannot collide with the earlier sections' helpers. The run command selects the section with `-k dispatch`.

```python
# ---- dispatch: the two seams route each row to stock, the tile or kv8 ---------------

# The second and third tile step boundaries at S = 20: below the floor the tile
# serves (from N0), from the floor to the ceiling kv8, from the ceiling the tile.
DISPATCH_S = 20
DISPATCH_FLOOR = 1921
DISPATCH_CEIL = 2561
# The counters the seams move in kv8.calls; the mirror's own counters belong to
# the mirror's tests.
_DISPATCH_KV8_KEYS = (
    "decode_kv8",
    "verify_kv8",
    "verify_rows_kv8",
    "verify_launches_kv8",
    "windows_capped",
)


@pytest.fixture
def dispatch_lines(monkeypatch):
    """The tile's flag at S = 20 with the stand-in tile, kv8's stand-in kernel, the
    floor at 1921 and the ceiling at 2561, all for this test only."""
    assert tileattn.boundaries(DISPATCH_CEIL, DISPATCH_S) == [
        tileattn.N0,
        DISPATCH_FLOOR,
        DISPATCH_CEIL,
    ]
    tfx.hooked(monkeypatch, splits=DISPATCH_S)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    monkeypatch.setattr(kv8, "N8", DISPATCH_FLOOR)
    kfx.ceiling(monkeypatch, DISPATCH_CEIL)


@contextlib.contextmanager
def _dispatch_open(monkeypatch, caches):
    """An unforced serving window over `caches`, as VLMEngine.decode() opens one
    around a generation, closed on the way out whatever the test asserted."""
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)
    window = kv8.open_window()
    try:
        yield window
    finally:
        kv8.close_window(caches)


def _dispatch_rows(t, *, seed):
    k_key, v_key = mx.random.split(mx.random.key(seed))
    shape = (1, tileattn.HKV, t, tileattn.D)
    keys = mx.random.normal(shape, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(shape, key=v_key).astype(mx.bfloat16)
    return keys, values


def _dispatch_cache(n, *, seed=0):
    """A KVCache holding n random bf16 rows, appended in one update as a prefill
    segment appends them."""
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.update_and_fetch(*_dispatch_rows(n, seed=seed))
    mx.eval(cache.keys, cache.values)
    return cache


def _dispatch_append(cache, t, *, seed):
    """t more rows through update_and_fetch, as a forward's projections append
    them; returns the K/V views the attention call receives."""
    keys, values = cache.update_and_fetch(*_dispatch_rows(t, seed=seed))
    mx.eval(keys, values)
    return keys, values


def _dispatch_queries(t, *, seed):
    """[1, 24, T, 256] bf16 query rows, a transposed [1, T, 24, 256] pool as
    q_norm and rope hand them over."""
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=mx.random.key(seed))
    pool = pool.astype(mx.bfloat16)
    mx.eval(pool)
    return pool.transpose(0, 2, 1, 3)


def _dispatch_decode(cache, *, seed):
    """One decode step: a row appended, then the decode hook's decision on it. An
    output of None means the hook handed the call to mlx-vlm's own function."""
    keys, values = _dispatch_append(cache, 1, seed=seed)
    q = _dispatch_queries(1, seed=seed + 1)
    call = {"cache": cache, "scale": tileattn.SCALE, "mask": None}
    return q, keys, values, tileattn._serve_decode((q, keys, values), call)


def _dispatch_snapshot():
    return dict(tileattn.calls), dict(kv8.calls)


def _dispatch_moved(before):
    """(tileattn counters, kv8's seam counters) that moved since `before`, by how much."""
    tile_before, kv8_before = before
    tile = {k: n - tile_before[k] for k, n in tileattn.calls.items() if n != tile_before[k]}
    ours = {
        k: kv8.calls[k] - kv8_before[k]
        for k in _DISPATCH_KV8_KEYS
        if kv8.calls[k] != kv8_before[k]
    }
    return tile, ours


def _dispatch_mirror(cache):
    return getattr(cache, kv8.MIRROR_ATTR, None)


def _dispatch_kv8_row(q, keys, values, m):
    """The one-row kv8 call at key count m over the real quantiser's codes of keys
    [0, m): what a row kv8 serves must equal bit for bit."""
    codes, sc = kv8.quantize(keys[:, :, :m], values[:, :, :m], m)
    chunk = tileattn.chunk_for(m, DISPATCH_S)
    return kfx.stand_in_tile8(q, codes, sc, m, chunk).transpose(0, 2, 1, 3)


def _dispatch_tile_row(q, keys, values, m):
    """The one-row call at key count m without kv8: the tile from N0, stock below."""
    return tileattn.decode(q, keys[:, :, :m], values[:, :, :m], DISPATCH_S)


def _dispatch_assert_rows(out, q, keys, values, prefix, kv8_rows):
    """Every verify row r equals its one-row call at prefix + r + 1 keys: kv8's for
    the key counts in `kv8_rows`, the tile's or stock's for the rest."""
    assert out.shape == (1, tileattn.HQ, q.shape[2], tileattn.D) and out.dtype == mx.bfloat16
    for r in range(q.shape[2]):
        m = prefix + r + 1
        qr = q[:, :, r : r + 1]
        if m in kv8_rows:
            want = _dispatch_kv8_row(qr, keys, values, m)
        else:
            want = _dispatch_tile_row(qr, keys, values, m)
        assert mx.array_equal(out[:, :, r : r + 1], want).item(), (prefix, r, m)


@pytest.mark.parametrize(
    ("n", "path"),
    [
        pytest.param(tileattn.N0 - 1, "stock", id="below N0"),
        pytest.param(tileattn.N0, "tile", id="at N0"),
        pytest.param(DISPATCH_FLOOR - 1, "tile", id="below the floor"),
        pytest.param(DISPATCH_FLOOR, "kv8", id="at the floor"),
        pytest.param(DISPATCH_CEIL - 1, "kv8", id="below the ceiling"),
    ],
)
def test_dispatch_a_decode_takes_its_path_by_key_count(dispatch_lines, monkeypatch, n, path):
    """Stock below N0, the tile from N0 to the floor, kv8 from the floor up to the
    ceiling; only a kv8 call builds a mirror, and decode_tile counts only the
    tile's own calls."""
    cache = _dispatch_cache(n - 1)
    with _dispatch_open(monkeypatch, [cache]) as window:
        before = _dispatch_snapshot()
        q, keys, values, out = _dispatch_decode(cache, seed=n)
        assert keys.shape[2] == cache.offset == n
        moved = _dispatch_moved(before)
        if path == "stock":
            assert out is None
            assert moved == ({"decode_stock": 1}, {})
        elif path == "tile":
            assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, n)).item()
            assert moved == ({"decode_tile": 1}, {})
        else:
            assert mx.array_equal(out, _dispatch_kv8_row(q, keys, values, n)).item()
            assert moved == ({}, {"decode_kv8": 1})
            assert _dispatch_mirror(cache).n == n
        assert (_dispatch_mirror(cache) is not None) == (path == "kv8")
        assert window.state != "capped"


def test_dispatch_a_decode_from_the_ceiling_caps_the_window_and_the_tile_serves_it(
    dispatch_lines, monkeypatch
):
    """The first call at the ceiling drops the mirror and caps the window, counted
    once; that call and every later one run on the tile."""
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    with _dispatch_open(monkeypatch, [cache]) as window:
        _dispatch_decode(cache, seed=31)
        assert _dispatch_mirror(cache) is not None
        _dispatch_append(cache, DISPATCH_CEIL - 1 - cache.offset, seed=32)
        for n in (DISPATCH_CEIL, DISPATCH_CEIL + 1):
            before = _dispatch_snapshot()
            q, keys, values, out = _dispatch_decode(cache, seed=n)
            assert keys.shape[2] == n
            assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, n)).item()
            capped = {"windows_capped": 1} if n == DISPATCH_CEIL else {}
            assert _dispatch_moved(before) == ({"decode_tile": 1}, capped)
            assert window.state == "capped" and _dispatch_mirror(cache) is None


@pytest.mark.parametrize(
    ("prefix", "t", "want_tile", "want_kv8", "kv8_rows"),
    [
        pytest.param(
            DISPATCH_FLOOR - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_tile": 3,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {"verify_kv8": 1, "verify_rows_kv8": 5, "verify_launches_kv8": 1},
            range(DISPATCH_FLOOR, DISPATCH_FLOOR + 5),
            id="straddles the floor",
        ),
        pytest.param(
            DISPATCH_FLOOR + 8,
            12,
            {"verify_calls": 1, "verify_t9_up": 1, "verify_straddles": 1},
            {"verify_kv8": 1, "verify_rows_kv8": 12, "verify_launches_kv8": 2},
            range(DISPATCH_FLOOR + 9, DISPATCH_FLOOR + 21),
            id="two kv8 runs",
        ),
        pytest.param(
            DISPATCH_CEIL - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_tile": 5,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {"verify_kv8": 1, "verify_rows_kv8": 3, "verify_launches_kv8": 1},
            range(DISPATCH_CEIL - 3, DISPATCH_CEIL),
            id="straddles the ceiling",
        ),
        pytest.param(
            tileattn.N0 - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_stock": 3,
                "verify_rows_tile": 5,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {},
            range(0),
            id="stock and tile only",
        ),
        pytest.param(
            1700,
            5,
            {"verify_calls": 1, "verify_t5": 1, "verify_rows_tile": 5, "verify_launches": 1},
            {},
            range(0),
            id="tile only",
        ),
    ],
)
def test_dispatch_a_verify_serves_each_run_by_its_first_row(
    dispatch_lines, monkeypatch, prefix, t, want_tile, want_kv8, kv8_rows
):
    """Each run of verify_plan goes stock, to kv8 or to the tile by its first row's
    key count; a call with no run in [floor, ceiling) builds no mirror, and a call
    that straddles the ceiling serves its rows below it and keeps its mirror."""
    cache = _dispatch_cache(prefix)
    with _dispatch_open(monkeypatch, [cache]) as window:
        keys, values = _dispatch_append(cache, t, seed=prefix)
        q = _dispatch_queries(t, seed=prefix + 1)
        before = _dispatch_snapshot()
        out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert _dispatch_moved(before) == (want_tile, want_kv8)
        _dispatch_assert_rows(out, q, keys, values, prefix, kv8_rows)
        assert (_dispatch_mirror(cache) is not None) == bool(kv8_rows)
        assert window.state != "capped"


def test_dispatch_a_verify_whose_first_row_is_at_the_ceiling_caps_and_the_tile_serves_it(
    dispatch_lines, monkeypatch
):
    """A mirror built earlier in the window is dropped by the first call that
    starts at the ceiling, and that call runs on the tile."""
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    with _dispatch_open(monkeypatch, [cache]) as window:
        _dispatch_decode(cache, seed=41)
        assert _dispatch_mirror(cache) is not None
        _dispatch_append(cache, DISPATCH_CEIL - 1 - cache.offset, seed=42)
        keys, values = _dispatch_append(cache, 5, seed=43)
        q = _dispatch_queries(5, seed=44)
        before = _dispatch_snapshot()
        out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert _dispatch_moved(before) == (
            {"verify_calls": 1, "verify_t5": 1, "verify_rows_tile": 5, "verify_launches": 1},
            {"windows_capped": 1},
        )
        assert window.state == "capped" and _dispatch_mirror(cache) is None
        _dispatch_assert_rows(out, q, keys, values, DISPATCH_CEIL - 1, range(0))


@pytest.mark.parametrize(
    ("prefix", "t", "rows"),
    [
        pytest.param(DISPATCH_FLOOR + 8, 12, DISPATCH_FLOOR + 20, id="two kv8 runs"),
        pytest.param(DISPATCH_CEIL - 4, 8, DISPATCH_CEIL, id="straddles the ceiling"),
    ],
)
def test_dispatch_a_verify_quantises_its_rows_once(dispatch_lines, monkeypatch, prefix, t, rows):
    """One prepare per call however many runs read the mirror; the window's first
    build covers the committed prefix and the call's rows up to the ceiling, never
    past it."""
    prepared, quantised = [], []
    prepare, quantize = kv8.prepare, kv8.quantize

    def counted_prepare(cache, count):
        prepared.append(count)
        return prepare(cache, count)

    def counted_quantize(k, v, out_rows):
        quantised.append(k.shape[2])
        return quantize(k, v, out_rows)

    monkeypatch.setattr(kv8, "prepare", counted_prepare)
    monkeypatch.setattr(kv8, "quantize", counted_quantize)
    cache = _dispatch_cache(prefix)
    with _dispatch_open(monkeypatch, [cache]):
        keys, values = _dispatch_append(cache, t, seed=prefix)
        q = _dispatch_queries(t, seed=prefix + 1)
        tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert prepared == [t]
        assert sum(quantised) == rows
        assert _dispatch_mirror(cache) is not None


def test_dispatch_serves_nothing_without_a_window(dispatch_lines, monkeypatch):
    """kv8 active but no decode running (a prefill, a load-time probe): both seams
    behave exactly as the tile does alone and no mirror is built."""
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    before = _dispatch_snapshot()
    q, keys, values, out = _dispatch_decode(cache, seed=51)
    assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, DISPATCH_FLOOR + 9)).item()
    keys, values = _dispatch_append(cache, 5, seed=52)
    q = _dispatch_queries(5, seed=53)
    out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
    _dispatch_assert_rows(out, q, keys, values, DISPATCH_FLOOR + 9, range(0))
    assert _dispatch_moved(before) == (
        {
            "decode_tile": 1,
            "verify_calls": 1,
            "verify_t5": 1,
            "verify_rows_tile": 5,
            "verify_launches": 1,
        },
        {},
    )
    assert _dispatch_mirror(cache) is None


def test_dispatch_with_the_floor_at_n0_kv8_takes_the_first_tile_step(
    dispatch_lines, monkeypatch
):
    """With the floor at N0, kv8 serves from N0 and the tile serves no decode below
    the ceiling."""
    kfx.small_floor(monkeypatch)
    assert kv8.N8 == tileattn.N0
    cache = _dispatch_cache(tileattn.N0 - 1)
    with _dispatch_open(monkeypatch, [cache]):
        before = _dispatch_snapshot()
        q, keys, values, out = _dispatch_decode(cache, seed=61)
        assert mx.array_equal(out, _dispatch_kv8_row(q, keys, values, tileattn.N0)).item()
        assert _dispatch_moved(before) == ({}, {"decode_kv8": 1})


def test_dispatch_the_verifier_hands_its_layer_cache_to_serve_verify(
    dispatch_lines, monkeypatch
):
    """Through mlx-vlm's exact verifier on the tiny model: each full-attention
    layer's call reaches serve_verify with that layer's own KVCache, and kv8
    serves its rows from a mirror on that cache."""
    lm = tfx.tiny_language_model()
    prefix = DISPATCH_FLOOR + 8
    ids = mx.random.randint(0, tfx.VOCAB, (prefix + 3,), key=mx.random.key(7)).tolist()
    cache = tfx.prefill_cache(lm, ids[:prefix])
    full = [c for c, layer in zip(cache, lm.layers, strict=True) if not layer.is_linear]
    assert len(full) == 2
    tfx.verify_ready(monkeypatch, lm)
    seen = []
    serve_verify = tileattn.serve_verify

    def recording(queries, keys, values, splits, layer_cache):
        seen.append(layer_cache)
        return serve_verify(queries, keys, values, splits, layer_cache)

    monkeypatch.setattr(tileattn, "serve_verify", recording)
    with _dispatch_open(monkeypatch, full):
        before = _dispatch_snapshot()
        tfx.verify_forward(lm, cache, ids[prefix : prefix + 3])
        assert [id(c) for c in seen] == [id(c) for c in full]
        assert _dispatch_moved(before) == (
            {"verify_calls": 2, "verify_t3": 2},
            {"verify_kv8": 2, "verify_rows_kv8": 6, "verify_launches_kv8": 2},
        )
        assert all(_dispatch_mirror(c) is not None for c in full)
    assert all(_dispatch_mirror(c) is None for c in full)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_kv8.py -v -k dispatch`
Expected: FAIL, 14 failed, 3 passed. These pass: the three decode cases `below N0`, `at N0` and `below the floor`, which route as the tile routes today. The rest fail in one of three ways:
- the two kv8 decode cases, the two cap tests and the floor-at-N0 test fail an `assert`, because the tile served a row that kv8 should have served;
- every direct `serve_verify` call raises `TypeError: serve_verify() takes 4 positional arguments but 5 were given`;
- the verifier test raises `TypeError: ... missing 1 required positional argument: 'layer_cache'`.

- [ ] **Step 3: Put the verify run loop in one place**

In `src/sous/engine/tileattn.py`, replace the whole of `verify` (lines 248–281), which is today:

```python
def verify(
    queries: Any,
    keys: Any,
    values: Any,
    splits: int,
    runs: list[tuple[int, int, int]] | None = None,
) -> Any:
    """The verifier's T-row causal call, row r over keys [0, n - T + r], as
    [1, 24, T, 256]: one tile call per run of verify_plan(), grouped stock calls for
    the runs below N0, concatenated in row order. `runs` is that plan when the
    caller already has it."""
    import mlx.core as mx

    from sous.engine.verifyattn import grouped_attention

    t = queries.shape[2]
    n = keys.shape[2]
    prefix = n - t
    if runs is None:
        runs = verify_plan(prefix, t, splits)
    if len(runs) == 1 and runs[0][2]:
        return tile(queries, keys, values, n, runs[0][2]).transpose(0, 2, 1, 3)
    parts = []
    for j, k, chunk in runs:
        q = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
        kk = keys if k == t else keys[:, :, : prefix + k, :]
        vv = values if k == t else values[:, :, : prefix + k, :]
        if chunk:
            parts.append(tile(q, kk, vv, prefix + k, chunk))
        else:
            stock = grouped_attention(q, kk, vv, SCALE, prefix + j, _arch())
            parts.append(stock.transpose(0, 2, 1, 3))
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.transpose(0, 2, 1, 3)
```

with:

```python
def _verify_runs(
    queries: Any,
    keys: Any,
    values: Any,
    runs: list[tuple[int, int, int]],
    served: Callable[[int, int, int], Any] | None = None,
) -> Any:
    """The run loop behind verify() and serve_verify(), as [1, 24, T, 256]. Each
    run (j, k, chunk), in row order, is answered by one of three: grouped stock
    calls when chunk is 0; `served(j, k, chunk)` when that returns a part (kv8,
    reading its int8 mirror); one tile call otherwise. Every part is a
    contiguous [1, k - j, 24, 256], and the parts are concatenated on the row
    axis. With one loop, a run kv8 leaves to the tile is sliced and launched
    exactly as it is with kv8 off. A single run's part is used as it is, with
    no concatenation."""
    import mlx.core as mx

    from sous.engine.verifyattn import grouped_attention

    t = queries.shape[2]
    prefix = keys.shape[2] - t
    parts = []
    for j, k, chunk in runs:
        part = served(j, k, chunk) if served is not None and chunk else None
        if part is None:
            q = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
            kk = keys if k == t else keys[:, :, : prefix + k, :]
            vv = values if k == t else values[:, :, : prefix + k, :]
            if chunk:
                part = tile(q, kk, vv, prefix + k, chunk)
            else:
                stock = grouped_attention(q, kk, vv, SCALE, prefix + j, _arch())
                part = stock.transpose(0, 2, 1, 3)
        parts.append(part)
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.transpose(0, 2, 1, 3)


def verify(
    queries: Any,
    keys: Any,
    values: Any,
    splits: int,
    runs: list[tuple[int, int, int]] | None = None,
) -> Any:
    """The verifier's T-row causal call, row r over keys [0, n - T + r], as
    [1, 24, T, 256]: one tile call per run of verify_plan(), grouped stock calls for
    the runs below N0, concatenated in row order. `runs` is that plan when the
    caller already has it."""
    if runs is None:
        t = queries.shape[2]
        runs = verify_plan(keys.shape[2] - t, t, splits)
    return _verify_runs(queries, keys, values, runs)
```

The old early return for a single tile run made the same `tile(queries, keys, values, n, chunk)` call the loop now makes for that run: (0, T) takes the whole `queries`, k = T takes the whole `keys` and `values`, and `prefix + T` is `n`. Its output is used without a concatenation, as before. So `verify` launches the same kernels on the same arrays as before, and every existing tile test pins it unchanged.

- [ ] **Step 4: Route decode rows to kv8 in the decode hook**

In `src/sous/engine/tileattn.py`, replace the whole of `_serve_decode` (lines 362–388), which is today:

```python
def _serve_decode(args: tuple, kwargs: dict) -> Any | None:
    """The tile's output for a one-row call it serves, else None, and the wrapper
    then makes the very call Qwen3_5Attention.__call__ made: deciding after the
    projections never appends to the cache twice. Multi-row (prefill) calls and
    calls while the flag is clear are not counted."""
    splits = _active_splits
    if not splits:
        return None
    try:
        queries, keys, values, cache, scale, mask, sinks = _bind(*args, **kwargs)
    except TypeError:
        return None
    if getattr(queries, "ndim", 0) != 4 or queries.shape[2] != 1:
        return None
    if not (
        mask is None
        and sinks is None
        and _plain_kv_cache(cache)
        and in_scope(queries, keys, values, scale)
    ):
        calls["decode_out"] += 1
        return None
    if keys.shape[2] < N0:
        calls["decode_stock"] += 1
        return None
    calls["decode_tile"] += 1
    return decode(queries, keys, values, splits)
```

with:

```python
def _serve_decode(args: tuple, kwargs: dict) -> Any | None:
    """The output for a one-row call the tile serves (or, from kv8's floor up to
    its ceiling inside a decode, kv8), else None. On None the wrapper makes the
    very call Qwen3_5Attention.__call__ made: deciding after the projections
    never appends to the cache twice. Multi-row (prefill) calls and calls while
    the flag is clear are not counted."""
    splits = _active_splits
    if not splits:
        return None
    try:
        queries, keys, values, cache, scale, mask, sinks = _bind(*args, **kwargs)
    except TypeError:
        return None
    if getattr(queries, "ndim", 0) != 4 or queries.shape[2] != 1:
        return None
    if not (
        mask is None
        and sinks is None
        and _plain_kv_cache(cache)
        and in_scope(queries, keys, values, scale)
    ):
        calls["decode_out"] += 1
        return None
    if keys.shape[2] < N0:
        calls["decode_stock"] += 1
        return None
    # Function-local: kv8 imports this module at its top.
    from sous.engine import kv8

    n = keys.shape[2]
    if kv8.serves(n):
        return kv8.decode(queries, cache, splits)
    # At or past the ceiling this caps the window and drops the mirror. The key
    # count never falls inside a decode, so every later call of it lands here too.
    kv8.observe_first_row(cache, n)
    calls["decode_tile"] += 1
    return decode(queries, keys, values, splits)
```

- [ ] **Step 5: Route verify runs to kv8 in `serve_verify`**

In `src/sous/engine/tileattn.py`, replace the whole of `serve_verify` (lines 416–431), which is today:

```python
def serve_verify(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """verify() for the exact verifier's hook, counted by T, by the rows each path
    took and by the calls cut into more than one run."""
    t = queries.shape[2]
    runs = verify_plan(keys.shape[2] - t, t, splits)
    calls["verify_calls"] += 1
    calls[f"verify_t{t}" if t <= MAX_RUN else "verify_t9_up"] += 1
    for j, k, chunk in runs:
        if chunk:
            calls["verify_rows_tile"] += k - j
            calls["verify_launches"] += 1
        else:
            calls["verify_rows_stock"] += k - j
    if len(runs) > 1:
        calls["verify_straddles"] += 1
    return verify(queries, keys, values, splits, runs)
```

with:

```python
def serve_verify(queries: Any, keys: Any, values: Any, splits: int, cache: Any) -> Any:
    """verify() for the exact verifier's hook, counted by T, by the rows each path
    took and by the calls cut into more than one run. `cache` is the layer's
    KVCache the rows were just appended to; a run kv8 serves reads that cache's
    int8 mirror instead of the bf16 keys."""
    # Function-local: kv8 imports this module at its top.
    from sous.engine import kv8

    t = queries.shape[2]
    prefix = keys.shape[2] - t
    runs = verify_plan(prefix, t, splits)
    calls["verify_calls"] += 1
    calls[f"verify_t{t}" if t <= MAX_RUN else "verify_t9_up"] += 1
    # Before any run asks kv8.serves: a call that starts at or past the ceiling
    # must drop the mirror, not refresh it.
    kv8.observe_first_row(cache, prefix + 1)
    # A run's first row decides for the whole run: the floor and the ceiling are
    # step boundaries, and verify_plan cuts at every one.
    on_kv8 = {j for j, _, chunk in runs if chunk and kv8.serves(prefix + j + 1)}
    for j, k, chunk in runs:
        if j in on_kv8:
            kv8.calls["verify_rows_kv8"] += k - j
            kv8.calls["verify_launches_kv8"] += 1
        elif chunk:
            calls["verify_rows_tile"] += k - j
            calls["verify_launches"] += 1
        else:
            calls["verify_rows_stock"] += k - j
    if len(runs) > 1:
        calls["verify_straddles"] += 1
    if not on_kv8:
        return _verify_runs(queries, keys, values, runs)
    kv8.calls["verify_kv8"] += 1
    # Once, before the first launch: every kv8 run reads the same mirror, so a call
    # cut into several runs quantises its new rows once.
    mirror = kv8.prepare(cache, t)

    def kv8_part(j: int, k: int, chunk: int) -> Any:
        return kv8.verify_run(queries, mirror, prefix, j, k, chunk) if j in on_kv8 else None

    return _verify_runs(queries, keys, values, runs, kv8_part)
```

- [ ] **Step 6: Pass the cache from verifyattn's wrapper**

In `src/sous/engine/verifyattn.py`, inside `_wrap`, replace line 438:

```python
                output = tileattn.serve_verify(queries, keys, values, splits)
```

with:

```python
                output = tileattn.serve_verify(queries, keys, values, splits, cache)
```

`cache` is the wrapper's own `_attention` parameter. `_in_scope` has already checked that it is an exact `KVCache` without bits or left padding.

Then confirm that nothing else calls the old four-argument form.

Run: `grep -rnw "serve_verify" src tests scripts`
Expected: exactly these hits:
- the definition in `src/sous/engine/tileattn.py`;
- the call in `src/sous/engine/verifyattn.py`;
- in `tests/test_kv8.py`, the direct calls (four tests), plus `def recording(...)` and the `serve_verify(...)` call inside it.

Every call passes five arguments.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kv8.py -v -k dispatch`
Expected: PASS, 17 passed.

- [ ] **Step 8: Run the neighbouring suites, then everything**

The seams sit on the tile's two hooks. With kv8 at its defaults (`_active` 0, no window), `kv8.serves` is False and `kv8.observe_first_row` does nothing, so every existing tile and verifier test must pass unchanged. That covers the counter tests through the real verifier (`test_verify_rows_follow_the_plan`, `test_a_declined_verify_computes_the_stock_attention`), the stock-architecture test that monkeypatches `verifyattn.grouped_attention`, the tile's load-time probe through `verify()`, and the contract tests.

Run: `uv run pytest tests/test_tileattn.py tests/test_tileattn_contract.py tests/test_verifyattn.py tests/test_kv8.py -v`
Expected: PASS. The only skips are the tile's and kv8's `nax` tests, where the tensor units are absent.

Run: `uv run pytest -m "not model"`
Expected: PASS.

- [ ] **Step 9: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/tileattn.py src/sous/engine/verifyattn.py tests/test_kv8.py
git commit -m "feat(engine): route decode and verify rows to the int8 mirror" -m "The attention tile's two hooks are where a row's path is decided, so kv8
joins there and nowhere else. The rule is the same in both seams. A row
that kv8 serves (its key count from the floor up to the ceiling, inside a
decode's window) reads the int8 mirror. A call that starts at or past the
ceiling caps the window first, and since the key count never falls inside
a decode, every later call of that decode stays on the bf16 tile.
Everything else runs exactly as before.

A verify call is routed run by run on each run's first row. That is
exact because the floor and the ceiling are step boundaries and
verify_plan already cuts at every one. The mirror is brought up to date
once per call, before the first launch, so a call cut into several kv8
runs quantises its new rows once. A call that straddles the ceiling keeps
its mirror and serves only its rows below the ceiling.

verify() and serve_verify() now share one run loop, so a run that kv8
leaves to the tile is sliced and launched as it is with kv8 off. The
verifier's wrapper already held the layer's cache and now hands it over.
decode_tile and the tile's verify row counters move only on the tile's
own path."
```


### Task 4: `enable()`, the gates, the source pins and the probe

**Files:**
- Modify: `src/sous/engine/kv8.py` (created by Tasks 1–3 of this plan):
  - the import block;
  - the gates section, appended at the end of the file, after the dispatch section.
- Test: `tests/test_kv8.py` (created by Tasks 1–3 of this plan):
  - the import block;
  - the enable section, appended at the end of the file.

Neither file exists on `main` or on `feat/int8-kv-mirror` before this plan, so this task replaces no line of the tree; both appends go after everything the earlier tasks wrote. Nothing else changes: `tileattn.py`, `verifyattn.py` and `projkernel.py` are only read.

**Interfaces:**
- Consumes:
  - Tasks 1–3 (`kv8`, names and signatures as the interface sheet fixes them):
    - constants `N8` (a module int the tests monkeypatch), `GROWTH = 4096`, `REL_RMS_BOUND = 1e-2`, `STOCK_RMS_BOUND = 3e-2`, `MIRROR_ATTR = "_sous_kv8"`;
    - state `_active`, `_ceiling`, `_window`, `calls` (keys `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_rows`, `mirror_trims`, `mirror_grown`, `windows`, `windows_capped`), the dataclasses `Window` and `Mirror`;
    - `quantize(k, v, out_rows) -> (codes, sc)`: reads strided bf16 views, writes rows `[0, rows)` of an `out_rows`-row output and nothing past them;
    - `dequant(codes, sc) -> (k32, v32)`;
    - `tile8(queries, codes, sc, n, chunk)`, the stand-in's swap target; `decode_codes(queries, codes, sc, n, splits)` and `verify_codes(queries, codes, sc, n, splits, runs=None)` both reach it through the module global, so a swapped `kv8.tile8` serves them;
    - `open_window(*, force=False)`, `close_window(caches) -> (state, rows)` (accepts an empty list, and caches with or without a mirror, and drops every mirror and its trim wrap on the caches it is given), `serving_now()`, `serves(n_row)`, `observe_first_row(cache, n_first)`;
    - `prepare(cache, t) -> Mirror`: under a forced window it treats L as unbounded, for rows and capacity alike; the first served call builds at `cap = round_up(offset + GROWTH, 256)` and installs the trim wrap; a mirror whose `serial` is not the open window's is rebuilt from row 0; it counts `mirror_builds` once per build or rebuild and `mirror_grown` once per growth, and returns with `mirror.n == cache.offset`;
    - `decode(queries, cache, splits)` (counts `decode_kv8`) and the seams: `tileattn._serve_decode` calls `kv8.serves`, `kv8.decode` and `kv8.observe_first_row`; `tileattn.serve_verify(queries, keys, values, splits, cache)` calls `kv8.observe_first_row` and `kv8.prepare`, counts `verify_kv8` once per call it serves and serves its runs through `kv8.verify_run`; `verifyattn._wrap` passes `cache`.
  - tileattn: `N0`, `GRAN`, `HQ`, `HKV`, `GQA`, `D`, `SCALE`, `MAX_RUN`, `MEASURED_SPLITS`, `PROBE_FAR_KEYS` (via `far_mark`), `chunk_for(n, splits)`, `boundaries(n_max, splits)`, `far_mark(splits)`, `_arch()`, `_DECODE_MODULE`, `_HOOK_MARK`, `_decode_wrapper(original)`, `_active_splits`, `_plain_kv_cache(cache)`, `calls`.
  - verifyattn: `_source_digest(module, qualname)` (sha256 of `inspect.getsource`, which follows `__wrapped__`; None when unreadable), `_attention_modules(model)`, `_TAG`.
  - projkernel: `_verifier_class()` (the exact verifier's class, constructed with no arguments), `calls`.
  - nax: `platform_reason()`, `Availability(available, reason=None)`. int8prefill: `_error_line(text)`.
  - `tests/tile_fixtures.py` (`tfx`): `guard_tile_state`, `tile_ready` (platform rule passes, 20 cores under this GPU's name, `tileattn.tile` the stand-in, `PROBE_FAR_KEYS = 3_000`), `tiny_language_model`, `verify_ready`.
  - `tests/kv8_fixtures.py` (`kfx`): `stand_in_tile8` (any chunk > 0) and the `nax` marker on `kv8.availability()`.
- Consumes from Tasks 1–2, in `kv8`, and never redefines: `availability() -> nax.Availability` (the platform rule, then `_compile_probe()`; refused as `the int8 K/V kernels failed: <error line>`), `_compile_probe()`, `_compile_ok`, `ceiling(window: int, splits: int) -> int` (pure; `ceiling(131072, 20) == 84_481`, `ceiling(262144, 20) == 170_241`, `ceiling(0, 20) == 0`) and `_floor_reason(splits) -> str | None`.
- Produces (in `kv8`):
  - `VALIDATED_KV8_SOURCES: dict[tuple[str, str], str]`: 17 sha256 digests keyed `(module, qualname)`, computed on mlx-vlm 0.7.2 (Step 4).
  - `probe(modules, splits: int, arch: str, verify_active: bool) -> str | None`: steps (1)–(7); leaves `_active` 0, `_window` None, the tile's flag and decode global, and `kv8.calls`, `tileattn.calls` and `projkernel.calls` as it found them; `mx.clear_cache()` in `finally`.
  - `enable(model, *, enabled: bool, tile_status, verify_status, window: int) -> dict`: keys `state`, `reason`, `probe_seconds`, `floor`, `ceiling`; never raises. Off: every value but `state` None. Active: `{"state": "active", "reason": None, "probe_seconds": <float>, "floor": N8, "ceiling": L}`, and one INFO line `int8 K/V mirror: {layers} layers, {N8} <= keys < {L}, probe {seconds:.2f}s`.
  - Private, reached by the tests: `_refuse(reason, probe_seconds=None, *, expected=False)`, `_tile_reason(tile_status)`, `_pins_reason()`, `_compile_ok`, `_compile_probe()`, the step functions `_probe_quantiser`, `_probe_exactness`, `_probe_parity`, `_probe_fence`, `_probe_mirror`, `_probe_stock`, `_probe_module`, `_probe_steps`, and the helpers `_reference_quantize`, `_tie_rows`, `_dequant_q`, `_reference_attention`, `_distance`, `_rows`, `_run`, `_restored`; constants `_VERIFY_ROWS = 5`, `_EXACT_ROWS = (1, 5, 8)`, `_QUANT_RANDOM = 37`, `_QUANT_TIES = 8`, `_FP16_MAX = 65504.0`.
  - Reason texts:
    - expected (one INFO line `int8 K/V mirror unavailable: <reason>`): `"attention tile off"`, the tile's reason verbatim, else `f"attention tile {state or 'unknown'}"`; `f"window below the kv8 floor (window {window}, ceiling {L}, floor {N8})"`;
    - warned (one `warnings.warn`, `sous: int8 K/V mirror unavailable (<reason>); decode and verify attention read the bf16 cache`): `f"mlx-vlm {qualname} changed"`; `f"the int8 K/V kernels failed: {error line}"`; `f"floor {N8} keys is not a tile step boundary at {splits} splits"`; the probe's reason; an exception's error line (logged as `int8 K/V mirror: enabling failed` with its traceback);
    - probe: `f"quantiser: the {random|tie|all-zero} rows differ from the reference"`, `"quantiser: rows added in two steps differ from one build"`, `f"exactness: T={t} at n={n} is {d:.3g} from the fp32 reference (bound 0.01)"`, `f"parity: verify rows at prefix {prefix}, T={t} differ from decode"`, `f"reads past n: T={t} at n={n} differs from the clean buffer"`, `"mirror: the first build differs from a bulk re-quantisation"`, `"mirror: a trim did not reach the mirror"`, `"mirror: a trim and an unserved append differ from a bulk re-quantisation"`, `"mirror: the shrinking block across the floor differs from a bulk re-quantisation"`, `"mirror: a mirror from another window differs from a bulk re-quantisation"`, `"mirror: a mirror from another window was not rebuilt"`, `"mirror: a growth differs from a bulk re-quantisation"`, `"mirror: a growth was not counted"`, `f"quantisation: n={n} is {d:.3g} from stock attention (bound 0.03)"`, `"the model's attention module did not reach kv8"`, `f"correctness: the model's module is {d:.3g} from the bf16 path (bound 0.03)"`, `"the exact verifier's attention did not reach kv8"`, `f"correctness: the verifier's attention is {d:.3g} from the bf16 path (bound 0.03)"`.

Notes for the implementer:
- **Order.** The first statement clears `_active`, `_window` and `_ceiling`; `enabled` false returns `off` before any guard. Then: the tile (expected), the pins (warned), and, timed from here so `probe_seconds` covers kv8's whole cost to a load, `availability()` (warned), the floor as a step boundary at the active S (warned), the ceiling L from the window (`L <= N8` expected), the probe (warned). `_active` and `_ceiling` are raised only after the probe passed.
- **The probe's steps** are module functions called by name from `_probe_steps`, in the spec's numbering: (1) `_probe_quantiser`, (2) `_probe_exactness`, (3) `_probe_parity`, (4) `_probe_fence`, (5) `_probe_mirror`, (6) `_probe_stock`, (7) `_probe_module`. Tests that aim at step 7 replace `_probe_parity`, the longest step, with a no-op. Steps 2–4 and 6 run with no window open, so a test kernel that raises only inside a window fails in step 7 alone; step 5 opens forced windows around `prepare`; step 7 opens one around each served call. The marks are the floor N8, the next two step boundaries and the far mark (`tileattn.far_mark`, the first boundary at or past `PROBE_FAR_KEYS`): 1,536, 1,921, 2,561 and 57,601 at S = 20; `tile_ready` lowers the far mark to 3,201.
- **What step 7 sets.** It installs the tile's decode hook if it is absent and sets `tileattn._active_splits = splits` (in a production load both already hold, since the tile is active), and `probe()` puts both back in `finally`.
- **Already defined.** `availability()`, `_compile_probe()`, `_compile_ok`, `ceiling()` and `_floor_reason()` come from Tasks 1–2. Do not redefine them: ruff's F811 rejects a second definition, and this task's `enable()` calls the existing ones. The fixtures' `nax` marker already reads `kv8.availability()` when `tests/kv8_fixtures.py` is imported.
- **Test helpers.** Every helper and test name below is new to `tests/test_kv8.py`; a collision with an earlier section is a copy error in this plan, not something to merge — stop and report it.
- **Thread.** `enable()` runs on the `sous-model-load` thread, so the probe's mlx work needs no `release_mlx_thread_state()` of its own.

- [ ] **Step 1: Write the failing tests**

The import block at the top of `tests/test_kv8.py` must hold at least these (keep every line the earlier tasks put there; `uv run ruff check --fix .` sorts the block):

```python
import contextlib
import importlib
import logging
import time
import types
import warnings

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import kv8, nax, projkernel, tileattn, verifyattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402
```

Append the enable section to the end of `tests/test_kv8.py`:

```python
# ---- enable(): the gates, the pins, availability, the ceiling and the probe ---------

_GATE_BUILD_ERROR = "Unable to build metal library\nprogram_source:3:1: error: boom"
# The attention tile's and exact verify's statuses as active loads report them.
_ACTIVE_TILE = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.5}
_ACTIVE_VERIFY = {"state": "active", "reason": None, "probe_seconds": 0.2}
# The default window; at S = 20 its ceiling is 84,481 keys.
_WINDOW = 131_072
_PINNED = {
    ("mlx_vlm.models.cache", "KVCache.update_and_fetch"),
    ("mlx_vlm.models.cache", "KVCache.trim"),
    ("mlx_vlm.models.cache", "KVCache.prefix_cache_reserve"),
    ("mlx_vlm.speculative.cache_state", "start_speculative_cache"),
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.commit"),
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.abort"),
    ("mlx_vlm.speculative.cache_state", "_trim_append_caches"),
    ("mlx_vlm.speculative.cache_state", "_cache_position_reader"),
    ("mlx_vlm.speculative.cache_state", "commit_speculative_round"),
    ("mlx_vlm.speculative.cache_state", "abort_speculative_round"),
    ("mlx_vlm.speculative.cache_state", "rollback_speculative_cache"),
    ("mlx_vlm.speculative.dflash", "_dflash_rounds"),
    ("mlx_vlm.speculative.dflash", "_dflash_verify"),
    ("mlx_vlm.speculative.dflash", "_dflash_verify_greedy"),
    ("mlx_vlm.speculative.dflash", "_reserve_dflash_target_cache"),
    ("mlx_vlm.models.qwen3_5.language", "LanguageModel.rollback_speculative_cache"),
    ("mlx_vlm.models.qwen3_5.speculative_verifier", "Qwen3_5BatchInvariantForward.verify"),
}


def _untouchable(*args, **kwargs):
    raise AssertionError("a guard or step ran that this test says must not")


def _language():
    return importlib.import_module("mlx_vlm.models.qwen3_5.language")


def _gate_state(monkeypatch) -> None:
    """Registers restores for what enable() and the probe leave behind for the
    rest of the process: kv8's flag, window, ceiling, floor and remembered
    compile verdict, and the tile's decode global, flag and verdict."""
    tfx.guard_tile_state(monkeypatch)
    for name in ("_active", "_window", "_ceiling", "_compile_ok", "N8"):
        monkeypatch.setattr(kv8, name, getattr(kv8, name))


def _kv8_ready(monkeypatch, *, fake_availability: bool = True, fake_probe: bool = True) -> dict:
    """Every guard that reads the machine passes on any GPU (tile_ready: the
    platform rule, 20 cores under this GPU's name, the tile's stand-in and a
    far mark of 3,201 keys), the stand-in is kv8's kernel and the floor is N0.
    Unless asked for, availability() and the probe are faked to pass, so a
    test reaches the guard it is about without compiling or probing. Returns
    the tile's active status to hand enable()."""
    _gate_state(monkeypatch)
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    monkeypatch.setattr(kv8, "N8", tileattn.N0)
    if fake_availability:
        monkeypatch.setattr(kv8, "availability", lambda: nax.Availability(True))
    if fake_probe:
        monkeypatch.setattr(kv8, "probe", lambda modules, splits, arch, verify_active: None)
    return dict(_ACTIVE_TILE)


class _GateRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _gate_log():
    """Every record kv8's logger emits inside the block, INFO included.
    Records still propagate, so a test's own caplog sees them too."""
    logger = logging.getLogger("sous.engine.kv8")
    handler, level = _GateRecords(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _refusal(lm, *, warned: bool, **kwargs) -> dict:
    """enable() on `lm`, expected to refuse, leaving the flag, the window and
    the ceiling clear. A load kv8 does not apply to (the tile, the window) is
    one INFO line and no warning, since most Macs load that way; one where it
    should have run and did not is exactly one warning, on one line, in kv8's
    words."""
    kwargs = {"tile_status": dict(_ACTIVE_TILE), "verify_status": None, "window": _WINDOW, **kwargs}
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = kv8.enable(lm, enabled=True, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert status["state"] == "unavailable"
    assert status["floor"] is None and status["ceiling"] is None
    if warned:
        assert len(messages) == 1, messages
        assert messages[0].startswith("sous: int8 K/V mirror unavailable (")
        assert "\n" not in messages[0]
        assert not any(m.startswith("int8 K/V mirror unavailable") for m in infos), infos
    else:
        assert messages == []
        assert infos == [f"int8 K/V mirror unavailable: {status['reason']}"]
    assert kv8._active == 0 and kv8._window is None and kv8._ceiling == 0
    return status


def _enabling_failures(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "sous.engine.kv8" and r.getMessage() == "int8 K/V mirror: enabling failed"
    ]


def test_enable_off_returns_before_any_guard(monkeypatch):
    _gate_state(monkeypatch)
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", kv8.Window(serial=99))
    monkeypatch.setattr(kv8, "_ceiling", 84_481)
    for name in ("_tile_reason", "_pins_reason", "availability", "ceiling", "probe"):
        monkeypatch.setattr(kv8, name, _untouchable)
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = kv8.enable(
            types.SimpleNamespace(), enabled=False, tile_status={}, verify_status=None, window=0
        )
    assert status == {
        "state": "off",
        "reason": None,
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert records == []
    assert kv8._active == 0 and kv8._window is None and kv8._ceiling == 0


@pytest.mark.parametrize(
    ("tile_status", "reason"),
    [
        ({"state": "off", "reason": None, "splits": None}, "attention tile off"),
        (
            {"state": "unavailable", "reason": "split target 16 not measured (measured: 20)"},
            "split target 16 not measured (measured: 20)",
        ),
        ({"state": "unavailable", "reason": None}, "attention tile unavailable"),
        (None, "attention tile unknown"),
    ],
)
def test_enable_rides_on_the_attention_tile(monkeypatch, tile_status, reason):
    _kv8_ready(monkeypatch)
    monkeypatch.setattr(kv8, "_pins_reason", _untouchable)
    status = _refusal(tfx.tiny_language_model(), warned=False, tile_status=tile_status)
    assert status["reason"] == reason and status["probe_seconds"] is None


_GUARD_REASONS = [
    "macOS 15.5 < 26.2",
    "mlx-vlm KVCache.trim changed",
    "the int8 K/V kernels failed: program_source:3:1: error: boom",
    "floor 1600 keys is not a tile step boundary at 20 splits",
    "window below the kv8 floor (window 3000, ceiling 1536, floor 1536)",
    "parity: verify rows at prefix 1535, T=6 differ from decode",
]
# Which of them warn: a pinned source that drifted, kernels this OS rejects, a
# floor that is no cut point and a failed probe. The tile and the window only
# say kv8 does not apply to this load.
_GUARD_WARNS = [False, True, True, True, False, True]


@pytest.mark.parametrize("first", range(len(_GUARD_REASONS)))
def test_enable_names_the_first_failing_guard(monkeypatch, first):
    """Guard `first` and every guard after it fail; the reason is guard `first`'s."""
    tile = _kv8_ready(monkeypatch)
    kwargs: dict = {"tile_status": tile, "verify_status": None, "window": _WINDOW}
    breakers = [
        lambda: kwargs.update(
            tile_status={"state": "unavailable", "reason": _GUARD_REASONS[0], "splits": None}
        ),
        lambda: monkeypatch.setitem(
            kv8.VALIDATED_KV8_SOURCES, ("mlx_vlm.models.cache", "KVCache.trim"), "0" * 64
        ),
        lambda: monkeypatch.setattr(
            kv8, "availability", lambda: nax.Availability(False, _GUARD_REASONS[2])
        ),
        lambda: monkeypatch.setattr(kv8, "N8", 1600),
        lambda: kwargs.update(window=3000),
        lambda: monkeypatch.setattr(
            kv8, "probe", lambda modules, splits, arch, verify_active: _GUARD_REASONS[5]
        ),
    ]
    for brk in breakers[first:]:
        brk()
    status = _refusal(tfx.tiny_language_model(), warned=_GUARD_WARNS[first], **kwargs)
    assert status["reason"] == _GUARD_REASONS[first]
    # The cost is reported once the kernels' compile has started.
    assert (status["probe_seconds"] is None) == (first < 2)


@pytest.mark.parametrize(("window", "limit"), [(0, 0), (3000, 1536)])
def test_enable_refuses_a_window_below_the_floor(monkeypatch, window, limit):
    """A window of 0, every test-built engine's, fits no ceiling at all; 3,000
    tokens fit one at the floor itself, below which kv8 serves nothing."""
    _kv8_ready(monkeypatch)
    monkeypatch.setattr(kv8, "probe", _untouchable)
    status = _refusal(tfx.tiny_language_model(), warned=False, window=window)
    assert status["reason"] == (
        f"window below the kv8 floor (window {window}, ceiling {limit}, floor 1536)"
    )
    assert isinstance(status["probe_seconds"], float)


def test_enable_refuses_a_floor_that_is_not_a_step_boundary(monkeypatch):
    _kv8_ready(monkeypatch)
    monkeypatch.setattr(kv8, "N8", 1600)
    monkeypatch.setattr(kv8, "probe", _untouchable)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == "floor 1600 keys is not a tile step boundary at 20 splits"
    assert isinstance(status["probe_seconds"], float)


def test_the_floor_is_a_step_boundary_at_every_measured_split():
    """Another core count joining the tile's measured splits fails here until
    the floor is re-cut: kv8 and the tile may meet only at step boundaries."""
    for splits in tileattn.MEASURED_SPLITS:
        assert kv8.N8 in tileattn.boundaries(kv8.N8, splits), splits


@pytest.mark.parametrize(
    ("window", "limit"),
    [
        (0, 0),
        (1535, 0),
        (3000, 1536),
        (4096, 1921),
        (65_536, 42_241),
        (131_072, 84_481),
        (262_144, 170_241),
    ],
)
def test_the_ceiling_is_the_last_step_boundary_whose_mirror_fits(window, limit):
    assert kv8.ceiling(window, 20) == limit


@pytest.mark.parametrize("splits", [10, 16, 20])
@pytest.mark.parametrize("window", [4096, 32_768, 131_072, 262_144])
def test_the_ceiling_fits_its_mirror_and_the_next_boundary_does_not(window, splits):
    def fits(b: int) -> bool:
        c = -(-b // 256) * 256
        return 1.504 * c + 0.504 * c / 16 + 272 <= window

    limit = kv8.ceiling(window, splits)
    bounds = tileattn.boundaries(window + 32 * splits, splits)
    assert limit in bounds and fits(limit)
    assert not fits(bounds[bounds.index(limit) + 1])


def test_the_kv8_pins_hold_on_the_locked_mlx_vlm():
    """A dependency bump that changes any function the mirror's validity rests
    on fails here until someone re-reads it against the trim wrap and adds its
    new hash."""
    assert set(kv8.VALIDATED_KV8_SOURCES) == _PINNED
    for (module, qualname), digest in kv8.VALIDATED_KV8_SOURCES.items():
        assert verifyattn._source_digest(module, qualname) == digest, qualname
    assert kv8._pins_reason() is None


@pytest.mark.parametrize(
    "key",
    [
        ("mlx_vlm.models.cache", "KVCache.trim"),
        ("mlx_vlm.speculative.cache_state", "_trim_append_caches"),
        ("mlx_vlm.models.qwen3_5.speculative_verifier", "Qwen3_5BatchInvariantForward.verify"),
    ],
)
def test_enable_refuses_a_changed_source(monkeypatch, key):
    _kv8_ready(monkeypatch)
    monkeypatch.setitem(kv8.VALIDATED_KV8_SOURCES, key, "0" * 64)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == f"mlx-vlm {key[1]} changed"
    assert status["probe_seconds"] is None


def test_a_pinned_function_mlx_vlm_removed_counts_as_changed(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    _kv8_ready(monkeypatch)
    monkeypatch.delattr(KVCache, "prefix_cache_reserve")
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == "mlx-vlm KVCache.prefix_cache_reserve changed"


def test_compile_probe_runs_the_quantiser_and_every_row_count_once(monkeypatch):
    _gate_state(monkeypatch)
    monkeypatch.setattr(nax, "platform_reason", lambda: None)
    monkeypatch.setattr(kv8, "_compile_ok", False)
    seen: list = []
    quantize = kv8.quantize

    def spy_quantize(k, v, out_rows):
        seen.append(("quantize", k.shape[2], out_rows, k.dtype))
        return quantize(k, v, out_rows)

    def spy_tile8(queries, codes, sc, n, chunk):
        seen.append(("tile8", queries.shape[2], n, chunk, queries.dtype))
        return kfx.stand_in_tile8(queries, codes, sc, n, chunk)

    monkeypatch.setattr(kv8, "quantize", spy_quantize)
    monkeypatch.setattr(kv8, "tile8", spy_tile8)
    assert kv8.availability() == nax.Availability(True)
    n0, chunk = tileattn.N0, tileattn.chunk_for(tileattn.N0, 20)
    assert seen == [("quantize", n0, n0 + 256, mx.bfloat16)] + [
        ("tile8", t, n0, chunk, mx.bfloat16) for t in range(1, tileattn.MAX_RUN + 1)
    ]
    assert kv8._compile_ok
    seen.clear()
    assert kv8.availability() == nax.Availability(True)
    assert seen == []


def test_availability_reports_the_error_line_and_remembers_only_success(monkeypatch, caplog):
    _gate_state(monkeypatch)
    monkeypatch.setattr(nax, "platform_reason", lambda: None)
    monkeypatch.setattr(kv8, "_compile_ok", False)

    def rejected(*args):
        raise RuntimeError(_GATE_BUILD_ERROR)

    monkeypatch.setattr(kv8, "tile8", rejected)
    with caplog.at_level(logging.WARNING, logger="sous.engine.kv8"):
        refused = kv8.availability()
    assert refused == nax.Availability(
        False, "the int8 K/V kernels failed: program_source:3:1: error: boom"
    )
    assert not kv8._compile_ok
    # The reason keeps one line; the log keeps the compiler's whole report.
    (record,) = [r for r in caplog.records if r.name == "sous.engine.kv8"]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        f"int8 K/V mirror: the kernels failed to build or run:\n{_GATE_BUILD_ERROR}"
    )
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    assert kv8.availability() == nax.Availability(True) and kv8._compile_ok
    monkeypatch.setattr(kv8, "tile8", rejected)
    assert kv8.availability() == nax.Availability(True)


def test_enable_refuses_kernels_that_fail_to_build_with_the_error_line(monkeypatch):
    _kv8_ready(monkeypatch, fake_availability=False)
    monkeypatch.setattr(kv8, "_compile_ok", False)

    def rejected(*args):
        raise RuntimeError(_GATE_BUILD_ERROR)

    monkeypatch.setattr(kv8, "tile8", rejected)
    monkeypatch.setattr(kv8, "probe", _untouchable)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == "the int8 K/V kernels failed: program_source:3:1: error: boom"
    assert isinstance(status["probe_seconds"], float)


def test_enable_refuses_a_failed_probe_with_its_cost(monkeypatch):
    _kv8_ready(monkeypatch)
    reason = "exactness: T=5 at n=1540 is 0.101 from the fp32 reference (bound 0.01)"
    monkeypatch.setattr(kv8, "probe", lambda modules, splits, arch, verify_active: reason)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == reason
    assert isinstance(status["probe_seconds"], float)


def test_enable_refuses_an_exception_in_a_guard_with_its_error_line(monkeypatch, caplog):
    _kv8_ready(monkeypatch)
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_ceiling", 84_481)
    error = RuntimeError(_GATE_BUILD_ERROR)

    def broken():
        raise error

    monkeypatch.setattr(kv8, "_pins_reason", broken)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == "program_source:3:1: error: boom"
    assert status["probe_seconds"] is None
    # The warning keeps one line; the log keeps where it was raised.
    (record,) = _enabling_failures(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


def test_enable_refuses_a_probe_that_raises_and_reports_what_it_cost(monkeypatch, caplog):
    _kv8_ready(monkeypatch)

    def raising(modules, splits, arch, verify_active):
        raise RuntimeError("device lost")

    monkeypatch.setattr(kv8, "probe", raising)
    status = _refusal(tfx.tiny_language_model(), warned=True)
    assert status["reason"] == "device lost"
    assert isinstance(status["probe_seconds"], float)
    (record,) = _enabling_failures(caplog)
    assert record.exc_info is not None and "device lost" in str(record.exc_info[1])


@pytest.mark.parametrize(
    ("verify_status", "verify_active"),
    [
        (None, False),
        ({"state": "off", "reason": None, "probe_seconds": None}, False),
        ({"state": "unavailable", "reason": "mlx 0.99.0 not validated"}, False),
        (_ACTIVE_VERIFY, True),
    ],
)
def test_enable_activates_and_probes_the_verifier_only_beside_exact_verify(
    monkeypatch, caplog, verify_status, verify_active
):
    tile = _kv8_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    seen: list = []

    def passing(modules, splits, arch, verify):
        seen.append(([id(m) for m in modules], splits, arch, verify, kv8._active, kv8._ceiling))

    monkeypatch.setattr(kv8, "probe", passing)
    with caplog.at_level(logging.INFO, logger="sous.engine.kv8"):
        status = kv8.enable(
            lm, enabled=True, tile_status=tile, verify_status=verify_status, window=_WINDOW
        )
    assert status == {
        "state": "active",
        "reason": None,
        "probe_seconds": status["probe_seconds"],
        "floor": 1536,
        "ceiling": 84_481,
    }
    assert isinstance(status["probe_seconds"], float)
    # The flag and the ceiling are raised only once the probe has passed.
    modules = [id(m) for m in verifyattn._attention_modules(lm)]
    assert seen == [(modules, 20, tileattn._arch(), verify_active, 0, 0)]
    assert kv8._active == 1 and kv8._ceiling == 84_481 and kv8._window is None
    assert [r.getMessage() for r in caplog.records if r.name == "sous.engine.kv8"] == [
        f"int8 K/V mirror: 2 layers, 1536 <= keys < 84481, probe {status['probe_seconds']:.2f}s"
    ]


def test_a_refused_load_after_an_active_one_leaves_nothing_serving(monkeypatch):
    tile = _kv8_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    status = kv8.enable(lm, enabled=True, tile_status=tile, verify_status=None, window=_WINDOW)
    assert status["state"] == "active" and kv8._ceiling == 84_481
    _refusal(lm, warned=False, tile_status={"state": "off", "reason": None, "splits": None})
    assert not kv8.serving_now() and not kv8.serves(2_000)


@pytest.mark.slow
def test_enable_goes_active_through_the_real_availability_and_probe(monkeypatch):
    tile = _kv8_ready(monkeypatch, fake_availability=False, fake_probe=False)
    monkeypatch.setattr(kv8, "_compile_ok", False)
    lm = tfx.tiny_language_model()
    tfx.verify_ready(monkeypatch, lm)
    status = kv8.enable(
        lm, enabled=True, tile_status=tile, verify_status=dict(_ACTIVE_VERIFY), window=_WINDOW
    )
    assert status["state"] == "active", status
    assert (status["floor"], status["ceiling"]) == (1536, 84_481)
    assert kv8._active == 1 and kv8._compile_ok


def _probe_setup(monkeypatch, tile8=None) -> tuple:
    """`tile8` (the stand-in by default) as kv8's kernel and the tile's
    stand-in as the tile, a far mark of 3,201 keys instead of 57,601 (the
    stand-ins are plain mlx, far slower than the kernels), the floor at N0,
    and the tiny model with its full-attention modules."""
    _gate_state(monkeypatch)
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8 if tile8 is None else tile8)
    monkeypatch.setattr(kv8, "N8", tileattn.N0)
    lm = tfx.tiny_language_model()
    return lm, verifyattn._attention_modules(lm)


def _counters() -> tuple:
    return dict(kv8.calls), dict(tileattn.calls), dict(projkernel.calls)


def _run_probe(modules, *, verify_active: bool) -> str | None:
    """probe() at S = 20, asserting it left the flag clear, the window closed,
    every counter, the tile's flag and mlx-vlm's decode global as it found
    them, whatever it decided."""
    installed, splits = _language().scaled_dot_product_attention, tileattn._active_splits
    before = _counters()
    reason = kv8.probe(modules, 20, tileattn._arch(), verify_active)
    assert kv8._active == 0 and kv8._window is None
    assert _counters() == before
    assert _language().scaled_dot_product_attention is installed
    assert tileattn._active_splits == splits
    return reason


@pytest.mark.slow
def test_probe_passes_the_stand_in_with_the_verifier_half(monkeypatch):
    lm, modules = _probe_setup(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    assert _run_probe(modules, verify_active=True) is None


@pytest.mark.slow
def test_probe_without_exact_verify_never_builds_the_verifier(monkeypatch):
    _, modules = _probe_setup(monkeypatch)
    monkeypatch.setattr(projkernel, "_verifier_class", _untouchable)
    assert _run_probe(modules, verify_active=False) is None


def test_probe_fails_parity_for_a_kernel_that_ignores_the_stepped_chunk(monkeypatch):
    def own_chunk(queries, codes, sc, n, chunk):
        return kfx.stand_in_tile8(queries, codes, sc, n, -(-n // 20))

    _, modules = _probe_setup(monkeypatch, own_chunk)
    reason = _run_probe(modules, verify_active=False)
    assert reason is not None and reason.startswith("parity:"), reason


def test_probe_fails_exactness_for_a_kernel_wrong_the_same_way_on_both_paths(monkeypatch):
    def scaled(queries, codes, sc, n, chunk):
        out = kfx.stand_in_tile8(queries, codes, sc, n, chunk)
        return (out.astype(mx.float32) * 1.1).astype(mx.bfloat16)

    _, modules = _probe_setup(monkeypatch, scaled)
    reason = _run_probe(modules, verify_active=False)
    assert reason is not None and reason.startswith("exactness: T=1 at n=1536 is "), reason


def test_probe_refuses_a_module_whose_calls_never_reach_kv8(monkeypatch):
    _, modules = _probe_setup(monkeypatch)
    # Parity cannot see this fault, so its sweep would only cost time.
    monkeypatch.setattr(kv8, "_probe_parity", lambda *args: None)
    monkeypatch.setattr(tileattn, "_plain_kv_cache", lambda cache: False)
    reason = _run_probe(modules, verify_active=False)
    assert reason == "the model's attention module did not reach kv8"


def test_probe_refuses_a_verifier_that_never_reaches_kv8(monkeypatch):
    lm, modules = _probe_setup(monkeypatch)
    monkeypatch.setattr(kv8, "_probe_parity", lambda *args: None)
    for module in tfx.verify_ready(monkeypatch, lm):
        object.__setattr__(module, verifyattn._TAG, 0)
    reason = _run_probe(modules, verify_active=True)
    assert reason == "the exact verifier's attention did not reach kv8"


@pytest.mark.parametrize("step", ["prepare", "tile8"])
def test_probe_restores_what_it_touched_when_a_step_raises(monkeypatch, step):
    """A raise from the mirror step (prepare) or from the module step (the
    kernel under a forced window, a mirror already built on its cache) still
    leaves the flag clear, the window closed, every counter and the tile's
    flag and decode global as they were."""
    original = {"prepare": kv8.prepare, "tile8": kfx.stand_in_tile8}[step]

    def fails_inside_a_window(*args):
        if kv8._window is not None:
            raise RuntimeError("device lost")
        return original(*args)

    _, modules = _probe_setup(monkeypatch)
    monkeypatch.setattr(kv8, step, fails_inside_a_window)
    monkeypatch.setattr(kv8, "_probe_parity", lambda *args: None)
    installed, splits = _language().scaled_dot_product_attention, tileattn._active_splits
    before = _counters()
    with pytest.raises(RuntimeError, match="device lost"):
        kv8.probe(modules, 20, tileattn._arch(), False)
    assert kv8._active == 0 and kv8._window is None
    assert _counters() == before
    assert _language().scaled_dot_product_attention is installed
    assert tileattn._active_splits == splits


@kfx.nax
def test_probe_passes_the_real_kernels_on_the_tiny_model(monkeypatch):
    _gate_state(monkeypatch)
    lm = tfx.tiny_language_model()
    tfx.verify_ready(monkeypatch, lm)
    start = time.perf_counter()
    reason = _run_probe(verifyattn._attention_modules(lm), verify_active=True)
    assert reason is None, f"{reason} (after {time.perf_counter() - start:.2f}s)"


@kfx.nax
def test_enable_activates_the_real_kernels_on_the_tiny_model(monkeypatch):
    _gate_state(monkeypatch)
    lm = tfx.tiny_language_model()
    tfx.verify_ready(monkeypatch, lm)
    status = kv8.enable(
        lm,
        enabled=True,
        tile_status=dict(_ACTIVE_TILE),
        verify_status=dict(_ACTIVE_VERIFY),
        window=_WINDOW,
    )
    assert status["state"] == "active", status
    assert (status["floor"], status["ceiling"]) == (kv8.N8, kv8.ceiling(_WINDOW, 20))
```

What each group pins:
- `_refusal` holds every refusal to its kind: an INFO line for the tile and the window, one one-line warning for the pins, the kernels, the floor, the probe and any exception, and the flag, the window and the ceiling clear either way.
- The gate-order table breaks guard `first` and every guard after it, so a guard that ran out of order names the wrong reason; its `probe_seconds` check pins that the cost is timed from the kernels' compile on.
- The ceiling tests pin L at the M5 Pro's split target for the windows that matter (84,481 at the default 131,072, 170,241 at 262,144) and, at three split targets, that L's mirror fits and the next boundary's does not.
- The pin tests use the real digests: the locked mlx-vlm passes them, a zeroed digest or a function mlx-vlm removed refuses with a warning.
- The availability tests pin the compile probe's inputs (one quantiser build at N0 + 256 rows, then T = 1..8 at N0 with the step's chunk, all bf16), that the error line reaches the reason and the whole report the log, and that only success is remembered.
- The activation test pins the status, the INFO line, what the probe is handed (the verifier half only with exact verify active) and that the flag and the ceiling rise only after the probe.
- The probe tests run the real probe on the tiny model with the stand-ins, each wrong in one way: a kernel that ignores the stepped chunk (parity), a kernel scaled by 1.1 on both paths (exactness), a cache the decode hook hands back to mlx-vlm (the module never reaches kv8), an untagged verifier (the verifier never reaches kv8), and raises inside a window from the mirror step and the module step (everything is put back).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_kv8.py -v`
Expected: FAIL. The earlier sections' tests still pass, and so do the tests this step added that read only what Tasks 1–2 defined: `test_the_floor_is_a_step_boundary_at_every_measured_split` (`kv8.N8` and tileattn), the 19 cases of `test_the_ceiling_is_the_last_step_boundary_whose_mirror_fits` and `test_the_ceiling_fits_its_mirror_and_the_next_boundary_does_not` (`kv8.ceiling`), and `test_availability_reports_the_error_line_and_remembers_only_success` and `test_compile_probe_runs_the_quantiser_and_every_row_count_once` (`kv8.availability`, `kv8._compile_probe`). Every other test this step added fails with an `AttributeError` naming a kv8 attribute this task adds (`enable`, `VALIDATED_KV8_SOURCES`, `_pins_reason`, `probe` or `_probe_parity`); the two `kfx.nax` tests fail the same way where the marker finds the kernels available and are skipped where it does not.

- [ ] **Step 3: Extend the module's imports**

The import block of `src/sous/engine/kv8.py` must hold at least these after this step (keep `from __future__ import annotations` and every line the earlier tasks added; `uv run ruff check --fix .` sorts the block). mlx, mlx_lm and mlx_vlm stay function-local:

```python
import importlib
import logging
import time
import traceback
import warnings
from collections.abc import Mapping
from typing import Any

from sous.engine import nax, projkernel, tileattn, verifyattn
from sous.engine.int8prefill import _error_line
```

The module must also define `logger = logging.getLogger("sous.engine.kv8")` at top level. If no earlier task added it, add it directly below the imports. Importing `projkernel` here makes no cycle: `projkernel` imports `nax`, `tileattn`, `verifyattn` and `int8prefill`, and `tileattn` imports `kv8` only inside its two seams.

- [ ] **Step 4: Compute the source pins**

Run (one line):

```bash
uv run python -c "from sous.engine.verifyattn import _source_digest as d; pins = [('mlx_vlm.models.cache', q) for q in ('KVCache.update_and_fetch', 'KVCache.trim', 'KVCache.prefix_cache_reserve')] + [('mlx_vlm.speculative.cache_state', q) for q in ('start_speculative_cache', 'SpeculativeCacheTransaction.commit', 'SpeculativeCacheTransaction.abort', '_trim_append_caches', '_cache_position_reader', 'commit_speculative_round', 'abort_speculative_round', 'rollback_speculative_cache')] + [('mlx_vlm.speculative.dflash', q) for q in ('_dflash_rounds', '_dflash_verify', '_dflash_verify_greedy', '_reserve_dflash_target_cache')] + [('mlx_vlm.models.qwen3_5.language', 'LanguageModel.rollback_speculative_cache'), ('mlx_vlm.models.qwen3_5.speculative_verifier', 'Qwen3_5BatchInvariantForward.verify')]; [print(m, q, d(m, q)) for m, q in pins]"
```

Expected (mlx-vlm 0.7.2, the lock; these are the values pasted into `VALIDATED_KV8_SOURCES` in Step 5):

```text
mlx_vlm.models.cache KVCache.update_and_fetch 9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b
mlx_vlm.models.cache KVCache.trim 8e020b3f950f96a9c9ec66502e8bb744764de418d527a7c8616cff80433dcb79
mlx_vlm.models.cache KVCache.prefix_cache_reserve 4233bfad20c4845ab27ddb87d23e7a3827f1d73556100d1a759563ee58aa4109
mlx_vlm.speculative.cache_state start_speculative_cache 5a462e518be86200b48291b75876dcedd2fe3cdab7192ac9c3062b4088273585
mlx_vlm.speculative.cache_state SpeculativeCacheTransaction.commit d4c6ae13b43a0054aeea966bd865156bca534504aa568ee670e11b869d9b52a3
mlx_vlm.speculative.cache_state SpeculativeCacheTransaction.abort f279604e5e1bc5c050cb1fe3bbf1a3d8de13ccf998b184ece45025615f55add2
mlx_vlm.speculative.cache_state _trim_append_caches 03056d9dae1435ff39b01f90d9244a5563d6c70d15d3c44ab9b9ad8fd176677d
mlx_vlm.speculative.cache_state _cache_position_reader 349251fa2854f1021103a62f5502ab79285aedec63c12ece8c060f29ba89e4b6
mlx_vlm.speculative.cache_state commit_speculative_round 58c3e8cf462dea7a576f56b42d0ff38d45daea051b95bc9e9bd8d1ef3698997c
mlx_vlm.speculative.cache_state abort_speculative_round 642f9cc3e46734e260f3cb776398ed4a18ff361f88086cfb01fe2d704d76d69c
mlx_vlm.speculative.cache_state rollback_speculative_cache 67de5bcb251299a27202019a8547635fe74b40d009764586dc3c53f04d27c053
mlx_vlm.speculative.dflash _dflash_rounds eee8dc9ab95a74ed87c74064607736003b7d0dafbdafc32c9fba91dc190d2b32
mlx_vlm.speculative.dflash _dflash_verify 39e4a6cd343ae7e054ae3584ff6879fc895494e413041f099ce08da14e3fc097
mlx_vlm.speculative.dflash _dflash_verify_greedy 1b472dc300554ab6017495bdeb18480be431578aa6958ff7b35c0145e8412971
mlx_vlm.speculative.dflash _reserve_dflash_target_cache a122a70af6663e7a233554b47997e2fb177f8e971c14b6b9aaf928b7bd5e789d
mlx_vlm.models.qwen3_5.language LanguageModel.rollback_speculative_cache 134fe07e0a9fb905a56d4397f8b7bba86729330cac8c8d9a8d16becadb316058
mlx_vlm.models.qwen3_5.speculative_verifier Qwen3_5BatchInvariantForward.verify 65153e7d0b6099869666eb4d10703056b0432c57025db76d808ff46e0c231ae9
```

If any line differs or prints `None`, stop: the installed mlx-vlm is not the lock's (`uv sync`), or the lock moved and every function whose line changed must be re-read against the trim wrap before its digest is replaced. The four digests verifyattn and projkernel already pin for the same functions (`KVCache.update_and_fetch`, `_dflash_rounds`, `_dflash_verify`, `_dflash_verify_greedy`) are the same values.

- [ ] **Step 5: Implement the gates, the pins, availability, the ceiling, the probe and `enable()`**

Append to the end of `src/sous/engine/kv8.py`:

```python
# ---- enable(): the gates, the pins, availability, the ceiling and the probe ---------

# What keeps the mirror's rows the cache's rows inside a decode: verify rows are
# appended through update_and_fetch, which truncates to the offset before it
# grows, so rows below the offset never move; nothing but KVCache.trim lowers an
# offset (prefix_cache_reserve pads and keeps rows and offset); and DFlash's
# commit and abort roll rows back only through the instance trim the mirror
# wraps, never by replaying them. mlx-vlm's rotating-cache path already replays
# accepted rows, and a KVCache moved onto such a path would leave the mirror
# stale with every other check green. sha256 of inspect.getsource, keyed
# (module, qualname), as verifyattn pins its own sources; getsource follows
# __wrapped__, so a class-level wrapper never reads as drift. Validated on
# mlx-vlm 0.7.2; add a digest only after re-reading that function against the
# trim wrap.
VALIDATED_KV8_SOURCES: dict[tuple[str, str], str] = {
    ("mlx_vlm.models.cache", "KVCache.update_and_fetch"): (
        "9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b"
    ),
    ("mlx_vlm.models.cache", "KVCache.trim"): (
        "8e020b3f950f96a9c9ec66502e8bb744764de418d527a7c8616cff80433dcb79"
    ),
    ("mlx_vlm.models.cache", "KVCache.prefix_cache_reserve"): (
        "4233bfad20c4845ab27ddb87d23e7a3827f1d73556100d1a759563ee58aa4109"
    ),
    ("mlx_vlm.speculative.cache_state", "start_speculative_cache"): (
        "5a462e518be86200b48291b75876dcedd2fe3cdab7192ac9c3062b4088273585"
    ),
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.commit"): (
        "d4c6ae13b43a0054aeea966bd865156bca534504aa568ee670e11b869d9b52a3"
    ),
    # Calls cache.trim on the instance, so an aborted round reaches the wrap.
    ("mlx_vlm.speculative.cache_state", "SpeculativeCacheTransaction.abort"): (
        "f279604e5e1bc5c050cb1fe3bbf1a3d8de13ccf998b184ece45025615f55add2"
    ),
    # Resolves getattr(cache, "trim"), so a partial commit reaches the wrap.
    ("mlx_vlm.speculative.cache_state", "_trim_append_caches"): (
        "03056d9dae1435ff39b01f90d9244a5563d6c70d15d3c44ab9b9ad8fd176677d"
    ),
    ("mlx_vlm.speculative.cache_state", "_cache_position_reader"): (
        "349251fa2854f1021103a62f5502ab79285aedec63c12ece8c060f29ba89e4b6"
    ),
    ("mlx_vlm.speculative.cache_state", "commit_speculative_round"): (
        "58c3e8cf462dea7a576f56b42d0ff38d45daea051b95bc9e9bd8d1ef3698997c"
    ),
    ("mlx_vlm.speculative.cache_state", "abort_speculative_round"): (
        "642f9cc3e46734e260f3cb776398ed4a18ff361f88086cfb01fe2d704d76d69c"
    ),
    ("mlx_vlm.speculative.cache_state", "rollback_speculative_cache"): (
        "67de5bcb251299a27202019a8547635fe74b40d009764586dc3c53f04d27c053"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_rounds"): (
        "eee8dc9ab95a74ed87c74064607736003b7d0dafbdafc32c9fba91dc190d2b32"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_verify"): (
        "39e4a6cd343ae7e054ae3584ff6879fc895494e413041f099ce08da14e3fc097"
    ),
    ("mlx_vlm.speculative.dflash", "_dflash_verify_greedy"): (
        "1b472dc300554ab6017495bdeb18480be431578aa6958ff7b35c0145e8412971"
    ),
    ("mlx_vlm.speculative.dflash", "_reserve_dflash_target_cache"): (
        "a122a70af6663e7a233554b47997e2fb177f8e971c14b6b9aaf928b7bd5e789d"
    ),
    ("mlx_vlm.models.qwen3_5.language", "LanguageModel.rollback_speculative_cache"): (
        "134fe07e0a9fb905a56d4397f8b7bba86729330cac8c8d9a8d16becadb316058"
    ),
    ("mlx_vlm.models.qwen3_5.speculative_verifier", "Qwen3_5BatchInvariantForward.verify"): (
        "65153e7d0b6099869666eb4d10703056b0432c57025db76d808ff46e0c231ae9"
    ),
}
# The verify depth the probe's verifier step runs: the block-5 round's rows.
_VERIFY_ROWS = 5
# The row counts the exactness step runs at each of its two marks.
_EXACT_ROWS = (1, 5, 8)
# The quantiser step's random and tie rows; an all-zero row follows them.
_QUANT_RANDOM = 37
_QUANT_TIES = 8
# fp16's largest finite value: the quantiser's scale saturates there.
_FP16_MAX = 65504.0


def _reference_quantize(k: Any, v: Any) -> tuple[Any, Any]:
    """The quantiser's formula in mlx ops, for the probe's bitwise check:
    per (head, row) s = fp16(min(amax / 127, fp16 max)), codes rint(x / s)
    clamped to +-127, divided by 1 instead where s rounded to 0. mx.round rounds
    half to even, as Metal's rint does. Returns (codes int8 [1, 4, R, 512], K
    then V; scales fp16 [1, 4, R, 2], sK then sV)."""
    import mlx.core as mx

    def one(x: Any) -> tuple[Any, Any]:
        x32 = x.astype(mx.float32)
        amax = mx.max(mx.abs(x32), axis=-1, keepdims=True)
        s = mx.minimum(amax / 127, _FP16_MAX).astype(mx.float16)
        f = mx.where(s > 0, s.astype(mx.float32), 1.0)
        return mx.clip(mx.round(x32 / f), -127, 127).astype(mx.int8), s

    k_codes, k_scales = one(k)
    v_codes, v_scales = one(v)
    return (
        mx.concatenate([k_codes, v_codes], axis=-1),
        mx.concatenate([k_scales, v_scales], axis=-1),
    )


def _tie_rows(rows: int) -> Any:
    """[1, 4, rows, 256] bf16 rows whose amax is 127 * 2**e, so the scale is
    exactly 2**e and x / s lands on a half for every element but the amax: the
    rows where the rounding rule shows. The scales come from Python floats, so
    every product is exact."""
    import mlx.core as mx

    lane = mx.arange(tileattn.D)
    halves = (lane % 254).astype(mx.float32) - 126.5
    base = mx.where(lane == 0, 127.0, halves)
    scales = mx.array([(-1) ** r * 2.0 ** (r % 8 - 3) for r in range(rows)], dtype=mx.float32)
    tied = scales[:, None] * base[None, :]
    shape = (1, tileattn.HKV, rows, tileattn.D)
    return mx.broadcast_to(tied[None, None], shape).astype(mx.bfloat16)


def _dequant_q(queries: Any) -> Any:
    """What the kernel's in-kernel Q quantisation means, in fp32: per (query
    head, row) codes rint(q / sq) clamped to +-127, times sq = amax / 127; a
    zero row stays zero."""
    import mlx.core as mx

    q32 = queries.astype(mx.float32)
    sq = mx.max(mx.abs(q32), axis=-1, keepdims=True) / 127
    f = mx.where(sq > 0, sq, 1.0)
    return mx.clip(mx.round(q32 / f), -127, 127) * sq


def _reference_attention(q32: Any, k32: Any, v32: Any) -> Any:
    """fp32 attention on the GPU (fp64 is CPU-only in mlx): T query rows over n
    keys, row r seeing keys [0, n - T + r], each KV head shared by the six
    query heads after it. Returns [1, 24, T, 256] fp32."""
    import mlx.core as mx

    t, n = q32.shape[2], k32.shape[2]
    q = q32.reshape(1, tileattn.HKV, tileattn.GQA, t, tileattn.D)
    scores = (q @ k32[:, :, None].swapaxes(-1, -2)) * tileattn.SCALE
    keep = mx.arange(n)[None, :] <= (n - t + mx.arange(t))[:, None]
    weights = mx.softmax(mx.where(keep, scores, float("-inf")), axis=-1)
    return (weights @ v32[:, :, None]).reshape(1, tileattn.HQ, t, tileattn.D)


def _distance(got: Any, want: Any) -> float:
    """Relative RMS of `got` from `want` in fp32; infinite when `got` is not
    finite everywhere, so a NaN fails every bound."""
    import mlx.core as mx

    if not mx.all(mx.isfinite(got)).item():
        return float("inf")
    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _rows(qpool: Any, start: int, t: int) -> Any:
    """T query rows of the [1, 20, 24, 256] pool as serving hands them over:
    a transposed [1, 24, T, 256] view."""
    return qpool[:, start : start + t].transpose(0, 2, 1, 3)


def _run(queries: Any, codes: Any, sc: Any, n: int, splits: int) -> Any:
    """[1, 24, T, 256]: the one-row call at T = 1, the verify call above it."""
    if queries.shape[2] == 1:
        return decode_codes(queries, codes, sc, n, splits)
    return verify_codes(queries, codes, sc, n, splits)


def _restored(offset: int, key: Any) -> Any:
    """A KVCache as a fork restore leaves it: keys, values, then offset
    assigned (the state setter would derive the offset from the padded shape),
    the 256-step padded buffer NaN past the live rows. The same key gives the
    same cache, so a served and an unserved call can start from one state."""
    import mlx.core as mx

    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    capacity = -(-(offset + tileattn.MAX_RUN + 1) // 256) * 256
    k_key, v_key = mx.random.split(key)
    live = (1, tileattn.HKV, offset, tileattn.D)
    tail = mx.full(
        (1, tileattn.HKV, capacity - offset, tileattn.D), float("nan"), dtype=mx.bfloat16
    )
    cache: Any = kv_cache()
    keys = mx.random.normal(live, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(live, key=v_key).astype(mx.bfloat16)
    cache.keys = mx.concatenate([keys, tail], axis=2)
    cache.values = mx.concatenate([values, tail], axis=2)
    cache.offset = offset
    mx.eval(cache.keys, cache.values)
    return cache


def _probe_quantiser(key: Any) -> str | None:
    """The quantiser bitwise against _reference_quantize on random rows, tie
    rows and an all-zero row, read through strided views of a padded buffer
    into a capacity-sized output, as a first build writes it; and rows added
    in two steps bitwise equal to one build."""
    import mlx.core as mx

    k_key, v_key = mx.random.split(key)
    random_shape = (1, tileattn.HKV, _QUANT_RANDOM, tileattn.D)
    ties = _tie_rows(_QUANT_TIES)
    zero = mx.zeros((1, tileattn.HKV, 1, tileattn.D), dtype=mx.bfloat16)
    pad = mx.zeros((1, tileattn.HKV, 256, tileattn.D), dtype=mx.bfloat16)
    kbuf = mx.concatenate(
        [mx.random.normal(random_shape, key=k_key).astype(mx.bfloat16), ties, zero, pad], axis=2
    )
    vbuf = mx.concatenate(
        [mx.random.normal(random_shape, key=v_key).astype(mx.bfloat16), -ties, zero, pad], axis=2
    )
    mx.eval(kbuf, vbuf)
    rows = _QUANT_RANDOM + _QUANT_TIES + 1
    k, v = kbuf[:, :, :rows], vbuf[:, :, :rows]
    codes, sc = quantize(k, v, -(-rows // 256) * 256)
    want_codes, want_sc = _reference_quantize(k, v)
    mx.eval(codes, sc, want_codes, want_sc)
    segments = (
        ("random", 0, _QUANT_RANDOM),
        ("tie", _QUANT_RANDOM, _QUANT_RANDOM + _QUANT_TIES),
        ("all-zero", rows - 1, rows),
    )
    for label, a, b in segments:
        if not (
            mx.array_equal(codes[:, :, a:b], want_codes[:, :, a:b]).item()
            and mx.array_equal(sc[:, :, a:b], want_sc[:, :, a:b]).item()
        ):
            return f"quantiser: the {label} rows differ from the reference"
    # A cut inside the random rows, at neither end and not on a power of two.
    cut = _QUANT_RANDOM // 2 + 1
    first_codes, first_sc = quantize(kbuf[:, :, :cut], vbuf[:, :, :cut], cut)
    rest_codes, rest_sc = quantize(kbuf[:, :, cut:rows], vbuf[:, :, cut:rows], rows - cut)
    steps_codes = mx.concatenate([first_codes, rest_codes], axis=2)
    steps_sc = mx.concatenate([first_sc, rest_sc], axis=2)
    if not (
        mx.array_equal(steps_codes, codes[:, :, :rows]).item()
        and mx.array_equal(steps_sc, sc[:, :, :rows]).item()
    ):
        return "quantiser: rows added in two steps differ from one build"
    return None


def _probe_exactness(
    kb: Any, vb: Any, qpool: Any, firsts: tuple[int, ...], splits: int
) -> str | None:
    """What parity cannot see in a kernel wrong the same way on both paths:
    the kernel within REL_RMS_BOUND of fp32 attention over the very codes it
    reads (K, V and Q dequantised), for calls whose first row sees each of
    `firsts` keys, at T = 1, 5 and 8, with x8 queries, a dominant key at
    n - 1 and a decoy at n, just past the live range, that no read may
    reach."""
    import mlx.core as mx

    ones = mx.ones((1, tileattn.HKV, 1, tileattn.D), dtype=mx.bfloat16)
    for first in firsts:
        for t in _EXACT_ROWS:
            n = first + t - 1
            q8 = (_rows(qpool, 0, t).astype(mx.float32) * 8).astype(mx.bfloat16)
            # The last row's own query per KV head: only that row sees key n - 1.
            heads = q8[:, :, t - 1].reshape(1, tileattn.HKV, tileattn.GQA, tileattn.D)
            dominant = heads[:, :, :1]
            kp = mx.concatenate([kb[:, :, : n - 1], dominant, dominant, kb[:, :, n + 1 :]], axis=2)
            vp = mx.concatenate([vb[:, :, : n - 1], ones, -ones, vb[:, :, n + 1 :]], axis=2)
            mx.eval(q8, kp, vp)
            codes, sc = quantize(kp, vp, kp.shape[2])
            mx.eval(codes, sc)
            got = _run(q8, codes, sc, n, splits)
            k32, v32 = dequant(codes[:, :, :n], sc[:, :, :n])
            want = _reference_attention(_dequant_q(q8), k32, v32)
            distance = _distance(got, want)
            if not distance <= REL_RMS_BOUND:
                return (
                    f"exactness: T={t} at n={n} is {distance:.3g} from the fp32 reference"
                    f" (bound {REL_RMS_BOUND})"
                )
    return None


def _probe_parity(codes: Any, sc: Any, qpool: Any, marks: list[int], splits: int) -> str | None:
    """What greedy parity with the drafter on or off rests on: every verify
    row array_equal to the one-row call at its own key count, at each mark
    for T = 1..MAX_RUN, with the rows below the mark, straddling it and from
    it (the tile's three placements). Query row i of the pool sits at
    position mark - 10 + i."""
    import mlx.core as mx

    for mark in marks:
        base = mark - 10
        dec = {
            pos + 1: decode_codes(_rows(qpool, pos - base, 1), codes, sc, pos + 1, splits)
            for pos in range(base, base + 20)
            if pos + 1 >= N8
        }
        mx.eval(list(dec.values()))
        for t in range(1, tileattn.MAX_RUN + 1):
            for prefix in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}):
                if prefix + 1 < N8:
                    continue  # rows below the floor are the tile's, not kv8's
                out = verify_codes(_rows(qpool, prefix - base, t), codes, sc, prefix + t, splits)
                same = mx.stack(
                    [mx.array_equal(out[:, :, r], dec[prefix + r + 1][:, :, 0]) for r in range(t)]
                )
                if not mx.all(same).item():
                    return f"parity: verify rows at prefix {prefix}, T={t} differ from decode"
    return None


def _probe_fence(
    codes: Any, sc: Any, qpool: Any, points: tuple[tuple[int, int], ...], splits: int
) -> str | None:
    """Reads stay in [0, n). Rows past n set to -128 codes (never a
    code the quantiser writes) and NaN scales, read whole and through a view
    ending exactly at n over the capacity-sized buffer (whose scale plane keeps
    its 16-byte head stride), give the clean result bitwise, and finite."""
    import mlx.core as mx

    cap = codes.shape[2]
    for n, t in points:
        queries = _rows(qpool, 0, t)
        want = _run(queries, codes, sc, n, splits)
        tail = (1, tileattn.HKV, cap - n)
        bad_codes = mx.concatenate(
            [codes[:, :, :n], mx.full((*tail, 2 * tileattn.D), -128, dtype=mx.int8)], axis=2
        )
        bad_sc = mx.concatenate(
            [sc[:, :, :n], mx.full((*tail, 2), float("nan"), dtype=mx.float16)], axis=2
        )
        mx.eval(want, bad_codes, bad_sc)
        for c, s in ((bad_codes, bad_sc), (bad_codes[:, :, :n], bad_sc[:, :, :n])):
            got = _run(queries, c, s, n, splits)
            if not (mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()):
                return f"reads past n: T={t} at n={n} differs from the clean buffer"
    return None


def _probe_mirror(key: Any, held: list[Any]) -> str | None:
    """The mirror rule on a real KVCache under a forced window: after
    every served call the mirror equals a bulk re-quantisation of the cache's
    rows. In order: a first build over a cache whose capacity is not a multiple
    of 4; a trim through the wrap, an unserved append over the trimmed rows,
    then a served call; the shrinking block across the floor; a mirror left
    from another window, which the next window rebuilds; a growth."""
    import mlx.core as mx

    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    # An odd-sized first segment: the truncating growth that follows it leaves
    # the cache a capacity that is not a multiple of 4.
    first = N8 + 1 if (N8 + 1) % 4 else N8 + 2
    k_key, v_key = mx.random.split(key)
    pool = (1, tileattn.HKV, first + GROWTH + 1024, tileattn.D)
    kpool = mx.random.normal(pool, key=k_key).astype(mx.bfloat16)
    vpool = mx.random.normal(pool, key=v_key).astype(mx.bfloat16)
    mx.eval(kpool, vpool)
    cache: Any = kv_cache()
    held.append(cache)
    drawn = 0

    def append(rows: int) -> None:
        # Fresh rows on every call, so an append over trimmed rows changes them.
        nonlocal drawn
        cache.update_and_fetch(kpool[:, :, drawn : drawn + rows], vpool[:, :, drawn : drawn + rows])
        mx.eval(cache.keys, cache.values)
        drawn += rows

    def requantised(mirror: Mirror) -> bool:
        off = cache.offset
        codes, sc = quantize(cache.keys[:, :, :off], cache.values[:, :, :off], off)
        return bool(
            mirror.n == off
            and mirror.cap >= off
            and mirror.cap % 256 == 0
            and mx.array_equal(mirror.codes[:, :, :off], codes).item()
            and mx.array_equal(mirror.sc[:, :, :off], sc).item()
        )

    open_window(force=True)
    append(first)
    append(256)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: the first build differs from a bulk re-quantisation"
    cache.trim(3)
    if getattr(cache, MIRROR_ATTR).n != cache.offset:
        return "mirror: a trim did not reach the mirror"
    append(3)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a trim and an unserved append differ from a bulk re-quantisation"
    # 16 rows served from just below the floor, rolled back below it, 8 rows
    # appended below it without a served call, then a served call.
    cache.trim(cache.offset - (N8 - 4))
    append(16)
    prepare(cache, 16)
    cache.trim(24)
    append(8)
    append(8)
    if not requantised(prepare(cache, 8)):
        return "mirror: the shrinking block across the floor differs from a bulk re-quantisation"
    # The cache moves under a mirror from another window with no wrap in the
    # way (the class's own trim), as a restore between decodes would.
    close_window([])
    type(cache).trim(cache, 8)
    append(8)
    open_window(force=True)
    builds = calls["mirror_builds"]
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a mirror from another window differs from a bulk re-quantisation"
    if calls["mirror_builds"] != builds + 1:
        return "mirror: a mirror from another window was not rebuilt"
    grown = calls["mirror_grown"]
    append(GROWTH + 256)
    append(1)
    if not requantised(prepare(cache, 1)):
        return "mirror: a growth differs from a bulk re-quantisation"
    if calls["mirror_grown"] != grown + 1:
        return "mirror: a growth was not counted"
    close_window([cache])
    return None


def _probe_stock(
    kb: Any, vb: Any, codes: Any, sc: Any, qpool: Any, ns: tuple[int, ...], splits: int
) -> str | None:
    """The format's own error: one-row calls over Gaussian K/V within
    STOCK_RMS_BOUND of stock bf16 attention over the same rows unquantised."""
    import mlx.core as mx

    queries = _rows(qpool, 0, 1)
    for n in ns:
        got = decode_codes(queries, codes, sc, n, splits)
        want = mx.fast.scaled_dot_product_attention(
            queries, kb[:, :, :n], vb[:, :, :n], scale=tileattn.SCALE, mask=None
        )
        distance = _distance(got, want)
        if not distance <= STOCK_RMS_BOUND:
            return (
                f"quantisation: n={n} is {distance:.3g} from stock attention"
                f" (bound {STOCK_RMS_BOUND})"
            )
    return None


def _probe_module(
    modules: list[Any], splits: int, verify_active: bool, n: int, key: Any, held: list[Any]
) -> str | None:
    """The loaded model's own attention module, projections and rope
    included, one row through mlx-vlm's decode global with a forced window open
    against the same call on the bf16 path, both over a restored-style KVCache;
    then, with exact verify active, the same module through the exact
    verifier's wrapper over _VERIFY_ROWS rows. Each served call must move its
    counter by one. Without a drafter kv8 serves decode only, so the verifier
    is never built."""
    import mlx.core as mx

    language: Any = importlib.import_module(tileattn._DECODE_MODULE)
    if not getattr(language.scaled_dot_product_attention, tileattn._HOOK_MARK, False):
        language.scaled_dot_product_attention = tileattn._decode_wrapper(
            language.scaled_dot_product_attention
        )
    tileattn._active_splits = splits
    module = modules[0]
    hidden = module.o_proj.weight.shape[0]
    x_key, c_key, v_key = mx.random.split(key, 3)
    x = mx.random.normal((1, 1, hidden), key=x_key).astype(mx.bfloat16)
    xs = mx.random.normal((1, _VERIFY_ROWS, hidden), key=v_key).astype(mx.bfloat16)
    mx.eval(x, xs)

    def restored(offset: int) -> Any:
        cache = _restored(offset, c_key)
        held.append(cache)
        return cache

    def served(call: Any, offset: int) -> Any:
        open_window(force=True)
        cache = restored(offset)
        out = call(cache)
        mx.eval(out)
        close_window([cache])
        return out

    want = module(x, mask=None, cache=restored(n - 1))
    mx.eval(want)
    before = calls["decode_kv8"]
    got = served(lambda cache: module(x, mask=None, cache=cache), n - 1)
    if calls["decode_kv8"] != before + 1:
        return "the model's attention module did not reach kv8"
    distance = _distance(got, want)
    if not distance <= STOCK_RMS_BOUND:
        return (
            f"correctness: the model's module is {distance:.3g} from the bf16 path"
            f" (bound {STOCK_RMS_BOUND})"
        )
    if not verify_active:
        return None
    verifier = projkernel._verifier_class()()

    def verify(cache: Any) -> Any:
        return verifier._attention(module, xs, "causal", cache, None, None)

    want = verify(restored(n))
    mx.eval(want)
    before = calls["verify_kv8"]
    got = served(verify, n)
    if calls["verify_kv8"] != before + 1:
        return "the exact verifier's attention did not reach kv8"
    distance = _distance(got, want)
    if not distance <= STOCK_RMS_BOUND:
        return (
            f"correctness: the verifier's attention is {distance:.3g} from the bf16 path"
            f" (bound {STOCK_RMS_BOUND})"
        )
    return None


def _probe_steps(
    modules: list[Any], splits: int, verify_active: bool, held: list[Any]
) -> str | None:
    """Every input is made with GPU ops and evaluated before the launches that
    read it, for the deadlock _compile_probe() describes. The marks: the floor
    N8, the next two step boundaries and the far mark, the first boundary at or
    past tileattn.PROBE_FAR_KEYS."""
    import mlx.core as mx

    global _active
    _active = 1
    marks = [b for b in tileattn.boundaries(N8 + 3 * tileattn.GRAN * splits, splits) if b >= N8]
    marks = marks[:3]
    far = max(tileattn.far_mark(splits), marks[-1])
    mid_step = (marks[1] + marks[2]) // 2
    q_key, k_key, v_key, quant_key, mirror_key, module_key = mx.random.split(mx.random.key(0), 6)

    reason = _probe_quantiser(quant_key)
    if reason is not None:
        return reason
    # Room past the far mark for the parity rows, the decoy and the fence.
    cap = -(-(far + 2 * tileattn.MAX_RUN + 1) // 256) * 256
    kv_shape = (1, tileattn.HKV, cap, tileattn.D)
    kb = mx.random.normal(kv_shape, key=k_key).astype(mx.bfloat16)
    vb = mx.random.normal(kv_shape, key=v_key).astype(mx.bfloat16)
    qpool = mx.random.normal((1, 20, tileattn.HQ, tileattn.D), key=q_key).astype(mx.bfloat16)
    mx.eval(kb, vb, qpool)
    codes, sc = quantize(kb, vb, cap)
    mx.eval(codes, sc)
    fence = ((N8, 1), (mid_step, 3), (far + tileattn.MAX_RUN - 1, tileattn.MAX_RUN))
    reason = (
        _probe_exactness(kb, vb, qpool, (N8, far), splits)
        or _probe_parity(codes, sc, qpool, sorted({*marks, far}), splits)
        or _probe_fence(codes, sc, qpool, fence, splits)
        or _probe_mirror(mirror_key, held)
        or _probe_stock(kb, vb, codes, sc, qpool, (N8, far), splits)
    )
    # Hundreds of MB at the far mark: gone before the module step allocates.
    del kb, vb, qpool, codes, sc
    if reason is not None:
        return reason
    return _probe_module(modules, splits, verify_active, mid_step, module_key, held)


def probe(modules: list[Any], splits: int, arch: str, verify_active: bool) -> str | None:
    """None when kv8 may serve `modules` on this GPU at `splits` splits, else
    why not. Runs on the loading thread after availability(); `arch` is the
    tile's GPU architecture, taken for the tile probe's signature, and every
    row probed here is at or past N8 >= N0, where no stock plan runs, so no
    step reads it. Raises the flag and opens forced windows only while it
    runs; in `finally` it closes the window and drops every mirror it built
    (with their trim wraps), clears the flag, puts back the tile's flag and
    mlx-vlm's decode global, restores kv8's, the tile's and the projection
    kernel's counters (a quantized model's projections reach the projection
    kernel's hooks in the module step) and clears mlx's cache. Its arrays
    are gone before the cache is cleared, so the budget measured next reads
    true."""
    import mlx.core as mx

    global _active, _window
    language: Any = importlib.import_module(tileattn._DECODE_MODULE)
    installed = language.scaled_dot_product_attention
    tile_splits = tileattn._active_splits
    counted = (dict(calls), dict(tileattn.calls), dict(projkernel.calls))
    held: list[Any] = []
    try:
        return _probe_steps(modules, splits, verify_active, held)
    except BaseException as e:
        # The traceback holds the steps' frames, and every probe array in them,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        try:
            if held:
                if _window is None:
                    open_window(force=True)
                close_window(held)
        finally:
            held.clear()
            _active, _window = 0, None
            language.scaled_dot_product_attention = installed
            tileattn._active_splits = tile_splits
            calls.update(counted[0])
            tileattn.calls.update(counted[1])
            projkernel.calls.update(counted[2])
            mx.clear_cache()


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use kv8: report why. An `expected` refusal says only
    that kv8 does not apply here (the tile is not active, the window fits no
    ceiling above the floor), which is how most Macs load a default-on kv8:
    one INFO line, and the reason in the status. Any other refusal means it
    should have run and did not (a pinned source that drifted, kernels this OS
    rejects, a floor that is no cut point, a failed probe): one warning, since
    a switch that is silently inert there hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("int8 K/V mirror unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: int8 K/V mirror unavailable ({reason}); "
            "decode and verify attention read the bf16 cache",
            stacklevel=3,
        )
    return {
        "state": "unavailable",
        "reason": reason,
        "probe_seconds": probe_seconds,
        "floor": None,
        "ceiling": None,
    }


def _tile_reason(tile_status: Mapping[str, Any] | None) -> str | None:
    """kv8 rides on the attention tile: its platform gate, S, the chunk
    schedule, verify_plan, MAX_RUN, the reduce kernel and, with a drafter,
    exact verify attention are all the tile's."""
    state = (tile_status or {}).get("state")
    if state == "active":
        return None
    if state == "off":
        return "attention tile off"
    return (tile_status or {}).get("reason") or f"attention tile {state or 'unknown'}"


def _pins_reason() -> str | None:
    """Which pinned mlx-vlm function changed, or None. An unreadable source (a
    function mlx-vlm removed) counts as changed."""
    for (module, qualname), digest in VALIDATED_KV8_SOURCES.items():
        if verifyattn._source_digest(module, qualname) != digest:
            return f"mlx-vlm {qualname} changed"
    return None


def enable(
    model: Any,
    *,
    enabled: bool,
    tile_status: Mapping[str, Any] | None,
    verify_status: Mapping[str, Any] | None,
    window: int,
) -> dict[str, Any]:
    """Decide whether this load's decode and verify attention read int8 K/V
    from N8 keys up to the ceiling L. `window` is [model].max_context_tokens,
    which L is computed from. Returns the status the engine exposes and never
    raises: a model load must not fail because an accelerator is missing.
    probe_seconds covers the kernels' compile and the probe, kv8's whole cost
    to a load."""
    global _active, _window, _ceiling
    # First, on every call: a flag, a window or a ceiling left from an earlier
    # load must never serve this one, whatever happens below.
    _active, _window, _ceiling = 0, None, 0
    if not enabled:
        return {
            "state": "off",
            "reason": None,
            "probe_seconds": None,
            "floor": None,
            "ceiling": None,
        }
    seconds: float | None = None
    try:
        reason = _tile_reason(tile_status)
        if reason is not None:
            return _refuse(reason, expected=True)
        splits = int((tile_status or {}).get("splits") or 0)
        reason = _pins_reason()
        if reason is not None:
            return _refuse(reason)
        modules = verifyattn._attention_modules(model)
        # The exact verifier runs only beside a drafter, so only then does the
        # probe take its half.
        verify_active = (verify_status or {}).get("state") == "active"
        limit = 0
        expected = False
        start = time.perf_counter()
        try:
            available = availability()
            if not available.available:
                reason = available.reason or "the int8 K/V kernels did not run"
            elif (reason := _floor_reason(splits)) is not None:
                pass
            else:
                limit = ceiling(window, splits)
                if limit <= N8:
                    expected = True
                    reason = (
                        f"window below the kv8 floor (window {window}, ceiling {limit}, floor {N8})"
                    )
                else:
                    reason = probe(modules, splits, tileattn._arch(), verify_active)
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds, expected=expected)
        _active, _ceiling = 1, limit
        logger.info(
            "int8 K/V mirror: %d layers, %d <= keys < %d, probe %.2fs",
            len(modules),
            N8,
            limit,
            seconds,
        )
        return {
            "state": "active",
            "reason": None,
            "probe_seconds": seconds,
            "floor": N8,
            "ceiling": limit,
        }
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active, _window, _ceiling = 0, None, 0
        # The warning carries one line; the traceback goes to the log.
        logger.warning("int8 K/V mirror: enabling failed", exc_info=True)
        return _refuse(str(e) or type(e).__name__, seconds)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_kv8.py -v`
Expected: PASS. All 61 tests this task adds pass, except that the two `kfx.nax` tests (`test_probe_passes_the_real_kernels_on_the_tiny_model`, `test_enable_activates_the_real_kernels_on_the_tiny_model`) are skipped wherever kv8's kernels are unavailable; they run on the M5 Pro. The three slow-marked runs, which take the probe through every step on the stand-ins, take several seconds each.

- [ ] **Step 7: Check the pins in a process with every wrapper installed**

Run:

```bash
uv run python -c "from sous.engine import int8prefill, kv8, projkernel, tileattn, verifyattn; print(kv8._pins_reason()); int8prefill.install_wrappers(); projkernel.install_hooks(); tileattn.install_decode_hook(); verifyattn.install_wrapper(); print(kv8._pins_reason())"
```

Expected: `None` twice. The first proves the digests against the locked mlx-vlm; the second that no class-level wrapper sous installs (int8's, the projection kernel's hooks, the tile's decode hook, the verifier's wrapper) moves a pinned source, since `getsource` follows `__wrapped__` and none of them wraps a pinned function.

- [ ] **Step 8: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add src/sous/engine/kv8.py tests/test_kv8.py
git commit -m "feat(engine): gate and probe the int8 K/V mirror at load" -m "The mirror serves decode and verify attention from int8 copies of the
cache, so a load must prove it may before the flag is raised. It rides
on the attention tile and refuses wherever the tile is not active. Its
validity rests on mlx-vlm appending verify rows through update_and_fetch
and rolling them back only through the instance trim, so seventeen
source pins cover every function on that path, and a changed one warns.

The ceiling L is the last step boundary whose mirror fits beside the
turn's bf16 KV inside the window-sized reserve, a function of the window
and S alone, so a request serves the same rows whatever else the daemon
holds; a window whose ceiling is not above the floor is an expected
refusal. The probe runs the quantiser bitwise against its formula, the
kernel against fp32 attention over the same codes, verify rows against
one-row calls at four marks, the fence past n, the mirror rule on a real
KVCache through the trim wrap, the format's error against stock, and the
loaded model's own module through both hooks."
git log -1 --format='%G? %an <%ae> %cn <%ce>%n%B'
```

Expected: ruff and ty report nothing, the full non-model suite passes, and the log line starts `G krcm0209 <17315774+krcm0209@users.noreply.github.com> krcm0209 <17315774+krcm0209@users.noreply.github.com>` with no trailer after the body.


### Task 5: Config and engine wiring

**Files:**
- Modify: `src/sous/config.py`:
  - `_KNOWN["model"]` at :31-34, gaining `"kv_int8"` after `"projection_kernel"`;
  - the new `kv_int8` field after `projection_kernel` at :155;
  - the new `_kv_int8` after `_projection_kernel`, which ends at :471;
  - `load_config`'s constructor call at :570-571.
- Modify: `src/sous/engine/promptcache.py`: `Speculation` at :463-475.
- Modify: `src/sous/engine/vlm.py`:
  - the module imports at :16;
  - the `__init__` signature at :78-80;
  - a new `kv8.enable` call after `_pin_block_size` at :157, before `draftctx.enable` at :158-160;
  - `decode()` at :455-479;
  - `unload()` at :590-595.
- Modify: `src/sous/engine/lm.py`: `__init__` at :58-59 and `speculation()` at :216-218.
- Modify: `src/sous/engine/base.py`:
  - `_default_factory`'s signature at :263-267 and its VLM branch at :326-334;
  - `default_engine_factory` at :379-380;
  - `ManagedEngine`, after the `projection_kernel_status` property at :427-432.
- Test:
  - `tests/test_config.py`, after the projection_kernel tests at :245-253;
  - `tests/test_engine_base.py` after :1322-1324, after :1551-1552, after :1650-1651, and at :2324-2344;
  - `tests/test_engine_unloaded.py` at :99-131 and a new test after :82;
  - `tests/test_kv8.py`: a new section appended at the end of the file.

Line numbers are those of the branch head `be778e7`. If an earlier task of this plan has already edited one of these files, find each replaced block by its quoted text; every quoted block is unique in its file.

**Interfaces:**
- Consumes (from the tasks that create `src/sous/engine/kv8.py` and `tests/test_kv8.py`; names and signatures as the interface sheet fixes them):
  - `kv8.enable(model, *, enabled: bool, tile_status, verify_status, window: int) -> dict` with keys `state`, `reason`, `probe_seconds`, `floor` and `ceiling`. With `enabled=False` its first statement clears `_active` and `_window` and it returns `{"state": "off", "reason": None, "probe_seconds": None, "floor": None, "ceiling": None}` before any gate, which is what keeps the stub models of the existing VLMEngine tests away from every kv8 gate.
  - `kv8.clear() -> None` (`_active = 0`, `_window = None`).
  - `kv8.open_window(*, force: bool = False) -> Window` (asserts no window is open; bumps `_serial`; `calls["windows"] += 1`).
  - `kv8.close_window(cache: Sequence[Any]) -> tuple[str, int]` (drops every mirror and trim wrap on the layers of `cache`, returns the window's `(state, rows)`, clears `_window`, calls `mx.clear_cache()` only when a mirror was built).
  - `kv8.serving_now() -> bool`, `kv8.Window` (mutable dataclass: `serial`, `state="idle"`, `rows=0`, `served=0`, `built=False`, `force=False`), and the module state `kv8._active`, `kv8._window`, `kv8._serial`, `kv8.calls`.
  - Existing: `projkernel.enable`, `tileattn.enable`, `verifyattn.enable`, `draftctx.enable`, `draftctx.OFF`, `vlm._pin_block_size`, `vlm.ForkStore`, `VLMEngine._fork_key_fields`, `base.measure_cache_budget`; the test helpers `_cfg`, `_stub`, `_positionless_model`, `_RecordingTokenizer` and `FakeEngine` in test_engine_base.py, and `_unloaded_lm`, `_unloaded_vlm` and `_ContextThatFailsToClose` in test_engine_unloaded.py.
- Produces:
  - `SousConfig.kv_int8: bool = True`, after `projection_kernel`, and `"kv_int8"` in `_KNOWN["model"]`.
  - `config._kv_int8(model: dict) -> bool`: absent gives True; a bool gives itself; anything else warns `sous config: [model].kv_int8 {value!r} must be true or false; using false` and gives False.
  - `Speculation.kv8: str = "off"` and `Speculation.kv8_rows: int = 0`, after `context`. `LMEngine.speculation()` keeps returning `Speculation()`, so both read `off` and 0 there.
  - `VLMEngine.__init__(..., attention_tile: bool = False, projection_kernel: bool = False, kv_int8: bool = False, kv_int8_window: int = 0, fork_dir, fork_budget, weights_identity)`.
  - `VLMEngine.kv_int8_status: dict`, set from `kv8.enable(self._model, enabled=kv_int8, tile_status=self.attention_tile_status, verify_status=self.verify_attention_status, window=kv_int8_window)`.
  - Load order in `VLMEngine.__init__`: int8 → drafter → verifyattn → tileattn → projkernel → block resolution and `_pin_block_size` → kv8 → draftctx → ForkStore → `measure_cache_budget`.
  - `VLMEngine.decode()`: when `kv8._active`, calls `kv8.open_window()` inside the draft-context `seed` block, immediately before the `stream_generate` loop, and `kv8.close_window(cache)` in that loop's `finally`, still inside the `seed` block; the `Speculation` built after the loop carries the pair as `kv8=` and `kv8_rows=`, and `("off", 0)` when no window opened. An exception out of the generation leaves `self._speculation` as it was.
  - `VLMEngine.unload()` calls `kv8.clear()` right after `projkernel.clear()`.
  - vlm.py's module imports gain `kv8`.
  - `LMEngine.kv_int8_status = {"state": "off", "reason": None, "probe_seconds": None, "floor": None, "ceiling": None}`, set before the fork store.
  - `base._default_factory(..., projection_kernel: bool = False, kv_int8: bool = False, fork_dir=..., fork_budget=...)`, passing `kv_int8=kv_int8` and `kv_int8_window=reserve_tokens` to `vlm.VLMEngine` only.
  - `default_engine_factory` passes `kv_int8=config.kv_int8`; the window travels as the `reserve_tokens=config.max_context_tokens` it already passes.
  - `ManagedEngine.kv_int8_status -> dict | None`, `getattr(self._inner, "kv_int8_status", None)`.

Why the defaults differ:
- The config default (`kv_int8 = True`) is what the daemon and `sous tune` get through `default_engine_factory(config)`.
- The constructors default to `kv_int8=False` and `kv_int8_window=0`, as `int8_prefill`, `attention_tile` and `projection_kernel` do. An engine built directly (the stub tests, the model tests, scripts) keeps the bf16 attention paths; and a window of 0 gives a ceiling of 0, so even an engine that asks for kv8 without a window serves nothing outside a forced window.
- `ManagedEngine.kv_int8_status` reads with a `None` default, not with an `off` dict, so a fake stays silent on the load line and in the status document, as it does for the tile and the projection kernel. The LM backend reports `off` itself.

Why kv8 enables after the pin and before draftctx:
- It reads the tile's status and the verifier's, both settled once `projkernel.enable` has run; it reads neither the projection kernel's status nor the block, so it sits after the block resolution, which belongs with the kernel.
- It must run before the fork store and before `measure_cache_budget`, so that the probe's mirrors and transient arrays are released (its `finally` calls `mx.clear_cache()`) when the budget is measured. draftctx wraps only the drafter's instance methods, which the kv8 probe never calls, so either order with draftctx is safe; before it keeps every kernel's enable in one run.

Why the window is read from `kv8._active` and not from `self.kv_int8_status`:
- `test_draftctx._engine`, `test_engine_positions._engine` and the kv8 contract tests build engines with `object.__new__(VLMEngine)` and set by hand only the attributes `decode()` reads. A module read adds no attribute to that list, and `verifyattn` already reads `tileattn._active_splits` the same way.
- After `unload()` the flag is 0, so a decode on a stale engine (which `_loaded()` refuses anyway) could never open a window.

- [ ] **Step 1: Write the failing config tests**

In `tests/test_config.py`, after the projection_kernel tests:

Replace:

```python
    with pytest.warns(
        UserWarning, match=r"\[model\]\.projection_kernel .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.projection_kernel is False
```

with:

```python
    with pytest.warns(
        UserWarning, match=r"\[model\]\.projection_kernel .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.projection_kernel is False


# ---- [model].kv_int8 ----------------------------------------------------------


def test_kv_int8_defaults_on(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x/y'\n")
    assert load_config(p).kv_int8 is True
    assert SousConfig().kv_int8 is True


def test_kv_int8_reads_false_without_an_unknown_key_warning(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nkv_int8 = false\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.kv_int8 is False


@pytest.mark.parametrize("bad", ['"yes"', "1", "0.5"])
def test_kv_int8_rejects_non_booleans_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nkv_int8 = {bad}\n")
    with pytest.warns(
        UserWarning, match=r"\[model\]\.kv_int8 .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.kv_int8 is False
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k kv_int8 -v`

Expected: 5 failed, 68 deselected.
- `test_kv_int8_defaults_on`: `AttributeError: 'SousConfig' object has no attribute 'kv_int8'`.
- `test_kv_int8_reads_false_without_an_unknown_key_warning` fails inside the error filter with `UserWarning: sous config: unknown key 'kv_int8' in [model]`.
- The three non-bool cases: `Failed: Regex pattern did not match any of the 1 warnings emitted.`, the one warning being that unknown-key one.

- [ ] **Step 3: Add the config field and its parser**

In `src/sous/config.py`:

Replace:

```python
        "attention_tile",
        "projection_kernel",
    },
}
```

with:

```python
        "attention_tile",
        "projection_kernel",
        "kv_int8",
    },
}
```

Replace:

```python
    # with the reason anywhere else. Read once, at daemon start.
    projection_kernel: bool = True
```

with:

```python
    # with the reason anywhere else. Read once, at daemon start.
    projection_kernel: bool = True
    # Decode's and the exact verifier's attention read an int8 copy of the
    # context's K/V (one symmetric scale per token and KV head) with int8
    # queries on the tensor units' int8 rows, instead of the bf16 cache, from
    # the kv8 floor up: half the bytes the attention tile streams. The
    # attention output moves by 0.73% relative RMS (median over the default
    # model's 16 full-attention layers, 1.76% at worst). The copy exists
    # only inside a generation, so prefill, forks and the fork store stay
    # bf16. It costs about half the bf16 KV of the rows it covers and must
    # fit beside them in the window-sized reserve, so it serves only below
    # a ceiling (about 84.5K keys at the 131072 window, 170K at 262144) and
    # the rest of the generation runs on the
    # tile. Which path a row takes depends only on its own key count, so
    # greedy output with the drafter equals greedy output without it. Not
    # bit-identical to the tile, hence the switch; false runs the tile.
    # Active only where the attention tile is, `unavailable` with the
    # reason anywhere else. Read once, at daemon start.
    kv_int8: bool = True
```

Directly after `_projection_kernel`:

Replace:

```python
    warnings.warn(
        f"sous config: [model].projection_kernel {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

with:

```python
    warnings.warn(
        f"sous config: [model].projection_kernel {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False


def _kv_int8(model: dict) -> bool:
    """[model].kv_int8: true or false; anything else warns and means false,
    the bf16 attention tile, by _int8_prefill's rule that a typo must not
    select changed numerics."""
    value = model.get("kv_int8", True)
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"sous config: [model].kv_int8 {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

In `load_config`:

Replace:

```python
        attention_tile=_attention_tile(model),
        projection_kernel=_projection_kernel(model),
```

with:

```python
        attention_tile=_attention_tile(model),
        projection_kernel=_projection_kernel(model),
        kv_int8=_kv_int8(model),
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`

Expected: 73 passed.

- [ ] **Step 5: Write the failing engine tests**

In `tests/test_engine_base.py`, after `test_default_factory_takes_a_direct_callers_block_as_an_explicit_block`, the factory's and the managed engine's kv8 plumbing:

Replace:

```python
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, draft_block_size=4)
    assert seen["vlm"]["draft_block_explicit"] is True
    assert seen["vlm"]["projection_kernel"] is False
```

with:

```python
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, draft_block_size=4)
    assert seen["vlm"]["draft_block_explicit"] is True
    assert seen["vlm"]["projection_kernel"] is False


# ---- kv_int8 plumbing ------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_engine_manager_threads_kv_int8_into_default_factory(monkeypatch, tmp_path, value):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(_cfg(tmp_path, kv_int8=value)).get()
    assert seen["kwargs"]["kv_int8"] is value


def test_default_factory_passes_kv_int8_and_its_window_to_the_vlm_engine_only(monkeypatch):
    """The window kv8 sizes its ceiling from is the reserve's, in tokens: the
    mirror must fit beside the turn's bf16 KV inside that reserve."""
    from sous.engine import base, lm, vlm

    seen: dict[str, dict] = {}

    class RecordingVLM:
        def __init__(self, model_id, **kwargs):
            seen["vlm"] = kwargs

    class RecordingLM:
        def __init__(self, model_id, **kwargs):
            seen["lm"] = kwargs

    monkeypatch.setattr(vlm, "VLMEngine", RecordingVLM)
    monkeypatch.setattr(lm, "LMEngine", RecordingLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"vision_config": {}})
    monkeypatch.setattr("sous.engine.window.kv_bytes_per_token", lambda cfg: 1024)
    base._default_factory(
        "m", 0.7, 0.8, 20, True, cache_budget=0, reserve_tokens=131072, kv_int8=True
    )
    assert seen["vlm"]["kv_int8"] is True
    assert seen["vlm"]["kv_int8_window"] == 131072
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3_5"})
    base._default_factory(
        "m", 0.7, 0.8, 20, True, cache_budget=0, reserve_tokens=131072, kv_int8=True
    )
    assert "kv_int8" not in seen["lm"]
    assert "kv_int8_window" not in seen["lm"]


def test_default_factory_leaves_kv_int8_off_for_a_direct_caller(monkeypatch):
    """Scripts and tests that call the factory themselves keep the bf16
    attention paths unless they ask, and without a window kv8 has no rows it
    could serve."""
    from sous.engine import base, vlm

    seen: dict[str, dict] = {}

    class RecordingVLM:
        def __init__(self, model_id, **kwargs):
            seen["vlm"] = kwargs

    monkeypatch.setattr(vlm, "VLMEngine", RecordingVLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"vision_config": {}})
    monkeypatch.setattr("sous.engine.window.kv_bytes_per_token", lambda cfg: 1024)
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0)
    assert seen["vlm"]["kv_int8"] is False
    assert seen["vlm"]["kv_int8_window"] == 0


def test_default_factory_passes_no_window_when_the_reserve_cannot_be_measured(monkeypatch):
    """With no bytes-per-token figure the factory reserves nothing, so there
    is no window for the mirror to fit beside: kv8 gets a window of 0 and
    serves nothing, rather than a ceiling sized against a reserve that does
    not exist."""
    import pytest

    from sous.engine import base, vlm

    seen: dict[str, dict] = {}

    class RecordingVLM:
        def __init__(self, model_id, **kwargs):
            seen["vlm"] = kwargs

    monkeypatch.setattr(vlm, "VLMEngine", RecordingVLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"vision_config": {}})
    monkeypatch.setattr("sous.engine.window.kv_bytes_per_token", lambda cfg: None)
    with pytest.warns(UserWarning, match="KV cost per token unknown"):
        base._default_factory(
            "m", 0.7, 0.8, 20, True, cache_budget=0, reserve_tokens=131072, kv_int8=True
        )
    assert seen["vlm"]["kv_int8"] is True
    assert seen["vlm"]["kv_int8_window"] == 0


def test_managed_engine_passes_the_kv_int8_status_through():
    inner = FakeEngine([])
    assert ManagedEngine(inner).kv_int8_status is None, "fakes without the attribute stay silent"
    status = {
        "state": "active",
        "reason": None,
        "probe_seconds": 0.12,
        "floor": 1536,
        "ceiling": 81920,
    }
    inner.kv_int8_status = status  # ty: ignore[unresolved-attribute]
    assert ManagedEngine(inner).kv_int8_status is status
```

After `test_lm_engine_reports_the_projection_kernel_off`, the LM backend's report:

Replace:

```python
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.projection_kernel_status == {"state": "off", "reason": None}
```

with:

```python
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.projection_kernel_status == {"state": "off", "reason": None}


def test_lm_engine_reports_kv_int8_off(monkeypatch):
    from sous.engine.lm import LMEngine

    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (object(), _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.kv_int8_status == {
        "state": "off",
        "reason": None,
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }
```

After `test_vlm_engine_reports_no_draft_block_without_a_drafter`, the VLM engine's enable order and its constructor default:

Replace:

```python
    # The constructor's default keeps the stock projections.
    assert engine.projection_kernel_status["state"] == "off"
```

with:

```python
    # The constructor's default keeps the stock projections.
    assert engine.projection_kernel_status["state"] == "off"


@pytest.mark.parametrize("drafter_loads", [True, False])
def test_vlm_engine_enables_kv8_after_the_projection_kernel_and_before_the_store_and_budget(
    monkeypatch, tmp_path, drafter_loads
):
    from sous.engine import base, draftctx, kv8, projkernel, tileattn, verifyattn, vlm

    model = _positionless_model()
    processor = types.SimpleNamespace(tokenizer=_RecordingTokenizer())
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)

    def load_drafter(m, draft_id):
        if not drafter_loads:
            raise RuntimeError("no such repo")
        return types.SimpleNamespace(prefer_requested_block_size=False), "dflash"

    monkeypatch.setattr(vlm, "_load_quantized_drafter", load_drafter)
    order: list[str] = []
    seen: dict[str, object] = {}
    tile_status = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43}
    kv8_status = {
        "state": "active",
        "reason": None,
        "probe_seconds": 0.12,
        "floor": 1536,
        "ceiling": 81920,
    }

    def verify_enable(m, *, enabled):
        return {"state": "active" if enabled else "off", "reason": None, "probe_seconds": None}

    def proj_enable(m, *, enabled, drafter_kind, tile_status):
        order.append("projection_kernel")
        return {"state": "off", "reason": None, "probe_seconds": None}

    def kv8_enable(m, *, enabled, tile_status, verify_status, window):
        order.append("kv_int8")
        seen.update(
            model=m,
            enabled=enabled,
            tile_status=tile_status,
            verify_status=verify_status,
            window=window,
        )
        return kv8_status

    def context_enable(m, d, kind):
        order.append("draft_context")
        return draftctx.OFF

    class Store:
        def __init__(self, *args):
            order.append("fork_store")

    def budget(reserve_bytes):
        order.append("budget")
        return 0

    monkeypatch.setattr(verifyattn, "enable", verify_enable)
    monkeypatch.setattr(
        tileattn, "enable", lambda m, *, enabled, drafter_kind, verify_status: tile_status
    )
    monkeypatch.setattr(projkernel, "enable", proj_enable)
    monkeypatch.setattr(kv8, "enable", kv8_enable)
    monkeypatch.setattr(draftctx, "enable", context_enable)
    monkeypatch.setattr(vlm, "ForkStore", Store)
    monkeypatch.setattr(vlm.VLMEngine, "_fork_key_fields", lambda self, weights: {})
    monkeypatch.setattr(base, "measure_cache_budget", budget)

    def build() -> vlm.VLMEngine:
        return vlm.VLMEngine(
            "test/model",
            draft_id="z-lab/drafter",
            prompt_cache=True,
            attention_tile=True,
            kv_int8=True,
            kv_int8_window=131072,
            fork_dir=tmp_path / "forks",
        )

    if drafter_loads:
        engine = build()
    else:
        with pytest.warns(UserWarning, match="speculative drafter"):
            engine = build()
    assert order == ["projection_kernel", "kv_int8", "draft_context", "fork_store", "budget"]
    assert seen["model"] is model and seen["enabled"] is True
    assert seen["tile_status"] is engine.attention_tile_status is tile_status
    assert seen["verify_status"] is engine.verify_attention_status
    assert engine.verify_attention_status["state"] == ("active" if drafter_loads else "off")
    assert seen["window"] == 131072
    assert engine.kv_int8_status is kv8_status


def test_vlm_engine_built_directly_leaves_kv8_off(monkeypatch):
    """Tests and scripts that build an engine themselves keep the bf16
    attention paths: the constructor's default refuses before any gate."""
    from sous.engine import kv8
    from sous.engine.vlm import VLMEngine

    _stub(
        monkeypatch,
        "mlx_vlm",
        load=lambda model_id: (_positionless_model(), _RecordingTokenizer()),
    )
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    # Left raised by an imagined earlier load: enable() must clear it.
    monkeypatch.setattr(kv8, "_active", 1)
    engine = VLMEngine("test/model", cache_budget=0)
    assert engine.kv_int8_status == {
        "state": "off",
        "reason": None,
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }
    assert kv8._active == 0
```

In `test_default_engine_factory_maps_every_model_value_onto_the_backend`, the config sets the new value away from its default, so the test proves it is read from the config:

Replace:

```python
        projection_kernel=False,
        speculative_block_explicit=True,
    )
    base.default_engine_factory(cfg)("org/m")
```

with:

```python
        projection_kernel=False,
        kv_int8=False,
        speculative_block_explicit=True,
    )
    base.default_engine_factory(cfg)("org/m")
```

Replace:

```python
        "projection_kernel": False,
        "fork_dir": None,
```

with:

```python
        "projection_kernel": False,
        "kv_int8": False,
        "fork_dir": None,
```

In `tests/test_engine_unloaded.py`, after `test_lm_reset_prompt_cache_works_after_unload`, the LM backend's default speculation:

Replace:

```python
    engine = _unloaded_lm()
    engine.reset_prompt_cache()  # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0
```

with:

```python
    engine = _unloaded_lm()
    engine.reset_prompt_cache()  # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0


def test_lm_speculation_reports_kv8_off():
    """No drafter and no int8 mirror on this backend: the turn line reads
    kv8=off from it, as from a VLM decode that opened no window."""
    from sous.engine.promptcache import Speculation

    spec = _unloaded_lm().speculation()
    assert spec == Speculation()
    assert (spec.kv8, spec.kv8_rows) == ("off", 0)
```

After `test_vlm_unload_turns_the_projection_kernel_off`:

Replace:

```python
    monkeypatch.setattr(projkernel, "_active", 1)
    _unloaded_vlm().unload()
    assert projkernel._active == 0
```

with:

```python
    monkeypatch.setattr(projkernel, "_active", 1)
    _unloaded_vlm().unload()
    assert projkernel._active == 0


def test_vlm_unload_turns_kv8_off(monkeypatch):
    from sous.engine import kv8

    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", kv8.Window(serial=7))
    _unloaded_vlm().unload()
    assert kv8._active == 0
    assert kv8._window is None
```

The test that pins the tile's and the kernel's clears before anything that can raise covers kv8's too, renamed to say so:

Replace:

```python
@pytest.mark.parametrize("failing", ["prompt cache", "draft context"])
def test_vlm_unload_clears_the_tile_and_projection_kernel_before_anything_can_raise(
    monkeypatch, failing
):
    from sous.engine import projkernel, tileattn

    def fails(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(tileattn, "_active_splits", 20)
    monkeypatch.setattr(projkernel, "_active", 1)
```

with:

```python
@pytest.mark.parametrize("failing", ["prompt cache", "draft context"])
def test_vlm_unload_clears_the_tile_projection_kernel_and_kv8_before_anything_can_raise(
    monkeypatch, failing
):
    from sous.engine import kv8, projkernel, tileattn

    def fails(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(tileattn, "_active_splits", 20)
    monkeypatch.setattr(projkernel, "_active", 1)
    monkeypatch.setattr(kv8, "_active", 1)
```

Replace:

```python
    with pytest.raises(RuntimeError, match="failed"):
        engine.unload()
    assert tileattn._active_splits == 0
    assert projkernel._active == 0
```

with:

```python
    with pytest.raises(RuntimeError, match="failed"):
        engine.unload()
    assert tileattn._active_splits == 0
    assert projkernel._active == 0
    assert kv8._active == 0
```

`unload()` must need no attribute beyond the ones `_unloaded_vlm()` sets, which is why it clears the module state and touches no cache: the mirrors live on caches that are dropped with the engine.

- [ ] **Step 6: Run the engine tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py tests/test_engine_unloaded.py -k "kv_int8 or kv8 or maps_every_model_value" -v`

Expected: 15 failed, 167 deselected.
- `KeyError: 'kv_int8'` from the two manager cases and from `test_default_factory_leaves_kv_int8_off_for_a_direct_caller`, and an `AssertionError` on the kwargs dict in `test_default_engine_factory_maps_every_model_value_onto_the_backend`, whose right side has 1 more item;
- `TypeError: _default_factory() got an unexpected keyword argument 'kv_int8'`, from `test_default_factory_passes_kv_int8_and_its_window_to_the_vlm_engine_only` and `test_default_factory_passes_no_window_when_the_reserve_cannot_be_measured`;
- `AttributeError: 'ManagedEngine' object has no attribute 'kv_int8_status'`;
- `AttributeError: 'LMEngine' object has no attribute 'kv_int8_status'`;
- `TypeError: VLMEngine.__init__() got an unexpected keyword argument 'kv_int8'` in both order cases (the no-drafter case also reports `DID NOT WARN`, since the constructor raises first), and `AttributeError: 'VLMEngine' object has no attribute 'kv_int8_status'` in the direct-build test;
- `AttributeError: 'Speculation' object has no attribute 'kv8'`;
- `assert 1 == 0` on `kv8._active` after `unload()`, three times.

- [ ] **Step 7: Add the speculation fields and wire the engines' load and unload**

`decode()` is left for Step 12, after its tests.

In `src/sous/engine/promptcache.py`:

Replace:

```python
@dataclass(frozen=True)
class Speculation:
    """What the drafter did in one decode call: rounds run, draft tokens
    proposed and accepted (the bonus token each round also yields is in
    neither), and the positions of context its first draft saw. The counts
    are zero without a drafter, or when the call never drafted; the context
    is measured only while the engine seeds it, and reads zero otherwise
    whatever the drafter saw."""

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    context: int = 0
```

with:

```python
@dataclass(frozen=True)
class Speculation:
    """What the drafter did in one decode call: rounds run, draft tokens
    proposed and accepted (the bonus token each round also yields is in
    neither), and the positions of context its first draft saw. The counts
    are zero without a drafter, or when the call never drafted; the context
    is measured only while the engine seeds it, and reads zero otherwise
    whatever the drafter saw.

    `kv8` and `kv8_rows` are the int8 K/V mirror's word on the same call,
    drafter or not: the state its serving window closed in (`idle`, `served`
    or `capped`; `off` when no window opened) and the rows it quantised."""

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    context: int = 0
    kv8: str = "off"
    kv8_rows: int = 0
```

In `src/sous/engine/vlm.py`:

Replace:

```python
from sous.engine import draftctx, forkio, projkernel, tileattn
```

with:

```python
from sous.engine import draftctx, forkio, kv8, projkernel, tileattn
```

Replace:

```python
        attention_tile: bool = False,
        projection_kernel: bool = False,
        fork_dir: Path | None = None,
```

with:

```python
        attention_tile: bool = False,
        projection_kernel: bool = False,
        kv_int8: bool = False,
        kv_int8_window: int = 0,
        fork_dir: Path | None = None,
```

Replace:

```python
        _pin_block_size(self._draft, self._draft_block_size)
        # The drafter's context on the prompt-cache path: armed once the
```

with:

```python
        _pin_block_size(self._draft, self._draft_block_size)
        # After the projection kernel, once the tile's and the verifier's
        # states are settled: kv8 serves only where the tile does and rides
        # on its step schedule, its verify plan and, with a drafter, exact
        # verify attention. Before the fork store and the budget, so the
        # probe's mirrors and transient arrays are released when the budget
        # is measured. The window is the served one: the mirror must fit
        # beside the turn's bf16 KV inside the reserve sized from it.
        self.kv_int8_status = kv8.enable(
            self._model,
            enabled=kv_int8,
            tile_status=self.attention_tile_status,
            verify_status=self.verify_attention_status,
            window=kv_int8_window,
        )
        # The drafter's context on the prompt-cache path: armed once the
```

In `unload()`:

Replace:

```python
        # First, before anything below can raise: until the next load decides
        # afresh, no call may reach the tile or the projection kernel on the
        # strength of this model's gates. The kernel's tags stay on this
        # model's modules, inert while its flag is clear.
        tileattn.clear()
        projkernel.clear()
```

with:

```python
        # First, before anything below can raise: until the next load decides
        # afresh, no call may reach the tile, the projection kernel or kv8 on
        # the strength of this model's gates. The kernel's tags stay on this
        # model's modules, inert while its flag is clear; kv8's mirrors live
        # on caches that go with the engine.
        tileattn.clear()
        projkernel.clear()
        kv8.clear()
```

In `src/sous/engine/lm.py`:

Replace:

```python
        self.projection_kernel_status = {"state": "off", "reason": None}
        # Forks on disk, when the factory handed us a directory: built after
```

with:

```python
        self.projection_kernel_status = {"state": "off", "reason": None}
        # kv8 serves only through the tile's hooks on an mlx-vlm qwen3_5
        # load, so mlx-lm's models never reach it; reported off, with the
        # keys kv8.enable's own `off` carries.
        self.kv_int8_status = {
            "state": "off",
            "reason": None,
            "probe_seconds": None,
            "floor": None,
            "ceiling": None,
        }
        # Forks on disk, when the factory handed us a directory: built after
```

Replace:

```python
    def speculation(self) -> Speculation:
        # No drafter on this backend: every decode is plain.
        return Speculation()
```

with:

```python
    def speculation(self) -> Speculation:
        # No drafter and no int8 mirror on this backend: every decode is
        # plain, and the mirror's fields keep their `off` and 0.
        return Speculation()
```

- [ ] **Step 8: Wire the factory and the managed engine**

In `src/sous/engine/base.py`, in `_default_factory`'s signature:

Replace:

```python
    attention_tile: bool = False,
    projection_kernel: bool = False,
    fork_dir: Path | None = None,
    fork_budget: int | None = None,
) -> Engine:
```

with:

```python
    attention_tile: bool = False,
    projection_kernel: bool = False,
    kv_int8: bool = False,
    fork_dir: Path | None = None,
    fork_budget: int | None = None,
) -> Engine:
```

In its VLM branch:

Replace:

```python
            attention_tile=attention_tile,
            projection_kernel=projection_kernel,
            fork_dir=fork_dir,
            fork_budget=fork_budget,
            weights_identity=weights,
        )
    # The drafter settings, the attention tile and the projection kernel stop
    # here: speculative decoding and both kernels' hooks are mlx-vlm paths,
    # and the mlx-lm backend has no parameter for any of them.
```

with:

```python
            attention_tile=attention_tile,
            projection_kernel=projection_kernel,
            kv_int8=kv_int8,
            # In tokens, the window the reserve above is sized from: kv8's
            # mirror must fit beside the turn's bf16 KV inside it. With no
            # reserve measured there is no window to fit, and kv8 serves nothing.
            kv_int8_window=reserve_tokens if bytes_per_token is not None else 0,
            fork_dir=fork_dir,
            fork_budget=fork_budget,
            weights_identity=weights,
        )
    # The drafter settings, the attention tile, the projection kernel and kv8
    # stop here: speculative decoding and the kernels' hooks are mlx-vlm
    # paths, and the mlx-lm backend has no parameter for any of them.
```

In `default_engine_factory`:

Replace:

```python
        projection_kernel=config.projection_kernel,
        fork_dir=fork_dir,
```

with:

```python
        projection_kernel=config.projection_kernel,
        kv_int8=config.kv_int8,
        fork_dir=fork_dir,
```

In `ManagedEngine`, after `projection_kernel_status`:

Replace:

```python
        return getattr(self._inner, "projection_kernel_status", None)
```

with:

```python
        return getattr(self._inner, "projection_kernel_status", None)

    @property
    def kv_int8_status(self) -> dict | None:
        # Optional on purpose, like attention_tile_status: the VLM backend
        # reports whether kv8 serves this load, the LM backend reports it
        # off, and fakes have neither.
        return getattr(self._inner, "kv_int8_status", None)
```

- [ ] **Step 9: Run the config and engine tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py -v`

Expected: 255 passed (73, 168 and 14), with the one warning the files already raise: `drafter context unavailable (drafter names no target layers)` from `test_vlm_engine_probes_verify_attention_only_with_a_drafter[True]`. The existing VLMEngine stub tests raise nothing new, because the constructor's `kv_int8=False` returns `off` before any kv8 gate.

- [ ] **Step 10: Write the decode window tests**

Append to the end of `tests/test_kv8.py`. The section imports function-locally so it appends cleanly whatever the file's header already imports; its helpers carry a `_dw_` prefix so they cannot collide with the file's own.

```python
# ---- the serving window inside VLMEngine.decode() ------------------------------


def _dw_isolate(monkeypatch, *, active: int) -> None:
    """kv8's module state as a fresh load leaves it, restored after the test:
    the window, its serial and the counters a real open_window moves."""
    from sous.engine import kv8

    monkeypatch.setattr(kv8, "_active", active)
    monkeypatch.setattr(kv8, "_window", None)
    monkeypatch.setattr(kv8, "_serial", kv8._serial)
    monkeypatch.setattr(kv8, "calls", dict.fromkeys(kv8.calls, 0))


def _dw_engine():
    """A VLMEngine holding only what decode() reads, built the way
    tests/test_engine_positions.py builds one: no weights, no drafter, and
    positions left to the model so the stub needs no rotary kwargs."""
    import threading
    import types

    import pytest

    pytest.importorskip("mlx.core")
    from sous.engine import draftctx
    from sous.engine.promptcache import PromptMemo, Speculation
    from sous.engine.vlm import VLMEngine

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        config=types.SimpleNamespace(eos_token_id=0)
    )
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()
    engine._positional = False
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._draft_context = draftctx.OFF
    engine._speculation = Speculation()
    return engine


def _dw_result(text: str):
    """One stream_generate result: accepted text, never a draft row."""
    import types

    return types.SimpleNamespace(is_draft=False, text=text, generation_tokens=1, finish_reason=None)


def test_decode_window_spans_the_generation_and_reports_its_state(monkeypatch):
    import sys
    import types

    from sous.engine import kv8

    _dw_isolate(monkeypatch, active=1)
    seen: dict[str, object] = {}

    def stream_generate(model, processor, prompt, **kwargs):
        window = kv8._window
        assert window is not None, "the window is open while the generation runs"
        seen["force"] = window.force
        seen["serving"] = kv8.serving_now()
        # What the seams would record had they served two calls.
        window.state = "served"
        window.rows = 37
        window.served = 2
        yield _dw_result("ok")

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )
    engine = _dw_engine()
    cache = [types.SimpleNamespace(), types.SimpleNamespace()]
    assert engine.decode(cache, [4, 5], 8) == "ok"
    assert seen["serving"] is True
    assert seen["force"] is False
    assert kv8._window is None, "the window closes with the decode"
    assert kv8.calls["windows"] == 1
    spec = engine.speculation()
    assert (spec.kv8, spec.kv8_rows) == ("served", 37)


def test_decode_window_stays_shut_while_kv8_is_inactive(monkeypatch):
    import sys
    import types

    from sous.engine import kv8

    _dw_isolate(monkeypatch, active=0)
    seen: dict[str, object] = {}

    def stream_generate(model, processor, prompt, **kwargs):
        seen["window"] = kv8._window
        yield _dw_result("ok")

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )
    engine = _dw_engine()
    assert engine.decode([types.SimpleNamespace()], [4, 5], 8) == "ok"
    assert seen["window"] is None
    assert kv8.calls["windows"] == 0
    spec = engine.speculation()
    assert (spec.kv8, spec.kv8_rows) == ("off", 0)


def test_decode_window_closes_on_an_exception_and_keeps_the_last_speculation(
    monkeypatch,
):
    import sys
    import types

    import pytest

    from sous.engine import kv8
    from sous.engine.promptcache import Speculation

    _dw_isolate(monkeypatch, active=1)

    def failing(model, processor, prompt, **kwargs):
        assert kv8._window is not None, "the window is open while the generation runs"
        yield _dw_result("partial")
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=failing))
    engine = _dw_engine()
    previous = Speculation(rounds=3, drafted=6, accepted=4, context=9, kv8="served", kv8_rows=11)
    engine._speculation = previous
    cache = [types.SimpleNamespace()]
    with pytest.raises(RuntimeError, match="boom"):
        engine.decode(cache, [4, 5], 8)
    assert kv8._window is None
    assert engine.speculation() is previous, "the retry's decode supplies the next values"

    # Nothing was left open for the next decode's open_window to trip on.
    def fine(model, processor, prompt, **kwargs):
        yield _dw_result("ok")

    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=fine))
    assert engine.decode(cache, [4, 5], 8) == "ok"
    assert kv8.calls["windows"] == 2
    assert engine.speculation().kv8 == "idle"


def test_decode_window_opens_inside_the_drafters_seed_and_closes_before_it_ends(monkeypatch):
    import contextlib
    import sys
    import types

    from sous.engine import kv8

    _dw_isolate(monkeypatch, active=1)
    order: list[object] = []
    cache = [types.SimpleNamespace(), types.SimpleNamespace()]

    @contextlib.contextmanager
    def seed(tail):
        order.append("seed")
        yield
        order.append("unseed")

    def open_window(*, force=False):
        order.append(("open", force))

    def close_window(c):
        order.append(("close", c is cache))
        return ("capped", 5)

    def stream_generate(model, processor, prompt, **kwargs):
        order.append("generate")
        yield _dw_result("ok")

    monkeypatch.setattr(kv8, "open_window", open_window)
    monkeypatch.setattr(kv8, "close_window", close_window)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )
    engine = _dw_engine()
    engine._draft_context = types.SimpleNamespace(seed=seed, fed=lambda: 0)
    assert engine.decode(cache, [4, 5], 8) == "ok"
    assert order == ["seed", ("open", False), "generate", ("close", True), "unseed"]
    spec = engine.speculation()
    assert (spec.kv8, spec.kv8_rows) == ("capped", 5)
```

- [ ] **Step 11: Run the decode window tests to verify they fail**

Run: `uv run pytest tests/test_kv8.py -k decode_window -v`

Expected: 3 failed, 1 passed, the rest of the file deselected.
- `test_decode_window_spans_the_generation_and_reports_its_state`: `AssertionError: the window is open while the generation runs`, raised inside the stub.
- `test_decode_window_closes_on_an_exception_and_keeps_the_last_speculation`: the same `AssertionError` escapes `pytest.raises(RuntimeError)`.
- `test_decode_window_opens_inside_the_drafters_seed_and_closes_before_it_ends`: an `AssertionError` comparing `['seed', 'generate', 'unseed']` with the expected order.
- `test_decode_window_stays_shut_while_kv8_is_inactive` passes already: it guards that an inactive kv8 changes nothing.

- [ ] **Step 12: Open and close the window in `decode()`**

In `src/sous/engine/vlm.py`, in `decode()`:

Replace:

```python
        before = draftctx.rounds_snapshot(self._draft)
        with self._draft_context.seed(draftctx.DraftContext.tail(context)):
            for r in stream_generate(
                model,
                processor,
                "",
                max_tokens=max_tokens,
                verbose=False,
                prompt_cache=cache,
                input_ids=mx.array(token_ids)[None],
                **self._positions(cache, len(token_ids)),
                **sampling,
                **draft_kwargs,
            ):
                # Draft rows are the speculator's proposals, not accepted output;
                # generate() skips them the same way.
                if r.is_draft:
                    continue
                chunks.append(r.text)
                if on_delta is not None:
                    on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
        rounds, accepted, drafted = draftctx.rounds_since(self._draft, before)
        self._speculation = Speculation(
            rounds=rounds, drafted=drafted, accepted=accepted, context=self._draft_context.fed()
        )
        return "".join(chunks)
```

with:

```python
        before = draftctx.rounds_snapshot(self._draft)
        kv8_state, kv8_rows = "off", 0
        with self._draft_context.seed(draftctx.DraftContext.tail(context)):
            # The int8 K/V mirror serves only inside this window, which spans
            # the whole generation, the one-row forward that ends its own
            # prefill segment included (that KV never outlives the turn), and
            # nothing else: every prefill() runs with no window, so the bf16
            # cache that forks, slots and the fork store see never meets a
            # mirror.
            kv8_window = bool(kv8._active)
            if kv8_window:
                kv8.open_window()
            try:
                for r in stream_generate(
                    model,
                    processor,
                    "",
                    max_tokens=max_tokens,
                    verbose=False,
                    prompt_cache=cache,
                    input_ids=mx.array(token_ids)[None],
                    **self._positions(cache, len(token_ids)),
                    **sampling,
                    **draft_kwargs,
                ):
                    # Draft rows are the speculator's proposals, not accepted
                    # output; generate() skips them the same way.
                    if r.is_draft:
                        continue
                    chunks.append(r.text)
                    if on_delta is not None:
                        on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
            finally:
                # Before PrefixCache restores or trims this cache, so no trim
                # wrap is left on a layer when it does, and on this thread, so
                # the mirrors' buffers leave mlx's cache before the publish
                # that follows. An exception leaves _speculation as it was:
                # a cold retry's decode supplies the next values.
                if kv8_window:
                    kv8_state, kv8_rows = kv8.close_window(cache)
        rounds, accepted, drafted = draftctx.rounds_since(self._draft, before)
        self._speculation = Speculation(
            rounds=rounds,
            drafted=drafted,
            accepted=accepted,
            context=self._draft_context.fed(),
            kv8=kv8_state,
            kv8_rows=kv8_rows,
        )
        return "".join(chunks)
```

- [ ] **Step 13: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_kv8.py -k decode_window -v`

Expected: 4 passed, the rest of the file deselected.

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py tests/test_engine_positions.py tests/test_draftctx.py -q`

Expected: every test passes. `test_engine_positions.py` and `test_draftctx.py` build engines with `object.__new__(VLMEngine)` and drive `decode()`; they pass unchanged because `kv8._active` is 0 there, so no window opens and their `Speculation` comparisons see `kv8="off"` and `kv8_rows=0`, the defaults their expected values already carry.

Then run the whole suite, under the GPU lock if another session may be using the GPU:

`uv run pytest -m "not model"`

Expected: every test passes. The skips are the `nax`-marked tests off the M5 and kv8's own real-kernel tests wherever `kv8.availability()` fails.

- [ ] **Step 14: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/config.py src/sous/engine/promptcache.py src/sous/engine/vlm.py src/sous/engine/lm.py src/sous/engine/base.py tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py tests/test_kv8.py
git commit -m "feat(engine): wire kv_int8 from the config into the decode window" -m "[model].kv_int8 (default true) reaches VLMEngine through both factories
with the served window, which sizes kv8's ceiling: the mirror must fit
beside the turn's bf16 KV inside the window-sized reserve. The engine
enables kv8 once the tile's and the verifier's states are settled and
before the fork store and the budget, so the probe's arrays are released
when the budget is measured.

decode() opens the serving window inside the drafter's seed, immediately
before the generation, and closes it in a finally, so the mirror exists
only while a generation runs: every prefill, fork and stored prompt stays
bf16, and no trim wrap is left when the prompt cache restores the cache.
The window's state and rows ride the decode's Speculation for the turn
line. A non-bool value means false, as the other numeric switches do; the
constructors keep the bf16 paths, so tests and scripts that build an
engine directly are unaffected, and an unload clears kv8 beside the tile
and the projection kernel, before anything there can raise."
```

Expected: `git log -1 --format='%G? %an <%ae>%n%B'` shows `G`, the configured identity and no attribution lines.


### Task 6: The mlx-vlm contract tests for the kv8 mirror

**Files:**
- Create: `tests/test_kv8_contract.py` (the whole file). No existing file changes: every seam these tests drive is in place from the earlier tasks.

**Interfaces:**
- Consumes:
  - `src/sous/engine/kv8.py` (the interface sheet's names):
    - state: `N8` (a module int, read at call time, so `kfx.small_floor` can move it), `MIRROR_ATTR = "_sous_kv8"`, `_active`, `_ceiling`, `_window: Window | None`;
    - `calls`, of whose keys these tests read `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_trims`, `windows` and `windows_capped`;
    - `Window` (`serial`, `state` in `idle` / `served` / `capped`, `rows`) and `Mirror` (`codes` int8 `[1, 4, cap, 512]`, `sc` fp16 `[1, 4, cap, 2]`, `cap`, `n`, `serial`);
    - `open_window(*, force: bool = False) -> Window`, `close_window(cache) -> tuple[str, int]`;
    - `serves(n_row) -> bool`, which both seams look up through the module on every call (the round-loop test wraps it);
    - `quantize(k, v, out_rows) -> (codes, sc)`, the real quantiser, which builds on any Metal GPU;
    - `tile8(queries, codes, sc, n, chunk)`, which `decode_codes` and `verify_run` look up through the module on every call (the stand-in and the path recorder replace it).
  - kv8 behaviour these tests pin, as the spec states it and the earlier tasks built it:
    - the trim wrap is an instance attribute named `trim` on a full-attention layer's `KVCache`, set when the mirror is built; after the original trim it sets `n = min(n, cache.offset)` and adds 1 to `calls["mirror_trims"]`, on every call;
    - `prepare(cache, t)` clamps `n` to `cache.offset − t` and quantises rows `[n, min(cache.offset, L))`; a first build quantises from row 0 into a `cap` that is a multiple of 256;
    - a call whose first row is at or past L (`n_keys` for decode, `offset − T + 1` for verify) caps an unforced window, adding 1 to `windows_capped` once per window and dropping that layer's mirror; a call that only straddles L keeps its mirror;
    - `close_window(cache)` removes the mirror attribute and the instance `trim` from every layer of `cache`, clears `_window` and returns the window's state and the rows it quantised (`rows > 0` exactly when it served a call);
    - `decode_kv8` counts each served one-row attention call; `verify_kv8` each verify attention call with at least one served run; `verify_rows_kv8` and `verify_launches_kv8` the served runs' rows and runs; `windows` each `open_window`.
  - The tileattn and verifyattn seams (Task 3): `tileattn._serve_decode`, past its scope checks and N0, returns `kv8.decode(queries, cache, splits)` when `kv8.serves(n_keys)` (leaving `decode_tile` alone), else calls `kv8.observe_first_row(cache, n_keys)` and runs the tile; `tileattn.serve_verify(queries, keys, values, splits, cache)` counts `verify_calls`, `verify_t*` and `verify_straddles` as before, calls `kv8.observe_first_row(cache, prefix + 1)`, calls `kv8.prepare(cache, t)` once when any run is served, and sends each run to stock (`verify_rows_stock`), to `kv8.verify_run` or to the tile (`verify_rows_tile`, `verify_launches`); `verifyattn._wrap` passes `cache`.
  - The engine (Task 5): `VLMEngine.decode()` opens a window with `kv8.open_window()` immediately before `stream_generate` when `kv8._active` is set, closes it in a `finally` with `kv8.close_window(cache)`, and after a loop that did not raise sets `Speculation(..., kv8=state, kv8_rows=rows)`; `promptcache.Speculation.kv8: str = "off"` and `.kv8_rows: int = 0`; `VLMEngine.kv_int8_status`.
  - `tests/kv8_fixtures.py` (Tasks 1–2), imported as `kfx`:
    - `stand_in_tile8(queries, codes, sc, n, chunk)`, with `kv8.tile8`'s contract: queries `[1, 24, T, 256]`, row t of a T-row call sees `n − T + t + 1` keys, the result is the contiguous `[1, T, 24, 256]` bf16, and a row's output depends on its own query, the codes it reads and the chunk, nothing else about the call; any chunk ≥ 1 is accepted (the sensitivity test hands it `ceil(n / 20)`);
    - `small_floor(monkeypatch)`: `kv8.N8` is `tileattn.N0` for the test;
    - `ceiling(monkeypatch, L)`: `kv8._ceiling` is L for the test;
    - `nax`: the skip marker on `kv8.availability()`.
  - tile_fixtures (existing): `VOCAB`, `tiny_language_model()`, `prefill_cache(lm, ids)`, `verify_forward(lm, cache, tokens)` (one verify forward, then the round aborted), `hooked(monkeypatch, *, splits, stand_in)` (the decode hook, the flag and `stand_in_tile`), `verify_ready(monkeypatch, lm)` (verifyattn's wrapper installed and the attention modules tagged).
  - Existing: `tileattn.N0`, `boundaries`, `verify_plan`, `calls` and `tile`; `draftctx.enable`; `promptcache.PromptMemo`.
  - mlx-vlm 0.7.2: `speculative.dflash._dflash_rounds` and the module global `_speculative_walk` it calls each round; `cache_state._trim_append_caches` (a partial commit) and `SpeculativeCacheTransaction.abort` (a round that raised), which both reach a KVCache's `trim` through an attribute lookup on the instance.
- Produces: tests only.

How the file drives mlx-vlm follows `tests/test_draftctx.py`: read its `_Drafter` (a drafter reduced to what the round loop touches), `_engine` (a `VLMEngine` built with `object.__new__`, every attribute `prefill()` and `decode()` read set by hand) and `_stub_mlx_vlm` (`mlx_vlm.generate` and `mlx_vlm.stream_generate` replaced on the real module). Here the drafter proposes a precomputed greedy continuation, and the stubs run the tiny model's real forwards.

Measured while planning (mlx 0.32.3, mlx-vlm 0.7.2, the M2; the tiny model with test_tileattn_contract's ids, `mx.random.key(132)`):
- The tiny model's full-attention layers are 1 and 3 (`full_attention_interval=2`); 0 and 2 are gated-delta layers. The exact verifier captures `target_layer_ids [0, 2]`.
- With the bf16 tile's stand-in serving every row, the real `_dflash_rounds` drives the scripted drafter exactly as the straddling test's comment says: from a 2,542-token prefix at block 5, max_tokens 32 and wrong drafts at round 3 position 1 and round 4 position 2, the walk records `(drafted, accepted)` = `(4, 4) (4, 4) (4, 4) (4, 1) (4, 2) (4, 4) (4, 4) (1, 1)`, commits `cont[1:32]`, leaves the offset at 2,574, and verifies exactly key counts 2,543–2,574, the ones the drafter-off decode computes. With wrong drafts at position 1 of rounds 0, 2 and 4 and the walk raising in round 6, every wrong draft was rejected and the abort left the offset at the committed prefix. Each test runs in 1–3 s.
- Against a throwaway mock of kv8 that follows the spec (an mlx-op quantiser, the trim wrap, the window, the two seams and the engine's window), the file passed with 15 tests run and 9 skipped in 10 s, and each injected fault broke exactly the tests aimed at it: a trim wrap that does not lower `n` (the hooks test, and the round-loop test through its bulk comparison even when only the partial commits' trims were ignored); a verify that caps on its last row instead of its first (the `(2558, 5)` parity case and the straddling test); an engine `decode()` without its `finally` (both engine tests).

- [ ] **Step 1: Write the contract test**

Create `tests/test_kv8_contract.py`:

```python
"""The int8 K/V mirror (kv8) through mlx-vlm's real qwen3_5 forward, exact
verifier and DFlash round loop, and through the engine's own prefill and decode.

A tiny random qwen3_5 language model at the 27B's attention shape (two
full-attention layers, from tile_fixtures) is prefilled past kv8's floor through
the real forward, with the attention tile's decode hook and verifyattn's wrapper
installed at S = 20, the floor N8 at N0 and the ceiling L at the step boundary
2,561. kv8.tile8 and tileattn.tile are the plain-mlx stand-ins on any GPU (the
real kernels under ``nax``); the quantiser is always the real one, which builds
on any Metal GPU. Every comparison with a one-row decode rests on stock's own
exactness on these ids, which test_tileattn_contract asserts.

Pinned here: both hooks reach kv8 only inside a window; every verify row equals
the one-row decode at its own key count, on the path the per-row rule gives it;
through mlx-vlm's own round loop the mirror follows partial commits and an abort
through the cache's instance trim and ends equal to quantising the committed
rows in one go; a round that straddles L keeps every row on its drafter-off path
and caps the window once; and the engine's prefill never builds a mirror while
its decode does, and closes its window whatever the generation raises.
"""

import threading
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import draftctx, kv8, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.promptcache import PromptMemo, Speculation  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402
from tests import kv8_fixtures as kfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402

SPLITS = 20
LAYERS = 2  # the tiny model's full-attention layers: every forward makes two attention calls
FULL = (1, 3)  # their indices among the model's layers, and in its cache
# L for every test here: the step boundary after 1,921 at S = 20. Far below the
# ceiling of any real window, so the straddling cases stay a few rows long.
CEILING = 2561
BLOCK = 5  # the verify block the round-loop tests run, as the M5 Pro does
PROMPT = 1600  # the engine tests' prompt: its last row lies past N8 and below CEILING
# The kv8 counters a window and its forwards move; the mirror's own counters
# (builds, rows, trims, growths) are read only where a test is about them.
COUNTED = (
    "decode_kv8",
    "verify_kv8",
    "verify_rows_kv8",
    "verify_launches_kv8",
    "windows",
    "windows_capped",
)
KV8_CASES = [
    (1000, 3),  # below N0: stock rows, and a window that serves nothing
    (1533, 5),  # two stock rows, then three kv8 rows from N8 (N0 here)
    (1700, 1),
    (1700, 2),  # the DFlash round loop's last round
    (1700, 3),
    (1700, 8),
    (1717, 5),  # where a kernel that ignores the schedule breaks parity (below)
    (1918, 5),  # straddles the step boundary at 1,921 keys: two kv8 runs
    (2558, 5),  # straddles L: two kv8 rows, then three tile rows; decode caps at 2,561
]


@pytest.fixture(scope="module")
def tiny():
    lm = tfx.tiny_language_model()
    # test_tileattn_contract's ids, whose stock exactness it asserts.
    ids = mx.random.randint(0, tfx.VOCAB, (2600,), key=mx.random.key(132)).tolist()
    return lm, ids


def _serving(monkeypatch: pytest.MonkeyPatch, *, stand_in: bool = True) -> None:
    """kv8 as an active load leaves it, behind the tile's decode hook at S = 20:
    the flag raised, the small floor, CEILING as L, no window open, the stand-ins
    for both kernels (or the real ones) and the real quantiser. monkeypatch
    restores all of it, a window a failed test left open included."""
    assert CEILING in tileattn.boundaries(CEILING, SPLITS)
    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=stand_in)
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, CEILING)
    if stand_in:
        monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)


def _prefill(lm, ids):
    """A fresh cache holding `ids`, prefilled in one call outside any window, and
    the greedy token after them."""
    cache = lm.make_cache()
    logits = lm(
        mx.array([ids], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(len(ids), dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    ).logits
    mx.eval([c.state for c in cache])
    return cache, int(mx.argmax(logits[0, -1]).item())


def _step(lm, cache, token: int):
    """One one-row decode forward: the logits row after `token`, [1, vocab]."""
    logits = lm(mx.array([[token]], dtype=mx.int32), cache=cache).logits
    mx.eval(logits)
    return logits[:, -1]


def _in_window(cache, run):
    """`run()` inside a window over `cache`, opened and closed as decode() does;
    returns run's result and close_window's (state, rows)."""
    kv8.open_window()
    try:
        out = run()
    finally:
        closed = kv8.close_window(cache)
    return out, closed


def _verify_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    logits, closed = _in_window(
        cache, lambda: tfx.verify_forward(lm, cache, ids[prefix : prefix + t])[0]
    )
    return [logits[:, r] for r in range(t)], closed


def _decode_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    return _in_window(cache, lambda: [_step(lm, cache, tok) for tok in ids[prefix : prefix + t]])


def _greedy_off(lm, ids, prefix, count):
    """Drafter-off greedy decoding of `count` tokens after ids[:prefix]: the first
    from the prefill, each later one from a one-row forward inside one window, as
    a decode without a drafter runs. Returns the tokens and close_window's
    (state, rows)."""
    cache, first = _prefill(lm, ids[:prefix])
    tokens = [first]

    def run():
        for _ in range(count - 1):
            tokens.append(int(mx.argmax(_step(lm, cache, tokens[-1])).item()))

    _, closed = _in_window(cache, run)
    return tokens, closed


def _same(a, b):
    return [mx.array_equal(x, y).item() for x, y in zip(a, b, strict=True)]


def _delta(before, after):
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def _counted(before, after):
    return {k: v for k, v in _delta(before, after).items() if k in COUNTED}


def _nonzero(counts):
    return {k: v for k, v in counts.items() if v}


def _path(n_row: int) -> str:
    """The path the per-row rule gives a row that sees n_row keys."""
    if n_row < tileattn.N0:
        return "stock"
    if kv8.N8 <= n_row < CEILING:
        return "kv8"
    return "tile"


def _state(capped: bool, served: bool) -> str:
    """close_window's state for a window that capped or served (or neither)."""
    if capped:
        return "capped"
    return "served" if served else "idle"


def _verify_counts(prefix, t):
    """(tileattn, kv8) counter deltas for one T-row verify forward in a window of
    its own: the tile's per-call counts as before, each run's rows to the path of
    its first row (every row of a run shares it: N8 and L are step boundaries,
    and verify_plan cuts at every one)."""
    runs = [(k - j, _path(prefix + j + 1)) for j, k, _ in tileattn.verify_plan(prefix, t, SPLITS)]
    rows = {p: sum(n for n, q in runs if q == p) for p in ("stock", "tile", "kv8")}
    launches = {p: sum(1 for _, q in runs if q == p) for p in ("tile", "kv8")}
    tile = {
        "verify_calls": LAYERS,
        f"verify_t{t}": LAYERS,
        "verify_rows_stock": LAYERS * rows["stock"],
        "verify_rows_tile": LAYERS * rows["tile"],
        "verify_launches": LAYERS * launches["tile"],
        "verify_straddles": LAYERS * (len(runs) > 1),
    }
    eight = {
        "verify_kv8": LAYERS * (launches["kv8"] > 0),
        "verify_rows_kv8": LAYERS * rows["kv8"],
        "verify_launches_kv8": LAYERS * launches["kv8"],
        "windows": 1,
        "windows_capped": int(prefix + 1 >= CEILING),
    }
    return _nonzero(tile), _nonzero(eight)


def _decode_counts(prefix, t):
    """(tileattn, kv8) counter deltas for t one-row forwards in a window of their own."""
    paths = [_path(prefix + r + 1) for r in range(t)]
    tile = {
        "decode_stock": LAYERS * paths.count("stock"),
        "decode_tile": LAYERS * paths.count("tile"),
    }
    eight = {
        "decode_kv8": LAYERS * paths.count("kv8"),
        "windows": 1,
        "windows_capped": int(prefix + t >= CEILING),
    }
    return _nonzero(tile), _nonzero(eight)


def _assert_mirror_is_bulk(cache) -> None:
    """Every full-attention layer's mirror covers exactly the cache's committed
    rows and holds what quantising them in one go gives: no row left from a
    rejected draft, none missing."""
    for i in FULL:
        layer = cache[i]
        mirror = getattr(layer, kv8.MIRROR_ATTR)
        n = mirror.n
        assert n == layer.offset, (i, n, layer.offset)
        codes, sc = kv8.quantize(layer.keys[:, :, :n, :], layer.values[:, :, :n, :], n)
        assert mx.array_equal(mirror.codes[:, :, :n], codes).item(), i
        assert mx.array_equal(mirror.sc[:, :, :n], sc).item(), i


def _dropped(cache) -> bool:
    """What close_window leaves on every full-attention layer: no mirror, and the
    class's own trim again."""
    return all(
        getattr(cache[i], kv8.MIRROR_ATTR, None) is None and "trim" not in vars(cache[i])
        for i in FULL
    )


# ---- the hooks ----------------------------------------------------------------


def test_both_hooks_reach_kv8_through_the_real_forward_with_a_window_open(tiny, monkeypatch):
    """Inside a window, verify rows reach kv8 through verifyattn's wrapper and a
    one-row decode through the tile's decode hook. The abort that ends the
    verify forward trims the mirror through its wrap, and the decode step after
    it, with a token the verify did not write, re-quantises the row it replaced.
    Without a window the same forwards are the tile's and quantise nothing,
    though kv8's flag is up."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    prefix, t = 1700, 5
    other = (ids[prefix] + 1) % tfx.VOCAB

    cache = tfx.prefill_cache(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    window = kv8.open_window()
    try:
        tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
        tile_verified, kv8_verified = dict(tileattn.calls), dict(kv8.calls)
        for i in FULL:
            mirror = getattr(cache[i], kv8.MIRROR_ATTR)
            assert (mirror.serial, mirror.n) == (window.serial, prefix)
            assert mirror.cap % 256 == 0 and mirror.cap >= prefix + t
        _step(lm, cache, other)
        _assert_mirror_is_bulk(cache)
        assert window.state == "served"
    finally:
        state, rows = kv8.close_window(cache)
    assert state == "served" and rows >= prefix
    assert kv8._window is None and _dropped(cache)
    assert _delta(tile_start, tile_verified) == {"verify_calls": LAYERS, "verify_t5": LAYERS}
    assert _counted(kv8_start, kv8_verified) == {
        "windows": 1,
        "verify_kv8": LAYERS,
        "verify_rows_kv8": LAYERS * t,
        "verify_launches_kv8": LAYERS,
    }
    assert _delta(tile_verified, tileattn.calls) == {}
    assert _counted(kv8_verified, kv8.calls) == {"decode_kv8": LAYERS}

    cache = tfx.prefill_cache(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
    _step(lm, cache, other)
    assert kv8.calls == kv8_start
    assert _delta(tile_start, tileattn.calls) == {
        "verify_calls": LAYERS,
        "verify_t5": LAYERS,
        "verify_rows_tile": LAYERS * t,
        "verify_launches": LAYERS,
        "decode_tile": LAYERS,
    }
    assert all(getattr(cache[i], kv8.MIRROR_ATTR, None) is None for i in FULL)


@pytest.mark.slow
@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=kfx.nax)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize(("prefix", "t"), KV8_CASES)
def test_kv8_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, prefix, t, stand_in, monkeypatch
):
    """Every verify row equals the one-row decode at its own key count, on the
    path the per-row rule gives it: stock below N0, kv8 from N8 below L, the tile
    from L. A verify that straddles L serves its rows below L, keeps its mirror
    and leaves its window uncapped; the decode caps its window at the first row
    at L."""
    lm, ids = tiny
    _serving(monkeypatch, stand_in=stand_in)
    tfx.verify_ready(monkeypatch, lm)
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    verify, verify_closed = _verify_rows(lm, ids, prefix, t)
    tile_verified, kv8_verified = dict(tileattn.calls), dict(kv8.calls)
    decode, decode_closed = _decode_rows(lm, ids, prefix, t)
    assert all(_same(verify, decode)), (prefix, t)

    tile_want, kv8_want = _verify_counts(prefix, t)
    assert _delta(tile_start, tile_verified) == tile_want
    assert _counted(kv8_start, kv8_verified) == kv8_want
    served = "verify_rows_kv8" in kv8_want
    assert verify_closed[0] == _state(prefix + 1 >= CEILING, served)
    assert (verify_closed[1] > 0) == served

    tile_want, kv8_want = _decode_counts(prefix, t)
    assert _delta(tile_verified, tileattn.calls) == tile_want
    assert _counted(kv8_verified, kv8.calls) == kv8_want
    served = "decode_kv8" in kv8_want
    assert decode_closed[0] == _state(prefix + t >= CEILING, served)
    assert (decode_closed[1] > 0) == served


def test_a_kv8_kernel_that_ignores_the_schedule_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: rows 1,718..1,722 share chunk 96, but a kernel that
    sizes its splits from the call's own key count (ceil(n / 20)) gives the
    verify call and the one-row call at 1,718 different partitions. The equality
    above is therefore not vacuous: the rows it compares are kv8's."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    monkeypatch.setattr(
        kv8,
        "tile8",
        lambda q, codes, sc, n, chunk: kfx.stand_in_tile8(q, codes, sc, n, -(-n // 20)),
    )
    verify, _ = _verify_rows(lm, ids, 1717, 5)
    decode, _ = _decode_rows(lm, ids, 1717, 5)
    assert not all(_same(verify, decode))


# ---- mlx-vlm's DFlash round loop ----------------------------------------------


class _Abort(Exception):
    """The walk failing part-way through a round."""


class _Scripted:
    """A DFlash drafter reduced to what the round loop touches, as
    test_draftctx's stub is, proposing the next block_size - 1 tokens of a
    precomputed greedy continuation, with a wrong token at the draft position
    `wrong` names for a round. Where it stands in the continuation is read off
    the acceptances the round loop records on it (each round commits its
    accepted drafts and a bonus), so after a rejection it proposes from the
    bonus the target chose."""

    prefer_requested_block_size = True

    def __init__(self, continuation: list[int], wrong: dict[int, int] | None = None):
        self.config = types.SimpleNamespace(target_layer_ids=[0, 2])
        self.continuation = continuation
        self.wrong = dict(wrong or {})
        self.round = -1
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def reset(self, target_model):
        return ["draft cache"]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        self.round += 1
        at = sum(int(a) + 1 for a in self.accept_lens)
        tokens = list(self.continuation[at + 1 : at + block_size])
        if self.round in self.wrong:
            q = self.wrong[self.round]
            tokens[q] = (tokens[q] + 1) % tfx.VOCAB
        return mx.array([tokens], dtype=token_dtype)

    draft_block_greedy = draft_block


def _walk(monkeypatch: pytest.MonkeyPatch, drafter: _Scripted, *, abort_round=None) -> list:
    """The round loop's own greedy walk, recording (round, drafted, accepted)
    for each round it finishes; in `abort_round` it raises _Abort instead, after
    the verify forward appended the round's rows and before the commit, so the
    loop aborts the round through SpeculativeCacheTransaction.abort."""
    from mlx_vlm.speculative import dflash

    real = dflash._speculative_walk
    record: list[tuple[int, int, int]] = []

    def walk(draft_tokens, target_tokens, budget):
        if drafter.round == abort_round:
            raise _Abort(f"round {drafter.round}")
        accepted, new_tokens = real(draft_tokens, target_tokens, budget)
        record.append((drafter.round, int(draft_tokens.shape[1]), accepted))
        return accepted, new_tokens

    monkeypatch.setattr(dflash, "_speculative_walk", walk)
    return record


def _rounds(lm, drafter: _Scripted, cache, first: int, max_tokens: int):
    """mlx-vlm's DFlash round loop over `cache`, greedy at block BLOCK, as
    generate_step hands it over after the prefill and the first bonus."""
    from mlx_vlm.speculative import dflash

    return dflash._dflash_rounds(
        types.SimpleNamespace(language_model=lm),  # ty: ignore[invalid-argument-type]
        drafter,  # ty: ignore[invalid-argument-type]
        cache,
        mx.zeros((1, 1, 2 * 256), dtype=mx.bfloat16),  # the prompt hidden: the stub never reads it
        first_bonus=first,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=BLOCK,
        greedy_sampling=True,
    )


class _Paths:
    """Which kernel served each key count, read off the two array-level calls
    every served row goes through: kv8.tile8 ("tile8") and tileattn.tile
    ("tile"). Row t of a T-row call sees n - T + t + 1 keys."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.seen: dict[int, set[str]] = {}
        for module, name in ((kv8, "tile8"), (tileattn, "tile")):
            monkeypatch.setattr(module, name, self._recording(name, getattr(module, name)))

    def _recording(self, kind, kernel):
        def run(queries, a, b, n, chunk):
            out = kernel(queries, a, b, n, chunk)
            for key_count in range(n - out.shape[1] + 1, n + 1):
                self.seen.setdefault(key_count, set()).add(kind)
            return out

        return run

    def take(self) -> dict[int, set[str]]:
        seen, self.seen = self.seen, {}
        return seen


@pytest.mark.slow
def test_the_mirror_follows_partial_commits_and_an_abort_through_the_real_round_loop(
    tiny, monkeypatch
):
    """mlx-vlm's own round loop, with kv8.serves wrapped to serve every other
    round only, as a per-round on/off harness wraps it. A served round's wrong
    draft is rejected and its rows trimmed off the cache at the commit
    (_trim_append_caches); the next round runs on the tile and writes those
    cache rows again with tokens the mirror never saw; the served round after
    it must quantise them anew. The
    clamp a served call runs first cannot see that, since by then the committed
    prefix has moved past the rejected rows: only the instance trim the commit
    reached can. The walk then raises in a served round, after its verify
    appended and quantised five rows, and the loop aborts the round through
    SpeculativeCacheTransaction.abort, which reaches the same trim. After all of
    it the mirror holds exactly what quantising the committed rows in one go
    gives."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    prefix = 1700
    cont, _ = _greedy_off(lm, ids, prefix, 40)
    drafter = _Scripted(cont, wrong={0: 1, 2: 1, 4: 1})
    serves = kv8.serves
    monkeypatch.setattr(kv8, "serves", lambda n_row: drafter.round % 2 == 0 and serves(n_row))
    record = _walk(monkeypatch, drafter, abort_round=6)
    cache, first = _prefill(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    kv8.open_window()
    try:
        with pytest.raises(_Abort):
            for _ in _rounds(lm, drafter, cache, first, max_tokens=64):
                pass
        assert [r for r, _, _ in record] == [0, 1, 2, 3, 4, 5]
        # Each served round was cut short by its wrong draft and followed by an
        # off round: the sequence the trim wrap alone keeps right.
        assert all(accepted < drafted for r, drafted, accepted in record if r % 2 == 0)
        assert cache[FULL[0]].offset == prefix + sum(a + 1 for _, _, a in record)
        _assert_mirror_is_bulk(cache)
    finally:
        kv8.close_window(cache)
    partial = sum(1 for _, drafted, accepted in record if accepted < drafted)
    moved = _delta(kv8_start, kv8.calls)
    assert moved.get("verify_kv8", 0) == LAYERS * 4  # rounds 0, 2, 4 and the aborted 6
    assert moved.get("verify_rows_kv8", 0) == LAYERS * BLOCK * 4
    # The wrap counts every trim it passes on: each partial commit, then the abort.
    assert moved.get("mirror_trims", 0) == LAYERS * (partial + 1)
    assert "decode_kv8" not in moved and "windows_capped" not in moved
    assert _delta(tile_start, tileattn.calls).get("verify_rows_tile", 0) == LAYERS * BLOCK * 3


@pytest.mark.slow
def test_a_round_that_straddles_the_ceiling_keeps_every_row_on_its_drafter_off_path(
    tiny, monkeypatch
):
    """L falls inside a verify round, and that round and the next reject drafts.
    Drafter off, one-row decode serves rows below L from the mirror and caps the
    window at the first row at L. Drafter on, the round loop commits the same
    tokens, every row it computes takes the path drafter off gave the row at
    that key count, and the window caps exactly once: at the first round whose
    first row is at or past L, never at a round that only straddles it."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    paths = _Paths(monkeypatch)
    prefix, tokens = 2542, 32
    capped = kv8.calls["windows_capped"]

    cont, closed = _greedy_off(lm, ids, prefix, tokens + 1)
    off = paths.take()
    assert off == {
        n: {"tile8" if n < CEILING else "tile"} for n in range(prefix + 1, prefix + tokens + 1)
    }
    assert closed[0] == "capped" and kv8.calls["windows_capped"] == capped + 1

    # Rounds 0-2 accept every draft (rows 2,543-2,557); round 3's rows are
    # 2,558-2,562 and its second draft is wrong, so it commits 2,558-2,559;
    # round 4's rows are 2,560-2,564 and its third draft is wrong, so it commits
    # 2,560-2,562; round 5 starts at 2,563 and caps; round 7 has one draft left.
    drafter = _Scripted(cont, wrong={3: 1, 4: 2})
    record = _walk(monkeypatch, drafter)
    cache, first = _prefill(lm, ids[:prefix])
    assert first == cont[0]
    kv8.open_window()
    try:
        produced = [token for token, _ in _rounds(lm, drafter, cache, first, max_tokens=tokens)]
    finally:
        closed = kv8.close_window(cache)
    on = paths.take()
    assert produced == cont[1:tokens]
    assert [(drafted, accepted) for _, drafted, accepted in record] == [
        (4, 4),
        (4, 4),
        (4, 4),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 4),
        (1, 1),
    ]
    assert set(on) == set(range(prefix + 1, prefix + tokens + 1))
    assert on == off
    assert closed[0] == "capped" and kv8.calls["windows_capped"] == capped + 2


# ---- the engine's prefill and decode -------------------------------------------


class _Boom(Exception):
    """A generation failing after its own prefill segment was served."""


class _Wrapper:
    """The mlx-vlm model the engine holds: the tiny language model, and the
    config decode() reads the end-of-sequence id from."""

    def __init__(self, lm):
        self.language_model = lm
        self.config = types.SimpleNamespace(model_type="qwen3_5", eos_token_id=0)


def _engine(lm) -> VLMEngine:
    """A VLMEngine with no load behind it, as test_draftctx builds one: every
    attribute prefill() and decode() read, set by hand. No drafter, so the draft
    context is off and decode() runs no round loop."""
    model = _Wrapper(lm)
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/tiny-qwen3_5"
    engine._model = model  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = None
    # The engine's own positions, as the qwen3_5 lineage gets them: the bare
    # language model cannot derive rope positions without a vision config.
    engine._positional = True
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._draft_context = draftctx.enable(model, None, "")
    engine._speculation = Speculation()
    engine.kv_int8_status = {
        "state": "active",
        "reason": None,
        "probe_seconds": None,
        "floor": tileattn.N0,
        "ceiling": CEILING,
    }
    return engine


def _stub_mlx_vlm(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
    """generate() and stream_generate() reduced to the tiny model's real
    forwards, as generate_step makes them: the prompt but its last row in one
    chunk, then the one-row step that ends every prefill; stream_generate then
    yields greedy tokens, one one-row forward between each. With `fail`,
    stream_generate raises right after that last prefill step."""
    import mlx_vlm

    def forward_prompt(m, kwargs):
        lm, cache, ids = m.language_model, kwargs["prompt_cache"], kwargs["input_ids"]
        positions = {k: kwargs[k] for k in ("position_ids", "rope_deltas") if k in kwargs}
        if ids.shape[1] > 1:
            lm(ids[:, :-1], cache=cache, **positions)
        logits = lm(ids[:, -1:], cache=cache, **positions).logits
        mx.eval(logits, [c.state for c in cache])
        return logits

    def generate(m, processor, prompt, **kwargs):
        assert kwargs["max_tokens"] == 0
        forward_prompt(m, kwargs)

    def stream_generate(m, processor, prompt, **kwargs):
        logits = forward_prompt(m, kwargs)
        if fail:
            raise _Boom("after the prefill segment")
        count = kwargs["max_tokens"]
        for i in range(count):
            token = int(mx.argmax(logits[0, -1]).item())
            yield types.SimpleNamespace(
                is_draft=False, text=f"{token} ", generation_tokens=i + 1, finish_reason=None
            )
            if i + 1 < count:
                cache = kwargs["prompt_cache"]
                logits = m.language_model(mx.array([[token]], dtype=mx.int32), cache=cache).logits

    # On the real module, as test_draftctx stubs it: decode() imports more of
    # mlx_vlm than a stand-in package would carry.
    monkeypatch.setattr(mlx_vlm, "generate", generate)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream_generate)


def test_a_prefill_builds_no_mirror_and_the_cold_paths_decode_builds_one(tiny, monkeypatch):
    """prefill() runs with no window, so even the one-row step that ends it past
    N8 is the tile's. decode() opens one around the generation: on the cold path
    the one-row step that ends its own prefill segment builds the mirror over
    the whole prompt and every token after it reads it, and its finally drops
    the mirror and reports what the window served."""
    lm, ids = tiny
    _serving(monkeypatch)
    _stub_mlx_vlm(monkeypatch)
    engine = _engine(lm)

    cache = lm.make_cache()
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    assert engine.prefill(cache, ids[:PROMPT]) is None
    assert cache[FULL[0]].offset == PROMPT
    assert kv8.calls == kv8_start
    assert _delta(tile_start, tileattn.calls) == {"decode_tile": LAYERS}
    assert all(getattr(cache[i], kv8.MIRROR_ATTR, None) is None for i in FULL)

    cache = lm.make_cache()
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    engine.decode(cache, ids[:PROMPT], 3)
    moved = _delta(kv8_start, kv8.calls)
    # The prefill segment's last step at 1,600 keys, then 1,601 and 1,602.
    assert _counted(kv8_start, kv8.calls) == {"windows": 1, "decode_kv8": 3 * LAYERS}
    assert moved.get("mirror_builds", 0) > 0
    assert _delta(tile_start, tileattn.calls) == {}
    spec = engine.speculation()
    assert spec.kv8 == "served" and spec.kv8_rows >= PROMPT
    assert kv8._window is None and _dropped(cache)


def test_an_exception_out_of_the_generation_closes_the_window(tiny, monkeypatch):
    """A generation that raises after kv8 built its mirror still leaves no
    window, no mirror and no trim wrap behind, and the turn's speculation as it
    was; the next decode opens a window of its own."""
    lm, ids = tiny
    _serving(monkeypatch)
    _stub_mlx_vlm(monkeypatch, fail=True)
    engine = _engine(lm)
    cache = lm.make_cache()
    builds = kv8.calls["mirror_builds"]
    with pytest.raises(_Boom):
        engine.decode(cache, ids[:PROMPT], 3)
    assert kv8.calls["mirror_builds"] > builds
    assert kv8._window is None and _dropped(cache)
    assert engine.speculation() == Speculation()

    _stub_mlx_vlm(monkeypatch)
    engine.decode(lm.make_cache(), ids[:PROMPT], 2)
    assert engine.speculation().kv8 == "served"
```

- [ ] **Step 2: Run the contract test**

Run: `uv run pytest tests/test_kv8_contract.py -v -rs`

Expected on the M2 and on CI's macos-15 (no tensor units, so `kfx.nax` skips): `15 passed, 9 skipped`. The skips are the nine `kernel` variants of `test_kv8_keeps_verify_rows_equal_to_decode_through_the_real_forward`, each with the reason `kfx.nax` gives. On the M5 Pro: `24 passed`.
- These tests pin behaviour the earlier tasks built, so they pass as written. A failed assertion is a bug in those tasks: fix it there and never loosen the assertion.
- If `_assert_mirror_is_bulk` fails on `n == layer.offset` in the round-loop test, the trim wrap is not where `_trim_append_caches` and `SpeculativeCacheTransaction.abort` look (`getattr(cache, "trim")` on the instance). If it fails on the codes with `n` right, the wrap is reached on some trims but not others: rows a rejected draft wrote are still in the mirror.
- If the straddling test's `record` differs from the scripted list, a round accepted fewer drafts than scripted, which means a verify row differed from the drafter-off decode at the same key count: the parity grid pins the same property at fixed prefixes and names the case.
- If `all(accepted < drafted ...)` fails in the round-loop test, a wrong draft was accepted: the target's own token coincided with the injected one, which this schedule never showed. Report it with the round rather than drop the assertion.
- The parity grid and the two round-loop tests carry `slow`, as the tile's own contract grid does; `-m "not model"` runs them.

- [ ] **Step 3: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add tests/test_kv8_contract.py
git commit -m "test(engine): pin the int8 K/V mirror through mlx-vlm's real paths" -m "The unit tests prove the quantiser, the mirror rule, the window and the
seams in isolation. These pin them through mlx-vlm's own code on a tiny
qwen3_5 model at the 27B's attention shape. Both hooks reach kv8 only
inside a window, and every verify row equals the one-row decode at its own
key count on the path the per-row rule gives it. Through the real DFlash
round loop, with kv8 serving every other round, the mirror follows partial
commits and an abort through the cache's instance trim and ends equal to
quantising the committed rows in one go; the clamp a served call runs first
cannot catch the rows a rejected draft left, so this is the trim wrap's
behavioural guard beside the source pins. A round that straddles the
ceiling keeps every row on its drafter-off path and caps the window once,
and the engine's prefill never builds a mirror while its decode does and
closes its window whatever the generation raises."
```

Expected: ruff and ty report nothing, and the full non-model suite passes.

---


### Task 7: Load line, status document, turn-line gauges and the fork-epoch deny-list

**Files:**
- Modify: `src/sous/engine/base.py`:
  - the load line in `EngineManager.get()`, after the `draft_block` token (:752-755);
  - `EngineManager.status()`, after the `draft_block` key (:1079-1080).
- Modify: `src/sous/engine/promptcache.py`:
  - `TURN_GAUGES` (:311-314);
  - `PromptCacheStats`, after `draft_context` (:401-404);
  - `_note_speculation` (:1490-1497).
- Modify: `src/sous/api/turn.py`: `TurnResult`'s last fields (:107-112) and its construction in `TurnRunner.run` (:346-347).
- Modify: `src/sous/api/routes.py`: `_turn_summary` (:245-246) and `_turn_line` (:273-274).
- Modify: `src/sous/engine/forkstore.py`: `_EPOCH_FILES` (:58) and the kernels loop in `engine_epoch()` (:248-250).
- Modify: `docs/observability.md`: the example turn line (:90-91), the gauges paragraph (:149-150) and the load-line paragraph (:215-217).
- Test:
  - `tests/test_engine_base.py`, appended after `test_status_carries_the_projection_kernel_and_the_draft_block`, the file's last test (:2784-2803);
  - `tests/test_promptcache.py`, after `test_the_drafters_gauges_follow_every_decode_path` (:670-671) and in `test_stats_as_dict_reports_every_counter` (:1191-1196);
  - `tests/test_api_turn.py`, after `test_gauge_fields_are_read_directly_not_as_deltas` (:617-620);
  - `tests/test_api_routes.py`: the line-length comment (:1665-1667), after `test_the_turn_line_attributes_a_hit` (:1752-1756) and in `test_a_served_turn_is_recorded_with_the_lines_fields` (:1949-1957);
  - `tests/test_forkstore.py`, after `test_the_projection_kernel_module_is_an_epoch_file` (:167-168).

Line numbers are those of the branch head `be778e7`. Task 5 edits `base.py`, `promptcache.py` and `tests/test_engine_base.py` before this task runs, so find each replaced block by its quoted text; every quoted block is unique in its file.

**Interfaces:**
- Consumes:
  - Task 5: `ManagedEngine.kv_int8_status -> dict | None` (`getattr(self._inner, "kv_int8_status", None)`, so a fake without the attribute reads None); `VLMEngine.kv_int8_status` and `LMEngine.kv_int8_status`, dicts with the keys `state`, `reason`, `probe_seconds`, `floor` and `ceiling`; `Speculation.kv8: str = "off"` and `Speculation.kv8_rows: int = 0`, which `VLMEngine.decode()` fills from `kv8.close_window(cache)`'s `(state, rows)` and `LMEngine.speculation()` leaves at their defaults.
  - Tasks 1–2: the kernel sources `src/sous/engine/kernels/attention_tile_kv8.metal` and `src/sous/engine/kernels/kv8_quant.metal` exist (the deny-list test checks the names against the tree).
  - The window states the gauges carry, as `kv8.close_window` returns them: `idle` (a window opened, nothing served), `served` (at least one call served), `capped` (a call whose first row was at or past the ceiling; rows may be 0); and `off` when no window opened.
  - Existing: `EngineManager.get()`/`status()`, `without_turn_gauges`, `PromptCacheStats.begin_turn`/`add`, `TurnRunner.run`, `_turn_summary`, `_turn_line`, `engine_epoch`, `_ENGINE_SOURCES`, `_EPOCH_FILES`; the test helpers `_positional_factory`, `_cfg` and `FakeEngine` (test_engine_base.py), `FakeHooks`, `PrefixCache`, `Speculation`, `STABLE_1`/`FULL_1`/`STABLE_2`/`FULL_2` (test_promptcache.py), `_runner`, `MSGS`, `RecordingSink` (test_api_turn.py), `_app`, `_post`, `_body`, `_turn_lines`, `_fields`, `_ALL_GAUGES`, `_recent` (test_api_routes.py).
- Produces:
  - The model-load line, after the `draft_block` token: ` kv_int8={state}`, then, only when the state is `active`, ` kv_int8_probe_s={probe_seconds} kv_int8_floor={floor} kv_int8_ceiling={ceiling}`. Engines without the attribute add nothing.
  - `EngineManager.status()`, after `draft_block`: `out["kv_int8"] = {"state": ..., "reason": ...}` when the engine reports a status.
  - `PromptCacheStats.kv8: str = "off"` and `PromptCacheStats.kv8_rows: int = 0`, both in `TURN_GAUGES` (so `begin_turn` resets them, `add` takes the max like every gauge, and `without_turn_gauges` drops them from the status document); `_note_speculation` copies `spec.kv8` and `spec.kv8_rows` after every decode.
  - `TurnResult.kv8: str = "off"` and `TurnResult.kv8_rows: int = 0`, read with `after.get("kv8", "off")` and `after.get("kv8_rows", 0)`.
  - `_turn_summary` keys `"kv8"` and `"kv8_rows"`, after `"draft_context"`; `_turn_line` prints ` kv8=<state> kv8_rows=<n>` right after `draft_context=<n>` and before `seconds=`.
  - `forkstore._EPOCH_EXCLUDED_KERNELS = ("attention_tile_kv8.metal", "kv8_quant.metal")`, skipped by `engine_epoch()`'s kernels scan. `_EPOCH_FILES` does not change: `kv8.py` stays out.

Why the gauges are per turn and read directly:
- They describe one decode, like `draft_context`: `close_window` returns the window's state and the rows it quantised, and `_note_speculation` assigns them after the decode, so a cold retry's decode replaces the failed warm attempt's values.
- A string gauge cannot be a delta. `add()` takes the max over owners for every gauge, which for these strings is a lexical max; that is one more reason the status document drops every turn gauge, and why the tests read them per owner, as the turn runner does.

Why the two kernels leave the epoch and the key gains nothing:
- Nothing kv8 serves is stored. The mirror lives only inside `decode()`, and the KV a generation appends is dropped: the hybrid path restores the snapshot taken before the decode, and the cold paths discard the cache. A stored fork's bits never depend on these kernels.
- Without the deny-list the `kernels/` scan would hash them, and every later edit to them would cold-start every fork store (a cold first turn was 175 s against 15.7 s from disk).
- The headers they include (`common.h`, `nax.h`) stay hashed, because the stored-KV kernels include them too.
- `vlm.py` and `tileattn.py` gain kv8 seams in Tasks 3 and 6 and are already epoch files, so this PR still changes the epoch once. That is deliberate and the PR says so.

- [ ] **Step 1: Write the failing load-line and status tests**

Append at the end of `tests/test_engine_base.py`, anchored on the last lines of the projection kernel's status test, the file's last test.

Replace:

```python
    inner.drafter = "z-lab/drafter"  # ty: ignore[unresolved-attribute]
    inner.draft_block = 5  # ty: ignore[unresolved-attribute]
    assert manager.status()["draft_block"] == 5
```

with:

```python
    inner.drafter = "z-lab/drafter"  # ty: ignore[unresolved-attribute]
    inner.draft_block = 5  # ty: ignore[unresolved-attribute]
    assert manager.status()["draft_block"] == 5


_KV8_ACTIVE = {
    "state": "active",
    "reason": None,
    "probe_seconds": 1.84,
    "floor": 1536,
    "ceiling": 84481,
}


@pytest.mark.parametrize(
    ("status", "tail"),
    [
        (
            _KV8_ACTIVE,
            " positions=engine kv_int8=active kv_int8_probe_s=1.84"
            " kv_int8_floor=1536 kv_int8_ceiling=84481",
        ),
        (
            {
                "state": "unavailable",
                "reason": "attention tile off",
                "probe_seconds": None,
                "floor": None,
                "ceiling": None,
            },
            " positions=engine kv_int8=unavailable",
        ),
        (
            {**_KV8_ACTIVE, "state": "unavailable", "reason": "probe: parity at n=1536"},
            " positions=engine kv_int8=unavailable",
        ),
        (
            {"state": "off", "reason": None, "probe_seconds": None, "floor": None, "ceiling": None},
            " positions=engine kv_int8=off",
        ),
        (None, " model=fake/model positions=engine"),
    ],
)
def test_get_logs_the_kv_int8_state(caplog, tmp_path, status, tail):
    """The probe's time, the floor and the ceiling print only when kv8 is
    active: an unavailable load serves no row, whatever its status still
    holds from a probe that ran."""
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        if status is not None:
            engine.kv_int8_status = status  # ty: ignore[unresolved-attribute]
        return engine

    mgr = EngineManager(_cfg(tmp_path), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and lines[0].endswith(tail), lines


def test_get_logs_the_kv_int8_tokens_after_the_draft_block(caplog, tmp_path):
    """kv8 enables after the block is pinned and rides on the tile, so its
    tokens close the line: the tile's, the kernel's, the block, then kv8's."""
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        engine.attention_tile_status = {  # ty: ignore[unresolved-attribute]
            "state": "active",
            "reason": None,
            "splits": 20,
            "probe_seconds": 0.43,
        }
        engine.projection_kernel_status = {  # ty: ignore[unresolved-attribute]
            "state": "active",
            "reason": None,
            "probe_seconds": 0.37,
        }
        engine.draft_block = 5  # ty: ignore[unresolved-attribute]
        engine.kv_int8_status = _KV8_ACTIVE  # ty: ignore[unresolved-attribute]
        return engine

    mgr = EngineManager(_cfg(tmp_path), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    expected = (
        " projection_kernel=active projection_kernel_probe_s=0.37 draft_block=5"
        " kv_int8=active kv_int8_probe_s=1.84 kv_int8_floor=1536 kv_int8_ceiling=84481"
    )
    assert len(lines) == 1 and lines[0].endswith(expected), lines


def test_status_carries_the_kv_int8_view_when_the_engine_reports_one(tmp_path):
    """State and reason only, as for the tile and the kernel: the floor, the
    ceiling and the probe's time are the load line's, said once per load."""
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "kv_int8" not in manager.status(), "fakes without the attribute stay silent"
    inner.kv_int8_status = _KV8_ACTIVE  # ty: ignore[unresolved-attribute]
    assert manager.status()["kv_int8"] == {"state": "active", "reason": None}
    inner.kv_int8_status = {  # ty: ignore[unresolved-attribute]
        "state": "unavailable",
        "reason": "window below the kv8 floor",
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }
    assert manager.status()["kv_int8"] == {
        "state": "unavailable",
        "reason": "window below the kv8 floor",
    }
```


`kv_int8_ceiling=84481` is the real ceiling at the 131072 window and S = 20: the largest step boundary (`640k + 1`) whose capacity `C = round_up(b, 256)` satisfies `1.504 C + 0.504 C / 16 + 272 <= 131072`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k "get_logs_the_kv_int8 or status_carries_the_kv_int8" -v`

Expected: 6 failed, 1 passed, the rest of the file deselected.
- The four `test_get_logs_the_kv_int8_state` cases with a status: `AssertionError` on a line that still ends at `positions=engine`.
- `test_get_logs_the_kv_int8_state` with `None` passes already: a fake without the attribute adds nothing.
- `test_get_logs_the_kv_int8_tokens_after_the_draft_block`: `AssertionError` on a line that ends at `draft_block=5`.
- `test_status_carries_the_kv_int8_view_when_the_engine_reports_one`: `KeyError: 'kv_int8'`.

- [ ] **Step 3: Write the load-line tokens and the status key**

In `src/sous/engine/base.py`, in `EngineManager.get()`:

Replace:

```python
        if engine.draft_block is not None:
            line += f" draft_block={engine.draft_block}"
        _logger.info(line)
        return engine
```

with:

```python
        if engine.draft_block is not None:
            line += f" draft_block={engine.draft_block}"
        kv8 = engine.kv_int8_status
        if kv8 is not None:
            line += f" kv_int8={kv8['state']}"
            # The rows kv8 serves run from the floor up to the ceiling, and
            # the ceiling follows [model].max_context_tokens, so the config
            # alone cannot say where int8 attention stops.
            if kv8["state"] == "active":
                line += (
                    f" kv_int8_probe_s={kv8['probe_seconds']}"
                    f" kv_int8_floor={kv8['floor']}"
                    f" kv_int8_ceiling={kv8['ceiling']}"
                )
        _logger.info(line)
        return engine
```


In `EngineManager.status()`:

Replace:

```python
                if self._engine.drafter is not None:
                    out["draft_block"] = self._engine.draft_block
```

with:

```python
                if self._engine.drafter is not None:
                    out["draft_block"] = self._engine.draft_block
                kv8 = self._engine.kv_int8_status
                if kv8 is not None:
                    out["kv_int8"] = {"state": kv8["state"], "reason": kv8["reason"]}
```


The status carries the state and the reason only, as it does for the tile and the projection kernel. The floor, the ceiling and the probe's time are said once per load, on the load line.

- [ ] **Step 4: Run the engine tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -q`

Expected: every test passes, with the one warning the file already raises (`drafter context unavailable (drafter names no target layers)`).

- [ ] **Step 5: Lint, type-check and commit the load line and the status**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): show kv_int8 on the load line and in the status document" -m "kv8 serves rows from a floor up to a ceiling that follows
[model].max_context_tokens, so the config alone cannot say where int8
attention runs. The model-load line names kv_int8= after draft_block=,
and when it is active also its probe time, floor and ceiling. The status
document's engine block carries kv_int8 {state, reason}. Fakes without
the attribute stay silent, as they do for the tile and the projection
kernel."
git log -1 --format='%G? %an <%ae>%n%B'
```

Expected: `G`, the configured identity, and no attribution lines.

- [ ] **Step 6: Write the failing gauge tests**

In `tests/test_promptcache.py`, after `test_the_drafters_gauges_follow_every_decode_path`:

Replace:

```python
    st = pc.stats()
    assert (st["draft_rounds"], st["drafted_tokens"], st["draft_context"]) == (1, 2, 3)
```

with:

```python
    st = pc.stats()
    assert (st["draft_rounds"], st["drafted_tokens"], st["draft_context"]) == (1, 2, 3)


def test_the_kv8_gauges_are_assigned_per_turn_from_the_last_decode():
    """The mirror's state and rows ride on the same Speculation as the
    drafter's counts: assigned after each decode, back to off and 0 with the
    next turn, and dropped from the daemon-wide view, where a max over the
    owners' states would mean nothing. Read per owner, as the turn runner
    reads them."""
    from sous.engine.promptcache import TURN_GAUGES, without_turn_gauges

    assert {"kv8", "kv8_rows"} <= TURN_GAUGES
    me = threading.current_thread()
    h = FakeHooks(trimmable=False)
    pc = PrefixCache(h)
    h.speculation_value = Speculation(kv8="capped", kv8_rows=84481)
    pc.generate(STABLE_1, FULL_1, 16)
    st = pc.stats(owner=me)
    assert (st["kv8"], st["kv8_rows"]) == ("capped", 84481)
    assert not {"kv8", "kv8_rows"} & set(without_turn_gauges(pc.stats()))
    h.speculation_value = Speculation()
    pc.generate(STABLE_2, FULL_2, 16)
    st = pc.stats(owner=me)
    assert (st["kv8"], st["kv8_rows"]) == ("off", 0)


def test_the_kv8_gauges_follow_the_decode_with_the_cache_off():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, enabled=False)
    h.speculation_value = Speculation(kv8="served", kv8_rows=40)
    pc.generate(STABLE_1, FULL_1, 16)
    st = pc.stats(owner=threading.current_thread())
    assert (st["kv8"], st["kv8_rows"]) == ("served", 40)
```


The file already imports `threading`, `PrefixCache` and `Speculation` at module level. In `test_stats_as_dict_reports_every_counter`, the expected dict gains the two defaults:

Replace:

```python
        "accepted_tokens": 0,
        "draft_context": 0,
    }


def test_stats_add_sums_pressure_evictions():
```

with:

```python
        "accepted_tokens": 0,
        "draft_context": 0,
        "kv8": "off",
        "kv8_rows": 0,
    }


def test_stats_add_sums_pressure_evictions():
```


In `tests/test_api_turn.py`, after `test_gauge_fields_are_read_directly_not_as_deltas`:

Replace:

```python
    assert 2.9 <= result.tokenize_seconds < 3.5


def test_queue_seconds_is_the_wait_for_the_endpoint_lock(tmp_path: Path):
```

with:

```python
    assert 2.9 <= result.tokenize_seconds < 3.5


def test_the_kv8_gauges_are_read_directly_and_default_off(tmp_path: Path):
    """The mirror's word on the decode is a gauge like draft_context, read
    from `after` as it stands: a lower row count than the previous turn's
    is this turn's, not a clamped delta. An engine whose stats never name
    the gauges (a fake, or an engine from before them) reads off and 0."""
    inner = FakeEngine(["a", "b"])
    runner, _ = _runner(tmp_path, inner)
    plain = runner.run(MSGS, [], 100, RecordingSink())
    assert (plain.kv8, plain.kv8_rows) == ("off", 0)
    inner.stats = {
        "hits": 0,
        "fork_hits": 0,
        "reused_tokens": 0,
        "kv8": "served",
        "kv8_rows": 90000,
    }
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        inner.stats = {**inner.stats, "kv8": "capped", "kv8_rows": 4096}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    result = runner.run(MSGS, [], 100, RecordingSink())
    assert (result.kv8, result.kv8_rows) == ("capped", 4096)


def test_queue_seconds_is_the_wait_for_the_endpoint_lock(tmp_path: Path):
```


In `tests/test_api_routes.py`, after `test_the_turn_line_attributes_a_hit`:

Replace:

```python
    assert len(f["tools"]) == 8 and len(f["system"]) == 8
    assert "lcp" not in f and "bounds" not in f


def test_the_turn_line_names_a_disk_restore(tmp_path: Path, capsys):
```

with:

```python
    assert len(f["tools"]) == 8 and len(f["system"]) == 8
    assert "lcp" not in f and "bounds" not in f


def test_the_turn_line_carries_the_kv8_gauges_after_the_drafters(tmp_path: Path, capsys):
    """kv8= and kv8_rows= follow draft_context=: both describe the decode
    behind decode_tps. A turn whose engine never reports them reads off and
    0, the LM backend's answer too."""
    inner = FakeEngine(["first", "second"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {
                **_ALL_GAUGES,
                "draft_context": 2047,
                "kv8": "served",
                "kv8_rows": 57344,
            }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    app = _app(tmp_path, inner)
    assert _post(app, _body()).status_code == 200
    assert _post(app, _body()).status_code == 200
    first, second = _turn_lines(capsys.readouterr().err)
    assert (_fields(first)["kv8"], _fields(first)["kv8_rows"]) == ("off", "0")
    assert (_fields(second)["kv8"], _fields(second)["kv8_rows"]) == ("served", "57344")
    assert " draft_context=2047 kv8=served kv8_rows=57344 seconds=" in second


def test_the_turn_line_names_a_disk_restore(tmp_path: Path, capsys):
```


In `test_a_served_turn_is_recorded_with_the_lines_fields`, the recorded summary carries both keys:

Replace:

```python
    assert turn["tools_hash"] is None and isinstance(turn["ts"], float)
    assert set(turn) == {
        "ts", "id", "model", "stream", "status", "error", "stop_reason", "cache", "took",
        "input_tokens", "output_tokens", "reused_tokens", "prefilled_tokens", "lcp",
        "lcp_region", "bounds", "forks", "evicted", "pressure", "load_s", "queue_s",
        "engine_wait_s", "tokenize_s", "ttft_s", "prefill_s", "decode_s", "persist_s",
        "restore_s", "prefill_tps", "decode_tps", "draft_rounds", "drafted", "accepted",
        "draft_context", "seconds", "tools_hash", "system_hash",
    }  # fmt: skip
```

with:

```python
    assert turn["tools_hash"] is None and isinstance(turn["ts"], float)
    assert turn["kv8"] == "off" and turn["kv8_rows"] == 0
    assert set(turn) == {
        "ts", "id", "model", "stream", "status", "error", "stop_reason", "cache", "took",
        "input_tokens", "output_tokens", "reused_tokens", "prefilled_tokens", "lcp",
        "lcp_region", "bounds", "forks", "evicted", "pressure", "load_s", "queue_s",
        "engine_wait_s", "tokenize_s", "ttft_s", "prefill_s", "decode_s", "persist_s",
        "restore_s", "prefill_tps", "decode_tps", "draft_rounds", "drafted", "accepted",
        "draft_context", "kv8", "kv8_rows", "seconds", "tools_hash", "system_hash",
    }  # fmt: skip
```


In `test_a_claimed_model_id_cannot_forge_or_bloat_a_log_line`, the comment's lengths move with the line (a `kv8=capped kv8_rows=262144` adds 27 characters, and the 600 cap still holds):

Replace:

```python
        # The attributed turn line is ~470 chars with its prefix (~510 with a
        # miss's diagnostics); the cap is here to catch an unbounded client
        # string, not to pin a length.
```

with:

```python
        # The attributed turn line is ~490 chars with its prefix (~550 with a
        # miss's diagnostics and kv8's longest state); the cap is here to
        # catch an unbounded client string, not to pin a length.
```


- [ ] **Step 7: Run them to verify they fail**

Run: `uv run pytest tests/test_promptcache.py tests/test_api_turn.py tests/test_api_routes.py -k "kv8 or reports_every_counter or recorded_with_the_lines_fields" -v`

Expected: 6 failed, the rest of the three files deselected.
- `test_the_kv8_gauges_are_assigned_per_turn_from_the_last_decode`: `AssertionError: assert {'kv8', 'kv8_rows'} <= frozenset({...})`.
- `test_the_kv8_gauges_follow_the_decode_with_the_cache_off`: `KeyError: 'kv8'`.
- `test_stats_as_dict_reports_every_counter`: `AssertionError` on the dict, whose right side has 2 more items.
- `test_the_kv8_gauges_are_read_directly_and_default_off`: `AttributeError: 'TurnResult' object has no attribute 'kv8'`.
- `test_the_turn_line_carries_the_kv8_gauges_after_the_drafters` and `test_a_served_turn_is_recorded_with_the_lines_fields`: `KeyError: 'kv8'`.

- [ ] **Step 8: Carry the gauges from the cache to the turn line**

In `src/sous/engine/promptcache.py`, `TURN_GAUGES`:

Replace:

```python
        "accepted_tokens",
        "draft_context",
    }
)
```

with:

```python
        "accepted_tokens",
        "draft_context",
        "kv8",
        "kv8_rows",
    }
)
```


`PromptCacheStats`, after `draft_context`:

Replace:

```python
    accepted_tokens: int = 0
    draft_context: int = 0

    def as_dict(self) -> dict:
```

with:

```python
    accepted_tokens: int = 0
    draft_context: int = 0
    # What the int8 K/V mirror did in the same decode: "off" when no serving
    # window opened (kv8 not active for this load), "idle" when the context
    # stayed below the floor, "served" when kv8 served a call, "capped" when
    # a call started at or past the ceiling; and the rows it quantised. A
    # string gauge is assigned, never summed, like took_kind.
    kv8: str = "off"
    kv8_rows: int = 0

    def as_dict(self) -> dict:
```


and `_note_speculation`:

Replace:

```python
def _note_speculation(stats: PromptCacheStats, hooks: CacheHooks) -> None:
    """Assign the decode's drafter gauges — the last decode call's, so a cold
    retry's counts replace the failed warm attempt's."""
    spec = hooks.speculation()
    stats.draft_rounds = spec.rounds
    stats.drafted_tokens = spec.drafted
    stats.accepted_tokens = spec.accepted
    stats.draft_context = spec.context
```

with:

```python
def _note_speculation(stats: PromptCacheStats, hooks: CacheHooks) -> None:
    """Assign the decode's drafter and kv8 gauges — the last decode call's,
    so a cold retry's values replace the failed warm attempt's."""
    spec = hooks.speculation()
    stats.draft_rounds = spec.rounds
    stats.drafted_tokens = spec.drafted
    stats.accepted_tokens = spec.accepted
    stats.draft_context = spec.context
    stats.kv8 = spec.kv8
    stats.kv8_rows = spec.kv8_rows
```


In `src/sous/api/turn.py`, `TurnResult`:

Replace:

```python
    accepted_tokens: int = 0
    draft_context: int = 0


@dataclass(frozen=True)
class CountResult:
```

with:

```python
    accepted_tokens: int = 0
    draft_context: int = 0
    # Whether that decode read int8 K/V ("off", "idle", "served" or
    # "capped") and the rows it quantised for it; "off" and 0 from an engine
    # that never reports them.
    kv8: str = "off"
    kv8_rows: int = 0


@dataclass(frozen=True)
class CountResult:
```


and its construction in `TurnRunner.run`:

Replace:

```python
                    draft_context=after.get("draft_context", 0),
                )
```

with:

```python
                    draft_context=after.get("draft_context", 0),
                    kv8=after.get("kv8", "off"),
                    kv8_rows=after.get("kv8_rows", 0),
                )
```


In `src/sous/api/routes.py`, `_turn_summary`:

Replace:

```python
        "draft_context": result.draft_context,
        "seconds": result.seconds,
```

with:

```python
        "draft_context": result.draft_context,
        # Whether that decode read int8 K/V, and the rows it quantised for
        # it: at long contexts decode_tps moves with this.
        "kv8": result.kv8,
        "kv8_rows": result.kv8_rows,
        "seconds": result.seconds,
```


and `_turn_line`:

Replace:

```python
        f"draft_context={s['draft_context']} "
        f"seconds={s['seconds']:.1f}{diag} "
```

with:

```python
        f"draft_context={s['draft_context']} "
        f"kv8={s['kv8']} kv8_rows={s['kv8_rows']} "
        f"seconds={s['seconds']:.1f}{diag} "
```


`sous/tune/bench.py` builds its own `TurnResult` from the raw gauges dict, so it needs no change; the endpoint's `TurnResult` keeps defaults for both fields, so its other constructors (tests building one by hand) keep working.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py tests/test_api_turn.py tests/test_api_routes.py -q`

Expected: 341 passed (211, 41 and 89).

Then: `uv run pytest tests/test_inflight.py tests/test_monitor.py tests/test_tui.py tests/test_engine_unloaded.py -q`

Expected: every test passes. The recent-turns ring and `sous top` read the summary by key, so two more keys change nothing there.

- [ ] **Step 10: Lint, type-check and commit the gauges**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/promptcache.py src/sous/api/turn.py src/sous/api/routes.py tests/test_promptcache.py tests/test_api_turn.py tests/test_api_routes.py
git commit -m "feat(api): put the kv8 state and rows on the turn line" -m "Whether a decode read int8 K/V changes its decode_tps at long contexts,
so the turn line says it for that decode: kv8=off|idle|served|capped and
kv8_rows=, the rows the decode quantised, right after draft_context=.
Both are per-turn prompt-cache gauges copied from the decode's
Speculation like the drafter's counts, so a cold retry's decode replaces
the warm attempt's values and the status document drops them. The turn
summary in the recent-turns ring carries the same two keys."
git log -1 --format='%G? %an <%ae>%n%B'
```

Expected: `G`, the configured identity, and no attribution lines.

- [ ] **Step 11: Write the failing fork-epoch tests**

In `tests/test_forkstore.py`, after `test_the_projection_kernel_module_is_an_epoch_file`:

Replace:

```python
def test_the_projection_kernel_module_is_an_epoch_file():
    assert "projkernel.py" in forkstore._EPOCH_FILES
```

with:

```python
def test_the_projection_kernel_module_is_an_epoch_file():
    assert "projkernel.py" in forkstore._EPOCH_FILES


def test_the_kv8_kernels_stay_out_of_the_engine_epoch(monkeypatch, tmp_path: Path):
    """The int8 K/V mirror's kernels serve only inside a generation, whose
    KV no fork keeps, so adding or editing them must leave every store
    warm; the tile's kernel and the headers they share still count."""
    src = tmp_path / "engine"
    (src / "kernels").mkdir(parents=True)
    for name in forkstore._EPOCH_FILES:
        (src / name).write_text(f"# {name}\n")
    (src / "kernels" / "attention_tile.metal").write_text("tile\n")
    (src / "kernels" / "common.h").write_text("header\n")
    monkeypatch.setattr(forkstore, "_ENGINE_SOURCES", src)
    forkstore.engine_epoch.cache_clear()
    without = engine_epoch()
    for name in ("attention_tile_kv8.metal", "kv8_quant.metal"):
        (src / "kernels" / name).write_text(f"// {name}\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() == without
    for name in ("attention_tile_kv8.metal", "kv8_quant.metal"):
        (src / "kernels" / name).write_text(f"// {name} changed\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() == without
    (src / "kernels" / "common.h").write_text("header changed\n")
    forkstore.engine_epoch.cache_clear()
    changed = engine_epoch()
    assert changed != without  # the shared header is compiled into the stored-KV kernels too
    (src / "kernels" / "attention_tile.metal").write_text("tile changed\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() != changed  # the tile's own kernel still writes stored KV
    forkstore.engine_epoch.cache_clear()


def test_the_kv8_exclusions_name_kernels_in_the_tree_and_kv8_is_no_epoch_file():
    """An excluded name that no longer matches a file excludes nothing, and
    the renamed kernel would cold-start every store on each later edit, so
    each name must still be a kernel in the tree. kv8.py stays out of the
    epoch files: nothing it serves is stored."""
    kernels = forkstore._ENGINE_SOURCES.joinpath("kernels")
    assert forkstore._EPOCH_EXCLUDED_KERNELS == ("attention_tile_kv8.metal", "kv8_quant.metal")
    for name in forkstore._EPOCH_EXCLUDED_KERNELS:
        assert kernels.joinpath(name).is_file(), name
    assert "kv8.py" not in forkstore._EPOCH_FILES
```


- [ ] **Step 12: Run them to verify they fail**

Run: `uv run pytest tests/test_forkstore.py -k kv8 -v`

Expected: 2 failed, the rest of the file deselected.
- `test_the_kv8_kernels_stay_out_of_the_engine_epoch`: `AssertionError` comparing two different 16-hex epochs, since the scan hashes the two new files.
- `test_the_kv8_exclusions_name_kernels_in_the_tree_and_kv8_is_no_epoch_file`: `AttributeError: module 'sous.engine.forkstore' has no attribute '_EPOCH_EXCLUDED_KERNELS'`.

- [ ] **Step 13: Skip the kv8 kernels in the epoch scan**

In `src/sous/engine/forkstore.py`:

Replace:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py", "projkernel.py")
```

with:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py", "projkernel.py")
# Kernel sources the kernels/ scan skips (engine_epoch says why).
_EPOCH_EXCLUDED_KERNELS = ("attention_tile_kv8.metal", "kv8_quant.metal")
```


In `engine_epoch()`:

Replace:

```python
    kernels = _ENGINE_SOURCES.joinpath("kernels")
    for entry in sorted(kernels.iterdir(), key=lambda e: e.name):
        if entry.name.endswith((".metal", ".h")):
```

with:

```python
    kernels = _ENGINE_SOURCES.joinpath("kernels")
    for entry in sorted(kernels.iterdir(), key=lambda e: e.name):
        # The int8 K/V mirror's quantiser and attention kernel run only
        # inside a generation, and the KV a generation appends is never
        # stored: the prompt cache restores the snapshot taken before the
        # decode, or drops the cache. Hashing them would cold-start every
        # store on each edit to kernels no stored KV depends on, which is
        # also why verifyattn.py is not in _EPOCH_FILES. The headers they
        # include stay hashed: the stored-KV kernels include them too.
        if entry.name in _EPOCH_EXCLUDED_KERNELS:
            continue
        if entry.name.endswith((".metal", ".h")):
```


- [ ] **Step 14: Run the fork-store tests to verify they pass**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py -q`

Expected: 94 passed (88 and 6).

- [ ] **Step 15: Lint, type-check and commit the deny-list**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/forkstore.py tests/test_forkstore.py
git commit -m "feat(engine): keep the kv8 kernels out of the fork-store epoch" -m "The int8 K/V mirror is built and read only inside a generation, and the
KV a generation appends is never stored: the prompt cache restores the
snapshot taken before the decode or drops the cache. So no stored fork
depends on the mirror's quantiser or attention kernel. Without a
deny-list the kernels/ scan would hash them, and every later edit to
them would cold-start every fork store. The headers they include stay
hashed, since the stored-KV kernels include them too, and kv8.py stays
out of the epoch files for the same reason verifyattn.py does.

The epoch still changes once with this branch: vlm.py and tileattn.py
gain the kv8 seams."
git log -1 --format='%G? %an <%ae>%n%B'
```

Expected: `G`, the configured identity, and no attribution lines.

- [ ] **Step 16: Document the load-line keys, the status key and the gauges**

In `docs/observability.md`, the example turn line:

Replace:

```
  prefill_tps=212.8 decode_tps=14.7 draft_rounds=1104 drafted=2208 accepted=1782 draft_context=1688
  seconds=207.4 tools=1b21cd75 system=9f8e7d6c
```

with:

```
  prefill_tps=212.8 decode_tps=14.7 draft_rounds=1104 drafted=2208 accepted=1782 draft_context=1688
  kv8=capped kv8_rows=84671 seconds=207.4 tools=1b21cd75 system=9f8e7d6c
```


The example is consistent at the 131072 window: kv8 served the decode's first calls from 84,335 keys and capped at the ceiling 84,481, after 146 committed rows plus the rows rejected drafts made it quantise again.

The gauges, after the drafter's paragraph:

Replace:

```
the first (with the prompt cache off the drafter sees the whole prompt,
and they are logged the same way). `seconds` is the total, running
```

with:

```
the first (with the prompt cache off the drafter sees the whole prompt,
and they are logged the same way). `kv8` says whether that decode's
attention read the int8 copy of the context's K/V: `off` when no copy
could be made (the load line does not say `kv_int8=active`), `idle` when
the context stayed below the floor (`kv_int8_floor=`) for the whole
decode, `served` when at least one decode or verify call read it, and
`capped` when a call started at or past the ceiling (`kv_int8_ceiling=`)
— from there every call of that decode ran on the bf16 attention tile,
whether or not earlier calls were served, as the turn above did once its
84,335-token prompt grew past a ceiling of 84,481. `kv8_rows` is how many
rows the decode quantised into the copy: the whole context on its first
served call, then each call's new rows, counting again the rows a
rejected draft rolled back and a later call re-quantised; 0 when nothing
was served. The first served call is the decode's own last prompt token,
so the one-time build of the copy is timed into `ttft_s` and `decode_s`.
Neither field touches prefill, the forks or the slots, which hold bf16
K/V only. `seconds` is the total, running
```


The load-line keys and the status key, after `draft_block=`:

Replace:

```
drafter's own policy; the status
document's engine block carries it as `draft_block`). One
more line names the Anthropic tool *types* a turn dropped, when any.
```

with:

```
drafter's own policy; the status
document's engine block carries it as `draft_block`), then
`kv_int8=active|unavailable|off` (whether decode's and speculative
verify's attention read an int8 copy of the context's K/V, built inside
each generation and dropped at its end, from `kv_int8_floor=` keys up to
`kv_int8_ceiling=`, with those two and `kv_int8_probe_s=` when it does;
the ceiling is the largest attention-tile step at which that copy still
fits beside the turn's bf16 K/V in the `[model].max_context_tokens`
reserve — about 84.5K keys at 131072 and 170K at 262144 — so the copy
never takes memory from the prompt-cache slots; `off` means
`[model].kv_int8 = false` or the mlx-lm backend, and `unavailable` a load
it cannot serve, every load whose attention tile is not `active` included,
since it rides on the tile, and a window too small to put the ceiling
above the floor — the status document's engine block carries `kv_int8`
with its state and reason, and the log splits INFO and `WARNING` the way
it does for the tile). One
more line names the Anthropic tool *types* a turn dropped, when any.
```


- [ ] **Step 17: Check the docs and commit**

Run: `grep -n "kv_int8\|kv8" docs/observability.md`

Expected: the example line (`kv8=capped kv8_rows=84671`), the gauges paragraph (`` `kv8` ``, `kv_int8=active`, `kv_int8_floor=`, `kv_int8_ceiling=`, `` `kv8_rows` ``) and the load-line paragraph (`kv_int8=active|unavailable|off`, `kv_int8_floor=`, `kv_int8_ceiling=`, `kv_int8_probe_s=`, `[model].kv_int8 = false`, `kv_int8` in the status document).

```bash
git add docs/observability.md
git commit -m "docs: describe kv_int8 on the load line, the status and the turn line" -m "The load line's kv_int8= with its probe time, floor and ceiling, the
status document's kv_int8 block, and the turn line's kv8= and kv8_rows=
gauges: what each state means, where the one-time build of the int8
copy is timed, and that prefill, forks and slots never hold int8 K/V."
git log -1 --format='%G? %an <%ae>%n%B'
```

Expected: `G`, the configured identity, and no attribution lines.


### Task 8: `sous tune` keys and notice, and the documentation

**Files:**
- Modify: `src/sous/tune/arms.py`: `Arm` at `:52-54`, `Arm.suite_key` at `:71-83`, `_arm`'s `Arm(...)` at `:219-221`.
- Modify: `src/sous/tune/suite/runner.py`:
  - `SuiteRun`'s fields at `:64-66`, `SuiteRun.key` at `:80-91`, `SuiteRun.from_dict` at `:102-107`;
  - the comment in `CountingEngine.__getattr__` at `:137-140`;
  - `_error_run` at `:235-236` and `run_one` at `:307-308`;
  - `_notice_unavailable` at `:349-359`, a new `_check_kv8` after `_check_proj` (`:373-381`), the call in `_run_suite` at `:468`.
- Modify: `src/sous/tune/decide.py:158`, `ArmSummary.key`'s annotation. Not in the task's file list: `ArmSummary.key` is assigned `Arm.suite_key`, and ty refuses an eight-tuple where the annotation says seven.
- Modify: `docs/tuning.md:64-73`; `docs/configuration.md`: the TOML block at `:28-30` and a new section after the projection-kernel section, which ends at `:226-228`.
- Modify: `CLAUDE.md`: the fork-store sentence at `:162-165`; a new bullet after the projkernel bullet, which ends at `:604-605`, before the draftctx bullet at `:606`.
- Test: `tests/test_tune_arms.py`, `tests/test_tune_runner.py`, `tests/test_tune_main.py`.
- Modify: `tests/test_tune_decide.py:237-238` (`_run` builds a `SuiteRun`, which gains a required field) and `tests/test_tune_report.py:289` (`_summary`'s literal key, only so ty accepts it: a seven-tuple is not assignable to the eight-tuple `ArmSummary.key` declares).

Line numbers are those of the branch head `be778e7`. If an earlier task of this plan has already edited one of these files, find each replaced block by its quoted text; every quoted block is unique in its file (the two marked `replace_all` occur exactly twice, both to be replaced).

**Interfaces:**
- Consumes:
  - Task 5: `SousConfig.kv_int8: bool = True` (so `SousConfig(..., kv_int8=False)` and `dataclasses.replace(config, kv_int8=False)` work), and `ManagedEngine.kv_int8_status -> dict | None` (`getattr(self._inner, "kv_int8_status", None)`: None for the fakes, `{"state": "off", ...}` from `LMEngine`, `kv8.enable`'s dict from `VLMEngine`).
  - The status dict `kv8.enable` returns (from the task that writes `kv8.enable`; keys `state`, `reason`, `probe_seconds`, `floor`, `ceiling`; `state` one of `off`, `unavailable`, `active`; `reason` `attention tile off` when the tile is off). The runner reads only `state` and `reason`.
  - For the documentation only (Step 9 checks they exist): `kv8.N8`, `kv8.GROWTH`, `kv8.MIRROR_ATTR`, `kv8.REL_RMS_BOUND`, `kv8.STOCK_RMS_BOUND`, `kv8.VALIDATED_KV8_SOURCES`, `kv8.Window`, `kv8.Mirror`, `kv8.ceiling`, `kv8.open_window`, `kv8.close_window`, `kv8.serves`, `kv8.observe_first_row`, `kv8.prepare`, `kv8.tile8`, `kv8.availability`, `kv8.enable`, `kv8.clear`, `kv8.probe`, `kv8.calls` and its ten keys, `kv8._active`, `kv8._window`, `kv8._ceiling`; `forkstore._EPOCH_EXCLUDED_KERNELS`; `tests/kv8_fixtures.py`'s `window`, `ceiling`, `small_floor`; the load-line tokens `kv_int8=`, `kv_int8_floor=`, `kv_int8_ceiling=`, the status key `kv_int8`, and the turn-line tokens `kv8=` and `kv8_rows=`.
  - Existing: `quick_arms`, `_arm`, `winner_stage_arms`, `SuiteRun`, `_notice_unavailable`, `_check_tile`, `_check_proj`, `_run_suite`, `run_suite`, `main`, `ArmSummary`; the test helpers `_winner`, `_candidates`, `_checkpoints` (arms), `_arm`, `_task`, `FINISH`, `PYTHON`, `FakeEngine` (runner), `_deps`, `_args`, `_cfg`, `M`, `D`, `N` (main).
- Produces:
  - `Arm.kv_int8: bool = True`, after `projection_kernel`; `Arm.suite_key -> tuple[str, str, int, bool, bool, bool, bool, bool]` = `(*self.key, int8_prefill, greedy, attention_tile, projection_kernel, kv_int8)`.
  - `_arm` sets `kv_int8=user.kv_int8`; `winner_stage_arms` carries it through `dataclasses.replace` unchanged.
  - `SuiteRun.kv_int8: bool`, required, after `projection_kernel`; `SuiteRun.key` the same eight-tuple; `SuiteRun.from_dict` fills a missing `kv_int8` with False (as it does `attention_tile` and `projection_kernel`).
  - `runner._notice_unavailable(arm, what, status, out, *, runs: str = "stock") -> None`: the existing helper with the fallback named; existing callers unchanged.
  - `runner._check_kv8(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None` (the argument order of its neighbours `_check_tile` and `_check_proj`): never raises; prints `f"  {arm.label}: int8 K/V unavailable here ({reason}); the engine runs bf16 K/V, as the daemon would"` only when `arm.kv_int8` is set and the engine reports `unavailable`; `reason` is the status's, else `no reason given`.
  - `decide.ArmSummary.key: tuple[str, str, int, bool, bool, bool, bool, bool]`.
  - Docs: the `kv_int8` paragraphs in `docs/configuration.md`, the shared inherited-settings paragraph in `docs/tuning.md`, the kv8 bullet and the fork-store sentence in `CLAUDE.md`.

**Design notes.**
- **Older rows read `kv_int8 = False`.** No row written before the setting existed read int8 K/V, on any machine, so it is a bf16 row, and a `--resume` under an arm that inherits `true` runs the task again instead of pooling bf16 and int8 rows. The key records the setting, not what the engine managed: on a Mac where kv8 cannot run, a resume across the upgrade re-runs tasks it did not strictly need to — the trade the tile's and the kernel's fallbacks already made.
- **The notice names bf16 K/V, not stock.** kv8 can be `unavailable` while the tile is `active` (a drifted pin, a compile or probe failure), and then attention runs on the bf16 tile, which is not the stock path. `_notice_unavailable` gains a keyword naming the fallback rather than a second copy of the notice. Where the tile itself is unavailable an arm prints both notices (and the projection kernel's), which is what the engine did.
- **No arm measures kv8.** The merge gate's suite step builds its on/off pair out of tree (two `Arm`s with `config` and `kv_int8` replaced, sharing `run_suite`'s scratch directory); `winner_stage_arms` gains nothing.
- **The documentation quotes no speed.** The spike's kernel ratios were measured on a variant this design changes (host-quantised Q, separate K and V); the gate's numbers go in after it, into the README with the rest. `docs/configuration.md` quotes only the format's own error, which does not depend on the kernel. `docs/observability.md` (the load-line keys, the status keys and the turn-line gauges) is not this task's; the configuration text links to its turn-line section, which exists already.
- **Ceiling figures.** "About 84K at 131072, about 170K at 262144" are `kv8.ceiling(131072, 20)` = 84,481 and `kv8.ceiling(262144, 20)` = 170,241 under the spec's inequality; Step 9 checks the implementation gives those.

- [ ] **Step 1: Write the failing tune tests**

In `tests/test_tune_arms.py`, replace the head of `_winner` (`:266-274`):

```python
def _winner(tmp_path, temperature=0.7, int8=False, tile=True, proj=True):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
        attention_tile=tile,
        projection_kernel=proj,
    )
```

with:

```python
def _winner(tmp_path, temperature=0.7, int8=False, tile=True, proj=True, kv8=True):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
        attention_tile=tile,
        projection_kernel=proj,
        kv_int8=kv8,
    )
```

In `tests/test_tune_arms.py`, replace the end of `_winner` and the first suite-key test (`:287-295`):

```python
        attention_tile=tile,
        projection_kernel=proj,
    )


def test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions(tmp_path):
    arm = _winner(tmp_path)
    assert arm.key == ("mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2", 3)
    assert arm.suite_key == (*arm.key, False, False, True, True)
```

with:

```python
        attention_tile=tile,
        projection_kernel=proj,
        kv_int8=kv8,
    )


def test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions(tmp_path):
    arm = _winner(tmp_path)
    assert arm.key == ("mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2", 3)
    assert arm.suite_key == (*arm.key, False, False, True, True, True)
```

In `tests/test_tune_arms.py`, in `test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_no_greedy_arm` (`:305`), replace:

```python
    assert int8.suite_key == (*int8.key, True, False, True, True)
```

with:

```python
    assert int8.suite_key == (*int8.key, True, False, True, True, True)
```

In `tests/test_tune_arms.py`, in `test_quick_arms_mirror_the_users_int8_and_greedy_settings` (`:365`), replace:

```python
        assert arm.suite_key == (*arm.key, True, True, True, True)
```

with:

```python
        assert arm.suite_key == (*arm.key, True, True, True, True, True)
```

In `tests/test_tune_arms.py`, in `test_quick_arms_mirror_the_users_attention_tile_and_key_their_suite_rows_by_it` (`:386`), replace:

```python
            assert arm.suite_key == (*arm.key, False, False, tile, True)
```

with:

```python
            assert arm.suite_key == (*arm.key, False, False, tile, True, True)
```

In `tests/test_tune_arms.py`, in `test_quick_arms_mirror_the_users_projection_kernel_and_key_their_suite_rows_by_it` (`:578`), replace:

```python
            assert arm.suite_key == (*arm.key, False, False, True, proj)
```

with:

```python
            assert arm.suite_key == (*arm.key, False, False, True, proj, True)
```

`test_the_winner_stage_carries_the_attention_tile_and_adds_no_arm_for_it` (`:397`) indexes the tile as `suite_key[5]`, which does not move; leave it. The projection kernel's own winner test (`:589`) read the kernel as the key's last element, which kv8 now is, so it moves to `suite_key[6]`, and the two kv8 tests follow it at the end of the file.

In `tests/test_tune_arms.py`, replace the last two lines of the file (`:588-589`):

```python
    assert all(a.projection_kernel is False and a.config.projection_kernel is False for a in arms)
    assert [a.suite_key[-1] for a in arms] == [False]
```

with:

```python
    assert all(a.projection_kernel is False and a.config.projection_kernel is False for a in arms)
    assert [a.suite_key[6] for a in arms] == [False]


def test_quick_arms_mirror_the_users_kv_int8_and_key_their_suite_rows_by_it(tmp_path):
    keys = {}
    for kv8 in (True, False):
        user = SousConfig(
            data_dir=tmp_path / "d",
            config_path=tmp_path / "c.toml",
            kv_int8=kv8,
            int8_prefill=False,
        )
        arms, refusals = quick_arms(
            user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
        )
        assert refusals == []
        for arm in arms:
            assert arm.kv_int8 is kv8 and arm.config.kv_int8 is kv8
            assert arm.suite_key == (*arm.key, False, False, True, True, kv8)
        keys[kv8] = {arm.suite_key for arm in arms}
    # A run resumed after the setting was flipped matches none of the old rows.
    assert keys[True].isdisjoint(keys[False])


def test_the_winner_stage_carries_kv_int8_and_adds_no_arm_for_it(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path, kv8=False), nax=True, checkpoint=cp)
    assert [a.label for a in arms] == ["27B + DFlash2 @3 + int8 prefill"]
    assert all(a.kv_int8 is False and a.config.kv_int8 is False for a in arms)
    assert [a.suite_key[-1] for a in arms] == [False]
```

In `tests/test_tune_runner.py`, in `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` (`:121`), replace:

```python
    assert run.key == (*_arm(tmp_path).key, False, False, True, True)
```

with:

```python
    assert run.key == (*_arm(tmp_path).key, False, False, True, True, True)
```

In `tests/test_tune_runner.py`, in `test_suite_runs_round_trip_through_dicts` (`:595-597`), replace:

```python
        attention_tile=False,
        projection_kernel=False,
        window=8192,
```

with:

```python
        attention_tile=False,
        projection_kernel=False,
        kv_int8=False,
        window=8192,
```

and its key assertion (`:612`):

```python
    assert run.key == ("org/m", "", 0, False, True, False, False)
```

with:

```python
    assert run.key == ("org/m", "", 0, False, True, False, False, False)
```

In `tests/test_tune_runner.py`, in the two `row = SuiteRun(...)` literals of `test_a_row_written_before_the_attention_tile_existed_reads_as_stock` (`:675-677`) and `test_a_row_written_before_the_projection_kernel_existed_reads_as_stock` (`:821-823`), replace (it appears twice, so use the Edit tool's `replace_all`):

```python
        attention_tile=True,
        projection_kernel=True,
        window=8192,
```

with:

```python
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        window=8192,
```

In `tests/test_tune_runner.py`, in the attention-tile row test (`:697`), replace:

```python
    assert old.key == (*arm.key, False, False, False, True)
```

with:

```python
    assert old.key == (*arm.key, False, False, False, True, True)
```

In `tests/test_tune_runner.py`, in the projection-kernel row test (`:845-849`), replace:

```python
    assert old.key == (*arm.key, False, False, True, False)
    assert old.key != arm.suite_key
    del row["attention_tile"]
    older = SuiteRun.from_dict(row)
    assert (older.attention_tile, older.projection_kernel) == (False, False)
```

with:

```python
    assert old.key == (*arm.key, False, False, True, False, True)
    assert old.key != arm.suite_key
    del row["attention_tile"]
    older = SuiteRun.from_dict(row)
    assert (older.attention_tile, older.projection_kernel) == (False, False)
```

In `tests/test_tune_runner.py`, replace the last two lines of the file (`:924-925`), appending the kv8 row test and the two notice tests:

```python
        assert outcome.runs[0].projection_kernel is case.projection_kernel, name
        assert not any("projection kernel" in line for line in lines), (name, lines)
```

with:

```python
        assert outcome.runs[0].projection_kernel is case.projection_kernel, name
        assert not any("projection kernel" in line for line in lines), (name, lines)


def test_a_row_written_before_kv_int8_existed_reads_as_bf16(tmp_path):
    arm = _arm(tmp_path)
    row = SuiteRun(
        task="t",
        index=0,
        label="m",
        model_id="org/m",
        drafter_id="",
        block_size=0,
        int8_prefill=False,
        greedy=False,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=True,
        window=8192,
        state="done",
        outcome="completed",
        turns=3,
        seconds=12.5,
        output_tokens=400,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=1.0,
        grade_detail="1/1",
        error=None,
        transcript_path=None,
    ).as_dict()
    del row["kv_int8"]
    old = SuiteRun.from_dict(row)
    # No run from before the setting existed read int8 K/V, on any machine:
    # the row is a bf16 row, and the settings it did record stay as written.
    assert old.kv_int8 is False
    assert (old.attention_tile, old.projection_kernel) == (True, True)
    # The arm inherits the setting's default, so a resume runs the task again
    # rather than counting the bf16 row toward the int8 arm.
    assert arm.kv_int8 is True
    assert old.key == (*arm.key, False, False, True, True, False)
    assert old.key != arm.suite_key


class _Kv8Unavailable(FakeEngine):
    drafter = ""
    kv_int8_status = {
        "state": "unavailable",
        "reason": "attention tile off",
        "probe_seconds": None,
        "floor": None,
        "ceiling": None,
    }


def test_an_arm_that_inherits_kv_int8_runs_bf16_where_the_engine_cannot_serve_it(tmp_path):
    """kv_int8 is inherited from the user's config and no arm measures it:
    an engine that reports it unavailable runs the arm on bf16 K/V after a
    notice, as the daemon would."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _Kv8Unavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert outcome.runs[0].kv_int8 is True
    assert (
        "  m: int8 K/V unavailable here (attention tile off); "
        "the engine runs bf16 K/V, as the daemon would"
    ) in lines


def test_no_kv8_notice_when_it_is_active_off_unreported_or_not_inherited(tmp_path):
    """`off` is the setting switched off or an engine without kv8 (the
    mlx-lm backend), and an engine that reports nothing has none either:
    neither is a machine where kv8 cannot run, so neither gets the notice."""

    class Active(FakeEngine):
        drafter = ""
        kv_int8_status = {
            "state": "active",
            "reason": None,
            "probe_seconds": 0.5,
            "floor": 1536,
            "ceiling": 84481,
        }

    class Off(FakeEngine):
        drafter = ""
        kv_int8_status = {
            "state": "off",
            "reason": None,
            "probe_seconds": None,
            "floor": None,
            "ceiling": None,
        }

    class Silent(FakeEngine):
        drafter = ""

    arm = _arm(tmp_path)
    bf16 = dataclasses.replace(
        arm, kv_int8=False, config=dataclasses.replace(arm.config, kv_int8=False)
    )
    for name, case, engine in (
        ("active", arm, Active),
        ("off", arm, Off),
        ("silent", arm, Silent),
        ("bf16", bf16, _Kv8Unavailable),
    ):
        lines = []
        outcome = run_suite(
            case,
            [_task()],
            runs=1,
            done=set(),
            record=lambda r: None,
            scratch=tmp_path / f"s-{name}",
            out=lines.append,
            factory=lambda mid, engine=engine: engine([FINISH]),
            python=PYTHON,
            active_memory=lambda: 0,
        )
        assert len(outcome.runs) == 1 and outcome.error is None, name
        assert outcome.runs[0].kv_int8 is case.kv_int8, name
        assert not any("int8 K/V" in line for line in lines), (name, lines)
```

In `tests/test_tune_main.py`, in `_suite`'s `SuiteRun(...)` (`:135-136`), replace:

```python
                    projection_kernel=arm.projection_kernel,
                    window=arm.window,
```

with:

```python
                    projection_kernel=arm.projection_kernel,
                    kv_int8=arm.kv_int8,
                    window=arm.window,
```

In `tests/test_tune_main.py`, in `test_a_stopped_suite_still_reports_the_runs_it_recorded`'s `SuiteRun(...)` (`:578-579`), replace:

```python
            projection_kernel=arm.projection_kernel,
            window=arm.window,
```

with:

```python
            projection_kernel=arm.projection_kernel,
            kv_int8=arm.kv_int8,
            window=arm.window,
```

In `tests/test_tune_main.py`, insert the resume test before `test_resume_retries_a_run_the_runner_recorded_as_an_error`: replace its first line (`:507`):

```python
def test_resume_retries_a_run_the_runner_recorded_as_an_error(tmp_path, capsys):
```

with:

```python
def test_resume_grades_every_suite_arm_again_after_kv_int8_was_flipped(tmp_path, capsys):
    """kv_int8 changes attention numerics past the floor, so suite rows are
    keyed by it: a resume after the flip pools none of the earlier rows with
    the new ones and grades every arm from scratch."""
    seen = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N), seen=seen)
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    assert first and all(done == [] for _, done in first)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    flipped = _cfg(tmp_path, kv_int8=False)
    assert main(_args(quick=False, resume=run_id), config=flipped, **deps) == 0
    assert seen == first
    results = (tmp_path / "tune" / run_id / "results.jsonl").read_text().splitlines()
    suite_rows = [r for r in map(json.loads, results) if r["kind"] == "suite"]
    assert {r["kv_int8"] for r in suite_rows} == {True, False}


def test_resume_retries_a_run_the_runner_recorded_as_an_error(tmp_path, capsys):
```

In `tests/test_tune_decide.py`, in `_run` (`:237-238`), replace:

```python
        projection_kernel=arm.projection_kernel,
        window=arm.window,
```

with:

```python
        projection_kernel=arm.projection_kernel,
        kv_int8=arm.kv_int8,
        window=arm.window,
```

In `tests/test_tune_report.py`, in `_summary` (`:289`), replace:

```python
        key=(model, "d", 3, False, False, True, True),
```

with:

```python
        key=(model, "d", 3, False, False, True, True, True),
```

- [ ] **Step 2: Run the tune tests to verify they fail**

Run: `uv run pytest tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`

Expected: 49 failed, 106 passed (arms 14 of 36, runner 7 of 32, main 11 of 37, decide 17 of 33, report 0 of 17). The failures are:
- arms: the nine tests that build `_winner` (`test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions`, `test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_no_greedy_arm`, `test_a_winner_the_user_set_greedy_keeps_greedy_on_its_int8_arm`, `test_without_nax_no_extra_arm_is_offered`, `test_a_checkpoint_int8_cannot_route_gets_no_int8_arm`, `test_a_winner_with_int8_on_gets_no_extra_arms`, `test_a_moe_checkpoint_gets_no_int8_arm`, the tile's and the kernel's winner-stage tests) and `test_the_winner_stage_carries_kv_int8_and_adds_no_arm_for_it` with `TypeError: Arm.__init__() got an unexpected keyword argument 'kv_int8'`; the int8/greedy, tile and kernel `quick_arms` tests with `AssertionError` on the seven-element `suite_key`; `test_quick_arms_mirror_the_users_kv_int8_and_key_their_suite_rows_by_it` with `AttributeError: 'Arm' object has no attribute 'kv_int8'`.
- runner: `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` on its key assertion; `test_suite_runs_round_trip_through_dicts`, the two older-row tests and `test_a_row_written_before_kv_int8_existed_reads_as_bf16` with `TypeError: SuiteRun.__init__() got an unexpected keyword argument 'kv_int8'`; `test_an_arm_that_inherits_kv_int8_runs_bf16_where_the_engine_cannot_serve_it` with `AttributeError: 'SuiteRun' object has no attribute 'kv_int8'`; `test_no_kv8_notice_when_it_is_active_off_unreported_or_not_inherited` with `TypeError: Arm.__init__() got an unexpected keyword argument 'kv_int8'` (from `dataclasses.replace`).
- main: every test whose suite builds a `SuiteRun` (both `test_a_full_run_never_measures_or_proposes_a_temperature` cases, `test_a_full_run_grades_each_models_fastest_arm_and_can_change_the_model`, `test_the_winner_stage_offers_only_what_the_hardware_allows`, `test_resume_skips_suite_runs_already_recorded`, the new `test_resume_grades_every_suite_arm_again_after_kv_int8_was_flipped`, `test_resume_retries_a_run_the_runner_recorded_as_an_error`, `test_the_daemon_is_asked_again_before_every_suite_arm`, `test_a_stopped_suite_still_reports_the_runs_it_recorded`, `test_a_stop_in_the_winner_stage_keeps_the_model_stages_choice`, `test_a_daemon_that_gets_busy_mid_suite_stops_with_the_resume_hint`) with `AttributeError: 'Arm' object has no attribute 'kv_int8'`.
- decide: the 17 tests built on `_run`, with the same `AttributeError`.

If any test fails with `TypeError: SousConfig.__init__() got an unexpected keyword argument 'kv_int8'` instead, Task 5 has not landed: stop and land it first.

- [ ] **Step 3: Mirror the flag on the arm and key it**

In `src/sous/tune/arms.py`, replace (in `Arm`, `:52-54`):

```python
    # Inherited and keyed the same way, for the same reason: kernel and stock
    # projections differ in their last bits.
    projection_kernel: bool = True
```

with:

```python
    # Inherited and keyed the same way, for the same reason: kernel and stock
    # projections differ in their last bits.
    projection_kernel: bool = True
    # Inherited and keyed the same way: attention over int8 K/V differs from
    # attention over bf16 K/V by the quantisation, not only in the last bits.
    kv_int8: bool = True
```

In `src/sous/tune/arms.py`, replace `Arm.suite_key` (`:71-83`):

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the one setting only the suite may change, and whether the
        sampler is greedy, the attention tile and the projection kernel,
        which every arm inherits."""
        return (
            *self.key,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
        )
```

with:

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the one setting only the suite may change, and whether the
        sampler is greedy, the attention tile, the projection kernel and
        int8 K/V, which every arm inherits."""
        return (
            *self.key,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
            self.kv_int8,
        )
```

In `src/sous/tune/arms.py`, replace the end of `_arm`'s `Arm(...)` (`:219-221`):

```python
        attention_tile=user.attention_tile,
        projection_kernel=user.projection_kernel,
    )
```

with:

```python
        attention_tile=user.attention_tile,
        projection_kernel=user.projection_kernel,
        kv_int8=user.kv_int8,
    )
```

In `src/sous/tune/decide.py`, replace (in `ArmSummary`, `:158`):

```python
    key: tuple[str, str, int, bool, bool, bool, bool]
```

with:

```python
    key: tuple[str, str, int, bool, bool, bool, bool, bool]
```

- [ ] **Step 4: Key the row and add the notice**

In `src/sous/tune/suite/runner.py`, replace (in `SuiteRun`, `:64-66`):

```python
    attention_tile: bool
    projection_kernel: bool
    window: int
```

with:

```python
    attention_tile: bool
    projection_kernel: bool
    kv_int8: bool
    window: int
```

In `src/sous/tune/suite/runner.py`, replace `SuiteRun.key` (`:80-91`):

```python
    @property
    def key(self) -> tuple[str, str, int, bool, bool, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (
            self.model_id,
            self.drafter_id,
            self.block_size,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
        )
```

with:

```python
    @property
    def key(self) -> tuple[str, str, int, bool, bool, bool, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (
            self.model_id,
            self.drafter_id,
            self.block_size,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
            self.kv_int8,
        )
```

In `src/sous/tune/suite/runner.py`, replace (in `SuiteRun.from_dict`, `:102-107`):

```python
        # A row written before a setting existed ran without what it switches
        # on — stock attention, stock projections — on every machine, so it
        # reads False, and a resume under an arm that has the setting runs
        # the task again rather than pool the two. Without the fallback,
        # resuming such a run would raise KeyError.
        d = {"attention_tile": False, "projection_kernel": False, **d}
```

with:

```python
        # A row written before a setting existed ran without what it switches
        # on — stock attention, stock projections, bf16 K/V — on every
        # machine, so it reads False, and a resume under an arm that has the
        # setting runs the task again rather than pool the two. Without the
        # fallback, resuming such a run would raise KeyError.
        d = {"attention_tile": False, "projection_kernel": False, "kv_int8": False, **d}
```

In `src/sous/tune/suite/runner.py`, replace the comment in `CountingEngine.__getattr__` (`:137-140`):

```python
        # drafter, draft_block, positions, int8_prefill_status,
        # verify_attention_status, draft_context_status, attention_tile_status,
        # projection_kernel_status: whatever the backend has, read through
        # getattr(..., None) by ManagedEngine.
```

with:

```python
        # drafter, draft_block, positions, int8_prefill_status,
        # verify_attention_status, draft_context_status, attention_tile_status,
        # projection_kernel_status, kv_int8_status: whatever the backend has,
        # read through getattr(..., None) by ManagedEngine.
```

In `src/sous/tune/suite/runner.py`, in `_error_run` (`:235-236`) and `run_one` (`:307-308`), replace (it appears twice, so use the Edit tool's `replace_all`):

```python
        projection_kernel=arm.projection_kernel,
        window=arm.window,
```

with:

```python
        projection_kernel=arm.projection_kernel,
        kv_int8=arm.kv_int8,
        window=arm.window,
```

In `src/sous/tune/suite/runner.py`, replace `_notice_unavailable` (`:349-359`):

```python
def _notice_unavailable(arm: Arm, what: str, status: dict | None, out: Callable[..., None]) -> None:
    """The notice for a setting the arm inherits and the engine reports
    unavailable; any other state, or no status, prints nothing."""
    status = status or {}
    if status.get("state") != "unavailable":
        return
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: {what} unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )
```

with:

```python
def _notice_unavailable(
    arm: Arm,
    what: str,
    status: dict | None,
    out: Callable[..., None],
    *,
    runs: str = "stock",
) -> None:
    """The notice for a setting the arm inherits and the engine reports
    unavailable; any other state, or no status, prints nothing. `runs` names
    what the engine falls back to."""
    status = status or {}
    if status.get("state") != "unavailable":
        return
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: {what} unavailable here ({reason}); "
        f"the engine runs {runs}, as the daemon would"
    )
```

In `src/sous/tune/suite/runner.py`, replace the end of `_check_proj` (`:380-381`), adding `_check_kv8` after it:

```python
    if arm.projection_kernel:
        _notice_unavailable(arm, "projection kernel", engine.projection_kernel_status, out)
```

with:

```python
    if arm.projection_kernel:
        _notice_unavailable(arm, "projection kernel", engine.projection_kernel_status, out)


def _check_kv8(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """int8 K/V is inherited and unmeasured like the tile, so the same holds:
    where the engine reports it unavailable, the daemon would serve this
    checkpoint's attention from the bf16 cache, and the arm runs the same way
    after a notice. The fallback is bf16 K/V, not stock: where only kv8
    refused, the tile still serves. `off`, or no status, is the setting
    switched off or an engine without it: no notice."""
    if arm.kv_int8:
        _notice_unavailable(arm, "int8 K/V", engine.kv_int8_status, out, runs="bf16 K/V")
```

In `src/sous/tune/suite/runner.py`, replace the call in `_run_suite` (`:468`):

```python
            _check_proj(arm, engine, out)
```

with:

```python
            _check_proj(arm, engine, out)
            _check_kv8(arm, engine, out)
```

- [ ] **Step 5: Run the tune tests to verify they pass**

Run: `uv run pytest tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`
Expected: 155 passed (arms 36, runner 32, main 37, decide 33, report 17).

- [ ] **Step 6: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add src/sous/tune/arms.py src/sous/tune/suite/runner.py src/sous/tune/decide.py tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py
git commit -m "feat(tune): key suite rows by kv_int8 and say when an arm runs bf16 K/V" -m "kv_int8 moves attention past its floor by the int8 quantisation, not
only in the last bits, so suite rows are keyed by it as by the attention
tile and the projection kernel: a resume after the setting was flipped
grades the arms again instead of pooling int8 and bf16 rows. Rows from
before the setting existed never read int8 K/V and read as false.

Every arm inherits the setting and none measures it. Where the engine
reports it unavailable the arm runs on as the daemon would, after a
notice that names bf16 K/V rather than stock: kv8 can refuse while the
tile still serves."
```

Expected: ruff and ty report nothing, the full non-model suite passes, and `git log -1 --format='%G? %an <%ae>%n%B'` shows `G`, the configured identity and no attribution lines.

- [ ] **Step 7: Document the setting and the suite key**

In `docs/tuning.md`, replace (`:64-73`):

```markdown
Every arm inherits `[model].attention_tile` and `[model].projection_kernel`
and no arm changes them; suite rows are keyed by both, so a `--resume` after
either was flipped runs those arms again instead of mixing kernel and stock
rows, and where the engine reports one `unavailable`, the arm prints a notice
with the reason and runs stock, as the daemon would (an `off` engine, such as
the mlx-lm backend's, gets no notice). Bench rows are keyed only by model,
drafter, block and window, so a `--resume` across a sous upgrade or a flip of
`attention_tile` or `projection_kernel` reuses decode and prefill timings
measured before it, which can rank the blocks the other way: start a fresh
tune after either.
```

with:

```markdown
Every arm inherits `[model].attention_tile`, `[model].projection_kernel` and
`[model].kv_int8` and no arm changes them; suite rows are keyed by all three,
so a `--resume` after any of them was flipped runs those arms again instead of
mixing rows measured with and without it (a row written before a setting
existed reads as `false` for it), and where the engine reports one
`unavailable`, the arm prints a notice with the reason and runs without it, as
the daemon would: stock attention or projections, or attention over the bf16
cache for `kv_int8` (an `off` engine, such as the mlx-lm backend's, gets no
notice). `kv_int8` matters most here of the three: attention past its floor
reads keys and values quantised to int8, which moves more than the last bits,
and the suite's grades are where that would show. Bench rows are keyed only by
model, drafter, block and window, so a `--resume` across a sous upgrade or a
flip of `attention_tile`, `projection_kernel` or `kv_int8` reuses decode and
prefill timings measured before it, which can rank the blocks the other way:
start a fresh tune after any of them.
```

In `docs/configuration.md`, replace the end of the TOML block (`:28-30`):

```markdown
attention_tile = true
projection_kernel = true
```
```

with:

```markdown
attention_tile = true
projection_kernel = true
kv_int8 = true
```
```

In `docs/configuration.md`, replace the last three lines of the `projection_kernel` section (`:226-228`), adding the `kv_int8` section after it:

```markdown
should have run here and did not. The on-disk forks are keyed by it: where it
is active, changing the setting starts the fork store cold once. The daemon
reads it at startup, so a change takes effect on its next start.
```

with:

```markdown
should have run here and did not. The on-disk forks are keyed by it: where it
is active, changing the setting starts the fork store cold once. The daemon
reads it at startup, so a change takes effect on its next start.

`[model].kv_int8` (default `true`) runs decode's and speculative verify's
attention over an int8 copy of the context's keys and values instead of the
bf16 cache, on the attention tile's key partition (sous's own kernels,
compiled at model load — no build step). Attention over a long context is
dominated by reading the keys and values, and int8 halves those bytes. Every
key and value row is quantised per token and KV head with one fp16 scale, and
each query row inside the kernel with one fp32 scale; on the default model's
keys and values from a real 35K-token prompt that costs about 1 % relative
error on the keys and values themselves and 0.7 % median (1.8 % in the worst
layer) on the attention output. So, unlike the tile and the projection
kernel, it moves more than the last bits: a reply can differ from one with
`false`, which reads the bf16 cache. Greedy output with the drafter still
equals output without it, since a verify row reads the same int8 rows as the
decode step at the same key count.

Only a generation reads the copy. It is built from the bf16 cache on the
first decode step at or past the floor and dropped when the generation ends,
so prefill, the prompt cache, the on-disk forks and everything stored stay
bf16, and the fork store does not depend on the setting. It serves attention
from the floor, 1,536 keys (the model-load line's `kv_int8_floor=`), up to a
ceiling set by `max_context_tokens` (`kv_int8_ceiling=`). The copy is half
the size of the bf16 cache it mirrors and lives inside the memory the daemon
already reserves for a turn's KV, so it serves only below the context length
at which the turn's bf16 cache and the copy still fit there together: about
84K keys at 131072, about 170K at 262144. Neither the reserve nor the
prompt-cache budget grows for it. A generation whose context reaches the
ceiling drops its copy and runs on over the bf16 cache, so one request takes
the same path whatever else the daemon holds; below the floor attention is
the tile's or the stock path, unchanged. The turn line's `kv8=` says what a
generation did — `served`, `capped` at the ceiling, `idle` (its context
stayed below the floor) or `off` — and `kv8_rows=` how many rows it quantised
(see [Observability](observability.md#the-turn-line)).

It runs only where the attention tile is active (so far a 20-core M5 Pro;
see `attention_tile` above), with the mlx-vlm functions it relies on unchanged
(sous pins their sources, as it does the tile's). Where the tile is off or
`unavailable`, the model-load line reads `kv_int8=unavailable` (`off` with
`false` and on the mlx-lm backend), the status document's `kv_int8` block
carries the reason, and attention reads the bf16 cache; that is one INFO line,
and a load warning about it means it should have run here and did not (its
kernels or load-time check failed, or a pinned mlx-vlm source changed under
it). A value other than `true` or `false` warns and means `false`. The daemon
reads it at startup, so a change takes effect on its next start.
```

- [ ] **Step 8: Add the kv8 bullet and the epoch exclusion to CLAUDE.md**

In `CLAUDE.md`, replace the fork-store sentence's file list (`:162-165`):

```markdown
  why the fork store's identity key hashes the engine sources
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  projkernel.py, kernels/*.metal and *.h) beside the versions, the GPU and
  the weights snapshot — any
```

with:

```markdown
  why the fork store's identity key hashes the engine sources
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  projkernel.py, kernels/*.metal and *.h except the two kv8 kernels,
  `attention_tile_kv8.metal` and `kv8_quant.metal`, which
  `_EPOCH_EXCLUDED_KERNELS` names because they serve only inside a
  generation and never touch stored KV) beside the versions, the GPU and
  the weights snapshot — any
```

In `CLAUDE.md`, replace the end of the projkernel bullet and the first line of the draftctx bullet (`:604-606`):

```markdown
  and remembers success only: the lane map was probed on Apple GPUs,
  so a GPU where it differs skips rather than fails.
- `engine/draftctx.py` hands the DFlash drafter the prompt's hidden states on
```

with:

```markdown
  and remembers success only: the lane map was probed on Apple GPUs,
  so a GPU where it differs skips rather than fails.
- `engine/kv8.py` serves decode's one-row attention and the exact
  verifier's attention from an int8 *mirror* of the bf16 KV cache, on the
  attention tile's partition: `kernels/attention_tile_kv8.metal`
  (`sous_kv8_tile_split`, the tile's split kernel over int8 K, V and Q; the
  tile's reduce kernel is reused unchanged) and `kernels/kv8_quant.metal`
  (`sous_kv8_quant`, the row quantiser, `common.h` only so it compiles on
  any Metal GPU and CI tests it). K and V are symmetric int8 per (token, KV
  head) with an fp16 scale (amax / 127, round half to even, codes in
  [−127, 127], a scale that rounds to 0 divides by 1, so a zero row gives
  codes 0); Q is quantised the same way inside the kernel with one fp32
  scale per (query head, row); the K and Q scales multiply the finished
  score and the V scale folds into the probability, so nothing is
  dequantised in the loop. The mirror is a pure per-row function of the
  bf16 rows, so bulk and incremental quantisation agree bit for bit and a
  verify row reads the same codes as the decode step at its key count:
  greedy parity with the drafter on or off holds as for the tile, while
  kv8 itself differs from the bf16 tile by the quantisation, not in the
  last bits. The mirror exists only inside `VLMEngine.decode()`, which
  calls `kv8.open_window()` inside the draft-context `seed` block
  immediately before `stream_generate` (so the one-row forward that ends
  the decode call's own prefill segment is covered) and
  `kv8.close_window(cache)` in that loop's `finally`: it drops every
  layer's mirror and trim wrap, returns `(state, rows)` for the
  `Speculation`, and calls `mx.clear_cache()` when a mirror was built.
  Every `prefill()`, the load-time probes and anything outside `decode()`
  run with no window and never build or read a mirror, so the KVCache,
  the slots, the forks and the fork store stay bf16 and nothing kv8
  serves is ever persisted. The window is a module global (`_window`, a
  `Window` with `serial`, `state` `idle`/`served`/`capped`, `rows`,
  `served`, `built`, `force`): one suffices because
  `ManagedEngine._gen_lock` serialises every forward and every `decode()`
  call site runs on the caller's session thread, and `close_window` runs
  before `PrefixCache` restores or trims the cache, so no trim wrap is in
  place when it does. A `Mirror` hangs off a full-attention layer's
  `KVCache` as `MIRROR_ATTR` (`_sous_kv8`, set with `object.__setattr__`;
  `fork_copy` builds a fresh cache, so a mirror never travels with a
  fork): `codes` int8 `[1, 4, cap, 512]` (each token's K then its V, so
  every 16-byte load stays aligned), `sc` fp16 `[1, 4, cap, 2]`, `cap` a
  multiple of 256 and deliberately not `keys.shape[2]` (a cache's
  capacity is routinely not a multiple of 4, and the 16-byte scale loads
  need the plane stride to be a multiple of 16), `n` valid rows and the
  `serial` of the window that built it. Validity follows the cache's
  offset through `trim`, not through the calls kv8 serves: inside a
  decode only `KVCache.trim` lowers an offset, and DFlash's commit and
  abort both reach it as an instance call (`_trim_append_caches` resolves
  `getattr(cache, "trim")`, `SpeculativeCacheTransaction.abort` calls
  `cache.trim`), so the mirror shadows `trim` on the instance and sets
  `n = min(n, offset)` after the original runs. The wrap is the
  mechanism; the clamp every served call runs first (`n = min(n, offset −
  T)`) and the rebuild from row 0 of a mirror from another window are
  defences. A served call quantises `[n, offset)` once
  (`kv8.prepare(cache, t)`, before a verify's run loop): the first build
  writes the cap-sized buffers in one dispatch per layer, later calls
  slice-assign a row-sized output in place, and a growth (`GROWTH` = 4096
  rows, never past `round_up(L, 256)`) goes layer by layer, each new
  buffer evaluated before the old one is dropped. A row is served iff
  `kv8.serves(n_row)` — a window is open and not capped and `N8 ≤ n_row
  < L` (or the window is forced) — the one predicate both tile seams call
  (decode: `n_keys`; a verify run: its first row's key count). `N8`
  (1536, a module constant equal to `N0` today, which may only ever move
  to a later step boundary, where per-round quantisation no longer costs
  more than kv8 saves) and `L` are tile step boundaries, so no
  `verify_plan` run straddles either and a call mixes kv8 and the bf16
  tile only across runs. `L` (`_ceiling`) is computed once by `enable()` with the pure
  `ceiling(window, splits)` from `[model].max_context_tokens`: the largest
  step boundary whose mirror fits beside the turn's bf16 KV inside the
  window-sized reserve at every instant (`1.504·C + 0.504·C/16 + 272 ≤
  window`, `C = round_up(L, 256)`: the bf16 KV, the mirror, one layer's old
  buffer during a growth, and the rows a straddling verify holds past L),
  about 84K at 131072 and 170K at 262144. The reserve, the slot budget and
  the pressure valve are deliberately untouched: scaling the reserve would
  cost 4–8 GiB of slot budget, and admitting against live memory would
  make a prompt's output depend on slot occupancy. A call whose first row
  is at or past `L` caps the window (`kv8.observe_first_row`: `state =
  "capped"`, the mirrors dropped) and every later call of that decode runs
  on the bf16 tile; the committed prefix never shrinks inside a decode, so
  the cap is sticky and drafter-on and drafter-off serve exactly the same
  rows. A window of 0 (every test-built engine) gives `L = 0`, which
  serves nothing; `open_window(force=True)` lifts `L` for the probe and the
  array-level tests only. `tileattn` imports `kv8` function-locally inside
  its two seams and `kv8` imports `tileattn` at module level, so there is
  no cycle. `enable()` runs after `projkernel.enable` (the tile's state is
  settled) and before the fork store and `measure_cache_budget` (the
  probe's arrays are released first); its first statement clears
  `_active` and `_window`. It checks, in order: the config (`off`); the
  tile (`active`, else one INFO line and `unavailable` with the tile's
  reason or `attention tile off`); the source pins
  (`VALIDATED_KV8_SOURCES`: `KVCache.update_and_fetch`, `trim` and
  `prefix_cache_reserve`, the `cache_state` transaction, commit, abort and
  rollback functions, `_dflash_rounds`, `_dflash_verify`,
  `_dflash_verify_greedy`, `_reserve_dflash_target_cache`, the qwen3_5
  `rollback_speculative_cache` and verifier `verify` — the mirror assumes
  rows are appended through `update_and_fetch` and rolled back through the
  instance `trim`, never replayed, and a cache moved onto a replaying path
  would go stale with every other gate green); `availability()` (the
  shared platform rule, then a compile and run of T = 1..8 and the
  quantiser, success remembered only); that `N8` is a step boundary for
  the active S; `L > N8` (else one INFO line); and the probe on the
  `sous-model-load` thread. Pin drift, a compile or probe failure and any
  exception (logged with its traceback) are one `warnings.warn` plus
  `unavailable`. The probe proves the quantiser bitwise against the
  mlx-op reference on random, tie (power-of-two scale) and all-zero rows,
  bulk equal to incremental; the kernel within `REL_RMS_BOUND` (1e-2)
  relative RMS of fp32 SDPA over the dequantised codes at N8 and the far
  mark (T = 1, 5, 8, with a dominant key at n − 1 and a decoy at n); every
  verify row `array_equal` its one-row call at N8, the next two step
  boundaries and the far mark, T = 1..8; NaN scales and −128 codes past n
  leaving the output bitwise unchanged and finite; the mirror rule on a
  real `KVCache` (a trim through the wrap, appends with and without a
  served call, the shrinking block across N8, a stale serial, a capacity
  not a multiple of 4, a growth); quantisation error within
  `STOCK_RMS_BOUND` (3e-2) of stock bf16 SDPA; and the model's own
  attention module through the decode hook and, when verify attention is
  active, through the verifier's wrapper, inside a forced window. It
  raises `_active` and opens its window only while it runs, and in
  `finally` closes the window, drops its mirrors, clears `_active`,
  restores `kv8.calls`, `tileattn.calls` and `projkernel.calls` and calls
  `mx.clear_cache()`. `VLMEngine.unload()` calls `kv8.clear()` (`_active`
  and `_window`, nothing else: the mirrors live on caches dropped with
  the engine). The constructors default it off (`VLMEngine(kv_int8=False,
  kv_int8_window=0)`); only `default_engine_factory` hands them the
  config's `true` and the window. Nothing is logged per turn; `kv8.calls`
  (`decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`,
  `mirror_builds`, `mirror_rows`, `mirror_trims`, `mirror_grown`,
  `windows`, `windows_capped`) counts every path for the tests and for
  harnesses, and the turn line carries `kv8=` and `kv8_rows=` through
  `Speculation` and the per-turn gauges. The fork key gains no field and
  `kv8.py` stays out of `_EPOCH_FILES`; the two kernel files are kept out
  of the kernels glob by `forkstore._EPOCH_EXCLUDED_KERNELS`, so a
  kv8-only kernel edit never cold-starts the store (the seams in `vlm.py`
  and `tileattn.py` are hashed as before). `sous tune` keys suite rows by
  `kv_int8` (`Arm.kv_int8`, `SuiteRun.kv_int8`, older rows reading
  `false`) and prints its notice only for `unavailable`, never for `off`.
  Tests swap the plain-mlx stand-in in for `kv8.tile8`
  (`tests/kv8_fixtures.py`), open windows through its `window()` context
  manager, set `L` with `ceiling(monkeypatch, L)` and lower the floor
  with `small_floor`; the `nax` tests skip on `kv8.availability()`, never
  the tile's. Every kernel edit re-runs the exactness tests and the
  T = 1..8 timings against the bf16 tile: the bits are what parity rests
  on and the speed is the only reason the mirror exists.
- `engine/draftctx.py` hands the DFlash drafter the prompt's hidden states on
```

- [ ] **Step 9: Check that every name the documentation cites exists, and the ceiling figures**

The bullet and the configuration text name symbols other tasks define; a renamed one would leave CLAUDE.md describing code that is not there.

```bash
uv run python - <<'EOF'
import re
from pathlib import Path

from sous.engine import forkstore, kv8

missing = [
    name
    for name in (
        "N8", "GROWTH", "MIRROR_ATTR", "REL_RMS_BOUND", "STOCK_RMS_BOUND",
        "VALIDATED_KV8_SOURCES", "Window", "Mirror", "ceiling", "open_window",
        "close_window", "serves", "observe_first_row", "prepare", "tile8",
        "availability", "enable", "clear", "probe", "calls", "_active",
        "_window", "_ceiling",
    )
    if not hasattr(kv8, name)
]
if not hasattr(forkstore, "_EPOCH_EXCLUDED_KERNELS"):
    missing.append("forkstore._EPOCH_EXCLUDED_KERNELS")
want = {
    "decode_kv8", "verify_kv8", "verify_rows_kv8", "verify_launches_kv8",
    "mirror_builds", "mirror_rows", "mirror_trims", "mirror_grown",
    "windows", "windows_capped",
}
if set(kv8.calls) != want:
    missing.append(f"kv8.calls keys {sorted(set(kv8.calls) ^ want)}")
fixtures = Path("tests/kv8_fixtures.py").read_text()
missing += [f"kv8_fixtures.{n}" for n in ("window", "ceiling", "small_floor")
            if not re.search(rf"^def {n}\(", fixtures, re.M)]
if kv8.MIRROR_ATTR != "_sous_kv8" or kv8.N8 != 1536 or kv8.GROWTH != 4096:
    missing.append("a constant the bullet quotes changed")
print("missing:", missing)
print("ceilings:", kv8.ceiling(131072, 20), kv8.ceiling(262144, 20))
print("excluded:", forkstore._EPOCH_EXCLUDED_KERNELS)
EOF
grep -c "kv8=\|kv8_rows" src/sous/api/routes.py
grep -c "kv_int8_floor\|kv_int8_ceiling" src/sous/engine/base.py
```

Expected:

```
missing: []
ceilings: 84481 170241
excluded: ('attention_tile_kv8.metal', 'kv8_quant.metal')
```

and both `grep -c` counts at least 1. A name in `missing` means another task named it differently: fix the documentation to the code's name (the interface sheet is the arbiter), not the code. Ceilings other than `84481 170241` mean `kv8.ceiling` departs from the inequality the docs quote: stop and reconcile with the task that wrote it before changing "about 84K" and "about 170K".

- [ ] **Step 10: Commit the documentation**

```bash
git add docs/tuning.md docs/configuration.md CLAUDE.md
git commit -m "docs: describe kv_int8, its floor and ceiling, and the kv8 engine rules" -m "configuration.md says what the setting changes (decode and verify
attention over an int8 copy, inside a generation only, with prefill and
everything stored left bf16), what the copy costs in memory and why that
caps it below the window instead of growing the reserve, and that it is
refused wherever the attention tile is. tuning.md adds it to the settings
every arm inherits and suite rows are keyed by.

CLAUDE.md gains the engine bullet: the window and its thread rules, the
mirror's validity through the trim wrap, the floor and the ceiling, what
the probe proves, the counters, and why the two kv8 kernels stay out of
the fork-store epoch, which the epoch sentence now says."
```

Expected: `git log -1 --format='%G? %an <%ae>%n%B'` shows `G`, the configured identity and no attribution lines.

---


### Task 9: The real-model test

**Files:**
- Create: `tests/test_kv8_model.py`

No existing file changes. The test is modelled on `tests/test_tileattn_model.py` (the engine context manager, `_corpus`, `_prompt`, the skip order and the "stock before, off after" pair) and `tests/test_projkernel_model.py` (blocks set on one engine, counter deltas per run).

**Interfaces:**
- Consumes:
  - The kv8 module (interface sheet): `kv8.availability() -> nax.Availability`; `kv8.ceiling(window: int, splits: int) -> int` (pure); the module constant `kv8.N8` and the module state `kv8._active` and `kv8._ceiling`; `kv8.calls` with the keys `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_rows`, `mirror_trims`, `mirror_grown`, `windows` and `windows_capped`.
  - Task 3: the seams. `serves(n_row)` is asked per call and per verify run and reads `kv8._ceiling` at call time, so a test may move the ceiling between two `generate()` calls. A call whose first row is at or past the ceiling caps the window (`windows_capped` once per window), and every later call of that decode runs on the bf16 tile.
  - Task 5: `VLMEngine(model_id, ..., draft_id=..., draft_block_size=..., attention_tile=..., projection_kernel=..., kv_int8: bool = False, kv_int8_window: int = 0)`; `VLMEngine.kv_int8_status` with the keys `state`, `reason`, `probe_seconds`, `floor` and `ceiling`; `VLMEngine.speculation() -> Speculation`, whose `kv8: str` and `kv8_rows: int` come from `kv8.close_window(cache)` after every decode (`"off"` and 0 when no window opened); `unload()` calls `kv8.clear()`.
  - The window states (interface sheet): `idle`, `served`, `capped`; `rows` counts the rows quantised in that decode, the first build included.
  - Existing, unchanged: `tileattn.availability()`, `tileattn.boundaries(n_max, splits)`, `tileattn.far_mark(splits)`, `tileattn.GRAN`, `tileattn.MEASURED_SPLITS`, `tileattn.calls`, `VLMEngine.attention_tile_status["splits"]`, `VLMEngine._draft_block_size`, `VLMEngine._encode`, `VLMEngine._tokenizer`, `VLMEngine.count_tokens`, `VLMEngine.generate`, `gpucores.matched_cores(device_name)`.
- Produces: one `-m model` test, `test_greedy_output_with_the_drafter_equals_output_without_it_on_kv8`. CI deselects it. It runs on the M5 Pro in the merge gate's step 1 (`uv run pytest -m model tests/test_kv8_model.py tests/test_tileattn_model.py`).

Facts the test is built on:
- An engine built without `prompt_cache=True` (the constructor's default) has its prompt cache off, and `cache_budget=0` skips the budget measurement, so every `generate()` is exactly one `decode()` over the whole prompt. That means one window per call (`windows` moves by exactly the number of calls), and `speculation()` after a call describes that call's decode. The decode's own prefill runs in multi-row chunks, which neither hook serves. The one-row forward that ends it goes through the decode hook at `n = prompt length`.
- The ceiling at the daemon's default window, 131072, with S = 20 is 84,481: `C = round_up(84481, 256) = 84736` gives `1.504 × 84736 + 0.504 × 84736 / 16 + 272 = 130,384 ≤ 131072`, and the next boundary, 85,121, gives `C = 85248` and 131,170, which does not fit. The test does not hard-code that number. It checks `kv8.ceiling()`'s answer against the inequality on both sides, so the test still holds if another S joins `MEASURED_SPLITS`.
- The floor `N8` may move above N0 in the gate's kernel step. The prompts are therefore placed relative to `kv8.N8` and to the step boundaries, never at fixed key counts:
  - `floor`: ends 86 tokens below N8, and its 520 and 521 generated tokens cross N8. The two caps end the last round at different row counts.
  - `far`: ends 40 tokens below the first step boundary at or past 57K (the tile probe's far mark), and its 160 tokens straddle that boundary. That is where kv8 earns its keep.
  - `cap`: ends 200 tokens below a small ceiling, and its 400 tokens cross that ceiling. The small ceiling is the first step boundary at least 256 keys above N8. With N8 = N0 = 1,536 and S = 20 that boundary is 1,921. The test sets it on `kv8._ceiling` for that prompt only. The prompt's own last row is above N8, so the decode is served and then capped.

  Every prompt's final key count is more than 16 rows from N8 and from the ceiling that applies to it. That is the condition under which the spec's gate requires drafter-on and drafter-off decodes to report the same `kv8=` state.
- Four engine loads, as in the tile test:
  1. kv8 off, drafter on: the reference, built before any kv8 engine has run in the process.
  2. kv8 on, drafter off.
  3. kv8 on, drafter on: at block 5 and at block 8 (set on the engine directly; config caps the block at 5).
  4. kv8 off, drafter on: after the kv8 engines have unloaded, it must reproduce engine 1 and make no kv8 call.

  Five cold prefills of the ~57.5K-token prompt dominate the run: about 35 minutes on the M5 Pro.
- All four engines keep the attention tile and the projection kernel on, as the daemon runs them. The block is 5, explicit. kv8 output is not compared with kv8-off output: the format changes numerics. That comparison is the merge gate's quality step, not this test's.

- [ ] **Step 1: Write the real-model test**

Create `tests/test_kv8_model.py`:

```python
"""kv8 on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 35 minutes, most of it four model loads and five
cold prefills of the ~57.5K-token prompt with the prompt cache off):
- With the tile active and the default 131072 window, an engine reports kv8 active, with its probe
  time printed, its floor at N8 and its ceiling the largest step boundary whose mirror fits
  beside the bf16 KV; the test skips, naming the reason, where kv8 cannot be active.
- Greedy output with the DFlash2 drafter equals output without it, at block 5 and at block 8, on
  a prompt whose generation crosses the floor, on one whose generation straddles the step
  boundary at or past 57K keys, and on one whose generation crosses a ceiling lowered for it.
- Every decode's gauges carry the window's state, with the rows it quantised: `served` on the
  first two prompts and `capped` on the third, the same with the drafter on and off; the
  counters show decode and verify served from the mirror, trims followed, one window per
  decode, one cap per capped decode, and nothing declined or out of scope on the tile.
- An engine with kv8 off, built after the kv8 engines unloaded, reproduces the output of one built
  before any kv8 engine ran in the process, reports `off` on every decode and makes no kv8 call.
"""

import contextlib
import math
from collections.abc import Iterator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores, kv8, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# The daemon's default [model].max_context_tokens: the window kv8's ceiling is sized against.
WINDOW = 131072
BLOCKS = (5, 8)
SLACK = 16
# The window state each prompt's decode must end in, drafter on or off.
EXPECTED = {"floor": "served", "far": "served", "cap": "capped"}


@contextlib.contextmanager
def _engine(*, kv_int8: bool, drafter: bool) -> Iterator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=5 if drafter else 0,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=kv_int8,
        kv_int8_window=WINDOW if kv_int8 else 0,
    )
    try:
        yield engine
    finally:
        engine.unload()


@contextlib.contextmanager
def _ceiling(line: int | None) -> Iterator[None]:
    """kv8's ceiling set to `line` for the calls inside, then put back; None leaves it alone.
    The seams read the ceiling on every call, so a loaded engine takes the new line at once."""
    saved = kv8._ceiling
    if line is not None:
        kv8._ceiling = line
    try:
        yield
    finally:
        kv8._ceiling = saved


def _fits(boundary: int) -> bool:
    """Whether a mirror ending at this boundary fits beside the turn's bf16 KV in WINDOW:
    the bf16 KV up to the capacity, the mirror at it, one layer's old mirror during a growth,
    and the rows a straddling verify may hold past the line plus one capacity step."""
    cap = math.ceil(boundary / 256) * 256
    return 1.504 * cap + 0.504 * cap / 16 + 272 <= WINDOW


def _targets(splits: int) -> dict[str, tuple[int, tuple[int, ...], int | None]]:
    """name -> (prompt tokens, max_tokens values, ceiling for these calls or None), placed
    against the floor and the step boundaries so the test holds wherever N8 sits."""
    reach = kv8.N8 + 256 + tileattn.GRAN * splits  # no step is wider than GRAN * splits keys
    line = next(b for b in tileattn.boundaries(reach, splits) if b >= kv8.N8 + 256)
    return {
        "floor": (kv8.N8 - 86, (520, 521), None),
        "far": (tileattn.far_mark(splits) - 40, (160,), None),
        "cap": (line - 200, (400,), line),
    }


def _corpus(engine, n):
    files = sorted((ROOT / "src" / "sous").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ids = engine._encode("".join(f.read_text() for f in files))
    assert len(ids) >= n, len(ids)
    return ids[:n]


def _prompt(engine, ids, target):
    """A one-message conversation whose rendered prompt is target tokens, within SLACK."""
    k, n, messages = target, 0, []
    for _ in range(8):
        messages = [{"role": "user", "content": HEAD + engine._tokenizer.decode(ids[:k])}]
        n = engine.count_tokens(messages, [])
        if abs(n - target) <= SLACK // 2:
            break
        k += target - n
    assert abs(n - target) <= SLACK, (target, n)
    return messages


def _generate(engine, prompts):
    """({(prompt, max_tokens): text}, {(prompt, max_tokens): (kv8 state, kv8 rows)},
    kv8 counter deltas, tile counter deltas) at the engine's current block."""
    kv8_before, tile_before = dict(kv8.calls), dict(tileattn.calls)
    texts, gauges = {}, {}
    for name, (messages, caps, line) in prompts.items():
        for cap in caps:
            with _ceiling(line):
                texts[(name, cap)] = engine.generate(messages, [], cap)
            spec = engine.speculation()
            gauges[(name, cap)] = (spec.kv8, spec.kv8_rows)
    kv8_counts = {k: v - kv8_before[k] for k, v in kv8.calls.items()}
    tile_counts = {k: v - tile_before[k] for k, v in tileattn.calls.items()}
    return texts, gauges, kv8_counts, tile_counts


def _check_status(status, splits):
    assert status["state"] == "active", status
    assert status["reason"] is None, status
    assert isinstance(status["probe_seconds"], float) and status["probe_seconds"] > 0, status
    assert status["floor"] == kv8.N8, status
    assert status["ceiling"] == kv8.ceiling(WINDOW, splits), status
    marks = tileattn.boundaries(WINDOW, splits)
    assert status["floor"] in marks and status["ceiling"] in marks, status
    assert kv8.N8 < status["ceiling"], status
    # The ceiling's memory rule on both sides: the ceiling fits, the next boundary does not.
    assert _fits(status["ceiling"]), status
    later = [b for b in marks if b > status["ceiling"]]
    assert not later or not _fits(later[0]), (status, later[:1])


def test_greedy_output_with_the_drafter_equals_output_without_it_on_kv8():
    avail = tileattn.availability()
    if not avail.available:
        pytest.skip(f"attention tile unavailable: {avail.reason}")
    cores, why = gpucores.matched_cores(str(mx.device_info()["device_name"]))
    if cores not in tileattn.MEASURED_SPLITS:
        pytest.skip(f"attention tile unavailable: {why or f'split target {cores} not measured'}")
    kernel = kv8.availability()
    if not kernel.available:
        pytest.skip(f"kv8 unavailable: {kernel.reason}")

    # First in the process, before any kv8 engine could open a window: the bf16 tile's paths.
    with _engine(kv_int8=False, drafter=True) as stock:
        assert stock.kv_int8_status["state"] == "off", stock.kv_int8_status
        assert stock.attention_tile_status["state"] == "active", stock.attention_tile_status
        splits = stock.attention_tile_status["splits"]
        targets = _targets(splits)
        ids = _corpus(stock, max(target for target, _, _ in targets.values()) + 64)
        prompts = {
            name: (_prompt(stock, ids, target), caps, line)
            for name, (target, caps, line) in targets.items()
        }
        reference, reference_gauges, _, _ = _generate(stock, prompts)

    with _engine(kv_int8=True, drafter=False) as plain:
        status = plain.kv_int8_status
        if status["state"] != "active":
            pytest.skip(f"kv8 {status['state']}: {status['reason']}")
        print(
            f"kv8 probe {status['probe_seconds']:.2f} s,"
            f" floor {status['floor']}, ceiling {status['ceiling']} at window {WINDOW}"
        )
        _check_status(status, splits)
        far_end = targets["far"][0] + max(targets["far"][1])
        assert far_end + SLACK < status["ceiling"], (far_end, status)
        undrafted, plain_gauges, plain_counts, plain_tile = _generate(plain, prompts)

    drafted, drafted_gauges, drafted_counts, drafted_tile = {}, {}, {}, {}
    with _engine(kv_int8=True, drafter=True) as engine:
        _check_status(engine.kv_int8_status, splits)
        assert engine._draft_block_size == 5
        for block in BLOCKS:
            engine._draft_block_size = block
            (
                drafted[block],
                drafted_gauges[block],
                drafted_counts[block],
                drafted_tile[block],
            ) = _generate(engine, prompts)

    with _engine(kv_int8=False, drafter=True) as off:
        assert off.kv_int8_status["state"] == "off", off.kv_int8_status
        assert kv8._active == 0
        again, again_gauges, off_counts, _ = _generate(off, prompts)

    # Parity: greedy output is the same with the drafter on and off, at every block.
    for block in BLOCKS:
        for key, text in undrafted.items():
            assert drafted[block][key] == text, (f"block {block}", *key)
    assert again == reference

    # The gauges: each decode's window state and the rows it quantised. A served or capped
    # window first built its mirror over the whole prefix, which is at least N8 rows.
    for key, (state, rows) in plain_gauges.items():
        assert state == EXPECTED[key[0]], (key, state, rows)
        assert rows >= kv8.N8, (key, state, rows)
    for block in BLOCKS:
        for key, (state, rows) in drafted_gauges[block].items():
            assert state == plain_gauges[key][0], (f"block {block}", key, state)
            assert rows >= kv8.N8, (f"block {block}", key, state, rows)
    for gauges in (reference_gauges, again_gauges):
        assert set(gauges.values()) == {("off", 0)}, gauges

    # The counters. One decode per call (the prompt cache is off), so one window per call, and
    # one cap for the one capped prompt per pass.
    calls = sum(len(caps) for _, caps, _ in prompts.values())
    capped = sum(len(caps) for name, (_, caps, _) in prompts.items() if EXPECTED[name] == "capped")
    assert plain_counts["windows"] == calls, plain_counts
    assert plain_counts["windows_capped"] == capped, plain_counts
    assert plain_counts["decode_kv8"] > 0, plain_counts
    assert plain_counts["mirror_builds"] > 0, plain_counts
    assert plain_counts["mirror_rows"] > 0, plain_counts
    assert plain_counts["verify_kv8"] == 0, plain_counts
    for block, counts in drafted_counts.items():
        assert counts["windows"] == calls, (block, counts)
        assert counts["windows_capped"] == capped, (block, counts)
        assert counts["decode_kv8"] > 0, (block, counts)
        assert counts["verify_kv8"] > 0, (block, counts)
        assert counts["verify_rows_kv8"] > 0, (block, counts)
        assert counts["verify_launches_kv8"] > 0, (block, counts)
        assert counts["mirror_trims"] > 0, (block, counts)
    assert drafted_tile[8]["verify_t8"] > 0, drafted_tile
    assert drafted_tile[8]["verify_straddles"] > 0, drafted_tile
    for counts in (plain_tile, *drafted_tile.values()):
        assert counts["decode_out"] == counts["verify_out"] == counts["verify_declined"] == 0
    assert not any(off_counts.values()), off_counts
```

Why each check holds:
- **`served` on `floor` and `far`.** `floor`'s decode crosses N8 below the real ceiling. `far` sits wholly between N8 and the real ceiling, which the test asserts (`far_end + SLACK < ceiling`). Its prefill's closing one-row forward is the first served call.
- **`capped` on `cap`.** Its prompt ends at least 56 keys above N8, so the closing one-row forward is served and builds the mirror. Once the committed prefix reaches the lowered ceiling, the next call's first row is at or past it. That call caps the window, exactly once.
- **`rows >= N8`.** The first served call quantises the whole prefix, from row 0 to that call's key count. That count is at least N8. A drafter-on decode may quantise more rows: its speculative rows run up to 16 past the committed length, and the rows a trim replaced are quantised again. Because of that, rows are bounded below and never compared across the two arms. The state is compared, and it is a function of committed key counts only.
- **`decode_kv8 > 0` with the drafter on.** `far`'s prefill ends in a one-row forward above N8. That forward is plain, not a verify, so it goes through the decode hook.
- **`mirror_trims > 0`.** A DFlash round that rejects part of its block trims the cache through the instance `trim`. Greedy acceptance on this model is about 0.84, so some of the ~2,300 tokens a pass generates are rejected.
- **`verify_t8` and `verify_straddles` at block 8.** Full block-8 rounds verify eight rows. `far`'s generation crosses a step boundary, so one of its rounds is cut into two runs. Both counters are the tile's, and `serve_verify` keeps counting them for every call kv8 serves.
- **No kv8 call on the last engine.** `unload()` cleared `_active`, and the off engine's `enable()` returned `off` before any gate. No window opens, so `serves` answers False everywhere and `close_window` never runs.

- [ ] **Step 2: Run the test here to show it collects and skips**

This test needs no implementation step. It pins behaviour that Tasks 1–7 built, and it can run only on the M5 Pro.

Run: `uv run pytest -m model tests/test_kv8_model.py -v -rs`

Expected on the maintainer's M2: `1 skipped`, with the reason `attention tile unavailable: macOS 15.5 < 26.2`. No model loads and nothing is fetched from the Hub, because the tile's platform rule refuses first.

Run: `uv run pytest -m "not model" tests/test_kv8_model.py`

Expected: `1 deselected`, which is how CI treats it.

On the M5 Pro the merge gate's step 1 runs it with `uv run pytest -m model tests/test_kv8_model.py tests/test_tileattn_model.py -rP`. `-rP` prints the passing test's captured output, which includes the line `kv8 probe <s> s, floor <N8>, ceiling 84481 at window 131072` at S = 20; record that line in the gate's report. The test must pass there. If an assertion fails, the bug is in Tasks 1–7: fix it there and never loosen the assertion. Three failures have known readings:
- **A text mismatch.** Before blaming kv8, find the first differing token and read the drafter-off run's top two logprobs at that token. A tie there is the argmax residual the projection kernel's notes describe (drafter-off greedy argmaxes `logits − logsumexp`, greedy verify argmaxes raw logits), not a kv8 fault. A mismatch with no tie at the first differing token means a verify row and its one-row call read different codes: re-run `uv run pytest tests/test_kv8.py tests/test_kv8_contract.py` on the M5 with the real kernel.
- **A `kv8` state that differs between the arms.** Report both arms' gauges and the prompt's final key count. The prompts are placed more than 16 rows from both lines, so a difference means the cap is not sticky or depends on uncommitted rows (Task 3's `observe_first_row` order).
- **`kv8 unavailable` or `kv8 <state>: <reason>` as the skip reason on the M5.** That is a load-time refusal, and the test cannot pass while it stands. Read the daemon-style warning that `enable()` printed (`warnings.warn` on a compile, probe or pin failure) and fix the cause.

- [ ] **Step 3: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model" tests/test_kv8_model.py
git add tests/test_kv8_model.py
git commit -m "test(engine): pin kv8's greedy parity and gauges on the real model" -m "The unit and contract tests prove the mirror, the seams and the kernel on
the tiny model and the stand-in; this holds the real 27B with DFlash2 to
the properties the design rests on. With kv8 active at the default
window, greedy output with the drafter equals output without it at
blocks 5 and 8, on a prompt that crosses the floor, one that straddles
the step boundary past 57K keys, and one that crosses a lowered ceiling.
Every decode reports the same window state on both arms, served or
capped, and an engine with kv8 off, built after kv8 engines ran, matches
one built before them and makes no kv8 call. It also prints the probe's
load-time cost and checks the floor and the ceiling the load reports
against the memory rule they come from."
```

Expected: ruff and ty report nothing, and pytest reports `1 deselected`.


### Task 10: The M5 Pro merge gate and the PR

This task is manual measurement, not code. Nothing it writes out of tree is ever committed. It runs the spec's merge gate (pre-registered 2026-10-12) exactly: preconditions, steps 1–6, the verdict, and the PR only after a passing verdict.

**Where it runs.**
- The M5 Pro over ssh, as `ssh -o BatchMode=yes m5pro '<command>'`, and rsync with `-e 'ssh -o BatchMode=yes'`.
- This Mac (the M2 Air) for gate step 1's M2 half, the CPU checks, the harness's authoring and the PR.

**What it never touches:**
- `~/code/personal/sous`, `~/.sous` and the maintainer's daemon on either Mac;
- `~/code/personal/sous-remote`'s checkout and environment on the M5: only `git fetch` and `git worktree add` run there.

**Trees on the M5** (the relay gate's worktree pattern):
- main runs from the detached worktree `~/code/personal/sous-remote-main` at origin/main;
- the branch runs from `~/code/personal/sous-remote-kv8` at the branch head.

Each has its own `.venv` from `uv sync`. The gate never checks out or syncs `~/code/personal/sous-remote` itself.

**GPU lock.**
- Every M5 GPU job runs under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py` with a `GPU_LABEL` and a hard timeout (`perl -e 'alarm shift @ARGV; exec @ARGV' <seconds> ...`). It waits for the production daemon to be idle and prints `[gpu_run_m5] daemon stayed idle during the job`, or `DAEMON-CONTAMINATED`, then `[gpu_run_m5] exit=<rc>`.
- Every GPU job on this Mac runs under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py`.
- A run counts only if its log says the daemon stayed idle and `exit=0`. Re-run any other.

**Production `max_tokens`.** Claude Code posts `max_tokens = 32000` on every `sous-local` subagent turn (its 32K output cap, which the spec's decision 6 sizes against). The replay set's builder posts the same value (`acceptance-131/build_replay.py`, checked in Step 2), and the transcripts record no other. Steps 3 and 4 replay every turn at 32000. The daemon still caps a generation at `window − input`.

**Time.** About 21 hours of M5 GPU time:

| Part | Gate step | Time |
|---|---|---|
| Non-model suite | 1 | 15 min |
| Model tests (two processes) | 1 | 1.2 h |
| Kernel timings | 2 | 40 min |
| Per-round runs (two seeds) | 3 | 2 h |
| ABBA (four runs) | 4 | 3.5 h |
| Parity and off-equals-main (four runs) | 5 | 4 h |
| Suite (four jobs) | 6(a) | 3.5 h |
| KL instrument | 6(b) | 6 h |

Step 1's M2 half takes about 20 minutes of this Mac's GPU.

**On any failure, stop and report the numbers. Open no PR.** These outcomes have a set response beyond that:
- **(5, 57K) under 1.35×** (Step 15). The spec's fallbacks run in their pre-registered order: (a) a two-pass Q prologue, (b) host-quantised Q as the spike timed it, (c) separate K and V planes. Each is a kernel change to the branch for the maintainer to approve. Report Step 15's table. After one lands, the gate restarts at Step 13 (the harness sync) and re-times.
- **The floor moves** (Step 15 prints `floor: N8 moves to <b>`). Raising `kv8.N8` is a branch change: the constant, and the tests and docs that state 1,536 as the floor. It goes to the maintainer with the round table. The gate restarts at Step 13 once it lands.
- **Step 3 says CLOSE.** The lever closes as the spec's rule says. Report `logs/step3.txt`; nothing later runs.
- **Step 6(a) says KEEP OFF.** `kv_int8` does not ship on by default. Report `logs/suite.txt`; the group-64 scale fallback (out of scope in the spec) is the maintainer's next question.
- **TO THE MAINTAINER** from Step 3, Step 6(a) or Step 6(b)'s tripwire: stop with that step's file.
- **A parity divergence** (Step 5): read `logs/gap.txt` before anything is blamed. One divergence still fails the gate.

**Files:**
- Out of tree, never committed:
  - Create on this Mac, in `~/code/sous-spikes/kv8-gate/` (beside the existing `suite_kv8.py` and `analyze_suite_kv8.py`), then rsync to `m5pro:code/personal/sous-spikes/kv8-gate/`:
    - `tests_m5.sh`, gate step 1 on the M5;
    - `ktime_kv8.py`, gate step 2;
    - `rounds_kv8.py` and `wrap_kv8.py`, the per-round shim and the GATE_WRAP daemon wrapper;
    - `run_kv8.sh` and `chain_kv8.sh`, the replay runs of gate steps 3–5;
    - `analyze_kv8.py`, gate step 3's analysis;
    - `judge_kv8.py`, the judgements of gate steps 3–5;
    - `kl_kv8.py` and `quality_m5.sh`, gate step 6;
    - `judge_all.sh`, every analysis.
  - Modify on this Mac: `~/code/sous-spikes/kv8-gate/suite_kv8.py` lines 42–43, so each suite arm's `Arm` carries `kv_int8` (Step 11).
  - Copy on the M5 into `~/code/personal/sous-spikes/kv8-gate/` (Step 13), so every run lands in that directory's `runs/`:
    - from `~/code/personal/sous-spikes/tile-132/gate/`: `gate132.py`, `analyze132.py`, `analyze133.py`;
    - from `~/code/personal/sous-spikes/p157/`: `gate157.py`, `build_replay.py`, `replay-157.json`, `analyze157.py`.
  - M5 trees: `~/code/personal/sous-remote-main` and `~/code/personal/sous-remote-kv8`, detached worktrees of `~/code/personal/sous-remote`.
- In-tree, only after a passing gate, and only if Step 21's figures say so: `README.md` line 56 (`and about 25 tok/s at 57K on an M5 Pro, sampling as shipped. It prefills`) and line 150 (`estimated 1.2–1.5x at short context and 1.3–1.5x at 51–57K. It drafts and`).

**Interfaces:**
- Consumes:
  - Task 1: `kv8.quantize(k, v, out_rows) -> (codes, sc)` over bf16 views `[1, 4, R, 256]`; `kv8.decode_codes(queries, codes, sc, n, splits) -> [1, 24, 1, 256]`; `kv8.verify_codes(queries, codes, sc, n, splits, runs=None) -> [1, 24, T, 256]`; `kv8.availability() -> nax.Availability` (`.available`, `.reason`); the constants `kv8.N8` (1536), `kv8.GROWTH` (4096), `kv8.BYTES_PER_ROW_PER_HEAD` (516).
  - Task 2: `kv8.open_window(*, force=False)`, `kv8.close_window(cache) -> (state, rows)`, `kv8.serves(n_row) -> bool`, `kv8.calls` (`decode_kv8`, `verify_kv8`, `verify_rows_kv8`, …); the mirror follows `KVCache.trim` through its instance wrap; a forced window has no ceiling.
  - Task 3: `tileattn._serve_decode` and `tileattn.serve_verify` import `kv8` function-locally and look up `kv8.serves`, `kv8.decode` and `kv8.observe_first_row` through the module on every call, so a module-level replacement reaches both seams; an off verify run moves no `kv8.calls` counter.
  - The interface sheet: `kv8.enable(model, *, enabled, tile_status, verify_status, window) -> dict`, which `VLMEngine.__init__` calls through the module attribute (`kv8.enable(self._model, ...)`), so the wrapper times it by replacing that attribute.
  - Task 5: `VLMEngine(..., attention_tile=..., projection_kernel=..., kv_int8: bool = False, kv_int8_window: int = 0)`; `VLMEngine.kv_int8_status` (`state`, `reason`, `probe_seconds`, `floor`, `ceiling`); `[model].kv_int8` in config.toml (`true` by default); `SousConfig.kv_int8`.
  - Task 7: the load-line tokens `kv_int8=`, `kv_int8_probe_s=`, `kv_int8_floor=`, `kv_int8_ceiling=` after `draft_block=`; the turn-line tokens `kv8=<off|idle|served|capped>` and `kv8_rows=<n>` after `draft_context=`.
  - Task 8: `Arm.kv_int8: bool = True`.
  - Task 9: `tests/test_kv8_model.py`.
  - Existing, unchanged: `tileattn.decode(q, k, v, splits)`, `tileattn.verify(q, k, v, splits)`, `tileattn.boundaries(n_max, splits)`, `tileattn.N0`, `tileattn.GRAN`, `tileattn.HQ/HKV/D`; `VLMEngine._ids(slot, messages, tools)`, `new_cache()`, `prefill(cache, ids)`, `_positions(cache, n)`, `_model`, `_tokenizer`, `_draft`, `_draft_id`, `_draft_block_size`, `_sampler`, `projection_kernel_status`, `attention_tile_status`, `unload()`; `sous.api.convert.parse_messages_request(body).messages/.tools`; mlx-vlm 0.7.2's `KVCache.trim(n)` and `ArraysCache.cache` (a list the recurrent layers assign into).
  - Out of tree:
    - `gate132.py` and `gate157.py`: `<label> <tree> <temperature> <max_tokens> [--model 'key = value']... [picks...]`, picks as `basename`, `basename:n` or `basename@i,j`; default picks #131's 64 turns; `GATE_WRAP`, `TILE132_COUNTERS`; an isolated HOME, `max_context_tokens = 262144`, port 8384; results in `runs/<label>/result.json` beside `daemon.out`, each result's `line` the parsed turn line and `tile` the wrapper's counters; the run's forks deleted on exit.
    - `analyze132.py <runs> equiv <a> <b>`, `analyze133.py <runs> runs|pair ...`, and `analyze157.py`'s module-level `REF`, `B`, `load`, `gap_fit`, `strat_parts`, `strat_metrics`, `ci` and `main`.
    - `suite_kv8.py <out.jsonl> <block>` and `analyze_suite_kv8.py <rows.jsonl>` (prints `rule: KEEP OFF|ELIGIBLE|TO THE MAINTAINER`).
    - `gpu_run_m5.py`, `detach.py <log> <cmd...>`, `wait_for.py <log> [timeout_s] [tail_lines]` (prints `DONE` once the log has `[gpu_run_m5] exit=`), and on this Mac `gpu_run.py`.
    - `acceptance-131/build_replay.py` and `replay-all.json`.
- Produces:
  - the gate verdict and its tables;
  - the PR, only after a passing verdict;
  - optionally, a `docs:` commit with the README's 57K figures.

- [ ] **Step 1: Rebase onto main with the relay in it, run step 1's M2 half, and push**

The gate's precondition is main at or after the relay merge (#181). This branch was cut on the relay's own commit, 860ba80.

```bash
gh pr view 181 --json state -q .state
git fetch -q origin
git log --oneline -1 origin/main --grep="split the attention tile's QK"
```

Expected: `MERGED`, then one line naming the relay's squash commit on main. If either is missing, stop: the precondition is unmet.

Replay only this branch's own commits onto main:

```bash
if git merge-base --is-ancestor 860ba80 HEAD; then git rebase --onto origin/main 860ba80; else git rebase origin/main; fi
```

A conflict means #181 changed in review after this branch was cut. Stop and report: every kernel number in this gate assumes the relay as merged.

Run gate step 1's M2 half, the non-model suite under this Mac's GPU lock, then the CPU checks:

```bash
mkdir -p ~/code/sous-spikes/kv8-gate/logs
GPU_LABEL=kv8-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py uv run pytest -m "not model" -rs > ~/code/sous-spikes/kv8-gate/logs/m2-notmodel.txt 2>&1
grep -E "[0-9]+ passed" ~/code/sous-spikes/kv8-gate/logs/m2-notmodel.txt | tail -n 1
uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check && echo "CPU CHECKS OK"
```

Expected:
- a summary line with passes, skips and deselections only, no `failed` or `error`. kv8's `nax` tests skip here with the platform reason (`macOS 15.5 < 26.2`);
- `CPU CHECKS OK`.

Push and note the head:

```bash
git push --force-with-lease -u origin feat/int8-kv-mirror
git rev-parse --short HEAD
```

- [ ] **Step 2: Check the M5's inputs**

```bash
ssh -o BatchMode=yes m5pro 'S=~/code/personal/sous-spikes; ls $S/acceptance-131/replay-all.json $S/acceptance-131/build_replay.py $S/tile-132/gate/gate132.py $S/tile-132/gate/analyze132.py $S/tile-132/gate/analyze133.py $S/p157/gate157.py $S/p157/build_replay.py $S/p157/replay-157.json $S/p157/analyze157.py $S/splash-port-2026-09-24/gpu_run_m5.py $S/splash-port-2026-09-24/detach.py $S/splash-port-2026-09-24/wait_for.py | wc -l; grep -c "\"max_tokens\": 32000" $S/acceptance-131/build_replay.py; sw_vers -productVersion'
```

Expected:
- `12`: all twelve files are listed;
- `1`: the replay set's builder posts the production `max_tokens`, 32000;
- the M5's macOS version (27.0 when this plan was written).

If anything is missing, stop and report.

- [ ] **Step 3: Prepare the M5 trees**

```bash
ssh -o BatchMode=yes m5pro 'cd ~/code/personal/sous-remote && git fetch -q origin && for spec in main:origin/main kv8:origin/feat/int8-kv-mirror; do w=$HOME/code/personal/sous-remote-${spec%%:*}; if [ -d $w ]; then git -C $w checkout -q --detach ${spec#*:}; else git worktree add -q --detach $w ${spec#*:}; fi && (cd $w && uv sync -q && echo "${w##*/}: $(git log -1 --oneline)") || exit 1; done'
```

Expected:
- `sous-remote-main:` followed by origin/main's head (the relay's squash commit or later);
- `sous-remote-kv8:` followed by the head Step 1 printed.

- [ ] **Step 4: Write gate step 1's M5 script**

Create `~/code/sous-spikes/kv8-gate/tests_m5.sh`:

```zsh
#!/bin/zsh
# Gate step 1 on the M5 Pro (out of tree, never committed): the non-model suite, then each model
# test file in a pytest process of its own, every one a GPU-lock job with a hard timeout. A model
# test's reference engine must be the first engine its process builds: in one process the kv8
# test's engines would install the tile's decode hook before the tile test's reference ran.
G=~/code/personal/sous-spikes/kv8-gate
L=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
cd ~/code/personal/sous-remote-kv8 || exit 1
mkdir -p $G/logs
rm -f $G/logs/TESTS
echo "tree $(git rev-parse --short HEAD) $(date +%H:%M:%S)" >> $G/logs/TESTS
env GPU_LABEL=kv8-tests python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 3600 \
  uv run pytest -m "not model" -rs > $G/logs/notmodel.log 2>&1
echo "notmodel exit=$? $(date +%H:%M:%S)" >> $G/logs/TESTS
for t in kv8 tileattn; do
  env GPU_LABEL=kv8-tests python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 5400 \
    uv run pytest -m model -rs tests/test_${t}_model.py > $G/logs/model-$t.log 2>&1
  echo "model-$t exit=$? $(date +%H:%M:%S)" >> $G/logs/TESTS
done
echo "TESTS DONE $(date +%H:%M:%S)" >> $G/logs/TESTS
```

- [ ] **Step 5: Write the kernel timings**

The cells and bars are the spec's gate step 2. The kv8 kernel and the relay tile both come from the branch tree, which carries the relay since Step 1. The chain, the buffers and the sampling order are `relay-gate/ktime_relay.py`'s.

Create `~/code/sous-spikes/kv8-gate/ktime_kv8.py`:

```python
"""Kernel timings for the kv8 merge gate (out of tree, never committed; from
relay-gate/ktime_relay.py): the int8 K/V kernel (kv8.decode_codes and kv8.verify_codes over a
packed mirror, Q quantised in the kernel) against the relay tile (tileattn.decode and
tileattn.verify over bf16 K/V), both from the branch tree, on the tile's ktime chain: 16
dependent layers (each layer's output is the next layer's queries) over distinct buffers, as
many as 6 GB of bf16 K/V holds, cycled; variants sampled round-robin with the order rotated
every round and reversed on odd rounds; medians; mx.synchronize around every sample.

Three kinds of cell, ms per 16-layer chain:
  kernel  T = 1..8 at 57K; T = 1, 3, 5, 6, 8 at 44K, 80K and 131K; T = 1 and 5 at N0, 4K, 8K
          and 16K. r = relay / kv8. Bars: r >= 1.35 at (5, 57K), (1, 57K) and (5, 80K); r >= 0.97
          at every other cell from 44K up; below 44K reported.
  round   per context, T = 5: each layer quantises the 5 rows the round appended (one quantiser
          dispatch), writes them into its mirror (two slice updates) and runs the kernel, which
          is what a served block-5 round pays. net = relay(T = 5) - round must be >= 0 wherever
          kv8 serves; the floor N8 is read from these cells.
  build   per context: a mirror's first build, one quantiser dispatch per layer writing the
          context's rows into a mirror of capacity round_up(context + GROWTH, 256), reported.
Every kernel and round cell carries an A/A pair of its kv8 variant; a gap past 1 % marks the
cell NOISY, and a NOISY run is re-timed whole.

usage: ktime_kv8.py [CTX,...]   (default every context above)
ends with `ktime PASS`, `ktime FAIL <cells>` or `ktime NOISY <cells>`, then one `floor:` line.
"""

import math
import statistics
import sys
import time

import mlx.core as mx

from sous.engine import kv8, tileattn

HQ, HKV, D, LAYERS = tileattn.HQ, tileattn.HKV, tileattn.D, 16
SPLITS = 20
WIDE = (1, 3, 5, 6, 8)
CELLS = {
    tileattn.N0: (1, 5),
    4096: (1, 5),
    8192: (1, 5),
    16384: (1, 5),
    44000: WIDE,
    57000: (1, 2, 3, 4, 5, 6, 7, 8),
    80000: WIDE,
    131072: WIDE,
}
BARS = {(5, 57000): 1.35, (1, 57000): 1.35, (5, 80000): 1.35}
FLOOR_CTX, FLOOR_R = 44000, 0.97
AA_MAX = 0.01
ROUND_T = 5


def round_up(x, m):
    return -(-x // m) * m


class Mirror:
    """One buffer pair's int8 copy, referenced from here only, so mlx donates its arrays to the
    round cell's slice updates and they write in place, as the engine's mirror does."""

    def __init__(self, codes, sc):
        self.codes, self.sc = codes, sc


def make_layers(n_max, budget_gb=6.0, seed=0):
    cap = round_up(n_max, 256)
    if cap == n_max:
        cap += 256
    per = 2 * HKV * cap * D * 2 / 1e9
    distinct = max(1, min(LAYERS, int(budget_gb // per)))
    bufs, mirrors = [], []
    for i in range(distinct):
        k1, k2 = mx.random.split(mx.random.key(seed + 1000 + 17 * i + n_max))
        kb = mx.random.normal((1, HKV, cap, D), key=k1).astype(mx.bfloat16)
        vb = mx.random.normal((1, HKV, cap, D), key=k2).astype(mx.bfloat16)
        codes, sc = kv8.quantize(kb, vb, cap)
        mx.eval(kb, vb, codes, sc)
        bufs.append((kb, vb))
        mirrors.append(Mirror(codes, sc))
    return bufs, mirrors


def views_for(bufs, n):
    out = [(kb[:, :, :n, :], vb[:, :, :n, :]) for kb, vb in (bufs[i % len(bufs)] for i in range(LAYERS))]
    mx.eval([a for kv in out for a in kv])
    return out


def queries(t):
    q = mx.random.normal((1, t, HQ, D), key=mx.random.key(77)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    mx.eval(q)
    return q


def relay_chain(q0, views):
    q = q0
    for k, v in views:
        q = tileattn.decode(q, k, v, SPLITS) if q.shape[2] == 1 else tileattn.verify(q, k, v, SPLITS)
    return q


def kv8_chain(q0, mirrors, n):
    q = q0
    for i in range(LAYERS):
        m = mirrors[i % len(mirrors)]
        if q.shape[2] == 1:
            q = kv8.decode_codes(q, m.codes, m.sc, n, SPLITS)
        else:
            q = kv8.verify_codes(q, m.codes, m.sc, n, SPLITS)
    return q


def round_chain(q0, bufs, mirrors, n, t):
    q = q0
    a = n - t
    for i in range(LAYERS):
        kb, vb = bufs[i % len(bufs)]
        m = mirrors[i % len(mirrors)]
        codes, sc = kv8.quantize(kb[:, :, a:n, :], vb[:, :, a:n, :], t)
        m.codes[:, :, a:n, :] = codes
        m.sc[:, :, a:n, :] = sc
        q = kv8.verify_codes(q, m.codes, m.sc, n, SPLITS)
    return q


def build_all(bufs, ctx):
    cap = round_up(ctx + kv8.GROWTH, 256)
    outs = []
    for i in range(LAYERS):
        kb, vb = bufs[i % len(bufs)]
        outs.extend(kv8.quantize(kb[:, :, :ctx, :], vb[:, :, :ctx, :], cap))
    return outs


def run_block(variants, min_ms=300.0, min_reps=7, max_reps=41):
    names = list(variants)
    est = {}
    for name in names:
        for _ in range(2):
            mx.eval(variants[name]())
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(variants[name]())
        mx.synchronize()
        est[name] = (time.perf_counter() - t0) * 1e3
    reps = int(min(max_reps, max(min_reps, math.ceil(min_ms / max(0.05, max(est.values()))))))
    tot = {k: [] for k in names}
    for r in range(reps):
        order = names[r % len(names) :] + names[: r % len(names)]
        if r % 2:
            order = order[::-1]
        for name in order:
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(variants[name]())
            mx.synchronize()
            tot[name].append((time.perf_counter() - t0) * 1e3)
    return {k: {"med": statistics.median(v), "reps": reps, "samples": v} for k, v in tot.items()}


def aa_gap(res, a, b):
    return statistics.median(x / y for x, y in zip(res[a]["samples"], res[b]["samples"], strict=True)) - 1


def rel_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w * w)))


def floor_line(nets):
    """nets: (context, net ms) for every round cell, ascending by context."""
    from_ctx = None
    for ctx, net in reversed(nets):
        if net < 0:
            break
        from_ctx = ctx
    if from_ctx is None:
        return "floor: kv8 does not pay at the largest context timed (net < 0 there)"
    if from_ctx == nets[0][0]:
        return f"floor: N8 stays {kv8.N8} (net >= 0 at every context timed)"
    step = tileattn.GRAN * SPLITS
    b = next(x for x in tileattn.boundaries(from_ctx + step, SPLITS) if x >= from_ctx)
    return f"floor: N8 moves to {b} (net >= 0 from {from_ctx} keys up, < 0 below)"


def main():
    ctxs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else sorted(CELLS)
    unknown = [c for c in ctxs if c not in CELLS]
    if unknown:
        sys.exit(f"no cells at {unknown}; the contexts are {sorted(CELLS)}")
    avail = kv8.availability()
    if not avail.available:
        sys.exit(f"kv8 unavailable: {avail.reason}")
    print(f"device {mx.device_info().get('architecture')} mlx {mx.__version__} S={SPLITS} N8={kv8.N8}", flush=True)
    failed, noisy, nets = [], [], []
    for ctx in ctxs:
        ts = CELLS[ctx]
        bufs, mirrors = make_layers(ctx + max(ts))
        for t in ts:
            n = ctx + t
            views = views_for(bufs, n)
            q0 = queries(t)
            a, b = relay_chain(q0, views), kv8_chain(q0, mirrors, n)
            mx.eval(a, b)
            rel = rel_rms(b, a)
            res = run_block(
                {
                    "relay": lambda: relay_chain(q0, views),
                    "kv8": lambda: kv8_chain(q0, mirrors, n),
                    "kv8#2": lambda: kv8_chain(q0, mirrors, n),
                }
            )
            aa = aa_gap(res, "kv8", "kv8#2")
            r = res["relay"]["med"] / res["kv8"]["med"]
            need = BARS.get((t, ctx), FLOOR_R if ctx >= FLOOR_CTX else None)
            if abs(aa) > AA_MAX:
                mark = "NOISY"
                noisy.append(f"kernel({t},{ctx})")
            elif need is None:
                mark = "reported"
            elif r >= need:
                mark = "ok"
            else:
                mark = "FAIL"
                failed.append(f"kernel({t},{ctx})")
            bar = f" bar {need:.2f}" if need is not None else ""
            print(
                f"kernel ctx={ctx} T={t} n={n} bufs={len(bufs)} relay {res['relay']['med']:.3f} ms  "
                f"kv8 {res['kv8']['med']:.3f} ms (A/A {aa * 100:+.2f}%, reps {res['kv8']['reps']})  "
                f"r {r:.3f}{bar}  relrms {rel:.2e}  {mark}",
                flush=True,
            )
            views = None
        t = ROUND_T
        n = ctx + t
        views = views_for(bufs, n)
        q0 = queries(t)
        res = run_block(
            {
                "relay": lambda: relay_chain(q0, views),
                "round": lambda: round_chain(q0, bufs, mirrors, n, t),
                "round#2": lambda: round_chain(q0, bufs, mirrors, n, t),
            }
        )
        aa = aa_gap(res, "round", "round#2")
        net = res["relay"]["med"] - res["round"]["med"]
        if abs(aa) > AA_MAX:
            mark = "NOISY"
            noisy.append(f"round({ctx})")
        else:
            mark = "ok" if net >= 0 else "net<0"
        nets.append((ctx, net))
        print(
            f"round  ctx={ctx} T={t} relay {res['relay']['med']:.3f} ms  round {res['round']['med']:.3f} ms "
            f"(A/A {aa * 100:+.2f}%, reps {res['round']['reps']})  net {net:+.3f} ms  {mark}",
            flush=True,
        )
        views = None
        res = run_block({"build": lambda: build_all(bufs, ctx)}, min_reps=5)
        ms = res["build"]["med"]
        # Per row and head: 1,024 B of bf16 K and V read, 512 B of codes and 4 B of scales written.
        gb = LAYERS * HKV * ctx * (2 * D * 2 + 2 * D + 4) / 1e9
        print(
            f"build  ctx={ctx} rows={ctx} cap={round_up(ctx + kv8.GROWTH, 256)} {ms:.2f} ms  "
            f"{gb / ms * 1e3:.0f} GB/s",
            flush=True,
        )
        bufs = mirrors = None
        mx.clear_cache()
    if noisy:
        print("ktime NOISY " + " ".join(noisy), flush=True)
        code = 2
    elif failed:
        print("ktime FAIL " + " ".join(failed), flush=True)
        code = 1
    else:
        print("ktime PASS", flush=True)
        code = 0
    print(floor_line(nets) if ctxs == sorted(CELLS) else "floor: not decided (not every context was timed)")
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the per-round shim and the daemon wrapper**

The spec's per-round arms, as `p157/rounds_gate.py` drew blocks, with the draw switching kv8 instead of the block:
- A round's draft and verify run between this round's `_dflash_next_block_size` and the next one's, so the arm drawn there covers exactly that round. This is the same boundary as the spec's wrap of the drafter's draft call, without stacking a third wrap on draftctx's instance wraps.
- `begin_decode()` turns kv8 on for the start of every decode. The engine opens the window as it does in production, and the decode's own prefill segment builds the mirror inside the turn's time to first token.
- Off rounds leave the mirror and its trim wrap in place, so the next on round re-quantises the backlog and an off round pays nothing for kv8.
- The window's serial is never touched.

The mlx-vlm source pins are `rounds_gate.py`'s, computed on mlx-vlm 0.7.2 and unchanged since #157 ran.

Create `~/code/sous-spikes/kv8-gate/rounds_kv8.py`:

```python
"""Per-round DFlash records with kv8 drawn on or off per round (out of tree, never committed;
from p157/rounds_gate.py, whose block mechanics, timing points and record fields it keeps, so
p157/analyze157.py reads its records as they are).

An arm is "<block><b|q>"; the gate draws 5b and 5q, block 5 either way. On a b round kv8.serves
answers False for every row, so the round's verify attention runs on the bf16 relay tile; on a
q round it answers as the engine would, so rows from N8 up to the ceiling read the int8 mirror.
The seams look kv8.serves up through the module on every call, so install() replacing that
attribute reaches both. A round's draft and verify run between this round's
_dflash_next_block_size and the next one's, so the arm drawn there covers exactly that round.
begin_decode() turns kv8 on for the start of every decode: the window the engine opened is left
as it is, the decode's own prefill segment (its last one-row forward) builds the mirror as it
does in production, and that cost lands in the turn's time to first token, never in a round.
Off rounds leave the mirror and its trim wrap in place, so the next on round quantises the rows
the off rounds appended (the clamp-and-update every served call runs) and an off round pays
nothing for kv8. The window's serial is never touched.

install(arms, seed=) patches three module globals of mlx-vlm 0.7.2's speculative/dflash.py, each
looked up per round inside _dflash_rounds (pinned below), and kv8.serves:
  - _dflash_next_block_size: when the drafter pins its requested block (sous sets
    prefer_requested_block_size), return min(drawn block, requested_block_total,
    remaining_budget) with the arm drawn from random.Random(seed); otherwise the original policy
    and no arm.
  - _sample_dflash_target_walk (sampled) and _speculative_walk (greedy): the originals,
    unchanged, timed. Both end in a host read, so the stamp after them is the round's end.
install(None) records rounds at the engine's own block and leaves kv8 serving as the engine does.

Each round record also carries kv8_on (the round's switch) and kv8_rows: the verify rows kv8
served between the round's dispatch and its walk's end (kv8.calls["verify_rows_kv8"]).
Records stay in memory (no I/O, no extra device sync on the generation thread).
"""

from __future__ import annotations

import hashlib
import inspect
import random
import time

from mlx_vlm.speculative import cache_state, common, dflash

from sous.engine import kv8

PINNED = {
    "_dflash_rounds": (dflash._dflash_rounds, "eee8dc9ab95a74ed"),
    "_dflash_next_block_size": (dflash._dflash_next_block_size, "5e161dccd09d2f73"),
    "_sample_dflash_target_walk": (dflash._sample_dflash_target_walk, "6cb5e7aebde0b9f6"),
    "_speculative_walk": (common._speculative_walk, "4487bf2099c8d3a6"),
    "_dflash_block_total": (common._dflash_block_total, "a9c660aef8d58aef"),
    "_reserve_dflash_target_cache": (dflash._reserve_dflash_target_cache, "a122a70af6663e7a"),
    "commit_speculative_round": (cache_state.commit_speculative_round, "58c3e8cf462dea7a"),
}
for _name, (_fn, _digest) in PINNED.items():
    _got = hashlib.sha256(inspect.getsource(_fn).encode()).hexdigest()[:16]
    assert _got == _digest, f"mlx-vlm drift: {_name} source {_got} != {_digest}"

_orig_next = dflash._dflash_next_block_size
_orig_sampled = dflash._sample_dflash_target_walk
_orig_greedy = dflash._speculative_walk
_orig_serves = kv8.serves
S = None  # the installed _State, or None


class _State:
    def __init__(self, arms, seed):
        self.arms = sorted(arms, key=lambda a: (int(a[:-1]), a[-1])) if arms else None
        for a in self.arms or ():
            assert a[-1] in "bq" and int(a[:-1]) >= 2, f"bad arm {a!r} (a block of 1 drafts nothing)"
        self.seed = seed
        self.rng = random.Random(seed)
        self.on = True
        self.records: list[dict] = []
        self.decode = -1
        self.round = 0
        self.rem0 = None
        self.prev_rem = None
        self.prev_t_end = None
        self.new_decode = True
        self.pending = None


def _serves(n_row: int) -> bool:
    st = S
    if st is not None and not st.on:
        return False
    return _orig_serves(n_row)


def install(arms=None, *, seed=0) -> None:
    global S
    S = _State(arms, seed)
    dflash._dflash_next_block_size = _next_block
    dflash._sample_dflash_target_walk = _sampled_walk
    dflash._speculative_walk = _greedy_walk
    kv8.serves = _serves


def begin_decode() -> None:
    if S is None:
        return
    # A turn's prefill and first token run as the daemon runs them, whatever the last round of
    # the previous turn drew.
    S.on = True
    S.new_decode = True


def drain() -> list[dict]:
    if S is None:
        return []
    out, S.records = S.records, []
    return out


def _next_block(draft_model, requested_block_total, remaining_budget, initial_block_size=None):
    st = S
    if st is None:
        return _orig_next(draft_model, requested_block_total, remaining_budget, initial_block_size)
    t_start = time.perf_counter()
    arm = None
    if st.arms is not None and getattr(draft_model, "prefer_requested_block_size", False):
        total = min(requested_block_total, remaining_budget)
        arm = st.rng.choice(st.arms) if total > 1 else None
        drawn = None if arm is None else int(arm[:-1])
        bs = total if drawn is None else min(drawn, total)
        if arm is not None:
            st.on = arm[-1] == "q"
    else:
        drawn = None
        bs = _orig_next(draft_model, requested_block_total, remaining_budget, initial_block_size)
    if bs <= 1:  # the loop breaks: no round
        st.pending = None
        return bs
    # remaining_budget strictly falls within a decode (every round emits >= 1 token)
    if st.new_decode or st.prev_rem is None or remaining_budget >= st.prev_rem:
        st.decode += 1
        st.round = 0
        st.rem0 = remaining_budget
        st.prev_t_end = None
        st.new_decode = False
    st.prev_rem = remaining_budget
    st.pending = {
        "decode": st.decode, "round": st.round, "drawn": drawn, "block": int(bs), "arm": arm,
        "kv8_on": st.on, "requested": int(requested_block_total), "rem": int(remaining_budget),
        "base": st.rem0 - remaining_budget + 1, "t_start": t_start,
        "rows0": kv8.calls["verify_rows_kv8"],
    }
    st.round += 1
    return bs


def _record(t_end, mode, accepted, new_tokens, drafted, pos=None):
    st = S
    p, st.pending = st.pending, None
    if p is None:  # a round the dispatcher did not open (the batched loop): not ours
        return
    target = p["drawn"] if p["drawn"] is not None else p["requested"]
    emitted = len(new_tokens)
    first = p["round"] == 0 or st.prev_t_end is None
    p.update(
        mode=mode, D=int(drafted), accepted=int(accepted), emitted=emitted,
        trunc=p["block"] < target or emitted < accepted + 1, t_end=t_end,
        dt=None if first else t_end - st.prev_t_end,
        gap=None if first else p["t_start"] - st.prev_t_end,
        kv8_rows=kv8.calls["verify_rows_kv8"] - p.pop("rows0"),
    )
    if pos is not None:
        p["pos"] = int(pos)
    st.records.append(p)
    st.prev_t_end = t_end


def _sampled_walk(logits, draft_tokens, sampler, budgets, *, row_ids, base_positions):
    out = _orig_sampled(
        logits, draft_tokens, sampler, budgets, row_ids=row_ids, base_positions=base_positions
    )
    t_end = time.perf_counter()
    if S is not None and len(budgets) == 1:
        _record(t_end, "sampled", out[0][0], out[1][0], draft_tokens.shape[1], base_positions[0])
    return out


def _greedy_walk(draft_tokens, target_tokens, budget):
    out = _orig_greedy(draft_tokens, target_tokens, budget)
    t_end = time.perf_counter()
    if S is not None:
        _record(t_end, "greedy", out[0], out[1], draft_tokens.shape[1])
    return out
```

Create `~/code/sous-spikes/kv8-gate/wrap_kv8.py`:

```python
"""GATE_WRAP shim for the kv8 merge gate (out of tree, never committed; from p157/wrap157.py,
which is p148/gate/wrap_gate.py, without wrap157's block-max override).

gate132.py and gate157.py run `<tree>/.venv/bin/python wrap_kv8.py` in place of `sous serve`.
This runs the branch's own `sous serve` and, after every VLMEngine.generate(), atomically writes
to $TILE132_COUNTERS what the drivers record with each turn: the turn's counter deltas, the
process totals, the engine's kv_int8, projection_kernel and attention_tile states and its block.
The counters are tileattn.calls as they are, projkernel.calls under `proj_`, verifyattn.calls
under `verifyattn_` and kv8.calls under `kv8_`. Every kv8.enable() logs its wall time to stderr,
the run's daemon.out.

Optional, by environment:
  GATE_ROUNDS=<path>  per-round records (rounds_kv8.py) and a summary after every generate(), as
                      JSON lines in analyze157.py's shape; the path must not exist yet
  GATE_ARMS=5b,5q     with GATE_ROUNDS: kv8 off (b) or on (q) drawn per round from GATE_SEED
                      (default 1810); refuses unless kv8 is active, a drafter is loaded and pins
                      its block, the configured block is the largest arm block and the sampler
                      is not greedy
  GATE_TRACE=<path>   per generate(), the decode call's token ids as JSON lines; with
                      GATE_TRACE_TOP2=1 also each token's two likeliest tokens and their
                      logprobs, a host read per token: parity runs only, never a timed one

usage: <tree>/.venv/bin/python wrap_kv8.py [sous cli args...]   (no args = serve)
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sous.engine import kv8, projkernel, tileattn, verifyattn, vlm  # noqa: E402

COUNTERS = os.environ.get("TILE132_COUNTERS", "")
ROUNDS = os.environ.get("GATE_ROUNDS", "")
ARMS = [a for a in os.environ.get("GATE_ARMS", "").split(",") if a]
SEED = int(os.environ.get("GATE_SEED", "1810"))
TRACE = os.environ.get("GATE_TRACE", "")
TOP2 = os.environ.get("GATE_TRACE_TOP2", "") == "1"
for _path in (ROUNDS, TRACE):
    if _path and os.path.exists(_path):
        raise SystemExit(f"kv8gate: {_path} exists; use a fresh path per run")
if ARMS and not ROUNDS:
    raise SystemExit("kv8gate: GATE_ARMS needs GATE_ROUNDS")
if ROUNDS:
    import rounds_kv8  # noqa: E402
_turns = 0
_rounds_installed = False
_trace: list[list] = []  # one list of (generation index, row) per decode call of this turn


def say(msg: str) -> None:
    print(f"kv8gate: {msg}", file=sys.stderr, flush=True)


def counters() -> dict[str, int]:
    out = dict(tileattn.calls)
    out.update({f"proj_{k}": v for k, v in projkernel.calls.items()})
    out.update({f"verifyattn_{k}": v for k, v in verifyattn.calls.items()})
    out.update({f"kv8_{k}": v for k, v in kv8.calls.items()})
    return out


def _atomic(path: str, snapshot: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp, path)
    except OSError as e:
        say(f"counters not written ({e})")


def _append(path: str, lines: list[dict]) -> None:
    if not lines:
        return
    try:
        with open(path, "a") as f:
            f.write("".join(json.dumps(line) + "\n" for line in lines))
    except OSError as e:
        say(f"{path} not written ({e})")


def _refusal(engine) -> str | None:
    if not ARMS:
        return None
    status = engine.kv_int8_status
    if status.get("state") != "active":
        return f"kv8 {status}"
    if engine._draft is None:
        return "no drafter loaded"
    if not getattr(engine._draft, "prefer_requested_block_size", False):
        return "the drafter does not pin its block"
    if engine._draft_block_size != max(int(a[:-1]) for a in ARMS):
        return f"configured block {engine._draft_block_size} is not the largest arm block"
    if engine._sampler is None:
        return "greedy: the per-round arms are judged on sampled turns"
    return None


def _timed_enable(original):
    @functools.wraps(original)
    def enable(*args, **kwargs):
        start = time.perf_counter()
        status = original(*args, **kwargs)
        say(
            f"kv8.enable seconds={time.perf_counter() - start:.3f} "
            f"state={status.get('state')} reason={status.get('reason')!r}"
        )
        return status

    return enable


def _install_trace() -> None:
    import mlx.core as mx
    import mlx_vlm

    original = mlx_vlm.stream_generate

    @functools.wraps(original)
    def stream_generate(*args, **kwargs):
        steps: list = []
        _trace.append(steps)
        gen = original(*args, **kwargs)
        try:
            for r in gen:
                if not getattr(r, "is_draft", False) and r.token is not None:
                    row = [int(r.token)]
                    if TOP2 and hasattr(r.logprobs, "reshape"):
                        lp = r.logprobs.reshape(-1).astype(mx.float32)
                        first = int(mx.argmax(lp).item())
                        rest = mx.where(mx.arange(lp.shape[0]) == first, -mx.inf, lp)
                        second = int(mx.argmax(rest).item())
                        row += [first, float(lp[first].item()), second, float(lp[second].item())]
                    steps.append((r.generation_tokens, row))
                yield r
        finally:
            gen.close()

    mlx_vlm.stream_generate = stream_generate


def install() -> None:
    kv8.enable = _timed_enable(kv8.enable)
    if TRACE:
        _install_trace()
    original = vlm.VLMEngine.generate

    @functools.wraps(original)
    def generate(self, messages, tools, max_tokens, on_delta=None):
        global _turns, _rounds_installed
        if ROUNDS:
            why = _refusal(self)
            if why:
                say(f"REFUSED: {why}")
                raise RuntimeError(f"kv8gate refused: {why}")
            if not _rounds_installed:
                rounds_kv8.install(ARMS or None, seed=SEED)
                _rounds_installed = True
                head = {
                    "type": "install", "pid": os.getpid(), "greedy": self._sampler is None,
                    "arms": ARMS or None, "seed": SEED, "configured_block": self._draft_block_size,
                    "drafter": self._draft_id, "kv8": dict(self.kv_int8_status),
                    "proj": dict(self.projection_kernel_status),
                }
                _append(ROUNDS, [head])
                say(f"rounds installed {head}")
            rounds_kv8.begin_decode()
        key = hashlib.sha256(
            json.dumps([messages, tools, max_tokens], sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        before = counters()
        _trace.clear()
        start = time.perf_counter()
        try:
            return original(self, messages, tools, max_tokens, on_delta)
        finally:
            seconds = time.perf_counter() - start
            now = counters()
            _turns += 1
            if COUNTERS:
                _atomic(
                    COUNTERS,
                    {
                        "turns": _turns,
                        "kv_int8": dict(self.kv_int8_status),
                        "projection_kernel": dict(self.projection_kernel_status),
                        "attention_tile": dict(self.attention_tile_status),
                        "draft_block": self._draft_block_size if self._draft is not None else None,
                        "last_turn": {k: v - before.get(k, 0) for k, v in now.items()},
                        "totals": now,
                        "last_turn_seconds": round(seconds, 3),
                    },
                )
            if ROUNDS:
                recs = rounds_kv8.drain()
                summary = {
                    "type": "turn", "turn": _turns, "key": key, "seconds": round(seconds, 3),
                    "rounds": len(recs),
                    "arms": dict(collections.Counter(str(r.get("arm")) for r in recs)),
                    "blocks": dict(collections.Counter(r["block"] for r in recs)),
                    "max_drafted": max((r["D"] for r in recs), default=0),
                    "tokens": sum(r["emitted"] for r in recs),
                    "kv8_rows": sum(r["kv8_rows"] for r in recs),
                }
                _append(
                    ROUNDS,
                    [{"type": "round", "turn": _turns, "key": key, **r} for r in recs] + [summary],
                )
            if TRACE:
                steps = dict(_trace[-1]) if _trace else {}
                rows = [steps[i] for i in sorted(steps)]
                _append(
                    TRACE,
                    [
                        {
                            "turn": _turns, "key": key, "decodes": len(_trace),
                            "tokens": [row[0] for row in rows],
                            "top2": [row[1:] for row in rows] if TOP2 else None,
                        }
                    ],
                )

    vlm.VLMEngine.generate = generate


if __name__ == "__main__":
    install()
    from sous.cli import main

    main(sys.argv[1:] or ["serve"])
```

- [ ] **Step 7: Write the replay runs**

Create `~/code/sous-spikes/kv8-gate/run_kv8.sh`:

```zsh
#!/bin/zsh
# The kv8 merge gate's replay runs, one after another, each its own GPU-lock job with a hard
# timeout (out of tree, never committed; from relay-gate/run_gate_relay.sh and p157/run157.sh).
# usage: run_kv8.sh <done-file> <spec>...   spec = label:driver:tree:temp:maxtok:variant:turnset:mode
#   driver   g132 (gate132.py) | g157 (gate157.py), both copied beside this script, so every run
#            lands in runs/<label> here
#   tree     main (~/code/personal/sous-remote-main, its own `sous serve`) | kv8
#            (~/code/personal/sous-remote-kv8, through wrap_kv8.py)
#   variant  - | off (kv_int8 = false) | nodraft (speculative_draft_id = "")
#   turnset  acc (#131's 64 turns, the drivers' default) | par (acc plus a6443762's continuation
#            turns 176, 239, 300 and 365, 131K to 229K tokens: 68 turns)
#   mode     - (counters only) | arms<seed> (5b against 5q drawn per round, with per-round
#            records) | tok (decode token ids) | top2 (token ids and each token's top two
#            logprobs); main takes neither a variant nor a mode
G=~/code/personal/sous-spikes/kv8-gate
L=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
MAIN=~/code/personal/sous-remote-main
KV8=~/code/personal/sous-remote-kv8
PY=$MAIN/.venv/bin/python
DONE=${1:A}; shift
PAR=(agent-a160cc42252609969.jsonl agent-a257d5dad84e70f90.jsonl agent-a51b90a20d6d54ff6.jsonl
     agent-a7bb20c376e462d83.jsonl agent-a6443762163413a72.jsonl:10 agent-a442f184ac548b226.jsonl
     agent-a6443762163413a72.jsonl@176,239,300,365)
mkdir -p $G/logs $G/rounds $G/traces $G/runs
rm -f $DONE $DONE.part
echo "trees main=$(git -C $MAIN rev-parse --short HEAD) kv8=$(git -C $KV8 rev-parse --short HEAD) $(date +%H:%M:%S)" >> $DONE.part
for spec in "$@"; do
  parts=(${(s/:/)spec})
  if [ ${#parts} != 8 ]; then echo "$spec: not 8 fields" >> $DONE.part; break; fi
  label=$parts[1]; driver=$parts[2]; tree=$parts[3]; temp=$parts[4]; maxtok=$parts[5]
  variant=$parts[6]; turnset=$parts[7]; mode=$parts[8]
  if [ -e $G/runs/$label ] || [ -e $G/rounds/$label.jsonl ] || [ -e $G/traces/$label.jsonl ]; then
    echo "$label: exists; pick a new label" >> $DONE.part; break
  fi
  extra=(--model 'prompt_cache_gb = 14')
  case $variant in
    -) ;;
    off) extra+=(--model 'kv_int8 = false');;
    nodraft) extra+=(--model 'speculative_draft_id = ""');;
    *) echo "$label: unknown variant $variant" >> $DONE.part; break;;
  esac
  case $turnset in
    acc) picks=();;
    par) picks=($PAR);;
    *) echo "$label: unknown turn set $turnset" >> $DONE.part; break;;
  esac
  case $driver in
    g132) drv=$G/gate132.py;;
    g157) drv=$G/gate157.py;;
    *) echo "$label: unknown driver $driver" >> $DONE.part; break;;
  esac
  case $tree in
    main) dir=$MAIN; wrap=;;
    kv8) dir=$KV8; wrap=$G/wrap_kv8.py;;
    *) echo "$label: unknown tree $tree" >> $DONE.part; break;;
  esac
  envs=()
  case $mode in
    -) ;;
    arms*) envs=(GATE_ROUNDS=$G/rounds/$label.jsonl GATE_ARMS=5b,5q GATE_SEED=${mode#arms});;
    tok) envs=(GATE_TRACE=$G/traces/$label.jsonl);;
    top2) envs=(GATE_TRACE=$G/traces/$label.jsonl GATE_TRACE_TOP2=1);;
    *) echo "$label: unknown mode $mode" >> $DONE.part; break;;
  esac
  if [ $tree = main ] && { [ $mode != - ] || [ $variant != - ]; }; then
    echo "$label: main takes no variant and no mode" >> $DONE.part; break
  fi
  env GATE_WRAP=$wrap $envs GPU_LABEL=kv8-gate python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 10800 \
    $PY $drv $label $dir $temp $maxtok $extra $picks > $G/logs/$label.log 2>&1
  echo "$label exit=$? $(date +%H:%M:%S)" >> $DONE.part
  # nothing of ours may be left on the gate port before the next run
  if lsof -nP -iTCP:8384 -sTCP:LISTEN >/dev/null 2>&1; then echo "$label: port 8384 still busy" >> $DONE.part; break; fi
  if pgrep -f "$drv $label " >/dev/null || pgrep -f "runs/$label" >/dev/null; then echo "$label: process left" >> $DONE.part; break; fi
done
mv $DONE.part $DONE
```

Create `~/code/sous-spikes/kv8-gate/chain_kv8.sh`. The runs:
- **`rounds`, gate step 3:** `r-s1` and `r-s2`, the branch through gate157.py, sampled at 0.7, production `max_tokens` 32000, #131's 64 turns, arms `5b,5q` drawn per round with seeds 1811 and 1812. The block is the M5's default, 5, which the shim requires.
- **`rest`, gate steps 4 and 5:**
  - the same-hour ABBA, the branch with `kv_int8` on against off, same settings: `on-1`, `off-1`, `off-2`, `on-2`;
  - parity: the branch with the drafter against without, greedy, `max_tokens` 256, on the 64 turns plus the continuation past 128K: `p-d-g` (token trace) and `p-nd-g` (token trace with each token's top two logprobs, for reading a divergence);
  - off equals main, greedy, 256, the same 68 turns: `off-d-g` (the branch with `kv_int8 = false`) and `main-d-g`.

```zsh
#!/bin/zsh
# The kv8 merge gate's replay phases (out of tree, never committed). `rounds` is gate step 3, the
# two per-round runs; `rest` is step 4's ABBA and then step 5's parity runs, each phase starting
# only when the one before it listed every run exit=0 and nothing else.
# usage: chain_kv8.sh rounds|rest
G=~/code/personal/sous-spikes/kv8-gate
phase() {  # phase <name> <runs> <spec>...
  local name=$1 n=$2
  shift 2
  zsh $G/run_kv8.sh $G/logs/${name}_DONE "$@" > $G/logs/chain-$name.txt 2>&1
  if [ "$(grep -c ' exit=0 ' $G/logs/${name}_DONE)" != $n ] || [ "$(grep -vc '^trees ' $G/logs/${name}_DONE)" != $n ]; then
    echo "phase $name incomplete; the chain stops"; cat $G/logs/${name}_DONE; exit 1
  fi
  echo "phase $name done $(date +%H:%M:%S)"
}
case $1 in
  rounds)
    phase ROUNDS 2 r-s1:g157:kv8:0.7:32000:-:acc:arms1811 r-s2:g157:kv8:0.7:32000:-:acc:arms1812;;
  rest)
    phase ABBA 4 on-1:g132:kv8:0.7:32000:-:acc:- off-1:g132:kv8:0.7:32000:off:acc:- \
      off-2:g132:kv8:0.7:32000:off:acc:- on-2:g132:kv8:0.7:32000:-:acc:-
    phase PARITY 4 p-d-g:g132:kv8:0.0:256:-:par:tok p-nd-g:g132:kv8:0.0:256:nodraft:par:top2 \
      off-d-g:g132:kv8:0.0:256:off:par:- main-d-g:g132:main:0.0:256:-:par:-;;
  *) echo "usage: chain_kv8.sh rounds|rest"; exit 2;;
esac
echo "chain $1 done $(date +%H:%M:%S)"
```

- [ ] **Step 8: Write gate step 3's analysis**

The metric is #157's: analyze157's PRIMARY, `tok/s (dt-attr, strat)`, arms drawn per round, a within-decode regression with a quadratic in base, 4000 resamples clustered over distinct turns, files paired by turn, both seeds pooled. The reference arm is `5b`.

Create `~/code/sous-spikes/kv8-gate/analyze_kv8.py`:

```python
"""Gate step 3 of the kv8 merge gate (out of tree, never committed): kv8 on (arm 5q) against off
(5b), drawn per round in one daemon process per seed, by analyze157's PRIMARY metric (tok/s
(dt-attr, strat), 95% percentile bootstrap over distinct turns, 4000 resamples, files paired by
turn) with 5b as the reference, and #157's rule on it:
  SHIP if 5q/5b >= 1.02 with the lower bound > 1.00; CLOSE if the lower bound <= 1.00 and the
  point < 1.02; otherwise TO THE MAINTAINER.
The ratio is gross of the mirror's first build, which lands in each turn's time to first token
(gate step 4 reports the net).

analyze157.py (copied beside this file) runs whole first, for its diagnostics; its trailing
`DECISION (p157 README ...)` header has no line for these arms and decides nothing here. Then
the context histogram, the same ratio per stratum (turns whose replies are at most 64 tokens,
then the other turns by prompt length in 44K bands; a stratum without mixed decodes reads n/a),
and the pooled ratio with the verdict.

usage: analyze_kv8.py <rounds.jsonl>:<result.json> [...]   (each per-round file with its run's
       gate157 result.json)
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("analyze157", os.path.join(HERE, "analyze157.py"))
a157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a157)
a157.REF = "5b"
ON = "5q"
BAND = 44000
SHORT = 64


def facts(results):
    """(file index, turn) -> (prompt tokens, reply tokens). The wrapper numbers a process's
    generate() calls from 1, one per replayed turn, in the driver's order."""
    out = {}
    for fi, path in enumerate(results):
        with open(path) as f:
            res = json.load(f)["results"]
        for t, r in enumerate(res, 1):
            line = r.get("line") or {}
            out[(fi, t)] = (int(line.get("input_tokens") or r.get("n") or 0), int(line.get("output_tokens") or 0))
    return out


def stratum(n, out):
    if out <= SHORT:
        return f"replies <= {SHORT} tokens"
    lo = n // BAND * 44
    return f"prompt {lo}-{lo + 44}K"


def primary(rounds, gb):
    keys = sorted({r["cluster"] for r in rounds}, key=str)
    blocks = sorted({r["block"] for r in rounds if not r["drop"]}, key=lambda a: (int(a[:-1]), a[-1]))
    if a157.REF not in blocks or ON not in blocks:
        return None
    A, c, ref, used, others = a157.strat_parts(rounds, blocks, keys)
    i = others.index(ON)
    if ref[:, 4].sum() == 0 or A.sum(0)[i, i] <= 1e-9:
        return None
    w = np.random.default_rng(118).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=a157.B)
    with np.errstate(invalid="ignore", divide="ignore"):
        spt = a157.strat_metrics(A.sum(0), c.sum(0), ref.sum(0), gb, blocks, others)
        sboot = a157.strat_metrics(
            np.einsum("bc,cij->bij", w, A), np.einsum("bc,cij->bij", w, c), w @ ref, gb, blocks, others
        )
        on, off = blocks.index(ON), blocks.index(a157.REF)
        lo, hi = a157.ci(sboot[:, on, 5] / sboot[:, off, 5])
        return float(spt[on, 5] / spt[off, 5]), float(lo), float(hi), len(keys), used


def main(args):
    pairs = [a.rsplit(":", 1) for a in args]
    rounds_paths = [p for p, _ in pairs]
    f = facts([r for _, r in pairs])
    a157.main(rounds_paths)
    rounds, _, mode = a157.load(rounds_paths)
    if mode != "sampled":
        raise SystemExit(f"mode {mode}: gate step 3 is judged on sampled files")
    for r in rounds:
        r["stratum"] = stratum(*f[(r["dkey"][0], r["turn"])])
    _, gb, _ = a157.gap_fit(rounds)

    print("\nCONTEXT HISTOGRAM (prompt tokens of the replayed turns, first file)")
    ns = sorted(n for (fi, _), (n, _) in f.items() if fi == 0)
    for band in sorted({n // BAND for n in ns}):
        print(f"  {band * 44}-{band * 44 + 44}K: {sum(n // BAND == band for n in ns)} turns")
    print(f"  smallest {ns[0]}, largest {ns[-1]}")
    print(f"  replies <= {SHORT} tokens: " + ", ".join(
        f"file {fi + 1} {sum(1 for (g, _), (_, o) in f.items() if g == fi and o <= SHORT)}" for fi in range(len(pairs))
    ))

    print(f"\nSTRATA ({ON}/{a157.REF} tok/s (dt-attr, strat), 95% CI over turns)")
    for s in sorted({r["stratum"] for r in rounds}):
        sub = [r for r in rounds if r["stratum"] == s]
        res = primary(sub, gb)
        replays = sum(1 for v in f.values() if stratum(*v) == s)
        cell = "n/a" if res is None else f"{res[0]:.4f} [{res[1]:.4f}, {res[2]:.4f}] over {res[3]} turns"
        print(f"  {s}: {replays} turn replays, {len(sub)} rounds: {cell}")

    res = primary(rounds, gb)
    if res is None:
        raise SystemExit("PRIMARY not available: the arms were not mixed within decodes")
    r, lo, hi, nturns, used = res
    if r >= 1.02 and lo > 1.00:
        v = "SHIP"
    elif lo <= 1.00 and r < 1.02:
        v = "CLOSE"
    else:
        v = "TO THE MAINTAINER"
    print(
        f"\nSTEP 3 (pre-registered rule): {ON}/{a157.REF} {r:.4f} [{lo:.4f}, {hi:.4f}] over {nturns} "
        f"distinct turns ({used} rounds in the regression) vs >= 1.02 with the lower bound > 1.00 -> {v}"
    )
    print(f"step3: {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 9: Write the judgements of gate steps 3–5**

Create `~/code/sous-spikes/kv8-gate/judge_kv8.py`:

```python
"""Judgements for the kv8 merge gate (out of tree, never committed). A judging subcommand prints
its evidence and ends with one line, `<name>: PASS` or `<name>: FAIL`, and exits 0 only on PASS;
gap and figures only report and always exit 0. Run it from the branch tree's venv: the mirror's
size comes from sous.engine.kv8's own constants.

usage: judge_kv8.py valid <gate dir> <label>=<turns>:<kind>...   kind drafted|nodraft|off|main
       judge_kv8.py rounds <rounds.jsonl> <turns>
       judge_kv8.py abba <runs> <on-1> <off-1> <off-2> <on-2>
       judge_kv8.py parity <runs> <drafted> <undrafted>
       judge_kv8.py gap <runs> <traces dir> <drafted> <undrafted>
       judge_kv8.py figures <runs> <README.md> <on-1> <off-1> <off-2> <on-2>
kinds: drafted = the branch, kv8 on, with the DFlash2 drafter; nodraft = the branch, kv8 on, no
drafter; off = the branch with kv_int8 = false; main = main's own `sous serve`.
"""

import collections
import json
import os
import random
import re
import sys

from sous.engine import kv8

KV = re.compile(r"(\w+)=(\S+)")
LAYERS, HEADS = 16, 4
GIB_GB = 2**30 / 1e9  # memory_gb is bytes / 1e9
EDGE = 16  # rows a drafter-on decode's last verify call may hold past the committed length


def verdict(name, ok):
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def result(runs, label):
    with open(os.path.join(runs, label, "result.json")) as f:
        return json.load(f)


def load_line(d):
    raw = d["load"][-1]["_raw"] if d.get("load") else ""
    return raw.split("INFO ", 1)[-1], dict(KV.findall(raw))


def num(line, key, cast=float):
    v = (line or {}).get(key)
    return None if v in (None, "-") else cast(v)


def where(r):
    return f"{r['agent'][6:14]}#{r['turn']}"


def by_turn(d):
    return {(r["agent"], r["turn"]): r for r in d["results"] if r.get("line")}


def round_up(x, m):
    return -(-x // m) * m


def cmd_valid(gate, specs):
    ok = True
    runs = os.path.join(gate, "runs")
    for spec in specs:
        label, _, want = spec.partition("=")
        turns, _, kind = want.partition(":")
        bad = []
        with open(os.path.join(gate, "logs", f"{label}.log"), errors="replace") as f:
            log = f.read()
        if "daemon stayed idle during the job" not in log or "DAEMON-CONTAMINATED" in log:
            bad.append("the daemon was not idle throughout")
        if "[gpu_run_m5] exit=0" not in log:
            bad.append("gpu_run_m5's exit is not 0")
        d = result(runs, label)
        res = d["results"]
        if len(res) != int(turns):
            bad.append(f"{len(res)} results, expected {turns}")
        if not all((r.get("line") or {}).get("status") == "200" for r in res):
            bad.append("a turn without a status=200 line")
        if len(d["load"]) != 1:
            bad.append(f"{len(d['load'])} model loads")
        raw, kv = load_line(d)
        if kv.get("attention_tile") != "active" or kv.get("projection_kernel") != "active":
            bad.append("the tile or the projection kernel is not active")
        with open(os.path.join(runs, label, "daemon.out"), errors="replace") as f:
            out = f.read()
        if "REFUSED" in out:
            bad.append("REFUSED in daemon.out")
        if os.path.isdir(os.path.join(runs, label, "home", ".sous", "forks")):
            bad.append("the run's forks were left behind")
        totals = ((res[-1].get("tile") or {}).get("totals") or {}) if res else {}
        k8 = {k[4:]: v for k, v in totals.items() if k.startswith("kv8_")}
        states = collections.Counter((r.get("line") or {}).get("kv8") for r in res)
        block = kv.get("draft_block", "-")
        if kind == "main":
            if "kv_int8" in kv or block != "5":
                bad.append(f"main's load line: kv_int8={kv.get('kv_int8')} draft_block={block}")
        elif kind in ("off", "drafted", "nodraft"):
            if "kv8gate: kv8.enable seconds=" not in out:
                bad.append("kv8.enable was never timed: the wrapper missed the load")
            if kind == "off":
                if kv.get("kv_int8") != "off" or block != "5":
                    bad.append(f"kv_int8={kv.get('kv_int8')} draft_block={block}")
                if not k8 or any(k8.values()):
                    bad.append(f"kv8 counters moved: {k8}")
                if set(states) != {"off"}:
                    bad.append(f"turn lines' kv8= {dict(states)}")
            else:
                if kv.get("kv_int8") != "active" or "kv_int8_floor" not in kv or "kv_int8_ceiling" not in kv:
                    bad.append(f"kv_int8={kv.get('kv_int8')} without its floor and ceiling")
                if kind == "drafted" and (block != "5" or not k8.get("verify_rows_kv8") or not k8.get("decode_kv8")):
                    bad.append(f"draft_block={block}, kv8 counters {k8}")
                if kind == "nodraft" and (block != "-" or not k8.get("decode_kv8") or k8.get("verify_kv8")):
                    bad.append(f"draft_block={block}, kv8 counters {k8}")
        else:
            bad.append(f"unknown kind {kind}")
        print(f"{label} ({kind}): {'ok' if not bad else 'FAIL: ' + '; '.join(bad)} | {raw}")
        ok &= not bad
    return verdict("valid", ok)


def cmd_rounds(path, turns):
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    inst = [x for x in recs if x["type"] == "install"]
    rounds = [x for x in recs if x["type"] == "round"]
    summaries = [x for x in recs if x["type"] == "turn"]
    bad = []
    if len(inst) != 1:
        bad.append(f"{len(inst)} install lines")
    else:
        i = inst[0]
        if (
            i["arms"] != ["5b", "5q"]
            or i["configured_block"] != 5
            or i["greedy"]
            or (i.get("kv8") or {}).get("state") != "active"
        ):
            bad.append(
                f"install: arms {i['arms']} configured_block {i['configured_block']} "
                f"greedy {i['greedy']} kv8 {i.get('kv8')}"
            )
    arms = collections.Counter(x["arm"] for x in rounds)
    if not arms.get("5b") or not arms.get("5q"):
        bad.append(f"rounds by arm {dict(arms)}")
    leaked = sum(1 for x in rounds if x["arm"] == "5b" and x["kv8_rows"])
    served = sum(1 for x in rounds if x["arm"] == "5q" and x["kv8_rows"])
    if leaked:
        bad.append(f"{leaked} off rounds served kv8 rows")
    if not served:
        bad.append("no on round served a kv8 row")
    if len(summaries) != turns:
        bad.append(f"{len(summaries)} turn summaries, expected {turns}")
    print(
        f"{os.path.basename(path)}: rounds by arm {dict(arms)}; on rounds with kv8 rows {served}; "
        f"off rounds with kv8 rows {leaked}; turns {len(summaries)}"
    )
    for b in bad:
        print("  " + b)
    return verdict("rounds", not bad)


def dec_ratio(ps):
    ao = sum(num(a["line"], "output_tokens", int) for a, _ in ps)
    ad = sum(num(a["line"], "decode_s") for a, _ in ps)
    bo = sum(num(b["line"], "output_tokens", int) for _, b in ps)
    bd = sum(num(b["line"], "decode_s") for _, b in ps)
    return (ao / ad) / (bo / bd) if ad and bd and bo else None


def boot(ps, stat, seed=181):
    rnd = random.Random(seed)
    vals = sorted(
        v for v in (stat([ps[rnd.randrange(len(ps))] for _ in ps]) for _ in range(4000)) if v is not None
    )
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def abba_pairs(runs, on1, off1, off2, on2):
    on, off = [result(runs, on1), result(runs, on2)], [result(runs, off1), result(runs, off2)]
    halves = []
    for a, b in zip(on, off, strict=True):
        A, B = by_turn(a), by_turn(b)
        halves.append([(A[k], B[k]) for k in A if k in B])
    return on, off, halves


def cmd_abba(runs, on1, off1, off2, on2):
    on, off, halves = abba_pairs(runs, on1, off1, off2, on2)
    pairs = [p for h in halves for p in h]
    _, kv = load_line(on[0])
    floor, ceil = int(kv["kv_int8_floor"]), int(kv["kv_int8_ceiling"])
    ok = True

    r = dec_ratio(pairs)
    lo, hi = boot(pairs, dec_ratio)
    good = r is not None and r >= 1.00
    print(
        f"decode, pooled over {len(pairs)} turn pairs: on/off {r:.3f} (95% CI {lo:.3f}-{hi:.3f}); "
        f"bar >= 1.00 -> {'ok' if good else 'FAIL'}"
    )
    ok &= good

    for key in ("prefill_s", "ttft_s"):
        both = [(a, b) for a, b in pairs if num(a["line"], key) is not None and num(b["line"], key) is not None]
        sa = sum(num(a["line"], key) for a, _ in both)
        sb = sum(num(b["line"], key) for _, b in both)
        rr = sa / sb if sb else None
        good = rr is not None and rr <= 1.03
        print(
            f"{key}: on {sa:.1f} s, off {sb:.1f} s over {len(both)} turn pairs, on/off "
            f"{'-' if rr is None else f'{rr:.3f}'}; bound <= 1.03 -> {'ok' if good else 'FAIL'}"
        )
        ok &= good

    for i, h in enumerate(halves, 1):
        a, b = max(h, key=lambda p: num(p[0]["line"], "input_tokens", int))
        n_in = num(a["line"], "input_tokens", int)
        out = num(a["line"], "output_tokens", int)
        # A decode whose prompt reaches the ceiling caps on its first call and builds nothing.
        rows = 0 if n_in >= ceil else min(round_up(n_in + out + kv8.GROWTH, 256), round_up(ceil, 256))
        mirror_gb = LAYERS * HEADS * kv8.BYTES_PER_ROW_PER_HEAD * rows / 1e9
        diff = (a.get("mem_peak_gb") or 0.0) - (b.get("mem_peak_gb") or 0.0)
        good = diff <= mirror_gb + GIB_GB
        print(
            f"half {i}: largest context {where(a)} n={n_in}: peak active memory on "
            f"{a.get('mem_peak_gb') or 0.0:.2f} GB, off {b.get('mem_peak_gb') or 0.0:.2f} GB, difference "
            f"{diff:+.2f} GB; the mirror's bytes {mirror_gb:.2f} GB ({rows} rows) + 1 GiB = bound "
            f"{mirror_gb + GIB_GB:.2f} GB -> {'ok' if good else 'FAIL'}"
        )
        ok &= good

    total = sum(num(r["line"], "pressure", int) or 0 for d in (*on, *off) for r in d["results"] if r.get("line"))
    print(f"pressure evictions over the four runs: {total} -> {'ok' if total == 0 else 'FAIL'}")
    ok &= total == 0

    mism = [
        f"{where(a)}: took {a['line'].get('took')}/{b['line'].get('took')} "
        f"evicted {a['line'].get('evicted')}/{b['line'].get('evicted')}"
        for a, b in pairs
        if a["line"].get("took") != b["line"].get("took") or a["line"].get("evicted") != b["line"].get("evicted")
    ]
    print(f"took classes and eviction counts equal on {len(pairs) - len(mism)}/{len(pairs)} turn pairs")
    for m in mism:
        print("  " + m)
    ok &= not mism

    bad, capped = [], []
    for a, b in pairs:
        n_in = num(a["line"], "input_tokens", int)
        final = n_in + num(a["line"], "output_tokens", int)
        state = a["line"].get("kv8")
        if state == "capped":
            capped.append(f"{where(a)} input {n_in} final {final}")
        fine = state in ("served", "capped") or (state == "idle" and final < floor)
        if not fine or b["line"].get("kv8") != "off":
            bad.append(f"{where(a)}: on {state}, off {b['line'].get('kv8')}, final {final}")
    print(
        f"kv8 states (on: served or capped, idle only below the floor {floor}; off: off): "
        f"{len(pairs) - len(bad)}/{len(pairs)} turn pairs as required"
    )
    for x in bad:
        print("  " + x)
    print(f"capped turns against the ceiling {ceil}: {capped or 'none'}")
    ok &= not bad
    return verdict("abba", ok)


def cmd_parity(runs, drafted, undrafted):
    a, b = result(runs, drafted), result(runs, undrafted)
    _, kv = load_line(a)
    floor, ceil = int(kv["kv_int8_floor"]), int(kv["kv_int8_ceiling"])
    same = near = 0
    bad = []
    for ra, rb in zip(a["results"], b["results"], strict=True):
        la, lb = ra.get("line") or {}, rb.get("line") or {}
        final = num(la, "input_tokens", int) + num(la, "output_tokens", int)
        if la.get("kv8") == lb.get("kv8") and la.get("kv8") in ("served", "capped", "idle"):
            same += 1
        elif min(abs(final - floor), abs(final - ceil)) <= EDGE:
            near += 1
            print(f"  {where(ra)} final {final}: {la.get('kv8')} / {lb.get('kv8')}, within {EDGE} rows of a line (exempt)")
        else:
            bad.append(f"{where(ra)} final {final}: drafter on {la.get('kv8')}, off {lb.get('kv8')}")
    print(
        f"kv8= identical on {same}/{len(a['results'])} turns; {near} exempt within {EDGE} rows of "
        f"the floor {floor} or the ceiling {ceil}"
    )
    for x in bad:
        print("  " + x)
    return verdict("parity-kv8", not bad)


def cmd_gap(runs, traces, drafted, undrafted):
    res = result(runs, undrafted)["results"]
    with open(os.path.join(traces, f"{drafted}.jsonl")) as f:
        on = [json.loads(line) for line in f]
    with open(os.path.join(traces, f"{undrafted}.jsonl")) as f:
        off = [json.loads(line) for line in f]
    if not (len(on) == len(off) == len(res)):
        print(f"gap: {len(on)} and {len(off)} traced turns for {len(res)} results")
        return 0
    differing = 0
    for i, (a, b) in enumerate(zip(on, off, strict=True)):
        if a["tokens"] == b["tokens"]:
            continue
        differing += 1
        n = min(len(a["tokens"]), len(b["tokens"]))
        j = next((k for k in range(n) if a["tokens"][k] != b["tokens"][k]), n)
        here = f"{where(res[i])} token {j}"
        if j == n:
            print(f"{here}: one reply is a prefix of the other (a stop)")
            continue
        first, lp1, second, lp2 = b["top2"][j]
        took = a["tokens"][j]
        tie = lp1 == lp2 and took in (first, second)
        why = (
            "a tie in drafter-off's bf16 logprobs and the drafter took the other: the argmax residual"
            if tie
            else "no tie at the first differing token: a kernel fault until shown otherwise"
        )
        print(
            f"{here}: drafter off took {first} ({lp1}), runner-up {second} ({lp2}), gap "
            f"{lp1 - lp2}; drafter on took {took} -> {why}"
        )
    print(f"gap: {differing} differing turns of {len(res)}")
    return 0


def cmd_figures(runs, readme, on1, off1, off2, on2):
    _, _, halves = abba_pairs(runs, on1, off1, off2, on2)
    ps = [p for h in halves for p in h if 32768 <= num(p[0]["line"], "input_tokens", int) < 65536]
    if not ps:
        print("figures: no 32-64K turn pair; the README's figures stay")
        return 0
    r = dec_ratio(ps)
    lo, hi = boot(ps, dec_ratio)
    if lo <= 1.0 <= hi:
        print(f"figures: 32-64K on/off {r:.3f} ({lo:.3f}-{hi:.3f}) includes 1.00; the README's figures stay")
        return 0
    with open(readme) as f:
        text = f.read()
    m = re.search(r"about (\d+(?:\.\d+)?) tok/s at 57K", text)
    s = re.search(r"(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)x at 51–57K", text)
    print(f"figures: 32-64K on/off {r:.3f} ({lo:.3f}-{hi:.3f}) over {len(ps)} turn pairs")
    if m:
        old = float(m.group(1))
        print(f"  57K decode: {m.group(1)} x {r:.3f} = {old * r:.2f} -> about {round(old * r * 2) / 2:g} tok/s")
    if s:
        a, b = float(s.group(1)), float(s.group(2))
        print(f"  Splash at 51-57K: {s.group(1)}-{s.group(2)}x / {r:.3f} -> {a / r:.1f}-{b / r:.1f}x")
    return 0


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "valid":
        code = cmd_valid(args[0], args[1:])
    elif cmd == "rounds":
        code = cmd_rounds(args[0], int(args[1]))
    elif cmd == "abba":
        code = cmd_abba(*args)
    elif cmd == "parity":
        code = cmd_parity(*args)
    elif cmd == "gap":
        code = cmd_gap(*args)
    elif cmd == "figures":
        code = cmd_figures(*args)
    else:
        sys.exit(__doc__)
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Write the KL instrument**

Gate step 6(b) scores each token's logits under kv8 on and off over the same prefix. A one-row forward appends to the cache, so the on forward is a detour that is undone:
- the attention layers are trimmed by one row through `KVCache.trim`, which kv8's trim wrap follows;
- the recurrent layers' state arrays are put back.

The script proves on the first step that this restores the cache exactly, or stops. The on forward runs in a forced window, so every row from N8 up is scored, the continuation's rows past the ceiling included.

Create `~/code/sous-spikes/kv8-gate/kl_kv8.py`:

```python
"""Gate step 6(b) of the kv8 merge gate (out of tree, never committed): how far kv8 moves the
default model's next-token distribution, scored token by token on greedy continuations.

Turns: #131's 64 (the drivers' default picks) and a6443762's continuation turns 176, 239, 300 and
365 (131K to 229K tokens), each rendered as the daemon renders it (sous.api.convert, then the
engine's own chat template and tokenizer) and prefilled into a fresh cache with no window open,
so the prompt's K/V are what the daemon computes. Then up to 256 steps of one-row forwards over
the same prefix, twice per step:
  - on: kv8 serving every row from N8 up (a forced window, whatever the ceiling), as a detour
    whose effect is undone: the attention layers trimmed by one row through KVCache.trim, which
    kv8's trim wrap follows, and the recurrent layers' state arrays put back;
  - off: kv8 serving nothing (kv8.serves answers False), kept. The continuation is greedy under
    kv8 off, the argmax of the off step's fp32 logprobs, and ends at an end token.
Per token: KL(off || on) in fp32 and whether the two argmaxes agree. Every kv8 decode call of an
on step also runs the relay tile on the same queries over the layer's bf16 K/V; the relative RMS
between the two outputs is kept per layer and per 4,096-key band. The first step of the first
turn checks that the detour is undone exactly: an off forward taken as a detour must equal the
kept off forward bit for bit, else the script stops.

Tripwire (pre-registered; everything else is information): mean KL <= 0.033, the int8-prefill
precedent.

usage: kl_kv8.py <out.json>   from the branch tree's venv, under gpu_run_m5.py
"""

import importlib.util
import json
import os
import statistics
import sys
import time

import mlx.core as mx

from sous.api.convert import parse_messages_request
from sous.engine import kv8, tileattn
from sous.engine.vlm import VLMEngine

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "mlx-community/Qwen3.8-27B-4bit"
WINDOW = 262144
MAX_TOKENS = 32000  # what the replay set's builder posts, as Claude Code does for a subagent
STEPS = 256
BAND = 4096
TRIPWIRE = 0.033
CONTINUATION = "agent-a6443762163413a72.jsonl@176,239,300,365"
ATTENTION_LAYERS = 16

ON = [False]
LAYER = [0]
PENDING: list = []
_orig_serves = kv8.serves
_orig_decode = kv8.decode


def serves(n_row):
    return ON[0] and _orig_serves(n_row)


def rel_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w * w))


def measured_decode(queries, cache, splits):
    out = _orig_decode(queries, cache, splits)
    n = cache.offset
    ref = tileattn.decode(queries, cache.keys[:, :, :n, :], cache.values[:, :, :n, :], splits)
    PENDING.append((LAYER[0], n, rel_rms(out, ref)))
    LAYER[0] += 1
    return out


def requests():
    spec = importlib.util.spec_from_file_location("gate157", os.path.join(HERE, "gate157.py"))
    g157 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g157)
    with open(os.path.join(HERE, "replay-157.json")) as f:
        data = {os.path.basename(r["file"]): r for r in json.load(f)}
    work = []
    for pick in [*g157.PICKS, CONTINUATION]:
        cap = ""
        if "@" in pick:
            base, _, sel = pick.partition("@")
            idx = [int(x) for x in sel.split(",")]
        else:
            base, _, cap = pick.partition(":")
            idx = None
        turns = [t for t in g157.turns_of(data[base]["file"]) if t["tools"] is not None]
        if idx is None:
            idx = list(range(len(turns)))[: int(cap) if cap else None]
        for k in idx:
            t = turns[k]
            body = {
                "model": "sous-local", "max_tokens": MAX_TOKENS, "system": t["system"],
                "tools": t["tools"], "messages": t["messages"],
            }
            work.append((base, k, body))
    return work


def main():
    out_path = sys.argv[1]
    if os.path.exists(out_path):
        sys.exit(f"{out_path} exists; pick a fresh path")
    from mlx_vlm.models.cache import ArraysCache, KVCache

    work = requests()
    print(f"{len(work)} turns", flush=True)
    engine = VLMEngine(
        MODEL, temperature=0.0, cache_budget=0, int8_prefill=True, attention_tile=True,
        projection_kernel=True, kv_int8=True, kv_int8_window=WINDOW,
    )
    status = engine.kv_int8_status
    print(f"kv_int8 {status}", flush=True)
    if status.get("state") != "active":
        sys.exit("kv8 is not active: the instrument cannot run")
    kv8.serves = serves
    kv8.decode = measured_decode
    lm = engine._model.language_model
    tok = engine._tokenizer
    ends = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int)}
    checked = False
    kl_all: list[float] = []
    agree_all = 0
    errors: dict[int, list[list[float]]] = {}
    turns_out = []
    start = time.time()
    for i, (base, k, body) in enumerate(work):
        chat = parse_messages_request(body)
        ids = engine._ids("full", chat.messages, chat.tools)
        cache = engine.new_cache()
        attn = [c for c in cache if type(c) is KVCache]
        rec = [c for c in cache if isinstance(c, ArraysCache)]
        if len(attn) != ATTENTION_LAYERS or len(attn) + len(rec) != len(cache):
            sys.exit(f"unexpected cache layout: {len(attn)} KVCache and {len(rec)} ArraysCache of {len(cache)}")
        t0 = time.time()
        engine.prefill(cache, ids[:-1])
        prefill_s = time.time() - t0

        def forward(token):
            out = lm(mx.array([[token]], dtype=mx.int32), cache=cache, **engine._positions(cache, 1))
            return out.logits[0, -1].astype(mx.float32)

        def detour(token):
            saved = [list(c.cache) for c in rec]
            logits = forward(token)
            mx.eval(logits)
            for c in attn:
                c.trim(1)
            for c, s in zip(rec, saved, strict=True):
                c.cache[:] = s
            return logits

        kls: list[float] = []
        agree = 0
        token = ids[-1]
        kv8.open_window(force=True)
        try:
            for step in range(STEPS):
                ON[0], LAYER[0] = True, 0
                PENDING.clear()
                on = detour(token)
                ON[0] = False
                if LAYER[0] != ATTENTION_LAYERS:
                    sys.exit(f"{base}#{k} step {step}: kv8 served {LAYER[0]} of {ATTENTION_LAYERS} layers")
                mx.eval([e for _, _, e in PENDING])
                for layer, n, e in PENDING:
                    errors.setdefault(n // BAND * BAND, [[] for _ in range(ATTENTION_LAYERS)])[layer].append(e.item())
                check = None if checked else detour(token)
                off = forward(token)
                mx.eval(off)
                if check is not None:
                    if not mx.array_equal(check, off).item():
                        sys.exit("RESTORE INEXACT: an undone forward changed the next one, so on and off would score different prefixes")
                    print("restore exact: an undone forward leaves the cache as it found it", flush=True)
                    checked = True
                lp_off = off - mx.logsumexp(off)
                lp_on = on - mx.logsumexp(on)
                kl = mx.sum(mx.exp(lp_off) * (lp_off - lp_on))
                top_off, top_on = mx.argmax(lp_off), mx.argmax(lp_on)
                mx.eval(kl, top_off, top_on)
                kls.append(kl.item())
                agree += int(top_off.item() == top_on.item())
                token = int(top_off.item())
                if token in ends:
                    break
        finally:
            ON[0] = False
            kv8.close_window(cache)
        kl_all += kls
        agree_all += agree
        turns_out.append(
            {
                "agent": base, "turn": k, "n": len(ids), "tokens": len(kls),
                "kl_mean": sum(kls) / len(kls), "kl_max": max(kls), "top1": agree / len(kls),
                "prefill_s": round(prefill_s, 1),
            }
        )
        print(
            f"[{i + 1}/{len(work)}] {base[6:14]}#{k} n={len(ids)} prefill {prefill_s:.0f}s "
            f"tokens={len(kls)} mean KL {sum(kls) / len(kls):.5f} max {max(kls):.4f} "
            f"top-1 {agree / len(kls):.4f} ({time.time() - start:.0f}s)",
            flush=True,
        )
        cache = attn = rec = None
        mx.clear_cache()

    layers = [4 * j + 3 for j in range(ATTENTION_LAYERS)]
    print(
        "\nPER-LAYER attention-output error, kv8 against the relay tile on the same queries and "
        "bf16 K/V (relative RMS %, median per 4,096-key band; columns are model layers)"
    )
    print("| keys from | " + " | ".join(f"L{x}" for x in layers) + " |")
    print("|---|" + "---|" * ATTENTION_LAYERS)
    for band in sorted(errors):
        cells = [f"{statistics.median(v) * 100:.2f}" if v else "-" for v in errors[band]]
        print(f"| {band} | " + " | ".join(cells) + " |")
    whole = [[e for per in errors.values() for e in per[j]] for j in range(ATTENTION_LAYERS)]
    print("| all, median | " + " | ".join(f"{statistics.median(v) * 100:.2f}" for v in whole) + " |")
    print("| all, max | " + " | ".join(f"{max(v) * 100:.2f}" for v in whole) + " |")
    mean_kl = sum(kl_all) / len(kl_all)
    top1 = agree_all / len(kl_all)
    print(
        f"\ntokens scored {len(kl_all)} over {len(work)} turns: mean KL(off || on) {mean_kl:.5f}, "
        f"top-1 agreement {top1:.4f}"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "turns": turns_out, "mean_kl": mean_kl, "top1": top1, "tokens": len(kl_all),
                "errors": {str(b): v for b, v in errors.items()},
            },
            f,
        )
    print(
        f"tripwire: {'PASS' if mean_kl <= TRIPWIRE else 'TO THE MAINTAINER'} "
        f"(mean KL {mean_kl:.5f}, bound {TRIPWIRE})",
        flush=True,
    )
    engine.unload()


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Point the suite at `kv_int8` and write gate step 6's M5 script**

`suite_kv8.py` already builds the on/off pair with `dataclasses.replace(cfg, kv_int8=on)`, sharing one scratch directory per job so a (job, task, seed) pair sends both arms one prompt and one seed. Its `Arm(...)` call does not yet set `Arm.kv_int8`, so both arms' rows would carry the field's default, `True`. The spec builds the pair as two `Arm`s with `config` and `kv_int8` replaced.

In `~/code/sous-spikes/kv8-gate/suite_kv8.py`, replace lines 42–43:

```python
        projection_kernel=cfg.projection_kernel,
    )
```

with:

```python
        projection_kernel=cfg.projection_kernel,
        kv_int8=on,
    )
```

Create `~/code/sous-spikes/kv8-gate/quality_m5.sh`:

```zsh
#!/bin/zsh
# Gate step 6 on the M5 Pro (out of tree, never committed): the suite's four jobs (job b runs
# seeds 2b and 2b+1 for both arms, off first for even b), then the KL instrument, each its own
# GPU-lock job with a hard timeout, from the branch tree's venv.
G=~/code/personal/sous-spikes/kv8-gate
L=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
cd ~/code/personal/sous-remote-kv8 || exit 1
mkdir -p $G/logs
rm -f $G/logs/QUALITY
if [ -e $G/suite.jsonl ] || [ -e $G/kl.json ]; then
  echo "suite.jsonl or kl.json exists; move it aside first" >> $G/logs/QUALITY; exit 1
fi
for b in 0 1 2 3; do
  env GPU_LABEL=kv8-suite python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 7200 \
    .venv/bin/python $G/suite_kv8.py $G/suite.jsonl $b > $G/logs/suite-b$b.log 2>&1
  echo "suite-b$b exit=$? $(date +%H:%M:%S)" >> $G/logs/QUALITY
done
env GPU_LABEL=kv8-kl python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 25200 \
  .venv/bin/python $G/kl_kv8.py $G/kl.json > $G/logs/kl.log 2>&1
echo "kl exit=$? $(date +%H:%M:%S)" >> $G/logs/QUALITY
echo "QUALITY DONE $(date +%H:%M:%S)" >> $G/logs/QUALITY
```

- [ ] **Step 12: Write every analysis**

Create `~/code/sous-spikes/kv8-gate/judge_all.sh`:

```zsh
#!/bin/zsh
# Every analysis of the kv8 merge gate, CPU only, after the last M5 GPU job (out of tree, never
# committed). Writes logs/ and prints each judgement's last line.
cd ~/code/personal/sous-remote-kv8 || exit 1
export HF_HUB_OFFLINE=1
G=~/code/personal/sous-spikes/kv8-gate
R=$G/runs
LG=$G/logs
PY=.venv/bin/python
$PY $G/judge_kv8.py valid $G r-s1=64:drafted r-s2=64:drafted on-1=64:drafted off-1=64:off \
  off-2=64:off on-2=64:drafted p-d-g=68:drafted p-nd-g=68:nodraft off-d-g=68:off \
  main-d-g=68:main > $LG/j_valid.txt
$PY $G/judge_kv8.py rounds $G/rounds/r-s1.jsonl 64 > $LG/j_rounds1.txt
$PY $G/judge_kv8.py rounds $G/rounds/r-s2.jsonl 64 > $LG/j_rounds2.txt
$PY $G/analyze_kv8.py $G/rounds/r-s1.jsonl:$R/r-s1/result.json \
  $G/rounds/r-s2.jsonl:$R/r-s2/result.json > $LG/step3.txt 2>&1
$PY $G/judge_kv8.py abba $R on-1 off-1 off-2 on-2 > $LG/j_abba.txt
$PY $G/analyze133.py $R runs on-1 off-1 off-2 on-2 > $LG/abba_runs.md
$PY $G/analyze133.py $R pair on-1 off-1 on-2 off-2 > $LG/abba_pair.md
$PY $G/analyze132.py $R equiv p-d-g p-nd-g > $LG/parity.txt
$PY $G/judge_kv8.py parity $R p-d-g p-nd-g > $LG/j_parity.txt
$PY $G/judge_kv8.py gap $R $G/traces p-d-g p-nd-g > $LG/gap.txt
$PY $G/analyze132.py $R equiv off-d-g main-d-g > $LG/off_main.txt
$PY $G/analyze_suite_kv8.py $G/suite.jsonl > $LG/suite.txt
$PY $G/judge_kv8.py figures $R README.md on-1 off-1 off-2 on-2 > $LG/figures.txt
grep -h model_load $R/on-1/daemon.out $R/off-1/daemon.out $R/p-nd-g/daemon.out $R/main-d-g/daemon.out \
  | sed 's/^.*INFO //' > $LG/load_lines.txt
for f in j_valid j_rounds1 j_rounds2 j_abba j_parity; do tail -n 1 $LG/$f.txt; done
grep -h '^step3:' $LG/step3.txt
head -n 1 $LG/parity.txt $LG/off_main.txt
tail -n 1 $LG/gap.txt
grep -h '^rule:' $LG/suite.txt
K=$LG/ktime.txt; [ -e $LG/ktime-2.txt ] && K=$LG/ktime-2.txt
grep -h -E '^(ktime|floor)' $K
grep -h '^tripwire' $LG/kl.log
grep -h -E '^(notmodel|model-)' $LG/TESTS
```

- [ ] **Step 13: Sync the harness to the M5 and check that it compiles and imports**

```bash
G=~/code/sous-spikes/kv8-gate
chmod +x $G/*.sh
rsync -a -e 'ssh -o BatchMode=yes' --exclude 'runs/' --exclude 'logs/' --exclude 'rounds/' --exclude 'traces/' --exclude 'data/' --exclude '__pycache__/' --exclude 'suite.jsonl' --exclude 'kl.json' $G/ m5pro:code/personal/sous-spikes/kv8-gate/
ssh -o BatchMode=yes m5pro 'S=~/code/personal/sous-spikes; G=$S/kv8-gate; K=~/code/personal/sous-remote-kv8/.venv/bin/python; M=~/code/personal/sous-remote-main/.venv/bin/python; cp $S/tile-132/gate/gate132.py $S/tile-132/gate/analyze132.py $S/tile-132/gate/analyze133.py $S/p157/gate157.py $S/p157/build_replay.py $S/p157/replay-157.json $S/p157/analyze157.py $G/ && mkdir -p $G/logs $G/rounds $G/traces $G/runs && $K -m py_compile $G/*.py && for f in $G/*.sh; do zsh -n $f || exit 1; done && cd $G && $K -c "import wrap_kv8, rounds_kv8, judge_kv8, analyze_kv8, suite_kv8" && $M -c "import gate132, gate157" && grep -c "kv_int8=on," suite_kv8.py && echo HARNESS OK'
```

Expected: `1`, then `HARNESS OK`.
- Importing `rounds_kv8` checks its seven pinned mlx-vlm sources in the branch's environment. An `AssertionError` naming one means the lock has moved off mlx-vlm 0.7.2: stop and report.
- Importing `suite_kv8` checks that the branch's `Arm` accepts `kv_int8`.
- Importing `gate157` under main's interpreter checks that both replay builders load.
- `ktime_kv8.py` and `kl_kv8.py` are compiled, not imported: each loads mlx and the engine at import.

- [ ] **Step 14: Run gate step 1 on the M5**

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/tests-chain.txt zsh $G/tests_m5.sh'
ssh -o BatchMode=yes m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/kv8-gate/logs/model-tileattn.log 540 6'
```

Repeat the `wait_for.py` call until it prints `DONE`; the three jobs take about 1.5 hours. Then:

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate/logs; cat $G/TESTS; grep -c -E "^SKIPPED .*tests/test_(kv8|tileattn)" $G/notmodel.log; for f in notmodel model-kv8 model-tileattn; do grep -E " passed| failed| error" $G/$f.log | tail -n 1; grep -c "daemon stayed idle during the job" $G/$f.log; done'
```

Expected:
- `TESTS` lists `notmodel exit=0`, `model-kv8 exit=0`, `model-tileattn exit=0` and `TESTS DONE`;
- the skip count is `0`: every kv8 and tile `nax` test ran on the M5;
- the non-model summary has passes and deselections only, no `failed` or `error`;
- each model log's summary is `1 passed`, with no skip;
- each idle count is `1`.

- [ ] **Step 15: Run gate step 2, the kernel timings, and read the floor**

```bash
ssh -o BatchMode=yes m5pro 'cd ~/code/personal/sous-remote-kv8 && G=~/code/personal/sous-spikes/kv8-gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/ktime.txt env GPU_LABEL=kv8-ktime python3 $L/gpu_run_m5.py perl -e "alarm shift @ARGV; exec @ARGV" 5400 .venv/bin/python -I $G/ktime_kv8.py'
ssh -o BatchMode=yes m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/kv8-gate/logs/ktime.txt 540 60'
```

Repeat the wait until `DONE`; it takes about 40 minutes. Expected:
- 31 `kernel` lines: every cell from 44K up ends `ok`; `(5, 57000)`, `(1, 57000)` and `(5, 80000)` carry `bar 1.35`; the cells below 44K end `reported`;
- 8 `round` lines ending `ok` (net ≥ 0) and 8 `build` lines;
- `ktime PASS`;
- `floor: N8 stays 1536 (net >= 0 at every context timed)`;
- `daemon stayed idle during the job`.

How to read anything else:
- `ktime NOISY`: an A/A pair moved by more than 1 %. Re-run the same command into `logs/ktime-2.txt`, at another hour if it repeats; Steps 20–22 then read that file in place of `ktime.txt`.
- `ktime FAIL` naming `kernel(5,57000)`: stop; the task head's fallbacks apply.
- `ktime FAIL` naming any other cell: stop and report the table.
- `floor: N8 moves to <b>`: stop; the task head's floor rule applies.
- `floor: kv8 does not pay at the largest context timed`: stop and report the table.

- [ ] **Step 16: Run gate step 3, the per-round runs, and judge it**

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/chain-rounds-outer.txt zsh $G/chain_kv8.sh rounds'
```

**Check the first load early.** About 10 minutes in, `r-s1` has loaded and taken its first turn. Read its load line and the wrapper's lines:

```bash
ssh -o BatchMode=yes m5pro 'R=~/code/personal/sous-spikes/kv8-gate/runs/r-s1; grep -h model_load $R/daemon.out | sed "s/^.*INFO //"; grep -h -E "kv8gate: (kv8.enable|rounds installed)" $R/daemon.out | cut -c1-240'
```

Expected:
- The load line ends `projection_kernel=active projection_kernel_probe_s=<s> draft_block=5 kv_int8=active kv_int8_probe_s=<s> kv_int8_floor=1536 kv_int8_ceiling=170241` (the ceiling at the drivers' 262144 window and S = 20).
- `kv8gate: kv8.enable seconds=<s> state=active reason=None`.
- `kv8gate: rounds installed {... 'arms': ['5b', '5q'], 'seed': 1811, 'configured_block': 5, ...}`.

If any reads otherwise, stop the chain and report:

```bash
ssh -o BatchMode=yes m5pro 'pkill -TERM -f "[k]v8-gate/chain_kv8.sh"; pkill -TERM -f "[k]v8-gate/run_kv8.sh"; pkill -TERM -f "[k]v8-gate/gate1"'
```

The drivers unwind on TERM: they stop their daemon's process group and delete the run's forks. The bracket in each pattern keeps `pkill` from matching the shell that runs it.

Wait for the phase, about 2 hours:

```bash
ssh -o BatchMode=yes m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/kv8-gate/logs/r-s2.log 540 6; cat ~/code/personal/sous-spikes/kv8-gate/logs/ROUNDS_DONE 2>/dev/null'
```

Repeat until `ROUNDS_DONE` lists `r-s1 exit=0` and `r-s2 exit=0`. Then judge the step:

```bash
ssh -o BatchMode=yes m5pro 'cd ~/code/personal/sous-remote-kv8 && G=~/code/personal/sous-spikes/kv8-gate && .venv/bin/python $G/judge_kv8.py valid $G r-s1=64:drafted r-s2=64:drafted && .venv/bin/python $G/judge_kv8.py rounds $G/rounds/r-s1.jsonl 64 && .venv/bin/python $G/judge_kv8.py rounds $G/rounds/r-s2.jsonl 64 && .venv/bin/python $G/analyze_kv8.py $G/rounds/r-s1.jsonl:$G/runs/r-s1/result.json $G/rounds/r-s2.jsonl:$G/runs/r-s2/result.json > $G/logs/step3.txt 2>&1; sed -n "/^CONTEXT HISTOGRAM/,\$p" $G/logs/step3.txt'
```

Expected:
- `valid: PASS`, then `rounds: PASS` twice: both arms drew rounds, no off round served a kv8 row, and on rounds did;
- the context histogram, the strata, then `step3: SHIP`.

`step3: CLOSE` or `step3: TO THE MAINTAINER` stops the gate (the task head). A run that fails `valid` is re-run under a new label (Step 18's rule) before anything is read.

- [ ] **Step 17: Run gate steps 4 and 5**

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/chain-rest-outer.txt zsh $G/chain_kv8.sh rest'
```

This runs 8 runs: about 3.5 hours of ABBA, then about 4 hours of parity. To follow progress:

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate/logs; cat $G/ABBA_DONE.part $G/ABBA_DONE $G/PARITY_DONE.part $G/PARITY_DONE 2>/dev/null; tail -n 2 "$(ls -t $G/*.log | head -1)"'
```

The chain is done when `PARITY_DONE` lists four `exit=0` lines:

```bash
ssh -o BatchMode=yes m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/kv8-gate/logs/main-d-g.log 540 6'
```

A `port 8384 still busy`, `process left` or `phase ABBA incomplete` line stops the chain. Report it before going on.

- [ ] **Step 18: Check that every replay run counts**

```bash
ssh -o BatchMode=yes m5pro 'cd ~/code/personal/sous-remote-kv8 && G=~/code/personal/sous-spikes/kv8-gate && .venv/bin/python $G/judge_kv8.py valid $G r-s1=64:drafted r-s2=64:drafted on-1=64:drafted off-1=64:off off-2=64:off on-2=64:drafted p-d-g=68:drafted p-nd-g=68:nodraft off-d-g=68:off main-d-g=68:main | tee $G/logs/validity.txt | tail -n 1'
```

Expected: `valid: PASS`. Each run:
- had the daemon idle throughout and `exit=0`;
- answered every turn with status 200 on one model load, tile and projection kernel active;
- left no forks;
- on the branch, had `kv8.enable` timed by the wrapper;
- showed its kind's load line and counters: kv8 active with its floor and ceiling and kv8 calls made (with and without the drafter), or `kv_int8=off` with no kv8 call and `kv8=off` on every turn line, or main's line without `kv_int8`.

Re-run any run that is not `ok` under a new label:
- append `b` to its label;
- run it with `run_kv8.sh`, its spec from `chain_kv8.sh` with the new label, and a done-file of its own;
- use the new label wherever the old one appears in `judge_all.sh`.

Its rounds and trace files take the new label too, so they start empty.

- [ ] **Step 19: Run gate step 6, the suite and the KL instrument**

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/quality-outer.txt zsh $G/quality_m5.sh'
```

The suite's four jobs take about 3.5 hours, then the instrument about 6 hours. To follow progress:

```bash
ssh -o BatchMode=yes m5pro 'G=~/code/personal/sous-spikes/kv8-gate/logs; cat $G/QUALITY 2>/dev/null; tail -n 2 "$(ls -t $G/suite-b*.log $G/kl.log 2>/dev/null | head -1)"'
```

**Check the instrument's start.** About 15 minutes after the fourth suite job ends, `kl.log` has its first turn:

```bash
ssh -o BatchMode=yes m5pro 'head -n 6 ~/code/personal/sous-spikes/kv8-gate/logs/kl.log'
```

Expected: `68 turns`, a `kv_int8 {'state': 'active', ...}` line, `restore exact: an undone forward leaves the cache as it found it`, and a `[1/68]` line. `RESTORE INEXACT` stops the instrument: report it, since on and off would then score different prefixes.

Wait until `QUALITY DONE`:

```bash
ssh -o BatchMode=yes m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/kv8-gate/logs/kl.log 540 8; cat ~/code/personal/sous-spikes/kv8-gate/logs/QUALITY'
```

Expected:
- `QUALITY` lists `suite-b0` to `suite-b3` and `kl`, each `exit=0`, then `QUALITY DONE`;
- each suite log has two `== on|off block <b>: 16 runs, released=True, error=None, ...` lines, the on arm's `kv8 calls` showing `decode_kv8` above 0 and the off arm's all 0, and `daemon stayed idle during the job`;
- `kl.log` ends with the per-layer table, `tokens scored ... mean KL(off || on) ..., top-1 agreement ...` and a `tripwire:` line, after `daemon stayed idle during the job`.

- [ ] **Step 20: Judge the gate**

On the M5, run every analysis. They are CPU-only; `analyze132.py equiv` needs the 27B's tokenizer, which is why they run there.

```bash
ssh -o BatchMode=yes m5pro 'zsh ~/code/personal/sous-spikes/kv8-gate/judge_all.sh'
```

Expected:
- `valid: PASS`, `rounds: PASS` twice, `abba: PASS` and `parity-kv8: PASS`;
- `step3: SHIP`;
- `equivalence p-d-g vs p-nd-g: 68/68 identical; cache histories equal on 68/68`;
- `equivalence off-d-g vs main-d-g: 68/68 identical; cache histories equal on 68/68`;
- `gap: 0 differing turns of 68`;
- `rule: ELIGIBLE`;
- `ktime PASS` and `floor: N8 stays 1536 (net >= 0 at every context timed)`;
- `tripwire: PASS (mean KL <x>, bound 0.033)`;
- `notmodel exit=0`, `model-kv8 exit=0`, `model-tileattn exit=0`.

Pull the results back to this Mac. The run homes and the suite's scratch stay behind:

```bash
rsync -a -e 'ssh -o BatchMode=yes' --exclude 'home/' --exclude 'data/' --exclude 'runs/b*/' m5pro:code/personal/sous-spikes/kv8-gate/ ~/code/sous-spikes/kv8-gate/
```

**The verdict.** Ship only if every row passes. Each row names the spec's gate step, the file that decides it and the reading it needs:

| Gate step | Decided by | Ship needs | Otherwise |
|---|---|---|---|
| Preconditions | Step 1, Step 3 | #181 `MERGED`, its squash commit on main, the rebase clean; both M5 trees at the expected heads | stop |
| 1. Tests | `logs/m2-notmodel.txt` and Step 1's `CPU CHECKS OK` (M2); `logs/TESTS`, `logs/notmodel.log`, `logs/model-kv8.log`, `logs/model-tileattn.log` (M5) | M2 summary without `failed` or `error`; three `exit=0`; 0 kv8 or tile skips on the M5; `1 passed` per model file | stop |
| 2. Kernel timings | `logs/ktime.txt` | `ktime PASS` and `floor: N8 stays 1536` | NOISY: re-time; (5, 57K) under 1.35×: the fallbacks; the floor moves: branch change; any other FAIL: stop |
| 3. Speed, primary | `logs/step3.txt`, `logs/j_rounds1.txt`, `logs/j_rounds2.txt` | `step3: SHIP`, `rounds: PASS` twice | `CLOSE`: the lever closes; `TO THE MAINTAINER`: stop |
| 4. Speed, secondary | `logs/j_abba.txt` (`logs/abba_runs.md` and `logs/abba_pair.md` reported) | `abba: PASS`: pooled decode ≥ 1.00; prefill and TTFT each ≤ +3 % (judged separately, which is stricter than the spec's sum); peak memory difference at the largest-context decode within the mirror's bytes + 1 GiB; 0 pressure evictions; `took` and evictions equal; every on turn `served` or `capped` (idle only below N8), every off turn `off`, capped turns listed against L | stop |
| 5. Parity | `logs/parity.txt`, `logs/j_parity.txt`, `logs/off_main.txt` (`logs/gap.txt` on any divergence) | 68/68 identical with equal cache histories on both pairs; `parity-kv8: PASS` | read `gap.txt`, then stop |
| 6(a). Suite | `logs/suite.txt` | `rule: ELIGIBLE` | `KEEP OFF`: kv_int8 does not ship on; `TO THE MAINTAINER`: stop |
| 6(b). KL instrument | `logs/kl.log` | `tripwire: PASS` | to the maintainer with the numbers |
| Validity | `logs/j_valid.txt` | `valid: PASS` | re-run the runs that are not `ok` |

If any row fails, stop and report the numbers and the files above. Open no PR.

- [ ] **Step 21: The README's 57K figures (only after a passing verdict)**

The spec changes the README's performance numbers only after the gate. The one gate figure that measures decode as shipped against kv8 off at the README's context is the ABBA's 32–64K turns, so it decides. `logs/figures.txt` says either that its CI includes 1.00, in which case the README stays, or what the two figures become.

```bash
cat ~/code/sous-spikes/kv8-gate/logs/figures.txt
```

If it prints new figures, edit `README.md` in this worktree:
- line 56, `and about 25 tok/s at 57K on an M5 Pro, sampling as shipped. It prefills`: replace `25` with the printed `about <x> tok/s` value;
- line 150, `estimated 1.2–1.5x at short context and 1.3–1.5x at 51–57K. It drafts and`: replace `1.3–1.5x` with the printed band, keeping the en dash;
- leave `about 31 tok/s with 2K tokens of context` and `1.2–1.5x at short context` alone: no gate run measures short context.

If either figure changed, commit and push:

```bash
git add README.md
git commit -m "docs: update the README's 57K decode figures with the int8 K/V mirror's gain" -m "The merge gate measured decode on the M5 Pro's 64 real subagent turns with
kv_int8 on against off on the branch, sampled, in same-hour ABBA order, and
the 32-64K turns' ratio cleared 1.00. The 57K figure and the gap to Splash
at 51-57K scale with that ratio; the 2K figure and the short-context gap
stay, since no gate run measured short context."
git log -1 --format='%G? %an <%ae> %cn <%ce>%n%B'
git push origin feat/int8-kv-mirror
```

Expected from `git log`: the first character is `G` (a good signature); author and committer are the identity in the global git config; the body is the message above and nothing else, with no trailer crediting anyone.

- [ ] **Step 22: Record the gate and open the PR (only after a passing verdict)**

Write the PR body's fixed head:

```bash
cat > ~/code/sous-spikes/kv8-gate/pr-head.md <<'EOF'
## What and why

Decode and DFlash verify attention on the default 27B now read an int8 copy of the context's
K/V (`src/sous/engine/kv8.py`, `src/sous/engine/kernels/attention_tile_kv8.metal` and
`kv8_quant.metal`) instead of the bf16 cache, from the floor N8 up to a ceiling the window sets.
The attention tile is bandwidth-bound at one and two rows and issue-bound from three rows; int8
K and V halve the bytes it reads, Q is quantised in the kernel, and QK runs on the tensor units'
int8 × int8 → int32 rows. The kernel timings below compare it with the relay tile from #181.

- On by default through `[model].kv_int8`; `false` runs today's paths, bitwise (below).
- Active only where the attention tile is active (the M5 Pro); `unavailable` with the tile's
  reason everywhere else, where nothing changes.
- The mirror exists only inside a generation. Prefill, the prompt cache, forks and the fork
  store stay bf16. K and V are per-token symmetric int8 with fp16 scales. Every verify row reads
  the same codes as its one-row decode, so greedy output with the DFlash2 drafter still equals
  output without it.
- Memory: the mirror is half the bf16 size of the rows it covers and fits inside the existing
  reserve. kv8 serves a row only below a ceiling (84,481 keys at the 131072 window, 170,241 at
  262144), and from there a generation runs on the bf16 tile. The reserve, the slot budget and
  the pressure valve do not change.
- The model-load line gains `kv_int8=` (with `kv_int8_probe_s=`, `kv_int8_floor=` and
  `kv_int8_ceiling=` when active), `/sous/status` a `kv_int8` block, and the turn line `kv8=`
  and `kv8_rows=`. `sous tune` keys suite rows by the setting.

**Fork store.** Landing this cold-starts every on-disk fork store once: `vlm.py` and
`tileattn.py` are hashed into the engine epoch. The two new kernels are excluded from it, so a
later kernel-only edit does not.

Spec: `docs/superpowers/specs/2026-10-12-int8-kv-mirror-design.md`. Builds on #181.
EOF
```

Assemble the body from the head and the gate's outputs:
- `logs/step3.txt` is taken from its context histogram on, because analyze157's install lines carry the rounds files' paths.
- analyze133's tables call the first label of each pair `tile` and the second `main`: here they are `kv_int8` on and off.

```bash
G=~/code/sous-spikes/kv8-gate
K=$G/logs/ktime.txt; [ -e $G/logs/ktime-2.txt ] && K=$G/logs/ktime-2.txt
{
  cat $G/pr-head.md
  printf '\n## Merge gate (M5 Pro, 20 GPU cores; pre-registered in the spec)\n\n### Tests\n\n~~~\n'
  grep -v '^tree ' $G/logs/TESTS
  for f in notmodel model-kv8 model-tileattn; do printf '%s: ' $f; grep -h -E '[0-9]+ passed' $G/logs/$f.log | tail -n 1; done
  printf 'M2 Air, not model: '; grep -h -E '[0-9]+ passed' $G/logs/m2-notmodel.txt | tail -n 1
  printf '~~~\n\n### Load lines\n\n~~~\n'; cat $G/logs/load_lines.txt; printf '~~~\n'
  printf '\n### Kernel timings: kv8 against the relay tile, ms per 16-layer chain\n\n~~~\n'; grep -h -E '^(device|kernel|round|build|ktime|floor)' $K; printf '~~~\n'
  printf '\n### Speed, primary: kv8 on (5q) against off (5b) drawn per round, two sampled seeds, production max_tokens\n\n~~~\n'; sed -n '/^CONTEXT HISTOGRAM/,$p' $G/logs/step3.txt; printf '~~~\n'
  printf '\n### Speed, secondary: same-hour ABBA, kv_int8 on against off, sampled, production max_tokens\n\nIn these tables `tile` is kv_int8 on and `main` is kv_int8 off.\n\n'
  cat $G/logs/abba_runs.md $G/logs/abba_pair.md
  printf '\n~~~\n'; cat $G/logs/j_abba.txt; printf '~~~\n'
  printf '\n### Parity: drafter on against off, greedy, 68 turns to 229K tokens\n\n~~~\n'; cat $G/logs/parity.txt $G/logs/j_parity.txt; tail -n 1 $G/logs/gap.txt; printf '~~~\n'
  printf '\n### kv_int8 = false against main, greedy, 68 turns\n\n~~~\n'; cat $G/logs/off_main.txt; printf '~~~\n'
  printf '\n### Tool-loop suite: 8 tasks x 8 seeds, kv_int8 on against off, paired by prompt and seed\n\n~~~\n'; cat $G/logs/suite.txt; printf '~~~\n'
  printf '\n### Next-token distribution under kv8: KL(off || on), top-1 agreement, attention error per layer\n\n'
  sed -n '/^PER-LAYER/,/^| all, max/p' $G/logs/kl.log
  printf '\n~~~\n'; grep -h -E '^(tokens scored|tripwire)' $G/logs/kl.log; printf '~~~\n'
} > $G/pr-body.md
if grep -n -i -e /Users/ -e /home/ -e "$(id -un)" -e "$(ssh -o BatchMode=yes m5pro id -un)" -e '\.ts\.net' -e 'co-authored-by' -e 'generated with' $G/pr-body.md; then echo "strip these lines before opening the PR"; fi
```

Expected: the check prints nothing. The body carries no person's name, no home path, no machine name and no attribution line.

Open the PR:

```bash
gh pr create --base main --head feat/int8-kv-mirror --title "perf(engine): serve decode and verify attention from an int8 K/V mirror" --body-file ~/code/sous-spikes/kv8-gate/pr-body.md
```

- [ ] **Step 23: Watch CI and clean up**

```bash
gh pr checks feat/int8-kv-mirror --watch --interval 60
```

Expected: all four jobs pass. On CI's runner (macos-15, no tensor units) kv8's `nax` tests skip on `kv8.availability()`, and the quantiser's tests, which are MPP-free, run.

Delete any forks the runs left behind. The drivers remove their own when they exit, and the suite and the instrument write none, so each `find` should print nothing:

```bash
ssh -o BatchMode=yes m5pro 'find ~/code/personal/sous-spikes/kv8-gate/runs ~/code/personal/sous-spikes/kv8-gate/data -path "*/.sous/forks" -type d -prune -print -exec rm -rf {} + 2>/dev/null'
find ~/code/sous-spikes/kv8-gate -path '*/.sous/forks' -type d -prune -print -exec rm -rf {} +
```

Expected: each `find` prints the fork directories it removed, or nothing.


