# Small-Row Projection Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the default 27B's small-row quantized projections through the Splash-port simdgroup MMA kernel. That covers every DFlash verify forward and every target call of ≤ 8 rows outside the verifier, decode included. It is on by default through `[model].projection_kernel`, keeps greedy parity with the drafter on or off, and makes verify block 5 the engine-resolved default wherever the kernel is active.

**Architecture:** A new `src/sous/engine/projkernel.py` holds:
- the kernel entries, the variant choice and `kernel_available()`;
- the tags, the flag and the two hooks;
- the gates, the pins, the warm-up, the probe, `enable()` and `static_reason()`.

It runs over a new `kernels/projection_mma.metal` plus `projection_mma.h`. The hooks sit in two places:
- verify rows reach the kernel through a class-level wrap of `Qwen3_5BatchInvariantForward._linear` and `_linears`;
- every other target call reaches it through a wrap of `nn.QuantizedLinear.__call__` that serves only modules the kernel tagged.

The kernel rides on the attention tile. Engine, config, fork-key and `sous tune` wiring make it visible, switchable and correctly keyed. Unless set explicitly, the block resolves to 5 where the kernel is active and 4 elsewhere.

**Tech Stack:** Python 3.14, mlx 0.32.2, mlx-vlm 0.7.2, mlx-lm 0.31.3 (the lock), Metal (`metal_simdgroup_matrix` via `mx.fast.metal_kernel`; no MPP), pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-10-03-projection-kernel-design.md`. Read it before starting any task.

**Branch:** `feat/projection-kernel`, based on the spec commit a4567ad (branch `docs/projection-kernel-spec`, itself on main 5e97e7f).

## Global Constraints

**Kernel source**
- Files: `src/sous/engine/kernels/projection_mma.h` and `src/sous/engine/kernels/projection_mma.metal`.
  - The header holds `mma8<TA,TB>` and the `PRESUM` switch.
  - The `.metal` file holds three bodies split by exact marker lines: plain, `// STAGED`, `// GROUP_SUMS`.
- Origin: ported from the spike's `kern.py` (plain) and `kern_st.py` (staged and group sums) at `~/code/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/`. Mode h only. Every other mode, `DBG` and every non-default knob are removed.
- Frozen arithmetic constants (they change bits):
  - `F = 4`, `NSG = 1`, `KS = 4`;
  - `GB = 4`, staged only;
  - `CH = 1`, `SPLITS = 1`, `PF = 1`, `SBV = 1`.
- Kernel names: `sous_proj_plain_n{1,2,3,4}`, `sous_proj_staged_n{1,2,3,4}`, `sous_proj_staged_ps_n{1,2,3,4}`, `sous_proj_group_sums`. Every one goes through `int8prefill._kernel(...)` with `compile_options={"math_mode": "safe"}` stated explicitly.

**Arithmetic contract**
- Weights dequantized exactly as `as_type<half2>(nibble | 0x64006400) - 1024`, times bf16 activations widened to fp32.
- 8×8×8 `simdgroup_multiply_accumulate`, accumulated in fp32.
- Per quant group, in order: `acc = fma(dot, scale, acc); acc = fma(sum_x, bias, acc)`. The four K partials are reduced in fixed order `sk0 + sk1 + sk2 + sk3`.
- Rows ≥ T are zeroed and x pointers clamped. The output is bf16.
- Rows are bitwise independent of T and of which linears share the dispatch, and the three variants are bitwise equal to each other.

**Variant choice**
- It depends only on K and `n_sum` (the summed N of the call's linears), never on T:
  - `n_sum >= 100_000` and `K % 1024 == 0`: staged with presum;
  - else `n_sum * K >= 40_000_000` and `K % 1024 == 0`: staged;
  - else plain.
- The plain variant needs `K // 64 >= 4`.

**Inputs the kernel takes**
- `nn.QuantizedLinear` with mode `affine`, `bits == 4`, `group_size == 64`, `biases is not None`, no `"bias"`, and `scales.dtype == mx.bfloat16`.
- x bf16 with 1 ≤ rows ≤ 8 (`MAX_ROWS = 8`), where rows is the product of the leading dimensions, made contiguous.
- 1–4 linears per call, all sharing x.

**Module surface** (`projkernel`)
- `MAX_ROWS = 8` and `REL_RMS_BOUND = 1e-2`.
- `linears(x, weights)`, `linear(x, w, s, b)`, and the seam `mma`.
- `eligible(module) -> bool`, `kernel_available() -> nax.Availability`.
- `enable(...)`, `clear()`, `static_reason(...)`.
- `calls`, `_active`, `_TAG = "_sous_projection_kernel"`, `_HOOK_MARK = "_sous_projection_kernel_hook"`.
- `VALIDATED_PROJ_SOURCES`.

**The flag**
- `projkernel._active` is a plain int, 1 or 0, never a model reference.
- The first statement of every `enable()` sets it to 0. Only an `active` result sets it to 1. `clear()` sets it to 0, and `VLMEngine.unload()` calls `clear()` beside `tileattn.clear()`, before anything there can raise.
- The probe raises it temporarily and always leaves it at 0.

**Tags**
- `enable()` tags every `nn.QuantizedLinear` in `model.language_model.named_modules()` with `object.__setattr__(m, _TAG, True)`. On the default 27B that is 497 modules.
- A failed probe removes the tags from the model it was given. Previous models are never touched.

**Hooks**
- Both are installed once per process, on the first activation. Each is a `functools.wraps` wrapper carrying `_HOOK_MARK`; installation is detected by the mark.
- **Hook 1** wraps `Qwen3_5BatchInvariantForward._linear` and `._linears` (class attributes in `mlx_vlm.models.qwen3_5.speculative_verifier`).
  - Routes when: `_active`, every linear in the call is tagged, x is bf16 with ndim 3.
  - Calls of more than 8 rows are split along T into runs of ≤ 8 and concatenated.
  - Otherwise it calls the original.
- **Hook 2** wraps `mlx.nn.QuantizedLinear.__call__`.
  - Routes when: `_active`, the module is tagged, x is bf16, and 1 ≤ rows ≤ 8.
  - Otherwise it calls what it wrapped.
- Both reach the kernel through the module attribute `projkernel.mma`, looked up at call time, so tests can swap in the stand-in.

**Counters.** `calls` keys exactly:
- `verify_kernel`: hook-1 calls served;
- `verify_split`: hook-1 calls of more than 8 rows that were split;
- `plain_kernel`: hook-2 calls served;
- `declined`: a call met the flag and the tags but was turned away (dtype or ndim).

**Gates**, in order. Each failure gives `unavailable` with its reason:
1. `_active = 0`.
2. `enabled` false gives `off`.
3. `tile_status["state"] != "active"` gives the tile's reason, or `"attention tile off"` when the tile state is `"off"`.
4. The model: `int8prefill._model_type(model) == "qwen3_5"` (the MoE variant refused) and every language-model `QuantizedLinear` eligible.
5. The drafter: `drafter_kind in (None, "dflash")`, and the language model has no attribute `speculative_verify_dflash_hidden`.
6. The pins: `VALIDATED_PROJ_SOURCES` hashes plus `verifyattn.VALIDATED_MLX`.
7. Warm-up: compile every pipeline the model needs, T = 1..8.
8. Tag, install the hooks, probe.

- **Refusal kinds:** steps 3–5 are expected, one INFO log line `projection kernel unavailable: <reason>`. Steps 6–8 and any exception are unexpected, one `warnings.warn` starting `sous: projection kernel unavailable (`, with `stacklevel=3`, and the reason cut by `int8prefill._error_line`; an exception is also logged with its traceback.
- **Pins:** `hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()` following `__wrapped__`, as in `verifyattn._source_digest`. They cover:
  - `Qwen3_5BatchInvariantForward._linear`, `_linears`, `_feed_forward`, `_gated_delta`, `_model`, `__call__`;
  - `Qwen3_5MLP.__call__`, `Qwen3_5GatedDeltaNet.__call__`, `Qwen3_5GatedDeltaNet._project_gates` (resolve the real owner of `_project_gates` in mlx-vlm 0.7.2);
  - `mlx_vlm.generate.ar.generate_step`;
  - `mlx.nn.QuantizedLinear.__call__`;
  - `mlx_vlm.speculative.dflash._dflash_verify` and `_dflash_verify_greedy`.

  The hex values are computed on mlx-vlm 0.7.2 and mlx 0.32.2 when the task runs, and pasted.
- **Status dict:** `{"state": "active"|"unavailable"|"off", "reason": str|None, "probe_seconds": float|None}`.
  - `probe_seconds` is rounded to 0.01 whenever the probe ran.
  - The LM backend sets exactly `{"state": "off", "reason": None}`.

**Config**
- `SousConfig.projection_kernel: bool = True`, listed in `_KNOWN["model"]`. A non-bool warns `sous config: [model].projection_kernel {value!r} must be true or false; using false` and means false.
- `SousConfig.speculative_block_explicit: bool = False`. `load_config` sets it True when `[model].speculative_block_size` is present and not rejected as invalid; a clamped value counts as explicit.
- `config.SPECULATIVE_BLOCK_KERNEL = 5`. `SPECULATIVE_BLOCK_DEFAULT = 4` and `SPECULATIVE_BLOCK_MAX = 5` are unchanged.
- The invalid-block warning ends `using the default`.

**Engine**
- `VLMEngine(..., projection_kernel: bool = False, draft_block_explicit: bool = True)` and `base._default_factory(..., projection_kernel: bool = False, draft_block_explicit: bool = True)`.
- `default_engine_factory` passes `projection_kernel=config.projection_kernel` and `draft_block_explicit=config.speculative_block_explicit`.
- `LMEngine` takes neither.
- **Load order in `VLMEngine.__init__`:** int8 → drafter load → verifyattn → tileattn → **projkernel** → **block resolution and `_pin_block_size`** → draftctx → ForkStore key → `measure_cache_budget`.
- **Block resolution:** when `draft_block_explicit` is false, a drafter loaded and the kernel is `active`, `self._draft_block_size = config.SPECULATIVE_BLOCK_KERNEL` before the pin.

**Visibility**
- The model-load line appends ` projection_kernel=<state>` after the attention_tile tokens, plus ` projection_kernel_probe_s=<x>` only when active. It then appends ` draft_block=<n>` whenever a drafter loaded.
- `EngineManager.status()` gains `projection_kernel: {state, reason}` and `draft_block: int | None` in its engine block.
- `ManagedEngine.projection_kernel_status` and `ManagedEngine.draft_block` read the engine with `getattr` defaults of None, so fakes stay silent on the load line. `VLMEngine.draft_block -> int | None` is the resolved block, or None without a drafter. LMEngine sets `{"state": "off", "reason": None}`.

**Fork store**
- `projkernel.py` joins `forkstore._EPOCH_FILES`. The kernel files are covered by the existing `kernels/*.metal`/`*.h` scan.
- `fork_key_fields` takes a required `proj_status` and records `proj` as `active` or `off` (`unavailable` counts as `off`).
- Landing the change cold-starts every store once. Say so in the PR.

**`sous tune`**
- `_arm` sets `speculative_block_explicit=True` whenever it pins a drafter's block.
- `Arm.projection_kernel` mirrors the config and joins `Arm.suite_key`. `SuiteRun.projection_kernel` joins `SuiteRun.key`; `from_dict` gives older rows the value their runs actually had.
- `runner._check_proj` prints a notice only for `unavailable`.
- The user's current arm, and `decide`'s comparison, use the effective block. That block comes from `projkernel.static_reason(config, model_config, drafter_kind) -> str | None`, where None means the kernel would activate:
  - the NAX platform rule and `gpucores.matched_cores`, with the core count in `tileattn.MEASURED_SPLITS`;
  - `attention_tile` and `projection_kernel` both true;
  - model type `qwen3_5` with attention 24/4/256 from `fetch_model_config`;
  - a drafter kind of `dflash` or none.

**Shared rules**
- Imports: mlx, mlx_lm and mlx_vlm imports are function-local everywhere in `src/` (the lint job runs on ubuntu without mlx), and mlx-vlm internals are resolved with `importlib`. At module level, `projkernel` may import only `int8prefill`, `nax`, `gpucores`, `tileattn` and `verifyattn`, none of which imports mlx at import time.
- Threads: no new threads. Warm-up and probe run inside the engine factory on the `sous-model-load` thread. Their inputs are built with GPU ops and evaluated before the first launch (the CPU-stream deadlock).
- Test modules: new ones start with `mx = pytest.importorskip("mlx.core")` and put `# noqa: E402` on deferred imports. Shared helpers live in `tests/proj_fixtures.py`, imported as `from tests import proj_fixtures as pfx`. Kernel tests skip with `pfx.KERNEL = projkernel.kernel_available()` read once at import, and the first CI run must show them running, not skipped.
- Test hygiene:
  - Tests read counter deltas.
  - Tests that install hooks restore `nn.QuantizedLinear.__call__`, the verifier's `_linear`/`_linears`, `_active`, int8's `_QUANTIZED_LINEAR_WRAPPED` flag and the compile verdict, through `pfx.guard_proj_state`.
  - `slow` marks subprocess and multi-second tests, which still run in CI.
  - Tests never touch `~/.sous`.
- Python and tooling:
  - Python ≥ 3.14 (`except A, B:` without parentheses is valid PEP 758 syntax).
  - Pragmas are `# ty: ignore[rule]`; the type checker is ty, not mypy.
  - `uv sync`, never pip.
  - Before each commit: `uv run ruff format . && uv run ruff check . && uv run ty check`, then the task's tests.
  - CI runs exactly `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .` and `uv lock --check`. `addopts` already has `-q`, so never add another.
- Comments: explain the non-obvious why and never restate code. Code and test comments never cite specs, plans, tasks, steps or issue bookkeeping.
- Privacy: never put the maintainer's name or home path in committed files or GitHub text; say "the maintainer". Plan text uses `~/` paths only.
- Commits: Conventional Commits, imperative lowercase subject, the why in the body, ending with `Co-Authored-By: Claude <noreply@anthropic.com>` (never a model name).
- Records: `docs/superpowers/**` records are never edited. The only file written there is this plan.
- GPU etiquette on the M5: run everything under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py` with the maintainer's daemon idle. Never touch `~/code/personal/sous`, `~/.sous` or the daemon. SSH through the maintainer's `m5pro-lan` host alias.
- Measured compile cost (M5 Pro, the spike kernels, 96 dispatch pipelines): 3.45 s cold (Metal cache busted), 0.02 s warm. Eager warm-up at load is therefore within the spec's bounds.

**Settled while drafting and dry-running**
- Fixtures:
  - `pfx.guard_proj_state(monkeypatch)` is a plain function, as `tfx.guard_tile_state` is, not a pytest fixture.
  - `pfx.tiny_quantized_model(seed=...)` is built in Task 3 and given its final body in Task 4: all 31 of its projections have K ≥ 256, because `eligible()` refuses K = 128.
  - `pfx.proj_ready(monkeypatch, *, real_pins=False)` returns the tile status to pass to `enable()`.
- `projkernel` details:
  - It logs through a module-level `logger`.
  - `linears()` raises ValueError on bad input and never returns None; falling back is the hooks' job.
  - `eligible()` also requires `K // 64 >= 4`.
  - A split hook-1 call counts as Task 3's code counts it.
- Provenance comments in the kernel files and `projkernel.py` match Task 9's notice:
  - incoai/splash at f43509a: the lane map, the register-operand MMA, the W·Xᵀ orientation with its nibble-pair reads and one-group prefetch, and the factored epilogue;
  - paperniuk's d81795d and d2f902e: the exact half weights `nibbles | 0x64006400u` minus 1024, times fp32 activations.
- Drafter blocks: DFlash2-27B's own block policy caps at `min(5, 8) = 5`, so block 0 does not exceed 8 rows with the default drafter. The model test exercises the split at block 12, set on the engine.

---

## File Structure

**Create**
- `src/sous/engine/kernels/projection_mma.h` and `src/sous/engine/kernels/projection_mma.metal` (Task 2).
- `src/sous/engine/projkernel.py` (Tasks 2–4).
- `tests/proj_fixtures.py` (Tasks 2–4): the plain-mlx stand-in `stand_in_mma`, synthetic quantized linears (`rand_linear`, `rand_module`), `KERNEL` and the `kernel` skip marker, `guard_proj_state`, `hooked`, `tiny_quantized_model`, `ACTIVE_TILE` and `proj_ready`.
- `tests/test_projkernel.py` (Tasks 2–4): arithmetic, hooks, gates.
- `tests/test_projkernel_contract.py` (Task 5).
- `tests/test_projkernel_model.py` (Task 10, `-m model`).

**Modify**
- `src/sous/engine/int8prefill.py` (Task 1): `functools.wraps` on `_wrap_quantized_linear`, `_wrap_mlp` and `_wrap_gdn`'s wrappers; the stale "≤ 6 rows" comment.
- `src/sous/engine/kernels/common.h` (Task 2): its first comment names the new kernel.
- `src/sous/config.py`, `src/sous/engine/vlm.py`, `src/sous/engine/lm.py`, `src/sous/engine/base.py`, and the `CountingEngine.__getattr__` comment in `src/sous/tune/suite/runner.py` (Task 6).
- `src/sous/engine/forkstore.py` and the load-line/status code in `base.py` (Task 7).
- `src/sous/tune/arms.py`, `src/sous/tune/decide.py`, `src/sous/tune/suite/runner.py`, `src/sous/tune/__init__.py`, `src/sous/tune/candidates.py` (`Checkpoint.config`) (Task 8).
- `THIRD_PARTY_NOTICES.md`, `src/sous/engine/kernels/__init__.py`, `CLAUDE.md`, `docs/configuration.md`, `docs/observability.md`, `docs/tuning.md`, `docs/how-it-works.md` (Task 9).
- Tests:
  - `tests/test_int8prefill.py` (Task 1);
  - `tests/test_config.py`, `tests/test_engine_base.py`, `tests/test_engine_unloaded.py` (Task 6);
  - `tests/test_forkstore.py`, `tests/test_engine_forkhooks.py`, `tests/test_engine_base.py` (Task 7);
  - `tests/test_tune_arms.py`, `tests/test_tune_runner.py`, `tests/test_tune_decide.py`, `tests/test_tune_main.py`, `tests/test_tune_report.py` (Task 8).

**Out of tree (merge gate, Task 11).** `~/code/sous-spikes/p148/gate/`, rsynced to `~/code/personal/sous-spikes/p148/gate/` on the M5, with a branch worktree `~/code/personal/sous-remote-proj` and a main worktree `~/code/personal/sous-remote-main`, both beside `~/code/personal/sous-remote`.

---

### Task 1: int8's wrappers keep their wrapped functions visible

**Files:**
- Modify: `src/sous/engine/int8prefill.py:28-30` (the `MIN_ROWS` comment), `:375-393` (`_wrap_mlp`), `:396-416` (`_wrap_gdn`), `:419-433` (`_wrap_quantized_linear`)
- Test: `tests/test_int8prefill.py:8-10` (the imports) and a new test after `test_install_wrappers_is_idempotent` (`:482-488`)

**Interfaces:**
- Consumes: nothing.
- Produces: each of int8's three wrappers is a `functools.wraps(orig)` wrapper, so `cls.__dict__["__call__"].__wrapped__ is orig` for `nn.QuantizedLinear`, `Qwen3_5MLP` and `Qwen3_5GatedDeltaNet` (and mlx-lm's `MLP`/`GatedDeltaNet`) once int8 installs. `inspect.getsource` unwraps through `__wrapped__`, so a source pin taken with it (`verifyattn._source_digest`, and Task 4's `VALIDATED_PROJ_SOURCES`) reads mlx's and mlx-vlm's own source whether or not int8 is active. `wraps` also copies the wrapped function's `__dict__`, so a mark attribute on a hook int8 later wraps (Task 3's `_HOOK_MARK`) stays readable on int8's wrapper.

- [ ] **Step 1: Write the failing test**

int8's install is permanent per process and guarded by `_WRAPPED` and `_QUANTIZED_LINEAR_WRAPPED`; earlier tests in the file (the `wrappers` fixture, `test_install_wrappers_is_idempotent`) have usually installed already. The test therefore monkeypatches both guards and the three class attributes to their current values, so it wraps afresh and teardown puts back exactly what it found. The witness is the class's own source (`inspect.getsource(cls)`), which no wrapper installed on the class can change: the unwrapped method's text must be a substring of it.

In `tests/test_int8prefill.py`, replace the imports:

```python
import types

import pytest
```

with:

```python
import hashlib
import inspect
import types

import pytest
```

Then insert after `test_install_wrappers_is_idempotent` (which ends with `    assert Qwen3_5MLP.__call__ is first`):

```python


def _digest(fn):
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def test_wrappers_leave_the_wrapped_source_readable(monkeypatch):
    """A source pin hashes inspect.getsource of a method, which follows __wrapped__:
    without it, an active int8 would read as mlx or mlx-vlm having changed the
    pinned method. The class's own source is the witness no wrapper can change."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet, Qwen3_5MLP

    classes = (nn.QuantizedLinear, Qwen3_5MLP, Qwen3_5GatedDeltaNet)
    # int8 installs once per process behind these two guards: reset both and the
    # three methods, so this test wraps afresh and teardown restores what it found.
    monkeypatch.setattr(i8, "_WRAPPED", set())
    monkeypatch.setattr(i8, "_QUANTIZED_LINEAR_WRAPPED", False)
    before = {cls: cls.__dict__["__call__"] for cls in classes}
    for cls, method in before.items():
        monkeypatch.setattr(cls, "__call__", method)
    i8.install_wrappers([(Qwen3_5MLP, "mlp"), (Qwen3_5GatedDeltaNet, "gdn")])
    for cls in classes:
        wrapper = cls.__dict__["__call__"]
        assert wrapper is not before[cls], cls
        assert wrapper.__wrapped__ is before[cls], cls
        source = inspect.getsource(wrapper)
        assert source.lstrip().startswith("def __call__("), cls
        assert source in inspect.getsource(cls), cls
        assert _digest(wrapper) == _digest(before[cls]), cls
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_int8prefill.py -v -k wrapped_source`
Expected: FAIL, 1 failed: `AttributeError: 'function' object has no attribute '__wrapped__'` at `assert wrapper.__wrapped__ is before[cls], cls`.

- [ ] **Step 3: Wrap with functools.wraps and correct the stale comment**

In `src/sous/engine/int8prefill.py` (which already imports `functools`), replace the `MIN_ROWS` comment and constant:

```python
# Below this many rows the Stage-A pass costs more than the faster GEMM saves
# (crossover measured between 4 and 8 rows; 1.6x by 128). Set well above the
# crossover so decode (1 row) and speculative verify (<= 6 rows) never route.
MIN_ROWS = 128
```

with:

```python
# Below this many rows the Stage-A pass costs more than the faster GEMM saves
# (crossover measured between 4 and 8 rows; 1.6x by 128). Set well above the
# crossover so decode (1 row) and a speculative verify (one drafted block, at
# most 16 rows under the drafter's own policy) never route.
MIN_ROWS = 128
```

In `_wrap_mlp`, replace:

```python
    orig = cls.__call__

    def call(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        if (
```

with:

```python
    orig = cls.__call__

    # wraps() keeps the original reachable through __wrapped__: source pins taken
    # with inspect.getsource follow it, and would otherwise hash this wrapper.
    @functools.wraps(orig)
    def call(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        if (
```

In `_wrap_gdn`, replace:

```python
    orig = cls.__call__

    def call(self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
```

with:

```python
    orig = cls.__call__

    @functools.wraps(orig)
    def call(self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
```

In `_wrap_quantized_linear`, replace:

```python
    orig = nn.QuantizedLinear.__call__

    def call(self: Any, x: Any) -> Any:
```

with:

```python
    orig = nn.QuantizedLinear.__call__

    @functools.wraps(orig)
    def call(self: Any, x: Any) -> Any:
```

and, at the end of the same function, replace:

```python
    nn.QuantizedLinear.__call__ = call
```

with:

```python
    nn.QuantizedLinear.__call__ = call  # ty: ignore[invalid-assignment]
```

ty 0.0.84 types the `wraps` result as `functools._Wrapped[...]` and reports `invalid-assignment` against the attribute's declared `(self, x) -> Unknown`; the two class wrappers already carry the same pragma.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_int8prefill.py -v`
Expected: PASS, 37 passed and 8 skipped on the maintainer's M2 (the 8 are the `nax` GEMM tests). The new test passes whether it runs alone (`before` holds mlx's own methods) or after the `wrappers` fixture (`before` holds int8's earlier wrappers, which now unwrap to the originals too).

Run: `uv run pytest tests/test_tileattn.py tests/test_verifyattn.py tests/test_draftctx.py`
Expected: PASS, 292 passed and 22 skipped on the maintainer's M2 (about two minutes): their source pins hash methods int8 never wraps, and draftctx's `__wrapped__` checks are on its own hooks.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/int8prefill.py tests/test_int8prefill.py
git commit -m "fix(engine): keep int8's wrapped methods visible to inspect" -m "int8 prefill replaces nn.QuantizedLinear.__call__ and the qwen3_5 MLP and
GatedDeltaNet __call__ with plain closures. A source pin hashes
inspect.getsource of a method, which follows __wrapped__ and nothing else,
so with int8 active the pin hashed int8's wrapper and read as mlx or
mlx-vlm having changed the method. functools.wraps sets __wrapped__ and
copies the wrapped function's __dict__, so a mark on a hook int8 wraps
later stays readable too. The MIN_ROWS comment also undercounted verify
rows: a drafted block reaches 16 under the drafter's own policy.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: The kernel, its entries and `kernel_available()`

**Files:**
- Create: `src/sous/engine/kernels/projection_mma.h`
- Create: `src/sous/engine/kernels/projection_mma.metal`
- Modify: `src/sous/engine/kernels/common.h:1-2` (the first comment: a third kernel prepends it)
- Create: `src/sous/engine/projkernel.py`
- Create: `tests/proj_fixtures.py`
- Test: `tests/test_projkernel.py` (create)
- Out of tree (not committed): `~/code/sous-spikes/p148/gate/port_vs_spike.py`

**Interfaces:**
- Consumes:
  - int8prefill: `_kernel(name, input_names, output_names, source, header, *, ensure_row_contiguous=True, compile_options=None)` (cached by name in `int8prefill._kernels`), `_kernel_text(name)` (`functools.cache`d), `_error_line(text)`.
  - `nax.Availability(available: bool, reason: str | None = None)`.
  - The spike's `kern.py`, `kern_st.py` and `verify_proj.py` at `~/code/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/` (Step 12 only).
- Produces:
  - `kernels/projection_mma.h`: `#include <metal_simdgroup_matrix>`, the `PRESUM` switch (`#ifndef PRESUM` / `#define PRESUM 0` / `#endif`) and `mma8<TA, TB>(thread float2 &c, vec<TA, 2> a, vec<TB, 2> b)`.
  - `kernels/projection_mma.metal`: the plain body, a line exactly `// STAGED`, the staged body, a line exactly `// GROUP_SUMS`, the group-sums body. The plain and staged bodies each hold one `__SELECT__` line, which Python replaces with the linear selection; each declares the frozen `constexpr int F = 4;`, `NSG = 1;`, `KS = 4;` and the staged body `GB = 4;`. Template arguments: `TR`, `KD`, `NLIN`, `N0`..`N3`, `CT1`..`CT3` (group sums: `TR`, `KD`). The output is always bf16.
  - In `projkernel`:
    - Public: `MAX_ROWS = 8`, `REL_RMS_BOUND = 1e-2`, `VARIANTS = ("plain", "staged", "staged_ps")`, `variant(k: int, n_sum: int) -> str`, `eligible(module: Any) -> bool`, `linears(x, weights) -> tuple[mx.array, ...]`, `linear(x, w, scales, biases) -> mx.array`, `mma = linears`, `kernel_available() -> nax.Availability`.
    - `linears` raises `ValueError` (it never returns None) for x not bf16, rows (the product of x's leading dimensions) outside 1..8, a linear count outside 1..4, a linear whose K is not x's, or a K the chosen variant cannot split (plain needs K / 64 >= 4, staged K % 1024 == 0). The hooks (Task 3) check dtype, tags and rows before they reach `mma`, so "unsupported input falls through to stock" is the hooks' behaviour; inside `linears` a refusal is a caller's bug.
    - `eligible` is the contract's input rule plus K / 64 >= 4 (the plain body's four K simdgroups): an `nn.QuantizedLinear`, mode `affine`, `bits == 4`, `group_size == 64`, uint32 2-D weights, bf16 2-D scales and biases of one shape, no `"bias"`, any N.
    - Private, used by later tasks and tests: `_MAX_LINEARS = 4`, `_GROUP = 64`, `_COLS = 32`, `_THREADS = 128`, `_KS = 4`, `_STAGED_K = 1024`, `_PRESUM_N = 100_000`, `_STAGED_WORK = 40_000_000`, `_STAGED_MARK = "// STAGED\n"`, `_SUMS_MARK = "// GROUP_SUMS\n"`, `_SELECT = "__SELECT__"`, `_SAFE_MATH = {"math_mode": "safe"}`; `_bodies()`, `_select(nl)`, `_source(kind, nl)`, `_header(presum)` (all but `_select` `functools.cache`d, so the hot path does no string work); `_projection_kernel(kind, nl)` (names `sous_proj_{plain,staged,staged_ps}_n{1..4}`, inputs `x`, then `sums` for `staged_ps`, then `w{i}`, `s{i}`, `b{i}`; outputs `y{i}`); `_group_sums_kernel()` (`sous_proj_group_sums`); `_group_sums(x2)`; `_launch(kind, x2, weights) -> list[mx.array]`, the only place a kernel is dispatched and the seam the probe tests replace; `_kernel_ok = False`; `_relative_rms(got, want) -> float`; `_kernel_probe() -> str | None`. The availability probe is named `_kernel_probe` so it cannot be confused with Task 4's model probe `probe(model)`.
  - In `tests/proj_fixtures.py`: `KERNEL = projkernel.kernel_available()` (read once at import), `kernel = pytest.mark.skipif(not KERNEL.available, reason=f"projection kernel unavailable: {KERNEL.reason}")`, `rand_linear(n, k, seed) -> (w, scales, biases)`, `activations(rows, k, seed) -> mx.array`, `stand_in_mma(x, weights) -> tuple[mx.array, ...]` (raises `ValueError` outside 1..8 bf16 rows, so a hook that forgets to split fails on it too).

The Metal is the spike's mode-h arithmetic, unchanged: only modes f, b and hh, `DBG`, the `CH`, `SPLITS`, `PF` and `SBV` knobs (each removed at its frozen value of 1) and the output-type template argument (always bf16) are gone. F, NSG, KS and GB become `constexpr int` in the bodies: mlx declares integer template arguments `template <int ...>`, so every index expression keeps the type it had in the spike. Step 12 proves the port bitwise equal to the spike on every variant, T = 1..8 and 1..4 linears; on the maintainer's M2 it printed `PORT == SPIKE, bitwise` when this plan was written. The header is `common.h`, then `#define PRESUM 1` for the presummed variant only, then `projection_mma.h`; every kernel is built with `compile_options={"math_mode": "safe"}` and mlx's default `ensure_row_contiguous=True`, which makes x contiguous.

- [ ] **Step 1: Write the kernel-source, variant and eligibility tests**

Create `tests/test_projkernel.py`. These tests need no kernel launch: the build test records `mx.fast.metal_kernel`'s arguments in a fresh `int8prefill._kernels`, so the recording kernels never outlive it.

```python
"""The small-row projection kernel (sous.engine.projkernel): its Metal source, the
variant choice, eligibility, availability, and the arithmetic every routing
decision rests on. The arithmetic tests run wherever the kernel compiles and
probes right (CI's macos-15 GPU included) and skip elsewhere."""

from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sous.engine import int8prefill, projkernel  # noqa: E402 — after the guards

# ---- kernel source ---------------------------------------------------------------


def _text(name: str) -> str:
    return files("sous.engine.kernels").joinpath(name).read_text()


def test_the_kernel_file_holds_three_bodies_split_at_marker_lines():
    text = _text("projection_mma.metal")
    assert text.count("\n// STAGED\n") == 1 and text.count("\n// GROUP_SUMS\n") == 1
    assert "#include" not in text
    plain, staged, sums = projkernel._bodies()
    assert plain.count("__SELECT__") == 1 and staged.count("__SELECT__") == 1
    assert "__SELECT__" not in sums
    assert "#if PRESUM" in staged and "#if PRESUM" not in plain + sums
    # Mode h only: half weights times fp32 activations, no other operand types.
    for body in (plain, staged):
        assert "mma8<half, float>(" in body and body.count("mma8<") == 1
        assert "as_type<half2>(nib | 0x64006400u) - half2(1024.0h)" in body
    assert "mpp::" not in text


def test_the_bodies_freeze_the_constants_the_launch_assumes():
    """F, NSG and KS (and GB when staged) fix the summation order; the Python side
    sizes the grid and the variants' K rules from the same numbers."""
    plain, staged, _ = projkernel._bodies()
    for body in (plain, staged):
        assert "constexpr int F = 4;" in body
        assert "constexpr int NSG = 1;" in body
        assert "constexpr int KS = 4;" in body
    assert "constexpr int GB = 4;" in staged and "GB" not in plain
    assert (projkernel._COLS, projkernel._THREADS) == (8 * 4 * 1, 32 * 1 * 4)
    assert (projkernel._KS, projkernel._STAGED_K) == (4, 64 * 4 * 4)


def test_the_header_holds_mma8_and_the_presum_switch():
    header = _text("projection_mma.h")
    assert "inline void mma8(" in header and "simdgroup_multiply_accumulate" in header
    assert "#ifndef PRESUM\n#define PRESUM 0\n#endif" in header
    assert "MetalPerformancePrimitives" not in header


def test_the_linear_select_is_generated_per_count():
    assert projkernel._select(1) == (
        "#define W_ w0\n#define S_ s0\n#define B_ b0\n#define Y_ y0\n  const uint NL_ = N0;"
    )
    three = projkernel._select(3)
    assert "const device uint32_t *W_ = (L == 0 ? w0 : (L == 1 ? w1 : w2));" in three
    assert "device bfloat *Y_ = (L == 0 ? y0 : (L == 1 ? y1 : y2));" in three
    assert "const uint NL_ = (L == 0 ? N0 : (L == 1 ? N1 : N2));" in three
    four = projkernel._select(4)
    assert "const device bfloat *S_ = (L == 0 ? s0 : (L == 1 ? s1 : (L == 2 ? s2 : s3)));" in four


def test_every_kernel_is_built_with_safe_math_under_its_own_name(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs
        return lambda **launch: []

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    for kind in projkernel.VARIANTS:
        for nl in range(1, 5):
            projkernel._projection_kernel(kind, nl)
    projkernel._group_sums_kernel()
    assert set(made) == {
        f"sous_proj_{kind}_n{nl}" for kind in projkernel.VARIANTS for nl in range(1, 5)
    } | {"sous_proj_group_sums"}
    for name, kwargs in made.items():
        assert kwargs["compile_options"] == {"math_mode": "safe"}, name
        assert kwargs.get("ensure_row_contiguous", True) is True, name
        assert "__SELECT__" not in kwargs["source"], name
        assert kwargs["header"].startswith(_text("common.h")), name
    ps = made["sous_proj_staged_ps_n2"]
    assert ps["input_names"] == ["x", "sums", "w0", "s0", "b0", "w1", "s1", "b1"]
    assert ps["output_names"] == ["y0", "y1"]
    assert ps["header"].index("#define PRESUM 1\n") < ps["header"].index("#ifndef PRESUM")
    assert "#define PRESUM 1" not in made["sous_proj_staged_n2"]["header"]
    assert made["sous_proj_plain_n1"]["input_names"] == ["x", "w0", "s0", "b0"]
    assert made["sous_proj_group_sums"]["output_names"] == ["sums"]


# ---- the variant choice ----------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n_sum", "kind"),
    [
        (5120, 248_320, "staged_ps"),  # lm_head
        (5120, 12_288 + 1024 + 1024, "staged"),  # q+k+v
        (5120, 2 * 17_408, "staged"),  # gate+up
        (5120, 10_240 + 6144 + 48 + 48, "staged"),  # in_proj_qkv+z+b+a
        (17_408, 5120, "staged"),  # down
        (5120, 12_288, "staged"),  # q alone
        (6144, 5120, "plain"),  # o_proj, out_proj
        (5120, 6144, "plain"),  # in_proj_z alone
        (5120, 1024, "plain"),  # k or v alone
        (5120, 48, "plain"),  # in_proj_b or a alone
        (1024, 39_062, "plain"),  # just under the staging work
        (1024, 39_063, "staged"),  # just over it
        (1024, 100_000, "staged_ps"),
        (5184, 248_320, "plain"),  # K % 1024 != 0: plain whatever the width
        (1088, 50_000, "plain"),
    ],
)
def test_the_variant_follows_k_and_the_summed_width(k, n_sum, kind):
    assert projkernel.variant(k, n_sum) == kind


# ---- eligibility -----------------------------------------------------------------


def _qlinear(k, n, *, bias=False, dtype=mx.bfloat16, **quant):
    parent = nn.Sequential(nn.Linear(k, n, bias=bias))
    parent.set_dtype(dtype)
    nn.quantize(parent, **{"group_size": 64, "bits": 4, **quant})
    return parent.layers[0]


def test_eligible_takes_only_what_the_kernel_decodes():
    assert projkernel.eligible(_qlinear(256, 48)), "N need not fill a tile"
    assert projkernel.eligible(_qlinear(1024, 40))
    assert not projkernel.eligible(_qlinear(256, 64, dtype=mx.float16)), "fp16 scales"
    assert not projkernel.eligible(_qlinear(256, 64, dtype=mx.float32)), "fp32 scales"
    assert not projkernel.eligible(_qlinear(256, 64, bits=8))
    assert not projkernel.eligible(_qlinear(256, 64, group_size=128))
    assert not projkernel.eligible(_qlinear(256, 64, group_size=32, mode="mxfp4"))
    assert not projkernel.eligible(_qlinear(256, 64, bias=True))
    assert not projkernel.eligible(_qlinear(128, 64)), "K / 64 below the four-way K split"
    assert not projkernel.eligible(nn.Linear(256, 64)), "not quantized"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_projkernel.py -v`
Expected: FAIL at collection: `ImportError: cannot import name 'projkernel' from 'sous.engine'`.

- [ ] **Step 3: Add the kernel source**

Create `src/sous/engine/kernels/projection_mma.h` exactly as below.

```metal
// The projection kernel's header, prepended by projkernel.py after common.h. The
// staged kernel that reads presummed activations has PRESUM set to 1 ahead of it.
#include <metal_simdgroup_matrix>

#ifndef PRESUM
#define PRESUM 0
#endif

// One 8x8x8 simdgroup_multiply_accumulate on a lane's two elements of each operand,
// in the probed lane map: A[fm][fn..fn+1] (weight column fm, k fn..fn+1),
// B[fm][fn..fn+1] (k fm, activation rows fn and fn+1), C and D[fm][fn..fn+1]. An
// output column reads only its own activation row, and the hardware sums its eight
// products onto C in a fixed sequential order.
template <typename TA, typename TB>
inline void mma8(thread float2 &c, vec<TA, 2> a, vec<TB, 2> b) {
  simdgroup_matrix<TA, 8, 8> A;
  simdgroup_matrix<TB, 8, 8> B;
  simdgroup_float8x8 C, D;
  reinterpret_cast<thread vec<TA, 2> &>(A.thread_elements()) = a;
  reinterpret_cast<thread vec<TB, 2> &>(B.thread_elements()) = b;
  reinterpret_cast<thread float2 &>(C.thread_elements()) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = reinterpret_cast<thread float2 &>(D.thread_elements());
}
```

Create `src/sous/engine/kernels/projection_mma.metal` exactly as below. There is no `#include`: `projkernel` prepends the header. Keep every expression as written, the ones that look redundant at the frozen constants included (`(KS > 1 ? KS - 1 : 1)`, `sgi % NSG`, the clamped `min(r0, uint(TR - 1))`): they are the spike's text, and Step 12 compares against it bit for bit. Never write the selection placeholder's name in a comment: Python replaces it everywhere in a body, and a multi-line replacement inside a `//` comment breaks the source.

```metal
// Small-row affine-Q4 (group 64) projections, Y = X W^T for 1-8 bf16 activation
// rows and 1-4 linears that share them, computed as W X^T in 8x8x8 simdgroup MMAs:
// the weight's output columns in the MMA's rows, the activation rows in its
// columns. Each weight is the exact half (nibble | 0x6400) - 1024 and each
// activation bf16 widened to fp32, so every product is exact; per quant group
// acc = fma(dot, scale, acc), then acc = fma(sum(x), bias, acc); four simdgroups
// split K and their partials are added in a fixed order. Rows past TR are zeroed
// and their reads clamped, and a tile serves one linear, so a row's bits depend on
// neither the row count nor the linears beside it, and the three bodies below
// agree bitwise. The lane map, the register-operand MMA, the W X^T orientation
// with its nibble-pair reads and one-group prefetch, and the factored epilogue
// follow Splash (incoai/splash at f43509a); the exact half weights times fp32
// activations follow its Apple7/8 port (paperniuk/splash, d81795d and d2f902e).
// Both are under the Apache License 2.0; see THIRD_PARTY_NOTICES.md. The K split,
// the staging, the group sums and the multi-linear dispatch are sous's own.
// Prepended by projkernel.py with common.h + projection_mma.h; the Python side
// splits this file at the STAGED and GROUP_SUMS marker lines and fills in the
// linear-selection code.
//
// Plain body. Template: TR (rows, 1..8), KD (K), NLIN, N0..N3 (each linear's N),
// CT1..CT3 (each later linear's first tile). Grid (128, tiles, 1), threadgroup
// (128, 1, 1): one threadgroup per 32-column tile of one linear. A lane (fm, fn)
// holds weight column fm and activation rows fn, fn + 1; within a quant group, MMA
// k index 2c + e at step (h, s) is physical k 16c + 8h + s + 4e.
  constexpr int F = 4;    // 8-column fragments per simdgroup
  constexpr int NSG = 1;  // simdgroups along N
  constexpr int KS = 4;   // simdgroups along K: part of the summation order
  constexpr uint G = KD / 64;
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint sn = sgi % NSG, sk = sgi / NSG;
  uint tgy = threadgroup_position_in_grid.y;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint c = fn >> 1;
  constexpr uint COLS = 8 * F * NSG;
  threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NSG * F * 32];

  // which linear this threadgroup's tile belongs to
  uint L = 0;
  __SELECT__

  const uint base = tgy * COLS + sn * (8 * F);
  const device uint8_t *wcol[F];
  const device bfloat *scol[F];
  const device bfloat *bcol[F];
  for (uint f = 0; f < F; ++f) {
    const uint n = min(base + f * 8 + fm, NL_ - 1);
    wcol[f] = (const device uint8_t *)W_ + ulong(n) * (KD / 2) + c * 8;
    scol[f] = (const device bfloat *)S_ + ulong(n) * G;
    bcol[f] = (const device bfloat *)B_ + ulong(n) * G;
  }
  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;

  float2 acc[F];
  for (uint f = 0; f < F; ++f) acc[f] = float2(0);

  // this simdgroup's quant groups
  const uint g0 = (sk * G) / KS, g1 = ((sk + 1) * G) / KS;

  // weights one group ahead
  uint2 wq[F];
  for (uint f = 0; f < F; ++f)
    wq[f] = (g0 < g1) ? *(const device uint2 *)(wcol[f] + ulong(g0) * 32) : uint2(0);
  for (uint g = g0; g < g1; ++g) {
    uint2 wv[F];
    for (uint f = 0; f < F; ++f) wv[f] = wq[f];
    if (g + 1 < g1)
      for (uint f = 0; f < F; ++f) wq[f] = *(const device uint2 *)(wcol[f] + ulong(g + 1) * 32);

    vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
    vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
    vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
    vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
    if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
    if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
    const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);

    // the group's activation sum per row, in a fixed tree across the fm lanes
    float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                       (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
    sm += simd_shuffle_xor(sm, 2);
    sm += simd_shuffle_xor(sm, 4);
    sm += simd_shuffle_xor(sm, 16);

    float2 dot[F];
    for (uint f = 0; f < F; ++f) dot[f] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
#pragma unroll
        for (uint f = 0; f < F; ++f) {
          const uint nib = ((h ? wv[f].y : wv[f].x) >> (4 * s)) & 0x000F000Fu;
          const float2 bf = h ? float2(xb0[s], xb1[s]) : float2(xa0[s], xa1[s]);
          const half2 a = as_type<half2>(nib | 0x64006400u) - half2(1024.0h);
          mma8<half, float>(dot[f], a, bf);
        }
      }
    }
    for (uint f = 0; f < F; ++f) {
      const float scale = float(scol[f][g]);
      const float bias = float(bcol[f][g]);
      acc[f] = fma(dot[f], scale, acc[f]);
      acc[f] = fma(sm, bias, acc[f]);
    }
  }

  // the K partials, added in simdgroup order
  if (KS > 1) {
    if (sk > 0)
      for (uint f = 0; f < F; ++f) red[(((sk - 1) * NSG + sn) * F + f) * 32 + lane] = acc[f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk > 0) return;
    for (uint k2 = 1; k2 < KS; ++k2)
      for (uint f = 0; f < F; ++f) acc[f] += red[(((k2 - 1) * NSG + sn) * F + f) * 32 + lane];
  }
  for (uint f = 0; f < F; ++f) {
    const uint n = base + f * 8 + fm;
    if (n >= NL_) continue;
    if (v0) Y_[ulong(r0) * NL_ + n] = static_cast<bfloat>(acc[f].x);
    if (v1) Y_[ulong(r1) * NL_ + n] = static_cast<bfloat>(acc[f].y);
  }

// STAGED
// Staged body: the plain body's arithmetic in the same order, with the weights
// brought in through threadgroup memory. Each simdgroup stages GB quant groups of its
// 32 columns with 16-byte loads, one stage ahead, and reads them back in the MMA
// lane map. Needs G % (KS * GB) == 0, so K % 1024 == 0. Template and launch as the
// plain body's; with PRESUM 1 it reads each group's activation sums from "sums"
// (the group-sums body below) instead of computing them.
  constexpr int F = 4;    // 8-column fragments per simdgroup
  constexpr int NSG = 1;  // simdgroups along N
  constexpr int KS = 4;   // simdgroups along K: part of the summation order
  constexpr int GB = 4;   // quant groups per staging stage
  constexpr uint G = KD / 64;
  constexpr uint NSGT = NSG * KS;
  constexpr uint CHUNKS = 8 * F * GB * 2;  // 16-byte chunks per stage per simdgroup
  constexpr uint NLD = CHUNKS / 32;        // uint4 loads per lane per stage
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const uint sn = sgi % NSG, sk = sgi / NSG;
  uint tgy = threadgroup_position_in_grid.y;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  constexpr uint COLS = 8 * F * NSG;
  threadgroup float2 red[(KS > 1 ? KS - 1 : 1) * NSG * F * 32];
  threadgroup uint2 stg_all[NSGT * GB * F * 32];
  threadgroup uint2 *stg = stg_all + sgi * (GB * F * 32);

  // which linear this threadgroup's tile belongs to
  uint L = 0;
  __SELECT__

  const uint base = tgy * COLS + sn * (8 * F);
  const device bfloat *scol[F];
  const device bfloat *bcol[F];
  for (uint f = 0; f < F; ++f) {
    const uint n = min(base + f * 8 + fm, NL_ - 1);
    scol[f] = (const device bfloat *)S_ + ulong(n) * G;
    bcol[f] = (const device bfloat *)B_ + ulong(n) * G;
  }
  // loader map: chunk q = i * 32 + lane -> column q / (2 GB), 16-byte chunk q % (2 GB)
  const device uint8_t *lsrc[NLD];
  uint ldst[NLD];
  for (uint i = 0; i < NLD; ++i) {
    const uint q = i * 32 + lane;
    const uint col = q / (2 * GB), ch = q % (2 * GB);
    const uint n = min(base + col, NL_ - 1);
    lsrc[i] = (const device uint8_t *)W_ + ulong(n) * (KD / 2) + ch * 16;
    // chunk ch = group gi (ch / 2), words 4 (ch & 1) .. +3 = uint2 slots c = 2 (ch & 1), +1
    const uint gi = ch >> 1, f = col >> 3, cm = col & 7;
    ldst[i] = ((gi * F + f) * 8 + cm) * 4 + 2 * (ch & 1);
  }
  const uint rd = fm * 4 + (fn >> 1);  // lane's uint2 slot within a (gi, f) block of 32

  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;

  float2 acc[F];
  for (uint f = 0; f < F; ++f) acc[f] = float2(0);

  const uint g0 = (sk * G) / KS, g1 = ((sk + 1) * G) / KS;  // multiples of GB
  uint4 pre[NLD];
  for (uint i = 0; i < NLD; ++i) pre[i] = *(const device uint4 *)(lsrc[i] + ulong(g0) * 32);
  for (uint gs = g0; gs < g1; gs += GB) {
    simdgroup_barrier(mem_flags::mem_threadgroup);  // previous stage fully read
    for (uint i = 0; i < NLD; ++i) {
      stg[ldst[i]] = pre[i].xy;
      stg[ldst[i] + 1] = pre[i].zw;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (gs + GB < g1)
      for (uint i = 0; i < NLD; ++i) pre[i] = *(const device uint4 *)(lsrc[i] + ulong(gs + GB) * 32);
    for (uint gi = 0; gi < GB; ++gi) {
      const uint g = gs + gi;
      uint2 wv[F];
      for (uint f = 0; f < F; ++f) wv[f] = stg[(gi * F + f) * 32 + rd];

      vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
      vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
      vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
      vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
      if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
      if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
      const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);
#if PRESUM
      const float2 sm = *(const device float2 *)(sums + g * 8 + fn);
#else
      float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                         (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
      sm += simd_shuffle_xor(sm, 2);
      sm += simd_shuffle_xor(sm, 4);
      sm += simd_shuffle_xor(sm, 16);
#endif

      float2 dot[F];
      for (uint f = 0; f < F; ++f) dot[f] = float2(0);
#pragma unroll
      for (uint h = 0; h < 2; ++h) {
#pragma unroll
        for (uint s = 0; s < 4; ++s) {
#pragma unroll
          for (uint f = 0; f < F; ++f) {
            const uint nib = ((h ? wv[f].y : wv[f].x) >> (4 * s)) & 0x000F000Fu;
            const float2 bf = h ? float2(xb0[s], xb1[s]) : float2(xa0[s], xa1[s]);
            const half2 a = as_type<half2>(nib | 0x64006400u) - half2(1024.0h);
            mma8<half, float>(dot[f], a, bf);
          }
        }
      }
      for (uint f = 0; f < F; ++f) {
        const float scale = float(scol[f][g]);
        const float bias = float(bcol[f][g]);
        acc[f] = fma(dot[f], scale, acc[f]);
        acc[f] = fma(sm, bias, acc[f]);
      }
    }
  }

  // the K partials, added in simdgroup order
  if (KS > 1) {
    if (sk > 0)
      for (uint f = 0; f < F; ++f) red[(((sk - 1) * NSG + sn) * F + f) * 32 + lane] = acc[f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk > 0) return;
    for (uint k2 = 1; k2 < KS; ++k2)
      for (uint f = 0; f < F; ++f) acc[f] += red[(((k2 - 1) * NSG + sn) * F + f) * 32 + lane];
  }
  for (uint f = 0; f < F; ++f) {
    const uint n = base + f * 8 + fm;
    if (n >= NL_) continue;
    if (v0) Y_[ulong(r0) * NL_ + n] = static_cast<bfloat>(acc[f].x);
    if (v1) Y_[ulong(r1) * NL_ + n] = static_cast<bfloat>(acc[f].y);
  }

// GROUP_SUMS
// Group-sums body: each quant group's activation sum per row, [K / 64, 8] fp32 with
// rows past TR zero, in exactly the plain body's arithmetic, so the staged body that
// reads it (PRESUM 1) agrees bitwise with the other two. Template: TR, KD. Grid
// (32 * K / 64, 1, 1), threadgroup (32, 1, 1): one simdgroup per group.
  const uint lane = thread_index_in_simdgroup;
  const uint g = threadgroup_position_in_grid.x;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint r0 = fn, r1 = fn + 1;
  const bool v0 = r0 < TR, v1 = r1 < TR;
  const uint koff = 16 * (fm >> 1) + 4 * (fm & 1);
  const device bfloat *x0 = (const device bfloat *)x + ulong(min(r0, uint(TR - 1))) * KD + koff;
  const device bfloat *x1 = (const device bfloat *)x + ulong(min(r1, uint(TR - 1))) * KD + koff;
  vec<bfloat, 4> ra0 = *(const device vec<bfloat, 4> *)(x0 + g * 64);
  vec<bfloat, 4> rb0 = *(const device vec<bfloat, 4> *)(x0 + g * 64 + 8);
  vec<bfloat, 4> ra1 = *(const device vec<bfloat, 4> *)(x1 + g * 64);
  vec<bfloat, 4> rb1 = *(const device vec<bfloat, 4> *)(x1 + g * 64 + 8);
  if (!v0) { ra0 = vec<bfloat, 4>(0); rb0 = vec<bfloat, 4>(0); }
  if (!v1) { ra1 = vec<bfloat, 4>(0); rb1 = vec<bfloat, 4>(0); }
  const float4 xa0 = float4(ra0), xb0 = float4(rb0), xa1 = float4(ra1), xb1 = float4(rb1);
  float2 sm = float2((xa0.x + xa0.y) + (xa0.z + xa0.w) + ((xb0.x + xb0.y) + (xb0.z + xb0.w)),
                     (xa1.x + xa1.y) + (xa1.z + xa1.w) + ((xb1.x + xb1.y) + (xb1.z + xb1.w)));
  sm += simd_shuffle_xor(sm, 2);
  sm += simd_shuffle_xor(sm, 4);
  sm += simd_shuffle_xor(sm, 16);
  if (fm == 0) { sums[g * 8 + fn] = sm.x; sums[g * 8 + fn + 1] = sm.y; }
```

In `src/sous/engine/kernels/common.h`, replace the first two lines:

```c
// Shared by every kernel; prepended by int8prefill.py and tileattn.py as the
// metal_kernel header.
```

with:

```c
// Shared by every kernel; prepended by int8prefill.py, tileattn.py and
// projkernel.py as the metal_kernel header.
```

- [ ] **Step 4: Create the module's source side**

Create `src/sous/engine/projkernel.py` with the constants, the source assembly, the kernel builders, `variant()` and `eligible()`. mlx is imported inside each function: the lint job runs on ubuntu without it.

```python
"""A simdgroup-MMA kernel for qwen3_5's small-row quantized projections.

Affine-Q4 (group 64) projections of 1..8 bf16 activation rows, 1..4 linears that
share the rows in one dispatch, run as 8x8x8 ``simdgroup_matrix`` MMAs
(``kernels/projection_mma.metal``): exact half weights times fp32 activations,
accumulated in fp32, with a fixed-order K split and per-group epilogue. A row's
bits depend on nothing else about the call: not on the row count, not on which
linears share the dispatch, not on which of the three variants served it. The
variants (plain; staged through threadgroup memory; staged with activation sums
presummed by a small prepare kernel) differ only in how they load, so the choice
between them is made by shape for speed and never changes an output. That
independence is what lets verify rows and one-row decode go through the same
kernel and agree bit for bit.

The kernel needs no tensor units and no MPP, so it compiles on any Metal GPU with
half x float simdgroup MMA; ``kernel_available()`` is what its tests skip on. Its
lane map, W x^T orientation and epilogue follow Splash (incoai/splash), and its
exact half-weight decode Splash's Apple7/8 port (paperniuk/splash), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from sous.engine.int8prefill import _kernel, _kernel_text

# Rows one kernel call serves: the MMA's eight columns.
MAX_ROWS = 8
# The kernel against a plain-mlx fp32 reference, in relative RMS. Its own error is
# bf16's rounding floor (~1e-3); a wrong lane map or K order lands near 1.
REL_RMS_BOUND = 1e-2
_MAX_LINEARS = 4
_GROUP = 64
# The Metal bodies' frozen F = 4, NSG = 1, KS = 4 and GB = 4, as the launch needs
# them: 32 columns per threadgroup, 128 threads, K / 64 >= KS for the plain body,
# K % (64 * KS * GB) == 0 for the staged one.
_COLS = 32
_THREADS = 128
_KS = 4
_STAGED_K = 1024
# Shape rules for speed, measured on the M5 Pro's 27B projections; the variants
# agree bitwise, so these never change an output. Presummed activations pay from
# lm_head's width; staging pays once the weights reach about 20 MB.
_PRESUM_N = 100_000
_STAGED_WORK = 40_000_000

VARIANTS = ("plain", "staged", "staged_ps")
# projection_mma.metal holds the plain body, this line, the staged body, the next
# line, then the group-sums body.
_STAGED_MARK = "// STAGED\n"
_SUMS_MARK = "// GROUP_SUMS\n"
# Where each body takes the code that picks its linear.
_SELECT = "__SELECT__"
# mlx's default, stated rather than inherited: the variants agree bitwise only if
# each rounds exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}


@functools.cache
def _bodies() -> tuple[str, str, str]:
    """The plain, staged and group-sums bodies."""
    plain, rest = _kernel_text("projection_mma.metal").split(_STAGED_MARK)
    staged, sums = rest.split(_SUMS_MARK)
    return plain, staged, sums


def _select(nl: int) -> str:
    """The code that points W_, S_, B_, Y_ and NL_ at this threadgroup's linear:
    linear l owns tiles [CT{l}, CT{l+1}), and nested ternaries keep the pointers in
    registers. One linear needs no choice."""
    if nl == 1:
        return "#define W_ w0\n#define S_ s0\n#define B_ b0\n#define Y_ y0\n  const uint NL_ = N0;"

    def pick(prefix: str) -> str:
        choice = f"{prefix}{nl - 1}"
        for i in range(nl - 2, -1, -1):
            choice = f"(L == {i} ? {prefix}{i} : {choice})"
        return choice

    lines = [
        "for (uint l = 1; l < NLIN; ++l) if (tgy >= (l == 1 ? CT1 : l == 2 ? CT2 : CT3)) L = l;",
        "tgy -= (L == 0 ? 0 : L == 1 ? CT1 : L == 2 ? CT2 : CT3);",
        f"const device uint32_t *W_ = {pick('w')};",
        f"const device bfloat *S_ = {pick('s')};",
        f"const device bfloat *B_ = {pick('b')};",
        f"device bfloat *Y_ = {pick('y')};",
        f"const uint NL_ = {pick('N')};",
    ]
    return "\n  ".join(lines)


@functools.cache
def _source(kind: str, nl: int) -> str:
    plain, staged, _ = _bodies()
    return (plain if kind == "plain" else staged).replace(_SELECT, _select(nl))


@functools.cache
def _header(presum: bool) -> str:
    switch = "#define PRESUM 1\n" if presum else ""
    return _kernel_text("common.h") + switch + _kernel_text("projection_mma.h")


def _projection_kernel(kind: str, nl: int) -> Callable[..., list[Any]]:
    """One kernel per variant and linear count: int8prefill's builder caches by name,
    and each count generates a different selection."""
    inputs = ["x", "sums"] if kind == "staged_ps" else ["x"]
    for i in range(nl):
        inputs += [f"w{i}", f"s{i}", f"b{i}"]
    return _kernel(
        f"sous_proj_{kind}_n{nl}",
        inputs,
        [f"y{i}" for i in range(nl)],
        _source(kind, nl),
        _header(kind == "staged_ps"),
        compile_options=_SAFE_MATH,
    )


def _group_sums_kernel() -> Callable[..., list[Any]]:
    return _kernel(
        "sous_proj_group_sums",
        ["x"],
        ["sums"],
        _bodies()[2],
        _header(False),
        compile_options=_SAFE_MATH,
    )


def variant(k: int, n_sum: int) -> str:
    """The variant for K and the summed N of a call's linears: never the row count,
    so every row count of a projection takes the same one."""
    if k % _STAGED_K == 0:
        if n_sum >= _PRESUM_N:
            return "staged_ps"
        if n_sum * k >= _STAGED_WORK:
            return "staged"
    return "plain"


def eligible(module: Any) -> bool:
    """An nn.QuantizedLinear the kernel decodes: affine, 4-bit, group 64, uint32
    weights, bf16 scales and biases, no bias term, and K / 64 at least the kernel's
    four-way K split. Any N: partial tiles are clamped on read and guarded on write."""
    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(module, nn.QuantizedLinear) or "bias" in module:
        return False
    if getattr(module, "mode", None) != "affine":
        return False
    if getattr(module, "bits", None) != 4 or getattr(module, "group_size", None) != _GROUP:
        return False
    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    if weight is None or scales is None or biases is None:
        return False
    if weight.dtype != mx.uint32 or weight.ndim != 2 or scales.ndim != 2:
        return False
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        return False
    if scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        return False
    k = scales.shape[1] * _GROUP
    return weight.shape[1] * 8 == k and k // _GROUP >= _KS
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_projkernel.py -v`
Expected: PASS, 21 passed.

- [ ] **Step 6: Create the shared test helpers**

Create `tests/proj_fixtures.py`. `stand_in_mma` is the plain-mlx model of `mma`'s contract that Tasks 3 to 5 swap into the hooks: each row through its own one-row matmul, each linear on its own, so its rows are T-invariant and group-invariant by construction.

```python
"""Shared helpers for the projection kernel's tests: synthetic quantized linears, a
plain-mlx stand-in for the Metal kernel, and the `kernel` skip marker.

mlx is imported inside each helper, so importing this module needs only what
`projkernel.kernel_available()` itself imports."""

from typing import Any

import pytest

from sous.engine import projkernel

# Read once: a failed probe is not remembered, so each call would compile again.
KERNEL = projkernel.kernel_available()
kernel = pytest.mark.skipif(
    not KERNEL.available, reason=f"projection kernel unavailable: {KERNEL.reason}"
)


def rand_linear(n: int, k: int, seed: int) -> tuple[Any, Any, Any]:
    """An affine-Q4 gs64 projection as mlx stores it, (weight uint32 [N, K/8],
    scales bf16 [N, K/64], biases bf16 [N, K/64]): a bf16 normal * 0.02 quantized
    on the GPU and evaluated."""
    import mlx.core as mx

    dense = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    w, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    mx.eval(w, scales, biases)
    return w, scales, biases


def activations(rows: int, k: int, seed: int) -> Any:
    """[rows, K] bf16 activations, built on the GPU and evaluated."""
    import mlx.core as mx

    x = (mx.random.normal((rows, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    return x


def stand_in_mma(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """A plain-mlx model of projkernel.mma's contract: every linear dequantized in
    fp32 and applied to each row on its own (the same one-row matmul whatever the
    row count), rounded to bf16. A row's output depends on its own activations and
    its own linear and nothing else, the property the routing tests rely on. It
    refuses what the kernel refuses, so a hook that forgets to split a long verify
    call fails here too."""
    import mlx.core as mx

    k = x.shape[-1]
    n_rows = x.size // k
    if x.dtype != mx.bfloat16 or not 1 <= n_rows <= projkernel.MAX_ROWS:
        raise ValueError(f"stand-in takes 1..{projkernel.MAX_ROWS} bf16 rows, got {x.shape}")
    rows = x.reshape(n_rows, k).astype(mx.float32)
    outs = []
    for w, scales, biases in weights:
        dense = mx.dequantize(
            w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=64, bits=4
        )
        y = mx.concatenate([rows[r : r + 1] @ dense.T for r in range(n_rows)], axis=0)
        outs.append(y.astype(mx.bfloat16).reshape(*x.shape[:-1], dense.shape[0]))
    return tuple(outs)
```

- [ ] **Step 7: Write the input-rule, stand-in, availability and arithmetic tests**

In `tests/test_projkernel.py`, replace the imports:

```python
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sous.engine import int8prefill, projkernel  # noqa: E402 — after the guards
```

with:

```python
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
np = pytest.importorskip("numpy")

from sous.engine import int8prefill, nax, projkernel  # noqa: E402 — after the guards
from tests import proj_fixtures as pfx  # noqa: E402
```

numpy is a transitive dependency (mlx-lm, mlx-vlm); the float64 reference is clearer in it than on mlx's CPU stream, and the importorskip keeps the module honest about it.

Append to the end of the file (after `assert not projkernel.eligible(nn.Linear(256, 64)), "not quantized"`). The `forced` fixture sends every call to one variant whatever its shape, so each arithmetic test runs once per variant; the shapes use K = 1024 or 2048, which every variant takes. The probe tests replace `_launch`, so they run on any GPU; the `pfx.kernel` tests run the Metal kernel and skip where `kernel_available()` failed.

```python
# ---- what linears() refuses (before any launch) ----------------------------------


@pytest.fixture
def no_launch(monkeypatch):
    def launched(*args):
        raise AssertionError("a refused call reached the kernel")

    monkeypatch.setattr(projkernel, "_launch", launched)


@pytest.mark.parametrize(
    ("shape", "dtype", "count", "match"),
    [
        ((3, 256), mx.float16, 1, "bf16"),
        ((3, 256), mx.float32, 1, "bf16"),
        ((9, 256), mx.bfloat16, 1, "rows"),
        ((3, 3, 256), mx.bfloat16, 1, "rows"),  # nine rows over two leading dims
        ((0, 256), mx.bfloat16, 1, "rows"),
        ((3, 256), mx.bfloat16, 0, "linears"),
        ((3, 256), mx.bfloat16, 5, "linears"),
        ((3, 512), mx.bfloat16, 1, "K"),
    ],
)
def test_linears_refuses_what_the_kernel_cannot_take(no_launch, shape, dtype, count, match):
    weights = [pfx.rand_linear(64, 256, 0)] * count
    with pytest.raises(ValueError, match=match):
        projkernel.linears(mx.zeros(shape, dtype=dtype), weights)


def test_linears_refuses_a_k_its_variant_cannot_split(no_launch, monkeypatch):
    """The plain body needs K / 64 >= 4 for its four K simdgroups, the staged ones
    K % 1024 == 0 for whole staging stages: a K that fits neither is refused rather
    than read past the end of the weights."""
    with pytest.raises(ValueError, match="K=128 does not fit the plain variant"):
        projkernel.linears(mx.zeros((2, 128), dtype=mx.bfloat16), [pfx.rand_linear(64, 128, 0)])
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: "staged")
    with pytest.raises(ValueError, match="K=1088 does not fit the staged variant"):
        projkernel.linears(mx.zeros((2, 1088), dtype=mx.bfloat16), [pfx.rand_linear(64, 1088, 0)])


# ---- the stand-in ----------------------------------------------------------------


def test_the_stand_in_keeps_rows_and_linears_apart():
    """What the routing tests assume of it: rows do not depend on the row count,
    and a fused call equals its linears one at a time."""
    weights = [pfx.rand_linear(n, 256, 60 + i) for i, n in enumerate((72, 48, 40))]
    x = pfx.activations(8, 256, seed=6)
    full = pfx.stand_in_mma(x[None], weights)
    assert [o.shape for o in full] == [(1, 8, 72), (1, 8, 48), (1, 8, 40)]
    assert all(o.dtype == mx.bfloat16 for o in full)
    for t in range(1, 9):
        part = pfx.stand_in_mma(x[:t], weights[1:2])
        assert mx.array_equal(part[0], full[1][0, :t]).item(), t
    with pytest.raises(ValueError, match="rows"):
        pfx.stand_in_mma(mx.zeros((9, 256), dtype=mx.bfloat16), weights)


# ---- availability ----------------------------------------------------------------


def test_kernel_available_reports_the_probes_reason(monkeypatch):
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    monkeypatch.setattr(projkernel, "_kernel_probe", lambda: "linear 0 is off its reference")
    assert projkernel.kernel_available() == nax.Availability(
        False, "the projection kernel failed: linear 0 is off its reference"
    )


def test_kernel_available_remembers_success_but_not_failure(monkeypatch):
    """One bad moment must not switch the kernel off for a long-lived daemon."""
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    calls = []

    def flaky():
        calls.append(1)
        return "transient" if len(calls) == 1 else None

    monkeypatch.setattr(projkernel, "_kernel_probe", flaky)
    assert not projkernel.kernel_available().available
    assert projkernel.kernel_available() == nax.Availability(True)
    assert projkernel.kernel_available() == nax.Availability(True)
    assert len(calls) == 2


_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:40:7: error: no matching function for call to 'mma8'\n"
    "      mma8<half, float>(dot[f], a, bf);\n"
)


def test_the_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(projkernel, "_launch", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.projkernel"):
        reason = projkernel._kernel_probe()
    assert reason == "program_source:40:7: error: no matching function for call to 'mma8'"
    assert "mma8<half, float>(dot[f], a, bf);" in caplog.text


def test_the_probe_refuses_a_kernel_that_computes_wrong(monkeypatch):
    def zeros(kind, x2, weights):
        return [mx.zeros((x2.shape[0], w.shape[0]), dtype=mx.bfloat16) for w, _, _ in weights]

    monkeypatch.setattr(projkernel, "_launch", zeros)
    reason = projkernel._kernel_probe()
    assert reason is not None and reason.startswith("linear 0 is off its reference")


def test_the_probe_refuses_a_variant_that_disagrees(monkeypatch):
    """Bitwise, not within tolerance: a staged body one rounding away from the
    plain one would break verify against decode."""

    def nudged(kind, x2, weights):
        outs = pfx.stand_in_mma(x2, weights)
        return [o * 1.0078125 if kind == "staged" else o for o in outs]  # about one bf16 ulp up

    monkeypatch.setattr(projkernel, "_launch", nudged)
    assert projkernel._kernel_probe() == "staged differs from plain in linear 0 of a 1-linear call"


def test_the_probe_refuses_a_row_that_depends_on_the_row_count(monkeypatch):
    def row_count(kind, x2, weights):
        outs = pfx.stand_in_mma(x2, weights)
        return list(outs) if x2.shape[0] == projkernel.MAX_ROWS else [o * 2 for o in outs]

    monkeypatch.setattr(projkernel, "_launch", row_count)
    assert projkernel._kernel_probe() == "plain: a row alone differs from the same row among eight"


@pfx.kernel
def test_the_real_probe_passes(monkeypatch):
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    assert projkernel._kernel_probe() is None


# ---- arithmetic (the real kernel) ------------------------------------------------


@pytest.fixture(params=projkernel.VARIANTS)
def forced(request, monkeypatch):
    """Every variant on every shape below: they share one arithmetic, so the choice
    by shape may send any call to any of them. K is a multiple of 1024 wherever this
    is used, which the staged bodies need."""
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: request.param)
    return request.param


@pfx.kernel
@pytest.mark.parametrize("k", [1024, 2048])
def test_rows_do_not_depend_on_the_row_count(forced, k):
    """Each row of a T-row call, T = 1..8, equals the same row of the eight-row
    call and the same row computed alone."""
    weights = [pfx.rand_linear(n, k, i) for i, n in enumerate((72, 48))]
    x = pfx.activations(8, k, seed=1)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        part = projkernel.linears(x[:t], weights)
        for got, want in zip(part, full, strict=True):
            assert got.shape == (t, want.shape[1]) and got.dtype == mx.bfloat16
            assert mx.array_equal(got, want[:t]).item(), (forced, k, t)
    for r in range(8):
        alone = projkernel.linears(x[r : r + 1], weights)
        for got, want in zip(alone, full, strict=True):
            assert mx.array_equal(got, want[r : r + 1]).item(), (forced, k, r)


@pfx.kernel
def test_the_plain_kernel_splits_an_uneven_k_the_same_way_for_every_row_count():
    """K / 64 = 17 does not divide by the four K simdgroups: the split is uneven
    but depends on K alone."""
    weights = [pfx.rand_linear(n, 1088, 10 + i) for i, n in enumerate((40, 48))]
    x = pfx.activations(8, 1088, seed=7)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        for got, want in zip(projkernel.linears(x[:t], weights), full, strict=True):
            assert mx.array_equal(got, want[:t]).item(), t


@pfx.kernel
@pytest.mark.parametrize("ns", [(72, 48), (40, 48, 8), (100, 48, 48, 72)])
def test_a_fused_call_equals_its_linears_one_at_a_time(forced, ns):
    weights = [pfx.rand_linear(n, 1024, 20 + i) for i, n in enumerate(ns)]
    x = pfx.activations(5, 1024, seed=2)
    fused = projkernel.linears(x, weights)
    for i, w in enumerate(weights):
        assert fused[i].shape == (5, ns[i])
        assert mx.array_equal(fused[i], projkernel.linear(x, *w)).item(), (forced, ns, i)


@pfx.kernel
@pytest.mark.parametrize("t", [1, 3, 8])
def test_the_three_variants_agree_bitwise(monkeypatch, t):
    """K = 2048 gives each K simdgroup two staging stages."""
    weights = [pfx.rand_linear(n, 2048, 30 + i) for i, n in enumerate((136, 48, 40))]
    x = pfx.activations(t, 2048, seed=3)
    outs = {}
    for kind in projkernel.VARIANTS:
        monkeypatch.setattr(projkernel, "variant", lambda k, n_sum, kind=kind: kind)
        outs[kind] = projkernel.linears(x, weights)
    for kind in ("staged", "staged_ps"):
        for got, want in zip(outs[kind], outs["plain"], strict=True):
            assert mx.array_equal(got, want).item(), (kind, t)


@pfx.kernel
def test_leading_dimensions_are_kept():
    weights = [pfx.rand_linear(n, 1024, 40 + i) for i, n in enumerate((72, 48))]
    x = pfx.activations(6, 1024, seed=8)
    flat = projkernel.linears(x, weights)
    for shape in ((1, 6, 1024), (2, 3, 1024)):
        shaped = projkernel.linears(x.reshape(shape), weights)
        for got, want in zip(shaped, flat, strict=True):
            assert got.shape == (*shape[:-1], want.shape[-1])
            assert mx.array_equal(got.reshape(want.shape), want).item()


@pfx.kernel
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_poisoned_row_leaves_the_other_rows_alone(forced, bad):
    weights = [pfx.rand_linear(n, 1024, 50 + i) for i, n in enumerate((72, 48))]
    x = pfx.activations(6, 1024, seed=4)
    clean = projkernel.linears(x, weights)
    rows = mx.arange(6)[:, None]
    cols = mx.arange(1024)[None, :]
    poisoned = mx.where((rows == 2) & (cols == 100), mx.array(bad, dtype=mx.bfloat16), x)
    got = projkernel.linears(poisoned, weights)
    keep = mx.array([0, 1, 3, 4, 5])
    for g, c in zip(got, clean, strict=True):
        assert mx.array_equal(g[keep], c[keep]).item(), (forced, bad)
        assert not mx.all(mx.isfinite(g[2])).item()


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _dequantize_f64(w, scales, biases):
    words = np.array(w)
    shifts = 4 * np.arange(8, dtype=np.uint32)
    codes = ((words[:, :, None] >> shifts) & 0xF).reshape(words.shape[0], -1)
    scale = np.repeat(_f64(scales), 64, axis=1)
    bias = np.repeat(_f64(biases), 64, axis=1)
    return codes.astype(np.float64) * scale + bias


@pfx.kernel
def test_the_kernel_rounds_like_a_float64_reference(forced):
    """Exact products, fp32 accumulation, one rounding to bf16: within half a bf16
    ulp of the float64 result plus fp32 accumulation noise, and correctly rounded
    almost everywhere. The kernel's output is evaluated before the CPU-side
    reference starts."""
    k = 2048
    weights = [pfx.rand_linear(n, k, 70 + i) for i, n in enumerate((136, 48))]
    x = pfx.activations(8, k, seed=5)
    got = projkernel.linears(x, weights)
    mx.eval(got)
    xs = _f64(x)
    for (w, scales, biases), y in zip(weights, got, strict=True):
        dense = _dequantize_f64(w, scales, biases)
        ref = xs @ dense.T
        magnitude = np.abs(xs) @ np.abs(dense).T
        y64 = _f64(y)
        assert np.all(np.abs(y64 - ref) <= np.abs(ref) * 2.0**-8 + magnitude * 2.0**-16)
        rounded = _f64(mx.array(ref.astype(np.float32)).astype(mx.bfloat16))
        assert np.mean(rounded == y64) >= 0.995, forced
```

- [ ] **Step 8: Run the tests to verify they fail**

Run: `uv run pytest tests/test_projkernel.py -v`
Expected: FAIL at collection, importing `tests/proj_fixtures.py`: `AttributeError: module 'sous.engine.projkernel' has no attribute 'kernel_available'`.

- [ ] **Step 9: Add the entries and kernel_available()**

In `src/sous/engine/projkernel.py`, replace the imports:

```python
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from sous.engine.int8prefill import _kernel, _kernel_text
```

with:

```python
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from sous.engine import nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text

logger = logging.getLogger("sous.engine.projkernel")
```

Append to the end of the file (after `return weight.shape[1] * 8 == k and k // _GROUP >= _KS`). `linears()` calls `variant()` and `_launch()` through the module globals at call time, which is how the tests force a variant and how the probe tests replace the launch.

```python
# --------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------


def _group_sums(x2: Any) -> Any:
    """Each quant group's activation sum per row of a contiguous [T, K] bf16 x, as
    [K / 64, 8] fp32 with rows past T zero: the staged body's own sums, computed
    once for every tile of a wide projection."""
    import mlx.core as mx

    rows, k = x2.shape
    (sums,) = _group_sums_kernel()(
        inputs=[x2],
        template=[("TR", rows), ("KD", k)],
        grid=(32 * (k // _GROUP), 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(k // _GROUP, 8)],
        output_dtypes=[mx.float32],
    )
    return sums


def _launch(kind: str, x2: Any, weights: list[tuple[Any, Any, Any]]) -> list[Any]:
    """One dispatch of one variant over [T, K] rows and 1..4 linears; [T, N_i] bf16
    each. Tiles are laid out linear by linear, a partial last tile per linear."""
    import mlx.core as mx

    rows, k = x2.shape
    nl = len(weights)
    ns = [int(w.shape[0]) for w, _, _ in weights]
    ct = [0]
    for n in ns:
        ct.append(ct[-1] + -(-n // _COLS))
    inputs = [x2, _group_sums(x2)] if kind == "staged_ps" else [x2]
    for w, scales, biases in weights:
        inputs += [w, scales, biases]
    template: list[tuple[str, Any]] = [("TR", rows), ("KD", k), ("NLIN", nl)]
    template += [(f"N{i}", ns[i] if i < nl else 0) for i in range(_MAX_LINEARS)]
    template += [(f"CT{i}", ct[i] if i < nl else 1 << 30) for i in (1, 2, 3)]
    return _projection_kernel(kind, nl)(
        inputs=inputs,
        template=template,
        grid=(_THREADS, ct[-1], 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, n) for n in ns],
        output_dtypes=[mx.bfloat16] * nl,
    )


def linears(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """x [..., K] bf16 with 1..MAX_ROWS rows (the product of its leading dimensions)
    through 1..4 affine-Q4 gs64 linears that share it, given as (weight uint32
    [N, K/8], scales bf16 [N, K/64], biases bf16 [N, K/64]); one [..., N_i] bf16 per
    linear. Raises ValueError for what the kernel cannot take; the hooks check their
    calls before reaching here, so a raise is a caller's bug, never a fallback."""
    import mlx.core as mx

    k = x.shape[-1]
    rows = x.size // k if k else 0
    if x.dtype != mx.bfloat16:
        raise ValueError(f"projection kernel needs bf16 activations, got {x.dtype}")
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"projection kernel takes 1..{MAX_ROWS} rows, got {rows}")
    if not 1 <= len(weights) <= _MAX_LINEARS:
        raise ValueError(f"projection kernel takes 1..{_MAX_LINEARS} linears, got {len(weights)}")
    for w, scales, _ in weights:
        if w.shape[-1] * 8 != k or scales.shape[-1] * _GROUP != k:
            raise ValueError(f"projection kernel: a linear's K is not the activations' K={k}")
    kind = variant(k, sum(int(w.shape[0]) for w, _, _ in weights))
    if k // _GROUP < _KS or (kind != "plain" and k % _STAGED_K):
        raise ValueError(f"projection kernel: K={k} does not fit the {kind} variant")
    outs = _launch(kind, x if x.ndim == 2 else x.reshape(rows, k), weights)
    if x.ndim == 2:
        return tuple(outs)
    return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outs)


def linear(x: Any, w: Any, scales: Any, biases: Any) -> Any:
    """One linear: linears() with a single (w, scales, biases)."""
    return linears(x, [(w, scales, biases)])[0]


# The seam the hooks call through, looked up at call time so tests can swap in a
# plain-mlx stand-in with the same contract.
mma = linears


# --------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------

# The probe's shape: K = 1024 lets every variant run; the four widths cover partial
# tiles, N = 48 (in_proj_b/a) and a single fragment.
_PROBE_K = _STAGED_K
_PROBE_NS = (40, 72, 48, 8)
# The row the probe also computes alone, against its row in the eight-row call.
_PROBE_ROW = 3

_kernel_ok = False


def _relative_rms(got: Any, want: Any) -> float:
    import mlx.core as mx

    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _kernel_probe() -> str | None:
    """Compile and run every variant and linear count once on eight rows, plus each
    variant on one row alone; None when every output is right, else why not. Every
    call must equal the plain kernel's four-linear call bitwise (the variants and
    the linear counts share one arithmetic), the lone row must equal its row of the
    eight, and the plain call must sit within REL_RMS_BOUND of an fp32 dequantized
    reference: an Apple GPU whose MMA lane map differed from the probed one fails
    here instead of serving wrong numbers. The inputs are built with GPU ops and
    evaluated first: a compile failure with a CPU-stream op in flight deadlocks
    inside mlx's exception path (0.32.2) instead of raising."""
    import mlx.core as mx

    try:
        keys = mx.random.split(mx.random.key(0), 1 + len(_PROBE_NS))
        x = mx.random.normal((MAX_ROWS, _PROBE_K), key=keys[0]).astype(mx.bfloat16)
        weights = []
        for i, n in enumerate(_PROBE_NS):
            dense = mx.random.normal((n, _PROBE_K), key=keys[1 + i]) * 0.02
            w, scales, biases = mx.quantize(dense.astype(mx.bfloat16), group_size=_GROUP, bits=4)
            weights.append((w, scales, biases))
        refs = [
            x.astype(mx.float32)
            @ mx.dequantize(
                w, s.astype(mx.float32), b.astype(mx.float32), group_size=_GROUP, bits=4
            ).T
            for w, s, b in weights
        ]
        mx.eval(x, weights, refs)
        base = _launch("plain", x, weights)
        mx.eval(base)
        for i, (got, ref) in enumerate(zip(base, refs, strict=True)):
            error = _relative_rms(got, ref)
            if not error <= REL_RMS_BOUND:
                return f"linear {i} is off its reference by relative RMS {error:.3g}"
        for kind in VARIANTS:
            for nl in range(1, _MAX_LINEARS + 1):
                outs = _launch(kind, x, weights[:nl])
                mx.eval(outs)
                for i, out in enumerate(outs):
                    if not mx.array_equal(out, base[i]).item():
                        return f"{kind} differs from plain in linear {i} of a {nl}-linear call"
            row = _PROBE_ROW
            (alone,) = _launch(kind, x[row : row + 1], weights[:1])
            if not mx.array_equal(alone, base[0][row : row + 1]).item():
                return f"{kind}: a row alone differs from the same row among eight"
    except Exception as e:  # noqa: BLE001 — any failure means the kernel cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("projection kernel: the kernel failed to build or run:\n%s", e)
        return _error_line(str(e))
    return None


def kernel_available() -> nax.Availability:
    """Whether the kernel compiles and computes right on this GPU. Only success is
    remembered: a failure is probed again on the next call, so one bad moment
    cannot switch the kernel off for a long-lived daemon."""
    global _kernel_ok
    if _kernel_ok:
        return nax.Availability(True)
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return nax.Availability(False, f"mlx unavailable: {e}")
    if not mx.metal.is_available():
        return nax.Availability(False, "Metal unavailable")
    failed = _kernel_probe()
    if failed is not None:
        return nax.Availability(False, f"the projection kernel failed: {failed}")
    _kernel_ok = True
    return nax.Availability(True)
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `uv run pytest tests/test_projkernel.py -v -rs`
Expected: PASS, 67 passed on the maintainer's M2 (macOS 15.5, MSL 3.2): 37 that need no kernel and 30 `pfx.kernel` tests, none skipped; a few seconds the first time Metal compiles the pipelines, under one once it has them cached. Where `kernel_available()` fails, the 30 skip with `projection kernel unavailable: the projection kernel failed: <reason>`; CI's macos-15 GPU must run them, which the first CI run of the branch shows.

- [ ] **Step 11: Run the neighbouring tests**

Run: `uv run pytest tests/test_int8prefill.py tests/test_forkstore.py tests/test_nax.py`
Expected: PASS, 132 passed and 8 skipped (int8's `nax` tests). `test_kernel_sources_ship_as_package_files` still holds: no MPP include reaches `common.h`. The two new kernel files and the `common.h` comment change `forkstore.engine_epoch()`, which hashes every `kernels/*.metal` and `*.h`; landing the branch cold-starts every fork store once, which Task 7 and the PR text already account for.

- [ ] **Step 12: Prove the port equals the spike, bit for bit (out of tree)**

Create `~/code/sous-spikes/p148/gate/port_vs_spike.py` (not committed; it uses synthetic weights; Task 11 reruns it on the M5 Pro beside `porteq.py`, which checks the real weights):

```python
"""The in-tree projection kernel against the spike's verify_proj, bitwise, on
synthetic weights: every variant, T = 1..8, 1..4 linears. Out of tree; run from a
sous checkout with `uv run python <this file>`. SPIKE_DIR overrides the location of
the spike's projection-kernels directory."""

import os
import sys
from pathlib import Path

import mlx.core as mx

SPIKE_DIR = Path(
    os.environ.get(
        "SPIKE_DIR", "~/code/sous-spikes/splash-port-2026-09-24/m5/projection-kernels"
    )
).expanduser()
sys.path.insert(0, str(SPIKE_DIR))

import kern  # noqa: E402
import kern_st  # noqa: E402
import verify_proj as vp  # noqa: E402

from sous.engine import projkernel  # noqa: E402

CASES = {
    "K1024 partial tiles": (1024, [40, 72, 48, 8]),
    "K2048 two stages": (2048, [136, 48, 40, 100]),
    "K1088 plain only": (1088, [72, 48]),
    "K5120 q+k+v shape": (5120, [3072, 1024, 1024]),
    "K5120 wide (presum)": (5120, [100352]),
    "K6144 o_proj shape": (6144, [5120]),
    "K5120 qkv+z+b+a shape": (5120, [10240, 6144, 48, 48]),
}


def quantized(n, k, seed):
    dense = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(dense, group_size=64, bits=4)


def main() -> int:
    info = mx.device_info()
    print(info.get("device_name"), info.get("architecture"), mx.__version__, flush=True)
    ok = True
    for name, (k, ns) in CASES.items():
        weights = [quantized(n, k, i) for i, n in enumerate(ns)]
        mx.eval(weights)
        for t in range(1, 9):
            x = (mx.random.normal((t, k), key=mx.random.key(50 + t)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            for nl in range(1, len(ns) + 1):
                sub = weights[:nl]
                pairs = [
                    ("chosen", projkernel.linears(x[None], sub), vp.mma_linears(x[None], sub)),
                    ("plain", projkernel._launch("plain", x, sub), kern.mma_linears(x, sub, **vp.SMALL)),
                ]
                if k % 1024 == 0:
                    pairs.append(("staged", projkernel._launch("staged", x, sub), kern_st.st_linears(x, sub, **vp.ST)))
                    pairs.append(("staged_ps", projkernel._launch("staged_ps", x, sub), kern_st.st_linears(x, sub, **vp.ST_LM)))
                for tag, ours, spike in pairs:
                    mx.eval(ours, spike)
                    if not all(mx.array_equal(a, b).item() for a, b in zip(ours, spike, strict=True)):
                        ok = False
                        print(f"DIFFERS: {name} T={t} linears={nl} {tag}", flush=True)
        print(f"{name}: checked", flush=True)
    print("PORT == SPIKE, bitwise" if ok else "PORT DIFFERS FROM SPIKE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Run (from the repo root): `uv run python ~/code/sous-spikes/p148/gate/port_vs_spike.py`
Expected: one line per case ending `checked`, then `PORT == SPIKE, bitwise`, exit code 0; about 70 s on the M2. Any `DIFFERS:` line means the port changed the arithmetic: fix the Metal, never the expectation. Task 11 runs it on the M5 Pro with `SPIKE_DIR` pointed at the spike's copy there.

- [ ] **Step 13: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/kernels/projection_mma.h src/sous/engine/kernels/projection_mma.metal src/sous/engine/kernels/common.h src/sous/engine/projkernel.py tests/proj_fixtures.py tests/test_projkernel.py
git commit -m "feat(engine): add the small-row projection kernel and its entries" -m "mlx-vlm's exact verifier keeps each verify row equal to one-row decode,
but its projections cost more with every row: 76 ms per 27B forward at 4
rows, 99 at 5, 179 at 8. A simdgroup MMA over exact half weights and fp32
activations, with a fixed-order K split and per-group epilogue, stays at
75-79 ms from 1 to 8 rows, and a row's bits depend on neither the row count
nor the linears sharing the dispatch, so verify rows and decode can share
it exactly. Three variants (plain, staged, staged with presummed
activations) agree bitwise and are chosen by shape for speed only.
kernel_available() compiles and checks every variant, so the arithmetic
tests skip on a GPU whose MMA lane map differs instead of failing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Tags, the flag, the two hooks and the counters

**Files:**
- Modify: `src/sous/engine/projkernel.py` (imports; the routing section appended after Task 2's code)
- Modify: `tests/proj_fixtures.py` (imports; `tiny_quantized_model`, `guard_proj_state` and `hooked` appended)
- Test: `tests/test_projkernel.py` (imports; the routing tests appended)

**Interfaces:**
- Consumes:
  - Task 1: int8's `nn.QuantizedLinear.__call__` wrapper is a `functools.wraps` wrapper, so it carries `__wrapped__` and a copy of the wrapped function's `__dict__` (the mark).
  - Task 2:
    - `projkernel.mma`, the seam: `mma(x, weights) -> tuple[mx.array, ...]`, x `[..., K]` bf16 with 1–8 rows, weights 1–4 `(w, scales, biases)` triples;
    - `projkernel.MAX_ROWS = 8` and `projkernel._kernel_ok`;
    - `pfx.stand_in_mma(x, weights)`, with `mma`'s contract and rows bitwise independent of T and of the linears beside them, as the kernel's are;
    - `pfx.rand_linear(n, k, seed) -> (w, scales, biases)`: an affine-Q4 gs64 projection as mlx stores it, bf16 scales and biases;
    - `projkernel._MAX_LINEARS = 4`.
  - int8prefill (existing): `_root(model)`, `_TAG`, `_QUANTIZED_LINEAR_WRAPPED`, `install_wrappers(classes)`, and `linear_int8(linear, x, stage=None)`, the seam its `nn.QuantizedLinear.__call__` wrapper reads at call time.
  - tile_fixtures (existing): `tiny_language_model(*, seed: int = 0)`.
- Produces:
  - In projkernel:
    - `_TAG = "_sous_projection_kernel"`, `_HOOK_MARK = "_sous_projection_kernel_hook"`, `_active = 0`, `_VERIFIER_MODULE = "mlx_vlm.models.qwen3_5.speculative_verifier"`.
    - `calls: dict[str, int]` with exactly the keys `verify_kernel`, `verify_split`, `plain_kernel` and `declined`, all starting at 0.
    - `_verifier_class() -> Any`: mlx-vlm's `Qwen3_5BatchInvariantForward`, resolved with importlib at call time.
    - `_tagged(module: Any) -> bool`.
    - `tag(model: Any) -> int`: every distinct `nn.QuantizedLinear` under `int8prefill._root(model)` (`model.language_model` for an mlx-vlm model) tagged with `object.__setattr__(m, _TAG, True)`; returns how many (31 on the tiny model, 497 on the 27B). It checks no eligibility: Task 4's `enable()` refuses a model with any ineligible linear before it tags.
    - `untag(model: Any) -> None`: every module under `_root(model)` that carries the tag gets `False`.
    - `clear() -> None`: `_active = 0`.
    - `_run_mma(x, weights) -> tuple`: `mma` read from the module at call time, at most `_MAX_LINEARS` linears per call. (Task 2's `_launch(kind, x2, weights)` keeps its name: it is the one place a kernel is dispatched.)
    - `_serve_verify(linears, x) -> tuple | None` and `_serve_plain(module, x) -> mx.array | None`.
    - `_wrap_linear(original)`, `_wrap_linears(original)` and `_wrap_call(original)`, each `(Callable[..., Any]) -> Callable[..., Any]`: `functools.wraps` wrappers with `_HOOK_MARK` set True. Each reads `_active` first and otherwise calls `original` with the call's own arguments.
    - `install_hooks() -> None`: wraps `Qwen3_5BatchInvariantForward._linear` and `._linears` (class attributes) and `nn.QuantizedLinear.__call__`, each unless it already carries the mark.
  - Routing and counting rules:
    - **The verifier hook** (`_linear`, `_linears`) serves a call when `_active` is set, the call names at least one linear, every linear is tagged, x is bf16 with ndim 3, the batch is at most `MAX_ROWS` and T ≥ 1. A served call adds 1 to `verify_kernel`. A call of more than `MAX_ROWS // batch` rows along T is cut into runs of that many, concatenated on axis 1, and adds 1 to `verify_split` as well.
    - **The module hook** (`nn.QuantizedLinear.__call__`) serves a call when `_active` is set, the module is tagged, rows (the product of the leading dimensions) is 1–8, and x is bf16 with at least two dimensions. A served call adds 1 to `plain_kernel`. Rows outside 1–8 (prefill chunks, int8's calls of 128 rows and more) are handed on uncounted.
    - **`declined`**: the flag is set and every linear is tagged, but the call is refused for its dtype or shape: another dtype, the wrong rank, a verify batch wider than `MAX_ROWS`, or a verify of no rows. Calls on untagged modules and calls while the flag is 0 are never counted.
  - In `tests/proj_fixtures.py`:
    - `rand_module(n: int, k: int, seed: int) -> Any`: `rand_linear`'s projection as an `nn.QuantizedLinear` (affine, 4-bit, group size 64, bf16 scales and biases, no `bias`);
    - `tiny_quantized_model(*, seed: int = 0) -> Any`: tile_fixtures' tiny qwen3_5 language model, quantized affine 4-bit at group size 64.
    - `guard_proj_state(monkeypatch: pytest.MonkeyPatch) -> None`.
    - `hooked(monkeypatch: pytest.MonkeyPatch, *, active: int = 1) -> None`.

`guard_proj_state` takes `monkeypatch`, as `tile_fixtures.guard_tile_state` does, rather than being a pytest fixture: a fixture defined in a helper module is not found by tests that import the module as `pfx`.

- [ ] **Step 1: Add the shared helpers**

In `tests/proj_fixtures.py`, add each of these import lines that Task 2's import block does not already have. `int8prefill` joins Task 2's `from sous.engine import ...` line; `uv run ruff check --fix tests/proj_fixtures.py` puts the block back in isort order.

```python
from typing import Any

import pytest

from sous.engine import int8prefill, projkernel
from tests import tile_fixtures as tfx
```

Append to the end of the file. `rand_module` wraps Task 2's `rand_linear` triple in the module the hooks see; `guard_proj_state` registers every restore before anything is installed, so a test can then install int8's wrapper and the hooks in either order. `hooked` installs through `install_hooks()` itself, so the tests drive the installer the engine uses, and the guard undoes it.

```python
def rand_module(n: int, k: int, seed: int) -> Any:
    """rand_linear's projection as the module a model holds: an nn.QuantizedLinear,
    affine 4-bit at group size 64, bf16 scales and biases, no bias term."""
    import mlx.nn as nn

    module = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    module.weight, module.scales, module.biases = rand_linear(n, k, seed)
    return module


def tiny_quantized_model(*, seed: int = 0) -> Any:
    """tile_fixtures' tiny qwen3_5 language model quantized as the 27B is: every
    linear affine 4-bit at group size 64 with bf16 scales and biases, lm_head
    included (31 QuantizedLinear modules), the embedding a QuantizedEmbedding.
    Layers 0 and 2 are GatedDeltaNet, 1 and 3 full attention."""
    import mlx.core as mx
    import mlx.nn as nn

    lm = tfx.tiny_language_model(seed=seed)
    nn.quantize(lm, group_size=64, bits=4)
    mx.eval(lm.parameters())
    return lm


def guard_proj_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers restores for what install_hooks(), enable() and the probe leave
    behind for the rest of the process: nn.QuantizedLinear.__call__ and the
    verifier's _linear and _linears (the hooks), the flag, int8's install flag (a
    test may install int8's wrapper in either order with ours) and the remembered
    compile verdict. A test that installs or activates the kernel then cannot
    leak any of them into the next."""
    import mlx.nn as nn

    verifier = projkernel._verifier_class()
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", nn.QuantizedLinear.__call__)
    monkeypatch.setattr(verifier, "_linear", verifier._linear)
    monkeypatch.setattr(verifier, "_linears", verifier._linears)
    monkeypatch.setattr(projkernel, "_active", projkernel._active)
    monkeypatch.setattr(
        int8prefill, "_QUANTIZED_LINEAR_WRAPPED", int8prefill._QUANTIZED_LINEAR_WRAPPED
    )
    monkeypatch.setattr(projkernel, "_kernel_ok", projkernel._kernel_ok)


def hooked(monkeypatch: pytest.MonkeyPatch, *, active: int = 1) -> None:
    """Both hooks installed and the flag at `active` for this test only, with the
    stand-in as the kernel; guard_proj_state's restores undo all of it."""
    guard_proj_state(monkeypatch)
    projkernel.install_hooks()
    monkeypatch.setattr(projkernel, "_active", active)
    monkeypatch.setattr(projkernel, "mma", stand_in_mma)
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_projkernel.py`, the header becomes the block below. Task 2's header already has every line from `from importlib.resources import files` down (`nn`, `np`, `int8prefill`, `nax` and `pfx` included); the three new lines are `import importlib`, `import inspect` and `import types`, above `from importlib.resources import files` in isort order.

```python
import importlib
import inspect
import types
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
np = pytest.importorskip("numpy")

from sous.engine import int8prefill, nax, projkernel  # noqa: E402 — after the guards
from tests import proj_fixtures as pfx  # noqa: E402
```

Append to the end of the file. How the tests are built:
- **Hand-on cases** are checked against a recording original: the call's own arguments must arrive and its answer come back. They need neither mlx-vlm's ops nor a kernel, and `mma` is replaced by a function that fails the test if reached.
- **Served cases** are compared with the stand-in. The split and verify-against-plain cases compare rows computed in one call with the same rows computed alone. That is exact because the stand-in's rows, like the kernel's, depend on nothing else in the call.
- **int8** is reached through `int8prefill.linear_int8`, the seam its wrapper reads at call time, replaced by a recorder, so the test needs no tensor units.
- **The drafter** is mlx-vlm's real `DFlashDraftModel` with its real `bind()`, quantized as the engine quantizes one.
- **Other families** are mlx-vlm 0.7.2's three subclasses of `Qwen3_5BatchInvariantForward`. gemma4's and qwen4_exp's inherit `_linear` and `_linears`; nemotron_h's overrides both but reaches them through `super()` for bf16.

```python
# ---- routing: the tags, the flag and the two hooks ---------------------------------


def _proj_counts_since(before):
    """The counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in projkernel.calls.items() if n != before[k]}


def _acts(*shape, dtype=None, seed=0):
    """Random activations of `shape`, bf16 unless `dtype` says otherwise."""
    x = mx.random.normal(shape, key=mx.random.key(seed))
    return x.astype(mx.bfloat16 if dtype is None else dtype)


def _tagged_linear(n, k, seed):
    linear = pfx.rand_module(n, k, seed)
    object.__setattr__(linear, projkernel._TAG, True)
    return linear


def _triples(*linears):
    return [(m.weight, m.scales, m.biases) for m in linears]


def _bitwise(a, b):
    return mx.array_equal(a, b).item()


def _no_kernel(x, weights):
    raise AssertionError("a call that should have been handed on reached the kernel")


@pytest.fixture(scope="module")
def tiny_target():
    return pfx.tiny_quantized_model()


@pytest.fixture
def fresh_target(tiny_target):
    """The tiny quantized model, untagged again after the test."""
    yield tiny_target
    projkernel.untag(tiny_target)


def test_the_counters_name_the_four_paths():
    assert set(projkernel.calls) == {"verify_kernel", "verify_split", "plain_kernel", "declined"}


def test_clear_drops_the_flag(monkeypatch):
    monkeypatch.setattr(projkernel, "_active", 1)
    projkernel.clear()
    assert projkernel._active == 0


def test_tag_marks_every_quantized_linear_of_the_language_model_and_nothing_else(fresh_target):
    from mlx.utils import tree_flatten

    # mlx-vlm's Model holds the text model as `language_model`, as enable() gets it.
    model = types.SimpleNamespace(language_model=fresh_target)
    assert projkernel.tag(model) == 31
    for path, module in fresh_target.named_modules():
        assert projkernel._tagged(module) == isinstance(module, nn.QuantizedLinear), path
    assert not any(projkernel._TAG in key for key, _ in tree_flatten(fresh_target.parameters()))
    projkernel.untag(model)
    assert not any(projkernel._tagged(module) for _, module in fresh_target.named_modules())


def test_the_drafters_own_linears_are_never_tagged(fresh_target):
    """DFlash binds the target's lm_head into the drafter: that head is the target's
    object and is tagged; every linear the drafter owns is not."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    vocab = fresh_target.args.vocab_size
    drafter = DFlashDraftModel(
        DFlashConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=vocab,
            mask_token_id=vocab - 1,
            target_layer_ids=[1],
            num_target_layers=4,
        )
    )
    drafter.set_dtype(mx.bfloat16)
    nn.quantize(drafter, group_size=64, bits=4)  # as the engine loads a drafter
    model = types.SimpleNamespace(language_model=fresh_target)
    drafter.bind(model)
    projkernel.tag(model)
    assert drafter.lm_head is fresh_target.lm_head and projkernel._tagged(drafter.lm_head)
    owned = [
        module
        for _, module in drafter.named_modules()
        if isinstance(module, nn.QuantizedLinear) and module is not fresh_target.lm_head
    ]
    assert len(owned) == 8
    assert not any(projkernel._tagged(module) for module in owned)


def test_install_hooks_wraps_each_method_once_over_its_original(monkeypatch):
    pfx.guard_proj_state(monkeypatch)
    verifier = projkernel._verifier_class()

    def methods():
        return (verifier._linear, verifier._linears, nn.QuantizedLinear.__call__)

    originals = [inspect.unwrap(method) for method in methods()]
    projkernel.install_hooks()
    first = methods()
    projkernel.install_hooks()
    assert all(now is then for now, then in zip(methods(), first, strict=True))
    for hook, original in zip(first, originals, strict=True):
        assert getattr(hook, projkernel._HOOK_MARK)
        assert inspect.unwrap(hook) is original
        # What the source pins hash on the next load in this process.
        assert inspect.getsource(hook) == inspect.getsource(original)


def test_the_verifier_hook_serves_tagged_verify_calls_through_the_class(monkeypatch):
    pfx.hooked(monkeypatch)
    verifier = projkernel._verifier_class()()
    q, k, v = (_tagged_linear(n, 256, seed) for seed, n in enumerate((512, 128, 128)))
    x = _acts(1, 5, 256)
    before = dict(projkernel.calls)
    single = verifier._linear(q, x)
    group = verifier._linears((q, k, v), x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}
    assert _bitwise(single, pfx.stand_in_mma(x, _triples(q))[0])
    assert isinstance(group, tuple)
    want = pfx.stand_in_mma(x, _triples(q, k, v))
    assert all(_bitwise(got, w) for got, w in zip(group, want, strict=True))


# case -> (flag, how many of the three linears are tagged, x, the counter it moves)
_VERIFY_HANDED_ON = {
    "flag clear": (0, 3, lambda: _acts(1, 3, 256), None),
    "untagged": (1, 0, lambda: _acts(1, 3, 256), None),
    "one linear untagged": (1, 2, lambda: _acts(1, 3, 256), None),
    "float16": (1, 3, lambda: _acts(1, 3, 256, dtype=mx.float16), "declined"),
    "float32": (1, 3, lambda: _acts(1, 3, 256, dtype=mx.float32), "declined"),
    "two dimensions": (1, 3, lambda: _acts(3, 256), "declined"),
    "a batch wider than eight rows": (1, 3, lambda: _acts(9, 1, 256), "declined"),
    "no rows": (1, 3, lambda: _acts(1, 0, 256), "declined"),
}


@pytest.mark.parametrize("method", ["_linear", "_linears"])
@pytest.mark.parametrize("case", list(_VERIFY_HANDED_ON))
def test_the_verifier_hook_hands_every_other_call_to_mlx_vlm(monkeypatch, method, case):
    """Checked against a recording original: the call's own arguments, and its
    answer returned."""
    flag, tagged, build, counter = _VERIFY_HANDED_ON[case]
    seen = []

    def original(self, linears, x):
        seen.append((self, linears, x))
        return "mlx-vlm's answer"

    wrap = projkernel._wrap_linear if method == "_linear" else projkernel._wrap_linears
    hook = wrap(original)
    assert getattr(hook, projkernel._HOOK_MARK) and inspect.unwrap(hook) is original
    monkeypatch.setattr(projkernel, "_active", flag)
    monkeypatch.setattr(projkernel, "mma", _no_kernel)
    group = [pfx.rand_module(128, 256, seed) for seed in range(3)]
    for linear in group[3 - tagged :]:  # the first is the one _linear gets
        object.__setattr__(linear, projkernel._TAG, True)
    linears = group[0] if method == "_linear" else tuple(group)
    verifier, x = object(), build()
    before = dict(projkernel.calls)
    assert hook(verifier, linears, x) == "mlx-vlm's answer"
    ((got_self, got_linears, got_x),) = seen
    assert got_self is verifier and got_linears is linears and got_x is x
    assert _proj_counts_since(before) == ({} if counter is None else {counter: 1})


@pytest.mark.parametrize(
    ("shape", "launches"),
    [((1, 9), [8, 1]), ((1, 12), [8, 4]), ((1, 16), [8, 8]), ((2, 6), [8, 4])],
)
def test_the_verifier_hook_cuts_a_long_verify_into_runs_equal_to_its_rows_alone(
    monkeypatch, shape, launches
):
    """Block 0 lets the drafter's own policy verify up to 16 rows. Each run holds
    at most eight rows, and every row equals the same row computed alone, as
    one-row decode computes it."""
    pfx.hooked(monkeypatch)
    seen = []

    def recording(x, weights):
        seen.append(x.size // x.shape[-1])
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", recording)
    verifier = projkernel._verifier_class()()
    group = [_tagged_linear(n, 256, seed) for seed, n in enumerate((256, 128))]
    x = _acts(*shape, 256)
    before = dict(projkernel.calls)
    outputs = verifier._linears(tuple(group), x)
    assert _proj_counts_since(before) == {"verify_kernel": 1, "verify_split": 1}
    assert seen == launches
    for out, linear in zip(outputs, group, strict=True):
        assert out.shape == (*shape, linear.weight.shape[0])
        for b in range(shape[0]):
            for t in range(shape[1]):
                (alone,) = pfx.stand_in_mma(x[b : b + 1, t : t + 1], _triples(linear))
                assert _bitwise(out[b : b + 1, t : t + 1], alone), (b, t)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (1, 8), (2, 4), (8,)])
def test_the_module_hook_serves_a_tagged_module_up_to_eight_rows(monkeypatch, shape):
    pfx.hooked(monkeypatch)
    linear = _tagged_linear(256, 256, 0)
    x = _acts(*shape, 256)
    before = dict(projkernel.calls)
    got = linear(x)
    assert _proj_counts_since(before) == {"plain_kernel": 1}
    assert _bitwise(got, pfx.stand_in_mma(x, _triples(linear))[0])


# case -> (flag, tagged, x, the counter it moves)
_PLAIN_HANDED_ON = {
    "flag clear": (0, True, lambda: _acts(1, 1, 256), None),
    "untagged": (1, False, lambda: _acts(1, 1, 256), None),
    "nine rows": (1, True, lambda: _acts(1, 9, 256), None),
    "nine rows across the batch": (1, True, lambda: _acts(3, 3, 256), None),
    "a prefill chunk": (1, True, lambda: _acts(1, 2048, 256), None),
    "no rows": (1, True, lambda: _acts(1, 0, 256), None),
    "float16": (1, True, lambda: _acts(1, 1, 256, dtype=mx.float16), "declined"),
    "float32": (1, True, lambda: _acts(1, 3, 256, dtype=mx.float32), "declined"),
    "one dimension": (1, True, lambda: _acts(256), "declined"),
}


@pytest.mark.parametrize("case", list(_PLAIN_HANDED_ON))
def test_the_module_hook_hands_every_other_call_to_what_it_wrapped(monkeypatch, case):
    flag, tagged, build, counter = _PLAIN_HANDED_ON[case]
    seen = []

    def original(self, x):
        seen.append((self, x))
        return "the wrapped method's answer"

    hook = projkernel._wrap_call(original)
    assert getattr(hook, projkernel._HOOK_MARK) and inspect.unwrap(hook) is original
    monkeypatch.setattr(projkernel, "_active", flag)
    monkeypatch.setattr(projkernel, "mma", _no_kernel)
    linear = pfx.rand_module(256, 256, 0)
    if tagged:
        object.__setattr__(linear, projkernel._TAG, True)
    x = build()
    before = dict(projkernel.calls)
    assert hook(linear, x) == "the wrapped method's answer"
    ((got_self, got_x),) = seen
    assert got_self is linear and got_x is x
    assert _proj_counts_since(before) == ({} if counter is None else {counter: 1})


def test_the_module_hook_leaves_untagged_modules_and_a_cleared_flag_on_mlx(monkeypatch):
    pfx.hooked(monkeypatch)
    stock = inspect.unwrap(nn.QuantizedLinear.__call__)
    other = pfx.rand_module(256, 256, 1)  # never tagged: a drafter's, another model's
    ours = _tagged_linear(256, 256, 2)
    x = _acts(1, 1, 256)
    before = dict(projkernel.calls)
    assert _bitwise(other(x), stock(other, x))
    projkernel.clear()
    assert _bitwise(ours(x), stock(ours, x))
    assert _proj_counts_since(before) == {}


def test_a_verify_row_equals_the_plain_row_through_the_two_hooks(monkeypatch, fresh_target):
    """The verifier's own _feed_forward at T = 5 against Qwen3_5MLP.__call__ one
    row at a time: gate+up fused and down alone on one side, three module calls
    on the other. A T- and group-invariant kernel gives both the same bits."""
    pfx.hooked(monkeypatch)
    projkernel.tag(fresh_target)
    mlp = fresh_target.model.layers[0].mlp
    x = _acts(1, 5, 256)
    before = dict(projkernel.calls)
    verify = projkernel._verifier_class()()._feed_forward(mlp, x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}
    before = dict(projkernel.calls)
    plain = [mlp(x[:, t : t + 1]) for t in range(5)]
    assert _proj_counts_since(before) == {"plain_kernel": 15}
    assert all(_bitwise(verify[:, t : t + 1], row) for t, row in enumerate(plain))


_OTHER_VERIFIERS = {
    "gemma4": ("mlx_vlm.models.gemma4.speculative_verifier", "Gemma4ExactSpeculativeVerifier"),
    "nemotron_h": (
        "mlx_vlm.models.nemotron_h.speculative_verifier",
        "NemotronHExactSpeculativeVerifier",
    ),
    "qwen4_exp": ("mlx_vlm.models.qwen4_exp.language", "Qwen4ExpBatchInvariantForward"),
}


@pytest.mark.parametrize("family", list(_OTHER_VERIFIERS))
def test_other_families_verifiers_inherit_the_hook_but_stay_stock_untagged(monkeypatch, family):
    """gemma4's and qwen4_exp's verifiers inherit the wrapped methods and
    nemotron_h's reaches them through super(): only the tags keep them stock."""
    from mlx_vlm.speculative.ops.linear import _target_verify_linear, _target_verify_linears

    module, name = _OTHER_VERIFIERS[family]
    pfx.hooked(monkeypatch)
    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, projkernel._verifier_class())
    verifier = cls()
    a, b = pfx.rand_module(256, 256, 1), pfx.rand_module(128, 256, 2)
    x = _acts(1, 3, 256)
    before = dict(projkernel.calls)
    single = verifier._linear(a, x)
    group = verifier._linears((a, b), x)
    assert _proj_counts_since(before) == {}
    assert _bitwise(single, _target_verify_linear(a, x))
    stock = _target_verify_linears((a, b), x)
    assert all(_bitwise(got, want) for got, want in zip(group, stock, strict=True))
    for linear in (a, b):
        object.__setattr__(linear, projkernel._TAG, True)
    before = dict(projkernel.calls)
    verifier._linear(a, x)
    verifier._linears((a, b), x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}


@pytest.mark.parametrize("order", ["int8 first", "kernel first"])
def test_int8_and_the_kernel_share_quantized_linear_in_either_order(monkeypatch, order):
    """int8 takes a tagged module's calls of 128 rows and more, the kernel its
    calls of eight and fewer, and the rows between go to mlx, whichever wrapper
    sits on top. int8's routing seam, linear_int8, is replaced by a recorder, so
    no tensor unit is needed."""
    pfx.guard_proj_state(monkeypatch)
    stock = inspect.unwrap(nn.QuantizedLinear.__call__)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", stock)
    monkeypatch.setattr(int8prefill, "_QUANTIZED_LINEAR_WRAPPED", False)
    installs = [lambda: int8prefill.install_wrappers([]), projkernel.install_hooks]
    for install in installs if order == "int8 first" else installs[::-1]:
        install()
    top = nn.QuantizedLinear.__call__
    assert getattr(top, projkernel._HOOK_MARK) and inspect.unwrap(top) is stock
    projkernel.install_hooks()
    assert nn.QuantizedLinear.__call__ is top, "the mark is found through int8's wrapper"

    int8_rows = []

    def int8_path(linear, x, stage=None):
        int8_rows.append(x.shape[-2])
        return mx.zeros((*x.shape[:-1], linear.weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(int8prefill, "linear_int8", int8_path)
    monkeypatch.setattr(projkernel, "mma", pfx.stand_in_mma)
    monkeypatch.setattr(projkernel, "_active", 1)
    linear = _tagged_linear(128, 256, 0)
    object.__setattr__(linear, int8prefill._TAG, True)
    before = dict(projkernel.calls)
    long, short, between = _acts(1, 130, 256), _acts(1, 3, 256), _acts(1, 20, 256)
    assert _bitwise(linear(long), mx.zeros((1, 130, 128), dtype=mx.bfloat16))
    assert int8_rows == [130]
    assert _bitwise(linear(short), pfx.stand_in_mma(short, _triples(linear))[0])
    assert _bitwise(linear(between), stock(linear, between))
    assert int8_rows == [130]
    assert _proj_counts_since(before) == {"plain_kernel": 1}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_projkernel.py -v -k "counters_name or clear_drops or tag_marks or drafters_own or install_hooks_wraps or verifier_hook or module_hook or verify_row_equals or other_families or int8_and_the_kernel"`
Expected: FAIL, 47 failed and 3 errors. Every failure is an `AttributeError: module 'sous.engine.projkernel' has no attribute ...`, naming `_verifier_class`, `_wrap_linear`, `_wrap_linears`, `_wrap_call`, `calls`, `tag` or `_active`. The 3 errors are the `fresh_target` fixture's teardown, which calls `untag`.

- [ ] **Step 4: Add the tags, the flag, the counters and both hooks**

In `src/sous/engine/projkernel.py`, add each of these import lines that Task 2's block does not already have. `_root` joins Task 2's `from sous.engine.int8prefill import ...` line (the one importing `_kernel` and `_kernel_text`); `uv run ruff check --fix src/sous/engine/projkernel.py` restores isort order. All four are standard-library or sous imports, so the module still imports no mlx at load.

```python
import functools
import importlib
import math
from collections.abc import Callable
from typing import Any

from sous.engine.int8prefill import _root
```

Append to the end of the file, after Task 2's `kernel_available()`. Notes on the code:
- **ty.** `quantized` is annotated `Any` because ty rejects a `(...) -> Any` wrapper assigned over the method's own signature (`invalid-assignment`), and `setattr` with a constant name would trip ruff's B010 instead. `mx.concatenate` gets `list(parts)` because its stub takes `list[array]`.
- **Cost.** Each wrapper reads `_active` before anything else, so a process that never activates pays one int read per call. `_serve_verify` and `_serve_plain` import `mlx.core` only past the tag check.
- **Cut length.** The verifier hook cuts along T by `MAX_ROWS // batch`, so a run never exceeds `MAX_ROWS` rows whatever the batch. The verifier always runs batch 1, where the runs are 8 rows.

```python
# --------------------------------------------------------------------------------
# Routing: the tags, the flag and the two hooks
# --------------------------------------------------------------------------------

# Set with object.__setattr__ so it lives in the module's __dict__: nn.Module's own
# __setattr__ would put a bool INTO the parameter tree.
_TAG = "_sous_projection_kernel"
# Set on all three wrappers so a second install never wraps again. functools.wraps
# copies the wrapped function's __dict__, so the mark is still found when int8's
# wrapper of nn.QuantizedLinear.__call__ sits on top of ours.
_HOOK_MARK = "_sous_projection_kernel_hook"
# 1 while the kernel serves, else 0: set only by an enable() that reached active,
# cleared first by every enable(), by clear() and by the probe on its way out. A
# plain int, so nothing it holds outlives an unload.
_active = 0
# Per projection call, not per turn: the tests read deltas, and so can a harness
# that wraps the daemon. Nothing is logged per turn.
calls: dict[str, int] = {
    "verify_kernel": 0,  # verifier _linear/_linears calls the kernel served
    "verify_split": 0,  # of those, calls over MAX_ROWS rows cut into runs
    "plain_kernel": 0,  # QuantizedLinear calls the kernel served
    "declined": 0,  # flag set and every linear tagged, refused for dtype or shape
}
_VERIFIER_MODULE = "mlx_vlm.models.qwen3_5.speculative_verifier"


def _verifier_class() -> Any:
    return importlib.import_module(_VERIFIER_MODULE).Qwen3_5BatchInvariantForward


def _tagged(module: Any) -> bool:
    return bool(getattr(module, _TAG, False))


def tag(model: Any) -> int:
    """Tag every QuantizedLinear of the loaded target's language model, lm_head
    included, and return how many; enable() has checked each one eligible. The
    drafter's own linears live in its own tree and are never reached. The
    lm_head DFlash binds into the drafter is the target's own object, so the
    drafter's sampled head goes through the kernel too, which changes proposals
    only."""
    import mlx.nn as nn

    seen: set[int] = set()
    for _, module in _root(model).named_modules():
        if isinstance(module, nn.QuantizedLinear) and id(module) not in seen:
            object.__setattr__(module, _TAG, True)
            seen.add(id(module))
    return len(seen)


def untag(model: Any) -> None:
    """Take the tags off `model`; both hooks then hand its calls on."""
    for _, module in _root(model).named_modules():
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, False)


def clear() -> None:
    """Stop serving: both hooks hand every call on from here on. Tags left on an
    unloaded model are inert while the flag is 0, and untagging it would need a
    reference that keeps its weights alive."""
    global _active
    _active = 0


def _weights(module: Any) -> tuple[Any, Any, Any]:
    return module.weight, module.scales, module.biases


def _run_mma(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """mma over the call's linears, at most _MAX_LINEARS per launch: a fused
    group's rows equal its singles', so where a longer list is cut changes no
    bit. `mma` is read from the module at call time, the seam tests swap the
    plain-mlx stand-in into."""
    out: list[Any] = []
    for i in range(0, len(weights), _MAX_LINEARS):
        out.extend(mma(x, weights[i : i + _MAX_LINEARS]))
    return tuple(out)


def _serve_verify(linears: Any, x: Any) -> tuple[Any, ...] | None:
    """The kernel's outputs for one verifier call while the flag is set, else
    None, and the wrapper calls mlx-vlm's method with the call's own arguments.
    A call of more than MAX_ROWS rows (block 0 lets the drafter's own policy
    verify 16) is cut along T into runs of at most MAX_ROWS: a row's bits do not
    depend on the rows beside it, so every run equals its rows computed one at a
    time, as decode computes them."""
    if not linears or not all(_tagged(m) for m in linears):
        return None
    import mlx.core as mx

    if getattr(x, "ndim", 0) != 3 or x.dtype != mx.bfloat16:
        calls["declined"] += 1
        return None
    batch, rows = x.shape[0], x.shape[1]
    step = MAX_ROWS // batch if batch else 0
    if not step or not rows:
        calls["declined"] += 1
        return None
    weights = [_weights(m) for m in linears]
    calls["verify_kernel"] += 1
    if rows <= step:
        return _run_mma(x, weights)
    calls["verify_split"] += 1
    runs = [_run_mma(x[:, j : j + step], weights) for j in range(0, rows, step)]
    return tuple(mx.concatenate(list(parts), axis=1) for parts in zip(*runs, strict=True))


def _serve_plain(module: Any, x: Any) -> Any | None:
    """The kernel's output for one call of a tagged module while the flag is set,
    else None, and the wrapper calls the method it wrapped: mlx's, or int8's,
    which takes its own calls of 128 rows and more. Rows are the product of the
    leading dimensions, so this serves one-row decode, a last prefill chunk of
    at most MAX_ROWS rows and the one-row forward that ends every prefill
    segment, and nothing longer."""
    if not _tagged(module):
        return None
    import mlx.core as mx

    shape = x.shape
    if not 1 <= math.prod(shape[:-1]) <= MAX_ROWS:
        return None
    if len(shape) < 2 or x.dtype != mx.bfloat16:
        calls["declined"] += 1
        return None
    calls["plain_kernel"] += 1
    return _run_mma(x, [_weights(module)])[0]


def _wrap_linear(original: Callable[..., Any]) -> Callable[..., Any]:
    """The verifier hook for its single projections: o_proj, down_proj, out_proj
    and the verify head."""

    # functools.wraps: the source pins follow __wrapped__ back to mlx-vlm's body.
    @functools.wraps(original)
    def _linear(self: Any, linear: Any, x: Any) -> Any:
        if _active:
            out = _serve_verify((linear,), x)
            if out is not None:
                return out[0]
        return original(self, linear, x)

    setattr(_linear, _HOOK_MARK, True)
    return _linear


def _wrap_linears(original: Callable[..., Any]) -> Callable[..., Any]:
    """The verifier hook for its fused groups: q+k+v, gate+up and qkv+z+b+a."""

    @functools.wraps(original)
    def _linears(self: Any, linears: Any, x: Any) -> Any:
        if _active:
            out = _serve_verify(linears, x)
            if out is not None:
                return out
        return original(self, linears, x)

    setattr(_linears, _HOOK_MARK, True)
    return _linears


def _wrap_call(original: Callable[..., Any]) -> Callable[..., Any]:
    """The module hook: nn.QuantizedLinear.__call__, for every call on the target
    outside the verifier."""

    @functools.wraps(original)
    def call(self: Any, x: Any) -> Any:
        if _active:
            out = _serve_plain(self, x)
            if out is not None:
                return out
        return original(self, x)

    setattr(call, _HOOK_MARK, True)
    return call


def install_hooks() -> None:
    """Wrap the exact verifier's _linear and _linears and nn.QuantizedLinear's
    __call__, once per process: a method already carrying the mark is left alone.
    enable() calls this on the first activation only, and the wrappers are never
    removed; while the flag is 0 each costs one int read per call.

    The verifier is wrapped on the class because its projections reach mlx-vlm's
    ops through names imported into its own module, and verifyattn's _attention
    wrapper calls self._linears and self._linear, so it goes through the same
    hook. gemma4's, nemotron_h's and qwen4_exp's verifiers inherit the wrap: only
    the tags keep them stock. Each wrapper falls through to what it wrapped, so
    int8's wrapper of nn.QuantizedLinear.__call__ may sit above or below this
    one: int8 takes 128 rows and more, this one 8 and fewer."""
    import mlx.nn as nn

    verifier = _verifier_class()
    if not getattr(verifier._linear, _HOOK_MARK, False):
        verifier._linear = _wrap_linear(verifier._linear)
    if not getattr(verifier._linears, _HOOK_MARK, False):
        verifier._linears = _wrap_linears(verifier._linears)
    # Any: ty would reject a wrapper assigned over the method's own signature.
    quantized: Any = nn.QuantizedLinear
    if not getattr(quantized.__call__, _HOOK_MARK, False):
        quantized.__call__ = _wrap_call(quantized.__call__)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_projkernel.py -v -k "counters_name or clear_drops or tag_marks or drafters_own or install_hooks_wraps or verifier_hook or module_hook or verify_row_equals or other_families or int8_and_the_kernel"`
Expected: PASS, 47 passed, in about 2 s on the M2 (the tiny model builds once per module).

- [ ] **Step 6: Run the neighbouring suites, then everything**

The hooks share the verifier class with verifyattn's `_attention` wrapper and `nn.QuantizedLinear.__call__` with int8's wrapper. The suites that drive both run after the new tests, to show nothing leaks out of `guard_proj_state`.

Run: `uv run pytest tests/test_int8prefill.py tests/test_projkernel.py tests/test_verifyattn.py tests/test_tileattn_contract.py tests/test_tileattn.py -v`
Expected: PASS. The only skips are:
- int8's and the tile's `nax` tests, where the tensor units are absent;
- Task 2's `kernel` tests, where `kernel_available()` failed.

It takes about 2.5 minutes on the M2. Run it, and the next command, under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py` when another session may be using the GPU.

Run: `uv run pytest -m "not model"`
Expected: PASS.

- [ ] **Step 7: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/projkernel.py tests/proj_fixtures.py tests/test_projkernel.py
git commit -m "feat(engine): route small-row projections to the kernel through two tag-gated hooks" -m "Two call sites reach the projection kernel. The exact verifier's _linear
and _linears are wrapped on the class, because its projections reach
mlx-vlm's ops through names imported into its own module, and
nn.QuantizedLinear.__call__ is wrapped for decode, short prefill tails and
the one-row forward that ends each prefill segment. Both serve only
modules tagged on the loaded target, and only while the flag is set, so
installing them changes nothing until the kernel is enabled. The other
families whose verifiers inherit the wrap stay stock for the same reason.

A verify of more than eight rows is cut into runs, which is exact because
the kernel's rows do not depend on the rows beside them. functools.wraps
and a mark mean that a second install, or int8's wrapper on top, neither
wraps twice nor hides the originals' source from the pins. The counters
show which path each call took.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

---

### Task 4: The gates, the pins, warm-up, the probe, `enable()` and `static_reason()`

**Files:**
- Modify: `src/sous/engine/projkernel.py`:
  - the import block;
  - the gates section, appended at the end of the file.
- Modify: `tests/proj_fixtures.py`: replace the body of Task 3's `tiny_quantized_model` (same name and signature); append `ACTIVE_TILE` and `proj_ready`.
- Test: `tests/test_projkernel.py`:
  - the import block;
  - the gates section, appended at the end of the file.

**Interfaces:**
- Consumes:
  - Task 2: `projkernel.mma` (looked up at call time, so tests swap the stand-in in), `projkernel.linears`, `projkernel.eligible(module)`, `projkernel.MAX_ROWS`, `projkernel.REL_RMS_BOUND`, `projkernel.logger` (`logging.getLogger("sous.engine.projkernel")`), `pfx.stand_in_mma`, `pfx.kernel`, `pfx.KERNEL`.
  - Task 3: `projkernel._active`, `projkernel.calls` (keys `verify_kernel`, `verify_split`, `plain_kernel`, `declined`), `projkernel._TAG`, `projkernel._HOOK_MARK`, `projkernel.tag(model) -> int`, `projkernel.untag(model)`, `projkernel.install_hooks()`, `projkernel.clear()`, `pfx.guard_proj_state(monkeypatch)`, `pfx.hooked(monkeypatch)`.
  - Task 1: int8's three wrappers carry `__wrapped__`, so the pins below pass with int8 installed.
  - int8prefill: `GROUP` (64), `_error_line`, `_model_type`, `_root`, `install_wrappers`, `_WRAPPED`, `_QUANTIZED_LINEAR_WRAPPED`.
  - verifyattn: `VALIDATED_MLX`, `_source_digest(module, qualname)`.
  - tileattn: `_split_target(device_name) -> (S | None, reason | None)` (the NAX core-count rule through `gpucores.matched_cores` and `MEASURED_SPLITS`), `HQ`, `HKV`, `D`.
  - nax: `platform_reason()`. gpucores: `read`, `GPUCores`.
  - tile_fixtures: `VOCAB`, `prefill_cache`, `verify_forward`, `tiny_language_model`.
- Produces:
  - `projkernel.SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})`, `projkernel.SUPPORTED_DRAFTER_KINDS = frozenset({"dflash"})`.
  - `projkernel.VALIDATED_PROJ_SOURCES: dict[str, str]`, keyed `"module:qualname"`, 13 sha256 digests computed on mlx 0.32.2 and mlx-vlm 0.7.2 (below).
  - `projkernel.warm_up(model) -> None`: every dispatch key of the model at T = 1..8 through `mma`, on zero inputs built on the GPU and evaluated first. Raises on a kernel that does not build.
  - `projkernel.probe(model) -> str | None`: the failure reason or None. Expects the model tagged and the hooks installed; leaves `_active` at 0 and `calls` as it found them, calls `mx.clear_cache()` in `finally`.
  - `projkernel.enable(model, *, enabled: bool, drafter_kind: str | None, tile_status: dict[str, Any]) -> dict[str, Any]`: `{"state", "reason", "probe_seconds"}`; never raises.
    - off: `{"state": "off", "reason": None, "probe_seconds": None}`;
    - active: `{"state": "active", "reason": None, "probe_seconds": round(seconds, 2)}`, where the seconds cover the warm-up and the probe;
    - INFO on success: `projection kernel: {n} projections, probe {seconds:.2f}s`.
  - `projkernel.static_reason(config, model_config: dict, drafter_kind: str | None) -> str | None`: `config` needs only `.attention_tile` and `.projection_kernel`.
  - Private helpers the tests reach: `_pins_reason() -> str | None`, `_stock_distance(got, want) -> float`, `_tile_reason`, `_model_reason`, `_drafter_reason`, `_refuse`, `_PROBE_ROWS = 5`, `_NAN_ROW = 3`.
  - Reason texts:
    - tile: `"attention tile off"`, the tile's own reason verbatim, else `f"attention tile {state}"`;
    - model: `f"unsupported model type {mt or 'unknown'!r} (qwen3_5 only)"`, `"no quantized projections"`, `f"projection {path} is not affine 4-bit gs64 with bf16 scales and no bias"`, `f"activation dtype {dtype} is not bfloat16"`;
    - drafter: `f"drafter kind {kind!r} (dflash only)"`, `"the language model defines speculative_verify_dflash_hidden: greedy verify would take its tokens outside the kernel"`;
    - pins: `f"mlx {version} not validated"`, `f"mlx-vlm {qualname} changed"`, `f"mlx {qualname} changed"`;
    - warm-up: `f"warm-up: {error line}"`;
    - probe: `"the model lacks a full-attention or a linear-attention layer"`, `"the exact verifier's projections did not reach the kernel"`, `"the model's own projections did not reach the kernel"`, `f"parity: verify row {r} of 5 differs from the one-row forward"`, `f"T-invariance: {paths} at T={t} differs from its rows alone"`, `f"grouping: {path} fused in {paths} differs from {path} alone"`, `f"correctness: {path} is {distance:.3g} from stock (bound 0.01)"`, `f"isolation: a NaN in row 3 changed the other rows of {path}"`;
    - static only: `"projection kernel off"`, `f"unsupported attention shape {q}/{kv}/{d} (24/4/256 only)"`, `"unquantized weights (affine 4-bit gs64 only)"`, `f"quantization {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"`, `f"projection {path} is {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"`, plus the platform's and the core reader's reasons verbatim.
  - `tests/proj_fixtures.py`:
    - `ACTIVE_TILE: dict[str, Any]`, an active tile status (`splits` 20);
    - `tiny_quantized_model(*, seed: int = 0)`, Task 3's builder with a new body: a qwen3_5 `LanguageModel`, affine 4-bit gs64, bf16, untied head; linear-attention layers 0 and 2, full-attention layers 1 and 3; 31 quantized projections, every K ≥ 256; nine dispatch keys. Task 3's body (tile_fixtures' model, quantized) has `linear_attn.out_proj` at K = 128, which `eligible()` refuses, so `enable()` could never activate on it; Task 3's tests read only the 31 projections, hidden size 256 and `tfx.VOCAB`, which the new body keeps;
    - `proj_ready(monkeypatch, *, real_pins: bool = False) -> dict[str, Any]`: calls `guard_proj_state`, fakes `nax.platform_reason` (None) and `gpucores.read` (20 cores under this GPU's name), swaps `projkernel.mma` for `stand_in_mma`, fakes `_pins_reason` (None) unless `real_pins`, and returns a copy of `ACTIVE_TILE`.

Notes for the implementer:
- **Order.** Gates 3–5 of `enable()` (the tile, the model, the drafter) are expected refusals, one INFO line each; the pins, the warm-up, the probe and any exception warn. The tile comes first because the kernel rides on it; the warm-up runs before tagging (it calls `mma` directly), and the probe runs after `tag()` and `install_hooks()` because it proves the hooks themselves.
- **The probe's layers.** It uses the model's first full-attention layer and its first linear-attention layer: layers 3 and 0 on the 27B, layers 1 and 0 on the tiny model.
- **Tags.** A failed probe, and any exception, untag the model `enable()` was given. Tags an earlier model kept are never touched: they are inert while `_active` is 0, and holding that model to untag it would keep its weights alive.
- **static_reason** follows `enable()`'s order without loading anything: the two switches first (so a switched-off kernel reads nothing), then the tile's machine and attention-shape gates, then the checkpoint's quantization (the default and every per-projection override under `language_model.`, embeddings excepted, since only `QuantizedLinear` projections reach the kernel; the OptiQ and oQ checkpoints carry 5- and 8-bit overrides), then the drafter kind.

- [ ] **Step 1: Give the tiny quantized model the kernel's shapes and add `proj_ready`**

`tests/proj_fixtures.py` already imports `pytest`, `Any` (from `typing`), `projkernel` and `tile_fixtures as tfx` at module level and defines `stand_in_mma`, `tiny_quantized_model(*, seed=0)` and `guard_proj_state(monkeypatch)` (Tasks 2–3). `proj_ready` calls `guard_proj_state` as a plain function, the way `tile_fixtures.tile_ready` calls `guard_tile_state`.

Replace Task 3's whole `tiny_quantized_model` definition (from `def tiny_quantized_model(` up to `def guard_proj_state(`) with the one below. Keep one definition: a second `def` would rebind the name for every caller, Task 3's included.

```python
def tiny_quantized_model(*, seed: int = 0) -> Any:
    """A random qwen3_5 language model quantized as the 27B is (affine 4-bit,
    group 64, bf16 scales, no bias, lm_head untied). Linear-attention layers 0
    and 2, full-attention layers 1 and 3, and 31 quantized projections, every
    one with K >= 256 so that each of the plain variant's four K splits gets a
    whole group. Its K/N shapes give nine dispatch keys."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    args = TextConfig(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=tfx.VOCAB,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,
    )
    mx.random.seed(seed)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=4)
    mx.eval(lm.parameters())
    return lm
```

Append to the end of the file:

```python
# The attention tile's status as an active load reports it; enable() rides on it.
ACTIVE_TILE: dict[str, Any] = {
    "state": "active",
    "reason": None,
    "splits": 20,
    "probe_seconds": 0.5,
}


def proj_ready(monkeypatch: pytest.MonkeyPatch, *, real_pins: bool = False) -> dict[str, Any]:
    """Every enable() and static_reason() guard that reads the machine passes
    on any GPU: the platform rule, and a core count of 20 under this GPU's own
    name. The stand-in is the kernel, and the pins pass unless `real_pins` (for
    the tests about the pins themselves). Registers guard_proj_state's restores
    and returns the attention tile's status to hand enable()."""
    import mlx.core as mx

    from sous.engine import gpucores, nax

    guard_proj_state(monkeypatch)
    device = str(mx.device_info()["device_name"])
    monkeypatch.setattr(nax, "platform_reason", lambda: None)
    monkeypatch.setattr(gpucores, "read", lambda: gpucores.GPUCores(20, device, None, "iokit"))
    monkeypatch.setattr(projkernel, "mma", stand_in_mma)
    if not real_pins:
        monkeypatch.setattr(projkernel, "_pins_reason", lambda: None)
    return dict(ACTIVE_TILE)
```

Check the model before anything depends on it:

Run: `uv run python -c "import mlx.nn as nn; from tests import proj_fixtures as pfx; lm = pfx.tiny_quantized_model(); print(sum(isinstance(m, nn.QuantizedLinear) for _, m in lm.named_modules()), [layer.is_linear for layer in lm.model.layers], {m.weight.shape[1] * 8 for _, m in lm.named_modules() if isinstance(m, nn.QuantizedLinear)})"`
Expected: `31 [True, False, True, False] {256, 512, 1024}`.

- [ ] **Step 2: Write the failing tests**

The import block at the top of `tests/test_projkernel.py` must hold at least these (keep everything Tasks 2–3 put there; `uv run ruff check --fix .` sorts the block):

```python
import contextlib
import copy
import importlib
import inspect
import logging
import types
import warnings

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores, int8prefill, nax, projkernel  # noqa: E402 — after the guard
from tests import proj_fixtures as pfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402
```

If a helper below shares a name with one Tasks 2–3 already defined in this module (`_verifier_cls`, `_projections`, `_tagged_count`, `_hooks_in_place`, `_rows`, `_scaled`), keep one definition and make sure it computes what the gates section expects: the verifier class, the language model's quantized projections, how many carry `_TAG`, whether all three hooks carry `_HOOK_MARK`, the product of the leading dimensions, and outputs scaled in fp32 and rounded back to bf16.

Append the gates section to the end of `tests/test_projkernel.py`:

```python
# ---- the gates, the pins, the warm-up, the probe and enable() -----------------------

# The tiny model's quantized projections: a one-row forward makes one hook-2 call
# for each. A verify forward makes 17 hook-1 calls: four per layer, then the head.
_PROJECTIONS = 31
_VERIFY_CALLS = 17
_BUILD_ERROR = "Unable to build metal library\nprogram_source:3:1: error: boom"


def _verifier_cls():
    return importlib.import_module(
        "mlx_vlm.models.qwen3_5.speculative_verifier"
    ).Qwen3_5BatchInvariantForward


def _projections(model) -> list:
    import mlx.nn as nn

    root = getattr(model, "language_model", model)
    return [m for _, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]


def _tagged_count(model) -> int:
    return sum(bool(getattr(m, projkernel._TAG, False)) for m in _projections(model))


def _hooks_in_place() -> bool:
    import mlx.nn as nn

    cls = _verifier_cls()
    return all(
        getattr(f, projkernel._HOOK_MARK, False)
        for f in (nn.QuantizedLinear.__call__, cls._linear, cls._linears)
    )


class _GateRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _gate_log():
    """Every record the kernel's logger emits inside the block, INFO included.
    Records still propagate, so a test's own caplog sees them too."""
    logger = logging.getLogger("sous.engine.projkernel")
    handler, level = _GateRecords(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _refusal(model, *, warned: bool, untagged: bool = True, **kwargs) -> dict:
    """enable() on `model`, expected to refuse, leaving the flag clear. A load
    the kernel does not apply to (the tile, the model, the drafter) is one INFO
    line and no warning, since most Macs load that way; one where it should
    have run and did not is exactly one warning, on one line, in the kernel's
    words. Unless the model kept its tags from an earlier activation, nothing
    on it is tagged afterwards."""
    kwargs = {"drafter_kind": None, "tile_status": dict(pfx.ACTIVE_TILE), **kwargs}
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = projkernel.enable(model, enabled=True, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert status["state"] == "unavailable"
    if warned:
        assert len(messages) == 1, messages
        assert messages[0].startswith("sous: projection kernel unavailable (")
        assert "\n" not in messages[0]
        assert not any(m.startswith("projection kernel unavailable") for m in infos), infos
    else:
        assert messages == []
        assert infos == [f"projection kernel unavailable: {status['reason']}"]
    assert projkernel._active == 0
    if untagged:
        assert _tagged_count(model) == 0
    return status


def _enabling_failures(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "sous.engine.projkernel"
        and r.getMessage() == "projection kernel: enabling failed"
    ]


def test_enable_off_returns_before_any_guard(monkeypatch):
    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 1)

    def untouchable(*args, **kwargs):
        raise AssertionError("a guard ran while the kernel is off")

    for name in ("_tile_reason", "_model_reason", "_pins_reason", "warm_up", "probe"):
        monkeypatch.setattr(projkernel, name, untouchable)
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = projkernel.enable(
            types.SimpleNamespace(), enabled=False, drafter_kind="mtp", tile_status={}
        )
    assert status == {"state": "off", "reason": None, "probe_seconds": None}
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert records == []
    assert projkernel._active == 0


_GUARD_REASONS = [
    "macOS 15.5 < 26.2",
    "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
    "drafter kind 'mtp' (dflash only)",
    "mlx 0.99.0 not validated",
    "warm-up: program_source:3:1: error: boom",
    "parity: verify row 2 of 5 differs from the one-row forward",
]
# Which of them warn: a pinned dependency that drifted, a kernel this OS rejects
# and a failed probe. The tile, the model and the drafter's kind only say the
# kernel does not apply to this load.
_GUARD_WARNS = [False, False, False, True, True, True]


@pytest.mark.parametrize("first", range(len(_GUARD_REASONS)))
def test_enable_names_the_first_failing_guard(monkeypatch, first):
    """Guard `first` and every guard after it fail; the reason is guard `first`'s."""
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    lm = pfx.tiny_quantized_model()
    kwargs: dict = {"drafter_kind": None, "tile_status": tile}

    def broken_warm_up(model):
        raise RuntimeError(_BUILD_ERROR)

    breakers = [
        lambda: kwargs.update(
            tile_status={"state": "unavailable", "reason": _GUARD_REASONS[0], "splits": None}
        ),
        lambda: monkeypatch.setattr(lm, "model_type", "qwen3_5_moe"),
        lambda: kwargs.update(drafter_kind="mtp"),
        lambda: monkeypatch.setattr(mx, "__version__", "0.99.0"),
        lambda: monkeypatch.setattr(projkernel, "warm_up", broken_warm_up),
        lambda: monkeypatch.setattr(projkernel, "probe", lambda model: _GUARD_REASONS[5]),
    ]
    for brk in breakers[first:]:
        brk()
    status = _refusal(lm, warned=_GUARD_WARNS[first], **kwargs)
    assert status["reason"] == _GUARD_REASONS[first]
    # The cost is reported once the warm-up has started.
    assert (status["probe_seconds"] is None) == (first < 4)


@pytest.mark.parametrize(
    ("tile_status", "reason"),
    [
        ({"state": "off", "reason": None, "splits": None}, "attention tile off"),
        (
            {"state": "unavailable", "reason": "split target 16 not measured (measured: 20)"},
            "split target 16 not measured (measured: 20)",
        ),
        ({"state": "unavailable", "reason": None}, "attention tile unavailable"),
    ],
)
def test_enable_rides_on_the_attention_tile(monkeypatch, tile_status, reason):
    pfx.proj_ready(monkeypatch)
    status = _refusal(pfx.tiny_quantized_model(), warned=False, tile_status=tile_status)
    assert status["reason"] == reason and status["probe_seconds"] is None


def test_enable_refuses_other_model_types(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    # The MoE variant reuses the verifier class and the dense MLP.
    monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
    status = _refusal(lm, warned=False)
    assert status["reason"] == "unsupported model type 'qwen3_5_moe' (qwen3_5 only)"


def test_enable_refuses_a_model_without_quantized_projections(monkeypatch):
    pfx.proj_ready(monkeypatch)
    status = _refusal(tfx.tiny_language_model(), warned=False)
    assert status["reason"] == "no quantized projections"


def _replace(lm, case: str) -> str:
    """Swap one projection for one the kernel cannot take; returns its path."""
    import mlx.nn as nn

    if case == "8-bit":
        head = nn.QuantizedLinear(256, tfx.VOCAB, bias=False, group_size=64, bits=8)
        head.set_dtype(mx.bfloat16)
        lm.lm_head = head
        return "lm_head"
    if case == "bias":
        out = nn.QuantizedLinear(256, 256, bias=True, group_size=64, bits=4)
        out.set_dtype(mx.bfloat16)
        lm.model.layers[0].linear_attn.out_proj = out
        return "model.layers.0.linear_attn.out_proj"
    down = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    down.set_dtype(mx.float16)
    lm.model.layers[1].mlp.down_proj = down
    return "model.layers.1.mlp.down_proj"


@pytest.mark.parametrize("case", ["8-bit", "bias", "fp16 scales"])
def test_enable_refuses_a_projection_the_kernel_cannot_take(monkeypatch, case):
    """Decode and verify route every projection of the language model, so one
    the kernel cannot take refuses the load rather than leaving it stock."""
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    path = _replace(lm, case)
    status = _refusal(lm, warned=False)
    assert status["reason"] == (
        f"projection {path} is not affine 4-bit gs64 with bf16 scales and no bias"
    )


def test_enable_refuses_activations_that_are_not_bf16(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    lm.model.norm.set_dtype(mx.float16)
    status = _refusal(lm, warned=False)
    assert status["reason"] == f"activation dtype {mx.float16} is not bfloat16"


@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
def test_enable_refuses_drafters_other_than_dflash(monkeypatch, kind):
    pfx.proj_ready(monkeypatch)
    status = _refusal(pfx.tiny_quantized_model(), warned=False, drafter_kind=kind)
    assert status["reason"] == f"drafter kind {kind!r} (dflash only)"


def test_enable_refuses_a_target_whose_greedy_verify_skips_the_head(monkeypatch):
    """mlx-vlm's greedy verify takes its tokens from quantized_argmax, which
    neither hook sees, once the target defines this method."""
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    object.__setattr__(lm, "speculative_verify_dflash_hidden", lambda *args, **kwargs: None)
    status = _refusal(lm, warned=False, drafter_kind="dflash")
    assert status["reason"] == (
        "the language model defines speculative_verify_dflash_hidden: "
        "greedy verify would take its tokens outside the kernel"
    )


def test_the_pins_hold_on_the_locked_mlx_and_mlx_vlm():
    assert set(projkernel.VALIDATED_PROJ_SOURCES) == {
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linear",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._feed_forward",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._gated_delta",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._model",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5MLP.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet._project_gates",
        "mlx_vlm.generate.ar:generate_step",
        "mlx.nn:QuantizedLinear.__call__",
        "mlx_vlm.speculative.dflash:_dflash_verify",
        "mlx_vlm.speculative.dflash:_dflash_verify_greedy",
    }
    assert projkernel._pins_reason() is None


def test_enable_refuses_an_unvalidated_mlx(monkeypatch):
    pfx.proj_ready(monkeypatch, real_pins=True)
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "mlx 0.99.0 not validated"


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        (
            "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears",
            "mlx-vlm Qwen3_5BatchInvariantForward._linears changed",
        ),
        (
            "mlx_vlm.speculative.dflash:_dflash_verify_greedy",
            "mlx-vlm _dflash_verify_greedy changed",
        ),
        ("mlx.nn:QuantizedLinear.__call__", "mlx QuantizedLinear.__call__ changed"),
    ],
)
def test_enable_refuses_a_changed_source(monkeypatch, key, reason):
    pfx.proj_ready(monkeypatch, real_pins=True)
    monkeypatch.setitem(projkernel.VALIDATED_PROJ_SOURCES, key, "0" * 64)
    assert _refusal(pfx.tiny_quantized_model(), warned=True)["reason"] == reason


@pytest.mark.parametrize("order", ["int8 first", "int8 after"])
def test_the_pins_follow_int8s_wrappers_and_the_hooks_to_the_originals(monkeypatch, order):
    """int8 prefill wraps nn.QuantizedLinear.__call__, Qwen3_5MLP.__call__ and
    Qwen3_5GatedDeltaNet.__call__, all three pinned here, in either order with
    the hooks. With int8 after, its wrapper sits on hook 2 and carries the mark,
    so a later load neither re-wraps nor reads the pins as drifted."""
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet, Qwen3_5MLP

    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    for cls in (Qwen3_5MLP, Qwen3_5GatedDeltaNet):
        monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(int8prefill, "_WRAPPED", set())
    monkeypatch.setattr(int8prefill, "_QUANTIZED_LINEAR_WRAPPED", False)

    def install_int8() -> None:
        int8prefill.install_wrappers([(Qwen3_5MLP, "mlp"), (Qwen3_5GatedDeltaNet, "gdn")])

    if order == "int8 first":
        install_int8()
    status = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    if order == "int8 after":
        install_int8()
    installed = nn.QuantizedLinear.__call__
    projkernel.clear()
    status = projkernel.enable(
        pfx.tiny_quantized_model(seed=1), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    assert nn.QuantizedLinear.__call__ is installed
    assert projkernel._pins_reason() is None


def test_a_second_load_activates_again_on_the_hooks_the_first_installed(monkeypatch):
    """An idle unload and reload, or sous tune's arms, enable a fresh model in a
    process whose hooks are in place: the pins follow __wrapped__ through them
    to mlx-vlm's and mlx's own sources, and nothing is wrapped twice."""
    import mlx.nn as nn

    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    first = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert first["state"] == "active", first
    cls = _verifier_cls()
    hooks = (nn.QuantizedLinear.__call__, cls.__dict__["_linear"], cls.__dict__["_linears"])
    projkernel.clear()
    second = projkernel.enable(
        pfx.tiny_quantized_model(seed=1), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert second["state"] == "active", second
    now = (nn.QuantizedLinear.__call__, cls.__dict__["_linear"], cls.__dict__["_linears"])
    assert all(a is b for a, b in zip(now, hooks, strict=True))


def test_enable_refuses_a_kernel_the_warm_up_cannot_build(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)
    error = RuntimeError(_BUILD_ERROR)

    def broken(model):
        raise error

    def untouchable(model):
        raise AssertionError("probed after a failed warm-up")

    monkeypatch.setattr(projkernel, "warm_up", broken)
    monkeypatch.setattr(projkernel, "probe", untouchable)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "warm-up: program_source:3:1: error: boom"
    assert isinstance(status["probe_seconds"], float)
    # The warning keeps one line; the log keeps the compiler's whole report.
    (record,) = [
        r
        for r in caplog.records
        if r.name == "sous.engine.projkernel"
        and r.getMessage() == "projection kernel: the warm-up failed"
    ]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


def test_warm_up_runs_every_dispatch_key_at_every_row_count(monkeypatch):
    pfx.guard_proj_state(monkeypatch)
    seen: list = []

    def spy(x, weights):
        seen.append((x.shape[-1], tuple(w.shape[0] for w, _, _ in weights), x.shape[1], x.dtype))
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", spy)
    projkernel.warm_up(pfx.tiny_quantized_model())
    keys = [
        (256, (2048, 512, 512)),  # q+k+v
        (256, (512, 512)),  # gate+up
        (256, (512, 256, 4, 4)),  # in_proj qkv+z+b+a
        (256, (512,)),  # lm_head, gate, up, k, v and in_proj_qkv alone
        (256, (2048,)),  # q alone
        (256, (256,)),  # in_proj_z and out_proj alone
        (256, (4,)),  # in_proj_b and in_proj_a alone
        (512, (256,)),  # down
        (1024, (256,)),  # o
    ]
    expected = sorted((k, n, t) for k, n in keys for t in range(1, projkernel.MAX_ROWS + 1))
    assert sorted((k, n, t) for k, n, t, _ in seen) == expected
    assert {dtype for *_, dtype in seen} == {mx.bfloat16}


def _probe_model(monkeypatch, mma=None):
    """A tiny model as enable() hands it to the probe: tagged, both hooks in
    place, the flag clear, and `mma` (the stand-in by default) as the kernel."""
    lm = pfx.tiny_quantized_model()
    pfx.guard_proj_state(monkeypatch)
    pfx.hooked(monkeypatch)
    if mma is not None:
        monkeypatch.setattr(projkernel, "mma", mma)
    monkeypatch.setattr(projkernel, "_active", 0)
    projkernel.tag(lm)
    return lm


def _run_probe(lm) -> str | None:
    """probe(), asserting it left the flag clear and the counters as it found
    them, whatever it decided."""
    before = dict(projkernel.calls)
    reason = projkernel.probe(lm)
    assert projkernel._active == 0
    assert projkernel.calls == before
    return reason


def _rows(x) -> int:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return rows


def _scaled(outs, factor: float) -> tuple:
    return tuple((o.astype(mx.float32) * factor).astype(mx.bfloat16) for o in outs)


def test_probe_passes_the_stand_in(monkeypatch):
    assert _run_probe(_probe_model(monkeypatch)) is None


@pfx.kernel
def test_probe_passes_the_real_kernel(monkeypatch):
    assert _run_probe(_probe_model(monkeypatch, projkernel.linears)) is None


def test_probe_refuses_projections_that_never_reach_the_kernel(monkeypatch):
    lm = _probe_model(monkeypatch)
    projkernel.untag(lm)
    assert _run_probe(lm) == "the exact verifier's projections did not reach the kernel"


def test_probe_refuses_when_the_models_own_calls_skip_hook_two(monkeypatch):
    import mlx.nn as nn

    lm = _probe_model(monkeypatch)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", inspect.unwrap(nn.QuantizedLinear.__call__))
    assert _run_probe(lm) == "the model's own projections did not reach the kernel"


def test_probe_fails_parity_for_a_kernel_whose_rows_move_with_the_row_count(monkeypatch):
    def by_rows(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + _rows(x) / 64)

    reason = _run_probe(_probe_model(monkeypatch, by_rows))
    assert reason == "parity: verify row 0 of 5 differs from the one-row forward"


def test_probe_fails_t_invariance_past_the_rows_the_hooks_step_used(monkeypatch):
    def past_five(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + (_rows(x) > 5) / 16)

    reason = _run_probe(_probe_model(monkeypatch, past_five))
    assert reason == (
        "T-invariance: model.layers.1.self_attn.q_proj+model.layers.1.self_attn.k_proj"
        "+model.layers.1.self_attn.v_proj at T=6 differs from its rows alone"
    )


def test_probe_fails_grouping_for_a_kernel_whose_rows_move_with_the_group(monkeypatch):
    def by_group(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + (len(weights) > 2) / 16)

    reason = _run_probe(_probe_model(monkeypatch, by_group))
    assert reason is not None and reason.startswith(
        "grouping: model.layers.1.self_attn.q_proj fused in "
    ), reason


def test_probe_fails_correctness_for_a_kernel_wrong_the_same_way_everywhere(monkeypatch):
    def scaled(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1.1)

    reason = _run_probe(_probe_model(monkeypatch, scaled))
    assert reason is not None and reason.startswith(
        "correctness: model.layers.1.self_attn.q_proj is "
    ), reason


def test_probe_fails_isolation_for_a_kernel_that_mixes_rows(monkeypatch):
    def mixes_rows(x, weights):
        return tuple(
            o + 0 * mx.sum(o, axis=-2, keepdims=True) for o in pfx.stand_in_mma(x, weights)
        )

    reason = _run_probe(_probe_model(monkeypatch, mixes_rows))
    assert reason == (
        "isolation: a NaN in row 3 changed the other rows of model.layers.1.self_attn.q_proj"
    )


def test_the_stock_distance_fails_every_bound_on_a_nan():
    want = mx.ones((1, 8, 64), dtype=mx.bfloat16)
    got = mx.where(mx.arange(64) == 7, mx.array(float("nan"), dtype=mx.bfloat16), want)
    assert projkernel._stock_distance(got, want) == float("inf")
    assert not projkernel._stock_distance(want, got) <= projkernel.REL_RMS_BOUND
    assert projkernel._stock_distance(want, want) == 0.0


def test_probe_restores_the_flag_and_the_counters_when_it_raises(monkeypatch):
    def fails_through_the_hooks(x, weights):
        if projkernel._active:
            raise RuntimeError("device lost")
        return pfx.stand_in_mma(x, weights)

    lm = _probe_model(monkeypatch, fails_through_the_hooks)
    before = dict(projkernel.calls)
    with pytest.raises(RuntimeError, match="device lost"):
        projkernel.probe(lm)
    assert projkernel._active == 0
    assert projkernel.calls == before


def test_enable_refuses_a_failed_probe_with_its_cost_and_untags_the_model(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    tagged_when_probed: list = []

    def failing(model):
        tagged_when_probed.append(_tagged_count(model))
        return "correctness: lm_head is 0.101 from stock (bound 0.01)"

    monkeypatch.setattr(projkernel, "probe", failing)
    status = _refusal(lm, warned=True)
    assert status["reason"] == "correctness: lm_head is 0.101 from stock (bound 0.01)"
    assert isinstance(status["probe_seconds"], float)
    assert tagged_when_probed == [_PROJECTIONS]


def test_a_failed_probe_untags_only_the_model_it_was_given(monkeypatch):
    tile = pfx.proj_ready(monkeypatch)
    earlier = pfx.tiny_quantized_model()
    status = projkernel.enable(earlier, enabled=True, drafter_kind=None, tile_status=tile)
    assert status["state"] == "active", status
    monkeypatch.setattr(
        projkernel,
        "probe",
        lambda model: "parity: verify row 0 of 5 differs from the one-row forward",
    )
    _refusal(pfx.tiny_quantized_model(seed=1), warned=True)
    # An earlier load's tags are inert while the flag is clear, and never touched.
    assert _tagged_count(earlier) == _PROJECTIONS


def test_enable_refuses_an_exception_in_a_guard_with_its_error_line(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 1)
    error = RuntimeError(_BUILD_ERROR)

    def broken(model):
        raise error

    monkeypatch.setattr(projkernel, "_model_reason", broken)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "program_source:3:1: error: boom"
    assert status["probe_seconds"] is None
    # The warning keeps one line; the log keeps where it was raised.
    (record,) = _enabling_failures(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


def test_enable_refuses_a_probe_that_raises_and_reports_what_it_cost(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)

    def fails_through_the_hooks(x, weights):
        if projkernel._active:
            raise RuntimeError("device lost")
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", fails_through_the_hooks)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "device lost"
    assert isinstance(status["probe_seconds"], float)
    (record,) = _enabling_failures(caplog)
    assert record.exc_info is not None and "device lost" in str(record.exc_info[1])


@pytest.mark.parametrize("kind", [None, "dflash"])
def test_enable_warms_up_untagged_then_tags_hooks_and_probes(monkeypatch, caplog, kind):
    tile = pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    order: list = []
    monkeypatch.setattr(
        projkernel, "warm_up", lambda model: order.append(("warm-up", _tagged_count(model)))
    )
    monkeypatch.setattr(
        projkernel,
        "probe",
        lambda model: order.append(("probe", _tagged_count(model), _hooks_in_place())),
    )
    with caplog.at_level(logging.INFO, logger="sous.engine.projkernel"):
        status = projkernel.enable(lm, enabled=True, drafter_kind=kind, tile_status=tile)
    assert status == {"state": "active", "reason": None, "probe_seconds": status["probe_seconds"]}
    assert isinstance(status["probe_seconds"], float)
    assert order == [("warm-up", 0), ("probe", _PROJECTIONS, True)]
    assert projkernel._active == 1
    assert [r.getMessage() for r in caplog.records if r.name == "sous.engine.projkernel"] == [
        f"projection kernel: {_PROJECTIONS} projections, probe {status['probe_seconds']:.2f}s"
    ]


def test_enable_goes_active_through_the_real_warm_up_and_probe(monkeypatch):
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    status = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    assert isinstance(status["probe_seconds"], float)
    assert projkernel._active == 1


@pytest.mark.parametrize(
    "ending", ["disabled", "failed gate", "other model", "failed probe", "cleared"]
)
def test_enable_then_a_later_load_or_clear_leaves_verify_and_decode_stock(monkeypatch, ending):
    tile = pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    ids = mx.random.randint(0, tfx.VOCAB, (43,), key=mx.random.key(5)).tolist()

    def verify() -> list:
        return tfx.verify_forward(lm, tfx.prefill_cache(lm, ids[:40]), ids[40:])

    def decode():
        cache = tfx.prefill_cache(lm, ids[:40])
        logits = lm(mx.array([ids[40:41]], dtype=mx.int32), cache=cache).logits
        mx.eval(logits)
        return logits

    verify_stock, decode_stock = verify(), decode()
    status = projkernel.enable(lm, enabled=True, drafter_kind=None, tile_status=tile)
    assert status["state"] == "active", status
    before = dict(projkernel.calls)
    verify()
    decode()
    # While active, both paths reach the kernel: what follows is not vacuous.
    assert projkernel.calls["verify_kernel"] - before["verify_kernel"] == _VERIFY_CALLS
    assert projkernel.calls["plain_kernel"] - before["plain_kernel"] == _PROJECTIONS

    if ending == "disabled":
        projkernel.enable(lm, enabled=False, drafter_kind=None, tile_status=tile)
    elif ending == "failed gate":
        _refusal(lm, warned=False, untagged=False, tile_status={"state": "off"})
    elif ending == "other model":
        monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
        _refusal(lm, warned=False, untagged=False)
    elif ending == "failed probe":
        monkeypatch.setattr(projkernel, "probe", lambda model: "parity: verify row 0 of 5 differs")
        _refusal(lm, warned=True)
    else:
        projkernel.clear()

    assert projkernel._active == 0
    before = dict(projkernel.calls)
    assert all(mx.array_equal(a, b).item() for a, b in zip(verify(), verify_stock, strict=True))
    assert mx.array_equal(decode(), decode_stock).item()
    for key in ("verify_kernel", "verify_split", "plain_kernel"):
        assert projkernel.calls[key] == before[key], key


_CONFIG_27B = {
    "model_type": "qwen3_5",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {"num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256},
}


def _model_config(**changes) -> dict:
    config = copy.deepcopy(_CONFIG_27B)
    for key, value in changes.items():
        config[key] = value
    return config


_STATIC_CASES = {
    "dflash": ({}, _CONFIG_27B, "dflash", None),
    "no drafter": ({}, _CONFIG_27B, None, None),
    "text-only layout": (
        {},
        {"model_type": "qwen3_5", "quantization": _CONFIG_27B["quantization"]}
        | _CONFIG_27B["text_config"],
        "dflash",
        None,
    ),
    "kernel off": ({"projection_kernel": False}, _CONFIG_27B, "dflash", "projection kernel off"),
    "tile off": ({"attention_tile": False}, _CONFIG_27B, "dflash", "attention tile off"),
    "another model type": (
        {},
        _model_config(model_type="qwen3_5_moe"),
        "dflash",
        "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
    ),
    "another attention shape": (
        {},
        _model_config(
            text_config={"num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256}
        ),
        "dflash",
        "unsupported attention shape 16/4/256 (24/4/256 only)",
    ),
    "unquantized": (
        {},
        {k: v for k, v in _CONFIG_27B.items() if k != "quantization"},
        "dflash",
        "unquantized weights (affine 4-bit gs64 only)",
    ),
    "mxfp4": (
        {},
        _model_config(quantization={"group_size": 32, "bits": 4, "mode": "mxfp4"}),
        "dflash",
        "quantization mxfp4 4-bit gs32 (affine 4-bit gs64 only)",
    ),
    "an 8-bit projection": (
        {},
        _model_config(
            quantization={
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "bits": 8,
                    "group_size": 64,
                },
            }
        ),
        "dflash",
        "projection language_model.model.layers.0.linear_attn.in_proj_qkv is affine 8-bit gs64"
        " (affine 4-bit gs64 only)",
    ),
    "an 8-bit embedding": (
        {},
        _model_config(
            quantization={
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
                "vision_tower.blocks.0.attn.qkv": {"bits": 8, "group_size": 64},
            }
        ),
        "dflash",
        None,
    ),
    "mtp drafter": ({}, _CONFIG_27B, "mtp", "drafter kind 'mtp' (dflash only)"),
}


@pytest.mark.parametrize("case", list(_STATIC_CASES))
def test_static_reason_reads_the_config_the_model_and_the_drafter(monkeypatch, case):
    pfx.proj_ready(monkeypatch)
    switches, model_config, drafter_kind, reason = _STATIC_CASES[case]
    config = types.SimpleNamespace(
        **{"attention_tile": True, "projection_kernel": True, **switches}
    )
    assert projkernel.static_reason(config, model_config, drafter_kind) == reason


@pytest.mark.parametrize("case", ["platform", "unreadable", "another GPU", "unmeasured"])
def test_static_reason_reads_the_machine_as_the_tile_does(monkeypatch, case):
    pfx.proj_ready(monkeypatch)
    device = str(mx.device_info()["device_name"])
    if case == "platform":
        monkeypatch.setattr(nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
        reason = "macOS 15.5 < 26.2"
    else:
        reading, reason = {
            "unreadable": (
                gpucores.GPUCores(None, None, "GPU core count unreadable: timed out", "none"),
                "GPU core count unreadable: timed out",
            ),
            "another GPU": (
                gpucores.GPUCores(20, "Apple M9 Ultra", None, "iokit"),
                f"IORegistry GPU 'Apple M9 Ultra' is not mlx's {device!r}",
            ),
            "unmeasured": (
                gpucores.GPUCores(16, device, None, "iokit"),
                "split target 16 not measured (measured: 20)",
            ),
        }[case]
        monkeypatch.setattr(gpucores, "read", lambda: reading)
    config = types.SimpleNamespace(attention_tile=True, projection_kernel=True)
    assert projkernel.static_reason(config, _CONFIG_27B, "dflash") == reason


def test_static_reason_reads_nothing_while_either_switch_is_off(monkeypatch):
    pfx.proj_ready(monkeypatch)

    def untouchable():
        raise AssertionError("read the machine for a switched-off kernel")

    monkeypatch.setattr(nax, "platform_reason", untouchable)
    monkeypatch.setattr(gpucores, "read", untouchable)
    for switches in ({"projection_kernel": False}, {"attention_tile": False}):
        config = types.SimpleNamespace(
            **{"attention_tile": True, "projection_kernel": True, **switches}
        )
        assert projkernel.static_reason(config, {}, "mtp") is not None
```

What each group pins:
- `_refusal` holds every refusal to its kind: an INFO line for the tile, the model and the drafter, one one-line warning for the pins, the warm-up, the probe and any exception, the flag clear either way, and no tags left behind.
- The gate-order table breaks guard `first` and every guard after it, as the tile's does, so a guard that ran out of order names the wrong reason. Its `probe_seconds` check pins that the cost is reported from the warm-up on.
- The pin tests use the real digests (`real_pins=True`): the locked versions pass them; a forced mlx version or a zeroed digest refuses with a warning; a second load in the same process, and int8's wrappers in either order, still pass, which is `__wrapped__` being followed.
- The probe tests run the real probe on the tiny model with a stand-in that is wrong in one way each, chosen so the earlier checks pass and the named check fails: row-count dependence (parity, through both hooks), dependence above five rows (T-invariance), dependence on three or more fused linears (grouping), a 10% scale (correctness), and a zero-weighted row sum (isolation, NaN only).
- The lifecycle test proves both real paths reach the kernel while active, then that each way of ending (a disabled load, a refused gate, another model, a failed probe, `clear()`) gives the stock bits again with no kernel calls.
- `static_reason` is a table: the config switches, the machine as the tile reads it, the model's type, attention shape and quantization from `config.json`, and the drafter kind.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_projkernel.py -v`
Expected: FAIL. Tasks 2–3's tests still pass. Every test this step added fails with an `AttributeError` naming a projkernel attribute this task adds: `_pins_reason` (the 42 that call `proj_ready`), `enable`, `probe`, `warm_up`, `VALIDATED_PROJ_SOURCES` or `_stock_distance`. That is 68 failures, plus `test_probe_passes_the_real_kernel`, which fails the same way where `pfx.KERNEL` is available and is skipped where it is not.

- [ ] **Step 4: Extend the module's imports**

The import block of `src/sous/engine/projkernel.py` must hold at least these after this step (keep every line Tasks 2–3 added; `uv run ruff check --fix .` sorts the block). mlx, mlx_lm and mlx_vlm stay function-local:

```python
import contextlib
import importlib
import logging
import time
import traceback
import warnings
from typing import Any

from sous.engine import nax, tileattn, verifyattn
from sous.engine.int8prefill import GROUP, _error_line, _model_type, _root
```

The module must also define `logger = logging.getLogger("sous.engine.projkernel")` at top level. Task 2 adds it for `kernel_available()`'s compiler report; if it is missing, add it directly below the imports.

- [ ] **Step 5: Implement the gates, the pins, the warm-up, the probe, `enable()` and `static_reason()`**

The digests below were computed on the pristine sources with the installed mlx 0.32.2 and mlx-vlm 0.7.2 (`verifyattn._source_digest`, before any wrapper was installed). They match the first 16 hex digits recorded during design. `_project_gates` is defined on `Qwen3_5GatedDeltaNet` itself in 0.7.2, not inherited. Append to the end of `src/sous/engine/projkernel.py`:

```python
# ---- the gates, the pins, the warm-up, the probe and enable() ---------------------

# The dense Qwen3.5-family text model. The MoE variant reuses the exact verifier's
# class and the dense MLP for its shared expert, beside routed experts the kernel
# never serves, so it is refused by model type rather than by class.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
# DFlash drafts with modules of its own, which are never tagged, and verifies
# through the exact verifier's _linear and _linears. Other drafters are unmeasured.
SUPPORTED_DRAFTER_KINDS = frozenset({"dflash"})
# What decides that every target projection of at most MAX_ROWS rows reaches one
# of the two hooks, and that greedy verify takes its tokens from the hooked head:
# sha256 of inspect.getsource, keyed "module:qualname". getsource follows
# __wrapped__ through both hooks and int8's wrappers, so these are mlx's and
# mlx-vlm's own sources whatever is installed over them. Validated on mlx 0.32.2
# and mlx-vlm 0.7.2; add a digest only after re-reading that function against
# both hooks.
VALIDATED_PROJ_SOURCES: dict[str, str] = {
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linear": (
        "668b9a7064f30c63348028fb72f22119b73e2897e2d246024c80fbe54e0404dd"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears": (
        "5cdaf151d00f3ab314f4722b475eec88a5228060be36c321c1363314ab249e0a"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._feed_forward": (
        "df7c1083833e60422732d7b1b5dfabfb6a45f5ef073c8d1416810588dbf95434"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._gated_delta": (
        "3df1521798b530554798c23f5b2d386de0e88e2ac50fb9f391c5035d4cf8f30e"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._model": (
        "cec9745522cd79893849777daa1a22cfd7736f10309ecd480db7e4f7d4778fbb"
    ),
    "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward.__call__": (
        "969ac680c045372ddc2dfc4abfac22503ad04f30b8102528f9d3a6c499e6309c"
    ),
    "mlx_vlm.models.qwen3_5.language:Qwen3_5MLP.__call__": (
        "dc08d40bfdf18cf60619caad43c76a1094f3de1b14d2c53b9e2228fa698bc40a"
    ),
    "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet.__call__": (
        "0ab0be71156568f50a248f898d431b657b36b399166e29e25c16c7a45f6cc81b"
    ),
    # Defined on Qwen3_5GatedDeltaNet itself in 0.7.2: decode's b and a projections.
    "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet._project_gates": (
        "3d1f08bfeaa29f50a1d95aa6d5dda3766a93bc7068e81f9ce99b75b304bdbee6"
    ),
    "mlx_vlm.generate.ar:generate_step": (
        "ace632f74028d998d34c8b867b44632c13816e0db4567d37401a0bc657dc7a6a"
    ),
    "mlx.nn:QuantizedLinear.__call__": (
        "f43bed1629a938c0f00e37b4a9c85db84be59dc93b49a27de97c44228e264b8b"
    ),
    "mlx_vlm.speculative.dflash:_dflash_verify": (
        "39e4a6cd343ae7e054ae3584ff6879fc895494e413041f099ce08da14e3fc097"
    ),
    "mlx_vlm.speculative.dflash:_dflash_verify_greedy": (
        "1b472dc300554ab6017495bdeb18480be431578aa6958ff7b35c0145e8412971"
    ),
}
# The probe's through-the-hooks rows: a verify depth above today's block of 4.
_PROBE_ROWS = 5
# The row the probe plants a NaN in: inside every T = MAX_ROWS call, at neither end.
_NAN_ROW = 3


def _in_features(module: Any) -> int:
    """K of an eligible projection, read off its per-group scales."""
    return module.scales.shape[1] * GROUP


def _dispatch_key(group: tuple[Any, ...]) -> tuple[int, tuple[int, ...]]:
    """What picks a pipeline: K and each linear's N, in order. The variant follows
    from K and their sum, and T is the template's own argument."""
    return _in_features(group[0]), tuple(m.weight.shape[0] for m in group)


def _dispatch_groups(root: Any) -> list[tuple[Any, ...]]:
    """One linear group per dispatch key the model's projections can reach: the
    exact verifier's three fused groups, and every projection alone, the way
    decode, a short prefill tail and the verifier's singles call them."""
    import mlx.nn as nn

    groups: list[tuple[Any, ...]] = []
    for layer in root.model.layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            groups.append((gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a))
        else:
            attention = layer.self_attn
            groups.append((attention.q_proj, attention.k_proj, attention.v_proj))
        groups.append((layer.mlp.gate_proj, layer.mlp.up_proj))
    groups += [(m,) for _, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]
    unique: dict[tuple[int, tuple[int, ...]], tuple[Any, ...]] = {}
    for group in groups:
        unique.setdefault(_dispatch_key(group), group)
    return list(unique.values())


def warm_up(model: Any) -> None:
    """Compile and run every pipeline the model's projections can reach, each
    dispatch key at T = 1..MAX_ROWS: no compile first happens mid-turn, and a
    Metal toolchain that rejects the kernel fails here, inside enable()'s
    refusal path, rather than inside a user's turn. The inputs are zeros built
    on the GPU and evaluated before the first launch: a compile failure with a
    CPU-stream op in flight deadlocks mlx's exception path instead of raising."""
    import mlx.core as mx

    groups = _dispatch_groups(_root(model))
    inputs = {
        k: mx.zeros((1, MAX_ROWS, k), dtype=mx.bfloat16)
        for k in sorted({_in_features(group[0]) for group in groups})
    }
    mx.eval(list(inputs.values()))
    for group in groups:
        weights = [(m.weight, m.scales, m.biases) for m in group]
        x = inputs[_in_features(group[0])]
        mx.eval([mma(x[:, :t], weights) for t in range(1, MAX_ROWS + 1)])


def _stock_distance(got: Any, want: Any) -> float:
    """The relative RMS distance of `got` from `want` in fp32; infinite when
    `got` is not finite everywhere. A NaN `want` gives NaN, which fails any
    `<=` bound, as it must."""
    import mlx.core as mx

    if not mx.all(mx.isfinite(got)).item():
        return float("inf")
    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _check_group(group: tuple[Any, ...], names: list[str], key: Any) -> str | None:
    """One dispatch on its real weights: every row at T = 1..MAX_ROWS equals the
    row alone, each linear fused equals it alone, the result sits within
    REL_RMS_BOUND of stock quantized_matmul (parity cannot see a kernel wrong
    the same way on every call), and a NaN row leaves the others untouched."""
    import mlx.core as mx

    weights = [(m.weight, m.scales, m.biases) for m in group]
    x = mx.random.normal((1, MAX_ROWS, _in_features(group[0])), key=key).astype(mx.bfloat16)
    nan = mx.array(float("nan"), dtype=mx.bfloat16)
    x_nan = mx.where(mx.arange(MAX_ROWS)[None, :, None] == _NAN_ROW, nan, x)
    mx.eval(x, x_nan)
    alone = [mma(x[:, r : r + 1], weights) for r in range(MAX_ROWS)]
    refs = [mx.concatenate([rows[i] for rows in alone], axis=1) for i in range(len(group))]
    mx.eval(refs)
    del alone
    label = "+".join(names)
    full: tuple[Any, ...] = ()
    for t in range(1, MAX_ROWS + 1):
        full = mma(x[:, :t], weights)
        same = mx.stack([mx.array_equal(o, ref[:, :t]) for o, ref in zip(full, refs, strict=True)])
        if not mx.all(same).item():
            return f"T-invariance: {label} at T={t} differs from its rows alone"
    if len(group) > 1:
        for i, name in enumerate(names):
            (single,) = mma(x, [weights[i]])
            if not mx.array_equal(single, full[i]).item():
                return f"grouping: {name} fused in {label} differs from {name} alone"
    for (w, scales, biases), got, name in zip(weights, full, names, strict=True):
        want = mx.quantized_matmul(
            x, w, scales=scales, biases=biases, transpose=True, group_size=GROUP, bits=4
        )
        distance = _stock_distance(got, want)
        if not distance <= REL_RMS_BOUND:
            return f"correctness: {name} is {distance:.3g} from stock (bound {REL_RMS_BOUND})"
    keep = mx.array([r for r in range(MAX_ROWS) if r != _NAN_ROW])
    for got, clean, name in zip(mma(x_nan, weights), full, names, strict=True):
        if not mx.array_equal(mx.take(got, keep, axis=1), mx.take(clean, keep, axis=1)).item():
            return f"isolation: a NaN in row {_NAN_ROW} changed the other rows of {name}"
    return None


def _probe_steps(root: Any) -> str | None:
    """Every input is made with GPU ops and evaluated before the launches that
    read it, for the deadlock warm_up() describes."""
    import mlx.core as mx

    global _active
    layers = list(root.model.layers)
    full = next((layer for layer in layers if not layer.is_linear), None)
    linear = next((layer for layer in layers if layer.is_linear), None)
    if full is None or linear is None:
        return "the model lacks a full-attention or a linear-attention layer"
    names = {id(module): path for path, module in root.named_modules()}

    # Through both hooks, on the first full-attention layer (layer 3 of the
    # 27B): the exact verifier's feed-forward over _PROBE_ROWS rows (gate+up
    # fused, then down) against the plain MLP one row at a time, as decode
    # calls it. Its rows must be identical, which is what greedy parity rests on.
    mlp = full.mlp
    verifier = importlib.import_module(
        "mlx_vlm.models.qwen3_5.speculative_verifier"
    ).Qwen3_5BatchInvariantForward()
    x = mx.random.normal((1, _PROBE_ROWS, _in_features(mlp.gate_proj)), key=mx.random.key(0))
    x = x.astype(mx.bfloat16)
    mx.eval(x)
    before = dict(calls)
    _active = 1
    verified = verifier._feed_forward(mlp, x)
    mx.eval(verified)
    middle = dict(calls)
    plain = [mlp(x[:, r : r + 1]) for r in range(_PROBE_ROWS)]
    mx.eval(plain)
    _active = 0
    if middle["verify_kernel"] - before["verify_kernel"] != 2:
        return "the exact verifier's projections did not reach the kernel"
    if calls["plain_kernel"] - middle["plain_kernel"] != 3 * _PROBE_ROWS:
        return "the model's own projections did not reach the kernel"
    for r in range(_PROBE_ROWS):
        if not mx.array_equal(verified[:, r : r + 1], plain[r]).item():
            return f"parity: verify row {r} of {_PROBE_ROWS} differs from the one-row forward"
    del x, verified, plain

    # Directly, on real weights: the full-attention layer's q+k+v, o, gate+up and
    # down, the first linear-attention layer's qkv+z+b+a and out_proj (layer 0
    # of the 27B), and the head, which covers the staged variants.
    attention, gdn = full.self_attn, linear.linear_attn
    groups: list[tuple[Any, ...]] = [
        (attention.q_proj, attention.k_proj, attention.v_proj),
        (attention.o_proj,),
        (mlp.gate_proj, mlp.up_proj),
        (mlp.down_proj,),
        (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
        (gdn.out_proj,),
    ]
    head = getattr(root, "lm_head", None)
    if head is not None:
        groups.append((head,))
    for seed, group in enumerate(groups, start=1):
        reason = _check_group(group, [names.get(id(m), "?") for m in group], mx.random.key(seed))
        if reason is not None:
            return reason
    return None


def probe(model: Any) -> str | None:
    """None when the kernel may serve `model`'s projections on this GPU, else
    why not. Runs on the loading thread after enable() has tagged the model and
    installed the hooks; raises the flag only around the calls that must reach
    them, and leaves the flag clear and the counters as it found them. Its
    arrays are gone before the cache is cleared, so the budget measured next
    reads true."""
    import mlx.core as mx

    global _active
    counted = dict(calls)
    try:
        return _probe_steps(_root(model))
    except BaseException as e:
        # The traceback holds the steps' frames, and every probe array in them,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _active = 0
        calls.update(counted)
        mx.clear_cache()


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use the kernel: report why. An `expected` refusal says
    only that the kernel does not apply here (the tile, the model, the
    drafter), which is how most Macs load a default-on kernel: one INFO line,
    and the reason in the status. Any other refusal means it should have run
    and did not (a pinned dependency that drifted, a kernel this OS rejects, a
    failed probe): one warning, since a switch that is silently inert there
    hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("projection kernel unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: projection kernel unavailable ({reason}); "
            "verify and decode run the stock projections",
            stacklevel=3,
        )
    return {"state": "unavailable", "reason": reason, "probe_seconds": probe_seconds}


def _tile_reason(tile_status: dict[str, Any] | None) -> str | None:
    """The kernel rides on the attention tile: the platform, the core count and,
    with a drafter, exact verify attention are all the tile's gates, and block
    5 was measured only beside it."""
    state = (tile_status or {}).get("state")
    if state == "active":
        return None
    if state == "off":
        return "attention tile off"
    return (tile_status or {}).get("reason") or f"attention tile {state or 'unknown'}"


def _model_reason(model: Any) -> str | None:
    """Why this model's projections are not the ones the kernel takes, or None:
    every quantized projection of the language model must be eligible, since
    decode and verify route them all, and the activations must be bf16."""
    import mlx.core as mx
    import mlx.nn as nn

    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)"
    root = _root(model)
    projections = [(p, m) for p, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]
    if not projections:
        return "no quantized projections"
    for path, module in projections:
        if not eligible(module):
            return f"projection {path} is not affine 4-bit gs64 with bf16 scales and no bias"
    dtype = root.model.norm.weight.dtype
    if dtype != mx.bfloat16:
        return f"activation dtype {dtype} is not bfloat16"
    return None


def _drafter_reason(model: Any, drafter_kind: str | None) -> str | None:
    """A drafter whose verify could leave the hooks, or None. A target that
    defines speculative_verify_dflash_hidden would have greedy verify take its
    tokens from quantized_argmax, which neither hook sees."""
    if drafter_kind is not None and drafter_kind not in SUPPORTED_DRAFTER_KINDS:
        return f"drafter kind {drafter_kind!r} (dflash only)"
    if hasattr(_root(model), "speculative_verify_dflash_hidden"):
        return (
            "the language model defines speculative_verify_dflash_hidden: "
            "greedy verify would take its tokens outside the kernel"
        )
    return None


def _pins_reason() -> str | None:
    """Which pinned dependency drifted, or None."""
    import mlx.core as mx

    version = mx.__version__  # ty: ignore[unresolved-attribute]
    if version not in verifyattn.VALIDATED_MLX:
        return f"mlx {version} not validated"
    for key, digest in VALIDATED_PROJ_SOURCES.items():
        module, qualname = key.split(":")
        if verifyattn._source_digest(module, qualname) != digest:
            owner = "mlx" if module.split(".")[0] == "mlx" else "mlx-vlm"
            return f"{owner} {qualname} changed"
    return None


def enable(
    model: Any,
    *,
    enabled: bool,
    drafter_kind: str | None,
    tile_status: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether this load's small-row target projections run on the
    kernel. Returns the status the engine exposes and never raises: a model
    load must not fail because an accelerator is missing. probe_seconds covers
    the warm-up and the probe, the kernel's whole cost to a load."""
    global _active
    # First, on every call: a flag left from an earlier load must never serve
    # this one, whatever happens below. Tags left on an earlier model are inert
    # while it is clear, and a reference kept to untag them would keep that
    # model's weights alive past its unload, so they stay.
    _active = 0
    if not enabled:
        return {"state": "off", "reason": None, "probe_seconds": None}
    seconds: float | None = None
    try:
        reason = _tile_reason(tile_status)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = _model_reason(model) or _drafter_reason(model, drafter_kind)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = _pins_reason()
        if reason is not None:
            return _refuse(reason)
        tagged = 0
        start = time.perf_counter()
        try:
            try:
                warm_up(model)
            except Exception as e:  # noqa: BLE001 — a kernel this OS rejects refuses the load
                # The one-line reason cannot carry a compiler's full report; the log can.
                logger.warning("projection kernel: the warm-up failed", exc_info=True)
                reason = f"warm-up: {_error_line(str(e) or type(e).__name__)}"
            else:
                tagged = tag(model)
                install_hooks()
                reason = probe(model)
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            untag(model)
            return _refuse(reason, seconds)
        _active = 1
        logger.info("projection kernel: %d projections, probe %.2fs", tagged, seconds)
        return {"state": "active", "reason": None, "probe_seconds": seconds}
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active = 0
        with contextlib.suppress(Exception):
            untag(model)
        # The warning carries one line; the traceback goes to the log.
        logger.warning("projection kernel: enabling failed", exc_info=True)
        return _refuse(str(e) or type(e).__name__, seconds)


def static_reason(config: Any, model_config: dict, drafter_kind: str | None) -> str | None:
    """Why the kernel would not be active for this config, model and drafter,
    or None when it would be, decided without loading anything: sous tune
    resolves the block an unset speculative_block_size runs at by it. It
    mirrors enable()'s order (the switches, then the tile's machine and model
    gates, then the kernel's own) and reads the model from its config.json. A
    refusal only a load can see, a probe failure, pin drift or exact verify
    attention missing beside the drafter, is beyond it."""
    if not config.projection_kernel:
        return "projection kernel off"
    if not config.attention_tile:
        return "attention tile off"
    reason = nax.platform_reason()
    if reason is not None:
        return reason
    import mlx.core as mx

    _, reason = tileattn._split_target(str(mx.device_info().get("device_name", "")))
    if reason is not None:
        return reason
    model_type = model_config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        return f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)"
    text = model_config.get("text_config", model_config)
    shape = (
        text.get("num_attention_heads"),
        text.get("num_key_value_heads"),
        text.get("head_dim"),
    )
    if shape != (tileattn.HQ, tileattn.HKV, tileattn.D):
        return f"unsupported attention shape {shape[0]}/{shape[1]}/{shape[2]} (24/4/256 only)"
    reason = _quantization_reason(model_config.get("quantization"))
    if reason is not None:
        return reason
    if drafter_kind is not None and drafter_kind not in SUPPORTED_DRAFTER_KINDS:
        return f"drafter kind {drafter_kind!r} (dflash only)"
    return None


def _quantization_reason(quantization: Any) -> str | None:
    """The config.json side of _model_reason's eligibility: the checkpoint's
    default and every per-projection override under the language model must be
    affine 4-bit gs64. Embeddings never reach the kernel, so their overrides
    do not count; nor does the vision tower's."""
    if not isinstance(quantization, dict):
        return "unquantized weights (affine 4-bit gs64 only)"

    def spec(entry: dict) -> tuple[Any, Any, Any]:
        return entry.get("mode", "affine"), entry.get("bits"), entry.get("group_size")

    mode, bits, group = spec(quantization)
    if (mode, bits, group) != ("affine", 4, GROUP):
        return f"quantization {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"
    for path, entry in quantization.items():
        if not (isinstance(entry, dict) and path.startswith("language_model.")):
            continue
        if path.endswith("embed_tokens"):
            continue
        mode, bits, group = spec(entry)
        if (mode, bits, group) != ("affine", 4, GROUP):
            return f"projection {path} is {mode} {bits}-bit gs{group} (affine 4-bit gs64 only)"
    return None
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_projkernel.py -v`
Expected: PASS. The 69 tests this task adds pass, `test_probe_passes_the_real_kernel` included wherever `pfx.KERNEL` is available (the M2, CI's macos-15 and the M5 Pro); where it is not, that one test is skipped with `projection kernel unavailable: <reason>`. The new tests take about 4 s on the M2.

- [ ] **Step 7: Check the pins in a process with every wrapper installed**

Run:

```bash
uv run python -c "from sous.engine import int8prefill, projkernel; print(projkernel._pins_reason()); int8prefill.install_wrappers(); projkernel.install_hooks(); print(projkernel._pins_reason())"
```

Expected: `None` twice. The first proves the digests against the locked mlx and mlx-vlm. The second proves `getsource` follows `__wrapped__` through int8's three wrappers (Task 1) and both hooks (Task 3) to the originals. If either line names a function, stop. Re-read that function against both hooks before changing any digest: a changed source can move a projection out of the hooks' reach.

- [ ] **Step 8: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add src/sous/engine/projkernel.py tests/proj_fixtures.py tests/test_projkernel.py
git commit -m "feat(engine): gate, warm up and probe the projection kernel at load" -m "The kernel serves every target projection of up to eight rows on the
loaded model, so a load must prove it may before the flag is raised. It
rides on the attention tile, takes only the qwen3_5 target with every
projection affine 4-bit gs64 and bf16 activations, and refuses a target
whose greedy verify would take its tokens outside the hooks. Thirteen
source pins cover every mlx and mlx-vlm function that decides a
projection reaches a hook, and follow __wrapped__ so a second load, or
int8's wrappers, never read as drift.

The warm-up compiles every dispatch key at T = 1..8 before any tag, so a
toolchain that rejects the kernel refuses the load rather than a turn.
The probe then runs the verifier's feed-forward through hook 1 against
the plain MLP through hook 2 and requires identical rows, and checks
T-invariance, grouping, closeness to stock and NaN isolation on the real
weights of one layer of each kind and the head. A refusal that only says
the kernel does not apply is an INFO line; drift, a build failure or a
failed probe warns. static_reason lets sous tune resolve the block an
unset size runs at without loading a model.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and the full non-model suite passes.

---

### Task 5: The mlx-vlm contract tests

**Files:**
- Create: `tests/test_projkernel_contract.py`

**Interfaces:**
- Consumes:
  - Task 2: `projkernel.linears`, `projkernel.mma`, `projkernel.MAX_ROWS`, `pfx.stand_in_mma`, `pfx.kernel`.
  - Task 3:
    - `projkernel._active`, `projkernel.tag(model)`, `pfx.guard_proj_state(monkeypatch)`, `pfx.hooked(monkeypatch)`;
    - `projkernel.calls`. Hook 1 adds one `verify_kernel` per `_linear`/`_linears` call it serves, plus one `verify_split` when that call has more than 8 rows, and launches `mma` once per run. Hook 2 adds one `plain_kernel` per call it serves.
  - Task 4: `pfx.tiny_quantized_model()` (Task 3's builder, with the body Task 4 gave it: every projection K ≥ 256), `pfx.proj_ready(monkeypatch, real_pins=True)`, `projkernel.enable(...)`.
  - tile_fixtures: `VOCAB`, `prefill_cache(lm, ids)`, `verify_forward(lm, cache, tokens)`.
  - mlx-vlm 0.7.2: `mlx_vlm.generate.ar.generate_step`, pinned by Task 4.
- Produces: tests only.

The tiny model's forwards have fixed shapes, measured on the real mlx-vlm 0.7.2 forward while planning:
- A verify forward makes 17 `_linear`/`_linears` calls: four per layer and the head.
- A one-row forward makes 31 `nn.QuantizedLinear.__call__` calls: every projection once.
- The 40-token prefix prefill makes none of eight rows or fewer.
- `generate_step` over a 38-token prompt at `prefill_step_size=32`, decoding two tokens, makes 31 five-row calls and 93 one-row calls.
- Stock verify rows equalled one-row decode bit for bit at T = 2, 3, 5, 8 and 12 with these ids on the M2.

- [ ] **Step 1: Write the contract test**

Create `tests/test_projkernel_contract.py`:

```python
"""The projection kernel through mlx-vlm's real qwen3_5 forward, exact verifier and generate_step.

A tiny random qwen3_5 language model, quantized affine 4-bit gs64 as the 27B is, is prefilled
through the real forward. Verify rows, as DFlash runs them, go through the exact verifier's
``_linear``/``_linears`` (hook 1); step-by-step one-row decode on a fresh prefill of the same prefix
goes through ``nn.QuantizedLinear.__call__`` (hook 2): a hybrid cache cannot rewind past a decode
step. Stock's own exactness is asserted first. With the kernel (the plain-mlx stand-in on any GPU,
the real kernel wherever it compiles) every verify row must stay bitwise equal to decode, both
hooks must be reached by every projection, and with the flag cleared both paths give the stock bits
again. The real ``generate_step`` must reach hook 2 on a short prefill tail and on decode.
"""

import collections
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import projkernel  # noqa: E402 — after the importorskip guard
from tests import proj_fixtures as pfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402

PREFIX = 40  # above MAX_ROWS: the prefill itself stays on stock in every test
# Hook-1 calls per verify forward: four per layer (q+k+v and o, or qkv+z+b+a and
# out_proj, then gate+up and down) and the head.
VERIFY_CALLS = 4 * 4 + 1
# Hook-2 calls per one-row forward: seven projections per full-attention layer,
# eight per linear-attention layer, and the head.
PLAIN_CALLS = 2 * 7 + 2 * 8 + 1


@pytest.fixture(scope="module")
def tiny():
    lm = pfx.tiny_quantized_model()
    ids = mx.random.randint(0, tfx.VOCAB, (PREFIX + 12,), key=mx.random.key(148)).tolist()
    return lm, ids


def _verify_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    logits = tfx.verify_forward(lm, cache, ids[prefix : prefix + t])[0]
    return [logits[:, r] for r in range(t)]


def _decode_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    rows = []
    for tok in ids[prefix : prefix + t]:
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache).logits
        mx.eval(logits)
        rows.append(logits[:, -1])
    return rows


def _same(a, b):
    return [mx.array_equal(x, y).item() for x, y in zip(a, b, strict=True)]


def _delta(before, after):
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def _kernel_on(monkeypatch, lm, *, stand_in: bool = True) -> None:
    """Both hooks in place, `lm` tagged and the flag set for this test only;
    the stand-in or the real kernel behind them."""
    pfx.guard_proj_state(monkeypatch)
    pfx.hooked(monkeypatch)
    if not stand_in:
        monkeypatch.setattr(projkernel, "mma", projkernel.linears)
    monkeypatch.setattr(projkernel, "_active", 1)
    projkernel.tag(lm)


def _rows_of(x) -> int:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return rows


@pytest.mark.parametrize("t", [2, 5, 8])
def test_stock_verify_rows_equal_step_by_step_decode_on_the_quantized_model(tiny, t, monkeypatch):
    """Stock's own exactness on this GPU, which every assertion below rests on."""
    lm, ids = tiny
    pfx.guard_proj_state(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 0)
    assert all(_same(_verify_rows(lm, ids, PREFIX, t), _decode_rows(lm, ids, PREFIX, t)))


@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize("t", [2, 3, 5, 8])
def test_the_kernel_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, t, stand_in, monkeypatch
):
    lm, ids = tiny
    pfx.guard_proj_state(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 0)
    stock_verify = _verify_rows(lm, ids, PREFIX, t)
    stock_decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(stock_verify, stock_decode)), "stock's own exactness"

    _kernel_on(monkeypatch, lm, stand_in=stand_in)
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, t)
    verified = dict(projkernel.calls)
    decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(verify, decode)), t
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS}
    assert _delta(verified, projkernel.calls) == {"plain_kernel": PLAIN_CALLS * t}

    monkeypatch.setattr(projkernel, "_active", 0)
    start = dict(projkernel.calls)
    assert all(_same(_verify_rows(lm, ids, PREFIX, t), stock_verify))
    assert all(_same(_decode_rows(lm, ids, PREFIX, t), stock_decode))
    assert projkernel.calls == start


def test_a_kernel_whose_rows_move_with_the_row_count_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: the equality above is not vacuous. A kernel whose
    rows depend on how many rows share the call gives a five-row verify and
    one-row decode different logits."""
    lm, ids = tiny
    _kernel_on(monkeypatch, lm)

    def by_rows(x, weights):
        factor = 1 + _rows_of(x) / 64
        return tuple(
            (o.astype(mx.float32) * factor).astype(mx.bfloat16)
            for o in pfx.stand_in_mma(x, weights)
        )

    monkeypatch.setattr(projkernel, "mma", by_rows)
    assert not all(_same(_verify_rows(lm, ids, PREFIX, 5), _decode_rows(lm, ids, PREFIX, 5)))


@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
def test_a_verify_of_more_than_eight_rows_is_split_and_equals_its_rows_alone(
    tiny, stand_in, monkeypatch
):
    """Block 0 hands the drafter's own policy the depth, which can verify up to
    16 rows: every projection call is cut into runs of at most MAX_ROWS rows,
    and each of its twelve rows still equals one-row decode."""
    lm, ids = tiny
    t = 12
    _kernel_on(monkeypatch, lm, stand_in=stand_in)
    kernel = projkernel.mma
    rows: list[int] = []

    def counting(x, weights):
        rows.append(_rows_of(x))
        return kernel(x, weights)

    monkeypatch.setattr(projkernel, "mma", counting)
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, t)
    verified = dict(projkernel.calls)
    launches = list(rows)
    decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(verify, decode))
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS, "verify_split": VERIFY_CALLS}
    assert len(launches) == 2 * VERIFY_CALLS
    assert max(launches) <= projkernel.MAX_ROWS and sum(launches) == t * VERIFY_CALLS
    assert _delta(verified, projkernel.calls) == {"plain_kernel": PLAIN_CALLS * t}


class _Wrapper:
    """The mlx-vlm model generate_step drives: a text-only embedding helper
    over the tiny language model, handing the explicit positions and zero
    delta the VLM engine hands mlx-vlm."""

    def __init__(self, lm):
        self.language_model = lm
        self.config = types.SimpleNamespace(model_type="qwen3_5", eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        n = input_ids.shape[1]
        return types.SimpleNamespace(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids),
            to_dict=lambda: {
                "inputs_embeds": None,
                "position_ids": mx.arange(n, dtype=mx.int32)[None],
                "rope_deltas": mx.zeros((1, 1), dtype=mx.int32),
            },
        )


def test_generate_step_reaches_hook_two_on_the_prefill_tail_and_on_decode(tiny, monkeypatch):
    """mlx-vlm's own generate loop: a 38-token prompt prefills as a 32-row
    chunk (stock), a 5-row tail and the one-row step that ends every prefill,
    then decodes two tokens. Every projection of the tail, of that step and of
    each decode step is served, which is why stored forks depend on the kernel."""
    from mlx_vlm.generate.ar import generate_step

    lm, _ = tiny
    _kernel_on(monkeypatch, lm)
    rows: list[int] = []

    def counting(x, weights):
        rows.append(_rows_of(x))
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", counting)
    prompt = mx.random.randint(0, tfx.VOCAB, (1, 38), key=mx.random.key(7)).astype(mx.int32)
    start = dict(projkernel.calls)
    tokens = [
        token
        for token, _ in generate_step(
            prompt,
            _Wrapper(lm),  # ty: ignore[invalid-argument-type]
            None,
            None,
            max_tokens=2,
            prefill_step_size=32,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        )
    ]
    assert len(tokens) == 2
    assert collections.Counter(rows) == {5: PLAIN_CALLS, 1: 3 * PLAIN_CALLS}
    assert _delta(start, projkernel.calls) == {"plain_kernel": 4 * PLAIN_CALLS}


def test_enable_on_the_tiny_model_serves_both_real_paths(tiny, monkeypatch):
    """enable() itself, its real pins, warm-up and probe and the hooks it
    installs, over the real forward and exact verifier."""
    lm, ids = tiny
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    status = projkernel.enable(lm, enabled=True, drafter_kind="dflash", tile_status=tile)
    assert status["state"] == "active", status
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, 5)
    verified = dict(projkernel.calls)
    decode = _decode_rows(lm, ids, PREFIX, 5)
    assert all(_same(verify, decode))
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS}
    assert _delta(verified, projkernel.calls) == {"plain_kernel": 5 * PLAIN_CALLS}
```

- [ ] **Step 2: Run the contract test**

Run: `uv run pytest tests/test_projkernel_contract.py -v -rs`

Expected wherever `pfx.KERNEL` is available (the M2, CI's macos-15 and the M5 Pro): `16 passed`. Where it is not: `11 passed, 5 skipped`. The skips are the five `kernel` variants, with reason `projection kernel unavailable: <reason>`.
- These tests pin behaviour that Tasks 2–4 built, so they pass as written. A failed parity, counter or split assertion is a bug in those tasks: fix it there, and never loosen the assertion.
- The sensitivity test is what makes the parity assertions meaningful. If it ever passes vacuously (its verify and decode rows equal), the hooks are not reaching `projkernel.mma` at call time.
- If `test_stock_verify_rows_equal_step_by_step_decode_on_the_quantized_model` ever fails on CI, stock mlx-vlm is not exact on that GPU for this model. Every kernel assertion rests on that precondition. Report it with the failing T rather than weakening a kernel assertion.

- [ ] **Step 3: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add tests/test_projkernel_contract.py
git commit -m "test(engine): pin the projection kernel's parity through mlx-vlm's real paths" -m "The unit tests prove the kernel entries, the hooks and the gates in
isolation. These pin the property the kernel exists for through mlx-vlm's
own code, on a tiny qwen3_5 model quantized as the 27B is. Every verify
row through the exact verifier's _linear and _linears equals one-row
decode through nn.QuantizedLinear.__call__, a verify above eight rows is
split and still equals decode, and the real generate_step reaches the
kernel on a short prefill tail, on the step that ends every prefill and on
decode. This holds with the stand-in on any GPU and with the real kernel
wherever it compiles, and enable() itself installs hooks that do all of
it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and the full non-model suite passes.

---

### Task 6: Config, engine wiring and the block default

**Files:**
- Modify: `src/sous/config.py`:
  - `_KNOWN` at :31-34;
  - the speculative comment and fields at :90-95, gaining `speculative_block_explicit` after `speculative_block_size`;
  - the new `projection_kernel` field after `attention_tile` at :110;
  - `SPECULATIVE_BLOCK_KERNEL` after `SPECULATIVE_BLOCK_DEFAULT` at :301;
  - `_speculative_block_size` at :310-335;
  - the new `_projection_kernel` after `_attention_tile`, which ends at :405;
  - `load_config` at :485-503.
- Modify: `src/sous/engine/vlm.py`:
  - the module imports at :11;
  - the `__init__` signature at :68-73;
  - the drafter pin at :109, which moves below a new `projkernel.enable` call placed after `tileattn.enable` (:120-125);
  - a new `draft_block` property after `drafter` (:173-178);
  - `unload()` at :550-553.
- Modify: `src/sous/engine/lm.py`: `__init__` at :55-56.
- Modify: `src/sous/engine/base.py`:
  - `_default_factory`'s signature at :248-253 and its VLM branch at :309-320;
  - `default_engine_factory` at :357-364;
  - `ManagedEngine`, after the `attention_tile_status` property at :405-409.
- Modify: `src/sous/tune/suite/runner.py`: the `CountingEngine.__getattr__` comment at :132-134 only.
- Test:
  - `tests/test_config.py` at :213-286;
  - `tests/test_engine_base.py` at :1207-1208, :1340-1403 and :2077-2090;
  - `tests/test_engine_unloaded.py` at :94-121.

Line numbers are those of the spec commit; Tasks 1–5 do not touch these files.

**Interfaces:**
- Consumes:
  - Task 4: `projkernel.enable(model, *, enabled: bool, drafter_kind: str | None, tile_status: dict) -> dict` with keys `state`, `reason` and `probe_seconds`. With `enabled=False` it clears `_active` and returns `off` before any gate, which is what keeps the stub models of the existing VLMEngine tests away from every gate.
  - Task 3: `projkernel.clear()` and `projkernel._active`.
  - Existing:
    - `tileattn.enable`, `VLMEngine.attention_tile_status`, `vlm._pin_block_size`, `draftctx.enable`, `base.measure_cache_budget`;
    - the test helpers `_positionless_model`, `_positional_factory`, `_stub`, `_cfg`, `_RecordingTokenizer` and `FakeEngine` in test_engine_base.py, and `_unloaded_vlm` in test_engine_unloaded.py.
- Produces:
  - `config.SPECULATIVE_BLOCK_KERNEL = 5`, beside the unchanged `SPECULATIVE_BLOCK_DEFAULT = 4` and `SPECULATIVE_BLOCK_MAX = 5`.
  - `SousConfig.speculative_block_explicit: bool = False`, after `speculative_block_size`.
  - `SousConfig.projection_kernel: bool = True`, after `attention_tile`, and `'projection_kernel'` in `_KNOWN['model']`.
  - `config._speculative_block_size(model: dict) -> tuple[int, bool]`: the block and whether it was set. Absent gives `(4, False)`; invalid gives `(4, False)` and a warning ending `; using the default`; above 5 gives `(5, True)` and the clamp warning; otherwise `(value, True)`.
  - `config._projection_kernel(model: dict) -> bool`: a non-bool warns `sous config: [model].projection_kernel {value!r} must be true or false; using false` and gives False.
  - `VLMEngine.__init__(model_id, temperature, top_p, top_k, prompt_cache, draft_id, draft_block_size: int = 0, draft_block_explicit: bool = True, cache_budget, reserve_bytes, int8_prefill, attention_tile: bool = False, projection_kernel: bool = False, fork_dir, fork_budget, weights_identity)`.
  - `VLMEngine.projection_kernel_status: dict`, set from `projkernel.enable(self._model, enabled=projection_kernel, drafter_kind=self._draft_kind if self._draft is not None else None, tile_status=self.attention_tile_status)`.
  - Load order in `VLMEngine.__init__`: int8 → drafter → verifyattn → tileattn → projkernel → block resolution and `_pin_block_size` → draftctx → ForkStore → `measure_cache_budget`.
  - Block resolution: when `draft_block_explicit` is False, a drafter loaded and `projection_kernel_status['state'] == 'active'`, `self._draft_block_size = SPECULATIVE_BLOCK_KERNEL`. `_pin_block_size(self._draft, self._draft_block_size)` then runs unconditionally; it does nothing without a drafter or at block 0.
  - `VLMEngine.draft_block -> int | None`, a property: `self._draft_block_size` when a drafter loaded, else None.
  - `VLMEngine.unload()` calls `projkernel.clear()` right after `tileattn.clear()`.
  - vlm.py's module imports gain `from sous.config import SPECULATIVE_BLOCK_KERNEL` and `projkernel`.
  - `LMEngine.projection_kernel_status = {'state': 'off', 'reason': None}`, set before the fork store.
  - `base._default_factory(..., draft_block_size: int = 0, draft_block_explicit: bool = True, ..., attention_tile: bool = False, projection_kernel: bool = False, fork_dir=..., fork_budget=...)`, passing both new values to `vlm.VLMEngine` only.
  - `default_engine_factory` passes `draft_block_explicit=config.speculative_block_explicit` and `projection_kernel=config.projection_kernel`.
  - `ManagedEngine.projection_kernel_status -> dict | None` and `ManagedEngine.draft_block -> int | None`, each `getattr(self._inner, name, None)`.

Why the defaults differ:
- The config defaults (`projection_kernel = True`, `speculative_block_explicit = False`) are what the daemon and `sous tune` get through `default_engine_factory(config)`.
- The constructors default to `projection_kernel=False` and `draft_block_explicit=True`, as `int8_prefill` and `attention_tile` do. An engine built directly (the stub tests, the model tests, scripts) keeps the stock projections and the block it was given.
- `ManagedEngine` reads both new attributes with a `None` default, not with `{"state": "off", ...}`. A fake then stays silent on the load line and in the status document, as it does for the tile. `test_get_logs_the_load_once_with_its_duration` (test_engine_base.py:94-105) pins a fake's line ending at ` model=fake/model`, and the `"fakes without the attribute stay silent"` status tests rest on the same rule. The LM backend still reports `off` itself.

Why the resolution and the pin sit after `projkernel.enable`:
- The kernel needs the tile's status, so it runs after `tileattn.enable`. The block needs the kernel's status, so it resolves after that.
- The pin used to run straight after the drafter loaded. Moving it is safe: verifyattn never sees the drafter, tileattn reads only its kind, and draftctx never reads `prefer_requested_block_size`.
- `self._draft_block_size` is reassigned before the pin, so the pin and `generate()` (`"draft_block_size": self._draft_block_size or None`) both see the resolved block.

- [ ] **Step 1: Write the failing config tests**

In `tests/test_config.py`, add the projection_kernel tests after the attention_tile ones, rewrite `test_speculative_defaults` for the unset flag and add a constant check after it.

Replace:

```python
    with pytest.warns(
        UserWarning, match=r"\[model\]\.attention_tile .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.attention_tile is False


def test_speculative_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    # 4 measured best on the M5 Pro on real subagent turns (#118); 0 would hand
    # the choice back to the drafter's own adaptive policy.
    assert cfg.speculative_block_size == 4
```

with:

```python
    with pytest.warns(
        UserWarning, match=r"\[model\]\.attention_tile .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.attention_tile is False


# ---- [model].projection_kernel ------------------------------------------------


def test_projection_kernel_defaults_on(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x/y'\n")
    assert load_config(p).projection_kernel is True
    assert SousConfig().projection_kernel is True


def test_projection_kernel_reads_false_without_an_unknown_key_warning(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprojection_kernel = false\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.projection_kernel is False


@pytest.mark.parametrize("bad", ['"yes"', "1", "0.5"])
def test_projection_kernel_rejects_non_booleans_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nprojection_kernel = {bad}\n")
    with pytest.warns(
        UserWarning, match=r"\[model\]\.projection_kernel .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.projection_kernel is False


def test_speculative_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    # Unset, the block is the engine's to resolve at load: 5 where the
    # projection kernel serves the load, 4 elsewhere. 0 would hand the choice
    # back to the drafter's own adaptive policy.
    assert cfg.speculative_block_size == 4
    assert cfg.speculative_block_explicit is False
    assert SousConfig().speculative_block_explicit is False


def test_speculative_block_kernel_is_five_within_the_clamp():
    from sous.config import (
        SPECULATIVE_BLOCK_DEFAULT,
        SPECULATIVE_BLOCK_KERNEL,
        SPECULATIVE_BLOCK_MAX,
    )

    assert (SPECULATIVE_BLOCK_DEFAULT, SPECULATIVE_BLOCK_KERNEL, SPECULATIVE_BLOCK_MAX) == (4, 5, 5)
```

In `test_speculative_keys_from_toml_without_unknown_key_warnings`:

Replace:

```python
    assert cfg.speculative_draft_id == ""
    assert cfg.speculative_block_size == 3
```

with:

```python
    assert cfg.speculative_draft_id == ""
    assert cfg.speculative_block_size == 3
    assert cfg.speculative_block_explicit is True
```

In `test_speculative_block_size_one_or_negative_warns_and_uses_the_default`, an invalid value counts as unset and its warning names no number:

Replace:

```python
    """mlx-vlm treats the override as the total verify-block size and ends the
    round loop at <= 1 — a configured 1 would silently truncate every response
    to one token. Invalid values degrade to the default (4) with a warning,
    matching the [context] policy stance."""
    for bad in ("1", "-3", "true", '"3"'):
        p = tmp_path / f"c{len(bad)}{bad[0]}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {bad}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == 4, bad
        assert any("speculative_block_size" in str(w.message) for w in caught), bad
```

with:

```python
    """mlx-vlm treats the override as the total verify-block size and ends the
    round loop at <= 1 — a configured 1 would silently truncate every response
    to one token. Invalid values degrade to the default with a warning,
    matching the [context] policy stance, and count as unset: the engine
    resolves the block as if the key were absent, so the warning cannot name
    a number."""
    for bad in ("1", "-3", "true", '"3"'):
        p = tmp_path / f"c{len(bad)}{bad[0]}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {bad}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == 4, bad
        assert cfg.speculative_block_explicit is False, bad
        assert any(
            "speculative_block_size" in str(w.message)
            and str(w.message).endswith("; using the default")
            for w in caught
        ), bad
```

In `test_speculative_block_size_above_five_warns_and_is_clamped_to_five`:

Replace:

```python
        assert cfg.speculative_block_size == 5, big
        assert any(
```

with:

```python
        assert cfg.speculative_block_size == 5, big
        assert cfg.speculative_block_explicit is True, big  # the intent stands
        assert any(
```

In `test_speculative_block_size_zero_and_two_to_five_accepted`, once inside the loop:

Replace:

```python
        assert cfg.speculative_block_size == ok
        assert not [w for w in caught if "speculative_block_size" in str(w.message)]
```

with:

```python
        assert cfg.speculative_block_size == ok
        assert cfg.speculative_block_explicit is True
        assert not [w for w in caught if "speculative_block_size" in str(w.message)]
```

and once for 0:

Replace:

```python
    assert cfg.speculative_block_size == 0
    assert not [w for w in caught if "speculative_block_size" in str(w.message)]
```

with:

```python
    assert cfg.speculative_block_size == 0
    assert cfg.speculative_block_explicit is True  # the drafter's policy, chosen
    assert not [w for w in caught if "speculative_block_size" in str(w.message)]
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "projection_kernel or speculative" -v`

Expected: 11 failed, 56 deselected.
- `test_projection_kernel_defaults_on`: `AttributeError: 'SousConfig' object has no attribute 'projection_kernel'`.
- The false-value test fails inside the error filter with `UserWarning: sous config: unknown key 'projection_kernel' in [model]`.
- The three non-bool cases: `Failed: Regex pattern did not match any of the 1 warnings emitted`, the one warning being that unknown-key one.
- `test_speculative_block_kernel_is_five_within_the_clamp`: `ImportError: cannot import name 'SPECULATIVE_BLOCK_KERNEL' from 'sous.config'`.
- The five speculative tests: `AttributeError: 'SousConfig' object has no attribute 'speculative_block_explicit'`.

- [ ] **Step 3: Add the config fields, the constant and the parsers**

In `src/sous/config.py`:

Replace:

```python
        "int8_prefill",
        "attention_tile",
    },
}
```

with:

```python
        "int8_prefill",
        "attention_tile",
        "projection_kernel",
    },
}
```

The block's comment now says who resolves an unset block, and the flag follows the field:

Replace:

```python
    # load), the engine logs and continues without it. Block size 4 measured
    # best on the M5 Pro on real subagent turns (#118: +3–5% decode over 3;
    # 2 and 6 ran clearly slower, 5 level with 3); 0 hands the choice back to
    # the drafter's adaptive policy. Above 5 is clamped (SPECULATIVE_BLOCK_MAX).
    speculative_draft_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    speculative_block_size: int = 4
```

with:

```python
    # load), the engine logs and continues without it. Left unset, the block
    # is resolved at load: SPECULATIVE_BLOCK_KERNEL where the projection
    # kernel serves the load, SPECULATIVE_BLOCK_DEFAULT where it does not.
    # Without the kernel, block size 4 measured best on the M5 Pro on real
    # subagent turns (#118: +3–5% decode over 3; 2 and 6 ran clearly slower,
    # 5 level with 3). 0 hands the choice back to the drafter's adaptive
    # policy. Above 5 is clamped (SPECULATIVE_BLOCK_MAX).
    speculative_draft_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    speculative_block_size: int = 4
    # Whether [model].speculative_block_size was set, a clamped value
    # included: only a block the user left unset is the engine's to resolve.
    speculative_block_explicit: bool = False
```

Replace:

```python
    attention_tile: bool = True
    # Reuse one KV cache across the turns of a conversation, prefilling only
```

with:

```python
    attention_tile: bool = True
    # The target's small-row quantized projections through one simdgroup-MMA
    # kernel whose rows do not depend on how many rows or linears share the
    # call: every verify forward's and, so that greedy output stays the same
    # with the drafter on or off, every other target call of up to 8 rows,
    # decode included. With it a 5-row verify round's projections cost about
    # what 4 rows' do, which makes block 5 pay: 1.10x sampled decode over
    # block 4 without the kernel on real subagent turns at 44-77K (M5 Pro).
    # Decode without a drafter runs at about 0.78x. Not bit-identical to
    # stock's kernels, hence the switch; false runs the stock paths, and an
    # unset speculative_block_size then resolves to 4. Active only where the
    # attention tile is, `unavailable` with the reason anywhere else. Read
    # once, at daemon start.
    projection_kernel: bool = True
    # Reuse one KV cache across the turns of a conversation, prefilling only
```

Replace:

```python
SPECULATIVE_BLOCK_DEFAULT = 4
```

with:

```python
SPECULATIVE_BLOCK_DEFAULT = 4
# What an unset speculative_block_size resolves to where the projection
# kernel serves the load: its verify projections cost about the same at 5
# rows as at 4, so the deeper block's extra accepted tokens come nearly free.
SPECULATIVE_BLOCK_KERNEL = 5
```

`_speculative_block_size` returns the flag with the block. It reads presence from the table itself, because `model.get(key, default)` cannot tell an unset key from one set to the default:

Replace:

```python
def _speculative_block_size(model: dict) -> int:
    """Validated [model].speculative_block_size: 0 (the drafter's own policy)
    or 2..5, degrading to the default with a warning — same stance as the rest
    of the file. This one is a silent-truncation knob: mlx-vlm treats the value
    as the total verify-block size and ends its round loop when it is <= 1,
    so a configured 1 (or a negative) would cap every response at a single
    token without any error. Above the maximum is clamped rather than
    defaulted: the intent ("as deep as pays") is clear, only the number is
    past where it pays."""
    value = model.get("speculative_block_size", SPECULATIVE_BLOCK_DEFAULT)
    if isinstance(value, bool) or not isinstance(value, int) or value == 1 or value < 0:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} must be 0 (auto) "
            f"or an integer from 2 to {SPECULATIVE_BLOCK_MAX}; using {SPECULATIVE_BLOCK_DEFAULT}",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_DEFAULT
    if value > SPECULATIVE_BLOCK_MAX:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} exceeds "
            f"{SPECULATIVE_BLOCK_MAX}, the largest verify block that has paid in "
            f"measurement (default model, M5 Pro); using {SPECULATIVE_BLOCK_MAX}",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_MAX
    return value
```

with:

```python
def _speculative_block_size(model: dict) -> tuple[int, bool]:
    """Validated [model].speculative_block_size, and whether it was set: 0
    (the drafter's own policy) or 2..5, degrading to the default with a
    warning — same stance as the rest of the file. This one is a
    silent-truncation knob: mlx-vlm treats the value as the total verify-block
    size and ends its round loop when it is <= 1, so a configured 1 (or a
    negative) would cap every response at a single token without any error.
    Above the maximum is clamped rather than defaulted: the intent ("as deep
    as pays") is clear, only the number is past where it pays, so a clamped
    value still counts as set. An invalid one does not: it falls back to the
    default, which the engine resolves at load as it does an unset one, so
    the warning names no number."""
    if "speculative_block_size" not in model:
        return SPECULATIVE_BLOCK_DEFAULT, False
    value = model["speculative_block_size"]
    if isinstance(value, bool) or not isinstance(value, int) or value == 1 or value < 0:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} must be 0 (auto) "
            f"or an integer from 2 to {SPECULATIVE_BLOCK_MAX}; using the default",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_DEFAULT, False
    if value > SPECULATIVE_BLOCK_MAX:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} exceeds "
            f"{SPECULATIVE_BLOCK_MAX}, the largest verify block that has paid in "
            f"measurement (default model, M5 Pro); using {SPECULATIVE_BLOCK_MAX}",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_MAX, True
    return value, True
```

Directly after `_attention_tile`:

Replace:

```python
    warnings.warn(
        f"sous config: [model].attention_tile {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

with:

```python
    warnings.warn(
        f"sous config: [model].attention_tile {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False


def _projection_kernel(model: dict) -> bool:
    """[model].projection_kernel: true or false; anything else warns and means
    false, the stock projections, by _int8_prefill's rule that a typo must not
    select changed numerics."""
    value = model.get("projection_kernel", True)
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"sous config: [model].projection_kernel {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

In `load_config`, unpack the pair before the constructor call; the warnings still point at `load_config`'s caller, since `_speculative_block_size` is still called from `load_config` itself:

Replace:

```python
    local_models, generation_timeout = _server_values(server)
    return SousConfig(
```

with:

```python
    local_models, generation_timeout = _server_values(server)
    block, block_explicit = _speculative_block_size(model)
    return SousConfig(
```

Replace:

```python
        speculative_block_size=_speculative_block_size(model),
        int8_prefill=_int8_prefill(model),
        attention_tile=_attention_tile(model),
```

with:

```python
        speculative_block_size=block,
        speculative_block_explicit=block_explicit,
        int8_prefill=_int8_prefill(model),
        attention_tile=_attention_tile(model),
        projection_kernel=_projection_kernel(model),
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`

Expected: 67 passed.

- [ ] **Step 5: Write the failing engine tests**

In `tests/test_engine_base.py`, after `test_default_factory_passes_attention_tile_to_the_vlm_engine_only`:

Replace:

```python
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, attention_tile=True)
    assert "attention_tile" not in seen["lm"]
```

with:

```python
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, attention_tile=True)
    assert "attention_tile" not in seen["lm"]


@pytest.mark.parametrize("value", [True, False])
def test_engine_manager_threads_projection_kernel_into_default_factory(
    monkeypatch, tmp_path, value
):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(_cfg(tmp_path, projection_kernel=value)).get()
    assert seen["kwargs"]["projection_kernel"] is value


@pytest.mark.parametrize("explicit", [True, False])
def test_engine_manager_threads_the_block_flag_into_default_factory(
    monkeypatch, tmp_path, explicit
):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(_cfg(tmp_path, speculative_block_explicit=explicit)).get()
    assert seen["kwargs"]["draft_block_explicit"] is explicit


def test_default_factory_passes_projection_kernel_and_the_block_flag_to_the_vlm_engine_only(
    monkeypatch,
):
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
        "m", 0.7, 0.8, 20, True, cache_budget=0, projection_kernel=True, draft_block_explicit=False
    )
    assert seen["vlm"]["projection_kernel"] is True
    assert seen["vlm"]["draft_block_explicit"] is False
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3_5"})
    base._default_factory(
        "m", 0.7, 0.8, 20, True, cache_budget=0, projection_kernel=True, draft_block_explicit=False
    )
    assert "projection_kernel" not in seen["lm"]
    assert "draft_block_explicit" not in seen["lm"]


def test_default_factory_takes_a_direct_callers_block_as_an_explicit_block(monkeypatch):
    """Scripts and tests that call the factory themselves get the block they
    pass, and the stock projections unless they ask for the kernel."""
    from sous.engine import base, vlm

    seen: dict[str, dict] = {}

    class RecordingVLM:
        def __init__(self, model_id, **kwargs):
            seen["vlm"] = kwargs

    monkeypatch.setattr(vlm, "VLMEngine", RecordingVLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"vision_config": {}})
    monkeypatch.setattr("sous.engine.window.kv_bytes_per_token", lambda cfg: 1024)
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, draft_block_size=4)
    assert seen["vlm"]["draft_block_explicit"] is True
    assert seen["vlm"]["projection_kernel"] is False
```

Replace the tile's enable-order test whole: it gains the kernel and the pin, and its name says so.

Replace:

```python
@pytest.mark.parametrize("drafter_loads", [True, False])
def test_vlm_engine_enables_the_attention_tile_after_the_verifier_and_before_the_budget(
    monkeypatch, drafter_loads
):
    from sous.engine import base, draftctx, tileattn, verifyattn, vlm

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
    verifier_status = {"state": "active", "reason": None, "probe_seconds": 0.21}
    tile_status = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43}

    def verify_enable(m, *, enabled):
        order.append("verify")
        return verifier_status

    def tile_enable(m, *, enabled, drafter_kind, verify_status):
        order.append("tile")
        seen.update(
            model=m, enabled=enabled, drafter_kind=drafter_kind, verify_status=verify_status
        )
        return tile_status

    def context_enable(m, d, kind):
        order.append("draft_context")
        return draftctx.OFF

    def budget(reserve_bytes):
        order.append("budget")
        return 0

    monkeypatch.setattr(verifyattn, "enable", verify_enable)
    monkeypatch.setattr(tileattn, "enable", tile_enable)
    monkeypatch.setattr(draftctx, "enable", context_enable)
    monkeypatch.setattr(base, "measure_cache_budget", budget)
    if drafter_loads:
        engine = vlm.VLMEngine("test/model", draft_id="z-lab/drafter", attention_tile=True)
    else:
        with pytest.warns(UserWarning, match="speculative drafter"):
            engine = vlm.VLMEngine("test/model", draft_id="z-lab/drafter", attention_tile=True)
    assert order == ["verify", "tile", "draft_context", "budget"]
    assert seen["model"] is model and seen["enabled"] is True
    assert seen["drafter_kind"] == ("dflash" if drafter_loads else None)
    assert seen["verify_status"] is engine.verify_attention_status is verifier_status
    assert engine.attention_tile_status is tile_status
```

with:

```python
@pytest.mark.parametrize("drafter_loads", [True, False])
def test_vlm_engine_enables_the_tile_then_the_projection_kernel_then_pins_the_block(
    monkeypatch, drafter_loads
):
    from sous.engine import base, draftctx, projkernel, tileattn, verifyattn, vlm

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
    verifier_status = {"state": "active", "reason": None, "probe_seconds": 0.21}
    tile_status = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43}
    proj_status = {"state": "active", "reason": None, "probe_seconds": 0.37}

    def verify_enable(m, *, enabled):
        order.append("verify")
        return verifier_status

    def tile_enable(m, *, enabled, drafter_kind, verify_status):
        order.append("tile")
        seen.update(
            model=m, enabled=enabled, drafter_kind=drafter_kind, verify_status=verify_status
        )
        return tile_status

    def proj_enable(m, *, enabled, drafter_kind, tile_status):
        order.append("projection_kernel")
        seen.update(
            proj_model=m,
            proj_enabled=enabled,
            proj_drafter_kind=drafter_kind,
            proj_tile_status=tile_status,
        )
        return proj_status

    def pin(drafter, block_size):
        order.append("pin")

    def context_enable(m, d, kind):
        order.append("draft_context")
        return draftctx.OFF

    def budget(reserve_bytes):
        order.append("budget")
        return 0

    monkeypatch.setattr(verifyattn, "enable", verify_enable)
    monkeypatch.setattr(tileattn, "enable", tile_enable)
    monkeypatch.setattr(projkernel, "enable", proj_enable)
    monkeypatch.setattr(vlm, "_pin_block_size", pin)
    monkeypatch.setattr(draftctx, "enable", context_enable)
    monkeypatch.setattr(base, "measure_cache_budget", budget)
    if drafter_loads:
        engine = vlm.VLMEngine(
            "test/model", draft_id="z-lab/drafter", attention_tile=True, projection_kernel=True
        )
    else:
        with pytest.warns(UserWarning, match="speculative drafter"):
            engine = vlm.VLMEngine(
                "test/model", draft_id="z-lab/drafter", attention_tile=True, projection_kernel=True
            )
    assert order == ["verify", "tile", "projection_kernel", "pin", "draft_context", "budget"]
    assert seen["model"] is model and seen["enabled"] is True
    assert seen["drafter_kind"] == ("dflash" if drafter_loads else None)
    assert seen["verify_status"] is engine.verify_attention_status is verifier_status
    assert engine.attention_tile_status is tile_status
    assert seen["proj_model"] is model and seen["proj_enabled"] is True
    assert seen["proj_drafter_kind"] == ("dflash" if drafter_loads else None)
    assert seen["proj_tile_status"] is tile_status
    assert engine.projection_kernel_status is proj_status
```

After `test_lm_engine_reports_the_attention_tile_off`, the LM backend's report, a helper that builds a drafted engine over stubs, and the block's resolution:

Replace:

```python
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.attention_tile_status == {"state": "off", "reason": None}
```

with:

```python
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.attention_tile_status == {"state": "off", "reason": None}


def test_lm_engine_reports_the_projection_kernel_off(monkeypatch):
    from sous.engine.lm import LMEngine

    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (object(), _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.projection_kernel_status == {"state": "off", "reason": None}


def _drafted_vlm(monkeypatch, proj_state: str, **kwargs):
    """A VLMEngine over stubs whose drafter loads and whose projection kernel
    reports `proj_state`; returned with its drafter and the blocks pinned."""
    from sous.engine import draftctx, projkernel, verifyattn, vlm

    model = _positionless_model()
    processor = types.SimpleNamespace(tokenizer=_RecordingTokenizer())
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    drafter = types.SimpleNamespace(prefer_requested_block_size=False)
    monkeypatch.setattr(vlm, "_load_quantized_drafter", lambda m, draft_id: (drafter, "dflash"))
    monkeypatch.setattr(
        verifyattn,
        "enable",
        lambda m, *, enabled: {"state": "active", "reason": None, "probe_seconds": 0.21},
    )
    status = {"state": proj_state, "reason": None, "probe_seconds": None}
    monkeypatch.setattr(
        projkernel, "enable", lambda m, *, enabled, drafter_kind, tile_status: status
    )
    monkeypatch.setattr(draftctx, "enable", lambda m, d, kind: draftctx.OFF)
    pinned: list[int] = []
    real_pin = vlm._pin_block_size

    def pin(d, block_size):
        pinned.append(block_size)
        real_pin(d, block_size)

    monkeypatch.setattr(vlm, "_pin_block_size", pin)
    engine = vlm.VLMEngine(
        "test/model", cache_budget=0, draft_id="z-lab/drafter", projection_kernel=True, **kwargs
    )
    return engine, drafter, pinned


@pytest.mark.parametrize(
    ("explicit", "proj_state", "block", "expected"),
    [
        (False, "active", 4, 5),
        (False, "unavailable", 4, 4),
        (False, "off", 4, 4),
        (True, "active", 4, 4),
        (True, "active", 3, 3),
        (True, "active", 0, 0),
    ],
)
def test_vlm_engine_resolves_an_unset_draft_block_to_five_only_where_the_kernel_is_active(
    monkeypatch, explicit, proj_state, block, expected
):
    engine, drafter, pinned = _drafted_vlm(
        monkeypatch, proj_state, draft_block_size=block, draft_block_explicit=explicit
    )
    assert engine._draft_block_size == expected and engine.draft_block == expected
    assert pinned == [expected], "the pin must see the resolved block"
    assert drafter.prefer_requested_block_size is (expected > 0)


def test_vlm_engine_built_directly_keeps_the_draft_block_it_was_given(monkeypatch):
    """Tests and scripts that build an engine themselves pass no flag; the
    block they ask for is the block they get, kernel or not."""
    engine, _, pinned = _drafted_vlm(monkeypatch, "active", draft_block_size=4)
    assert engine.draft_block == 4 and pinned == [4]


def test_vlm_engine_reports_no_draft_block_without_a_drafter(monkeypatch):
    from sous.engine.vlm import VLMEngine

    _stub(
        monkeypatch,
        "mlx_vlm",
        load=lambda model_id: (_positionless_model(), _RecordingTokenizer()),
    )
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    engine = VLMEngine("test/model", cache_budget=0, draft_block_size=4, draft_block_explicit=False)
    assert engine.draft_block is None and engine._draft_block_size == 4
    # The constructor's default keeps the stock projections.
    assert engine.projection_kernel_status["state"] == "off"
```

In `test_default_engine_factory_maps_every_model_value_onto_the_backend`, the config sets both new values away from their defaults, so the test proves they are read from the config:

Replace:

```python
        int8_prefill=True,
        attention_tile=False,
    )
    base.default_engine_factory(cfg)("org/m")
```

with:

```python
        int8_prefill=True,
        attention_tile=False,
        projection_kernel=False,
        speculative_block_explicit=True,
    )
    base.default_engine_factory(cfg)("org/m")
```

Replace:

```python
        "draft_block_size": 2,
        "cache_budget": int(1.5 * (1 << 30)),
```

with:

```python
        "draft_block_size": 2,
        "draft_block_explicit": True,
        "cache_budget": int(1.5 * (1 << 30)),
```

Replace:

```python
        "attention_tile": False,
        "fork_dir": None,
```

with:

```python
        "attention_tile": False,
        "projection_kernel": False,
        "fork_dir": None,
```

In `tests/test_engine_unloaded.py`, after `test_vlm_unload_turns_the_attention_tile_off`:

Replace:

```python
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    _unloaded_vlm().unload()
    assert tileattn._active_splits == 0
```

with:

```python
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    _unloaded_vlm().unload()
    assert tileattn._active_splits == 0


def test_vlm_unload_turns_the_projection_kernel_off(monkeypatch):
    from sous.engine import projkernel

    monkeypatch.setattr(projkernel, "_active", 1)
    _unloaded_vlm().unload()
    assert projkernel._active == 0
```

The test that pins the tile's clear before anything that can raise covers the kernel's clear too, renamed to say so:

Replace:

```python
@pytest.mark.parametrize("failing", ["prompt cache", "draft context"])
def test_vlm_unload_turns_the_attention_tile_off_before_anything_that_can_raise(
    monkeypatch, failing
):
    from sous.engine import tileattn

    def fails(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(tileattn, "_active_splits", 20)
```

with:

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

Replace:

```python
    with pytest.raises(RuntimeError, match="failed"):
        engine.unload()
    assert tileattn._active_splits == 0
```

with:

```python
    with pytest.raises(RuntimeError, match="failed"):
        engine.unload()
    assert tileattn._active_splits == 0
    assert projkernel._active == 0
```

`unload()` must need no attribute beyond the ones `_unloaded_vlm()` sets, which is why it clears the module flag and leaves the tags alone.

- [ ] **Step 6: Run the engine tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py tests/test_engine_unloaded.py -k "projection_kernel or draft_block or block_flag or explicit_block or maps_every_model_value" -v`

Expected: 21 failed, 132 deselected.
- `KeyError: 'projection_kernel'` (2) and `KeyError: 'draft_block_explicit'` (3) from the manager and direct-call factory tests, and an `AssertionError` on the kwargs dict in `test_default_engine_factory_maps_every_model_value_onto_the_backend`, whose right side has 2 more items;
- `TypeError: _default_factory() got an unexpected keyword argument 'projection_kernel'`;
- `TypeError: VLMEngine.__init__() got an unexpected keyword argument 'projection_kernel'` in the order test (its no-drafter case also reports `DID NOT WARN`, since the constructor raises first), the six resolution cases and the direct-build test, and `... 'draft_block_explicit'` in the no-drafter test;
- `AttributeError: 'LMEngine' object has no attribute 'projection_kernel_status'`;
- `assert 1 == 0` on `projkernel._active` after `unload()`, three times.

- [ ] **Step 7: Wire the engines**

In `src/sous/engine/vlm.py`:

Replace:

```python
from sous.engine import draftctx, forkio, tileattn
```

with:

```python
from sous.config import SPECULATIVE_BLOCK_KERNEL
from sous.engine import draftctx, forkio, projkernel, tileattn
```

Replace:

```python
        draft_block_size: int = 0,
        cache_budget: int | None = None,
        reserve_bytes: int = 0,
        int8_prefill: bool = False,
        attention_tile: bool = False,
        fork_dir: Path | None = None,
```

with:

```python
        draft_block_size: int = 0,
        draft_block_explicit: bool = True,
        cache_budget: int | None = None,
        reserve_bytes: int = 0,
        int8_prefill: bool = False,
        attention_tile: bool = False,
        projection_kernel: bool = False,
        fork_dir: Path | None = None,
```

The pin leaves the drafter branch:

Replace:

```python
            _pin_block_size(self._draft, draft_block_size)
        # Only speculation runs the verifier, so only a drafter that loaded
```

with:

```python
        # Only speculation runs the verifier, so only a drafter that loaded
```

and lands after the kernel, behind the resolution:

Replace:

```python
            verify_status=self.verify_attention_status,
        )
        # The drafter's context on the prompt-cache path: armed once the
```

with:

```python
            verify_status=self.verify_attention_status,
        )
        # After the tile, whose status it reads: the kernel serves only where
        # the tile does, so it inherits the tile's GPU gate and, with a
        # drafter, exact verify attention. Before the fork-store key, which
        # reads it, and before the budget, so the probe's transient arrays are
        # already released when the budget is measured.
        self.projection_kernel_status = projkernel.enable(
            self._model,
            enabled=projection_kernel,
            drafter_kind=self._draft_kind if self._draft is not None else None,
            tile_status=self.attention_tile_status,
        )
        # An unset block resolves only now that the kernel has settled: with
        # the kernel a 5-row verify round's projections cost about what a
        # 4-row round's do, so block 5 pays; without it 4 does. The pin
        # follows the resolution, and moving it this late changes nothing
        # above: neither the verifier's nor the tile's gates read the pin.
        if (
            not draft_block_explicit
            and self._draft is not None
            and self.projection_kernel_status["state"] == "active"
        ):
            self._draft_block_size = SPECULATIVE_BLOCK_KERNEL
        _pin_block_size(self._draft, self._draft_block_size)
        # The drafter's context on the prompt-cache path: armed once the
```

Replace:

```python
        engine survives with a warning rather than an error."""
        return self._draft_id
```

with:

```python
        engine survives with a warning rather than an error."""
        return self._draft_id

    @property
    def draft_block(self) -> int | None:
        """The verify block speculative decoding runs with, as resolved at
        load (0 is the drafter's own policy), or None without a drafter: the
        model-load line's and the status document's `draft_block`."""
        return self._draft_block_size if self._draft is not None else None
```

In `unload()`:

Replace:

```python
        # First, before anything below can raise: until the next load decides
        # afresh, no one-row call may reach the tile on the strength of this
        # model's gates.
        tileattn.clear()
```

with:

```python
        # First, before anything below can raise: until the next load decides
        # afresh, no call may reach the tile or the projection kernel on the
        # strength of this model's gates. The kernel's tags stay on this
        # model's modules, inert while its flag is clear.
        tileattn.clear()
        projkernel.clear()
```

In `src/sous/engine/lm.py`:

Replace:

```python
        self.attention_tile_status = {"state": "off", "reason": None}
        # Forks on disk, when the factory handed us a directory: built after
```

with:

```python
        self.attention_tile_status = {"state": "off", "reason": None}
        # The projection kernel serves only modules an mlx-vlm qwen3_5 load
        # tagged, so mlx-lm's never reach it; reported off, as the tile is.
        self.projection_kernel_status = {"state": "off", "reason": None}
        # Forks on disk, when the factory handed us a directory: built after
```

- [ ] **Step 8: Wire the factory and the managed engine**

In `src/sous/engine/base.py`, in `_default_factory`'s signature:

Replace:

```python
    draft_block_size: int = 0,
    cache_budget: int | None = None,
    reserve_tokens: int = 0,
    int8_prefill: bool = False,
    attention_tile: bool = False,
    fork_dir: Path | None = None,
```

with:

```python
    draft_block_size: int = 0,
    # False only from the config, for a block the user left unset: the VLM
    # engine resolves that one at load. A direct caller's block is taken as
    # given.
    draft_block_explicit: bool = True,
    cache_budget: int | None = None,
    reserve_tokens: int = 0,
    int8_prefill: bool = False,
    attention_tile: bool = False,
    projection_kernel: bool = False,
    fork_dir: Path | None = None,
```

In its VLM branch:

Replace:

```python
            draft_block_size=draft_block_size,
            cache_budget=cache_budget,
            reserve_bytes=reserve_bytes,
            int8_prefill=int8_prefill,
            attention_tile=attention_tile,
            fork_dir=fork_dir,
            fork_budget=fork_budget,
            weights_identity=weights,
        )
    # The drafter settings and the attention tile stop here: speculative
    # decoding and the tile's hooks are mlx-vlm paths, and the mlx-lm backend
    # has no parameter for either.
```

with:

```python
            draft_block_size=draft_block_size,
            draft_block_explicit=draft_block_explicit,
            cache_budget=cache_budget,
            reserve_bytes=reserve_bytes,
            int8_prefill=int8_prefill,
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

In `default_engine_factory`:

Replace:

```python
        draft_block_size=config.speculative_block_size,
        cache_budget=(
```

with:

```python
        draft_block_size=config.speculative_block_size,
        draft_block_explicit=config.speculative_block_explicit,
        cache_budget=(
```

Replace:

```python
        attention_tile=config.attention_tile,
        fork_dir=fork_dir,
```

with:

```python
        attention_tile=config.attention_tile,
        projection_kernel=config.projection_kernel,
        fork_dir=fork_dir,
```

In `ManagedEngine`, after `attention_tile_status`:

Replace:

```python
        return getattr(self._inner, "attention_tile_status", None)
```

with:

```python
        return getattr(self._inner, "attention_tile_status", None)

    @property
    def projection_kernel_status(self) -> dict | None:
        # Optional on purpose, like attention_tile_status: the VLM backend
        # reports whether the kernel serves this load, the LM backend reports
        # it off, and fakes have neither.
        return getattr(self._inner, "projection_kernel_status", None)

    @property
    def draft_block(self) -> int | None:
        """The verify block the drafter runs with, as the engine resolved it
        at load (0 is the drafter's own policy): None without a drafter, and
        from a backend or a fake that has no notion of one."""
        return getattr(self._inner, "draft_block", None)
```

In `src/sous/tune/suite/runner.py`, the list of attributes `CountingEngine` passes through:

Replace:

```python
        # drafter, positions, int8_prefill_status, verify_attention_status,
        # draft_context_status, attention_tile_status: whatever the backend
        # has, read through getattr(..., None) by ManagedEngine.
```

with:

```python
        # drafter, draft_block, positions, int8_prefill_status,
        # verify_attention_status, draft_context_status, attention_tile_status,
        # projection_kernel_status: whatever the backend has, read through
        # getattr(..., None) by ManagedEngine.
```

- [ ] **Step 9: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py -v`

Expected: 220 passed (67, 141 and 12), with the one warning the file already raises: `drafter context unavailable (drafter names no target layers)` from `test_vlm_engine_probes_verify_attention_only_with_a_drafter[True]`. The existing VLMEngine stub tests raise nothing new, because the constructor's `projection_kernel=False` returns `off` before any gate.

Then run the whole suite, under the GPU lock if another session may be using the GPU:

`uv run pytest -m "not model"`

Expected: every test passes. The skips are the `nax`-marked tests off the M5 and the projection kernel's arithmetic tests wherever `projkernel.kernel_available()` fails.

- [ ] **Step 10: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/config.py src/sous/engine/vlm.py src/sous/engine/lm.py src/sous/engine/base.py src/sous/tune/suite/runner.py tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py
git commit -m "feat(engine): default the projection kernel on and resolve an unset block at load" -m "With the kernel, a 5-row verify round's projections cost about what a
4-row round's do, so block 5 pays only where the kernel is active: 1.10x
sampled decode over block 4 without it, on real subagent turns. An unset
[model].speculative_block_size therefore resolves at load, to 5 where the
kernel is active and to 4 elsewhere. An explicit value always wins, a
clamped one included, and the drafter's block is pinned only after the
resolution.

[model].projection_kernel switches the kernel off. A non-bool value means
false, as the other numeric switches do. The engine constructors keep the
stock projections and the block they are given, so tests and scripts that
build an engine directly are unaffected. An unload clears the kernel's flag
beside the tile's, before anything there can raise.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

---

### Task 7: Load line, status document and fork store

**Files:**
- Modify: `src/sous/engine/base.py`:
  - the load line in `EngineManager.get()`, after the attention_tile tokens (:719-724 once Task 6 has landed, :696-701 before it);
  - `EngineManager.status()`, after the attention_tile block (:1039-1041 once Task 6 has landed, :1016-1018 before it).
- Modify: `src/sous/engine/forkstore.py`:
  - `_EPOCH_FILES` at :58;
  - `fork_key_fields`'s signature and docstring at :155-176;
  - its `fields` dict at :191-192;
  - `engine_epoch()`'s header comment at :240-242.
- Modify: `src/sous/engine/vlm.py`: the fork-store comment and `_fork_key_fields` (:155-158 and :230-232 once Task 6 has landed).
- Modify: `src/sous/engine/lm.py`: `_fork_key_fields` (:95-97 once Task 6 has landed).
- Test:
  - `tests/test_engine_base.py`, appended after `test_status_carries_the_attention_tile_view_when_the_engine_reports_one`, the file's last test;
  - `tests/test_forkstore.py` at :138-160 and :1408-1484;
  - `tests/test_engine_forkhooks.py` at :32-55.

**Interfaces:**
- Consumes:
  - Task 6: `ManagedEngine.projection_kernel_status` and `ManagedEngine.draft_block`; `VLMEngine.projection_kernel_status` and `LMEngine.projection_kernel_status`, both set before the fork store is built.
  - Task 2: `src/sous/engine/projkernel.py` exists, so `engine_epoch()` can read it.
  - Existing: `ManagedEngine.drafter` (`""` from the VLM backend without a drafter, None from the LM backend and fakes), `key_name`, `engine_epoch`, `os_release`, and the test helpers `_positional_factory`, `_cfg`, `FakeEngine` and `_vlm_fields`.
- Produces:
  - The model-load line, after the attention_tile tokens: ` projection_kernel={state}`, then ` projection_kernel_probe_s={probe_seconds}` only when active, then ` draft_block={n}` whenever `engine.draft_block` is not None, 0 included.
  - `EngineManager.status()`, after `attention_tile`: `out['projection_kernel'] = {'state': ..., 'reason': ...}` when the engine reports one, and `out['draft_block'] = engine.draft_block` whenever `engine.drafter is not None` (None while the VLM backend runs without a drafter).
  - `forkstore._EPOCH_FILES = ('vlm.py', 'lm.py', 'int8prefill.py', 'tileattn.py', 'projkernel.py')`. The kernel's `.metal` and `.h` files are already hashed by the `kernels/` scan.
  - `fork_key_fields(*, backend, backend_version, weights, gpu, positions, int8_status, tile_status, proj_status: Mapping[str, Any]) -> dict[str, str]`, `proj_status` required, recording `'proj'` as `'active'` when `proj_status.get('state') == 'active'`, else `'off'`.
  - vlm.py and lm.py `_fork_key_fields` pass `proj_status=self.projection_kernel_status`; the LM backend's is always `off`.

Why `proj` is in the key:
- The kernel serves the one-row forward that ends every prefill segment, and any prefill tail of up to 8 rows. A fork written with it holds different last bits in those tokens' KV and, through their hidden states, in every later layer's KV and GDN state.
- `unavailable` leaves the flag clear, so it runs `off`'s numerics and shares `off`'s key. The probe's time and the reason are reports, so they stay out.
- The kernel's routing does not depend on whether a drafter loaded, so the store's premise that the drafter is absent from the key still holds.

Consequence: landing this changes the key and the engine epoch, so every fork store starts cold once. The PR says so.

- [ ] **Step 1: Write the failing load-line and status tests**

Append at the end of `tests/test_engine_base.py`, anchored on the last lines of the tile's status test, the file's last test:

Replace:

```python
    assert manager.status()["attention_tile"] == {
        "state": "unavailable",
        "reason": "split target 16 not measured (measured: 20)",
    }
```

with:

```python
    assert manager.status()["attention_tile"] == {
        "state": "unavailable",
        "reason": "split target 16 not measured (measured: 20)",
    }


@pytest.mark.parametrize(
    ("status", "tail"),
    [
        (
            {"state": "active", "reason": None, "probe_seconds": 0.37},
            " positions=engine projection_kernel=active projection_kernel_probe_s=0.37",
        ),
        (
            {"state": "unavailable", "reason": "attention tile off", "probe_seconds": None},
            " positions=engine projection_kernel=unavailable",
        ),
        (
            {"state": "unavailable", "reason": "probe: row 3 differs", "probe_seconds": 0.37},
            " positions=engine projection_kernel=unavailable",
        ),
        ({"state": "off", "reason": None}, " positions=engine projection_kernel=off"),
        (None, " model=fake/model positions=engine"),
    ],
)
def test_get_logs_the_projection_kernel_state(caplog, tmp_path, status, tail):
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        if status is not None:
            engine.projection_kernel_status = status  # ty: ignore[unresolved-attribute]
        return engine

    mgr = EngineManager(_cfg(tmp_path), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and lines[0].endswith(tail), lines


@pytest.mark.parametrize(
    ("block", "tail"),
    [
        (5, " projection_kernel=active projection_kernel_probe_s=0.37 draft_block=5"),
        (0, " projection_kernel=active projection_kernel_probe_s=0.37 draft_block=0"),
        (None, " projection_kernel=active projection_kernel_probe_s=0.37"),
    ],
)
def test_get_logs_the_resolved_draft_block_after_the_kernel_tokens(caplog, tmp_path, block, tail):
    """The tile's tokens, then the kernel's, then the block whenever a drafter
    loaded, 0 (the drafter's own policy) included: the config alone cannot
    say which block runs once the engine resolves an unset one."""
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
        engine.draft_block = block  # ty: ignore[unresolved-attribute]
        return engine

    mgr = EngineManager(_cfg(tmp_path), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    expected = " attention_tile=active attention_tile_splits=20 attention_tile_probe_s=0.43" + tail
    assert len(lines) == 1 and lines[0].endswith(expected), lines


def test_status_carries_the_projection_kernel_and_the_draft_block(tmp_path):
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    status = manager.status()
    assert "projection_kernel" not in status, "fakes without the attribute stay silent"
    assert "draft_block" not in status, "as does a backend with no notion of a drafter"
    inner.projection_kernel_status = {  # ty: ignore[unresolved-attribute]
        "state": "unavailable",
        "reason": "attention tile off",
        "probe_seconds": None,
    }
    inner.drafter = ""  # ty: ignore[unresolved-attribute]
    inner.draft_block = None  # ty: ignore[unresolved-attribute]
    status = manager.status()
    assert status["projection_kernel"] == {"state": "unavailable", "reason": "attention tile off"}
    assert status["draft_block"] is None  # a backend that knows drafters, running without one
    inner.drafter = "z-lab/drafter"  # ty: ignore[unresolved-attribute]
    inner.draft_block = 5  # ty: ignore[unresolved-attribute]
    assert manager.status()["draft_block"] == 5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k "get_logs_the_projection_kernel or get_logs_the_resolved_draft_block or status_carries_the_projection_kernel" -v`

Expected: 8 failed, 1 passed, 141 deselected. The passing case is the fake without the attribute, whose line already ends at `positions=engine`. The load-line cases fail with an `AssertionError` on a line that still ends at `positions=engine` or at `attention_tile_probe_s=0.43`; the status test fails with `KeyError: 'projection_kernel'`.

- [ ] **Step 3: Write the load-line tokens and the status keys**

In `src/sous/engine/base.py`, in `EngineManager.get()`:

Replace:

```python
                line += (
                    f" attention_tile_splits={tile['splits']}"
                    f" attention_tile_probe_s={tile['probe_seconds']}"
                )
        _logger.info(line)
        return engine
```

with:

```python
                line += (
                    f" attention_tile_splits={tile['splits']}"
                    f" attention_tile_probe_s={tile['probe_seconds']}"
                )
        proj = engine.projection_kernel_status
        if proj is not None:
            line += f" projection_kernel={proj['state']}"
            if proj["state"] == "active":
                line += f" projection_kernel_probe_s={proj['probe_seconds']}"
        # Resolved at load, so the config alone cannot say which block runs:
        # an unset one is 5 where the kernel is active and 4 elsewhere.
        if engine.draft_block is not None:
            line += f" draft_block={engine.draft_block}"
        _logger.info(line)
        return engine
```

In `EngineManager.status()`:

Replace:

```python
                tile = self._engine.attention_tile_status
                if tile is not None:
                    out["attention_tile"] = {"state": tile["state"], "reason": tile["reason"]}
```

with:

```python
                tile = self._engine.attention_tile_status
                if tile is not None:
                    out["attention_tile"] = {"state": tile["state"], "reason": tile["reason"]}
                proj = self._engine.projection_kernel_status
                if proj is not None:
                    out["projection_kernel"] = {"state": proj["state"], "reason": proj["reason"]}
                # The block the drafter runs, as resolved at load: present
                # whenever the backend knows drafters, None while it runs
                # without one.
                if self._engine.drafter is not None:
                    out["draft_block"] = self._engine.draft_block
```

Whether the status carries `draft_block` turns on whether the backend knows drafters, not on the block itself: a VLM engine without a drafter says `null`, while the LM backend and fakes say nothing.

- [ ] **Step 4: Run the engine tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -v`

Expected: 150 passed, with the same one warning as in Task 6.

- [ ] **Step 5: Lint, type-check and commit the visibility**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): show the projection kernel and the resolved block on the load line" -m "An unset block now resolves at load from the kernel's state, so the
config alone no longer says which block runs. The model-load line names
projection_kernel=, with its probe time when active, and draft_block=
whenever a drafter loaded, 0 included. The status document's engine block
carries projection_kernel {state, reason} and draft_block, which is null
while the VLM backend runs without a drafter. Fakes and backends without
the attributes stay silent, as they do for the tile.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing fork-store tests**

In `tests/test_forkstore.py`, in `test_engine_epoch_changes_when_a_source_byte_changes`:

Replace:

```python
    for name in ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py"):
        (src / name).write_text(f"# {name}\n")
```

with:

```python
    for name in ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py", "projkernel.py"):
        (src / name).write_text(f"# {name}\n")
```

At the end of the same test, a change to `projkernel.py` must move the epoch, and a new test pins the file list:

Replace:

```python
    assert engine_epoch() != changed  # the tile's schedule decides which kernel runs
    forkstore.engine_epoch.cache_clear()


def test_the_attention_tile_module_is_an_epoch_file():
    assert "tileattn.py" in forkstore._EPOCH_FILES
```

with:

```python
    assert engine_epoch() != changed  # the tile's schedule decides which kernel runs
    changed = engine_epoch()
    (src / "projkernel.py").write_text("# projkernel.py changed\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() != changed  # its routing decides which calls the kernel serves
    forkstore.engine_epoch.cache_clear()


def test_the_attention_tile_module_is_an_epoch_file():
    assert "tileattn.py" in forkstore._EPOCH_FILES


def test_the_projection_kernel_module_is_an_epoch_file():
    assert "projkernel.py" in forkstore._EPOCH_FILES
```

In `test_fork_key_fields_carry_everything_that_changes_the_kv`, the first call:

Replace:

```python
        tile_status={"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43},
    )
    assert f["backend"] == "vlm" and f["backend_version"] == "0.7.1"
```

with:

```python
        tile_status={"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43},
        proj_status={"state": "active", "reason": None, "probe_seconds": 0.37},
    )
    assert f["backend"] == "vlm" and f["backend_version"] == "0.7.1"
```

its assertions:

Replace:

```python
    assert f["tile"] == "active:20" and f["os"] == forkstore.os_release()
    assert f["env"] == "MLX_ENABLE_TF32=1"
```

with:

```python
    assert f["tile"] == "active:20" and f["os"] == forkstore.os_release()
    assert f["proj"] == "active"
    assert f["env"] == "MLX_ENABLE_TF32=1"
```

the second call and its assertions:

Replace:

```python
        int8_status={"state": "unavailable", "reason": "no tensor units", "routed": 0},
        tile_status={"state": "off", "reason": None},
    )
    assert off["int8"] == "off"  # off and unavailable ran the same numerics
    assert off["tile"] == "off" and off["os"] == forkstore.os_release()
```

with:

```python
        int8_status={"state": "unavailable", "reason": "no tensor units", "routed": 0},
        tile_status={"state": "off", "reason": None},
        proj_status={"state": "off", "reason": None},
    )
    assert off["int8"] == "off"  # off and unavailable ran the same numerics
    assert off["tile"] == "off" and off["os"] == forkstore.os_release()
    assert off["proj"] == "off"
```

and the last call:

Replace:

```python
        int8_status={"state": "off", "reason": None, "routed": 0},
        tile_status={"state": "off", "reason": None},
    )
```

with:

```python
        int8_status={"state": "off", "reason": None, "routed": 0},
        tile_status={"state": "off", "reason": None},
        proj_status={"state": "off", "reason": None},
    )
```

The shared helper takes the kernel's status too:

Replace:

```python
_INT8_OFF = {"state": "off", "reason": None, "routed": 0}


def _vlm_fields(*, tile=_TILE_OFF, int8=_INT8_OFF) -> dict[str, str]:
    from sous.engine.forkstore import fork_key_fields

    return fork_key_fields(
        backend="vlm",
        backend_version="0.7.2",
        weights="sha",
        gpu="applegpu_g17s",
        positions="engine",
        int8_status=int8,
        tile_status=tile,
    )
```

with:

```python
_INT8_OFF = {"state": "off", "reason": None, "routed": 0}
_PROJ_OFF = {"state": "off", "reason": None, "probe_seconds": None}
_PROJ_ON = {"state": "active", "reason": None, "probe_seconds": 0.37}


def _vlm_fields(*, tile=_TILE_OFF, int8=_INT8_OFF, proj=_PROJ_OFF) -> dict[str, str]:
    from sous.engine.forkstore import fork_key_fields

    return fork_key_fields(
        backend="vlm",
        backend_version="0.7.2",
        weights="sha",
        gpu="applegpu_g17s",
        positions="engine",
        int8_status=int8,
        tile_status=tile,
        proj_status=proj,
    )
```

After `test_the_key_ignores_what_a_tile_status_only_reports`:

Replace:

```python
    probe = {**mac, "reason": "correctness: n=1536 is 0.101 from stock", "probe_seconds": 0.5}
    assert key_name(_vlm_fields(tile=mac)) == key_name(_vlm_fields(tile=probe))
```

with:

```python
    probe = {**mac, "reason": "correctness: n=1536 is 0.101 from stock", "probe_seconds": 0.5}
    assert key_name(_vlm_fields(tile=mac)) == key_name(_vlm_fields(tile=probe))


def test_the_key_moves_with_the_projection_kernel():
    """A store written with the kernel is never restored without it. An
    unavailable kernel ran off's numerics, so it shares off's key, and the
    probe's time and the reason are reports, not numerics."""
    unavailable = {"state": "unavailable", "reason": "attention tile off", "probe_seconds": None}
    assert _vlm_fields(proj=_PROJ_ON)["proj"] == "active"
    assert _vlm_fields(proj=unavailable)["proj"] == "off"
    assert key_name(_vlm_fields(proj=_PROJ_ON)) != key_name(_vlm_fields())
    assert key_name(_vlm_fields(proj=unavailable)) == key_name(_vlm_fields())
    later = {**_PROJ_ON, "reason": "noted", "probe_seconds": 1.2}
    assert key_name(_vlm_fields(proj=later)) == key_name(_vlm_fields(proj=_PROJ_ON))
```

In `test_the_os_build_is_in_every_key`:

Replace:

```python
    variants = [{}, {"int8": int8}, {"tile": _TILE_20}]
```

with:

```python
    variants = [{}, {"int8": int8}, {"tile": _TILE_20}, {"proj": _PROJ_ON}]
```

In `tests/test_engine_forkhooks.py`, the VLM key-field test reads the kernel's state, renamed to say so:

Replace:

```python
def test_vlm_key_fields_read_positions_and_the_realised_int8_and_tile_states(monkeypatch):
    engine = vlm.VLMEngine.__new__(vlm.VLMEngine)
    engine._positional = True
    engine.int8_prefill_status = {"state": "active", "reason": None, "routed": 12}
    engine.attention_tile_status = {
        "state": "active",
        "reason": None,
        "splits": 20,
        "probe_seconds": 0.43,
    }
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "vlm" and fields["positions"] == "engine"
    assert fields["int8"] == "active:12" and fields["weights"] == "sha"
    assert fields["tile"] == "active:20"
```

with:

```python
def test_vlm_key_fields_read_positions_and_the_realised_kernel_states(monkeypatch):
    engine = vlm.VLMEngine.__new__(vlm.VLMEngine)
    engine._positional = True
    engine.int8_prefill_status = {"state": "active", "reason": None, "routed": 12}
    engine.attention_tile_status = {
        "state": "active",
        "reason": None,
        "splits": 20,
        "probe_seconds": 0.43,
    }
    engine.projection_kernel_status = {"state": "active", "reason": None, "probe_seconds": 0.37}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "vlm" and fields["positions"] == "engine"
    assert fields["int8"] == "active:12" and fields["weights"] == "sha"
    assert fields["tile"] == "active:20" and fields["proj"] == "active"
```

and the LM one reads its `off`:

Replace:

```python
    engine.attention_tile_status = {"state": "off", "reason": None}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off" and fields["tile"] == "off"
```

with:

```python
    engine.attention_tile_status = {"state": "off", "reason": None}
    engine.projection_kernel_status = {"state": "off", "reason": None}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off" and fields["tile"] == "off" and fields["proj"] == "off"
```

- [ ] **Step 7: Run the fork-store tests to verify they fail**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py -v`

Expected: 9 failed, 83 passed.
- The epoch test: `AssertionError`, since a byte change to `projkernel.py` does not move the epoch.
- `assert 'projkernel.py' in ('vlm.py', 'lm.py', 'int8prefill.py', 'tileattn.py')`.
- Five key-field tests (the direct one and the four that go through `_vlm_fields`): `TypeError: fork_key_fields() got an unexpected keyword argument 'proj_status'`.
- The two engine key-field tests: `KeyError: 'proj'`.

- [ ] **Step 8: Key the store by the kernel**

In `src/sous/engine/forkstore.py`:

Replace:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py")
```

with:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py", "projkernel.py")
```

Replace:

```python
    tile_status: Mapping[str, Any],
) -> dict[str, str]:
    """Everything that changes the KV arrays behind identical token ids, and
    nothing else: the backend and its package, mlx, the GPU, the weights,
    the engine sources (epoch) and the file layout, which side supplies the
    rotary positions, whether int8 prefill actually ran, whether the
    attention tile served the load and at which split target, the macOS
    build, and the two mlx-core environment switches that change matmul
    precision and attention accumulation.

    The tile runs the one-row forward that ends every prefill segment, so it
    moves the last bits of that token's KV and, through its hidden state,
    every later layer's KV and GDN state. `unavailable` runs `off`'s
    numerics for both kernels.
```

with:

```python
    tile_status: Mapping[str, Any],
    proj_status: Mapping[str, Any],
) -> dict[str, str]:
    """Everything that changes the KV arrays behind identical token ids, and
    nothing else: the backend and its package, mlx, the GPU, the weights,
    the engine sources (epoch) and the file layout, which side supplies the
    rotary positions, whether int8 prefill actually ran, whether the
    attention tile served the load and at which split target, whether the
    projection kernel served it, the macOS build, and the two mlx-core
    environment switches that change matmul precision and attention
    accumulation.

    The tile and the projection kernel both run the one-row forward that
    ends every prefill segment, and the projection kernel any prefill tail
    of up to 8 rows too, so each moves the last bits of those tokens' KV
    and, through their hidden states, every later layer's KV and GDN state.
    `unavailable` runs `off`'s numerics for all three kernels.
```

Replace:

```python
    the last bits any of them produces. The template, tokenizer, sampling,
    drafter and window are deliberately absent: ids are compared exactly,
    and none of them touches a prefill."""
```

with:

```python
    the last bits any of them produces. The template, tokenizer, sampling,
    drafter and window are deliberately absent: ids are compared exactly,
    and none of them touches a prefill (the projection kernel routes a
    prefill the same with a drafter loaded or without one)."""
```

Replace:

```python
        "tile": f"active:{tile_status['splits']}" if tile_active else "off",
        "os": os_release(),
```

with:

```python
        "tile": f"active:{tile_status['splits']}" if tile_active else "off",
        "proj": "active" if proj_status.get("state") == "active" else "off",
        "os": os_release(),
```

In `engine_epoch()`, the header comment names what the kernel's header carries:

Replace:

```python
    # The .h files are compiled into every kernel (int8's nibble-decode and
    # K-order contract, the tile's lane layout), so they are part of the
    # numerics too.
```

with:

```python
    # The .h files are compiled into every kernel (int8's nibble-decode and
    # K-order contract, the tile's lane layout, the projection kernel's MMA
    # helper), so they are part of the numerics too.
```

In `src/sous/engine/vlm.py`, the fork-store comment in `__init__`:

Replace:

```python
        # Forks on disk, when the factory handed us a directory: built after
        # the drafter, int8 and the attention tile have settled, since the key
        # reads them. Tests and sous tune never pass one, so nothing they build
        # touches ~/.sous.
```

with:

```python
        # Forks on disk, when the factory handed us a directory: built after
        # the drafter, int8, the attention tile and the projection kernel have
        # settled, since the key reads them. Tests and sous tune never pass
        # one, so nothing they build touches ~/.sous.
```

and `_fork_key_fields`:

Replace:

```python
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
        )
```

with:

```python
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
            proj_status=self.projection_kernel_status,
        )
```

In `src/sous/engine/lm.py`, `_fork_key_fields`:

Replace:

```python
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
        )
```

with:

```python
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
            proj_status=self.projection_kernel_status,
        )
```

- [ ] **Step 9: Run the tests to verify they pass, then the wider checks**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py tests/test_engine_unloaded.py tests/test_engine_base.py tests/test_config.py -v`

Expected: 321 passed (86, 6, 12, 150 and 67), with the same one warning. That includes `test_a_working_store_construction_leaves_forks_on`, whose LM engine builds its key from the `off` status Task 6 set before the store.

Then:

```bash
uv run pytest tests/test_engine_forkio.py tests/test_promptcache.py -v
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected: every test passes, and ruff and ty report nothing.

- [ ] **Step 10: Commit the key**

```bash
git add src/sous/engine/forkstore.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_forkstore.py tests/test_engine_forkhooks.py
git commit -m "feat(engine): key on-disk forks by the projection kernel" -m "The kernel serves the one-row forward that ends every prefill segment and
any prefill tail of up to 8 rows, so a fork written with it holds
different bits than one written without it. The key records proj as
active or off, unavailable sharing off's key since it runs off's numerics,
and projkernel.py joins the engine epoch; the kernel's .metal and .h files
were already hashed by the kernels directory scan.

The kernel routes a prefill the same with a drafter loaded or without one,
so the drafter stays out of the key.

Landing this cold-starts every fork store once, because both the key and
the engine epoch change.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

---

### Task 8: `sous tune`: the projection kernel's keys and notice, and the block the daemon runs

**Files:**
- Modify: `src/sous/tune/candidates.py`: the import at `:9`, `Checkpoint` at `:112-124`, `describe`'s return at `:148-160`. Not in the File Structure: see the third design note.
- Modify: `src/sous/tune/arms.py`: the imports and `BLOCK_SIZES` at `:4-15`, `Arm` at `:39-64`, two new functions before `_arm` (`:106`), `_arm` at `:116-142`.
- Modify: `src/sous/tune/suite/runner.py`: `SuiteRun` at `:54-103`, `_error_run` and `run_one` (`:228`, `:286`), a new `_check_proj` after `_check_tile` (`:328-344`), the call at `:429`. The comment in `CountingEngine.__getattr__` is Task 6's and is not touched again.
- Modify: `src/sous/tune/decide.py`: `_changes` at `:57-58`, `ArmSummary.key` at `:151`.
- Modify: `src/sous/tune/__init__.py`: `main` at `:259`. Not in the File Structure: see the second design note.
- Test: `tests/test_tune_candidates.py`, `tests/test_tune_arms.py`, `tests/test_tune_runner.py`, `tests/test_tune_main.py`, `tests/test_tune_decide.py`.
- Modify: `tests/test_tune_report.py:289`, `_summary`'s literal key, only so ty accepts it: a six-tuple is not assignable to the seven-tuple `ArmSummary.key` declares.

**Interfaces:**
- Consumes:
  - Task 4: `projkernel.static_reason(config, model_config: dict, drafter_kind: str | None) -> str | None`, None meaning the kernel would activate. The tune calls it as `projkernel.static_reason(...)` at call time, so tests monkeypatch the module attribute and never depend on the machine they run on.
  - Task 6: `SousConfig.projection_kernel`, `SousConfig.speculative_block_explicit`, `config.SPECULATIVE_BLOCK_KERNEL`, `load_config` setting the explicit flag when the file sets the block, and `ManagedEngine.projection_kernel_status` (None when the engine has no such attribute, as for fakes; the LM backend reports `{"state": "off", "reason": None}`).
  - Existing: `quick_arms`, `_arm`, `winner_stage_arms`, `SuiteRun`, `_check_tile`, `_run_suite`, `quick_decision`, `full_decision`, `_changes`, `main`.
- Produces:
  - `candidates.Checkpoint.config: dict`, default `{}`, `compare=False`, `repr=False`, set by `describe` to the config it read.
  - `arms.drafter_kind(config: dict) -> str`: mlx-vlm's own resolution for a drafter loaded with no kind given (`_expected_drafter_kind(model_type, config) or DEFAULT_DRAFTER_KIND` from `mlx_vlm.speculative.drafters`, reached through `importlib`), `"unresolved"` when the module or either name is missing.
  - `arms.resolve_block(user: SousConfig, checkpoints: dict[str, Checkpoint]) -> SousConfig`: `user` itself when the block is explicit, no drafter is configured, either checkpoint is undescribed or `static_reason` returns a reason; otherwise `dataclasses.replace(user, speculative_block_size=SPECULATIVE_BLOCK_KERNEL)`, still not explicit.
  - `Arm.projection_kernel: bool = True`, after `attention_tile`; `Arm.suite_key -> tuple[str, str, int, bool, bool, bool, bool]` = `(*self.key, int8_prefill, greedy, attention_tile, projection_kernel)`.
  - `_arm` sets `speculative_block_explicit=True` on every drafter arm's config (a drafterless arm keeps the user's flag) and `projection_kernel=user.projection_kernel`.
  - `SuiteRun.projection_kernel: bool`, required, after `attention_tile`; `SuiteRun.key` the same seven-tuple; `SuiteRun.from_dict` fills a missing `projection_kernel` (and, as before, `attention_tile`) with False.
  - `runner._check_proj(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None`: never raises; prints `f"  {arm.label}: projection kernel unavailable here ({reason}); the engine runs stock, as the daemon would"` only when `arm.projection_kernel` is set and the engine reports `unavailable`; `reason` is the status's, else `no reason given`.
  - `decide._changes` writes `speculative_block_size` for a drafter arm whose block differs from the user's (resolved) block, or whose model or drafter differs from the user's while the user's block is unset.
  - `decide.ArmSummary.key: tuple[str, str, int, bool, bool, bool, bool]`.
  - `main` replaces `user` with `arms_mod.resolve_block(user, checkpoints)` right after describing the checkpoints, and prints `speculative_block_size is unset and the projection kernel can run here: the configured arm is block 5` when that moved the block.

**Design notes.**
- **Older rows read `projection_kernel = False`.** `attention_tile`'s fallback is False because every row written before the tile existed ran stock attention. The same holds here, on every machine: no row from before the setting existed went through the kernel. So an older row is a stock-projection row, and a `--resume` under an arm that inherits `true` runs the task again instead of pooling stock and kernel rows. The key records the setting, not what the engine managed, so on a Mac where the kernel cannot run a resume across the upgrade re-runs tasks it did not strictly need to; that is the same trade the tile's fallback made.
- **One resolution, in `main`.** `quick_arms`' current-arm mark and `decide`'s comparison both need the effective block, and both receive `user` from `main`, so resolving it once there, before any arm exists, makes `quick_arms`' `block == user.speculative_block_size` mark, its `sorted({*BLOCK_SIZES, user.speculative_block_size})`, and `_changes`' `arm.block_size != user.speculative_block_size` all compare against the block the daemon runs, with no signature change. `quick_arms` and `decide` keep taking a plain `SousConfig`, so their existing tests do not move with the machine they run on.
- **`Checkpoint.config`.** `static_reason` reads `model_type` and the attention shape (24/4/256) from the raw `fetch_model_config` dict, and the drafter's kind needs the drafter's raw config. `describe` already fetched both, through the `describe` collaborator `main`'s tests inject; carrying the dict on `Checkpoint` avoids a second fetch that those tests would send to the Hub. It is out of equality and the hash, so nothing that compares checkpoints changes.
- **The drafter's kind** comes from mlx-vlm's own resolver applied to that config (`fx.dflash2_27b()`, `model_type` `qwen3`, resolves to `dflash`), so the static check refuses an MTP or EAGLE drafter exactly as the engine's load would.
- **Another model or drafter writes its block.** The resolved block is the daemon's for the user's own pair only. Without this, a full run whose winner is another model at block 5 would write no block, and the daemon could resolve that model's unset block to 4.
- **Known gap**, as the spec says: a refusal only a load shows (a failed probe, a drifted pin) leaves the current-arm mark one block off. The arms themselves are unaffected: each pins its own block.

- [ ] **Step 1: Write the failing candidates and arms tests**

In `tests/test_tune_candidates.py`, replace the first line:

```python
from datetime import date
```

with:

```python
import dataclasses
from datetime import date
```

Append to the end of `tests/test_tune_candidates.py`:

```python


def test_describe_keeps_the_config_it_read_for_checks_that_read_more_of_it():
    cp = describe("t", config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 1)
    assert cp.config == fx.qwen_27b()
    # Equality and the hash stay about the summarised fields.
    assert cp == dataclasses.replace(cp, config={})
    assert hash(cp) == hash(dataclasses.replace(cp, config={}))
```

In `tests/test_tune_arms.py`, replace the imports (`:1-4`):

```python
import dataclasses

from sous.config import SousConfig
from sous.tune.arms import BLOCK_SIZES, Arm, quick_arms, winner_stage_arms
```

with:

```python
import dataclasses
import sys
import types

from sous.config import SPECULATIVE_BLOCK_KERNEL, SousConfig
from sous.tune.arms import (
    BLOCK_SIZES,
    Arm,
    drafter_kind,
    quick_arms,
    resolve_block,
    winner_stage_arms,
)
```

In `tests/test_tune_arms.py`, replace the head of `_winner` (`:252-259`):

```python
def _winner(tmp_path, temperature=0.7, int8=False, tile=True):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
        attention_tile=tile,
    )
```

with:

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

In `tests/test_tune_arms.py`, replace the end of `_winner` (`:271-273`):

```python
        greedy=temperature == 0,
        attention_tile=tile,
    )
```

with:

```python
        greedy=temperature == 0,
        attention_tile=tile,
        projection_kernel=proj,
    )
```

Change the existing `suite_key` assertions, each to one more element:

- In `test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions` (`:279`), `    assert arm.suite_key == (*arm.key, False, False, True)` becomes:

  ```python
      assert arm.suite_key == (*arm.key, False, False, True, True)
  ```
- In `test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_a_sampled_winner_a_greedy_arm` (`:289-290`), these two lines:

  ```python
      assert int8.suite_key == (*int8.key, True, False, True)
      assert greedy.suite_key == (*greedy.key, False, True, True)
  ```
  become:

  ```python
      assert int8.suite_key == (*int8.key, True, False, True, True)
      assert greedy.suite_key == (*greedy.key, False, True, True, True)
  ```
- In `test_quick_arms_mirror_the_users_int8_and_greedy_settings` (`:347`), `        assert arm.suite_key == (*arm.key, True, True, True)` becomes:

  ```python
          assert arm.suite_key == (*arm.key, True, True, True, True)
  ```
- In `test_quick_arms_mirror_the_users_attention_tile_and_key_their_suite_rows_by_it` (`:365`), `            assert arm.suite_key == (*arm.key, False, False, tile)` becomes:

  ```python
              assert arm.suite_key == (*arm.key, False, False, tile, True)
  ```
- In `test_the_winner_stage_carries_the_attention_tile_and_adds_no_arm_for_it` (`:376`), the tile is no longer the key's last element: `    assert [a.suite_key[-1] for a in arms] == [False, False]` becomes:

  ```python
      assert [a.suite_key[5] for a in arms] == [False, False]
  ```

Append to the end of `tests/test_tune_arms.py`:

```python


def _static_check(monkeypatch, reason=None):
    """Stand in for the projection kernel's static check, recording what the
    tune asked it: the real one reads this machine's GPU."""
    from sous.engine import projkernel

    asked = []

    def static_reason(config, model_config, drafter_kind):
        asked.append((config, model_config, drafter_kind))
        return reason

    monkeypatch.setattr(projkernel, "static_reason", static_reason)
    return asked


def test_an_unset_block_resolves_to_the_kernels_where_the_static_check_passes(
    tmp_path, monkeypatch
):
    asked = _static_check(monkeypatch)
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    assert user.speculative_block_explicit is False
    resolved = resolve_block(user, _checkpoints())
    assert resolved.speculative_block_size == SPECULATIVE_BLOCK_KERNEL == 5
    # Still unset: a proposal must never write the resolved block back.
    assert resolved.speculative_block_explicit is False
    assert asked == [(user, fx.qwen_27b(), "dflash")]
    arms, _ = quick_arms(
        resolved, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    current = [a for a in arms if a.current]
    assert [(a.drafter_id, a.block_size) for a in current] == [("z-lab/Qwen3.8-27B-DFlash2", 5)]
    assert current[0].label == "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @5"


def test_an_unset_block_stays_at_the_default_where_the_static_check_refuses(tmp_path, monkeypatch):
    _static_check(monkeypatch, reason="attention tile off")
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    assert resolve_block(user, _checkpoints()) is user


def test_only_an_unset_block_with_a_described_model_and_drafter_is_resolved(tmp_path, monkeypatch):
    asked = _static_check(monkeypatch)
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    explicit = dataclasses.replace(user, speculative_block_explicit=True)
    undrafted = dataclasses.replace(user, speculative_draft_id="")
    assert resolve_block(explicit, _checkpoints()) is explicit
    assert resolve_block(undrafted, _checkpoints()) is undrafted
    for missing in ("mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2"):
        cps = _checkpoints()
        del cps[missing]
        assert resolve_block(user, cps) is user
    assert asked == []


def test_every_drafter_arm_pins_the_block_its_label_names(tmp_path):
    """An unset block resolves at load, and replace() would hand every arm the
    user's unset flag: where the projection kernel runs, each drafter arm
    would then load at block 5 whatever its label says."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms, _ = quick_arms(
        user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
    )
    drafted = [a for a in arms if a.drafter_id]
    assert len(drafted) == 2 * len(BLOCK_SIZES)
    for arm in drafted:
        assert arm.config.speculative_block_explicit is True
        assert arm.config.speculative_block_size == arm.block_size
    assert all(not a.config.speculative_block_explicit for a in arms if not a.drafter_id)


def test_the_drafter_kind_is_mlx_vlms_own_resolution():
    assert drafter_kind(fx.dflash2_27b()) == "dflash"
    assert drafter_kind({"model_type": "qwen3_5_mtp"}) == "mtp"
    assert drafter_kind({"model_type": "qwen3", "num_nextn_predict_layers": 1}) == "mtp"
    assert drafter_kind({"speculators_model_type": "eagle3"}) == "eagle3"


def test_a_drafter_kind_mlx_vlm_cannot_resolve_reads_as_unresolved(monkeypatch):
    name = "mlx_vlm.speculative.drafters"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    assert drafter_kind(fx.dflash2_27b()) == "unresolved"
    monkeypatch.setitem(sys.modules, name, None)
    assert drafter_kind(fx.dflash2_27b()) == "unresolved"


def test_quick_arms_mirror_the_users_projection_kernel_and_key_their_suite_rows_by_it(tmp_path):
    keys = {}
    for proj in (True, False):
        user = SousConfig(
            data_dir=tmp_path / "d", config_path=tmp_path / "c.toml", projection_kernel=proj
        )
        arms, refusals = quick_arms(
            user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
        )
        assert refusals == []
        for arm in arms:
            assert arm.projection_kernel is proj and arm.config.projection_kernel is proj
            assert arm.suite_key == (*arm.key, False, False, True, proj)
        keys[proj] = {arm.suite_key for arm in arms}
    # A run resumed after the setting was flipped matches none of the old rows.
    assert keys[True].isdisjoint(keys[False])


def test_the_winner_stage_carries_the_projection_kernel_and_adds_no_arm_for_it(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path, proj=False), nax=True, checkpoint=cp)
    assert [a.label for a in arms] == ["27B + DFlash2 @3 + int8 prefill", "27B + DFlash2 @3 greedy"]
    assert all(a.projection_kernel is False and a.config.projection_kernel is False for a in arms)
    assert [a.suite_key[-1] for a in arms] == [False, False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run them separately, since a collection error stops pytest before any test runs:

Run: `uv run pytest tests/test_tune_candidates.py -v`
Expected: 1 failed, 15 passed. `test_describe_keeps_the_config_it_read_for_checks_that_read_more_of_it` fails with `AttributeError: 'Checkpoint' object has no attribute 'config'`.

Run: `uv run pytest tests/test_tune_arms.py -v`
Expected: ERROR at collection, `ImportError: cannot import name 'drafter_kind' from 'sous.tune.arms'`.

- [ ] **Step 3: Carry the config on `Checkpoint`, and add the resolution, the kernel key and the explicit block to the arms**

In `src/sous/tune/candidates.py`, replace the import (`:9`):

```python
from dataclasses import dataclass
```

with:

```python
from dataclasses import dataclass, field
```

In `src/sous/tune/candidates.py`, replace the end of `Checkpoint`:

```python
    quant: QuantSummary
    vocab_size: int | None = None
```

with:

```python
    quant: QuantSummary
    vocab_size: int | None = None
    # The config.json itself, for checks that read more of it than the
    # fields above (the projection kernel's static check). Out of equality
    # and the hash: a checkpoint is what it was summarised to.
    config: dict = field(default_factory=dict, compare=False, repr=False)
```

In `src/sous/tune/candidates.py`, replace the end of `describe`:

```python
        vocab_size=text.get("vocab_size"),
    )
```

with:

```python
        vocab_size=text.get("vocab_size"),
        config=config,
    )
```

In `src/sous/tune/arms.py`, replace the imports and `BLOCK_SIZES`:

```python
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from sous.config import MIN_CONTEXT_TOKENS, SPECULATIVE_BLOCK_MAX, SousConfig
from sous.tune.candidates import Candidate, Checkpoint, Fit, drafter_compatible, fit

# The verify depths worth measuring per machine, up to the config's clamp: 4
# measured best on the M5 Pro (#118), 2 and 3 are for machines where a verify
# row costs more.
BLOCK_SIZES = tuple(range(2, SPECULATIVE_BLOCK_MAX + 1))
```

with:

```python
from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass

from sous.config import (
    MIN_CONTEXT_TOKENS,
    SPECULATIVE_BLOCK_KERNEL,
    SPECULATIVE_BLOCK_MAX,
    SousConfig,
)
from sous.tune.candidates import Candidate, Checkpoint, Fit, drafter_compatible, fit

# The verify depths worth measuring per machine, up to the config's clamp: 5
# pays on the M5 Pro where the projection kernel runs and 4 where it does not;
# 2 and 3 are for machines where a verify row costs more.
BLOCK_SIZES = tuple(range(2, SPECULATIVE_BLOCK_MAX + 1))
```

In `src/sous/tune/arms.py`, replace (in `Arm`):

```python
    attention_tile: bool = True
    # True only for the winner stage's own int8 arm
```

with:

```python
    attention_tile: bool = True
    # Inherited and keyed the same way, for the same reason: kernel and stock
    # projections differ in their last bits.
    projection_kernel: bool = True
    # True only for the winner stage's own int8 arm
```

In `src/sous/tune/arms.py`, replace:

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the two settings only the suite may change, and the attention
        tile every arm inherits."""
        return (*self.key, self.int8_prefill, self.greedy, self.attention_tile)
```

with:

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the two settings only the suite may change, and the attention
        tile and projection kernel every arm inherits."""
        return (
            *self.key,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
        )
```

Add `drafter_kind` and `resolve_block` before `_arm`. The `projkernel` import is function-local, as `winner_stage_arms`' import of `int8prefill` is, and `static_reason` is read off the module at call time so a test's monkeypatch reaches it.

In `src/sous/tune/arms.py`, replace the head of `_arm`:

```python
def _arm(
    user: SousConfig,
```

with:

```python
def drafter_kind(config: dict) -> str:
    """The round-loop kind mlx-vlm resolves at load for a drafter with this
    config.json and no kind given, by its own resolver, so the static check
    below needs no download. A resolver this cannot reach reads as
    "unresolved", which the check refuses like any kind but dflash."""
    try:
        drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    except ImportError:
        return "unresolved"
    expected = getattr(drafters, "_expected_drafter_kind", None)
    default = getattr(drafters, "DEFAULT_DRAFTER_KIND", None)
    if expected is None or not isinstance(default, str):
        return "unresolved"
    model_type = config.get("model_type") or config.get("speculators_model_type")
    return str(expected(model_type, config) or default)


def resolve_block(user: SousConfig, checkpoints: dict[str, Checkpoint]) -> SousConfig:
    """The user's configuration with the verify block the daemon would run.
    An unset speculative_block_size resolves at load to
    SPECULATIVE_BLOCK_KERNEL where the projection kernel goes active and stays
    at the default elsewhere; the tune cannot load a model to find out, so it
    puts the kernel's static check to the checkpoints it already described.
    The current-arm mark, the user's own block among the measured ones and
    every proposed change then compare against what the daemon runs. A
    refusal the static check cannot see (a failed probe, a drifted source
    pin) leaves the mark one block off; the arms themselves are unaffected,
    since each pins the block its label names."""
    if user.speculative_block_explicit or not user.speculative_draft_id:
        return user
    target = checkpoints.get(user.model_id)
    drafter = checkpoints.get(user.speculative_draft_id)
    if target is None or drafter is None:
        return user
    from sous.engine import projkernel

    if projkernel.static_reason(user, target.config, drafter_kind(drafter.config)) is not None:
        return user
    return dataclasses.replace(user, speculative_block_size=SPECULATIVE_BLOCK_KERNEL)


def _arm(
    user: SousConfig,
```

In `src/sous/tune/arms.py`, replace the config `_arm` builds:

```python
    config = dataclasses.replace(
        user,
        model_id=cand.id,
        speculative_draft_id=drafter_id,
        speculative_block_size=block if drafter_id else user.speculative_block_size,
        max_context_tokens=min(user.max_context_tokens, f.window),
    )
```

with:

```python
    config = dataclasses.replace(
        user,
        model_id=cand.id,
        speculative_draft_id=drafter_id,
        speculative_block_size=block if drafter_id else user.speculative_block_size,
        # Explicit, or the copy of the user's unset flag would let the engine
        # resolve the block at load: where the projection kernel runs, every
        # drafter arm would measure block 5 whatever its label says.
        speculative_block_explicit=True if drafter_id else user.speculative_block_explicit,
        max_context_tokens=min(user.max_context_tokens, f.window),
    )
```

In `src/sous/tune/arms.py`, replace the end of `_arm`:

```python
        greedy=user.temperature == 0,
        attention_tile=user.attention_tile,
    )
```

with:

```python
        greedy=user.temperature == 0,
        attention_tile=user.attention_tile,
        projection_kernel=user.projection_kernel,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_candidates.py tests/test_tune_arms.py -v`
Expected: 45 passed (16 candidates, 29 arms). `test_the_drafter_kind_is_mlx_vlms_own_resolution` imports the real mlx-vlm resolver, which CI's macos-15 job has.

- [ ] **Step 5: Write the failing runner, main and decide tests, and update the hand-built rows**

In `tests/test_tune_runner.py`, make these replacements first, then append the new tests. The replacements must come first: the appended row test repeats the text the last of them matches.

1. In `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` (`:121`), `    assert run.key == (*_arm(tmp_path).key, False, False, True)` becomes:

   ```python
       assert run.key == (*_arm(tmp_path).key, False, False, True, True)
   ```
2. In `test_suite_runs_round_trip_through_dicts` (`:522-523`), replace:

   ```python
           greedy=True,
           attention_tile=False,
   ```
   with:

   ```python
           greedy=True,
           attention_tile=False,
           projection_kernel=False,
   ```
   and its last line, `    assert run.key == ("org/m", "", 0, False, True, False)`, with:

   ```python
       assert run.key == ("org/m", "", 0, False, True, False, False)
   ```
3. In `test_a_row_written_before_the_attention_tile_existed_reads_as_stock` (`:601-602`), replace:

   ```python
           greedy=False,
           attention_tile=True,
   ```
   with:

   ```python
           greedy=False,
           attention_tile=True,
           projection_kernel=True,
   ```
   and `    assert old.key == (*arm.key, False, False, False)` (`:623`) with:

   ```python
       assert old.key == (*arm.key, False, False, False, True)
   ```

Append to the end of `tests/test_tune_runner.py`:

```python


def test_a_row_written_before_the_projection_kernel_existed_reads_as_stock(tmp_path):
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
    del row["projection_kernel"]
    old = SuiteRun.from_dict(row)
    # No run from before the setting existed went through the kernel, on any
    # machine: the row is a stock-projection row.
    assert old.projection_kernel is False and old.attention_tile is True
    # The arm inherits the kernel's default, so a resume runs the task again
    # rather than counting the stock row toward the kernel arm.
    assert arm.projection_kernel is True
    assert old.key == (*arm.key, False, False, True, False)
    assert old.key != arm.suite_key
    del row["attention_tile"]
    older = SuiteRun.from_dict(row)
    assert (older.attention_tile, older.projection_kernel) == (False, False)


class _ProjUnavailable(FakeEngine):
    drafter = ""
    projection_kernel_status = {
        "state": "unavailable",
        "reason": "attention tile off",
        "probe_seconds": None,
    }


def test_an_arm_that_inherits_the_kernel_runs_stock_where_the_engine_cannot_serve_it(tmp_path):
    """Like the tile, the projection kernel is inherited from the user's
    config and no arm measures it: an engine that reports it unavailable
    runs the arm stock after a notice, as the daemon would."""
    lines = []
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "s",
        out=lines.append,
        factory=lambda mid: _ProjUnavailable([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert len(outcome.runs) == 1 and outcome.error is None
    assert outcome.runs[0].projection_kernel is True
    assert (
        "  m: projection kernel unavailable here (attention tile off); "
        "the engine runs stock, as the daemon would"
    ) in lines


def test_no_kernel_notice_when_it_is_active_off_unreported_or_not_inherited(tmp_path):
    class Active(FakeEngine):
        drafter = ""
        projection_kernel_status = {"state": "active", "reason": None, "probe_seconds": 0.3}

    class Off(FakeEngine):
        drafter = ""
        projection_kernel_status = {"state": "off", "reason": None}

    class Silent(FakeEngine):
        drafter = ""

    arm = _arm(tmp_path)
    stock = dataclasses.replace(
        arm,
        projection_kernel=False,
        config=dataclasses.replace(arm.config, projection_kernel=False),
    )
    for name, case, engine in (
        ("active", arm, Active),
        ("off", arm, Off),
        ("silent", arm, Silent),
        ("stock", stock, _ProjUnavailable),
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
        assert outcome.runs[0].projection_kernel is case.projection_kernel, name
        assert not any("projection kernel" in line for line in lines), (name, lines)
```

In `tests/test_tune_main.py`, replace `_cfg` (`:192-196`). The file it writes sets the block, so the config it hands `main` says so too: the block-3 convention of these tests then never meets the resolution:

```python
def _cfg(tmp_path, **over):
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\nspeculative_block_size = 3\n")
    over.setdefault("speculative_block_size", 3)
    return SousConfig(data_dir=tmp_path, config_path=p, **over)
```

with:

```python
def _cfg(tmp_path, **over):
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\nspeculative_block_size = 3\n")
    # The file sets the block, which load_config records as explicit: no test
    # built on it moves with how the daemon would resolve an unset one.
    over.setdefault("speculative_block_size", 3)
    over.setdefault("speculative_block_explicit", True)
    return SousConfig(data_dir=tmp_path, config_path=p, **over)
```

Add the field to both hand-built rows. In `_suite` (`:134-135`, 20-space indent), replace:

```python
                    attention_tile=arm.attention_tile,
                    window=arm.window,
```

with:

```python
                    attention_tile=arm.attention_tile,
                    projection_kernel=arm.projection_kernel,
                    window=arm.window,
```

In `test_a_stopped_suite_still_reports_the_runs_it_recorded` (`:558-559`, 12-space indent), replace:

```python
            attention_tile=arm.attention_tile,
            window=arm.window,
```

with:

```python
            attention_tile=arm.attention_tile,
            projection_kernel=arm.projection_kernel,
            window=arm.window,
```

Append to the end of `tests/test_tune_main.py`:

```python


def _static_check(monkeypatch, reason=None):
    """Stand in for the projection kernel's static check: the real one reads
    this machine's GPU, and these tests must not move with it."""
    from sous.engine import projkernel

    asked = []

    def static_reason(config, model_config, drafter_kind):
        asked.append((config.model_id, model_config, drafter_kind))
        return reason

    monkeypatch.setattr(projkernel, "static_reason", static_reason)
    return asked


def _unset(tmp_path):
    """A config file that leaves the block to the daemon."""
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\n")
    return SousConfig(data_dir=tmp_path, config_path=p)


B4 = "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @4"
B5 = "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @5"


def test_an_unset_block_is_measured_as_the_block_the_daemon_would_run(
    tmp_path, capsys, monkeypatch
):
    asked = _static_check(monkeypatch)
    deps, _ = _deps(tmp_path, scores={B5: 20.0}, cached=(M, D, N))
    assert main(_args(), config=_unset(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert asked == [(M, fx.qwen_27b(), "dflash")]
    assert "speculative_block_size is unset and the projection kernel can run here" in out
    choice = out.split("## Choice", 1)[1]
    assert f"  {B5}" in choice.splitlines()
    assert "the current arm is already the fastest measured" in choice
    assert "no config change" in choice and "Apply these changes" not in out
    assert (tmp_path / "config.toml").read_text() == "# mine\n[model]\n"


def test_a_block_faster_than_the_resolved_one_is_written_explicitly(tmp_path, capsys, monkeypatch):
    _static_check(monkeypatch)
    deps, _ = _deps(tmp_path, scores={B4: 20.0}, cached=(M, D, N), answers=("y",))
    assert main(_args(), config=_unset(tmp_path), **deps) == 0
    assert "+speculative_block_size = 4" in capsys.readouterr().out
    written = load_config(tmp_path / "config.toml")
    assert written.speculative_block_size == 4 and written.speculative_block_explicit is True


def test_an_unset_block_stays_at_the_default_where_the_kernel_cannot_run(
    tmp_path, capsys, monkeypatch
):
    _static_check(monkeypatch, reason="attention tile off")
    deps, _ = _deps(tmp_path, scores={B4: 20.0}, cached=(M, D, N))
    assert main(_args(), config=_unset(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "projection kernel can run here" not in out
    choice = out.split("## Choice", 1)[1]
    assert f"  {B4}" in choice.splitlines() and "no config change" in choice


def test_a_block_the_file_sets_is_never_resolved(tmp_path, capsys, monkeypatch):
    asked = _static_check(monkeypatch)
    deps, _ = _deps(tmp_path, scores={CUR: 20.0}, cached=(M, D, N))
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert asked == []
    assert "no config change" in capsys.readouterr().out.split("## Choice", 1)[1]
```

In `tests/test_tune_decide.py`, `_run` (`:236-238`), replace:

```python
        attention_tile=arm.attention_tile,
        window=arm.window,
        state=state,
```

with:

```python
        attention_tile=arm.attention_tile,
        projection_kernel=arm.projection_kernel,
        window=arm.window,
        state=state,
```

In `tests/test_tune_decide.py`, replace `_user` and its comment (`:257-261`):

```python
# Every user here keeps block 3, the block the fixtures' current arm and rows
# are written for, so these tests do not move with the shipped default.
def _user(tmp_path, **over):
    over.setdefault("speculative_block_size", 3)
    return SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", **over)
```

with:

```python
# Every user here sets block 3, the block the fixtures' current arm and rows
# are written for, so these tests move neither with the shipped default nor
# with how an unset block resolves.
def _user(tmp_path, **over):
    over.setdefault("speculative_block_size", 3)
    over.setdefault("speculative_block_explicit", True)
    return SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", **over)
```

Append to the end of `tests/test_tune_decide.py`:

```python


def test_a_resolved_unset_block_proposes_only_a_block_the_daemon_would_not_run(tmp_path):
    """The tune hands decide the user's block as the daemon resolves it
    (arms.resolve_block): an unset block where the projection kernel runs is
    already 5, so a block-5 winner changes nothing and any other is written."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_block_size=5)
    assert user.speculative_block_explicit is False
    arms = [_arm(user, "b5", block=5, current=True), _arm(user, "b4", block=4)]
    rows = [_row("b5", block=5, d16k=20.0), _row("b4", block=4, d16k=19.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "b5" and choice.changes == {}
    rows = [_row("b5", block=5, d16k=20.0), _row("b4", block=4, d16k=21.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.changes == {"model": {"speculative_block_size": 4}}


def test_a_drafter_arm_of_another_pair_writes_its_block_when_the_users_is_unset(tmp_path):
    """The resolved block is the daemon's for the user's own model and drafter
    only: an unset block could resolve differently for another pair, so its
    block is written even when the numbers match."""
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", speculative_block_size=5)
    cur = _arm(user, "27 @5", block=5, current=True, fit_window=131072)
    nine = _arm(user, "9 + d @5", drafter="z/9d", block=5, model=M9, fit_window=131072)
    runs = _runs(cur, [1.0], seconds=100.0) + _runs(nine, [1.0], seconds=50.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None and choice.label == "9 + d @5"
    assert choice.changes == {
        "model": {"id": M9, "speculative_draft_id": "z/9d", "speculative_block_size": 5}
    }
    explicit = dataclasses.replace(user, speculative_block_explicit=True)
    choice = full_decision(explicit, [cur, nine], runs, [], runs_per_task=1)
    assert choice is not None
    assert choice.changes == {"model": {"id": M9, "speculative_draft_id": "z/9d"}}
```

In `tests/test_tune_report.py`, `_summary` (`:289`), `        key=(model, "d", 3, False, False, True),` becomes:

```python
        key=(model, "d", 3, False, False, True, True),
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`

Expected: 33 failed, 77 passed (runner 6 of 27, main 11 of 34, decide 16 of 32, report 0 of 17). The failures are:
- 28 tests fail with `TypeError: SuiteRun.__init__() got an unexpected keyword argument 'projection_kernel'`: every test that builds a `SuiteRun` with the new keyword — 3 in the runner file, the 9 full-run tests in `tests/test_tune_main.py`, and all 16 decide failures, `_run`-based each, `test_a_drafter_arm_of_another_pair_writes_its_block_when_the_users_is_unset` included (its own assertion is reached only once the runner takes the field).
- The two runner notice tests fail with `AttributeError: 'SuiteRun' object has no attribute 'projection_kernel'`, and `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` fails its seven-tuple assertion.
- `test_an_unset_block_is_measured_as_the_block_the_daemon_would_run` fails on `assert asked == [...]` (nothing asks the static check yet), and `test_a_block_faster_than_the_resolved_one_is_written_explicitly` on `"+speculative_block_size = 4" in ...` (the current arm is still block 4).

Three new tests already pass, and should: `test_an_unset_block_stays_at_the_default_where_the_kernel_cannot_run`, `test_a_block_the_file_sets_is_never_resolved` and `test_a_resolved_unset_block_proposes_only_a_block_the_daemon_would_not_run` pin behaviour this task must keep.

- [ ] **Step 7: Key the row, add the notice, compare against the resolved block**

In `src/sous/tune/suite/runner.py`, replace (in `SuiteRun`):

```python
    attention_tile: bool
    window: int
```

with:

```python
    attention_tile: bool
    projection_kernel: bool
    window: int
```

In `src/sous/tune/suite/runner.py`, replace:

```python
    @property
    def key(self) -> tuple[str, str, int, bool, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (
            self.model_id,
            self.drafter_id,
            self.block_size,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
        )
```

with:

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

In `src/sous/tune/suite/runner.py`, replace (in `from_dict`):

```python
        # A row written before the tile existed ran stock attention; without
        # the fallback, resuming such a run would raise KeyError.
        d = {"attention_tile": False, **d}
```

with:

```python
        # A row written before a setting existed ran without what it switches
        # on — stock attention, stock projections — on every machine, so it
        # reads False, and a resume under an arm that has the setting runs
        # the task again rather than pool the two. Without the fallback,
        # resuming such a run would raise KeyError.
        d = {"attention_tile": False, "projection_kernel": False, **d}
```

The comment in `CountingEngine.__getattr__` already names `draft_block` and `projection_kernel_status`: Task 6 updated it, so it is left alone here.

In both `_error_run` and `run_one`, replace every occurrence of this text (it appears twice, so use the Edit tool's `replace_all`):

```python
        attention_tile=arm.attention_tile,
        window=arm.window,
```

with:

```python
        attention_tile=arm.attention_tile,
        projection_kernel=arm.projection_kernel,
        window=arm.window,
```

In `src/sous/tune/suite/runner.py`, replace the end of `_check_tile`, adding `_check_proj` after it:

```python
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: attention tile unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )
```

with:

```python
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: attention tile unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )


def _check_proj(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """The projection kernel is inherited and unmeasured like the tile, so
    the same holds: where the engine reports it unavailable, the daemon would
    serve this checkpoint with stock projections, and the arm runs the same
    way after a notice. Its block is unaffected — every drafter arm pins the
    one its label names. `off`, or no status, is the setting switched off or
    an engine without the kernel: no notice."""
    if not arm.projection_kernel:
        return
    status = engine.projection_kernel_status or {}
    if status.get("state") != "unavailable":
        return
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: projection kernel unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )
```

In `src/sous/tune/suite/runner.py`, replace the call site in `_run_suite`:

```python
            _check_tile(arm, engine, out)
```

with:

```python
            _check_tile(arm, engine, out)
            _check_proj(arm, engine, out)
```

In `src/sous/tune/decide.py`, replace the block clause of `_changes`:

```python
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
```

with:

```python
    # `user` arrives with its block resolved (arms.resolve_block): an unset
    # block already reads as the one the daemon runs for the user's own model
    # and drafter, so a winner at that block writes nothing. For another pair
    # an unset block could resolve differently at load, so a drafter arm that
    # changes the pair writes its block whatever the number.
    other_pair = (arm.model_id, arm.drafter_id) != (user.model_id, user.speculative_draft_id)
    if arm.drafter_id and (
        arm.block_size != user.speculative_block_size
        or (other_pair and not user.speculative_block_explicit)
    ):
        model["speculative_block_size"] = arm.block_size
```

In `src/sous/tune/decide.py`, replace (in `ArmSummary`):

```python
    key: tuple[str, str, int, bool, bool, bool]
```

with:

```python
    key: tuple[str, str, int, bool, bool, bool, bool]
```

In `src/sous/tune/__init__.py`, replace (in `main`):

```python
    checkpoints = _describe_all(ids, describe, out)
    arms, refusals = arms_mod.quick_arms(
```

with:

```python
    checkpoints = _describe_all(ids, describe, out)
    # Before any arm exists: the current-arm mark and every proposed change
    # compare against the block the daemon would run, and an unset block
    # resolves at load by whether the projection kernel can run here.
    resolved = arms_mod.resolve_block(user, checkpoints)
    if resolved.speculative_block_size != user.speculative_block_size:
        out(
            "speculative_block_size is unset and the projection kernel can run here: "
            f"the configured arm is block {resolved.speculative_block_size}"
        )
    user = resolved
    arms, refusals = arms_mod.quick_arms(
```

- [ ] **Step 8: Run the tune tests to verify they pass**

Run: `uv run pytest tests/test_tune_candidates.py tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`
Expected: 155 passed (candidates 16, arms 29, runner 27, main 34, decide 32, report 17).

- [ ] **Step 9: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add src/sous/tune/candidates.py src/sous/tune/arms.py src/sous/tune/suite/runner.py src/sous/tune/decide.py src/sous/tune/__init__.py tests/test_tune_candidates.py tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py
git commit -m "feat(tune): measure the block the daemon runs and key suite rows by the projection kernel" -m "An unset speculative_block_size now resolves at load to 5 where the
projection kernel is active. dataclasses.replace copied the user's unset
flag onto every arm, so each drafter arm on such a Mac would have measured
block 5 whatever its label said; _arm now pins the block it names. The
current-arm mark and every proposal compare against the block the daemon
would run, resolved once without a load by the kernel's static check over
the checkpoints the tune already described, with the drafter's kind read
through mlx-vlm's own resolver. A winner with another model or drafter
writes its block, since an unset one could resolve differently for it.

The kernel changes output, so suite rows are keyed by it as by the
attention tile; rows from before it existed ran stock projections and read
as false. An engine that cannot run it prints a notice and runs stock, as
the daemon would.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and the full non-model suite passes.

---

### Task 9: Notices and documentation

**Files:**
- Modify: `THIRD_PARTY_NOTICES.md`: the oMLX section's last sentence (`:13-14`), and a new Splash section at the end.
- Modify: `src/sous/engine/kernels/__init__.py`: the module docstring (the whole file).
- Modify: `CLAUDE.md`:
  - the epoch-file list at `:163-165`;
  - a new bullet after the tileattn bullet, which ends at `:413`, before the draftctx bullet at `:414`.
- Modify: `docs/configuration.md`: the TOML block at `:25-28`, a new section after the attention-tile section (which ends at `:184`), and the speculative-decoding paragraph at `:197-211`.
- Modify: `docs/observability.md`: the load-line paragraph at `:200-203`.
- Modify: `docs/tuning.md:57-62`.
- Modify: `docs/how-it-works.md:252-253`. Not in the File Structure: the fork-store paragraph lists what a stored fork must match, and Task 7 adds the projection kernel's state to that key; leaving it out would be a stale sentence.

**Interfaces:**
- Consumes: every name pinned in Tasks 1–8:
  - `projkernel`'s `MAX_ROWS`, `_TAG`, `_HOOK_MARK`, `_active`, `calls` and its four keys, `mma`, `enable`, `clear`, `kernel_available`, `static_reason`, `VALIDATED_PROJ_SOURCES`;
  - the kernel's frozen constants and the `// STAGED` and `// GROUP_SUMS` markers;
  - `SousConfig.projection_kernel`, `speculative_block_explicit`, `SPECULATIVE_BLOCK_KERNEL`;
  - the load-line tokens `projection_kernel=`, `projection_kernel_probe_s=` and `draft_block=`, the status keys `projection_kernel` and `draft_block`;
  - the fork-key field `proj` and `projkernel.py` in `_EPOCH_FILES`;
  - `arms.resolve_block`, `_arm`'s explicit block, `runner._check_proj`;
  - `tests/proj_fixtures.py`'s `guard_proj_state` and `hooked`, `tests/test_projkernel.py`, `tests/test_projkernel_model.py`.
- Produces: docs only.

**What the provenance comparison found** (run against the spike's `kern.py` and `kern_st.py` while this plan was written; Step 1 repeats it against the in-tree files):
- **Upstream Splash** (incoai/splash at f43509a): the simdgroup lane map (`common/sgmatrix.h`, `sgmatrix::lane_map`; its `qid`/`fm`/`fn` expressions match the port line for line), the MMA on plain-register operands (`sgmatrix::mma_acc`, the matrices built only inside each multiply-accumulate through `thread_elements()`), and, in `decode/linear_q4_sgmatrix.metal`, the W·Xᵀ orientation with `col * 32 + c * 8` weight addressing (`c = fn / 2`), the `(word >> (4 * s)) & 0x000F000Fu` nibble pairs, the one-group prefetch and the `fma(dot, scale, acc); fma(sum, bias, acc)` epilogue. Splash's own `THIRD_PARTY_NOTICES` covers only ggml material in its GGUF kernels, none of which these files touch.
- **paperniuk's Apple7/8 port** (branch `apple7-m1-kernels`): the exact half weights times fp32 activations, `as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h)` fed to a mixed half×float MMA, in d81795d's prefill tile and d2f902e's `decode/linear_q4_mma.metal`; the port's `h`/`s`/`f` loop around them follows d2f902e's.
- **a450058**, the `SimdgroupF32` commit, contributes nothing the port keeps: its operands are fp32 nibbles times fp32 (the spike's mode `f`, dropped), and its coherent-partials fix belongs to a grid split-K the port does not use. The notice therefore cites d81795d and d2f902e, not a450058.
- **sous's own**: reading mlx's [N, K/8] weights and [T, K] activations in place (Splash repacks into 256-column tiles and reads a prepared Xᵀ table plus precomputed sums), the k-bijection that lets each lane read four consecutive activations, the in-kernel shuffle-tree row sums and the group-sums kernel that reproduces them, the fixed-order in-threadgroup K split (Splash splits K across threadgroups with coherent partials and an atomic counter), the staging, the zeroed rows past T, the variant choice and the one-to-four-linear codegen.

- [ ] **Step 1: Compare the port with upstream, line by line**

Fetch the upstream sources at pinned commits into a scratch directory:

```bash
P=$(mktemp -d)
fetch() { gh api "repos/$1/contents/$2?ref=$3" -H "Accept: application/vnd.github.raw" > "$P/$4"; }
fetch incoai/splash runtime/metal/kernels/common/sgmatrix.h f43509a2f03d83b4442a0fab748e8b0fb49aa817 inc_sgmatrix.h
fetch incoai/splash runtime/metal/kernels/common/q4_sgmatrix.h f43509a2f03d83b4442a0fab748e8b0fb49aa817 inc_q4_sgmatrix.h
fetch incoai/splash runtime/metal/kernels/decode/linear_q4_sgmatrix.metal f43509a2f03d83b4442a0fab748e8b0fb49aa817 inc_decode_linear_q4_sgmatrix.metal
fetch paperniuk/splash runtime/metal/kernels/common/q4_sgmatrix.h a450058d862c079204be19719b89793f8b681abf pk_a450058_q4_sgmatrix.h
fetch paperniuk/splash runtime/metal/kernels/decode/linear_q4_sgmatrix.metal a450058d862c079204be19719b89793f8b681abf pk_a450058_decode_linear_q4_sgmatrix.metal
fetch paperniuk/splash runtime/metal/kernels/prefill/linear_q4_mma.metal d81795d008c755988bff6bd7d871fa001754de27 pk_d81795d_prefill_linear_q4_mma.metal
fetch paperniuk/splash runtime/metal/kernels/decode/linear_q4_mma.metal d2f902e8d4aec55b2afce4eb0666015d1f505a98 pk_d2f902e_decode_linear_q4_mma.metal
wc -l "$P"/*
gh api repos/incoai/splash --jq .license.spdx_id
gh api repos/paperniuk/splash --jq .license.spdx_id
grep -il copyright "$P"/* || echo "no copyright header"
```

Expected: `wc` lists, alphabetically, `inc_decode_linear_q4_sgmatrix.metal` 186, `inc_q4_sgmatrix.h` 46, `inc_sgmatrix.h` 44, `pk_a450058_decode_linear_q4_sgmatrix.metal` 189, `pk_a450058_q4_sgmatrix.h` 71, `pk_d2f902e_decode_linear_q4_mma.metal` 222 and `pk_d81795d_prefill_linear_q4_mma.metal` 219 lines (977 total); then `Apache-2.0` twice and `no copyright header`.

Write the comparison script beside them. It normalises whitespace, drops trivial lines, and for every remaining line of the port finds an exact or near (difflib ratio ≥ 0.8) upstream line, then labels the hit with what the notice attributes; a hit the notice does not cover prints `UNATTRIBUTED` and the script exits 1.

```bash
cat > "$P/compare.py" <<'EOF'
"""Every non-trivial line of the port against every line of the upstream
sources (whitespace-normalised): an exact match or a near one (difflib ratio
>= 0.8) is classified by what THIRD_PARTY_NOTICES.md attributes; anything
it does not attribute prints UNATTRIBUTED."""

import difflib
import re
import sys
from pathlib import Path

TRIVIAL = re.compile(r"^(\}|\{|\};|//.*|#pragma unroll|using namespace metal;|#include <metal_\w+>|#endif|#else|)$")
ATTRIBUTED = [
    ("lane map", re.compile(r"qid = lane >> 2|fm = \(qid & 4\)|fn = \(\(qid & 2\) << 1\)|c = fn (>> 1|/ 2)")),
    ("register MMA", re.compile(r"simdgroup_matrix<|simdgroup_float8x8|simdgroup_half8x8|thread_elements\(\)|simdgroup_multiply_accumulate|inline void mma8\(")),
    ("nibble pairs", re.compile(r">> \(4 \* s\)\) & 0x000F000Fu")),
    ("half weights", re.compile(r"0x64006400u\) - half2\(1024\.0h\)")),
    ("epilogue", re.compile(r"= fma\(.*(scale|bias)")),
    ("W.X^T columns", re.compile(r"base \+ n?f \* 8 \+ fm")),
    ("generic", re.compile(r"^for \(uint |^float2 (acc|dot)\[|^uint2 w|barrier\(|thread_index_in_simdgroup|simdgroup_index_in_threadgroup|threadgroup_position_in_grid|sgi? % \w+, \w+ = sgi? / |^\} else \{|^#(if|define)")),
]


def norm(line: str) -> str:
    return " ".join(line.split())


def kept(path: str) -> list[tuple[int, str]]:
    out = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = norm(raw)
        if line and not TRIVIAL.match(line):
            out.append((n, line))
    return out


def main(port: list[str], upstream: list[str]) -> int:
    up = [(f, n, line) for f in upstream for n, line in kept(f)]
    exact = {}
    for f, n, line in up:
        exact.setdefault(line, f"{f}:{n}")
    unattributed = 0
    for pf in port:
        for n, line in kept(pf):
            if line in exact:
                kind, source = "EXACT", exact[line]
            else:
                f, m, best = max(up, key=lambda u: difflib.SequenceMatcher(None, line, u[2]).ratio())
                if difflib.SequenceMatcher(None, line, best).ratio() < 0.8:
                    continue
                kind, source = "NEAR ", f"{f}:{m}"
            label = next((name for name, rx in ATTRIBUTED if rx.search(line)), None)
            if label is None:
                unattributed += 1
                label = "UNATTRIBUTED"
            print(f"{kind} {label:13s} {pf}:{n}: {line}  <= {source}")
    print(f"{unattributed} unattributed")
    return 1 if unattributed else 0


if __name__ == "__main__":
    split = sys.argv.index("--")
    sys.exit(main(sys.argv[1:split], sys.argv[split + 1 :]))
EOF
uv run python "$P/compare.py" src/sous/engine/kernels/projection_mma.h src/sous/engine/kernels/projection_mma.metal -- "$P"/*.h "$P"/*.metal
```

Expected: the last line is `0 unattributed` and the exit status 0. The hits fall in these labels only:
- `lane map`: the three lines `const uint qid = lane >> 2;`, `const uint fm = (qid & 4) | ((lane >> 1) & 3);` and `const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);` as EXACT, once per body that computes them, plus `c = fn >> 1` as NEAR;
- `register MMA`: the `mma8` helper's lines, `simdgroup_multiply_accumulate(D, A, B, C);` EXACT;
- `nibble pairs`, `half weights`, `epilogue` and `W.X^T columns`: NEAR, one per occurrence in the bodies;
- `generic`: loop headers, accumulator declarations, barriers and thread-position builtins.

Run on Task 2's kernel text as this plan writes it, the same command printed 72 hits (14 EXACT, 58 NEAR) and `0 unattributed`; on the spike's own kernels, 88 hits and `0 unattributed`. Any `UNATTRIBUTED` line means the port carries something from upstream the notice below does not name: add it to the notice's derived list and a label for it to `ATTRIBUTED`, and re-run before committing. Nothing from this step is committed.

- [ ] **Step 2: Add the Splash notice**

In `THIRD_PARTY_NOTICES.md`, replace the oMLX section's last sentence:

```markdown
`attention_tile.metal` in the same directory is sous's own work and is not derived
from oMLX.
```

with:

```markdown
`attention_tile.metal` in the same directory is sous's own work and is not derived
from oMLX; neither are `projection_mma.h` and `projection_mma.metal` (next section).
```

Append to the end of `THIRD_PARTY_NOTICES.md` (it already ends with a newline, so the section opens with one blank line):

```markdown

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
`acc = fma(dot, scale, acc); acc = fma(sum, bias, acc)`. The rest is sous's
own: reading mlx's own [N, K/8] weights and [T, K] activations in place, where
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
```

`pyproject.toml`'s `license-files` already lists `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`; nothing there changes.

- [ ] **Step 3: Name the kernel in the kernels package docstring**

Check the two facts the docstring states about Task 2's files:

```bash
grep -c '_kernel_text("common.h")' src/sous/engine/projkernel.py
grep -nxE '// (STAGED|GROUP_SUMS)' src/sous/engine/kernels/projection_mma.metal
```

Expected: a count of at least 1 (projkernel prepends `common.h` like int8prefill and tileattn), and exactly the two marker lines. If the count is 0, write "``common.h`` is prepended to the int8 and tile bodies" in place of "``common.h`` is prepended to every body" below.

In `src/sous/engine/kernels/__init__.py`, replace the whole file:

```python
"""Metal sources for sous's runtime-compiled kernels: int8 prefill and the
attention tile.

Read by ``sous.engine.int8prefill`` and ``sous.engine.tileattn`` through
``importlib.resources`` and handed to ``mx.fast.metal_kernel``, which compiles them
at model load. The ``.metal`` files are kernel *bodies* (mlx generates the
signature), so they do not compile standalone. ``common.h`` is prepended to every
body; ``nax.h``, which includes the Metal-4 tensor-op header, only to the tensor-op
kernels: the int8 GEMM and the tile's split kernel. ``attention_tile.metal`` holds
two bodies, the split kernel and the reduce kernel, divided at its ``// REDUCE``
line.

The int8 kernels are derived from oMLX (jundot/omlx#3548, Apache License 2.0); see
THIRD_PARTY_NOTICES.md. ``attention_tile.metal`` is sous's own.
"""
```

with:

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
None of the three uses tensor ops, so they compile on any Metal GPU and CI runs
their tests.

The int8 kernels are derived from oMLX (jundot/omlx#3548) and the projection
kernel in part from Splash (incoai/splash and its Apple7/8 port), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md. ``attention_tile.metal`` is
sous's own.
"""
```

- [ ] **Step 4: Add `projkernel.py` to CLAUDE.md's epoch list**

In `CLAUDE.md`, replace:

```markdown
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  kernels/*.metal and *.h) beside the versions, the GPU and the weights
  snapshot — any
```

with:

```markdown
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  projkernel.py, kernels/*.metal and *.h) beside the versions, the GPU and
  the weights snapshot — any
```

- [ ] **Step 5: Add the projection kernel's gotcha**

The bullet goes after tileattn's and before draftctx's, in the same register: what the parity rests on, the hooks and their scope, the flag, the guards and the probe, int8, the argmax residual, the block resolution, the fork key and the re-test rule.

In `CLAUDE.md`, replace:

```markdown
  in for `tileattn.tile`; the `nax` tests skip on
  `tileattn.availability()`, never int8's.
- `engine/draftctx.py` hands the DFlash drafter the prompt's hidden states on
```

with:

```markdown
  in for `tileattn.tile`; the `nax` tests skip on
  `tileattn.availability()`, never int8's.
- `engine/projkernel.py` serves the target model's small-row quantized
  projections from one simdgroup-matrix MMA kernel
  (`kernels/projection_mma.metal` and `.h`: plain, staged and group-sums
  bodies split at `// STAGED` and `// GROUP_SUMS`; no MPP and no `nax.h`,
  so it compiles on any Metal GPU and CI runs its arithmetic tests): every
  exact-verifier projection and the verify head, and every target call of
  1–8 rows outside the verifier — one-row decode, a prefill's last chunk of
  ≤ 8 rows and the one-row forward that ends every prefill segment, and the
  sampled DFlash drafter's call of the target's shared `lm_head`. Greedy
  parity with the drafter on or off holds only because the kernel's rows
  are bitwise independent of the row count T and of which linears share a
  dispatch: weights dequantize exactly (`as_type<half2>(nibble |
  0x64006400) - 1024`) against bf16 activations widened to fp32, so every
  product is exact; each 8×8×8 `simdgroup_multiply_accumulate` sums in a
  fixed order in fp32; each quant group's epilogue is
  `acc = fma(dot, scale, acc); acc = fma(sum_x, bias, acc)`; the four K
  partials reduce as `sk0 + sk1 + sk2 + sk3`; rows ≥ T are zeroed with
  their x pointers clamped; each tile serves exactly one linear; and the
  variant (plain, staged, staged over presummed x) depends only on K and
  the summed N of the call's linears, never on T, the three being bitwise
  equal. That is why a fused verify group (q+k+v, gate+up, qkv+z+b+a)
  equals the same projections decoded one at a time. F = 4, NSG = 1,
  KS = 4, GB = 4 (staged), CH = 1, SPLITS = 1, PF = 1 and SBV = 1 are
  frozen: another KS, or CH or SPLITS above 1, changes the summation order
  and the bits. Two hooks, each a `functools.wraps` wrapper carrying
  `_HOOK_MARK`, installed once per process on the first activation and
  never removed (int8's three wrappers carry `wraps` too, so every source
  pin follows `__wrapped__` to mlx's and mlx-vlm's own code whatever is
  wrapped on top): hook 1 wraps `Qwen3_5BatchInvariantForward._linear` and
  `_linears` on the class and routes when the flag is set, every linear in
  the call is tagged and x is bf16 with ndim 3, cutting a call of more than
  8 rows (block 0, the drafter's own policy, verifies up to 16) along T
  into runs of ≤ 8, exact because the kernel is T-invariant; hook 2 wraps
  `nn.QuantizedLinear.__call__` and routes when the flag is set, the module
  is tagged, x is bf16 and its rows (the product of the leading
  dimensions) are 1–8. Anything else falls through to what was wrapped,
  and both reach the kernel through `projkernel.mma`, looked up per call.
  The Gemma4, NemotronH and Qwen4Exp verifiers subclass
  `Qwen3_5BatchInvariantForward` and inherit hook 1, so only the tag keeps
  them stock. Tags (`_TAG`, set with `object.__setattr__` so they stay out
  of mlx's parameter tree) go on every `QuantizedLinear` of the loaded
  target's language model, `lm_head` included (497 on the default 27B);
  the drafter's own linears are never tagged, and the greedy drafter's
  head argmax (`quantized_argmax`) stays stock, since it shapes proposals
  only. int8 prefill wraps the same `__call__` for its own tags at ≥ 128
  rows on `shape[-2]`: the row ranges are disjoint and each wrapper falls
  through to what it wrapped, so install order does not matter — keep the
  two gates disjoint if either moves. The flag `_active` is a plain int, 1
  or 0, never a model reference: the first statement of every `enable()`
  clears it, only an `active` result sets it, the probe raises it only
  while it runs, and `VLMEngine.unload()` clears it through
  `projkernel.clear()` beside `tileattn.clear()`. A failed probe untags the
  model it was given; tags left on an unloaded model are inert while the
  flag is 0, and holding a reference to untag it would keep its weights
  alive. `enable()` checks, in order: the config (`off`); the attention
  tile (`active`, else the tile's reason or `attention tile off` — the
  kernel rides on the tile, inheriting its M5 Pro gate and, with a drafter,
  exact verify attention); the model (`qwen3_5`, every language-model
  `QuantizedLinear` affine 4-bit, group 64, bf16 scales and biases, no
  `bias`); the drafter (none or `dflash`, and the target must not define
  `speculative_verify_dflash_hidden`, which would send greedy verify's
  tokens through `quantized_argmax`, past both hooks); the source pins
  (`VALIDATED_PROJ_SOURCES`: the verifier's `_linear`, `_linears`,
  `_feed_forward`, `_gated_delta`, `_model` and `__call__`, `Qwen3_5MLP`
  and `Qwen3_5GatedDeltaNet`, `generate_step`,
  `nn.QuantizedLinear.__call__`, `_dflash_verify` and
  `_dflash_verify_greedy`) and `verifyattn.VALIDATED_MLX`; a warm-up that
  compiles every pipeline the model's dispatch keys need at T = 1..8; then
  the tags, the hooks and the probe, on the `sous-model-load` thread. The
  probe drives layer 3's verifier `_feed_forward` at T = 5 and the same
  layer's plain modules at T = 1 through both hooks (`calls` must show
  both) and requires the rows `array_equal`; then, on real linears (layer
  3's q+k+v, o and gate+up+down, layer 0's qkv+z+b+a and out_proj,
  `lm_head`), every row at T = 1..8 equal to its one-row single, every
  fused group equal to its singles, relative RMS within 1e-2 of stock
  `quantized_matmul` (parity cannot catch a kernel that is wrong the same
  way on both paths) and a NaN in one row leaving the others alone. It
  restores `calls`, leaves the flag 0 and calls `mx.clear_cache()` in
  `finally`. A refusal at the tile, model or drafter step only says the
  kernel does not apply to this load: one INFO line, `projection kernel
  unavailable: <reason>`. Pin drift, a compile or probe failure and any
  exception (logged with its traceback) say it should have run: one
  `warnings.warn`, `sous: projection kernel unavailable (…)`. Both are
  `unavailable`; `off` is never a refusal. Parity still rests on one thing
  the probe cannot cover: drafter-off greedy takes its argmax over bf16
  `logits − logsumexp(logits)` (`generate/ar.py`) and greedy verify over
  raw logits (`sampler(logits)` in `dflash.py`), so the two can differ only
  where the subtraction rounds the top two logits into a tie, in a flat row
  whose top logit sits below half the logsumexp. Stock and the tile rest on
  the same; parity runs on real turns are its only witness, and a
  divergence is read at the first differing token's top-2 logit gap before
  the kernel is blamed. Because the kernel makes a deeper block pay, an
  unset `[model].speculative_block_size` (`speculative_block_explicit`
  false) resolves in `VLMEngine.__init__` to `SPECULATIVE_BLOCK_KERNEL` = 5
  when a drafter loaded and the kernel is `active`, before
  `_pin_block_size`; an explicit value always wins and 0 keeps its meaning.
  `sous tune` cannot load a model to ask, so `arms.resolve_block` puts
  `projkernel.static_reason` to the checkpoints it described, and `_arm`
  marks every drafter arm's block explicit, or every arm would measure 5.
  The engine constructors default it off
  (`VLMEngine(projection_kernel=False, draft_block_explicit=True)`); only
  `default_engine_factory` hands them the config. Nothing is logged per
  turn; `projkernel.calls` (`verify_kernel`, `verify_split`,
  `plain_kernel`, `declined`) counts every path for the tests and for
  harnesses, and the load line carries `projection_kernel=` and, with a
  drafter, `draft_block=`. The fork key records `proj` (`active`, else
  `off`) and `projkernel.py` is in `_EPOCH_FILES`: the one-row forward that
  ends every prefill segment, and any tail of ≤ 8 rows, runs on the
  kernel, so stored KV and GDN state depend on it, while routing never
  depends on whether a drafter loaded, so the drafter stays out of the
  key. Every kernel edit, and every change to the pins or `VALIDATED_MLX`,
  re-runs the exactness tests (`tests/test_projkernel.py`, and `uv run
  pytest -m model tests/test_projkernel_model.py` on the M5 Pro) and the
  timing check — the 12 dispatch keys at T = 1..8 against the kernel before
  the edit, interleaved in one process with an A/A pair for the noise
  floor, the per-forward sum within ±2%: the bits are the parity contract
  and the speed is the only reason the kernel exists. Tests install the
  hooks through `pfx.guard_proj_state` and `pfx.hooked`, swap the plain-mlx
  stand-in in for `projkernel.mma`, and skip the kernel's own tests on
  `projkernel.kernel_available()`, which compiles and runs every variant
  once and remembers success only: the lane map was probed on Apple GPUs,
  so a GPU where it differs skips rather than fails.
- `engine/draftctx.py` hands the DFlash drafter the prompt's hidden states on
```

- [ ] **Step 6: Document `[model].projection_kernel` and the resolved block in the configuration reference**

The sample config shows the block commented out: an uncommented `speculative_block_size = 4` copied from it would be explicit and keep the block at 4 where the kernel runs.

In `docs/configuration.md`, replace:

```toml
speculative_draft_id = "z-lab/Qwen3.8-27B-DFlash2"
speculative_block_size = 4
int8_prefill = false
attention_tile = true
```

with:

```toml
speculative_draft_id = "z-lab/Qwen3.8-27B-DFlash2"
# speculative_block_size = 5  # unset: 5 where the projection kernel is active, else 4
int8_prefill = false
attention_tile = true
projection_kernel = true
```

In `docs/configuration.md`, replace the end of the attention-tile section, adding the projection-kernel section after it:

```markdown
update starts the store cold, whatever the setting). The daemon reads it at
startup, so a change takes effect on its next start.

`[server].generation_timeout_minutes` bounds a turn at both ends: how long it
```

with:

```markdown
update starts the store cold, whatever the setting). The daemon reads it at
startup, so a change takes effect on its next start.

`[model].projection_kernel` (default `true`) runs the target model's quantized
projections through one small-row kernel for every speculative verify and for
every call of eight rows or fewer outside it: each decode step, and the short
tail that ends a prefill (sous's simdgroup-matrix kernel, derived in part from
Splash under Apache-2.0 — see `THIRD_PARTY_NOTICES.md` — and compiled at model
load, no build step). Its cost stays nearly flat from one verify row to eight,
which is what makes a deeper verify block pay: on an M5 Pro with the default
model and drafter, a block-5 round with the kernel cost 5 ms more than a
block-4 round without it (115 against 110 ms) and produced 3.46 tokens against
3.00, so sampled decode at block 5 ran 1.10x block 4 without the kernel (95% CI
1.07–1.14) on 64 real subagent turns at 44–77K of context (#148). That is why
an unset `speculative_block_size` is 5 wherever the kernel is active. Each
row's output is bitwise independent of how many rows a call carries and of
which projections share it, so greedy output with the drafter stays identical
to output without it — and that is also why plain decode goes through the
kernel, at a cost: with no drafter, decode runs about 0.78x what it does with
`false` (10.7 against 13.6 tok/s on those turns). The kernel is closer to an
exact reference than mlx's own one-row kernel, but not bit-identical to it, so
a near-tie can resolve the other way than with `false`, which runs the stock
paths. It is active only where the attention tile is active, which so far
means a 20-core M5 Pro, with a dense Qwen3.5-family model whose projections
are affine 4-bit at group size 64, no drafter or a DFlash one, and the mlx and
mlx-vlm sources it was validated with. Anywhere else the model-load line reads
`projection_kernel=unavailable` (`off` on the mlx-lm backend), the status
document's `projection_kernel` block carries the reason, the projections run
stock and an unset block stays 4. As with the tile, that is one INFO line, and
a `WARNING` saying `sous: projection kernel unavailable (…)` means the kernel
should have run here and did not. The on-disk forks are keyed by it: where it
is active, changing the setting starts the fork store cold once. The daemon
reads it at startup, so a change takes effect on its next start.

`[server].generation_timeout_minutes` bounds a turn at both ends: how long it
```

In `docs/configuration.md`, replace the speculative-decoding paragraph up to its clamp sentence:

```markdown
Speculative decoding (`speculative_draft_id`, `speculative_block_size`) is
~1.8x decode on the default model at short context with the shipped
sampling; `""` disables it. Greedy (temperature 0) speculative decode ran
2.0x plain greedy decode on an M5 Pro (27.1 against 13.6 tok/s at block 4;
1.83x at block 3), on 64 real subagent turns at 44–77K tokens, with output
identical to plain decode on every turn (#118).
The "~2.4x greedy" figure earlier versions quoted ran an argmax sampler
through the sampled speculative walk, before the engine reached mlx-vlm's
greedy branch, and that branch also drafts differently (#87). Block size 4
measured best on those turns, sampled and greedy: +3–5% decode over block 3,
with 2 and 6 clearly slower (about 0.8x block 3) and 5 level with 3, its
interval overlapping 4's (#118). 0 lets the drafter's adaptive policy
pick the depth. Anything above 5 is clamped, because no larger block has
paid: block 6 ran 0.8x block 3 on the M5 Pro, and without the attention tile
6+ verify rows also leave mlx's fused attention kernel. It auto-disables with a
```

with:

```markdown
Speculative decoding (`speculative_draft_id`, `speculative_block_size`) is
~1.8x decode on the default model at short context with the shipped
sampling; `""` disables it. Left unset, the block size is 5 where the
projection kernel is active and 4 elsewhere; a value you set always wins, and
the model-load line's `draft_block=` says which block the engine runs.
Greedy (temperature 0) speculative decode at block 5 with the projection
kernel ran about 2.3x stock plain greedy decode on an M5 Pro — stock meaning
drafter-off decode with `projection_kernel = false`, 13.6 tok/s on 64 real
subagent turns at 44–77K tokens — against 2.0x at block 4 without the kernel
(27.1 against 13.6 tok/s; 1.83x at block 3). Either way, output with the
drafter was identical to plain decode under the same setting on every turn
(#118, #148). Measured against the kernel's own drafter-off decode, which is
slower (see `projection_kernel` above), the same runs are about 2.9x.
The "~2.4x greedy" figure earlier versions quoted ran an argmax sampler
through the sampled speculative walk, before the engine reached mlx-vlm's
greedy branch, and that branch also drafts differently (#87). Without the
projection kernel, block 4 measured best on those turns, sampled and greedy:
+3–5% decode over block 3, with 2 and 6 clearly slower (about 0.8x block 3)
and 5 level with 3, its interval overlapping 4's (#118); with it, block 5 is
the faster one. 0 lets the drafter's adaptive policy pick the depth. Anything
above 5 is clamped: block 6 ran 0.8x block 3 on the M5 Pro without the
projection kernel, has been measured with it only in an out-of-tree spike, and
without the attention tile 6+ verify rows also leave mlx's fused attention
kernel. It auto-disables with a
```

- [ ] **Step 7: Document the load-line tokens**

In `docs/observability.md`, replace:

```markdown
where the tile does not apply to the Mac, the model or the drafter, and as
a `WARNING py.warnings` line where it should have run and did not). One
more line names the Anthropic tool *types* a turn dropped, when any.
```

with:

```markdown
where the tile does not apply to the Mac, the model or the drafter, and as
a `WARNING py.warnings` line where it should have run and did not), then
`projection_kernel=active|unavailable|off` (whether the target's quantized
projections for speculative verify and for every call of up to eight rows —
decode and short prefill tails — run on the small-row projection kernel, with
`projection_kernel_probe_s=` when they do; `off` means
`[model].projection_kernel = false` or the mlx-lm backend, and `unavailable`
a load the kernel cannot serve, every load whose attention tile is not
`active` included, since the kernel rides on the tile — the status document's
`projection_kernel` block carries the reason, and the log splits INFO and
`WARNING` the way it does for the tile), and, whenever a drafter loaded,
`draft_block=` (the verify block the engine pinned: the configured
`speculative_block_size`, or for an unset one 5 where the projection kernel
is active and 4 elsewhere, 0 being the drafter's own policy; the status
document's engine block carries it as `draft_block`). One
more line names the Anthropic tool *types* a turn dropped, when any.
```

- [ ] **Step 8: Say in the tuning docs how the kernel is keyed and how the current arm is found**

In `docs/tuning.md`, replace:

```markdown
Every arm inherits `[model].attention_tile` and no arm changes it; suite
rows are keyed by it, so a `--resume` after it was flipped runs those arms
again instead of mixing tile and stock rows, and where the engine reports
the tile `unavailable`, the arm prints a notice with the reason and runs
stock, as the daemon would (an `off` engine, such as the mlx-lm backend's,
gets no notice).
```

with:

```markdown
Every arm inherits `[model].attention_tile` and `[model].projection_kernel`
and no arm changes them; suite rows are keyed by both, so a `--resume` after
either was flipped runs those arms again instead of mixing kernel and stock
rows, and where the engine reports one `unavailable`, the arm prints a notice
with the reason and runs stock, as the daemon would (an `off` engine, such as
the mlx-lm backend's, gets no notice).

Every drafter arm runs exactly the block its label names. Your configured arm
is the block the daemon would run: with `speculative_block_size` unset, that
is 5 where the projection kernel can run and 4 elsewhere. The tune decides
which without loading the model — from the platform rule, the GPU's core
count, `attention_tile` and `projection_kernel`, the model's config and the
drafter's kind — and prints a line when the unset block resolves to 5. A
refusal only a load would show (a failed probe, a pinned mlx-vlm source that
changed) leaves that mark one block off; the measurements themselves are
unaffected. A proposal then leaves an unset block alone when the winner runs
the block it already resolves to, writes any other block explicitly, and
always writes the block of a winner with another model or drafter, since an
unset block could resolve differently for it.
```

- [ ] **Step 9: Name the kernel in how-it-works' fork-store paragraph**

In `docs/how-it-works.md`, replace:

```markdown
positions owner, int8 state, attention-tile state and macOS build match the
ones that wrote it; anything else
```

with:

```markdown
positions owner, int8 state, attention-tile state, projection-kernel state
and macOS build match the ones that wrote it; anything else
```

- [ ] **Step 10: Check nothing stale is left, and commit**

```bash
grep -n "projkernel.py" CLAUDE.md
grep -n "^speculative_block_size" docs/configuration.md
grep -c "projection_kernel" docs/configuration.md docs/observability.md docs/tuning.md
grep -n "^## " THIRD_PARTY_NOTICES.md
uv run python -c "import sous.engine.kernels as k; print(k.__doc__.splitlines()[0])"
git diff --stat
```

Expected:
- The first grep prints four lines: the epoch list (`  projkernel.py, kernels/*.metal and *.h) beside the versions, the GPU and`), and three in the new bullet — its first line, its `_EPOCH_FILES` sentence and its `tests/test_projkernel.py`.
- The second prints nothing: no uncommented block size is left in the sample config.
- Every count is at least 1.
- The notices have two sections, `## oMLX INT8-activation prefill kernels` and `## Splash simdgroup-matrix Q4 projection kernel`.
- The docstring's first line is `Metal sources for sous's runtime-compiled kernels: int8 prefill, the`.
- The diff touches exactly `CLAUDE.md`, `THIRD_PARTY_NOTICES.md`, `docs/configuration.md`, `docs/how-it-works.md`, `docs/observability.md`, `docs/tuning.md` and `src/sous/engine/kernels/__init__.py`.

```bash
uv run ruff check . && uv run ruff format --check .
git add CLAUDE.md THIRD_PARTY_NOTICES.md docs/configuration.md docs/how-it-works.md docs/observability.md docs/tuning.md src/sous/engine/kernels/__init__.py
git commit -m "docs: record the projection kernel's contract, its Splash provenance and the resolved block" -m "The kernel keeps greedy parity only through its arithmetic contract, two
tag-scoped hooks, a plain-int flag and a load-time probe, and every one of
them is a trap for whoever next edits the kernel or bumps mlx or mlx-vlm,
so CLAUDE.md records them beside the tile's, with the argmax residual the
probe cannot cover and the rule that a kernel edit re-runs the exactness
tests and the timing check. A line-by-line comparison with Splash and its
Apple7/8 port settled what the kernel derives from them, and the notice
names exactly that. The user docs say what [model].projection_kernel does
and costs without a drafter, that an unset block is 5 where it is active,
what the load line and the status document report, and how sous tune finds
the configured arm.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: The real-model test

**Files:**
- Create: `tests/test_projkernel_model.py`

**Interfaces:**
- Consumes:
  - Task 2: `projkernel.kernel_available()`.
  - Task 3: `projkernel.calls` (keys `verify_kernel`, `verify_split`, `plain_kernel`, `declined`) and `projkernel._active`.
  - Task 4: `projkernel.static_reason(config, model_config, drafter_kind)`.
  - Task 6:
    - `VLMEngine(model_id, ..., draft_id=..., draft_block_size=..., attention_tile=..., projection_kernel=..., draft_block_explicit=...)`;
    - `projection_kernel_status`, `_draft_block_size`, `_draft`, `unload()`;
    - `SousConfig()` defaults `attention_tile=True` and `projection_kernel=True`;
    - the block resolution: an unexplicit block becomes `SPECULATIVE_BLOCK_KERNEL` (5) only when a drafter loaded and the kernel is `active`.
  - `tileattn.availability()` and `sous.engine.base.fetch_model_config(model_id)`, both unchanged.
- Produces: one `-m model` test. It runs on the M5 Pro in Task 11 and is deselected in CI.

Facts the test is built on (read in mlx-vlm 0.7.2):
- With `draft_block_size` 0, `VLMEngine.decode` passes `None`, and `_dflash_block_total` takes the drafter's own ceiling. For DFlash2 that is `min(config.block_size, runtime_block_size)`. The 27B's drafter ships `block_size` 8 and no `runtime_block_size`, which `DFlash2Config.from_dict` sets to `min(5, 8)` = 5. So the drafter's own policy verifies 3 to 5 rows on this model, never more than 8.
- The verify split therefore needs a block above 8 set on the engine. The drafter builds a block of any length (`draft_block` takes `block_size` as given), and nothing in the verifier caps T. The test uses block 12, verified as runs of 8 and 4.
- `_pin_block_size` pins `prefer_requested_block_size` only for a block above 0. An engine built with block 5 keeps that pin when the test sets 8 or 12 on it, so those rounds verify exactly that many rows, less the budget's cut at the end. Block 0 needs an engine built with 0, so that nothing is pinned.

- [ ] **Step 1: Write the real-model test**

Create `tests/test_projkernel_model.py`:

```python
"""The projection kernel on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 15 minutes, most of it five model loads and
drafter-off decode on the kernel):
- With the attention tile and the kernel on, an engine reports the kernel active; the test skips,
  naming the reason, where it cannot be.
- An unset block resolves to 5 with the kernel active and to 4 with it off.
- Greedy output with the DFlash2 drafter equals output without it at block 5, at blocks 8 and 12
  set on the engine directly (config caps the block at 5), and at block 0, the drafter's own
  policy. DFlash2's own policy verifies at most 5 rows (its runtime block is min(5, 8)), so block
  12 is the one whose verify rows exceed 8 and are split into runs.
- Over the kernel runs both hooks are reached, nothing is declined, and only block 12 splits.
- An engine with the kernel off, built after the kernel engines unloaded, reproduces the output
  of one built before any of them, and makes no kernel call.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.config import SousConfig  # noqa: E402 — after the importorskip guard
from sous.engine import projkernel, tileattn  # noqa: E402
from sous.engine.base import fetch_model_config  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# name -> (prompt tokens, max_tokens values): one prompt whose generation stays below the
# attention tile's 1,536 keys and one above them, so the kernel is checked beside both attention
# paths; two caps on the short prompt end its last round at different row counts.
PROMPTS = {"short": (600, (256, 257)), "long": (3000, (256,))}
SLACK = 16
SPLIT_BLOCK = 12


@contextlib.contextmanager
def _engine(
    *, kernel: bool, drafter: bool, block: int = 4, explicit: bool = False
) -> Iterator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=block if drafter else 0,
        draft_block_explicit=explicit,
        attention_tile=True,
        projection_kernel=kernel,
    )
    try:
        yield engine
    finally:
        engine.unload()


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
    """({(prompt, max_tokens): text}, projkernel counter deltas) at the engine's current block."""
    before = dict(projkernel.calls)
    texts = {}
    for name, (messages, caps) in prompts.items():
        for cap in caps:
            texts[(name, cap)] = engine.generate(messages, [], cap)
    return texts, {k: v - before[k] for k, v in projkernel.calls.items()}


def test_greedy_output_with_the_drafter_equals_output_without_it_on_the_kernel():
    tile = tileattn.availability()
    if not tile.available:
        pytest.skip(f"attention tile unavailable: {tile.reason}")
    kernel = projkernel.kernel_available()
    if not kernel.available:
        pytest.skip(f"projection kernel unavailable: {kernel.reason}")
    why = projkernel.static_reason(SousConfig(), fetch_model_config(DEFAULT_27B), "dflash")
    if why is not None:
        pytest.skip(f"projection kernel unavailable: {why}")

    # Before any engine of this test activates the kernel: in a process that has run no kernel
    # engine, these are main's paths with no hook installed.
    with _engine(kernel=False, drafter=True) as stock:
        assert stock.projection_kernel_status["state"] == "off", stock.projection_kernel_status
        assert stock._draft_block_size == 4
        ids = _corpus(stock, max(target for target, _ in PROMPTS.values()) + 64)
        prompts = {
            name: (_prompt(stock, ids, target), caps) for name, (target, caps) in PROMPTS.items()
        }
        reference, _ = _generate(stock, prompts)

    with _engine(kernel=True, drafter=False) as plain:
        status = plain.projection_kernel_status
        if status["state"] != "active":
            pytest.skip(f"projection kernel {status['state']}: {status['reason']}")
        undrafted, plain_counts = _generate(plain, prompts)

    drafted, drafted_counts = {}, {}
    with _engine(kernel=True, drafter=True) as engine:
        assert engine.projection_kernel_status["state"] == "active", engine.projection_kernel_status
        assert engine._draft_block_size == 5
        for block in (5, 8, SPLIT_BLOCK):
            engine._draft_block_size = block
            drafted[block], drafted_counts[block] = _generate(engine, prompts)

    with _engine(kernel=True, drafter=True, block=0, explicit=True) as policy:
        assert policy.projection_kernel_status["state"] == "active", policy.projection_kernel_status
        assert policy._draft_block_size == 0
        assert not getattr(policy._draft, "prefer_requested_block_size", False)
        drafted[0], drafted_counts[0] = _generate(policy, prompts)

    with _engine(kernel=False, drafter=True) as off:
        assert off.projection_kernel_status["state"] == "off", off.projection_kernel_status
        assert off._draft_block_size == 4
        assert projkernel._active == 0
        again, off_counts = _generate(off, prompts)

    for block, texts in drafted.items():
        for key, text in undrafted.items():
            assert texts[key] == text, (f"block {block}", *key)
    assert again == reference
    assert plain_counts["plain_kernel"] > 0, plain_counts
    assert plain_counts["verify_kernel"] == 0, plain_counts
    for block, counts in drafted_counts.items():
        assert counts["verify_kernel"] > 0, (block, counts)
        assert counts["plain_kernel"] > 0, (block, counts)
        assert (counts["verify_split"] > 0) == (block == SPLIT_BLOCK), (block, counts)
    assert all(c["declined"] == 0 for c in (plain_counts, *drafted_counts.values()))
    assert not any(off_counts.values()), off_counts
```

Why each count holds:
- The drafter-off engine verifies nothing, so its `verify_kernel` is 0. Every decode token and the one-row forward that ends each prefill reach hook 2, so its `plain_kernel` is above 0.
- Every drafted engine also ends each prefill in a one-row forward, so `plain_kernel` is above 0 there as well, beside the verify calls. The greedy drafter's own head goes through `quantized_argmax`, which neither hook serves.
- Block 12 is the only block above 8 rows. Its budget-cut final rounds can verify fewer, but its full rounds split.

- [ ] **Step 2: Run the test here to show it collects and skips**

This test needs no implementation step. It pins behaviour that Tasks 2–6 built, and it can run only on the M5 Pro.

Run: `uv run pytest -m model tests/test_projkernel_model.py -v -rs`

Expected on the maintainer's M2: `1 skipped`, with reason `attention tile unavailable: macOS 15.5 < 26.2`. No model loads, and nothing is fetched from the Hub, because the tile's platform rule refuses before `fetch_model_config` runs.

Run: `uv run pytest -m "not model" tests/test_projkernel_model.py`

Expected: `1 deselected`, which is how CI treats it.

The test runs on the M5 in Task 11, where it must pass. If an assertion fails there, the bug is in Tasks 2–6: fix it there and never loosen the assertion. Two failures have known readings:
- **A text mismatch.** Before blaming the kernel, find the first differing token. Then read the drafter-off run's top two logprobs at that token, as Task 11's `judge.py gap` does. A tie there is the argmax residual the spec names, not a kernel fault.
- **An exception inside mlx-vlm at block 12.** Report it with the traceback before changing the test. The split itself is pinned on synthetic weights by Task 3's tests.

- [ ] **Step 3: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model" tests/test_projkernel_model.py
git add tests/test_projkernel_model.py
git commit -m "test(engine): pin the projection kernel's greedy parity on the real model" -m "The unit and contract tests prove the kernel and its hooks on synthetic
weights; this holds the real 27B to the property the kernel's design
exists for. With the kernel active, greedy output with the DFlash2 drafter
equals output without it at block 5, at 8 and 12 set on the engine (12
verifies more than 8 rows, which the verify hook splits into runs) and at
block 0, the drafter's own policy. It also pins the engine-resolved block,
5 with the kernel and 4 without, and that an engine with the kernel off,
built after kernel engines ran in the process, reproduces one built before
them.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and pytest reports `1 deselected`.

---

---

### Task 11: The M5 Pro merge gate and the PR

This task is manual measurement, not code. Nothing it writes out of tree is ever committed.

**Where it runs.** The M5 Pro over ssh, through `m5.sh` (Step 3). It reaches the M5 by its LAN alias `m5pro-lan` and checks the host key recorded for the tailnet name, which it reads from `ssh -G m5pro` at run time. So no step spells out a machine's name. The M2 Air part runs on this Mac.

**What it never touches:**
- `~/code/personal/sous`, `~/.sous` and the maintainer's daemon;
- `~/code/personal/sous-remote`'s checkout and environment: only `git fetch` and `git worktree add` run there.

**Trees on the M5.** Main runs from a detached worktree `~/code/personal/sous-remote-main` at origin/main, and the branch from `~/code/personal/sous-remote-proj` at the branch head. Each has its own `.venv`. That is how #132's gate ran its main tree, so the gate never checks out or syncs `~/code/personal/sous-remote` itself.

**GPU lock.**
- Every M5 GPU job runs under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py`. It waits for the daemon to be idle and prints `daemon stayed idle during the job`, or `DAEMON-CONTAMINATED`.
- Every GPU job on this Mac runs under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py`.
- A run counts only if its log says the daemon stayed idle. Re-run any other.

**Time.** About nine hours of M5 GPU time:

| Part | Time |
|---|---|
| Load timings | 5 min |
| Non-model suite | 15 min |
| Model tests | 1 h |
| Port equals spike | 10 min |
| Kernel timings | 30 min |
| Replay chain A (speed ABBA, per-round arms) | 2.3 h |
| Replay chain B (drafter off, parity, off equals main) | 3.5 h |
| Suite | 50 min |
| `api_smoke.py` | 5 min |

The M2 part takes about 40 minutes of this Mac's GPU.

**On any failure, stop and report the numbers. Open no PR.** Three outcomes have a set response beyond that:
- **Speed, primary.** A 5k/4v point in [1.00, 1.05) with its CI above 1.00 is not a pass. It goes back to the maintainer with the numbers.
- **Cold load over 10 s.** Per-T lazy compilation must replace the load-time warm-up before merge. That is a code change for the maintainer to approve: report the four load timings.
- **Per-forward sum off #133's by more than 2%,** while every in-tree / spike ratio beside it is within 0.98–1.02. Re-run `ktime.py forward` once at another hour, since the M5 has run several percent slower at some hours of the day (#143's gate). If it fails again, report both tables.

**Files:**
- Out of tree, never committed:
  - Create on this Mac, in `~/code/sous-spikes/p148/gate/`, then rsync to `m5pro-lan:code/personal/sous-spikes/p148/gate/`:
    - `m5.sh`, the ssh helper;
    - `loadtime.py` and `loadtime.sh`, the load timings;
    - `wrap_gate.py`, the GATE_WRAP shim, and `rounds_gate.py`, its per-round records and arms;
    - `run_gate.sh` and `chains.sh`, the replay runs;
    - `porteq.py`, port equals spike on the real weights;
    - `port_vs_spike.py`, Task 2's synthetic port check (each variant forced, 1–4 linears), already in this directory: Step 16 runs it on the M5 beside `porteq.py`, as Task 2 says;
    - `ktime.py`, the kernel timings;
    - `suite/config.toml`, `suite_pair.py`, `run_suite.sh` and `analyze_pair.py`, the suite;
    - `judge.py` and `judge_all.sh`, the judgements.
  - Copy on the M5 into `~/code/personal/sous-spikes/p148/gate/`:
    - from `~/code/personal/sous-spikes/tile-132/gate/`: `gate132.py`, `analyze132.py` and `analyze133.py`;
    - from `~/code/personal/sous-spikes/p148/`: `analyze148.py`.
  - M5 trees: `~/code/personal/sous-remote-main` and `~/code/personal/sous-remote-proj`, detached worktrees of `~/code/personal/sous-remote`.
  - M2 (this Mac), in `~/code/sous-spikes/p148/m2/`:
    - copies of #143's harness, `gate143_m2.py` and `bodies-small.json`, from `~/code/sous-spikes/tile-132/m2/`;
    - the worktrees `main/` and `branch/`, each with its own `.venv`.
- May modify in-tree, only if the gate justifies it: `README.md`'s 57K decode figure and its Splash gap at 51–57K.

**Interfaces:**
- Consumes:
  - Task 2: `projkernel.linears(x, weights)` and `projkernel.linear(x, w, scales, biases)`, with x `[..., K]` of 1–8 rows (porteq.py, ktime.py); `~/code/sous-spikes/p148/gate/port_vs_spike.py`, which also forces each variant through `projkernel._launch(kind, x2, weights)` and reads the spike from `SPIKE_DIR`.
  - Task 3: `projkernel.calls` (`verify_kernel`, `verify_split`, `plain_kernel`, `declined`) and `projkernel._active`. The flag is read on every call, so the arms shim sets it per round.
  - Task 4: `projkernel.enable`. `vlm.py` reaches it through the module attribute (`projkernel.enable(...)`), as it does `tileattn.enable`, so the shim and `loadtime.py` time it by replacing that attribute. Both fail loudly if their timer never fires.
  - Task 6:
    - `VLMEngine(..., attention_tile=..., projection_kernel=..., draft_block_explicit=...)`;
    - `projection_kernel_status`, `attention_tile_status`, `_draft`, `_draft_block_size`, `_sampler`, `_draft_id`;
    - `[model].projection_kernel` in config.toml.
  - Task 7:
    - the load-line tokens `projection_kernel=`, `projection_kernel_probe_s=` and `draft_block=`, in that order after the attention tile's;
    - `/sous/status`'s `engine.projection_kernel` `{state, reason}`.
  - Task 8: `Arm.projection_kernel`. `suite_pair.py` builds an `Arm` in both trees and passes it only where the field exists.
  - Task 10: `tests/test_projkernel_model.py`.
  - Out of tree:
    - `gate132.py`'s interface: `GATE_WRAP`, `TILE132_COUNTERS`, `--model 'key = value'`, picks as `basename`, `basename:n` or `basename@i,j`, and results in `runs/<label>/result.json` with `daemon.out` beside it. It writes `max_context_tokens = 262144`, an isolated HOME and port 8384, and deletes the run's forks.
    - `analyze132.py equiv`, `analyze133.py runs|pair` and `analyze148.py`;
    - the spike's `verify_proj.py` and `bench.py` in `~/code/personal/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/`;
    - `gpu_run_m5.py`, `detach.py` and `wait_for.py`, and on this Mac `gpu_run.py`;
    - `gate143_m2.py` (`GATE_BODIES`, `GATE_MODEL_TOML`).
- Produces:
  - the gate verdict and its tables;
  - the PR, only after a passing gate;
  - optionally, a `docs:` commit with the README's 57K figures.

- [ ] **Step 1: Check the inputs on this Mac**

```bash
ls ~/code/sous-spikes/tile-132/m2/gate143_m2.py ~/code/sous-spikes/tile-132/m2/bodies-small.json \
  ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py \
  ~/code/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/verify_proj.py
```

Expected: all four are listed.
- The first two are #143's M2 harness, copied out of `/private/tmp` before #132's gate.
- The last is this Mac's copy of the spike kernels, which Step 4 sends to the M5 if its own copy is missing.
- If either M2 file is missing, the M2 criterion cannot run as specified: stop and report.

- [ ] **Step 2: Push the branch**

```bash
git fetch -q origin
git merge-base --is-ancestor origin/main HEAD && echo "based on origin/main" || echo "main moved"
```

If it prints `main moved`, rebase and re-run exactly what CI runs:

```bash
git rebase origin/main
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
```

Then push and note the head:

```bash
git push --force-with-lease -u origin feat/projection-kernel
git rev-parse --short HEAD
```

- [ ] **Step 3: Write the ssh helper**

```bash
mkdir -p ~/code/sous-spikes/p148/gate/suite
```

Create `~/code/sous-spikes/p148/gate/m5.sh`:

```sh
#!/bin/sh
# ssh to the M5 Pro checked against the host key recorded for its tailnet name, so its LAN alias
# works when Tailscale's DNS does not: `m5.sh m5pro-lan '<command>'`, and as rsync's remote
# shell: `rsync -e ~/code/sous-spikes/p148/gate/m5.sh ...`.
exec ssh -o HostKeyAlias="$(ssh -G m5pro | awk '$1 == "hostname" {print $2}')" "$@"
```

```bash
chmod +x ~/code/sous-spikes/p148/gate/m5.sh
~/code/sous-spikes/p148/gate/m5.sh m5pro-lan 'sw_vers -productVersion; sysctl -n kern.osversion'
```

Expected: the M5's macOS version and build (27.0 when this plan was written).

- [ ] **Step 4: Prepare the M5 trees and check the M5's inputs**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote && git fetch -q origin && for spec in main:origin/main proj:origin/feat/projection-kernel; do w=$HOME/code/personal/sous-remote-${spec%%:*}; if [ -d $w ]; then git -C $w checkout -q --detach ${spec#*:}; else git worktree add -q --detach $w ${spec#*:}; fi && (cd $w && uv sync -q && echo "${w##*/}: $(git log -1 --oneline)") || exit 1; done'
```

Expected:
- `sous-remote-main:` followed by origin/main's head.
- `sous-remote-proj:` followed by the head Step 2 printed.

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'S=~/code/personal/sous-spikes; K=$S/splash-port-2026-09-24/m5/projection-kernels; ls $S/acceptance-131/replay-all.json $S/acceptance-131/build_replay.py $S/tile-132/gate/gate132.py $S/tile-132/gate/analyze132.py $S/tile-132/gate/analyze133.py $S/p148/analyze148.py $K/verify_proj.py $K/kern.py $K/kern_st.py $K/bench.py $K/kern_mpp.py $K/kern_dq.py $K/kern_mppst.py $S/splash-port-2026-09-24/gpu_run_m5.py $S/splash-port-2026-09-24/detach.py $S/splash-port-2026-09-24/wait_for.py'
```

Expected: all 16 files are listed. The spike's `bench.py` imports every `kern*.py` beside it.

If only files under `projection-kernels/` are missing, send this Mac's copy and list again:

```bash
G=~/code/sous-spikes/p148/gate
rsync -a -e $G/m5.sh --include '*.py' --exclude '*' ~/code/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/ m5pro-lan:code/personal/sous-spikes/splash-port-2026-09-24/m5/projection-kernels/
```

If anything else is missing, stop and report.

- [ ] **Step 5: Write the load timings**

The spec bounds what the kernel adds to a load:
- **Warm:** at most 1 s.
- **Cold, empty Metal cache:** at most 10 s.

Metal keeps compiled pipelines in an on-disk cache shared across processes. So "the first load after install" is cold only if nothing on the M5 has compiled the 27B's pipelines from these sources before, and Task 10's run or an earlier test run may already have done so.

`loadtime.py` makes the cold case certain instead. It appends a fresh nonce comment to the header of every `sous_proj_*` kernel. That changes the source text Metal keys its cache on, and no arithmetic. A second process with the same nonce is then the warm case. This is the method #148's own compile-cost measurement used (`p148/compile_time.py`: 3.45 s cold, 0.02 s warm on the spike's kernels).

The `plain` run, the shipped sources and the first GPU job after Step 4's install, records the spec's "first load after install" as it stands. The `off` run is the baseline the others are measured against.

Create `~/code/sous-spikes/p148/gate/loadtime.py`:

```python
"""What the projection kernel adds to a model load (out of tree, never committed).

One load per process: the default 27B with its DFlash2 drafter, the attention tile on, the kernel
on or off, and projkernel.enable() timed from outside. Metal keeps compiled pipelines in an
on-disk cache shared by processes, so a cold load needs kernel sources Metal has never seen: with
a nonce, a comment carrying it is appended to the header of every kernel whose name starts with
sous_proj_ before any is built, which changes no arithmetic. The same nonce in a second process
is a warm load of the same sources; no nonce loads the sources as shipped.

usage: loadtime.py plain|off|cold|warm [nonce]
  plain      the kernel on, the sources as shipped
  off        projection_kernel = false: the load without the kernel
  cold/warm  the kernel on with the nonce: the same one for both, cold first
prints: loadtime mode=.. enable_s=.. state=.. reason=.. probe_s=.. draft_block=.. load_s=..
"""

import sys
import time

import mlx.core as mx

MODE = sys.argv[1]
NONCE = sys.argv[2] if len(sys.argv) > 2 else ""
assert MODE in ("plain", "off", "cold", "warm"), MODE
assert bool(NONCE) == (MODE in ("cold", "warm")), "cold and warm take a nonce, plain and off none"

if NONCE:
    _metal_kernel = mx.fast.metal_kernel

    def _with_nonce(*args, **kwargs):
        if str(kwargs.get("name", "")).startswith("sous_proj_"):
            kwargs["header"] = (kwargs.get("header") or "") + f"\n// gate nonce {NONCE}\n"
        return _metal_kernel(*args, **kwargs)

    mx.fast.metal_kernel = _with_nonce

from sous.engine import projkernel  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

timed: list[float] = []
_enable = projkernel.enable


def _timed_enable(*args, **kwargs):
    start = time.perf_counter()
    try:
        return _enable(*args, **kwargs)
    finally:
        timed.append(time.perf_counter() - start)


projkernel.enable = _timed_enable
start = time.perf_counter()
engine = VLMEngine(
    "mlx-community/Qwen3.8-27B-4bit",
    temperature=0.7,
    cache_budget=0,
    draft_id="z-lab/Qwen3.8-27B-DFlash2",
    draft_block_size=4,
    draft_block_explicit=False,
    attention_tile=True,
    projection_kernel=MODE != "off",
)
load_s = time.perf_counter() - start
status = engine.projection_kernel_status
assert len(timed) == 1, f"projkernel.enable ran {len(timed)} times: the timer missed the load"
print(
    f"loadtime mode={MODE} enable_s={timed[0]:.3f} state={status['state']} "
    f"reason={status['reason']!r} probe_s={status.get('probe_seconds')} "
    f"draft_block={engine._draft_block_size} load_s={load_s:.2f}",
    flush=True,
)
engine.unload()
```

Create `~/code/sous-spikes/p148/gate/loadtime.sh`:

```zsh
#!/bin/zsh
# The load timings in order, each its own process (out of tree, never committed): the shipped
# sources first (the first load since the branch tree was installed), the load without the
# kernel, then a fresh nonce cold and the same nonce warm.
cd ~/code/personal/sous-remote-proj || exit 1
G=~/code/personal/sous-spikes/p148/gate
N=$(python3 -c 'import secrets; print(secrets.token_hex(8))')
for args in plain off "cold $N" "warm $N"; do
  .venv/bin/python $G/loadtime.py ${=args} || exit 1
done
```

- [ ] **Step 6: Write the GATE_WRAP shim and its per-round records**

The shim is #132's `wrap_branch.py` plus three switches:
- per-round records in `analyze148.py`'s shape;
- the per-round arm draw of #148's spike, now through `projkernel._active` rather than the spike's own hooks;
- a token trace for the parity runs, so a divergence's top-2 logit gap can be read without re-running anything.

Speed ABBA runs use none of the three, so the branch carries no harness cost that main does not.

Create `~/code/sous-spikes/p148/gate/rounds_gate.py`. Its source pins were computed on mlx-vlm 0.7.2 when this plan was written, and are unchanged from `blocks148.py`:

```python
"""Per-round DFlash records and, in the arms mode, the arm drawn per round (out of tree, never
committed; from p148/blocks148.py, whose block mechanics and record fields it keeps unchanged, so
p148/analyze148.py reads its records as they are).

An arm is "<block><v|k>". v clears projkernel's flag for the round, so both hooks pass every
call through to mlx-vlm's exact verifier and stock qmv; k sets it, so every target call the
round makes takes the kernel: the verify projections and head, and the sampled drafter's call of
the target's head. Both hooks read the flag on every call, and a round's draft and verify are
built between this round's _dflash_next_block_size and the next one's, so setting the flag there
covers exactly that round. install(arms, seed=) patches three module globals of mlx-vlm 0.7.2's
speculative/dflash.py, each looked up per round inside _dflash_rounds:
  - _dflash_next_block_size: when the drafter pins its requested block (sous sets
    prefer_requested_block_size), return min(drawn, requested_block_total, remaining_budget)
    with the arm drawn from random.Random(seed); otherwise the original policy.
  - _sample_dflash_target_walk (sampled) and _speculative_walk (greedy): the originals,
    unchanged, timed. Both end in a host read, so the stamp after them is the round's end.
install(None) records rounds at the engine's own block and never touches the flag.

Per-round blocks are safe in 0.7.2's _dflash_rounds for the reasons blocks148.py gives (pinned
below): the target-cache reservation is sized once from the configured block, so any drawn block
at or under it fits, and nothing carries a block length from one round to the next.

Records stay in memory (no I/O, no extra device sync on the generation thread).
"""

from __future__ import annotations

import hashlib
import inspect
import random
import time

from mlx_vlm.speculative import cache_state, common, dflash

from sous.engine import projkernel

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
S = None  # the installed _State, or None


class _State:
    def __init__(self, arms, seed):
        self.arms = sorted(arms, key=lambda a: (int(a[:-1]), a[-1])) if arms else None
        for a in self.arms or ():
            assert a[-1] in "vk" and int(a[:-1]) >= 2, f"bad arm {a!r} (a block of 1 drafts nothing)"
        self.seed = seed
        self.rng = random.Random(seed)
        self.records: list[dict] = []
        self.decode = -1
        self.round = 0
        self.rem0 = None
        self.prev_rem = None
        self.prev_t_end = None
        self.new_decode = True
        self.pending = None


def install(arms=None, *, seed=0) -> None:
    global S
    S = _State(arms, seed)
    dflash._dflash_next_block_size = _next_block
    dflash._sample_dflash_target_walk = _sampled_walk
    dflash._speculative_walk = _greedy_walk


def begin_decode() -> None:
    if S is None:
        return
    if S.arms is not None:
        # A turn's prefill and first token run as the daemon runs them, whatever the last round
        # of the previous turn drew.
        projkernel._active = 1
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
            projkernel._active = 1 if arm[-1] == "k" else 0
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
        "kern": projkernel._active == 1,
        "requested": int(requested_block_total), "rem": int(remaining_budget),
        "base": st.rem0 - remaining_budget + 1, "t_start": t_start,
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

Create `~/code/sous-spikes/p148/gate/wrap_gate.py`:

```python
"""GATE_WRAP shim for the projection-kernel merge gate (out of tree, never committed; from
tile-132/gate/wrap_branch.py and p148/wrap148.py).

gate132.py runs `<tree>/.venv/bin/python wrap_gate.py` in place of `sous serve`. This runs the
branch's own `sous serve` and, after every VLMEngine.generate(), atomically writes to
$TILE132_COUNTERS what gate132.py records with each turn: the turn's counter deltas, the process
totals, the engine's projection_kernel and attention_tile states and its block. The counters are
tileattn.calls as they are, projkernel.calls under `proj_` and verifyattn.calls under
`verifyattn_`. Every projkernel.enable() logs its wall time to stderr, the run's daemon.out.

Optional, by environment:
  GATE_ROUNDS=<path>  per-round records (rounds_gate.py) and a summary after every generate(),
                      as JSON lines in p148/analyze148.py's shape; the path must not exist yet
  GATE_ARMS=4v,5k     with GATE_ROUNDS: the arm drawn per round from GATE_SEED (default 148);
                      refuses unless the kernel is active, the drafter pins its block and the
                      configured block is the largest arm block
  GATE_TRACE=<path>   per generate(), the decode call's token ids as JSON lines; with
                      GATE_TRACE_TOP2=1 also each token's two likeliest tokens and their logprobs,
                      a host read per token: parity runs only, never a timed one

usage: <tree>/.venv/bin/python wrap_gate.py [sous cli args...]   (no args = serve)
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

from sous.engine import projkernel, tileattn, verifyattn, vlm  # noqa: E402

COUNTERS = os.environ.get("TILE132_COUNTERS", "")
ROUNDS = os.environ.get("GATE_ROUNDS", "")
ARMS = [a for a in os.environ.get("GATE_ARMS", "").split(",") if a]
SEED = int(os.environ.get("GATE_SEED", "148"))
TRACE = os.environ.get("GATE_TRACE", "")
TOP2 = os.environ.get("GATE_TRACE_TOP2", "") == "1"
for _path in (ROUNDS, TRACE):
    if _path and os.path.exists(_path):
        raise SystemExit(f"gate148: {_path} exists; use a fresh path per run")
if ARMS and not ROUNDS:
    raise SystemExit("gate148: GATE_ARMS needs GATE_ROUNDS")
if ROUNDS:
    import rounds_gate  # noqa: E402
_turns = 0
_rounds_installed = False
_trace: list[list] = []  # one list of (generation index, row) per decode call of this turn


def say(msg: str) -> None:
    print(f"gate148: {msg}", file=sys.stderr, flush=True)


def counters() -> dict[str, int]:
    out = dict(tileattn.calls)
    out.update({f"proj_{k}": v for k, v in projkernel.calls.items()})
    out.update({f"verifyattn_{k}": v for k, v in verifyattn.calls.items()})
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
    status = engine.projection_kernel_status
    if status.get("state") != "active":
        return f"projection kernel {status}"
    if engine._draft is None:
        return "no drafter loaded"
    if not getattr(engine._draft, "prefer_requested_block_size", False):
        return "the drafter does not pin its block"
    if engine._draft_block_size != max(int(a[:-1]) for a in ARMS):
        return f"configured block {engine._draft_block_size} is not the largest arm block"
    return None


def _timed_enable(original):
    @functools.wraps(original)
    def enable(*args, **kwargs):
        start = time.perf_counter()
        status = original(*args, **kwargs)
        say(
            f"projkernel.enable seconds={time.perf_counter() - start:.3f} "
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
    projkernel.enable = _timed_enable(projkernel.enable)
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
                raise RuntimeError(f"gate148 refused: {why}")
            if not _rounds_installed:
                rounds_gate.install(ARMS or None, seed=SEED)
                _rounds_installed = True
                head = {
                    "type": "install", "pid": os.getpid(), "greedy": self._sampler is None,
                    "arms": ARMS or None, "seed": SEED, "configured_block": self._draft_block_size,
                    "drafter": self._draft_id, "proj": dict(self.projection_kernel_status),
                }
                _append(ROUNDS, [head])
                say(f"rounds installed {head}")
            rounds_gate.begin_decode()
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
                        "projection_kernel": dict(self.projection_kernel_status),
                        "attention_tile": dict(self.attention_tile_status),
                        "draft_block": self._draft_block_size if self._draft is not None else None,
                        "last_turn": {k: v - before.get(k, 0) for k, v in now.items()},
                        "totals": now,
                        "last_turn_seconds": round(seconds, 3),
                    },
                )
            if ROUNDS:
                recs = rounds_gate.drain()
                summary = {
                    "type": "turn", "turn": _turns, "key": key, "seconds": round(seconds, 3),
                    "rounds": len(recs),
                    "arms": dict(collections.Counter(str(r.get("arm")) for r in recs)),
                    "blocks": dict(collections.Counter(r["block"] for r in recs)),
                    "max_drafted": max((r["D"] for r in recs), default=0),
                    "tokens": sum(r["emitted"] for r in recs),
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

Create `~/code/sous-spikes/p148/gate/run_gate.sh`:

```zsh
#!/bin/zsh
# The projection-kernel merge gate's replay runs, one after another, each its own GPU-lock job
# with a hard timeout (out of tree, never committed; from tile-132/gate/run_gate.sh).
# usage: run_gate.sh <done-file> <spec>...   spec = label:tree:temp:maxtok:variant:turnset:mode
#   tree     main (~/code/personal/sous-remote-main, `sous serve`) or branch
#            (~/code/personal/sous-remote-proj, through wrap_gate.py)
#   variant  - | nodraft (speculative_draft_id = "") | off (projection_kernel = false)
#            | b5 (speculative_block_size = 5)
#   turnset  acc (gate132's default: #131's 64 turns) | first16 (the first 16 of those 64,
#            a160cc42's turns 0-15) | par (acc plus a6443762's continuation turns 108, 176, 239,
#            300 and 365, 97K to 229K tokens: 69 turns)
#   mode     - (counters only) | rounds (per-round records) | arms<seed> (4v against 5k drawn
#            per round, with rounds) | tok (decode token ids) | top2 (token ids and each
#            token's top two logprobs); every mode but - is the branch's alone
G=~/code/personal/sous-spikes/p148/gate
L=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
MAIN=~/code/personal/sous-remote-main
BRANCH=~/code/personal/sous-remote-proj
PY=$MAIN/.venv/bin/python
DONE=$1; shift
ACC=(agent-a160cc42252609969.jsonl agent-a257d5dad84e70f90.jsonl agent-a51b90a20d6d54ff6.jsonl
     agent-a7bb20c376e462d83.jsonl agent-a6443762163413a72.jsonl:10 agent-a442f184ac548b226.jsonl)
PAR=($ACC agent-a6443762163413a72.jsonl@108,176,239,300,365)
mkdir -p $G/logs $G/rounds $G/traces
cd $MAIN
rm -f $DONE $DONE.part
for spec in "$@"; do
  parts=(${(s/:/)spec})
  label=$parts[1]; tree=$parts[2]; temp=$parts[3]; maxtok=$parts[4]
  variant=$parts[5]; turnset=$parts[6]; mode=$parts[7]
  extra=(--model 'prompt_cache_gb = 14')
  case $variant in
    -) ;;
    nodraft) extra+=(--model 'speculative_draft_id = ""');;
    off) extra+=(--model 'projection_kernel = false');;
    b5) extra+=(--model 'speculative_block_size = 5');;
    *) echo "$label: unknown variant $variant" >> $DONE.part; break;;
  esac
  case $turnset in
    acc) picks=();;
    first16) picks=(agent-a160cc42252609969.jsonl:16);;
    par) picks=($PAR);;
    *) echo "$label: unknown turn set $turnset" >> $DONE.part; break;;
  esac
  case $tree in
    main) dir=$MAIN; wrap=;;
    branch) dir=$BRANCH; wrap=$G/wrap_gate.py;;
    *) echo "$label: unknown tree $tree" >> $DONE.part; break;;
  esac
  envs=()
  case $mode in
    -) ;;
    rounds) envs=(GATE_ROUNDS=$G/rounds/$label.jsonl);;
    arms*) envs=(GATE_ROUNDS=$G/rounds/$label.jsonl GATE_ARMS=4v,5k GATE_SEED=${mode#arms});;
    tok) envs=(GATE_TRACE=$G/traces/$label.jsonl);;
    top2) envs=(GATE_TRACE=$G/traces/$label.jsonl GATE_TRACE_TOP2=1);;
    *) echo "$label: unknown mode $mode" >> $DONE.part; break;;
  esac
  if [ $tree = main ] && [ $mode != - ]; then echo "$label: main takes no mode" >> $DONE.part; break; fi
  env GATE_WRAP=$wrap $envs GPU_LABEL=proj-gate python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 7200 \
    $PY $G/gate132.py $label $dir $temp $maxtok $extra $picks > $G/logs/$label.log 2>&1
  echo "$label exit=$? $(date +%H:%M:%S)" >> $DONE.part
  # nothing of ours may be left on the gate port before the next run
  if lsof -nP -iTCP:8384 -sTCP:LISTEN >/dev/null 2>&1; then echo "$label: port 8384 still busy" >> $DONE.part; break; fi
  if pgrep -f "gate132.py $label" >/dev/null || pgrep -f "runs/$label" >/dev/null; then echo "$label: process left" >> $DONE.part; break; fi
done
mv $DONE.part $DONE
```

Create `~/code/sous-spikes/p148/gate/chains.sh`. The runs, in order:
- **Chain A:**
  - the sampled ABBA at each tree's default: `main-s1`, `br-s1`, `br-s2`, `main-s2`;
  - the two per-round arms runs at configured block 5, seeds 248 and 249: `arms-1`, `arms-2`.
- **Chain B:**
  - the drafter-off ABBA on the first 16 turns: `main-nd1`, `br-nd1`, `br-nd2`, `main-nd2`;
  - parity, the branch with the drafter against without, greedy, max_tokens 256, on the 64 turns plus the continuation: `t-d-g`, `t-nd-g`;
  - off equals main, greedy, max_tokens 256, 64 turns: `main-g256`, `off-g256`.

The spec's greedy report on identical-output turns is read from `t-d-g` against `main-g256`. Both are drafter-on greedy runs at each tree's default, so the gate needs no further greedy pair. `t-d-g` carries only the token-id trace, one list append per token.

```zsh
#!/bin/zsh
# Replay chains A and B of the projection-kernel merge gate, back to back (out of tree, never
# committed): chain B starts only when chain A's done-file lists six exit=0 lines and nothing else.
G=~/code/personal/sous-spikes/p148/gate
zsh $G/run_gate.sh $G/logs/A_DONE \
  main-s1:main:0.7:1024:-:acc:- br-s1:branch:0.7:1024:-:acc:- br-s2:branch:0.7:1024:-:acc:- \
  main-s2:main:0.7:1024:-:acc:- arms-1:branch:0.7:1024:b5:acc:arms248 \
  arms-2:branch:0.7:1024:b5:acc:arms249 > $G/logs/chain-a.txt 2>&1
if [ "$(grep -c ' exit=0 ' $G/logs/A_DONE)" != 6 ] || [ "$(wc -l < $G/logs/A_DONE | tr -d ' ')" != 6 ]; then
  echo "chain A incomplete; chain B not started"; cat $G/logs/A_DONE; exit 1
fi
zsh $G/run_gate.sh $G/logs/B_DONE \
  main-nd1:main:0.7:1024:nodraft:first16:- br-nd1:branch:0.7:1024:nodraft:first16:- \
  br-nd2:branch:0.7:1024:nodraft:first16:- main-nd2:main:0.7:1024:nodraft:first16:- \
  t-d-g:branch:0.0:256:-:par:tok t-nd-g:branch:0.0:256:nodraft:par:top2 \
  main-g256:main:0.0:256:-:acc:- off-g256:branch:0.0:256:off:acc:rounds > $G/logs/chain-b.txt 2>&1
echo "chains done"; cat $G/logs/A_DONE $G/logs/B_DONE
```

- [ ] **Step 8: Write port equals spike**

Create `~/code/sous-spikes/p148/gate/porteq.py`. It covers 31 dispatches over the 12 dispatch keys:
- 2 for each fused group and for `down`, `q`, `in_proj_qkv` and `z`;
- 4 for `o/out_proj`, `k/v`, `gate/up` and `b/a`;
- 1 for `lm_head`.

The layer names were checked against the 27B's `model.safetensors.index.json`.

```python
"""Port equals spike, for the projection-kernel merge gate (out of tree, never committed; from
p148/par148.py).

On the default 27B's real weights, for each of the 12 dispatch keys the model makes, the in-tree
kernel (projkernel.linears) must give outputs bitwise equal (mx.array_equal, shapes included) to
the spike's verify_proj.mma_linears at every T = 1..8: the same activations into both, in the
call shape the hooks use ([1, T, K] bf16). A fused key runs its group as one dispatch; a single
key runs each of its modules alone. Two layers per family, so one layer's values cannot carry it.

usage: porteq.py   (one line per dispatch and layer, then PORT == SPIKE or MISMATCH)
"""

import glob
import json
import os
import sys

import mlx.core as mx

SPIKE = os.path.expanduser(
    "~/code/personal/sous-spikes/splash-port-2026-09-24/m5/projection-kernels"
)
sys.path.insert(0, SPIKE)
import verify_proj  # noqa: E402

from sous.engine import projkernel  # noqa: E402

IDX = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/"
        "model.safetensors.index.json"
    )
)[0]
WMAP = json.load(open(IDX))["weight_map"]
P = "language_model.model.layers."
ATTN = (3, 63)  # full attention every fourth layer
GDN = (0, 62)
MLP = (3, 62)


def attn(i, *names):
    return [f"{P}{i}.self_attn.{n}" for n in names]


def gdn(i, *names):
    return [f"{P}{i}.linear_attn.{n}" for n in names]


def mlp(i, *names):
    return [f"{P}{i}.mlp.{n}" for n in names]


def singles(names):
    return [[n] for n in names]


# dispatch key -> the dispatches it is checked on; each dispatch is one call sharing x
KEYS = {
    "q+k+v": [attn(i, "q_proj", "k_proj", "v_proj") for i in ATTN],
    "gate+up": [mlp(i, "gate_proj", "up_proj") for i in MLP],
    "qkv+z+b+a": [gdn(i, "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a") for i in GDN],
    "o/out_proj": singles([n for i in ATTN for n in attn(i, "o_proj")])
    + singles([n for i in GDN for n in gdn(i, "out_proj")]),
    "down": singles([n for i in MLP for n in mlp(i, "down_proj")]),
    "lm_head": [["language_model.lm_head"]],
    "q": singles([n for i in ATTN for n in attn(i, "q_proj")]),
    "k/v": singles([n for i in ATTN for n in attn(i, "k_proj", "v_proj")]),
    "gate/up": singles([n for i in MLP for n in mlp(i, "gate_proj", "up_proj")]),
    "in_proj_qkv": singles([n for i in GDN for n in gdn(i, "in_proj_qkv")]),
    "z": singles([n for i in GDN for n in gdn(i, "in_proj_z")]),
    "b/a": singles([n for i in GDN for n in gdn(i, "in_proj_b", "in_proj_a")]),
}
_shards: dict = {}


def weights(name):
    shard = os.path.join(os.path.dirname(IDX), WMAP[name + ".weight"])
    if shard not in _shards:
        _shards.clear()  # one shard mapped at a time
        _shards[shard] = mx.load(shard)
    d = _shards[shard]
    out = tuple(mx.contiguous(d[f"{name}.{part}"]) for part in ("weight", "scales", "biases"))
    mx.eval(*out)
    return out


def main():
    print(mx.device_info().get("architecture"), "mlx", mx.__version__, flush=True)
    ok = True
    dispatches = 0
    for key, groups in KEYS.items():
        for names in groups:
            ws = [weights(n) for n in names]
            k = ws[0][0].shape[1] * 8
            same = []
            for t in range(1, 9):
                x = (mx.random.normal((1, t, k), key=mx.random.key(1000 * t + k)) * 2).astype(
                    mx.bfloat16
                )
                mx.eval(x)
                tree = projkernel.linears(x, ws)
                spike = verify_proj.mma_linears(x, ws)
                eq = len(tree) == len(spike) and all(
                    a.shape == b.shape and mx.array_equal(a, b).item()
                    for a, b in zip(tree, spike, strict=True)
                )
                same.append(eq)
                ok &= eq
            dispatches += 1
            layer = names[0].split(".")[3] if P in names[0] else "-"
            short = "+".join(n.rsplit(".", 1)[-1] for n in names)
            marks = " ".join("=" if s else "X" for s in same)
            print(f"{key:<12} layer {layer:>2} {short:<40} T=1..8 {marks}", flush=True)
            del ws
        mx.clear_cache()
    print(f"{dispatches} dispatches x T = 1..8", flush=True)
    print("PORT == SPIKE" if ok else "MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Write the kernel timings**

Create `~/code/sous-spikes/p148/gate/ktime.py`.
- **#133's per-forward figures** are its `mma-only` column (`p133/m5/analysis.txt`): 75.00, 77.16, 78.55, 78.12, 78.22, 77.24, 78.20 and 78.92 ms at T = 1..8. They were measured with the spike bench's `measure()` on single linears, so the forward sum here uses the same function and the same calls per shape.
- **The cells** time each of the 12 dispatch keys as the hooks dispatch them, fused groups included.

```python
"""Kernel timings for the projection-kernel merge gate (out of tree, never committed; from
p133/bench133.py, reusing the spike bench's weights, dependent chain and glue subtraction).

Cells: each of the 27B's 12 dispatch keys at T = 1..8. A sample is one dependent chain over the
key's rotating weight copies (call i+1 reads call i's first output, cut or padded back to K),
less the median of the padding's own chain, divided by the copies: ms per call. Per cell the
variants are sampled round-robin, in an order rotated every round and reversed on odd rounds so
drift spreads evenly: the in-tree kernel (projkernel.linears) twice, an A/A pair whose gap is the
noise floor, the spike's verify_proj.mma_linears, and the padding alone. A cell passes when
tree / spike is within the larger of 3% and the A/A gap. `tree == spike` compares the outputs
bitwise on these random weights (informational; porteq.py decides it on the real ones).

Per forward: #133's sum, the in-tree single-linear kernel over the 27B's verify-forward
projections (#133's calls per shape, in_proj_a/b excluded), timed with the spike bench's own
measure() at T = 1..8, the spike's sum beside it. It passes when it is within 2% of #133's
figure at that T; tree / spike tells a drift of the machine from one of the port.

usage: ktime.py [cells|forward|all]   (default all)
"""

import math
import os
import statistics
import sys
import time

import mlx.core as mx

SPIKE = os.path.expanduser(
    "~/code/personal/sous-spikes/splash-port-2026-09-24/m5/projection-kernels"
)
sys.path.insert(0, SPIKE)
import bench  # noqa: E402  the spike's bench: make_weights, n_rot, reps_for, measure, SHAPES
import verify_proj  # noqa: E402

from sous.engine import projkernel  # noqa: E402

KEYS = {
    "q+k+v": (5120, [12288, 1024, 1024]),
    "gate+up": (5120, [17408, 17408]),
    "qkv+z+b+a": (5120, [10240, 6144, 48, 48]),
    "o/out_proj": (6144, [5120]),
    "down": (17408, [5120]),
    "lm_head": (5120, [248320]),
    "q": (5120, [12288]),
    "k/v": (5120, [1024]),
    "gate/up": (5120, [17408]),
    "in_proj_qkv": (5120, [10240]),
    "z": (5120, [6144]),
    "b/a": (5120, [48]),
}
TS = range(1, 9)
# #133's verify forward: calls per bench.SHAPES shape, in_proj_a/b excluded.
FORWARD = {"q": 16, "kv": 32, "o": 64, "qkv": 48, "z": 48, "gate_up": 128, "down": 64, "lm_head": 1}
# #133's simdgroup-kernel sums over FORWARD, ms per forward (p133/m5/analysis.txt, mma-only).
P133 = {1: 75.00, 2: 77.16, 3: 78.55, 4: 78.12, 5: 78.22, 6: 77.24, 7: 78.20, 8: 78.92}
MIN_MS, MIN_REPS, MAX_REPS = 200.0, 9, 41


def tree(x, per):
    return projkernel.linears(x, [(w, s, b) for w, s, b, _ in per])


def spike(x, per):
    return verify_proj.mma_linears(x, [(w, s, b) for w, s, b, _ in per])


def tree1(x, per):
    w, s, b, _ = per[0]
    return (projkernel.linear(x, w, s, b),)


def spike1(x, per):
    w, s, b, _ = per[0]
    return (verify_proj.mma_linear(x, w, s, b),)


def chain(fn, wsets, x0, pad):
    t, k = x0.shape

    def run():
        x = x0
        extras = []
        for per in wsets:
            ys = fn(x, per)
            extras.extend(ys[1:])
            y = ys[0]
            if y.shape[-1] >= k:
                x = y.reshape(-1)[: t * k].reshape(t, k)
            else:
                x = mx.concatenate([y, pad], axis=1)
        mx.eval(x, *extras)

    return run


def glue(wsets, x0, pad):
    n0 = x0.shape[1] - pad.shape[1]

    def run():
        x = x0
        for _ in wsets:
            x = mx.concatenate([x[:, :n0], pad], axis=1)
        mx.eval(x)

    return run


def sample(run):
    mx.synchronize()
    t0 = time.perf_counter()
    run()
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def cell(k, ns, t, wsets):
    x0 = (mx.random.normal((t, k), key=mx.random.key(7)) * 0.5).astype(mx.bfloat16)
    pad = (mx.random.normal((t, max(k - ns[0], 1)), key=mx.random.key(8)) * 0.5).astype(mx.bfloat16)
    mx.eval(x0, pad)
    runs = {
        "tree": chain(tree, wsets, x0, pad),
        "spike": chain(spike, wsets, x0, pad),
        "tree#2": chain(tree, wsets, x0, pad),
    }
    if ns[0] < k:
        runs["glue"] = glue(wsets, x0, pad)
    for run in runs.values():  # compile, then warm
        run()
        run()
    est = max(sample(run) for run in runs.values())
    reps = int(min(MAX_REPS, max(MIN_REPS, math.ceil(MIN_MS / max(est, 0.05)))))
    names = list(runs)
    samples = {n: [] for n in names}
    for r in range(reps):
        order = names[r % len(names) :] + names[: r % len(names)]
        if r % 2:
            order = order[::-1]
        for n in order:
            samples[n].append(sample(runs[n]))
    floor = statistics.median(samples["glue"]) if "glue" in samples else 0.0
    net = {n: [(s - floor) / len(wsets) for s in samples[n]] for n in ("tree", "spike", "tree#2")}
    med = {n: statistics.median(v) for n, v in net.items()}
    gap = abs(statistics.median(a / b for a, b in zip(net["tree"], net["tree#2"], strict=True)) - 1)
    same = all(
        mx.array_equal(a, b).item()
        for a, b in zip(tree(x0, wsets[0]), spike(x0, wsets[0]), strict=True)
    )
    return med, gap, same


def cells():
    print("| key | T | tree ms | spike ms | A/A gap | tree / spike | band | tree == spike | cell |")
    print("|---|---|---|---|---|---|---|---|---|")
    failures = 0
    for name, (k, ns) in KEYS.items():
        wsets = bench.make_weights(k, ns, bench.n_rot(k, ns))
        for t in TS:
            med, gap, same = cell(k, ns, t, wsets)
            ratio = med["tree"] / med["spike"]
            band = max(0.03, gap)
            ok = abs(ratio - 1.0) <= band
            failures += not ok
            print(
                f"| {name} | {t} | {med['tree']:.4f} | {med['spike']:.4f} | {gap * 100:.1f}% | "
                f"{ratio:.3f} | ±{band * 100:.1f}% | {same} | {'ok' if ok else 'FAIL'} |",
                flush=True,
            )
        del wsets
        mx.clear_cache()
    return failures


def forward():
    sums = {t: {"tree": 0.0, "spike": 0.0} for t in TS}
    for shape, calls in FORWARD.items():
        k, n = bench.SHAPES[shape]
        wsets = bench.make_weights(k, [n], bench.n_rot(k, [n]))
        reps = bench.reps_for(k, [n])
        for t in TS:
            order = ("tree", "spike") if t % 2 else ("spike", "tree")
            for name in order:
                fn = tree1 if name == "tree" else spike1
                _, (dep, _), (glue_ms, _) = bench.measure(fn, wsets, k, [n], t, reps)
                sums[t][name] += calls * (dep - glue_ms)
        del wsets
        mx.clear_cache()
    print("\n| T | tree ms / forward | spike ms / forward | #133 | tree / #133 | tree / spike | forward |")
    print("|---|---|---|---|---|---|---|")
    failures = 0
    for t in TS:
        tr, sp = sums[t]["tree"], sums[t]["spike"]
        ok = abs(tr / P133[t] - 1.0) <= 0.02
        failures += not ok
        print(
            f"| {t} | {tr:.2f} | {sp:.2f} | {P133[t]:.2f} | {tr / P133[t]:.3f} | {tr / sp:.3f} | "
            f"{'ok' if ok else 'FAIL'} |",
            flush=True,
        )
    return failures


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert what in ("cells", "forward", "all"), what
    bench.header()
    failures = {}
    if what in ("cells", "all"):
        failures["cells"] = cells()
    if what in ("forward", "all"):
        failures["forward"] = forward()
    bad = {k: v for k, v in failures.items() if v}
    print(f"ktime {'PASS' if not bad else f'FAIL {bad}'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Write the suite**

The spec's suite design is #148's suiteg, path-neutral and paired:
- **Prompts.** A seed is one prompt set, because sous's sampling is deterministic for a given prompt, and a prompt states its project root.
- **Runs.** Each (seed, tree) runs on a fresh load in a process of that tree, with the project root identical in both trees and the same run order.
- **Interpreter.** The tasks' commands and graders run under main's interpreter in both trees. A test runner's output then cannot differ by a venv path.

Seeds 5000–5005 are six prompt sets that no earlier suite used.

Create `~/code/sous-spikes/p148/gate/suite/config.toml`. It sets no block, so each tree runs its own default:

```toml
[server]
port = 8384
upstream_url = "http://127.0.0.1:9"

[model]
max_context_tokens = 262144
temperature = 0.7
```

Create `~/code/sous-spikes/p148/gate/suite_pair.py`:

```python
"""The projection-kernel merge gate's tool-loop suite, one (seed, tree) per process (out of tree,
never committed; from p148/suite148g.py).

sous tune's eight tasks on the default model at temperature 0.7, each tree at its own default:
main's block 4 against the branch's kernel and block 5. sous's sampling is deterministic for a
given prompt and a suite prompt states its project root, so a seed is one prompt set: the root is
<gate>/suite/w<seed>/arm/<task>-1/project in both trees (every Arm is labelled "arm"), and the run
directory moves to <gate>/suite/runs/<tree>-<seed> afterwards. The tasks' commands and graders run
under one interpreter for both trees ($SUITE_PAIR_PYTHON), so no tool output differs between them
by a path. run_suite loads the engine once and releases it: one load per process. On the branch
each row carries the run's projkernel.calls deltas; every row carries the engine's block.

usage: <tree>/.venv/bin/python suite_pair.py <pairs.jsonl> <seed> <tree label>
env:   SUITE_PAIR_CONFIG  the suite's config.toml; SUITE_PAIR_PYTHON  the tasks' interpreter;
       run under an isolated HOME
"""

import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from sous.config import load_config
from sous.engine.base import default_engine_factory
from sous.tune.arms import Arm
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import run_suite

try:
    from sous.engine import projkernel
except ImportError:  # main has no projection kernel
    projkernel = None


def counters() -> dict[str, int]:
    return dict(projkernel.calls) if projkernel is not None else {}


def main():
    out_path, seed, tree = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("sous.engine").setLevel(logging.INFO)
    cfg = load_config(Path(os.environ["SUITE_PAIR_CONFIG"]))
    config = dataclasses.replace(cfg, temperature=0.7, int8_prefill=False)
    mirrored = {}
    if "projection_kernel" in {f.name for f in dataclasses.fields(Arm)}:
        mirrored["projection_kernel"] = config.projection_kernel
    arm = Arm(
        label="arm", config=config, model_id=config.model_id,
        drafter_id=config.speculative_draft_id, block_size=config.speculative_block_size,
        window=config.max_context_tokens, tier="", current=False,
        attention_tile=config.attention_tile, **mirrored,
    )
    tasks = load_tasks()
    assert len(tasks) == 8, [t.name for t in tasks]
    root = Path(out_path).parent
    work = root / f"w{seed}"
    archive = root / "runs" / f"{tree}-{seed}"
    if archive.exists():
        raise SystemExit(f"{archive} exists: this (seed, tree) already ran")
    if work.exists():
        shutil.rmtree(work)
    state: dict = {}
    last = [counters()]

    def factory(model_id):
        engine = default_engine_factory(config, forks=False)(model_id)
        state.update(
            block=getattr(engine, "_draft_block_size", None),
            proj=dict(getattr(engine, "projection_kernel_status", None) or {"state": "absent"}),
        )
        return engine

    def record(run):
        now = counters()
        delta = {k: v - last[0].get(k, 0) for k, v in now.items() if v != last[0].get(k, 0)}
        last[0] = now
        row = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(run).items()}
        row.update(
            tree=tree, seed=seed, block=state.get("block"), proj=state.get("proj"),
            proj_calls=delta, archive=str(archive), ts=time.time(),
        )
        with open(out_path, "a") as out:
            out.write(json.dumps(row, default=str) + "\n")
        print(
            f"{tree} s{seed} {run.task} grade={run.grade} {run.seconds:.1f}s turns={run.turns} "
            f"out={run.output_tokens} block={state.get('block')} calls={delta} | "
            f"{run.grade_detail[:100]}",
            flush=True,
        )

    t0 = time.time()
    outcome = run_suite(
        arm, tasks, runs=1, done=set(), record=record, scratch=work, factory=factory,
        python=Path(os.environ["SUITE_PAIR_PYTHON"]),
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work), str(archive))
    print(
        f"== {tree} seed {seed} done in {time.time() - t0:.0f}s: released={outcome.released} "
        f"error={outcome.error} block={state.get('block')} proj={state.get('proj')}",
        flush=True,
    )
    if not outcome.released or outcome.error:
        raise SystemExit("the suite did not finish cleanly")


if __name__ == "__main__":
    main()
```

Create `~/code/sous-spikes/p148/gate/run_suite.sh`:

```zsh
#!/bin/zsh
# The suite's (seed, tree) processes in order, under one GPU lock (out of tree, never committed).
# usage: run_suite.sh <first seed> <last seed>   even seeds run main then the branch, odd seeds
#        the reverse, so drift falls on both trees alike; every process has the suite's own HOME
#        and runs the tasks' commands under main's interpreter.
G=~/code/personal/sous-spikes/p148/gate
MAIN=~/code/personal/sous-remote-main
BRANCH=~/code/personal/sous-remote-proj
for seed in $(seq $1 $2); do
  if (( seed % 2 )); then order=(branch main); else order=(main branch); fi
  for tree in $order; do
    if [ $tree = main ]; then dir=$MAIN; else dir=$BRANCH; fi
    echo "== seed $seed tree $tree $(date +%H:%M:%S)"
    env HOME=$G/suite/home HF_HUB_CACHE=$HOME/.cache/huggingface/hub \
      SUITE_PAIR_CONFIG=$G/suite/home/.sous/config.toml SUITE_PAIR_PYTHON=$MAIN/.venv/bin/python \
      perl -e 'alarm shift @ARGV; exec @ARGV' 1800 \
      $dir/.venv/bin/python $G/suite_pair.py $G/suite/pairs.jsonl $seed $tree || exit 1
  done
done
```

Create `~/code/sous-spikes/p148/gate/analyze_pair.py`:

```python
"""The suite's paired check (out of tree, never committed; from p148/analyze_suiteg.py): pairs are
(task, seed), compared on perfect runs with a one-sided exact McNemar test in the harm direction.

usage: analyze_pair.py <pairs.jsonl>
"""

import json
import sys
from math import comb

TREES = ("main", "branch")


def main():
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    by = {(r["task"], r["seed"], r["tree"]): r for r in rows}
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    for tree in TREES:
        rs = [r for r in rows if r["tree"] == tree]
        print(
            f"{tree}: {len(rs)} runs, mean grade {sum(r['grade'] for r in rs) / max(len(rs), 1):.3f}, "
            f"perfect {sum(r['grade'] >= 1 for r in rs)}/{len(rs)}, "
            f"blocks {sorted({str(r['block']) for r in rs})}, "
            f"task seconds {sum(r['seconds'] for r in rs):.0f}"
        )
    branch = [r for r in rows if r["tree"] == "branch"]
    routed = sum(
        1
        for r in branch
        if r["proj_calls"].get("verify_kernel", 0) > 0
        and r["proj_calls"].get("plain_kernel", 0) > 0
        and not r["proj_calls"].get("declined", 0)
    )
    stock = sum(1 for r in rows if r["tree"] == "main" and r["proj_calls"])
    print(
        f"branch runs on the kernel (verify and plain calls, none declined): {routed}/{len(branch)}; "
        f"main runs with kernel calls: {stock}"
    )
    print("per task (perfect runs):")
    for t in tasks:
        cells = "  ".join(
            f"{tree} {sum(by[(t, s, tree)]['grade'] >= 1 for s in seeds if (t, s, tree) in by)}"
            f"/{len(seeds)}"
            for tree in TREES
        )
        print(f"  {t:<24} {cells}")
    pairs = [
        (by[(t, s, "main")]["grade"] >= 1, by[(t, s, "branch")]["grade"] >= 1)
        for t in tasks
        for s in seeds
        if (t, s, "main") in by and (t, s, "branch") in by
    ]
    harm = sum(1 for m, b in pairs if m and not b)
    gain = sum(1 for m, b in pairs if b and not m)
    n = harm + gain
    p = sum(comb(n, i) for i in range(harm, n + 1)) / 2**n if n else 1.0
    verdict = "HARMFUL (p < 0.10)" if p < 0.10 else "no harm detected"
    print(
        f"branch vs main over {len(pairs)} (task, prompt) pairs: main-perfect/branch-not {harm}, "
        f"branch-perfect/main-not {gain}; one-sided exact McNemar p = {p:.4f} -> {verdict}"
    )
    print("Imperfect runs:")
    for r in sorted(rows, key=lambda r: (r["task"], r["seed"], r["tree"])):
        if r["grade"] < 1:
            print(
                f"  {r['tree']} s{r['seed']} {r['task']} grade={r['grade']:.2f} turns={r['turns']} "
                f"{r['grade_detail'][:120]} | {r['archive']}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Write the judgements**

Create `~/code/sous-spikes/p148/gate/judge.py`. It uses the standard library only, so it runs under any Python 3.10 or later; the steps below run it under a tree's own `.venv`.

```python
"""Judgements for the projection-kernel merge gate (out of tree, never committed). A judging
subcommand prints its evidence and ends with one line, `<name>: PASS` or `<name>: FAIL`, and
exits 0 only on PASS; `arms` also has `<name>: TO THE MAINTAINER` (exit 2). Report-only
subcommands (greedy, gap, figures) always exit 0. Standard library only.

usage: judge.py loadtime <loadtime.txt>
       judge.py load <runs> <label>=<state>:<block>...   state active|off; block 5|4|- (none)
       judge.py path <runs> <label>=<kind>...            kind drafted|nodraft|off
       judge.py off <runs> <label> <rounds.jsonl>
       judge.py arms <analyze148 output>
       judge.py abba <analyze133 pair output> <min ratio> [--prefill-memory]
       judge.py greedy <runs> <branch label> <main label>
       judge.py gap <runs> <traces dir> <drafted label> <undrafted label>
       judge.py figures <analyze133 pair output>
       judge.py m2 <runs> <main label> <branch label>
"""

import json
import os
import re
import sys

KV = re.compile(r"(\w+)=(\S+)")


def verdict(name, ok, extra=""):
    print(f"{name}: {'PASS' if ok else 'FAIL'}{extra}")
    return 0 if ok else 1


def result(runs, label):
    with open(os.path.join(runs, label, "result.json")) as f:
        return json.load(f)


def load_line(d):
    raw = d["load"][-1]["_raw"] if d.get("load") else ""
    return raw.split("INFO ", 1)[-1], dict(KV.findall(raw))


def norm(content):
    return json.dumps([{k: v for k, v in b.items() if k != "id"} for b in content or []], sort_keys=True)


def cmd_loadtime(path):
    rows = {}
    for line in open(path):
        if line.startswith("loadtime "):
            kv = dict(KV.findall(line))
            rows[kv["mode"]] = kv
            print(line.rstrip())
    if set(rows) != {"plain", "off", "cold", "warm"}:
        return verdict("loadtime", False, f" (modes {sorted(rows)})")
    base = float(rows["off"]["enable_s"])
    added = {m: float(rows[m]["enable_s"]) - base for m in ("plain", "cold", "warm")}
    states = all(
        rows[m]["state"] == "active" and rows[m]["draft_block"] == "5" for m in ("plain", "cold", "warm")
    ) and (rows["off"]["state"], rows["off"]["draft_block"]) == ("off", "4")
    print(
        f"added by the kernel: cold {added['cold']:.2f} s (bound 10), warm {added['warm']:.2f} s "
        f"(bound 1), first load after install {added['plain']:.2f} s (bound 10)"
    )
    ok = states and added["cold"] <= 10.0 and added["warm"] <= 1.0 and added["plain"] <= 10.0
    return verdict("loadtime", ok)


def enable_seconds(runs, label):
    out = []
    with open(os.path.join(runs, label, "daemon.out"), errors="replace") as f:
        for line in f:
            m = re.search(r"gate148: projkernel\.enable seconds=([\d.]+)", line)
            if m:
                out.append(float(m.group(1)))
    return out


def cmd_load(runs, specs):
    ok = True
    for spec in specs:
        label, _, want = spec.partition("=")
        state, _, block = want.partition(":")
        raw, kv = load_line(result(runs, label))
        seconds = enable_seconds(runs, label)
        good = kv.get("projection_kernel") == state and kv.get("draft_block", "-") == block
        good &= bool(seconds)
        if state == "active":
            good &= "projection_kernel_probe_s" in kv and kv.get("attention_tile") == "active"
            # every gate run loads after the load timings compiled the kernels: a warm load
            good &= max(seconds, default=99.0) <= 1.0
        print(f"{label}: {'ok' if good else 'FAIL'} enable_s={seconds} | {raw}")
        ok &= good
    return verdict("load", ok)


def cmd_path(runs, specs):
    ok = True
    for spec in specs:
        label, _, kind = spec.partition("=")
        d = result(runs, label)
        snap = (d["results"][-1].get("tile") or {}) if d["results"] else {}
        totals = snap.get("totals") or {}
        proj = {k[5:]: v for k, v in totals.items() if k.startswith("proj_")}
        state = (snap.get("projection_kernel") or {}).get("state")
        if kind == "off":
            good = state == "off" and bool(proj) and not any(proj.values())
        else:
            good = (
                state == "active"
                and proj.get("declined", 1) == 0
                and proj.get("plain_kernel", 0) > 0
                and (kind != "drafted" or proj.get("verify_kernel", 0) > 0)
            )
        print(f"{label} ({kind}): {'ok' if good else 'FAIL'} state={state} {proj}")
        ok &= good
    return verdict("path", ok)


def cmd_off(runs, label, rounds_path):
    d = result(runs, label)
    _, kv = load_line(d)
    over = []
    for r in d["results"]:
        ln = r.get("line") or {}
        if int(ln.get("drafted") or 0) > 3 * int(ln.get("draft_rounds") or 0):
            over.append(f"{r['agent'][6:14]}#{r['turn']}")
    recs = [json.loads(line) for line in open(rounds_path) if line.strip()]
    installs = [x for x in recs if x["type"] == "install"]
    rounds = [x for x in recs if x["type"] == "round"]
    max_d = max((x["D"] for x in rounds), default=None)
    print(
        f"{label}: load line draft_block={kv.get('draft_block')}, configured block "
        f"{[x['configured_block'] for x in installs]}, {len(rounds)} rounds, most drafted in one "
        f"round {max_d}, turn lines over 3 drafted per round: {over or 'none'}"
    )
    ok = (
        kv.get("draft_block") == "4"
        and [x["configured_block"] for x in installs] == [4]
        and bool(rounds)
        and max_d is not None
        and max_d <= 3
        and not over
    )
    return verdict("off", ok)


def cmd_arms(path):
    text = open(path).read()
    m = re.search(r"^\s+5k/4v: ([\d.]+) \[([\d.]+), ([\d.]+)\]$", text, re.M)
    if m is None:
        return verdict("arms", False, " (no PRIMARY 5k/4v line)")
    r, lo, hi = (float(g) for g in m.groups())
    print(f"5k/4v {r:.4f} [{lo:.4f}, {hi:.4f}]: pass at >= 1.05 with the CI's lower bound > 1.00")
    if lo > 1.0 and r >= 1.05:
        return verdict("arms", True)
    if lo > 1.0:
        print("arms: TO THE MAINTAINER (a point in [1.00, 1.05) with the CI above 1.00 is not a pass)")
        return 2
    return verdict("arms", False)


def pair_row(text, bucket):
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 15 and cells[1] == bucket:
            return cells
    return None


def ratio_ci(cells):
    lo, hi = (float(x) for x in cells[8].split("–"))
    return float(cells[7].strip("*")), lo, hi


def cmd_abba(path, min_ratio, prefill_memory):
    text = open(path).read()
    cells = pair_row(text, "all")
    try:
        r, lo, hi = ratio_ci(cells)
    except (TypeError, ValueError):
        return verdict("abba", False, " (no `all` row)")
    ok = r >= min_ratio
    print(f"all: branch/main {r:.3f} (95% CI {lo:.3f}-{hi:.3f}); bar >= {min_ratio:.2f}")
    if prefill_memory:
        tp, mp = float(cells[14]), float(cells[15])
        m = re.search(r"peak memory_gb over these turns: tile ([\d.]+), main ([\d.]+)", text)
        tm, mm = (float(g) for g in m.groups()) if m else (float("nan"), float("nan"))
        ok &= abs(tp / mp - 1.0) <= 0.03 and abs(tm - mm) <= 0.2
        print(
            f"prefill {tp} vs {mp} tok/s ({tp / mp - 1:+.1%}; bound ±3%), peak memory {tm:.2f} vs "
            f"{mm:.2f} GB (bound ±0.2)"
        )
    return verdict("abba", ok)


def cmd_greedy(runs, branch, main):
    a_runs = {(r["agent"], r["turn"]): r for r in result(runs, branch)["results"]}
    b_runs = {(r["agent"], r["turn"]): r for r in result(runs, main)["results"]}
    out = dec_a = dec_b = 0.0
    n = 0
    for key, b in b_runs.items():
        a = a_runs.get(key)
        if a is None or not a.get("line") or not b.get("line"):
            continue
        if norm(a["content"]) != norm(b["content"]) or a["stop"] != b["stop"]:
            continue
        tokens = int(a["line"]["output_tokens"])
        if tokens < 50:
            continue
        n += 1
        out += tokens
        dec_a += float(a["line"]["decode_s"])
        dec_b += float(b["line"]["decode_s"])
    if not n:
        print("greedy: no identical-output turn with >= 50 tokens")
        return 0
    print(
        f"greedy, identical-output turns with >= 50 tokens: {n} of {len(b_runs)}, {int(out)} tokens: "
        f"{branch} {out / dec_a:.2f} tok/s, {main} {out / dec_b:.2f} tok/s, {dec_b / dec_a:.3f}x"
    )
    return 0


def cmd_gap(runs, traces, drafted, undrafted):
    res = result(runs, undrafted)["results"]
    on = [json.loads(line) for line in open(os.path.join(traces, f"{drafted}.jsonl"))]
    off = [json.loads(line) for line in open(os.path.join(traces, f"{undrafted}.jsonl"))]
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
        where = f"{res[i]['agent'][6:14]}#{res[i]['turn']} token {j}"
        if j == n:
            print(f"{where}: one reply is a prefix of the other (a stop)")
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
            f"{where}: drafter off took {first} ({lp1}), runner-up {second} ({lp2}), gap "
            f"{lp1 - lp2}; drafter on took {took} -> {why}"
        )
    print(f"gap: {differing} differing turns of {len(res)}")
    return 0


def cmd_figures(path):
    cells = pair_row(open(path).read(), "32-64K")
    try:
        r, lo, hi = ratio_ci(cells)
    except (TypeError, ValueError):
        print("figures: no 32-64K row; the README's figures stay")
        return 0
    if lo <= 1.0 <= hi:
        print(f"figures: 32-64K branch/main {r:.3f} ({lo:.3f}-{hi:.3f}) includes 1.00; the README stays")
        return 0
    print(f"figures: 32-64K branch/main {r:.3f} ({lo:.3f}-{hi:.3f})")
    print(f"  57K decode: 23.5 x {r:.3f} = {23.5 * r:.2f} -> about {round(23.5 * r * 2) / 2:g} tok/s")
    print(f"  Splash at 51-57K: 1.4-1.6x / {r:.3f} -> {1.4 / r:.1f}-{1.6 / r:.1f}x")
    return 0


def cmd_m2(runs, main, branch):
    m, b = result(runs, main), result(runs, branch)
    errors = [r for r in (*m["results"], *b["results"]) if not r.get("id")]
    same = [
        norm(x.get("content")) == norm(y.get("content")) and x.get("stop") == y.get("stop")
        for x, y in zip(m["results"], b["results"], strict=True)
    ]
    raw, kv = load_line(b)
    status = (b.get("engine_status") or {}).get("projection_kernel") or {}
    tile = (b.get("engine_status") or {}).get("attention_tile") or {}
    over = [
        f"{r['agent']}#{r['turn']}"
        for r in b["results"]
        if r.get("line") and int(r["line"].get("drafted") or 0) > 3 * int(r["line"].get("draft_rounds") or 0)
    ]
    print(f"HTTP errors: {len(errors)}")
    print(f"greedy replies identical on {sum(same)}/{len(same)} turns")
    print(f"branch load line: {raw}")
    print(f"branch projection_kernel: {status}; attention_tile: {tile}")
    print(f"branch turns over 3 drafted per round: {over or 'none'}")
    ok = (
        not errors
        and all(same)
        and kv.get("projection_kernel") == "unavailable"
        and kv.get("draft_block") == "4"
        and status.get("state") == "unavailable"
        and bool(tile.get("reason"))
        and status.get("reason") == tile.get("reason")
        and not over
    )
    return verdict("m2", ok)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "loadtime":
        code = cmd_loadtime(args[0])
    elif cmd == "load":
        code = cmd_load(args[0], args[1:])
    elif cmd == "path":
        code = cmd_path(args[0], args[1:])
    elif cmd == "off":
        code = cmd_off(*args)
    elif cmd == "arms":
        code = cmd_arms(args[0])
    elif cmd == "abba":
        code = cmd_abba(args[0], float(args[1]), "--prefill-memory" in args[2:])
    elif cmd == "greedy":
        code = cmd_greedy(*args)
    elif cmd == "gap":
        code = cmd_gap(*args)
    elif cmd == "figures":
        code = cmd_figures(args[0])
    elif cmd == "m2":
        code = cmd_m2(*args)
    else:
        sys.exit(__doc__)
    sys.exit(code)


if __name__ == "__main__":
    main()
```

Create `~/code/sous-spikes/p148/gate/judge_all.sh`:

```zsh
#!/bin/zsh
# Every analysis of the projection-kernel merge gate, CPU only, after the last M5 GPU job (out of
# tree, never committed). Writes logs/ and prints each judgement's last line.
cd ~/code/personal/sous-remote-proj || exit 1
G=~/code/personal/sous-spikes/p148/gate
R=$G/runs
LG=$G/logs
PY=.venv/bin/python
$PY $G/judge.py loadtime $LG/loadtime.txt > $LG/j_loadtime.txt
$PY $G/judge.py load $R br-s1=active:5 br-s2=active:5 arms-1=active:5 arms-2=active:5 \
  br-nd1=active:- br-nd2=active:- t-d-g=active:5 t-nd-g=active:- off-g256=off:4 > $LG/j_load.txt
$PY $G/judge.py path $R br-s1=drafted br-s2=drafted arms-1=drafted arms-2=drafted \
  br-nd1=nodraft br-nd2=nodraft t-d-g=drafted t-nd-g=nodraft off-g256=off > $LG/j_path.txt
$PY $G/analyze148.py $G/rounds/arms-1.jsonl $G/rounds/arms-2.jsonl > $LG/arms.txt 2>&1
$PY $G/judge.py arms $LG/arms.txt > $LG/j_arms.txt
$PY $G/analyze133.py $R runs main-s1 br-s1 br-s2 main-s2 main-nd1 br-nd1 br-nd2 main-nd2 > $LG/decode_runs.md
$PY $G/analyze133.py $R pair br-s1 main-s1 br-s2 main-s2 > $LG/decode_pair.md
$PY $G/judge.py abba $LG/decode_pair.md 1.00 --prefill-memory > $LG/j_abba.txt
$PY $G/analyze133.py $R pair br-nd1 main-nd1 br-nd2 main-nd2 > $LG/nodraft_pair.md
$PY $G/judge.py abba $LG/nodraft_pair.md 0.75 > $LG/j_nodraft.txt
$PY $G/judge.py greedy $R t-d-g main-g256 > $LG/greedy.txt
$PY $G/analyze132.py $R equiv t-d-g t-nd-g > $LG/parity.txt
$PY $G/judge.py gap $R $G/traces t-d-g t-nd-g > $LG/gap.txt
$PY $G/analyze132.py $R equiv off-g256 main-g256 > $LG/off_main.txt
$PY $G/judge.py off $R off-g256 $G/rounds/off-g256.jsonl > $LG/j_off.txt
$PY $G/analyze_pair.py $G/suite/pairs.jsonl > $LG/suite.md
$PY $G/judge.py figures $LG/decode_pair.md > $LG/figures.txt
grep -h model_load $R/br-s1/daemon.out $R/br-nd1/daemon.out $R/off-g256/daemon.out $R/main-s1/daemon.out \
  | sed 's/^.*INFO //' > $LG/load_lines.txt
for f in j_loadtime j_load j_path j_arms j_abba j_nodraft j_off; do tail -n 1 $LG/$f.txt; done
head -n 1 $LG/parity.txt $LG/off_main.txt
grep -h -E '^PORT|^MISMATCH' $LG/porteq.txt $LG/port_synth.txt; grep -h '^ktime' $LG/ktime.txt
tail -n 1 $LG/greedy.txt $LG/gap.txt
grep -h -E 'no harm detected|HARMFUL' $LG/suite.md
```

- [ ] **Step 12: Sync the harness to the M5 and check that it compiles and imports**

```bash
G=~/code/sous-spikes/p148/gate
chmod +x $G/*.sh
rsync -a -e $G/m5.sh --exclude 'runs/' --exclude 'logs/' --exclude 'rounds/' --exclude 'traces/' --exclude 'suite/home/' --exclude 'suite/runs/' --exclude 'suite/w*/' --exclude 'suite/pairs.jsonl' $G/ m5pro-lan:code/personal/sous-spikes/p148/gate/
$G/m5.sh m5pro-lan 'S=~/code/personal/sous-spikes; G=$S/p148/gate; B=~/code/personal/sous-remote-proj/.venv/bin/python; M=~/code/personal/sous-remote-main/.venv/bin/python; cp $S/tile-132/gate/gate132.py $S/tile-132/gate/analyze132.py $S/tile-132/gate/analyze133.py $S/p148/analyze148.py $G/ && mkdir -p $G/logs $G/rounds $G/traces && $B -m py_compile $G/*.py && for f in $G/*.sh; do zsh -n $f || exit 1; done && cd $G && $B -c "import wrap_gate, rounds_gate, suite_pair, judge" && $M -c "import suite_pair" && echo HARNESS OK'
```

Expected: `HARNESS OK`.
- Importing `rounds_gate` checks its seven pinned mlx-vlm sources in the branch's environment. An `AssertionError` naming one means the lock has moved off mlx-vlm 0.7.2: stop and report.
- Importing `suite_pair` under main's interpreter checks that it builds without `projkernel`.
- `port_vs_spike.py`, `porteq.py` and `ktime.py` are compiled, not imported: each loads the spike's modules at import, and `port_vs_spike.py` looks for them under this Mac's path unless `SPIKE_DIR` says otherwise (Step 16 sets it).

- [ ] **Step 13: Run the load timings, the first GPU job on the branch tree**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/loadtime.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py zsh $G/loadtime.sh'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/loadtime.txt 540 12'
```

Repeat the `wait_for.py` call until it prints `DONE`; it takes about 3 minutes. Expected:
- four `loadtime` lines, `mode=plain`, `off`, `cold` and `warm`;
- `plain`, `cold` and `warm` read `state=active` and `draft_block=5`, with a `probe_s=` value;
- `off` reads `state=off` and `draft_block=4`;
- the log ends with `daemon stayed idle during the job` and `exit=0`.

Step 23 judges the bounds.

- [ ] **Step 14: Run the non-model suite on the M5**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/p148/gate/logs/notmodel.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py uv run pytest -m "not model" -rs'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/notmodel.txt 540 30'
```

Repeat the `wait_for.py` call until `DONE`. Then:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'L=~/code/personal/sous-spikes/p148/gate/logs/notmodel.txt; grep -c -E "^SKIPPED .*tests/test_(projkernel|tileattn)" $L; grep -E " passed| failed| error" $L | tail -2; tail -2 $L'
```

Expected:
- The count is `0`. Every projection-kernel test ran, the contract tests' kernel cases included, and so did every tile `nax` test.
- The summary line has passes and skips only, with no `failed` or `error`.
- The log ends with `daemon stayed idle during the job` and `exit=0`.

- [ ] **Step 15: Run the model tests on the M5**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/p148/gate/logs/model.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py zsh -c "rc=0; for t in projkernel tileattn verifyattn; do uv run pytest -m model -rs tests/test_\${t}_model.py || rc=1; done; exit \$rc"'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/model.txt 540 20'
```

Repeat the wait until `DONE`; it takes about an hour. Each file runs in a pytest process of its own, under one GPU lock. Each test's reference engine must be the first engine in its process, which is what its docstring rests on: the projection-kernel test's stock engine loads with no projection hook installed, and the tile test's tile-off engine with no tile hook. In one process the projection-kernel test's engines would install the tile's decode hook before the tile test's reference ran, and that test would still pass without proving what it says.

Expected:
- Three pytest summaries, each `1 passed`, with no skip.
- `daemon stayed idle during the job` and `exit=0`.

- [ ] **Step 16: Run port equals spike**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/porteq.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py .venv/bin/python $G/porteq.py'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/porteq.txt 540 40'
```

Expected:
- 31 dispatch lines, each ending in eight `=`.
- `31 dispatches x T = 1..8`, then `PORT == SPIKE`.
- `daemon stayed idle during the job` and `exit=0`.

Any `X` fails the gate: the port then does not carry the spike's parity and quality evidence.

Then run Task 2's synthetic check on the M5, the same way. `porteq.py` sees only the variant each real shape chooses; this one forces each variant a case's K allows (plain always, staged and staged with presum where K % 1024 = 0) at T = 1..8 and 1–4 linears. `SPIKE_DIR` points it at the M5's copy of the spike, since its default is this Mac's path:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/port_synth.txt env GPU_LABEL=proj-gate SPIKE_DIR=$L/m5/projection-kernels python3 $L/gpu_run_m5.py .venv/bin/python $G/port_vs_spike.py'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/port_synth.txt 540 12'
```

Expected:
- Seven lines ending `checked` and no `DIFFERS:` line.
- `PORT == SPIKE, bitwise`.
- `daemon stayed idle during the job` and `exit=0`.

A `DIFFERS:` line fails the gate as an `X` does.

- [ ] **Step 17: Run the kernel timings**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/ktime.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py .venv/bin/python $G/ktime.py'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/ktime.txt 540 16'
```

Repeat the wait until `DONE`. Expected:
- 96 cell rows and 8 forward rows, each ending `ok`.
- `ktime PASS`.
- `daemon stayed idle during the job`.

The re-run for a forward-only failure, as the task's head describes it, uses the same command with `$G/ktime.py forward` and the log `ktime-forward.txt`. If the re-run passes, the gate reads its forward table in place of the first, and the PR body carries both.

- [ ] **Step 18: Run the replay chains**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/chains.txt zsh $G/chains.sh'
```

This runs 14 runs: about 2.3 hours for chain A and 3.5 hours for chain B. To follow progress:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate/logs; cat $G/A_DONE.part $G/A_DONE $G/B_DONE.part $G/B_DONE 2>/dev/null; tail -n 2 "$(ls -t $G/*.log | head -1)"'
```

**Check the first branch load early.** About 25 minutes in, `br-s1` has loaded. Read its load line and the shim's timing:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'R=~/code/personal/sous-spikes/p148/gate/runs/br-s1; grep -h model_load $R/daemon.out | sed "s/^.*INFO //"; grep -h "gate148: projkernel.enable" $R/daemon.out'
```

Expected:
- The load line ends `attention_tile=active attention_tile_splits=20 attention_tile_probe_s=<s> projection_kernel=active projection_kernel_probe_s=<s> draft_block=5`.
- The timing line reads `gate148: projkernel.enable seconds=<under 1> state=active reason=None`.

If either reads anything else, stop the chain and report:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'pkill -TERM -f "[p]148/gate/chains.sh"; pkill -TERM -f "[p]148/gate/run_gate.sh"; pkill -TERM -f "[p]148/gate/gate132.py"'
```

gate132.py unwinds on TERM: it stops its daemon's process group and deletes that run's forks. The bracket in each pattern keeps `pkill` from matching the shell that runs it.

The chains are done when `B_DONE` lists eight `exit=0` lines:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/off-g256.log 540 6'
```

A `port 8384 still busy`, `process left` or `chain A incomplete` line stops the chain. Report it before going on.

- [ ] **Step 19: Check that every replay run counts**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate; for l in main-s1 br-s1 br-s2 main-s2 arms-1 arms-2 main-nd1 br-nd1 br-nd2 main-nd2 t-d-g t-nd-g main-g256 off-g256; do f=$G/logs/$l.log; printf "%-10s idle=%s gone=%s\n" $l "$(grep -c "daemon stayed idle during the job" $f)" "$(grep -c "process group gone; port 8384 free" $f)"; done | tee $G/logs/validity.txt; echo "refused: $(cat $G/runs/*/daemon.out | grep -c "gate148: REFUSED")"'
```

Expected:
- `idle=1 gone=1` on all 14 lines.
- `refused: 0`: both arms runs met the shim's preconditions.

A run counts only if the lock reported the daemon idle throughout and the harness logged the child's process group gone and its port free. Re-run any other run under a new label:
- append `b` to its label;
- run it with `run_gate.sh` and a done-file of its own;
- use the new label wherever the old one appears in Step 23.

Its rounds and trace files take the new label too, so they start empty.

- [ ] **Step 20: Run the suite: 8 tasks × 6 prompts per tree, in two GPU-lock jobs**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate && mkdir -p $G/suite/home/.sous && cp $G/suite/config.toml $G/suite/home/.sous/config.toml'
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/suite-a.log env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py zsh $G/run_suite.sh 5000 5002'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/suite-a.log 540 14'
```

Repeat the wait until `DONE`, about 25 minutes. Then run the second job the same way, with log `suite-b.log` and seeds `5003 5005`:

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/p148/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/suite-b.log env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py zsh $G/run_suite.sh 5003 5005'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/suite-b.log 540 14'
```

Expected in both logs:
- Six `== <tree> seed <s> done` lines, each saying `released=True error=None`.
- The branch's lines read `block=5 proj={'state': 'active', ...}`, and main's read `block=4 proj={'state': 'absent'}`.
- `daemon stayed idle during the job` and `exit=0`.

- [ ] **Step 21: Run `api_smoke.py` on the M5**

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-proj && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/p148/gate/logs/smoke.txt env GPU_LABEL=proj-gate python3 $L/gpu_run_m5.py uv run python scripts/api_smoke.py'
$G/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/p148/gate/logs/smoke.txt 540 6'
```

Expected:
- `PASS: a tool_use block came back`.
- `daemon stayed idle during the job`.
- `exit=0`.

- [ ] **Step 22: Run the M2 (this Mac)**

Make the harness copy and a main and a branch worktree, each with its own `.venv`:

```bash
M2=~/code/sous-spikes/p148/m2
mkdir -p $M2
cp ~/code/sous-spikes/tile-132/m2/gate143_m2.py ~/code/sous-spikes/tile-132/m2/bodies-small.json $M2/
git fetch -q origin
git worktree add --detach $M2/main origin/main
git worktree add --detach $M2/branch origin/feat/projection-kernel
uv sync -q --directory $M2/main
uv sync -q --directory $M2/branch
```

Run #143's M2 task set greedy on each tree: the 9B and its DFlash drafter at max_tokens 512, about 7 minutes each. Neither run sets a block, so each tree takes its own default:

```bash
M2=~/code/sous-spikes/p148/m2
GATE_BODIES=$M2/bodies-small.json GATE_MODEL_TOML='id = "mlx-community/Qwen3.5-9B-MLX-4bit"\nspeculative_draft_id = "z-lab/Qwen3.5-9B-DFlash"\n' GPU_LABEL=proj-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py python3 $M2/gate143_m2.py main-d-g $M2/main 0.0 512 > $M2/main-d-g.log 2>&1
GATE_BODIES=$M2/bodies-small.json GATE_MODEL_TOML='id = "mlx-community/Qwen3.5-9B-MLX-4bit"\nspeculative_draft_id = "z-lab/Qwen3.5-9B-DFlash"\n' GPU_LABEL=proj-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py python3 $M2/gate143_m2.py br-d-g $M2/branch 0.0 512 > $M2/br-d-g.log 2>&1
$M2/branch/.venv/bin/python ~/code/sous-spikes/p148/gate/judge.py m2 $M2/runs main-d-g br-d-g | tee $M2/m2.txt
```

Expected:
- `HTTP errors: 0`.
- `greedy replies identical on 12/12 turns`.
- The branch load line ends `projection_kernel=unavailable draft_block=4`.
- `branch projection_kernel:` has `'state': 'unavailable'` and the tile's reason, `macOS 15.5 < 26.2` on this Mac.
- `branch turns over 3 drafted per round: none`.
- `m2: PASS`.

Run `api_smoke.py` in the branch tree:

```bash
M2=~/code/sous-spikes/p148/m2
GPU_LABEL=proj-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py uv run --directory $M2/branch python scripts/api_smoke.py 2>&1 | tail -4 | tee $M2/smoke.txt
```

Expected: `PASS: a tool_use block came back`.

Record each tree's non-model suite here, for Step 25's check that CI ran the kernel tests:

```bash
M2=~/code/sous-spikes/p148/m2
for t in main branch; do GPU_LABEL=proj-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py uv run --directory $M2/$t pytest -m "not model" -rs > $M2/notmodel-$t.txt 2>&1; grep -E "[0-9]+ passed" $M2/notmodel-$t.txt | tail -1; done
grep -c -E "^SKIPPED .*tests/test_projkernel" $M2/notmodel-branch.txt
```

Expected:
- Two summary lines with passes and skips only.
- A count of `0`: the kernel compiles and runs on this Mac's GPU, so none of its tests skips here.

- [ ] **Step 23: Judge the gate**

On the M5, run every analysis. They are CPU-only; `analyze132.py equiv` needs the 27B's tokenizer, which is why they run there.

```bash
G=~/code/sous-spikes/p148/gate
$G/m5.sh m5pro-lan 'zsh ~/code/personal/sous-spikes/p148/gate/judge_all.sh'
```

Expected:
- `loadtime: PASS`, `load: PASS`, `path: PASS`, `arms: PASS`, `abba: PASS` twice and `off: PASS`.
- `equivalence t-d-g vs t-nd-g: 69/69 identical; cache histories equal on 69/69`.
- `equivalence off-g256 vs main-g256: 64/64 identical; cache histories equal on 64/64`.
- `PORT == SPIKE`, `PORT == SPIKE, bitwise` and `ktime PASS`.
- `gap: 0 differing turns of 69`, the last line of `greedy.txt`, and `... -> no harm detected`.

Pull the results back to this Mac, with the suite's archived runs. The run homes stay behind:

```bash
G=~/code/sous-spikes/p148/gate
rsync -a -e $G/m5.sh --exclude 'home/' --exclude 'suite/w*/' m5pro-lan:code/personal/sous-spikes/p148/gate/ ~/code/sous-spikes/p148/gate/
```

**Compare the suite's failing diffs**, as the spike did, for every branch run under `Imperfect runs:` in `logs/suite.md`:
1. Open the branch run's `transcript.jsonl` and `project/` in `~/code/sous-spikes/p148/gate/suite/runs/branch-<seed>/arm/<task>-1/`. The line's own path is the M5's copy of the same directory.
2. Open main's run of the same task and seed, in `suite/runs/main-<seed>/arm/<task>-1/`.
3. Write one line per run into `~/code/sous-spikes/p148/gate/logs/suite_failures.md`: the task, the seed, the branch's failure mode, and main's grade and failure mode on the same prompt. Include no path.
4. If the branch has no imperfect run, write `No imperfect branch run.`

The gate passes only if every criterion below holds. Each is quoted from the spec, followed by the output that decides it.

- **Load.** "The branch's load line shows `projection_kernel=active draft_block=5`. Warm load: warm-up plus probe adds at most 1 s. Cold load: the first load after install (empty Metal cache) is recorded and must add at most 10 s."
  - `logs/j_load.txt` ends `load: PASS`. Every branch run's load line carries the expected state and block. Every active run carries `projection_kernel_probe_s=`, and its `projkernel.enable` took at most 1 s: these are warm loads, after Step 13 compiled the sources.
  - `logs/j_loadtime.txt` ends `loadtime: PASS`, with the cold (nonce) addition ≤ 10 s, the warm one ≤ 1 s and the first load after install ≤ 10 s. All three are recorded.
- **Port equals spike.** "On the 27B's real weights, for every dispatch key at T = 1..8, the in-tree kernel's output is `array_equal` to the spike's `verify_proj`."
  - Decided by `logs/porteq.txt`: `31 dispatches x T = 1..8` and `PORT == SPIKE`.
  - And by `logs/port_synth.txt`, Task 2's check with each variant forced on synthetic weights: no `DIFFERS:` line and `PORT == SPIKE, bitwise`.
- **Path.** "`declined` = 0 in every branch run. `verify_kernel` > 0 in every branch run with a drafter, and `plain_kernel` > 0 in every branch run. A `projection_kernel = false` run makes 0 kernel calls and shows `draft_block=4`."
  - Decided by `logs/j_path.txt`: nine `ok` lines and `path: PASS`.
  - The `off-g256` line has state `off` and every count 0. Its `draft_block=4` is in `logs/j_load.txt`.
- **Speed, primary.** "Pass: 5k/4v ≥ 1.05 with the CI lower bound > 1.00. A point in [1.00, 1.05) with the CI above 1.00 is not a pass; it goes back to the maintainer with the numbers."
  - Decided by `logs/j_arms.txt`: `arms: PASS`.
  - `logs/arms.txt` is analyze148's full report over both seeds: rounds per arm, the stratified table and `PRIMARY`.
  - `arms: TO THE MAINTAINER` stops the gate with `logs/arms.txt`.
- **Speed, secondary.** "Pooled `out_tokens / decode_s` branch/main is reported, and must not be below 1.00." "Prefill tok/s is within ±3% of main in the same ABBA, and peak `memory_gb` within ±0.2 GB." "Greedy is reported on identical-output turns only."
  - Decided by `logs/j_abba.txt`: the `all` row's ratio ≥ 1.00, prefill within ±3%, peak memory within ±0.2 GB, and `abba: PASS`.
  - `logs/decode_runs.md` and `logs/decode_pair.md` are reported.
  - `logs/greedy.txt` is reported, not judged.
- **Drafter off.** "`speculative_draft_id = ""`, sampled, ABBA on the first 16 of the 64 turns in gate132's order: branch/main ≥ 0.75 pooled."
  - The first 16 in gate132's order are a160cc42's turns 0–15, the pick `agent-a160cc42252609969.jsonl:16`.
  - Decided by `logs/j_nodraft.txt`: the ratio ≥ 0.75 and `abba: PASS`.
- **Parity.** "The branch with the drafter against the branch without it, greedy, max_tokens 256, on all 64 turns plus the a6443762 continuation at ≥ 128K. Every reply and stop reason must be identical, and the prompt-cache history must be equal on every pair. One divergence fails the gate. Before ruling, the first differing token's top-2 logit gap is read."
  - Decided by `logs/parity.txt`: `69/69 identical; cache histories equal on 69/69`, with no further line.
  - The 69 turns include a6443762@176, @239, @300 and @365, which run from 131K to 229K tokens.
  - On any divergence, `logs/gap.txt` gives each differing turn's first differing token, the drafter-off run's top two logprobs there and the drafter-on token. Report it with `logs/parity.txt`: the spec's argmax residual is a bf16 tie at that token, and anything else is a kernel fault.
- **Off equals main.** "`projection_kernel = false`, greedy, same settings: reply-identical to main on the 64 turns, with equal cache histories." "`draft_block=4` on the load line, and at most 3 drafted tokens per round on every turn line."
  - Decided by `logs/off_main.txt`: `64/64 identical; cache histories equal on 64/64`.
  - And by `logs/j_off.txt`: `draft_block=4`, configured block `[4]`, the most drafted in one round ≤ 3, no turn line over 3 per round, and `off: PASS`.
- **Suite.** "8 tasks × 6 prompts, branch default against main default ... no harm at a one-sided exact McNemar p < 0.10 on perfect runs. Failing diffs are compared, as the spike did."
  - Decided by `logs/suite.md`:
    - `main: 48 runs` with blocks `['4']`, and `branch: 48 runs` with blocks `['5']`;
    - `branch runs on the kernel ...: 48/48; main runs with kernel calls: 0`;
    - the McNemar line ends `no harm detected`.
  - `logs/suite_failures.md` is written.
- **Kernel timings.** "Each cell's band is the larger of ±3% and the A/A spread. The per-forward sum (#133: 75–79 ms) must land within ±2%."
  - Decided by `logs/ktime.txt`: 96 cells `ok`, 8 forward rows `ok` and `ktime PASS`. After a forward re-run, the forward rows come from `logs/ktime-forward.txt`, which must read `ktime PASS`.
- **M2 Air.** "The load line reads `projection_kernel=unavailable` with the tile's reason and `draft_block=4`; greedy replies on the #143 M2 task set are identical to main's; drafted tokens per round are at most 3."
  - Decided by `~/code/sous-spikes/p148/m2/m2.txt`: `m2: PASS`.
- **`scripts/api_smoke.py`** "passes on both Macs."
  - Decided by `logs/smoke.txt` and `~/code/sous-spikes/p148/m2/smoke.txt`: both read `PASS: a tool_use block came back`.

If any criterion fails, stop and report the numbers and the files above. Open no PR.

- [ ] **Step 24: Record the gate and open the PR**

**Decode figures.** Only after a passing gate.

The spec changes the README's decode figures only if the gate's numbers justify it. The one gate figure that measures decode as shipped against main at the README's context is the secondary ABBA's `32-64K` bucket, so it decides. `logs/figures.txt` says either that its CI includes 1.00, in which case the README stays, or what the two figures become:
- `README.md` (line 56 when this plan was written): in `about 23.5 tok/s at 57K`, replace 23.5 with the printed value, when it differs.
- `README.md` (line 148): in `1.4–1.6x at 51–57K`, replace both bounds with the printed band, when it differs. Keep the README's en dash.
- Leave `about 31 tok/s with 2K tokens of context` and `1.2–1.5x at short context` alone. No gate run measures short context.

If either edit changed a figure, commit and push:

```bash
git add README.md
git commit -m "docs: update the README's 57K decode figures with the projection kernel's gain" -m "The merge gate measured decode on the M5 Pro's 64 real subagent turns with
the branch against main, each at its default, sampled, in ABBA order, and
the 32-64K bucket's ratio cleared 1.00. The 57K figure and the gap to
Splash at 51-57K scale with that ratio; the 2K figure and the short-context
gap stay, since no gate run measured short context.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin feat/projection-kernel
```

**The PR body.** Write its fixed head:

```bash
cat > ~/code/sous-spikes/p148/gate/pr-head.md <<'EOF'
## What and why

The default 27B's small-row quantized projections now run through a simdgroup MMA kernel ported
from the Splash spike (`src/sous/engine/projkernel.py`,
`src/sous/engine/kernels/projection_mma.{h,metal}`): every DFlash verify forward, and every target
call of up to 8 rows outside the verifier, decode included. mlx-vlm's exact verifier keeps each
row bitwise equal to one-row decode, but its projection cost climbs with rows; the kernel's stays
nearly flat from 1 to 8 rows, which makes verify block 5 pay. Its rows are bitwise independent of
the row count and of which linears share a dispatch, so a verify row equals the decode of the same
token, and greedy output with the DFlash2 drafter stays identical to output without it.

- On by default through `[model].projection_kernel`; `false` runs today's paths.
- Active only where the attention tile is active (the M5 Pro), on the default dense qwen3_5 4-bit
  model with no drafter or a DFlash one; `unavailable` with its reason everywhere else, where
  nothing changes.
- An unset `[model].speculative_block_size` now resolves to 5 where the kernel is active and to 4
  elsewhere; an explicit value always wins, and 0 keeps its meaning.
- The cost: decode without a drafter must take the kernel too, for greedy parity, and runs slower
  where the kernel is active (measured below).
- The model-load line gains `projection_kernel=` (with `projection_kernel_probe_s=` when active)
  and `draft_block=`, and `/sous/status` a `projection_kernel` block and `draft_block`.
- `sous tune` keys suite rows by the setting, pins each block arm's block explicitly, and marks
  the current arm by the block the engine would resolve.
- THIRD_PARTY_NOTICES.md gains a Splash section (Apache-2.0).

**Fork store.** Landing this cold-starts every on-disk fork store once, including stores with the
kernel off: the fork key gains `proj`, and `projkernel.py` joins the engine epoch.

Spec: `docs/superpowers/specs/2026-10-03-projection-kernel-design.md`. Plan:
`docs/superpowers/plans/2026-10-03-projection-kernel.md`.

Issue: #148. Still open and out of scope: block 6, and the suite's per-arm project paths (#149).
EOF
```

Assemble the body from the head and the gate's outputs:
- The suite's `Imperfect runs:` list is left out, because it carries archive paths.
- analyze148's install lines are left out too, because they carry the rounds files' paths.

```bash
G=~/code/sous-spikes/p148/gate
M2=~/code/sous-spikes/p148/m2
{
  cat $G/pr-head.md
  printf '\n## Merge gate (M5 Pro, 20 GPU cores)\n\n### Load\n\n~~~\n'; cat $G/logs/load_lines.txt; grep -h -E '^loadtime|^added by' $G/logs/j_loadtime.txt; printf '~~~\n'
  printf '\n### Port equals spike: real 27B weights, every dispatch key, T = 1..8\n\n~~~\n'; grep -v '^\[gpu_run' $G/logs/porteq.txt; grep -h -E '^PORT|^DIFFERS' $G/logs/port_synth.txt | sed 's/^/synthetic weights, each variant forced: /'; printf '~~~\n'
  printf '\n### Path (process totals per branch run)\n\n~~~\n'; cat $G/logs/j_path.txt; printf '~~~\n'
  printf '\n### Speed, primary: 4v against 5k drawn per round in one branch process, two sampled seeds\n\n~~~\n'; sed -n '/^PRIMARY/,$p' $G/logs/arms.txt; cat $G/logs/j_arms.txt; printf '~~~\n'
  printf '\n### Speed, secondary: sampled ABBA on the 64 turns, each tree at its default\n\n'; cat $G/logs/decode_runs.md $G/logs/decode_pair.md
  printf '\n~~~\n'; cat $G/logs/j_abba.txt $G/logs/greedy.txt; printf '~~~\n'
  printf '\n### Drafter off, sampled ABBA on the first 16 turns\n\n'; cat $G/logs/nodraft_pair.md; printf '\n~~~\n'; cat $G/logs/j_nodraft.txt; printf '~~~\n'
  printf '\n### Parity: drafter on against off, greedy, 69 turns to 228,630 tokens\n\n~~~\n'; cat $G/logs/parity.txt; tail -n 1 $G/logs/gap.txt; printf '~~~\n'
  printf '\n### projection_kernel = false against main, greedy, 64 turns\n\n~~~\n'; cat $G/logs/off_main.txt $G/logs/j_off.txt; printf '~~~\n'
  printf '\n### Tool-loop suite: 8 tasks x 6 prompts, paired by prompt\n\n~~~\n'; sed '/^Imperfect runs:/,$d' $G/logs/suite.md; printf '~~~\n\n'; cat $G/logs/suite_failures.md
  printf '\n### Kernel timings: in-tree against the spike, ms per call and per forward\n\n'; grep -h -E '^\||^ktime' $G/logs/ktime.txt $(ls $G/logs/ktime-forward.txt 2>/dev/null)
  printf '\n### M2 Air (9B + DFlash, the #143 task set)\n\n~~~\n'; cat $M2/m2.txt; printf '~~~\n'
  printf '\n`scripts/api_smoke.py`: M5 `%s`, M2 `%s`.\n' "$(grep -h '^PASS\|^FAIL' $G/logs/smoke.txt)" "$(grep -h '^PASS\|^FAIL' $M2/smoke.txt)"
  printf '\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n'
} > $G/pr-body.md
if grep -n -e /Users/ -e /home/ -e "$(id -un)" -e "$($G/m5.sh m5pro-lan id -un)" -e '\.ts\.net' $G/pr-body.md; then echo "strip these lines before opening the PR"; fi
```

Expected: the privacy check prints nothing. The body must carry no person's name, no home path and no machine name.

Open the PR:

```bash
gh pr create --base main --head feat/projection-kernel --title "feat(engine): run verify and decode projections on a small-row simdgroup kernel" --body-file ~/code/sous-spikes/p148/gate/pr-body.md
```

- [ ] **Step 25: Confirm that CI ran the kernel tests**

CI runs `pytest -m "not model"` under `-q` with no `-rs`, so its log shows only the summary counts.
- **Why counts suffice.** The kernel tests skip wherever `projkernel.kernel_available()` fails, and on this Mac none of them skips (Step 22). The tests both trees share skip alike on both runs of one machine, so they cancel out of a difference. A test the branch adds that skips on CI adds to CI's difference beyond this Mac's only if it runs here and not there, and a kernel test on a GPU where the lane map differs is the case the spec names.
- **The check.** The PR's CI skip count minus main's latest green CI run's, which Step 2 rebased onto, must be no larger than this Mac's branch-minus-main difference.

```bash
gh pr checks feat/projection-kernel --watch --interval 60
pr=$(gh run list --branch feat/projection-kernel --workflow CI --limit 1 --json databaseId -q '.[0].databaseId')
base=$(gh run list --branch main --workflow CI --status success --limit 1 --json databaseId -q '.[0].databaseId')
for r in $pr $base; do gh run view $r --log | grep -E '[0-9]+ passed' | tail -n 1; done
grep -h -E '[0-9]+ passed' ~/code/sous-spikes/p148/m2/notmodel-branch.txt ~/code/sous-spikes/p148/m2/notmodel-main.txt | tail -n 2
```

Expected: every check passes, and the PR's skipped count minus main's on CI is at most the branch's minus main's on this Mac.

If CI's difference is larger, the kernel tests skipped on CI's GPU: the lane-map risk the spec names. Report the four summary lines to the maintainer before anything else.

- [ ] **Step 26: Clean up**

Delete any forks the runs left behind. gate132.py removes its own when it exits, but gate143_m2.py does not:

```bash
~/code/sous-spikes/p148/gate/m5.sh m5pro-lan 'find ~/code/personal/sous-spikes/p148/gate/runs ~/code/personal/sous-spikes/p148/gate/suite/home -path "*/.sous/forks" -type d -prune -print -exec rm -rf {} +'
find ~/code/sous-spikes/p148/m2/runs -path '*/home/.sous/forks' -type d -prune -print -exec rm -rf {} +
```

Expected: each `find` prints the fork directories it removed, or nothing.

---

---

## Self-Review Notes

**Spec coverage**
- **What the measurements established:** grounded in the spike. Task 11 re-measures every claim the merge gate depends on.
- **Decisions:**
  - design (b): Tasks 3 (both hooks) and 4 (gates);
  - on by default with an opt-out: Task 6;
  - rides on the tile: Task 4, gate 3;
  - engine-resolved block default: Tasks 6 (engine) and 8 (`sous tune`);
  - the full kernel family: Task 2;
  - two tag-gated hooks: Task 3;
  - the greedy drafter's argmax left stock: Task 3, which hooks nothing else;
  - lives in sous with runtime gates: Task 4.
- **Design:**
  - Files and Kernel: Tasks 1 and 2.
  - Routing: Task 3.
  - Guards: Task 4.
  - Block default: Tasks 6 and 8.
  - Wiring and visibility: Tasks 6, 7 and 8.
  - Fork store: Task 7.
  - Notices: Task 9.
- **Testing:**
  - CI: Tasks 1–8.
  - Contract tests: Task 5.
  - Model tests: Task 10.
- **Merge gate:** Task 11, every pass criterion quoted with the output that decides it.
- **Documentation:** Task 9.
- **Out of scope:** untouched.

**Deliberate refinements of the spec's wording**
- **Block 0 with DFlash2:** the drafter's own policy caps at `min(5, 8) = 5`, so with the default drafter block 0 never verifies more than 8 rows. The split still serves other DFlash drafters, and the model test exercises it at block 12, set on the engine.
- **`ManagedEngine` defaults:** `projection_kernel_status` and `draft_block` default to None, so fakes stay silent on the load line. The LM backend reports `off`.
- **`static_reason` and quantization:** it also checks the checkpoint's quantization, including per-layer overrides. Otherwise mxfp4 and mixed-precision checkpoints would get their current tune arm marked at block 5.
- **`sous tune` resolution:**
  - It resolves the user's effective block once, in `main`, through `arms.resolve_block(user, checkpoints)`, and `Checkpoint.config` carries the raw configs `describe` already fetched.
  - `decide` also writes the block when the winner changes model or drafter while the user's block is unset.
  - Suite rows written before this change read `projection_kernel = False`, because none of their runs used the kernel.
- **`probe_seconds`** covers the warm-up as well as the probe.
- **The notice** cites incoai/splash at f43509a and paperniuk's d81795d and d2f902e. a450058 contributes nothing the port keeps.

**Placeholders:** none. Every code step carries complete code, and Tasks 1–9 were executed as written.

**Type and name consistency**
- The dry run proved Tasks 1–9 against each other: one `_launch` (Task 3's helper is `_run_mma`), and `rand_linear` (a triple) beside `rand_module` (a module).
- `tiny_quantized_model` has one definition, whose body Task 4 replaces.
- Task 11 was reviewed against the dry-run code's real names: `projkernel` functions, `calls` keys, the status dict, load-line tokens, `VLMEngine` kwargs and properties, `Arm`/`SuiteRun` fields and the `proj` fork-key field.

**Measured compile cost.** On the M5 Pro, the spike kernels' 96 dispatch pipelines took 3.45 s cold (Metal's cache busted with a nonce) and 0.02 s warm. Eager warm-up at load is within the spec's bounds.

**Dry run (2026-10-03).** Tasks 1–9 were executed as written in a scratch worktree on a4567ad, on the maintainer's M2 (Apple M2, macOS 15.5).
- **Plan fixes:** 14 in all, folded in above.
  - Task 3 redefined Task 2's `_launch`, which silently skipped every kernel test.
  - Task 4 duplicated `tiny_quantized_model`; Task 3's version was unusable at K = 128.
  - Task 8 had a stale `runner.py` anchor.
  - Three provenance comments were wrong.
  - The rest were small: one helper type, one append location, one import block, one duplicated constant, and contract wording.
- **Results:**
  - Every Expected line then matched.
  - The kernel tests ran on the M2's GPU, not skipped.
  - `port_vs_spike.py` printed `PORT == SPIKE, bitwise` for every variant, 1–8 rows and 1–4 linears.
- **Full CI set** after each task:

  | after task | passed |
  |---|---|
  | 1 | 1772 |
  | 2 | 1839 |
  | 3 | 1886 |
  | 4 | 1955 |
  | 5 | 1971 |
  | 6 | 1993 |
  | 7 | 2004 |
  | 8 | 2022 |
  | 9 | 2022 |

  Every run had 39 skipped (the tile's `nax` tests) and 41 deselected (`model`).
- **Tasks 10–11** run only on the M5 Pro. They were reviewed against the dry-run code, and every embedded script passes `py_compile`, `bash -n` or `zsh -n`. Task 10's test skips on the M2 with `macOS 15.5 < 26.2`.
