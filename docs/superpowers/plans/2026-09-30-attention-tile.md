# Tensor-Unit Attention Tile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run decode's one-row attention and the exact verifier's attention on the M5's tensor units, through one GQA-packed MPP tile with a key partition that steps with context, on by default through `[model].attention_tile` and keeping greedy parity with the drafter on or off.

**Architecture:** A new `src/sous/engine/tileattn.py` (schedule, kernel entries, hooks, probe, gates, `enable()`) over a new `kernels/attention_tile.metal`, with the tensor-unit platform rule shared through a new `nax.py` and the split target read by a new `gpucores.py`. Decode reaches it through a replacement of the one module global `Qwen3_5Attention.__call__` calls after its projections; verify rows reach it through a tile branch in verifyattn's existing verifier wrapper. Engine, config, fork-key and `sous tune` wiring make it visible, switchable and correctly keyed.

**Tech Stack:** Python 3.14, mlx 0.32.2, mlx-vlm 0.7.2, mlx-lm 0.31.3 (the lock), Metal (MPP `matmul2d` via `mx.fast.metal_kernel`), ctypes IOKit, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-30-attention-tile-design.md`. Read it before starting any task.

**Branch:** `feat/attention-tile`, based on e9aa76b.

## Global Constraints

- **Schedule:** `N0 = 1536`; `chunk_for(n, S)` is 0 (stock) below N0, else `32 * ceil(n / (32 * S))` with `GRAN = 32`; each step is 32·S keys wide and holds at most S splits.
- **Schedule at S = 20:** 408 step boundaries up to 262,144, the first four [1536, 1921, 2561, 3201]; exactly 20 splits for every n ≥ 11,553.
- **Runs:** `verify_plan(prefix, t, splits)` cuts a T-row verify into contiguous runs that never cross N0 or a step boundary and never exceed `MAX_RUN = 8` rows, so any T (T > 8 included) is served.
- **Stock runs:** runs below N0 go to `verifyattn.grouped_attention` (T = 1 and 2 included) with tileattn's own `_arch()` (`mx.device_info()["architecture"]`), never `verifyattn._ARCH`.
- **Split target S:** the GPU core count, read once per process by `gpucores.read()` (IOKit through ctypes, fallback `/usr/sbin/ioreg -a -r -c AGXAccelerator -d 1` parsed with plistlib, absolute path, 5 s timeout); exactly one AGXAccelerator service, an int core count > 0, and `model == mx.device_info()["device_name"]`.
- **Measured splits:** the tile is active only for S in `MEASURED_SPLITS = frozenset({20})`; any other value is `unavailable` with that reason.
- **S in CI:** every CI test that asserts a tile state monkeypatches S (`tileattn.gpucores.read`, or the flag directly) and never reads IOKit.
- **Kernel source:** one file, `src/sous/engine/kernels/attention_tile.metal`: the split body, a line exactly `// REDUCE`, then the reduce body.
- **Kernel specialisation:** bf16, 24 query / 4 KV heads (GQA 6), head dim 256, scale 1/16 (`SCALE = 0.0625`, compared with `==`), fp32 probabilities (`typedef float PE`); lane layout from `nax_coord` in `kernels/nax.h`.
- **Kernel codegen:** all four local cooperative-tensor types wrapped in `metal::remove_addrspace_t`; `sg / 2` and `sg - rg * 2` stay exactly as written (the equivalent shifts gave identical output but ran 1.4x slower at T = 3 and 8).
- **Kernel build:** both kernels through `int8prefill._kernel(...)` with `compile_options={"math_mode": "safe"}` stated explicitly.
- **Split kernel:** `sous_attention_tile_split`, inputs q, k, v, params; outputs part, stats; header `common.h` + `nax.h`; `ensure_row_contiguous=False` (the source reads `q_strides`/`k_strides`/`v_strides`).
- **Reduce kernel:** `sous_attention_tile_reduce`, inputs part, stats, params; output out; header `common.h`.
- **`tile(queries, keys, values, n, chunk)`:** returns a contiguous `[1, T, 24, 256]` bf16 array; row t sees keys [0, n − T + t]; every K/V read is clamped to n − 1.
- **`decode()` / `verify()`:** return `[1, 24, T, 256]` and look up the module-global `tile` at call time, so tests can swap in the stand-in.
- **`tileattn.in_scope(q, k, v, scale)`:** the one post-projection predicate both hooks use: mx arrays, batch 1, 24/4/256 heads, q/k/v all bf16, `scale == 1/16`, K and V the same length ≥ T.
- **Decode hook install:** only the first `enable()` that reaches `active` replaces `mlx_vlm.models.qwen3_5.language.scaled_dot_product_attention`, with a `functools.wraps` wrapper that forwards `*args, **kwargs` unchanged.
- **Decode hook scope:** serves a call only when the flag is set, mask is None, sinks is None, the query is one row, `in_scope` holds, n ≥ N0, and `type(cache) is KVCache` with no `bits`, no `left_padding` and no `_qwen3_5_decode_left_padding`; every other call goes to the original with the same arguments.
- **Decode hook in tests:** installed through `monkeypatch`, so the global is restored after each test; a process that never activates the tile runs mlx-vlm's function untouched.
- **Verify hook, before the projections:** verifyattn's `_in_scope` decides; while the tile is active it takes T ≥ 1 with no upper bound, mask None at T = 1 and `"causal"` above, bf16 only.
- **Verify hook, after the projections:** rows go to `tileattn.serve_verify()` when `_post_ok` and `in_scope` both hold; otherwise the wrapper computes today's path for that T (T ≥ 3 `grouped_attention`, or `_row_loop` when `_post_ok` fails; T = 2 one bool-masked call; T = 1 one unmasked call) and never re-enters the original method.
- **Verify hook, tile inactive:** verifyattn behaves exactly as it does today.
- **Flag:** `tileattn._active_splits` is a plain int, the active S or 0, never a model reference; the first statement of every `enable()` clears it, only an `active` result sets it, `tileattn.clear()` clears it, and `VLMEngine.unload()` calls `clear()`.
- **Guards, in order, each failure `unavailable` with its reason:** (1) `enabled` false → `off`; (2) `nax.platform_reason()` (macOS ≥ 26.2, GPU generation ≥ 17, ≥ 18 for 'p' parts), early so an M2 reports the NAX reason whatever its model; (3) model type `qwen3_5` and every full-attention module at 24/4/256, scale 1/16, bf16; (4) no drafter, or a kind in `{"dflash"}` with `verify_status["state"] == "active"`; (5) `verifyattn._gate()` (mlx in `VALIDATED_MLX = frozenset({"0.32.2"})`, `MLX_SDPA_BLOCKS` unset or 0, every verifyattn source hash); (6) the three decode-source hashes; (7) the split target; (8) the probe, on the `sous-model-load` thread.
- **Decode-source hashes** (`hashlib.sha256(inspect.getsource(obj).encode("utf-8"))` in `mlx_vlm.models.qwen3_5.language`, checked on mlx-vlm 0.7.2):
  - `Qwen3_5Attention.__call__` b051c66057acc90f4ed78691f90254685cfdb20d5814b0d58beec4d2bb34c184
  - `Qwen3_5Model.__call__` 9dfc2048db0d818595bcdeb306a59e80cf1217331784fe7361c92a2dbf13ab65
  - `LanguageModel.__call__` af7f3f6427c2e59ab5266798b7518eaf026c3f7f749dd0962e38a6acece49cc9
- **Probe constants:** `PROBE_FAR_KEYS = 57_000` (the far mark is the first step boundary ≥ it, 57,601 at S = 20; CI tests monkeypatch it to 3_000); `REL_RMS_BOUND = 1e-2` (finite and relative RMS ≤ the bound, written so NaN fails).
- **Probe hygiene:** inputs built with GPU ops only (no float64) and evaluated before the first launch; every T = 1..8 compiled first; the flag left at 0 and `tileattn.calls` and the language global as it found them; its arrays deleted and `mx.clear_cache()` called before returning; a compile failure logs the compiler's full report and keeps its first `error:` line as the reason.
- **Failure policy:** `enable()` never raises and a model load never fails because of it; every refusal is exactly one `warnings.warn` (text starting `sous: attention tile unavailable (`, `stacklevel=3`, reason cut to one line by `int8prefill._error_line`) plus `state: unavailable`.
- **Status dict:** `{"state": "active"|"unavailable"|"off", "reason": str|None, "splits": int|None, "probe_seconds": float|None}`; `splits` only when active; `probe_seconds` rounded to 0.01 whenever the probe ran; the LM backend sets exactly `{"state": "off", "reason": None}`.
- **Config:** `[model].attention_tile` → `SousConfig.attention_tile: bool = True`, listed in `_KNOWN["model"]`; a non-bool warns `sous config: [model].attention_tile {value!r} must be true or false; using false` and means false; read once at daemon start.
- **Engine defaults:** `VLMEngine(..., attention_tile: bool = False)` and `base._default_factory(..., attention_tile: bool = False)` default off, as `int8_prefill` does; the config's `True` reaches the daemon and `sous tune` through `default_engine_factory(config)`; `LMEngine` takes no tile argument.
- **Visibility:** the model-load line appends ` attention_tile=<state>` after the draft_context tokens, plus ` attention_tile_splits=<S> attention_tile_probe_s=<x>` only when active; `EngineManager.status()` gains `attention_tile: {state, reason}`; nothing is logged per turn; `tileattn.calls` counts every path (keys pinned in Task 5).
- **Fork store:** `tileattn.py` joins `forkstore._EPOCH_FILES` (`attention_tile.metal` is covered by the `kernels/*.metal` glob); `fork_key_fields` takes a required `tile_status` and records `tile` as `active:<S>` or `off` (`unavailable` counts as `off`), plus `os` = `forkstore.os_release()` (`<product version> (<build>)`) whenever the tile or int8 prefill is active.
- **Cold starts:** landing the change cold-starts every fork store once; a macOS update cold-starts it again while either kernel is active. Say so in the PR.
- **Unchanged:** `SPECULATIVE_BLOCK_MAX = 5`; verifyattn stays out of `_EPOCH_FILES`; the status document's `config` block; bench row keys; monitor, TUI and `sous status` rendering; the LM backend's numerics.
- **Out of scope:** the prefill attention tile; int8 KV; block 8 and cheaper verify projections (#126, #133); other model families, head dims and dtypes; the LM backend; split targets other than the measured ones; per-chip tuning.
- **Imports:** mlx, mlx_lm and mlx_vlm imports are function-local everywhere in `src/` (the lint job runs on ubuntu without mlx); mlx-vlm internals are resolved with `importlib`.
- **Import graph:** `verifyattn` imports the `tileattn` module at module level; `tileattn` imports `verifyattn` only inside functions; at module level `tileattn` imports `gpucores`, `nax`, and `_error_line`, `_kernel`, `_kernel_text`, `_model_type` from `int8prefill` (none imports mlx at import time).
- **Monkeypatch points:** cross-module calls go through module attributes (`nax.platform_reason()`, `gpucores.read()`), so one monkeypatch reaches every caller.
- **Threads:** no new threads; the probe runs inside the engine factory on `EngineManager._load`'s `sous-model-load` thread, which already calls `release_mlx_thread_state()` on exit.
- **Test modules:** new test modules start with `mx = pytest.importorskip("mlx.core")` and put `# noqa: E402` on the deferred imports.
- **Shared helpers:** `tests/tile_fixtures.py`, imported as `from tests import tile_fixtures as tfx` (tests is a package); mlx imports there are function-local, apart from what `tileattn.availability()` does.
- **`nax` tests:** use `tfx.nax`, a `skipif` on `tfx.TILE = tileattn.availability()` read once at import (the platform rule plus a compile-and-run of every T), never int8's availability.
- **Where `nax` tests run:** the maintainer's M2 (macOS 15.5, applegpu_g14g) and CI (macos-15) cannot compile the split kernel (no MetalPerformancePrimitives); `nax` tests skip on both and run in the M5 gate.
- **Test hygiene:** tests read counter deltas (`before = dict(tileattn.calls)`); `slow` marks subprocess and multi-second tests, which still run in CI; tests never touch `~/.sous` and use tmp_path config_path and data_dir.
- **Python and tooling:** Python ≥ 3.14 (`except A, B:` without parentheses is valid PEP 758 syntax); pragmas are `# ty: ignore[rule]` with the rule ty reports, never `# type: ignore`; the type checker is ty, not mypy; `uv sync`, never pip.
- **Comments:** explain non-obvious why and never restate code; code and test comments never cite specs, plans, task or step numbers, or issue bookkeeping — they state the fact.
- **Privacy:** never put the maintainer's name or home path in committed files or GitHub text; say "the maintainer". Plan text uses `~/` paths only.
- **Commits:** Conventional Commits, imperative lowercase subject, the why in the body, ending with `Co-Authored-By: Claude <noreply@anthropic.com>` (never a model name).
- **Before each commit:** `uv run ruff format . && uv run ruff check . && uv run ty check`, then the task's tests.
- **CI runs exactly:** `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`; `addopts` already has `-q`, so never add another.
- **Records:** `docs/superpowers/**` records are never edited; the only file written there is this plan.
- **GPU etiquette on this Mac:** run GPU-heavy commands under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py` whenever another session may be using the GPU.
- **GPU etiquette on the M5:** run everything under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py` with the maintainer's daemon idle, and never touch `~/code/personal/sous`, `~/.sous` or the daemon.
- **Durable harness copy:** when execution starts, copy the #143 gate's M2 harness into `~/code/sous-spikes/tile-132/m2/` (see "Before Task 1"); it lives under `/private/tmp` and can vanish before the gate.
- **Port sources:** the kernel, the GPU core reader and the probe were verified on the M2 in the spike and are embedded verbatim below; the spike harness is mirrored on the M5 at `~/code/personal/sous-spikes/tile-132/model/`.

---

## File Structure

**Create**
- `src/sous/engine/nax.py` holds the tensor-unit ("NAX") platform rule shared by int8 prefill and the tile: `Availability` and `platform_reason()`. It has no numerics and imports no mlx at import time.
- `src/sous/engine/gpucores.py` reads the GPU core count and model from the IORegistry, once per process: IOKit through ctypes, with `/usr/sbin/ioreg` plus plistlib as the fallback. `matched_cores()` pairs the reading with mlx's device name. It never imports mlx.
- `src/sous/engine/tileattn.py` is the attention tile, shaped like `int8prefill.py` and `verifyattn.py`:
  - the schedule: `chunk_for`, `step_of`, `boundaries`, `verify_plan`;
  - the kernel entries: `tile`, `decode`, `verify`, `in_scope`;
  - its own `availability()`;
  - the `calls` counters and the `_active_splits` flag;
  - the decode hook and `serve_verify`;
  - the load-time probe, the gates and `enable()`.
- `src/sous/engine/kernels/attention_tile.metal` holds the split kernel body, a line `// REDUCE`, then the reduce kernel body. sous's own code; no new third-party code.
- `tests/test_nax.py` covers the platform-rule table.
- `tests/test_gpucores.py` covers IORegistry parsing, the ioreg fallback, model matching, caching and a live Darwin read.
- `tests/tile_fixtures.py` holds shared helpers, imported as `from tests import tile_fixtures as tfx`:
  - the plain-mlx stand-in kernel;
  - serving-layout q/k/v;
  - the tiny 24/4/256 qwen3_5 model and its prefill;
  - the hook, flag and verifier installers;
  - `TILE` and the `nax` skip marker.
- `tests/test_tileattn.py` covers the schedule, the kernel entries (stand-in, plus the real kernel under `nax`), the hooks and counters, the probe, the gates, `enable()`, the flag lifecycle, and the broken-kernel subprocess.
- `tests/test_tileattn_contract.py` runs mlx-vlm's real forward and exact verifier on the tiny model. It asserts main's precondition first, then greedy parity with the stand-in (CI) and with the real kernel (`nax`).
- `tests/test_tileattn_model.py` is `-m model`, manual, on the M5 Pro: the real 27B, drafter on against drafter off at blocks 3 and 8, and tile-off equal to main.
- `docs/superpowers/plans/2026-09-30-attention-tile.md` is this plan. It is the only file written under `docs/superpowers/**`.

**Modify**
- `src/sous/engine/int8prefill.py`:
  - `availability()` calls `nax.platform_reason()`;
  - `Availability` is imported from `nax` and stays importable here;
  - `_kernel()` gains keyword-only `ensure_row_contiguous` and `compile_options`.
- `src/sous/engine/kernels/__init__.py`, `common.h`, `nax.h`: the docstring and first comment lines name the tile as a second user (`nax.h` now also goes to the tile's split kernel).
- `THIRD_PARTY_NOTICES.md`: one sentence saying `attention_tile.metal` is sous's own, not oMLX-derived.
- `src/sous/engine/verifyattn.py`:
  - a module-level `tileattn` import;
  - `_in_scope(..., splits=0)` widens while the tile is active;
  - `_stock_verify()`;
  - the tile branch in `_wrap`;
  - the module docstring's T = 2 sentence.
- `src/sous/config.py`: the `attention_tile` field, the `_attention_tile()` parser and the `_KNOWN` entry.
- `src/sous/engine/vlm.py`:
  - the `attention_tile` kwarg;
  - `tileattn.enable()` right after `verifyattn.enable()`;
  - `unload()` calls `tileattn.clear()`;
  - later, `_fork_key_fields` passes `tile_status`.
- `src/sous/engine/lm.py`: `attention_tile_status = {"state": "off", "reason": None}`; later it passes that to `fork_key_fields`.
- `src/sous/engine/base.py`:
  - `_default_factory` and `default_engine_factory` plumbing;
  - `ManagedEngine.attention_tile_status`;
  - the model-load line tokens;
  - the `status()` engine block.
- `src/sous/engine/forkstore.py`: `tileattn.py` joins `_EPOCH_FILES`; `fork_key_fields(tile_status=...)` records `tile` and `os`; new `os_release()`; `engine_epoch()`'s header comment names every kernel.
- `src/sous/tune/arms.py`: `Arm.attention_tile`, `suite_key`, `_arm`.
- `src/sous/tune/suite/runner.py`:
  - `SuiteRun.attention_tile` and `key`;
  - the `from_dict` fallback for older rows;
  - `_check_tile`;
  - the `CountingEngine` pass-through comment.
- `src/sous/tune/decide.py`: the `ArmSummary.key` annotation only.
- Tests:
  - `tests/test_verifyattn.py` (the `_scope` helper plus the tile-active scope table);
  - `tests/test_config.py`, `tests/test_engine_base.py`, `tests/test_engine_unloaded.py`;
  - `tests/test_forkstore.py`, `tests/test_engine_forkhooks.py`;
  - `tests/test_tune_arms.py`, `tests/test_tune_runner.py`, `tests/test_tune_main.py`, `tests/test_tune_decide.py`, `tests/test_tune_report.py`.
- Docs:
  - `CLAUDE.md`: the new gotcha, the epoch list, the verifyattn T = 2 amendment and the int8 MPP/platform-rule wording;
  - `docs/configuration.md`, `docs/observability.md`, `docs/tuning.md`, `docs/how-it-works.md`;
  - `README.md`: decode figures only if the merge gate justifies it.

**Out of tree (merge gate only, never committed)**
- On the M5: `~/code/personal/sous-spikes/tile-132/gate/`, authored here in `~/code/sous-spikes/tile-132/gate/` and rsynced.
  - New: `m5.sh`, `wrap_branch.py` (the GATE_WRAP counter shim), `run_gate.sh`, `suite_ab.py`, `ktime.py`, `suite/config.toml`.
  - Copied from `tile-132/model/`: `gate132.py`, `analyze132.py`, `analyze133.py`, `analyze_suite.py`, `bench_model.py`, `cmp_decode.py`.
  - The branch worktree `~/code/personal/sous-remote-tile`, beside the main checkout `~/code/personal/sous-remote`.
- On the M2: `~/code/sous-spikes/tile-132/m2/` holds a copy of the #143 M2 harness (`gate143_m2.py`, `bodies-small.json`, `build_bodies.py`) plus a main checkout and a branch checkout, each with its own `.venv`.

---

## Before Task 1: Keep the M2 gate harness

The #143 gate's M2 harness lives under `/private/tmp`, which can be cleared before Task 13 needs it. Copy it out first:

```bash
mkdir -p ~/code/sous-spikes/tile-132/m2
src="$(dirname "$(find /private/tmp -path '*scratchpad/m2/gate143_m2.py' 2>/dev/null | head -1)")"
cp -n "$src/gate143_m2.py" "$src/bodies-small.json" "$src/build_bodies.py" ~/code/sous-spikes/tile-132/m2/
ls ~/code/sous-spikes/tile-132/m2/
```

Expected: `bodies-small.json`, `build_bodies.py` and `gate143_m2.py` are listed. If `find` returns nothing, say so before starting: Task 13's M2 criterion cannot run as specified without it.

---

### Task 1: The shared NAX platform rule

**Files:**
- Create: `src/sous/engine/nax.py`
- Modify: `src/sous/engine/int8prefill.py:15-23` (the imports) and `:128-166` (`Availability`, `_ARCH` and the platform half of `availability()`; its GEMM probe at `:166-169` stays)
- Test: `tests/test_nax.py` (create)
- Must still pass unchanged: `tests/test_int8prefill.py`, `tests/test_tune_hardware.py`, `tests/test_tune_report.py`

**Interfaces:**
- Consumes (existing only):
  - `int8prefill._gemm_probe() -> str | None`.
  - Every importer of `from sous.engine.int8prefill import Availability`: `src/sous/tune/hardware.py:14`, `tests/test_tune_hardware.py:1`, `tests/test_tune_report.py:7`.
  - The int8 platform tests at `tests/test_int8prefill.py:113-139`, which monkeypatch `platform.system`, `platform.mac_ver`, `mx.device_info` and `i8._gemm_probe`.
- Produces:
  - `sous.engine.nax.Availability`: `@dataclass(frozen=True)` with `available: bool` and `reason: str | None = None`, moved unchanged.
  - `sous.engine.nax.platform_reason() -> str | None`: None when mlx 0.32.2's own `is_nax_available()` rule passes (Darwin; mlx importable; `mx.metal.is_available()`; `platform.mac_ver()` release ≥ (26, 2); `mx.device_info()['architecture']` matching `applegpu_g(\d+)(\D?)$` with generation ≥ 17, or ≥ 18 for a 'p' suffix). Otherwise exactly today's int8prefill texts: `'not macOS'`, `f'mlx unavailable: {e}'`, `'Metal unavailable'`, `f'macOS {release} < 26.2'`, `f'unrecognized GPU architecture {arch!r}'`, `f'GPU {arch} predates the neural accelerators (gen {gen} < {need})'`. It reads the platform and mlx at call time; the mlx import is function-local.
  - `sous.engine.int8prefill.Availability` is `nax.Availability`, imported into int8prefill so existing importers keep working.
  - `int8prefill.availability() -> Availability`: `reason = nax.platform_reason()`; if it is not None, `Availability(False, reason)`; otherwise `_gemm_probe()` exactly as today. int8prefill calls the rule through the module (`from sous.engine import nax`), so one monkeypatch reaches both kernels.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nax.py`. The platform table is int8's own (`tests/test_int8prefill.py:121-139`), now asserted against the whole reason text:

```python
"""The tensor-unit platform rule int8 prefill and the attention tile share
(sous.engine.nax). The platform and the device are faked, so the table runs on
any Mac, CI's macos-15 included."""

import platform

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax  # noqa: E402 — after the importorskip guard


def _fake_platform(monkeypatch, mac_ver: str, arch: str):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: (mac_ver, ("", "", ""), "arm64"))
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})


@pytest.mark.parametrize(
    ("mac_ver", "arch", "reason"),
    [
        ("26.6.2", "applegpu_g17s", None),
        ("26.2", "applegpu_g17g", None),
        ("26.6.2", "applegpu_g18p", None),
        (
            "26.6.2",
            "applegpu_g17p",
            "GPU applegpu_g17p predates the neural accelerators (gen 17 < 18)",
        ),
        (
            "26.6.2",
            "applegpu_g16s",
            "GPU applegpu_g16s predates the neural accelerators (gen 16 < 17)",
        ),
        ("26.1", "applegpu_g17s", "macOS 26.1 < 26.2"),
        ("26.6.2", "something_else", "unrecognized GPU architecture 'something_else'"),
    ],
)
def test_platform_reason_mirrors_mlx_nax_rule(monkeypatch, mac_ver, arch, reason):
    _fake_platform(monkeypatch, mac_ver, arch)
    assert nax.platform_reason() == reason


def test_platform_reason_refuses_other_systems(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert nax.platform_reason() == "not macOS"


def test_platform_reason_refuses_a_machine_without_metal(monkeypatch):
    _fake_platform(monkeypatch, "26.6.2", "applegpu_g17s")
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert nax.platform_reason() == "Metal unavailable"


def test_int8_availability_takes_the_shared_rule_before_its_gemm_probe(monkeypatch):
    """One monkeypatch point for both kernels, and a refused platform never
    compiles the GEMM."""

    def compiled():
        raise AssertionError("the GEMM probe ran on a refused platform")

    monkeypatch.setattr(nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(int8prefill, "_gemm_probe", compiled)
    assert int8prefill.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_int8prefill_keeps_exporting_the_shared_availability():
    """tune/hardware.py and the tune tests import it from int8prefill."""
    assert int8prefill.Availability is nax.Availability
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nax.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'nax' from 'sous.engine'`.

- [ ] **Step 3: Create the shared rule**

Create `src/sous/engine/nax.py`. The body of `platform_reason()` is lines 144-165 of today's `int8prefill.availability()`, each `return Availability(False, text)` now `return text`:

```python
"""The tensor-unit ("NAX") platform rule sous's tensor-op kernels share.

Both runtime-compiled kernels that use the Metal-4 tensor ops, int8 prefill's GEMM
and the attention tile's split kernel, serve only where mlx itself would use the
neural accelerators. On an older GPU the tensor ops run on the plain shader path,
where neither kernel buys anything, so it is refused rather than merely slow.
This module holds that rule and nothing else: no numerics, and mlx is imported
only when the rule is read.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass

_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


def platform_reason() -> str | None:
    """None when mlx 0.32.2's own is_nax_available() (device.cpp) would pass:
    macOS >= 26.2 and an Apple GPU of generation >= 17 (>= 18 for 'p'-suffix
    parts), the M5 family and later. Otherwise why not. Read at call time, so a
    test that fakes the platform or the device reaches every kernel's caller."""
    if platform.system() != "Darwin":
        return "not macOS"
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return f"mlx unavailable: {e}"
    if not mx.metal.is_available():
        return "Metal unavailable"
    release = platform.mac_ver()[0]
    parts = tuple(int(p) for p in release.split(".") if p.isdigit())
    if parts[:2] < (26, 2):
        return f"macOS {release} < 26.2"
    arch = str(mx.device_info().get("architecture", ""))
    match = _ARCH.match(arch)
    if match is None:
        return f"unrecognized GPU architecture {arch!r}"
    gen = int(match.group(1))
    need = 18 if match.group(2) == "p" else 17
    if gen < need:
        return f"GPU {arch} predates the neural accelerators (gen {gen} < {need})"
    return None
```

- [ ] **Step 4: Point int8prefill at it**

In `src/sous/engine/int8prefill.py`, `platform`, `re` and `dataclass` served only the rule that moved. Replace the imports (`:15-23`):

```python
import functools
import logging
import platform
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, cast
```

with:

```python
import functools
import logging
import warnings
from collections.abc import Callable, Iterable
from importlib.resources import files
from typing import Any, cast

from sous.engine import nax
from sous.engine.nax import Availability
```

Then replace the `Availability` dataclass, `_ARCH` and the platform half of `availability()` (`:128-166`), keeping the GEMM probe that follows:

```python
@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


def availability() -> Availability:
    """mlx's own is_nax_available() (device.cpp, 0.32.2) — macOS >= 26.2 and an
    Apple GPU of generation >= 17 (>= 18 for 'p'-suffix parts), the M5 family and
    later — then `_gemm_probe()`, which compiles and runs the kernels, so this is
    mlx work, not a device query. Older GPUs run the Metal-4 tensor ops on the
    plain shader path, where int8 buys nothing, so they are `unavailable`, not
    merely slow."""
    if platform.system() != "Darwin":
        return Availability(False, "not macOS")
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return Availability(False, f"mlx unavailable: {e}")
    if not mx.metal.is_available():
        return Availability(False, "Metal unavailable")
    release = platform.mac_ver()[0]
    parts = tuple(int(p) for p in release.split(".") if p.isdigit())
    if parts[:2] < (26, 2):
        return Availability(False, f"macOS {release} < 26.2")
    arch = str(mx.device_info().get("architecture", ""))
    match = _ARCH.match(arch)
    if match is None:
        return Availability(False, f"unrecognized GPU architecture {arch!r}")
    gen = int(match.group(1))
    need = 18 if match.group(2) == "p" else 17
    if gen < need:
        return Availability(
            False, f"GPU {arch} predates the neural accelerators (gen {gen} < {need})"
        )
    rejected = _gemm_probe()
```

with:

```python
def availability() -> Availability:
    """The tensor-unit rule int8 prefill shares with the attention tile
    (`nax.platform_reason()`: macOS >= 26.2 and an M5-family or later GPU), then
    `_gemm_probe()`, which compiles and runs the kernels, so this is mlx work, not
    a device query."""
    reason = nax.platform_reason()
    if reason is not None:
        return Availability(False, reason)
    rejected = _gemm_probe()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nax.py -v`
Expected: PASS, 11 passed.

- [ ] **Step 6: Run the unchanged int8 and tune tests**

Run: `uv run pytest tests/test_int8prefill.py tests/test_tune_hardware.py tests/test_tune_report.py -v`
Expected: PASS, 57 passed and 8 skipped: the skips are int8's `nax` GEMM tests, which run only where the tensor units are (the M5 Pro). `test_availability_mirrors_mlx_nax_rule` still passes because `nax.platform_reason()` reads `platform` and `mx.device_info` at call time.

- [ ] **Step 7: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/int8prefill.py src/sous/engine/nax.py tests/test_nax.py
git commit -m "refactor(engine): share the tensor-unit platform rule between kernels" -m "The attention tile needs the same macOS 26.2 / M5-family rule that int8
prefill applies before it compiles anything, and the two kernels must never
disagree about which machines have tensor units. The rule moves to nax.py
with its texts unchanged; int8prefill keeps exporting Availability for the
tune code that imports it from there.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: The GPU core-count reader

**Files:**
- Create: `src/sous/engine/gpucores.py`
- Test: `tests/test_gpucores.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `gpucores.GPUCores`: `@dataclass(frozen=True)` with `cores: int | None`, `model: str | None`, `reason: str | None` (None when cores and model were read) and `source: str` (`'iokit'`, `'ioreg'` or `'none'`).
  - `gpucores.parse_services(services: list[dict[str, Any]], source: str) -> GPUCores`: exactly one service, an int (not bool) `'gpu-core-count'` > 0 and a non-empty str `'model'`. Reasons: `'no AGXAccelerator service'`, `f'{len(services)} AGXAccelerator services'`, `f'gpu-core-count is {cores!r}'`, `f'model is {model!r}'`.
  - `gpucores._iokit_services() -> list[dict[str, Any]]`: ctypes IOKit and CoreFoundation, every CF/IO object released; raises OSError when IOKit cannot be used.
  - `gpucores._ioreg_services() -> list[dict[str, Any]]`: `subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-c', 'AGXAccelerator', '-d', '1'], capture_output=True, check=True, timeout=5.0)` parsed with `plistlib.loads`; empty stdout gives `[]`, a non-list plist raises ValueError.
  - `gpucores.read() -> GPUCores`: `@functools.cache`, never raises; IOKit first, `except OSError, AttributeError, ValueError:` falls back to ioreg, and both failing gives `GPUCores(None, None, f'GPU core count unreadable: {e}', 'none')`.
  - `gpucores.matched_cores(device_name: str) -> tuple[int | None, str | None]`: `(cores, None)` when `read()` succeeded and its model equals `device_name`, else `(None, reason)` with `read().reason` or `f"IORegistry GPU {model!r} is not mlx's {device_name!r}"`. It calls `read()` through the module global, so `gpucores.read` is the single monkeypatch point for S.
  - Constants `_IOKIT`, `_CORE_FOUNDATION`, `_IOREG = '/usr/sbin/ioreg'`, `_IOREG_TIMEOUT_S = 5.0`, `_SERVICE_CLASS = b'AGXAccelerator'`, `_KEYS = ('gpu-core-count', 'model')`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gpucores.py`. An autouse fixture clears `read()`'s cache around every test. The live test must not assert success: a virtualised CI GPU may expose no AGXAccelerator service.

```python
"""The GPU core count (sous.engine.gpucores): IORegistry parsing, the ioreg
fallback, the match against mlx's device name, and one live read on a Mac."""

import plistlib
import subprocess
import sys

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores  # noqa: E402 — after the importorskip guard


@pytest.fixture(autouse=True)
def _fresh_read():
    gpucores.read.cache_clear()
    yield
    gpucores.read.cache_clear()


@pytest.mark.parametrize(
    ("services", "expected"),
    [
        (
            [{"gpu-core-count": 20, "model": "Apple M5 Pro"}],
            gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"),
        ),
        ([], gpucores.GPUCores(None, None, "no AGXAccelerator service", "iokit")),
        (
            [{"gpu-core-count": 8, "model": "Apple M2"}] * 2,
            gpucores.GPUCores(None, None, "2 AGXAccelerator services", "iokit"),
        ),
        (
            [{"gpu-core-count": True, "model": "Apple M2"}],
            gpucores.GPUCores(None, None, "gpu-core-count is True", "iokit"),
        ),
        (
            [{"gpu-core-count": 0, "model": "Apple M2"}],
            gpucores.GPUCores(None, None, "gpu-core-count is 0", "iokit"),
        ),
        (
            [{"gpu-core-count": 8, "model": None}],
            gpucores.GPUCores(None, None, "model is None", "iokit"),
        ),
    ],
)
def test_parse_services_needs_one_service_with_a_core_count_and_a_model(services, expected):
    assert gpucores.parse_services(services, "iokit") == expected


def _unusable():
    raise OSError("IOKit unusable")


def test_read_falls_back_to_ioreg_when_iokit_cannot_be_used(monkeypatch):
    monkeypatch.setattr(gpucores, "_iokit_services", _unusable)
    monkeypatch.setattr(
        gpucores, "_ioreg_services", lambda: [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]
    )
    assert gpucores.read() == gpucores.GPUCores(20, "Apple M5 Pro", None, "ioreg")


def test_read_reports_why_when_neither_reader_works(monkeypatch):
    def timed_out():
        raise subprocess.TimeoutExpired(["/usr/sbin/ioreg"], 5.0)

    monkeypatch.setattr(gpucores, "_iokit_services", _unusable)
    monkeypatch.setattr(gpucores, "_ioreg_services", timed_out)
    reading = gpucores.read()
    assert (reading.cores, reading.model, reading.source) == (None, None, "none")
    assert reading.reason is not None
    assert reading.reason.startswith("GPU core count unreadable")


def _fake_ioreg(monkeypatch, stdout: bytes) -> list:
    seen = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_ioreg_services_parse_the_plist_from_an_absolute_path_with_a_timeout(monkeypatch):
    entries = [{"gpu-core-count": 20, "model": "Apple M5 Pro", "IOClass": "AGXAcceleratorG17X"}]
    seen = _fake_ioreg(monkeypatch, plistlib.dumps(entries))
    assert gpucores._ioreg_services() == [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]
    (argv, kwargs), *_ = seen
    assert argv == ["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"]
    assert kwargs["timeout"] == 5.0 and kwargs["check"] is True


def test_ioreg_services_are_empty_when_nothing_matches(monkeypatch):
    _fake_ioreg(monkeypatch, b"")
    assert gpucores._ioreg_services() == []


def test_ioreg_services_refuse_a_plist_that_is_not_an_array(monkeypatch):
    _fake_ioreg(monkeypatch, plistlib.dumps({"model": "Apple M5 Pro"}))
    with pytest.raises(ValueError, match="not return a plist array"):
        gpucores._ioreg_services()


def _reading(monkeypatch, reading):
    monkeypatch.setattr(gpucores, "read", lambda: reading)


def test_matched_cores_are_the_count_when_the_models_agree(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"))
    assert gpucores.matched_cores("Apple M5 Pro") == (20, None)


def test_matched_cores_refuse_another_gpu(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"))
    assert gpucores.matched_cores("Apple M5 Max") == (
        None,
        "IORegistry GPU 'Apple M5 Pro' is not mlx's 'Apple M5 Max'",
    )


def test_matched_cores_carry_the_readers_reason(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(None, None, "no AGXAccelerator service", "iokit"))
    assert gpucores.matched_cores("Apple M5 Pro") == (None, "no AGXAccelerator service")


def test_read_happens_once_per_process(monkeypatch):
    seen = []

    def services():
        seen.append(1)
        return [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]

    monkeypatch.setattr(gpucores, "_iokit_services", services)
    assert gpucores.read() is gpucores.read()
    assert seen == [1]


@pytest.mark.skipif(sys.platform != "darwin", reason="the IORegistry is macOS's")
def test_a_live_read_never_raises_and_names_mlxs_gpu():
    """A virtualised CI GPU may expose no AGXAccelerator service, so only a
    reading that succeeded is checked against mlx."""
    reading = gpucores.read()
    if reading.reason is not None:
        return
    assert reading.cores is not None and reading.cores > 0
    assert reading.source in ("iokit", "ioreg")
    assert reading.model == mx.device_info()["device_name"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gpucores.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'gpucores' from 'sous.engine'`.

- [ ] **Step 3: Create the reader**

Create `src/sous/engine/gpucores.py`. It imports no mlx anywhere: the caller pairs the reading with mlx's device name. The unparenthesised `except OSError, AttributeError, ValueError:` is PEP 758 syntax, valid on 3.14 and what `ruff format` produces.

```python
"""The GPU's core count, read from the IORegistry: the attention tile's split target.

mlx exposes no core count (device_info carries the name and the architecture
only), so it is read from the AGXAccelerator service's "gpu-core-count", the
figure System Information shows. IOKit through ctypes first; /usr/sbin/ioreg as
the fallback when IOKit itself cannot be called. Never raises, and never imports
mlx: the caller pairs the reading with mlx's device name.
"""

from __future__ import annotations

import ctypes
import functools
import plistlib
import subprocess
from dataclasses import dataclass
from typing import Any

_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_IOREG = "/usr/sbin/ioreg"
_IOREG_TIMEOUT_S = 5.0
_SERVICE_CLASS = b"AGXAccelerator"
_KEYS = ("gpu-core-count", "model")
_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_SINT64 = 4  # kCFNumberSInt64Type


@dataclass(frozen=True)
class GPUCores:
    cores: int | None
    model: str | None
    reason: str | None  # None when cores and model were read
    source: str  # "iokit", "ioreg" or "none"


def parse_services(services: list[dict[str, Any]], source: str) -> GPUCores:
    """Exactly one service with an integer core count > 0 and a model string."""
    if not services:
        return GPUCores(None, None, "no AGXAccelerator service", source)
    if len(services) > 1:
        return GPUCores(None, None, f"{len(services)} AGXAccelerator services", source)
    cores = services[0].get("gpu-core-count")
    model = services[0].get("model")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores <= 0:
        return GPUCores(None, None, f"gpu-core-count is {cores!r}", source)
    if not isinstance(model, str) or not model:
        return GPUCores(None, None, f"model is {model!r}", source)
    return GPUCores(cores, model, None, source)


def _iokit_services() -> list[dict[str, Any]]:
    """Every AGXAccelerator service's two properties; OSError when IOKit cannot be used."""
    iokit = ctypes.CDLL(_IOKIT)
    cf = ctypes.CDLL(_CORE_FOUNDATION)
    vp, u32 = ctypes.c_void_p, ctypes.c_uint32
    iokit.IOServiceMatching.restype = vp
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
    iokit.IOServiceGetMatchingServices.argtypes = [u32, vp, ctypes.POINTER(u32)]
    iokit.IOIteratorNext.restype = u32
    iokit.IOIteratorNext.argtypes = [u32]
    iokit.IORegistryEntryCreateCFProperty.restype = vp
    iokit.IORegistryEntryCreateCFProperty.argtypes = [u32, vp, vp, u32]
    iokit.IOObjectRelease.restype = ctypes.c_int
    iokit.IOObjectRelease.argtypes = [u32]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, u32]
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [vp]
    cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetTypeID.restype = ctypes.c_ulong
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, u32]
    cf.CFRelease.restype = None
    cf.CFRelease.argtypes = [vp]

    def value(ref: int) -> Any:
        tid = cf.CFGetTypeID(ref)
        if tid == cf.CFNumberGetTypeID():
            out = ctypes.c_int64()
            return out.value if cf.CFNumberGetValue(ref, _SINT64, ctypes.byref(out)) else None
        if tid == cf.CFStringGetTypeID():
            buf = ctypes.create_string_buffer(256)
            if cf.CFStringGetCString(ref, buf, len(buf), _UTF8):
                return buf.value.decode("utf-8")
        return None

    matching = iokit.IOServiceMatching(_SERVICE_CLASS)
    if not matching:
        raise OSError("IOServiceMatching returned NULL")
    iterator = u32(0)
    # kIOMainPortDefault is MACH_PORT_NULL; the call consumes `matching`.
    kr = iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator))
    if kr != 0:
        raise OSError(f"IOServiceGetMatchingServices returned {kr:#x}")
    keys = {k: cf.CFStringCreateWithCString(None, k.encode(), _UTF8) for k in _KEYS}
    services: list[dict[str, Any]] = []
    try:
        while service := iokit.IOIteratorNext(iterator):
            try:
                props: dict[str, Any] = {}
                for name, key in keys.items():
                    ref = iokit.IORegistryEntryCreateCFProperty(service, key, None, 0)
                    if ref:
                        try:
                            props[name] = value(ref)
                        finally:
                            cf.CFRelease(ref)
                services.append(props)
            finally:
                iokit.IOObjectRelease(service)
    finally:
        for key in keys.values():
            if key:
                cf.CFRelease(key)
        iokit.IOObjectRelease(iterator)
    return services


def _ioreg_services() -> list[dict[str, Any]]:
    """The same two properties from ioreg's plist; OSError/ValueError when unreadable."""
    out = subprocess.run(
        [_IOREG, "-a", "-r", "-c", _SERVICE_CLASS.decode(), "-d", "1"],
        capture_output=True,
        check=True,
        timeout=_IOREG_TIMEOUT_S,
    ).stdout
    if not out.strip():  # ioreg prints nothing when no service matches
        return []
    entries = plistlib.loads(out)
    if not isinstance(entries, list):
        raise ValueError("ioreg did not return a plist array")
    return [{k: e.get(k) for k in _KEYS} for e in entries if isinstance(e, dict)]


@functools.cache
def read() -> GPUCores:
    """Read once per process: the core count cannot change under a running daemon."""
    try:
        return parse_services(_iokit_services(), "iokit")
    except OSError, AttributeError, ValueError:
        pass
    try:
        return parse_services(_ioreg_services(), "ioreg")
    except (OSError, ValueError, subprocess.SubprocessError, plistlib.InvalidFileException) as e:
        return GPUCores(None, None, f"GPU core count unreadable: {e}", "none")


def matched_cores(device_name: str) -> tuple[int | None, str | None]:
    """(cores, None) when the IORegistry's GPU is the one mlx runs on, else
    (None, why not). Through the module global, so `gpucores.read` is the one
    place a test sets the split target."""
    reading = read()
    if reading.reason is not None:
        return None, reading.reason
    if reading.model != device_name:
        return None, f"IORegistry GPU {reading.model!r} is not mlx's {device_name!r}"
    return reading.cores, None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gpucores.py -v`
Expected: PASS, 16 passed. On the maintainer's M2, `read()` gives `GPUCores(cores=8, model='Apple M2', reason=None, source='iokit')` in well under a millisecond; the ioreg fallback gives the same reading in about 20 ms.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/gpucores.py tests/test_gpucores.py
git commit -m "feat(engine): read the GPU's core count from the IORegistry" -m "The attention tile partitions keys for one wave of threadgroups, S
splits per KV head on S cores, and mlx exposes no core count. IOKit through
ctypes reads the AGXAccelerator's gpu-core-count, the figure System
Information shows, with /usr/sbin/ioreg as the fallback; a reading counts
only when its model is the GPU mlx runs on.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: The pure schedule

**Files:**
- Create: `src/sous/engine/tileattn.py` (the module docstring for the whole feature, the constants and the schedule)
- Test: `tests/test_tileattn.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces, in `sous.engine.tileattn`:
  - Constants `HQ = 24`, `HKV = 4`, `GQA = 6`, `D = 256`, `SCALE = 0.0625`, `MAX_RUN = 8`, `N0 = 1536`, `GRAN = 32`, `MEASURED_SPLITS = frozenset({20})`.
  - `logger = logging.getLogger('sous.engine.tileattn')`.
  - `chunk_for(n: int, splits: int) -> int`: 0 below N0, else `GRAN * -(-n // (GRAN * splits))`.
  - `step_of(n: int, splits: int) -> tuple[int, int, int]`: (first n, last n, chunk) of n's step; `(1, N0 - 1, 0)` below N0, else, with `k = chunk // GRAN`, `(max(N0, GRAN * splits * (k - 1) + 1), GRAN * splits * k, chunk)`.
  - `boundaries(n_max: int, splits: int) -> list[int]`: every key count B ≤ n_max whose step differs from B − 1's, N0 first.
  - `verify_plan(prefix: int, t: int, splits: int) -> list[tuple[int, int, int]]`: contiguous runs `(j, k, chunk)` over rows 0..t−1, where row r has `prefix + r + 1` keys; a run never crosses N0 or a step boundary and never exceeds MAX_RUN rows; chunk 0 means stock.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tileattn.py`. It starts with the importorskip guard because it will hold the GPU tests too. The sweep over every n up to 262,144 takes about 0.1 s per split target.

```python
"""The attention tile (sous.engine.tileattn).

The schedule is pure Python. The kernel entries run on any Metal GPU with a
plain-mlx stand-in for the Metal kernel, which CI's GPU (macos-15) cannot
compile; the `nax`-marked tests run the real kernel where the tensor units are.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import tileattn  # noqa: E402 — after the importorskip guard

N0 = tileattn.N0
SPLIT_TARGETS = [10, 16, 20, 40]


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_below_n0_is_the_stock_path(splits):
    assert tileattn.chunk_for(N0 - 1, splits) == 0
    assert tileattn.chunk_for(N0, splits) > 0
    assert tileattn.step_of(N0 - 1, splits) == (1, N0 - 1, 0)


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_every_step_past_the_first_is_32_s_keys_wide(splits):
    marks = tileattn.boundaries(262_144, splits)
    for mark in marks[1:]:
        first, last, chunk = tileattn.step_of(mark, splits)
        assert (first, last - first + 1) == (mark, 32 * splits), mark
        assert chunk == tileattn.chunk_for(mark, splits) == tileattn.chunk_for(last, splits)
        assert tileattn.chunk_for(mark - 1, splits) == chunk - tileattn.GRAN


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_no_row_gets_more_than_s_splits(splits):
    for n in range(N0, 262_145):
        assert -(-n // tileattn.chunk_for(n, splits)) <= splits, n


def test_twenty_splits_from_the_tail_of_step_nineteen():
    assert -(-11_552 // tileattn.chunk_for(11_552, 20)) == 19
    assert all(-(-n // tileattn.chunk_for(n, 20)) == 20 for n in range(11_553, 262_145))


@pytest.mark.parametrize(
    ("splits", "first_four"),
    [
        (20, [1536, 1921, 2561, 3201]),
        (10, [1536, 1601, 1921, 2241]),
        (16, [1536, 1537, 2049, 2561]),  # N0 is a step of one key at 16 splits
        (40, [1536, 2561, 3841, 5121]),
    ],
)
def test_boundaries_start_at_n0(splits, first_four):
    assert tileattn.boundaries(262_144, splits)[:4] == first_four


def test_the_m5_pros_schedule_has_408_steps_to_262k():
    assert len(tileattn.boundaries(262_144, 20)) == 408


def test_step_of_either_side_of_the_57600_boundary():
    assert tileattn.step_of(57_600, 20) == (56_961, 57_600, 2_880)
    assert tileattn.step_of(57_601, 20) == (57_601, 58_240, 2_912)


@pytest.mark.parametrize(
    ("prefix", "t", "runs"),
    [
        # rows 1531..1535 are stock; 1536..1543 fill a run; 1544..1546 continue the step
        (1530, 16, [(0, 5, 0), (5, 13, 96), (13, 16, 96)]),
        # a run cut at 8 rows, then the step boundary at 57601 inside the call
        (57_590, 16, [(0, 8, 2880), (8, 10, 2880), (10, 16, 2912)]),
        (5000, 12, [(0, 8, 256), (8, 12, 256)]),
    ],
)
def test_verify_plan_cuts_at_n0_steps_and_eight_rows(prefix, t, runs):
    assert tileattn.verify_plan(prefix, t, 20) == runs


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_verify_plan_runs_cover_every_row_at_its_own_chunk(splits):
    for prefix in range(N0 - 20, N0 + 3 * 32 * splits + 1):
        for t in range(1, 17):
            runs = tileattn.verify_plan(prefix, t, splits)
            assert [j for j, _, _ in runs] == [0, *[k for _, k, _ in runs[:-1]]]
            assert runs[-1][1] == t
            for j, k, chunk in runs:
                assert 1 <= k - j <= tileattn.MAX_RUN
                assert all(tileattn.chunk_for(prefix + r + 1, splits) == chunk for r in range(j, k))
            for (j, k, chunk), (_, _, following) in zip(runs, runs[1:], strict=False):
                assert chunk != following or k - j == tileattn.MAX_RUN, (prefix, t, runs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'tileattn' from 'sous.engine'`.

- [ ] **Step 3: Create the module with its docstring, constants and schedule**

Create `src/sous/engine/tileattn.py`. The docstring is written once here for the whole feature, so later tasks add code only. No mlx at module level.

```python
"""A tensor-unit attention tile for qwen3_5's one-row decode and exact verifier.

The six query heads that share a KV head are packed with a call's rows into MPP
``matmul2d`` fragments on the M5's tensor units: a split kernel per (key split,
KV head) with an fp32 online softmax in registers, then a reduce kernel that
combines the splits in a fixed order. The kernels (``kernels/attention_tile.metal``)
are specialised to bf16, 24 query / 4 KV heads, head dim 256 and scale 1/16.

Exactness rests on one property: a row's output depends on its key partition,
the chunk each split covers, and on nothing else about the call. Its 32-key
blocks align the same way in every call, keys past its limit get probability
exactly 0, and a (row, key) product does not depend on the fragment row it sits
in. So the chunk is fixed within each step of 32 * S keys (S the GPU's core
count, one wave of threadgroups); a verify call is cut into runs that never
cross N0, a step boundary or 8 rows; and below N0 decode and verify both take
stock paths that are exact against each other. Every verify row then equals a
one-row decode at its own length, so greedy output with a drafter equals greedy
output without one, whatever the block size.

Two hooks reach it. One-row decode goes through a replacement of the
``scaled_dot_product_attention`` global in mlx-vlm's qwen3_5 language module,
the call ``Qwen3_5Attention.__call__`` makes after its projections; verify rows
come from verifyattn's wrapper of the exact verifier. Both decide after the
projections and fall back to the very call the model would have made, and both
act only while ``_active_splits`` is set. Only an ``enable()`` that passed every
gate and the load-time probe sets it; every ``enable()`` and every unload clears
it. Nothing here raises into a model load.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sous.engine.tileattn")

HQ = 24
HKV = 4
GQA = 6
D = 256
# D ** -0.5, compiled into the kernel as SCALE_LOG2.
SCALE = 0.0625
# The split kernel is instantiated for T = 1..MAX_RUN rows; longer verify calls are cut.
MAX_RUN = 8
# The first key count the tile serves. Below it stock one-row SDPA is as fast (T = 1
# wall time against stock: 0.97x at 1024 keys, 1.02-1.04x at 1101-1281, 1.08x at
# 1536, 1.11x at 2048).
N0 = 1536
# The kernel's key block: a chunk is a multiple of it, and a step is GRAN * S keys.
GRAN = 32
# Split targets the merge gate measured for speed: the M5 Pro's 20 GPU cores.
MEASURED_SPLITS = frozenset({20})


def chunk_for(n: int, splits: int) -> int:
    """The split size for a row that sees n keys; 0 means the stock path. Step k
    covers n in (GRAN*S*(k-1), GRAN*S*k] and uses chunk GRAN*k, so no row gets more
    than S splits: S splits x HKV heads is one wave of threadgroups on S cores."""
    if n < N0:
        return 0
    return GRAN * -(-n // (GRAN * splits))


def step_of(n: int, splits: int) -> tuple[int, int, int]:
    """(first n, last n, chunk) of the step that holds key count n."""
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return (1, N0 - 1, 0)
    k = chunk // GRAN
    return (max(N0, GRAN * splits * (k - 1) + 1), GRAN * splits * k, chunk)


def boundaries(n_max: int, splits: int) -> list[int]:
    """Every key count B <= n_max whose step differs from B - 1's, N0 first."""
    out = [N0]
    n = N0
    while True:
        n = step_of(n, splits)[1] + 1
        if n > n_max:
            return out
        out.append(n)


def verify_plan(prefix: int, t: int, splits: int) -> list[tuple[int, int, int]]:
    """A T-row verify behind `prefix` keys as runs (j, k, chunk) of rows [j, k);
    row r sees prefix + r + 1 keys and chunk 0 means stock. A run never crosses N0
    or a step boundary and never holds more than MAX_RUN rows, so any T is served."""
    runs: list[tuple[int, int, int]] = []
    j = 0
    while j < t:
        chunk = chunk_for(prefix + j + 1, splits)
        k = j + 1
        while k < t and k - j < MAX_RUN and chunk_for(prefix + k + 1, splits) == chunk:
            k += 1
        runs.append((j, k, chunk))
        j = k
    return runs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -v`
Expected: PASS, 26 passed, in about half a second.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/tileattn.py tests/test_tileattn.py
git commit -m "feat(engine): schedule the attention tile's key partition by context step" -m "A verify row is bit-identical to a one-row decode at its own length only
when both split the keys the same way. The chunk is therefore fixed per
step of 32 x S keys and stock below N0, and a verify call is cut into
runs that never cross a step or exceed the kernel's 8 row variants, so
any drafter block stays exact.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: The kernel, its decode/verify entries and the CI stand-in

**Files:**
- Create: `src/sous/engine/kernels/attention_tile.metal`
- Modify: `src/sous/engine/int8prefill.py:55-79` (`_kernel` gains two keyword-only arguments)
- Modify: `src/sous/engine/tileattn.py` (the imports, two constants, the kernel pair, the entries, the compile probe and `availability()`)
- Modify: `src/sous/engine/kernels/__init__.py` (the docstring), `src/sous/engine/kernels/common.h:1` and `src/sous/engine/kernels/nax.h:1-3` (the first comment lines)
- Modify: `THIRD_PARTY_NOTICES.md:12` (one sentence)
- Create: `tests/tile_fixtures.py`
- Test: `tests/test_tileattn.py`

**Interfaces:**
- Consumes:
  - Task 1: `nax.platform_reason()`, `nax.Availability`.
  - Task 3: `HQ`, `HKV`, `GQA`, `D`, `SCALE`, `MAX_RUN`, `N0`, `GRAN`, `MEASURED_SPLITS`, `chunk_for`, `step_of`, `boundaries`, `verify_plan`.
  - int8prefill: `_kernel_text(name: str) -> str` (`functools.cache`d), `_kernels: dict[str, Callable[..., list[Any]]]`, `_error_line(text: str) -> str`.
  - `verifyattn.grouped_attention(queries, keys, values, scale: float, prefix: int, arch: str)`, imported inside `verify()`: verifyattn will import tileattn at module level.
- Produces:
  - `int8prefill._kernel(name: str, input_names: list[str], output_names: list[str], source: str, header: str, *, ensure_row_contiguous: bool = True, compile_options: dict[str, str] | None = None) -> Callable[..., list[Any]]`. Both are passed to `mx.fast.metal_kernel` on every build; int8's calls pass neither, so they get mlx's own defaults.
  - `kernels/attention_tile.metal`: the split body, a line exactly `// REDUCE`, then the reduce body.
  - In tileattn:
    - `_REDUCE_MARK = '// REDUCE\n'` and `_SAFE_MATH = {'math_mode': 'safe'}`.
    - `_kernel_pair() -> tuple[Callable[..., list[Any]], Callable[..., list[Any]]]`.
    - `tile(queries, keys, values, n: int, chunk: int) -> mx.array`: contiguous `[1, T, 24, 256]` bf16.
    - `_arch() -> str`: `@functools.cache`, `str(mx.device_info().get('architecture', ''))`.
    - `in_scope(queries, keys, values, scale: float) -> bool`.
    - `decode(queries, keys, values, splits: int) -> mx.array`: `[1, 24, 1, 256]`.
    - `verify(queries, keys, values, splits: int) -> mx.array`: `[1, 24, T, 256]`.
    - `_compile_ok = False` and `_compile_probe() -> str | None`.
    - `availability() -> nax.Availability`: the platform reason, else `Availability(False, f'the attention tile kernel failed: {line}')`, else `Availability(True)`.
    - Module-level imports: `from sous.engine import nax` and `from sous.engine.int8prefill import _error_line, _kernel, _kernel_text`. `gpucores` (on the first line) and `_model_type` (on the second) join these two lines in Task 7, whose gates first call them: ruff's F401 fails an import nothing uses yet, so they cannot land in this task.
  - In `tests/tile_fixtures.py`: `stand_in_tile(queries, keys, values, n: int, chunk: int) -> mx.array` with `tile`'s contract; `_tree_sum(x, axis: int)`; `serving_qkv(n: int, t: int, *, seed: int = 0) -> tuple[mx.array, mx.array, mx.array]`; `TILE = tileattn.availability()`, read once at import; `nax = pytest.mark.skipif(not TILE.available, reason=f'attention tile unavailable: {TILE.reason}')`.

- [ ] **Step 1: Write the kernel-source tests**

In `tests/test_tileattn.py`, replace the imports:

```python
import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import tileattn  # noqa: E402 — after the importorskip guard
```

with:

```python
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, tileattn  # noqa: E402 — after the importorskip guard
```

Append the kernel-source tests to the end of the file. The kwargs test swaps in a fresh `int8prefill._kernels` dict rather than deleting the two names from the real one: a `monkeypatch.delitem` of a name that was not cached records nothing to restore, so the recording kernels would stay cached for every later test.

```python
# ---- kernel source ---------------------------------------------------------------


def test_the_kernel_file_holds_a_split_body_and_a_reduce_body():
    text = files("sous.engine.kernels").joinpath("attention_tile.metal").read_text()
    assert text.count("\n// REDUCE\n") == 1 and "#include" not in text
    split, reduce = text.split(tileattn._REDUCE_MARK)
    assert "mpp::tensor_ops" in split and "nax_coord(lane)" in split
    # MSL 4.1 gives locals a generic address space: every local cooperative-tensor
    # type is stripped of it, or macOS 27's compiler rejects the operands.
    assert split.count("metal::remove_addrspace_t<decltype(") == 4
    assert "mpp::" not in reduce


def test_both_kernels_are_built_with_safe_math_and_the_split_reads_strides(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs
        return lambda **launch: []

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    tileattn._kernel_pair()
    split = made["sous_attention_tile_split"]
    reduce = made["sous_attention_tile_reduce"]
    assert (split["input_names"], split["output_names"]) == (
        ["q", "k", "v", "params"],
        ["part", "stats"],
    )
    assert (reduce["input_names"], reduce["output_names"]) == (
        ["part", "stats", "params"],
        ["out"],
    )
    assert split["ensure_row_contiguous"] is False and "k_strides" in split["source"]
    assert reduce["ensure_row_contiguous"] is True
    assert split["compile_options"] == reduce["compile_options"] == {"math_mode": "safe"}
    assert "MetalPerformancePrimitives" in split["header"]
    assert "MetalPerformancePrimitives" not in reduce["header"]
    int8prefill._kernel("sous_test_plain", ["x"], ["y"], "y[0] = x[0];", "")
    plain = made["sous_test_plain"]
    assert plain["ensure_row_contiguous"] is True and plain["compile_options"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -v -k "kernel_file or safe_math"`
Expected: FAIL, 2 failed: `FileNotFoundError` for `attention_tile.metal`, and `AttributeError: module 'sous.engine.tileattn' has no attribute '_kernel_pair'`.

- [ ] **Step 3: Add the kernel source**

Create `src/sous/engine/kernels/attention_tile.metal` exactly as below: sous's own code, built from `nax.h`'s lane layout. There is no `#include`: `tileattn` prepends `common.h` + `nax.h` to the split body and `common.h` to the reduce body. Keep `sg / 2` and `sg - rg * 2` as written, and all four `metal::remove_addrspace_t` wrappers (macOS 27's MSL 4.1 rejects the unwrapped operands).

```metal
// Exact GQA-packed attention for decode and verify on the tensor units: bf16,
// 24 query / 4 KV heads, head dim 256, scale 1/16, fp32 probabilities.
// Prepended by tileattn.py with common.h + nax.h (the split body) or common.h (the
// reduce body); the Python side splits this file at the REDUCE marker line.
//
// Split body. Template: TR (rows T, 1..8), MR (= 6*TR packed rows per KV head,
// m = t*6 + h), RGT (ceil(MR/16) 16-row fragment groups). Grid (64*RGT, nsplit, 4),
// threadgroup (64*RGT, 1, 1): one threadgroup per (key split, KV head), two
// simdgroups per 16-row group, each accumulating half of the 256 output dims.
// params = {n, chunk, nsplit}. Row t sees keys [0, n - TR + t]; every K/V read is
// clamped to n - 1 and keys past a row's limit get probability exactly 0, so a row's
// output depends on `chunk` and nothing else about the call.
//
// Head-dim permutation: the QK reduction over D is order-free, so k-step kk /
// fragment column c reads physical dim (c/4)*64 + kk*4 + c%4 for both Q and K (each
// lane owns 64 contiguous dims of every row it touches: 16-byte loads). PV's output
// columns use the same trick: tile nn, half hf, column fn+j is physical dim
// fn*16 + nn*8 + hf*4 + j, so V loads are 16-byte runs and O partials are stored in
// order. q, k and v need innermost stride 1 and 16-byte-aligned rows; only the head
// and token strides are read.
  constexpr int GQ = 6;
  constexpr int DD = 256;
  constexpr float SCALE_LOG2 = 0.0625f * 1.4426950408889634f;
  constexpr float NEG = -1.0e30f;
  typedef float PE;  // fp32 probabilities into the PV matmul

  const ushort lane = ushort(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  // Keep these as a division and a multiply-subtract: written as sg >> 1 and sg & 1 (same values) the
  // compiler emits code that runs 1.4x slower at T >= 3 (57K, M5 Pro, macOS 27.0), same output.
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

  const device bfloat* kbase = k + kvh * k_strides[1] + fn * 16;
  const device bfloat* vbase = v + kvh * v_strides[1] + fn * 16;
  const int kst = int(k_strides[2]);
  const int vst = int(v_strides[2]);

  int lim[2];
  const device bfloat* qp[2];
  UNROLL for (int i = 0; i < 2; ++i) {
    const int m = min(rg * 16 + fm + 8 * i, MR - 1);
    const int t = m / GQ;
    const int h = m - t * GQ;
    lim[i] = prefix + t + 1;
    qp[i] = q + (GQ * kvh + h) * q_strides[1] + t * q_strides[2] + fn * 16;
  }
  // Q staged once in threadgroup memory in lane order; each half stages 8 of its group's 16 k-steps.
  threadgroup bfloat4 qtg[RGT * 1024];
  UNROLL for (int kq = 0; kq < 8; ++kq) {
    const int kk = half_ * 8 + kq;
    UNROLL for (int i = 0; i < 2; ++i) {
      qtg[((rg * 16 + kk) * 2 + i) * 32 + lane] = *reinterpret_cast<const device bfloat4*>(qp[i] + kk * 4);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
  auto ct_q = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_k = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  using ct_q_t = metal::remove_addrspace_t<decltype(ct_q)>;
  using ct_k_t = metal::remove_addrspace_t<decltype(ct_k)>;
  auto ct_s = qk_op.template get_destination_cooperative_tensor<ct_q_t, ct_k_t, float>();

  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
  auto ct_p = pv_op.template get_left_input_cooperative_tensor<PE, bfloat, float>();
  auto ct_v = pv_op.template get_right_input_cooperative_tensor<PE, bfloat, float>();
  using ct_p_t = metal::remove_addrspace_t<decltype(ct_p)>;
  using ct_v_t = metal::remove_addrspace_t<decltype(ct_v)>;
  auto ct_o = pv_op.template get_destination_cooperative_tensor<ct_p_t, ct_v_t, float>();

  float O[4][16];
  UNROLL for (int nn = 0; nn < 4; ++nn) { UNROLL for (int e = 0; e < 16; ++e) O[nn][e] = 0.0f; }
  float rmax[2] = {NEG, NEG};
  float rsum[2] = {0.0f, 0.0f};

  for (int kb0 = k_begin; kb0 < k_end; kb0 += 32) {
    const device bfloat* kr[4];
    const device bfloat* vr[4];
    UNROLL for (int r = 0; r < 4; ++r) {
      const int key = min(kb0 + fm + 8 * r, n - 1);
      kr[r] = kbase + key * kst;
      vr[r] = vbase + key * vst;
    }
    float S[16];
    UNROLL for (int e = 0; e < 16; ++e) S[e] = 0.0f;

    // S = Q K^T: 16 k-steps of 16 (permuted) dims, two k-steps per 16-byte K load
    UNROLL for (int l2 = 0; l2 < 8; ++l2) {
      uint4 kw[4];
      UNROLL for (int r = 0; r < 4; ++r) kw[r] = *reinterpret_cast<const device uint4*>(kr[r] + l2 * 8);
      UNROLL for (int s = 0; s < 2; ++s) {
        const int l = l2 * 2 + s;
        UNROLL for (int i = 0; i < 2; ++i) {
          const bfloat4 qv = bfloat4(qtg[((rg * 16 + l) * 2 + i) * 32 + lane]);
          UNROLL for (int j = 0; j < 4; ++j) ct_q[i * 4 + j] = qv[j];
        }
        UNROLL for (int r = 0; r < 4; ++r) {
          const bfloat4 b = as_type<bfloat4>(s == 0 ? kw[r].xy : kw[r].zw);
          UNROLL for (int j = 0; j < 4; ++j) ct_k[r * 4 + j] = b[j];
        }
        UNROLL for (int e = 0; e < 16; ++e) ct_s[e] = S[e];
        qk_op.run(ct_q, ct_k, ct_s);
        UNROLL for (int e = 0; e < 16; ++e) S[e] = ct_s[e];
      }
    }

    // scale, causal mask (keys past n are clamped copies and always masked), online softmax (exp2)
    UNROLL for (int e = 0; e < 16; ++e) S[e] *= SCALE_LOG2;
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
      UNROLL for (int nn = 0; nn < 4; ++nn) {
        UNROLL for (int hh = 0; hh < 2; ++hh) {
          UNROLL for (int i = 0; i < 2; ++i) {
            UNROLL for (int j = 0; j < 4; ++j) O[nn][hh * 8 + i * 4 + j] *= fac[i];
          }
        }
      }
    }

    // O += P V over the block's two 16-key halves, this simdgroup's 4 output tiles
    UNROLL for (int ks = 0; ks < 2; ++ks) {
      UNROLL for (int e = 0; e < 8; ++e) ct_p[e] = PE(S[ks * 8 + e]);
      UNROLL for (int nn = 0; nn < 4; ++nn) {
        uint4 vw[2];
        UNROLL for (int i = 0; i < 2; ++i) vw[i] = *reinterpret_cast<const device uint4*>(vr[ks * 2 + i] + (nn0 + nn) * 8);
        UNROLL for (int i = 0; i < 2; ++i) {
          const bfloat4 lo = as_type<bfloat4>(vw[i].xy);
          const bfloat4 hi = as_type<bfloat4>(vw[i].zw);
          UNROLL for (int j = 0; j < 4; ++j) { ct_v[i * 4 + j] = lo[j]; ct_v[8 + i * 4 + j] = hi[j]; }
        }
        UNROLL for (int e = 0; e < 16; ++e) ct_o[e] = O[nn][e];
        pv_op.run(ct_p, ct_v, ct_o);
        UNROLL for (int e = 0; e < 16; ++e) O[nn][e] = ct_o[e];
      }
    }
  }

  // partials [kvh][split][m][256] (physical dim order) and {max (log2 units), sum} per row
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

// REDUCE
// Reduce body. Template: MR. Grid (256, MR, 4), threadgroup (256, 1, 1). Combines a
// row's splits in split order (fixed order) and writes out [1, T, 24, 256], the
// model's [B, L, H, D] layout.
  constexpr int DD = 256;
  const int d = int(thread_position_in_threadgroup.x);
  const int m = int(threadgroup_position_in_grid.y);
  const int kvh = int(threadgroup_position_in_grid.z);
  const int nsplit = params[2];
  const device float* st = stats + (size_t(kvh) * size_t(nsplit) * MR + m) * 2;
  float gmax = -1.0e30f;
  for (int s = 0; s < nsplit; ++s) gmax = max(gmax, st[size_t(s) * MR * 2]);
  float num = 0.0f;
  float den = 0.0f;
  const device float* pp = part + (size_t(kvh) * size_t(nsplit) * MR + m) * DD + d;
  for (int s = 0; s < nsplit; ++s) {
    const float w = fast::exp2(st[size_t(s) * MR * 2] - gmax);
    num += w * pp[size_t(s) * MR * DD];
    den += w * st[size_t(s) * MR * 2 + 1];
  }
  const int t = m / 6;
  const int h = m - t * 6;
  out[(size_t(t) * 24 + 6 * kvh + h) * DD + d] = static_cast<bfloat>(num / den);
```

- [ ] **Step 4: Give int8prefill's kernel builder the two keyword-only arguments**

In `src/sous/engine/int8prefill.py`, replace `_kernel` (`:55-78`):

```python
def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str,
) -> Callable[..., list[Any]]:
    """One metal_kernel object per name, built on first use. mlx compiles each
    (source, template) instantiation once per process behind it."""
    kernel = _kernels.get(name)
    if kernel is None:
        import mlx.core as mx

        kernel = cast(
            "Callable[..., list[Any]]",
            mx.fast.metal_kernel(
                name=name,
                input_names=input_names,
                output_names=output_names,
                source=source,
                header=header,
            ),
        )
        _kernels[name] = kernel
```

with:

```python
def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str,
    *,
    ensure_row_contiguous: bool = True,
    compile_options: dict[str, str] | None = None,
) -> Callable[..., list[Any]]:
    """One metal_kernel object per name, built on first use. mlx compiles each
    (source, template) instantiation once per process behind it."""
    kernel = _kernels.get(name)
    if kernel is None:
        import mlx.core as mx

        kernel = cast(
            "Callable[..., list[Any]]",
            mx.fast.metal_kernel(
                name=name,
                input_names=input_names,
                output_names=output_names,
                source=source,
                header=header,
                ensure_row_contiguous=ensure_row_contiguous,
                compile_options=compile_options,
            ),
        )
        _kernels[name] = kernel
```

- [ ] **Step 5: Build the kernel pair in tileattn**

In `src/sous/engine/tileattn.py`, replace the imports:

```python
from __future__ import annotations

import logging

logger = logging.getLogger("sous.engine.tileattn")
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

logger = logging.getLogger("sous.engine.tileattn")
```

Replace the `MEASURED_SPLITS` line, adding the two constants after it:

```python
MEASURED_SPLITS = frozenset({20})
```

with:

```python
MEASURED_SPLITS = frozenset({20})
# attention_tile.metal holds the split body, this line, then the reduce body.
_REDUCE_MARK = "// REDUCE\n"
# mlx's default, stated rather than inherited: parity between the row variants needs
# each to round exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}
```

Append `_kernel_pair()` to the end of the file:

```python
def _kernel_pair() -> tuple[Callable[..., list[Any]], Callable[..., list[Any]]]:
    """The split and reduce kernels, built on first use. The split kernel reads K/V
    in place through their strides: a KVCache hands back a slice of its padded
    buffer, which ensure_row_contiguous would copy whole on every call. Only the
    split kernel uses the tensor ops, so only its header carries nax.h."""
    split_source, reduce_source = _kernel_text("attention_tile.metal").split(_REDUCE_MARK)
    common = _kernel_text("common.h")
    split = _kernel(
        "sous_attention_tile_split",
        ["q", "k", "v", "params"],
        ["part", "stats"],
        split_source,
        common + _kernel_text("nax.h"),
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    reduce = _kernel(
        "sous_attention_tile_reduce",
        ["part", "stats", "params"],
        ["out"],
        reduce_source,
        common,
        compile_options=_SAFE_MATH,
    )
    return split, reduce
```

- [ ] **Step 6: Run the kernel-source tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -v -k "kernel_file or safe_math"`
Expected: PASS, 2 passed.

- [ ] **Step 7: Create the shared test helpers**

Create `tests/tile_fixtures.py`. `stand_in_tile` is a plain-mlx model of the kernel's partition: each row on its own, fixed-order reductions whose pairing depends only on the chunk, so its output depends on the chunk and nothing else about the call, which is the property every exactness test relies on. Each split's partial is rounded through bf16 on purpose: with fp32 partials a chunk change flips about one bf16 element in 6,144, too few for the sensitivity test.

```python
"""Shared helpers for the attention tile's tests: a plain-mlx stand-in for the
Metal kernel, serving-layout inputs, and the `nax` skip marker.

mlx is imported inside each helper, so importing this module needs only what
`tileattn.availability()` itself imports."""

from typing import Any

import pytest

from sous.engine import tileattn

# Read once: a failed compile is not remembered, so each call would compile again.
TILE = tileattn.availability()
nax = pytest.mark.skipif(not TILE.available, reason=f"attention tile unavailable: {TILE.reason}")


def _tree_sum(x: Any, axis: int) -> Any:
    """Sum over `axis` by pairwise halving of a zero-padded power-of-two length: every
    step is an elementwise add, so the pairing depends only on the axis length."""
    import mlx.core as mx

    axis = axis % x.ndim
    size = x.shape[axis]
    width = 1 << max(size - 1, 0).bit_length()
    if width != size:
        pad = list(x.shape)
        pad[axis] = width - size
        x = mx.concatenate([x, mx.zeros(pad, dtype=x.dtype)], axis=axis)
    while width > 1:
        width //= 2
        lo = [slice(None)] * x.ndim
        hi = [slice(None)] * x.ndim
        lo[axis] = slice(0, width)
        hi[axis] = slice(width, 2 * width)
        x = x[tuple(lo)] + x[tuple(hi)]
    return mx.squeeze(x, axis=axis)


def stand_in_tile(queries: Any, keys: Any, values: Any, n: int, chunk: int) -> Any:
    """A plain-mlx model of the kernel's partition, with tileattn.tile's contract.
    Row t of a T-row call sees keys [0, n - T + t]; keys are cut into splits at
    multiples of `chunk`, reads are clamped to n - 1 and masked past the row's
    limit; each row and split keeps an fp32 max, sum and partial output, and the
    splits are combined in split order. Rows never share an op that mixes them (no
    matmul across packed rows) and every reduction's order depends on the chunk
    alone, so a row's output depends on the chunk and nothing else about the call.
    Returns the kernel's contiguous [1, T, 24, 256] bf16."""
    import mlx.core as mx

    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    t = queries.shape[2]
    nsplit = -(-n // chunk)
    prefix = n - t
    neg = mx.array(-float("inf"), dtype=mx.float32)
    rows = []
    for r in range(t):
        limit = prefix + r + 1
        # [1, 4, 6, 1, 256]: the row's six query heads per KV head
        q = queries[:, :, r, :].astype(mx.float32).reshape(1, hkv, gqa, 1, d)
        maxes, sums, partials = [], [], []
        for s in range(nsplit):
            pos = s * chunk + mx.arange(chunk)
            idx = mx.minimum(pos, n - 1)
            k = mx.take(keys, idx, axis=2).astype(mx.float32)[:, :, None]  # [1, 4, 1, c, 256]
            v = mx.take(values, idx, axis=2).astype(mx.float32)[:, :, None]
            score = _tree_sum(q * k, axis=-1) * tileattn.SCALE  # [1, 4, 6, c]
            score = mx.where(pos < limit, score, neg)
            m = mx.max(score, axis=-1, keepdims=True)  # a max is order-free
            ref = mx.where(m == neg, mx.zeros_like(m), m)
            p = mx.exp(score - ref)
            maxes.append(m)
            sums.append(_tree_sum(p, axis=-1)[..., None])
            # Rounded through bf16: with fp32 partials a partition change flips about
            # one element in 6,144 of the bf16 output, too few for a sensitivity test.
            partial = _tree_sum(p[..., None] * v, axis=-2)  # [1, 4, 6, 256]
            partials.append(partial.astype(mx.bfloat16).astype(mx.float32))
        gmax = maxes[0]
        for m in maxes[1:]:
            gmax = mx.maximum(gmax, m)
        num = mx.zeros_like(partials[0])
        den = mx.zeros_like(sums[0])
        for split_max, split_sum, split_out in zip(maxes, sums, partials, strict=True):
            w = mx.exp(split_max - gmax)
            num = num + w * split_out
            den = den + w * split_sum
        rows.append((num / den).reshape(1, 1, tileattn.HQ, d))
    return mx.concatenate(rows, axis=1).astype(mx.bfloat16)


def serving_qkv(n: int, t: int, *, seed: int = 0) -> tuple[Any, Any, Any]:
    """Random bf16 inputs laid out as serving hands them over: queries a transposed
    [1, 24, T, 256] slice of a [1, T, 24, 256] pool (q_norm and rope output), keys
    and values [..., :n, :] slices of a KVCache-style buffer grown in 256-token steps
    with at least one spare row. Built with GPU ops and evaluated."""
    import mlx.core as mx

    cap = -(-(n + 1) // 256) * 256
    q_key, k_key, v_key = mx.random.split(mx.random.key(seed), 3)
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=q_key).astype(mx.bfloat16)
    keys = mx.random.normal((1, tileattn.HKV, cap, tileattn.D), key=k_key).astype(mx.bfloat16)
    values = mx.random.normal((1, tileattn.HKV, cap, tileattn.D), key=v_key).astype(mx.bfloat16)
    mx.eval(pool, keys, values)
    return pool.transpose(0, 2, 1, 3), keys[:, :, :n, :], values[:, :, :n, :]
```

- [ ] **Step 8: Write the entry, stand-in and availability tests**

In `tests/test_tileattn.py`, replace the imports:

```python
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, tileattn  # noqa: E402 — after the importorskip guard
```

with:

```python
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax, tileattn, verifyattn  # noqa: E402 — after the guard
from tests import tile_fixtures as tfx  # noqa: E402
```

Append to the end of the file. The `tile_kernel` fixture runs a test with the stand-in everywhere and again with the real kernel under `tfx.nax`, which skips on the maintainer's M2 and on CI (MetalPerformancePrimitives is missing there) and runs in the M5 gate. The parity tests over T = 1..16 are `slow`: about 25 s together with the stand-in on the M2.

```python
# ---- entries, with the stand-in (and the real kernel where it compiles) ----------


@pytest.fixture(params=[pytest.param("stand-in"), pytest.param("real", marks=tfx.nax)])
def tile_kernel(request, monkeypatch):
    """The plain-mlx stand-in everywhere; the Metal kernel where the tensor units are."""
    if request.param == "stand-in":
        monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    return request.param


def test_in_scope_takes_the_serving_layout():
    assert tileattn.in_scope(*tfx.serving_qkv(2000, 3), tileattn.SCALE)


_OUT_OF_SCOPE = {
    "batch 2": lambda q, k, v, s: (
        mx.concatenate([q, q]),
        mx.concatenate([k, k]),
        mx.concatenate([v, v]),
        s,
    ),
    "16 query heads": lambda q, k, v, s: (q[:, :16], k, v, s),
    "2 KV heads": lambda q, k, v, s: (q, k[:, :2], v[:, :2], s),
    "head dim 128": lambda q, k, v, s: (q[..., :128], k[..., :128], v[..., :128], s),
    "float16": lambda q, k, v, s: (
        q.astype(mx.float16),
        k.astype(mx.float16),
        v.astype(mx.float16),
        s,
    ),
    "float16 queries": lambda q, k, v, s: (q.astype(mx.float16), k, v, s),
    "scale 0.1": lambda q, k, v, s: (q, k, v, 0.1),
    "fewer keys than rows": lambda q, k, v, s: (q, k[:, :, :2], v[:, :, :2], s),
    "unequal K and V": lambda q, k, v, s: (q, k, v[:, :, :-1], s),
    "quantized keys": lambda q, k, v, s: (q, (k, k), v, s),
    "rank 3": lambda q, k, v, s: (q[0], k, v, s),
}


@pytest.mark.parametrize("broken", list(_OUT_OF_SCOPE))
def test_in_scope_refuses_each_broken_property(broken):
    q, k, v = tfx.serving_qkv(2000, 3)
    assert not tileattn.in_scope(*_OUT_OF_SCOPE[broken](q, k, v, tileattn.SCALE))


@pytest.mark.parametrize(
    ("n", "t", "chunk"), [(2000, 3, 96), (2000, 8, 96), (4096, 8, 224), (1537, 5, 64)]
)
def test_a_stand_in_row_is_the_one_row_call_at_its_own_length(n, t, chunk):
    q, k, v = tfx.serving_qkv(n, t)
    out = tfx.stand_in_tile(q, k, v, n, chunk)
    assert out.shape == (1, t, tileattn.HQ, tileattn.D) and out.dtype == mx.bfloat16
    for r in range(t):
        m = n - t + r + 1
        one = tfx.stand_in_tile(q[:, :, r : r + 1], k[:, :, :m], v[:, :, :m], m, chunk)
        assert mx.array_equal(out[:, r], one[:, 0]).item(), r


@pytest.mark.parametrize(("n", "chunk"), [(2000, 96), (4096, 224), (1537, 64)])
def test_a_stand_in_row_moves_with_its_chunk(n, chunk):
    """So an equality between two paths proves they used the same chunk."""
    q, k, v = tfx.serving_qkv(n, 1)
    same_step = tfx.stand_in_tile(q, k, v, n, chunk)
    next_step = tfx.stand_in_tile(q, k, v, n, chunk + tileattn.GRAN)
    assert not mx.array_equal(same_step, next_step).item()


def _assert_rows_equal_decode(splits: int, mark: int, ts) -> None:
    """Every verify row around `mark` equals decode() at its own key count. Pool
    row i is the query at position base + i, which sees base + i + 1 keys; the
    prefixes put a call's rows all below, straddling and all from `mark`."""
    base = mark - 20
    q, k, v = tfx.serving_qkv(mark + 20, 40, seed=mark)
    decoded = {}

    def decode_at(pos):
        if pos not in decoded:
            i = pos - base
            decoded[pos] = tileattn.decode(
                q[:, :, i : i + 1], k[:, :, : pos + 1], v[:, :, : pos + 1], splits
            )
        return decoded[pos]

    for t in ts:
        for prefix in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}):
            n = prefix + t
            i = prefix - base
            out = tileattn.verify(q[:, :, i : i + t], k[:, :, :n], v[:, :, :n], splits)
            assert out.shape == (1, tileattn.HQ, t, tileattn.D) and out.dtype == mx.bfloat16
            for r in range(t):
                same = mx.array_equal(out[:, :, r : r + 1], decode_at(prefix + r)).item()
                assert same, (splits, mark, prefix, t, r)


@pytest.mark.slow
@pytest.mark.parametrize("splits", [10, 20])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_stand_in_verify_rows_equal_decode_rows(monkeypatch, splits, step):
    """At N0 and the first two step boundaries, T = 1..16: straddles, runs cut at
    8 rows, and the rows below N0 against stock one-row attention."""
    monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    mark = tileattn.boundaries(4096, splits)[step]
    _assert_rows_equal_decode(splits, mark, range(1, 17))


@tfx.nax
@pytest.mark.parametrize("mark", [1536, 1921, 2561, 12161, 57601, 131201, 261761])
def test_real_verify_rows_equal_decode_rows_at_step_boundaries(mark):
    assert mark in tileattn.boundaries(262_144, 20)
    _assert_rows_equal_decode(20, mark, range(1, 17))


def test_the_entries_return_the_models_layouts(tile_kernel):
    q, k, v = tfx.serving_qkv(2000, 1)
    assert tileattn.tile(q, k, v, 2000, 96).shape == (1, 1, tileattn.HQ, tileattn.D)
    assert tileattn.decode(q, k, v, 20).shape == (1, tileattn.HQ, 1, tileattn.D)
    for n, t in ((2000, 5), (tileattn.N0 + 2, 5), (2000, 12)):
        q, k, v = tfx.serving_qkv(n, t)
        out = tileattn.verify(q, k, v, 20)
        assert out.shape == (1, tileattn.HQ, t, tileattn.D) and out.dtype == mx.bfloat16


@pytest.mark.parametrize(("n", "t"), [(tileattn.N0, 1), (2000, 3), (1600, 8), (1540, 12)])
def test_reads_stay_below_n(tile_kernel, n, t):
    """NaN planted past n in every KV head, and a buffer that ends exactly at n,
    both give the clean buffer's output."""
    q, k, v = tfx.serving_qkv(n, t)
    want = tileattn.verify(q, k, v, 20)
    nan = mx.full((1, tileattn.HKV, 256, tileattn.D), float("nan"), dtype=mx.bfloat16)
    k_nan = mx.concatenate([k, nan], axis=2)
    v_nan = mx.concatenate([v, nan], axis=2)
    k_end, v_end = mx.contiguous(k), mx.contiguous(v)
    mx.eval(k_nan, v_nan, k_end, v_end)
    assert mx.all(mx.isnan(k_nan[:, :, n:])).item() and not mx.any(mx.isnan(k_nan[:, :, :n])).item()
    for keys, values in ((k_nan[:, :, :n], v_nan[:, :, :n]), (k_end, v_end)):
        got = tileattn.verify(q, keys, values, 20)
        assert mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()


def test_a_wrong_partition_would_differ(monkeypatch):
    """The sensitivity half: a straddling run served as one call at a single chunk,
    and decode at the neighbouring step's chunk, both change the bits, so the
    equalities above are not vacuous."""
    monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    splits = 10
    mark = tileattn.boundaries(4096, splits)[1]
    prefix, t = mark - 3, 4
    runs = tileattn.verify_plan(prefix, t, splits)
    assert [(j, k) for j, k, _ in runs] == [(0, 2), (2, 4)]
    q, k, v = tfx.serving_qkv(prefix + t, t)
    right = tileattn.verify(q, k, v, splits)
    one_call = tfx.stand_in_tile(q, k, v, prefix + t, runs[-1][2]).transpose(0, 2, 1, 3)
    assert not mx.array_equal(one_call[:, :, :2], right[:, :, :2]).item()
    first, _, chunk = tileattn.step_of(mark, splits)
    assert first == mark
    q1, k1, v1 = q[:, :, 1:2], k[:, :, :mark], v[:, :, :mark]
    previous_step = tfx.stand_in_tile(q1, k1, v1, mark, chunk - tileattn.GRAN)
    decoded = tileattn.decode(q1, k1, v1, splits)
    assert not mx.array_equal(decoded, previous_step.transpose(0, 2, 1, 3)).item()


def test_decode_below_n0_is_stock_one_row_attention(monkeypatch):
    def launched(*args):
        raise AssertionError("the tile ran below N0")

    monkeypatch.setattr(tileattn, "tile", launched)
    q, k, v = tfx.serving_qkv(tileattn.N0 - 1, 1)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=tileattn.SCALE, mask=None)
    assert mx.array_equal(tileattn.decode(q, k, v, 20), stock).item()


def test_verify_below_n0_plans_with_its_own_architecture(monkeypatch):
    """verifyattn._ARCH is set only by verifyattn's own probe; the tile reads mlx."""
    seen = []

    def grouped(queries, keys, values, scale, prefix, arch):
        seen.append((prefix, arch, scale))
        return mx.zeros(queries.shape, dtype=queries.dtype)

    monkeypatch.setattr(verifyattn, "grouped_attention", grouped)
    monkeypatch.setattr(verifyattn, "_ARCH", "applegpu_g13s")
    tileattn.verify(*tfx.serving_qkv(1000, 3), 20)
    assert seen == [(997, mx.device_info()["architecture"], tileattn.SCALE)]


def test_arch_is_read_from_mlx_once(monkeypatch):
    tileattn._arch.cache_clear()
    try:
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g17s"})
        assert tileattn._arch() == "applegpu_g17s"
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g14g"})
        assert tileattn._arch() == "applegpu_g17s"
    finally:
        tileattn._arch.cache_clear()


# ---- availability and the compile probe ------------------------------------------


def test_availability_refuses_the_platform_before_compiling(monkeypatch):
    def compiled():
        raise AssertionError("the kernel compiled on a refused platform")

    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(tileattn, "_compile_probe", compiled)
    assert tileattn.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_availability_reports_the_compilers_error_line(monkeypatch):
    line = "program_source:12:3: error: use of undeclared identifier 'x'"
    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(tileattn, "_compile_probe", lambda: line)
    assert tileattn.availability() == nax.Availability(
        False, f"the attention tile kernel failed: {line}"
    )


_TILE_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
    "  x = sg / 2;\n"
)


def test_compile_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_TILE_COMPILER_ERROR)

    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "tile", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.tileattn"):
        reason = tileattn._compile_probe()
    assert reason == "program_source:12:3: error: use of undeclared identifier 'x'"
    assert "x = sg / 2;" in caplog.text


def test_compile_probe_runs_every_row_variant_and_remembers_only_success(monkeypatch):
    rows = []

    def flaky(queries, keys, values, n, chunk):
        rows.append(queries.shape[2])
        if len(rows) == 1:
            raise RuntimeError("transient")
        return mx.zeros((1, queries.shape[2], tileattn.HQ, tileattn.D), dtype=mx.bfloat16)

    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "tile", flaky)
    assert tileattn._compile_probe() == "transient"
    assert tileattn._compile_probe() is None
    assert tileattn._compile_probe() is None
    assert rows == [1, *range(1, tileattn.MAX_RUN + 1)]


# ---- the real kernel (needs the tensor units; skipped where they are absent) -----


@tfx.nax
def test_every_row_variant_compiles_and_runs(monkeypatch):
    monkeypatch.setattr(tileattn, "_compile_ok", False)
    assert tileattn._compile_probe() is None


def _relative_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2))).item()


def _fp32_reference(q, k, v):
    """softmax(q k^T * SCALE) v in float32 on the GPU, from the bf16 inputs; query
    head h reads KV head h // 6, as mlx's GQA does."""
    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    qf = q.astype(mx.float32).reshape(1, hkv, gqa, 1, d)
    kf = k.astype(mx.float32)[:, :, None]
    vf = v.astype(mx.float32)[:, :, None]
    p = mx.softmax((qf @ kf.swapaxes(-1, -2)) * tileattn.SCALE, axis=-1)
    return (p @ vf).reshape(1, tileattn.HQ, 1, d)


@tfx.nax
@pytest.mark.parametrize("n", [2048, 57_344, 196_608])
@pytest.mark.parametrize("gain", [1, 8])
def test_real_kernel_is_no_less_accurate_than_stock(n, gain):
    q, k, v = tfx.serving_qkv(n, 1, seed=n)
    q = q * gain  # exact in bf16: x8 sharpens the softmax
    want = _fp32_reference(q, k, v)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=tileattn.SCALE, mask=None)
    tile_error = _relative_rms(tileattn.decode(q, k, v, 20), want)
    stock_error = _relative_rms(stock, want)
    assert tile_error <= 1.05 * stock_error, (tile_error, stock_error)
```

- [ ] **Step 9: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -v`
Expected: FAIL at collection, importing `tests/tile_fixtures.py`: `AttributeError: module 'sous.engine.tileattn' has no attribute 'availability'`.

- [ ] **Step 10: Add the entries, the compile probe and availability()**

Append to the end of `src/sous/engine/tileattn.py`. `decode()` and `verify()` look up the module-global `tile` at call time, which is how the tests swap in the stand-in.

```python
def tile(queries: Any, keys: Any, values: Any, n: int, chunk: int) -> Any:
    """One kernel call over T <= MAX_RUN rows: row t sees keys [0, n - T + t], every
    K/V read is clamped to n - 1, and keys are split at multiples of `chunk`.
    Queries are [1, 24, T, 256], K/V [1, 4, >= n, 256], all bf16 with innermost
    stride 1. Returns the contiguous [1, T, 24, 256] bf16 output."""
    import mlx.core as mx

    split, reduce = _kernel_pair()
    t = queries.shape[2]
    mr = GQA * t
    rgt = -(-mr // 16)
    nsplit = -(-n // chunk)
    params = mx.array([n, chunk, nsplit], dtype=mx.int32)
    part, stats = split(
        inputs=[queries, keys, values, params],
        template=[("TR", t), ("MR", mr), ("RGT", rgt)],
        grid=(64 * rgt, nsplit, HKV),
        threadgroup=(64 * rgt, 1, 1),
        output_shapes=[(HKV * nsplit * mr * D,), (HKV * nsplit * mr * 2,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = reduce(
        inputs=[part, stats, params],
        template=[("MR", mr)],
        grid=(D, mr, HKV),
        threadgroup=(D, 1, 1),
        output_shapes=[(1, t, HQ, D)],
        output_dtypes=[mx.bfloat16],
    )
    return out


@functools.cache
def _arch() -> str:
    """This process's GPU architecture, for grouped_attention's plan. Read here, never
    from verifyattn._ARCH, which is set only when verifyattn's own probe passed."""
    import mlx.core as mx

    return str(mx.device_info().get("architecture", ""))


def in_scope(queries: Any, keys: Any, values: Any, scale: float) -> bool:
    """What decode() and verify() assume of the projected tensors; both hooks check
    it after the projections."""
    import mlx.core as mx

    return (
        isinstance(queries, mx.array)
        and isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[0] == keys.shape[0] == values.shape[0] == 1
        and queries.shape[1] == HQ
        and keys.shape[1] == values.shape[1] == HKV
        and queries.shape[3] == keys.shape[3] == values.shape[3] == D
        and queries.dtype == keys.dtype == values.dtype == mx.bfloat16
        and queries.shape[2] >= 1
        and keys.shape[2] == values.shape[2] >= queries.shape[2]
        and scale == SCALE
    )


def decode(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The one-row call over all n = keys.shape[2] keys, [1, 24, 1, 256]. Stock one-row
    SDPA below N0, where the decode hook never calls this; the probe and the tests
    compare against it there."""
    import mlx.core as mx

    n = keys.shape[2]
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=SCALE, mask=None)
    return tile(queries, keys, values, n, chunk).transpose(0, 2, 1, 3)


def verify(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The verifier's T-row causal call, row r over keys [0, n - T + r], as
    [1, 24, T, 256]: one tile call per run of verify_plan(), grouped stock calls for
    the runs below N0, concatenated in row order."""
    import mlx.core as mx

    from sous.engine.verifyattn import grouped_attention

    t = queries.shape[2]
    n = keys.shape[2]
    prefix = n - t
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


_compile_ok = False


def _compile_probe() -> str | None:
    """Compile and run every row variant, T = 1..MAX_RUN, on serving-layout inputs;
    None when they all work, else the compiler's error line. The kernel is compiled
    at runtime against the OS's own Metal toolchain, so a new macOS can reject a
    source that compiled before. The inputs are built with GPU ops and evaluated
    first: a compile failure with a CPU-stream op in flight deadlocks inside mlx's
    exception path (0.32.2) instead of raising. Only success is remembered, so one
    bad moment cannot switch the tile off for a long-lived daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        q_key, k_key, v_key = mx.random.split(mx.random.key(0), 3)
        rows = mx.random.normal((1, MAX_RUN, HQ, D), key=q_key).astype(mx.bfloat16)
        # One 256-token step past N0: the K/V are slices of a padded buffer, as served.
        keys = mx.random.normal((1, HKV, N0 + 256, D), key=k_key).astype(mx.bfloat16)
        values = mx.random.normal((1, HKV, N0 + 256, D), key=v_key).astype(mx.bfloat16)
        mx.eval(rows, keys, values)
        queries = rows.transpose(0, 2, 1, 3)
        chunk = chunk_for(N0, min(MEASURED_SPLITS))
        for t in range(1, MAX_RUN + 1):
            mx.eval(tile(queries[:, :, :t], keys[:, :, :N0], values[:, :, :N0], N0, chunk))
    except Exception as e:  # noqa: BLE001 — any failure means the kernel cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("attention tile: the kernel failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares with int8 prefill, then a compile and run
    of every row variant. The `nax` tests skip on this, never on int8's: the two
    kernels can fail to compile on different toolchains."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the attention tile kernel failed: {failed}")
    return nax.Availability(True)
```

- [ ] **Step 11: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -v -rs`
Expected: PASS, 66 passed and 19 skipped on the maintainer's M2, each skip reading `attention tile unavailable: macOS 15.5 < 26.2` (CI's reason names its own macOS). Run it under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py` when another session may be using the GPU.

- [ ] **Step 12: Name the tile in the kernel package's text**

Replace the whole of `src/sous/engine/kernels/__init__.py` (it is only a docstring) with:

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

In `src/sous/engine/kernels/common.h`, replace the first line:

```c
// Shared by both kernels; prepended by int8prefill.py as the metal_kernel header.
```

with:

```c
// Shared by every kernel; prepended by int8prefill.py and tileattn.py as the
// metal_kernel header.
```

In `src/sous/engine/kernels/nax.h`, replace the first three lines:

```c
// Prepended after common.h for the GEMM only: the Metal-4 tensor-op header needs
// macOS 26.2+, and keeping it out of Stage A lets that kernel compile (and CI test
// it) on any Metal GPU.
```

with:

```c
// Prepended after common.h for the tensor-op kernels only, the int8 GEMM and the
// attention tile's split kernel: the Metal-4 tensor-op header needs macOS 26.2+, and
// keeping it out of Stage A and the tile's reduce kernel lets those compile (and CI
// test them) on any Metal GPU.
```

In `THIRD_PARTY_NOTICES.md`, replace the line that ends the oMLX paragraph's first block (`:12`), adding one sentence after it:

```markdown
`mx.fast.metal_kernel` sources with the Q5 and per-group-scaling variants removed.
```

with:

```markdown
`mx.fast.metal_kernel` sources with the Q5 and per-group-scaling variants removed.
`attention_tile.metal` in the same directory is sous's own work and is not derived
from oMLX.
```

- [ ] **Step 13: Run the kernel-package and neighbouring tests**

Run: `uv run pytest tests/test_int8prefill.py tests/test_verifyattn.py tests/test_nax.py -v`
Expected: PASS, 128 passed and 8 skipped (int8's `nax` tests). `test_kernel_sources_ship_as_package_files` still holds: the tensor-op include stays in `nax.h` only, and no comment added to `common.h` names it.

- [ ] **Step 14: Lint, type-check and commit**

The comment edits to `common.h` and `nax.h` and the new `.metal` file change `forkstore.engine_epoch()`, which hashes `kernels/*.metal` and `*.h`, so landing the branch cold-starts every fork store once; Task 9 and the PR text already account for it.

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add THIRD_PARTY_NOTICES.md src/sous/engine/int8prefill.py src/sous/engine/kernels/__init__.py src/sous/engine/kernels/attention_tile.metal src/sous/engine/kernels/common.h src/sous/engine/kernels/nax.h src/sous/engine/tileattn.py tests/test_tileattn.py tests/tile_fixtures.py
git commit -m "feat(engine): add the tensor-unit attention tile kernel with its decode and verify entries" -m "Verify attention is most of decode's context-dependent cost, and the M5's
tensor units cut it by 2-2.6x at T = 3 when the six query heads of a KV head
are packed with the rows into MPP fragments. The split kernel reads K/V
through their strides so a KVCache's padded buffer is never copied, both
kernels state safe math, and availability() compiles and runs every row
variant, so the nax tests skip wherever the toolchain rejects the source.
CI's GPU cannot compile it: a plain-mlx stand-in with the same partition
contract carries the exactness tests there.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: The decode hook, the verify branch and the call counters

**Files:**
- Modify: `src/sous/engine/tileattn.py` (the flag, the hook mark, the counters, the decode hook and `serve_verify`)
- Modify: `src/sous/engine/verifyattn.py:22-24` (the docstring's T = 2 paragraph), `:38` (the import), `:317-327` (`_in_scope`), `:360-389` (`_stock_verify` and the tile branch in `_wrap`)
- Modify: `tests/tile_fixtures.py` (the tiny model, its prefill and verify forward, the hook installers)
- Test: `tests/test_tileattn.py`
- Test: `tests/test_verifyattn.py:354-380` (`_scope` gains `splits: int = 0` and `dtype=mx.float32`) and the new tile-active scope tests, inserted before `_fresh_model` at `:423`

**Interfaces:**
- Consumes:
  - Task 4: `tile`, `decode`, `verify`, `in_scope`, `_arch`, `tfx.stand_in_tile`, `tfx.serving_qkv`.
  - Task 3: the constants and `verify_plan`.
  - verifyattn (existing): `_TAG`, `_KVCACHE`, `_ARCH`, `calls` (keys `'grouped'`, `'loop'`, `'original'`), `grouped_attention`, `_row_loop(queries, keys, values, cache, scale)`, `_post_ok(queries, keys, values, length) -> bool`, `install_wrapper()`, `_attention_modules(model) -> list`, and the `_wrap(cls)` body at `:360-393`.
- Produces:
  - In tileattn:
    - `calls: dict[str, int]` with exactly the keys `decode_tile`, `decode_stock`, `decode_out`, `verify_calls`, `verify_rows_tile`, `verify_rows_stock`, `verify_launches`, `verify_straddles`, `verify_declined`, `verify_out`, `verify_t1` … `verify_t8` and `verify_t9_up`, all starting at 0.
    - `_active_splits = 0`, `clear() -> None` and `_HOOK_MARK = '_sous_attention_tile'`.
    - `_bind(queries, keys, values, cache, scale, mask, sinks=None) -> tuple`, mirroring mlx-vlm base's parameter list.
    - `_plain_kv_cache(cache: Any) -> bool`, `_serve_decode(args: tuple, kwargs: dict) -> Any | None`.
    - `_decode_wrapper(original: Callable[..., Any]) -> Callable[..., Any]`: `functools.wraps`, with `_HOOK_MARK` set True on the wrapper.
    - `install_decode_hook() -> None`: idempotent; replaces `mlx_vlm.models.qwen3_5.language.scaled_dot_product_attention` unless the current value carries `_HOOK_MARK`.
    - `serve_verify(queries, keys, values, splits: int) -> mx.array`: counts, then returns `verify(...)`.
  - Counting rules:
    - `decode_tile`: a one-row call the tile served; `decode_stock`: an in-scope one-row call below N0, which the original function answers; `decode_out`: a one-row call while the flag is set but out of scope. Multi-row (prefill) calls and calls while the flag is 0 are not counted.
    - `serve_verify`: +1 to `verify_calls` and to `verify_t{T}` (`verify_t9_up` for T ≥ 9); rows in tile runs to `verify_rows_tile` and rows in stock runs to `verify_rows_stock`; tile runs to `verify_launches`; +1 to `verify_straddles` when the plan has more than one run.
    - verifyattn: `tileattn.calls['verify_declined']` for a post-projection decline while the flag is set, and `tileattn.calls['verify_out']` for a tagged module the pre-projection scope refuses while the flag is set.
  - In verifyattn: `from sous.engine import tileattn` at module level; `_in_scope(verifier, attention, x, mask, cache, splits: int = 0) -> bool`; `_stock_verify(queries, keys, values, cache, scale: float, length: int) -> mx.array`; `_wrap`'s `_attention` reads `splits = tileattn._active_splits` once per call.
  - In `tests/tile_fixtures.py`: `VOCAB = 512`; `tiny_language_model(*, seed: int = 0, heads: int = 24, kv_heads: int = 4)`; `prefill_cache(lm, ids: list[int]) -> list`; `verify_forward(lm, cache, tokens: list[int]) -> list`; `hooked(monkeypatch, *, splits: int = 20, stand_in: bool = True) -> None`; `verify_ready(monkeypatch, lm) -> list`.

- [ ] **Step 1: Add the tiny model and the hook installers to the shared helpers**

In `tests/tile_fixtures.py`, replace the docstring's first paragraph:

```python
"""Shared helpers for the attention tile's tests: a plain-mlx stand-in for the
Metal kernel, serving-layout inputs, and the `nax` skip marker.
```

with:

```python
"""Shared helpers for the attention tile's tests: a plain-mlx stand-in for the
Metal kernel, serving-layout inputs, the `nax` skip marker, a tiny qwen3_5 model
at the 27B's attention shape, and the installers for both hooks.
```

Replace the sous import, adding `verifyattn` and `VOCAB`:

```python
from sous.engine import tileattn
```

with:

```python
from sous.engine import tileattn, verifyattn

VOCAB = 512
```

Append to the end of the file. `hooked` installs the decode hook only through `monkeypatch`, so the language module's global is restored after every test; `verify_ready` tags with `object.__setattr__`, as verifyattn's `enable()` does, which keeps the tag out of mlx's parameter tree.

```python
def tiny_language_model(*, seed: int = 0, heads: int = 24, kv_heads: int = 4) -> Any:
    """A random qwen3_5 language model with two full-attention layers (1 and 3) at
    the 27B's attention shape by default, bf16, unquantized."""
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    args = TextConfig(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=heads,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=kv_heads,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,
    )
    mx.random.seed(seed)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    return lm


def prefill_cache(lm: Any, ids: list[int]) -> list:
    """A fresh cache holding `ids`, prefilled in one call. Explicit positions and a
    zero delta, as the VLM engine hands them: the bare language model cannot derive
    rope positions without a vision config."""
    import mlx.core as mx

    cache = lm.make_cache()
    lm(
        mx.array([ids], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(len(ids), dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    )
    mx.eval([c.state for c in cache])
    return cache


def verify_forward(lm: Any, cache: list, tokens: list[int]) -> list:
    """One verify forward as DFlash runs it, then the speculative round aborted so
    the cache is back where it was: the logits and three layers' hidden states."""
    import mlx.core as mx

    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=[0, 1, 2],
        speculative_verify=True,
    )
    arrays = [out.logits, *out.hidden_states]
    mx.eval(arrays)
    out.gdn_states.abort()
    return arrays


def hooked(monkeypatch: pytest.MonkeyPatch, *, splits: int = 20, stand_in: bool = True) -> None:
    """The decode hook installed and the flag set for this test only; monkeypatch
    restores the language module's global and the flag afterwards."""
    import importlib

    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    current = language.scaled_dot_product_attention
    if not getattr(current, tileattn._HOOK_MARK, False):
        monkeypatch.setattr(
            language, "scaled_dot_product_attention", tileattn._decode_wrapper(current)
        )
    monkeypatch.setattr(tileattn, "_active_splits", splits)
    if stand_in:
        monkeypatch.setattr(tileattn, "tile", stand_in_tile)


def verify_ready(monkeypatch: pytest.MonkeyPatch, lm: Any) -> list:
    """verifyattn's wrapper installed and `lm`'s full-attention modules tagged, as
    its enable() leaves them. Tags stay on the modules; a test that needs them off
    sets them to 0."""
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    verifyattn.install_wrapper()
    monkeypatch.setattr(verifyattn, "_ARCH", mx.device_info()["architecture"])
    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)
    modules = verifyattn._attention_modules(lm)
    for module in modules:
        object.__setattr__(module, verifyattn._TAG, tileattn.GQA)
    return modules
```

- [ ] **Step 2: Write the decode-hook tests**

In `tests/test_tileattn.py`, replace the imports:

```python
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax, tileattn, verifyattn  # noqa: E402 — after the guard
from tests import tile_fixtures as tfx  # noqa: E402
```

with:

```python
import importlib
import inspect
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax, tileattn, verifyattn  # noqa: E402 — after the guard
from tests import tile_fixtures as tfx  # noqa: E402
```

Append to the end of the file. Each refused call is compared bitwise with mlx-vlm's own function; the quantized cache is checked against a recording original instead, because mlx-vlm's quantized path needs quantized K/V.

```python
# ---- the decode hook -------------------------------------------------------------


def _tile_counts_since(before):
    """The counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in tileattn.calls.items() if n != before[k]}


def _language_module():
    return importlib.import_module("mlx_vlm.models.qwen3_5.language")


def _kv_cache_with(**attrs):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    for name, value in attrs.items():
        setattr(cache, name, value)
    return cache


def _subclass_cache():
    from mlx_vlm.models.cache import KVCache

    class Subclass(KVCache):
        pass

    return Subclass()


def _decode_call(n=2000, t=1, *, queries=None, dtype=None, cache=None, **kwargs):
    q, k, v = tfx.serving_qkv(n, t)
    if dtype is not None:
        q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    cache = _kv_cache_with() if cache is None else cache
    call = {"cache": cache, "scale": tileattn.SCALE, "mask": None, **kwargs}
    return (q if queries is None else queries(q), k, v), call


# case -> (the call, the counter it moves; None moves none)
_PASSED_THROUGH = {
    "causal mask": (lambda: _decode_call(mask="causal"), "decode_out"),
    "array mask": (
        lambda: _decode_call(mask=mx.ones((1, 1, 1, 2000), dtype=mx.bool_)),
        "decode_out",
    ),
    "sinks": (lambda: _decode_call(sinks=mx.zeros((24,), dtype=mx.bfloat16)), "decode_out"),
    "float16": (lambda: _decode_call(dtype=mx.float16), "decode_out"),
    "scale 0.1": (lambda: _decode_call(scale=0.1), "decode_out"),
    "16 query heads": (lambda: _decode_call(queries=lambda q: q[:, :16]), "decode_out"),
    "KVCache subclass": (lambda: _decode_call(cache=_subclass_cache()), "decode_out"),
    "left padding": (
        lambda: _decode_call(cache=_kv_cache_with(left_padding=mx.array([1]))),
        "decode_out",
    ),
    "decode left padding": (
        lambda: _decode_call(cache=_kv_cache_with(_qwen3_5_decode_left_padding=[1])),
        "decode_out",
    ),
    "flag clear": (lambda: _decode_call(), None),
    "two rows": (lambda: _decode_call(t=2), None),
    "below N0": (lambda: _decode_call(n=tileattn.N0 - 1), "decode_stock"),
}


def test_the_decode_hook_serves_a_plain_one_row_call(monkeypatch):
    tfx.hooked(monkeypatch)
    q, k, v = tfx.serving_qkv(2000, 1)
    before = dict(tileattn.calls)
    got = _language_module().scaled_dot_product_attention(
        q, k, v, cache=_kv_cache_with(), scale=tileattn.SCALE, mask=None
    )
    assert _tile_counts_since(before) == {"decode_tile": 1}
    assert mx.array_equal(got, tileattn.decode(q, k, v, 20)).item()


@pytest.mark.parametrize("case", list(_PASSED_THROUGH))
def test_the_decode_hook_hands_every_other_call_to_mlx_vlm(monkeypatch, case):
    from mlx_vlm.models.base import scaled_dot_product_attention as original

    build, counter = _PASSED_THROUGH[case]
    tfx.hooked(monkeypatch, splits=0 if case == "flag clear" else 20)
    args, kwargs = build()
    before = dict(tileattn.calls)
    got = _language_module().scaled_dot_product_attention(*args, **kwargs)
    assert _tile_counts_since(before) == ({} if counter is None else {counter: 1})
    assert mx.array_equal(got, original(*args, **kwargs)).item()


def test_the_decode_hook_hands_a_quantized_cache_over_untouched(monkeypatch):
    """mlx-vlm's quantized path needs quantized K/V, so this one is checked against a
    recording original: the same arguments, and its answer returned."""
    seen = []

    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return "mlx-vlm's answer"

    wrapped = tileattn._decode_wrapper(original)
    assert inspect.unwrap(wrapped) is original and getattr(wrapped, tileattn._HOOK_MARK)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    q, k, v = tfx.serving_qkv(2000, 1)
    cache = _kv_cache_with(bits=4)
    before = dict(tileattn.calls)
    assert wrapped(q, k, v, cache=cache, scale=tileattn.SCALE, mask=None) == "mlx-vlm's answer"
    ((args, kwargs),) = seen
    assert all(a is b for a, b in zip(args, (q, k, v), strict=True))
    assert kwargs == {"cache": cache, "scale": tileattn.SCALE, "mask": None}
    assert _tile_counts_since(before) == {"decode_out": 1}


def test_install_decode_hook_is_idempotent(monkeypatch):
    from mlx_vlm.models.base import scaled_dot_product_attention as original

    language = _language_module()
    # Registers the restore: the replacement must not outlive this test.
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    tileattn.install_decode_hook()
    first = language.scaled_dot_product_attention
    tileattn.install_decode_hook()
    assert language.scaled_dot_product_attention is first
    assert getattr(first, tileattn._HOOK_MARK) and inspect.unwrap(first) is original


@pytest.fixture(scope="module")
def tiny_verifier():
    """The tiny model, its token ids, and prefilled caches by prefix. A verify
    forward rolls its cache back, so those caches are shared; a test that decodes
    builds its own."""
    lm = tfx.tiny_language_model()
    ids = mx.random.randint(0, tfx.VOCAB, (2000,), key=mx.random.key(7)).tolist()
    return lm, ids, {}


def test_a_real_attention_module_reaches_the_tile_once(tiny_verifier, monkeypatch):
    """One-row decode through Qwen3_5Attention.__call__ itself: its projections, rope
    and cache append, then the replaced global."""
    lm, ids, _ = tiny_verifier
    index = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    module = lm.layers[index].self_attn
    stock_cache = tfx.prefill_cache(lm, ids[:1700])[index]
    tile_cache = tfx.prefill_cache(lm, ids[:1700])[index]
    x = mx.random.normal((1, 1, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    tfx.hooked(monkeypatch, splits=0)
    want = module(x, mask=None, cache=stock_cache)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    got = module(x, mask=None, cache=tile_cache)
    mx.eval(want, got)
    assert stock_cache.offset == tile_cache.offset == 1701
    assert _tile_counts_since(before) == {"decode_tile": 1}
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    assert (mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2))).item() <= 1e-2
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -v -k "decode_hook or real_attention_module"`
Expected: FAIL, 16 failed: 14 with `AttributeError: module 'sous.engine.tileattn' has no attribute '_HOOK_MARK'` (raised in `tfx.hooked`), one with `... has no attribute '_decode_wrapper'` and one with `... has no attribute 'install_decode_hook'`.

- [ ] **Step 4: Add the flag, the counters and the decode hook**

In `src/sous/engine/tileattn.py`, replace the first two imports, adding `importlib`:

```python
import functools
import logging
```

with:

```python
import functools
import importlib
import logging
```

Replace the `_SAFE_MATH` line, adding the flag, the hook mark and the counters after it:

```python
_SAFE_MATH = {"math_mode": "safe"}
```

with:

```python
_SAFE_MATH = {"math_mode": "safe"}

# The active split target S, or 0: set only by an enable() that reached active,
# cleared first by every enable() and by clear(). A plain int, so nothing it holds
# outlives an unload.
_active_splits = 0
# Set on the decode wrapper so a second install never wraps it again.
_HOOK_MARK = "_sous_attention_tile"
# Per attention call (one layer of one forward), not per turn: the tests read deltas
# and the merge gate exports them. Nothing is logged per turn.
calls: dict[str, int] = {
    "decode_tile": 0,  # one-row call served by the kernel (n >= N0)
    "decode_stock": 0,  # one-row call in scope below N0: mlx-vlm's function answered
    "decode_out": 0,  # one-row call while active but out of scope: mlx-vlm's function
    "verify_calls": 0,
    "verify_rows_tile": 0,
    "verify_rows_stock": 0,  # rows below N0, through grouped_attention
    "verify_launches": 0,  # tile runs
    "verify_straddles": 0,  # calls cut into more than one run
    "verify_declined": 0,  # in scope before the projections, refused after them
    "verify_out": 0,  # tagged verifier calls while active but out of scope
    "verify_t1": 0,
    "verify_t2": 0,
    "verify_t3": 0,
    "verify_t4": 0,
    "verify_t5": 0,
    "verify_t6": 0,
    "verify_t7": 0,
    "verify_t8": 0,
    "verify_t9_up": 0,
}
```

Append to the end of the file. `language` is annotated `Any` because ty resolves the literal module and rejects assigning the wrapper to a function-typed attribute; `setattr` with a constant name would trip ruff's B010 instead.

```python
def clear() -> None:
    """Stop serving: both hooks pass every call through from here on."""
    global _active_splits
    _active_splits = 0


def _bind(queries, keys, values, cache, scale, mask, sinks=None) -> tuple:
    """mlx-vlm's base.scaled_dot_product_attention parameter list, which verifyattn's
    source gate pins."""
    return queries, keys, values, cache, scale, mask, sinks


def _plain_kv_cache(cache: Any) -> bool:
    """mlx-vlm's single-sequence KVCache itself (its subclasses keep other layouts),
    unquantized and without left padding. The left-padded helpers hand the global a
    None or batch cache, so they never pass."""
    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    return (
        type(cache) is kv_cache
        and not hasattr(cache, "bits")
        and getattr(cache, "_qwen3_5_decode_left_padding", None) is None
        and getattr(cache, "left_padding", None) is None
    )


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


def _decode_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """mlx-vlm's attention function with the one-row calls the tile serves taken
    out; every other call reaches `original` with its own arguments."""

    @functools.wraps(original)
    def scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Any:
        out = _serve_decode(args, kwargs)
        return original(*args, **kwargs) if out is None else out

    setattr(scaled_dot_product_attention, _HOOK_MARK, True)
    return scaled_dot_product_attention


def install_decode_hook() -> None:
    """Replace the global Qwen3_5Attention.__call__ calls after its projections; the
    module has no other seam there. Idempotent. enable() installs it on the first
    activation only, so a process that never activates the tile runs mlx-vlm's
    function untouched, and the wrapper passes every call through while the flag
    is clear, so an unload leaves it inert."""
    language: Any = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    current = language.scaled_dot_product_attention
    if not getattr(current, _HOOK_MARK, False):
        language.scaled_dot_product_attention = _decode_wrapper(current)


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
    return verify(queries, keys, values, splits)
```

- [ ] **Step 5: Run the decode-hook tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -v -k "decode_hook or real_attention_module"`
Expected: PASS, 16 passed.

- [ ] **Step 6: Write the verify-branch tests**

In `tests/test_verifyattn.py`, replace `_scope` (`:354-380`), giving it the flag and the hidden states' dtype; the defaults keep every existing scope test on today's table:

```python
def _scope(
    monkeypatch,
    *,
    t=3,
    batch=1,
    mask="causal",
    cache=None,
    head_dim=256,
    tag=GQA,
    heads=24,
    kv_heads=4,
):
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)
    attention = types.SimpleNamespace(
        head_dim=head_dim, num_attention_heads=heads, num_key_value_heads=kv_heads
    )
    setattr(attention, verifyattn._TAG, tag)
    return verifyattn._in_scope(
        Qwen3_5BatchInvariantForward(),
        attention,
        mx.zeros((batch, t, 8)),
        mask,
        KVCache() if cache is None else cache,
    )
```

with:

```python
def _scope(
    monkeypatch,
    *,
    t=3,
    batch=1,
    mask="causal",
    cache=None,
    head_dim=256,
    tag=GQA,
    heads=24,
    kv_heads=4,
    splits=0,
    dtype=mx.float32,
):
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)
    attention = types.SimpleNamespace(
        head_dim=head_dim, num_attention_heads=heads, num_key_value_heads=kv_heads
    )
    setattr(attention, verifyattn._TAG, tag)
    return verifyattn._in_scope(
        Qwen3_5BatchInvariantForward(),
        attention,
        mx.zeros((batch, t, 8), dtype=dtype),
        mask,
        KVCache() if cache is None else cache,
        splits,
    )
```

Replace the `def _fresh_model():` line, adding the tile-active scope tests before it:

```python
def _fresh_model():
```

with:

```python
# The tile active at the M5 Pro's split target, over a bf16 model's hidden states.
TILE_ACTIVE = {"splits": 20, "dtype": mx.bfloat16}


def test_scope_with_the_tile_active_takes_every_t_from_one(monkeypatch):
    assert _scope(monkeypatch, t=1, mask=None, **TILE_ACTIVE)
    assert _scope(monkeypatch, t=2, **TILE_ACTIVE)
    assert _scope(monkeypatch, t=16, **TILE_ACTIVE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"t": 1},  # one row under a causal mask: not what the verifier builds
        {"mask": None},
        {"dtype": mx.float32},
        {"batch": 2},
        {"mask": "left_padded_decode"},
        {"head_dim": 128},
        {"tag": 0},
        {"tag": 4},
    ],
)
def test_scope_with_the_tile_active_refuses_everything_else(monkeypatch, overrides):
    assert not _scope(monkeypatch, **{**TILE_ACTIVE, **overrides})


def test_scope_with_the_tile_active_refuses_array_masks_and_other_caches(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    class Subclass(KVCache):
        pass

    quantized = KVCache()
    quantized.bits = 4  # ty: ignore[unresolved-attribute]
    padded = KVCache()
    padded._qwen3_5_decode_left_padding = [1]  # ty: ignore[unresolved-attribute]
    assert not _scope(monkeypatch, mask=mx.ones((1, 1, 3, 3), dtype=mx.bool_), **TILE_ACTIVE)
    for cache in (Subclass(), quantized, padded):
        assert not _scope(monkeypatch, cache=cache, **TILE_ACTIVE), type(cache).__name__


def test_scope_without_the_tile_keeps_three_to_eight_rows_for_bf16_too(monkeypatch):
    bf16 = {"dtype": mx.bfloat16}
    assert _scope(monkeypatch, t=3, **bf16) and _scope(monkeypatch, t=8, **bf16)
    assert not _scope(monkeypatch, t=1, mask=None, **bf16)
    assert not _scope(monkeypatch, t=2, **bf16)
    assert not _scope(monkeypatch, t=9, **bf16)


def _fresh_model():
```

Append the forward-level tests to the end of `tests/test_tileattn.py`. They run mlx-vlm's real exact verifier on the tiny model; the plan-following test is `slow` (about 11 s on the M2).

```python
# ---- the verify branch, through mlx-vlm's exact verifier -------------------------


def _prefilled_cache(verifier, prefix):
    lm, ids, caches = verifier
    if prefix not in caches:
        caches[prefix] = tfx.prefill_cache(lm, ids[:prefix])
    return caches[prefix]


@pytest.mark.slow
@pytest.mark.parametrize("prefix", [1000, 1530, 1700, 1915])
def test_verify_rows_follow_the_plan(tiny_verifier, monkeypatch, prefix):
    """Below N0, above it, and straddling N0 and the step at 1921, T = 1..16: each of
    the two full-attention layers sends one call, cut as verify_plan says."""
    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    tfx.hooked(monkeypatch)
    cache = _prefilled_cache(tiny_verifier, prefix)
    for t in range(1, 17):
        runs = tileattn.verify_plan(prefix, t, 20)
        tile_rows = sum(k - j for j, k, chunk in runs if chunk)
        expected = {
            "verify_calls": 2,
            f"verify_t{t}" if t <= tileattn.MAX_RUN else "verify_t9_up": 2,
            "verify_rows_tile": 2 * tile_rows,
            "verify_rows_stock": 2 * (t - tile_rows),
            "verify_launches": 2 * sum(1 for *_, chunk in runs if chunk),
            "verify_straddles": 2 * (len(runs) > 1),
        }
        before = dict(tileattn.calls)
        tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
        assert _tile_counts_since(before) == {k: n for k, n in expected.items() if n}, (prefix, t)
        assert all(c.offset == prefix for c in cache if hasattr(c, "offset"))


def test_an_untagged_verifier_never_reaches_the_tile(tiny_verifier, monkeypatch):
    lm, ids, _ = tiny_verifier
    for module in tfx.verify_ready(monkeypatch, lm):
        object.__setattr__(module, verifyattn._TAG, 0)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700:1703]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {}
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))


def test_a_tagged_verify_the_scope_refuses_is_counted_out_of_scope(tiny_verifier, monkeypatch):
    """The gate requires none of these in a served turn, so they must be visible."""

    class OtherCache:
        pass

    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    monkeypatch.setattr(verifyattn, "_KVCACHE", OtherCache)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700:1703]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before, original = dict(tileattn.calls), verifyattn.calls["original"]
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {"verify_out": 2}
    assert verifyattn.calls["original"] - original == 2
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))


@pytest.mark.parametrize(("owner", "check"), [(verifyattn, "_post_ok"), (tileattn, "in_scope")])
@pytest.mark.parametrize("t", [1, 2, 3, 5])
def test_a_declined_verify_computes_todays_attention(tiny_verifier, monkeypatch, owner, check, t):
    """After the projections the rows are already appended, so a decline must
    reproduce the stock verifier (T = 1, 2) or the grouped path (T >= 3) itself."""
    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700 : 1700 + t]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(owner, check, lambda *a: False)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {"verify_declined": 2}
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))
```

- [ ] **Step 7: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py -v -k scope`
Expected: FAIL, 23 failed, each with `TypeError: _in_scope() takes 5 positional arguments but 6 were given`.

Run: `uv run pytest tests/test_tileattn.py -v -k "follow_the_plan or untagged or out_of_scope or declined"`
Expected: FAIL, 13 failed and 1 passed: the counter deltas are empty (`assert {} == {'verify_calls': 2, ...}`) because nothing sends rows to the tile yet. The untagged test already passes; it proves the tile stays out of an inactive verifier.

- [ ] **Step 8: Add the tile branch to verifyattn**

In `src/sous/engine/verifyattn.py`, replace the docstring's last paragraph (`:22-24`):

```python
T = 2 is left alone. The stock verifier already makes one call there, and
grouping it would change output where rows straddle a plan transition.
"""
```

with:

```python
While the attention tile (tileattn) is inactive, T = 2 is left alone: the stock
verifier already makes one call there, and grouping it would change output where
rows straddle a plan transition. While it is active, every T from 1 goes to
tileattn, whose rows below its N0 take grouped_attention, T = 2 included, so that
they equal the one-row decode at their own lengths.
"""
```

Replace the int8prefill import (`:38`), importing tileattn at module level too (tileattn imports verifyattn only inside `verify()`, so there is no cycle):

```python
from sous.engine.int8prefill import _model_type, _root
```

with:

```python
from sous.engine import tileattn
from sous.engine.int8prefill import _model_type, _root
```

Replace the head of `_in_scope` (`:317-327`); everything from the cache-type check down is unchanged:

```python
def _in_scope(verifier: Any, attention: Any, x: Any, mask: Any, cache: Any) -> bool:
    """Whether this call takes the grouped path. Decided before any projection:
    _prepare_projected_qkv appends the verify rows to the KV cache, so after it
    the original method could only append them a second time."""
    gqa = getattr(attention, _TAG, 0)
    if not gqa or _KVCACHE is None:
        return False
    if x.ndim != 3 or x.shape[0] != 1 or not (MIN_ROWS <= x.shape[1] <= MAX_ROWS):
        return False
    if not (isinstance(mask, str) and mask == "causal"):
        return False
```

with:

```python
def _in_scope(
    verifier: Any, attention: Any, x: Any, mask: Any, cache: Any, splits: int = 0
) -> bool:
    """Whether this call leaves mlx-vlm's method. Decided before any projection:
    _prepare_projected_qkv appends the verify rows to the KV cache, so after it
    the original method could only append them a second time. While the tile is
    active (splits > 0) every T from 1 is taken, bf16 only, with the mask the
    verifier builds for it: None for one row, "causal" above."""
    gqa = getattr(attention, _TAG, 0)
    if not gqa or _KVCACHE is None:
        return False
    if x.ndim != 3 or x.shape[0] != 1:
        return False
    rows = x.shape[1]
    if splits:
        import mlx.core as mx

        if rows < 1 or x.dtype != mx.bfloat16:
            return False
        if not (mask is None if rows == 1 else (isinstance(mask, str) and mask == "causal")):
            return False
    elif not (MIN_ROWS <= rows <= MAX_ROWS and isinstance(mask, str) and mask == "causal"):
        return False
```

Replace the head of `_wrap` (`:360-371`) with `_stock_verify` and a head that reads the flag once per call. `_stock_verify` copies the stock verifier's own T = 2 and T = 1 calls (`speculative_verifier.py:114-129` and `:154-162`), which `_attention`'s source hash already covers:

```python
def _wrap(cls: Any) -> None:
    import mlx.core as mx

    orig = cls._attention

    # functools.wraps: the source gate follows __wrapped__ back to mlx-vlm's body.
    @functools.wraps(orig)
    def _attention(self, attention, x, mask, cache, position_ids, position_embeddings):
        if not _in_scope(self, attention, x, mask, cache):
            if getattr(attention, _TAG, 0):
                calls["original"] += 1
            return orig(self, attention, x, mask, cache, position_ids, position_embeddings)
```

with:

```python
def _stock_verify(
    queries: Any, keys: Any, values: Any, cache: Any, scale: float, length: int
) -> Any:
    """Today's attention for a call the tile declined after the projections, which
    have already appended the rows, so the original method cannot be re-entered:
    grouped (or mlx-vlm's row loop) from T = 3, and below that the stock verifier's
    own single call, bool-masked at T = 2 and unmasked at T = 1."""
    import mlx.core as mx
    from mlx_vlm.models.base import kv_sequence_length, scaled_dot_product_attention

    if length >= MIN_ROWS:
        if _post_ok(queries, keys, values, length):
            calls["grouped"] += 1
            prefix = keys.shape[-2] - length
            return grouped_attention(queries, keys, values, scale, prefix, _ARCH)
        calls["loop"] += 1
        return _row_loop(queries, keys, values, cache, scale)
    mask = None
    if length == 2:
        key_length = kv_sequence_length(keys)
        prefix = key_length - length
        mask = (
            mx.arange(key_length)[None, None, None, :]
            < (prefix + mx.arange(length) + 1)[None, None, :, None]
        )
    return scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)


def _wrap(cls: Any) -> None:
    import mlx.core as mx

    orig = cls._attention

    # functools.wraps: the source gate follows __wrapped__ back to mlx-vlm's body.
    @functools.wraps(orig)
    def _attention(self, attention, x, mask, cache, position_ids, position_embeddings):
        splits = tileattn._active_splits
        if not _in_scope(self, attention, x, mask, cache, splits):
            if getattr(attention, _TAG, 0):
                calls["original"] += 1
                if splits:
                    tileattn.calls["verify_out"] += 1
            return orig(self, attention, x, mask, cache, position_ids, position_embeddings)
```

Replace the post-projection branch (`:382-389`); the flag-0 arm is today's code, byte for byte:

```python
        if output is None:
            if _post_ok(queries, keys, values, length):
                calls["grouped"] += 1
                prefix = keys.shape[-2] - length
                output = grouped_attention(queries, keys, values, attention.scale, prefix, _ARCH)
            else:
                calls["loop"] += 1
                output = _row_loop(queries, keys, values, cache, attention.scale)
```

with:

```python
        if output is None and splits:
            if _post_ok(queries, keys, values, length) and tileattn.in_scope(
                queries, keys, values, attention.scale
            ):
                output = tileattn.serve_verify(queries, keys, values, splits)
            else:
                tileattn.calls["verify_declined"] += 1
                output = _stock_verify(queries, keys, values, cache, attention.scale, length)
        elif output is None:
            if _post_ok(queries, keys, values, length):
                calls["grouped"] += 1
                prefix = keys.shape[-2] - length
                output = grouped_attention(queries, keys, values, attention.scale, prefix, _ARCH)
            else:
                calls["loop"] += 1
                output = _row_loop(queries, keys, values, cache, attention.scale)
```

- [ ] **Step 9: Run the verify-branch tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py -v -k scope`
Expected: PASS, 23 passed.

Run: `uv run pytest tests/test_tileattn.py -v -k "follow_the_plan or untagged or out_of_scope or declined"`
Expected: PASS, 14 passed.

- [ ] **Step 10: Run the whole tile and verifier suites, then everything**

Run: `uv run pytest tests/test_tileattn.py tests/test_verifyattn.py -v`
Expected: PASS, 189 passed and 19 skipped on the M2 (the tile's `nax` tests); every existing `test_verifyattn.py` test passes with the flag at 0. Run this and the next command under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py` when another session may be using the GPU.

Run: `uv run pytest -m "not model"`
Expected: PASS. The only skips are the int8 and attention-tile `nax` tests where the tensor units are absent.

- [ ] **Step 11: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/tileattn.py src/sous/engine/verifyattn.py tests/test_tileattn.py tests/test_verifyattn.py tests/tile_fixtures.py
git commit -m "feat(engine): route one-row decode and exact verify through the attention tile when active" -m "The tile serves two call sites that already exist: the global
Qwen3_5Attention.__call__ calls after its projections, and verifyattn's
verifier wrapper. Both decide after the projections and fall back to the
very call the model would have made, so a refusal never appends to the
cache twice, and both act only while the flag is set, so installing the
hook changes nothing until the tile is enabled. The counters show which
path every call took, for the tests and the merge gate.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: The load-time probe

**Files:**
- Modify: `src/sous/engine/tileattn.py`:
  - add `traceback` to the stdlib imports;
  - append the probe section at the end of the file, after Task 5's `serve_verify`.
- Test: `tests/test_tileattn.py`:
  - add to the import block;
  - append the probe section at the end.

**Interfaces:**
- Consumes:
  - Task 4: `tile`, `decode`, `verify`, `_compile_probe`, `_compile_ok`, `_arch`, `_error_line`.
  - Task 5: `calls`, `_active_splits`, `_decode_wrapper`, `_HOOK_MARK`, `_plain_kv_cache`, `tfx.hooked`, `tfx.tiny_language_model`, `tfx.prefill_cache`.
  - Task 3: `boundaries`, `step_of`, `chunk_for`.
  - verifyattn: `plan_transitions(arch: str, gqa: int) -> tuple[int, ...]` and `_attention_modules(model) -> list` (function-local import).
  - mlx-vlm `KVCache` via importlib.
- Produces:
  - `PROBE_FAR_KEYS = 57_000` and `REL_RMS_BOUND = 1e-2`.
  - `far_mark(splits: int) -> int`: the first step boundary ≥ PROBE_FAR_KEYS; 57,601 at S = 20.
  - `parity_cases(splits: int, arch: str) -> tuple[list[int], list[tuple[int, int]]]`, returning (tile marks, stock cases):
    - tile marks = `sorted({N0, b[1], b[2], far_mark(splits)})`, with b the step boundaries from N0;
    - stock cases = `sorted({(p, t) for m in verifyattn.plan_transitions(arch, GQA) if m < N0 for t in (1, 2) for p in (m - t - 1, m - 1 - t // 2, m - 1)})`.
  - `_rel_rms(got, want) -> float`.
  - `_close(got, want) -> bool`: finite, and `_rel_rms <= REL_RMS_BOUND`, written so that NaN fails.
  - `probe(modules: list[Any], splits: int, arch: str) -> str | None`: None when the tile may serve. Otherwise the reason, which starts with `'kernel: '`, `'parity: '`, `'reads past n: '` or `'correctness: '`, or is exactly `"the model's attention module did not reach the tile"`.

The probe is the one verified on the M2 with the stand-in, with these changes:
- step 1 is `_compile_probe()`, reported as `f"kernel: {reason}"`, and runs before the probe allocates anything;
- the far mark is `far_mark(splits)`: the first boundary at or past 57,000 keys, not the nearest one;
- NaN is planted past n along the token axis in every KV head;
- `probe()` restores `calls`, the flag and mlx-vlm's global in a `finally`, and clears the cache there;
- step 5 installs the decode wrapper only for the probe's own duration, and only when the global is not already marked.

- [ ] **Step 1: Write the failing probe tests**

In `tests/test_tileattn.py`, add `import time` to the stdlib block above `import pytest`, keeping it alphabetical (Task 5 already added `importlib`, and `verifyattn` is already on the `from sous.engine import ...` line). Then append:

```python
# ---- the load-time probe -----------------------------------------------------------

_REJECTED_BUILD = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
)


def _qwen_language():
    return importlib.import_module("mlx_vlm.models.qwen3_5.language")


def _guard_probe(monkeypatch) -> None:
    """Registers restores for every global a probe run touches, so a probe
    that fails a test cannot leak a hook, a flag or a compile verdict into
    the next one."""
    language = _qwen_language()
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    monkeypatch.setattr(tileattn, "_active_splits", tileattn._active_splits)
    monkeypatch.setattr(tileattn, "_compile_ok", tileattn._compile_ok)


def _probe_setup(monkeypatch, tile=tfx.stand_in_tile) -> list:
    """`tile` as the kernel, a far mark of 3,201 keys instead of 57,601 (the
    stand-in is plain mlx, far slower than the kernel), a compile check that
    really runs, and the tiny model's full-attention modules."""
    _guard_probe(monkeypatch)
    monkeypatch.setattr(tileattn, "tile", tile)
    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    return verifyattn._attention_modules(tfx.tiny_language_model())


def _run_probe(modules, arch: str | None = None) -> str | None:
    """probe() at S = 20, asserting it left the flag clear and the counters
    and mlx-vlm's global as it found them, whatever it decided."""
    installed = _qwen_language().scaled_dot_product_attention
    before = dict(tileattn.calls)
    reason = tileattn.probe(modules, 20, arch or tileattn._arch())
    assert tileattn._active_splits == 0
    assert tileattn.calls == before
    assert _qwen_language().scaled_dot_product_attention is installed
    return reason


def test_probe_far_mark_is_the_first_step_boundary_at_or_past_the_probe_keys(monkeypatch):
    assert tileattn.far_mark(20) == 57_601
    assert tileattn.far_mark(10) == 57_281
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    assert tileattn.far_mark(20) == 3_201
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_201)
    assert tileattn.far_mark(20) == 3_201  # a boundary is its own far mark


def test_probe_parity_cases_cover_the_first_steps_the_far_mark_and_stock_transitions():
    marks, stock = tileattn.parity_cases(20, "applegpu_g17s")
    assert marks == [1536, 1921, 2561, 57601]
    # 's' parts change plan at 1024 and 1025 keys, both below N0: rows on both sides
    assert stock == [(1021, 2), (1022, 1), (1022, 2), (1023, 1), (1023, 2), (1024, 1), (1024, 2)]
    # 'g' parts change plan only at 4096 keys, where the tile serves
    assert tileattn.parity_cases(20, "applegpu_g14g") == ([1536, 1921, 2561, 57601], [])


@pytest.mark.slow
def test_probe_passes_the_stand_in(monkeypatch):
    assert _run_probe(_probe_setup(monkeypatch)) is None


@pytest.mark.slow
def test_probe_checks_the_stock_side_on_any_gpu_and_reuses_an_installed_hook(monkeypatch):
    modules = _probe_setup(monkeypatch)
    # The hook an earlier load installed, and the flag enable() cleared first.
    tfx.hooked(monkeypatch, splits=20)
    monkeypatch.setattr(tileattn, "_active_splits", 0)
    hook = _qwen_language().scaled_dot_product_attention
    assert tileattn.parity_cases(20, "applegpu_g17s")[1]  # the stock cases run on this GPU
    assert _run_probe(modules, "applegpu_g17s") is None
    assert _qwen_language().scaled_dot_product_attention is hook


def test_probe_fails_parity_for_a_kernel_that_ignores_the_stepped_chunk(monkeypatch):
    modules = _probe_setup(
        monkeypatch, lambda q, k, v, n, chunk: tfx.stand_in_tile(q, k, v, n, -(-n // 20))
    )
    reason = _run_probe(modules)
    assert reason is not None and reason.startswith("parity:"), reason


def test_probe_fails_correctness_for_a_kernel_wrong_the_same_way_on_both_paths(monkeypatch):
    def scaled(q, k, v, n, chunk):
        out = tfx.stand_in_tile(q, k, v, n, chunk)
        return (out.astype(mx.float32) * 1.1).astype(mx.bfloat16)

    modules = _probe_setup(monkeypatch, scaled)
    # Parity cannot see this kernel, so the sweep past N0 would only cost time.
    monkeypatch.setattr(tileattn, "parity_cases", lambda splits, arch: ([tileattn.N0], []))
    reason = _run_probe(modules)
    assert reason is not None and reason.startswith("correctness:"), reason


def test_probe_reports_a_kernel_that_fails_to_build_with_its_error_line(monkeypatch):
    def rejected(q, k, v, n, chunk):
        raise RuntimeError(_REJECTED_BUILD)

    modules = _probe_setup(monkeypatch, rejected)
    reason = _run_probe(modules)
    assert reason == "kernel: program_source:12:3: error: use of undeclared identifier 'x'"


def test_probe_refuses_a_module_that_never_reaches_the_tile(monkeypatch):
    modules = _probe_setup(monkeypatch)
    monkeypatch.setattr(tileattn, "parity_cases", lambda splits, arch: ([tileattn.N0], []))
    monkeypatch.setattr(tileattn, "_plain_kv_cache", lambda cache: False)
    assert _run_probe(modules) == "the model's attention module did not reach the tile"


def test_probe_restores_what_it_touched_when_it_raises(monkeypatch):
    def fails_past_the_first_step(q, k, v, n, chunk):
        if n >= 1921:
            raise RuntimeError("device lost")
        return tfx.stand_in_tile(q, k, v, n, chunk)

    modules = _probe_setup(monkeypatch, fails_past_the_first_step)
    installed = _qwen_language().scaled_dot_product_attention
    before = dict(tileattn.calls)
    with pytest.raises(RuntimeError, match="device lost"):
        tileattn.probe(modules, 20, tileattn._arch())
    assert tileattn._active_splits == 0
    assert tileattn.calls == before
    assert _qwen_language().scaled_dot_product_attention is installed


@tfx.nax
def test_probe_passes_the_real_kernel(monkeypatch):
    _guard_probe(monkeypatch)
    modules = verifyattn._attention_modules(tfx.tiny_language_model())
    start = time.perf_counter()
    reason = _run_probe(modules)
    assert reason is None, f"{reason} (after {time.perf_counter() - start:.2f}s)"
```

The two full probe runs are `slow`: with the plain-mlx stand-in they take 7–11 s each on the M2, even at a far mark of 3,201 keys. The correctness and module-reach tests trim the parity sweep to N0 through `parity_cases`, since the two passing runs already prove the sweep and those two tests are about later steps.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -k test_probe_ -v`

Expected: 9 failed, 1 skipped. The failures come from `_probe_setup` and the far-mark test:
- `AttributeError: <module 'sous.engine.tileattn' ...> has no attribute 'PROBE_FAR_KEYS'`
- `AttributeError: module 'sous.engine.tileattn' has no attribute 'far_mark'`
- `AttributeError: module 'sous.engine.tileattn' has no attribute 'parity_cases'`

`test_probe_passes_the_real_kernel` skips on this M2 and on CI with `attention tile unavailable: macOS 15.5 < 26.2` or the CI runner's own platform reason.

- [ ] **Step 3: Implement the probe**

Add `import traceback` to the stdlib imports at the top of `src/sous/engine/tileattn.py`, keeping them alphabetical (`importlib` is already there from the decode hook). Then append at the end of the file:

```python
# The probe's far parity mark is the first step boundary at or past this many
# keys: every row runs its full S splits there (from 11,553 keys at S = 20),
# and the subagent turns the tile was measured on sit at 44-77K.
PROBE_FAR_KEYS = 57_000
# How far the probe lets the tile land from stock one-row SDPA, as a relative
# RMS difference. The tile's error against a float64 reference is at or below
# stock's, and one bf16 ulp of the output is about 4e-3.
REL_RMS_BOUND = 1e-2


def far_mark(splits: int) -> int:
    """The first step boundary at or past PROBE_FAR_KEYS."""
    reach = PROBE_FAR_KEYS + GRAN * splits  # no step is wider than GRAN * splits keys
    return next(b for b in boundaries(reach, splits) if b >= PROBE_FAR_KEYS)


def parity_cases(splits: int, arch: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(tile marks, stock cases). Each tile mark (N0, the next two step
    boundaries and the far mark) is probed at T = 1..8 with rows below it,
    straddling it and from it. The stock cases put T = 1 and 2 rows on both
    sides of every verifyattn plan transition below N0, where verify runs
    grouped_attention and must still equal stock one-row SDPA."""
    from sous.engine.verifyattn import plan_transitions

    first = boundaries(N0 + 2 * GRAN * splits, splits)
    marks = sorted({first[0], first[1], first[2], far_mark(splits)})
    stock = sorted(
        {
            (p, t)
            for m in plan_transitions(arch, GQA)
            if m < N0
            for t in (1, 2)
            for p in (m - t - 1, m - 1 - t // 2, m - 1)
        }
    )
    return marks, stock


def _rel_rms(got: Any, want: Any) -> float:
    import mlx.core as mx

    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _close(got: Any, want: Any) -> bool:
    """Finite and within REL_RMS_BOUND of `want`: a NaN anywhere fails, since
    no comparison with NaN is true."""
    import mlx.core as mx

    return bool(mx.all(mx.isfinite(got)).item()) and _rel_rms(got, want) <= REL_RMS_BOUND


def probe(modules: list[Any], splits: int, arch: str) -> str | None:
    """None when the tile may serve `modules` on this GPU at `splits` splits,
    else why not. Runs on the loading thread with the flag clear, and leaves
    the flag clear and the counters and mlx-vlm's global as it found them.
    Its arrays are gone before the cache is cleared, so the budget measured
    next reads true."""
    import mlx.core as mx

    global _active_splits
    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    installed = language.scaled_dot_product_attention
    counted = dict(calls)
    try:
        return _probe_steps(modules, splits, arch, language)
    except BaseException as e:
        # The traceback holds the steps' frame, and every probe array in it,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _active_splits = 0
        language.scaled_dot_product_attention = installed
        calls.update(counted)
        mx.clear_cache()


def _probe_steps(modules: list[Any], splits: int, arch: str, language: Any) -> str | None:
    """Every input is made with GPU ops (no float64, which is CPU-only in mlx)
    and evaluated before the first launch: a kernel failure with a CPU-stream
    op in flight deadlocks mlx's exception path. Inputs are laid out as
    serving lays them out: transposed query rows, K/V slices of 256-step
    padded buffers."""
    import mlx.core as mx

    global _active_splits
    # Every row variant compiles now, so no compile first happens mid-turn.
    failed = _compile_probe()
    if failed is not None:
        return f"kernel: {failed}"

    marks, stock_cases = parity_cases(splits, arch)
    far = far_mark(splits)
    first = boundaries(N0 + 2 * GRAN * splits, splits)
    mid_step = (first[1] + first[2]) // 2
    top = max(far + 16, max((p + t for p, t in stock_cases), default=0))
    cap = -(-(top + 1) // 256) * 256
    k_key, v_key, q_key, x_key, m_key = mx.random.split(mx.random.key(0), 5)
    kb = mx.random.normal((1, HKV, cap, D), key=k_key).astype(mx.bfloat16)
    vb = mx.random.normal((1, HKV, cap, D), key=v_key).astype(mx.bfloat16)
    qpool = mx.random.normal((1, 20, HQ, D), key=q_key).astype(mx.bfloat16)
    nan_tail = mx.full((1, HKV, 256, D), float("nan"), dtype=mx.bfloat16)
    mx.eval(kb, vb, qpool, nan_tail)

    def rows(start: int, t: int) -> Any:
        return qpool[:, start : start + t].transpose(0, 2, 1, 3)

    # Parity: every verify row equals decode() at its own key count.
    def parity(point: int, cases: list[tuple[int, int]]) -> str | None:
        base = point - 10  # query row i sits at position base + i and sees base + i + 1 keys
        dec = {
            pos + 1: decode(rows(pos - base, 1), kb[:, :, : pos + 1], vb[:, :, : pos + 1], splits)
            for pos in range(base, base + 20)
        }
        for prefix, t in cases:
            n = prefix + t
            out = verify(rows(prefix - base, t), kb[:, :, :n], vb[:, :, :n], splits)
            same = mx.stack(
                [mx.array_equal(out[:, :, r], dec[prefix + r + 1][:, :, 0]) for r in range(t)]
            )
            if not mx.all(same).item():
                return f"parity: verify rows at prefix {prefix}, T={t} differ from decode"
        return None

    for mark in marks:
        cases = [
            (p, t)
            for t in range(1, MAX_RUN + 1)
            for p in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1})
        ]
        if (reason := parity(mark, cases)) is not None:
            return reason
    for prefix, t in stock_cases:
        if (reason := parity(prefix + 1, [(prefix, t)])) is not None:
            return reason

    # Reads stay in [0, n): NaN past n in every KV head, and a buffer that
    # ends exactly at n, give the clean result.
    for n, t in ((N0, 1), (mid_step, 3), (far, MAX_RUN)):
        want = verify(rows(0, t), kb[:, :, :n], vb[:, :, :n], splits)
        k_nan = mx.concatenate([kb[:, :, :n], nan_tail], axis=2)
        v_nan = mx.concatenate([vb[:, :, :n], nan_tail], axis=2)
        k_end, v_end = mx.contiguous(kb[:, :, :n]), mx.contiguous(vb[:, :, :n])
        mx.eval(want, k_nan, v_nan, k_end, v_end)
        for keys, values in ((k_nan[:, :, :n], v_nan[:, :, :n]), (k_end, v_end)):
            got = verify(rows(0, t), keys, values, splits)
            if not (mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()):
                return f"reads past n: T={t} at n={n} differs from the clean buffer"

    # Correctness, which parity cannot see in a kernel wrong the same way on
    # both paths: x8 queries, a dominant key at n - 1, and a second one at n,
    # just past the live range, that the tile must not read.
    q8 = (qpool[:, :1].astype(mx.float32) * 8).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    dominant = q8[:, :, 0].reshape(1, HKV, GQA, D)[:, :, :1]
    ones = mx.ones((1, HKV, 1, D), dtype=mx.bfloat16)
    for n in sorted({N0, mid_step, far}):
        kp = mx.concatenate([kb[:, :, : n - 1], dominant, dominant, kb[:, :, n + 1 :]], axis=2)
        vp = mx.concatenate([vb[:, :, : n - 1], ones, -ones, vb[:, :, n + 1 :]], axis=2)
        mx.eval(kp, vp)
        got = decode(q8, kp[:, :, :n], vp[:, :, :n], splits)
        want = mx.fast.scaled_dot_product_attention(
            q8, kp[:, :, :n], vp[:, :, :n], scale=SCALE, mask=None
        )
        if not _close(got, want):
            rms = _rel_rms(got, want)
            return f"correctness: n={n} is {rms:.3g} from stock (bound {REL_RMS_BOUND})"

    # The loaded model's own module, with its projections and rope, through
    # mlx-vlm's global, over a KVCache as a fork restore leaves it: keys,
    # values and offset assigned, the padded buffer NaN past the live keys.
    module = modules[0]
    n = mid_step
    x = mx.random.normal((1, 1, module.o_proj.weight.shape[0]), key=x_key).astype(mx.bfloat16)
    capacity = -(-n // 256) * 256
    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache

    def restored() -> Any:
        cache = kv_cache()
        k_rows, v_rows = mx.random.split(m_key)
        tail = nan_tail[:, :, : capacity - n + 1]
        live = (1, HKV, n - 1, D)
        cache.keys = mx.concatenate(
            [mx.random.normal(live, key=k_rows).astype(mx.bfloat16), tail], axis=2
        )
        cache.values = mx.concatenate(
            [mx.random.normal(live, key=v_rows).astype(mx.bfloat16), tail], axis=2
        )
        cache.offset = n - 1
        mx.eval(cache.keys, cache.values)
        return cache

    if not getattr(language.scaled_dot_product_attention, _HOOK_MARK, False):
        language.scaled_dot_product_attention = _decode_wrapper(
            language.scaled_dot_product_attention
        )
    _active_splits = 0
    want = module(x, mask=None, cache=restored())
    mx.eval(want)
    before = calls["decode_tile"]
    _active_splits = splits
    got = module(x, mask=None, cache=restored())
    mx.eval(got)
    _active_splits = 0
    if calls["decode_tile"] != before + 1:
        return "the model's attention module did not reach the tile"
    if not _close(got, want):
        return f"correctness: the model's module is {_rel_rms(got, want):.3g} from stock"
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -k test_probe_ -v`

Expected: 9 passed and 1 skipped (the `nax` test), in 20–30 s on the M2. Most of it is the two `slow` runs at 7–11 s each.

The probe tests put 20–30 s of GPU work through the stand-in. If another session may be using the GPU, run them under the lock instead:

`GPU_LABEL=tile-probe python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py uv run pytest tests/test_tileattn.py -k test_probe_ -v`

- [ ] **Step 5: Run the wider checks**

```bash
uv run pytest tests/test_tileattn.py tests/test_verifyattn.py -v
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected:
- every test passes;
- the `nax` tests skip on this M2 and on CI;
- ruff and ty report nothing.

`_probe_steps` takes the language module as `Any`, so the wrapper assignment there needs no `# ty: ignore`.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/tileattn.py tests/test_tileattn.py
git commit -m "feat(engine): prove the attention tile exact and correct on this GPU at load" -m "A kernel compiled at load by the OS's Metal toolchain can be rejected by a
new macOS, and the tile's parity argument holds only if this GPU and the
loaded model's own attention behave as the schedule assumes. The probe
compiles every row variant up front, so no compile first happens mid-turn.
It then checks, on this device:
- verify rows are bitwise equal to decode at the first step boundaries, at
  the far mark, and at the stock plan transitions below N0;
- reads stay inside n;
- the tile is close to stock SDPA, which catches a kernel wrong the same way
  on both paths;
- one real attention module reaches the tile over a restored-style cache.

It leaves the flag, the counters and mlx-vlm's global as it found them, and
clears the cache before the budget is measured.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: The gates, enable() and the flag lifecycle

**Files:**
- Modify: `src/sous/engine/tileattn.py`:
  - the stdlib, `collections.abc` and sous imports;
  - a constants block directly above the probe section from Task 6;
  - `probe()`'s module name;
  - the gates and `enable()`, appended at the end of the file.
- Modify: `tests/tile_fixtures.py`: a module-level `import importlib`; append `guard_tile_state` and `tile_ready`.
- Test: `tests/test_tileattn.py`:
  - the probe helpers switch to `tfx.guard_tile_state`;
  - the import block;
  - the enable section, appended at the end.

**Interfaces:**
- Consumes:
  - Task 1: `nax.platform_reason()`.
  - Task 2: `gpucores.matched_cores(device_name)`, `gpucores.read`, `gpucores.GPUCores`.
  - Task 5: `_active_splits`, `clear()`, `install_decode_hook()`, `calls`, `tfx.hooked`, `tfx.verify_ready`, `tfx.verify_forward`, `tfx.tiny_language_model`.
  - Task 6: `probe(modules, splits, arch)`, `PROBE_FAR_KEYS`.
  - Task 4: `_arch()`, `_compile_ok`, `tfx.stand_in_tile`.
  - verifyattn: `_gate() -> str | None`, `_source_digest(module: str, qualname: str) -> str | None`, `_attention_modules(model)`.
  - int8prefill: `_model_type(model) -> str`, `_error_line`.
- Produces:
  - `SUPPORTED_MODEL_TYPES = frozenset({'qwen3_5'})`.
  - `SUPPORTED_DRAFTER_KINDS = frozenset({'dflash'})`.
  - `_DECODE_MODULE = 'mlx_vlm.models.qwen3_5.language'`.
  - `VALIDATED_DECODE_SOURCES: dict[str, frozenset[str]]`, with the three pinned sha256 values for `'Qwen3_5Attention.__call__'`, `'Qwen3_5Model.__call__'` and `'LanguageModel.__call__'`.
  - `_refuse(reason: str, probe_seconds: float | None = None) -> dict[str, Any]`:
    - reduces the reason with `_error_line`;
    - one `warnings.warn(f'sous: attention tile unavailable ({reason}); decode and verify run the stock attention paths', stacklevel=3)`;
    - returns `{'state': 'unavailable', 'reason': reason, 'splits': None, 'probe_seconds': probe_seconds}`.
  - `_model_reason(model: Any) -> str | None`.
  - `_drafter_reason(drafter_kind: str | None, verify_status: Mapping[str, Any] | None) -> str | None`.
  - `_decode_sources_reason() -> str | None`.
  - `_split_target(device_name: str) -> tuple[int | None, str | None]`.
  - `enable(model: Any, *, enabled: bool, drafter_kind: str | None, verify_status: Mapping[str, Any] | None) -> dict[str, Any]`. All three keyword arguments are required.
    - off → `{'state': 'off', 'reason': None, 'splits': None, 'probe_seconds': None}`;
    - active → `{'state': 'active', 'reason': None, 'splits': S, 'probe_seconds': round(seconds, 2)}`.
  - Reason texts:
    - `f'unsupported model type {mt or "unknown"!r} (qwen3_5 only)'`
    - `'no full-attention layers'`
    - `f'unsupported attention shape {q}/{kv}/{d} (24/4/256 only)'`
    - `f'attention scale {scale} is not 1/16'`
    - `f'attention dtype {dtype} is not bfloat16'`
    - `f'drafter kind {kind!r} (dflash only)'`
    - `f'exact verify attention is {state}: the verifier would run stock attention beside the tile'`
    - `f'mlx-vlm {qualname} changed'`
    - `f'split target {cores} not measured (measured: 20)'`, built from `sorted(MEASURED_SPLITS)`
    - verifyattn's own `_gate()` texts, e.g. `'mlx 0.99.0 not validated'` and `'MLX_SDPA_BLOCKS is set'`
    - the reader's reason or the mismatch reason, verbatim
    - the probe's reason
  - In `tests/tile_fixtures.py`:
    - `guard_tile_state(monkeypatch) -> None`: registers restores for the language global, `tileattn._active_splits` and `tileattn._compile_ok`;
    - `tile_ready(monkeypatch, *, cores: int = 20) -> None`:
      - patches `tileattn.nax.platform_reason` → None;
      - patches `tileattn.gpucores.read` → `GPUCores(cores, mx.device_info()['device_name'], None, 'iokit')`;
      - patches `tileattn.tile` → the stand-in;
      - patches `tileattn.PROBE_FAR_KEYS` → 3_000;
      - calls `guard_tile_state`.

- [ ] **Step 1: Share the tile-state guard through the fixtures**

In `tests/tile_fixtures.py`, add `import importlib` as the first module-level import, above `from typing import Any` (the one inside `hooked` stays; it is harmless). Then append:

```python
def guard_tile_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers restores for what enable() and the probe leave behind for the
    rest of the process: mlx-vlm's decode global (the hook), the flag and the
    remembered compile verdict. A test that activates the tile, or runs the
    stand-in through the compile check, then cannot leak either into the next."""
    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    monkeypatch.setattr(tileattn, "_active_splits", tileattn._active_splits)
    monkeypatch.setattr(tileattn, "_compile_ok", tileattn._compile_ok)


def tile_ready(monkeypatch: pytest.MonkeyPatch, *, cores: int = 20) -> None:
    """Every enable() guard that reads the machine passes on any GPU: the
    platform rule, and a core count of `cores` under this GPU's own name. The
    stand-in is the kernel, and the probe's far mark is one the stand-in
    reaches in seconds."""
    import mlx.core as mx

    from sous.engine import gpucores

    device = str(mx.device_info()["device_name"])
    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(
        tileattn.gpucores, "read", lambda: gpucores.GPUCores(cores, device, None, "iokit")
    )
    monkeypatch.setattr(tileattn, "tile", stand_in_tile)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    guard_tile_state(monkeypatch)
```

`tileattn.gpucores` exists only once Step 7 adds the import; the probe tests in Step 3 do not call `tile_ready`, so they run before it.

- [ ] **Step 2: Point the probe tests at the shared guard**

In `tests/test_tileattn.py`, delete the test-local guard. Replace:

```python
def _guard_probe(monkeypatch) -> None:
    """Registers restores for every global a probe run touches, so a probe
    that fails a test cannot leak a hook, a flag or a compile verdict into
    the next one."""
    language = _qwen_language()
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    monkeypatch.setattr(tileattn, "_active_splits", tileattn._active_splits)
    monkeypatch.setattr(tileattn, "_compile_ok", tileattn._compile_ok)


def _probe_setup(monkeypatch, tile=tfx.stand_in_tile) -> list:
```

with:

```python
def _probe_setup(monkeypatch, tile=tfx.stand_in_tile) -> list:
```

In `_probe_setup`, replace:

```python
    _guard_probe(monkeypatch)
    monkeypatch.setattr(tileattn, "tile", tile)
```

with:

```python
    tfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(tileattn, "tile", tile)
```

In `test_probe_passes_the_real_kernel`, replace:

```python
    _guard_probe(monkeypatch)
    modules = verifyattn._attention_modules(tfx.tiny_language_model())
```

with:

```python
    tfx.guard_tile_state(monkeypatch)
    modules = verifyattn._attention_modules(tfx.tiny_language_model())
```

- [ ] **Step 3: Run the probe tests to verify the move changed nothing**

Run: `uv run pytest tests/test_tileattn.py -k test_probe_ -v`

Expected: 9 passed and 1 skipped (`nax`), as at the end of Task 6.

- [ ] **Step 4: Write the failing enable tests**

In `tests/test_tileattn.py`, add these to the stdlib block above `import pytest`, keeping it alphabetical:
- `import json`, `import logging`, `import subprocess`, `import sys`, `import types`, `import warnings`;
- `from pathlib import Path`.

Then append:

```python
# ---- the gates and enable() ---------------------------------------------------------


def _refusal(lm, **kwargs) -> dict:
    """enable() on `lm`, expected to refuse: exactly one warning, on one line,
    in the tile's words, and the flag left clear."""
    kwargs = {"drafter_kind": None, "verify_status": None, **kwargs}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = tileattn.enable(lm, enabled=True, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1, messages
    assert messages[0].startswith("sous: attention tile unavailable (")
    assert "\n" not in messages[0]
    assert status["state"] == "unavailable" and status["splits"] is None
    assert tileattn._active_splits == 0
    return status


def test_enable_off_returns_before_any_guard(monkeypatch):
    tfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(tileattn, "_active_splits", 20)

    def untouchable(*args, **kwargs):
        raise AssertionError("a guard ran while the tile is off")

    monkeypatch.setattr(tileattn.nax, "platform_reason", untouchable)
    monkeypatch.setattr(tileattn, "_model_reason", untouchable)
    monkeypatch.setattr(tileattn, "probe", untouchable)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = tileattn.enable(
            types.SimpleNamespace(), enabled=False, drafter_kind="mtp", verify_status=None
        )
    assert status == {"state": "off", "reason": None, "splits": None, "probe_seconds": None}
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert tileattn._active_splits == 0


def test_enable_reports_the_platform_whatever_the_model(monkeypatch):
    tfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    status = _refusal(types.SimpleNamespace(), drafter_kind="mtp")
    assert status["reason"] == "macOS 15.5 < 26.2" and status["probe_seconds"] is None


_GUARD_REASONS = [
    "macOS 15.5 < 26.2",
    "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
    "drafter kind 'mtp' (dflash only)",
    "mlx 0.99.0 not validated",
    "mlx-vlm Qwen3_5Attention.__call__ changed",
    "split target 16 not measured (measured: 20)",
    "parity: verify rows at prefix 1535, T=6 differ from decode",
]


@pytest.mark.parametrize("first", range(len(_GUARD_REASONS)))
def test_enable_names_the_first_failing_guard(monkeypatch, first):
    """Guard `first` and every guard after it fail; the reason is guard `first`'s."""
    from sous.engine import gpucores

    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    device = str(mx.device_info()["device_name"])
    kwargs: dict = {"drafter_kind": None, "verify_status": None}
    breakers = [
        lambda: monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: _GUARD_REASONS[0]),
        lambda: monkeypatch.setattr(lm, "model_type", "qwen3_5_moe"),
        lambda: kwargs.update(drafter_kind="mtp"),
        lambda: monkeypatch.setattr(mx, "__version__", "0.99.0"),
        lambda: monkeypatch.setitem(
            tileattn.VALIDATED_DECODE_SOURCES, "Qwen3_5Attention.__call__", frozenset({"0" * 64})
        ),
        lambda: monkeypatch.setattr(
            tileattn.gpucores, "read", lambda: gpucores.GPUCores(16, device, None, "iokit")
        ),
        lambda: monkeypatch.setattr(
            tileattn, "probe", lambda modules, splits, arch: _GUARD_REASONS[6]
        ),
    ]
    for brk in breakers[first:]:
        brk()
    assert _refusal(lm, **kwargs)["reason"] == _GUARD_REASONS[first]


def test_enable_refuses_other_model_types(monkeypatch):
    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    # The MoE variant reuses these very attention classes.
    monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
    assert _refusal(lm)["reason"] == "unsupported model type 'qwen3_5_moe' (qwen3_5 only)"


def test_enable_refuses_a_model_without_full_attention(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(verifyattn, "_attention_modules", lambda model: [])
    assert _refusal(tfx.tiny_language_model())["reason"] == "no full-attention layers"


def test_enable_refuses_another_attention_shape(monkeypatch):
    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model(heads=6, kv_heads=1)
    assert _refusal(lm)["reason"] == "unsupported attention shape 6/1/256 (24/4/256 only)"


def test_enable_refuses_another_attention_scale(monkeypatch):
    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    monkeypatch.setattr(verifyattn._attention_modules(lm)[1], "scale", 0.1)
    assert _refusal(lm)["reason"] == "attention scale 0.1 is not 1/16"


def test_enable_refuses_another_attention_dtype(monkeypatch):
    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    lm.set_dtype(mx.float16)
    assert _refusal(lm)["reason"] == f"attention dtype {mx.float16} is not bfloat16"


@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
def test_enable_refuses_drafters_other_than_dflash(monkeypatch, kind):
    tfx.tile_ready(monkeypatch)
    status = _refusal(
        tfx.tiny_language_model(), drafter_kind=kind, verify_status={"state": "active"}
    )
    assert status["reason"] == f"drafter kind {kind!r} (dflash only)"


@pytest.mark.parametrize(
    "verify_status",
    [
        {"state": "unavailable", "reason": "mlx 0.33.0 not validated", "probe_seconds": None},
        {"state": "off", "reason": None, "probe_seconds": None},
        None,
    ],
)
def test_enable_needs_exact_verify_beside_a_dflash_drafter(monkeypatch, verify_status):
    tfx.tile_ready(monkeypatch)
    status = _refusal(tfx.tiny_language_model(), drafter_kind="dflash", verify_status=verify_status)
    state = verify_status["state"] if verify_status else None
    assert status["reason"] == (
        f"exact verify attention is {state}: the verifier would run stock attention beside the tile"
    )


def test_enable_passes_no_drafter_and_a_dflash_one_with_exact_verify():
    assert tileattn._drafter_reason(None, None) is None
    assert tileattn._drafter_reason("dflash", {"state": "active", "reason": None}) is None


def test_enable_refuses_an_unvalidated_mlx(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    assert _refusal(tfx.tiny_language_model())["reason"] == "mlx 0.99.0 not validated"


def test_enable_refuses_a_forced_sdpa_block_count(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "4")
    assert _refusal(tfx.tiny_language_model())["reason"] == "MLX_SDPA_BLOCKS is set"


def test_enable_validates_the_locked_mlx_vlm_decode_sources():
    assert set(tileattn.VALIDATED_DECODE_SOURCES) == {
        "Qwen3_5Attention.__call__",
        "Qwen3_5Model.__call__",
        "LanguageModel.__call__",
    }
    assert tileattn._decode_sources_reason() is None


def test_enable_refuses_a_changed_decode_source(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setitem(
        tileattn.VALIDATED_DECODE_SOURCES, "Qwen3_5Model.__call__", frozenset({"0" * 64})
    )
    reason = _refusal(tfx.tiny_language_model())["reason"]
    assert reason == "mlx-vlm Qwen3_5Model.__call__ changed"


@pytest.mark.parametrize("case", ["unreadable", "another GPU", "unmeasured"])
def test_enable_needs_a_measured_core_count_of_this_gpu(monkeypatch, case):
    from sous.engine import gpucores

    tfx.tile_ready(monkeypatch)
    device = str(mx.device_info()["device_name"])
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
    monkeypatch.setattr(tileattn.gpucores, "read", lambda: reading)
    assert _refusal(tfx.tiny_language_model())["reason"] == reason


def test_enable_refuses_a_failed_probe_with_its_cost(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(
        tileattn, "probe", lambda modules, splits, arch: "correctness: n=1536 is 0.101 from stock"
    )
    status = _refusal(tfx.tiny_language_model())
    assert status["reason"] == "correctness: n=1536 is 0.101 from stock"
    assert isinstance(status["probe_seconds"], float)


def test_enable_refuses_an_exception_in_a_guard_with_its_error_line(monkeypatch):
    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(tileattn, "_active_splits", 20)

    def broken(model):
        raise RuntimeError("Unable to build metal library\nprogram_source:3:1: error: boom")

    monkeypatch.setattr(verifyattn, "_attention_modules", broken)
    status = _refusal(tfx.tiny_language_model())
    assert status["reason"] == "program_source:3:1: error: boom"
    assert status["probe_seconds"] is None


@pytest.mark.parametrize(
    ("kind", "verify_status"),
    [(None, None), ("dflash", {"state": "active", "reason": None, "probe_seconds": 0.21})],
)
def test_enable_activates_the_tile_when_the_probe_passes(monkeypatch, caplog, kind, verify_status):
    tfx.tile_ready(monkeypatch)
    lm = tfx.tiny_language_model()
    seen: list = []

    def passing(modules, splits, arch):
        seen.append(([id(m) for m in modules], splits, arch))

    monkeypatch.setattr(tileattn, "probe", passing)
    with caplog.at_level(logging.INFO, logger="sous.engine.tileattn"):
        status = tileattn.enable(lm, enabled=True, drafter_kind=kind, verify_status=verify_status)
    assert status["state"] == "active" and status["reason"] is None
    assert status["splits"] == 20 and isinstance(status["probe_seconds"], float)
    assert seen == [([id(m) for m in verifyattn._attention_modules(lm)], 20, tileattn._arch())]
    assert tileattn._active_splits == 20
    assert getattr(_qwen_language().scaled_dot_product_attention, tileattn._HOOK_MARK, False)
    assert [r.getMessage() for r in caplog.records if r.name == "sous.engine.tileattn"] == [
        f"attention tile: 2 layers, 20 splits, probe {status['probe_seconds']:.2f}s"
    ]


@pytest.mark.slow
def test_enable_goes_active_through_the_real_probe(monkeypatch):
    tfx.tile_ready(monkeypatch)
    status = tileattn.enable(
        tfx.tiny_language_model(), enabled=True, drafter_kind=None, verify_status=None
    )
    assert status["state"] == "active" and status["splits"] == 20, status
    assert isinstance(status["probe_seconds"], float)
    assert tileattn._active_splits == 20
    assert getattr(_qwen_language().scaled_dot_product_attention, tileattn._HOOK_MARK, False)


@pytest.mark.parametrize("ending", ["disabled", "failed gate", "other model", "cleared"])
def test_enable_then_a_later_load_or_clear_leaves_decode_and_verify_stock(monkeypatch, ending):
    from mlx_vlm.models.base import scaled_dot_product_attention as original
    from mlx_vlm.models.cache import KVCache

    from sous.engine import gpucores

    tfx.tile_ready(monkeypatch)
    monkeypatch.setattr(tileattn, "probe", lambda modules, splits, arch: None)
    lm = tfx.tiny_language_model()
    tfx.verify_ready(monkeypatch, lm)
    ids = mx.random.randint(0, tfx.VOCAB, (2003,), key=mx.random.key(5)).tolist()
    cache = tfx.prefill_cache(lm, ids[:2000])
    reference = tfx.verify_forward(lm, cache, ids[2000:])
    queries, keys, values = tfx.serving_qkv(2000, 1)

    def decode_row():
        return _qwen_language().scaled_dot_product_attention(
            queries, keys, values, cache=KVCache(), scale=tileattn.SCALE, mask=None
        )

    stock = original(queries, keys, values, cache=KVCache(), scale=tileattn.SCALE, mask=None)
    status = tileattn.enable(lm, enabled=True, drafter_kind=None, verify_status=None)
    assert status["state"] == "active"
    before = dict(tileattn.calls)
    mx.eval(decode_row())
    tfx.verify_forward(lm, cache, ids[2000:])
    # While active, both paths reach the tile: what follows is not vacuous.
    assert tileattn.calls["decode_tile"] == before["decode_tile"] + 1
    assert tileattn.calls["verify_rows_tile"] == before["verify_rows_tile"] + 2 * 3

    if ending == "disabled":
        tileattn.enable(lm, enabled=False, drafter_kind=None, verify_status=None)
    elif ending == "failed gate":
        device = str(mx.device_info()["device_name"])
        monkeypatch.setattr(
            tileattn.gpucores, "read", lambda: gpucores.GPUCores(16, device, None, "iokit")
        )
        _refusal(lm)
    elif ending == "other model":
        monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
        _refusal(lm)
    else:
        tileattn.clear()

    assert tileattn._active_splits == 0
    before = dict(tileattn.calls)
    assert mx.array_equal(decode_row(), stock).item()
    after = tfx.verify_forward(lm, cache, ids[2000:])
    assert all(mx.array_equal(a, r).item() for a, r in zip(after, reference, strict=True))
    for key in ("decode_tile", "verify_calls", "verify_rows_tile"):
        assert tileattn.calls[key] == before[key], key


@tfx.nax
def test_enable_activates_the_real_kernel_on_the_tiny_model(monkeypatch):
    from sous.engine import gpucores

    tfx.guard_tile_state(monkeypatch)
    device = str(mx.device_info()["device_name"])
    monkeypatch.setattr(
        tileattn.gpucores, "read", lambda: gpucores.GPUCores(20, device, None, "iokit")
    )
    status = tileattn.enable(
        tfx.tiny_language_model(), enabled=True, drafter_kind=None, verify_status=None
    )
    assert status["state"] == "active" and status["splits"] == 20, status


# Patched before anything compiles the kernel: a built pair is cached by name
# for the life of the process and a passing compile is remembered, so this
# needs an interpreter of its own.
_BROKEN_KERNEL = """
import json

import mlx.core as mx

from sous.engine import gpucores, tileattn

source = tileattn._kernel_text


def broken(name):
    text = source(name)
    if name != "attention_tile.metal":
        return text
    split, reduce = text.split(tileattn._REDUCE_MARK)
    split = "  int sous_broken = sous_undeclared_identifier;\\n" + split
    return split + tileattn._REDUCE_MARK + reduce


tileattn._kernel_text = broken
from tests import tile_fixtures as tfx

device = str(mx.device_info()["device_name"])
gpucores.read = lambda: gpucores.GPUCores(20, device, None, "iokit")
status = tileattn.enable(
    tfx.tiny_language_model(), enabled=True, drafter_kind=None, verify_status=None
)
print(json.dumps(status))
"""


@tfx.nax
@pytest.mark.slow
def test_enable_reports_a_kernel_this_os_rejects_with_its_error_line():
    done = subprocess.run(
        [sys.executable, "-c", _BROKEN_KERNEL],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    status = json.loads(done.stdout.strip().splitlines()[-1])
    assert status["state"] == "unavailable", status
    assert status["reason"].startswith("kernel: ") and "error:" in status["reason"], status
```

About these tests:
- `_refusal` is the one place that asserts the failure policy, so every refusal test also proves exactly one warning, on one line, with the flag clear.
- The lifecycle test stubs the probe, so it stays fast. Its positive control asserts that decode and verify both reached the tile while active, so the equality checks after deactivation are not vacuous.
- The subprocess test patches the kernel source before anything compiles it, in an interpreter of its own:
  - `int8prefill._kernels` caches a built pair by name for the life of the process;
  - `_compile_ok` remembers a passing compile;
  - importing `tests.tile_fixtures` compiles at import.

  Its child was dry-run on the M2 with the platform rule forced open. It exits 0 and its last stdout line is the status as JSON, where the reason is the MPP header's `fatal error:` line. On the M5 the reason is the undeclared identifier's `error:` line.

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tileattn.py -k test_enable_ -v`

Expected: 36 failed and 2 skipped. The two `nax` tests skip on this M2 and on CI. The failures are:
- `AttributeError: module 'sous.engine.tileattn' has no attribute 'enable'`
- `AttributeError: module 'sous.engine.tileattn' has no attribute 'VALIDATED_DECODE_SOURCES'`
- `AttributeError: module 'sous.engine.tileattn' has no attribute '_drafter_reason'`
- `AttributeError: <module 'sous.engine.tileattn' ...> has no attribute '_model_reason'`
- `AttributeError: module 'sous.engine.tileattn' has no attribute 'gpucores'` (from `tfx.tile_ready`, 32 of the 36)

- [ ] **Step 6: Add the constants and point the probe at them**

In `src/sous/engine/tileattn.py`, insert this block directly above the comment `# The probe's far parity mark is the first step boundary at or past this many`, which opens Task 6's section. Leave two blank lines on either side:

```python
# The dense Qwen3.5-family text model. The families that reuse its language
# module, qwen3_5_moe among them, would reach the same hooked global, so they
# are refused by model type, not by class.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
# DFlash drafts with mx.fast.scaled_dot_product_attention of its own, so its
# proposals never reach the hooked global. mlx-vlm's qwen3_5 MTP drafter builds
# its layers from the target's attention class over plain KVCaches: the decode
# hook would serve its proposals, and that path is unmeasured.
SUPPORTED_DRAFTER_KINDS = frozenset({"dflash"})
# Home of Qwen3_5Attention and of the global it calls after its projections.
_DECODE_MODULE = "mlx_vlm.models.qwen3_5.language"
# The one-row forward's path to that global with mask None, which decode parity
# rests on: sha256 of inspect.getsource, as verifyattn pins its own sources.
VALIDATED_DECODE_SOURCES: dict[str, frozenset[str]] = {
    "Qwen3_5Attention.__call__": frozenset(
        {"b051c66057acc90f4ed78691f90254685cfdb20d5814b0d58beec4d2bb34c184"}
    ),
    "Qwen3_5Model.__call__": frozenset(
        {"9dfc2048db0d818595bcdeb306a59e80cf1217331784fe7361c92a2dbf13ab65"}
    ),
    "LanguageModel.__call__": frozenset(
        {"af7f3f6427c2e59ab5266798b7518eaf026c3f7f749dd0962e38a6acece49cc9"}
    ),
}
```

In `probe()`, replace:

```python
    language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    installed = language.scaled_dot_product_attention
```

with:

```python
    language = importlib.import_module(_DECODE_MODULE)
    installed = language.scaled_dot_product_attention
```

- [ ] **Step 7: Implement the gates and enable()**

At the top of `src/sous/engine/tileattn.py`:
- add `import time` and `import warnings` to the stdlib imports, keeping them alphabetical;
- replace `from collections.abc import Callable` with `from collections.abc import Callable, Mapping`;
- this task is the first user of `gpucores` and `_model_type`, so replace the two sous imports:

```python
from sous.engine import nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text
```

with:

```python
from sous.engine import gpucores, nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text, _model_type
```

Append at the end of the file:

```python
def _refuse(reason: str, probe_seconds: float | None = None) -> dict[str, Any]:
    """This load cannot use the tile: say so once and report why. A default-on
    switch that is silently inert is worse than one warning line."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    warnings.warn(
        f"sous: attention tile unavailable ({reason}); "
        "decode and verify run the stock attention paths",
        stacklevel=3,
    )
    return {
        "state": "unavailable",
        "reason": reason,
        "splits": None,
        "probe_seconds": probe_seconds,
    }


def _model_reason(model: Any) -> str | None:
    """Why this model's attention is not the one the kernel is specialised
    for, or None."""
    import mlx.core as mx

    from sous.engine import verifyattn

    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)"
    modules = verifyattn._attention_modules(model)
    if not modules:
        return "no full-attention layers"
    for module in modules:
        q, kv, d = module.num_attention_heads, module.num_key_value_heads, module.head_dim
        if (q, kv, d) != (HQ, HKV, D):
            return f"unsupported attention shape {q}/{kv}/{d} (24/4/256 only)"
        if module.scale != SCALE:
            return f"attention scale {module.scale} is not 1/16"
        dtype = module.k_norm.weight.dtype
        if dtype != mx.bfloat16:
            return f"attention dtype {dtype} is not bfloat16"
    return None


def _drafter_reason(
    drafter_kind: str | None, verify_status: Mapping[str, Any] | None
) -> str | None:
    """With a drafter, the verifier must be on its exact grouped path too:
    otherwise decode runs the tile while verify runs stock SDPA, and output
    with the drafter no longer equals output without it."""
    if drafter_kind is None:
        return None
    if drafter_kind not in SUPPORTED_DRAFTER_KINDS:
        return f"drafter kind {drafter_kind!r} (dflash only)"
    state = (verify_status or {}).get("state")
    if state != "active":
        return (
            f"exact verify attention is {state}: "
            "the verifier would run stock attention beside the tile"
        )
    return None


def _decode_sources_reason() -> str | None:
    """Which function on the one-row forward's path changed, or None."""
    from sous.engine.verifyattn import _source_digest

    for qualname, digests in VALIDATED_DECODE_SOURCES.items():
        if _source_digest(_DECODE_MODULE, qualname) not in digests:
            return f"mlx-vlm {qualname} changed"
    return None


def _split_target(device_name: str) -> tuple[int | None, str | None]:
    """(S, None) when this GPU's core count is a measured split target, else
    (None, why). The guards prove exactness, not speed: an unmeasured core
    count may run the tile slower than stock."""
    cores, reason = gpucores.matched_cores(device_name)
    if cores is None:
        return None, reason
    if cores not in MEASURED_SPLITS:
        measured = ", ".join(str(s) for s in sorted(MEASURED_SPLITS))
        return None, f"split target {cores} not measured (measured: {measured})"
    return cores, None


def enable(
    model: Any,
    *,
    enabled: bool,
    drafter_kind: str | None,
    verify_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether this load's decode and verify attention run on the tile.
    Returns the status the engine exposes and never raises: a model load must
    not fail because an accelerator is missing."""
    global _active_splits
    # First, on every call: a flag left from an earlier load must never serve
    # this one, whatever happens below.
    _active_splits = 0
    if not enabled:
        return {"state": "off", "reason": None, "splits": None, "probe_seconds": None}
    seconds: float | None = None
    try:
        # Before anything imports mlx, so a GPU without tensor units reports
        # that whatever its model.
        reason = nax.platform_reason()
        if reason is not None:
            return _refuse(reason)
        import mlx.core as mx

        from sous.engine import verifyattn

        reason = (
            _model_reason(model)
            or _drafter_reason(drafter_kind, verify_status)
            or verifyattn._gate()
            or _decode_sources_reason()
        )
        if reason is not None:
            return _refuse(reason)
        splits, reason = _split_target(str(mx.device_info().get("device_name", "")))
        if splits is None:
            return _refuse(reason or "no split target")
        modules = verifyattn._attention_modules(model)
        start = time.perf_counter()
        try:
            reason = probe(modules, splits, _arch())
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds)
        install_decode_hook()
        _active_splits = splits
        logger.info(
            "attention tile: %d layers, %d splits, probe %.2fs", len(modules), splits, seconds
        )
        return {"state": "active", "reason": None, "splits": splits, "probe_seconds": seconds}
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active_splits = 0
        return _refuse(str(e) or type(e).__name__, seconds)
```

Notes on this code:
- The platform guard is the first statement inside the `try`, ahead of the first mlx import, so a GPU without tensor units reports the NAX reason whatever its model.
- Everything that can raise sits inside one `try`, whose `except` clears the flag again and refuses with the probe's seconds when the probe ran.
- The success path installs the hook and sets the flag last, so an exception anywhere before them leaves both untouched.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tileattn.py -k test_enable_ -v`

Expected: 36 passed and 2 skipped (`nax`). `test_enable_goes_active_through_the_real_probe` is the only `slow` one, at 7–13 s on the M2. The two `nax` tests run in the M5 gate.

Then run the whole file, under the GPU lock if another session may be using the GPU:

`uv run pytest tests/test_tileattn.py -v`

Expected: every test passes, and the `nax` tests skip on this M2 and on CI.

- [ ] **Step 9: Run the wider checks**

```bash
uv run pytest tests/test_verifyattn.py tests/test_int8prefill.py tests/test_nax.py tests/test_gpucores.py -v
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected:
- every test passes;
- `tile_ready` patches `nax.platform_reason` through monkeypatch only, so int8's platform tests still see the real rule;
- ruff and ty report nothing new.

- [ ] **Step 10: Commit**

```bash
git add src/sous/engine/tileattn.py tests/tile_fixtures.py tests/test_tileattn.py
git commit -m "feat(engine): gate the attention tile and enable it per load without ever failing one" -m "The tile ships on by default, so every load must decide afresh whether it
may serve, and say why when it may not.

enable() clears the flag first, so no load inherits an earlier model's
verdict. It then runs the guards in order:
- the platform rule;
- the model's attention shape, scale and dtype;
- the drafter, which needs exact verify beside it;
- verifyattn's mlx gate;
- the decode-path source hashes;
- a measured split target read from the IORegistry;
- the probe.

Only an active result installs the hook and sets the flag. Every refusal is
one warning and an unavailable status, and nothing raises into a model load.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Config, engine wiring, load line and status

**Files:**
- Modify: `src/sous/config.py`:
  - `_KNOWN` at :17-33;
  - the new field after `int8_prefill` at :102;
  - the new `_attention_tile` after `_int8_prefill` at :371-382;
  - `load_config` at :454-482.
- Modify: `src/sous/engine/vlm.py`:
  - the module imports at :11;
  - the `__init__` signature at :60-75;
  - the new call after :112-114;
  - the fork-store comment at :119-121;
  - `unload` at :532-547.
- Modify: `src/sous/engine/lm.py`: `__init__`, before the fork store at :52-55.
- Modify: `src/sous/engine/base.py`:
  - `_default_factory` at :241-332;
  - `default_engine_factory` at :335-362;
  - `ManagedEngine`, after the `draft_context_status` property at :395-399;
  - the load line in `EngineManager.get()` at :666-682;
  - `EngineManager.status()` at :991-997.
- Modify: `src/sous/tune/suite/runner.py`: the `CountingEngine.__getattr__` comment at :120-124 only.
- Test:
  - `tests/test_config.py`, after the int8 tests at :162-187;
  - `tests/test_engine_base.py`: :1148-1169, :1282-1298, :1948-1985 and the end of the file;
  - `tests/test_engine_unloaded.py`, after :85-88.

**Interfaces:**
- Consumes:
  - Task 7: `tileattn.enable(model, *, enabled, drafter_kind, verify_status) -> dict` with keys state, reason, splits and probe_seconds.
  - Task 5: `tileattn.clear()`.
  - Existing:
    - `verifyattn.enable`;
    - `VLMEngine._draft`, `_draft_kind` and `verify_attention_status`;
    - `draftctx.enable`;
    - `measure_cache_budget`;
    - the test helpers `_positional_factory`, `_positionless_model`, `_stub`, `_cfg` and `_RecordingTokenizer`, and `_unloaded_vlm` in test_engine_unloaded.py.
- Produces:
  - `SousConfig.attention_tile: bool = True`, placed after `int8_prefill`.
  - `config._attention_tile(model: dict) -> bool`.
  - `'attention_tile'` in `_KNOWN['model']`.
  - `load_config(...)` passes `attention_tile=_attention_tile(model)`.
  - `VLMEngine.__init__(..., int8_prefill: bool = False, attention_tile: bool = False, fork_dir: Path | None = None, ...)`.
  - `VLMEngine.attention_tile_status: dict[str, Any]`.
  - `VLMEngine.unload()` calls `tileattn.clear()`.
  - vlm.py's module-level import becomes `from sous.engine import draftctx, forkio, tileattn`.
  - `LMEngine.attention_tile_status = {'state': 'off', 'reason': None}`.
  - `base._default_factory(..., int8_prefill: bool = False, attention_tile: bool = False, fork_dir=..., fork_budget=...)`, passing `attention_tile` to `vlm.VLMEngine` only.
  - `default_engine_factory` passes `attention_tile=config.attention_tile`.
  - `ManagedEngine.attention_tile_status -> dict | None`, returning `getattr(self._inner, 'attention_tile_status', None)`.
  - Load line, after the draft_context tokens: `f' attention_tile={state}'`, plus, only when active, `f' attention_tile_splits={splits} attention_tile_probe_s={probe_seconds}'`.
  - `status()`, after `draft_context`: `out['attention_tile'] = {'state': tile['state'], 'reason': tile['reason']}`.

Why the defaults differ:
- The config default is `True`, which is what the daemon and `sous tune` get through `default_engine_factory(config)`.
- The engine constructors default to `False`, as `int8_prefill` does, so direct constructions keep today's paths unless they opt in: the stub tests, the model tests and the scripts.
- That also keeps `tileattn.enable` off the stub models that the existing VLMEngine tests load, whose `mlx_vlm` is a `SimpleNamespace`.

- [ ] **Step 1: Write the failing config tests**

In `tests/test_config.py`, insert after `test_int8_prefill_rejects_non_booleans_with_a_warning` (which ends at :187), before `test_speculative_defaults`:

```python
# ---- [model].attention_tile ---------------------------------------------------


def test_attention_tile_defaults_on(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x/y'\n")
    assert load_config(p).attention_tile is True
    assert SousConfig().attention_tile is True


def test_attention_tile_reads_false_without_an_unknown_key_warning(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nattention_tile = false\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.attention_tile is False


@pytest.mark.parametrize("bad", ['"yes"', "1", "0.5"])
def test_attention_tile_rejects_non_booleans_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nattention_tile = {bad}\n")
    with pytest.warns(
        UserWarning, match=r"\[model\]\.attention_tile .* must be true or false; using false"
    ):
        cfg = load_config(p)
    assert cfg.attention_tile is False
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k attention_tile -v`

Expected: 5 failed.
- `test_attention_tile_defaults_on` fails with `AttributeError: 'SousConfig' object has no attribute 'attention_tile'`.
- The false-value test fails inside the error filter with `UserWarning: sous config: unknown key 'attention_tile' in [model]`.
- The three non-bool cases fail with `Failed: Regex pattern did not match any of the 1 warnings emitted`, because the only warning is that unknown-key one.

- [ ] **Step 3: Add the config field and its parser**

In `src/sous/config.py`, replace:

```python
        "int8_prefill",
    },
}
```

with:

```python
        "int8_prefill",
        "attention_tile",
    },
}
```

Replace:

```python
    int8_prefill: bool = False
    # Reuse one KV cache across the turns of a conversation, prefilling only
```

with:

```python
    int8_prefill: bool = False
    # Decode's one-row attention and the exact verifier's on the M5's tensor
    # units, through one GQA-packed tile whose key partition steps with the
    # context: 1.22x decode on 64 real subagent turns at 44-77K (M5 Pro). On
    # by default because it is as accurate as stock and greedy output is the
    # same with the drafter on or off; it is not bit-identical to stock's
    # kernels, hence the switch, and false runs the stock paths. Active only
    # on a measured split target (the M5 Pro's 20 GPU cores), `unavailable`
    # with the reason anywhere else. Read once, at daemon start.
    attention_tile: bool = True
    # Reuse one KV cache across the turns of a conversation, prefilling only
```

Directly after `_int8_prefill` (which ends with `return False` at :382), insert, two blank lines below it:

```python
def _attention_tile(model: dict) -> bool:
    """[model].attention_tile: true or false; anything else warns and means
    false, the stock attention paths, by _int8_prefill's rule that a typo must
    not select changed numerics."""
    value = model.get("attention_tile", True)
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"sous config: [model].attention_tile {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

In `load_config`, replace:

```python
        int8_prefill=_int8_prefill(model),
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
```

with:

```python
        int8_prefill=_int8_prefill(model),
        attention_tile=_attention_tile(model),
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`

Expected: every test in the file passes, the five new ones included.

- [ ] **Step 5: Write the failing engine tests**

In `tests/test_engine_base.py`, insert after `test_default_factory_passes_int8_prefill_to_both_engines` (which ends at :1169):

```python
@pytest.mark.parametrize("value", [True, False])
def test_engine_manager_threads_attention_tile_into_default_factory(monkeypatch, tmp_path, value):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(_cfg(tmp_path, attention_tile=value)).get()
    assert seen["kwargs"]["attention_tile"] is value


def test_default_factory_passes_attention_tile_to_the_vlm_engine_only(monkeypatch):
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
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, attention_tile=True)
    assert seen["vlm"]["attention_tile"] is True
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3_5"})
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, attention_tile=True)
    assert "attention_tile" not in seen["lm"]
```

Insert after `test_lm_engine_enables_int8_prefill_on_the_loaded_model` (:1282-1298 in the file as it was before these insertions):

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


def test_lm_engine_reports_the_attention_tile_off(monkeypatch):
    from sous.engine.lm import LMEngine

    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (object(), _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    engine = LMEngine("test/model", cache_budget=0)
    assert engine.attention_tile_status == {"state": "off", "reason": None}
```

In `test_default_engine_factory_maps_every_model_value_onto_the_backend`, replace:

```python
        max_context_tokens=65536,
        int8_prefill=True,
    )
```

with:

```python
        max_context_tokens=65536,
        int8_prefill=True,
        attention_tile=False,
    )
```

In the same test, replace:

```python
        "int8_prefill": True,
        "fork_dir": None,
```

with:

```python
        "int8_prefill": True,
        "attention_tile": False,
        "fork_dir": None,
```

The config says `False` so the test proves the value is read from the config, not from a default.

Append at the end of the file:

```python
@pytest.mark.parametrize(
    ("status", "tail"),
    [
        (
            {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43},
            " positions=engine attention_tile=active attention_tile_splits=20"
            " attention_tile_probe_s=0.43",
        ),
        (
            {"state": "unavailable", "reason": "probe", "splits": None, "probe_seconds": 0.43},
            " positions=engine attention_tile=unavailable",
        ),
        ({"state": "off", "reason": None}, " positions=engine attention_tile=off"),
        (None, " model=fake/model positions=engine"),
    ],
)
def test_get_logs_the_attention_tile_state(caplog, tmp_path, status, tail):
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        if status is not None:
            engine.attention_tile_status = status  # ty: ignore[unresolved-attribute]
        return engine

    mgr = EngineManager(_cfg(tmp_path), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and lines[0].endswith(tail), lines


def test_status_carries_the_attention_tile_view_when_the_engine_reports_one(tmp_path):
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "attention_tile" not in manager.status(), "fakes without the attribute stay silent"
    inner.attention_tile_status = {  # ty: ignore[unresolved-attribute]
        "state": "unavailable",
        "reason": "split target 16 not measured (measured: 20)",
        "splits": None,
        "probe_seconds": 0.43,
    }
    assert manager.status()["attention_tile"] == {
        "state": "unavailable",
        "reason": "split target 16 not measured (measured: 20)",
    }
```

In `tests/test_engine_unloaded.py`, insert after `test_vlm_reset_prompt_cache_works_after_unload` (which ends at :88), with two blank lines on either side:

```python
def test_vlm_unload_turns_the_attention_tile_off(monkeypatch):
    from sous.engine import tileattn

    monkeypatch.setattr(tileattn, "_active_splits", 20)
    _unloaded_vlm().unload()
    assert tileattn._active_splits == 0
```

`unload()` must not need any attribute beyond the ones `_unloaded_vlm()` sets, which is why it clears the module flag rather than anything on the engine.

- [ ] **Step 6: Run the engine tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py tests/test_engine_unloaded.py -k "attention_tile or maps_every_model_value" -v`

Expected: 12 failed and 1 passed. The passing one is the load-line case for a fake without the attribute, which already ends at `positions=engine`. The failures are:
- `KeyError: 'attention_tile'` in the two manager tests (`default_engine_factory` does not pass it yet), and an `AssertionError` on the kwargs dict in `test_default_engine_factory_maps_every_model_value_onto_the_backend`;
- `TypeError: _default_factory() got an unexpected keyword argument 'attention_tile'`;
- `TypeError: VLMEngine.__init__() got an unexpected keyword argument 'attention_tile'`;
- `AttributeError: 'LMEngine' object has no attribute 'attention_tile_status'`;
- `AssertionError` on the load lines, which still end at `positions=engine`;
- `KeyError: 'attention_tile'` from `status()`;
- `assert 20 == 0` after `unload()`.

- [ ] **Step 7: Wire the engines**

In `src/sous/engine/vlm.py`, replace:

```python
from sous.engine import draftctx, forkio
```

with:

```python
from sous.engine import draftctx, forkio, tileattn
```

Replace:

```python
        int8_prefill: bool = False,
        fork_dir: Path | None = None,
        fork_budget: int | None = None,
        weights_identity: str = "",
    ):
```

with:

```python
        int8_prefill: bool = False,
        attention_tile: bool = False,
        fork_dir: Path | None = None,
        fork_budget: int | None = None,
        weights_identity: str = "",
    ):
```

Replace:

```python
        self.verify_attention_status = verifyattn.enable(
            self._model, enabled=self._draft is not None
        )
```

with:

```python
        self.verify_attention_status = verifyattn.enable(
            self._model, enabled=self._draft is not None
        )
        # After the verifier, whose status it reads: with a drafter, decode may
        # take the tile only if verify does too, or output with the drafter
        # would differ from output without it. Before the budget is measured,
        # so the probe's transient K/V is already released.
        self.attention_tile_status = tileattn.enable(
            self._model,
            enabled=attention_tile,
            drafter_kind=self._draft_kind if self._draft is not None else None,
            verify_status=self.verify_attention_status,
        )
```

Replace:

```python
        # Forks on disk, when the factory handed us a directory: built after
        # the drafter and int8 have settled, since the key reads both. Tests
        # and sous tune never pass one, so nothing they build touches ~/.sous.
```

with:

```python
        # Forks on disk, when the factory handed us a directory: built after
        # the drafter, int8 and the attention tile have settled, since the key
        # reads them. Tests and sous tune never pass one, so nothing they build
        # touches ~/.sous.
```

In `unload()`, replace:

```python
        self._draft_context.close()
        self._draft_context = draftctx.OFF
        gc.collect()
```

with:

```python
        self._draft_context.close()
        self._draft_context = draftctx.OFF
        # Until the next load decides afresh, no one-row call may reach the
        # tile on the strength of this model's gates.
        tileattn.clear()
        gc.collect()
```

In `src/sous/engine/lm.py`, replace:

```python
        self._tokenize_lock = threading.Lock()
        # Forks on disk, when the factory handed us a directory: built after
```

with:

```python
        self._tokenize_lock = threading.Lock()
        # The attention tile is an mlx-vlm qwen3_5 path that mlx-lm's models
        # never reach; reported off so the load line and the status document
        # read it from either backend.
        self.attention_tile_status = {"state": "off", "reason": None}
        # Forks on disk, when the factory handed us a directory: built after
```

- [ ] **Step 8: Wire the factory, the load line and the status document**

In `src/sous/engine/base.py`, in `_default_factory`'s signature, replace:

```python
    int8_prefill: bool = False,
    fork_dir: Path | None = None,
    fork_budget: int | None = None,
) -> Engine:
```

with:

```python
    int8_prefill: bool = False,
    attention_tile: bool = False,
    fork_dir: Path | None = None,
    fork_budget: int | None = None,
) -> Engine:
```

In its VLM branch (`:301-317`), replace:

```python
            int8_prefill=int8_prefill,
            fork_dir=fork_dir,
            fork_budget=fork_budget,
            weights_identity=weights,
        )
    # The drafter settings stop here: speculative decoding is an mlx-vlm
    # feature, and the mlx-lm backend has no parameter for it.
```

with:

```python
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

In `default_engine_factory` (`:359-362`), replace:

```python
        int8_prefill=config.int8_prefill,
        fork_dir=fork_dir,
        fork_budget=fork_budget,
    )
```

with:

```python
        int8_prefill=config.int8_prefill,
        attention_tile=config.attention_tile,
        fork_dir=fork_dir,
        fork_budget=fork_budget,
    )
```

In `ManagedEngine` (`:399`), replace:

```python
        return getattr(self._inner, "draft_context_status", None)
```

with:

```python
        return getattr(self._inner, "draft_context_status", None)

    @property
    def attention_tile_status(self) -> dict | None:
        # Optional on purpose: the VLM backend reports whether the tile serves
        # this load, the LM backend reports it off, and fakes have neither.
        return getattr(self._inner, "attention_tile_status", None)
```

In `EngineManager.get()` (`:680-682`), replace:

```python
            if context["state"] == "active":
                line += f" draft_context_window={context['window']}"
        _logger.info(line)
```

with:

```python
            if context["state"] == "active":
                line += f" draft_context_window={context['window']}"
        tile = engine.attention_tile_status
        if tile is not None:
            line += f" attention_tile={tile['state']}"
            if tile["state"] == "active":
                line += (
                    f" attention_tile_splits={tile['splits']}"
                    f" attention_tile_probe_s={tile['probe_seconds']}"
                )
        _logger.info(line)
```

In `EngineManager.status()` (`:996-997`), replace:

```python
                        "window": context["window"],
                    }
```

with:

```python
                        "window": context["window"],
                    }
                tile = self._engine.attention_tile_status
                if tile is not None:
                    out["attention_tile"] = {"state": tile["state"], "reason": tile["reason"]}
```

In `src/sous/tune/suite/runner.py`, replace:

```python
        # drafter, positions, int8_prefill_status, verify_attention_status,
        # draft_context_status: whatever the backend has, read through
        # getattr(..., None) by ManagedEngine.
```

with:

```python
        # drafter, positions, int8_prefill_status, verify_attention_status,
        # draft_context_status, attention_tile_status: whatever the backend
        # has, read through getattr(..., None) by ManagedEngine.
```

- [ ] **Step 9: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py -v`

Expected: every test passes. The existing VLMEngine stub tests still pass with no new warning: the constructor's default `attention_tile=False` returns `off` before any guard.

Then run the whole suite, under the GPU lock if another session may be using the GPU:

`uv run pytest -m "not model"`

Expected: every test passes. The only skips are `nax`-marked tests, int8's and the tile's, which skip on this M2 and on CI. The run takes about 3 minutes on the M2.

- [ ] **Step 10: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/config.py src/sous/engine/vlm.py src/sous/engine/lm.py src/sous/engine/base.py src/sous/tune/suite/runner.py tests/test_config.py tests/test_engine_base.py tests/test_engine_unloaded.py
git commit -m "feat(engine): turn the attention tile on by default with a config switch" -m "The tile measured 1.22x decode on real subagent turns and keeps greedy
parity with the drafter on or off, so the daemon runs it by default.

It is not bit-identical to stock attention, hence [model].attention_tile. A
non-bool value means false, as int8_prefill's switch does.

The engine constructors keep defaulting off, as int8 does, so tests and
scripts that build an engine directly keep today's paths. The load line and
the status document say whether the tile serves the load, and why not. An
unload clears the flag, so no call reaches the tile between models.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: The fork-store key

**Files:**
- Modify: `src/sous/engine/forkstore.py`:
  - the stdlib imports at :13-26;
  - `_EPOCH_FILES` at :57;
  - `fork_key_fields` at :146-183 and its docstring;
  - a new `os_release()` between `fork_key_fields` and `engine_epoch` at :186;
  - `engine_epoch()`'s header comment at :195-196.
- Modify: `src/sous/engine/vlm.py`: `_fork_key_fields`, at :175-187 before Task 8's edits.
- Modify: `src/sous/engine/lm.py`: `_fork_key_fields`, at :75-89 before Task 8's edits.
- Test: `tests/test_forkstore.py`:
  - the imports at :6-15;
  - :134-151 and :1387-1422;
  - new tests after each.
- Test: `tests/test_engine_forkhooks.py`: :32-47.

**Interfaces:**
- Consumes:
  - Task 8: `VLMEngine.attention_tile_status` and `LMEngine.attention_tile_status`, with keys `state` and `splits`.
  - Existing: `key_name(fields)`, `engine_epoch()`, `FORK_LAYOUT`, and the ctypes `sysctlbyname` pattern in `base.kernel_memory_pressure`.
- Produces:
  - `forkstore._EPOCH_FILES = ('vlm.py', 'lm.py', 'int8prefill.py', 'tileattn.py')`.
  - `forkstore.os_release() -> str`, `@functools.cache`: `f'{platform.mac_ver()[0]} ({build})'`, with build read from `sysctlbyname('kern.osversion')` through ctypes.
    - `'unknown'` when the build cannot be read;
    - `''` off macOS.
  - `fork_key_fields(*, backend: str, backend_version: str, weights: str, gpu: str, positions: str, int8_status: Mapping[str, Any], tile_status: Mapping[str, Any]) -> dict[str, str]`:
    - `'tile'` is `f"active:{tile_status['splits']}"` when `tile_status.get('state') == 'active'`, else `'off'`;
    - `'os'` is `os_release()` when the tile or int8 is active.
  - vlm.py and lm.py `_fork_key_fields` pass `tile_status=self.attention_tile_status`.

Why the tile is in the key:
- The tile serves the one-row forward that ends every prefill segment, and the fork floor of 4,096 keys is above N0. So it moves the last bits of that token's KV and, through its hidden state, every later layer's KV and GDN state.
- `unavailable` keeps the flag clear, so it has `off`'s numerics and shares `off`'s key.

Why the OS build is in the key: runtime-compiled kernels, the tile and int8 prefill alike, can change their last bits with macOS.

Consequences:
- Landing this changes both the key and the epoch, which cold-starts every fork store once.
- While the tile or int8 is active, a macOS update cold-starts it again.
- The merge gate writes both into the PR.

- [ ] **Step 1: Write the failing tests**

In `tests/test_forkstore.py`, add `import platform` to the stdlib imports, between `import os` and `import re`.

In `test_engine_epoch_changes_when_a_source_byte_changes`, replace:

```python
    for name in ("vlm.py", "lm.py", "int8prefill.py"):
        (src / name).write_text(f"# {name}\n")
```

with:

```python
    for name in ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py"):
        (src / name).write_text(f"# {name}\n")
```

In the same test, replace:

```python
    assert engine_epoch() != changed  # the headers are compiled into the kernels
    forkstore.engine_epoch.cache_clear()
```

with:

```python
    assert engine_epoch() != changed  # the headers are compiled into the kernels
    changed = engine_epoch()
    (src / "tileattn.py").write_text("# tileattn.py changed\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() != changed  # the tile's schedule decides which kernel runs
    forkstore.engine_epoch.cache_clear()


def test_the_attention_tile_module_is_an_epoch_file():
    assert "tileattn.py" in forkstore._EPOCH_FILES
```

In `test_fork_key_fields_carry_everything_that_changes_the_kv`:

Replace:

```python
        int8_status={"state": "active", "reason": None, "routed": 336},
    )
    assert f["backend"] == "vlm" and f["backend_version"] == "0.7.1"
```

with:

```python
        int8_status={"state": "active", "reason": None, "routed": 336},
        tile_status={"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43},
    )
    assert f["backend"] == "vlm" and f["backend_version"] == "0.7.1"
```

Replace:

```python
    assert f["positions"] == "engine" and f["int8"] == "active:336"
    assert f["env"] == "MLX_ENABLE_TF32=1"
```

with:

```python
    assert f["positions"] == "engine" and f["int8"] == "active:336"
    assert f["tile"] == "active:20" and f["os"] == forkstore.os_release()
    assert f["env"] == "MLX_ENABLE_TF32=1"
```

Replace:

```python
        int8_status={"state": "unavailable", "reason": "no tensor units", "routed": 0},
    )
    assert off["int8"] == "off"  # off and unavailable ran the same numerics
```

with:

```python
        int8_status={"state": "unavailable", "reason": "no tensor units", "routed": 0},
        tile_status={"state": "off", "reason": None},
    )
    assert off["int8"] == "off"  # off and unavailable ran the same numerics
    assert off["tile"] == "off" and "os" not in off
```

Replace the test's last call:

```python
        positions="model",
        int8_status={"state": "off", "reason": None, "routed": 0},
    )
```

with:

```python
        positions="model",
        int8_status={"state": "off", "reason": None, "routed": 0},
        tile_status={"state": "off", "reason": None},
    )
```

Append at the end of `tests/test_forkstore.py`:

```python
_TILE_OFF = {"state": "off", "reason": None, "splits": None, "probe_seconds": None}
_TILE_20 = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43}
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


def test_the_key_moves_with_the_tile_and_its_split_target():
    """A store written with the tile is never restored without it, or with
    another S; unavailable ran off's numerics, so it shares off's key."""
    unavailable = {**_TILE_OFF, "state": "unavailable", "reason": "macOS 15.5 < 26.2"}
    tile_16 = {**_TILE_20, "splits": 16}
    names = {key_name(_vlm_fields(tile=t)) for t in (_TILE_20, tile_16, _TILE_OFF)}
    assert len(names) == 3
    assert _vlm_fields(tile=unavailable)["tile"] == "off"
    assert key_name(_vlm_fields(tile=unavailable)) == key_name(_vlm_fields())


def test_the_os_build_is_in_the_key_while_a_runtime_compiled_kernel_is_active(monkeypatch):
    int8 = {"state": "active", "reason": None, "routed": 336}
    assert "os" not in _vlm_fields()
    assert _vlm_fields(int8=int8)["os"] == forkstore.os_release()
    assert _vlm_fields(tile=_TILE_20)["os"] == forkstore.os_release()
    off, tiled = key_name(_vlm_fields()), key_name(_vlm_fields(tile=_TILE_20))
    monkeypatch.setattr(forkstore, "os_release", lambda: "15.6 (24G84)")
    assert key_name(_vlm_fields(tile=_TILE_20)) != tiled
    assert key_name(_vlm_fields()) == off  # a macOS update leaves a stock store alone


@pytest.mark.skipif(platform.system() != "Darwin", reason="reads this Mac's build")
def test_os_release_is_the_version_and_the_build():
    release = forkstore.os_release()
    assert re.fullmatch(r"\d+(\.\d+)+ \([0-9A-Za-z]+\)", release), release
    assert release.startswith(f"{platform.mac_ver()[0]} (")
```

In `tests/test_engine_forkhooks.py`, replace both key-field tests, :32-47:

```python
def test_vlm_key_fields_read_positions_and_the_realised_int8_state(monkeypatch):
    engine = vlm.VLMEngine.__new__(vlm.VLMEngine)
    engine._positional = True
    engine.int8_prefill_status = {"state": "active", "reason": None, "routed": 12}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "vlm" and fields["positions"] == "engine"
    assert fields["int8"] == "active:12" and fields["weights"] == "sha"
    assert fields["gpu"]  # mx.device_info()["architecture"] on this Mac


def test_lm_key_fields_name_the_lm_backend(monkeypatch):
    engine = lm.LMEngine.__new__(lm.LMEngine)
    engine.int8_prefill_status = {"state": "off", "reason": None, "routed": 0}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off"
```

with:

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
    assert fields["gpu"]  # mx.device_info()["architecture"] on this Mac


def test_lm_key_fields_name_the_lm_backend(monkeypatch):
    engine = lm.LMEngine.__new__(lm.LMEngine)
    engine.int8_prefill_status = {"state": "off", "reason": None, "routed": 0}
    engine.attention_tile_status = {"state": "off", "reason": None}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off" and fields["tile"] == "off"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py -v`

Expected: 8 failed, and the rest pass:
- the epoch test: `AssertionError`, since the tileattn.py byte change does not move the epoch;
- `assert 'tileattn.py' in ('vlm.py', 'lm.py', 'int8prefill.py')`;
- the three key-field tests: `TypeError: fork_key_fields() got an unexpected keyword argument 'tile_status'`;
- the OS test: `AttributeError: module 'sous.engine.forkstore' has no attribute 'os_release'`;
- the two engine key-field tests: `KeyError: 'tile'`.

- [ ] **Step 3: Implement the key fields and the OS reading**

In `src/sous/engine/forkstore.py`, add `import platform` to the stdlib imports, between `import os` and `import shutil`.

Replace:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py")
```

with:

```python
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py", "tileattn.py")
```

Replace `fork_key_fields` from its `positions: str,` parameter through its `fields = {` line:

```python
    positions: str,
    int8_status: Mapping[str, Any],
) -> dict[str, str]:
    """Everything that changes the KV arrays behind identical token ids, and
    nothing else: the backend and its package, mlx, the GPU, the weights,
    the engine sources (epoch) and the file layout, which side supplies the
    rotary positions, whether int8 prefill actually ran (`off` and
    `unavailable` are the same numerics), and the two mlx-core environment
    switches that change matmul precision and attention accumulation. The
    template, tokenizer, sampling, drafter and window are deliberately
    absent: ids are compared exactly, and none of them touches a prefill."""
    from importlib.metadata import version

    fields = {
```

with:

```python
    positions: str,
    int8_status: Mapping[str, Any],
    tile_status: Mapping[str, Any],
) -> dict[str, str]:
    """Everything that changes the KV arrays behind identical token ids, and
    nothing else: the backend and its package, mlx, the GPU, the weights,
    the engine sources (epoch) and the file layout, which side supplies the
    rotary positions, whether int8 prefill actually ran, whether the
    attention tile served the load and at which split target, the macOS
    build while either of those runtime-compiled kernels is active, and the
    two mlx-core environment switches that change matmul precision and
    attention accumulation.

    The tile runs the one-row forward that ends every prefill segment, so it
    moves the last bits of that token's KV and, through its hidden state,
    every later layer's KV and GDN state. Both kernels are compiled at load
    against the OS's Metal toolchain, whose output can change with a macOS
    update. For both, `unavailable` runs `off`'s numerics. The template,
    tokenizer, sampling, drafter and window are deliberately absent: ids are
    compared exactly, and none of them touches a prefill."""
    from importlib.metadata import version

    int8_active = int8_status.get("state") == "active"
    tile_active = tile_status.get("state") == "active"
    fields = {
```

Replace:

```python
        "int8": (
            f"active:{int8_status.get('routed', 0)}"
            if int8_status.get("state") == "active"
            else "off"
        ),
    }
    env = ";".join(f"{k}={os.environ[k]}" for k in _NUMERIC_ENV if k in os.environ)
    if env:
        fields["env"] = env
    return fields
```

with:

```python
        "int8": f"active:{int8_status.get('routed', 0)}" if int8_active else "off",
        "tile": f"active:{tile_status['splits']}" if tile_active else "off",
    }
    if int8_active or tile_active:
        fields["os"] = os_release()
    env = ";".join(f"{k}={os.environ[k]}" for k in _NUMERIC_ENV if k in os.environ)
    if env:
        fields["env"] = env
    return fields


@functools.cache
def os_release() -> str:
    """This Mac's macOS version and build, e.g. "15.5 (24F74)": the build
    moves with every update, even one that keeps the version. "unknown" when
    the build cannot be read, "" off macOS. A libc call, no subprocess."""
    if platform.system() != "Darwin":
        return ""
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buf = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(len(buf))
        if libc.sysctlbyname(b"kern.osversion", buf, ctypes.byref(size), None, 0) != 0:
            return "unknown"
        build = buf.value.decode("ascii")
    except Exception:  # noqa: BLE001 — a key field that degrades, never a failure
        return "unknown"
    return f"{platform.mac_ver()[0]} ({build})" if build else "unknown"
```

`fork_key_fields` calls `os_release()` through the module global, so the test's monkeypatch reaches it.

In `engine_epoch()`, the header comment still says the `.h` files go into "both kernels"; they now also go into the tile's. Replace:

```python
    # The .h files are compiled into both kernels (the nibble-decode and
    # K-order contract lives there), so they are part of the numerics too.
```

with:

```python
    # The .h files are compiled into every kernel (int8's nibble-decode and
    # K-order contract, the tile's lane layout), so they are part of the
    # numerics too.
```

- [ ] **Step 4: Pass the tile's status from both engines**

In `src/sous/engine/vlm.py`, in `_fork_key_fields`, replace:

```python
            positions=self.positions,
            int8_status=self.int8_prefill_status,
        )
```

with:

```python
            positions=self.positions,
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
        )
```

In `src/sous/engine/lm.py`, in `_fork_key_fields`, replace:

```python
            positions="model",
            int8_status=self.int8_prefill_status,
        )
```

with:

```python
            positions="model",
            int8_status=self.int8_prefill_status,
            tile_status=self.attention_tile_status,
        )
```

Both engines set `attention_tile_status` before they build the fork store (Task 8), so the key always has it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py tests/test_engine_unloaded.py -v`

Expected: every test passes. That includes the LM engine built with stubs and a fork directory (`test_a_working_store_construction_leaves_forks_on`), whose key now reads the `off` tile status Task 8 set before the store.

- [ ] **Step 6: Run the wider checks**

```bash
uv run pytest tests/test_engine_base.py tests/test_engine_forkio.py tests/test_promptcache.py -v
uv run ruff format . && uv run ruff check . && uv run ty check
```

Expected: every test passes, and ruff and ty report nothing.

- [ ] **Step 7: Commit**

```bash
git add src/sous/engine/forkstore.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_forkstore.py tests/test_engine_forkhooks.py
git commit -m "feat(engine): key on-disk forks by the attention tile and the macOS build" -m "The tile runs the one-row forward that ends every prefill segment, so a
fork written with it holds different last bits than one written without
it, or at another split target.

Both runtime-compiled kernels are built against the OS's Metal toolchain,
so their output can move with a macOS update. The key therefore records the
tile's state and S, and, while the tile or int8 prefill is active, the
macOS version and build. unavailable shares off's key, since it runs off's
numerics.

Landing this cold-starts every fork store once, because both the key and
the engine epoch change. A macOS update does it again while either kernel
is active.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: sous tune keys and the unavailable notice

**Files:**
- Modify: `src/sous/tune/arms.py`:
  - `:34-45`, the mirrored fields;
  - `:54-58`, `suite_key`;
  - `:123-135`, `_arm`.
- Modify: `src/sous/tune/suite/runner.py`:
  - `:54-92`, `SuiteRun`;
  - `:207-230`, `_error_run`;
  - `:233-287`, `run_one`;
  - a new `_check_tile` after `_check_int8` (`:290-312`);
  - the call site at `:395-396`.
- Modify: `src/sous/tune/decide.py:151`, the `ArmSummary.key` annotation only. `summarize` stores `arm.suite_key` there, and ty rejects a six-tuple where a five-tuple is declared ("a tuple of length 5 is not assignable to a tuple of length 6").
- Modify: `docs/tuning.md:50-56`.
- Test: `tests/test_tune_arms.py`, `tests/test_tune_runner.py`.
- Modify: the tests that build rows by hand:
  - `tests/test_tune_main.py:125-147` and `:547-567`, the two `SuiteRun(...)` calls;
  - `tests/test_tune_decide.py:217-240`, `_run`;
  - `tests/test_tune_report.py:286-300`, `_summary`'s literal key, for the same ty reason.

**Interfaces:**
- Consumes:
  - Task 8: `SousConfig.attention_tile` and `ManagedEngine.attention_tile_status`.
  - Existing: `Arm`, `_arm`, `quick_arms`, `winner_stage_arms`, `SuiteRun`, `_check_int8`, `run_suite`, and `decide.py`'s use of `arm.suite_key`.
- Produces:
  - `Arm.attention_tile: bool = True`, after `greedy`, mirrored from the config.
  - `Arm.suite_key -> tuple[str, str, int, bool, bool, bool]` = `(*self.key, self.int8_prefill, self.greedy, self.attention_tile)`.
  - `_arm` sets `attention_tile=user.attention_tile`.
  - `SuiteRun.attention_tile: bool`, required, after `greedy`.
  - `SuiteRun.key -> tuple[str, str, int, bool, bool, bool]` = `(model_id, drafter_id, block_size, int8_prefill, greedy, attention_tile)`.
  - `SuiteRun.from_dict` fills a missing `'attention_tile'` with `False`, because rows written before the tile existed ran stock attention.
  - `runner._check_tile(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None`. It never raises.
    - When `arm.attention_tile` is set and the engine reports anything but active, it prints `f'  {arm.label}: attention tile unavailable here ({reason}); the engine runs stock, as the daemon would'`.
    - `reason` is the status's own reason, else `engine reports <state>`, or `engine reports nothing` when there is no status.
  - `decide.ArmSummary.key: tuple[str, str, int, bool, bool, bool]`.
  - Unchanged:
    - `winner_stage_arms` adds no tile arm; the field rides through `dataclasses.replace`.
    - `decide._changes` proposes no tile change.

- [ ] **Step 1: Write the failing arms tests**

In `tests/test_tune_arms.py`, replace `_winner` (`:252-271`) with:

```python
def _winner(tmp_path, temperature=0.7, int8=False, tile=True):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
        attention_tile=tile,
    )
    return Arm(
        label="27B + DFlash2 @3",
        config=cfg,
        model_id="mlx-community/Qwen3.8-27B-4bit",
        drafter_id="z-lab/Qwen3.8-27B-DFlash2",
        block_size=3,
        window=131072,
        tier="27b-dense",
        current=True,
        fit_window=131072,
        int8_prefill=int8,
        greedy=temperature == 0,
        attention_tile=tile,
    )
```

Change the three existing `suite_key` assertions:

- In `test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions` (`:277`), `    assert arm.suite_key == (*arm.key, False, False)` becomes:

  ```python
      assert arm.suite_key == (*arm.key, False, False, True)
  ```
- In `test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_a_sampled_winner_a_greedy_arm` (`:287-288`), these two lines:

  ```python
      assert int8.suite_key == (*int8.key, True, False)
      assert greedy.suite_key == (*greedy.key, False, True)
  ```
  become:

  ```python
      assert int8.suite_key == (*int8.key, True, False, True)
      assert greedy.suite_key == (*greedy.key, False, True, True)
  ```
- In `test_quick_arms_mirror_the_users_int8_and_greedy_settings` (`:345`), `        assert arm.suite_key == (*arm.key, True, True)` becomes:

  ```python
          assert arm.suite_key == (*arm.key, True, True, True)
  ```

Append to the end of the file:

```python


def test_quick_arms_mirror_the_users_attention_tile_and_key_their_suite_rows_by_it(tmp_path):
    keys = {}
    for tile in (True, False):
        user = SousConfig(
            data_dir=tmp_path / "d", config_path=tmp_path / "c.toml", attention_tile=tile
        )
        arms, refusals = quick_arms(
            user, _candidates(), _checkpoints(), working_set_bytes=fx.M5_PRO_WORKING_SET
        )
        assert refusals == []
        for arm in arms:
            assert arm.attention_tile is tile and arm.config.attention_tile is tile
            assert arm.suite_key == (*arm.key, False, False, tile)
        keys[tile] = {arm.suite_key for arm in arms}
    # A run resumed after the setting was flipped matches none of the old rows.
    assert keys[True].isdisjoint(keys[False])


def test_the_winner_stage_carries_the_attention_tile_and_adds_no_arm_for_it(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path, tile=False), nax=True, checkpoint=cp)
    assert [a.label for a in arms] == ["27B + DFlash2 @3 + int8 prefill", "27B + DFlash2 @3 greedy"]
    assert all(a.attention_tile is False and a.config.attention_tile is False for a in arms)
    assert [a.suite_key[-1] for a in arms] == [False, False]
```

- [ ] **Step 2: Run the arms tests to verify they fail**

Run: `uv run pytest tests/test_tune_arms.py -v`

Expected failures:
- Every test that builds `_winner` fails with `TypeError: Arm.__init__() got an unexpected keyword argument 'attention_tile'`.
- `test_quick_arms_mirror_the_users_int8_and_greedy_settings` fails its six-tuple assertion.
- `test_quick_arms_mirror_the_users_attention_tile_and_key_their_suite_rows_by_it` fails with `AttributeError: 'Arm' object has no attribute 'attention_tile'`.

The other arms tests pass.

- [ ] **Step 3: Add the field, the key and the mirror to `arms.py`**

In `src/sous/tune/arms.py`, replace:

```python
    # The quality-affecting dimensions the winner stage measures, mirrored
    # from `config` so a row can be keyed without reading the config back.
    int8_prefill: bool = False
    greedy: bool = False
```

with:

```python
    # The quality-affecting dimensions the winner stage measures, mirrored
    # from `config` so a row can be keyed without reading the config back.
    int8_prefill: bool = False
    greedy: bool = False
    # Mirrored the same way but never measured: every arm inherits the
    # user's setting. It still keys suite rows, because tile and stock
    # attention differ in their last bits.
    attention_tile: bool = True
```

Replace:

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key plus the two settings only the suite may change."""
        return (*self.key, self.int8_prefill, self.greedy)
```

with:

```python
    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the two settings only the suite may change, and the attention
        tile every arm inherits."""
        return (*self.key, self.int8_prefill, self.greedy, self.attention_tile)
```

In `_arm`, replace:

```python
        int8_prefill=user.int8_prefill,
        greedy=user.temperature == 0,
    )
```

with:

```python
        int8_prefill=user.int8_prefill,
        greedy=user.temperature == 0,
        attention_tile=user.attention_tile,
    )
```

- [ ] **Step 4: Run the arms tests to verify they pass**

Run: `uv run pytest tests/test_tune_arms.py -v`
Expected: all pass.

- [ ] **Step 5: Write the failing runner tests and update the hand-built rows**

In `tests/test_tune_runner.py`:

1. Add `import dataclasses` as the first line of the stdlib import block, above `import json`.

2. In `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` (`:120`), change `    assert run.key == (*_arm(tmp_path).key, False, False)` to:

   ```python
       assert run.key == (*_arm(tmp_path).key, False, False, True)
   ```

3. In `test_an_arm_that_merely_inherits_int8_runs_when_the_engine_cannot_route_it`, replace the last line, `    assert any("runs stock" in line for line in lines)`, with the lines below. `FakeEngine` has no tile status, so this arm now also prints the tile's notice, and that notice says "runs stock" too:

   ```python
       # The inherited attention tile's notice says "runs stock" too: match int8's own line.
       assert any("int8 prefill unavailable here (no tensor units)" in line for line in lines)
   ```

4. Replace `test_suite_runs_round_trip_through_dicts` (`:511-536`) with:

   ```python
   def test_suite_runs_round_trip_through_dicts(tmp_path):
       run = SuiteRun(
           task="t",
           index=0,
           label="m",
           model_id="org/m",
           drafter_id="",
           block_size=0,
           int8_prefill=False,
           greedy=True,
           attention_tile=False,
           window=8192,
           state="done",
           outcome="completed",
           turns=3,
           seconds=12.5,
           output_tokens=400,
           malformed=0,
           repetitions=0,
           approvals_denied=1,
           grade=0.5,
           grade_detail="1/2",
           error=None,
           transcript_path=None,
       )
       assert SuiteRun.from_dict(json.loads(json.dumps(run.as_dict()))) == run
       assert run.key == ("org/m", "", 0, False, True, False)
   ```

5. Append to the end of the file:

   ```python


   def test_a_row_written_before_the_attention_tile_existed_reads_as_stock(tmp_path):
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
       del row["attention_tile"]
       old = SuiteRun.from_dict(row)
       assert old.attention_tile is False
       # The arm inherits the tile's default, so a resume runs the task again
       # rather than counting the stock row toward the tile arm.
       assert arm.attention_tile is True
       assert old.key == (*arm.key, False, False, False)
       assert old.key != arm.suite_key


   class _TileUnavailable(FakeEngine):
       drafter = ""
       attention_tile_status = {
           "state": "unavailable",
           "reason": "macOS 15.5 < 26.2",
           "splits": None,
           "probe_seconds": None,
       }


   def test_an_arm_that_inherits_the_tile_runs_stock_where_the_engine_cannot_serve_it(tmp_path):
       """Every arm inherits attention_tile from the user's config, and none is
       built to measure it: an engine that reports the tile unavailable must
       not refuse the arm, since the daemon would serve that checkpoint with
       stock attention and one warning."""
       lines = []
       outcome = run_suite(
           _arm(tmp_path),
           [_task()],
           runs=1,
           done=set(),
           record=lambda r: None,
           scratch=tmp_path / "s",
           out=lines.append,
           factory=lambda mid: _TileUnavailable([FINISH]),
           python=PYTHON,
           active_memory=lambda: 0,
       )
       assert len(outcome.runs) == 1 and outcome.error is None
       assert outcome.runs[0].attention_tile is True
       assert (
           "  m: attention tile unavailable here (macOS 15.5 < 26.2); "
           "the engine runs stock, as the daemon would"
       ) in lines


   def test_no_tile_notice_when_the_tile_is_active_or_the_arm_runs_stock(tmp_path):
       class Active(FakeEngine):
           drafter = ""
           attention_tile_status = {
               "state": "active",
               "reason": None,
               "splits": 20,
               "probe_seconds": 0.4,
           }

       arm = _arm(tmp_path)
       stock = dataclasses.replace(
           arm, attention_tile=False, config=dataclasses.replace(arm.config, attention_tile=False)
       )
       for case, engine in ((arm, Active), (stock, _TileUnavailable)):
           lines = []
           outcome = run_suite(
               case,
               [_task()],
               runs=1,
               done=set(),
               record=lambda r: None,
               scratch=tmp_path / f"s-{case.attention_tile}",
               out=lines.append,
               factory=lambda mid, engine=engine: engine([FINISH]),
               python=PYTHON,
               active_memory=lambda: 0,
           )
           assert len(outcome.runs) == 1 and outcome.error is None
           assert outcome.runs[0].attention_tile is case.attention_tile
           assert not any("attention tile" in line for line in lines), lines
   ```

In `tests/test_tune_main.py`, add the field to both hand-built rows. The two edits differ only in indentation.

In `_suite` (`:125`, 20-space indent), replace:

```python
                    greedy=arm.greedy,
                    window=arm.window,
```

with:

```python
                    greedy=arm.greedy,
                    attention_tile=arm.attention_tile,
                    window=arm.window,
```

In `test_a_stopped_suite_still_reports_the_runs_it_recorded` (`:547`, 12-space indent), replace:

```python
            greedy=arm.greedy,
            window=arm.window,
```

with:

```python
            greedy=arm.greedy,
            attention_tile=arm.attention_tile,
            window=arm.window,
```

In `tests/test_tune_decide.py`, `_run` (`:220`), replace:

```python
        greedy=arm.greedy,
        window=arm.window,
        state=state,
```

with:

```python
        greedy=arm.greedy,
        attention_tile=arm.attention_tile,
        window=arm.window,
        state=state,
```

In `tests/test_tune_report.py`, `_summary` (`:289`), change `        key=(model, "d", 3, False, False),` to:

```python
        key=(model, "d", 3, False, False, True),
```

- [ ] **Step 6: Run the runner tests to verify they fail**

Run: `uv run pytest tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`

Expected: 29 failed and 71 passed. The failures are:
- Every test that builds a `SuiteRun` by hand, including the whole of `tests/test_tune_main.py`'s suite path, fails with `TypeError: SuiteRun.__init__() got an unexpected keyword argument 'attention_tile'`.
- `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` fails its six-tuple assertion.
- `test_an_arm_that_inherits_the_tile_runs_stock_where_the_engine_cannot_serve_it` and `test_no_tile_notice_when_the_tile_is_active_or_the_arm_runs_stock` fail with `AttributeError: 'SuiteRun' object has no attribute 'attention_tile'` before their notice assertions.

- [ ] **Step 7: Key the row, fall back for old rows, add the notice**

In `src/sous/tune/suite/runner.py`, replace:

```python
    int8_prefill: bool
    greedy: bool
    window: int
```

with:

```python
    int8_prefill: bool
    greedy: bool
    attention_tile: bool
    window: int
```

Replace:

```python
    @property
    def key(self) -> tuple[str, str, int, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (self.model_id, self.drafter_id, self.block_size, self.int8_prefill, self.greedy)
```

with:

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

Replace:

```python
    @classmethod
    def from_dict(cls, d: dict) -> SuiteRun:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})
```

with:

```python
    @classmethod
    def from_dict(cls, d: dict) -> SuiteRun:
        # A row written before the tile existed ran stock attention; without
        # the fallback, resuming such a run would raise KeyError.
        d = {"attention_tile": False, **d}
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})
```

In both `_error_run` (`:216-217`) and `run_one` (`:273-274`), replace every occurrence of this text (it appears twice, so use the Edit tool's `replace_all`):

```python
        greedy=arm.greedy,
        window=arm.window,
```

with:

```python
        greedy=arm.greedy,
        attention_tile=arm.attention_tile,
        window=arm.window,
```

Add `_check_tile` after `_check_int8`. Replace:

```python
    out(
        f"  {arm.label}: int8 prefill unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )


def _slug(label: str) -> str:
```

with:

```python
    out(
        f"  {arm.label}: int8 prefill unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )


def _check_tile(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """Every arm inherits the attention tile from the user's config and none
    exists to measure it, so unlike int8 there is nothing to fail: wherever
    the engine reports the tile anything but active, the daemon would serve
    this checkpoint with stock attention and one warning, and the arm runs
    the same way after a notice."""
    if not arm.attention_tile:
        return
    status = engine.attention_tile_status or {}
    if status.get("state") == "active":
        return
    reason = status.get("reason") or f"engine reports {status.get('state', 'nothing')}"
    out(
        f"  {arm.label}: attention tile unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )


def _slug(label: str) -> str:
```

At the call site in `_run_suite`, replace:

```python
            _check_drafter(arm, engine)
            _check_int8(arm, engine, out)
```

with:

```python
            _check_drafter(arm, engine)
            _check_int8(arm, engine, out)
            _check_tile(arm, engine, out)
```

In `src/sous/tune/decide.py`, `ArmSummary`, replace `    key: tuple[str, str, int, bool, bool]` with:

```python
    key: tuple[str, str, int, bool, bool, bool]
```

- [ ] **Step 8: Run the tune tests to verify they pass**

Run: `uv run pytest tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py -v`
Expected: all pass.

- [ ] **Step 9: Say in the tuning docs how the tile is keyed**

In `docs/tuning.md`, replace:

```markdown
flag to understand. The full run may therefore change `[model].id`, the
drafter and block size, the window, `int8_prefill` and `temperature`.
```

with:

```markdown
flag to understand. The full run may therefore change `[model].id`, the
drafter and block size, the window, `int8_prefill` and `temperature`.
Every arm inherits `[model].attention_tile` and no arm changes it; suite
rows are keyed by it, so a `--resume` after it was flipped runs those arms
again instead of mixing tile and stock rows, and where the tile cannot run,
the arm prints a notice and runs stock, as the daemon would.
```

- [ ] **Step 10: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add src/sous/tune/arms.py src/sous/tune/suite/runner.py src/sous/tune/decide.py docs/tuning.md tests/test_tune_arms.py tests/test_tune_runner.py tests/test_tune_main.py tests/test_tune_decide.py tests/test_tune_report.py
git commit -m "feat(tune): key suite rows by the attention tile and say where it cannot run" -m "A resumed tune matched suite rows on the bench key plus int8 prefill and
greedy, so a run resumed after [model].attention_tile was flipped would
have pooled tile and stock rows into one arm's grade. Arms now mirror the
setting, it joins both keys, and rows written before it existed read as
stock. An arm whose engine cannot run the tile prints a notice and runs
stock, as the daemon would, instead of failing: no arm exists to measure
the tile.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and the full non-model suite passes.

---

### Task 11: The mlx-vlm contract test and the real-model test

**Files:**
- Create: `tests/test_tileattn_contract.py`
- Create: `tests/test_tileattn_model.py`

**Interfaces:**
- Consumes:
  - Task 5: `tfx.tiny_language_model`, `tfx.prefill_cache`, `tfx.verify_forward`, `tfx.hooked`, `tfx.verify_ready` and `tfx.VOCAB`.
  - Task 5: `tileattn.calls`, including `verify_t2`, `verify_t8`, `verify_rows_stock` and `verify_straddles`.
  - Task 4: `tfx.stand_in_tile`, `tfx.nax`, `tileattn.availability()`.
  - Task 3: `tileattn.verify_plan`, `tileattn.N0`, `tileattn.MEASURED_SPLITS`.
  - Task 2: `gpucores.matched_cores(device_name)`.
  - Task 8:
    - `VLMEngine(model_id, ..., draft_id=..., draft_block_size=..., attention_tile=...)`;
    - `attention_tile_status`;
    - `unload()`;
    - `tileattn._active_splits`.
  - Task 7: `enable()`, which the engine calls.
  - verifyattn: `calls`, `_TAG`, reached through `tfx.verify_ready`.
- Produces: tests only.

- [ ] **Step 1: Write the contract test**

Create `tests/test_tileattn_contract.py`:

```python
"""The attention tile through mlx-vlm's real qwen3_5 forward and exact verifier.

A tiny random language model at the 27B's attention shape (24 query and 4 KV heads, head dim 256,
bf16, two full-attention layers) is prefilled through the real forward. Verify rows, as DFlash
runs them, are compared with step-by-step one-row decode on a fresh prefill of the same prefix:
a hybrid cache cannot rewind past a decode step. Main's own exactness is asserted first; the
tile (the plain-mlx stand-in on any GPU, the real kernel under ``nax``) must then keep every verify
row bitwise equal to decode, and with the flag cleared both paths must give main's bits again.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import tileattn  # noqa: E402 — after the importorskip guard
from tests import tile_fixtures as tfx  # noqa: E402

SPLITS = 20
LAYERS = 2  # the tiny model's full-attention layers: every forward makes two attention calls
TILE_CASES = [
    (1000, 2),  # below N0: grouped stock rows, T = 2 included
    (1000, 3),
    (1533, 5),  # two stock rows, then three tile rows from N0
    (1700, 1),
    (1700, 2),  # the DFlash round loop's last round
    (1700, 3),
    (1700, 8),
    (1918, 5),  # straddles the first step boundary above N0 (1921 at S = 20)
]


@pytest.fixture(scope="module")
def tiny():
    lm = tfx.tiny_language_model()
    ids = mx.random.randint(0, tfx.VOCAB, (2600,), key=mx.random.key(132)).tolist()
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


def _verify_counts(prefix, t):
    plan = tileattn.verify_plan(prefix, t, SPLITS)
    tile_rows = sum(k - j for j, k, chunk in plan if chunk)
    counts = {
        "verify_calls": LAYERS,
        f"verify_t{t}": LAYERS,
        "verify_rows_tile": LAYERS * tile_rows,
        "verify_rows_stock": LAYERS * (t - tile_rows),
        "verify_launches": LAYERS * sum(1 for *_, chunk in plan if chunk),
        "verify_straddles": LAYERS * (len(plan) > 1),
    }
    return {k: v for k, v in counts.items() if v}


def _decode_counts(prefix, t):
    tile = sum(1 for r in range(t) if prefix + r + 1 >= tileattn.N0)
    counts = {"decode_tile": LAYERS * tile, "decode_stock": LAYERS * (t - tile)}
    return {k: v for k, v in counts.items() if v}


@pytest.mark.parametrize(("prefix", "t"), [(1000, 2), (1000, 3), (1700, 3)])
def test_mains_verify_rows_equal_step_by_step_decode(tiny, prefix, t, monkeypatch):
    """Main's own exactness on this GPU, which every tile assertion below
    rests on. Should it ever fail on CI, quantize the tiny model
    (nn.quantize(lm, group_size=64, bits=4)), as the 27B is, rather than
    weaken a tile assertion."""
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    assert tileattn._active_splits == 0
    assert all(_same(_verify_rows(lm, ids, prefix, t), _decode_rows(lm, ids, prefix, t)))


@pytest.mark.slow
@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=tfx.nax)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize(("prefix", "t"), TILE_CASES)
def test_the_tile_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, prefix, t, stand_in, monkeypatch
):
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    main_verify = _verify_rows(lm, ids, prefix, t)
    main_decode = _decode_rows(lm, ids, prefix, t)
    assert all(_same(main_verify, main_decode)), "main's own exactness"

    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=stand_in)
    start = dict(tileattn.calls)
    verify = _verify_rows(lm, ids, prefix, t)
    verified = dict(tileattn.calls)
    decode = _decode_rows(lm, ids, prefix, t)
    assert all(_same(verify, decode)), (prefix, t)
    assert _delta(start, verified) == _verify_counts(prefix, t)
    assert _delta(verified, tileattn.calls) == _decode_counts(prefix, t)

    monkeypatch.setattr(tileattn, "_active_splits", 0)
    start = dict(tileattn.calls)
    assert all(_same(_verify_rows(lm, ids, prefix, t), main_verify))
    assert all(_same(_decode_rows(lm, ids, prefix, t), main_decode))
    assert tileattn.calls == start


def test_a_kernel_that_ignores_the_schedule_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: rows 1718..1722 share chunk 96, but a kernel
    that sizes its splits from the call's own key count (ceil(n / 20)) gives
    the verify call and the one-row call at 1,718 different partitions, so
    the equality above is not vacuous."""
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=True)
    monkeypatch.setattr(
        tileattn, "tile", lambda q, k, v, n, chunk: tfx.stand_in_tile(q, k, v, n, -(-n // 20))
    )
    assert not all(_same(_verify_rows(lm, ids, 1717, 5), _decode_rows(lm, ids, 1717, 5)))
```

- [ ] **Step 2: Run the contract test**

Run: `uv run pytest tests/test_tileattn_contract.py -v -rs`

Expected on the maintainer's M2 and on CI: `12 passed, 8 skipped`.
- The skips are the `kernel` variants, with reason `attention tile unavailable: macOS 15.5 < 26.2` on the M2 and the platform reason on CI.
- These tests pin behaviour that Tasks 4–8 built, so they pass as written. A failed parity or counter assertion is a bug in those tasks: fix it there, and never loosen the assertion.
- Main's precondition was measured on the M2 before planning. With this exact model, these ids (`mx.random.key(132)`) and every `TILE_CASES` prefix and T, verify rows equalled one-row decode bit for bit, unquantized.
- If `test_mains_verify_rows_equal_step_by_step_decode` ever fails on CI, add these lines to the `tiny` fixture after `tfx.tiny_language_model()` and keep every tile assertion as it is:

  ```python
      import mlx.nn as nn

      nn.quantize(lm, group_size=64, bits=4)
      mx.eval(lm.parameters())
  ```

- [ ] **Step 3: Write the real-model test**

Create `tests/test_tileattn_model.py`:

```python
"""The attention tile on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 30 minutes, most of it five cold prefills of the
~57.5K-token prompt with the prompt cache off):
- An engine with the tile reports it active; the test skips, naming the reason, where it cannot be.
- Greedy output with the DFlash2 drafter equals output without it, at block 3 and at block 8 (set
  on the engine directly: config caps the block at 5), on a ~1,450-token prompt whose generation
  crosses N0 (1,536 keys) and the first step boundary above it (1,921), and on a prompt ending just
  below the 57,600-key step boundary.
- Over the tile runs, the counters show verify rows below N0, a T = 2 round, straddling verify
  calls and, at block 8, T = 8 on the tile, with nothing declined or out of scope.
- An engine with the tile off, built after the tile engines unloaded, reproduces the output of
  one built before any tile engine ran in the process, which is main's.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# name -> (prompt tokens, max_tokens values). A response's last round verifies T = 2 only when
# the cap leaves exactly one token to emit; block-3 rounds emit one to three tokens, so one of
# three consecutive caps lands on it.
PROMPTS = {"short": (1450, (520, 521, 522)), "long": (57_560, (160,))}
SLACK = 16


@contextlib.contextmanager
def _engine(*, tile: bool, drafter: bool) -> Iterator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=3 if drafter else 0,
        attention_tile=tile,
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


def _generate(engine, prompts, blocks):
    """({(block, prompt, max_tokens): text}, {block: tile counter deltas}); block 0 keeps the
    engine's own block size."""
    texts, counts = {}, {}
    for block in blocks:
        if block:
            engine._draft_block_size = block
        before = dict(tileattn.calls)
        for name, (messages, caps) in prompts.items():
            for cap in caps:
                texts[(block, name, cap)] = engine.generate(messages, [], cap)
        counts[block] = {k: v - before[k] for k, v in tileattn.calls.items()}
    return texts, counts


def test_greedy_output_with_the_drafter_equals_output_without_it_on_the_tile():
    avail = tileattn.availability()
    if not avail.available:
        pytest.skip(f"attention tile unavailable: {avail.reason}")
    cores, why = gpucores.matched_cores(str(mx.device_info()["device_name"]))
    if cores not in tileattn.MEASURED_SPLITS:
        pytest.skip(f"attention tile unavailable: {why or f'split target {cores} not measured'}")

    # First in the process, before any tile engine could install the decode hook: main's paths.
    with _engine(tile=False, drafter=True) as main:
        assert main.attention_tile_status["state"] == "off", main.attention_tile_status
        ids = _corpus(main, max(target for target, _ in PROMPTS.values()) + 64)
        prompts = {
            name: (_prompt(main, ids, target), caps) for name, (target, caps) in PROMPTS.items()
        }
        reference, _ = _generate(main, prompts, (3,))

    with _engine(tile=True, drafter=False) as plain:
        status = plain.attention_tile_status
        if status["state"] != "active":
            pytest.skip(f"attention tile {status['state']}: {status['reason']}")
        undrafted, plain_counts = _generate(plain, prompts, (0,))

    with _engine(tile=True, drafter=True) as drafted:
        assert drafted.attention_tile_status["state"] == "active", drafted.attention_tile_status
        drafted_texts, drafted_counts = _generate(drafted, prompts, (3, 8))

    with _engine(tile=False, drafter=True) as off:
        assert off.attention_tile_status["state"] == "off", off.attention_tile_status
        assert tileattn._active_splits == 0
        again, off_counts = _generate(off, prompts, (3,))

    for (_, name, cap), text in undrafted.items():
        assert drafted_texts[(3, name, cap)] == text, ("block 3", name, cap)
        assert drafted_texts[(8, name, cap)] == text, ("block 8", name, cap)
    assert again == reference
    tile = {
        k: plain_counts[0][k] + drafted_counts[3][k] + drafted_counts[8][k] for k in tileattn.calls
    }
    assert plain_counts[0]["decode_tile"] > 0, plain_counts
    assert tile["verify_rows_stock"] > 0, tile
    assert tile["verify_t2"] > 0, tile
    assert tile["verify_straddles"] > 0, tile
    assert drafted_counts[8]["verify_t8"] > 0, drafted_counts
    assert tile["decode_out"] == tile["verify_out"] == tile["verify_declined"] == 0, tile
    assert not any(off_counts[3].values()), off_counts
```

- [ ] **Step 4: Check that the model test collects and skips where the tile cannot run**

Run: `uv run pytest -m model tests/test_tileattn_model.py -v -rs`
Expected on the maintainer's M2: `1 skipped` with reason `attention tile unavailable: macOS 15.5 < 26.2`. No model loads, because the platform rule refuses first.

Run: `uv run pytest -m "not model" tests/test_tileattn_model.py`
Expected: `1 deselected`, which is how CI treats it. The test itself runs on the M5 in Task 13.

- [ ] **Step 5: Lint, type-check, run the suite and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
uv run pytest -m "not model"
git add tests/test_tileattn_contract.py tests/test_tileattn_model.py
git commit -m "test(engine): pin the attention tile's greedy parity through mlx-vlm's real forward" -m "The unit tests prove the kernel entries and hooks in isolation; this pins
the property the feature exists for through mlx-vlm's own forward and
exact verifier: every verify row equals step-by-step decode at its own key
count below N0, across it and across a step boundary, with the stand-in on
any GPU and the real kernel on the M5. The manual model test holds the
real 27B's drafter-on output to drafter-off at blocks 3 and 8, and a
tile-off engine to main.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Expected: ruff and ty report nothing, and the full non-model suite passes, including the new `slow` parity cases.

---

### Task 12: Documentation

**Files:**
- Modify: `CLAUDE.md`:
  - the epoch-file list at `:163-164`;
  - the int8 bullets at `:281-303`;
  - the verifyattn bullet at `:304-323`;
  - a new bullet after it, before the draftctx bullet at `:324`.
- Modify: `docs/configuration.md`: the TOML block at `:8-28`, and a new section after the int8 section at `:138-152`.
- Modify: `docs/observability.md`: the load-line paragraph at `:182-193`.
- Modify: `docs/how-it-works.md:249-253`. The fork-store paragraph lists what a stored fork must match, and it gains the tile and the macOS build. This is a stale sentence the fork-key change leaves behind.

**Interfaces:**
- Consumes:
  - Every name pinned in Tasks 1–10: `tileattn`'s `N0`, `MAX_RUN`, `MEASURED_SPLITS`, `enable`, `clear`, `calls` and `availability`; `nax.platform_reason()`; `gpucores.read()`.
  - The fork-key fields `tile` and `os`.
  - The load-line tokens `attention_tile=`, `attention_tile_splits=` and `attention_tile_probe_s=`.
  - `[model].attention_tile`.
- Produces: docs only.

- [ ] **Step 1: Add `tileattn.py` to CLAUDE.md's epoch list**

In `CLAUDE.md`, replace:

```markdown
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, kernels/*.metal
  and *.h) beside the versions, the GPU and the weights snapshot — any
```

with:

```markdown
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  kernels/*.metal and *.h) beside the versions, the GPU and the weights
  snapshot — any
```

- [ ] **Step 2: Amend the int8 bullets: `nax.h` and the shared platform rule**

In `CLAUDE.md`, replace:

```markdown
  `model_type` `qwen3_5` (the MoE variant reuses the same classes and is refused), and only the
  GEMM's header includes the Metal-4 `MetalPerformancePrimitives` header, which
  needs macOS 26.2+ — the Stage-A kernel must keep compiling on any Metal GPU
  so CI (macos-15, no tensor units) can test it.
```

with:

```markdown
  `model_type` `qwen3_5` (the MoE variant reuses the same classes and is refused), and only the
  tensor-op kernels — the GEMM and the attention tile's split kernel — get
  `nax.h`, whose Metal-4 `MetalPerformancePrimitives` include needs macOS
  26.2+: the Stage-A kernel and the tile's reduce kernel stay MPP-free, and
  Stage A must keep compiling on any Metal GPU so CI (macos-15, no tensor
  units) can test it.
```

Replace:

```markdown
  reject a kernel that compiled before. `int8prefill.availability()` therefore
  compiles and runs the GEMM once where the tensor units exist (`_gemm_probe`,
  GPU work only, success remembered, the full compiler report logged): a
  rejected kernel reads as `unavailable` and the `nax` tests skip, where a
  compile failure with a CPU-stream op in flight deadlocks in mlx's exception
  path. Being mlx ops, it inherits the thread rules above.
```

with:

```markdown
  reject a kernel that compiled before. The platform half of the rule (macOS
  >= 26.2, GPU generation >= 17, >= 18 for 'p' parts) is
  `nax.platform_reason()`, shared by both tensor-op kernels; past it
  `int8prefill.availability()` compiles and runs the GEMM once where the
  tensor units exist (`_gemm_probe`) and `tileattn.availability()` every tile
  variant (`_compile_probe`) — GPU work only, success remembered, the full
  compiler report logged: a rejected kernel reads as `unavailable` and that
  kernel's `nax` tests skip, where a compile failure with a CPU-stream op in
  flight deadlocks in mlx's exception path. Being mlx ops, both inherit the
  thread rules above.
```

- [ ] **Step 3: Amend verifyattn's T-range and "T = 2 is left alone"**

In `CLAUDE.md`, replace:

```markdown
- `engine/verifyattn.py` replaces mlx-vlm's per-row SDPA loop in the exact
  verifier (`Qwen3_5BatchInvariantForward._attention`, T = 3..8) with one
  stock `mx.fast.scaled_dot_product_attention` per group of rows that share
  mlx's kernel plan. Grouping is bit-exact only because inside one plan mlx
```

with:

```markdown
- `engine/verifyattn.py` replaces mlx-vlm's per-row SDPA loop in the exact
  verifier (`Qwen3_5BatchInvariantForward._attention`, T = 3..8 while the
  attention tile is inactive) with one stock
  `mx.fast.scaled_dot_product_attention` per group of rows that share mlx's
  kernel plan. Grouping is bit-exact only because inside one plan mlx
```

Replace:

```markdown
  environment can resolve a newer mlx-vlm than the lock. T = 2 is left
  alone: stock already makes one call there, and grouping it would change
  output at plan straddles. Scope is decided before the projections —
```

with:

```markdown
  environment can resolve a newer mlx-vlm than the lock. While the tile is
  inactive T = 2 is left alone: stock already makes one call there, and
  grouping it would change output at plan straddles. While it is active the
  wrapper's scope widens to every T from 1 and the rows go to `tileattn`
  (next bullet). Scope is decided before the projections —
```

- [ ] **Step 4: Add the tile's gotcha**

In `CLAUDE.md`, insert this bullet after the verifyattn bullet, which ends `verify output is bit-identical.`, and before the bullet that starts ``- `engine/draftctx.py` hands the DFlash drafter``:

```markdown
- `engine/tileattn.py` serves decode's one-row attention and the exact
  verifier's attention from one GQA-packed MPP tile on the M5's tensor
  units (`kernels/attention_tile.metal`: a split kernel and a fixed-order
  reduce, specialised to bf16, 24/4/256 heads and scale 1/16). Greedy
  parity with the drafter on or off holds only because a row's output
  depends on its chunk and nothing else about the call, and the chunk is
  fixed per 32·S-key step: `chunk_for(n, S)` is 0 (stock) below `N0` =
  1536 keys and `32 * ceil(n / (32 * S))` from there. `verify_plan` cuts a
  T-row verify into runs at every step boundary, at `N0` and every
  `MAX_RUN` = 8 rows, so any T stays exact; runs below `N0` go to
  `verifyattn.grouped_attention` with tileattn's own `_arch()`, never
  `verifyattn._ARCH`. Decode replaces one module global,
  `mlx_vlm.models.qwen3_5.language.scaled_dot_product_attention`, on the
  first `enable()` that reaches `active` and never earlier. The global
  cannot tell which module calls it, so its scope rests on the flag plus
  the post-projection checks (one row, mask and sinks None, `in_scope`,
  n ≥ `N0`, `type(cache) is KVCache` without bits or left padding): the
  families that reuse the qwen3_5 language model are refused by model
  type, DFlash drafters call `mx.fast` directly and MTP drafters are
  refused. Deciding after the projections is safe because the fallback is
  the very call the method made, so nothing is appended twice. Verify rows
  arrive through verifyattn's wrapper, whose scope widens to every T from
  1 while the flag is set. The flag `_active_splits` is a plain int, the
  active S or 0, never a model reference: the first statement of every
  `enable()` clears it, only an `active` result sets it, and
  `VLMEngine.unload()` clears it through `tileattn.clear()`. `enable()`
  checks, in order: the config; the shared platform rule
  (`nax.platform_reason()`); the model (qwen3_5, every full-attention
  module at 24/4/256, scale 1/16, bf16); the drafter (none, or `dflash`
  with verify attention active — else the verifier would run stock beside
  a tiled decode and drafter-on output would stop equalling drafter-off);
  verifyattn's whole `_gate()`; the source hashes of the three decode
  functions (`VALIDATED_DECODE_SOURCES`); the split target, the GPU core
  count read once per process by `gpucores.read()` (IOKit, then
  `/usr/sbin/ioreg`), which must be in `MEASURED_SPLITS` — a chip joins
  only by passing the merge gate's kernel timings, since the guards prove
  exactness, not speed; and a probe on the `sous-model-load` thread. The
  probe compiles every T = 1..8, proves verify rows `array_equal` to
  decode at `N0`, the first two step boundaries and the first one past
  57K keys, plants NaN past n to prove every read stops at n, compares
  with stock one-row SDPA within 1e-2 relative RMS (parity cannot catch a
  kernel that is wrong the same way on both paths) and runs one of the
  model's own attention modules through a restored-style `KVCache`. Like
  int8 it never raises: every refusal is one warning plus `unavailable`.
  The engine constructors default it off (`VLMEngine(attention_tile=False)`,
  like `int8_prefill`); only `default_engine_factory` hands them the
  config's `true`, so tests and scripts keep today's paths unless they opt
  in. Nothing is logged per turn; `tileattn.calls` counts every path for
  the tests and the merge gate. The fork key records `tile` (`active:S`,
  else `off`) and, while the tile or int8 prefill is active, `os` (the
  macOS build, whose Metal toolchain compiles both kernels at load);
  `tileattn.py` is in `_EPOCH_FILES`. Every kernel edit and every
  `VALIDATED_MLX` extension re-runs the tile's exactness tests and its
  T = 1, 3 and 8 timings: codegen is fragile, and `sg >> 1` for `sg / 2`
  kept the output but ran 1.4x slower. Tests install the decode hook
  through monkeypatch, monkeypatch S (never IOKit) and swap the plain-mlx
  stand-in in for `tileattn.tile`; the `nax` tests skip on
  `tileattn.availability()`, never int8's.
```

- [ ] **Step 5: Document `[model].attention_tile` in the configuration reference**

In `docs/configuration.md`, replace:

```toml
speculative_block_size = 3
int8_prefill = false
```

with:

```toml
speculative_block_size = 3
int8_prefill = false
attention_tile = true
```

Replace:

```markdown
a checkpoint with no eligible projection warns once; in a mixed checkpoint, ineligible
projections fall through per projection.
```

with:

```markdown
a checkpoint with no eligible projection warns once; in a mixed checkpoint, ineligible
projections fall through per projection.

`[model].attention_tile` (default `true`) runs decode's one-row attention and
speculative verify's attention on the M5 GPU's neural accelerators, as one
GQA-packed tile whose key partition steps with the context (sous's own
kernel, compiled at model load — no build step). Measured on an M5 Pro with
the default model and drafter: the verify forward's attention at block 3 is
2–2.6x faster than the grouped stock calls it replaces, and decode is 1.22x
faster on 64 real subagent turns at 44–77K of context. Greedy output with the
drafter stays identical to output without it — a verify row and the decode
step at the same key count use the same partition — and the tile is as
accurate as the stock kernels, but not bit-identical to them: a near-tie can
resolve the other way than with `false`, which runs the stock paths. It is
active only where it has been measured: a 20-core M5 Pro GPU on macOS 26.2+,
running a dense Qwen3.5-family model (`model_type` `qwen3_5`, 24 query and 4
KV heads, head dim 256, bf16 attention, as the default model is) with no
drafter or a DFlash one. Anywhere else the model-load line reads
`attention_tile=unavailable`, the status document's `attention_tile` block
carries the reason, and decode and verify run stock. The on-disk forks are
keyed by it: changing it, or updating macOS while it is active, starts the
fork store cold once. The daemon reads it at startup, so a change takes
effect on its next start.
```

- [ ] **Step 6: Document the load-line tokens**

In `docs/observability.md`, replace:

```markdown
`unavailable` a drafter this cannot seed — the status document's
`draft_context` block carries the reason). One
more line names the Anthropic tool *types* a turn dropped, when any.
```

with:

```markdown
`unavailable` a drafter this cannot seed — the status document's
`draft_context` block carries the reason), and on both backends
`attention_tile=active|unavailable|off` (whether decode's and speculative
verify's attention run on the tensor-unit tile, with
`attention_tile_splits=` and `attention_tile_probe_s=` when it does; `off`
means `[model].attention_tile = false` or the mlx-lm backend, and
`unavailable` a load the tile cannot serve — the status document's
`attention_tile` block carries the reason). One
more line names the Anthropic tool *types* a turn dropped, when any.
```

- [ ] **Step 7: Name the tile and the macOS build in how-it-works' fork-store paragraph**

In `docs/how-it-works.md`, replace:

```markdown
(mlx-vlm or mlx-lm), mlx version, GPU, weights snapshot, engine sources,
positions owner and int8 state match the ones that wrote it; anything else
```

with:

```markdown
(mlx-vlm or mlx-lm), mlx version, GPU, weights snapshot, engine sources,
positions owner, int8 state and attention-tile state (and, while either
kernel is active, macOS build) match the ones that wrote it; anything else
```

- [ ] **Step 8: Check nothing stale is left, and commit**

```bash
grep -n "T = 2 is left" CLAUDE.md
grep -n "attention_tile" docs/configuration.md docs/observability.md
git diff --stat
```

Expected:
- The first grep prints exactly one line, the amended one: `  inactive T = 2 is left alone: stock already makes one call there, and`. No unamended copy is left.
- The second grep shows the TOML line, the new section and the load-line text.
- The diff touches exactly `CLAUDE.md`, `docs/configuration.md`, `docs/observability.md` and `docs/how-it-works.md`.

```bash
uv run ruff check . && uv run ruff format --check .
git add CLAUDE.md docs/configuration.md docs/observability.md docs/how-it-works.md
git commit -m "docs: record how the attention tile stays exact and how to switch it off" -m "The tile's parity rests on a stepped partition, verify runs cut at steps,
at N0 and at eight rows, a module-global decode hook scoped by a plain
flag, and a load-time probe; each is a trap for whoever next edits the
kernel or bumps mlx or mlx-vlm, so CLAUDE.md records them beside
verifyattn's, whose T = 2 rule now holds only while the tile is inactive.
The user docs say what [model].attention_tile does, where it runs, what the
load line and the status document report, and that a change, or a macOS
update while it is active, starts the fork store cold.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: The M5 Pro merge gate

This task is manual measurement, not code, and nothing it writes out of tree is ever committed.

**Where it runs.** The M5 Pro, reached over ssh. The `m5.sh` helper (Step 3) reaches the M5 by its LAN alias `m5pro-lan` and checks it against the host key recorded for its tailnet name. It reads that name from `ssh -G m5pro` at run time, so this plan names no machine.

**What it never touches:** `~/code/personal/sous`, `~/.sous` and the maintainer's daemon.

**GPU lock.** Every M5 GPU job runs under `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py`. It waits for the daemon to be idle and prints `daemon stayed idle during the job`, or `DAEMON-CONTAMINATED`. Every GPU job on this Mac runs under `~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py`.

**Time.** About nine hours of M5 GPU time:

| Part | Time |
|---|---|
| Replay chain 1 | 2.2 h |
| Replay chain 2 | 4 h |
| Suite | 45 min |
| Model tests | 40 min |
| Kernel timings | 15 min |

**On any failure, stop and report the numbers. Open no PR.**

**Files:**
- Out of tree, never committed:
  - Create on this Mac, then rsync to `m5pro-lan:code/personal/sous-spikes/tile-132/gate/`, in `~/code/sous-spikes/tile-132/gate/`:
    - `m5.sh`;
    - `wrap_branch.py`, the GATE_WRAP counter shim;
    - `run_gate.sh`;
    - `suite_ab.py`;
    - `ktime.py`;
    - `suite/config.toml`.
  - Copy on the M5, from `~/code/personal/sous-spikes/tile-132/model/` into `tile-132/gate/`:
    - `gate132.py`, `analyze132.py`, `analyze133.py`, `analyze_suite.py`, `bench_model.py`;
    - `cmp_decode.py`, which pools the greedy report over identical-output turns and imports `analyze132`.
  - M5 checkouts:
    - `~/code/personal/sous-remote`, on main;
    - a new detached worktree `~/code/personal/sous-remote-tile`, at the branch head.
  - M2 (this Mac): `~/code/sous-spikes/tile-132/m2/` holds:
    - the #143 harness copy: `gate143_m2.py`, `bodies-small.json` and `build_bodies.py`;
    - the worktrees `main/` and `branch/`, each with its own `.venv`.
- May modify in-tree, only if the gate justifies it:
  - `README.md:54-56` and `:147-148`, the decode figures;
  - `docs/configuration.md`, the decode figure in the `[model].attention_tile` paragraph.

**Interfaces:**
- Consumes:
  - `tileattn.calls` and `verifyattn.calls`.
  - `sous.engine.vlm.VLMEngine.generate(self, messages, tools, max_tokens, on_delta=None)` and `VLMEngine.attention_tile_status`.
  - `sous.cli.main(argv)`.
  - The load-line tokens, and the `/sous/status` `engine.attention_tile` block.
  - `gate132.py`'s interface:
    - `GATE_WRAP`: runs `<tree>/.venv/bin/python <script>` in place of `sous serve`;
    - `TILE132_COUNTERS`: the path the shim writes, read after each turn;
    - `--model 'key = value'` extra lines;
    - picks written as `basename@i,j,k`;
    - results in `runs/<label>/result.json`;
    - it writes `max_context_tokens = 262144`, an isolated HOME and port 8384, and deletes the run's forks.
  - `gpu_run_m5.py`, `detach.py` and `wait_for.py` in `~/code/personal/sous-spikes/splash-port-2026-09-24/`, and `gpu_run.py` in `~/code/sous-spikes/splash-port-2026-09-24/`.
- Produces:
  - The gate verdict and its tables.
  - The PR, only after a passing gate.
  - Optionally, a `docs:` commit updating the README and configuration decode figures.

- [ ] **Step 1: Check the M2 harness copy made before Task 1**

```bash
mkdir -p ~/code/sous-spikes/tile-132/m2 ~/code/sous-spikes/tile-132/gate/suite
ls ~/code/sous-spikes/tile-132/m2/gate143_m2.py ~/code/sous-spikes/tile-132/m2/bodies-small.json ~/code/sous-spikes/tile-132/m2/build_bodies.py
```

Expected: all three files are listed. The copy should have been made before Task 1, because `/private/tmp` can be cleared at any time. If any file is missing, copy them now:

```bash
src="$(dirname "$(find /private/tmp -path '*scratchpad/m2/gate143_m2.py' 2>/dev/null | head -1)")"
cp -n "$src/gate143_m2.py" "$src/bodies-small.json" "$src/build_bodies.py" ~/code/sous-spikes/tile-132/m2/
```

If `find` returns nothing, the M2 criterion cannot run as specified: stop and report.

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
git push --force-with-lease -u origin feat/attention-tile
git rev-parse --short HEAD
```

- [ ] **Step 3: Write the ssh helper**

Create `~/code/sous-spikes/tile-132/gate/m5.sh`:

```sh
#!/bin/sh
# ssh to the M5 Pro checked against the host key recorded for its tailnet name, so
# its LAN alias works when Tailscale's DNS does not: `m5.sh m5pro-lan '<command>'`,
# and as rsync's remote shell: `rsync -e ~/code/sous-spikes/tile-132/gate/m5.sh ...`.
exec ssh -o HostKeyAlias="$(ssh -G m5pro | awk '$1 == "hostname" {print $2}')" "$@"
```

```bash
chmod +x ~/code/sous-spikes/tile-132/gate/m5.sh
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'sw_vers -productVersion; sysctl -n kern.osversion'
```

Expected: the M5's macOS version and build (27.0 and 26A428 when this plan was written).

- [ ] **Step 4: Prepare the M5 trees**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote && git fetch -q origin && git checkout -q main && git merge -q --ff-only origin/main && uv sync -q && git log -1 --oneline'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote && if [ -d ~/code/personal/sous-remote-tile ]; then git -C ~/code/personal/sous-remote-tile checkout -q --detach origin/feat/attention-tile; else git worktree add -q --detach ~/code/personal/sous-remote-tile origin/feat/attention-tile; fi && cd ~/code/personal/sous-remote-tile && uv sync -q && git log -1 --oneline && mkdir -p ~/code/personal/sous-spikes/tile-132/gate/logs ~/code/personal/sous-spikes/tile-132/gate/suite'
```

Expected:
- The main checkout prints origin/main's head.
- The branch worktree prints the head that Step 2 printed.

- [ ] **Step 5: Run the non-model suite on the M5**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/tile-132/gate/logs/notmodel.txt env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py uv run pytest -m "not model" -rs'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/notmodel.txt 540 30'
```

Repeat the `wait_for.py` call until it prints `DONE`. Then:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'L=~/code/personal/sous-spikes/tile-132/gate/logs/notmodel.txt; grep -c "attention tile unavailable" $L; grep -E " passed| failed| error" $L | tail -2; tail -2 $L'
```

Expected:
- The count is `0`, so every tile `nax` test ran, the contract test's `kernel` variants included.
- The summary line has passes and skips only, with no `failed` or `error`.
- The log ends `[gpu_run_m5] daemon stayed idle during the job` and `[gpu_run_m5] exit=0`.
- Re-run the step if the log says `DAEMON-CONTAMINATED`.

- [ ] **Step 6: Run the model tests on the M5**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/tile-132/gate/logs/model.txt env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py uv run pytest -m model -rs tests/test_tileattn_model.py tests/test_verifyattn_model.py'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/model.txt 540 20'
```

Repeat the `wait_for.py` call until `DONE`; it takes about 40 minutes. Expected:
- `2 passed`, with no skip.
- The log ends `daemon stayed idle during the job` and `exit=0`.

- [ ] **Step 7: Write the GATE_WRAP counter shim**

Create `~/code/sous-spikes/tile-132/gate/wrap_branch.py`:

```python
"""GATE_WRAP shim for the attention-tile merge gate (out of tree, never committed).

gate132.py runs `<tree>/.venv/bin/python wrap_branch.py` in place of `sous serve`. This runs the
branch's own `sous serve` unchanged and, after every VLMEngine.generate(), atomically writes to
$TILE132_COUNTERS what gate132.py records with each turn: the turn's call-counter deltas, the
process totals and the engine's attention_tile status. The counters are tileattn.calls plus
verifyattn.calls under a `verifyattn_` prefix.

usage: <tree>/.venv/bin/python wrap_branch.py [sous cli args...]   (no args = serve)
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time

from sous.engine import tileattn, verifyattn, vlm

COUNTERS = os.environ.get("TILE132_COUNTERS", "")
_turns = 0


def counters() -> dict[str, int]:
    out = dict(tileattn.calls)
    out.update({f"verifyattn_{k}": v for k, v in verifyattn.calls.items()})
    return out


def _write(snapshot: dict) -> None:
    if not COUNTERS:
        return
    tmp = COUNTERS + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp, COUNTERS)
    except OSError as e:
        print(f"wrap_branch: counters not written ({e})", file=sys.stderr, flush=True)


def install() -> None:
    original = vlm.VLMEngine.generate

    @functools.wraps(original)
    def generate(self, messages, tools, max_tokens, on_delta=None):
        global _turns
        before = counters()
        start = time.perf_counter()
        try:
            return original(self, messages, tools, max_tokens, on_delta)
        finally:
            now = counters()
            _turns += 1
            _write(
                {
                    "turns": _turns,
                    "state": dict(getattr(self, "attention_tile_status", None) or {}),
                    "last_turn": {k: v - before.get(k, 0) for k, v in now.items()},
                    "totals": now,
                    "last_turn_seconds": round(time.perf_counter() - start, 3),
                }
            )

    vlm.VLMEngine.generate = generate


if __name__ == "__main__":
    install()
    from sous.cli import main

    main(sys.argv[1:] or ["serve"])
```

- [ ] **Step 8: Write the run chain**

Create `~/code/sous-spikes/tile-132/gate/run_gate.sh`:

```zsh
#!/bin/zsh
# The attention-tile merge gate's replay runs, one after another, each its own GPU-lock job with
# a hard timeout (out of tree, never committed; adapted from tile-132/model/run133.sh).
# usage: run_gate.sh <done-file> <spec>...   spec = label:tree:temp:maxtok:variant:turnset
#   tree     main (~/code/personal/sous-remote, `sous serve`) or branch
#            (~/code/personal/sous-remote-tile, through wrap_branch.py)
#   variant  - | nodraft (speculative_draft_id = "") | off (attention_tile = false)
#   turnset  acc (gate132's default: #131's 64 turns) | sub16 | full (sub16 plus a6443762's
#            continuation to 228,630 tokens: 23 turns)
G=~/code/personal/sous-spikes/tile-132/gate
L=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
MAIN=~/code/personal/sous-remote
BRANCH=~/code/personal/sous-remote-tile
PY=$MAIN/.venv/bin/python
DONE=$1; shift
SUB16=(agent-a160cc42252609969.jsonl@0,4,9,14,19 agent-a7bb20c376e462d83.jsonl@2,6
       agent-a51b90a20d6d54ff6.jsonl@1,7 agent-a257d5dad84e70f90.jsonl@1,3,6
       agent-a442f184ac548b226.jsonl@0,3,7,11)
FULL=($SUB16 agent-a6443762163413a72.jsonl@0,9,108,176,239,300,365)
mkdir -p $G/logs
cd $MAIN
rm -f $DONE $DONE.part
for spec in "$@"; do
  parts=(${(s/:/)spec})
  label=$parts[1]; tree=$parts[2]; temp=$parts[3]; maxtok=$parts[4]; variant=$parts[5]; turnset=$parts[6]
  extra=(--model 'prompt_cache_gb = 14')
  case $variant in
    nodraft) extra+=(--model 'speculative_draft_id = ""');;
    off) extra+=(--model 'attention_tile = false');;
  esac
  case $turnset in
    acc) picks=();;
    sub16) picks=($SUB16);;
    full) picks=($FULL);;
    *) echo "$label: unknown turn set $turnset" >> $DONE.part; break;;
  esac
  case $tree in
    main) dir=$MAIN; wrap=;;
    branch) dir=$BRANCH; wrap=$G/wrap_branch.py;;
    *) echo "$label: unknown tree $tree" >> $DONE.part; break;;
  esac
  env GATE_WRAP=$wrap GPU_LABEL=tile-gate python3 $L perl -e 'alarm shift @ARGV; exec @ARGV' 5400 \
    $PY $G/gate132.py $label $dir $temp $maxtok $extra $picks > $G/logs/$label.log 2>&1
  echo "$label exit=$? $(date +%H:%M:%S)" >> $DONE.part
  # nothing of ours may be left on the gate port before the next run
  if lsof -nP -iTCP:8384 -sTCP:LISTEN >/dev/null 2>&1; then echo "$label: port 8384 still busy" >> $DONE.part; break; fi
  if pgrep -f "gate132.py $label" >/dev/null || pgrep -f "runs/$label" >/dev/null; then echo "$label: process left" >> $DONE.part; break; fi
done
mv $DONE.part $DONE
```

- [ ] **Step 9: Write the suite A/B and its config**

Create `~/code/sous-spikes/tile-132/gate/suite/config.toml`:

```toml
[server]
port = 8384
upstream_url = "http://127.0.0.1:9"

[model]
max_context_tokens = 262144
temperature = 0.7
speculative_block_size = 3
```

Create `~/code/sous-spikes/tile-132/gate/suite_ab.py`:

```python
"""The attention-tile merge gate's tool-loop suite (out of tree, never committed; adapted from
tile-132/model/suite133.py): sous tune's eight tasks on the default model at block 3 and
temperature 0.7, [model].attention_tile true ('tile') against false ('stock'), in one process on
the branch tree.

run_suite loads the engine once per call and releases it at the end, so every arm and round gets
a fresh load, and each load's tileattn.enable() runs with the arm's own setting: the stock arm's
load clears the flag the tile arm's load set, which is the lifecycle the daemon relies on. Each
row carries the per-run deltas of tileattn.calls and verifyattn.calls, which prove the path:
stock runs make no tile call. Rows keep suite133's shape, so analyze_suite.py reads them as is.

usage: suite_ab.py <out.jsonl> <first_round> <last_round>   (rounds inclusive; one run per task per
       arm per round; even rounds run stock then tile, odd rounds tile then stock)
env:   SUITE_AB_CONFIG  config.toml (default <this dir>/suite/config.toml); run under an isolated HOME
"""

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

from sous.config import load_config
from sous.engine import tileattn, verifyattn
from sous.engine.base import default_engine_factory
from sous.tune.arms import Arm
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import run_suite

HERE = os.path.dirname(os.path.abspath(__file__))


def counters() -> dict[str, int]:
    out = dict(tileattn.calls)
    out.update({f"verifyattn_{k}": v for k, v in verifyattn.calls.items()})
    return out


def main():
    out_path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("sous.engine").setLevel(logging.INFO)
    cfg = load_config(Path(os.environ.get("SUITE_AB_CONFIG", os.path.join(HERE, "suite", "config.toml"))))
    base = dataclasses.replace(cfg, speculative_block_size=3, temperature=0.7, int8_prefill=False)
    shown = ("model_id", "speculative_draft_id", "speculative_block_size", "temperature",
             "max_context_tokens", "prompt_cache_gb", "int8_prefill")
    print("config:", {k: getattr(base, k) for k in shown}, flush=True)
    arms = {}
    for name, on in (("stock", False), ("tile", True)):
        config = dataclasses.replace(base, attention_tile=on)
        arms[name] = Arm(label=name, config=config, model_id=config.model_id,
                         drafter_id=config.speculative_draft_id, block_size=3,
                         window=config.max_context_tokens, tier="", current=False, attention_tile=on)
    tasks = load_tasks()
    print("tasks:", [t.name for t in tasks], flush=True)
    scratch_root = Path(out_path).with_suffix(".scratch")
    with open(out_path, "a") as out:
        for rnd in range(first, last + 1):
            order = ["stock", "tile"] if rnd % 2 == 0 else ["tile", "stock"]
            for name in order:
                arm = arms[name]
                state: dict = {}
                last_snap = [counters()]

                def factory(model_id, arm=arm, state=state):
                    engine = default_engine_factory(arm.config, forks=False)(model_id)
                    state.clear()
                    state.update(getattr(engine, "attention_tile_status", None) or {})
                    return engine

                def record(run, arm=arm, rnd=rnd, state=state, last_snap=last_snap):
                    now = counters()
                    delta = {k: v - last_snap[0].get(k, 0) for k, v in now.items()
                             if v != last_snap[0].get(k, 0)}
                    last_snap[0] = now
                    row = {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in dataclasses.asdict(run).items()}
                    row.update(arm=arm.label, round=rnd, switch=arm.attention_tile,
                               tile_state={"status": state.get("state"), "reason": state.get("reason"),
                                           "splits": state.get("splits"),
                                           "probe_seconds": state.get("probe_seconds")},
                               tile_calls=delta, ts=time.time())
                    out.write(json.dumps(row, default=str) + "\n")
                    out.flush()
                    print(f"{arm.label} r{rnd} {run.task} grade={run.grade} {run.seconds:.1f}s "
                          f"turns={run.turns} out={run.output_tokens} calls={delta} | "
                          f"{run.grade_detail[:100]}", flush=True)

                t0 = time.time()
                print(f"== round {rnd} arm {name}", flush=True)
                outcome = run_suite(arm, tasks, runs=1, done=set(), record=record,
                                    scratch=scratch_root / f"{name}-{rnd}", factory=factory)
                print(f"== round {rnd} arm {name} done in {time.time() - t0:.0f}s: "
                      f"released={outcome.released} error={outcome.error} tile_state={state}",
                      flush=True)
                if not outcome.released:
                    raise SystemExit("weights not released; stopping")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Write the kernel timings**

Create `~/code/sous-spikes/tile-132/gate/ktime.py`:

```python
"""Kernel timings for the attention-tile merge gate (out of tree, never committed), adapted from
tile-132/model/bench_model.py's `time` command.

One process. Each cell's variants are sampled round-robin, in an order rotated every round and
reversed on odd rounds, so drift spreads evenly across them. A sample is one dependent chain over
the 27B's 16 full-attention layers: layer l's output is layer l+1's query, and each layer reads
its own [1, 4, cap, 256] bf16 buffer through the strided [:, :, :n, :] slice KVCache hands the
model (past a 6 GB budget the layers cycle through as many distinct buffers as fit).

Variants per cell: the baseline (stock M=1 SDPA at T = 1, verifyattn.grouped_attention at T = 3
and 8), the in-tree tileattn entry at S = 20 twice (an A/A pair whose gap is the noise floor), and
the spike's tileattn.py loaded under another module name. A cell passes when the in-tree / spike
median ratio is within 0.97-1.03 and the in-tree speedup over the baseline is within 5% of the
spec's microbench table. `tree == spike` compares the two kernels' chains bitwise (informational).

usage: ktime.py [CTX,...]   (default 2048,8192,57000,131072,262136; a cell runs n = CTX + T keys)
"""

import importlib.util
import math
import os
import statistics
import sys
import time

import mlx.core as mx

from sous.engine import tileattn, verifyattn

SPIKE = os.path.expanduser("~/code/personal/sous-spikes/tile-132/model/tileattn.py")
HQ, HKV, D, LAYERS = 24, 4, 256, 16
SCALE = D**-0.5
SPLITS = 20
TS = (1, 3, 8)
CONTEXTS = [2048, 8192, 57000, 131072, 262136]
# The spec's microbench, ms per 16-layer chain: {T: {context: (baseline, tile)}}.
SPEC = {
    1: {2048: (1.10, 0.99), 8192: (2.82, 2.38), 57000: (15.63, 13.48),
        131072: (35.25, 30.61), 262136: (70.45, 59.41)},
    3: {2048: (2.11, 1.11), 8192: (5.77, 2.62), 57000: (32.53, 13.95),
        131072: (74.39, 30.85), 262136: (160.99, 62.87)},
    8: {2048: (4.89, 1.66), 8192: (14.42, 3.62), 57000: (82.07, 20.63),
        131072: (192.12, 46.97), 262136: (417.16, 98.06)},
}
ARCH = str(mx.device_info()["architecture"])


def _load_spike():
    spec = importlib.util.spec_from_file_location("spike_tileattn", SPIKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spike = _load_spike()


def make_layers(n_max, budget_gb=6.0, seed=0):
    cap = -(-n_max // 256) * 256
    if cap == n_max:
        cap += 256
    per = 2 * HKV * cap * D * 2 / 1e9
    distinct = max(1, min(LAYERS, int(budget_gb // per)))
    bufs = []
    for i in range(distinct):
        k1, k2 = mx.random.split(mx.random.key(seed + 1000 + 17 * i + n_max))
        kb = mx.random.normal((1, HKV, cap, D), key=k1).astype(mx.bfloat16)
        vb = mx.random.normal((1, HKV, cap, D), key=k2).astype(mx.bfloat16)
        mx.eval(kb, vb)
        bufs.append((kb, vb))
    return [bufs[i % distinct] for i in range(LAYERS)]


def views(layers, n):
    out = [(kb[:, :, :n, :], vb[:, :, :n, :]) for kb, vb in layers]
    mx.eval([a for kv in out for a in kv])
    return out


def queries(t, seed=77):
    q = mx.random.normal((1, t, HQ, D), key=mx.random.key(seed)).astype(mx.bfloat16)
    q = q.transpose(0, 2, 1, 3)
    mx.eval(q)
    return q


def stock_m1(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=None)


def grouped(q, k, v):
    return verifyattn.grouped_attention(q, k, v, SCALE, k.shape[2] - q.shape[2], ARCH)


def tree(q, k, v):
    if q.shape[2] == 1:
        return tileattn.decode(q, k, v, SPLITS)
    return tileattn.verify(q, k, v, SPLITS)


def spiked(q, k, v):
    if q.shape[2] == 1:
        return spike.decode(q, k, v, spike.SCALE)
    return spike.verify(q, k, v, spike.SCALE)


def chain(fn, q0, layer_views):
    q = q0
    for k, v in layer_views:
        q = fn(q, k, v)
    return q


def run_block(variants, q0, lv, min_ms=300.0, min_reps=7, max_reps=41):
    names = list(variants)
    est = {}
    for name in names:
        for _ in range(2):
            mx.eval(chain(variants[name], q0, lv))
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(chain(variants[name], q0, lv))
        mx.synchronize()
        est[name] = (time.perf_counter() - t0) * 1e3
    reps = int(min(max_reps, max(min_reps, math.ceil(min_ms / max(0.05, max(est.values()))))))
    samples = {name: [] for name in names}
    for r in range(reps):
        order = names[r % len(names):] + names[: r % len(names)]
        if r % 2:
            order = order[::-1]
        for name in order:
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(chain(variants[name], q0, lv))
            mx.synchronize()
            samples[name].append((time.perf_counter() - t0) * 1e3)
    return samples


def main():
    ctxs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else CONTEXTS
    print(f"device {ARCH} mlx {mx.__version__} N0={tileattn.N0} S={SPLITS} "
          "spike=tile-132/model/tileattn.py", flush=True)
    print("| ctx | T | baseline | baseline ms | tree ms | spike ms | A/A gap | tree / spike | "
          "speedup | spec speedup | tree == spike | cell |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    failures = 0
    for ctx in ctxs:
        layers = make_layers(ctx + max(TS))
        for t in TS:
            lv = views(layers, ctx + t)
            q0 = queries(t)
            base = "stock" if t == 1 else "grouped"
            variants = {base: stock_m1 if t == 1 else grouped, "tree": tree, "spike": spiked,
                        "tree#2": tree}
            samples = run_block(variants, q0, lv)
            med = {name: statistics.median(s) for name, s in samples.items()}
            gap = statistics.median(
                a / b for a, b in zip(samples["tree"], samples["tree#2"], strict=True)) - 1.0
            ratio = med["tree"] / med["spike"]
            speedup = med[base] / med["tree"]
            same = mx.array_equal(chain(tree, q0, lv), chain(spiked, q0, lv)).item()
            spec = SPEC.get(t, {}).get(ctx)
            spec_speedup = spec[0] / spec[1] if spec else None
            ok = 0.97 <= ratio <= 1.03 and (
                spec_speedup is None or abs(speedup / spec_speedup - 1.0) <= 0.05)
            failures += not ok
            spec_cell = "-" if spec_speedup is None else f"{spec_speedup:.2f}x"
            print(f"| {ctx} | {t} | {base} | {med[base]:.3f} | {med['tree']:.3f} | "
                  f"{med['spike']:.3f} | {gap * 100:+.1f}% | {ratio:.3f} | {speedup:.2f}x | "
                  f"{spec_cell} | {same} | {'ok' if ok else 'FAIL'} |", flush=True)
            del lv, q0
        del layers
        mx.clear_cache()
    print(f"ktime {'PASS' if failures == 0 else f'FAIL ({failures} cells)'}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Sync the harness to the M5 and copy the spike's scripts beside it**

```bash
rsync -a -e ~/code/sous-spikes/tile-132/gate/m5.sh ~/code/sous-spikes/tile-132/gate/ m5pro-lan:code/personal/sous-spikes/tile-132/gate/
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'M=~/code/personal/sous-spikes/tile-132/model; G=~/code/personal/sous-spikes/tile-132/gate; cp $M/gate132.py $M/analyze132.py $M/analyze133.py $M/analyze_suite.py $M/bench_model.py $M/cmp_decode.py $G/ && ~/code/personal/sous-remote-tile/.venv/bin/python -m py_compile $G/*.py && zsh -n $G/run_gate.sh && cd $G && ~/code/personal/sous-remote-tile/.venv/bin/python -c "import wrap_branch, suite_ab" && ls $G'
```

Expected:
- Nothing fails to compile or import.
- The listing shows the six copied scripts beside `m5.sh`, `wrap_branch.py`, `run_gate.sh`, `suite_ab.py`, `ktime.py`, `suite/` and `logs/`.

Which analysis commands the gate uses:
- `analyze132.py`: its `equiv` and `turns` commands. Its `summary` reads the spike's own tile-state keys and is not used.
- `analyze133.py`, `cmp_decode.py` and `analyze_suite.py`: read the rows exactly as written.

- [ ] **Step 12: Run replay chain 1: speed ABBA and the greedy report**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/tile-132/gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/chain-speed.txt zsh $G/run_gate.sh $G/logs/SPEED_DONE main-s1:main:0.7:1024:-:acc tile-s1:branch:0.7:1024:-:acc tile-s2:branch:0.7:1024:-:acc main-s2:main:0.7:1024:-:acc main-g:main:0.0:1024:-:acc tile-g:branch:0.0:1024:-:acc'
```

Six runs of about 21 minutes each. To follow progress, and to learn when the chain is done:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/tile-132/gate/logs; cat $G/SPEED_DONE.part $G/SPEED_DONE 2>/dev/null; tail -3 $G/tile-s1.log 2>/dev/null'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/tile-g.log 540 6'
```

The chain is done when `SPEED_DONE` exists and lists six `exit=0` lines. A `port 8384 still busy` or `process left` line stops the chain: report it before going on.

The first branch run's log shows the load line. Check it early:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'grep -h model_load ~/code/personal/sous-spikes/tile-132/gate/runs/tile-s1/daemon.out'
```

Expected: `attention_tile=active attention_tile_splits=20 attention_tile_probe_s=<seconds>`. If it reads anything else, stop.

- [ ] **Step 13: Run replay chain 2: drafter off, parity, off equals main**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/tile-132/gate && python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/detach.py $G/logs/chain-rest.txt zsh $G/run_gate.sh $G/logs/REST_DONE main-nd1:main:0.7:1024:nodraft:sub16 tile-nd1:branch:0.7:1024:nodraft:sub16 tile-nd2:branch:0.7:1024:nodraft:sub16 main-nd2:main:0.7:1024:nodraft:sub16 t-d-g:branch:0.0:256:-:full t-nd-g:branch:0.0:256:nodraft:full main-g256:main:0.0:256:-:acc off-g256:branch:0.0:256:off:acc'
```

About four hours. Follow it as in Step 12, reading `REST_DONE.part`/`REST_DONE` and waiting on `off-g256.log`. It is done when `REST_DONE` lists eight `exit=0` lines.

- [ ] **Step 14: Check that every replay run counts**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/tile-132/gate; for l in main-s1 tile-s1 tile-s2 main-s2 main-g tile-g main-nd1 tile-nd1 tile-nd2 main-nd2 t-d-g t-nd-g main-g256 off-g256; do f=$G/logs/$l.log; printf "%-10s idle=%s gone=%s\n" $l "$(grep -c "daemon stayed idle during the job" $f)" "$(grep -c "process group gone; port 8384 free" $f)"; done | tee $G/logs/validity.txt'
```

Expected: `idle=1 gone=1` on all 14 lines.

A run counts only if the lock reported the daemon idle throughout and the harness logged the child's process group gone and its port free. Re-run any other run under a new label: append `b`, run it with `run_gate.sh` and a done-file of its own, and use that label wherever the old one appears below.

- [ ] **Step 15: Run the kernel timings**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && G=~/code/personal/sous-spikes/tile-132/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/ktime.txt env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py .venv/bin/python $G/ktime.py'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/ktime.txt 540 22'
```

Repeat the wait until `DONE`. Expected:
- 15 table rows, each ending `ok`.
- `ktime PASS`.
- `daemon stayed idle during the job`.

- [ ] **Step 16: Run the suite: 8 tasks × 6 runs per arm, in two GPU-lock jobs**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'G=~/code/personal/sous-spikes/tile-132/gate && mkdir -p $G/suite/home/.sous && cp $G/suite/config.toml $G/suite/home/.sous/config.toml'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && G=~/code/personal/sous-spikes/tile-132/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/suite-a.log env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py perl -e "alarm shift @ARGV; exec @ARGV" 3600 env HOME=$G/suite/home HF_HUB_CACHE=$HOME/.cache/huggingface/hub SUITE_AB_CONFIG=$G/suite/home/.sous/config.toml .venv/bin/python $G/suite_ab.py $G/suite/suite.jsonl 0 2'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/suite-a.log 540 12'
```

Repeat the wait until `DONE`, about 22 minutes. Then run the second job the same way, with log `suite-b.log` and rounds `3 5`:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && G=~/code/personal/sous-spikes/tile-132/gate && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py $G/logs/suite-b.log env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py perl -e "alarm shift @ARGV; exec @ARGV" 3600 env HOME=$G/suite/home HF_HUB_CACHE=$HOME/.cache/huggingface/hub SUITE_AB_CONFIG=$G/suite/home/.sous/config.toml .venv/bin/python $G/suite_ab.py $G/suite/suite.jsonl 3 5'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/suite-b.log 540 12'
```

Expected in both logs:
- Every `== round N arm X done` line says `released=True error=None`.
- The tile arm's `tile_state` has `'state': 'active'` and `'splits': 20`; the stock arm's has `'state': 'off'`.
- `daemon stayed idle during the job` and `exit=0`.

- [ ] **Step 17: Run the M2 (this Mac)**

Make a main checkout and a branch checkout, each with its own `.venv`:

```bash
git fetch -q origin
git worktree add --detach ~/code/sous-spikes/tile-132/m2/main origin/main
git worktree add --detach ~/code/sous-spikes/tile-132/m2/branch origin/feat/attention-tile
uv sync -q --directory ~/code/sous-spikes/tile-132/m2/main
uv sync -q --directory ~/code/sous-spikes/tile-132/m2/branch
```

Run #143's M2 task set greedy on each tree. These are the 9B and its DFlash drafter at max_tokens 512, about 7 minutes each; run each in the background:

```bash
GATE_BODIES=$HOME/code/sous-spikes/tile-132/m2/bodies-small.json GATE_MODEL_TOML='id = "mlx-community/Qwen3.5-9B-MLX-4bit"\nspeculative_draft_id = "z-lab/Qwen3.5-9B-DFlash"\n' GPU_LABEL=tile-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py python3 ~/code/sous-spikes/tile-132/m2/gate143_m2.py main-d-g ~/code/sous-spikes/tile-132/m2/main 0.0 512 > ~/code/sous-spikes/tile-132/m2/main-d-g.log 2>&1
GATE_BODIES=$HOME/code/sous-spikes/tile-132/m2/bodies-small.json GATE_MODEL_TOML='id = "mlx-community/Qwen3.5-9B-MLX-4bit"\nspeculative_draft_id = "z-lab/Qwen3.5-9B-DFlash"\n' GPU_LABEL=tile-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py python3 ~/code/sous-spikes/tile-132/m2/gate143_m2.py br-d-g ~/code/sous-spikes/tile-132/m2/branch 0.0 512 > ~/code/sous-spikes/tile-132/m2/br-d-g.log 2>&1
```

Compare the two runs and read the branch's load line:

```bash
uv run python - <<'EOF' | tee ~/code/sous-spikes/tile-132/m2/m2.txt
import json
import os

runs = os.path.expanduser("~/code/sous-spikes/tile-132/m2/runs")
main, branch = (json.load(open(f"{runs}/{label}/result.json")) for label in ("main-d-g", "br-d-g"))


def reply(r):
    blocks = [{k: v for k, v in b.items() if k != "id"} for b in r.get("content") or []]
    return json.dumps(blocks, sort_keys=True), r.get("stop")


errors = [r for r in (*main["results"], *branch["results"]) if not r.get("id")]
same = [reply(a) == reply(b) for a, b in zip(main["results"], branch["results"], strict=True)]
print(f"HTTP errors: {len(errors)}")
print(f"greedy replies identical on {sum(same)}/{len(same)} turns")
print("branch load line:", branch["load"][-1]["_raw"].split("INFO ")[-1])
print("branch attention_tile:", branch["engine_status"].get("attention_tile"))
EOF
```

Expected:
- `HTTP errors: 0`.
- `identical on 12/12 turns`.
- The branch load line ends `attention_tile=unavailable`.
- The status reason is the NAX platform rule's, `macOS 15.5 < 26.2` on this Mac.

Run `api_smoke.py` in the branch tree:

```bash
GPU_LABEL=tile-m2 python3 ~/code/sous-spikes/splash-port-2026-09-24/gpu_run.py uv run --directory ~/code/sous-spikes/tile-132/m2/branch python scripts/api_smoke.py 2>&1 | tail -4 | tee ~/code/sous-spikes/tile-132/m2/smoke.txt
```

Expected: `PASS: a tool_use block came back`.

- [ ] **Step 18: Run `api_smoke.py` on the M5**

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && L=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $L/detach.py ~/code/personal/sous-spikes/tile-132/gate/logs/smoke.txt env GPU_LABEL=tile-gate python3 $L/gpu_run_m5.py uv run python scripts/api_smoke.py'
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/tile-132/gate/logs/smoke.txt 540 6'
```

Expected:
- `PASS: a tool_use block came back`.
- `daemon stayed idle during the job`.
- `exit=0`.

- [ ] **Step 19: Judge the gate**

First, on the M5, run the equivalence checks. They need the 27B's tokenizer for any reply that differs, so they run there. Save the load lines at the same time:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'cd ~/code/personal/sous-remote-tile && G=~/code/personal/sous-spikes/tile-132/gate; .venv/bin/python $G/analyze132.py $G/runs equiv t-d-g t-nd-g > $G/logs/parity.txt; .venv/bin/python $G/analyze132.py $G/runs equiv off-g256 main-g256 > $G/logs/off_main.txt; .venv/bin/python $G/analyze132.py $G/runs equiv tile-g main-g > $G/logs/greedy_equiv.txt; grep -h model_load $G/runs/tile-s1/daemon.out $G/runs/tile-nd1/daemon.out $G/runs/off-g256/daemon.out $G/runs/main-s1/daemon.out > $G/logs/load_lines.txt; head -1 $G/logs/parity.txt $G/logs/off_main.txt $G/logs/greedy_equiv.txt; cat $G/logs/load_lines.txt'
```

Then pull the results back to this Mac. This excludes the run homes and the suite's scratch copies:

```bash
rsync -a -e ~/code/sous-spikes/tile-132/gate/m5.sh --exclude 'home/' --exclude '*.scratch/' m5pro-lan:code/personal/sous-spikes/tile-132/gate/ ~/code/sous-spikes/tile-132/gate/
```

Then run the CPU-only analyses here:

```bash
G=~/code/sous-spikes/tile-132/gate
uv run python $G/analyze133.py $G/runs runs main-s1 tile-s1 tile-s2 main-s2 main-g tile-g main-nd1 tile-nd1 tile-nd2 main-nd2 | tee $G/logs/decode_runs.md
uv run python $G/analyze133.py $G/runs pair tile-s1 main-s1 tile-s2 main-s2 | tee $G/logs/decode_pair.md
uv run python $G/analyze133.py $G/runs pair tile-nd1 main-nd1 tile-nd2 main-nd2 | tee $G/logs/nodraft_pair.md
uv run python $G/cmp_decode.py $G/runs tile-g main-g | tee $G/logs/greedy_identical.txt
uv run python $G/analyze_suite.py $G/suite/suite.jsonl | tee $G/logs/suite.md
cat $G/logs/ktime.txt
```

Check the path counters, from each branch run's last snapshot of the process totals:

```bash
uv run python - <<'EOF' | tee ~/code/sous-spikes/tile-132/gate/logs/path.txt
import json
import os

runs = os.path.expanduser("~/code/sous-spikes/tile-132/gate/runs")
drafted = {"tile-s1", "tile-s2", "tile-g", "t-d-g"}
branch = ["tile-s1", "tile-s2", "tile-g", "tile-nd1", "tile-nd2", "t-d-g", "t-nd-g"]
for label in [*branch, "off-g256"]:
    snap = json.load(open(f"{runs}/{label}/result.json"))["results"][-1].get("tile") or {}
    totals = snap.get("totals") or {}
    state = (snap.get("state") or {}).get("state")
    tile = {k: v for k, v in totals.items() if v and not k.startswith("verifyattn_")}
    if label == "off-g256":
        ok = state == "off" and not tile
    else:
        ok = (
            state == "active"
            and totals.get("decode_tile", 0) > 0
            and (label not in drafted or totals.get("verify_calls", 0) > 0)
            and not any(totals.get(k, 0) for k in ("decode_out", "verify_out", "verify_declined"))
        )
    print(f"{label}: {'PASS' if ok else 'FAIL'} state={state} {tile}")
EOF
```

List every max_tokens stop against main's run of the same turn:

```bash
uv run python - <<'EOF' | tee ~/code/sous-spikes/tile-132/gate/logs/max_tokens.txt
import json
import os

runs = os.path.expanduser("~/code/sous-spikes/tile-132/gate/runs")
for tile, main in (("tile-s1", "main-s1"), ("tile-s2", "main-s2"), ("tile-g", "main-g")):
    a = json.load(open(f"{runs}/{tile}/result.json"))["results"]
    b = json.load(open(f"{runs}/{main}/result.json"))["results"]
    for x, y in zip(a, b, strict=True):
        if "max_tokens" in (x["stop"], y["stop"]):
            print(
                f"{tile} vs {main} {x['agent'][6:14]}#{x['turn']} n={x['n']}: stop "
                f"{x['stop']}/{y['stop']}, out {x['usage']['output_tokens']}/"
                f"{y['usage']['output_tokens']}"
            )
print("done")
EOF
```

The gate passes only if every criterion below holds. Each is quoted from the spec, followed by the output that decides it.

- **Load.** "The M5 Pro's load line shows `attention_tile=active attention_tile_splits=20`, and the probe's cost is recorded."
  - Decided by `logs/load_lines.txt`.
  - The `tile-s1` and `tile-nd1` lines show both tokens plus `attention_tile_probe_s=`; record that value.
  - The `off-g256` line ends `attention_tile=off`.
- **Path.** "In every branch run, tile decode and verify counts are > 0, with 0 declined and 0 out of scope. The off run makes 0 tile calls."
  - Verify counts apply to the drafter-on runs; a drafter-off run makes no verify call.
  - Decided by `logs/path.txt`, which must print `PASS` on all eight lines.
- **Speed.** "Sampled, in ABBA order (main, branch, branch, main) on the 64 turns:"
  - "pooled `out_tokens / decode_s` is ≥ 1.15x branch/main, and the turn-bootstrap 95% CI excludes 1.10"
  - "the difference in tokens per round has a CI that includes 0"
  - "prefill tok/s is within ±3% of main in the same ABBA, and peak `memory_gb` within ±0.2 GB."
  - "Greedy is reported on identical-output turns only. Any max_tokens stop or repetition loop is listed and compared with main's run of the same turn."

  Decided by `logs/decode_pair.md`:
  - The `all` row's ratio is ≥ 1.150, and its CI's lower end is > 1.100.
  - In `tokens/round tile − main: d (95% CI lo..hi)`, lo ≤ 0 ≤ hi.
  - The `all` row's tile prefill tok/s is within ±3% of its main prefill tok/s.
  - In `peak memory_gb over these turns: tile X, main Y`, |X − Y| ≤ 0.2.

  The greedy figure is the last line of `logs/greedy_identical.txt`, and every max_tokens stop is in `logs/max_tokens.txt`. Both are reported, not judged.
- **Drafter off.** "With `speculative_draft_id = ""`, sampled, ABBA on a 16-turn subset: branch/main ≥ 1.00 pooled."
  - Decided by `logs/nodraft_pair.md`: the `all` row's ratio is ≥ 1.000.
- **Parity.** "The branch with the drafter against the branch without it, greedy, max_tokens 256."
  - "At least 16 turns, including the a6443762 continuation at ≥ 128K."
  - "Every reply and stop reason must be identical."
  - "The prompt-cache history (hit or miss, slot taken, reused and prefilled tokens) must be equal on every compared pair."
  - "One divergence fails the gate."

  Decided by `logs/parity.txt`:
  - It reads `equivalence t-d-g vs t-nd-g: 23/23 identical; cache histories equal on 23/23` with no further lines.
  - The 23 turns include a6443762@176, @239, @300 and @365, which run from 131K to 229K.
- **Off equals main.** "`attention_tile = false`, greedy, with the same settings: reply-identical to main on the 64 turns, with equal cache histories."
  - Decided by `logs/off_main.txt`: `64/64 identical; cache histories equal on 64/64`.
- **Suite.** "8 tasks × 6 runs per arm, in alternating rounds."
  - "The branch's mean grade is at least stock's − 0.02."
  - "No task that is 6/6 on stock drops below 5/6 on the branch."
  - "No new failure mode appears; the failing diffs are compared, as the spike did."

  Decided by `logs/suite.md`:
  - Each arm has 48 runs.
  - The tile arm's mean grade is ≥ stock's − 0.02.
  - For every task whose six stock grades are all 1, at least five tile grades are 1.
  - The "Tile counters per arm" lines show tile calls only on the tile arm.
  - For every tile run under "Imperfect runs", open its transcript and the matching stock failure: the same failure mode is allowed, and a new one fails.
- **Kernel timings.** "In one process, interleaved, the in-tree entries run against stock M=1 / #128 grouped and against the spike's `tileattn.py` kernel, at T = 1, 3 and 8 from 2K to 262K, with an A/A pair to show the noise floor."
  - "In-tree / spike median ratio within 0.97–1.03 in every cell."
  - "Speedup over stock or grouped within 5% of the spike's table."

  Decided by `logs/ktime.txt`, which reads `ktime PASS`.
- **M2 Air.** "Running the 9B with its DFlash drafter:"
  - "the load line reads `attention_tile=unavailable` with the NAX reason;"
  - "greedy replies on the #143 M2 task set are identical to main's."

  Decided by `~/code/sous-spikes/tile-132/m2/m2.txt`, as Step 17 expects.
- **`scripts/api_smoke.py`** "passes on both Macs."
  - Decided by `logs/smoke.txt` and `~/code/sous-spikes/tile-132/m2/smoke.txt`: both read `PASS: a tool_use block came back`.

If any criterion fails, stop and report the numbers and the files above. Open no PR.

- [ ] **Step 20: Record the gate and open the PR**

**Decode figures.** Only after a passing gate. From `logs/decode_pair.md`, take R_all (the `all` row's ratio) and R_mid (the `32-64K` row's ratio). Apply each of these edits only when its figure changes:
- `docs/configuration.md`, the `[model].attention_tile` paragraph: if R_all rounded to two decimals is not 1.22, replace `1.22x` in `decode is 1.22x faster on 64 real subagent turns` with R_all to two decimals.
- `README.md:56`: in `about 19.5 tok/s at 57K`, replace 19.5 with 19.5 × R_mid rounded to the nearest 0.5.
- `README.md:148`: in `1.7–1.9x at 51–57K`, divide both bounds by R_mid and round each to one decimal.
- Leave `about 31 tok/s with 2K tokens of context` alone, because no gate run measures short context.

If any of these edits changed a figure, commit them and push:

```bash
git add README.md docs/configuration.md
git commit -m "docs: update the decode figures with the attention tile's measured gain" -m "The merge gate measured decode on the M5 Pro's 64 real subagent turns with
the tile against main, sampled, in ABBA order. The 57K figure and the gap
to Splash at 51-57K scale with its 32-64K ratio, and the tile's own gain is
the pooled ratio; the 2K figure stays, since no gate run measured short
context.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin feat/attention-tile
```

**The PR body.** Write its fixed head:

```bash
cat > ~/code/sous-spikes/tile-132/gate/pr-head.md <<'EOF'
## What and why

Decode's one-row attention and the exact verifier's attention now run on the M5's tensor units,
through one GQA-packed MPP tile whose key partition steps with context
(`src/sous/engine/tileattn.py`, `src/sous/engine/kernels/attention_tile.metal`). Verify attention
is most of decode's context-dependent cost at subagent context, and the tile cuts it 2–2.6x at
block 3. A row's output depends only on its chunk and the chunk is fixed per 640-key step, so a
verify row is bit-identical to the one-row decode at its own key count: greedy output with the
DFlash2 drafter stays identical to output without it.

- On by default through `[model].attention_tile`; `false` runs today's paths.
- Active only on the measured split target, the M5 Pro's 20 GPU cores, with macOS 26.2+, a dense
  qwen3_5 model at 24/4/256 bf16 and no drafter or a DFlash one; `unavailable` with its reason
  everywhere else, where nothing changes.
- The model-load line gains `attention_tile=` (with `attention_tile_splits=` and
  `attention_tile_probe_s=` when active), and `/sous/status` an `attention_tile` block.
- `sous tune` keys suite rows by the setting, so a resume never mixes tile and stock rows.

**Fork store.** Landing this cold-starts every on-disk fork store once, including stores with the
tile off: the fork key gains `tile` (and `os` whenever the tile or int8 prefill is active), and
`tileattn.py` joins the engine epoch. While the tile or int8 prefill is active, a macOS update
cold-starts the store again, because the OS's Metal toolchain compiles those kernels at load.

Spec: `docs/superpowers/specs/2026-09-30-attention-tile-design.md`. Plan:
`docs/superpowers/plans/2026-09-30-attention-tile.md`.

Issue: #132. Still open and out of scope: block 8 and cheaper verify projections, #126 and #133.
EOF
```

Assemble the body from the head and the gate's outputs. The suite's list of imperfect runs is left out: it carries transcript paths.

```bash
G=~/code/sous-spikes/tile-132/gate
M2=~/code/sous-spikes/tile-132/m2
{
  cat $G/pr-head.md
  printf '\n## Merge gate (M5 Pro, 20 GPU cores)\n\n### Load\n\n~~~\n'; cat $G/logs/load_lines.txt; printf '~~~\n'
  printf '\n### Path (process totals per branch run)\n\n~~~\n'; cat $G/logs/path.txt; printf '~~~\n'
  printf '\n### Decode, sampled ABBA on the 64 turns\n\n'; cat $G/logs/decode_runs.md $G/logs/decode_pair.md
  printf '\n### Greedy, identical-output turns only\n\n~~~\n'; tail -n 2 $G/logs/greedy_identical.txt; printf '~~~\n'
  printf '\n### max_tokens stops, branch against main on the same turn\n\n~~~\n'; cat $G/logs/max_tokens.txt; printf '~~~\n'
  printf '\n### Drafter off, sampled ABBA on 16 turns\n\n'; cat $G/logs/nodraft_pair.md
  printf '\n### Parity: drafter on against off, greedy, 23 turns to 228,630 tokens\n\n~~~\n'; cat $G/logs/parity.txt; printf '~~~\n'
  printf '\n### attention_tile = false against main, greedy, 64 turns\n\n~~~\n'; cat $G/logs/off_main.txt; printf '~~~\n'
  printf '\n### Tool-loop suite, 8 tasks x 6 runs per arm\n\n'; sed '/^Imperfect runs:/,$d' $G/logs/suite.md
  printf '\n### Kernel timings, ms per 16-layer chain\n\n'; grep -E '^\||^ktime' $G/logs/ktime.txt
  printf '\n### M2 Air (9B + DFlash, the #143 task set)\n\n~~~\n'; cat $M2/m2.txt; printf '~~~\n'
  printf '\n`scripts/api_smoke.py`: M5 `%s`, M2 `%s`.\n' "$(grep -h '^PASS\|^FAIL' $G/logs/smoke.txt)" "$(grep -h '^PASS\|^FAIL' $M2/smoke.txt)"
  printf '\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n'
} > $G/pr-body.md
if grep -n -e /Users/ -e "$(id -un)" $G/pr-body.md; then echo "strip these lines before opening the PR"; fi
```

Expected: the privacy check prints nothing. The body must carry no person's name or home path.

Open the PR:

```bash
gh pr create --base main --head feat/attention-tile --title "feat(engine): run decode and verify attention on the M5's tensor units" --body-file ~/code/sous-spikes/tile-132/gate/pr-body.md
```

**Clean-up.** Delete any forks the runs left behind; gate132.py already removes its own:

```bash
~/code/sous-spikes/tile-132/gate/m5.sh m5pro-lan 'find ~/code/personal/sous-spikes/tile-132/gate/runs -path "*/home/.sous/forks" -type d -prune -print -exec rm -rf {} +'
find ~/code/sous-spikes/tile-132/m2/runs -path '*/home/.sous/forks' -type d -prune -print -exec rm -rf {} +
```

---

## Self-Review Notes

- **Spec coverage:**
  - What the measurements established › the schedule: Task 3 (`chunk_for`, `step_of`, `boundaries`, `verify_plan`, the 408-boundary and 20-split facts).
  - Decisions:
    - on by default with an opt-out: Task 8;
    - existing hook points: Task 5 (decode global, verifyattn branch, only T = 1 and 2 reproduced in `_stock_verify`);
    - one kernel with a stepped partition, runs cut at 8: Tasks 3–4;
    - split target S from IOKit, only measured values on: Tasks 2 and 7;
    - drafters, none or DFlash: Task 7;
    - lives in sous with runtime gates: Tasks 1 and 7.
  - Design › Files: Tasks 1 (`nax.py`), 4 (`attention_tile.metal`, safe math, `remove_addrspace_t`, `nax.h` lane layout), 5 (verifyattn branch).
  - Design › Schedule and its layout invariant: Tasks 3–4, with the real producers' layouts exercised by Task 6's real-module step.
  - Design › Decode hook and Verify hook: Task 5; first activation installs: Task 7.
  - Design › Guards 1–8, the flag and the failure policy: Tasks 6–7.
  - Design › Wiring and visibility: Task 8 (vlm.py, lm.py, config.py, base.py), Task 10 (`sous tune`), Task 5 (the `calls` dict).
  - Design › Fork store: Task 9, plus `engine_epoch()`'s header comment.
  - Testing › pure logic: Tasks 2, 3, 7, 8, 9, 10; stand-in kernel (decode scope, verify scope, output, sensitivity, probe decisions, lifecycle, fixtures): Tasks 4–7; contract test: Task 11; `nax` tests: Tasks 4, 6, 7, 11; model test: Task 11.
  - Merge gate: Task 13, every pass criterion quoted with the output that decides it.
  - Documentation: Task 12 (CLAUDE.md gotcha and amendments, configuration, observability, how-it-works), Task 10 (tuning), Task 13 (README figures only if justified).
  - Out of scope: untouched.
- **Deliberate refinements of the spec's wording:**
  - The probe's far parity mark is the first step boundary at or past 57,000 keys (57,601 at S = 20), not "the boundary nearest 57K" (56,961): a boundary at or past the target always puts every probed row at its full S splits, and the rule is the same for any S.
  - "Exactly 20 splits" holds from n = 11,553, not 12,161 as the spec's prose says: 11,553 is the first n whose chunk (608) needs 20 splits, and Task 3 pins it; the spec's figure is a later n where it also holds.
  - The lifecycle test's "then an unload" ends with `tileattn.clear()`, the call `VLMEngine.unload()` makes; Task 8 separately pins that `unload()` clears the flag.
- **Placeholders:** none. Every code step carries complete code; the only angle-bracket values are gate outputs (`<seconds>`) that the gate reads, not writes.
- **Type and name consistency** (checked across tasks):
  - `tile(queries, keys, values, n, chunk)`, `decode(queries, keys, values, splits)`, `verify(queries, keys, values, splits)`, `in_scope(queries, keys, values, scale)`, `serve_verify(queries, keys, values, splits)`.
  - `probe(modules, splits, arch)`, `far_mark(splits)`, `parity_cases(splits, arch)`, `enable(model, *, enabled, drafter_kind, verify_status)`, `clear()`, `install_decode_hook()`, `_decode_wrapper(original)`, `_HOOK_MARK`, `_active_splits`, `_compile_ok`, `_compile_probe()`, `availability()`.
  - Status keys `state`, `reason`, `splits`, `probe_seconds`; LM status `{"state": "off", "reason": None}`.
  - `calls` keys `decode_tile`, `decode_stock`, `decode_out`, `verify_calls`, `verify_rows_tile`, `verify_rows_stock`, `verify_launches`, `verify_straddles`, `verify_declined`, `verify_out`, `verify_t1`…`verify_t8`, `verify_t9_up` — the same in Tasks 5, 6, 7, 11 and the gate's `path.txt` check.
  - Fork-key fields `tile` (`active:<S>`/`off`) and `os`; `forkstore.os_release()`.
  - `Arm.attention_tile`, `Arm.suite_key` and `SuiteRun.key` six-tuples; `decide.ArmSummary.key` annotation matches.
  - `tfx` helpers: `stand_in_tile`, `serving_qkv`, `TILE`, `nax`, `VOCAB`, `tiny_language_model`, `prefill_cache`, `verify_forward`, `hooked`, `verify_ready`, `guard_tile_state`, `tile_ready`.
- **Import staging:** `gpucores` and `_model_type` join tileattn's module-level imports in Task 7, their first user, so no task leaves an unused import for ruff's F401; `tfx.tile_ready` reaches `tileattn.gpucores` only from Task 7 on.
- **Line numbers** were checked against e9aa76b; each task that edits a file an earlier task already changed names the anchor text to replace, so the numbers are a guide and the replace blocks are exact.
- **Test counts** in the Expected lines were derived from the current collection (`test_int8prefill.py` 43, `test_tune_hardware.py` 5, `test_tune_report.py` 17, `test_verifyattn.py` 82 with 12 matching `-k scope`) plus the tests each task adds.
- **Dry run (2026-09-30):** Tasks 1–12 were executed as written in a scratch worktree on e9aa76b; every Expected count matched, the full CI set passed on the M2 (1757 passed, 38 skipped — all `nax`), and the tile's `nax` tests passed on the M5 Pro with none skipped (real `enable()`: `active`, 20 splits, 0.16 s probe). The fixes it found are folded in: every replace block's old text ends at a full source line (Task 7 Step 2, Task 12 Step 3), the failure reasons in Task 7 Step 5, Task 8 Step 6 and Task 10 Step 6 are the ones actually printed, and Task 11's model test is carried in `ruff format` form.
