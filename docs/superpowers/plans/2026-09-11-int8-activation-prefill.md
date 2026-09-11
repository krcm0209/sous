# INT8-Activation Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the prefill matmuls of affine-Q4/gs64 Qwen3.5-family models through an INT8-activation GEMM on the M5 tensor units, behind `[model] int8_prefill = false`, for a measured ≥1.4x prefill at 4,096 tokens on the M5 Pro with no change to decode, verify, or any machine where the key is off or the hardware lacks tensor units.

**Architecture:** One new module, `src/sous/engine/int8prefill.py`, holds two Metal kernels as source strings compiled at model load through `mx.fast.metal_kernel` (Stage A: per-row INT8 activation quantization; GEMM: int8 × packed-Q4 → int32 on the neural accelerators with the affine correction per 64-group), plus the routing layer: eligibility, per-model tagging of projections, and process-wide class wrappers on the Qwen3.5 MLP / GDN classes and `nn.QuantizedLinear` that act only on tagged modules with ≥128 rows. The engines call `enable(model, enabled=...)` right after `load()`; it degrades to stock kernels on any failure and reports a status dict the manager surfaces. No native build, no new dependency.

**Tech Stack:** Python 3.14, uv, mlx 0.32.2 (`mx.fast.metal_kernel`, Metal-4 `MetalPerformancePrimitives` tensor ops), mlx-vlm 0.6.17 / mlx-lm 0.31.3 (class targets), pytest, ty, ruff. No numpy (not a dependency; tests use mlx ops for references).

**Spec:** `docs/superpowers/specs/2026-09-11-int8-activation-prefill-design.md` — read it first; this plan argues from it. Its *What the spike established* section holds the measured numbers the acceptance bars come from; its *Non-goals* (default-on, attention routing = #76, per-group scaling, Q5/mxfp/MoE, decode) are not to be added here. Issue #77 covers baseline profiling and is unrelated to this plan.

## Global Constraints

- Python >= 3.14; everything runs via `uv run`. Never pip. `except A, B:` without parentheses is valid 3.14 syntax (PEP 758) — don't "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`. `uv run ty check` covers tests and scripts too. `mx.fast.metal_kernel(...)` is typed as returning `object`; the module wraps it once (`_kernel()`, Task 1) with a `cast` so call sites type-check without pragmas.
- **mlx / mlx_lm / mlx_vlm imports are function-local in `src/`** (absent on non-macOS; the lint job runs on ubuntu and must import `sous.engine.int8prefill`). Test files may bind `mx = pytest.importorskip("mlx.core")` at module level: tests run on `macos-15` only.
- **Any thread that touches mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits** (ml-explore/mlx#4327). Nothing in this plan adds a thread; the kernels run on whatever generation thread calls the model, and `enable()` runs on the thread that constructs the engine. The runtime gate (Task 7) includes a `ManagedEngine` session that exits.
- **Only rows ≥ `MIN_ROWS = 128` route** (`x.shape[-2]`). Decode (1 row) and speculative verify (≤ `SPECULATIVE_BLOCK_MAX + 1 = 6` rows) never touch the kernels. The constant is not a config knob.
- **Only tagged modules route.** Tags are set with `object.__setattr__` (so they live in the module's `__dict__`, not in mlx's parameter tree — `nn.Module.__setattr__` would store a tuple or bool *inside* the tree and it would show up in `parameters()`/`named_modules()`). The drafter, `lm_head`, embeddings, vision towers and every `self_attn.*` projection are never tagged.
- **The key off, or the hardware unavailable, means byte-identical behaviour to today**: no kernel compiled, no wrapper installed, nothing tagged. A model load never fails because of this feature — every failure in `enable()` degrades with one `warnings.warn` and returns `state == "unavailable"`.
- **No token content is logged.** Counts, shapes, and exception text only.
- **The kernel reads the checkpoint's packed weights in place**; no per-projection copies of anything are kept (Task 2 decides the scale/bias read layout by measurement; neither outcome keeps a resident copy).
- **The K-order invariant** (Task 1): slot `16c + 4t + j` of a 64-code group holds code `k = 16c + 8·(t>>1) + 2j + (t&1)`. The Python reorder and the GEMM's nibble decode implement the two halves of one contract — never change one without the other; the bit-exactness test in Task 2 is what catches a mismatch.
- Only affine Q4 gs64 `nn.QuantizedLinear` projections with fp16/bf16 scales, no bias, `N % 64 == 0`, `K % 64 == 0` are eligible. mxfp4/mxfp8/8-bit/gs128 fall through silently, per projection.
- Tests never touch the real `~/.sous`. `model`-marked tests download weights and run locally only (Task 7). `slow`-marked tests are run, not skipped.
- `docs/superpowers/**` are point-in-time records: never edit an existing spec or plan (this plan is a new file). The spec (55b8419) and this plan are committed on branch `feat/int8-prefill`; all work lands there. `main` is protected.
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: exactly `Co-Authored-By: Claude <noreply@anthropic.com>` (model-less, per the user's global CLAUDE.md; that instruction overrides any harness default). Never include a session URL. CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. Every commit step runs `uv run ruff format .` *before* `uv run ruff check .`.
- The kernel sources are a derivative of oMLX's Apache-2.0 `qwen35_oq_a8.metal`, `qwen35_oq_a8_nax.metal` and `oq_a8_decode.h` (jundot/omlx#3548). The module header and `THIRD_PARTY_NOTICES.md` (Task 2) carry the notice; sous stays MIT.
- Comments explain non-obvious *why*; never restate what code does.

---

## Decisions this plan locks (beyond the spec)

Settled against the installed mlx 0.32.2 / mlx-vlm 0.6.17 / mlx-lm 0.31.3 and the current code; implementers should not reopen them.

1. **Tag names.** Eligible projections carry `_sous_int8_prefill = True`; a handed-down shared Stage A sits on `_sous_int8_stage = (x, stage) | None`, both set via `object.__setattr__`. `stage` is the `(qa, sa, ra)` tuple `stage_a()` returns.
2. **Tagging walks `named_modules()` of the language root** — `model.language_model` when the attribute exists (mlx-vlm), else `model` (mlx-lm) — and considers a module only when its dotted path contains `.layers.` and not `self_attn`. That excludes `lm_head`, embeddings, vision towers and the attention projections without naming any class. An MLP (a module with `gate_proj`, `up_proj`, `down_proj`) is tagged all-or-nothing: one ineligible member leaves all three stock, so the MLP wrapper's fall-through never lets the `QuantizedLinear` wrapper route gate/up alone. Every other eligible projection under `.layers.` (the GDN `in_proj_qkv`, `in_proj_z`, `out_proj`) is tagged individually; `in_proj_a`/`in_proj_b` (N = 48) fail eligibility.
3. **`enable()` with zero eligible projections is `unavailable`** with reason `"no eligible projections (affine Q4 gs64 required)"`, not `active` with `routed == 0` — a mxfp checkpoint must read as "this feature did nothing".
4. **Wrapper targets are class objects.** `install_wrappers(classes)` takes `(cls, kind)` pairs, `kind in {"mlp", "gdn"}`; the default list resolves `mlx_vlm.models.qwen3_5.language.Qwen3_5MLP` / `Qwen3_5GatedDeltaNet` and `mlx_lm.models.qwen3_5.MLP` / `GatedDeltaNet` (the mlx-lm `MLP` name is `qwen3_next.Qwen3NextMLP`, shared with Qwen3-Next — harmless, untagged modules fall through), skipping any that fail to import. Tests inject tiny classes through the same parameter. The `nn.QuantizedLinear.__call__` wrapper is installed once per process alongside them.
5. **SwiGLU is `nn.silu(gate) * up`** inline in the MLP wrapper — what both mlx-vlm's and mlx-lm's `swiglu` compute — rather than importing either library's helper into a module that must import on ubuntu.
6. **`M` is a one-element int32 input, `K`/`N`/`T` are template constants** — one compiled instantiation per distinct projection shape (five on the default model), none per chunk length.
7. **The stored-layout read of scales/biases is implemented first (`SLAYOUT = 0`) and benchmarked against the transposed layout (`SLAYOUT = 1`) in Task 2**; the losing branch is deleted in the same task. The spec's rule: keep the stored layout unless it costs > 3% at M = 2048; otherwise transpose per call inside `qmm` — never a resident copy either way.
8. **Test fixtures quantize bf16 modules.** `nn.quantize` on an fp32 `nn.Linear` yields fp32 scales, which are ineligible by design; fixtures call `parent.set_dtype(mx.bfloat16)` before `nn.quantize(parent, group_size=64, bits=4)`, and reach the `QuantizedLinear` as `parent.layers[0]` (quantize replaces *children*, never the module passed in).
9. **Degradation warns via `warnings.warn`, activation logs via `logging`** — the engines already warn with `warnings.warn` when the drafter fails (`vlm.py`), and `MCPServer` installs a root stderr handler at INFO for the `int8 prefill: routed N projections` line.
10. **The runtime gate's numerics prompt** is the concatenation of `src/sous/engine/promptcache.py` and `src/sous/gateway/turn.py`, encoded by the engine, first 2,048 tokens, KL over the last 256 positions — the spike's standard prompt.

## File structure

| File | Responsibility |
|---|---|
| `src/sous/engine/int8prefill.py` (create) | Metal sources; `_kernel()` cache; `reorder_k`, `stage_a`, `qmm`, `linear_int8`; `availability()`; `eligible_linear`; tagging; class wrappers; `enable()` |
| `src/sous/config.py` (modify) | `_KNOWN["model"]`, `SousConfig.int8_prefill`, `_int8_prefill()` validator, loader |
| `src/sous/engine/base.py` (modify) | `_default_factory(int8_prefill=)`, `EngineManager` plumbing, `ManagedEngine.int8_prefill_status`, `EngineManager.status()` field |
| `src/sous/engine/vlm.py`, `src/sous/engine/lm.py` (modify) | constructor parameter + `enable()` call after `load()` |
| `THIRD_PARTY_NOTICES.md` (create) | Apache-2.0 attribution for the derived kernels |
| `README.md`, `CLAUDE.md` (modify) | config docs, gotchas |
| `tests/test_int8prefill.py` (create) | CI tests (Stage A, reorder, eligibility, routing, enable) + NAX-only exactness tests (`skipif`) |
| `tests/test_int8prefill_model.py` (create) | `model`-marked runtime gate |
| `tests/test_config.py`, `tests/test_engine_base.py` (modify) | config and plumbing tests |

---

### Task 1: Stage A kernel and the K-order

**Files:**
- Create: `src/sous/engine/int8prefill.py`
- Test: `tests/test_int8prefill.py`

**Interfaces:**
- Produces: `GROUP = 64`, `MIN_ROWS = 128`, `TILE_WM = 1`, `TILE_WN = 2`, `BM = 32`, `BN = 64`; `reorder_k(qa: mx.array) -> mx.array` (int8 `[M, K]` → same shape, fragment K-order, contiguous); `stage_a(x: mx.array) -> tuple[mx.array, mx.array, mx.array]` = `(qa int8 [M, K] reordered, sa float32 [M], ra int16 [K/64, M])` for `x` of shape `[..., K]` in bf16/fp16; `_kernel(name, input_names, output_names, source, header) -> Callable[..., list[Any]]` (cached per name).

- [ ] **Step 1: Confirm the branch holds the spec and this plan**

```bash
git checkout feat/int8-prefill && git log --oneline -2
```
Expected: the plan commit on top of `55b8419 docs: spec for INT8-activation prefill on the M5 tensor units`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_int8prefill.py`:

```python
"""INT8-activation prefill: Stage A, the K-order, eligibility, routing, enablement.

Everything here runs on any Metal GPU (CI is macos-15, no tensor units); the GEMM
itself is exercised by the `nax`-gated tests at the bottom, which skip where the
neural accelerators are absent and run on the M5 Pro.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sous.engine import int8prefill as i8  # noqa: E402 — after the importorskip guards


def _unreorder(qa):
    """Inverse of reorder_k: (c, a, b, j) -> (c, a, j, b) within each 64-code group."""
    m, k = qa.shape
    return qa.reshape(m, k // 64, 4, 2, 2, 4).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)


# ---- K-order --------------------------------------------------------------------


def test_reorder_k_slot_formula():
    """Slot 16c + 4t + j of a group holds code 16c + 8*(t>>1) + 2j + (t&1); the
    second group is the same pattern offset by 64."""
    ids = mx.arange(128, dtype=mx.int32).astype(mx.int8).reshape(1, 128)
    out = i8.reorder_k(ids).tolist()[0]
    for c in range(4):
        for t in range(4):
            for j in range(4):
                assert out[16 * c + 4 * t + j] == 16 * c + 8 * (t >> 1) + 2 * j + (t & 1)
    assert out[64:] == [64 + v for v in out[:64]]


def test_reorder_k_round_trips_and_is_contiguous():
    x = mx.random.randint(-127, 128, (3, 256)).astype(mx.int8)
    out = i8.reorder_k(x)
    assert out.shape == x.shape and out.dtype == mx.int8
    assert mx.array_equal(_unreorder(out), x).item()
    assert out.flags["row_contiguous"]


# ---- Stage A --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
def test_stage_a_matches_the_reference(dtype):
    m, k = 37, 1024
    x = (mx.random.normal((m, k)) * 0.5).astype(dtype)
    qa, sa, ra = i8.stage_a(x)
    mx.eval(qa, sa, ra)
    assert (qa.shape, qa.dtype) == ((m, k), mx.int8)
    assert (sa.shape, sa.dtype) == ((m,), mx.float32)
    assert (ra.shape, ra.dtype) == ((k // 64, m), mx.int16)

    xf = x.astype(mx.float32)
    amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    q_ref = mx.clip(mx.round(xf * (127.0 / amax)), -127, 127)
    q_got = _unreorder(qa).astype(mx.float32)
    diff = mx.abs(q_got - q_ref)
    # fp32 (kernel) vs fp32-from-the-same-inputs (mx ops) differ only at rounding
    # ties: never by more than one code, and rarely (the spike measured 0.03%).
    assert mx.max(diff).item() <= 1
    assert mx.mean(diff > 0).item() <= 1e-3
    assert mx.allclose(sa, (amax / 127.0).reshape(-1), rtol=1e-6, atol=0.0).item()
    # Ra is the sum of the kernel's OWN codes, exactly — the affine correction
    # relies on it — so it is checked against q_got, not q_ref.
    ra_ref = q_got.reshape(m, k // 64, 64).sum(axis=-1).astype(mx.int16).T
    assert mx.array_equal(ra, ra_ref).item()


def test_stage_a_zero_row_emits_zeros_not_nan():
    rest = (mx.random.normal((4, 128)) * 0.5).astype(mx.bfloat16)
    x = mx.concatenate([mx.zeros((1, 128), dtype=mx.bfloat16), rest])
    qa, sa, ra = i8.stage_a(x)
    assert mx.all(qa[0] == 0).item()
    assert sa[0].item() == 0.0
    assert mx.all(ra[:, 0] == 0).item()
    assert not mx.any(mx.isnan(sa)).item()


def test_stage_a_flattens_leading_dims():
    x = (mx.random.normal((1, 200, 128)) * 0.5).astype(mx.bfloat16)
    qa, sa, ra = i8.stage_a(x)
    assert qa.shape == (200, 128) and sa.shape == (200,) and ra.shape == (2, 200)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_int8prefill.py -q`
Expected: `ImportError: cannot import name 'int8prefill'` (collection error).

- [ ] **Step 4: Create the module with the Stage-A kernel**

Create `src/sous/engine/int8prefill.py`:

```python
"""INT8-activation prefill for affine-Q4/gs64 projections on the M5 tensor units.

Prefill GEMMs (>= MIN_ROWS rows) of tagged ``nn.QuantizedLinear`` projections run as
int8 activations x the checkpoint's own packed Q4 weights -> int32 on the neural
accelerators, with the affine correction applied per 64-code group. Two Metal
kernels live here as source strings and are compiled by mlx at model load
(``mx.fast.metal_kernel``): no native build, no ABI coupling to an mlx version.

Derived from oMLX (jundot/omlx#3548, Apache License 2.0): qwen35_oq_a8.metal,
qwen35_oq_a8_nax.metal and oq_a8_decode.h. See THIRD_PARTY_NOTICES.md.

Spec: docs/superpowers/specs/2026-09-11-int8-activation-prefill-design.md.
"""

from __future__ import annotations

import logging
import platform
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger("sous.engine.int8prefill")

GROUP = 64
# Below this many rows the Stage-A pass costs more than the faster GEMM saves
# (crossover measured between 4 and 8 rows; 1.6x by 128). Set well above the
# crossover so decode (1 row) and speculative verify (<= 6 rows) never route.
MIN_ROWS = 128
# The PR's best Q4 tile: BM = 2 * 16 * WM rows, BN = 32 * WN columns, 32 * WM * WN
# threads per threadgroup. The seven tile variants are within ~3% of each other.
TILE_WM = 1
TILE_WN = 2
BM = 2 * 16 * TILE_WM
BN = 32 * TILE_WN

_COMMON_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
"""

# Stage A: one 256-thread threadgroup (eight simdgroups) per row. Each simdgroup owns
# whole 64-code groups, so the group sums never straddle simdgroups and reduce with
# one simd_sum. Ra is built from the ROUNDED codes, not from x: the affine bias
# correction in the GEMM is exact only against the codes actually multiplied.
_STAGE_A_SOURCE = r"""
  constexpr int SG = 8;
  const int row = int(threadgroup_position_in_grid.x);
  constexpr int groups = K / 64;
  const int simd_gid = int(simdgroup_index_in_threadgroup);
  const int simd_lid = int(thread_index_in_simdgroup);
  const device T* xr = x + size_t(row) * size_t(K);
  device int8_t* qr = qa + size_t(row) * size_t(K);
  device short* rr = ra + size_t(row) * size_t(groups);
  threadgroup float partial_amax[SG];
  float amax = 0.0f;
  for (int g = simd_gid; g < groups; g += SG) {
    const int base = g * 64 + simd_lid;
    amax = max(amax, abs(float(xr[base])));
    amax = max(amax, abs(float(xr[base + 32])));
  }
  amax = simd_max(amax);
  if (simd_lid == 0) partial_amax[simd_gid] = amax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_amax = 0.0f;
  UNROLL for (int i = 0; i < SG; ++i) row_amax = max(row_amax, partial_amax[i]);
  // An all-zero row has no representable scale: emit zeros rather than NaN and
  // let the affine bias term carry that row's output.
  const float inv = row_amax > 0.0f ? (127.0f / row_amax) : 0.0f;
  if (simd_gid == 0 && simd_lid == 0) sa[row] = row_amax > 0.0f ? (row_amax / 127.0f) : 0.0f;
  for (int g = simd_gid; g < groups; g += SG) {
    const int base = g * 64 + simd_lid;
    const int q0 = int(clamp(rint(float(xr[base]) * inv), -127.0f, 127.0f));
    const int q1 = int(clamp(rint(float(xr[base + 32]) * inv), -127.0f, 127.0f));
    qr[base] = int8_t(q0);
    qr[base + 32] = int8_t(q1);
    // |Ra| <= 64 * 127 = 8128, so int16 holds it.
    const int gsum = simd_sum(q0 + q1);
    if (simd_lid == 0) rr[g] = short(gsum);
  }
"""

_kernels: dict[str, Callable[..., list[Any]]] = {}


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
    return kernel


def reorder_k(qa: Any) -> Any:
    """Put each 64-code group into the GEMM's fragment K-order.

    Slot ``16c + 4t + j`` of a group holds code ``k = 16c + 8*(t>>1) + 2j + (t&1)``:
    the order in which the GEMM's nibble decode hands weights to the tensor-op
    fragment. Legal because a dot product is order-independent and the affine
    group is the same set of 64 either way, so scales, biases and Ra are untouched.
    Writing ``t = 2a + b`` the map is the transpose of the last two axes of
    ``(c, a, j, b)``; one contiguous copy of an int8 [M, K].
    """
    import mlx.core as mx

    m, k = qa.shape
    return mx.contiguous(
        qa.reshape(m, k // GROUP, 4, 2, 4, 2).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)
    )


def stage_a(x: Any) -> tuple[Any, Any, Any]:
    """bf16/fp16 ``[..., K]`` -> ``(qa int8 [M, K] reordered, sa f32 [M], ra int16 [K/64, M])``.

    Ra comes back group-major so a group's row sums are contiguous across the lanes
    that need them in the GEMM.
    """
    import mlx.core as mx

    k = x.shape[-1]
    m = x.size // k
    x2 = x.reshape(m, k)
    qa, sa, ra = _kernel(
        "sous_int8_stage_a", ["x"], ["qa", "sa", "ra"], _STAGE_A_SOURCE, _COMMON_HEADER
    )(
        inputs=[x2],
        template=[("T", x.dtype), ("K", k)],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,), (m, k // GROUP)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    )
    return reorder_k(qa), sa, mx.contiguous(ra.T)
```

(`platform`, `re`, `warnings`, `dataclass`, `Iterable`, `logger` are used by Tasks 2–3; ruff will flag the unused ones now — that is expected and resolved within this task's ruff step by keeping only `logging`, `Callable`, `Any`, `cast` until the later tasks add their uses. Remove the unused imports before committing this task; Tasks 2 and 3 add them back with their code.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_int8prefill.py -q`
Expected: 6 passed (the first call compiles the Stage-A kernel; a second run is faster).

- [ ] **Step 6: Lint, format, type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: all clean. If ty complains about `x.shape` on `Any`, the parameter annotations are `Any` on purpose — check that `_kernel` returns through `cast`.

- [ ] **Step 7: Commit**

```bash
git add src/sous/engine/int8prefill.py tests/test_int8prefill.py
git commit -m "feat(engine): stage-A int8 activation quantization for prefill

Per-row int8 codes, scale and group sums in one runtime-compiled Metal
kernel, plus the fragment K-order the tensor-op GEMM (next commit) reads.
Nothing routes yet.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: The Q4 × INT8 tensor-unit GEMM, availability, and the notice

**Files:**
- Modify: `src/sous/engine/int8prefill.py`
- Create: `THIRD_PARTY_NOTICES.md`
- Test: `tests/test_int8prefill.py`

**Interfaces:**
- Consumes: `reorder_k`, `stage_a`, `_kernel`, `GROUP`, `BM`, `BN`, `TILE_WM`, `TILE_WN` (Task 1).
- Produces: `@dataclass(frozen=True) Availability(available: bool, reason: str | None = None)`; `availability() -> Availability`; `qmm(qa, sa, ra, weight, scales, biases) -> mx.array` (`[M, N]` in `scales.dtype`; `scales`/`biases` in their stored `[N, K/64]` layout); `linear_int8(linear, x, stage=None) -> mx.array` (`[..., N]`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_int8prefill.py`:

```python
# ---- availability -----------------------------------------------------------------


def _fake_platform(monkeypatch, mac_ver: str, arch: str):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: (mac_ver, ("", "", ""), "arm64"))
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})


@pytest.mark.parametrize(
    ("mac_ver", "arch", "available", "needle"),
    [
        ("26.6.2", "applegpu_g17s", True, None),
        ("26.2", "applegpu_g17g", True, None),
        ("26.6.2", "applegpu_g18p", True, None),
        ("26.6.2", "applegpu_g17p", False, "gen 17 < 18"),
        ("26.6.2", "applegpu_g16s", False, "gen 16 < 17"),
        ("26.1", "applegpu_g17s", False, "macOS 26.1 < 26.2"),
        ("26.6.2", "something_else", False, "unrecognized"),
    ],
)
def test_availability_mirrors_mlx_nax_rule(monkeypatch, mac_ver, arch, available, needle):
    _fake_platform(monkeypatch, mac_ver, arch)
    a = i8.availability()
    assert a.available is available
    if needle is not None:
        assert a.reason is not None and needle in a.reason


def test_availability_on_this_machine_reports_a_reason_when_unavailable():
    a = i8.availability()
    assert a.available or (isinstance(a.reason, str) and a.reason)


# ---- GEMM (needs the tensor units; skipped where they are absent) ----------------

nax = pytest.mark.skipif(
    not i8.availability().available, reason=f"no tensor units: {i8.availability().reason}"
)


def _unpack_q4(w):
    """uint32 [N, K/8] -> codes int32 [N, K]; code i of a word sits at bits 4i."""
    shifts = (mx.arange(8, dtype=mx.uint32) * 4).reshape(1, 1, 8)
    codes = (w[:, :, None] >> shifts) & mx.array(0xF, dtype=mx.uint32)
    return codes.reshape(w.shape[0], -1).astype(mx.int32)


def _random_words(n, words):
    hi = mx.random.randint(0, 1 << 16, (n, words)).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, (n, words)).astype(mx.uint32)
    return (hi << mx.array(16, dtype=mx.uint32)) | lo


@nax
@pytest.mark.parametrize(
    ("k", "n", "m"),
    [(64, 64, 32), (128, 64, 32), (1024, 128, 37), (1024, 128, 64), (256, 192, 300)],
)
def test_gemm_is_bit_exact_with_power_of_two_metadata(k, n, m):
    """With power-of-two scales and biases every fp32 partial sum is exact, so the
    kernel's only rounding is the final bf16 cast: it must equal the fp64-exact
    reference rounded to bf16. This is the test that fails if the fragment layout
    or the K-order is wrong."""
    qa_nat = mx.random.randint(-127, 128, (m, k)).astype(mx.int8)
    ra = qa_nat.astype(mx.int32).reshape(m, k // 64, 64).sum(axis=-1)  # [M, G]
    sa = mx.ones((m,), dtype=mx.float32)
    w = _random_words(n, k // 8)
    s, b = 2.0**-8, -(2.0**-5)
    scales = mx.full((n, k // 64), s, dtype=mx.bfloat16)
    biases = mx.full((n, k // 64), b, dtype=mx.bfloat16)
    # Integer-exact in fp32 on the CPU stream: |products| <= 127*15, sums < 2^24.
    dots = mx.matmul(
        qa_nat.astype(mx.float32), _unpack_q4(w).astype(mx.float32).T, stream=mx.cpu
    )
    ref = dots * s + ra.sum(axis=-1, keepdims=True).astype(mx.float32) * b
    got = i8.qmm(i8.reorder_k(qa_nat), sa, mx.contiguous(ra.astype(mx.int16).T), w, scales, biases)
    assert got.shape == (m, n) and got.dtype == mx.bfloat16
    assert mx.array_equal(got, ref.astype(mx.bfloat16)).item()


@nax
def test_gemm_applies_the_row_scale():
    k, n, m = 128, 64, 40
    qa_nat = mx.random.randint(-127, 128, (m, k)).astype(mx.int8)
    ra = qa_nat.astype(mx.int32).reshape(m, 2, 64).sum(axis=-1)
    sa = mx.array([2.0 ** (i % 5) for i in range(m)], dtype=mx.float32)
    w = _random_words(n, k // 8)
    scales = mx.full((n, 2), 2.0**-8, dtype=mx.bfloat16)
    biases = mx.full((n, 2), -(2.0**-5), dtype=mx.bfloat16)
    dots = mx.matmul(qa_nat.astype(mx.float32), _unpack_q4(w).astype(mx.float32).T, stream=mx.cpu)
    ref = (dots * 2.0**-8 + ra.sum(axis=-1, keepdims=True).astype(mx.float32) * -(2.0**-5)) * sa[
        :, None
    ]
    got = i8.qmm(i8.reorder_k(qa_nat), sa, mx.contiguous(ra.astype(mx.int16).T), w, scales, biases)
    assert mx.array_equal(got, ref.astype(mx.bfloat16)).item()


@nax
def test_gemm_tracks_stock_qmm_within_the_int8_noise():
    k, n, m = 5120, 1024, 300
    w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4, mode="affine")
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    got = i8.qmm(*i8.stage_a(x), packed, scales, biases).astype(mx.float32)
    ref = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True, group_size=64, bits=4, mode="affine"
    ).astype(mx.float32)
    rel_rms = mx.sqrt(mx.mean((got - ref) ** 2)) / mx.sqrt(mx.mean(ref**2))
    # Per-row int8 activation quantization noise; the spike measured 9.2e-3.
    assert rel_rms.item() < 1.5e-2


@nax
def test_linear_int8_keeps_leading_dims_and_matches_qmm():
    parent = nn.Sequential(nn.Linear(256, 128, bias=False))
    parent.set_dtype(mx.bfloat16)
    nn.quantize(parent, group_size=64, bits=4)
    linear = parent.layers[0]
    x = (mx.random.normal((1, 200, 256)) * 0.5).astype(mx.bfloat16)
    out = i8.linear_int8(linear, x)
    assert out.shape == (1, 200, 128) and out.dtype == mx.bfloat16
    stage = i8.stage_a(x)
    direct = i8.qmm(*stage, linear.weight, linear.scales, linear.biases).reshape(1, 200, 128)
    assert mx.array_equal(out, direct).item()
    # A handed-down stage is used as-is.
    assert mx.array_equal(i8.linear_int8(linear, x, stage), direct).item()


def test_qmm_rejects_untiled_shapes():
    qa = mx.zeros((32, 64), dtype=mx.int8)
    sa = mx.ones((32,), dtype=mx.float32)
    ra = mx.zeros((1, 32), dtype=mx.int16)
    with pytest.raises(ValueError, match="N=48"):
        i8.qmm(qa, sa, ra, mx.zeros((48, 8), dtype=mx.uint32), mx.ones((48, 1)), mx.zeros((48, 1)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_int8prefill.py -q -k "availability or gemm or linear_int8 or qmm"`
Expected: FAIL with `AttributeError: module 'sous.engine.int8prefill' has no attribute 'availability'` (and `qmm`).

- [ ] **Step 3: Add availability, the GEMM kernel (both metadata layouts, for the benchmark), `qmm`, `linear_int8`**

Add to `src/sous/engine/int8prefill.py` after `_STAGE_A_SOURCE`:

```python
_NAX_HEADER = (
    _COMMON_HEADER
    + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Mirror of mlx::steel::BaseNAXFrag::get_coord() (mlx 0.32.2, steel/gemm/nax.h): the
// lane's (column x, row y) inside a 16x16 fragment. A lane's slots r*4 + j are
// (row y + 8r, col x + j) of its fragment; a 16x32 destination is two such
// fragments along n.
inline short2 nax_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}
"""
)

# oq_a8_qmm_t_nax_v8<T, BITS=4, ACT_MODE=0, WM, WN> from oMLX, with K/N as template
# constants, M as a one-element input, and SLAYOUT selecting how scales/biases are
# read (0: the checkpoint's stored [N, K/64]; 1: transposed [K/64, N]).
#
#   out[m, n] = sa[m] * sum_g( Sw[n,g] * sum_{k in g} qa[m,k]*qw[n,k] + Bw[n,g] * Ra[g,m] )
#
# The inner sum is four 16x32x16 int8 tensor-op steps per group into int32; the
# fp32 correction lands once per group as two fmas. Weights are the packed uint32
# words unmodified: one uint2 per weight row per group holds the lane's 16-code run
# and the nibbles are decoded straight into the right-hand fragment. Rows past M
# read row M-1 and are never stored, so the K loop has no bounds branch.
_GEMM_SOURCE = r"""
  constexpr int kFragM = 16, kFragN = 32, kFragK = 16, kDestElems = 16, kSteps = 4;
  constexpr int TM = 2;
  constexpr int TBM = TM * kFragM * WM;
  constexpr int TBN = kFragN * WN;
  constexpr int words = 8;
  constexpr int groups = K / 64;
  const int M = Mp[0];

  const int sg = int(simdgroup_index_in_threadgroup);
  const int sg_m = sg % WM;
  const int sg_n = sg / WM;
  const int row_base = int(threadgroup_position_in_grid.y) * TBM + sg_m * (TM * kFragM);
  const int col_base = int(threadgroup_position_in_grid.x) * TBN + sg_n * kFragN;
  const short2 coord = nax_coord(thread_index_in_simdgroup);

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM, kFragN, kFragK, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kFragM, kFragN, kFragK, false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  const int n_run0 = col_base + int(coord.x);
  const int m_base = row_base + int(coord.y);

  float Cf[TM][kDestElems];
  UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int e = 0; e < kDestElems; ++e) Cf[i][e] = 0.0f; }

  const device uint32_t* wbase = w + size_t(col_base + int(coord.y)) * size_t(groups) * words;
  const int w_stride8 = 8 * groups * words;
  const int w_stride16 = kFragM * groups * words;
  const int cx = int(coord.x) >> 2;

  for (int g = 0; g < groups; ++g) {
    uint2 wg[4];
    UNROLL for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 + (q >> 1) * w_stride16 + size_t(g) * words;
      wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
    }
    UNROLL for (int t = 0; t < kSteps; ++t) {
      UNROLL for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        const uint32_t word = wg[q][t >> 1];
        const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
      UNROLL for (int hf = 0; hf < 2; ++hf) {
        UNROLL for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + hf * kFragM + r * 8, M - 1);
          const char4 quad = as_type<char4>(*reinterpret_cast<const device uint32_t*>(
              qa + size_t(m) * size_t(K) + size_t(g) * 64 + size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else        { if (hf == 0) op.run(ct_a, ct_b, acc0);     else op.run(ct_a, ct_b, acc1); }
      }
    }
    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    if (SLAYOUT == 0) {
      UNROLL for (int h = 0; h < 2; ++h) {
        UNROLL for (int j = 0; j < 4; ++j) {
          const size_t n = size_t(n_run0 + h * kFragM + j);
          sv[h][j] = scales[n * size_t(groups) + size_t(g)];
          bv[h][j] = biases[n * size_t(groups) + size_t(g)];
        }
      }
    } else {
      const device T* srow = scales + size_t(g) * size_t(N);
      const device T* brow = biases + size_t(g) * size_t(N);
      UNROLL for (int h = 0; h < 2; ++h) {
        const int n0 = n_run0 + h * kFragM;
        sv[h] = *reinterpret_cast<const device vec<T, 4>*>(srow + n0);
        bv[h] = *reinterpret_cast<const device vec<T, 4>*>(brow + n0);
      }
    }
    const device short* rrow = ra + size_t(g) * size_t(M);
    float r_g[TM][2];
    UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int r = 0; r < 2; ++r) {
      const int m = min(m_base + i * kFragM + r * 8, M - 1);
      r_g[i][r] = float(rrow[m]);
    } }
    UNROLL for (int e = 0; e < kDestElems; ++e) {
      const int r = ((e & 7) >> 2);
      const float swc = float(sv[e >> 3][e & 3]);
      const float bwc = float(bv[e >> 3][e & 3]);
      Cf[0][e] = metal::fma(swc, float(acc0[e]), metal::fma(bwc, r_g[0][r], Cf[0][e]));
      Cf[1][e] = metal::fma(swc, float(acc1[e]), metal::fma(bwc, r_g[1][r], Cf[1][e]));
    }
  }
  UNROLL for (int i = 0; i < TM; ++i) { UNROLL for (int e = 0; e < kDestElems; ++e) {
    const int ee = e & 7;
    const int r = ee >> 2;
    const int m = row_base + i * kFragM + int(coord.y) + r * 8;
    if (m < M) {
      const int n = col_base + (e >> 3) * kFragM + int(coord.x) + (ee & 3);
      out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(sa[m] * Cf[i][e]);
    }
  } }
"""


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


def availability() -> Availability:
    """Mirror of mlx's own is_nax_available() (device.cpp, 0.32.2): macOS >= 26.2
    and an Apple GPU of generation >= 17 (>= 18 for 'p'-suffix parts) — the M5
    family and later. Older GPUs run the Metal-4 tensor ops on the plain shader
    path, where int8 buys nothing, so they are `unavailable`, not merely slow."""
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
    return Availability(True)


def qmm(qa: Any, sa: Any, ra: Any, weight: Any, scales: Any, biases: Any) -> Any:
    """``[M, N]`` in ``scales.dtype`` from Stage-A outputs and a projection's packed
    affine-Q4 weight with its scales/biases in their stored ``[N, K/64]`` layout."""
    import mlx.core as mx

    m, k = qa.shape
    n = weight.shape[0]
    if n % BN or k % GROUP:
        raise ValueError(f"int8 prefill GEMM needs N % {BN} == 0 and K % {GROUP} == 0; got N={n} K={k}")
    (out,) = _kernel(
        "sous_int8_q4_gemm",
        ["qa", "sa", "ra", "w", "scales", "biases", "Mp"],
        ["out"],
        _GEMM_SOURCE,
        _NAX_HEADER,
    )(
        inputs=[qa, sa, ra, weight, scales, biases, mx.array([m], dtype=mx.int32)],
        template=[
            ("T", scales.dtype),
            ("WM", TILE_WM),
            ("WN", TILE_WN),
            ("K", k),
            ("N", n),
            ("SLAYOUT", 0),
        ],
        grid=(32 * (n // BN), TILE_WM * ((m + BM - 1) // BM), TILE_WN),
        threadgroup=(32, TILE_WM, TILE_WN),
        output_shapes=[(m, n)],
        output_dtypes=[scales.dtype],
    )
    return out


def linear_int8(linear: Any, x: Any, stage: tuple[Any, Any, Any] | None = None) -> Any:
    """One tagged projection on the INT8 path; ``stage`` is a Stage A already taken
    for this exact ``x`` (shared across gate/up or qkv/z), else it is taken here."""
    qa, sa, ra = stage if stage is not None else stage_a(x)
    out = qmm(qa, sa, ra, linear.weight, linear.scales, linear.biases)
    return out.reshape(*x.shape[:-1], linear.weight.shape[0])
```

Restore the imports the module now uses: `platform`, `re`, `dataclass` (keep `warnings`, `Iterable`, `logger` for Task 3 — if ruff flags them, remove and re-add in Task 3).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_int8prefill.py -q`
Expected: on the M5 Pro every test passes (the first GEMM call compiles for a few seconds per distinct `(K, N)`); on a non-NAX Mac the `nax`-marked tests skip and the rest pass.

If `test_gemm_is_bit_exact_with_power_of_two_metadata` fails: first suspect the fragment layout (`nax_coord`, the `base`/`ct_a` indices) and the K-order — compare `_GEMM_SOURCE` against the spec's *GEMM* section literally; do not loosen the test.

- [ ] **Step 5: Benchmark the two metadata layouts and lock one (the spec's ≤ 3% rule)**

Run this throwaway script (not committed) from the repo root:

```bash
uv run python - <<'EOF'
import time
import mlx.core as mx
from sous.engine import int8prefill as i8

def timeit(fn, warm=3, it=15):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / it

def run(layout, qa, sa, ra, w, scales, biases):
    m, k = qa.shape; n = w.shape[0]
    s, b = (scales, biases) if layout == 0 else (mx.contiguous(scales.T), mx.contiguous(biases.T))
    mx.eval(s, b)
    kern = i8._kernel("sous_int8_q4_gemm", ["qa","sa","ra","w","scales","biases","Mp"], ["out"], i8._GEMM_SOURCE, i8._NAX_HEADER)
    return lambda: kern(inputs=[qa, sa, ra, w, s, b, mx.array([m], dtype=mx.int32)],
        template=[("T", mx.bfloat16), ("WM", i8.TILE_WM), ("WN", i8.TILE_WN), ("K", k), ("N", n), ("SLAYOUT", layout)],
        grid=(32 * (n // i8.BN), i8.TILE_WM * ((m + i8.BM - 1) // i8.BM), i8.TILE_WN),
        threadgroup=(32, i8.TILE_WM, i8.TILE_WN), output_shapes=[(m, n)], output_dtypes=[mx.bfloat16])[0]

M = 2048
for name, K, N in [("gate_up", 5120, 17408), ("down", 17408, 5120), ("gdn_qkv", 5120, 10240), ("gdn_out", 6144, 5120)]:
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4, mode="affine")
    x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
    qa, sa, ra = i8.stage_a(x); mx.eval(qa, sa, ra, packed, scales, biases)
    t0 = timeit(run(0, qa, sa, ra, packed, scales, biases))
    t1 = timeit(run(1, qa, sa, ra, packed, scales, biases))
    print(f"{name:<8} stored[N,G] {2*M*N*K/t0/1e12:5.1f} TOP/s  transposed[G,N] {2*M*N*K/t1/1e12:5.1f} TOP/s  stored/transposed = {t1/t0:.3f}")
EOF
```

Decision: if `stored/transposed >= 0.97` on every shape (the stored layout costs ≤ 3%), keep `SLAYOUT == 0`: delete the `else` branch and the `("SLAYOUT", 0)` template entry, and replace the `if (SLAYOUT == 0) {` / `}` wrapper with its body. Otherwise keep the transposed branch only: delete the `if (SLAYOUT == 0)` body and the template entry, and make `qmm` transpose per call — replace `inputs=[qa, sa, ra, weight, scales, biases, ...]` with

```python
        inputs=[
            qa,
            sa,
            ra,
            weight,
            mx.contiguous(scales.T),
            mx.contiguous(biases.T),
            mx.array([m], dtype=mx.int32),
        ],
```

and in the `else` branch's comment note the measured ratio. Either way, record both TOP/s columns for the four shapes in a comment above `_GEMM_SOURCE` (e.g. `# Layout benchmark 2026-09-xx, M=2048: stored 50.1/42.5/49.8/46.9 vs transposed 51.1/42.8/50.0/46.8 TOP/s -> stored kept.`) so the decision is reproducible.

Then rerun `uv run pytest tests/test_int8prefill.py -q` — the exactness tests must still pass with the locked layout.

- [ ] **Step 6: Write the third-party notice**

Create `THIRD_PARTY_NOTICES.md`:

```markdown
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
```

- [ ] **Step 7: Lint, format, type-check, full CI-equivalent test run**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q`
Expected: clean; all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/sous/engine/int8prefill.py tests/test_int8prefill.py THIRD_PARTY_NOTICES.md
git commit -m "feat(engine): int8 x packed-Q4 prefill GEMM on the M5 tensor units

Runtime-compiled port of oMLX's oq_a8 NAX kernel (Apache-2.0, see
THIRD_PARTY_NOTICES.md): int8 activations against the checkpoint's own
packed affine-Q4 words, affine correction per 64-group, bit-exact against
an integer reference. Availability mirrors mlx's is_nax_available rule.
Nothing routes yet.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Eligibility, tagging, class wrappers, `enable()`

**Files:**
- Modify: `src/sous/engine/int8prefill.py`
- Test: `tests/test_int8prefill.py`

**Interfaces:**
- Consumes: `stage_a`, `linear_int8`, `availability`, `MIN_ROWS`, `BN`, `GROUP` (Tasks 1–2).
- Produces: `eligible_linear(linear) -> bool`; `install_wrappers(classes: Iterable[tuple[type, str]] | None = None) -> None`; `enable(model, *, enabled: bool) -> dict[str, Any]` returning `{"state": "off" | "active" | "unavailable", "reason": str | None, "routed": int}`; module-private `_tag(model) -> int`, `_untag(model) -> None`, `_warm_up(model) -> None`, `_TAG = "_sous_int8_prefill"`, `_STAGE = "_sous_int8_stage"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_int8prefill.py`:

```python
# ---- eligibility --------------------------------------------------------------


def _qlinear(k, n, *, bias=False, dtype=mx.bfloat16, **quant):
    parent = nn.Sequential(nn.Linear(k, n, bias=bias))
    parent.set_dtype(dtype)
    nn.quantize(parent, **{"group_size": 64, "bits": 4, **quant})
    return parent.layers[0]


def test_eligible_linear_table():
    assert i8.eligible_linear(_qlinear(128, 256))
    assert i8.eligible_linear(_qlinear(128, 256, dtype=mx.float16))
    assert not i8.eligible_linear(_qlinear(128, 256, bits=8))
    assert not i8.eligible_linear(_qlinear(128, 256, group_size=128))
    assert not i8.eligible_linear(_qlinear(128, 256, group_size=32, mode="mxfp4"))
    assert not i8.eligible_linear(_qlinear(128, 48)), "N must tile by 64"
    assert not i8.eligible_linear(_qlinear(128, 256, bias=True))
    assert not i8.eligible_linear(_qlinear(128, 256, dtype=mx.float32)), "fp32 scales"
    assert not i8.eligible_linear(nn.Linear(128, 256)), "not quantized"


# ---- routing (any Metal GPU: qmm is replaced by an exact-enough reference) ------


def _reference_qmm(qa, sa, ra, w, scales, biases):
    x_hat = _unreorder(qa).astype(mx.float32) * sa[:, None]
    wd = mx.dequantize(w, scales, biases, group_size=64, bits=4, mode="affine")
    return (x_hat @ wd.astype(mx.float32).T).astype(scales.dtype)


class _TinyGDN(nn.Module):
    """Stand-in with the projection names the GDN wrapper looks for."""

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(128, 256, bias=False)
        self.in_proj_z = nn.Linear(128, 128, bias=False)
        self.in_proj_a = nn.Linear(128, 48, bias=False)
        self.out_proj = nn.Linear(128, 128, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        mixed = self.in_proj_qkv(inputs)[..., :128] + self.in_proj_z(inputs)
        return self.out_proj(mixed) + self.in_proj_a(inputs)[..., :1]


class _Counter:
    def __init__(self, monkeypatch):
        self.stage_calls = 0
        self.qmm_calls = 0
        real_stage_a = i8.stage_a

        def counting_stage_a(x):
            self.stage_calls += 1
            return real_stage_a(x)

        def counting_qmm(*args):
            self.qmm_calls += 1
            return _reference_qmm(*args)

        monkeypatch.setattr(i8, "stage_a", counting_stage_a)
        monkeypatch.setattr(i8, "qmm", counting_qmm)


def _mlp():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    mlp = Qwen3_5MLP(128, 256)
    mlp.set_dtype(mx.bfloat16)
    nn.quantize(mlp, group_size=64, bits=4)
    return mlp


def _tag_all(module):
    for _, m in module.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            object.__setattr__(m, i8._TAG, True)


@pytest.fixture
def wrappers():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    i8.install_wrappers([(Qwen3_5MLP, "mlp"), (_TinyGDN, "gdn")])


def test_mlp_routes_with_one_shared_stage_above_the_floor(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    mlp = _mlp()
    x = (mx.random.normal((1, 130, 128)) * 0.5).astype(mx.bfloat16)
    stock = mlp(x)
    _tag_all(mlp)
    routed = mlp(x)
    assert (counter.stage_calls, counter.qmm_calls) == (2, 3), "gate+up share; down has its own"
    rel = mx.sqrt(mx.mean((routed - stock).astype(mx.float32) ** 2)) / mx.sqrt(
        mx.mean(stock.astype(mx.float32) ** 2)
    )
    assert rel.item() < 3e-2


def test_rows_below_the_floor_take_the_stock_path(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    mlp = _mlp()
    _tag_all(mlp)
    mlp((mx.random.normal((1, 127, 128)) * 0.5).astype(mx.bfloat16))
    mlp((mx.random.normal((1, 1, 128)) * 0.5).astype(mx.bfloat16))
    assert counter.qmm_calls == 0


def test_untagged_modules_are_never_routed(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    other = _mlp()  # same class, never tagged: a second resident model
    other((mx.random.normal((1, 130, 128)) * 0.5).astype(mx.bfloat16))
    assert counter.qmm_calls == 0


def test_gdn_shares_one_stage_between_qkv_and_z_and_clears_it(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    gdn = _TinyGDN()
    gdn.set_dtype(mx.bfloat16)
    nn.quantize(gdn, group_size=64, bits=4)
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        object.__setattr__(getattr(gdn, name), i8._TAG, True)
    x = (mx.random.normal((1, 200, 128)) * 0.5).astype(mx.bfloat16)
    gdn(x)
    # one Stage A for qkv+z, one for out_proj; in_proj_a (N=48) untagged -> stock
    assert (counter.stage_calls, counter.qmm_calls) == (2, 3)
    assert getattr(gdn.in_proj_qkv, i8._STAGE) is None
    assert getattr(gdn.in_proj_z, i8._STAGE) is None


def test_install_wrappers_is_idempotent():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    i8.install_wrappers([(Qwen3_5MLP, "mlp")])
    first = Qwen3_5MLP.__call__
    i8.install_wrappers([(Qwen3_5MLP, "mlp")])
    assert Qwen3_5MLP.__call__ is first


# ---- enable() -----------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self, down_bits=4):
        super().__init__()
        self.mlp = _mlp()
        if down_bits != 4:
            self.mlp.down_proj = _qlinear(256, 128, bits=down_bits)
        self.linear_attn = _TinyGDN()
        self.linear_attn.set_dtype(mx.bfloat16)
        nn.quantize(self.linear_attn, group_size=64, bits=4)
        self.self_attn = nn.Sequential(nn.Linear(128, 256, bias=False))
        self.self_attn.set_dtype(mx.bfloat16)
        nn.quantize(self.self_attn, group_size=64, bits=4)


class _Inner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class _LanguageModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _Inner(layers)
        self.lm_head = _qlinear(128, 256)


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.language_model = _LanguageModel(layers)


def _tags(model):
    return sorted(
        path
        for path, m in model.named_modules()
        if isinstance(m, nn.QuantizedLinear) and getattr(m, i8._TAG, False)
    )


def test_enable_off_touches_nothing():
    model = _Model([_Layer()])
    assert i8.enable(model, enabled=False) == {"state": "off", "reason": None, "routed": 0}
    assert _tags(model) == []


def test_enable_unavailable_tags_nothing(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(False, "no tensor units"))
    model = _Model([_Layer()])
    status = i8.enable(model, enabled=True)
    assert status == {"state": "unavailable", "reason": "no tensor units", "routed": 0}
    assert _tags(model) == []


def test_enable_tags_mlp_and_gdn_but_not_attention_or_lm_head(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(), _Layer()])
    status = i8.enable(model, enabled=True)
    assert status == {"state": "active", "reason": None, "routed": 12}
    tagged = _tags(model)
    assert len(tagged) == 12
    assert all(".layers." in p for p in tagged)
    assert not any("self_attn" in p or "lm_head" in p or "in_proj_a" in p for p in tagged)


def test_enable_tags_an_mlp_all_or_nothing(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(down_bits=8)])
    status = i8.enable(model, enabled=True)
    assert status["routed"] == 3, "only the GDN projections; the MLP with an 8-bit down stays stock"
    assert not any(".mlp." in p for p in _tags(model))


def test_enable_with_no_eligible_projection_is_unavailable(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    model = _Model([])
    status = i8.enable(model, enabled=True)
    assert status["state"] == "unavailable"
    assert status["reason"] is not None and "no eligible projections" in status["reason"]


def test_enable_degrades_when_warm_up_fails(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))

    def boom(model):
        raise RuntimeError("compiler said no")

    monkeypatch.setattr(i8, "_warm_up", boom)
    model = _Model([_Layer()])
    with pytest.warns(UserWarning, match="compiler said no"):
        status = i8.enable(model, enabled=True)
    assert status == {"state": "unavailable", "reason": "compiler said no", "routed": 0}
    assert _tags(model) == []


def test_warm_up_runs_one_int8_linear_per_distinct_shape(monkeypatch):
    seen = []

    def fake_linear_int8(linear, x, stage=None):
        seen.append((x.shape, linear.weight.shape[0], x.dtype))
        return mx.zeros((x.shape[0], linear.weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(i8, "linear_int8", fake_linear_int8)
    model = _Model([_Layer(), _Layer()])
    i8._tag(model)
    i8._warm_up(model)
    # gate/up (128->256), down (256->128), qkv (128->256), z (128->128), out (128->128):
    # distinct (K, N, dtype) = {(128,256), (256,128), (128,128)}
    assert set(seen) == {
        ((128, 128), 256, mx.bfloat16),
        ((128, 256), 128, mx.bfloat16),
        ((128, 128), 128, mx.bfloat16),
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_int8prefill.py -q -k "eligible or routes or floor or untagged or gdn or idempotent or enable or warm_up"`
Expected: FAIL with `AttributeError: module 'sous.engine.int8prefill' has no attribute 'eligible_linear'` and friends.

- [ ] **Step 3: Implement eligibility, tagging, wrappers, enable**

Append to `src/sous/engine/int8prefill.py`:

```python
# --------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------

# Set with object.__setattr__ so they live in the module's __dict__: nn.Module's own
# __setattr__ would put a bool or a tuple of arrays INTO the parameter tree, where
# parameters() and named_modules() would find it.
_TAG = "_sous_int8_prefill"
_STAGE = "_sous_int8_stage"


def eligible_linear(linear: Any) -> bool:
    """Affine Q4 gs64 ``nn.QuantizedLinear`` with fp16/bf16 scales, no bias, and a
    column count the GEMM tiles by 64 — the only shape the kernel decodes."""
    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(linear, nn.QuantizedLinear) or "bias" in linear:
        return False
    if getattr(linear, "mode", None) != "affine":
        return False
    if getattr(linear, "bits", None) != 4 or getattr(linear, "group_size", None) != GROUP:
        return False
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return False
    if weight.dtype != mx.uint32 or weight.ndim != 2 or scales.ndim != 2:
        return False
    if scales.dtype not in (mx.float16, mx.bfloat16) or biases.dtype != scales.dtype:
        return False
    if scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        return False
    n = weight.shape[0]
    k = scales.shape[1] * GROUP
    return weight.shape[1] * 32 == k * 4 and n % BN == 0 and k % GROUP == 0


def _tagged(module: Any) -> bool:
    return bool(getattr(module, _TAG, False))


def _rows_ok(x: Any) -> bool:
    import mlx.core as mx

    return x.ndim >= 2 and x.shape[-2] >= MIN_ROWS and x.dtype in (mx.bfloat16, mx.float16)


def _root(model: Any) -> Any:
    # mlx-vlm wraps the text model as `language_model`; mlx-lm's Model is the root.
    return getattr(model, "language_model", model)


def _is_mlp(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj"))


def _tag(model: Any) -> int:
    """Tag eligible projections under the decoder layers; return how many."""
    import mlx.nn as nn

    count = 0
    seen: set[int] = set()
    for path, module in _root(model).named_modules():
        if ".layers." not in f".{path}." or "self_attn" in path:
            continue
        if _is_mlp(module):
            trio = (module.gate_proj, module.up_proj, module.down_proj)
            # All or nothing: with only gate/up tagged the MLP wrapper would fall
            # through to the original call, whose gate/up would then route alone.
            if all(eligible_linear(m) for m in trio):
                for m in trio:
                    if id(m) not in seen:
                        object.__setattr__(m, _TAG, True)
                        seen.add(id(m))
                        count += 1
            continue
        if isinstance(module, nn.QuantizedLinear) and eligible_linear(module):
            if id(module) in seen or _in_mlp_path(path):
                continue
            object.__setattr__(module, _TAG, True)
            seen.add(id(module))
            count += 1
    return count


def _in_mlp_path(path: str) -> bool:
    return path.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")


def _untag(model: Any) -> None:
    for _, module in _root(model).named_modules():
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, False)
        if _STAGE in getattr(module, "__dict__", {}):
            object.__setattr__(module, _STAGE, None)


_WRAPPED: set[type] = set()
_QUANTIZED_LINEAR_WRAPPED = False

_TARGETS = (
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5MLP", "mlp"),
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5GatedDeltaNet", "gdn"),
    ("mlx_lm.models.qwen3_5", "MLP", "mlp"),
    ("mlx_lm.models.qwen3_5", "GatedDeltaNet", "gdn"),
)


def _resolve_targets() -> list[tuple[type, str]]:
    import importlib

    found: list[tuple[type, str]] = []
    for module_name, class_name, kind in _TARGETS:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        found.append((cls, kind))
    return found


def _wrap_mlp(cls: type) -> None:
    import mlx.nn as nn

    orig = cls.__call__

    def call(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            _rows_ok(x)
            and _tagged(self.gate_proj)
            and _tagged(self.up_proj)
            and _tagged(self.down_proj)
        ):
            shared = stage_a(x)
            gate = linear_int8(self.gate_proj, x, shared)
            up = linear_int8(self.up_proj, x, shared)
            return linear_int8(self.down_proj, nn.silu(gate) * up)
        return orig(self, x, *args, **kwargs)

    cls.__call__ = call  # ty: ignore[invalid-assignment]


def _wrap_gdn(cls: type) -> None:
    orig = cls.__call__

    def call(self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        share = [
            proj
            for proj in (getattr(self, "in_proj_qkv", None), getattr(self, "in_proj_z", None))
            if proj is not None and _tagged(proj)
        ]
        if not share or not _rows_ok(inputs):
            return orig(self, inputs, *args, **kwargs)
        stage = stage_a(inputs)
        for proj in share:
            object.__setattr__(proj, _STAGE, (inputs, stage))
        try:
            return orig(self, inputs, *args, **kwargs)
        finally:
            for proj in share:
                object.__setattr__(proj, _STAGE, None)

    cls.__call__ = call  # ty: ignore[invalid-assignment]


def _wrap_quantized_linear() -> None:
    import mlx.nn as nn

    orig = nn.QuantizedLinear.__call__

    def call(self: Any, x: Any) -> Any:
        if _tagged(self) and _rows_ok(x):
            handed = getattr(self, _STAGE, None)
            # Identity, not equality: the GDN wrapper handed a stage for this exact
            # input object; any other array gets its own Stage A.
            stage = handed[1] if handed is not None and handed[0] is x else None
            return linear_int8(self, x, stage)
        return orig(self, x)

    nn.QuantizedLinear.__call__ = call  # ty: ignore[invalid-assignment]


def install_wrappers(classes: Iterable[tuple[type, str]] | None = None) -> None:
    """Wrap the MLP / GDN classes (default: the Qwen3.5 classes of mlx-vlm and
    mlx-lm that import) and ``nn.QuantizedLinear``, once per process. Wrappers
    act only on tagged modules, so installing them is behaviour-neutral for every
    other model in the process."""
    global _QUANTIZED_LINEAR_WRAPPED
    for cls, kind in _resolve_targets() if classes is None else classes:
        if cls in _WRAPPED:
            continue
        (_wrap_mlp if kind == "mlp" else _wrap_gdn)(cls)
        _WRAPPED.add(cls)
    if not _QUANTIZED_LINEAR_WRAPPED:
        _wrap_quantized_linear()
        _QUANTIZED_LINEAR_WRAPPED = True


def _warm_up(model: Any) -> None:
    """Compile every kernel instantiation the tagged projections need, on a zero
    input of MIN_ROWS rows: first-turn latency stays flat, and a compiler that
    rejects the source or a changed metal_kernel signature fails here, inside
    enable()'s degradation path, rather than inside a user's turn."""
    import mlx.core as mx
    import mlx.nn as nn

    done: set[tuple[int, int, Any]] = set()
    for _, module in _root(model).named_modules():
        if not (isinstance(module, nn.QuantizedLinear) and _tagged(module)):
            continue
        k = module.scales.shape[1] * GROUP
        key = (k, module.weight.shape[0], module.scales.dtype)
        if key in done:
            continue
        done.add(key)
        mx.eval(linear_int8(module, mx.zeros((MIN_ROWS, k), dtype=module.scales.dtype)))


def enable(model: Any, *, enabled: bool) -> dict[str, Any]:
    """Opt one loaded model in. Returns the status the engine exposes; never raises
    — a model load must not fail because a prefill accelerator is missing."""
    if not enabled:
        return {"state": "off", "reason": None, "routed": 0}
    avail = availability()
    if not avail.available:
        return {"state": "unavailable", "reason": avail.reason, "routed": 0}
    try:
        routed = _tag(model)
        if routed == 0:
            _untag(model)
            return {
                "state": "unavailable",
                "reason": "no eligible projections (affine Q4 gs64 required)",
                "routed": 0,
            }
        install_wrappers()
        _warm_up(model)
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _untag(model)
        warnings.warn(
            f"sous: int8 prefill unavailable ({e}); prefilling with stock kernels",
            stacklevel=2,
        )
        return {"state": "unavailable", "reason": str(e), "routed": 0}
    logger.info("int8 prefill: routed %d projections", routed)
    return {"state": "active", "reason": None, "routed": routed}
```

Make sure the module imports `warnings` and `Iterable` (Task 1's header listed them).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_int8prefill.py -q`
Expected: all pass (on a non-NAX Mac the `nax` tests skip). If `test_enable_tags_mlp_and_gdn_but_not_attention_or_lm_head` counts 14 instead of 12, `_tag` is tagging `self_attn` — check the path filter; if 6, the GDN projections were skipped by `_in_mlp_path` — check that helper only matches the three MLP leaf names.

- [ ] **Step 5: Lint, format, type-check, full run**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q`
Expected: clean. If ty rejects the `cls.__call__ = call` assignments even with the pragma, use `setattr(cls, "__call__", call)` instead and drop the pragma.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/int8prefill.py tests/test_int8prefill.py
git commit -m "feat(engine): route tagged Qwen3.5 MLP/GDN prefill through the int8 GEMM

Per-model tagging (object.__setattr__, outside mlx's parameter tree) plus
process-wide wrappers that act only on tagged modules with >= 128 rows,
sharing one Stage A across gate/up and qkv/z. enable() degrades to stock
kernels on any failure and reports a status dict; nothing calls it yet.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: The `[model] int8_prefill` config key

**Files:**
- Modify: `src/sous/config.py:46-58` (`_KNOWN["model"]`), `src/sous/config.py:100-108` (`SousConfig` fields near `speculative_block_size`), `src/sous/config.py:222-253` (validators), `src/sous/config.py:436-446` (loader)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SousConfig.int8_prefill: bool = False`; `_int8_prefill(model: dict) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (after the `prompt_cache_gb` tests, ~line 322):

```python
# ---- [model].int8_prefill -----------------------------------------------------


def test_int8_prefill_defaults_off(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x/y'\n")
    assert load_config(p).int8_prefill is False
    assert SousConfig().int8_prefill is False


def test_int8_prefill_reads_true_without_an_unknown_key_warning(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nint8_prefill = true\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.int8_prefill is True


@pytest.mark.parametrize("bad", ['"yes"', "1", "0.5"])
def test_int8_prefill_rejects_non_booleans_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nint8_prefill = {bad}\n")
    with pytest.warns(UserWarning, match=r"\[model\]\.int8_prefill"):
        cfg = load_config(p)
    assert cfg.int8_prefill is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q -k int8_prefill`
Expected: FAIL — `AttributeError: 'SousConfig' object has no attribute 'int8_prefill'`, and the `true` case warns about an unknown key.

- [ ] **Step 3: Add the key**

In `src/sous/config.py`, `_KNOWN["model"]`: add `"int8_prefill",` after `"speculative_block_size",`.

In `SousConfig`, after `speculative_block_size: int = 3`:

```python
    # Prefill matmuls of affine-Q4/gs64 projections on the M5 tensor units with
    # int8 activations (spec 2026-09-11): 1.5x prefill measured on the M5 Pro, but
    # int8 activations change numerics (KL 0.033 vs stock on the standard prompt,
    # inside the 4-bit weight envelope of 0.052), so it ships off until the
    # tool-loop A/B says otherwise. Ignored, with a status reason, on GPUs
    # without neural accelerators (pre-M5) or macOS < 26.2.
    int8_prefill: bool = False
```

After `_prompt_cache_gb`:

```python
def _int8_prefill(model: dict) -> bool:
    """[model].int8_prefill: true or false; anything else warns and means false,
    the same stance as the other [model] knobs (a typo must not turn on a path
    that changes numerics)."""
    value = model.get("int8_prefill", False)
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"sous config: [model].int8_prefill {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False
```

In `load_config`, after `speculative_block_size=_speculative_block_size(model),`:

```python
        int8_prefill=_int8_prefill(model),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/sous/config.py tests/test_config.py
git commit -m "feat(config): [model].int8_prefill, default off

A boolean that opts the loaded model into the int8-activation prefill
kernels; off by default because int8 activations change numerics and the
tool-loop A/B that would justify default-on has not run yet.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Engine plumbing and status

**Files:**
- Modify: `src/sous/engine/base.py:200-210` (`_default_factory` signature), `:238-268` (engine construction), `:269-330` (`ManagedEngine`), `:452-476` (`EngineManager.__init__`), `:528-539` (`status()`)
- Modify: `src/sous/engine/vlm.py:50-70` (`VLMEngine.__init__`), `src/sous/engine/lm.py:14-33` (`LMEngine.__init__`)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: `int8prefill.enable(model, *, enabled) -> dict` (Task 3), `SousConfig.int8_prefill` (Task 4).
- Produces: `_default_factory(..., int8_prefill: bool = False)`; `VLMEngine(..., int8_prefill: bool = False)` and `LMEngine(..., int8_prefill: bool = False)`, each with `self.int8_prefill_status: dict[str, Any]`; `ManagedEngine.int8_prefill_status -> dict | None` (property); `EngineManager.status()["int8_prefill"]` when the engine reports one.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py` after `test_engine_manager_threads_drafter_config_into_default_factory` (~line 466):

```python
# ---- int8 prefill plumbing --------------------------------------------------------


def test_engine_manager_threads_int8_prefill_into_default_factory(monkeypatch):
    import sous.engine.base as base

    seen: dict[str, dict] = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen["kwargs"] = kwargs
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    EngineManager(SousConfig(int8_prefill=True)).get()
    assert seen["kwargs"]["int8_prefill"] is True


def test_default_factory_passes_int8_prefill_to_both_engines(monkeypatch):
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
    monkeypatch.setattr("sous.context.kv_bytes_per_token", lambda cfg: 1024)
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, int8_prefill=True)
    assert seen["vlm"]["int8_prefill"] is True
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3_5"})
    base._default_factory("m", 0.7, 0.8, 20, True, cache_budget=0, int8_prefill=True)
    assert seen["lm"]["int8_prefill"] is True


def test_status_carries_the_int8_prefill_view_when_the_engine_reports_one(tmp_path):
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "int8_prefill" not in manager.status(), "fakes without the attribute stay silent"
    inner.int8_prefill_status = {"state": "active", "reason": None, "routed": 336}
    assert manager.status()["int8_prefill"] == {"state": "active", "reason": None, "routed": 336}


def test_vlm_engine_enables_int8_prefill_on_the_loaded_model(monkeypatch):
    from sous.engine import int8prefill
    from sous.engine.vlm import VLMEngine

    model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="fake"))
    processor = types.SimpleNamespace(tokenizer=_RecordingTokenizer())
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)
    seen: dict[str, object] = {}

    def fake_enable(m, *, enabled):
        seen["model"], seen["enabled"] = m, enabled
        return {"state": "off", "reason": None, "routed": 0}

    monkeypatch.setattr(int8prefill, "enable", fake_enable)
    engine = VLMEngine("test/model", cache_budget=0, int8_prefill=True)
    assert seen == {"model": model, "enabled": True}
    assert engine.int8_prefill_status == {"state": "off", "reason": None, "routed": 0}


def test_lm_engine_enables_int8_prefill_on_the_loaded_model(monkeypatch):
    from sous.engine import int8prefill
    from sous.engine.lm import LMEngine

    model = object()
    _stub(monkeypatch, "mlx_lm", load=lambda model_id: (model, _RecordingTokenizer()))
    _stub(monkeypatch, "mlx_lm.sample_utils", make_sampler=lambda **kw: None)
    seen: dict[str, object] = {}

    def fake_enable(m, *, enabled):
        seen["model"], seen["enabled"] = m, enabled
        return {"state": "unavailable", "reason": "test", "routed": 0}

    monkeypatch.setattr(int8prefill, "enable", fake_enable)
    engine = LMEngine("test/model", cache_budget=0, int8_prefill=False)
    assert seen == {"model": model, "enabled": False}
    assert engine.int8_prefill_status["state"] == "unavailable"
```

Check how the existing `_hammer_ids` VLM test (~line 575) stubs `mlx_vlm.utils` and whether `PrefixCache(...)` construction in the engine needs anything else stubbed with `cache_budget=0`; mirror exactly what that test stubs.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -q -k int8`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'int8_prefill'` and `KeyError: 'int8_prefill'`.

- [ ] **Step 3: Plumb the flag**

`src/sous/engine/base.py`, `_default_factory` signature — add after `reserve_tokens: int = 0,`:

```python
    int8_prefill: bool = False,
```

and pass `int8_prefill=int8_prefill,` to both `vlm.VLMEngine(...)` and `lm.LMEngine(...)` calls (the LM call's comment about drafter settings stopping at the VLM stays true; this flag is for both backends).

`EngineManager.__init__` lambda — add after the `reserve_tokens=...` argument:

```python
                int8_prefill=config.int8_prefill,
```

`ManagedEngine` — add after the `model_id` property:

```python
    @property
    def int8_prefill_status(self) -> dict | None:
        # Optional on purpose: fakes and older engines have no such attribute.
        return getattr(self._inner, "int8_prefill_status", None)
```

`EngineManager.status()` — inside `if self._engine is not None:` after the `prompt_cache` line:

```python
                int8 = self._engine.int8_prefill_status
                if int8 is not None:
                    out["int8_prefill"] = dict(int8)
```

`src/sous/engine/vlm.py`, `VLMEngine.__init__` — add parameter `int8_prefill: bool = False,` after `reserve_bytes: int = 0,`; import `from sous.engine import int8prefill` next to the existing `from sous.engine.base import measure_cache_budget`; and right after `self._model, self._processor = load(model_id)`:

```python
        # Before the drafter loads (so it is never tagged) and before the cache
        # budget is measured (so warm-up temporaries are already released).
        self.int8_prefill_status = int8prefill.enable(self._model, enabled=int8_prefill)
```

`src/sous/engine/lm.py`, `LMEngine.__init__` — same parameter, same import, and after the `load(model_id)` line:

```python
        self.int8_prefill_status = int8prefill.enable(self._model, enabled=int8_prefill)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, format, type-check, full run**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/base.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_engine_base.py
git commit -m "feat(engine): enable int8 prefill at model load and surface its status

Both engines opt the freshly loaded model in (before the drafter, before
the cache-budget measurement) and keep the status dict; the manager copies
it into status() so sous status and server_status show state and count.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md:294-313` (the `[model]` TOML block), `README.md:~400` (after the `prompt_cache_gb` paragraph), `CLAUDE.md:19-` (Gotchas)

- [ ] **Step 1: README config block**

In the `[model]` block after `speculative_block_size = 3`:

```toml
# INT8-activation prefill on the M5 tensor units (M5-family or newer, macOS 26.2+;
# silently unavailable elsewhere). ~1.5x prefill on the default model, but int8
# activations change numerics — measured drift is inside what 4-bit weights
# already add — so it stays off until a tool-loop A/B says otherwise.
int8_prefill = false
```

- [ ] **Step 2: README prose**

After the `[model].prompt_cache_gb` paragraph (the one ending with the fork/budget explanation, ~line 400), add:

```markdown
`[model].int8_prefill` (default `false`) runs the prefill matmuls of the MLP and
linear-attention projections as INT8 activations against the checkpoint's packed
4-bit weights on the M5 GPU's neural accelerators (Apache-2.0 kernel derived from
oMLX, compiled at model load — no build step). Measured on an M5 Pro with the
default model: 484 → 754 tok/s at 4K tokens, 386 → 580 tok/s at 32K. Decode and
speculative verify are untouched. It changes prefill numerics (KL 0.033 vs the
stock path on a code prompt; 4-bit weights alone are 0.052 vs 8-bit), which is
why it ships off. Needs an M5-family or newer GPU and macOS 26.2+; anywhere else
`sous status` reports `int8_prefill: unavailable` with the reason and prefill
runs stock. Only affine 4-bit, group-size-64 checkpoints route (the default
model does); mxfp4/mxfp8 ones fall through silently.
```

- [ ] **Step 3: CLAUDE.md gotchas**

Append two bullets to `## Gotchas`:

```markdown
- `engine/int8prefill.py` is a runtime-compiled Metal kernel pair (Apache-2.0,
  derived from oMLX — see THIRD_PARTY_NOTICES.md). The Python `reorder_k` and
  the GEMM's nibble decode are two halves of one K-order contract (slot
  `16c+4t+j` holds code `16c+8*(t>>1)+2j+(t&1)`): never change one alone; the
  power-of-two bit-exactness test is what catches a mismatch. Only rows >= 128
  route (decode/verify never do), only tagged modules route (tags are set with
  `object.__setattr__` so they stay out of mlx's parameter tree), and only the
  GEMM's header includes the Metal-4 `MetalPerformancePrimitives` header, which
  needs macOS 26.2+ — the Stage-A kernel must keep compiling on any Metal GPU
  so CI (macos-15, no tensor units) can test it.
- `int8prefill.enable()` never raises and a model load never fails because of
  it: every failure is one `warnings.warn` plus `state: unavailable`. Tests
  that need the GEMM without tensor units monkeypatch `int8prefill.qmm`.
```

- [ ] **Step 4: Check formatting and commit**

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: clean (Markdown is untouched by ruff; this confirms nothing else drifted).

```bash
git add README.md CLAUDE.md
git commit -m "docs: document [model].int8_prefill and the kernel's invariants

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Runtime gate (local, `model`-marked) and the manual soak

**Files:**
- Create: `tests/test_int8prefill_model.py`

**Interfaces:**
- Consumes: `VLMEngine(model_id, cache_budget=0, int8_prefill=...)`, `engine.new_cache()`, `engine.prefill(cache, ids)`, `engine._encode(text)`, `engine._loaded()`, `engine.int8_prefill_status`, `engine.unload()` (existing engine API plus Task 5); `ManagedEngine`, `GenerationSession` (`sous.engine.base`).

- [ ] **Step 1: Write the gate tests**

Create `tests/test_int8prefill_model.py`:

```python
"""Runtime gate for int8 prefill: real weights, real tensor units, local only.

Acceptance from the spec: routed count 336 on the default model, >= 1.4x prefill
at 4,096 tokens and >= 1.3x at 32,768, KL(stock || int8) <= 0.06 nats on the
standard prompt, and a generation session that exits cleanly with the kernels on.
Takes ~6 minutes on the M5 Pro (two 27B loads, two 32K prefills).
"""

import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill as i8  # noqa: E402

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not i8.availability().available, reason=f"no tensor units: {i8.availability().reason}"
    ),
]

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
HYBRID_VLM = "mlx-community/Qwen3.5-9B-MLX-4bit"
ROOT = Path(__file__).resolve().parents[1]


def _standard_text() -> str:
    return (ROOT / "src/sous/engine/promptcache.py").read_text() + (
        ROOT / "src/sous/gateway/turn.py"
    ).read_text()


def _repeat_to(ids: list[int], n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


def _prefill_seconds(engine, ids: list[int]) -> float:
    best = float("inf")
    for _ in range(2):
        cache = engine.new_cache()
        mx.synchronize()
        t = time.perf_counter()
        engine.prefill(cache, ids)
        mx.synchronize()
        best = min(best, time.perf_counter() - t)
        del cache
        mx.clear_cache()
    return best


def _last_logits(engine, ids: list[int], positions: int = 256):
    model, _ = engine._loaded()
    cache = engine.new_cache()
    out = model.language_model(mx.array(ids)[None], cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    logits = logits[0, -positions:].astype(mx.float32)
    mx.eval(logits)
    return logits


def _kl(p_logits, q_logits) -> float:
    lp = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    lq = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
    return mx.mean(mx.sum(mx.exp(lp) * (lp - lq), axis=-1)).item()


def test_default_model_meets_the_acceptance_bars():
    from sous.engine.vlm import VLMEngine

    stock = VLMEngine(DEFAULT_27B, cache_budget=0)
    assert stock.int8_prefill_status["state"] == "off"
    base = stock._encode(_standard_text())
    ids_4k, ids_32k, ids_2k = _repeat_to(base, 4096), _repeat_to(base, 32768), base[:2048]
    stock_4k = _prefill_seconds(stock, ids_4k)
    stock_32k = _prefill_seconds(stock, ids_32k)
    stock_logits = _last_logits(stock, ids_2k)
    stock.unload()
    del stock
    mx.clear_cache()

    fast = VLMEngine(DEFAULT_27B, cache_budget=0, int8_prefill=True)
    assert fast.int8_prefill_status == {"state": "active", "reason": None, "routed": 336}
    fast_4k = _prefill_seconds(fast, ids_4k)
    fast_32k = _prefill_seconds(fast, ids_32k)
    fast_logits = _last_logits(fast, ids_2k)
    kl = _kl(stock_logits, fast_logits)
    fast.unload()

    print(
        f"\nprefill 4096: {4096 / stock_4k:.0f} -> {4096 / fast_4k:.0f} tok/s ({stock_4k / fast_4k:.2f}x)"
        f"\nprefill 32768: {32768 / stock_32k:.0f} -> {32768 / fast_32k:.0f} tok/s ({stock_32k / fast_32k:.2f}x)"
        f"\nKL(stock || int8) over 256 positions: {kl:.4f} nats"
    )
    assert stock_4k / fast_4k >= 1.4
    assert stock_32k / fast_32k >= 1.3
    assert kl <= 0.06


def test_hybrid_9b_routes_its_real_gdn_and_mlp_classes_and_still_generates():
    """The CI routing tests use a stand-in GDN; this is the real
    Qwen3_5GatedDeltaNet / Qwen3_5MLP pair from mlx-vlm on a smaller affine-Q4
    checkpoint, through a ManagedEngine session so the generation thread exits
    with the kernels loaded (the mlx thread-teardown rule)."""
    from sous.engine.base import ManagedEngine
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    engine = VLMEngine(HYBRID_VLM, cache_budget=0, int8_prefill=True)
    assert engine.int8_prefill_status["state"] == "active"
    assert engine.int8_prefill_status["routed"] > 0
    managed = ManagedEngine(engine)
    prompt = "Summarize this in one sentence:\n" + _standard_text()[:6000]
    msgs = [{"role": "user", "content": prompt}]
    session = managed.session()
    try:
        out = session.generate(msgs, WORKER_TOOLS, max_tokens=32)
    finally:
        session.close()
        assert session.join(timeout=30)
    assert isinstance(out, str) and out.strip()
    engine.unload()
```

Check `GenerationSession.generate`'s exact signature in `src/sous/engine/base.py:409` and `ManagedEngine.session()` at `:319` before running; adjust the call (it may take `on_delta`) — the point is that a generation runs on a session thread that is then closed and joined.

- [ ] **Step 2: Run the gate**

Run: `uv run pytest tests/test_int8prefill_model.py -m model -q -s`
Expected: 2 passed, with the printed throughput and KL lines. Record them in the PR description. If the 4K bar misses narrowly (1.35–1.4x), rerun once with nothing else on the GPU (the daemon idle-unloaded: `sous status` shows no loaded model) before concluding; if it still misses, the routing set or the layout decision in Task 2 is the suspect — do not lower the bar.

- [ ] **Step 3: Full CI-equivalent run and commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q && uv lock --check`
Expected: clean.

```bash
git add tests/test_int8prefill_model.py
git commit -m "test(engine): runtime gate for int8 prefill on the default model

Speed, numerics and routed-count bars from the spec, plus the real
Qwen3.5 GDN/MLP classes on the 9B hybrid through a session that exits.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual soak on the real daemon (only with Kyle's go-ahead — touches `~/.sous`)**

1. Add `int8_prefill = true` under `[model]` in `~/.sous/config.toml`.
2. Restart the unmanaged daemon the way the memory note prescribes (fork + `setsid`; a plain `nohup` dies with the shell).
3. Delegate one small task through Claude Code (e.g. "add a `mul()` function with a test to `mathx.py`" in a scratch project) and let it finish.
4. `sous status` must show `int8_prefill: state=active routed=336`; `~/.sous/daemon.err.log` must show `int8 prefill: routed 336 projections` and no segfault; the task's result must be `done`.
5. Revert the config line (the feature ships default-off) and restart the daemon again.

- [ ] **Step 5: Open the PR**

Use the `write-pr` skill. The description leads with the measured numbers from Step 2, links the spec and this plan, states that the default is off and why (numerics; A/B gate follow-up spec), links #76 (attention routing) and #77 (baseline profiling), and carries the `🤖 Generated with [Claude Code](https://claude.com/claude-code)` footer with no session URL.

---

## Self-review notes

- **Spec coverage.** Stage A (Task 1); GEMM, layout rule, availability, notice (Task 2); eligibility, tagging rules, wrappers, shared stages, `enable()` states, warm-up, logging (Task 3); config (Task 4); engine hook order, status field (Task 5); README/CLAUDE.md (Task 6); acceptance bars, real GDN classes, thread-exit soak (Task 7). The spec's "byte-identical when off" is covered by `test_enable_off_touches_nothing` and the `off` state short-circuiting before any import of the Metal side.
- **Type consistency.** `enable(model, *, enabled: bool) -> dict[str, Any]` everywhere; `linear_int8(linear, x, stage=None)` positional `stage` in the wrappers and the warm-up fake; `Availability(available, reason=None)`; `_TAG`/`_STAGE` string constants used by tests via `i8._TAG`; `int8_prefill_status` on both engines, the `ManagedEngine` property and `FakeEngine` (set ad hoc in the status test — `FakeEngine` itself gains no attribute).
- **Known follow-ups, not in this plan:** the A/B harness spec (gates default-on), #76, #77.
