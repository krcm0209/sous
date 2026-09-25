# Exact Grouped Verify Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mlx-vlm's per-row SDPA loop in the speculative verifier with grouped stock `mx.fast.scaled_dot_product_attention` calls. The result must be bit-identical, proven at load and fall back to the stock loop whenever the proof fails, and it lifts decode by about 10% at subagent context.

**Architecture:** A new `src/sous/engine/verifyattn.py`, shaped like `int8prefill.py`.
- `plan()` mirrors mlx 0.32.2's SDPA dispatch, and `groups()` cuts verify rows into runs that share a plan.
- A class-level wrap of `Qwen3_5BatchInvariantForward._attention` acts only on tagged modules.
- `enable()` runs four guards before it tags anything: the mlx version, the mlx-vlm source hashes, a load-time probe, and the per-call scope check.
- `VLMEngine` calls `enable()` only when a drafter loaded. `base.py` surfaces the result on the model-load line and in the status document.

**Tech Stack:** Python 3.14, mlx 0.32.2, mlx-vlm 0.7.1 (the lock; the M5 daemon's tool env runs 0.7.2, whose relevant source is byte-identical), pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-24-exact-verify-attention-design.md`. Read it before starting any task.

## Global Constraints

- Scope is exactly **3 ≤ T ≤ 8**. T = 2 always takes mlx-vlm's original path, since stock already makes one call there.
- Accept only mask `== "causal"` (the string), and only when `type(cache) is KVCache` (exact type, not `isinstance`).
- `VALIDATED_MLX = frozenset({"0.32.2"})`. `MLX_SDPA_BLOCKS` set to anything but empty or `0` makes the path unavailable.
- Source hashes are sha256 of `inspect.getsource(obj).encode("utf-8")`, and the validated values are fixed (identical in mlx-vlm 0.7.1 and 0.7.2):
  - `Qwen3_5BatchInvariantForward._attention` 7790ae37e7a0d78d2287217a3050e03a4b9b7c10eb1e60839e4315226186cdf1
  - `Qwen3_5Attention._prepare_projected_qkv` da21fd76217c72aa7ef15d4097e1fe428ff0022259f9556eb8fd95b59217c109
  - `_create_qwen3_5_attention_mask` a74a1135eb644698e1beb6e43a5d51e27e6490d0b0f4f6d951d08dd5e908c681
  - `_qwen3_5_left_padded_attention` e5d9f69065da9885674a6c3e11a65807948c5e9f9d7a1fef6211628328e7b312
  - `_qwen3_5_left_padding_info` 1cea363e3150f9262f7fb23d11d525b6be32625d3407cd324850d82d6a151fd5
  - `KVCache.update_and_fetch` 9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b
  - `KVCache.make_mask` d65ba0d83b52a93bb00faae6bcbac8add067724ebe9dd8ba6aaa716f81b6b620
- Only head dim 256 and `model_type == "qwen3_5"` are supported.
- `enable()` never raises and a model load never fails because of it. Every failure is one `warnings.warn` (text starting `"sous: "`, `stacklevel=3`) plus `state: unavailable`. The status dict is `{"state": "off"|"unavailable"|"active", "reason": str|None, "probe_seconds": float|None}`.
- mlx and mlx_vlm imports stay **function-local** in `src/` (the lint job runs on ubuntu without mlx). Tests use `mx = pytest.importorskip("mlx.core")` at the top, with `# noqa: E402` on the deferred imports.
- Leave these alone:
  - the fork key (`_fork_key_fields`);
  - `forkstore._EPOCH_FILES`: do not add `verifyattn.py`;
  - `engine/kernels/`, the config, the LM backend, the TUI;
  - `SPECULATIVE_BLOCK_MAX`.
- `vlm.py` *is* in `_EPOCH_FILES`, so this branch invalidates every on-disk fork once after deploy (one cold start). Say so in the PR.
- Type-suppression pragmas are `# ty: ignore[<rule>]` with the rule ty actually reports. Never write `# type: ignore`.
- Comments explain non-obvious *why*. Never cite the spec, the plan, or task and step numbers in code or test comments; state the fact instead.
- Every command that touches the GPU on the M2 dev machine runs under the lock when another session may be using it. A plain `uv run pytest ...` is fine when nothing else is running.
- Commits use Conventional Commits with an imperative lowercase subject and the *why* in the body, ending with `Co-Authored-By: Claude <noreply@anthropic.com>`. No model name.
- Before every commit run `uv run ruff format . && uv run ruff check . && uv run ty check`, and fix what they report.

---

## File Structure

- **Create `src/sous/engine/verifyattn.py`.** The whole feature: plan mirror, grouping, probe, gates, hook, `enable()`. About 330 lines, the same shape as `int8prefill.py`.
- **Create `tests/test_verifyattn.py`.** Plan and grouping tables, probe, gates, scope predicate, the tiny-model contract test, `enable()`, and slow subprocess tests under `MLX_METAL_GPU_ARCH` overrides.
- **Create `tests/test_verifyattn_model.py`.** The `-m model` real-27B bit-equality test (manual, M5 Pro).
- **Modify `src/sous/engine/vlm.py`.** Call `verifyattn.enable()` after the drafter loads.
- **Modify `src/sous/engine/base.py`.** A `ManagedEngine.verify_attention_status` property, the load-line tokens, and a `status()` block.
- **Modify `src/sous/tune/suite/runner.py:120-121`.** The pass-through comment lists attribute names; add the new one.
- **Modify `tests/test_engine_base.py`.** Tests for the engine wiring, the load line and the status document.
- **Modify `tests/test_server.py`.** The `svc` fixture releases mlx thread state on pytest's main thread, so any real-mlx test that runs after it fails (an existing bug; see Task 2).
- **Modify `CLAUDE.md`, `README.md` and the spec.** A gotcha bullet, the load-line description, and two measured numbers.

---

### Task 1: The plan mirror, grouping and probe cases

**Files:**
- Create: `src/sous/engine/verifyattn.py`
- Test: `tests/test_verifyattn.py`

**Interfaces:**
- Produces:
  - `plan(arch: str, n_keys: int, gqa: int, q_len: int) -> tuple[str, int]`, returning `("fallback", 0)`, `("1pass", 0)` or `("2pass", blocks)`.
  - `groups(arch: str, prefix: int, t: int, gqa: int) -> list[tuple[int, int]]`.
  - `plan_transitions(arch: str, gqa: int) -> tuple[int, ...]`.
  - `probe_cases(arch: str, gqa: int) -> tuple[tuple[int, int], ...]` of `(prefix, T)` pairs.
  - Constants `HEAD_DIM = 256`, `MIN_ROWS = 3`, `MAX_ROWS = 8`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verifyattn.py`:

```python
"""Exact grouped verify attention (sous.engine.verifyattn).

The plan and grouping tables are pure Python. The GPU tests run on any Metal
GPU (CI is macos-15), and the slow tests repeat them in subprocesses with
MLX_METAL_GPU_ARCH set, so the 's' and 'd' dispatch tables the M5 Pro takes are
proven bit-exact on whatever GPU runs the suite.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import verifyattn  # noqa: E402 — after the importorskip guard

S, D, G = "applegpu_g13s", "applegpu_g13d", "applegpu_g14g"


@pytest.mark.parametrize(
    ("arch", "n_keys", "gqa", "q_len", "expected"),
    [
        # the vector path's limits
        (S, 5000, 6, 9, ("fallback", 0)),
        (S, 5000, 6, 6, ("fallback", 0)),  # 6 x 6 > 32
        (S, 2, 6, 3, ("fallback", 0)),  # more queries than keys
        # 's': two passes from 1024 keys; more blocks only past 1024 and n_simds > 4
        (S, 1023, 6, 1, ("1pass", 0)),
        (S, 1024, 6, 1, ("2pass", 64)),
        (S, 1025, 6, 1, ("2pass", 128)),
        (S, 8192, 6, 1, ("2pass", 128)),
        (S, 8193, 6, 1, ("2pass", 256)),
        (S, 32768, 6, 1, ("2pass", 256)),
        (S, 32769, 6, 1, ("2pass", 512)),
        (S, 65536, 6, 1, ("2pass", 512)),
        (S, 65537, 6, 1, ("2pass", 1024)),
        (S, 5000, 4, 1, ("2pass", 64)),  # n_simds 4 is not > 4
        (S, 5000, 4, 2, ("2pass", 128)),
        # 'd'
        (D, 1023, 6, 1, ("1pass", 0)),
        (D, 1024, 6, 1, ("2pass", 128)),
        (D, 16383, 6, 1, ("2pass", 128)),
        (D, 16384, 6, 1, ("2pass", 512)),
        (D, 65535, 6, 1, ("2pass", 512)),
        (D, 65536, 6, 1, ("2pass", 1024)),
        (D, 8193, 2, 1, ("2pass", 256)),  # n_simds <= 2 past 8192
        (D, 20000, 4, 1, ("2pass", 128)),  # n_simds 4: neither branch
        # every other suffix: two passes only with GQA from 4096 keys
        (G, 4095, 6, 1, ("1pass", 0)),
        (G, 4096, 6, 1, ("2pass", 64)),
        (G, 4096, 2, 1, ("2pass", 32)),
        (G, 100000, 1, 1, ("1pass", 0)),
    ],
)
def test_plan_mirrors_mlx_0_32_2_dispatch(arch, n_keys, gqa, q_len, expected):
    assert verifyattn.plan(arch, n_keys, gqa, q_len) == expected


@pytest.mark.parametrize(
    ("arch", "prefix", "t", "gqa", "expected"),
    [
        (G, 4093, 3, 6, [(0, 2), (2, 3)]),  # rows at 4094, 4095 | 4096
        (G, 4093, 8, 6, [(0, 2), (2, 7), (7, 8)]),  # past 5 rows the call falls back
        (S, 1021, 4, 6, [(0, 2), (2, 3), (3, 4)]),  # 1024 and 1025 are two transitions
        (S, 8190, 8, 6, [(0, 2), (2, 7), (7, 8)]),
        (S, 32766, 3, 6, [(0, 2), (2, 3)]),
        (S, 57000, 8, 6, [(0, 5), (5, 8)]),
        (S, 57000, 1, 6, [(0, 1)]),
    ],
)
def test_groups_split_at_every_plan_transition(arch, prefix, t, gqa, expected):
    assert verifyattn.groups(arch, prefix, t, gqa) == expected


@pytest.mark.parametrize(
    ("arch", "gqa", "expected"),
    [
        (S, 6, (1024, 1025, 8193, 32769, 65537)),
        (D, 6, (1024, 16384, 65536)),
        (G, 6, (4096,)),
        (G, 4, (4096,)),
    ],
)
def test_plan_transitions_are_the_dispatch_thresholds(arch, gqa, expected):
    assert verifyattn.plan_transitions(arch, gqa) == expected


def test_probe_cases_straddle_every_transition_and_stop_past_the_last():
    cases = verifyattn.probe_cases(S, 6)
    assert max(p + t for p, t in cases) == 65537 + 7
    for n in verifyattn.plan_transitions(S, 6):
        for t in range(verifyattn.MIN_ROWS, verifyattn.MAX_ROWS + 1):
            # some row sits at n - 1 and the next at n
            assert any(p + 1 <= n - 1 and n <= p + t for p, tt in cases if tt == t), (n, t)
    assert all(verifyattn.MIN_ROWS <= t <= verifyattn.MAX_ROWS for _, t in cases)
    assert max(p + t for p, t in verifyattn.probe_cases(G, 6)) == 4096 + 7
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: FAIL at collection with `ImportError: cannot import name 'verifyattn'`.

- [ ] **Step 3: Write the module with the pure functions**

Create `src/sous/engine/verifyattn.py`:

```python
"""Exact grouped attention for mlx-vlm's speculative verifier (qwen3_5).

The exact verifier (``Qwen3_5BatchInvariantForward._attention``) runs a T > 2
verify forward's attention as T single-query SDPA calls, so the whole KV prefix
is streamed once per row. Here rows are grouped into one stock
``mx.fast.scaled_dot_product_attention`` call wherever mlx gives the group the
kernel plan each row would get alone. Inside one plan, mlx assigns key i to
simdgroup ``i % 32`` (one pass) or block ``i % blocks`` (two passes) whatever
the key count, and the causal mask skips excluded keys. So every row stays
bit-identical to the loop and to the M=1 decode.

``plan()`` mirrors mlx 0.32.2's dispatch (backend/metal/
scaled_dot_product_attention.cpp), which sous cannot inspect at runtime. So four
guards stand between it and a served turn:
- the mlx version;
- the source of every mlx-vlm function the hook reads;
- a load-time probe on this GPU;
- a per-call scope check, decided before the projections run. The projections
  append the verify rows to the KV cache, so the original method cannot be
  re-entered after them.

T = 2 is left alone. The stock verifier already makes one call there, and
grouping it would change output where rows straddle a plan transition.
"""

from __future__ import annotations

import functools

HEAD_DIM = 256
MIN_ROWS = 3
MAX_ROWS = 8
# Past mlx 0.32.2's last SDPA dispatch threshold (65536 keys).
_SCAN_TO = 70_000


def plan(arch: str, n_keys: int, gqa: int, q_len: int) -> tuple[str, int]:
    """The kernel mlx 0.32.2 picks for one head-dim-256 SDPA call:
    ("fallback", 0), ("1pass", 0) or ("2pass", blocks). Mirrors
    has_fused_kernel/use_fallback and sdpa_vector_2pass's block table."""
    if q_len > 8 or q_len > n_keys or q_len * gqa > 32:
        return ("fallback", 0)
    suffix = arch[-1:]
    if not ((suffix in ("d", "s") and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096)):
        return ("1pass", 0)
    n_simds = gqa * q_len
    if suffix == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
    elif suffix == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return ("2pass", blocks)


def groups(arch: str, prefix: int, t: int, gqa: int) -> list[tuple[int, int]]:
    """Contiguous row runs [j, k) whose one causal call gets every member row's
    singleton plan; a single row is always exact on its own."""
    out: list[tuple[int, int]] = []
    j = 0
    while j < t:
        k = j + 1
        while k < t:
            call = plan(arch, prefix + k + 1, gqa, k + 1 - j)
            if call[0] == "fallback" or any(
                plan(arch, prefix + r + 1, gqa, 1) != call for r in range(j, k + 1)
            ):
                break
            k += 1
        out.append((j, k))
        j = k
    return out


@functools.cache
def plan_transitions(arch: str, gqa: int) -> tuple[int, ...]:
    """Every key count N at which some verify group's plan differs from N - 1's.
    The scan starts past MAX_ROWS keys: below that the q_len > n_keys fallback
    moves with N, and no verify reaches it."""
    qs = range(1, min(MAX_ROWS, 32 // gqa) + 1)
    prev = [plan(arch, MAX_ROWS + 1, gqa, q) for q in qs]
    out: list[int] = []
    for n in range(MAX_ROWS + 2, _SCAN_TO):
        cur = [plan(arch, n, gqa, q) for q in qs]
        if cur != prev:
            out.append(n)
        prev = cur
    return tuple(out)


@functools.cache
def probe_cases(arch: str, gqa: int) -> tuple[tuple[int, int], ...]:
    """(prefix, T) pairs that put verify rows on both sides of every plan
    transition, plus one interior prefix between neighbours. Past the last
    transition nothing changes, so nothing beyond it needs proving."""
    marks = plan_transitions(arch, gqa)
    cases: set[tuple[int, int]] = set()
    for n in marks:
        for t in range(MIN_ROWS, MAX_ROWS + 1):
            cases.update((p, t) for p in (n - t, n - t // 2 - 1, n - 1))
    edges = (0, *marks) if marks else (0, 1024)
    for lo, hi in zip(edges, edges[1:], strict=False):
        cases.update(((lo + hi) // 2, t) for t in range(MIN_ROWS, MAX_ROWS + 1))
    return tuple(sorted(cases))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: all tests pass. `plan_transitions` scans 70,000 key counts once per `(arch, gqa)` and is cached. If the run takes more than about a second, note it for Task 5's probe budget.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/verifyattn.py tests/test_verifyattn.py
git commit -m "feat(engine): mirror mlx's SDPA plan to group verify rows exactly" -m "mlx-vlm's exact verifier streams the KV prefix once per verify row. Rows
that share mlx's kernel plan can share one stock SDPA call bit-exactly;
this is the pure plan mirror and grouping those calls will use.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Grouped attention and the load-time probe

**Files:**
- Modify: `src/sous/engine/verifyattn.py`
- Modify: `tests/test_server.py:46-52` (the `svc` fixture)
- Test: `tests/test_verifyattn.py`

**Interfaces:**
- Consumes: `groups`, `probe_cases`, `HEAD_DIM`, `MAX_ROWS` from Task 1.
- Produces:
  - `grouped_attention(queries, keys, values, scale: float, prefix: int, arch: str) -> mx.array`. Queries are `[1, Hq, T, D]`; keys and values are `[1, Hkv, prefix + T, D]`.
  - `_row_loop(queries, keys, values, cache, scale: float) -> mx.array`: the stock verifier's T > 2 loop.
  - `probe(arch: str, q_heads: int, kv_heads: int, dtype) -> str | None`: `None` when exact, otherwise the reason.

- [ ] **Step 0: Stop the status-document tests from releasing the main thread's mlx state**

`Daemon.status_document()` ends with `release_mlx_thread_state()`, as the pool thread that serves it must. Three tests in `tests/test_server.py` call it on pytest's main thread through the `svc` fixture:
- `test_the_memory_read_is_the_last_thing_the_document_does`
- `test_server_status_reports_the_served_config`
- `test_the_status_document_carries_no_task_fields`

After that call, every later real-mlx op on that thread fails with `RuntimeError: There is no Stream(gpu, N) in current thread`. `tests/test_verifyattn.py` is the first real-mlx file that sorts after `test_server.py`, so the full suite breaks. This was reproduced on a clean copy of `main` with a one-line mlx test sorted last.

Replace the fixture header in `tests/test_server.py`:

```python
@pytest.fixture()
def svc(tmp_path: Path, monkeypatch):
    # status_document() releases the calling thread's mlx state, as the pool
    # thread that serves it must; on pytest's main thread that release leaves
    # every later real-mlx test in the session without a stream.
    from sous import server

    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: None)
    root = tmp_path / "proj"
```

Keep the rest of the fixture body as it is. The two tests that assert the release (around `:324` and `:338`) patch it again themselves, so they still pass.

Run: `uv run pytest tests/test_server.py`
Expected: all pass.

- [ ] **Step 1: Write the failing tests**

Add `import os`, `import subprocess`, `import sys` and `from pathlib import Path` to the stdlib block at the top of `tests/test_verifyattn.py`, above `import pytest`. Add `ROOT = Path(__file__).resolve().parents[1]` next to the `S, D, G` constants. Then append:

```python


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(24, 4), (16, 4)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_probe_proves_grouping_exact_on_this_gpu(q_heads, kv_heads, dtype):
    arch = mx.device_info()["architecture"]
    assert verifyattn.probe(arch, q_heads, kv_heads, getattr(mx, dtype)) is None


def test_probe_catches_a_grouping_that_ignores_the_plan(monkeypatch):
    # One call over as many rows as the vector path takes, across transitions:
    # the probe must see the rows whose plan it changed.
    monkeypatch.setattr(
        verifyattn,
        "groups",
        lambda arch, prefix, t, gqa: (
            [(0, min(t, 32 // gqa))] + [(r, r + 1) for r in range(min(t, 32 // gqa), t)]
        ),
    )
    arch = mx.device_info()["architecture"]
    reason = verifyattn.probe(arch, 24, 4, mx.bfloat16)
    assert reason is not None and reason.startswith("grouped attention differs")


@pytest.mark.slow
@pytest.mark.parametrize("arch", [S, D])
def test_exactness_holds_under_every_dispatch_table(arch):
    """mlx reads MLX_METAL_GPU_ARCH once, when it builds the device, and routes
    SDPA by it; a child process with it set runs the other suffixes' tables on
    this GPU."""
    env = dict(os.environ, MLX_METAL_GPU_ARCH=arch)
    tests = [
        "tests/test_verifyattn.py::test_probe_proves_grouping_exact_on_this_gpu",
        "tests/test_verifyattn.py::test_probe_catches_a_grouping_that_ignores_the_plan",
    ]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "passed" in done.stdout and "skipped" not in done.stdout
```

Also add this sanity test, so the override test cannot pass vacuously. It spawns a process, so it is `slow` too:

```python
@pytest.mark.slow
def test_the_arch_override_reaches_mlx():
    code = "import mlx.core as mx; print(mx.device_info()['architecture'])"
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, MLX_METAL_GPU_ARCH=S),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.stdout.strip() == S, done.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py -k "probe or override"`
Expected: FAIL with `AttributeError: module 'sous.engine.verifyattn' has no attribute 'probe'`. The override sanity test should already pass.

- [ ] **Step 3: Implement grouped attention, the row loop and the probe**

Add `from typing import Any` to the module's imports, then add below `probe_cases`:

```python
def grouped_attention(
    queries: Any, keys: Any, values: Any, scale: float, prefix: int, arch: str
) -> Any:
    """One stock SDPA call per group of rows that share their singleton plan:
    causal for a group, unmasked for a lone row, which is the M=1 decode call."""
    import mlx.core as mx

    t = queries.shape[2]
    gqa = queries.shape[1] // keys.shape[1]
    parts = [
        mx.fast.scaled_dot_product_attention(
            queries[:, :, j:k, :],
            keys[:, :, : prefix + k, :],
            values[:, :, : prefix + k, :],
            scale=scale,
            mask="causal" if k - j > 1 else None,
        )
        for j, k in groups(arch, prefix, t, gqa)
    ]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def _row_loop(queries: Any, keys: Any, values: Any, cache: Any, scale: float) -> Any:
    """mlx-vlm's own T > 2 loop: one single-query call per row over its own
    prefix, through mlx-vlm's helpers so quantized KV tuples still work."""
    import mlx.core as mx
    from mlx_vlm.models.base import (
        kv_sequence_length,
        scaled_dot_product_attention,
        slice_kv_sequence,
    )

    t = queries.shape[2]
    prefix = kv_sequence_length(keys) - t
    return mx.concatenate(
        [
            scaled_dot_product_attention(
                queries[:, :, i : i + 1, :],
                slice_kv_sequence(keys, prefix + i + 1),
                slice_kv_sequence(values, prefix + i + 1),
                cache=cache,
                scale=scale,
                mask=None,
            )
            for i in range(t)
        ],
        axis=2,
    )


def probe(arch: str, q_heads: int, kv_heads: int, dtype: Any) -> str | None:
    """Prove grouped_attention bit-identical to the per-row loop on this GPU at
    every plan transition. Returns why not, or None. Inputs are laid out as
    serving lays them out: queries are a transposed [1, T, Hq, D] slice, K/V
    slices of a buffer grown in KVCache's 256-token steps."""
    import mlx.core as mx

    cases = probe_cases(arch, q_heads // kv_heads)
    capacity = -(-max(p + t for p, t in cases) // 256) * 256
    k_key, v_key, q_key = mx.random.split(mx.random.key(0), 3)
    keys = mx.random.normal((1, kv_heads, capacity, HEAD_DIM), key=k_key).astype(dtype)
    values = mx.random.normal((1, kv_heads, capacity, HEAD_DIM), key=v_key).astype(dtype)
    rows = mx.random.normal((1, MAX_ROWS, q_heads, HEAD_DIM), key=q_key).astype(dtype)
    scale = HEAD_DIM**-0.5
    try:
        for prefix, t in cases:
            queries = rows[:, :t].transpose(0, 2, 1, 3)
            k = keys[..., : prefix + t, :]
            v = values[..., : prefix + t, :]
            got = grouped_attention(queries, k, v, scale, prefix, arch)
            want = _row_loop(queries, k, v, None, scale)
            if not mx.array_equal(got, want).item():
                return f"grouped attention differs from the per-row loop at prefix {prefix}, T={t}"
    finally:
        del keys, values, rows
        mx.clear_cache()
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: all pass, including the `slow` subprocess cases. On an M2 the whole file takes about 17 s, 16 s of it in the slow subprocesses. The 's' and 'd' probes cost 2.5–2.9 s cold each, because they reach 65,544 keys.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/verifyattn.py tests/test_verifyattn.py tests/test_server.py
git commit -m "feat(engine): prove grouped verify attention exact at load" -m "Grouping is exact only while plan() mirrors mlx's dispatch on this GPU.
The probe checks every plan transition with serving's input layout, and
the slow tests re-run it under the 's' and 'd' dispatch tables so CI
proves the tables the M5 Pro takes.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: The mlx version and mlx-vlm source gates

**Files:**
- Modify: `src/sous/engine/verifyattn.py`
- Test: `tests/test_verifyattn.py`

**Interfaces:**
- Produces:
  - `VALIDATED_MLX: frozenset[str]`
  - `VALIDATED_MLX_VLM_SOURCES: dict[str, frozenset[str]]`
  - `_SOURCES: dict[str, str]`, mapping qualname to module
  - `_source_digest(module: str, qualname: str) -> str | None`
  - `_gate() -> str | None`: the first reason the path is unavailable, or None.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifyattn.py`:

```python
def test_the_locked_mlx_is_validated():
    """A dependency bump fails here until someone re-reads mlx's SDPA dispatch
    against plan() and extends VALIDATED_MLX."""
    assert mx.__version__ in verifyattn.VALIDATED_MLX


def test_the_locked_mlx_vlm_sources_are_validated():
    """A dependency bump that changes any function the hook reads fails here
    until someone re-reads it and adds its new hash."""
    for qualname, module in verifyattn._SOURCES.items():
        digest = verifyattn._source_digest(module, qualname)
        assert digest in verifyattn.VALIDATED_MLX_VLM_SOURCES[qualname], qualname
    assert verifyattn._gate() is None


def test_gate_refuses_an_unvalidated_mlx(monkeypatch):
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    assert verifyattn._gate() == "mlx 0.99.0 not validated"


def test_gate_refuses_a_forced_sdpa_block_count(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "128")
    assert verifyattn._gate() == "MLX_SDPA_BLOCKS is set"


def test_gate_ignores_a_zero_sdpa_block_count(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "0")
    assert verifyattn._gate() is None


def test_gate_refuses_a_changed_mlx_vlm_function(monkeypatch):
    monkeypatch.setitem(
        verifyattn.VALIDATED_MLX_VLM_SOURCES, "KVCache.make_mask", frozenset({"0" * 64})
    )
    assert verifyattn._gate() == "mlx-vlm KVCache.make_mask changed"


def test_an_unreadable_source_counts_as_changed():
    assert verifyattn._source_digest("mlx_vlm.models.cache", "KVCache.no_such_method") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py -k "gate or validated or source"`
Expected: FAIL with `AttributeError: ... has no attribute 'VALIDATED_MLX'`.

- [ ] **Step 3: Implement the gates**

In `src/sous/engine/verifyattn.py`:
- Add `import hashlib`, `import importlib`, `import inspect` and `import os` to the imports. Keep `functools`; `logging`, `time` and `warnings` come in Task 5.
- Add these constants after `_SCAN_TO`:

```python
# mlx releases whose SDPA dispatch plan() has been read against; re-read
# backend/metal/scaled_dot_product_attention.cpp and extend this on a bump.
VALIDATED_MLX = frozenset({"0.32.2"})
# Every mlx-vlm function the hook calls or re-implements around, by the module
# that defines it. The daemon's tool environment can resolve a newer mlx-vlm
# than the lock, so these are checked at load, not only in CI.
_SOURCES = {
    "Qwen3_5BatchInvariantForward._attention": "mlx_vlm.models.qwen3_5.speculative_verifier",
    "Qwen3_5Attention._prepare_projected_qkv": "mlx_vlm.models.qwen3_5.language",
    "_create_qwen3_5_attention_mask": "mlx_vlm.models.qwen3_5.language",
    "_qwen3_5_left_padded_attention": "mlx_vlm.models.qwen3_5.language",
    "_qwen3_5_left_padding_info": "mlx_vlm.models.qwen3_5.language",
    "KVCache.update_and_fetch": "mlx_vlm.models.cache",
    "KVCache.make_mask": "mlx_vlm.models.cache",
}
# sha256 of inspect.getsource() as validated (identical in mlx-vlm 0.7.1 and
# 0.7.2). Add a hash only after re-reading that function against the hook.
VALIDATED_MLX_VLM_SOURCES: dict[str, frozenset[str]] = {
    "Qwen3_5BatchInvariantForward._attention": frozenset(
        {"7790ae37e7a0d78d2287217a3050e03a4b9b7c10eb1e60839e4315226186cdf1"}
    ),
    "Qwen3_5Attention._prepare_projected_qkv": frozenset(
        {"da21fd76217c72aa7ef15d4097e1fe428ff0022259f9556eb8fd95b59217c109"}
    ),
    "_create_qwen3_5_attention_mask": frozenset(
        {"a74a1135eb644698e1beb6e43a5d51e27e6490d0b0f4f6d951d08dd5e908c681"}
    ),
    "_qwen3_5_left_padded_attention": frozenset(
        {"e5d9f69065da9885674a6c3e11a65807948c5e9f9d7a1fef6211628328e7b312"}
    ),
    "_qwen3_5_left_padding_info": frozenset(
        {"1cea363e3150f9262f7fb23d11d525b6be32625d3407cd324850d82d6a151fd5"}
    ),
    "KVCache.update_and_fetch": frozenset(
        {"9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b"}
    ),
    "KVCache.make_mask": frozenset(
        {"d65ba0d83b52a93bb00faae6bcbac8add067724ebe9dd8ba6aaa716f81b6b620"}
    ),
}
```

Then add these functions after `probe`:

```python
def _source_digest(module: str, qualname: str) -> str | None:
    """sha256 of an mlx-vlm function's source, or None when it cannot be read."""
    try:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()
    except ImportError, AttributeError, OSError, TypeError:
        return None


def _gate() -> str | None:
    """Why plan() or the hook cannot be trusted in this process, or None."""
    import mlx.core as mx

    version = mx.__version__  # ty: ignore[unresolved-attribute]
    if version not in VALIDATED_MLX:
        return f"mlx {version} not validated"
    # mlx honours this override of its two-pass block count; plan() does not.
    if os.environ.get("MLX_SDPA_BLOCKS", "0") not in ("", "0"):
        return "MLX_SDPA_BLOCKS is set"
    for qualname, module in _SOURCES.items():
        if _source_digest(module, qualname) not in VALIDATED_MLX_VLM_SOURCES[qualname]:
            return f"mlx-vlm {qualname} changed"
    return None
```

The unparenthesised `except A, B:` is PEP 758 syntax, valid on 3.14 and what `ruff format` produces. `ty` does not know that `mlx.core` has `__version__`, hence the pragma. Reading the version through `importlib.metadata` instead would break the test that monkeypatches `mx.__version__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: all pass.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/verifyattn.py tests/test_verifyattn.py
git commit -m "feat(engine): gate grouped verify attention on validated mlx and mlx-vlm" -m "plan() mirrors compiled mlx dispatch and the hook re-implements the body of
an mlx-vlm method, so both are pinned: mlx by version, mlx-vlm by the
source of every function the hook reads, checked at load because the
daemon's tool env can drift from the lock.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: The verifier hook and the tiny-model contract test

**Files:**
- Modify: `src/sous/engine/verifyattn.py`
- Test: `tests/test_verifyattn.py`

**Interfaces:**
- Consumes: `grouped_attention`, `_row_loop`, `HEAD_DIM`, `MIN_ROWS`, `MAX_ROWS`.
- Produces:
  - `_TAG = "_sous_verify_attention"`. The tag value on an attention module is the gqa the probe proved; 0 means untagged.
  - Module globals `_ARCH: str`, `_KVCACHE: type | None` and `calls: dict[str, int]`, with keys `"grouped"`, `"loop"` and `"original"`.
  - `_in_scope(verifier, attention, x, mask, cache) -> bool`
  - `_post_ok(queries, keys, values, length: int) -> bool`
  - `install_wrapper() -> None`: idempotent, installed once per process.
  - `_attention_modules(model) -> list`
  - `_untag(model) -> None`

- [ ] **Step 1: Write the failing tests**

Add `import types` to the stdlib block at the top of `tests/test_verifyattn.py`. Then append:

```python
GQA = 6
VOCAB = 512


def _tiny_language_model():
    """A random qwen3_5 language model with two full-attention layers at the
    27B's head dim and GQA ratio; mlx-vlm's verify path takes it unquantized."""
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
        num_attention_heads=GQA,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=1,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,  # layers 1 and 3 are full attention
    )
    mx.random.seed(0)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    return lm


@pytest.fixture(scope="module")
def tiny():
    """The tiny model prefilled to just below this GPU's first plan transition,
    so verify rows of T = 3..5 straddle it."""
    arch = mx.device_info()["architecture"]
    lm = _tiny_language_model()
    prefix = verifyattn.plan_transitions(arch, GQA)[0] - 3
    ids = mx.random.randint(0, VOCAB, (prefix + 8,), key=mx.random.key(7)).tolist()
    cache = lm.make_cache()
    # Explicit positions and a zero delta, as the VLM engine hands them: the
    # bare language model cannot derive rope positions without a vision config.
    lm(
        mx.array([ids[:prefix]], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(prefix, dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    )
    mx.eval([c.state for c in cache])
    return lm, cache, ids, prefix


def _tag(lm, gqa):
    for module in verifyattn._attention_modules(lm):
        object.__setattr__(module, verifyattn._TAG, gqa)


def _ready(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    verifyattn.install_wrapper()
    monkeypatch.setattr(verifyattn, "_ARCH", mx.device_info()["architecture"])
    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)


def _verify(lm, cache, tokens):
    """One verify forward as DFlash runs it, then the speculative round aborted
    so the cache is back at the prefix."""
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


@pytest.mark.parametrize("t", [2, 3, 4, 5])
def test_hooked_verify_forward_is_bit_equal_to_stock(tiny, t, monkeypatch):
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + t]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    _tag(lm, GQA)
    before = dict(verifyattn.calls)
    try:
        ours = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert all(c.offset == prefix for c in cache if hasattr(c, "offset"))
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))
    grouped = verifyattn.calls["grouped"] - before["grouped"]
    original = verifyattn.calls["original"] - before["original"]
    # T = 2 is stock's own single call; both full-attention layers group from T = 3
    assert (grouped, original) == ((0, 2) if t == 2 else (2, 0))


def test_one_call_over_the_straddle_would_differ(tiny, monkeypatch):
    """The sensitivity half: ignoring the plan changes the logits, so the
    bit-equality above is not vacuous."""
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + 3]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    monkeypatch.setattr(verifyattn, "groups", lambda arch, p, t, gqa: [(0, t)])
    _tag(lm, GQA)
    try:
        bad = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert not mx.array_equal(stock[0], bad[0]).item()


@pytest.mark.parametrize("t", [3, 5])
def test_the_post_projection_fallback_is_the_stock_loop(tiny, t, monkeypatch):
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + t]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    monkeypatch.setattr(verifyattn, "_post_ok", lambda *a: False)
    _tag(lm, GQA)
    before = verifyattn.calls["loop"]
    try:
        ours = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))
    assert verifyattn.calls["loop"] - before == 2


def test_install_wrapper_is_idempotent():
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    verifyattn.install_wrapper()
    first = Qwen3_5BatchInvariantForward._attention
    verifyattn.install_wrapper()
    assert Qwen3_5BatchInvariantForward._attention is first


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


def test_scope_takes_a_plain_causal_verify_of_three_to_eight_rows(monkeypatch):
    assert _scope(monkeypatch, t=3) and _scope(monkeypatch, t=8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"t": 2},  # stock's own single call
        {"t": 1},
        {"t": 9},
        {"batch": 2},
        {"mask": None},
        {"mask": "left_padded_decode"},
        {"head_dim": 128},
        {"tag": 0},  # untagged
        {"tag": 4},  # probed for another GQA ratio
    ],
)
def test_scope_refuses_everything_else(monkeypatch, overrides):
    assert not _scope(monkeypatch, **overrides)


def test_scope_refuses_an_array_mask(monkeypatch):
    assert not _scope(monkeypatch, mask=mx.ones((1, 1, 3, 3), dtype=mx.bool_))


def test_scope_refuses_kvcache_subclasses_quantized_and_left_padded_caches(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    class Subclass(KVCache):
        pass

    quantized = KVCache()
    quantized.bits = 4  # ty: ignore[unresolved-attribute]
    padded = KVCache()
    padded._qwen3_5_decode_left_padding = [1]  # ty: ignore[unresolved-attribute]
    for cache in (Subclass(), quantized, padded):
        assert not _scope(monkeypatch, cache=cache), type(cache).__name__
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py -k "hook or scope or straddle or fallback or wrapper"`
Expected: FAIL with `AttributeError: ... has no attribute 'install_wrapper'`, or `_attention_modules`.

- [ ] **Step 3: Implement the hook**

Add to `src/sous/engine/verifyattn.py`.

First, the state, directly after `VALIDATED_MLX_VLM_SOURCES`:

```python
# Set with object.__setattr__ so it stays out of mlx's parameter tree; the value
# is the GQA ratio the probe proved, 0 when untagged.
_TAG = "_sous_verify_attention"
_ARCH = ""
_KVCACHE: type | None = None
_WRAPPED: set[type] = set()
# Per-path call counts; the tests read them to prove which path ran.
calls = {"grouped": 0, "loop": 0, "original": 0}
```

Then the hook, after `_gate`:

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
    # The exact type: mlx-vlm's KVCache subclasses keep other layouts.
    if type(cache) is not _KVCACHE or hasattr(cache, "bits"):
        return False
    if getattr(cache, "_qwen3_5_decode_left_padding", None) is not None:
        return False
    info = verifier._helpers()._qwen3_5_left_padding_info(cache)
    if info is not None and info[1] > 0:
        return False
    heads = getattr(attention, "num_attention_heads", 0)
    kv_heads = getattr(attention, "num_key_value_heads", 0)
    return (
        getattr(attention, "head_dim", None) == HEAD_DIM
        and kv_heads > 0
        and heads == gqa * kv_heads
    )


def _post_ok(queries: Any, keys: Any, values: Any, length: int) -> bool:
    """What the grouped call assumes of the projected tensors."""
    import mlx.core as mx

    return (
        isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.dtype == keys.dtype == values.dtype
        and queries.shape[2] == length
        and keys.shape[-2] == values.shape[-2] >= length
        and queries.shape[1] % keys.shape[1] == 0
    )


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
        batch, length, _ = x.shape
        q_proj_output, keys, values = self._linears(
            (attention.q_proj, attention.k_proj, attention.v_proj), x
        )
        queries, keys, values, gate, mask = attention._prepare_projected_qkv(
            q_proj_output, keys, values, cache, position_ids, position_embeddings, mask
        )
        output = self._helpers()._qwen3_5_left_padded_attention(
            queries, keys, values, cache=cache, scale=attention.scale, mask=mask
        )
        if output is None:
            if _post_ok(queries, keys, values, length):
                calls["grouped"] += 1
                prefix = keys.shape[-2] - length
                output = grouped_attention(queries, keys, values, attention.scale, prefix, _ARCH)
            else:
                calls["loop"] += 1
                output = _row_loop(queries, keys, values, cache, attention.scale)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output * mx.sigmoid(gate))

    cls._attention = _attention


def install_wrapper() -> None:
    """Wrap the exact verifier's attention once per process. The wrapper acts only
    on tagged modules, so installing it changes nothing for any other model."""
    cls = importlib.import_module(
        "mlx_vlm.models.qwen3_5.speculative_verifier"
    ).Qwen3_5BatchInvariantForward
    if cls not in _WRAPPED:
        _wrap(cls)
        _WRAPPED.add(cls)


def _attention_modules(model: Any) -> list[Any]:
    """The loaded target's full-attention modules; linear-attention layers have none."""
    root = getattr(model, "language_model", model)
    layers = getattr(getattr(root, "model", None), "layers", None) or []
    return [
        layer.self_attn
        for layer in layers
        if not getattr(layer, "is_linear", True) and hasattr(layer, "self_attn")
    ]


def _untag(model: Any) -> None:
    for module in _attention_modules(model):
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, 0)
```

If `ty` rejects `cls._attention = _attention` or `setattr(attention, verifyattn._TAG, tag)` in the tests, add `# ty: ignore[<rule>]` with exactly the rule it reports. `cls: Any` should already prevent the first one.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: all pass. The tiny-model fixture takes about 0.2 s and each case about 0.01 s on an M2.

- [ ] **Step 5: Add the contract test to the slow override run**

In `test_exactness_holds_under_every_dispatch_table`, extend `tests` with the tiny-model contract so the hook itself is proven under 's' and 'd'. On those tables `plan_transitions(arch, 6)[0] - 3` is 1021.

```python
        "tests/test_verifyattn.py::test_hooked_verify_forward_is_bit_equal_to_stock",
        "tests/test_verifyattn.py::test_one_call_over_the_straddle_would_differ",
```

Run: `uv run pytest tests/test_verifyattn.py -m slow`
Expected: 3 passed: the two dispatch-table runs and the override sanity test. Each child runs about ten tests, taking 7–9 s on an M2.

- [ ] **Step 6: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/verifyattn.py tests/test_verifyattn.py
git commit -m "feat(engine): hook the exact verifier's attention for grouped calls" -m "The wrapper re-runs mlx-vlm's attention body with the per-row loop
replaced, deciding scope before the projections append to the cache. A
tiny random qwen3_5 model proves logits and captured hidden states stay
bit-equal across a plan transition, on this GPU and under 's' and 'd'.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `enable()`

**Files:**
- Modify: `src/sous/engine/verifyattn.py`
- Test: `tests/test_verifyattn.py`

**Interfaces:**
- Consumes: `_gate`, `probe`, `install_wrapper`, `_attention_modules`, `_untag`, `_TAG`, `_ARCH`, `_KVCACHE`, `HEAD_DIM`; and `_model_type` from `sous.engine.int8prefill`.
- Produces:
  - `SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})`
  - `enable(model, *, enabled: bool) -> dict[str, Any]`, returning `{"state", "reason", "probe_seconds"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifyattn.py`:

```python
def _fresh_model():
    return _tiny_language_model()


def test_enable_off_does_nothing():
    lm = _fresh_model()
    assert verifyattn.enable(lm, enabled=False) == {
        "state": "off",
        "reason": None,
        "probe_seconds": None,
    }
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_tags_every_full_attention_layer_with_the_proved_ratio():
    lm = _fresh_model()
    status = verifyattn.enable(lm, enabled=True)
    try:
        assert status["state"] == "active" and status["reason"] is None
        assert isinstance(status["probe_seconds"], float)
        modules = verifyattn._attention_modules(lm)
        assert len(modules) == 2 and all(getattr(m, verifyattn._TAG) == GQA for m in modules)
    finally:
        verifyattn._untag(lm)


def test_enable_refuses_other_model_types(monkeypatch):
    lm = _fresh_model()
    # The language model carries model_type itself, and _model_type reads it first.
    monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
    with pytest.warns(UserWarning, match="unsupported model type 'qwen3_5_moe'"):
        status = verifyattn.enable(lm, enabled=True)
    assert status == {
        "state": "unavailable",
        "reason": "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
        "probe_seconds": None,
    }


def test_enable_reports_a_failed_gate(monkeypatch):
    lm = _fresh_model()
    monkeypatch.setattr(verifyattn, "_gate", lambda: "mlx 0.99.0 not validated")
    with pytest.warns(UserWarning, match="mlx 0.99.0 not validated"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable" and status["reason"] == "mlx 0.99.0 not validated"
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_reports_a_failed_probe_with_its_cost(monkeypatch):
    lm = _fresh_model()
    monkeypatch.setattr(verifyattn, "probe", lambda *a: "grouped attention differs at prefix 1")
    with pytest.warns(UserWarning, match="differs at prefix 1"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable"
    assert isinstance(status["probe_seconds"], float)
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_never_raises(monkeypatch):
    lm = _fresh_model()

    def boom(*a):
        raise RuntimeError("metal said no")

    monkeypatch.setattr(verifyattn, "probe", boom)
    with pytest.warns(UserWarning, match="metal said no"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable" and status["reason"] == "metal said no"
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_verifyattn.py -k enable`
Expected: FAIL with `AttributeError: ... has no attribute 'enable'`.

- [ ] **Step 3: Implement `enable()`**

In `src/sous/engine/verifyattn.py`:
- Add `import logging`, `import time` and `import warnings` to the imports.
- Add `from sous.engine.int8prefill import _model_type` after the stdlib imports.
- Add `logger = logging.getLogger("sous.engine.verifyattn")` after the imports.
- Add the constant directly after `VALIDATED_MLX`:

```python
# The MoE variant reuses the same verifier class with other attention shapes.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
```

Then, at the end of the file:

```python
def _refuse(reason: str, probe_seconds: float | None = None) -> dict[str, Any]:
    """The verifier keeps mlx-vlm's per-row attention: say so once, and why."""
    warnings.warn(
        f"sous: exact grouped verify attention unavailable ({reason}); "
        "verifying with mlx-vlm's per-row attention",
        stacklevel=3,
    )
    return {"state": "unavailable", "reason": reason, "probe_seconds": probe_seconds}


def enable(model: Any, *, enabled: bool) -> dict[str, Any]:
    """Opt one loaded model's verifier in. Returns the status the engine exposes;
    never raises — a model load must not fail because of an optimisation."""
    global _ARCH, _KVCACHE
    if not enabled:
        return {"state": "off", "reason": None, "probe_seconds": None}
    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return _refuse(f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)")
    seconds: float | None = None
    try:
        import mlx.core as mx

        reason = _gate()
        if reason is not None:
            return _refuse(reason)
        modules = _attention_modules(model)
        shapes = {(m.num_attention_heads, m.num_key_value_heads, m.head_dim) for m in modules}
        if len(shapes) != 1:
            return _refuse(f"expected one attention shape, found {sorted(shapes)}")
        q_heads, kv_heads, head_dim = shapes.pop()
        if head_dim != HEAD_DIM or kv_heads <= 0 or q_heads % kv_heads:
            return _refuse(f"unsupported attention shape {q_heads}/{kv_heads}/{head_dim}")
        arch = str(mx.device_info().get("architecture", ""))
        start = time.perf_counter()
        reason = probe(arch, q_heads, kv_heads, modules[0].k_norm.weight.dtype)
        seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds)
        _ARCH = arch
        _KVCACHE = importlib.import_module("mlx_vlm.models.cache").KVCache
        install_wrapper()
        for module in modules:
            object.__setattr__(module, _TAG, q_heads // kv_heads)
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _untag(model)
        return _refuse(str(e) or type(e).__name__, seconds)
    logger.info("exact verify attention: %d layers, probe %.2fs", len(modules), seconds)
    return {"state": "active", "reason": None, "probe_seconds": seconds}
```

The `_refuse` calls inside `enable` keep `stacklevel=3` pointing at `enable`'s caller.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verifyattn.py`
Expected: all pass.

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/verifyattn.py tests/test_verifyattn.py
git commit -m "feat(engine): enable grouped verify attention behind its gates and probe" -m "One call per model load runs the gates and the probe and tags the full-
attention layers only when every guard passes; any failure is one warning
and state unavailable, never a failed load.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Wire it into the engine, the load line and the status document

**Files:**
- Modify: `src/sous/engine/vlm.py:72` (import) and `:102` (after `_pin_block_size`)
- Modify: `src/sous/engine/base.py`:
  - `ManagedEngine`: after the `int8_prefill_status` property, around `:384-388`
  - `EngineManager.get()`: the load line, around `:654-661`
  - `EngineManager.status()`: after the int8 block, around `:965-967`
- Modify: `src/sous/tune/suite/runner.py:120-121` (comment)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: `verifyattn.enable(model, *, enabled)` from Task 5.
- Produces: `VLMEngine.verify_attention_status` (the dict) and `ManagedEngine.verify_attention_status -> dict | None`.
  - Load line tokens: `verify_attention=<state>`, plus `verify_attention_probe_s=<x>` when a probe ran.
  - Status document: `status()["verify_attention"] == {"state", "reason"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py`. `_positionless_model`, `_RecordingTokenizer`, `_stub`, `_cfg` and `_positional_factory` already exist in that file.

```python
@pytest.mark.parametrize("drafter_loads", [True, False])
def test_vlm_engine_probes_verify_attention_only_with_a_drafter(monkeypatch, drafter_loads):
    from sous.engine import verifyattn, vlm

    model = _positionless_model()
    processor = types.SimpleNamespace(tokenizer=_RecordingTokenizer())
    _stub(monkeypatch, "mlx_vlm", load=lambda model_id: (model, processor))
    _stub(monkeypatch, "mlx_vlm.sample_utils", make_sampler=lambda **kw: None)

    def load_drafter(m, draft_id):
        if not drafter_loads:
            raise RuntimeError("no such repo")
        return types.SimpleNamespace(prefer_requested_block_size=False), "dflash"

    monkeypatch.setattr(vlm, "_load_quantized_drafter", load_drafter)
    seen: dict[str, object] = {}

    def fake_enable(m, *, enabled):
        seen["model"], seen["enabled"] = m, enabled
        return {"state": "active" if enabled else "off", "reason": None, "probe_seconds": None}

    monkeypatch.setattr(verifyattn, "enable", fake_enable)
    if drafter_loads:
        engine = vlm.VLMEngine("test/model", cache_budget=0, draft_id="z-lab/drafter")
    else:
        with pytest.warns(UserWarning, match="speculative drafter"):
            engine = vlm.VLMEngine("test/model", cache_budget=0, draft_id="z-lab/drafter")
    assert seen == {"model": model, "enabled": drafter_loads}
    assert engine.verify_attention_status["state"] == ("active" if drafter_loads else "off")


def test_get_logs_the_verify_attention_state_and_probe_cost(caplog):
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        engine.verify_attention_status = {  # ty: ignore[unresolved-attribute]
            "state": "active",
            "reason": None,
            "probe_seconds": 0.21,
        }
        return engine

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1
    assert lines[0].endswith(
        " positions=engine verify_attention=active verify_attention_probe_s=0.21"
    )


def test_get_logs_no_probe_cost_when_no_probe_ran(caplog):
    import logging

    def factory(model_id):
        engine = _positional_factory(model_id)
        engine.verify_attention_status = {  # ty: ignore[unresolved-attribute]
            "state": "off",
            "reason": None,
            "probe_seconds": None,
        }
        return engine

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert lines[0].endswith(" verify_attention=off")


def test_status_carries_the_verify_attention_view_when_the_engine_reports_one(tmp_path):
    inner = FakeEngine([])
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert "verify_attention" not in manager.status(), "fakes without the attribute stay silent"
    inner.verify_attention_status = {  # ty: ignore[unresolved-attribute]
        "state": "unavailable",
        "reason": "mlx 0.33.0 not validated",
        "probe_seconds": None,
    }
    assert manager.status()["verify_attention"] == {
        "state": "unavailable",
        "reason": "mlx 0.33.0 not validated",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k "verify_attention or no_probe_cost"`
Expected: FAIL. The first test fails with `AttributeError: ... has no attribute 'verify_attention_status'`, and the load-line and status tests fail on missing tokens and keys.

- [ ] **Step 3: Wire `vlm.py`**

In `src/sous/engine/vlm.py`, change the import at `:72` to:

```python
        from sous.engine import int8prefill, verifyattn
```

`_pin_block_size(self._draft, draft_block_size)` sits inside the `if draft_id:` block (12-space indent). Insert this after that block ends, at `__init__` body indentation (8 spaces), directly before the `# Forks on disk` comment:

```python
        # Only speculation runs the verifier, so only a drafter that loaded
        # earns the probe; before the budget is measured, so the probe's
        # transient K/V is already released.
        self.verify_attention_status = verifyattn.enable(
            self._model, enabled=self._draft is not None
        )
```

At body level it runs whether or not a drafter was requested, so a load with no drafter reports `off`.

- [ ] **Step 4: Wire `base.py`**

After the `int8_prefill_status` property of `ManagedEngine`, add:

```python
    @property
    def verify_attention_status(self) -> dict | None:
        # Optional on purpose: only the VLM backend sets this; the LM backend
        # and fakes have no such attribute.
        return getattr(self._inner, "verify_attention_status", None)
```

In `EngineManager.get()`, after the `positions=` append and before `_logger.info(line)`, add:

```python
        verify = engine.verify_attention_status
        if verify is not None:
            line += f" verify_attention={verify['state']}"
            if verify.get("probe_seconds") is not None:
                line += f" verify_attention_probe_s={verify['probe_seconds']}"
```

In `EngineManager.status()`, after the `int8_prefill` block, add:

```python
                verify = self._engine.verify_attention_status
                if verify is not None:
                    out["verify_attention"] = {
                        "state": verify["state"],
                        "reason": verify["reason"],
                    }
```

- [ ] **Step 5: Update the pass-through comment**

In `src/sous/tune/suite/runner.py`, change the comment above `__getattr__` (around `:120-121`) so it lists the new name:

```python
        # drafter, positions, int8_prefill_status, verify_attention_status:
        # whatever the backend has, read through getattr(..., None) by ManagedEngine.
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -m "not model"`
Expected: all pass (about 110 s on an M2). The existing load-line tests still pass, because their fakes have no `verify_attention_status` and the new tokens come after `positions=`. `test_vlm_key_fields_read_positions_and_the_realised_int8_state` in `tests/test_engine_forkhooks.py` passes unchanged, which is the fork key staying as it was.

- [ ] **Step 7: Lint, type-check and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine/vlm.py src/sous/engine/base.py src/sous/tune/suite/runner.py tests/test_engine_base.py
git commit -m "feat(engine): run exact verify attention whenever a drafter loads" -m "The VLM engine opts its target in after the drafter loads and before the
cache budget is measured; the load line and the status document say
whether it is active and what the probe cost. vlm.py is hashed into the
fork store's epoch, so on-disk forks rebuild once after this lands.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: The real-model test, docs and spec numbers

**Files:**
- Create: `tests/test_verifyattn_model.py`
- Modify:
  - `CLAUDE.md`: a new Gotchas bullet after the `int8prefill.enable()` bullet, before `## Security boundary`
  - `README.md:486-490`: the load-line description
  - `docs/superpowers/specs/2026-09-24-exact-verify-attention-design.md`: the probe cap and measured cost

- [ ] **Step 1: Write the model test**

Create `tests/test_verifyattn_model.py`:

```python
"""The real default model's verify forward: grouped attention against mlx-vlm's
per-row loop.

Acceptance bars:
- The engine reports verify attention active with the default drafter.
- At prefixes that straddle the M5 Pro's plan transitions (1021, 8190, 32766)
  and at a subagent-sized 57,000, for T = 3..8, the logits and every DFlash2
  capture hidden state are bit-equal with the hook on and off.
- The grouped path runs on all 16 full-attention layers.

Takes about 6 minutes on the M5 Pro: one 27B load and a 57K prefill.
"""

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import verifyattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
CAPTURE = [5, 19, 33, 47, 61]  # DFlash2's target_layer_ids on the 27B
PREFIXES = [1021, 8190, 32766, 57000]
ROOT = Path(__file__).resolve().parents[1]


def _corpus(engine, n):
    files = sorted((ROOT / "src" / "sous").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ids = engine._encode("".join(f.read_text() for f in files))
    assert len(ids) >= n, len(ids)
    return ids[:n]


def _offset(cache):
    return max((int(getattr(c, "offset", 0) or 0) for c in cache), default=0)


def _verify(lm, cache, tokens):
    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=CAPTURE,
        speculative_verify=True,
    )
    arrays = [out.logits, *out.hidden_states]
    mx.eval(arrays)
    out.gdn_states.abort()
    return arrays


def test_grouped_verify_forward_is_bit_equal_to_mlx_vlms_loop():
    engine = VLMEngine(DEFAULT_27B, cache_budget=0, draft_id=DRAFTER)
    try:
        status = engine.verify_attention_status
        assert status["state"] == "active", status
        modules = verifyattn._attention_modules(engine._model)
        assert len(modules) == 16
        gqa = getattr(modules[0], verifyattn._TAG)
        assert engine._model is not None
        lm = engine._model.language_model
        ids = _corpus(engine, max(PREFIXES) + 2056)
        cache = engine.new_cache()
        for target in PREFIXES:
            while _offset(cache) < target:
                start = _offset(cache)
                engine.prefill(cache, ids[start : start + min(2048, target - start)])
                mx.eval([c.state for c in cache if hasattr(c, "state")])
                assert _offset(cache) > start, "prefill did not advance"
            prefix = _offset(cache)
            for t in range(3, 9):
                tokens = ids[prefix : prefix + t]
                for module in modules:
                    object.__setattr__(module, verifyattn._TAG, 0)
                stock = _verify(lm, cache, tokens)
                for module in modules:
                    object.__setattr__(module, verifyattn._TAG, gqa)
                before = verifyattn.calls["grouped"]
                ours = _verify(lm, cache, tokens)
                assert verifyattn.calls["grouped"] - before == len(modules), (prefix, t)
                assert all(
                    mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True)
                ), (prefix, t)
    finally:
        engine.unload()
```

Run: `uv run pytest tests/test_verifyattn.py tests/test_verifyattn_model.py -m "not model"`
Expected: the model test is deselected; the others pass. The model test itself runs on the M5 in Task 8.

- [ ] **Step 2: Add the CLAUDE.md gotcha**

In `CLAUDE.md`, insert after the bullet that starts ``- `int8prefill.enable()` never raises`` and before `## Security boundary`:

```markdown
- `engine/verifyattn.py` replaces mlx-vlm's per-row SDPA loop in the exact
  verifier (`Qwen3_5BatchInvariantForward._attention`, T = 3..8) with one
  stock `mx.fast.scaled_dot_product_attention` per group of rows that share
  mlx's kernel plan. Grouping is bit-exact only because inside one plan mlx
  assigns key i to simdgroup `i % 32` or block `i % blocks` whatever the key
  count and the causal mask skips excluded keys, and `plan()` mirrors mlx
  0.32.2's dispatch (`VALIDATED_MLX`) — re-read
  `scaled_dot_product_attention.cpp` and extend it on every mlx bump; every
  mlx-vlm function the hook reads is pinned by source hash
  (`VALIDATED_MLX_VLM_SOURCES`), checked at load because the daemon's tool
  environment can resolve a newer mlx-vlm than the lock. T = 2 is left
  alone: stock already makes one call there, and grouping it would change
  output at plan straddles. Scope is decided before the projections —
  `_prepare_projected_qkv` appends to the KV cache, so the wrapper never
  re-enters the original method after them. `enable()` runs only when a
  drafter loaded, probes every plan transition on this GPU before tagging,
  and like int8 never raises. CI proves the 's' and 'd' tables through
  `MLX_METAL_GPU_ARCH` subprocesses. The module stays out of
  `forkstore._EPOCH_FILES` on purpose: prefill never enters the verifier and
  verify output is bit-identical.
```

- [ ] **Step 3: Update the README's load-line description**

In `README.md:486-490`, change the sentence about the load line so that after ``positions=engine|model`` on the VLM backend it continues:

```markdown
..., plus `positions=engine|model` on the VLM backend (which side supplies the
rotary positions behind a warm cache) and `verify_attention=active|unavailable|off`
(whether speculative verify runs its attention as grouped exact calls; with
`verify_attention_probe_s=` when the load-time exactness probe ran).
```

Keep the surrounding sentences intact. Match the paragraph's existing wrapping.

- [ ] **Step 4: Correct the spec's probe numbers**

In the spec's "Load-time probe" bullet:
- Replace `capped at the last transition + 8, which is 65545 keys on 's' and 'd' and 4104 on other suffixes` with `capped at the last transition + 7 keys: 65544 on 's', 65543 on 'd' and 4103 on other suffixes`.
- Replace the "Memory and time" sub-bullet with: `**Memory and time.** Measured on an M2. On its own 'g' table the probe takes 0.9 s cold (it is the process's first attention call, so the SDPA kernels compile inside it) and 0.1 s warm, with a 51 MB peak. Under the 's' and 'd' tables it takes 2.5–2.9 s cold and 1.4–2.2 s warm, with about 0.8 GB peak. The M5 Pro's figure is recorded on its load line (`verify_attention_probe_s`).`

Also add one sentence to "Wiring and visibility" under **Unchanged**: `` `vlm.py` itself is in `_EPOCH_FILES`, so this change invalidates the on-disk forks once after it lands.``

- [ ] **Step 5: Run the checks and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model"
git add tests/test_verifyattn_model.py CLAUDE.md README.md docs/superpowers/specs/2026-09-24-exact-verify-attention-design.md
git commit -m "docs: record how grouped verify attention stays exact" -m "The plan mirror, both gates and the scope rule are what keep the path
bit-exact, and each has a trap for whoever next bumps mlx or mlx-vlm; the
model test pins the real 27B at the M5 Pro's plan straddles.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Gate on the M5 Pro

This task is manual measurement, not code. It runs over `ssh m5pro` in `~/code/personal/sous-remote`, under `gpu_run_m5.py`, with the daemon idle. Never touch `~/code/personal/sous`, `~/.sous` or the daemon itself. `S` below stands for `~/code/personal/sous-spikes/splash-port-2026-09-24`.

**Files:**
- Create (out of tree, on this Mac, then rsync): `~/code/sous-spikes/splash-port-2026-09-24/m5/gate/port_branch.py`, plus copies of `e2e.py` and `agg.py` from `m5/integrated/`.

- [ ] **Step 1: Push the branch and update the M5 checkout**

```bash
git push -u origin feat/exact-verify-attention
ssh m5pro 'cd ~/code/personal/sous-remote && git fetch -q && git checkout -q feat/exact-verify-attention && git merge -q --ff-only origin/feat/exact-verify-attention && uv sync -q && git log -1 --oneline'
```

- [ ] **Step 2: Run the test suites on the M5**

```bash
ssh m5pro 'cd ~/code/personal/sous-remote && S=~/code/personal/sous-spikes/splash-port-2026-09-24 && GPU_LABEL=gate-tests python3 $S/gpu_run_m5.py uv run pytest -m "not model" tests/test_verifyattn.py tests/test_engine_base.py 2>&1 | tail -15'
ssh m5pro 'cd ~/code/personal/sous-remote && S=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $S/detach.py $S/m5/gate/model.txt env GPU_LABEL=gate-model python3 $S/gpu_run_m5.py uv run pytest -m model tests/test_verifyattn_model.py'
ssh m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/splash-port-2026-09-24/m5/gate/model.txt 540'
```

Expected:
- The first command passes, slow tests included. It names the two files because a full-suite run on macOS 27 needs `--deselect tests/test_int8prefill.py` (#123).
- The model test prints `1 passed`, and its log ends `exit=0` and `daemon stayed idle`.
- Repeat `wait_for.py` until it prints DONE.
- If any output says `DAEMON-CONTAMINATED`, re-run that command.

- [ ] **Step 3: Build the end-to-end harness shim**

With the branch installed, "stock" is the untagged path (exactly main's behaviour) and "attn" is the tagged path. Create `m5/gate/port_branch.py`:

```python
"""e2e.py's `port` interface over the branch's verifyattn: stock = untagged
(main's code path), attn = tagged (grouped verify attention)."""

from sous.engine import verifyattn

COUNTS = verifyattn.calls
_SAVED: dict[int, tuple[object, int]] = {}


def install():
    pass


def bind(model):
    for module in verifyattn._attention_modules(model):
        _SAVED[id(module)] = (module, getattr(module, verifyattn._TAG, 0))


def variant(name, proj_ts=()):
    for module, gqa in _SAVED.values():
        object.__setattr__(module, verifyattn._TAG, gqa if name == "attn" else 0)


def reset_counts():
    for key in COUNTS:
        COUNTS[key] = 0
```

Then:
1. Copy `m5/integrated/e2e.py` and `m5/integrated/agg.py` into `m5/gate/`.
2. In the copied `e2e.py`, replace `import port` with `import port_branch as port`.
3. Directly after the line that constructs the `VLMEngine`, add `port.bind(eng._model)`, using whatever that line's variable name is.
4. Check that the engine's load line or `eng.verify_attention_status` says `active` and fail fast if not: add `assert eng.verify_attention_status["state"] == "active"` after the bind.
5. Sync the folder: `rsync -a ~/code/sous-spikes/splash-port-2026-09-24/m5/gate/ m5pro:code/personal/sous-spikes/splash-port-2026-09-24/m5/gate/`.

- [ ] **Step 4: Run the gate: block 3, greedy and sampled, ~2K and ~57K, stock vs attn**

```bash
ssh m5pro 'cd ~/code/personal/sous-remote && S=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $S/detach.py $S/m5/gate/short.txt env GPU_LABEL=gate-e2e python3 $S/gpu_run_m5.py uv run python $S/m5/gate/e2e.py $S/m5/gate/e2e.jsonl short 0,1,2,3 3 0,1 stock,attn 5 256'
ssh m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/splash-port-2026-09-24/m5/gate/short.txt 540'
ssh m5pro 'cd ~/code/personal/sous-remote && S=~/code/personal/sous-spikes/splash-port-2026-09-24 && python3 $S/detach.py $S/m5/gate/long.txt env GPU_LABEL=gate-e2e python3 $S/gpu_run_m5.py uv run python $S/m5/gate/e2e.py $S/m5/gate/e2e.jsonl long 0,1,2,3 3 0,1 stock,attn 5 256'
ssh m5pro 'python3 ~/code/personal/sous-spikes/splash-port-2026-09-24/wait_for.py ~/code/personal/sous-spikes/splash-port-2026-09-24/m5/gate/long.txt 540'
ssh m5pro 'cd ~/code/personal/sous-remote && S=~/code/personal/sous-spikes/splash-port-2026-09-24 && uv run python $S/m5/gate/agg.py $S/m5/gate/e2e.jsonl'
```

- Repeat each `wait_for.py` until DONE. The long run prefills four 57K prompts and takes about 30–40 minutes.
- `agg.py` is CPU-only.

- [ ] **Step 5: Judge the gate and record it**

The gate passes only if all three hold:
- Greedy output is token-identical to the AR greedy reference for every prompt at block 3 in the `attn` variant. `e2e.py` records parity per run.
- Pooled `attn` / `stock` decode tok/s is at least 1.05 at ~57K, for greedy and for sampled.
- At ~2K, `attn` / `stock` is at least 0.95.

Also record `verify_attention_probe_s` from the model-load line in `short.txt`. Pull the results back with `rsync -a m5pro:code/personal/sous-spikes/splash-port-2026-09-24/m5/gate/ ~/code/sous-spikes/splash-port-2026-09-24/m5/gate/`.

If the gate fails, stop and report the numbers. Do not open the PR.

- [ ] **Step 6: Open the PR**

Only after a passing gate. Use the `commit-commands:commit-push-pr` flow or `gh pr create`. The body must include:
- what changed and why, with the spec path;
- the gate's table (greedy and sampled, 2K and 57K, stock vs attn, parity) and the probe's load-time cost on the M5;
- the note that on-disk forks rebuild once after this lands (`vlm.py` is in `_EPOCH_FILES`);
- links to #126 and #127;
- `🤖 Generated with [Claude Code](https://claude.com/claude-code)` as the last line.

Never put a person's name or a home path in the PR text.

---

## Self-Review Notes

- **Spec coverage:**
  - mechanism: Tasks 1–2
  - hook and scope: Task 4
  - four guards: Tasks 2, 3 and 4, assembled in Task 5
  - wiring and visibility: Task 6
  - testing:
    - pure tables: Task 1
    - GPU and the `MLX_METAL_GPU_ARCH` subprocesses: Tasks 2 and 4
    - gates: Task 3
    - contract: Task 4
    - model test: Task 7
    - gate: Task 8
  - docs: Task 7
  - out of scope: untouched
- **The spec's cap numbers were off by one:** the largest key count probed is the last transition + 7 (65544 on 's'), so the "+ 8" wording overcounted. Task 7 Step 4 corrects them in the spec.
- **Type consistency:** every task uses the same names and signatures:
  - `grouped_attention(queries, keys, values, scale, prefix, arch)`
  - `_row_loop(queries, keys, values, cache, scale)`
  - `probe(arch, q_heads, kv_heads, dtype)`
  - `enable(model, *, enabled)`
  - the status keys `state`, `reason` and `probe_seconds`
  - `_TAG`, `calls`, `_attention_modules`, `_untag` and `install_wrapper`
