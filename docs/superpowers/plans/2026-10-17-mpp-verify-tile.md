# MPP verify tile over slab weights (stage 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the DFlash round's verify projections on the M5 Pro's tensor units from slab-ordered weights, with the MLP linears held slab-only and every other path keeping today's bits, under the block-invariance contract.

**Architecture:** A small `slab` module owns the layout (repack, inverse, the slab marker every reader tests) so that `projkernel`, `int8prefill` and the new `projtile` can all read it without an import cycle. `projtile` owns the slab-only conversion of gate/up/down, the stacked copies of the attention and GDN linears, #126's arm-A tensor-op kernel at its tuned plans, a decode window and the routing predicate the projection kernel's two seams ask. The projection kernel, the one-row kernel and the int8 GEMM gain slab addressing with their arithmetic unchanged (bitwise against their mlx-layout forms), and the int8 GEMM takes the 9–127-row MLP calls a slab-only module can no longer hand to stock. The engine enables the tile after the drafter route and before kv8, keys the fork store on the conversion, prints the scope, and exposes the status; `sous tune` keys suite rows on the flag.

**Tech Stack:** Python 3.14, mlx 0.32.3 / mlx-vlm 0.7.2 (pinned `==`), `mx.fast.metal_kernel` with Metal 4 MPP (`nax.h`, macOS 26.2+), pytest (`uv run pytest -m "not model"`), ty, ruff.

**Spec:** `docs/superpowers/specs/2026-10-16-mpp-verify-tile-design.md` (approved 2026-10-17). The plan argues from it; read both. Revised 2026-10-17 after a three-lens review (coverage, code facts, executability).

## Global Constraints

- Branch `feat/projection-tile` from `main` **after PR #188 (the drafter route) merges**: the plan names `_DRAFT_TAG`, `_draft_active` and `enable_drafter`, which #188 adds. `main` is protected: branch + PR, all four CI jobs green (`uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`).
- Python ≥ 3.14; `# ty: ignore[rule]`, never `# type: ignore`; `uv`, never pip; `addopts=-q` is set, never pass another `-q`.
- mlx / mlx_lm / mlx_vlm imports stay function-local. `projkernel` and `int8prefill` import `projtile` only inside functions (projkernel imports int8prefill at module level and projtile imports projkernel, so a module-level import would cycle); the shared types live in `slab.py`, which imports nothing of sous's. Every thread that touches mlx releases its state (`engine.base.release_mlx_thread_state`) before it exits; everything in `enable()` runs on the `sous-model-load` thread; nothing is left lazy at the end of a load.
- A preprocessor switch is a header `#define`, never a `metal_kernel` template entry (`#if` cannot see a template parameter): the precedent is `PRESUM` in `projkernel._header`.
- Tests never touch the real `~/.sous`; `docs/superpowers/**` is never edited (the spec and this plan included, once committed); no spec, plan, task or step numbers in code or test comments — state the fact.
- Kernels compile under `compile_options={"math_mode": "safe"}`; the tensor-op kernel gets `nax.h` and `remove_addrspace_t` on every cooperative-tensor `decltype`; CI (macos-15, no tensor units) skips tensor-op tests through an `Availability` mark and never fails for their absence.
- Numbers the spec fixes: `MAX_ROWS = 8`; slab tile 256 columns, 32-byte (column, group) units, planes `[T][G][256]`; `TILE_COPY_MIN_BUDGET_GIB = 14`; `TILE_FLOOR_FACTOR = 1.05`; `REL_RMS_BOUND = 1e-2` (cancelling layouts only); int8 slab floor `SLAB_MIN_ROWS = 9`; `MEASURED_CORES = {20}`; the plan table as decision 5; config values `true | false | "mlp" | "all"`; fork-key field `ptile` ∈ {`active`, `off`}; `_EPOCH_FILES` gains `slab.py` and `projtile.py`, `_EPOCH_EXCLUDED_KERNELS` gains `projection_tile.metal` and `projection_tile.h`; the int8 counter is `int8prefill.calls["int8_small"]`; the status and load line report GB of 1e9 bytes (`copy_gb`) while every decision is made in bytes.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body; signed as `krcm0209 <17315774+krcm0209@users.noreply.github.com>`; no attribution lines; never `--no-gpg-sign` or `--no-verify`. Splash (Apache-2.0) is attributed in `THIRD_PARTY_NOTICES.md` for every file that adopts its layout or kernel.
- Out-of-tree harness (`~/code/sous-spikes/tile-gate`) is never committed; every GPU job on the M5 runs under `gpu_run_m5.py` with the production daemon idle.

## Review Focus

1. **A prefill that begins with the tile's window still open** (a decode that raised past its `finally`): KV with tile bits would enter a fork under a key that cannot tell. `prefill()` must raise, not serve — Task 8's `test_prefill_refuses_an_open_tile_window` (CI, stubbed engine).
2. **An MLP width that is not a multiple of 256** (another checkpoint): padded planes, kernels that write or read past `n_real`, the slice back, the inverse that drops the padding — Task 2's `test_repack_round_trip_is_bitwise[n4160]`, Task 3's `test_slab_bodies_equal_mlx_bodies[...-n544]`, Task 4's `test_the_slab_gemm_equals_the_mlx_layout_gemm[n4160-...]`, Task 5's `test_every_row_is_independent_of_the_row_count[...-n96]`/`[...-n4160]`.
3. **An explicit `prompt_cache_gb` of 0 or a small value with `projection_tile = true` or `"all"`**, and an unreadable headroom: no warning, the copy decided by whether the configured budget still fits beside it — Task 7's `test_copy_decision_table`.
4. **A verifier call at scope `mlp` whose linears are all attention or GDN** (the four-linear GDN call, q+k+v, out_proj), or any call whose key has no plan: the whole call stays on the MMA kernel, no run is split between paths — Task 6's `test_a_call_with_an_unservable_linear_stays_on_the_mma_kernel_whole` and `test_a_key_outside_the_plan_table_is_served_by_the_mma_kernel`.
5. **Planes outliving the model** (a module-level reference): `sous tune`'s 1 GiB release check would fail on every arm — Task 7's `test_no_plane_outlives_its_model`.

---

### Task 1: The `projection_tile` config key

**Files:**
- Modify: `src/sous/config.py` (`_KNOWN["model"]`, `SousConfig`, a `_projection_tile` parser beside `_projection_kernel`, `load_config`'s field assignment, a comment block beside `projection_kernel: bool = True`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SousConfig.projection_tile: bool | str = True` with values `True`, `False`, `"mlp"`, `"all"`; `_projection_tile(model: dict) -> bool | str`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`, in the file's own pattern (`p = tmp_path / "config.toml"; p.write_text(...); load_config(p)`, `SousConfig()` for the dataclass default, `pytest.warns(UserWarning, match=...)` for bad values):

```python
def test_projection_tile_defaults_on(tmp_path): ...        # load_config of "[model]\nid = 'x/y'\n" → True; SousConfig().projection_tile is True
@pytest.mark.parametrize("value, want", [("true", True), ("false", False), ('"mlp"', "mlp"), ('"all"', "all")])
def test_projection_tile_reads_its_four_values(tmp_path, value, want): ...   # under warnings.simplefilter("error"): no unknown-key warning
@pytest.mark.parametrize("bad", ["1", '"on"', '"ALL"', "[]"])
def test_projection_tile_rejects_anything_else_with_a_warning(tmp_path, bad):
    # match r'\[model\]\.projection_tile .* must be true, false, "mlp" or "all"; using false' and reads False
```

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -k projection_tile` → FAIL.
- [ ] **Step 3: Implement** `_projection_tile(model: dict) -> bool | str` (a bool, or exactly one of the two lowercase strings; `1` is not `True` here — test the type, not equality; anything else warns with `stacklevel=3` and returns `False`), the field with a comment in the style of the block above `projection_kernel` (what it serves, the two module forms, the copy and its 14 GiB rule, the four values, where it is refused), the `_KNOWN` entry and the `load_config` assignment.
- [ ] **Step 4: Run** `uv run pytest tests/test_config.py` → PASS; `uv run ty check` clean.
- [ ] **Step 5: Commit** `feat(config): add the projection_tile key`.

### Task 2: The slab layout, the slab-only conversion, the stacks and the test fixtures

**Files:**
- Create: `src/sous/engine/slab.py` (the layout and the marker: pure mlx, no sous imports)
- Create: `src/sous/engine/projtile.py` (conversion, revert, stacks, `fork_key_state`; the kernel, routing and `enable()` are added in later tasks)
- Create: `tests/projtile_fixtures.py`
- Test: `tests/test_projtile.py`

**Interfaces:**
- Produces, in `slab`:
  - `TILE = 256`, `UNIT_BYTES = 32`, `GROUP = 64`, `SLAB_ATTR = "_sous_slab"`
  - `@dataclass(frozen=True) class SlabInfo: n_real: int; k: int; tiles: int` with properties `n_pad` (`tiles * 256`) and `groups` (`k // 64`)
  - `repack(w, scales, biases) -> tuple[Any, Any, Any, SlabInfo]`: flat uint32 codes of `T·G·256·8` words, flat bf16 scales and biases of `T·G·256`, zero-padded, evaluated; `unpack(codes, scales, biases, info) -> tuple[Any, Any, Any]`: the inverse, `[n_real, K/8]`, `[n_real, G]`, `[n_real, G]`
  - `slab_info(module) -> SlabInfo | None` (`getattr(module, SLAB_ATTR, None)`)
- Produces, in `projtile`:
  - `STACK_ATTR = "_sous_tile"`; `@dataclass(frozen=True) class Stack: codes: Any; scales: Any; biases: Any; info: SlabInfo; slices: tuple[tuple[int, int], ...]; key: tuple[int, tuple[int, ...]]` (`key` = `(K, Ns)` with the members' real widths, in call order); `stack_of(module) -> Stack | None`
  - `convert_mlp(root) -> int` (gate, up, down of every layer: assigns the flat planes to `weight`/`scales`/`biases` through the module's own attribute assignment so mlx's parameter tree holds them, sets `_sous_slab` with `object.__setattr__`; one linear at a time, `mx.eval` after each, mlx's arrays dropped before the next, `mx.clear_cache()` at the end); `revert_mlp(root) -> int` (the inverse; deletes the marker)
  - `stack_groups(root) -> list[tuple[Any, ...]]` (per attention layer `(q, k, v)`; per GDN layer `(in_proj_qkv, in_proj_z)` and `(in_proj_b, in_proj_a)`; `(o_proj,)`; `(out_proj,)`), walking `root.model.layers` as `projkernel._dispatch_groups` does (`layer.is_linear`, `layer.linear_attn`, `layer.self_attn`, `layer.mlp`)
  - `build_stacks(root, *, keys_with_plans: Callable[[tuple], bool]) -> int` (concatenates each group's `w`, `scales`, `biases` along N in call order before padding, repacks once, records the members' `(start, stop)`, sets the one `Stack` on every member with `object.__setattr__`; a group whose `key` the predicate rejects is skipped; returns the bytes built); `drop_stacks(root) -> None`; `copy_bytes(root, *, keys_with_plans) -> int` (the sum the stacks would take: `tiles·256·K·9//16` per group, computable before building)
  - `fork_key_state(root) -> str`: `"active"` when **any** MLP linear carries `_sous_slab` (a half-converted model after a failed revert must key as converted), else `"off"`
  - Module level holds no arrays: no id-keyed maps, no `functools.cache` over arrays.
- Produces, in `tests/projtile_fixtures.py` (the data-layer part; later tasks add to it): `tiny_slab_model(*, seed=0, scope="all")` = `pfx.tiny_quantized_model(seed=seed)` (hidden 256, intermediate 512: every K ∈ {256, 512, 1024} and every MLP width is a multiple of 256, so no padding on the tiny model) after `convert_mlp` and, at `"all"`, `build_stacks` with every key accepted; `padded_module(n, k, seed)` = `pfx.rand_module` at a width that is not a multiple of 256.

- [ ] **Step 1: Write the failing tests** in `tests/test_projtile.py`:

```python
@pytest.mark.parametrize("n", [256, 4160, 48, 17408], ids=lambda n: f"n{n}")
def test_repack_round_trip_is_bitwise(n): ...                       # K = 1024: unpack(repack(w,s,b)) array_equal to w, s, b
def test_repack_addresses_match_the_slab_formula(): ...             # code byte of (n, k) at ((t*G+g)*256+r)*32 + j//2, low nibble first; scale at (t*G+g)*256+r
def test_padded_columns_are_zero(): ...                             # n = 48: planes past column 48 all zero
def test_convert_mlp_replaces_the_arrays_and_marks_the_module(): ...   # weight.ndim == 1 and size T*G*256*8; scales.ndim == 1; slab_info(m).n_real == N; module.parameters() holds the planes
def test_stock_quantized_matmul_raises_on_a_converted_module(): ...    # mx.quantized_matmul(...) and module(x) raise ValueError
def test_revert_mlp_restores_the_arrays_bitwise_and_clears_the_marker(): ...
def test_build_stacks_concatenates_before_padding_and_records_slices(): ...  # tiny model: q+k+v slices [(0,2048),(2048,2560),(2560,3072)], key (256,(2048,512,512)); b+a one tile, n_pad 256, key (256,(4,4))
def test_build_stacks_skips_a_group_without_a_plan(): ...           # keys_with_plans rejecting the b+a key: no Stack on in_proj_b/in_proj_a
def test_copy_bytes_is_the_stacks_padded_bytes(): ...               # equals what build_stacks returns, with the same predicate
def test_fork_key_state_follows_the_markers(): ...                  # off → active after convert → off after revert
def test_a_half_converted_model_keys_active(): ...                  # convert one layer's gate only → "active"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_projtile.py` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `slab.py`, the `projtile` data layer and the fixtures. The repack is `w.reshape(T, 256, G, 8).transpose(0, 2, 1, 3)` made contiguous then flattened, and `s.reshape(T, 256, G).transpose(0, 2, 1)` for both planes, after zero-padding N to `T·256`; `unpack` reverses the transposes and slices `[:n_real]`.
- [ ] **Step 4: Run** `uv run pytest tests/test_projtile.py` → PASS; `uv run ty check`, `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] **Step 5: Commit** `feat(engine): slab layout, slab-only MLP conversion and stacked copies for the projection tile`.

### Task 3: The weight descriptor and the slab bodies of the MMA and one-row kernels

**Files:**
- Modify: `src/sous/engine/projkernel.py` (`_weights`, `_as_weights` (new), `linears`, `linear`, `_launch`, `_dispatch`, `_launch_row`, `_row_dispatch`, `_kernel_spec`, `_projection_kernel`, `_header`, `_row_source`, `_row_kernel`, `_in_features`, `_dispatch_key`, `warm_up`, `_check_group`, `_check_rows`)
- Modify: `src/sous/engine/kernels/projection_mma.h` (`#ifndef SLAB` / `#define SLAB 0`), `src/sous/engine/kernels/projection_mma.metal` (`#if SLAB` around the address lines of the plain and staged bodies), `src/sous/engine/kernels/projection_row.metal` (a second body after a `// SLAB` marker line)
- Modify: `tests/proj_fixtures.py` (`stand_in_mma`, `stand_in_one_row` accept a slab entry)
- Test: `tests/test_projkernel.py`

**Interfaces:**
- Produces:
  - `class Weights(NamedTuple): w: Any; scales: Any; biases: Any; slab: SlabInfo | None` in `projkernel` (`SlabInfo` imported from `slab`, which has no sous imports). **`_weights(module)` returns the bare 3-tuple for an mlx-layout module and a `Weights` only for a slab-only one**, so every existing stand-in, probe and test that unpacks three fields or splats into `one_row(x2, w, s, b)` keeps working on unconverted models. Readers normalise with `_as_weights(entry) -> Weights` and never splat an entry.
  - `linears(x, weights)`: entries of one call must share the form (a mixed list raises `ValueError`); K and N come from the descriptor (`slab.k`, `slab.n_real`) when present, else from the arrays as today.
  - `_header(presum: bool, slab: bool)` prepends `#define SLAB 1` (the body's header defaults it to 0); `_kernel_spec(kind, nl, slab)` names the kernel `sous_proj_{kind}_n{nl}` or `sous_proj_{kind}_slab_n{nl}`; `_dispatch(rows, k, ns)` is unchanged (the template carries no SLAB entry); the output shape stays `(rows, n_real)` and the body writes columns `< NL_` as today.
  - `_launch_row(x2, w, scales, biases, slab: SlabInfo | None = None)`: the mlx form is the call `one_row(x2, w, scales, biases)` as today; the slab form is called with `slab=info` only, and runs `sous_proj_row_slab` (`_row_source()` splits `projection_row.metal` at `// SLAB`; grid `(256 * tiles, 1, 1)`, threadgroup `(256, 1, 1)`, output `(1, n_real)`).
  - `_in_features(module)` and `_dispatch_key(group)` read the descriptor; `_check_group` unpacks through `_as_weights`; `warm_up` unchanged in logic.
- Metal: in the plain and staged bodies, under `#if SLAB`: `wcol[f] = W_ + (ulong(n >> 8) * G * 256 + (n & 255)) * 32 + c * 8`, group stride `8192` bytes in place of `32`, `scol[f] = S_ + ulong(n >> 8) * G * 256 + (n & 255)` and `scol[f][ulong(g) * 256]` (bias alike), the staged loader `lsrc[i] = W_ + (ulong(n >> 8) * G * 256 + (n & 255)) * 32 + (ch >> 1) * 8192 + (ch & 1) * 16` with stage stride `8192`; `#else` today's lines. The slab one-row body is the slab spike's (`sous-spikes/slab/slabkern.py:ROW_SRC`): one lane per output column, `gsum[G]` in threadgroup memory from the fixed tree, per lane the four quarters `[(q·G)/4, ((q+1)·G)/4)`, each group's dot as the MMA order's fma chain from +0, the two-fma epilogue, `((p0 + p1) + p2) + p3`, one bf16 rounding; template `KD`, `ND`; `n < ND` guards the store.

- [ ] **Step 1: Write the failing tests** in `tests/test_projkernel.py` (`pfx.kernel` where the kernel runs; stand-in tests everywhere):

```python
def test_the_bodies_carry_a_slab_switch_and_the_row_kernel_a_slab_body(): ...   # '#if SLAB' in plain and staged; one '// SLAB' marker; '#define SLAB 0' default in the header; no ("SLAB", ...) template entry in _dispatch
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("k, n", [(1024, 256), (1024, 544), (2048, 1024)], ids=["k1024-n256", "k1024-n544", "k2048-n1024"])
def test_slab_bodies_equal_mlx_bodies(forced, layout, k, n): ...               # T = 1..8, every variant: array_equal; n544 exercises padding
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_slab_one_row_kernel_equals_the_mlx_one_row_kernel(layout): ...     # K 1024, N 256 and 544
def test_weights_is_a_bare_tuple_for_an_mlx_module_and_a_descriptor_for_a_slab_module(): ...
def test_linears_refuses_a_call_mixing_slab_and_mlx_forms(): ...
def test_the_descriptor_gives_k_and_n_for_a_converted_module(): ...             # _in_features, _dispatch_key on tiny_slab_model
def test_the_stand_ins_accept_a_slab_entry(): ...                               # stand_in_mma(x, [Weights(...)]) == stand_in_mma over unpacked arrays; stand_in_one_row(x2, w, s, b, slab=info) likewise
def test_warm_up_covers_a_converted_model(): ...                                # every key and one-row shape once, counters as found
```

- [ ] **Step 2: Run** `uv run pytest tests/test_projkernel.py -k "slab or descriptor or mixing"` → FAIL.
- [ ] **Step 3: Implement** the descriptor, the header define, the two kernel names per variant, the metal switch and the slab row body. Keep `_kernel_probe` (`kernel_available()`) on the mlx forms; `_check_rows` runs `one_row` against `mma` on whatever form the module holds.
- [ ] **Step 4: Run** `uv run pytest tests/test_projkernel.py tests/test_projkernel_contract.py` → PASS (every existing stand-in untouched).
- [ ] **Step 5: Commit** `feat(engine): slab addressing for the MMA and one-row projection kernels`.

### Task 4: The int8 GEMM on slab units and the 9-row floor for slab-only modules

**Files:**
- Modify: `src/sous/engine/int8prefill.py` (`qmm`, `linear_int8`, `_rows_ok`, `_wrap_mlp`, `_wrap_quantized_linear`, a `calls` dict), `src/sous/engine/kernels/q4_int8_gemm.metal` (`#ifndef SLAB` / `#define SLAB 0`, then `#if SLAB` around the weight and scale reads)
- Test: `tests/test_int8prefill.py`

**Interfaces:**
- Produces: `SLAB_MIN_ROWS = 9`; `calls: dict[str, int] = {"int8_small": 0}` (the 9–127-row calls served); `qmm(qa, sa, ra, weight, scales, biases, *, slab: SlabInfo | None = None)` — with a slab the kernel is `sous_int8_q4_gemm_slab` with header `common.h + "#define SLAB 1\n" + nax.h`, the planes are passed in place (no `mx.contiguous(scales.T)`), `n = slab.n_real`; without one, exactly today's call; `linear_int8(linear, x, stage=None)` reads `slab.slab_info(linear)` (module-level import of `slab` is fine) and passes `slab=` only when it is not None, reshaping to `n_real`; `_rows_ok(x, slab: SlabInfo | None = None) -> bool` = `x.ndim >= 2 and x.dtype in (bf16, fp16) and (x.shape[-2] >= MIN_ROWS or (slab is not None and x.shape[-2] >= SLAB_MIN_ROWS))`; `_wrap_mlp` passes `slab_info(self.gate_proj)` so a 9–127-row MLP chunk keeps the shared Stage A across gate and up; `_wrap_gdn` unchanged; `_wrap_quantized_linear` passes `slab_info(self)`. No change to `_warm_up` (int8 enables before anything is converted; Task 7's warm-up covers the slab GEMM).
- Metal: under `#if SLAB`: `wbase = w + (size_t((col_base + coord.y) >> 8) * groups * 256 + ((col_base + coord.y) & 255)) * words` with `w_stride8 = 8 * words`, `w_stride16 = kFragM * words` and the per-group step `size_t(g) * 256 * words`; `srow = scales + (size_t(n_run0 >> 8) * groups + g) * 256 + (n_run0 & 255)` (bias alike) with the two `vec<T,4>` loads at `+ h * kFragM`; `BN` = 64 divides 256 and `col_base` is a multiple of 32, so no read crosses a slab tile (assert `TILE % BN == 0` in Python beside the constants).

- [ ] **Step 1: Write the failing tests** in `tests/test_int8prefill.py`:

```python
@nax
@pytest.mark.parametrize("n", [1024, 4160], ids=["n1024", "n4160"])
@pytest.mark.parametrize("rows", [9, 64, 128])
def test_the_slab_gemm_equals_the_mlx_layout_gemm(n, rows): ...         # same qa/sa/ra; weight repacked: array_equal, random weights and the power-of-two exactness case
def test_rows_ok_admits_nine_rows_on_a_slab_only_module_and_not_otherwise(): ...   # (9, slab) ok; (8, slab) no; (9, None) no; (128, None) ok
def test_a_nine_row_mlp_call_on_a_converted_module_routes_to_int8_with_one_stage(): ...   # stand-in qmm counting calls and stage objects: gate and up share one Stage A; calls["int8_small"] += 1; an 8-row call does not route
def test_qmm_reads_slab_planes_in_place(): ...                            # with a slab, the kernel inputs are the flat planes (capture _kernel's inputs)
def test_the_gemm_source_defaults_slab_to_zero(): ...                     # source scan: '#ifndef SLAB' and '#if SLAB'
```

- [ ] **Step 2: Run** `uv run pytest tests/test_int8prefill.py -k "slab or nine"` → FAIL.
- [ ] **Step 3: Implement.** The stand-in `qmm` patches in this file keep their six-parameter form for unconverted models; the slab routing test's stand-in accepts `slab=` and unpacks through `slab.unpack`.
- [ ] **Step 4: Run** `uv run pytest tests/test_int8prefill.py` → PASS (nax tests skip on CI, run on the M5).
- [ ] **Step 5: Commit** `feat(engine): int8 GEMM reads slab-ordered MLP weights and takes their 9-row calls`.

### Task 5: The tile kernel, its availability and plans

**Files:**
- Create: `src/sous/engine/kernels/projection_tile.h`, `src/sous/engine/kernels/projection_tile.metal`
- Modify: `src/sous/engine/projtile.py` (kernel section), `src/sous/engine/kernels/__init__.py` (docstring)
- Test: `tests/test_projtile.py`

**Interfaces:**
- Produces, in `projtile`:
  - `@dataclass(frozen=True) class Plan: tn: int; s: int; groups: int; pipe: bool; splits: int = 1` with `label` (`TN128/S8/G80/pipe x8` style)
  - `PLANS: dict[tuple[int, tuple[int, ...]], Plan]`, keyed by `(K, Ns)` with the real widths in call order. **From `plans_tuned.json`'s arm-A cells (six):** `(5120, (12288, 1024, 1024))` → `Plan(128, 8, 80, True)`; `(6144, (5120,))` → `Plan(256, 8, 20, True, 2)`; `(5120, (10240, 6144))` → `Plan(128, 8, 80, True)`; `(5120, (48, 48))` → `Plan(128, 8, 2, True, 8)`; `(5120, (17408, 17408))` → `Plan(128, 4, 100, True)` (the dual); `(17408, (5120,))` → `Plan(256, 8, 20, True, 2)`. **The alone forms the window reaches through hook 2, at the spec's decision-5 values:** `(5120, (12288,))` → `Plan(128, 8, 80, True)`; `(5120, (1024,))` → `Plan(128, 8, 8, True)`; `(5120, (10240,))` → `Plan(128, 4, 80, True)`; `(5120, (6144,))` → `Plan(128, 8, 48, True)`; `(5120, (48,))` → the b+a plan over the shared tile; `(5120, (17408,))` → `Plan(256, 4, 68, True)`.
  - `MEASURED_CORES = frozenset({20})`; `plan_for(key) -> Plan | None`; `plan_for_call(linears) -> Plan | None` (the stack's `key` when every linear shares one stack; `(K, (n_real,))` for a slab-only or a lone stack member; `(K, (Ng, Nu))` for gate+up; None when any linear has no planes or the key has no plan); `check_plan(info, plan) -> str | None` (K % 256, n_pad % tn, tn ∈ {128, 256}, s ∈ {4, 8}, 1 ≤ splits ≤ K/256, the dual needs splits 1); `MLP_KEYS(root) -> list[key]` (the dual, gate/up alone, down for every distinct MLP shape)
  - `availability() -> nax.Availability` (the platform rule; then, remembered on success only, a compile and run on synthetic slab planes at **K = 2048** (so every split count in `PLANS` is legal), N = 512: the tile at every `DISTINCT_SHAPES = sorted({(p.tn, p.s, p.groups, p.pipe, p.splits) for p in PLANS.values()})` plus the dual, the reduce kernel, the slab MMA bodies, the slab one-row kernel and the int8 slab GEMM; the compiler report logged on failure, the reason one line)
  - `pad8(x) -> Any` (`[T, K]` → `[8, K]`, zero rows appended, no copy at T = 8); `run(x8, codes, scales, biases, info, plan) -> Any` (`[8, n_pad]` bf16; `splits > 1` runs the `SPLITOUT` kernel then the reduce); `run_dual(x8, gate_planes, up_planes, info, plan) -> tuple[Any, Any]`; the seams `tile = run` and `tile_dual = run_dual` (everything above `run`/`run_dual` — `verify`, `plain`, `warm_up`, the probe — calls the seams, so tests swap stand-ins in)
  - kernel names `sous_proj_tile_d{0|1}_s{0|1}` and `sous_proj_tile_reduce`
- The port: `projection_tile.h` = `sous-spikes/p126/kern126.py:HEADER` (lines 38–178: `q4_store_input_sums`, `q4_traversal`, `VISIT`, `STORE_ELEM`, `RUN_GROUP`, `FINISH_GROUP` — which holds the refill barrier — and the `DUAL`/`SPLITOUT` switches, set per kernel name as header defines) and `projection_tile.metal` = `BODY` (180–273, holding the prologue and closing barriers) then a `// REDUCE` line then `REDUCE_BODY` (275–282); template constants `N, K, TN, S, GROUPS, PIPE, SPLITS`; the `NATIVE` arm-B switch is dropped, which fixes `CSTRIDE = 64`, `GSTRIDE = 8192` and arm A's `tw0`/`tw1` tile base `wb0 + tile*G*256*32 + tile_offset*32`; the epilogue stays sous's two fmas; `#pragma clang fp reassociate(off)` as the spike. Grid `(GROUPS·S·32, SPLITS, 1)`, threadgroup `(S·32, 1, 1)`, `output_shapes=[(8, n_pad)]` bf16 or `[(splits, 8, n_pad)]` fp32 under `SPLITOUT`; reduce grid `(ceil(8·n_pad/256)·256, 1, 1)`, threadgroup 256.
- `tests/projtile_fixtures.py` gains: `TILE = projtile.availability()`, `tile_nax` mark; `TINY_PLANS` mapping the tiny model's keys to `Plan(128, 4, 2, True)` — q+k+v `(256, (2048, 512, 512))`, gate+up `(256, (512, 512))`, qkv+z `(256, (512, 256))`, b+a `(256, (4, 4))`, down `(512, (256,))`, o `(1024, (256,))`, `(256, (256,))`, `(256, (512,))`, `(256, (2048,))`, `(256, (4,))`; `tiny_plans(monkeypatch)` sets `projtile.PLANS` to `{**PLANS, **TINY_PLANS}`; `stand_in_tile(x8, codes, scales, biases, info, plan)` (unpack, then `pfx.stand_in_mma`'s arithmetic over the padded rows, `[8, n_pad]`) and `stand_in_tile_dual`.

- [ ] **Step 1: Write the failing tests** in `tests/test_projtile.py`:

```python
def test_the_six_fused_plans_match_the_spike_and_the_alone_forms_the_spec(): ...   # the twelve entries above; plan_for((5120,(248320,))) is None; plan_for((4096,(256,))) is None
def test_plan_for_call_needs_planes_and_a_plan(): ...
def test_check_plan_refuses_what_the_kernel_cannot_take(): ...
def test_measured_cores_is_the_m5_pro(): assert projtile.MEASURED_CORES == frozenset({20})
def test_the_kernel_sources_split_at_reduce_and_carry_three_barriers(): ...   # one '// REDUCE' in the .metal; 'threadgroup_barrier(' count == 2 in the .metal (prologue, closing) and 1 in the .h (FINISH_GROUP's refill)
def test_pad8_appends_zero_rows_and_keeps_eight(): ...
@tile_nax
@pytest.mark.parametrize("shape", DISTINCT_SHAPES)
@pytest.mark.parametrize("n", [512, 96, 4160], ids=lambda n: f"n{n}")
def test_every_row_is_independent_of_the_row_count(shape, n): ...     # K = 2048 random planes: rows at T=1..8 array_equal to T=1 (pad8); NaN in row 3 leaves the others unchanged; columns past n_real zero
@tile_nax
def test_the_split_reduce_adds_partials_in_order(): ...               # SPLITOUT partials summed in Python 0..S-1 then bf16 == run()
@tile_nax
def test_the_dual_launch_equals_two_single_launches(): ...
@tile_nax
def test_the_tile_is_at_the_bf16_floor_against_an_fp32_reference(): ...   # rel RMS <= TILE_FLOOR_FACTOR * floor (scales/biases upcast to fp32)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_projtile.py -k "plan or kernel or pad8 or reduce"` → FAIL.
- [ ] **Step 3: Port the kernel and implement** the table, `plan_for_call`, `check_plan`, `availability()`, `pad8`, `run`, `run_dual`, the seams, the fixtures. Register `projection_tile.metal`/`.h` in `kernels/__init__.py`'s docstring (origin: Splash `q4_mpp_tiles.h` at 0d5a035).
- [ ] **Step 4: Run** on the M2 (everything but `tile_nax` passes, the rest skip) and on the M5 (`uv run pytest tests/test_projtile.py -rs` with no skip, under `gpu_run_m5.py`).
- [ ] **Step 5: Commit** `feat(engine): the MPP projection tile kernel at its tuned plans`.

### Task 6: Routing — the window, the round flag, the two seams

**Files:**
- Modify: `src/sous/engine/projtile.py` (routing section), `src/sous/engine/projkernel.py` (`calls`, `_serve_verify`, `_serve_plain`), `tests/projtile_fixtures.py` (the guard and `window()`)
- Test: `tests/test_projtile.py`, `tests/test_projkernel.py`

**Interfaces:**
- Produces, in `projtile`: `_active: int`, `_window: Window | None` (`Window(serial: int)`), `_round_flag: int`, `_scope: str | None`; `open_window() -> None` (also sets `_round_flag = 1`), `close_window() -> None` (never raises), `set_round_flag(value: int) -> None`; `serves(rows: int, *, hook: int) -> bool` = `_active and _window is not None and _round_flag and (hook == 1 or 2 <= rows <= MAX_ROWS)`; `verify(linears, x, plan) -> tuple[Any, ...]` (x `[1, T, K]`; one launch per stack or dual; slices back to each linear's `[1, T, n_real]`); `plain(module, x, plan) -> Any` (**a stack member launches only its own tiles**: its `(start, stop)` is a whole-tile range for q, k, v, in_proj_qkv and in_proj_z, so the launch reads the slice of the flat planes `[t0·G·256·8, t1·G·256·8)` at the member's alone plan; in_proj_b and in_proj_a run the b+a stack and slice); `clear() -> None` (flag, window, round flag, scope)
- `projkernel.calls` gains `"verify_tile"` and `"plain_tile"`. `_serve_verify`, after its own conditions and its cut into runs of at most 8 rows: `plan = projtile.plan_for_call(linears)` once per call; each run is served by `projtile.verify(linears, run, plan)` when `plan is not None and projtile.serves(run_rows, hook=1)`, else `_run_mma(run, weights)`; `verify_tile` counts served runs, `verify_kernel` the call as today. `_serve_plain`: `plan = projtile.plan_for_call((module,))`; rows 2–8 with `plan is not None and projtile.serves(rows, hook=2)` → `projtile.plain(module, x, plan)`, `plain_tile += 1`; one row → the one-row kernel; rows 9–127 → `None` as today (int8's wrapper takes a slab-only module); otherwise the MMA kernel. The drafter's own linears (`_DRAFT_TAG`) never reach the tile. The GDN four-linear call arrives as one `_linears` call: `plan_for_call` of a call whose linears span two stacks (qkv+z and b+a) returns the pair's plans and `verify` issues the two launches.
- `tests/projtile_fixtures.py` gains `guard_tile_state(monkeypatch)` (restores `_active`, `_window`, `_round_flag`, `_scope`, `PLANS`, `projkernel.calls`, `int8prefill.calls`), `window()` (a context manager opening and closing the window), `hooked_tile(monkeypatch)` (= `pfx.hooked` + `tiny_plans` + `projtile._active = 1` + the stand-in seams).

- [ ] **Step 1: Write the failing tests** (`tests/test_projtile.py` unless noted):

```python
def test_serves_needs_the_flag_the_window_and_the_round_flag(): ...                 # table over the three bits and hooks 1/2 at rows 1, 2, 8, 9
def test_open_window_resets_the_round_flag_and_close_window_never_raises(): ...
def test_verify_slices_a_stack_back_to_its_members(): ...                           # q+k+v call → one stand-in launch, three outputs of real widths
def test_the_gdn_call_runs_as_two_launches(): ...
def test_plain_on_a_stack_member_launches_its_own_tiles(): ...                      # q alone: the stand-in sees planes of q's tile range and the alone plan; in_proj_b: the b+a stack
# in tests/test_projkernel.py, with hooked_tile and the stand-ins:
def test_the_verifier_hook_serves_every_run_on_the_tile_while_the_window_is_open(): ...   # T = 5 (one run), T = 12 (8 + 4) and T = 9 (8 + 1): verify_tile == runs, no MMA launch
def test_a_call_with_an_unservable_linear_stays_on_the_mma_kernel_whole(): ...     # scope mlp (no stacks): the GDN call → verify_kernel, verify_tile unchanged
def test_a_key_outside_the_plan_table_is_served_by_the_mma_kernel(): ...           # PLANS without the tiny gate+up key: the MLP call → MMA whole
def test_the_module_hook_routes_two_to_eight_rows_to_the_tile_inside_the_window_only(): ...  # 1 row → one_row; 2..8 → plain_tile; outside the window → plain_kernel
def test_nine_to_127_rows_on_a_slab_only_module_fall_through_to_int8(): ...        # _serve_plain returns None; int8 stand-in qmm called; int8prefill.calls["int8_small"] += 1
def test_the_round_flag_closes_the_tile_for_one_round(): ...                        # set_round_flag(0) → MMA path; open_window() → tile again
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_projtile.py tests/test_projkernel.py tests/test_projkernel_contract.py` → PASS.
- [ ] **Step 5: Commit** `feat(engine): route verify runs and short decode calls to the projection tile inside a decode window`.

### Task 7: `enable()`, the guards, the memory rule, the warm-up, the probe, `static_reason`

**Files:**
- Modify: `src/sous/engine/projtile.py` (enable section), `tests/projtile_fixtures.py` (`tile_ready`)
- Test: `tests/test_projtile.py`

**Interfaces:**
- Produces: `TILE_COPY_MIN_BUDGET_GIB = 14`, `TILE_FLOOR_FACTOR = 1.05`;
  `enable(model, *, config_value: bool | str, kernel_status: dict, tile_status: dict, int8_status: dict, drafter_kind: str | None, cache_budget: int | None, headroom_bytes: int | None, reserve_bytes: int) -> dict` returning the one five-key shape `{"state": "off"|"unavailable"|"active", "scope": "all"|"mlp"|None, "reason": str|None, "copy_gb": float|None, "probe_seconds": float|None}` (`copy_gb = copy_bytes / 1e9`, the status and load-line figure);
  `copy_decision(config_value, cache_budget: int | None, headroom_bytes: int | None, copy_bytes: int, reserve_bytes: int) -> tuple[bool, str]` — all bytes; `"mlp"` → `(False, "scope mlp by config")`; `"all"` → `(True, "scope all by config; budget left X GiB")`; `True` with `headroom_bytes is None` → `(False, "headroom unreadable")`; `True` with `cache_budget is None` → copy iff `headroom − copy − reserve − CACHE_BUDGET_SLACK ≥ TILE_COPY_MIN_BUDGET_GIB·2**30`, else `(False, "copy would leave X GiB of prompt-cache budget (< 14)")`; `True` with an explicit budget → copy iff `headroom − copy − reserve − CACHE_BUDGET_SLACK ≥ cache_budget`, else `(False, "configured budget X GiB does not fit beside the copy")`; the text always carries the budget left in GiB;
  `_refuse(reason, probe_seconds=None, *, expected=False)` in the kernel's style (`projection tile unavailable: …` / `sous: projection tile unavailable (…); verify runs the projection kernel`); `_pins_reason() -> str | None` (tests patch it); `_compile_real_shapes(root) -> None` (the slab MMA bodies at T = 1..8, the slab one-row kernel and the int8 slab GEMM on synthetic planes at every distinct MLP `(K, n_real)` the model holds, **before** anything is converted); `warm_up(root, scope) -> None` (every `(key, plan)` the model reaches on the tile and the dual, the slab bodies per MLP shape, the int8 slab GEMM once per distinct slab `(K, n_real)` — gate/up `(5120, 17408)` and down `(17408, 5120)` on the 27B; M is a runtime input, so one row count suffices); `probe(root, scope) -> str | None`; `static_reason(config, model_config, drafter_kind) -> str | None` (`projection tile off` when `False`; then `projkernel.static_reason`; then `no drafter`; then `int8 prefill off` when `not config.int8_prefill`; then the core count via `gpucores.read()` against `MEASURED_CORES`; then `no tuned plan for <key>` for the MLP keys read from `model_config` — `hidden_size`, `intermediate_size`); `static_copy_bytes(model_config) -> int` (the attention and GDN stacks' padded bytes from `config.json`: `num_hidden_layers`, the linear/full layer split (`layer_types` or `full_attention_interval`), `hidden_size`, `num_attention_heads × head_dim`, `num_key_value_heads × head_dim`, `linear_num_key_heads × linear_key_head_dim`, `linear_num_value_heads × linear_value_head_dim`)
- Order inside `enable()`: clear state → config (`False` → `off`) → kernel → tile → drafter → int8 → cores → model (eligibility; every key of `MLP_KEYS(root)` in `PLANS`, else INFO `no tuned plan for <key>`) → `_pins_reason` (warning) → `availability()` (warning) → `_compile_real_shapes` (warning) → `copy_decision` → `convert_mlp` → `build_stacks(root, keys_with_plans=lambda key: plan_for(key) is not None)` when copying → `warm_up` → `probe`; a probe failure or an exception clears the exception's frames, then `revert_mlp` and `drop_stacks`, then one `warnings.warn` plus `unavailable`; if the revert raises, the markers stay authoritative. Success sets `_active = 1`, `_scope`, logs `projection tile: scope=<s> copy=<x.y> GB budget left <y.y> GiB probe <t>s`.
- **The probe** (real linears: layer 3's q+k+v, o, gate+up, down; layer 0's qkv+z, b+a, out_proj — on the tiny model the corresponding modules): on the tile seams, every row at T = 1..8 `array_equal` to the same row at T = 1 through the same plan (the dual and the two-launch GDN form included); accuracy against a float32 reference from `unpack`ed planes dequantised with scales and biases upcast to fp32: the floor is that reference rounded to bf16, the tile's rel RMS ≤ `TILE_FLOOR_FACTOR` × floor on every probed linear; a NaN planted in one row leaving the others bitwise unchanged; on the slab-only MLP linears the slab MMA bodies at T = 1..8, the slab one-row kernel and the int8 slab GEMM at 9, 64 and 128 rows each `array_equal` to the mlx-layout kernels over the `unpack`ed planes; one `_feed_forward` and one plain MLP call through the hooks inside a forced window showing `verify_tile` and `plain_tile`, then the same call outside it on the MMA path. It restores `calls`, leaves `_active` 0 and `mx.clear_cache()`s in `finally`.
- `tests/projtile_fixtures.py` gains `tile_ready(monkeypatch) -> dict` modelled on `pfx.proj_ready`: calls `pfx.proj_ready(monkeypatch)` (platform rule, 20 cores, projkernel stand-ins, pins), sets `projtile.availability` to `lambda: nax.Availability(True)`, `projtile._pins_reason` to `lambda: None`, `projtile._compile_real_shapes` to a no-op, `projtile.tile`/`tile_dual` to the stand-ins, `int8prefill.qmm` to a slab-aware reference (`tests/test_int8prefill.py`'s `_reference_qmm` over `slab.unpack`, moved into the fixtures), applies `tiny_plans`, and returns active `kernel_status`, `tile_status` and `int8_status` dicts.

- [ ] **Step 1: Write the failing tests** in `tests/test_projtile.py`:

```python
def test_enable_off_returns_before_any_guard(): ...
def test_enable_names_the_first_failing_guard_in_order(): ...          # kernel, tile, drafter, int8, cores, model/plan, pins, availability, real shapes — INFO vs warning as the spec says
@pytest.mark.parametrize("case", COPY_CASES, ids=...)
def test_copy_decision_table(case): ...   # bytes; G = 2**30: (True, None, 35G, 4G, 16G) → False (13 < 14); (True, None, 44G, 4G, 16G) → True; (True, 12G, 35G, 4G, 16G) → True; (True, 14G, 35G, 4G, 16G) → False; (True, 0, 20G, 4G, 16G) → False, no warning; (True, None, None, 4G, 16G) → False 'headroom unreadable'; ("mlp", ...) → False; ("all", None, 20G, 4G, 16G) → True with the budget left in the text
def test_enable_converts_the_mlp_and_builds_stacks_at_scope_all(): ...  # tile_ready; markers set; status {"state": "active", "scope": "all", ...}; copy_gb == copy_bytes/1e9
def test_enable_serves_mlp_only_when_the_copy_is_skipped(): ...         # scope mlp, reason names the shortfall, no stacks
def test_a_failed_probe_reverts_the_conversion_bitwise_and_drops_the_stacks(): ...   # arrays equal the pre-enable arrays; markers gone; unavailable with a warning
def test_enable_compiles_the_real_shapes_before_converting(): ...       # a failing _compile_real_shapes leaves the model untouched (no marker ever set)
def test_no_plane_outlives_its_model(): ...                              # weakref to a stack's codes and a converted weight; del model; gc.collect(); both dead
def test_probe_fails_t_invariance_floor_and_isolation_for_broken_stand_ins(): ...  # three stand-ins, three reasons
def test_warm_up_runs_the_int8_slab_gemm_once_per_mlp_shape(): ...
def test_static_reason_mirrors_enable(): ...                             # off, the kernel's reason, no drafter, int8 off, cores, no tuned plan
def test_static_copy_bytes_matches_copy_bytes_on_the_tiny_model(): ...
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_projtile.py` → PASS; ty and ruff clean.
- [ ] **Step 5: Commit** `feat(engine): enable the projection tile behind its guards, memory rule and probe`.

### Task 8: Engine wiring — load order, the window in `decode()`, the fork key, the status and the load line

**Files:**
- Modify: `src/sous/engine/vlm.py` (constructor parameter `projection_tile: bool | str = False`; `projtile.enable(...)` after `enable_drafter` and before `kv8.enable`, passing `cache_budget`, `live_headroom()` (bytes or None) and `reserve_bytes`; `self._ptile = projtile.fork_key_state(root)` right after; `decode()` opens the tile window beside kv8's and closes it first in the `finally`, in its own `try/finally`; `prefill()`'s first statement raises `RuntimeError("projection tile window open across a prefill")` when `projtile._window is not None`; `_fork_key_fields` passes `ptile=getattr(self, "_ptile", "off")`; `unload()` calls `projtile.clear()` beside `projkernel.clear()`)
- Modify: `src/sous/engine/base.py` (`_default_factory(..., projection_tile: bool | str = False)`, `default_engine_factory` passes `config.projection_tile`; `ManagedEngine.projection_tile_status` property; the load line after `projection_kernel_drafter=` and before `draft_block=`: ` projection_tile=<state>` and, when active, `:<scope> projection_tile_probe_s=<s> projection_tile_budget_gib=<y.y>` plus, at scope `all`, ` projection_tile_copy_gb=<x.y>` and, at `mlp`, ` projection_tile_copy=skipped`; the status document gains `out["projection_tile"] = {k: tile.get(k) for k in ("state", "scope", "reason", "copy_gb", "probe_seconds")}`; the stale "an unset one is 5 where the kernel is active" comment becomes 8)
- Modify: `src/sous/engine/lm.py` (`self.projection_tile_status = {"state": "off", "scope": None, "reason": None, "copy_gb": None, "probe_seconds": None}`; `fork_key_fields(..., ptile="off")`)
- Modify: `src/sous/engine/forkstore.py` (`fork_key_fields(..., ptile: str)` → `fields["ptile"] = ptile`; the docstring's last sentence amended; `_EPOCH_FILES` += `"slab.py", "projtile.py"`; `_EPOCH_EXCLUDED_KERNELS` += `("projection_tile.metal", "projection_tile.h")`)
- Create: `tests/test_projtile_engine.py` (CI: the engine built as `tests/test_kv8.py::_dw_engine` does — `object.__new__(VLMEngine)` with `mlx_vlm` stubbed through `monkeypatch.setitem(sys.modules, "mlx_vlm", ...)`)
- Test: `tests/test_engine_base.py`, `tests/test_forkstore.py`, `tests/test_engine_forkhooks.py`, `tests/test_projtile_engine.py`

**Interfaces:**
- Consumes: Task 7's `enable` and status dict, Task 6's window; `enable()` needs a budget `reason` text with the budget left for the load line's `projection_tile_budget_gib` — `enable()` returns it in the status as `"budget_left_gib": float | None` (a sixth key the status document does not copy).

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_engine_base.py
def test_the_load_line_carries_the_projection_tile_after_the_drafter_route(): ...   # 'projection_kernel_drafter=active projection_tile=active:all projection_tile_probe_s=1.2 projection_tile_budget_gib=13.1 projection_tile_copy_gb=4.1 draft_block=8'
def test_the_load_line_prints_the_mlp_scope_and_unavailable(): ...                   # 'projection_tile=active:mlp ... projection_tile_copy=skipped'; 'projection_tile=unavailable'
def test_the_status_document_reports_the_projection_tile_with_get(): ...             # five keys; an LM-style off dict does not raise
def test_default_engine_factory_hands_the_tile_value_to_the_vlm_engine(): ...        # True, False, "mlp", "all"
# tests/test_forkstore.py
def test_the_key_changes_with_ptile_and_the_epoch_hashes_slab_and_projtile_but_not_the_tile_kernels(): ...
# tests/test_engine_forkhooks.py
def test_vlm_key_fields_read_ptile_off_without_a_model(): ...        # the VLMEngine.__new__ stub without _ptile
def test_lm_key_fields_carry_ptile_off(): ...
# tests/test_projtile_engine.py
def test_prefill_refuses_an_open_tile_window(): ...                  # projtile._window set; prefill raises RuntimeError before _loaded()
def test_decode_closes_the_tile_window_even_when_kv8s_close_raises(): ...   # kv8._active = 1, kv8.close_window raising, projtile._active = 1; afterwards projtile._window is None and the error propagates
def test_unload_clears_the_tile(): ...
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the wiring in the four files; keep `vlm.py`'s load-sequence comment style (one paragraph on why the tile sits after the drafter route and before kv8, the fork key and the budget).
- [ ] **Step 4: Run** `uv run pytest -m "not model"` → PASS; ty, ruff, `uv lock --check` clean.
- [ ] **Step 5: Commit** `feat(engine): wire the projection tile into the load, the decode window, the fork key and the status`.

### Task 9: `sous tune` — the suite key, the scope report, the int8 stage arm, the notice

**Files:**
- Modify: `src/sous/tune/arms.py` (`Arm.projection_tile: bool | str = True`; `Arm.suite_key` gains it — **not** `Arm.key`, the bench key bench rows and resume match on; `_arm` passes `user.projection_tile`; `winner_stage_arms`'s int8 arm sets both `projection_tile=False` and `config=dataclasses.replace(winner.config, int8_prefill=True, projection_tile=False)`; `_arm`/`quick_arms` pass `tile_copy_bytes = projtile.static_copy_bytes(target.config)` to `fit()` only when `projtile.static_reason(user, target.config, drafter_kind) is None`, else `None`), `src/sous/tune/candidates.py` (`fit(..., tile_copy_bytes: int | None = None, tile_config: bool | str = False, cache_budget: int | None = None)`; `Fit.tile_scope: str | None` from `projtile.copy_decision` with headroom = `working_set − target.bytes − drafter.bytes / DRAFTER_QUANT_DIVISOR`; `needed` and the window unchanged; the detail line names the scope), `src/sous/tune/decide.py` (`ArmSummary.key` widened), `src/sous/tune/suite/runner.py` (`SuiteRun.projection_tile`, `SuiteRun.key`, `from_dict`'s `"projection_tile": False` default; `_check_tile(arm, engine, out)` printing `_notice_unavailable` except when the reason starts with `no drafter`)
- Test: `tests/test_tune_arms.py`, `tests/test_tune_candidates.py`, `tests/test_tune_decide.py`, `tests/test_tune_runner.py`, `tests/test_tune_suite.py`

- [ ] **Step 1: Write the failing tests** (one per file: `suite_key`/`SuiteRun.key`/`ArmSummary.key` carry the flag and old rows read `False`; `Arm.key` is unchanged; the int8 stage arm's Arm and config both read `False`; `fit()` reports `tile_scope` without changing `window` or `needed`, and `None` when `tile_copy_bytes is None`; the runner prints no notice for `no drafter` and one for any other `unavailable`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** the tune tests → PASS. **Step 5: Commit** `feat(tune): key suite rows by projection_tile and report the copy's scope`.

### Task 10: Documentation, the contract rewrite and the notices

**Files:**
- Modify: `CLAUDE.md` (one `engine/projtile.py` bullet in the kernel's style: what the tile serves and when, the window and its close order, the two module forms and `slab.py`, the stacks, the plan table and `MEASURED_CORES`, the guard order and the two refusal kinds, the memory rule in bytes, the probe and `TILE_FLOOR_FACTOR`, the counters, the fork-key field and the epoch entries, the import rule; the contract sentences in the `verifyattn`, `tileattn`, `projkernel` and `kv8` bullets become conditional as the spec's "The contract, restated" words them; the fork-store sentence names `slab.py`, `projtile.py` and the excluded tile files), `docs/configuration.md` (the key, its four values, the memory arithmetic at 131072 and 262144 on a 64 GB Mac, where it is refused, the int8 path for 9–127-row MLP chunks, stage 2; **and** the three contract paragraphs under `attention_tile`, `projection_kernel` and `kv_int8` made conditional), `docs/observability.md` (load-line keys, status keys, counters), `docs/tuning.md` (the suite key, `fit()`'s scope, the int8 stage arm), `docs/how-it-works.md` (the fork-key list gains `ptile`), `THIRD_PARTY_NOTICES.md` (Splash `q4_mpp_tiles.h` at 0d5a035; the slab layout attributed for `projection_mma.metal`'s `SLAB` bodies, `projection_row.metal`'s slab body, `q4_int8_gemm.metal`'s slab address and `slab.py`'s repack; the "reads mlx's own weights where Splash repacks weights into 256-column tiles" sentence corrected), `src/sous/config.py` comments on `attention_tile`, `projection_kernel` and `kv_int8` (the conditional contract). The README changes only in Task 12 step 4, with the gate's figure.

- [ ] **Step 1: Write the docs.** **Step 2:** `grep -rn "drafter on or off\|without it\|equals output without\|identical to output without" CLAUDE.md docs src/sous/config.py README.md` and make every hit conditional on the tile (the README hits are listed for Task 12). **Step 3:** `uv run ruff format --check .` on the touched Python. **Step 4: Commit** `docs: the projection tile, its memory rule and the block-invariance contract`.

### Task 11: The model tests (M5 Pro)

**Files:**
- Create: `tests/test_projtile_model.py` (`pytestmark = pytest.mark.model`)
- Keep unchanged: `tests/test_projkernel_model.py`, `tests/test_tileattn_model.py`, `tests/test_kv8_model.py`

**Interfaces:** engines built as `VLMEngine(..., int8_prefill=True, attention_tile=True, projection_kernel=True, projection_tile="all", kv_int8=True, kv_int8_window=131072, cache_budget=12 * 2**30, ...)` and a second at `"mlp"`; assert `kv_int8_status["state"] == "active"` beside the tile's scope.

- [ ] **Step 1: Write** `test_greedy_output_is_block_invariant_on_the_tile` (two prompts, about 1,450 and 57,560 tokens, 256 tokens: blocks 2, 12 — set on the engine — and 0 against 8, identical token ids; `calls["verify_tile"]` moved on the block-12 run's split runs; at both scopes), `test_the_tile_sits_at_the_bf16_floor_on_every_linear` (the spike's `acc126` ported: float64 reference, element bound, `TILE_FLOOR_FACTOR`), `test_the_slab_only_mlp_paths_equal_a_tile_off_engine` (gate/up/down outputs at 1, 5, 8 and 128 rows `array_equal`; at 9, 64 and 127 rows within `REL_RMS_BOUND` of stock), `test_drafter_on_against_drafter_off_within_the_measured_perturbation` (the gate's step-6 rule on the two prompts; the logits come from a recording entry injected into `stream_generate`'s `logits_processors` — `generate_step` hands each processor the raw `outputs.logits[:, -1, :]` before the logsumexp — and, for verify rows, from a wrapper installed over `mlx_vlm.speculative.dflash._dflash_verify_greedy` that records `verify_out.logits` before its sampler; D and the margins asserted), `test_the_status_and_load_line_name_the_scope`.
- [ ] **Step 2: Run on the M5** under `gpu_run_m5.py`, each file in its own process: `tests/test_projtile_model.py`, then `test_projkernel_model.py`, `test_tileattn_model.py`, `test_kv8_model.py`, `test_int8prefill_model.py` → PASS.
- [ ] **Step 3: Commit** `test: block invariance, the bf16 floor and the slab paths on the 27B`.

### Task 12: The gate harness (out of tree, never committed) and the merge

**Files (create under `~/code/sous-spikes/tile-gate/`, mirrored to the M5's `~/code/personal/sous-spikes/tile-gate/`; the branch tree at `~/code/personal/sous-remote-tile` with its own venv):**
- `README.md` — the spec's merge gate copied verbatim as the pre-registration, with run labels, seeds, done-files, and the cold-load procedure (the Metal shader cache directory removed before the cold load, as `p148/gate/loadtime.py`'s cold mode does); amendments only with a timestamp.
- Copied from `b8d-gate`: `gate157.py`, `replay-157.json`, `picks145.txt`, `build_replay.py` (step 3's turns), `gate132.py` (steps 5–7b's 68 turns), `rounds_drafter.py` → `rounds_tile.py` (arm `8t` calls `projtile.set_round_flag(1)` and `8k` `set_round_flag(0)` in `_next_block`; the PINNED mlx-vlm hashes, timing points and record fields kept; `draft_calls` replaced by `tile_calls` = the `verify_tile` delta), `wrap_b8d.py` → `wrap_tile.py` (installs the recorder; the 7b mode wraps `projtile.open_window` to set the flag back to 0; the KV-hash recorder after every `prefill()` — SHA-256 of each layer's K and V up to its offset and of each recurrent state — with the `int8_small` delta; the raw top-2 logit tracer as Task 11 describes; the `projtile.enable` timer for step 2's four modes).
- Copied from `kv8-gate`: `analyze133.py` → `analyze_abba.py` (4a), `suite_kv8.py` → `suite_tile.py` and `quality_m5.sh` (step 8, arms `"all"`/`false`), `tests_m5.sh` (step 1).
- New: `porteq126.py` (adds `~/code/sous-spikes/p126` to `sys.path`; every dispatch key at T = 1..8 through `projtile.run`/`run_dual` and `kern126`'s arm A at the same plan; `PORT == SPIKE` per key; the SHA-256 of `projection_tile.metal` and `.h`), `bench_int8small.py` (4c: 9/64/127-row MLP calls through int8 on slabs against stock on main's layout), `analyze_tile.py` (= `analyze157.py` with REF `8k` and the step-3 rule), `judge_tile.py`, `check_tile.py` (validity, the scope on every load line), `run_tile.sh`, `chain_tile.sh`, `judge_all.sh`; the M2 step (the 9B with its drafter, #143's 12-turn set) and `scripts/api_smoke.py` on both Macs are shell lines in `judge_all.sh`'s README section.

- [ ] **Step 1: Write the harness.** **Step 2: Smoke** one 2-turn run on the M5 at each scope, the daemon idle. **Step 3: Pre-register** (README timestamped) and run `chain_tile.sh all`; judge. **Step 4:** With the gate's block-invariance figure, rewrite README.md's "Speed never changes the answer" paragraph and its kv8 results line on the branch and commit `docs: the README's answer-stability claim under the projection tile`. **Step 5:** Put the verdict, the numbers and the `PORT == SPIKE` lines in the PR body. The maintainer merges.

---

## Self-review notes

- Spec coverage: decisions 1–8 → Tasks 6/1/2–4/7/5/8 (block default untouched)/8/9; Design subsections → Tasks 2–8; Testing → Tasks 2–8 (CI) and 11 (model); Merge gate → Task 12; Documentation → Tasks 10 and 12; the contract rewrite → Task 10, with the three existing model tests kept (Task 11).
- Names used across tasks: `slab.SlabInfo`/`slab_info`/`repack`/`unpack`/`SLAB_ATTR`; `projtile.Stack`/`stack_of`/`convert_mlp`/`revert_mlp`/`build_stacks`/`drop_stacks`/`copy_bytes`/`static_copy_bytes`/`fork_key_state`/`Plan`/`PLANS`/`plan_for`/`plan_for_call`/`check_plan`/`MLP_KEYS`/`MEASURED_CORES`/`availability`/`pad8`/`run`/`run_dual`/`tile`/`tile_dual`/`serves(rows, *, hook)`/`open_window`/`close_window`/`set_round_flag`/`verify(linears, x, plan)`/`plain(module, x, plan)`/`clear`/`enable`/`copy_decision`/`_pins_reason`/`_compile_real_shapes`/`warm_up`/`probe`/`static_reason`; `projkernel.Weights`/`_as_weights`/`calls["verify_tile"|"plain_tile"]`; `int8prefill.SLAB_MIN_ROWS`/`calls["int8_small"]`/`qmm(..., slab=)`/`_rows_ok(x, slab=None)`; `projection_tile_status` (five keys plus `budget_left_gib`); `ptile`.
- Review Focus items 1–5 are pinned in Tasks 8, 2/3/4/5, 7, 6 and 7.
- Changes from the first draft (the review of 2026-10-17): `slab.py` breaks the import cycle; `_weights` keeps the bare 3-tuple for mlx modules so no stand-in changes; the SLAB switch is a header define; the plan is part of servability and the model guard refuses an MLP key without one; stack members launch their own tiles at the alone plans; `TINY_PLANS`/`tiny_plans` and `tile_ready` make the CI tests runnable; `copy_decision` works in bytes and names the headroom-unreadable case; the status has one five-key shape everywhere and the load line prints the budget left; `fork_key_state` is `active` on any marker; the engine tests move to a stubbed-engine CI file; `_wrap_mlp` shares one Stage A for the 9–127-row chunks; the int8 slab GEMM is warmed per MLP shape by `projtile.warm_up`; `Arm.suite_key` (not `key`) and the int8 stage arm's config; `fit()`'s inputs named; kern126 line ranges and the barrier count corrected; the harness file list follows the b8d and kv8 precedents; the README rewrite is a gate step.
