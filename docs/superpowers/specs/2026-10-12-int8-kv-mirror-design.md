# sous — INT8 K/V through the attention tile (the `kv_int8` mirror)

*2026-10-12. The in-tree design for the L4 lever of the block-5 round profile (profile report of 2026-10-12, the maintainer's `sous-spikes/prof-b5/results.md`) and its stage-1 spike (`sous-spikes/l4-int8kv/results.md`, same day). Grounded on the maintainer's M5 Pro (applegpu_g17s, 20 GPU cores, macOS 27.0, mlx 0.32.3, mlx-vlm 0.7.2, the default Qwen3.8-27B-4bit with DFlash2). The maintainer ruled "take i8q to stage 2 and PR the relay first" on 2026-10-12; the split-QK relay is its own PR (`feat/attention-tile-split-qk`) and this design assumes it.*

## What the measurements established

**The attention tile is bandwidth-bound at T ≤ 5 and issue-bound from the second 16-row fragment group on.** With the relay in, its time at 57K keys per 16-layer chain is 13.2 ms at T = 1 and 13.8 ms at T = 5, reading K/V at 270–283 GB/s; a window diagnostic (every address pinned to one 32-key block) still costs 0.93 of the streaming time at T = 5.

**Halving the K/V bytes alone does not pay at T = 5.** The spike's first variant (int8 K/V, the tile's load pattern) was 1.8× at T = 1 and 0.83× at T = 3..5. Three further changes, each measured on the ktime chain with an A/A pair, carried it over the bar (ms per chain at 57K; `bf16` is the in-tree tile):

| variant | T = 5 | ratio | T = 1 | ratio |
|---|---|---|---|---|
| bf16 tile | 13.84 | 1.00 | 13.21 | 1.00 |
| int8 K/V, 16-byte V and scale loads (26 load instructions per lane-block, not 48) | 11.58 | 1.19 | 7.23 | 1.83 |
| + split-QK relay (now in the tile) | 10.31 | 1.34 | 7.03 | 1.88 |
| + int8 Q, QK on the `int8_t × int8_t → int32_t` rows | **8.17** | **1.69** | 7.33 | 1.80 |

At 44K the last variant is 1.65×, at 80K 1.72×, at T = 6 1.60×; a replicate with a new buffer seed and the variant order reversed reproduced every ratio within 0.01; on the real 35K-token tools fork it is 1.60× at T = 5. Converting int8 to bf16 in registers instead of feeding MPP's mixed rows is 0.80× at T = 5.

**Quantisation error is small on real K/V.** After `k_norm` the keys have no outlier channels (the largest channel's amax is 2.5–4.9× the median channel's), so one symmetric scale per (token, KV head) costs 1.0 % relative RMS on K and V. Attention outputs against the bf16 tile, on the fork's 16 layers with two query proxies: 0.43 % median, 0.72 % max, with int8 Q adding 0.04 %. (Group-64 scales would cost 0.7 %; they do not fold into the tile's permutation and stay the fallback if the suite objects.)

**The kernel is exact in the tile's sense.** Every verify row equals its one-row call bit for bit (235/235 rows at N0, the next two step boundaries, 35,001 real tokens and 57,601, T = 1..8), the NaN fence past n holds, and the output is within 2.0e-3 relative RMS of an fp32 reference on the same codes.

**Per-round quantisation is cheap with one dispatch per layer.** Quantising a round's five appended rows for 16 layers costs 0.46 ms through a fused kernel plus two slice updates per layer (0.92 ms through plain mlx ops), against a 5.6 ms saving per block-5 round at 57K.

## Decisions

1. **An int8 mirror beside the bf16 cache, built during decode only.** The KVCache stays bf16 and is what prefill, forks, the fork store, slots and every existing gate see. Decode and verify attention at n ≥ N0 read an int8 copy of the committed rows that lives on the cache object for the length of one `decode()` call. Prefill, including the one-row forward that ends every prefill segment, runs on the bf16 tile as today, so nothing this design serves reaches a stored fork, a turn slot or the budget's slot arithmetic.
2. **Per-token symmetric int8 with fp16 scales for K and V; int8 Q with an fp32 scale per row, quantised inside the kernel.** Scale = amax / 127, round to nearest, codes in [−127, 127]. The K scale multiplies the finished score, the V scale folds into the probability, the Q scale multiplies the score: no dequantisation in the loop.
3. **The verify rows read the same codes as decode rows.** A (row, key) score depends on the row's codes and scale, the key's codes and scale, and nothing else; the mirror is a pure per-row function of the bf16 rows, so bulk and incremental quantisation agree bit for bit, and a verify row equals the one-row call at its own length. Greedy output with the drafter equals greedy output without it, as for the tile.
4. **One flag, `[model].kv_int8`, default `true`, refused where the tile is not active.** The kernel rides on the tile (its platform gate, S, N0, the chunk schedule, `verify_plan`, `MAX_RUN`, the reduce kernel and, with a drafter, exact verify attention), so it inherits every tile refusal. A typo warns and means `false`.
5. **Memory: the in-flight turn pays 1.5× KV; slots and forks pay nothing.** The reserve the factory sizes for one window of KV is scaled by 1.5 inside the engine once `kv_int8` is active, before the budget is measured. The mirror is dropped at the end of every `decode()`, so a cache that becomes a slot never carries one.
6. **The fork key does not change for the flag; the kernel files join the epoch by the existing glob.** Nothing served by this design writes a fork, so the key stays as it is; the new `.metal` files are hashed into the epoch like every kernel source (one cold fork store after deployment, as for every kernel edit).

## Design

### Module layout

- `src/sous/engine/kv8.py` — the mirror, the quantiser dispatch, the kernel dispatch, `enable()`, `clear()`, the probe, `calls`, `static_reason()`.
- `src/sous/engine/kernels/attention_tile_kv8.metal` — the split body over int8 K/V/Q (the tile's relay kernel with int8 operands); the tile's reduce kernel is reused unchanged.
- `src/sous/engine/kernels/kv8_quant.metal` — the fused row quantiser.
- Seams in existing files: `tileattn._serve_decode` and `tileattn.serve_verify` (which gains the `cache` argument `verifyattn._wrap` already holds) hand a served call to `kv8` when it is active; `VLMEngine` enables it after the projection kernel, scales the reserve, sets the serving flag around `stream_generate`, drops the mirrors in `decode()`'s `finally`, clears it in `unload()`; `base.default_engine_factory` passes the config; `config.py` parses the key; the load line and the status document carry its state.

### The mirror

A `Mirror` object attached to each full-attention layer's `KVCache` as `_sous_kv8` (a plain attribute set with `object.__setattr__`, like the left-padding attributes mlx-vlm itself sets):

- `codes`: uint8 `[1, 4, cap, 512]`, each token's 256 K codes then its 256 V codes (the tile reads K and V as strided views of one buffer: token stride 512 B, V at +256, 16-byte-aligned rows);
- `sc`: fp16 `[1, 4, cap, 2]` = {sK, sV} per (token, head);
- `n`: the number of rows mirrored, from 0.

`cap` follows the cache's padded capacity (`keys.shape[2]`), growing by concatenation when the cache grows, as `KVCache.update_and_fetch` does.

**Update rule, at every served call** with `n` keys of which the last `T` were appended by this call (T = 1 for decode): `start = min(mirror.n, n − T)`; if `start < n`, quantise rows `[start, n)` from `cache.keys` and `cache.values` and slice-assign them; then `mirror.n = n`. Rows below `n − T` are the committed prefix: the cache rewrites rows only at its offset after a trim, and after any trim the next append's `n − T` is at or below the trimmed offset, so the rule re-quantises exactly the rows a rollback replaced. The first served call of a decode quantises the whole prefix (one pass of about 20–40 ms at 57K); later calls quantise T rows plus whatever a rejection rolled back. The mirror is never read for rows at or past `n`.

**Lifetime.** Built on the first served call inside `VLMEngine.decode()` and dropped from every layer's cache in that call's `finally`. The serving flag (`kv8.serving`, set and cleared by `decode()` on the session thread) is what makes prefill's one-row forward, the fork probe and `count_tokens` stay on the bf16 tile; the hooks serve `kv8` only while it is set and `kv8` is active.

### The quantiser (`kv8_quant.metal`)

Grid (256, R, 4), threadgroup (256, 1, 1): thread d of (head, row) loads `k[row, d]` and `v[row, d]` as bf16 from the cache's strided views, takes the row's amax by `simd_max` and an 8-value threadgroup reduce, sets `s = half(amax / 127)` (a zero row gives s = 0 and codes 0; −128 is never written), writes `codes[row, d]` and `codes[row, 256 + d]` as `clamp(rint(x / float(s)), −127, 127)` and, from thread 0, `sc[row] = {sK, sV}`. One dispatch per layer per served call, then two slice updates. Round-to-nearest-even is the kernel's rule; the tests' reference quantiser reproduces it (an explicit half-even formula in mlx ops), and the probe compares the two on random rows with ties present.

### The kernel (`attention_tile_kv8.metal`)

The relay kernel with these changes, everything else as the tile (grid, threadgroup, the 32·S chunk schedule, the online softmax, the fixed-order relay, resident accumulators, the clamp past n, the zero-probability fence, the partial/stats layout):

- **Inputs** `q` bf16 `[1, 24, T, 256]`, `codes` uint8 and `sc` fp16 as above (read through strides, `ensure_row_contiguous=False`), `params`.
- **Q quantised in the kernel.** Each half stages its 8 k-steps of Q as today, then: the lane's amax over its staged dims per row → `simd_shuffle_xor` over the four lanes sharing a row → the two halves exchange their row maxima through threadgroup memory (one barrier) → `sq = amax / 127` in fp32 per row → each lane rewrites its staged Q as packed int8 codes (`rint`, clamp) in place → one more barrier. Three barriers per call, none per block.
- **QK** on `get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>` / right the same, `multiply_accumulate` into an int32 destination, zeroed per block; the relay sums int32 partials (exact); then `S = float(Si) × (sq[row] × sK[key] × SCALE_LOG2)`.
- **Loads per lane per 32-key block:** K as 16-byte runs (two per key row per half), V as 16-byte runs covering two output tiles, scales as one 16-byte load per 16-key half when the block lies inside n and clamped `half2` loads in a call's last block. The head-dim permutation is the tile's; it keeps every lane's K and V bytes contiguous.
- **PV** on `<float, int8_t, float>` rows: `ct_p = p × sV[key]` (masked keys are exactly 0, so a NaN scale past n cannot leak), the row sum on the unscaled p.
- **MSL** as the tile: `remove_addrspace_t` on every cooperative-tensor `decltype`, `math_mode: safe`, `common.h + nax.h` as the header, compiled under its own kernel name.

### Dispatch (`kv8.py`)

- `decode(queries, cache, splits)` and `verify(queries, cache, splits, runs)` mirror `tileattn.decode/verify`: below N0 the stock paths over the bf16 cache (unchanged, exact against each other); from N0 the kv8 kernel over the mirror, cut by `verify_plan` exactly as the tile, with the tile's reduce kernel.
- `tileattn._serve_decode`: after its existing checks and `keys.shape[2] >= N0`, `if kv8.serving_now(): return kv8.decode(queries, cache, splits)`; else the tile as today. `tileattn.serve_verify(queries, keys, values, splits, cache)` likewise; `verifyattn._wrap` passes `cache`.
- Counters `kv8.calls`: `decode_kv8`, `verify_kv8`, `verify_rows_kv8`, `verify_launches_kv8`, `mirror_builds`, `mirror_rows`, `declined` (active but not serving, or out of scope). Nothing is logged per turn.
- `enable(model, *, enabled, tile_status)`: first statement clears the plain-int flag; `off` when the config says so; `unavailable` (one INFO line, `expected=True`) when the tile is not active (its reason or `attention tile off`); then the compile probe (every T = 1..8) and the probe on the loading thread: parity (every verify row `array_equal` its one-row call at N0, the next two step boundaries and the far mark, T = 1..8 at the tile's three placements), the fence (NaN scales and −128 codes past n), the mirror rule on a real `KVCache` (append, trim, re-append), the quantiser against the reference on rows with ties, relative RMS ≤ 1e-2 against the fp32 reference on the same codes, and ≤ 3e-2 against stock bf16 SDPA on Gaussian K/V (the quantisation's own error, ≈ 0.7 % there). A compile or probe failure, any exception (logged with its traceback) and pin drift are `warnings.warn` plus `unavailable`. The probe leaves the flag clear, restores `calls`, and calls `mx.clear_cache()` in `finally`.
- `clear()` from `VLMEngine.unload()`; `static_reason(config, model_config, drafter_kind)` = `kv_int8 off` or the tile's static reason, for `sous tune`'s notices.
- Source pins: the kernel reads nothing of mlx-vlm beyond what the tile pins; the tile's `VALIDATED_DECODE_SOURCES` and verifyattn's gates cover the hooks.

### Engine, config, status

- `config.py`: `kv_int8: bool = True` on `SousConfig`, parsed by `_kv_int8` with the warn-and-False rule; `KNOWN_MODEL_KEYS` (or the equivalent set) gains it.
- `base.default_engine_factory` passes `kv_int8=config.kv_int8`; `VLMEngine.__init__` takes `kv_int8: bool = False`, enables it after `projkernel.enable` (so the tile's state is settled) and before `measure_cache_budget`, and scales `reserve_bytes` by 1.5 when active: the in-flight turn's KV is bf16 plus the mirror while it decodes.
- `VLMEngine.decode()`: sets `kv8.serving` for the `stream_generate` call and in `finally` clears it and drops every layer's mirror; `unload()` calls `kv8.clear()`.
- The load line carries `kv_int8=<state>` and `kv_int8_probe_s=`; the status document's `engine` block carries `kv_int8: {state, reason}`; `sous tune`'s per-arm notice names `unavailable`, never `off`.
- `window.kv_bytes_per_token` stays the bf16 cost: it sizes slots and forks, which stay bf16.

## Testing

- `tests/kv8_fixtures.py`: a plain-mlx stand-in for the kv8 kernel (the tile's stand-in over dequantised codes with the same partition), the reference quantiser (half-even), serving-layout inputs, the `nax` skip marker reused from `tile_fixtures`.
- `tests/test_kv8.py` (CI, stand-in): the mirror rule (first call quantises the prefix; T rows per call; a trim followed by an append re-quantises exactly the replaced rows; capacity growth; never reads past n; dropped at the end of decode); dispatch (below N0 stock, from N0 kv8; `verify_plan` runs; counters); the serving flag (prefill's one-row forward and the fork probe never build a mirror); enable's ordering and refusals (off, tile not active expected-INFO, compile and probe failures warn, exception path); `clear()`; config parsing; the factory passes the flag and the engine scales the reserve; the load line and the status document.
- `tests/test_kv8.py` (`nax`-marked, real kernel): compile at T = 1..8; parity, fence, the quantiser against the reference with ties, relative RMS bounds; the kernel's Q quantisation against the reference on the host.
- `tests/test_kv8_model.py` (`model`-marked, the M5): the tiny-model path through both hooks (`calls` shows `decode_kv8` and `verify_kv8`), greedy parity drafter on vs off on the tiny model, `enable()` on the real 27B with the probe time logged.
- Existing suites unchanged: the tile's and verifyattn's tests keep passing with the kv8 flag clear (the default in every test and script).

## Merge gate (pre-registered 2026-10-12, before any in-model number)

On the M5 Pro, the branch tree in its own worktree and venv, every job under `gpu_run_m5.py` with the production daemon idle:

1. **Tests:** `uv run pytest -m "not model"` green on the M2 and the M5 (nax tests included), `uv run pytest -m model tests/test_kv8_model.py tests/test_tileattn_model.py` green on the M5; ty, ruff, lock clean.
2. **Kernel timings** (ktime chain, A/A pair): kv8 kernel against the relay tile, T = 1, 3, 5, 6, 8 at 8K / 57K / 131K: ≥ 1.35× at (5, 57K), (1, 57K) and (5, 131K); no cell below 1.00× at 8K.
3. **Speed, primary:** #131's 64 real subagent turns replayed through a real `sous serve` with a per-round shim that draws `kv8` on or off each round (both arms keep the mirror current, so the off arm pays nothing and the on arm pays its own quantisation); block 5, sampled at 0.7, two seeds; **ship** if the stratified dt-attributed tokens/s ratio on/off ≥ 1.02 with the 95 % turn-clustered bootstrap lower bound > 1.00 (#157's rule); **close** if the lower bound ≤ 1.00 and the point < 1.02; else to the maintainer.
4. **Speed, secondary:** same-hour per-daemon ABBA against main on the 64 turns (sampled, 1024 tokens): pooled ≥ 1.00, prefill within ±3 %, peak memory within the scaled reserve, `pressure_evictions` 0.
5. **Parity:** greedy (temperature 0, 256 tokens) drafter on against drafter off within the branch on the 64 turns plus the a6443762 continuation past 128K: every reply and stop reason identical, every cache history equal; one divergence fails, read at the first differing token's top-2 logit gap before the kernel is blamed. With `kv_int8 = false` the branch equals main bitwise on the same replays.
6. **Suite:** #123's paired design, 8 tasks × 8 seeds, arms `kv_int8` on and off sharing one run directory and seed per pair; keep off if the point difference is below −0.05; eligible if the 95 % lower bound ≥ −0.05, completed ≥ stock − 8 and repetitions ≤ stock's.
7. **Block 6:** a block-6 arm is carried in step 3 (arms 5k/6k drawn per round with kv8 on) as information for #157; it decides nothing here.

## Documentation

- `docs/configuration.md`: the `kv_int8` key, what it changes (decode and verify attention read int8 K/V from the second 1,536 keys; prefill and stored prompts stay bf16), the memory note (the in-flight turn's KV is 1.5× while it decodes; the reserve grows with it), and that it is refused wherever the attention tile is.
- `CLAUDE.md`: one engine bullet in the tile's style (the mirror's lifetime and update rule, the serving flag, the thread rules, what the probe proves, the counters, the fork-epoch note).
- README's performance numbers only after the gate.

## Out of scope

- An int8-only cache (the memory win: smaller slots, forks and reserve). It needs a cache class forkio, fork_copy, every `type(cache) is KVCache` gate and the fork layout version admit; the mirror keeps those untouched and can become that later.
- Group-64 K scales (the quality fallback): needs the group-aligned head-dim permutation; built only if the suite fails at per-token.
- Prefill attention over int8 history.
- Changing the default block: #157's rule on the relay and kv8 arms decides that separately.
