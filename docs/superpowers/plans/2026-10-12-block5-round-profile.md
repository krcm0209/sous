# Block-5 Round Profile (M5 Pro) — Pre-registered Measurement Plan

> **For agentic workers:** this is a measurement plan, not a feature plan. Nothing here
> changes sous. The harness is throwaway and out of tree (`~/code/sous-spikes/prof-b5/`,
> mirrored to `~/code/personal/sous-spikes/prof-b5/` on the M5 Pro). Execute the tasks in
> order with superpowers:executing-plans; a fresh reviewer gates Tasks 2, 3, 4 and 5
> before `run.sh` starts. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure, on the production M5 Pro configuration, where the time inside one
block-5 DFlash2 round goes (GPU time by stage, host time, GPU idle), at T = 1, 5 and 6 and
across the replay's contexts, so the remaining decode levers are ranked from measured
ceilings instead of isolated-bench arithmetic.

**Architecture:** three instruments on the same replayed turns, inside the same daemon
process wherever two numbers are compared. (A) the per-round recorder from the #148/#157
spikes, unchanged, gives the reference round time and draws block 5 or 6 per round. (B) a
shim that wraps the verifier's, the drafter's and the decode layers' methods records each
stage's host build time (lazy mode) and, with an `mx.eval` before and after every leaf
stage, its GPU time (sync modes); the mode is chosen per post, so plain, lazy and sync
rounds of one turn come from one process. (C) `xctrace` Metal System Trace over the replay
daemon gives GPU busy and idle intervals per round, the pure GPU time of every sync-mode
eval, and the GPU performance state, without touching the code path.

**Tech Stack:** Python 3.14 in the sous-remote venv on the M5 (mlx 0.32.3, mlx-vlm 0.7.2,
main at or after 03ed9e1), `p157/gate157.py` as the replay driver's base with p157's
`build_replay.py` and `replay-157.json`, `rounds_gate.py`, the shared `gpu_run_m5.py`,
`xcrun xctrace` (Xcode 27 on macOS 27.0), numpy for the analysis.

**Spec:** none. This plan is its own pre-registration. The question comes from the
2026-10-12 re-rank: every remaining decode lever (fused GDN verify with deferred state
commit, GDN prework fusion, the 8-layer `async_eval` flush, drafter fusions, T = 5 → 6,
INT8 KV through the attention tile) draws on the part of the round that has never been
measured in situ.

## Global Constraints

- The production daemon on the M5 Pro is never stopped, reconfigured or attached to.
  Every GPU job runs under the shared `~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py`
  (one GPU job at a time across sessions; waits for the production daemon to be idle;
  flags contamination) in an isolated HOME on port 8384, as every spike since #132 has.
- The M5 tree is `~/code/personal/sous-remote` at main. Never `~/code/personal/sous`.
- The shim patches nothing in the sous tree and nothing in site-packages: class attributes
  and module globals are rebound in the shim's process only.
- Every mlx-vlm and sous function the shim rebinds or times is source-pinned by hash at
  import; a mismatch refuses the run.
- Sampled decoding at the production default (temperature 0.7) is the primary
  configuration. Greedy is report-only. Per-round stage *milliseconds* are compared, never
  tok/s (sampled texts differ post to post; #160's lesson).
- The isolated config carries `max_context_tokens = 262144`, `prompt_cache_gb = 24` (so a
  turn's predecessor slot is retained as a copy beside the published slot and the forks;
  the budget does not enter the round time, so #157's 119.8 ms at `prompt_cache_gb = 14`
  stays comparable) and `idle_unload_minutes = 1440` (one load per daemon).
- Per-turn `max_tokens` is 512 in profile posts and 128 in the keep-warm pass. Replay
  decodes end at EOS after ≈ 38 rounds (#157: 10,951 rounds over 290 decodes), so the count
  rules below are met by posting a turn several times inside one daemon, never by a longer
  budget.
- **Repost rule.** A repost of an identical request cannot take the slot its first post
  published (`promptcache.reuse_length`: an exact match is a miss) and would re-prefill
  the whole conversation from the tools fork. Every turn k in this plan is therefore
  warmed by posting its predecessor turn k−1 (same subagent, rebuilt by the same builder),
  then posted once untimed, and every timed post of k takes the k−1 slot and prefills only
  the k−1 → k delta. A timed post must log `took=turn@n_{k−1}` or `turn-moved@n_{k−1}`
  (same take length, same class); a `fork`, `disk` or `miss` take invalidates the post.
  A subagent's first turn is warmed by itself and its timed posts take `fork@<header>`.
- **Freeze.** No number or rule in sections 2–4 changes after the first job of `run.sh`
  (D0) starts. Task 3 step 0 (the attach and performance-state probes) and the M2 smoke
  are exempt probes whose outcomes select pre-registered branches, not amendments. A
  harness-only amendment is allowed before the data it affects has been analysed; it is
  logged in `README.md` with a timestamp and the passes it touches. No analysis script
  runs on any rounds file before D3 completes; the ledger counts and the driver's repost
  loop are the only reads. A definition changed after its data was seen makes that
  quantity exploratory, and no lever decision may rest on it.
- Nothing is posted to GitHub by the executor. The report goes to the maintainer first.

## Threats to validity (each pinned by a check below)

1. **Lazy evaluation puts GPU work in the wrong bucket.** `mx.eval(out)` runs every
   unevaluated ancestor of `out`, so a leaf's eval would absorb the unsynced work before it
   (conv/SiLU/split/norm before the recurrence kernel; rope/norm/cache append before the
   attention tile), and an enclosing stage's eval would find nothing left. Pinned by: every
   leaf wrapper evaluates its *inputs* first and records that time as a named `pre` bucket
   (section 3.2); the only enclosing stage that evaluates is the layer, at every layer
   boundary, so no bucket spans layers; residues are sums of measured buckets, never
   subtractions.
2. **Sync mode distorts.** A sync serialises host and GPU, adds a round trip, and makes the
   host encode each stage's dispatches while the GPU waits. Pinned by: the pure GPU time of
   every sync-mode eval read from xctrace in D3, giving an additive per-eval overhead per
   bucket that is subtracted from D1's host-timer buckets (section 3.6); round-trip and
   no-op costs measured in situ per turn as a cross-check only; the closure check (H6)
   against the plain round of the same process.
3. **Wrong hook order.** A timer inside projkernel's hook would time the stock path.
   Pinned by: timers installed at the first `generate()` (after `enable()`), a guard that
   refuses a post unless the shim's wrappers are the outermost attributes, and per-round
   `projkernel.calls`/`tileattn.calls` ratios equal across modes within 1 %.
4. **Thermal and process drift.** The M5 drifts 5–20 % across jobs and ≈ 9 % by time of day.
   Pinned by: every comparison is within one daemon process and within one turn (modes
   alternate per post in ABBA order; blocks 5 and 6 are drawn per round); L1's T = 1
   recurrence term comes from an in-process microbench, not from the T = 1 daemon; two
   canaries (compute: 20 × 4096² bf16 matmuls; bandwidth: 5 sums over a 1 GiB bf16 array)
   timed after every post's decode flag a post whose canary deviates > 5 % from the median
   of the same turn's posts.
5. **Isolated sums are not in-situ time.** #133's per-forward projection sums were measured
   one linear at a time. Pinned by: H1 is decided on the in-situ number alone, with the
   matched prior stated (one-row kernel isolated ≈ 61 ms against the ctx-128 decode step of
   63.1 ms).
6. **Replay turns are not the field, and few turns give wide intervals.** Pinned by: 18
   turns spread evenly over 44–80K, content-deduplicated, one per subagent where possible,
   every stage fitted against context and reported at 56K with a bootstrap over clusters
   and a t-interval; buckets descriptive only; a 128K/190K pair report-only.
7. **The first rounds are special.** draftctx feeds the drafter the prompt tail on its
   first call; the first post of a turn restores or extends a slot. Pinned by: rounds 1–2
   of every decode excluded; the k−1 warm-up and one untimed post of k precede every timed
   post; D0 writes the tools forks to disk once.
8. **Tracing perturbs.** Metal System Trace adds per-command-buffer instrumentation.
   Pinned by: traced and untraced plain posts of the same turn in the same daemon when
   `--attach` works; when only `--launch` works, each D3 turn's canary-normalised ratio
   `r = (own_D3 / canary_D3) / (own_D1 / canary_D1)` is computed, and any `r > 1.03` sends
   L2 to the maintainer whatever its value.
9. **Instrumentation can pre-warm the GPU.** Pinned by: D6 runs no canary, no begin-decode
   marker and no microbench; its only shim GPU work is the marker after each decode's last
   round, before the driver's sleep.

---

## 1. What is already known (provenance)

| fact | value | source |
|---|---|---|
| Block-5 round, sampled, kernel active, 44–88K, 127 distinct turns, stratified | dt 119.8 ms [117.9, 121.7]; own 119.5; tokens/round 3.32; 2.36 accepted per round | #157 pooled, `p157/verdict.md` |
| Host gap between rounds (walk end → next block dispatch) | 0.350 ms, flat in emitted tokens | #157 `gap fit over 10400 rounds` |
| Block-6 round, same turns | dt 133.0 ms, own 132.7; +13.2 ms/round (stratified; 13.0 unstratified) for +11 % tokens/round; 2.72 accepted per round | #157 |
| Rounds per decode in the replay | ≈ 38 (10,951 rounds / 290 decodes at max_tokens 1024; 5 budget-truncated; 762 rounds sat in decodes under 8 timed rounds) | #157 |
| One-row step, drafter off, kernel active | 74.7 ms (13.38 tok/s) | #160 stage-2 gate |
| Isolated per-forward projection sums (MMA kernel) at T = 1..8 | 75.0, 77.2, 78.6, 78.1, 78.2, 77.2, 78.2, 78.9 ms; the one-row kernel ≈ 61 ms at T = 1 (1.012× qmv vs 1.244×) | #133 `mma-only`; #160 stage 1 |
| Stock decode at ctx 128, T = 1 (pre-kernel) | 63.1 ms total; MLP 61 %, GDN 27 % (recurrence 14 % ≈ 8.8 ms; 12.5 % ≈ 8.3 ms at 16K), attention 8 % | 2026-09-17 layer profile |
| Verify attention tile, 16 attention layers, 57K | T = 3 14.8 ms (≈ 252 GB/s); T = 8 16.9 ms (≈ 221 GB/s) | #132 microbench |
| Drafter steady state at block 3 / block 8, 2.4K and 57K | 10 ms / 21 ms, flat in context | #131 timing |
| Model | 64 layers: 48 gated-delta, 16 full attention (every 4th); attention 24/4/256; GDN 48 v-heads × 128, 16 k-heads × 128; vocab 248,320; GDN layers use `ArraysCache(size=2)`: `cache[0]` conv window, `cache[1]` recurrent state | `config.json`, `language.py:1935` |
| Drafter (z-lab/Qwen3.8-27B-DFlash2) | 5 layers, hidden 5120, 32/8 heads, head dim 128, all sliding-window 2048; quantized to 4-bit at load (`vlm.py:49`); `bind()` attaches the target's `embed_tokens` and `lm_head` as its own attributes, and `_logits` calls that `lm_head` (0.72 GB) through projkernel hook 2 | drafter `config.json`, `qwen3_dflash/dflash.py:214-227`, `dflash2.py:157` |
| GDN state per layer | (48, 128, 128) fp32 = 3 MiB; the with-states verify kernel writes T−1 intermediates: 576 MiB per round at T = 5 (write floor 2.24 ms at 270 GB/s); the commit's state select is a view (≈ 0 GPU) | `gated_delta.py`, `cache.py:update_recurrent`, `cache.py:1061` |
| KV read per verify forward | 64 KiB per token (16 × 4 × 256 × 2 B × K+V): 2.9 GB at 44K, 3.7 GB at 56K, 5.2 GB at 80K | arithmetic |
| Weights streamed per forward | ≈ 14.4 GB (25.6 B params at 4.5 bits, embedding excluded) ≈ 53 ms at 270 GB/s | arithmetic from `config.json` |
| Bandwidth constant used below | 260 GB/s (the tile's measured rate; weights use 270 where the prior did) | #132 |
| Turn-slot semantics | an exact-match repost is a miss (`reuse_length`); a turn slot is copied and left in place when `_make_room` can hold the copy, else moved; only the lowest boundary (the tools fork) is written to disk | `promptcache.py`, CLAUDE.md |
| xctrace on the M5 over ssh | `--launch -- <python>` works: `metal-gpu-intervals` (1,063 rows for a 5 s loop; columns include start, duration, process, cmdbuffer-id, start-latency, state) and `gpu-performance-state-intervals` present; `gpu-shader-profiler-interval` empty; the "log archive is corrupt" error concerns os_log only. **`--attach` is untested.** | probe 2026-10-12, `/tmp/xt2.trace` on the M5 |
| Production log | no organic `sous claude` use: 586 of 936 turns are the 2026-09-26 fan-out spike, the rest gates and smokes | daemon.log 2026-10-12 |

**The round** (mlx-vlm 0.7.2 `speculative/dflash.py:_dflash_rounds`, sous hooks in place):
`_dflash_next_block_size` → draft (`DFlash2DraftModel.draft_block` or the inherited
`draft_block_greedy`, wrapped per instance by draftctx) → the sampler RNG helper's
`mx.async_eval` and then `mx.async_eval(draft_tokens)` (host encodes the draft graph) →
verify (`_dflash_verify` → `LanguageModel.__call__(…, speculative_verify=True)` →
`Qwen3_5BatchInvariantForward.__call__` → `_model` → 64 × `_layer` → `_gated_delta` /
`_attention`, then `_feed_forward`; projections through projkernel hook 1; verify
attention through verifyattn's wrapper to `tileattn.tile`; the head through `_linear`) →
`mx.concatenate(hidden_states)` + `mx.async_eval(hidden)` (host encodes the verify graph
up to the last captured layer; nothing blocks) → **walk**: sampled,
`_sample_dflash_target_walk` does `draft_tokens.tolist()` and then, per drafted position
until the first mismatch, `sampler(logprobs[:, position])` + `mx.eval` — one GPU round
trip per position, ≈ 3.4 per round — the first of them completing the un-encoded tail
(remaining layers, final norm, head, logsumexp) and waiting for the verify; greedy,
`_speculative_walk` is two `tolist()` calls → `commit_speculative_round` → yield. The
recorder stamps `t_start` at `_dflash_next_block_size` and `t_end` after the walk, so
`own` = draft + verify + walk and `gap` (0.35 ms) holds commit and emission.

## 2. Hypotheses (pre-registered)

Each states a predicted range and what the measurement decides. Predictions are for
T = 5, sampled, at 56K context (the fitted value, section 3.6), as GPU milliseconds of the
plain round `own` unless noted.

- **H1 Projections in situ.** Verify projections plus head (`proj@gdn` + `proj@attn` +
  `proj@mlp` + `proj@verify_call`, excluding the drafter's own linears and its call of the
  target head) take 55–79 ms. Decided on the measured value; the report states it beside
  #133's 78.2 ms isolated sum.
- **H2 Draft.** The drafter's call (`draft` GPU, its target-head call included) takes
  10–18 ms and is flat in context: the 95 % bootstrap CI of slope × 36,000 tokens (the
  44K → 80K change) lies inside [−1.5, +1.5] ms (sliding window 2048, head dim 128; a rise
  would point at the context projection path, not at drafter KV).
- **H3 GDN chain.** The gated-delta chain excluding projections (`pre@gdn/gdn_kernel` =
  prework, `gdn_kernel` including g/β prep, `gdn_norm`) takes 10–20 ms at T = 5; the
  in-process microbench gives `kernel_T1` 7.5–11 ms (prior 8.3–8.8) and a with-states
  store cost `kernel_T5_states − kernel_T5` of 2.0–4.0 ms (floor 2.24 ms).
- **H4 Attention.** `floor_turn = (prompt tokens + median generated position of the kept
  rounds) × 65,536 B / 260 GB/s`; at 56K the floor is 14.2 ms. H4 holds when the 95 %
  bootstrap CI of the fitted `sdpa / floor` ratio at 56K lies inside [1.00, 1.15]
  (band 14.2–16.3 ms); per-turn ratios are reported.
- **H5 Host.** `max(0, verify.build + verify.encode − draft_gpu)` ≤ 2 ms (lazy posts; the
  walk's first-eval host time added as the upper bound, since `verify.encode` excludes the
  un-encoded tail), and GPU idle inside `own` (xctrace, D3 plain posts) ≤ 5 % of `own`.
- **H6 Closure.** `draft + Σ layers + pre@verify_call/proj + head + walk_gpu` from the
  burst-corrected sync decomposition, plus measured idle, reproduces the plain round
  within ±5 %; sync-mode `walk.first_wait` must be ≈ a no-op (checked). A miss means no
  lever is ranked spike-worthy from the stage table; the report says which instrument
  disagrees and goes to the maintainer.
- **H7 T = 5 → 6.** Of the per-round 6k − 5k difference (paired within turn, same process;
  #157: +13.2 ms), projections contribute ≤ 2 ms, the drafter 1–3 ms, attention ≤ 1 ms
  (≈ 0.4 ms per row from #132), the GDN chain ≤ 3 ms, the walk 0.1–0.5 ms (+0.37 accepted
  positions per round × H10's per-position cost). Unattributed = Δown(plain) −
  [Δdraft_gpu + Δ(Σ layer_gpu) + Δhead + Δpre@verify_call/proj + Δwalk.pingpong(lazy) +
  max(0, Δ(verify.build + verify.encode)(lazy) − Δdraft_gpu)], all paired within turn at
  56K. H7 fails if the unattributed share exceeds 30 % of Δown.
- **H8 Keep-warm.** After an idle gap ≥ 3 s the next post pays 0.1–0.5 s more (slower first
  rounds and prefill while the GPU re-enters its high performance state); the
  performance-state track, if the probe shows it informative, shows the ramp.
- **H9 Progress summaries.** In the 2026-09-26 cap-1 runs, Claude Code's progress-summary
  calls are 3–10 % of local engine seconds, and Σ `queue_s` charged to real turns is
  0.5–3 % of their wall clock.
- **H10 Walk.** The sampled walk's per-position round trips (all evals after the first,
  `walk.pingpong` from lazy posts, with D3's `host − burst` as the round-trip part) cost
  1–3 ms per round.

## 3. Measurement design

### 3.1 Workload

- **Profile set (18 turns).** Pool = p157's 145 replay turns (`replay-157.json`, order of
  `picks145.txt`, transcripts under `~/code/personal/sous-spikes/p157/transcripts/`) that
  are eligible, have a predecessor turn k−1 in the same transcript (or are the subagent's
  first turn), and whose *recorded* sous-local response has `usage.output_tokens` ≥ 200
  (a transcript property, checked on one long answer against its text length before
  trusting it; relaxed to ≥ 100 as a stated step if the pool is short), deduplicated by
  analyze157's content key (sha256 of converted messages, tools and max_tokens; the first
  in picks145 order kept). Targets: 18 values `t_i = 44,000 + (i + 0.5) × 2,000`. For each
  target in order, the nearest unused pool turn within ±2,000 tokens whose subagent and
  content key are not yet used; a target with none is skipped and stated; remaining slots
  are filled from the nearest targets allowing a second turn per subagent, still within
  ±2,000 of the target. Descriptive buckets are the sorted thirds. Task 1 also lists three
  reserve turns per third, in order, for floor replacements: the unused pool turns nearest
  to that third's six grid targets (A: 45–55K, B: 57–67K, C: 69–79K), ties by picks145
  order. The list, each turn's `n`, predecessor and recorded
  length are written to `README.md` before any timing.
- **Long pair (report-only):** from the full `replay-157.json` set (647 turns, 44K–229K; the
  #130 sweep log holds only usage, and its request bodies were built from a since-changed
  tree), the eligible turns nearest 128K and 190K rendered tokens whose ids extend their
  predecessor, each with that predecessor.
- **Keep-warm turn:** the first profile-set turn in picks145 order whose k−1 → k delta is
  1,000–4,000 tokens.
- **Greedy check (report-only):** the first turn of each third at temperature 0.
- **xctrace turns:** the first turn of each third.

### 3.2 Instrument (B): stage timers

**Where a wrapper sits.** The recorder (`rounds_gate.install(ARMS or None,
seed=int(PROF_SEED))`) and the shim's wrappers are installed at the first `generate()` of
the process, whatever its mode, after the engine's `enable()` calls, so every wrapper is
outermost (projkernel hook 1 wraps `_linear`/`_linears`, hook 2 wraps
`nn.QuantizedLinear.__call__`, both at `enable()`; verifyattn wraps the verifier's
attention; tileattn replaces the decode SDPA global). The `install` record is written
then. On a plain post the shim restores every original it rebound (kept with its owner in
`_orig`) except the recorder, the T = 1 step recorder (which in plain mode stamps `dt`
only) and the import-time drafter wraps, which check the mode on their first line; on the
next non-plain post it re-installs. Each wrapper records `build` (call return minus call
start: host graph-construction time) and, in sync modes, `eval` and `pre` as below.
Records are keyed `(parent, stage, layer_index)`: `parent` is the top of a stack the
wrapper pushes and pops (`draft`, `verify`, `verify_call`, `model`, `gdn`, `attn`, `mlp`,
`walk`, `step`, or `-`), `layer_index` is −1 outside layers; the layer counter wraps
*outside* the layer timer, and the `model` wrappers reset it to −1 on return. Verifier
projections are keyed `proj_in` (from `_linears`) and `proj_out` (from `_linear`); stock
T = 1 `QuantizedLinear` calls are keyed `proj#<ordinal within parent>`. A wrapper whose
stage is already on the stack passes through (DFlash2's inherited `draft_block_greedy`
may call `draft_block`; `verify_greedy` wraps `_dflash_verify_greedy`, which calls
`_dflash_verify` = `verify`).

| stage | wrapped attribute | leaf rule (call time) | T = 5 / 6 | T = 1 |
|---|---|---|---|---|
| draft | `DFlashDraftModel.draft_block`, `.draft_block_greedy`, `DFlash2DraftModel.draft_block` (class, at import; draftctx binds the instance methods by `getattr` at `enable()`, so its tail feeding wraps *outside* the timer; round 1 is excluded anyway) | leaf in sync and syncfine | yes | — |
| verify / verify_greedy | `dflash._dflash_verify` / `dflash._dflash_verify_greedy` (module globals read per round) | never | yes | — |
| verify_call | `Qwen3_5BatchInvariantForward.__call__` | never | yes | — |
| model | `Qwen3_5BatchInvariantForward._model`; `language.Qwen3_5Model.__call__` | never (resets the layer counter) | yes | yes |
| layer | `Qwen3_5BatchInvariantForward._layer`; `language.Qwen3_5DecoderLayer.__call__` | leaf in sync and syncfine (every layer boundary); in fine layers its eval holds the final residual add | yes | yes |
| gdn / attn / mlp | `._gated_delta` / `._attention` / `._feed_forward`; `language.Qwen3_5GatedDeltaNet/Qwen3_5Attention/Qwen3_5MLP.__call__` | never | yes | yes |
| proj_in / proj_out / proj#k | `._linears` / `._linear` (outside hook 1); `nn.QuantizedLinear.__call__` (outside hook 2) only for modules carrying projkernel's `_TAG` | leaf when the parent is `verify_call`, `step` or `draft` (the head, in sync and syncfine, any layer index); else leaf only in syncfine inside `FINE_LAYERS` | yes | yes |
| gdn_kernel | `speculative_verifier.gated_delta_update` and `language.gated_delta_update` (two names, one function); extra input: `cache.cache[1]` (the recurrent state; `cache[0]` is already the new window) | leaf in syncfine inside `FINE_LAYERS` | yes | yes |
| gdn_norm | `language.Qwen3_5RMSNormGated.__call__` | same | yes | yes |
| sdpa | `tileattn.tile`; `verifyattn.grouped_attention` (also reached by tileattn's per-call import; tileattn has no module-level name of its own) | same | yes | yes (decode's one-row tile) |
| walk | `dflash._sample_dflash_target_walk`, `dflash._speculative_walk`, rebound *after* `rounds_gate.install()` so the shim sits outside the recorder's stamps; pushes `walk`, stamps entry and exit | — | yes | — |
| encode | `dflash.mx` and `common.mx` rebound to a proxy whose `eval`/`async_eval` stamp call and return under the current parent and in round order: the async_evals between the draft wrapper's exit and the verify wrapper's entry sum to `draft.encode`; the one after the verify returns is `verify.encode`; inside `walk`, the first eval is `walk.first_wait` and the rest `walk.pingpong` (greedy: the walk's host time stands in) | — | yes | — |
| step (T = 1 only) | `language.LanguageModel.__call__`, installed only when `engine._draft is None`; acts only when `inputs.shape[1] == 1`, `cache is not None`, `speculative_verify` is not set and the post's decode has started (the first `on_delta` of the post): `begin_round()`, push `step`, emit a `step` record on return with `dt` = entry-to-entry, `decode` and `round` indices, and in sync modes `own` = entry → `mx.eval(out.logits)` | — | — | yes |

**Leaf protocol in sync modes.** A leaf wrapper first evaluates its array arguments and
keyword arguments (`_arrays` walks tuples, lists, dicts and `LanguageModelOutput`, never
`nn.Module`), timing that as `pre@<parent>/<leaf>`; then calls; then evaluates its
outputs, timing that as `eval`. Every eval stamps call and return (`perf_counter`) for the
trace pairing and increments `n_sync` when its duration exceeds the midpoint of the
turn's `noop_us` and `rt_us`, else `n_noop`. The named `pre` buckets:

| bucket | holds |
|---|---|
| `pre@gdn/proj_in` | input layernorm (the previous layer's residual add is in that layer's own eval) |
| `pre@gdn/gdn_kernel` | conv-state concat, conv1d, SiLU, split, q/k norm, the conv-window copy; the recurrent-state select is a view |
| `pre@gdn/gdn_norm`, `pre@gdn/proj_out` | ≈ 0 (already evaluated) — a check |
| `pre@attn/proj_in` | input layernorm |
| `pre@attn/sdpa` | rope, q/k norm, cache append (`_prepare_projected_qkv`) |
| `pre@attn/proj_out` | transpose/reshape, gate sigmoid |
| `pre@mlp/proj_in` | post-attention layernorm + the attention/GDN residual add |
| `pre@mlp/proj_out` | swiglu |
| `pre@verify_call/proj_out` | final norm (`model.norm`) |
| `pre@draft/proj_out` | the drafter body up to its final norm (its own layers) |
| `eval@model/layer` (fine layers) | the final residual add |

Totals are sums: `gdn_chain = pre@gdn/gdn_kernel + gdn_kernel + gdn_norm + pre@gdn/gdn_norm
+ pre@gdn/proj_out`; `gdn_total = gdn_chain + proj_in@gdn + proj_out@gdn + pre@gdn/proj_in`;
`attn_other = pre@attn/sdpa + pre@attn/proj_out`; `mlp_act = pre@mlp/proj_out`;
`head = eval proj_out@verify_call`; `draft` GPU = `pre@draft/proj_out + eval proj_out@draft
+ eval draft` (the candidate selector). `build` is a host cost only in lazy mode.

**Modes** (read per post from `PROF_MODE_FILE`, written by the driver before each post;
`PROF_MODE_SEQ`, a cycled list, serves the M2 smoke instead):

- `plain` (also written for warm-up posts): recorder only; originals restored. The
  reference round.
- `lazy`: timers record `build` only; no eval anywhere. The lazy overhead is the paired
  `(lazy − plain) / plain` fitted at 56K over the 18 turns; if its bootstrap upper bound is
  ≤ 3 %, lazy posts are used; otherwise `lazymin` posts (a stage allowlist: `draft`,
  `verify`, `walk` and the encode proxy only) are added for every turn and used instead.
- `sync`: `mx.eval` after the draft, after every `layer`, after the final norm and the
  head projection, and inside the walk: ≈ 70 syncs per round.
- `syncfine`: as `sync`, plus the leaf protocol inside `FINE_LAYERS` = {1, 2, 4} (gated-
  delta) and {7} (attention) — not 0 and 3, whose first evals absorb the verify-input
  concat, embedding, masks and position embeddings. Per-stage times extrapolate × 48/3
  and × 16/1, checked against the per-layer totals: Σ over a fine layer of (all its buckets
  + its own eval) must equal the `sync`-mode layer time of the same layer type within 5 %.

**Overheads, measured in situ.** At every sync-mode decode's end, on the session thread,
inside `with mx.stream(generation_stream)`: the median of 20 one-element round trips
(`rt_us`), 20 evals of an already-evaluated array (`noop_us`) and 20 chains of 64 trivial
ops (`chain_us`; `(chain − rt)/63` reported as the per-dispatch encode cost). These are
cross-checks; the correction that enters the ceilings is D3's per-eval overhead
(section 3.6).

**Guards** (every post): the shim's wrappers are the outermost attributes it rebound
(else the post is `REFUSED` and a `post` record says so); `projection_kernel_status
["state"]` and `attention_tile_status["state"]` are `active` on every pass (the M2 smoke
alone sets `PROF_ALLOW_INACTIVE=1`, recorded in the install record; `analyze_prof.py`
refuses files that carry it); with a drafter, `verify_attention_status["state"]` is
`active` and `engine._draft_context.active` is true; the block matches `PROF_BLOCK`; per
round, `projkernel.calls["verify_kernel"]` and `tileattn.calls` verify paths divided by
the round count agree across modes within 1 % and `decode_tile` is 0 in drafted rounds.

### 3.3 Instrument (C): xctrace

`xcrun xctrace record --template 'Metal System Trace'` over the replay daemon: `--attach
<pid>` if Task 3 step 0 proves it, else `--launch`. Under `--launch`, xctrace runs with the
real HOME and the daemon's environment is passed explicitly through `/usr/bin/env
HOME=<isolated> HF_HUB_CACHE=… HF_HUB_OFFLINE=1 GATE_ROUNDS=… PROF_…=… <venv python>
prof.py`, so the pid is the daemon's; at the end the driver SIGTERMs the daemon pid read
from `daemon.lock`, waits for xctrace to finalise (minutes), and only then signals the
group. `GATE_XT` is the trace length in seconds (1500 for D3, 1200 for D6). Exported
tables: `metal-gpu-intervals` (rows filtered to the daemon's pid through the `process`
cell's `fmt`) and `gpu-performance-state-intervals` (column mnemonics read from the
export's schema; the parser fails if a column is missing).

**Alignment by in-band marker.** The shim calibrates at install a matmul repeat count
that runs ≈ 20 ms (recorded with its measured duration), then at each marker point runs
three such bursts separated by 50 ms sleeps and records `perf_counter()` at each submit
and return. The parser unions the pid's rows, splits them into bursts at gaps > 5 ms,
matches three bursts of the recorded duration ±30 % with 40–60 ms gaps, fits the
`perf_counter → trace time` offset from the three edges (spread < 2 ms, else it fails),
and checks the end marker lands within 1 ms. Marker points: `begin_decode` and after the
decode's last round (D3); after the last round only (D6).

**Outputs per plain post:** per decode window (round 3's `t_start` to the last round's
`t_end`) the GPU busy fraction, idle per round = `(1 − busy) × dt − mean measured gap`,
`start-latency` p50/p90, and a gap histogram (0.2–1, 1–5, > 5 ms). Gap placement (before
draft, draft → verify, mid-verify, walk) comes from the traced `lazy` post of the same
turn, whose proxy stamps mark the draft and verify submissions.

**Outputs per sync post:** every stamped eval (`pre`, `eval`, proxy evals and async_evals,
the step's eval) is mapped into trace time; each `metal-gpu-intervals` row is assigned to
the event whose [call, next call) window contains its start; a round is valid when every
real-sync leaf eval received at least one interval. The summed interval duration is that
eval's pure GPU time, and `host_eval − burst` its encode and sync overhead.

### 3.4 In-process microbench (drafted daemon, at every turn end, session thread; never in D6)

48 independent copies of layer 1's real shapes with fresh random inputs of the real dtype
(q, k `(1, T, 16, 128)` bf16; v `(1, T, 48, 128)` bf16; a, b `(1, T, 48)` bf16; `A_log`,
`dt_bias` from layer 1; state `(1, 48, 128, 128)` fp32; no cache), called as
`gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=S, mask=None, use_kernel=True)`
and, for the with-states variant, `state_steps=4` (three outputs, all evaluated). Per rep,
the 48 calls are built and evaluated in **one** `mx.eval`; 20 interleaved reps per variant
(T = 1, T = 5, T = 5 with states); `kernel_X = median(chain) − rt_us` of the same loop.
L1's floor is `kernel_T1`; H3's store cost is `kernel_T5_states − kernel_T5`.

### 3.5 Passes (daemons; one isolated HOME `~/sous-prof-home` shared by all; forks kept until the chain ends, then `rm -rf ~/sous-prof-home/.sous/forks`)

Every turn's post sequence is: k−1 (untimed warm-up), k once (untimed, mode `plain`),
then the listed timed posts of k.

| # | label | config | timed posts per turn (mode sequence) | turns | tokens | purpose |
|---|---|---|---|---|---|---|
| D0 | warmforks | production | one post per distinct tools hash among all turns below, `max_tokens 1`, untimed | ≈ 5–10 | 1 | write the tools forks to disk once |
| D1 | on | drafter on, `speculative_block_size = 6`, shim lifts `SPECULATIVE_BLOCK_MAX` to 6, arms `5k,6k` drawn per round (seed 180) | plain, lazy, sync, syncfine, syncfine, sync, lazy, plain | 18 | 512 | H1–H3, H5–H7, H10; microbench |
| D2 | off | drafter off (`speculative_draft_id = ""`) | plain, sync, syncfine, syncfine, sync, plain | 18 | 512 | T = 1 stage shares; step time |
| D3 | xt | production (block 5, no arms), under xctrace | plain, lazy, sync, syncfine, plain (traced); with `--attach`, plus plain, plain untraced | 3 | 512 | H5 idle, per-eval overheads (3.6), threat 8 |
| D4 | long | production | plain, syncfine, syncfine, plain | 2 | 256 | attention scaling (report-only) |
| D5 | greedy | production, temperature 0 | plain, sync, plain | 3 | 512 | greedy verify path (report-only) |
| D6 | warm | production, `GATE_NOPOLL=1`, `PROF_CANARY=0`, `PROF_MICROBENCH=0`, traced whole or not at all | 12 timed posts of the keep-warm turn, sleeps before each of 0, 1, 3, 10, 30, 90, 90, 30, 10, 3, 1, 0 s | 1 | 128 | H8 / L6 |

Budget (prefill as its own term; the k−1 warm-up restores the tools fork and prefills to
`n_{k−1}`: ≈ 60 s mean, 2 min at 80K; a timed post prefills the k−1 → k delta, ≈ 8 s, and
decodes ≈ 5 s plain or 10–15 s sync): D0 ≈ 15 min; D1 ≈ 18 × (60 + 13 + 8 × 16) s ≈ 60 min
plus the microbench ≈ 3 min; D2 ≈ 18 × (60 + 13 + 6 × 14) s ≈ 47 min; D3 ≈ 9 min plus
export; D4 ≈ 11 + 22 min of cold prefill plus posts ≈ 42 min; D5 ≈ 6 min; D6 ≈ 10 min;
seven loads ≈ 12 min; one repost per mode for a third of the turns ≈ 15 min. Total ≈ 3.7 h
against a 5 h cap. Drop order if the cap approaches: D5, D4, D2 (first shrunk to the six
turns nearest 46K/56K/72K, two each, then dropped), D6. D1 and D3 are never dropped. If the
running total passes 4 h at a daemon boundary, the chain stops there and resumes in a
second session from the next daemon. Each job runs under `perl -e 'alarm 7200; exec
@ARGV'`.

### 3.6 Exclusions, counting, statistics

- Rounds excluded: rounds 1–2 of every decode, budget-truncated rounds, the EOS round, any
  round with the `trunc` flag, rounds of refused posts, rounds of posts whose take was a
  `fork`, `disk` or `miss` (D1–D5), and rounds of posts whose canary deviates > 5 % from
  the median of the same turn's posts (listed per turn). Decodes with fewer than 8 kept
  rounds are excluded (as #157's stratification did) and count toward the repost trigger;
  they never void a daemon. Steps (T = 1): the same, by decode token index.
- A daemon's data counts when `gpu_run_m5` logs the daemon idle and uncontaminated, the
  job exits 0, every post is HTTP 200, the load line shows every exact path active, and the
  floors hold: ≥ 30 kept rounds per turn, mode **and arm** in D1 (5k and 6k separately),
  ≥ 30 per turn and mode in D2–D5. The driver's repost loop reads the rounds file and
  reposts a turn in a mode (≤ 2 extra posts) while it is under the floor; a turn still
  short is replaced by the first reserve turn of its third (posted at the daemon's end) and
  the replacement is stated. A daemon that fails a validity line is re-run once under a `b`
  label, which replaces it; a `b` that also fails voids that daemon and every decision that
  needs it goes to the maintainer.
- Per turn, mode and arm: the median over kept rounds of each bucket. Headline: for each
  quantity, OLS of the per-turn median on `(n − 56,000)` over the 18 turns, reported at 56K
  with a 95 % percentile bootstrap (4,000 resamples, refit each; clusters = subagents when
  any subagent contributes two turns) and the t-interval (df = clusters − 2) beside it.
  Thirds are descriptive (ranges, no CI). `own5` is the fitted plain `own` of the 5k rounds
  at 56K in D1; every share is `ceiling / own5` bootstrapped jointly on the same resamples.
  H7 and the lazy-overhead check are paired within turn.
- GPU-side correction: for each bucket and fine layer, D3's sync and syncfine posts give an
  additive per-eval overhead `o_b = median over valid rounds of (host_eval_b − burst_b) /
  n_evals_b`. D1's corrected bucket = raw `host_eval_b − n_evals_b × o_b`, with no `rt` or
  `noop` subtraction on top (the `rt`-adjusted value is a cross-check only). The ceiling
  bootstrap resamples D3 rounds for `o_b` jointly with D1 turns. A lever whose ceiling
  uses a bucket without a valid pairing cannot be spike-worthy or closed: it goes to the
  maintainer, flagged *encode-inclusive*.

## 4. Decision rules (pre-registered; the output is a ranked lever list)

Every ceiling is a GPU-side quantity at 56K unless noted. The bar is **4 % of `own5`**
(≈ 4.8 ms): a spike that realises half its ceiling then still reaches the +2 % ship bar the
project's gates use (#157's rule). Three outcomes per lever, on the bootstrap of
`ceiling / own5` (4,000 resamples, ceiling and `own5` refit on the same resample):

- **spike-worthy**: point ≥ 4 % and the 95 % lower bound ≥ 2 %;
- **closed**: the 95 % upper bound < 4 %;
- **to the maintainer**: anything else. Residue clause: if the 95 % bootstrap CI of the
  mean signed closure residue (ms, at 56K) excludes 0, every lever whose `ceiling_ms` is
  below the CI's larger absolute bound goes to the maintainer; otherwise the clause does
  not apply.

Ceilings (ranked, never summed: L1 and L2 both claim prework host cost; L2 and L8 both
claim walk idle, so L2 is also reported excluding walk-placed idle):

- **L1 fused GDN verify (deferred state commit + prework fusion).**
  `ceiling = gdn_chain(T=5) − kernel_T1` (section 3.4). If H7 puts ≥ 50 % of the paired
  6k − 5k difference in `gdn_chain`, an L1 spike re-tests block 6.
- **L2 host-bound idle (`async_eval` flush, prework host cost).**
  `ceiling = idle per round inside own` from D3's plain posts (no CI: three turns, reported
  per turn; spike-worthy if the smallest of the three is ≥ 4 %, closed if the largest is
  < 2 %, else maintainer; under `--launch`, any canary-normalised `r > 1.03` sends it to
  the maintainer). Reported beside it, as the explanation and not the bound:
  `max(0, verify.build + verify.encode − draft_gpu)`, `draft.build + draft.encode`, and
  `walk.pingpong`.
- **L3 drafter (fusions, routing).** `ceiling = draft_gpu − (own_param_bytes + head_bytes
  + drafter_kv_bytes) / 260 GB/s`: `own_param_bytes` = Σ `nbytes` over the drafter's
  parameters excluding every array reachable from its bound `embed_tokens` and `lm_head`
  (identity check against the target's modules); `head_bytes` = the target `lm_head`'s
  `nbytes`; `drafter_kv_bytes` = Σ over its cache arrays at turn end. `proj_out@draft` is
  reported, not subtracted.
- **L4 INT8 KV through the tile.** `ceiling = 0.5 × sdpa(T=5)`.
- **L8 positioned target sampling (one sync per round).** `ceiling = Σ over pingpong evals
  of (host − burst GPU)` from D3 when the pairing is valid (the per-position sampler GPU
  work stays under a batched sampler), else `walk.pingpong` (lazy) flagged
  *encode-inclusive*; `walk.pingpong − rt` reported as the conservative value.
- **L5 T = 6.** No ceiling of its own; #157's close stands unless L1's rule above fires.
- **L6 keep-warm ticker.** `Penalty(g) = mean over the pair at g of [ttft_s − mean ttft_s
  of the g = 0 pair] + Σ_{r=1..5} (own_r − mean over the g = 0 pair of own_r)`, in seconds
  (`ttft_s` read unrounded from `/sous/status`). Implement if `Penalty(3) ≥ 0.5 s` and
  `Penalty(10) ≥ 0.5 s` and the two g = 0 posts' `ttft_s` differ by < 0.15 s; unresolved
  otherwise (to the maintainer). The lower bar is deliberate: a keep-warm ticker carries no
  parity risk.
- **L7 progress summaries.** If summary turns are ≥ 5 % of local engine seconds in the
  cap-1 runs, or Σ `queue_s` charged to real turns ≥ 2 % of their wall clock, the number
  goes to the maintainer with the identification method; no sous change is pre-committed.

Output: L1–L4 and L8 sorted by ceiling as a share of `own5`, each with its 56K value, CI,
third ranges and flags. The next M5 spike is the top spike-worthy lever. If none is
spike-worthy and none is "to the maintainer", the profile closes the decode-kernel line
for the 27B on this GPU, and the report says what remains (weights and model choice, out
of scope here).

## 5. Report

`results.md` in the harness directory: the known-facts table beside the measured values
(H1–H10 each held / failed with the number), the bucket table at T = 1, 5 and 6 (host
`build`, host `eval`, burst GPU, corrected value, share of `own5`, CI; third ranges), the
closure line, the xctrace busy/idle figures, gap placement and `start-latency`, the
traced/untraced check, the keep-warm table, the summary-share numbers with the
unclassified count, the lever table with ceilings and the outcome each rule produced, the
ledger and the amendments log. The maintainer sees it before anything reaches GitHub; the
memory file records the findings the same day.

---

## Task 1: Harness skeleton, pins and the fixed turn list

**Files:**
- Create: `~/code/sous-spikes/prof-b5/README.md` (sections 2–4 of this plan verbatim, the
  turn list with predecessors and reserves, the long pairs, the keep-warm turn, an empty
  "Amendments" section, the tree hash)
- Create: `~/code/sous-spikes/prof-b5/pins.py`, `pick_turns.py`
- Copy unchanged: `p157/gate157.py` (driver base), `p148/gate/rounds_gate.py`,
  `p157/wrap157.py` (reference only). On the M5 only: symlink p157's `build_replay.py`
  and `replay-157.json` into `prof-b5/` (the driver resolves data through
  `P157 = ~/code/personal/sous-spikes/p157`). `gpu_run_m5.py` is **not** copied (its lock
  lives in its own directory; the chain calls the shared copy by path).

**Interfaces:**
- Produces: `pins.PINNED` and `pins.check()` (raises `SystemExit` naming the first drifted
  function); `turns.txt` (18 lines `third basename@index n_k n_{k-1} recorded_out target
  key`), `reserve.txt`, `long.txt` (two `basename@index n_k n_{k-1}` lines), `warm.txt`.

- [ ] **Step 1: Copy the reference scripts**

```bash
mkdir -p ~/code/sous-spikes/prof-b5 && cd ~/code/sous-spikes/prof-b5
cp ../p157/gate157.py ../p148/gate/rounds_gate.py . && cp ../p157/wrap157.py ref_wrap157.py
```

- [ ] **Step 2: Write `pins.py`**

```python
"""Source pins for everything prof.py rebinds or times (mlx-vlm 0.7.2 and sous main;
tileattn/verifyattn entries pin sous sources, so an edit there forces a re-pin)."""
import hashlib, inspect
import mlx.nn as nn
from mlx_vlm.speculative import cache_state, dflash
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
from mlx_vlm.models.qwen3_5 import language, speculative_verifier
from sous.engine import tileattn, verifyattn

V = speculative_verifier.Qwen3_5BatchInvariantForward

def _digest(fn):
    return hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest()[:16]

# Digests are filled by `python pins.py --print` on the M5 tree before the first timing
# job; rounds_gate's four digests are reused as they are.
PINNED = {
    "_dflash_rounds": (dflash._dflash_rounds, "eee8dc9ab95a74ed"),
    "_dflash_verify": (dflash._dflash_verify, ""),
    "_dflash_verify_greedy": (dflash._dflash_verify_greedy, ""),
    "_sample_dflash_target_walk": (dflash._sample_dflash_target_walk, "6cb5e7aebde0b9f6"),
    "_speculative_walk": (dflash._speculative_walk, "4487bf2099c8d3a6"),
    "commit_speculative_round": (cache_state.commit_speculative_round, "58c3e8cf462dea7a"),
    "DFlash.draft_block": (DFlashDraftModel.draft_block, ""),
    "DFlash.draft_block_greedy": (DFlashDraftModel.draft_block_greedy, ""),
    "DFlash2.draft_block": (DFlash2DraftModel.draft_block, ""),
    "V._layer": (V._layer, ""), "V._gated_delta": (V._gated_delta, ""),
    "V._attention": (V._attention, ""), "V._feed_forward": (V._feed_forward, ""),
    "V._model": (V._model, ""), "V.__call__": (V.__call__, ""),
    "V._linear": (V._linear, ""), "V._linears": (V._linears, ""),
    "gated_delta_update": (speculative_verifier.gated_delta_update, ""),
    "language.gated_delta_update": (language.gated_delta_update, ""),
    "DecoderLayer": (language.Qwen3_5DecoderLayer.__call__, ""),
    "GatedDeltaNet": (language.Qwen3_5GatedDeltaNet.__call__, ""),
    "Attention": (language.Qwen3_5Attention.__call__, ""),
    "MLP": (language.Qwen3_5MLP.__call__, ""),
    "RMSNormGated": (language.Qwen3_5RMSNormGated.__call__, ""),
    "Qwen3_5Model": (language.Qwen3_5Model.__call__, ""),
    "LanguageModel": (language.LanguageModel.__call__, ""),
    "QuantizedLinear": (nn.QuantizedLinear.__call__, ""),
    "tile": (tileattn.tile, ""), "grouped_attention": (verifyattn.grouped_attention, ""),
}

def check():
    for name, (fn, want) in PINNED.items():
        got = _digest(fn)
        if got != want:
            raise SystemExit(f"prof-b5: pin drift {name}: {got} != {want}")

if __name__ == "__main__":
    for name, (fn, _) in PINNED.items():
        print(f"{name}: {_digest(fn)}")
```

- [ ] **Step 3: Fill the digests from the M5 tree**

```bash
scp -q pins.py m5pro:~/code/personal/sous-spikes/prof-b5/pins.py
ssh m5pro 'cd ~/code/personal/sous-remote && git fetch -q && git checkout -q main && git pull -q --ff-only && uv sync -q && .venv/bin/python ~/code/personal/sous-spikes/prof-b5/pins.py --print && git rev-parse --short HEAD'
```

Expected: 29 lines `name: <16 hex>` and the tree hash; the four reused digests match.
Paste the digests into `PINNED` and the hash into README "Tree:". The same digests must
print on the M2 worktree (same lock, same sources), which Task 2's smoke relies on.

- [ ] **Step 4: Write `pick_turns.py` and fix the turn list (runs on the M5, inside p157)**

```python
"""Fixed profile set from p157's replay list (section 3.1). Reads replay-157.json,
picks145.txt and the transcript copies through gate157's turns_of; never a timing.
usage: python pick_turns.py ~/code/personal/sous-spikes/p157"""
import hashlib, json, sys
P = sys.argv[1]; sys.path.insert(0, P)
from gate157 import turns_of, parse_picks          # gate157 dispatches old/new builders
data = {r["file"].rsplit("/", 1)[-1]: r for r in json.load(open(f"{P}/replay-157.json"))}
order = parse_picks(open(f"{P}/picks145.txt").read().split(), data)   # [(base, i), ...], old picks included
MIN_OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 200

def content_key(t):   # analyze157's cluster key: converted messages, tools, max_tokens
    return hashlib.sha256(json.dumps([t["messages"], t["tools"], 512], sort_keys=True, default=str).encode()).hexdigest()[:16]

pool = []
for base, i in order:
    turns = [t for t in turns_of(data[base]["file"]) if t["tools"] is not None]
    rec = data[base]["turns"]
    n = {t["k"]: t["n"] for t in rec}.get(i)
    if n is None or not {t["k"]: t.get("eligible", True) for t in rec}[i] or turns[i].get("compacted"): continue
    out = ((turns[i].get("usage") or {}).get("output_tokens")) or 0
    if out < MIN_OUT: continue
    prev = i - 1 if i >= 1 else None
    pool.append({"base": base, "i": i, "n": n, "prev": prev, "out": out, "key": content_key(turns[i])})
seen_keys, picked = set(), []
pool = [p for p in pool if not (p["key"] in seen_keys or seen_keys.add(p["key"]))]
used_sub, used_key = {}, set()
targets = [44000 + (j + 0.5) * 2000 for j in range(18)]
def take(t, max_per_sub):
    cands = [p for p in pool if abs(p["n"] - t) <= 2000 and p["key"] not in used_key and used_sub.get(p["base"], 0) < max_per_sub]
    if not cands: return None
    p = min(cands, key=lambda p: abs(p["n"] - t)); used_key.add(p["key"]); used_sub[p["base"]] = used_sub.get(p["base"], 0) + 1
    return p
skipped = []
for t in targets:
    p = take(t, 1)
    if p is None: skipped.append(t)
    else: picked.append((t, p))
for t in skipped:
    p = take(t, 2)
    if p is not None: picked.append((t, p))
picked.sort(key=lambda x: x[1]["n"])
for j, (t, p) in enumerate(picked):
    third = ("A", "B", "C")[min(2, j * 3 // max(1, len(picked)))]
    print(f"{third} {p['base']}@{p['i']} {p['n']} prev={p['prev']} out={p['out']} target={int(t)}")
print(f"# pool {len(pool)} distinct; picked {len(picked)}; skipped targets {[int(t) for t in skipped if all(pp[0] != t for pp in picked)]}", file=sys.stderr)
```

Also print the per-third candidate counts before selection and the three reserves per
third (the next nearest unused pool turns). Expected: 18 picks (any skipped target and
any second-per-subagent fill stated in README); `prev` present for every pick that is not
a subagent's first turn. Spot-check `usage.output_tokens` against one long answer's text.
If `gate157` exposes no `parse_picks`, parse as it does (`base@i,j`, `base:cap`, bare
base). Then write `turns.txt`, `reserve.txt`, `long.txt` (the two long turns, whose
predecessors the driver derives as index − 1) and `warm.txt` (the keep-warm turn).

## Task 2: The stage-timer shim `prof.py`

**Files:**
- Create: `~/code/sous-spikes/prof-b5/stages.py`, `prof.py`

**Interfaces:**
- Consumes: `rounds_gate.install(arms, seed=)`, `.begin_decode()`, `.drain()`,
  `pins.check()`.
- Produces (one JSONL, `GATE_ROUNDS`, absolute): `install` {`mode_file`, `block`, `arms`,
  `seed`, `greedy`, `drafter_param_bytes`, `own_param_bytes`, `head_bytes`, `wall0`,
  `marker_reps`, `marker_ms`, `proj`, `tile`, `verify_attention`, `allow_inactive`};
  `round` (rounds_gate's fields plus `post`, `mode`, `turn_key`, `took`, `canary_dev`,
  `wall`, `stages` {`"<parent>/<stage>/<layer>"`: {`n`, `n_sync`, `n_noop`, `build`, `pre`,
  `eval`, `stamps`: [[call, ret], …]}}, `walk` {`entry`, `exit`, `first_wait`, `pingpong`,
  `syncs`}, `encode` {`draft`, `verify`}, `drafter_kv_bytes`); `step` (T = 1: `decode`,
  `round`, `dt`, `own`, plus the same extras); `overhead` per sync decode {`rt_us`,
  `noop_us`, `chain_us`}; `microbench` per drafted turn {`kernel_T1`, `kernel_T5`,
  `kernel_T5_states`}; `marker` {`post`, `where`, `submit`: [3], `done`: [3]}; `post`
  {`post`, `mode`, `refused`: reason|null, `canary_ms`, `bw_canary_ms`}.
  Environment: `PROF_MODE_FILE` or `PROF_MODE_SEQ`, `PROF_BLOCK` (`5`, `6`, `0` = no block
  check), `PROF_ARMS`, `PROF_SEED`, `GATE_ROUNDS`, `PROF_ALLOW_INACTIVE`, `PROF_MARKER`,
  `PROF_CANARY` (default 1), `PROF_MICROBENCH` (default 1).

- [ ] **Step 1: Write `stages.py`**

```python
"""Stage timers with a parent stack, a layer counter and the leaf protocol (pre-eval of
inputs, then call, then eval of outputs). Records accumulate per round in CUR."""
from __future__ import annotations
import collections, functools, time
import mlx.core as mx, mlx.nn as nn

MODE = "plain"
ALLOW = None                      # lazymin: a set of stage names; None = all
FINE_LAYERS = {1, 2, 4, 7}
HEAD_PARENTS = ("verify_call", "step", "draft")
SYNC_SPLIT_US = 30.0              # reset per turn to the midpoint of noop_us and rt_us
STACK: list[str] = []
LAYER = [-1]
ORD: dict = collections.Counter()  # proj ordinal per parent within a round
CUR: dict = collections.defaultdict(lambda: {"n": 0, "n_sync": 0, "n_noop": 0, "build": 0.0, "pre": 0.0, "eval": 0.0, "stamps": []})

def _arrays(obj):
    if isinstance(obj, nn.Module): return []
    if isinstance(obj, mx.array): return [obj]
    if isinstance(obj, (tuple, list)): return [a for o in obj for a in _arrays(o)]
    if isinstance(obj, dict): return [a for o in obj.values() for a in _arrays(o)]
    if hasattr(obj, "logits"): return _arrays(obj.logits) + _arrays(obj.hidden_states or [])
    return []

def _sync(arrs, rec):
    t0 = time.perf_counter(); mx.eval(*arrs); t1 = time.perf_counter()
    rec["stamps"].append([t0, t1])
    rec["n_sync" if (t1 - t0) * 1e6 > SYNC_SPLIT_US else "n_noop"] += 1
    return t1 - t0

def parent():
    return STACK[-1] if STACK else "-"

def key(stage):
    return f"{parent()}/{stage}/{LAYER[0]}"

def leaf_now(stage, fine_only):
    if MODE not in ("sync", "syncfine"): return False
    if stage == "layer" or stage == "draft": return True
    if stage.startswith("proj") and parent() in HEAD_PARENTS: return True
    return fine_only and MODE == "syncfine" and LAYER[0] in FINE_LAYERS

def timed(stage, *, fine_only=False, extra_inputs=None, ordinal=False):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if MODE == "plain" or stage in STACK or (ALLOW is not None and stage not in ALLOW):
                return fn(*a, **k)
            name = f"{stage}#{ORD[(parent(), stage)]}" if ordinal else stage
            if ordinal: ORD[(parent(), stage)] += 1
            rec = CUR[key(name)]
            rec["n"] += 1
            do_sync = leaf_now(stage, fine_only)
            if do_sync:
                ins = _arrays(a) + _arrays(k) + (_arrays(extra_inputs(*a, **k)) if extra_inputs else [])
                if ins: rec["pre"] += _sync(ins, rec)
            STACK.append(stage)
            t0 = time.perf_counter()
            try:
                out = fn(*a, **k)
            finally:
                STACK.pop()
            rec["build"] += time.perf_counter() - t0
            if do_sync:
                outs = _arrays(out)
                if outs: rec["eval"] += _sync(outs, rec)
            return out
        wrapper._prof_stage = stage
        return wrapper
    return deco

def layer_counter(fn):          # applied OUTSIDE timed("layer"), so the key sees the new index
    @functools.wraps(fn)
    def wrapper(*a, **k):
        LAYER[0] += 1
        return fn(*a, **k)
    return wrapper

def resets_layers(fn):           # applied to the model wrappers
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try: return fn(*a, **k)
        finally: LAYER[0] = -1
    return wrapper

def begin_round():
    CUR.clear(); STACK.clear(); ORD.clear(); LAYER[0] = -1

def drain_round() -> dict:
    out = {kk: dict(v) for kk, v in CUR.items()}
    begin_round()
    return out

def overhead(stream, n=20):
    x = mx.array([1.0]); mx.eval(x)
    with mx.stream(stream):
        rt = sorted(_t(lambda: mx.eval(x + 1)) for _ in range(n))[n // 2]
        noop = sorted(_t(lambda: mx.eval(x)) for _ in range(n))[n // 2]
        def chain():
            y = x
            for _ in range(64): y = y + 1
            mx.eval(y)
        ch = sorted(_t(chain) for _ in range(n))[n // 2]
    return {"rt_us": rt * 1e6, "noop_us": noop * 1e6, "chain_us": ch * 1e6}

def _t(f):
    t = time.perf_counter(); f(); return time.perf_counter() - t
```

- [ ] **Step 2: Write `prof.py`**

```python
"""GATE_WRAP shim for the block-5 round profile (out of tree, never committed)."""
from __future__ import annotations
import functools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sous.config
BLOCK = int(os.environ.get("PROF_BLOCK", "5"))
if BLOCK == 6:
    sous.config.SPECULATIVE_BLOCK_MAX = 6
import pins; pins.check()
import stages, rounds_gate
import mlx.core as mx, mlx.nn as nn
from mlx_vlm.speculative import common, dflash
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
from mlx_vlm.models.qwen3_5 import language, speculative_verifier
from sous.engine import projkernel, tileattn, verifyattn, vlm

MODE_FILE = os.environ.get("PROF_MODE_FILE"); MODE_SEQ = [m for m in os.environ.get("PROF_MODE_SEQ", "").split(",") if m]
ROUNDS = os.environ["GATE_ROUNDS"]
ARMS = [a for a in os.environ.get("PROF_ARMS", "").split(",") if a]
SEED = int(os.environ.get("PROF_SEED", "180"))
ALLOW_INACTIVE = os.environ.get("PROF_ALLOW_INACTIVE") == "1"
MARKER = os.environ.get("PROF_MARKER") == "1"
CANARY = os.environ.get("PROF_CANARY", "1") == "1"
MICROBENCH = os.environ.get("PROF_MICROBENCH", "1") == "1"
assert os.path.isabs(ROUNDS) and os.path.isdir(os.path.dirname(ROUNDS)), "GATE_ROUNDS must be absolute, in an existing directory"
if os.path.exists(ROUNDS): raise SystemExit(f"prof-b5: {ROUNDS} exists; use a fresh path per daemon")

# Import-time drafter wraps: draftctx binds the instance methods by getattr at enable(),
# after this, so it picks up the timed methods (its tail feeding runs outside the timer;
# round 1 is excluded anyway). Both classes, so the M2's 9B drafter is covered too.
for cls, name in ((DFlashDraftModel, "draft_block"), (DFlashDraftModel, "draft_block_greedy"), (DFlash2DraftModel, "draft_block")):
    setattr(cls, name, stages.timed("draft")(getattr(cls, name)))

V = speculative_verifier.Qwen3_5BatchInvariantForward
_orig: dict = {}; _wrapped: dict = {}
_evals: list = []       # proxy stamps of the current round: (name, parent, call, ret)

class _MxProxy:
    """Stamps eval/async_eval under the current parent; everything else is mlx.core."""
    def __getattr__(self, name):
        if name in ("eval", "async_eval"):
            real = getattr(mx, name)
            def call(*a, **k):
                t0 = time.perf_counter(); out = real(*a, **k)
                _evals.append((name, stages.parent(), t0, time.perf_counter())); return out
            return call
        return getattr(mx, name)

def _targets(engine):
    tagged = lambda m: getattr(m, projkernel._TAG, False)
    norm_cls = type(engine._model.language_model.model.layers[0].linear_attn.norm)
    state_in = lambda *a, cache=None, **k: [cache.cache[1]] if cache is not None and hasattr(cache, "cache") else []
    layer = lambda f: stages.layer_counter(stages.timed("layer")(f))
    model = lambda f: stages.resets_layers(stages.timed("model")(f))
    t = [
        (V, "_linears", stages.timed("proj_in", fine_only=True)), (V, "_linear", stages.timed("proj_out", fine_only=True)),
        (V, "_gated_delta", stages.timed("gdn")), (V, "_attention", stages.timed("attn")),
        (V, "_feed_forward", stages.timed("mlp")), (V, "_model", model), (V, "__call__", stages.timed("verify_call")),
        (V, "_layer", layer),
        (speculative_verifier, "gated_delta_update", stages.timed("gdn_kernel", fine_only=True, extra_inputs=state_in)),
        (language, "gated_delta_update", stages.timed("gdn_kernel", fine_only=True, extra_inputs=state_in)),
        (norm_cls, "__call__", stages.timed("gdn_norm", fine_only=True)),
        (tileattn, "tile", stages.timed("sdpa", fine_only=True)),
        (verifyattn, "grouped_attention", stages.timed("sdpa", fine_only=True)),
        (language.Qwen3_5DecoderLayer, "__call__", layer),
        (language.Qwen3_5GatedDeltaNet, "__call__", stages.timed("gdn")),
        (language.Qwen3_5Attention, "__call__", stages.timed("attn")),
        (language.Qwen3_5MLP, "__call__", stages.timed("mlp")),
        (language.Qwen3_5Model, "__call__", model),
        (nn.QuantizedLinear, "__call__", lambda f: _only_tagged(f, stages.timed("proj", fine_only=True, ordinal=True)(f), tagged)),
        (dflash, "_dflash_verify", stages.timed("verify")),
        (dflash, "_dflash_verify_greedy", stages.timed("verify_greedy")),
        (dflash, "mx", lambda _: _MxProxy()), (common, "mx", lambda _: _MxProxy()),
    ]
    return t

def _only_tagged(orig, timed_fn, tagged):
    @functools.wraps(orig)
    def call(self, x, *a, **k):
        return (timed_fn if tagged(self) else orig)(self, x, *a, **k)
    return call

def _install(engine):
    for owner, name, make in _targets(engine):
        k = (owner, name)
        if k not in _orig: _orig[k] = getattr(owner, name)    # AttributeError here = a bad target; fail loudly
        _wrapped[k] = make(_orig[k]); setattr(owner, name, _wrapped[k])
    for name in ("_sample_dflash_target_walk", "_speculative_walk"):   # outside rounds_gate's stamps
        k = (dflash, name)
        if k not in _orig: _orig[k] = getattr(dflash, name)
        _wrapped[k] = _walk_wrapper(_orig[k]); setattr(dflash, name, _wrapped[k])

def _restore():
    for (owner, name), orig in _orig.items(): setattr(owner, name, orig)

def _guard(engine) -> str | None:
    if stages.MODE != "plain":
        for (owner, name), w in _wrapped.items():
            if getattr(owner, name) is not w: return f"{getattr(owner, '__name__', owner)}.{name} is no longer outermost"
    if not ALLOW_INACTIVE:
        for attr in ("projection_kernel_status", "attention_tile_status"):
            if getattr(engine, attr).get("state") != "active": return f"{attr} is not active"
        if engine._draft is not None:
            if engine.verify_attention_status.get("state") != "active": return "verify_attention is not active"
            if not engine._draft_context.active: return "draft_context is not active"
    if BLOCK and engine._draft is not None and engine._draft_block_size != BLOCK:
        return f"block {engine._draft_block_size} != PROF_BLOCK {BLOCK}"
    return None
```

The rest of `prof.py`, in code when written (its contracts): `install()` wraps
`vlm.VLMEngine.generate` as `wrap_gate.install()` does and ends with `from sous.cli import
main; main(sys.argv[1:] or ["serve"])`. The generate wrapper, per post: reads the mode
(`warmup` → `plain`; `lazymin` sets `stages.ALLOW`); on the process's first call installs
`rounds_gate.install(ARMS or None, seed=SEED)`, wraps `dflash._dflash_next_block_size`
so `stages.begin_round()` and `_evals.clear()` run before every round, calls
`_install(engine)` once, calibrates the marker, measures parameter bytes and writes the
`install` record; restores on plain posts and re-installs on non-plain ones; refuses on
`_guard` (writing a `post` record); records `took` from the engine's per-turn gauges;
after the decode, in D1–D5, times the two canaries and, in sync modes, writes the
`overhead` record (`stages.overhead(common.generation_stream)`) and resets
`stages.SYNC_SPLIT_US`; in drafted daemons with `MICROBENCH`, runs section 3.4 and writes
its record; runs the marker at `begin_decode` (unless `PROF_CANARY=0`, the D6 setting)
and after the last round when `MARKER`. It attaches `stages.drain_round()`, the proxy
events (`encode`, `walk` fields), `mode`, `post`, `turn_key`, `took`, `canary_dev` and
`wall` to each round by wrapping `rounds_gate._record` (a module global the recorder
looks up per call) to set them on `rounds_gate.S.pending` before delegating. The T = 1
step recorder is installed when `engine._draft is None` and acts per section 3.2; in
plain mode it stamps `dt` only. The walk wrapper pushes `walk`, stamps entry/exit and
reads the proxy events made inside it.

- [ ] **Step 3: Smoke on the M2 (9B + its DFlash drafter; kernel and tile unavailable there)**

Through `p158/stage2/gate158.py` (synthetic bodies, a `[model]` block, `GATE_TURNS`
posts per daemon), one daemon per mode list, fresh labels and rounds files:

```bash
cd ~/code/sous-spikes/prof-b5 && mkdir -p /tmp/prof
for seq in plain,syncfine,plain sync,lazy; do
GATE_WRAP=$PWD/prof.py PROF_MODE_SEQ=$seq PROF_BLOCK=3 PROF_ALLOW_INACTIVE=1 \
GATE_ROUNDS=/tmp/prof/smoke-$seq.jsonl GATE_PORT=8384 GATE_TURNS=3 \
GATE_BODIES=~/code/sous-spikes/p148/m2/bodies-small.json \
GATE_MODEL_TOML='id = "mlx-community/Qwen3.5-9B-MLX-4bit"
speculative_draft_id = "z-lab/Qwen3.5-9B-DFlash"
speculative_block_size = 3
idle_unload_minutes = 1440
' python3 ~/code/sous-spikes/p158/stage2/gate158.py smoke-m2-$seq ~/code/sous/.claude/worktrees/omlx-prioritization-qwen-8d665c 0.7 64
done
```

Expected: the install record carries `allow_inactive: true`; syncfine rounds have keys
`-/draft/-1`, `-/verify/-1`, `verify/verify_call/-1`, `verify_call/model/-1`,
`model/layer/0..L−1` with `n_sync == L`, `gdn/proj_in/<i>`, `gdn/gdn_kernel/<i>`,
`gdn/gdn_norm/<i>`, `gdn/proj_out/<i>`, `attn/proj_in/<i>`, `mlp/proj_in/<i>`,
`verify_call/proj_out/-1` with `n_sync ≥ 1` (the head), `draft/proj_out/-1` with `n_sync
≥ 1`; `pre` and `eval` > 0 and `n_sync` > 0 for inner leaves only in layers 1, 2, 4, 7 (or
the 9B's equivalents); `pre` of `gdn/gdn_kernel` > 0 and of `gdn/gdn_norm` ≈ 0;
`model/layer/3` has `attn/*` children; the `walk` field has `syncs ≥ 1`; the `overhead`
record has `rt_us` in 20–500; `sdpa` keys are optional on the M2. Plain rounds carry no
stages and `V._linear is _orig[(V, "_linear")]` after a plain post; `sync` rounds carry
`layer` evals with `n_sync == L` and head evals, no inner `pre`; lazy rounds carry `build`
only. With `PROF_BLOCK=4` the guard refuses for the block reason, and a `_targets` entry
whose owner lacks the attribute fails the install loudly (test by adding one).

- [ ] **Step 4: Lazy overhead on the M2 (report only; the M5 check in D1 is binding)**

`PROF_MODE_SEQ=plain,lazy,lazy,plain`, `GATE_TURNS=4` on one body, 128 tokens; record the
paired difference of median `own`.

- [ ] **Step 5: Reviewer gate**

A fresh reviewer reads `prof.py`, `stages.py` and this task against threats 1–3 and
confirms: leaf pre-eval before the call and eval after; the call-time leaf rule (layer and
draft always, head parents always, inner leaves only in fine layers of syncfine); the layer
counter outside the timer and the reset in the model wrappers; `n_sync`/`n_noop` counted
only on evals with the in-situ threshold; `_arrays` skips modules; recorder and timers
installed at the first `generate()` and restored on plain posts; the drafter wraps at
import on both classes; the walk wrappers rebound after `rounds_gate.install`; the step
recorder installed only drafter-off, filtered and kept in plain mode; the `QuantizedLinear`
wrap tagged-only; nothing writes to disk from the session thread except at post end.

## Task 3: xctrace capture and parser

**Files:**
- Create: `~/code/sous-spikes/prof-b5/xt_probe.sh`, `xt_parse.py`

**Interfaces:**
- Produces: `xt_parse.py <trace> <pid> <rounds.jsonl> <label>` prints JSON per post
  `{"post": k, "aligned": bool, "offset_s": x, "decodes": [{"busy": x, "idle_ms_per_round":
  x, "gaps": {...}, "placement": {...}, "start_latency_us": {"p50": x, "p90": x}}],
  "evals": [{"round": r, "key": "...", "seq": i, "host_ms": x, "gpu_ms": x, "valid": bool}],
  "pstates": [...]}`.

- [ ] **Step 0: Probes on the M5 (exempt from the freeze; outcomes select pre-registered branches)**

```bash
ssh m5pro 'PY=$HOME/code/personal/sous-remote/.venv/bin/python; PY=$PY python3 -c "import subprocess,os; subprocess.Popen([os.environ[\"PY\"],\"-I\",\"/tmp/xt_load.py\"], start_new_session=True, close_fds=True)"; sleep 1; PID=$(pgrep -n -f xt_load.py); rm -rf /tmp/xt3.trace; xcrun xctrace record --template "Metal System Trace" --attach $PID --time-limit 6s --output /tmp/xt3.trace 2>&1 | tail -3; xcrun xctrace export --input /tmp/xt3.trace --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"metal-gpu-intervals\"]" | grep -c "($PID)"'
```

(`/tmp/xt_load.py` is the 5 s matmul loop from the earlier probe, lengthened to 20 s.)
Branch: a count > 100 → `--attach` (D3 gains two untraced plain posts per turn); else
`--launch` for D3 and D6 with the `/usr/bin/env` wrapper of section 3.3, after a launched
probe that prints the child's `HOME`, `GATE_ROUNDS` and one `PROF_*` variable to confirm
the environment arrives. Then dump the distinct `process`, `channel-name`, `state` and
`event-depth` values and the `gpu-performance-state-intervals` schema mnemonics and
distinct state values from `/tmp/xt2.trace`: branch: the state differs between the idle
head and the loop → the track is used for H8; else H8 is judged on timings only.

- [ ] **Step 1: `xt_parse.py` core (tables, pid filter, union, marker alignment)**

```python
import json, subprocess, sys, xml.etree.ElementTree as ET   # the XML is xctrace's own export
trace, pid, rounds_path, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def table(schema):
    xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                          f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'],
                         capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(xml)
    by_id = {el.attrib["id"]: el for el in root.iter() if "id" in el.attrib}
    def cell(el):            # resolve ref=; keep text and fmt (process cells carry only fmt)
        el = by_id[el.attrib["ref"]] if "ref" in el.attrib else el
        return {"text": el.text, "fmt": el.attrib.get("fmt")}
    schema_el = root.find(".//schema")
    if schema_el is None: raise SystemExit(f"no schema for {schema}")
    cols = [c.find("mnemonic").text for c in schema_el.findall("col")]
    return cols, [{cols[i]: cell(c) for i, c in enumerate(list(row)) if i < len(cols)} for row in root.iter("row")]

cols, rows = table("metal-gpu-intervals")
for need in ("start", "duration", "process"):
    if need not in cols: raise SystemExit(f"metal-gpu-intervals lacks {need}: {cols}")
mine = sorted((int(r["start"]["text"]) / 1e9, (int(r["start"]["text"]) + int(r["duration"]["text"])) / 1e9,
               float(r["start-latency"]["text"]) / 1e3 if r.get("start-latency", {}).get("text") else None)
              for r in rows if (r["process"]["fmt"] or "").endswith(f"({pid})"))
others = sum(1 for r in rows) - len(mine)

def bursts(ivals, split_s):
    """Union of intervals, split into bursts at gaps > split_s."""
    out, cur = [], None
    for a, b, _ in ivals:
        if cur and a - cur[1] <= split_s: cur[1] = max(cur[1], b)
        else:
            if cur: out.append(tuple(cur))
            cur = [a, b]
    if cur: out.append(tuple(cur))
    return out

def union_busy(lo, hi):
    busy, cur, gaps = 0.0, lo, []
    for a, b, _ in mine:
        if b <= lo or a >= hi: continue
        a, b = max(a, lo), min(b, hi)
        if a > cur: gaps.append((cur, a))
        busy += max(0.0, b - max(a, cur)); cur = max(cur, b)
    if cur < hi: gaps.append((cur, hi))
    return busy, gaps

def fit_offset(marker, marker_ms):
    """marker: {'submit': [3], 'done': [3]} perf_counter seconds; three bursts of marker_ms
    ±30 % with 40–60 ms gaps. Returns trace_t − perf_t."""
    lo, hi = marker_ms * 0.7e-3, marker_ms * 1.3e-3
    cands = [(a, b) for a, b in bursts(mine, 0.005) if lo <= b - a <= hi]
    for i in range(len(cands) - 2):
        (a0, b0), (a1, b1), (a2, b2) = cands[i:i + 3]
        if 0.04 <= a1 - b0 <= 0.06 and 0.04 <= a2 - b1 <= 0.06:
            offs = [b0 - marker["done"][0], b1 - marker["done"][1], b2 - marker["done"][2]]
            if max(offs) - min(offs) < 0.002:
                return sum(offs) / 3
    raise SystemExit(f"{label}: marker pattern not found for pid {pid}")
```

The rest of the script: per post with a `marker` record, fit the offset (D3: from the
begin-decode marker, checking the end marker within 1 ms; D6: from the end marker), map
every round's `t_start`/`t_end` and every stamped eval; per decode window compute `busy`,
`idle_ms_per_round`, the gap histogram, `start-latency` p50/p90; for the traced lazy post
place gaps by the proxy stamps; for sync posts assign each row to the stamped event whose
[call, next call) window contains its start, sum per event (`gpu_ms`), mark an event
`valid` when it was a real sync and received ≥ 1 row, and emit the per-event list;
`pstates` from `gpu-performance-state-intervals` by schema mnemonics (fail if absent).
Print `others` (rows from other processes) beside the pid's count.

- [ ] **Step 2: Validate on a re-recorded probe**

Re-record the loop on the M5 with the shim's own marker function prepended (importing
`prof`'s marker helper, calibrated the same way), under the branch chosen in step 0.
Expected: the marker found with offset spread < 2 ms; busy ≥ 0.9 inside the loop's own
window; the pid filter removing other processes' rows; `pstates` non-empty.

- [ ] **Step 3: Reviewer gate**

A fresh reviewer checks the pid filter, the union, the burst split, the marker fit, the
event assignment rule and that the parser fails loudly on a missing column or marker.

## Task 4: Analysis `analyze_prof.py`

**Files:**
- Create: `~/code/sous-spikes/prof-b5/analyze_prof.py`

**Interfaces:**
- Consumes: the rounds JSONL files of every daemon and `xt_parse.py`'s JSON.
- Produces: the tables of section 5 as Markdown on stdout, and `levers.json`
  `{"own5_ms": x, "levers": {"L1": {"ceiling_ms": x, "share": x, "ci": [lo, hi], "outcome":
  "spike|closed|maintainer", "encode_inclusive": bool}, ...}, "closure": {"plain_own_ms": x,
  "gpu_sum_ms": x, "idle_ms": x, "residue_ms_ci": [lo, hi]}, "H": {"H1": {...}, ...}}`.

- [ ] **Step 1: Loading and exclusion**

```python
def load(path):
    recs = [json.loads(l) for l in open(path)]
    install = next(r for r in recs if r["type"] == "install")
    if install.get("allow_inactive"): raise SystemExit(f"{path}: M2 smoke file, not analysable")
    posts = {r["post"]: r for r in recs if r["type"] == "post"}
    groups = collections.defaultdict(list)
    for r in recs:
        if r["type"] in ("round", "step"): groups[(r["type"], r["turn_key"], r["post"], r["decode"])].append(r)
    kept = []
    for (_, _, post, _), rs in groups.items():
        p = posts.get(post, {})
        if p.get("refused") or p.get("took_class") in ("fork", "disk", "miss") or abs(p.get("canary_dev", 0)) > 0.05: continue
        rs.sort(key=lambda r: r["round"])
        body = [r for r in rs[2:-1] if not r.get("trunc")]
        if len(body) >= 8: kept += body            # short decodes feed the repost loop, never void a daemon
    return install, kept, [r for r in recs if r["type"] in ("overhead", "microbench", "marker")]
```

(`took_class` and `canary_dev` are computed by the driver into the `post` record: the
take class from the turn line, the canary deviation against the median of the same
turn's posts.)

- [ ] **Step 2: Buckets, overhead correction, fine-layer scaling**

Per record key `parent/stage/layer`: `build_ms`, `pre_ms`, `eval_ms` raw. Correction per
section 3.6: `o_b` per bucket from D3's valid events (`host − gpu` per eval, median over
rounds), `corrected = raw − n_evals × o_b`; buckets without a valid pairing keep the raw
value and are flagged. Sum fine layers by type (`layer` ∈ {1, 2, 4} gated-delta, {7}
attention), scale × 48/3 and × 16; `head = proj_out@verify_call`; layer-type totals for
every layer from the `sync` posts; the fine-layer identity check within 5 %, printed.

- [ ] **Step 3: Per-turn medians, the 56K fit, intervals**

Per turn, mode and arm (5k/6k): median over kept rounds of each bucket. OLS on
`(n − 56,000)` over the 18 turns; value at 56K; percentile bootstrap over clusters (4,000
resamples, refit, D3 rounds resampled jointly for `o_b`); t-interval with df = clusters −
2. `own5` from the plain 5k rounds. H7 per bucket: within-turn paired 6k − 5k medians,
then the same fit.

- [ ] **Step 4: Closure, hypotheses, levers**

Closure per D1 turn: `gpu_sum = draft + Σ layer_gpu + pre@verify_call/proj_out + head +
walk_gpu` (corrected sync buckets; `walk.first_wait` must be ≈ no-op in sync, printed) +
`idle_ms_per_round` (D3, pooled) against the plain `own`; the signed residue per turn and
the bootstrap CI of its mean. H1–H10 evaluated exactly as section 2 words them, each
printed with its number and verdict. L1–L8 as section 4 with the three outcomes and the
residue clause. Keep-warm and summary numbers (Tasks 6 and 7) merge by hand into
`results.md`.

- [ ] **Step 5: Unit check on the M2 smoke files; Step 6: Reviewer gate**

The reviewer checks every bucket sum against section 3.2's table, the exclusions against
3.6, the correction against 3.6, and that no threshold in the code differs from section 4.

## Task 5: Driver and M5 run chain

**Files:**
- Create: `~/code/sous-spikes/prof-b5/driver.py` (from `gate157.py`, data paths pointed
  at `P157`; adds `GATE_HOME` (shared, `makedirs(exist_ok=True)`, config rewritten per
  daemon, no fork `rmtree`), `GATE_MODES` (the timed-post mode sequence; the driver posts
  k−1, then k as `plain`, then k once per listed mode, writing `PROF_MODE_FILE` before each
  post), `GATE_SLEEP` (seconds slept before each timed post), `GATE_NOPOLL` (no `MemPoll`),
  `GATE_XT=<seconds>` with `GATE_XT_LAUNCH=1` for the launch branch, the repost loop (reads the rounds file; ≤ 2 extra posts per turn
  and mode while under the floor, per arm in D1), reserve substitution, `ttft_s` and the
  take from `/sous/status` per post, the `post` records' `took_class` and `canary_dev`,
  and a start-up self-check that names any missing file)
- Create: `~/code/sous-spikes/prof-b5/run.sh`

- [ ] **Step 1: Mirror the harness and the tree**

```bash
rsync -a --exclude __pycache__ ~/code/sous-spikes/prof-b5/ m5pro:~/code/personal/sous-spikes/prof-b5/
ssh m5pro 'cd ~/code/personal/sous-remote && git fetch -q && git checkout -q main && git pull -q --ff-only && uv sync -q && git rev-parse --short HEAD'
```

- [ ] **Step 2: `run.sh`**

```bash
#!/bin/zsh
set -u
cd ~/code/personal/sous-spikes/prof-b5
TREE=~/code/personal/sous-remote
GPU=~/code/personal/sous-spikes/splash-port-2026-09-24/gpu_run_m5.py
export GATE_HOME=~/sous-prof-home PROF_MODE_FILE=$PWD/mode PROF_SEED=180
mkdir -p rounds logs traces
TURNS=(${(f)"$(cut -d' ' -f2 turns.txt)"})          # zsh does not word-split $TURNS
XT=(${(f)"$(awk 'NR==1||NR==7||NR==13{print $2}' turns.txt)"})
LONG=(${(f)"$(cut -d' ' -f1 long.txt)"})
job() {  # job <label> <temp> <block> <drafter:on|off> <arms> <max_tokens> <modes> <picks...>
  local label=$1 temp=$2 block=$3 drafter=$4 arms=$5 toks=$6 modes=$7; shift 7
  local extra=(--model 'prompt_cache_gb = 24' --model 'idle_unload_minutes = 1440')
  [[ $drafter == off ]] && extra+=(--model 'speculative_draft_id = ""')
  [[ $block == 6 ]] && extra+=(--model 'speculative_block_size = 6')
  GPU_LABEL=prof-b5-$label GATE_WRAP=$PWD/prof.py PROF_BLOCK=$block PROF_ARMS=$arms \
  GATE_ROUNDS=$PWD/rounds/$label.jsonl GATE_MODES=$modes \
  python3 $GPU perl -e 'alarm 7200; exec @ARGV' \
    $TREE/.venv/bin/python driver.py $label $TREE $temp $toks "${extra[@]}" "$@" > logs/$label.out 2>&1
  echo "$label exit=$?"
}
GATE_TOOLS_ONLY=1 job warmforks 0.7 5 on "" 1 plain "${TURNS[@]}" "${LONG[@]}" --picks-file warm.txt
job on     0.7 6 on 5k,6k 512 plain,lazy,sync,syncfine,syncfine,sync,lazy,plain "${TURNS[@]}"
job off    0.7 0 off ""   512 plain,sync,syncfine,syncfine,sync,plain           "${TURNS[@]}"
GATE_XT=1500 PROF_MARKER=1 job xt 0.7 5 on "" 512 plain,lazy,sync,syncfine,plain "${XT[@]}"
job long   0.7 5 on ""    256 plain,syncfine,syncfine,plain "${LONG[@]}"
job greedy 0   5 on ""    512 plain,sync,plain "${XT[@]}"
GATE_SLEEP=0,1,3,10,30,90,90,30,10,3,1,0 GATE_NOPOLL=1 GATE_XT=1200 PROF_MARKER=1 PROF_CANARY=0 PROF_MICROBENCH=0 \
job warm   0.7 5 on ""    128 plain,plain,plain,plain,plain,plain,plain,plain,plain,plain,plain,plain --picks-file warm.txt
rm -rf ~/sous-prof-home/.sous/forks
```

(`PROF_BLOCK=0` with the drafter off disables the block check. `GATE_TOOLS_ONLY=1` makes
D0 post one turn per distinct tools hash. With the attach branch, the `xt` job's mode list
gains two trailing `plain` posts after the trace's time limit; with the launch branch,
`GATE_XT_LAUNCH=1` is added to the `xt` and `warm` jobs. The driver writes `plain` to the
mode file for warm-ups.)

- [ ] **Step 2b: Reviewer gate and dry run**

A fresh reviewer reads `driver.py` and `run.sh`; then a GPU-free dry run on the M2 with a
fake status server exercises the mode-file sequence, the sleep list, the repost trigger,
the reserve substitution and the `took` class check.

- [ ] **Step 3: Launch detached, inside the GUI screen session when one exists, else `setsid`**

```bash
ssh m5pro '(screen -S gui -X screen zsh -lc "cd ~/code/personal/sous-spikes/prof-b5 && ./run.sh > run.out 2>&1" 2>/dev/null || python3 -c "import subprocess; subprocess.Popen([\"zsh\",\"-lc\",\"cd ~/code/personal/sous-spikes/prof-b5 && ./run.sh > run.out 2>&1\"], start_new_session=True, close_fds=True)")'
```

Poll `run.out` every 10–15 minutes (≈ 3.7 h). The production daemon must read idle on every
poll (`sous status`); a contaminated job is re-run under a `b` label at the end of the chain.

- [ ] **Step 4: Validity ledger**

For every job: `gpu_run_m5` idle line present, `DAEMON-CONTAMINATED` absent, exit 0, all
posts 200, load line paths active, take class and length per timed post, floors met per
turn, mode and arm, reposts and reserve substitutions, canary exclusions. Write
`ledger.md`.

## Task 6: Keep-warm analysis

- [ ] **Step 1:** From `rounds/warm.jsonl` (12 timed posts in order) and the per-post
  `ttft_s` and take: per post `g`, `ttft_s`, round 1 `own`, rounds 2–5 `own`. Pair the two
  posts of each `g`; the g = 0 pair is the reference; print `Penalty(g)` per section 4 and
  the two g = 0 `ttft_s` values.
- [ ] **Step 2:** From the warm trace (if traced and the track is informative), the
  performance-state intervals across the sleeps beside the post times.
- [ ] **Step 3:** Apply L6's rule and write the verdict line into `results.md`.

## Task 7: Progress-summary share (M2, log-only)

**Files:**
- Create: `~/code/sous-spikes/prof-b5/summ.py`
- Data: a copy of the M5's `~/.sous/daemon.log` (hashes, ids and counts only), the run
  directories under `~/sous130-fanout/` (start and end times listed in README before
  counting), and the subagent transcripts (`~/.claude/projects/*/*/subagents/agent-*.jsonl`)
  whose timestamps fall inside those runs, copied to the session scratchpad under
  `untrusted/summ/`.

- [ ] **Step 1: Identification rule (fixed here).** Runs are the `~/sous130-fanout` run
  directories' time spans (the three cap-1 runs and the 4-way run); the #141 branch run is
  reported separately. Within a run, group served `sous-local` turns by `(system, tools)`
  hash. Transcript responses are grouped by `message.id`, taking the last line's timestamp
  and the maximum `usage.output_tokens`. A turn is a *real* turn when a response's last
  timestamp is within ±2 s of the turn's completion and its output tokens match
  `output_tokens` ± 2. A turn is a *progress summary* when it is not real **and**
  `output_tokens` ≤ 300 **and** its arrival (`completion − seconds − queue_s`) is within
  20–45 s of the previous arrival in its group. A turn whose transcript shows a
  `compact_boundary` within ±60 s is a *compaction* turn. Everything else is
  *unclassified*; a run with > 10 % unclassified is reported as a range.
- [ ] **Step 2:** Per run: counts, engine seconds (`prefill_s + decode_s`) and share by
  class; Σ `queue_s` over real turns and its share of the run's wall clock; the same for
  summary turns.
- [ ] **Step 3:** Apply L7's rule and write the verdict line into `results.md`.

## Task 8: Report

- [ ] **Step 1:** `results.md` per section 5, every H1–H10 with its number and verdict, the
  lever table from `levers.json`, the ledger, the amendments.
- [ ] **Step 2:** Copy `rounds/`, `traces/*.json` (parsed, not the `.trace` bundles),
  `results.md`, `ledger.md` and `README.md` to the local harness directory.
- [ ] **Step 3:** Update the memory file (findings, each rule's outcome, the next M5 spike
  or the close) and hand `results.md` to the maintainer. GitHub posting (a new issue
  carrying the profile, or comments on the lever issues) waits for their word.

## Self-review

- **Coverage:** H1 ← D1 `proj_*` buckets with D3's per-eval correction; H2 ← D1 `draft`
  and the fit's slope; H3 ← D1 `gdn_chain` and the microbench; H4 ← D1 `sdpa` per turn;
  H5 ← D1 lazy `verify.build`/`.encode` and sync `draft`, D3 idle; H6 ← Task 4 step 4;
  H7 ← D1 paired arms; H8 ← D6 and Task 6; H9 ← Task 7; H10 ← D1 `walk.pingpong` and
  D3's `host − burst`. Every ceiling names only measured quantities; L1's T = 1 term and
  its T = 5 term come from the same process.
- **Same-process rule:** plain/lazy/sync/syncfine per turn in one daemon; 5k/6k per round;
  the closure's plain `own` from the same turn; the only cross-process terms are D3's
  per-eval overheads (additive, bootstrapped jointly) and D3's idle (per turn, no CI,
  canary-normalised under `--launch`).
- **Placeholders:** the `pins.py` digests are blank by design until Task 1 step 3 fills
  them; `prof.py`'s generate wrapper and `driver.py` are specified by contract and base
  script and are gated by reviewers before `run.sh`.
- **Consistency:** record keys are `parent/stage/layer` with stages `draft`, `verify`,
  `verify_greedy`, `verify_call`, `model`, `layer`, `gdn`, `attn`, `mlp`, `proj_in`,
  `proj_out`, `proj#k`, `gdn_kernel`, `gdn_norm`, `sdpa`, `walk`, `step`; the same names
  appear in `stages.py`, the smoke expectations, `analyze_prof.py` and section 3.2.
- **Threats:** each of the nine threats names the step that pins it.
