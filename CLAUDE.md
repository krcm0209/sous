# CLAUDE.md

sous is a local daemon that serves Claude Code's subagents from a local MLX
model and forwards everything else upstream unchanged; Claude Code executes
every tool under its own permission system. macOS / Apple silicon only. The
goal is plan economics: subagent output is generated locally for free so
heavy Claude Code use stretches further — evaluate features against that
goal.

## Commands

- `uv sync` — full setup (installs Python 3.14 and everything). Never pip.
- `uv run pytest -m "not model"` — the test suite. `model`-marked tests
  download multi-GB weights and are local/manual only; `slow`-marked tests
  spawn real processes and take seconds — run them, don't skip or mock them.
- `uv run ty check` — type check (ty, NOT mypy; covers tests and scripts too).
- `uv run ruff check . && uv run ruff format --check .` — lint/format.
- `uv lock --check` — lockfile sync. CI runs exactly these four jobs.
- `uv run python scripts/api_smoke.py` — one served turn against a tiny real model; run it after any dependency change, since CI cannot load a model.

## Gotchas

- Python >=3.14 required. `except A, B:` without parentheses (PEP 758, e.g.
  src/sous/cli.py) is valid 3.14 syntax, not a Python 2 bug — don't "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- mlx / mlx_lm / mlx_vlm imports are deliberately function-local (absent on
  non-macOS; the lint CI job runs on ubuntu; tests use fake engines). Don't
  hoist them to module level.
- Any thread that touches mlx MUST call
  `engine.base.release_mlx_thread_state()` before it exits — mlx >= 0.32.1
  (ml-explore/mlx#4327) segfaults the whole daemon in the exiting thread's
  TLS teardown otherwise. CI cannot catch this (model tests are local-only);
  after dependency changes, verify with one real served turn. After the
  release that thread cannot run another mlx op that needs a stream — any
  array op, `mx.eval`, a model load — with "There is no Stream(gpu, 0) in
  current thread"; device and allocator calls (`device_info`,
  `get_active_memory`, `get_cache_memory`, `clear_cache`) and the freeing of
  arrays made earlier still work, which is what `server._mlx_memory_gb`
  relies on from threads that are reused. So a thread
  that releases before each unit of work ends — an endpoint pool thread,
  which releases after every turn — may keep querying and freeing, but must
  never run a load: a load runs on a `sous-model-load` thread of its own
  (`EngineManager._load`). Before it had one, a load on a pool thread made
  every later cold start on that thread fail. A thread that controls its own
  exit, like the endpoint's session thread, touches mlx freely and releases
  once on the way out.
- Tests must never touch the real `~/.sous` — always pass tmp_path-based
  config_path/data_dir.
- `docs/superpowers/**` are point-in-time design/plan records: never edit,
  reformat, or "sync" them with current code (they are also ruff-excluded).
  Once a record's work has shipped and nothing cites it as a current rule,
  it leaves the tree: `git rm` it and add a permalink row to
  `docs/design-records.md`.
- Engine `on_delta` callbacks (`engine/base.py:Delta`) fire on the generation
  thread from inside the decode loop: never block or raise in one, and expect
  late deltas from a stalled-and-abandoned session.
- `PrefixCache` refuses its cold retry once any delta has reached the client
  (a retry would replay the turn to it). That is deliberate, not a missing
  retry. A non-streaming turn's `on_delta` is accounting-only and wrapped in
  `ReplaySafe` (`engine/base.py`), so it still retries cold on a warm-cache
  failure — nothing was sent that a re-run would send twice.
- Prompt-cache slots (`engine/promptcache.py`) are owned by the thread that
  built them (#34) and looked up only by that thread. `reset_prompt_cache`
  and `prompt_cache_stats` take an `owner`: the turn runner retires a
  stalled session's thread (`session.thread`) and never calls the bare
  `reset()`, which drops every other owner's slots too. A `fork` slot is a *copy* taken while
  a turn prefills past a boundary (`fork_point`, 4096-token floor). There are
  two: the *tools* boundary — Qwen3.5/3.8 render `# Tools` *before* the
  client's system text, so sessions and projects presenting the same tool
  array share ~45–56K rendered tokens (Phase 3a's "cross-session reuse is
  impossible" read the request body's order, not the render's) — and the
  *header* boundary, the whole system block. Both come from probe pairs
  (`promptcache.probe_boundaries`), never from rendering the system turn
  alone, which Qwen3.8's template refuses. The lowest boundary is also
  written to disk (`engine/forkstore.py`, `engine/forkio.py`): the live
  cache's full padded `keys`/`values` at the boundary, before the resident
  copy, outside the prefill timer, with its own handler; a miss reads the
  longest stored prefix into the turn's own cache on its own thread and the
  boundary loop republishes a resident fork there (`reuse <= b`). Restore a
  `KVCache` by assigning `keys`, `values`, then `offset` from metadata —
  never through `state`, whose setter derives the offset from the padded
  shape and mispositions every appended token — and `mx.eval` the arrays: a
  lazy array is invisible to the memory the pressure valve reads (fork
  copies are lazy too; `eval_cache` after every `fork_copy`). Save only
  evaluated, contiguous buffers: a slice or a transpose is materialised
  whole on save. Files are shared across owner threads; only arrays are
  owner-scoped. Only `default_engine_factory` hands an engine a store
  directory — tests and `sous tune` (`forks=False`) never reach `~/.sous`.
  Warm turns resolve the probe too:
  a turn that starts at another session's tools fork must still publish its
  own header fork. A `turn` slot is *copied* and left in place by the turn
  that extends it when `_make_room` can hold the copy (Claude Code branches
  a background subagent's conversation every 30 s with a progress-summary
  call, and a consumed slot left the real conversation only the header
  fork), and *moved* — removed, its arrays adopted — when it cannot, always
  at `prompt_cache_gb = 0`. The copy is priced at the size the turn will
  publish (the parent's own size passes budgets the publish then cannot
  honour), and a take retires the lengths below its slot
  (`_retire_ancestors`, read off `held`, never recorded), so a conversation
  holds its current and previous lengths and a chain a cold retry breaks
  heals on the next take. A slot is priced at `slot_bytes`, every layer's
  `nbytes`, so the whole padded buffers: `VLMEngine.decode()`, and nothing
  else, grows the full-attention `KVCache`s by `window.DECODE_KV_STEP` =
  4096 rows (mlx-vlm's step is 256; every `prefill()` call keeps it, while
  the rows the decode call prefills itself, the whole prompt with the cache
  off, grow by the decode's step, and `_decode_kv_step` puts it back on
  every way out of the decode, with the instance's `prefix_cache_reserve`,
  which skips mlx-vlm's DFlash reserve while the rows fit: rounded to 4096 it
  would copy nearly every decode's whole cache before the first round), so a
  published turn slot carries up to about 4,100 rows (the step plus the
  block the decode's reserve pads for) past its decode's last token,
  256 MiB on the default model, while a `fork_copy` holds only the offset's
  rows. Inline `role:"system"` messages after the first user message
  render as `<system-reminder>` blocks in the preceding user turn (after its
  tool results), or as a user turn of their own after an assistant turn
  (`api/convert._place_inline_system`), never hoisted into the system
  block: the hoist moved the header boundary and killed every prefix behind
  it on the turn an attachment arrived.
  Never rewind a cache to make a slot — a hybrid model's recurrent layers
  cannot.
  `base.measure_cache_budget`/`live_headroom` deliberately do not call
  `release_mlx_thread_state()`: they run on threads whose caches are live.
  The pressure valve reads Metal's headroom and the kernel's
  `kern.memorystatus_vm_pressure_level`, never psutil's free RAM: right after
  a model load the weight files sit in the page cache as "active", that
  figure read low, and the valve evicted the forks a cold turn had just made.
- Every `prefill()` and `decode()` on the VLM backend hands mlx-vlm explicit
  `position_ids` covering the cache from 0 through the tokens being appended,
  plus a zero `rope_deltas` (`VLMEngine._positions`), for the model families
  whose text-only embedding helper returns positions of its own (probed once
  per engine: the Qwen lineage — `qwen2_vl`, `qwen2_5_vl`, `qwen3_5`,
  `qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` and the families that reuse
  their model classes (verified on `qwen3_5` and `qwen2_vl`; the rest share the
  rotary contract — `qwen3_omni_moe` has classes of its own over the same
  `apply_rotary`); the other mRoPE families return none, position from the
  cache offset themselves, and several would fail on an array of another
  rank). The probe calls the helper the way `generate_step` does (ids, no
  pixels, `mask=None`), once, at load — several helpers outside the lineage
  null the model's rotary state when run; a helper that cannot be probed gets
  no kwargs and one warning. Load-bearing: mlx-vlm's
  helper returns positions local to the ids it is given and `generate_step`
  merges them into the model's kwargs, and the Qwen language models slice that
  array at the cache offset — past its end for any continuation `generate_step`
  chunks — so on 0.7.x the fused MRoPE Metal kernel reads a zero-length buffer
  out of bounds and every continued token lands at position 0 (on 0.6.17 the
  same tokens landed at `0..n-1`; measured 2026-09-14 on the default model:
  keys off by 127–170 % either way, prefill and decode, drafter or not). Only
  from 0.7.0 does `generate_step` re-apply a caller's positions over the
  helper's — hence mlx-vlm is never pinned below 0.7.0; below it the kwargs are
  silently overwritten. A suffix-only array is sliced the same way, so the array is
  always absolute from 0; `rope_deltas` goes with it so the engine owns the
  delta the model adopts into the state a drafter run nulls and positions the
  generated tokens from. mlx-vlm's own `_prime_cached_prefix_rope_state` does
  this only on its `prompt_cache_state`/APC paths, which sous never enters.
  `tests/test_engine_positions.py` pins the kwargs and the probe;
  `tests/test_engine_vlm.py` pins the positions by comparing cached keys, not
  greedy text — a mispositioned block has reproduced the reference's words
  exactly — reading every layer on a pure-attention model but only the first
  attention layer on a hybrid, whose deeper layers drift by tens of percent
  between a one-pass and a split prefill for reasons that have nothing to do
  with positions, and it witnesses generated tokens too — the model positions
  those from its cache offset and the delta it adopted, never from
  `position_ids`. The LM backend needs none of this (mlx-lm's text models take
  positions from the cache offset). The model-load line and the status
  document's `positions` record which side owns them (`engine` when the
  helper returned them, `model` otherwise); `position_ids`/`rope_deltas` are
  pass-through kwargs mlx-vlm's `GenerateKwargs` does not declare, so across
  an mlx-vlm bump the guards are that token, the contract test in
  `tests/test_engine_positions.py` (the real `generate_step` over a stub
  model, in CI: the engine's kwargs must reach the language model over the
  helper's) and a manual `uv run pytest -m model tests/test_engine_vlm.py` on
  the M5 Pro for the kernels themselves — CI cannot load the models. A fork
  on disk written by a build with wrong KV would survive a restart, which is
  why the fork store's identity key hashes the engine sources
  (`forkstore.engine_epoch`: vlm.py, lm.py, int8prefill.py, tileattn.py,
  projkernel.py, slab.py, projtile.py, kernels/*.metal and *.h except the
  two kv8 kernels, `attention_tile_kv8.metal` and `kv8_quant.metal`, and
  the projection tile's `projection_tile.metal` and `projection_tile.h`,
  which `_EPOCH_EXCLUDED_KERNELS` names because they serve only inside a
  generation and never touch stored KV) beside the versions, the GPU and
  the weights snapshot — any
  change to what a token id produces invalidates every file; `rm -rf
  ~/.sous/forks` is the manual eraser.
- Prompt-cache per-turn gauges (`promptcache.TURN_GAUGES`, reset by
  `PromptCacheStats.begin_turn` on every `generate()` and again on a cold
  retry) are assigned per turn and read back directly from the owner-scoped
  `after` snapshot in `TurnRunner.run` — never as `after − before`. Counters
  (`forks`, `evictions`, `pressure_evictions`, `reused_tokens`) are deltas.
  Timers read `promptcache._clock` so tests can drive them; `Slot.last_used`
  keeps the real clock. The gauges exist for the endpoint's turn line only:
  the status document's `prompt_cache` block (`EngineManager.status`) goes
  through `without_turn_gauges`, because there a gauge is a max over every
  owner ever seen beside daemon-long counters — a number no reader can use.
- `EngineManager` never holds its lock across a model load or unload
  (`_loading`/`_unloading` on a `Condition`): `status()`, `hold()` and the
  idle sweep must answer during either, and `/sous/status` reports
  `loading`. A `hold(pid, create_time)` pins the model while that process
  lives — the sweep, `status()` and `hold()` all prune holders with psutil
  (pid gone, a zombie, or start time off by more than a second = reused
  pid), and whichever prunes the last one restarts the idle clock; the
  preload thread `hold()` may start calls `get()` and then
  `release_mlx_thread_state()` like every mlx-touching thread. An endpoint
  turn re-checks `abandoned` after `engines.get()`: that is where it now
  waits out another thread's load (the lease no longer does). The `/sous/`
  routes (`sous/monitor.py`) are mounted in `create_server` *before* the
  endpoint's, behind the loopback guard in
  `sous/loopback.py` that the endpoint shares; a `/sous/` path never reaches
  the upstream, and a failure inside a route is a JSON 500, never
  Starlette's text one. `sous claude` speaks plain HTTP and nothing else:
  `/sous/status`, or
  a 404 from whatever holds `daemon.lock` = "restart the daemon" (from
  anything else = "not sous on this port"), no compatibility fallback. The
  status document is one shape everywhere (`Daemon.status_document`):
  `engine`, `inflight`, `config`, plus `recent_turns` when the caller asks
  for it.
- `sous/inflight.py` is the in-flight turn registry: an endpoint turn
  registers under its `msg_` id in `TurnRunner.run`; `progress()` runs on
  the engine's session thread from inside the decode loop, so every
  registry method is a dict write under one lock and never raises (an
  unknown id is ignored). The runner retires a turn *before* it releases
  the endpoint lock — the lock is not a queue, so a finished turn left
  registered could still be the entry a reader takes as the one on the
  pass — and `snapshot()` lists the turn past `queued` first (the one whose
  phase moved most recently), then the queue in arrival order. The
  prefill's size is not stamped by the runner — it is inside
  `session.generate()` for the whole prefill — but read by a *probe* the
  registry calls from a status reader's thread: the probe answers only
  once the owner's `hits`/`misses` have moved past the turn's baseline and
  the `prefilled_tokens` gauge is non-zero, because `begin_turn` zeroes
  that gauge only inside `generate()`, after the hand-off, the engine-lock
  wait and the render — so a reader landing before the take sees no size
  rather than the previous turn's. The registry re-asks on every snapshot
  while the turn prefills; a later answer supersedes the first (a warm
  attempt retried cold), and a new size restarts the phase clock.
  `stats(owner)` is a locked dict copy no prefill holds the lock across.
  `/sous/events` polls `Daemon.status_version()` — the registry's
  version, `EngineManager.version` (load, unload, hold, release, and
  every idle-clock reset: `touch()` and a `get()` hit) and the config
  file's mtime and size — ten times a second while the last document
  showed a turn, a load or an unload
  (`engine.unloading`), and twice a second otherwise, rebuilding
  the whole document on a worker thread when the tuple moved; the
  once-a-second heartbeat runs only while busy (the TUI reads a silent
  stream during a turn as a stalled daemon), never idle — the idle clock is
  the TUI's to run
  (`Top._engine_now`, seeded by every reset the engine version carries).
  Summaries are per build. Never
  wake the loop from a writer. The routes record a summary for *every*
  `POST /v1/messages id=…` line — refused, abandoned and failed included —
  and the served line is printed from that summary (`_turn_line`,
  `_failure_line`), so a row and a line cannot disagree; the dict is in
  memory only and does not survive a restart. Textual is imported only
  inside the `sous top` command function: `sous serve`, `sous claude` and
  `sous statusline` (no Textual, httpx or psutil; half-second budget) never
  load it. The terminal runs with `ansi_color=True`
  and the `ansi-dark` theme pinned explicitly — foreground and background
  are the terminal's own — and paints hex accents on top of that ground,
  each chosen to clear 3:1 on black and on white (a test asserts it); nothing
  in `tui.py` sets a `background:` other than `ansi_default`. The chef's
  frames, the dial and the steam are picked from the app's clock, never
  tweened, so a pinned clock is a pinned picture.
- Claude Code auto-compacts a `sous-local` subagent when *its own* token
  count nears `CLAUDE_CODE_MAX_CONTEXT_TOKENS − 33K` (sooner with its
  precompute trigger) — two extra local calls, ~6 minutes at 131072 for a
  four-turn agent, and the agent then answers from a summary. Every
  compaction switch is global to the session (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`
  is a *minimum* with the model window; `DISABLE_AUTO_COMPACT`,
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`), so `sous claude` sets none; the
  per-local-model lever is `[model].max_context_tokens` (the default
  model's native length is 262144, at +8 GiB of KV reserve) — the classic
  threshold scales with it, the precompute fraction is remote-configured
  per window size and unmeasured at 262144.
- `sous claude` sets `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS=1` and
  `CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS=1` unless the user set them to
  a value Claude Code reads: digits from 1 up (the workflow cap to 256);
  empty, 0 or text is unset to Claude Code too, and it would run its default
  of 20 behind a launch line naming the value (`cli._CONCURRENCY_CAPS`,
  `cli._claude_code_reads_cap`). Overlapped subagents evict each other's
  prompt-cache slots: four at once re-prefilled 56% of their engine time
  (#139). The workflow cap queues agents. The subagent cap *refuses* the extra
  Agent calls ("Do not retry"), at every depth, so a running subagent's own
  launches are refused too. A Sonnet 5 main loop relaunched each refused
  agent after the running one finished (#140); other main-loop models are
  unmeasured. Claude Code can lift the subagent cap itself in some modes, so
  the daemon must still cope with overlap.
- The endpoint forwards every request it does not serve (`api/upstream.py`)
  as a transparent proxy: never re-serialize a forwarded body, never add or
  alter an end-to-end header (only `Host`, the hop-by-hop set and a buffered
  body's `Content-Length` change; responses lose the hop-by-hop set and gain
  `Via`, with uvicorn's own `Date`/`Server` replacing the upstream's), never turn
  `trust_env` on. The routing predicate is the decoded body's `model` after
  stripping one trailing `[…]` suffix; a body that does not decode is the
  upstream's, not a 400. Any third-party library that enters the endpoint's
  request path gets its logger pinned in `mount_endpoint`: sse-starlette logs
  each SSE frame, httpx the full upstream URL with its query string, httpcore
  response header values — and `configure_daemon_logging` sets the root level
  to INFO, so none of that is hypothetical. huggingface-hub 2.x imports the
  same two libraries as `httpx2`/`httpcore2`, and every model load fetches
  `config.json` through the hub, so those names carry the same pin.
- `engine/int8prefill.py` compiles the Metal kernel pair in `engine/kernels/` at
  model load (Apache-2.0, derived from oMLX — see THIRD_PARTY_NOTICES.md). The Python `reorder_k` and
  the GEMM's nibble decode are two halves of one K-order contract (slot
  `16c+4t+j` holds code `16c+8*(t>>1)+2j+(t&1)`): never change one alone; the
  power-of-two bit-exactness test is what catches a mismatch. Only rows >= 128
  on `shape[-2]` route (decode/verify never do) — except on a slab-only MLP
  linear (`engine/projtile.py` bullet), which `_rows_ok(x, slab)` admits
  from `SLAB_MIN_ROWS` = 9 rows counted as the product of the leading
  dimensions, since stock cannot read its flat planes and the projection
  kernel takes it only up to 8 — only tagged modules route (tags are set with
  `object.__setattr__` so they stay out of mlx's parameter tree) and only on
  `model_type` `qwen3_5` (the MoE variant reuses the same classes and is refused), and only the
  tensor-op kernels — the GEMM, the attention tile's and kv8's split kernels
  and the projection tile — get
  `nax.h`, whose Metal-4 `MetalPerformancePrimitives` include needs macOS
  26.2+: the Stage-A kernel and both tiles' reduce kernels stay MPP-free, and
  Stage A must keep compiling on any Metal GPU so CI (macos-15, no tensor
  units) can test it.
- `int8prefill.enable()` never raises and a model load never fails because of
  it, and being on by default its refusals come in two kinds, like the
  attention tile's (`_refuse(expected=...)`): one that only says int8 does not
  apply to this load — the platform rule, a model type other than dense
  `qwen3_5`, no routable affine Q4 gs64 projection — is one INFO line plus
  `unavailable`; one that says it should have run and did not — the GEMM probe
  or the warm-up failing — is one `warnings.warn` plus `unavailable`. Tests
  that need the GEMM without tensor units monkeypatch `int8prefill.qmm`.
- mlx (>= 0.32.1) compiles runtime kernels as the newest MSL the OS has — 4.1 on
  macOS 27, where `decltype` of a local cooperative tensor carries `thread` and
  must be wrapped in `metal::remove_addrspace_t` (#123) — so any macOS major can
  reject a kernel that compiled before. Whether a tensor-op kernel may run
  is therefore decided in two halves. The platform half (macOS >= 26.2, GPU
  generation >= 17, >= 18 for 'p' parts) is `nax.platform_reason()`, shared
  by every tensor-op kernel; past it, the compile half:
  `int8prefill.availability()` compiles and runs the GEMM once where the
  tensor units exist (`_gemm_probe`), `tileattn.availability()` every tile
  variant (`_compile_probe`), and kv8's and the projection tile's
  `availability()` theirs — GPU work only, success remembered, the full
  compiler report logged: a rejected kernel reads as `unavailable` and that
  kernel's `nax` tests skip, where a compile failure with a CPU-stream op in
  flight deadlocks in mlx's exception path. Being mlx ops, they all inherit
  the thread rules above.
- `engine/verifyattn.py` replaces mlx-vlm's per-row SDPA loop in the exact
  verifier (`Qwen3_5BatchInvariantForward._attention`, T = 3..8 while the
  attention tile is inactive) with one stock
  `mx.fast.scaled_dot_product_attention` per group of rows that share mlx's
  kernel plan. Grouping is bit-exact only because inside one plan mlx
  assigns key i to simdgroup `i % 32` or block `i % blocks` whatever the key
  count and the causal mask skips excluded keys, and `plan()` mirrors the
  dispatch of the mlx releases in `VALIDATED_MLX` — on every mlx bump re-read
  `scaled_dot_product_attention.cpp`, and the `is_nax_available()`
  (`device.cpp`) and `BaseNAXFrag::get_coord()` (`steel/gemm/nax.h`) that
  `nax.py` and `kernels/nax.h` mirror, before extending it; every
  mlx-vlm function the hook reads is pinned by source hash
  (`VALIDATED_MLX_VLM_SOURCES`), checked at load because a tool environment
  can still be handed another mlx-vlm than the lock. `pyproject.toml` pins
  mlx and mlx-vlm with `==` to the releases these sets, tileattn's and
  projkernel's were validated on (#168). Another mlx-vlm loads the exact
  paths only while every hooked source still hashes the same, so move a pin
  only once the hash tests pass on the new release (extending the sets where
  a source changed) and its M5 Pro checks have run;
  `tests/test_dependency_pins.py` holds the pins to `VALIDATED_MLX`, its own
  `VALIDATED_MLX_VLM` and the installed versions. While the tile is
  inactive T = 2 is left alone: stock already makes one call there, and
  grouping it would change output at plan straddles. While it is active the
  wrapper's scope widens to every T from 1 and the rows go to `tileattn`
  (next bullet). Scope is decided before the projections —
  `_prepare_projected_qkv` appends to the KV cache, so the wrapper never
  re-enters the original method after them. `enable()` runs only when a
  drafter loaded, probes every plan transition on this GPU before tagging,
  and like int8 never raises. CI proves the 's' and 'd' tables through
  `MLX_METAL_GPU_ARCH` subprocesses. The module stays out of
  `forkstore._EPOCH_FILES` on purpose: prefill never enters the verifier, so
  no stored KV depends on it. Its grouping is bit-identical to the per-row
  loop and to M=1 decode, and while the tile is active the rows from `N0` up
  are tileattn's (keyed by `tileattn.py`) and equal the tile's decode
  instead; tileattn still sends every sub-`N0` run through
  `grouped_attention` (T = 1 and 2 included) and relies on it equalling
  stock one-row SDPA.
- `engine/tileattn.py` serves decode's one-row attention and the exact
  verifier's attention from one GQA-packed MPP tile on the M5's tensor
  units (`kernels/attention_tile.metal`: a split kernel and a fixed-order
  reduce, specialised to bf16, 24/4/256 heads and scale 1/16). A verify row
  equals its one-row decode call — what greedy parity with the drafter on
  or off rests on while the projection tile is not active, and what block
  invariance on the drafter's path rests on while it is active
  (`engine/projtile.py` bullet) — only because a row's output
  depends on its chunk and nothing else about the call, and the chunk is
  fixed per 32·S-key step: `chunk_for(n, S)` is 0 (stock) below `N0` =
  1536 keys and `32 * ceil(n / (32 * S))` from there. `verify_plan` cuts a
  T-row verify into runs at every step boundary, at `N0` and every
  `MAX_RUN` = 8 rows, so any T stays exact; runs below `N0` go to
  `verifyattn.grouped_attention` with tileattn's own `_arch()`, never
  `verifyattn._ARCH`. Decode replaces one module global,
  `mlx_vlm.models.qwen3_5.language.scaled_dot_product_attention`, on the
  first `enable()` that reaches `active` and never earlier (the probe wraps
  it only while it runs and puts back what it found).
  The global cannot tell which module calls it, so its scope rests on the
  flag plus the post-projection checks (one row, mask and sinks None,
  `in_scope`, n ≥ `N0`, `type(cache) is KVCache` without bits or left
  padding): the families that reuse the qwen3_5 language model are refused
  by model type, DFlash drafters call `mx.fast` directly and MTP drafters
  are refused. Deciding after the projections is safe because the fallback
  is the very call the method made, so nothing is appended twice. Verify
  rows arrive through verifyattn's wrapper, whose scope widens to every T
  from 1 while the flag is set. Inside a decode's serving window the rows
  from `N8` up to the ceiling `L` go to kv8 instead of the tile
  (`engine/kv8.py` bullet). The flag `_active_splits` is a plain int, the
  active S or 0, never a model reference: the first statement of every
  `enable()` clears it, only an `active` result sets it, and
  `VLMEngine.unload()` clears it through `tileattn.clear()`. `enable()`
  checks, in order: the config; the shared platform rule
  (`nax.platform_reason()`); the model (qwen3_5, every full-attention
  module at 24/4/256, scale 1/16, bf16); the drafter (none, or `dflash`
  with verify attention active — else the verifier would run stock beside
  a tiled decode and a verify row would stop equalling its one-row call);
  verifyattn's whole `_gate()`; the source hashes of the three decode
  functions (`VALIDATED_DECODE_SOURCES`); the split target, the GPU core
  count read once per process by `gpucores.read()` (IOKit, then
  `/usr/sbin/ioreg`), which must be in `MEASURED_SPLITS` — a chip joins
  only after the tile's T = 1, 3 and 8 timings are measured on it, since
  the guards prove exactness, not speed; and a probe on the
  `sous-model-load` thread. The probe compiles every T = 1..8, proves
  verify rows `array_equal` to decode at `N0`, the next two step
  boundaries and the first one at or past 57K keys, plants NaN past n to
  prove every read stops at n, compares with stock one-row SDPA within
  1e-2 relative RMS (parity cannot catch a kernel that is wrong the same
  way on both paths) and runs one of the model's own attention modules
  through a restored-style `KVCache`. Like int8 it never raises and, being
  on by default, its refusals come in two kinds (`_refuse(expected=...)`). One
  that only says the tile does not apply to this load — the platform rule,
  the split target (unreadable, another GPU, unmeasured), the model's type,
  shape, scale or dtype, a drafter kind other than `dflash` — is one INFO
  line plus `unavailable` with its reason: that is how most Macs load, and
  a WARNING there reads as a fault nobody can act on. One that says it
  should have run and did not — a compile or probe failure, any exception
  (logged with its traceback), mlx, `MLX_SDPA_BLOCKS` or source-hash
  drift, exact verify not active beside a DFlash drafter — is one
  `warnings.warn` plus `unavailable`. `sous tune` prints its per-arm
  notice only for `unavailable`, never for `off`. The engine constructors
  default it off (`VLMEngine(attention_tile=False)`, like `int8_prefill`);
  only `default_engine_factory` hands them the config's `true`, so tests
  and scripts keep the stock paths unless they opt in. Nothing is logged per
  turn; `tileattn.calls` counts every path for the tests and for harnesses
  that wrap the daemon. The fork key records `tile` (`active:S`, else
  `off`) and always `os` (the macOS build), because every stored KV depends
  on Metal compiled at load — mlx-vlm's gated-delta kernel and `mx.compile`
  fusions, mlx's JIT kernels, and the tile's and int8's kernels — so a
  macOS update starts every fork store cold once; `tileattn.py` is in
  `_EPOCH_FILES`. Every kernel edit and every `VALIDATED_MLX` extension
  re-runs the tile's exactness tests and its T = 1, 3 and 8 timings:
  codegen is fragile, and `sg >> 1` for `sg / 2` kept the output but ran
  1.4x slower. A newer mlx-vlm makes the tile `unavailable` until the
  three decode sources (`VALIDATED_DECODE_SOURCES`) are re-read and
  re-pinned, as verifyattn's are. Tests install the decode hook through
  monkeypatch, monkeypatch S (never IOKit) and swap the plain-mlx stand-in
  in for `tileattn.tile`; the `nax` tests skip on
  `tileattn.availability()`, never int8's.
- `engine/projkernel.py` serves the target model's small-row quantized
  projections from a simdgroup-matrix MMA kernel
  (`kernels/projection_mma.metal` and `.h`: plain, staged and group-sums
  bodies split at `// STAGED` and `// GROUP_SUMS`; no MPP and no `nax.h`,
  so it compiles on any Metal GPU and CI runs its arithmetic tests), and
  one-row calls from a one-row kernel with the same bits (below): every
  exact-verifier projection and the verify head, and every target call of
  1–8 rows outside the verifier — one-row decode, a prefill's last chunk of
  ≤ 8 rows and the one-row forward that ends every prefill segment, and the
  sampled DFlash drafter's call of the target's shared `lm_head`. Inside
  a decode window the projection tile takes a verify run or a call of 2–8
  rows instead where it can (`engine/projtile.py` bullet). With the tile
  not active, greedy parity with the drafter on or off holds only because the
  kernel's rows are bitwise independent of the row count T and of which
  linears share a dispatch (with it active the verify projections it holds
  planes for run on the tile, whose rows are T-invariant but not the
  one-row kernel's bits,
  so the contract there is block invariance, and this independence is what
  keeps every path outside the tile's window at the same bits): weights
  dequantize exactly (`as_type<half2>(nibble |
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
  and the bits. A call of one row (decode, the one-row forward that ends a
  prefill segment, a one-row tail) never reaches the MMA kernel: `linears()`
  sends it, one dispatch per linear, through the `one_row` seam to the
  one-row kernel (`kernels/projection_row.metal`, sous's own, compiled
  under `math_mode: safe` like the rest), which writes the same order out
  by hand — quarters as the group ranges `[q·G/4, (q+1)·G/4)`, each group's
  64 products as one fma chain from +0 in the MMA's order (h, s, then MMA
  k = 2c + e: the hardware's sequential C + p0 + … + p7, probed on the M2
  and the M5 Pro), the group sum's tree, the two-fma epilogue,
  `((a0 + a1) + a2) + a3` — so its output equals the MMA kernel's at T = 1
  bit for bit. Every step is an explicit `fma` or add: safe math still
  contracts a plain `acc + d * s` into an fma. RC = 1 and NSG = 4 (one
  column per simdgroup) are frozen for speed only — the tile changes no
  bit — and its threadgroup memory (NSG·G·(RC + 1) fp32) caps K at
  `_ROW_MAX_K` = 65536, past which a one-row call stays on the MMA kernel.
  Its bits are checked against the MMA kernel on random rows and in
  `CANCELLING`'s five layouts, where large terms cancel exactly. Random
  rows at the probe's widths show none of the reorders tried: a reordered
  quarter sum or epilogue moves only a few outputs in tens of thousands of
  a real-width linear, and a lane sum's tree none. So each step has a
  layout that shows it — `quarter` and `middle` the quarters' order and
  pairing (`(a3 + a2) + (a1 + a0)` passes `quarter` alone), `group` a
  group's products and its sum's tree (the one that would move were
  another GPU's MMA to sum in another order), `lane` the tree inside a
  lane's sum, `pair` the epilogue's order and its fmas — and the negative
  control in `tests/test_projkernel.py` holds each to a reordered copy of
  the kernel. A kernel edit that changes a sum's shape needs a layout that
  shows it. Two hooks, each a `functools.wraps` wrapper carrying
  `_HOOK_MARK`, installed once per process on the first activation and
  never removed (int8's three wrappers carry `wraps` too, so every source
  pin follows `__wrapped__` to mlx's and mlx-vlm's own code whatever is
  wrapped on top): hook 1 wraps `Qwen3_5BatchInvariantForward._linear` and
  `_linears` on the class and routes when the flag is set, every linear in
  the call is tagged and x is bf16 with ndim 3, cutting a call of more than
  8 rows (block 0, the drafter's own policy, verifies up to the checkpoint's
  runtime block: 16 for a v1 DFlash, 5 for the default DFlash2) along T
  into runs of ≤ 8, exact because the kernel is T-invariant; hook 2 wraps
  `nn.QuantizedLinear.__call__` and routes when the flag is set, the module
  is tagged, x is bf16 and its rows (the product of the leading
  dimensions) are 1–8. Anything else falls through to what was wrapped,
  and both reach the kernel through `projkernel.mma`, looked up per call —
  except on a slab-only module (`engine/projtile.py` bullet), which has no
  reader but its slab paths: both hooks serve it whatever the flag or its
  tag says, and a call on one that neither they nor int8 take (not bf16,
  1-D, zero rows, a verifier batch past 8 rows) raises a `RuntimeError`
  naming the module (`slab.PATH_ATTR`, set by `convert_mlp`), its shape and
  why, rather than reach mlx's kernels on the flat planes; past 8 rows hook 2
  hands a call on only when int8 will take it, so the wrapper order still
  does not matter.
  The Gemma4, NemotronH and Qwen4Exp verifiers subclass
  `Qwen3_5BatchInvariantForward` and inherit hook 1, so only the tag keeps
  them stock. Tags (`_TAG`, set with `object.__setattr__` so they stay out
  of mlx's parameter tree) go on every `QuantizedLinear` of the loaded
  target's language model, `lm_head` included (497 on the default 27B).
  The drafter's own linears carry a tag of their own (`_DRAFT_TAG`) under a
  flag of their own (`_draft_active`, set by `enable_drafter()`, which
  `VLMEngine.__init__` calls with the resolved block after `enable()`, the
  block resolution and `_pin_block_size` and before kv8; `enable()` clears
  both flags, as `clear()` does), which hook 2 serves
  only from `_DRAFT_MIN_ROWS` = 6
  rows to 8: stock `quantized_matmul` serves the drafter's shapes at
  135–197 GB/s up to five rows and falls to about 100 at eight, where the
  kernel reads at 126–177 (M5 Pro, 2026-10-16), so at block 8 the draft's
  eight-row calls go to the kernel, and so do the context calls (`fc` at
  K = 25600, every layer's `k_proj` and `v_proj` on `x_ctx`) of a round
  that committed six to eight tokens, since mlx-vlm's `_dflash_rounds` cuts
  the context to `accepted + 1` rows, and an unseeded first draft's
  seven-row generation prompt (a floor moved off six must bench both
  shapes); at blocks 1–5 no steady-state call reaches six rows and the
  route is `off` (below). They shape proposals only, so no bitwise or
  T-invariance contract is owed to
  them, their probe is relrms against stock at 6, 7 and 8 rows on every
  (K, N) the drafter holds (the warm-up compiles the same), and the drafter
  stays out of the fork key. Whether the drafter's calls reach hook 2 is
  guarded on two sides, since the warm-up and the relrms step call `mma`
  directly. The probe ends by tagging the drafter and calling one of its
  linears at 8 rows as a module with the flag raised: `draft_kernel` must
  move, or the route is `unavailable` and untagged. That proves only that a
  drafter linear's own `__call__` is the hooked one (a `QuantizedLinear`
  subclass with a `__call__` of its own fails it); it never runs the
  drafter's forward. The forward is pinned instead: `VALIDATED_DRAFT_SOURCES`
  hashes, as `VALIDATED_PROJ_SOURCES` does, every mlx-vlm function from
  `draft_block` and `draft_block_greedy` to the linears (`_hidden`;
  `_project_context_kv`, which already fuses every layer's k/v into one
  matmul when they are `nn.Linear`; the decoder layer, the attention and its
  `_project_kv`, the MLP; DFlash2's convolutions' `prepare` and its
  selector's `select`), for the drafter classes in `VALIDATED_DRAFTERS`
  (v1 DFlash and DFlash2, by `module:qualname`). `_draft_pins_reason`, just
  before the warm-up, gives two kinds of refusal. A drafter of any other
  class (Laguna, the DSpark variants, a subclass of either) is one the route
  does not apply to: `unavailable`, one INFO line, `drafter <class> not
  validated`. In a validated class it reads each pinned function through the
  loaded drafter's own classes, so a forward that fused the quantized k/v or
  called `quantized_matmul` directly, or a component overriding a pinned
  function, reads as mlx-vlm drift: `unavailable`, one warning, `mlx-vlm
  <qualname> changed`. `enable_drafter()` reports through the
  kernel's `_refuse` (subject `projection kernel drafter`) in its
  vocabulary: `off` with the kernel switched off or a verify block of 1–5
  (whose draft calls stay below the floor in steady state), checked in that
  order before anything else. Block 0, the drafter's own policy, proceeds
  because a drafter's own policy may exceed five rows (a v1 DFlash
  checkpoint's 16); under the default DFlash2's runtime block of 5 an
  `active` route there serves nothing in steady state, and `draft_kernel`
  stays 0. `unavailable` with one INFO line where the route does
  not apply (the kernel not active, no quantized linear, a linear `eligible()`
  rejects, named with the K and N its packed weight holds and the
  requirement it misses, a drafter class not validated) and with one
  `warnings.warn` where it should have run (a validated drafter's forward
  off its pins, a warm-up or probe that failed or
  raised, each under its own name, any exception — an exception's
  traceback logged beside it, as the kernel's). The greedy drafter's head
  argmax (`quantized_argmax`) stays stock, since it too shapes proposals
  only. int8 prefill wraps the same `__call__` for its own tags at ≥ 128
  rows on `shape[-2]`, and on a slab-only MLP linear from 9 rows counted
  as hook 2 counts them, the product of the leading dimensions: the row
  ranges are disjoint (on a slab-only module complementary, since stock
  cannot read its planes) and each wrapper falls
  through to what it wrapped, so install order does not matter — keep the
  two gates disjoint if either moves. The flag `_active` is a plain int, 1
  or 0, never a model reference: the first statement of every `enable()`
  clears it, only an `active` result sets it, the probe raises it only
  while it runs, and `VLMEngine.unload()` clears it through
  `projkernel.clear()` beside `tileattn.clear()`. It switches the kernel off
  for modules in mlx's layout only: a converted load's MLP is served from
  its slab paths regardless, so at 0 that load runs its MLP there beside
  stock attention and GDN projections, and a harness that wants a round off
  the tile on a converted load clears the tile's round flag
  (`projtile.set_round_flag`), not `_active`. A failed probe untags the
  model it was given; tags left on an unloaded model are inert while the
  flag is 0, and holding a reference to untag it would keep its weights
  alive. `enable()` checks, in order: the config (`off`); the attention
  tile (`active`, else the tile's reason or `attention tile off` — the
  kernel rides on the tile, inheriting its M5 Pro gate and, with a drafter,
  exact verify attention); the model (`qwen3_5`, every language-model
  `QuantizedLinear` affine 4-bit, group 64, bf16 scales and biases, no
  `bias`, and at least 8 scale values — mlx binds a kernel input of fewer
  elements `constant`, which the bodies' device pointers cannot compile
  against); the drafter's kind (none or `dflash`); a target that does not
  define `speculative_verify_dflash_hidden`, which would send greedy
  verify's tokens through `quantized_argmax`, past both hooks, and which a
  qwen3_5 target gains only through mlx-vlm drift; the source pins
  (`VALIDATED_PROJ_SOURCES`: the verifier's `_linear`, `_linears`,
  `_feed_forward`, `_gated_delta`, `_layer`, `_model` and `__call__`,
  `Qwen3_5DecoderLayer`, `Qwen3_5MLP` and `Qwen3_5GatedDeltaNet`,
  `generate_step`, `nn.QuantizedLinear.__call__`, `_dflash_verify`,
  `_dflash_verify_greedy` and `_dflash_rounds`) and
  `verifyattn.VALIDATED_MLX`; a warm-up that
  compiles every pipeline the model's dispatch keys need at T = 1..8 and
  every linear's (K, N) on the one-row kernel; then
  the tags, the hooks and the probe, on the `sous-model-load` thread. The
  probe drives layer 3's verifier `_feed_forward` at T = 5 and the same
  layer's plain modules at T = 1 through both hooks (`calls` must show
  both) and requires the rows `array_equal`; then, on real linears (layer
  3's q+k+v, o and gate+up+down, layer 0's qkv+z+b+a and out_proj,
  `lm_head`), every row at T = 1..8 equal to its one-row single, every
  fused group equal to its singles, relative RMS within 1e-2 of stock
  `quantized_matmul` (parity cannot catch a kernel that is wrong the same
  way on both paths) and a NaN in one row leaving the others alone; last,
  on every dispatch key the model reaches and on a synthetic cancelling
  linear per K, a random row and a row in each `CANCELLING` layout through
  `one_row` `array_equal` to its row of one multi-row `mma` call (at one row
  `mma` is the one-row kernel itself), a mismatch making the whole kernel
  `unavailable` with a warning, since parity rests on it (a one-row kernel
  off on random rows already fails the earlier parity and T-invariance
  steps, which compare several-row calls with `mma` at one row). It
  restores `calls`, leaves the flag 0 and calls `mx.clear_cache()` in
  `finally`. A refusal at the tile, model or drafter-kind step only says
  the kernel does not apply to this load: one INFO line, `projection kernel
  unavailable: <reason>`. A target defining
  `speculative_verify_dflash_hidden`, pin drift, a compile or probe failure
  and any exception (logged with its traceback) say it should have run: one
  `warnings.warn`, `sous: projection kernel unavailable (…)`. Both are
  `unavailable`; `off` is never a refusal. With the projection tile not
  active, parity still rests on one thing
  the probe cannot cover: drafter-off greedy takes its argmax over bf16
  `logits − logsumexp(logits)` (`generate/ar.py`) and greedy verify over
  raw logits (`sampler(logits)` in `dflash.py`), so the two can differ only
  where the subtraction rounds the top two logits into a tie, in a flat row
  whose top logit sits below half the logsumexp. Stock and the tile rest on
  the same; parity runs on real turns are its only witness, and a
  divergence is read at the first differing token's top-2 logit gap before
  the kernel is blamed. Because the kernel makes a deeper block pay, an
  unset `[model].speculative_block_size` (`speculative_block_explicit`
  false) resolves in `VLMEngine.__init__`, when a drafter loaded and before
  `_pin_block_size`, to `SPECULATIVE_BLOCK_KERNEL` = 8 where the kernel is
  `active`, else `SPECULATIVE_BLOCK_TILE` = 4 where the attention tile is
  (the M5 Pro with the kernel switched off, where #118 measured 4, or with
  a kernel that refused the load), else
  `SPECULATIVE_BLOCK_DEFAULT` = 3: mlx-vlm's exact verifier collapses from
  T=4 on the M2 (#156: 3.56 tok/s at block 4 against 13.46 at 3, identical
  replies), and no other GPU is measured. An explicit value always wins and
  0 keeps its meaning. `sous tune` cannot load a model to ask, so
  `arms.resolve_block` puts `projkernel.static_reason` and then
  `tileattn.static_reason` (the tile's half of the kernel's check, which
  `projkernel.static_reason` calls first) to the checkpoints it described,
  and `_arm` marks every drafter arm's block explicit, or every arm would
  measure the resolved one.
  The engine constructors default it off
  (`VLMEngine(projection_kernel=False, draft_block_explicit=True)`); only
  `default_engine_factory` hands them the config. Nothing is logged per
  turn; `projkernel.calls` (`verify_kernel`, which counts the calls whose
  runs the projection tile served too, `verify_split`, `verify_tile`,
  `plain_kernel`, `plain_tile`, `draft_kernel`, `declined`, and `one_row`,
  one per one-row dispatch `linears()` serves; warm-up and probe leave them
  all as found) counts
  every path for the tests and for
  harnesses, and the load line carries `projection_kernel=`, then on the
  VLM backend the drafter route's `projection_kernel_drafter=active|unavailable|off` (with
  `projection_kernel_drafter_probe_s=` when active; the status document
  carries it as `projection_kernel.drafter`, and `sous tune`'s `_check_proj`
  gives an `unavailable` one a notice of its own),
  and, with a drafter, `draft_block=`. The fork key records `proj` (`active`, else
  `off`) and `projkernel.py` is in `_EPOCH_FILES`: the one-row forward that
  ends every prefill segment, and any tail of ≤ 8 rows, runs on the
  kernel, so stored KV and GDN state depend on it, while the kernel routes
  a prefill the same with a drafter loaded or without, so the drafter
  stays out of the key — its presence reaches stored KV only through the
  projection tile's conversion, which `ptile` keys. Every kernel edit, and
  every change to the pins or `VALIDATED_MLX`,
  re-runs the exactness tests (`tests/test_projkernel.py`, and `uv run
  pytest -m model tests/test_projkernel_model.py` on the M5 Pro) and the
  timing check — the 12 dispatch keys at T = 1..8 against the kernel before
  the edit, interleaved in one process with an A/A pair for the noise
  floor, the per-forward sum within ±2%: the bits are the parity contract
  (with the projection tile active, what every path outside its window
  keeps) and the speed is the only reason the kernel exists. Tests install the
  hooks through `pfx.guard_proj_state` and `pfx.hooked`, swap the plain-mlx
  stand-ins in for `projkernel.mma` and `projkernel.one_row`
  (`pfx.stand_ins`; `pfx.real_kernel` puts both real kernels back), and
  skip the kernel's own tests on `projkernel.kernel_available()`, which
  compiles and runs every variant and the one-row kernel once (the one-row
  kernel bitwise against the MMA kernel at T = 1, random and cancelling)
  and remembers success only: the lane map was probed on Apple GPUs,
  so a GPU where it differs skips rather than fails.
- `engine/projtile.py` serves the DFlash round's verify projections from an
  MPP tile on the M5 Pro's tensor units over slab-ordered weights
  (`kernels/projection_tile.metal`: the tile body and, after its `// REDUCE`
  line, the split's reduce; `projection_tile.h`, prepended to the tile body
  alone; Splash's `q4_mpp_tiles.h` at 0d5a035): per quant group one
  `matmul2d` of 8 bf16 rows against the group's 4-bit codes into a fresh
  fp32 cooperative tensor, then `acc = fma(P, scale, acc); acc = fma(rowsum,
  bias, acc)` in ascending group order from +0 and one bf16 rounding; a
  split plan writes fp32 partials that `sous_proj_tile_reduce` adds in split
  order. Every launch computes 8 rows (`pad8` appends zero rows), so a row's
  bits depend on neither T nor which rows share the call, but they are not
  the one-row kernel's: the tensor op adds adjacent product pairs first, an
  order no sequential fp32 sum reproduces. So while the tile is active the
  contract is block invariance on the drafter's path — a position's logits
  depend on neither the block size nor which rows share a call (the tile's
  rows are independent, the attention tile and kv8 give a verify row its
  one-row bits, the GDN verifier is batch-invariant), so for a given load
  and scope greedy output with the drafter is the same at any block — and
  drafter-on output differs from drafter-off by bf16 rounding inside the
  projections and, on prefill chunks of 9–127 rows, by int8 prefill's
  activations on the MLP (a drafterless load is refused with `no drafter`
  and never converts, so stock `quantized_matmul` serves those chunks
  there), so the two agree except at near-ties. Where it is not active
  (`projection_tile = false`, or `unavailable`: no DFlash drafter, int8
  prefill off, another Mac) drafter on equals drafter off bit for bit, as
  before the tile. It serves only while `serves(rows, hook=)` holds — `_active`, a
  decode window open, the round flag set — through projkernel's two hooks:
  hook 1 (`_serve_verify`) run by run after its cut into runs of ≤ 8, at any
  rows, a one-row remainder included, so a verify of more than 8 rows never
  mixes paths; hook 2 (`_serve_plain`) at 2–8 rows, the short chunk the
  decode call prefills itself. One row stays on the one-row kernel;
  `lm_head` (never copied) and the drafter's own linears never reach the
  tile. A call goes to it only when `plan_for_call(linears)` gives every
  linear a launch at a tuned plan — a lone linear keyed by its K and width,
  gate+up as one dual launch, a stacked call one launch per run of a stack's
  members whole and in order, so the GDN four are qkv+z then b+a — else the
  whole call stays on the MMA kernel. `plan_for_call` returns the `Launch`es
  (key, members, plan) that `verify()` and `plain()` launch from
  (`_check_call` refuses another call's by identity), so a call's launches
  are worked out once, not per run; `slab.slab_info` and `stack_of` read
  their markers with `vars(m).get`, since a `getattr` miss goes through
  `nn.Module.__getattr__` and raises inside it, six times the cost, on a
  path every projection of every round takes. `VLMEngine.decode()` opens the
  window (`open_window()`, which also sets the round flag to 1) after kv8's,
  inside the draft-context `seed` block just before `stream_generate`, when
  `projtile._active`, and closes it first in that loop's `finally`, in a
  `try` of its own, so kv8's close runs whatever happens (`close_window()`
  never raises). The KV step is a context manager (`_decode_kv_step`)
  entered on the same `with` as the draft context, before either window
  opens and left after both have closed, so nothing sits between the opens
  and the `try`, and every exit path, an open or a close that raised
  included, hands the layers back carrying what they came with (neither
  close reads the step). `prefill()` finding a window open closes it and warns,
  naming the window's serial, before the forward: KV with the tile's bits
  would reach a fork under a key that cannot tell, and a refusal would fail
  every prompt-cache turn (each prefills before the decode that replaces
  the window) until a reload.
  The round flag (`set_round_flag`) is a harness's per-round lever that only
  `serves()` reads. One module-global window suffices:
  `ManagedEngine._gen_lock` serialises every forward and `decode()` runs on
  the caller's session thread. The weights take two forms, both on the
  modules (nothing at module level holds an array). The MLP's gate, up and
  down are slab-only: `convert_mlp` replaces their `weight`, `scales` and `biases`
  through the module's own `__setattr__`, one linear at a time (mlx's
  parameter tree holds the planes, mlx's arrays are dropped), and marks each
  `_sous_slab` (`slab.SLAB_ATTR`, a `SlabInfo(n_real, k, tiles)`, set with
  `object.__setattr__`); stock `quantized_matmul` raises on the flat planes
  rather than compute garbage, and every path reads them — the MMA kernel's
  `SLAB` bodies and the one-row kernel's slab body (bitwise the mlx-layout
  bodies), int8's GEMM through its slab address, the tile. The attention and
  GDN linears keep mlx's arrays for prefill and the one-row kernel and, at
  scope `all`, gain a stacked slab copy: one `Stack` shared by its members
  as `_sous_tile`, each member's place in it as `_sous_tile_index` — q+k+v
  and o_proj per attention layer, qkv+z, b+a and out_proj per GDN layer (the
  four GDN inputs as one stack ran 1.06x the kernel's time); a member called
  alone launches only its own tiles when its columns sit on tile edges, else
  its whole stack (b and a share one tile). Every reader takes the
  descriptor `projkernel._weights(m)` returns — the bare `(weight, scales,
  biases)` 3-tuple for an mlx-layout module, `Weights(w, scales, biases,
  slab)` for a slab-only one, normalised by `_as_weights` — and derives K
  and N from it, never from `w.shape`. `engine/slab.py` holds the layout
  (codes `[T][G][256][8 words]`, each 32-byte unit one column's one group of
  64 codes, mlx's bytes verbatim; bf16 scales and biases `[T][G][256]`; N
  zero-padded to whole 256-column tiles, whose padding computes exactly 0),
  `repack` and `unpack` (a bitwise round trip), `SlabInfo` and the marker,
  and imports nothing of sous's: projtile imports projkernel and int8prefill
  at module level, so `projkernel` imports `projtile` only inside its two
  seams and `int8prefill` never does — both read the marker through `slab`.
  A slab-only linear's calls of 9–127 rows, which hook 2 hands on and stock
  cannot read, go to int8's GEMM: `_rows_ok(x, slab)` admits a slab-only
  module from `SLAB_MIN_ROWS` = 9 rows counted as the product of the leading
  dimensions, as hook 2 counts them; that is why int8 prefill must be
  active, and why the fork key reads the conversion — those prefill chunks
  carry int8's bits where stock's were. `PLANS` maps a key, (K, the members'
  real widths in call order), to a `Plan(tn, s, groups, pipe, splits)`: the
  27B's fused verifier keys at the plans tuned for them and the alone forms
  hook 2 reaches, 12 keys, all tuned on the 20-core M5 Pro (`MEASURED_CORES`
  = {20}). A key outside the table stays on the MMA kernel, never on a
  default plan (one GROUPS step off cost 15–20 %), and split counts are
  frozen per key, since another count adds in another order. `enable()` runs
  on the `sous-model-load` thread after `projkernel.enable` and before block
  resolution, the drafter's route (which reads none of the target's modules
  and runs once the block has resolved), kv8, the fork store and
  `measure_cache_budget`, never raises and starts with `clear()`. It checks, in order: the config
  (`false` is `off`); the projection kernel `active`; the attention tile
  `active`; a DFlash drafter (else `no drafter`); int8 prefill `active`; the
  GPU core count in `MEASURED_CORES`; the model (`_model_reason`: a dense
  MLP, every MLP and to-be-copied linear eligible and tagged by the kernel,
  every MLP linear tagged by int8, every MLP key planned and passing
  `check_plan`) — each one INFO line and `unavailable`
  (`_refuse(expected=True)`); then the kernel's source pins
  (`projkernel._pins_reason()`) and `availability()` (the platform rule,
  then one compile and run of every distinct plan, the dual, the slab MMA
  bodies, the slab one-row body and int8's slab GEMM, success remembered
  only) — one `warnings.warn` and `unavailable`. Then, all counted in
  `probe_seconds`: `_compile_real_shapes` (every slab path at the MLP's own
  shapes on zero planes, evaluated before the first launch for the deadlock
  above, and before anything is converted, so a toolchain that rejects one
  refuses the tile with the model untouched), the memory rule, the
  conversion, the copy (nothing built leaves scope `mlp`, "no stack key has
  a plan"), `warm_up` (every launch the model reaches on the tile and every
  slab path, on zero rows, so nothing compiles inside a turn) and the probe.
  A failure there warns, and once the conversion started also undoes it
  (`revert_mlp`, bitwise, then `drop_stacks`), an exception's frames cleared
  first so its arrays go before the revert; a revert that raises leaves the
  markers authoritative, since every reader and the fork key are
  marker-driven, and the refusal's reason adds how many MLP linears stay
  slab-only. The memory rule (`copy_decision`) decides in bytes: the
  conversion is size-neutral; `"mlp"` never copies and `"all"` always does;
  `true` under the automatic budget copies iff `budget_left` —
  `live_headroom()` read before the copy with the drafter loaded, less
  `copy_bytes`, the turn's reserve and `CACHE_BUDGET_SLACK` — is at least
  `TILE_COPY_MIN_BUDGET_GIB` = 14 GiB (a fork beside a retaining
  conversation: a copy that evicts the forks costs more prefill than it
  saves decode), under a configured `prompt_cache_gb` iff that budget fits
  beside the copy, with `[model].prompt_cache` off (`cache_enabled`, which
  `VLMEngine.__init__` hands `enable()` from `prompt_cache`) iff the copy
  fits beside the reserve and the slack, as under a budget of 0, since no
  slot or fork holds any, and keeps `mlp` when the headroom is unreadable;
  never a warning. The copy is 4.09 GB on the 27B (b+a pads to one tile). Measured on
  the 64 GB M5 Pro with the automatic budget: at 262144 the budget is about
  17.9 GiB without the copy and 14.07 GiB (15,102,367,760 bytes) with it,
  0.07 GiB above the rule, so the automatic scope is `active:all` there but
  sits on the rule's edge (a change to the weights, the drafter, mlx, the
  window or the reserve can flip it to `mlp`), and a maintainer who wants a
  fixed scope pins `"all"` or `"mlp"`. At 131072 the same arithmetic (the
  reserve 8 GiB smaller) leaves about 22 GiB and the scope `all` — derived,
  not measured. `measure_cache_budget` runs after the copy and finds
  it inside `active`. The probe takes the first
  full-attention layer's MLP (layer 3 of the 27B) and, at `all`, its
  attention and the first GDN layer's linears (layer 0), every call and each
  of its linears alone: every row at T = 1..8 `array_equal` to the same row
  alone; each linear's relative RMS against an elementwise fp32 reference
  within `TILE_FLOOR_FACTOR` = 1.05x the bf16 floor (that reference rounded
  to bf16 — T-invariance compares the tile with itself, this catches one
  that lost precision); a NaN in one row leaving the others bitwise alone.
  Then the slab-only MLP's other paths, bitwise against mlx's layout (the
  MMA kernel at T = 1..8 and, for a lone linear, the one-row kernel and
  int8's GEMM at 9, 64 and 128 rows) on its real linears, gate+up and a
  synthetic linear per K in each `CANCELLING` layout; last, the verifier's
  `_feed_forward` and the MLP's own call at 5 rows through both hooks, on
  the tile inside a window and on the MMA kernel outside it. It leaves
  `_active` 0, the window shut, the round flag and both counter dicts as
  found, and calls `mx.clear_cache()` in `finally`. `projection_tile_status`
  carries `state`, `scope`, `reason` (at `mlp`, why the copy was skipped),
  `copy_gb` (GB of 1e9 bytes, at `all`), `probe_seconds` and
  `budget_left_gib`: the status document copies the first five, and the load
  line prints `projection_tile=<state>`, active as `active:<scope>` followed
  by `projection_tile_probe_s=`, `projection_tile_budget_gib=` (what the
  automatic budget has left beside what was copied) and
  `projection_tile_copy_gb=` or `projection_tile_copy=skipped`. Nothing is
  logged per turn; `projkernel.calls["verify_tile"]` (runs) and
  `["plain_tile"]` and `int8prefill.calls["int8_small"]` count its paths.
  The fork key's `ptile` (`fork_key_state`: `active` when every MLP linear
  carries the marker, `off` when none does, `partial:<marked>/<total>:<8
  hex of sha256 over the marked linears' sorted paths>` after a revert that
  raised part-way, whose layers split between int8's bits and stock's on a
  9–127-row chunk, KV neither other load computes, and which layers split
  matters as much as how many) keys the
  moved 9–127-row chunks; `slab.py` and `projtile.py` are in
  `_EPOCH_FILES` (the repack decides the path every MLP read takes) and
  `projection_tile.metal` and `.h` in `_EPOCH_EXCLUDED_KERNELS`: they serve
  only inside the window and no stored KV carries their bits, while the
  three re-addressed kernels' slab bodies stay hashed. `clear()`
  (`VLMEngine.unload()`, beside `projkernel.clear()`) drops `_active`, the
  window, the round flag and the scope, every one a plain value; the planes
  go with the modules. The constructors default it off
  (`VLMEngine(projection_tile=False)`; `LMEngine` reports `off`) and only
  `default_engine_factory` hands them the config's value. Tests swap
  plain-mlx stand-ins in for `projtile.tile` and `tile_dual`
  (`tests/projtile_fixtures.py`: `guard_tile_state`, `window()`,
  `hooked_tile`, `tile_ready`), and the `tile_nax` tests skip on
  `projtile.availability()`, never another kernel's.
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
  verify row reads the same codes as the decode step at its key count, so
  a verify row equals its one-row call: as for the tile, what greedy
  parity with the drafter on or off rests on while the projection tile is
  not active and block invariance while it is. kv8 itself differs from the
  bf16 tile by the quantisation, not in the
  last bits. The mirror exists only inside `VLMEngine.decode()`, which,
  when kv8 is active, calls `kv8.open_window()` inside the draft-context `seed` block
  just before `stream_generate`, only the projection tile's window opening
  between them (so the one-row forward that ends
  the decode call's own prefill segment is covered), and
  `kv8.close_window(cache)` in that loop's `finally`, after the projection
  tile's close: it drops every
  layer's mirror and trim wrap, returns `(state, rows)` for the
  `Speculation` (`rows` one full-attention layer's count: every layer
  quantises the same rows, and the window sums them over its `layers`), and
  calls `mx.clear_cache()` when a mirror was built.
  Every `prefill()`, the load-time probes and anything outside `decode()`
  run with no window and never build or read a mirror, so the KVCache,
  the slots, the forks and the fork store stay bf16 and nothing kv8
  serves is ever persisted. The window is a module global (`_window`, a
  `Window` with `serial`, `state` `idle`/`served`/`capped`, `rows`,
  `served`, `built`, `force`, `layers`): one suffices because
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
  defences. A served call quantises `[n, min(offset, L))` once, so no row
  at or past `L` is ever written (`kv8.prepare(cache, t)`, before a
  verify's run loop): the first build writes the cap-sized buffers in one
  dispatch per layer, later calls
  slice-assign a row-sized output in place, and a growth (`GROWTH` = 4096
  rows, never past `round_up(L, 256)`) goes layer by layer, each new
  buffer evaluated before the old one is dropped. A row is served iff
  `kv8.serves(n_row)` — a window is open and not capped and `N8 ≤ n_row
  < L` (or the window is forced) — the one predicate both tile seams call
  (decode: `n_keys`; a verify run: its first row's key count). `N8`
  (4481, a module constant: the merge gate's round cells on the M5 Pro
  measured kv8's saving net of the per-round quantiser negative at `N0`
  and non-negative from 4,096 keys, hence the step boundary at or past
  4,096) and `L` are tile step boundaries, so no
  `verify_plan` run straddles either and a call mixes kv8 and the bf16
  tile only across runs. `L` (`_ceiling`) is computed once by `enable()` with the pure
  `ceiling(window, splits)` from `[model].max_context_tokens`: the largest
  step boundary whose mirror fits beside the turn's bf16 KV inside the
  window-sized reserve at every instant (`1.504·C + 0.504·C/16 + 272 ≤
  window`, `C = round_up(L, 256)`: the bf16 KV, the mirror, one layer's old
  buffer during a growth, and the rows a straddling verify holds past L),
  about 84K at 131072 and 170K at 262144. A decode grows its KV by
  `window.DECODE_KV_STEP` = 4096 rows, which can overshoot the window-sized
  reserve by up to 4,095 rows (about 256 MiB on the default model), mirror
  or none; under the automatic budget `promptcache.CACHE_BUDGET_SLACK`
  (2 GiB) absorbs it, while a configured `prompt_cache_gb` holds nothing
  back for it and a decode near L can then run about 256 MiB past the
  recommended working set (Metal paging, not a wrong row), and either way
  the ceiling keeps its 16 + 256 overhang term and the rows kv8 serves do
  not depend on the decode's growth step. The reserve, the slot budget and
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
  no cycle. `enable()` runs after `projkernel.enable` and `projtile.enable`
  (the tile's state is settled) and before the fork store and
  `measure_cache_budget` (the
  probe's arrays are released first); its first statement clears
  `_active`, `_window` and `_ceiling`. It checks, in order: the config (`off`); the
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
  `sous-model-load` thread. A floor that is not a step boundary, pin
  drift, a compile or probe failure and any
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
  the hybrid prompt-cache path (#142). mlx-vlm's round loop
  (`speculative/dflash.py:_dflash_rounds`) gives the drafter, on its first
  draft call after `reset`, the hidden states of whatever the *decode call*
  prefilled — the 7-token generation prompt there, since `PrefixCache._run`
  prefills the stable render in a drafter-less call and stops at the anchor.
  So every segment a turn prefills before its decode — the fork stops'
  and the last — runs with a `capture` the orchestrator threads through
  (`True` starts one, the engine's return continues it, None when the
  engine has nothing to seed): the prefill passes mlx-vlm's own
  `capture_layer_ids` kwarg to `generate()` and a class-level post-hook on
  the language model's forward (armed per instance with
  `object.__setattr__`, like int8's tags) keeps the newest window of each
  forward's `hidden_states` in one `Sink` — `generate_step` deletes every
  chunk's output, so the hook is the only observer, and the prefill itself
  runs exactly as uncaptured, so the KV is bit-identical. A sink that fails
  part-way voids itself: a tail ending short of the generation prompt would
  seed the wrong positions. The decode assembles the tail (its own buffer:
  a slice of the chunks would pin twice its size) and `seed()`s it into
  instance-level wraps of the drafter's `draft_block`, `draft_block_greedy`
  and `reset`, which prepend it once (the greedy wrap may delegate to the
  sampled one — the first-call flag is consumed by the outer call). The
  wrap hands over at most one window — the newest rows of the tail that
  fit beside the call's own rows, none when those already fill it: the
  drafter's sliding layers would skip the rest, and on the trimmable path
  the call's own rows are everything after the last fork stop. The
  `_Seeded` context drops its reference once the drafter holds the tail,
  so the tail is freed by the first draft call, not the end of the decode.
  On the trimmable path the decode call
  prefills the last segment itself, so only fork stops feed the capture
  there. `unload()` must `close()` the context first: the wraps close over
  the drafter and the drafter binds the target's embedding and head, so
  the weights would outlive the unload. Nothing here changes a token — the
  verify is exact — so the gates are structural (kind `dflash`, none of the
  hooks the round loop looks for, target layers named, no
  `_batch_invariant_decode`), not source hashes, and a drift shows on the
  turn line as `draft_context=7` (the capture lost) or as `draft_context=0`
  beside `draft_context=unavailable` on the load line (a gate tripped) —
  the context gauge is measured only while the seeding is active; the
  count gauges are 0 when no drafter ran and when it ran no round (one
  token asked for), the prompt cache off included — there the drafter runs
  over the whole prompt and the counts are logged the same way.
  `tests/test_draftctx.py` drives the real `generate_step` and
  `_dflash_rounds` on stubs (CPU mlx suffices) to pin both contracts.
  Per-decode acceptance (`rounds_since`, mlx-vlm's lifetime counters,
  which its per-call `accept_lens` are not) reaches the turn line through
  the prompt cache's per-turn gauges (`draft_rounds`, `drafted_tokens`,
  `accepted_tokens`, `draft_context`) via `hooks.speculation()` after every
  decode.

## Security boundary

sous executes nothing. `src/sous/api/` never runs a tool (Claude Code does,
under its own permission mode: auto mode's classifier, the allow/deny rules
and the sandbox are the boundary around the local model) and never logs a
request body, header value or query string. Its lines go through `sous.logs`
(one timestamped, levelled shape on the root handler); `_log_turn` runs on
the event loop, so nothing that blocks may ever be added to it — do such
work on the turn's thread. It forwards the client's credentials to
`[server].upstream_url` and nowhere else, and stores none. The fork store
(`~/.sous/forks`, 0700/0600) is the one place the daemon keeps
prompt-derived data at rest: rendered tool-block and system-text token ids
plus their KV, never a header or a credential; it logs counts and bytes,
never a token. `sous claude`
(`cli.py`) never sets `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, a tier
variable or a permission mode. Do not add confinement, allowlists or approval
flows to sous: the daemon guards its own HTTP surface (`sous/loopback.py`)
and nothing else. A change that makes any of these otherwise needs the spec
(`docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`, as amended by
`docs/superpowers/specs/2026-09-19-remove-mcp-path-design.md`) changed first.

`src/sous/tune/suite/tools.py` runs commands on copies of the suite's
synthetic projects inside `sous tune` only. It is a test harness, not a
boundary, and must never be reachable from the daemon.

## Workflow

- Comments explain non-obvious *why*; never restate what code does.
- Conventional Commits (`feat:`/`fix:`/`docs:`/...), imperative lowercase
  subject, *why* in the body.
- `main` is protected: branch + PR, all four CI jobs green.
