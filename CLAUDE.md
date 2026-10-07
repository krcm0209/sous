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
  heals on the next take. Inline `role:"system"` messages after the first user message
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
  helper's — hence the `mlx-vlm>=0.7.0` floor; below it the kwargs are silently
  overwritten. A suffix-only array is sliced the same way, so the array is
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
  projkernel.py, kernels/*.metal and *.h) beside the versions, the GPU and
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
  to INFO, so none of that is hypothetical.
- `engine/int8prefill.py` compiles the Metal kernel pair in `engine/kernels/` at
  model load (Apache-2.0, derived from oMLX — see THIRD_PARTY_NOTICES.md). The Python `reorder_k` and
  the GEMM's nibble decode are two halves of one K-order contract (slot
  `16c+4t+j` holds code `16c+8*(t>>1)+2j+(t&1)`): never change one alone; the
  power-of-two bit-exactness test is what catches a mismatch. Only rows >= 128
  route (decode/verify never do), only tagged modules route (tags are set with
  `object.__setattr__` so they stay out of mlx's parameter tree) and only on
  `model_type` `qwen3_5` (the MoE variant reuses the same classes and is refused), and only the
  tensor-op kernels — the GEMM and the attention tile's split kernel — get
  `nax.h`, whose Metal-4 `MetalPerformancePrimitives` include needs macOS
  26.2+: the Stage-A kernel and the tile's reduce kernel stay MPP-free, and
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
  by both tensor-op kernels; past it, the compile half:
  `int8prefill.availability()` compiles and runs the GEMM once where the
  tensor units exist (`_gemm_probe`) and `tileattn.availability()` every tile
  variant (`_compile_probe`) — GPU work only, success remembered, the full
  compiler report logged: a rejected kernel reads as `unavailable` and that
  kernel's `nax` tests skip, where a compile failure with a CPU-stream op in
  flight deadlocks in mlx's exception path. Being mlx ops, both inherit the
  thread rules above.
- `engine/verifyattn.py` replaces mlx-vlm's per-row SDPA loop in the exact
  verifier (`Qwen3_5BatchInvariantForward._attention`, T = 3..8 while the
  attention tile is inactive) with one stock
  `mx.fast.scaled_dot_product_attention` per group of rows that share mlx's
  kernel plan. Grouping is bit-exact only because inside one plan mlx
  assigns key i to simdgroup `i % 32` or block `i % blocks` whatever the key
  count and the causal mask skips excluded keys, and `plan()` mirrors mlx
  0.32.2's dispatch (`VALIDATED_MLX`) — re-read
  `scaled_dot_product_attention.cpp` and extend it on every mlx bump; every
  mlx-vlm function the hook reads is pinned by source hash
  (`VALIDATED_MLX_VLM_SOURCES`), checked at load because a tool environment
  can still be handed another mlx-vlm than the lock. `pyproject.toml` pins
  mlx and mlx-vlm with `==` to exactly what these sets, tileattn's and
  projkernel's validate (#168): raise a pin only in the commit that extends
  them; `tests/test_dependency_pins.py` holds the two together. While the tile is
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
  from 1 while the flag is set. The flag `_active_splits` is a plain int,
  the active S or 0, never a model reference: the first statement of every
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
  false) resolves in `VLMEngine.__init__`, when a drafter loaded and before
  `_pin_block_size`, to `SPECULATIVE_BLOCK_KERNEL` = 5 where the kernel is
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
  turn; `projkernel.calls` (`verify_kernel`, `verify_split`,
  `plain_kernel`, `declined`, and `one_row`, one per one-row dispatch
  `linears()` serves; warm-up and probe leave them all as found) counts
  every path for the tests and for
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
  stand-ins in for `projkernel.mma` and `projkernel.one_row`
  (`pfx.stand_ins`; `pfx.real_kernel` puts both real kernels back), and
  skip the kernel's own tests on `projkernel.kernel_available()`, which
  compiles and runs every variant and the one-row kernel once (the one-row
  kernel bitwise against the MMA kernel at T = 1, random and cancelling)
  and remembers success only: the lane map was probed on Apple GPUs,
  so a GPU where it differs skips rather than fails.
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
