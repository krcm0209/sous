# CLAUDE.md

sous is an MCP daemon that delegates mechanical coding tasks from Claude
Code to a sandboxed local MLX worker. macOS / Apple silicon only. The goal
is plan economics: volume output is generated locally for free so heavy
Claude Code use stretches further — evaluate features against that goal.

## Commands

- `uv sync` — full setup (installs Python 3.14 and everything). Never pip.
- `uv run pytest -m "not model"` — the test suite. `model`-marked tests
  download multi-GB weights and are local/manual only; `slow`-marked tests
  spawn real processes and take seconds — run them, don't skip or mock them.
- `uv run ty check` — type check (ty, NOT mypy; covers tests and scripts too).
- `uv run ruff check . && uv run ruff format --check .` — lint/format.
- `uv lock --check` — lockfile sync. CI runs exactly these four jobs.
- `uv run python scripts/e2e_smoke.py` — agent loop against a tiny real model.

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
  after dependency changes, verify with one real delegated task. After the
  release that thread cannot run another mlx op that needs a stream — any
  array op, `mx.eval`, a model load — with "There is no Stream(gpu, 0) in
  current thread"; device and allocator calls (`device_info`,
  `get_active_memory`, `get_cache_memory`, `clear_cache`) and the freeing of
  arrays made earlier still work, which is what `server._mlx_memory_gb` and
  `context._live_memory` rely on from threads that are reused. So a thread
  that releases before each unit of work ends — a gateway pool thread, which
  releases after every turn — may keep querying and freeing, but must never
  run a load: a load runs on a `sous-model-load` thread of its own
  (`EngineManager._load`). Before it had one, a load on a pool thread made
  every later cold start on that thread fail. A thread that controls its own
  exit, like the worker loop, touches mlx freely and releases once on the
  way out.
- e2e_smoke.py often ends `failed` or `budget-exhausted` even when it worked —
  the 0.6B model can't reliably emit `finish`. Judge by hello.txt content.
- Budget exhaustion is `done` with outcome `budget-exhausted`, never `failed`.
- Tests must never touch the real `~/.sous` — always pass tmp_path-based
  config_path/data_dir.
- `docs/superpowers/**` are point-in-time design/plan records: never edit,
  reformat, or "sync" them with current code (they are also ruff-excluded).
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
  and `prompt_cache_stats` take an `owner`: the worker retires its session's
  thread (`session.thread`) at task end and never calls the bare `reset()`,
  which drops the gateway's slots too. A `fork` slot is a *copy* taken while
  a turn prefills past a boundary (`fork_point`, 4096-token floor). There are
  two: the *tools* boundary — Qwen3.5/3.8 render `# Tools` *before* the
  client's system text, so sessions and projects presenting the same tool
  array share ~45–56K rendered tokens (Phase 3a's "cross-session reuse is
  impossible" read the request body's order, not the render's) — and the
  *header* boundary, the whole system block. Both come from probe pairs
  (`promptcache.probe_boundaries`), never from rendering the system turn
  alone, which Qwen3.8's template refuses. Warm turns resolve the probe too:
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
  (`gateway/convert._place_inline_system`), never hoisted into the system
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
  the M5 Pro for the kernels themselves — CI cannot load the models. Slots
  built before this fix hold mispositioned keys: only a daemon restart drops
  them.
- Prompt-cache per-turn gauges (`promptcache.TURN_GAUGES`, reset by
  `PromptCacheStats.begin_turn` on every `generate()` and again on a cold
  retry) are assigned per turn and read back directly from the owner-scoped
  `after` snapshot in `TurnRunner.run` — never as `after − before`. Counters
  (`forks`, `evictions`, `pressure_evictions`, `reused_tokens`) are deltas.
  Timers read `promptcache._clock` so tests can drive them; `Slot.last_used`
  keeps the real clock. The gauges exist for the gateway's turn line only:
  every MCP-facing `prompt_cache` block (a task report, `server_status` via
  `EngineManager.status`) goes through `without_turn_gauges`, because there a
  gauge is a task's last `generate()` beside task-long counters, or a max
  over every owner ever seen — tokens the frontier model pays to read nothing.
- `EngineManager` never holds its lock across a model load or unload
  (`_loading`/`_unloading` on a `Condition`): `status()`, `hold()` and the
  idle sweep must answer during either, and `/sous/status` reports
  `loading`. A `hold(pid, create_time)` pins the model while that process
  lives — the sweep, `status()` and `hold()` all prune holders with psutil
  (pid gone, a zombie, or start time off by more than a second = reused
  pid), and whichever prunes the last one restarts the idle clock; the
  preload thread `hold()` may start calls `get()` and then
  `release_mlx_thread_state()` like every mlx-touching thread. A gateway
  turn re-checks `abandoned` after `engines.get()`: that is where it now
  waits out another thread's load (the lease no longer does). The `/sous/`
  routes (`sous/monitor.py`) are mounted in `create_server` *before* the
  gateway and whether or not it is enabled, behind the loopback guard in
  `sous/loopback.py` that the gateway shares; a `/sous/` path never reaches
  the upstream, and a failure inside a route is a JSON 500, never
  Starlette's text one. `sous claude` has no MCP client: `/sous/status`, or
  a 404 from whatever holds `daemon.lock` = "restart the daemon" (from
  anything else = "not sous on this port"), no compatibility fallback. The
  status document is one shape everywhere (`SousService.status_document`):
  `engine`, `inflight`, `queue`, `config`, and over HTTP `recent_turns` and
  `recent_tasks` — the MCP tool drops the two lists because a frontier
  model pays for every key it reads.
- `sous/inflight.py` is the in-flight turn registry: a gateway turn
  registers under its `msg_` id in `TurnRunner.run`; `progress()` runs on
  the engine's session thread from inside the decode loop, so every
  registry method is a dict write under one lock and never raises (an
  unknown id is ignored). The runner retires a turn *before* it releases
  the gateway lock — the lock is not a queue, so a finished turn left
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
  `/sous/events` polls `SousService.status_version()` — the registry's
  version, `EngineManager.version` (load, unload, hold, release, and
  every idle-clock reset: `touch()` and a `get()` hit),
  `TaskStore.version` (any connection that inserted, updated or deleted a
  row) and the config
  file's mtime and size — ten times a second while the last document
  showed a turn, a running task (`queue.running`), a load or an unload
  (`engine.unloading`), and twice a second otherwise, rebuilding
  the whole document on a worker thread when the tuple moved; the
  once-a-second heartbeat runs only while busy (the TUI reads a silent
  stream during a turn as a stalled daemon, and a task's clock is in the
  document), never idle — the idle clock is the TUI's to run
  (`Top._engine_now`, seeded by every reset the engine version carries).
  The build memoises the task counts and listing on the store's version
  (two slots: the narrow MCP build never reads the listing) and the
  allowlist on the config file's mtime and size; summaries are per build. Never
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
  per-local-model lever is `[gateway].max_context_tokens` (the default
  model's native length is 262144, at +8 GiB of KV reserve) — the classic
  threshold scales with it, the precompute fraction is remote-configured
  per window size and unmeasured at 262144.
- The gateway forwards every request it does not serve (`gateway/upstream.py`)
  as a transparent proxy: never re-serialize a forwarded body, never add or
  alter an end-to-end header (only `Host`, the hop-by-hop set and a buffered
  body's `Content-Length` change; responses lose the hop-by-hop set and gain
  `Via`, with uvicorn's own `Date`/`Server` replacing the upstream's), never turn
  `trust_env` on. The routing predicate is the decoded body's `model` after
  stripping one trailing `[…]` suffix; a body that does not decode is the
  upstream's, not a 400. Any third-party library that enters the gateway's
  request path gets its logger pinned in `mount_gateway`: sse-starlette logs
  each SSE frame, httpx the full upstream URL with its query string, httpcore
  response header values — and `MCPServer.__init__` installs a root stderr
  handler at INFO, so none of that is hypothetical.
- `engine/int8prefill.py` compiles the Metal kernel pair in `engine/kernels/` at
  model load (Apache-2.0, derived from oMLX — see THIRD_PARTY_NOTICES.md). The Python `reorder_k` and
  the GEMM's nibble decode are two halves of one K-order contract (slot
  `16c+4t+j` holds code `16c+8*(t>>1)+2j+(t&1)`): never change one alone; the
  power-of-two bit-exactness test is what catches a mismatch. Only rows >= 128
  route (decode/verify never do), only tagged modules route (tags are set with
  `object.__setattr__` so they stay out of mlx's parameter tree) and only on
  `model_type` `qwen3_5` (the MoE variant reuses the same classes and is refused), and only the
  GEMM's header includes the Metal-4 `MetalPerformancePrimitives` header, which
  needs macOS 26.2+ — the Stage-A kernel must keep compiling on any Metal GPU
  so CI (macos-15, no tensor units) can test it.
- `int8prefill.enable()` never raises and a model load never fails because of
  it: every failure is one `warnings.warn` plus `state: unavailable`. Tests
  that need the GEMM without tensor units monkeypatch `int8prefill.qmm`.

## Security boundary

`src/sous/toolexec.py` is the sandbox (path confinement, command allowlist,
process-group kill, stat audit). Any change there needs a test that fails
without it. Odd-looking code is load-bearing (the un-reaped zombie during
the group kill, EPERM suppression, ctime in the audit) — read the comments
before touching. Suspected-flaky tests get run in a loop, not judged on one
pass.

`src/sous/gateway/` is deliberately outside that boundary: it never executes a
tool (Claude Code does, under its own permissions) and never logs a request
body, header value or query string. Its lines go through `sous.logs` (one
timestamped, levelled shape on the root handler); `_log_turn` runs on the
event loop, so nothing that blocks (a database write) may ever be added to
it — do such work on the turn's thread. It forwards the client's
credentials to `[gateway].upstream_url` and nowhere else, and stores none.
`sous claude` (`cli.py`) never sets `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_API_KEY` or a tier variable. A change that makes any of these
otherwise needs the spec
(`docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`) changed first.

## Workflow

- Comments explain non-obvious *why*; never restate what code does.
- Conventional Commits (`feat:`/`fix:`/`docs:`/...), imperative lowercase
  subject, *why* in the body.
- `main` is protected: branch + PR, all four CI jobs green.
