# sous — Gateway Observability: attribute a turn, watch it live, keep the record

*2026-09-10. Continues issue #41 after Phase 3c (PR #72), PR #74 (`lcp=` on
misses) and PR #75 (kernel pressure valve). Grounded in an audit of every
observability surface the daemon has (the `sous status` command, the MCP
`server_status` tool, `~/.sous/daemon.err.log`, Claude Code's own transcripts)
and in a forensic reconstruction of a 13m14s subagent session that the current
log could account for only by hand. Four PRs in this order: **1** attribute a
turn, **2** live status and the TUI, **3** persist turns and `sous turns`,
**4** docs (folded into 1–3 where they land). PR 1 ships before anything in
the companion spec `2026-09-10-gateway-cache-continuity-design.md`, whose
changes are verified with PR 1's fields; that spec's PR B introduces the
`/sous/` route module and `GET /sous/status` that PR 2 here extends.*

## What the audit established

`sous status` never talks to the daemon: it probes the TCP port and reads
`tasks.db` directly (`cli.py:_cmd_status`). Gateway turns are not tasks
(`Gateway` is constructed without a store) and are persisted nowhere. The only
record of a turn is one stderr line per **completed** turn, with no timestamp
(`routes._log` is a bare `print`), in an unrotated `daemon.err.log` that
already holds 38 restarts and 79 local turns. During a 13-minute turn the
gateway itself says nothing; the model-load window is bracketed only by
huggingface_hub's own chatter.

That file's name is a launchd artefact, not a description: it is the
`StandardErrorPath` from the plist `sous install-launchd` writes, and the
split between it and `daemon.log` is by Unix stream, not severity. Python's
logging module writes to stderr by default, the gateway's `_log` prints there
on purpose, `warnings.warn` goes there, the MCP SDK installs a stderr handler
at INFO (a `RichHandler` that wraps every line at 80 columns), and
huggingface_hub's progress bars land there too. So the "error" file holds
essentially everything and `daemon.log` holds the startup banner. CONTRIBUTING
tells readers to watch `daemon.log` for gateway lines that never appear in it.

The turn line carries `seconds` as one number. `started` is stamped *after*
`TurnRunner._lock` is acquired, so it excludes the lock wait (measurably 9.6,
2.1 and 2.5 s in the session studied, when Claude Code's progress-summary
calls held the lock) and includes the model load (~13 s on T1), `count_tokens`
on a 66K prompt, the fork probe, prefill, snapshot, decode and restore, with no
split. Both mlx backends discard the `prompt_tps`/`generation_tps` the
libraries compute; on the VLM path prefill and decode are already separate
calls in `PrefixCache._run`, so a split is a handful of monotonic reads. No
line says what a turn *published* (`forks=`), what it *took*, whether the
pressure valve *charged* it (`pressure_evictions` exists since PR #75 and never
reaches the log), or where a miss diverged relative to the fork boundaries
(`lcp=62851` was legible only after the header boundary, 62,856, was
reconstructed from another turn's `reused_tokens`). A locally served
`count_tokens` can load the model and logs nothing on success. Failure lines
carry no `seconds`.

Across all 79 turns ever logged, 21 misses account for 3,837.5 s — 81% of all
local generation time — at a mean of 182.7 s against 8.7 s for a hit. The
project's stated goal (stretch the subscription) is not measured anywhere.

## Goal

Three questions answered without archaeology: *what is the local model doing
right now, and how far along is it* (live, from a second terminal or inside
Claude Code's status line); *why did that turn take that long* (from one log
line, or one row); *what has it done this week* (`sous status`, with totals).
Every new datum is a count, a duration, a hash or an identifier — never a body,
a header value or a query string.

## Non-goals

- True intra-prefill progress. mlx-vlm exposes no prefill progress callback
  (mlx-lm's is popped when a drafter is in use), so sous would have to chunk
  `VLMEngine.prefill` itself — a throughput-sensitive change gated on its own
  measurement. Until then the prefill phase shows elapsed and an ETA from a
  rolling prefill rate.
- In-band progress events to the API client. Only `ping` is documented as
  skipped by both official SDKs; an unknown SSE event type risks Claude Code's
  accumulator.
- Rotating the daemon log. launchd appends; the file is 908 KB after a month
  and 38 restarts; the durable record moves to SQLite (PR 3), and the log
  stays a log.
- Claude Desktop integration (not pursued).

## Design

### PR 1 — attribute a turn

**Log streams: one file, one shape.** Five changes, all in this PR:

1. *One file.* The plist `sous install-launchd` writes points both
   `StandardOutPath` and `StandardErrorPath` at `~/.sous/daemon.log`; launchd
   accepts the same path for both and interleaves writes in order.
   `daemon.err.log` stops existing as a name.
2. *Migration.* `sous install-launchd` learns to re-run: today it only
   `bootstrap`s, which fails on an already-loaded label. It will `bootout` an
   existing job first, then, with the daemon lock free (the `flock` probe
   `sous status` already uses — a hand-started daemon still holding it makes
   the command refuse, so no open descriptor keeps writing into an unlinked
   file), append an existing `daemon.err.log` onto `daemon.log` and remove it,
   then `bootstrap` the new plist. The history stays in the one file; the
   misnomer goes. An unmanaged `sous serve` redirects wherever its launcher
   said and is not sous's to migrate.
3. *Every line sous emits carries a timestamp and a level.* `routes._log`
   becomes a thin wrapper over a `logging` logger (`sous.gateway`; the
   continuity spec's engine lines use `sous.engine`, the monitor
   `sous.monitor`), and every record is formatted as
   `2026-09-10T19:26:14.025Z INFO sous.gateway: …` — ISO-8601 UTC with
   milliseconds, Python's level names. `INFO` for a served or forwarded
   request; `WARNING` for a request sous refused (4xx, 529, abandoned while
   queued) and for the dropped-tools note; `ERROR` for a failure sous produced
   (5xx from an exception, an unreachable upstream). `grep ' ERROR '` then
   means what the old filename promised.
4. *Library lines take the same shape.* After `MCPServer(...)` is constructed
   in `create_server`, the `RichHandler` the SDK's `configure_logging("INFO")`
   installed on the root logger is replaced by a plain `StreamHandler` to
   stderr with the formatter above, so MCP, uvicorn and huggingface_hub lines
   stop wrapping at 80 columns and become greppable; `logging.captureWarnings
   (True)` routes the prompt cache's `warnings.warn` through it as `WARNING
   py.warnings: …`. The existing logger pins (sse-starlette at INFO, httpx and
   httpcore at WARNING) stay exactly where they are, since the no-bodies rule
   depends on them.
5. *Docs.* README names the one file and the line shape and tells existing
   installs to re-run `sous install-launchd`; CONTRIBUTING's two references
   become true.

The observability of a turn does not depend on where the line goes: a
foreground `sous serve` prints the same lines to its terminal.

**The turn line.** Success lines become:

```
2026-09-10T19:26:14.025Z INFO sous.gateway:
  POST /v1/messages id=msg_… model=sous-local stream=1 status=200
  input_tokens=84335 output_tokens=2887 stop=end_turn
  cache=hit took=turn@82647 reused_tokens=82647 prefilled_tokens=1681
  forks=0 evicted=1 pressure=0
  load_s=0.0 queue_s=2.5 tokenize_s=1.1 ttft_s=9.8 prefill_s=7.9 decode_s=196.6
  prefill_tps=213 decode_tps=14.7 seconds=207.4
  tools=1b21cd75 system=9f8e7d6c
```

(one line; broken here for reading). On a miss, `took=none` and two more
fields: `lcp=62851 lcp_region=system bounds=[62298,62856]`. Field semantics:

| field | meaning | measured where |
|---|---|---|
| `id` | the response's `msg_` id — joins the line to PR 3's row and to the client's transcript | `TurnAssembler` |
| `took` | which slot served the turn and its length: `turn@N` (copied, retained), `turn-moved@N`, `fork@N`, `none` | new `PromptCacheStats` gauges `took_kind`/`took_len`, set in `generate` |
| `prefilled_tokens` | stable tokens actually prefilled this turn (`len(stable) − reused`) | `_run` |
| `forks` / `evicted` / `pressure` | slots this turn published / slots evicted while it ran, charged to this owner / of which by the pressure valve | owner-scoped before/after deltas of existing counters (`forks`, `evictions`, `pressure_evictions`) in `TurnRunner.run` |
| `load_s` | wall time of `engines.get()` (≈0 when loaded) | `TurnRunner.run` |
| `queue_s` | wait for `TurnRunner._lock` | stamped before `acquire` |
| `tokenize_s` | `count_tokens` plus the probe renders | around `engine.count_tokens` and inside the probe closure |
| `ttft_s` | from `generate()` entry to the first `Delta` | `on_delta` in `TurnRunner.run` |
| `prefill_s` / `decode_s` | the two phases of `_run` (the fork-copy and snapshot time counted into `prefill_s`; the restore into `decode_s`) | four `time.monotonic()` reads in `_run`, written to gauges |
| `prefill_tps` / `decode_tps` | `prefilled_tokens/prefill_s`, `output_tokens/decode_s`; `-` when the divisor is 0 | `_log_turn` |
| `seconds` | unchanged: lock acquired → result | `TurnRunner.run` |
| `tools` / `system` | `blake2b(digest_size=4)` hex of `json.dumps(tools, sort_keys=True)` and of the rendered system text (after volatile stripping) | `convert.parse_messages_request`, carried on `ChatRequest` |
| `lcp_region` | `tools` if `lcp < bounds[0]`, `system` if `< bounds[1]`, else `conversation`; `-` when no bounds | `_log_turn` |
| `bounds` | the boundaries the probe verified this turn, ascending | gauge set in `_fork_boundaries` before the `reuse < b` filter |

Gauges follow `miss_lcp`'s pattern (assigned per turn, max-folded in
`_GAUGES`, read from `after` directly, never as a delta). Every path that
bypasses `_run` (`enabled=False`, the not-a-strict-prefix bail) zeroes the
phase gauges so a line never reports a previous turn's numbers. The two hashes
are of *rendered* text, truncated to 8 hex characters: comparable across lines,
not reversible.

**Other lines.** Failure lines (`529`, `TurnAbandoned`, `GenerationStalled`,
any 5xx) gain `seconds=` measured from request receipt. Locally served
`count_tokens` logs `POST /v1/messages/count_tokens model=… input_tokens=N
load_s=… seconds=…` on success. `EngineManager.get()` logs `sous engine: model
loaded in N.N s` when it constructs an engine (the continuity spec's preload
path reuses it).

**Plumbing.** `TurnResult` gains the new fields; `_log_turn` formats them;
`PromptCacheStats` gains the gauges and `prefilled_tokens`; `ChatRequest`
gains the two hashes. No new module.

### PR 2 — live status, and a terminal that shows it

**Registry.** `sous/monitor.py` (introduced by the continuity spec's PR B for
`/sous/hold` and the minimal `/sous/status`) gains an `Inflight` registry: a
lock-guarded dict keyed by the turn's `msg_` id holding `started_at`,
`phase` (`queued` → `loading` → `tokenizing` → `prefill` → `decode` →
`done`), `input_tokens`, `reused_tokens`, `to_prefill`, `generated_tokens`,
a rolling decode rate, and an ETA (`to_prefill / rolling_prefill_tps` during
prefill, `(max_tokens − generated)/decode_tps` capped by the client's
`max_tokens` during decode). `TurnRunner.run` gains a `turn_id` argument (the
`msg_` id the route already minted for the response), registers on entry
(phase `queued`), advances phases at the same points PR 1 stamps, and removes
in its `finally`; the sink's `delta` path increments `generated_tokens`. Every update
is a dict write under a `threading.Lock` — the sink contract (never block,
never raise, runs on the engine's session thread mid-decode) is kept. The
registry also keeps the last 50 completed turns' summaries (PR 1's fields) in
memory for the TUI's recent table before PR 3 makes them durable. Delegated
tasks are read from `TaskStore` as `sous status` does today.

**Routes** (loopback-guarded, mounted before the gateway's catch-all):

- `GET /sous/status` — one JSON document: `engine` (`loaded`, `loading`,
  `model_id`, `idle_seconds`, `holders`, `memory_gb`, `prompt_cache` counters
  incl. `slots`, `resident_bytes`), `inflight` (list, usually 0–1 entries; more
  when turns are queued), `queue` (task counts), `recent_turns` (last 50),
  `recent_tasks` (last 10, id/state/title/duration), `config` (as today).
  `server_status` (MCP) returns the same document minus `recent_*`, so an MCP
  client sees the in-flight turn too.
- `GET /sous/events` — an SSE stream: the full status document as the first
  event, then a `status` event whenever the registry changes, coalesced to at
  most 10 events per second, and a `ping` every 10 s. `sse-starlette` is
  already a dependency. Closing the connection ends the stream; nothing is
  buffered for a client that is gone.

**The TUI.** `sous status --watch` (alias `sous top`) is a Textual application
in `sous/tui.py`, imported function-locally so `sous serve` and every other
subcommand never load it. Textual (≥8; adds `linkify-it-py`,
`mdit-py-plugins`, `platformdirs` beyond the already-locked `rich`) is chosen
for the presence oh-my-pi has and rich's `Live` cannot give: it takes the
alternate screen and restores the terminal on exit, so nothing lands in
scrollback; its compositor redraws only dirty regions behind synchronized
output, so nothing tears; it tweens properties with easing (`animate`) and
ticks timers at frame rate (`set_interval`); it handles resize and the mouse.

What the screen shows, top to bottom, at 100×30:

```
 sous · 127.0.0.1:8383 · Qwen3.8-27B-4bit + DFlash2         ● loaded · held by 1 · 18.4 GB
┌ now ───────────────────────────────────────────────────────────────────────────────────────┐
│ msg_3f9a2c1  sous-local  general-purpose turn                          decode · 00:42     │
│ prefill  ████████████████████████████████████████████████████ 1,681 / 1,681 tok  7.9 s    │
│ decode   █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   612 tok · eta ~00:18          │
│                                                     14.7 tok/s   ▁▂▃▅▆▇▇▆▆▇▇▇▆▅▆▇▇▇▇▇▇     │
├ recent turns ───────────────────────────────────────────────────────────────────────────────┤
│ 19:26:14  hit   turn@82647   84,335 in   2,887 out   pre 7.9s  dec 196.6s   207.4 s       │
│ 19:21:22  hit   turn@66114   82,654 in      60 out   pre 82.7s dec   3.5s    87.5 s       │
│ 19:19:53  miss  lcp=62851 system  66,121 in  73 out   pre 197s  dec   4.3s   202.8 s      │
├ cache ──────────────────────────┬ tasks ────────────────────────────────────────────────────┤
│ slots 5 · 19.6 / 23.0 GiB       │ a11e04b  done      add docstrings to engine/base.py       │
│ hits 40 · forks 18 · misses 21  │ 9c02f7d  running   scaffold pytest fixtures for tasks.py  │
│ evicted 6 (pressure 0)          │                                                            │
└─────────────────────────────────┴────────────────────────────────────────────────────────────┘
 q quit · ↑↓ scroll · 27B · high › ● loaded 18.4 GB › cache 5 slots › queue 1 › 10 Hz
```

The presence rules, which are requirements, not styling notes:

- **Alternate screen, restored on exit**; `q`, `Esc` and Ctrl-C all restore.
- **Counters move per token**, not per second: `generated_tokens` and the
  decode bar update on every `status` event; between events the bar's value
  is tweened (150 ms, ease-out) so a 10 Hz feed reads as continuous motion.
- **Elapsed timers tick at 10 fps** from `started_at`, independent of the
  event feed, so a stalled turn is visibly stalled (the timer runs, the
  counters do not, and the phase label dims after 5 s without an event).
- **Prefill shows honest progress**: an eased bar driven by elapsed ×
  rolling prefill rate, capped at 95% until the phase changes, with the ETA
  beside it; when no rate is known yet (first turn after a restart) it is an
  indeterminate pulse.
- **A tok/s sparkline** (`Sparkline`, last 60 samples) under the decode bar;
  a large-digit readout (`Digits`) of the current rate when the terminal is
  ≥120 columns.
- **New rows slide into the recent table** from the top with a 200 ms
  highlight; the table is a `DataTable` that scrolls with the keyboard and the
  mouse.
- **Phase transitions animate**: the "now" panel's border colour eases
  between phase colours (queued grey → loading amber → prefill blue → decode
  green); a miss flashes the cache panel's `misses` counter once.
- **Engine load** shows as its own panel state (`loading…` with a
  `LoadingIndicator`), since a cold turn spends its first 10–15 s there.
- **Resize** reflows immediately. At 80×24 the cache and tasks panels stack
  and the recent table keeps three rows; below 60 columns the TUI shows the
  footer line only.
- **Theme** follows the terminal's colours (Textual's default dark/light
  detection); no hard-coded background.
- Frame budget: 30 fps for animations, 10 Hz data. CPU while idle under 2%
  (verified by running it for a minute against an idle daemon).

Data path: the TUI opens `/sous/events`; if the connection drops it shows a
`reconnecting…` banner and retries with backoff; it never falls back to
polling (same daemon, same port). If the daemon is down it says so and exits
non-zero after `q`.

**`sous statusline`.** A subcommand for Claude Code's `statusLine` setting: it
reads and ignores Claude Code's JSON on stdin, fetches `/sous/status` once,
and prints one line — `sous: decode 612 tok · 14.7 tok/s · eta 18s` during a
turn, `sous: idle · 5 slots · held` otherwise, `sous: daemon down` on
connection failure. It runs once a second under Claude Code, so it imports
nothing beyond the stdlib (`urllib`, not httpx, not Textual) and returns in
well under a second. It exists so the README recipe needs no `curl`/`jq`; it
does not touch `~/.claude/settings.json`.

### PR 3 — persist turns, and `sous turns`

**Table.** `TaskStore` gains `gateway_turns` (created in `__init__`, same
`CREATE TABLE IF NOT EXISTS` pattern as `tasks`): `id` (the `msg_` id,
primary key), `ts` (REAL, epoch), `model` (the bounded label the log uses),
`stream`, `status`, `stop_reason`, `input_tokens`, `output_tokens`,
`reused_tokens`, `prefilled_tokens`, `cache` (`hit|fork|miss`), `took`,
`lcp`, `lcp_region`, `forks`, `evicted`, `pressure`, `load_s`, `queue_s`,
`tokenize_s`, `ttft_s`, `prefill_s`, `decode_s`, `seconds`, `tools_hash`,
`system_hash`. Metadata only — exactly the turn line's fields, never
`chat.messages`, never a system prompt, never a tool name. `record_turn()`,
`list_recent_turns(limit)`, `turn_totals(since)` and a turn-specific
`prune(retention)`.

**Write path.** `_log_turn` runs on the event loop on both branches; the
SQLite write therefore happens on the turn's own thread, right after the
result is assembled and before the slot is released — never on the loop.
`Gateway` receives the store through `mount_gateway` (the one signature
change). Failures to write log one line and never fail the turn.

**Retention.** `[gateway].turn_retention` (default 2000 rows; turns outnumber
tasks ~10:1), pruned after each write like tasks are.

**`sous status`.** Becomes: the daemon line; an engine line (`loaded`,
`holders`, `slots`, resident GiB); the last 10 turns with local timestamps,
cache outcome, tokens and seconds; a totals line for the last 24 h (`N turns
· X tokens generated locally · Y cache-read tokens · Z misses costing M
min`); then the recent tasks as today. It stays a static print; `--watch` is
the TUI.

**`sous turns`.** A read-only reader of Claude Code's own subagent transcripts
for runs that predate the table (and as an independent cross-check): it globs
`~/.claude/projects/*/*/subagents/*.jsonl`, keeps records whose
`message.model` is one of the daemon's `local_models`, and prints per-agent
timelines — timestamp, role, stop reason, `input_tokens`,
`cache_read_input_tokens`, `output_tokens`, and the gap to the next row.
`--since <duration>` and `--project <path>` filter. It prints **metadata
only**: no `message.content`, no tool names, no paths from inside messages.
Gaps are client-side wall time and are labelled as such; the transcript
format is Claude Code's private one, so the command is documented as a
convenience that may break with a Claude Code release.

### PR 4 — docs (folded)

- README "Managing the daemon": `~/.sous/daemon.log` is the one log (both
  streams), its line shape and level names, and that existing installs re-run
  `sous install-launchd`, which folds the old `daemon.err.log` in and removes
  it. CONTRIBUTING's "Watch `~/.sous/daemon.log`" lines are reworded to name
  the level tokens now that they are true.
- README "Gateway mode": the new turn line, field by field; the `/sous/`
  routes (loopback-only, what they return); `sous status --watch` /
  `sous top` with a screenshot-as-text; `sous statusline` and the settings
  recipe — `{"statusLine": {"type": "command", "command": "sous statusline",
  "refreshInterval": 1}}` — stating that without `refreshInterval` the line
  refreshes only on message events and would freeze for a whole turn; a
  `SubagentStart`/`SubagentStop` hook example that appends `{ts, agent_id,
  agent_type}` to a file, with the caveat that it brackets a subagent without
  seeing the progress-summary calls.
- README configuration: `[gateway].turn_retention`.
- CLAUDE.md: `_log_turn` runs on the event loop (never do I/O there); sinks
  never block; `/v1/*` is the upstream's namespace and `/sous/*` the daemon's.

## Invariants preserved

- No request body, header value, query string or response text reaches a log
  line, a row, an SSE event or the TUI. Hashes are of rendered text, 8 hex
  chars. Model labels go through the existing `_log_token` bound.
- All `/sous/` routes are loopback-only (Host and Origin), like the gateway's.
- The logger pins that keep library bodies, URLs and header values out of the
  log survive the handler swap: the swap replaces the root *handler*, never a
  library logger's *level*.
- Cache statistics are read owner-scoped on the turn thread; the registry is
  the only thing the event loop reads.
- Sinks and registry updates never block or raise on the engine thread.
- Nothing new is imported by `sous serve` (Textual is function-local in the
  CLI); the lint job on ubuntu imports every module without mlx, as today.

## Testing

**PR 1 (`tests/test_gateway_routes.py`, `test_gateway_turn.py`,
`test_promptcache.py`, `test_gateway_convert.py`):** the line format is parsed
back field by field on hit, fork, miss and moved-take turns from the fake
engine; `lcp_region` for an `lcp` below, between and above the two bounds and
`-` without bounds; phase gauges zeroed on the bypass paths; `queue_s` > 0
when a second turn waits on the lock; `load_s` > 0 exactly when the factory
ran; `ttft_s` ≤ `seconds`; `tools`/`system` hashes stable across two identical
requests and different when one tool description or one system character
changes; `count_tokens` and failure lines carry `seconds`; the timestamp
prefix parses as ISO-8601 UTC. Log streams (`tests/test_cli.py`,
`tests/test_server.py`): the generated plist has both standard paths equal to
`<data_dir>/daemon.log`; `install-launchd` appends an existing
`daemon.err.log` onto `daemon.log` byte for byte and removes it, does nothing
when the file is absent, and refuses while the daemon lock is held; after
`create_server` the root logger has no `RichHandler`, a record formats as
`<iso-utc-ms>Z LEVEL name: message`, a `warnings.warn` arrives through it as
`WARNING py.warnings`, and the sse-starlette/httpx/httpcore pins are intact;
`_log` emits INFO for a 200 line, WARNING for a 4xx/529/abandoned line and
ERROR for a 5xx line (asserted with `caplog`).

**PR 2 (`tests/test_monitor.py`, `tests/test_tui.py`):** registry phases in
order and removal on success, error and abandonment; `generated_tokens`
increments per delta; `/sous/status` schema; `/sous/events` first event is the
snapshot, updates are coalesced (100 registry changes in 100 ms produce ≤ 2
events), `ping` cadence, disconnect ends the generator; loopback refusal;
`server_status` mirrors `inflight`. The TUI is tested with Textual's
`App.run_test()` pilot against a fake event feed: the decode bar reaches the
fed value, the timer advances without events, the phase label dims after 5 s
of silence, resize to 80×24 stacks the panels, `q` exits and restores; a
snapshot test of the 100×30 render. `sous statusline` output for the three
states against a fake server, and an assertion via `sys.modules` that it
imported neither `textual` nor `httpx` nor `mlx`.

**PR 3 (`tests/test_tasks.py`, `tests/test_cli.py`, `tests/test_turns.py`):**
`record_turn` round-trips every field; migration creates the table on a
pre-existing DB; retention prunes oldest first; a write error does not fail the
turn (fake store raising); `sous status` output against a seeded DB; `sous
turns` against fixture JSONL (a subagent with local and frontier rows; asserts
no `content` string appears in the output; `--since`/`--project` filters).

**Real-daemon gates (recorded in each PR):** PR 1 — one `sous claude` session;
the log's `seconds` equals the transcript's assistant-to-assistant gaps minus
`queue_s` to within 0.5 s. PR 2 — the TUI open in a second terminal during a
subagent turn; a screen recording or three screenshots (loading, prefill,
decode) attached. PR 3 — `sous status` shows the session's turns and totals
matching the log.

## Documentation

As PR 4 above; each PR lands its own README paragraph rather than a trailing
docs PR.

## Risks and open questions

- **Textual is the first UI framework in the tree.** It is confined to
  `sous/tui.py`, imported by one subcommand, and can be dropped without
  touching the daemon. Python 3.14 is in its classifiers (8.2.x).
- **Coalescing hides tokens.** At 10 Hz and 15 tok/s every token is visible;
  a faster model would show steps of 2–3 tokens per frame. Acceptable.
- **The registry is per-process memory.** A daemon restart loses the in-flight
  view and the last-50 buffer; PR 3's table is the durable record.
- **The prefill ETA lies on the first turn after a restart** (no rate yet).
  It is shown as indeterminate rather than guessed.
- **`sous turns` depends on Claude Code's transcript layout.** Documented as
  such; PR 3's own table is the supported source.
- **A 32-bit-wide hash collides** eventually across thousands of tool sets. It
  is a diagnostic, compared within one log, not a key.
