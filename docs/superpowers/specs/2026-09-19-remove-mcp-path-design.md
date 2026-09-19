# Remove the MCP delegate path and refocus sous on the endpoint

Design record, 2026-09-19. Point-in-time: describes the decision and the
target state; it is not kept in sync with later code.

## Decision

sous stops being an MCP server with a sandboxed local worker. It becomes one
thing: a local daemon that serves Claude Code's subagents from a local MLX
model and forwards everything else upstream unchanged. Claude Code executes
every tool under its own permission system. The worker, its task queue, its
eight-tool loop and the application-level sandbox (`toolexec.py`) are deleted
in one breaking release, 0.7.0, with no deprecation period.

Decisions taken during design, in order:

1. Remove outright in one release. No compatibility shim.
2. Keep the PyPI distribution name `sous-mcp` for now and explain it in the
   README; a PEP 541 request for the bare name `sous` is open
   (pypi/support#11881). The `sous-mcp` console script goes; `sous` stays.
3. The security stance is documentation only: `sous claude` sets no permission
   flags and prints no notice about permission modes.
4. Version 0.7.0. Tags only until 1.0, as before.
5. Removal plus consolidation: names and config are regularised so the product
   reads as one thing, not as a gateway bolted onto a daemon.

## Why

The MCP path existed so a model trusted less than a frontier model could act
on a repository behind a confinement layer sous owned. Two things changed:

- Claude Code's auto mode (GA on all plans, the default starting mode on Max)
  puts a frontier classifier between any subagent and the filesystem. Verified
  on the 2026-09-14 gate session: with a `sous-local` subagent in auto mode,
  each shell command, network operation, protected-path write and the
  subagent's spawn description and final report went to a Claude Sonnet 5
  classifier request through sous (1.2–1.8 s each); reads and
  working-directory edits were auto-approved without a request. The
  classifier sees the user's messages and the tool calls with tool results
  stripped; three consecutive or twenty total blocks fall back to prompting.
  Claude Code's optional Seatbelt sandbox adds an OS-level filesystem and
  network boundary that `toolexec.py` never had (no network sandbox, a `setsid`
  escape, deletions invisible to the audit — #36).
- Every measured gain since 0.5.0 landed on the endpoint: cross-session
  prompt-cache forks, retained turn slots, the observability surface,
  `sous tune`. Delegation through the queue only paid when the diff dwarfed
  the spec, and nothing has been promoted, so there is no installed base to
  carry.

Keeping a second, weaker boundary that no longer gates anything the product
recommends is cost without benefit.

## Scope

### Removed

| Area | What goes |
|---|---|
| Modules | `tasks.py`, `toolexec.py`, `worker.py`, `proxy.py`, `context.py` (helpers relocated, see Tune) |
| MCP surface | the `mcp` dependency, `MCPServer`, the instructions text, all six tools, the `/mcp/` redirect, the stdio bridge (`sous mcp`) |
| CLI | `sous wait`, `sous mcp`; task counts in `sous stop`; recent tasks in `sous status` |
| Config | `[gateway]` (keys move, see Config), `[budgets]`, `[commands]`, `[context]`, `[tasks]`; the worker meaning of `[model].max_context_tokens` |
| Status document | `queue`, `recent_tasks`, `config.max_turns`, `config.max_minutes`, `config.allowlist`, `config.context`, the nested `config.gateway` |
| Assets | `server.json`, `.github/workflows/publish-mcp-registry.yml`, `skills/delegating-to-local/`, `scripts/e2e_smoke.py`, the `mcp-name` comment in the README |
| Tests | `test_commands.py`, `test_confinement.py`, `test_tasks.py`, `test_worker.py`, `test_proxy.py`, `test_context.py`, `tests/daemon_stub.py`; worker cases in `test_cli.py`, `test_config.py`, `test_server.py`, `test_monitor.py`, `test_tui.py` |
| Support | Claude Desktop. The stdio bridge was its only route in, and the desktop app cannot point Claude Code at a local base URL. |

### Kept, renamed

| Before | After |
|---|---|
| `src/sous/gateway/` | `src/sous/api/` (logger `sous.api`) |
| `gateway/routes.py: Gateway`, `mount_gateway` | `api/routes.py: Endpoint`, `mount_endpoint` |
| `server.py: SousService` | `server.py: Daemon` |
| `config.py: GATEWAY_MIN_CONTEXT_TOKENS` | `MIN_CONTEXT_TOKENS` (still 48 × 1024) |
| `SousConfig.gateway_*` fields | `local_models`, `max_context_tokens`, `generation_timeout_minutes`, `upstream_url`; `gateway_enabled` deleted |
| `tests/test_gateway_*.py` | `tests/test_api_*.py` |
| `scripts/e2e_smoke.py` | `scripts/api_smoke.py` (rewritten, see Scripts) |

### Untouched

`engine/*` (algorithms, kernels, prompt cache, positions), `inflight.py`,
`loopback.py`, `logs.py`, `sse.py`, `monitor.py` beyond the dropped fields,
`tune/*` beyond the changes listed under Tune, `docs/superpowers/**`,
`THIRD_PARTY_NOTICES.md`, the four CI jobs and the PyPI publish job.

## Runtime

`create_server` builds a plain `starlette.applications.Starlette` with the
`/sous/*` monitor routes mounted before the `/v1/*` endpoint routes, both
behind the existing loopback guard, and the same lifespan as today: on
shutdown it awaits `Endpoint.aclose()` so the session thread reaches
`release_mlx_thread_state()` and the upstream connection pool closes. `serve`
hands that app to uvicorn with the existing `uvicorn_config` (access log off,
logger shapes pinned). `starlette`, `uvicorn`, `anyio` and `sse-starlette` are
already direct dependencies; `mcp` is removed and its transitive pins stop
mattering. `sous.logs` remains the only root handler; the pins on
`sse_starlette`, `httpx` and `httpcore` loggers stay because those libraries
still sit in the request path.

### Idle unload

The idle sweep moves from the worker loop (its only caller) into
`EngineManager`: `start_idle_sweep(interval_s=15)` starts one daemon thread,
`sous-idle-sweep`, that calls `unload_if_idle()` every 15 s until a stop
`Event` is set; `stop_idle_sweep()` sets the event and joins. The daemon's
lifespan starts it after the engine manager exists and stops it on shutdown.
The thread never loads a model, so it may touch mlx freely during an unload
and calls `release_mlx_thread_state()` once on the way out. Each tick is a
lock acquisition and a clock comparison, which retires the twice-a-second
SQLite poll (#103). `hold()`, `touch()`, `get()`, holder pruning and the
version counter keep their semantics: the sweep is the only thing that
changes cadence, not what "idle" means.

## Config

Two sections. `[model]` is the engine; `[server]` is the endpoint.

```toml
[model]
id = "mlx-community/Qwen3.8-27B-4bit"
idle_unload_minutes = 30
max_context_tokens = 131072        # the served window; floor 49152
# temperature, top_p, top_k, prompt_cache, prompt_cache_gb, speculative_draft_id,
# speculative_block_size and int8_prefill are unchanged

[server]
port = 8383
upstream_url = "https://api.anthropic.com"
local_models = ["sous-local"]
generation_timeout_minutes = 30
```

`[model].max_context_tokens` changes meaning: it was the worker's fixed window
(default 32768) and becomes the served window (default 131072), the value
`sous claude` exports as `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and the endpoint's
render cap. The existing floor check carries over, so a worker-era 32768, or
one written by an earlier `sous tune --quick`, is warned about and clamped to
49152 rather than honoured. Every other validation message names the new
section.

Obsolete keys produce one `warnings.warn` at load, listing each key and, where
one exists, its new home: `[gateway].enabled` (gone: the endpoint is always
on), `[gateway].local_models`, `[gateway].upstream_url`,
`[gateway].generation_timeout_minutes` (now `[server].*`),
`[gateway].max_context_tokens` (now `[model].max_context_tokens`, and the
warning states the value in effect), and the whole of `[budgets]`,
`[commands]`, `[context]`, `[tasks]`. The daemon starts regardless. Nothing
under `~/.sous/tasks/` or `~/.sous/tasks.db` is touched; the README's upgrade
note says they can be deleted.

`EngineManager`'s KV reserve is sized from `[model].max_context_tokens`
alone; the `max()` over two windows disappears.

## Status document and its readers

One shape everywhere, as before, minus the task fields:

- `engine` — unchanged.
- `inflight` — unchanged.
- `config` — flat: `model_id`, `idle_unload_minutes`, `port`, `local_models`,
  `max_context_tokens`, `upstream_url`, `generation_timeout_minutes`.
- `recent_turns` — HTTP only, unchanged.

`Daemon.status_version()` becomes the in-flight registry's version, the
engine version and the config file's mtime and size; the task store's version
is gone. `/sous/events` polls fast while a turn is in flight, a load or an
unload is under way, and slowly otherwise; the busy-only heartbeat rule is
unchanged.

Readers: `sous top` drops the task rows, `TASK_WORDS` and the
`TASKS ON RAIL · COOKING` line, and its render snapshot
(`tests/snapshots/sous-top-100x30.svg`) is regenerated once; everything on the
rail, the slip, the ticket and the rate panel already renders turns.
`sous statusline` reads only `engine` and `inflight` and does not change.
`sous status` becomes the plain-text status: daemon up or down, engine state
and holders, in-flight turns, and the last five turns; `--watch` still opens
the TUI. Exit codes keep their current meaning.

## CLI

Commands after: `serve`, `claude`, `status`, `top`, `statusline`, `stop`,
`install-launchd`, `uninstall-launchd`, `tune`. `sous claude` reads
`[server].local_models` and `[model].max_context_tokens`, keeps its hold,
its `/sous/status` restart check and its env line, and loses the branch that
refused to start when the gateway was disabled. `sous stop` reports nothing
about tasks. Only the `sous` console script remains in `pyproject.toml`.

## Tune

PR #107 (`sous tune` full mode) merges before this work starts; this branch
is cut from `main` afterwards. Changes:

- `tune/bench.py` gets its synthetic tool array from a new `tune/payload.py`
  instead of the worker's `WORKER_TOOLS`; the payload is a fixed array of
  eight tool schemas whose rendered token count a test pins, so bench numbers
  stay comparable across runs.
- `kv_bytes_per_token`, `native_max_tokens` and `TOKEN_STEP` move from
  `context.py` to `engine/window.py`; `tune/candidates.py` imports them from
  there.
- `tune/arms.py` and `tune/decide.py` propose one window,
  `[model].max_context_tokens`; the separate gateway window and the
  `gateway_enabled` branches go.
- `tune/daemon.py: ready_for_tune` is ready when `inflight` is empty and
  `engine.holders` is empty, instead of reading `queue`.
- `protocol.py` keeps `ToolSet`, `parse_tool_calls` and `ParseError` (the
  endpoint uses them) and drops `WORKER_TOOLS` and `WORKER_TOOLSET`.

## Tests

Deleted: the six worker-only files and `tests/daemon_stub.py`. Pruned: the
`wait` cases in `test_cli.py`; the allowlist, context and budgets cases in
`test_config.py`; the delegate, task status, result, cancel, respond and
recent-tasks cases in `test_server.py`; the two events-loop cases that wake on
a task change in `test_monitor.py`; any queue-line cases in `test_tui.py`.
Renamed: `test_gateway_*.py` to `test_api_*.py` with imports updated, no
behaviour change.

Added, each failing before its change:

- `EngineManager.start_idle_sweep` unloads an idle engine on the next tick and
  stops cleanly; the sweep thread releases its mlx state (a fake engine
  records the call).
- Obsolete config keys produce exactly one warning naming each key and its new
  home, and the daemon still builds.
- The status document has no `queue` or `recent_tasks` key and a flat
  `config`; `status_version` has three components.
- `sous status` prints engine state, holders, in-flight turns and recent turns
  against a stubbed `/sous/status`.
- The lifespan closes the endpoint on shutdown (existing gateway test, kept
  under the new name).
- `scripts/api_smoke.py`'s request builder produces a body the endpoint routes
  locally (pure function, no model).

The `model`-marked tests keep their markers and stay local-only; CI runs
`uv run pytest -m "not model"`, `uv run ty check`, ruff and `uv lock --check`
exactly as today.

## Scripts and CI

`scripts/api_smoke.py` starts `sous serve` on a free port with a temp config
and data dir pointing at the same tiny model `e2e_smoke.py` used, posts one
streaming `/v1/messages` request for `sous-local` carrying a one-tool array
and a prompt that asks for that tool, and passes when a `tool_use` content
block arrives before a timeout. It judges the block, not whether the model
finishes the turn, for the reason the old smoke judged `hello.txt` (#56
closes). `CLAUDE.md` names it as the post-dependency-change check: one real
served turn.

`server.json` and `publish-mcp-registry.yml` are deleted. `ci.yml` is
unchanged.

## Documentation and packaging

`README.md` is rewritten in this order: tagline; Why (plan economics, the
hybrid argument); How sous compares (full-local replacements, bare routers,
and what sous adds over a subagent delegate: cross-session prompt-cache forks,
retained turn slots, a byte-for-byte transparent proxy that stores no
credential, `sous top` and attributed turn lines, `sous tune`, holds and idle
unload); Requirements; Install (`uv tool install sous-mcp`, one paragraph on
the name and the PEP 541 request); Quick start (`sous serve` or launchd, then
`sous claude`); How it works (routing on the model id, forwarding, what a
locally served turn gives up); Prompt cache and continuity; Observability;
Configuration (`[model]`, `[server]`); Tuning; Smaller machines; Security
model; Validation status (the endpoint gate numbers already recorded in the
gateway PRs); Limitations; Upgrading from 0.6; Development.

The Security model section states: the boundary around the local model is
Claude Code's permission system; sous executes nothing and logs no request
body, header value or query string; it forwards the client's credentials to
`[server].upstream_url` and nowhere else and stores none. In auto mode, the
Max default, a classifier reviews the local subagent's shell commands,
network operations and protected-path writes, and its spawn description and
final report (Claude Sonnet 5, requested through sous, about 1.5 s each);
reads and working-directory edits are auto-approved; Claude Code's sandbox
adds an OS-level filesystem and network boundary; in Manual mode every
non-read action prompts, and under `-p` prompts become denials unless
`--permission-mode auto` is passed. Cited to the Claude Code docs pages on
permission modes and classifier billing, with the version those facts were
read at.

Upgrading from 0.6 lists: the MCP server and tools are gone, so
`claude mcp remove sous` and any `claude_desktop_config.json` entry; the
obsolete config keys and their new homes; `~/.sous/tasks/` and `tasks.db` can
be deleted; Claude Desktop is no longer supported.

`CLAUDE.md`: the description line becomes the new one-sentence product
statement; the `e2e_smoke.py` and `budget-exhausted` gotchas go; the Security
boundary section is rewritten around `src/sous/api/` (executes nothing, logs
nothing sensitive, forwards credentials only to `[server].upstream_url`;
`sous claude` never sets `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or a tier
variable), with the rule that no confinement is added to sous because the
boundary is Claude Code's; the commands list names `api_smoke.py`.

`CONTRIBUTING.md`: "Touching the sandbox" goes; "Verifying the endpoint" is
the smoke recipe; the scope note no longer mentions the task lifecycle or the
MCP surface.

`pyproject.toml`: description "The sous-chef for Claude's kitchen: serve
Claude Code's subagents from a local MLX model on Apple silicon, so heavy
Claude Code use stretches further on the same plan"; keywords drop `mcp`,
`mcp-server`, `model-context-protocol`, `delegation` and add `claude-code`,
`subagent`, `local-model`; the `mcp` dependency and the `sous-mcp` script go;
version 0.7.0 in the final commit.

## Sequencing and gate

1. Merge PR #107.
2. Branch from `main`. One PR, `refactor!: remove the MCP delegate path`,
   built as ordered commits that each leave the four CI jobs green, roughly:
   idle sweep into the engine; plain Starlette app and MCP removal; worker,
   tasks, sandbox, proxy and their tests deleted; config and status document
   reshaped; CLI surface; tune; package and test renames; smoke script;
   README, CLAUDE.md, CONTRIBUTING, pyproject; version bump. Code and docs
   travel together because the current README tells users to register an MCP
   server.
3. After merge: tag `v0.7.0`; reinstall the M5 Pro daemon from `main`; run
   `scripts/api_smoke.py` and one real `sous claude` session in auto mode
   whose subagent runs on `sous-local`, checking the turn lines and one
   classifier request in the daemon log.
4. Close #36, #56, #103 and #28 as moot. #40, #44, #39, #87 and #89 stay.

## Non-goals

No new endpoint features, no engine changes, no distribution rename, no
permission-mode handling in `sous claude`, no edits under
`docs/superpowers/**` other than adding this file, and no promotion copy.
