# sous — Design Spec

*2026-08-14. Approved through collaborative brainstorming; supersedes the working draft.*

**sous** is the sous-chef for Claude's kitchen: a macOS MCP application that lets Claude Code / Claude Desktop (on a Pro/Max plan) delegate lower-priority, mechanical, or volume-heavy coding tasks to a locally hosted MLX model running on Apple silicon — so the Anthropic-hosted models keep their role as the primary reasoning agents. The directory is `local-mlx-mcp`; the package, CLI, and MCP server name are all `sous`. (If published to PyPI and `sous` is taken, the fallback package name is `sous-mcp`.)

## Goals

- Claude delegates self-contained coding tasks; a local model executes them autonomously in a tool-use loop and reports back.
- Delegation is asynchronous: Claude queues tasks, keeps working, and collects results.
- One shared daemon: the model loads once regardless of how many Claude sessions are open.
- The worker is sandboxed: path-confined file access plus an allowlist of verification commands, with a realtime human-approval escalation path.
- Claude always reviews delegated output; delegated code is a draft, not a merge.

## Non-goals (v1)

- Image/vision inputs for tasks (the default model is multimodal, but tasks are text-only).
- Parallel model execution (the GPU is single-tenant; the queue serializes).
- Network access for the worker.
- Non-macOS platforms.

## Core decisions

| Decision | Choice |
|---|---|
| Local model's role | **Autonomous worker** — runs its own tool-use loop, reports back |
| Worker capabilities | **Files + safe commands** — file tools plus allowlisted verify commands (tests, linters, formatters), with an approval escalation flow |
| Delegation style | **Async task queue** — `delegate_task` returns an ID immediately; Claude polls and collects |
| Model hosting | **Embedded** — the daemon loads the model in-process via MLX |
| Default model | **`mlx-community/Qwen3.8-27B-8bit`** (~29.5 GB). Configurable. True FP8 checkpoints (e.g. `Qwen/Qwen3.8-27B-FP8`) are CUDA-only; MLX's FP8-family equivalent is `mxfp8`, which mlx-community has not yet published for 3.8. The 8-bit pick is quality/size-equivalent; moving to `-mxfp8` later is an optional one-line config change. |
| Topology | **Single shared daemon**, MCP over streamable HTTP on 127.0.0.1 |
| MCP SDK | Official `modelcontextprotocol/python-sdk` **v2** — high-level `MCPServer` API (`mcp.server.mcpserver`), protocol revision 2026-07-28 |

## Architecture

A single Python daemon, **`sous`**, exposing MCP over **streamable HTTP bound to 127.0.0.1 only**. Claude Code registers it once with `claude mcp add --transport http`.

```
Claude Code ──MCP/HTTP──▶ ┌────────────────────────────────────┐
Claude Desktop ─────────▶ │  sous daemon                       │
                          │                                    │
                          │  MCP layer (SDK v2 MCPServer)      │
                          │      │                             │
                          │  Task manager ── SQLite            │
                          │      │        (~/.sous/)           │
                          │  Agent worker (loop + budgets)     │
                          │      │              │              │
                          │  MLX engine     Tool executor      │
                          │  (mlx-vlm /     (files + cmds,     │
                          │   mlx-lm)        path-confined)    │
                          └────────────────────────────────────┘
```

Five components, each independently testable:

1. **MCP layer** — official Python MCP SDK v2, high-level `MCPServer` API (v2's rename of FastMCP; *not* the third-party FastMCP package). Host/port/transport configured in `run()` per v2 conventions. Translates MCP tool calls into task-manager operations. Knows nothing about MLX.
2. **Task manager** — the async queue. Tasks persist to SQLite in `~/.sous/` so results survive daemon restarts. One worker consumes the queue serially; "parallelism" means Claude queues many tasks and they stream through one at a time.
3. **Agent worker** — the harness around the local model: builds the system prompt, runs the *generate → parse tool call → execute tool → append result* loop, enforces budgets, and produces a final structured report.
4. **Tool executor** — implements the worker's tools (read / write / edit / glob / grep + allowlisted commands), path-confined to the task's declared project root.
5. **MLX engine** — a thin abstraction with two backends, selected automatically from the downloaded model's `config.json` (a `vision_config`/multimodal `model_type` selects mlx-vlm; otherwise mlx-lm): **mlx-vlm** for multimodal models (required for the default Qwen3.8-27B, which mlx-lm can't load; used text-only in v1) and **mlx-lm** for text-only models (so switching to e.g. a fast MoE coder like Qwen3-Coder-30B just works). Lazy-loads the model on the first task; unloads after a configurable idle period (default 30 min) so the daemon idles at near-zero memory. Model ID is config, not code.

### Lifecycle — "what if the daemon isn't running?"

1. **launchd LaunchAgent** (via `sous install-launchd`) with `KeepAlive` — starts at login, restarts if it dies. Idle cost is near zero because the model unloads when unused.
2. **Companion Claude Code skill** — if the MCP server is unreachable, Claude runs `sous status`, boots it (`launchctl kickstart` or `sous serve`) from Bash, then retries the MCP call. Claude self-heals the connection instead of giving up.

CLI: `sous serve`, `sous status`, `sous install-launchd`.

## MCP tool surface

Six tools, written so their descriptions themselves steer Claude toward correct use (delegate mechanical/volume work; keep reasoning; make instructions self-contained because the worker has **no conversation context**):

| Tool | Arguments | Returns |
|---|---|---|
| `delegate_task` | `title`, `instructions` (self-contained: goal, constraints, acceptance criteria), `project_root` (absolute path — the confinement boundary), `context_files` (optional paths to read first), `verify_commands` (optional, must match allowlist) | `task_id`, queue position |
| `task_status` | `task_id` (optional — omit for all active/recent) | status, queue position, turns/time used, last activity (e.g. "editing src/foo.py"); for `awaiting_approval`, the exact command requested |
| `task_result` | `task_id`, `include_diff` (optional bool) | final report: what was done, files changed with per-file summaries, verify-command outputs (exit code + output tail), worker's notes/uncertainties, budget usage; unified diff if requested; transcript path |
| `cancel_task` | `task_id` | confirmation; a running task stops at the next tool boundary |
| `respond_to_command_request` | `task_id`, `approve` (bool), `persist_to_allowlist` (bool) | resolves an `awaiting_approval` task: run the command once, add it to the config allowlist permanently, or deny |
| `server_status` | — | model loaded? memory use, queue depth, config summary (model ID, budgets, allowlist) |

### Task lifecycle

`queued → running → awaiting_approval? → done | failed | cancelled`, persisted in SQLite.

**Realtime allowlist escalation:** the allowlist is re-read from `config.toml` on *every* command execution, so manual edits apply instantly — no watcher needed. When the worker requests a command that isn't allowlisted, the task pauses as `awaiting_approval`; `task_status` exposes the exact command; the skill instructs Claude to surface it to the user and answer via `respond_to_command_request`. Unanswered requests time out (default 10 min) as a denial and the worker continues without the command. Trade-off: a paused task holds the single worker slot, so the queue waits behind it — bounded by the timeout.

**Auditability:** the full worker transcript (every turn, tool call, and result) is saved to `~/.sous/tasks/<id>/transcript.jsonl`; `task_result` includes the path so Claude (or the user) can audit exactly what the worker did. Old tasks are pruned past a configurable retention count (default 200).

**File-change tracking:** the tool executor records every write/edit (path + before/after content hash), so `task_result` reports precise changes without requiring git. When `project_root` is a git repo, the diff comes from `git diff` for familiar output.

**Progress:** polling via `task_status` — deliberately no push/progress-notification machinery (SDK v2 removed the experimental Tasks API; polling is simpler and fits Claude's tool-call model).

**Queueing:** FIFO, single worker. Claude can queue many tasks; `task_status` shows positions so it can keep doing its own work while the queue drains.

**Native task-list mirroring:** these tools are *analogous to*, not part of, Claude Code's internal workflow/task system — MCP servers can't inject into that UI. The companion skill instructs Claude to mirror every delegated task into its native task list: create a Claude Code task on `delegate_task`, update it while polling `task_status`, complete it on `task_result`. Delegated work then shows up in the normal Claude Code progress UI.

## Worker agent loop

For each task, the agent worker builds a fresh conversation: a system prompt (role: focused coding subcontractor; make minimal changes; follow existing style; report honestly), the task `instructions`, a top-level directory listing of `project_root`, and — before the loop starts — the harness pre-reads any `context_files` into the transcript so the model doesn't burn turns fetching them.

Then: **generate → parse tool calls → execute → append results → repeat.** Tool calls use the model's native tool-call format via its chat template; the harness parses them itself (no OpenAI-compat layer to trust). A malformed call gets an error + format reminder back; repeated malformed calls (default 3 consecutive) fail the task cleanly.

**Worker's tools:** `read_file`, `write_file`, `edit_file` (exact-match replace), `list_dir`, `glob`, `grep`, `run_command` (allowlist only), and `finish(report)` — the explicit "I'm done" tool whose argument becomes the task report.

**Budgets** (all config, defaults shown): max 40 turns, 15 min wall-clock, 4,096 tokens per generation, and a **32k-token context cap** — the model's native window is 262k, but KV cache at that length would blow past 64 GB RAM next to 29.5 GB of weights. Near the cap, oldest tool results are elided with a placeholder note. Hitting any budget ends the task as `done (budget-exhausted)` with a partial report — never a hang.

## Safety & confinement

- **Path confinement:** every path is symlink-resolved and must land inside `project_root`; writes into `.git/` are denied (prevents hook injection); reads outside the root are denied too.
- **Approval flow, worker's view:** a non-allowlisted `run_command` call simply blocks until it resolves — the worker eventually receives either the command output (approved) or "denied — proceed without it." The pause/escalation machinery is invisible to the model; no extra tools for it to misuse.
- **Command execution:** no shell — argv built with shlex, `shell=False`, so `;`, `|`, `$()`, backticks, and redirects are inert. Commands must match an allowlist of prefixes, where matching means the command's leading argv tokens equal an allowlist entry's tokens exactly (`["npm", "test"]` matches `npm test --workspaces` but not `npm testx` or `npm install`); defaults below, editable in config. Run with `cwd=project_root`, 120 s timeout, output truncated to 16 KB, environment scrubbed to a minimal PATH (no `*_TOKEN`/`*_KEY`/`*_SECRET` vars leak from the daemon).
- **No network:** the worker has no fetch/URL tools.
- **Prompt-injection stance:** file contents the worker reads are data, and a 27B local model can't be fully trusted to remember that — so the real defense is the sandbox: confinement + allowlist + budgets bound the blast radius to "edited files inside one project root," which Claude then reviews.
- **Concurrency guard:** one task runs at a time globally; the skill tells Claude not to edit files a running task is touching (file-change hashes in `task_result` let Claude detect conflicts after the fact).

## Configuration (`~/.sous/config.toml`)

```toml
[server]
port = 8383                     # streamable HTTP, 127.0.0.1 only

[model]
id = "mlx-community/Qwen3.8-27B-8bit"
idle_unload_minutes = 30
max_context_tokens = 32768

[budgets]
max_turns = 40
max_minutes = 15
max_tokens_per_generation = 4096

[commands]
allowlist = ["pytest", "python -m pytest", "npm test", "npx eslint",
             "npx prettier", "ruff", "black", "mypy", "go test",
             "cargo test", "cargo check", "make test"]
timeout_seconds = 120
approval_timeout_minutes = 10   # awaiting_approval → auto-deny

[tasks]
retention = 200
```

Missing file → defaults; unknown keys → warn, don't crash. The `[commands] allowlist` is re-read on every command execution (realtime updates); other sections are read at startup.

## Error handling

| Failure | Behavior |
|---|---|
| Model load fails (OOM, missing repo, disk) | Task → `failed` with the real error; `server_status` reports engine state |
| Daemon dies mid-task | On restart, `running` tasks → `failed (interrupted)`, transcript preserved |
| Generation stall | Per-generation timeout fails the task cleanly |
| 3 consecutive malformed tool calls | Task → `failed (model-confused)` with transcript |
| HF download interrupted | huggingface_hub resume; clear progress/error via `server_status` |
| Any MCP-layer exception | Structured tool error to Claude; daemon never crashes |

Budget exhaustion is **not** a failure — it's `done (budget-exhausted)` with a partial report.

## Testing strategy

- **Unit** (no model): tool executor — path confinement is the security-critical core, so symlink-escape, `../`, absolute-path, and `.git/`-write cases get exhaustive tests; allowlist matching; env scrubbing; tool-call parser (well-formed + malformed); task manager state transitions and restart persistence; approval-flow state machine (approve / deny / persist / timeout).
- **Integration** (no model): the agent loop against a **fake engine** with scripted responses — verifies loop mechanics, budgets, `finish`, malformed-call handling, and the awaiting-approval pause/resume. Runs in CI on any machine.
- **E2E** (local, manual): the engine abstraction lets smoke tests use a tiny real model (~0.5 GB 4-bit) to prove real generation + tool-call parsing; the full Qwen3.8-27B path is a manual checklist.
- TDD throughout, per the superpowers workflow.

## The delegation skill (ships in this repo)

A Claude Code skill, `skills/delegating-to-local/SKILL.md`, that teaches Claude:

- **When to delegate:** mechanical, repetitive, volume-heavy, low-risk work — boilerplate, test scaffolding, bulk renames/migrations, docstrings, lint-fix sweeps, generating fixtures.
- **When NOT to:** architecture, tricky debugging, security-sensitive code, anything needing conversation context or taste.
- **How to write instructions:** self-contained (the worker sees nothing of the chat), explicit acceptance criteria, `verify_commands` from the allowlist.
- **The workflow:** queue → mirror into the native Claude Code task list → continue own work → poll (surfacing any `awaiting_approval` command requests to the user) → collect → **always review the diff** before accepting.
- **Self-healing:** if the MCP server is unreachable — `sous status`, then `launchctl kickstart` or `sous serve` via Bash, then retry.

## Repo layout

```
local-mlx-mcp/
├── pyproject.toml            # uv-managed, Python 3.14 (mlx ships cp314 arm64 wheels; MCP SDK v2 supports 3.10–3.14; standard build, not free-threaded 3.14t)
├── src/sous/
│   ├── cli.py                # serve / status / install-launchd
│   ├── server.py             # MCP layer (SDK v2 MCPServer)
│   ├── tasks.py              # queue + SQLite persistence
│   ├── worker.py             # agent loop + budgets
│   ├── toolexec.py           # worker tools, confinement, allowlist
│   ├── config.py
│   └── engine/               # base protocol + vlm.py / lm.py backends
├── skills/delegating-to-local/SKILL.md
├── tests/
└── docs/superpowers/specs/
```

## Future work (explicitly out of v1)

- Switch default to `mlx-community/Qwen3.8-27B-mxfp8` when published.
- Vision-input tasks (screenshot-driven UI work) via the already-multimodal default model.
- Multiple concurrent workers if a smaller/faster model makes batching worthwhile.
