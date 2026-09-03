# sous

<!-- mcp-name: io.github.krcm0209/sous -->

The sous-chef for Claude's kitchen: delegate mechanical, volume-heavy coding
tasks from Claude Code / Claude Desktop to a local MLX model on your Mac.
Claude designs the menu; sous does the prep — in a sandboxed, auditable,
autonomous tool loop. The sandbox is application-level, not an OS jail
([what that means](#security-model)). You (and Claude) review everything it
cooks.

## Why

Heavy Claude Code use runs into plan usage limits, and most of what consumes
them is generated output. sous exists to stretch that budget: the mechanical,
volume-heavy output comes from a local model at zero marginal cost, while
Claude spends its much cheaper input-side attention writing instructions and
reviewing the resulting diff. Claude stays the head chef, and the same plan
carries further into the week.

This is a hybrid local + cloud arrangement, deliberately not an
all-or-nothing switch to a local model: pointing Claude Code itself at a
local endpoint trades away the frontier reasoning you are paying for, while
sous offloads only the small, mechanical work your Apple silicon Mac can
handle on its own.

## How sous compares

Other ways to put local models next to Claude Code make a different trade:

- **Full-local replacements** point Claude Code (or a fork of it) at a local
  endpoint — every task drops to local-model quality, including the ones you
  wanted a frontier model for.
- **Routers/proxies** swap models per request, but the work still runs
  synchronously inside your session, and the only thing between the local
  model and your files is a permission ruleset calibrated for a frontier
  model's judgement.
- **Subagent/skill delegates** hand tasks to a local model with your
  session's permissions and none of the rest: no persistent queue, no
  project-root confinement, no allowlist scoped to a model you trust less,
  no audit trail.

sous is the hybrid: Claude keeps the reasoning, and an asynchronous queue
hands the mechanical work to a local worker that is sandboxed, budgeted,
approval-gated, and journaled — with the diff always reviewed before it
counts.

The cost is capability, and it is deliberate. The worker runs its own loop
over eight fixed tools (`read_file`, `write_file`, `edit_file`, `list_dir`,
`glob`, `grep`, `run_command`, `finish`); it cannot reach your MCP servers,
your skills, or your hooks, all of which a subagent delegate inherits. sous
buys confinement, persistence, and a reviewable diff by giving the local
model a much smaller world to work in. Work that genuinely needs the harness
is work to keep in Claude.

## Requirements

- Apple silicon Mac (tested on M-series with 64 GB; the default model needs
  ~18 GB free unified memory — for 16 GB machines, see
  [Smaller machines](#smaller-machines))
- Python 3.14 (standard build) via [uv](https://docs.astral.sh/uv/)
- Claude Code or Claude Desktop with a Pro/Max plan

## Install

```bash
uv tool install sous-mcp   # PyPI package name; the CLI it installs is `sous`
sous install-launchd       # start at login, keep alive (recommended)
claude mcp add --transport http sous http://127.0.0.1:8383/mcp
```

From a checkout, `uv tool install .` works instead of the PyPI package.
(The bare `sous` name on PyPI is an unrelated, abandoned placeholder — the
package you want is `sous-mcp`.)

### Claude Desktop

`claude_desktop_config.json` launches MCP servers as stdio subprocesses, so it
cannot take the HTTP URL above. Use `sous mcp`, which bridges stdio to the
daemon:

```json
{
  "mcpServers": {
    "sous": { "command": "sous", "args": ["mcp"] }
  }
}
```

`sous mcp` holds no state and loads no model — it forwards messages to the one
daemon. Open several clients and they share it, so the model is resident once
no matter how many are connected. If no daemon is running it starts one, which
is what makes this work without `sous install-launchd`.

Discovery needs no extra setup: the daemon publishes MCP server instructions
that clients put in front of Claude, saying when to delegate and why.
Optionally install the delegation skill as a supplement — a fuller playbook
(mirroring delegations into Claude's task list, approval etiquette,
restarting a downed daemon):

```bash
cp -r skills/delegating-to-local ~/.claude/skills/
```

First delegation downloads the model (~16.1 GB for the default) — one time.

### Managing the daemon

```bash
sous status             # is it up, and what has it been doing
sous wait <task-id>     # block until a task finishes or needs approval
sous stop               # stop it (see below)
sous uninstall-launchd  # stop it starting at login, and remove the agent
```

`sous stop` deliberately refuses when launchd is managing the daemon, because
`KeepAlive` would restart it a second later and the command would look like it
did nothing. It tells you which command you actually want. Stopping is for
daemons nothing is supervising — including one `sous mcp` started for you.

Stopping the daemon also ends any running `sous mcp` bridges; their clients
reconnect and start a fresh one on the next call.

## What Claude gets

| Tool | Purpose |
|---|---|
| `delegate_to_local_model` | queue a self-contained task (returns immediately) |
| `task_status` | poll progress / queue position / approval requests |
| `task_result` | fetch report, changed files, verify output, diff |
| `cancel_task` | stop a queued or running task |
| `respond_to_command_request` | approve/deny a non-allowlisted command |
| `server_status` | model + queue + config health |

## Gateway mode (experimental)

sous can also stand in for `api.anthropic.com`: with `[gateway].enabled =
true` the daemon serves `POST /v1/messages` (streaming and not),
`POST /v1/messages/count_tokens` and `HEAD /api/hello` on the same
`127.0.0.1:8383`, answering requests for the model id `sous-local` with the
local model. Claude Code executes the tool calls itself, with its usual
permission prompts; the model only decides what to call.

This is the first half of the hybrid design in issue #41. The intended mode
is a frontier main loop with local subagents — `CLAUDE_CODE_SUBAGENT_MODEL=sous-local`,
tier variables left alone, so a frontier model always reviews the local
model's work — and it needs the routing half and the `sous claude` launcher,
which come next. Until then a Claude Code session pointed at the gateway runs
*entirely* on the local model: exactly the trade "Why" and "How sous
compares" above argue against, supported only as a way to exercise the
endpoint, not as a mode. The recipe for that lives in
[CONTRIBUTING.md](CONTRIBUTING.md#verifying-the-gateway-endpoint).

No gateway response tells Claude Code how large the window is, and `sous-local`
is not an id it has a built-in size for, so it falls back to its own default
unless `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is set (it honours that variable only
for non-`claude-*` ids — one reason the served id is honest). Set it (and
`CLAUDE_CODE_AUTO_COMPACT_WINDOW`) to `[gateway].max_context_tokens` in the
environment that launches Claude Code, or requests eventually exceed the
server limit and return `400 invalid_request_error: prompt is too long`. The
planned `sous claude` launcher (Phase 2) will set these automatically.

What a locally served turn gives up, stated plainly:

- **No sandbox.** The gateway returns `tool_use` blocks and never runs a
  tool; `toolexec.py` (path confinement, allowlist, audit) is not in this
  path. Claude Code's own permission system is the boundary, and a 27B model
  inherits whatever permissiveness you configured for frontier subagents.
- **No Anthropic server-side or built-in tool types (no client-supplied
  schema).** `WebSearch`, `WebFetch`-as-server-tool and code execution run
  inside Anthropic's API; `bash_*`, `text_editor_*` and the other built-ins
  run on the client but arrive with their schema implied by the type. Both are
  dropped from the request (logged as `dropped N tool(s) with no
  client-supplied schema`) — the local chat template can only offer a tool it
  has an explicit schema for. Claude Code sends custom-typed equivalents when
  it drives a non-claude model, so a delegated turn keeps its file and shell
  tools.
- **No thinking, no request-level sampling.** `thinking`, `temperature`,
  `top_p`, `top_k`, `stop_sequences` and `tool_choice` are accepted and
  ignored; the daemon's `[model]` sampler applies. Images and documents in
  messages become a one-line `[image omitted: sous serves text only]`
  placeholder.
- **One turn at a time, one cache slot.** Turns are serialized behind the
  same lock as delegated tasks, and the prompt cache has one slot: a
  subagent's consecutive turns reuse it (turn 2 of a 57k-token prompt
  prefills only the new tokens), but while a delegated task and a gateway
  conversation overlap — or two conversations interleave — every turn on
  both sides is a cold prefill, because each evicts the other. Do not run
  both at once until keyed cache slots land.
- **A client that disconnects does not stop the model.** The turn runs to
  completion (so the next request never waits on a wedged lock); aborting
  mid-generation comes with batching, later.

Each `/v1/messages` turn logs one metadata-only line to the daemon's
stderr — method, model, stream flag, status, token counts, stop reason,
cache hit/miss, seconds — plus one line naming the Anthropic tool *types*
it dropped, when any. `/api/hello` and `count_tokens` add nothing beyond
uvicorn's access log; no request body or header value is ever logged.
Errors are Anthropic-shaped
(`{"type": "error", "error": {"type": ..., "message": ...}}`); an unknown
model id is a `404 not_found_error`, an oversized body (over 32 MiB) a
`413 request_too_large`, and a prompt that fills the window an
`invalid_request_error` saying `prompt is too long`.

## Configuration — `~/.sous/config.toml`

```toml
[server]
port = 8383

[model]
id = "mlx-community/Qwen3.8-27B-4bit"
idle_unload_minutes = 30
max_context_tokens = 32768
prompt_cache = true
temperature = 0.7
top_p = 0.8
top_k = 20
# Speculative decoding: ~1.8x decode on the default model with the shipped
# sampling, up to ~2.4x greedy. "" disables it; block size 0 lets the
# drafter's own policy pick the depth. Auto-disables with a warning when
# the drafter can't serve the configured model.
speculative_draft_id = "z-lab/Qwen3.8-27B-DFlash2"
speculative_block_size = 0

[budgets]
max_turns = 40
max_minutes = 15
max_tokens_per_generation = 4096

[commands]
allowlist = ["pytest", "python -m pytest", "npm test", "npx eslint",
             "npx prettier", "ruff", "black", "mypy", "go test",
             "cargo test", "cargo check", "make test", "uv run pytest",
             "uv run python -m pytest", "uv run ruff", "uv run black",
             "uv run mypy", "uv run ty"]
timeout_seconds = 120
approval_timeout_minutes = 10

[context]
mode = "fixed"     # "auto": size the window per task from free memory
fraction = 0.8     # auto: share of remaining memory headroom the KV cache may use
min_tokens = 8192  # auto: never shrink the window below this

[tasks]
retention = 200

[gateway]
# EXPERIMENTAL — see "Gateway mode" above. Serve Claude Code from the local
# model over an Anthropic-compatible /v1/messages on the same port.
enabled = false
local_models = ["sous-local"]   # model ids served locally; anything else is a 404 for now.
                                # Never claude-*: Claude Code ignores its context-window
                                # env vars for those ids (the config rejects them).
max_context_tokens = 65536      # server-side limit on prompt + reply tokens; positive
                                # values below 49152 are raised to it. Claude Code does NOT
                                # learn this from the gateway: set CLAUDE_CODE_MAX_CONTEXT_TOKENS
                                # (and CLAUDE_CODE_AUTO_COMPACT_WINDOW) to the same value in
                                # its environment, or a long conversation grows past it and
                                # fails with "prompt is too long".
generation_timeout_minutes = 30
```

Every value is optional; the allowlist is re-read on every command execution,
so edits apply instantly. Swap `[model].id` for any MLX text or vision model
(e.g. the `-8bit`/`-4bit` conversions, or a fast MoE coder via mlx-lm).

`[context] mode = "auto"` sizes the worker's context window per task instead
of using the fixed `[model].max_context_tokens`: when a task starts, sous
measures the remaining memory headroom (the tighter of the Metal working-set
ceiling and available system RAM), lets the KV cache have `fraction` of it,
and clamps the result between `min_tokens` and the model's native maximum.
The window is a cap, not a reservation. With cache reuse on (the shipped
default), the KV cache lives for the whole task, so `fraction` bounds
sustained residency, and a `run_command` subprocess competes with a live
cache that used to be freed between turns. With
`[model].prompt_cache = false` it bounds only a per-generation peak.
Residency still tracks the tokens a task actually uses, not the window.
Every task's report records the window it ran with and why
(`budget.context_tokens` / `budget.context_reason`). The default model's
hybrid attention makes context unusually cheap (only 16 of its 64 layers
accumulate KV — about 64 KiB per token), so an otherwise-idle 64 GB machine
gets the full native 262k window. If sizing fails for any reason, the task
runs with the fixed `max_context_tokens` and a warning.
Cache reuse pays most in `auto` mode: since elision is the only thing that
discards the cache, and elision fires only when the prompt exceeds the window,
a window the task never reaches means the cache survives the whole task. The
shipped default is `fixed` at 32768 tokens.

`[model].prompt_cache` (default `true`) reuses one KV cache across the turns
of a task, prefilling only what the conversation gained instead of the whole
thing every turn. All of a task's generations run on one worker-owned thread
so the cache survives between turns; measured on the default model in one
process, six growing turns took 29.5s warm against 77s cold, with per-turn
time flat instead of growing. Set it to `false` to prefill every turn from
scratch.

`temperature`/`top_p`/`top_k` control the worker's sampler (Qwen's own
documented non-thinking-mode defaults). Greedy decoding (temperature 0)
sounds safer but isn't: it gives the model no way to escape a bad
completion once it happens, since a near-identical prompt plus a nudge
still argmaxes to the same wrong output every time.

## Smaller machines

The default model fits 64 GB and 32 GB machines. The alternative below shares
the default's `qwen3_5` architecture, so it loads through the exact same
mlx-vlm path — edit `[model].id` in `~/.sous/config.toml` and the next
delegation downloads and uses it.

| Unified memory | `[model].id` | Weights |
|---|---|---|
| 64 GB / 32 GB (default) | `mlx-community/Qwen3.8-27B-4bit` | ~16.1 GB |
| 16 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | ~6 GB |

The default is the affine 4-bit Qwen3.8-27B: half the footprint of the
8-bit-class quants with no measured tool-loop quality loss. It also keeps
mlx-vlm's speculative-decoding fast path available, which requires affine
quantization (4-, 5-, or 8-bit); mxfp quants fall into a much slower
per-token verify fallback (krcm0209/sous#58 has the measurements behind
both claims). The 16 GB pick
drops to the 9B tier because an 8-bit 9B (~11 GB) would crowd the ≈10.7 GB
Metal working-set limit of a 16 GB machine once the KV cache lands on top of
the weights; on 16 GB, also consider `[model].max_context_tokens = 16384` if
you see memory pressure. (`mlx-community/Qwen3.5-9B-mxfp8`/`-mxfp4` look
like the obvious picks, but as of 2026-08 they are empty placeholder repos
with no weights.)

The 16 GB alternative passed the same worker-path validation as the default
(see [Validation status](#validation-status)), run on the 64 GB test
machine — which validates the models and quants through sous's whole stack,
not the memory fit on physical 32 GB / 16 GB hardware (that remains
arithmetic: weights plus KV-cache headroom). Smaller workers still fail more
tasks in general and make reviewing the diff matter more — but a reviewed
draft from a small local model costs your plan nothing.

## Security model

- Workers are confined to the `project_root` of their task (symlink-resolved;
  `.git/` writes denied). The sous control directory (`~/.sous/` — config,
  allowlist, task db, transcripts) is never writable from inside the sandbox,
  and a `project_root` that contains it is rejected outright.
- No shell: commands run as argv (never through a shell), with an
  environment scrubbed down to `PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`,
  `TMPDIR` (anything else, including `*_TOKEN`/`*_KEY`/`*_SECRET`/`*_PASSWORD`
  vars, is stripped). Only allowlisted commands run without approval.
  `PATH` itself is adopted from your login shell once at daemon startup, so
  allowlisted commands resolve exactly as they do in your terminal no matter
  how the daemon was launched (launchd starts agents with the bare system
  `PATH`, which would otherwise turn every `uv run ...` into an approval).
- Non-allowlisted commands pause the task for explicit human approval
  (auto-deny after `approval_timeout_minutes`).
- Path confinement bounds the worker's *edits*, not what an allowlisted
  command can do. Allowlisting a command that executes repo-resident code —
  any test runner (`pytest`, `npm test`, `make test`, ...) — is equivalent
  to granting arbitrary local code execution over code the worker just
  wrote: the worker can write a `conftest.py` or test file that does
  anything the command's process can (network egress, reading any
  user-readable file), and the allowlisted verify run executes it without
  approval. Calibrate the allowlist accordingly, and review diffs before
  trusting verify output.
- There is no network sandbox: an allowlisted or human-approved command can
  still reach the network if the command itself does (e.g. `npm test`
  hitting a registry). Keep the allowlist narrow and review approval
  requests before saying yes.
- Command timeouts kill the command's whole process group (SIGTERM, short
  grace, SIGKILL) before file changes are audited — but a descendant that
  double-forks and calls `setsid()` escapes into a new session and survives
  the group kill. Closing that residual requires cgroup/OS-level confinement
  that macOS does not offer.
- The before/after file audit around each command is stat-based:
  `(mtime_ns, size, ctime_ns)` per file. mtime and size alone are forgeable
  by code the command executes (equal-length rewrite + `os.utime` restore);
  ctime is what makes the audit tamper-resistant, because no userspace API
  can set it and `os.utime` itself bumps it. That resistance has limits: a
  process running as root (e.g. via a mount trick or raw-device write) or
  manipulation of the system clock between the two snapshots could still
  hide a change. The audit is a safety net against the sandboxed worker and
  the code it runs — not against a privileged attacker.
- Every worker turn is journaled to `~/.sous/tasks/<id>/transcript.jsonl`.
- The MCP endpoint binds to 127.0.0.1 only.
- **Gateway mode bypasses the sandbox by design.** A locally served Claude
  Code turn never touches `toolexec.py`: the gateway hands `tool_use` blocks
  back and Claude Code executes them under its own permission rules. The
  gateway binds to `127.0.0.1` only, stores no credential, and never logs
  request bodies or header values — including the `Authorization` header
  Claude Code sends with every request. It is off by default.

## Validation status

Validated end to end on an M5 Pro / 64 GB (originally against
`mlx-community/Qwen3.8-27B-mxfp8`, which remains a supported `[model].id`):

- **Worker path** — a delegated "add type hints and docstrings" task completed
  in 57s over 3 turns (`done` / `completed`). The worker edited the file, chose
  to run `pytest` to check itself, and reported accurately; an independent
  re-run of the tests confirmed it.
- **MCP surface** — driven by a real MCP client over streamable HTTP, the same
  path Claude Code uses: all six tools registered with the expected names, and
  a delegated task ran to `done` / `completed` in 42s with a correct diff and
  verify output.
- **Current default (`Qwen3.8-27B-4bit`)** — three delegated tasks
  (module-from-spec, docstring sweep, test scaffolding) through the real
  worker loop on 2026-08-29, all `done` / `completed` in 4/7/3 turns; every
  artifact passed independent grading, including hidden spec tests
  (krcm0209/sous#58).
- **Other models through the same stack** — the same worker-path check also
  passed on `Qwen3.8-27B-mxfp4` in 35s over 4 turns — confirming mlx-vlm
  loads mxfp4-mode quants — and on the 16 GB pick `Qwen3.5-9B-MLX-4bit` in
  25s over 8 turns.

Tool-call parsing accepts both the XML-ish `<function=…>/<parameter=…>` format
that Qwen3 emits and the hermes JSON format used by other MLX models.

## Limitations

- MLX generation cannot be aborted mid-stream. Generations are serialized by
  a per-engine lock (and the engine is never idle-unloaded while one is in
  flight), so a truly wedged generation delays subsequent tasks until the
  daemon is restarted. Running the worker in a separate process (process
  isolation) is the future fix.
- `scripts/e2e_smoke.py` uses a 0.6B model so it stays cheap to run. That model
  is too small to reliably emit a `finish` call, so the script usually ends
  `failed` or `budget-exhausted` even when it writes the right file — it
  exercises the plumbing, not model competence. The real-model runs above are
  the meaningful end-to-end evidence.
- Gateway mode (experimental) serves one turn at a time on a single-slot
  prompt cache, drops Anthropic server-side and built-in tool types (the ones
  that carry no client-supplied schema), ignores request-level sampling and
  thinking, and finishes a turn even after the client hangs up.
  It serves the model ids in `[gateway].local_models` locally and answers
  every other id with `404 not_found_error`; routing those upstream (the
  hybrid in issue #41) is not built yet.

## Development

Setup, the checks CI runs, and the pull-request process are in
[CONTRIBUTING.md](CONTRIBUTING.md).

Manual E2E with the real model: `sous serve`, register with `claude mcp add`,
then ask Claude to delegate something trivial and watch `sous status`.

Design spec: `docs/superpowers/specs/2026-08-14-sous-design.md`.
