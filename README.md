# sous

<!-- mcp-name: io.github.krcm0209/sous -->

The sous-chef for Claude's kitchen: delegate mechanical, volume-heavy coding
tasks from Claude Code / Claude Desktop to a local MLX model on your Mac.
Claude designs the menu; sous does the prep — in a sandboxed, auditable,
autonomous tool loop. You (and Claude) review everything it cooks.

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
  synchronously inside your session, and nothing sandboxes what the local
  model does to your files.
- **Subagent/skill delegates** hand tasks to a local model without a
  persistent queue, path confinement, command allowlisting, or an audit
  trail.

sous is the hybrid: Claude keeps the reasoning, and an asynchronous queue
hands the mechanical work to a local worker that is sandboxed, budgeted,
approval-gated, and journaled — with the diff always reviewed before it
counts.

## Requirements

- Apple silicon Mac (tested on M-series with 64 GB; the default model needs
  ~30 GB free unified memory — for 32 GB and 16 GB machines, see
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

First delegation downloads the model (~28.7 GB for the default) — one time.

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

## Configuration — `~/.sous/config.toml`

```toml
[server]
port = 8383

[model]
id = "mlx-community/Qwen3.8-27B-mxfp8"
idle_unload_minutes = 30
max_context_tokens = 32768
temperature = 0.7
top_p = 0.8
top_k = 20

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

[tasks]
retention = 200
```

Every value is optional; the allowlist is re-read on every command execution,
so edits apply instantly. Swap `[model].id` for any MLX text or vision model
(e.g. the `-8bit`/`-4bit` conversions, or a fast MoE coder via mlx-lm).

`temperature`/`top_p`/`top_k` control the worker's sampler (Qwen's own
documented non-thinking-mode defaults). Greedy decoding (temperature 0)
sounds safer but isn't: it gives the model no way to escape a bad
completion once it happens, since a near-identical prompt plus a nudge
still argmaxes to the same wrong output every time.

## Smaller machines

The default model wants a 64 GB machine. Both alternatives below share the
default's `qwen3_5` architecture, so they load through the exact same mlx-vlm
path — edit `[model].id` in `~/.sous/config.toml` and the next delegation
downloads and uses them.

| Unified memory | `[model].id` | Weights |
|---|---|---|
| 64 GB (default) | `mlx-community/Qwen3.8-27B-mxfp8` | ~28.7 GB |
| 32 GB | `mlx-community/Qwen3.8-27B-mxfp4` | ~15.2 GB |
| 16 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | ~6 GB |

The 32 GB pick is the same Qwen3.8-27B checkpoint at mxfp4 — same model at
half the footprint for a modest quality cost. The 16 GB pick drops to the 9B
tier because an 8-bit 9B (~11 GB) would crowd the ≈10.7 GB Metal working-set
limit of a 16 GB machine once the KV cache lands on top of the weights; on
16 GB, also consider `[model].max_context_tokens = 16384` if you see memory
pressure. (`mlx-community/Qwen3.5-9B-mxfp8`/`-mxfp4` look like the obvious
picks, but as of 2026-08 they are empty placeholder repos with no weights.)

Both alternatives passed the same worker-path validation as the default
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

## Validation status

Validated end to end against the default model
(`mlx-community/Qwen3.8-27B-mxfp8`) on an M5 Pro / 64 GB:

- **Worker path** — a delegated "add type hints and docstrings" task completed
  in 57s over 3 turns (`done` / `completed`). The worker edited the file, chose
  to run `pytest` to check itself, and reported accurately; an independent
  re-run of the tests confirmed it.
- **MCP surface** — driven by a real MCP client over streamable HTTP, the same
  path Claude Code uses: all six tools registered with the expected names, and
  a delegated task ran to `done` / `completed` in 42s with a correct diff and
  verify output.
- **Smaller-machine models** — both [Smaller machines](#smaller-machines)
  picks passed the same worker-path check on the same machine (delegated
  type-hints task, worker self-verified with `pytest`, independent re-run
  confirmed, accurate report): `Qwen3.8-27B-mxfp4` in 35s over 4 turns —
  confirming mlx-vlm loads mxfp4-mode quants — and `Qwen3.5-9B-MLX-4bit` in
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

## Development

Setup, the checks CI runs, and the pull-request process are in
[CONTRIBUTING.md](CONTRIBUTING.md).

Manual E2E with the real model: `sous serve`, register with `claude mcp add`,
then ask Claude to delegate something trivial and watch `sous status`.

Design spec: `docs/superpowers/specs/2026-08-14-sous-design.md`.
