# sous

The sous-chef for Claude's kitchen: delegate mechanical, volume-heavy coding
tasks from Claude Code / Claude Desktop to a local MLX model on your Mac.
Claude designs the menu; sous does the prep — in a sandboxed, auditable,
autonomous tool loop. You (and Claude) review everything it cooks.

## Requirements

- Apple silicon Mac (tested on M-series with 64 GB; the default model needs
  ~30 GB free unified memory)
- Python 3.14 (standard build) via [uv](https://docs.astral.sh/uv/)
- Claude Code or Claude Desktop with a Pro/Max plan

## Install

```bash
uv tool install .          # from a checkout; PyPI/brew distribution planned
sous install-launchd       # start at login, keep alive (recommended)
claude mcp add --transport http sous http://127.0.0.1:8383/mcp
```

Optionally install the delegation skill so Claude knows when and how to use it:

```bash
cp -r skills/delegating-to-local ~/.claude/skills/
```

First delegation downloads the model (~28.7 GB for the default) — one time.

## What Claude gets

| Tool | Purpose |
|---|---|
| `delegate_task` | queue a self-contained task (returns immediately) |
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
             "cargo test", "cargo check", "make test"]
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

## Security model

- Workers are confined to the `project_root` of their task (symlink-resolved;
  `.git/` writes denied). The sous control directory (`~/.sous/` — config,
  allowlist, task db, transcripts) is never writable from inside the sandbox,
  and a `project_root` that contains it is rejected outright.
- No shell: commands run as argv (never through a shell), with an
  environment scrubbed down to `PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`,
  `TMPDIR` (anything else, including `*_TOKEN`/`*_KEY`/`*_SECRET`/`*_PASSWORD`
  vars, is stripped). Only allowlisted commands run without approval.
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
- Every worker turn is journaled to `~/.sous/tasks/<id>/transcript.jsonl`.
- The MCP endpoint binds to 127.0.0.1 only.

## Limitations

- MLX generation cannot be aborted mid-stream. Generations are serialized by
  a per-engine lock (and the engine is never idle-unloaded while one is in
  flight), so a truly wedged generation delays subsequent tasks until the
  daemon is restarted. Running the worker in a separate process (process
  isolation) is the future fix.

## Development

```bash
uv run pytest -m "not model"        # fast suite, no model needed (CI)
uv run pytest -m model              # engine tests, downloads tiny models
uv run python scripts/e2e_smoke.py  # full loop with a 0.6B model
```

Manual E2E with the real model: `sous serve`, register with `claude mcp add`,
then ask Claude to delegate something trivial and watch `sous status`.

Design spec: `docs/superpowers/specs/2026-08-14-sous-design.md`.
