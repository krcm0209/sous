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
  after dependency changes, verify with one real delegated task.
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

## Security boundary

`src/sous/toolexec.py` is the sandbox (path confinement, command allowlist,
process-group kill, stat audit). Any change there needs a test that fails
without it. Odd-looking code is load-bearing (the un-reaped zombie during
the group kill, EPERM suppression, ctime in the audit) — read the comments
before touching. Suspected-flaky tests get run in a loop, not judged on one
pass.

`src/sous/gateway/` is deliberately outside that boundary: it never executes a
tool (Claude Code does, under its own permissions) and never logs a request
body, header value or query string. It forwards the client's credentials to
`[gateway].upstream_url` and nowhere else, and stores none. `sous claude`
(`cli.py`) never sets `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or a tier
variable. A change that makes any of these otherwise needs the spec
(`docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`) changed first.

## Workflow

- Comments explain non-obvious *why*; never restate what code does.
- Conventional Commits (`feat:`/`fix:`/`docs:`/...), imperative lowercase
  subject, *why* in the body.
- `main` is protected: branch + PR, all four CI jobs green.
