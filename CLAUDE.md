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

## Security boundary

`src/sous/toolexec.py` is the sandbox (path confinement, command allowlist,
process-group kill, stat audit). Any change there needs a test that fails
without it. Odd-looking code is load-bearing (the un-reaped zombie during
the group kill, EPERM suppression, ctime in the audit) — read the comments
before touching. Suspected-flaky tests get run in a loop, not judged on one
pass.

## Workflow

- Comments explain non-obvious *why*; never restate what code does.
- Conventional Commits (`feat:`/`fix:`/`docs:`/...), imperative lowercase
  subject, *why* in the body.
- `main` is protected: branch + PR, all four CI jobs green.
