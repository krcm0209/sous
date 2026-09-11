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
  own header fork. A `turn` slot is *moved* into the turn that extends it.
  Never rewind a cache to make a slot — a hybrid model's recurrent layers
  cannot.
  `base.measure_cache_budget`/`live_headroom` deliberately do not call
  `release_mlx_thread_state()`: they run on threads whose caches are live.
  The pressure valve reads Metal's headroom and the kernel's
  `kern.memorystatus_vm_pressure_level`, never psutil's free RAM: right after
  a model load the weight files sit in the page cache as "active", that
  figure read low, and the valve evicted the forks a cold turn had just made.
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
- `engine/int8prefill.py` is a runtime-compiled Metal kernel pair (Apache-2.0,
  derived from oMLX — see THIRD_PARTY_NOTICES.md). The Python `reorder_k` and
  the GEMM's nibble decode are two halves of one K-order contract (slot
  `16c+4t+j` holds code `16c+8*(t>>1)+2j+(t&1)`): never change one alone; the
  power-of-two bit-exactness test is what catches a mismatch. Only rows >= 128
  route (decode/verify never do), only tagged modules route (tags are set with
  `object.__setattr__` so they stay out of mlx's parameter tree), and only the
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
