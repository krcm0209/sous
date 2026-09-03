# Contributing to sous

Thanks for looking. Contributions are welcome — bug reports especially.

A few things worth knowing before you spend time on this.

## Before you start

**You need an Apple silicon Mac.** This is not a preference. `mlx-metal` is
`sys_platform == 'darwin'` gated, the worker runs on Metal, and CI itself runs
on macOS ARM runners. On any other machine you will not be able to run the
test suite, so there is no practical way to verify a change. Python 3.14 is
also required, though uv installs that for you.

**This is a single-maintainer project.** Reviews may take a while. If you are
planning anything beyond a focused fix, open an issue first — it is no fun to
write a large PR and then discover it conflicts with where the project is
going.

## What's in scope

Good candidates: bug reports with a reproduction, focused fixes, clearer
documentation, additional test coverage — particularly around the sandbox.

Please open an issue before starting on: new tools exposed to the worker,
changes to the task lifecycle or MCP surface, or anything that widens what a
worker is permitted to do.

sous deliberately targets Apple silicon. Porting it to Linux or CUDA is not a
small PR, and is not currently a goal.

## Setup

```bash
uv sync
```

That is the whole thing. uv resolves the interpreter and every dependency from
`uv.lock`.

## Checks before you push

CI runs four jobs. You can reproduce all of them locally:

```bash
uv run pytest -m "not model"                        # Tests
uv run ty check                                     # Type check
uv run ruff check . && uv run ruff format --check . # Lint
uv lock --check                                     # Lockfile in sync
```

`ruff format .` (without `--check`) applies the formatting rather than just
reporting it.

Two suites do **not** run in CI, and are worth running yourself when you touch
the engine layer:

```bash
uv run pytest -m model              # engine tests; downloads real models
uv run python scripts/e2e_smoke.py  # the full agent loop, tiny model
```

`model`-marked tests are excluded from CI because they need multi-GB
downloads. `slow`-marked tests — the process-group kill tests, which use real
processes on purpose — *do* run in CI, so don't skip them locally.

## Pull requests

`main` is protected, so work on a branch and open a PR. All four CI jobs must
be green.

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `test:`, `refactor:`. Keep the
subject imperative and lowercase after the type.

Explain *why* in the body, not just what — the diff already says what changed.
If a fix is subtle, say what you verified and how, especially for anything
timing-dependent.

## Conventions

- **ruff** with `line-length = 100`; config lives in `pyproject.toml`.
- **ty** type-checks the whole repo, tests included. Test doubles that
  deliberately implement only part of an interface use `cast`, with a comment
  saying why — see `tests/test_worker.py`.
- **`docs/`** is excluded from ruff. The design spec and implementation plan
  are point-in-time records; reformatting the Python inside their code blocks
  rewrites history for no benefit.
- Match the comment density and style of the surrounding code. This codebase
  explains non-obvious *reasoning* in comments and is fairly light on
  restating what the code says.

## Touching the sandbox

`src/sous/toolexec.py` is the security boundary: path confinement, the command
allowlist, and the process-group kill that stops a timed-out command's
descendants from writing files after the audit. The guarantees it makes are
described under [Security model](README.md#security-model).

Changes there need a test that fails without them, and will get a closer read.
Some of the behaviour is OS-level and genuinely unmockable — the existing
tests spawn real processes for that reason. If you are fixing something
intermittent, run the affected test in a loop before concluding it is fixed; a
single green run proves very little.

## Verifying the gateway endpoint

Until the routing half of issue #41 lands, every request that reaches the
gateway is served locally, so the only way to drive it from Claude Code is a
*whole-session-local* run. That is a verification setup, not a supported
mode (see README, "Gateway mode"). With `[gateway].enabled = true` and the
daemon restarted:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8383 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=sous-local \
  ANTHROPIC_DEFAULT_SONNET_MODEL=sous-local \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=sous-local \
  CLAUDE_CODE_SUBAGENT_MODEL=sous-local \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 \
  CLAUDE_CODE_AUTO_COMPACT_WINDOW=65536 \
  API_TIMEOUT_MS=3000000 \
  claude --disallowedTools LSP
```

Do **not** set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`: either one
switches Claude Code from your subscription login to API-credit billing the
moment traffic goes upstream again. The two context variables must match
`[gateway].max_context_tokens` (Claude Code honours them only for model ids
that are not `claude-*`, which is why the served id is honest).
`API_TIMEOUT_MS` covers model load plus a long prefill; `--disallowedTools
LSP` keeps a language server from appending its schema mid-session and
re-prefilling the whole conversation. Watch `~/.sous/daemon.log` for the
`sous gateway:` lines — the first turn is a cold prefill, later turns should
report `cache=hit`.

(The plan's Task 10 uses this recipe verbatim plus headless flags; keep the two identical.)

## Questions

Open an issue.

For anything security-sensitive, please don't — use GitHub's private
vulnerability reporting instead (**Security** tab → **Report a vulnerability**),
so a live weakness isn't described in public while it is unfixed. See
[SECURITY.md](SECURITY.md).
