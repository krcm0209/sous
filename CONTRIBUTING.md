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

## Questions

Open an issue. For anything security-sensitive, please say so in the title so
it gets looked at first.
