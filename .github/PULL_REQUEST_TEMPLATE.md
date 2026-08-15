<!--
CI already reports whether lint, types, and the fast test suite pass, so this
template only asks for the things it cannot check for you.
-->

## What and why

<!-- What changed, and what problem it solves. The diff covers what; the why
     is the part that is hard to recover later. -->

## How you verified it

<!-- CI runs `pytest -m "not model"` and nothing else. The model-marked engine
     tests and scripts/e2e_smoke.py never run there, so if you touched the
     engine layer, this is the only place that evidence exists. Commands and
     their output beat "tested locally". -->

## Checklist

- [ ] Touched the engine layer (`src/sous/engine/`): ran `uv run pytest -m model`,
      and/or `uv run python scripts/e2e_smoke.py`
- [ ] Touched the sandbox (`src/sous/toolexec.py`): included a test that fails
      without this change
- [ ] Fixing something intermittent: ran it repeatedly rather than once, and
      said how many times above
- [ ] Commit subject follows [Conventional Commits](https://www.conventionalcommits.org/)

<!-- Strike out or delete any line that does not apply. -->
