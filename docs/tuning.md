# Tuning: the suite and the candidate table

`sous tune` measures this machine and grades the candidate models; the README's
Tuning section says what it does and what it may change. This note is for
changing what it measures.

## Adding a suite task

A task is a directory under `src/sous/tune/suite/tasks/<name>/`:

```
task.toml     title, category, instructions, context_files, verify_commands,
              max_turns (default 16), max_minutes (default 10)
project/      the fixture the candidate sees, copied to a scratch root per run
grade/        hidden: test_*.py unittest modules, or a grade.py
solution/     the complete solved project, used only by CI to prove the grader
```

Rules the loader and CI enforce (`tests/test_tune_suite.py`):

- `category` is one of implement-from-spec, test-scaffolding, mechanical-sweep,
  cross-file-refactor, bug-fix, codegen, feature-slice; every `context_files`
  entry exists under `project/`; `grade/` holds a `grade.py` or at least one
  `test_*.py`; there are no other keys.
- The grader scores `solution/` at 1.0 and the untouched `project/` at 0.0.
  The easy way to guarantee the second half: write every hidden test so it
  fails before the change — import the new name, use the new flag, read the
  migrated file — and put the "nothing else changed" assertions inside those
  same tests rather than in tests of their own.
- Standard library only, in the fixture, the grader and the solution: an end
  user's machine has nothing else, and the suite loop runs only a task's
  verify commands and the test runners (`python -m unittest`,
  `python -m pytest`, their `python3` spellings and bare `pytest`).
- A task must never require deleting, moving or renaming a file: the suite's tools
  read, write and edit files and run those commands; there is no `rm` or `mv`.
  Ask for a rewrite, or say "leave the old file in place".
- Running `python -m unittest` executes the candidate's own code on your
  machine, like every test runner would; the suite's tools confine file edits
  to the scratch copy, not what its tests execute.
- Hidden `test_*.py` modules run in a subprocess with the candidate's project as
  the working directory (`python -m sous.tune.suite.unittests grade/`), so they
  import the candidate's modules by bare name; import inside the test methods so
  a missing module fails that test rather than the whole module.
- A `grade.py` defines `grade(project: Path, tests) -> tuple[float, str]`,
  where `tests(cwd, tests_dir=None)` runs unittest modules (`tests_dir`
  defaults to the hidden ones) and returns `(passed, total, detail)`. See
  `tests_for_slugify` (mutation scoring) and `docstring_sweep` (an AST check
  over the behaviour tests).
- Keep a task small enough that a 27B finishes it in a dozen turns: the
  suite's purpose is parity between arms, not a leaderboard.

The fixture trees are excluded from `ty` (their imports resolve only inside a
copied project) and linted and formatted by ruff like everything else.

## Adding a curated candidate

`src/sous/tune/candidates.toml` lists the models `sous tune` measures. A row
is an id, a tier, the drafters known to pair with it and a note; everything
else — bytes, KV cost per token, quantization layout, the exact-verifier and
INT8-prefill eligibility, drafter compatibility — is read from the
checkpoint's `config.json` at run time, so a row never carries a number that
can go stale. Rows are ordered largest tier first; the full run's fallback
reference (when the configured model does not fit) is the first fitting one.

Bump `checked` to the day you last verified the table against the Hub and
set `validated_with` to the mlx-vlm version you ran the suite on. The report
warns when `checked` is more than 90 days old, and `--discover` (a later
release) is what a freshness bot would run.
