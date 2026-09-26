# Tuning

`sous tune` chooses the model and the `[model]` settings for the machine it
runs on, so you never read a tok/s table or a quantization format. It ends
with a report, a unified diff of `~/.sous/config.toml`, and a question:

```
Apply these changes to ~/.sous/config.toml? [y/N]
```

Nothing is written before that yes; a backup is kept beside the config file,
and the tune prints the daemon-restart command every applied change needs
(`sous stop`, then `sous serve`, or `launchctl kickstart -k gui/<uid>/<label>`
for a managed daemon — the daemon reads `[model]` once at startup).

**`sous tune --quick`** (about 15 minutes on an M5 Pro for three fitting
candidates) detects the chip, the Metal working set and whether the GPU has
tensor units, fits every curated candidate to memory (printing the
arithmetic for each one it refuses), asks about every download it would need
one by one, then measures prefill and decode throughput of every arm — a
model, a drafter or none, a block size — through sous's own engine
(`--repeat` sets the attempts per measurement, default 2; the best wins and
the spread is printed). It proposes only the settings that cannot change
what the model says: the drafter, its block size, and a window that fits.
Other models are measured and reported with a "quality untested" label; a
quick run never changes the model.

**`sous tune`** (about three hours on an M5 Pro for three fitting models; the
estimate is printed after the quick stage from the measured speeds and covers
the model stage; the winner stage adds up to two arms of the same size) does
all of the above, then grades the candidates: for each model's fastest
quality-neutral arm, and for your current configuration, it runs a suite of
eight mechanical coding tasks — implement a module from its spec, write
tests for one, a docstring sweep, a cross-file rename, a bug fix, a dataclass
from a JSON schema, a CLI flag, a config-format migration — through an agent
loop of its own over a scratch copy of each task's project, running only that
task's verify commands and the suite's test runners, and denying nothing
because there is nothing to approve. Each run is scored by a
hidden grader (`--runs` sets the runs per task, default 2). The rule, printed
in full with every number:

- the **reference** is your current configuration when it fits this machine,
  else the fastest arm of the largest tier that does;
- an arm is **eligible** when its mean grade is within 0.05 of the
  reference's, it completed at least as many runs as the reference minus one
  task's worth, and it looped on a tool no more often;
- the **winner** is the eligible arm with the lowest total suite wall time
  (a tie goes to the smaller memory footprint).

On the winner, the same rule then judges one extra arm per quality-affecting
setting: INT8 prefill (where the tensor units and the checkpoint allow it)
and greedy sampling (`temperature = 0`, which also lets the drafter's
exact-match verify run). A setting lands in the diff only when its own
measured arm is eligible and faster — that is why there is no `--greedy`
flag to understand. The full run may therefore change `[model].id`, the
drafter and block size, the window, `int8_prefill` and `temperature`.

The daemon is asked to release the model first (`POST /sous/unload`) and
refuses while a `sous claude` session holds it, a turn is in flight, or a
load or unload is under way — the tune waits for none of them, it tells you,
and it asks again right before its first bench load and before every model the
suite loads. Results (`results.jsonl` with every bench row and suite run,
`hardware.json`, `report.md`, and each suite run's project and transcript under
`suite/`) land in `~/.sous/tune/<run-id>/`; `--resume <run-id>` continues an
interrupted run from the rows it already has; `--models ID ...` measures ids of
your own; `--yes` answers every prompt for scripted use, `--apply` skips only
the final one. Adding a suite task or a curated candidate is described
[below](#changing-what-it-measures).

## Changing what it measures

The section above says what `sous tune` does and what it may change. This
one is for changing what it measures.

### Adding a suite task

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

### Adding a curated candidate

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
