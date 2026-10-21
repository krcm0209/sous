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
what the model says beyond a near-tie: the drafter, its block size, and a
window that fits — a block size, or one drafter for another, changes
nothing, and a drafter against none moves a near-tie only where the
projection tile is active (the 20-core M5 Pro; see `projection_tile` in
[Configuration](configuration.md)).
Other models are measured and reported with a "quality untested" label; a
quick run never changes the model.

**`sous tune`** (about three hours on an M5 Pro for three fitting models; the
estimate is printed after the quick stage from the measured speeds and covers
the model stage; the winner stage adds at most one arm of the same size) does
all of the above, then grades the candidates: for each model's fastest
quality-neutral arm, and for your current configuration, it runs a suite of
eight mechanical coding tasks — implement a module from its spec, write
tests for one, a docstring sweep, a cross-file rename, a bug fix, a dataclass
from a JSON schema, a CLI flag, a config-format migration — through an agent
loop of its own over a scratch copy of each task's project, running only that
task's verify commands and the suite's test runners, and denying nothing
because there is nothing to approve. Each run is scored by a
hidden grader (`--runs` sets the runs per task, default 2). Every arm and run
works on its copy at one path, `<run-id>/project`, which the prompt names, so
arms are compared on the same prompt; run *i* of every sampled arm samples from
seed *i*, so a task's runs differ on purpose rather than replaying one stream.
With `temperature = 0` set, every arm is greedy and draws nothing, so its runs
of a task repeat one trajectory and count as that one result more than once.
The rule, printed in full with every number:

- the **reference** is your current configuration when it fits this machine,
  else the fastest arm of the largest tier that does;
- an arm is **eligible** when its mean grade is within 0.05 of the
  reference's, it completed at least as many runs as the reference minus one
  task's worth, and it looped on a tool no more often;
- the **winner** is the eligible arm with the lowest total suite wall time
  (a tie goes to the smaller memory footprint).

On the winner, the same rule then judges one extra arm: INT8 prefill, only
for a winner that has it off (it ships on) and only where the tensor units and
the checkpoint allow it. That arm keeps your `projection_tile`, like every
other setting: the tile serves only beside active int8 prefill, so where it can
run the winner ran without it and the int8 arm with it, and the grade is that of
the configuration the diff (which writes `int8_prefill` alone) then deploys. It
lands in the diff only when its own measured arm is eligible and faster. The tune never changes `temperature`: every arm inherits
yours, and greedy decoding (`temperature = 0`) is never proposed, because it
leaves a model that starts repeating itself no way out and, as above, the suite
cannot measure that risk; it runs only when you set it. The full run may
therefore change `[model].id`, the drafter and block size, the window and
`int8_prefill`.
Every arm inherits `[model].attention_tile`, `[model].projection_kernel`,
`[model].kv_int8` and `[model].projection_tile`, and no arm changes them;
suite rows are keyed by all four, so a `--resume` after any of them was flipped
runs those arms again instead of
mixing rows measured with and without it (a row written before a setting
existed reads as `false` for it). For `projection_tile` a row records what the
arm's load reported — the scope `all` or `mlp` when the tile went active, else
`unavailable` or `off` — rather than the setting, since `"mlp"` and `"all"` put
different projections on the tile and `true` picks either at load, from the
headroom it finds: rows of two scopes never pool. An arm is graded on the rows
of the one state that finished the most runs; a `--resume` loads it again only
while no state has finished every run, and then counts as done only the runs
finished under the state that load reports, so a load that picks the other
scope runs the arm's tasks again under it. Where the engine reports one
`unavailable`, the arm prints a notice with the reason and runs without it, as
the daemon would: stock attention or projections, attention over the bf16
cache for `kv_int8`, or no tile for `projection_tile` (an `off` engine, such as
the mlx-lm backend's, gets no notice, and neither does the projection tile's
`no drafter` refusal, which every drafterless arm would print). The projection
kernel's route for the drafter's own linears reports apart from the kernel and
gets a notice of its own only when it is `unavailable`, the arm then drafting
on stock projections: where the kernel serves the target but the drafter's
probe failed, say, and also wherever the kernel itself is `unavailable` and the
arm's block is 0 or 6 and up, which prints both notices with the same reason.
A block of 1 to 5 leaves the route `off`, with no notice. `kv_int8` matters
most here of the four: attention past its floor reads keys and values
quantised to int8, which moves more than the last bits, and the suite's grades
are where that would show. Fitting a candidate also works out the scope the
projection tile's memory rule would pick at the fitted window (the copy is
elective, so it never lowers the window), but the tune prints that scope
nowhere today; the daemon's model-load line (`projection_tile=active:<scope>`)
is where a load's scope shows. Bench rows are keyed only by model, drafter,
block and window, so a `--resume` across a sous upgrade or a flip of
`attention_tile`, `projection_kernel`, `kv_int8` or `projection_tile` reuses
decode and prefill timings measured before it, which can rank the blocks the
other way: start a fresh tune after any of them.

Every drafter arm runs exactly the block its label names. Your configured arm
is the block the daemon would run: with `speculative_block_size` unset, that
is 8 where the projection kernel can run, 4 where only the attention tile can
and 3 elsewhere. The tune decides which without loading the model — from the
platform rule, the GPU's core count, `attention_tile` and `projection_kernel`,
the model's config and the drafter's kind — and prints a line when the unset
block resolves to 8 or 4. A refusal only a load would show (a failed probe, a
pinned mlx-vlm source that changed) leaves that mark a block or two off; the
measurements themselves are unaffected. A proposal then leaves an unset block
alone when the winner runs the block it already resolves to, writes any other
block explicitly, and always writes the block of a winner with another model
or drafter, since an unset block could resolve differently for it.

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
