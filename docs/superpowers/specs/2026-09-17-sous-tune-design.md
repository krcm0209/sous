# sous — `sous tune`: benchmark the machine, pick the model, write the config

*2026-09-17. Grounded in a same-day research pass: the open issues, the mlx /
mlx-lm / mlx-vlm release notes and unreleased main branches, 64 real gateway
turns from the M5 Pro daemon log, and a per-layer-type timing probe of the
default model on that machine. Approved shape: staged (throughput first, then
a quality suite), the task suite in the repo, per-download consent up front,
a standalone process, and a printed diff followed by an apply prompt.*

## What the research established

Every remaining "sous feels slow" lever except one is a decision no user
should have to make by hand, and every one of them is blocked on the same
missing thing: a harness that runs real delegated work through the real
worker loop and grades it. #78 (`int8_prefill` default-on), #87 (greedy vs
sampled), #28 (reasoning effort), #90 (KV quantization) and the mxfp4 and
35B-A3B model questions all end with "measure on the tool loop first".

The numbers behind that, measured on the M5 Pro (64 GB) with
`mlx-community/Qwen3.8-27B-4bit`, mlx 0.32.2, mlx-vlm 0.7.0:

| where the time goes | prefill, 2048-token chunk | decode, 16K context |
|---|---:|---:|
| MLP projections | 58% | 56% |
| GDN projections + recurrence | 22% (recurrence alone 6%) | 23% (recurrence 12.5%) |
| full attention | ~0–5% | 14.5%, rising to ~35% by 60K |
| total | 4.28 s (479 tok/s) | 66.7 ms/token (15 tok/s) |

Plain decode streams the 16.1 GB of weights at ~300 GB/s and is within ~10%
of that ceiling, so no kernel tweak moves it much. What does move it: a
drafter (token-exact, 1.8x at short context, ~1.1x by 60K because a 3-row
verify costs 1.17x a single row at 128 tokens and 1.25x at 16K), a model with
fewer active parameters, and never paying the cold prefill twice. In the
daemon log a cold turn is 55–88K tokens at 324–390 tok/s (141–272 s); a warm
turn answers in 1.0–1.4 s and decodes at 16–20 tok/s at ~60K context against
38–40 tok/s at 28 tokens.

Two machines have to be served: the M5 Pro (NAX tensor units, macOS 26.6,
working set 51.8 GiB) and an M2 Air (8-core GPU, 16 GB, macOS 15.5, working
set 10.7 GiB) where the 27B cannot load at all and the right answer is a
9B or 4B checkpoint with a compatible drafter, a smaller window and a smaller
prefill chunk. The same tool has to produce both answers.

On quantization variants: oMLX's oQ checkpoints are standard affine
mixed-precision safetensors (`Qwen3.8-27B-oQ4`: 4-bit base, 160 layers at
5-bit, one at 6-bit, 16.6 GB) and load in mlx-vlm unchanged; mlx-vlm's exact
speculative verifier accepts affine 4/5/8-bit layers, so oQ4 keeps the
drafter's speed while `oQ6` (6-bit base, 23.3 GB) loses it entirely and
`OptiQ-4bit` (261 layers at 8-bit, 20.7 GB) keeps it at ~30% more bytes per
token. None of those properties belongs in a hand-maintained table: they
are all readable from the checkpoint's `config.json`.

## Goal

`sous tune` chooses the model and the `[model]` settings for the machine it
runs on, so an end user never reads a tok/s table or a quantization format:

- **`sous tune --quick`** (stage B1): detect the hardware, list the
  candidates that fit, ask consent for each download, measure prefill and
  decode throughput of every arm through sous's own engine classes, and
  recommend only the settings that cannot change what the model says —
  drafter, block size, and a window that fits.
- **`sous tune`** (stage B2): everything above, then run a graded suite of
  mechanical coding tasks through the real worker loop for the candidate
  models, pick the fastest arm whose quality is within a fixed margin of the
  reference, then settle the quality-affecting settings (INT8 prefill,
  sampling) on that winner.
- Both print the hardware, the arms, the scores, the chosen arm with the
  rule that chose it, a unified diff of `config.toml`, and then ask before
  applying. Results and the report are kept under `~/.sous/tune/`.
- Everything a candidate needs is derived from its checkpoint, so a model id
  the user passes by hand is treated exactly like a curated one, and
  `--discover` can offer new checkpoints without a sous release.

Acceptance, on the M5 Pro and the M2 Air:

- `--quick` completes in about 15 minutes for three fitting candidates on
  the M5 Pro and recommends `speculative_block_size = 3` for the default
  model (the value measured best on that machine on 2026-09-05), or reports
  why a different value won;
- the full run completes, with `--resume` picking up after an interruption,
  its reference arm is eligible against itself, and with
  `--models mlx-community/Qwen3.8-27B-mxfp8` added it reproduces the
  2026-08-29 finding that the affine-4bit 27B is not worse than mxfp8 on the
  tool loop, in whichever direction the suite's margin allows;
- on the M2 Air it refuses the 27B with the memory arithmetic printed,
  offers the 9B, 4B and 2B tiers, and the applied config runs a delegated
  task end to end;
- CI exercises every decision path with fake engines and grades every suite
  task's reference solution at 1.0 and its untouched fixture at 0.0 without
  a model.

## Non-goals

- Per-task-type model switching. A model swap is a 16–20 GB reload; v1 picks
  one model per machine.
- Online tuning inside the daemon, or replaying the user's own `tasks.db`.
  Real tasks have no ground truth; a `--from-history` mode is a later idea.
- Non-Python suite tasks, or tasks that need anything beyond the standard
  library and `unittest`.
- KV-quantization arms (#90): the engine has no such setting yet. The arm
  model is generic so one can be added when it does.
- A curated-list freshness bot in CI. Designed for (`--discover` in
  metadata-only mode is what it would run) but a follow-up.
- Any change to how the daemon, the gateway or the worker generate. The only
  engine change is the one line that lets `temperature = 0` reach mlx-vlm's
  greedy speculative path (#87), because without it a greedy arm would
  measure the sampled code path under an argmax sampler.

## Design

### Command and process model

`sous tune [--quick] [--models ID ...] [--runs N] [--repeat N]
[--resume RUN_ID] [--discover] [--yes] [--apply]` is a subcommand in
`src/sous/cli.py` that
delegates to `sous.tune.main`. It runs as its own process — the daemon's
engine is built for one configured model, and the suite needs the worker
loop in-process with a scratch task store, the way `scripts/e2e_smoke.py`
already drives it. mlx is imported function-locally, as everywhere else.

Each arm runs on a dedicated thread that loads the engine through
`sous.engine.base._default_factory` with a `dataclasses.replace`d
`SousConfig`, measures, unloads (`engine.unload()`, `mx.clear_cache()`) and
calls `release_mlx_thread_state()` on its way out. The next arm starts only
once `mx.get_active_memory()` has fallen back to the pre-load level, so two
models are never resident together.

`--yes` answers every consent prompt with yes (downloads and apply) for
scripted runs; `--apply` applies without the final prompt but still asks for
downloads. Without a TTY and without those flags the command prints and exits
0 without changing anything.

### Preconditions and `POST /sous/unload`

The tune refuses to start while the daemon has the model loaded or loading,
a `sous claude` session holds it (`engine.holders > 0`), a task is running,
or a gateway turn is in flight — the machine cannot hold two models, and a
live session's turns would stall behind the bench. It reads `GET /sous/status`
for all of that and prints the exact reason.

To keep the user out of daemon mechanics it first asks the daemon to release
the weights: a new loopback-guarded `POST /sous/unload` (same
`check_loopback` and fetch-metadata rules as `/sous/hold`, empty body)
calls `EngineManager.unload_now()`, which is `unload_if_idle` without the
idle-clock test (a request with a body is refused the way a malformed hold
is): it refuses (HTTP 409, `{"unloaded": false, "reason": ...}`)
under an in-flight generation, a held lease, any holder or a running task,
and otherwise unloads and answers `{"unloaded": true}`. The daemon reloads
lazily on its next request, as after any idle unload. A daemon that is not
running is fine; the tune does not start one.

### Hardware detection (`sous/tune/hardware.py`)

One `Hardware` record, printed as the report header and saved as
`hardware.json`: chip (`mx.device_info()['device_name']`), unified memory,
Metal recommended working set, GPU architecture string and NAX eligibility
(reusing `int8prefill.availability()`), macOS version, mlx / mlx-lm /
mlx-vlm / sous versions, the Hugging Face cache path and its free disk.
Nothing here needs a model loaded.

### Candidates (`sous/tune/candidates.py`, `candidates.toml`)

The curated table is data, shipped in the package:

```toml
checked = 2026-09-17            # when the rows below were last reviewed
validated_with = "mlx-vlm 0.7.1"
trusted_orgs = ["mlx-community", "z-lab"]   # converters --discover may offer
publishers = ["Qwen"]                        # base-model orgs it accepts

[[candidate]]
id = "mlx-community/Qwen3.8-27B-4bit"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "shipped default; affine 4-bit keeps the fast exact verifier"

[[candidate]]
id = "mlx-community/Qwen3.8-27B-oQ4"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]

[[candidate]]
id = "mlx-community/Qwen3.8-27B-mxfp4"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "fast exact verifier only from mlx-vlm 0.7.1"

[[candidate]]
id = "mlx-community/Qwen3.8-27B-OptiQ-4bit"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]

[[candidate]]
id = "mlx-community/Qwen3.8-27B-oQ6"
tier = "27b-dense"
drafters = []
note = "6-bit base loses the fast exact verifier; quality tier only"

[[candidate]]
id = "mlx-community/Qwen3.5-35B-A3B-4bit"
tier = "35b-moe"
drafters = ["z-lab/Qwen3.5-35B-A3B-DFlash"]

[[candidate]]
id = "mlx-community/Qwen3.5-9B-MLX-4bit"
tier = "9b"
drafters = ["z-lab/Qwen3.5-9B-DFlash", "mlx-community/Qwen3.5-9B-MTP-4bit"]

[[candidate]]
id = "mlx-community/Qwen3.5-4B-MLX-4bit"
tier = "4b"
drafters = ["z-lab/Qwen3.5-4B-DFlash"]

[[candidate]]
id = "mlx-community/Qwen3.5-2B-MLX-4bit"
tier = "2b"
drafters = []
```

A row carries only what cannot be read off the checkpoint: its tier, the
drafters known to pair with it, and a note. Everything else is derived by
`describe(model_id)` from the checkpoint's `config.json` and the Hub file
listing, and cached in the run directory:

- **bytes**: the sum of the safetensors sizes (Hub `files_metadata`), or the
  on-disk size when already cached; a drafter's resident size is its bf16
  bytes ÷ 3.5 (4-bit codes plus group scales), since sous quantizes drafters
  to 4-bit at load;
- **KV bytes per token** and **native window** from `context.kv_bytes_per_token`
  and `native_max_tokens`, so hybrid models count only their attention layers;
- **per-layer quantization**: the base `quantization` block plus overrides,
  summarised as the fraction of linears the exact verifier serves with its
  fast kernels (affine, bits in {4, 5, 8}, biases present) and the fraction
  `int8prefill` can route (affine 4-bit, group size 64);
- **backend** via `select_backend`, and **drafter compatibility** by the same
  hidden-size and layer-count checks `validate_drafter_compatibility` makes,
  so an incompatible pair is dropped before anything is downloaded.

**Fit.** A candidate fits when
`bytes + drafter_bytes + kv_per_token × window + 2 GiB ≤ working set`, with
`window` the larger of the worker's `max_context_tokens` and, when the
gateway is enabled, its window. A candidate that does not fit at the
configured window but does at a smaller one is kept with the smaller window
recorded as that arm's setting — this is how the M2 Air gets a 9B at 32K
rather than nothing. The arithmetic is printed for every refused candidate.
There is one table for every machine: the fit rule is the machine filter,
and the *reference* differs per machine (below).

The table's `checked` date is printed with the header; past 90 days the
report says so and suggests `--discover`.

### Download consent (`sous/tune/hub.py`)

After the fit pass, every snapshot an arm needs that is not in the Hub cache
is listed once, in one block, before anything runs:

```
Downloads needed (free disk 412 GB at ~/.cache/huggingface/hub):
  1. mlx-community/Qwen3.5-35B-A3B-4bit   20.4 GB   candidate (35b-moe tier)
  2. z-lab/Qwen3.5-35B-A3B-DFlash          0.8 GB   drafter for #1; refusing keeps #1 without a drafter
  3. mlx-community/Qwen3.8-27B-oQ4        16.6 GB   candidate (27b-dense tier)
Download #1? [y/N]
```

Each answer is separate. Refusing a candidate removes its arms; refusing a
drafter keeps the model's no-drafter arm. Approved snapshots are fetched with
`huggingface_hub.snapshot_download` before any measurement, so the bench
never stalls on a download mid-run. `config.json` and file listings are
metadata (kilobytes) and are fetched without asking; offline, a candidate
with no cached metadata is reported as "unknown, offline" and skipped.

### Arms and stages (`sous/tune/arms.py`)

An **arm** is a `SousConfig` derived from the user's: `model_id`,
`speculative_draft_id` (or `""`), `speculative_block_size`, `int8_prefill`,
`temperature`/`top_p`/`top_k`, and the worker's and gateway's
`max_context_tokens` (only ever lowered, for fit). Everything else is the
user's own config. The user's current configuration is always one of the
arms, whether or not its model is in the curated table. Each arm gets a
scratch `data_dir` holding a generated `config.toml` (the worker reads the
allowlist from `config_path` on every command), so a run never touches
`~/.sous`.

Two kinds of dimension:

- **quality-neutral** — drafter, block size, window. A verified drafter emits
  the target's own distribution, so these are settled by throughput alone in
  every mode;
- **quality-affecting** — model, `int8_prefill`, sampling. Settled only by
  the suite, and only in the full run.

Stages of the full run:

1. **Quick** (also the whole of `--quick`): for every fitting candidate ×
   drafter-or-none, the throughput bench at block sizes 2, 3 and 5 (drafter
   arms only). Output: the best block per (model, drafter), and each model's
   fastest quality-neutral arm.
2. **Model stage**: the suite runs each candidate model in its fastest
   quality-neutral arm, plus the reference arm. The decision rule picks the
   winner.
3. **Winner stage**: on the winner only, the suite runs `int8_prefill = true`
   (when the hardware allows it and the checkpoint routes) and greedy
   sampling (`temperature = 0`), one extra arm each, judged by the same rule
   against the winner's own result. This is why no user ever chooses
   `--greedy`: the tool measures it where it matters and explains the answer.

### Throughput bench (`sous/tune/bench.py`)

Through the real engine (`VLMEngine`/`LMEngine` from `_default_factory`),
one `GenerationSession` per arm so warm turns reuse the slot the way the
worker and the gateway do. The prompt is a synthetic, deterministic
code-shaped conversation (a system message, a tool array — `WORKER_TOOLS` —
and user content built from a fixed seed), sized to 2K and 16K tokens (16K
only when the arm's window allows it):

| measurement | how |
|---|---|
| load seconds | around `EngineManager.get()` |
| prefill tok/s at 2K and at 16K | a cold `generate(max_tokens=1)`; `prefilled_tokens / prefill_seconds` from the owner-scoped `prompt_cache_stats` gauges |
| decode tok/s at 1K and at 16K context | a warm follow-up `generate(max_tokens=256)`; output tokens ÷ `decode_seconds` |
| warm-turn TTFT | a third turn adding ~100 tokens; `generate()` entry to first delta |
| peak memory | `mx.get_peak_memory()` after the arm |

Block sizes are swept on the decode measurements only. Each number is the
best of two attempts (`--repeat` raises it); the spread is printed so a
thermally throttled laptop shows as noise rather than a finding. The gauges
are per-turn readings, so the bench reads them right after each `generate()`
on the session thread, never as differences of counters.

### Quality suite (`sous/tune/suite/`)

Eight tasks ship in the package, one directory each:

```
suite/<name>/task.toml     title, instructions, context_files, verify_commands,
                           category, max_turns (default 16), max_minutes (default 10)
suite/<name>/project/      the fixture the worker sees (copied to a temp root)
suite/<name>/grade/        hidden: unittest modules or a grade.py script
suite/<name>/solution/     a reference result, used only by CI to prove the grader
```

Categories and graders, all standard library only so an end user's machine
needs nothing beyond Python:

| task | category | grader |
|---|---|---|
| implement a module from a spec | implement-from-spec | hidden `unittest` suite, score = passed/total |
| add tests for an existing module | test-scaffolding | the worker's tests run against the pristine module and against a mutated copy; score = pass on pristine × catches mutations |
| docstring sweep | mechanical-sweep | AST check of every public function plus behaviour unchanged |
| rename a symbol across files | cross-file-refactor | hidden tests import the new name; grep proves the old one is gone |
| fix a failing test | bug-fix | hidden tests, including the regression |
| dataclass from a JSON schema | codegen | hidden tests round-trip the schema |
| add a CLI flag | feature-slice | hidden tests run the CLI in a subprocess |
| migrate a config format | mechanical-sweep | hidden tests read the migrated files |

The runner copies `project/` to a temp root, enqueues the task in a temp
`TaskStore`, and runs `run_worker_loop` with the arm's config: the shipped
`DEFAULT_ALLOWLIST` plus `python -m unittest`, the tune's own interpreter
first on the worker's `PATH`, and a poller that denies any approval request
the moment it appears and counts it. The suite never sees `grade/` or
`solution/`. Per run it records: state and outcome, turns used, wall seconds
(`finished_at − started_at`), output tokens, malformed tool calls
(transcript `event == "malformed"`), repetition incidents (three identical
consecutive tool calls, parsed from the transcript's `generation` events with
`protocol.parse_tool_calls`), approvals denied, and the grade in `[0, 1]`
with the grader's details. Two runs per task by default (`--runs`); every
run is appended to `results.jsonl` as it finishes, and `--resume RUN_ID`
skips whatever is already there.

### Decision rule (`sous/tune/decide.py`)

Printed in full, with every number, before the choice:

- **reference** = the user's current arm when it fits this machine, else the
  fastest quality-neutral arm of the largest fitting tier;
- **eligible** = mean grade ≥ reference − 0.05, and
  `completed_runs ≥ reference_completed_runs − runs_per_task` (a run is
  completed when its task ends in state `done`), and repetition incidents ≤
  the reference's;
- **winner** = the eligible arm with the lowest suite wall time (the sum of
  run seconds, which weighs prefill and decode the way a user experiences
  them); ties go to the arm with the smaller memory footprint.

The winner stage applies the same test with the winner as reference. In
`--quick` mode the rule never changes the model: it chooses among the current
model's quality-neutral arms and reports the other models' throughput with
an explicit "quality untested" label.

### Report, diff, apply (`sous/tune/report.py`)

Plain-text tables (no new dependencies): hardware; candidates with fit
arithmetic and derived properties; the bench; the suite; the rule and the
choice. Then the config change as a unified diff between the user's
`config.toml` and a `tomlkit` round-trip with the winning arm's keys set —
the same formatting-preserving path `persist_allowlist_entry` uses, so
comments and ordering survive — followed by:

```
Apply these changes to ~/.sous/config.toml? [y/N]
```

Yes writes a backup (`config.toml.bak-tune-<run-id>`) and the new file. A
model or `int8_prefill` change adds one line saying the daemon must restart
to pick it up (`sous stop` for an unmanaged daemon, `launchctl kickstart -k
gui/<uid>/<label>` for a managed one). `report.md`, `results.jsonl` and
`hardware.json` land in `~/.sous/tune/<run-id>/` regardless.

### Discovery (`--discover`)

Metadata only, no model needed: for each `trusted_orgs` entry, list
checkpoints whose `config.json` `model_type` is a family the installed
mlx-vlm or mlx-lm ships, whose model card's `base_model` names a repo under
one of the table's `publishers` (`Qwen/` for the Qwen families), and that
fit the machine; drafters by the same compatibility checks against each
candidate.
They join the run as "unvetted" arms with the same per-download consent, and
the report marks them. Family support tracks the installed libraries, so a
new mlx-vlm release widens discovery without a sous release; a family whose
template cannot tool-call scores zero in the suite and is rejected by the
rule, never by a hand-kept list.

### Engine change for the greedy arm

`VLMEngine.decode` passes `sampler=None, temperature=0` instead of an argmax
sampler when the configured temperature is 0, so mlx-vlm's greedy
speculative branch is what the arm measures (#87). At any other temperature
nothing changes. The LM backend keeps its sampler either way.

### Delivery

Three PRs, each shippable on its own:

1. **`sous tune --quick`**: `/sous/unload` and `unload_now`, hardware,
   candidates and `describe()`, the download plan and consent, arms, the
   throughput bench, the quick-mode rule, report, diff and apply, the CLI
   subcommand, README.
2. **The full run**: the suite (eight tasks with graders and solutions), the
   runner and its metrics, the model and winner stages, the full decision
   rule, `--resume`, the greedy engine change.
3. **`--discover`**: Hub discovery, the `publishers` test, the "unvetted"
   labelling, the `checked`-age warning.

`results.jsonl` holds bench rows and suite runs alike from PR 1 on, so
`--resume` covers both once PR 2 lands.

## Invariants preserved

- The daemon never runs a benchmark: `sous tune` is a separate process and
  starts only when the daemon holds no model, no session and no task.
- Two models are never resident at once; every mlx-touching thread releases
  its streams before it exits.
- The suite runs the real worker loop under the real sandbox with the shipped
  allowlist plus `python -m unittest`; approvals are denied, never granted.
- No config change without the user's answer to the apply prompt (or the
  explicit `--apply`/`--yes`), and every change is a diff they read first.
- No download without the user's answer for that specific snapshot; metadata
  fetches are the only network access without consent.
- `--quick` never changes the model or a quality-affecting setting.
- `/sous/unload` refuses everything the idle sweep refuses; it only removes
  the clock.

## Testing

Fake engines and no model, in CI:

- `describe()` on saved `config.json` fixtures for every curated row:
  bytes, KV per token, verifier-fast and int8-routable fractions, drafter
  compatibility; the fit rule on the M5 Pro and M2 Air numbers, including
  the window-lowering path;
- arm enumeration for quick and full runs, with and without NAX;
- the download plan and consent parsing from a fake stdin (each answer
  removes exactly what the prompt said it would);
- the decision rule on fixture `results.jsonl` files: a faster arm below the
  margin loses, a tie goes to the smaller footprint, `--quick` never changes
  the model;
- report formatting and the diff/apply round-trip on a temp config with
  comments, including the backup and the restart note;
- `POST /sous/unload`: 200 on an idle loaded engine, 409 with the reason
  under a generation, a lease, a holder and a running task, and the loopback
  and fetch-metadata refusals shared with `/sous/hold`;
- every suite task: the grader scores `solution/` at 1.0 and the untouched
  `project/` at 0.0, `task.toml` validates, and `importlib.resources` finds
  the fixture in the installed package (hatchling ships package data);
- the runner's metric extraction on a fixture transcript (malformed,
  repetition, approvals denied);
- the greedy path: with temperature 0 the engine hands mlx-vlm no sampler
  (a fake `stream_generate` records its kwargs).

Model-marked, local only: one quick arm on `mlx-community/Qwen3-0.6B-4bit`
end to end, and the smallest suite task through the runner on it (outcome
recorded, grade computed, no assertion on the grade — the 0.6B cannot pass).

## Documentation

- README: a "Tuning" section with the two commands, what each may change,
  the consent and apply prompts, where results go, the daemon-restart note,
  and the memory-tier table replaced by "run `sous tune`" with the old table
  kept as the fallback for offline machines.
- The `[model]` config reference: which keys `sous tune` writes.
- `docs/` note on adding a suite task and a curated row, and the
  `checked` date discipline.

## Risks and open questions

- **Bench variance.** A throttled or busy machine moves tok/s by 5–10%; the
  best-of-two and the printed spread bound it, and the rule's model choice
  rests on the suite's wall time rather than a single tok/s reading.
- **Suite noise.** Two sampled runs per task at temperature 0.7 is a parity
  check, not a ranking; the 0.05 margin and the one-task allowance encode
  that, and `--runs 3` is there for a slower, surer answer.
- **Suite representativeness.** Eight stdlib Python tasks stand in for all
  delegated work. They are mechanical by construction, which is sous's
  stated scope; a category that matters to a user and is missing is a task
  to add, not a rule to bend.
- **Run time.** A full run on the M5 Pro with three fitting models is about
  three hours; the ETA is printed after the quick stage from the measured
  speeds, and `--resume` makes an interruption cheap.
- **Non-Qwen families.** Discovery may offer a family whose template or
  tool-call format sous's parser does not speak; the suite scores it zero
  and the rule rejects it, at the cost of the run time spent finding out.
  A tool-call smoke turn before the suite (one forced `finish` call) would
  save that time and is worth adding if it happens in practice.
- **Hub availability.** Offline, the curated table still works for cached
  checkpoints; discovery and uncached candidates are skipped with a line
  saying so.
