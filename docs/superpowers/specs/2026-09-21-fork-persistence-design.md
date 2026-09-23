# sous — Persist prompt-cache fork slots to disk

*2026-09-21. Issue #89. Follows the tools-boundary fork (Phase 3c, PR #72),
cache continuity (PRs #93, #94) and the observability PRs (#85, #95), and
takes up the "SSD tier" spike the continuity spec deferred, narrowed to what
the daemon log says is worth keeping. Grounded in the maintainer's M5 Pro
daemon log (2026-09-13 to 2026-09-22), six measurement probes against mlx
0.32.2 / mlx-vlm 0.7.1 on an M2 Air, and a read of mlx-vlm's own disk cache.
Where a number below was measured on the M2 Air it says so; the M5 Pro
numbers the README will print are owed by the live gate.*

## What the log established

Fork slots die with the weights. `VLMEngine.unload` resets the prompt cache,
so an idle unload (`[model].idle_unload_minutes`, 30 by default) or a daemon
restart drops every fork, and the next `claude` session's first subagent turn
prefills the ~45–57K-token tool block from zero.

The daemon log over 8.5 days (71 locally served turns, 13 model loads):

- Every one of the 13 model loads was followed by a `cache=miss` first turn.
  Zero sessions started warm after a load, in a log whose only two
  `cache=fork` lines are mid-session.
- 13 misses prefilled 20K tokens or more: 44.6K–88.2K tokens at 113–272 s
  each, 1,957 s of prefill in total. Each sat 117–182 s after a
  `model_load` line — a post-load cold start, not a changed prompt.
- 8 distinct tool arrays (`tools=` hashes) appeared; four of them account for
  52 of the 71 turns. Six of the 24 misses presented a tool array the log had
  already seen, so a tools fork that survived the unload would have served
  them. Estimated saving, `prefill_s × tools_boundary / prefilled_tokens`
  minus a 2 s restore: ~831 s of the 1,957 s.
- The header boundary sat 574–2,511 tokens above the tools boundary
  (`bounds=[56578, 57152]` is typical). A header fork on disk would have
  added ~7.6 s over the same nine days: `system=` in the log hashes the
  client's system *text*, and the header boundary's ids include the tool
  block, so a turn with a new tool array can hit neither fork.

So the win is one file per tool array, restored across unloads and
restarts, and the tools boundary is where nearly all of it lives.

## What the probes established

Measured on the M2 Air with mlx 0.32.2 unless noted; the numbers that decide
the design's shape, each load-bearing below:

- **Saving is free only for evaluated, contiguous arrays.** `mx.save_safetensors`
  of an evaluated contiguous buffer adds no device memory. A lazy slice
  (`keys[..., :offset, :]`, which is exactly what `KVCache.state` returns) is
  materialised first — for a multi-tensor save, *every* tensor at once —
  costing a transient equal to the whole payload (+186.8 MiB on 187 MiB;
  ~3.5 GiB on a real fork). A non-contiguous array costs the same and 5× the
  wall time. mlx-vlm's own disk cache slices unconditionally and measured
  +100 %.
- **A lazy array cannot be evaluated from another thread.** Saving arrays
  another thread built but did not evaluate fails deterministically with
  `There is no Stream(gpu, 0) in current thread` — a clean `RuntimeError`,
  not a hang or a crash. The write must run on the owner thread.
- **`mx.load` is lazy and mmap-backed.** The call reads the header only
  (0.1–0.7 ms on a 512 MiB file, zero memory); evaluating one small tensor
  reads only its bytes; evaluating after `os.unlink` still returns the data.
  One unexplained anomaly: a *second* `mx.load` of a path already loaded and
  then deleted in the same process raised the stream `RuntimeError` on a
  small tensor 3/3 times in one script shape and 0/5 in five near-identical
  ones. The design makes that path unreachable rather than relying on it.
- **`KVCache.state`'s setter derives `offset` from the array's shape.** A
  full padded buffer restored through it comes back with `offset` equal to
  the padded capacity (1000 → 1024 measured). That is not 24 tokens of
  zeroed KV treated as content: `VLMEngine._positions` reads `c.offset` to
  place every appended token, so the whole continuation is mispositioned,
  silently. Restoring by assigning `keys`, `values` and then `offset` from
  metadata reproduces the live cache exactly — same values, same capacity,
  and a further chunk extends both identically (bit-exact, `nbytes` equal).
- **`fork_copy`'s copies are lazy.** `mx.array(a)` allocates nothing until
  evaluated; `nbytes` already reports the full size, `mx.get_active_memory()`
  does not move. `live_headroom()` therefore over-reports by the forks a
  turn just published until something evaluates them. A restored cache is
  the opposite: evaluated, and visible to the valve at once.
- **Write throughput at full size is 1.73 GiB/s** (3.5 GiB in 2.0 s, the
  free page pool exhausted for the file's lifetime; 2.9–4 GiB/s on files the
  page cache absorbs whole). Warm read plus eval: 12–14.5 GiB/s. A cold read
  after a reboot was not measured anywhere and is owed by the gate. The
  kernel's `kern.memorystatus_vm_pressure_level` stayed at 1 through a
  3.5 GiB write on the idle Air; on a loaded 64 GB daemon that is unmeasured.
- **The safetensors header is plain JSON**: 8 bytes of length, then the
  header. Parsing it in pure Python takes 41 µs; `mx.load(...,
  return_metadata=True)` takes 202 µs and creates thread-bound lazy arrays.
- **The two-step id check is cheap**: blake2b-16 over 57K ids 0.94 ms; an
  exact `loaded_ids.tolist() == ids` 1.5 ms (versus 30 ms through `mx.all`).
- **`mx.save_safetensors` appends `.safetensors`** to a name that lacks it.
- **int8 prefill changes the KV**: it routes the MLP trio and GatedDeltaNet's
  own `in_proj_qkv`/`in_proj_z`, so every later layer's K/V and the recurrent
  state differ. `int8prefill.enable` degrades to `unavailable` rather than
  raising, so the *realised* state, not the config flag, is what identifies
  a load.
- **mlx-vlm's APC exact-cache store is not reusable**: +100 % transient on
  write, 4.0 GiB/s restore against 14.1 for `mx.load` (it reads each tensor
  with `open`/`read`/copy), everything beyond three method names is private
  and was rewritten between 0.6.17, 0.7.0 and 0.7.2 with the on-disk
  `exact_cache_v1` string untouched, and construction spawns daemon threads
  that clear their mlx streams only on an explicit `close()`. Its per-layer
  encoding conventions are worth copying; nothing else is.

## Goal

A tools fork survives the weights. A daemon that has served one turn with a
given tool array serves the first turn of every later session presenting it
warm — across idle unloads, daemon restarts, upgrades that leave the engine's
numerics alone, and reboots — paying seconds of SSD read instead of two to
four minutes of prefill. Measured against the log: the six repeat-array misses
become `cache=disk took=disk@~50K` turns prefilling 574–2,511 tokens plus the
brief, with `restore_s` in low single digits.

The same mechanism gives a machine that cannot afford resident forks
(`prompt_cache_gb = 0`, the README's advice for 48 GB) a warm cold-start it
has never had: the disk copy is written from the live cache whether or not a
resident copy fits.

## Non-goals

- **The header boundary on disk.** Measured at ~7.6 s over nine days against
  ~831 s for the tools boundary, and at 52–59 GiB for the maintainer's own
  workload if both were kept. The header fork stays resident-only, exactly
  as today. Revisit only if a workload appears whose tools arrays recur far
  less than its (tools, system) pairs.
- **Turn slots on disk**, and the continuity spec's append-only chunk
  mirroring of conversations. Conversation state has a short half-life; the
  log puts the value entirely in the shared boundary.
- **A `sous forks` command.** `rm -rf ~/.sous/forks` is the eraser, the
  daemon tolerates it under its feet (below), and the store's INFO line at
  construction says where it is and what it found. File one if the #99
  experience recurs.
- **A second-process lock.** The daemon is singleton by `daemon.lock`, and
  `sous tune` gets no store (below), so no second writer exists. An
  `flock` around the eviction sweep is a one-liner if one ever does.
- **`fsync` before the rename.** A 3.5 GiB fsync is seconds on the turn's
  critical path. The panic window is accepted and bounded: a truncated file
  fails the size check, a zeroed one fails the exact ids check, both delete
  the file. A partial write that leaves the ids intact and loses KV blocks
  is the residual risk, stated here.
- **Eager restore at model load** (the load thread releases its mlx streams
  and exits; its arrays would be unusable, and it would read every file
  instead of the one the turn needs), **a deferred write after decode**
  (after decode the cache is back at the anchor — `trim_to`/`restore` —
  and at `prompt_cache_gb = 0` there is no resident slot either, so in the
  case the feature exists for there is nothing to write), and **reusing
  mlx-vlm's `DiskBlockStore`** (above).
- **KV quantization of the file** (#90). The format records the dtype and
  the epoch changes when #90 lands.
- **Waking `/sous/events` for a persist or restore.** Both happen inside a
  turn; the document catches up when the turn retires.

## Design

### 1. What is written, and when (`engine/promptcache.py`)

At each fork boundary a turn stops at, the live cache holds exactly the KV
the boundary's ids produce. Today the loop in `_run` prefills to the
boundary, then charges, copies and publishes a resident fork. The change:
**before the resident copy, and independently of it, the live cache is
written to disk when the store has no file for those ids.**

- `_fork_boundaries` returns, per boundary, two flags instead of one
  decision: `need_slot` (this owner holds no resident fork with those ids —
  today's rule) and `need_file` (the store's index has no file for them).
  A boundary is returned when either is true. Its early return
  `if self.max_bytes <= 0: return []` becomes `if self.max_bytes <= 0 and
  self._store is None`, so a budget-0 machine still resolves the probe and
  still writes. The `reuse < b` filter becomes `reuse <= b`: a turn that
  starts exactly at a boundary — a disk restore, or a fork lost to a
  pressure eviction — republishes a resident fork there through the
  existing charged path (`hooks.prefill` with an empty list is a no-op on
  both backends), so residency heals with no new publish path.
- Only the **lowest** boundary that clears `FORK_MIN_TOKENS` is ever written
  (`need_file` is false for every other). On the default template that is
  the tools boundary; on a template that renders the system text first it
  is the header, which is then the only boundary anyway. One file per turn.
- In `_run`, after `hooks.prefill(cache, stable_ids[reuse:boundary])` and
  `reuse = boundary`: the prefill timer is closed, `self._persist(stats,
  cache, stable_ids[:boundary])` runs under its own timer into a new
  `persist_seconds` turn gauge, and the prefill timer reopens before `price
  = slot_bytes(cache)`. `_persist` has its own `try`/`except`/`warnings.warn`
  ("fork persist failed (…); continuing") — mandatory, because on a cold
  turn `generate` re-raises rather than retrying, and a disk error must
  never turn a viable 170 s prefill into a 500. It is charged to no budget:
  the live cache is the turn's own, covered by `reserve_bytes`.
- The persist is **not** gated on owner retirement. A stalled session's
  thread reaching a boundary writes a valid file: retirement exists because
  its *arrays* live on a dead thread's streams, and a file has no thread.
  This is the one case where the disk copy is strictly better than today.
- `hooks.persist(cache, path, ids, meta)` — a new `CacheHooks` method — is
  called only with the live working cache, at a boundary, on its owner
  thread, holding no lock. Never from `_publish`, the pressure valve,
  `reset()`, or with a `Slot` (whose arrays are lazy copies).

An unload cannot land mid-write: `EngineManager._refusal` refuses while a
generation is in flight, and the persist runs inside one.

### 2. File format (`engine/forkstore.py` rules, engine hooks for the arrays)

One safetensors file per fork, `<data_dir>/forks/<key>/<n>-<digest>.safetensors`,
where `n` is the token count and `digest` is blake2b-16 over the ids as
little-endian int32 bytes. Tensors:

- `ids`: int32[n], the exact boundary ids.
- Per layer `i`: for a `KVCache`, `c{i}_k` and `c{i}_v` — the **full padded
  buffers** (`c.keys`, `c.values`, at most 255 tokens × 64 KiB = 16 MiB of
  padding), with `c{i}_offset` in metadata. Never the `state` slice: that is
  the transient the probes measured. For an `ArraysCache`, `c{i}_s{j}` per
  state array, with `c{i}_s{j}_none = "1"` in metadata for an absent one (a
  missing key must mean corruption, never "None"). Any other layer class
  refuses the persist (the engine returns False, one warning the first
  time): the two classes above are what `make_prompt_cache` builds for every
  model sous serves, and a `RotatingKVCache` cannot be forked in any case.
- Metadata: `format = sous-fork-v1`, every identity-key field (§3) spelled
  out, `n_tokens`, `boundary` (`tools`|`header`), the layer-kind list,
  `file_bytes` (the expected total, checked with `stat` at scan and at
  restore — a free truncation guard), `created` (ISO 8601), the sous
  version.

The engine's `persist` first `mx.eval`s the arrays it is about to write
(free when they already are — a lazy graph would otherwise be evaluated
inside the timer), writes to `<final>.<pid>.tmp.safetensors` in the same
directory (the suffix is mandatory: mlx appends `.safetensors` to a name
without it), then `os.replace`. A failed save unlinks its temp in a
`finally`. Files are created 0o600 in a 0o700 directory that also holds a
zero-byte `.metadata_never_index`.

Bytes beyond `offset` in a padded buffer are never read on restore. They may
hold KV of the previous turn's generated tokens when the cache was adopted
from a moved turn slot; the security note in the README says so.

### 3. Identity key

The key is *everything that changes the KV arrays computed for identical
token ids*, and nothing else. Ids are compared exactly at restore, so the
chat template, tokenizer, sampling settings, drafter (prefill passes none)
and context window are deliberately absent: a template change is a natural
miss, and none of the others touches a prefill.

Fields, each read once at engine construction and memoised:

1. `backend`: `vlm` | `lm`, and that package's version
   (`importlib.metadata.version("mlx-vlm")` or `("mlx-lm")`; mlx-lm prefills
   in 2048-token chunks where mlx-vlm does not, so their KV can differ).
2. `mlx`: `importlib.metadata.version("mlx")` (mlx has no `__version__`).
3. `gpu`: `mx.device_info()["architecture"]` (e.g. `applegpu_g14g`). A
   `~/.sous` copied between Macs must not serve one GPU's keys to another.
4. `weights`: for a Hub id, the snapshot commit — `fetch_model_config`
   already downloads `config.json`, and that path's parent directory *is*
   the commit sha, so it costs no second Hub round-trip and works offline
   once cached. For a local path, blake2b over `config.json`'s bytes plus
   the sorted `(name, size, mtime_ns)` of every `*.safetensors` in the
   directory — a re-quantisation into the same directory at the same bits
   leaves `config.json` byte-identical, so `config.json` alone cannot see
   the weights change. Quantization rides inside either form.
5. `epoch`: **derived, not remembered.** blake2b-8 over the bytes of the
   modules that decide ids → KV — `engine/vlm.py`, `engine/lm.py`,
   `engine/int8prefill.py`, `engine/kernels/*.metal` — read once at import,
   plus a manual `FORK_LAYOUT = 1` integer that changes only when the file
   format does. Persistence removes the safety net the #99 note in CLAUDE.md
   leans on ("only a daemon restart drops them"): a sous-side positions fix
   with the versions and weights unchanged would otherwise be restored
   after every restart, forever, by a build whose author forgot a constant.
   A source-derived epoch invalidates exactly when the numerics could have
   changed (a formatter pass costs one cold start; a semantic change can
   never be missed) and never on the "daemon reinstalled from main" cadence
   the sous version would.
6. `positions`: `engine` | `model` (`VLMEngine.positions`). Redundant with
   (model family, mlx-vlm version) in principle, but the probe returns
   `model` on any exception, so one bad load would otherwise share its key.
7. `int8`: `(int8_prefill_status["state"] == "active", routed)` — the
   realised state, not the config flag; `off` and `unavailable` are the same
   numerics.
8. `env`: `MLX_ENABLE_TF32` and `MLX_SDPA_BLOCKS` when set (mlx-core reads
   both; the first changes matmul precision, the second attention
   accumulation order). Absent from the key when unset.

The directory name is blake2b-16 over the joined fields; the fields are
written into every file's metadata and once into `<key>/key.json` for
forensics. The layer-kind list and a future KV-quantization field are *not*
in the key: both are functions of `config.json` and the epoch; the
layer-kind list is verified at restore instead.

### 4. Restore (`engine/promptcache.py` `generate`)

On a miss — after `_take` found nothing this owner holds and after the
pre-turn `_evict_caps` pass — the miss branch is **replaced**, not extended
(its `stats.misses += 1` is what makes the turn line read `cache=miss`):

1. **Lookup** (pure Python, forkstore): for each stored length `n` in the
   current key's index, descending, with `0 < n < len(stable_ids)` (the
   `reuse_length` rule: an exact match leaves nothing to decode and is a
   miss), compare blake2b-16 of `stable_ids[:n]` with the file's digest.
   Under 8 ms for a store of eight lengths. No candidate: the existing miss
   code runs verbatim.
2. **Room.** If `hooks.headroom()` is known and below the file's bytes, drop
   least-recently-used resident slots (nothing is protected — a miss took
   nothing) one at a time until it fits or none remain, the shape of
   `_evict_pressure`'s Metal half. If it still does not fit, or
   `hooks.pressure()` is at `KERNEL_PRESSURE_CRITICAL`, skip the restore:
   plain cold miss, file kept, `restore_skips` incremented. The restore is
   the one place the design allocates a whole fork in one step; the cold
   path grows the same bytes over minutes with the valve consulted at each
   publish.
3. **Guard.** If this turn's fork clone just failed (`_copy_of` returned
   None), skip the restore: that was an allocation failure of the size
   about to be requested again.
4. **Restore** into the turn's **own working cache**: `hooks.new_cache()`,
   then `hooks.restore(path, cache, ids)` on the turn's thread — `mx.load`
   (lazy), assign `c.keys`, `c.values`, then `c.offset = n` from metadata
   for every `KVCache` (never the `state` setter), `c.state = [...]` for
   every `ArraysCache`, then `mx.eval` everything so the arrays are owned by
   the thread that will use them and visible to `mx.get_active_memory()`
   before the valve next reads it. Verify: `format` and key fields match,
   `file_bytes` matches `stat`, layer count and kinds match `new_cache()`,
   every `KVCache` has `offset == n` and `keys.shape[2] >= n`, and
   `loaded_ids.tolist() == stable_ids[:n]`.
5. **Account** as a hit that came from disk: `hits += 1`, `disk_hits += 1`,
   `reused_tokens += n`, `took_len = n`, `took_kind = "disk"`,
   `restore_seconds` set. `misses`, `miss_lcp`, `fork_hits`, `retained`,
   `moved` untouched. `reuse = n`; `_run` proceeds exactly as after a fork
   take. The cache is never planted: it is the turn's own, uncharged,
   covered by `reserve_bytes`. Residency comes from §1's `reuse <= b`: the
   boundary loop republishes a resident fork through the charged path, so
   the machine holds live plus copy (as on any fork take today), never
   restored plus planted plus copy.

Failure handling, by cause:

- A verification failure (format, key, size, layer, offset or ids mismatch,
  a safetensors decode error): warn once, **delete the file**, cold miss.
- `MemoryError`, an mlx `RuntimeError`, `OSError`: warn, **keep the file**,
  cold miss. On a loaded 48 GB machine the likeliest restore failure is
  memory, and one pressure event must not delete the artefact worth
  minutes.
- A file missing from disk (`rm -rf` under the daemon): drop the index
  entry, plain miss, no warning.
- The prefill that follows a successful restore fails: `reuse > 0`, so the
  existing cold-retry path runs; the file is kept; the retry's
  `begin_turn()` zeroes `restore_seconds` like every other turn gauge, so
  the line correctly reads `cache=miss took=none` — the counters below
  survive.
- Three consecutive verification failures in one daemon life disable the
  store (`state: unavailable`, reason recorded): a store whose files all
  fail would otherwise delete itself one 170 s turn at a time.

Ordering invariant, stated rather than accidental: the restore runs before
`_fork_boundaries` resolves the probe (`_run` calls it after `generate` has
fixed `reuse`), so a turn warm at a restored tools fork still publishes its
header fork — the "warm turns resolve the probe too" rule holds unchanged.
A restored fork is used whole or not at all; nothing trims it (the hybrid
rule).

### 5. Store rules (`engine/forkstore.py`, mlx-free)

The module imports no mlx, like `promptcache.py`, so every rule is testable
in CI with fakes. It owns: the identity key rendering, the directory index,
the digest and longest-prefix rule, the byte budget and LRU, naming,
temp/rename discipline, the pure-Python header parse (refusing a header
length above 1 MiB before allocating — a 57K-id file's header is 12.8 KB),
the free-space floor, and the in-flight bookkeeping. The engines own every
array: `persist`, `restore`, the exact ids comparison, `eval_cache`.

- **Construction** (on the model-load thread, touching no mlx): `mkdir` the
  root; sweep `*.<pid>.tmp.safetensors` whose pid is not alive (psutil, as
  `EngineManager` prunes holders); one `os.scandir` across **every** key
  directory collecting `(path, size, mtime)` for the budget and LRU
  (microseconds per file, no header reads); header parses for the
  **current** key's files only, for lookup, skipping any whose `format`,
  key fields or `file_bytes` do not match (deleted). Enforce the budget
  here too, so lowering the knob takes effect at the next start. One INFO
  line: `prompt-cache forks on disk: <dir> budget=16.0GiB found=3 (9.4 GiB)`.
  A root that cannot be created or written (`EACCES`, `EROFS`, a full disk)
  disables the store here with one warning, never on a turn.
- **Budget**: `[model].prompt_cache_disk_gb`, the `prompt_cache_gb` shape —
  `"auto"` (default), a number of GiB, or `0` (off). `auto` is
  `min(16 GiB, 25 % of the volume's free bytes at construction)`
  (`shutil.disk_usage`): 16 GiB holds four to five tools forks of the
  default model, more than the four arrays that carried 52 of 71 turns in
  the log. Before each write: skip it, warning once, if it would leave the
  volume with less than `max(10 GiB, 5 % of its size)` free — a skip, not a
  disable; the store stays readable.
- **Eviction**: at write time, least-recently-used by mtime across every
  key directory, down to 0.9 × the budget so a write near the cap does not
  sweep every turn. Stale keys (a model switch, a version bump, an epoch
  change) age out this way; nothing deletes them eagerly, so a switch and
  back keeps its forks. `disk_evictions` counts the files. A path in the
  **restoring set** (below) is never evicted.
- **Touch**: `os.utime` on every restore *and* whenever a resident fork
  serves a turn (the file of a fork that is resident is otherwise the oldest
  in the store, and the first out). `_fork_boundaries` touches the file of
  every boundary in `wanted` whose file exists; a fork take touches its
  slot's digest.
- **In-flight sets**, under the store's own short lock, never held across a
  write, a read or a status build: digests being written (a second writer
  of the same digest skips, never waits — the endpoint session thread and
  its post-stall replacement can both prefill the same array) and paths
  being restored (skipped by eviction; this is what makes the probe's
  unexplained delete-during-second-load anomaly unreachable).
- **`disk` counters** are two plain ints (`forks`, `bytes`) the store
  updates under its lock and `PrefixCache.stats()` reads without one,
  beside `slots`/`resident_bytes`, so no status reader ever waits behind a
  2 s write and `without_turn_gauges` never sees them.
- **`rm -rf ~/.sous/forks` under a running daemon**: the store `mkdir`s
  before every write, treats `ENOENT` as a normal outcome everywhere, and
  disables itself only on `ENOSPC`, `EDQUOT`, `EACCES`, `EROFS`.

### 6. Wiring and switches (`engine/base.py`, `config.py`, `tune/`)

- Both engines take `fork_store: ForkStore | None = None`. **Off unless
  given.** The only place that builds one is `default_engine_factory`, from
  `config.data_dir / "forks"`; `_default_factory` gains `fork_dir: Path |
  None = None`. No module-level default path anywhere: every direct engine
  construction — the whole test suite, the model tests on the maintainer's
  own Mac — must be unable to reach `~/.sous`.
- `sous tune` passes `fork_dir=None` at both of its factory sites
  (`tune/suite/runner.py`, `tune/bench.py`). This is correctness, not
  hygiene: an `Arm.config` inherits the user's `data_dir` verbatim, `sous
  tune` takes no daemon lock, and its arms differ in model, drafter and
  int8 — each a key of its own writing synthetic forks into the live
  store's LRU, with a 2 s write landing inside the prefill the arm is
  timing. A test builds an arm's engine and asserts the store is absent.
- The store is not constructed when `prompt_cache = false` (`generate`
  returns before anything here runs), when `prompt_cache_disk_gb = 0`, or
  when `reserve_bytes == 0` (KV cost per token unknown: the pressure valve
  is off, and the store must not be the one thing allocating 3.5 GiB at a
  stroke on a model the daemon cannot size — the existing warning gains
  "and prompt-cache forks are not persisted"). `load_config` warns when
  `prompt_cache_disk_gb` is set beside `prompt_cache = false`.
- `prompt_cache_disk_gb` joins `_KNOWN["model"]`, gets a validator beside
  `_prompt_cache_gb` (same rejections: bool, non-finite, negative), a
  `SousConfig` field, a line in the README's sample config, and a row in
  the status document's `config` block.
- `count_tokens` never reaches the store: it tokenises only and never
  enters `PrefixCache.generate`.
- `VLMEngine.unload` / `reset(None)` drop every slot and leave the store
  alone. That is the feature.

### 7. Observability

- **Turn line**: `cache=disk took=disk@N` — a new `TurnResult.from_disk`
  from a new `disk_hits` counter, `cache = "disk" if from_disk else "fork"
  if forked else "hit" if cache_hit else "miss"`, so `cache=fork` keeps
  meaning "a resident fork was copied" and the log series stays comparable.
  Two new fields, `persist_s=` and `restore_s=`, from two new `TURN_GAUGES`
  (`persist_seconds`, `restore_seconds`), never folded into
  `prefill_seconds`: a tools-fork restore prefills ~574 tokens in ~1.5 s,
  and a restore folded in would misreport `prefill_tps` by ~40 % on the one
  turn that shows the feature working. A disk-restore turn will also print
  `forks=1` — it consumed a file and republished a resident fork; the
  README says so.
- **Counters** (sum daemon-wide, survive a cold retry): `persists`,
  `restores`, `disk_hits`, `disk_evictions`, `restore_skips`. No
  `restore_failures` in `as_dict()`: a failure already warns and counts a
  miss; the store keeps its own count for the disable rule.
- **Status document**: `prompt_cache.disk = {state: off|unavailable|active,
  reason, forks, bytes, budget_bytes}` — the `int8_prefill` shape, so
  "off by config", "disabled after an IO error" and "never constructed"
  stop being one indistinguishable zero. `sous status` prints one line
  (`forks on disk: 3 · 9.4 GB` or the `unavailable` reason); `sous top`
  extends its wide `PROMPT CACHE` line with ` · disk {forks} · {gb:.1f} GB`
  (the panel's line count is fixed — a new row would push the tickets block
  off) and leaves the compact layout alone; `sous statusline` is unchanged.
- **Log lines**: the one INFO line at construction (§5), which is also the
  first-run disk notice. None per turn: the turn line is the single source
  of per-turn facts, and `promptcache.py` emits `warnings.warn` only.

### 8. Peak memory, stated

`kv_bytes_per_token` for the default model is 2 × 16 layers × 4 heads × 256 ×
2 B = 64 KiB. A 50K-token tools fork is ~3.1 GiB padded, a 57K header fork
~3.5 GiB. `reserve_bytes` is one window: 8 GiB at 131072, 16 GiB at 262144;
the README's auto `max_bytes` on a 64 GB machine is ~27 GiB and ~19 GiB
respectively. A restored working cache (≤ 3.5 GiB) is uncharged and inside
the reserve; its republished resident copy is charged against `max_bytes`
and made room for first. A persist allocates nothing beyond the live cache
when the arrays are evaluated and contiguous, which §2 guarantees. The new
`CacheHooks.eval_cache(cache)` — `mx.eval` over every layer's state, owner
thread, no release — is called after `fork_copy` in `_copy_of` and in
`_run`'s loop, and after `hooks.restore`, so `slot_bytes`, `max_bytes` and
`live_headroom` describe the same bytes: today the valve at a cold turn's
publish believes in ~7 GiB of headroom that two lazy forks have already
claimed. The eval is work the fork pays anyway on first use.

## Invariants preserved

- Slots are owned by the thread that built them and looked up only by it
  (#34). The disk store shares *files* across owners — the endpoint's
  session thread, its replacement after a stall, any future owner — while
  every restoring thread allocates its own arrays on its own streams. The
  invariant is about arrays, not content; `base.py` already states the
  content rule (ids published with the slot, strict prefix match). The
  `PrefixCache` and `Slot` docstrings gain this sentence, or the next reader
  files the disk path as a bug.
- Never rewind a cache to make a slot: a restored fork is used whole,
  `reuse = n` exactly, nothing trims.
- The bookkeeping lock is held only across list and dict work. The persist,
  the restore, the digest scan and the header parses all run outside it;
  `stats()` reads two plain ints.
- No new thread. Persist on the owner thread inside `_run`; restore on the
  turn's thread inside `generate`; the directory scan touches no mlx.
- Charged before allocated: the restored cache is the turn's own; the
  resident copy goes through `_make_room` as every fork does.
- A cold turn's `generate` re-raises: every disk operation has its own
  handler and can only warn.
- `promptcache.py` and `forkstore.py` import no mlx; every mlx import in
  the engines stays function-local.
- The API never logs a request body; the store writes token ids and KV to
  a 0o700 directory the README describes, and nothing about them to the
  log.

## Testing

**`tests/test_forkstore.py` (pure Python, fakes):** header parse of a
hand-built safetensors header (8-byte length + JSON) and refusal above
1 MiB; digest and longest-prefix (descending, `0 < n < len(stable_ids)`, an
exact-length file is not a candidate); the identity-key rendering and that
a change in any field changes the directory; the derived epoch changes when
a source byte does; LRU across two key directories to 0.9 × budget with
the restoring set protected; the free-space floor as a skip, not a disable;
`auto` budget arithmetic; temp-file sweep only for dead pids; `mkdir`
before write and `ENOENT` as a plain miss; `ENOSPC`/`EACCES`/`EROFS`
disable, `ENOENT` never; the in-flight digest skip; `disk_evictions` and
the disable-after-three rule.

**`tests/test_promptcache.py` (FakeStore, FakeTrimmable/FakeRecurrent):**
the persist runs at the lowest boundary only, once per digest, before the
resident copy, outside the prefill timer, and a raising `persist` hook does
not fail a cold turn; `_fork_boundaries` returns a boundary when either
flag is set and `reuse <= b` republishes at a restored boundary; the miss
branch becomes a disk hit with exactly `hits`, `disk_hits`, `reused_tokens`,
`took_kind == "disk"` moving and `misses` not; the room rule drops LRU
slots before a restore and skips at critical pressure; the restore is
skipped after a failed clone; the file is deleted on a verification failure
and kept on `MemoryError`/`OSError`/a failed follow-on prefill (which
cold-retries); a retired owner still persists; `restore_seconds` and
`persist_seconds` are in `TURN_GAUGES` and the counters are not (the two
tests that pin the field sets are updated); `stats()` reports `disk` ints
without taking the store's lock; the touch on a resident fork take.

**`tests/test_engine_forkstore.py` (real `mlx_vlm.models.cache` classes,
tiny arrays, runs in CI on macos-15):** a `KVCache` built to an offset that
is not a multiple of 256 plus an `ArraysCache(size=2)`, persisted through
the engine hook and restored into fresh caches: bit-exact keys, values and
states, `offset` from metadata (not the padded shape), and one further
`update_and_fetch` producing identical arrays *and* identical capacity to a
never-serialised twin; a file whose ids tensor is altered fails
verification; `mx.save_safetensors`'s appended suffix is handled; a
`RotatingKVCache` layer is refused.

**`tests/test_engine_base.py`, `tests/test_config.py`, `tests/test_api_turn.py`,
`tests/test_api_routes.py`, `tests/test_tune_*.py`:** the factory builds a
store only from `default_engine_factory` and a tune arm's engine has none;
the knob's validator and the `prompt_cache = false` warning; `from_disk`
reaches the turn line as `cache=disk took=disk@N persist_s= restore_s=`;
the status document's `disk` block in all three states.

**Model tests (local, `@pytest.mark.model`):** on the hybrid and the
pure-attention test models, prefill a header-shaped prompt, persist at a
boundary, drop the live cache, restore, and assert `mx.array_equal` on
every `KVCache` key/value slice `[..., :offset, :]` and every `ArraysCache`
array, plus `offset` and layer kinds — a serialisation-identity claim, so
every layer must match exactly. Then one further chunk on the restored
cache and on the source cache, compared on cached keys: every layer on the
pure-attention model, the first attention layer on the hybrid (the #99
witness rule). **Never** greedy text, and never against a cold one-pass
prefill: a split prefill drifts on a hybrid's deeper layers, and the
budget-0 change makes every cold turn a split.

**Live gate on the M5 Pro (before merge, recorded in the PR), five arms:**

1. Cold: a daemon restart *after a reboot or `sudo purge`*, then a new
   session's first subagent turn logs `cache=disk took=disk@~50K` with the
   genuinely cold `restore_s` — the number the README prints.
2. The write turn: `prefill_tps` within 5 % of the log's cold-miss baselines
   (395 tok/s at 44.6K, 366 at 63.6K) and `persist_s` recorded; sample
   `kern.memorystatus_vm_pressure_level` and `vm_stat` before the persist,
   at its end and at the publish that follows, with the weights loaded and
   two resident forks. If the level moves to warn, the smallest fix is to
   skip the kernel half of `_evict_pressure` for the publish that follows a
   persist in the same turn (never the Metal half, and never `F_NOCACHE`,
   which would forfeit the warm restore).
3. A second restart: `persist_s=0.0` on the first turn, the file's mtime and
   inode unchanged (the dedupe holds), `forks=1` (residency republished).
4. `prompt_cache_gb = 0`: the configuration the write path exists for and
   the one the 64 GB machine never runs — `took=disk@N` with nothing
   planted, and the probe's per-turn cost visible in `tokenize_s`.
5. `rm -rf ~/.sous/forks` under the running daemon, then one cold turn:
   the directory reappears and a file is written; `sous status` shows
   `active` throughout.

## Documentation

- README "Prompt cache and continuity": the two "Forks live as long as the
  weights" sentences (the section and the `prompt_cache_gb` paragraph) are
  rewritten — the tools fork now lives on disk under the knob below, the
  header fork with the weights; the turn-line paragraph gains `cache=disk`,
  `took=disk@N`, `persist_s=`, `restore_s=` and their exclusion from the
  phase partition; "Configuration" gains `prompt_cache_disk_gb` with the
  `auto` arithmetic, the free-space floor, `0`, and `rm -rf ~/.sous/forks`
  as the eraser; a sentence recommending a Time Machine exclusion for
  `~/.sous/forks`.
- README "Security model": sous now keeps data derived from prompts at
  rest — the rendered tool block and system text of the sessions that
  produced a fork, as token ids and as KV, unencrypted, under a 0o700
  directory, cleared by `rm -rf ~/.sous/forks` or `prompt_cache_disk_gb =
  0`. CLAUDE.md's "Security boundary" gets the same bullet.
- CLAUDE.md gotchas: the #99 sentence "only a daemon restart drops them" is
  replaced with what the epoch guarantees (a change to the engine sources
  invalidates every file; a version or weights change does too); the
  fork-slot paragraph gains the disk rule (lowest boundary only, live cache
  written before the resident copy, restore on the turn's thread, files
  shared across owners while arrays stay owner-scoped); the mlx gotcha
  gains "restore `KVCache` by assigning `keys`/`values`/`offset`, never
  through `state`" and "save only evaluated, contiguous buffers".

## Risks and open questions

- **The cold read is unmeasured.** Every restore number anywhere in this
  document is page-cache warm. After a restart the file usually is too;
  after a reboot it is not. Gate arm 1 is the number.
- **Write throughput on the M5 Pro is unmeasured** and the M2 figure (1.73
  GiB/s at full size, 2 s per 3.5 GiB) already halves the draft's estimate.
  Two seconds on a 113–272 s cold turn is noise; on a 15 s warm turn that
  writes its first tools file it is 13 %, once per tool array.
- **The kernel pressure level during a full-size write on a loaded 64 GB
  daemon** is unmeasured (gate arm 2). A single `warn` reading at the
  publish that follows would cost one slot per publish — the feature
  evicting the residency it protects. The fix is named above and small.
- **The derived epoch** invalidates on any byte change to four engine
  files, including comments and formatting. That is the intended trade: a
  cold start per engine edit, never a silently stale fork. If it proves too
  eager in practice, hash the AST instead of the bytes.
- **The probe now resolves on every turn at `prompt_cache_gb = 0`** — four
  renders and memoised encodes under the tokenize lock, in `tokenize_s`.
  Small, new, and on the configuration the write path is aimed at; gate
  arm 4 records it.
- **A 16 GiB `auto` cap** is a judgment: four to five forks of the default
  model, sized to the log's dominant arrays. A user with more tool arrays
  raises it. The alternative considered was 32 GiB with a 20 GiB floor;
  16 with a `max(10 GiB, 5 %)` floor was chosen because the feature is on
  by default on a disk sous otherwise knows nothing about.
- **Tools boundary only** is a measured call against one maintainer's log
  (7.6 s marginal for the header). A workload whose tool arrays churn while
  its system text holds would move it; the `boundary` field in metadata and
  the `need_file` rule make adding the header a one-line change.
