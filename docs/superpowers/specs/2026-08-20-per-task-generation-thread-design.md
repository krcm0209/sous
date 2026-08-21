# sous — Per-Task Generation Thread Design Spec

*2026-08-20. Addresses issue #34. Companion to the 2026-08-19 prompt-cache-reuse
spec, which built the cache machinery this spec lets engage.*

Issue #33 landed cross-turn prompt cache reuse, but shipped it default-off: on
the real daemon path every warm generation fails and retries cold. The cause is
thread shape, not cache logic. `_generate_with_timeout` runs every generation on
a fresh daemon thread, and each thread destroys its mlx streams at exit — a
mandatory step per ml-explore/mlx#4327. The KV cache arrays are bound to the
streams of the thread that created them, so the next turn's new thread cannot
touch them.

This spec moves all of a task's generations onto one thread, so the cache and
the thread have the same lifetime. Nothing about the cache itself changes.

## Why a per-task thread is required, not merely chosen

The obvious alternative — keep per-generation threads but pin every generation
to one long-lived shared stream — was probed empirically against this repo's
mlx 0.32.0 and fails at the first operation: mlx streams are thread-scoped for
*use*. A stream created on one thread cannot be used from another, with either
`with mx.stream(s)` or a per-op `stream=` argument. The only unit that can hold
the cache alive across turns is the thread itself.

## Goals

- A real delegated task on the default model, with `prompt_cache = true`,
  reports `cold_retries: 0` and non-zero `reused_tokens`.
- The stall path still ends the task rather than hanging the daemon.
- No thread that touched mlx exits without `release_mlx_thread_state()`.
- The late-waiter identity leak (issue #34, comment: consideration 7) is closed
  where the lock lives.
- Cache reference ownership has one deliberate answer (consideration 4), not
  per-path luck.

## Non-goals

- Any change to `PrefixCache`, the strict-prefix rule, the epoch guard,
  snapshot/restore, or the stats shape. Consideration 5 in the issue: this is
  only about where the generation runs.
- Process isolation for wedged generations (README limitation, unchanged).
- Cross-task cache reuse.

## Core decisions

| Decision | Choice |
|---|---|
| Thread lifetime | One daemon thread per task, owned by a new `GenerationSession`. |
| Session owner | `ManagedEngine.session()` — identity lives where `_gen_lock` lives. |
| Per-turn deadline | Request queue + reply queue; the reply wait carries the turn's timeout. |
| Stall response | Set `abandoned`, raise `GenerationStalled`, never wait on the thread again. |
| Late-waiter fix | The thread re-checks `abandoned` **after** it acquires `_gen_lock`. |
| Reset ownership | The worker thread, exactly as today: once at task start, once in the `finally`. |
| `CLOSE` semantics | Ends the session loop, nothing else. Sent unconditionally, never joined. |
| Queue discipline | `put_nowait` on both maxsize-1 queues: an invariant break crashes loudly. |
| mlx release | Once per session thread, in its `finally`. |

## The new shape

### `GenerationSession` (src/sous/engine/base.py)

`ManagedEngine.session()` creates one per task. It owns a daemon thread
(started eagerly, handle kept on the session so tests can join it), a request
queue (maxsize 1), a reply queue (maxsize 1), and an `_abandoned`
`threading.Event`.

The thread loop:

1. Block on the request queue.
2. On `CLOSE`: exit the loop. The `finally` releases mlx thread state.
3. On a generation request: acquire `_gen_lock`. **Under the lock**, check
   `_abandoned`; if set, exit without generating. This closes consideration 7:
   a request abandoned while queued on the lock can no longer run under the
   next task's identity, because abandonment is checked at the moment and
   place the lock is granted.
4. Run `inner.generate(...)`. Catch `BaseException`: a `KeyboardInterrupt` must
   relay to the worker (existing restart-recovery contract), never kill the
   loop silently.
5. After the lock is released, check `_abandoned` again; if set, drop the
   result and exit. Otherwise `put_nowait` the reply, then rebind the local to
   `None` so a parked thread never pins a reply payload (an `("err", e)` reply
   would otherwise pin the generation's KV cache through the traceback).

`session.generate(messages, tools, max_tokens, timeout)`: `put_nowait` the
request, wait on the reply queue with the caller's timeout. On `queue.Empty`:
set `_abandoned`, raise `GenerationStalled`. An `("err", e)` reply re-raises
`e`. One request is in flight at a time, and a session is never used after
`close()`; the session asserts both.

`session.close()`: `put_nowait(CLOSE)`, unconditionally, and return. No join,
no return value.

- Healthy thread: it is idle on the request queue, dequeues `CLOSE`, releases,
  exits.
- Abandoned-while-idle (the reply landed inside the timeout race window): same
  — `CLOSE` is what un-leaks it. Everything it pinned becomes garbage.
- Wedged thread: never dequeues `CLOSE`; when and if it unwedges, the
  post-lock or post-generation `_abandoned` check exits it. Until then it is
  leaked deliberately, exactly like today's abandoned generation threads, and
  `_gen_lock` keeps the next task off the engine. A leaked thread never
  exits, so it never triggers the TLS segfault.

Because `CLOSE` carries no reset, sending it to an abandoned session is safe:
there is no late cache or stats mutation to race the next task.

`GenerationStalled` moves from `worker.py` to `engine/base.py`; the worker
imports it, so `worker.GenerationStalled` keeps resolving.

### `run_task` (src/sous/worker.py)

- The `engine` parameter becomes `ManagedEngine`. `EngineManager.get()` already
  returns one; its annotation narrows to say so. Tests wrap `FakeEngine` in
  `ManagedEngine`, which is what production always did.
- `session = engine.session()` after the start-of-task reset.
- Each turn calls `session.generate(...)` with the remaining wall-clock budget
  as the timeout. `_generate_with_timeout` is deleted.
- The `finally` becomes `session.close()` followed by the same unconditional
  `engine.reset_prompt_cache()` as today. Reset count and ownership are
  unchanged: two per task, both on the worker thread.

### Reset ownership: the deliberate answer to consideration 4

The #4327 hazard is a thread that touched mlx *exiting* without clearing its
streams — TLS teardown. Cross-thread array *deallocation* is a different
operation: allocator-level buffer release, no stream use. It is also today's
shipped behavior — `run_task`'s `finally` already drops, on the worker thread,
cache arrays created on generation threads, on every cache-enabled task, and
never failed. The design therefore keeps one reset owner (the worker thread)
on every path, instead of splitting ownership per path. Two backstops make the
worst case contractual rather than lucky: the worker loop thread already calls
`release_mlx_thread_state()` at exit, and the real-model verification below
exercises exactly this teardown on the default model.

### What deliberately does not change

- `ManagedEngine.generate()` (the lockful direct path) stays — it is the
  `Engine` protocol and the semantic definition of the lock.
- `count_tokens` runs on the worker thread. It is pure tokenizer work with no
  mlx arrays.
- `PromptMemo` stays safe cross-thread by text-equality, as designed.
- `unload_if_idle` still gates on `_gen_lock.locked()`, and still cannot run
  concurrently with a task because the worker loop is single-threaded.
- `run_worker_loop`'s own `release_mlx_thread_state()` stays: that thread still
  loads and unloads models and deallocates foreign-thread arrays.

## Testing strategy

CI has no mlx, so the session is exercised with real threads around fake
engines — the pattern the worker tests already rely on. Discipline rules,
from adversarial review of this design:

- Every test choreographs with `threading.Event`s and joins the exposed
  session thread handle before asserting. No sleeps, no ident-recycling
  assumptions — cross-task freshness is asserted on `Thread` objects, not
  idents.
- Any test that abandons a session unwedges the fake and joins the leaked
  thread before returning, so no zombie fires a later test's monkeypatch.
- `FakeEngine` grows the recording the pins need once, centrally: the thread
  object seen by each `generate`, and the ident seen by each
  `reset_prompt_cache`.
- Worker tests wrap via one helper (`inner = FakeEngine(...); run_task(...,
  ManagedEngine(inner), ...)`) and keep every existing assertion verbatim
  against the inner fake.

The pins:

- All of a task's generations observe the same thread, different from the
  caller's; consecutive tasks on one `ManagedEngine` observe different
  `Thread` objects, and a release per task.
- `release_mlx_thread_state` (monkeypatched in `sous.engine.base`, the only
  namespace `run_task`-level tests can reach it through) runs exactly once per
  session, and its ident equals the ident the fake recorded during `generate`.
- **Consideration-7 regression:** session A wedges inside `inner.generate`
  holding the lock (event-controlled); session B times out queued on the lock;
  the test unwedges A and joins both threads; B's inner `generate` never ran.
  This fails against a design that checks `abandoned` outside the lock.
- Stalled-then-completed generation: the fake gates on the session's
  `_abandoned` event so the interleaving is pinned; the result is dropped, no
  reply is queued, and after `close()` the thread exits and releases.
- Reset ownership: no reset ever runs on a session thread — every recorded
  reset ident is the worker's. This fails against a design that sneaks a reset
  back into the `CLOSE` path, where it would race the next task.
- Zero-generation task (context overflow, pre-turn cancel): session closes
  promptly, resets stay 2, release fires once.
- Session-level: an exception in `generate` propagates from
  `session.generate`, the same session then serves another request, and
  `close()` still releases once. `KeyboardInterrupt` relays likewise (existing
  restart-recovery pin strengthened to assert the release).
- Rewritten pins: `test_generation_thread_releases_mlx_state_before_exit`
  becomes once-per-task-on-the-session-thread;
  `test_generation_timeout_at_wall_budget_is_budget_exhausted` trades its
  `time.sleep(2)` for an event-gated fake plus join;
  `test_genuine_stall_with_budget_remaining_fails` injects the stall through a
  stub session (a blocking engine cannot reach that branch: the timeout *is*
  the remaining budget, so a real stall always lands at the deadline).

## Verification before the default flips

CI cannot see any of this; per CLAUDE.md it is verified locally, in order:

1. `scripts/e2e_smoke.py` (0.6B, trimmable path): `cold_retries: 0`, `hits > 0`.
2. One real delegated task on the default `Qwen3.8-27B-mxfp8` with
   `prompt_cache = true`: `cold_retries: 0`, `reused_tokens > 0`, task
   completes, no segfault (the mlx thread-state rule in CLAUDE.md).
3. Wall-clock before/after **in one machine state**: one process, weights
   loaded once, the same multi-turn workload run with the cache off and on by
   toggling `PrefixCache.enabled` between runs. This avoids the paging that
   invalidated #33's comparison.
4. A quality judgement (consideration 6): with reuse actually engaging,
   delegated tasks must still complete correctly. Incremental prefill is not
   numerically equal to single-shot prefill on the hybrid architecture; the
   judgement is behavioural, on real tasks, not tensor equality.

Only after 1–4 hold does `SousConfig.prompt_cache` flip to `true`
(config default, `from_toml` fallback, README/config docs, and the now-stale
sentence in `e2e_smoke.py`'s docstring).
