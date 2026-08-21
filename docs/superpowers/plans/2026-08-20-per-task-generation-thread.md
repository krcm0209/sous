# Per-Task Generation Thread Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all of a task's generations on one thread so the cross-turn prompt cache survives between turns, then flip `prompt_cache` to default-on once real-model verification passes.

**Architecture:** A new `GenerationSession` (owned by `ManagedEngine`) holds one daemon thread per task, a request/reply queue pair, and an `abandoned` event checked under `_gen_lock`. `run_task` creates one session per task and keeps every cache reset on the worker thread. `CLOSE` only ends the session loop; it carries no reset.

**Tech Stack:** Python 3.14, threading + queue stdlib, pytest, ty, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-20-per-task-generation-thread-design.md`

## Global Constraints

- Python >= 3.14. `except A, B:` without parentheses is valid syntax; do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- mlx / mlx_lm / mlx_vlm imports stay function-local. CI has no mlx.
- Every thread that touches mlx MUST call `release_mlx_thread_state()` before it exits (ml-explore/mlx#4327), or never exit.
- Tests must never touch the real `~/.sous`; always tmp_path-based config_path/data_dir.
- Tests choreograph with `threading.Event` and bounded `join`; never with sleeps as synchronization. Compare `Thread` objects, not idents, across tasks (idents recycle).
- Conventional Commits; imperative lowercase subject; why in the body.
- `docs/superpowers/**` records are append-only: add new files, never edit old ones.
- CI gates: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`.

---

### Task 1: `GenerationSession` in engine/base.py

**Files:**
- Modify: `src/sous/engine/base.py` (add `GenerationStalled`, `_CLOSE`, `GenerationSession`, `ManagedEngine.session()`; narrow `EngineManager.get() -> ManagedEngine`)
- Modify: `tests/fake_engine.py` (record generate threads and reset idents)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: `ManagedEngine._gen_lock`, `ManagedEngine._inner`, `release_mlx_thread_state()` — all already in `src/sous/engine/base.py`.
- Produces: `GenerationStalled(Exception)`; `GenerationSession` with `generate(messages: list[dict], tools: list[dict], max_tokens: int, timeout: float) -> str` (raises `GenerationStalled` on timeout, re-raises engine exceptions), `close() -> None`, test-visible `_thread`, `_abandoned`, `_replies`, `_requests`; `ManagedEngine.session() -> GenerationSession`; `EngineManager.get() -> ManagedEngine`.

- [ ] **Step 1: Extend the fake engine's recording**

In `tests/fake_engine.py`, add `import threading` at the top, then extend `__init__`, `generate`, and `reset_prompt_cache`:

```python
"""Scripted Engine implementation for model-free tests."""

import threading


class FakeEngine:
    model_id = "fake/model"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.max_tokens_seen: list[int] = []
        self.unloaded = False
        self.resets = 0
        self.stats: dict = {}
        # Which thread ran each call: the per-task-thread design (issue #34)
        # is pinned on these. Thread objects, not idents — idents recycle.
        self.generate_threads: list[threading.Thread] = []
        self.reset_idents: list[int] = []

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        self.generate_threads.append(threading.current_thread())
        self.calls.append([dict(m) for m in messages])
        self.max_tokens_seen.append(max_tokens)
        if not self.script:
            raise AssertionError("FakeEngine script exhausted")
        return self.script.pop(0)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def reset_prompt_cache(self) -> None:
        self.resets += 1
        self.reset_idents.append(threading.get_ident())

    def prompt_cache_stats(self) -> dict:
        return dict(self.stats)

    def unload(self) -> None:
        self.unloaded = True
```

Note: subclasses that override `generate` without calling `super()` do not record threads; no test on such a subclass may assert `generate_threads`.

- [ ] **Step 2: Write the failing session tests**

Append to `tests/test_engine_base.py`. Add `import pytest` and extend the base import line to `from sous.engine.base import EngineManager, GenerationStalled, ManagedEngine, select_backend`.

```python
def _msgs() -> list[dict]:
    return [{"role": "user", "content": "x"}]


def test_session_runs_all_generations_on_one_fresh_thread():
    inner = FakeEngine(["a", "b"])
    session = ManagedEngine(inner).session()
    assert session.generate(_msgs(), [], 8, timeout=5) == "a"
    assert session.generate(_msgs(), [], 8, timeout=5) == "b"
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()
    assert inner.generate_threads[0] is inner.generate_threads[1] is session._thread
    assert session._thread is not threading.current_thread()


def test_session_releases_mlx_state_once_on_its_own_thread(monkeypatch):
    import sous.engine.base as base

    released_in: list[int] = []
    monkeypatch.setattr(
        base, "release_mlx_thread_state", lambda: released_in.append(threading.get_ident())
    )
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    assert session.generate(_msgs(), [], 8, timeout=5) == "a"
    session.close()
    session._thread.join(5)
    assert released_in == [session._thread.ident]


def test_session_relays_exceptions_and_survives_them():
    class Flaky(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            out = super().generate(messages, tools, max_tokens)
            if out == "boom":
                raise ValueError("boom")
            return out

    inner = Flaky(["boom", "ok"])
    session = ManagedEngine(inner).session()
    with pytest.raises(ValueError, match="boom"):
        session.generate(_msgs(), [], 8, timeout=5)
    # The same session, the same thread: an engine error must not kill the loop.
    assert session.generate(_msgs(), [], 8, timeout=5) == "ok"
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()


def test_session_close_without_any_generation():
    session = ManagedEngine(FakeEngine([])).session()
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()


def test_stalled_generation_is_abandoned_and_its_late_result_dropped():
    gate = threading.Event()

    class Gated(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            gate.wait(10)
            return super().generate(messages, tools, max_tokens)

    inner = Gated(["late"])
    session = ManagedEngine(inner).session()
    with pytest.raises(GenerationStalled):
        session.generate(_msgs(), [], 8, timeout=0.05)
    assert session._abandoned.is_set()
    gate.set()  # ordering pin: the generation completes only after abandonment
    session._thread.join(5)
    assert not session._thread.is_alive()
    assert session._replies.empty()  # the late result was dropped, not queued


def test_abandoned_waiter_on_the_lock_never_generates():
    """Issue #34, consideration 7: a generation abandoned while QUEUED on
    _gen_lock must exit when the lock frees, never run under the next task's
    identity. Fails if the session checks _abandoned before taking the lock
    instead of after."""
    entered = threading.Event()
    release = threading.Event()

    class Wedged(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            entered.set()
            release.wait(10)
            return super().generate(messages, tools, max_tokens)

    inner = Wedged(["a"])
    managed = ManagedEngine(inner)
    session_a = managed.session()
    session_b = managed.session()
    a_result: list[str] = []
    threading.Thread(
        target=lambda: a_result.append(session_a.generate(_msgs(), [], 8, timeout=10)),
        daemon=True,
    ).start()
    assert entered.wait(5)  # A is wedged inside generate, holding _gen_lock
    with pytest.raises(GenerationStalled):
        session_b.generate(_msgs(), [], 8, timeout=0.05)  # B abandoned on the lock
    release.set()
    session_b._thread.join(5)
    assert not session_b._thread.is_alive()
    session_a.close()
    session_a._thread.join(5)
    assert a_result == ["a"]
    assert len(inner.calls) == 1  # B's request never reached the engine


def test_close_unleaks_an_idle_thread_holding_an_unconsumed_reply():
    """The reply-vs-timeout race can abandon a session whose thread already
    queued its reply and parked. CLOSE must wake it so it exits and releases —
    otherwise an ("err", e) reply would pin the KV cache through its traceback
    for the daemon's lifetime."""
    inner = FakeEngine(["a"])
    session = ManagedEngine(inner).session()
    # Drive the loop directly: a reply lands, but no caller consumes it.
    session._requests.put_nowait((_msgs(), [], 8))
    for _ in range(1000):
        if not session._replies.empty():
            break
        time.sleep(0.005)
    else:
        pytest.fail("session thread never produced the reply")
    session._abandoned.set()  # what generate() does when it times out
    session.close()
    session._thread.join(5)
    assert not session._thread.is_alive()
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -q`
Expected: ImportError — `GenerationStalled` does not exist in `sous.engine.base`.

- [ ] **Step 4: Implement `GenerationSession`**

In `src/sous/engine/base.py`: add `import queue` to the stdlib imports. Put `GenerationStalled` and `_CLOSE` right after `release_mlx_thread_state`; put `GenerationSession` after `ManagedEngine` (it uses `_gen_lock`) and before `EngineManager`:

```python
class GenerationStalled(Exception):
    """A generation produced no reply within its deadline."""


_CLOSE = object()  # session shutdown sentinel; ends the loop, carries no reset


class GenerationSession:
    """One task's generations, all on one daemon thread (issue #34).

    mlx KV cache arrays are usable only from the thread whose streams created
    them (streams are thread-scoped for use — probed empirically, see the
    design spec), and every thread that touched mlx must call
    release_mlx_thread_state() before it exits (ml-explore/mlx#4327). A fresh
    thread per generation therefore killed the prompt cache every turn; one
    thread per task lets turn N+1 reuse turn N's cache.

    The loop re-checks `_abandoned` while it HOLDS _gen_lock: a request whose
    task gave up while still queued on the lock exits instead of running
    under the next task's identity (issue #34, consideration 7).

    close() sends _CLOSE and never joins. A healthy or abandoned-but-idle
    thread dequeues it, releases its mlx state, and exits — this is also what
    un-leaks a thread whose reply lost the timeout race, so nothing it pinned
    (worst case an ("err", e) traceback holding the KV cache) outlives the
    task. A wedged thread never dequeues it and is leaked deliberately, like
    the abandoned per-generation threads before this class: it never exits,
    so it never hits the TLS-teardown segfault, and _gen_lock keeps the next
    task off the engine meanwhile. _CLOSE deliberately carries no cache
    reset — a late reset from a stale session thread would race the next
    task's cache and stats, the same class of bug as consideration 7. Every
    reset belongs to the worker thread.
    """

    def __init__(self, managed: ManagedEngine):
        self._managed = managed
        # maxsize=1 plus put_nowait everywhere: the loop always dequeues a
        # request before parking again, so Full is unreachable — and if a
        # future edit breaks that, failing loudly beats deadlocking the
        # worker inside run_task's finally.
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._replies: queue.Queue = queue.Queue(maxsize=1)
        self._abandoned = threading.Event()
        self._closed = False
        # Kept as an attribute so tests can join it; production never joins —
        # a wedged generation must not block task teardown.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                req = self._requests.get()
                if req is _CLOSE:
                    return
                with self._managed._gen_lock:
                    if self._abandoned.is_set():
                        return
                    try:
                        reply = ("ok", self._managed._inner.generate(*req))
                    except BaseException as e:  # noqa: BLE001 — relayed to the caller
                        reply = ("err", e)
                if self._abandoned.is_set():
                    return
                self._replies.put_nowait(reply)
                # A parked thread must not pin the reply: an ("err", e) entry
                # holds the whole generation frame — KV cache included —
                # through the traceback.
                del reply
        finally:
            release_mlx_thread_state()

    def generate(
        self, messages: list[dict], tools: list[dict], max_tokens: int, timeout: float
    ) -> str:
        assert not self._closed and not self._abandoned.is_set(), (
            "session reused after close() or a stall"
        )
        self._requests.put_nowait((messages, tools, max_tokens))
        try:
            kind, value = self._replies.get(timeout=timeout)
        except queue.Empty:
            self._abandoned.set()
            raise GenerationStalled(f"generation stalled (> {round(timeout, 1)}s)") from None
        if kind == "err":
            raise value
        return value

    def close(self) -> None:
        """End the session thread; never joined (see the class docstring)."""
        if self._closed:
            return
        self._closed = True
        self._requests.put_nowait(_CLOSE)
```

Add to `ManagedEngine`:

```python
    def session(self) -> GenerationSession:
        """One task's generation thread; run_task creates one per task."""
        return GenerationSession(self)
```

Narrow `EngineManager.get`'s annotation from `-> Engine` to `-> ManagedEngine` (the body already returns one).

Note: the file has `from __future__ import annotations`, so the `managed: ManagedEngine` annotation resolves regardless of definition order; the placement above (session after `ManagedEngine`) also keeps it valid at runtime.

- [ ] **Step 5: Run the session tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -q`
Expected: PASS.

- [ ] **Step 6: Mutation checks (do not commit these edits)**

1. Move the first `if self._abandoned.is_set(): return` to just BEFORE `with self._managed._gen_lock:`. Run `uv run pytest tests/test_engine_base.py::test_abandoned_waiter_on_the_lock_never_generates -q`. Expected: FAIL (B generates after the lock frees). Revert.
2. Delete the second `if self._abandoned.is_set(): return` (post-generation check). Run `uv run pytest tests/test_engine_base.py::test_stalled_generation_is_abandoned_and_its_late_result_dropped -q`. Expected: FAIL (`_replies` is not empty). Revert.

- [ ] **Step 7: Full local gates**

Run: `uv run pytest -m "not model" -q && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: all pass (worker still uses its own thread-per-generation path; nothing else changed).

- [ ] **Step 8: Commit**

```bash
git add src/sous/engine/base.py tests/fake_engine.py tests/test_engine_base.py
git commit -m "feat: add GenerationSession, one generation thread per task"
```

Body: the session exists so the prompt cache and its creating thread share a lifetime (#34); the abandoned check runs under _gen_lock to close the late-waiter identity leak.

---

### Task 2: run_task on the session

**Files:**
- Modify: `src/sous/worker.py` (delete `GenerationStalled` class and `_generate_with_timeout`; use `engine.session()`)
- Modify: `tests/test_worker.py` (wrap fakes in `ManagedEngine`; rewrite three thread-shape tests; add new pins)
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `GenerationSession.generate(messages, tools, max_tokens, timeout)`, `GenerationSession.close()`, `ManagedEngine.session()`, `GenerationStalled` — all from Task 1.
- Produces: `run_task(task, store, engine: ManagedEngine, config, context=None)`; `sous.worker.GenerationStalled` re-exported via import.

- [ ] **Step 1: Rewire worker.py**

In `src/sous/worker.py`:

1. Remove `import queue`.
2. Change the engine import line to:

```python
from sous.engine.base import (
    Engine,
    EngineManager,
    GenerationStalled,
    ManagedEngine,
    release_mlx_thread_state,
)
```

3. Delete the `class GenerationStalled(Exception)` block and the whole `_generate_with_timeout` function.
4. Change `run_task`'s signature parameter `engine: Engine` to `engine: ManagedEngine` (docstringless signature otherwise unchanged). `_elide_if_needed` and `_failure_extra` keep `engine: Engine`.
5. Directly under the existing start-of-task `engine.reset_prompt_cache()` (before the `try:`), add:

```python
    # One generation thread for the whole task: the prompt cache lives on
    # that thread's mlx streams, so this is what lets turn N+1 reuse it (#34).
    session = engine.session()
```

6. Replace the generation call inside the loop:

```python
            remaining = max(0.1, deadline - time.monotonic())
            try:
                text = session.generate(
                    messages,
                    WORKER_TOOLS,
                    min(config.max_tokens_per_generation, output_room),
                    timeout=remaining,
                )
```

(The `except GenerationStalled` / `except Exception` arms are unchanged.)

7. Replace the `finally:` block at the end of `run_task`:

```python
    finally:
        # Runs on every exit from the try — normal finish, any early return
        # in the loop, or an exception escaping it. The session thread ends
        # here; the reset stays on the worker thread, the single reset owner
        # on every path, so the cache never outlives the task that built it.
        session.close()
        engine.reset_prompt_cache()
```

- [ ] **Step 2: Migrate test_worker.py call sites**

In `tests/test_worker.py`:

1. Add to the imports: `from sous.engine.base import GenerationStalled, ManagedEngine`.
2. Add the capture helper after the `FINISH` constant:

```python
class SessionCapturingEngine(ManagedEngine):
    """Keeps every session it hands out so tests can join session threads."""

    def __init__(self, inner):
        super().__init__(inner)
        self.sessions: list = []

    def session(self):
        s = super().session()
        self.sessions.append(s)
        return s


def _join_sessions(engine: SessionCapturingEngine, timeout: float = 5.0) -> None:
    for s in engine.sessions:
        s._thread.join(timeout)
        assert not s._thread.is_alive()
```

3. At every `run_task(...)` call site that passes a `FakeEngine` (or subclass) directly, wrap it: `run_task(task, store, ManagedEngine(engine), cfg)` — keeping every existing assertion against the inner fake variable. Sites that pass an inline fake with no assertions (e.g. `run_task(task, store, FakeEngine([FINISH]), cfg)`) wrap inline: `run_task(task, store, ManagedEngine(FakeEngine([FINISH])), cfg)`.
4. `test_worker_loop_survives_bookkeeping_exception` needs no change: `EngineManager` wraps its factory's product itself.

- [ ] **Step 3: Rewrite the three thread-shape tests**

Replace `test_generation_timeout_at_wall_budget_is_budget_exhausted` with:

```python
def test_generation_timeout_at_wall_budget_is_budget_exhausted(env):
    """C1: the generation timeout IS the remaining wall-clock budget, so a
    timeout with the deadline passed means the budget ran out — per spec that
    ends the task as done/budget-exhausted with a partial report, never
    failed."""
    root, cfg, store = env
    cfg = dataclasses.replace(cfg, max_minutes=0.005)  # 0.3 s wall budget
    unwedge = threading.Event()

    class StallingEngine(FakeEngine):
        def generate(self, messages, tools, max_tokens):
            unwedge.wait(10)  # released by the test AFTER run_task returns
            return FINISH

    inner = StallingEngine([FINISH])
    engine = SessionCapturingEngine(inner)
    task = _start(store, root)
    run_task(task, store, engine, cfg)
    got = store.get(task.id)
    assert got.state == TaskState.DONE
    assert got.outcome == "budget-exhausted"
    assert "files_changed" in got.report  # partial report still assembled
    assert Path(got.report["transcript_path"]).exists()
    # Both resets ran on the worker thread — the abandoned session must not
    # reset anything, and the leaked thread must not outlive the test where
    # it could fire a later test's monkeypatched hooks.
    assert inner.resets == 2
    assert all(i == threading.get_ident() for i in inner.reset_idents)
    unwedge.set()
    _join_sessions(engine)
```

Replace `test_genuine_stall_with_budget_remaining_fails` with:

```python
def test_genuine_stall_with_budget_remaining_fails(env, monkeypatch):
    """C1 (the other side): a stall while wall-clock budget genuinely remains
    is still a failure, not budget exhaustion. The stall is injected — a real
    blocking engine cannot reach this branch, because the per-turn timeout IS
    the remaining budget."""
    from sous.engine.base import GenerationSession

    root, cfg, store = env  # max_minutes=1: plenty of budget remains

    def stall_immediately(self, messages, tools, max_tokens, timeout):
        raise GenerationStalled("generation stalled (> 5s)")

    monkeypatch.setattr(GenerationSession, "generate", stall_immediately)
    task = _start(store, root)
    run_task(task, store, ManagedEngine(FakeEngine([FINISH])), cfg)
    got = store.get(task.id)
    assert got.state == TaskState.FAILED
    assert "stalled" in got.report["error"]
    assert "files_changed" in got.report
```

Replace `test_generation_thread_releases_mlx_state_before_exit` with:

```python
def test_session_thread_releases_mlx_state_once_per_task(env, monkeypatch):
    """mlx >= 0.32.1 requires mx.clear_streams() at the end of every thread
    that touched mlx (ml-explore/mlx#4327). With one generation thread per
    task the release runs once per task, on that thread. Patching
    sous.engine.base is sufficient: run_task-level tests never invoke the
    copy imported into sous.worker (that one belongs to run_worker_loop)."""
    import sous.engine.base as base

    released_in: list[int] = []
    monkeypatch.setattr(
        base, "release_mlx_thread_state", lambda: released_in.append(threading.get_ident())
    )
    root, cfg, store = env
    task = _start(store, root)
    inner = FakeEngine(
        [
            CALL.format(name="write_file", args='{"path": "out.txt", "content": "x"}'),
            FINISH,
        ]
    )
    engine = SessionCapturingEngine(inner)
    run_task(task, store, engine, cfg)
    assert store.get(task.id).state == TaskState.DONE
    _join_sessions(engine)
    gen_idents = {t.ident for t in inner.generate_threads}
    assert len(gen_idents) == 1, "all generations on one thread"
    assert released_in == list(gen_idents), "one release, on the generation thread"
```

- [ ] **Step 4: Add the new worker-level pins**

```python
def test_consecutive_tasks_get_fresh_session_threads(env, monkeypatch):
    """One session thread per task (issue #34): a session reused across tasks
    would resurrect the original bug after any abandoned task. Thread objects,
    not idents — idents recycle."""
    import sous.engine.base as base

    released: list[bool] = []
    monkeypatch.setattr(base, "release_mlx_thread_state", lambda: released.append(True))
    root, cfg, store = env
    inner = FakeEngine([FINISH, FINISH])
    engine = SessionCapturingEngine(inner)
    for _ in range(2):
        task = _start(store, root)
        run_task(task, store, engine, cfg)
        assert store.get(task.id).state == TaskState.DONE
    _join_sessions(engine)
    assert len(inner.generate_threads) == 2
    assert inner.generate_threads[0] is not inner.generate_threads[1]
    assert len(released) == 2  # one release per task
```

Extend `test_run_task_resets_the_prompt_cache_at_start_and_end` (keep name):

```python
def test_run_task_resets_the_prompt_cache_at_start_and_end(env):
    root, cfg, store = env
    task = _start(store, root)
    inner = FakeEngine([FINISH])
    engine = SessionCapturingEngine(inner)
    run_task(task, store, engine, cfg)
    assert inner.resets == 2  # once at entry, once in the finally
    _join_sessions(engine)
    here = threading.get_ident()
    assert inner.reset_idents == [here, here]  # the worker thread owns every reset
```

Extend `test_context_over_cap_with_nothing_to_elide_fails_cleanly` — after the existing asserts, using a `SessionCapturingEngine`-wrapped inner and a `time.monotonic()` bracket around `run_task`:

```python
    # Zero-generation task: the session was created and must close promptly —
    # no join wait, both resets present, thread gone.
    assert elapsed < 3.0
    assert inner.resets == 2
    _join_sessions(engine)
```

Extend `test_engine_exception_fails_task_cleanly` — wrap in `SessionCapturingEngine`; after the existing asserts add `assert inner.resets == 2` and `_join_sessions(engine)`.

Extend `test_restart_recovery_reports_files_changed_and_transcript` — wrap in `SessionCapturingEngine`; after `pytest.raises(KeyboardInterrupt)` add `_join_sessions(engine)` (pins that the finally closed the session before the exception propagated).

- [ ] **Step 5: Run the worker tests**

Run: `uv run pytest tests/test_worker.py -q`
Expected: PASS, including every migrated assertion verbatim against the inner fakes.

- [ ] **Step 6: Full local gates**

Run: `uv run pytest -m "not model" -q && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/sous/worker.py tests/test_worker.py
git commit -m "feat: run each task's generations on its session thread"
```

Body: warm cache reuse needs the same mlx streams across turns; per-generation threads made that impossible (#34).

---

### Task 3: Real-model verification (local only; CI cannot see this)

**Files:**
- Create: scratchpad harnesses only (not committed).
- No repo changes in this task.

**Interfaces:**
- Consumes: the shipped `scripts/e2e_smoke.py`; `SousConfig(prompt_cache=True)`; `EngineManager`.
- Produces: recorded numbers for the PR body; the go/no-go decision for Task 4.

- [ ] **Step 1: Smoke on the 0.6B (trimmable path)**

Run: `uv run python scripts/e2e_smoke.py`
Expected: `prompt_cache` block with `cold_retries: 0` and `hits > 0`; judge task success by `hello.txt` content, not the final state (the 0.6B is unreliable at `finish`).

- [ ] **Step 2: One real delegated task on the default 27B (hybrid path)**

Scratchpad harness modeled on `e2e_smoke.py` but with the default `model_id`, `prompt_cache=True`, temp dirs, and a small multi-turn coding task. Expected: outcome `completed`, `cold_retries: 0`, `reused_tokens > 0`, `snapshot_bytes` ≈ 147 MiB, and no daemon segfault (the CLAUDE.md thread-state rule).

- [ ] **Step 3: Wall-clock, one machine state**

Scratchpad harness: load the engine once, replay a fixed synthetic multi-turn conversation (max_tokens small so prefill dominates), toggling `PrefixCache.enabled` off → on → off → on between replays in the same process. Record the per-turn wall-clock table. Expected: warm turns flat, cold turns growing, no paging (single load).

- [ ] **Step 4: Quality judgement (issue consideration 6)**

At least two further real delegated tasks with the cache on (different shapes: read-then-edit, multi-file). Expected: outcomes correct by inspection of the diffs/files. Record observations. If quality is degraded, STOP: Task 4 does not run, the default stays `false`, and the findings go to the PR/issue instead.

---

### Task 4: Flip the default (gated on Task 3)

**Files:**
- Modify: `src/sous/config.py` (default `True`, `from_toml` fallback `True`)
- Modify: `tests/test_config.py` (default expectation)
- Modify: `scripts/e2e_smoke.py` (docstring sentence about the cold fallback)
- Modify: `src/sous/engine/promptcache.py` (the `PrefixCache.__init__` comment naming the shipped default)
- Modify: `README.md` (only if it states the default; check with grep)

**Interfaces:**
- Consumes: verification results from Task 3.
- Produces: `SousConfig.prompt_cache: bool = True`.

- [ ] **Step 1: Update the config default test first**

In `tests/test_config.py`, find the `prompt_cache` default assertion (`grep -n prompt_cache tests/test_config.py`) and flip its expectation to `True`, keeping the override case.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_config.py -q` — expected: the flipped assertion FAILS against the current default.

- [ ] **Step 3: Flip the default**

In `src/sous/config.py`: `prompt_cache: bool = True` and `model.get("prompt_cache", True)`. Update the stale comment in `PrefixCache.__init__` and the `e2e_smoke.py` docstring sentence ("currently falls back to a cold prefill every turn…"). `grep -rn "prompt_cache" README.md docs/*.md` (excluding `docs/superpowers/**`, which is append-only) and update any stated default.

- [ ] **Step 4: Run the gates**

Run: `uv run pytest -m "not model" -q && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: enable prompt cache reuse by default"
```

Body: cite the Task 3 numbers (cold_retries 0, reused tokens, wall-clock table).

---

### Task 5: Review and PR

- [ ] **Step 1: Adversarial code review of the whole branch diff** (code-review workflow; fix anything confirmed, re-run gates).
- [ ] **Step 2: `uv lock --check` and the full four CI gates one final time.**
- [ ] **Step 3: Open the PR with the write-pr skill.** The body carries: the design summary, the verification numbers from Task 3, and the acceptance-shape checklist from issue #34. Do not merge; do not comment on the issue.
