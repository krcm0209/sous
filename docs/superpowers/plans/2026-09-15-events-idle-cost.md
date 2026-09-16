# Events Idle Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GET /sous/events` sends the status document only when something on it changed while the daemon is idle, so an idle daemon with a `sous top` connected costs well under 1 % of a core; the document build opens no SQLite connection and parses no TOML unless a task or the config changed; `sous top` runs the idle clocks itself.

**Architecture:** The event stream polls one composite version — the in-flight registry's, a new one on the engine manager (load, unload, hold, release) and a new one on the task store (any row written) — instead of the registry's alone, and keeps its once-a-second heartbeat only while a document shows a turn in flight, a delegated task running or a load under way (the terminal's stall detection and its task clock need it then; the cost is nothing beside a generation). Idle, the loop polls every half second and sends nothing but sse-starlette's 10 s ping. The service memoises its task-store reads on the store's version and the allowlist on the config file's mtime. The terminal advances the three idle clocks from the last document's `idle_seconds` plus the time since it arrived, on its existing half-second idle tick, repainting only lines whose words changed.

**Tech Stack:** Python 3.14, Starlette/uvicorn, sse-starlette, SQLite (WAL), Textual 8; tests with pytest, Textual's pilot.

**Spec:** GitHub issue #96 (proposal items 1–3 and its acceptance section, plus the maintainer's comment with the build-cost split) and the measurements below, which this plan argues from.

## What was measured (M5 Pro, 2026-09-15)

Throwaway daemon from the `main` checkout on port 8385, model never loaded, CPU-time deltas over 45 s windows (`ps -o cputime`), one core = 100 %:

| variant | daemon CPU |
| --- | --- |
| idle, no events client | 0.45 % |
| + 1 events client, defaults (10 Hz tick, 1 Hz heartbeat) | 0.92 % |
| + 2 clients | 1.36 % |
| + 1 client, heartbeat off | 0.64 % |
| + 1 client, heartbeat off, tick 1 s | 0.53 % |

One client costs 0.47 %: the heartbeat's rebuild, encode and send 0.28 % (about 2.8 ms per heartbeat — the 1 ms build plus the thread hand-off, the JSON and the frame), the 10 Hz tick loop 0.11 %, the open connection 0.08 %. The 0.45 % with no client at all is the worker loop's `claim_next` — an `UPDATE … RETURNING` write transaction on a fresh SQLite connection twice a second — plus the idle sweep; that is a separate issue (to be filed) and NOT this plan's scope. Expected after this plan: idle daemon with one client about 0.55 %, of which 0.45 % is the worker poll.

The gate in the PR #95 review used `ps -o %cpu`, a decaying average, sampled right after activity; the recipe in Task 6 replaces it.

## Global Constraints

- Any thread that touches mlx calls `engine.base.release_mlx_thread_state()` before it exits; `status_document` already does and keeps doing so on every path. Nothing in this plan adds an mlx call.
- Tests never touch the real `~/.sous`: every `TaskStore`, `SousConfig` and daemon under test takes a `tmp_path`-based `config_path`/`data_dir`.
- Comments explain non-obvious *why*, never restate code; no test or code comment cites this plan, the issue's item numbers, a task or a step — state the fact instead.
- Type-suppression pragmas are `# ty: ignore[rule]`; `uv run ty check` must pass on tests too.
- `addopts = -q` is set: never add another `-q` to a pytest invocation; use `-p no:cacheprovider`.
- The gateway and the monitor never log a request body, header value or query string; the events loop keeps logging only an exception's type.
- `docs/superpowers/**` is never edited after commit.
- Conventional Commits, imperative lowercase subject, *why* in the body; every commit ends with `Co-Authored-By: Claude <noreply@anthropic.com>` and nothing else in that trailer. Nothing committed or posted names the maintainer or a session.
- `EVENT_TICK_SECONDS = 0.1` and `EVENT_HEARTBEAT_SECONDS = 1.0` keep their names and values (tests monkeypatch them); a new `EVENT_IDLE_TICK_SECONDS = 0.5` joins them.
- The events loop's contract while busy is unchanged: a registry change is on the wire within `EVENT_TICK_SECONDS`, a document at least every `EVENT_HEARTBEAT_SECONDS`.

---

### Task 1: `TaskStore.version`

**Files:**
- Modify: `src/sous/tasks.py` (`__init__`, `_conn`, a `version` property)
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TaskStore.version -> int` (property), bumped once per connection that changed at least one row; reads and no-op writes leave it alone. Task 3 reads it; Task 4 polls it through Task 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tasks.py`:

```python
def test_version_moves_once_per_connection_that_changed_a_row(tmp_path: Path):
    """The event stream polls this number: a read, and a claim that found
    nothing to claim, must not move it, or an idle daemon would rebuild
    its document twice a second for the worker's empty polls."""
    store = TaskStore(tmp_path / "tasks.db")
    v0 = store.version
    store.count_by_state()
    store.list_recent(limit=10)
    assert store.claim_next() is None
    assert store.version == v0
    task = store.enqueue("t", "do it", str(tmp_path), [], [])
    v1 = store.version
    assert v1 == v0 + 1
    assert store.claim_next() is not None
    v2 = store.version
    assert v2 == v1 + 1
    store.set_activity(task.id, "working", 1)
    assert store.version == v2 + 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_tasks.py -p no:cacheprovider -k version`
Expected: FAIL with `AttributeError: 'TaskStore' object has no attribute 'version'`.

- [ ] **Step 3: Implement**

In `src/sous/tasks.py`, add `import threading` beside the other imports. In `TaskStore.__init__`, before the `with self._conn() as c:` block (the schema setup goes through `_conn`, which must find these attributes):

```python
        # Bumped once per connection that changed a row. The event stream
        # polls it to learn that a task moved without a heartbeat; the
        # worker and the server share this one instance, so an in-process
        # counter sees every write. The lock is for the increment only —
        # readers take one int.
        self._version = 0
        self._version_lock = threading.Lock()
```

Add the property after `__init__`:

```python
    @property
    def version(self) -> int:
        """Moves with every write that changed a row and with nothing else:
        a reader that saw the same number twice saw the same rows twice.
        Read it BEFORE the reads it guards, so a write landing during them
        shows up as a newer number on the next check rather than hiding
        behind one read after the fact."""
        return self._version
```

Replace `_conn`:

```python
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # `with sqlite3.connect(...)` only ends the transaction — it leaves the
        # descriptor open. Returning the bare connection leaked db + WAL per
        # call, crash-looping the daemon at launchd's 256-descriptor soft limit.
        conn = sqlite3.connect(self.db_path)
        try:
            # Setup belongs inside the try: the WAL pragma takes a lock and can
            # raise SQLITE_BUSY under contention, which would otherwise escape
            # the finally and leak the descriptor we just opened.
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
            # After the commit. total_changes counts the rows this connection
            # inserted, updated or deleted — DDL and an UPDATE that matched
            # nothing (the worker's idle claim) leave it at zero.
            if conn.total_changes:
                with self._version_lock:
                    self._version += 1
        finally:
            conn.close()
```

- [ ] **Step 4: Run the test and the file**

Run: `uv run pytest tests/test_tasks.py -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/sous/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): count the writes that changed a row" -m "The event stream will poll it to learn that a task moved without rebuilding the document on a clock. A read, and the worker's idle claim that matched nothing, leave it alone."
```

---

### Task 2: `EngineManager.version`

**Files:**
- Modify: `src/sous/engine/base.py` (`EngineManager.__init__`, `get`, `unload_if_idle`, `hold`, `_prune_holders`, a `version` property)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EngineManager.version -> int` (property), bumped on: a load starting, a load finishing (or failing), an unload starting, an unload finishing, a hold taken, a holder pruned, and every reset of the idle clock — `touch()`, and `get()` on an already loaded engine — because the terminal extrapolates that clock between documents, and a reset it never hears about (a `count_tokens` call registers no turn) is a wrong number on screen until the next document. NOT bumped by `status()`. Task 3 reads it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_base.py`:

```python
def test_version_moves_on_load_unload_hold_release_and_idle_resets_and_nothing_else():
    """What the event stream polls between documents. A read moves nothing;
    every reset of the idle clock does, because the terminal runs that
    clock itself between documents and must hear about a restart."""
    created: list[FakeEngine] = []

    def factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    alive = {"ok": True}
    cfg = SousConfig(idle_unload_minutes=0)
    m = EngineManager(cfg, engine_factory=factory, holder_alive=lambda pid, started: alive["ok"])
    v = m.version
    m.get()
    assert m.version > v, "a load moved nothing"
    v = m.version
    m.status()
    assert m.version == v, "a read moved the version"
    m.get()
    assert m.version == v + 1, "a hit resets the idle clock and moved nothing"
    m.touch()
    assert m.version == v + 2, "a touch resets the idle clock and moved nothing"
    v = m.version
    m.hold(4242, 1.0)
    assert m.version == v + 1, "a hold moved nothing"
    m.status()
    assert m.version == v + 1, "a status read with a live holder moved the version"
    alive["ok"] = False
    m.status()
    assert m.version == v + 2, "a pruned holder moved nothing"
    v = m.version
    time.sleep(0.01)
    assert m.unload_if_idle() is True
    assert m.version == v + 2, "an unload is two changes: it started, it finished"
    assert m.unload_if_idle() is False
    assert m.version == v + 2, "a refused unload moved the version"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_engine_base.py -p no:cacheprovider -k version_moves`
Expected: FAIL with `AttributeError: 'EngineManager' object has no attribute 'version'`.

- [ ] **Step 3: Implement**

In `EngineManager.__init__`, after `self._clock = clock`:

```python
        # Bumped on the transitions the status document shows and the
        # in-flight registry cannot: a load starting or ending, an unload
        # starting or ending, a hold taken, a holder pruned, and every
        # reset of the idle clock — the terminal runs that clock itself
        # between documents, so a reset it never hears about is a wrong
        # number on screen until the next one.
        self._version = 0
```

Add after `__init__`:

```python
    @property
    def version(self) -> int:
        """Read without the lock — one int; see Inflight.version for why a
        poller reads it before, not after, the snapshot it describes."""
        return self._version

    def _bump(self) -> None:
        """Lock held by the caller."""
        self._version += 1
```

In `get()`: the hit path becomes

```python
            if self._engine is not None:
                self._last_used = self._clock()
                self._bump()
                return self._engine
```

the `self._loading = True` line becomes

```python
            self._loading = True
            self._bump()
```

the failure path's `self._loading = False` (inside `except BaseException:`) becomes

```python
                self._loading = False
                self._bump()
```

and the success path's block becomes

```python
        with self._changed:
            self._engine = engine
            self._loading = False
            self._last_used = self._clock()
            self._bump()
            self._changed.notify_all()
```

In `unload_if_idle()`: the `self._unloading = True` line becomes

```python
            self._unloading = True
            self._bump()
```

and the `finally` block becomes

```python
        finally:
            with self._changed:
                self._unloading = False
                self._bump()
                self._changed.notify_all()
```

In `hold()`: after `self._holders[pid] = create_time` add `self._bump()`.

`touch()` becomes

```python
    def touch(self) -> None:
        with self._lock:
            self._last_used = self._clock()
            self._bump()
```

In `_prune_holders()`: after the `for pid in gone:` loop's `_logger.info(...)` line (inside the loop is fine too; once after the loop is cleaner):

```python
        if gone:
            self._bump()
```

placed before the existing `if gone and not self._holders and self._last_used is not None:`.

- [ ] **Step 4: Run the test and the file**

Run: `uv run pytest tests/test_engine_base.py -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): count the load, unload, hold, release and idle-clock transitions" -m "The event stream will poll it beside the registry's version: these are the engine changes a status document shows that no registry write announces, and the heartbeat that used to carry them costs an idle daemon a document a second. The idle clock's resets count too: the terminal runs that clock between documents."
```

---

### Task 3: `SousService.status_version()` and a memoised build

**Files:**
- Modify: `src/sous/server.py` (`SousService.__init__`, `status_document`, new `status_version`, `_task_reads`, `_allowlist_now`; the `mount_monitor(...)` call in `create_server`)
- Modify: `src/sous/monitor.py` (`mount_monitor` signature only — the loop itself is Task 4)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `TaskStore.version` (Task 1), `EngineManager.version` (Task 2), `Inflight.version`.
- Produces: `SousService.status_version() -> tuple[int, int, int]` = `(inflight.version, engines.version, store.version)`; `mount_monitor(mcp, engines, status, version)` where `version: Callable[[], object]` replaces the `inflight` parameter (Task 4 reads it in the loop). `status_document` semantics unchanged; it opens no SQLite connection while `store.version` is unchanged and parses no TOML while the config file's mtime is unchanged; the narrow build (`recent=False`, the MCP tool) still reads only the counts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def test_status_version_is_the_three_counters(svc):
    service, store, _ = svc
    before = service.status_version()
    assert before == (service.inflight.version, service.engines.version, store.version)
    store.enqueue("t", "do it", "/tmp/nowhere", [], [])
    after = service.status_version()
    assert after[2] == before[2] + 1 and after[:2] == before[:2]


def test_the_build_reads_the_task_store_only_when_it_changed(svc, monkeypatch):
    """Ten documents a second during a turn opened twenty SQLite connections
    a second for counts and a listing that change only when a task does."""
    import sqlite3

    service, store, root = svc
    opened: list[str] = []
    real_connect = sqlite3.connect

    def counting(path, *args, **kwargs):
        opened.append(str(path))
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    service.status_document(recent=False)
    assert len(opened) == 1, "the narrow build reads the counts and nothing else"
    service.status_document(recent=True)
    assert len(opened) == 2, "the wide build adds the listing, once"
    service.status_document(recent=True)
    service.status_document(recent=False)
    assert len(opened) == 2, "a build with no task change opened a connection"
    out = service.delegate_task("t", "do it", str(root))
    opened.clear()
    doc = service.status_document(recent=True)
    assert opened, "a task change did not reach the build"
    assert doc["queue"]["queued"] == 1
    assert [t["id"] for t in doc["recent_tasks"]] == [out["task_id"]]


def test_a_running_tasks_seconds_keep_moving_between_task_changes(svc, monkeypatch):
    """What is cached is the rows, never the summaries: a running task's
    `seconds` is the clock minus its start, and a cached summary froze it."""
    from sous import server as server_module

    service, store, root = svc
    out = service.delegate_task("t", "do it", str(root))
    task = store.claim_next()
    assert task is not None and task.id == out["task_id"]
    started = task.started_at
    assert started is not None, "claim_next stamps started_at"
    monkeypatch.setattr(server_module.time, "time", lambda: started + 5.0)
    assert service.status_document(recent=True)["recent_tasks"][0]["seconds"] == 5
    monkeypatch.setattr(server_module.time, "time", lambda: started + 9.0)
    assert service.status_document(recent=True)["recent_tasks"][0]["seconds"] == 9


def test_the_allowlist_is_reparsed_only_when_the_config_file_changed(svc, monkeypatch):
    """An edit still takes effect on the next build — the file's mtime is
    the key — but ten builds a second no longer parse TOML ten times."""
    import os

    from sous import server as server_module

    service, _, _ = svc
    parses: list[Path] = []
    real = server_module.current_allowlist

    def counting(path):
        parses.append(path)
        return real(path)

    monkeypatch.setattr(server_module, "current_allowlist", counting)
    assert service.status_document(recent=False)["config"]["allowlist"] == [["pytest"]]
    service.status_document(recent=False)
    assert len(parses) == 1
    path = service.config.config_path
    path.write_text('[commands]\nallowlist = ["pytest", "ruff"]\n')
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert service.status_document(recent=False)["config"]["allowlist"] == [
        ["pytest"],
        ["ruff"],
    ]
    assert len(parses) == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_server.py -p no:cacheprovider -k "status_version or only_when_it_changed or keep_moving or reparsed"`
Expected: the first FAILS with `AttributeError: ... 'status_version'`; the second and fourth FAIL on their count assertions; the third passes already (it pins what must stay true once the cache exists).

- [ ] **Step 3: Implement**

In `SousService.__init__`, after `self.inflight = ...`:

```python
        # The status document is built up to ten times a second during a
        # turn, and the task counts, the recent listing and the allowlist
        # change only when a task or the config file does: each is read
        # once per change and served from here in between. A whole tuple
        # is replaced at once, so the builder threads never see half of one.
        self._counts_cache: tuple[int, dict] | None = None
        self._recent_cache: tuple[int, list[Task]] | None = None
        self._allowlist_cache: tuple[int, list[list[str]]] | None = None
```

Add these methods to `SousService`, before `status_document`:

```python
    def status_version(self) -> tuple[int, int, int]:
        """What the event stream polls between documents: the in-flight
        registry's version, the engine manager's and the task store's.
        Three ints read without a lock; a change to any of them is a
        document worth sending, and nothing else is."""
        return (self.inflight.version, self.engines.version, self.store.version)

    def _task_reads(self, *, recent: bool) -> tuple[dict, list[Task]]:
        # The version is read before the queries, so a write that lands
        # while they run is a newer number next time, never masked. Two
        # slots: the narrow build (the MCP tool) never pays for a listing
        # it does not show.
        version = self.store.version
        counts = self._counts_cache
        if counts is None or counts[0] != version:
            counts = (version, self.store.count_by_state())
            self._counts_cache = counts
        if not recent:
            return counts[1], []
        tasks = self._recent_cache
        if tasks is None or tasks[0] != version:
            tasks = (version, self.store.list_recent(limit=10))
            self._recent_cache = tasks
        return counts[1], tasks[1]

    def _allowlist_now(self) -> list[list[str]]:
        try:
            stamp = self.config.config_path.stat().st_mtime_ns
        except OSError:
            stamp = -1  # no file: the defaults, which current_allowlist knows
        cached = self._allowlist_cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        entries = current_allowlist(self.config.config_path)
        self._allowlist_cache = (stamp, entries)
        return entries
```

In `status_document`, replace `counts = self.store.count_by_state()` with `counts, tasks = self._task_reads(recent=recent)`, replace `"allowlist": current_allowlist(self.config.config_path),` with `"allowlist": self._allowlist_now(),`, and replace the `recent_tasks` line with

```python
            document["recent_tasks"] = [self._task_summary(t) for t in tasks]
```

In `create_server`, the mount line becomes

```python
    mount_monitor(mcp, engines, lambda: svc.status_document(recent=True), svc.status_version)
```

In `src/sous/monitor.py`, change the signature and docstring of `mount_monitor` to

```python
def mount_monitor(
    mcp: MCPServer,
    engines: EngineManager,
    status: Callable[[], dict],
    version: Callable[[], object],
) -> None:
    """Register GET /sous/status, GET /sous/events, POST /sous/hold and a
    404 for every other /sous/ path. `status` builds the full status
    document (recent turns and tasks included); `version` is what the event
    stream polls — a value that differs from the last one whenever the
    document would. The status and hold handlers hand their work to a
    thread — status() reads the task store and hold() takes the engine
    manager's lock, and neither belongs on the event loop — and the event
    stream builds each of its documents the same way."""
```

and the `EventSourceResponse(_status_events(status, inflight), ...)` call to `_status_events(status, version)`; in `_status_events`, for this task only, rename the parameter to `version: Callable[[], object]`, change `seen: int | None = None` to `seen: object = None  # nothing sent yet, whatever the clock says`, read `current = version()` where it read `inflight.version`, compare `current != seen`, and record `seen, sent = current, now` (Task 4 rewrites the loop; this keeps the suite green and `ty` clean in between). Drop the now-unused `Inflight` import from `monitor.py`.

- [ ] **Step 4: Run the suite parts this touches**

Run: `uv run pytest tests/test_server.py tests/test_monitor.py tests/test_cli.py -p no:cacheprovider`
Expected: all pass. Then `uv run ty check` and `uv run ruff check . && uv run ruff format --check .`: clean.

- [ ] **Step 5: Commit**

```bash
git add src/sous/server.py src/sous/monitor.py tests/test_server.py
git commit -m "perf(server): build the status document from memoised task and config reads" -m "During a turn the document is built up to ten times a second, and each build opened two SQLite connections and parsed the config file for values that change only when a task or an edit does. The rows are cached on the store's version and the allowlist on the file's mtime; summaries stay per build so a running task's seconds keep moving. status_version() is the composite the event stream will poll."
```

---

### Task 4: The event stream sends on change, and beats only while busy

**Files:**
- Modify: `src/sous/monitor.py` (`EVENT_IDLE_TICK_SECONDS`, `_status_events`, new `_busy`)
- Test: `tests/test_monitor.py` (`_collect_events` gains a duration mode; two tests rewritten, three added)

**Interfaces:**
- Consumes: `version()` from Task 3.
- Produces: the stream contract: a document at connect; then one whenever `version()` differs from the last sent, checked every `EVENT_TICK_SECONDS` while the last document was busy (a turn in flight, a delegated task running — `queue.running`, which also counts a task awaiting approval, deliberately: its clock runs too — or a load under way) and every `EVENT_IDLE_TICK_SECONDS` otherwise; plus one at least every `EVENT_HEARTBEAT_SECONDS` only while busy; the `ping` every `EVENT_PING_SECONDS` regardless. `_app` in `tests/test_monitor.py` gains `factory`, `store` and `idle_minutes` keyword parameters. Task 5 relies on the idle contract.

- [ ] **Step 1: Give the collector a duration mode**

In `tests/test_monitor.py`, change `_collect_events`'s signature to

```python
def _collect_events(
    app, *, want: int | None, seconds: float = 3.0, headers=(), during=None
) -> list:
```

change the counter line in its `send` to

```python
                if want is not None and len([f for f in frames if f[0] != "ping"]) >= want:
                    got.set()
```

and its wait block to:

```python
        task = asyncio.create_task(app(scope, receive, send))
        try:
            if want is None:
                # A quiet stream: nothing to wait for but the clock.
                await asyncio.sleep(seconds)
            elif want:
                await asyncio.wait_for(got.wait(), seconds)
            else:
                # A refused request completes on its own; nothing to wait for.
                try:
                    await asyncio.wait_for(task, seconds)
                except TimeoutError:
                    pytest.fail(f"the request was served, not refused: status={status}")
        finally:
```

and extend the docstring's first sentence with "— or, with `want=None`, for `seconds` regardless —".

- [ ] **Step 2: Write the failing tests**

Replace `test_events_beat_once_a_second_when_nothing_changes` with:

```python
def test_a_quiet_daemon_sends_no_status_frame_between_pings(tmp_path: Path, monkeypatch):
    """The heartbeat cost an idle daemon a document rebuild a second per
    connected terminal — a quarter of a percent of a core each, for a clock
    the terminal can run itself. Idle, only the ping goes out."""
    monkeypatch.setattr(monitor, "EVENT_PING_SECONDS", 0.2)
    monkeypatch.setattr(monitor, "EVENT_IDLE_TICK_SECONDS", 0.05)
    app, _ = _app(tmp_path)
    _, frames = _collect_events(app, want=None, seconds=1.2)
    assert [e for e, _ in frames if e == "status"] == ["status"]
    assert len([e for e, _ in frames if e == "ping"]) >= 3


def test_events_beat_once_a_second_while_a_turn_is_in_flight(tmp_path: Path, monkeypatch):
    """The terminal reads a silent stream during a turn as a stalled daemon
    (a prefill can run for a minute without a registry change), so the
    heartbeat stays for as long as a document shows a turn."""
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.2)
    registry = Inflight()
    registry.begin("msg_1", model="sous-local", stream=True, max_tokens=1)
    app, _ = _app(tmp_path, inflight=registry)
    started = time.monotonic()
    _, frames = _collect_events(app, want=3)
    assert len([e for e, _ in frames if e == "status"]) >= 3
    assert time.monotonic() - started < 2.0


def test_events_beat_while_a_delegated_task_is_running(tmp_path: Path, monkeypatch):
    """The worker writes the store once per tool call, but the task's own
    clock and the cache counters it moves are in the document: a running
    task keeps the heartbeat the way a gateway turn does."""
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.2)
    store = TaskStore(tmp_path / "tasks.db")
    app, _ = _app(tmp_path, store=store)
    store.enqueue("t", "do it", str(tmp_path), [], [])
    assert store.claim_next() is not None
    started = time.monotonic()
    _, frames = _collect_events(app, want=3)
    statuses = [d for e, d in frames if e == "status"]
    assert len(statuses) >= 3 and all(d["queue"]["running"] == 1 for d in statuses)
    assert time.monotonic() - started < 2.0


def test_events_beat_while_a_load_is_under_way(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.2)
    started = threading.Event()
    release = threading.Event()

    def slow_factory(model_id: str):
        started.set()
        release.wait(5.0)
        return FakeEngine([])

    app, engines = _app(tmp_path, factory=slow_factory)
    loader = threading.Thread(target=engines.get, daemon=True)
    loader.start()
    assert started.wait(2.0)
    try:
        _, frames = _collect_events(app, want=3)
    finally:
        release.set()
        loader.join(5.0)
    statuses = [d for e, d in frames if e == "status"]
    assert len(statuses) >= 3 and all(d["engine"]["loading"] for d in statuses[:2])


def test_a_load_a_hold_a_release_a_task_change_and_an_unload_each_reach_the_client(
    tmp_path: Path, monkeypatch
):
    """None of these is a registry change; each used to ride the heartbeat."""
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(monitor, "EVENT_IDLE_TICK_SECONDS", 0.05)
    alive = {"ok": True}
    store = TaskStore(tmp_path / "tasks.db")
    app, engines = _app(
        tmp_path, alive=lambda pid, started: alive["ok"], store=store, idle_minutes=0
    )

    def load() -> None:
        engines.get()

    _, frames = _collect_events(app, want=2, during=load)
    assert [d["engine"]["loaded"] for e, d in frames if e == "status"] == [False, True]

    _, frames = _collect_events(app, want=2, during=lambda: engines.hold(4242, 1.0))
    assert [d["engine"]["holders"] for e, d in frames if e == "status"] == [0, 1]

    def release() -> None:
        alive["ok"] = False

    _, frames = _collect_events(app, want=2, during=release)
    assert [d["engine"]["holders"] for e, d in frames if e == "status"][-1] == 0

    _, frames = _collect_events(
        app, want=2, during=lambda: store.enqueue("t", "do it", str(tmp_path), [], [])
    )
    assert [d["queue"]["queued"] for e, d in frames if e == "status"] == [0, 1]

    time.sleep(0.01)  # idle_unload_minutes=0 still needs the clock to have moved
    _, frames = _collect_events(app, want=2, during=engines.unload_if_idle)
    assert [d["engine"]["loaded"] for e, d in frames if e == "status"] == [True, False]
```

The `release` case needs the sweep or a status read to prune the holder: `_status_events` builds a document only when `version()` moved, and pruning happens inside `engines.status()`, which the build calls — so the holder's departure is noticed by whichever build runs next. That build must be caused by something: make `release()` also call `engines.status()` (the launcher's own `/sous/status` poll and the worker's sweep do this in the daemon):

```python
    def release() -> None:
        alive["ok"] = False
        engines.status()
```

Rewrite `test_events_ping_on_the_configured_cadence` so the collector does not depend on the heartbeat:

```python
def test_events_ping_on_the_configured_cadence(tmp_path: Path, monkeypatch):
    """The ping is sse-starlette's, on the shared interval, whether or not
    a document goes out."""
    monkeypatch.setattr(monitor, "EVENT_PING_SECONDS", 0.2)
    app, _ = _app(tmp_path)
    _, frames = _collect_events(app, want=None, seconds=0.7)
    assert ("ping", {"type": "ping"}) in frames
```

`_app` in `tests/test_monitor.py` needs three new keyword parameters: `factory=None` (used as `engine_factory=factory or (lambda mid: FakeEngine([]))`), `idle_minutes=None` (passed as `idle_unload_minutes=idle_minutes` into the `SousConfig(...)` call when not None) and `store=None` (used as `store = store or TaskStore(tmp_path / "tasks.db")` — the version counter is per instance, so a test that writes must write through the app's own store, the way `inflight=` already hands in the registry). `threading` and `TaskStore` are already imported there.

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_monitor.py -p no:cacheprovider -k "quiet_daemon or beat_ or each_reach or ping_on"`
Expected: `quiet_daemon` FAILS (several `status` frames arrive); `each_reach` FAILS (`EVENT_IDLE_TICK_SECONDS` missing, then the `hold`/`store` cases stall until the collector's timeout once Task 3's version poll exists); the busy-beat tests pass already (the old unconditional heartbeat covers them) and must keep passing.

- [ ] **Step 4: Implement**

In `src/sous/monitor.py`, after `EVENT_HEARTBEAT_SECONDS = 1.0`:

```python
# Idle — no turn in flight, no load under way — nothing on the registry
# moves without a turn starting, so the poll slows to this: a new order is
# on screen within half a second, and the loop costs a fifth of the 10 Hz
# one. Busy, the fast tick is what puts a token delta on the wire in time.
EVENT_IDLE_TICK_SECONDS = 0.5
```

Replace the comment above `EVENT_HEARTBEAT_SECONDS` with:

```python
# While a turn is in flight, a delegated task is running or a load is
# under way the document also goes out this often unchanged: the terminal
# reads a silent stream during a turn as a stalled daemon, a prefill runs
# for a minute without a registry change, and a task's clock and the cache
# counters it moves are in the document while the worker writes the store
# once per tool call. Idle there is no heartbeat at all — a load, a hold
# and a task change bump a version of their own, and the idle clock is the
# terminal's to run.
```

Replace `_status_events`:

```python
def _busy(document: dict) -> bool:
    # A delegated task counts: the worker never writes the registry, and
    # `running` includes a task awaiting approval, whose clock runs too.
    engine = document.get("engine") or {}
    queue = document.get("queue") or {}
    return bool(document.get("inflight") or queue.get("running") or engine.get("loading"))


async def _status_events(
    status: Callable[[], dict], version: Callable[[], object]
) -> AsyncIterator[ServerSentEvent]:
    """The document now, then again whenever `version()` moved since the
    last one went out — checked every tick — and, only while the last
    document showed a turn, a running task or a load, at least once a
    heartbeat regardless.
    Built on a worker thread each time. Nothing is buffered for a client
    that is gone: the response cancels this generator on disconnect, and
    the tick's sleep is where that lands. A failure building or encoding
    the document ends the stream rather than raising through uvicorn: the
    client sees the connection close and redials, the same as any other
    daemon failure logs its type and never its message."""
    seen: object = None  # nothing sent yet, whatever the clock says
    sent = time.monotonic()
    busy = True
    while True:
        # Read before the build, not after: a change landing during it is
        # then a newer value on the next check, never one masked.
        current = version()
        now = time.monotonic()
        if current != seen or (busy and now - sent >= EVENT_HEARTBEAT_SECONDS):
            try:
                document = await run_sync(status)
                # Encoded under the same guard, and as strictly as /sous/status
                # (Starlette's JSONResponse refuses non-finite numbers): a value
                # json cannot serialize would otherwise leave as a traceback
                # through uvicorn, and a NaN as a token no other parser reads.
                data = json.dumps(document, separators=(",", ":"), allow_nan=False)
            except Exception as e:  # noqa: BLE001 — logged and the stream ends, never raised
                _logger.error(f"GET /sous/events failed ({type(e).__name__})")
                return
            yield ServerSentEvent(event="status", data=data, sep=_SEP)
            seen, sent = current, now
            busy = _busy(document)
        await asyncio.sleep(EVENT_TICK_SECONDS if busy else EVENT_IDLE_TICK_SECONDS)
```

- [ ] **Step 5: Run the monitor tests, then the suite**

Run: `uv run pytest tests/test_monitor.py -p no:cacheprovider`
Expected: all pass, including `test_events_follow_registry_changes_coalesced` (its burst ends with a turn in flight, so the stream is busy) and `test_events_document_is_built_off_the_event_loop`.
Run: `uv run pytest -m "not model" -p no:cacheprovider` — all pass; `uv run ty check`; `uv run ruff check . && uv run ruff format --check .`.

- [ ] **Step 6: Commit**

```bash
git add src/sous/monitor.py tests/test_monitor.py
git commit -m "perf(monitor): send the status document only when something changed while idle" -m "The once-a-second heartbeat rebuilt, encoded and sent the whole document per connected client — a quarter of a percent of a core each on an idle daemon — to carry a load, a hold and the idle clock, none of which the registry announces. The stream now polls the composite version (registry, engine, task store) and keeps the heartbeat only while a document shows a turn, a running task or a load, where the terminal's stall detection and task clock need it and the cost is nothing beside a generation. Idle, the poll slows to half a second and only the ping goes out."
```

---

### Task 5: `sous top` runs the idle clocks itself

**Files:**
- Modify: `src/sous/tui.py` (`Top.__init__`, `_apply`'s engine read and idler rule, `_refresh_footer`, `_idle_tick`, new `_engine_now` and `_show_idle`; the comment block above the widgets that says the feed sends a document every second)
- Test: `tests/test_tui.py` (two tests rewritten, one added)

**Interfaces:**
- Consumes: the idle stream contract from Task 4 (no documents while idle; `idle_seconds` in the last one).
- Produces: nothing other modules consume.

- [ ] **Step 1: Write the failing tests**

Replace `test_a_quiet_kitchens_idle_tick_only_moves_the_steam` with:

```python
def test_a_quiet_kitchens_idle_tick_moves_the_steam_and_the_clock():
    """The daemon sends no document while nothing moves, so the idle clock
    is the terminal's to run: from the last document's idle_seconds plus
    the time since it arrived, advanced on the idle tick, and painted only
    when the second it shows changes."""

    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        document = _doc(None, recent=RECENT)
        document["engine"]["holders"] = 0  # the countdown is drawn only unheld
        await _deliver(feed, pilot, document)
        body = _plain(app, "#card-body")
        assert "no orders on the rail for 4m 12s" in body
        assert "LIGHTS OUT in 25m 48s" in body
        steam_0 = _plain(app, "#card-art")
        clock.now = BASE + 0.5
        await pilot.pause(0.6)
        steam_1 = _plain(app, "#card-art")
        assert steam_1 != steam_0
        assert _plain(app, "#card-body") == body
        clock.now = BASE + 1.0
        await pilot.pause(0.6)
        steam_2 = _plain(app, "#card-art")
        assert steam_2 == steam_0 and steam_2 != steam_1
        body_1 = _plain(app, "#card-body")
        assert "no orders on the rail for 4m 13s" in body_1
        assert "LIGHTS OUT in 25m 47s" in body_1
        assert "idle 4m 13s" in app.query_one(tui.LinePanel).border_subtitle
        assert "KITCHEN QUIET 4m 13s" in app.query_one(tui.Strip).render().plain
        # Below 60 columns the footer is the only clock on screen.
        app._width = 50
        app._refresh_footer()
        assert _plain(app, "#footer") == " KITCHEN QUIET 4m 13s  q"

    _run(test)
```

Replace `test_a_heartbeat_repaints_only_the_lines_whose_words_changed` with:

```python
def test_the_idle_tick_repaints_only_the_lines_whose_clock_changed(monkeypatch):
    """A whole screen redrawn every second is 2–6 % of a core. With motion
    off the idle tick still runs against a live daemon — the clock must
    move — but it may repaint only the lines that show the clock, and only
    when the second they show changes."""
    painted: list[str] = []
    real_refresh = tui.Widget.refresh

    def spy(self, *args, **kwargs):
        painted.append(f"{type(self).__name__}#{self.id}")
        return real_refresh(self, *args, **kwargs)

    async def test(app, pilot, feed, clock):
        app.motion = False
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        assert app._idler is not None and app._idler._active.is_set()
        assert "no orders on the rail for 4m 12s" in _plain(app, "#card-body")
        monkeypatch.setattr(tui.Widget, "refresh", spy)
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        assert painted == [], f"a document that changed nothing repainted {painted}"
        clock.now = BASE + 0.5
        await pilot.pause(0.6)
        assert painted == [], f"a tick inside the same second repainted {painted}"
        for n in (1, 2, 3):
            painted.clear()
            clock.now = BASE + n
            await pilot.pause(0.6)
            # The card's own `since …` subtitle counts up with the same clock.
            assert len(painted) <= 4, painted
            assert "Line#card-body" in painted and "Card#card" in painted, painted
            assert f"no orders on the rail for 4m {12 + n}s" in _plain(app, "#card-body")
        # The memo is at the widget, not over the document: a turn arriving
        # still draws the slip.
        painted.clear()
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT))
        assert "Line#headline" in painted and "Slip#slip" in painted, painted
        assert app.query_one(tui.Slip).display
        assert _plain(app, "#headline").startswith(" PLATING  decode 412 tok")

    _run(test)
```

(The `_deliver` calls above happen at `clock.now == BASE`, so a re-delivered document restarts the extrapolation at the same instant and changes nothing.)

Add:

```python
def test_the_idle_clock_does_not_run_while_the_feed_is_down():
    """A stale document's idle clock is a number about a daemon that is
    gone; the card shows the redial countdown instead, and the panel
    subtitle says when the document arrived."""

    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        await _deliver(feed, pilot, None)
        clock.now = BASE + 2.0
        await pilot.pause(0.6)
        assert "4m 12s" not in _plain(app, "#card-body")
        assert "as of" in app.query_one(tui.LinePanel).border_subtitle

    _run(test)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_tui.py -p no:cacheprovider -k "idle_tick or idle_clock"`
Expected: the first two FAIL (the clock does not advance; with motion off the idler is paused); the third passes already and pins the disconnected case.

- [ ] **Step 3: Implement**

In `Top.__init__`, beside `self._received_at = 0.0`:

```python
        # The idle second last painted, so the tick repaints the clock's
        # lines once a second and never twice for the same words.
        self._idle_shown: int | None = None
```

In `_apply`, change `engine = document.get("engine") or {}` to

```python
        # On a fresh document the offset is zero; on a resize it is the
        # clock already on screen, never the document's older one.
        engine = self._engine_now(now)
        # The document paints its own second; the tick repaints only once
        # the clock has moved past it.
        idle_now = engine.get("idle_seconds")
        self._idle_shown = int(idle_now) if idle_now is not None else None
```

(outside the `if not resize:` block, which sets `self._document` and `self._received_at` before it); and change the idler rule at the end of `_apply` to:

```python
        if self._idler is not None:
            # Idle against a live daemon the tick runs the idle clock — the
            # daemon sends no document while nothing moves — and
            # disconnected it is the only thing that moves the redial
            # countdown, so it runs whenever nothing is on the pass, with
            # the motion flag deciding only what it animates.
            self._idler.resume() if not busy else self._idler.pause()
```

Add to `Top`, before `_idle_tick`:

```python
    def _engine_now(self, now: float) -> dict:
        """The last document's engine block with its idle clock advanced by
        the time since that document arrived. The daemon sends no document
        while nothing moves, so between them the clock is the terminal's;
        anything that would reset it (a turn, a hold leaving) is a document."""
        engine = dict(self._document.get("engine") or {})
        idle = engine.get("idle_seconds")
        if idle is not None and self._connected:
            engine["idle_seconds"] = idle + max(0.0, now - self._received_at)
        return engine

    def _show_idle(self, engine: dict, now: float) -> None:
        """Repaint the lines that show the idle clock — the quiet card's
        span, its lights-out countdown and its `since …` subtitle, THE
        LINE's subtitle, the headline, the narrow footer — from the
        advanced clock. Each widget skips text that did not change, so
        this is three or four repaints a second."""
        document = self._document
        config = document.get("config") or {}
        recent = document.get("recent_turns") or []
        queue = document.get("queue") or {}
        card = self.query_one(Card)
        if card.display and not engine.get("loading"):
            card.show_quiet(engine, config, recent, now, self.motion)
        mood = self._mood(engine, now)
        self.query_one(LinePanel).show(mood, engine, config, queue, recent, now, self.motion)
        headline, colour = self._headline(engine, recent, now)
        self.query_one(Strip).show(headline, colour)
        self._refresh_footer()
```

In `_refresh_footer`, change `engine = self._document.get("engine") or {}` to `engine = self._engine_now(now)` so the narrow footer shows the same advanced clock.

Replace `_idle_tick`:

```python
    def _idle_tick(self) -> None:
        now = self._clock()
        try:
            engine = self._engine_now(now)
            mood = self._mood(engine, now)
            self.query_one("#line-chef", Chef).show(mood, now, motion=self.motion)
            card = self.query_one(Card)
            if card.display:
                # Only the chef's motion, the pot's steam, and — disconnected
                # — the redial/closed countdown, never the rest of the
                # card's text: that only a real document, a retry or the
                # idle clock's next second changes.
                wait = max(0.0, (self._retry_at or now) - now) if not self._connected else None
                card.tick(now, motion=self.motion, wait=wait)
            if self._connected and self._turn is None and engine.get("idle_seconds") is not None:
                second = int(engine["idle_seconds"])
                if second != self._idle_shown:
                    self._idle_shown = second
                    # The whole second, so the span and the countdown it is
                    # subtracted from never disagree by one.
                    engine["idle_seconds"] = float(second)
                    self._show_idle(engine, now)
        except NoMatches:
            return
```

Update the comment block above the widgets section (the one beginning "The feed sends a status document every second whether or not anything on it moved") so its first sentence reads "The feed sends a status document whenever something on it moved — while a turn is in flight, every second — and a document is applied to every widget at once."; keep the rest of that block's reasoning about repaints as it stands.

- [ ] **Step 4: Run the TUI tests, then the suite**

Run: `uv run pytest tests/test_tui.py -p no:cacheprovider`
Expected: all pass (`test_a_quiet_kitchen_counts_down_to_lights_out_only_when_nothing_holds_the_model` still passes: `_deliver` lands at `BASE`, so the advanced clock equals the document's).
Run: `uv run pytest -m "not model" -p no:cacheprovider`; `uv run ty check`; `uv run ruff check . && uv run ruff format --check .`.

- [ ] **Step 5: Commit**

```bash
git add src/sous/tui.py tests/test_tui.py
git commit -m "feat(tui): run the idle clocks from the last document and the terminal's clock" -m "The daemon no longer sends a document a second while nothing moves, so the quiet card's span, its lights-out countdown, THE LINE's subtitle and the headline advance on the idle tick from the last document's idle_seconds plus the time since it arrived, repainting only when the second they show changes. The idle tick now runs whenever nothing is on the pass; the motion flag decides only what it animates."
```

---

### Task 6: Docs and the M5 Pro gate

**Files:**
- Modify: `README.md` (the `GET /sous/events` bullet), `CLAUDE.md` (the `/sous/events` sentence in the in-flight registry gotcha)
- No code.

- [ ] **Step 1: README**

Replace the `GET /sous/events` bullet with:

```markdown
- `GET /sous/events` — the same document as a Server-Sent Events stream:
  once at connect, then whenever something on it changed — the turn in
  flight (at most ten times a second), a load, an unload, a hold or its
  release, a task — and, only while a turn is in flight, a delegated task
  running or a load under way, at least once a second; idle, nothing but
  a `ping` every 10 s.
```

- [ ] **Step 2: CLAUDE.md**

Replace the sentence that begins "`/sous/events` polls `Inflight.version` ten times a second" (through "never wake it from a writer.") with:

```markdown
  `/sous/events` polls `SousService.status_version()` — the registry's
  version, `EngineManager.version` (load, unload, hold, release, and
  every idle-clock reset: `touch()` and a `get()` hit) and
  `TaskStore.version` (any connection that changed a row) — ten times a
  second while the last document showed a turn, a running task
  (`queue.running`) or a load, and twice a second otherwise, rebuilding
  the whole document on a worker thread when the tuple moved; the
  once-a-second heartbeat runs only while busy (the TUI reads a silent
  stream during a turn as a stalled daemon, and a task's clock is in the
  document), never idle — the idle clock is the TUI's to run
  (`Top._engine_now`, seeded by every reset the engine version carries).
  The build memoises the task counts and listing on the store's version
  (two slots: the narrow MCP build never reads the listing) and the
  allowlist on the config file's mtime; summaries are per build. Never
  wake the loop from a writer.
```

- [ ] **Step 3: Checks and commit**

Run: `uv run ruff check . && uv run ruff format --check .` (CLAUDE.md and README are not linted; this is for the tree), then:

```bash
git add README.md CLAUDE.md
git commit -m "docs: the event stream sends on change and beats only while busy"
```

- [ ] **Step 4: The gate (manual, M5 Pro, after the whole branch is reviewed)**

On the M5 Pro, from `~/code/personal/sous-remote` on the branch, with a throwaway daemon (HOME under `/tmp`, `[server] port = 8385`, the checkout's venv python, model never loaded), measure CPU-time deltas over 45 s windows with `ps -o cputime=` before and after (`(after − before) / 45 × 100` = % of a core):

1. daemon idle, no client;
2. daemon + one `curl -sN http://127.0.0.1:8385/sous/events > /dev/null`;
3. daemon + `sous top` (the installed tool with the same HOME, inside the GUI `screen` session; measure the python process — `pgrep -n -f "sous top"` returns the newest match, which is the shell wrapper's child; take the pid whose command line starts with the tool venv's python) — both the daemon's and the terminal's figures;
4. then, with the model loaded and one `sous claude -p "reply with exactly: ok" --model sous-local` turn running against the live daemon, confirm `sous top` shows the prefill and decode phases (the heartbeat while busy) and that a load, an unload (set `idle_unload_minutes = 1` on the throwaway) and a hold each appear on `sous top` within a second.

Acceptance (issue #96): 2 and 3 well under 1 % — expected about 0.55 %, of which 0.45 % is the worker poll measured in 1, to be reported separately; each transition in 4 visible within a second. Record the table in the PR body with the recipe; no name, no session link.
