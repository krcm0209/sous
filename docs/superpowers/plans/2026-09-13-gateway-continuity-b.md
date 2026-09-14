# Gateway Cache Continuity B — Preload and Hold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `sous claude` session never pays a model load or a from-zero prefill because the daemon went idle between two of its subagents: launching `sous claude` preloads the model and holds it for as long as that Claude Code process lives, and the launcher's preflight stops speaking MCP.

**Architecture:** Three pieces, none on the request→render→cache path. (1) `EngineManager` gains a holder registry keyed by PID + process start time, a `hold()` that starts one detached preload thread when nothing is loaded, and an idle sweep that prunes dead holders (PID gone, or reused — start time differs) and refuses to unload while any remain; the manager stops holding its lock across a model load or unload so `status()` and `hold()` answer during one. (2) A new daemon-level route module `sous/monitor.py` mounts `GET /sous/status`, `POST /sous/hold` and a `/sous/` 404 on the daemon's Starlette app before the gateway's catch-all, whatever the gateway flag says, behind the same loopback Host/Origin guard the gateway uses (moved to `sous/loopback.py` so both share it). (3) `sous claude` fetches `/sous/status` with `httpx`, treats a 404 as "daemon predates this CLI" (exit 1, no compatibility mode), posts `/sous/hold` with its own PID and start time, prints one line, and `execve`s — the holder *is* the Claude Code process.

**Tech Stack:** Python 3.14, `threading.Condition`, `httpx` and `psutil` (both already dependencies), Starlette routes via the MCP SDK's `custom_route`, pytest with the existing `FakeEngine`/`FakeUpstream` seams and a stdlib `http.server` for the launcher's HTTP tests. No new dependencies. No mlx code changes beyond the preload thread's `release_mlx_thread_state()`.

**Spec:** `docs/superpowers/specs/2026-09-10-gateway-cache-continuity-design.md` — "Design" B (engine manager, route, launcher, "Not in scope, stated"), "Observability hooks this spec adds", "Invariants preserved", "Testing" B and the real-daemon gate, "Documentation", "Risks and open questions" (PID watch cadence, daemon restart). Part A landed as #93 (`bda9ebc`). The observability spec (`…-gateway-observability-design.md`, PR 2 section) extends `sous/monitor.py` and `/sous/status` later; this plan creates the module and the route it extends. PR 2's document reshapes today's `model` key into `engine`, which will also touch the launcher's one read of it. Section "What the continuity A gate added" below is this plan's own finding, folded in where it changes a task or the gate.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`; `ty` flags an *unused* pragma as its own diagnostic — add one only when `ty check` names the rule.
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local. Every thread that touches mlx calls `engine.base.release_mlx_thread_state()` before it exits (mlx ≥ 0.32.1 segfaults the daemon otherwise); the preload thread this plan adds loads the model, so it is such a thread.
- The gateway and the new routes never log a request body, a header value or a query string. The only body-derived value any new line carries is a PID (not a secret). Nothing that blocks (a database read, a psutil call) runs on the event loop: route handlers hand such work to a thread.
- Prompt-cache slots are owned by the thread that built them; the preload thread never builds a slot (it calls `get()` and exits).
- `sous claude` sets no `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or `ANTHROPIC_DEFAULT_*_MODEL`, and never a global Claude Code compaction variable (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `DISABLE_AUTO_COMPACT`, `DISABLE_COMPACT`, `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`): each is global and would change the frontier main loop's behaviour. The hold request carries no credential.
- No compatibility fallbacks in the launcher: a daemon that does not answer `/sous/status` is one to restart, not one to talk MCP to (Kyle's ruling, 2026-09-10; CLI and daemon ship as one package at major version 0).
- Tests never touch the real `~/.sous` or `~/.claude`: every path comes from `tmp_path`; every port from a free-port helper; every HTTP call in a test goes to an in-process app or a server the test started.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` (this plan and its spec included) once committed.
- **No plan language in committed code.** Code and test comments/docstrings must never cite a spec, a plan, a task or a step ("Task 2's helper", "Spec:", "the spec promises", "docs/superpowers/…"). Where a code block below carries such a reference, restate the underlying fact instead. Sweep the branch's own additions before the PR (the tree already carries a few `Phase 1` / `gate 2` / `decision 10` references from earlier work — out of scope, leave them): `git diff main -- src tests | grep -nE '^\+.*(Task [0-9]|Step [0-9]|[Dd]ecision [0-9]|Phase [0-9]|gate [0-9]|[Ss]pec[: ]|docs/superpowers)'` must print nothing.
- Never loosen an existing assertion to make a test pass. When this plan changes a behaviour an existing test pinned (named per task below), restate the test's expected value and its docstring; if a test's premise is gone, delete it and say so in the commit body.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>` — model-less, whatever the harness suggests. No session link anywhere.
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` (never pipe without `pipefail`; `pyproject.toml` already sets `-q`, so never add another `-q` — it suppresses the summary line), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`. Line length is 100 and `E501` is enforced: `ruff format` reflows code but never splits a string literal, so a long expected string is written as adjacent literals over two lines.
- The suite on `main` at `bda9ebc` collects 978 tests under `-m "not model"`: `949 passed, 29 deselected` on macOS 26.2+, and `941 passed, 8 skipped, 29 deselected` on this Mac (macOS 15.5 — eight `tests/test_int8prefill.py` cases skip without tensor units). Each task's "Expected" line below gives the delta and the this-Mac line; on a machine with tensor units `passed` is 8 higher and there are no skips.
- Branch: `feat/gateway-continuity-b` off `main` at `bda9ebc`. `main` is protected: PR with all four CI jobs green.
- Kyle's daemon (on the M5 Pro, unmanaged, pid in `~/.sous/daemon.lock`) is not restarted by any task. The real-daemon gate at the end is the orchestrator's, after Kyle's explicit go-ahead, since it installs branch code over his working `sous`.

---

## What the continuity A gate added

The A gate (2026-09-13, `claude -p` under `sous claude`, one background general-purpose subagent, session `81e23232`) logged six `sous-local` turns for a four-turn agent. Reconciled against the subagent's transcript (`~/.claude/projects/…/subagents/agent-ab3fb4f17cdc9756e.jsonl`, metadata only) and Claude Code 2.1.269's binary:

| call | log line | what it was |
|---|---|---|
| 1–4 | miss, then `cache=hit took=turn@N` ×3, `stop=tool_use` | the agent's real turns |
| 5 | `cache=hit took=turn@78611 prefilled_tokens=1367 output_tokens=3442 stop=end_turn seconds=215.1` | **the auto-compaction summary call**: the transcript's `compact_boundary` entry has `trigger: "auto"`, `preTokens: 95050`, `postTokens: 23938`, `durationMs: 215102` — this call to the token |
| 6 | `cache=fork took=fork@50312 prefilled_tokens=29441 output_tokens=839 seconds=163.1`, same `tools=`/`system=` hashes as call 5 | **the post-compaction continuation**: the summary as a user message plus Claude Code's re-attachments (CLAUDE.md, the agent listing, re-read files), then the agent's final report |

Two calls, 378 s of a 654 s session (58%), zero subscription tokens: a wall-clock and answer-quality cost (the agent reported from a summary of itself), not a plan-economics one. Neither is a cache defect: call 5 extended the conversation's own slot and paid 1.4K tokens of prefill plus 210 s of decode at the model's 16 tok/s; call 6 is genuinely new text after the tools boundary (the header fork would have saved ~700 tokens). What is wrong is that the compaction fired at all with a 131,072-token window.

**Why it fired (read from the binary):** Claude Code's auto-compact threshold for a model is `effective_window − 13000`, where `effective_window = window − min(max_output_tokens, 20000)` — for `sous-local` under `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072` that is 98,072 — and a *precompute* trigger can fire earlier still, at `effective_window × (1 − fraction)` with a fraction taken from remote configuration keyed by window size. The count it compares is Claude Code's own tally (95,050 when the API's total for the last real turn was 78,739). So every local general-purpose subagent that runs four or five tool turns compacts today; a frontier subagent with the same conversation has a 200K window and does not. The switches are all global: `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (a minimum with the model's window, so it cannot *raise* the local threshold and would lower the main loop's), `DISABLE_AUTO_COMPACT`/`DISABLE_COMPACT`, the `autoCompactEnabled` setting, `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` (a test override). None may be set by `sous claude` (Global Constraints). **The one per-local-model lever is the window itself:** `[gateway].max_context_tokens`, which the launcher already passes as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. The default model's native context is 262,144 (`text_config.max_position_embeddings`), so `262144` is a legal value; it raises the classic threshold to 229,144 and the precompute one proportionally, and costs the auto prompt-cache budget one more window of KV reserve (+8 GiB at 64 KiB/token — the README's "about 27 GiB on a 64 GB machine" becomes about 19 GiB, still above the ~14 GiB it names for a fork beside a retaining conversation).

One caveat the documentation must carry: the observed trigger was the *precompute* one (95,050 is below the classic 98,072), and its fraction is keyed by window size in remote configuration, so where a subagent actually compacts at 262144 is not known until measured — the classic arithmetic is the upper bound, not a prediction.

**What this plan does with it:** no code. Task 5 documents the arithmetic and the lever in the README's `max_context_tokens` comment and under "What a locally served turn gives up", and adds one CLAUDE.md line so nobody reaches for a global variable. The real-daemon gate (orchestrator step 3) names compaction pairs explicitly, reports them separately from the per-turn cache criterion, and asks Kyle whether to run with `max_context_tokens = 262144` first so the gate measures this plan. A follow-up for the observability spec's PR 3: `sous turns` can mark compaction boundaries from the subagent transcript's `compact_boundary` metadata, which the turn line cannot see (same system, same tools).

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/engine/base.py` | `EngineManager`: `_loading`/`_unloading` flags on a `Condition` so `get()`/`unload_if_idle()` never hold the lock across a load or unload; `hold(pid, create_time)`, `_run_preload`, `_prune_holders`; `status()` gains `loading` and `holders`; `holder_alive`/`clock` injection; module-level `_holder_alive` (psutil). |
| `src/sous/loopback.py` | `ALLOWED_HOSTS`, `ALLOWED_ORIGIN_HOSTS`, `check_loopback(request)` — moved from `gateway/routes.py`, imported by it and by `monitor.py`. |
| `src/sous/monitor.py` | `mount_monitor(mcp, engines, status)`: `GET /sous/status`, `POST /sous/hold` (1 KiB cap, strict shape), `/sous/{path}` → 404; `HOLD_BODY_LIMIT`. |
| `src/sous/server.py` | `create_server` mounts the monitor before the gateway, unconditionally. |
| `src/sous/gateway/routes.py` | Loses `_check_loopback` and the two allow-lists; imports `check_loopback`. |
| `src/sous/gateway/turn.py` | One comment: `get()` (not the lease) waits out another thread's load. |
| `src/sous/cli.py` | `_daemon_status` over `GET /sous/status`; `_hold` over `POST /sous/hold`; MCP client (`_server_status`, `_failure_names`) and the pre-routing branch deleted; the preload line. |
| `tests/test_engine_base.py`, `tests/test_monitor.py` (new), `tests/test_loopback.py` (new, small), `tests/test_gateway_routes.py`, `tests/test_cli.py` | Tests per task below. |
| `README.md`, `CLAUDE.md` | Docs per Task 5. |

## Decisions this plan makes beyond the spec's text

1. **`EngineManager` never holds its lock across a model load or unload.** Today `get()` runs the factory under `_lock` and `unload_if_idle()` runs `engine.unload()` under it — minutes for the 27B. The spec's `status()` gains `loading`, which is only observable if `status()` answers *during* the load; and `hold()` must return to the launcher inside its 15 s timeout while the worker or a gateway turn is mid-load (otherwise the hold is lost in exactly the case it exists for). A `_loading`/`_unloading` pair on a `threading.Condition` over the same lock keeps every invariant: one load at a time, no load during an unload, no unload during a load, no unload under a generation or lease. `lease()` keeps its meaning (a cross-call pin on a *loaded* engine); the turn runner's `get()` now does the waiting the lease comment attributed to entering the lease, and `load_seconds` is measured from `started` so its value is unchanged. The one thing that does run under the lock and did not before is the psutil liveness check per holder in the sweep (`_prune_holders`): sub-millisecond, once per poll, and it must see a consistent registry.
2. **The loopback guard becomes `sous/loopback.py`.** `_check_loopback` and its allow-lists move out of `gateway/routes.py` unchanged in behaviour: the monitor must not import the gateway's *route* module (sse-starlette, the upstream forwarder) for a 25-line check, and the gateway must not import the monitor. Both still import `RequestError` from `gateway/convert.py` — the error vocabulary is the API's and lives with the request parser. The error message changes from "the gateway serves loopback hosts only" to "loopback hosts only"; the one existing test that pins the allow-list by name (`test_host_header_must_be_loopback`) moves that pin to the new module.
3. **`/sous` and `/sous/{path:path}` → 404, registered by the monitor.** The spec says `/sous/anything-else` is never forwarded upstream; the SDK appends custom routes in registration order after `/mcp`, so the monitor mounting before the gateway is what guarantees it, and an explicit 404 route (Anthropic-shaped `not_found_error`) makes it true with the gateway on *and* testable with it off. `/sous/{path:path}` does not match a bare `/sous`, and with the gateway on that path would full-match the gateway's catch-all and be forwarded with the client's credentials — hence the second registration.
4. **Route handlers run `status()` and `hold()` on a worker thread** (`from anyio.to_thread import run_sync` — `ty` cannot resolve `anyio.to_thread.run_sync` through a bare `import anyio`, since `anyio/__init__` answers unknown attributes with a placeholder type; `src/sous/proxy.py` already imports it this way): `server_status()` reads SQLite for the queue counts and `hold()` takes the manager lock; the same rule `_log_turn` lives under. The MCP SDK already runs the sync `server_status` tool on that same anyio pool, so `_mlx_memory_gb()`'s per-call `release_mlx_thread_state()` on a pooled thread is today's behaviour, not new; nothing this plan dispatches there keeps mlx state between calls, and PR 2 must keep it that way.
5. **Liveness and time are injected.** `EngineManager(config, engine_factory, *, holder_alive=None, clock=time.monotonic)`; the default `_holder_alive(pid, create_time)` is psutil (`Process(pid).create_time()` within 1 s of the recorded value; any `psutil.Error` — no such process, access denied, zombie — is "gone"). Tests pass a dict-backed liveness function and a settable clock instead of sleeping or faking psutil; the psutil helper has its own tests (this process is alive under its real start time, dead under a start time one hour off, a pid nothing wears is dead, and an `AccessDenied` is dead).
6. **The launcher's pre-routing check is deleted with its test.** A daemon that answers `/sous/status` has `[gateway].upstream_url` by construction (routing landed three releases before this route), so `"upstream_url" not in gateway` cannot be true; the 404 path is the one message for every older daemon. (An older daemon with the gateway on forwards the path to `api.anthropic.com`, which answers an unauthenticated `GET /sous/status` with 404 — checked 2026-09-13; with the gateway off, Starlette 404s. Either way the message is "restart".)
7. **No second "model loaded in N s" line.** The spec lists one; `get()` already logs `model_load seconds=… model=…` (the observability PR 1's bracket) on every load, preload included. Adding a duplicate would put two load lines on a cold start. `hold()` logs `preloading <model>` after starting the thread and outside the lock, so on a fast load that line can land *after* `model_load seconds=`; the gate asserts that all three lines precede the first turn, not their order.
8. **`sous claude --help`, `-h`, `--version` and `-v` as the first argument skip the hold.** Those four make `claude` exit at once; holding for them would start a multi-minute 27B load nobody waits for and, because the prune restarts the idle clock, leave the weights resident for a fresh `idle_unload_minutes`. Anything else (`claude mcp list`, `claude doctor`, …) the launcher cannot classify and holds like a session — documented, not engineered around. The preflight's checks run for every invocation, as today.
9. **`hold()` during an unload starts the preload thread anyway** — its `get()` waits for the unload to finish and then loads. The launcher sees `loading: true` and prints "preloading".
10. **The sweep prunes holders first, before every early return.** The spec says "before the idle test"; placing the prune after the `_engine is None` guard would leave a holder registered forever after a failed preload (nothing loaded, so the sweep returns early), and skip it under every generation or lease. The idle-clock restart writes `_last_used` directly: `touch()` takes the same non-reentrant lock the sweep already holds.
11. **The preload thread releases its mlx state right after `get()` returns, on the thread that loaded the weights.** That is the turn runner's precedent (`gateway/turn.py`, end of `run`: "engines.get() may have loaded the model on this thread … keeping it unconditional is what makes it checkable"): the weights are arrays other threads use afterwards, the streams are the loading thread's own. `measure_cache_budget`'s "no release here" rule is about releasing *inside* the load, while the caller's streams are live. CI cannot exercise the real call (fake engines), so the gate's preload criterion is a completed generation after a preload, not just a load.
12. **The compaction finding changes documentation and the gate, not code** (section above).

---

### Task 1: `EngineManager` never holds its lock across a load or an unload (`base.py`)

**Files:**
- Modify: `src/sous/engine/base.py` (`EngineManager` at lines 469–571), `src/sous/gateway/turn.py` (the comment at lines 158–160)
- Test: `tests/test_engine_base.py` (new tests after `test_get_logs_the_load_once_with_its_duration`, line 96)

**Interfaces:**
- Consumes: `ManagedEngine`, `_logger`, `SousConfig.idle_unload_minutes` (all present).
- Produces: `EngineManager._changed: threading.Condition` (over `_lock`), `EngineManager._loading: bool`, `EngineManager._unloading: bool`; `status()` dict gains `"loading": bool`. `get()`, `touch()`, `lease()`, `unload_if_idle()`, `status()` keep their signatures. Task 2 builds on `_changed` and `_loading`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py` after `test_get_logs_the_load_once_with_its_duration`:

```python
class _GatedFactory:
    """A model factory that blocks until released, so a test can look at the
    manager mid-load. `started` is set once the factory has been entered."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.created: list[FakeEngine] = []

    def __call__(self, model_id: str) -> FakeEngine:
        self.started.set()
        assert self.release.wait(10)
        engine = FakeEngine([])
        self.created.append(engine)
        return engine


def _gated_manager(idle_minutes: int = 30) -> tuple[EngineManager, _GatedFactory]:
    factory = _GatedFactory()
    return EngineManager(SousConfig(idle_unload_minutes=idle_minutes), engine_factory=factory), factory


def test_status_answers_while_the_model_loads():
    """The worker's get() used to hold the manager lock for the whole load —
    minutes for a 27B — so status() (and everything else on the lock) waited
    it out. A load in progress must be reportable, not a blocking call."""
    mgr, factory = _gated_manager()
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        t0 = time.monotonic()
        s = mgr.status()
        assert time.monotonic() - t0 < 1.0
        assert s["loaded"] is False and s["loading"] is True
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()
    s = mgr.status()
    assert s["loaded"] is True and s["loading"] is False


def test_concurrent_gets_share_one_load():
    mgr, factory = _gated_manager()
    engines: list = []
    threads = [
        threading.Thread(target=lambda: engines.append(mgr.get()), daemon=True) for _ in range(3)
    ]
    for t in threads:
        t.start()
    try:
        assert factory.started.wait(5)
    finally:
        factory.release.set()
    for t in threads:
        t.join(5)
        assert not t.is_alive()
    assert len(factory.created) == 1
    assert all(e is engines[0] for e in engines)


def test_a_failed_load_leaves_the_manager_empty_and_loadable():
    calls: list[str] = []

    def factory(model_id: str):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("weights missing")
        return FakeEngine([])

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    with pytest.raises(RuntimeError):
        mgr.get()
    s = mgr.status()
    assert s["loaded"] is False and s["loading"] is False
    assert mgr.get() is mgr.get() and len(calls) == 2


def test_a_failed_load_releases_the_threads_waiting_on_it():
    """A waiter parks until the load in progress ends; a load that ends by
    raising must wake it too, or the daemon is wedged behind one bad load.
    Each waiter then finds nothing loaded and loads on its own."""
    entered = threading.Event()
    proceed = threading.Event()
    calls: list[str] = []

    def factory(model_id: str):
        calls.append(model_id)
        if len(calls) == 1:
            entered.set()
            assert proceed.wait(10)
            raise RuntimeError("weights missing")
        return FakeEngine([])

    mgr = EngineManager(SousConfig(), engine_factory=factory)
    failed = threading.Event()

    def load_and_fail():
        try:
            mgr.get()
        except RuntimeError:
            failed.set()

    loader = threading.Thread(target=load_and_fail, daemon=True)
    loader.start()
    assert entered.wait(5)
    got: list = []
    waiter = threading.Thread(target=lambda: got.append(mgr.get()), daemon=True)
    waiter.start()
    proceed.set()
    loader.join(5)
    assert failed.is_set()
    waiter.join(5)
    assert not waiter.is_alive(), "a failed load left its waiter parked"
    assert len(calls) == 2 and got[0] is mgr.get()


def test_unload_if_idle_is_refused_at_once_during_a_load():
    """Refused, and refused without waiting: the sweep runs on the worker's
    thread every poll, and parking it behind a load for minutes is the
    stall the lock change removes."""
    mgr, factory = _gated_manager(idle_minutes=0)
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        t0 = time.monotonic()
        assert mgr.unload_if_idle() is False
        assert time.monotonic() - t0 < 1.0
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()


class _SlowUnloadEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__([])
        self.unloading = threading.Event()
        self.release = threading.Event()

    def unload(self) -> None:
        self.unloading.set()
        assert self.release.wait(10)
        super().unload()


def test_get_during_an_unload_waits_for_it_and_loads_fresh():
    """Freeing a 27B takes seconds and get() must never hand out an engine
    that is being torn down, nor load a second copy beside it: it waits, then
    loads. status() meanwhile reports neither loaded nor loading."""
    slow = _SlowUnloadEngine()
    second_started = threading.Event()
    made: list[FakeEngine] = []

    def factory(model_id: str):
        if made:
            second_started.set()
        made.append(slow if not made else FakeEngine([]))
        return made[-1]

    mgr = EngineManager(SousConfig(idle_unload_minutes=0), engine_factory=factory)
    mgr.get()
    time.sleep(0.01)
    sweeper = threading.Thread(target=mgr.unload_if_idle, daemon=True)
    sweeper.start()
    assert slow.unloading.wait(5)
    got: list = []
    getter = threading.Thread(target=lambda: got.append(mgr.get()), daemon=True)
    getter.start()
    # Parked behind the unload, not loading a second copy beside it.
    assert not second_started.wait(0.2)
    assert got == []
    s = mgr.status()
    assert s["loading"] is False and s["loaded"] is False
    slow.release.set()
    sweeper.join(5)
    getter.join(5)
    assert not getter.is_alive()
    assert len(made) == 2 and got[0] is not slow and slow.unloaded is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k "model_loads or share_one_load or failed_load or refused_at_once or during_an_unload" -v`
Expected: four FAIL, two pass. `test_status_answers_while_the_model_loads`: `status()` blocks on the lock the loader holds inside the factory, so it returns only after the factory's 10 s wait gives up (an `AssertionError` in the loader thread, printed by `threading.excepthook`), then fails on `< 1.0`. `test_concurrent_gets_share_one_load` and `test_a_failed_load_releases_the_threads_waiting_on_it` PASS already — today the lock itself serializes and a raising factory releases it; keep both, they pin the properties through the change (the second catches a `notify_all()` dropped from the failure path, which today's code cannot lose). `test_a_failed_load_leaves_the_manager_empty_and_loadable` FAILS with `KeyError: 'loading'`. `test_unload_if_idle_is_refused_at_once_during_a_load`: same 10 s stall, then `< 1.0` fails. `test_get_during_an_unload…`: the sweeper holds the lock through `unload()`, which times out after 10 s and raises before `_engine` is cleared, so `status()` then reports the slow engine still loaded and `KeyError: 'loading'` fails first (`loading` is read before `loaded`).

- [ ] **Step 3: Rewrite `get()`, `unload_if_idle()` and `status()`**

In `src/sous/engine/base.py`, replace `EngineManager.__init__`'s last four lines, `get`, `unload_if_idle` and `status` with:

```python
        self._lock = threading.Lock()
        # A load or an unload is never done under _lock (minutes for a 27B):
        # get() and unload_if_idle() mark which is in progress and wait on
        # this for it to finish, so status() and every other short call on
        # the lock answer meanwhile.
        self._changed = threading.Condition(self._lock)
        self._engine: ManagedEngine | None = None
        self._loading = False
        self._unloading = False
        self._last_used: float | None = None
        self._leases = 0

    def get(self) -> ManagedEngine:
        with self._changed:
            while self._loading or self._unloading:
                self._changed.wait()
            if self._engine is not None:
                self._last_used = time.monotonic()
                return self._engine
            self._loading = True
        loading = time.monotonic()
        try:
            engine = ManagedEngine(self._factory(self._config.model_id))
        except BaseException:
            with self._changed:
                self._loading = False
                self._changed.notify_all()
            raise
        with self._changed:
            self._engine = engine
            self._loading = False
            self._last_used = time.monotonic()
            self._changed.notify_all()
        # The one line that brackets a cold start in the daemon log —
        # before it, only huggingface_hub's own chatter said a load
        # happened, and a turn's `seconds` could not be split.
        _logger.info(f"model_load seconds={time.monotonic() - loading:.1f} model={engine.model_id}")
        return engine
```

(`touch()` and `lease()` are unchanged.)

```python
    def unload_if_idle(self) -> bool:
        with self._changed:
            # _engine is None while a load or an unload is in progress too,
            # so both are refused here without a second check.
            if self._engine is None or self._last_used is None:
                return False
            if self._engine.generation_in_flight() or self._leases:
                # Never free the model weights under an active (possibly
                # abandoned-as-stalled) generation, nor under a caller that is
                # holding this engine across calls that take no _gen_lock.
                return False
            idle = time.monotonic() - self._last_used
            if idle <= self._config.idle_unload_minutes * 60:
                return False
            engine, self._engine = self._engine, None
            self._unloading = True
        try:
            engine.unload()
        finally:
            with self._changed:
                self._unloading = False
                self._changed.notify_all()
        return True

    def status(self) -> dict:
        with self._lock:
            idle = (time.monotonic() - self._last_used) if self._last_used else None
            out = {
                "loaded": self._engine is not None,
                "loading": self._loading,
                "model_id": self._config.model_id,
                "idle_seconds": idle,
            }
            if self._engine is not None:
                # promptcache imports this module, so the import cannot be global.
                from sous.engine.promptcache import without_turn_gauges

                # Counts and byte totals only; never a token id.
                out["prompt_cache"] = without_turn_gauges(self._engine.prompt_cache_stats())
                int8 = self._engine.int8_prefill_status
                if int8 is not None:
                    out["int8_prefill"] = dict(int8)
            return out
```

- [ ] **Step 4: Fix the turn runner's comment**

In `src/sous/gateway/turn.py` lines 158–160, replace

```python
                # From `started`, not from here: entering the lease waits out
                # a load or unload another thread is in the middle of, which
                # costs this turn exactly what loading the model itself would.
```

with

```python
                # From `started`, not from here: get() waits out a load or
                # unload another thread is in the middle of, which costs this
                # turn exactly what loading the model itself would.
```

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_engine_base.py tests/test_gateway_turn.py tests/test_worker.py -v`
Expected: all PASS, including the existing `test_a_lease_holds_off_the_idle_unload`, `test_unload_refused_while_generation_in_flight`, `test_a_running_turn_pins_the_engine_against_the_idle_unload` and `test_get_logs_the_load_once_with_its_duration` (the log line text is unchanged).

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: +6 — `947 passed, 8 skipped, 29 deselected` on this Mac.

Run: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/base.py src/sous/gateway/turn.py tests/test_engine_base.py
git commit -m "refactor(engine): load and unload outside the manager lock

get() ran the model factory under EngineManager._lock and unload_if_idle()
ran engine.unload() under it — minutes for the default model — so
status() and every other short call on that lock stalled for the whole
load. A load-in-progress must be reportable (status() now says
loading=True) and a short call must answer during one. A Condition over
the same lock keeps one load at a time, no load during an unload and no
unload under a generation or lease.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Holders, the PID watch and the preload thread (`base.py`)

**Files:**
- Modify: `src/sous/engine/base.py` (`EngineManager`; a module-level `_holder_alive` next to `release_mlx_thread_state`)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: Task 1's `_changed`, `_loading`; `release_mlx_thread_state` (present — the preload thread calls it after `get()` returns, the same shape as the gateway's turn-pool thread at the end of `TurnRunner.run`, which may also have loaded the model); the test helpers Task 1 left in `tests/test_engine_base.py` (`_GatedFactory`, `_gated_manager`, `_SlowUnloadEngine`).
- Produces: `EngineManager.__init__(config, engine_factory=None, *, holder_alive: Callable[[int, float], bool] | None = None, clock: Callable[[], float] = time.monotonic)`; `EngineManager.hold(pid: int, create_time: float) -> dict` returning `{"loaded": bool, "loading": bool, "holders": int}`; `status()` gains `"holders": int`; `base._holder_alive(pid: int, create_time: float) -> bool`. Log lines on `sous.engine`: `hold pid=N (holders=M)`, `hold released pid=N (holders=M)`, `preloading <model_id>`, `preload failed (<ExceptionType>)`. Task 3's route calls `hold`; Task 4's launcher reads the dict.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py`:

```python
class _Liveness:
    """A stand-in for the psutil check: which (pid, create_time) pairs are
    live processes right now."""

    def __init__(self) -> None:
        self.live: set[tuple[int, float]] = set()

    def __call__(self, pid: int, create_time: float) -> bool:
        return (pid, create_time) in self.live


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _held_manager(idle_minutes: int = 30, factory=None):
    created: list[FakeEngine] = []

    def default_factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    live, clock = _Liveness(), _Clock()
    mgr = EngineManager(
        SousConfig(idle_unload_minutes=idle_minutes),
        engine_factory=factory or default_factory,
        holder_alive=live,
        clock=clock,
    )
    return mgr, created, live, clock


def _join_preload(mgr: EngineManager) -> None:
    """Wait for a preload the test just triggered. The manager forgets the
    thread as its last act, so a missing thread means a finished preload —
    which the status must then agree with."""
    thread = mgr._preload
    if thread is None:
        assert mgr.status()["loading"] is False
        return
    thread.join(5)
    assert not thread.is_alive()


def test_hold_starts_exactly_one_preload_which_releases_its_mlx_state(monkeypatch):
    """The release is monkeypatched, so this pins that it happens on the
    preload thread and once, not what the real call does to a loaded model;
    the gateway's turn-pool thread already releases after a get() that may
    have loaded the weights, and the real-daemon run is what exercises it."""
    from sous.engine import base

    released_in: list[int] = []
    monkeypatch.setattr(
        base, "release_mlx_thread_state", lambda: released_in.append(threading.get_ident())
    )
    mgr, factory = _gated_manager()
    first = mgr.hold(101, 5.0)
    assert first == {"loaded": False, "loading": True, "holders": 1}
    assert factory.started.wait(5)
    preload = mgr._preload
    assert preload is not None and preload.daemon and preload.name == "sous-preload"
    second = mgr.hold(102, 6.0)
    assert second == {"loaded": False, "loading": True, "holders": 2}
    assert mgr._preload is preload  # no second thread while the first loads
    s = mgr.status()
    assert s["loaded"] is False and s["loading"] is True and s["holders"] == 2
    factory.release.set()
    preload.join(5)
    assert not preload.is_alive()
    assert len(factory.created) == 1
    assert released_in == [preload.ident]
    s = mgr.status()
    assert s["loaded"] is True and s["loading"] is False and s["holders"] == 2
    assert mgr.hold(103, 7.0) == {"loaded": True, "loading": False, "holders": 3}


def test_a_hold_during_another_threads_load_starts_no_preload():
    """The worker or a gateway turn may already be loading; a hold then only
    registers itself and reports the load in progress."""
    mgr, factory = _gated_manager()
    loader = threading.Thread(target=mgr.get, daemon=True)
    loader.start()
    try:
        assert factory.started.wait(5)
        assert mgr.hold(51, 1.0) == {"loaded": False, "loading": True, "holders": 1}
        assert mgr._preload is None
        assert mgr.status()["loading"] is True
    finally:
        factory.release.set()
    loader.join(5)
    assert not loader.is_alive()
    assert len(factory.created) == 1  # one load, not two


@pytest.mark.parametrize("preloaded", [False, True])
def test_hold_never_reports_neither_loaded_nor_loading(preloaded):
    """The launcher prints one of two lines from these flags; a hold on an
    unloaded model always starts or joins a load, so no third state exists."""
    mgr, created, live, _ = _held_manager()
    if preloaded:
        mgr.get()
    answer = mgr.hold(81, 1.0)
    _join_preload(mgr)
    assert answer["loaded"] or answer["loading"]


def test_a_failed_preload_is_logged_and_never_raised(caplog):
    import logging

    def factory(model_id: str):
        raise RuntimeError("weights missing")

    mgr, _, live, _ = _held_manager(factory=factory)
    live.live.add((7, 1.0))
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        assert mgr.hold(7, 1.0)["loading"] is True
        _join_preload(mgr)
    messages = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert "hold pid=7 (holders=1)" in messages
    assert f"preloading {mgr._config.model_id}" in messages
    assert "preload failed (RuntimeError)" in messages
    assert "weights missing" not in "".join(messages)
    assert mgr.status()["loading"] is False and mgr.status()["holders"] == 1


def test_the_sweep_keeps_the_model_while_any_holder_lives(caplog):
    import logging

    mgr, created, live, clock = _held_manager(idle_minutes=30)
    live.live.update({(11, 1.0), (12, 2.0)})
    mgr.hold(11, 1.0)
    mgr.hold(12, 2.0)
    _join_preload(mgr)
    clock.now += 3600  # an hour idle: past the threshold, but held
    assert mgr.unload_if_idle() is False
    live.live.discard((11, 1.0))  # one holder exits
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        assert mgr.unload_if_idle() is False
    assert "hold released pid=11 (holders=1)" in [r.getMessage() for r in caplog.records]
    assert mgr.status()["holders"] == 1 and created[0].unloaded is False


def test_a_reused_pid_is_a_different_holder():
    """PID reuse: the same number with a different start time is not the
    process that asked for the hold."""
    mgr, created, live, clock = _held_manager(idle_minutes=0)
    live.live.add((21, 100.0))
    mgr.hold(21, 100.0)
    _join_preload(mgr)
    clock.now += 1
    assert mgr.unload_if_idle() is False  # held
    live.live = {(21, 100.0 + 5.0)}  # a new process wearing the old pid
    assert mgr.unload_if_idle() is False  # pruned: the idle clock restarts on this sweep
    assert mgr.status()["holders"] == 0
    clock.now += 1
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True


def test_the_idle_clock_restarts_when_the_last_holder_leaves():
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    live.live.add((31, 1.0))
    mgr.hold(31, 1.0)
    _join_preload(mgr)
    clock.now += 7200  # two hours held
    live.live.clear()
    assert mgr.unload_if_idle() is False  # released now: idle clock restarts here
    assert mgr.status()["holders"] == 0
    clock.now += 30 * 60  # exactly the threshold
    assert mgr.unload_if_idle() is False
    clock.now += 1
    assert mgr.unload_if_idle() is True
    assert created[0].unloaded is True


def test_a_hold_during_an_unload_reloads_after_it():
    slow = _SlowUnloadEngine()
    made: list[FakeEngine] = []

    def factory(model_id: str):
        made.append(slow if not made else FakeEngine([]))
        return made[-1]

    live, clock = _Liveness(), _Clock()
    mgr = EngineManager(
        SousConfig(idle_unload_minutes=0), engine_factory=factory, holder_alive=live, clock=clock
    )
    mgr.get()
    clock.now += 1
    sweeper = threading.Thread(target=mgr.unload_if_idle, daemon=True)
    sweeper.start()
    assert slow.unloading.wait(5)
    live.live.add((41, 1.0))
    assert mgr.hold(41, 1.0) == {"loaded": False, "loading": True, "holders": 1}
    slow.release.set()
    sweeper.join(5)
    _join_preload(mgr)
    assert len(made) == 2 and mgr.status()["loaded"] is True


def test_holder_alive_is_this_process_under_its_real_start_time_only():
    import os

    import psutil

    from sous.engine.base import _holder_alive

    start = psutil.Process().create_time()
    assert _holder_alive(os.getpid(), start) is True
    assert _holder_alive(os.getpid(), start - 3600.0) is False
    assert _holder_alive(2**30, start) is False  # beyond every platform's pid range


def test_a_holder_we_cannot_read_counts_as_gone(monkeypatch):
    """psutil raises AccessDenied for another user's process; the daemon and
    its holders share a user, so that is never ours and must not pin the
    model. Any other psutil error is the same answer."""
    import psutil

    from sous.engine.base import _holder_alive

    def denied(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    assert _holder_alive(1234, 1.0) is False


def test_holders_are_pruned_even_with_nothing_loaded():
    """A load that failed leaves no engine; the registry must still forget a
    holder whose process is gone, or /sous/status reports sessions that
    ended hours ago."""

    def factory(model_id: str):
        raise RuntimeError("weights missing")

    mgr, _, live, _ = _held_manager(factory=factory)
    live.live.add((71, 1.0))
    mgr.hold(71, 1.0)
    _join_preload(mgr)
    assert mgr.status()["holders"] == 1
    live.live.clear()
    assert mgr.unload_if_idle() is False  # nothing loaded, but the sweep still prunes
    assert mgr.status()["holders"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k "hold or holder or sweep or reused or idle_clock or preload" -v`
Expected: every new test FAILS — `TypeError: EngineManager.__init__() got an unexpected keyword argument 'holder_alive'`, `AttributeError: 'EngineManager' object has no attribute 'hold'`, `ImportError: cannot import name '_holder_alive'`.

- [ ] **Step 3: Implement holders, the prune and the preload**

In `src/sous/engine/base.py`, add after `release_mlx_thread_state`:

```python
def _holder_alive(pid: int, create_time: float) -> bool:
    """Whether the process that registered a hold is still the one wearing
    `pid`: alive, and started within a second of the time it reported. A
    reused pid fails the second test; a zombie, a vanished process or one
    this user cannot read (never ours — the daemon and its holders share a
    user) all count as gone rather than pin the model forever."""
    import psutil

    try:
        return abs(psutil.Process(pid).create_time() - create_time) <= 1.0
    except psutil.Error:
        return False
```

Change `EngineManager.__init__`'s signature and add the fields:

```python
    def __init__(
        self,
        config: SousConfig,
        engine_factory: Callable[[str], Engine] | None = None,
        *,
        holder_alive: Callable[[int, float], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
```

and, after `self._leases = 0`:

```python
        # Processes that pin the model: pid -> the start time it reported.
        # `sous claude` registers itself and then execs Claude Code, which
        # keeps the pid, so the holder IS that session; the sweep drops a
        # holder whose process is gone.
        self._holders: dict[int, float] = {}
        self._holder_alive = holder_alive or _holder_alive
        self._preload: threading.Thread | None = None
        self._clock = clock
```

Replace every `time.monotonic()` inside `EngineManager` (in `get`, `touch`, `unload_if_idle`, `status` — not the `loading = …` stopwatch in `get`, which measures the load itself) with `self._clock()`.

Add the methods after `lease`:

```python
    def hold(self, pid: int, create_time: float) -> dict:
        """Pin the model for as long as process `pid` (started at
        `create_time`) lives, loading it now if nothing is loaded and no load
        is under way. Never waits for the load and never raises because of
        it: a hold is an optimization, and the first turn's own get() reports
        a real failure."""
        with self._lock:
            self._holders[pid] = create_time
            holders = len(self._holders)
            loaded = self._engine is not None
            preloading = not loaded and not self._loading and self._preload is None
            if preloading:
                self._preload = threading.Thread(
                    target=self._run_preload, name="sous-preload", daemon=True
                )
                self._preload.start()
            loading = self._loading or self._preload is not None
        _logger.info(f"hold pid={pid} (holders={holders})")
        if preloading:
            _logger.info(f"preloading {self._config.model_id}")
        return {"loaded": loaded, "loading": loading, "holders": holders}

    def _run_preload(self) -> None:
        try:
            self.get()
        except Exception as exc:  # noqa: BLE001 — the load's own error surfaces on the first turn
            # The type only: a loader's message can name a path.
            _logger.warning(f"preload failed ({type(exc).__name__})")
        finally:
            # This thread loaded the model, so it touched mlx: the weights are
            # arrays every later thread can use, the streams are this one's
            # own (the gateway's turn thread releases the same way after a
            # get() that loaded). Released before the registry forgets the
            # thread, so `_preload is None` means it is done with everything.
            release_mlx_thread_state()
            with self._lock:
                self._preload = None

    def _prune_holders(self) -> bool:
        """Drop every holder whose process is gone; True when that emptied
        the registry. Lock held by the caller — the psutil check is
        sub-millisecond, once per poll, and the registry it reads must not
        change under it."""
        gone = [pid for pid, started in self._holders.items() if not self._holder_alive(pid, started)]
        for pid in gone:
            del self._holders[pid]
            _logger.info(f"hold released pid={pid} (holders={len(self._holders)})")
        return bool(gone) and not self._holders
```

Replace `unload_if_idle` (Task 1's version) with — the prune runs first, before every early return, so a holder is forgotten whether or not anything is loaded and whether or not a generation is running; `touch()` is not used for the clock restart because it takes the lock the sweep already holds:

```python
    def unload_if_idle(self) -> bool:
        with self._changed:
            if self._prune_holders() and self._last_used is not None:
                # The last session just left: its idle time starts now, so a
                # relaunch inside idle_unload_minutes never pays a reload.
                self._last_used = self._clock()
            # _engine is None while a load or an unload is in progress too,
            # so both are refused here without a second check.
            if self._engine is None or self._last_used is None:
                return False
            if self._engine.generation_in_flight() or self._leases:
                # Never free the model weights under an active (possibly
                # abandoned-as-stalled) generation, nor under a caller that is
                # holding this engine across calls that take no _gen_lock.
                return False
            if self._holders:
                return False
            idle = self._clock() - self._last_used
            if idle <= self._config.idle_unload_minutes * 60:
                return False
            engine, self._engine = self._engine, None
            self._unloading = True
        try:
            engine.unload()
        finally:
            with self._changed:
                self._unloading = False
                self._changed.notify_all()
        return True
```

In `status()`, add `"holders": len(self._holders),` after `"loading": …,` and make the `loading` value `self._loading or self._preload is not None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -v`
Expected: all PASS. (`test_status_when_never_loaded` still passes: it asserts `loaded` and `model_id` only.)

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: +12 — `959 passed, 8 skipped, 29 deselected` on this Mac.

Run: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .`
Expected: clean. If `ty` objects to `self._holder_alive = holder_alive or _holder_alive` as an instance attribute holding a function, annotate the field: `self._holder_alive: Callable[[int, float], bool] = holder_alive or _holder_alive`.

- [ ] **Step 5: Commit**

```bash
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): hold the model for a live process and preload on request

A sous claude session's subagents run minutes apart; with the daemon idle
between them the weights were unloaded and the next turn paid the load
and a from-zero prefill. hold(pid, create_time) pins the model while that
process lives — the idle sweep prunes holders whose pid is gone or reused
(start time differs) with psutil — and starts one detached preload thread
when nothing is loaded, which releases its mlx thread state on exit. The
idle clock restarts when the last holder leaves, so a relaunch inside
idle_unload_minutes never reloads.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: The daemon's `/sous/` routes (`loopback.py`, `monitor.py`, `server.py`)

**Files:**
- Create: `src/sous/loopback.py`, `src/sous/monitor.py`
- Modify: `src/sous/gateway/routes.py` (lines 85–96 and 297–319: the allow-lists and `_check_loopback` move out; the three call sites become `check_loopback`; line 22's `urlsplit` import goes), `src/sous/server.py` (`create_server`, lines 406–407; the import block, after line 30's `from sous.logs import …`)
- Test: create `tests/test_monitor.py`, `tests/test_loopback.py`; `tests/test_gateway_routes.py` (`test_routes_are_absent_when_the_gateway_is_disabled`, lines 158–162, and `test_host_header_must_be_loopback`, lines 196–208, which pins `routes._ALLOWED_HOSTS` by name)

**Interfaces:**
- Consumes: Task 2's `EngineManager.hold`; `SousService.server_status` (present, `server.py:224`); `RequestError` (`gateway/convert.py:61`).
- Produces: `loopback.ALLOWED_HOSTS`, `loopback.ALLOWED_ORIGIN_HOSTS`, `loopback.check_loopback(request: Request) -> None` (raises `RequestError`); `monitor.HOLD_BODY_LIMIT = 1024`; `monitor.mount_monitor(mcp: MCPServer, engines: EngineManager, status: Callable[[], dict]) -> None`. Routes: `GET /sous/status` → 200 with `server_status()`'s dict; `POST /sous/hold` with `{"pid": int, "create_time": number}` → 200 with `hold()`'s dict; any other `/sous/…` → 404 `not_found_error`. Task 4's launcher speaks to these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loopback.py`:

```python
"""The Host/Origin guard shared by the gateway's routes and the daemon's
/sous/ routes: what it lets through and what it refuses."""

import pytest
from starlette.requests import Request

from sous.gateway.convert import RequestError
from sous.loopback import check_loopback


def _request(host: str | None, origin: str | None = None) -> Request:
    headers = []
    if host is not None:
        headers.append((b"host", host.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "query_string": b""})


@pytest.mark.parametrize("host", ["127.0.0.1:8383", "localhost", "[::1]:8383", "LOCALHOST:1"])
def test_loopback_hosts_pass(host):
    check_loopback(_request(host))


@pytest.mark.parametrize("host", ["sous.example:8383", "10.0.0.5", "", None])
def test_other_hosts_are_refused(host):
    with pytest.raises(RequestError) as exc:
        check_loopback(_request(host))
    assert exc.value.status == 403 and exc.value.error_type == "permission_error"


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:5173", "http://localhost", "http://[::1]:9", None]
)
def test_loopback_and_absent_origins_pass(origin):
    check_loopback(_request("127.0.0.1", origin))


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://[::1", ""])
def test_foreign_and_malformed_origins_are_refused(origin):
    with pytest.raises(RequestError) as exc:
        check_loopback(_request("127.0.0.1", origin))
    assert exc.value.status == 403


def test_the_allow_lists_are_exactly_the_loopback_names():
    from sous.loopback import ALLOWED_HOSTS, ALLOWED_ORIGIN_HOSTS

    assert set(ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}
    assert set(ALLOWED_ORIGIN_HOSTS) == {"127.0.0.1", "localhost", "::1"}
```

In `tests/test_gateway_routes.py`, `test_host_header_must_be_loopback` (lines 196–208) asserts `set(routes._ALLOWED_HOSTS) == {…}` after `import sous.gateway.routes as routes`; that pin has just moved to `tests/test_loopback.py`. Delete the import line and the final `assert set(routes._ALLOWED_HOSTS) …` line; the behavioural assertions stay:

```python
def test_host_header_must_be_loopback(tmp_path: Path):
    """Custom routes skip the /mcp transport's Host check; a page whose hostname
    re-resolves to 127.0.0.1 must not get to drive the local model."""
    app = _app(tmp_path, FakeEngine([]))
    for host in ("evil.example:8383", "evil.example"):
        r = _post(app, _body(), headers={"host": host})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 403
    for host in ("127.0.0.1:8383", "localhost", "[::1]:8383", "[::1]", "LOCALHOST:8383"):
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 200
```

Create `tests/test_monitor.py`:

```python
"""The daemon's own loopback routes under /sous/: reachable with the gateway
off, never forwarded with it on, guarded like the gateway's routes."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import psutil
import pytest
from starlette.requests import Request

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError
from sous.monitor import HOLD_BODY_LIMIT, _hold_body
from sous.server import create_server
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine
from tests.fake_upstream import FakeUpstream


def _app(tmp_path: Path, *, gateway_enabled: bool = False, upstream=None, alive=None):
    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=gateway_enabled
    )
    engines = EngineManager(
        cfg, engine_factory=lambda mid: FakeEngine([]), holder_alive=alive or (lambda pid, ct: True)
    )
    store = TaskStore(tmp_path / "tasks.db")
    upstream = upstream or FakeUpstream().upstream()
    return create_server(store, engines, cfg, upstream=upstream).streamable_http_app(), engines


def _request(app, method: str, path: str, body=None, headers=None) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8383"
        ) as client:
            # A dict is JSON-encoded; bytes, str, None and an async iterator
            # (a chunked body) go to httpx as they are.
            content = json.dumps(body) if isinstance(body, dict) else body
            return await client.request(
                method, path, content=content, headers={"content-type": "application/json", **(headers or {})}
            )

    return asyncio.run(go())


def _hold_body(**overrides) -> dict:
    return {"pid": os.getpid(), "create_time": psutil.Process().create_time(), **overrides}


def _await_loaded(app, timeout: float = 5.0) -> dict:
    """The preload runs on its own thread; poll the route the launcher would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        model = _request(app, "GET", "/sous/status").json()["model"]
        if model["loaded"] and not model["loading"]:
            return model
        time.sleep(0.01)
    raise AssertionError("the preload never finished")


def test_status_is_the_server_status_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "GET", "/sous/status")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    doc = r.json()
    assert doc["model"]["loaded"] is False
    assert doc["model"]["loading"] is False
    assert doc["model"]["holders"] == 0
    assert doc["config"]["gateway"] == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }
    assert set(doc) == {"model", "memory_gb", "queue", "config"}


def test_hold_registers_the_caller_and_preloads(tmp_path: Path):
    app, engines = _app(tmp_path)
    r = _request(app, "POST", "/sous/hold", _hold_body())
    assert r.status_code == 200
    assert r.json() == {"loaded": False, "loading": True, "holders": 1}
    model = _await_loaded(app)
    assert model["holders"] == 1
    s = engines.status()
    assert s["loaded"] is True and s["loading"] is False and s["holders"] == 1
    r = _request(app, "POST", "/sous/hold", _hold_body(pid=os.getpid() + 1))
    assert r.json() == {"loaded": True, "loading": False, "holders": 2}


def test_hold_refuses_every_other_shape_with_an_anthropic_shaped_400(tmp_path: Path):
    app, engines = _app(tmp_path)
    bad = [
        b"not json",
        b"[]",
        json.dumps({"pid": os.getpid()}),
        json.dumps(_hold_body(extra=1)),
        json.dumps(_hold_body(pid=True)),
        json.dumps(_hold_body(pid=0)),
        json.dumps(_hold_body(pid="12")),
        json.dumps(_hold_body(create_time="now")),
        json.dumps(_hold_body(create_time=1e999)),
        b'{"pid": 1, "create_time": NaN}',
        json.dumps(_hold_body(create_time=1.0)).encode() + b" " * HOLD_BODY_LIMIT,
    ]
    for body in bad:
        r = _request(app, "POST", "/sous/hold", body)
        assert r.status_code == 400, body[:40]
        answer = r.json()
        assert answer["type"] == "error" and answer["error"]["type"] == "invalid_request_error"
        assert isinstance(answer["error"]["message"], str)

    async def chunks():
        # httpx sends an async byte iterator as several ASGI messages, so the
        # cap has to be enforced as the body accumulates, not on one read.
        for _ in range(4):
            yield b"x" * 400

    r = _request(app, "POST", "/sous/hold", chunks())
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    s = engines.status()
    assert s["holders"] == 0 and s["loading"] is False and s["loaded"] is False


def test_a_client_that_drops_mid_body_is_a_400_not_a_traceback():
    """httpx's in-process transport cannot hang up mid-body, so the handler's
    body reader is driven directly with a receive that disconnects."""

    async def receive():
        return {"type": "http.disconnect"}

    request = Request(
        {"type": "http", "method": "POST", "path": "/sous/hold", "headers": [], "query_string": b""},
        receive,
    )
    with pytest.raises(RequestError) as exc:
        asyncio.run(_hold_body(request))
    assert exc.value.status == 400 and exc.value.error_type == "invalid_request_error"


def test_routes_are_loopback_guarded(tmp_path: Path):
    app, engines = _app(tmp_path)
    for method, path, body in (("GET", "/sous/status", None), ("POST", "/sous/hold", _hold_body())):
        r = _request(app, method, path, body, headers={"host": "sous.example:8383"})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        r = _request(app, method, path, body, headers={"origin": "https://evil.example"})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
    assert engines.status()["holders"] == 0


def test_anything_else_under_sous_is_404_and_never_forwarded(tmp_path: Path):
    fake = FakeUpstream()
    app, _ = _app(tmp_path, gateway_enabled=True, upstream=fake.upstream())
    # `/sous` without the slash would full-match the gateway's catch-all
    # and be forwarded with the client's credentials; a method the real
    # routes do not take falls through to the 404 too.
    for method, path in (
        ("GET", "/sous"),
        ("GET", "/sous/"),
        ("POST", "/sous/events"),
        ("DELETE", "/sous/hold/1"),
        ("POST", "/sous/status"),
        ("GET", "/sous/hold"),
    ):
        r = _request(app, method, path)
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error", path
    assert _request(app, "GET", "/sous/status").status_code == 200
    assert _request(app, "HEAD", "/sous/status").status_code == 200
    assert _request(app, "POST", "/sous/hold", _hold_body()).status_code == 200
    assert fake.requests == []
    # The gateway's own catch-all still forwards what is not ours.
    assert _request(app, "GET", "/api/hello").status_code == 200
    assert [s["path"] for s in fake.requests] == ["/api/hello"]


def test_routes_are_up_with_the_gateway_off(tmp_path: Path):
    app, _ = _app(tmp_path, gateway_enabled=False)
    assert _request(app, "GET", "/sous/status").status_code == 200
    assert _request(app, "POST", "/sous/hold", _hold_body()).status_code == 200
    assert _request(app, "GET", "/sous/nope").status_code == 404
    assert _request(app, "GET", "/sous").status_code == 404
    assert _request(app, "GET", "/api/hello").status_code == 404  # no gateway, no forwarding
```

In `tests/test_gateway_routes.py`, extend `test_routes_are_absent_when_the_gateway_is_disabled` with one line so it keeps meaning "the *gateway's* routes":

```python
def test_routes_are_absent_when_the_gateway_is_disabled(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]), gateway_enabled=False)
    assert _request(app, "HEAD", "/api/hello").status_code == 404
    assert _post(app, _body()).status_code == 404
    assert _request(app, "GET", "/sous/status").status_code == 200  # the daemon's own routes stay
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_loopback.py tests/test_monitor.py -v`
Expected: collection errors — `ModuleNotFoundError: No module named 'sous.loopback'` / `'sous.monitor'`.

Run: `uv run pytest tests/test_gateway_routes.py::test_routes_are_absent_when_the_gateway_is_disabled tests/test_gateway_routes.py::test_host_header_must_be_loopback -v`
Expected: the first FAILS — `/sous/status` is 404 (nothing mounts it); the second still PASSES (its pin moved, its behaviour is unchanged).

- [ ] **Step 3: Create `src/sous/loopback.py`**

```python
"""The Host/Origin check every custom route on the daemon's loopback bind
applies — the pair the MCP transport checks on /mcp, which custom routes get
none of. Shared by the gateway's routes and the daemon's own /sous/ routes."""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.requests import Request

from sous.gateway.convert import RequestError

# Without the Host check a web page whose hostname re-resolves to 127.0.0.1
# could drive the local model. Same allow-list as the SDK's.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
# The SDK checks Origin as well, and Host alone does not cover what it covers:
# a cross-origin fetch with Content-Type: text/plain is a CORS simple request,
# so it skips the preflight and arrives with a perfectly legitimate loopback
# Host. The page cannot read the reply, but the turn it starts holds the
# gateway lock, the engine lock and a prompt-cache slot for a whole generation
# timeout — a drive-by DoS of the daemon. urlsplit unwraps the IPv6 brackets a
# netloc carries, so these are bare addresses.
ALLOWED_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")


def check_loopback(request: Request) -> None:
    """Refuse (403, permission_error) a Host that is not loopback or an
    Origin that is not — a browser's, never Claude Code's or curl's."""
    host = request.headers.get("host", "").lower()
    # Strip the port; an IPv6 literal keeps its brackets.
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    else:
        host = host.split(":", 1)[0]
    if host not in ALLOWED_HOSTS:
        raise RequestError(403, "permission_error", "loopback hosts only")
    # Only a browser sends Origin, and only a browser can be someone else's
    # page: absent (Claude Code, httpx, curl) passes untouched. `null` — a
    # sandboxed frame or a file:// page — has no hostname and is refused.
    origin = request.headers.get("origin")
    if origin is None:
        return
    try:
        # An unbalanced IPv6 bracket makes urlsplit raise instead of
        # returning; a malformed Origin is refused like any foreign one.
        hostname = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        hostname = ""
    if hostname not in ALLOWED_ORIGIN_HOSTS:
        raise RequestError(403, "permission_error", "loopback origins only")
```

In `src/sous/gateway/routes.py`: delete `_ALLOWED_HOSTS`, `_ALLOWED_ORIGIN_HOSTS` and their comments (lines 85–96, from the `# Custom routes get none of the Host validation…` comment through the `_ALLOWED_ORIGIN_HOSTS = …` line — lines 98–101, `_LOG_TYPES` and `_LOG_TYPE_CHARS`, stay), delete `_check_loopback` (lines 297–319, the `def` through its last `raise`), delete line 22 (`from urllib.parse import urlsplit` — nothing else in the file uses it; `ruff` F401 confirms), add `from sous.loopback import check_loopback` to the imports, and replace the three `_check_loopback(request)` calls (in `passthrough`, `messages` and `count_tokens`) with `check_loopback(request)`.

- [ ] **Step 4: Create `src/sous/monitor.py`**

```python
"""The daemon's own loopback routes under /sous/: what `sous claude` speaks
to the daemon. Mounted before the gateway's routes and whatever the gateway
flag says, so a path under /sous — the bare /sous included — is answered
here or 404s here, and is never forwarded to the upstream."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from anyio.to_thread import run_sync
from mcp.server import MCPServer
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from sous.engine.base import EngineManager
from sous.gateway.convert import RequestError
from sous.loopback import check_loopback

# A hold body is two numbers; anything larger is not one.
HOLD_BODY_LIMIT = 1024
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def _refused(e: RequestError) -> Response:
    return JSONResponse(e.body(), status_code=e.status)


def _invalid(message: str) -> RequestError:
    return RequestError(400, "invalid_request_error", message)


async def _hold_body(request: Request) -> tuple[int, float]:
    """`{"pid": int, "create_time": number}` and nothing else. Read in chunks
    under the cap rather than whole: the route is loopback-only, but a body
    that is not a hold is refused before it is held in memory."""
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > HOLD_BODY_LIMIT:
                raise _invalid(f"hold body exceeds {HOLD_BODY_LIMIT} bytes")
            chunks.append(chunk)
    except ClientDisconnect:
        # Nobody will read this response; it exists so a disconnect is not a
        # 500 with a traceback in the daemon log.
        raise _invalid("client disconnected mid-body") from None
    try:
        body = json.loads(b"".join(chunks))
    except ValueError:
        raise _invalid("hold body is not JSON") from None
    if not isinstance(body, dict) or set(body) != {"pid", "create_time"}:
        raise _invalid("hold body must be exactly {pid, create_time}")
    pid, create_time = body["pid"], body["create_time"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise _invalid("pid must be a positive integer")
    if (
        isinstance(create_time, bool)
        or not isinstance(create_time, int | float)
        or not math.isfinite(create_time)
    ):
        raise _invalid("create_time must be a finite number")
    return pid, float(create_time)


def mount_monitor(mcp: MCPServer, engines: EngineManager, status: Callable[[], dict]) -> None:
    """Register GET /sous/status, POST /sous/hold and a 404 for every other
    /sous/ path. `status` is the daemon's server_status document. Both real
    handlers hand their work to a thread: status() reads the task store and
    hold() takes the engine manager's lock, and neither belongs on the event
    loop."""

    async def sous_status(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return JSONResponse(await run_sync(status))

    async def sous_hold(request: Request) -> Response:
        try:
            check_loopback(request)
            pid, create_time = await _hold_body(request)
        except RequestError as e:
            return _refused(e)
        return JSONResponse(await run_sync(engines.hold, pid, create_time))

    async def sous_unknown(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return _refused(RequestError(404, "not_found_error", "no such route"))

    mcp.custom_route("/sous/status", methods=["GET"])(sous_status)
    mcp.custom_route("/sous/hold", methods=["POST"])(sous_hold)
    # Registered after the two real routes and before the gateway mounts its
    # catch-all (the SDK keeps registration order), so no path under /sous
    # can reach the upstream with the gateway on. The bare /sous needs its
    # own entry: `/sous/{path:path}` does not match it, and the gateway's
    # `/{path:path}` would.
    mcp.custom_route("/sous", methods=_METHODS)(sous_unknown)
    mcp.custom_route("/sous/{path:path}", methods=_METHODS)(sous_unknown)
```

- [ ] **Step 5: Mount it in `create_server`**

In `src/sous/server.py`, add `from sous.monitor import mount_monitor` to the imports (a new line after line 30, `from sous.logs import …` — alphabetical), and change the end of `create_server` to:

```python
    # The daemon's own routes go on before the gateway's: the gateway ends
    # with a catch-all that forwards upstream, and a /sous/ path must never
    # get there. Mounted whatever the gateway flag says — `sous claude`
    # asks /sous/status whether the gateway is on.
    mount_monitor(mcp, engines, svc.server_status)
    if config.gateway_enabled:
        mounted_gateway.append(mount_gateway(mcp, engines, config, upstream=upstream))

    return mcp
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_loopback.py tests/test_monitor.py tests/test_gateway_routes.py tests/test_gateway_http.py tests/test_server.py -v`
Expected: all PASS, including the gateway's `test_host_header_must_be_loopback` (as edited), `test_a_foreign_origin_is_rejected_before_the_model_is_touched` (it asserts `"origins" in` the message, which "loopback origins only" satisfies), `test_a_malformed_origin_is_rejected_not_raised`, `test_loopback_and_absent_origins_pass` and `test_create_server_registers_six_tools` (routes are not tools).

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: +24 (17 loopback cases, 7 monitor tests) — `983 passed, 8 skipped, 29 deselected` on this Mac.

Run: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/sous/loopback.py src/sous/monitor.py src/sous/server.py src/sous/gateway/routes.py tests/test_loopback.py tests/test_monitor.py tests/test_gateway_routes.py
git commit -m "feat(server): daemon-level /sous/status and /sous/hold routes

The launcher's preflight spoke MCP to read server_status, and there was
no way for a process to pin the model. /sous/status serves the same
document over plain HTTP; /sous/hold registers the caller's pid and start
time with the engine manager and starts a preload. Both are mounted
before the gateway's routes and whether or not the gateway is on, behind
the gateway's own loopback Host/Origin guard, now in sous/loopback.py so
neither module imports the other; every other /sous/ path is a 404 here
rather than a request to the upstream.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `sous claude` speaks HTTP, holds the model, and stops speaking MCP (`cli.py`)

**Files:**
- Modify: `src/sous/cli.py` (`_STATUS_TIMEOUT_SECONDS` comment at 44–46; `_failure_names`, `_server_status`, `_daemon_status` at 100–166; `_cmd_claude` at 168–253)
- Test: `tests/test_cli.py` (`_claude_setup` at 848–883; `test_claude_refuses_a_pre_routing_daemon` at 1045–1057; the `@pytest.mark.slow` `test_daemon_status_reads_the_real_daemons_gateway_config` at ~1095, whose docstring names the MCP call this task deletes; new tests)

**Interfaces:**
- Consumes: Task 3's routes and response shapes; `claude_env`, `claude_argv`, `_port_open` (present).
- Produces: `cli._daemon_status(port: int) -> dict | None` (same name and contract; now HTTP; `SystemExit(1)` on 404); `cli._hold(port: int) -> dict | None`; `cli._PREDATES_MESSAGE` (the 404 text). `_server_status` and `_failure_names` are gone.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, add to `_claude_setup` (after the `_daemon_status` monkeypatch) so every existing launcher test keeps its fake daemon:

```python
    holds: list[int] = []
    monkeypatch.setattr(
        cli,
        "_hold",
        lambda port: (holds.append(port), {"loaded": True, "loading": False, "holders": 1})[1],
    )
```

and return `holds` alongside: change the last line to `return cfg, calls, holds` — then update every existing `cfg, calls = _claude_setup(…)` / `_, calls = _claude_setup(…)` call in the file to unpack three values (`cfg, calls, _ = …` / `_, calls, _ = …`). Delete `test_claude_refuses_a_pre_routing_daemon` (its premise — a daemon that answers the status document without `upstream_url` — cannot occur once the document comes from `/sous/status`, which every routing-era daemon lacks and every pre-routing daemon 404s).

Add a real-socket fake daemon and the new tests at the end of the `sous claude` section:

```python
class _FakeSousHTTP:
    """A daemon that speaks only the two /sous/ routes, over a real socket:
    the launcher's HTTP client is under test, so no transport is faked."""

    def __init__(
        self,
        status: dict | None,
        status_code: int = 200,
        hold_code: int = 200,
        raw: bytes | None = None,
    ):
        """`raw`, when given, is sent verbatim (with the status code) for both
        routes: what a wrong daemon on the port might answer."""
        import http.server

        self.holds: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # quiet
                pass

            def _send(self, code: int, payload: bytes) -> None:
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _reply(self, code: int, body: dict) -> None:
                self._send(code, raw if raw is not None else json.dumps(body).encode())

            def do_GET(self) -> None:
                if self.path == "/sous/status" and status_code == 200:
                    self._reply(200, status or {})
                else:
                    error = {"type": "error", "error": {"type": "not_found_error", "message": ""}}
                    self._reply(status_code, error)

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                outer.holds.append(json.loads(self.rfile.read(length)))
                self._reply(hold_code, {"loaded": False, "loading": True, "holders": 1})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_daemon_status_reads_sous_status_over_http():
    from sous import cli

    fake = _FakeSousHTTP(_status())
    try:
        assert cli._daemon_status(fake.port) == _status()
    finally:
        fake.close()


def test_daemon_status_treats_a_404_as_a_daemon_that_predates_this_cli(capsys):
    """An older daemon has no /sous/status: with the gateway on it forwards
    the path upstream and relays the API's 404, with it off Starlette 404s.
    Either way the answer is restart, not a fallback to MCP."""
    from sous import cli

    fake = _FakeSousHTTP(None, status_code=404)
    try:
        with pytest.raises(SystemExit) as exc:
            cli._daemon_status(fake.port)
    finally:
        fake.close()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "predates this CLI" in err and "sous stop" in err and "sous serve" in err


def test_daemon_status_reports_a_listener_that_does_not_answer(capsys):
    from sous import cli

    fake = _FakeSousHTTP(None, status_code=500)
    try:
        assert cli._daemon_status(fake.port) is None
    finally:
        fake.close()
    err = capsys.readouterr().err
    assert "did not answer /sous/status" in err and "500" in err


def test_daemon_status_is_none_when_nothing_listens():
    from sous import cli

    assert cli._daemon_status(1) is None  # port 1: never up


def test_a_status_that_is_not_json_is_a_daemon_that_does_not_answer(capsys):
    """Something else on the port — a 200 that is not the document."""
    from sous import cli

    fake = _FakeSousHTTP(None, raw=b"<html>nope</html>")
    try:
        assert cli._daemon_status(fake.port) is None
    finally:
        fake.close()
    assert "did not answer /sous/status" in capsys.readouterr().err


def test_hold_posts_this_process_and_returns_the_daemons_answer():
    import psutil

    from sous import cli

    fake = _FakeSousHTTP(_status())
    try:
        assert cli._hold(fake.port) == {"loaded": False, "loading": True, "holders": 1}
    finally:
        fake.close()
    [body] = fake.holds
    assert body["pid"] == os.getpid()
    assert abs(body["create_time"] - psutil.Process().create_time()) < 1.0
    assert set(body) == {"pid", "create_time"}


def test_a_refused_hold_warns_and_returns_none(capsys):
    from sous import cli

    fake = _FakeSousHTTP(_status(), hold_code=500)
    try:
        assert cli._hold(fake.port) is None
    finally:
        fake.close()
    err = capsys.readouterr().err
    assert "warning" in err and "hold" in err and "launching anyway" in err


def test_a_hold_answered_with_something_that_is_not_an_object_warns(capsys):
    from sous import cli

    fake = _FakeSousHTTP(_status(), raw=b"[1, 2]")
    try:
        assert cli._hold(fake.port) is None
    finally:
        fake.close()
    assert "unexpected answer" in capsys.readouterr().err


def test_claude_holds_the_model_after_the_checks_and_says_so(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude"])
    assert holds == [8383] and len(calls) == 1
    err = capsys.readouterr().err
    assert "sous claude: mlx-community/Qwen3.8-27B-4bit already loaded; held while this session runs" in err


def test_claude_reports_a_preload(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_hold", lambda port: {"loaded": False, "loading": True, "holders": 1})
    cli.main(["claude"])
    assert len(calls) == 1
    assert "sous claude: preloading mlx-community/Qwen3.8-27B-4bit; held while this session runs" in (
        capsys.readouterr().err
    )


def test_claude_launches_even_when_the_hold_is_refused(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, _ = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_hold", lambda port: None)
    cli.main(["claude", "-p", "hi"])
    [(exe, argv, env)] = calls
    assert argv == ["/opt/bin/claude", "-p", "hi", "--disallowedTools", "LSP"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8383"
    assert "held while this session runs" not in capsys.readouterr().err


def test_claude_does_not_hold_when_a_check_fails(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch, status=_status(enabled=False))
    with pytest.raises(SystemExit):
        cli.main(["claude"])
    assert holds == [] and calls == []


@pytest.mark.parametrize("flag", ["--help", "-h", "--version", "-v"])
def test_claude_does_not_hold_for_an_invocation_that_exits_at_once(tmp_path, capsys, monkeypatch, flag):
    """A hold for `claude --version` would start a multi-minute load nobody
    waits for and, once the process is gone, restart the idle clock under
    weights nobody is using. The preflight still runs; only the hold is skipped."""
    from sous import cli

    _, calls, holds = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", flag])
    assert holds == [] and len(calls) == 1
    assert calls[0][1] == ["/opt/bin/claude", flag, "--disallowedTools", "LSP"]
    assert "held while this session runs" not in capsys.readouterr().err
```

Restate the docstring of the existing `@pytest.mark.slow` `test_daemon_status_reads_the_real_daemons_gateway_config` (it keeps passing — it drives a real `create_server` app under uvicorn, which now mounts the monitor — and it is the one test that reaches `/sous/status` through a real socket rather than the in-process transport):

```python
    """The launcher's one source of truth, against the real server: a plain
    GET /sous/status on the daemon's port, and None when nothing is
    listening."""
```

`_status()` must carry the model id the launcher prints: add `"model": {"model_id": "mlx-community/Qwen3.8-27B-4bit", "loaded": True, "loading": False, "holders": 0},` as a top-level key of the dict `_status` returns (beside `"config"`), and restate its docstring: "the shape of `GET /sous/status` — `SousService.server_status()` — of which the launcher reads `config.gateway` for its checks and `model.model_id` for its one line". Add `import json` to the file's imports (`os`, `threading` and `pytest` are already there).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "claude or daemon_status or hold" -v`
Expected: the new `_daemon_status` tests FAIL (the MCP client gets a non-MCP answer: `did not answer server_status`), `_hold` tests FAIL with `AttributeError: module 'sous.cli' has no attribute '_hold'`, and `_claude_setup` itself fails on the same attribute for every launcher test.

- [ ] **Step 3: Rewrite the preflight**

In `src/sous/cli.py`, replace the comment at lines 44–46 and the block from `def _failure_names` through the end of `_daemon_status` (lines 100–166) with:

```python
# Nothing about /sous/status or /sous/hold is slow: one reads the queue
# counts and the config snapshot, the other registers a pid; neither waits
# for the model.
_STATUS_TIMEOUT_SECONDS = 15.0
_PREDATES_MESSAGE = (
    "sous claude: the running daemon predates this CLI; restart it (sous stop; sous serve)"
)


def _daemon_status(port: int) -> dict | None:
    """What the daemon on `port` reports about itself (GET /sous/status), or
    None if nothing there answers. Exits 1 on a 404: that is a daemon from
    before the route existed, and the fix is a restart — the CLI and the
    daemon ship in one package, so there is no older dialect to fall back to.

    The config FILE is not the truth: the daemon loads it once at startup and
    holds that snapshot for its whole life, so an edit since then — a changed
    `local_models`, a wider `max_context_tokens`, `enabled` flipped — is a
    setting the running gateway does not have. /sous/status is where the
    daemon reports the gateway config it is actually serving.
    """
    import httpx

    if not _port_open(port):
        return None
    try:
        reply = httpx.get(f"http://127.0.0.1:{port}/sous/status", timeout=_STATUS_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        # The type only: an httpx message can carry the URL it was building.
        print(
            f"sous claude: 127.0.0.1:{port} did not answer /sous/status ({type(exc).__name__})",
            file=sys.stderr,
        )
        return None
    if reply.status_code == 404:
        print(_PREDATES_MESSAGE, file=sys.stderr)
        raise SystemExit(1)
    body: object = None
    if reply.status_code == 200:
        try:
            body = reply.json()
        except ValueError:
            body = None
    if not isinstance(body, dict):
        print(
            f"sous claude: 127.0.0.1:{port} did not answer /sous/status "
            f"(status {reply.status_code})",
            file=sys.stderr,
        )
        return None
    return body


def _hold(port: int) -> dict | None:
    """Ask the daemon to keep the model loaded while this process lives.
    execve keeps the pid, so the process that leaves is the Claude Code
    session, and its start time tells the daemon a reused pid from a new
    one. Any failure is a warning: the session works without the hold, it
    just pays a reload after an idle unload."""
    import httpx
    import psutil

    body = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
    try:
        reply = httpx.post(
            f"http://127.0.0.1:{port}/sous/hold", json=body, timeout=_STATUS_TIMEOUT_SECONDS
        )
        reply.raise_for_status()
        answer = reply.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(
            f"sous claude: warning: could not hold the model ({type(exc).__name__}); "
            "launching anyway",
            file=sys.stderr,
        )
        return None
    if not isinstance(answer, dict):
        print(
            "sous claude: warning: could not hold the model (unexpected answer); launching anyway",
            file=sys.stderr,
        )
        return None
    return answer
```

In `_cmd_claude`: delete the `if "upstream_url" not in gateway:` block — lines 190–199, the three-line comment through its `raise SystemExit(1)`, keeping line 189's `gateway = status.get(...)` (its condition can no longer hold). Add a module constant next to `_DISALLOWED_FLAGS`:

```python
# Invocations that exit at once: holding the model for them would start a
# load nobody waits for and, once the process is gone, restart the idle
# clock under weights nobody is using. Only these four are knowable here.
_NO_HOLD_FLAGS = ("--help", "-h", "--version", "-v")
```

After `env = claude_env(...)` and before the env `print`, add:

```python
    held = None if user_args and user_args[0] in _NO_HOLD_FLAGS else _hold(config.server_port)
    if held is not None:
        model_id = status.get("model", {}).get("model_id", "the model")
        if held.get("loaded"):
            line = f"sous claude: {model_id} already loaded; held while this session runs"
        elif held.get("loading"):
            line = f"sous claude: preloading {model_id}; held while this session runs"
        else:
            # Not a state hold() produces, but this is another process's JSON.
            line = f"sous claude: {model_id} held while this session runs"
        print(line, file=sys.stderr)
```

Delete `import json` at the top of `cli.py`: `_server_status` was its only user (ruff F401 confirms). The function-local `anyio` imports go with the two deleted functions; `anyio` stays a package dependency (server.py uses it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS. `test_claude_execs_claude_with_the_gateway_environment` still asserts `"warning" not in err` — the "already loaded" line contains no "warning".

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: +15 (16 new cases incl. the four-way parametrization, −1 deleted) — `998 passed, 8 skipped, 29 deselected` on this Mac.

Run: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .`
Expected: clean. `mcp` is still a dependency of the package (the daemon is an MCP server); only the CLI's client-side import is gone.

- [ ] **Step 5: Commit**

```bash
git add src/sous/cli.py tests/test_cli.py
git commit -m "feat(cli): preflight over /sous/status; hold the model for the session

sous claude read server_status through an MCP client session — an async
client and a transport handshake to fetch one dict — and a daemon that
went idle between two subagents unloaded the model under the session.
The preflight now GETs /sous/status (a 404 is a daemon older than this
CLI: restart it; no compatibility dialect) and, once the checks pass,
POSTs /sous/hold with its own pid and start time before exec'ing claude,
which keeps the pid. The pre-routing check goes: a daemon that answers
/sous/status has an upstream_url by construction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md` ("Gateway mode": the paragraph ending "…is bounded by `CLAUDE_CODE_MAX_CONTEXT_TOKENS` instead." at lines ~194–209; the "gives up" list at ~226–300; the config block's `idle_unload_minutes` at line 404 and `max_context_tokens` comment at ~460–470; the `server_status` sentence at ~556), `CLAUDE.md` (the gotchas list)

**Interfaces:** none. Every sentence below is checked against the code as shipped in Tasks 1–4; an inaccuracy is the defect.

- [ ] **Step 1: README — what `sous claude` does at launch**

After the paragraph that ends "…the subagent's window is bounded by `CLAUDE_CODE_MAX_CONTEXT_TOKENS` instead." add:

> Before it execs, `sous claude` asks the daemon to keep the model loaded for the session: it `POST`s `/sous/hold` with its own process id and start time, and since `exec` keeps the process id, the holder *is* the Claude Code process. If nothing is loaded the daemon starts loading right away, on its own thread — the launcher prints `sous claude: preloading <model>; held while this session runs` (or `already loaded`) and does not wait, so the first subagent turn finds the weights resident or the load already under way instead of paying it. The daemon's idle sweep checks its holders on every poll between delegated tasks: when the `claude` process exits, the hold goes with it at the next sweep — within a second on an idle daemon; a delegated task in flight defers the sweep until it finishes — and the idle clock restarts from that moment, so quitting and relaunching inside `[model].idle_unload_minutes` never reloads either. A hold the daemon refuses is a warning, not a stop. Two caveats: a daemon restart forgets its holders (the model then unloads after `idle_unload_minutes` and the next subagent turn reloads it); and `sous claude --help`, `-h`, `--version` and `-v` skip the hold, while any other invocation that exits at once (`sous claude mcp list`, say) holds like a session — the load it may start is one nobody waits for, and the weights then stay resident for a fresh `idle_unload_minutes`.
>
> The preflight itself is plain HTTP: `GET /sous/status` on the daemon's port returns the same document the MCP `server_status` tool does. A `404` means the running daemon predates this CLI (the route did not exist); `sous claude` says so and exits 1 — restart the daemon from the same install (`sous stop`, then `sous serve` or `sous install-launchd`). There is no compatibility mode: the CLI and the daemon ship as one package.

- [ ] **Step 2: README — Claude Code compacts a local subagent early**

Add a bullet to "What a locally served turn gives up", after "**Mid-conversation system messages become `<system-reminder>` blocks.**":

> - **A shorter window, and Claude Code compacts inside it.** A frontier subagent has a 200K-token window; a local one has `[gateway].max_context_tokens` (131072 by default), which `sous claude` passes as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. Claude Code auto-compacts a conversation when *its own* token count reaches the window minus its output reserve (20K) minus a 13K buffer — 98K at the default window — and its "precompute" variant fires earlier still, at a fraction of that it takes from remote configuration. In practice a general-purpose subagent compacts after four or five tool turns at 131072: one call to write the summary (a warm hit, but a ~3.4K-token generation at the model's decode speed — about 3.5 minutes on the default model) and one to continue from it (the summary plus every re-attached file, ~30K tokens of prefill from the tools fork), and the agent then reports from a summary of itself. Nothing on the sous side changes this — both calls are served as cheaply as their content allows — and none of Claude Code's compaction switches (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `DISABLE_AUTO_COMPACT`, `autoCompactEnabled`) is per-model: each would also change the frontier main loop, so `sous claude` sets none of them. The lever is the window: the default model's native context is 262144, and `max_context_tokens = 262144` moves the classic threshold to ~229K at the cost of one more window of KV reserved out of the auto prompt-cache budget (8 GiB on the default model — the ~27 GiB above becomes ~19 GiB on a 64 GB machine, still room for a tools fork beside a retaining conversation). Where the earlier precompute trigger lands at that window is not known until measured: its fraction comes from Claude Code's remote configuration, keyed by window size. Restart the daemon after the edit; `sous claude` passes the running daemon's value.

- [ ] **Step 3: README — config comments**

In the config block, change

```toml
idle_unload_minutes = 30
```

to

```toml
idle_unload_minutes = 30   # a live `sous claude` session pins the model; the clock
                           # restarts when its last one exits
```

and append to the `max_context_tokens` comment (after "…grows past it and fails with "prompt is too long"."):

```toml
                                # Claude Code auto-compacts a local subagent once its own
                                # count nears this minus ~33K (sooner with its precompute
                                # trigger); 262144 — the default model's native length —
                                # pushes that out at the cost of 8 GiB of prompt-cache budget.
```

- [ ] **Step 4: README — `server_status`**

Change the sentence at ~556 "`server_status` reports `prompt_cache` — slots, resident bytes, hits, fork hits, retained and moved turn-slot takes, evictions and the subset the pressure valve took — counts only." to:

> `server_status` (and `GET /sous/status`, the same document over HTTP) reports `holders` (live `sous claude` sessions pinning the model), `loading` (a load in progress, a preload included) and `prompt_cache` — slots, resident bytes, hits, fork hits, retained and moved turn-slot takes, evictions and the subset the pressure valve took — counts only.

- [ ] **Step 5: CLAUDE.md gotchas**

Add two bullets to "## Gotchas", after the "Prompt-cache per-turn gauges" bullet:

> - `EngineManager` never holds its lock across a model load or unload
>   (`_loading`/`_unloading` on a `Condition`): `status()`, `hold()` and the
>   idle sweep must answer during either, and `/sous/status` reports
>   `loading`. A `hold(pid, create_time)` pins the model while that process
>   lives — the sweep prunes holders with psutil (pid gone, or start time
>   off by more than a second = reused pid) and restarts the idle clock when
>   the last leaves; the preload thread it may start calls `get()` and then
>   `release_mlx_thread_state()` like every mlx-touching thread. The `/sous/`
>   routes (`sous/monitor.py`) are mounted in `create_server` *before* the
>   gateway and whether or not it is enabled, behind the loopback guard in
>   `sous/loopback.py` that the gateway shares; a `/sous/` path never reaches
>   the upstream. `sous claude` has no MCP client: `/sous/status` or a 404 =
>   "restart the daemon", no compatibility fallback.
> - Claude Code auto-compacts a `sous-local` subagent when *its own* token
>   count nears `CLAUDE_CODE_MAX_CONTEXT_TOKENS − 33K` (sooner with its
>   precompute trigger) — two extra local calls, ~6 minutes at 131072 for a
>   four-turn agent, and the agent then answers from a summary. Every
>   compaction switch is global to the session (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`
>   is a *minimum* with the model window; `DISABLE_AUTO_COMPACT`,
>   `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`), so `sous claude` sets none; the
>   per-local-model lever is `[gateway].max_context_tokens` (the default
>   model's native length is 262144, at +8 GiB of KV reserve) — the classic
>   threshold scales with it, the precompute fraction is remote-configured
>   per window size and unmeasured at 262144.

- [ ] **Step 6: Verify the wording against the code and commit**

Read `EngineManager.hold`, `_prune_holders`, `unload_if_idle`, `monitor.mount_monitor`, `create_server` and `cli._daemon_status`/`_hold`/`_cmd_claude` once more and check every claim above: "at the next sweep — within a second on an idle daemon" (the worker polls every 0.5 s and sweeps only when `claim_next()` returns nothing, so a running task defers it), "the idle clock restarts from that moment", the four flags in `_NO_HOLD_FLAGS`, the exact stderr lines, "the same document", "no compatibility mode", the 20K/13K arithmetic (98,072 at 131072; 229,144 at 262144), "8 GiB" (131072 tokens × 64 KiB), "~27 → ~19 GiB" (the README's own auto-budget figure minus that), and that every compaction sentence calls the 229K figure the *classic* threshold and the precompute one unmeasured.

```bash
git add README.md CLAUDE.md
git commit -m "docs: describe preload and hold, the /sous/ routes and early compaction

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Orchestrator steps after Task 5

1. **Whole-branch review** (fresh Opus reviewer, the full diff against `main`): the plan-language sweep; the docs' every claim against the code; and specifically whether any path still runs the factory or `unload()` under `EngineManager._lock`, whether the preload thread can exit without `release_mlx_thread_state()`, whether any route handler does blocking work on the event loop (and that `run_sync` is imported from `anyio.to_thread`, not reached through `anyio.`), whether any path under `/sous` — the bare `/sous` included — can reach `Gateway.passthrough` with the gateway on, whether the sweep prunes before every early return, and whether `cli.py` still imports `mcp` or `json` anywhere.
2. **PR** `feat/gateway-continuity-b` → `main`, title `feat(engine,server,cli): preload and hold the model for a sous claude session`, body from the commits plus the compaction finding (one paragraph, the table above condensed), footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and nothing beneath it.
3. **Real-daemon gate (needs Kyle's go-ahead — installs branch code over his daemon):** on the M5 Pro, from `~/code/personal/sous-remote` at the branch: `uv sync`, the non-model suite, `uv tool install --force --reinstall .`, `sous stop`, wait for `~/.sous/daemon.lock` to clear, restart detached (`Popen(start_new_session=True)`), and **do not warm the model**. Ask Kyle first whether to set `[gateway].max_context_tokens = 262144` in his `config.toml` for this run (his call; it needs the restart anyway and costs 8 GiB of cache budget) so the agent does not compact and the gate measures this plan. Then one `sous claude -p` session inside the GUI `screen` session with one background general-purpose subagent of ≥3 tool-using turns, and:
   - **preload:** the launcher's stderr shows `preloading …; held while this session runs`; the daemon log shows `hold pid=N (holders=1)`, `preloading <model>` and `model_load seconds=…` — all three before the first `POST /v1/messages` line, in any order (`preloading` is logged after the thread starts and can trail a fast load); that first turn logs `load_s=0.0` **and completes** (`output_tokens` > 0, `status=200`), and the daemon pid in `~/.sous/daemon.lock` is the same afterwards — the generation after a preload is what exercises the preload thread's mlx release, which no fake-engine test can;
   - **status during the load:** while `preloading` is on stderr and before the first turn, `curl -s 127.0.0.1:8383/sous/status` three times a second; every answer returns within a second and reports `"loading": true, "loaded": false` until `model_load seconds=` lands;
   - **hold:** `curl -s 127.0.0.1:8383/sous/status` during the session reports `"holders": 1` and `"loaded": true`; after `claude` exits, the log shows `hold released pid=N (holders=0)` within a few seconds, `/sous/status` reports `"holders": 0` with `"loaded": true`, and `idle_seconds` counts from the release; a second reading a few minutes later still reports `"loaded": true` with `idle_seconds` below `idle_unload_minutes × 60` — the spec's "model still loaded for `idle_unload_minutes`";
   - **cache (as the A gate):** every real subagent turn after the first logs `cache=hit took=turn@N`, and `lcp` never appears above the header on any line;
   - **compaction accounting:** join the daemon-log turn lines to the subagent transcript by `id=msg_…` (the reconcile script from the A gate); a `compact_boundary` entry marks the call before it as the summary call and the call after it as the continuation. Report both with their seconds and tokens, separately from the per-turn criterion, and state the window the session ran with. If Kyle chose 262144 and a compaction still occurs, that is a finding to record, not a gate failure.
   Record the turn lines, the `/sous/status` readings and the compaction accounting on the PR. Reinstall from `main` after merge; restore Kyle's window setting if he asked for that.
4. **Memory:** the compaction finding (Claude Code's arithmetic, the levers, the ruling on global variables) is a project fact worth its own memory file; the B execution record updates `gateway-continuity-a-executed`'s "next" pointer and the specs memory.
