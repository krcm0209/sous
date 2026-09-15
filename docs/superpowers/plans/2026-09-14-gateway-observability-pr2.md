# Gateway Observability PR 2 — Live Status and `sous top` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** *What is the local model doing right now, and how far along is it* — answered live from a second terminal (`sous top`), from Claude Code's status line (`sous statusline`), and from one JSON document (`GET /sous/status`, `GET /sous/events`, the MCP `server_status` tool), without reading the log.

**Architecture:** Four pieces. (1) A process-wide `Inflight` registry (`sous/inflight.py`): every gateway turn registers under the `msg_` id its client already holds, advances through `queued → loading → tokenizing → prefill → decode` at the points the turn runner already stamps, counts generated tokens per delta, and is removed in the runner's `finally`; the routes record each completed or failed turn's summary (the turn line's fields) into a 50-entry ring. Every write is a dict update under one lock — nothing blocks, nothing raises, so the decode-loop path keeps the sink contract. (2) The status document is reshaped: `model` becomes `engine` (with `memory_gb` inside it), `inflight` is added, and the `/sous/status` variant carries `recent_turns` and `recent_tasks`; `GET /sous/events` streams that document over SSE — the first event at once, then one whenever the registry's version changed (polled ten times a second, so a burst is one event), and at least once a second so engine state that lives outside the registry (a load, a hold, the idle clock) still moves. (3) `sous statusline`: a stdlib-only subcommand that prints one line for Claude Code's `statusLine` setting. (4) `sous top` (alias `sous status --watch`): a Textual application in `sous/tui.py`, imported by that subcommand alone, fed by `/sous/events` — the presence rules in the spec are requirements here, and the look is the maintainer's: a whimsical fusion of chef-y motifs and early-90s nostalgia (an order slip, a kitchen-timer dial, a pixel chef, Memphis shapes, a HI-SCORE), specified in full in Task 5.

**Tech Stack:** Python 3.14; `threading.Lock`; Starlette routes via the MCP SDK's `custom_route`; `sse-starlette` (already a dependency) for `/sous/events`; `httpx` (already a dependency) as the TUI's SSE client; **Textual `>=8.2,<9` — the one new package**, verified to install on Python 3.14 (8.2.8 on 2026-09-14) and confined to `sous/tui.py`, plus `rich>=15` promoted from transitive to declared because `tui.py` imports `rich.text.Text` directly; `urllib` for the status line; pytest with the existing `FakeEngine`/`FakeUpstream` seams, Textual's `App.run_test()` pilot, and a hand-driven ASGI call for SSE tests. No mlx code changes. No new dev dependency (the render snapshot is a committed SVG compared as text — Textual's `export_screenshot()` is deterministic, verified).

**Spec:** `docs/superpowers/specs/2026-09-10-gateway-observability-design.md` — "PR 2 — live status, and a terminal that shows it" (Registry, Routes, The TUI, `sous statusline`), "Invariants preserved", "Testing" PR 2 and the real-daemon gate, "Documentation" (the PR 2 lines of "PR 4 — docs (folded)"), "Risks and open questions" (Textual, coalescing, per-process registry, the first-turn prefill ETA). PR 1 landed as #85 (`53c393a`); the continuity spec's B landed as #94 (`0bb08c9`) and created `sous/monitor.py`, `GET /sous/status` and `POST /sous/hold`, which this plan extends. PR 3 (persist turns, `sous turns`) follows this one and will persist exactly the summary dict Task 2 defines.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`; `ty` flags an *unused* pragma as its own diagnostic — add one only when `ty check` names the rule.
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local. Nothing this plan adds runs on a thread that keeps mlx state between calls: the status document is built on the anyio worker pool exactly as `server_status` is today (`_mlx_memory_gb()` releases per call), and the prefill probe reads the prompt cache's counters (a locked dict copy), never an array.
- **Textual is imported at module level only in `src/sous/tui.py`, and `sous.tui` only inside the `sous top` / `sous status --watch` command function** (`cli.py`), so `sous serve`, `sous claude` and `sous statusline` never load it (tests in Tasks 4 and 5 assert this in a fresh interpreter). `sous statusline` imports nothing beyond the stdlib (`urllib`, not `httpx`; a test asserts it).
- Nothing that blocks runs on the event loop: route handlers hand document building to `anyio.to_thread.run_sync` (imported from `anyio.to_thread`, as `monitor.py` and `proxy.py` already do); `_log_turn` gains one registry append (a lock held for microseconds by threads that hold it for microseconds) and nothing else.
- Sinks and the registry never block or raise on the engine's session thread: every `Inflight` method swallows an unknown id, takes no lock but its own, and does O(1) work under it (`snapshot()` is the one reader-side exception and is never called from a turn's thread).
- No request body, header value, query string, prompt text, tool name or file path reaches a log line, the registry, an SSE event, the status line or the TUI. Task titles (user-written, shown by `sous status` today) may appear in `recent_tasks`; model labels go through the existing `_log_token` bound.
- All `/sous/` routes are loopback-only (Host and Origin) via `sous.loopback.check_loopback`; nothing under `/sous` reaches the gateway's forwarder.
- The logger pins in `mount_gateway` (sse-starlette at INFO, httpx and httpcore at WARNING) are untouched; the TUI's httpx client runs in a different process and logs nothing (it configures no logging).
- `sous claude` sets no credential, tier or compaction variable (unchanged; the launcher's only edit here is the renamed key it reads).
- Tests never touch the real `~/.sous` or `~/.claude`: every path comes from `tmp_path`; every port from a free-port helper; every HTTP call in a test goes to an in-process app or a server the test started. The TUI's tests feed the app from an in-memory async iterator, never a socket; the one exception connects the real SSE reader to a closed loopback port and asserts it is `FeedDown` at once.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` (this plan and its spec included) once committed.
- **No plan language in committed code.** Code and test comments/docstrings must never cite a spec, a plan, a task or a step ("Task 2's helper", "Spec:", "the spec promises", "docs/superpowers/…"). Where a code block below carries such a reference, restate the underlying fact instead. Sweep the branch's own additions before the PR: `git diff main -- src tests | grep -nE '^\+.*(Task [0-9]|Step [0-9]|[Dd]ecision [0-9]|Phase [0-9]|gate [0-9]|[Ss]pec[: ]|docs/superpowers|PR [0-9])'` must print nothing.
- Never loosen an existing assertion to make a test pass. When this plan changes a behaviour an existing test pinned (named per task below), restate the test's expected value and its docstring; if a test's premise is gone, delete it and say so in the commit body.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>` — model-less, whatever the harness suggests. No session link anywhere.
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` (never pipe without `pipefail`; `pyproject.toml` already sets `-q`, so never add another `-q` — it suppresses the summary line), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`, and after Task 5 `uv lock --check`. Line length is 100 and `E501` is enforced: `ruff format` reflows code but never splits a string literal, so a long expected string is written as adjacent literals over two lines.
- The suite on `main` at `0bb08c9` collects 1050 tests under `-m "not model"`: `1013 passed, 8 skipped, 29 deselected in ~87 s` on this Mac (macOS 15.5 — eight `tests/test_int8prefill.py` cases skip without tensor units), `1021 passed, 29 deselected` on the M5 Pro. Each task's "Expected" line below gives the delta and the this-Mac line; on a machine with tensor units `passed` is 8 higher and there are no skips.
- Branch: `feat/gateway-observability-pr2` off `main` at `0bb08c9`. `main` is protected: PR with all four CI jobs green (`uv lock --check` on ubuntu — the lockfile must carry Textual and its four new transitive packages `linkify-it-py`, `mdit-py-plugins`, `platformdirs`, `uc-micro-py`; `markdown-it-py`, `mdurl`, `pygments` and `typing-extensions` are already locked, and `rich` 15.0.0 is already locked and becomes a declared dependency).
- the maintainer's daemon (on the M5 Pro, unmanaged, pid in `~/.sous/daemon.lock`) is not restarted by any task. The real-daemon gate at the end is the orchestrator's, after the maintainer's explicit go-ahead, since it installs branch code over his working `sous`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/inflight.py` (new) | `Inflight`: the in-flight turn registry (phases, tokens, rolling decode rate, the prefill probe, ETAs), the 50-entry ring of completed-turn summaries, the rolling prefill rate and typical output length, a `version` counter, `snapshot()`. Pure Python, injectable clock, no Starlette. |
| `src/sous/gateway/turn.py` | `TurnRunner(engines, config, inflight)`; `run(…, *, turn_id, model, stream)` registers, stamps phases, counts deltas, removes in `finally`; `_prefill_plan` probe. |
| `src/sous/gateway/routes.py` | `Gateway(…, inflight)` passes the id into `run`; `_turn_summary` (the turn line's fields as a dict), `_turn_line` (the identical line from it), `_failure_summary`; every `POST /v1/messages id=…` line also records a summary; `_METHODS` shared. |
| `src/sous/loopback.py` | Gains `ALL_METHODS` (the seven verbs the gateway catch-all and the `/sous/` 404 both register). |
| `src/sous/server.py` | `SousService(store, engines, config, inflight)`; `status_document(*, recent)`; `server_status()` = the document without `recent_*`; `create_server` builds one `Inflight` and hands it to the service, the monitor and the gateway. |
| `src/sous/monitor.py` | `mount_monitor(mcp, engines, status, inflight)`; `GET /sous/events` (SSE, 10 Hz poll of `inflight.version`, 1 Hz heartbeat, 10 s ping); constants `EVENT_TICK_SECONDS`, `EVENT_HEARTBEAT_SECONDS`, `EVENT_PING_SECONDS`. |
| `src/sous/cli.py` | `sous statusline` (stdlib only), `sous top` and `sous status --watch` (function-local `from sous.tui import run_top`), the launcher's one read (`engine.model_id`). |
| `src/sous/tui.py` (new) | The Textual app: the SSE feed reader with backoff, the vocabulary, palette and chef-sprite constants, pure view functions, widgets (the slip, the chef, the line, the book), CSS, motion; `run_top(port) -> int`. |
| `pyproject.toml`, `uv.lock` | `textual>=8.2,<9` and `rich>=15`. |
| `tests/test_inflight.py` (new), `tests/test_gateway_turn.py`, `tests/test_gateway_routes.py`, `tests/test_server.py`, `tests/test_monitor.py`, `tests/test_cli.py`, `tests/test_tui.py` (new), `tests/snapshots/sous-top-100x30.svg` (new) | Tests per task below. |
| `README.md`, `CLAUDE.md`, `SECURITY.md`, `CONTRIBUTING.md` | Docs per Task 7. |

## Decisions this plan makes beyond the spec's text

1. **The registry is its own module, `sous/inflight.py`, not part of `sous/monitor.py`.** The spec places it in the monitor; but `gateway/turn.py` has to import it, and the monitor is a route module (Starlette, the MCP server, sse-starlette). A turn runner must not import a route module for a data structure — the same reasoning that put the loopback guard in `sous/loopback.py` rather than have the monitor import the gateway's routes. The monitor imports the registry; nothing imports the monitor but `server.py`.
2. **`/sous/events` polls the registry's version at 10 Hz instead of being woken by writers.** Writers run on turn threads and the engine's session thread and have no handle on the event loop; a `call_soon_threadsafe` per delta would be a cross-thread wake-up per token. A 100 ms poll of one integer costs nothing, delivers a change within 100 ms, and *is* the spec's "coalesced to at most 10 events per second". The stream also sends the document once a second when nothing changed, because a load starting, a holder arriving or leaving and the idle clock are not registry changes and the TUI must show them; sse-starlette's own `ping` every 10 s stays as the spec's keepalive.
3. **The `status` event carries the whole document, built on a worker thread, at most 10 times a second.** One shape for the first event and every later one keeps the TUI a pure function of the last document. The build is `engines.status()` (its lock, never held across a load since #94), one `count_by_state` and one `list_recent(10)` on SQLite (sub-millisecond each), the registry snapshot, and `_mlx_memory_gb()` — at 10 Hz during a turn, a fraction of a percent of one core; measured in the gate.
4. **The prefill size is read by a probe, not stamped by the runner.** The runner is blocked inside `session.generate()` for the whole prefill and only learns `reused_tokens`/`prefilled_tokens` afterwards; but the cache adds the reuse when it takes the slot and assigns the `prefilled_tokens` gauge on entering `_run`, both *before* the first prefill call, and `stats(owner)` is a locked dict copy the cache's lock is never held across a prefill for (`_run` calls `hooks.prefill` outside it — read 2026-09-14). So the runner registers the prefill phase with a closure `_prefill_plan(engine, session.thread, before)`; `Inflight.snapshot()` calls it (outside the registry lock, from the status builder's worker thread) until it answers, then never again for that turn. Before the cache decides, the entry says `to_prefill: null` and the TUI pulses, which is the spec's "indeterminate" case made general. A cache with `enabled=False` never decides; the pulse is what it shows.
5. **The decode ETA and bar use a typical output length, capped by `max_tokens`.** The spec's `(max_tokens − generated)/decode_tps` alone is an upper bound of half an hour at Claude Code's 32 000 `max_tokens`, and the spec's own mock (`612 tok · eta ~00:18`) is plainly not that. The registry offers `expected_output_tokens` = the median `output_tokens` of the last 20 completed turns once five exist, capped by the turn's `max_tokens`; the ETA is `(expected − generated)/decode_tps` while `generated < expected`, and `null` otherwise (the label then shows the rate and the elapsed time only). The bar is `generated/expected`, capped at 95 % until the turn ends, and indeterminate with no estimate.
6. **The rolling decode rate is tokens over the last 2 s of deltas; the rolling prefill rate is an EMA (weight ⅓) over completed turns that prefilled ≥ 512 tokens.** A drafter emits 1–4 tokens per delta in bursts; two seconds smooths that and still shows a stall within two seconds. Small prefills are fixed cost (the probe, a fork copy) and would drag the rate that sizes the next ETA.
7. **Every `POST /v1/messages id=…` line records a summary, failures included.** The ring answers "why did that turn take that long", and a refused (529), abandoned (499), stalled or errored turn is an answer. Failure summaries carry `ts, id, model, stream, status, error, seconds`; success summaries carry every turn-line field under the spec's PR 3 column names (`stop_reason`, `cache`, `took`, `lcp`, `lcp_region`, `forks`, `evicted`, `pressure`, `load_s`, `queue_s`, `engine_wait_s`, `tokenize_s`, `ttft_s`, `prefill_s`, `decode_s`, `seconds`, `tools_hash`, `system_hash`), so PR 3 persists the dict as it is. The turn line is now *printed from* that dict (`_turn_line`), byte-identical to today's — the existing line-parsing tests are the proof.
8. **`server_status` (MCP) changes shape: `model` → `engine`, `memory_gb` inside it, `inflight` added; `/sous/status` adds `recent_turns` and `recent_tasks`.** The spec asks for it; the one consumer outside the tests is the launcher's single read of `model.model_id`, updated in Task 3. There is no compatibility key: CLI and daemon ship as one package.
9. **The TUI keeps the terminal's own ground and paints its accents in full colour.** `App(ansi_color=True)` with the `ansi-dark` theme pinned explicitly (`ansi_color` alone does not switch the theme — verified on 8.2.8) makes foreground and background `ansi_default`, i.e. whatever the terminal paints; hex accents still render as truecolor on top of it (verified: the SVG export carries them). That is the spec's "follows the terminal's colours; no hard-coded background". Every accent is chosen at relative luminance 0.19–0.29, the band that clears 3:1 against a near-black ground and against white, and a test asserts it; numbers are always the default foreground, so removing colour costs mood and nothing else. Six accents cannot blend, so a phase change is a slate step for a tenth of a second and then the phase colour (`Slip.show`), the spec's intent kept in the medium's idiom.
10. **The render snapshot is a committed SVG compared as text, no plugin.** `pytest-textual-snapshot` would add `syrupy`, whose pytest-9 support is unverified (the lock has pytest 9.1.1); `App.export_screenshot()` is deterministic for a deterministic screen (verified), so the test compares its output with `tests/snapshots/sous-top-100x30.svg`, writes `…​.actual.svg` beside it on a mismatch, and rewrites the file when `SOUS_UPDATE_SNAPSHOTS=1`. The screen is made deterministic by an injected clock and by waiting out the animations. A Textual upgrade that changes the rendering is a one-command update, documented in the test.
11. **The TUI's 10 fps timer runs only while something is in flight or loading.** Idle CPU under 2 % is a spec requirement; Textual's own idle is near zero, so the timers are what would spend it. The heartbeat event (1 Hz) keeps the idle screen current.
12. **`Ctrl-C` quits.** Textual 8 binds it to `help_quit` (a notification pointing at Ctrl-Q); the spec says Ctrl-C restores the terminal, so the app binds it to `quit` with priority.
13. **Two cheap hardenings deferred from #94 land here, in the files they touch:** the seven-verb list is one tuple (`sous.loopback.ALL_METHODS`) used by the gateway catch-all and the `/sous/` 404; and `tests/test_cli.py`'s `_claude_setup` docstring stops saying "a daemon that answers server_status" now that the launcher speaks HTTP.
14. **`sous status` stays static** (its rewrite is PR 3's); this PR adds `--watch` to it and `sous top` as the alias, both landing in the same function.
15. **Every frame animation reads the clock; only gauges tween.** The chef's frames, the timer dial, the pot's steam and the VHS bands are `int(now × fps) % n` from the app's injected clock, so a test with a pinned clock gets a pinned picture and the render snapshot is byte-stable; the two gauges and the row highlight tween, and `m` snaps them. The chef's faces are ASCII so no ambiguous-width glyph can shift one by a cell; the decorative glyphs (squiggle, triangle, dot, dial, gingham) are measured once at start-up and swapped for ASCII if the terminal counts any of them as two cells.

---

### Task 1: The in-flight registry (`inflight.py`)

**Files:**
- Create: `src/sous/inflight.py`
- Test: `tests/test_inflight.py` (new)

**Interfaces:**
- Consumes: nothing in the tree (stdlib only).
- Produces: `Inflight(*, clock=time.time)` with `version: int` (property), `begin(turn_id, *, model, stream, max_tokens)`, `phase(turn_id, phase, *, probe=None)`, `sized(turn_id, input_tokens)`, `progress(turn_id, generated_tokens)`, `end(turn_id)`, `finished(summary: dict)`, `snapshot() -> {"inflight": [entry…], "recent_turns": [summary…]}`; module constants `PHASES`, `RECENT_TURNS = 50`, `RATE_WINDOW_SECONDS = 2.0`, `RATE_MIN_PREFILL_TOKENS = 512`, `RATE_EMA = 0.33`, `TYPICAL_OUTPUT_TURNS = 20`, `TYPICAL_OUTPUT_MIN_TURNS = 5`; the `Probe` type alias `Callable[[], tuple[int, int] | None]`. An entry's keys: `id, model, stream, started_at, phase, phase_since, input_tokens, max_tokens, reused_tokens, to_prefill, generated_tokens, expected_output_tokens, decode_tps, eta_seconds`. Tasks 2–5 build on exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inflight.py`:

```python
"""The in-flight turn registry: phases, rates, the recent ring, and that
nothing in it blocks or raises on the threads that write to it."""

import threading

from sous.inflight import (
    RECENT_TURNS,
    TYPICAL_OUTPUT_MIN_TURNS,
    Inflight,
)


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _registry() -> tuple[Inflight, Clock]:
    clock = Clock()
    return Inflight(clock=clock), clock


def _summary(**overrides) -> dict:
    base = {
        "ts": 1_000.0,
        "id": "msg_a",
        "model": "sous-local",
        "stream": 1,
        "status": 200,
        "error": None,
        "stop_reason": "end_turn",
        "cache": "hit",
        "took": "turn@55175",
        "input_tokens": 58302,
        "output_tokens": 192,
        "reused_tokens": 55175,
        "prefilled_tokens": 3120,
        "prefill_s": 4.0,
        "decode_s": 12.0,
        "seconds": 16.5,
    }
    base.update(overrides)
    return base


def _begin(reg: Inflight, turn_id: str = "msg_a", max_tokens: int = 4096) -> None:
    reg.begin(turn_id, model="sous-local", stream=True, max_tokens=max_tokens)


def _entry(reg: Inflight, turn_id: str = "msg_a") -> dict:
    (entry,) = [e for e in reg.snapshot()["inflight"] if e["id"] == turn_id]
    return entry


def test_a_turn_walks_the_phases_and_leaves():
    reg, clock = _registry()
    _begin(reg)
    entry = _entry(reg)
    assert entry["phase"] == "queued" and entry["started_at"] == 1_000.0
    assert entry["model"] == "sous-local" and entry["stream"] is True
    assert entry["max_tokens"] == 4096 and entry["input_tokens"] == 0
    for phase in ("loading", "tokenizing", "prefill"):
        clock.tick(1.0)
        reg.phase("msg_a", phase)
        entry = _entry(reg)
        assert entry["phase"] == phase and entry["phase_since"] == clock.now
    reg.sized("msg_a", 58302)
    assert _entry(reg)["input_tokens"] == 58302
    clock.tick(1.0)
    reg.progress("msg_a", 1)
    entry = _entry(reg)
    # The first delta is what ends the prefill: nobody else can stamp it.
    assert entry["phase"] == "decode" and entry["phase_since"] == clock.now
    assert entry["generated_tokens"] == 1
    reg.end("msg_a")
    assert reg.snapshot()["inflight"] == []


def test_queued_turns_are_listed_in_arrival_order():
    reg, _ = _registry()
    _begin(reg, "msg_a")
    _begin(reg, "msg_b")
    assert [e["id"] for e in reg.snapshot()["inflight"]] == ["msg_a", "msg_b"]


def test_generated_tokens_and_the_decode_rate_follow_the_deltas():
    reg, clock = _registry()
    _begin(reg)
    reg.phase("msg_a", "prefill")
    for n in range(1, 31):  # 30 tokens over 2 s: 15 tok/s
        clock.tick(1 / 15)
        reg.progress("msg_a", n)
    entry = _entry(reg)
    assert entry["generated_tokens"] == 30
    assert entry["decode_tps"] == 15.0
    clock.tick(5.0)  # a stall: every sample has left the rate window
    assert _entry(reg)["decode_tps"] is None


def test_a_single_delta_has_no_rate_yet():
    reg, _ = _registry()
    _begin(reg)
    reg.progress("msg_a", 3)
    entry = _entry(reg)
    assert entry["decode_tps"] is None and entry["eta_seconds"] is None


def test_the_decode_eta_uses_the_typical_output_length_capped_by_max_tokens():
    reg, clock = _registry()
    assert _begin(reg, "msg_early") is None
    reg.progress("msg_early", 1)
    clock.tick(1.0)
    reg.progress("msg_early", 16)
    # Fewer than the minimum number of completed turns: no estimate, no bar.
    early = _entry(reg, "msg_early")
    assert early["expected_output_tokens"] is None and early["eta_seconds"] is None
    reg.end("msg_early")
    for n in range(TYPICAL_OUTPUT_MIN_TURNS):
        reg.finished(_summary(id=f"msg_{n}", output_tokens=600))
    _begin(reg)
    for n in range(1, 31):
        clock.tick(1 / 15)
        reg.progress("msg_a", n)
    entry = _entry(reg)
    assert entry["expected_output_tokens"] == 600
    assert entry["eta_seconds"] == 38.0  # (600 - 30) / 15 tok/s
    _begin(reg, "msg_b", max_tokens=100)
    for n in range(1, 31):
        clock.tick(1 / 15)
        reg.progress("msg_b", n)
    entry = _entry(reg, "msg_b")
    assert entry["expected_output_tokens"] == 100  # never past the client's cap
    for n in range(31, 101):
        reg.progress("msg_b", n)
    # Past the estimate the ETA is unknown again rather than negative.
    assert _entry(reg, "msg_b")["eta_seconds"] is None


def test_the_typical_output_is_a_median_over_recent_turns_only():
    reg, _ = _registry()
    for n in range(TYPICAL_OUTPUT_MIN_TURNS):
        reg.finished(_summary(id=f"old_{n}", output_tokens=5000))
    for n in range(20):
        reg.finished(_summary(id=f"new_{n}", output_tokens=200 + n))
    reg.finished(_summary(id="failed", status=500, error="api_error", output_tokens=0))
    _begin(reg)
    reg.progress("msg_a", 1)
    # The 20 newest with an output: 200..219, median 209 (the failure's 0
    # does not count as an output).
    assert _entry(reg)["expected_output_tokens"] == 209


def test_the_prefill_eta_comes_from_the_probe_and_the_rolling_rate():
    reg, clock = _registry()
    reg.finished(_summary(prefilled_tokens=4000, prefill_s=2.0))  # 2000 tok/s
    _begin(reg)
    answers: list[tuple[int, int] | None] = [None, (55175, 3000)]
    reg.phase("msg_a", "prefill", probe=lambda: answers.pop(0))
    entry = _entry(reg)
    # The cache has not decided yet: nothing to size the bar with.
    assert entry["reused_tokens"] is None and entry["to_prefill"] is None
    assert entry["eta_seconds"] is None
    clock.tick(0.5)
    entry = _entry(reg)
    assert entry["reused_tokens"] == 55175 and entry["to_prefill"] == 3000
    assert entry["eta_seconds"] == 1.0  # 3000 / 2000 tok/s, less the 0.5 s spent
    assert answers == []  # decided once; never asked again
    clock.tick(5.0)
    assert _entry(reg)["eta_seconds"] == 0.0
    reg.progress("msg_a", 1)
    entry = _entry(reg)
    assert entry["phase"] == "decode" and entry["to_prefill"] == 3000


def test_a_probe_that_raises_or_answers_late_is_not_an_error():
    reg, _ = _registry()
    _begin(reg)
    reg.phase("msg_a", "prefill", probe=lambda: 1 // 0)
    assert _entry(reg)["to_prefill"] is None
    reg.phase("msg_a", "prefill", probe=lambda: (0, 10))
    reg.end("msg_a")
    assert reg.snapshot()["inflight"] == []


def test_the_rolling_prefill_rate_ignores_small_prefills_and_smooths_the_rest():
    reg, _ = _registry()
    reg.finished(_summary(prefilled_tokens=100, prefill_s=1.0))  # below the floor
    _begin(reg)
    reg.phase("msg_a", "prefill", probe=lambda: (0, 2000))
    assert _entry(reg)["eta_seconds"] is None  # no rate yet
    reg.end("msg_a")
    reg.finished(_summary(prefilled_tokens=2000, prefill_s=1.0))  # 2000 tok/s
    reg.finished(_summary(prefilled_tokens=1000, prefill_s=1.0))  # 1000 tok/s
    reg.finished(_summary(prefilled_tokens=0, prefill_s=0.0))  # a hit with nothing to prefill
    _begin(reg, "msg_b")
    reg.phase("msg_b", "prefill", probe=lambda: (0, 1670))
    # 0.33 * 1000 + 0.67 * 2000 = 1670 tok/s; the zero-prefill hit changed nothing.
    assert _entry(reg, "msg_b")["eta_seconds"] == 1.0


def test_recent_turns_are_newest_first_and_capped():
    reg, _ = _registry()
    for n in range(RECENT_TURNS + 5):
        reg.finished(_summary(id=f"msg_{n}"))
    recent = reg.snapshot()["recent_turns"]
    assert len(recent) == RECENT_TURNS
    assert recent[0]["id"] == f"msg_{RECENT_TURNS + 4}"
    assert recent[-1]["id"] == "msg_5"
    recent[0]["id"] = "mutated"
    assert reg.snapshot()["recent_turns"][0]["id"] != "mutated"  # a copy, not the ring


def test_unknown_ids_are_ignored_never_raised():
    reg, _ = _registry()
    before = reg.version
    reg.phase("msg_x", "decode")
    reg.sized("msg_x", 5)
    reg.progress("msg_x", 5)
    reg.end("msg_x")
    assert reg.snapshot() == {"inflight": [], "recent_turns": []}
    assert reg.version == before


def test_every_change_bumps_the_version_and_a_snapshot_does_not():
    reg, _ = _registry()
    seen = [reg.version]

    def bumped() -> None:
        assert reg.version > seen[-1]
        seen.append(reg.version)

    _begin(reg)
    bumped()
    reg.phase("msg_a", "loading")
    bumped()
    reg.sized("msg_a", 10)
    bumped()
    reg.progress("msg_a", 1)
    bumped()
    reg.snapshot()
    assert reg.version == seen[-1]
    reg.end("msg_a")
    bumped()
    reg.finished(_summary())
    bumped()


def test_writes_from_many_threads_stay_consistent():
    reg, _ = _registry()
    ids = [f"msg_{n}" for n in range(16)]
    for turn_id in ids:
        _begin(reg, turn_id)

    def hammer(turn_id: str) -> None:
        for n in range(1, 201):
            reg.progress(turn_id, n)
        reg.snapshot()

    threads = [threading.Thread(target=hammer, args=(t,)) for t in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    assert {e["generated_tokens"] for e in reg.snapshot()["inflight"]} == {200}
    assert len(reg.snapshot()["inflight"]) == 16
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inflight.py 2>&1 | tail -3`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'sous.inflight'`.

- [ ] **Step 3: Write the registry**

Create `src/sous/inflight.py`:

```python
"""What the local model is doing right now, and what it just did.

A registry of in-flight gateway turns keyed by the `msg_` id the client
already holds, plus the last fifty completed turns' summaries: what
`/sous/status`, `/sous/events` and the MCP `server_status` tool read. Every
write is a dict update under one lock, microseconds long — `progress` runs
on the engine's session thread from inside the decode loop, where nothing
may block or raise — so no method raises for an unknown id and none takes
any lock but its own. Every value is a count, a duration, an identifier or
a phase name: never a token, a prompt or a tool name.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

PHASES = ("queued", "loading", "tokenizing", "prefill", "decode")
RECENT_TURNS = 50
# The rolling decode rate is tokens over the deltas of the last few seconds:
# long enough to smooth a drafter's bursts of one to four tokens, short
# enough that a stall reads as one.
RATE_WINDOW_SECONDS = 2.0
# A prefill below this size is mostly fixed cost (the probe, a fork copy) and
# would drag the rate that sizes the next turn's prefill ETA.
RATE_MIN_PREFILL_TOKENS = 512
# The newest turn's prefill rate carries a third of the estimate: one odd
# turn cannot swing the next ETA, a real change shows within three.
RATE_EMA = 0.33
# How many completed turns the typical output length is the median of, and
# how many it needs before it is offered at all.
TYPICAL_OUTPUT_TURNS = 20
TYPICAL_OUTPUT_MIN_TURNS = 5

# Answers (reused_tokens, to_prefill) once the prompt cache has decided them
# for a turn, None while it is still deciding. Called from a status reader's
# thread, off this module's lock — never from the turn's own thread, which
# is inside the generation for the whole prefill.
Probe = Callable[[], tuple[int, int] | None]


@dataclass
class _Turn:
    id: str
    model: str
    stream: bool
    max_tokens: int
    started_at: float
    phase_since: float
    phase: str = "queued"
    input_tokens: int = 0
    generated_tokens: int = 0
    reused_tokens: int | None = None
    to_prefill: int | None = None
    # (time, generated so far) per delta, enough for the rate window at a
    # hundred tokens a second.
    samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=256))
    probe: Probe | None = None


def _decode_rate(samples: deque[tuple[float, int]], now: float) -> float | None:
    """Tokens per second over the samples inside the rate window, or None
    with fewer than two of them — a stall, or a turn that just started."""
    recent = [(t, n) for t, n in samples if now - t <= RATE_WINDOW_SECONDS]
    if len(recent) < 2:
        return None
    (t0, n0), (t1, n1) = recent[0], recent[-1]
    if t1 <= t0 or n1 <= n0:
        return None
    return (n1 - n0) / (t1 - t0)


def _prefill_rate(summary: dict) -> float | None:
    tokens = summary.get("prefilled_tokens") or 0
    seconds = summary.get("prefill_s") or 0.0
    if tokens < RATE_MIN_PREFILL_TOKENS or seconds <= 0:
        return None
    return tokens / seconds


class Inflight:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._turns: dict[str, _Turn] = {}
        self._recent: deque[dict] = deque(maxlen=RECENT_TURNS)
        self._prefill_tps: float | None = None
        self._version = 0

    @property
    def version(self) -> int:
        """Bumped by every change: a reader that saw the same number twice
        saw the same registry twice. Read without the lock — one int."""
        return self._version

    def begin(self, turn_id: str, *, model: str, stream: bool, max_tokens: int) -> None:
        now = self._clock()
        with self._lock:
            self._turns[turn_id] = _Turn(turn_id, model, stream, max_tokens, now, now)
            self._version += 1

    def phase(self, turn_id: str, phase: str, *, probe: Probe | None = None) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn.phase, turn.phase_since, turn.probe = phase, self._clock(), probe
            self._version += 1

    def sized(self, turn_id: str, input_tokens: int) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn.input_tokens = input_tokens
            self._version += 1

    def progress(self, turn_id: str, generated_tokens: int) -> None:
        """One delta. The first one is what ends the prefill, so it moves
        the turn to `decode` itself: nothing else can see that moment."""
        now = self._clock()
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            if turn.phase != "decode":
                turn.phase, turn.phase_since, turn.probe = "decode", now, None
            turn.generated_tokens = generated_tokens
            turn.samples.append((now, generated_tokens))
            self._version += 1

    def end(self, turn_id: str) -> None:
        with self._lock:
            if self._turns.pop(turn_id, None) is not None:
                self._version += 1

    def finished(self, summary: dict) -> None:
        """A completed or failed turn's summary — the turn line's fields —
        kept newest first. A prefill of real size also feeds the rolling
        prefill rate the next turn's ETA is sized with."""
        with self._lock:
            self._recent.appendleft(dict(summary))
            rate = _prefill_rate(summary)
            if rate is not None:
                self._prefill_tps = (
                    rate
                    if self._prefill_tps is None
                    else RATE_EMA * rate + (1 - RATE_EMA) * self._prefill_tps
                )
            self._version += 1

    def snapshot(self) -> dict:
        """`{"inflight": [...], "recent_turns": [...]}`, every value
        JSON-ready and every dict a copy. A turn still in its prefill is
        asked for its size through its probe — here, outside this lock,
        because the probe takes the prompt cache's — so a caller belongs on
        a worker thread, never on the event loop."""
        with self._lock:
            pending = [
                (turn.id, turn.probe)
                for turn in self._turns.values()
                if turn.phase == "prefill" and turn.probe is not None and turn.to_prefill is None
            ]
        decided: dict[str, tuple[int, int]] = {}
        for turn_id, probe in pending:
            try:
                answer = probe()
            except Exception:  # noqa: BLE001 — the turn may have ended under the probe
                answer = None
            if answer is not None:
                decided[turn_id] = answer
        now = self._clock()
        with self._lock:
            for turn_id, (reused, to_prefill) in decided.items():
                turn = self._turns.get(turn_id)
                if turn is not None and turn.phase == "prefill":
                    turn.reused_tokens, turn.to_prefill = reused, to_prefill
            typical = self._typical_output()
            return {
                "inflight": [self._entry(turn, now, typical) for turn in self._turns.values()],
                "recent_turns": [dict(s) for s in self._recent],
            }

    def _typical_output(self) -> int | None:
        """Lock held by the caller. The median output of the newest turns
        that produced one; None until enough have."""
        outputs = [s["output_tokens"] for s in self._recent if s.get("output_tokens")][
            :TYPICAL_OUTPUT_TURNS
        ]
        if len(outputs) < TYPICAL_OUTPUT_MIN_TURNS:
            return None
        return int(statistics.median(outputs))

    def _entry(self, turn: _Turn, now: float, typical: int | None) -> dict:
        """Lock held by the caller."""
        tps = _decode_rate(turn.samples, now)
        eta: float | None = None
        expected: int | None = None
        if turn.phase == "prefill" and turn.to_prefill and self._prefill_tps:
            eta = max(0.0, turn.to_prefill / self._prefill_tps - (now - turn.phase_since))
        elif turn.phase == "decode":
            # Capped by the client's max_tokens, which is the hard stop; the
            # median is the likely one.
            expected = min(typical, turn.max_tokens) if typical else None
            if tps and expected and turn.generated_tokens < expected:
                eta = (expected - turn.generated_tokens) / tps
        return {
            "id": turn.id,
            "model": turn.model,
            "stream": turn.stream,
            "started_at": turn.started_at,
            "phase": turn.phase,
            "phase_since": turn.phase_since,
            "input_tokens": turn.input_tokens,
            "max_tokens": turn.max_tokens,
            "reused_tokens": turn.reused_tokens,
            "to_prefill": turn.to_prefill,
            "generated_tokens": turn.generated_tokens,
            "expected_output_tokens": expected,
            "decode_tps": None if tps is None else round(tps, 1),
            "eta_seconds": None if eta is None else round(eta, 1),
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inflight.py 2>&1 | tail -3`
Expected: `13 passed`.

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1026 passed, 8 skipped, 29 deselected`; `uv run ty check`; `uv run ruff format . && uv run ruff check . && uv run ruff format --check .` — all clean.

- [ ] **Step 5: Commit**

```bash
git add src/sous/inflight.py tests/test_inflight.py
git commit -m "feat(inflight): register in-flight gateway turns and keep recent summaries" -m "The daemon could say what a turn cost only once it was over. This registry holds every turn in flight — phase, tokens, a rolling decode rate, the prefill size once the cache decides it, an ETA — and the last fifty completed turns' summaries, under one lock held for microseconds, so the engine's decode loop can report a delta without blocking or raising. Nothing reads it yet." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: The turn runner registers, the routes record (`turn.py`, `routes.py`, `loopback.py`)

**Files:**
- Modify: `src/sous/gateway/turn.py` (`TurnRunner.__init__` at lines 109–124, `run` at lines 126–271), `src/sous/gateway/routes.py` (imports lines 12–46; `_rate` at 143–144; `Gateway.__init__` at 324–364; `messages` at 487–599; `_log_stream_failure` at 724–734; `_log_turn` at 736–768; `mount_gateway` at 771–802), `src/sous/loopback.py` (append `ALL_METHODS`), `src/sous/monitor.py` (line 28: `_METHODS` → the shared tuple)
- Test: `tests/test_gateway_turn.py` (append), `tests/test_gateway_routes.py` (`test_rates_and_optional_fields_print_a_dash_when_undefined` at lines 1802–1806 is restated; new tests appended)

**Interfaces:**
- Consumes: `Inflight` and `Probe` from Task 1; `ManagedEngine.prompt_cache_stats(owner=)`; `TurnAssembler.message_id`/`stop_reason`; `ChatRequest.tools_hash`/`system_hash`.
- Produces: `TurnRunner(engines, config, inflight: Inflight | None = None)`; `TurnRunner.run(messages, tools, max_tokens, sink, abandoned=None, *, turn_id: str = "", model: str = "-", stream: bool = False)`; `Gateway(engines, config, upstream=None, inflight: Inflight | None = None)` with `Gateway._inflight`; `mount_gateway(mcp, engines, config, *, upstream=None, inflight=None)`; module functions `_turn_summary(chat, result, assembler, *, stream) -> dict`, `_turn_line(summary) -> str`, `_failure_summary(turn_id, model, stream, status, error, received) -> dict`, `_per_second(count, seconds) -> float | None`; `sous.loopback.ALL_METHODS`. Task 3 hands one registry to `mount_gateway`; Task 5 reads the summary keys.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gateway_turn.py`:

```python
class _Recording(Inflight):
    """Every registry call in order, so a turn's walk through the phases
    can be pinned without racing the turn thread."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def begin(self, turn_id, *, model, stream, max_tokens):
        self.calls.append(("begin", turn_id, model, stream, max_tokens))
        super().begin(turn_id, model=model, stream=stream, max_tokens=max_tokens)

    def phase(self, turn_id, phase, *, probe=None):
        self.calls.append(("phase", turn_id, phase, probe is not None))
        super().phase(turn_id, phase, probe=probe)

    def sized(self, turn_id, input_tokens):
        self.calls.append(("sized", turn_id, input_tokens))
        super().sized(turn_id, input_tokens)

    def progress(self, turn_id, generated_tokens):
        self.calls.append(("progress", turn_id, generated_tokens))
        super().progress(turn_id, generated_tokens)

    def end(self, turn_id):
        self.calls.append(("end", turn_id))
        super().end(turn_id)


def test_a_turn_walks_the_registry_and_leaves_it(tmp_path: Path):
    inner = ChunkedFakeEngine(["Hel|lo"])
    live = _Recording()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    runner = TurnRunner(engines, _cfg(tmp_path), live)
    runner.run(MSGS, [], 4096, RecordingSink(), turn_id="msg_1", model="sous-local", stream=True)
    tokens = inner.count_tokens(MSGS, [])
    assert live.calls == [
        ("begin", "msg_1", "sous-local", True, 4096),
        ("phase", "msg_1", "loading", False),
        ("phase", "msg_1", "tokenizing", False),
        ("sized", "msg_1", tokens),
        ("phase", "msg_1", "prefill", True),
        ("progress", "msg_1", 1),
        ("progress", "msg_1", 2),
        ("end", "msg_1"),
    ]
    assert live.snapshot()["inflight"] == []


def test_the_registry_sees_the_turn_while_it_generates(tmp_path: Path):
    """What a status reader on another thread finds mid-turn: the entry in
    its prefill phase, and — through the probe — the sizes the cache has
    already decided, read off the owner's own counters."""
    inner = FakeEngine(["ok"])
    live = Inflight()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    runner = TurnRunner(engines, _cfg(tmp_path), live)
    seen: list[dict] = []
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        # Before the cache decides: the probe has nothing to say.
        seen.append(live.snapshot()["inflight"][0])
        inner.stats = {"reused_tokens": 55175, "prefilled_tokens": 3120}
        seen.append(live.snapshot()["inflight"][0])
        return original(messages, tools, max_tokens, on_delta)

    inner.generate = generate  # ty: ignore[invalid-assignment]
    runner.run(MSGS, [], 4096, RecordingSink(), turn_id="msg_1", model="sous-local")
    undecided, decided = seen
    assert undecided["phase"] == "prefill"
    assert undecided["reused_tokens"] is None and undecided["to_prefill"] is None
    assert decided["reused_tokens"] == 55175 and decided["to_prefill"] == 3120
    assert decided["input_tokens"] == inner.count_tokens(MSGS, [])


def test_a_failed_turn_leaves_the_registry_too(tmp_path: Path):
    class Boom(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            raise RuntimeError("no")

    live = Inflight()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: Boom([]))
    runner = TurnRunner(engines, _cfg(tmp_path), live)
    with pytest.raises(RuntimeError):
        runner.run(MSGS, [], 4096, RecordingSink(), turn_id="msg_1")
    assert live.snapshot()["inflight"] == []


def test_a_turn_refused_at_the_lock_leaves_the_registry(tmp_path: Path):
    live = Inflight()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: FakeEngine([]))
    runner = TurnRunner(engines, _cfg(tmp_path, gateway_generation_timeout_minutes=1), live)
    runner._timeout = 0.05
    with runner._lock, pytest.raises(GatewayBusy):
        runner.run(MSGS, [], 4096, RecordingSink(), turn_id="msg_1")
    assert live.snapshot()["inflight"] == []


def test_an_abandoned_turn_leaves_the_registry(tmp_path: Path):
    live = _Recording()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: FakeEngine(["never"]))
    runner = TurnRunner(engines, _cfg(tmp_path), live)
    gone = threading.Event()
    gone.set()
    with pytest.raises(TurnAbandoned):
        runner.run(MSGS, [], 4096, RecordingSink(), gone, turn_id="msg_1")
    assert live.calls == [
        ("begin", "msg_1", "-", False, 4096),
        ("phase", "msg_1", "loading", False),
        ("end", "msg_1"),
    ]
    assert live.snapshot()["inflight"] == []


def test_a_turn_without_an_id_never_touches_the_registry(tmp_path: Path):
    live = _Recording()
    engines = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: FakeEngine(["ok"]))
    runner = TurnRunner(engines, _cfg(tmp_path), live)
    runner.run(MSGS, [], 4096, RecordingSink())
    assert live.calls == []


def test_the_prefill_probe_reads_the_owners_counters_against_the_turns_baseline():
    from sous.gateway.turn import _prefill_plan

    inner = FakeEngine([])
    engine = ManagedEngine(inner)
    owner = threading.current_thread()
    inner.stats = {"reused_tokens": 40, "prefilled_tokens": 0}
    assert _prefill_plan(engine, owner, {"reused_tokens": 40}) is None
    inner.stats = {"reused_tokens": 90, "prefilled_tokens": 12}
    assert _prefill_plan(engine, owner, {"reused_tokens": 40}) == (50, 12)
    inner.stats = {"reused_tokens": 40, "prefilled_tokens": 12}
    assert _prefill_plan(engine, owner, {"reused_tokens": 40}) == (0, 12)
    assert inner.stats_owners[-1] is owner
```

Add to the file's imports: `from sous.engine.base import Delta, EngineManager, GenerationStalled, ManagedEngine, ReplaySafe` (add `ManagedEngine`) and `from sous.inflight import Inflight`.

Append to `tests/test_gateway_routes.py`:

```python
# --- the registry and the recent ring --------------------------------------------


def _recent(gateway: Gateway) -> list[dict]:
    return gateway._inflight.snapshot()["recent_turns"]


def test_a_served_turn_is_recorded_with_the_lines_fields(tmp_path: Path):
    inner = FakeEngine(["first", "second reply here"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        inner.stats["hits"] += 1
        inner.stats["reused_tokens"] += 100
        inner.stats["prefilled_tokens"] = 6
        inner.stats["prefill_seconds"] = 2.0
        inner.stats["decode_seconds"] = 4.0
        inner.stats["took_kind"], inner.stats["took_len"] = "turn", 100
        return original(messages, tools, max_tokens, on_delta)

    inner.generate = generate  # ty: ignore[invalid-assignment]
    gateway, app = _gateway_app(tmp_path, inner)
    r = _post(app, _body(stream=True))
    started = [d for e, d in _events(r.text) if e == "message_start"]
    message_id = started[0]["message"]["id"]
    (turn,) = _recent(gateway)
    assert turn["id"] == message_id and turn["model"] == "sous-local"
    assert turn["status"] == 200 and turn["error"] is None and turn["stream"] == 1
    assert turn["cache"] == "hit" and turn["took"] == "turn@100"
    assert turn["stop_reason"] == "end_turn"
    assert turn["output_tokens"] == 1 and turn["reused_tokens"] == 100
    assert turn["prefilled_tokens"] == 6 and turn["prefill_s"] == 2.0 and turn["decode_s"] == 4.0
    assert turn["prefill_tps"] == 3.0 and turn["decode_tps"] == 0.25
    assert turn["lcp"] is None and turn["lcp_region"] is None and turn["bounds"] is None
    assert turn["tools_hash"] is None and isinstance(turn["ts"], float)
    assert set(turn) == {
        "ts", "id", "model", "stream", "status", "error", "stop_reason", "cache", "took",
        "input_tokens", "output_tokens", "reused_tokens", "prefilled_tokens", "lcp",
        "lcp_region", "bounds", "forks", "evicted", "pressure", "load_s", "queue_s",
        "engine_wait_s", "tokenize_s", "ttft_s", "prefill_s", "decode_s", "prefill_tps",
        "decode_tps", "seconds", "tools_hash", "system_hash",
    }  # fmt: skip
    _post(app, _body(stream=False))
    ids = [t["id"] for t in _recent(gateway)]
    assert len(ids) == 2 and ids[1] == message_id  # newest first


def test_a_miss_is_recorded_with_its_lcp_and_bounds(tmp_path: Path):
    inner = FakeEngine(["reply"])
    inner.stats = dict(_ALL_GAUGES, misses=1, miss_lcp=62851, bound_lo=62298, bound_hi=62856)
    gateway, app = _gateway_app(tmp_path, inner)
    _post(app, _body())
    (turn,) = _recent(gateway)
    assert turn["cache"] == "miss" and turn["took"] == "none"
    assert turn["lcp"] == 62851 and turn["lcp_region"] == "system"
    assert turn["bounds"] == [62298, 62856]


def test_failed_and_refused_turns_are_recorded_with_their_error(tmp_path: Path, monkeypatch):
    class Boom(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            raise RuntimeError("engine down")

    gateway, app = _gateway_app(tmp_path, Boom([]))
    _post(app, _body(stream=False))
    _post(app, _body(stream=True))
    recent = _recent(gateway)
    assert [(t["status"], t["error"], t["stream"]) for t in recent] == [
        (200, "api_error", 1),  # the stream's headers had gone out
        (500, "api_error", 0),
    ]
    assert all(t["id"].startswith("msg_") and t["seconds"] >= 0 for t in recent)
    assert all("stop_reason" not in t for t in recent)
    monkeypatch.setattr("sous.gateway.routes.MAX_PENDING_TURNS", 0)
    gateway, app = _gateway_app(tmp_path, FakeEngine(["never"]))
    assert _post(app, _body()).status_code == 529
    (refused,) = _recent(gateway)
    assert (refused["status"], refused["error"]) == (529, "overloaded_error")


def test_an_abandoned_stream_is_recorded_as_499(tmp_path: Path, monkeypatch):
    gateway, app = _gateway_app(tmp_path, FakeEngine(["never"]))

    def gone(*args, **kwargs):
        raise TurnAbandoned

    monkeypatch.setattr(gateway._runner, "run", gone)
    _post(app, _body(stream=True))
    (turn,) = _recent(gateway)
    assert (turn["status"], turn["error"], turn["stream"]) == (499, "abandoned", 1)


def test_the_turn_line_is_printed_from_the_recorded_summary(tmp_path: Path, capsys):
    """One dict, two consumers: the log line and the recent ring must never
    disagree, so the line is formatted from the summary the ring keeps."""
    inner = FakeEngine(["reply"])
    inner.stats = dict(_ALL_GAUGES)
    gateway, app = _gateway_app(tmp_path, inner)
    _post(app, _body())
    (turn,) = _recent(gateway)
    (line,) = [ln for ln in _turn_lines(capsys.readouterr().err) if "status=200" in ln]
    fields = _fields(line)
    assert fields["id"] == turn["id"] and fields["cache"] == turn["cache"]
    assert fields["seconds"] == f"{turn['seconds']:.1f}"
    assert fields["prefill_tps"] == "-" and turn["prefill_tps"] is None


def test_the_registry_sees_a_streamed_turn_under_its_message_id(tmp_path: Path):
    inner = FakeEngine(["reply"])
    gateway, app = _gateway_app(tmp_path, inner)
    seen: list[dict] = []
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        seen.extend(gateway._inflight.snapshot()["inflight"])
        return original(messages, tools, max_tokens, on_delta)

    inner.generate = generate  # ty: ignore[invalid-assignment]
    r = _post(app, _body(stream=True))
    started = [d for e, d in _events(r.text) if e == "message_start"]
    (entry,) = seen
    assert entry["id"] == started[0]["message"]["id"]
    assert entry["phase"] == "prefill" and entry["stream"] is True
    assert entry["model"] == "sous-local" and entry["max_tokens"] == 4096
    assert gateway._inflight.snapshot()["inflight"] == []
```

Add `from sous.gateway.turn import TurnAbandoned` to the file's imports (after `from sous.gateway.routes import …`). Restate `test_rates_and_optional_fields_print_a_dash_when_undefined` (lines 1802–1806) as:

```python
def test_rates_and_optional_fields_print_a_dash_when_undefined():
    from sous.gateway.routes import _opt, _per_second

    assert _per_second(100, 0.0) is None and _per_second(100, 4.0) == 25.0
    assert _opt(None) == "-" and _opt(1.234) == "1.2" and _opt(_per_second(59, 4.0)) == "14.8"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_turn.py tests/test_gateway_routes.py 2>&1 | tail -3`
Expected: FAIL — `TypeError: TurnRunner.__init__() takes 3 positional arguments but 4 were given`, `run() got an unexpected keyword argument 'turn_id'`, `AttributeError: 'Gateway' object has no attribute '_inflight'`, `ImportError: cannot import name '_per_second'`.

- [ ] **Step 3: The runner registers**

In `src/sous/gateway/turn.py`, add `from sous.inflight import Inflight` after the `sous.engine.base` import block, and add this module function before `class TurnRunner`:

```python
def _prefill_plan(
    engine: ManagedEngine, owner: threading.Thread, before: dict
) -> tuple[int, int] | None:
    """(reused, to_prefill) once the cache has taken a slot for the turn and
    sized what it must prefill — it adds the reuse at the take and assigns
    the prefilled gauge on entering its run, both before the first prefill
    call — or None while it is still deciding, when both read zero. Called
    from a status reader's thread: the stats are a copy under the cache's
    own lock, which no prefill holds."""
    after = engine.prompt_cache_stats(owner=owner)
    reused = max(0, after.get("reused_tokens", 0) - before.get("reused_tokens", 0))
    to_prefill = after.get("prefilled_tokens", 0)
    if not reused and not to_prefill:
        return None
    return reused, to_prefill
```

Change `TurnRunner.__init__`'s signature to `def __init__(self, engines: EngineManager, config: SousConfig, inflight: Inflight | None = None):` and add, after `self._timeout = …`:

```python
        # Where a turn says what phase it is in and how far along. A runner
        # built without one (tests, count_tokens) keeps a private registry
        # nobody reads; the gateway hands in the daemon's.
        self._inflight = inflight or Inflight()
```

Replace `run` (lines 126–271) with a thin wrapper and the body it wraps:

```python
    def run(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sink: Sink,
        abandoned: threading.Event | None = None,
        *,
        turn_id: str = "",
        model: str = "-",
        stream: bool = False,
    ) -> TurnResult:
        """`turn_id` is the response's `msg_` id: with one, the turn is
        visible in the registry from here until it returns or raises."""
        live = self._inflight if turn_id else None
        if live is not None:
            live.begin(turn_id, model=model, stream=stream, max_tokens=max_tokens)
        try:
            return self._turn(messages, tools, max_tokens, sink, abandoned, live, turn_id)
        finally:
            if live is not None:
                live.end(turn_id)

    def _turn(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sink: Sink,
        abandoned: threading.Event | None,
        live: Inflight | None,
        turn_id: str,
    ) -> TurnResult:
        queued = time.monotonic()
        if not self._lock.acquire(timeout=self._timeout):
            raise GatewayBusy(f"no generation slot within {self._timeout:.0f}s")
        if self._closing:
            # Queued behind another turn while close() was giving up on it;
            # that other turn's finally already dropped (or will drop) the
            # session, so this one must not start a new generation on a
            # gateway that has already been told to shut down.
            self._lock.release()
            raise GatewayBusy("gateway is shutting down")
        started = time.monotonic()
        queue_seconds = started - queued
        if live is not None:
            live.phase(turn_id, "loading")
        try:
            # The idle sweep runs on the worker's thread, and _gen_lock only
            # covers generate(): without a lease the model could be unloaded
            # under count_tokens(), which is seconds of work on a long prompt.
            with self._engines.lease():
                if abandoned is not None and abandoned.is_set():
                    # Queued behind another turn, and the client gave up meanwhile:
                    # a generation nobody reads would only delay the live requests
                    # behind it. (A turn that has started still drains — the GPU
                    # cannot be interrupted and the lock discipline depends on it.)
                    raise TurnAbandoned
                engine = self._engines.get()
                # From `started`, not from here: get() waits out a load or
                # unload another thread is in the middle of, which costs this
                # turn exactly what loading the model itself would.
                load_seconds = time.monotonic() - started
                if abandoned is not None and abandoned.is_set():
                    # That wait is minutes when a hold started the load, and
                    # the client may have left during it: the check above ran
                    # before it, and the lease no longer parks a turn behind
                    # the load the way the manager lock once did.
                    raise TurnAbandoned
                if live is not None:
                    live.phase(turn_id, "tokenizing")
                session = self._session_for(engine)
                counting = time.monotonic()
                input_tokens = engine.count_tokens(messages, tools)
                tokenize_seconds = time.monotonic() - counting
                room = self._window - input_tokens
                if room <= 0:
                    raise PromptTooLong(input_tokens, self._window)
                if live is not None:
                    live.sized(turn_id, input_tokens)
                # Owner-scoped, so the before/after delta is exact: only this
                # session's thread moves these counters, and the turn holds
                # the gateway lock, so nothing else's hit or reset can land
                # between the two reads.
                before = engine.prompt_cache_stats(owner=session.thread)
                sink.started(input_tokens)
                final: Delta | None = None
                first_delta_at: float | None = None

                def on_delta(delta: Delta) -> None:
                    nonlocal final, first_delta_at
                    if first_delta_at is None:
                        first_delta_at = time.monotonic()
                    final = delta
                    if live is not None:
                        live.progress(turn_id, delta.output_tokens)
                    sink.delta(delta)

                # ReplaySafe tells the prompt cache this callback's output
                # never reaches a client, so a warm-cache failure may still
                # be retried cold — true exactly when the sink itself is.
                callback = ReplaySafe(on_delta) if sink.replay_safe else on_delta

                if live is not None:
                    # The registry learns the prefill's size from the cache's
                    # counters, read by whoever asks for status while this
                    # thread is inside the generation.
                    owner = session.thread
                    live.phase(
                        turn_id, "prefill", probe=lambda: _prefill_plan(engine, owner, before)
                    )
                generating = time.monotonic()
                try:
                    text = session.generate(
                        messages,
                        tools,
                        min(max_tokens, room),
                        timeout=self._timeout,
                        on_delta=callback,
                    )
                except GenerationStalled:
                    # The session is unusable after a stall (its thread may still
                    # be generating, holding the engine lock); the next turn gets a
                    # fresh one and waits on the lock like the worker would. Retire
                    # the stalled session's thread too: when the abandoned thread
                    # finishes it would publish the KV cache it built on ITS
                    # streams, and a cache is usable only from the thread that
                    # built it (#34). Retirement makes that late publish drop
                    # itself, and leaves the worker's slots alone.
                    stalled = session.thread
                    self._drop_session()
                    # Best-effort: a reset that raises would replace GenerationStalled
                    # with a generic 500 and lose the stall's classification.
                    with contextlib.suppress(Exception):
                        engine.reset_prompt_cache(owner=stalled)
                    raise
                after = engine.prompt_cache_stats(owner=session.thread)

                def delta_of(key: str) -> int:
                    return max(0, after.get(key, 0) - before.get(key, 0))

                cache_hit = delta_of("hits") > 0
                return TurnResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=final.output_tokens if final else 0,
                    finish_reason=final.finish_reason if final else "stop",
                    cache_hit=cache_hit,
                    forked=delta_of("fork_hits") > 0,
                    reused_tokens=delta_of("reused_tokens"),
                    # miss_lcp is a gauge the engine assigns per miss, so
                    # `after` holds this turn's value exactly when this turn
                    # missed — and a stale one from an earlier miss otherwise.
                    lcp=0 if cache_hit else after.get("miss_lcp", 0),
                    seconds=time.monotonic() - started,
                    # Gauges: assigned per turn by the cache, read directly like miss_lcp.
                    prefilled_tokens=after.get("prefilled_tokens", 0),
                    took_len=after.get("took_len", 0),
                    took_kind=after.get("took_kind", ""),
                    forks=delta_of("forks"),
                    evictions=delta_of("evictions"),
                    pressure_evictions=delta_of("pressure_evictions"),
                    load_seconds=load_seconds,
                    queue_seconds=queue_seconds,
                    engine_wait_seconds=session.lock_wait_seconds,
                    tokenize_seconds=tokenize_seconds + after.get("probe_seconds", 0.0),
                    ttft_seconds=None if first_delta_at is None else first_delta_at - generating,
                    prefill_seconds=after.get("prefill_seconds", 0.0),
                    decode_seconds=after.get("decode_seconds", 0.0),
                    bounds=(after.get("bound_lo", 0), after.get("bound_hi", 0)),
                )
        finally:
            self._engines.touch()
            if self._closing:
                # close() gave up on this turn's lock and returned False;
                # nothing else will ever retry the drop, so the turn that
                # held the lock does it here, while it still owns it. The
                # session's own thread dequeues _CLOSE and exits on its own
                # after this — joining it would make a turn thread wait on
                # shutdown work, which it must never do.
                self._drop_session()
            self._lock.release()
            # engines.get() may have loaded the model on this thread. Pool
            # threads outlive the call, but the invariant is per thread that
            # touched mlx (ml-explore/mlx#4327), and keeping it unconditional
            # is what makes it checkable.
            release_mlx_thread_state()
```

(The body is today's `run` with six `live` lines added; every existing comment stays.)

- [ ] **Step 4: The routes record, and print the line from the record**

In `src/sous/gateway/routes.py`:

Imports: add `import functools` (alphabetically after `import contextlib`), `from sous.inflight import Inflight` (after `from sous.gateway.upstream import …`), and change `from sous.loopback import check_loopback` to `from sous.loopback import ALL_METHODS, check_loopback`.

Replace `_rate` (lines 143–144) with:

```python
def _per_second(count: int, seconds: float) -> float | None:
    return None if seconds <= 0 else count / seconds
```

After `_classify` (line 199), add the three summary functions:

```python
def _turn_summary(
    chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
) -> dict:
    """The turn line's fields as one JSON-ready dict: what the registry keeps
    for the live view and what the line is printed from, so the two can
    never disagree. Counts, durations, hashes and identifiers only."""
    cache = "fork" if result.forked else "hit" if result.cache_hit else "miss"
    lo, hi = result.bounds
    miss = cache == "miss"
    return {
        "ts": time.time(),
        "id": assembler.message_id,
        "model": _model_label(chat),
        "stream": int(stream),
        "status": 200,
        "error": None,
        "stop_reason": assembler.stop_reason,
        "cache": cache,
        # A hit retried cold took no slot in the end: the cache zeroes its
        # per-turn gauges with the retry, so took_kind is "" there too.
        "took": f"{result.took_kind}@{result.took_len}" if result.took_kind else "none",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reused_tokens": result.reused_tokens,
        "prefilled_tokens": result.prefilled_tokens,
        # A hit already says how much was reused; a miss says how far the
        # render agreed with the closest slot and in which region — the
        # number that tells a changed tool array from a changed system text.
        "lcp": result.lcp if miss else None,
        "lcp_region": _lcp_region(result.lcp, lo, hi) if miss else None,
        "bounds": [b for b in (lo, hi) if b] if miss else None,
        "forks": result.forks,
        "evicted": result.evictions,
        "pressure": result.pressure_evictions,
        "load_s": result.load_seconds,
        "queue_s": result.queue_seconds,
        "engine_wait_s": result.engine_wait_seconds,
        "tokenize_s": result.tokenize_seconds,
        "ttft_s": result.ttft_seconds,
        "prefill_s": result.prefill_seconds,
        "decode_s": result.decode_seconds,
        "prefill_tps": _per_second(result.prefilled_tokens, result.prefill_seconds),
        "decode_tps": _per_second(result.output_tokens, result.decode_seconds),
        "seconds": result.seconds,
        "tools_hash": chat.tools_hash or None,
        "system_hash": chat.system_hash or None,
    }


def _turn_line(s: dict) -> str:
    diag = ""
    if s["cache"] == "miss":
        bounds = ",".join(str(b) for b in s["bounds"])
        diag = f" lcp={s['lcp']} lcp_region={s['lcp_region']} bounds=[{bounds}]"
    return (
        f"POST /v1/messages id={s['id']} model={s['model']} "
        f"stream={s['stream']} status=200 "
        f"input_tokens={s['input_tokens']} output_tokens={s['output_tokens']} "
        f"stop={s['stop_reason']} cache={s['cache']} took={s['took']} "
        f"reused_tokens={s['reused_tokens']} prefilled_tokens={s['prefilled_tokens']} "
        f"forks={s['forks']} evicted={s['evicted']} pressure={s['pressure']} "
        f"load_s={s['load_s']:.1f} queue_s={s['queue_s']:.1f} "
        f"engine_wait_s={s['engine_wait_s']:.1f} "
        f"tokenize_s={s['tokenize_s']:.1f} ttft_s={_opt(s['ttft_s'])} "
        f"prefill_s={s['prefill_s']:.1f} decode_s={s['decode_s']:.1f} "
        f"prefill_tps={_opt(s['prefill_tps'])} decode_tps={_opt(s['decode_tps'])} "
        f"seconds={s['seconds']:.1f}{diag} "
        f"tools={s['tools_hash'] or '-'} system={s['system_hash'] or '-'}"
    )


def _failure_summary(
    turn_id: str, model: str, stream: bool, status: int, error: str, received: float
) -> dict:
    """What the ring keeps for a turn that produced no result: the same head
    as a success plus the error type and the client's wait."""
    return {
        "ts": time.time(),
        "id": turn_id,
        "model": model,
        "stream": int(stream),
        "status": status,
        "error": error,
        "seconds": time.monotonic() - received,
    }
```

`Gateway.__init__`: signature becomes `def __init__(self, engines: EngineManager, config: SousConfig, upstream: Upstream | None = None, inflight: Inflight | None = None):`; replace `self._runner = TurnRunner(engines, config)` with:

```python
        # The daemon's registry when create_server hands one in (the status
        # routes read it); a private one otherwise, so the runner always has
        # somewhere to report.
        self._inflight = inflight or Inflight()
        self._runner = TurnRunner(engines, config, self._inflight)
```

In `messages()`:

- the 529 branch (lines 521–527): after the `_log(...)` call and before `return _error_response(...)`, add
  ```python
            self._inflight.finished(
                _failure_summary(
                    assembler.message_id, _model_label(chat), chat.stream, 529, "overloaded_error", received
                )
            )
  ```
- the stream branch's `turn()` (lines 548–556): the `self._runner.run(...)` call becomes
  ```python
                            self._runner.run(
                                chat.messages,
                                chat.tools,
                                chat.max_tokens,
                                sink,
                                abandoned,
                                turn_id=assembler.message_id,
                                model=_model_label(chat),
                                stream=True,
                            ),
  ```
  and the `except TurnAbandoned:` branch adds, after its `_log(...)`:
  ```python
                    self._inflight.finished(
                        _failure_summary(
                            assembler.message_id, _model_label(chat), True, 499, "abandoned", received
                        )
                    )
  ```
- the non-streaming submit (lines 575–583) becomes
  ```python
        future = self._submit(
            self._turns,
            self._pending,
            functools.partial(
                self._runner.run,
                chat.messages,
                chat.tools,
                chat.max_tokens,
                _NullSink(),
                turn_id=assembler.message_id,
                model=_model_label(chat),
                stream=False,
            ),
        )
  ```
  and its `except Exception as e:` branch adds, after the `_log(...)`:
  ```python
            self._inflight.finished(
                _failure_summary(
                    assembler.message_id, _model_label(chat), False, status, error_type, received
                )
            )
  ```

`_log_stream_failure` (lines 724–734): after the `_log(...)` call add
```python
        self._inflight.finished(
            # The client saw 200: the SSE headers had gone out before the failure.
            _failure_summary(
                assembler.message_id, _model_label(chat), True, 200, error_type, received
            )
        )
```

(`status` stays in use by the existing `level=_status_level(status)`.)

Replace `_log_turn` (lines 736–768) with:

```python
    def _log_turn(
        self, chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
    ) -> None:
        # On the event loop: the record is a lock held for microseconds by
        # threads that hold it for microseconds, and nothing else.
        summary = _turn_summary(chat, result, assembler, stream=stream)
        self._inflight.finished(summary)
        _log(_turn_line(summary))
```

`mount_gateway`: signature gains `inflight: Inflight | None = None` after `upstream`; `gateway = Gateway(engines, config, upstream, inflight)`; the catch-all registration's method list becomes `methods=list(ALL_METHODS)`.

Append to `src/sous/loopback.py`:

```python
# Every verb the gateway's catch-all forwards and the daemon's /sous/ 404
# answers: the two lists must stay one, or a verb would fall through one
# and reach the other.
ALL_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
```

In `src/sous/monitor.py`: delete line 28 (`_METHODS = [...]`), change the import to `from sous.loopback import ALL_METHODS, check_loopback`, and the two `methods=_METHODS` registrations (lines 121–122) to `methods=list(ALL_METHODS)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_turn.py tests/test_gateway_routes.py tests/test_monitor.py 2>&1 | tail -3`
Expected: all pass (the turn-line tests at lines 1719–1880 are what prove `_turn_line` is byte-identical).

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1039 passed, 8 skipped, 29 deselected`; `uv run ty check`; ruff format/check — clean.

- [ ] **Step 6: Commit**

```bash
git add src/sous/gateway/turn.py src/sous/gateway/routes.py src/sous/loopback.py src/sous/monitor.py tests/test_gateway_turn.py tests/test_gateway_routes.py
git commit -m "feat(gateway): report every turn to the registry, print the line from its record" -m "A turn now registers under its msg_ id before it queues for the gateway lock, stamps loading/tokenizing/prefill at the points its timings are taken, counts generated tokens per delta on the engine thread (a locked dict write, never a block), hands the registry a probe that reads the prefill's size off the cache's counters, and leaves in its finally. Every POST /v1/messages line — served, refused, abandoned, failed — also records a summary; the served line is printed from that summary so a row and a line cannot disagree. The seven-verb list the catch-all and the /sous/ 404 both register is one tuple now." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: One status document, `/sous/status` in full, `/sous/events` (`server.py`, `monitor.py`, `cli.py`)

**Files:**
- Modify: `src/sous/server.py` (`SousService.__init__` at lines 59–62, `_status_entry` at 107–120, `server_status` at 225–256, `create_server` at 303–415), `src/sous/monitor.py` (module docstring; imports; `mount_monitor` at lines 85–122), `src/sous/cli.py` (line 317: `status.get("model", {})` → `status.get("engine", {})`; the `_daemon_status` docstring's "GET /sous/status" sentence is unchanged)
- Test: `tests/test_server.py` (`test_server_status` at 337–344, `test_server_status_reports_context_policy`/`…gateway_config` unchanged; new tests), `tests/test_monitor.py` (`_await_loaded` at 62–70, `test_status_is_the_server_status_document` at 73–87, `test_hold_registers_the_caller_and_preloads` at 90–100, `test_a_failing_status_or_hold_is_an_error_body_not_a_bare_500` at 204–236; new tests), `tests/test_cli.py` (`_status` at 832–853: the `"model"` key becomes `"engine"`; its docstring names `engine.model_id`; `_claude_setup`'s docstring at 856–858 becomes "A gateway-enabled config file, a `claude` on PATH, a daemon that answers `GET /sous/status`, and an execve that records instead of replacing the process.")

**Interfaces:**
- Consumes: `Inflight.snapshot()`/`version` (Task 1); `Gateway`/`mount_gateway(…, inflight=)` (Task 2).
- Produces: `SousService(store, engines, config, inflight: Inflight | None = None)` with `.inflight`; `SousService.status_document(*, recent: bool) -> dict` and `server_status()`; document keys `engine` (`loaded, loading, model_id, idle_seconds, holders, memory_gb, prompt_cache?, int8_prefill?`), `inflight` (list of Task 1 entries), `queue` (`queued, running`), `config` (as today), and with `recent=True` `recent_turns` (Task 2 summaries, newest first) and `recent_tasks` (`[{id, state, title, seconds}]`, newest first, ≤ 10); `mount_monitor(mcp, engines, status, inflight)`; `GET /sous/events` — `event: status` with the document as `data`, first at once, then on registry change (≤ 10/s) and at least every `EVENT_HEARTBEAT_SECONDS`, plus `event: ping` every `EVENT_PING_SECONDS`; constants `EVENT_TICK_SECONDS = 0.1`, `EVENT_HEARTBEAT_SECONDS = 1.0`, `EVENT_PING_SECONDS = 10`. Tasks 4 and 5 read this document; Task 5 reads this stream.

- [ ] **Step 1: Write the failing tests**

In `tests/test_server.py`, restate `test_server_status` (lines 337–344):

```python
def test_server_status(svc):
    service, _, root = svc
    service.delegate_task("t", "x", str(root))
    s = service.server_status()
    assert s["queue"]["queued"] == 1
    assert s["engine"]["loaded"] is False and s["engine"]["loading"] is False
    assert s["engine"]["holders"] == 0 and "memory_gb" in s["engine"]
    assert s["inflight"] == []
    assert s["config"]["model_id"]
    assert ["pytest"] in s["config"]["allowlist"]
    # The MCP tool's answer is the document without the two recent lists:
    # a frontier model pays to read every key.
    assert set(s) == {"engine", "inflight", "queue", "config"}
```

and append:

```python
def test_status_document_with_recents_lists_turns_and_tasks(svc):
    from sous.inflight import Inflight

    service, store, root = svc
    assert isinstance(service.inflight, Inflight)
    tid = service.delegate_task("scaffold the fixtures", "x", str(root))["task_id"]
    store.claim_next()
    service.inflight.begin("msg_1", model="sous-local", stream=True, max_tokens=64)
    service.inflight.finished({"ts": 1.0, "id": "msg_0", "status": 200})
    doc = service.status_document(recent=True)
    assert set(doc) == {"engine", "inflight", "queue", "config", "recent_turns", "recent_tasks"}
    assert [t["id"] for t in doc["inflight"]] == ["msg_1"]
    assert doc["recent_turns"] == [{"ts": 1.0, "id": "msg_0", "status": 200}]
    (task,) = doc["recent_tasks"]
    assert task["id"] == tid and task["state"] == "running"
    assert task["title"] == "scaffold the fixtures" and task["seconds"] >= 0
    assert set(task) == {"id", "state", "title", "seconds"}
    assert doc["queue"] == {"queued": 0, "running": 1}


def test_recent_tasks_are_capped_at_ten_newest_first(svc):
    service, store, root = svc
    for n in range(12):
        store.enqueue(
            title=f"t{n}",
            instructions="x",
            project_root=str(root),
            context_files=[],
            verify_commands=[],
        )
    titles = [t["title"] for t in service.status_document(recent=True)["recent_tasks"]]
    assert titles == [f"t{n}" for n in range(11, 1, -1)]


```

`create_server` does not hand its service back, so the sharing of one registry between the gateway and the status routes is proven through the app. Append to `tests/test_monitor.py`:

```python
def test_status_carries_the_recent_ring_the_gateway_writes(tmp_path: Path):
    """create_server builds one registry for the gateway and the status
    routes: a turn the gateway served shows up in /sous/status."""
    from tests.fake_engine import FakeEngine as _Fake

    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: _Fake(["reply"]))
    app = create_server(
        TaskStore(tmp_path / "tasks.db"), engines, cfg, upstream=FakeUpstream().upstream()
    ).streamable_http_app()
    body = {
        "model": "sous-local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _request(app, "POST", "/v1/messages", body).status_code == 200
    doc = _request(app, "GET", "/sous/status").json()
    (turn,) = doc["recent_turns"]
    assert turn["status"] == 200 and turn["model"] == "sous-local"
```

In `tests/test_monitor.py`, restate `_await_loaded` (lines 62–70) to read `["engine"]`:

```python
def _await_loaded(app, timeout: float = 5.0) -> dict:
    """The preload runs on its own thread; poll the route the launcher would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        engine = _request(app, "GET", "/sous/status").json()["engine"]
        if engine["loaded"] and not engine["loading"]:
            return engine
        time.sleep(0.01)
    raise AssertionError("the preload never finished")
```

restate `test_status_is_the_server_status_document` (lines 73–87):

```python
def test_status_is_the_full_status_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "GET", "/sous/status")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    doc = r.json()
    assert doc["engine"]["loaded"] is False
    assert doc["engine"]["loading"] is False
    assert doc["engine"]["holders"] == 0
    assert "memory_gb" in doc["engine"]
    assert doc["inflight"] == [] and doc["recent_turns"] == [] and doc["recent_tasks"] == []
    assert doc["config"]["gateway"] == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
    }
    assert set(doc) == {"engine", "inflight", "queue", "config", "recent_turns", "recent_tasks"}
```

and in `test_hold_registers_the_caller_and_preloads` (lines 95–96) rename the local `model = _await_loaded(app)` / `model["holders"]` to `engine`. In `test_a_failing_status_or_hold_is_an_error_body_not_a_bare_500` (lines 204–236) the route no longer calls `server_status`, so change only these two lines — the docstring, `failing_hold`, the body-shape and the caplog assertions stay:

```python
    def failing_status(self, *, recent):
        raise RuntimeError("tasks.db is locked")

    monkeypatch.setattr(SousService, "status_document", failing_status)
```

Add `from sous import monitor` to the file's imports (line 18's `from sous.monitor import HOLD_BODY_LIMIT` stays). Then append:

```python
# --- /sous/events -------------------------------------------------------------------


def _frames(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data))
    return out


def _collect_events(app, *, want: int, seconds: float = 3.0, headers=(), during=None) -> list:
    """GET /sous/events straight through the ASGI app — httpx's transport
    buffers a response to its end, and this one has none — until `want`
    non-ping frames have arrived, then disconnect and wait for the handler
    to finish. `during(loop)` runs once the first frame is in."""

    async def go():
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/sous/events",
            "raw_path": b"/sous/events",
            "query_string": b"",
            "headers": [(b"host", b"127.0.0.1:8383"), *headers],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8383),
        }
        disconnect = asyncio.Event()
        got = asyncio.Event()
        chunks: list[bytes] = []
        status: list[int] = []

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                status.append(message["status"])
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                frames = _frames(b"".join(chunks).decode())
                if len(frames) == 1 and during is not None:
                    during()
                if len([f for f in frames if f[0] != "ping"]) >= want:
                    got.set()

        task = asyncio.create_task(app(scope, receive, send))
        try:
            if want:
                await asyncio.wait_for(got.wait(), seconds)
            else:
                # A refused request completes on its own; nothing to wait for.
                await asyncio.wait_for(task, seconds)
        finally:
            disconnect.set()
            # The disconnect must end the stream: a generator that kept
            # running would be a client's whole session of work for nobody.
            await asyncio.wait_for(task, 5)
        return status, _frames(b"".join(chunks).decode())

    return asyncio.run(go())


def test_events_start_with_the_full_document(tmp_path: Path):
    app, _ = _app(tmp_path)
    status, frames = _collect_events(app, want=1)
    assert status == [200]
    event, doc = frames[0]
    assert event == "status"
    assert set(doc) == {"engine", "inflight", "queue", "config", "recent_turns", "recent_tasks"}
    assert doc["engine"]["loaded"] is False


def test_events_follow_registry_changes_coalesced(tmp_path: Path, monkeypatch):
    """A hundred changes inside one tick are one event, not a hundred; and
    the heartbeat must not add to the count within the window."""
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 60.0)
    registry = Inflight()
    app, _ = _app(tmp_path, inflight=registry)

    def burst() -> None:
        for n in range(100):
            registry.begin(f"msg_{n}", model="sous-local", stream=True, max_tokens=1)
            registry.end(f"msg_{n}")
        registry.begin("msg_last", model="sous-local", stream=True, max_tokens=1)

    _, frames = _collect_events(app, want=2, during=burst)
    statuses = [d for e, d in frames if e == "status"]
    assert 2 <= len(statuses) <= 3
    assert [t["id"] for t in statuses[-1]["inflight"]] == ["msg_last"]


def test_events_beat_once_a_second_when_nothing_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.2)
    app, _ = _app(tmp_path)
    started = time.monotonic()
    _, frames = _collect_events(app, want=3)
    assert len([e for e, _ in frames if e == "status"]) >= 3
    assert time.monotonic() - started < 2.0


def test_events_ping_on_the_configured_cadence(tmp_path: Path, monkeypatch):
    """The heartbeat keeps status frames flowing so the collector stays
    connected past the first ping; the ping itself is sse-starlette's."""
    monkeypatch.setattr(monitor, "EVENT_PING_SECONDS", 0.2)
    monkeypatch.setattr(monitor, "EVENT_HEARTBEAT_SECONDS", 0.15)
    app, _ = _app(tmp_path)
    _, frames = _collect_events(app, want=3, seconds=3.0)
    assert ("ping", {"type": "ping"}) in frames


def test_events_are_loopback_guarded_and_get_only(tmp_path: Path):
    app, _ = _app(tmp_path)
    status, frames = _collect_events(
        app, want=0, headers=[(b"origin", b"https://evil.example")], seconds=1.0
    )
    assert status == [403] and frames == []
    r = _request(app, "POST", "/sous/events")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_events_document_is_built_off_the_event_loop(tmp_path: Path, monkeypatch):
    from sous.server import SousService

    threads: set[str] = set()
    original = SousService.status_document

    def recording(self, *, recent):
        threads.add(threading.current_thread().name)
        return original(self, recent=recent)

    monkeypatch.setattr(SousService, "status_document", recording)
    app, _ = _app(tmp_path)
    _collect_events(app, want=1)
    assert threads and "MainThread" not in threads
```

`test_events_follow_registry_changes_coalesced` needs the registry the app was built with: `_app` in this file forwards one to `create_server`. Restate `_app` (lines 26–37):

```python
def _app(
    tmp_path: Path, *, gateway_enabled: bool = False, upstream=None, alive=None, inflight=None
):
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        gateway_enabled=gateway_enabled,
    )
    engines = EngineManager(
        cfg, engine_factory=lambda mid: FakeEngine([]), holder_alive=alive or (lambda pid, ct: True)
    )
    store = TaskStore(tmp_path / "tasks.db")
    upstream = upstream or FakeUpstream().upstream()
    server = create_server(store, engines, cfg, upstream=upstream, inflight=inflight)
    return server.streamable_http_app(), engines
```

Add `import threading` and `from sous.inflight import Inflight` to the file's imports.

In `tests/test_cli.py`, restate `_status` (lines 832–853) with the key renamed and the docstring "…of which the launcher reads `config.gateway` for its checks and `engine.model_id` for its one line", and `_claude_setup`'s docstring as given in Files above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_server.py tests/test_monitor.py tests/test_cli.py 2>&1 | tail -3`
Expected: FAIL — `KeyError: 'engine'`, `AttributeError: 'SousService' object has no attribute 'status_document'`, the events tests time out with `status == [404]`, and `test_claude_holds_the_model_after_the_checks_and_says_so` prints `the model` instead of the id.

- [ ] **Step 3: The service builds one document**

In `src/sous/server.py`: add `from sous.inflight import Inflight` after `from sous.gateway.upstream import Upstream`. Change `SousService.__init__` to:

```python
    def __init__(
        self,
        store: TaskStore,
        engines: EngineManager,
        config: SousConfig,
        inflight: Inflight | None = None,
    ):
        self.store = store
        self.engines = engines
        self.config = config
        # The gateway's turns report here; create_server hands the same
        # registry to both, so the status document sees what the gateway does.
        self.inflight = inflight or Inflight()
```

Replace `server_status` (lines 225–256) with:

```python
    def _task_summary(self, t: Task) -> dict:
        end = t.finished_at or time.time()
        return {
            "id": t.id,
            "state": t.state,
            "title": t.title,
            "seconds": round(end - t.started_at) if t.started_at else 0,
        }

    def status_document(self, *, recent: bool) -> dict:
        """The one document every status surface serves. `recent` adds the
        last fifty turns and the last ten tasks — for the HTTP routes and
        the terminal, not for the MCP tool, where a frontier model pays for
        every key it reads. Runs on a worker thread: the task store is
        SQLite, the engine's status takes its lock, and the registry's
        snapshot may call into the prompt cache."""
        counts = self.store.count_by_state()
        engine = self.engines.status()
        engine["memory_gb"] = _mlx_memory_gb()
        live = self.inflight.snapshot()
        document = {
            "engine": engine,
            "inflight": live["inflight"],
            "queue": {
                "queued": counts.get(TaskState.QUEUED, 0),
                "running": counts.get(TaskState.RUNNING, 0)
                + counts.get(TaskState.AWAITING_APPROVAL, 0),
            },
            "config": {
                "model_id": self.config.model_id,
                "port": self.config.server_port,
                "max_turns": self.config.max_turns,
                "max_minutes": self.config.max_minutes,
                "allowlist": current_allowlist(self.config.config_path),
                "context": {
                    "mode": self.config.context_mode,
                    "fraction": self.config.context_fraction,
                    "min_tokens": self.config.context_min_tokens,
                    # Fixed mode's operative value — without it a client sees
                    # THAT the policy is fixed but not what it's fixed to.
                    "max_context_tokens": self.config.max_context_tokens,
                },
                "gateway": {
                    "enabled": self.config.gateway_enabled,
                    "local_models": list(self.config.gateway_local_models),
                    "max_context_tokens": self.config.gateway_max_context_tokens,
                    "upstream_url": self.config.gateway_upstream_url,
                },
            },
        }
        if recent:
            document["recent_turns"] = live["recent_turns"]
            document["recent_tasks"] = [
                self._task_summary(t) for t in self.store.list_recent(limit=10)
            ]
        return document

    def server_status(self) -> dict:
        return self.status_document(recent=False)
```

`create_server`'s signature gains a keyword-only `inflight: Inflight | None = None` after `upstream` (a test that drives the app directly hands in the registry it will write to); its docstring-free body replaces `svc = SousService(store, engines, config)` with

```python
    # One registry: the gateway's turns write it, the status routes and the
    # MCP tool read it. A caller may pass its own (a test that writes to it
    # while driving the app).
    inflight = inflight or Inflight()
    svc = SousService(store, engines, config, inflight)
```

and the two mounts (lines 411–413) with

```python
    mount_monitor(mcp, engines, lambda: svc.status_document(recent=True), inflight)
    if config.gateway_enabled:
        mounted_gateway.append(
            mount_gateway(mcp, engines, config, upstream=upstream, inflight=inflight)
        )
```

- [ ] **Step 4: The events route**

In `src/sous/monitor.py`, module docstring becomes:

```python
"""The daemon's own loopback routes under /sous/: what `sous claude`, `sous
top` and `sous statusline` speak to the daemon. Mounted before the gateway's
routes and whatever the gateway flag says, so a path under /sous — the bare
/sous included — is answered here or 404s here for every method the routes
register (a verb none of them lists gets Starlette's own 405 first), and is
never forwarded to the upstream."""
```

Imports: add `import asyncio`, `import time`, `from collections.abc import AsyncIterator, Callable` (replacing the bare `Callable` import), `from sse_starlette import EventSourceResponse, ServerSentEvent`, `from sous.inflight import Inflight`. After `HOLD_BODY_LIMIT`, add:

```python
# The event stream polls the registry's version this often: a burst of
# changes inside one tick is one event, and a change is on the wire within
# a tick. Ten a second is what a terminal can show and what a token-per-
# delta feed produces at the default model's decode speed.
EVENT_TICK_SECONDS = 0.1
# Engine state (a load starting, a holder leaving, the idle clock) is not
# in the registry, so the document also goes out this often unchanged.
EVENT_HEARTBEAT_SECONDS = 1.0
# sse-starlette's keepalive, for a client behind something that closes a
# silent connection; the daemon's own /v1 stream uses the same interval.
EVENT_PING_SECONDS = 10
_SEP = "\n"
_PING = ServerSentEvent(event="ping", data='{"type": "ping"}', sep=_SEP)


async def _status_events(
    status: Callable[[], dict], inflight: Inflight
) -> AsyncIterator[ServerSentEvent]:
    """The document now, then again whenever the registry changed since the
    last one went out — checked every tick — and at least once a heartbeat
    regardless. Built on a worker thread each time. Nothing is buffered for
    a client that is gone: the response cancels this generator on
    disconnect, and the tick's sleep is where that lands."""
    seen: int | None = None
    sent = float("-inf")
    while True:
        version = inflight.version
        now = time.monotonic()
        if version != seen or now - sent >= EVENT_HEARTBEAT_SECONDS:
            document = await run_sync(status)
            yield ServerSentEvent(
                event="status", data=json.dumps(document, separators=(",", ":")), sep=_SEP
            )
            seen, sent = version, now
        await asyncio.sleep(EVENT_TICK_SECONDS)
```

`mount_monitor` becomes:

```python
def mount_monitor(
    mcp: MCPServer, engines: EngineManager, status: Callable[[], dict], inflight: Inflight
) -> None:
    """Register GET /sous/status, GET /sous/events, POST /sous/hold and a
    404 for every other /sous/ path. `status` builds the full status
    document (recent turns and tasks included); `inflight` is the registry
    whose version the event stream watches. Every handler hands its work to
    a thread: status() reads the task store and hold() takes the engine
    manager's lock, and neither belongs on the event loop."""

    async def sous_status(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return await _served("GET /sous/status", status)

    async def sous_events(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return EventSourceResponse(
            _status_events(status, inflight),
            ping=EVENT_PING_SECONDS,
            ping_message_factory=lambda: _PING,
            sep=_SEP,
        )

    async def sous_hold(request: Request) -> Response:
        try:
            check_loopback(request)
            pid, create_time = await _hold_body(request)
        except RequestError as e:
            return _refused(e)
        return await _served("POST /sous/hold", engines.hold, pid, create_time)

    async def sous_unknown(request: Request) -> Response:
        try:
            check_loopback(request)
        except RequestError as e:
            return _refused(e)
        return _refused(RequestError(404, "not_found_error", "no such route"))

    mcp.custom_route("/sous/status", methods=["GET"])(sous_status)
    mcp.custom_route("/sous/events", methods=["GET"])(sous_events)
    mcp.custom_route("/sous/hold", methods=["POST"])(sous_hold)
    # Registered after the real routes and before the gateway mounts its
    # catch-all (the SDK keeps registration order), so no path under /sous
    # can reach the upstream with the gateway on. The bare /sous needs its
    # own entry: `/sous/{path:path}` does not match it, and the gateway's
    # `/{path:path}` would.
    mcp.custom_route("/sous", methods=list(ALL_METHODS))(sous_unknown)
    mcp.custom_route("/sous/{path:path}", methods=list(ALL_METHODS))(sous_unknown)
```

(`EVENT_PING_SECONDS` and the heartbeat are read at call time, as the tests' monkeypatching needs; `_status_events` reads the module globals on every loop, which is what makes `EVENT_HEARTBEAT_SECONDS` patchable.)

In `src/sous/cli.py` line 317: `model_id = status.get("engine", {}).get("model_id", "the model")`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_server.py tests/test_monitor.py tests/test_cli.py tests/test_gateway_routes.py 2>&1 | tail -3`
Expected: all pass; the events tests take ~3 s together.

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1048 passed, 8 skipped, 29 deselected`; `uv run ty check`; ruff — clean.

- [ ] **Step 6: Commit**

```bash
git add src/sous/server.py src/sous/monitor.py src/sous/cli.py tests/test_server.py tests/test_monitor.py tests/test_cli.py
git commit -m "feat(server,monitor): serve one status document with the live turn and stream it on GET /sous/events" -m "The status document gains the registry: what the model is doing now (inflight) and, over HTTP, the last fifty turns and ten tasks; the engine block absorbs memory_gb under its own name. /sous/events streams that document — at once, then whenever the registry's version moved (polled ten times a second, so a burst is one event) and at least once a second so a load or a hold shows without a turn — with sse-starlette's ping as the keepalive. The MCP tool serves the same document without the recent lists. The launcher reads engine.model_id." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `sous statusline` (`cli.py`)

**Files:**
- Modify: `src/sous/cli.py` (`import contextlib` in the import block at lines 5–18; constants after `_PREDATES_MESSAGE` at line 59; new functions after `_cmd_status` at line 379; `main` at lines 767–819: a `statusline` subparser and dispatch)
- Test: `tests/test_cli.py` (append; `_FakeSousHTTP` at lines 1095–1154 is reused as is)

**Interfaces:**
- Consumes: the Task 3 document over `GET /sous/status`.
- Produces: `sous statusline`; `statusline_text(document: dict, now: float) -> str` (pure, tested directly); `_STATUSLINE_TIMEOUT_SECONDS = 0.5`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
# --- sous statusline --------------------------------------------------------------


def _document(**overrides) -> dict:
    doc = {
        "engine": {"loaded": True, "loading": False, "holders": 0, "prompt_cache": {"slots": 5}},
        "inflight": [],
        "queue": {"queued": 0, "running": 0},
        "config": {},
        "recent_turns": [],
        "recent_tasks": [],
    }
    doc.update(overrides)
    return doc


def _turn(**overrides) -> dict:
    turn = {
        "id": "msg_1",
        "phase": "decode",
        "started_at": 1_000.0,
        "phase_since": 1_010.0,
        "generated_tokens": 612,
        "to_prefill": None,
        "decode_tps": 14.7,
        "eta_seconds": 18.4,
    }
    turn.update(overrides)
    return turn


def test_statusline_text_for_each_state():
    from sous.cli import statusline_text

    now = 1_042.0
    assert statusline_text(_document(inflight=[_turn()]), now) == (
        "sous: decode 612 tok · 14.7 tok/s · eta 18s"
    )
    assert statusline_text(_document(inflight=[_turn(decode_tps=None, eta_seconds=None)]), now) == (
        "sous: decode 612 tok · 00:42"
    )
    assert statusline_text(
        _document(inflight=[_turn(phase="prefill", to_prefill=3120, eta_seconds=4.0)]), now
    ) == "sous: prefill 3,120 tok · eta 4s"
    assert statusline_text(_document(inflight=[_turn(phase="prefill")]), now) == (
        "sous: prefill · 00:42"
    )
    assert statusline_text(_document(inflight=[_turn(phase="loading")]), now) == (
        "sous: loading · 00:42"
    )
    queued = _document(inflight=[_turn(), _turn(id="msg_2", phase="queued")])
    assert statusline_text(queued, now) == (
        "sous: decode 612 tok · 14.7 tok/s · eta 18s · +1 queued"
    )
    assert statusline_text(_document(), now) == "sous: idle · 5 slots"
    assert statusline_text(_document(engine={"loaded": True, "holders": 2}), now) == (
        "sous: idle · held"
    )
    assert statusline_text(_document(engine={"loaded": False, "loading": True}), now) == (
        "sous: loading"
    )
    assert statusline_text(_document(engine={"loaded": False, "loading": False}), now) == (
        "sous: idle · model unloaded"
    )
    assert statusline_text({}, now) == "sous: idle · model unloaded"


def _statusline_config(tmp_path, monkeypatch, port: int):
    from sous import cli
    from sous.config import SousConfig

    cfg = SousConfig(server_port=port, data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)


def test_statusline_reads_the_daemon_and_prints_one_line(tmp_path, capsys, monkeypatch):
    import io

    from sous import cli

    fake = _FakeSousHTTP(_document(inflight=[_turn()]))
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        # Claude Code pipes its own JSON in; it is read and ignored.
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"model": {"id": "claude-opus-5"}}'))
        cli.main(["statusline"])
    finally:
        fake.close()
    out = capsys.readouterr().out
    assert out == "sous: decode 612 tok · 14.7 tok/s · eta 18s\n"


def test_statusline_says_daemon_down_and_exits_zero(tmp_path, capsys, monkeypatch):
    from sous import cli

    _statusline_config(tmp_path, monkeypatch, _free_cli_port())
    cli.main(["statusline"])
    assert capsys.readouterr().out == "sous: daemon down\n"


def test_statusline_treats_a_wrong_answer_as_daemon_down(tmp_path, capsys, monkeypatch):
    from sous import cli

    fake = _FakeSousHTTP(None, raw=b"<html>not sous</html>")
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        cli.main(["statusline"])
    finally:
        fake.close()
    assert capsys.readouterr().out == "sous: daemon down\n"


def test_statusline_ignores_a_configured_proxy(tmp_path, capsys, monkeypatch):
    from sous import cli

    fake = _FakeSousHTTP(_document())
    try:
        _statusline_config(tmp_path, monkeypatch, fake.port)
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        cli.main(["statusline"])
    finally:
        fake.close()
    assert capsys.readouterr().out == "sous: idle · 5 slots\n"


def test_statusline_imports_nothing_heavy(tmp_path):
    """It runs once a second under Claude Code: the stdlib only. Checked in
    a fresh interpreter, since this test process has long since imported
    httpx for the launcher's tests."""
    fake = _FakeSousHTTP(_document())
    try:
        probe = (
            "import sys\n"
            "from pathlib import Path\n"
            "from sous import cli\n"
            "from sous.config import SousConfig\n"
            f"cfg = SousConfig(server_port={fake.port}, data_dir=Path(sys.argv[1]), "
            "config_path=Path(sys.argv[1]) / 'c.toml')\n"
            "cli.load_config = lambda: cfg\n"
            "cli.main(['statusline'])\n"
            "print(sorted(m for m in ('textual', 'httpx', 'mlx', 'psutil') if m in sys.modules))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    finally:
        fake.close()
    assert done.returncode == 0, done.stderr
    assert done.stdout == "sous: idle · 5 slots\n[]\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k statusline 2>&1 | tail -3`
Expected: FAIL — `ImportError: cannot import name 'statusline_text'`, and `main(["statusline"])` exits 2 with argparse's "invalid choice".

- [ ] **Step 3: Write the subcommand**

In `src/sous/cli.py`, add `import contextlib` between `import argparse` and `import fcntl`, and after `_PREDATES_MESSAGE` add:

```python
# Claude Code runs the status-line command about once a second and shows
# whatever it prints; a slow answer is a frozen line, so the whole call is
# bounded well inside that second.
_STATUSLINE_TIMEOUT_SECONDS = 0.5
```

After `_cmd_status` add:

```python
def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def statusline_text(document: dict, now: float) -> str:
    """One line for Claude Code's status bar from the status document: the
    turn in flight if there is one, else the engine. Every number is the
    daemon's; `now` is only for the elapsed time of a turn."""
    engine = document.get("engine") or {}
    inflight = document.get("inflight") or []
    if not inflight:
        if not engine.get("loaded"):
            return "sous: loading" if engine.get("loading") else "sous: idle · model unloaded"
        parts = ["sous: idle"]
        slots = (engine.get("prompt_cache") or {}).get("slots")
        if slots is not None:
            parts.append(f"{slots} slots")
        if engine.get("holders"):
            parts.append("held")
        return " · ".join(parts)
    turn = inflight[0]
    phase = turn.get("phase", "queued")
    head = f"sous: {phase}"
    parts: list[str] = []
    if phase == "decode":
        head = f"{head} {turn.get('generated_tokens', 0):,} tok"
        if turn.get("decode_tps"):
            parts.append(f"{turn['decode_tps']:.1f} tok/s")
    elif phase == "prefill" and turn.get("to_prefill"):
        head = f"{head} {turn['to_prefill']:,} tok"
    eta = turn.get("eta_seconds")
    if eta is not None and (phase == "decode" or (phase == "prefill" and turn.get("to_prefill"))):
        parts.append(f"eta {int(eta)}s")
    elif turn.get("started_at") is not None:
        parts.append(_clock(now - turn["started_at"]))
    if len(inflight) > 1:
        parts.append(f"+{len(inflight) - 1} queued")
    return " · ".join([head, *parts])


def _cmd_statusline() -> None:
    import urllib.request

    config = load_config()
    # Claude Code writes its own JSON (model, cwd, cost) to stdin; none of
    # it is needed here, but a pipe nobody reads can block the writer. A
    # stdin that cannot be read (a closed handle, pytest's capture) is as
    # harmless as an empty one.
    with contextlib.suppress(OSError):
        if not sys.stdin.isatty():
            sys.stdin.read()
    # No proxy, whatever the environment says: 127.0.0.1 is loopback, the
    # same rule as the launcher's httpx client and the gateway's forwarder.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{config.server_port}/sous/status"
    try:
        with opener.open(url, timeout=_STATUSLINE_TIMEOUT_SECONDS) as reply:
            document = json.loads(reply.read(_REPLY_LIMIT))
    except OSError, ValueError:
        # urllib's URLError and HTTPError are OSErrors; a body that is not
        # JSON, or is JSON but not an object, is not the daemon's.
        print("sous: daemon down")
        return
    if not isinstance(document, dict):
        print("sous: daemon down")
        return
    print(statusline_text(document, time.time()))
```

In `main`: after `sub.add_parser("status", …)` add

```python
    sub.add_parser(
        "statusline",
        help="one line for Claude Code's statusLine setting (reads and ignores its stdin JSON)",
    )
```

and in the dispatch chain, after the `status` branch:

```python
    elif args.command == "statusline":
        _cmd_statusline()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k statusline 2>&1 | tail -3`
Expected: `6 passed`.

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1054 passed, 8 skipped, 29 deselected`; `uv run ty check`; ruff — clean.

- [ ] **Step 5: Commit**

```bash
git add src/sous/cli.py tests/test_cli.py
git commit -m "feat(cli): add sous statusline for Claude Code's status bar" -m "One line from GET /sous/status — the turn in flight with its tokens, rate and ETA, else the engine's state — printed by a subcommand that imports nothing beyond the stdlib and gives up inside half a second, because Claude Code runs it once a second and shows whatever it gets. It reads and discards Claude Code's stdin JSON so the pipe never blocks, and ignores proxy variables for the loopback call like every other client in this package." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `sous top` — THE PASS: the terminal, its chef, its vocabulary and its feed (`tui.py`, `cli.py`, `pyproject.toml`)

**Files:**
- Modify: `pyproject.toml` (`dependencies`: `rich` after `"psutil>=7",`, `textual` after `"starlette>=1.6",`), `uv.lock` (by `uv add`)
- Create: `src/sous/tui.py`
- Modify: `src/sous/cli.py` (module docstring line 1; new `_cmd_top` after `_cmd_statusline`; `main`: `status` gains `--watch`, a `top` subparser, dispatch), `src/sous/server.py` (`status_document`'s `config` block gains `idle_unload_minutes` — see the note below)
- Test: `tests/test_tui.py` (new), `tests/test_cli.py` (append), `tests/test_server.py` (one assertion added to `test_server_status`)

**Interfaces:**
- Consumes: the Task 3 document over `GET /sous/events` and its `inflight` entries (Task 1's keys), `recent_turns` (Task 2's summary keys), `recent_tasks`, `config.idle_unload_minutes`.
- Produces: `sous.tui` — pure functions `clock_text`, `span_text`, `kilo_text`, `seconds_text`, `local_time`, `engine_word`, `stop_word`, `why_text(turn)`, `rail_row(turn)`, `hi_score(recent, live)`, `turn_view(turn, now, received, *, stalled=False) -> TurnView`, `sprite_frame(mood, now, *, motion=True)`, `sprite_text(mood, now, *, motion=True)`, `ticket_text(turn)`, `motifs()`; `FeedDown`, `sse_feed(port)`; widgets `Strip`, `Rule`, `Gauge`, `Chef`, `Slip`, `Stub`, `Card`, `LinePanel`, `RatePanel`, `Rail`, `Overlay`; the app `Top(feed, *, port, clock=time.time)` with `motion: bool`; `run_top(port) -> int`; the vocabulary, sprite and palette constants. `sous top` and `sous status --watch`. Task 6 pins the render.

**One addition to Task 3's document, made here:** the quiet card and THE LINE need `[model].idle_unload_minutes`, which the document's `config` block does not carry. Add `"idle_unload_minutes": self.config.idle_unload_minutes,` to the `config` dict in `SousService.status_document` (after `"model_id"`), and to `tests/test_server.py`'s `test_server_status` the assertion `assert s["config"]["idle_unload_minutes"] == 30`. Do this in Step 5 below, in the same commit.

**The look, which this task implements (the maintainer's brief: "a beautiful, whimsical fusion of chef-y motifs and early 90s nostalgia", and, after a first draft, "not a DOS console — visual motifs nostalgic of chef-y things, and early 90s era vibes"; direction B + A's slip and timer, approved 2026-09-14):** a restaurant's pass, drawn the way 1991 drew things. The turn in flight is an **ORDER SLIP** — a perforated border with `ORDER № msg_…` in its top edge and `tear here` in its bottom — carrying `PLATING  decode 412 tok`, a **kitchen-timer dial** `◴◵◶◷` turning beside the elapsed clock, the ETA, the SEAR and PLATE gauges, a neon graphic-EQ of tok/s, the walk-in's verdict (`REHEAT hit · reused 53,296 tok · 8,845 fresh`) and the order's numbers; a queued turn is a second, smaller slip beneath it. On **THE LINE** to its right stands a three-row pixel **chef** whose face is the state — `(>_<)` with a knife for KNIFE WORK, `(=_=)` over a flame for SEARING, `(^_^)` plating, `(-_o)` side-eye when the feed stalls and a `?` at ten seconds, `(-_-) zZ` when the kitchen is quiet, `(o_o)` while the stove fires up, `(x_x)` with a dropped toque when no daemon answers, VHS tracking bands when the feed is redialing — beside the engine, the prompt cache's counters and the ticket counts. A **Memphis title strip** frames it (`∿▲● sous top ●▲∿  THE PASS`, a run of squiggles, triangles and confetti, and the **HI-SCORE** — the best decode rate on record — parked top-right like an arcade attract screen), a squiggle rule under it, and a **gingham rule** `▞▚▞▚▞▚ RECENT ORDERS` over the book. With nothing on the pass the slip gives way to **THE PASS IS CLEAR**: the chef asleep beside a steaming stock pot, how long the rail has been empty, when the lights go out, the last order, the HI-SCORE and today's tally. The book is a plain, readable table with English column heads; the `why` column names how the ticket ended and what it cost (`ORDER UP · lcp 0 · new tools`, `PLATE FULL`, `DROPPED IT · 529 overloaded_error`). Two laws keep it legible: **whimsy decorates, numbers inform** — every number is the terminal's own foreground — and **a kitchen word never appears without its literal term or its number in the same cell**. The terminal's own background is never painted, and every accent is a mid-luminance colour (teal `#17A2A2`, hot pink `#D6488A`, purple `#8E68DE`, mustard `#A87A10`, chilli `#CE5450`, slate `#79798E`) that clears 3:1 on a black ground and on a white one — a test asserts it. The render at 100×30 mid-decode with a second order queued — the text of Task 6's snapshot, exactly the scenario its test scripts (five orders in the book, a HI-SCORE of 11.4 set during the run, clocks in UTC), which the README carries until the gate replaces it with a capture of the real daemon:

```text
∿▲● sous top ●▲∿  THE PASS   ∿   ▲    ●   ∿   ▲    ●   ∿   ▲    ●   ∿   ▲    HI-SCORE 11.4 tok/s ▲●∿
∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
┏╍ ORDER № msg_f4cfe7 ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓ ╭─ THE LINE ────────────────────────╮
╏ PLATING  decode 412 tok    ◶ 00:41.3            ETA 01:02  ╏ │   ▟███▙   LINE IS OPEN  loaded    │
╏                                                            ╏ │   (^_^)   Qwen3.8-27B-4bit        │
╏ SEAR  ▐█████████████████████▌  prefill 8,845 tok · done    ╏ │   ▐███▌═  31.4 GB · hold 1        │
╏ PLATE ▐████████▁▁▁▁▁▁▁▁▁▁▁▁▁▌  412 / 1,024 tok · 40%       ╏ │                                   │
╏                                                            ╏ │ PROMPT CACHE   6 slots · 11.8 GB  │
╏ tok/s ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▇█ 9.7 now                      ╏ │ REHEAT  41 hit    DRAWER 18 fork  │
╏ REHEAT hit · reused 53,296 tok · 8,845 fresh               ╏ │ SCRATCH  7 miss  TOSSED  3 evict  │
╏                                                            ╏ │ WALK-IN FULL 1 pressure evict     │
╏ in 62,141 tok   max 1,024   started 22:12:38   1st of 2    ╏ │ reused 1,820,000 tok              │
┗╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍ tear here ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┛ │                                   │
┏╍ ORDER № msg_64f2b3 ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓ │ TICKETS  ON RAIL 2 · COOKING 1    │
╏ ON THE RAIL  queued 00:07   next up · 8,912 tok in         ╏ │ SERVED 3 · BURNT 1 · 86'D 1       │
┗╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍ tear here ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┛ ╰───────────────────── idle 4m 12s ─╯
▞▚▞▚▞▚ RECENT ORDERS ▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞
 time      ticket      cache         in      out    plate   total   why
 22:12:20  msg_000000  REHEAT 61.9k  62,000  1,024  41.2s   44.1s   PLATE FULL
 22:11:20  msg_000000  DRAWER 53.3k  62,000  1,024  41.2s   44.1s   ORDER UP · WALK-IN FULL ×2
 22:10:20  msg_000000  SCRATCH 0     62,000  1,024  41.2s   44.1s   ORDER UP · lcp 0 · new tools
 22:09:20  msg_000000  DROPPED IT    —       —      —       44.1s   DROPPED IT · 529 overloaded_e
 22:08:20  msg_000000  DROPPED IT    —       —      —       44.1s   WALKED OUT · 499 abandoned




 TASK  COOKING 0:41  rename Delta callback across engine tests
 SEAR prefill  PLATE decode  REHEAT hit  DRAWER fork  SCRATCH miss  TOSSED evicted  86'D cancelled
 q quit   l legend   ? about   m motion   p pause   ↑↓ orders   enter ticket   r reconnect
```

At 80×24 the columns stack: the slip goes full width and loses its numbers line, the queued order becomes one perforated line (`┄┄ ON THE RAIL  queued 00:07   next up · 8,912 tok in ┄┄`), THE LINE folds its counters beside the chef in three rows, the book keeps three rows and drops `in`/`out`, and the delegated-task line stays (the slip gives up the spacer under its headline to pay for it). Below 60 columns only the footer renders (` PLATING 412/1,024 9.7t/s 00:41  q`). At 120 the book gains `took` and `sear`, the tasks line becomes two, and a **PLATE RATE** panel under THE LINE shows the live tok/s as three-row `Digits` with the 60 s floor and the HI-SCORE in its edge. Keys: `q`/`Esc`/`Ctrl-C` quit; `↑↓` scroll the book; `enter` opens the ticket under the cursor (every field of its turn line); `p` pauses the book; `l` the legend; `?` the About card ("brought to you by one Mac, one 27B chef, and no tokens billed to the dining room"); `m` motion off; `r` redials at once.

**Rulings this task makes on the design brief:** the chef's frames, the dial, the steam, the EQ's decoration and the VHS bands are all picked from the clock (`int(now × fps) % n`), never tweened, so a pinned clock is a pinned picture and the render snapshot is byte-stable; Textual's `dashed` border is heavy-dashed (`┏╍╍┓ ╏`) and is the perforation — there is no light-dashed border to draw; the book's `cache` cell is the kitchen word beside the tokens it reused (`REHEAT 61.9k`, `DRAWER 53.3k`, `SCRATCH 0`) and `took` keeps its slot identifier (`turn@61904`) at 120 columns, where there is room; the design's `t tasks` key is not built — the newest task rides on one line under the book (two at 120); `enter` (the ticket card) is built, since the book's row cannot carry queue, load, knife and first-token times; the redialing state replaces the slip with a slate **VHS TRACKING** card and dims the rest with `as of <time>` on THE LINE; a kitchen closed is the only state that changes the chef's shape (a dropped toque, a greyed apron), so it reads with colour stripped; decorative glyphs are measured with `rich.cells.cell_len` at start-up and swapped for ASCII if any is two cells wide, and none ever sits in a table cell.

- [ ] **Step 1: The two dependencies**

Run: `uv add "textual>=8.2,<9"` and `uv add "rich>=15"` — these edit `pyproject.toml` and `uv.lock`. Then place the two lines in `pyproject.toml`'s alphabetical `dependencies` with these comments: `rich` after `"psutil>=7",`:

```toml
    # `sous top` (src/sous/tui.py) imports rich.text.Text directly. Textual
    # pulls rich in, but a direct import must be a direct dependency — same
    # rule as the sse-starlette block below.
    "rich>=15",
```

and `textual` after `"starlette>=1.6",`:

```toml
    # `sous top` only (src/sous/tui.py), imported inside that command
    # function: the daemon, the launcher and the status line never load it.
    # Pinned below the next major: Textual's CSS and widget APIs move between
    # majors, and the render snapshot in tests/snapshots pins one rendering.
    "textual>=8.2,<9",
```

Run: `uv lock --check` → clean; `uv run python -c "import textual, rich; print(textual.__version__, rich.__version__)"` → `8.2.x 15.x`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tui.py`:

```python
"""`sous top`: the words and fractions a document becomes, the chef's
frames, the feed reader, and the app under Textual's pilot against an
in-memory feed."""

import asyncio
import time
from pathlib import Path

import pytest
from rich.text import Text

from sous import tui
from sous.tui import Top, turn_view

BASE = 1_700_000_000.0  # 2023-11-14 22:13:20 UTC


class Clock:
    def __init__(self, now: float = BASE) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _doc(turn=None, *, loaded=True, loading=False, recent=None, tasks=None, behind=()):
    return {
        "engine": {
            "loaded": loaded,
            "loading": loading,
            "model_id": "mlx-community/Qwen3.8-27B-4bit",
            "idle_seconds": 252.0,
            "holders": 1,
            "memory_gb": 31.4,
            "prompt_cache": {
                "slots": 6,
                "resident_bytes": int(11.8 * (1 << 30)),
                "hits": 41,
                "fork_hits": 18,
                "misses": 7,
                "forks": 12,
                "evictions": 3,
                "pressure_evictions": 1,
                "reused_tokens": 1_820_000,
            },
        },
        "inflight": ([turn] if turn else []) + list(behind),
        "queue": {"queued": 2, "running": 1},
        "config": {
            "model_id": "mlx-community/Qwen3.8-27B-4bit",
            "idle_unload_minutes": 30,
            "gateway": {"enabled": True, "max_context_tokens": 262144},
        },
        "recent_turns": recent or [],
        "recent_tasks": tasks or [],
    }


def _turn(**overrides) -> dict:
    turn = {
        "id": "msg_f4cfe7d6880edc5168039e50",
        "model": "sous-local",
        "stream": True,
        "started_at": BASE - 41.3,
        "phase": "decode",
        "phase_since": BASE - 37.1,
        "input_tokens": 62141,
        "max_tokens": 1024,
        "reused_tokens": 53296,
        "to_prefill": 8845,
        "generated_tokens": 412,
        "expected_output_tokens": 1024,
        "decode_tps": 9.7,
        "eta_seconds": 62.0,
    }
    turn.update(overrides)
    return turn


def _summary(n: int, cache: str = "hit", **overrides) -> dict:
    summary = {
        "ts": BASE - 60 * n,
        "id": f"msg_{n:024x}",
        "model": "sous-local",
        "stream": 1,
        "status": 200,
        "error": None,
        "stop_reason": "end_turn",
        "cache": cache,
        "took": "turn@61904" if cache == "hit" else "none",
        "input_tokens": 62000,
        "output_tokens": 1024,
        "reused_tokens": 61904 if cache == "hit" else 0,
        "prefilled_tokens": 96 if cache == "hit" else 62000,
        "lcp": None,
        "lcp_region": None,
        "bounds": None,
        "forks": 0,
        "evicted": 0,
        "pressure": 0,
        "load_s": 0.0,
        "queue_s": 0.0,
        "engine_wait_s": 0.0,
        "tokenize_s": 0.2,
        "ttft_s": 0.4,
        "prefill_s": 0.4,
        "decode_s": 41.2,
        "prefill_tps": 240.0,
        "decode_tps": 9.0 + n * 0.3,
        "seconds": 44.1,
        "tools_hash": "1b21cd75",
        "system_hash": "9f8e7d6c",
    }
    summary.update(overrides)
    return summary


RECENT = [
    _summary(1, stop_reason="max_tokens"),
    _summary(2, "fork", took="fork@53296", reused_tokens=53296, evicted=2, pressure=2),
    _summary(3, "miss", lcp=0, lcp_region="tools", bounds=[1, 2], load_s=12.8),
    _summary(4, error="overloaded_error", status=529),
    _summary(5, error="abandoned", status=499),
]
TASKS = [
    {
        "id": "a",
        "state": "running",
        "title": "rename Delta callback across engine tests",
        "seconds": 41,
    },
    {
        "id": "b",
        "state": "done",
        "title": "add a regression test for the pressure valve",
        "seconds": 72,
    },
]


# --- words, strings and fractions -----------------------------------------------------


def test_clock_span_and_number_text():
    assert tui.clock_text(41.34) == "00:41" and tui.clock_text(41.3, tenths=True) == "00:41.3"
    assert tui.task_time(794) == "13:14" and tui.task_time(3725) == "62:05"
    assert tui.clock_text(-3) == "00:00" and tui.clock_text(3725) == "62:05"
    assert tui.span_text(252) == "4m 12s" and tui.span_text(48) == "48s"
    assert tui.span_text(3 * 3600 + 5 * 60) == "3h 05m"
    assert tui.kilo_text(61904) == "61.9k" and tui.kilo_text(96) == "96"
    assert tui.kilo_text(None) == "—"
    assert tui.seconds_text(104.94) == "104.9s" and tui.seconds_text(None) == "—"


def test_every_kitchen_word_is_glued_to_its_literal_or_its_number():
    """The legibility guard: a cache word beside the tokens it reused, a
    stop word beside how the ticket ended, a phase word beside its phase."""
    row = tui.rail_row(_summary(1, "fork", reused_tokens=53296, took="fork@53296"))
    assert row["cache"] == "DRAWER 53.3k" and row["took"] == "fork@53296"
    assert tui.rail_row(_summary(1, "miss"))["cache"] == "SCRATCH 0"
    assert tui.rail_row(_summary(1))["cache"] == "REHEAT 61.9k"
    failed = tui.rail_row(_summary(1, error="api_error", status=500))
    assert failed["cache"] == "DROPPED IT" and failed["in"] == "—" and failed["took"] == "none"
    assert failed["why"] == "DROPPED IT · 500 api_error"
    assert tui.stop_word(_summary(1, error="abandoned", status=499)) == "WALKED OUT"
    assert tui.stop_word(_summary(1, stop_reason="max_tokens")) == "PLATE FULL"
    assert tui.stop_word(_summary(1, stop_reason="tool_use")) == "ORDER UP"
    for line in tui.LEGEND.splitlines():
        assert line.strip(), "the legend has no blank line to lose a row to"
    for word in ("SEAR", "PLATE", "REHEAT", "DRAWER", "SCRATCH"):
        assert word in tui.LEGEND_STRIP and word in tui.LEGEND_STRIP_COMPACT


def test_why_column_names_the_stop_then_the_cost():
    assert tui.why_text(_summary(1)) == "ORDER UP"
    assert tui.why_text(_summary(1, stop_reason="max_tokens")) == "PLATE FULL"
    miss = _summary(1, "miss", lcp=62851, lcp_region="system", bounds=[62298, 62856])
    assert tui.why_text(miss) == "ORDER UP · lcp 62.9k · header"
    assert tui.why_text(_summary(1, "miss", lcp=0, lcp_region="tools")) == (
        "ORDER UP · lcp 0 · new tools"
    )
    assert tui.why_text(_summary(1, "miss", lcp=0, lcp_region=None)) == (
        "ORDER UP · lcp 0 · nothing to reheat"
    )
    assert tui.why_text(_summary(1, evicted=2)) == "ORDER UP · TOSSED ×2"
    assert tui.why_text(_summary(1, evicted=2, pressure=1)) == "ORDER UP · WALK-IN FULL ×1"
    assert tui.why_text(_summary(1, load_s=12.8, queue_s=2.5)) == (
        "ORDER UP · PREHEATING 12.8s · waited 2.5s"
    )
    assert tui.why_text(_summary(1, load_s=0.4, queue_s=0.9)) == "ORDER UP"
    assert tui.why_text(_summary(1, error="overloaded_error", status=529)) == (
        "DROPPED IT · 529 overloaded_error"
    )


def test_turn_view_fractions_and_words():
    now, received = BASE + 1.0, BASE
    view = turn_view(_turn(), now, received)
    assert view.phase == "decode" and view.word == "PLATING"
    assert view.elapsed == pytest.approx(42.3) and view.in_phase == pytest.approx(38.1)
    assert view.prefill_fraction == 1.0
    assert view.decode_fraction == pytest.approx(412 / 1024)
    assert view.eta == 62.0 and view.tps == 9.7
    assert turn_view(_turn(), now, received, stalled=True).word == "IN THE WINDOW"
    # Prefill with a size and an ETA: the bar is elapsed over (elapsed at
    # the event + eta), capped, and keeps moving after the event.
    prefill = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=6.0)
    assert turn_view(prefill, received, received).prefill_fraction == pytest.approx(37.1 / 43.1)
    later = turn_view(prefill, received + 2.0, received)
    assert later.prefill_fraction == pytest.approx(39.1 / 43.1)
    assert turn_view(prefill, received + 60.0, received).prefill_fraction == tui.PREFILL_CAP
    undecided = _turn(phase="prefill", to_prefill=None, reused_tokens=None, eta_seconds=None)
    assert turn_view(undecided, now, received).prefill_fraction is None
    assert turn_view(_turn(expected_output_tokens=None), now, received).decode_fraction is None
    assert turn_view(_turn(phase="tokenizing"), now, received).prefill_fraction == 0.0
    huge = _turn(generated_tokens=5000, expected_output_tokens=1024)
    assert turn_view(huge, now, received).decode_fraction == tui.PREFILL_CAP


def test_the_hi_score_is_the_best_rate_on_record():
    assert tui.hi_score(RECENT, []) == (10.5, BASE - 300)  # summary 5: 9.0 + 5 * 0.3
    assert tui.hi_score(RECENT, [9.0, 14.7]) == (14.7, None)
    assert tui.hi_score([], []) == (0.0, None)


def test_the_chef_has_a_face_for_every_state_and_holds_still_without_motion():
    for mood, (fps, frames) in tui.SPRITES.items():
        for frame in frames:
            assert len(frame) == 3 and all(len(row) == 7 for row in frame), mood
        # A pinned clock is a pinned frame; motion off is frame zero.
        assert tui.sprite_frame(mood, 12.34) == tui.sprite_frame(mood, 12.34)
        assert tui.sprite_frame(mood, 12.34, motion=False) == frames[0]
        if fps and len(frames) > 1:
            seen = {tui.sprite_frame(mood, n / fps) for n in range(len(frames) * 2)}
            assert seen == set(frames), mood
    quiet = tui.sprite_text("quiet", 0.0)
    assert "(-_-)" in quiet.plain and quiet.plain.count("\n") == 2
    assert "(x_x)" in tui.sprite_text("closed", 0.0).plain
    # The toque and the face are the terminal's own ink, bold; the apron teal.
    spans = [(quiet.plain[s.start : s.end], str(s.style)) for s in quiet.spans]
    assert ("▟███▙", "bold") in spans and ("▐███▌", tui.TEAL) in spans


def test_the_ticket_card_carries_every_field_of_the_turn():
    text = tui.ticket_text(_summary(3, "miss", lcp=0, lcp_region="tools", load_s=12.8))
    assert "ticket" in text and _summary(3)["id"] in text
    assert "SCRATCH miss" in text and "0.0s / 12.8s" in text
    assert "ORDER UP · lcp 0 · new tools · PREHEATING 12.8s" in text


def test_the_palette_reads_on_a_black_and_a_white_ground():
    """Every accent clears 3:1 against #1e1e1e and against white: the
    terminal's ground is never painted, so both are real."""

    def luminance(hex_colour: str) -> float:
        channels = []
        for n in (1, 3, 5):
            c = int(hex_colour[n : n + 2], 16) / 255
            channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = channels
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(a: float, b: float) -> float:
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)

    for colour in (tui.TEAL, tui.PINK, tui.PURPLE, tui.MUSTARD, tui.CHILLI, tui.SLATE):
        lum = luminance(colour)
        assert contrast(lum, luminance("#1e1e1e")) >= 3.0, colour
        assert contrast(lum, luminance("#ffffff")) >= 3.0, colour
    source = Path(tui.__file__).read_text()
    assert "background:" not in source.replace("background: ansi_default", "")


# --- the app --------------------------------------------------------------------------


class Feed:
    """A scripted feed: documents put on the queue are delivered in order;
    None makes the stream drop (FeedDown) and the next factory call
    reconnects."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.connections = 0

    def __call__(self):
        self.connections += 1
        return self._stream()

    async def _stream(self):
        while True:
            document = await self.queue.get()
            if document is None:
                raise tui.FeedDown("ConnectError")
            yield "status", document


def _run(test, *, size=(100, 30), clock=None):
    """Run `test(app, pilot, feed, clock)` inside the pilot."""
    clock = clock or Clock()
    feed = Feed()

    async def go():
        app = Top(feed, port=8383, clock=clock)
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            await test(app, pilot, feed, clock)
        return app

    return asyncio.run(go())


async def _deliver(feed: Feed, pilot, document: dict) -> None:
    await feed.queue.put(document)
    await pilot.pause(0.15)


def _plain(app, selector: str) -> str:
    return app.query_one(selector).render().plain


def test_the_slip_shows_the_turn_from_the_document():
    async def test(app, pilot, feed, clock):
        behind = _turn(
            id="msg_64f2b36f3e4a528198cee4ea",
            phase="queued",
            started_at=BASE - 7,
            input_tokens=8912,
        )
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT, tasks=TASKS, behind=[behind]))
        slip = app.query_one(tui.Slip)
        assert slip.display and slip.border_title == "ORDER № msg_f4cfe7"
        assert slip.has_class("phase-decode") and slip.border_subtitle == "tear here"
        headline = _plain(app, "#headline")
        assert headline.startswith(" PLATING  decode 412 tok") and "00:41.3" in headline
        assert headline.rstrip().endswith("ETA 01:02")
        assert app.query_one("#plate", tui.Gauge).fraction == pytest.approx(412 / 1024)
        assert app.query_one("#sear", tui.Gauge).fraction == 1.0
        assert _plain(app, "#shelf") == " REHEAT hit · reused 53,296 tok · 8,845 fresh"
        assert _plain(app, "#order").endswith("1st of 2")
        stub = app.query_one(tui.Stub)
        assert stub.display and stub.border_title == "ORDER № msg_64f2b3"
        assert "ON THE RAIL  queued 00:07   next up · 8,912 tok in" in stub.render().plain
        assert not app.query_one(tui.Card).display
        line = _plain(app, "#line-engine")
        assert line.startswith("  LINE IS OPEN  loaded") and "hold 1" in line
        assert "REHEAT  41 hit    DRAWER 18 fork" in _plain(app, "#line-body")
        assert app.query_one("#line-chef", tui.Chef).mood == "decode"
        assert "HI-SCORE 10.5 tok/s" in app.query_one(tui.Strip).render().plain
        assert app.query_one(tui.Rail).row_count == 5
        assert _plain(app, "#tasks").startswith(" TASK  COOKING 0:41  rename Delta callback")
        assert _plain(app, "#footer") == tui.KEYS
        long_job = {
            "id": "c",
            "state": "running",
            "title": "the whole test suite",
            "seconds": 794,
        }
        await _deliver(feed, pilot, _doc(_turn(), tasks=[long_job]))
        assert _plain(app, "#tasks").startswith(" TASK  COOKING 13:14  the whole test suite")

    _run(test)


def test_prefill_shows_the_pulse_until_the_cache_decides():
    async def test(app, pilot, feed, clock):
        undecided = _turn(
            phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=None,
            to_prefill=None, reused_tokens=None, expected_output_tokens=None,
        )  # fmt: skip
        await _deliver(feed, pilot, _doc(undecided))
        assert app.query_one("#sear", tui.Gauge).fraction is None
        assert _plain(app, "#shelf") == " the walk-in is deciding"
        assert "ETA —" in _plain(app, "#headline")
        assert app.query_one(tui.Slip).has_class("phase-prefill")
        assert app.query_one("#line-chef", tui.Chef).mood == "prefill"
        decided = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=3.0)
        await _deliver(feed, pilot, _doc(decided))
        sear = app.query_one("#sear", tui.Gauge)
        assert sear.fraction == pytest.approx(37.1 / 40.1)
        assert sear.detail == "prefill 8,845 tok · ETA 00:03"

    _run(test)


def test_clocks_run_from_the_clock_and_a_silent_feed_reads_as_stalled():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        clock.now = BASE + 3.0
        await pilot.pause(0.25)  # the 10 fps ticker reads the clock
        slip = app.query_one(tui.Slip)
        assert "00:44.3" in _plain(app, "#headline") and not slip.has_class("stalled")
        assert app.query_one("#plate", tui.Gauge).fraction == pytest.approx(412 / 1024)
        clock.now = BASE + 6.0
        await pilot.pause(0.25)
        assert slip.has_class("stalled")
        assert _plain(app, "#headline").startswith(" IN THE WINDOW  stalled 00:06")
        assert slip.border_subtitle == "IN THE WINDOW · no event 6s"
        assert app.query_one("#line-chef", tui.Chef).mood == "stalled"
        clock.now = BASE + 12.0
        await pilot.pause(0.25)
        assert app.query_one("#line-chef", tui.Chef).mood == "stalled-long"

    _run(test)


def test_a_quiet_kitchen_shows_the_card_and_stops_the_ticker():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        card = app.query_one(tui.Card)
        assert card.display and not app.query_one(tui.Slip).display
        assert card.border_title == "THE PASS IS CLEAR" and card.has_class("card-quiet")
        body = _plain(app, "#card-body")
        assert "KITCHEN QUIET  no orders on the rail for 4m 12s" in body
        assert "LIGHTS OUT in 25m 48s   idle unload at 30m" in body
        assert "HI-SCORE 10.5 tok/s  best plate rate today, set" in body
        assert "3 SERVED · 1 SCRATCH miss · 1 BURNT · 1 WALKED OUT" in body
        assert "s o u s" not in body and "▒▒▒▒▒" in _plain(app, "#card-art")
        assert app.query_one("#card-chef", tui.Chef).mood in ("done", "quiet")
        assert "KITCHEN QUIET 4m 12s" in app.query_one(tui.Strip).render().plain
        assert app.query_one("#firing", tui.LoadingIndicator).auto_refresh is None
        # Idle, the 10 fps clock is paused: a moved clock changes no headline.
        clock.now = BASE + 30.0
        await pilot.pause(0.25)
        assert "no orders on the rail for 4m 12s" in _plain(app, "#card-body")
        await _deliver(feed, pilot, _doc(None, loaded=False, loading=True))
        firing = app.query_one("#firing", tui.LoadingIndicator)
        assert card.border_title == "FIRING UP" and firing.display
        assert firing.auto_refresh == pytest.approx(1 / 16)
        assert app.query_one("#card-chef", tui.Chef).mood == "loading"
        clock.now = BASE + 40.0
        await pilot.pause(0.25)
        assert card.border_subtitle == "10s loading"

    _run(test)


def test_a_dropped_feed_redials_and_a_daemon_that_never_answered_exits_one():
    async def test(app, pilot, feed, clock):
        card = app.query_one(tui.Card)
        assert card.border_title == "KITCHEN CLOSED" and card.has_class("card-closed")
        assert app.query_one("#card-chef", tui.Chef).mood == "closed"
        await feed.queue.put(None)
        await pilot.pause(0.15)
        assert card.border_subtitle.startswith("attempt 1 · next try in")
        await pilot.pause(0.6)  # the first retry: 0.5 s
        assert feed.connections == 2
        await _deliver(feed, pilot, _doc(_turn()))
        assert app.query_one(tui.Slip).display and not app.query_one("#right").has_class("stale")
        await feed.queue.put(None)
        await pilot.pause(0.15)
        assert card.display and card.border_title == "VHS TRACKING"
        assert app.query_one("#right").has_class("stale")
        assert app.query_one(tui.LinePanel).border_subtitle == f"as of {tui.local_time(BASE)}"
        assert "redialing the daemon · try 1" in _plain(app, "#card-lead")
        bands = _plain(app, "#card-chef")
        clock.now = BASE + 0.25  # two VHS frames on: the card keeps its clock
        await pilot.pause(0.6)
        assert _plain(app, "#card-chef") != bands
        await pilot.press("r")  # retry now, ahead of the backoff
        await pilot.pause(0.1)
        assert feed.connections == 3
        await pilot.press("q")

    app = _run(test)
    assert app.return_code == 0

    async def never(app, pilot, feed, clock):
        await feed.queue.put(None)
        await pilot.pause(0.15)
        await pilot.press("q")

    assert _run(never).return_code == 1


def test_resize_reflows_to_every_breakpoint():
    async def test(app, pilot, feed, clock):
        behind = _turn(id="msg_64f2b36f3e4a528198cee4ea", phase="queued", started_at=BASE - 7)
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT, tasks=TASKS, behind=[behind]))
        rail = app.query_one(tui.Rail)
        assert tuple(rail._widths) == tui.RAIL_COLUMNS and rail._widths["why"] == 29
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.15)
        assert app.screen.has_class("compact")
        assert not app.query_one("#line-body").display and not app.query_one("#order").display
        assert app.query_one(tui.Stub).size.height == 1
        assert app.query_one("#tasks").display and app.query_one(tui.Rail).size.height == 4
        assert tuple(rail._widths) == tui.RAIL_COLUMNS_COMPACT
        assert _plain(app, "#legend") == tui.LEGEND_STRIP_COMPACT
        assert _plain(app, "#footer") == tui.KEYS_COMPACT
        await pilot.resize_terminal(50, 20)
        await pilot.pause(0.15)
        assert app.screen.has_class("narrow")
        for selector in ("#strip", "#rule", "#main", "#gingham", "#rail", "#legend"):
            assert not app.query_one(selector).display, selector
        assert _plain(app, "#footer") == " PLATING 412/1,024 9.7t/s 00:41  q"
        await pilot.resize_terminal(120, 36)
        await pilot.pause(0.15)
        assert app.screen.has_class("wide") and not app.screen.has_class("compact")
        assert app.query_one(tui.RatePanel).display
        assert app.query_one("#digits", tui.Digits).value == "9.7"
        assert tuple(rail._widths) == tui.RAIL_COLUMNS_WIDE
        assert rail.get_row(RECENT[1]["id"])[3] == "fork@53296"

    _run(test)


def test_keys_pause_the_rail_open_the_overlays_and_quit():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT[:2]))
        rail = app.query_one(tui.Rail)
        await pilot.press("p")
        await pilot.pause()
        assert rail.paused and "paused" in app.query_one("#gingham").render().plain
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT))
        assert rail.row_count == 2
        await pilot.press("p")
        await pilot.pause()
        assert rail.row_count == 5 and "paused" not in app.query_one("#gingham").render().plain
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, tui.Overlay)
        assert (
            app.screen.query_one("#overlay").render().plain.startswith(tui.LEGEND.splitlines()[0])
        )
        await pilot.resize_terminal(80, 24)  # reflows the pass, not the overlay
        await pilot.pause(0.15)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, tui.Overlay)
        assert app.screen.has_class("compact") and not app.query_one("#line-body").display
        await pilot.resize_terminal(100, 30)
        await pilot.pause(0.15)
        await pilot.press("question_mark")
        await pilot.pause()
        assert "brought to you by" in app.screen.query_one("#overlay").render().plain
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, tui.Overlay)
        assert app.screen.query_one("#overlay").border_title == "ORDER № msg_000000"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert app.motion is False
        await pilot.press("ctrl+c")

    assert _run(test).return_code == 0

    async def escape(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        await pilot.press("escape")

    assert _run(escape).return_code == 0


def test_the_app_keeps_the_terminals_own_ground():
    async def test(app, pilot, feed, clock):
        assert app.ansi_color is True and app.theme == "ansi-dark"
        assert app.current_theme.background == "ansi_default"

    _run(test)


# --- the feed reader --------------------------------------------------------------------


def test_a_closed_port_is_feed_down_at_once():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    async def go():
        started = time.monotonic()
        with pytest.raises(tui.FeedDown) as exc:
            async for _ in tui.sse_feed(port):
                pass
        assert "ConnectError" in str(exc.value)
        assert time.monotonic() - started < 3.0

    asyncio.run(go())


# --- motion ---------------------------------------------------------------------------


def test_the_plate_gauge_tweens_to_the_fed_value_and_snaps_without_motion():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(generated_tokens=100)))
        plate = app.query_one("#plate", tui.Gauge)
        await pilot.wait_for_animation()
        assert plate.fraction == pytest.approx(100 / 1024)
        await feed.queue.put(_doc(_turn(generated_tokens=900)))
        await pilot.pause(0.03)
        mid = plate.fraction
        assert 100 / 1024 < mid < 900 / 1024, "the bar is between the two values mid-tween"
        await pilot.wait_for_animation()
        assert plate.fraction == pytest.approx(900 / 1024)
        app.motion = False
        await feed.queue.put(_doc(_turn(generated_tokens=200)))
        await pilot.pause(0.03)
        assert plate.fraction == pytest.approx(200 / 1024)

    _run(test)


def test_a_phase_change_ramps_through_the_settling_step():
    async def test(app, pilot, feed, clock):
        prefill = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=3.0)
        await _deliver(feed, pilot, _doc(prefill))
        slip = app.query_one(tui.Slip)
        assert slip.has_class("phase-prefill") and not slip.has_class("settling")
        await feed.queue.put(_doc(_turn()))
        await pilot.pause(0.03)
        assert slip.has_class("phase-decode") and slip.has_class("settling")
        await pilot.pause(0.2)
        assert not slip.has_class("settling")
        await _deliver(feed, pilot, _doc(_turn(generated_tokens=500)))
        assert not slip.has_class("settling")
        app.motion = False
        await feed.queue.put(_doc(prefill))
        await pilot.pause(0.03)
        assert slip.has_class("phase-prefill") and not slip.has_class("settling")

    _run(test)


def test_a_new_ticket_on_the_rail_is_highlighted_then_settles():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT[1:]))
        rail = app.query_one(tui.Rail)
        assert not isinstance(rail.get_row(RECENT[1]["id"])[0], Text)
        await feed.queue.put(_doc(_turn(), recent=RECENT))
        await pilot.pause(0.03)
        cell = rail.get_row(RECENT[0]["id"])[0]
        assert isinstance(cell, Text) and str(cell.style) == "reverse"
        assert not isinstance(rail.get_row(RECENT[1]["id"])[0], Text)
        await pilot.pause(0.12)
        assert str(rail.get_row(RECENT[0]["id"])[0].style) == "bold"
        await pilot.pause(0.15)
        assert not isinstance(rail.get_row(RECENT[0]["id"])[0], Text)
        assert str(rail.get_row(RECENT[0]["id"])[2].style) == tui.TEAL
        app.motion = False
        await _deliver(feed, pilot, _doc(_turn(), recent=[_summary(9), *RECENT]))
        assert not isinstance(rail.get_row(_summary(9)["id"])[0], Text)

    _run(test)


def test_a_miss_or_a_pressure_eviction_flashes_the_line_once():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        line = app.query_one(tui.LinePanel)
        assert not line.has_class("flash")
        bumped = _doc(_turn())
        bumped["engine"]["prompt_cache"]["misses"] += 1
        await feed.queue.put(bumped)
        await pilot.pause(0.03)
        assert line.has_class("flash") and line.styles.border_top[1].hex.upper() == tui.PINK
        await pilot.pause(0.35)
        assert not line.has_class("flash") and line.styles.border_top[1].hex.upper() == tui.TEAL
        app.motion = False
        quiet = _doc(_turn())
        quiet["engine"]["prompt_cache"]["pressure_evictions"] += 2
        await feed.queue.put(quiet)
        await pilot.pause(0.03)
        assert not line.has_class("flash")

    _run(test)
```

Append to `tests/test_cli.py`:

```python
# --- sous top ---------------------------------------------------------------------------


def test_top_and_status_watch_run_the_terminal(tmp_path, monkeypatch):
    from sous import cli, tui
    from sous.config import SousConfig

    cfg = SousConfig(server_port=8391, data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    ports: list[int] = []
    monkeypatch.setattr(tui, "run_top", lambda port: (ports.append(port), 3)[1])
    with pytest.raises(SystemExit) as exc:
        cli.main(["top"])
    assert exc.value.code == 3
    with pytest.raises(SystemExit) as exc:
        cli.main(["status", "--watch"])
    assert exc.value.code == 3
    assert ports == [8391, 8391]


def test_only_sous_top_imports_textual(tmp_path):
    """The daemon, the launcher's module and the status line load without
    the UI framework; a fresh interpreter proves it, since this process has
    imported it for the terminal's own tests."""
    probe = (
        "import sys\n"
        "import sous.cli, sous.server, sous.monitor, sous.inflight\n"
        "print('textual' in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "False\n"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tui.py tests/test_cli.py -k "top or textual" 2>&1 | tail -3`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'sous.tui'`.

- [ ] **Step 4: Write the terminal**

Create `src/sous/tui.py`:

```python
"""`sous top` — THE PASS: the daemon's live status as a terminal.

The one module in the tree that imports Textual, imported by the `sous top`
command function and nowhere else, so nothing the daemon runs loads it.
Everything on screen comes from the last status document `/sous/events`
delivered plus the clock: the functions in the first half turn a document
into words, strings and fractions and are tested without a terminal; the
widgets in the second half only display them.

The look: a restaurant's pass, drawn the way 1991 drew things. The turn in
flight is an order slip — a perforated border with ORDER № in its top edge
and "tear here" in its bottom — with the SEAR and PLATE gauges, a neon
graphic-EQ of tok/s and a kitchen-timer dial turning beside the clock. A
three-row pixel chef on THE LINE wears the engine's mood on its face.
Squiggles, triangles and confetti frame the title; gingham rules divide the
sections; the best rate today is the HI-SCORE. Two laws keep it legible:
whimsy decorates and numbers inform — every number is the terminal's own
foreground — and a kitchen word never appears without its literal term or
its number in the same cell. The terminal's own background is never
painted; every accent is a mid-luminance colour that reads on a black and a
white ground alike.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import localtime, strftime
from time import time as _wall_clock

import httpx
from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DataTable, Digits, LoadingIndicator, Sparkline, Static

Feed = Callable[[], AsyncIterator[tuple[str, dict]]]
Clock = Callable[[], float]

# --- vocabulary --------------------------------------------------------------------------
# The inline-gloss rule: a kitchen word shares its cell with the literal term
# or the number it stands for (`SEAR ▐███▌ prefill 4,296 tok`, `REHEAT 41 hit`).
# The legend strip above the footer and the `l` overlay carry the table.

PHASE_WORDS = {
    "queued": "ON THE RAIL",
    "loading": "PREHEATING",
    "tokenizing": "KNIFE WORK",
    "prefill": "SEARING",
    "decode": "PLATING",
}
DONE_WORD = "ORDER UP"
STALLED_WORD = "IN THE WINDOW"
CACHE_WORDS = {"hit": "REHEAT", "fork": "WARM DRAWER", "miss": "SCRATCH"}
RAIL_CACHE_WORDS = {"hit": "REHEAT", "fork": "DRAWER", "miss": "SCRATCH"}
STOP_WORDS = {
    "end_turn": "ORDER UP",
    "tool_use": "ORDER UP",
    "max_tokens": "PLATE FULL",
}
FAILED_WORD = "DROPPED IT"
ABANDONED_WORD = "WALKED OUT"
ENGINE_WORDS = {
    "unloaded": "LIGHTS OUT",
    "loading": "FIRING UP",
    "loaded": "LINE IS OPEN",
}
QUIET_WORD = "KITCHEN QUIET"
CLOSED_WORD = "KITCHEN CLOSED"
TASK_WORDS = {
    "queued": "ON RAIL",
    "running": "COOKING",
    "awaiting_approval": "NEEDS CHEF",
    "done": "SERVED",
    "failed": "BURNT",
    "cancelled": "86'D",
}
REGION_WORDS = {
    "tools": "new tools",
    "system": "header",
    "tools-or-system": "tools or header",
    "conversation": "conversation",
}
LEGEND_STRIP = (
    " SEAR prefill  PLATE decode  REHEAT hit  DRAWER fork  SCRATCH miss"
    "  TOSSED evicted  86'D cancelled"
)
LEGEND_STRIP_COMPACT = " SEAR prefill · PLATE decode · REHEAT hit · DRAWER fork · SCRATCH miss"
KEYS = " q quit   l legend   ? about   m motion   p pause   ↑↓ orders   enter ticket   r reconnect"
KEYS_COMPACT = " q quit  l legend  ? about  m motion  p pause  ↑↓ orders  enter ticket"
LEGEND = """\
 ON THE RAIL    queued       admitted, waiting for a free slot
 PREHEATING     loading      model weights loading (10–15 s cold)
 KNIFE WORK     tokenizing   the prompt being counted
 SEARING        prefill      one fast burst of heat: reading the prompt
 PLATING        decode       slow and steady: writing tokens out
 ORDER UP       done         the turn finished
 IN THE WINDOW  stalled      no word from the daemon for 5 s; the clock runs
 PLATE FULL     max_tokens   the turn hit its output cap
 WALKED OUT     abandoned    the client left while the turn was queued
 DROPPED IT     failed       the turn failed or was refused
 REHEAT         hit          the prompt's prefix reused from a resident shelf
 WARM DRAWER    fork         a copy of another conversation's prefix, held hot
 SCRATCH        miss         nothing reusable; the prompt read cold
 TOSSED         evicted      a shelf dropped to make room
 WALK-IN FULL   pressure     shelves dropped under memory pressure
 LIGHTS OUT / FIRING UP / LINE IS OPEN   no model / loading / model resident
 hold N         holders      live `sous claude` sessions pinning the model
 ON RAIL / COOKING / NEEDS CHEF / SERVED / BURNT / 86'D
                             a delegated task queued / running / awaiting
                             your approval / done / failed / cancelled
"""
ABOUT = """\

   ∿▲● sous top ●▲∿

   brought to you by one Mac, one 27B chef,
   and no tokens billed to the dining room.

   any key
"""

# Seconds without a status event before the turn is shown as stalled, and
# before the chef grows a question mark.
STALL_SECONDS = 5.0
STALL_QUESTION_SECONDS = 10.0
# The reconnect backoff: first wait, then doubled up to the cap.
RETRY_SECONDS = (0.5, 8.0)
# Samples on the tok/s sparkline, one a second.
SPARK_SAMPLES = 60
# Breakpoints: below the first only the footer shows; below the second the
# columns stack; at the third the big rate readout appears.
NARROW_COLUMNS = 60
COMPACT_COLUMNS = 100
WIDE_COLUMNS = 120
# The prefill bar never claims more than this before the phase changes.
PREFILL_CAP = 0.95
DIAL = "◴◵◶◷"

# --- palette (relative luminance 0.19–0.29: ≥ 3:1 on #1e1e1e and on white) ------------
TEAL = "#17A2A2"  # chrome, rules, the apron, LINE IS OPEN, REHEAT, SERVED
PINK = "#D6488A"  # the title word, confetti, PLATING, the EQ's top, HI-SCORE, a flash
PURPLE = "#8E68DE"  # the order slip, ORDER №, the dial, the PLATE fill, KNIFE WORK
MUSTARD = "#A87A10"  # the SEAR fill, SEARING, PREHEATING, PLATE FULL, the stall ?
CHILLI = "#CE5450"  # BURNT, DROPPED IT, WALK-IN FULL, KITCHEN CLOSED
SLATE = "#79798E"  # legend, perforation, IN THE WINDOW, LIGHTS OUT, 86'D, steam, VHS
PHASE_COLOURS = {
    "queued": SLATE,
    "loading": MUSTARD,
    "tokenizing": PURPLE,
    "prefill": MUSTARD,
    "decode": PINK,
}
CACHE_COLOURS = {"hit": TEAL, "fork": PURPLE, "miss": MUSTARD}

# Decorative glyphs are East-Asian Ambiguous width in some terminals; a
# start-up measurement swaps the whole set for ASCII rather than let one
# two-cell squiggle shift a column. None of them ever sits in a table cell.
MOTIFS = {"squiggle": "∿", "triangle": "▲", "dot": "●", "dial": DIAL, "gingham": "▞▚"}
MOTIFS_ASCII = {
    "squiggle": "~",
    "triangle": "^",
    "dot": "o",
    "dial": "|/-\\",
    "gingham": "/\\",
}


def motifs() -> dict[str, str]:
    return MOTIFS if all(cell_len(g) == len(g) for g in MOTIFS.values()) else MOTIFS_ASCII


# --- words, strings and fractions -----------------------------------------------------


def clock_text(seconds: float, *, tenths: bool = False) -> str:
    """`mm:ss`, or `mm:ss.t`; never negative, hours roll into minutes."""
    seconds = max(0.0, seconds)
    if tenths:
        # Rounded to the tenth first: 41.3 is 41.299… in binary, and a
        # truncation would print it as 41.2.
        whole, tenth = divmod(round(seconds * 10), 10)
        return f"{whole // 60:02d}:{whole % 60:02d}.{tenth}"
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def task_time(seconds: float) -> str:
    """`m:ss` for a delegated task, minutes unbounded (a task can run an hour)."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def span_text(seconds: float) -> str:
    """`4m 12s`, `1h 03m`, `48s`: the idle and unload spans."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def kilo_text(count: int | None) -> str:
    """`26.5k` for a column; `—` for none."""
    if count is None:
        return "—"
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def seconds_text(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.1f}s"


def local_time(epoch: float) -> str:
    return strftime("%H:%M:%S", localtime(epoch))


def engine_word(engine: dict) -> str:
    if engine.get("loaded"):
        return ENGINE_WORDS["loaded"]
    return ENGINE_WORDS["loading" if engine.get("loading") else "unloaded"]


def stop_word(turn: dict) -> str:
    if turn.get("error"):
        return ABANDONED_WORD if turn.get("error") == "abandoned" else FAILED_WORD
    return STOP_WORDS.get(turn.get("stop_reason") or "", DONE_WORD)


def why_text(turn: dict) -> str:
    """The book's last column: how the ticket ended, then what made it cost
    what it cost — the miss and where it diverged, the walk-in's evictions,
    a cold load, a queue wait."""
    parts = [stop_word(turn)]
    if turn.get("error"):
        parts.append(f"{turn.get('status', '-')} {turn['error']}")
        return " · ".join(parts)
    if turn.get("cache") == "miss":
        region = REGION_WORDS.get(turn.get("lcp_region") or "", "nothing to reheat")
        parts.append(f"lcp {kilo_text(turn.get('lcp') or 0)} · {region}")
    if turn.get("pressure"):
        parts.append(f"WALK-IN FULL ×{turn['pressure']}")
    elif turn.get("evicted"):
        parts.append(f"TOSSED ×{turn['evicted']}")
    if (turn.get("load_s") or 0.0) >= 1.0:
        parts.append(f"PREHEATING {turn['load_s']:.1f}s")
    if (turn.get("queue_s") or 0.0) >= 1.0:
        parts.append(f"waited {turn['queue_s']:.1f}s")
    return " · ".join(parts)


# Column widths; `why` takes what is left. Each column costs its width plus
# two cells of the table's padding.
RAIL_WIDTHS = {
    "time": 8, "ticket": 10, "cache": 12, "took": 10, "in": 6, "out": 5,
    "sear": 6, "plate": 6, "total": 6, "why": 22,
}  # fmt: skip
RAIL_COLUMNS = ("time", "ticket", "cache", "in", "out", "plate", "total", "why")
RAIL_COLUMNS_WIDE = (
    "time", "ticket", "cache", "took", "in", "out", "sear", "plate", "total", "why",
)  # fmt: skip
RAIL_COLUMNS_COMPACT = ("time", "ticket", "cache", "plate", "total", "why")


def rail_row(turn: dict) -> dict[str, str]:
    """One completed turn as the book's cells, keyed by column name. The
    cache cell is the kitchen word beside the tokens it reused."""
    failed = bool(turn.get("error"))
    cache = turn.get("cache") or ""
    return {
        "time": local_time(turn.get("ts") or 0.0),
        "ticket": (turn.get("id") or "")[:10],
        "cache": FAILED_WORD
        if failed
        else f"{RAIL_CACHE_WORDS.get(cache, '—')} {kilo_text(turn.get('reused_tokens') or 0)}",
        "took": "none" if failed else turn.get("took") or "none",
        "in": "—" if failed else f"{turn.get('input_tokens', 0):,}",
        "out": "—" if failed else f"{turn.get('output_tokens', 0):,}",
        "sear": "—" if failed else seconds_text(turn.get("prefill_s")),
        "plate": "—" if failed else seconds_text(turn.get("decode_s")),
        "total": seconds_text(turn.get("seconds")),
        "why": why_text(turn),
    }


def hi_score(recent: list[dict], live: list[float]) -> tuple[float, float | None]:
    """The best decode rate on record — the completed turns' and the live
    samples' — and the time of the turn that set it."""
    best, when = 0.0, None
    for turn in recent:
        rate = turn.get("decode_tps") or 0.0
        if rate > best:
            best, when = rate, turn.get("ts")
    if live and max(live) > best:
        best, when = max(live), None
    return best, when


@dataclass(frozen=True)
class TurnView:
    """What the slip shows for the turn in flight, computed once per event or
    tick from the entry, the clock, and when the entry arrived."""

    id: str
    model: str
    phase: str
    word: str
    started_at: float
    elapsed: float
    in_phase: float
    input_tokens: int
    max_tokens: int
    reused: int | None
    to_prefill: int | None
    generated: int
    expected: int | None
    tps: float | None
    eta: float | None
    # 0–1 for a bar, None for the indeterminate pulse.
    prefill_fraction: float | None
    decode_fraction: float | None


def turn_view(turn: dict, now: float, received: float, *, stalled: bool = False) -> TurnView:
    """`received` is when the entry's document arrived — the moment its ETA
    was true — so the prefill bar keeps moving between events."""
    phase = turn.get("phase") or "queued"
    started = turn.get("started_at") or now
    since = turn.get("phase_since") or started
    eta = turn.get("eta_seconds")
    to_prefill = turn.get("to_prefill")
    prefill_fraction: float | None
    if phase == "decode":
        prefill_fraction = 1.0
    elif phase != "prefill":
        prefill_fraction = 0.0
    elif to_prefill and eta is not None:
        total = max(0.1, (received - since) + eta)
        prefill_fraction = min(PREFILL_CAP, (now - since) / total)
    else:
        prefill_fraction = None
    expected = turn.get("expected_output_tokens")
    generated = turn.get("generated_tokens") or 0
    decode_fraction = min(PREFILL_CAP, generated / expected) if expected else None
    return TurnView(
        id=turn.get("id") or "",
        model=turn.get("model") or "-",
        phase=phase,
        word=STALLED_WORD if stalled else PHASE_WORDS.get(phase, phase.upper()),
        started_at=started,
        elapsed=now - started,
        in_phase=now - since,
        input_tokens=turn.get("input_tokens") or 0,
        max_tokens=turn.get("max_tokens") or 0,
        reused=turn.get("reused_tokens"),
        to_prefill=to_prefill,
        generated=generated,
        expected=expected,
        tps=turn.get("decode_tps"),
        eta=eta,
        prefill_fraction=prefill_fraction,
        decode_fraction=decode_fraction,
    )


# --- the chef ----------------------------------------------------------------------------
# Every frame is 3 rows × 7 columns: columns 1–5 the chef (toque, face,
# apron), columns 0 and 6 a prop. Faces are ASCII so no ambiguous-width glyph
# can shift one by a cell; the frame is picked from the clock, so a pinned
# clock is a pinned picture.

SPRITES: dict[str, tuple[float, tuple[tuple[str, str, str], ...]]] = {
    "queued": (
        2,
        ((" ▟███▙ ", " (._.) ", " ▐███▌≡"), (" ▟███▙ ", " (._.) ", " ▐███▌=")),
    ),
    "loading": (
        3,
        ((" ▟███▙ ", " (o_o) ", " ▐███▌."), (" ▟███▙ ", " (o_o) ", " ▐███▌^")),
    ),
    "tokenizing": (
        6,
        (("╱▟███▙ ", " (>_<) ", " ▐███▌╴"), (" ▟███▙ ", "╱(>_<) ", " ▐███▌╴")),
    ),
    "prefill": (
        6,
        (
            (" ▟███▙ ", " (=_=)^", " ▐███▌▄"),
            (" ▟███▙ ", " (=_=)'", " ▐███▌▄"),
            (" ▟███▙ ", " (=_=) ", " ▐███▌▄"),
        ),
    ),
    "decode": (
        4,
        ((" ▟███▙ ", " (^_^) ", " ▐███▌═"), (" ▟███▙ ", " (^o^) ", " ▐███▌-")),
    ),
    "stalled": (
        1,
        ((" ▟███▙ ", " (-_o) ", " ▐███▌ "), (" ▟███▙ ", " (o_-) ", " ▐███▌ ")),
    ),
    "stalled-long": (1, ((" ▟███▙?", " (o_-) ", " ▐███▌ "),)),
    "done": (6, ((" ▟███▙*", " (^0^) ", " ▐███▌="), (" ▟███▙ ", " (^0^) ", " ▐███▌="))),
    "quiet": (
        0.5,
        ((" ▟███▙z", " (-_-) ", " ▐███▌ "), (" ▟███▙Z", " (-_-) ", " ▐▄█▄▌ ")),
    ),
    "closed": (0, ((" ▙███▟ ", " (x_x) ", " ▐░░░▌ "),)),
    "vhs": (
        8,
        (
            (" ▟███▙ ", " (=_=) ", "░▒▓░▒▓░"),
            (" ▟███▙ ", "▒▓░▒▓░▒", " ▐███▌ "),
            ("▓░▒▓░▒▓", " (=_=) ", " ▐███▌ "),
        ),
    ),
}
PROP_COLOURS = {
    "╱": MUSTARD, "╴": MUSTARD, "^": MUSTARD, "'": MUSTARD, "▄": MUSTARD, "?": MUSTARD,
    "═": PURPLE, "-": PURPLE, "≡": PURPLE, "=": PURPLE, ".": PURPLE,
    "*": PINK, "!": PINK,
    "z": SLATE, "Z": SLATE,
}  # fmt: skip
STEAM = ("    ∿ ∿    ", "   ∿   ∿   ")
POT = ("  ╭─────╮  ", " ═┫▒▒▒▒▒┣═ ", "  ╰─────╯  ")


def sprite_frame(mood: str, now: float, *, motion: bool = True) -> tuple[str, str, str]:
    fps, frames = SPRITES[mood]
    index = int(now * fps) % len(frames) if motion and fps else 0
    return frames[index]


def sprite_text(mood: str, now: float, *, motion: bool = True) -> Text:
    """The frame as styled text: toque and face in the terminal's own ink,
    bold; the apron teal; the props their own colours; the noise bands and
    a closed kitchen's apron in slate."""
    rows = []
    for n, row in enumerate(sprite_frame(mood, now, motion=motion)):
        left, chef, right = row[0], row[1:6], row[6]
        if set(row) <= set("░▒▓ "):
            rows.append(Text(row, style=SLATE))
            continue
        chef_style = (
            f"bold {SLATE}" if mood == "closed" and n == 2 else (TEAL if n == 2 else "bold")
        )
        rows.append(
            Text.assemble(
                (left, PROP_COLOURS.get(left, SLATE)),
                (chef, chef_style),
                (right, PROP_COLOURS.get(right, SLATE)),
            )
        )
    return Text("\n").join(rows)


# --- the feed ---------------------------------------------------------------------------


class FeedDown(Exception):
    """The daemon did not answer, refused, or hung up."""


async def sse_feed(port: int) -> AsyncIterator[tuple[str, dict]]:
    """`GET /sous/events` as (event, data) pairs until the daemon hangs up.
    The read timeout sits well past the daemon's 10 s ping, so a silent
    connection is one the daemon has left, not one between events."""
    url = f"http://127.0.0.1:{port}/sous/events"
    timeout = httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0)
    try:
        # trust_env=False: 127.0.0.1 is loopback, and a proxy variable must
        # never route or see this call — the launcher's rule, kept here.
        async with (
            httpx.AsyncClient(timeout=timeout, trust_env=False) as client,
            client.stream("GET", url) as reply,
        ):
            if reply.status_code != 200:
                raise FeedDown(f"status {reply.status_code}")
            event: str | None = None
            data = ""
            async for line in reply.aiter_lines():
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: "):
                    data = line[len("data: ") :]
                elif line == "" and event is not None:
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        payload = {}
                    yield event, payload if isinstance(payload, dict) else {}
                    event, data = None, ""
    except httpx.HTTPError as exc:
        # The type only: an httpx message can carry the URL it was building.
        raise FeedDown(type(exc).__name__) from None
    raise FeedDown("stream ended")


# --- widgets ------------------------------------------------------------------------------


class Strip(Widget):
    """The title strip: the name in its shapes on the left, a repeating run
    of shapes across, and the kitchen's headline — the HI-SCORE while the
    line is open, its state otherwise — on the right."""

    DEFAULT_CSS = "Strip { height: 1; }"

    def __init__(self) -> None:
        super().__init__(id="strip")
        self._headline = ""
        self._headline_colour = TEAL

    def show(self, headline: str, colour: str) -> None:
        self._headline, self._headline_colour = headline, colour
        self.refresh()

    def render(self) -> Text:
        m = motifs()
        s, t, d = m["squiggle"], m["triangle"], m["dot"]
        left = Text.assemble(
            (f"{s}{t}{d} ", TEAL),
            ("sous top", f"bold {PINK}"),
            (f" {d}{t}{s}", TEAL),
            ("  THE PASS", "bold"),
        )
        right = Text.assemble(
            (self._headline, f"bold {self._headline_colour}"), (f" {t}{d}{s}", TEAL)
        )
        room = self.size.width - left.cell_len - right.cell_len - 2
        pattern = Text()
        cycle = ((f"   {s}", TEAL), (f"   {t}", PINK), (f"    {d}", MUSTARD))
        while room > 0:
            for glyphs, colour in cycle:
                if pattern.cell_len + len(glyphs) > room:
                    pattern.append(" " * (room - pattern.cell_len))
                    room = 0
                    break
                pattern.append(glyphs, colour)
            else:
                continue
            break
        return Text.assemble(left, pattern, "  ", right)


class Rule(Widget):
    """A squiggle rule, or a gingham rule with a section name in it."""

    DEFAULT_CSS = "Rule { height: 1; }"

    def __init__(self, title: str = "", *, id: str) -> None:
        super().__init__(id=id)
        self._title = title

    def render(self) -> Text:
        m = motifs()
        width = self.size.width
        if not self._title:
            return Text(m["squiggle"] * width, style=TEAL)
        pair = m["gingham"]
        head = pair * 3
        tail_len = max(0, width - len(head) - len(self._title) - 2)
        out = Text.assemble((head, TEAL), (f" {self._title} ", f"bold {PINK}"))
        for n in range((tail_len + 1) // 2):
            out.append(pair[: max(0, min(2, tail_len - 2 * n))], TEAL if n % 2 == 0 else PINK)
        return out


class Gauge(Widget):
    """`SEAR  ▐████▁▁▁▌  prefill 4,296 tok · 6.2s` on one line: the kitchen
    word, a capped track, then the literal and whatever else the caller says.
    A fraction of None is the indeterminate pulse: a short bright band
    sliding along the track, positioned from the clock so a frozen clock is a
    frozen picture."""

    DEFAULT_CSS = "Gauge { height: 1; }"
    # The detail column's width, so the two gauges and the tok/s row share
    # one x-axis whatever their text says; the app sets it per breakpoint.
    detail_width = 30
    fraction: reactive[float | None] = reactive(0.0)

    def __init__(self, word: str, colour: str, *, id: str) -> None:
        super().__init__(id=id)
        self._word = word
        self._colour = colour
        self.detail = ""
        self.phase_time = 0.0

    def show(
        self,
        fraction: float | None,
        detail: str,
        phase_time: float,
        *,
        ease: float = 0.0,
    ) -> None:
        """`ease` is a tween's duration in seconds; 0 snaps. Only a number
        tweens to a number — the pulse (None) never tweens."""
        self.detail, self.phase_time = detail, phase_time
        if ease and self.fraction is not None and fraction is not None:
            self.animate("fraction", fraction, duration=ease, easing="out_cubic")
        else:
            self.fraction = fraction
        self.refresh()

    def track_width(self) -> int:
        return max(8, self.size.width - 9 - self.detail_width)

    def render(self) -> Text:
        track = self.track_width()
        if self.fraction is None:
            band = 6
            span = max(1, track - band)
            step = int(self.phase_time * 12) % (2 * span)
            offset = step if step < span else 2 * span - step
            cells = Text.assemble(
                ("▁" * offset, SLATE),
                ("█" * band, self._colour),
                ("▁" * (track - offset - band), SLATE),
            )
        else:
            filled = int(round(self.fraction * track))
            cells = Text.assemble(("█" * filled, self._colour), ("▁" * (track - filled), SLATE))
        return Text.assemble(
            (f" {self._word:<6}", f"bold {self._colour}"),
            ("▐", SLATE),
            cells,
            ("▌", SLATE),
            f"  {self.detail[: self.detail_width - 2]}",
        )


class Chef(Widget):
    """The three-row pixel chef; its mood is the state it stands in."""

    DEFAULT_CSS = "Chef { width: 7; height: 3; }"

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.mood = "closed"
        self.now = 0.0
        self.motion = True

    def show(self, mood: str, now: float, *, motion: bool) -> None:
        self.mood, self.now, self.motion = mood, now, motion
        self.refresh()

    def render(self) -> Text:
        return sprite_text(self.mood, self.now, motion=self.motion)


class Slip(Vertical):
    """The order slip: the turn in flight on a perforated ticket."""

    def __init__(self) -> None:
        super().__init__(id="slip")
        self._phase = ""
        self._stats = ""

    def compose(self) -> ComposeResult:
        yield Static("", id="headline", markup=False)
        yield Static("", id="spacer1")
        yield Gauge("SEAR", MUSTARD, id="sear")
        yield Gauge("PLATE", PURPLE, id="plate")
        yield Static("", id="spacer2")
        with Horizontal(id="tps-row"):
            yield Static(" tok/s ", id="tps-label", markup=False)
            yield Sparkline([0.0], id="spark")
            yield Static("", id="tps", markup=False)
        yield Static("", id="shelf", markup=False)
        yield Static("", id="spacer3")
        yield Static("", id="order", markup=False)

    def _motion(self) -> bool:
        return getattr(self.app, "motion", True)

    def show(self, view: TurnView, behind: int, now: float, spark: list[float]) -> None:
        changed = view.phase != self._phase
        self._phase = view.phase
        for phase in PHASE_WORDS:
            self.set_class(phase == view.phase, f"phase-{phase}")
        if changed and self._motion():
            # Six accents cannot blend: the ramp is a slate step for a tenth
            # of a second, then the phase colour.
            self.add_class("settling")
            self.set_timer(0.1, lambda: self.remove_class("settling"))
        self.border_title = f"ORDER № {view.id[:10]}"
        sep = " · " if not self.has_class("compact") else " "
        sear = f"prefill {view.to_prefill:,} tok" if view.to_prefill else "prefill sizing…"
        if view.phase == "prefill" and view.eta is not None:
            sear += f"{sep}ETA {clock_text(view.eta)}"
        elif view.phase == "decode":
            sear += f"{sep}done"
        self.query_one("#sear", Gauge).show(view.prefill_fraction, sear, view.in_phase)
        plate = f"{view.generated:,}"
        if view.expected:
            plate += f" / {view.expected:,} tok{sep}{int(100 * view.generated / view.expected)}%"
        else:
            plate += " tok"
        self.query_one("#plate", Gauge).show(
            view.decode_fraction if view.phase == "decode" else 0.0,
            plate,
            view.in_phase,
            ease=0.15 if self._motion() else 0.0,
        )
        padded = [0.0] * max(0, SPARK_SAMPLES - len(spark)) + spark[-SPARK_SAMPLES:]
        self.query_one("#spark", Sparkline).data = padded
        self._stats = f" {view.tps:.1f} now" if view.tps is not None else " — now"
        self.query_one("#tps", Static).update(self._stats)
        if view.reused is None:
            shelf = "the walk-in is deciding"
        elif view.reused:
            shelf = f"REHEAT hit · reused {view.reused:,} tok · {view.to_prefill or 0:,} fresh"
        else:
            shelf = f"SCRATCH miss · nothing reused · {view.to_prefill or 0:,} fresh"
        self.query_one("#shelf", Static).update(f" {shelf}")
        order = (
            f" in {view.input_tokens:,} tok   max {view.max_tokens:,}   "
            f"started {local_time(view.started_at)}"
        )
        if behind:
            order += f"   1st of {behind + 1}"
        self.query_one("#order", Static).update(order)
        self.tick(view, stalled=False, silence=0.0)

    def tick(self, view: TurnView, *, stalled: bool, silence: float) -> None:
        """The clocks, the dial and the phase word: run from the clock, not
        the feed."""
        self.set_class(stalled, "stalled")
        motion = self._motion()
        dial = motifs()["dial"][int(view.elapsed * 2) % 4 if motion else 0]
        clock = f"{dial} {clock_text(view.elapsed, tenths=True)}"
        literal = view.phase
        if stalled:
            head = f"{STALLED_WORD}  stalled {clock_text(silence)}"
            self.border_subtitle = f"{STALLED_WORD} · no event {silence:.0f}s"
        else:
            head = f"{view.word}  {literal}"
            if view.phase == "decode":
                head += f" {view.generated:,} tok"
            self.border_subtitle = "tear here"
        eta = f"ETA {clock_text(view.eta)}" if view.eta is not None and not stalled else "ETA —"
        width = max(20, self.size.width - 2)
        gap = max(2, width - 1 - len(head) - len(clock) - len(eta) - 8)
        self.query_one("#headline", Static).update(
            f" {head}{' ' * (gap // 2)}{clock}{' ' * (gap - gap // 2 + 8)}{eta}"[:width]
        )
        sear = self.query_one("#sear", Gauge)
        sear.show(
            view.prefill_fraction,
            sear.detail,
            view.in_phase,
            ease=0.12 if motion else 0.0,
        )

    def reset(self) -> None:
        self._phase = ""
        for phase in PHASE_WORDS:
            self.remove_class(f"phase-{phase}")
        self.remove_class("stalled")


class Stub(Widget):
    """A queued turn behind the one on the slip: a slip of its own at full
    width, one perforated line when the columns stack."""

    def __init__(self) -> None:
        super().__init__(id="stub")
        self._turn: dict | None = None
        self._more = 0
        self._now = 0.0

    def show(self, turn: dict | None, more: int, now: float) -> None:
        self._turn, self._more, self._now = turn, more, now
        self.display = turn is not None
        if turn is not None:
            self.border_title = f"ORDER № {turn.get('id', '')[:10]}"
            self.border_subtitle = "tear here"
        self.refresh()

    def render(self) -> Text:
        if self._turn is None:
            return Text("")
        t = self._turn
        waited = clock_text(self._now - (t.get("started_at") or self._now))
        tokens = t.get("input_tokens") or 0
        line = f"{PHASE_WORDS['queued']}  queued {waited}   next up · {tokens:,} tok in"
        if self._more:
            line += f" · +{self._more} more"
        if self.has_class("compact"):
            pair = "┄┄"
            return Text.assemble((f" {pair} ", SLATE), (line, ""), (f" {pair}", SLATE))
        return Text.assemble(
            (" ", ""),
            (PHASE_WORDS["queued"], f"bold {SLATE}"),
            (line[len(PHASE_WORDS["queued"]) :], ""),
        )


class Card(Vertical):
    """The left column when there is no slip: the pass is clear, the stove
    is firing up, the feed is redialing, or the kitchen is closed."""

    def __init__(self) -> None:
        super().__init__(id="card")

    def compose(self) -> ComposeResult:
        with Horizontal(id="card-top"):
            yield Chef(id="card-chef")
            yield Static("", id="card-art", markup=False)
            yield Static("", id="card-lead", markup=False)
        yield LoadingIndicator(id="firing")
        yield Static("", id="card-body", markup=False)

    def show_quiet(
        self, engine: dict, config: dict, recent: list[dict], now: float, motion: bool
    ) -> None:
        self._state("quiet", TEAL)
        self.border_title = "THE PASS IS CLEAR"
        idle = engine.get("idle_seconds") or 0.0
        minutes = config.get("idle_unload_minutes")
        self.query_one("#card-chef", Chef).show("quiet", now, motion=motion)
        steam = STEAM[int(now * 2) % 2 if motion else 0]
        self.query_one("#card-art", Static).update("\n".join([steam, *POT]))
        self.query_one("#card-lead", Static).update("\n\n  stock on, nobody ordering")
        lines = ["", f" {QUIET_WORD}  no orders on the rail for {span_text(idle)}"]
        if minutes and minutes * 60 > idle:
            lines.append(
                f" {ENGINE_WORDS['unloaded']} in {span_text(minutes * 60 - idle)}   "
                f"idle unload at {minutes}m"
            )
        else:
            lines.append(f" {ENGINE_WORDS['unloaded']} · the next order preheats the model")
        if recent:
            last = recent[0]
            lines += [
                "",
                f" LAST ORDER  {last.get('id', '')[:10]}  {stop_word(last)}"
                f"            {seconds_text(last.get('seconds'))} total",
            ]
            best, when = hi_score(recent, [])
            if best:
                set_at = f", set {local_time(when)}" if when else ""
                lines.append(f" HI-SCORE {best:.1f} tok/s  best plate rate today{set_at}")
            served = sum(1 for t in recent if not t.get("error"))
            scratch = sum(1 for t in recent if t.get("cache") == "miss")
            burnt = sum(1 for t in recent if t.get("error") and t.get("error") != "abandoned")
            walked = sum(1 for t in recent if t.get("error") == "abandoned")
            lines += [
                "",
                f" {served} SERVED · {scratch} SCRATCH miss · {burnt} BURNT · {walked} WALKED OUT",
            ]
            oldest = min(t.get("ts") or now for t in recent)
            self.border_subtitle = f"since {local_time(oldest)[:5]} · {span_text(now - oldest)}"
        else:
            self.border_subtitle = ""
        self.query_one("#card-body", Static).update("\n".join(lines))

    def show_firing(
        self,
        model_id: str,
        memory: float | None,
        elapsed: float,
        now: float,
        motion: bool,
    ) -> None:
        self._state("firing", MUSTARD)
        self.border_title = ENGINE_WORDS["loading"]
        self.border_subtitle = f"{elapsed:.0f}s loading"
        self.query_one("#card-chef", Chef).show("loading", now, motion=motion)
        self.query_one("#card-art", Static).update("")
        memory_text = f" · {memory:.1f} GB to read" if memory else ""
        self.query_one("#card-lead", Static).update(
            f"\n  loading {model_id.rsplit('/', 1)[-1]}{memory_text}"
        )
        self.query_one("#card-body", Static).update("")

    def show_redial(
        self, attempt: int, wait: float, why: str, as_of: str, now: float, motion: bool
    ) -> None:
        self._state("vhs", SLATE)
        self.border_title = "VHS TRACKING"
        self.border_subtitle = f"as of {as_of}" if as_of else ""
        self.query_one("#card-chef", Chef).show("vhs", now, motion=motion)
        self.query_one("#card-art", Static).update("")
        self.query_one("#card-lead", Static).update(
            f"\n  redialing the daemon · try {attempt} · next in {wait:.0f}s\n  ({why})"
        )
        self.query_one("#card-body", Static).update("")

    def show_closed(self, port: int, attempt: int, wait: float, now: float) -> None:
        self._state("closed", CHILLI)
        self.border_title = CLOSED_WORD
        self.border_subtitle = f"attempt {attempt} · next try in {wait:.0f}s" if attempt else ""
        self.query_one("#card-chef", Chef).show("closed", now, motion=False)
        self.query_one("#card-art", Static).update("")
        self.query_one("#card-lead", Static).update(
            f"\n  {CLOSED_WORD} — no daemon on 127.0.0.1:{port}\n  start it with:  sous serve"
        )
        self.query_one("#card-body", Static).update(
            "\n  (or: sous install-launchd)   q leaves; the screen keeps trying meanwhile"
        )

    def _state(self, state: str, colour: str) -> None:
        for s in ("quiet", "firing", "vhs", "closed"):
            self.set_class(s == state, f"card-{s}")
        firing = self.query_one("#firing", LoadingIndicator)
        firing.display = state == "firing"
        # Its own mount starts a 16 Hz refresh; off the firing state that would
        # be the only timer running in a quiet kitchen.
        firing.auto_refresh = 1 / 16 if state == "firing" else None
        self.query_one("#card-art").display = state == "quiet"


class LinePanel(Vertical):
    """THE LINE: the chef, the engine, the walk-in's counters and the tickets."""

    def __init__(self) -> None:
        super().__init__(id="line")
        self.border_title = "THE LINE"

    def compose(self) -> ComposeResult:
        with Horizontal(id="line-top"):
            yield Chef(id="line-chef")
            yield Static("", id="line-engine", markup=False)
        yield Static("", id="line-body", markup=False)

    def show(
        self,
        mood: str,
        engine: dict,
        config: dict,
        queue: dict,
        recent: list[dict],
        now: float,
        motion: bool,
    ) -> None:
        self.query_one("#line-chef", Chef).show(mood, now, motion=motion)
        model = (config.get("model_id") or "-").rsplit("/", 1)[-1]
        memory = engine.get("memory_gb")
        holders = engine.get("holders") or 0
        c = engine.get("prompt_cache") or {}
        resident = (c.get("resident_bytes") or 0) / (1 << 30)
        served = sum(1 for t in recent if not t.get("error"))
        burnt = sum(1 for t in recent if t.get("error") and t.get("error") != "abandoned")
        cancelled = sum(1 for t in recent if t.get("error") == "abandoned")
        word = engine_word(engine)
        literal = (
            "loaded" if engine.get("loaded") else "loading" if engine.get("loading") else "unloaded"
        )
        gb = f"{memory:.1f} GB" if memory else "—"
        full = c.get("pressure_evictions", 0)
        compact = self.has_class("compact")
        if compact:
            self.query_one("#line-engine", Static).update(
                f"  {word}  {literal}   {model}   {gb}   hold {holders}\n"
                f"  REHEAT {c.get('hits', 0)} hit   DRAWER {c.get('fork_hits', 0)} fork   "
                f"SCRATCH {c.get('misses', 0)} miss\n"
                f"  TOSSED {c.get('evictions', 0)} evict  WALK-IN FULL {full}"
                f"   slots {c.get('slots', 0)} · {resident:.1f} GB"
            )
            self.query_one("#line-body", Static).update("")
        else:
            self.query_one("#line-engine", Static).update(
                f"  {word}  {literal}\n  {model}\n  {gb} · hold {holders}"
            )
            lines = [
                "",
                f" PROMPT CACHE   {c.get('slots', 0)} slots · {resident:.1f} GB",
                f" REHEAT {c.get('hits', 0):>3} hit    DRAWER {c.get('fork_hits', 0):>2} fork",
                f" SCRATCH {c.get('misses', 0):>2} miss  TOSSED {c.get('evictions', 0):>2} evict",
                f" WALK-IN FULL {c.get('pressure_evictions', 0)} pressure evict",
                f" reused {c.get('reused_tokens', 0):,} tok",
            ]
            # `size` is the content box: the sprite's three rows come off it.
            # The tickets block needs three more rows than the wide layout
            # leaves once the rate panel has taken its five.
            if self.size.height - 3 >= len(lines) + 3:
                lines += [
                    "",
                    f" TICKETS  ON RAIL {queue.get('queued', 0)}"
                    f" · COOKING {queue.get('running', 0)}",
                    f" SERVED {served} · BURNT {burnt} · 86'D {cancelled}",
                ]
            self.query_one("#line-body", Static).update("\n".join(lines))
        idle = engine.get("idle_seconds")
        tail = f"idle {span_text(idle)}" if idle is not None and engine.get("loaded") else literal
        if compact:
            tail = f"ON RAIL {queue.get('queued', 0)} · COOKING {queue.get('running', 0)} · {tail}"
        self.border_subtitle = tail


class RatePanel(Vertical):
    """PLATE RATE: the live tok/s as three-row digits, at 120 columns."""

    def __init__(self) -> None:
        super().__init__(id="rate")
        self.border_title = "PLATE RATE  tok/s"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Digits("", id="digits")
            yield Static("", id="rate-text", markup=False)

    def show(self, tps: float | None, floor: float | None, best: float, word: str) -> None:
        self.query_one("#digits", Digits).update(f"{tps:.1f}" if tps is not None else "")
        self.query_one("#rate-text", Static).update(
            f"tok/s now\nfloor {floor:.1f} · 60s\n{word}"
            if floor is not None
            else f"tok/s now\n\n{word}"
        )
        self.border_subtitle = f"HI-SCORE {best:.1f} tok/s" if best else ""


class Rail(DataTable):
    """RECENT ORDERS: the last fifty completed tickets, newest on top."""

    def __init__(self) -> None:
        super().__init__(cursor_type="row", id="rail")
        self._columns: tuple[str, ...] = ()
        self._widths: dict[str, int] = {}
        self._shown: list[str] = []
        self.turns: list[dict] = []
        self.paused = False
        self._pending: list[dict] | None = None

    def set_columns(self, columns: tuple[str, ...], screen_width: int) -> None:
        """The columns for a screen width; `why` takes what the others leave
        (the table's own padding costs two cells per column)."""
        widths = {name: RAIL_WIDTHS[name] for name in columns}
        used = sum(w + 2 for name, w in widths.items() if name != "why")
        widths["why"] = max(RAIL_WIDTHS["why"], screen_width - 2 - used - 2)
        if columns == self._columns and widths == self._widths:
            return
        self._columns, self._widths = columns, widths
        self.clear(columns=True)
        for name in columns:
            self.add_column(Text(name, style=SLATE), width=widths[name], key=name)
        self._shown = []

    def show(self, turns: list[dict]) -> list[str]:
        """Rebuild when the set of ids changed; returns the ids that are new."""
        if self.paused:
            self._pending = turns
            return []
        ids = [t.get("id") or "" for t in turns]
        if ids == self._shown:
            return []
        new = [i for i in ids if i not in self._shown]
        had_rows = bool(self._shown)
        self.clear()
        self.turns = turns
        for turn in turns:
            cells = rail_row(turn)
            row = []
            for c in self._columns:
                value = cells[c][: self._widths[c]]
                if c == "cache" and not turn.get("error"):
                    row.append(Text(value, style=CACHE_COLOURS.get(turn.get("cache") or "", "")))
                elif c == "cache":
                    row.append(Text(value, style=CHILLI))
                else:
                    row.append(value)
            self.add_row(*row, key=turn.get("id") or None)
        self._shown = ids
        if new and had_rows and getattr(self.app, "motion", True):
            self._highlight(new)
        return new

    def _highlight(self, ids: list[str]) -> None:
        """Three 100 ms steps — reverse, bold, then what the rebuild put
        there — on every cell of a row that just landed."""
        originals = {
            (row_key, name): self.get_cell(row_key, name)
            for row_key in ids
            if row_key in self._shown
            for name in self._columns
        }

        def restyle(style: str | None) -> None:
            for (row_key, name), original in originals.items():
                if row_key not in self._shown:
                    continue
                if style is None:
                    self.update_cell(row_key, name, original)
                else:
                    plain = original.plain if isinstance(original, Text) else original
                    self.update_cell(row_key, name, Text(plain, style=style))

        restyle("reverse")
        self.set_timer(0.1, lambda: restyle("bold"))
        self.set_timer(0.2, lambda: restyle(None))

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        if not self.paused and self._pending is not None:
            pending, self._pending = self._pending, None
            self.show(pending)
        return self.paused

    def selected(self, row_key: str | None = None) -> dict | None:
        """The ticket under the cursor, or the one a row selection named."""
        if row_key is not None:
            return next((t for t in self.turns if t.get("id") == row_key), None)
        if not self.turns or self.cursor_row is None:
            return None
        return self.turns[self.cursor_row] if self.cursor_row < len(self.turns) else None


class Overlay(ModalScreen[None]):
    """The legend, the About card, or a ticket: a box in the middle, any key
    closes it."""

    BINDINGS = [Binding("escape", "dismiss", show=False)]

    def __init__(self, text: str | Text, title: str = "") -> None:
        super().__init__()
        self._text = text
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._text, id="overlay", markup=False)

    def on_mount(self) -> None:
        self.query_one("#overlay").border_title = self._title

    def on_key(self, event: events.Key) -> None:
        event.stop()
        self.dismiss(None)


def ticket_text(turn: dict) -> str:
    """One completed ticket, every field of the turn line, for `enter`."""
    rows = [
        ("ticket", turn.get("id", "")),
        ("stop", stop_word(turn)),
        (
            "cache",
            f"{CACHE_WORDS.get(turn.get('cache') or '', '—')} {turn.get('cache') or ''}",
        ),
        ("took", turn.get("took") or "none"),
        (
            "in / out",
            f"{turn.get('input_tokens', 0):,} / {turn.get('output_tokens', 0):,} tok",
        ),
        (
            "reused / fresh",
            f"{turn.get('reused_tokens', 0):,} / {turn.get('prefilled_tokens', 0):,} tok",
        ),
        (
            "queue / load",
            f"{seconds_text(turn.get('queue_s'))} / {seconds_text(turn.get('load_s'))}",
        ),
        (
            "knife / sear / plate",
            " / ".join(seconds_text(turn.get(k)) for k in ("tokenize_s", "prefill_s", "decode_s")),
        ),
        ("first token", seconds_text(turn.get("ttft_s"))),
        ("total", seconds_text(turn.get("seconds"))),
        (
            "forks / tossed / pressure",
            f"{turn.get('forks', 0)} / {turn.get('evicted', 0)} / {turn.get('pressure', 0)}",
        ),
        ("why", why_text(turn)),
    ]
    return "\n".join(f" {k:<26} {v}" for k, v in rows)


class Top(App[int]):
    """`sous top`."""

    TITLE = "sous top"
    CSS = f"""
    Screen {{ background: ansi_default; color: ansi_default; layout: vertical; }}
    #main {{ height: 14; layout: horizontal; }}
    #left {{ width: 62; layout: vertical; margin-right: 1; }}
    #slip {{ height: 1fr; border: dashed {PURPLE}; border-title-color: {PURPLE};
             border-title-style: bold; border-subtitle-color: {PURPLE};
             border-subtitle-align: center; }}
    #slip.stalled {{ border: dashed {SLATE}; border-title-color: {SLATE};
                     border-subtitle-color: {SLATE}; }}
    #slip.settling {{ border: dashed {SLATE}; }}
    #slip #spacer1, #slip #spacer2, #slip #spacer3 {{ height: 1; }}
    #tps-row {{ height: 1; }}
    #tps-label {{ width: auto; color: {SLATE}; }}
    #spark {{ width: 1fr; height: 1; }}
    #spark > .sparkline--max-color {{ color: {PINK}; }}
    #spark > .sparkline--min-color {{ color: {TEAL}; }}
    #tps {{ width: 30; }}
    #shelf {{ color: {TEAL}; }}
    #order {{ color: ansi_default; }}
    #stub {{ height: 3; border: dashed {SLATE}; border-title-color: {SLATE};
             border-subtitle-color: {SLATE}; border-subtitle-align: center; }}
    #card {{ height: 1fr; border: round {TEAL}; border-title-style: bold;
             border-subtitle-align: right; }}
    #card.card-quiet {{ border: round {TEAL}; border-title-color: {TEAL}; }}
    #card.card-firing {{ border: round {MUSTARD}; border-title-color: {MUSTARD}; }}
    #card.card-vhs {{ border: round {SLATE}; border-title-color: {SLATE}; }}
    #card.card-closed {{ border: round {CHILLI}; border-title-color: {CHILLI}; }}
    #card-top {{ height: 4; }}
    #card-chef {{ margin: 0 1 0 2; }}
    #card-art {{ width: 11; color: {SLATE}; margin: 0 2; }}
    #card-lead {{ width: 1fr; }}
    #firing {{ height: 1; color: {MUSTARD}; }}
    #line {{ width: 1fr; border: round {TEAL}; border-title-color: {TEAL}; border-title-style: bold;
             border-subtitle-align: right; border-subtitle-color: {SLATE}; }}
    #line.flash {{ border: round {PINK}; border-title-color: {PINK};
                   border-subtitle-color: {PINK}; }}
    #line-top {{ height: 3; }}
    #line-chef {{ margin: 0 0 0 2; }}
    #line-engine {{ width: 1fr; }}
    #rate {{ display: none; height: 5; border: round {PINK}; border-title-color: {PINK};
             border-subtitle-align: right; border-subtitle-color: {PINK}; }}
    #digits {{ width: auto; color: {PINK}; margin: 0 2 0 2; }}
    #rate-text {{ width: 1fr; }}
    #rail {{ height: 1fr; overflow-x: hidden; scrollbar-size: 0 0; }}
    #tasks {{ height: 1; color: ansi_default; }}
    #legend {{ height: 1; color: {SLATE}; }}
    #footer {{ height: 1; }}
    .stale {{ text-style: dim; }}
    Screen.compact #main {{ layout: vertical; height: 14; }}
    Screen.compact #left {{ width: 100%; height: 9; }}
    Screen.compact #slip {{ height: 8; }}
    Screen.compact #slip #spacer1, Screen.compact #slip #spacer3, Screen.compact #order {{
        display: none; }}
    Screen.compact #stub {{ height: 1; border: none; }}
    Screen.compact #line {{ height: 5; }}
    Screen.compact #line-body {{ display: none; }}
    Screen.compact #tps {{ width: 12; }}
    Screen.narrow #strip, Screen.narrow #rule, Screen.narrow #main, Screen.narrow #gingham,
    Screen.narrow #rail, Screen.narrow #tasks, Screen.narrow #legend {{ display: none; }}
    Screen.wide #main {{ height: 16; }}
    Screen.wide #left {{ width: 78; }}
    Screen.wide #rate {{ display: block; }}
    Screen.wide #tasks {{ height: 3; }}
    Overlay {{ align: center middle; }}
    #overlay {{ width: 84; border: round {PINK}; border-title-color: {PINK}; padding: 0 2;
                background: ansi_default; }}
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", show=False, priority=True),
        Binding("q", "quit", "quit"),
        Binding("escape", "quit", show=False),
        Binding("p", "pause", "pause"),
        Binding("l", "legend", "legend"),
        Binding("m", "motion", "motion"),
        Binding("question_mark", "about", "about"),
        Binding("enter", "ticket", "ticket"),
        Binding("r", "reconnect", "reconnect"),
    ]

    def __init__(self, feed: Feed, *, port: int, clock: Clock = _wall_clock) -> None:
        super().__init__(ansi_color=True)
        # ansi_color alone leaves the default theme in place; the ANSI theme
        # is what makes foreground and background the terminal's own, and
        # hex accents still render as they are.
        self.theme = "ansi-dark"
        self._feed = feed
        self._port = port
        self._clock = clock
        self.motion = True
        self._document: dict = {}
        self._turn: dict | None = None
        self._behind: list[dict] = []
        self._received_at = 0.0
        self._connected = False
        self._ever_connected = False
        self._attempt = 0
        self._retry_at: float | None = None
        self._retry_now = asyncio.Event()
        self._why = ""
        self._firing_since: float | None = None
        self._done_until = 0.0
        self._spark: deque[float] = deque(maxlen=SPARK_SAMPLES)
        self._spark_at = float("-inf")
        self._counters: dict[str, int] = {}
        self._ticker: Timer | None = None
        self._idler: Timer | None = None
        self._base_screen: Screen | None = None
        self._width = 0

    # -- layout ---------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Strip()
        yield Rule(id="rule")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Slip()
                yield Stub()
                yield Card()
            with Vertical(id="right"):
                yield LinePanel()
                yield RatePanel()
        yield Rule("RECENT ORDERS", id="gingham")
        yield Rail()
        yield Static("", id="tasks", markup=False)
        yield Static(LEGEND_STRIP, id="legend", markup=False)
        yield Static(KEYS, id="footer", markup=False)

    def on_mount(self) -> None:
        self._base_screen = self.screen
        self._width = self.size.width
        self._breakpoints(self._width)
        self._show_left(None)
        self._ticker = self.set_interval(0.1, self._tick, pause=True)
        self._idler = self.set_interval(0.5, self._idle_tick, pause=True)
        self.query_one(Rail).focus()
        self.run_worker(self._pump(), exclusive=True)

    def on_resize(self, event: events.Resize) -> None:
        self._width = event.size.width
        self._breakpoints(self._width)
        if self._document:
            self._apply(self._document, resize=True)
        else:
            self._show_left(None)
            self._refresh_footer()

    def _breakpoints(self, width: int) -> None:
        screen = self._base_screen or self.screen  # not the overlay that may be up
        screen.set_class(width < NARROW_COLUMNS, "narrow")
        screen.set_class(width < COMPACT_COLUMNS, "compact")
        screen.set_class(width >= WIDE_COLUMNS, "wide")
        for widget in (
            self.query_one(Slip),
            self.query_one(Stub),
            self.query_one(LinePanel),
        ):
            widget.set_class(width < COMPACT_COLUMNS, "compact")
        for gauge in self.query(Gauge):
            gauge.detail_width = 30 if width >= COMPACT_COLUMNS else 26
        self.query_one("#legend", Static).update(
            LEGEND_STRIP if width >= COMPACT_COLUMNS else LEGEND_STRIP_COMPACT
        )
        self.query_one(Rail).set_columns(
            RAIL_COLUMNS_COMPACT
            if width < COMPACT_COLUMNS
            else RAIL_COLUMNS_WIDE
            if width >= WIDE_COLUMNS
            else RAIL_COLUMNS,
            width,
        )

    # -- the feed -------------------------------------------------------------------

    async def _pump(self) -> None:
        """Read the feed for as long as the app runs. A drop is shown and
        retried with backoff (or at once on `r`) — never a fallback to
        polling: same daemon, same port, and a daemon that is gone answers
        neither."""
        delay = RETRY_SECONDS[0]
        while True:
            try:
                async for event, payload in self._feed():
                    if event == "status":
                        self._apply(payload)
                        delay = RETRY_SECONDS[0]
            except FeedDown as exc:
                self._dropped(str(exc))
            self._attempt += 1
            self._retry_at = self._clock() + delay
            self._show_left(None)
            self._refresh_footer()
            self._retry_now.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._retry_now.wait(), delay)
            delay = min(delay * 2, RETRY_SECONDS[1])

    def _dropped(self, why: str) -> None:
        self._connected = False
        self._why = why
        self._turn = None
        if self._ticker is not None:
            self._ticker.pause()
        if self._idler is not None:
            self._idler.resume()  # the redial countdown and the tracking bands
        for selector in ("#right", "#rail", "#tasks"):
            self.query_one(selector).add_class("stale")
        self.query_one(LinePanel).border_subtitle = (
            f"as of {local_time(self._received_at)}" if self._ever_connected else ""
        )
        self._show_left(None)
        self._refresh_footer()

    # -- the document ---------------------------------------------------------------

    def _apply(self, document: dict, *, resize: bool = False) -> None:
        now = self._clock()
        if not resize:
            self._document = document
            self._received_at = now
            self._connected = self._ever_connected = True
            self._attempt = 0
            self._retry_at = None
            for selector in ("#right", "#rail", "#tasks"):
                self.query_one(selector).remove_class("stale")
        engine = document.get("engine") or {}
        config = document.get("config") or {}
        inflight = document.get("inflight") or []
        recent = document.get("recent_turns") or []
        queue = document.get("queue") or {}
        previous = self._turn
        self._turn = inflight[0] if inflight else None
        self._behind = inflight[1:]
        if not resize and previous is not None and self._turn is None and self.motion:
            # The ticket just left the pass: the chef's flourish, for a beat.
            self._done_until = now + 1.2
        if self._turn is not None:
            view = turn_view(self._turn, now, self._received_at)
            if not resize and view.phase == "decode" and now - self._spark_at >= 1.0:
                self._spark.append(view.tps or 0.0)
                self._spark_at = now
            self._show_left(view)
        else:
            self._show_left(None)
        if not engine.get("loading"):
            self._firing_since = None
        mood = self._mood(engine, now)
        self.query_one(LinePanel).show(mood, engine, config, queue, recent, now, self.motion)
        best, _ = hi_score(recent, list(self._spark))
        live = [r for r in self._spark if r]
        self.query_one(RatePanel).show(
            self._turn and turn_view(self._turn, now, self._received_at).tps,
            min(live) if live else None,
            best,
            f"{PHASE_WORDS.get(self._turn.get('phase', ''), '')}  {self._turn.get('phase', '')}"
            if self._turn
            else QUIET_WORD,
        )
        headline, colour = self._headline(engine, recent, now)
        self.query_one(Strip).show(headline, colour)
        self.query_one(Rail).show(recent)
        self._show_tasks(document.get("recent_tasks") or [])
        cache = engine.get("prompt_cache") or {}
        if not resize:
            rose = {
                key: cache.get(key, 0) > self._counters.get(key, cache.get(key, 0))
                for key in ("misses", "pressure_evictions")
            }
            self._counters = {k: cache.get(k, 0) for k in ("misses", "pressure_evictions")}
            if self.motion and (rose["misses"] or rose["pressure_evictions"]):
                self.query_one(LinePanel).add_class("flash")
                self.set_timer(0.25, lambda: self.query_one(LinePanel).remove_class("flash"))
        self._refresh_footer()
        busy = self._turn is not None or bool(engine.get("loading"))
        if self._ticker is not None:
            self._ticker.resume() if busy else self._ticker.pause()
        if self._idler is not None:
            self._idler.resume() if (not busy and self.motion) else self._idler.pause()

    def _mood(self, engine: dict, now: float) -> str:
        if not self._connected:
            return "vhs" if self._ever_connected else "closed"
        if self._turn is not None:
            silence = now - self._received_at
            if silence > STALL_QUESTION_SECONDS:
                return "stalled-long"
            if silence > STALL_SECONDS:
                return "stalled"
            return self._turn.get("phase") or "queued"
        if now < self._done_until:
            return "done"
        if engine.get("loading"):
            return "loading"
        # An unloaded engine on a live daemon is still a quiet kitchen (the
        # headline says LIGHTS OUT); the closed face is for a daemon that is gone.
        return "quiet"

    def _headline(self, engine: dict, recent: list[dict], now: float) -> tuple[str, str]:
        if not self._connected:
            return (CLOSED_WORD, CHILLI) if not self._ever_connected else ("REDIALING...", SLATE)
        if self._turn is None and engine.get("loading"):
            return ENGINE_WORDS["loading"], MUSTARD
        if self._turn is None and engine.get("loaded"):
            return f"{QUIET_WORD} {span_text(engine.get('idle_seconds') or 0)}", TEAL
        if self._turn is None:
            return ENGINE_WORDS["unloaded"], SLATE
        best, _ = hi_score(recent, list(self._spark))
        return (f"HI-SCORE {best:.1f} tok/s", PINK) if best else ("first order", PINK)

    def _show_left(self, view: TurnView | None) -> None:
        """The slip and the stub when a turn is on the pass; the card when
        not — quiet, firing up, redialing or closed."""
        now = self._clock()
        slip, stub, card = (
            self.query_one(Slip),
            self.query_one(Stub),
            self.query_one(Card),
        )
        engine = self._document.get("engine") or {}
        config = self._document.get("config") or {}
        recent = self._document.get("recent_turns") or []
        if view is not None and self._connected:
            slip.display, card.display = True, False
            slip.show(view, len(self._behind), now, list(self._spark))
            stub.show(self._behind[0] if self._behind else None, len(self._behind) - 1, now)
            return
        slip.display, card.display = False, True
        stub.show(None, 0, now)
        slip.reset()
        wait = max(0.0, (self._retry_at or now) - now)
        if not self._connected and not self._ever_connected:
            card.show_closed(self._port, self._attempt, wait, now)
        elif not self._connected:
            card.show_redial(
                self._attempt,
                wait,
                self._why,
                local_time(self._received_at),
                now,
                self.motion,
            )
        elif engine.get("loading"):
            if self._firing_since is None:
                self._firing_since = now
            card.show_firing(
                config.get("model_id", "-"),
                engine.get("memory_gb"),
                now - self._firing_since,
                now,
                self.motion,
            )
        else:
            card.show_quiet(engine, config, recent, now, self.motion)

    def _show_tasks(self, tasks: list[dict]) -> None:
        rows = []
        for t in tasks[: 2 if self._width >= WIDE_COLUMNS else 1]:
            word = TASK_WORDS.get(t.get("state", ""), "?")
            seconds = t.get("seconds") or 0
            rows.append(
                f" TASK  {word} {task_time(seconds) if seconds else '—'}  "
                f"{(t.get('title') or '')[:70]}"
            )
        self.query_one("#tasks", Static).update(
            "\n".join(rows) if rows else " TASK  none on the rail"
        )

    def _refresh_footer(self) -> None:
        footer = self.query_one("#footer", Static)
        width = self._width
        now = self._clock()
        if width < NARROW_COLUMNS:
            if self._turn is not None:
                view = turn_view(self._turn, now, self._received_at)
                gen = f"{view.generated:,}" + (f"/{view.expected:,}" if view.expected else "")
                rate = f" {view.tps:.1f}t/s" if view.tps else ""
                footer.update(f" {view.word} {gen}{rate} {clock_text(view.elapsed)}  q")
            elif self._connected:
                engine = self._document.get("engine") or {}
                footer.update(f" {QUIET_WORD} {span_text(engine.get('idle_seconds') or 0)}  q")
            else:
                footer.update(f" {CLOSED_WORD}  q")
            return
        footer.update(KEYS if width >= COMPACT_COLUMNS else KEYS_COMPACT)

    # -- timers ---------------------------------------------------------------------

    def _tick(self) -> None:
        now = self._clock()
        engine = self._document.get("engine") or {}
        config = self._document.get("config") or {}
        try:
            if self._turn is not None:
                silence = now - self._received_at
                stalled = silence > STALL_SECONDS
                view = turn_view(self._turn, now, self._received_at, stalled=stalled)
                self.query_one(Slip).tick(view, stalled=stalled, silence=silence)
                self.query_one("#line-chef", Chef).show(
                    self._mood(engine, now), now, motion=self.motion
                )
            elif engine.get("loading") and self._firing_since is not None:
                self.query_one(Card).show_firing(
                    config.get("model_id", "-"),
                    engine.get("memory_gb"),
                    now - self._firing_since,
                    now,
                    self.motion,
                )
                self.query_one("#line-chef", Chef).show("loading", now, motion=self.motion)
            if self._width < NARROW_COLUMNS:
                self._refresh_footer()
        except NoMatches:
            # A tick that lands while the screen is being torn down: nothing
            # left to draw on.
            return

    def _idle_tick(self) -> None:
        now = self._clock()
        try:
            engine = self._document.get("engine") or {}
            mood = self._mood(engine, now)
            self.query_one("#line-chef", Chef).show(mood, now, motion=self.motion)
            card = self.query_one(Card)
            if card.display and self._connected and not engine.get("loading"):
                card.show_quiet(
                    engine,
                    self._document.get("config") or {},
                    self._document.get("recent_turns") or [],
                    now,
                    self.motion,
                )
            elif card.display and not self._connected:
                self._show_left(None)
        except NoMatches:
            return

    # -- keys ------------------------------------------------------------------------

    async def action_quit(self) -> None:
        self.exit(return_code=0 if self._ever_connected else 1)

    def action_pause(self) -> None:
        paused = self.query_one(Rail).toggle_pause()
        self.query_one("#gingham", Rule)._title = (
            "RECENT ORDERS · paused" if paused else "RECENT ORDERS"
        )
        self.query_one("#gingham").refresh()

    def action_legend(self) -> None:
        self.push_screen(Overlay(LEGEND, "the legend"))

    def action_about(self) -> None:
        self.push_screen(Overlay(ABOUT, "about"))

    def action_ticket(self) -> None:
        self._open_ticket(self.query_one(Rail).selected())

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # The rail has focus and took the enter for itself: same card.
        key = event.row_key.value if event.row_key is not None else None
        self._open_ticket(self.query_one(Rail).selected(key))

    def _open_ticket(self, turn: dict | None) -> None:
        if turn is not None:
            self.push_screen(Overlay(ticket_text(turn), f"ORDER № {turn.get('id', '')[:10]}"))

    def action_reconnect(self) -> None:
        self._retry_now.set()

    def action_motion(self) -> None:
        self.motion = not self.motion
        if self._document:
            self._apply(self._document, resize=True)
        self.notify("motion off" if not self.motion else "motion on", timeout=1.5)


def run_top(port: int) -> int:
    """`sous top`: the app on the alternate screen until q, Esc or Ctrl-C;
    1 when no daemon ever answered, 0 otherwise."""
    app = Top(lambda: sse_feed(port), port=port)
    app.run()
    return app.return_code or 0
```

(This module was run headless under Textual 8.2.8 on Python 3.14 with a scripted feed at 100×30, 80×24, 50×20 and 120×36 through every state — decode with a queued order, prefill with and without a size, firing up, quiet, stalled at 6 s and 12 s, redialing, closed — and passes the tests above and `ruff check`/`ruff format` at line length 100 as written. `ty` may still name a rule: fix the cause, never suppress. `Rail` reads `getattr(self.app, "motion", True)` and `Slip._motion()` does the same because `App` has no `motion` attribute in `ty`'s eyes. `Top._breakpoints` classes `self._base_screen`, captured in `on_mount`, because `App.screen` is whichever overlay is up and a resize under the legend would otherwise reflow the overlay and leave the pass half-done.)

- [ ] **Step 5: The two commands, and the minutes in the document**

In `src/sous/cli.py`, after `_cmd_statusline` add:

```python
def _cmd_top() -> None:
    """The terminal. Textual is imported here and nowhere else in sous, so
    the daemon and every other subcommand run without it."""
    config = load_config()
    from sous.tui import run_top

    raise SystemExit(run_top(config.server_port))
```

In `main`: the `status` parser gains a flag —

```python
    status = sub.add_parser("status", help="check the daemon and recent tasks")
    status.add_argument("--watch", action="store_true", help="the live terminal (sous top)")
```

(replacing `sub.add_parser("status", help="check the daemon and recent tasks")`), then after the `statusline` parser:

```python
    sub.add_parser("top", help="watch the pass live: the order on it, the line, the recent orders")
```

and in the dispatch chain, replace `_cmd_status()` with

```python
    elif args.command == "status":
        _cmd_top() if args.watch else _cmd_status()
    elif args.command == "top":
        _cmd_top()
```

(one `elif` for `status`, one for `top`). The module docstring's first line becomes `"""sous CLI: serve / status / top / statusline / wait / stop / mcp / claude / install- and` with `uninstall-launchd."""` on the next line.

In `src/sous/server.py`, `status_document`: add `"idle_unload_minutes": self.config.idle_unload_minutes,` to the `config` dict right after `"model_id": self.config.model_id,`; in `tests/test_server.py`'s `test_server_status`, add `assert s["config"]["idle_unload_minutes"] == 30` after the `model_id` assertion.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tui.py tests/test_cli.py tests/test_server.py 2>&1 | tail -3`
Expected: all pass; `test_tui.py` in about 12 s (the pilot tests pause for tenths of a second each).

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1077 passed, 8 skipped, 29 deselected`; `uv run ty check`; ruff format/check; `uv lock --check` — clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/sous/tui.py src/sous/cli.py src/sous/server.py tests/test_tui.py tests/test_cli.py tests/test_server.py
git commit -m "feat(cli): watch the pass live with sous top" -m "A Textual application fed by GET /sous/events: the order on the pass as a perforated slip with its gauges, a kitchen-timer dial, the rate as a neon EQ and the walk-in's verdict; a pixel chef on the line whose face is the state; the engine, the prompt cache and the tickets beside it; the recent orders on a plain table whose last column says how each ended and what it cost. Drawn the way 1991 drew things, on the terminal's own background, with every accent legible on black and on white and every kitchen word glued to its literal or its number. Reconnects with backoff, exits 1 when no daemon ever answered, reflows at 100, 80 and 60 columns. Textual is imported inside the one command function that needs it." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: The render snapshot, and the idle budget (`tui.py` tests)

**Files:**
- Modify: `tests/test_tui.py` (append), `.gitignore` (one line)
- Create: `tests/snapshots/sous-top-100x30.svg` (written by the test itself on first run — see Step 3)

**Interfaces:**
- Consumes: Task 5's app, its `Clock` and `Feed` test helpers.
- Produces: the snapshot file and its `SOUS_UPDATE_SNAPSHOTS=1` rewrite switch; the idle-CPU recipe the gate runs. No new names outside the tests.

- [ ] **Step 1: Write the failing test**

First add `import os` to `tests/test_tui.py`'s import block, between `import asyncio` and `import time` (an appended module-level import would be `E402`). Then append to `tests/test_tui.py`:

```python
SNAPSHOT = Path(__file__).parent / "snapshots" / "sous-top-100x30.svg"


def test_the_pass_renders_as_committed(monkeypatch):
    """The 100×30 screen mid-decode, cell for cell (an SVG of the screen).
    A deliberate change to the look — or a Textual upgrade that renders
    differently — is a rerun with SOUS_UPDATE_SNAPSHOTS=1 and a commit of
    the new file; the diff is the review."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    clock = Clock(BASE - 8)

    async def test(app, pilot, feed, clock):
        for n in range(1, 9):
            clock.now = BASE - 8 + n
            await feed.queue.put(_doc(_turn(generated_tokens=400 + n, decode_tps=9.0 + n * 0.3)))
            await pilot.pause(0.05)
        clock.now = BASE
        behind = _turn(
            id="msg_64f2b36f3e4a528198cee4ea",
            phase="queued",
            started_at=BASE - 7,
            input_tokens=8912,
        )
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT, tasks=TASKS, behind=[behind]))
        await pilot.wait_for_scheduled_animations()
        await pilot.pause(0.35)
        rendered = app.export_screenshot()
        if os.environ.get("SOUS_UPDATE_SNAPSHOTS") == "1":
            SNAPSHOT.parent.mkdir(exist_ok=True)
            SNAPSHOT.write_text(rendered)
        assert SNAPSHOT.exists(), "run once with SOUS_UPDATE_SNAPSHOTS=1 to write the snapshot"
        expected = SNAPSHOT.read_text()
        if rendered != expected:
            actual = SNAPSHOT.with_suffix(".actual.svg")
            actual.write_text(rendered)
            raise AssertionError(
                f"the render differs from {SNAPSHOT.name}; the new render is at {actual.name} — "
                "open both, and if the change is intended rerun with SOUS_UPDATE_SNAPSHOTS=1"
            )

    _run(test, clock=clock)
```

Add the line `tests/snapshots/*.actual.svg` to `.gitignore`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_tui.py -k committed 2>&1 | tail -3`
Expected: FAIL — `AssertionError: run once with SOUS_UPDATE_SNAPSHOTS=1 to write the snapshot`.

- [ ] **Step 3: Write the snapshot, run it again**

Run: `SOUS_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_tui.py -k committed 2>&1 | tail -3` → `1 passed`, and `tests/snapshots/sous-top-100x30.svg` exists (about 50 KB). Open it in a browser once and check it against the render in Task 5 (the slip with its dial and ETA, the queued order, the chef plating, `HI-SCORE 11.4 tok/s` in the strip, the book's five rows, the gingham rule, the legend and the keys); then `uv run pytest tests/test_tui.py 2>&1 | tail -3` → all pass, the snapshot byte-identical on the second run. Run the file three times in a row: the render must not depend on the wall clock (every frame reads the pinned clock) or on the test's timing (the highlight and the settling step are over after the pause).

Then: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` → `1078 passed, 8 skipped, 29 deselected`; `uv run ty check`; ruff — clean.

- [ ] **Step 4: The idle budget, by hand**

With no daemon running (`sous top` on the closed kitchen, chef frozen) and then against an idle one, in a 100×30 terminal for one minute each: `ps -o %cpu= -p $(pgrep -f "sous top")` every 10 s must read under 2 % on every sample. The only timers running while nothing cooks are the 0.5 s idle tick (the chef's `zZ` and the pot's steam, under 20 cells) and the feed's 1 Hz heartbeat; the card's `LoadingIndicator` starts a 16 Hz refresh of its own when it mounts, which `Card._state` turns off in every state but firing (the quiet-kitchen test asserts `auto_refresh is None`). If a sample exceeds 2 %, the cause is a timer left running or a rebuild on every heartbeat (`Rail.show` returns early when the ids are unchanged; `Strip`/`LinePanel` repaint only their own line) — fix it, do not raise the budget. Record the readings in the commit body.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tui.py tests/snapshots/sous-top-100x30.svg .gitignore
git commit -m "test(tui): pin the pass at 100×30" -m "A committed SVG of the screen mid-decode with a queued order, rendered from a pinned clock and a scripted feed, so a change to the look — or a Textual release that renders differently — shows up as a diff of the picture. SOUS_UPDATE_SNAPSHOTS=1 rewrites it. Idle CPU measured at <readings> against a closed and an idle daemon." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Documentation (`README.md`, `CLAUDE.md`, `SECURITY.md`, `CONTRIBUTING.md`)

**Files:**
- Modify: `README.md` ("Managing the daemon" command block at lines 109–115; "Gateway mode" — the paragraph beginning "The preflight itself is plain HTTP" at lines 224–229 and the `server_status` paragraph at lines 617–621), `CLAUDE.md` (the `EngineManager`/`/sous/` gotcha ending "no compatibility fallback." at lines 96–113; a new gotcha after it), `SECURITY.md` (the `/sous/` routes bullet at lines 63–71), `CONTRIBUTING.md` (the sentence "Watch `~/.sous/daemon.log` for the `INFO sous.gateway:` lines" at lines 147–149)
- Test: none (docs); the README's screenshot-as-text is Task 5's headless render at 100×30, pasted verbatim.

**Interfaces:** Consumes every name Tasks 1–6 produce; produces nothing.

- [ ] **Step 1: README — "Managing the daemon"**

Replace the command block (lines 109–115) with:

```bash
sous status             # is it up, and what has it been doing
sous top                # watch it live (alias: sous status --watch); q quits
sous statusline         # one line for Claude Code's status bar (below)
sous wait <task-id>     # block until a task finishes or needs approval
sous claude             # Claude Code with local subagents (gateway mode, below)
sous stop               # stop it (see below)
sous uninstall-launchd  # stop it starting at login, and remove the agent
```

- [ ] **Step 2: README — "Gateway mode": the routes, the terminal, the status line**

Replace the paragraph beginning "The preflight itself is plain HTTP" (lines 224–229) with:

````markdown
The preflight itself is plain HTTP. The daemon's own routes live under
`/sous/` on the same port, loopback-only like the gateway's, whether or not
the gateway is on:

- `GET /sous/status` — one JSON document: `engine` (`loaded`, `loading`,
  `model_id`, `idle_seconds`, `holders`, `memory_gb`, the `prompt_cache`
  counters), `inflight` (the turn the model is serving right now — its
  `msg_` id, phase, tokens so far, rate and ETA — usually empty or one
  entry), `queue` (delegated task counts), `recent_turns` (the last 50,
  every field of the turn line below), `recent_tasks` (the last 10) and
  `config`. The MCP `server_status` tool returns the same document without
  the two `recent_*` lists.
- `GET /sous/events` — the same document as a Server-Sent Events stream:
  once at connect, then whenever the turn in flight changes (at most ten
  times a second) and at least once a second, with a `ping` every 10 s.
- `POST /sous/hold` — what `sous claude` posts (above).

A `404` from `/sous/status` means the running daemon predates this CLI (the
route did not exist); `sous claude` says so and exits 1 — restart the daemon
from the same install (`sous stop`, then `sous serve` or `sous
install-launchd`). There is no compatibility mode: the CLI and the daemon
ship as one package.

**`sous top`** (or `sous status --watch`) is that stream in a terminal —
open it beside a `sous claude` session and watch a subagent's order go from
the rail to the plate — a perforated slip with a timer dial, a pixel chef
whose face is the state, the walk-in's counters, and the recent orders:

```text
∿▲● sous top ●▲∿  THE PASS   ∿   ▲    ●   ∿   ▲    ●   ∿   ▲    ●   ∿   ▲    HI-SCORE 11.4 tok/s ▲●∿
∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
┏╍ ORDER № msg_f4cfe7 ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓ ╭─ THE LINE ────────────────────────╮
╏ PLATING  decode 412 tok    ◶ 00:41.3            ETA 01:02  ╏ │   ▟███▙   LINE IS OPEN  loaded    │
╏                                                            ╏ │   (^_^)   Qwen3.8-27B-4bit        │
╏ SEAR  ▐█████████████████████▌  prefill 8,845 tok · done    ╏ │   ▐███▌═  31.4 GB · hold 1        │
╏ PLATE ▐████████▁▁▁▁▁▁▁▁▁▁▁▁▁▌  412 / 1,024 tok · 40%       ╏ │                                   │
╏                                                            ╏ │ PROMPT CACHE   6 slots · 11.8 GB  │
╏ tok/s ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▇█ 9.7 now                      ╏ │ REHEAT  41 hit    DRAWER 18 fork  │
╏ REHEAT hit · reused 53,296 tok · 8,845 fresh               ╏ │ SCRATCH  7 miss  TOSSED  3 evict  │
╏                                                            ╏ │ WALK-IN FULL 1 pressure evict     │
╏ in 62,141 tok   max 1,024   started 22:12:38   1st of 2    ╏ │ reused 1,820,000 tok              │
┗╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍ tear here ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┛ │                                   │
┏╍ ORDER № msg_64f2b3 ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓ │ TICKETS  ON RAIL 2 · COOKING 1    │
╏ ON THE RAIL  queued 00:07   next up · 8,912 tok in         ╏ │ SERVED 3 · BURNT 1 · 86'D 1       │
┗╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍ tear here ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┛ ╰───────────────────── idle 4m 12s ─╯
▞▚▞▚▞▚ RECENT ORDERS ▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞
 time      ticket      cache         in      out    plate   total   why
 22:12:20  msg_000000  REHEAT 61.9k  62,000  1,024  41.2s   44.1s   PLATE FULL
 22:11:20  msg_000000  DRAWER 53.3k  62,000  1,024  41.2s   44.1s   ORDER UP · WALK-IN FULL ×2
 22:10:20  msg_000000  SCRATCH 0     62,000  1,024  41.2s   44.1s   ORDER UP · lcp 0 · new tools
 22:09:20  msg_000000  DROPPED IT    —       —      —       44.1s   DROPPED IT · 529 overloaded_e
 22:08:20  msg_000000  DROPPED IT    —       —      —       44.1s   WALKED OUT · 499 abandoned




 TASK  COOKING 0:41  rename Delta callback across engine tests
 SEAR prefill  PLATE decode  REHEAT hit  DRAWER fork  SCRATCH miss  TOSSED evicted  86'D cancelled
 q quit   l legend   ? about   m motion   p pause   ↑↓ orders   enter ticket   r reconnect
```

`q`, `Esc` or `Ctrl-C` leave the screen exactly as it was; `enter` opens
the order under the cursor with every field of its turn line, `l` the
legend, `?` the About card, `m` turns the motion off. The vocabulary is
glossed on the screen itself (`SEAR ▐███▌ prefill`, `REHEAT 41 hit`) and
on its legend line; the numbers are the turn line's, in the terminal's own
foreground, and nothing paints over the terminal's background. It reconnects
with backoff if the daemon restarts and says so if none is running. It is
the one command in sous that imports [Textual](https://textual.textualize.io);
nothing else (the daemon included) loads it.

**`sous statusline`** prints one line for Claude Code's `statusLine`
setting — `sous: decode 612 tok · 14.7 tok/s · eta 18s` during a turn,
`sous: idle · 5 slots · held` between turns, `sous: daemon down` when
nothing answers — and reads (and ignores) the JSON Claude Code pipes to it.
Add to `~/.claude/settings.json`:

```json
{"statusLine": {"type": "command", "command": "sous statusline", "refreshInterval": 1}}
```

`refreshInterval` matters: without it Claude Code re-runs the command only on
message events, and the line would freeze for a whole subagent turn. The
command imports nothing beyond the standard library and gives up inside
half a second, so it costs the status bar nothing.

To bracket a subagent from the outside as well, a `SubagentStart` /
`SubagentStop` hook can append its own record — `{ts, agent_id,
agent_type}` — to a file; it sees the subagent's start and end, not the
progress-summary calls Claude Code makes in between, which only the turn
line and `sous top` show.
````

Replace the `server_status` paragraph (lines 617–621) with:

```markdown
`server_status` (and `GET /sous/status`, the same document over HTTP, plus
the recent turns and tasks) reports the engine — `holders` (live `sous
claude` sessions pinning the model), `loading` (a load in progress, a
preload included), `memory_gb` and `prompt_cache` — slots, resident bytes,
hits, fork hits, retained and moved turn-slot takes, evictions and the
subset the pressure valve took — counts only — and `inflight`, the turn
being served right now with its phase, tokens and rate.
```

- [ ] **Step 3: CLAUDE.md — two gotchas**

In the gotcha that begins "`EngineManager` never holds its lock across a model load or unload", replace its last sentence (from "`sous claude` has no MCP client:" to "no compatibility fallback.") with:

```markdown
  `sous claude` has no MCP client: `/sous/status`, or
  a 404 from whatever holds `daemon.lock` = "restart the daemon" (from
  anything else = "not sous on this port"), no compatibility fallback. The
  status document is one shape everywhere (`SousService.status_document`):
  `engine`, `inflight`, `queue`, `config`, and over HTTP `recent_turns` and
  `recent_tasks` — the MCP tool drops the two lists because a frontier
  model pays for every key it reads.
```

After that gotcha, add a new bullet:

```markdown
- `sous/inflight.py` is the in-flight turn registry: a gateway turn
  registers under its `msg_` id in `TurnRunner.run` and is removed in its
  `finally`; `progress()` runs on the engine's session thread from inside
  the decode loop, so every registry method is a dict write under one lock
  and never raises (an unknown id is ignored). The prefill's size is not
  stamped by the runner — it is inside `session.generate()` for the whole
  prefill — but read by a *probe* the registry calls from a status reader's
  thread: the cache adds the reuse at the take and assigns `prefilled_tokens`
  on entering `_run`, both before the first prefill call, and `stats(owner)`
  is a locked dict copy no prefill holds the lock across. `/sous/events`
  polls `Inflight.version` ten times a second and rebuilds the whole
  document on a worker thread when it moved (and once a second regardless,
  because a load, a hold and the idle clock are not registry changes);
  never wake it from a writer. The routes record a summary for *every*
  `POST /v1/messages id=…` line — refused, abandoned and failed included —
  and the served line is printed from that summary (`_turn_line`), so a
  row and a line cannot disagree; PR 3 persists the dict as it is. Textual
  is imported only inside the `sous top` command function: `sous serve`,
  `sous claude` and `sous statusline` (stdlib only, half-second budget)
  never load it. The terminal runs with `ansi_color=True` and the
  `ansi-dark` theme pinned explicitly — foreground and background are the
  terminal's own — and paints hex accents on top of that ground, each
  chosen to clear 3:1 on black and on white (a test asserts it); nothing in
  `tui.py` sets a `background:` other than `ansi_default`. The chef's
  frames, the dial and the steam are picked from the app's clock, never
  tweened, so a pinned clock is a pinned picture.
```

- [ ] **Step 4: SECURITY.md — the routes bullet**

Replace the `/sous/` routes bullet (lines 63–71, through its closing parenthetical) with:

```markdown
- **The `/sous/` routes** (`GET /sous/status`, `GET /sous/events`,
  `POST /sous/hold`) — the daemon's own loopback routes, mounted whether or
  not the gateway is enabled, behind the same `Host`/`Origin` refusal.
  `/sous/status` and `/sous/events` serve the status document (engine
  state, the turn in flight, recent turns and tasks — counts, durations,
  hashes and identifiers; task titles are the user's own); `/sous/hold`
  pins the model in memory while a named process lives. In scope:
  reachability from anything but a loopback client; a path under `/sous`
  reaching the gateway's forwarder; anything of a hold body beyond a pid and
  a start time being acted on; any request body, prompt text, tool name or
  file path reaching the document, the event stream, `sous top` or `sous
  statusline`.
```

(The old bullet's trailing parenthetical — the pid being logged to attribute a hold — is superseded by the new "beyond a pid and a start time" clause; do not carry it forward.)

- [ ] **Step 5: CONTRIBUTING.md — one sentence**

Replace "Watch `~/.sous/daemon.log` for the `INFO sous.gateway:` lines (both streams land there; every line carries a timestamp and a level)." (lines 147–149) with:

```markdown
Watch `~/.sous/daemon.log` for the `INFO sous.gateway:` lines (both streams
land there; every line carries a timestamp and a level), or `sous top` in a
second terminal for the same turns as they happen.
```

- [ ] **Step 6: Check, then commit**

Run: `uv run ruff check . && uv run ruff format --check .` (the docs are excluded, but the commit must not carry a stray formatting change), and read each edited section once for a claim the code does not make (the README's route list against `mount_monitor`, the status-line strings against `statusline_text`).

```bash
git add README.md CLAUDE.md SECURITY.md CONTRIBUTING.md
git commit -m "docs: document the /sous/ routes, sous top and sous statusline" -m "The README names the three loopback routes and the document they serve, shows the terminal at 100×30 and gives the statusLine recipe with the refreshInterval it needs (without it the line freezes for a whole turn). CLAUDE.md records the registry's contract (a dict write under one lock, never raising, on the engine thread), the probe that reads the prefill's size, the poll-never-wake rule for /sous/events, and where Textual is and is not imported. SECURITY.md adds /sous/events to the routes in scope." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Orchestrator steps after Task 7

1. **Whole-branch review** (fresh Opus reviewer, the full diff against `main`): the plan-language sweep (the grep in Global Constraints); the docs' every claim against the code; and specifically — whether any `Inflight` method can raise or block on the engine thread (no lock but its own, no I/O, every lookup `.get`), whether `snapshot()` calls a probe under the registry lock, whether anything on the event loop does more than a registry append (`_log_turn`) or a `run_sync`, whether `_turn_line` can differ from the pre-refactor line for any `TurnResult` (compare field by field against the version in `main`), whether Textual is imported anywhere but `tui.py`, and `sous.tui` anywhere but inside the `sous top` command function (`grep -rniE "textual|from sous\.tui" src/sous --include=*.py` must show `tui.py`'s imports and, in `cli.py`, one function-local `from sous.tui import run_top` plus its docstring), whether anything in `tui.py` paints a background (`grep -n 'background:' src/sous/tui.py` shows `ansi_default` only) and whether the palette test still covers every accent constant, whether the TUI's 10 fps timer runs while idle, whether `statusline` imports anything beyond the stdlib at module or function level, whether `/sous/events` reads `EVENT_*` at call time, and whether the launcher, the monitor tests and the server tests all read `engine` (no `"model"` status key left: `grep -rn '"model"' src/sous/cli.py tests/test_cli.py | grep -v model_id` and `grep -rn '\["model"\]' tests/test_monitor.py tests/test_server.py` both print nothing — a turn record's own `"model": "sous-local"` field in a fixture is not a hit).
2. **PR** `feat/gateway-observability-pr2` → `main`, title `feat(monitor,cli): live status, GET /sous/events, sous top and sous statusline`, body from the commits plus one paragraph on the registry's contract and one on the TUI's look (with the 100×30 mock), footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and nothing beneath it.
3. **Real-daemon gate (needs the maintainer's go-ahead — installs branch code over their daemon):** on the M5 Pro, from `~/code/personal/sous-remote` at the branch: `uv sync`, the non-model suite, `uv tool install --force --reinstall .`, `sous stop`, wait until the old **pid is gone** (`kill -0`; `daemon.lock` can outlive the process as a stale file), restart detached (`/usr/bin/python3` + `subprocess.Popen([~/.local/bin/sous, "serve"], start_new_session=True, stdin=DEVNULL, stdout=stderr=open(~/.sous/daemon.log, "ab"))`), model unloaded, window as configured (262144). Then, in the GUI `screen` session, one detached window running `sous top` (`screen -S gui -X screen -t top zsh -lc 'sous top'`) and one `sous claude -p` session with one background general-purpose subagent of ≥3 tool-using turns, and:
   - **the three screenshots:** `screen -S gui -p top -X hardcopy -h ~/top-<phase>.txt` taken during the load (the launcher's preload), during the first turn's prefill and during a decode — driven by polling `/sous/status` (`gate_poll.py`, 3/s) and firing the hardcopy on each phase's first sighting. Each capture must show the phase's own panel state (the loading indicator; a prefill bar or pulse with the elapsed time; the decode bar, the token count moving between two captures 2 s apart, the rate). Attach the three text captures to the PR.
   - **the stream:** `curl -sN 127.0.0.1:8383/sous/events | head -c 20000` during a turn shows a first `status` event with every key, then events no closer than 100 ms apart and no farther than ~1 s, and a `ping` within 10 s of silence; `inflight[0].phase` walks `loading`/`tokenizing`/`prefill`/`decode` across the run, `to_prefill` appears during the cold turn's prefill (the probe), `generated_tokens` climbs by 1–4 per event during decode, `expected_output_tokens` is `null` until five turns have completed and a number after.
   - **the status line:** `echo '{}' | sous statusline` during decode prints `sous: decode N tok · R tok/s · eta …` and returns in under 0.5 s (`time`); between turns `sous: idle · N slots · held`; with the daemon stopped `sous: daemon down` in under 0.5 s.
   - **idle CPU:** with the TUI open against the idle daemon for 60 s, `ps -o %cpu= -p <sous top pid>` sampled at 10 s intervals reads under 2 % on every sample; the daemon's own `%cpu` with one `/sous/events` client connected and no turn is likewise under 2 %.
   - **during a turn:** the daemon's `%cpu` with the TUI connected is compared against the turn line's `decode_tps` of a turn served with no client connected — the same model, the same session — and any difference is reported (decision 3's cost, measured).
   - **cache and compaction, as the B gate:** every real subagent turn after the first logs `cache=hit took=turn@N`; the reconcile script joins the log's lines to the subagent transcript by `id=msg_…` and reports any `compact_boundary`.
   - **the recent table:** after the session, `/sous/status`'s `recent_turns` has one entry per `POST /v1/messages id=…` line in the log, in the same order, with matching `id`, `cache`, `input_tokens`, `output_tokens` and `seconds` to one decimal.
   Record the captures, the stream excerpt, the timings and the CPU readings on the PR. Reinstall from `main` after merge; kill the `top` screen window.
4. **Memory:** the PR 2 execution record (merge sha, rulings, the measured CPU costs, the TUI vocabulary the maintainer approved) as its own file; update the specs memory's "next" pointer to the PR 3 plan; note the `expected_output_tokens` estimate and the probe as facts PR 3's `sous status` rewrite can reuse.
