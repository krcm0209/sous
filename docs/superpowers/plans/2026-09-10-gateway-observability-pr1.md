# Gateway Observability PR 1 — Attribute a Turn, One Daemon Log — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every line the daemon writes carries a timestamp and a level and lands in one `~/.sous/daemon.log`; every locally served gateway turn logs where its seconds went (load, queue, tokenize, time to first token, prefill, decode), what slot it took and published, what was evicted under it, and — on a miss — which region of the prompt diverged, plus hashes of the tool array and system text.

**Architecture:** A new `sous/logs.py` owns the line shape (one formatter, one stderr handler that resolves `sys.stderr` at emit time so pytest's capture sees it, an idempotent installer that removes the MCP SDK's `RichHandler`). The gateway's `_log` becomes a `logging` call with a level. Timing lives where the work happens: `PrefixCache._run` and `_fork_boundaries` write per-turn gauges onto `PromptCacheStats` (the existing `miss_lcp` pattern), `TurnRunner.run` stamps queue/load/tokenize/first-delta and reads the gauges back owner-scoped, `convert.parse_messages_request` hashes the rendered tools and system text, and `Gateway._log_turn` prints it all. `sous install-launchd` learns to re-run and folds the misnamed `daemon.err.log` into `daemon.log`.

**Tech Stack:** Python 3.14, stdlib `logging`/`hashlib`/`plistlib`, Starlette routes already in place, pytest with `capsys`/`caplog`/`monkeypatch`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-gateway-observability-design.md` — section "PR 1 — attribute a turn" (including "Log streams: one file, one shape"), "Invariants preserved", "Testing" (PR 1 paragraph), "Documentation". The companion spec `docs/superpowers/specs/2026-09-10-gateway-cache-continuity-design.md` is verified with this PR's fields and must not be started before this lands.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local; none of this PR touches them.
- The gateway never logs a request body, a header value or a query string. Every new field is a count, a duration, an 8-hex-character hash of *rendered* text, or an identifier already bounded by `_log_token`.
- Tests never touch the real `~/.sous` or `~/.claude`: every path comes from `tmp_path`.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` (this plan and its spec included) once committed.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3` (never pipe without `pipefail` — it masked a failure once), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`. Several code blocks below are written wide for reading; `ruff format` reflows them (line length 100). `uv lock --check` after any `pyproject.toml` change (none expected here).
- `ty` flags an unused `# ty: ignore[...]` as a diagnostic: add a pragma only when `ty check` names the rule.
- Branch: `feat/gateway-observability-pr1` off `main`. `main` is protected: PR with all four CI jobs green.
- The daemon Kyle runs is unmanaged (`sous serve` started by hand, pid in `~/.sous/daemon.lock`). Nothing in this plan restarts it; the real-daemon gate at the end is run by the orchestrator, not a task subagent.

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/logs.py` (new) | `UTCFormatter`, `StderrHandler` (dynamic `sys.stderr`), `configure_daemon_logging()` (idempotent installer), `enable_warning_capture()`, `format_line()` (the same shape for the one print that must stay a print). No sous imports. |
| `src/sous/server.py` | `main()` configures logging first, enables warning capture, disables huggingface progress bars, and logs its own startup lines through `sous.server`; `create_server` re-runs the idempotent installer after `MCPServer(...)`; `uvicorn_config` passes `log_config=None`; the signal handler's print takes the line shape via `format_line`. |
| `src/sous/worker.py` | The worker-loop error print becomes `sous.worker` at ERROR. |
| `src/sous/gateway/upstream.py` | `_error` marks the responses it synthesizes (`sous_synthesized = True`) so the forwarded-request line can log them at the right level. |
| `src/sous/gateway/routes.py` | `_log(message, level)`, `_status_level`, `received` stamps on every path, failure lines with `seconds=`, the `count_tokens` success line, the new `_log_turn`, `_lcp_region`/`_rate`/`_opt` helpers. |
| `src/sous/gateway/turn.py` | `TurnResult` new fields, `CountResult`, timing in `run()` and `count_tokens()`. |
| `src/sous/gateway/convert.py` | `ChatRequest.tools_hash` / `system_hash`, `_digest`. |
| `src/sous/engine/promptcache.py` | Gauges on `PromptCacheStats` (+ `begin_turn`), `_clock`, timers in `_run`, probe timing and bounds in `_fork_boundaries`, `took_len` in `generate`. |
| `src/sous/engine/base.py` | `EngineManager.get()` logs `model loaded in N.N s`. |
| `src/sous/cli.py` | `launchd_plist` both streams → `daemon.log`; `_cmd_install_launchd` re-runnable with `_fold_legacy_stderr_log`. |
| `tests/test_logs.py` (new), `tests/test_server.py`, `tests/test_gateway_routes.py`, `tests/test_gateway_turn.py`, `tests/test_gateway_convert.py`, `tests/test_promptcache.py`, `tests/test_engine_base.py`, `tests/test_cli.py` | Tests per task below. |
| `README.md`, `CONTRIBUTING.md`, `CLAUDE.md` | Docs per Task 9. |

---

### Task 1: `sous/logs.py` — one line shape, one handler, idempotent install

**Files:**
- Create: `src/sous/logs.py`
- Modify: `src/sous/server.py` (`create_server` at ~line 298–330, the signal handler at ~line 482–488, `uvicorn_config` at ~line 509, `main()` at ~line 536–570), `src/sous/worker.py` (the worker-loop error print at ~line 474)
- Test: `tests/test_logs.py` (new), `tests/test_server.py`

**Interfaces:**
- Produces: `sous.logs.SOUS_HANDLER_NAME: str`; `sous.logs.UTCFormatter(logging.Formatter)`; `sous.logs.StderrHandler(logging.StreamHandler)`; `sous.logs.configure_daemon_logging() -> None`; `sous.logs.enable_warning_capture() -> None`; `sous.logs.format_line(level: str, name: str, message: str) -> str` (the formatter's output for code that cannot go through logging — the SIGTERM handler, which runs where logging's lock is not safe to take). Later tasks log through `logging.getLogger("sous.gateway")` / `"sous.engine"` and rely on the root handler this installs. After this task every daemon-side line sous itself emits has the shape: the two startup lines and the worker-loop error become logger calls, the signal handler's line is formatted by hand. The CLI's user-facing `print`s (`sous status`, `sous claude …`) are terminal output, not daemon log, and stay as they are.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_logs.py`:

```python
"""The daemon's one line shape: `<iso-utc-ms>Z LEVEL name: message`, on a
handler that follows sys.stderr so pytest's capture sees it, installed
idempotently over whatever the MCP SDK put on the root logger."""

import logging
import re
import warnings
from datetime import UTC, datetime

import pytest

from sous.logs import (
    SOUS_HANDLER_NAME,
    StderrHandler,
    UTCFormatter,
    configure_daemon_logging,
    enable_warning_capture,
    format_line,
)

LINE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (INFO|WARNING|ERROR) sous\.test: hello$"
)


class RichHandler(logging.Handler):
    """Stands in for rich.logging.RichHandler: the installer matches the SDK's
    handler by class name, and rich is only a transitive dependency here."""

    def emit(self, record: logging.LogRecord) -> None:
        pass


def _sous_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == SOUS_HANDLER_NAME]


@pytest.fixture(autouse=True)
def _clean_root():
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
    for h in before:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)
    logging.captureWarnings(False)


def test_formatter_prints_iso_utc_millis_level_and_name():
    record = logging.LogRecord("sous.test", logging.INFO, __file__, 1, "hello", None, None)
    record.created = 1789000000.123
    record.msecs = 123.0
    text = UTCFormatter().format(record)
    assert LINE.match(text), text
    # Pinned to the UTC rendering: a localtime regression would still match
    # the regex, so the exact stamp is the assertion that matters.
    assert text.startswith("2026-09-10T00:26:40.123Z "), text
    stamp = datetime.fromisoformat(text.split(" ", 1)[0].replace("Z", "+00:00"))
    assert stamp == datetime.fromtimestamp(1789000000.123, UTC).replace(microsecond=123000)


def test_format_line_matches_the_formatter_shape():
    text = format_line("WARNING", "sous.test", "hello")
    assert LINE.match(text), text
    assert " WARNING sous.test: hello" in text


def test_configure_is_idempotent_and_installs_exactly_one_handler():
    configure_daemon_logging()
    configure_daemon_logging()
    assert len(_sous_handlers()) == 1
    assert isinstance(_sous_handlers()[0], StderrHandler)
    assert isinstance(_sous_handlers()[0].formatter, UTCFormatter)
    assert logging.getLogger().level == logging.INFO


def test_configure_removes_the_sdks_rich_handler():
    root = logging.getLogger()
    root.addHandler(RichHandler())
    configure_daemon_logging()
    assert not [h for h in root.handlers if type(h).__name__ == "RichHandler"]


def test_handler_writes_to_the_current_sys_stderr(capsys):
    """Bound to sys.stderr at emit time: capsys swaps the stream per test,
    and a handler bound at construction would write past the capture."""
    configure_daemon_logging()
    logging.getLogger("sous.test").info("hello")
    err = capsys.readouterr().err
    assert LINE.match(err.strip()), err


def test_levels_are_spelled_out_in_the_line(capsys):
    configure_daemon_logging()
    logging.getLogger("sous.test").warning("hello")
    logging.getLogger("sous.test").error("hello")
    lines = capsys.readouterr().err.splitlines()
    assert " WARNING sous.test: hello" in lines[0]
    assert " ERROR sous.test: hello" in lines[1]


def test_warning_capture_routes_warnings_through_logging(caplog):
    enable_warning_capture()
    with caplog.at_level(logging.WARNING, logger="py.warnings"):
        warnings.warn("sous prompt cache: boom", stacklevel=1)
    assert any("sous prompt cache: boom" in r.getMessage() for r in caplog.records)
    assert all(r.name == "py.warnings" and r.levelno == logging.WARNING for r in caplog.records)
```

Append to `tests/test_server.py`:

```python
def test_create_server_installs_the_daemon_log_handler_and_drops_the_rich_one(svc):
    import logging

    from sous.logs import SOUS_HANDLER_NAME
    from sous.server import create_server

    class RichHandler(logging.Handler):  # the SDK's, matched by class name
        def emit(self, record: logging.LogRecord) -> None:
            pass

    service, store, _ = svc
    root = logging.getLogger()
    root.addHandler(RichHandler())
    try:
        create_server(store, service.engines, service.config)
        names = [h.get_name() for h in root.handlers]
        assert names.count(SOUS_HANDLER_NAME) == 1
        assert not [h for h in root.handlers if type(h).__name__ == "RichHandler"]
    finally:
        for h in list(root.handlers):
            if h.get_name() == SOUS_HANDLER_NAME or type(h).__name__ == "RichHandler":
                root.removeHandler(h)


def test_uvicorn_config_leaves_logging_to_the_daemon():
    """uvicorn's default dictConfig gives its loggers their own handlers and
    stops propagation, so its lines would keep the `INFO:     …` shape
    instead of the daemon's. log_config=None lets them reach the root."""
    from sous.server import uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.log_config is None
    assert cfg.access_log is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_logs.py tests/test_server.py -q -k "log or uvicorn" 2>&1 | tail -5`
Expected: FAIL — `ModuleNotFoundError: No module named 'sous.logs'` for the new file; `AssertionError` on `log_config is None` (uvicorn's default is its `LOGGING_CONFIG` dict).

- [ ] **Step 3: Write `src/sous/logs.py`**

```python
"""One shape for every line the daemon writes.

`<iso-utc-ms>Z LEVEL name: message` — the timestamp launchd's log never
had, the level the filename `daemon.err.log` falsely promised, and the
logger name (`sous.gateway`, `sous.engine`, `mcp.…`, `uvicorn.error`) that
says which part spoke. Installed on the root logger, replacing the
`RichHandler` the MCP SDK's `configure_logging("INFO")` puts there from
`MCPServer.__init__` — that handler wraps every record at 80 columns, which
made the log ungreppable.
"""

from __future__ import annotations

import logging
import sys
import time

SOUS_HANDLER_NAME = "sous-daemon-log"


class UTCFormatter(logging.Formatter):
    """`2026-09-10T19:26:14.025Z INFO sous.gateway: …` — UTC, milliseconds,
    Python's level names. Exceptions are appended the way logging always
    formats them."""

    def format(self, record: logging.LogRecord) -> str:
        line = format_line(
            record.levelname, record.name, record.getMessage(), record.created, record.msecs
        )
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def format_line(
    level: str, name: str, message: str, created: float | None = None, msecs: float | None = None
) -> str:
    """The line shape, for the formatter and for the one place that cannot
    log: the SIGTERM handler in server.py, which runs asynchronously to
    whatever thread holds logging's lock and must print instead."""
    now = time.time() if created is None else created
    millis = int((now % 1) * 1000) if msecs is None else int(msecs)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    return f"{stamp}.{millis:03d}Z {level} {name}: {message}"


class StderrHandler(logging.StreamHandler):
    """A StreamHandler whose stream is whatever `sys.stderr` is at emit time.

    The stdlib's own `logging._StderrHandler` (behind `lastResort`) works this
    way for the same reason: the daemon is one process for hours and pytest
    swaps `sys.stderr` per test, so a stream captured at construction is the
    wrong one later. The setter is a no-op so `StreamHandler.__init__` and
    `setStream` cannot pin a stale stream."""

    def __init__(self) -> None:
        super().__init__()
        self.set_name(SOUS_HANDLER_NAME)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


def configure_daemon_logging() -> None:
    """Install the daemon's handler on the root logger, once.

    Idempotent, so `create_server` can call it every time it assembles the
    app (tests build many). Removes any earlier copy of this handler and the
    SDK's RichHandler (matched by class name: rich is a transitive dependency
    and this module must not import it). Sets the root level to INFO when it
    is unset or higher — the SDK sets INFO too, but this must not depend on
    call order. Library logger *levels* are untouched: the gateway pins
    sse-starlette/httpx/httpcore above where they log bodies and URLs, and
    a handler swap must never loosen that."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == SOUS_HANDLER_NAME or type(handler).__name__ == "RichHandler":
            root.removeHandler(handler)
    handler = StderrHandler()
    handler.setFormatter(UTCFormatter())
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def enable_warning_capture() -> None:
    """Route `warnings.warn` (the prompt cache's degradation notices) through
    logging as `WARNING py.warnings: …`, so they take the line shape too.
    Called from the daemon entry point only: under pytest the warnings plugin
    owns `showwarning`, and tests assert on `pytest.warns`."""
    logging.captureWarnings(True)
```

`ty` accepts the `stream` property override as written (verified); add a `# ty: ignore[<rule>]` only if a future ty names one — an unused pragma is itself a diagnostic.

- [ ] **Step 4: Wire it into `server.py`**

In `create_server`, immediately after `mcp = MCPServer("sous", instructions=_INSTRUCTIONS, lifespan=_lifespan)`:

```python
    # MCPServer.__init__ just ran the SDK's configure_logging("INFO"), which
    # basicConfig's a RichHandler onto the root logger — every record wrapped
    # at 80 columns. Replace it with the daemon's one-line shape now, before
    # the first line anyone cares about is written.
    configure_daemon_logging()
```

Add `from sous.logs import configure_daemon_logging, enable_warning_capture` to the imports.

In `uvicorn_config`, add `log_config=None,` with the comment:

```python
        # uvicorn's default dictConfig gives `uvicorn` and `uvicorn.error`
        # their own stderr handlers and stops propagation, so its lines would
        # keep the `INFO:     Started server process` shape beside the
        # daemon's timestamped ones. None leaves logging to sous.logs.
        log_config=None,
```

In `main()`, right after `_lock = _acquire_singleton_lock(config.data_dir)`:

```python
    # The line shape from the first line on: install the handler before the
    # startup lines below (create_server re-runs the idempotent installer
    # after MCPServer.__init__ puts the SDK's RichHandler back). Two more
    # things only the daemon entry point may do to process-wide state: route
    # warnings.warn through logging (pytest owns showwarning in tests), and
    # silence huggingface_hub's tqdm bars — terminal animation that lands in
    # a log file as garbage (`Fetching 13 files: 100%|██████████|`).
    configure_daemon_logging()
    enable_warning_capture()
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
```

Then convert the daemon's own bare prints, so "every line sous emits" is true. Add `_logger = logging.getLogger("sous.server")` at module level in `server.py` (with `import logging` if absent) and:

- `main()`: `print(f"sous: marked {interrupted} interrupted task(s) as failed")` → `_logger.warning(f"marked {interrupted} interrupted task(s) as failed")`; the gateway banner `print(f"sous: gateway (experimental) serving …")` → `_logger.info(f"gateway (experimental) serving …")` with the same text minus the `sous: ` prefix (the logger name carries it).
- The SIGTERM handler (`handle()` inside `_install_shutdown_handler`, ~line 482–488): keep the `print(..., file=sys.stderr)` — a signal handler may interrupt a thread holding logging's lock, and `os._exit(0)` follows — but shape it: `print(format_line("WARNING", "sous.server", f"killed {killed} running command group(s)"), file=sys.stderr)`. Import `format_line` alongside the others.
- `src/sous/worker.py` ~line 474: `print(f"sous: worker loop error (continuing): {e}", file=sys.stderr)` → `logging.getLogger("sous.worker").error(f"worker loop error (continuing): {e}")` (add `import logging`; if `sys` then has no other use in `worker.py`, remove `import sys` — ruff F401). `grep -rn "worker loop error\|command group\|interrupted task" tests/` finds no test asserting on these prints (checked); if one appears, switch it to `caplog`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logs.py tests/test_server.py -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 6: Run the whole suite, types, lint**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
```
Expected: green. If a routes test that asserts on `capsys.readouterr().err` now sees extra library lines and fails on an exact-equality check, note it for Task 2 (which rewrites those assertions) rather than weakening it here.

- [ ] **Step 7: Commit**

```bash
git add src/sous/logs.py src/sous/server.py src/sous/worker.py tests/test_logs.py tests/test_server.py
git commit -m "feat(server): one daemon log line shape with timestamps and levels

daemon.err.log was launchd's stderr, not an error log: Python logging,
the gateway's prints, warnings and the MCP SDK's RichHandler all wrote
there, the SDK's handler wrapped every record at 80 columns, and nothing
carried a timestamp. Install one formatter/handler on the root logger
(idempotently, replacing the SDK's), let uvicorn's loggers propagate to
it, and capture warnings into it at the daemon entry point.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: the gateway logs through `logging`, with levels

**Files:**
- Modify: `src/sous/gateway/routes.py` (`_log` at line 123; `import sys` at line 19; call sites at lines 354, 404, 430, 452, 461, 471, 482, 517, 544, 631, 653), `src/sous/gateway/upstream.py` (`_error` at lines 107–112)
- Test: `tests/test_gateway_routes.py` (ten occurrences of the old `sous gateway:` prefix: lines ~732, ~825, ~1277–1286, ~1316; the new tests go after `test_the_log_carries_lcp_only_on_a_miss`)

**Interfaces:**
- Consumes: root handler from Task 1 (installed by `create_server`, which `_app()` in the routes tests calls).
- Produces: `routes._log(message: str, level: int = logging.INFO) -> None`; `routes._status_level(status: int) -> int` (`ERROR` for `status >= 500 and status != 529`, else `WARNING`); `upstream.SynthesizedError(JSONResponse)` with class attribute `sous_synthesized = True`, returned by `upstream._error` for the 400/499/502/504 responses the forwarder makes itself, so `_forward` can log those at `_status_level` while a status the upstream returned stays INFO (it is the upstream's verdict, not sous's failure — and by status alone the two are indistinguishable). Every existing line keeps its text; only the prefix changes from `sous gateway: ` to `<ts> LEVEL sous.gateway: `.
- Not tested end to end: the WARNING on the "abandoned while queued" line needs a queued client to disconnect mid-wait; its level is the literal `logging.WARNING` at the call site and `_status_level` is unit-tested.

- [ ] **Step 1: Update the existing assertions and write the failing level tests**

In `tests/test_gateway_routes.py` the prefix moves from the start of the line to after the timestamp and level, so every `startswith` on it must become a substring check:

- ~line 732 (`test_log_lines_carry_metadata_only`): `"sous gateway: POST /v1/messages" in err` → `"sous.gateway: POST /v1/messages" in err`.
- ~line 825 (`test_a_dropped_tool_type_cannot_forge_a_log_line`): the forged string becomes `"x\nsous.gateway: POST /v1/messages status=200"` and its assertion uses the new prefix.
- ~lines 1277–1286 (`test_forwarding_logs_one_bounded_metadata_line_and_never_a_body_header_or_query`): the filter becomes `lines = [line for line in err.splitlines() if "sous.gateway: upstream" in line]`, and each `assert lines[N].startswith("sous gateway: upstream <rest>")` becomes `assert "sous.gateway: upstream <rest>" in lines[N]` with the same `<rest>`. `_root_stderr_logging()` (just above it) adds a bare `StreamHandler` with no formatter, so each record is also emitted as the plain message; only the daemon handler's copy contains `sous.gateway:`, so `assert len(lines) == 5` still holds. Leave that helper alone.
- ~line 1316 (`test_a_claimed_model_id_cannot_forge_or_bloat_a_log_line`): the filter becomes `lines = [line for line in err.splitlines() if "sous.gateway:" in line]`, and its forged id becomes `"sous-local[\nsous.gateway: FORGED status=200]"`. (Its `len(line) < 200` bound is Task 7's to raise.)

Then add:

```python
def _gateway_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "sous.gateway"]


def test_a_served_turn_logs_at_info(tmp_path: Path, caplog):
    app = _app(tmp_path, FakeEngine(["ok"]))
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        assert _post(app, _body()).status_code == 200
    turn = [r for r in _gateway_records(caplog) if "POST /v1/messages" in r.getMessage()]
    assert turn and turn[-1].levelno == logging.INFO


def test_a_refused_request_logs_at_warning(tmp_path: Path, caplog):
    """4xx and 529 are refusals sous chose — a client's malformed body, or
    back-pressure the SDKs retry — not failures."""
    app = _app(tmp_path, FakeEngine([]))
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        assert _post(app, _body(max_tokens=0)).status_code == 400
    assert [r.levelno for r in _gateway_records(caplog)] == [logging.WARNING]


def test_a_failure_sous_produced_logs_at_error(tmp_path: Path, caplog):
    class Boom(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            raise RuntimeError("engine exploded")

    app = _app(tmp_path, Boom([]))
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        assert _post(app, _body()).status_code == 500
    assert logging.ERROR in [r.levelno for r in _gateway_records(caplog)]


def test_a_full_queue_logs_at_warning_not_error(tmp_path: Path, monkeypatch, caplog):
    """529 is the one 5xx that is back-pressure, not a fault."""
    import sous.gateway.routes as routes

    monkeypatch.setattr(routes, "MAX_PENDING_TURNS", 0)
    app = _app(tmp_path, FakeEngine([]))
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        assert _post(app, _body()).status_code == 529
    assert [r.levelno for r in _gateway_records(caplog)] == [logging.WARNING]


def test_status_level_splits_refusals_from_failures():
    from sous.gateway.routes import _status_level

    assert _status_level(400) == _status_level(413) == _status_level(529) == logging.WARNING
    assert _status_level(500) == _status_level(502) == _status_level(504) == logging.ERROR


def test_an_unreachable_upstream_logs_at_error_but_a_forwarded_status_at_info(
    tmp_path: Path, caplog
):
    """502/504 the forwarder synthesizes are sous's failures; a 5xx the real
    upstream answered is the upstream's verdict and stays INFO — the same
    number, told apart by the marker `_error` sets, never by status."""
    from sous.gateway.upstream import Upstream

    unreachable = Upstream("https://127.0.0.1:9")  # the discard port: refused at once
    app = _app(tmp_path, FakeEngine([]), upstream=unreachable)
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        r = _post(app, _body(model="claude-opus-5"))
    assert r.status_code == 502
    upstream_lines = [r for r in _gateway_records(caplog) if "upstream POST" in r.getMessage()]
    assert [r.levelno for r in upstream_lines] == [logging.ERROR]

    caplog.clear()
    app = _app(tmp_path, FakeEngine([]))  # FakeUpstream answers 200
    with caplog.at_level(logging.INFO, logger="sous.gateway"):
        assert _post(app, _body(model="claude-opus-5")).status_code == 200
    upstream_lines = [r for r in _gateway_records(caplog) if "upstream POST" in r.getMessage()]
    assert [r.levelno for r in upstream_lines] == [logging.INFO]


def test_mounting_pins_the_noisy_library_loggers(tmp_path: Path, monkeypatch):
    """The handler swap (sous.logs) replaces the root handler, never a library
    logger's level — the no-bodies rule rests on these three pins."""
    for name in ("sse_starlette", "httpx", "httpcore"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.DEBUG)
    _app(tmp_path, FakeEngine([]))  # create_server → configure_daemon_logging → mount_gateway
    assert logging.getLogger("sse_starlette").level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
```

`logging` is already imported at the top of the test module. `MAX_PENDING_TURNS` is read in `Gateway.__init__`, so patching the module constant before `_app` builds the app works (`threading.BoundedSemaphore(0)` is valid and refuses every non-blocking acquire); the existing `test_a_full_queue_answers_529_with_an_anthropic_shaped_body` patches it to 1 and pre-acquires the slot — either shape is fine. If `Upstream("https://127.0.0.1:9")` is rejected by the constructor (it validates an https origin), use the URL form `test_gateway_upstream.py` already uses for its connection-refused case.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_routes.py -q -k "log" 2>&1 | tail -8`
Expected: the renamed-prefix tests FAIL (`sous.gateway:` not in stderr — the print still says `sous gateway:`); the level tests FAIL with no `sous.gateway` records at all.

- [ ] **Step 3: Rewrite `_log` and classify the call sites**

In `routes.py`, replace `_log`:

```python
_logger = logging.getLogger("sous.gateway")


def _log(message: str, level: int = logging.INFO) -> None:
    """One metadata line per event, at a level that means something: INFO for
    a request served or forwarded, WARNING for one sous refused (a 4xx, a
    529, a client gone while queued, tools it had to drop), ERROR for a
    failure sous produced (a 5xx from an exception, an unreachable upstream).
    The root handler (sous.logs) adds the timestamp and the name."""
    _logger.log(level, message)


def _status_level(status: int) -> int:
    # 529 is back-pressure both official SDKs retry with backoff, not a fault.
    return logging.ERROR if status >= 500 and status != 529 else logging.WARNING
```

Delete `import sys` from `routes.py`'s imports (line 19): `_log`'s `file=sys.stderr` was its only use, and ruff's F401 fails the lint job otherwise.

In `upstream.py`, mark what the forwarder makes itself:

```python
class SynthesizedError(JSONResponse):
    """An Anthropic-shaped error the forwarder made itself (unreachable
    upstream, timeout, client gone) — as opposed to a status the upstream
    answered, which is forwarded verbatim. The gateway's log line tells the
    two apart by this class, never by the number: a 502 can be either."""

    sous_synthesized = True


def _error(status: int, message: str) -> SynthesizedError:
    return SynthesizedError(
        {"type": "error", "error": {"type": "api_error", "message": message}},
        status_code=status,
        headers={"via": VIA},
    )
```

(`ty` rejects an ad-hoc attribute on a `JSONResponse` instance — verified — so the flag is a class attribute. Every existing test that checks `_error`'s body and headers keeps passing: the subclass changes nothing on the wire.)

Then at each call site:

| line (before edit) | change |
|---|---|
| 354 `_log("closed" …)` | unchanged (INFO) |
| 404 `_log(f"upstream …")` | `level=_status_level(response.status_code) if getattr(response, "sous_synthesized", False) else logging.INFO` |
| 430 count 529 | add `level=logging.WARNING` |
| 452, 461 `RequestError` on messages | add `level=_status_level(e.status)` |
| 471 dropped tools | add `level=logging.WARNING` |
| 482 messages 529 | add `level=logging.WARNING` |
| 517 abandoned while queued | add `level=logging.WARNING` |
| 544 non-stream failure | add `level=_status_level(status)` |
| 631 stream failure | add `level=_status_level(status)` |
| 653 turn line | unchanged (INFO) |

`logging` is already imported in `routes.py` (it pins the library loggers in `mount_gateway`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_routes.py -q 2>&1 | tail -3`
Expected: PASS. If a test built on `_gateway_app` (which calls `mount_gateway` on a bare `MCPServer("test")`, never `create_server`) asserts on stderr, add `configure_daemon_logging()` (imported from `sous.logs`) as the first line of `_gateway_app` — the installer is idempotent.

- [ ] **Step 5: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/routes.py src/sous/gateway/upstream.py tests/test_gateway_routes.py
git commit -m "feat(gateway): log through logging with a level per line

A bare print gave every gateway line the same weight; the daemon log
now carries a level, so a refused request (4xx, 529, abandoned) is a
WARNING and a failure sous produced (any other 5xx) is an ERROR, and
'grep ERROR' means what daemon.err.log's name used to promise.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: seconds on every line; the `count_tokens` line; the model-load line

**Files:**
- Modify: `src/sous/gateway/turn.py` (`count_tokens` at lines 207–215; add `CountResult` next to `TurnResult`), `src/sous/gateway/routes.py` (`messages()` at 446–560, `count_tokens()` at 411–445, `_stream()` at ~590–640), `src/sous/engine/base.py` (`EngineManager.get` at lines 482–487)
- Test: `tests/test_gateway_turn.py` (`test_count_tokens_uses_the_engine_and_releases` at ~line 349), `tests/test_gateway_routes.py`, `tests/test_engine_base.py`

**Interfaces:**
- Produces: `turn.CountResult(count: int, load_seconds: float, seconds: float)` (frozen dataclass); `TurnRunner.count_tokens(...) -> CountResult`. Log lines: failures on `/v1/messages` and `/v1/messages/count_tokens` end in ` seconds=S`; a served count logs `POST /v1/messages/count_tokens model=M input_tokens=N load_s=L seconds=S` at INFO; `sous.engine` logs `model loaded in N.N s (<model_id>)` at INFO when `EngineManager.get()` constructs an engine.

- [ ] **Step 1: Write the failing tests**

`tests/test_gateway_turn.py` — replace the body of `test_count_tokens_uses_the_engine_and_releases`:

```python
def test_count_tokens_uses_the_engine_and_releases(tmp_path: Path, monkeypatch):
    import sous.gateway.turn as turn

    released: list[bool] = []
    monkeypatch.setattr(turn, "release_mlx_thread_state", lambda: released.append(True))
    inner = FakeEngine([])
    runner, _ = _runner(tmp_path, inner)
    result = runner.count_tokens(MSGS, [])
    assert isinstance(result, CountResult)
    assert result.count == inner.count_tokens(MSGS, [])
    assert result.load_seconds >= 0 and result.seconds >= result.load_seconds
    assert released == [True]


def test_count_tokens_reports_the_load_it_paid_for(tmp_path: Path):
    """A count can be the request that loads the model; the line must say so."""
    inner = FakeEngine([])

    def slow_factory(model_id: str):
        time.sleep(0.05)
        return inner

    engines = EngineManager(_cfg(tmp_path), engine_factory=slow_factory)
    runner = TurnRunner(engines, _cfg(tmp_path))
    first = runner.count_tokens(MSGS, [])
    second = runner.count_tokens(MSGS, [])
    assert first.load_seconds >= 0.05
    assert second.load_seconds < 0.05
```

Add `CountResult` to the `from sous.gateway.turn import (...)` list.

`tests/test_gateway_routes.py` — add:

```python
def _turn_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "sous.gateway: POST /v1/messages" in line]


def test_every_failure_line_ends_with_seconds(tmp_path: Path, capsys):
    """A refusal's wall time matters as much as a success's: a 529 that took
    ten seconds to be refused is a different problem from one that took none."""
    app = _app(tmp_path, FakeEngine([]))
    assert _post(app, _body(max_tokens=0)).status_code == 400
    assert _post(app, {"model": "sous-local", "max_tokens": 4}, path="/v1/messages/count_tokens").status_code == 400
    lines = _turn_lines(capsys.readouterr().err)
    assert len(lines) == 2
    assert all(" seconds=" in line for line in lines), lines


def test_a_served_count_logs_one_line_with_load_and_seconds(tmp_path: Path, capsys):
    app = _app(tmp_path, FakeEngine([]))
    body = {"model": "sous-local", "messages": [{"role": "user", "content": "hi"}]}
    assert _post(app, body, path="/v1/messages/count_tokens").status_code == 200
    (line,) = _turn_lines(capsys.readouterr().err)
    assert "POST /v1/messages/count_tokens model=sous-local input_tokens=" in line
    assert " load_s=" in line and " seconds=" in line
    assert "hi" not in line.split("input_tokens=")[0]  # body text never reaches the log
```

`tests/test_engine_base.py` — add:

```python
def test_get_logs_the_load_once_with_its_duration(caplog):
    import logging

    mgr, created = _manager()
    with caplog.at_level(logging.INFO, logger="sous.engine"):
        mgr.get()
        mgr.get()
    lines = [r.getMessage() for r in caplog.records if r.name == "sous.engine"]
    assert len(lines) == 1 and len(created) == 1
    assert lines[0].startswith("model loaded in ") and lines[0].endswith(" s (fake/model)")
```

(`_manager()`'s config has the default `model_id`; `FakeEngine.model_id` is `"fake/model"` — the line names the `model_id` the *engine* reports, `self._engine.model_id`, not the config's.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_turn.py tests/test_gateway_routes.py tests/test_engine_base.py -q -k "count or failure_line or load" 2>&1 | tail -8`
Expected: `ImportError: cannot import name 'CountResult'`; the routes tests FAIL on missing ` seconds=` / missing count line; the engine test FAIL with no records.

- [ ] **Step 3: `CountResult` and timing in `turn.py`**

Below `TurnResult`:

```python
@dataclass(frozen=True)
class CountResult:
    count: int
    # A count can be the request that loads the model — seconds of work the
    # client sees as latency and the log must attribute.
    load_seconds: float
    seconds: float
```

Replace `TurnRunner.count_tokens`:

```python
    def count_tokens(self, messages: list[dict], tools: list[dict]) -> CountResult:
        started = time.monotonic()
        try:
            # Same race as run(): this whole call happens outside _gen_lock.
            with self._engines.lease():
                loading = time.monotonic()
                engine = self._engines.get()
                load_seconds = time.monotonic() - loading
                count = engine.count_tokens(messages, tools)
            self._engines.touch()
            return CountResult(count, load_seconds, time.monotonic() - started)
        finally:
            release_mlx_thread_state()
```

- [ ] **Step 4: Seconds on every line in `routes.py`**

In `messages()`: first statement `received = time.monotonic()`. Define once, near `_status_level`:

```python
def _elapsed(since: float) -> str:
    return f" seconds={time.monotonic() - since:.1f}"
```

Append `{_elapsed(received)}` to the message string of every failure line in `messages()` (the two `RequestError` lines, the 529 line, the non-stream failure line) and inside the `turn()` closure's abandoned line (`received` is in its enclosing scope). Give `_stream` a `received: float` parameter (pass `received` at the one call site) and append `{_elapsed(received)}` to its failure line.

In `count_tokens()`: first statement `received = time.monotonic()`. The two `RequestError` returns currently log nothing — add before each `return JSONResponse(e.body(), status_code=e.status)`:

```python
            _log(
                f"POST /v1/messages/count_tokens status={e.status} error={e.error_type}"
                f"{_elapsed(received)}",
                level=_status_level(e.status),
            )
```

Append `{_elapsed(received)}` to the 529 line. Replace the tail of the method:

```python
        try:
            result = await self._submit(
                self._counts,
                self._pending_counts,
                self._runner.count_tokens,
                chat.messages,
                chat.tools,
            )
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            status, error_type, message = _classify(e)
            _log(
                f"POST /v1/messages/count_tokens model={_model_label(chat)} status={status} "
                f"error={error_type}{_elapsed(received)}",
                level=_status_level(status),
            )
            return _error_response(status, error_type, message)
        _log(
            f"POST /v1/messages/count_tokens model={_model_label(chat)} "
            f"input_tokens={result.count} load_s={result.load_seconds:.1f} "
            f"seconds={result.seconds:.1f}"
        )
        return JSONResponse({"input_tokens": result.count})
```

- [ ] **Step 5: The load line in `base.py`**

Add `import logging` to `engine/base.py`'s imports and, at module level near the top, `_logger = logging.getLogger("sous.engine")`. Replace `EngineManager.get`:

```python
    def get(self) -> ManagedEngine:
        with self._lock:
            if self._engine is None:
                loading = time.monotonic()
                self._engine = ManagedEngine(self._factory(self._config.model_id))
                # The one line that brackets a cold start in the daemon log —
                # before it, only huggingface_hub's own chatter said a load
                # happened, and a turn's `seconds` could not be split.
                _logger.info(
                    f"model loaded in {time.monotonic() - loading:.1f} s ({self._engine.model_id})"
                )
            self._last_used = time.monotonic()
            return self._engine
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_turn.py tests/test_gateway_routes.py tests/test_engine_base.py -q 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 7: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/turn.py src/sous/gateway/routes.py src/sous/engine/base.py tests/test_gateway_turn.py tests/test_gateway_routes.py tests/test_engine_base.py
git commit -m "feat(gateway): seconds on failure lines, a count_tokens line, a model-load line

Failure lines carried no duration, a locally served count_tokens logged
nothing on success even when it loaded the model, and the only evidence
of a load was huggingface_hub's chatter. Every line now ends in seconds
measured from request receipt, a served count logs its load and total,
and EngineManager.get logs the load it performed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: per-turn gauges in the prompt cache

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`_GAUGES`/`PromptCacheStats` at lines 280–315; `_fork_boundaries` at 539–595; `generate` at 720–912; `_run` at 913–1000)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Produces on `PromptCacheStats`: `prefilled_tokens: int`, `took_len: int`, `bound_lo: int`, `bound_hi: int`, `probe_seconds: float`, `prefill_seconds: float`, `decode_seconds: float` (all gauges, max-folded, zeroed by `begin_turn()` at the start of every `generate`); `promptcache._clock: Callable[[], float]` (defaults to `time.monotonic`; tests replace it). `stats()` dicts carry every field via `as_dict()` unchanged.
- Semantics: `prefilled_tokens = len(stable_ids) − reuse at entry` (what the turn had to prefill); `took_len` = length of the slot `_take` returned (0 on a miss); `bound_lo`/`bound_hi` = the probe's candidates that clear the floor and lie inside the render, ascending — `(0, b)` when there is one, `(0, 0)` when none or when `max_bytes <= 0`; `probe_seconds` = wall time of the `fork_at()` closure; `prefill_seconds` = every `hooks.prefill` call plus fork copies plus the snapshot; `decode_seconds` = the `hooks.decode` call plus the restore. On a pure-attention (all-trimmable) model prefill and decode are one fused call, reported as decode. A cold retry accumulates both attempts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py` (after the memo tests; `FakeHooks`, `STABLE_1`, `FULL_1`, `STABLE_2`, `FULL_2`, `ROOMY`, `FORK_MIN_TOKENS` are already defined there):

```python
# ---- per-turn gauges -------------------------------------------------------


class FakeClock:
    """A monotonic clock the fakes advance by hand, so phase timings are exact."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TimedHooks(FakeHooks):
    """Every prefill takes 2 s, every decode 5 s, on the fake clock."""

    def __init__(self, clock: FakeClock, **kw):
        super().__init__(**kw)
        self.clock = clock

    def prefill(self, cache, token_ids):
        self.clock.advance(2.0)
        super().prefill(cache, token_ids)

    def decode(self, cache, token_ids, max_tokens, on_delta=None):
        self.clock.advance(5.0)
        return super().decode(cache, token_ids, max_tokens, on_delta)


@pytest.fixture()
def clock(monkeypatch) -> FakeClock:
    import sous.engine.promptcache as promptcache

    c = FakeClock()
    monkeypatch.setattr(promptcache, "_clock", c)
    return c


def test_a_cold_non_trimmable_turn_splits_prefill_and_decode(clock):
    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    s = pc.stats()
    assert s["prefilled_tokens"] == len(STABLE_1)
    assert s["prefill_seconds"] == pytest.approx(2.0)  # one prefill to the anchor
    assert s["decode_seconds"] == pytest.approx(5.0)
    assert s["took_len"] == 0


def test_a_warm_turn_reports_the_slot_it_took_and_only_the_new_tokens(clock):
    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    pc.generate(STABLE_2, FULL_2, 16)
    s = pc.stats()
    assert s["took_len"] == len(STABLE_1)
    assert s["prefilled_tokens"] == len(STABLE_2) - len(STABLE_1)


def test_gauges_are_zeroed_when_a_turn_bypasses_the_cache(clock):
    """A render that is not a strict prefix of its own full prompt decodes
    cold without touching _run; the gauges must not keep the previous turn's."""
    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    assert pc.stats()["prefill_seconds"] > 0
    with pytest.warns(UserWarning, match="not the stable render"):
        pc.generate([1, 2, 3], [9, 9, 9], 16)
    s = pc.stats()
    assert (s["prefill_seconds"], s["decode_seconds"], s["prefilled_tokens"], s["took_len"]) == (0.0, 0.0, 0, 0)


def test_trimmable_reports_the_fused_pass_as_decode(clock):
    h = TimedHooks(clock, trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    s = pc.stats()
    assert s["prefill_seconds"] == 0.0 and s["decode_seconds"] == pytest.approx(5.0)
    assert s["prefilled_tokens"] == len(STABLE_1)


def test_the_probe_is_timed_and_the_bounds_recorded(clock):
    long = list(range(1, FORK_MIN_TOKENS * 3))
    tools_at, header_at = FORK_MIN_TOKENS + 10, FORK_MIN_TOKENS * 2

    def probe():
        clock.advance(0.5)
        return [header_at, tools_at]  # any order; the cache sorts

    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(long, long + [90, 91], 16, fork_at=probe)
    s = pc.stats()
    assert s["probe_seconds"] == pytest.approx(0.5)
    assert (s["bound_lo"], s["bound_hi"]) == (tools_at, header_at)
    assert s["forks"] == 2


def test_one_boundary_is_reported_as_the_upper_bound(clock):
    long = list(range(1, FORK_MIN_TOKENS * 2))
    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(long, long + [90, 91], 16, fork_at=[FORK_MIN_TOKENS + 5])
    s = pc.stats()
    assert (s["bound_lo"], s["bound_hi"]) == (0, FORK_MIN_TOKENS + 5)


def test_no_budget_means_no_bounds(clock):
    long = list(range(1, FORK_MIN_TOKENS * 2))
    h = TimedHooks(clock, trimmable=False)
    pc = PrefixCache(h)  # max_bytes=0: the probe is never resolved
    pc.generate(long, long + [90, 91], 16, fork_at=lambda: [FORK_MIN_TOKENS + 5])
    s = pc.stats()
    assert (s["bound_lo"], s["bound_hi"], s["probe_seconds"]) == (0, 0, 0.0)


def test_fork_copies_and_the_snapshot_count_as_prefill(clock):
    """The copy at a boundary and the pre-decode snapshot are prefill-side
    costs of building the cache, not decode."""
    long = list(range(1, FORK_MIN_TOKENS * 2))
    h = TimedHooks(clock, trimmable=False)

    def slow_copy(hooks, a):
        clock.advance(0.1)
        # The module-level fake, never h.copy_array: that method dispatches
        # back into copy_impl and would recurse forever.
        return copy_array(a)

    h.copy_impl = slow_copy
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(long, long + [90, 91], 16, fork_at=[FORK_MIN_TOKENS + 5])
    s = pc.stats()
    assert s["prefill_seconds"] > 4.0  # two prefills (2 s each) plus copies
    assert s["decode_seconds"] == pytest.approx(5.0, abs=0.5)  # decode plus restore copies


def test_gauges_fold_by_max_like_the_existing_ones():
    a = PromptCacheStats(prefill_seconds=3.0, took_len=10, bound_hi=500)
    a.add(PromptCacheStats(prefill_seconds=1.0, took_len=40, bound_hi=200))
    assert (a.prefill_seconds, a.took_len, a.bound_hi) == (3.0, 40, 500)
```

Also update the one existing whole-dict shape test, `test_stats_as_dict_reports_every_counter` (`tests/test_promptcache.py:873`; its expected literal at ~886 enumerates every key): add `"prefilled_tokens": 0, "took_len": 0, "bound_lo": 0, "bound_hi": 0, "probe_seconds": 0.0, "prefill_seconds": 0.0, "decode_seconds": 0.0` to the expected dict (the constructor call above it may stay as is — the new fields default). `_empty_stats()` at line 155 derives from `PromptCacheStats().as_dict()` and needs no edit.

And pin the disabled path the spec names: in the existing `test_disabled_never_reuses_and_never_counts`, add `assert pc.stats()["prefill_seconds"] == 0.0 and pc.stats()["took_len"] == 0` after its last assertion. (Implementation note for Step 4: `generate` returns before `_stats_for` on a disabled cache, so no gauge is ever written there and `begin_turn` is not needed on that path; the assertion is what keeps that true.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -q -k "gauge or clock or split or took or bound or probe or fused or bypass or snapshot_count or disabled" 2>&1 | tail -8`
Expected: FAIL — `AttributeError: module 'sous.engine.promptcache' has no attribute '_clock'`, then `KeyError: 'prefilled_tokens'`.

- [ ] **Step 3: Gauges and `begin_turn` on `PromptCacheStats`**

Replace `_GAUGES` and extend the dataclass:

```python
_GAUGES = frozenset(
    {
        "snapshot_bytes",
        "miss_lcp",
        "prefilled_tokens",
        "took_len",
        "bound_lo",
        "bound_hi",
        "probe_seconds",
        "prefill_seconds",
        "decode_seconds",
    }
)

# The clock the per-turn timers read. A module attribute rather than a bare
# time.monotonic so a test can swap in a hand-advanced one and assert exact
# phase durations; Slot.last_used keeps the real clock.
_clock = time.monotonic
```

Add to `PromptCacheStats` after `pressure_evictions`:

```python
    # Per-turn gauges, zeroed by begin_turn() at the top of every generate()
    # and assigned as the turn runs, so a turn that bypasses _run (cache off,
    # render not a strict prefix) reports zeros rather than the last turn's.
    prefilled_tokens: int = 0  # stable tokens this turn had to prefill
    took_len: int = 0  # length of the slot _take returned; 0 on a miss
    bound_lo: int = 0  # the probe's lower boundary (tools), 0 when absent
    bound_hi: int = 0  # the probe's upper boundary (header), 0 when absent
    probe_seconds: float = 0.0
    prefill_seconds: float = 0.0  # every hooks.prefill, plus fork copies and the snapshot
    decode_seconds: float = 0.0  # hooks.decode, plus the restore

    def begin_turn(self) -> None:
        self.prefilled_tokens = self.took_len = self.bound_lo = self.bound_hi = 0
        self.probe_seconds = self.prefill_seconds = self.decode_seconds = 0.0
```

Update the `add` docstring's gauge sentence to "except the gauges (`_GAUGES`)…".

- [ ] **Step 4: Timers in `generate`, `_run` and `_fork_boundaries`**

In `generate`, right after `stats = self._stats_for(owner)` inside the first `with self._lock:` block, add `stats.begin_turn()`. In the take branch (`if slot is not None:` at ~line 827), insert one line after `reuse = len(slot.held)` and change nothing else — the branch reads, after the edit:

```python
        if slot is not None:
            reuse = len(slot.held)
            stats.took_len = reuse  # the slot this turn took; stays 0 on a miss
            if warm is None:
                warm = slot.cache
            stats.hits += 1
            stats.reused_tokens += reuse
```

In `_run`, thread the timers. `_fork_boundaries` gains `stats` as its first parameter; it has two callers — `_run` (`promptcache.py:929`) becomes `self._fork_boundaries(stats, owner, stable_ids, fork_at, reuse)`, and `tests/test_engine_forkpoint.py:143–145` (the `Recording` fake's `generate`) becomes

```python
            fork_at = self._resolver._fork_boundaries(
                PromptCacheStats(), threading.current_thread(), list(stable_ids), fork_at, 0
            )
```

with `PromptCacheStats` added to that file's `from sous.engine.promptcache import (...)` line. Inside `_run`:

```python
        hooks = self._hooks
        anchor = len(stable_ids)
        entry_reuse = reuse
        stats.prefilled_tokens += anchor - entry_reuse
        published: list[Slot] = []
        for boundary in self._fork_boundaries(stats, owner, stable_ids, fork_at, reuse):
            ...
            started = _clock()
            hooks.prefill(cache, list(stable_ids[reuse:boundary]))
            reuse = boundary
            ...
            try:
                if self._make_room(price, protect=published):
                    copy = hooks.new_cache()
                    fork_copy(cache, copy, hooks.copy_array)
                    ...
            except Exception as e:
                ...
            finally:
                # The copy is part of building the cache, so it is prefill time.
                stats.prefill_seconds += _clock() - started
                copy = slot = None
```

(Keep every existing statement between; only the `started = _clock()` before the prefill and the `stats.prefill_seconds +=` line in the `finally` are new. The `finally` already exists — add the timer line above its `copy = slot = None`.)

Then the tail of `_run`:

```python
        published = []
        if all_trimmable(cache):
            started = _clock()
            text = hooks.decode(cache, list(full_ids[reuse:]), max_tokens, on_delta)
            trim_to(cache, anchor)
            stats.decode_seconds += _clock() - started
            return text
        started = _clock()
        hooks.prefill(cache, list(stable_ids[reuse:]))
        snap, nbytes = snapshot(cache, hooks.copy_array)
        stats.snapshot_bytes = nbytes
        stats.prefill_seconds += _clock() - started
        started = _clock()
        text = hooks.decode(cache, list(full_ids[anchor:]), max_tokens, on_delta)
        restore(cache, snap, hooks.copy_array)
        stats.decode_seconds += _clock() - started
        return text
```

Keep both existing comments in that tail exactly where they are (`# Everything rewinds, so (the rest of) prefill and decode fuse…` above the `all_trimmable` branch, ~line 982, and `# A recurrent layer cannot rewind, so stop at the anchor…` above the final prefill, ~line 988); only the `_clock()` reads and the `stats.*_seconds +=` lines are new.

In `_fork_boundaries`, add `stats: PromptCacheStats` as the first parameter and time the closure and record bounds:

```python
        if isinstance(fork_at, Sequence):
            candidates = list(fork_at)
        else:
            started = _clock()
            try:
                candidates = list(fork_at())
            except Exception as e:
                warnings.warn(...)  # unchanged
                return []
            finally:
                stats.probe_seconds += _clock() - started
        # The boundaries the probe verified inside this render, ascending —
        # recorded before the `reuse < b` filter so a warm turn that started
        # past one still reports where both sit (the log's lcp_region needs
        # them on the *next* turn's miss line).
        inside = sorted({b for b in candidates if b >= FORK_MIN_TOKENS and b < len(stable_ids)})
        if len(inside) >= 2:
            stats.bound_lo, stats.bound_hi = inside[0], inside[-1]
        elif inside:
            stats.bound_lo, stats.bound_hi = 0, inside[0]
        wanted = sorted({b for b in inside if reuse < b})
```

The early `if self.max_bytes <= 0: return []` stays first, so no probe and no bounds at budget 0.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py tests/test_engine_forkpoint.py -q 2>&1 | tail -3`
Expected: PASS (including the updated `as_dict` shape test and the fork-point tests through the new signature).

- [ ] **Step 6: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add src/sous/engine/promptcache.py tests/test_promptcache.py tests/test_engine_forkpoint.py
git commit -m "feat(engine): per-turn prompt-cache gauges for phase timing and slot geometry

A turn's seconds were one number. The cache now records, per turn, what
it prefilled, which slot it took, the two fork boundaries the probe
verified, and how long the probe, prefill (with copies and snapshot) and
decode (with restore) took — as max-folded gauges zeroed at the start of
every generate, so a turn that bypasses the cache reports nothing stale.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: hashes of the rendered tool array and system text

**Files:**
- Modify: `src/sous/gateway/convert.py` (`ChatRequest` at lines 70–80; `parse_messages_request` at 300–325)
- Test: `tests/test_gateway_convert.py`

**Interfaces:**
- Produces: `ChatRequest.tools_hash: str = ""`, `ChatRequest.system_hash: str = ""` — 8 lowercase hex characters (`blake2b`, `digest_size=4`) of `json.dumps(tools, sort_keys=True, separators=(",", ":"))` over the *converted* tool list and of the rendered system text; empty string when there are no tools / no system text. `convert._digest(text: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gateway_convert.py`:

```python
# --- content hashes: comparable across log lines, never reversible --------------


def test_hashes_are_eight_hex_chars_of_rendered_tools_and_system():
    req = parse_messages_request(_body(system="Be terse.", tools=[READ_TOOL]))
    assert len(req.tools_hash) == 8 and int(req.tools_hash, 16) >= 0
    assert len(req.system_hash) == 8 and int(req.system_hash, 16) >= 0
    assert req.tools_hash != req.system_hash


def test_hashes_are_stable_and_change_with_one_character():
    a = parse_messages_request(_body(system="Be terse.", tools=[READ_TOOL]))
    b = parse_messages_request(_body(system="Be terse.", tools=[READ_TOOL]))
    assert (a.tools_hash, a.system_hash) == (b.tools_hash, b.system_hash)
    c = parse_messages_request(_body(system="Be terse!", tools=[READ_TOOL]))
    assert c.system_hash != a.system_hash and c.tools_hash == a.tools_hash
    tweaked = {**READ_TOOL, "description": "Read a file."}
    d = parse_messages_request(_body(system="Be terse.", tools=[tweaked]))
    assert d.tools_hash != a.tools_hash and d.system_hash == a.system_hash


def test_hashes_are_empty_when_there_is_nothing_to_hash():
    req = parse_messages_request(_body())
    assert (req.tools_hash, req.system_hash) == ("", "")


def test_the_system_hash_is_over_the_rendered_text_after_volatile_stripping():
    """Two requests that differ only by the <total_tokens> marker render the
    same system text, so they must hash the same — the marker is what the
    stripping exists to hide from the cache."""
    a = parse_messages_request(_body(system="Be terse."))
    b = parse_messages_request(
        _body(system="Be terse.\n\n<total_tokens>1234 tokens left</total_tokens>")
    )
    assert a.system_hash == b.system_hash != ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_convert.py -q -k hash 2>&1 | tail -5`
Expected: FAIL — `AttributeError: 'ChatRequest' object has no attribute 'tools_hash'`.

- [ ] **Step 3: Implement**

Add `import hashlib` and `import json` (check: `json` may already be imported) to `convert.py`. Extend `ChatRequest`:

```python
    dropped_tool_types: list[str] = field(default_factory=list)
    # 8 hex chars of blake2b over the rendered tool array and system text:
    # enough to see "same tools, different system" across log lines, and
    # nothing a reader could turn back into either. Empty when absent.
    tools_hash: str = ""
    system_hash: str = ""
```

Add near `_invalid`:

```python
def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=4).hexdigest()
```

Replace the tail of `parse_messages_request`:

```python
    tools, dropped = chat_tools(body.get("tools"))
    rendered = chat_messages(body.get("system"), messages)
    system_text = rendered[0]["content"] if rendered and rendered[0]["role"] == "system" else ""
    return ChatRequest(
        model,
        rendered,
        tools,
        max_tokens,
        stream,
        dropped,
        tools_hash=_digest(json.dumps(tools, sort_keys=True, separators=(",", ":"))) if tools else "",
        system_hash=_digest(system_text) if system_text else "",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_convert.py -q 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/convert.py tests/test_gateway_convert.py
git commit -m "feat(gateway): hash the rendered tool array and system text per request

A miss's lcp says where two renders diverged but not what changed. Two
truncated blake2b digests, comparable across log lines and not
reversible, tell a changed tool array from a changed system text
without a byte of either leaving the process.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: `TurnRunner` measures the rest and carries it on `TurnResult`

**Files:**
- Modify: `src/sous/gateway/turn.py` (`TurnResult` at lines 60–74; `run()` at 95–205)
- Test: `tests/test_gateway_turn.py`

**Interfaces:**
- Consumes: gauge keys from Task 4 in the owner-scoped `prompt_cache_stats` dict (`prefilled_tokens`, `took_len`, `bound_lo`, `bound_hi`, `probe_seconds`, `prefill_seconds`, `decode_seconds`), counters `forks`, `evictions`, `pressure_evictions`.
- Produces on `TurnResult` (all with defaults, after `lcp`): `prefilled_tokens: int = 0`, `took_len: int = 0`, `forks: int = 0`, `evictions: int = 0`, `pressure_evictions: int = 0`, `load_seconds: float = 0.0`, `queue_seconds: float = 0.0`, `tokenize_seconds: float = 0.0`, `ttft_seconds: float | None = None`, `prefill_seconds: float = 0.0`, `decode_seconds: float = 0.0`, `bounds: tuple[int, int] = (0, 0)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gateway_turn.py`:

```python
class _SlowCount(ChunkedFakeEngine):
    """count_tokens costs 50 ms, so tokenize_s cannot be satisfied by the
    probe term alone."""

    def count_tokens(self, messages, tools):
        time.sleep(0.05)
        return super().count_tokens(messages, tools)


class _Gated(ChunkedFakeEngine):
    """Sets `entered` from inside generate(), i.e. once the caller holds the
    gateway lock — the handshake test_busy_gateway_gives_up_after_the_timeout
    already uses instead of a sleep that races the scheduler."""

    def __init__(self, script, delay):
        super().__init__(script, delay)
        self.entered = threading.Event()

    def generate(self, messages, tools, max_tokens, on_delta=None):
        self.entered.set()
        return super().generate(messages, tools, max_tokens, on_delta)


def test_a_turn_carries_its_phase_timings(tmp_path: Path):
    inner = _SlowCount(["Hel|lo|there"], delay=0.02)
    inner.stats = {
        "hits": 0, "fork_hits": 0, "reused_tokens": 0, "forks": 0, "evictions": 0,
        "pressure_evictions": 0, "prefilled_tokens": 0, "took_len": 0, "bound_lo": 0,
        "bound_hi": 0, "probe_seconds": 0.0, "prefill_seconds": 0.0, "decode_seconds": 0.0,
    }
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        inner.stats = {
            **inner.stats, "forks": 2, "evictions": 1, "pressure_evictions": 1,
            "prefilled_tokens": 321, "took_len": 0, "bound_lo": 100, "bound_hi": 200,
            "probe_seconds": 0.25, "prefill_seconds": 1.5, "decode_seconds": 4.0,
        }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    runner, _ = _runner(tmp_path, inner)
    result = runner.run(MSGS, [], 100, RecordingSink())
    assert result.queue_seconds >= 0 and result.queue_seconds < 0.5
    assert result.load_seconds >= 0
    # 0.05 s of count_tokens plus the 0.25 s probe gauge; the upper bound only
    # guards against a wildly wrong sum on a loaded runner.
    assert 0.30 <= result.tokenize_seconds < 0.80
    assert result.ttft_seconds is not None and 0.02 <= result.ttft_seconds < result.seconds
    assert (result.prefill_seconds, result.decode_seconds) == (1.5, 4.0)
    assert (result.prefilled_tokens, result.took_len, result.bounds) == (321, 0, (100, 200))
    assert (result.forks, result.evictions, result.pressure_evictions) == (2, 1, 1)


def test_counter_fields_are_deltas_not_totals(tmp_path: Path):
    """forks/evictions/pressure are this turn's, read as owner-scoped
    before/after deltas like reused_tokens already is."""
    inner = FakeEngine(["a", "b"])
    inner.stats = {"hits": 0, "fork_hits": 0, "reused_tokens": 0, "forks": 5, "evictions": 3, "pressure_evictions": 1}
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {**inner.stats, "forks": 6, "evictions": 3, "pressure_evictions": 1}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    runner, _ = _runner(tmp_path, inner)
    first = runner.run(MSGS, [], 100, RecordingSink())
    second = runner.run(MSGS, [], 100, RecordingSink())
    assert (first.forks, first.evictions) == (0, 0)
    assert (second.forks, second.evictions, second.pressure_evictions) == (1, 0, 0)


def test_queue_seconds_is_the_wait_for_the_gateway_lock(tmp_path: Path):
    inner = _Gated(["slow|turn", "fast"], delay=0.1)
    runner, _ = _runner(tmp_path, inner)
    results: dict[str, TurnResult] = {}

    def first():
        results["first"] = runner.run(MSGS, [], 100, RecordingSink())

    t = threading.Thread(target=first)
    t.start()
    assert inner.entered.wait(5)  # the first turn holds the lock
    results["second"] = runner.run(MSGS, [], 100, RecordingSink())
    t.join()
    assert results["first"].queue_seconds < 0.05
    # The first turn still has ≥ 0.1 s of scripted generation ahead of it when
    # the second arrives; the gate, not the clock, is what this test pins.
    assert results["second"].queue_seconds > 0.05


def test_a_turn_with_no_delta_has_no_ttft(tmp_path: Path):
    class Silent(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            return self._take(messages, tools, max_tokens)  # never calls on_delta

    runner, _ = _runner(tmp_path, Silent(["ok"]))
    result = runner.run(MSGS, [], 100, RecordingSink())
    assert result.ttft_seconds is None and result.output_tokens == 0


def test_load_seconds_is_paid_once(tmp_path: Path):
    inner = FakeEngine(["a", "b"])

    def slow_factory(model_id: str):
        time.sleep(0.05)
        return inner

    engines = EngineManager(_cfg(tmp_path), engine_factory=slow_factory)
    runner = TurnRunner(engines, _cfg(tmp_path))
    first = runner.run(MSGS, [], 100, RecordingSink())
    second = runner.run(MSGS, [], 100, RecordingSink())
    assert first.load_seconds >= 0.05 and second.load_seconds < 0.05
```

(`threading`, `time`, `pytest`, `EngineManager`, `TurnResult`, `ChunkedFakeEngine` are already imported at the top of this test module.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_turn.py -q -k "phase or deltas or queue_seconds or ttft or load_seconds" 2>&1 | tail -8`
Expected: FAIL — `AttributeError: 'TurnResult' object has no attribute 'queue_seconds'` (and friends).

- [ ] **Step 3: Extend `TurnResult` and `run()`**

Append to `TurnResult` after `lcp: int = 0`:

```python
    # Where the seconds went, and what the cache did — every one a count or a
    # duration; formatted by the gateway's turn line.
    prefilled_tokens: int = 0  # stable tokens the cache had to prefill
    took_len: int = 0  # length of the slot that served the turn; 0 on a miss
    forks: int = 0  # fork slots this turn published
    evictions: int = 0  # slots dropped while it ran, charged to this session
    pressure_evictions: int = 0  # of which by the pressure valve
    load_seconds: float = 0.0  # engines.get(): the model load, ≈0 when resident
    queue_seconds: float = 0.0  # wait for the gateway lock, before `seconds` starts
    tokenize_seconds: float = 0.0  # count_tokens plus the fork probe's renders
    ttft_seconds: float | None = None  # generate() entry → first delta; None if none came
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    bounds: tuple[int, int] = (0, 0)  # the probe's (tools, header) boundaries; 0 = absent
```

In `run()`:

```python
        queued = time.monotonic()
        if not self._lock.acquire(timeout=self._timeout):
            raise GatewayBusy(f"no generation slot within {self._timeout:.0f}s")
        if self._closing:
            self._lock.release()
            raise GatewayBusy("gateway is shutting down")
        started = time.monotonic()
        queue_seconds = started - queued
        try:
            with self._engines.lease():
                if abandoned is not None and abandoned.is_set():
                    raise TurnAbandoned
                loading = time.monotonic()
                engine = self._engines.get()
                load_seconds = time.monotonic() - loading
                session = self._session_for(engine)
                counting = time.monotonic()
                input_tokens = engine.count_tokens(messages, tools)
                tokenize_seconds = time.monotonic() - counting
                room = self._window - input_tokens
                if room <= 0:
                    raise PromptTooLong(input_tokens, self._window)
                before = engine.prompt_cache_stats(owner=session.thread)
                sink.started(input_tokens)
                final: Delta | None = None
                first_delta_at: float | None = None

                def on_delta(delta: Delta) -> None:
                    nonlocal final, first_delta_at
                    if first_delta_at is None:
                        first_delta_at = time.monotonic()
                    final = delta
                    sink.delta(delta)

                callback = ReplaySafe(on_delta) if sink.replay_safe else on_delta
                generating = time.monotonic()
                try:
                    text = session.generate(...)  # unchanged
                except GenerationStalled:
                    ...  # unchanged
                after = engine.prompt_cache_stats(owner=session.thread)
                cache_hit = after.get("hits", 0) > before.get("hits", 0)

                def delta_of(key: str) -> int:
                    return max(0, after.get(key, 0) - before.get(key, 0))

                return TurnResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=final.output_tokens if final else 0,
                    finish_reason=final.finish_reason if final else "stop",
                    cache_hit=cache_hit,
                    forked=after.get("fork_hits", 0) > before.get("fork_hits", 0),
                    reused_tokens=delta_of("reused_tokens"),
                    lcp=0 if cache_hit else after.get("miss_lcp", 0),
                    seconds=time.monotonic() - started,
                    # Gauges: assigned per turn by the cache, read directly like miss_lcp.
                    prefilled_tokens=after.get("prefilled_tokens", 0),
                    took_len=after.get("took_len", 0),
                    forks=delta_of("forks"),
                    evictions=delta_of("evictions"),
                    pressure_evictions=delta_of("pressure_evictions"),
                    load_seconds=load_seconds,
                    queue_seconds=queue_seconds,
                    tokenize_seconds=tokenize_seconds + after.get("probe_seconds", 0.0),
                    ttft_seconds=None if first_delta_at is None else first_delta_at - generating,
                    prefill_seconds=after.get("prefill_seconds", 0.0),
                    decode_seconds=after.get("decode_seconds", 0.0),
                    bounds=(after.get("bound_lo", 0), after.get("bound_hi", 0)),
                )
```

Keep every existing comment; the existing `reused_tokens=max(0, …)` expression becomes `delta_of("reused_tokens")`, same value.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_turn.py -q 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/turn.py tests/test_gateway_turn.py
git commit -m "feat(gateway): measure queue, load, tokenize and first-token time per turn

TurnResult carried one duration that started after the lock was taken
and folded the model load, tokenization, prefill and decode together.
It now carries each phase, the first-delta latency, the slot it took,
what it published and what was evicted under it — read owner-scoped
from the cache's per-turn gauges and counter deltas.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: the attributed turn line

**Files:**
- Modify: `src/sous/gateway/routes.py` (`_log_turn` at lines 645–658)
- Test: `tests/test_gateway_routes.py`

**Interfaces:**
- Consumes: `TurnResult` fields (Task 6), `ChatRequest.tools_hash`/`system_hash` (Task 5), `assembler.message_id`.
- Produces: the line format below; helpers `routes._lcp_region(lcp: int, lo: int, hi: int) -> str`, `routes._rate(count: int, seconds: float) -> str`, `routes._opt(value: float | None) -> str`.

Line (one line; wrapped here):

```
POST /v1/messages id=msg_… model=sous-local stream=1 status=200
 input_tokens=N output_tokens=N stop=end_turn
 cache=hit took=turn@82647 reused_tokens=82647 prefilled_tokens=1681
 forks=0 evicted=1 pressure=0
 load_s=0.0 queue_s=2.5 tokenize_s=1.1 ttft_s=9.8 prefill_s=7.9 decode_s=196.6
 prefill_tps=213 decode_tps=15 seconds=207.4
 tools=1b21cd75 system=9f8e7d6c
```

On a miss, `took=none` and, before `tools=`: ` lcp=62851 lcp_region=system bounds=[62298,62856]`. `took` is `fork@N` when `forked`, `turn@N` when a hit, `none` on a miss. `bounds=[]` when both are 0, `bounds=[b]` when only the upper is set. `prefill_tps`/`decode_tps` print with one decimal (the spec's example is `decode_tps=14.7`, and at ~15 tok/s the decimal is the signal), `-` when the divisor is 0. `ttft_s` is `-` when no delta came. Hashes print `-` when empty.

Deviation from the spec's field table, deliberate: `took_kind` and the `turn-moved@N` spelling are not in PR 1. Today `_take` removes every `turn` slot and leaves every `fork` slot, so the kind is exactly `result.forked` and a separate gauge could only disagree with it; the continuity spec introduces the retained turn slot and adds the copied/moved distinction with it. `turn@N` here means "the turn slot this session held". The spec's "moved-take" test case is therefore not owed by this PR.

- [ ] **Step 1: Write the failing tests**

First, in the existing `test_a_claimed_model_id_cannot_forge_or_bloat_a_log_line` (~line 1316), the attributed line is ~413 characters with its prefix, so the `len(line) < 200` bound no longer fits; the bound exists to catch an unbounded client string, not to pin a length, so replace the loop body with:

```python
    for line in lines:
        # The attributed turn line is ~410 chars with its prefix; the cap is
        # here to catch an unbounded client string, not to pin a length.
        assert len(line) < 500, line
        assert "A" * 20 not in line, line  # the oversized id never reaches it
        if "model=" in line:
            assert "model=-" in line, line
```

Then append to `tests/test_gateway_routes.py`:

```python
def _fields(line: str) -> dict[str, str]:
    """`key=value` tokens of a turn line; `bounds=[a,b]` stays one token."""
    return dict(tok.split("=", 1) for tok in line.split(" ") if "=" in tok)


_ALL_GAUGES = {
    "hits": 0, "fork_hits": 0, "reused_tokens": 0, "miss_lcp": 0, "forks": 0, "evictions": 0,
    "pressure_evictions": 0, "prefilled_tokens": 0, "took_len": 0, "bound_lo": 0, "bound_hi": 0,
    "probe_seconds": 0.0, "prefill_seconds": 0.0, "decode_seconds": 0.0,
}


def test_the_turn_line_attributes_a_hit(tmp_path: Path, capsys):
    inner = FakeEngine(["first", "second reply here"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {
                **_ALL_GAUGES, "hits": 1, "reused_tokens": 40, "prefilled_tokens": 6, "took_len": 40,
                "bound_lo": 10, "bound_hi": 30, "forks": 1, "evictions": 2, "pressure_evictions": 1,
                "prefill_seconds": 2.0, "decode_seconds": 4.0,
            }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    app = _app(tmp_path, inner)
    assert _post(app, _body(system="Be terse.", tools=[READ_TOOL])).status_code == 200
    r = _post(app, _body(system="Be terse.", tools=[READ_TOOL]))
    assert r.status_code == 200
    line = _turn_lines(capsys.readouterr().err)[1]
    f = _fields(line)
    assert f["id"] == r.json()["id"] and f["id"].startswith("msg_")
    assert f["cache"] == "hit" and f["took"] == "turn@40"
    assert f["reused_tokens"] == "40" and f["prefilled_tokens"] == "6"
    assert (f["forks"], f["evicted"], f["pressure"]) == ("1", "2", "1")
    assert f["prefill_s"] == "2.0" and f["decode_s"] == "4.0"
    # 6 prefilled tokens / 2.0 s; "second reply here" is 3 words = 3 output tokens / 4.0 s
    assert f["prefill_tps"] == "3.0" and f["decode_tps"] == "0.8"
    for key in ("load_s", "queue_s", "tokenize_s", "ttft_s", "seconds"):
        assert key in f, key
    assert len(f["tools"]) == 8 and len(f["system"]) == 8
    assert "lcp" not in f and "bounds" not in f


def test_the_turn_line_diagnoses_a_miss(tmp_path: Path, capsys):
    inner = FakeEngine(["reply"])
    inner.stats = {**_ALL_GAUGES, "miss_lcp": 25, "bound_lo": 10, "bound_hi": 30, "prefilled_tokens": 50}
    app = _app(tmp_path, inner)
    assert _post(app, _body()).status_code == 200
    f = _fields(_turn_lines(capsys.readouterr().err)[0])
    assert f["cache"] == "miss" and f["took"] == "none"
    assert (f["lcp"], f["lcp_region"], f["bounds"]) == ("25", "system", "[10,30]")
    assert (f["tools"], f["system"]) == ("-", "-")  # no tools, no system text in _body()


@pytest.mark.parametrize(
    ("lcp", "lo", "hi", "region"),
    [(5, 10, 30, "tools"), (15, 10, 30, "system"), (35, 10, 30, "conversation"),
     (15, 0, 30, "system"), (35, 0, 30, "conversation"), (15, 0, 0, "-")],
)
def test_lcp_region_places_the_divergence(lcp, lo, hi, region):
    from sous.gateway.routes import _lcp_region

    assert _lcp_region(lcp, lo, hi) == region


def test_rates_and_optional_fields_print_a_dash_when_undefined():
    from sous.gateway.routes import _opt, _rate

    assert _rate(100, 0.0) == "-" and _rate(100, 4.0) == "25.0" and _rate(59, 4.0) == "14.8"
    assert _opt(None) == "-" and _opt(1.234) == "1.2"


def test_a_fork_hit_prints_took_fork(tmp_path: Path, capsys):
    inner = FakeEngine(["a", "b"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {**_ALL_GAUGES, "hits": 1, "fork_hits": 1, "reused_tokens": 4000, "took_len": 4000}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    app = _app(tmp_path, inner)
    _post(app, _body())
    _post(app, _body())
    f = _fields(_turn_lines(capsys.readouterr().err)[1])
    assert f["cache"] == "fork" and f["took"] == "fork@4000"
```

The existing `test_the_log_says_cache_fork…` and `test_the_log_carries_lcp_only_on_a_miss` keep passing: `cache=` and `lcp=` keep their names and positions relative to each other (`lcp=` still appears only on a miss). Their `inner.stats` dicts lack the new keys; `after.get(..., 0)` in `TurnRunner.run` makes every missing key a zero, so they need no change.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_routes.py -q -k "turn_line or lcp_region or dash or took_fork" 2>&1 | tail -8`
Expected: FAIL — `KeyError: 'id'` / `ImportError: cannot import name '_lcp_region'`.

- [ ] **Step 3: Implement**

Near `_status_level` in `routes.py`:

```python
def _lcp_region(lcp: int, lo: int, hi: int) -> str:
    """Which part of the render a miss diverged in, given the two fork
    boundaries the probe verified: below the tools boundary the tool array
    changed, below the header the system text did, else the conversation.
    `-` when the probe never ran (no fork budget)."""
    if not hi:
        return "-"
    if lo and lcp < lo:
        return "tools"
    if lcp < hi:
        return "system"
    return "conversation"


def _rate(count: int, seconds: float) -> str:
    return "-" if seconds <= 0 else f"{count / seconds:.1f}"


def _opt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"
```

Replace `_log_turn`:

```python
    def _log_turn(
        self, chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
    ) -> None:
        cache = "fork" if result.forked else "hit" if result.cache_hit else "miss"
        took = "none" if cache == "miss" else f"{'fork' if cache == 'fork' else 'turn'}@{result.took_len}"
        lo, hi = result.bounds
        # A hit already says how much was reused; a miss says how far the
        # render agreed with the closest slot and in which region — the
        # number that tells a changed tool array from a changed system text.
        diag = ""
        if cache == "miss":
            bounds = ",".join(str(b) for b in (lo, hi) if b)
            diag = f" lcp={result.lcp} lcp_region={_lcp_region(result.lcp, lo, hi)} bounds=[{bounds}]"
        _log(
            f"POST /v1/messages id={assembler.message_id} model={_model_label(chat)} "
            f"stream={int(stream)} status=200 "
            f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
            f"stop={assembler.stop_reason} cache={cache} took={took} "
            f"reused_tokens={result.reused_tokens} prefilled_tokens={result.prefilled_tokens} "
            f"forks={result.forks} evicted={result.evictions} pressure={result.pressure_evictions} "
            f"load_s={result.load_seconds:.1f} queue_s={result.queue_seconds:.1f} "
            f"tokenize_s={result.tokenize_seconds:.1f} ttft_s={_opt(result.ttft_seconds)} "
            f"prefill_s={result.prefill_seconds:.1f} decode_s={result.decode_seconds:.1f} "
            f"prefill_tps={_rate(result.prefilled_tokens, result.prefill_seconds)} "
            f"decode_tps={_rate(result.output_tokens, result.decode_seconds)} "
            f"seconds={result.seconds:.1f}{diag} "
            f"tools={chat.tools_hash or '-'} system={chat.system_hash or '-'}"
        )
```

`assembler.message_id` is the `msg_` id minted in `messages()` and echoed in `message_start` and the non-streaming body, so the line joins to the client's transcript.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_routes.py -q 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/routes.py tests/test_gateway_routes.py
git commit -m "feat(gateway): attribute every served turn on its log line

The line said seconds and cache=; it could not say why a 289 s turn
took 289 s. It now carries the message id, the slot it took, what it
prefilled and published, what was evicted under it, every phase's
seconds and rate, and on a miss which region of the render diverged
between the two fork boundaries, plus hashes of the tool array and
system text. Counts, durations, hashes — never a byte of a body.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: one launchd log; `install-launchd` re-runs and folds the old file in

**Files:**
- Modify: `src/sous/cli.py` (`launchd_plist` at lines 255–274; `_cmd_install_launchd` at 511–525)
- Test: `tests/test_cli.py` (plist tests at lines 20–48)

**Interfaces:**
- Produces: `cli.launchd_plist` with both `StandardOutPath` and `StandardErrorPath` = `<log_dir>/daemon.log`; `cli._fold_legacy_stderr_log(data_dir: Path) -> bool`; `_cmd_install_launchd` sequence: `_bootout(LABEL)` → (if it unloaded) `_await_port_closed` → refuse if `_lock_is_held(data_dir)` → `_fold_legacy_stderr_log` → write plist → `launchctl bootstrap`.

- [ ] **Step 1: Update the plist tests and write the failing install tests**

In `tests/test_cli.py`, in `test_plist_is_valid_and_correct` and `test_plist_escapes_xml_special_characters`, change both `StandardErrorPath` assertions to expect `…/daemon.log` (same value as `StandardOutPath`). Extend the module's import to `from sous.cli import _BOOTOUT_NOT_LOADED, LABEL, launchd_plist`. Then add:

```python
def _install_env(tmp_path, monkeypatch, *, bootout_code: int, lock_held: bool):
    """install-launchd with launchctl, the lock probe and the plist path faked."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    calls: dict = {"bootout": [], "run": []}
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/opt/sous")
    monkeypatch.setattr(cli, "_bootout", lambda label: (calls["bootout"].append(label), bootout_code)[1])
    monkeypatch.setattr(cli, "_await_port_closed", lambda port, seconds: True)
    monkeypatch.setattr(cli, "_lock_is_held", lambda data_dir: lock_held)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check: calls["run"].append(cmd))
    return cli, plist, calls


def test_install_launchd_boots_out_then_writes_and_bootstraps(tmp_path, capsys, monkeypatch):
    """Re-running must work: bootstrap fails on an already-loaded label, and
    a plist that changed (both streams now go to daemon.log) reloads only if
    the old job is out first."""
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    cli.main(["install-launchd"])
    assert calls["bootout"] == [cli.LABEL]
    data = plistlib.loads(plist.read_bytes())
    assert data["StandardOutPath"] == data["StandardErrorPath"] == str(tmp_path / "daemon.log")
    assert calls["run"] and calls["run"][0][:2] == ["launchctl", "bootstrap"]
    assert calls["run"][0][-1] == str(plist)


def test_install_launchd_when_nothing_was_loaded(tmp_path, capsys, monkeypatch):
    """A first install: bootout reports "not loaded" and that is fine."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=_BOOTOUT_NOT_LOADED, lock_held=False
    )
    cli.main(["install-launchd"])
    assert calls["bootout"] == [cli.LABEL]  # tried, and tolerated "not loaded"
    assert plist.exists() and calls["run"]


def test_install_launchd_refuses_while_a_daemon_holds_the_lock(tmp_path, capsys, monkeypatch):
    """A hand-started daemon still writing daemon.err.log must not have the
    file appended and unlinked under it; the user stops it first."""
    cli, plist, calls = _install_env(
        tmp_path, monkeypatch, bootout_code=_BOOTOUT_NOT_LOADED, lock_held=True
    )
    (tmp_path / "daemon.err.log").write_text("old\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1
    assert "sous stop" in capsys.readouterr().out
    assert not plist.exists() and (tmp_path / "daemon.err.log").exists() and not calls["run"]


def test_install_launchd_folds_the_legacy_stderr_log_into_daemon_log(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    (tmp_path / "daemon.log").write_bytes(b"out-1\n")
    (tmp_path / "daemon.err.log").write_bytes(b"err-1\nerr-2\n")
    cli.main(["install-launchd"])
    assert (tmp_path / "daemon.log").read_bytes() == b"out-1\nerr-1\nerr-2\n"
    assert not (tmp_path / "daemon.err.log").exists()
    assert "folded daemon.err.log into daemon.log" in capsys.readouterr().out


def test_install_launchd_without_a_legacy_log_touches_nothing(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=0, lock_held=False)
    cli.main(["install-launchd"])
    assert not (tmp_path / "daemon.log").exists() or (tmp_path / "daemon.log").read_bytes() == b""
    assert "folded" not in capsys.readouterr().out


def test_install_launchd_keeps_the_plist_when_bootout_really_fails(tmp_path, capsys, monkeypatch):
    cli, plist, calls = _install_env(tmp_path, monkeypatch, bootout_code=5, lock_held=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["install-launchd"])
    assert exc.value.code == 1 and not plist.exists() and not calls["run"]
```

`_cfg(tmp_path, port)` at line 88 of the test module returns a config whose `data_dir` is `tmp_path`, so the log files above sit in `tmp_path`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -q -k "plist or install" 2>&1 | tail -8`
Expected: the plist tests FAIL on `daemon.err.log` vs `daemon.log`; `…boots_out_then_writes…` and `…when_nothing_was_loaded` FAIL on the missing `bootout` call; `…refuses_while_a_daemon_holds_the_lock` FAILS because today's command never checks the lock (no `SystemExit`); `…folds_the_legacy_stderr_log…` FAILS on the missing `folded` line. `…without_a_legacy_log_touches_nothing` is a negative test and passes before and after.

- [ ] **Step 3: Implement**

In `launchd_plist`, replace the two path entries with:

```python
            # Both streams to one file, in write order. The stderr file was
            # named daemon.err.log and held everything the daemon said — Python
            # logging, the gateway's lines, warnings, the SDK's handler all
            # write to stderr — while daemon.log held the banner. One file,
            # and the level on every line (sous.logs) says what is an error.
            "StandardOutPath": f"{log_dir}/daemon.log",
            "StandardErrorPath": f"{log_dir}/daemon.log",
```

Add above `_cmd_install_launchd`:

```python
def _fold_legacy_stderr_log(data_dir: Path) -> bool:
    """Append a pre-one-file `daemon.err.log` onto `daemon.log` and remove it.

    The history is worth keeping; the name is not — it was launchd's stderr
    path, not an error log, and it will never be written again. Byte for
    byte, so nothing in it is reformatted. Only ever called with no daemon
    holding the lock: an open descriptor would keep writing into the unlinked
    inode and those lines would be lost."""
    legacy = data_dir / "daemon.err.log"
    if not legacy.exists():
        return False
    target = data_dir / "daemon.log"
    with legacy.open("rb") as src, target.open("ab") as dst:
        shutil.copyfileobj(src, dst)
    legacy.unlink()
    print(f"folded {legacy.name} into {target.name}")
    return True
```

Replace `_cmd_install_launchd`:

```python
def _cmd_install_launchd() -> None:
    config = load_config()
    exe = shutil.which("sous") or sys.argv[0]
    plist_path = _plist_path()
    # Out first, unconditionally: `bootstrap` fails on a label that is already
    # loaded, so a re-run (a new plist — this release moved both log streams
    # to daemon.log — or a moved executable) never applied. Same rule and
    # same helper as uninstall: anything but "unloaded" or "was not loaded"
    # leaves a live KeepAlive job, and writing a plist over it would report
    # success for a daemon still running the old one.
    code = _bootout(LABEL)
    if code not in (0, _BOOTOUT_NOT_LOADED):
        print(f"sous: launchctl bootout failed (exit {code}); not touching {plist_path}")
        raise SystemExit(1)
    if code == 0:
        _await_port_closed(config.server_port, _UNLOAD_GRACE_SECONDS)
    if _lock_is_held(config.data_dir):
        # A daemon launchd did not start — `sous serve` by hand — is still
        # writing its log files; folding one under it would lose lines.
        print("sous: a daemon is running and holds the lock; stop it first: sous stop")
        raise SystemExit(1)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    _fold_legacy_stderr_log(config.data_dir)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(launchd_plist(exe, config.data_dir))
    print(f"wrote {plist_path}")
    cmd = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)]
    try:
        subprocess.run(cmd, check=True)
        print("daemon loaded; it will start at login and stay alive")
    except subprocess.CalledProcessError, FileNotFoundError:
        print("could not load automatically; run manually:")
        print("  " + " ".join(cmd))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -q 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Suite, types, lint; commit**

```bash
set -o pipefail; uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
git add src/sous/cli.py tests/test_cli.py
git commit -m "feat(cli): one daemon.log for both streams; install-launchd re-runs and folds the old file

daemon.err.log was launchd's stderr path and held everything the daemon
said; daemon.log held a banner. Both streams now go to daemon.log, with
the level on each line saying what is an error. install-launchd boots an
existing job out before bootstrapping (a re-run used to fail on the
loaded label), refuses while a hand-started daemon holds the lock, and
appends an existing daemon.err.log onto daemon.log before removing it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: documentation

**Files:**
- Modify: `README.md` ("Managing the daemon" at lines 107–124; the turn-line paragraph in "Gateway mode" at lines 265–280, beginning "Each `/v1/messages` turn served locally logs one metadata-only line"), `CONTRIBUTING.md` (lines 147–148 and 156–160), `CLAUDE.md` (the gateway bullet under "Gotchas" and the "Security boundary" paragraph on logging)

- [ ] **Step 1: README — "Managing the daemon"**

After the `sous stop` paragraph, add:

```markdown
The daemon writes one log, `~/.sous/daemon.log`, both streams. Every line
sous emits reads `2026-09-10T19:26:14.025Z INFO sous.gateway: …` — UTC
timestamp with milliseconds, then `INFO` (served or forwarded), `WARNING`
(a request sous refused: a 4xx, a 529, a client gone while queued, tools it
had to drop) or `ERROR` (a failure sous produced), then which part spoke.
Library lines (`mcp.…`, `uvicorn.error`, `huggingface_hub…`) take the same
shape, and `warnings.warn` arrives as `WARNING py.warnings`. Earlier
releases split stdout and stderr into `daemon.log` and `daemon.err.log`;
re-run `sous install-launchd` once — it boots the old job out, appends
`daemon.err.log` onto `daemon.log`, removes it, and loads the new plist.
```

- [ ] **Step 2: README — the turn line**

Replace the paragraph that begins "Each `/v1/messages` turn served locally logs one metadata-only line" (through "…plus one line naming the Anthropic tool *types* it dropped, when any.") with the following — the whole `~~~` block is the replacement text, including the inner triple-backtick code fence:

~~~markdown
Each `/v1/messages` turn served locally logs one metadata-only line, for
example (wrapped here):

```
2026-09-10T19:26:14.025Z INFO sous.gateway: POST /v1/messages id=msg_… model=sous-local
  stream=1 status=200 input_tokens=84335 output_tokens=2887 stop=end_turn
  cache=hit took=turn@82647 reused_tokens=82647 prefilled_tokens=1681 forks=0 evicted=1 pressure=0
  load_s=0.0 queue_s=2.5 tokenize_s=1.1 ttft_s=9.8 prefill_s=7.9 decode_s=196.6
  prefill_tps=212.8 decode_tps=14.7 seconds=207.4 tools=1b21cd75 system=9f8e7d6c
```

`id` is the response's message id (the same one in `message_start`, so the
line joins to Claude Code's transcript). `cache` is `hit` (this
conversation's own slot), `fork` (a copy of a shared boundary — ~45–56K
reused tokens is a tools fork, ~57K a header fork) or `miss`; `took` names
the slot and its length. `prefilled_tokens` is what the turn had to
prefill; `forks`/`evicted`/`pressure` are what it published and what was
dropped under it (by budget, or by the pressure valve). The phases:
`load_s` (a model load, ≈0 when resident), `queue_s` (the wait for the
gateway lock — Claude Code's small background calls hold it too),
`tokenize_s`, `ttft_s` (to the first token), `prefill_s`, `decode_s`, with
`prefill_tps`/`decode_tps` as rates; `seconds` runs from the lock to the
result, so client-visible latency is `queue_s` more. On a miss the line
adds `lcp=` (how many leading tokens the render shared with the closest
resident slot), `lcp_region=` (`tools` below the tools boundary — the tool
array changed; `system` below the header — the system text did; else
`conversation`) and `bounds=[tools,header]`, the boundaries the probe
verified. `tools=` and `system=` are 8-hex-character hashes of the rendered
tool array and system text: comparable across lines, not reversible.
Refused requests log the same way at `WARNING` with `status=` and
`seconds=`; a locally served `count_tokens` logs its `input_tokens`,
`load_s` and `seconds`; the engine logs `model loaded in N s` when it loads.
One more line names the Anthropic tool *types* a turn dropped, when any.
~~~

Keep the sentences that follow in the original paragraph (forwarded-request line, uvicorn access log off, "Nothing else is logged…", the error-shape sentences) unchanged, except: in the forwarded-request sentence, add that a `502`/`504`/`499` the forwarder itself produced (unreachable upstream, timeout, client gone) logs at `ERROR`/`WARNING`, while a status the upstream answered logs at `INFO` whatever the number.

- [ ] **Step 3: CONTRIBUTING**

Both sentences are line-wrapped in the file, so edit them across the wrap. Lines 147–148 read

```
re-prefilling the whole conversation. Watch `~/.sous/daemon.log` for the
`sous gateway:` lines. The main loop's turns should now report `cache=hit`
```

→ make them

```
re-prefilling the whole conversation. Watch `~/.sous/daemon.log` for the
`INFO sous.gateway:` lines (both streams land there; every line carries a
timestamp and a level). The main loop's turns should now report `cache=hit`
```

Lines 157–160 read

```
`~/.sous/daemon.log` is enough: every forwarded request logs an `upstream
<METHOD> <path> model=<id> status=<code>` line, every local turn a `POST
/v1/messages model=sous-local ...` line. A hybrid session should show the
```

→ make them

```
`~/.sous/daemon.log` is enough: every forwarded request logs an `INFO
sous.gateway: upstream <METHOD> <path> model=<id> status=<code>` line, every
local turn a `POST /v1/messages id=… model=sous-local …` line with its phase
timings. A hybrid session should show the
```

Re-wrap the touched paragraphs to the file's ~79-column width.

- [ ] **Step 4: CLAUDE.md**

In the "Security boundary" paragraph on the gateway, after "never logs a request body, header value or query string.", add: "Its lines go through `sous.logs` (one timestamped, levelled shape on the root handler); `_log_turn` runs on the event loop, so nothing that blocks (a database write) may ever be added to it — do such work on the turn's thread." In the Gotchas list, add a bullet:

```markdown
- Prompt-cache per-turn gauges (`PromptCacheStats.begin_turn`, `_GAUGES`) are
  assigned per turn and read back directly from the owner-scoped `after`
  snapshot in `TurnRunner.run` — never as `after − before`. Counters
  (`forks`, `evictions`, `pressure_evictions`, `reused_tokens`) are deltas.
  Timers read `promptcache._clock` so tests can drive them; `Slot.last_used`
  keeps the real clock.
```

- [ ] **Step 5: Verify the docs render and nothing else changed**

Run: `git diff --stat` — only the three files. `uv run ruff check . && uv run ruff format --check .` (unchanged code, sanity).

- [ ] **Step 6: Commit**

```bash
git add README.md CONTRIBUTING.md CLAUDE.md
git commit -m "docs: the one daemon log and the attributed turn line

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: full verification and the pull request

**Files:** none new.

- [ ] **Step 1: The four CI jobs, exactly**

```bash
set -o pipefail
uv run pytest -m "not model" -q 2>&1 | tail -3
uv run ty check
uv run ruff check . && uv run ruff format --check .
uv lock --check
```
Expected: all green. The suite has one known flaky test under load (`tests/test_commands.py::test_timeout_kills_descendants_before_recording`); if it fails once, run it alone ten times in a loop on exit codes before deciding anything.

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin feat/gateway-observability-pr1
```

PR title: `feat(gateway): attribute every turn; one timestamped daemon log`. Body (write to a scratch file and pass with `--body-file`): what a turn line now says and why each field exists (the 13m14s session as the motivating case: two 200 s misses, two 90 s re-prefills, a 197 s decode, none of it distinguishable before), the log-stream change and the `install-launchd` migration (re-run once; the old file is folded in and removed), the invariants kept (no bodies, hashes of rendered text only, loopback), and the verification list. End with the standard footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and no session link.

- [ ] **Step 3 (orchestrator, not a subagent): the real-daemon gate**

After CI is green and before merge: `uv tool install --force --reinstall .` from the branch checkout; stop the unmanaged daemon (`sous stop`) and restart it detached (the fork+setsid launcher; plain `nohup … &` dies with the shell); run one `sous claude` session with a subagent of at least two turns; confirm in `~/.sous/daemon.log` (note: the unmanaged launcher chooses the file — point it at `daemon.log` for both streams) that every line carries the prefix, that the turn lines parse, and that for each real turn `seconds + queue_s` matches the transcript's assistant-to-assistant gap within 0.5 s. Record the lines (they contain no bodies) in the PR. Then merge, reinstall from `main`, restart the daemon, delete the branch.
