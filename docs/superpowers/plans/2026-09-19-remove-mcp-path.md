# Remove the MCP Delegate Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the MCP server, worker loop, task queue and application-level sandbox, and reshape sous into one daemon that serves Claude Code's subagents from a local MLX model, in a single 0.7.0 release.

**Architecture:** The daemon becomes a plain Starlette app (monitor routes, then the Anthropic-shaped endpoint routes), the idle-unload sweep moves into `EngineManager`, config collapses to `[model]` (engine) and `[server]` (endpoint), and the status document loses its task fields. `sous tune` keeps grading candidates through a tune-owned loop and scratch-directory executor that reuse the worker's prompt and tool schema as fixtures, so grades stay comparable.

**Tech Stack:** Python 3.14, Starlette + uvicorn + sse-starlette (already direct deps), mlx / mlx-lm / mlx-vlm (function-local imports), Textual (TUI), pytest, ty, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-19-remove-mcp-path-design.md`

## Global Constraints

- Python `>=3.14`. `except A, B:` without parentheses is valid PEP 758 syntax in this repo; never "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- `mlx`, `mlx_lm`, `mlx_vlm` imports stay function-local.
- Every thread that touches mlx calls `engine.base.release_mlx_thread_state()` once before it exits; a thread that released must never run a model load.
- Tests never touch the real `~/.sous`: always pass tmp_path-based `config_path` / `data_dir`.
- Never edit anything under `docs/superpowers/**` except adding this plan's files.
- No spec, plan, task or step numbers in code or test comments; state the fact instead.
- Comments explain non-obvious *why*, never what the code does.
- Conventional Commits, imperative lowercase subject, *why* in the body. Every commit ends with the trailer `Co-Authored-By: Claude <noreply@anthropic.com>`. Never put a person's name or home path in a commit, file, PR or issue.
- The four CI checks, run before every commit: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. `slow`-marked tests run; they are never skipped or mocked.
- Package name stays `sous-mcp`; CLI stays `sous`; version becomes `0.7.0` in the last task only.
- Served window floor: `MIN_CONTEXT_TOKENS = 48 * 1024` (49152). Served window default: `131072`. Idle sweep interval: `15.0` s.
- Renames fixed by the spec: `src/sous/gateway/` → `src/sous/api/` (logger `sous.api`); `Gateway` → `Endpoint`, `mount_gateway` → `mount_endpoint`; `SousService` → `Daemon`; `GATEWAY_MIN_CONTEXT_TOKENS` → `MIN_CONTEXT_TOKENS`; `GATEWAY_DEFAULT_UPSTREAM` → `DEFAULT_UPSTREAM`; `tests/test_gateway_*.py` → `tests/test_api_*.py`.
- Spec deviation, agreed here: the payload pin is a content hash of the tool schema and system prompt, not a rendered token count, because CI has no tokenizer.
- Spec deviation, agreed here: `run_loop` takes its budget, tools and transcript as explicit keyword arguments instead of the spec's `(task, engine, window, *, python, transcript)` sketch; the interpreter is selected by the runner's existing `interpreter_first`, so the loop never sees `python`.

## File Structure

| Path | Responsibility after this plan |
|---|---|
| `src/sous/engine/base.py` | unchanged algorithms; `EngineManager` gains `start_idle_sweep` / `stop_idle_sweep`; `default_engine_factory` reserves `config.max_context_tokens` |
| `src/sous/engine/window.py` (new) | `TOKEN_STEP`, `kv_bytes_per_token`, `native_max_tokens` (moved from `context.py`) |
| `src/sous/tune/payload.py` (new) | `TOOLS`, `TOOLSET`, `SYSTEM_TEMPLATE`, `build_system_prompt` — the fixtures bench and the suite loop feed the model |
| `src/sous/tune/suite/tools.py` (new) | `ScratchTools`: the eight tools over a scratch project, `ToolError` |
| `src/sous/tune/suite/loop.py` (new) | `Budget`, `LoopResult`, `Transcript`, `run_loop` |
| `src/sous/tune/suite/runner.py` | grades through `run_loop`; no store, no worker, no denier |
| `src/sous/tune/{bench,candidates,arms,decide,daemon}.py` | one window, readiness from holders and in-flight turns |
| `src/sous/server.py` | `Daemon` (status document), `create_server` → Starlette app, `serve`, `main`; no MCP, no worker |
| `src/sous/monitor.py` | `monitor_routes(...)` returns the `/sous/*` routes; `_busy` without the queue |
| `src/sous/api/{routes,turn,convert,response,upstream}.py` | the endpoint (renamed package); `Endpoint.routes()`, `mount_endpoint` |
| `src/sous/config.py` | `[model]` + `[server]`, obsolete-key warning, `MIN_CONTEXT_TOKENS`, `DEFAULT_UPSTREAM` |
| `src/sous/cli.py` | `serve`, `claude`, `status` (plain-text status), `top`, `statusline`, `stop`, `install-launchd`, `uninstall-launchd`, `tune` |
| `src/sous/tui.py` | turns only; no task rows |
| `src/sous/protocol.py` | `ToolSet`, `ToolCall`, `ParseError`, `parse_tool_calls(text, toolset)`; no worker schema |
| deleted | `src/sous/{tasks,toolexec,worker,proxy,context}.py`, `server.json`, `.github/workflows/publish-mcp-registry.yml`, `skills/`, `scripts/e2e_smoke.py`, `tests/{test_commands,test_confinement,test_tasks,test_worker,test_proxy,test_context}.py`, `tests/daemon_stub.py` |
| `scripts/api_smoke.py` (new) | the post-dependency-change check: one served turn against a tiny real model |
| `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/tuning.md`, `pyproject.toml` | rewritten around the endpoint |

Task order matters: every task leaves the four CI checks green, and later tasks assume earlier ones landed.

---

### Task 1: Idle-unload sweep owned by EngineManager

The worker loop is today the only caller of `unload_if_idle()` (`src/sous/worker.py:462`). Give the engine manager its own sweep thread so the worker can go.

**Files:**
- Modify: `src/sous/engine/base.py` (`EngineManager.__init__` at 532-567; add three methods after `unload_if_idle` at 796-806)
- Test: `tests/test_engine_base.py` (uses the existing `_held_manager` fixture at 434-449)

**Interfaces:**
- Produces: `IDLE_SWEEP_SECONDS = 15.0` (module constant); `EngineManager.start_idle_sweep(interval_s: float = IDLE_SWEEP_SECONDS) -> None`; `EngineManager.stop_idle_sweep(timeout: float = 5.0) -> None`. Task 6 starts the sweep from the daemon's lifespan.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py` (the file already imports `threading`, `time`, `EngineManager`, `SousConfig` and defines `_held_manager`, `_Clock`, `_Liveness`; add `import sous.engine.base as base` if not present):

```python
def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_idle_sweep_unloads_an_idle_model_on_its_own_thread():
    mgr, created, live, clock = _held_manager(idle_minutes=1)
    mgr.get()
    assert mgr.status()["loaded"]
    clock.now += 61
    mgr.start_idle_sweep(interval_s=0.01)
    try:
        assert _wait_until(lambda: not mgr.status()["loaded"])
        assert any(t.name == "sous-idle-sweep" for t in threading.enumerate())
    finally:
        mgr.stop_idle_sweep()
    assert not any(t.name == "sous-idle-sweep" for t in threading.enumerate())


def test_the_idle_sweep_keeps_a_fresh_model_and_starts_once():
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    mgr.get()
    mgr.start_idle_sweep(interval_s=0.01)
    mgr.start_idle_sweep(interval_s=0.01)
    try:
        time.sleep(0.05)
        assert mgr.status()["loaded"]
        assert sum(t.name == "sous-idle-sweep" for t in threading.enumerate()) == 1
    finally:
        mgr.stop_idle_sweep()
    mgr.stop_idle_sweep()  # idempotent when nothing runs


def test_the_idle_sweep_releases_its_mlx_state_once_on_exit(monkeypatch):
    released: list[str] = []
    monkeypatch.setattr(base, "release_mlx_thread_state", lambda: released.append("x"))
    mgr, created, live, clock = _held_manager(idle_minutes=30)
    mgr.start_idle_sweep(interval_s=0.01)
    time.sleep(0.03)
    mgr.stop_idle_sweep()
    assert released == ["x"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k idle_sweep -v`
Expected: FAIL with `AttributeError: 'EngineManager' object has no attribute 'start_idle_sweep'`

- [ ] **Step 3: Implement the sweep**

In `src/sous/engine/base.py`, add the constant near the other module constants (top of file, after imports):

```python
# The idle sweep's cadence. Each tick is a lock and a clock comparison, so the
# only thing it buys by being slow is nothing; idle_unload_minutes is minutes.
IDLE_SWEEP_SECONDS = 15.0
```

In `EngineManager.__init__`, after `self._version = 0`:

```python
        self._sweep: threading.Thread | None = None
        self._sweep_stop: threading.Event | None = None
```

After `unload_if_idle` (line ~806), add:

```python
    def start_idle_sweep(self, interval_s: float = IDLE_SWEEP_SECONDS) -> None:
        """Retire an idle model without anyone asking: one daemon thread,
        `sous-idle-sweep`, calls unload_if_idle() every `interval_s` seconds
        until stop_idle_sweep(). A second start while one runs is a no-op."""
        with self._lock:
            if self._sweep is not None:
                return
            stop = threading.Event()
            thread = threading.Thread(
                target=self._run_idle_sweep,
                args=(interval_s, stop),
                name="sous-idle-sweep",
                daemon=True,
            )
            self._sweep, self._sweep_stop = thread, stop
            thread.start()

    def stop_idle_sweep(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread, stop = self._sweep, self._sweep_stop
            self._sweep, self._sweep_stop = None, None
        if thread is None or stop is None:
            return
        stop.set()
        thread.join(timeout)

    def _run_idle_sweep(self, interval_s: float, stop: threading.Event) -> None:
        # An unload frees mlx arrays on this thread, and a thread that touched
        # mlx must release its state once before it exits (ml-explore/mlx#4327).
        # This thread never loads, so releasing on the way out is enough.
        try:
            while not stop.wait(interval_s):
                try:
                    self.unload_if_idle()
                except Exception as e:  # noqa: BLE001 — one bad tick must not end the sweep
                    _logger.warning(f"idle sweep failed ({type(e).__name__}: {e})")
        finally:
            release_mlx_thread_state()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -v`
Expected: PASS (all, including the three new ones)

- [ ] **Step 5: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): idle-unload sweep on a thread the engine manager owns" -m "The worker loop is the only caller of unload_if_idle today, and it is going away. A 15 s tick is a lock and a clock read, which also retires the twice-a-second SQLite poll the sweep rode on." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Window helpers and the tune payload

Move the context-window arithmetic tune needs out of `context.py` (which dies with the worker) and give tune its own copy of the eight-tool schema and system prompt.

**Files:**
- Create: `src/sous/engine/window.py`, `src/sous/tune/payload.py`, `tests/test_engine_window.py`, `tests/test_tune_payload.py`
- Modify: `src/sous/context.py` (lines 25-27 and 45-84 become an import), `src/sous/tune/candidates.py:13`, `src/sous/tune/bench.py` (lines 15, 143, 149, 195, 362, 370), `tests/test_context.py` (move tests at lines 36-76)

**Interfaces:**
- Produces: `sous.engine.window.TOKEN_STEP: int`, `kv_bytes_per_token(model_config: dict) -> int | None`, `native_max_tokens(model_config: dict) -> int | None`; `sous.tune.payload.TOOLS: list[dict]`, `TOOLSET: ToolSet`, `SYSTEM_TEMPLATE: str`, `build_system_prompt(project_root: Path) -> str`.
- Consumes: `sous.protocol.ToolSet.from_tools(tools: list[dict], *, strict: bool = True) -> ToolSet`.

- [ ] **Step 1: Create `tests/test_engine_window.py` from the five helper tests**

Move the five test functions at `tests/test_context.py` lines 36-76 (`test_kv_bytes_counts_only_full_attention_layers`, `test_kv_bytes_plain_gqa_uses_all_layers`, `test_kv_bytes_derives_head_dim_when_absent`, `test_kv_bytes_none_when_shape_unknown`, `test_native_max_nested_and_flat`) verbatim into a new `tests/test_engine_window.py` whose imports are:

```python
from sous.engine.window import kv_bytes_per_token, native_max_tokens
```

Delete those five functions (and any now-unused import) from `tests/test_context.py`.

- [ ] **Step 2: Write the failing payload tests**

`tests/test_tune_payload.py`:

```python
import hashlib
import json
from pathlib import Path

from sous.tune import payload


def test_the_payload_names_the_eight_tools_in_order():
    names = [t["function"]["name"] for t in payload.TOOLS]
    assert names == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "run_command",
        "finish",
    ]
    assert payload.TOOLSET.names == frozenset(names)


def test_the_payload_is_pinned():
    # bench numbers and suite grades are only comparable across runs while
    # the model sees the same schema and prompt; a change here is deliberate
    # and re-baselines both.
    digest = hashlib.sha256(
        (json.dumps(payload.TOOLS, sort_keys=True) + payload.SYSTEM_TEMPLATE).encode()
    ).hexdigest()
    assert digest == "PIN"


def test_the_system_prompt_lists_the_root_and_its_entries(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "README.md").write_text("x")
    text = payload.build_system_prompt(tmp_path)
    assert f"Project root: {tmp_path}" in text
    assert "README.md\npkg/" in text
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_window.py tests/test_tune_payload.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sous.engine.window'` / `'sous.tune.payload'`

- [ ] **Step 4: Create `src/sous/engine/window.py`**

```python
"""Context-window arithmetic from a model's config.json: what one token of
KV cache costs, and the length the model was trained to attend to."""

from __future__ import annotations

# mlx-lm's KVCache grows its buffers in 256-token steps; an unaligned window
# ends in a partially-usable step.
TOKEN_STEP = 256


def _text_config(model_config: dict) -> dict:
    # VLMs (the default model included) nest the language model's shape under
    # text_config; text-only models keep it at the top level.
    return model_config.get("text_config", model_config)
```

then paste `kv_bytes_per_token` (docstring and body, `src/sous/context.py` lines 51-80) and `native_max_tokens` (lines 83-84) verbatim below it.

- [ ] **Step 5: Point `context.py` and `candidates.py` at the new module**

In `src/sous/context.py`: delete lines 25-27 (`TOKEN_STEP`), 45-48 (`_text_config`) and 51-84 (the two functions); add `from sous.engine.window import TOKEN_STEP, kv_bytes_per_token, native_max_tokens` to the imports so the rest of the module (`decide_context`, `auto_context_tokens`) keeps working until Task 9 deletes it.

In `src/sous/tune/candidates.py` line 13: `from sous.engine.window import TOKEN_STEP, kv_bytes_per_token, native_max_tokens`.

- [ ] **Step 6: Create `src/sous/tune/payload.py`**

```python
"""What the local model sees during a tune: the eight-tool schema and the
system prompt the worker path used. Kept as fixtures so bench numbers and
suite grades stay comparable across releases — a change here re-baselines
both, and the pin test says so."""

from __future__ import annotations

from pathlib import Path

from sous.protocol import ToolSet


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS: list[dict] = [
```

Paste the eight `_tool(...)` calls from `src/sous/protocol.py` lines 46-107 verbatim (the list body of `WORKER_TOOLS`), close the list, then:

```python
TOOLSET = ToolSet.from_tools(TOOLS)

SYSTEM_TEMPLATE = """..."""  # paste src/sous/worker.py lines 27-41 verbatim


def build_system_prompt(project_root: Path) -> str:
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in project_root.iterdir())[:50]
    return SYSTEM_TEMPLATE.format(root=project_root, listing="\n".join(entries))
```

(`SYSTEM_TEMPLATE` is the exact 15-line triple-quoted string at `worker.py:27-41`, beginning `You are sous, a focused coding subcontractor working alone \`.)

- [ ] **Step 7: Compute the pin and put it in the test**

Run:

```bash
uv run python -c 'import hashlib, json; from sous.tune import payload; print(hashlib.sha256((json.dumps(payload.TOOLS, sort_keys=True) + payload.SYSTEM_TEMPLATE).encode()).hexdigest())'
```

Replace `"PIN"` in `tests/test_tune_payload.py` with the printed hex string.

- [ ] **Step 8: Bench reads the schema from the payload**

In `src/sous/tune/bench.py`: line 15 becomes `from sous.tune.payload import TOOLS`; replace `WORKER_TOOLS` with `TOOLS` at lines 143, 149, 195, 362 and 370. `grep -n WORKER_TOOLS src/sous/tune` must print nothing.

- [ ] **Step 9: Run the tests and the four checks; commit**

Run: `uv run pytest tests/test_engine_window.py tests/test_tune_payload.py tests/test_tune_bench.py tests/test_context.py tests/test_tune_candidates.py -v`
Expected: PASS

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/engine/window.py src/sous/tune/payload.py src/sous/context.py src/sous/tune/candidates.py src/sous/tune/bench.py tests/test_engine_window.py tests/test_tune_payload.py tests/test_context.py
git commit -m "refactor(tune): own the model's schema and prompt; window maths under engine" -m "The worker path is going away and bench and the suite still need what it showed the model. A content pin makes any later change to the payload a deliberate re-baseline." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Scratch tools for the suite

**Files:**
- Create: `src/sous/tune/suite/tools.py`, `tests/test_tune_suite_tools.py`

**Interfaces:**
- Produces: `ScratchTools(root: Path, verify_commands: tuple[str, ...] = (), *, command_timeout: float = 120.0)` with `.root: Path`, `.denied: int`, and methods `read_file(path, offset=0, limit=2000) -> str`, `write_file(path, content) -> str`, `edit_file(path, old, new) -> str`, `list_dir(path=".") -> str`, `glob(pattern) -> str`, `grep(pattern, glob_pattern="**/*") -> str`, `run_command(command, timeout: float | None = None) -> str`; `class ToolError(Exception)`; constants `MAX_TOOL_OUTPUT = 16_000`, `MAX_GREP_HITS = 200`, `MAX_GLOB_HITS = 500`, `ACCEPTED_RUNNERS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_suite_tools.py`:

```python
import sys
from pathlib import Path

import pytest

from sous.tune.suite.tools import ScratchTools, ToolError


def _tools(tmp_path: Path, **kw) -> ScratchTools:
    root = tmp_path / "project"
    root.mkdir()
    return ScratchTools(root, **kw)


def test_paths_resolve_under_the_root_and_escapes_are_refused(tmp_path: Path):
    t = _tools(tmp_path)
    assert t.write_file("pkg/a.py", "x = 1\n") == "wrote 6 chars to pkg/a.py"
    assert t.read_file("pkg/a.py") == "1\tx = 1"
    with pytest.raises(ToolError):
        t.read_file("../outside.txt")
    with pytest.raises(ToolError):
        t.write_file(str(tmp_path / "outside.txt"), "no")
    assert not (tmp_path / "outside.txt").exists()


def test_edit_requires_exactly_one_match(tmp_path: Path):
    t = _tools(tmp_path)
    t.write_file("a.py", "a\nb\na\n")
    assert t.edit_file("a.py", "a", "c") == (
        "edit rejected: 2 matches for old text in a.py (need exactly 1)"
    )
    assert t.edit_file("a.py", "b", "c") == "edited a.py"
    assert t.read_file("a.py") == "1\ta\n2\tc\n3\ta"


def test_list_glob_and_grep_read_the_scratch_project(tmp_path: Path):
    t = _tools(tmp_path)
    t.write_file("pkg/a.py", "def f():\n    return 1\n")
    t.write_file("README.md", "hi\n")
    assert t.list_dir() == "README.md\npkg/"
    assert t.glob("**/*.py") == "pkg/a.py"
    assert t.grep("return", "**/*.py") == "pkg/a.py:2:return 1"


def test_only_verify_commands_and_the_test_runners_run(tmp_path: Path):
    verify = f"{sys.executable} -c 'print(7)'"
    t = _tools(tmp_path, verify_commands=(verify,))
    assert t.run_command(verify) == "exit code 0\n7\n"
    assert t.run_command("rm -rf .") == "command denied (not allowlisted): rm -rf ."
    assert t.denied == 1
    assert t.run_command("pytest --version").startswith("exit code 0")
    assert t.run_command("") == "command rejected: empty"
    assert t.run_command("echo 'unterminated").startswith("command rejected: unparseable")


def test_a_command_past_its_timeout_is_reported_not_raised(tmp_path: Path):
    sleep = f"{sys.executable} -c 'import time; time.sleep(5)'"
    t = _tools(tmp_path, verify_commands=(sleep,))
    assert t.run_command(sleep, timeout=0.2) == f"command timed out after 0.2s: {sleep}"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_tune_suite_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sous.tune.suite.tools'`

- [ ] **Step 3: Create `src/sous/tune/suite/tools.py`**

```python
"""The suite loop's tools over a scratch copy of a synthetic project.

Not a boundary: the project is a throwaway copy under a temp dir, the only
commands accepted are the task's own verify commands and the suite's test
runners, and nothing here audits, scrubs or kills process groups. Output
shapes match what the worker's tools printed, so the model reads the same
results the suite was graded against."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

MAX_TOOL_OUTPUT = 16_000
MAX_GREP_HITS = 200
MAX_GLOB_HITS = 500
# What a task may run besides its own verify commands: the runners the suite's
# tasks are written for, matched as argv prefixes like the old allowlist was.
ACCEPTED_RUNNERS = (
    "python -m unittest",
    "python3 -m unittest",
    "python -m pytest",
    "python3 -m pytest",
    "pytest",
)


class ToolError(Exception):
    """A call the scratch project refuses; the message is what the model reads."""


def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + "\n[truncated]"
    return text


class ScratchTools:
    def __init__(
        self,
        root: Path,
        verify_commands: tuple[str, ...] = (),
        *,
        command_timeout: float = 120.0,
    ):
        self.root = root.resolve()
        self.denied = 0
        self._accepted = [shlex.split(c) for c in (*verify_commands, *ACCEPTED_RUNNERS)]
        self._command_timeout = command_timeout

    def _confined(self, path: str) -> Path:
        raw = Path(path)
        candidate = (raw if raw.is_absolute() else self.root / raw).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise ToolError(f"path escapes the project: {path}")
        return candidate

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        p = self._confined(path)
        # Streamed, not read_text().splitlines(): reading five lines of a
        # large file must not materialize the whole file first.
        numbered: list[str] = []
        joined_len = -1  # each entry costs len(entry) + 1 joining "\n"
        with p.open(errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if lineno <= offset:
                    continue
                if lineno > offset + limit:
                    break
                entry = f"{lineno}\t" + line.removesuffix("\n")
                numbered.append(entry)
                joined_len += len(entry) + 1
                if joined_len > MAX_TOOL_OUTPUT:
                    break
        return _truncate("\n".join(numbered) or "(empty file)")

    def write_file(self, path: str, content: str) -> str:
        p = self._confined(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._confined(path)
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"edit rejected: {n} matches for old text in {path} (need exactly 1)"
        p.write_text(text.replace(old, new, 1))
        return f"edited {path}"

    def list_dir(self, path: str = ".") -> str:
        p = self._confined(path)
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return _truncate("\n".join(entries) or "(empty dir)")

    def glob(self, pattern: str) -> str:
        hits = sorted(
            str(p.relative_to(self.root))
            for p in self.root.glob(pattern)
            if ".git" not in p.parts and p.resolve().is_relative_to(self.root)
        )
        return _truncate("\n".join(hits[:MAX_GLOB_HITS]) or "(no matches)")

    def grep(self, pattern: str, glob_pattern: str = "**/*") -> str:
        rx = re.compile(pattern)
        out: list[str] = []
        for p in sorted(self.root.glob(glob_pattern)):
            if not p.is_file() or ".git" in p.parts or not p.resolve().is_relative_to(self.root):
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{p.relative_to(self.root)}:{i}:{line.strip()}")
                        if len(out) >= MAX_GREP_HITS:
                            return _truncate("\n".join(out) + "\n[max hits reached]")
            except OSError:
                continue
        return _truncate("\n".join(out) or "(no matches)")

    def run_command(self, command: str, timeout: float | None = None) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"command rejected: unparseable ({e})"
        if not argv:
            return "command rejected: empty"
        if not any(argv[: len(prefix)] == prefix for prefix in self._accepted):
            self.denied += 1
            return f"command denied (not allowlisted): {command}"
        budget = self._command_timeout if timeout is None else timeout
        try:
            proc = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True, timeout=budget, check=False
            )
        except FileNotFoundError:
            return f"command not found: {argv[0]}"
        except subprocess.TimeoutExpired:
            return f"command timed out after {budget}s: {command}"
        body = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        return _truncate(f"exit code {proc.returncode}\n{body}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_suite_tools.py -v`
Expected: PASS

- [ ] **Step 5: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/tune/suite/tools.py tests/test_tune_suite_tools.py
git commit -m "feat(tune): scratch-project tools for the suite loop" -m "The suite grades a candidate on a throwaway copy of a synthetic project. It needs the worker's tool behaviours, not its sandbox: paths under the copy, the task's own verify commands, and the same output shapes the grades were taken against." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: The suite loop

**Files:**
- Create: `src/sous/tune/suite/loop.py`, `tests/test_tune_suite_loop.py`

**Interfaces:**
- Consumes: `sous.tune.payload.TOOLS`, `TOOLSET`, `build_system_prompt`; `sous.tune.suite.tools.ScratchTools`, `ToolError`; `sous.protocol.parse_tool_calls(text, toolset) -> list[ToolCall]`, `ParseError`, `ToolCall(name, arguments)`; `sous.engine.base.ManagedEngine.session() -> GenerationSession` (`.generate(messages, tools, max_tokens, timeout=...) -> str`, `.thread`, `.close()`), `ManagedEngine.count_tokens(messages, tools) -> int`, `ManagedEngine.reset_prompt_cache(owner=...)`, `GenerationStalled`.
- Produces: `Budget(turns: int, minutes: int, tokens_per_generation: int = 4096)`; `LoopResult(outcome: str, turns: int, seconds: float, error: str | None, summary: str, concerns: str)`; `Transcript(path: Path)` with `.path` and `.log(**event)`; `run_loop(*, instructions: str, context_files: Sequence[str], root: Path, engine: ManagedEngine, window: int, budget: Budget, tools: ScratchTools, transcript: Transcript, command_timeout: float = 120.0) -> LoopResult`. Outcomes: `"completed"`, `"failed"` (with `error` set), `"budget-exhausted"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_suite_loop.py`:

```python
import json
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.tune.suite.loop import Budget, Transcript, run_loop
from sous.tune.suite.tools import ScratchTools
from tests.fake_engine import FakeEngine

CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "done", "concerns": ""}')
WRITE = CALL.format(name="write_file", args='{"path": "hello.txt", "content": "hi"}')


def _run(tmp_path: Path, script: list[str], budget: Budget = Budget(turns=8, minutes=5)):
    root = tmp_path / "project"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    mgr = EngineManager(cfg, engine_factory=lambda _mid: FakeEngine(script))
    transcript = Transcript(tmp_path / "transcript.jsonl")
    try:
        result = run_loop(
            instructions="write hello.txt",
            context_files=(),
            root=root,
            engine=mgr.get(),
            window=8192,
            budget=budget,
            tools=ScratchTools(root),
            transcript=transcript,
        )
    finally:
        mgr.unload_now()
    events = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    return result, events, root


def test_a_finished_task_is_completed_with_its_summary(tmp_path: Path):
    result, events, root = _run(tmp_path, [WRITE, FINISH])
    assert result.outcome == "completed"
    assert result.error is None
    assert result.turns == 2
    assert result.summary == "done"
    assert (root / "hello.txt").read_text() == "hi"
    assert [e["event"] for e in events] == ["generation", "tool", "generation", "finished"]
    assert events[1]["name"] == "write_file"
    assert events[-1]["outcome"] == "completed"


def test_running_out_of_turns_is_budget_exhausted_not_failed(tmp_path: Path):
    result, events, _ = _run(tmp_path, [WRITE, WRITE], Budget(turns=1, minutes=5))
    assert result.outcome == "budget-exhausted"
    assert result.error is None
    assert result.turns == 1
    assert events[-1] == {"event": "finished", "outcome": "budget-exhausted"}


def test_three_malformed_turns_fail_the_task_and_are_counted(tmp_path: Path):
    bad = "<tool_call>{not json}</tool_call>"
    result, events, _ = _run(tmp_path, [bad, bad, bad])
    assert result.outcome == "failed"
    assert result.error == "model-confused: 3 consecutive malformed tool calls"
    assert sum(e["event"] == "malformed" for e in events) == 3


def test_three_turns_without_a_tool_call_fail_the_task(tmp_path: Path):
    result, _, _ = _run(tmp_path, ["thinking...", "still thinking", "hmm"])
    assert result.outcome == "failed"
    assert result.error == "model-confused: 3 consecutive turns without a tool call"


def test_an_engine_error_is_a_failed_run(tmp_path: Path):
    result, events, _ = _run(tmp_path, [])  # FakeEngine raises once its script is exhausted
    assert result.outcome == "failed"
    assert result.error is not None and result.error.startswith("engine error: ")
    assert events[0]["event"] == "engine_error"


def test_finish_without_a_summary_is_a_tool_error_and_the_loop_goes_on(tmp_path: Path):
    empty = CALL.format(name="finish", args='{"summary": ""}')
    result, events, _ = _run(tmp_path, [empty, FINISH])
    assert result.outcome == "completed"
    assert events[1] == {
        "event": "tool",
        "name": "finish",
        "arguments": {"summary": ""},
        "result": "error: finish requires a non-empty summary",
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_tune_suite_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sous.tune.suite.loop'`

- [ ] **Step 3: Create `src/sous/tune/suite/loop.py`**

```python
"""One suite task through the local model: the shape of the loop the worker
ran, minus the queue, the approvals and the audit a throwaway scratch project
does not need. Writes the transcript events the runner's metrics read."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sous.engine.base import GenerationStalled, ManagedEngine
from sous.protocol import ParseError, ToolCall, parse_tool_calls
from sous.tune.payload import TOOLS, TOOLSET, build_system_prompt
from sous.tune.suite.tools import ScratchTools, ToolError

MAX_CONSECUTIVE_MALFORMED = 3
FORMAT_REMINDER = (
    "Your tool call could not be parsed ({error}). Re-emit it using exactly "
    "the tool-call format specified in your instructions."
)
NUDGE = "You must call a tool to make progress. Call finish when done."


@dataclass(frozen=True)
class Budget:
    turns: int
    minutes: int
    tokens_per_generation: int = 4096


@dataclass(frozen=True)
class LoopResult:
    outcome: str
    turns: int
    seconds: float
    error: str | None
    summary: str
    concerns: str


class Transcript:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, **event) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")


def _tool_result_message(name: str, result: str) -> dict:
    return {"role": "user", "content": f'<tool_result name="{name}">\n{result}\n</tool_result>'}


def _execute(call: ToolCall, tools: ScratchTools, deadline: float, command_timeout: float) -> str:
    try:
        a = call.arguments
        match call.name:
            case "read_file":
                return tools.read_file(a["path"], a.get("offset", 0), a.get("limit", 2000))
            case "write_file":
                return tools.write_file(a["path"], a["content"])
            case "edit_file":
                return tools.edit_file(a["path"], a["old"], a["new"])
            case "list_dir":
                return tools.list_dir(a.get("path", "."))
            case "glob":
                return tools.glob(a["pattern"])
            case "grep":
                return tools.grep(a["pattern"], a.get("glob_pattern", "**/*"))
            case "run_command":
                # One command may not run past the task's own deadline.
                remaining = deadline - time.monotonic()
                return tools.run_command(a["command"], timeout=max(1, min(command_timeout, remaining)))
            case _:
                return f"error: unhandled tool {call.name}"
    except ToolError as e:
        return f"error: {e}"
    except KeyError as e:
        return f"error: missing required argument {e}"
    except OSError as e:
        return f"error: {e}"
    except Exception as e:  # noqa: BLE001 — a tool's failure is the model's to read, never the loop's to raise
        return f"error: {e}"


def run_loop(
    *,
    instructions: str,
    context_files: Sequence[str],
    root: Path,
    engine: ManagedEngine,
    window: int,
    budget: Budget,
    tools: ScratchTools,
    transcript: Transcript,
    command_timeout: float = 120.0,
) -> LoopResult:
    session = engine.session()
    try:
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(root)},
            {"role": "user", "content": instructions},
        ]
        for cf in context_files:
            try:
                content = tools.read_file(cf)
            except (ToolError, OSError) as e:
                content = f"error: {e}"
            messages.append({"role": "user", "content": f"Contents of {cf}:\n{content}"})

        started = time.monotonic()
        deadline = started + budget.minutes * 60
        turns = malformed = 0
        summary = concerns = ""
        outcome, error = "budget-exhausted", None

        def fail(reason: str) -> None:
            nonlocal outcome, error
            outcome, error = "failed", reason

        while turns < budget.turns and time.monotonic() < deadline:
            token_count = engine.count_tokens(messages, TOOLS)
            output_room = window - token_count
            if output_room <= 0:
                # No elision here: a suite task that fills its window is a
                # failed run, which is what the worker reported once it had
                # nothing left to elide.
                reason = f"context overflow: {token_count} tokens fill the {window}-token window"
                transcript.log(event="context_overflow", error=reason)
                fail(reason)
                break
            remaining = max(0.1, deadline - time.monotonic())
            try:
                text = session.generate(
                    messages,
                    TOOLS,
                    min(budget.tokens_per_generation, output_room),
                    timeout=remaining,
                )
            except GenerationStalled as e:
                if time.monotonic() >= deadline:
                    transcript.log(event="budget-exhausted", error=str(e))
                    break
                transcript.log(event="stalled", error=str(e))
                fail(str(e))
                break
            except Exception as e:  # noqa: BLE001 — the engine's failure is the run's outcome, not a crash of the suite
                transcript.log(event="engine_error", error=str(e))
                fail(f"engine error: {e}")
                break
            turns += 1
            transcript.log(event="generation", turn=turns, text=text)
            messages.append({"role": "assistant", "content": text})

            try:
                calls = parse_tool_calls(text, TOOLSET)
            except ParseError as e:
                malformed += 1
                transcript.log(event="malformed", error=str(e))
                if malformed >= MAX_CONSECUTIVE_MALFORMED:
                    fail("model-confused: 3 consecutive malformed tool calls")
                    break
                messages.append({"role": "user", "content": FORMAT_REMINDER.format(error=e)})
                continue
            if not calls:
                malformed += 1
                if malformed >= MAX_CONSECUTIVE_MALFORMED:
                    fail("model-confused: 3 consecutive turns without a tool call")
                    break
                messages.append({"role": "user", "content": NUDGE})
                continue
            malformed = 0

            finished = False
            for call in calls:
                if call.name == "finish":
                    raw_summary = call.arguments.get("summary")
                    if raw_summary is None or not str(raw_summary).strip():
                        result = "error: finish requires a non-empty summary"
                        transcript.log(
                            event="tool", name=call.name, arguments=call.arguments, result=result
                        )
                        messages.append(_tool_result_message(call.name, result))
                        continue
                    summary = str(raw_summary)
                    concerns = str(call.arguments.get("concerns", ""))
                    outcome, finished = "completed", True
                    break
                result = _execute(call, tools, deadline, command_timeout)
                transcript.log(
                    event="tool", name=call.name, arguments=call.arguments, result=result[:2000]
                )
                messages.append(_tool_result_message(call.name, result))
            if finished:
                break

        transcript.log(event="finished", outcome=outcome)
        return LoopResult(outcome, turns, time.monotonic() - started, error, summary, concerns)
    finally:
        session.close()
        engine.reset_prompt_cache(owner=session.thread)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_suite_loop.py -v`
Expected: PASS. If the malformed test fails because `parse_tool_calls` returns no calls for `{not json}` instead of raising, use `'<tool_call>{"name": "no_such_tool", "arguments": {}}</tool_call>'` as `bad`: an unknown tool name is a `ParseError` under a strict toolset. If `FakeEngine.count_tokens` is not defined, check `tests/fake_engine.py`: the `Engine` protocol requires it, and every existing fake implements it; do not add one to the loop.

- [ ] **Step 5: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/tune/suite/loop.py tests/test_tune_suite_loop.py
git commit -m "feat(tune): a suite loop of its own over the scratch tools" -m "The runner drove the worker loop through a scratch task store. Its shape is what the grades were taken against, so the loop keeps the prompt, the tool schema, the budget rules and the transcript events, and drops the queue, approvals and audit that only mattered on a real repository." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: The runner grades through the loop

**Files:**
- Modify: `src/sous/tune/suite/runner.py` (imports 22-48; `SUITE_ALLOWLIST` 49; `SuiteRun.completed` 86-88; `_Denier` 210-249; `_scratch_config` 252-267; `run_one` 296-373)
- Modify: `tests/test_tune_runner.py`

**Interfaces:**
- Consumes: Task 3's `ScratchTools`, Task 4's `Budget`, `Transcript`, `run_loop`.
- Produces: `run_one(task, index, arm, engine, counting, scratch, *, python, grade_timeout=120.0) -> SuiteRun` (the `poll` parameter is gone). `SuiteRun` fields unchanged; `state` is `"done"` or `"failed"`; `outcome` is `None` on a failed run.

- [ ] **Step 1: Adapt the tests first**

In `tests/test_tune_runner.py`:
- Delete `from sous.tasks import TaskStore` (line 15) and `SUITE_ALLOWLIST` / `DEFAULT_ALLOWLIST` from the imports (line 20 and the `sous.config` import).
- Delete `test_the_suite_allowlist_is_the_shipped_one_plus_unittest` (98), `test_the_denier_survives_a_store_error_and_records_it` (166) and `test_a_denier_failure_marks_the_run_as_an_error` (180), and the `BustedStore` / `FailingDenier` helpers they use.
- In `test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run` (119): the transcript is now `scratch / "transcript.jsonl"`; assert `run.state == "done"`, `run.outcome == "completed"`, `run.approvals_denied == 0`.
- In `test_a_command_outside_the_allowlist_is_denied_and_counted` (137): keep the scripted `run_command` call for a command the task does not accept (e.g. `rm -rf /`); assert `run.approvals_denied == 1` and that the transcript's `tool` event for it has `result == "command denied (not allowlisted): rm -rf /"`.
- Wherever a test passes `poll=` to `run_one`/`run_suite`, remove it. Comments naming `run_task` (232, 245) now name `run_loop`.

Run: `uv run pytest tests/test_tune_runner.py -v`
Expected: FAIL — `run_one` still takes `poll`, the transcript path differs, denials come from the denier.

- [ ] **Step 2: Rewrite the runner's task execution**

In `src/sous/tune/suite/runner.py`:
- Imports: drop `from sous.config import DEFAULT_ALLOWLIST, SousConfig` (keep `SousConfig` only if something else in the file still uses it — grep), `from sous.context import ContextDecision`, `from sous.tasks import TaskState, TaskStore`, `from sous.worker import run_task`; add `from sous.tune.suite.loop import Budget, Transcript, run_loop` and `from sous.tune.suite.tools import ScratchTools`.
- Delete `SUITE_ALLOWLIST` (49), `_POLL_SECONDS`, the `_Denier` class (210-249) and `_scratch_config` (252-267).
- `SuiteRun.completed` returns `self.state == "done"`.
- Replace `run_one` (296-373) with:

```python
def run_one(
    task: SuiteTask,
    index: int,
    arm: Arm,
    engine: ManagedEngine,
    counting: CountingEngine,
    scratch: Path,
    *,
    python: Path,
    grade_timeout: float = 120.0,
) -> SuiteRun:
    """One task, once, through the suite loop, then the grade over what it
    left in the scratch copy of the project."""
    project = scratch / "project"
    shutil.copytree(task.project, project)
    transcript = Transcript(scratch / "transcript.jsonl")
    tools = ScratchTools(project, task.verify_commands)
    before = counting.output_tokens
    with interpreter_first(python):
        result = run_loop(
            instructions=task.instructions,
            context_files=task.context_files,
            root=project,
            engine=engine,
            window=arm.window,
            budget=Budget(turns=task.max_turns, minutes=task.max_minutes),
            tools=tools,
            transcript=transcript,
        )
    malformed, repetitions = metrics_from_transcript(transcript.path)
    grade = grade_task(task, project, python=python, timeout=grade_timeout)
    failed = result.error is not None
    return SuiteRun(
        task=task.name,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        window=arm.window,
        state="failed" if failed else "done",
        outcome=None if failed else result.outcome,
        turns=result.turns,
        seconds=result.seconds,
        output_tokens=counting.output_tokens - before,
        malformed=malformed,
        repetitions=repetitions,
        approvals_denied=tools.denied,
        grade=grade.score,
        grade_detail=grade.detail,
        error=result.error,
        transcript_path=str(transcript.path),
    )
```

- `grep -n "TaskState\|TaskStore\|run_task\|poll" src/sous/tune/suite/runner.py` must print nothing; fix any `run_suite` call site that passed `poll`.

- [ ] **Step 3: Run the tune tests**

Run: `uv run pytest tests/test_tune_runner.py tests/test_tune_suite.py tests/test_tune_main.py -v`
Expected: PASS

- [ ] **Step 4: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/tune/suite/runner.py tests/test_tune_runner.py
git commit -m "refactor(tune): grade through the suite loop, not the worker" -m "Nothing of the queue, the approval thread or the scratch config survives in the runner; a denied command is a counted tool error, as the denier made it." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: The daemon without the queue

Stop the daemon from reading, running or reporting tasks. The worker module still exists after this task (Task 9 deletes it); nothing starts it any more.

**Files:**
- Modify: `src/sous/server.py` (`SousService` 77-369, `create_server` 416-536, `main` 679-737, imports 24-40), `src/sous/monitor.py` (`_busy` 59-69), `src/sous/tune/daemon.py` (`ready_for_tune` 24-101)
- Test: `tests/test_server.py`, `tests/test_monitor.py`, `tests/test_gateway_http.py:43-64`, `tests/test_cli.py:1401-1441`, the tune readiness tests (`grep -ln '"queue"' tests/test_tune_*.py`)

**Interfaces:**
- Produces: `SousService(engines: EngineManager, config: SousConfig, inflight: Inflight | None = None)`; `SousService.status_version() -> tuple[int, int, tuple[int, int]]`; `SousService.status_document(*, recent: bool) -> dict` with keys `engine`, `inflight`, `config` (`model_id`, `idle_unload_minutes`, `port`, `gateway{enabled, local_models, max_context_tokens, upstream_url}`) and, with `recent=True`, `recent_turns`; `create_server(engines: EngineManager, config: SousConfig, *, upstream: Upstream | None = None, inflight: Inflight | None = None) -> MCPServer` whose lifespan starts and stops the idle sweep. `server_status`, `delegate_task`, `task_status`, `task_result`, `cancel_task`, `respond_to_command_request` no longer exist.

- [ ] **Step 1: Write the failing test for the new document shape**

Append to `tests/test_server.py` (after replacing the `svc` fixture as in Step 3):

```python
def test_the_status_document_carries_no_task_fields(svc):
    service, _root = svc
    doc = service.status_document(recent=True)
    assert set(doc) == {"engine", "inflight", "config", "recent_turns"}
    assert set(doc["config"]) == {"model_id", "idle_unload_minutes", "port", "gateway"}
    assert set(service.status_document(recent=False)) == {"engine", "inflight", "config"}
    assert len(service.status_version()) == 3
```

Run: `uv run pytest tests/test_server.py -k no_task_fields -v`
Expected: FAIL (the fixture still returns three values and the document still has `queue`)

- [ ] **Step 2: Reshape `SousService`**

In `src/sous/server.py`:
- `__init__(self, engines, config, inflight=None)`: keep `self.engines`, `self.config`, `self.inflight`; delete `self.store` and the three `_Memo` attributes (and the `_Memo` class if nothing else uses it).
- Delete `delegate_task`, `_status_entry`, `task_status`, `task_result`, `_diff`, `_is_git_repo`, `cancel_task`, `respond_to_command_request`, `_task_summary`, `_task_reads`, `_allowlist_now`, `server_status`.
- `status_version` returns `(self.inflight.version, self.engines.version, self._config_stamp())`; update its docstring to three reads.
- `status_document`: remove the `counts, tasks = ...` line, the `"queue"` entry, `config.max_turns`, `config.max_minutes`, `config.allowlist`, `config.context`, and the `recent_tasks` line; the docstring no longer mentions tasks or SQLite.
- Delete `_INSTRUCTIONS` (379-413) and the six `@mcp.tool()` functions in `create_server`; construct `MCPServer("sous", lifespan=_lifespan)`.
- `create_server(engines, config, *, upstream=None, inflight=None)`: `svc = SousService(engines, config, inflight)`; the lifespan becomes

```python
    @contextlib.asynccontextmanager
    async def _lifespan(_: MCPServer) -> AsyncIterator[None]:
        engines.start_idle_sweep()
        try:
            yield None
        finally:
            engines.stop_idle_sweep()
            for gateway in mounted_gateway:
                await gateway.aclose()
```

- `main()`: delete the `store = TaskStore(...)` / `recover_interrupted` block and the worker `threading.Thread`; call `create_server(engines, config)`. Keep `stop`, `_install_shutdown_handler(stop)` and `_login_shell_path` for now (Task 9 removes both worker leftovers).
- Imports: drop `TaskStore`, `FINISHED_STATES`, `Task`, `TaskState`, `run_worker_loop`, `current_allowlist`, `persist_allowlist_entry`, `_is_within`, `canonical_command_for_allowlist`, `command_allowed`, `format_line` and any stdlib import ruff now reports unused (`re`, `shlex`, `subprocess`, `time`).

- [ ] **Step 3: Update the daemon's other readers and the tests that build it**

- `src/sous/monitor.py` `_busy`: delete the `queue` lookup and `or queue.get("running")`; the comment about a delegated task goes.
- `src/sous/tune/daemon.py` `ready_for_tune`: replace the queue block with

```python
    inflight = len(doc.get("inflight") or [])
    if inflight:
        return Readiness(False, f"the daemon is busy: {inflight} turn(s) in flight")
```

- `tests/test_server.py`: the `svc` fixture becomes `SousConfig(...)` + `EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))` + `return SousService(engines, cfg), root` (no store, no allowlist file); delete the tests at lines 39, 46, 51, 56, 64, 71, 84, 105, 111, 117, 125, 135, 140, 150, 168, 206, 223, 240, 250, 257, 263, 276, 305, 322, 337, 353, 396, 410, 428, 445, 461, 478, 765, 875, 898, 914, 943, 964; every remaining `create_server(store, ...)` call drops its first argument; `test_server_status_reports_gateway_config` (813) reads `service.status_document(recent=False)["config"]["gateway"]`.
- `tests/test_monitor.py`: `_app` helper (29-56) loses `store`; the bare call at 271-273 becomes `create_server(engines, cfg, upstream=FakeUpstream().upstream())`; delete `test_events_beat_while_a_delegated_task_is_running` (466); in `test_an_idle_stream_polls_on_the_idle_tick_and_a_busy_one_on_the_fast_tick` (418) the idle-safe change at 437-449 becomes `engines.touch()` (it bumps the engine version and keeps the stream idle); in the test at 563 drop the task-change segment (570, 597-600) and rename it `test_a_load_a_hold_a_release_and_an_unload_each_reach_the_client`; `test_status_is_the_full_status_document` (92) and `test_events_start_with_the_full_document` (372) expect keys `{"engine", "inflight", "config", "recent_turns"}`.
- `tests/test_gateway_http.py` `_app` (43-64): `create_server(engines, cfg, upstream=upstream)`.
- `tests/test_cli.py:1436`: `create_server(engines, cfg)`.
- Tune readiness tests: any document fixture carrying `"queue"` loses it; a test that expected the "task(s) running" message now expects `"the daemon is busy: 1 turn(s) in flight"` for a document with one `inflight` entry. Add one if none exists:

```python
def test_a_turn_in_flight_makes_the_daemon_busy():
    doc = {"engine": {"loaded": True, "holders": 0}, "inflight": [{"id": "msg_1"}]}
    request = lambda port, method, path: (200, json.dumps(doc).encode())  # noqa: E731
    r = ready_for_tune(8383, request=request, port_open=lambda port: True)
    assert r == Readiness(False, "the daemon is busy: 1 turn(s) in flight")
```

- [ ] **Step 4: Run the affected tests**

Run: `uv run pytest tests/test_server.py tests/test_monitor.py tests/test_gateway_http.py tests/test_cli.py tests/test_tune_main.py -v`
Expected: PASS

- [ ] **Step 5: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add -A src/sous/server.py src/sous/monitor.py src/sous/tune/daemon.py tests
git commit -m "refactor(server): the daemon stops running and reporting tasks" -m "The six MCP tools, the task store and the worker thread leave the daemon; the status document keeps engine, inflight, config and recent turns, and the idle sweep now starts from the lifespan since the worker loop no longer runs it." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: The terminal without task rows

**Files:**
- Modify: `src/sous/tui.py` (`TASK_WORDS` 83-90; the `#tasks` widget in `compose`; `_show_tasks` 1754-1763 and its call at 1645; `queue` reads at 1621 and 1875; the `TASKS` line at 1116-1123; the compact tail at 1126-1127)
- Test: `tests/test_tui.py` (`_doc` 27-66; tests at 423 and 836; the snapshot test 1139-1259), `tests/snapshots/sous-top-100x30.svg`

- [ ] **Step 1: Adapt the tests**

In `tests/test_tui.py`: remove the `queue` (58) and `recent_tasks` (65) entries and the `tasks` parameter from `_doc`; delete the `TASKS` constant; in `test_the_slip_shows_the_turn_from_the_document` (423) drop `tasks=TASKS` and the assertions at 454 (`" TASKS  ON RAIL 2 · COOKING 1"`), 460 and 462-469 (`#tasks`); in `test_resize_reflows_to_every_breakpoint` (836) drop `tasks=TASKS` (839) and the `#tasks` assertion (850); in the snapshot test drop `tasks=TASKS` (1238).

Run: `uv run pytest tests/test_tui.py -x -q`
Expected: FAIL — the app still queries `#tasks` and reads `document["queue"]`

- [ ] **Step 2: Remove the task surface from the TUI**

In `src/sous/tui.py`: delete `TASK_WORDS`; delete the `Line` widget with `id="tasks"` from `compose` (grep `"tasks"`); delete `_show_tasks` and its call; delete the two `queue = document.get("queue") or {}` lines and the `f" {'TASKS':<7}ON RAIL ..."` entry in the `lines += [...]` block (keep the `ORDERS` and the blank-prefixed `ABANDONED` lines); the compact `tail` no longer carries the `ON RAIL … · COOKING …` prefix. Any CSS rule for `#tasks` goes with the widget.

- [ ] **Step 3: Regenerate the snapshot and run the TUI tests**

```bash
SOUS_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_tui.py -k renders_as_committed -q
uv run pytest tests/test_tui.py -q
```
Expected: the second run PASSES with the new `tests/snapshots/sous-top-100x30.svg`. Open the SVG and confirm no `TASK` row and no `ON RAIL` text remains.

- [ ] **Step 4: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/tui.py tests/test_tui.py tests/snapshots/sous-top-100x30.svg
git commit -m "refactor(tui): the pass shows turns only" -m "The status document no longer carries tasks or a queue; the rail, the slip, the ticket and the rates already render turns." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: The CLI surface

**Files:**
- Modify: `src/sous/cli.py` (`_cmd_status` 375-388; `_cmd_wait` 503-539 and `_arg_timeout`/`_arg_interval` if only `wait` uses them; `_cmd_stop` 668-717; `main` registration 916-963 and dispatch 964-992)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `status_lines(document: dict, now: float) -> list[str]` (public, tested); `sous status` prints those lines or the not-running hint; `sous wait` and `sous mcp` are gone.
- Consumes: `_sous_request(port, method, path) -> tuple[int, bytes]`, `_json_object(raw) -> dict | None`, `_port_open(port) -> bool`, `restart_hint(*, managed)`, `_clock(seconds) -> str` — all already in `cli.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
from sous.cli import status_lines


def test_status_lines_show_the_engine_the_turn_and_the_recent_ring():
    doc = {
        "engine": {
            "loaded": True,
            "loading": False,
            "unloading": False,
            "model_id": "org/m",
            "idle_seconds": 12.0,
            "holders": 1,
        },
        "inflight": [{"id": "msg_1", "model": "sous-local", "phase": "decode", "started_at": 100.0}],
        "config": {"port": 8383},
        "recent_turns": [
            {
                "id": "msg_0",
                "model": "sous-local",
                "status": 200,
                "stop_reason": "tool_use",
                "cache": "hit",
                "input_tokens": 66262,
                "output_tokens": 50,
                "seconds": 4.9,
            }
        ],
    }
    lines = status_lines(doc, now=130.0)
    assert lines[0] == "sous daemon: listening on 127.0.0.1:8383"
    assert lines[1].startswith("  engine: org/m loaded · holders 1 · idle ")
    assert lines[2].startswith("  turn msg_1 sous-local decode ")
    assert lines[3] == "  recent turns:"
    assert lines[4] == "    msg_0 sous-local status=200 stop=tool_use cache=hit in=66262 out=50 4.9s"


def test_status_lines_for_an_idle_unloaded_daemon():
    doc = {
        "engine": {"loaded": False, "loading": False, "unloading": False, "model_id": "org/m",
                   "idle_seconds": None, "holders": 0},
        "inflight": [],
        "config": {"port": 8383},
        "recent_turns": [],
    }
    assert status_lines(doc, now=0.0) == [
        "sous daemon: listening on 127.0.0.1:8383",
        "  engine: org/m unloaded · holders 0",
        "  turns in flight: none",
    ]


def test_status_prints_the_document_when_the_daemon_answers(monkeypatch, capsys, tmp_path):
    import json

    from sous import cli

    doc = {"engine": {"loaded": False, "model_id": "org/m", "holders": 0}, "inflight": [],
           "config": {"port": 8383}, "recent_turns": []}
    monkeypatch.setattr(cli, "load_config", lambda: SousConfig(server_port=8383, data_dir=tmp_path, config_path=tmp_path / "config.toml"))
    monkeypatch.setattr(cli, "_port_open", lambda port: True)
    monkeypatch.setattr(cli, "_sous_request", lambda port, method, path, json=None: (200, json_dumps(doc)))
    cli.main(["status"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "sous daemon: listening on 127.0.0.1:8383"
    assert out[1] == "  engine: org/m unloaded · holders 0"


def json_dumps(doc: dict) -> bytes:
    import json

    return json.dumps(doc).encode()


def test_wait_and_mcp_are_no_longer_commands(capsys):
    from sous import cli

    for verb in ("wait", "mcp"):
        with pytest.raises(SystemExit):
            cli.main([verb])
        assert "invalid choice" in capsys.readouterr().err
```

(`SousConfig` and `pytest` are already imported at the top of `tests/test_cli.py`; add them if not.)

Run: `uv run pytest tests/test_cli.py -k "status_lines or status_prints or no_longer_commands" -v`
Expected: FAIL (`status_lines` does not exist; `wait`/`mcp` still parse)

- [ ] **Step 2: Delete `wait` and `mcp`, trim `stop`, rewrite `status`**

In `src/sous/cli.py`:
- Delete `_cmd_wait` (503-539) and, if `grep -n "_arg_timeout\|_arg_interval" src/sous/cli.py` shows no other use, those two helpers; delete the `wait` registration (926-934) and dispatch (981-982); delete the `mcp` registration (936) and dispatch (969-974).
- In `_cmd_stop`: delete the `TaskStore` block (690-701) and the final `print("any running `sous mcp` bridges will exit on their own")`.
- Help texts: `serve` → `"run the daemon (the endpoint on 127.0.0.1)"`; `status` → `"the daemon, its engine and the recent turns"`.
- Replace `_cmd_status` with:

```python
def status_lines(document: dict, now: float) -> list[str]:
    """`sous status` from the status document: the engine, the turns in
    flight and the last five served, newest first. `now` is only for a
    turn's elapsed time, like statusline_text."""
    engine = document.get("engine") or {}
    config = document.get("config") or {}
    if engine.get("loading"):
        state = "loading"
    elif engine.get("unloading"):
        state = "unloading"
    elif engine.get("loaded"):
        state = "loaded"
    else:
        state = "unloaded"
    engine_parts = [
        f"engine: {engine.get('model_id', '?')} {state}",
        f"holders {engine.get('holders') or 0}",
    ]
    idle = engine.get("idle_seconds")
    if state == "loaded" and idle is not None:
        engine_parts.append(f"idle {_clock(idle)}")
    lines = [
        f"sous daemon: listening on 127.0.0.1:{config.get('port', '?')}",
        "  " + " · ".join(engine_parts),
    ]
    inflight = document.get("inflight") or []
    if inflight:
        for turn in inflight:
            since = turn.get("started_at")
            age = f" {_clock(now - since)}" if since is not None else ""
            lines.append(
                f"  turn {turn.get('id', '?')} {turn.get('model', '?')} "
                f"{turn.get('phase', 'queued')}{age}"
            )
    else:
        lines.append("  turns in flight: none")
    recent = list(document.get("recent_turns") or [])[-5:]
    if recent:
        lines.append("  recent turns:")
        for s in reversed(recent):
            lines.append(
                f"    {s.get('id', '?')} {s.get('model', '?')} status={s.get('status', '?')} "
                f"stop={s.get('stop_reason') or '-'} cache={s.get('cache') or '-'} "
                f"in={s.get('input_tokens', '-')} out={s.get('output_tokens', '-')} "
                f"{s.get('seconds', '-')}s"
            )
    return lines


def _cmd_status() -> None:
    config = load_config()
    if not _port_open(config.server_port):
        print(f"sous daemon: not running (port {config.server_port})")
        print("start it with: sous serve   (or: sous install-launchd)")
        return
    import httpx

    try:
        status, raw = _sous_request(config.server_port, "GET", "/sous/status")
    except httpx.HTTPError as e:
        print(
            f"sous daemon: port {config.server_port} did not answer /sous/status "
            f"({type(e).__name__}); is it sous?"
        )
        raise SystemExit(1)
    document = _json_object(raw) if status == 200 else None
    if document is None:
        print(
            f"sous daemon: port {config.server_port} answered {status} to /sous/status; "
            f"restart it ({restart_hint(managed=False)})"
        )
        raise SystemExit(1)
    for line in status_lines(document, now=time.time()):
        print(line)
```

Use for `now` the same clock `_statusline_fetch` passes to `statusline_text` (check that call: if it passes `time.monotonic()`, pass `time.monotonic()` here too — `started_at` must be on the same clock).

- [ ] **Step 3: Prune the old CLI tests**

Delete `test_mcp_subcommand_dispatches_to_the_proxy` (69), `test_stop_warns_about_all_active_tasks` (249), and the whole `wait` block 686-823 (`_wait_store`, `_enqueue` and the seven `test_wait_*` functions).

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 4: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add src/sous/cli.py tests/test_cli.py
git commit -m "feat(cli): sous status reports the engine and the turns; wait and mcp go" -m "wait and mcp only made sense with a task queue and an MCP transport. status now prints what the daemon actually does: the engine, the turn in flight and the recent ring." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Delete the worker path

**Files:**
- Delete: `src/sous/tasks.py`, `src/sous/toolexec.py`, `src/sous/worker.py`, `src/sous/proxy.py`, `src/sous/context.py`, `scripts/e2e_smoke.py`, `tests/test_commands.py`, `tests/test_confinement.py`, `tests/test_tasks.py`, `tests/test_worker.py`, `tests/test_proxy.py`, `tests/test_context.py`, `tests/daemon_stub.py`
- Modify: `src/sous/protocol.py` (`_tool` 31-43, `WORKER_TOOLS` 46-107, `WORKER_TOOLSET` 208, `parse_tool_calls` 246), `src/sous/server.py` (`_login_shell_path` 539-568, `_install_shutdown_handler` 604-632, `main`), `tests/test_protocol.py`, `tests/test_server.py`, `tests/test_config.py:15`, `CLAUDE.md:17`

**Interfaces:**
- Produces: `parse_tool_calls(text: str, toolset: ToolSet) -> list[ToolCall]` (toolset required).

- [ ] **Step 1: Delete the modules and their tests**

```bash
git rm src/sous/tasks.py src/sous/toolexec.py src/sous/worker.py src/sous/proxy.py src/sous/context.py scripts/e2e_smoke.py \
  tests/test_commands.py tests/test_confinement.py tests/test_tasks.py tests/test_worker.py tests/test_proxy.py tests/test_context.py tests/daemon_stub.py
```

- [ ] **Step 2: Remove the worker's schema from `protocol.py` and make the toolset explicit**

In `src/sous/protocol.py`: delete `_tool` (31-43), `WORKER_TOOLS` (46-107) and `WORKER_TOOLSET` (208); change line 246 to `def parse_tool_calls(text: str, toolset: ToolSet) -> list[ToolCall]:`. Callers: `src/sous/gateway/response.py:169` already passes one; `src/sous/tune/suite/loop.py` passes `TOOLSET`.

In `tests/test_protocol.py`: delete `test_worker_tools_names_and_shape` (17), `test_default_toolset_is_the_strict_worker_set` (327) and `EXPECTED_TOOLS` (5-14) if nothing else uses it; add `from sous.tune.payload import TOOLSET` and turn every one-argument `parse_tool_calls(<expr>)` into `parse_tool_calls(<expr>, TOOLSET)`:

```bash
uv run python - <<'EOF'
import re, pathlib
p = pathlib.Path("tests/test_protocol.py")
s = p.read_text()
s = re.sub(r"parse_tool_calls\(([^,()]*(?:\([^()]*\))?[^,()]*)\)", r"parse_tool_calls(\1, TOOLSET)", s)
p.write_text(s)
EOF
grep -n "parse_tool_calls(" tests/test_protocol.py | grep -v TOOLSET
```
The final grep must print nothing; fix any call the regex missed by hand.

- [ ] **Step 3: Remove the worker leftovers from `server.py`**

- `_install_shutdown_handler`: drop the `terminate_active_commands()` call and the `sous.toolexec` import.
- Delete `_login_shell_path` (539-568) and its use in `main` (the `if login_path := _login_shell_path(): os.environ["PATH"] = login_path` block and its comment): it existed so sandboxed commands resolved like the user's shell, and the daemon runs no commands now. Delete the tests in `tests/test_server.py` whose names contain `login` (`grep -n "def test_.*login" tests/test_server.py`).
- In `tests/test_config.py` delete the test at line 15 (it imports `sous.toolexec`); the rest of the allowlist tests go in Task 10.
- `CLAUDE.md`: delete line 17 (`uv run python scripts/e2e_smoke.py — agent loop against a tiny real model.`); Task 13 adds its replacement.

- [ ] **Step 4: Run everything**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
grep -rn "sous.tasks\|sous.toolexec\|sous.worker\|sous.proxy\|sous.context\|WORKER_TOOLS" src tests scripts
```
Expected: all four checks PASS; the grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor!: remove the worker loop, the task queue and the sandbox" -m "Claude Code's permission system, auto mode and sandbox now gate every action a local subagent takes, and the endpoint is the product. The application-level sandbox, the queue and the stdio bridge guarded a path nothing recommends any more." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Config: `[model]` is the engine, `[server]` is the endpoint

**Files:**
- Modify: `src/sous/config.py` (whole file: `_KNOWN` 45-73, constants 76-78, `SousConfig` 82-153, `_warn_unknown` 171-179, `_gateway_values` 307-369, `_upstream_url` 403-440, `load_config` 443-485; delete `DEFAULT_ALLOWLIST` 19-43, `_context_values`, `current_allowlist`, `persist_allowlist_entry`), `src/sous/engine/base.py:295-318` (`default_engine_factory`), `src/sous/gateway/turn.py:137`, `src/sous/gateway/routes.py` and `upstream.py` (field names), `src/sous/server.py` (`status_document`, `create_server`, `main`), `src/sous/cli.py:171,255-345`, `src/sous/tune/arms.py` (`Arm` 18-34, `_arm` 109-152, `quick_arms` 162-166 and the `gateway_window` computation near 192), `src/sous/tune/decide.py:50-60`, `src/sous/tune/report.py`, `src/sous/tune/bench.py`
- Test: `tests/test_config.py`, `tests/test_cli.py` (`_status` 832-851, 1401-1441, 1466-1472), `tests/test_server.py:813`, `tests/test_monitor.py:92-108`, `tests/test_gateway_http.py:56,71`, `tests/test_tune_*.py`

**Interfaces:**
- Produces: `SousConfig` fields `server_port`, `upstream_url`, `local_models: tuple[str, ...]`, `generation_timeout_minutes`, `model_id`, `idle_unload_minutes`, `max_context_tokens` (default `131072`), the sampler / prompt-cache / speculative / int8 fields unchanged, `data_dir`, `config_path`. Constants `MIN_CONTEXT_TOKENS = 48 * 1024`, `DEFAULT_UPSTREAM = "https://api.anthropic.com"`. `load_config(config_path: Path | None = None) -> SousConfig` warns once for 0.6 settings. Status document `config` is flat: `model_id`, `idle_unload_minutes`, `port`, `local_models`, `max_context_tokens`, `upstream_url`, `generation_timeout_minutes`. `Arm` loses `gateway_window` and `fit_gateway_window`; `_arm(user, cand, drafter_id, block, f, own, *, current=None)`.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_config.py`:

```python
def test_settings_from_0_6_are_ignored_with_one_warning_naming_their_new_homes(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[gateway]\nenabled = true\nlocal_models = ["sous-local"]\nmax_context_tokens = 131072\n\n'
        '[budgets]\nmax_turns = 3\n\n[commands]\nallowlist = ["pytest"]\n\n'
        '[context]\nmode = "auto"\n\n[tasks]\nretention = 5\n'
    )
    with pytest.warns(UserWarning) as caught:
        cfg = load_config(path)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    m = messages[0]
    assert m.startswith("sous config: settings from 0.6 ignored: ")
    for note in (
        "[gateway].enabled (removed: the endpoint is always on)",
        "[gateway].local_models (now [server].local_models)",
        "[gateway].max_context_tokens (now [model].max_context_tokens; the served window is 131072)",
        "[budgets] (removed with the worker path)",
        "[commands] (removed with the worker path)",
        "[context] (removed with the worker path)",
        "[tasks] (removed with the worker path)",
    ):
        assert note in m
    assert cfg.max_context_tokens == 131072
    assert cfg.local_models == ("sous-local",)


def test_a_worker_era_window_is_clamped_to_the_floor(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nmax_context_tokens = 32768\n")
    with pytest.warns(
        UserWarning,
        match=r"\[model\]\.max_context_tokens 32768 is below Claude Code's 49152-token floor",
    ):
        cfg = load_config(path)
    assert cfg.max_context_tokens == 49152


def test_server_and_model_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.upstream_url == "https://api.anthropic.com"
    assert cfg.local_models == ("sous-local",)
    assert cfg.generation_timeout_minutes == 30
    assert cfg.max_context_tokens == 131072
    assert not hasattr(cfg, "gateway_enabled")
```

Run: `uv run pytest tests/test_config.py -k "0_6 or worker_era or server_and_model_defaults" -v`
Expected: FAIL

- [ ] **Step 2: Rewrite `config.py`**

Replace the known-keys table, constants and dataclass:

```python
_KNOWN = {
    "server": {"port", "upstream_url", "local_models", "generation_timeout_minutes"},
    "model": {
        "id",
        "idle_unload_minutes",
        "max_context_tokens",
        "temperature",
        "top_p",
        "top_k",
        "prompt_cache",
        "prompt_cache_gb",
        "speculative_draft_id",
        "speculative_block_size",
        "int8_prefill",
    },
}

# Settings 0.7.0 removed with the worker path, and where the ones that moved
# now live: a config written for 0.6 is told so once at load, instead of
# steering nothing in silence.
_OBSOLETE: dict[str, dict[str, str | None]] = {
    "gateway": {
        "enabled": None,
        "local_models": "[server].local_models",
        "upstream_url": "[server].upstream_url",
        "generation_timeout_minutes": "[server].generation_timeout_minutes",
        "max_context_tokens": "[model].max_context_tokens",
    },
    "budgets": {},
    "commands": {},
    "context": {},
    "tasks": {},
}

MIN_CONTEXT_TOKENS = 48 * 1024
DEFAULT_UPSTREAM = "https://api.anthropic.com"
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class SousConfig:
    server_port: int = 8383
    upstream_url: str = DEFAULT_UPSTREAM
    local_models: tuple[str, ...] = ("sous-local",)
    generation_timeout_minutes: int = 30
    model_id: str = "mlx-community/Qwen3.8-27B-4bit"
    idle_unload_minutes: int = 30
    # The served window: what `sous claude` exports as CLAUDE_CODE_MAX_CONTEXT_TOKENS
    # and the render cap of a local turn. Below 49152 Claude Code compacts
    # constantly, so the floor is enforced at load.
    max_context_tokens: int = 131072
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    speculative_draft_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    speculative_block_size: int = 3
    int8_prefill: bool = False
    prompt_cache: bool = True
    prompt_cache_gb: float | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    config_path: Path = DEFAULT_CONFIG_PATH
```

(keep the existing rationale comments on the sampler, speculative, int8 and prompt-cache fields.) Delete `DEFAULT_ALLOWLIST`, `_context_values`, `current_allowlist`, `persist_allowlist_entry` and `_gateway_values`; drop the `tomlkit` and `shlex` imports if nothing in the file still uses them (the dependency stays: tune writes config files with tomlkit).

`_warn_unknown` skips obsolete sections:

```python
def _warn_unknown(raw: dict) -> None:
    for section, values in raw.items():
        if section in _OBSOLETE:
            continue
        ...  # unchanged
```

Add:

```python
def _warn_obsolete(raw: dict, window: int) -> None:
    notes: list[str] = []
    for section, keys in _OBSOLETE.items():
        values = raw.get(section)
        if values is None:
            continue
        if not keys or not isinstance(values, dict):
            notes.append(f"[{section}] (removed with the worker path)")
            continue
        for key in values:
            home = keys.get(key)
            if key not in keys:
                notes.append(f"[{section}].{key} (unknown)")
            elif home is None:
                notes.append(f"[{section}].{key} (removed: the endpoint is always on)")
            elif key == "max_context_tokens":
                notes.append(f"[{section}].{key} (now {home}; the served window is {window})")
            else:
                notes.append(f"[{section}].{key} (now {home})")
    if notes:
        warnings.warn(
            "sous config: settings from 0.6 ignored: "
            + "; ".join(notes)
            + " — see README, Upgrading from 0.6",
            stacklevel=3,
        )


def _server_values(server: dict) -> tuple[tuple[str, ...], int]:
    """local_models and generation_timeout_minutes, validated."""
    # paste the `models` validation from the old _gateway_values (lines 319-344)
    # and the `timeout` validation (360-367), with every message saying
    # [server].local_models / [server].generation_timeout_minutes
    return tuple(models), timeout


def _model_window(model: dict) -> int:
    window = model.get("max_context_tokens", 131072)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        warnings.warn(
            f"sous config: [model].max_context_tokens {window!r} must be a positive "
            "integer; using 131072",
            stacklevel=3,
        )
        return 131072
    if window < MIN_CONTEXT_TOKENS:
        warnings.warn(
            f"sous config: [model].max_context_tokens {window} is below Claude Code's "
            f"{MIN_CONTEXT_TOKENS}-token floor; using {MIN_CONTEXT_TOKENS}",
            stacklevel=3,
        )
        return MIN_CONTEXT_TOKENS
    return window
```

`_upstream_url(server: dict)` keeps its body; its message says `[server].upstream_url` and `DEFAULT_UPSTREAM`. `load_config` becomes:

```python
def load_config(config_path: Path | None = None) -> SousConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_toml(path)
    _warn_unknown(raw)
    server = _section(raw, "server")
    model = _section(raw, "model")
    window = _model_window(model)
    _warn_obsolete(raw, window)
    local_models, generation_timeout = _server_values(server)
    return SousConfig(
        server_port=server.get("port", 8383),
        upstream_url=_upstream_url(server),
        local_models=local_models,
        generation_timeout_minutes=generation_timeout,
        model_id=model.get("id", "mlx-community/Qwen3.8-27B-4bit"),
        idle_unload_minutes=model.get("idle_unload_minutes", 30),
        max_context_tokens=window,
        temperature=model.get("temperature", 0.7),
        top_p=model.get("top_p", 0.8),
        top_k=model.get("top_k", 20),
        prompt_cache=model.get("prompt_cache", True),
        prompt_cache_gb=_prompt_cache_gb(model),
        speculative_draft_id=model.get("speculative_draft_id", "z-lab/Qwen3.8-27B-DFlash2"),
        speculative_block_size=_speculative_block_size(model),
        int8_prefill=_int8_prefill(model),
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
        config_path=path,
    )
```

- [ ] **Step 3: Rename the fields everywhere, then fix what a rename cannot**

```bash
grep -rl "gateway_max_context_tokens\|gateway_local_models\|gateway_upstream_url\|gateway_generation_timeout_minutes\|GATEWAY_MIN_CONTEXT_TOKENS\|GATEWAY_DEFAULT_UPSTREAM" src tests | xargs sed -i '' \
  -e 's/gateway_max_context_tokens/max_context_tokens/g' \
  -e 's/gateway_local_models/local_models/g' \
  -e 's/gateway_upstream_url/upstream_url/g' \
  -e 's/gateway_generation_timeout_minutes/generation_timeout_minutes/g' \
  -e 's/GATEWAY_MIN_CONTEXT_TOKENS/MIN_CONTEXT_TOKENS/g' \
  -e 's/GATEWAY_DEFAULT_UPSTREAM/DEFAULT_UPSTREAM/g'
grep -rn "gateway_enabled\|gateway_window\|fit_gateway_window\|serving_window\|\[gateway\]\|\"gateway\"" src tests
```

Resolve every hit of the second grep:
- `src/sous/engine/base.py` `default_engine_factory`: `reserve_tokens=config.max_context_tokens` (the `max(...)` and its comment go).
- `src/sous/server.py`: `status_document["config"]` becomes the flat seven keys listed in Interfaces; `create_server` mounts the gateway unconditionally (`mounted_gateway.append(...)` with no `if`); `main()` logs `f"serving {', '.join(config.local_models)} at http://127.0.0.1:{config.server_port}/v1/messages; everything else is forwarded to {config.upstream_url}"` unconditionally, without "(experimental)".
- `src/sous/cli.py` `_cmd_claude`: delete the `gateway = status.get("config", {}).get("gateway", {})` / `if not gateway.get("enabled")` block; read `local_models = status.get("config", {}).get("local_models")` and `max_context_tokens = status.get("config", {}).get("max_context_tokens")`; the drift check compares `config.local_models` and `config.max_context_tokens`; the docstring at 171 no longer mentions `enabled`.
- `src/sous/tune/arms.py`: `Arm` drops `gateway_window` and `fit_gateway_window` (and a `serving_window` property becomes `return self.window` or is removed if `window` serves); `_arm(user, cand, drafter_id, block, f, own, *, current=None)` builds `config = dataclasses.replace(user, model_id=cand.id, speculative_draft_id=drafter_id, speculative_block_size=block if drafter_id else user.speculative_block_size, max_context_tokens=min(user.max_context_tokens, f.window))` and `Arm(label=..., config=config, model_id=cand.id, drafter_id=drafter_id, block_size=block if drafter_id else 0, window=min(user.max_context_tokens, f.window), tier=cand.tier, current=current, fit_window=min(user.max_context_tokens, own.window), int8_prefill=user.int8_prefill, greedy=user.temperature == 0)`; `quick_arms`: `window = user.max_context_tokens` and `floor = MIN_CONTEXT_TOKENS`; delete the `gateway_window = min(...)` computation and every `gateway_window` argument.
- `src/sous/tune/decide.py` `_changes`: delete the gateway branch (the `if user.gateway_enabled and gateway is not None and gateway < ...: changes["gateway"] = {...}` lines and the `gateway` value that feeds them); `src/sous/tune/report.py` and `bench.py`: read `arm.window` where they read the gateway window.
- `tests/test_gateway_http.py:56,71`: drop `gateway_enabled=True`.
- `tests/test_cli.py` `_status()` (832-851): `"config"` becomes `{"model_id": "org/m", "idle_unload_minutes": 30, "port": 8383, "local_models": ["sous-local"], "max_context_tokens": 131072, "upstream_url": "https://api.anthropic.com", "generation_timeout_minutes": 30}`; line 1472 → `del status["config"]["local_models"]`; the assertion at 1441 compares the flat dict; delete any test asserting the "has the gateway off" refusal (`grep -n "gateway off" tests/test_cli.py`).
- `tests/test_server.py:813-823` and the Task 6 shape test: the flat seven keys.
- `tests/test_monitor.py:102-107`: the flat keys.
- `tests/test_tune_*.py`: `Arm(...)` constructions drop `gateway_window=`/`fit_gateway_window=`; `SousConfig(gateway_enabled=...)` kwargs go; assertions on a proposed `{"gateway": {...}}` diff go (one window is proposed, under `model`).
- `tests/test_config.py`: delete the allowlist tests (73, 81, 99, 109, 137, 149, 159, 171, 179, 186, 198, 210, 223), the context tests (231, 240, 247, 263), the defaults assertions at 38-43 and line 51; rewrite the `[gateway]` tests (424-577): `enabled` tests are deleted, `local_models` / `upstream_url` / `generation_timeout_minutes` tests write a `[server]` table and assert the renamed fields, `max_context_tokens` tests write a `[model]` table.

- [ ] **Step 4: Run everything**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
grep -rn "gateway_enabled\|gateway_window\|\[gateway\]" src tests
```
Expected: PASS; the grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "refactor(config)!: [model] is the engine, [server] is the endpoint" -m "The worker's window, budgets, allowlist and context policy leave the config; the endpoint's keys move under [server] and its window under [model], since there is one window now. A config written for 0.6 gets one warning naming each setting's new home, and a worker-era window is clamped to the floor rather than honoured." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: A plain Starlette app, no MCP SDK

**Files:**
- Modify: `src/sous/monitor.py` (`mount_monitor` 184-270), `src/sous/gateway/routes.py` (`_MCP_PATH` 99, `Gateway.passthrough` redirect ~490-497, `mount_gateway` 901-935), `src/sous/server.py` (`create_server`, `serve` 671-676, `main`, imports), `pyproject.toml:45`, `uv.lock`
- Test: `tests/test_gateway_http.py:43-76`, `tests/test_server.py`, `tests/test_monitor.py`

**Interfaces:**
- Produces: `monitor_routes(engines: EngineManager, status: Callable[[], dict], version: Callable[[], object]) -> list[BaseRoute]`; `Gateway.routes() -> list[BaseRoute]`; `mount_gateway(engines, config, *, upstream=None, inflight=None) -> Gateway`; `create_server(engines, config, *, upstream=None, inflight=None) -> Starlette`; `serve(app: Starlette, host: str, port: int) -> None`.

- [ ] **Step 1: Write the failing lifespan test**

Append to `tests/test_server.py` (mark it the way `tests/test_monitor.py` marks its async tests):

```python
async def test_the_lifespan_runs_the_idle_sweep_and_closes_the_endpoint(tmp_path):
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    app = create_server(engines, cfg, upstream=FakeUpstream().upstream())
    assert isinstance(app, Starlette)
    async with app.router.lifespan_context(app):
        assert any(t.name == "sous-idle-sweep" for t in threading.enumerate())
    assert not any(t.name == "sous-idle-sweep" for t in threading.enumerate())
```

Run: `uv run pytest tests/test_server.py -k lifespan_runs -v`
Expected: FAIL (`create_server` returns an `MCPServer`)

- [ ] **Step 2: Routes as lists**

`src/sous/monitor.py`: rename `mount_monitor` to `monitor_routes(engines, status, version) -> list[BaseRoute]`, keep the handler closures, and return in this order:

```python
    return [
        Route("/sous/status", sous_status, methods=["GET"]),
        Route("/sous/events", sous_events, methods=["GET"]),
        Route("/sous/hold", sous_hold, methods=["POST"]),
        Route("/sous/unload", sous_unload, methods=["POST"]),
        # After the real routes and before the endpoint's catch-all, so no
        # path under /sous can reach the upstream. The bare /sous needs its
        # own entry: `/sous/{path:path}` does not match it.
        Route("/sous", sous_unknown, methods=list(ALL_METHODS)),
        Route("/sous/{path:path}", sous_unknown, methods=list(ALL_METHODS)),
    ]
```

with `from starlette.routing import BaseRoute, Route` and no `MCPServer` import.

`src/sous/gateway/routes.py`: delete `_MCP_PATH` and the `/mcp/` redirect block in `passthrough` (the catch-all forwards everything it gets); `mount_gateway(engines, config, *, upstream=None, inflight=None) -> Gateway` keeps the logger pins and returns `Gateway(engines, config, upstream, inflight)`; add

```python
    def routes(self) -> list[BaseRoute]:
        return [
            Route("/v1/messages", self.messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", self.count_tokens, methods=["POST"]),
            # Matched last: a method the two routes above do not take
            # (GET /v1/messages) falls through to the upstream's own answer.
            Route("/{path:path}", self.passthrough, methods=list(ALL_METHODS)),
        ]
```

- [ ] **Step 3: The app**

`src/sous/server.py`:

```python
def create_server(
    engines: EngineManager,
    config: SousConfig,
    *,
    upstream: Upstream | None = None,
    inflight: Inflight | None = None,
) -> Starlette:
    # One registry: the endpoint's turns write it, the status routes read it.
    inflight = inflight or Inflight()
    svc = SousService(engines, config, inflight)
    gateway = mount_gateway(engines, config, upstream=upstream, inflight=inflight)

    @contextlib.asynccontextmanager
    async def _lifespan(_: Starlette) -> AsyncIterator[None]:
        engines.start_idle_sweep()
        try:
            yield None
        finally:
            # Fires on every path that drives the app's ASGI lifespan —
            # uvicorn's graceful shutdown and its SIGTERM path alike. Without
            # it the session thread stays parked in _requests.get() and never
            # reaches release_mlx_thread_state() (ml-explore/mlx#4327), and
            # the upstream forwarder's connection pool stays open.
            engines.stop_idle_sweep()
            await gateway.aclose()

    # The daemon's own routes go on before the endpoint's: the endpoint ends
    # with a catch-all that forwards upstream, and a /sous/ path must never
    # get there.
    return Starlette(
        routes=[
            *monitor_routes(engines, lambda: svc.status_document(recent=True), svc.status_version),
            *gateway.routes(),
        ],
        lifespan=_lifespan,
    )


def serve(app: Starlette, host: str, port: int) -> None:
    """uvicorn with a bounded graceful shutdown (see uvicorn_config)."""
    anyio.run(uvicorn.Server(uvicorn_config(app, host, port)).serve)
```

Remove the `configure_daemon_logging()` re-run inside `create_server` and the comments about the SDK's RichHandler (in `create_server` and at the top of `main`); `main()` calls `app = create_server(engines, config)` and `serve(app, "127.0.0.1", config.server_port)`. Imports: `from starlette.applications import Starlette`; drop `from mcp.server import MCPServer`.

- [ ] **Step 4: Drop the dependency**

In `pyproject.toml` delete the `"mcp>=2.0,<3",` line and rewrite the three comments that justify `anyio`, `sse-starlette`/`starlette` and `httpx` as "mcp already pulls it in" — they are simply direct imports now. Then:

```bash
uv lock
grep -c '^name = "mcp"' uv.lock
```
Expected: `0`.

- [ ] **Step 5: Fix the tests that built the app**

- `tests/test_gateway_http.py` `_gateway_app` (67-76): `Starlette(routes=mount_gateway(engines, cfg).routes())`, no `MCPServer`.
- Any test of the `/mcp/` redirect (`grep -n '"/mcp' tests`): delete.
- `tests/test_server.py`: tests calling `serve(mcp, ...)` or asserting `MCPServer` pass the Starlette app instead; the existing test that the lifespan closes the gateway keeps passing through `app.router.lifespan_context`.
- `tests/test_monitor.py`: any use of `mount_monitor` becomes `Starlette(routes=monitor_routes(...))`.

- [ ] **Step 6: Run everything and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
grep -rn "from mcp\|import mcp\|MCPServer" src tests pyproject.toml
git add -A src tests pyproject.toml uv.lock
git commit -m "refactor(server)!: the daemon is a Starlette app; the MCP SDK goes" -m "Nothing speaks MCP any more. The app is the monitor routes and the endpoint routes in that order, with the lifespan owning the idle sweep and the endpoint's shutdown." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```
Expected: PASS; the grep prints nothing.

---

### Task 12: Names: `sous.api`, `Endpoint`, `Daemon`

**Files:**
- Rename: `src/sous/gateway/` → `src/sous/api/`; `tests/test_gateway_*.py` → `tests/test_api_*.py`
- Modify: every importer (`src/sous/server.py`, `src/sous/monitor.py`, `src/sous/loopback.py`), `src/sous/logs.py:5,22`, `src/sous/engine/base.py:427`, `src/sous/api/turn.py` (`GatewayBusy`), the tests

- [ ] **Step 1: Move and rename mechanically**

```bash
git mv src/sous/gateway src/sous/api
for f in tests/test_gateway_*.py; do git mv "$f" "${f/test_gateway_/test_api_}"; done
grep -rl "sous\.gateway\|sous/gateway" src tests scripts CLAUDE.md CONTRIBUTING.md README.md | xargs sed -i '' -e 's/sous\.gateway/sous.api/g' -e 's#sous/gateway#sous/api#g'
grep -rl "mount_gateway\|GatewayBusy\|Gateway\|SousService" src tests | xargs perl -pi -e \
  's/mount_gateway/mount_endpoint/g; s/GatewayBusy/EndpointBusy/g; s/\bGateway\b/Endpoint/g; s/SousService/Daemon/g'
grep -rn "gateway" src/sous/server.py src/sous/monitor.py src/sous/api/routes.py | grep -v "^.*#"
```

Rename the remaining local variables and docstrings the last grep shows (`gateway = mount_endpoint(...)` → `endpoint`, "the gateway's turns" → "the endpoint's turns"); the logger name is now `"sous.api"` and the module docstring of `src/sous/logs.py` names it.

- [ ] **Step 2: Run everything and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add -A
git commit -m "refactor: the endpoint package is sous.api; SousService is Daemon" -m "One product, one set of names: the package that serves /v1/messages is the api, the object that builds the status document is the daemon, and no name says gateway any more." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: `scripts/api_smoke.py`

**Files:**
- Create: `scripts/api_smoke.py`, `tests/test_api_smoke.py`
- Modify: `CLAUDE.md` (the Commands list)

**Interfaces:**
- Produces: `build_request(model: str = "sous-local") -> dict` (pure; tested), `main() -> int`.
- Consumes: `sous.api.convert.chat_tools(tools) -> tuple[list[dict], <dropped>]` (as `routes.py` calls it).

- [ ] **Step 1: Write the failing test**

`tests/test_api_smoke.py`:

```python
import importlib.util
from pathlib import Path

from sous.api.convert import chat_tools


def _smoke():
    path = Path(__file__).resolve().parents[1] / "scripts" / "api_smoke.py"
    spec = importlib.util.spec_from_file_location("api_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_smoke_request_is_one_the_endpoint_serves_locally():
    body = _smoke().build_request()
    assert body["model"] == "sous-local"
    assert body["stream"] is True
    tools, dropped = chat_tools(body["tools"])
    assert len(tools) == 1
    assert not dropped
    assert body["messages"][0]["role"] == "user"
```

Run: `uv run pytest tests/test_api_smoke.py -v`
Expected: FAIL (`scripts/api_smoke.py` does not exist)

- [ ] **Step 2: Write the script**

`scripts/api_smoke.py`:

```python
"""One served turn against a tiny real model: the check to run after a
dependency change, since CI cannot load a model.

Starts `sous serve` on a free port with a config and data dir under a temp
HOME, posts one streaming /v1/messages request for `sous-local` that carries
a tool, and passes when a tool_use block streams back. Judged by the block,
not by the model finishing its turn: a 0.6B model cannot reliably close a
turn, and the plumbing is what this checks.

    uv run python scripts/api_smoke.py
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TINY = "mlx-community/Qwen3-0.6B-4bit"
START_TIMEOUT = 120.0
TURN_TIMEOUT = 600.0


def build_request(model: str = "sous-local") -> dict:
    return {
        "model": model,
        "max_tokens": 200,
        "stream": True,
        "system": "You are a coding assistant. Use the write_file tool to write files.",
        "tools": [
            {
                "name": "write_file",
                "description": "Write a file at path with content.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Write hello.txt containing exactly: hello from sous"}
        ],
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise SystemExit("the daemon did not open its port")


def main() -> int:
    import httpx

    port = _free_port()
    home = Path(tempfile.mkdtemp(prefix="sous-smoke-"))
    sous_dir = home / ".sous"
    sous_dir.mkdir()
    # An empty drafter id disables speculative decoding, as tune's drafterless
    # arms do; the floor window keeps the KV reserve small for a 0.6B model.
    (sous_dir / "config.toml").write_text(
        f'[server]\nport = {port}\n\n[model]\nid = "{TINY}"\nmax_context_tokens = 49152\n'
        'speculative_draft_id = ""\n'
    )
    # HOME points the daemon at the temp config and data dir; HF_HOME keeps
    # the real model cache so the tiny model is not downloaded again.
    env = {
        **os.environ,
        "HOME": str(home),
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    }
    daemon = subprocess.Popen(
        [sys.executable, "-c", "from sous.cli import main; main()", "serve"],
        env=env,
        stdout=sys.stderr,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(port, time.monotonic() + START_TIMEOUT)
        saw_tool_use = False
        with (
            httpx.Client(timeout=TURN_TIMEOUT, trust_env=False) as client,
            client.stream(
                "POST", f"http://127.0.0.1:{port}/v1/messages", json=build_request()
            ) as reply,
        ):
            print(f"status {reply.status_code}", file=sys.stderr)
            for line in reply.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                block = event.get("content_block") or {}
                if event.get("type") == "content_block_start" and block.get("type") == "tool_use":
                    saw_tool_use = True
                    print(f"tool_use: {json.dumps(block)}", file=sys.stderr)
                if event.get("type") == "message_delta":
                    stop = (event.get("delta") or {}).get("stop_reason")
                    print(f"stop_reason: {stop}", file=sys.stderr)
        verdict = "PASS: a tool_use block came back" if saw_tool_use else "FAIL: no tool_use block"
        print(verdict, file=sys.stderr)
        return 0 if saw_tool_use else 1
    finally:
        daemon.send_signal(signal.SIGTERM)
        try:
            daemon.wait(30)
        except subprocess.TimeoutExpired:
            daemon.kill()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the test, then the script once for real (local, needs the tiny model)**

```bash
uv run pytest tests/test_api_smoke.py -v
uv run python scripts/api_smoke.py
```
Expected: the test PASSES; the script prints `status 200`, a `tool_use:` line and `PASS`, exit 0. If the model is not cached, the first run downloads it (~400 MB).

- [ ] **Step 4: Name it in CLAUDE.md**

Add to the Commands list in `CLAUDE.md`: ``- `uv run python scripts/api_smoke.py` — one served turn against a tiny real model; run it after any dependency change, since CI cannot load a model.``

- [ ] **Step 5: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add scripts/api_smoke.py tests/test_api_smoke.py CLAUDE.md
git commit -m "feat(scripts): api_smoke drives one served turn against a tiny model" -m "The old smoke drove the worker loop. The plumbing worth checking after a dependency bump is now the endpoint: a request in, a tool_use block out." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: Assets and packaging

**Files:**
- Delete: `server.json`, `.github/workflows/publish-mcp-registry.yml`, `skills/delegating-to-local/SKILL.md`
- Modify: `pyproject.toml` (`description` line 7, `keywords` 15-27, `[project.scripts]` 94-98), `uv.lock`

- [ ] **Step 1: Delete the MCP assets**

```bash
git rm server.json .github/workflows/publish-mcp-registry.yml
git rm -r skills
grep -rn "server.json\|skills/" .github pyproject.toml CONTRIBUTING.md
```
The grep shows only lines Task 15 rewrites (docs) or nothing.

- [ ] **Step 2: Describe the package as what it is**

In `pyproject.toml`:
- `description = "The sous-chef for Claude's kitchen: serve Claude Code's subagents from a local MLX model on Apple silicon, so heavy Claude Code use stretches further on the same plan"`
- `keywords = ["claude", "claude-code", "subagent", "local-llm", "local-model", "mlx", "apple-silicon", "qwen", "ai-agents"]`
- `[project.scripts]` keeps only `sous = "sous.cli:main"`.

```bash
uv lock
uv build
```
Expected: the wheel builds; `unzip -p dist/*.whl '*/entry_points.txt'` lists only `sous`.

- [ ] **Step 3: Run the four checks and commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add -A pyproject.toml uv.lock
git commit -m "chore: drop the MCP registry manifest, the delegation skill and the sous-mcp script" -m "Nothing registers as an MCP server any more; the distribution name stays until the bare name on PyPI is granted." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 15: Documentation

**Files:**
- Rewrite: `README.md`
- Modify: `CLAUDE.md` (description 3-6; gotchas 45-47; the Security boundary section 267-286; every `[gateway].` key name), `CONTRIBUTING.md` (20-31, 93-105, 106-167), `docs/tuning.md` (31-41)

- [ ] **Step 1: README**

Write the new `README.md` in this order. "Carry" means move the named lines of the current file with only the edits stated; every other section is new text given here.

1. Title and tagline (new):

```markdown
# sous

The sous-chef for Claude's kitchen: serve Claude Code's subagents from a
local MLX model on your Mac, so heavy Claude Code use stretches further on
the same plan. Claude designs the menu, the local model does the prep, and
Claude Code runs every tool itself under its own permission system.
```

2. `## Why` — carry lines 12-25; replace the last sentence of the second paragraph with: "sous serves the subagents — the volume-heavy half of the work — from a local model and leaves the main loop on the frontier model you are paying for."

3. `## How sous compares` (new):

```markdown
## How sous compares

- **Full-local replacements** point Claude Code (or a fork of it) at a local
  endpoint: every task drops to local-model quality, including the ones you
  wanted a frontier model for. sous keeps the main loop on the frontier model
  and serves only the subagents locally.
- **Bare routers and proxies** swap models per request and stop there. sous is
  one at its core — a byte-for-byte transparent proxy that stores no
  credential — and adds what makes a local subagent usable: a prompt cache
  whose forks are shared across sessions and projects (a new session's first
  local turn went from 170 s to 16 s on the default model), retained turn slots
  so the conversation Claude Code branches every 30 s keeps its cache, a live
  terminal and attributed turn lines, `sous tune` to settle the model and its
  window for your machine, and model holds with idle unload.
- **What gates the local model is Claude Code, not sous.** Claude Code executes
  every tool call under its permission mode: in auto mode a frontier
  classifier reviews shell commands, network access and protected-path writes,
  reads and edits inside the working directory are approved as they are for
  any subagent, and the optional sandbox adds an OS-level filesystem and
  network boundary. sous runs nothing. See [Security model](#security-model).
```

4. `## Requirements` — carry 56-63.

5. `## Install` (new; replaces 64-149, which described `claude mcp add`, the Claude Desktop bridge and the daemon management — the launchd paragraphs from `### Managing the daemon` (107-149) are carried into this section after the name paragraph, minus any sentence about `sous mcp` or the queue):

```markdown
## Install

    uv tool install sous-mcp

The package is `sous-mcp` and the command is `sous`. The name dates from the
MCP server sous shipped as before 0.7.0; the bare name `sous` on PyPI is held
by an abandoned placeholder and a PEP 541 request to claim it is open
(pypi/support#11881). When it is granted the package will be `sous`.
```

6. `## Quick start` (new):

```markdown
## Quick start

    sous serve            # or: sous install-launchd, to start at login
    sous claude           # Claude Code with its subagents served locally

`sous claude` launches Claude Code with `ANTHROPIC_BASE_URL` pointing at the
daemon, `CLAUDE_CODE_SUBAGENT_MODEL=sous-local` (forced) and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` set to the served window, and holds the model
loaded while the session lives. Every argument after `claude` passes through
to Claude Code. The first run downloads the model (~16 GB for the default),
once. Claude Desktop is not supported: it cannot point Claude Code at a local
base URL.
```

7. `## How it works` — carry 161-258 and 290-330 (the routing and forwarding; `[gateway]`-named settings become `[server]`/`[model]` names, "gateway" becomes "the endpoint" or "sous", and every "(experimental)" goes), then carry the "what a locally served turn gives up" list 330-347 and 401-443 with the **No sandbox** bullet replaced by: "**No sandbox of sous's own.** The endpoint returns `tool_use` blocks and never runs a tool; Claude Code's permission mode is the boundary (see Security model)."

7b. `## Prompt cache and continuity` — carry the cache bullets 348-400 (one turn at a time and keyed slots; usage split; mid-conversation system messages) under their own heading, as prose rather than "gives up" bullets where the sentence allows.

8. `## Observability` — carry 259-289 (`sous top`, `sous statusline`) and add: "`sous status` prints the same document once: the engine, the turn in flight and the last five turns."

9. `## Configuration — ~/.sous/config.toml` — carry 528-725 with the table reduced to the two sections below and every explanation of a `[budgets]`, `[commands]`, `[context]`, `[tasks]` or `[gateway].enabled` key removed; `[gateway]` key explanations move under `[server]` (and `max_context_tokens` under `[model]`):

```toml
[server]
port = 8383
upstream_url = "https://api.anthropic.com"
local_models = ["sous-local"]
generation_timeout_minutes = 30

[model]
id = "mlx-community/Qwen3.8-27B-4bit"
idle_unload_minutes = 30
max_context_tokens = 131072
temperature = 0.7
top_p = 0.8
top_k = 20
prompt_cache = true
# prompt_cache_gb = 8
speculative_draft_id = "z-lab/Qwen3.8-27B-DFlash2"
speculative_block_size = 3
int8_prefill = false
```

10. `## Tuning` — carry 726-793. `## Smaller machines` — carry 794-827 minus any worker sentence.

11. `## Security model` (new):

```markdown
## Security model

sous executes nothing. The daemon binds to `127.0.0.1`, refuses foreign
`Host`/`Origin` values and cross-site fetch metadata on every route, forwards
the `Authorization` header Claude Code sends to `[server].upstream_url`
unmodified and nowhere else, stores no credential, and never logs a request
body, header value or query string.

The boundary around the local model is Claude Code's permission system, the
same one that governs a frontier subagent:

- **Auto mode** (the default starting mode on Pro, Max and Team) sends the
  local subagent's shell commands, network operations and protected-path
  writes, plus its spawn description and its final report, to a frontier
  classifier (Claude Sonnet 5 by default). Through sous each check is one
  request forwarded upstream, about 1.5 s. Reads and edits inside the working
  directory are approved without a check. Three consecutive or twenty total
  blocks fall back to prompting you.
- **Manual mode** prompts for every non-read action the local model takes;
  under `-p`, prompts become denials unless you pass `--permission-mode auto`.
- **The sandbox** (`/sandbox` in Claude Code) adds an OS-level filesystem and
  network boundary around every shell command, local or frontier.

These are Claude Code behaviours, read from its documentation on permission
modes and classifier billing at version 2.1.27x; sous adds no permission flag
and prints no notice about them.
```

12. `## Validation status` (new; replaces 886-911):

```markdown
## Validation status

Validated end to end on an M5 Pro / 64 GB with the default model
(`mlx-community/Qwen3.8-27B-4bit`) and its DFlash drafter:

- **Cross-session prompt-cache forks** (2026-09-06): a new Claude Code
  session's first local turn went from 170 s to 16 s once another session had
  built the tools fork, across processes and projects.
- **Same-session forks** (2026-09-04): 8.3 s against 168 s cold for the
  second subagent of a session.
- **Continuity** (2026-09-13/14): retained turn slots kept a subagent's cache
  through the progress-summary branches Claude Code makes every 30 s, at a
  262144-token window.
- **Auto mode** (2026-09-14): with a `sous-local` subagent under Claude Code's
  auto mode, each classified action was one Sonnet 5 request through sous,
  1.2–1.8 s; reads and working-directory edits were approved without one.

Tool-call parsing accepts both the XML-ish `<function=…>/<parameter=…>`
format Qwen3 emits and the hermes JSON format other MLX models use.
```

13. `## Limitations` — carry 912-933 minus the `e2e_smoke.py` bullet; the first bullet says "delays subsequent turns" instead of tasks; the last bullet loses "(experimental)" and names `[server].local_models` and `[server].upstream_url`.

14. `## Upgrading from 0.6` (new):

```markdown
## Upgrading from 0.6

0.7.0 removed the MCP server, the task queue, the worker and its sandbox.

- Remove the MCP registration: `claude mcp remove sous`, and any `sous mcp`
  entry in `claude_desktop_config.json`. Claude Desktop is no longer supported.
- Config: `[gateway]` keys moved — `local_models`, `upstream_url` and
  `generation_timeout_minutes` to `[server]`, `max_context_tokens` to
  `[model]`; `enabled` is gone (the endpoint is always on); `[budgets]`,
  `[commands]`, `[context]` and `[tasks]` are gone. The daemon warns once at
  startup about any of them and starts anyway. A `[model].max_context_tokens`
  written for the worker (32768) is clamped to the 49152 floor; set it to the
  window you want.
- `~/.sous/tasks/` and `~/.sous/tasks.db` are no longer read and can be
  deleted.
- `sous wait` and `sous mcp` are gone; `sous status` now prints the engine and
  the turns.
```

15. `## Development` — carry 934-end, replacing `scripts/e2e_smoke.py` with `scripts/api_smoke.py` and deleting any sentence about the sandbox tests. Delete the `<!-- mcp-name: ... -->` comment at line 3.

Then: `grep -n -i "mcp\|worker\|delegat\|allowlist\|sandboxed\|toolexec\|experimental" README.md` — every remaining hit must be in the Install name paragraph, the Upgrading section or the Security model's sandbox bullet.

- [ ] **Step 2: CLAUDE.md**

- Lines 3-6 become: "sous is a local daemon that serves Claude Code's subagents from a local MLX model and forwards everything else upstream unchanged; Claude Code executes every tool under its own permission system. macOS / Apple silicon only. The goal is plan economics: subagent output is generated locally for free so heavy Claude Code use stretches further — evaluate features against that goal."
- Delete the two gotcha bullets at 45-47 (`e2e_smoke.py` outcomes; `budget-exhausted`).
- Every `[gateway].max_context_tokens` becomes `[model].max_context_tokens`, every `[gateway].` other key becomes `[server].`; "the gateway" in prose becomes "the endpoint" where it names the package or its routes.
- Replace the `## Security boundary` section with:

```markdown
## Security boundary

sous executes nothing. `src/sous/api/` never runs a tool (Claude Code does,
under its own permission mode: auto mode's classifier, the allow/deny rules
and the sandbox are the boundary around the local model) and never logs a
request body, header value or query string. Its lines go through `sous.logs`
(one timestamped, levelled shape on the root handler); `_log_turn` runs on
the event loop, so nothing that blocks may ever be added to it — do such
work on the turn's thread. It forwards the client's credentials to
`[server].upstream_url` and nowhere else, and stores none. `sous claude`
(`cli.py`) never sets `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, a tier
variable or a permission mode. Do not add confinement, allowlists or approval
flows to sous: the daemon guards its own HTTP surface (`sous/loopback.py`)
and nothing else. A change that makes any of these otherwise needs the spec
(`docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`, as amended by
`docs/superpowers/specs/2026-09-19-remove-mcp-path-design.md`) changed first.

`src/sous/tune/suite/tools.py` runs commands on copies of the suite's
synthetic projects inside `sous tune` only. It is a test harness, not a
boundary, and must never be reachable from the daemon.
```

- [ ] **Step 3: CONTRIBUTING.md and docs/tuning.md**

- `CONTRIBUTING.md`: delete `## Touching the sandbox` (93-105); retitle `## Verifying the gateway endpoint` to `## Verifying the endpoint` and open it with "`uv run python scripts/api_smoke.py` is the short form; the recipe below drives a whole Claude Code session."; in `## What's in scope` (20-31) replace "changes to the task lifecycle or MCP surface" with "changes to the endpoint's request or response contract or to the status document".
- `docs/tuning.md` lines 31-41: "the worker's allowlist is the shipped one plus `python -m unittest`" → "the suite loop runs only a task's verify commands and `python -m unittest` / `python -m pytest`"; "the worker's tools read, write and edit files and run allowlisted commands, and neither `rm` nor `mv` is allowlisted" → "the suite's tools read, write and edit files and run those commands; there is no `rm` or `mv`"; the bullet beginning "Allowlisting `python -m unittest` runs the worker's own code" → "Running `python -m unittest` executes the candidate's own code on your machine, like every test runner would; the suite's tools confine file edits to the scratch copy, not what its tests execute."; "with the worker's project as the working directory … import the worker's modules" → "the candidate's project".

- [ ] **Step 4: Check and commit**

```bash
grep -rn -i "mcp\|delegate_to_local\|toolexec\|e2e_smoke\|\[gateway\]\|\[budgets\]\|\[commands\]\|\[context\]\|\[tasks\]" README.md CLAUDE.md CONTRIBUTING.md docs/tuning.md
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add README.md CLAUDE.md CONTRIBUTING.md docs/tuning.md
git commit -m "docs: sous is the endpoint" -m "The README, the contributor guide and the working notes described an MCP daemon with a sandboxed worker. They now describe what ships: a daemon that serves Claude Code's subagents locally, with Claude Code's permission system as the boundary, and how to move a 0.6 install." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```
The grep's only hits are the README's Install and Upgrading sections and CLAUDE.md's spec paths.

---

### Task 16: Version 0.7.0

**Files:**
- Modify: `pyproject.toml:6`, `uv.lock`

- [ ] **Step 1: Bump and lock**

```bash
sed -i '' 's/^version = "0.6.1"$/version = "0.7.0"/' pyproject.toml
uv lock
uv run sous --help
```
Expected: the help lists exactly `serve`, `status`, `statusline`, `top`, `stop`, `install-launchd`, `uninstall-launchd`, `tune`, `claude`.

- [ ] **Step 2: The four checks, then commit**

```bash
uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check
git add pyproject.toml uv.lock
git commit -m "chore: bump version to 0.7.0" -m "The MCP delegate path is gone; the endpoint is the product." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## After the last task

1. Push the branch and open one PR against `main` titled `refactor!: remove the MCP delegate path and refocus on the endpoint`, whose description summarises the spec's Decision, Scope and Upgrading sections and links the spec and this plan. All four CI jobs green.
2. After merge: tag `v0.7.0` (tags trigger nothing; `tag.gpgsign` signs it); reinstall the daemon on the M5 Pro from `main`; run `uv run python scripts/api_smoke.py`; run one real `sous claude` session in auto mode whose subagent runs on `sous-local`, and check the daemon log for its turn lines and one forwarded `claude-sonnet-5` classifier request.
3. Close #36, #56, #103 and #28 as moot, each with a one-line comment naming 0.7.0.
