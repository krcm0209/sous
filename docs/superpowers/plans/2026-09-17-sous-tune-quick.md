# `sous tune --quick` Implementation Plan (PR 1 of the sous tune spec)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `sous tune --quick`: detect the hardware, fit the curated candidates to memory, ask consent per download, measure every quality-neutral arm through sous's own engine classes, print the report and the `config.toml` diff, and apply on a yes.

**Architecture:** A new `sous.tune` package runs as its own process, one engine per arm on a dedicated thread, and reads throughput from the engine's per-turn prompt-cache gauges. The daemon gains one loopback route, `POST /sous/unload`, backed by `EngineManager.unload_now()`, so the tune can free the weights without the user touching the daemon. Nothing in this PR changes the model or any quality-affecting key: the quick rule chooses only drafter, block size and a window that fits.

**Tech Stack:** Python 3.14, mlx / mlx-vlm / mlx-lm (function-local imports only), huggingface_hub (sizes, cache checks, downloads), tomlkit (formatting-preserving config edits), httpx (daemon calls, already used by `sous.cli`), pytest with `tests/fake_engine.py`.

**Spec:** `docs/superpowers/specs/2026-09-17-sous-tune-design.md` — this plan implements its "Delivery" item 1; the suite, the full run and `--discover` are PRs 2 and 3.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid and used in this repo.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- `mlx`, `mlx_lm`, `mlx_vlm` imports stay function-local (the lint job runs on ubuntu). Every thread that touched mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits.
- No new dependencies. `uv lock --check` must stay green.
- CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. Run all four before every commit.
- Tests never touch the real `~/.sous` or `~/.cache/huggingface`: every path comes from `tmp_path`, every network call is injected.
- `pytest` `addopts = "-q"` already; never add another `-q`.
- Comments explain non-obvious *why*; never cite this plan, the spec, task or step numbers in code or test comments.
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Never put the maintainer's name or work home path in commits, files or PR text.
- Line length 100 (ruff); ruff rules E, F, I, UP, B, SIM.

## File Structure

| file | responsibility |
|---|---|
| `src/sous/engine/base.py` (modify) | `EngineManager.unload_now()` — the sweep's refusals without its clock |
| `src/sous/monitor.py` (modify) | `POST /sous/unload` route, 409 on refusal |
| `src/sous/tune/__init__.py` (create) | `main(args)`: orchestration of the quick run, prompts, exit codes |
| `src/sous/tune/hardware.py` (create) | `Hardware` record and `detect()` |
| `src/sous/tune/candidates.py` + `candidates.toml` (create) | curated table, `describe()` from a checkpoint's config, drafter compatibility, the fit rule |
| `src/sous/tune/hub.py` (create) | snapshot sizes, cache checks, the download plan, consent, fetching |
| `src/sous/tune/arms.py` (create) | `Arm` and the quick-mode enumeration |
| `src/sous/tune/rundir.py` (create) | `~/.sous/tune/<run-id>/` with `results.jsonl`, `hardware.json`, `report.md` |
| `src/sous/tune/bench.py` (create) | the throughput bench of one arm through the real engine |
| `src/sous/tune/decide.py` (create) | the quick-mode rule and the config changes it yields |
| `src/sous/tune/report.py` (create) | plain-text report, tomlkit diff, backup and apply, restart note |
| `src/sous/tune/daemon.py` (create) | preconditions: status read, unload request, refusal reasons |
| `src/sous/cli.py` (modify) | the `tune` subcommand |
| `README.md` (modify) | "Tuning" section; the memory table points at `sous tune` |
| `tests/test_engine_base.py`, `tests/test_monitor.py` (modify) | unload_now and the route |
| `tests/tune_fixtures.py` (create) | config.json dicts for the candidates, shared by the tune tests |
| `tests/test_tune_*.py` (create) | one file per module |
| `tests/test_tune_model.py` (create) | model-marked: one quick arm on the 0.6B |

---

### Task 1: `EngineManager.unload_now()`

**Files:**
- Modify: `src/sous/engine/base.py` (after `unload_if_idle`, ~line 766)
- Test: `tests/test_engine_base.py`

**Interfaces:**
- Consumes: `EngineManager._changed`, `_prune_holders`, `_engine`, `_leases`, `_holders`, `_unloading`, `_bump` — all existing.
- Produces: `EngineManager.unload_now() -> dict` returning `{"unloaded": bool, "reason": str | None}`. Reasons are the literal strings `"nothing loaded"`, `"a generation is in flight"`, `"the engine is leased by a turn"`, `"held by N session(s)"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py`:

```python
def test_unload_now_frees_a_fresh_engine_and_reports_it():
    """The tune asks the daemon to release the weights whatever the idle
    clock says: the refusals are the sweep's, the clock is not."""
    mgr, created = _manager(idle_minutes=30)
    mgr.get()
    mgr.touch()
    assert mgr.unload_now() == {"unloaded": True, "reason": None}
    assert created[0].unloaded is True
    assert mgr.status()["loaded"] is False


def test_unload_now_refuses_with_a_reason_when_nothing_is_loaded():
    mgr, _ = _manager()
    assert mgr.unload_now() == {"unloaded": False, "reason": "nothing loaded"}


def test_unload_now_refuses_under_a_lease_a_holder_and_a_generation():
    created: list[FakeEngine] = []

    def factory(model_id: str):
        e = FakeEngine([])
        created.append(e)
        return e

    # The default holder check asks psutil about the pid; a fake pid would be
    # pruned before the refusal is reached.
    mgr = EngineManager(SousConfig(), engine_factory=factory, holder_alive=lambda pid, ct: True)
    engine = mgr.get()
    with mgr.lease():
        assert mgr.unload_now() == {"unloaded": False, "reason": "the engine is leased by a turn"}
    mgr.hold(4242, 1.0)
    assert mgr.unload_now() == {"unloaded": False, "reason": "held by 1 session(s)"}
    mgr._holders.clear()
    with engine._gen_lock:
        assert mgr.unload_now() == {"unloaded": False, "reason": "a generation is in flight"}
    assert created[0].unloaded is False
    assert mgr.unload_now()["unloaded"] is True


def test_unload_now_bumps_the_version_and_clears_the_idle_clock():
    mgr, _ = _manager()
    mgr.get()
    before = mgr.version
    mgr.unload_now()
    assert mgr.version > before
    assert mgr.status()["idle_seconds"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py -k unload_now -v`
Expected: 4 failures, `AttributeError: 'EngineManager' object has no attribute 'unload_now'`.

- [ ] **Step 3: Implement `unload_now`**

In `src/sous/engine/base.py`, directly after `unload_if_idle`:

```python
    def unload_now(self) -> dict:
        """Free the weights on request — `sous tune` needs the memory — with
        every refusal the idle sweep makes and none of its clock: a session
        holding the model, a leased engine or an in-flight generation each
        keep it, and the caller learns which. `idle_seconds` reads None
        afterwards, as after any unload."""
        with self._changed:
            self._prune_holders()
            if self._engine is None:
                return {"unloaded": False, "reason": "nothing loaded"}
            if self._engine.generation_in_flight():
                return {"unloaded": False, "reason": "a generation is in flight"}
            if self._leases:
                return {"unloaded": False, "reason": "the engine is leased by a turn"}
            if self._holders:
                return {"unloaded": False, "reason": f"held by {len(self._holders)} session(s)"}
            engine, self._engine = self._engine, None
            self._last_used = None
            self._unloading = True
            self._bump()
        try:
            engine.unload()
        finally:
            with self._changed:
                self._unloading = False
                self._bump()
                self._changed.notify_all()
        return {"unloaded": True, "reason": None}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_base.py -v`
Expected: all pass (the four new ones included).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/engine/base.py tests/test_engine_base.py
git commit -m "feat(engine): unload the model on request, with the sweep's refusals" -m "sous tune runs as its own process and needs the daemon's memory; it must never wait out the idle clock or fight a live session for it, so the request is refused for exactly the reasons the idle sweep refuses." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `POST /sous/unload`

**Files:**
- Modify: `src/sous/monitor.py` (add the handler next to `sous_hold`, register it before the catch-all `sous_unknown` routes)
- Test: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `EngineManager.unload_now()` from Task 1; `check_loopback`, `_refused`, `_served`, `RequestError`, `_invalid` — existing.
- Produces: `POST /sous/unload` → 200 `{"unloaded": true, "reason": null}`; 409 with the Anthropic-shaped error body `{"type": "error", "error": {"type": "conflict_error", "message": <reason>}}` when refused; a non-empty body is a 400 like a malformed hold; loopback and fetch-metadata refusals exactly as `/sous/hold`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_monitor.py` (the file already has `_app`, `_request`, `_await_loaded`):

```python
def test_unload_frees_a_loaded_idle_engine(tmp_path: Path):
    app, engines = _app(tmp_path)
    engines.get()
    r = _request(app, "POST", "/sous/unload", b"")
    assert r.status_code == 200
    assert r.json() == {"unloaded": True, "reason": None}
    assert engines.status()["loaded"] is False


def test_unload_is_a_409_with_the_reason_while_a_session_holds_the_model(tmp_path: Path):
    app, engines = _app(tmp_path)
    engines.get()
    engines.hold(os.getpid(), 1.0)
    r = _request(app, "POST", "/sous/unload", b"")
    assert r.status_code == 409
    assert r.json()["error"] == {"type": "conflict_error", "message": "held by 1 session(s)"}
    assert engines.status()["loaded"] is True


def test_unload_with_nothing_loaded_is_a_409_too(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "POST", "/sous/unload", b"")
    assert r.status_code == 409
    assert r.json()["error"]["message"] == "nothing loaded"


def test_unload_refuses_a_body_and_a_cross_site_caller_like_hold(tmp_path: Path):
    app, _ = _app(tmp_path)
    r = _request(app, "POST", "/sous/unload", {"pid": 1})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    r = _request(app, "POST", "/sous/unload", b"", headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403
```

If `_app` in this file does not already load a `FakeEngine` on `engines.get()`, note that its default factory is `lambda mid: FakeEngine([])`, which is enough here. `os` is already imported in the file (the hold tests use `os.getpid()`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_monitor.py -k unload -v`
Expected: 4 failures with status 404 (the catch-all answers).

- [ ] **Step 3: Add the handler and register it**

In `src/sous/monitor.py`, add a body guard next to `_hold_body`:

```python
async def _require_empty_body(request: Request) -> None:
    """`/sous/unload` takes no body: one that carries something is a request
    for a different route, refused the way a malformed hold is."""
    try:
        async for chunk in request.stream():
            if chunk:
                raise _invalid("unload takes no body")
    except ClientDisconnect:
        raise _invalid("client disconnected mid-body") from None
```

and, inside `mount_monitor`, after `sous_hold`:

```python
    async def sous_unload(request: Request) -> Response:
        try:
            check_loopback(request)
            await _require_empty_body(request)
        except RequestError as e:
            return _refused(e)
        reply = await _served("POST /sous/unload", engines.unload_now)
        if reply.status_code != 200:
            return reply
        body = json.loads(bytes(reply.body))
        if body.get("unloaded"):
            return reply
        # The engine is busy or empty: a state the caller must wait out or
        # respect, not an error in its request.
        return _refused(RequestError(409, "conflict_error", body["reason"]))
```

Register it before the catch-alls, next to the hold registration:

```python
    mcp.custom_route("/sous/unload", methods=["POST"])(sous_unload)
```

Update the `mount_monitor` docstring's route list to name `POST /sous/unload`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: all pass. If the 400 test fails because `_invalid` produces a different `type`, read `sous/gateway/convert.py:_invalid` and assert the type it produces; do not change `_invalid`.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -m "not model" -x
git add src/sous/monitor.py tests/test_monitor.py
git commit -m "feat(monitor): POST /sous/unload releases the model for a tune run" -m "A benchmark cannot share the machine with the daemon's resident weights. The route lets sous tune ask for the memory without the user stopping the daemon, and a 409 with the sweep's own reason tells it what to wait for." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 3: Hardware detection

**Files:**
- Create: `src/sous/tune/__init__.py` (empty docstring module for now; `main` arrives in Task 10)
- Create: `src/sous/tune/hardware.py`
- Test: `tests/test_tune_hardware.py`

**Interfaces:**
- Consumes: `sous.engine.int8prefill.availability() -> Availability(available: bool, reason: str | None)`.
- Produces:

```python
@dataclass(frozen=True)
class Hardware:
    chip: str
    memory_bytes: int
    working_set_bytes: int
    architecture: str
    nax: bool
    nax_reason: str | None
    macos: str
    versions: dict[str, str]        # {"mlx": ..., "mlx-vlm": ..., "mlx-lm": ..., "sous-mcp": ...}
    hub_cache: str
    disk_free_bytes: int

    def as_dict(self) -> dict: ...

def detect(*, device_info=None, mac_ver=None, disk_usage=None, availability=None,
           version=None, hub_cache=None) -> Hardware
def gib(n: int) -> str            # "51.8 GiB"
```

Every keyword is an injection point for tests; production passes none and the function reaches for `mlx.core.device_info`, `platform.mac_ver`, `shutil.disk_usage`, `int8prefill.availability`, `importlib.metadata.version` and `huggingface_hub.constants.HF_HUB_CACHE`.

- [ ] **Step 1: Write the failing test**

`tests/test_tune_hardware.py`:

```python
from sous.engine.int8prefill import Availability
from sous.tune.hardware import Hardware, detect, gib


def _info():
    return {
        "device_name": "Apple M5 Pro",
        "memory_size": 68719476736,
        "max_recommended_working_set_size": 55662788608,
        "architecture": "applegpu_g17s",
    }


def test_detect_reads_every_field_through_its_injection_point(tmp_path):
    hw = detect(
        device_info=_info,
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda path: (1, 2, 412 * 2**30),
        availability=lambda: Availability(True),
        version=lambda name: {"mlx": "0.32.2", "mlx-vlm": "0.7.1", "mlx-lm": "0.31.3",
                              "sous-mcp": "0.6.1"}[name],
        hub_cache=str(tmp_path),
    )
    assert hw == Hardware(
        chip="Apple M5 Pro",
        memory_bytes=68719476736,
        working_set_bytes=55662788608,
        architecture="applegpu_g17s",
        nax=True,
        nax_reason=None,
        macos="26.6.2",
        versions={"mlx": "0.32.2", "mlx-vlm": "0.7.1", "mlx-lm": "0.31.3", "sous-mcp": "0.6.1"},
        hub_cache=str(tmp_path),
        disk_free_bytes=412 * 2**30,
    )
    assert hw.as_dict()["working_set_bytes"] == 55662788608


def test_a_pre_m5_gpu_reports_why_int8_is_unavailable(tmp_path):
    hw = detect(
        device_info=lambda: {**_info(), "device_name": "Apple M2", "architecture": "applegpu_g14g"},
        mac_ver=lambda: ("15.5", ("", "", ""), "arm64"),
        disk_usage=lambda path: (1, 2, 3),
        availability=lambda: Availability(False, "macOS 15.5 < 26.2"),
        version=lambda name: "x",
        hub_cache=str(tmp_path),
    )
    assert hw.nax is False and hw.nax_reason == "macOS 15.5 < 26.2"


def test_a_missing_distribution_reads_as_absent_not_a_crash(tmp_path):
    from importlib.metadata import PackageNotFoundError

    def version(name):
        raise PackageNotFoundError(name)

    hw = detect(device_info=_info, mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
                disk_usage=lambda p: (1, 2, 3), availability=lambda: Availability(True),
                version=version, hub_cache=str(tmp_path))
    assert hw.versions == {"mlx": "absent", "mlx-vlm": "absent", "mlx-lm": "absent",
                           "sous-mcp": "absent"}


def test_gib_formats_one_decimal():
    assert gib(55662788608) == "51.8 GiB"
    assert gib(0) == "0.0 GiB"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_tune_hardware.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune'`.

- [ ] **Step 3: Implement**

`src/sous/tune/__init__.py`:

```python
"""`sous tune`: benchmark this machine, pick the settings, show the diff."""
```

`src/sous/tune/hardware.py`:

```python
"""What the tune knows about the machine before any model is loaded."""

from __future__ import annotations

import dataclasses
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as dist_version

DISTRIBUTIONS = ("mlx", "mlx-vlm", "mlx-lm", "sous-mcp")
_GIB = 1 << 30


@dataclass(frozen=True)
class Hardware:
    chip: str
    memory_bytes: int
    working_set_bytes: int
    architecture: str
    nax: bool
    nax_reason: str | None
    macos: str
    versions: dict[str, str]
    hub_cache: str
    disk_free_bytes: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def gib(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


def _device_info() -> dict:
    import mlx.core as mx

    return dict(mx.device_info())


def _availability():
    from sous.engine import int8prefill

    return int8prefill.availability()


def _hub_cache() -> str:
    from huggingface_hub import constants

    return str(constants.HF_HUB_CACHE)


def detect(
    *,
    device_info: Callable[[], dict] | None = None,
    mac_ver: Callable[[], tuple] | None = None,
    disk_usage: Callable[[str], tuple] | None = None,
    availability: Callable[[], object] | None = None,
    version: Callable[[str], str] | None = None,
    hub_cache: str | None = None,
) -> Hardware:
    """Every reading through an injectable so the tests need no GPU. The mlx
    call is a device query, not an op, so it is safe on any thread."""
    info = (device_info or _device_info)()
    release = (mac_ver or platform.mac_ver)()[0]
    cache = hub_cache if hub_cache is not None else _hub_cache()
    free = int((disk_usage or shutil.disk_usage)(cache)[2])
    nax = (availability or _availability)()
    read = version or dist_version
    versions: dict[str, str] = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = read(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return Hardware(
        chip=str(info.get("device_name", "unknown")),
        memory_bytes=int(info.get("memory_size", 0)),
        working_set_bytes=int(info.get("max_recommended_working_set_size", 0)),
        architecture=str(info.get("architecture", "")),
        nax=bool(getattr(nax, "available", False)),
        nax_reason=getattr(nax, "reason", None),
        macos=release,
        versions=versions,
        hub_cache=cache,
        disk_free_bytes=free,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_hardware.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/__init__.py src/sous/tune/hardware.py tests/test_tune_hardware.py
git commit -m "feat(tune): detect the machine a tune run measures" -m "The report header and the fit rule both need the chip, the Metal working set, NAX eligibility and the library versions; reading them through injectables keeps the tests off the GPU." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Candidates — the curated table, `describe()`, drafter compatibility, the fit rule

**Files:**
- Create: `src/sous/tune/candidates.toml`
- Create: `src/sous/tune/candidates.py`
- Create: `tests/tune_fixtures.py`
- Test: `tests/test_tune_candidates.py`

**Interfaces:**
- Consumes: `sous.context.kv_bytes_per_token(config) -> int | None`, `sous.context.native_max_tokens(config) -> int | None`, `sous.engine.base.select_backend(config) -> str`, `sous.engine.base.fetch_model_config(model_id) -> dict`, `sous.config.GATEWAY_MIN_CONTEXT_TOKENS`.
- Produces:

```python
@dataclass(frozen=True)
class Candidate:
    id: str
    tier: str
    drafters: tuple[str, ...]
    note: str = ""

@dataclass(frozen=True)
class Table:
    checked: datetime.date
    validated_with: str
    trusted_orgs: tuple[str, ...]
    publishers: tuple[str, ...]
    candidates: tuple[Candidate, ...]

def load_table(path: Path | None = None) -> Table

@dataclass(frozen=True)
class QuantSummary:
    mode: str | None            # base mode, e.g. "affine", "mxfp4"; None = unquantized
    bits: int | None
    group_size: int | None
    overrides: int              # per-layer entries in the quantization block
    verifier_fast: bool         # base layers take mlx-vlm's fast exact verifier
    verifier_slow_layers: int   # overrides that fall to the per-token path
    int8_routable: bool         # base layers are affine 4-bit, group size 64
    int8_excluded_layers: int   # overrides int8 prefill leaves on the stock path

def summarize_quantization(config: dict) -> QuantSummary

@dataclass(frozen=True)
class Checkpoint:
    id: str
    model_type: str
    backend: str                # "vlm" | "lm"
    bytes: int | None           # None when unknown (offline, uncached)
    kv_bytes_per_token: int | None
    native_max_tokens: int | None
    hidden_size: int | None
    num_layers: int | None
    num_target_layers: int | None   # drafters only
    quant: QuantSummary

def describe(model_id: str, *, config_fn=None, size_fn=None) -> Checkpoint
def drafter_compatible(target: Checkpoint, drafter: Checkpoint) -> tuple[bool, str]
DRAFTER_QUANT_DIVISOR = 3.5
SLACK_BYTES = 2 * 2**30
WINDOW_STEP = 256

@dataclass(frozen=True)
class Fit:
    fits: bool
    window: int                 # the window this arm runs at (lowered when needed)
    needed_bytes: int           # at that window
    working_set_bytes: int
    detail: str                 # the arithmetic, one line

def fit(target: Checkpoint, drafter: Checkpoint | None, *, window: int, floor: int,
        working_set_bytes: int) -> Fit
```

`fit` rule: `needed = bytes + drafter.bytes / DRAFTER_QUANT_DIVISOR + kv × window + SLACK_BYTES`. If that exceeds the working set, the window is lowered to the largest multiple of `WINDOW_STEP` that fits; below `floor` the candidate does not fit. Unknown `bytes` or `kv_bytes_per_token` means `fits=False` with `detail` saying which is unknown.

- [ ] **Step 1: Write the fixtures**

`tests/tune_fixtures.py`:

```python
"""config.json shapes of the checkpoints the tune reasons about, reduced to
the keys sous reads. Hybrid layer lists follow Qwen3.5: every fourth layer
is full attention."""


def _layer_types(n: int) -> list[str]:
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]


def qwen_27b(quantization: dict | None = None, model_type: str = "qwen3_5") -> dict:
    return {
        "model_type": model_type,
        "vision_config": {"model_type": "qwen3_5"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "num_attention_heads": 24,
            "head_dim": 256,
            "hidden_size": 5120,
            "max_position_embeddings": 262144,
            "layer_types": _layer_types(64),
        },
        "quantization": quantization or {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def qwen_9b() -> dict:
    return {
        "model_type": "qwen3_5",
        "vision_config": {"model_type": "qwen3_5"},
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "num_attention_heads": 16,
            "head_dim": 256,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "layer_types": _layer_types(32),
        },
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def oq4_quantization() -> dict:
    """A 4-bit base with sensitivity boosts: 160 layers at 5-bit and one at
    6-bit, the shape mlx-community/Qwen3.8-27B-oQ4 ships."""
    q: dict = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(160):
        q[f"language_model.model.layers.{i % 64}.linear_attn.out_proj.{i}"] = {
            "group_size": 64, "bits": 5, "mode": "affine",
        }
    q["language_model.model.layers.3.self_attn.k_proj"] = {"group_size": 64, "bits": 6,
                                                            "mode": "affine"}
    return q


def dflash2_27b() -> dict:
    return {"model_type": "qwen3", "num_hidden_layers": 5, "hidden_size": 5120,
            "num_target_layers": 64, "head_dim": 128, "num_attention_heads": 32,
            "num_key_value_heads": 8, "max_position_embeddings": 262144}


def dflash_9b() -> dict:
    return {"model_type": "qwen3", "num_hidden_layers": 6, "hidden_size": 4096,
            "num_target_layers": 32, "head_dim": 128, "num_attention_heads": 32,
            "num_key_value_heads": 8, "max_position_embeddings": 262144}


M5_PRO_WORKING_SET = 55662788608   # 51.8 GiB
M2_AIR_WORKING_SET = 11453251584   # 10.7 GiB
GB = 10**9
```

- [ ] **Step 2: Write the failing tests**

`tests/test_tune_candidates.py`:

```python
from datetime import date

import pytest

from sous.config import GATEWAY_MIN_CONTEXT_TOKENS
from sous.tune.candidates import (
    Checkpoint,
    describe,
    drafter_compatible,
    fit,
    load_table,
    summarize_quantization,
)
from tests import tune_fixtures as fx


def test_the_shipped_table_loads_and_names_the_default_model_first():
    table = load_table()
    assert isinstance(table.checked, date)
    assert table.candidates[0].id == "mlx-community/Qwen3.8-27B-4bit"
    assert table.candidates[0].drafters == ("z-lab/Qwen3.8-27B-DFlash2",)
    assert "mlx-community" in table.trusted_orgs and "Qwen" in table.publishers
    tiers = {c.tier for c in table.candidates}
    assert {"27b-dense", "35b-moe", "9b", "4b", "2b"} <= tiers


def test_quantization_summary_of_a_plain_4bit_checkpoint():
    q = summarize_quantization(fx.qwen_27b())
    assert (q.mode, q.bits, q.group_size, q.overrides) == ("affine", 4, 64, 0)
    assert q.verifier_fast and q.verifier_slow_layers == 0
    assert q.int8_routable and q.int8_excluded_layers == 0


def test_quantization_summary_of_oq4_keeps_the_fast_verifier_but_counts_the_6bit_layer():
    q = summarize_quantization(fx.qwen_27b(fx.oq4_quantization()))
    assert q.overrides == 161
    assert q.verifier_fast and q.verifier_slow_layers == 1
    assert q.int8_routable and q.int8_excluded_layers == 161


def test_quantization_summary_of_mxfp4_and_6bit_and_unquantized():
    mx4 = summarize_quantization(fx.qwen_27b({"group_size": 32, "bits": 4, "mode": "mxfp4"}))
    assert mx4.verifier_fast is False and mx4.int8_routable is False
    six = summarize_quantization(fx.qwen_27b({"group_size": 64, "bits": 6, "mode": "affine"}))
    assert six.verifier_fast is False
    none = summarize_quantization({"model_type": "qwen3_5"})
    assert none.mode is None and none.verifier_fast is False


def test_describe_reads_bytes_shape_and_kv_cost_through_its_injection_points():
    cp = describe(
        "mlx-community/Qwen3.8-27B-4bit",
        config_fn=lambda mid: fx.qwen_27b(),
        size_fn=lambda mid: 16_100_000_000,
    )
    assert cp.backend == "vlm" and cp.model_type == "qwen3_5"
    assert cp.bytes == 16_100_000_000
    assert cp.kv_bytes_per_token == 2 * 16 * 4 * 256 * 2   # 16 attention layers, 64 KiB
    assert cp.native_max_tokens == 262144
    assert cp.hidden_size == 5120 and cp.num_layers == 64 and cp.num_target_layers is None


def test_describe_of_a_drafter_records_its_target_layer_count():
    cp = describe("z-lab/Qwen3.8-27B-DFlash2", config_fn=lambda m: fx.dflash2_27b(),
                  size_fn=lambda m: 3_850_000_000)
    assert cp.num_target_layers == 64 and cp.hidden_size == 5120
    assert cp.kv_bytes_per_token is not None   # a drafter's own KV is real but small


def test_drafter_compatibility_matches_hidden_size_and_layer_count():
    target = describe("t", config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 1)
    good = describe("d", config_fn=lambda m: fx.dflash2_27b(), size_fn=lambda m: 1)
    wrong = describe("w", config_fn=lambda m: fx.dflash_9b(), size_fn=lambda m: 1)
    assert drafter_compatible(target, good) == (True, "")
    ok, why = drafter_compatible(target, wrong)
    assert ok is False and "hidden size" in why


def _cp(bytes_, kv=65536, native=262144):
    return Checkpoint(id="m", model_type="qwen3_5", backend="vlm", bytes=bytes_,
                      kv_bytes_per_token=kv, native_max_tokens=native, hidden_size=1,
                      num_layers=1, num_target_layers=None,
                      quant=summarize_quantization(fx.qwen_27b()))


def test_fit_keeps_the_configured_window_when_everything_fits():
    drafter = _cp(3_850_000_000, kv=2 * 5 * 8 * 128 * 2)
    f = fit(_cp(16_100_000_000), drafter, window=131072, floor=8192,
            working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert f.fits and f.window == 131072
    assert "131072" in f.detail and "GiB" in f.detail


def test_fit_lowers_the_window_to_a_256_multiple_before_giving_up():
    f = fit(_cp(6_000_000_000, kv=32768), None, window=131072, floor=8192,
            working_set_bytes=fx.M2_AIR_WORKING_SET)
    assert f.fits and f.window % 256 == 0 and 8192 <= f.window < 131072
    assert f.needed_bytes <= fx.M2_AIR_WORKING_SET


def test_fit_refuses_a_model_that_cannot_reach_the_floor_and_prints_the_arithmetic():
    f = fit(_cp(16_100_000_000), None, window=131072, floor=8192,
            working_set_bytes=fx.M2_AIR_WORKING_SET)
    assert f.fits is False and f.window == 8192
    assert "16.1 GB" in f.detail or "15.0 GiB" in f.detail
    assert "10.7 GiB" in f.detail


def test_fit_refuses_when_a_size_is_unknown():
    f = fit(_cp(None), None, window=8192, floor=8192, working_set_bytes=10**12)
    assert f.fits is False and "unknown" in f.detail


def test_fit_floor_for_the_gateway_is_claude_codes_minimum():
    assert GATEWAY_MIN_CONTEXT_TOKENS == 48 * 1024
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_candidates.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.candidates'`.

- [ ] **Step 4: Write the table**

`src/sous/tune/candidates.toml`:

```toml
# The curated candidates `sous tune` measures. A row carries only what a
# checkpoint's config.json cannot say: its tier, the drafters known to pair
# with it, and a note. Bytes, KV cost, quantization layout and drafter
# compatibility are read from the checkpoint at run time.
checked = 2026-09-17
validated_with = "mlx-vlm 0.7.1"
trusted_orgs = ["mlx-community", "z-lab"]
publishers = ["Qwen"]

[[candidate]]
id = "mlx-community/Qwen3.8-27B-4bit"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "shipped default; affine 4-bit keeps the fast exact verifier"

[[candidate]]
id = "mlx-community/Qwen3.8-27B-oQ4"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "oMLX mixed precision; one 6-bit layer takes the slow verifier path"

[[candidate]]
id = "mlx-community/Qwen3.8-27B-mxfp4"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "fast exact verifier only from mlx-vlm 0.7.1"

[[candidate]]
id = "mlx-community/Qwen3.8-27B-OptiQ-4bit"
tier = "27b-dense"
drafters = ["z-lab/Qwen3.8-27B-DFlash2"]
note = "half the layers at 8-bit: ~30% more bytes per token"

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

- [ ] **Step 5: Implement `candidates.py`**

```python
"""The curated candidate table, what a checkpoint's config says about
itself, and whether it fits this machine."""

from __future__ import annotations

import datetime
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from sous.context import kv_bytes_per_token, native_max_tokens
from sous.engine.base import fetch_model_config, select_backend

# A drafter ships bf16 and sous quantizes it to 4-bit at load: 4 bits of
# code plus group scales against 16, so ~3.5x smaller resident.
DRAFTER_QUANT_DIVISOR = 3.5
# Working memory a turn needs beyond weights and KV: activations, the
# drafter's captured hidden states, the allocator's slack.
SLACK_BYTES = 2 * 2**30
# mlx grows KV buffers in 256-token steps; a window is a multiple of it.
WINDOW_STEP = 256
_GIB = 1 << 30
_GB = 10**9
_FAST_VERIFIER_BITS = (4, 5, 8)


@dataclass(frozen=True)
class Candidate:
    id: str
    tier: str
    drafters: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Table:
    checked: datetime.date
    validated_with: str
    trusted_orgs: tuple[str, ...]
    publishers: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def load_table(path: Path | None = None) -> Table:
    text = path.read_text() if path else files("sous.tune").joinpath("candidates.toml").read_text()
    raw = tomllib.loads(text)
    return Table(
        checked=raw["checked"],
        validated_with=str(raw["validated_with"]),
        trusted_orgs=tuple(raw["trusted_orgs"]),
        publishers=tuple(raw["publishers"]),
        candidates=tuple(
            Candidate(
                id=c["id"], tier=c["tier"], drafters=tuple(c.get("drafters", ())),
                note=c.get("note", ""),
            )
            for c in raw["candidate"]
        ),
    )


@dataclass(frozen=True)
class QuantSummary:
    mode: str | None
    bits: int | None
    group_size: int | None
    overrides: int
    verifier_fast: bool
    verifier_slow_layers: int
    int8_routable: bool
    int8_excluded_layers: int


def _fast(mode: str | None, bits: int | None) -> bool:
    # mlx-vlm's exact verifier runs its fused kernels only for affine 4/5/8-bit
    # layers; every other layout falls to one matmul per verify position.
    return mode == "affine" and bits in _FAST_VERIFIER_BITS


def _routable(mode: str | None, bits: int | None, group: int | None) -> bool:
    return mode == "affine" and bits == 4 and group == 64


def summarize_quantization(config: dict) -> QuantSummary:
    q = config.get("quantization") or config.get("quantization_config") or {}
    mode = q.get("mode", "affine") if q else None
    bits = q.get("bits") if q else None
    group = q.get("group_size") if q else None
    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    slow = sum(
        1 for v in overrides.values() if not _fast(v.get("mode", mode), v.get("bits", bits))
    )
    excluded = sum(
        1
        for v in overrides.values()
        if not _routable(v.get("mode", mode), v.get("bits", bits), v.get("group_size", group))
    )
    return QuantSummary(
        mode=mode,
        bits=bits,
        group_size=group,
        overrides=len(overrides),
        verifier_fast=_fast(mode, bits),
        verifier_slow_layers=slow,
        int8_routable=_routable(mode, bits, group),
        int8_excluded_layers=excluded,
    )


@dataclass(frozen=True)
class Checkpoint:
    id: str
    model_type: str
    backend: str
    bytes: int | None
    kv_bytes_per_token: int | None
    native_max_tokens: int | None
    hidden_size: int | None
    num_layers: int | None
    num_target_layers: int | None
    quant: QuantSummary


def _text(config: dict) -> dict:
    return config.get("text_config", config)


def describe(
    model_id: str,
    *,
    config_fn: Callable[[str], dict] | None = None,
    size_fn: Callable[[str], int | None] | None = None,
) -> Checkpoint:
    """Everything the tune needs from a checkpoint without loading it. The
    size callable is `hub.snapshot_bytes` in production and may answer None
    offline; the config comes through `fetch_model_config` (a cached
    kilobyte, the one network read that needs no consent)."""
    from sous.tune import hub

    config = (config_fn or fetch_model_config)(model_id)
    text = _text(config)
    return Checkpoint(
        id=model_id,
        model_type=str(config.get("model_type", "")),
        backend=select_backend(config),
        bytes=(size_fn or hub.snapshot_bytes)(model_id),
        kv_bytes_per_token=kv_bytes_per_token(config),
        native_max_tokens=native_max_tokens(config),
        hidden_size=text.get("hidden_size"),
        num_layers=text.get("num_hidden_layers"),
        num_target_layers=config.get("num_target_layers"),
        quant=summarize_quantization(config),
    )


def drafter_compatible(target: Checkpoint, drafter: Checkpoint) -> tuple[bool, str]:
    """The checks mlx-vlm's validate_drafter_compatibility makes, made here
    before any download: a drafter reads the target's hidden states, so the
    widths must agree and its declared target depth must be the target's."""
    if drafter.hidden_size != target.hidden_size:
        return False, (
            f"hidden size {drafter.hidden_size} does not match the target's {target.hidden_size}"
        )
    if drafter.num_target_layers is not None and drafter.num_target_layers != target.num_layers:
        return False, (
            f"declares {drafter.num_target_layers} target layers; the target has "
            f"{target.num_layers}"
        )
    return True, ""


@dataclass(frozen=True)
class Fit:
    fits: bool
    window: int
    needed_bytes: int
    working_set_bytes: int
    detail: str


def _needed(target_bytes: int, drafter_bytes: int, kv: int, window: int) -> int:
    return int(target_bytes + drafter_bytes / DRAFTER_QUANT_DIVISOR + kv * window + SLACK_BYTES)


def fit(
    target: Checkpoint,
    drafter: Checkpoint | None,
    *,
    window: int,
    floor: int,
    working_set_bytes: int,
) -> Fit:
    """Weights, the quantized drafter, one window of KV and the slack against
    Metal's recommended working set. A window that does not fit is lowered
    to the largest step that does; below `floor` the candidate is refused,
    and the arithmetic goes into `detail` either way so a refusal explains
    itself."""
    kv = target.kv_bytes_per_token
    dbytes = 0 if drafter is None else drafter.bytes
    if target.bytes is None or kv is None or dbytes is None:
        what = "weights size" if target.bytes is None else "KV cost" if kv is None else "drafter size"
        return Fit(False, window, 0, working_set_bytes, f"{what} unknown (offline?)")
    needed = _needed(target.bytes, dbytes, kv, window)
    chosen = window
    if needed > working_set_bytes:
        room = working_set_bytes - _needed(target.bytes, dbytes, kv, 0)
        chosen = max(0, room // kv) // WINDOW_STEP * WINDOW_STEP
        chosen = min(chosen, window)
        if chosen < floor:
            chosen = floor
        needed = _needed(target.bytes, dbytes, kv, chosen)
    fits = needed <= working_set_bytes
    drafter_note = f" + drafter {dbytes / _GB:.1f} GB/{DRAFTER_QUANT_DIVISOR}" if drafter else ""
    detail = (
        f"weights {target.bytes / _GB:.1f} GB ({target.bytes / _GIB:.1f} GiB){drafter_note}"
        f" + KV {kv} B/token x {chosen} + {SLACK_BYTES / _GIB:.0f} GiB slack"
        f" = {needed / _GIB:.1f} GiB vs working set {working_set_bytes / _GIB:.1f} GiB"
        + ("" if chosen == window else f" (window lowered from {window})")
        + ("" if fits else f"; below the {floor}-token floor")
    )
    return Fit(fits, chosen, needed, working_set_bytes, detail)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_candidates.py -v`
Expected: all pass. `hub.snapshot_bytes` does not exist yet: the tests always inject `size_fn`, and the import inside `describe` is deferred, so nothing fails until Task 5 adds it. If ruff flags the unused deferred import path, keep the import where it is; it resolves in Task 5.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/candidates.py src/sous/tune/candidates.toml tests/tune_fixtures.py tests/test_tune_candidates.py
git commit -m "feat(tune): describe candidates from their checkpoints and fit them to memory" -m "The table carries only what a config.json cannot say. Bytes, KV cost, the verifier and int8 layouts and drafter compatibility come from the checkpoint, so a hand-passed id gets the same treatment as a curated one, and the fit arithmetic is printed with every refusal." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 5: Hub — sizes, cache checks, the download plan, consent, fetching

**Files:**
- Create: `src/sous/tune/hub.py`
- Test: `tests/test_tune_hub.py`

**Interfaces:**
- Consumes: `huggingface_hub.HfApi().model_info(repo_id, files_metadata=True).siblings[*].rfilename/.size`, `huggingface_hub.snapshot_download(repo_id, local_files_only=True)` (raises `huggingface_hub.errors.LocalEntryNotFoundError` when absent), `huggingface_hub.snapshot_download(repo_id)`.
- Produces:

```python
def snapshot_bytes(repo_id: str, *, api=None, cached_path=None) -> int | None
    # cached: the on-disk size of the snapshot; else the Hub's safetensors sizes; None offline
def is_cached(repo_id: str, *, cached_path=None) -> bool

@dataclass(frozen=True)
class Download:
    repo_id: str
    bytes: int | None
    role: str                   # "candidate (27b-dense tier)" | "drafter for <model>" | "reference"
    removes: str                # what a refusal takes away, in the user's words

def plan_downloads(items: list[tuple[str, str, str]], *, cached=is_cached, size=snapshot_bytes)
    -> list[Download]           # items: (repo_id, role, removes); duplicates collapse; cached ones drop
def ask_consent(plan: list[Download], free_bytes: int, *, ask=input, out=print) -> set[str]
def fetch(repo_ids: Iterable[str], *, download=None, out=print) -> None
```

`ask_consent` prints the whole block first (numbered, size, role, free disk, what refusing removes), then asks `Download #N? [y/N] ` for each in order; only `y`/`yes` (case-insensitive) approves. EOF on stdin counts as no.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_hub.py`:

```python
from types import SimpleNamespace

from sous.tune.hub import Download, ask_consent, fetch, is_cached, plan_downloads, snapshot_bytes


class _Api:
    def __init__(self, files):
        self.files = files

    def model_info(self, repo_id, files_metadata=False):
        assert files_metadata is True
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in self.files.items()]
        )


def test_snapshot_bytes_sums_the_hubs_safetensors_sizes_when_not_cached():
    api = _Api({"model.safetensors": 10, "model-2.safetensors": 5, "config.json": 999})
    assert snapshot_bytes("x/y", api=api, cached_path=lambda rid: None) == 15


def test_snapshot_bytes_prefers_the_on_disk_size_of_a_cached_snapshot(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "config.json").write_bytes(b"{}")
    api = _Api({"model.safetensors": 10})
    assert snapshot_bytes("x/y", api=api, cached_path=lambda rid: tmp_path) == 102


def test_snapshot_bytes_is_none_offline():
    class Down:
        def model_info(self, *a, **k):
            raise OSError("no network")

    assert snapshot_bytes("x/y", api=Down(), cached_path=lambda rid: None) is None


def test_is_cached_follows_the_local_only_lookup(tmp_path):
    assert is_cached("x/y", cached_path=lambda rid: tmp_path) is True
    assert is_cached("x/y", cached_path=lambda rid: None) is False


def test_plan_collapses_duplicates_drops_cached_and_keeps_order():
    plan = plan_downloads(
        [
            ("org/big", "candidate (27b-dense tier)", "its arms"),
            ("org/draft", "drafter for org/big", "org/big runs without a drafter"),
            ("org/big", "reference", "its arms"),
            ("org/cached", "candidate (9b tier)", "its arms"),
        ],
        cached=lambda rid: rid == "org/cached",
        size=lambda rid: {"org/big": 20 * 10**9, "org/draft": None}[rid],
    )
    assert [d.repo_id for d in plan] == ["org/big", "org/draft"]
    assert plan[0] == Download("org/big", 20 * 10**9, "candidate (27b-dense tier)", "its arms")
    assert plan[1].bytes is None


def test_consent_prints_the_block_up_front_then_asks_each_separately(capsys):
    plan = [
        Download("org/big", 20 * 10**9, "candidate (35b-moe tier)", "its arms"),
        Download("org/draft", 8 * 10**8, "drafter for org/big", "org/big runs without a drafter"),
    ]
    answers = iter(["y", "n"])
    approved = ask_consent(plan, free_bytes=412 * 2**30, ask=lambda prompt: next(answers))
    out = capsys.readouterr().out
    assert out.index("org/big") < out.index("org/draft") < out.index("Download #1?")
    assert "20.0 GB" in out and "0.8 GB" in out and "412.0 GiB" in out
    assert "org/big runs without a drafter" in out
    assert approved == {"org/big"}


def test_consent_treats_eof_and_anything_but_yes_as_no():
    plan = [Download("a/b", 1, "candidate (2b tier)", "its arms")]

    def eof(prompt):
        raise EOFError

    assert ask_consent(plan, free_bytes=1, ask=eof, out=lambda *a, **k: None) == set()
    assert ask_consent(plan, free_bytes=1, ask=lambda p: "maybe", out=lambda *a, **k: None) == set()
    assert ask_consent(plan, free_bytes=1, ask=lambda p: "YES", out=lambda *a, **k: None) == {"a/b"}


def test_fetch_downloads_each_approved_snapshot_once(capsys):
    seen = []
    fetch(["a/b", "c/d", "a/b"], download=lambda rid: seen.append(rid))
    assert seen == ["a/b", "c/d"]
    assert "a/b" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_hub.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.hub'`.

- [ ] **Step 3: Implement**

`src/sous/tune/hub.py`:

```python
"""Hugging Face Hub reads the tune makes: sizes, cache presence, the
download plan, consent for each snapshot, and the downloads themselves."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

_GB = 10**9
_GIB = 1 << 30


def _cached_path(repo_id: str) -> Path | None:
    """The local snapshot directory, or None: a lookup that never touches the
    network, so an absent snapshot is a plain answer rather than a download."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except LocalEntryNotFoundError, OSError, ValueError:
        return None


def is_cached(repo_id: str, *, cached_path: Callable[[str], Path | None] | None = None) -> bool:
    return (cached_path or _cached_path)(repo_id) is not None


def _hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def snapshot_bytes(
    repo_id: str, *, api=None, cached_path: Callable[[str], Path | None] | None = None
) -> int | None:
    """On disk when cached (every file, symlinks followed), else the Hub's
    safetensors sizes; None when neither can answer (offline, private)."""
    local = (cached_path or _cached_path)(repo_id)
    if local is not None:
        return sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, names in os.walk(local, followlinks=True)
            for f in names
        )
    try:
        info = (api or _hub_api()).model_info(repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001 — offline or refused: unknown, not an error
        return None
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", "")
        if name.endswith(".safetensors"):
            total += int(getattr(sibling, "size", 0) or 0)
    return total or None


@dataclass(frozen=True)
class Download:
    repo_id: str
    bytes: int | None
    role: str
    removes: str


def plan_downloads(
    items: list[tuple[str, str, str]],
    *,
    cached: Callable[[str], bool] = is_cached,
    size: Callable[[str], int | None] = snapshot_bytes,
) -> list[Download]:
    """Every snapshot the arms need that is not on disk, once each, in first-
    mention order — so the user sees one block and answers each entry once."""
    plan: list[Download] = []
    seen: set[str] = set()
    for repo_id, role, removes in items:
        if repo_id in seen or cached(repo_id):
            continue
        seen.add(repo_id)
        plan.append(Download(repo_id, size(repo_id), role, removes))
    return plan


def _size(n: int | None) -> str:
    return "size unknown" if n is None else f"{n / _GB:.1f} GB"


def ask_consent(
    plan: list[Download],
    free_bytes: int,
    *,
    ask: Callable[[str], str] = input,
    out: Callable[..., None] = print,
) -> set[str]:
    """The whole plan first, then one question per snapshot: a user decides
    each download knowing everything the run wants and what a no costs."""
    if not plan:
        return set()
    hub = os.path.join("~", ".cache", "huggingface", "hub")
    out(f"Downloads needed (free disk {free_bytes / _GIB:.1f} GiB at {hub}):")
    for n, d in enumerate(plan, start=1):
        out(f"  {n}. {d.repo_id:45s} {_size(d.bytes):>13s}   {d.role}; refusing: {d.removes}")
    approved: set[str] = set()
    for n, d in enumerate(plan, start=1):
        try:
            answer = ask(f"Download #{n} ({d.repo_id}, {_size(d.bytes)})? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            approved.add(d.repo_id)
    return approved


def fetch(
    repo_ids: Iterable[str],
    *,
    download: Callable[[str], object] | None = None,
    out: Callable[..., None] = print,
) -> None:
    """Every approved snapshot before any measurement, so a bench never
    stalls on the network mid-run."""
    if download is None:
        from huggingface_hub import snapshot_download

        download = snapshot_download
    done: set[str] = set()
    for repo_id in repo_ids:
        if repo_id in done:
            continue
        done.add(repo_id)
        out(f"downloading {repo_id} ...")
        download(repo_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_hub.py tests/test_tune_candidates.py -v`
Expected: all pass (the candidates tests now resolve `hub.snapshot_bytes` too).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/hub.py tests/test_tune_hub.py
git commit -m "feat(tune): plan downloads and ask consent for each snapshot up front" -m "A tune may want tens of gigabytes. The user sees the whole plan with sizes, roles and what a refusal removes, answers each entry once, and every approved snapshot lands before a single measurement runs." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Arms — the quick-mode enumeration

**Files:**
- Create: `src/sous/tune/arms.py`
- Test: `tests/test_tune_arms.py`

**Interfaces:**
- Consumes: `Checkpoint`, `Candidate`, `Fit`, `fit()`, `drafter_compatible()` from Task 4; `SousConfig`, `GATEWAY_MIN_CONTEXT_TOKENS`.
- Produces:

```python
BLOCK_SIZES = (2, 3, 5)

@dataclass(frozen=True)
class Arm:
    label: str                  # "Qwen3.8-27B-4bit + DFlash2 @3"
    config: SousConfig          # the user's config with this arm's keys replaced
    model_id: str
    drafter_id: str             # "" for no drafter
    block_size: int             # 0 for no drafter
    window: int                 # worker window this arm runs at
    gateway_window: int | None  # lowered gateway window, None when the gateway is off
    tier: str
    current: bool               # the user's own configuration
    unvetted: bool = False

@dataclass(frozen=True)
class Refusal:
    model_id: str
    drafter_id: str
    reason: str

def quick_arms(user: SousConfig, candidates: list[Candidate], checkpoints: dict[str, Checkpoint],
               *, working_set_bytes: int) -> tuple[list[Arm], list[Refusal]]
```

Rules: the user's current model is always a candidate (tier `"current"` when not in the table). For each candidate model that fits: one no-drafter arm, and for each compatible drafter one arm per block size in `BLOCK_SIZES`. The fit runs with the drafter that costs the most memory (all arms of a model share the window it yields) and with `window = max(user.max_context_tokens, gateway window if enabled)`, `floor = GATEWAY_MIN_CONTEXT_TOKENS` when the gateway is on else `user.context_min_tokens`. A lowered window lowers `max_context_tokens` and, when the gateway is on, `gateway_max_context_tokens` in the arm's config. `current` marks the arm equal to the user's model, drafter and block size. A model whose checkpoint is unknown (offline) is refused with the fit detail.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_arms.py`:

```python
from sous.config import SousConfig
from sous.tune.arms import BLOCK_SIZES, quick_arms
from sous.tune.candidates import Candidate, describe
from tests import tune_fixtures as fx


def _checkpoints():
    return {
        "mlx-community/Qwen3.8-27B-4bit": describe(
            "mlx-community/Qwen3.8-27B-4bit", config_fn=lambda m: fx.qwen_27b(),
            size_fn=lambda m: 16_100_000_000),
        "z-lab/Qwen3.8-27B-DFlash2": describe(
            "z-lab/Qwen3.8-27B-DFlash2", config_fn=lambda m: fx.dflash2_27b(),
            size_fn=lambda m: 3_850_000_000),
        "mlx-community/Qwen3.5-9B-MLX-4bit": describe(
            "mlx-community/Qwen3.5-9B-MLX-4bit", config_fn=lambda m: fx.qwen_9b(),
            size_fn=lambda m: 6_000_000_000),
        "z-lab/Qwen3.5-9B-DFlash": describe(
            "z-lab/Qwen3.5-9B-DFlash", config_fn=lambda m: fx.dflash_9b(),
            size_fn=lambda m: 2_600_000_000),
    }


def _candidates():
    return [
        Candidate("mlx-community/Qwen3.8-27B-4bit", "27b-dense", ("z-lab/Qwen3.8-27B-DFlash2",)),
        Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ("z-lab/Qwen3.5-9B-DFlash",)),
    ]


def test_on_the_m5_pro_every_model_gets_a_no_drafter_arm_and_one_per_block_size(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms, refusals = quick_arms(user, _candidates(), _checkpoints(),
                                working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert refusals == []
    by_model = {}
    for a in arms:
        by_model.setdefault(a.model_id, []).append(a)
    assert len(by_model["mlx-community/Qwen3.8-27B-4bit"]) == 1 + len(BLOCK_SIZES)
    assert len(by_model["mlx-community/Qwen3.5-9B-MLX-4bit"]) == 1 + len(BLOCK_SIZES)
    current = [a for a in arms if a.current]
    assert len(current) == 1
    assert (current[0].drafter_id, current[0].block_size) == ("z-lab/Qwen3.8-27B-DFlash2", 3)
    assert current[0].config.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    assert current[0].config.speculative_block_size == 3
    assert current[0].window == user.max_context_tokens
    plain = [a for a in by_model["mlx-community/Qwen3.8-27B-4bit"] if not a.drafter_id][0]
    assert plain.block_size == 0 and plain.config.speculative_draft_id == ""
    assert plain.label == "Qwen3.8-27B-4bit"
    assert current[0].label == "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3"


def test_on_the_m2_air_the_27b_is_refused_and_the_9b_runs_at_a_lowered_window(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml",
                      max_context_tokens=131072)
    arms, refusals = quick_arms(user, _candidates(), _checkpoints(),
                                working_set_bytes=fx.M2_AIR_WORKING_SET)
    assert [r.model_id for r in refusals] == ["mlx-community/Qwen3.8-27B-4bit"]
    assert "GiB" in refusals[0].reason
    assert {a.model_id for a in arms} == {"mlx-community/Qwen3.5-9B-MLX-4bit"}
    assert all(a.window < 131072 and a.window % 256 == 0 for a in arms)
    assert all(a.config.max_context_tokens == a.window for a in arms)
    assert not any(a.current for a in arms)


def test_the_gateway_window_is_lowered_too_and_floors_at_claude_codes_minimum(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml",
                      gateway_enabled=True, gateway_max_context_tokens=262144)
    arms, _ = quick_arms(user, _candidates(), _checkpoints(),
                         working_set_bytes=fx.M2_AIR_WORKING_SET)
    nine = [a for a in arms if a.model_id == "mlx-community/Qwen3.5-9B-MLX-4bit"]
    assert nine and all(a.gateway_window is not None and a.gateway_window >= 48 * 1024
                        for a in nine)
    assert all(a.config.gateway_max_context_tokens == a.gateway_window for a in nine)


def test_a_current_model_outside_the_table_is_still_an_arm(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml",
                      model_id="mlx-community/Qwen3.5-9B-MLX-4bit", speculative_draft_id="")
    arms, _ = quick_arms(user, _candidates()[:1], _checkpoints(),
                         working_set_bytes=fx.M5_PRO_WORKING_SET)
    nine = [a for a in arms if a.model_id == "mlx-community/Qwen3.5-9B-MLX-4bit"]
    assert nine and nine[0].tier == "current"
    assert [a for a in nine if a.current][0].drafter_id == ""


def test_an_incompatible_drafter_is_dropped_before_any_arm_exists(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    cps = _checkpoints()
    cands = [Candidate("mlx-community/Qwen3.8-27B-4bit", "27b-dense", ("z-lab/Qwen3.5-9B-DFlash",))]
    arms, refusals = quick_arms(user, cands, cps, working_set_bytes=fx.M5_PRO_WORKING_SET)
    assert [a.drafter_id for a in arms if a.model_id == "mlx-community/Qwen3.8-27B-4bit"] == [""]
    assert any("hidden size" in r.reason for r in refusals)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_arms.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.arms'`.

- [ ] **Step 3: Implement**

`src/sous/tune/arms.py`:

```python
"""One arm is one configuration the bench measures: a model, a drafter or
none, a block size, and the windows it fits at."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from sous.config import GATEWAY_MIN_CONTEXT_TOKENS, SousConfig
from sous.tune.candidates import Candidate, Checkpoint, Fit, drafter_compatible, fit

# The verify depths worth measuring: 2 and 3 pay on this family's GQA ratio,
# 5 is the most rows mlx's fused attention kernel takes.
BLOCK_SIZES = (2, 3, 5)


@dataclass(frozen=True)
class Arm:
    label: str
    config: SousConfig
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    gateway_window: int | None
    tier: str
    current: bool
    unvetted: bool = False


@dataclass(frozen=True)
class Refusal:
    model_id: str
    drafter_id: str
    reason: str


def _short(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def label_for(model_id: str, drafter_id: str, block_size: int) -> str:
    if not drafter_id:
        return _short(model_id)
    return f"{_short(model_id)} + {_short(drafter_id)} @{block_size}"


def _with_current(user: SousConfig, candidates: list[Candidate]) -> list[Candidate]:
    if any(c.id == user.model_id for c in candidates):
        return list(candidates)
    drafters = (user.speculative_draft_id,) if user.speculative_draft_id else ()
    return [Candidate(user.model_id, "current", drafters, "the configured model"), *candidates]


def _arm(user: SousConfig, cand: Candidate, drafter_id: str, block: int, f: Fit,
         gateway_window: int | None) -> Arm:
    config = dataclasses.replace(
        user,
        model_id=cand.id,
        speculative_draft_id=drafter_id,
        speculative_block_size=block if drafter_id else user.speculative_block_size,
        max_context_tokens=min(user.max_context_tokens, f.window),
        gateway_max_context_tokens=(
            gateway_window if gateway_window is not None else user.gateway_max_context_tokens
        ),
    )
    current = (
        cand.id == user.model_id
        and drafter_id == user.speculative_draft_id
        and (not drafter_id or block == user.speculative_block_size)
    )
    return Arm(
        label=label_for(cand.id, drafter_id, block),
        config=config,
        model_id=cand.id,
        drafter_id=drafter_id,
        block_size=block if drafter_id else 0,
        window=min(user.max_context_tokens, f.window),
        gateway_window=gateway_window,
        tier=cand.tier,
        current=current,
    )


def quick_arms(
    user: SousConfig,
    candidates: list[Candidate],
    checkpoints: dict[str, Checkpoint],
    *,
    working_set_bytes: int,
) -> tuple[list[Arm], list[Refusal]]:
    """Every quality-neutral arm of every candidate that fits, the user's own
    configuration among them. All arms of a model share one window: the one
    the fit yields with the model's heaviest drafter, so a block size is never
    measured at a window another block size could not run at."""
    window = max(user.max_context_tokens, user.gateway_max_context_tokens if user.gateway_enabled else 0)
    floor = GATEWAY_MIN_CONTEXT_TOKENS if user.gateway_enabled else user.context_min_tokens
    arms: list[Arm] = []
    refusals: list[Refusal] = []
    for cand in _with_current(user, candidates):
        target = checkpoints.get(cand.id)
        if target is None:
            refusals.append(Refusal(cand.id, "", "checkpoint unknown (offline?)"))
            continue
        drafters: list[Checkpoint] = []
        for drafter_id in cand.drafters:
            drafter = checkpoints.get(drafter_id)
            if drafter is None:
                refusals.append(Refusal(cand.id, drafter_id, "drafter unknown (offline?)"))
                continue
            ok, why = drafter_compatible(target, drafter)
            if ok:
                drafters.append(drafter)
            else:
                refusals.append(Refusal(cand.id, drafter_id, why))
        heaviest = max(drafters, key=lambda d: d.bytes or 0, default=None)
        f = fit(target, heaviest, window=window, floor=floor, working_set_bytes=working_set_bytes)
        if not f.fits:
            refusals.append(Refusal(cand.id, "", f.detail))
            continue
        gateway_window = None
        if user.gateway_enabled:
            gateway_window = min(user.gateway_max_context_tokens, f.window)
        arms.append(_arm(user, cand, "", 0, f, gateway_window))
        for drafter in drafters:
            for block in BLOCK_SIZES:
                arms.append(_arm(user, cand, drafter.id, block, f, gateway_window))
    return arms, refusals
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_arms.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/arms.py tests/test_tune_arms.py
git commit -m "feat(tune): enumerate the quality-neutral arms a quick run measures" -m "Drafter, block size and window never change what the model says, so they are the only dimensions a quick run varies; every model's arms share the window its heaviest drafter fits at." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 7: The run directory and the throughput bench

**Files:**
- Create: `src/sous/tune/rundir.py`
- Create: `src/sous/tune/bench.py`
- Test: `tests/test_tune_rundir.py`, `tests/test_tune_bench.py`

**Interfaces:**
- Consumes: `Arm` (Task 6); `EngineManager(config, engine_factory=...)`, `EngineManager.get()`, `EngineManager.unload_now()` (Task 1), `ManagedEngine.session()`, `ManagedEngine.count_tokens(messages, tools)`, `ManagedEngine.prompt_cache_stats(owner=thread)`, `GenerationSession.generate(messages, tools, max_tokens, timeout, on_delta)`, `GenerationSession.thread`, `GenerationSession.close()`, `Delta.output_tokens`, `release_mlx_thread_state()`, `sous.protocol.WORKER_TOOLS`.
- Produces:

```python
class RunDir:
    path: Path; run_id: str
    @classmethod new(cls, base: Path, *, clock=time.time) -> RunDir     # base/<YYYYmmdd-HHMMSS>
    @classmethod existing(cls, base: Path, run_id: str) -> RunDir       # FileNotFoundError if absent
    append(kind: str, row: dict) -> None                                 # one JSON line in results.jsonl
    rows(kind: str) -> list[dict]
    write_json(name: str, obj) -> Path
    write_text(name: str, text: str) -> Path

@dataclass(frozen=True)
class BenchRow:
    label: str; model_id: str; drafter_id: str; block_size: int; window: int
    ok: bool; error: str | None
    load_seconds: float | None
    prefill_tps_2k: float | None; prefill_tps_16k: float | None
    decode_tps_1k: float | None; decode_tps_16k: float | None
    ttft_seconds: float | None
    peak_memory_bytes: int | None
    spread: float | None            # (best − worst) / best of decode_tps_1k over the repeats
    def as_dict(self) -> dict
    @classmethod from_dict(cls, d: dict) -> BenchRow

LONG_CONTEXT = 16384
SHORT_CONTEXT = 1024
PREFILL_CONTEXT = 2048
DECODE_TOKENS = 256

def build_prompt(count, target_tokens: int, *, seed: int = 0) -> list[dict]
def bench_arm(arm: Arm, *, repeat: int = 2, factory=None, timeout: float = 600.0,
              peak_memory=None, reset_peak=None, active_memory=None, out=print) -> BenchRow
```

`bench_arm` runs everything on a thread of its own that calls `release_mlx_thread_state()` on exit, loads through `EngineManager`, and ends with `unload_now()` plus a wait (up to 30 s) for `active_memory()` to fall back within 1 GiB of what it was before the load. Any exception becomes `ok=False` with `error=f"{type(e).__name__}: {e}"`; the arm is never retried.

- [ ] **Step 1: Write the failing rundir tests**

`tests/test_tune_rundir.py`:

```python
import json

import pytest

from sous.tune.rundir import RunDir


def test_new_run_dir_is_named_by_the_clock_and_created(tmp_path):
    rd = RunDir.new(tmp_path / "tune", clock=lambda: 1_800_000_000.0)
    assert rd.path.is_dir() and rd.path.parent == tmp_path / "tune"
    assert rd.run_id == rd.path.name and len(rd.run_id) == 15   # YYYYmmdd-HHMMSS


def test_rows_round_trip_by_kind_and_survive_a_reopen(tmp_path):
    rd = RunDir.new(tmp_path / "tune")
    rd.append("bench", {"label": "a", "decode_tps_1k": 25.5})
    rd.append("bench", {"label": "b", "decode_tps_1k": None})
    rd.append("note", {"text": "x"})
    again = RunDir.existing(tmp_path / "tune", rd.run_id)
    assert again.rows("bench") == [{"label": "a", "decode_tps_1k": 25.5},
                                   {"label": "b", "decode_tps_1k": None}]
    assert again.rows("note") == [{"text": "x"}]
    lines = (rd.path / "results.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "bench"


def test_existing_refuses_an_unknown_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunDir.existing(tmp_path / "tune", "nope")


def test_json_and_text_files_land_in_the_run_dir(tmp_path):
    rd = RunDir.new(tmp_path / "tune")
    p = rd.write_json("hardware.json", {"chip": "M2"})
    assert json.loads(p.read_text()) == {"chip": "M2"}
    assert rd.write_text("report.md", "# hi\n").read_text() == "# hi\n"
```

- [ ] **Step 2: Implement `rundir.py`**

```python
"""Where a tune run keeps what it measured: one directory per run under the
data dir, results appended as they land so an interrupted run can resume."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        self.run_id = path.name

    @classmethod
    def new(cls, base: Path, *, clock: Callable[[], float] = time.time) -> RunDir:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
        path = base / run_id
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    @classmethod
    def existing(cls, base: Path, run_id: str) -> RunDir:
        path = base / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"no tune run {run_id!r} under {base}")
        return cls(path)

    @property
    def results(self) -> Path:
        return self.path / "results.jsonl"

    def append(self, kind: str, row: dict) -> None:
        # One line per result, flushed at once: a run killed mid-arm keeps
        # every arm that finished, and --resume reads exactly those.
        with self.results.open("a") as f:
            f.write(json.dumps({"kind": kind, **row}, allow_nan=False) + "\n")

    def rows(self, kind: str) -> list[dict]:
        if not self.results.exists():
            return []
        out: list[dict] = []
        for line in self.results.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.pop("kind", None) == kind:
                out.append(row)
        return out

    def write_json(self, name: str, obj: object) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.write_text(text)
        return p
```

Run: `uv run pytest tests/test_tune_rundir.py -v` → 4 passed.

- [ ] **Step 3: Write the failing bench tests**

`tests/test_tune_bench.py`:

```python
import dataclasses

from sous.config import SousConfig
from sous.protocol import WORKER_TOOLS
from sous.tune.arms import Arm
from sous.tune.bench import (
    DECODE_TOKENS,
    LONG_CONTEXT,
    PREFILL_CONTEXT,
    BenchRow,
    bench_arm,
    build_prompt,
)
from tests.fake_engine import FakeEngine


def _count(messages, tools):
    return sum(len(str(m)) for m in messages) // 4


def test_build_prompt_reaches_the_target_deterministically():
    a = build_prompt(_count, 2048)
    b = build_prompt(_count, 2048)
    assert a == b
    assert _count(a, WORKER_TOOLS) >= 2048
    assert a[0]["role"] == "system" and a[1]["role"] == "user"
    assert "def " in a[1]["content"]


def _arm(tmp_path, window=131072):
    cfg = SousConfig(data_dir=tmp_path / "d", config_path=tmp_path / "c.toml",
                     max_context_tokens=window, speculative_draft_id="z/draft",
                     speculative_block_size=3)
    return Arm(label="m + draft @3", config=cfg, model_id="org/m", drafter_id="z/draft",
               block_size=3, window=window, gateway_window=None, tier="t", current=True)


def _fake(script_len=12):
    engines = []

    def factory(model_id):
        e = FakeEngine(["w " * DECODE_TOKENS] * script_len)
        e.stats = {"prefill_seconds": 2.0, "prefilled_tokens": PREFILL_CONTEXT,
                   "decode_seconds": 4.0}
        engines.append(e)
        return e

    return factory, engines


def test_bench_arm_reads_throughput_from_the_engines_gauges(tmp_path):
    factory, engines = _fake()
    row = bench_arm(
        _arm(tmp_path), factory=factory, repeat=2,
        peak_memory=lambda: 4096, reset_peak=lambda: None, active_memory=lambda: 100,
        out=lambda *a, **k: None,
    )
    assert row.ok and row.error is None
    assert row.model_id == "org/m" and row.drafter_id == "z/draft" and row.block_size == 3
    assert row.prefill_tps_2k == PREFILL_CONTEXT / 2.0
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0
    assert row.decode_tps_16k == DECODE_TOKENS / 4.0
    assert row.prefill_tps_16k == PREFILL_CONTEXT / 2.0   # the fake's gauge, whatever the prompt
    assert row.ttft_seconds is not None and row.ttft_seconds >= 0
    assert row.peak_memory_bytes == 4096 and row.spread == 0.0
    assert row.load_seconds is not None
    assert engines[0].unloaded is True
    # Every turn ran on the one session thread the prompt cache is scoped to.
    assert len({t for t in engines[0].generate_threads}) == 1


def test_a_small_window_skips_the_long_context_measurements(tmp_path):
    factory, engines = _fake()
    row = bench_arm(_arm(tmp_path, window=8192), factory=factory, repeat=1,
                    peak_memory=lambda: 1, reset_peak=lambda: None, active_memory=lambda: 0,
                    out=lambda *a, **k: None)
    assert row.ok and row.decode_tps_16k is None and row.prefill_tps_16k is None
    assert row.decode_tps_1k == DECODE_TOKENS / 4.0
    assert LONG_CONTEXT > 8192


def test_a_failing_load_is_a_row_not_an_exception(tmp_path):
    def factory(model_id):
        raise RuntimeError("no weights")

    row = bench_arm(_arm(tmp_path), factory=factory, peak_memory=lambda: 0,
                    reset_peak=lambda: None, active_memory=lambda: 0, out=lambda *a, **k: None)
    assert row.ok is False and row.error == "RuntimeError: no weights"
    assert row.decode_tps_1k is None


def test_rows_round_trip_through_dicts():
    row = BenchRow(label="a", model_id="m", drafter_id="", block_size=0, window=1, ok=True,
                   error=None, load_seconds=1.0, prefill_tps_2k=2.0, prefill_tps_16k=None,
                   decode_tps_1k=3.0, decode_tps_16k=None, ttft_seconds=0.5,
                   peak_memory_bytes=7, spread=0.1)
    assert BenchRow.from_dict(row.as_dict()) == row
    assert dataclasses.asdict(row)["label"] == "a"
```

- [ ] **Step 4: Run the bench tests to verify they fail**

Run: `uv run pytest tests/test_tune_bench.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.bench'`.

- [ ] **Step 5: Implement `bench.py`**

```python
"""Throughput of one arm through sous's own engine: prefill and decode at a
short and a long context, warm-turn TTFT, load time and peak memory, read
from the prompt cache's per-turn gauges the way the gateway's turn line is."""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sous.engine.base import Delta, EngineManager, release_mlx_thread_state
from sous.protocol import WORKER_TOOLS
from sous.tune.arms import Arm

LONG_CONTEXT = 16384
SHORT_CONTEXT = 1024
PREFILL_CONTEXT = 2048
DECODE_TOKENS = 256
# A long-context measurement needs the prompt plus its answers to fit.
_LONG_HEADROOM = 1024
_UNLOAD_WAIT_SECONDS = 30.0
_UNLOAD_SLACK_BYTES = 1 << 30

_SYSTEM = (
    "You are a coding assistant. Answer with code only, no prose, continuing the module "
    "in the same style."
)


@dataclass(frozen=True)
class BenchRow:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    ok: bool
    error: str | None
    load_seconds: float | None
    prefill_tps_2k: float | None
    prefill_tps_16k: float | None
    decode_tps_1k: float | None
    decode_tps_16k: float | None
    ttft_seconds: float | None
    peak_memory_bytes: int | None
    spread: float | None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchRow:
        return cls(**{f.name: d.get(f.name) for f in dataclasses.fields(cls)})


def _code_lines(rng: random.Random, n: int, start: int) -> list[str]:
    lines: list[str] = []
    for i in range(start, start + n):
        a, b = rng.randint(2, 97), rng.randint(2, 97)
        lines.append(f"def fn_{i}(x: int, y: int = {b}) -> int:")
        lines.append(f'    """Combine x with {a}; y offsets the result."""')
        lines.append(f"    total = x * {a} + y")
        lines.append(f"    return total % {a + b}")
        lines.append("")
    return lines


def build_prompt(
    count: Callable[[list[dict], list[dict]], int], target_tokens: int, *, seed: int = 0
) -> list[dict]:
    """A code-shaped conversation of at least `target_tokens` tokens by the
    engine's own count, grown geometrically so a 16K prompt costs a handful
    of tokenizations rather than hundreds. Deterministic per seed."""
    rng = random.Random(seed)
    lines: list[str] = []

    def render() -> list[dict]:
        body = "Review this module and continue it in the same style.\n\n" + "\n".join(lines)
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": body}]

    fn = 0
    while count(render(), WORKER_TOOLS) < target_tokens:
        n = max(16, len(lines) // 5)
        lines.extend(_code_lines(rng, n, fn))
        fn += n
    return render()


class _Turn:
    """One generate() through a session, with the gauges read on that
    session's thread right after — they are per-turn readings, never
    counter differences."""

    def __init__(self, engine, session, timeout: float):
        self.engine, self.session, self.timeout = engine, session, timeout

    def run(self, messages: list[dict], max_tokens: int) -> tuple[str, dict, int, float | None]:
        first: list[float] = []
        tokens: list[int] = [0]
        started = time.monotonic()

        def on_delta(d: Delta) -> None:
            if not first:
                first.append(time.monotonic() - started)
            tokens[0] = d.output_tokens

        text = self.session.generate(messages, WORKER_TOOLS, max_tokens, self.timeout, on_delta)
        gauges = self.engine.prompt_cache_stats(owner=self.session.thread)
        return text, gauges, tokens[0], (first[0] if first else None)


def _rate(tokens: float | None, seconds: float | None) -> float | None:
    if not tokens or not seconds or seconds <= 0:
        return None
    return tokens / seconds


def _measure(arm: Arm, repeat: int, factory, timeout: float, peak_memory, reset_peak,
             active_memory, out) -> BenchRow:
    base = dict(label=arm.label, model_id=arm.model_id, drafter_id=arm.drafter_id,
                block_size=arm.block_size, window=arm.window)
    baseline = active_memory()
    reset_peak()
    manager = EngineManager(arm.config, engine_factory=factory)
    loading = time.monotonic()
    engine = manager.get()
    load_seconds = time.monotonic() - loading
    session = engine.session()
    turn = _Turn(engine, session, timeout)
    count = engine.count_tokens
    try:
        decode_1k: list[float] = []
        ttfts: list[float] = []
        for _ in range(max(1, repeat)):
            short = build_prompt(count, SHORT_CONTEXT)
            text, gauges, produced, _ = turn.run(short, DECODE_TOKENS)
            rate = _rate(produced, gauges.get("decode_seconds"))
            if rate is not None:
                decode_1k.append(rate)
            warm = [*short, {"role": "assistant", "content": text},
                    {"role": "user", "content": "Continue with the next function."}]
            _, _, _, ttft = turn.run(warm, 8)
            if ttft is not None:
                ttfts.append(ttft)
        _, gauges, _, _ = turn.run(build_prompt(count, PREFILL_CONTEXT, seed=1), 1)
        prefill_2k = _rate(gauges.get("prefilled_tokens"), gauges.get("prefill_seconds"))
        prefill_16k = decode_16k = None
        if arm.window >= LONG_CONTEXT + _LONG_HEADROOM:
            long = build_prompt(count, LONG_CONTEXT, seed=2)
            text, gauges, _, _ = turn.run(long, 1)
            prefill_16k = _rate(gauges.get("prefilled_tokens"), gauges.get("prefill_seconds"))
            warm = [*long, {"role": "assistant", "content": text},
                    {"role": "user", "content": "Continue with the next function."}]
            _, gauges, produced, _ = turn.run(warm, DECODE_TOKENS)
            decode_16k = _rate(produced, gauges.get("decode_seconds"))
        spread = None
        if decode_1k:
            spread = (max(decode_1k) - min(decode_1k)) / max(decode_1k)
        return BenchRow(
            **base, ok=True, error=None, load_seconds=load_seconds,
            prefill_tps_2k=prefill_2k, prefill_tps_16k=prefill_16k,
            decode_tps_1k=max(decode_1k) if decode_1k else None, decode_tps_16k=decode_16k,
            ttft_seconds=min(ttfts) if ttfts else None,
            peak_memory_bytes=int(peak_memory()), spread=spread,
        )
    finally:
        session.close()
        session.join(timeout)
        manager.unload_now()
        deadline = time.monotonic() + _UNLOAD_WAIT_SECONDS
        while active_memory() > baseline + _UNLOAD_SLACK_BYTES and time.monotonic() < deadline:
            time.sleep(0.2)
        out(f"  {arm.label}: done")


def _peak_memory() -> int:
    import mlx.core as mx

    return int(mx.get_peak_memory())


def _reset_peak() -> None:
    import mlx.core as mx

    mx.reset_peak_memory()


def _active_memory() -> int:
    import mlx.core as mx

    mx.clear_cache()
    return int(mx.get_active_memory())


def bench_arm(
    arm: Arm,
    *,
    repeat: int = 2,
    factory: Callable[[str], object] | None = None,
    timeout: float = 600.0,
    peak_memory: Callable[[], int] | None = None,
    reset_peak: Callable[[], None] | None = None,
    active_memory: Callable[[], int] | None = None,
    out: Callable[..., None] = print,
) -> BenchRow:
    """The arm on a thread of its own, which releases its mlx state on the
    way out; a failure is a row that says so, and the next arm still runs."""
    outcome: list[BenchRow | BaseException] = []

    def run() -> None:
        try:
            outcome.append(_measure(
                arm, repeat, factory, timeout,
                peak_memory or _peak_memory, reset_peak or _reset_peak,
                active_memory or _active_memory, out,
            ))
        except BaseException as e:  # noqa: BLE001 — becomes the row's error
            outcome.append(e)
        finally:
            release_mlx_thread_state()

    worker = threading.Thread(target=run, name=f"sous-tune-{arm.label}", daemon=True)
    worker.start()
    worker.join()
    result = outcome[0]
    if isinstance(result, BaseException):
        return BenchRow(
            label=arm.label, model_id=arm.model_id, drafter_id=arm.drafter_id,
            block_size=arm.block_size, window=arm.window, ok=False,
            error=f"{type(result).__name__}: {result}", load_seconds=None,
            prefill_tps_2k=None, prefill_tps_16k=None, decode_tps_1k=None,
            decode_tps_16k=None, ttft_seconds=None, peak_memory_bytes=None, spread=None,
        )
    return result
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_bench.py tests/test_tune_rundir.py -v`
Expected: all pass. If `test_bench_arm_reads_throughput_from_the_engines_gauges` fails on `spread`, the fake's decode rate is identical every repeat, so the spread is exactly `0.0`; if it fails on the script running out, count the turns: `repeat × 2 + 1 + 2 = 7` for `repeat=2`, which the 12-reply script covers.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/rundir.py src/sous/tune/bench.py tests/test_tune_rundir.py tests/test_tune_bench.py
git commit -m "feat(tune): measure an arm's throughput through the real engine" -m "Prefill, decode, warm-turn TTFT, load time and peak memory come from the engine's own per-turn gauges on one session thread, so the numbers are what a task or a gateway turn would see; every arm loads, measures and unloads alone." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 8: The quick-mode decision rule

**Files:**
- Create: `src/sous/tune/decide.py`
- Test: `tests/test_tune_decide.py`

**Interfaces:**
- Consumes: `Arm` (Task 6), `BenchRow` (Task 7), `SousConfig`.
- Produces:

```python
@dataclass(frozen=True)
class QuickChoice:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    gateway_window: int | None
    changes: dict[str, dict[str, object]]   # {"model": {...}, "gateway": {...}}, only keys that differ
    reasons: list[str]                      # one line per fact behind the choice

def score(row: BenchRow) -> float | None          # decode_tps_16k when measured, else decode_tps_1k
def best_per_model(rows: list[BenchRow]) -> dict[str, BenchRow]
def quick_decision(user: SousConfig, arms: list[Arm], rows: list[BenchRow]) -> QuickChoice | None
```

The quick rule never changes the model: it ranks the current model's successful arms by `score` (ties keep the current arm), and the changes are the keys of the winning arm that differ from the user's config — `speculative_draft_id`, `speculative_block_size` (drafter arms only), `max_context_tokens` when the arm's window is lower, `gateway.max_context_tokens` when the gateway is on and the arm's gateway window is lower. `None` when the current model has no successful row (it did not fit, or every arm failed); the caller prints the other models' rows with a "quality untested" label either way.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_decide.py`:

```python
from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.decide import best_per_model, quick_decision, score

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _row(label, model=M, drafter=D, block=3, d1k=25.0, d16k=18.0, ok=True):
    return BenchRow(label=label, model_id=model, drafter_id=drafter, block_size=block,
                    window=131072, ok=ok, error=None if ok else "x", load_seconds=5.0,
                    prefill_tps_2k=480.0, prefill_tps_16k=400.0, decode_tps_1k=d1k,
                    decode_tps_16k=d16k, ttft_seconds=1.0, peak_memory_bytes=1, spread=0.02)


def _arm(user, label, drafter=D, block=3, window=131072, gateway_window=None, current=False,
         model=M):
    return Arm(label=label, config=user, model_id=model, drafter_id=drafter, block_size=block,
               window=window, gateway_window=gateway_window, tier="27b-dense", current=current)


def test_score_prefers_the_long_context_measurement():
    assert score(_row("a", d16k=18.0, d1k=25.0)) == 18.0
    assert score(_row("a", d16k=None, d1k=25.0)) == 25.0
    assert score(_row("a", ok=False)) is None


def test_best_per_model_picks_the_top_score_of_each_model():
    rows = [_row("a", d16k=18.0), _row("b", block=2, d16k=19.5),
            _row("c", model="o/9b", drafter="", block=0, d16k=40.0)]
    best = best_per_model(rows)
    assert best[M].label == "b" and best["o/9b"].label == "c"


def test_a_faster_block_size_becomes_the_only_change(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d16k=18.0), _row("b2", block=2, d16k=19.5)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "b2"
    assert choice.changes == {"model": {"speculative_block_size": 2}}
    assert any("19.5" in r and "18.0" in r for r in choice.reasons)


def test_a_tie_keeps_the_current_arm_and_yields_no_changes(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "b2", block=2)]
    rows = [_row("cur", d16k=18.0), _row("b2", block=2, d16k=18.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None and choice.label == "cur" and choice.changes == {}


def test_no_drafter_winning_clears_the_drafter_key(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "plain", drafter="", block=0)]
    rows = [_row("cur", d16k=10.0), _row("plain", drafter="", block=0, d16k=12.0)]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    assert choice.changes == {"model": {"speculative_draft_id": ""}}


def test_a_lowered_window_is_written_for_the_worker_and_the_gateway(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml",
                      max_context_tokens=131072, gateway_enabled=True,
                      gateway_max_context_tokens=262144)
    arms = [_arm(user, "cur", current=True, window=65536, gateway_window=65536)]
    rows = [_row("cur")]
    choice = quick_decision(user, arms, rows)
    assert choice is not None
    assert choice.changes == {"model": {"max_context_tokens": 65536},
                              "gateway": {"max_context_tokens": 65536}}


def test_other_models_never_win_and_a_missing_current_model_is_none(tmp_path):
    user = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    arms = [_arm(user, "cur", current=True), _arm(user, "moe", model="o/moe", drafter="", block=0)]
    rows = [_row("cur", d16k=18.0), _row("moe", model="o/moe", drafter="", block=0, d16k=60.0)]
    assert quick_decision(user, arms, rows).model_id == M
    assert quick_decision(user, arms, [rows[1]]) is None
    assert quick_decision(user, arms, [_row("cur", ok=False)]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_decide.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.decide'`.

- [ ] **Step 3: Implement**

`src/sous/tune/decide.py`:

```python
"""The quick rule: among the current model's quality-neutral arms, the
fastest decode wins, and only the keys that differ from the user's config
become changes. The model itself is never this rule's to change."""

from __future__ import annotations

from dataclasses import dataclass, field

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow


@dataclass(frozen=True)
class QuickChoice:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    gateway_window: int | None
    changes: dict[str, dict[str, object]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def score(row: BenchRow) -> float | None:
    """Decode at the long context when it was measured — the gateway's regime
    and where the drafter's gain is smallest — else at the short one."""
    if not row.ok:
        return None
    return row.decode_tps_16k if row.decode_tps_16k is not None else row.decode_tps_1k


def best_per_model(rows: list[BenchRow]) -> dict[str, BenchRow]:
    best: dict[str, BenchRow] = {}
    for row in rows:
        s = score(row)
        if s is None:
            continue
        held = best.get(row.model_id)
        if held is None or s > (score(held) or 0.0):
            best[row.model_id] = row
    return best


def _changes(user: SousConfig, arm: Arm) -> dict[str, dict[str, object]]:
    model: dict[str, object] = {}
    if arm.drafter_id != user.speculative_draft_id:
        model["speculative_draft_id"] = arm.drafter_id
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
    if arm.window < user.max_context_tokens:
        model["max_context_tokens"] = arm.window
    changes: dict[str, dict[str, object]] = {}
    if model:
        changes["model"] = model
    if (
        user.gateway_enabled
        and arm.gateway_window is not None
        and arm.gateway_window < user.gateway_max_context_tokens
    ):
        changes["gateway"] = {"max_context_tokens": arm.gateway_window}
    return changes


def quick_decision(user: SousConfig, arms: list[Arm], rows: list[BenchRow]) -> QuickChoice | None:
    by_label = {a.label: a for a in arms}
    mine = [r for r in rows if r.model_id == user.model_id and score(r) is not None]
    if not mine:
        return None
    current = next((r for r in mine if by_label.get(r.label) and by_label[r.label].current), None)
    # A strict improvement over the current arm is required: a tie changes
    # nothing, and a re-run on a noisy afternoon must not flip a setting.
    winner = current
    for row in mine:
        if winner is None or (score(row) or 0.0) > (score(winner) or 0.0):
            winner = row
    assert winner is not None
    arm = by_label[winner.label]
    which = "decode at 16K context" if winner.decode_tps_16k is not None else "decode at 1K context"
    reasons = [f"{which}: {score(winner):.1f} tok/s for {winner.label}"]
    if current is not None and current.label != winner.label:
        reasons.append(f"{which}: {score(current):.1f} tok/s for the current arm ({current.label})")
    if current is not None and current.label == winner.label:
        reasons.append("the current arm is already the fastest measured")
    return QuickChoice(
        label=arm.label, model_id=arm.model_id, drafter_id=arm.drafter_id,
        block_size=arm.block_size, window=arm.window, gateway_window=arm.gateway_window,
        changes=_changes(user, arm), reasons=reasons,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_decide.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/decide.py tests/test_tune_decide.py
git commit -m "feat(tune): pick the fastest quality-neutral arm of the current model" -m "A quick run may change the drafter, the block size and a window that fits, nothing else, and only on a strict improvement, so a noisy re-run cannot flip a setting." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Report, diff, apply, restart note

**Files:**
- Create: `src/sous/tune/report.py`
- Test: `tests/test_tune_report.py`

**Interfaces:**
- Consumes: `Hardware`, `gib` (Task 3); `Table`, `Checkpoint`, `Fit` (Task 4); `Refusal` (Task 6); `BenchRow` (Task 7); `QuickChoice` (Task 8); `tomlkit`.
- Produces:

```python
def render_report(*, hardware: Hardware, table_age_days: int, checkpoints: dict[str, Checkpoint],
                  refusals: list[Refusal], rows: list[BenchRow], choice: QuickChoice | None,
                  current_model: str) -> str
def config_diff(config_path: Path, changes: dict[str, dict[str, object]]) -> tuple[str, str]
def apply_changes(config_path: Path, new_text: str, run_id: str) -> Path | None
def restart_note(changes: dict[str, dict[str, object]], *, managed: bool, label: str) -> str | None
```

`render_report` has four sections in this order: `Hardware`, `Candidates` (one line per checkpoint with bytes, KV per token, verifier and int8 layout, and the refusal reason when refused), `Throughput` (one line per row: label, prefill 2K/16K, decode 1K/16K, TTFT, peak memory, spread, or the error), `Choice` (the winner and its reasons, or the guidance line when `choice is None`), and every model other than the current one is marked `quality untested`. Past 90 days the header says the table is stale and suggests `--discover`.

`config_diff` parses the file with tomlkit (an absent file starts from an empty document), sets each `changes[section][key]`, and returns the new text and a unified diff. `apply_changes` copies the old file to `config.toml.bak-tune-<run_id>` (None when there was no file), writes the new text, and returns the backup path. `restart_note` returns a line only when `changes` touch `model.id` or `model.int8_prefill`: `sous stop` then `sous serve` for an unmanaged daemon, `launchctl kickstart -k gui/<uid>/<label>` for a managed one.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune_report.py`:

```python
import os

from sous.config import load_config
from sous.engine.int8prefill import Availability
from sous.tune.arms import Refusal
from sous.tune.bench import BenchRow
from sous.tune.candidates import describe
from sous.tune.decide import QuickChoice
from sous.tune.hardware import detect
from sous.tune.report import apply_changes, config_diff, render_report, restart_note
from tests import tune_fixtures as fx

M = "mlx-community/Qwen3.8-27B-4bit"


def _hardware(tmp_path):
    return detect(
        device_info=lambda: {"device_name": "Apple M5 Pro", "memory_size": 64 * 2**30,
                             "max_recommended_working_set_size": fx.M5_PRO_WORKING_SET,
                             "architecture": "applegpu_g17s"},
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda p: (1, 2, 400 * 2**30),
        availability=lambda: Availability(True),
        version=lambda n: "1.0", hub_cache=str(tmp_path),
    )


def _row(label, model=M, ok=True):
    return BenchRow(label=label, model_id=model, drafter_id="d", block_size=3, window=131072,
                    ok=ok, error=None if ok else "RuntimeError: boom", load_seconds=5.2,
                    prefill_tps_2k=479.4, prefill_tps_16k=401.0, decode_tps_1k=25.6,
                    decode_tps_16k=18.3, ttft_seconds=1.1, peak_memory_bytes=20 * 2**30,
                    spread=0.03)


def test_the_report_has_every_section_and_labels_other_models_quality_untested(tmp_path):
    cps = {M: describe(M, config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 16_100_000_000),
           "o/9b": describe("o/9b", config_fn=lambda m: fx.qwen_9b(), size_fn=lambda m: 6 * 10**9)}
    choice = QuickChoice(label="cur", model_id=M, drafter_id="d", block_size=3, window=131072,
                         gateway_window=None, changes={}, reasons=["the current arm is already the fastest measured"])
    text = render_report(hardware=_hardware(tmp_path), table_age_days=3, checkpoints=cps,
                         refusals=[Refusal("o/big", "", "weights 30.0 GB ... below the floor")],
                         rows=[_row("cur"), _row("nine", model="o/9b"), _row("bad", ok=False)],
                         choice=choice, current_model=M)
    for heading in ("Hardware", "Candidates", "Throughput", "Choice"):
        assert heading in text
    assert "Apple M5 Pro" in text and "51.8 GiB" in text
    assert "16.1 GB" in text and "64 KiB/token" in text
    assert "o/big" in text and "below the floor" in text
    assert "479" in text and "18.3" in text and "RuntimeError: boom" in text
    assert "quality untested" in text.split("nine")[1].splitlines()[0]
    assert "already the fastest" in text
    assert "stale" not in text


def test_a_stale_table_and_a_missing_choice_are_said_plainly(tmp_path):
    text = render_report(hardware=_hardware(tmp_path), table_age_days=120, checkpoints={},
                         refusals=[], rows=[], choice=None, current_model=M)
    assert "stale" in text and "--discover" in text
    assert "cannot recommend" in text and M in text


def test_config_diff_preserves_comments_and_untouched_keys(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('# mine\n[model]\nid = "x/y"   # keep\nspeculative_block_size = 3\n\n[gateway]\nenabled = true\n')
    new, diff = config_diff(p, {"model": {"speculative_block_size": 2},
                                "gateway": {"max_context_tokens": 65536}})
    assert "# mine" in new and 'id = "x/y"   # keep' in new
    assert "speculative_block_size = 2" in new and "max_context_tokens = 65536" in new
    assert "-speculative_block_size = 3" in diff and "+speculative_block_size = 2" in diff
    assert "+max_context_tokens = 65536" in diff
    assert load_config(p).speculative_block_size == 3   # nothing written yet


def test_config_diff_starts_from_an_empty_document_when_the_file_is_absent(tmp_path):
    p = tmp_path / "config.toml"
    new, diff = config_diff(p, {"model": {"speculative_draft_id": ""}})
    assert 'speculative_draft_id = ""' in new and diff.startswith("---")


def test_apply_writes_a_backup_then_the_new_text(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nspeculative_block_size = 3\n")
    new, _ = config_diff(p, {"model": {"speculative_block_size": 2}})
    backup = apply_changes(p, new, "20260917-101500")
    assert backup == tmp_path / "config.toml.bak-tune-20260917-101500"
    assert backup.read_text() == "[model]\nspeculative_block_size = 3\n"
    assert load_config(p).speculative_block_size == 2
    assert apply_changes(tmp_path / "fresh.toml", new, "x") is None


def test_restart_note_only_for_keys_the_daemon_reads_at_load():
    assert restart_note({"model": {"speculative_block_size": 2}}, managed=False, label="l") is None
    unmanaged = restart_note({"model": {"id": "o/9b"}}, managed=False, label="com.sous.daemon")
    assert unmanaged and "sous stop" in unmanaged and "sous serve" in unmanaged
    managed = restart_note({"model": {"int8_prefill": True}}, managed=True, label="com.sous.daemon")
    assert managed and f"launchctl kickstart -k gui/{os.getuid()}/com.sous.daemon" in managed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_report.py -v`
Expected: `ModuleNotFoundError: No module named 'sous.tune.report'`.

- [ ] **Step 3: Implement**

`src/sous/tune/report.py`:

```python
"""What the user reads: the report, the config diff, and the apply."""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path

import tomlkit

from sous.tune.arms import Refusal
from sous.tune.bench import BenchRow
from sous.tune.candidates import Checkpoint
from sous.tune.decide import QuickChoice
from sous.tune.hardware import Hardware, gib

STALE_AFTER_DAYS = 90
_GB = 10**9
_KIB = 1024


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def _quant(cp: Checkpoint) -> str:
    q = cp.quant
    if q.mode is None:
        return "unquantized"
    layout = f"{q.mode} {q.bits}-bit gs{q.group_size}"
    if q.overrides:
        layout += f", {q.overrides} layer overrides"
    verifier = "fast verifier" if q.verifier_fast else "slow verifier"
    if q.verifier_slow_layers:
        verifier += f" ({q.verifier_slow_layers} slow layers)"
    int8 = "int8-routable" if q.int8_routable else "no int8 prefill"
    return f"{layout}; {verifier}; {int8}"


def _candidate_line(cp: Checkpoint, refusal: Refusal | None) -> str:
    size = "size unknown" if cp.bytes is None else f"{cp.bytes / _GB:.1f} GB"
    kv = "KV unknown" if cp.kv_bytes_per_token is None else f"{cp.kv_bytes_per_token // _KIB} KiB/token"
    line = f"  {cp.id:45s} {size:>12s}  {kv:>13s}  {_quant(cp)}"
    if refusal is not None:
        line += f"\n      refused: {refusal.reason}"
    return line


def _row_line(row: BenchRow, current_model: str) -> str:
    tag = "" if row.model_id == current_model else "  [quality untested]"
    if not row.ok:
        return f"  {row.label:45s} failed: {row.error}{tag}"
    peak = "-" if row.peak_memory_bytes is None else gib(row.peak_memory_bytes)
    return (
        f"  {row.label:45s} prefill {_fmt(row.prefill_tps_2k)}/{_fmt(row.prefill_tps_16k)} tok/s"
        f"  decode {_fmt(row.decode_tps_1k)}/{_fmt(row.decode_tps_16k)} tok/s"
        f"  ttft {_fmt(row.ttft_seconds, ' s')}  peak {peak}"
        f"  spread {_fmt(None if row.spread is None else row.spread * 100, '%', 0)}"
        f"  load {_fmt(row.load_seconds, ' s')}{tag}"
    )


def render_report(
    *,
    hardware: Hardware,
    table_age_days: int,
    checkpoints: dict[str, Checkpoint],
    refusals: list[Refusal],
    rows: list[BenchRow],
    choice: QuickChoice | None,
    current_model: str,
) -> str:
    refused = {r.model_id: r for r in refusals if not r.drafter_id}
    out = ["# sous tune --quick", "", "## Hardware", ""]
    out.append(
        f"  {hardware.chip}, {gib(hardware.memory_bytes)} unified memory, Metal working set "
        f"{gib(hardware.working_set_bytes)}, macOS {hardware.macos}"
    )
    nax = "NAX tensor units: yes" if hardware.nax else f"NAX tensor units: no ({hardware.nax_reason})"
    out.append(f"  {nax}; " + ", ".join(f"{k} {v}" for k, v in hardware.versions.items()))
    out.append(f"  hub cache {hardware.hub_cache}: {gib(hardware.disk_free_bytes)} free")
    if table_age_days > STALE_AFTER_DAYS:
        out.append(
            f"  the candidate table is {table_age_days} days old and may be stale; "
            "`sous tune --discover` can look for newer checkpoints"
        )
    out += ["", "## Candidates", "  (prefill/decode at 2K/16K and 1K/16K tokens of context)", ""]
    for cp in checkpoints.values():
        out.append(_candidate_line(cp, refused.get(cp.id)))
    for r in refusals:
        if r.drafter_id:
            out.append(f"  {r.drafter_id:45s} dropped for {r.model_id}: {r.reason}")
        elif r.model_id not in checkpoints:
            out.append(f"  {r.model_id:45s} refused: {r.reason}")
    out += ["", "## Throughput", ""]
    out += [_row_line(r, current_model) for r in rows] or ["  (nothing measured)"]
    out += ["", "## Choice", ""]
    if choice is None:
        out.append(
            f"  cannot recommend a setting: no successful measurement of the configured model "
            f"({current_model}). A quick run never changes the model; if it does not fit this "
            "machine, run the full `sous tune` (a later release) or set [model].id by hand to "
            "one of the candidates above."
        )
    else:
        out.append(f"  {choice.label}")
        out += [f"    {r}" for r in choice.reasons]
        if not choice.changes:
            out.append("    no config change")
    return "\n".join(out) + "\n"


def config_diff(config_path: Path, changes: dict[str, dict[str, object]]) -> tuple[str, str]:
    """The new file text and its unified diff against the current one. tomlkit
    keeps comments, ordering and spacing, so the diff is the changed keys and
    nothing else."""
    old = config_path.read_text() if config_path.is_file() else ""
    doc = tomlkit.parse(old) if old else tomlkit.document()
    for section, keys in changes.items():
        table = doc.setdefault(section, tomlkit.table())
        for key, value in keys.items():
            table[key] = value
    new = tomlkit.dumps(doc)
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(config_path),
            tofile=f"{config_path} (tuned)",
        )
    )
    return new, diff


def apply_changes(config_path: Path, new_text: str, run_id: str) -> Path | None:
    backup = None
    if config_path.is_file():
        backup = config_path.with_name(f"{config_path.name}.bak-tune-{run_id}")
        shutil.copyfile(config_path, backup)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(new_text)
    return backup


def restart_note(
    changes: dict[str, dict[str, object]], *, managed: bool, label: str
) -> str | None:
    """Keys the daemon reads once at load need a restart to take effect;
    everything a quick run writes is read per task or per turn."""
    model = changes.get("model", {})
    if "id" not in model and "int8_prefill" not in model:
        return None
    if managed:
        return f"restart the daemon to load it: launchctl kickstart -k gui/{os.getuid()}/{label}"
    return "restart the daemon to load it: sous stop, then sous serve"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune_report.py -v`
Expected: 6 passed. If the "64 KiB/token" assertion fails, check `_candidate_line`: 65536 // 1024 is 64, so the line reads `64 KiB/token`.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
git add src/sous/tune/report.py tests/test_tune_report.py
git commit -m "feat(tune): render the report and the config diff, apply on request" -m "The user reads the machine, every candidate's arithmetic, every arm's numbers and the rule's choice before any diff, and the diff is a tomlkit round-trip so comments and ordering survive an apply." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 10: Preconditions, orchestration, and the `sous tune` subcommand

**Files:**
- Create: `src/sous/tune/daemon.py`
- Modify: `src/sous/tune/__init__.py` (add `main`)
- Modify: `src/sous/cli.py` (the `tune` subparser and dispatch, ~line 897 and ~line 940)
- Test: `tests/test_tune_daemon.py`, `tests/test_tune_main.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `sous.cli._sous_request(port, method, path, json=None) -> tuple[int, bytes]`, `sous.cli._port_open(port)`, `sous.cli._launchd_loaded(label)`, `sous.cli.LABEL`; everything from Tasks 3–9.
- Produces:

```python
# daemon.py
@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str                 # what happened, one line, printed either way

def ready_for_tune(port: int, *, request=None, port_open=None) -> Readiness

# __init__.py
def main(args: argparse.Namespace, **deps) -> int
```

`ready_for_tune`: no daemon on the port → ready ("no daemon on port N"). `GET /sous/status` not 200 → not ready ("port N answered <status>; is it sous?"). `engine.holders > 0` → not ready ("a sous claude session holds the model; finish it first"). `queue.running` or `inflight` non-empty → not ready ("the daemon is busy: N task(s) running, M turn(s) in flight"). `engine.loaded` or `engine.loading` → `POST /sous/unload`: 200 → ready ("the daemon released the model"); 409 → not ready (the body's `error.message`); anything else → not ready ("unload answered <status>"). Otherwise ready ("the daemon holds no model").

`main` runs the quick stage end to end; every external effect is a keyword in `deps` with a production default, so the test drives it with fakes: `detect`, `load_table`, `describe`, `ready`, `bench`, `is_cached`, `size`, `download`, `ask`, `isatty`, `managed`, `out`. Exit codes: 0 done (with or without an apply), 2 refused before measuring (precondition, no arms, usage), 1 a failure while measuring.

- [ ] **Step 1: Write the failing daemon tests**

`tests/test_tune_daemon.py`:

```python
import json

from sous.tune.daemon import ready_for_tune


def _status(**engine):
    doc = {"engine": {"loaded": False, "loading": False, "holders": 0, **engine},
           "inflight": [], "queue": {"queued": 0, "running": 0}}
    return doc


def _request(status_doc, unload=(200, {"unloaded": True, "reason": None})):
    calls = []

    def request(port, method, path, json=None):
        calls.append((method, path))
        if path == "/sous/status":
            return 200, json_bytes(status_doc)
        if path == "/sous/unload":
            code, body = unload
            return code, json_bytes(body)
        raise AssertionError(path)

    return request, calls


def json_bytes(obj):
    return json.dumps(obj).encode()


def test_no_daemon_is_ready():
    r = ready_for_tune(8383, request=lambda *a, **k: (0, b""), port_open=lambda p: False)
    assert r.ready and "no daemon" in r.reason


def test_a_loaded_idle_daemon_is_asked_to_unload():
    request, calls = _request(_status(loaded=True))
    r = ready_for_tune(8383, request=request, port_open=lambda p: True)
    assert r.ready and "released" in r.reason
    assert calls == [("GET", "/sous/status"), ("POST", "/sous/unload")]


def test_a_held_model_a_running_task_and_a_turn_each_refuse_without_unloading():
    for doc, word in (
        (_status(loaded=True, holders=1), "sous claude"),
        ({**_status(loaded=True), "queue": {"queued": 0, "running": 1}}, "busy"),
        ({**_status(loaded=True), "inflight": [{"id": "msg_1"}]}, "busy"),
    ):
        request, calls = _request(doc)
        r = ready_for_tune(8383, request=request, port_open=lambda p: True)
        assert r.ready is False and word in r.reason
        assert calls == [("GET", "/sous/status")]


def test_a_refused_unload_reports_the_daemons_reason():
    request, _ = _request(
        _status(loading=True),
        unload=(409, {"type": "error",
                      "error": {"type": "conflict_error", "message": "a generation is in flight"}}),
    )
    r = ready_for_tune(8383, request=request, port_open=lambda p: True)
    assert r.ready is False and r.reason == "a generation is in flight"


def test_something_else_on_the_port_is_not_ready():
    r = ready_for_tune(8383, request=lambda *a, **k: (404, b"nope"), port_open=lambda p: True)
    assert r.ready is False and "404" in r.reason
```

- [ ] **Step 2: Implement `daemon.py`**

```python
"""Whether a tune may start: the daemon must hold no model, no session and
no work, and it is asked to release the weights rather than stopped."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str


def _body(raw: bytes) -> dict:
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def ready_for_tune(
    port: int,
    *,
    request: Callable[..., tuple[int, bytes]] | None = None,
    port_open: Callable[[int], bool] | None = None,
) -> Readiness:
    from sous.cli import _port_open, _sous_request

    call = request or _sous_request
    if not (port_open or _port_open)(port):
        return Readiness(True, f"no daemon on port {port}")
    status, raw = call(port, "GET", "/sous/status")
    if status != 200:
        return Readiness(False, f"port {port} answered {status} to /sous/status; is it sous?")
    doc = _body(raw)
    engine = doc.get("engine") or {}
    if engine.get("holders"):
        return Readiness(False, "a sous claude session holds the model; finish it first")
    running = (doc.get("queue") or {}).get("running") or 0
    inflight = len(doc.get("inflight") or [])
    if running or inflight:
        return Readiness(
            False, f"the daemon is busy: {running} task(s) running, {inflight} turn(s) in flight"
        )
    if not (engine.get("loaded") or engine.get("loading")):
        return Readiness(True, "the daemon holds no model")
    status, raw = call(port, "POST", "/sous/unload")
    if status == 200:
        return Readiness(True, "the daemon released the model")
    if status == 409:
        message = ((_body(raw).get("error") or {}).get("message")) or "the daemon refused"
        return Readiness(False, str(message))
    return Readiness(False, f"unload answered {status}")
```

Run: `uv run pytest tests/test_tune_daemon.py -v` → 5 passed.

- [ ] **Step 3: Write the failing orchestration tests**

`tests/test_tune_main.py`:

```python
import argparse
from datetime import date

from sous.config import SousConfig, load_config
from sous.tune import main
from sous.tune.bench import BenchRow
from sous.tune.candidates import Candidate, Table
from sous.tune.candidates import describe as real_describe
from sous.tune.daemon import Readiness
from sous.tune.hardware import Hardware
from tests import tune_fixtures as fx

M = "mlx-community/Qwen3.8-27B-4bit"
D = "z-lab/Qwen3.8-27B-DFlash2"


def _args(**over):
    base = dict(quick=True, models=None, repeat=1, resume=None, yes=False, apply=False)
    base.update(over)
    return argparse.Namespace(**base)


def _hardware(tmp_path, working_set=fx.M5_PRO_WORKING_SET):
    return Hardware(chip="M5 Pro", memory_bytes=64 * 2**30, working_set_bytes=working_set,
                    architecture="applegpu_g17s", nax=True, nax_reason=None, macos="26.6",
                    versions={}, hub_cache=str(tmp_path / "hub"), disk_free_bytes=10**12)


def _table():
    return Table(checked=date.today(), validated_with="x", trusted_orgs=("mlx-community",),
                 publishers=("Qwen",),
                 candidates=(Candidate(M, "27b-dense", (D,)),
                             Candidate("mlx-community/Qwen3.5-9B-MLX-4bit", "9b", ())))


def _configs():
    return {M: fx.qwen_27b(), D: fx.dflash2_27b(),
            "mlx-community/Qwen3.5-9B-MLX-4bit": fx.qwen_9b()}


def _sizes():
    return {M: 16_100_000_000, D: 3_850_000_000, "mlx-community/Qwen3.5-9B-MLX-4bit": 6 * 10**9}


def _bench(scores):
    def bench(arm, **kw):
        s = scores.get(arm.label, 10.0)
        return BenchRow(label=arm.label, model_id=arm.model_id, drafter_id=arm.drafter_id,
                        block_size=arm.block_size, window=arm.window, ok=True, error=None,
                        load_seconds=1.0, prefill_tps_2k=400.0, prefill_tps_16k=350.0,
                        decode_tps_1k=s + 5, decode_tps_16k=s, ttft_seconds=1.0,
                        peak_memory_bytes=1, spread=0.0)
    return bench


def _deps(tmp_path, *, scores, cached=(), answers=(), ready=Readiness(True, "no daemon"),
          isatty=True):
    configs, sizes = _configs(), _sizes()
    it = iter(answers)
    downloaded = []
    return dict(
        detect=lambda: _hardware(tmp_path), load_table=lambda: _table(),
        describe=lambda mid, config_fn=None, size_fn=None: real_describe(
            mid, config_fn=lambda m: configs[m], size_fn=lambda m: sizes[m]),
        ready=lambda port: ready, bench=_bench(scores),
        is_cached=lambda rid: rid in cached, size=lambda rid: sizes[rid],
        download=downloaded.append, ask=lambda prompt: next(it, "n"),
        isatty=lambda: isatty, managed=lambda: False, out=print,
    ), downloaded


def _cfg(tmp_path, **over):
    p = tmp_path / "config.toml"
    p.write_text("# mine\n[model]\nspeculative_block_size = 3\n")
    return SousConfig(data_dir=tmp_path, config_path=p, **over)


def test_a_quick_run_measures_the_fitting_arms_and_applies_on_yes(tmp_path, capsys):
    deps, downloaded = _deps(tmp_path, scores={"Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0,
                                               "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3": 18.0},
                             cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"), answers=("y",))
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "## Throughput" in out and "quality untested" in out
    assert "+speculative_block_size = 2" in out
    assert "Apply these changes" in out
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2
    assert "# mine" in (tmp_path / "config.toml").read_text()
    assert list((tmp_path / "tune").iterdir())   # the run dir with results and the report
    assert downloaded == []


def test_downloads_are_asked_one_by_one_and_a_refused_model_loses_its_arms(tmp_path, capsys):
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("n",))
    code = main(_args(), config=_cfg(tmp_path), **deps)
    out = capsys.readouterr().out
    assert code == 0
    assert "Download #1" in out and "Qwen3.5-9B-MLX-4bit" in out
    assert downloaded == [] and "Qwen3.5-9B-MLX-4bit" not in out.split("## Throughput")[1]


def test_an_approved_download_is_fetched_before_any_measurement(tmp_path, capsys):
    order = []
    deps, downloaded = _deps(tmp_path, scores={}, cached=(M, D), answers=("y",))
    real_bench = deps["bench"]
    deps["bench"] = lambda arm, **kw: (order.append("bench"), real_bench(arm, **kw))[1]
    deps["download"] = lambda rid: order.append(f"download {rid}")
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert order[0] == "download mlx-community/Qwen3.5-9B-MLX-4bit"


def test_a_busy_daemon_refuses_before_anything_is_measured(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, ready=Readiness(False, "the daemon is busy: 1 task(s)"))
    called = []
    deps["bench"] = lambda arm, **kw: called.append(arm)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 2
    assert "busy" in capsys.readouterr().out and called == []


def test_a_model_that_does_not_fit_gets_guidance_and_no_diff(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    deps["detect"] = lambda: _hardware(tmp_path, working_set=fx.M2_AIR_WORKING_SET)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "refused" in out and "cannot recommend" in out and "Apply these changes" not in out


def test_without_a_tty_nothing_is_applied_unless_apply_is_passed(tmp_path, capsys):
    scores = {"Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @2": 20.0}
    deps, _ = _deps(tmp_path, scores=scores, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"),
                    isatty=False)
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 3
    assert main(_args(apply=True), config=_cfg(tmp_path), **deps) == 0
    assert load_config(tmp_path / "config.toml").speculative_block_size == 2


def test_resume_skips_arms_already_in_the_run(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, "mlx-community/Qwen3.5-9B-MLX-4bit"))
    seen = []
    real = deps["bench"]
    deps["bench"] = lambda arm, **kw: (seen.append(arm.label), real(arm, **kw))[1]
    assert main(_args(), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    assert main(_args(resume=run_id), config=_cfg(tmp_path), **deps) == 0
    assert seen == [] and first


def test_a_full_run_is_refused_in_this_version(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={})
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 2
    assert "--quick" in capsys.readouterr().out
```

- [ ] **Step 4: Implement `main`**

Replace `src/sous/tune/__init__.py` with:

```python
"""`sous tune`: benchmark this machine, pick the settings, show the diff."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date

from sous.config import SousConfig, load_config
from sous.tune import arms as arms_mod
from sous.tune import bench as bench_mod
from sous.tune import candidates as cand_mod
from sous.tune import daemon as daemon_mod
from sous.tune import hardware as hw_mod
from sous.tune import hub as hub_mod
from sous.tune import report as report_mod
from sous.tune.decide import quick_decision
from sous.tune.rundir import RunDir

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _managed() -> bool:
    from sous.cli import LABEL, _launchd_loaded

    return _launchd_loaded(LABEL)


def _describe_all(ids: set[str], describe, out) -> dict[str, cand_mod.Checkpoint]:
    found: dict[str, cand_mod.Checkpoint] = {}
    for model_id in sorted(ids):
        try:
            found[model_id] = describe(model_id)
        except Exception as e:  # noqa: BLE001 — an unreadable checkpoint is skipped, named
            out(f"  {model_id}: cannot read its config ({type(e).__name__}); skipped")
    return found


def _consented(arms, plan, approved) -> list:
    """Arms whose snapshots are all on disk or approved: a refused model loses
    every arm, a refused drafter leaves the model's no-drafter arm."""
    refused = {d.repo_id for d in plan} - approved
    return [a for a in arms if a.model_id not in refused and a.drafter_id not in refused]


def main(args: argparse.Namespace, *, config: SousConfig | None = None, **deps) -> int:
    out: Callable[..., None] = deps.get("out", print)
    if not getattr(args, "quick", False):
        out("sous tune: only the quick stage exists in this version; run `sous tune --quick`")
        return EXIT_REFUSED
    user = config or load_config()
    hardware = deps.get("detect", hw_mod.detect)()
    table = deps.get("load_table", cand_mod.load_table)()
    ready = deps.get("ready", daemon_mod.ready_for_tune)(user.server_port)
    out(f"daemon: {ready.reason}")
    if not ready.ready:
        return EXIT_REFUSED

    candidates = list(table.candidates)
    if getattr(args, "models", None):
        candidates = [
            next((c for c in table.candidates if c.id == m), cand_mod.Candidate(m, "manual", ()))
            for m in args.models
        ]
    ids = {c.id for c in candidates} | {d for c in candidates for d in c.drafters}
    ids.add(user.model_id)
    if user.speculative_draft_id:
        ids.add(user.speculative_draft_id)
    describe = deps.get("describe", cand_mod.describe)
    checkpoints = _describe_all(ids, describe, out)
    arms, refusals = arms_mod.quick_arms(
        user, candidates, checkpoints, working_set_bytes=hardware.working_set_bytes
    )
    if not arms:
        out("no candidate fits this machine at the configured window:")
        for r in refusals:
            out(f"  {r.model_id}: {r.reason}")
        return EXIT_REFUSED

    tiers = {a.model_id: a.tier for a in arms}
    items = []
    for a in arms:
        items.append((a.model_id, f"candidate ({tiers[a.model_id]} tier)", "its arms"))
        if a.drafter_id:
            items.append((a.drafter_id, f"drafter for {a.model_id}",
                          f"{a.model_id} runs without a drafter"))
    plan = hub_mod.plan_downloads(
        items, cached=deps.get("is_cached", hub_mod.is_cached), size=deps.get("size", hub_mod.snapshot_bytes)
    )
    if getattr(args, "yes", False):
        approved = {d.repo_id for d in plan}
    else:
        approved = hub_mod.ask_consent(plan, hardware.disk_free_bytes, ask=deps.get("ask", input), out=out)
    arms = _consented(arms, plan, approved)
    if not arms:
        out("nothing left to measure after the downloads you declined")
        return EXIT_REFUSED
    hub_mod.fetch(sorted(approved), download=deps.get("download"), out=out)

    base = user.data_dir / "tune"
    run = RunDir.existing(base, args.resume) if getattr(args, "resume", None) else RunDir.new(base)
    run.write_json("hardware.json", hardware.as_dict())
    done = {r["label"] for r in run.rows("bench")}
    bench = deps.get("bench", bench_mod.bench_arm)
    out(f"measuring {len(arms)} arm(s); results in {run.path}")
    for arm in arms:
        if arm.label in done:
            continue
        out(f"  {arm.label} ...")
        row = bench(arm, repeat=getattr(args, "repeat", 2))
        run.append("bench", row.as_dict())
    rows = [bench_mod.BenchRow.from_dict(r) for r in run.rows("bench")]

    choice = quick_decision(user, arms, rows)
    text = report_mod.render_report(
        hardware=hardware, table_age_days=(date.today() - table.checked).days,
        checkpoints=checkpoints, refusals=refusals, rows=rows, choice=choice,
        current_model=user.model_id,
    )
    run.write_text("report.md", text)
    out(text)
    if choice is None or not choice.changes:
        return EXIT_OK
    new_text, diff = report_mod.config_diff(user.config_path, choice.changes)
    out(diff)
    apply = getattr(args, "apply", False) or getattr(args, "yes", False)
    if not apply:
        if not deps.get("isatty", sys.stdin.isatty)():
            out("not a terminal: nothing applied (pass --apply to write the changes)")
            return EXIT_OK
        try:
            answer = deps.get("ask", input)(f"Apply these changes to {user.config_path}? [y/N] ")
        except EOFError:
            answer = ""
        apply = answer.strip().lower() in ("y", "yes")
    if not apply:
        out("nothing applied")
        return EXIT_OK
    backup = report_mod.apply_changes(user.config_path, new_text, run.run_id)
    out(f"applied; backup at {backup}" if backup else "applied")
    note = report_mod.restart_note(choice.changes, managed=deps.get("managed", _managed)(), label="com.sous.daemon")
    if note:
        out(note)
    return EXIT_OK
```

Run: `uv run pytest tests/test_tune_main.py -v` → 8 passed. If ruff complains about line length in `main`, wrap the long calls; if ty complains about `deps.get(...)` callables, annotate `deps: dict[str, Callable]` via `**deps: Callable`.

- [ ] **Step 5: Wire the CLI**

In `src/sous/cli.py`, add the subparser after the `uninstall-launchd` one:

```python
    tune = sub.add_parser(
        "tune",
        help="measure this machine and propose [model] settings (quick stage: drafter, block size, window)",
    )
    tune.add_argument("--quick", action="store_true", help="the throughput stage only")
    tune.add_argument("--models", nargs="+", metavar="ID", help="candidate ids instead of the table")
    tune.add_argument("--repeat", type=int, default=2, help="attempts per measurement (best wins)")
    tune.add_argument("--resume", metavar="RUN_ID", help="continue a run under ~/.sous/tune")
    tune.add_argument("--yes", action="store_true", help="approve every download and apply")
    tune.add_argument("--apply", action="store_true", help="apply the diff without asking")
```

and the dispatch branch:

```python
    elif args.command == "tune":
        from sous.tune import main as tune_main

        raise SystemExit(tune_main(args))
```

Add to `tests/test_cli.py` (add `import pytest` at the top if the file lacks it):

```python
def test_tune_subcommand_dispatches_with_its_flags(monkeypatch):
    import sous.tune

    seen = {}
    monkeypatch.setattr(sous.tune, "main", lambda args, **k: seen.update(vars(args)) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(["tune", "--quick", "--repeat", "3", "--models", "a/b", "c/d", "--yes"])
    assert e.value.code == 0
    assert seen["quick"] is True and seen["repeat"] == 3
    assert seen["models"] == ["a/b", "c/d"] and seen["yes"] is True and seen["apply"] is False
```

Run: `uv run pytest tests/test_cli.py -k tune -v` → passed.

- [ ] **Step 6: Lint, type-check, the whole suite, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -m "not model" && uv lock --check
git add src/sous/tune/__init__.py src/sous/tune/daemon.py src/sous/cli.py tests/test_tune_daemon.py tests/test_tune_main.py tests/test_cli.py
git commit -m "feat(cli): sous tune --quick measures the machine and proposes the settings" -m "The daemon is asked to release the model rather than stopped, every download is a separate answer, every arm is measured through the real engine, and the diff is applied only on the user's yes." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Documentation

**Files:**
- Modify: `README.md` (a "Tuning" section before "Smaller machines"; the memory table's introduction points at `sous tune`; the "Managing the daemon" section lists `sous tune`)
- Modify: `README.md` configuration reference under `[model]`: which keys `sous tune --quick` may write

- [ ] **Step 1: Write the "Tuning" section**

Insert before `## Smaller machines`:

```markdown
## Tuning

`sous tune --quick` measures this machine and proposes the `[model]` settings
that cannot change what the model says: the drafter, its block size, and a
context window that fits. It detects the chip, the Metal working set and
whether the GPU has tensor units, fits every curated candidate to memory
(and prints the arithmetic for each one it refuses), lists every download it
would need and asks about each one separately, then measures prefill and
decode throughput of every arm through sous's own engine — the numbers a
delegated task or a gateway turn would see. It ends with a report, a diff
of `~/.sous/config.toml`, and a question:

```
Apply these changes to ~/.sous/config.toml? [y/N]
```

Nothing is written before that yes; a backup is kept beside the file. The
daemon is asked to release the model first (`POST /sous/unload`) and
refuses while a `sous claude` session holds it or a task is running — the
tune waits for neither, it tells you. Results and the report land under
`~/.sous/tune/<run-id>/`; `--resume <run-id>` continues an interrupted run,
`--models ID ...` measures ids of your own, `--yes` answers every prompt
for scripted use. Other models than the configured one are measured and
reported with a "quality untested" label; only the full run (a later
release) may propose a model change.
```

- [ ] **Step 2: Point the memory table at the tune**

Replace the first paragraph of `## Smaller machines` with:

```markdown
`sous tune --quick` tells you what fits and how fast it runs here; the table
below is the fallback for a machine that cannot reach the Hub. The
alternative shares the default's `qwen3_5` architecture, so it loads
through the exact same mlx-vlm path — edit `[model].id` in
`~/.sous/config.toml` and the next delegation downloads and uses it.
```

- [ ] **Step 3: The config reference and the daemon commands**

Under the `[model]` reference (the commented example config), add after the `speculative_block_size` line:

```toml
# `sous tune --quick` may rewrite speculative_draft_id, speculative_block_size
# and max_context_tokens (and [gateway].max_context_tokens) after showing you
# the diff; it never changes id or int8_prefill.
```

In "Managing the daemon", add one bullet: `sous tune --quick` — measure this machine and propose settings (see Tuning).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe sous tune --quick" -m "The tune replaces reading a memory table with running a command; the table stays as the offline fallback." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Model-marked end-to-end check

**Files:**
- Create: `tests/test_tune_model.py`

- [ ] **Step 1: Write the test**

```python
"""Local only: one quick arm through the real engine on the smallest model,
so the bench's gauge reads, the session thread and the unload are exercised
against mlx itself, not a fake."""

import pytest

from sous.config import SousConfig
from sous.tune.arms import Arm
from sous.tune.bench import bench_arm

pytestmark = pytest.mark.model

TINY = "mlx-community/Qwen3-0.6B-4bit"


def test_bench_arm_measures_the_tiny_model_and_releases_it(tmp_path):
    cfg = SousConfig(data_dir=tmp_path / "d", config_path=tmp_path / "c.toml",
                     model_id=TINY, speculative_draft_id="", max_context_tokens=8192)
    arm = Arm(label="tiny", config=cfg, model_id=TINY, drafter_id="", block_size=0,
              window=8192, gateway_window=None, tier="test", current=True)
    row = bench_arm(arm, repeat=1, out=lambda *a, **k: None)
    assert row.ok, row.error
    assert row.prefill_tps_2k and row.prefill_tps_2k > 0
    assert row.decode_tps_1k and row.decode_tps_1k > 0
    assert row.decode_tps_16k is None   # the window is too small for the long context
    assert row.ttft_seconds is not None and row.load_seconds is not None
    assert row.peak_memory_bytes and row.peak_memory_bytes > 0
```

- [ ] **Step 2: Run it locally (downloads ~350 MB on first use)**

Run: `uv run pytest -m model tests/test_tune_model.py -v`
Expected: 1 passed. On a machine without the weights this is the one test allowed to download.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tune_model.py
git commit -m "test(tune): bench one arm on the tiny model end to end" -m "The fakes cannot prove the gauge reads and the unload against mlx; the 0.6B can, in seconds, wherever the model tests run." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage (PR 1 scope):** unload route and `unload_now` (Tasks 1–2); hardware (3); table, `describe`, derived properties, fit with window lowering and printed arithmetic (4); download plan and per-snapshot consent up front, fetch before measuring (5); arms with the user's current configuration always present (6); bench through the real engine on one session thread, gauges read per turn, best-of-`--repeat`, spread, peak memory, unload between arms (7); the quick rule with its invariants (8); report sections, stale-table warning, tomlkit diff, backup, apply prompt, restart note (9); preconditions, orchestration, `--resume`, non-TTY behaviour, the CLI (10); README (11); a model-marked check (12). Discovery, the suite and the greedy engine change are PRs 2 and 3 by the spec's delivery section.
- **Deferred detail an implementer must not "fix":** `describe()` imports `sous.tune.hub` lazily because Task 5 lands after Task 4; the import stays lazy afterwards so `candidates` never pulls the Hub client at import time.
- **Type consistency:** `Refusal(model_id, drafter_id, reason)` is produced in Task 6 and consumed by `render_report` in Task 9 and `main` in Task 10; `BenchRow.as_dict/from_dict` round-trip is what `RunDir` rows carry; `QuickChoice.changes` is the `{section: {key: value}}` shape `config_diff`, `apply_changes` and `restart_note` take.
