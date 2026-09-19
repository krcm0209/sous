# `sous tune` full run Implementation Plan (PR 2 of the sous tune spec)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the full `sous tune` run: a graded suite of eight mechanical coding tasks through the real worker loop, the model stage and the winner stage (INT8 prefill, greedy sampling), the full decision rule, `--resume` over suite runs, the ETA after the quick stage, and the one-line engine change that lets `temperature = 0` reach mlx-vlm's greedy speculative path.

**Architecture:** `sous tune` (no `--quick`) runs PR 1's quick stage unchanged, then loads each model's fastest quality-neutral arm once and drives every suite task through `sous.worker.run_task` under the real sandbox with a scratch task store, denying every approval. Each run is graded by a hidden grader shipped with the package (unittest modules run in a subprocess with the worker's project as the working directory, or a `grade.py` for the two categories that need more), appended to `results.jsonl` as `kind: "suite"` rows the way bench rows already are, and the rule in `decide.full_decision` picks the fastest eligible arm. The winner then gets one extra arm per quality-affecting setting, judged by the same rule against the winner's own result. Everything still ends in the report, the diff and the apply prompt PR 1 built.

**Tech Stack:** Python 3.14 stdlib (`unittest`, `ast`, `subprocess`, `tomllib`, `importlib.resources`), the existing `sous.worker` / `sous.tasks` / `sous.toolexec` / `sous.engine` modules, mlx-vlm (function-local imports only), tomlkit, pytest with `tests/fake_engine.py`.

**Spec:** `docs/superpowers/specs/2026-09-17-sous-tune-design.md` — this plan implements its "Delivery" item 2 (the full run, the suite, the runner and its metrics, the model and winner stages, the full decision rule, `--resume`, the greedy engine change). `--discover` is PR 3. PR 1 (`--quick`, merged as 778d6b5) is the base: read `src/sous/tune/*.py` before starting any task here.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid and used in this repo.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`.
- `mlx`, `mlx_lm`, `mlx_vlm` imports stay function-local (the lint job runs on ubuntu). Every thread that touched mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits.
- No new dependencies. `uv lock --check` must stay green. The suite's tasks, graders and reference solutions use the standard library only.
- CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. Run all four before every commit.
- Tests never touch the real `~/.sous` or `~/.cache/huggingface`: every path comes from `tmp_path`, every network call and every model load is injected.
- `pytest` `addopts = "-q"` already; never add another `-q`.
- Comments explain non-obvious *why*; never cite this plan, the spec, task or step numbers in code or test comments.
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Never put the maintainer's name or work home path in commits, files or PR text.
- Line length 100 (ruff); ruff rules E, F, I, UP, B, SIM. Suite fixtures are Python files inside the package and are linted and formatted like everything else.
- The suite runs the real worker loop under the real sandbox: `DEFAULT_ALLOWLIST` plus `python -m unittest` and `python3 -m unittest`, approvals denied, never granted.
- `--quick` never changes the model or a quality-affecting setting; its code path must behave exactly as before this PR (every PR 1 test keeps passing unchanged except the one that asserted the full run is refused).
- Two models are never resident at once: the suite loads one arm's engine per dedicated thread and releases it the way the bench does.

## Decisions this plan makes where the spec left room

- **Repetition incidents are read from the transcript's `tool` events**, not by re-parsing `generation` text: a `tool` event is exactly one executed call with its name and arguments, so three identical consecutive ones is the metric the spec defines with nothing to re-parse (a `finish` call is never a `tool` event and cannot repeat).
- **The suite calls `run_task` directly**, on the arm's dedicated thread, after `store.claim_next()`: that is the whole agent loop the daemon runs (generate → parse → execute → append, the sandbox, the approval hook, the budgets, the verify commands); `run_worker_loop` only adds queue polling and the idle sweep around it, and it would need a second thread that owns the engine's mlx state.
- **Output tokens** come from the engine's own deltas: the runner wraps the engine in a proxy that hands the generate call a `ReplaySafe` accounting callback (the worker passes none), so the count is what `Delta.output_tokens` reports and a warm-cache failure still retries cold.
- **`solution/` is the complete solved project**, not an overlay, so a grader test can point at it directly and a task whose reference result deletes files (the config migration) needs no manifest.
- **The winner stage adopts at most one extra arm**: the rule is run over `{winner, int8 arm, greedy arm}` with the winner as reference, so a setting is adopted only when its own measured arm is eligible and faster, and two settings never land together untested as a pair.
- **`grade.py` is optional**: a task whose `grade/` holds only `test_*.py` modules is scored `passed / total` by the shared runner; the two categories the spec singles out (test scaffolding with mutants, the docstring sweep's AST check) ship a `grade.py` with `grade(project, tests) -> (score, detail)`.
- **The spec's `--runs` default is 2** and the completed-runs allowance is `runs_per_task`, exactly as written.

## File Structure

| file | responsibility |
|---|---|
| `src/sous/engine/vlm.py` (modify) | no sampler at `temperature = 0`; `decode` passes `sampler=None, temperature=0` |
| `src/sous/engine/base.py` (modify) | `default_engine_factory(config)` — the factory `EngineManager` builds by default, callable by the runner |
| `src/sous/tune/arms.py` (modify) | `Arm.int8_prefill`, `Arm.greedy`, `Arm.suite_key`; `winner_stage_arms()` |
| `src/sous/tune/bench.py` (modify) | `release()` — the arm teardown, shared with the suite runner |
| `src/sous/tune/suite/__init__.py` (create) | `SuiteTask`, `load_tasks()`, validation of `task.toml` |
| `src/sous/tune/suite/grading.py` (create) | `Grade`, `run_tests()` (the unittest subprocess), `grade_task()` |
| `src/sous/tune/suite/unittests.py` (create) | `python -m sous.tune.suite.unittests DIR` — runs a directory of unittest modules and prints JSON |
| `src/sous/tune/suite/runner.py` (create) | `SuiteRun`, `CountingEngine`, `run_one()`, `run_suite()`, `metrics_from_transcript()`, `estimate_seconds()` |
| `src/sous/tune/suite/tasks/<name>/` (create ×8) | `task.toml`, `project/`, `grade/`, `solution/` |
| `src/sous/tune/decide.py` (modify) | `ArmSummary`, `summarize()`, `model_stage()`, `FullChoice`, `full_decision()` |
| `src/sous/tune/report.py` (modify) | the Suite section, the rule's lines, `FullChoice` in the Choice section |
| `src/sous/tune/__init__.py` (modify) | the full run: ETA, model stage, winner stage, `--resume` over suite rows |
| `src/sous/cli.py` (modify) | `--runs`, help text |
| `pyproject.toml` (modify) | `[tool.ty.src] exclude` for the fixture trees (their imports resolve only inside a copied project) |
| `README.md`, `docs/tuning.md` (modify/create) | the full run, what it may change, adding a task or a row |
| `tests/test_engine_vlm_greedy.py`, `tests/test_tune_suite.py`, `tests/test_tune_runner.py` (create); `tests/test_tune_arms.py`, `tests/test_tune_bench.py`, `tests/test_tune_decide.py`, `tests/test_tune_report.py`, `tests/test_tune_main.py`, `tests/test_tune_model.py`, `tests/test_cli.py`, `tests/test_engine_base.py` (modify) | |

Package data: `[tool.hatch.build.targets.wheel] packages = ["src/sous"]` ships every file under `src/sous`, `candidates.toml` included today, so the fixture trees need no packaging change. `importlib.resources.files("sous.tune.suite") / "tasks"` resolves them installed and editable alike.

---

### Task 1: Greedy sampling reaches mlx-vlm's greedy speculative path

**Files:**
- Modify: `src/sous/engine/vlm.py` (`VLMEngine.__init__` sampler line; `decode`'s `stream_generate` call)
- Test: `tests/test_engine_vlm_greedy.py` (create)

**Interfaces:**
- Consumes: mlx-vlm 0.7.x `generate_step` computes `sampler_is_greedy = sampler is None and temperature == 0` (`mlx_vlm/generate/ar.py`), and `stream_generate` forwards its kwargs to it. The speculative walk takes the exact-match verify branch only when that flag is true.
- Produces: `VLMEngine._sampler` is `None` at `temperature == 0`; `decode` passes `sampler=None, temperature=0` then, and the configured sampler (no `temperature` kwarg) otherwise. The LM backend is untouched.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_vlm_greedy.py
"""At temperature 0 the VLM engine hands mlx-vlm no sampler: its generate_step
takes `sampler is None and temperature == 0` as the greedy case, and only
that case runs the speculative walk's exact-match verify. An argmax sampler
built by make_sampler is a callable, so it measured the sampled code path
under a greedy pick."""

import sys
import threading
import types

from sous.engine.promptcache import PromptMemo
from sous.engine.vlm import VLMEngine


class _Layer:
    def __init__(self, offset: int = 0):
        self.offset = offset


def _engine(sampler) -> VLMEngine:
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = types.SimpleNamespace(config=types.SimpleNamespace(eos_token_id=0))  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = sampler
    engine._positional = False
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    return engine


def _stub(monkeypatch, calls: dict):
    def stream_generate(model, processor, prompt, **kwargs):
        calls.update(kwargs)
        return iter(())

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )


def test_temperature_zero_builds_no_sampler(monkeypatch):
    made = []
    # The import inside _make_sampler is `from mlx_vlm.sample_utils import
    # make_sampler`: a stub under that name in sys.modules is what it finds.
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **kw: made.append(kw) or object()),
    )
    assert VLMEngine._make_sampler(temperature=0, top_p=0.8, top_k=20) is None
    assert made == []
    assert VLMEngine._make_sampler(temperature=0.7, top_p=0.8, top_k=20) is not None
    assert made == [{"temp": 0.7, "top_p": 0.8, "top_k": 20}]


def test_decode_without_a_sampler_asks_for_greedy_generation(monkeypatch):
    calls: dict = {}
    _stub(monkeypatch, calls)
    _engine(None).decode([_Layer()], [4, 5], 8)
    assert calls["sampler"] is None
    assert calls["temperature"] == 0


def test_decode_with_a_sampler_passes_it_and_no_temperature(monkeypatch):
    calls: dict = {}
    _stub(monkeypatch, calls)
    sampler = object()
    _engine(sampler).decode([_Layer()], [4, 5], 8)
    assert calls["sampler"] is sampler
    assert "temperature" not in calls
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_vlm_greedy.py -v`
Expected: FAIL — `VLMEngine` has no `_make_sampler`; `calls["temperature"]` raises `KeyError`.

- [ ] **Step 3: Implement**

In `src/sous/engine/vlm.py`, add a static method to `VLMEngine` and use it in `__init__`:

```python
    @staticmethod
    def _make_sampler(*, temperature: float, top_p: float, top_k: int):
        """None at temperature 0: mlx-vlm's generate_step treats no sampler
        plus temperature 0 as greedy, and only then does the speculative walk
        take its exact-match verify; an argmax sampler is a callable like any
        other and gets the sampled path (krcm0209/sous#87)."""
        if temperature == 0:
            return None
        from mlx_vlm.sample_utils import make_sampler

        return make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
```

Replace `from mlx_vlm.sample_utils import make_sampler` and `self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)` in `__init__` with:

```python
        self._sampler = self._make_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
```

In `decode`, before the `stream_generate` loop:

```python
        # No sampler means greedy to generate_step only together with an
        # explicit temperature of 0; the configured sampler needs no
        # temperature at all.
        sampling = {"sampler": None, "temperature": 0} if self._sampler is None else {
            "sampler": self._sampler
        }
```

and replace `sampler=self._sampler,` in the call with `**sampling,`.

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_engine_vlm_greedy.py tests/test_engine_positions.py tests/test_engine_vlm.py -m "not model" -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS. (`test_engine_positions.py` builds engines with `engine._sampler = object()` and must still pass.)

- [ ] **Step 5: Commit**

```bash
git add src/sous/engine/vlm.py tests/test_engine_vlm_greedy.py
git commit -m "feat(engine): hand mlx-vlm no sampler at temperature 0

generate_step reads 'sampler is None and temperature == 0' as greedy and only
then runs the speculative walk's exact-match verify. An argmax sampler from
make_sampler is a callable, so a greedy configuration measured the sampled
path (#87). The LM backend keeps its sampler either way.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Arms carry the winner-stage dimensions

**Files:**
- Modify: `src/sous/tune/arms.py`
- Test: `tests/test_tune_arms.py` (append)

**Interfaces:**
- Consumes: `Arm` (PR 1), `Hardware.nax`, `Checkpoint.model_type`, `Checkpoint.quant.int8_routable`, `int8prefill.SUPPORTED_MODEL_TYPES` (`frozenset({"qwen3_5"})`).
- Produces:
  - `Arm.int8_prefill: bool = False`, `Arm.greedy: bool = False` (mirrors of `Arm.config`).
  - `Arm.suite_key -> tuple[str, str, int, bool, bool]` = `(model_id, drafter_id, block_size, int8_prefill, greedy)`.
  - `winner_stage_arms(winner: Arm, *, nax: bool, checkpoint: Checkpoint) -> list[Arm]`: the INT8 arm (label `"<winner.label> + int8 prefill"`) when `nax` and the checkpoint's `model_type` is in `SUPPORTED_MODEL_TYPES` and `quant.int8_routable` and the winner is not already int8; the greedy arm (label `"<winner.label> greedy"`) when `winner.config.temperature != 0`. Both `current=False`, `dataclasses.replace`d from the winner so every window and fit value carries over.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tune_arms.py`)

```python
from sous.tune.arms import Arm, winner_stage_arms


def _winner(tmp_path, temperature=0.7, int8=False):
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        temperature=temperature,
        int8_prefill=int8,
    )
    return Arm(
        label="27B + DFlash2 @3",
        config=cfg,
        model_id="mlx-community/Qwen3.8-27B-4bit",
        drafter_id="z-lab/Qwen3.8-27B-DFlash2",
        block_size=3,
        window=131072,
        gateway_window=None,
        tier="27b-dense",
        current=True,
        fit_window=131072,
        int8_prefill=int8,
    )


def test_the_suite_key_extends_the_bench_key_with_the_quality_dimensions(tmp_path):
    arm = _winner(tmp_path)
    assert arm.key == ("mlx-community/Qwen3.8-27B-4bit", "z-lab/Qwen3.8-27B-DFlash2", 3)
    assert arm.suite_key == (*arm.key, False, False)


def test_on_nax_a_routable_checkpoint_gets_an_int8_arm_and_a_sampled_winner_a_greedy_arm(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    arms = winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)
    assert [a.label for a in arms] == ["27B + DFlash2 @3 + int8 prefill", "27B + DFlash2 @3 greedy"]
    int8, greedy = arms
    assert int8.int8_prefill and int8.config.int8_prefill and not int8.greedy
    assert greedy.greedy and greedy.config.temperature == 0 and not greedy.int8_prefill
    assert int8.suite_key == (*int8.key, True, False)
    assert greedy.suite_key == (*greedy.key, False, True)
    assert all(not a.current and a.fit_window == 131072 for a in arms)


def test_without_nax_only_the_greedy_arm_is_offered(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    assert [a.label for a in winner_stage_arms(_winner(tmp_path), nax=False, checkpoint=cp)] == [
        "27B + DFlash2 @3 greedy"
    ]


def test_a_checkpoint_int8_cannot_route_gets_no_int8_arm(tmp_path):
    cp = describe(
        "mlx-community/Qwen3.8-27B-mxfp4",
        config_fn=lambda m: fx.qwen_27b({"group_size": 32, "bits": 4, "mode": "mxfp4"}),
        size_fn=lambda m: 15_000_000_000,
    )
    arms = winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)
    assert [a.greedy for a in arms] == [True]


def test_a_winner_already_greedy_and_int8_gets_no_extra_arms(tmp_path):
    cp = _checkpoints()["mlx-community/Qwen3.8-27B-4bit"]
    assert winner_stage_arms(_winner(tmp_path, temperature=0, int8=True), nax=True, checkpoint=cp) == []


def test_a_moe_checkpoint_gets_no_int8_arm(tmp_path):
    cp = describe(
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        config_fn=lambda m: fx.qwen_27b(model_type="qwen3_5_moe"),
        size_fn=lambda m: 20_000_000_000,
    )
    assert [a.greedy for a in winner_stage_arms(_winner(tmp_path), nax=True, checkpoint=cp)] == [True]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_arms.py -v`
Expected: FAIL — `Arm.__init__() got an unexpected keyword argument 'int8_prefill'`, no `winner_stage_arms`.

- [ ] **Step 3: Implement** (in `src/sous/tune/arms.py`)

Add to the `Arm` dataclass after `fit_gateway_window`:

```python
    # The quality-affecting dimensions the winner stage measures, mirrored
    # from `config` so a row can be keyed without reading the config back.
    int8_prefill: bool = False
    greedy: bool = False

    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key plus the two settings only the suite may change."""
        return (*self.key, self.int8_prefill, self.greedy)
```

Add at module level (after `quick_arms`):

```python
def winner_stage_arms(winner: Arm, *, nax: bool, checkpoint: Checkpoint) -> list[Arm]:
    """One extra arm per quality-affecting setting the winner does not have
    yet: INT8 prefill where the tensor units and the checkpoint's quantization
    allow it (the engine refuses anything else with a status, and the arm
    would measure the stock path under the int8 label), and greedy sampling
    for a winner that samples. Each is judged on its own against the winner."""
    from sous.engine.int8prefill import SUPPORTED_MODEL_TYPES

    arms: list[Arm] = []
    routable = (
        nax
        and checkpoint.model_type in SUPPORTED_MODEL_TYPES
        and checkpoint.quant.int8_routable
    )
    if routable and not winner.int8_prefill:
        arms.append(
            dataclasses.replace(
                winner,
                label=f"{winner.label} + int8 prefill",
                config=dataclasses.replace(winner.config, int8_prefill=True),
                current=False,
                int8_prefill=True,
            )
        )
    if winner.config.temperature != 0:
        arms.append(
            dataclasses.replace(
                winner,
                label=f"{winner.label} greedy",
                config=dataclasses.replace(winner.config, temperature=0.0),
                current=False,
                greedy=True,
            )
        )
    return arms
```

`int8prefill` imports nothing from mlx at module level (check: `src/sous/engine/int8prefill.py` imports mlx inside functions), so the import is only kept local to keep `arms` free of the engine package's load-time cost.

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_arms.py tests/test_tune_decide.py tests/test_tune_main.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sous/tune/arms.py tests/test_tune_arms.py
git commit -m "feat(tune): arms carry the int8 and greedy dimensions the winner stage measures

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: The engine factory and the arm teardown become reusable

**Files:**
- Modify: `src/sous/engine/base.py` (`EngineManager.__init__`)
- Modify: `src/sous/tune/bench.py` (`_measure`)
- Test: `tests/test_engine_base.py` (append), `tests/test_tune_bench.py` (append)

**Interfaces:**
- Produces:
  - `sous.engine.base.default_engine_factory(config: SousConfig) -> Callable[[str], Engine]`: the factory `EngineManager` builds when none is given — every `[model]` value mapped onto `_default_factory` exactly as today. `EngineManager.__init__` uses it (`engine_factory or default_engine_factory(config)`).
  - `sous.tune.bench.release(manager: EngineManager, session, *, baseline: int, active_memory: Callable[[], int], label: str, out) -> str | None`: closes the session (5 s join), `unload_now()`, waits up to `_UNLOAD_WAIT_SECONDS` for active memory to fall back within `_UNLOAD_SLACK_BYTES` of `baseline`, prints what it sees, and returns the error text when the weights stayed resident (`"unload refused: …"`, `"memory not released: …"`, `"teardown failed: …"`), `None` when they were freed. Never raises. `_measure` calls it and applies `row.with_teardown_error(error)` when it returns text.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_base.py`:

```python
def test_default_engine_factory_maps_every_model_value_onto_the_backend(monkeypatch, tmp_path):
    from sous.engine import base

    seen = {}

    def fake(model_id, *args, **kwargs):
        seen["model_id"] = model_id
        seen["args"] = args
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(base, "_default_factory", fake)
    cfg = SousConfig(
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        model_id="org/m",
        temperature=0.0,
        top_p=0.9,
        top_k=5,
        prompt_cache=False,
        speculative_draft_id="z/d",
        speculative_block_size=2,
        prompt_cache_gb=1.5,
        max_context_tokens=4096,
        gateway_enabled=True,
        gateway_max_context_tokens=65536,
        int8_prefill=True,
    )
    base.default_engine_factory(cfg)("org/m")
    assert seen["model_id"] == "org/m"
    assert seen["args"] == (0.0, 0.9, 5, False)
    assert seen["kwargs"] == {
        "draft_id": "z/d",
        "draft_block_size": 2,
        "cache_budget": int(1.5 * (1 << 30)),
        "reserve_tokens": 65536,
        "int8_prefill": True,
    }


def test_engine_manager_without_a_factory_uses_the_default_one(monkeypatch, tmp_path):
    from sous.engine import base

    calls = []
    monkeypatch.setattr(
        base, "default_engine_factory", lambda cfg: lambda mid: calls.append((cfg.model_id, mid))
    )
    cfg = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", model_id="org/m")
    base.EngineManager(cfg)._factory("org/m")
    assert calls == [("org/m", "org/m")]
```

(`SousConfig` is already imported at the top of `tests/test_engine_base.py`; check and add the import if it is not.)

Append to `tests/test_tune_bench.py` (the manager shape is the one the existing bench tests use, `EngineManager(arm.config, engine_factory=lambda mid: engine)`):

```python
from sous.engine.base import EngineManager
from sous.tune.bench import release


def test_release_frees_the_engine_and_returns_none(tmp_path):
    engine = FakeEngine([])
    manager = EngineManager(_arm(tmp_path).config, engine_factory=lambda mid: engine)
    managed = manager.get()
    session = managed.session()
    lines = []
    error = release(
        manager, session, baseline=0, active_memory=lambda: 0, label="m", out=lines.append
    )
    assert error is None
    assert engine.unloaded
    assert manager.status()["loaded"] is False
    assert lines == []


def test_release_reports_a_refused_unload_without_waiting(tmp_path, monkeypatch):
    engine = FakeEngine([])
    manager = EngineManager(_arm(tmp_path).config, engine_factory=lambda mid: engine)
    manager.get()
    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 30.0)
    with manager.lease():
        error = release(
            manager, None, baseline=0, active_memory=lambda: 0, label="m", out=lambda *a: None
        )
    assert error == "unload refused: the engine is leased by a turn"
    assert not engine.unloaded


def test_release_reports_memory_that_never_comes_back(tmp_path, monkeypatch):
    engine = FakeEngine([])
    manager = EngineManager(_arm(tmp_path).config, engine_factory=lambda mid: engine)
    manager.get()
    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 0.0)
    error = release(
        manager,
        None,
        baseline=0,
        active_memory=lambda: 3 * 2**30,
        label="m",
        out=lambda *a: None,
    )
    assert error == "memory not released: 3.0 GiB still resident"
    assert engine.unloaded
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_base.py tests/test_tune_bench.py -v`
Expected: FAIL — no `default_engine_factory`, no `release`.

- [ ] **Step 3: Implement**

In `src/sous/engine/base.py`, add above `class EngineManager`:

```python
def default_engine_factory(config: SousConfig) -> Callable[[str], Engine]:
    """The factory EngineManager builds when given none: every [model] value
    mapped onto the backend. Public so a process that must wrap the real
    engine (the tune's suite runner counts its deltas) builds the same one
    rather than a second copy of this mapping."""
    return lambda model_id: _default_factory(
        model_id,
        config.temperature,
        config.top_p,
        config.top_k,
        config.prompt_cache,
        draft_id=config.speculative_draft_id,
        draft_block_size=config.speculative_block_size,
        cache_budget=(
            None if config.prompt_cache_gb is None else int(config.prompt_cache_gb * (1 << 30))
        ),
        # The largest cache one turn can build on this daemon: the
        # gateway's window when it is on, else the worker's.
        reserve_tokens=max(
            config.max_context_tokens,
            config.gateway_max_context_tokens if config.gateway_enabled else 0,
        ),
        int8_prefill=config.int8_prefill,
    )
```

and in `EngineManager.__init__` replace the whole `self._factory = engine_factory or (lambda model_id: _default_factory(...))` expression with:

```python
        self._factory = engine_factory or default_engine_factory(config)
```

(`_default_factory`'s comments about the reserve move with the code; leave nothing behind in `__init__`.)

In `src/sous/tune/bench.py`, add above `_measure`:

```python
def release(
    manager: EngineManager,
    session,
    *,
    baseline: int,
    active_memory: Callable[[], int],
    label: str,
    out: Callable[..., None],
) -> str | None:
    """Free one arm's engine: the text of what went wrong when the weights
    stayed resident (a refused unload, memory that never came back, a
    teardown step that raised), None when they were freed. Never raises —
    the caller's measurement must survive its own teardown — and the run
    must stop on any text, because the next arm's peak-memory reading would
    be two models' worth."""
    try:
        if session is not None:
            session.close()
            # A wedged generation never dequeues _CLOSE, so this can't wait
            # for one; it only gives a healthy thread time to release its
            # mlx state.
            session.join(5.0)
        released = manager.unload_now()
        if not released["unloaded"]:
            # The weights are still resident — say so, and don't wait for
            # memory that cannot come back.
            out(f"  {label}: model not released ({released['reason']})")
            return f"unload refused: {released['reason']}"
        deadline = time.monotonic() + _UNLOAD_WAIT_SECONDS
        while active_memory() > baseline + _UNLOAD_SLACK_BYTES and time.monotonic() < deadline:
            time.sleep(0.2)
        resident = active_memory() - baseline
        if resident > _UNLOAD_SLACK_BYTES:
            # The unload ran but the memory never came back: the same dirty
            # machine as a refused unload.
            out(f"  {label}: {resident / _GIB:.1f} GiB still resident after the unload")
            return f"memory not released: {resident / _GIB:.1f} GiB still resident"
        return None
    except Exception as e:  # noqa: BLE001 — the caller's row survives; only teardown failed
        out(f"  {label}: teardown failed ({type(e).__name__}: {e})")
        return f"teardown failed: {type(e).__name__}: {e}"
```

Replace `_measure`'s whole second `try:` block (from `# Teardown always runs, …` through `row = row.with_teardown_error(f"teardown failed: …")`) with:

```python
    # Teardown always runs, over whatever the try/except above produced: a
    # row already built in `row` can no longer be discarded by a teardown
    # problem the way a bare `finally` that returned from `try` could.
    error = release(
        manager, session, baseline=baseline, active_memory=active_memory, label=arm.label, out=out
    )
    if error is not None:
        row = row.with_teardown_error(error)
```

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_engine_base.py tests/test_tune_bench.py tests/test_tune_main.py tests/test_worker.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS — every existing bench teardown test (`test_a_refused_unload_is_recorded_and_skips_the_settle_wait`, `test_a_teardown_exception_keeps_the_finished_row`, `test_memory_that_never_comes_back_after_the_unload_stops_the_run`) still passes through `release`.

- [ ] **Step 5: Commit**

```bash
git add src/sous/engine/base.py src/sous/tune/bench.py tests/test_engine_base.py tests/test_tune_bench.py
git commit -m "refactor(engine,tune): expose the default engine factory and the arm teardown

The suite runner loads the same engine the bench does and must free it the
same way; a second copy of either mapping would drift.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Suite tasks: the package, the loader and the first task

**Files:**
- Create: `src/sous/tune/suite/__init__.py`
- Create: `src/sous/tune/suite/tasks/implement_rpn/task.toml`, `project/rpn.py`, `grade/test_rpn.py`, `solution/rpn.py`
- Modify: `pyproject.toml` (`[tool.ty.src] exclude`)
- Test: `tests/test_tune_suite.py` (create)

**Interfaces:**
- Produces:
  - `SuiteTask` (frozen dataclass): `name: str`, `path: Path`, `title: str`, `category: str`, `instructions: str`, `context_files: tuple[str, ...]`, `verify_commands: tuple[str, ...]`, `max_turns: int`, `max_minutes: int`; properties `project = path / "project"`, `grade_dir = path / "grade"`, `solution = path / "solution"`.
  - `load_tasks(root: Path | None = None) -> list[SuiteTask]`: every directory under `root` (default: the package's `tasks/`) holding a `task.toml`, sorted by name; `load_task(path: Path) -> SuiteTask` validates one. `ValueError` names the task and the problem for: a missing or empty `title`/`category`/`instructions`; `context_files`/`verify_commands` not lists of non-empty strings; `max_turns`/`max_minutes` not positive ints; an unknown key; a missing `project/`, `grade/` or `solution/`; a context file absent from `project/`; a `grade/` with neither `grade.py` nor a `test_*.py`.
  - `CATEGORIES = ("implement-from-spec", "test-scaffolding", "mechanical-sweep", "cross-file-refactor", "bug-fix", "codegen", "feature-slice")` — `category` must be one of them.
  - Defaults: `max_turns = 16`, `max_minutes = 10`, `context_files = []`, `verify_commands = []`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tune_suite.py
"""The suite's tasks: what ships, what a task.toml must say, and that the
fixture trees resolve from the installed package."""

from importlib.resources import files
from pathlib import Path

import pytest

from sous.tune.suite import CATEGORIES, SuiteTask, load_task, load_tasks

_MINIMAL = 'title = "t"\ncategory = "bug-fix"\ninstructions = "do it"\n'


def _task_dir(tmp_path, toml=_MINIMAL, *, project=("a.py",), grade=("test_a.py",), solution=True):
    d = tmp_path / "task_x"
    d.mkdir()
    (d / "task.toml").write_text(toml)
    (d / "project").mkdir()
    for name in project:
        (d / "project" / name).write_text("x = 1\n")
    (d / "grade").mkdir()
    for name in grade:
        (d / "grade" / name).write_text("import unittest\n")
    if solution:
        (d / "solution").mkdir()
    return d


def test_the_shipped_tasks_load_from_the_package():
    tasks = load_tasks()
    assert [t.name for t in tasks] == sorted(t.name for t in tasks)
    assert "implement_rpn" in [t.name for t in tasks]
    for t in tasks:
        assert isinstance(t, SuiteTask)
        assert t.category in CATEGORIES
        assert t.project.is_dir() and t.grade_dir.is_dir() and t.solution.is_dir()
        for cf in t.context_files:
            assert (t.project / cf).is_file(), (t.name, cf)


def test_the_fixture_tree_is_package_data():
    root = Path(str(files("sous.tune.suite").joinpath("tasks")))
    assert (root / "implement_rpn" / "task.toml").is_file()


def test_defaults_and_derived_paths(tmp_path):
    t = load_task(_task_dir(tmp_path))
    assert t.name == "task_x" and t.title == "t" and t.category == "bug-fix"
    assert t.context_files == () and t.verify_commands == ()
    assert t.max_turns == 16 and t.max_minutes == 10
    assert t.project == t.path / "project" and t.grade_dir == t.path / "grade"
    assert t.solution == t.path / "solution"


@pytest.mark.parametrize(
    "toml, problem",
    [
        ('category = "bug-fix"\ninstructions = "x"\n', "title"),
        ('title = "t"\ncategory = "bug-fix"\ninstructions = ""\n', "instructions"),
        ('title = "t"\ncategory = "nope"\ninstructions = "x"\n', "category"),
        (_MINIMAL + "context_files = [1]\n", "context_files"),
        (_MINIMAL + 'verify_commands = ""\n', "verify_commands"),
        (_MINIMAL + "max_turns = 0\n", "max_turns"),
        (_MINIMAL + "max_minutes = true\n", "max_minutes"),
        (_MINIMAL + 'extra = "?"\n', "extra"),
        (_MINIMAL + 'context_files = ["missing.py"]\n', "missing.py"),
    ],
)
def test_a_bad_task_toml_is_a_valueerror_naming_the_problem(tmp_path, toml, problem):
    with pytest.raises(ValueError, match=problem) as e:
        load_task(_task_dir(tmp_path, toml))
    assert "task_x" in str(e.value)


def test_a_task_needs_its_three_directories_and_a_grader(tmp_path):
    with pytest.raises(ValueError, match="solution"):
        load_task(_task_dir(tmp_path, solution=False))
    with pytest.raises(ValueError, match="grade.py or a test_"):
        load_task(_task_dir(tmp_path, grade=("notes.txt",)))
    d = _task_dir(tmp_path, grade=("grade.py",))
    assert load_task(d).grade_dir == d / "grade"


def test_load_tasks_skips_directories_without_a_task_toml(tmp_path):
    _task_dir(tmp_path)
    (tmp_path / "junk").mkdir()
    assert [t.name for t in load_tasks(tmp_path)] == ["task_x"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_suite.py -v`
Expected: FAIL — `ModuleNotFoundError: sous.tune.suite`.

- [ ] **Step 3: Create the package and the loader**

```python
# src/sous/tune/suite/__init__.py
"""The graded suite: mechanical coding tasks the worker runs for real, each
with a hidden grader and a reference solution. A task is a directory —
task.toml, project/ (what the worker sees), grade/ (what scores it),
solution/ (the solved project, so CI can prove the grader)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

CATEGORIES = (
    "implement-from-spec",
    "test-scaffolding",
    "mechanical-sweep",
    "cross-file-refactor",
    "bug-fix",
    "codegen",
    "feature-slice",
)
DEFAULT_MAX_TURNS = 16
DEFAULT_MAX_MINUTES = 10
_KEYS = {
    "title",
    "category",
    "instructions",
    "context_files",
    "verify_commands",
    "max_turns",
    "max_minutes",
}


@dataclass(frozen=True)
class SuiteTask:
    name: str
    path: Path
    title: str
    category: str
    instructions: str
    context_files: tuple[str, ...]
    verify_commands: tuple[str, ...]
    max_turns: int
    max_minutes: int

    @property
    def project(self) -> Path:
        return self.path / "project"

    @property
    def grade_dir(self) -> Path:
        return self.path / "grade"

    @property
    def solution(self) -> Path:
        return self.path / "solution"


def tasks_root() -> Path:
    return Path(str(files("sous.tune.suite").joinpath("tasks")))


def _text(raw: dict, key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"suite task {name}: {key} must be a non-empty string")
    return value


def _strings(raw: dict, key: str, name: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"suite task {name}: {key} must be a list of non-empty strings")
    return tuple(value)


def _positive(raw: dict, key: str, default: int, name: str) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"suite task {name}: {key} must be a positive integer")
    return value


def load_task(path: Path) -> SuiteTask:
    name = path.name
    raw = tomllib.loads((path / "task.toml").read_text())
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"suite task {name}: unknown key(s) {', '.join(sorted(unknown))}")
    category = _text(raw, "category", name)
    if category not in CATEGORIES:
        raise ValueError(f"suite task {name}: category {category!r} is not one of {CATEGORIES}")
    task = SuiteTask(
        name=name,
        path=path,
        title=_text(raw, "title", name),
        category=category,
        instructions=_text(raw, "instructions", name),
        context_files=_strings(raw, "context_files", name),
        verify_commands=_strings(raw, "verify_commands", name),
        max_turns=_positive(raw, "max_turns", DEFAULT_MAX_TURNS, name),
        max_minutes=_positive(raw, "max_minutes", DEFAULT_MAX_MINUTES, name),
    )
    for sub in ("project", "grade", "solution"):
        if not (path / sub).is_dir():
            raise ValueError(f"suite task {name}: no {sub}/ directory")
    for cf in task.context_files:
        if not (task.project / cf).is_file():
            raise ValueError(f"suite task {name}: context file {cf} is not in project/")
    if not (task.grade_dir / "grade.py").is_file() and not any(task.grade_dir.glob("test_*.py")):
        raise ValueError(f"suite task {name}: grade/ needs a grade.py or a test_*.py module")
    return task


def load_tasks(root: Path | None = None) -> list[SuiteTask]:
    base = root if root is not None else tasks_root()
    return [load_task(p) for p in sorted(base.iterdir()) if (p / "task.toml").is_file()]
```

- [ ] **Step 4: Create the first task**

`src/sous/tune/suite/tasks/implement_rpn/task.toml`:

```toml
title = "Implement the RPN calculator"
category = "implement-from-spec"
instructions = """rpn.py documents two functions in its module docstring and leaves both \
unimplemented (they raise NotImplementedError). Implement tokenize and evaluate exactly as \
the docstring describes, keeping the docstring and both signatures as they are. Do not add \
any other file."""
context_files = ["rpn.py"]
max_turns = 12
```

`src/sous/tune/suite/tasks/implement_rpn/project/rpn.py`:

```python
"""Reverse Polish notation over integers.

tokenize(text) splits the text on whitespace and returns the tokens as a
list of strings; empty or blank text gives an empty list.

evaluate(tokens) evaluates the tokens as a reverse Polish expression: a
numeric token (an int literal, possibly negative) is pushed on a stack; an
operator token ("+", "-", "*", "/") pops the right operand, then the left
one, applies the operator and pushes the result. "/" is floor division (the
// operator). It returns the single value left on the stack. It raises
ValueError for a token that is neither a number nor an operator, for an
operator with fewer than two values on the stack, and when anything but
exactly one value is left at the end.
"""


def tokenize(text: str) -> list[str]:
    raise NotImplementedError


def evaluate(tokens: list[str]) -> int:
    raise NotImplementedError
```

`src/sous/tune/suite/tasks/implement_rpn/grade/test_rpn.py`:

```python
import unittest


class TokenizeTests(unittest.TestCase):
    def test_splits_on_whitespace_and_blank_is_empty(self):
        from rpn import tokenize

        self.assertEqual(tokenize("3  4 +\n"), ["3", "4", "+"])
        self.assertEqual(tokenize("   "), [])


class EvaluateTests(unittest.TestCase):
    def test_addition(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["3", "4", "+"]), 7)

    def test_nested_expression(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["5", "1", "2", "+", "4", "*", "+", "3", "-"]), 14)

    def test_operand_order_for_subtraction_and_division(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["10", "3", "-"]), 7)
        self.assertEqual(evaluate(["7", "2", "/"]), 3)

    def test_floor_division_of_a_negative(self):
        from rpn import evaluate

        self.assertEqual(evaluate(["-7", "2", "/"]), -4)

    def test_errors(self):
        from rpn import evaluate

        with self.assertRaises(ValueError):
            evaluate(["1", "x", "+"])
        with self.assertRaises(ValueError):
            evaluate(["+"])
        with self.assertRaises(ValueError):
            evaluate(["1", "2"])
```

`src/sous/tune/suite/tasks/implement_rpn/solution/rpn.py`: the project file's docstring verbatim, then:

```python
_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b,
}


def tokenize(text: str) -> list[str]:
    return text.split()


def evaluate(tokens: list[str]) -> int:
    stack: list[int] = []
    for token in tokens:
        if token in _OPERATORS:
            if len(stack) < 2:
                raise ValueError(f"operator {token!r} needs two operands")
            right, left = stack.pop(), stack.pop()
            stack.append(_OPERATORS[token](left, right))
        else:
            try:
                stack.append(int(token))
            except ValueError:
                raise ValueError(f"unknown token {token!r}") from None
    if len(stack) != 1:
        raise ValueError(f"{len(stack)} values left on the stack")
    return stack[0]
```

Append to `pyproject.toml`, after `[tool.ty.environment]`:

```toml
[tool.ty.src]
# The suite's fixture projects import each other by bare name (`from rpn
# import evaluate`) and resolve only inside a copied project; the shipped
# graders prove them in CI instead.
exclude = ["src/sous/tune/suite/tasks"]
```

- [ ] **Step 5: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_suite.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: PASS. If `ruff format --check` flags a fixture, run `uv run ruff format src/sous/tune/suite` — fixtures are formatted like everything else.

- [ ] **Step 6: Commit**

```bash
git add src/sous/tune/suite pyproject.toml tests/test_tune_suite.py
git commit -m "feat(tune): the suite package, its task loader and the first task

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Grading: the unittest subprocess and the grade contract

**Files:**
- Create: `src/sous/tune/suite/unittests.py`, `src/sous/tune/suite/grading.py`
- Test: `tests/test_tune_suite.py` (append)

**Interfaces:**
- Produces:
  - `python -m sous.tune.suite.unittests DIR`: discovers `test_*.py` under `DIR` (top-level dir `DIR`), runs them with the process's working directory as the importable project, prints one JSON line `{"passed": int, "total": int, "failed": [str, ...]}` to stdout, exits 0 (a failing test is a result, not an exit code). `total` is `testsRun` — a module that fails to import is one erroring test.
  - `grading.Grade` (frozen dataclass): `score: float` in `[0, 1]`, `detail: str`.
  - `grading.run_tests(cwd: Path, tests_dir: Path, *, python: Path, timeout: float) -> tuple[int, int, str]`: runs the module above in a subprocess (`[python, "-m", "sous.tune.suite.unittests", tests_dir]`, `cwd=cwd`, the current environment) and returns `(passed, total, detail)`; a timeout, a crash or unparsable output is `(0, 0, "<what happened>")`.
  - `grading.grade_task(task: SuiteTask, project: Path, *, python: Path = Path(sys.executable), timeout: float = 120.0) -> Grade`: with `grade/grade.py` present, loads it (`importlib.util.spec_from_file_location`) and calls `grade(project, tests)` where `tests(cwd, tests_dir=None)` is `run_tests` bound to `python`/`timeout` with `task.grade_dir` as the default `tests_dir`; without one, `passed / total` over the hidden modules (`0.0` when `total == 0`). The score is clamped to `[0, 1]`; a grader that raises is `Grade(0.0, "grader failed: <type>: <msg>")`.
  - The `grade.py` contract, documented in `docs/tuning.md` in Task 11: `def grade(project: Path, tests) -> tuple[float, str]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tune_suite.py`)

```python
import json
import subprocess
import sys

from sous.tune.suite.grading import Grade, grade_task, run_tests


def _project(tmp_path, source="def add(a, b):\n    return a + b\n"):
    p = tmp_path / "proj"
    p.mkdir(parents=True, exist_ok=True)
    (p / "calc.py").write_text(source)
    return p


def _hidden(tmp_path, body=None):
    h = tmp_path / "hidden"
    h.mkdir(exist_ok=True)
    (h / "test_calc.py").write_text(
        body
        or "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_add(self):\n        from calc import add\n\n        self.assertEqual(add(1, 2), 3)\n\n"
        "    def test_add_negative(self):\n        from calc import add\n\n"
        "        self.assertEqual(add(-1, 1), 0)\n"
    )
    return h


def test_the_unittest_runner_prints_counts_as_json(tmp_path):
    proj, hidden = _project(tmp_path), _hidden(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "sous.tune.suite.unittests", str(hidden)],
        cwd=proj,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {
        "passed": 2,
        "total": 2,
        "failed": [],
    }


def test_run_tests_counts_a_failing_test_and_names_it(tmp_path):
    # abs(a) + b: right for (1, 2), wrong for (-1, 1) — one of the two fails.
    proj = _project(tmp_path, "def add(a, b):\n    return abs(a) + b\n")
    hidden = _hidden(tmp_path)
    passed, total, detail = run_tests(proj, hidden, python=Path(sys.executable), timeout=60)
    assert (passed, total) == (1, 2)
    assert "test_add_negative" in detail and "1/2 hidden tests passed" in detail


def test_a_module_that_cannot_import_is_one_erroring_test(tmp_path):
    proj = _project(tmp_path)
    hidden = _hidden(tmp_path, "from nothing import nobody\n")
    assert run_tests(proj, hidden, python=Path(sys.executable), timeout=60)[:2] == (0, 1)


def test_a_hung_test_is_a_timeout_not_a_hang(tmp_path):
    proj = _project(tmp_path)
    hidden = _hidden(
        tmp_path,
        "import time\nimport unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_spin(self):\n        time.sleep(60)\n",
    )
    passed, total, detail = run_tests(proj, hidden, python=Path(sys.executable), timeout=2)
    assert (passed, total) == (0, 0)
    assert "timed out" in detail


def _suite_task(tmp_path, grade_py=None):
    d = tmp_path / "task_calc"
    d.mkdir()
    (d / "task.toml").write_text('title = "t"\ncategory = "bug-fix"\ninstructions = "x"\n')
    (d / "project").mkdir()
    (d / "solution").mkdir()
    grade = d / "grade"
    grade.mkdir()
    if grade_py is None:
        (grade / "test_calc.py").write_text((_hidden(tmp_path) / "test_calc.py").read_text())
    else:
        (grade / "grade.py").write_text(grade_py)
    return load_task(d)


def test_grade_task_without_a_script_is_the_hidden_tests_pass_fraction(tmp_path):
    task = _suite_task(tmp_path)
    good = grade_task(task, _project(tmp_path), timeout=60)
    assert good == Grade(1.0, good.detail) and "2/2" in good.detail
    bad = grade_task(task, _project(tmp_path / "b", "def add(a, b):\n    return abs(a) + b\n"), timeout=60)
    assert bad.score == 0.5


def test_grade_task_with_a_script_hands_it_the_project_and_the_test_runner(tmp_path):
    script = (
        "from pathlib import Path\n\n\n"
        "def grade(project: Path, tests) -> tuple[float, str]:\n"
        "    assert (project / 'calc.py').is_file()\n"
        "    passed, total, _ = tests(project, project.parent / 'own')\n"
        "    return 2.5, f'{passed}/{total} own tests'\n"
    )
    task = _suite_task(tmp_path, script)
    proj = _project(tmp_path)
    own = tmp_path / "own"
    own.mkdir()
    (own / "test_own.py").write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
    )
    g = grade_task(task, proj, timeout=60)
    assert g == Grade(1.0, "1/1 own tests")  # clamped to [0, 1]


def test_a_grader_that_raises_is_a_zero_with_the_error(tmp_path):
    task = _suite_task(tmp_path, "def grade(project, tests):\n    raise KeyError('boom')\n")
    g = grade_task(task, _project(tmp_path), timeout=60)
    assert g.score == 0.0 and "grader failed: KeyError" in g.detail
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_suite.py -v`
Expected: FAIL — `ModuleNotFoundError: sous.tune.suite.grading`.

- [ ] **Step 3: Implement**

```python
# src/sous/tune/suite/unittests.py
"""`python -m sous.tune.suite.unittests DIR`: run the unittest modules under
DIR and print the counts as one JSON line. The grader runs this in a
subprocess with the worker's project as the working directory — `python -m`
puts that directory first on sys.path, so the hidden tests import the
worker's modules by name — and a project that hangs or crashes takes the
subprocess with it, never the tune."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def run(tests_dir: Path) -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(tests_dir))
    result = unittest.TestResult()
    suite.run(result)
    failed = [str(case) for case, _ in [*result.failures, *result.errors]]
    total = result.testsRun
    return {"passed": total - len(failed), "total": total, "failed": failed}


def main(argv: list[str]) -> int:
    print(json.dumps(run(Path(argv[1]).resolve())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

```python
# src/sous/tune/suite/grading.py
"""How a suite run is scored: hidden unittest modules run against the
worker's project in a subprocess, or a task's own grade.py over that same
runner for the categories a pass count cannot express."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sous.tune.suite import SuiteTask

TestRunner = Callable[..., tuple[int, int, str]]


@dataclass(frozen=True)
class Grade:
    score: float
    detail: str


def run_tests(cwd: Path, tests_dir: Path, *, python: Path, timeout: float) -> tuple[int, int, str]:
    """(passed, total, detail) of the test_*.py modules under `tests_dir`,
    run with `cwd` importable. A timeout, a crash or output that is not the
    runner's JSON is (0, 0, why): the worker's code is untrusted and a
    project that hangs must not hang the tune."""
    argv = [str(python), "-m", "sous.tune.suite.unittests", str(tests_dir)]
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return 0, 0, f"tests timed out after {timeout:.0f} s"
    except OSError as e:
        return 0, 0, f"could not run the tests: {e}"
    lines = proc.stdout.strip().splitlines()
    try:
        counts = json.loads(lines[-1]) if lines else {}
        passed, total = int(counts["passed"]), int(counts["total"])
        failed = [str(f) for f in counts.get("failed", [])]
    except (ValueError, KeyError, TypeError, IndexError):
        tail = proc.stderr.strip().splitlines()[-1:] or ["no output"]
        return 0, 0, f"test runner exited {proc.returncode}: {tail[0]}"
    detail = f"{passed}/{total} hidden tests passed"
    if failed:
        detail += "; failed: " + ", ".join(failed)
    return passed, total, detail


def _load_grader(script: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"sous_tune_suite_grade_{name}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade


def grade_task(
    task: SuiteTask,
    project: Path,
    *,
    python: Path = Path(sys.executable),
    timeout: float = 120.0,
) -> Grade:
    """The task's score over the worker's finished project, in [0, 1]. With
    a grade.py the task decides — it gets the project and a test runner
    whose default test directory is the hidden one — else the hidden
    modules' pass fraction. A grader that raises scores zero and says so:
    the run is a result either way, and the suite goes on."""

    def tests(cwd: Path, tests_dir: Path | None = None) -> tuple[int, int, str]:
        return run_tests(
            cwd, tests_dir if tests_dir is not None else task.grade_dir, python=python, timeout=timeout
        )

    script = task.grade_dir / "grade.py"
    try:
        if script.is_file():
            score, detail = _load_grader(script, task.name)(project, tests)
        else:
            passed, total, detail = tests(project)
            score = passed / total if total else 0.0
    except Exception as e:  # noqa: BLE001 — a grader bug is a zero, named, not a dead run
        return Grade(0.0, f"grader failed: {type(e).__name__}: {e}")
    return Grade(min(1.0, max(0.0, float(score))), str(detail))
```

- [ ] **Step 4: Add the CI proof of every shipped grader** (append to `tests/test_tune_suite.py`)

```python
import shutil


@pytest.mark.parametrize("task", load_tasks(), ids=lambda t: t.name)
def test_every_shipped_grader_scores_the_solution_one_and_the_fixture_zero(task, tmp_path):
    solved = tmp_path / "solved"
    shutil.copytree(task.solution, solved)
    untouched = tmp_path / "untouched"
    shutil.copytree(task.project, untouched)
    good = grade_task(task, solved, timeout=120)
    bad = grade_task(task, untouched, timeout=120)
    assert good.score == 1.0, (task.name, good.detail)
    assert bad.score == 0.0, (task.name, bad.detail)
```

- [ ] **Step 5: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_suite.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS, including `implement_rpn` in the parametrized proof.

- [ ] **Step 6: Commit**

```bash
git add src/sous/tune/suite/grading.py src/sous/tune/suite/unittests.py tests/test_tune_suite.py
git commit -m "feat(tune): grade a suite run with hidden tests in a subprocess or a task's own grader

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Three more tasks: test scaffolding, the docstring sweep, the rename

**Files:**
- Create: `src/sous/tune/suite/tasks/tests_for_slugify/…`, `src/sous/tune/suite/tasks/docstring_sweep/…`, `src/sous/tune/suite/tasks/rename_symbol/…`
- Test: the parametrized proof in `tests/test_tune_suite.py` (no new test code; it discovers the tasks)

**Interfaces:**
- Consumes: the `grade.py` contract from Task 5: `grade(project: Path, tests) -> tuple[float, str]`, `tests(cwd, tests_dir=None) -> (passed, total, detail)`.
- Produces: three task directories the loader accepts and the proof scores 1.0 / 0.0.

Design rule for every task in this PR: **each hidden test fails on the untouched fixture**, so the proof's `0.0` holds without special cases, and a partial result scores a fraction.

- [ ] **Step 1: Run the proof to see the three tasks are absent**

Run: `uv run pytest tests/test_tune_suite.py -k every_shipped_grader -v`
Expected: PASS for `implement_rpn` only (one parametrized case).

- [ ] **Step 2: Create `tests_for_slugify`** (category `test-scaffolding`; graded by the worker's own tests on the pristine module and on four mutants)

`task.toml`:

```toml
title = "Write tests for slugify"
category = "test-scaffolding"
instructions = """slugify.py has no tests. Write tests/test_slugify.py, a unittest module with one \
test per behaviour the module docstring documents: lower-casing, a run of characters that are \
not letters or digits becoming one dash, leading and trailing dashes being removed, the cut at \
max_length, and a dash left at the end by the cut being removed. Use assertEqual on the exact \
slug. Run `python -m unittest discover -s tests` and make sure every test passes before you \
finish. Do not change slugify.py."""
context_files = ["slugify.py"]
verify_commands = ["python -m unittest discover -s tests"]
```

`project/slugify.py`:

```python
"""slugify(text, max_length=40) turns a title into a URL slug:

- letters are lower-cased;
- every run of characters that are not ASCII letters or digits becomes one
  "-";
- leading and trailing "-" are removed;
- the result is cut to at most max_length characters, and a "-" the cut
  leaves at the end is removed too.
"""

import re

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 40) -> str:
    slug = _NON_WORD.sub("-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-")
```

`project/tests/__init__.py`: empty file (the directory has to exist in git for the worker to find it listed).

`grade/grade.py`:

```python
"""Score = 1 if the worker's tests pass on the pristine module, times the
share of four mutants — one per documented behaviour — those tests catch.
No tests, or tests that fail on the real module, score zero."""

import shutil
import tempfile
from pathlib import Path

_HEAD = 'import re\n\n_NON_WORD = re.compile(r"[^a-z0-9]+")\n\n\n'
MUTANTS = {
    "no lower-casing": _HEAD + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text).strip("-")\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "leading and trailing dashes kept": _HEAD + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text.lower())\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "a run becomes several dashes": (
        'import re\n\n_NON_WORD = re.compile(r"[^a-z0-9]")\n\n\n'
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text.lower()).strip("-")\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "max_length ignored": _HEAD + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    return _NON_WORD.sub("-", text.lower()).strip("-")\n'
    ),
}


def _fails(tests, project: Path) -> bool:
    passed, total, _ = tests(project, project / "tests")
    return total == 0 or passed < total


def grade(project: Path, tests) -> tuple[float, str]:
    passed, total, detail = tests(project, project / "tests")
    if total == 0 or passed < total:
        return 0.0, f"the worker's tests on the pristine module: {detail}"
    caught = []
    for name, source in MUTANTS.items():
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "project"
            shutil.copytree(project, copy)
            (copy / "slugify.py").write_text(source)
            if _fails(tests, copy):
                caught.append(name)
    return (
        len(caught) / len(MUTANTS),
        f"{total} tests pass on the pristine module; caught {len(caught)}/{len(MUTANTS)} "
        f"mutants ({', '.join(caught) or 'none'})",
    )
```

`solution/slugify.py`: identical to `project/slugify.py`. `solution/tests/__init__.py`: empty. `solution/tests/test_slugify.py`:

```python
import unittest

from slugify import slugify


class SlugifyTests(unittest.TestCase):
    def test_letters_are_lower_cased(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_a_run_of_non_alphanumerics_becomes_one_dash(self):
        self.assertEqual(slugify("a  --  b!!c"), "a-b-c")

    def test_leading_and_trailing_dashes_are_removed(self):
        self.assertEqual(slugify("  hello! "), "hello")

    def test_the_slug_is_cut_to_max_length(self):
        self.assertEqual(slugify("abcdefghij", max_length=5), "abcde")

    def test_a_dash_left_by_the_cut_is_removed(self):
        self.assertEqual(slugify("abc def", max_length=4), "abc")
```

- [ ] **Step 3: Create `docstring_sweep`** (category `mechanical-sweep`; AST check over the four undocumented public functions, times the hidden behaviour tests)

`task.toml`:

```toml
title = "Docstring sweep over shapes.py"
category = "mechanical-sweep"
instructions = """Add a one-line docstring to every public function in shapes.py that has none. \
Leave the functions that already have a docstring, the private helper, every signature and \
every line of behaviour exactly as they are. Do not add or remove any other file."""
context_files = ["shapes.py"]
```

`project/shapes.py`:

```python
"""Areas and perimeters of simple shapes."""

import math


def _check_positive(*values: float) -> None:
    for value in values:
        if value <= 0:
            raise ValueError("dimensions must be positive")


def area_rect(width: float, height: float) -> float:
    """Area of a rectangle."""
    _check_positive(width, height)
    return width * height


def perimeter_rect(width: float, height: float) -> float:
    _check_positive(width, height)
    return 2 * (width + height)


def area_circle(radius: float) -> float:
    """Area of a circle."""
    _check_positive(radius)
    return math.pi * radius * radius


def perimeter_circle(radius: float) -> float:
    _check_positive(radius)
    return 2 * math.pi * radius


def area_triangle(base: float, height: float) -> float:
    _check_positive(base, height)
    return base * height / 2


def scale(value: float, factor: float) -> float:
    _check_positive(value, factor)
    return value * factor
```

`grade/grade.py`:

```python
"""Score = the share of the originally undocumented public functions that
now carry a docstring, provided the hidden behaviour tests still all pass;
a behaviour change scores zero whatever was documented."""

import ast
from pathlib import Path

UNDOCUMENTED = ("perimeter_rect", "perimeter_circle", "area_triangle", "scale")


def grade(project: Path, tests) -> tuple[float, str]:
    try:
        tree = ast.parse((project / "shapes.py").read_text())
    except (OSError, SyntaxError) as e:
        return 0.0, f"shapes.py unreadable: {e}"
    documented = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
    }
    done = [name for name in UNDOCUMENTED if name in documented]
    passed, total, detail = tests(project)
    if total == 0 or passed < total:
        return 0.0, f"behaviour changed: {detail}"
    return len(done) / len(UNDOCUMENTED), f"{len(done)}/{len(UNDOCUMENTED)} documented; {detail}"
```

`grade/test_shapes.py`:

```python
import math
import unittest


class BehaviourTests(unittest.TestCase):
    def test_rectangle(self):
        from shapes import area_rect, perimeter_rect

        self.assertEqual(area_rect(2, 3), 6)
        self.assertEqual(perimeter_rect(2, 3), 10)

    def test_circle(self):
        from shapes import area_circle, perimeter_circle

        self.assertAlmostEqual(area_circle(1), math.pi)
        self.assertAlmostEqual(perimeter_circle(1), 2 * math.pi)

    def test_triangle_and_scale(self):
        from shapes import area_triangle, scale

        self.assertEqual(area_triangle(4, 3), 6)
        self.assertEqual(scale(2, 1.5), 3)

    def test_non_positive_dimensions_are_refused(self):
        from shapes import area_rect, scale

        with self.assertRaises(ValueError):
            area_rect(0, 1)
        with self.assertRaises(ValueError):
            scale(1, -1)
```

`solution/shapes.py`: the project file with these one-line docstrings added, nothing else changed: `perimeter_rect` → `"""Perimeter of a rectangle."""`, `perimeter_circle` → `"""Perimeter (circumference) of a circle."""`, `area_triangle` → `"""Area of a triangle from its base and height."""`, `scale` → `"""Scale a positive value by a positive factor."""`.

- [ ] **Step 4: Create `rename_symbol`** (category `cross-file-refactor`; hidden tests import the new name and prove the old one is gone from every file)

`task.toml`:

```toml
title = "Rename ItemStore to InventoryStore"
category = "cross-file-refactor"
instructions = """Rename the class ItemStore to InventoryStore everywhere in this project: its \
definition in store.py, and every import and use in report.py, cli.py and tests/test_store.py. \
Change nothing else — no behaviour, no other names. Run `python -m unittest discover -s tests` \
and make sure it passes before you finish."""
context_files = ["store.py"]
verify_commands = ["python -m unittest discover -s tests"]
```

`project/store.py`:

```python
"""An in-memory count of named items."""


class ItemStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, name: str, quantity: int = 1) -> None:
        self._counts[name] = self._counts.get(name, 0) + quantity

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    def names(self) -> list[str]:
        return sorted(self._counts)
```

`project/report.py`:

```python
from store import ItemStore


def summarize(store: ItemStore) -> str:
    parts = [f"{name} x{store.count(name)}" for name in store.names()]
    return f"{len(parts)} items: {', '.join(parts)}"
```

`project/cli.py`:

```python
"""python cli.py NAME[:QUANTITY] ... prints a summary of the items given."""

import sys

from report import summarize
from store import ItemStore


def main(argv: list[str]) -> int:
    store = ItemStore()
    for arg in argv:
        name, _, quantity = arg.partition(":")
        store.add(name, int(quantity) if quantity else 1)
    print(summarize(store))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

`project/tests/test_store.py`:

```python
import unittest

from report import summarize
from store import ItemStore


class StoreTests(unittest.TestCase):
    def test_counts_add_up(self):
        store = ItemStore()
        store.add("bolt", 3)
        store.add("bolt")
        self.assertEqual(store.count("bolt"), 4)

    def test_summary_lists_items_in_name_order(self):
        store = ItemStore()
        store.add("nut")
        store.add("bolt", 3)
        self.assertEqual(summarize(store), "2 items: bolt x3, nut x1")
```

`grade/test_rename.py`:

```python
import unittest
from pathlib import Path


class RenameTests(unittest.TestCase):
    def test_the_new_name_is_the_store(self):
        from store import InventoryStore

        store = InventoryStore()
        store.add("bolt", 3)
        self.assertEqual(store.count("bolt"), 3)

    def test_the_report_takes_the_new_name(self):
        from report import summarize
        from store import InventoryStore

        store = InventoryStore()
        store.add("nut")
        store.add("bolt", 3)
        self.assertEqual(summarize(store), "2 items: bolt x3, nut x1")

    def test_the_old_name_is_gone_from_every_file(self):
        import store

        self.assertFalse(hasattr(store, "ItemStore"))
        for path in sorted(Path.cwd().rglob("*.py")):
            self.assertNotIn("ItemStore", path.read_text(), path.name)

    def test_the_cli_and_the_tests_use_the_new_name(self):
        for name in ("cli.py", "tests/test_store.py"):
            self.assertIn("InventoryStore", Path(name).read_text(), name)
```

`solution/`: the four project files with `ItemStore` replaced by `InventoryStore` in every occurrence (`store.py` class name; the import and the annotation in `report.py`; the import and the constructor call in `cli.py`; the import and both constructor calls in `tests/test_store.py`).

- [ ] **Step 5: Run the proof, the loader tests and the four CI checks**

Run: `uv run pytest tests/test_tune_suite.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS with four parametrized proof cases (`implement_rpn`, `tests_for_slugify`, `docstring_sweep`, `rename_symbol`). If `ruff format --check` flags a fixture, format it (`uv run ruff format src/sous/tune/suite`) — a solution and its project file must stay byte-identical where the task says "unchanged", so format both.

- [ ] **Step 6: Commit**

```bash
git add src/sous/tune/suite/tasks
git commit -m "feat(tune): suite tasks for test scaffolding, a docstring sweep and a cross-file rename

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: The last four tasks: the bug fix, the dataclass, the CLI flag, the config migration

**Files:**
- Create: `src/sous/tune/suite/tasks/fix_intervals/…`, `src/sous/tune/suite/tasks/dataclass_from_schema/…`, `src/sous/tune/suite/tasks/cli_flag/…`, `src/sous/tune/suite/tasks/migrate_config/…`
- Test: the parametrized proof in `tests/test_tune_suite.py`

**Interfaces:**
- Consumes: the hidden-tests-only path of `grade_task` (no `grade.py` in any of these four).
- Produces: four task directories; with Tasks 4 and 6 the suite is the spec's eight.

- [ ] **Step 1: Create `fix_intervals`** (category `bug-fix`)

`task.toml`:

```toml
title = "Fix the failing interval test"
category = "bug-fix"
instructions = """Run `python -m unittest discover -s tests`: one test in tests/test_intervals.py \
fails. Fix the bug in intervals.py so that every test passes. The tests are correct — do not \
modify them, and do not change what merge returns for the cases that already pass."""
context_files = ["intervals.py", "tests/test_intervals.py"]
verify_commands = ["python -m unittest discover -s tests"]
max_turns = 12
```

`project/intervals.py` (the bug is the `<` that should be `<=`; touching intervals are not merged):

```python
"""merge(intervals) merges a list of closed integer intervals [start, end]:
the result is sorted by start, and any two intervals that overlap or touch
(one's start equal to the other's end) become one interval."""


def merge(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

`project/tests/test_intervals.py`:

```python
import unittest

from intervals import merge


class MergeTests(unittest.TestCase):
    def test_disjoint_intervals_stay_apart(self):
        self.assertEqual(merge([[1, 2], [4, 5]]), [[1, 2], [4, 5]])

    def test_overlapping_intervals_merge(self):
        self.assertEqual(merge([[1, 4], [2, 6]]), [[1, 6]])

    def test_touching_intervals_merge(self):
        self.assertEqual(merge([[1, 3], [3, 5]]), [[1, 5]])
```

`grade/test_fix.py` (every test has a touching case, so all fail before the fix; the disjoint and overlapping cases guard the behaviour that already worked):

```python
import unittest


class FixTests(unittest.TestCase):
    def test_a_touching_pair_merges(self):
        from intervals import merge

        self.assertEqual(merge([[1, 3], [3, 5]]), [[1, 5]])
        self.assertEqual(merge([[1, 2], [4, 5]]), [[1, 2], [4, 5]])

    def test_a_touching_chain_merges_whatever_the_input_order(self):
        from intervals import merge

        self.assertEqual(merge([[5, 7], [1, 3], [3, 5]]), [[1, 7]])
        self.assertEqual(merge([[1, 4], [2, 6]]), [[1, 6]])

    def test_touching_beside_disjoint(self):
        from intervals import merge

        self.assertEqual(merge([[1, 2], [2, 3], [5, 6]]), [[1, 3], [5, 6]])
```

`solution/intervals.py`: the project file with `start <= merged[-1][1]`. `solution/tests/test_intervals.py`: identical to the project's.

- [ ] **Step 2: Create `dataclass_from_schema`** (category `codegen`)

`task.toml`:

```toml
title = "A dataclass from a JSON schema"
category = "codegen"
instructions = """Write record.py: a dataclass named Record generated from schema.json. One field \
per property, in the schema's order, with the matching Python type (integer -> int, string -> str, \
boolean -> bool, an array of strings -> list[str], number -> float). Required properties have no \
default; an optional one defaults to the schema's default when it gives one, else to None. Add a \
classmethod from_dict(data) that raises ValueError when a required key is missing, and a method \
to_dict() returning a plain dict that from_dict accepts back unchanged. Standard library only."""
context_files = ["schema.json"]
```

`project/schema.json`:

```json
{
  "title": "Record",
  "type": "object",
  "required": ["id", "name", "active"],
  "properties": {
    "id": {"type": "integer"},
    "name": {"type": "string"},
    "active": {"type": "boolean"},
    "tags": {"type": "array", "items": {"type": "string"}, "default": []},
    "score": {"type": "number"}
  }
}
```

`grade/test_record.py`:

```python
import dataclasses
import unittest

FULL = {"id": 1, "name": "a", "active": True, "tags": ["x"], "score": 2.5}


class RecordTests(unittest.TestCase):
    def test_round_trip(self):
        from record import Record

        record = Record.from_dict(FULL)
        self.assertEqual(record.to_dict(), FULL)
        self.assertEqual(Record.from_dict(record.to_dict()), record)

    def test_optional_fields_default(self):
        from record import Record

        record = Record.from_dict({"id": 2, "name": "b", "active": False})
        self.assertEqual(record.tags, [])
        self.assertIsNone(record.score)

    def test_a_missing_required_key_is_a_valueerror(self):
        from record import Record

        with self.assertRaises(ValueError):
            Record.from_dict({"id": 3})

    def test_it_is_a_dataclass_with_the_schema_fields(self):
        from record import Record

        self.assertTrue(dataclasses.is_dataclass(Record))
        self.assertEqual(
            [f.name for f in dataclasses.fields(Record)], ["id", "name", "active", "tags", "score"]
        )
```

`solution/schema.json`: identical to the project's. `solution/record.py`:

```python
import dataclasses
from dataclasses import dataclass, field

_REQUIRED = ("id", "name", "active")


@dataclass
class Record:
    id: int
    name: str
    active: bool
    tags: list[str] = field(default_factory=list)
    score: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise ValueError(f"missing required keys: {', '.join(missing)}")
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            active=bool(data["active"]),
            tags=list(data.get("tags", [])),
            score=None if data.get("score") is None else float(data["score"]),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
```

- [ ] **Step 3: Create `cli_flag`** (category `feature-slice`; hidden tests run the CLI in a subprocess)

`task.toml`:

```toml
title = "Add --lines to wc.py"
category = "feature-slice"
instructions = """Add a --lines flag to wc.py. With it, the tool counts lines (as str.splitlines \
counts them) instead of words; the output format, the per-file lines and the total line stay \
exactly as they are. Give the flag a one-line help text so it shows up in --help. Change nothing \
else."""
context_files = ["wc.py"]
```

`project/wc.py`:

```python
"""python wc.py FILE [FILE ...] prints the word count of each file as
"<count> <file>", and a "<count> total" line when more than one file is
given."""

import argparse
import sys


def count_words(text: str) -> int:
    return len(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="count words in files")
    parser.add_argument("files", nargs="+", help="files to count")
    args = parser.parse_args(argv)
    total = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            count = count_words(f.read())
        total += count
        print(f"{count} {name}")
    if len(args.files) > 1:
        print(f"{total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`grade/test_wc.py`:

```python
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _wc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "wc.py", *args], capture_output=True, text=True, timeout=30, check=False
    )


class LinesFlagTests(unittest.TestCase):
    def test_lines_counts_lines_and_words_stay_the_default(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f.txt"
            f.write_text("a b\nc\n")
            self.assertEqual(_wc("--lines", str(f)).stdout.strip(), f"2 {f}")
            self.assertEqual(_wc(str(f)).stdout.strip(), f"3 {f}")

    def test_two_files_print_a_line_total(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.txt", Path(td) / "b.txt"
            a.write_text("a\nb\nc\n")
            b.write_text("x\n")
            out = _wc("--lines", str(a), str(b))
            self.assertEqual(out.stdout.strip().splitlines(), [f"3 {a}", f"1 {b}", "4 total"])

    def test_help_describes_the_flag(self):
        out = _wc("--help")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--lines", out.stdout)
```

`solution/wc.py`: the project file with a `count_lines` helper and the flag:

```python
"""python wc.py FILE [FILE ...] prints the word count of each file as
"<count> <file>", and a "<count> total" line when more than one file is
given. With --lines it counts lines instead."""

import argparse
import sys


def count_words(text: str) -> int:
    return len(text.split())


def count_lines(text: str) -> int:
    return len(text.splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="count words in files")
    parser.add_argument("files", nargs="+", help="files to count")
    parser.add_argument("--lines", action="store_true", help="count lines instead of words")
    args = parser.parse_args(argv)
    total = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            text = f.read()
        count = count_lines(text) if args.lines else count_words(text)
        total += count
        print(f"{count} {name}")
    if len(args.files) > 1:
        print(f"{total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create `migrate_config`** (category `mechanical-sweep`)

`task.toml`:

```toml
title = "Migrate the INI configs to JSON"
category = "mechanical-sweep"
instructions = """Migrate the configuration files from INI to JSON. For each configs/<name>.ini \
write configs/<name>.json holding the same sections and keys — each section an object, every \
value the string configparser reads (port "8080" stays a string) — then delete the .ini file. \
Change settings.load to read configs/<name>.json instead, returning the same nested dict it \
returns today. There are three files: alpha, beta and gamma."""
context_files = ["settings.py", "configs/alpha.ini"]
```

`project/settings.py`:

```python
"""Named configurations under configs/: load(name) returns
{section: {key: value}} with every value a string."""

import configparser
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def load(name: str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(CONFIG_DIR / f"{name}.ini")
    return {section: dict(parser[section]) for section in parser.sections()}
```

`project/configs/alpha.ini`:

```ini
[server]
host = alpha.local
port = 8080

[limits]
retries = 3
```

`project/configs/beta.ini`: the same shape with `beta.local`, `8081`, `5`. `project/configs/gamma.ini`: `gamma.local`, `9000`, `1`.

`grade/test_migration.py`:

```python
import json
import unittest
from pathlib import Path

EXPECTED = {
    "alpha": {"server": {"host": "alpha.local", "port": "8080"}, "limits": {"retries": "3"}},
    "beta": {"server": {"host": "beta.local", "port": "8081"}, "limits": {"retries": "5"}},
    "gamma": {"server": {"host": "gamma.local", "port": "9000"}, "limits": {"retries": "1"}},
}
CONFIGS = Path("configs")


class MigrationTests(unittest.TestCase):
    def test_every_ini_became_a_json_file(self):
        for name in EXPECTED:
            self.assertTrue((CONFIGS / f"{name}.json").is_file(), name)
            self.assertFalse((CONFIGS / f"{name}.ini").exists(), name)

    def test_the_json_holds_the_sections_with_string_values(self):
        for name, expected in EXPECTED.items():
            with (CONFIGS / f"{name}.json").open() as f:
                self.assertEqual(json.load(f), expected, name)

    def test_load_reads_the_json(self):
        from settings import load

        self.assertFalse((CONFIGS / "beta.ini").exists())
        self.assertEqual(load("beta"), EXPECTED["beta"])
        self.assertEqual(load("gamma"), EXPECTED["gamma"])
```

`solution/settings.py`:

```python
"""Named configurations under configs/: load(name) returns
{section: {key: value}} with every value a string."""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def load(name: str) -> dict[str, dict[str, str]]:
    with (CONFIG_DIR / f"{name}.json").open() as f:
        return json.load(f)
```

`solution/configs/alpha.json`, `beta.json`, `gamma.json`: the `EXPECTED` objects above, one per file, as indented JSON; no `.ini` files in `solution/configs/`.

- [ ] **Step 5: Run the proof, the loader tests and the four CI checks**

Run: `uv run pytest tests/test_tune_suite.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS with eight parametrized proof cases. `test_the_shipped_tasks_load_from_the_package` must list all eight names.

- [ ] **Step 6: Commit**

```bash
git add src/sous/tune/suite/tasks
git commit -m "feat(tune): suite tasks for a bug fix, codegen, a CLI flag and a config migration

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: The suite runner: one arm, every task, through the real worker loop

**Files:**
- Create: `src/sous/tune/suite/runner.py`
- Test: `tests/test_tune_runner.py` (create)

**Interfaces:**
- Consumes: `sous.worker.run_task(task, store, engine, config, context=ContextDecision)`, `TaskStore` (`enqueue`, `claim_next`, `get`, `respond_approval` — returns True only for the decision that took effect), `ManagedEngine`, `EngineManager`, `default_engine_factory` (Task 3), `bench.release` (Task 3), `bench._check_drafter`, `bench._active_memory`, `grading.grade_task` (Task 5), `Arm.suite_key` (Task 2), `ReplaySafe`.
- Produces:
  - `SuiteRun` (frozen dataclass): `task, index, label, model_id, drafter_id, block_size, int8_prefill, greedy, window, state, outcome, turns, seconds, output_tokens, malformed, repetitions, approvals_denied, grade, grade_detail, error, transcript_path`; `key` (the arm's `suite_key`), `completed` (`state == "done"`), `as_dict()`, `from_dict()`.
  - `SuiteOutcome` (frozen dataclass): `runs: list[SuiteRun]`, `released: bool`, `error: str | None`.
  - `SUITE_ALLOWLIST = (*DEFAULT_ALLOWLIST, "python -m unittest", "python3 -m unittest")`.
  - `CountingEngine(inner: Engine)`: the `Engine` protocol by delegation, `output_tokens` summed over every `generate` from `Delta.output_tokens`.
  - `metrics_from_transcript(path: Path) -> tuple[int, int]`: `(malformed, repetitions)`.
  - `interpreter_first(python: Path)`: a context manager putting `python.parent` first on `PATH` for its span.
  - `run_one(task, index, arm, engine: ManagedEngine, counting: CountingEngine, scratch: Path, *, python: Path, poll: float = 0.1, grade_timeout: float = 120.0) -> SuiteRun`.
  - `run_suite(arm, tasks, *, runs: int, done: set[tuple[str, int]], record: Callable[[SuiteRun], None], scratch: Path, out, factory=None, python=None, active_memory=None, poll=0.1) -> SuiteOutcome`.
  - `estimate_seconds(arms: list[Arm], rows: list[BenchRow], *, tasks: int, runs: int) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tune_runner.py
"""The suite runner drives sous's own worker loop: the sandbox and the
allowlist are real, approvals are denied, and every number in a run is read
back from the store, the transcript and the engine's deltas."""

import json
import os
import sys
import threading
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import ManagedEngine, ReplaySafe
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import (
    SUITE_ALLOWLIST,
    CountingEngine,
    SuiteRun,
    estimate_seconds,
    interpreter_first,
    metrics_from_transcript,
    run_one,
    run_suite,
)
from tests.fake_engine import FakeEngine

PYTHON = Path(sys.executable)
CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
FINISH = CALL.format(name="finish", args='{"summary": "done", "concerns": ""}')


def _task(name="implement_rpn"):
    return next(t for t in load_tasks() if t.name == name)


def _arm(tmp_path, **over):
    cfg = SousConfig(
        data_dir=tmp_path / "never-used",
        config_path=tmp_path / "never-used.toml",
        model_id="org/m",
        speculative_draft_id="",
        max_context_tokens=8192,
        approval_timeout_minutes=1,
        **over,
    )
    return Arm(
        label="m",
        config=cfg,
        model_id="org/m",
        drafter_id="",
        block_size=0,
        window=8192,
        gateway_window=None,
        tier="t",
        current=True,
    )


def _solution_write(task):
    content = (task.solution / "rpn.py").read_text()
    return CALL.format(name="write_file", args=json.dumps({"path": "rpn.py", "content": content}))


def test_metrics_from_a_transcript_count_malformed_calls_and_repetition_streaks(tmp_path):
    p = tmp_path / "transcript.jsonl"
    same = {"event": "tool", "name": "read_file", "arguments": {"path": "a.py"}, "result": "x"}
    other = {"event": "tool", "name": "read_file", "arguments": {"path": "b.py"}, "result": "y"}
    events = [
        {"event": "generation", "turn": 1, "text": "..."},
        {"event": "malformed", "error": "no call"},
        same,
        same,
        same,
        same,  # one streak of four is one incident
        other,
        {"event": "malformed", "error": "bad json"},
        same,
        same,
        same,  # a second streak is a second incident
        {"event": "finished", "outcome": "completed"},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    assert metrics_from_transcript(p) == (2, 2)
    assert metrics_from_transcript(tmp_path / "missing.jsonl") == (0, 0)


def test_interpreter_first_prepends_the_interpreters_directory_and_restores(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    with interpreter_first(Path("/venv/bin/python3.14")):
        assert os.environ["PATH"].split(os.pathsep) == ["/venv/bin", "/usr/bin"]
    assert os.environ["PATH"] == "/usr/bin"


def test_the_suite_allowlist_is_the_shipped_one_plus_unittest():
    from sous.config import DEFAULT_ALLOWLIST

    assert SUITE_ALLOWLIST[: len(DEFAULT_ALLOWLIST)] == tuple(DEFAULT_ALLOWLIST)
    assert SUITE_ALLOWLIST[len(DEFAULT_ALLOWLIST) :] == ("python -m unittest", "python3 -m unittest")


def test_a_counting_engine_reads_the_deltas_through_a_replay_safe_callback():
    inner = FakeEngine(["one two three", "four"])
    counting = CountingEngine(inner)
    counting.generate([], [], 8)
    counting.generate([], [], 8)
    assert counting.output_tokens == 4
    assert all(isinstance(cb, ReplaySafe) for cb in inner.on_deltas_seen)
    assert counting.model_id == "fake/model"
    assert ManagedEngine(counting).drafter is None


def test_run_one_grades_a_solved_task_and_reads_every_metric_from_the_run(tmp_path):
    task = _task()
    inner = FakeEngine([_solution_write(task), FINISH])
    counting = CountingEngine(inner)
    engine = ManagedEngine(counting)
    run = run_one(task, 0, _arm(tmp_path), engine, counting, tmp_path / "s", python=PYTHON)
    assert run.task == "implement_rpn" and run.index == 0 and run.label == "m"
    assert run.state == "done" and run.outcome == "completed" and run.completed
    assert run.turns == 2 and run.grade == 1.0, run.grade_detail
    assert run.output_tokens == counting.output_tokens > 0
    assert run.malformed == 0 and run.repetitions == 0 and run.approvals_denied == 0
    assert run.error is None and run.seconds >= 0
    assert run.transcript_path and Path(run.transcript_path).is_file()
    assert (tmp_path / "s" / "project" / "rpn.py").read_text() == (task.solution / "rpn.py").read_text()
    assert run.key == (*_arm(tmp_path).key, False, False)


def test_a_command_outside_the_allowlist_is_denied_and_counted(tmp_path):
    task = _task()
    denied = CALL.format(name="run_command", args='{"command": "echo hi"}')
    allowed = CALL.format(name="run_command", args='{"command": "python -m unittest discover"}')
    inner = FakeEngine([denied, allowed, FINISH])
    counting = CountingEngine(inner)
    run = run_one(
        task, 1, _arm(tmp_path), ManagedEngine(counting), counting, tmp_path / "s", python=PYTHON
    )
    assert run.approvals_denied == 1 and run.state == "done" and run.index == 1
    lines = [json.loads(l) for l in Path(run.transcript_path).read_text().splitlines()]
    tools = [e for e in lines if e.get("event") == "tool"]
    assert "command denied" in tools[0]["result"]
    assert tools[1]["result"].startswith("exit code")
    assert run.grade == 0.0  # nothing was implemented


def test_an_engine_failure_is_a_failed_run_with_a_zero_grade(tmp_path):
    task = _task()
    counting = CountingEngine(FakeEngine([]))  # the script is exhausted at once
    run = run_one(
        task, 0, _arm(tmp_path), ManagedEngine(counting), counting, tmp_path / "s", python=PYTHON
    )
    assert run.state == "failed" and not run.completed
    assert run.error and "engine error" in run.error
    assert run.grade == 0.0 and run.turns == 0


def test_run_suite_loads_once_runs_what_is_not_done_records_in_order_and_releases(tmp_path):
    task = _task()
    scripts = [_solution_write(task), FINISH] * 3
    inner = FakeEngine(scripts)
    loads = []

    def factory(model_id):
        loads.append(model_id)
        return inner

    recorded = []
    outcome = run_suite(
        _arm(tmp_path),
        [task],
        runs=3,
        done={("implement_rpn", 1)},
        record=recorded.append,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert loads == ["org/m"]
    assert [(r.task, r.index) for r in outcome.runs] == [("implement_rpn", 0), ("implement_rpn", 2)]
    assert recorded == outcome.runs
    assert all(r.grade == 1.0 for r in outcome.runs)
    assert outcome.released and outcome.error is None
    assert inner.unloaded
    assert (tmp_path / "scratch" / "m" / "implement_rpn-3" / "project" / "rpn.py").is_file()


def test_run_suite_runs_on_a_thread_of_its_own(tmp_path):
    task = _task()
    names = []
    run_suite(
        _arm(tmp_path),
        [task],
        runs=1,
        done=set(),
        record=lambda r: names.append(threading.current_thread().name),
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    # Each run is recorded from the thread that owns the arm's engine — the
    # one that releases its mlx state on the way out — never the caller's.
    assert names == ["sous-tune-suite-m"]


def test_a_load_failure_is_an_outcome_with_no_runs(tmp_path):
    def factory(model_id):
        raise RuntimeError("no weights")

    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=factory,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == [] and outcome.released
    assert outcome.error == "load failed: RuntimeError: no weights"


def test_an_arm_whose_drafter_or_int8_did_not_load_runs_nothing(tmp_path):
    class Engine(FakeEngine):
        drafter = ""
        int8_prefill_status = {"state": "unavailable", "reason": "no tensor units", "routed": 0}

    arm = _arm(tmp_path)
    drafted = Arm(**{**vars(arm), "drafter_id": "z/d", "block_size": 3, "label": "m + d @3"})
    lines = []
    outcome = run_suite(
        drafted,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "a",
        out=lines.append,
        factory=lambda mid: Engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == [] and outcome.released
    assert outcome.error == "drafter z/d requested, engine runs with none"
    int8 = Arm(**{**vars(arm), "int8_prefill": True, "label": "m + int8 prefill"})
    outcome = run_suite(
        int8,
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "b",
        out=lines.append,
        factory=lambda mid: Engine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert outcome.runs == []
    assert outcome.error == "int8 prefill requested, engine reports unavailable: no tensor units"


def test_memory_is_judged_against_the_baseline_taken_before_the_load(tmp_path, monkeypatch):
    from sous.tune import bench

    monkeypatch.setattr(bench, "_UNLOAD_WAIT_SECONDS", 0.0)
    readings = iter([5 * 2**30, 5 * 2**30, 9 * 2**30])
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: next(readings),
    )
    # Baseline 5 GiB before the load, 5 GiB after the unload's wait, then 9
    # GiB on the final reading: 4 GiB more than the baseline stayed resident.
    assert not outcome.released
    assert outcome.error == "memory not released: 4.0 GiB still resident"
    assert [r.state for r in outcome.runs] == ["done"]


def test_a_run_that_raises_outside_the_worker_is_recorded_as_an_error_and_the_suite_goes_on(
    tmp_path, monkeypatch
):
    from sous.tune.suite import runner

    calls = []
    real = runner.run_one

    def flaky(task, index, *args, **kwargs):
        calls.append(index)
        if index == 0:
            raise OSError("disk full")
        return real(task, index, *args, **kwargs)

    monkeypatch.setattr(runner, "run_one", flaky)
    outcome = run_suite(
        _arm(tmp_path),
        [_task()],
        runs=2,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=lambda mid: FakeEngine([FINISH]),
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert calls == [0, 1]
    assert [r.state for r in outcome.runs] == ["error", "done"]
    assert outcome.runs[0].error == "OSError: disk full" and outcome.runs[0].grade == 0.0


def test_suite_runs_round_trip_through_dicts(tmp_path):
    run = SuiteRun(
        task="t", index=0, label="m", model_id="org/m", drafter_id="", block_size=0,
        int8_prefill=False, greedy=True, window=8192, state="done", outcome="completed",
        turns=3, seconds=12.5, output_tokens=400, malformed=0, repetitions=0,
        approvals_denied=1, grade=0.5, grade_detail="1/2", error=None, transcript_path=None,
    )
    assert SuiteRun.from_dict(json.loads(json.dumps(run.as_dict()))) == run
    assert run.key == ("org/m", "", 0, False, True)


def test_estimate_seconds_scales_with_the_measured_speeds(tmp_path):
    arm = _arm(tmp_path)
    row = BenchRow(
        label="m", model_id="org/m", drafter_id="", block_size=0, window=8192, ok=True,
        error=None, load_seconds=1.0, prefill_tps_2k=300.0, prefill_tps_16k=None,
        decode_tps_1k=30.0, decode_tps_16k=None, ttft_seconds=1.0, peak_memory_bytes=1,
        spread=None,
    )
    per_run = 8 * (3000 / 300 + 300 / 30)
    assert estimate_seconds([arm], [row], tasks=8, runs=2) == per_run * 16
    assert estimate_seconds([arm], [], tasks=8, runs=2) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: sous.tune.suite.runner`.

- [ ] **Step 3: Implement**

```python
# src/sous/tune/suite/runner.py
"""One suite run is one task through sous's own worker loop — the sandbox,
the allowlist, the approval hook, the budgets, the verify commands — on an
engine the arm loaded once for every task; graded afterwards, recorded as
it finishes, and released the way the bench releases its arm."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sous.config import DEFAULT_ALLOWLIST, SousConfig
from sous.context import ContextDecision
from sous.engine.base import (
    Delta,
    Engine,
    EngineManager,
    ManagedEngine,
    OnDelta,
    ReplaySafe,
    default_engine_factory,
    release_mlx_thread_state,
)
from sous.tasks import TaskState, TaskStore
from sous.tune.arms import Arm
from sous.tune.bench import BenchRow, _active_memory, _check_drafter, release
from sous.tune.suite import SuiteTask
from sous.tune.suite.grading import grade_task
from sous.worker import run_task

SUITE_ALLOWLIST = (*DEFAULT_ALLOWLIST, "python -m unittest", "python3 -m unittest")
_POLL_SECONDS = 0.1
# The ETA's picture of one run — the turns a task takes and what each turn
# prefills and decodes — from the M5 Pro's delegated tasks of 2026-09.
ETA_TURNS = 8
ETA_PROMPT_TOKENS = 3000
ETA_OUTPUT_TOKENS = 300


@dataclass(frozen=True)
class SuiteRun:
    task: str
    index: int
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    int8_prefill: bool
    greedy: bool
    window: int
    state: str
    outcome: str | None
    turns: int
    seconds: float
    output_tokens: int
    malformed: int
    repetitions: int
    approvals_denied: int
    grade: float
    grade_detail: str
    error: str | None
    transcript_path: str | None

    @property
    def key(self) -> tuple[str, str, int, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (self.model_id, self.drafter_id, self.block_size, self.int8_prefill, self.greedy)

    @property
    def completed(self) -> bool:
        return self.state == TaskState.DONE

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SuiteRun:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


@dataclass(frozen=True)
class SuiteOutcome:
    runs: list[SuiteRun]
    # False when the weights stayed resident after the arm: the caller must
    # stop, as it does for a bench row, rather than load a second model.
    released: bool
    error: str | None


class CountingEngine:
    """The real engine behind a callback that sums what every generate
    produced: run_task passes no on_delta of its own, so this is the one
    reader of Delta.output_tokens on a suite run. The callback is
    ReplaySafe — nothing it sees leaves the process — so a warm attempt
    that fails may still be retried cold, as with no callback at all."""

    def __init__(self, inner: Engine):
        self._inner = inner
        self.output_tokens = 0
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def __getattr__(self, name: str):
        # drafter, positions, int8_prefill_status: whatever the backend has,
        # read through getattr(..., None) by ManagedEngine.
        return getattr(self._inner, name)

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        produced = [0]

        def count(d: Delta) -> None:
            # The count restarts on a cold retry; the attempt that finished
            # is the largest one seen.
            produced[0] = max(produced[0], d.output_tokens)
            if on_delta is not None:
                on_delta(d)

        callback = ReplaySafe(count) if on_delta is None or isinstance(on_delta, ReplaySafe) else count
        text = self._inner.generate(messages, tools, max_tokens, callback)
        with self._lock:
            self.output_tokens += produced[0]
        return text

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self._inner.count_tokens(messages, tools)

    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self._inner.reset_prompt_cache(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._inner.prompt_cache_stats(owner)

    def unload(self) -> None:
        self._inner.unload()


def metrics_from_transcript(path: Path) -> tuple[int, int]:
    """(malformed tool calls, repetition incidents) from a task's
    transcript. An incident is the model looping on one tool: three identical
    consecutive executed calls, counted once per streak of three or more. A
    `tool` event is exactly one executed call with its arguments, so nothing
    needs re-parsing; a `finish` never becomes one and cannot repeat."""
    if not path.is_file():
        return 0, 0
    malformed = repetitions = streak = 0
    last: tuple | None = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("event") if isinstance(event, dict) else None
        if kind == "malformed":
            malformed += 1
        elif kind == "tool":
            call = (event.get("name"), json.dumps(event.get("arguments"), sort_keys=True))
            streak = streak + 1 if call == last else 1
            last = call
            if streak == 3:
                repetitions += 1
    return malformed, repetitions


@contextlib.contextmanager
def interpreter_first(python: Path) -> Iterator[None]:
    """`python` on the worker's PATH resolves to the tune's own interpreter
    for the span: the sandbox passes PATH through (toolexec.scrubbed_env)
    and the `python -m unittest` the worker runs must be the Python the
    graders use, with the standard library and nothing else."""
    before = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join([str(python.parent), *([before] if before else [])])
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = before


class _Denier:
    """Denies every approval the worker asks for the moment it appears, and
    counts it: a suite run has no human, and a command outside the
    allowlist must cost the run a denial, not the approval timeout."""

    def __init__(self, store: TaskStore, task_id: str, poll: float):
        self._store, self._task_id, self._poll = store, task_id, poll
        self.denied = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sous-tune-denier", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            task = self._store.get(self._task_id)
            # respond_approval is true only for the decision that took
            # effect, so a request seen twice before the worker polls it
            # is counted once.
            if (
                task is not None
                and task.state == TaskState.AWAITING_APPROVAL
                and self._store.respond_approval(self._task_id, approve=False)
            ):
                self.denied += 1
            self._stop.wait(self._poll)

    def __enter__(self) -> _Denier:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5.0)


def _scratch_config(arm: Arm, scratch: Path, task: SuiteTask) -> SousConfig:
    """The arm's config over a scratch control directory: the worker reads
    the allowlist from config_path on every command, so the file has to
    exist, and the data dir holds this run's store and transcript."""
    config_path = scratch / "config.toml"
    entries = ", ".join(json.dumps(e) for e in SUITE_ALLOWLIST)
    config_path.write_text(f"[commands]\nallowlist = [{entries}]\n")
    return dataclasses.replace(
        arm.config,
        data_dir=scratch / "data",
        config_path=config_path,
        max_turns=task.max_turns,
        max_minutes=task.max_minutes,
        context_mode="fixed",
        max_context_tokens=arm.window,
    )


def _error_run(task: SuiteTask, index: int, arm: Arm, error: str) -> SuiteRun:
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
        state="error",
        outcome=None,
        turns=0,
        seconds=0.0,
        output_tokens=0,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=0.0,
        grade_detail="",
        error=error,
        transcript_path=None,
    )


def run_one(
    task: SuiteTask,
    index: int,
    arm: Arm,
    engine: ManagedEngine,
    counting: CountingEngine,
    scratch: Path,
    *,
    python: Path,
    poll: float = _POLL_SECONDS,
    grade_timeout: float = 120.0,
) -> SuiteRun:
    """One task, once, through run_task — the loop the daemon runs, minus
    the queue polling around it — then the grade over what it left."""
    project = scratch / "project"
    shutil.copytree(task.project, project)
    config = _scratch_config(arm, scratch, task)
    store = TaskStore(scratch / "tasks.db")
    queued = store.enqueue(
        title=task.title,
        instructions=task.instructions,
        project_root=str(project),
        context_files=list(task.context_files),
        verify_commands=list(task.verify_commands),
    )
    claimed = store.claim_next()
    if claimed is None or claimed.id != queued.id:
        raise RuntimeError("the scratch store handed back another task")
    before = counting.output_tokens
    with _Denier(store, claimed.id, poll) as denier, interpreter_first(python):
        run_task(claimed, store, engine, config, context=ContextDecision(arm.window, "tune"))
    final = store.get(claimed.id)
    if final is None:
        raise RuntimeError("the task vanished from the scratch store")
    transcript = config.data_dir / "tasks" / claimed.id / "transcript.jsonl"
    malformed, repetitions = metrics_from_transcript(transcript)
    grade = grade_task(task, project, python=python, timeout=grade_timeout)
    finished = final.finished_at if final.finished_at is not None else time.time()
    started = final.started_at if final.started_at is not None else final.created_at
    report = final.report or {}
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
        state=final.state,
        outcome=final.outcome,
        turns=final.turns_used,
        seconds=finished - started,
        output_tokens=counting.output_tokens - before,
        malformed=malformed,
        repetitions=repetitions,
        approvals_denied=denier.denied,
        grade=grade.score,
        grade_detail=grade.detail,
        error=None if final.state == TaskState.DONE else str(report.get("error") or final.state),
        transcript_path=str(transcript),
    )


def _check_int8(arm: Arm, engine: ManagedEngine) -> None:
    """Like the drafter check: the engine refuses INT8 prefill with a status
    rather than an error, and a run under the int8 label over the stock
    path would put that setting in the config on stock numbers."""
    if not arm.int8_prefill:
        return
    status = engine.int8_prefill_status or {}
    if status.get("state") != "active":
        reason = status.get("reason") or "no status"
        raise RuntimeError(
            f"int8 prefill requested, engine reports {status.get('state', 'nothing')}: {reason}"
        )


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


def _run_suite(
    arm: Arm,
    tasks: list[SuiteTask],
    runs: int,
    done: set[tuple[str, int]],
    record: Callable[[SuiteRun], None],
    scratch: Path,
    out: Callable[..., None],
    factory: Callable[[str], Engine] | None,
    python: Path,
    active_memory: Callable[[], int],
    poll: float,
) -> SuiteOutcome:
    baseline = active_memory()
    base = factory or default_engine_factory(arm.config)
    counters: list[CountingEngine] = []

    def wrapped(model_id: str) -> Engine:
        counters.append(CountingEngine(base(model_id)))
        return counters[-1]

    manager = EngineManager(arm.config, engine_factory=wrapped)
    try:
        engine = manager.get()
    except Exception as e:  # noqa: BLE001 — the arm is skipped, named; nothing is resident
        return SuiteOutcome([], True, f"load failed: {type(e).__name__}: {e}")
    results: list[SuiteRun] = []
    check_error: str | None = None
    try:
        _check_drafter(arm, engine)
        _check_int8(arm, engine)
    except RuntimeError as e:
        check_error = str(e)
    if check_error is None:
        for task in tasks:
            for index in range(runs):
                if (task.name, index) in done:
                    continue
                out(f"  {arm.label}: {task.name} #{index + 1} ...")
                run_scratch = scratch / _slug(arm.label) / f"{task.name}-{index + 1}"
                # A resumed run may have died mid-task here; nothing in a
                # half-run scratch is worth keeping.
                if run_scratch.exists():
                    shutil.rmtree(run_scratch)
                run_scratch.mkdir(parents=True)
                try:
                    result = run_one(
                        task, index, arm, engine, counters[-1], run_scratch, python=python, poll=poll
                    )
                except Exception as e:  # noqa: BLE001 — one run's failure, recorded; the suite goes on
                    result = _error_run(task, index, arm, f"{type(e).__name__}: {e}")
                record(result)
                results.append(result)
                out(
                    f"    {task.name} #{index + 1}: {result.state}, grade {result.grade:.2f}, "
                    f"{result.seconds:.0f} s, {result.turns} turn(s)"
                )
    teardown = release(
        manager, None, baseline=baseline, active_memory=active_memory, label=arm.label, out=out
    )
    return SuiteOutcome(results, teardown is None, teardown or check_error)


def run_suite(
    arm: Arm,
    tasks: list[SuiteTask],
    *,
    runs: int,
    done: set[tuple[str, int]],
    record: Callable[[SuiteRun], None],
    scratch: Path,
    out: Callable[..., None] = print,
    factory: Callable[[str], Engine] | None = None,
    python: Path | None = None,
    active_memory: Callable[[], int] | None = None,
    poll: float = _POLL_SECONDS,
) -> SuiteOutcome:
    """Every (task, run index) of the arm not in `done`, on a thread of its
    own that loads the engine once, hands each run to `record` as it
    finishes, and releases its mlx state on the way out."""
    outcome: list[SuiteOutcome | BaseException] = []

    def run() -> None:
        try:
            outcome.append(
                _run_suite(
                    arm,
                    tasks,
                    runs,
                    done,
                    record,
                    scratch,
                    out,
                    factory,
                    python or Path(sys.executable),
                    active_memory or _active_memory,
                    poll,
                )
            )
        except BaseException as e:  # noqa: BLE001 — becomes the outcome's error
            outcome.append(e)
        finally:
            release_mlx_thread_state()

    worker = threading.Thread(target=run, name=f"sous-tune-suite-{arm.label}", daemon=True)
    worker.start()
    worker.join()
    result = outcome[0]
    if isinstance(result, BaseException):
        # Only a failure before the load finished reaches here (_run_suite
        # turns everything after it into an outcome), so nothing is resident.
        return SuiteOutcome([], True, f"{type(result).__name__}: {result}")
    return result


def estimate_seconds(
    arms: list[Arm], rows: list[BenchRow], *, tasks: int, runs: int
) -> float | None:
    """A rough suite duration from the quick stage's speeds: every arm's
    short-context prefill and decode over ETA_TURNS turns of a typical
    task's prompt and answer, times the tasks and runs. None when an arm has
    no usable row — a guess would be read as a measurement."""
    by_key = {r.key: r for r in rows if r.ok}
    total = 0.0
    for arm in arms:
        row = by_key.get(arm.key)
        if row is None or not row.prefill_tps_2k or not row.decode_tps_1k:
            return None
        per_turn = ETA_PROMPT_TOKENS / row.prefill_tps_2k + ETA_OUTPUT_TOKENS / row.decode_tps_1k
        total += ETA_TURNS * per_turn * tasks * runs
    return total
```

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_runner.py tests/test_worker.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS. The runner tests take a few seconds each (a real `python -m sous.tune.suite.unittests` subprocess per grade); that is intended — they exercise the shipped grader end to end.

- [ ] **Step 5: Commit**

```bash
git add src/sous/tune/suite/runner.py tests/test_tune_runner.py
git commit -m "feat(tune): run every suite task through the real worker loop and record each run

One engine per arm on a thread of its own, the shipped allowlist plus
python -m unittest, approvals denied and counted, output tokens from the
engine's own deltas, and the grade over what the worker left.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: The full decision rule

**Files:**
- Modify: `src/sous/tune/decide.py`
- Test: `tests/test_tune_decide.py` (append)

**Interfaces:**
- Consumes: `Arm.suite_key`, `Arm.int8_prefill`, `Arm.greedy` (Task 2), `SuiteRun` (Task 8), `BenchRow`, `score()` and `_changes()` (PR 1), `sous.tune.hardware.gib`.
- Produces:
  - `MARGIN = 0.05`.
  - `ArmSummary` (frozen dataclass): `label, key, model_id, runs, completed, mean_grade, wall_seconds, repetitions, malformed, approvals_denied, output_tokens, peak_memory_bytes: int | None`.
  - `summarize(arm: Arm, runs: list[SuiteRun], rows: list[BenchRow]) -> ArmSummary | None` — `None` when the arm has no run; `peak_memory_bytes` from the arm's ok bench row (by `Arm.key`), else `None`.
  - `model_stage(arms: list[Arm], rows: list[BenchRow]) -> list[Arm]` — each model's fastest quality-neutral arm (one scoring unit per model, as `quick_decision` does across a model's rows; ties prefer the current arm, then the drafterless one), in the arms' order, plus the current arm when it is not already chosen and has an ok row.
  - `FullChoice` (frozen dataclass): `label: str`, `arm: Arm`, `reference_label: str`, `changes: dict[str, dict[str, object]]`, `reasons: list[str]`.
  - `full_decision(user: SousConfig, arms: list[Arm], runs: list[SuiteRun], rows: list[BenchRow], *, runs_per_task: int, reference: Arm | None = None) -> FullChoice | None`: the spec's rule. `reference=None` means the current arm when it has runs, else the first summarized arm (the largest fitting tier: the table lists tiers largest first and `quick_arms` keeps that order); a given `reference` (the winner stage) must have runs or the result is `None`. `reasons` holds the rule with every number, then one line per arm ending in `-> eligible` or `-> not eligible: <why>`, then the winner line.
  - `_changes(user, arm, *, full: bool = False)`: with `full`, also `id` (a different model), `int8_prefill` (when it differs), `temperature = 0.0` (a greedy arm over a sampling config).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tune_decide.py`; add `import dataclasses` to the module's imports)

```python
from sous.tune.decide import ArmSummary, FullChoice, full_decision, model_stage, summarize
from sous.tune.suite.runner import SuiteRun

M9 = "mlx-community/Qwen3.5-9B-MLX-4bit"


def _run(arm, task="t", index=0, *, grade=1.0, seconds=60.0, state="done", repetitions=0, tokens=100):
    return SuiteRun(
        task=task,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        window=arm.window,
        state=state,
        outcome="completed" if state == "done" else None,
        turns=4,
        seconds=seconds,
        output_tokens=tokens,
        malformed=0,
        repetitions=repetitions,
        approvals_denied=0,
        grade=grade,
        grade_detail="",
        error=None if state == "done" else "x",
        transcript_path=None,
    )


def _runs(arm, grades, **kw):
    return [_run(arm, task=f"t{i}", grade=g, **kw) for i, g in enumerate(grades)]


def _user(tmp_path, **over):
    return SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml", **over)


def _peak(label, model, drafter, block, gib):
    row = _row(label, model=model, drafter=drafter, block=block)
    return dataclasses.replace(row, peak_memory_bytes=int(gib * 2**30))


def test_model_stage_picks_each_models_fastest_arm_and_keeps_the_current_one(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", block=3, current=True)
    fast = _arm(user, "27 @5", block=5)
    plain = _arm(user, "27", drafter="", block=0)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    nine_d = _arm(user, "9 + d @3", drafter="z/9d", block=3, model=M9)
    rows = [
        _row("27 @3", block=3, d16k=18.0),
        _row("27 @5", block=5, d16k=20.0),
        _row("27", drafter="", block=0, d16k=15.0),
        _row("9", model=M9, drafter="", block=0, d16k=40.0),
        _row("9 + d @3", model=M9, drafter="z/9d", block=3, d16k=40.0),
    ]
    stage = model_stage([cur, fast, plain, nine, nine_d], rows)
    assert [a.label for a in stage] == ["27 @5", "9", "27 @3"]


def test_model_stage_drops_a_model_without_a_successful_row(tmp_path):
    user = _user(tmp_path)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    assert model_stage([nine], [_row("9", model=M9, drafter="", block=0, ok=False)]) == []


def test_summarize_reads_the_runs_and_the_bench_peak(tmp_path):
    user = _user(tmp_path)
    arm = _arm(user, "27 @3", current=True)
    runs = _runs(arm, [1.0, 0.5], seconds=30.0) + [_run(arm, "t2", state="failed", grade=0.0)]
    s = summarize(arm, runs, [_peak("27 @3", M, D, 3, 20.0)])
    assert s == ArmSummary(
        label="27 @3",
        key=arm.suite_key,
        model_id=M,
        runs=3,
        completed=2,
        mean_grade=0.5,
        wall_seconds=120.0,
        repetitions=0,
        malformed=0,
        approvals_denied=0,
        output_tokens=300,
        peak_memory_bytes=20 * 2**30,
    )
    assert summarize(_arm(user, "27 @5", block=5), runs, []) is None


def test_a_faster_arm_within_the_margin_wins_and_changes_the_model(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True, fit_window=131072)
    nine = _arm(user, "9", drafter="", block=0, model=M9, fit_window=131072)
    runs = _runs(cur, [1.0, 0.9, 0.9, 1.0], seconds=100.0) + _runs(nine, [0.9] * 4, seconds=40.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "9" and choice.reference_label == "27 @3"
    assert choice.changes == {"model": {"id": M9, "speculative_draft_id": ""}}
    text = "\n".join(choice.reasons)
    assert "reference: 27 @3 (the configured arm)" in text
    assert "mean grade >= 0.90 (0.95 - 0.05)" in text
    assert "9: grade 0.90, completed 4/4, wall 160 s" in text and "-> eligible" in text
    assert "winner: 9 (160 s vs 400 s for the reference)" in text


def test_a_faster_arm_below_the_margin_loses(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0, 1.0], seconds=100.0) + _runs(nine, [0.9, 0.98], seconds=10.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3" and choice.changes == {}
    assert any("9: " in r and "not eligible: grade 0.94 < 0.95" in r for r in choice.reasons)
    assert "winner: 27 @3 (the reference; nothing eligible is faster)" in choice.reasons[-1]


def test_too_few_completed_runs_or_more_repetitions_make_an_arm_ineligible(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0] * 4, seconds=100.0) + [
        _run(nine, "t0", grade=1.0, seconds=10.0),
        _run(nine, "t1", grade=1.0, seconds=10.0, state="failed"),
        _run(nine, "t2", grade=1.0, seconds=10.0, state="failed"),
        _run(nine, "t3", grade=1.0, seconds=10.0, state="failed"),
    ]
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3"
    assert any("not eligible: completed 1 < 2" in r for r in choice.reasons)
    runs = _runs(cur, [1.0] * 2, seconds=100.0) + _runs(nine, [1.0] * 2, seconds=10.0, repetitions=1)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=2)
    assert choice is not None and choice.label == "27 @3"
    assert any("not eligible: repetitions 2 > 0" in r for r in choice.reasons)


def test_a_wall_time_tie_goes_to_the_smaller_peak_memory(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    runs = _runs(cur, [1.0], seconds=50.0) + _runs(nine, [1.0], seconds=50.0)
    rows = [_peak("27 @3", M, D, 3, 20.0), _peak("9", M9, "", 0, 8.0)]
    choice = full_decision(user, [cur, nine], runs, rows, runs_per_task=1)
    assert choice is not None and choice.label == "9"
    assert "a tie goes to the smaller peak memory" in "\n".join(choice.reasons)
    # An exact tie in both keeps the reference.
    rows = [_peak("27 @3", M, D, 3, 8.0), _peak("9", M9, "", 0, 8.0)]
    assert full_decision(user, [cur, nine], runs, rows, runs_per_task=1).label == "27 @3"


def test_without_runs_of_the_current_arm_the_first_measured_arm_is_the_reference(tmp_path):
    user = _user(tmp_path)
    cur = _arm(user, "27 @3", current=True)
    nine = _arm(user, "9", drafter="", block=0, model=M9)
    four = _arm(user, "4", drafter="", block=0, model="mlx-community/Qwen3.5-4B-MLX-4bit")
    runs = _runs(nine, [0.8], seconds=50.0) + _runs(four, [0.8], seconds=20.0)
    choice = full_decision(user, [cur, nine, four], runs, [], runs_per_task=1)
    assert choice is not None and choice.reference_label == "9" and choice.label == "4"
    assert "reference: 9 (the fastest arm of the largest fitting tier" in choice.reasons[0]
    assert full_decision(user, [cur], [], [], runs_per_task=1) is None


def test_the_winner_stage_judges_each_extra_arm_against_the_winner(tmp_path):
    user = _user(tmp_path)
    winner = _arm(user, "27 @3", current=True)
    int8 = dataclasses.replace(
        winner,
        label="27 @3 + int8 prefill",
        config=dataclasses.replace(user, int8_prefill=True),
        current=False,
        int8_prefill=True,
    )
    greedy = dataclasses.replace(
        winner,
        label="27 @3 greedy",
        config=dataclasses.replace(user, temperature=0.0),
        current=False,
        greedy=True,
    )
    runs = (
        _runs(winner, [1.0, 1.0], seconds=100.0)
        + _runs(int8, [1.0, 0.96], seconds=80.0)
        + _runs(greedy, [1.0, 1.0], seconds=70.0)
    )
    choice = full_decision(user, [winner, int8, greedy], runs, [], runs_per_task=2, reference=winner)
    assert choice is not None and choice.label == "27 @3 greedy"
    assert choice.changes == {"model": {"temperature": 0.0}}
    assert "reference: 27 @3 (the model stage's winner)" in choice.reasons[0]
    runs = _runs(winner, [1.0, 1.0], seconds=100.0) + _runs(int8, [1.0, 0.96], seconds=80.0)
    choice = full_decision(user, [winner, int8], runs, [], runs_per_task=2, reference=winner)
    assert choice.changes == {"model": {"int8_prefill": True}}
    runs = _runs(winner, [1.0, 1.0], seconds=100.0) + _runs(int8, [1.0, 0.88], seconds=80.0)
    assert full_decision(user, [winner, int8], runs, [], runs_per_task=2, reference=winner).changes == {}
    assert full_decision(user, [winner, int8], [], [], runs_per_task=2, reference=winner) is None


def test_a_full_choice_over_another_model_writes_its_drafter_block_and_fit_window(tmp_path):
    user = _user(tmp_path, gateway_enabled=True)
    cur = _arm(user, "27 @3", current=True, fit_window=131072, fit_gateway_window=131072)
    nine = _arm(
        user,
        "9 + d @2",
        drafter="z/9d",
        block=2,
        model=M9,
        window=65536,
        gateway_window=65536,
        fit_window=65536,
        fit_gateway_window=65536,
    )
    runs = _runs(cur, [1.0], seconds=100.0) + _runs(nine, [1.0], seconds=50.0)
    choice = full_decision(user, [cur, nine], runs, [], runs_per_task=1)
    # The worker's window is not written: the arm's fit (65536) is not below
    # the configured 32768, and a window is only ever lowered. The gateway's
    # is, from 131072.
    assert choice.changes == {
        "model": {"id": M9, "speculative_draft_id": "z/9d", "speculative_block_size": 2},
        "gateway": {"max_context_tokens": 65536},
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_decide.py -v`
Expected: FAIL — `ImportError: cannot import name 'full_decision'`.

- [ ] **Step 3: Implement** (in `src/sous/tune/decide.py`)

Replace the module docstring with:

```python
"""The rules. Quick: among the current model's quality-neutral arms, the
fastest decode wins, and the model itself is never this rule's to change.
Full: the fastest arm whose suite result is within a fixed margin of the
reference, printed with every number so the choice explains itself."""
```

Add the imports `from sous.tune.hardware import gib` and `from sous.tune.suite.runner import SuiteRun`; add `MARGIN = 0.05` after the imports. Extend `_changes`:

```python
def _changes(user: SousConfig, arm: Arm, *, full: bool = False) -> dict[str, dict[str, object]]:
    model: dict[str, object] = {}
    # Only the full run may write the model and the quality-affecting keys:
    # a quick run measured throughput and nothing about what the model says.
    if full and arm.model_id != user.model_id:
        model["id"] = arm.model_id
    if arm.drafter_id != user.speculative_draft_id:
        model["speculative_draft_id"] = arm.drafter_id
    if arm.drafter_id and arm.block_size != user.speculative_block_size:
        model["speculative_block_size"] = arm.block_size
    # The arm's own fit, not the model's shared measurement window: that one
    # reserves memory for the heaviest drafter, which this arm may not load.
    window = arm.fit_window if arm.fit_window is not None else arm.window
    if window < user.max_context_tokens:
        model["max_context_tokens"] = window
    if full and arm.int8_prefill != user.int8_prefill:
        model["int8_prefill"] = arm.int8_prefill
    if full and arm.greedy and user.temperature != 0:
        model["temperature"] = 0.0
    changes: dict[str, dict[str, object]] = {}
    if model:
        changes["model"] = model
    gateway = arm.fit_gateway_window if arm.fit_gateway_window is not None else arm.gateway_window
    if user.gateway_enabled and gateway is not None and gateway < user.gateway_max_context_tokens:
        changes["gateway"] = {"max_context_tokens": gateway}
    return changes
```

Append after `quick_decision`:

```python
@dataclass(frozen=True)
class ArmSummary:
    label: str
    key: tuple[str, str, int, bool, bool]
    model_id: str
    runs: int
    completed: int
    mean_grade: float
    wall_seconds: float
    repetitions: int
    malformed: int
    approvals_denied: int
    output_tokens: int
    peak_memory_bytes: int | None


def summarize(arm: Arm, runs: list[SuiteRun], rows: list[BenchRow]) -> ArmSummary | None:
    """The arm's suite result in the rule's terms; None when it has no run.
    The peak comes from the bench row of the same model, drafter and block:
    the suite measures time and quality, the bench measured memory."""
    mine = [r for r in runs if r.key == arm.suite_key]
    if not mine:
        return None
    peak = next((r.peak_memory_bytes for r in rows if r.ok and r.key == arm.key), None)
    return ArmSummary(
        label=arm.label,
        key=arm.suite_key,
        model_id=arm.model_id,
        runs=len(mine),
        completed=sum(1 for r in mine if r.completed),
        mean_grade=sum(r.grade for r in mine) / len(mine),
        wall_seconds=sum(r.seconds for r in mine),
        repetitions=sum(r.repetitions for r in mine),
        malformed=sum(r.malformed for r in mine),
        approvals_denied=sum(r.approvals_denied for r in mine),
        output_tokens=sum(r.output_tokens for r in mine),
        peak_memory_bytes=peak,
    )


def model_stage(arms: list[Arm], rows: list[BenchRow]) -> list[Arm]:
    """What the suite runs in the model stage: each model's fastest
    quality-neutral arm by the bench (one scoring unit per model, as the
    quick rule scores; a tie keeps the current arm, then the drafterless
    one), and the current arm besides, so the reference is always measured.
    A model none of whose arms measured is not a candidate for the suite."""
    by_key = {a.key: a for a in arms}
    ok = [r for r in rows if r.ok and r.key in by_key and r.window == by_key[r.key].window]
    chosen: list[Arm] = []
    seen: set[str] = set()
    for arm in arms:
        if arm.model_id in seen:
            continue
        seen.add(arm.model_id)
        mine = [r for r in ok if r.model_id == arm.model_id]
        long = all(r.decode_tps_16k is not None for r in mine)
        scored = [(r, s) for r in mine if (s := score(r, long)) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda rs: (rs[1], by_key[rs[0].key].current, not rs[0].drafter_id))
        chosen.append(by_key[best[0].key])
    current = next((a for a in arms if a.current), None)
    if current is not None and current not in chosen and any(r.key == current.key for r in ok):
        chosen.append(current)
    return chosen


@dataclass(frozen=True)
class FullChoice:
    label: str
    arm: Arm
    reference_label: str
    changes: dict[str, dict[str, object]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _peak_text(s: ArmSummary) -> str:
    return "-" if s.peak_memory_bytes is None else gib(s.peak_memory_bytes)


def full_decision(
    user: SousConfig,
    arms: list[Arm],
    runs: list[SuiteRun],
    rows: list[BenchRow],
    *,
    runs_per_task: int,
    reference: Arm | None = None,
) -> FullChoice | None:
    """The eligible arm with the lowest suite wall time. Eligible: mean
    grade within MARGIN of the reference's, completed runs within one
    task's worth of the reference's, no more repetition incidents. The
    reference is the current arm, or the first measured one when the
    configured model did not fit (the largest fitting tier: the table is
    ordered largest first), or the arm given — the winner stage's winner.
    Every number goes into `reasons`, so the report is the rule."""
    summaries = [(a, s) for a in arms if (s := summarize(a, runs, rows)) is not None]
    if not summaries:
        return None
    if reference is not None:
        ref = next(((a, s) for a, s in summaries if a.suite_key == reference.suite_key), None)
        why = "the model stage's winner"
    else:
        ref = next(((a, s) for a, s in summaries if a.current), None)
        why = "the configured arm"
        if ref is None:
            ref = summaries[0]
            why = "the fastest arm of the largest fitting tier; the configured arm has no runs"
    if ref is None:
        return None
    ref_arm, ref_sum = ref
    grade_floor = ref_sum.mean_grade - MARGIN
    completed_floor = ref_sum.completed - runs_per_task
    reasons = [
        f"reference: {ref_arm.label} ({why})",
        f"eligible: mean grade >= {grade_floor:.2f} ({ref_sum.mean_grade:.2f} - {MARGIN:.2f}), "
        f"completed runs >= {completed_floor} ({ref_sum.completed} - {runs_per_task} per task), "
        f"repetition incidents <= {ref_sum.repetitions}",
        "winner: the eligible arm with the lowest suite wall time; "
        "a tie goes to the smaller peak memory",
    ]
    eligible: list[tuple[Arm, ArmSummary]] = []
    for arm, s in summaries:
        problems = []
        if s.mean_grade < grade_floor:
            problems.append(f"grade {s.mean_grade:.2f} < {grade_floor:.2f}")
        if s.completed < completed_floor:
            problems.append(f"completed {s.completed} < {completed_floor}")
        if s.repetitions > ref_sum.repetitions:
            problems.append(f"repetitions {s.repetitions} > {ref_sum.repetitions}")
        line = (
            f"{arm.label}: grade {s.mean_grade:.2f}, completed {s.completed}/{s.runs}, "
            f"wall {s.wall_seconds:.0f} s, repetitions {s.repetitions}, peak {_peak_text(s)}"
        )
        if problems:
            reasons.append(f"{line} -> not eligible: {'; '.join(problems)}")
        else:
            reasons.append(f"{line} -> eligible")
            eligible.append((arm, s))
    # The reference is eligible against itself by construction. It goes
    # first so an exact tie keeps it: a re-run must not flip the model on
    # equal numbers.
    eligible.sort(key=lambda e: e[0] is not ref_arm)
    winner, w = min(
        eligible,
        key=lambda e: (
            e[1].wall_seconds,
            e[1].peak_memory_bytes if e[1].peak_memory_bytes is not None else float("inf"),
        ),
    )
    if winner is ref_arm:
        reasons.append(f"winner: {ref_arm.label} (the reference; nothing eligible is faster)")
    else:
        reasons.append(
            f"winner: {winner.label} ({w.wall_seconds:.0f} s vs {ref_sum.wall_seconds:.0f} s "
            f"for the reference)"
        )
    return FullChoice(
        label=winner.label,
        arm=winner,
        reference_label=ref_arm.label,
        changes=_changes(user, winner, full=True),
        reasons=reasons,
    )
```

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_decide.py tests/test_tune_main.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS. (`decide` now imports `sous.tune.suite.runner`, which imports `sous.worker`; no cycle — `runner` does not import `decide`.)

- [ ] **Step 5: Commit**

```bash
git add src/sous/tune/decide.py tests/test_tune_decide.py
git commit -m "feat(tune): the full rule picks the fastest arm within the suite's quality margin

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: The report shows the suite and the rule

**Files:**
- Modify: `src/sous/tune/report.py`
- Test: `tests/test_tune_report.py` (append)

**Interfaces:**
- Consumes: `ArmSummary`, `FullChoice` (Task 9), `QuickChoice` (PR 1).
- Produces: `render_report(*, hardware, table_age_days, checkpoints, refusals, rows, choice: QuickChoice | FullChoice | None, current_model: str, quick: bool = True, suite: list[ArmSummary] | None = None, tasks: int = 0, runs: int = 0) -> str`. With `quick=False`: the title is `# sous tune`, the throughput rows carry no `[quality untested]` tag, a `## Suite` section follows `## Throughput` (one line per summary), and `## Choice` prints the `FullChoice`'s label and every `reasons` line. A `None` choice in full mode says no arm completed the suite. The quick rendering is byte-for-byte what PR 1 produced.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tune_report.py`)

```python
from sous.tune.decide import ArmSummary, FullChoice


def _summary(label, model=M, grade=0.91, wall=812.0, peak=20 * 2**30):
    return ArmSummary(
        label=label,
        key=(model, "d", 3, False, False),
        model_id=model,
        runs=16,
        completed=16,
        mean_grade=grade,
        wall_seconds=wall,
        repetitions=0,
        malformed=1,
        approvals_denied=2,
        output_tokens=41203,
        peak_memory_bytes=peak,
    )


def _full_report(tmp_path, choice, suite):
    return render_report(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints={M: describe(M, config_fn=lambda m: fx.qwen_27b(), size_fn=lambda m: 1)},
        refusals=[],
        rows=[_row("27B @3"), _row("9B", model="mlx-community/Qwen3.5-9B-MLX-4bit")],
        choice=choice,
        current_model=M,
        quick=False,
        suite=suite,
        tasks=8,
        runs=2,
    )


def test_a_full_report_has_the_suite_section_and_no_quality_untested_tag(tmp_path):
    from sous.tune.arms import Arm

    arm = Arm(
        label="9B",
        config=load_config(tmp_path / "c.toml"),
        model_id="mlx-community/Qwen3.5-9B-MLX-4bit",
        drafter_id="",
        block_size=0,
        window=131072,
        gateway_window=None,
        tier="9b",
        current=False,
    )
    choice = FullChoice(
        label="9B",
        arm=arm,
        reference_label="27B @3",
        changes={"model": {"id": arm.model_id}},
        reasons=["reference: 27B @3 (the configured arm)", "winner: 9B (400 s vs 812 s for the reference)"],
    )
    text = _full_report(tmp_path, choice, [_summary("27B @3"), _summary("9B", model=arm.model_id, wall=400.0, peak=8 * 2**30)])
    assert text.startswith("# sous tune\n")
    assert "[quality untested]" not in text
    assert "## Suite" in text and "(8 tasks x 2 runs per arm" in text
    assert (
        f"  {'27B @3':45s} grade 0.91  completed 16/16  wall 812 s"
        "  repetitions 0  malformed 1  denied 2  output 41203 tok  peak 20.0 GiB"
    ) in text
    assert "## Choice\n\n  9B\n    reference: 27B @3 (the configured arm)\n    winner: 9B" in text


def test_a_full_report_without_a_choice_says_no_arm_completed_the_suite(tmp_path):
    text = _full_report(tmp_path, None, [])
    assert "  (no suite runs)" in text
    assert "cannot recommend a setting: no arm completed the suite" in text


def test_the_quick_report_is_unchanged_by_the_full_fields(tmp_path):
    kwargs = dict(
        hardware=_hardware(tmp_path),
        table_age_days=1,
        checkpoints={},
        refusals=[],
        rows=[_row("27B @3")],
        choice=None,
        current_model=M,
    )
    assert render_report(**kwargs) == render_report(**kwargs, quick=True, suite=None)
    assert render_report(**kwargs).startswith("# sous tune --quick\n")
    assert "## Suite" not in render_report(**kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_report.py -v`
Expected: FAIL — `render_report() got an unexpected keyword argument 'quick'`.

- [ ] **Step 3: Implement** (in `src/sous/tune/report.py`)

Change the import `from sous.tune.decide import QuickChoice` to `from sous.tune.decide import ArmSummary, FullChoice, QuickChoice`. Replace `_row_line` with:

```python
def _row_line(row: BenchRow, current_model: str, *, tag: bool) -> str:
    # In a quick run only the current model's quality is known; the full run
    # measured every model's, so nothing is untested there.
    untested = "  [quality untested]" if tag and row.model_id != current_model else ""
    if not row.ok:
        return f"  {row.label:45s} failed: {row.error}{untested}"
    peak = "-" if row.peak_memory_bytes is None else gib(row.peak_memory_bytes)
    line = (
        f"  {row.label:45s} prefill {_fmt(row.prefill_tps_2k)}/{_fmt(row.prefill_tps_16k)} tok/s"
        f"  decode {_fmt(row.decode_tps_1k)}/{_fmt(row.decode_tps_16k)} tok/s"
        f"  ttft {_fmt(row.ttft_seconds, ' s')}  peak {peak}"
        f"  spread {_fmt(None if row.spread is None else row.spread * 100, '%', 0)}"
        f"  load {_fmt(row.load_seconds, ' s')}{untested}"
    )
    if row.error:
        # A finished measurement whose teardown was refused or failed: the
        # weights stayed resident, and every arm measured after it says so
        # here, not only on the console of the run that hit it.
        line += f"\n      {row.error}"
    return line
```

Add:

```python
def _suite_line(s: ArmSummary) -> str:
    return (
        f"  {s.label:45s} grade {s.mean_grade:.2f}  completed {s.completed}/{s.runs}"
        f"  wall {s.wall_seconds:.0f} s  repetitions {s.repetitions}  malformed {s.malformed}"
        f"  denied {s.approvals_denied}  output {s.output_tokens} tok"
        f"  peak {'-' if s.peak_memory_bytes is None else gib(s.peak_memory_bytes)}"
    )
```

Change `render_report`'s signature and body:

```python
def render_report(
    *,
    hardware: Hardware,
    table_age_days: int,
    checkpoints: dict[str, Checkpoint],
    refusals: list[Refusal],
    rows: list[BenchRow],
    choice: QuickChoice | FullChoice | None,
    current_model: str,
    quick: bool = True,
    suite: list[ArmSummary] | None = None,
    tasks: int = 0,
    runs: int = 0,
) -> str:
    refused = {r.model_id: r for r in refusals if not r.drafter_id}
    out = ["# sous tune --quick" if quick else "# sous tune", "", "## Hardware", ""]
```

… the Hardware and Candidates sections stay as they are; the Throughput lines become `out += [_row_line(r, current_model, tag=quick) for r in rows] or ["  (nothing measured)"]`; after them, before `## Choice`:

```python
    if not quick:
        out += [
            "",
            "## Suite",
            f"  ({tasks} tasks x {runs} runs per arm; grade = mean hidden-grader score in [0, 1]; "
            "wall = the sum of every run's seconds)",
            "",
        ]
        out += [_suite_line(s) for s in suite or []] or ["  (no suite runs)"]
```

and the `## Choice` section becomes:

```python
    out += ["", "## Choice", ""]
    if choice is None and quick:
        out.append(
            f"  cannot recommend a setting: no successful measurement of the configured model "
            f"({current_model}). A quick run never changes the model; if it does not fit this "
            "machine, run the full `sous tune` or set [model].id by hand to one of the "
            "candidates above."
        )
    elif choice is None:
        out.append("  cannot recommend a setting: no arm completed the suite")
    else:
        out.append(f"  {choice.label}")
        out += [f"    {r}" for r in choice.reasons]
        if not choice.changes:
            out.append("    no config change")
    return "\n".join(out) + "\n"
```

The quick-mode "cannot recommend" text loses its "(a later release)" parenthetical; no test asserts that substring (`grep -n "later release" tests/` finds nothing), so nothing else changes.

- [ ] **Step 4: Run the tests and the four CI checks**

Run: `uv run pytest tests/test_tune_report.py tests/test_tune_main.py -v && uv run ty check && uv run ruff check . && uv run ruff format --check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sous/tune/report.py tests/test_tune_report.py
git commit -m "feat(tune): the report shows the suite's numbers and the rule that chose

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: The full run: model stage, winner stage, `--resume`, `--runs`, docs

**Files:**
- Modify: `src/sous/tune/__init__.py`, `src/sous/cli.py`, `README.md`
- Create: `docs/tuning.md`
- Test: `tests/test_tune_main.py` (modify + append), `tests/test_cli.py` (modify one test)

**Interfaces:**
- Consumes: everything above — `model_stage`, `full_decision`, `summarize`, `winner_stage_arms`, `run_suite`, `SuiteRun`, `SuiteOutcome`, `estimate_seconds`, `load_tasks`, `render_report(quick=…, suite=…)`.
- Produces: `main(args, *, …, suite=runner_mod.run_suite, load_tasks=suite_mod.load_tasks, …)` — two new injectable collaborators, both keyword-only with the real one as default like the rest. `sous tune` without `--quick` is the full run; `--runs N` (default 2). Exit codes unchanged: 0 done (applied or not), 1 a failure that leaves rows for `--resume`, 2 refused before anything ran.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tune_main.py`, change `_args` to include `runs=2`:

```python
def _args(**over):
    base = dict(quick=True, models=None, repeat=1, runs=2, resume=None, yes=False, apply=False)
    base.update(over)
    return argparse.Namespace(**base)
```

Delete `test_a_full_run_is_refused_in_this_version`. Add to `_deps` two entries, `suite` and `load_tasks`, driven by new keyword arguments `grades=None, seconds=None, seen=None`:

```python
from sous.tune.suite import SuiteTask
from sous.tune.suite.runner import SuiteOutcome, SuiteRun


def _tasks(tmp_path):
    return [
        SuiteTask(
            name=n,
            path=tmp_path,
            title=n,
            category="bug-fix",
            instructions="x",
            context_files=(),
            verify_commands=(),
            max_turns=1,
            max_minutes=1,
        )
        for n in ("a", "b")
    ]


def _suite(grades, seconds, seen):
    def suite(arm, tasks, *, runs, done, record, scratch, out, **kw):
        seen.append((arm.label, sorted(done)))
        results = []
        for task in tasks:
            for i in range(runs):
                if (task.name, i) in done:
                    continue
                r = SuiteRun(
                    task=task.name,
                    index=i,
                    label=arm.label,
                    model_id=arm.model_id,
                    drafter_id=arm.drafter_id,
                    block_size=arm.block_size,
                    int8_prefill=arm.int8_prefill,
                    greedy=arm.greedy,
                    window=arm.window,
                    state="done",
                    outcome="completed",
                    turns=3,
                    seconds=(seconds or {}).get(arm.label, 10.0),
                    output_tokens=100,
                    malformed=0,
                    repetitions=0,
                    approvals_denied=0,
                    grade=(grades or {}).get(arm.label, 1.0),
                    grade_detail="",
                    error=None,
                    transcript_path=None,
                )
                record(r)
                results.append(r)
        return SuiteOutcome(results, True, None)

    return suite
```

and in `_deps(...)`, add the parameters `grades=None, seconds=None, seen=None` and the two dict entries:

```python
        suite=_suite(grades, seconds, seen if seen is not None else []),
        load_tasks=lambda: _tasks(tmp_path),
```

Then append:

```python
N = "mlx-community/Qwen3.5-9B-MLX-4bit"
NINE = "Qwen3.5-9B-MLX-4bit"
CUR = "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @3"


def test_a_full_run_grades_each_models_fastest_arm_and_can_change_the_model(tmp_path, capsys):
    seen = []
    deps, _ = _deps(
        tmp_path,
        scores={CUR: 30.0, "Qwen3.8-27B-4bit + Qwen3.8-27B-DFlash2 @5": 28.0, NINE: 45.0},
        cached=(M, D, N),
        grades={NINE: 0.9, CUR: 0.92},
        seconds={NINE: 5.0, CUR: 20.0},
        seen=seen,
    )
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    # The model stage: each model's fastest arm, the current one among them.
    assert [label for label, _ in seen[:2]] == [CUR, NINE]
    # The winner stage on the 9B: nax is on and the fixture is affine 4-bit gs64.
    assert [label for label, _ in seen[2:]] == [f"{NINE} + int8 prefill", f"{NINE} greedy"]
    assert "## Suite" in out and "suite: 2 tasks x 2 runs on 2 arm(s)" in out
    assert "roughly" in out and "min from the measured speeds" in out
    assert f'+id = "{N}"' in out
    assert "restart the daemon to apply id, speculative_draft_id" in out
    assert f'id = "{N}"' in (tmp_path / "config.toml").read_text()


def test_the_winner_stage_offers_only_what_the_hardware_allows(tmp_path, capsys):
    seen = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N), seen=seen)
    deps["detect"] = lambda: dataclasses.replace(_hardware(tmp_path), nax=False, nax_reason="pre-M5")
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    assert [label for label, _ in seen if "int8" in label] == []
    assert [label for label, _ in seen if label.endswith(" greedy")]


def test_a_greedy_arm_that_wins_writes_temperature_zero(tmp_path, capsys):
    deps, _ = _deps(
        tmp_path,
        scores={CUR: 30.0, NINE: 20.0},
        cached=(M, D, N),
        seconds={f"{CUR} greedy": 5.0},
    )
    assert main(_args(quick=False, yes=True), config=_cfg(tmp_path), **deps) == 0
    out = capsys.readouterr().out
    assert "+temperature = 0.0" in out and "+id" not in out


def test_resume_skips_suite_runs_already_recorded(tmp_path, capsys):
    seen = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N), seen=seen)
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 0
    first = list(seen)
    run_id = sorted(p.name for p in (tmp_path / "tune").iterdir())[-1]
    seen.clear()
    assert main(_args(quick=False, resume=run_id), config=_cfg(tmp_path), **deps) == 0
    assert first and seen == []  # every (task, run) of every arm is already on disk
    rows = [json.loads(l) for l in (tmp_path / "tune" / run_id / "results.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"bench", "suite"}


def test_a_suite_arm_whose_weights_stay_resident_stops_the_run(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))
    deps["suite"] = lambda arm, tasks, **kw: SuiteOutcome([], False, "unload refused: held by 1 session(s)")
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 1
    out = capsys.readouterr().out
    assert "could not be released (unload refused: held by 1 session(s))" in out
    assert "--resume" in out


def test_the_daemon_is_asked_again_before_every_suite_arm(tmp_path, capsys):
    calls = []
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))
    real = deps["ready"]
    deps["ready"] = lambda port: (calls.append(port), real(port))[1]
    assert main(_args(quick=False), config=_cfg(tmp_path), **deps) == 0
    # once up front, once before the first bench load, once per suite arm
    # (two models, then two winner-stage arms)
    assert len(calls) == 2 + 4


def test_a_quick_run_never_touches_the_suite(tmp_path, capsys):
    deps, _ = _deps(tmp_path, scores={}, cached=(M, D, N))

    def never(*a, **k):
        raise AssertionError("the suite ran in a quick run")

    deps["suite"] = never
    deps["load_tasks"] = never
    assert main(_args(quick=True), config=_cfg(tmp_path), **deps) == 0
    assert "## Suite" not in capsys.readouterr().out
```

(`dataclasses` is already imported at the top of `tests/test_tune_main.py`; add `import json`.) In `tests/test_cli.py::test_tune_subcommand_dispatches_with_its_flags`, add `"--runs", "3"` to the argv and `assert seen["runs"] == 3` alongside the existing assertions.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tune_main.py tests/test_cli.py -k "tune" -v`
Expected: FAIL — `main() got an unexpected keyword argument 'suite'`; the CLI test fails on `--runs`.

- [ ] **Step 3: Implement `main`**

Rewrite `src/sous/tune/__init__.py` so the module reads as follows (the quick path is the existing code, kept verbatim where shown as `…same as before…`; everything else is new):

```python
"""`sous tune`: benchmark this machine, grade the candidates, pick the
settings, show the diff. `--quick` stops after the throughput stage."""

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
from sous.tune import suite as suite_mod
from sous.tune.decide import (
    ArmSummary,
    FullChoice,
    QuickChoice,
    full_decision,
    model_stage,
    quick_decision,
    summarize,
)
from sous.tune.rundir import RunDir
from sous.tune.suite import runner as runner_mod

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
_GIB = 1 << 30
```

`_managed`, `_describe_all`, `_consented`, `_snapshots`, `_manual_candidate`: unchanged. Add:

```python
def _suite_stage(
    stage: list[arms_mod.Arm],
    tasks: list[suite_mod.SuiteTask],
    runs_per_task: int,
    run: RunDir,
    suite_runs: list[runner_mod.SuiteRun],
    *,
    suite: Callable[..., runner_mod.SuiteOutcome],
    ready: Callable[[int], daemon_mod.Readiness],
    user: SousConfig,
    out: Callable[..., None],
) -> str | None:
    """Every arm's (task, run) not already in the run's rows, one arm at a
    time. Returns the text of what must stop the run — a daemon that is no
    longer free, or weights left resident — else None."""
    for arm in stage:
        done = {
            (r.task, r.index)
            for r in suite_runs
            if r.key == arm.suite_key and r.window == arm.window
        }
        if len(done) >= len(tasks) * runs_per_task:
            continue
        # Minutes to hours have passed since the last check: a session or a
        # task that arrived since would load beside the suite's engine.
        readiness = ready(user.server_port)
        if not readiness.ready:
            return f"daemon: {readiness.reason}"
        out(f"  {arm.label}: suite ...")

        def record(r: runner_mod.SuiteRun) -> None:
            run.append("suite", r.as_dict())
            suite_runs.append(r)

        outcome = suite(
            arm,
            tasks,
            runs=runs_per_task,
            done=done,
            record=record,
            scratch=run.path / "suite",
            out=out,
        )
        if outcome.error:
            out(f"  {arm.label}: {outcome.error}")
        if not outcome.released:
            return (
                f"the model could not be released ({outcome.error}); restart the daemon or "
                f"this process and re-run with --resume {run.run_id}"
            )
    return None


def _full_stages(
    user: SousConfig,
    hardware: hw_mod.Hardware,
    checkpoints: dict[str, cand_mod.Checkpoint],
    arms: list[arms_mod.Arm],
    rows: list[bench_mod.BenchRow],
    tasks: list[suite_mod.SuiteTask],
    runs_per_task: int,
    run: RunDir,
    *,
    suite: Callable[..., runner_mod.SuiteOutcome],
    ready: Callable[[int], daemon_mod.Readiness],
    out: Callable[..., None],
) -> tuple[FullChoice | None, list[ArmSummary], str | None]:
    """The model stage, then the winner stage on its choice. Returns the
    choice, the summaries the report shows, and the text of a failure that
    stops the run."""
    stage = model_stage(arms, rows)
    eta = runner_mod.estimate_seconds(stage, rows, tasks=len(tasks), runs=runs_per_task)
    line = f"suite: {len(tasks)} tasks x {runs_per_task} runs on {len(stage)} arm(s)"
    if eta is not None:
        line += f"; roughly {eta / 60:.0f} min from the measured speeds"
    out(line)
    suite_runs = [runner_mod.SuiteRun.from_dict(r) for r in run.rows("suite")]
    stop = _suite_stage(
        stage, tasks, runs_per_task, run, suite_runs, suite=suite, ready=ready, user=user, out=out
    )
    if stop is not None:
        return None, [], stop
    choice = full_decision(user, stage, suite_runs, rows, runs_per_task=runs_per_task)
    extra: list[arms_mod.Arm] = []
    if choice is not None:
        winner = choice.arm
        extra = arms_mod.winner_stage_arms(
            winner, nax=hardware.nax, checkpoint=checkpoints[winner.model_id]
        )
        if extra:
            out(f"winner stage on {winner.label}: " + ", ".join(a.label for a in extra))
            stop = _suite_stage(
                extra,
                tasks,
                runs_per_task,
                run,
                suite_runs,
                suite=suite,
                ready=ready,
                user=user,
                out=out,
            )
            if stop is not None:
                return None, [], stop
            choice = full_decision(
                user,
                [winner, *extra],
                suite_runs,
                rows,
                runs_per_task=runs_per_task,
                reference=winner,
            )
    summaries = [s for a in [*stage, *extra] if (s := summarize(a, suite_runs, rows)) is not None]
    return choice, summaries, None
```

`main`'s signature gains, after `bench=bench_mod.bench_arm,`:

```python
    suite: Callable[..., runner_mod.SuiteOutcome] = runner_mod.run_suite,
    load_tasks: Callable[[], list[suite_mod.SuiteTask]] = suite_mod.load_tasks,
```

Its body: delete the `if not getattr(args, "quick", False): … return EXIT_REFUSED` block and start with `quick = bool(getattr(args, "quick", False))`; keep everything through the bench loop and its `except` as before. Replace the tail — from `choice = quick_decision(user, arms, rows)` through `out(text)` — with:

```python
    choice: QuickChoice | FullChoice | None
    summaries: list[ArmSummary] = []
    tasks: list[suite_mod.SuiteTask] = []
    runs_per_task = getattr(args, "runs", 2)
    if quick or released_failure:
        choice = quick_decision(user, arms, rows) if quick else None
    else:
        tasks = load_tasks()
        choice, summaries, stop = _full_stages(
            user,
            hardware,
            checkpoints,
            arms,
            rows,
            tasks,
            runs_per_task,
            run,
            suite=suite,
            ready=ready,
            out=out,
        )
        if stop is not None:
            out(f"sous tune: {stop}")
            released_failure = True
    text = report_mod.render_report(
        hardware=hardware,
        table_age_days=(date.today() - table.checked).days,
        checkpoints=checkpoints,
        refusals=refusals,
        rows=rows,
        choice=choice,
        current_model=user.model_id,
        quick=quick,
        suite=summaries,
        tasks=len(tasks),
        runs=runs_per_task,
    )
    run.write_text("report.md", text)
    out(text)
```

The rest of `main` (the `released_failure` return, the diff, the apply prompt, the restart note) stays exactly as it is.

- [ ] **Step 4: The CLI**

In `src/sous/cli.py`, the `tune` parser becomes:

```python
    tune = sub.add_parser(
        "tune",
        help="measure this machine, grade the candidates and propose [model] settings "
        "(--quick: throughput only — drafter, block size, window)",
    )
    tune.add_argument(
        "--quick", action="store_true", help="the throughput stage only; never changes the model"
    )
    tune.add_argument(
        "--models", nargs="+", metavar="ID", help="candidate ids instead of the table"
    )
    tune.add_argument("--runs", type=int, default=2, help="suite runs per task (full run)")
    tune.add_argument("--repeat", type=int, default=2, help="attempts per measurement (best wins)")
    tune.add_argument("--resume", metavar="RUN_ID", help="continue a run under ~/.sous/tune")
    tune.add_argument("--yes", action="store_true", help="approve every download and apply")
    tune.add_argument("--apply", action="store_true", help="apply the diff without asking")
```

- [ ] **Step 5: Run the tests and the four CI checks**

Run: `uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: PASS, whole suite.

- [ ] **Step 6: Docs**

`README.md`, the commands block: change the `sous tune --quick` line to two lines:

```
sous tune               # grade the candidates on this machine and propose settings (see Tuning)
sous tune --quick       # the throughput stage only, in about 15 minutes
```

`README.md`, the `## Tuning` section: replace it entirely with:

```markdown
## Tuning

`sous tune` chooses the model and the `[model]` settings for the machine it
runs on, so you never read a tok/s table or a quantization format. It ends
with a report, a unified diff of `~/.sous/config.toml`, and a question:

```
Apply these changes to ~/.sous/config.toml? [y/N]
```

Nothing is written before that yes; a backup is kept beside the config file,
and the tune prints the daemon-restart command every applied change needs
(`sous stop`, then `sous serve`, or `launchctl kickstart -k gui/<uid>/<label>`
for a managed daemon — the daemon reads `[model]` once at startup).

**`sous tune --quick`** (about 15 minutes on an M5 Pro for three fitting
candidates) detects the chip, the Metal working set and whether the GPU has
tensor units, fits every curated candidate to memory (printing the
arithmetic for each one it refuses), asks about every download it would need
one by one, then measures prefill and decode throughput of every arm — a
model, a drafter or none, a block size — through sous's own engine
(`--repeat` sets the attempts per measurement, default 2; the best wins and
the spread is printed). It proposes only the settings that cannot change
what the model says: the drafter, its block size, and a window that fits.
Other models are measured and reported with a "quality untested" label; a
quick run never changes the model.

**`sous tune`** (about three hours on an M5 Pro for three fitting models; the
estimate is printed after the quick stage from the measured speeds) does all
of the above, then grades the candidates: for each model's fastest
quality-neutral arm, and for your current configuration, it runs a suite of
eight mechanical coding tasks — implement a module from its spec, write
tests for one, a docstring sweep, a cross-file rename, a bug fix, a dataclass
from a JSON schema, a CLI flag, a config-format migration — through the real
worker loop, under the real sandbox with the shipped allowlist plus
`python -m unittest`, denying every approval request. Each run is scored by a
hidden grader (`--runs` sets the runs per task, default 2). The rule, printed
in full with every number:

- the **reference** is your current configuration when it fits this machine,
  else the fastest arm of the largest tier that does;
- an arm is **eligible** when its mean grade is within 0.05 of the
  reference's, it completed at least as many runs as the reference minus one
  task's worth, and it looped on a tool no more often;
- the **winner** is the eligible arm with the lowest total suite wall time
  (a tie goes to the smaller memory footprint).

On the winner, the same rule then judges one extra arm per quality-affecting
setting: INT8 prefill (where the tensor units and the checkpoint allow it)
and greedy sampling (`temperature = 0`, which also lets the drafter's
exact-match verify run). A setting lands in the diff only when its own
measured arm is eligible and faster — that is why there is no `--greedy`
flag to understand. The full run may therefore change `[model].id`, the
drafter and block size, the windows, `int8_prefill` and `temperature`.

The daemon is asked to release the model first (`POST /sous/unload`) and
refuses while a `sous claude` session holds it, a task is running or queued,
or a load or unload is under way — the tune waits for none of them, it tells
you, and it asks again before every model it loads. Results (`results.jsonl`
with every bench row and suite run, `hardware.json`, `report.md`, and each
suite run's project and transcript under `suite/`) land in
`~/.sous/tune/<run-id>/`; `--resume <run-id>` continues an interrupted run
from the rows it already has; `--models ID ...` measures ids of your own;
`--yes` answers every prompt for scripted use, `--apply` skips only the final
one. Adding a suite task or a curated candidate is described in
[docs/tuning.md](docs/tuning.md).
```

`README.md`, the `[model]` reference comment: replace the three lines starting `# \`sous tune --quick\` may rewrite` with:

```
# `sous tune --quick` may rewrite speculative_draft_id, speculative_block_size
# and max_context_tokens (and [gateway].max_context_tokens) after showing you
# the diff; the full `sous tune` may also rewrite id, int8_prefill and
# temperature, each only when its suite run earned it.
```

Also in the README's `## Smaller machines` section, change "only the full run (a later release) may propose a model change" wording wherever it appears (grep `later release` in `README.md` and `src/`; there must be none left).

`docs/tuning.md` (new):

```markdown
# Tuning: the suite and the candidate table

`sous tune` measures this machine and grades the candidate models; the README's
Tuning section says what it does and what it may change. This note is for
changing what it measures.

## Adding a suite task

A task is a directory under `src/sous/tune/suite/tasks/<name>/`:

```
task.toml     title, category, instructions, context_files, verify_commands,
              max_turns (default 16), max_minutes (default 10)
project/      the fixture the worker sees, copied to a scratch root per run
grade/        hidden: test_*.py unittest modules, or a grade.py
solution/     the complete solved project, used only by CI to prove the grader
```

Rules the loader and CI enforce (`tests/test_tune_suite.py`):

- `category` is one of implement-from-spec, test-scaffolding, mechanical-sweep,
  cross-file-refactor, bug-fix, codegen, feature-slice; every `context_files`
  entry exists under `project/`; `grade/` holds a `grade.py` or at least one
  `test_*.py`; there are no other keys.
- The grader scores `solution/` at 1.0 and the untouched `project/` at 0.0.
  The easy way to guarantee the second half: write every hidden test so it
  fails before the change — import the new name, use the new flag, read the
  migrated file — and put the "nothing else changed" assertions inside those
  same tests rather than in tests of their own.
- Standard library only, in the fixture, the grader and the solution: an end
  user's machine has nothing else, and the worker's allowlist is the shipped
  one plus `python -m unittest`.
- Hidden `test_*.py` modules run in a subprocess with the worker's project as
  the working directory (`python -m sous.tune.suite.unittests grade/`), so they
  import the worker's modules by bare name; import inside the test methods so
  a missing module fails that test rather than the whole module.
- A `grade.py` defines `grade(project: Path, tests) -> tuple[float, str]`,
  where `tests(cwd, tests_dir=None)` runs unittest modules (`tests_dir`
  defaults to the hidden ones) and returns `(passed, total, detail)`. See
  `tests_for_slugify` (mutation scoring) and `docstring_sweep` (an AST check
  over the behaviour tests).
- Keep a task small enough that a 27B finishes it in a dozen turns: the
  suite's purpose is parity between arms, not a leaderboard.

The fixture trees are excluded from `ty` (their imports resolve only inside a
copied project) and linted and formatted by ruff like everything else.

## Adding a curated candidate

`src/sous/tune/candidates.toml` lists the models `sous tune` measures. A row
is an id, a tier, the drafters known to pair with it and a note; everything
else — bytes, KV cost per token, quantization layout, the exact-verifier and
INT8-prefill eligibility, drafter compatibility — is read from the
checkpoint's `config.json` at run time, so a row never carries a number that
can go stale. Rows are ordered largest tier first; the full run's fallback
reference (when the configured model does not fit) is the first fitting one.

Bump `checked` to the day you last verified the table against the Hub and
set `validated_with` to the mlx-vlm version you ran the suite on. The report
warns when `checked` is more than 90 days old, and `--discover` (a later
release) is what a freshness bot would run.
```

- [ ] **Step 7: Run the four CI checks, then commit**

Run: `uv run pytest -m "not model" && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: PASS.

```bash
git add src/sous/tune/__init__.py src/sous/cli.py README.md docs/tuning.md tests/test_tune_main.py tests/test_cli.py
git commit -m "feat(cli): sous tune grades the candidates and settles the quality-affecting settings

The full run: each model's fastest quality-neutral arm and the current one
through the suite, the rule over their results, then INT8 prefill and greedy
sampling judged on the winner alone; --runs, --resume over suite rows, the
ETA after the quick stage.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: The model-marked runner test

**Files:**
- Modify: `tests/test_tune_model.py` (append)

**Interfaces:**
- Consumes: `run_suite`, `load_tasks`, the tiny model `mlx-community/Qwen3-0.6B-4bit` (text-only, mlx-lm backend, ~350 MB, already used by `e2e_smoke.py` and the PR 1 model test).

- [ ] **Step 1: Write the test** (append to `tests/test_tune_model.py`)

```python
from sous.tune.suite import load_tasks
from sous.tune.suite.runner import run_suite


def test_the_smallest_suite_task_runs_through_the_runner_on_the_tiny_model(tmp_path):
    """Outcome recorded, grade computed, nothing asserted about the grade —
    the 0.6B cannot pass a task; what this proves is the real engine, the
    real sandbox and the grader wired together, and the weights released."""
    cfg = SousConfig(
        data_dir=tmp_path / "d",
        config_path=tmp_path / "c.toml",
        model_id=TINY,
        speculative_draft_id="",
        max_context_tokens=8192,
    )
    arm = Arm(
        label="tiny",
        config=cfg,
        model_id=TINY,
        drafter_id="",
        block_size=0,
        window=8192,
        gateway_window=None,
        tier="test",
        current=True,
    )
    task = next(t for t in load_tasks() if t.name == "docstring_sweep")
    recorded = []
    outcome = run_suite(
        arm,
        [task],
        runs=1,
        done=set(),
        record=recorded.append,
        scratch=tmp_path / "scratch",
        out=lambda *a, **k: None,
    )
    assert outcome.error is None and outcome.released, outcome.error
    assert len(outcome.runs) == 1 and recorded == outcome.runs
    run = outcome.runs[0]
    assert run.state in ("done", "failed")
    assert 0.0 <= run.grade <= 1.0 and run.grade_detail
    assert run.turns >= 1 and run.seconds > 0 and run.output_tokens > 0
    assert run.transcript_path and Path(run.transcript_path).is_file()
```

(add `from pathlib import Path` to the module's imports.)

- [ ] **Step 2: Run it locally**

Run: `uv run pytest -m model tests/test_tune_model.py -v`
Expected: PASS in a minute or two on this machine (the 0.6B is cached from PR 1's model test). The non-model job never runs it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tune_model.py
git commit -m "test(tune): one suite task through the runner on the tiny model

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Live gate (manual, after CI is green; the spec's acceptance)

Not tasks for an implementer — the maintainer's checks over ssh on the M5 Pro and locally on the M2 Air, in `~/code/personal/sous-remote` on the M5 (never the maintainer's working copy), with the daemon built from this branch:

1. `sous tune --quick` still passes on both Macs exactly as PR 1's gate did (no suite, no change in what it proposes).
2. `sous tune` on the M5 Pro with every download declined (the 27B-4bit + DFlash2 and, if cached, the oQ4): the model stage runs the current arm and each cached model's fastest arm; the reference is eligible against itself; the winner stage runs the int8 and greedy arms on the winner; the report prints the rule with every number; the diff is what the rule says. Expect about three hours; interrupt it once mid-suite and `--resume` it to prove the suite rows are skipped.
3. `sous tune --models mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.8-27B-mxfp8` on the M5 Pro, with the mxfp8 download approved (~29 GB): the spec's reproduction of the 2026-08-29 finding — the affine 4-bit is not worse than mxfp8 on the tool loop in whichever direction the margin allows.
4. `sous tune` on the M2 Air: the 27B refused with the arithmetic, the 9B (cached) offered and graded; apply, restart the daemon, and run one delegated task end to end on the applied config.
5. In every run: `mx.get_active_memory()` back at the baseline between arms (the console's "released" lines), no two models ever resident, the daemon untouched while it held a session.

## Self-review notes (already applied while writing)

- Every suite task's hidden tests fail on the untouched fixture, so the parametrized proof needs no per-task exception.
- `Arm.key` (bench rows) is unchanged; `suite_key` extends it. `summarize` joins the two by `key`, so a winner-stage arm inherits its bench peak from the winner's row.
- The quick path is untouched: `quick_decision`, the quick report and every PR 1 test keep passing; only the "full run is refused" test goes.
- `decide` now imports `sous.tune.suite.runner`, which imports `sous.worker`; `runner` imports `bench` and `arms`, never `decide` or the package `__init__` — no cycle.
- No `--greedy` flag anywhere; the greedy arm exists only inside the winner stage.
