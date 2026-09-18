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
    except ValueError, KeyError, TypeError, IndexError:
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
            cwd,
            tests_dir if tests_dir is not None else task.grade_dir,
            python=python,
            timeout=timeout,
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
