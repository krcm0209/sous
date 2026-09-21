"""How a suite run is scored: hidden unittest modules run against the
candidate's project in a subprocess, or a task's own grade.py over that same
runner for the categories a pass count cannot express."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from sous.tune.suite import SuiteTask, spawn

# Enough for the runner's counts line, failed test ids included, or the last
# line of a traceback.
_TAIL_BYTES = 256 * 1024


@dataclass(frozen=True)
class Grade:
    score: float
    detail: str


def run_tests(cwd: Path, tests_dir: Path, *, python: Path, timeout: float) -> tuple[int, int, str]:
    """(passed, total, detail) of the test_*.py modules under `tests_dir`,
    run with `cwd` importable. A timeout, a crash or output that is not the
    runner's JSON is (0, 0, why): the candidate's code is untrusted and a
    project that hangs must not hang the tune."""
    argv = [str(python), "-m", "sous.tune.suite.unittests", str(tests_dir)]
    try:
        with spawn.bounded(argv, cwd=cwd, timeout=timeout, merged=False) as run:
            if run.stopped == "timeout":
                return 0, 0, f"tests timed out after {timeout:.0f} s"
            if run.stopped == "output":
                return (
                    0,
                    0,
                    f"test runner stopped: more than {spawn.MAX_SPOOL_BYTES} bytes of output",
                )
            returncode = run.returncode
            counts_line = _last_line(run.stdout)
            error_line = _last_line(run.stderr)
    except OSError as e:
        return 0, 0, f"could not run the tests: {e}"
    try:
        counts = json.loads(counts_line) if counts_line else {}
        passed, total = int(counts["passed"]), int(counts["total"])
        failed = [str(f) for f in counts.get("failed", [])]
    except ValueError, KeyError, TypeError:
        return 0, 0, f"test runner exited {returncode}: {error_line or 'no output'}"
    detail = f"{passed}/{total} hidden tests passed"
    if failed:
        detail += "; failed: " + ", ".join(failed)
    return passed, total, detail


def _last_line(spool: IO[bytes]) -> str:
    """The last non-blank line of a spool, read from its tail: the runner's
    counts are its final line, and a traceback's last line names the error."""
    size = spool.seek(0, os.SEEK_END)
    spool.seek(max(0, size - _TAIL_BYTES))
    lines = spool.read().decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


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
    """The task's score over the candidate's finished project, in [0, 1]. With
    a grade.py the task decides — it gets the project and a test runner
    whose default test directory is the hidden one — else the hidden
    modules' pass fraction. A grader that raises or returns something that
    is not a (score, detail) pair scores zero and says so: the run is a
    result either way, and the suite goes on."""

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
        grade = Grade(min(1.0, max(0.0, float(score))), str(detail))
    except Exception as e:  # noqa: BLE001 — a grader bug is a zero, named, not a dead run
        return Grade(0.0, f"grader failed: {type(e).__name__}: {e}")
    return grade
