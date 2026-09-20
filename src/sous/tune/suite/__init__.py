"""The graded suite: mechanical coding tasks the candidate runs for real, each
with a hidden grader and a reference solution. A task is a directory —
task.toml, project/ (what the candidate sees), grade/ (what scores it),
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
