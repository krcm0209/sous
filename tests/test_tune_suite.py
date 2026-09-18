"""The suite's tasks: what ships, what a task.toml must say, and that the
fixture trees resolve from the installed package."""

from importlib.resources import files
from pathlib import Path

import pytest

from sous.tune.suite import CATEGORIES, SuiteTask, load_task, load_tasks

_MINIMAL = 'title = "t"\ncategory = "bug-fix"\ninstructions = "do it"\n'


def _task_dir(tmp_path, toml=_MINIMAL, *, project=("a.py",), grade=("test_a.py",), solution=True):
    import shutil

    d = tmp_path / "task_x"
    d.mkdir(exist_ok=True)
    (d / "task.toml").write_text(toml)
    # Clean up and recreate subdirectories to handle multiple calls with different parameters
    for sub in ("project", "grade", "solution"):
        sub_path = d / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
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
