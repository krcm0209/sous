"""The suite's tasks: what ships, what a task.toml must say, and that the
fixture trees resolve from the installed package."""

import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from sous.tune.suite import CATEGORIES, SuiteTask, load_task, load_tasks
from sous.tune.suite.grading import Grade, grade_task, run_tests

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
        or (
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_add(self):\n        from calc import add\n\n"
            "        self.assertEqual(add(1, 2), 3)\n\n"
            "    def test_add_negative(self):\n        from calc import add\n\n"
            "        self.assertEqual(add(-1, 1), 0)\n"
        )
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
    bad_project = _project(tmp_path / "b", "def add(a, b):\n    return abs(a) + b\n")
    bad = grade_task(task, bad_project, timeout=60)
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


def test_a_grader_that_returns_garbage_is_a_zero_with_the_error(tmp_path):
    task = _suite_task(tmp_path, "def grade(project, tests):\n    return None, 'looks fine'\n")
    g = grade_task(task, _project(tmp_path), timeout=60)
    assert g.score == 0.0 and "grader failed: TypeError" in g.detail


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
