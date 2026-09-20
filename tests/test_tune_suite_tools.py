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
