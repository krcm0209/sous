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


def test_glob_and_grep_never_reach_past_the_root(tmp_path: Path):
    (tmp_path / "secret.txt").write_text("TOKEN=hunter2\n")
    t = _tools(tmp_path)
    t.write_file("a.py", "TOKEN = None\n")
    assert "secret" not in t.glob("../*")
    assert t.grep("TOKEN", "../*") == "(no matches)"
    assert t.grep("TOKEN") == "a.py:1:TOKEN = None"


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


def test_a_blank_verify_command_accepts_nothing(tmp_path: Path):
    t = _tools(tmp_path, verify_commands=(" ",))
    assert t.run_command("rm -rf .") == "command denied (not allowlisted): rm -rf ."
    assert t.denied == 1


def test_the_cd_idiom_runs_its_command_from_the_root(tmp_path: Path):
    """Local models write `cd <project> && <cmd>` whatever the prompt says;
    the cd is dropped, never a cwd, and anything past one command is refused."""
    verify = f"{sys.executable} -c 'print(7)'"
    t = _tools(tmp_path, verify_commands=(verify,))
    t.write_file("pkg/__init__.py", "")
    assert t.run_command(f"cd {t.root} && {verify}") == "exit code 0\n7\n"
    assert t.run_command(f"cd pkg && {verify}") == "exit code 0\n7\n"
    guidance = "command rejected: run a single command from the project root"
    assert t.run_command("cd").startswith(guidance)
    assert t.run_command(f"cd {t.root} && {verify} && echo x").startswith(guidance)
    assert t.run_command(f"cd .. && {verify}") == (
        "command rejected: cd target .. is outside the project"
    )
    assert t.run_command(f"cd missing && {verify}") == (
        "command rejected: cd target missing is not a directory in the project"
    )
    assert t.run_command(f"cd {t.root} && rm -rf .").startswith("command denied")
    assert t.denied == 1


def test_a_command_that_reads_stdin_gets_eof(tmp_path: Path):
    reader = f"{sys.executable} -c 'import sys; print(len(sys.stdin.read()))'"
    t = _tools(tmp_path, verify_commands=(reader,))
    assert t.run_command(reader) == "exit code 0\n0\n"


def test_a_command_past_its_timeout_is_reported_not_raised(tmp_path: Path):
    sleep = f"{sys.executable} -c 'import time; time.sleep(5)'"
    t = _tools(tmp_path, verify_commands=(sleep,))
    assert t.run_command(sleep, timeout=0.2) == f"command timed out after 0.2s: {sleep}"


def test_command_output_keeps_the_verdict_by_capping_head_and_tail(tmp_path: Path):
    # A verify command whose stdout is long keeps both head (for context) and
    # tail (for verdict): "3 failed" or similar is at the end.
    verdict = f"{sys.executable} -c \"print('x' * 40000); print('VERDICT')\""
    t = _tools(tmp_path, verify_commands=(verdict,))
    result = t.run_command(verdict)
    assert result.startswith("exit code 0")
    assert "[... " in result and " characters elided ...]" in result
    assert "VERDICT" in result and result.rstrip().endswith("VERDICT")
    # Capped size: two halves + marker + newlines
    from sous.tune.suite.tools import MAX_TOOL_OUTPUT

    assert len(result) <= MAX_TOOL_OUTPUT + 100  # marker + newlines


def test_command_with_odd_bytes_does_not_raise(tmp_path: Path):
    # A verify command that writes bytes the locale codec cannot decode
    # should not raise UnicodeDecodeError; instead, it should return a
    # tool result with the decodable parts and replacement characters.
    odd = f"{sys.executable} -c \"import sys; sys.stdout.buffer.write(b'\\xff\\xfe ok\\n')\""
    t = _tools(tmp_path, verify_commands=(odd,))
    result = t.run_command(odd)
    assert result.startswith("exit code 0")
    assert "ok" in result
