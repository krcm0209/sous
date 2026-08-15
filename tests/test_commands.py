import sys
from pathlib import Path

import pytest

from sous.toolexec import ToolExecutor, command_allowed, scrubbed_env


def test_allowlist_token_matching():
    allow = [["npm", "test"], ["pytest"]]
    assert command_allowed(["npm", "test"], allow)
    assert command_allowed(["npm", "test", "--workspaces"], allow)
    assert command_allowed(["pytest", "tests/x.py", "-v"], allow)
    assert not command_allowed(["npm", "testx"], allow)
    assert not command_allowed(["npm", "install"], allow)
    assert not command_allowed(["npm"], allow)  # shorter than entry
    assert not command_allowed([], allow)


def test_scrubbed_env(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", "x")
    monkeypatch.setenv("AWS_SECRET", "x")
    monkeypatch.setenv("DB_PASSWORD", "x")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = scrubbed_env()
    assert env["PATH"] == "/usr/bin"
    assert "MY_API_TOKEN" not in env
    assert "AWS_SECRET" not in env
    assert "DB_PASSWORD" not in env


@pytest.fixture()
def ex(tmp_path: Path) -> ToolExecutor:
    root = tmp_path / "proj"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[commands]\nallowlist = ["/bin/echo", "true"]\n')
    return ToolExecutor(root, cfg)


def test_allowlisted_command_runs(ex: ToolExecutor):
    out = ex.run_command("/bin/echo hello")
    assert "exit code 0" in out and "hello" in out


def test_shell_metacharacters_inert(ex: ToolExecutor, tmp_path: Path):
    out = ex.run_command("/bin/echo hi; touch pwned")
    assert "exit code 0" in out
    assert "hi; touch pwned" in out            # echoed literally
    assert not (tmp_path / "proj" / "pwned").exists()


def test_denied_without_hook(ex: ToolExecutor):
    out = ex.run_command("rm -rf /")
    assert out.startswith("command denied")


def test_hook_approves(ex: ToolExecutor):
    out = ex.run_command("/usr/bin/printf ok", approval=lambda cmd: True)
    assert "exit code 0" in out and "ok" in out


def test_hook_denies(ex: ToolExecutor):
    out = ex.run_command("/usr/bin/printf no", approval=lambda cmd: False)
    assert out.startswith("command denied")


def test_allowlist_hot_reload(ex: ToolExecutor):
    assert ex.run_command("/usr/bin/true").startswith("command denied")
    ex.config_path.write_text('[commands]\nallowlist = ["/usr/bin/true"]\n')
    assert "exit code 0" in ex.run_command("/usr/bin/true")


def test_timeout(ex: ToolExecutor):
    ex.config_path.write_text('[commands]\nallowlist = ["/bin/sleep"]\n')
    out = ex.run_command("/bin/sleep 5", timeout=1)
    assert "timed out" in out


def test_empty_allowlist_entry_does_not_match(ex: ToolExecutor):
    """Empty allowlist entry must not match any command (prevent allow-all gate)."""
    assert not command_allowed(["rm", "-rf", "/"], [[], ["npm", "test"]])


def test_empty_allowlist_entry_in_config(ex: ToolExecutor):
    """End-to-end: empty entry in config cannot bypass security gate."""
    ex.config_path.write_text('[commands]\nallowlist = ["", "/bin/echo"]\n')
    # Empty entry should not allow /usr/bin/true
    assert ex.run_command("/usr/bin/true").startswith("command denied")
    # But /bin/echo should still be allowed (non-empty entry)
    out = ex.run_command("/bin/echo hi")
    assert "exit code 0" in out


# --- B3: command-induced file changes must be recorded ---

@pytest.fixture()
def pyex(tmp_path: Path) -> ToolExecutor:
    """Executor whose allowlist contains the current Python interpreter, the
    stand-in for the default-allowlisted formatters (black, prettier, ruff)
    whose whole job is modifying files."""
    root = tmp_path / "proj"
    root.mkdir()
    # config in its own dir: data_dir defaults to config_path.parent and is
    # write-protected, so it must not be an ancestor of the project root.
    cfg = tmp_path / "sous-home" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(f'[commands]\nallowlist = ["{sys.executable}"]\n')
    return ToolExecutor(root, cfg)


def test_command_created_file_recorded(pyex: ToolExecutor):
    out = pyex.run_command(
        f"{sys.executable} -c \"open('gen.txt', 'w').write('made-by-command')\"")
    assert "exit code 0" in out
    changes = {c.path: c for c in pyex.changed_files()}
    assert "gen.txt" in changes
    assert changes["gen.txt"].kind == "created"
    assert changes["gen.txt"].before_sha is None
    assert changes["gen.txt"].after_sha


def test_command_modified_file_recorded(pyex: ToolExecutor):
    (pyex.project_root / "fmt.py").write_text("x=1\n")
    out = pyex.run_command(
        f"{sys.executable} -c \"open('fmt.py', 'w').write('x = 1  # formatted')\"")
    assert "exit code 0" in out
    changes = {c.path: c for c in pyex.changed_files()}
    assert "fmt.py" in changes
    assert changes["fmt.py"].kind == "modified"
    # the pre-command snapshot is stat-only: prior content hash is unknown
    assert changes["fmt.py"].before_sha is None


def test_command_edit_refreshes_stale_hash_of_tracked_file(pyex: ToolExecutor):
    """A file the worker wrote and a formatter then rewrote must carry the
    formatter's content hash, not the stale pre-command one."""
    pyex.write_file("t.txt", "one")
    [before] = pyex.changed_files()
    pyex.run_command(
        f"{sys.executable} -c \"open('t.txt', 'w').write('two-formatted')\"")
    [after] = [c for c in pyex.changed_files() if c.path == "t.txt"]
    assert after.kind == "created"          # original kind survives
    assert after.after_sha != before.after_sha  # hash refreshed


def test_command_changes_in_git_dir_not_recorded(pyex: ToolExecutor):
    (pyex.project_root / ".git").mkdir()
    pyex.run_command(
        f"{sys.executable} -c \"open('.git/junk', 'w').write('x')\"")
    assert all(".git" not in c.path for c in pyex.changed_files())
