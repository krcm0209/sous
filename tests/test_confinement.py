import os
from pathlib import Path

import pytest

from sous.toolexec import ChangedFile, PathViolation, ToolExecutor, resolve_confined


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "src").mkdir()
    (tmp_path / "proj" / "src" / "a.py").write_text("x = 1\ny = 2\n")
    (tmp_path / "proj" / ".git").mkdir()
    (tmp_path / "proj" / ".git" / "config").write_text("[core]\n")
    (tmp_path / "outside.txt").write_text("secret")
    return tmp_path / "proj"


@pytest.fixture()
def ex(root: Path, tmp_path: Path) -> ToolExecutor:
    return ToolExecutor(root, tmp_path / "config.toml")


# --- resolve_confined: the security core ---

def test_relative_path_ok(root: Path):
    assert resolve_confined(root, "src/a.py", False) == root / "src" / "a.py"

def test_dotdot_escape_denied(root: Path):
    with pytest.raises(PathViolation):
        resolve_confined(root, "../outside.txt", False)

def test_absolute_outside_denied(root: Path, tmp_path: Path):
    with pytest.raises(PathViolation):
        resolve_confined(root, str(tmp_path / "outside.txt"), False)

def test_absolute_inside_ok(root: Path):
    assert resolve_confined(root, str(root / "src/a.py"), False).name == "a.py"

def test_symlink_escape_denied(root: Path, tmp_path: Path):
    os.symlink(tmp_path / "outside.txt", root / "link.txt")
    with pytest.raises(PathViolation):
        resolve_confined(root, "link.txt", False)

def test_symlinked_dir_escape_denied(root: Path, tmp_path: Path):
    os.symlink(tmp_path, root / "updir")
    with pytest.raises(PathViolation):
        resolve_confined(root, "updir/outside.txt", False)

def test_git_write_denied_but_read_ok(root: Path):
    with pytest.raises(PathViolation):
        resolve_confined(root, ".git/hooks/pre-commit", True)
    assert resolve_confined(root, ".git/config", False)

def test_prefix_sibling_denied(root: Path, tmp_path: Path):
    (tmp_path / "projX").mkdir()
    (tmp_path / "projX" / "f.txt").write_text("no")
    with pytest.raises(PathViolation):
        resolve_confined(root, str(tmp_path / "projX" / "f.txt"), False)


# --- adversarial tests: confirmed exploits ---

def test_dotdot_in_unresolved_tail_denied(ex: ToolExecutor, tmp_path: Path):
    """C1: x/../../pwned.txt escapes via unresolved tail."""
    with pytest.raises(PathViolation):
        ex.write_file("x/../../pwned.txt", "X")
    # Verify no file was created outside root
    assert not (tmp_path / "pwned.txt").exists()

def test_dotdot_before_git_denied(root: Path):
    """C2: q/../.git/hooks/pre-commit bypasses first-part check."""
    with pytest.raises(PathViolation):
        resolve_confined(root, "q/../.git/hooks/pre-commit", for_write=True)

def test_git_case_insensitive_write_denied(root: Path):
    """C3: .GIT (uppercase) bypasses exact string match on case-insensitive FS."""
    with pytest.raises(PathViolation):
        resolve_confined(root, ".GIT/hooks/evil", for_write=True)

def test_nested_git_write_denied(root: Path):
    """I4: sub/.git/hooks/x (nested at any depth) must be rejected."""
    with pytest.raises(PathViolation):
        resolve_confined(root, "sub/.git/hooks/x", for_write=True)

def test_symlinked_parent_escape_denied(ex: ToolExecutor, tmp_path: Path):
    """Symlinked parent dir pointing outside, with non-existent tail."""
    os.symlink(tmp_path, ex.project_root / "out")
    with pytest.raises(PathViolation):
        ex.write_file("out/newfile.txt", "X")

def test_git_read_still_allowed(root: Path):
    """.git/config read is allowed (only writes are blocked)."""
    result = resolve_confined(root, ".git/config", for_write=False)
    assert result.exists()


# --- file tools ---

def test_read_file_line_numbers(ex: ToolExecutor):
    out = ex.read_file("src/a.py")
    assert "1\tx = 1" in out and "2\ty = 2" in out

def test_read_offset_limit(ex: ToolExecutor):
    out = ex.read_file("src/a.py", offset=1, limit=1)
    assert "y = 2" in out and "x = 1" not in out

def test_write_creates_and_tracks(ex: ToolExecutor):
    ex.write_file("src/new.py", "z = 3\n")
    changes = ex.changed_files()
    assert any(c.path == "src/new.py" and c.kind == "created" for c in changes)

def test_write_modify_tracks_before_hash(ex: ToolExecutor):
    ex.write_file("src/a.py", "x = 9\n")
    [c] = [c for c in ex.changed_files() if c.path == "src/a.py"]
    assert c.kind == "modified" and c.before_sha and c.after_sha != c.before_sha

def test_edit_exact_unique(ex: ToolExecutor):
    ex.edit_file("src/a.py", "x = 1", "x = 42")
    assert "x = 42" in (ex.project_root / "src/a.py").read_text()

def test_edit_nonunique_rejected(ex: ToolExecutor):
    ex.write_file("src/dup.py", "a\na\n")
    out = ex.edit_file("src/dup.py", "a", "b")
    assert "2 matches" in out  # error text, not exception

def test_edit_missing_rejected(ex: ToolExecutor):
    out = ex.edit_file("src/a.py", "nope", "x")
    assert "0 matches" in out

def test_glob_and_grep(ex: ToolExecutor):
    assert "src/a.py" in ex.glob("**/*.py")
    assert "src/a.py:1" in ex.grep("x = 1")

def test_grep_skips_git_dir(ex: ToolExecutor):
    assert ".git" not in ex.grep("core")

def test_output_truncated(ex: ToolExecutor):
    ex.write_file("big.txt", "line\n" * 20000)
    out = ex.read_file("big.txt", limit=20000)
    assert len(out) <= 16_000 + len("\n[truncated]") and out.endswith("[truncated]")
