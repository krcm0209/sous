import os
from pathlib import Path

import pytest

from sous.toolexec import PathViolation, ToolExecutor, resolve_confined


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
    # The config lives in its own data dir (like ~/.sous), NOT directly in
    # the parent of the project root: data_dir defaults to config_path.parent
    # and is write-protected, so placing config.toml at tmp_path would shield
    # the whole tmp tree including the project root itself.
    return ToolExecutor(root, tmp_path / "sous-home" / "config.toml")


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


def test_git_symlink_write_denied(tmp_path: Path):
    """A1 (bypass #3): `.git` as a symlink to a directory inside the project.
    Resolution strips the `.git` name (resolved path is realgit/...), so a
    resolved-only guard lets the write land in the real git dir. The lexical
    path still names `.git` and must be checked too."""
    proj = tmp_path / "proj-gitlink"
    (proj / "realgit").mkdir(parents=True)
    os.symlink(proj / "realgit", proj / ".git")
    ex = ToolExecutor(proj, tmp_path / "sous-home" / "config.toml")
    with pytest.raises(PathViolation):
        ex.write_file(".git/hooks/pre-commit", "INJECTED")
    assert not (proj / "realgit" / "hooks" / "pre-commit").exists()


def test_git_symlink_case_variant_write_denied(tmp_path: Path):
    """A1 + C3 combined: a case variant of the `.git` symlink must be denied
    too (case-folded lexical check)."""
    proj = tmp_path / "proj-gitlink-case"
    (proj / "realgit").mkdir(parents=True)
    os.symlink(proj / "realgit", proj / ".git")
    with pytest.raises(PathViolation):
        resolve_confined(proj, ".GIT/hooks/evil", for_write=True)


def test_nested_git_symlink_write_denied(tmp_path: Path):
    """A1 at depth: sub/.git as a symlink is denied like top-level .git."""
    proj = tmp_path / "proj-gitlink-nested"
    (proj / "sub" / "realgit").mkdir(parents=True)
    os.symlink(proj / "sub" / "realgit", proj / "sub" / ".git")
    with pytest.raises(PathViolation):
        resolve_confined(proj, "sub/.git/hooks/x", for_write=True)


# --- C1: the sous data dir is write-protected inside the sandbox ---


@pytest.fixture()
def home_ex(tmp_path: Path) -> ToolExecutor:
    """Executor whose project root CONTAINS the sous data dir (the $HOME
    delegation case): data_dir defaults to config_path.parent."""
    home = tmp_path / "home"
    (home / ".sous" / "tasks" / "x").mkdir(parents=True)
    (home / ".sous" / "config.toml").write_text('[commands]\nallowlist = ["pytest"]\n')
    (home / ".sous" / "tasks.db").write_text("db-bytes")
    (home / ".sous" / "tasks" / "x" / "transcript.jsonl").write_text('{"event":"tool"}\n')
    (home / "notes.txt").write_text("plain project file")
    return ToolExecutor(home, home / ".sous" / "config.toml")


def test_resolve_confined_protected_write_denied(root: Path):
    guarded = root / "src"
    with pytest.raises(PathViolation):
        resolve_confined(root, "src/a.py", True, protected=(guarded.resolve(),))
    # reads of a protected path stay allowed
    assert resolve_confined(root, "src/a.py", False, protected=(guarded.resolve(),))


def test_data_dir_config_write_denied_and_unchanged(home_ex: ToolExecutor):
    before = (home_ex.project_root / ".sous" / "config.toml").read_text()
    with pytest.raises(PathViolation):
        home_ex.write_file(".sous/config.toml", '[commands]\nallowlist = ["bash"]\n')
    assert (home_ex.project_root / ".sous" / "config.toml").read_text() == before


def test_data_dir_tasksdb_write_denied(home_ex: ToolExecutor):
    with pytest.raises(PathViolation):
        home_ex.write_file(".sous/tasks.db", "junk")
    assert (home_ex.project_root / ".sous" / "tasks.db").read_text() == "db-bytes"


def test_data_dir_transcript_write_denied(home_ex: ToolExecutor):
    with pytest.raises(PathViolation):
        home_ex.write_file(".sous/tasks/x/transcript.jsonl", "[]")
    assert (
        home_ex.project_root / ".sous" / "tasks" / "x" / "transcript.jsonl"
    ).read_text() == '{"event":"tool"}\n'


def test_data_dir_edit_denied(home_ex: ToolExecutor):
    with pytest.raises(PathViolation):
        home_ex.edit_file(".sous/config.toml", "pytest", "bash")


def test_data_dir_read_still_allowed(home_ex: ToolExecutor):
    assert "allowlist" in home_ex.read_file(".sous/config.toml")


def test_write_outside_data_dir_still_works(home_ex: ToolExecutor):
    home_ex.write_file("notes.txt", "updated")
    assert (home_ex.project_root / "notes.txt").read_text() == "updated"


def test_explicit_data_dir_overrides_default(root: Path, tmp_path: Path):
    (root / "sub").mkdir()
    ex = ToolExecutor(root, tmp_path / "config.toml", data_dir=root / "sub")
    with pytest.raises(PathViolation):
        ex.write_file("sub/f.txt", "x")


def _fs_is_case_insensitive(tmp_path: Path) -> bool:
    probe = tmp_path / "sous_case_probe_x"
    probe.write_text("x")
    hit = (tmp_path / "sous_case_probe_X").exists()
    probe.unlink()
    return hit


def test_data_dir_case_variant_write_denied(tmp_path: Path):
    """C1 (case bypass): on a case-insensitive FS (APFS/HFS+), a project root
    given in a different case than the real dir must NOT let the worker write
    the real ~/.sous/config.toml. Path.resolve() preserves case on macOS, so
    case-sensitive is_relative_to would miss it while the write lands on the
    real inode."""
    if not _fs_is_case_insensitive(tmp_path):
        pytest.skip("requires a case-insensitive filesystem (APFS/HFS+)")
    home = tmp_path / "home"
    (home / ".sous").mkdir(parents=True)
    real_cfg = home / ".sous" / "config.toml"
    real_cfg.write_text('[commands]\nallowlist = ["pytest"]\n')
    before = real_cfg.read_bytes()
    # project_root given with a case variant of the real "home" directory
    ex = ToolExecutor(tmp_path / "HOME", home / ".sous" / "config.toml", data_dir=home / ".sous")
    with pytest.raises(PathViolation):
        ex.write_file(".sous/config.toml", '[commands]\nallowlist = ["bash"]\n')
    assert real_cfg.read_bytes() == before  # real file byte-unchanged


# --- I1: glob/grep must not read through escaping symlinks ---


def test_grep_does_not_follow_escaping_symlink(ex: ToolExecutor, tmp_path: Path):
    (tmp_path / "secret.txt").write_text("TOPSECRET-MARKER")
    os.symlink(tmp_path / "secret.txt", ex.project_root / "leak.txt")
    assert "TOPSECRET-MARKER" not in ex.grep("TOPSECRET-MARKER")
    assert "src/a.py:1" in ex.grep("x = 1")  # in-root grep still works


def test_glob_does_not_list_escaping_symlink(ex: ToolExecutor, tmp_path: Path):
    (tmp_path / "secret.txt").write_text("TOPSECRET-MARKER")
    os.symlink(tmp_path / "secret.txt", ex.project_root / "leak.txt")
    hits = ex.glob("**/*")
    assert "leak.txt" not in hits
    assert "src/a.py" in hits  # in-root glob still works


def test_grep_skips_escaping_symlinked_dir(ex: ToolExecutor, tmp_path: Path):
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "s.txt").write_text("TOPSECRET-MARKER")
    os.symlink(tmp_path / "vault", ex.project_root / "vaultlink")
    assert "TOPSECRET-MARKER" not in ex.grep("TOPSECRET-MARKER", "vaultlink/*")


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


def test_read_file_window_from_large_file_is_memory_bounded(ex: ToolExecutor):
    """E1: read_file must stream. A small window from a large file must not
    materialize the whole file — read_text().splitlines() spiked peak RSS
    from 27 MB to 638 MB on a 143 MB file before the fix."""
    import resource
    import sys

    big = ex.project_root / "big_stream.txt"
    row = "y" * 90 + "\n"
    with big.open("w") as f:
        chunk = row * 1000  # ~91 KB — written in pieces, never held whole
        for _ in range(440):  # ~40 MB, 440_000 lines total
            f.write(chunk)
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: B on macOS, KB on Linux
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    out = ex.read_file("big_stream.txt", offset=200_000, limit=5)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    lines = out.splitlines()
    assert lines == [f"{200_000 + i}\t" + "y" * 90 for i in range(1, 6)]
    # Generous margin, but far below the ~40 MB the whole file would cost.
    assert after - before < 20 * 1024 * 1024
