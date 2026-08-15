"""Worker tool execution: path-confined file ops and a no-shell command runner."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sous.config import current_allowlist

MAX_TOOL_OUTPUT = 16_000

ApprovalHook = Callable[[str], bool]

_ENV_PASS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")


def command_allowed(argv: list[str], allowlist: list[list[str]]) -> bool:
    """Check if argv is allowlisted by leading-token equality.

    Skips falsy entries to prevent empty allowlist entries from matching all commands.
    """
    return any(
        len(argv) >= len(entry) and argv[: len(entry)] == entry
        for entry in allowlist if entry
    ) if argv else False


def scrubbed_env() -> dict[str, str]:
    """Return an environment restricted to a strict pass-list.

    The pass-list IS the secret-scrubbing mechanism: anything not named in
    _ENV_PASS (tokens, keys, cloud credentials, ...) is simply dropped.
    """
    return {k: v for k, v in os.environ.items() if k in _ENV_PASS}


class PathViolation(Exception):
    pass


def _is_within(path: Path, ancestor: Path) -> bool:
    """True if `path` equals or lives inside `ancestor`, folding case so a
    case-variant path cannot slip past on a case-insensitive filesystem
    (APFS/HFS+ are the product's target). Path.resolve() preserves case on
    macOS, and os.path.normcase is a no-op on POSIX, so we fold each path
    component explicitly rather than relying on either. Callers should pass
    already-.resolve()d paths."""
    p = [part.lower() for part in path.parts]
    a = [part.lower() for part in ancestor.parts]
    return len(p) >= len(a) and p[:len(a)] == a


def resolve_confined(project_root: Path, candidate: str, for_write: bool,
                     *, protected: tuple[Path, ...] = ()) -> Path:
    root = project_root.resolve()
    raw = Path(candidate)
    joined = raw if raw.is_absolute() else root / raw
    # .resolve() collapses ".." AND resolves symlinks in existing
    # components, for existing and non-existent paths alike (strict=False
    # is the default). This is what closes the unresolved-tail hole.
    resolved = joined.resolve()
    if not resolved.is_relative_to(root):
        raise PathViolation(f"path escapes project root: {candidate}")
    rel = resolved.relative_to(root)
    if for_write:
        # Check for a .git component on BOTH the resolved path and the
        # lexical path (".."/"." collapsed, symlinks NOT followed). If .git
        # is itself a symlink to a directory inside the project, resolution
        # strips the ".git" name entirely — only the lexical path still
        # shows it. Case-folded, any depth, either route → denied.
        lexical = Path(os.path.normpath(joined))
        lex_parts = lexical.parts
        if _is_within(lexical, root):
            lex_parts = lex_parts[len(root.parts):]
        if any(part.lower() == ".git" for part in (*rel.parts, *lex_parts)):
            raise PathViolation("writes into .git/ are not allowed")
    if for_write and any(_is_within(resolved, shield) for shield in protected):
        # The sous control directory (config.toml with the command allowlist,
        # tasks.db, transcripts) must never be writable from inside the
        # sandbox, even when the project root contains it (e.g. root=$HOME).
        # Case-folded so a path-case variant can't reach the real inode on a
        # case-insensitive FS.
        raise PathViolation(f"writes into the sous data dir are not allowed: {candidate}")
    return resolved


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


@dataclass
class ChangedFile:
    path: str
    kind: str  # "created" | "modified"
    before_sha: str | None
    after_sha: str


def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + "\n[truncated]"
    return text


class ToolExecutor:
    def __init__(self, project_root: Path, config_path: Path,
                 data_dir: Path | None = None):
        self.project_root = project_root.resolve()
        self.config_path = config_path
        # Default matches how SousConfig derives data_dir from config_path.
        self._data_dir = (data_dir if data_dir is not None else config_path.parent).resolve()
        self._changes: dict[str, ChangedFile] = {}

    # -- files --

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        p = resolve_confined(self.project_root, path, for_write=False)
        lines = p.read_text(errors="replace").splitlines()
        window = lines[offset:offset + limit]
        numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(window)]
        return _truncate("\n".join(numbered) or "(empty file)")

    def _record_change(self, p: Path, before: bytes | None, after: bytes) -> None:
        rel = str(p.relative_to(self.project_root))
        prior = self._changes.get(rel)
        original_sha = prior.before_sha if prior else (_sha(before) if before is not None else None)
        kind = prior.kind if prior else ("modified" if before is not None else "created")
        self._changes[rel] = ChangedFile(rel, kind, original_sha, _sha(after))

    def write_file(self, path: str, content: str) -> str:
        p = resolve_confined(self.project_root, path, for_write=True,
                             protected=(self._data_dir,))
        before = p.read_bytes() if p.is_file() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self._record_change(p, before, content.encode())
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = resolve_confined(self.project_root, path, for_write=True,
                             protected=(self._data_dir,))
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"edit rejected: {n} matches for old text in {path} (need exactly 1)"
        before = text.encode()
        updated = text.replace(old, new, 1)
        p.write_text(updated)
        self._record_change(p, before, updated.encode())
        return f"edited {path}"

    def list_dir(self, path: str = ".") -> str:
        p = resolve_confined(self.project_root, path, for_write=False)
        entries = sorted(
            e.name + ("/" if e.is_dir() else "") for e in p.iterdir()
        )
        return _truncate("\n".join(entries) or "(empty dir)")

    def glob(self, pattern: str) -> str:
        hits = sorted(
            str(p.relative_to(self.project_root))
            for p in self.project_root.glob(pattern)
            if ".git" not in p.parts
            # Skip symlinks (or paths under symlinked dirs) that escape the
            # root: their NAME is inside but their target is not.
            and p.resolve().is_relative_to(self.project_root)
        )
        return _truncate("\n".join(hits[:500]) or "(no matches)")

    def grep(self, pattern: str, glob_pattern: str = "**/*") -> str:
        rx = re.compile(pattern)
        out: list[str] = []
        for p in sorted(self.project_root.glob(glob_pattern)):
            if not p.is_file() or ".git" in p.parts:
                continue
            if not p.resolve().is_relative_to(self.project_root):
                continue  # escaping symlink: reads outside the root are denied
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{p.relative_to(self.project_root)}:{i}:{line.strip()}")
                        if len(out) >= 200:
                            return _truncate("\n".join(out) + "\n[max hits reached]")
            except OSError:
                continue
        return _truncate("\n".join(out) or "(no matches)")

    def changed_files(self) -> list[ChangedFile]:
        return list(self._changes.values())

    # -- command execution --

    def run_command(self, command: str, approval: ApprovalHook | None = None,
                    timeout: int = 120) -> str:
        """Run a command with allowlist checking and optional approval hook.

        Args:
            command: Command string to parse and execute
            approval: Optional approval hook to override allowlist check
            timeout: Timeout in seconds (default 120)

        Returns:
            Command output: "exit code N\n<stdout>\n<stderr>" or error message
        """
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"command rejected: unparseable ({e})"
        if not argv:
            return "command rejected: empty"
        if not command_allowed(argv, current_allowlist(self.config_path)):
            if approval is None or not approval(command):
                return f"command denied (not allowlisted): {command}"
        try:
            proc = subprocess.run(
                argv, shell=False, cwd=self.project_root, env=scrubbed_env(),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"command timed out after {timeout}s: {command}"
        except FileNotFoundError:
            return f"command not found: {argv[0]}"
        out = f"exit code {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        return _truncate(out)
