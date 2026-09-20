"""The suite loop's tools over a scratch copy of a synthetic project.

Not a boundary: the project is a throwaway copy under a temp dir, the only
commands accepted are the task's own verify commands and the suite's test
runners, and nothing here audits, scrubs or kills process groups. The output
shapes are fixed, so every arm the suite grades reads its results the same
way. `approvals_denied` in a suite run counts the commands refused against
this set, which is narrower than the 0.6 allowlist."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

MAX_TOOL_OUTPUT = 16_000
MAX_GREP_HITS = 200
MAX_GLOB_HITS = 500
# One command's wall-clock bound, unless the task's own deadline is nearer.
COMMAND_TIMEOUT = 120.0
# What a task may run besides its own verify commands: the runners the suite's
# tasks are written for, matched as argv prefixes.
ACCEPTED_RUNNERS = (
    "python -m unittest",
    "python3 -m unittest",
    "python -m pytest",
    "python3 -m pytest",
    "pytest",
)
_CAP_HALF = MAX_TOOL_OUTPUT // 2
_SHELL_OPERATORS = ("&&", "||", ";", "|", "<", ">")
CD_GUIDANCE = "run a single command from the project root, without cd, &&, pipes or redirection"


class ToolError(Exception):
    """A call the scratch project refuses; the message is what the model reads."""


def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + "\n[truncated]"
    return text


def _spooled(f) -> str:
    """A command's spooled output, read back under the cap: all of a short
    one, the head and tail of a long one with the elided count between. Tail
    as well as head because a verify command's verdict ("3 failed", the
    traceback, the summary) is at the end, which head-only truncation would
    hide."""
    size = f.seek(0, os.SEEK_END)
    f.seek(0)
    if size <= MAX_TOOL_OUTPUT:
        return f.read().decode(errors="replace")
    head = f.read(_CAP_HALF).decode(errors="replace")
    f.seek(size - _CAP_HALF)
    tail = f.read().decode(errors="replace")
    return f"{head}\n[... {size - 2 * _CAP_HALF} characters elided ...]\n{tail}"


class ScratchTools:
    def __init__(self, root: Path, verify_commands: tuple[str, ...] = ()):
        self.root = root.resolve()
        self.denied = 0
        # A blank entry would split to an empty prefix, and an empty prefix
        # matches every command.
        self._accepted = [
            argv for argv in (shlex.split(c) for c in (*verify_commands, *ACCEPTED_RUNNERS)) if argv
        ]

    def _inside(self, resolved: Path) -> bool:
        return resolved.is_relative_to(self.root)

    def _confined(self, path: str) -> Path:
        raw = Path(path)
        candidate = (raw if raw.is_absolute() else self.root / raw).resolve()
        if not self._inside(candidate):
            raise ToolError(f"path escapes the project: {path}")
        return candidate

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        p = self._confined(path)
        # Streamed, not read_text().splitlines(): reading five lines of a
        # large file must not materialize the whole file first.
        numbered: list[str] = []
        joined_len = -1  # each entry costs len(entry) + 1 joining "\n"
        with p.open(errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                if lineno <= offset:
                    continue
                if lineno > offset + limit:
                    break
                entry = f"{lineno}\t" + line.removesuffix("\n")
                numbered.append(entry)
                joined_len += len(entry) + 1
                if joined_len > MAX_TOOL_OUTPUT:
                    break
        return _truncate("\n".join(numbered) or "(empty file)")

    def write_file(self, path: str, content: str) -> str:
        p = self._confined(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self._confined(path)
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"edit rejected: {n} matches for old text in {path} (need exactly 1)"
        p.write_text(text.replace(old, new, 1))
        return f"edited {path}"

    def list_dir(self, path: str = ".") -> str:
        p = self._confined(path)
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return _truncate("\n".join(entries) or "(empty dir)")

    def glob(self, pattern: str) -> str:
        hits = sorted(
            str(p.relative_to(self.root))
            for p in self.root.glob(pattern)
            if ".git" not in p.parts and self._inside(p.resolve())
        )
        return _truncate("\n".join(hits[:MAX_GLOB_HITS]) or "(no matches)")

    def grep(self, pattern: str, glob_pattern: str = "**/*") -> str:
        rx = re.compile(pattern)
        out: list[str] = []
        for p in sorted(self.root.glob(glob_pattern)):
            if not p.is_file() or ".git" in p.parts or not self._inside(p.resolve()):
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{p.relative_to(self.root)}:{i}:{line.strip()}")
                        if len(out) >= MAX_GREP_HITS:
                            return _truncate("\n".join(out) + "\n[max hits reached]")
            except OSError:
                continue
        return _truncate("\n".join(out) or "(no matches)")

    def _strip_cd(self, argv: list[str]) -> list[str]:
        """`cd <dir> && <command>` down to `<command>`: every command already
        runs at the project root, and local models emit the idiom whatever the
        prompt says. `<dir>` must be a directory inside the project (a failed
        cd short-circuits the && in a real shell), the remainder one command
        with no further shell operator, and cwd is never taken from it."""
        if not argv or argv[0] != "cd":
            return argv
        if len(argv) < 4 or argv[2] != "&&":
            raise ToolError(f"command rejected: {CD_GUIDANCE}")
        rest = argv[3:]
        if any(op in token for token in rest for op in _SHELL_OPERATORS):
            raise ToolError(f"command rejected: {CD_GUIDANCE}")
        try:
            target = self._confined(argv[1])
        except ToolError:
            raise ToolError(
                f"command rejected: cd target {argv[1]} is outside the project"
            ) from None
        if not target.is_dir():
            raise ToolError(
                f"command rejected: cd target {argv[1]} is not a directory in the project"
            )
        return rest

    def run_command(self, command: str, timeout: float | None = None) -> str:
        try:
            argv = self._strip_cd(shlex.split(command))
        except ValueError as e:
            return f"command rejected: unparseable ({e})"
        except ToolError as e:
            return str(e)
        if not argv:
            return "command rejected: empty"
        if not any(argv[: len(prefix)] == prefix for prefix in self._accepted):
            self.denied += 1
            return f"command denied (not allowlisted): {command}"
        budget = COMMAND_TIMEOUT if timeout is None else timeout
        # Spooled to one unlinked temp file, not pipes: capture_output would
        # hold a runaway test's whole output in this process, beside the
        # resident weights, before the cap could apply. Both streams in write
        # order, as a terminal would show them. stdin is closed: a child that
        # reads it must see EOF, not the operator's terminal.
        with tempfile.TemporaryFile() as spool:
            try:
                proc = subprocess.run(
                    argv,
                    cwd=self.root,
                    stdin=subprocess.DEVNULL,
                    stdout=spool,
                    stderr=spool,
                    timeout=budget,
                    check=False,
                )
            except FileNotFoundError:
                return f"command not found: {argv[0]}"
            except subprocess.TimeoutExpired:
                return f"command timed out after {budget}s: {command}"
            body = _spooled(spool)
        return f"exit code {proc.returncode}\n{body}"
