"""The suite loop's tools over a scratch copy of a synthetic project.

Not a boundary: the project is a throwaway copy under a temp dir, the only
commands accepted are the task's own verify commands and the suite's test
runners, and nothing here audits, scrubs or kills process groups. Output
shapes match what the worker's tools printed, so the model reads the same
results the suite was graded against."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

MAX_TOOL_OUTPUT = 16_000
MAX_GREP_HITS = 200
MAX_GLOB_HITS = 500
# What a task may run besides its own verify commands: the runners the suite's
# tasks are written for, matched as argv prefixes like the old allowlist was.
ACCEPTED_RUNNERS = (
    "python -m unittest",
    "python3 -m unittest",
    "python -m pytest",
    "python3 -m pytest",
    "pytest",
)


class ToolError(Exception):
    """A call the scratch project refuses; the message is what the model reads."""


def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_OUTPUT:
        return text[:MAX_TOOL_OUTPUT] + "\n[truncated]"
    return text


class ScratchTools:
    def __init__(
        self,
        root: Path,
        verify_commands: tuple[str, ...] = (),
        *,
        command_timeout: float = 120.0,
    ):
        self.root = root.resolve()
        self.denied = 0
        self._accepted = [shlex.split(c) for c in (*verify_commands, *ACCEPTED_RUNNERS)]
        self._command_timeout = command_timeout

    def _confined(self, path: str) -> Path:
        raw = Path(path)
        candidate = (raw if raw.is_absolute() else self.root / raw).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
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
            if ".git" not in p.parts and p.resolve().is_relative_to(self.root)
        )
        return _truncate("\n".join(hits[:MAX_GLOB_HITS]) or "(no matches)")

    def grep(self, pattern: str, glob_pattern: str = "**/*") -> str:
        rx = re.compile(pattern)
        out: list[str] = []
        for p in sorted(self.root.glob(glob_pattern)):
            if not p.is_file() or ".git" in p.parts or not p.resolve().is_relative_to(self.root):
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

    def run_command(self, command: str, timeout: float | None = None) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"command rejected: unparseable ({e})"
        if not argv:
            return "command rejected: empty"
        if not any(argv[: len(prefix)] == prefix for prefix in self._accepted):
            self.denied += 1
            return f"command denied (not allowlisted): {command}"
        budget = self._command_timeout if timeout is None else timeout
        try:
            proc = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True, timeout=budget, check=False
            )
        except FileNotFoundError:
            return f"command not found: {argv[0]}"
        except subprocess.TimeoutExpired:
            return f"command timed out after {budget}s: {command}"
        body = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        return _truncate(f"exit code {proc.returncode}\n{body}")
