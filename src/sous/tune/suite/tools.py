"""The suite loop's tools over a scratch copy of a synthetic project.

Not a boundary: the project is a throwaway copy under a temp dir, the only
commands accepted are the task's own verify commands and the suite's test
runners, and nothing here audits or scrubs. The bounds a command runs under
(`spawn`) keep a runaway candidate from taking the tune with it; they confine
nothing. The output shapes are fixed, so every arm the suite grades reads its
results the same way. `approvals_denied` in a suite run counts the commands
refused against this set, which is narrower than the 0.6 allowlist."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from sous.tune.suite import spawn

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
# Whole tokens, matched after shlex has unquoted: no shell runs here, so
# an operator would reach the command as a literal argument and its complaint
# would send the model down a wrong path; a `>` inside a quoted argument is
# not one.
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "&", "|", "|&", "<", ">", ">>", "2>", "2>&1", "&>"})
# ...and a token glued to one (`>out.txt`, `2>err`, `cmd;`), which would
# reach the command as an argument too.
_GLUED_OPERATOR = re.compile(r"^\d?[<>|&;]|[<>|&;]$")
CD_GUIDANCE = (
    "run a single command from the project root, or after one `cd <dir> &&`; "
    "no pipes, redirection or second command"
)


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
    return f"{head}\n[... {size - 2 * _CAP_HALF} bytes elided ...]\n{tail}"


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

    def _parse(self, command: str) -> tuple[list[str], Path]:
        """The argv and the directory to run it in. `cd <dir> && <command>`
        runs <command> in <dir>, as a shell would: local models emit the idiom
        whatever the prompt says, and one that named a directory expects its
        relative paths resolved there — run at the root instead, its "file not
        found" contradicts the listing it just read. <dir> must be a directory
        inside the project (a failed cd short-circuits the && in a real shell).
        Anything else shell-shaped is refused with guidance rather than run
        as literal arguments."""
        argv = shlex.split(command)
        cwd = self.root
        if argv and argv[0] == "cd":
            if len(argv) < 4 or argv[2] != "&&":
                raise ToolError(f"command rejected: {CD_GUIDANCE}")
            cwd = self._cd_target(argv[1])
            argv = argv[3:]
        if any(token in _SHELL_OPERATORS or _GLUED_OPERATOR.search(token) for token in argv):
            raise ToolError(f"command rejected: {CD_GUIDANCE}")
        return argv, cwd

    def _cd_target(self, arg: str) -> Path:
        try:
            target = self._confined(arg)
        except ToolError:
            raise ToolError(f"command rejected: cd target {arg} is outside the project") from None
        if not target.is_dir():
            raise ToolError(f"command rejected: cd target {arg} is not a directory in the project")
        return target

    def run_command(self, command: str, timeout: float | None = None) -> str:
        try:
            argv, cwd = self._parse(command)
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
        try:
            with spawn.bounded(argv, cwd=cwd, timeout=budget) as run:
                if run.stopped == "timeout":
                    return f"command timed out after {budget}s: {command}"
                body = _spooled(run.stdout)
                if run.stopped == "output":
                    stopped = f"command stopped: more than {spawn.MAX_SPOOL_BYTES} bytes of output"
                    return f"{stopped}\n{body}"
                return f"exit code {run.returncode}\n{body}"
        except FileNotFoundError:
            return f"command not found: {argv[0]}"
