"""Worker tool execution: path-confined file ops and a no-shell command runner."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shlex
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sous.config import current_allowlist

MAX_TOOL_OUTPUT = 16_000

ApprovalHook = Callable[[str], bool]

_ENV_PASS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")


def command_allowed(argv: list[str], allowlist: list[list[str]]) -> bool:
    """Check if argv is allowlisted by leading-token equality.

    Skips falsy entries to prevent empty allowlist entries from matching all commands.
    """
    return (
        any(len(argv) >= len(entry) and argv[: len(entry)] == entry for entry in allowlist if entry)
        if argv
        else False
    )


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
    return len(p) >= len(a) and p[: len(a)] == a


def resolve_confined(
    project_root: Path, candidate: str, for_write: bool, *, protected: tuple[Path, ...] = ()
) -> Path:
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
            lex_parts = lex_parts[len(root.parts) :]
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


# Command output keeps the HEAD and the TAIL of the combined stdout+stderr,
# half the budget each: for a verify command the verdict — "3 failed", the
# traceback, the summary line — is at the END, which the head-only _truncate
# (still right for the file tools) would hide.
_CAP_HALF = MAX_TOOL_OUTPUT // 2


def _read_combined_span(stdout_f, n_out: int, stderr_f, n_err: int, start: int, length: int) -> str:
    """Read `length` bytes at offset `start` from the virtual concatenation
    stdout + b"\\n" + stderr, seeking within the spool files rather than
    loading them (they can be hundreds of MB). Decoded with errors="replace"
    so binary-ish output — or a multi-byte sequence split at a cut point —
    cannot raise."""
    end = start + length
    chunks: list[bytes] = []
    if start < n_out:
        stdout_f.seek(start)
        chunks.append(stdout_f.read(min(end, n_out) - start))
    if start <= n_out < end:
        chunks.append(b"\n")  # the separator the return shape puts between streams
    err_lo = max(start - n_out - 1, 0)
    err_hi = min(end - n_out - 1, n_err)
    if err_hi > err_lo:
        stderr_f.seek(err_lo)
        chunks.append(stderr_f.read(err_hi - err_lo))
    return b"".join(chunks).decode(errors="replace")


def _capped_command_output(stdout_f, stderr_f) -> str:
    """Bounded read-back of a command's spooled stdout+stderr: the whole
    thing when it fits in MAX_TOOL_OUTPUT, otherwise the head and the tail
    with an elision marker between them. Never holds more than
    ~MAX_TOOL_OUTPUT bytes in memory no matter how much the command wrote."""
    n_out = os.fstat(stdout_f.fileno()).st_size
    n_err = os.fstat(stderr_f.fileno()).st_size
    total = n_out + 1 + n_err  # the "\n" joining the two streams
    if total <= MAX_TOOL_OUTPUT:
        return _read_combined_span(stdout_f, n_out, stderr_f, n_err, 0, total)
    head = _read_combined_span(stdout_f, n_out, stderr_f, n_err, 0, _CAP_HALF)
    tail = _read_combined_span(stdout_f, n_out, stderr_f, n_err, total - _CAP_HALF, _CAP_HALF)
    return f"{head}\n[... {total - 2 * _CAP_HALF} bytes elided ...]\n{tail}"


_KILL_GRACE_SECONDS = 2.0


def _kill_process_group(pgid: int, proc: subprocess.Popen) -> None:
    """Kill a timed-out command's entire process group with escalation:
    SIGTERM first (so well-behaved test runners can clean up), a short grace
    period, then SIGKILL for whatever is left. Reaps the direct child.
    ProcessLookupError means the group is already gone — not an error.

    stdout/stderr go to temp files, not pipes, so no surviving descendant
    can hold a pipe write-end open and hang either wait() here: waitpid()
    blocks only on the direct child's exit, never on stream EOF."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        # Grace period for SIGTERM handlers to run before the hard kill.
        proc.wait(timeout=_KILL_GRACE_SECONDS)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        # Reap the direct child so it doesn't linger as a zombie.
        proc.wait(timeout=5)


class ToolExecutor:
    def __init__(self, project_root: Path, config_path: Path, data_dir: Path | None = None):
        self.project_root = project_root.resolve()
        self.config_path = config_path
        # Default matches how SousConfig derives data_dir from config_path.
        self._data_dir = (data_dir if data_dir is not None else config_path.parent).resolve()
        self._changes: dict[str, ChangedFile] = {}

    # -- files --

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        p = resolve_confined(self.project_root, path, for_write=False)
        # Streamed on purpose: read_text().splitlines() materialized the
        # ENTIRE file before the window applied (reading 5 lines of a 143 MB
        # file spiked peak RSS from 27 MB to 638 MB — and this is the
        # worker's most-used tool). Iterate line by line, skip to offset,
        # collect at most limit lines, and stop early once the joined output
        # is already past MAX_TOOL_OUTPUT — _truncate discards the rest.
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

    def _record_change(self, p: Path, before: bytes | None, after: bytes) -> None:
        rel = str(p.relative_to(self.project_root))
        prior = self._changes.get(rel)
        original_sha = prior.before_sha if prior else (_sha(before) if before is not None else None)
        kind = prior.kind if prior else ("modified" if before is not None else "created")
        self._changes[rel] = ChangedFile(rel, kind, original_sha, _sha(after))

    def write_file(self, path: str, content: str) -> str:
        p = resolve_confined(self.project_root, path, for_write=True, protected=(self._data_dir,))
        before = p.read_bytes() if p.is_file() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self._record_change(p, before, content.encode())
        return f"wrote {len(content)} chars to {path}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = resolve_confined(self.project_root, path, for_write=True, protected=(self._data_dir,))
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
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
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

    def _tree_snapshot(self) -> dict[str, tuple[int, int, int]]:
        """Cheap stat-based snapshot of the project tree: rel path ->
        (mtime_ns, size, ctime_ns). Skips .git (case-folded) and never follows
        symlinks — a symlink's target may live outside the root, and its
        content must be neither read nor reported. Stat-only on purpose: this
        runs before AND after every command, so no hashing here.

        ctime is in the signature specifically because mtime and size are
        forgeable by executed code (allowlisted test runners execute code the
        worker just wrote): an equal-length rewrite plus os.utime restores
        both. ctime — the inode CHANGE time — cannot be set from userspace;
        os.utime only sets atime/mtime, and calling it bumps ctime itself, so
        the forgery is self-defeating. Costs nothing: it rides in the same
        stat struct already being read."""
        snap: dict[str, tuple[int, int, int]] = {}
        stack: list[str] = [str(self.project_root)]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name.lower() != ".git":
                                    stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                rel = os.path.relpath(entry.path, self.project_root)
                                snap[rel] = (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
                        except OSError:
                            continue
            except OSError:
                continue
        return snap

    def _record_command_changes(self, before: dict[str, tuple[int, int, int]]) -> None:
        """Record files a command created or modified (the default allowlist
        ships formatters — black, npx prettier, ruff — whose whole job is
        editing files). Content is read only for files whose stat changed.
        Deletions cannot be expressed by the ChangedFile shape (after_sha is
        required) and are not recorded."""
        for rel, sig in self._tree_snapshot().items():
            if before.get(rel) == sig:
                continue
            try:
                content = (self.project_root / rel).read_bytes()
            except OSError:
                continue  # vanished between snapshot and read
            prior = self._changes.get(rel)
            if prior is not None:
                # already tracked: refresh the now-stale content hash, keep
                # the original kind and before_sha
                self._changes[rel] = ChangedFile(rel, prior.kind, prior.before_sha, _sha(content))
            else:
                kind = "modified" if rel in before else "created"
                # before_sha unknown: the pre-command snapshot is stat-only
                self._changes[rel] = ChangedFile(rel, kind, None, _sha(content))

    def run_command(
        self, command: str, approval: ApprovalHook | None = None, timeout: int = 120
    ) -> str:
        """Run a command with allowlist checking and optional approval hook.

        The command runs in its own session/process group (setsid), and on
        timeout the WHOLE group is killed — SIGTERM, a short grace, SIGKILL —
        before changes are recorded, so descendants spawned by test runners
        (pytest-xdist workers, npm's node, make's compilers) cannot keep
        running and mutating files after the timeout and audit. Best-effort,
        not a guarantee: a descendant that double-forks and calls setsid()
        itself escapes into a new session the group kill cannot reach (not
        closable without OS-level confinement macOS doesn't offer).

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
        if not command_allowed(argv, current_allowlist(self.config_path)) and (
            approval is None or not approval(command)
        ):
            return f"command denied (not allowlisted): {command}"
        before_snap = self._tree_snapshot()
        # stdout/stderr are spooled to unlinked temp files, NOT pipes:
        # communicate() would buffer the ENTIRE output in RAM before the
        # 16 KB cap could apply (a ~300 MB-noisy test run drove peak RSS
        # past 1 GB on a machine sharing 64 GB with a ~28.7 GB resident
        # model). Files also make the reap unhangable: no descendant can
        # keep a pipe write-end open, because there is no pipe.
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            try:
                # start_new_session: the child becomes a session/process-group
                # leader, so a timeout can kill its whole descendant tree, not
                # just the direct child. stdin=DEVNULL: a non-interactive
                # worker must never block on input, and a session leader has
                # no controlling terminal for inherited TTY stdin to make
                # sense.
                proc = subprocess.Popen(
                    argv,
                    shell=False,
                    cwd=self.project_root,
                    env=scrubbed_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    start_new_session=True,
                )
            except FileNotFoundError:
                return f"command not found: {argv[0]}"
            # Capture the group id while the child is certainly un-reaped (it
            # may have exited, but stays a zombie until wait() below).
            pgid = os.getpgid(proc.pid)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole group FIRST, then record: snapshotting while
                # descendants are still alive would let them modify files
                # after the audit.
                _kill_process_group(pgid, proc)
                # A timed-out command may already have modified files.
                self._record_command_changes(before_snap)
                return f"command timed out after {timeout}s: {command}"
            except BaseException:  # noqa: BLE001 — deliberate: KeyboardInterrupt/
                # SystemExit during daemon shutdown must not orphan a running
                # process group that keeps writing files. Kill the group, then
                # let the exception propagate (subprocess.run had the same
                # kill-on-any-exception backstop for the direct child). No
                # change recording here: the exception is propagating, no
                # report is being assembled.
                _kill_process_group(pgid, proc)
                raise
            self._record_command_changes(before_snap)
            body = _capped_command_output(out_f, err_f)
            return f"exit code {proc.returncode}\n{body}"
