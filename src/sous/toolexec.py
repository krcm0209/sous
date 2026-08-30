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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sous.config import current_allowlist

MAX_TOOL_OUTPUT = 16_000

ApprovalHook = Callable[[str], bool]

_ENV_PASS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")


_CD_GUIDANCE = (
    "commands already run at the project root without a shell; "
    "run a single command per call, without 'cd'"
)


_CD_SHELL_PUNCTUATION = ";&|<>"


def _parse_cd_idiom(command: str) -> tuple[str, list[str]] | None:
    """Parse `cd <dir> && <command...>`, returning (<dir>, <command argv>).

    None when `command` is not a `cd ...` invocation at all. Raises ValueError
    when it is a `cd` but not the honored idiom: no `&&`, an empty remainder,
    or any further shell operator in the remainder.

    Tokenized with `punctuation_chars` so shell operators are their own tokens
    even glued to a word (`echo hi>out`, `a|b`), closing the gap a plain
    shlex.split leaves — it would hand back `>`/`|` as ordinary arguments and
    the shell-less runner would execute them literally, which is different
    semantics from what the operator asked for. Shared by normalize_cd_prefix
    and canonical_command_for_allowlist so both judge the idiom identically.
    A quoted operator in the remainder is over-rejected (rare, and the model
    can drop the `cd`); that is preferred to running a redirection literally.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_CD_SHELL_PUNCTUATION)
    lexer.whitespace_split = True
    tokens = list(lexer)
    if not tokens or tokens[0] != "cd":
        return None
    if len(tokens) < 3 or tokens[2] != "&&":
        raise ValueError(_CD_GUIDANCE)
    directory, rest = tokens[1], tokens[3:]
    if not rest:
        raise ValueError(_CD_GUIDANCE)
    if any(set(tok) <= set(_CD_SHELL_PUNCTUATION) for tok in (directory, *rest)):
        raise ValueError(
            "chained shell commands, pipes, and redirections are not supported; "
            "run one command per call (commands already run at the project root)"
        )
    return directory, rest


def normalize_cd_prefix(command: str, argv: list[str], project_root: Path) -> list[str]:
    """Strip the shell idiom `cd <dir> && <command...>` down to <command...>.

    The runner is deliberately shell-less, so local models' habitual
    `cd proj && pytest` used to fail twice over: the `cd` prefix missed the
    leading-token allowlist (burning the approval timeout headless), and even
    an approved run exec'd a nonexistent `cd` binary. Every command already
    runs at the project root, so the `cd` is dropped and the remainder runs
    there — for the overwhelmingly common `cd <project-root> && <cmd>` this is
    identical, and cwd is never taken from `<dir>`, so there is no path for a
    `cd` to widen confinement or move the working directory out of the root.

    - The remainder must be a single command: `_parse_cd_idiom` rejects any
      further shell operator (`&&`, `||`, `;`, `|`, and redirections `<`/`>`),
      including one glued to a token, rather than run it as a literal argument.
    - `<dir>` must resolve to an existing directory inside the root. A failed
      `cd` short-circuits `&&` in a real shell, so a missing or non-directory
      target rejects the whole command instead of running the remainder; an
      out-of-root target is rejected too (for the message — it is never cwd).
    - Anything that is not exactly `cd <dir> && ...` (bare `cd x`, `cd` with
      no target) is rejected with guidance rather than left to fail as
      "command not found: cd".

    Non-cd argv is returned untouched: '&&' stays an inert literal argument
    everywhere else, per the existing metacharacter tests. Raises ValueError
    (shape/escape); run_command turns it into a rejection message.
    """
    if not argv or argv[0] != "cd":
        return argv
    parsed = _parse_cd_idiom(command)
    if parsed is None:  # argv[0] == "cd" but the lexer disagreed; treat as bad
        raise ValueError(_CD_GUIDANCE)
    directory, rest = parsed
    # resolve_confined bounds the path (out-of-root → clear message); is_dir
    # enforces the `&&` short-circuit — a failed cd must not run the remainder.
    # The resolved path is otherwise discarded: <dir> never becomes cwd, so
    # there is no TOCTOU race between this check and the launch.
    resolved = resolve_confined(project_root, directory, for_write=False)
    if not resolved.is_dir():
        raise ValueError(f"cd target is not a directory: {directory}")
    return rest


def command_allowed(argv: list[str], allowlist: list[list[str]]) -> bool:
    """Check if argv is allowlisted by leading-token equality.

    Skips falsy entries to prevent empty allowlist entries from matching all commands.
    """
    return (
        any(len(argv) >= len(entry) and argv[: len(entry)] == entry for entry in allowlist if entry)
        if argv
        else False
    )


def canonical_command_for_allowlist(command: str) -> str:
    """The form of `command` that `command_allowed` will actually match — i.e.
    with a `cd <dir> &&` prefix stripped, since run_command matches the
    stripped argv. Persisting the raw `cd ... && ...` string instead would
    write an entry no future request can ever match (leading token 'cd'), so
    an approve-and-persist on a cd-prefixed command would keep prompting.

    Pure and filesystem-free: shares _parse_cd_idiom's shape judgement (so the
    persisted form matches exactly what run_command strips), but skips its
    confinement/is_dir checks. Returns `command` unchanged when it is not the
    clean cd idiom or cannot be parsed."""
    try:
        parsed = _parse_cd_idiom(command)
    except ValueError:
        return command
    if parsed is None:
        return command
    return shlex.join(parsed[1])


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
_KILL_POLL_SECONDS = 0.02

# ESRCH and EPERM both mean "nothing in this group can be signalled" — see
# _kill_process_group for why EPERM is not a failure here.
_ALREADY_DEAD = (ProcessLookupError, PermissionError)


def _await_exit_without_reaping(pid: int, timeout: float) -> None:
    """Wait up to `timeout` for `pid` to exit, deliberately leaving it
    un-reaped so its pid stays allocated."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None:
                return
        except ChildProcessError:
            return  # already reaped; nothing left to wait for
        if time.monotonic() >= deadline:
            return
        time.sleep(_KILL_POLL_SECONDS)


# Process groups of commands running right now. start_new_session puts each
# child in its own session, so it outlives the daemon and keeps writing to the
# user's project unless the group is killed on the way out. Registered here
# rather than plumbed through the worker: the groups are created in this file
# and the kill escalation already lives here, so the boundary stays in one
# place. The worker runs one task at a time today; a set costs nothing and
# does not assume that.
_active_groups: set[int] = set()
_active_groups_lock = threading.Lock()
# Latched by terminate_active_commands(). A daemon that has begun shutting down
# never runs another command, so this is deliberately one-way.
_registration_closed = False


def _register_group(pgid: int) -> bool:
    """Claim a group, or report that shutdown has already closed registration.

    The flag and the set are read and written under one acquisition, so a
    command either lands in the snapshot terminate_active_commands() takes or
    is told to kill itself. There is no gap between the two for it to fall
    through, which is what a plain snapshot got wrong.
    """
    with _active_groups_lock:
        if _registration_closed:
            return False
        _active_groups.add(pgid)
        return True


def _unregister_group(pgid: int) -> None:
    """Release a group id. Idempotent, and safe to call from any exit path.

    Call this the moment the leader is reaped, not when the command finishes:
    a reaped pid belongs to the OS again, and everything after the reap — the
    project-tree audit, the output read-back — is time in which a shutdown
    could signal whatever inherited the number.
    """
    with _active_groups_lock:
        _active_groups.discard(pgid)


def terminate_active_commands() -> int:
    """Kill every command group still running, returning how many there were.

    SIGKILL immediately, with no grace period, unlike the timeout path. Two
    reasons, and the second is the important one:

    The daemon exits as soon as this returns, so nothing would observe a
    well-behaved teardown — the grace period buys a timed-out test runner a
    tidy exit, but buys a dying daemon nothing.

    More importantly, waiting here is unsafe in a way it is not on the timeout
    path. There, _kill_process_group is called by the worker thread itself, so
    it can hold the group leader un-reaped across the escalation. Here the
    worker is concurrently blocked in proc.wait() and reaps the leader the
    moment it dies — so any delay between signalling and SIGKILL is a window
    where the pgid is free to be recycled onto an unrelated group. Signalling
    once, immediately, keeps that window as small as this design allows.

    Registration is closed first so a task mid-flight cannot start another
    command behind the snapshot: stop.set() does not interrupt run_task, and
    its agent loop is free to issue another run_command as soon as the current
    one dies.
    """
    global _registration_closed
    with _active_groups_lock:
        _registration_closed = True
        groups = list(_active_groups)
    for pgid in groups:
        with contextlib.suppress(*_ALREADY_DEAD):
            os.killpg(pgid, signal.SIGKILL)
    return len(groups)


def _kill_process_group(pgid: int, proc: subprocess.Popen) -> None:
    """Kill a timed-out command's entire process group with escalation:
    SIGTERM first (so well-behaved test runners can clean up), a short grace
    period, then SIGKILL for whatever is left. Reaps the direct child last.

    start_new_session makes the child its own group leader, so the pgid IS
    its pid. It is therefore left un-reaped until after the SIGKILL: reaping
    it during the grace period would release that pid and let the group id be
    recycled, aiming the escalation at an unrelated group.

    ESRCH and EPERM both mean the group has nothing left to kill. macOS
    returns EPERM rather than ESRCH once the group's remaining members are
    un-reaped zombies; while any member is still signalable the call succeeds
    and kills it, so suppressing EPERM cannot mask a survivor.

    stdout/stderr go to temp files, not pipes, so no surviving descendant
    can hold a pipe write-end open and hang the wait() here: waitpid()
    blocks only on the direct child's exit, never on stream EOF."""
    with contextlib.suppress(*_ALREADY_DEAD):
        os.killpg(pgid, signal.SIGTERM)
    # Grace period for SIGTERM handlers, without reaping the group leader.
    _await_exit_without_reaping(proc.pid, _KILL_GRACE_SECONDS)
    with contextlib.suppress(*_ALREADY_DEAD):
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
        self, command: str, approval: ApprovalHook | None = None, timeout: float = 120
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
            timeout: Timeout in seconds (default 120). Fractional values are
                honoured — callers clamp it to the remaining task budget.

        Returns:
            Command output: "exit code N\n<stdout>\n<stderr>" or error message
        """
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"command rejected: unparseable ({e})"
        if not argv:
            return "command rejected: empty"
        # Before the allowlist/approval gate, on purpose: a confinement
        # violation in the cd target must be unapprovable, and the allowlist
        # must match the command that will actually run (the cd prefix
        # stripped). The command always runs at the project root.
        try:
            argv = normalize_cd_prefix(command, argv, self.project_root)
        except (ValueError, PathViolation) as e:
            return f"command rejected: {e}"
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
            # start_new_session made the child its own group leader, so the
            # pgid IS its pid — no syscall needed to learn it. os.getpgid()
            # here was worse than redundant: macOS returns ESRCH for a zombie
            # (Linux does not), so any command exiting before this line raised
            # ProcessLookupError. The pid stays valid because Popen leaves the
            # child un-reaped until the wait() below; see _kill_process_group
            # for why it must stay that way through the kill escalation.
            pgid = proc.pid
            if not _register_group(pgid):
                # Shutdown began between Popen and here. Kill what we just
                # started rather than let it outlive the daemon: the snapshot
                # was taken before this group existed, so nothing else will.
                _kill_process_group(pgid, proc)
                return "command aborted: the daemon is shutting down"
            try:
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Kill the whole group FIRST, then record: snapshotting while
                    # descendants are still alive would let them modify files
                    # after the audit.
                    _kill_process_group(pgid, proc)
                    _unregister_group(pgid)  # reaped in there; the id is not ours
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
                    _unregister_group(pgid)
                    raise
                # wait() reaped the leader, so release the id before the audit
                # and the output read-back: both can take a while on a large
                # project, and holding a recycled id across them means shutdown
                # would SIGKILL whatever inherited it.
                _unregister_group(pgid)
                self._record_command_changes(before_snap)
                body = _capped_command_output(out_f, err_f)
                return f"exit code {proc.returncode}\n{body}"
            finally:
                # Backstop for any path above that returns or raises before
                # reaching its own release. discard() makes this idempotent.
                _unregister_group(pgid)
