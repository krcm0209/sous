"""Run a candidate's command under the harness's bounds.

The command is the candidate's own code — a local model wrote it — so its
output is spooled to unlinked temp files rather than pipes (capture_output held
a runaway test's whole output in this process, beside the resident weights),
its stdin is closed (a child that reads it must see EOF, not the operator's
terminal), and it is stopped past a wall clock or a size cap: the spool sits on
the boot volume, and a test printing in a loop fills that faster than a
two-minute clock notices. Either stop kills the command's whole process group,
so nothing it started outlives it, holding the spool's blocks or writing into
the project while the project is graded. None of this confines the command;
it keeps a runaway from taking the tune, or the machine, with it."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

# Far above what any reader keeps — the tools cap a result at 16 000
# characters, the grader reads one line of each stream — and about ten
# milliseconds of a runaway, so the stop lands within one poll.
MAX_SPOOL_BYTES = 16 * 1024 * 1024
_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class Spawned:
    """A finished command. `stopped` is None when it exited on its own,
    "timeout" or "output" when the harness killed it — then `returncode` is
    the kill's, not the command's. The streams are open only inside the
    `bounded` block that yields this; with `merged`, both are one file."""

    returncode: int
    stopped: str | None
    stdout: IO[bytes]
    stderr: IO[bytes]


@contextlib.contextmanager
def bounded(
    argv: list[str], *, cwd: Path, timeout: float, merged: bool = True
) -> Iterator[Spawned]:
    """Run `argv` in `cwd` and yield it finished. `merged` puts both streams
    in one spool, in write order; otherwise each has its own. A command that
    cannot start raises what Popen raised (FileNotFoundError, PermissionError)."""
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(tempfile.TemporaryFile())
        err = out if merged else stack.enter_context(tempfile.TemporaryFile())
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        stopped = _watch(proc, (out,) if merged else (out, err), timeout)
        yield Spawned(proc.returncode, stopped, out, err)


def _watch(proc: subprocess.Popen, spools: tuple[IO[bytes], ...], timeout: float) -> str | None:
    deadline = time.monotonic() + timeout
    while True:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_POLL_SECONDS)
            return None
        if sum(os.fstat(spool.fileno()).st_size for spool in spools) > MAX_SPOOL_BYTES:
            _kill_group(proc)
            return "output"
        if time.monotonic() >= deadline:
            _kill_group(proc)
            return "timeout"


def _kill_group(proc: subprocess.Popen) -> None:
    # start_new_session made the command a group leader, so its pid names
    # the group; a group already gone is nothing to kill.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
