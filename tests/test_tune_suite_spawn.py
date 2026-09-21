import subprocess
import sys
from pathlib import Path

import pytest

from sous.tune.suite import spawn


def test_separate_spools_keep_the_streams_apart(tmp_path: Path):
    argv = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    with spawn.bounded(argv, cwd=tmp_path, timeout=10, merged=False) as run:
        assert run.stopped is None and run.returncode == 0
        run.stdout.seek(0)
        run.stderr.seek(0)
        assert run.stdout.read() == b"out\n"
        assert run.stderr.read() == b"err\n"


def test_an_interrupt_inside_the_watch_kills_the_command(tmp_path: Path, monkeypatch):
    """Ctrl-C during a tune unwinds through the watch; the command, in a
    session of its own where the terminal's interrupt cannot reach it, must
    die with the tune rather than play on in the scratch project."""
    made: list[subprocess.Popen] = []
    real = subprocess.Popen

    class Recording(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(spawn.subprocess, "Popen", Recording)

    def interrupted(proc, spools, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(spawn, "_watch", interrupted)
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(KeyboardInterrupt), spawn.bounded(argv, cwd=tmp_path, timeout=5):
        pass
    assert len(made) == 1
    assert made[0].poll() is not None
