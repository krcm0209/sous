import io
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import SousService
from tests.fake_engine import FakeEngine


@pytest.fixture(autouse=True)
def _no_leaked_warning_capture():
    """server.main() turns logging.captureWarnings on for the process. Left on,
    a later captureWarnings(True) anywhere in the suite is a no-op once pytest
    has restored showwarning — and that test passes or fails by file order."""
    yield
    logging.captureWarnings(False)


@pytest.fixture()
def svc(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    return SousService(engines, cfg), root


def test_the_memory_read_is_the_last_thing_the_document_does(svc, monkeypatch):
    """Reading mlx's memory releases this thread's mlx state; the registry's
    snapshot probes the prompt cache, which can free mlx arrays. Freeing
    after the release is what segfaults the thread on its way out."""
    from sous import server as server_module

    service, _ = svc
    order: list[str] = []
    monkeypatch.setattr(server_module, "_mlx_memory_gb", lambda: order.append("memory") or 1.5)
    snapshot = service.inflight.snapshot

    def probing_snapshot() -> dict:
        order.append("snapshot")
        return snapshot()

    monkeypatch.setattr(service.inflight, "snapshot", probing_snapshot)
    doc = service.status_document(recent=True)
    assert order == ["snapshot", "memory"]
    assert doc["engine"]["memory_gb"] == 1.5


# --- only one daemon may run: a second would load its own copy of the
# --- model beside the first's.


def test_lock_excludes_a_separate_process(tmp_path: Path):
    """flock, not a pidfile: the kernel drops it when the holder dies, so a
    SIGKILLed daemon can never wedge the next start out of the lock."""
    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()

    probe = (
        "import sys; from pathlib import Path;"
        "from sous.server import _acquire_singleton_lock;"
        f"_acquire_singleton_lock(Path({str(data)!r}));"
        "print('ACQUIRED')"
    )
    holder = _acquire_singleton_lock(data)
    try:
        blocked = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert blocked.returncode != 0, "second process acquired a held lock"
        assert "ACQUIRED" not in blocked.stdout
    finally:
        holder.close()

    # holder released -> the next process gets it
    freed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert "ACQUIRED" in freed.stdout, freed.stderr


def test_non_contention_lock_failure_is_not_reported_as_another_daemon(tmp_path: Path, monkeypatch):
    """ENOLCK, or a filesystem without flock support, is a real startup failure.

    Contention is EAGAIN (EWOULDBLOCK). Translating every OSError into "another
    daemon already holds ..." would send the operator hunting for a process that
    does not exist, hiding the actual cause.
    """
    import errno
    import fcntl

    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()

    def no_locks_available(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks_available)
    with pytest.raises(OSError) as exc:
        _acquire_singleton_lock(data)
    assert exc.value.errno == errno.ENOLCK


def test_lock_file_records_the_holder_pid(tmp_path: Path):
    """The lock file names its holder, so a client can tell a restarted daemon
    from the one it connected to — a port check alone cannot."""
    from sous.server import _acquire_singleton_lock

    data = tmp_path / "data"
    data.mkdir()
    holder = _acquire_singleton_lock(data)
    try:
        assert (data / "daemon.lock").read_text().strip() == str(os.getpid())
    finally:
        holder.close()


def test_main_installs_the_shutdown_handler_before_serving(tmp_path: Path, monkeypatch):
    """The handler is useless if main() never installs it, and its body is
    untestable in-process (it ends in os._exit) — so pinning this one-line
    link is the only coverage the wiring gets.
    """
    import sous.server as server

    data = tmp_path / "data"
    data.mkdir()
    installed = []
    with socket.socket() as occupied:  # make mcp.run() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / "config.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        monkeypatch.setattr(
            server, "_install_shutdown_handler", lambda stop: installed.append(stop)
        )
        with pytest.raises(SystemExit):
            server.main()
    assert installed, "main() served without installing the shutdown handler"


def test_main_routes_config_load_warnings_through_the_daemon_shape(
    tmp_path: Path, monkeypatch, caplog
):
    """load_config() runs inside main(), which must already have installed the
    warnings-to-logging redirect by the time it does — otherwise a config typo
    reaches stderr as a raw `UserWarning`, unshaped and unleveled, while every
    other line the daemon writes carries a timestamp and a level.
    """
    import dataclasses
    import logging

    import sous.server as server
    from sous.config import load_config as real_load_config

    data = tmp_path / "data"
    data.mkdir()
    config_path = tmp_path / "config.toml"
    config_path.write_text("[nonsense]\n")  # unknown section: load_config warns

    with socket.socket() as occupied:  # make mcp.run() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)

        def fake_load_config():
            # The real parser, so the warning is genuine — only the paths
            # (never the real ~/.sous) and port are stood in for.
            cfg = real_load_config(config_path)
            return dataclasses.replace(cfg, data_dir=data, server_port=occupied.getsockname()[1])

        monkeypatch.setattr(server, "load_config", fake_load_config)
        # logging.captureWarnings(True) only rebinds warnings.showwarning the
        # first time it is called in a process — a no-op on every later call
        # while it thinks capture is already on. An earlier test in this same
        # file already left it "on" from a stale rebinding, so reset it here
        # (and restore that reset afterward) rather than depend on suite order.
        logging.captureWarnings(False)
        try:
            with caplog.at_level("WARNING"), pytest.raises(SystemExit):
                server.main()
        finally:
            logging.captureWarnings(False)

    warning_records = [r for r in caplog.records if r.name == "py.warnings"]
    assert warning_records, "config warning never reached logging as py.warnings"
    assert "unknown section" in warning_records[0].getMessage()


class _FakeStderr(io.StringIO):
    """A stream main() can log through (write/flush both work) whose isatty()
    is controlled by the test, unlike a real TextIOWrapper's."""

    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _run_main_against_an_occupied_port(tmp_path: Path, monkeypatch, *, subdir: str) -> None:
    """make serve() fail fast instead of actually serving, so main() runs its
    startup steps and exits via the bind failure."""
    import sous.server as server

    data = tmp_path / subdir
    data.mkdir()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / f"{subdir}.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        with pytest.raises(SystemExit):
            server.main()


def test_main_disables_progress_bars_only_when_stderr_is_not_a_tty(tmp_path: Path, monkeypatch):
    """Under launchd stderr is a pipe with nobody to animate a bar for; a
    `sous serve` run by hand for a first multi-GB model download has a real
    terminal, and the download's own progress should still show there."""
    calls: list[bool] = []
    monkeypatch.setattr("huggingface_hub.utils.disable_progress_bars", lambda: calls.append(True))

    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty=True))
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="tty")
    assert calls == []  # interactive: progress bars stay on

    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty=False))
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="no-tty")
    assert calls == [True]  # non-interactive: progress bars are silenced

    # fd 2 closed at exec (`sous serve 2>&-`): Python sets sys.stderr to None,
    # and the daemon must still start rather than die silently on isatty().
    monkeypatch.setattr(sys, "stderr", None)
    _run_main_against_an_occupied_port(tmp_path, monkeypatch, subdir="no-stderr")
    assert calls == [True, True]


def test_status_memory_probe_releases_mlx_thread_state(svc, monkeypatch):
    """status_document runs in whatever short-lived worker thread the monitor
    and gateway routes hand it; any mlx state it created must be released
    before that thread can exit (ml-explore/mlx#4327)."""
    import sous.server as server

    service, _ = svc
    released = []
    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: released.append(True))
    service.status_document(recent=False)
    assert released


def test_status_document_releases_mlx_thread_state_when_the_build_fails(svc, monkeypatch):
    """The engine's status and the registry's snapshot can free mlx arrays
    on the building thread before the memory read releases its state; the
    routes catch a failure in between and hand the pooled thread back, so
    the release cannot depend on reaching the read."""
    import sous.server as server

    service, _ = svc
    released = []
    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: released.append(True))

    def boom():
        raise RuntimeError("the probe failed")

    monkeypatch.setattr(service.inflight, "snapshot", boom)
    with pytest.raises(RuntimeError):
        service.status_document(recent=True)
    assert released


def test_server_status_reports_the_served_config(svc):
    """The status document is how a user confirms which model ids the daemon
    would serve locally, and at what window."""
    service, _ = svc
    assert service.status_document(recent=False)["config"] == {
        "model_id": "mlx-community/Qwen3.8-27B-4bit",
        "idle_unload_minutes": 30,
        "port": 8383,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "upstream_url": "https://api.anthropic.com",
        "generation_timeout_minutes": 30,
    }


def test_uvicorn_config_bounds_graceful_shutdown():
    """uvicorn holds SIGTERM while serving and, unbounded, waits for open
    connections — a long non-streaming gateway turn would defer sous's own
    shutdown handler by minutes. The bound is what keeps `sous stop` prompt."""
    from sous.server import GRACEFUL_SHUTDOWN_SECONDS, uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_SECONDS == 5
    assert cfg.host == "127.0.0.1"
    # uvicorn's access log prints the raw request target, query string
    # included, at INFO — the daemon's own level.
    assert cfg.access_log is False


def test_create_server_installs_the_daemon_log_handler_and_drops_the_rich_one(svc):
    import logging

    from sous.logs import SOUS_HANDLER_NAME
    from sous.server import create_server

    class RichHandler(logging.Handler):  # the SDK's, matched by class name
        def emit(self, record: logging.LogRecord) -> None:
            pass

    service, _ = svc
    root = logging.getLogger()
    root.addHandler(RichHandler())
    try:
        create_server(service.engines, service.config)
        names = [h.get_name() for h in root.handlers]
        assert names.count(SOUS_HANDLER_NAME) == 1
        assert not [h for h in root.handlers if type(h).__name__ == "RichHandler"]
    finally:
        for h in list(root.handlers):
            if h.get_name() == SOUS_HANDLER_NAME or type(h).__name__ == "RichHandler":
                root.removeHandler(h)


def test_uvicorn_config_leaves_logging_to_the_daemon():
    """uvicorn's default dictConfig gives its loggers their own handlers and
    stops propagation, so its lines would keep the `INFO:     …` shape
    instead of the daemon's. log_config=None lets them reach the root."""
    from sous.server import uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.log_config is None
    assert cfg.access_log is False


def test_the_status_document_carries_no_task_fields(svc):
    service, _root = svc
    doc = service.status_document(recent=True)
    assert set(doc) == {"engine", "inflight", "config", "recent_turns"}
    assert set(doc["config"]) == {
        "model_id",
        "idle_unload_minutes",
        "port",
        "local_models",
        "max_context_tokens",
        "upstream_url",
        "generation_timeout_minutes",
    }
    assert set(service.status_document(recent=False)) == {"engine", "inflight", "config"}
    assert len(service.status_version()) == 3
