import io
import logging
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import anyio
import pytest
from starlette.applications import Starlette

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import Daemon, create_server
from tests.fake_engine import FakeEngine
from tests.fake_upstream import FakeUpstream


@pytest.fixture(autouse=True)
def _no_leaked_warning_capture():
    """server.main() turns logging.captureWarnings on for the process. Left on,
    a later captureWarnings(True) anywhere in the suite is a no-op once pytest
    has restored showwarning — and that test passes or fails by file order."""
    yield
    logging.captureWarnings(False)


@pytest.fixture(autouse=True)
def _no_leaked_tqdm_switch():
    """server.main() throws tqdm's process-wide mp_lock switch, which has no
    undo of its own. Left thrown, every later tqdm bar in the suite runs
    without the multiprocessing lock tqdm would otherwise give it."""
    from tqdm.std import TqdmDefaultWriteLock

    had = "mp_lock" in vars(TqdmDefaultWriteLock)
    before = vars(TqdmDefaultWriteLock).get("mp_lock")
    yield
    if had:
        TqdmDefaultWriteLock.mp_lock = before
    elif "mp_lock" in vars(TqdmDefaultWriteLock):
        del TqdmDefaultWriteLock.mp_lock


@pytest.fixture()
def svc(tmp_path: Path, monkeypatch):
    # status_document() releases the calling thread's mlx state, as the pool
    # thread that serves it must; on pytest's main thread that release leaves
    # every later real-mlx test in the session without a stream.
    from sous import server

    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: None)
    root = tmp_path / "proj"
    root.mkdir()
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    return Daemon(engines, cfg), root


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
    doc = service.status_document()
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


def test_the_daemon_gives_tqdm_no_multiprocessing_lock(monkeypatch):
    """tqdm's first bar would otherwise take a multiprocessing RLock — a
    named semaphore the resource tracker reports as leaked when the daemon
    leaves through os._exit. With the switch thrown, the first bar's lock is
    the thread lock alone and nothing is registered."""
    import multiprocessing.resource_tracker as resource_tracker

    import tqdm.std
    from tqdm.std import TqdmDefaultWriteLock

    import sous.server as server

    # A bare delattr(raising=False) records no undo for an absent attribute,
    # and the switch and get_lock() below would then outlive the test. The
    # setattr records each starting state, absence included; the delattr
    # gives the test the fresh-process state it needs.
    for owner, name in ((TqdmDefaultWriteLock, "mp_lock"), (tqdm.std.tqdm, "_lock")):
        monkeypatch.setattr(owner, name, None, raising=False)
        monkeypatch.delattr(owner, name)
    registered: list[tuple[str, str]] = []
    monkeypatch.setattr(
        resource_tracker, "register", lambda name, rtype: registered.append((name, rtype))
    )
    server._drop_tqdm_process_lock()
    lock = tqdm.std.tqdm.get_lock()
    assert TqdmDefaultWriteLock.mp_lock is None
    assert lock.locks == [TqdmDefaultWriteLock.th_lock]
    assert registered == []


def test_main_installs_the_shutdown_handler_before_serving(tmp_path: Path, monkeypatch):
    """The handler is useless if main() never installs it, and its body is
    untestable in-process (it ends in os._exit) — so pinning this one-line
    link is the only coverage the wiring gets. The same is true of the
    daemon's log handler: nothing but main() installs it any more (the
    endpoint test files' autouse fixtures call configure_daemon_logging()
    themselves, so they would stay green even if main() stopped calling it),
    so this test removes any copy left by an earlier test before asserting
    that main() is what put one back.
    """
    import sous.server as server
    from sous.logs import SOUS_HANDLER_NAME

    root = logging.getLogger()
    for h in list(root.handlers):
        if h.name == SOUS_HANDLER_NAME:
            root.removeHandler(h)
    data = tmp_path / "data"
    data.mkdir()
    installed = []
    with socket.socket() as occupied:  # make serve() fail fast instead of serving
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        cfg = SousConfig(
            data_dir=data,
            config_path=tmp_path / "config.toml",
            server_port=occupied.getsockname()[1],
        )
        cfg.config_path.write_text("")
        monkeypatch.setattr(server, "load_config", lambda: cfg)
        monkeypatch.setattr(server, "_install_shutdown_handler", lambda: installed.append(True))
        monkeypatch.setattr(server, "_drop_tqdm_process_lock", lambda: installed.append("tqdm"))
        with pytest.raises(SystemExit):
            server.main()
    assert True in installed, "main() served without installing the shutdown handler"
    assert "tqdm" in installed, "main() served without throwing the tqdm switch"
    assert sum(1 for h in root.handlers if h.name == SOUS_HANDLER_NAME) == 1


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

    with socket.socket() as occupied:  # make serve() fail fast instead of serving
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
    and endpoint routes hand it; any mlx state it created must be released
    before that thread can exit (ml-explore/mlx#4327)."""
    import sous.server as server

    service, _ = svc
    released = []
    monkeypatch.setattr(server, "release_mlx_thread_state", lambda: released.append(True))
    service.status_document()
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
        service.status_document()
    assert released


def test_server_status_reports_the_served_config(svc):
    """The status document is how a user confirms which model ids the daemon
    would serve locally, and at what window."""
    service, _ = svc
    assert service.status_document()["config"] == {
        "model_id": "mlx-community/Qwen3.8-27B-4bit",
        "idle_unload_minutes": 30,
        "port": 8383,
        "local_models": ["sous-local"],
        "max_context_tokens": 131072,
        "prompt_cache_disk_gb": None,
        "upstream_url": "https://api.anthropic.com",
        "generation_timeout_minutes": 30,
    }


def test_uvicorn_config_bounds_graceful_shutdown():
    """uvicorn holds SIGTERM while serving and, unbounded, waits for open
    connections — a long non-streaming endpoint turn would defer sous's own
    shutdown handler by minutes. The bound is what keeps `sous stop` prompt."""
    from sous.server import GRACEFUL_SHUTDOWN_SECONDS, uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_SECONDS == 5
    assert cfg.host == "127.0.0.1"
    # uvicorn's access log prints the raw request target, query string
    # included, at INFO — the daemon's own level.
    assert cfg.access_log is False


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
    doc = service.status_document()
    assert set(doc) == {"engine", "inflight", "config", "recent_turns"}
    assert set(doc["config"]) == {
        "model_id",
        "idle_unload_minutes",
        "port",
        "local_models",
        "max_context_tokens",
        "prompt_cache_disk_gb",
        "upstream_url",
        "generation_timeout_minutes",
    }
    assert set(service.status_document()) == {"engine", "inflight", "recent_turns", "config"}
    assert len(service.status_version()) == 3


def test_the_status_version_is_the_registry_the_engine_and_the_config_stamp(svc):
    service, _root = svc
    path = service.config.config_path
    assert not path.exists()
    assert service._config_stamp() == (-1, -1)  # no file to stamp
    path.write_text("[server]\nport = 8383\n")
    st = path.stat()
    assert service._config_stamp() == (st.st_mtime_ns, st.st_size)
    assert service.status_version() == (
        service.inflight.version,
        service.engines.version,
        (st.st_mtime_ns, st.st_size),
    )


def test_a_config_rewrite_under_the_old_mtime_still_moves_the_stamp(svc):
    """`cp -p` and a backup restore both put a different file back under the
    mtime it had; the size is what catches that one."""
    service, _root = svc
    path = service.config.config_path
    path.write_text("[server]\nport = 8383\n")
    before = path.stat()
    stamp = service._config_stamp()
    path.write_text('[server]\nport = 8383\nlocal_models = ["sous-local"]\n')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns  # the mtime is back
    assert service._config_stamp() != stamp
    assert service.status_version()[2] == service._config_stamp()


def test_the_lifespan_runs_the_idle_sweep(tmp_path: Path):
    cfg = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    engines = EngineManager(cfg, engine_factory=lambda mid: FakeEngine([]))
    app = create_server(engines, cfg, upstream=FakeUpstream().upstream())
    assert isinstance(app, Starlette)

    async def inner() -> None:
        async with app.router.lifespan_context(app):
            assert any(t.name == "sous-idle-sweep" for t in threading.enumerate())
        assert not any(t.name == "sous-idle-sweep" for t in threading.enumerate())

    anyio.run(inner)
