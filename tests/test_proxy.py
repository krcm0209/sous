import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _daemon_command(data: Path, port: int) -> list[str]:
    return [sys.executable, "-m", "tests.daemon_stub", str(data), str(port)]


def _wait_for_port(port: int, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def test_ensure_daemon_starts_one_when_nothing_is_listening(tmp_path: Path):
    """A user who never ran install-launchd has no daemon; the proxy starts one."""
    from sous.proxy import ensure_daemon

    port = _free_port()
    data = tmp_path / "data"
    proc = None
    try:
        assert ensure_daemon(port, _daemon_command(data, port), wait_seconds=25.0)
        assert _wait_for_port(port, 1.0)
    finally:
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)
        if proc:
            proc.kill()


def test_ensure_daemon_does_not_spawn_when_one_is_already_listening(tmp_path: Path):
    """Second terminal: the port is open, so no second daemon is launched."""
    from sous.proxy import ensure_daemon

    port = _free_port()
    marker = tmp_path / "spawns.log"
    # A start command that records every invocation instead of starting anything.
    recorder = [sys.executable, "-c", f"open({str(marker)!r}, 'a').write('x')"]

    with socket.socket() as listening:
        listening.bind(("127.0.0.1", port))
        listening.listen(1)
        assert ensure_daemon(port, recorder, wait_seconds=5.0)

    assert not marker.exists(), "spawned a daemon while one was already listening"


def test_two_proxies_share_one_daemon(tmp_path: Path):
    """N proxies, one daemon, one model.

    The first proxy starts a daemon; the second must attach to it rather than
    start a second one, which would mean a second EngineManager and a second
    copy of the model resident.
    """
    from sous.proxy import ensure_daemon

    port = _free_port()
    data = tmp_path / "data"
    try:
        assert ensure_daemon(port, _daemon_command(data, port), wait_seconds=25.0)

        # Second proxy: hand it a start command that would be a loud failure if
        # it were ever run, proving it attached to the existing daemon instead.
        tripwire = [sys.executable, "-c", "raise SystemExit('second daemon started')"]
        assert ensure_daemon(port, tripwire, wait_seconds=5.0)

        holders = subprocess.run(
            ["pgrep", "-f", f"daemon_stub {data} {port}"], capture_output=True, text=True
        )
        assert len(holders.stdout.split()) == 1, "more than one daemon process is running"
    finally:
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)


def test_racing_proxies_leave_exactly_one_daemon(tmp_path: Path):
    """Two proxies starting at once both spawn; the flock settles it.

    The loser exits rather than running a second worker and loading a second
    model, so the race degrades to the same one-daemon outcome.
    """
    port = _free_port()
    data = tmp_path / "data"
    data.mkdir(parents=True)
    try:
        racers = [
            subprocess.Popen(
                _daemon_command(data, port),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            for _ in range(3)
        ]
        assert _wait_for_port(port, 25.0), "no daemon came up"
        time.sleep(2.0)  # let the losers lose

        survivors = subprocess.run(
            ["pgrep", "-f", f"daemon_stub {data} {port}"], capture_output=True, text=True
        )
        assert len(survivors.stdout.split()) == 1, "the lock did not settle the race"

        losers = [p for p in racers if p.poll() is not None]
        assert losers, "expected at least one racer to lose"
        complaints = [p.stderr.read().decode() for p in losers if p.stderr is not None]
        assert any("another daemon already holds" in c for c in complaints)
    finally:
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)


@pytest.mark.slow
def test_proxy_forwards_tool_calls_over_stdio(tmp_path: Path):
    """End to end: a real MCP stdio client -> `sous mcp` -> the HTTP daemon.

    Uses the SDK's own stdio client, i.e. the same transport Claude Desktop
    speaks, so the pump is exercised rather than simulated.
    """
    import anyio
    from mcp import ClientSession, StdioServerParameters, stdio_client

    port = _free_port()
    data = tmp_path / "data"
    daemon = subprocess.Popen(
        _daemon_command(data, port),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert _wait_for_port(port, 25.0), "daemon never came up"

    proxy_src = (
        "from pathlib import Path;"
        "from sous.config import SousConfig;"
        "from sous.proxy import run;"
        f"run(SousConfig(data_dir=Path({str(data)!r}), "
        f"config_path=Path({str(data / 'config.toml')!r}), server_port={port}))"
    )

    async def drive() -> list[str]:
        params = StdioServerParameters(command=sys.executable, args=["-c", proxy_src])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            result = await session.call_tool("server_status", {})
            assert result is not None
            return [t.name for t in listed.tools]

    try:
        names = anyio.run(drive)
        assert "server_status" in names and "delegate_task" in names
    finally:
        daemon.kill()
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)
