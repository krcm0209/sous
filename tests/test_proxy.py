import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _daemon_command(data: Path, port: int) -> list[str]:
    return [sys.executable, "-m", "tests.daemon_stub", str(data), str(port)]


def _readline(stream, seconds: float) -> bytes:
    """Read one line, giving up after `seconds` instead of blocking forever."""
    result: list[bytes] = []
    reader = threading.Thread(target=lambda: result.append(stream.readline()), daemon=True)
    reader.start()
    reader.join(seconds)
    return result[0] if result else b""


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


@pytest.mark.slow
def test_proxy_exits_when_the_daemon_dies(tmp_path: Path):
    """A dead daemon must end the bridge, not leave it waiting on stdin.

    Both SDK transports own background task groups. With the bridge's own task
    group innermost, its __aexit__ absorbs the cancellation before those
    transports unwind, and stdio_server() goes on waiting for stdin forever —
    the client sees a live server with a dead backend behind it.
    """
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
    proxy = subprocess.Popen(
        [sys.executable, "-c", proxy_src],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        # A real session first. An idle proxy has no live daemon stream to
        # notice a death on — the HTTP client only opens its SSE stream once
        # `initialized` arrives — so skipping the handshake would test nothing.
        assert proxy.stdin is not None and proxy.stdout is not None
        proxy.stdin.write(
            b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":'
            b'"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}\n'
        )
        proxy.stdin.flush()
        assert _readline(proxy.stdout, 20.0), "no initialize response through the bridge"
        proxy.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        proxy.stdin.flush()
        time.sleep(2.0)
        assert proxy.poll() is None, "proxy died before the daemon did"

        daemon.kill()
        daemon.wait()

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and proxy.poll() is None:
            time.sleep(0.3)
        assert proxy.poll() is not None, "proxy still running after the daemon died"
    finally:
        proxy.kill()
        daemon.kill()
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)


class _StdioTransport:
    """Adapts the stdio client to the Client transport protocol."""

    def __init__(self, params):
        self._params = params
        self._cm = None

    async def __aenter__(self):
        from mcp.client.stdio import stdio_client

        self._cm = stdio_client(self._params)
        return await self._cm.__aenter__()

    async def __aexit__(self, *exc):
        assert self._cm is not None
        return await self._cm.__aexit__(*exc)


@pytest.mark.slow
def test_modern_protocol_client_works_through_the_bridge(tmp_path: Path):
    """A `mode='auto'` client (which probes `server/discover`) must still work.

    stdio carries no HTTP headers, so the bridge has no `Mcp-Method`/`Mcp-Name`
    metadata to forward and the daemon sees the session as legacy — the SDK's
    documented fallback. Tools must still resolve and call. The bridge does not
    synthesize those headers: the daemon's validation ladder checks them only
    when present, so omitting them is skipped whereas a wrong guess would be a
    HEADER_MISMATCH rejection.
    """
    import anyio
    from mcp import StdioServerParameters
    from mcp.client import Client

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

    async def drive():
        params = StdioServerParameters(command=sys.executable, args=["-c", proxy_src])
        async with Client(_StdioTransport(params), mode="auto") as client:
            listed = await client.list_tools()
            result = await client.call_tool("server_status", {})
            return [t.name for t in listed.tools], result

    try:
        names, result = anyio.run(drive)
        assert "server_status" in names and "delegate_task" in names
        assert result is not None
    finally:
        daemon.kill()
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)


@pytest.mark.slow
def test_proxy_exits_when_the_daemon_restarts(tmp_path: Path):
    """A restarted daemon leaves the bridge alive but useless — it must exit.

    launchd's KeepAlive puts a new daemon on the same port within a second, so
    a liveness check never sees it go down; meanwhile every old session is gone
    and each call comes back "Session not found" forever. The client cannot
    recover, because a bridge that stays alive is never restarted. Identity, not
    liveness, is the signal — hence the pid in daemon.lock.
    """
    port = _free_port()
    data = tmp_path / "data"
    first = subprocess.Popen(
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
    proxy = subprocess.Popen(
        [sys.executable, "-c", proxy_src],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    second = None
    try:
        assert proxy.stdin is not None and proxy.stdout is not None
        proxy.stdin.write(
            b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":'
            b'"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}\n'
        )
        proxy.stdin.flush()
        assert _readline(proxy.stdout, 20.0), "no initialize response through the bridge"
        proxy.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        proxy.stdin.flush()
        time.sleep(2.0)
        assert proxy.poll() is None, "proxy died before the restart"

        first.kill()
        first.wait()
        second = subprocess.Popen(
            _daemon_command(data, port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert _wait_for_port(port, 25.0), "replacement daemon never came up"

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and proxy.poll() is None:
            time.sleep(0.3)
        assert proxy.poll() is not None, "bridge survived a daemon restart as a dead session"
    finally:
        proxy.kill()
        first.kill()
        if second:
            second.kill()
        subprocess.run(["pkill", "-f", f"daemon_stub {data} {port}"], check=False)
