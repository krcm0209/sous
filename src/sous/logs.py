"""One shape for every line the daemon writes.

`<iso-utc-ms>Z LEVEL name: message` — the timestamp launchd's log never
had, the level the filename `daemon.err.log` falsely promised, and the
logger name (`sous.gateway`, `sous.engine`, `mcp.…`, `uvicorn.error`) that
says which part spoke. Installed on the root logger, replacing the
`RichHandler` the MCP SDK's `configure_logging("INFO")` puts there from
`MCPServer.__init__` — that handler wraps every record at 80 columns, which
made the log ungreppable.
"""

from __future__ import annotations

import logging
import sys
import time

SOUS_HANDLER_NAME = "sous-daemon-log"


class UTCFormatter(logging.Formatter):
    """`2026-09-10T19:26:14.025Z INFO sous.gateway: …` — UTC, milliseconds,
    Python's level names. Exceptions are appended the way logging always
    formats them."""

    def format(self, record: logging.LogRecord) -> str:
        line = format_line(
            record.levelname, record.name, record.getMessage(), record.created, record.msecs
        )
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def format_line(
    level: str, name: str, message: str, created: float | None = None, msecs: float | None = None
) -> str:
    """The line shape, for the formatter and for the one place that cannot
    log: the SIGTERM handler in server.py, which runs asynchronously to
    whatever thread holds logging's lock and must print instead."""
    now = time.time() if created is None else created
    millis = int((now % 1) * 1000) if msecs is None else int(msecs)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    return f"{stamp}.{millis:03d}Z {level} {name}: {message}"


class StderrHandler(logging.StreamHandler):
    """A StreamHandler whose stream is whatever `sys.stderr` is at emit time.

    The stdlib's own `logging._StderrHandler` (behind `lastResort`) works this
    way for the same reason: the daemon is one process for hours and pytest
    swaps `sys.stderr` per test, so a stream captured at construction is the
    wrong one later. The setter is a no-op so `StreamHandler.__init__` and
    `setStream` cannot pin a stale stream."""

    def __init__(self) -> None:
        super().__init__()
        self.set_name(SOUS_HANDLER_NAME)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


def configure_daemon_logging() -> None:
    """Install the daemon's handler on the root logger, once.

    Idempotent, so `create_server` can call it every time it assembles the
    app (tests build many). Removes any earlier copy of this handler and the
    SDK's RichHandler (matched by class name: rich is a transitive dependency
    and this module must not import it). Sets the root level to INFO when it
    is unset or higher — the SDK sets INFO too, but this must not depend on
    call order. Library logger *levels* are untouched: the gateway pins
    sse-starlette/httpx/httpcore above where they log bodies and URLs, and
    a handler swap must never loosen that."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == SOUS_HANDLER_NAME or type(handler).__name__ == "RichHandler":
            root.removeHandler(handler)
    handler = StderrHandler()
    handler.setFormatter(UTCFormatter())
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def enable_warning_capture() -> None:
    """Route `warnings.warn` (the prompt cache's degradation notices) through
    logging as `WARNING py.warnings: …`, so they take the line shape too.
    Called from the daemon entry point only: under pytest the warnings plugin
    owns `showwarning`, and tests assert on `pytest.warns`."""
    logging.captureWarnings(True)
