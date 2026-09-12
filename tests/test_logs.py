"""The daemon's one line shape: `<iso-utc-ms>Z LEVEL name: message`, on a
handler that follows sys.stderr so pytest's capture sees it, installed
idempotently over whatever the MCP SDK put on the root logger."""

import logging
import re
import warnings
from datetime import UTC, datetime

import pytest

from sous.logs import (
    SOUS_HANDLER_NAME,
    StderrHandler,
    UTCFormatter,
    configure_daemon_logging,
    enable_warning_capture,
    format_line,
)

LINE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z (INFO|WARNING|ERROR) sous\.test: hello$"
)


class RichHandler(logging.Handler):
    """Stands in for rich.logging.RichHandler: the installer matches the SDK's
    handler by class name, and rich is only a transitive dependency here."""

    def emit(self, record: logging.LogRecord) -> None:
        pass


def _sous_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == SOUS_HANDLER_NAME]


@pytest.fixture(autouse=True)
def _clean_root():
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
    for h in before:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)
    logging.captureWarnings(False)


def test_formatter_prints_iso_utc_millis_level_and_name():
    record = logging.LogRecord("sous.test", logging.INFO, __file__, 1, "hello", None, None)
    record.created = 1789000000.123
    record.msecs = 123.0
    text = UTCFormatter().format(record)
    assert LINE.match(text), text
    # Pinned to the UTC rendering: a localtime regression would still match
    # the regex, so the exact stamp is the assertion that matters.
    assert text.startswith("2026-09-10T00:26:40.123Z "), text
    stamp = datetime.fromisoformat(text.split(" ", 1)[0].replace("Z", "+00:00"))
    assert stamp == datetime.fromtimestamp(1789000000.123, UTC).replace(microsecond=123000)


def test_format_line_matches_the_formatter_shape():
    text = format_line("WARNING", "sous.test", "hello")
    assert LINE.match(text), text
    assert " WARNING sous.test: hello" in text


def test_configure_is_idempotent_and_installs_exactly_one_handler():
    configure_daemon_logging()
    configure_daemon_logging()
    assert len(_sous_handlers()) == 1
    assert isinstance(_sous_handlers()[0], StderrHandler)
    assert isinstance(_sous_handlers()[0].formatter, UTCFormatter)
    assert logging.getLogger().level == logging.INFO


def test_configure_removes_the_sdks_rich_handler():
    root = logging.getLogger()
    root.addHandler(RichHandler())
    configure_daemon_logging()
    assert not [h for h in root.handlers if type(h).__name__ == "RichHandler"]


def test_handler_writes_to_the_current_sys_stderr(capsys):
    """Bound to sys.stderr at emit time: capsys swaps the stream per test,
    and a handler bound at construction would write past the capture."""
    configure_daemon_logging()
    logging.getLogger("sous.test").info("hello")
    err = capsys.readouterr().err
    assert LINE.match(err.strip()), err


def test_levels_are_spelled_out_in_the_line(capsys):
    configure_daemon_logging()
    logging.getLogger("sous.test").warning("hello")
    logging.getLogger("sous.test").error("hello")
    lines = capsys.readouterr().err.splitlines()
    assert " WARNING sous.test: hello" in lines[0]
    assert " ERROR sous.test: hello" in lines[1]


def test_warning_capture_routes_warnings_through_logging(caplog):
    enable_warning_capture()
    with caplog.at_level(logging.WARNING, logger="py.warnings"):
        warnings.warn("sous prompt cache: boom", stacklevel=1)
    assert any("sous prompt cache: boom" in r.getMessage() for r in caplog.records)
    assert all(r.name == "py.warnings" and r.levelno == logging.WARNING for r in caplog.records)
