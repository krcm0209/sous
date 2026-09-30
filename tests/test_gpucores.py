"""The GPU core count (sous.engine.gpucores): IORegistry parsing, the ioreg
fallback, the match against mlx's device name, and one live read on a Mac."""

import os
import plistlib
import subprocess
import sys

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores  # noqa: E402 — after the importorskip guard


@pytest.fixture(autouse=True)
def _fresh_read():
    gpucores.read.cache_clear()
    yield
    gpucores.read.cache_clear()


@pytest.mark.parametrize(
    ("services", "expected"),
    [
        (
            [{"gpu-core-count": 20, "model": "Apple M5 Pro"}],
            gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"),
        ),
        ([], gpucores.GPUCores(None, None, "no AGXAccelerator service", "iokit")),
        (
            [{"gpu-core-count": 8, "model": "Apple M2"}] * 2,
            gpucores.GPUCores(None, None, "2 AGXAccelerator services", "iokit"),
        ),
        (
            [{"gpu-core-count": True, "model": "Apple M2"}],
            gpucores.GPUCores(None, None, "gpu-core-count is True", "iokit"),
        ),
        (
            [{"gpu-core-count": 0, "model": "Apple M2"}],
            gpucores.GPUCores(None, None, "gpu-core-count is 0", "iokit"),
        ),
        (
            [{"gpu-core-count": 8, "model": None}],
            gpucores.GPUCores(None, None, "model is None", "iokit"),
        ),
    ],
)
def test_parse_services_needs_one_service_with_a_core_count_and_a_model(services, expected):
    assert gpucores.parse_services(services, "iokit") == expected


def _unusable():
    raise OSError("IOKit unusable")


def test_read_falls_back_to_ioreg_when_iokit_cannot_be_used(monkeypatch):
    monkeypatch.setattr(gpucores, "_iokit_services", _unusable)
    monkeypatch.setattr(
        gpucores, "_ioreg_services", lambda: [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]
    )
    assert gpucores.read() == gpucores.GPUCores(20, "Apple M5 Pro", None, "ioreg")


def test_read_reports_why_when_neither_reader_works(monkeypatch):
    def timed_out():
        raise subprocess.TimeoutExpired(["/usr/sbin/ioreg"], 5.0)

    monkeypatch.setattr(gpucores, "_iokit_services", _unusable)
    monkeypatch.setattr(gpucores, "_ioreg_services", timed_out)
    reading = gpucores.read()
    assert (reading.cores, reading.model, reading.source) == (None, None, "none")
    assert reading.reason is not None
    assert reading.reason.startswith("GPU core count unreadable")


def _fake_ioreg(monkeypatch, stdout: bytes) -> list:
    seen = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_ioreg_services_parse_the_plist_from_an_absolute_path_with_a_timeout(monkeypatch):
    entries = [{"gpu-core-count": 20, "model": "Apple M5 Pro", "IOClass": "AGXAcceleratorG17X"}]
    seen = _fake_ioreg(monkeypatch, plistlib.dumps(entries))
    assert gpucores._ioreg_services() == [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]
    (argv, kwargs), *_ = seen
    assert argv == ["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"]
    assert kwargs["timeout"] == 5.0 and kwargs["check"] is True


@pytest.mark.parametrize(
    "stdout",
    [
        plistlib.dumps([{"gpu-core-count": 8, "model": "Apple M2"}])[:60],  # truncated: ExpatError
        b"<plist><key>a</key></plist>",  # a key without a value: IndexError
    ],
)
def test_read_reports_a_malformed_ioreg_plist_instead_of_raising(monkeypatch, stdout):
    """plistlib's parser leaks expat's and its own stack errors, none a ValueError."""
    monkeypatch.setattr(gpucores, "_iokit_services", _unusable)
    _fake_ioreg(monkeypatch, stdout)
    reading = gpucores.read()
    assert (reading.cores, reading.model, reading.source) == (None, None, "none")
    assert reading.reason is not None
    assert reading.reason.startswith("GPU core count unreadable")


def test_ioreg_services_are_empty_when_nothing_matches(monkeypatch):
    _fake_ioreg(monkeypatch, b"")
    assert gpucores._ioreg_services() == []


def test_ioreg_services_refuse_a_plist_that_is_not_an_array(monkeypatch):
    _fake_ioreg(monkeypatch, plistlib.dumps({"model": "Apple M5 Pro"}))
    with pytest.raises(ValueError, match="not return a plist array"):
        gpucores._ioreg_services()


def _reading(monkeypatch, reading):
    monkeypatch.setattr(gpucores, "read", lambda: reading)


def test_matched_cores_are_the_count_when_the_models_agree(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"))
    assert gpucores.matched_cores("Apple M5 Pro") == (20, None)


def test_matched_cores_refuse_another_gpu(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(20, "Apple M5 Pro", None, "iokit"))
    assert gpucores.matched_cores("Apple M5 Max") == (
        None,
        "IORegistry GPU 'Apple M5 Pro' is not mlx's 'Apple M5 Max'",
    )


def test_matched_cores_carry_the_readers_reason(monkeypatch):
    _reading(monkeypatch, gpucores.GPUCores(None, None, "no AGXAccelerator service", "iokit"))
    assert gpucores.matched_cores("Apple M5 Pro") == (None, "no AGXAccelerator service")


def test_read_happens_once_per_process(monkeypatch):
    seen = []

    def services():
        seen.append(1)
        return [{"gpu-core-count": 20, "model": "Apple M5 Pro"}]

    monkeypatch.setattr(gpucores, "_iokit_services", services)
    assert gpucores.read() is gpucores.read()
    assert seen == [1]


@pytest.mark.skipif(sys.platform != "darwin", reason="the IORegistry is macOS's")
@pytest.mark.skipif(
    bool(os.environ.get("CI")), reason="CI never reads IOKit; S is monkeypatched there"
)
def test_a_live_read_never_raises_and_names_mlxs_gpu():
    """A virtualised GPU may expose no AGXAccelerator service, so only a
    reading that succeeded is checked against mlx."""
    reading = gpucores.read()
    if reading.reason is not None:
        return
    assert reading.cores is not None and reading.cores > 0
    assert reading.source in ("iokit", "ioreg")
    assert reading.model == mx.device_info()["device_name"]
