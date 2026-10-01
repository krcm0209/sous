"""The tensor-unit platform rule int8 prefill and the attention tile share
(sous.engine.nax). The platform and the device are faked, so the table runs on
any Mac, CI's macos-15 included."""

import platform

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax  # noqa: E402 — after the importorskip guard


def _fake_platform(monkeypatch, mac_ver: str, arch: str):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: (mac_ver, ("", "", ""), "arm64"))
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})


@pytest.mark.parametrize(
    ("mac_ver", "arch", "reason"),
    [
        ("26.6.2", "applegpu_g17s", None),
        ("26.2", "applegpu_g17g", None),
        ("26.6.2", "applegpu_g18p", None),
        (
            "26.6.2",
            "applegpu_g17p",
            "GPU applegpu_g17p predates the neural accelerators (gen 17 < 18)",
        ),
        (
            "26.6.2",
            "applegpu_g16s",
            "GPU applegpu_g16s predates the neural accelerators (gen 16 < 17)",
        ),
        ("26.1", "applegpu_g17s", "macOS 26.1 < 26.2"),
        ("26.6.2", "something_else", "unrecognized GPU architecture 'something_else'"),
    ],
)
def test_platform_reason_mirrors_mlx_nax_rule(monkeypatch, mac_ver, arch, reason):
    _fake_platform(monkeypatch, mac_ver, arch)
    assert nax.platform_reason() == reason


def test_platform_reason_refuses_other_systems(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert nax.platform_reason() == "not macOS"


def test_platform_reason_refuses_a_machine_without_metal(monkeypatch):
    _fake_platform(monkeypatch, "26.6.2", "applegpu_g17s")
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert nax.platform_reason() == "Metal unavailable"


def test_int8_availability_takes_the_shared_rule_before_its_gemm_probe(monkeypatch):
    """One monkeypatch point for both kernels, and a refused platform never
    compiles the GEMM."""

    def compiled():
        raise AssertionError("the GEMM probe ran on a refused platform")

    monkeypatch.setattr(nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(int8prefill, "_gemm_probe", compiled)
    assert int8prefill.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_int8prefill_keeps_exporting_the_shared_availability():
    """tune/hardware.py and the tune tests import it from int8prefill."""
    assert int8prefill.Availability is nax.Availability
