from sous.engine.int8prefill import Availability
from sous.tune.hardware import Hardware, detect, gib


def _info():
    return {
        "device_name": "Apple M5 Pro",
        "memory_size": 68719476736,
        "max_recommended_working_set_size": 55662788608,
        "architecture": "applegpu_g17s",
    }


def test_detect_reads_every_field_through_its_injection_point(tmp_path):
    hw = detect(
        device_info=_info,
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda path: (1, 2, 412 * 2**30),
        availability=lambda: Availability(True),
        version=lambda name: {
            "mlx": "0.32.2",
            "mlx-vlm": "0.7.1",
            "mlx-lm": "0.31.3",
            "sous-mcp": "0.6.1",
        }[name],
        hub_cache=str(tmp_path),
    )
    assert hw == Hardware(
        chip="Apple M5 Pro",
        memory_bytes=68719476736,
        working_set_bytes=55662788608,
        architecture="applegpu_g17s",
        nax=True,
        nax_reason=None,
        macos="26.6.2",
        versions={"mlx": "0.32.2", "mlx-vlm": "0.7.1", "mlx-lm": "0.31.3", "sous-mcp": "0.6.1"},
        hub_cache=str(tmp_path),
        disk_free_bytes=412 * 2**30,
    )
    assert hw.as_dict()["working_set_bytes"] == 55662788608


def test_a_pre_m5_gpu_reports_why_int8_is_unavailable(tmp_path):
    hw = detect(
        device_info=lambda: {**_info(), "device_name": "Apple M2", "architecture": "applegpu_g14g"},
        mac_ver=lambda: ("15.5", ("", "", ""), "arm64"),
        disk_usage=lambda path: (1, 2, 3),
        availability=lambda: Availability(False, "macOS 15.5 < 26.2"),
        version=lambda name: "x",
        hub_cache=str(tmp_path),
    )
    assert hw.nax is False and hw.nax_reason == "macOS 15.5 < 26.2"


def test_a_missing_distribution_reads_as_absent_not_a_crash(tmp_path):
    from importlib.metadata import PackageNotFoundError

    def version(name):
        raise PackageNotFoundError(name)

    hw = detect(
        device_info=_info,
        mac_ver=lambda: ("26.6.2", ("", "", ""), "arm64"),
        disk_usage=lambda p: (1, 2, 3),
        availability=lambda: Availability(True),
        version=version,
        hub_cache=str(tmp_path),
    )
    assert hw.versions == {
        "mlx": "absent",
        "mlx-vlm": "absent",
        "mlx-lm": "absent",
        "sous-mcp": "absent",
    }


def test_gib_formats_one_decimal():
    assert gib(55662788608) == "51.8 GiB"
    assert gib(0) == "0.0 GiB"
