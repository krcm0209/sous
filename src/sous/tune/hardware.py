"""What the tune knows about the machine before any model is loaded."""

from __future__ import annotations

import dataclasses
import os
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from sous.engine.int8prefill import Availability

DISTRIBUTIONS = ("mlx", "mlx-vlm", "mlx-lm", "sous-mcp")
_GIB = 1 << 30


@dataclass(frozen=True)
class Hardware:
    chip: str
    memory_bytes: int
    working_set_bytes: int
    architecture: str
    nax: bool
    nax_reason: str | None
    macos: str
    versions: dict[str, str]
    hub_cache: str
    disk_free_bytes: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def gib(n: int) -> str:
    return f"{n / _GIB:.1f} GiB"


def _device_info() -> dict:
    import mlx.core as mx

    return dict(mx.device_info())


def _availability() -> Availability:
    from sous.engine import int8prefill

    return int8prefill.availability()


def hub_cache_dir() -> str:
    """Where the Hub keeps snapshots on this machine; the one lookup the
    tune's modules share."""
    from huggingface_hub import constants

    return str(constants.HF_HUB_CACHE)


def _nearest_existing(path: str) -> str:
    """Walk up from `path` to the nearest directory that exists: the Hub
    cache is created on its first download, so `disk_usage` on it raises
    `FileNotFoundError` on a machine that has never fetched a model."""
    current = path
    while current and not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current if current and os.path.exists(current) else os.path.expanduser("~")


def detect(
    *,
    device_info: Callable[[], dict] | None = None,
    mac_ver: Callable[[], tuple] | None = None,
    disk_usage: Callable[[str], tuple] | None = None,
    availability: Callable[[], Availability] | None = None,
    version: Callable[[str], str] | None = None,
    hub_cache: str | None = None,
) -> Hardware:
    """Every reading through an injectable so the tests need no GPU. The mlx
    call is a device query, not an op, so it is safe on any thread."""
    info = (device_info or _device_info)()
    release = (mac_ver or platform.mac_ver)()[0]
    cache = hub_cache if hub_cache is not None else hub_cache_dir()
    free = int((disk_usage or shutil.disk_usage)(_nearest_existing(cache))[2])
    nax = (availability or _availability)()
    read = version or dist_version
    versions: dict[str, str] = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = read(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return Hardware(
        chip=str(info.get("device_name", "unknown")),
        memory_bytes=int(info.get("memory_size", 0)),
        working_set_bytes=int(info.get("max_recommended_working_set_size", 0)),
        architecture=str(info.get("architecture", "")),
        nax=bool(getattr(nax, "available", False)),
        nax_reason=getattr(nax, "reason", None),
        macos=release,
        versions=versions,
        hub_cache=cache,
        disk_free_bytes=free,
    )
