"""The GPU's core count, read from the IORegistry: the attention tile's split target.

mlx exposes no core count (device_info carries the name and the architecture
only), so it is read from the AGXAccelerator service's "gpu-core-count", the
figure System Information shows. IOKit through ctypes first; /usr/sbin/ioreg as
the fallback when IOKit itself cannot be called. Never raises, and never imports
mlx: the caller pairs the reading with mlx's device name.
"""

from __future__ import annotations

import ctypes
import functools
import plistlib
import subprocess
from dataclasses import dataclass
from typing import Any

_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_IOREG = "/usr/sbin/ioreg"
_IOREG_TIMEOUT_S = 5.0
_SERVICE_CLASS = b"AGXAccelerator"
_KEYS = ("gpu-core-count", "model")
_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_SINT64 = 4  # kCFNumberSInt64Type


@dataclass(frozen=True)
class GPUCores:
    cores: int | None
    model: str | None
    reason: str | None  # None when cores and model were read
    source: str  # "iokit", "ioreg" or "none"


def parse_services(services: list[dict[str, Any]], source: str) -> GPUCores:
    """Exactly one service with an integer core count > 0 and a model string."""
    if not services:
        return GPUCores(None, None, "no AGXAccelerator service", source)
    if len(services) > 1:
        return GPUCores(None, None, f"{len(services)} AGXAccelerator services", source)
    cores = services[0].get("gpu-core-count")
    model = services[0].get("model")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores <= 0:
        return GPUCores(None, None, f"gpu-core-count is {cores!r}", source)
    if not isinstance(model, str) or not model:
        return GPUCores(None, None, f"model is {model!r}", source)
    return GPUCores(cores, model, None, source)


def _iokit_services() -> list[dict[str, Any]]:
    """Every AGXAccelerator service's two properties; OSError when IOKit cannot be used."""
    iokit = ctypes.CDLL(_IOKIT)
    cf = ctypes.CDLL(_CORE_FOUNDATION)
    vp, u32 = ctypes.c_void_p, ctypes.c_uint32
    iokit.IOServiceMatching.restype = vp
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
    iokit.IOServiceGetMatchingServices.argtypes = [u32, vp, ctypes.POINTER(u32)]
    iokit.IOIteratorNext.restype = u32
    iokit.IOIteratorNext.argtypes = [u32]
    iokit.IORegistryEntryCreateCFProperty.restype = vp
    iokit.IORegistryEntryCreateCFProperty.argtypes = [u32, vp, vp, u32]
    iokit.IOObjectRelease.restype = ctypes.c_int
    iokit.IOObjectRelease.argtypes = [u32]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, u32]
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [vp]
    cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetTypeID.restype = ctypes.c_ulong
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, u32]
    cf.CFRelease.restype = None
    cf.CFRelease.argtypes = [vp]

    def value(ref: int) -> Any:
        tid = cf.CFGetTypeID(ref)
        if tid == cf.CFNumberGetTypeID():
            out = ctypes.c_int64()
            return out.value if cf.CFNumberGetValue(ref, _SINT64, ctypes.byref(out)) else None
        if tid == cf.CFStringGetTypeID():
            buf = ctypes.create_string_buffer(256)
            if cf.CFStringGetCString(ref, buf, len(buf), _UTF8):
                return buf.value.decode("utf-8")
        return None

    matching = iokit.IOServiceMatching(_SERVICE_CLASS)
    if not matching:
        raise OSError("IOServiceMatching returned NULL")
    iterator = u32(0)
    # kIOMainPortDefault is MACH_PORT_NULL; the call consumes `matching`.
    kr = iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator))
    if kr != 0:
        raise OSError(f"IOServiceGetMatchingServices returned {kr:#x}")
    keys = {k: cf.CFStringCreateWithCString(None, k.encode(), _UTF8) for k in _KEYS}
    services: list[dict[str, Any]] = []
    try:
        while service := iokit.IOIteratorNext(iterator):
            try:
                props: dict[str, Any] = {}
                for name, key in keys.items():
                    ref = iokit.IORegistryEntryCreateCFProperty(service, key, None, 0)
                    if ref:
                        try:
                            props[name] = value(ref)
                        finally:
                            cf.CFRelease(ref)
                services.append(props)
            finally:
                iokit.IOObjectRelease(service)
    finally:
        for key in keys.values():
            if key:
                cf.CFRelease(key)
        iokit.IOObjectRelease(iterator)
    return services


def _ioreg_services() -> list[dict[str, Any]]:
    """The same two properties from ioreg's plist; OSError/ValueError when unreadable."""
    out = subprocess.run(
        [_IOREG, "-a", "-r", "-c", _SERVICE_CLASS.decode(), "-d", "1"],
        capture_output=True,
        check=True,
        timeout=_IOREG_TIMEOUT_S,
    ).stdout
    if not out.strip():  # ioreg prints nothing when no service matches
        return []
    entries = plistlib.loads(out)
    if not isinstance(entries, list):
        raise ValueError("ioreg did not return a plist array")
    return [{k: e.get(k) for k in _KEYS} for e in entries if isinstance(e, dict)]


@functools.cache
def read() -> GPUCores:
    """Read once per process: the core count cannot change under a running daemon."""
    try:
        return parse_services(_iokit_services(), "iokit")
    except OSError, AttributeError, ValueError:
        pass
    try:
        return parse_services(_ioreg_services(), "ioreg")
    except Exception as e:  # noqa: BLE001 — plistlib leaks ExpatError and IndexError; never raises
        return GPUCores(None, None, f"GPU core count unreadable: {e}", "none")


def matched_cores(device_name: str) -> tuple[int | None, str | None]:
    """(cores, None) when the IORegistry's GPU is the one mlx runs on, else
    (None, why not). Through the module global, so `gpucores.read` is the one
    place a test sets the split target."""
    reading = read()
    if reading.reason is not None:
        return None, reading.reason
    if reading.model != device_name:
        return None, f"IORegistry GPU {reading.model!r} is not mlx's {device_name!r}"
    return reading.cores, None
