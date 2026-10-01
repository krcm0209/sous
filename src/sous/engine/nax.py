"""The tensor-unit ("NAX") platform rule sous's tensor-op kernels share.

Both runtime-compiled kernels that use the Metal-4 tensor ops, int8 prefill's GEMM
and the attention tile's split kernel, serve only where mlx itself would use the
neural accelerators. On an older GPU the tensor ops run on the plain shader path,
where neither kernel buys anything, so it is refused rather than merely slow.
This module holds that rule and nothing else: no numerics, and mlx is imported
only when the rule is read.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass

_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


def platform_reason() -> str | None:
    """None when mlx 0.32.2's own is_nax_available() (device.cpp) would pass:
    macOS >= 26.2 and an Apple GPU of generation >= 17 (>= 18 for 'p'-suffix
    parts), the M5 family and later. Otherwise why not. Read at call time, so a
    test that fakes the platform or the device reaches every kernel's caller."""
    if platform.system() != "Darwin":
        return "not macOS"
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return f"mlx unavailable: {e}"
    if not mx.metal.is_available():
        return "Metal unavailable"
    release = platform.mac_ver()[0]
    parts = tuple(int(p) for p in release.split(".") if p.isdigit())
    if parts[:2] < (26, 2):
        return f"macOS {release} < 26.2"
    arch = str(mx.device_info().get("architecture", ""))
    match = _ARCH.match(arch)
    if match is None:
        return f"unrecognized GPU architecture {arch!r}"
    gen = int(match.group(1))
    need = 18 if match.group(2) == "p" else 17
    if gen < need:
        return f"GPU {arch} predates the neural accelerators (gen {gen} < {need})"
    return None
