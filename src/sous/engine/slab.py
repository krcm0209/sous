"""The slab weight layout the projection tile reads, its inverse, and the marker a
slab-only module carries.

One linear's (or one stack's) affine-Q4 group-64 weights become three flat planes,
N zero-padded to whole 256-column tiles: codes ``[T][G][256][8 words]``, each
32-byte unit one column's one group of 64 codes, mlx's bytes verbatim (low nibble
first, nothing requantised); and bf16 scales and biases ``[T][G][256]``. One
group of one tile is then a contiguous 8 KiB of codes and 512 bytes of each
plane. The repack is a pure block transpose, so the round trip is bitwise, and
padded columns hold zero codes, scales and biases, which compute exactly 0.

The layout is Splash's (incoai/splash), under the Apache License 2.0.

Nothing of sous is imported here, so projkernel, int8prefill and projtile can all
read the marker without an import cycle. mlx is imported inside each function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TILE = 256
UNIT_BYTES = 32
GROUP = 64
# Set with object.__setattr__ so it lives in the module's __dict__, out of mlx's
# parameter tree; every reader of a module's arrays tests it first.
SLAB_ATTR = "_sous_slab"
# Where a slab-only module sits in its model, set beside the marker the same way:
# a call no slab path can serve fails naming it, since mlx's own kernels cannot
# read the planes either.
PATH_ATTR = "_sous_slab_path"

_UNIT_WORDS = UNIT_BYTES // 4


@dataclass(frozen=True)
class SlabInfo:
    """What the flat planes no longer say: the real width, K and the tile count."""

    n_real: int
    k: int
    tiles: int

    @property
    def n_pad(self) -> int:
        return self.tiles * TILE

    @property
    def groups(self) -> int:
        return self.k // GROUP

    @property
    def units(self) -> int:
        """(column, group) units the planes hold, padding included: the scales'
        and the biases' size, each a unit's one value."""
        return self.n_pad * self.groups

    @property
    def code_words(self) -> int:
        """The codes plane's size in uint32 words: one 32-byte unit per (column,
        group)."""
        return self.units * UNIT_BYTES // 4


def repack(w: Any, scales: Any, biases: Any) -> tuple[Any, Any, Any, SlabInfo]:
    """mlx's ``weight`` [N, K/8] uint32 and ``scales``/``biases`` [N, K/64] as the
    flat slab planes, zero-padded to whole tiles and evaluated. Refuses any other
    layout: a group size other than 64 would transpose without error into planes
    every reader misaddresses."""
    import mlx.core as mx

    if w.ndim != 2 or w.dtype != mx.uint32:
        raise ValueError(f"repack takes a [N, K/8] uint32 weight, got {w.dtype} {w.shape}")
    n, k = w.shape[0], w.shape[1] * 8
    groups = k // GROUP
    if k % GROUP or scales.shape != (n, groups) or biases.shape != (n, groups):
        raise ValueError(
            f"repack takes affine 4-bit group-{GROUP} planes, got weight {w.shape}, "
            f"scales {scales.shape}, biases {biases.shape}"
        )
    info = SlabInfo(n_real=n, k=k, tiles=-(-n // TILE))
    pad = info.n_pad - n
    if pad:
        w, scales, biases = (mx.pad(a, [(0, pad), (0, 0)]) for a in (w, scales, biases))
    codes = mx.contiguous(
        w.reshape(info.tiles, TILE, groups, _UNIT_WORDS).transpose(0, 2, 1, 3)
    ).reshape(-1)
    scales, biases = (
        mx.contiguous(p.reshape(info.tiles, TILE, groups).transpose(0, 2, 1)).reshape(-1)
        for p in (scales, biases)
    )
    mx.eval(codes, scales, biases)
    return codes, scales, biases, info


def unpack(codes: Any, scales: Any, biases: Any, info: SlabInfo) -> tuple[Any, Any, Any]:
    """repack's inverse, lazy: ``weight`` [n_real, K/8], ``scales`` and ``biases``
    [n_real, K/64], the padding dropped."""
    t, g = info.tiles, info.groups
    w = codes.reshape(t, g, TILE, _UNIT_WORDS).transpose(0, 2, 1, 3)
    w = w.reshape(info.n_pad, g * _UNIT_WORDS)
    s, b = (
        p.reshape(t, g, TILE).transpose(0, 2, 1).reshape(info.n_pad, g) for p in (scales, biases)
    )
    return w[: info.n_real], s[: info.n_real], b[: info.n_real]


def slab_info(module: Any) -> SlabInfo | None:
    """The marker of a slab-only module, None for one holding mlx's layout. Read
    from the instance dict, where object.__setattr__ put it: a getattr that misses
    goes through nn.Module's __getattr__ and raises inside it, six times the cost,
    on a path every projection call of a decode takes."""
    return vars(module).get(SLAB_ATTR)
