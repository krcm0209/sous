"""Shared helpers for the projection tile's tests: the tiny qwen3_5 model with its
MLP held slab-only (and, at scope "all", its attention and GDN linears stacked), a
projection whose width pads, the `tile_nax` skip marker, plans for the tiny model's
keys, plain-mlx stand-ins for the tile's two launches, and the routing state's
guard, its decode window and the hooks with the stand-ins behind them.

mlx is imported inside each helper, so importing this module needs only what
`projtile.availability()` itself imports."""

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

from sous.engine import int8prefill, projkernel, projtile, slab
from tests import proj_fixtures as pfx

# Read once: a failed compile is not remembered, so each call would compile again.
TILE = projtile.availability()
tile_nax = pytest.mark.skipif(
    not TILE.available, reason=f"projection tile unavailable: {TILE.reason}"
)

# One plan for every dispatch key of the tiny model: its K are 256, 512 and 1024 and
# every padded width a multiple of 256, so TN 128 and no split fit them all.
TINY_PLAN = projtile.Plan(128, 4, 2, True)
TINY_PLANS: dict[projtile.Key, projtile.Plan] = {
    key: TINY_PLAN
    for key in (
        (256, (2048, 512, 512)),  # q+k+v
        (256, (512, 512)),  # gate+up, the dual
        (256, (512, 256)),  # in_proj_qkv+in_proj_z
        (256, (4, 4)),  # in_proj_b+in_proj_a
        (512, (256,)),  # down
        (1024, (256,)),  # o_proj
        (256, (256,)),  # out_proj, in_proj_z alone
        (256, (512,)),  # gate, up, k, v, in_proj_qkv alone
        (256, (2048,)),  # q alone
        (256, (4,)),  # in_proj_b, in_proj_a alone
    )
}


def tiny_slab_model(*, seed: int = 0, scope: str = "all") -> Any:
    """pfx.tiny_quantized_model() with every gate, up and down slab-only and, at
    scope "all", every attention and GDN group stacked whatever its key. Its K are
    256, 512 and 1024 and every MLP width a multiple of 256, so nothing on its MLP
    pads (b+a, N 8, is the one stack that does)."""
    if scope not in ("all", "mlp"):
        raise ValueError(f'scope must be "all" or "mlp", got {scope!r}')
    root = pfx.tiny_quantized_model(seed=seed)
    projtile.convert_mlp(root)
    if scope == "all":
        projtile.build_stacks(root, keys_with_plans=lambda key: True)
    return root


def padded_module(n: int, k: int, seed: int) -> Any:
    """pfx.rand_module() at a width N off a multiple of the slab tile, so its planes
    carry zero columns past N."""
    if n % slab.TILE == 0:
        raise ValueError(f"padded_module needs N off a multiple of {slab.TILE}, got {n}")
    return pfx.rand_module(n, k, seed)


def tiny_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan table with the tiny model's keys added, for this test only."""
    monkeypatch.setattr(projtile, "PLANS", {**projtile.PLANS, **TINY_PLANS})


def _check_launch(x8: Any, info: slab.SlabInfo, plan: projtile.Plan, *, dual: bool) -> None:
    import mlx.core as mx

    reason = projtile.check_plan(info, plan, dual=dual)
    if reason is not None:
        raise ValueError(f"stand-in tile: {reason}")
    if tuple(x8.shape) != (projtile.MAX_ROWS, info.k) or x8.dtype != mx.bfloat16:
        raise ValueError(f"stand-in tile takes [8, {info.k}] bf16 rows, got {x8.shape}")


def stand_in_tile(
    x8: Any, codes: Any, scales: Any, biases: Any, info: slab.SlabInfo, plan: projtile.Plan
) -> Any:
    """A plain-mlx model of projtile.tile's contract: the planes unpacked and the
    eight padded rows run through pfx.stand_in_mma's arithmetic, each row on its
    own, then the columns past N zero, so [8, n_pad] bf16 as the kernel writes it.
    It refuses what the kernel refuses."""
    import mlx.core as mx

    _check_launch(x8, info, plan, dual=False)
    (y,) = pfx.stand_in_mma(x8, [projkernel.Weights(codes, scales, biases, info)])
    pad = info.n_pad - info.n_real
    return mx.pad(y, [(0, 0), (0, pad)]) if pad else y


def stand_in_tile_dual(
    x8: Any,
    gate_planes: tuple[Any, Any, Any],
    up_planes: tuple[Any, Any, Any],
    info: slab.SlabInfo,
    plan: projtile.Plan,
) -> tuple[Any, Any]:
    """projtile.tile_dual's contract as two stand_in_tile launches."""
    _check_launch(x8, info, plan, dual=True)
    return stand_in_tile(x8, *gate_planes, info, plan), stand_in_tile(x8, *up_planes, info, plan)


def guard_tile_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers restores for what the routing tests move: the flag, the window,
    the round flag, the scope, the plan table, and the projection kernel's and
    int8's counters (each count restored in place, since the modules read the
    dicts they hold)."""
    for name in ("_active", "_window", "_round_flag", "_scope", "PLANS"):
        monkeypatch.setattr(projtile, name, getattr(projtile, name))
    for counters in (projkernel.calls, int8prefill.calls):
        for key, count in counters.items():
            monkeypatch.setitem(counters, key, count)


@contextlib.contextmanager
def window() -> Iterator[projtile.Window]:
    """The tile's decode window open for the with-block, as decode() opens it, and
    closed on the way out unless the block closed it itself."""
    projtile.open_window()
    opened = projtile._window
    assert opened is not None
    try:
        yield opened
    finally:
        if projtile._window is opened:
            projtile.close_window()


def hooked_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    """pfx.hooked() (both hooks, the kernel's flag, its stand-ins), the tiny model's
    plans, the tile's flag and the stand-ins behind the tile's two seams, for this
    test only. Serving still needs a window: window()."""
    guard_tile_state(monkeypatch)
    pfx.hooked(monkeypatch)
    tiny_plans(monkeypatch)
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(projtile, "tile", stand_in_tile)
    monkeypatch.setattr(projtile, "tile_dual", stand_in_tile_dual)
