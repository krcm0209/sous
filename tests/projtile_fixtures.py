"""Shared helpers for the projection tile's tests: the tiny qwen3_5 model with its
MLP held slab-only (and, at scope "all", its attention and GDN linears stacked),
and a projection whose width pads.

mlx is imported inside each helper, so importing this module needs only what
`tests.proj_fixtures` itself imports."""

from typing import Any

from sous.engine import projtile, slab
from tests import proj_fixtures as pfx


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
