"""The projection tile: the drafter's verify projections on the M5 Pro's tensor
units, over slab-ordered weights (``sous.engine.slab``).

The weights take two forms. gate, up and down are held slab-only: their
``weight``, ``scales`` and ``biases`` are replaced by the flat planes, which mlx's
parameter tree then holds, and the module carries ``slab.SLAB_ATTR``. The
attention and GDN linears keep mlx's arrays for prefill and the one-row kernel
and gain a slab copy, one stack per group of linears the verifier calls
together, shared by every member as ``STACK_ATTR``. Both live on the modules and
go with them at unload: nothing at module level holds an array (no id-keyed
map, no cache over arrays), so nothing here can keep a model's weights alive.

mlx is imported inside each function.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from sous.engine import slab
from sous.engine.slab import SlabInfo

# --------------------------------------------------------------------------------
# The two weight forms: slab-only MLP linears and stacked copies
# --------------------------------------------------------------------------------

# Set with object.__setattr__, like the kernels' tags, so the stack stays out of
# mlx's parameter tree whatever it holds: its planes are a copy, not the module's
# parameters, and no tree walk may see or count them twice.
STACK_ATTR = "_sous_tile"

# (K, the members' real widths in call order): what picks a stack's plan.
Key = tuple[int, tuple[int, ...]]

_MLP_NAMES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True, eq=False)
class Stack:
    """One slab copy of a group of linears the verifier calls together: N
    concatenated in call order before padding, so the group pads once; ``slices``
    holds each member's (start, stop) in it. Compared by identity, since its planes
    have no truth value."""

    codes: Any
    scales: Any
    biases: Any
    info: SlabInfo
    slices: tuple[tuple[int, int], ...]
    key: Key


def stack_of(module: Any) -> Stack | None:
    return getattr(module, STACK_ATTR, None)


def _mlp_linears(root: Any) -> Iterator[Any]:
    """gate, up and down of every layer with a dense MLP; any other layout (a MoE
    block, a model without layers) yields nothing, so the fork key can read any
    model the engine loads."""
    for layer in getattr(getattr(root, "model", None), "layers", ()):
        mlp = getattr(layer, "mlp", None)
        for name in _MLP_NAMES:
            module = getattr(mlp, name, None)
            if module is not None:
                yield module


def _convert(module: Any) -> None:
    """One linear slab-only. The planes go through the module's own __setattr__, so
    mlx's parameter tree holds them and mlx's arrays are dropped here; repack has
    evaluated the planes, so no graph keeps the old arrays alive."""
    codes, scales, biases, info = slab.repack(module.weight, module.scales, module.biases)
    module.weight, module.scales, module.biases = codes, scales, biases
    object.__setattr__(module, slab.SLAB_ATTR, info)


def _revert(module: Any, info: SlabInfo) -> None:
    """One slab-only linear back to mlx's arrays, bitwise. The marker goes last, so
    a revert that raises leaves the module marked as what it still holds."""
    import mlx.core as mx

    w, scales, biases = slab.unpack(module.weight, module.scales, module.biases, info)
    mx.eval(w, scales, biases)
    module.weight, module.scales, module.biases = w, scales, biases
    object.__delattr__(module, slab.SLAB_ATTR)


def convert_mlp(root: Any) -> int:
    """Hold every layer's gate, up and down slab-only and return how many were
    converted; a module already marked is left alone. One linear at a time, so the
    peak is one linear's transient rather than the MLP's."""
    import mlx.core as mx

    converted = 0
    for module in _mlp_linears(root):
        if slab.slab_info(module) is None:
            _convert(module)
            converted += 1
    mx.clear_cache()
    return converted


def revert_mlp(root: Any) -> int:
    """convert_mlp's inverse: every marked MLP linear back to mlx's arrays,
    bitwise, its marker deleted; returns how many."""
    import mlx.core as mx

    reverted = 0
    for module in _mlp_linears(root):
        info = slab.slab_info(module)
        if info is not None:
            _revert(module, info)
            reverted += 1
    mx.clear_cache()
    return reverted


def stack_groups(root: Any) -> list[tuple[Any, ...]]:
    """The attention and GDN linears grouped as the verifier calls them, layer by
    layer: a GDN layer's in_proj_qkv+in_proj_z, in_proj_b+in_proj_a and out_proj;
    an attention layer's q+k+v and o_proj. The verifier calls the four GDN input
    projections at once, but as one stack they ran slower on the tile than on the
    projection kernel (1.06x its time, the 27B on the M5 Pro), and as two stacks
    faster."""
    groups: list[tuple[Any, ...]] = []
    for layer in root.model.layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            groups += [
                (gdn.in_proj_qkv, gdn.in_proj_z),
                (gdn.in_proj_b, gdn.in_proj_a),
                (gdn.out_proj,),
            ]
        else:
            attention = layer.self_attn
            groups += [
                (attention.q_proj, attention.k_proj, attention.v_proj),
                (attention.o_proj,),
            ]
    return groups


def _key(group: tuple[Any, ...]) -> Key:
    return group[0].scales.shape[1] * slab.GROUP, tuple(int(m.weight.shape[0]) for m in group)


def _stack_bytes(key: Key) -> int:
    """A stack's planes: half a byte per code plus a bf16 scale and bias per 64,
    9/16 of a byte per padded weight."""
    k, ns = key
    tiles = -(-sum(ns) // slab.TILE)
    return tiles * slab.TILE * k * 9 // 16


def build_stacks(root: Any, *, keys_with_plans: Callable[[Key], bool]) -> int:
    """A stack for every group whose key ``keys_with_plans`` accepts, set as one
    shared Stack on each member; returns the bytes built. One group at a time,
    evaluated by repack, so the peak is one group's transient."""
    import mlx.core as mx

    built = 0
    for group in stack_groups(root):
        key = _key(group)
        if not keys_with_plans(key):
            continue
        arrays = (
            mx.concatenate([getattr(m, name) for m in group], axis=0)
            for name in ("weight", "scales", "biases")
        )
        codes, scales, biases, info = slab.repack(*arrays)
        slices = tuple(itertools.pairwise(itertools.accumulate(key[1], initial=0)))
        stack = Stack(codes, scales, biases, info, slices, key)
        for member in group:
            object.__setattr__(member, STACK_ATTR, stack)
        built += codes.nbytes + scales.nbytes + biases.nbytes
    mx.clear_cache()
    return built


def drop_stacks(root: Any) -> None:
    """Take the stack off every member; its planes go with the last reference."""
    import mlx.core as mx

    for group in stack_groups(root):
        for member in group:
            if STACK_ATTR in vars(member):
                object.__delattr__(member, STACK_ATTR)
    mx.clear_cache()


def copy_bytes(root: Any, *, keys_with_plans: Callable[[Key], bool]) -> int:
    """What build_stacks would build with the same predicate, from the shapes
    alone, so the copy can be decided before anything is allocated."""
    keys = (_key(group) for group in stack_groups(root))
    return sum(_stack_bytes(key) for key in keys if keys_with_plans(key))


def fork_key_state(root: Any) -> str:
    """The fork key's ``ptile``: "active" while any MLP linear is slab-only, else
    "off". Any, not all: a revert that raised part-way leaves linears that still
    take the slab paths, and their 9-to-127-row calls int8's bits rather than
    stock's, so the model must key as converted. The stacks are not read: they
    serve only inside a decode, and no stored KV comes from one."""
    converted = any(slab.slab_info(m) is not None for m in _mlp_linears(root))
    return "active" if converted else "off"
