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

The kernel (``kernels/projection_tile.metal`` and ``.h``) is Splash's affine Q4
MPP tile (incoai/splash, Apache License 2.0; see THIRD_PARTY_NOTICES.md): per
quant group one ``matmul2d`` of eight bf16 rows against the group's 4-bit codes
on the tensor units, then sous's two-fma fp32 epilogue. Every launch computes
eight rows, a call of fewer padded with zero rows, so a row's bits never depend
on how many rows share the call. It runs only at the plans in ``PLANS``, tuned
on the M5 Pro; a key outside the table stays on the projection kernel.

It serves only inside a decode's window (``open_window``), reached through the
projection kernel's two hooks: the verifier's runs (``verify``) and a module's
calls of two to eight rows (``plain``), each when ``serves()`` and
``plan_for_call()`` say so; every other call keeps the projection kernel's path.

``enable()`` decides it per load, after int8 prefill and the projection kernel and
before the drafter's route, which VLMEngine settles only once the block has
resolved: guards that say whether the tile applies here, a compile of the
slab paths at the model's own shapes, the memory rule that picks the scope
(``copy_decision``: the MLP alone, or the attention and GDN copies too), the
conversion, a warm-up and a probe. A failure after the conversion reverts it.

mlx is imported inside each function.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import operator
import time
import traceback
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from sous.engine import gpucores, int8prefill, nax, projkernel, slab, tileattn
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text
from sous.engine.promptcache import CACHE_BUDGET_SLACK
from sous.engine.slab import SlabInfo

logger = logging.getLogger("sous.engine.projtile")

# --------------------------------------------------------------------------------
# The two weight forms: slab-only MLP linears and stacked copies
# --------------------------------------------------------------------------------

# Set with object.__setattr__, like the kernels' tags, so the stack stays out of
# mlx's parameter tree whatever it holds: its planes are a copy, not the module's
# parameters, and no tree walk may see or count them twice.
STACK_ATTR = "_sous_tile"
# Each member's position in its stack's slices, set beside STACK_ATTR the same way:
# a call is a stack's launch only when its linears are the members in order, and a
# member called alone finds its own columns by it.
INDEX_ATTR = "_sous_tile_index"

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
    # From the instance dict, for the reason slab.slab_info reads its marker there.
    return vars(module).get(STACK_ATTR)


def _mlp_trios(root: Any) -> Iterator[tuple[Any, ...]]:
    """gate, up and down of each layer with a dense MLP, one tuple per layer; any
    other layout (a MoE block, a model without layers) yields nothing, so the fork
    key can read any model the engine loads."""
    for layer in getattr(getattr(root, "model", None), "layers", ()):
        mlp = getattr(layer, "mlp", None)
        trio = tuple(
            module for name in _MLP_NAMES if (module := getattr(mlp, name, None)) is not None
        )
        if trio:
            yield trio


def _mlp_linears(root: Any) -> Iterator[Any]:
    """Every layer's gate, up and down, flat."""
    for trio in _mlp_trios(root):
        yield from trio


def _mlp_named(root: Any) -> Iterator[tuple[str, Any]]:
    """_mlp_linears() with each linear's path, as named_modules() spells it."""
    for index, layer in enumerate(getattr(getattr(root, "model", None), "layers", ())):
        mlp = getattr(layer, "mlp", None)
        for name in _MLP_NAMES:
            module = getattr(mlp, name, None)
            if module is not None:
                yield f"model.layers.{index}.mlp.{name}", module


# mlx's QuantizedLinear._extra_repr unpacks weight.shape as [N, K/8], which a
# flat plane is not, so repr() of a converted module, of anything holding one (its
# MLP, its layer, the model, a Launch) would raise for the life of the load. A
# converted module carries its own, set in its instance dict, which an attribute
# lookup reads before the class's method.
_REPR_ATTR = "_extra_repr"


def _slab_repr(module: Any) -> Callable[[], str]:
    """A converted module's _extra_repr: its slab form. It holds the module
    weakly: a bound method would close a cycle through the module's own dict,
    which only the cyclic collector breaks, keeping the planes past the last
    reference to the model."""
    ref = weakref.ref(module)

    def extra_repr() -> str:
        m = ref()
        info = None if m is None else slab.slab_info(m)
        if info is None:
            return "slab"
        return (
            f"slab n_real={info.n_real} k={info.k} tiles={info.tiles}, "
            f"bits={getattr(m, 'bits', None)}, group_size={getattr(m, 'group_size', None)}"
        )

    return extra_repr


def _convert(module: Any, path: str | None = None) -> None:
    """One linear slab-only. The planes go through the module's own __setattr__, so
    mlx's parameter tree holds them and mlx's arrays are dropped here; repack has
    evaluated the planes, so no graph keeps the old arrays alive. ``path`` names
    the module in the error a call no slab path serves raises."""
    codes, scales, biases, info = slab.repack(module.weight, module.scales, module.biases)
    module.weight, module.scales, module.biases = codes, scales, biases
    if path is not None:
        object.__setattr__(module, slab.PATH_ATTR, path)
    object.__setattr__(module, _REPR_ATTR, _slab_repr(module))
    object.__setattr__(module, slab.SLAB_ATTR, info)


def _unmark(module: Any) -> None:
    """Take off what _convert() set beside the planes: the marker, the path and
    the repr, so mlx's own repr reads mlx's arrays again."""
    object.__delattr__(module, slab.SLAB_ATTR)
    for name in (slab.PATH_ATTR, _REPR_ATTR):
        if name in vars(module):
            object.__delattr__(module, name)


def _revert_group(modules: Sequence[Any]) -> int:
    """Every slab-only linear of a layer's MLP back to mlx's arrays, bitwise, all or
    none: the unpacking and its evaluation come first, the assignments after, so a
    failure while unpacking leaves the whole group slab-only. A group that was left
    half reverted would fuse gate and up in two forms, which the projection kernel's
    fused call refuses on every drafted decode. Each module's marker goes after its
    own planes, so a module is never marked as what it no longer holds. Returns how
    many were reverted."""
    import mlx.core as mx

    marked = [(m, info) for m in modules if (info := slab.slab_info(m)) is not None]
    unpacked = [slab.unpack(m.weight, m.scales, m.biases, info) for m, info in marked]
    mx.eval(*(array for planes in unpacked for array in planes))
    for (module, _), (w, scales, biases) in zip(marked, unpacked, strict=True):
        module.weight, module.scales, module.biases = w, scales, biases
        _unmark(module)
    return len(marked)


def convert_mlp(root: Any) -> int:
    """Hold every layer's gate, up and down slab-only and return how many were
    converted; a module already marked is left alone. One linear at a time, so the
    peak is one linear's transient rather than the MLP's."""
    import mlx.core as mx

    converted = 0
    for path, module in _mlp_named(root):
        if slab.slab_info(module) is None:
            _convert(module, path)
            converted += 1
    mx.clear_cache()
    return converted


def revert_mlp(root: Any) -> int:
    """convert_mlp's inverse: every marked MLP linear back to mlx's arrays,
    bitwise, its marker deleted; returns how many. One layer at a time and a
    layer's three together (_revert_group), so an interrupted revert leaves whole
    layers in one form each."""
    import mlx.core as mx

    reverted = 0
    for trio in _mlp_trios(root):
        reverted += _revert_group(trio)
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
    shared Stack on each member beside the member's index in it; returns the bytes
    built. One group at a time, evaluated by repack, so the peak is one group's
    transient."""
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
        for index, member in enumerate(group):
            object.__setattr__(member, STACK_ATTR, stack)
            object.__setattr__(member, INDEX_ATTR, index)
        built += codes.nbytes + scales.nbytes + biases.nbytes
    mx.clear_cache()
    return built


def drop_stacks(root: Any) -> None:
    """Take the stack and the index off every member; the planes go with the last
    reference."""
    import mlx.core as mx

    for group in stack_groups(root):
        for member in group:
            for name in (STACK_ATTR, INDEX_ATTR):
                if name in vars(member):
                    object.__delattr__(member, name)
    mx.clear_cache()


def copy_bytes(root: Any, *, keys_with_plans: Callable[[Key], bool]) -> int:
    """What build_stacks would build with the same predicate, from the shapes
    alone, so the copy can be decided before anything is allocated."""
    keys = (_key(group) for group in stack_groups(root))
    return sum(_stack_bytes(key) for key in keys if keys_with_plans(key))


def fork_key_state(root: Any) -> str:
    """The fork key's ``ptile``: "active" when every MLP linear is slab-only, "off"
    when none is, and "partial:<slab-only>/<MLP linears>:<digest>" otherwise. A
    revert that raised part-way leaves some layers slab-only, whose 9-to-127-row
    prefill calls carry int8's bits, beside layers back on stock: KV that matches
    neither a converted load's nor an unconverted one's, so it keys apart from
    both. Which layers matter, not only how many (a failed conversion leaves a
    prefix slab-only, a failed revert a suffix), so the digest is the first 8 hex
    of sha256 over the slab-only linears' sorted paths. The stacks are not read:
    they serve only inside a decode, and no stored KV comes from one."""
    named = list(_mlp_named(root))
    converted = sorted(path for path, m in named if slab.slab_info(m) is not None)
    if not converted:
        return "off"
    if len(converted) == len(named):
        return "active"
    digest = hashlib.sha256("\n".join(converted).encode()).hexdigest()[:8]
    return f"partial:{len(converted)}/{len(named)}:{digest}"


# --------------------------------------------------------------------------------
# The kernel and its plans
# --------------------------------------------------------------------------------

# Rows one launch computes: the tile is compiled at M = 8 alone, and a call of fewer
# rows is padded with zero rows (pad8), so T is never an argument.
MAX_ROWS = 8
# The tile against an fp32 reference: its relative RMS within this factor of the
# floor, that reference rounded to bf16. On every linear of the 27B the tile's
# relative RMS against that elementwise fp32 reference equalled the floor to four
# significant digits (M5 Pro).
TILE_FLOOR_FACTOR = 1.05
# GPU core counts the plans were tuned on: the M5 Pro's 20. A plan's GROUPS and
# split count are chosen for one wave of threadgroups on that chip.
MEASURED_CORES = frozenset({20})

# The row sums are taken one 256-input block (four quant groups) at a time, and a
# split covers whole blocks.
_BLOCK = 256
_TILE_WIDTHS = (128, 256)
_SIMDGROUPS = (4, 8)
# mlx's default, stated rather than inherited: the kernel's bits are its two fmas
# per group and the reduce's ordered adds, which relaxed and fast math may contract
# or reorder. It runs at fast-math speed.
_SAFE_MATH = {"math_mode": "safe"}
# projection_tile.metal holds the tile body, this line, then the reduce body.
_REDUCE_MARK = "// REDUCE\n"
_REDUCE_THREADS = 256


@dataclass(frozen=True)
class Plan:
    """How one launch covers its key: TN columns per tile, S simdgroups, GROUPS
    persistent threadgroups, PIPE (two groups' matmuls issued before their
    epilogues) and SPLITS (K cut into that many ranges whose fp32 partials a
    fixed-order reduce adds; frozen per key, since another count adds in another
    order)."""

    tn: int
    s: int
    groups: int
    pipe: bool
    splits: int = 1

    @property
    def label(self) -> str:
        split = f" x{self.splits}" if self.splits > 1 else ""
        return f"TN{self.tn}/S{self.s}/G{self.groups}/{'pipe' if self.pipe else 'seq'}{split}"


# The 27B's dispatch keys on the M5 Pro. The six fused keys the verifier calls carry
# the plans tuned for them (1,440 candidates, 0.835x the projection kernel's
# forward); the alone forms, which decode's two-to-eight-row calls reach through the
# module hook, carry the plans chosen for each linear alone. One GROUPS step off a
# tuned plan cost 15-20 %, so a key outside the table gets no default plan: the
# projection kernel serves it.
PLANS: dict[Key, Plan] = {
    (5120, (12288, 1024, 1024)): Plan(128, 8, 80, True),  # q+k+v
    (6144, (5120,)): Plan(256, 8, 20, True, 2),  # o_proj, out_proj
    (5120, (10240, 6144)): Plan(128, 8, 80, True),  # in_proj_qkv+in_proj_z
    (5120, (48, 48)): Plan(128, 8, 2, True, 8),  # in_proj_b+in_proj_a
    (5120, (17408, 17408)): Plan(128, 4, 100, True),  # gate+up, one dual launch
    (17408, (5120,)): Plan(256, 8, 20, True, 2),  # down
    (5120, (12288,)): Plan(128, 8, 80, True),  # q
    (5120, (1024,)): Plan(128, 8, 8, True),  # k, v
    (5120, (10240,)): Plan(128, 4, 80, True),  # in_proj_qkv
    (5120, (6144,)): Plan(128, 8, 48, True),  # in_proj_z
    # in_proj_b, in_proj_a: alone, each runs the b+a stack's one tile.
    (5120, (48,)): Plan(128, 8, 2, True, 8),
    (5120, (17408,)): Plan(256, 4, 68, True),  # gate, up
}


def _shape(plan: Plan) -> tuple[int, int, int, bool, int]:
    return plan.tn, plan.s, plan.groups, plan.pipe, plan.splits


# What availability() compiles: every distinct plan of the table on one launch, and
# the dual at gate+up's, the one key a dual launch serves. Read from the table at
# import, so a test that swaps PLANS changes neither.
DISTINCT_SHAPES = sorted({_shape(p) for p in PLANS.values()})
_DUAL_SHAPE = _shape(PLANS[(5120, (17408, 17408))])


def plan_for(key: Key) -> Plan | None:
    """The tuned plan for a dispatch key, None for a key the table lacks."""
    return PLANS.get(key)


# One launch of a call: its key, and the linears it serves in call order.
_Launch = tuple[Key, tuple[Any, ...]]


def _launches(linears: Sequence[Any]) -> list[_Launch] | None:
    """The launches a call of ``linears`` takes, in call order, each as its key and
    the linears it serves: a lone linear keyed by its own K and width (the alone
    form, whether it is slab-only or a stack member); gate+up (two slab-only
    linears of one shape) as one dual launch; and, for a call of stacked linears,
    one launch per run of them that is a stack's members whole and in order (the
    GDN call makes two, qkv+z and b+a). None when a linear has no planes, the
    forms mix, or a run is not its stack's members in order: widths alone would
    take q+v+k for q+k+v, k and v being one width."""
    if not linears:
        return None
    infos = [slab.slab_info(m) for m in linears]
    stacks = [stack_of(m) for m in linears]
    if any(info is None and stack is None for info, stack in zip(infos, stacks, strict=True)):
        return None
    if len(linears) == 1:
        (module,) = linears
        if infos[0] is None and vars(module).get(INDEX_ATTR) is None:
            return None
        key = (projkernel._in_features(module), (projkernel._out_features(module),))
        return [(key, (module,))]
    if all(info is not None for info in infos):
        first = infos[0]
        if len(infos) == 2 and first is not None and infos[1] == first:
            return [((first.k, (first.n_real, first.n_real)), tuple(linears))]
        return None
    if any(info is not None for info in infos):
        return None
    launches: list[_Launch] = []
    start = 0
    while start < len(linears):
        stack = stacks[start]
        stop = start + 1
        while stop < len(linears) and stacks[stop] is stack:
            stop += 1
        run = tuple(linears[start:stop])
        indices = [vars(m).get(INDEX_ATTR) for m in run]
        if stack is None or indices != list(range(len(stack.slices))):
            return None
        launches.append((stack.key, run))
        start = stop
    return launches


class Launch(NamedTuple):
    """One launch of a call: its key, the linears it serves in call order, and the
    plan it runs at."""

    key: Key
    members: tuple[Any, ...]
    plan: Plan


def plan_for_call(linears: Sequence[Any]) -> tuple[Launch, ...] | None:
    """The launches a verifier or module call of ``linears`` takes, in call order,
    each with its plan, or None when _launches refuses the call or any launch's
    key has no plan: the whole call then stays on the projection kernel, never
    split between the two. verify() and plain() launch from what this returns, so
    a call's launches are worked out once per call, not again per run."""
    launches = _launches(linears)
    if launches is None:
        return None
    out: list[Launch] = []
    for key, members in launches:
        plan = PLANS.get(key)
        if plan is None:
            return None
        out.append(Launch(key, members, plan))
    return tuple(out)


def check_plan(info: SlabInfo, plan: Plan, *, dual: bool = False) -> str | None:
    """Why the kernel cannot run ``plan`` over planes holding ``info``, None when it
    can: the row sums read whole 256-input blocks, the tiles must cover the padded
    width exactly, the kernel is tuned at TN 128/256 and S 4/8, a split covers at
    least one block, and the dual launch has no split form."""
    if info.k % _BLOCK:
        return f"K={info.k} is not a multiple of {_BLOCK}"
    if info.n_pad % plan.tn:
        return f"N={info.n_pad} is not a multiple of TN={plan.tn}"
    if plan.tn not in _TILE_WIDTHS:
        return f"TN={plan.tn} is not 128 or 256"
    if plan.s not in _SIMDGROUPS:
        return f"S={plan.s} is not 4 or 8"
    if plan.groups < 1:
        return f"GROUPS={plan.groups} is below 1"
    if not 1 <= plan.splits <= info.k // _BLOCK:
        return f"SPLITS={plan.splits} is outside 1..{info.k // _BLOCK}"
    if dual and plan.splits != 1:
        return f"a dual launch takes SPLITS=1, got {plan.splits}"
    return None


def MLP_KEYS(root: Any) -> list[Key]:
    """The keys a model's MLP calls reach on the tile, for every distinct MLP shape:
    gate+up's dual, gate and up alone, and down. Read from either form, so it holds
    before anything is converted."""
    keys: list[Key] = []
    for layer in getattr(getattr(root, "model", None), "layers", ()):
        mlp = getattr(layer, "mlp", None)
        linears = [getattr(mlp, name, None) for name in _MLP_NAMES]
        if any(m is None for m in linears):
            continue
        (k_gate, n_gate), (k_up, n_up), (k_down, n_down) = (
            (projkernel._in_features(m), projkernel._out_features(m)) for m in linears
        )
        for key in (
            (k_gate, (n_gate, n_up)),
            (k_gate, (n_gate,)),
            (k_up, (n_up,)),
            (k_down, (n_down,)),
        ):
            if key not in keys:
                keys.append(key)
    return keys


@functools.cache
def _sources() -> tuple[str, str, str, str]:
    """common.h, nax.h with the tile's helpers, and the tile and reduce bodies,
    read once: per-launch text work slows the decode hot path."""
    tile_body, reduce_body = _kernel_text("projection_tile.metal").split(_REDUCE_MARK)
    nax_tile = _kernel_text("nax.h") + _kernel_text("projection_tile.h")
    return _kernel_text("common.h"), nax_tile, tile_body, reduce_body


@functools.cache
def _tile_spec(dual: bool, splitout: bool) -> tuple[str, list[str], list[str], str, str]:
    """One kernel object per (DUAL, SPLITOUT), each switch a header define: the
    preprocessor cannot see a template argument."""
    common, nax_tile, body, _ = _sources()
    defines = f"#define DUAL {int(dual)}\n#define SPLITOUT {int(splitout)}\n"
    inputs = ["x", "w0", "sc0", "bi0"] + (["w1", "sc1", "bi1"] if dual else [])
    outputs = ["y0", "y1"] if dual else ["y0"]
    name = f"sous_proj_tile_d{int(dual)}_s{int(splitout)}"
    return name, inputs, outputs, body, common + defines + nax_tile


@functools.cache
def _launch_args(info: SlabInfo, plan: Plan, dual: bool, splitout: bool) -> dict[str, Any]:
    """What a (SlabInfo, plan) fixes about a tile launch, checked and built once:
    a decoded token launches every projection of the model."""
    import mlx.core as mx

    reason = check_plan(info, plan, dual=dual)
    if reason is not None:
        raise ValueError(f"projection tile: {reason}")
    n = info.n_pad
    if splitout:
        shapes, dtypes = [(plan.splits, MAX_ROWS, n)], [mx.float32]
    else:
        shapes, dtypes = [(MAX_ROWS, n)], [mx.bfloat16]
    template = [
        ("N", n),
        ("K", info.k),
        ("TN", plan.tn),
        ("S", plan.s),
        ("GROUPS", plan.groups),
        ("PIPE", plan.pipe),
        ("SPLITS", plan.splits),
    ]
    return {
        "template": template,
        "grid": (plan.groups * plan.s * 32, plan.splits, 1),
        "threadgroup": (plan.s * 32, 1, 1),
        "output_shapes": shapes * 2 if dual else shapes,
        "output_dtypes": dtypes * 2 if dual else dtypes,
    }


@functools.cache
def _reduce_args(splits: int, n: int) -> dict[str, Any]:
    import mlx.core as mx

    total = MAX_ROWS * n
    return {
        "template": [("TOTAL", total), ("SPLITS", splits)],
        "grid": (-(-total // _REDUCE_THREADS) * _REDUCE_THREADS, 1, 1),
        "threadgroup": (_REDUCE_THREADS, 1, 1),
        "output_shapes": [(MAX_ROWS, n)],
        "output_dtypes": [mx.bfloat16],
    }


def _check(x8: Any, planes: Sequence[Any], info: SlabInfo) -> None:
    """Refuse rows or planes the kernel would misread: it reads eight rows of K and
    every plane at the addresses ``info`` gives."""
    import mlx.core as mx

    if tuple(x8.shape) != (MAX_ROWS, info.k) or x8.dtype != mx.bfloat16:
        raise ValueError(
            f"projection tile takes [{MAX_ROWS}, {info.k}] bf16 rows, "
            f"got {tuple(x8.shape)} {x8.dtype}"
        )
    codes, scales, biases = planes
    if (
        codes.dtype != mx.uint32
        or codes.size != info.code_words
        or any(p.dtype != mx.bfloat16 or p.size != info.units for p in (scales, biases))
    ):
        raise ValueError(f"projection tile: the planes do not hold {info}")


def run(x8: Any, codes: Any, scales: Any, biases: Any, info: SlabInfo, plan: Plan) -> Any:
    """x8 [8, K] bf16 through one slab-ordered projection or stack at ``plan``:
    [8, n_pad] bf16, the columns past n_real computing exactly 0 for finite
    activations. A split plan runs the split kernel, then the reduce that adds its
    partials in order."""
    if plan.splits > 1:
        return _reduce(_partials(x8, codes, scales, biases, info, plan))
    _check(x8, (codes, scales, biases), info)
    (y,) = _kernel(*_tile_spec(False, False), compile_options=_SAFE_MATH)(
        inputs=[x8, codes, scales, biases], **_launch_args(info, plan, False, False)
    )
    return y


def _partials(x8: Any, codes: Any, scales: Any, biases: Any, info: SlabInfo, plan: Plan) -> Any:
    """A split plan's first pass: [SPLITS, 8, n_pad] fp32, each split's unrounded
    sum over its range of quant groups."""
    _check(x8, (codes, scales, biases), info)
    (part,) = _kernel(*_tile_spec(False, True), compile_options=_SAFE_MATH)(
        inputs=[x8, codes, scales, biases], **_launch_args(info, plan, False, True)
    )
    return part


def _reduce(part: Any) -> Any:
    """[SPLITS, 8, N] fp32 partials added in split order from 0, rounded to bf16
    once: [8, N]."""
    splits, _, n = part.shape
    common, _, _, body = _sources()
    (y,) = _kernel(
        "sous_proj_tile_reduce", ["part"], ["y"], body, common, compile_options=_SAFE_MATH
    )(inputs=[part], **_reduce_args(splits, n))
    return y


def run_dual(
    x8: Any,
    gate_planes: Sequence[Any],
    up_planes: Sequence[Any],
    info: SlabInfo,
    plan: Plan,
) -> tuple[Any, Any]:
    """gate and up, two slab-ordered projections of one shape, in one launch that
    shares every activation slice and row sum: two [8, n_pad] bf16, each bitwise
    the single launch's at the same plan. A dual plan has no split."""
    _check(x8, gate_planes, info)
    _check(x8, up_planes, info)
    y0, y1 = _kernel(*_tile_spec(True, False), compile_options=_SAFE_MATH)(
        inputs=[x8, *gate_planes, *up_planes], **_launch_args(info, plan, True, False)
    )
    return y0, y1


def pad8(x: Any) -> Any:
    """[T, K] rows, T = 1..8, as the [8, K] a launch reads: zero rows appended,
    which make their row sums 0 and keep anything non-finite out of the tensor op;
    at T = 8 the rows themselves, uncopied."""
    import mlx.core as mx

    if x.ndim != 2 or not 1 <= x.shape[0] <= MAX_ROWS:
        raise ValueError(f"pad8 takes [1..{MAX_ROWS}, K] rows, got {tuple(x.shape)}")
    t, k = x.shape
    if t == MAX_ROWS:
        return x
    return mx.concatenate([x, mx.zeros((MAX_ROWS - t, k), dtype=x.dtype)], axis=0)


# The seams everything above the launches calls through (verify, plain, the warm-up
# and the probe), looked up at call time so tests can swap in plain-mlx stand-ins
# with the same contracts.
tile = run
tile_dual = run_dual


# --------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------

# availability()'s synthetic projection: K = 2048 takes every split count in the
# table (up to K / 256 = 8), and N = 512 is a whole number of tiles at either TN.
_PROBE_K = 2048
_PROBE_N = 512

_compile_ok = False


def _synthetic(n: int, k: int, key: Any) -> tuple[Any, Any, Any, SlabInfo]:
    """A random affine-Q4 projection as slab planes, built on the GPU and evaluated."""
    import mlx.core as mx

    dense = (mx.random.normal((n, k), key=key) * 0.02).astype(mx.bfloat16)
    return slab.repack(*mx.quantize(dense, group_size=slab.GROUP, bits=4))


def _compile_probe() -> str | None:
    """Compile and run every kernel a converted load serves from slab planes: the
    tile at every distinct plan of the table (the split plans reach the reduce
    kernel) and the dual at gate+up's, then the projection kernel's slab bodies, its
    one-row slab body and the int8 GEMM's slab read. None when all of them run,
    else the compiler's error line, the full report going to the log. They compile
    at runtime against the OS's own Metal toolchain, so a new macOS can reject a
    source that compiled before. The inputs are built with GPU ops and evaluated
    first: a compile failure with a CPU-stream op in flight deadlocks inside mlx's
    exception path instead of raising. Only success is remembered, so one bad
    moment cannot switch the tile off for a long-lived daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        x_key, gate_key, up_key = mx.random.split(mx.random.key(0), 3)
        x = mx.random.normal((int8prefill.SLAB_MIN_ROWS, _PROBE_K), key=x_key)
        x = x.astype(mx.bfloat16)
        x8 = mx.contiguous(x[:MAX_ROWS])
        codes, scales, biases, info = _synthetic(_PROBE_N, _PROBE_K, gate_key)
        up_codes, up_scales, up_biases, _ = _synthetic(_PROBE_N, _PROBE_K, up_key)
        mx.eval(x, x8)
        gate, up = (codes, scales, biases), (up_codes, up_scales, up_biases)
        outs = [run(x8, *gate, info, Plan(*shape)) for shape in DISTINCT_SHAPES]
        outs += run_dual(x8, gate, up, info, Plan(*_DUAL_SHAPE))
        weights = [projkernel.Weights(codes, scales, biases, info)]
        for kind in projkernel.VARIANTS:
            outs += projkernel._launch(kind, x8, weights)
        outs.append(projkernel._launch_row(x8[:1], codes, scales, biases, slab=info))
        outs.append(int8prefill.qmm(*int8prefill.stage_a(x), *gate, slab=info))
        mx.eval(outs)
    except Exception as e:  # noqa: BLE001 — any failure means the kernels cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("projection tile: the kernels failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares with int8 prefill and the attention
    tile (`nax.platform_reason()`), then a compile and run of every kernel a
    converted load serves. The `tile_nax` tests skip on this, never on another
    kernel's: each can fail to compile on a toolchain the others pass."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the projection tile kernels failed: {failed}")
    return nax.Availability(True)


# --------------------------------------------------------------------------------
# Routing: the flag, the decode window, the round flag and the launches
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One decode's window: the tile serves only while one is open."""

    serial: int


# 1 while the tile serves: set only by an enable() that reached active, cleared by
# every enable() first and by clear(). A plain int, as every flag here is, so
# nothing it holds outlives an unload.
_active = 0
# What enable() made servable: "all" (the MLP slab-only, the attention and GDN
# linears stacked) or "mlp"; None while the tile is not active.
_scope: str | None = None
# The open window, or None. One module global suffices: the engine's generation
# lock serialises every forward, and decode() opens and closes it on the caller's
# session thread.
_window: Window | None = None
# The newest window's serial, never reset, so no two windows of a process share one.
_serial = 0
# A harness's per-round lever, read by serves() and never by the window: at 0 an
# open window's calls stay on the projection kernel. Every open_window() sets it
# to 1, so production, which never touches it, serves every round of every decode.
_round_flag = 0


def open_window() -> None:
    """Open one decode's window and set the round flag: until close_window(),
    serves() answers for the tile. A window still open is replaced, not refused:
    whatever left it open, this decode's window is the one that should serve."""
    global _window, _serial, _round_flag
    _serial += 1
    _window = Window(serial=_serial)
    _round_flag = 1


def close_window() -> None:
    """Close the window, open or not. Never raises: decode() calls it first in its
    finally, where a raise would mask the error in flight and skip kv8's close."""
    global _window
    _window = None


def set_round_flag(value: int) -> None:
    """0 keeps the calls of the rounds that follow on the projection kernel inside
    an open window; anything else serves them on the tile again."""
    global _round_flag
    _round_flag = 1 if value else 0


def serves(rows: int, *, hook: int) -> bool:
    """Whether a call of ``rows`` rows runs on the tile: only while it is active, a
    window is open and the round flag is set. Hook 1 (a verifier run, at most
    MAX_ROWS rows after the cut) at any rows, a one-row remainder included, so a
    verify never mixes paths; hook 2 (a module call) at 2..MAX_ROWS, one row being
    the one-row kernel's. The projection kernel's seams look this up through the
    module on every call, so a harness may wrap it."""
    return bool(
        _active and _window is not None and _round_flag and (hook == 1 or 2 <= rows <= MAX_ROWS)
    )


def clear() -> None:
    """Stop serving (VLMEngine.unload()): the flag, the window, the round flag and
    the scope go. The planes live on the modules and go with the model."""
    global _active, _window, _round_flag, _scope
    _active, _window, _round_flag, _scope = 0, None, 0, None


def _padded(x: Any) -> tuple[Any, int]:
    """x [..., K] as the [8, K] a launch reads, and its rows: the product of the
    leading dimensions."""
    k = x.shape[-1]
    rows = x.size // k
    return pad8(x.reshape(rows, k)), rows


def _cut(y8: Any, rows: int, x: Any, start: int, stop: int) -> Any:
    """Columns [start, stop) of the call's rows of a launch's [8, n_pad] output, in
    x's leading shape. A cut of the whole output adds no slice, and 2-D rows no
    reshape: each graph node costs host time on every run of every round."""
    whole = rows == MAX_ROWS and start == 0 and stop == y8.shape[1]
    y = y8 if whole else y8[:rows, start:stop]
    return y if x.ndim == 2 else y.reshape(*x.shape[:-1], stop - start)


def _alone(module: Any, x8: Any, rows: int, x: Any, plan: Plan) -> Any:
    """One linear's own launch at its alone plan. A slab-only linear reads its
    planes. A stack member whose columns start and end on tile edges reads only
    its own tiles, a contiguous range of every plane ([T][G][256] units), so a
    member called alone costs its own bytes rather than its stack's; one sharing
    a tile with another member (in_proj_b and in_proj_a share b+a's one tile)
    runs its whole stack and takes its columns."""
    info = slab.slab_info(module)
    if info is not None:
        y = tile(x8, module.weight, module.scales, module.biases, info, plan)
        return _cut(y, rows, x, 0, info.n_real)
    stack = stack_of(module)
    if stack is None:
        raise ValueError("projection tile: a linear without planes")
    start, stop = stack.slices[vars(module)[INDEX_ATTR]]
    whole = stack.info
    if start % slab.TILE or (stop % slab.TILE and stop != whole.n_real):
        y = tile(x8, stack.codes, stack.scales, stack.biases, whole, plan)
        return _cut(y, rows, x, start, stop)
    t0, t1 = start // slab.TILE, -(-stop // slab.TILE)
    planes = (stack.codes, stack.scales, stack.biases)
    if (t0, t1) != (0, whole.tiles):
        # Per tile: the planes are [T][G][256] units, whole tiles end to end.
        units = whole.units // whole.tiles
        words = whole.code_words // whole.tiles
        planes = (
            stack.codes[t0 * words : t1 * words],
            stack.scales[t0 * units : t1 * units],
            stack.biases[t0 * units : t1 * units],
        )
    own = SlabInfo(n_real=stop - start, k=whole.k, tiles=t1 - t0)
    return _cut(tile(x8, *planes, own, plan), rows, x, 0, stop - start)


def _check_call(linears: Sequence[Any], launches: Sequence[Launch]) -> None:
    """Refuse launches that are not a call of exactly ``linears``, in order, by
    identity: the hook hands each call its own plan_for_call(), so a mismatch is a
    caller's bug, never a fallback."""
    members = [member for launch in launches for member in launch.members]
    if len(members) != len(linears) or not all(map(operator.is_, members, linears, strict=True)):
        raise ValueError("projection tile: the launches are not plan_for_call's for this call")


def verify(linears: Sequence[Any], x: Any, launches: Sequence[Launch]) -> tuple[Any, ...]:
    """One verifier run on the tile: x [..., K] bf16 of 1..MAX_ROWS rows (the
    product of its leading dimensions) through ``linears`` as ``launches``,
    plan_for_call's for them; one launch per stack, dual or lone linear, and each
    linear's columns cut back to [..., n_real], in call order. Raises ValueError
    when ``launches`` are not that call's."""
    _check_call(linears, launches)
    x8, rows = _padded(x)
    outs: list[Any] = []
    for _, members, plan in launches:
        if len(members) == 1:
            outs.append(_alone(members[0], x8, rows, x, plan))
            continue
        info = slab.slab_info(members[0])
        if info is not None:
            gate, up = members
            ys = tile_dual(
                x8,
                (gate.weight, gate.scales, gate.biases),
                (up.weight, up.scales, up.biases),
                info,
                plan,
            )
            outs += [_cut(y, rows, x, 0, info.n_real) for y in ys]
            continue
        stack = stack_of(members[0])
        if stack is None:
            raise ValueError("projection tile: a linear without planes")
        y = tile(x8, stack.codes, stack.scales, stack.biases, stack.info, plan)
        outs += [_cut(y, rows, x, start, stop) for start, stop in stack.slices]
    return tuple(outs)


def plain(module: Any, x: Any, launches: Sequence[Launch]) -> Any:
    """One module call on the tile: x [..., K] bf16 of 2..MAX_ROWS rows through
    ``module`` as ``launches``, plan_for_call's for the module alone; [...,
    n_real]. A stack member launches only its own tiles where it can (_alone).
    Raises ValueError when ``launches`` are not that call's, as verify() does."""
    _check_call((module,), launches)
    x8, rows = _padded(x)
    return _alone(module, x8, rows, x, launches[0].plan)


# --------------------------------------------------------------------------------
# The guards, the memory rule, the warm-up, the probe and enable()
# --------------------------------------------------------------------------------

# The prompt-cache budget, in GiB, the automatic rule must still leave beside the
# attention and GDN copies, else the tile serves the MLP alone: the budget
# docs/configuration.md gives a fork kept beside a retaining conversation. A copy
# that evicts the forks costs more prefill than it saves decode.
TILE_COPY_MIN_BUDGET_GIB = 14
_GIB = 1 << 30
# The probe's through-the-hooks rows and the row it plants a NaN in, as the
# projection kernel's probe takes them.
_PROBE_ROWS = 5
_NAN_ROW = 3
# The rows the probe holds int8's slab GEMM to its mlx-layout read at: its floor
# for a slab-only module, a middle count, and where it takes every module's calls.
_INT8_ROWS = (int8prefill.SLAB_MIN_ROWS, 64, int8prefill.MIN_ROWS)
# The width of the probe's cancelling linears: one slab tile, and N % 64 == 0 as
# int8's GEMM needs.
_CANCEL_N = slab.TILE


def _status(
    state: str,
    *,
    scope: str | None = None,
    reason: str | None = None,
    copy_gb: float | None = None,
    probe_seconds: float | None = None,
    budget_left_gib: float | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "scope": scope,
        "reason": reason,
        "copy_gb": copy_gb,
        "probe_seconds": probe_seconds,
        "budget_left_gib": budget_left_gib,
    }


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use the tile: report why. An `expected` refusal says only
    that the tile does not apply here (the projection kernel, the attention tile,
    the drafter, int8 prefill, the GPU, the model), which is how most Macs load a
    default-on tile: one INFO line, and the reason in the status. Any other
    refusal means it should have run and did not (pin drift, a kernel this OS
    rejects, a failed probe): one warning, since a switch that is silently inert
    there hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("projection tile unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: projection tile unavailable ({reason}); verify runs the projection kernel",
            stacklevel=3,
        )
    return _status("unavailable", reason=reason, probe_seconds=probe_seconds)


def _kernel_reason(kernel_status: dict[str, Any] | None) -> str | None:
    """The tile rides on the projection kernel: its hooks are the only way in, and
    its paths serve everything the tile does not."""
    state = (kernel_status or {}).get("state")
    if state == "active":
        return None
    if state == "off":
        return "projection kernel off"
    return (kernel_status or {}).get("reason") or f"projection kernel {state or 'unknown'}"


def _drafter_reason(drafter_kind: str | None) -> str | None:
    """The tile serves the DFlash round's verify: without a drafter a decode has no
    verify run, and its two-to-eight-row calls alone do not pay for the load."""
    if drafter_kind is None:
        return "no drafter"
    return projkernel._drafter_kind_reason(drafter_kind)


def _int8_reason(int8_status: dict[str, Any] | None) -> str | None:
    """A slab-only MLP linear's calls of nine rows and more can go nowhere but
    int8's slab GEMM: stock reads mlx's layout only."""
    state = (int8_status or {}).get("state")
    return None if state == "active" else f"int8 prefill {state or 'unknown'}"


def _cores_reason() -> str | None:
    """The plans are tuned for one wave of threadgroups on the measured core
    counts; another GPU would run them, but at a speed nobody measured."""
    import mlx.core as mx

    cores, reason = gpucores.matched_cores(str(mx.device_info().get("device_name", "")))
    if cores is None:
        return reason
    if cores not in MEASURED_CORES:
        measured = ", ".join(str(c) for c in sorted(MEASURED_CORES))
        return f"GPU core count {cores} not measured (measured: {measured})"
    return None


def _has_plan(key: Key) -> bool:
    return plan_for(key) is not None


def _no_plan(key: Key) -> str:
    """Shared by enable() and static_reason(): one text for both."""
    return f"no tuned plan for {key}"


def _path(root: Any, module: Any) -> str:
    return next((p for p, m in root.named_modules() if m is module), "?")


def _model_reason(root: Any) -> str | None:
    """Why this model is not one the tile takes, or None. Every MLP linear is
    converted whatever the scope, so each must be eligible (affine 4-bit gs64,
    bf16 scales and biases, mlx's layout still) and routed by both of the paths
    that will serve it: the projection kernel's hooks up to MAX_ROWS rows and
    int8's wrappers from nine; stock would raise on its flat planes. Every MLP key
    needs a tuned plan the kernel can run. A copied linear needs only to be
    eligible: a group without a plan is not copied, and the projection kernel
    keeps it."""
    mlp = list(_mlp_linears(root))
    if not mlp:
        return "no dense MLP"
    copied = [m for group in stack_groups(root) if _has_plan(_key(group)) for m in group]
    for module in mlp + copied:
        if not projkernel.eligible(module):
            return (
                f"projection {_path(root, module)} is not affine 4-bit gs64 with bf16 "
                "scales in mlx's layout"
            )
        if not projkernel._tagged(module):
            return f"projection {_path(root, module)} is not routed by the projection kernel"
    for module in mlp:
        if not int8prefill._tagged(module):
            return f"projection {_path(root, module)} is not routed by int8 prefill"
    for key in MLP_KEYS(root):
        plan = plan_for(key)
        if plan is None:
            return _no_plan(key)
        k, ns = key
        dual = len(ns) == 2
        n = ns[0] if dual else sum(ns)
        reason = check_plan(SlabInfo(n_real=n, k=k, tiles=-(-n // slab.TILE)), plan, dual=dual)
        if reason is not None:
            return f"the plan for {key}: {reason}"
    return None


def _pins_reason() -> str | None:
    """Which pinned dependency drifted, or None. The tile is reached only through
    the projection kernel's hooks, and its slab paths only through the kernel's
    and int8's own seams, so the kernel's pins (mlx itself, and every mlx-vlm
    function that decides a projection reaches a hook) are the tile's."""
    return projkernel._pins_reason()


def budget_left(headroom_bytes: int | None, copy_bytes: int, reserve_bytes: int) -> int | None:
    """The prompt-cache budget the automatic rule (promptcache.auto_cache_budget)
    leaves once the copy is made: headroom, read before the copy with the drafter
    already loaded, minus the copy, the turn's own reserve and the slack. None
    when the headroom could not be read."""
    if headroom_bytes is None:
        return None
    return headroom_bytes - copy_bytes - reserve_bytes - CACHE_BUDGET_SLACK


def copy_decision(
    config_value: bool | str,
    cache_budget: int | None,
    headroom_bytes: int | None,
    copy_bytes: int,
    reserve_bytes: int,
    *,
    cache_enabled: bool,
) -> tuple[bool, str]:
    """Whether to make the attention and GDN copies, and why, all in bytes: the
    MLP's conversion is size-neutral, the copies are not. "mlp" never copies,
    "all" always does; true copies only while the prompt cache keeps a budget
    worth having beside them: TILE_COPY_MIN_BUDGET_GIB under the automatic
    budget, the configured budget itself under an explicit one. With the prompt
    cache off (``cache_enabled`` false) no slot or fork holds any budget, so true
    copies whenever the copy fits beside the reserve and the slack, as under a
    configured budget of 0. Never a warning: a skipped copy still serves the MLP,
    and a configured budget is the user's choice. Shared with sous tune's fit(),
    which names the scope a load would take in its detail."""
    left = budget_left(headroom_bytes, copy_bytes, reserve_bytes)
    # Two decimals: at one, 13.96 GiB would read "14.0 GiB (< 14)".
    after = "unknown" if left is None else f"{left / _GIB:.2f} GiB"
    if config_value == "mlp":
        return False, "scope mlp by config"
    if config_value == "all":
        return True, f"scope all by config; budget left {after}"
    if left is None:
        return False, "headroom unreadable"
    if not cache_enabled:
        fits = "fits" if left >= 0 else "does not fit"
        return (
            left >= 0,
            f"prompt cache off; the copy {fits} beside the reserve; budget left {after}",
        )
    if cache_budget is None:
        if left >= TILE_COPY_MIN_BUDGET_GIB * _GIB:
            return True, f"budget left {after}"
        return False, (
            f"copy would leave {after} of prompt-cache budget (< {TILE_COPY_MIN_BUDGET_GIB})"
        )
    configured = f"configured budget {cache_budget / _GIB:.2f} GiB"
    if left >= cache_budget:
        return True, f"{configured} fits beside the copy; budget left {after}"
    return False, f"{configured} does not fit beside the copy; budget left {after}"


def _zero_weights(n: int, k: int) -> projkernel.Weights:
    """Slab planes of zeros for an [n, k] projection, unevaluated."""
    import mlx.core as mx

    info = SlabInfo(n_real=n, k=k, tiles=-(-n // slab.TILE))
    codes = mx.zeros((info.code_words,), dtype=mx.uint32)
    scales = mx.zeros((info.units,), dtype=mx.bfloat16)
    biases = mx.zeros((info.units,), dtype=mx.bfloat16)
    return projkernel.Weights(codes, scales, biases, info)


def _mlp_groups(root: Any) -> list[tuple[Any, ...]]:
    """One group per projection-kernel dispatch a slab-only MLP reaches: gate+up as
    the verifier fuses them, then each linear alone, deduplicated by K and widths."""
    groups: dict[tuple[int, tuple[int, ...]], tuple[Any, ...]] = {}
    for layer in getattr(getattr(root, "model", None), "layers", ()):
        mlp = getattr(layer, "mlp", None)
        linears = [getattr(mlp, name, None) for name in _MLP_NAMES]
        if any(m is None for m in linears):
            continue
        gate, up, down = linears
        for group in ((gate, up), (gate,), (up,), (down,)):
            key = (
                projkernel._in_features(group[0]),
                tuple(projkernel._out_features(m) for m in group),
            )
            groups.setdefault(key, group)
    return list(groups.values())


def _slab_paths(groups: Sequence[Sequence[projkernel.Weights]], inputs: dict[int, Any]) -> None:
    """Run each path a slab-only linear takes outside the tile once: the projection
    kernel's slab bodies at T = 1..MAX_ROWS over every group, and over every lone
    linear its one-row slab body and int8's slab GEMM. The GEMM's row count is a
    runtime input, so one count compiles it; ``inputs`` holds SLAB_MIN_ROWS rows
    per K, already evaluated."""
    import mlx.core as mx

    for weights in groups:
        info = weights[0].slab
        if info is None:
            raise ValueError("projection tile: a slab path handed mlx's layout")
        x = inputs[info.k]
        mx.eval([projkernel.mma(x[:t], list(weights)) for t in range(1, MAX_ROWS + 1)])
        if len(weights) > 1:
            continue
        (entry,) = weights
        # The real `mma` already sends T = 1 to the one-row kernel; this states it,
        # so a stand-in `mma` that does not still leaves it covered.
        if info.k <= projkernel._ROW_MAX_K:
            mx.eval(projkernel._row(x[:1], entry))
        stage = int8prefill.stage_a(x)
        mx.eval(int8prefill.qmm(*stage, entry.w, entry.scales, entry.biases, slab=info))


def _compile_real_shapes(root: Any) -> None:
    """Compile and run, on zero planes at every distinct shape the model's MLP
    holds, every path a slab-only MLP linear is served by outside the tile: the
    projection kernel's slab bodies at T = 1..MAX_ROWS (gate+up fused, each linear
    alone), its one-row slab body and int8's slab GEMM. Before anything is
    converted, so a toolchain that rejects one at these shapes refuses the tile
    with the model untouched rather than through the revert. Raises what the
    kernels raise. The planes and rows are built and evaluated before the first
    launch: a compile failure with a CPU-stream op in flight deadlocks mlx's
    exception path. The counters are left as found."""
    import mlx.core as mx

    planes: dict[tuple[int, int], projkernel.Weights] = {}
    groups: list[list[projkernel.Weights]] = []
    for group in _mlp_groups(root):
        k = projkernel._in_features(group[0])
        entries = []
        for module in group:
            n = projkernel._out_features(module)
            if (k, n) not in planes:
                planes[(k, n)] = _zero_weights(n, k)
            entries.append(planes[(k, n)])
        groups.append(entries)
    inputs = {
        k: mx.zeros((int8prefill.SLAB_MIN_ROWS, k), dtype=mx.bfloat16)
        for k in sorted({k for k, _ in planes})
    }
    mx.eval(list(planes.values()), list(inputs.values()))
    counted = dict(projkernel.calls)
    try:
        _slab_paths(groups, inputs)
    finally:
        projkernel.calls.update(counted)


def _call_shape(launches: Sequence[Launch]) -> tuple[Any, ...]:
    """What tells two calls' launches apart: each launch's key, member count and
    form."""
    return tuple(
        (launch.key, len(launch.members), slab.slab_info(launch.members[0]) is not None)
        for launch in launches
    )


def _tile_calls(root: Any, scope: str) -> list[tuple[tuple[Any, ...], tuple[Launch, ...]]]:
    """One call, with its launches, per distinct set of launches the model reaches
    on the tile: each layer's verifier calls (gate+up and down; at scope "all" the
    attention or GDN input projections whole and o_proj or out_proj) and every
    linear of them alone, as a module call of two to eight rows reaches it. A call
    whose key has no plan stays on the projection kernel and is left out."""
    candidates: list[tuple[Any, ...]] = []
    for layer in root.model.layers:
        mlp = layer.mlp
        candidates += [(mlp.gate_proj, mlp.up_proj), (mlp.down_proj,)]
        candidates += [(mlp.gate_proj,), (mlp.up_proj,)]
        if scope != "all":
            continue
        if layer.is_linear:
            gdn = layer.linear_attn
            inputs = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
            output = gdn.out_proj
        else:
            attention = layer.self_attn
            inputs = (attention.q_proj, attention.k_proj, attention.v_proj)
            output = attention.o_proj
        candidates += [inputs, (output,), *((m,) for m in inputs)]
    seen: set[tuple[Any, ...]] = set()
    calls_: list[tuple[tuple[Any, ...], tuple[Launch, ...]]] = []
    for call in candidates:
        launches = plan_for_call(call)
        if launches is None or _call_shape(launches) in seen:
            continue
        seen.add(_call_shape(launches))
        calls_.append((call, launches))
    return calls_


def warm_up(root: Any, scope: str) -> None:
    """Compile and run, once the model is converted (and, at scope "all", copied),
    every launch it reaches on the tile, the dual included, and every path its
    slab-only MLP takes outside the tile, each on zero rows: no compile first
    happens inside a user's turn, and a kernel that fails at the model's real
    planes fails here, inside enable()'s revert. The rows are built and evaluated
    before the first launch, for the deadlock _compile_real_shapes() describes;
    the counters are left as found."""
    import mlx.core as mx

    calls_ = _tile_calls(root, scope)
    groups = [
        [projkernel._as_weights(projkernel._weights(m)) for m in group]
        for group in _mlp_groups(root)
    ]
    ks = {projkernel._in_features(call[0]) for call, _ in calls_}
    ks |= {entry.slab.k for group in groups for entry in group if entry.slab is not None}
    inputs = {k: mx.zeros((int8prefill.SLAB_MIN_ROWS, k), dtype=mx.bfloat16) for k in sorted(ks)}
    mx.eval(list(inputs.values()))
    counted = dict(projkernel.calls)
    try:
        for call, launches in calls_:
            x = inputs[projkernel._in_features(call[0])]
            mx.eval(verify(call, x[:MAX_ROWS], launches))
        _slab_paths(groups, inputs)
    finally:
        projkernel.calls.update(counted)


def _as_mlx(module: Any) -> tuple[Any, Any, Any]:
    """The planes the tile reads for ``module`` back in mlx's layout, evaluated: a
    slab-only linear's own, or its columns of its stack's copy."""
    import mlx.core as mx

    info = slab.slab_info(module)
    if info is not None:
        arrays = slab.unpack(module.weight, module.scales, module.biases, info)
    else:
        stack = stack_of(module)
        if stack is None:
            raise ValueError("projection tile: a linear without planes")
        start, stop = stack.slices[vars(module)[INDEX_ATTR]]
        whole = slab.unpack(stack.codes, stack.scales, stack.biases, stack.info)
        arrays = tuple(a[start:stop] for a in whole)
    w, scales, biases = (mx.contiguous(a) for a in arrays)
    mx.eval(w, scales, biases)
    return w, scales, biases


def _reference(x: Any, w: Any, scales: Any, biases: Any) -> Any:
    """x [T, K] through one linear in fp32: the weights dequantised with scales and
    biases upcast to fp32, every product and sum an fp32 elementwise op, not a
    matmul, which on the tensor units may run at TF32. A row at a time, so one
    [N, K] product is alive at once; evaluated."""
    import mlx.core as mx

    dense = mx.dequantize(
        w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=slab.GROUP, bits=4
    )
    x32 = x.astype(mx.float32)
    rows = []
    for r in range(x.shape[0]):
        row = mx.sum(dense * x32[r], axis=-1)
        mx.eval(row)
        rows.append(row)
    out = mx.stack(rows)
    mx.eval(out)
    return out


def _check_tile_call(
    call: tuple[Any, ...],
    launches: tuple[Launch, ...],
    names: list[str],
    x: Any,
    refs: dict[int, tuple[Any, float]],
) -> str | None:
    """One call on the tile, at its plans, on its real planes, over ``x``
    [MAX_ROWS, K] bf16, evaluated: every row at T = 1..MAX_ROWS equals the same
    row at T = 1; each linear's rows sit within TILE_FLOOR_FACTOR of the bf16
    floor against an fp32 reference (block invariance compares the tile with
    itself; this catches a kernel that lost precision); and a NaN in one row
    leaves the others bitwise alone. ``refs`` holds each linear's reference and
    floor by id, computed on first use: the probe calls a fused group's linears
    alone too, over the same ``x``, and the reference is the costly step."""
    import mlx.core as mx

    label = "+".join(names)
    nan = mx.array(float("nan"), dtype=mx.bfloat16)
    x_nan = mx.where(mx.arange(MAX_ROWS)[:, None] == _NAN_ROW, nan, x)
    mx.eval(x_nan)
    alone = [verify(call, x[r : r + 1], launches) for r in range(MAX_ROWS)]
    rows = [mx.concatenate([outs[i] for outs in alone]) for i in range(len(call))]
    mx.eval(rows)
    del alone
    full: tuple[Any, ...] = ()
    for t in range(1, MAX_ROWS + 1):
        full = verify(call, x[:t], launches)
        same = mx.stack([mx.array_equal(o, r[:t]) for o, r in zip(full, rows, strict=True)])
        if not mx.all(same).item():
            return f"T-invariance: {label} at T={t} differs from its rows alone"
    for module, got, name in zip(call, full, names, strict=True):
        if id(module) not in refs:
            ref = _reference(x, *_as_mlx(module))
            refs[id(module)] = ref, tileattn._rel_rms(ref.astype(mx.bfloat16), ref)
        ref, floor = refs[id(module)]
        error = projkernel._stock_distance(got, ref)
        if not error <= TILE_FLOOR_FACTOR * floor:
            return (
                f"floor: {name} is {error:.4g} from its fp32 reference, above "
                f"{TILE_FLOOR_FACTOR}x the bf16 floor {floor:.4g}"
            )
    keep = mx.array([r for r in range(MAX_ROWS) if r != _NAN_ROW])
    for got, clean, name in zip(verify(call, x_nan, launches), full, names, strict=True):
        if not mx.array_equal(mx.take(got, keep, axis=0), mx.take(clean, keep, axis=0)).item():
            return f"isolation: a NaN in row {_NAN_ROW} changed the other rows of {name}"
    return None


def _slab_cases(mlp: Any, names: dict[int, str]) -> list[tuple[str, Any, list[Any], list[Any]]]:
    """(label, [MIN_ROWS, K] rows, the slab entries, the same linears in mlx's
    layout): the MLP's own linears alone, gate+up fused, then a synthetic linear
    at each of the MLP's K in each CANCELLING layout, repacked. Evaluated."""
    import mlx.core as mx

    linears = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    entries = {id(m): projkernel._weights(m) for m in linears}
    mlx_form = {id(m): _as_mlx(m) for m in linears}
    rows: dict[int, Any] = {}
    cases: list[tuple[str, Any, list[Any], list[Any]]] = []
    for i, module in enumerate(linears):
        k = projkernel._in_features(module)
        if k not in rows:
            rows[k] = mx.random.normal((int8prefill.MIN_ROWS, k), key=mx.random.key(200 + i))
            rows[k] = rows[k].astype(mx.bfloat16)
        cases.append(
            (names.get(id(module), "?"), rows[k], [entries[id(module)]], [mlx_form[id(module)]])
        )
    gate, up = mlp.gate_proj, mlp.up_proj
    cases.append(
        (
            f"{names.get(id(gate), '?')}+{names.get(id(up), '?')}",
            rows[projkernel._in_features(gate)],
            [entries[id(gate)], entries[id(up)]],
            [mlx_form[id(gate)], mlx_form[id(up)]],
        )
    )
    for i, k in enumerate(sorted(rows)):
        keys = mx.random.split(mx.random.key(300 + i), len(projkernel.CANCELLING))
        for j, layout in enumerate(projkernel.CANCELLING):
            x, (linear,) = projkernel._cancelling_case(
                int8prefill.MIN_ROWS, k, (_CANCEL_N,), layout, keys[j]
            )
            codes, scales, biases, info = slab.repack(*linear)
            cases.append(
                (
                    f"a {layout}-cancelling K={k} linear",
                    x,
                    [projkernel.Weights(codes, scales, biases, info)],
                    [linear],
                )
            )
    mx.eval(list(rows.values()), [case[2:] for case in cases])
    return cases


def _check_slab(label: str, x: Any, entries: list[Any], mlx_form: list[Any]) -> str | None:
    """One case's slab paths bitwise against mlx's layout: the projection kernel at
    T = 1..MAX_ROWS and, for a lone linear, the one-row kernel on each of those
    rows and int8's GEMM at each of _INT8_ROWS."""
    import mlx.core as mx

    checks: list[tuple[str, Any]] = []
    for t in range(1, MAX_ROWS + 1):
        got, want = projkernel.mma(x[:t], entries), projkernel.mma(x[:t], mlx_form)
        where = f"the projection kernel at T={t}"
        checks += [(where, mx.array_equal(g, w)) for g, w in zip(got, want, strict=True)]
    if len(entries) == 1:
        (entry,), (linear,) = entries, mlx_form
        for r in range(MAX_ROWS):
            row = x[r : r + 1]
            same = mx.array_equal(projkernel._row(row, entry), projkernel._row(row, linear))
            checks.append(("the one-row kernel", same))
        for m in _INT8_ROWS:
            stage = int8prefill.stage_a(x[:m])
            got = int8prefill.qmm(*stage, entry.w, entry.scales, entry.biases, slab=entry.slab)
            checks.append(
                (f"int8's GEMM at {m} rows", mx.array_equal(got, int8prefill.qmm(*stage, *linear)))
            )
    same = mx.stack([check for _, check in checks])
    mx.eval(same)
    for i, (where, _) in enumerate(checks):
        if not same[i].item():
            return f"slab: {label} differs from mlx's layout on {where}"
    return None


def _check_routing(mlp: Any) -> str | None:
    """One decode-shaped call through both hooks: the exact verifier's
    feed-forward over _PROBE_ROWS rows and the MLP's own call of as many. Inside a
    window the tile serves all of it; outside, none of it, and the projection
    kernel serves it. The flag is raised for the step; probe() clears it."""
    import mlx.core as mx

    global _active
    x = mx.random.normal(
        (1, _PROBE_ROWS, projkernel._in_features(mlp.gate_proj)), key=mx.random.key(0)
    )
    x = x.astype(mx.bfloat16)
    mx.eval(x)
    verifier = projkernel._verifier_class()()
    counts = projkernel.calls
    _active = 1
    open_window()
    try:
        before = dict(counts)
        mx.eval(verifier._feed_forward(mlp, x))
        middle = dict(counts)
        mx.eval(mlp(x))
        inside = dict(counts)
    finally:
        close_window()
    mx.eval(verifier._feed_forward(mlp, x), mlp(x))
    after = dict(counts)
    if middle["verify_tile"] - before["verify_tile"] != 2:
        return "routing: the exact verifier's MLP did not reach the tile inside the window"
    if inside["plain_tile"] - middle["plain_tile"] != 3:
        return "routing: the MLP's own calls did not reach the tile inside the window"
    if after["verify_tile"] != inside["verify_tile"] or after["plain_tile"] != inside["plain_tile"]:
        return "routing: the tile served the MLP outside the window"
    if (
        after["verify_kernel"] - inside["verify_kernel"],
        after["plain_kernel"] - inside["plain_kernel"],
    ) != (2, 3):
        return "routing: outside the window the MLP did not reach the projection kernel"
    return None


def _probe_steps(root: Any, scope: str) -> str | None:
    """Every input is made with GPU ops and evaluated before the launches that
    read it, for the deadlock _compile_real_shapes() describes."""
    import mlx.core as mx

    layers = list(root.model.layers)
    full = next((layer for layer in layers if not layer.is_linear), None)
    linear = next((layer for layer in layers if layer.is_linear), None)
    if full is None or linear is None:
        return "the model lacks a full-attention or a linear-attention layer"
    names = {id(module): path for path, module in root.named_modules()}

    # On the tile, the first full-attention layer's MLP (layer 3 of the 27B) and,
    # at scope "all", its attention and the first linear-attention layer's GDN
    # (layer 0): the verifier's calls, then each of their linears alone.
    mlp = full.mlp
    calls_: list[tuple[Any, ...]] = [(mlp.gate_proj, mlp.up_proj), (mlp.down_proj,)]
    if scope == "all":
        attention, gdn = full.self_attn, linear.linear_attn
        calls_ = [
            (attention.q_proj, attention.k_proj, attention.v_proj),
            (attention.o_proj,),
            *calls_,
            (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
            (gdn.out_proj,),
        ]
    calls_ += [(m,) for call in list(calls_) if len(call) > 1 for m in call]
    # The guards gave every MLP key a plan; a copied group without one was not
    # copied, and the projection kernel keeps its calls.
    converted = {id(mlp.gate_proj), id(mlp.up_proj), id(mlp.down_proj)}
    # One x per K, so a linear's fp32 reference serves its fused call and its
    # call alone; at most a few MB of [MAX_ROWS, N] fp32 rows, dropped before
    # the steps below.
    xs: dict[int, Any] = {}
    refs: dict[int, tuple[Any, float]] = {}
    for call in calls_:
        launches = plan_for_call(call)
        call_names = [names.get(id(m), "?") for m in call]
        if launches is None:
            if id(call[0]) in converted:
                return f"no launch on the tile for {'+'.join(call_names)}"
            continue
        k = projkernel._in_features(call[0])
        if k not in xs:
            x = mx.random.normal((MAX_ROWS, k), key=mx.random.key(len(xs) + 1))
            xs[k] = x.astype(mx.bfloat16)
            mx.eval(xs[k])
        reason = _check_tile_call(call, launches, call_names, xs[k], refs)
        if reason is not None:
            return reason
    del xs, refs

    # Off the tile: the slab-only MLP's other paths, bitwise against mlx's layout.
    for case in _slab_cases(mlp, names):
        reason = _check_slab(*case)
        if reason is not None:
            return reason

    return _check_routing(mlp)


def probe(root: Any, scope: str) -> str | None:
    """None when the tile may serve this converted (and, at scope "all", copied)
    model on this GPU, else why not. Runs on the loading thread after the
    warm-up; raises the flag and opens a window only around the calls that must
    reach the hooks, and leaves the flag clear, the window closed, the round flag
    and the counters as it found them. Its arrays are gone before the cache is
    cleared."""
    import mlx.core as mx

    global _active, _window, _round_flag
    round_flag = _round_flag
    counted = (dict(projkernel.calls), dict(int8prefill.calls))
    try:
        return _probe_steps(root, scope)
    except BaseException as e:
        # The traceback holds the steps' frames, and every probe array in them,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _active, _window, _round_flag = 0, None, round_flag
        projkernel.calls.update(counted[0])
        int8prefill.calls.update(counted[1])
        mx.clear_cache()


def _undo(root: Any) -> str | None:
    """Take a refused load back to mlx's layout: the MLP reverted, then the copies
    dropped. A revert that raises leaves the layers it did not reach slab-only and
    marked, each layer whole (revert_mlp moves a layer's gate, up and down
    together); every reader is marker-driven, so they keep their slab paths, and
    the fork key reads the model as converted or partly so. Returns None when the
    MLP is back in mlx's layout, else what the refusal's reason adds: how many
    linears stay slab-only."""
    failed = False
    try:
        revert_mlp(root)
    except Exception:  # noqa: BLE001 — the markers stay authoritative
        failed = True
        logger.warning(
            "projection tile: the revert failed; the linears it did not reach keep "
            "their slab paths",
            exc_info=True,
        )
    try:
        drop_stacks(root)
    except Exception:  # noqa: BLE001 — a copy left behind is served by nothing
        logger.warning("projection tile: dropping the copies failed", exc_info=True)
    if not failed:
        return None
    linears = list(_mlp_linears(root))
    left = sum(slab.slab_info(m) is not None for m in linears)
    return f"the revert failed with {left} of {len(linears)} MLP linears slab-only"


def _with_undo(reason: str, root: Any) -> str:
    """A refusal's one-line reason once the conversion is undone, with what a
    failed revert left behind."""
    note = _undo(root)
    line = _error_line(reason)
    return line if note is None else f"{line}; {note}"


def enable(
    model: Any,
    *,
    config_value: bool | str,
    kernel_status: dict[str, Any],
    tile_status: dict[str, Any],
    int8_status: dict[str, Any],
    drafter_kind: str | None,
    cache_enabled: bool,
    cache_budget: int | None,
    headroom_bytes: int | None,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Decide whether this load's verify projections run on the tile, and at which
    scope; on the loading thread, after int8 prefill and the projection kernel
    and before the drafter's route, so neither the drafter's tags nor its flag
    are set while the guards, the warm-up and the probe run (the probe drives
    the target's MLP through both hooks). ``cache_enabled`` is whether the
    prompt cache is on, ``cache_budget`` the configured prompt-cache budget in
    bytes (None when automatic), ``headroom_bytes`` live_headroom() read now,
    ``reserve_bytes`` the turn's own reserve. Returns the status the engine
    exposes and never raises: a model load must not fail because an accelerator
    is missing. probe_seconds covers the real-shape compile, the conversion, the
    copy, the warm-up and the probe, the tile's whole cost to a load."""
    global _active, _scope
    # First, on every call: a flag, a window or a scope left from an earlier load
    # must never serve this one, whatever happens below.
    clear()
    if not config_value:
        return _status("off")
    root: Any = None
    touched = False
    seconds: float | None = None
    try:
        reason = (
            _kernel_reason(kernel_status)
            or projkernel._tile_reason(tile_status)
            or _drafter_reason(drafter_kind)
            or _int8_reason(int8_status)
            or _cores_reason()
        )
        if reason is not None:
            return _refuse(reason, expected=True)
        root = int8prefill._root(model)
        reason = _model_reason(root)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = _pins_reason()
        if reason is None:
            available = availability()
            if not available.available:
                reason = available.reason or "the projection tile kernels did not run"
        if reason is not None:
            return _refuse(reason)
        built, why, scope = 0, None, "mlp"
        start = time.perf_counter()
        try:
            try:
                _compile_real_shapes(root)
            except Exception as e:  # noqa: BLE001 — a kernel this OS rejects refuses the load
                # The one-line reason cannot carry a compiler's full report; the log can.
                logger.warning(
                    "projection tile: the kernels failed at the model's shapes", exc_info=True
                )
                reason = f"real shapes: {_error_line(str(e) or type(e).__name__)}"
            else:
                wanted = copy_bytes(root, keys_with_plans=_has_plan)
                copy, why = copy_decision(
                    config_value,
                    cache_budget,
                    headroom_bytes,
                    wanted,
                    reserve_bytes,
                    cache_enabled=cache_enabled,
                )
                touched = True
                if convert_mlp(root) == 0:
                    raise RuntimeError("the MLP conversion converted no linear")
                if copy:
                    built = build_stacks(root, keys_with_plans=_has_plan)
                    # Nothing built serves exactly what scope "mlp" serves: say so.
                    scope, why = ("all", why) if built else ("mlp", "no stack key has a plan")
                warm_up(root, scope)
                reason = probe(root, scope)
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(_with_undo(reason, root) if touched else reason, seconds)
        left = budget_left(headroom_bytes, built, reserve_bytes)
        left_gib = None if left is None else round(left / _GIB, 2)
        _active, _scope = 1, scope
        logger.info(
            "projection tile: scope=%s copy=%.1f GB budget left %s probe %.2fs%s",
            scope,
            built / 1e9,
            "unknown" if left_gib is None else f"{left_gib:.1f} GiB",
            seconds,
            f" ({why})" if scope == "mlp" else "",
        )
        return _status(
            "active",
            scope=scope,
            reason=why if scope == "mlp" else None,
            copy_gb=built / 1e9 if scope == "all" else None,
            probe_seconds=seconds,
            budget_left_gib=left_gib,
        )
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        clear()
        # The frames hold the failed step's arrays: release them before the revert.
        traceback.clear_frames(e.__traceback__)
        # The warning carries one line; the traceback goes to the log.
        logger.warning("projection tile: enabling failed", exc_info=True)
        reason = str(e) or type(e).__name__
        return _refuse(_with_undo(reason, root) if touched else reason, seconds)


def static_reason(config: Any, model_config: dict, drafter_kind: str | None) -> str | None:
    """Why the tile would not be active for this config, model and drafter, or None
    when it would be, decided without loading anything (sous tune). It mirrors
    enable()'s order and words: the switch, the projection kernel's whole static
    check (which carries the attention tile's), the drafter, int8 prefill, the
    core count, then the plans for the MLP's keys read from config.json. What only
    a load can see, a probe or a compile failure, pin drift, is beyond it."""
    if not config.projection_tile:
        return "projection tile off"
    reason = projkernel.static_reason(config, model_config, drafter_kind)
    if reason is not None:
        return reason
    reason = _drafter_reason(drafter_kind)
    if reason is not None:
        return reason
    if not config.int8_prefill:
        return "int8 prefill off"
    reason = _cores_reason()
    if reason is not None:
        return reason
    text = model_config.get("text_config", model_config)
    hidden, inner = text.get("hidden_size"), text.get("intermediate_size")
    for key in ((hidden, (inner, inner)), (hidden, (inner,)), (inner, (hidden,))):
        if not _has_plan(key):
            return _no_plan(key)
    return None


def _layer_kinds(text: dict) -> list[bool]:
    """Whether each layer is linear attention, from config.json's layer_types or,
    as mlx-vlm reads it, every full_attention_interval-th layer full."""
    types = text.get("layer_types")
    if types:
        return [kind == "linear_attention" for kind in types]
    interval = text.get("full_attention_interval", 4)
    return [(i + 1) % interval != 0 for i in range(text["num_hidden_layers"])]


def static_copy_bytes(model_config: dict) -> int:
    """copy_bytes() from config.json alone, for sous tune: the padded planes of
    every attention and GDN stack whose key has a plan, layer by layer as
    stack_groups() builds them. q_proj carries the attention output gate, so it is
    twice the heads' width."""
    text = model_config.get("text_config", model_config)
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    head_dim = text.get("head_dim") or hidden // heads
    kv = text["num_key_value_heads"] * head_dim
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    v_heads = text["linear_num_value_heads"]
    value_dim = v_heads * text["linear_value_head_dim"]
    linear: list[Key] = [
        (hidden, (2 * key_dim + value_dim, value_dim)),
        (hidden, (v_heads, v_heads)),
        (value_dim, (hidden,)),
    ]
    full: list[Key] = [(hidden, (2 * heads * head_dim, kv, kv)), (heads * head_dim, (hidden,))]
    return sum(
        _stack_bytes(key)
        for is_linear in _layer_kinds(text)
        for key in (linear if is_linear else full)
        if _has_plan(key)
    )
