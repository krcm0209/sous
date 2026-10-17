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

mlx is imported inside each function.
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from sous.engine import int8prefill, nax, projkernel, slab
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text
from sous.engine.slab import SlabInfo

logger = logging.getLogger("sous.engine.projtile")

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


# --------------------------------------------------------------------------------
# The kernel and its plans
# --------------------------------------------------------------------------------

# Rows one launch computes: the tile is compiled at M = 8 alone, and a call of fewer
# rows is padded with zero rows (pad8), so T is never an argument.
MAX_ROWS = 8
# The tile against an fp32 reference: its relative RMS within this factor of the
# floor, that reference rounded to bf16. The spike's tile sat on the floor to four
# digits on every linear of the 27B.
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


def _call_keys(linears: Sequence[Any]) -> tuple[Key, ...] | None:
    """The key of each launch a call of ``linears`` takes, in call order: a lone
    linear's own K and width (the alone form, whether it is slab-only or a stack
    member); gate+up's (two slab-only linears of one shape, one dual launch); a
    stack's key for each run of linears that covers its stack whole (the GDN call
    makes two, qkv+z and b+a). None when a linear has no planes, the forms mix, or
    a run covers part of a stack."""
    if not linears:
        return None
    infos = [slab.slab_info(m) for m in linears]
    stacks = [stack_of(m) for m in linears]
    if any(info is None and stack is None for info, stack in zip(infos, stacks, strict=True)):
        return None
    if len(linears) == 1:
        (module,) = linears
        return ((projkernel._in_features(module), (projkernel._out_features(module),)),)
    if all(info is not None for info in infos):
        first = infos[0]
        if len(infos) == 2 and first is not None and infos[1] == first:
            return ((first.k, (first.n_real, first.n_real)),)
        return None
    if any(info is not None for info in infos):
        return None
    keys: list[Key] = []
    start = 0
    while start < len(linears):
        stack = stacks[start]
        stop = start + 1
        while stop < len(linears) and stacks[stop] is stack:
            stop += 1
        widths = tuple(projkernel._out_features(m) for m in linears[start:stop])
        if stack is None or widths != stack.key[1]:
            return None
        keys.append(stack.key)
        start = stop
    return tuple(keys)


def plan_for_call(linears: Sequence[Any]) -> tuple[Plan, ...] | None:
    """One plan per launch a verifier or module call of ``linears`` takes, in call
    order, or None when any linear has no planes or any launch's key has no plan:
    the whole call then stays on the projection kernel, never split between the
    two."""
    keys = _call_keys(linears)
    if keys is None:
        return None
    plans: list[Plan] = []
    for key in keys:
        plan = PLANS.get(key)
        if plan is None:
            return None
        plans.append(plan)
    return tuple(plans)


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
    units = info.n_pad * info.groups
    if (
        codes.dtype != mx.uint32
        or codes.size * 4 != units * slab.UNIT_BYTES
        or any(p.dtype != mx.bfloat16 or p.size != units for p in (scales, biases))
    ):
        raise ValueError(f"projection tile: the planes do not hold {info}")


def run(x8: Any, codes: Any, scales: Any, biases: Any, info: SlabInfo, plan: Plan) -> Any:
    """x8 [8, K] bf16 through one slab-ordered projection or stack at ``plan``:
    [8, n_pad] bf16, the columns past n_real zero. A split plan runs the split
    kernel, then the reduce that adds its partials in order."""
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
