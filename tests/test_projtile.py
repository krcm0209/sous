"""The projection tile (sous.engine.projtile) and the slab layout it reads
(sous.engine.slab): the layout and its inverse, the slab-only MLP conversion and
its revert, the stacked copies of the attention and GDN linears, the fork key's
reading of the markers, the tile kernel with its plans, and the routing state and
launches the projection kernel's hooks reach it through, and enable() with its
guards, memory rule, warm-up and probe. Plain mlx except the `tile_nax` tests,
which run the kernel on the tensor units."""

import contextlib
import gc
import logging
import types
import warnings
import weakref
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
nn = pytest.importorskip("mlx.nn")

from sous.engine import (  # noqa: E402 — after the guards
    gpucores,
    int8prefill,
    nax,
    projkernel,
    projtile,
    slab,
)
from tests import proj_fixtures as pfx  # noqa: E402
from tests import projtile_fixtures as ptfx  # noqa: E402
from tests.projtile_fixtures import tile_nax  # noqa: E402

K = 1024
_UINT = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}


def _same(got, want) -> bool:
    """Shape, dtype and every bit (a bf16 -0.0 is not +0.0 here)."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return False
    as_bits = _UINT[want.dtype.size]
    return bool(mx.array_equal(got.view(as_bits), want.view(as_bits)).item())


def _mlp_linears(root) -> list:
    return [
        getattr(layer.mlp, name)
        for layer in root.model.layers
        for name in ("gate_proj", "up_proj", "down_proj")
    ]


def _arrays(module) -> tuple:
    return module.weight, module.scales, module.biases


# ---- the layout ------------------------------------------------------------------


@pytest.mark.parametrize("n", [256, 4160, 48, 17408], ids=lambda n: f"n{n}")
def test_repack_round_trip_is_bitwise(n):
    w, s, b = pfx.rand_linear(n, K, seed=n)
    codes, scales, biases, info = slab.repack(w, s, b)
    tiles = -(-n // 256)
    assert info == slab.SlabInfo(n_real=n, k=K, tiles=tiles)
    assert (info.n_pad, info.groups) == (tiles * 256, K // 64)
    assert codes.dtype == mx.uint32 and codes.shape == (tiles * (K // 64) * 256 * 8,)
    assert scales.shape == biases.shape == (tiles * (K // 64) * 256,)
    assert scales.dtype == biases.dtype == mx.bfloat16
    for got, want in zip(slab.unpack(codes, scales, biases, info), (w, s, b), strict=True):
        assert _same(got, want)


def test_repack_addresses_match_the_slab_formula():
    """Column n = 256t + r, input k = 64g + j: its code sits in byte
    ((tG + g)256 + r)32 + j/2, the low nibble for even j, and its scale and bias
    at (tG + g)256 + r. N 300 pads, so a second tile and its offsets are seen."""
    n, k = 300, 512
    groups = k // 64
    w, s, b = pfx.rand_linear(n, k, seed=7)
    codes, scales, biases, info = slab.repack(w, s, b)
    assert info.tiles == 2

    rows, cols = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    words = np.array(w)
    # mlx packs code k into word k // 8 at bits 4 (k % 8).
    want = (words[rows, cols // 8] >> (4 * (cols % 8))) & 0xF
    unit = ((rows // 256) * groups + cols // 64) * 256 + rows % 256
    code_bytes = np.array(codes).view(np.uint8)
    j = cols % 64
    got = (code_bytes[unit * 32 + j // 2] >> (4 * (j % 2))) & 0xF
    np.testing.assert_array_equal(got, want)

    rows, gs = np.meshgrid(np.arange(n), np.arange(groups), indexing="ij")
    at = ((rows // 256) * groups + gs) * 256 + rows % 256
    for plane, source in ((scales, s), (biases, b)):
        flat = np.array(plane.view(mx.uint16))
        np.testing.assert_array_equal(flat[at], np.array(source.view(mx.uint16)))


def test_padded_columns_are_zero():
    w, s, b = pfx.rand_linear(48, K, seed=3)
    codes, scales, biases, info = slab.repack(w, s, b)
    groups = info.groups
    assert info.tiles == 1
    unit_codes = codes.reshape(1, groups, 256, 8)
    assert not mx.any(unit_codes[:, :, 48:]).item()
    assert mx.any(unit_codes[:, :, :48]).item()
    for plane in (scales, biases):
        bits = plane.view(mx.uint16).reshape(1, groups, 256)
        assert not mx.any(bits[:, :, 48:]).item()
        assert mx.any(bits[:, :, :48]).item()


def test_repack_refuses_a_layout_other_than_affine_q4_group_64():
    dense = (mx.random.normal((256, K), key=mx.random.key(0)) * 0.02).astype(mx.bfloat16)
    w, s, b = mx.quantize(dense, group_size=32, bits=4)
    with pytest.raises(ValueError, match="group-64"):
        slab.repack(w, s, b)
    w, s, b = pfx.rand_linear(256, K, seed=1)
    with pytest.raises(ValueError, match="uint32"):
        slab.repack(w.reshape(-1), s, b)


# ---- the slab-only MLP -----------------------------------------------------------


def test_convert_mlp_replaces_the_arrays_and_marks_the_module():
    root = pfx.tiny_quantized_model(seed=0)
    linears = _mlp_linears(root)
    before = [_arrays(m) for m in linears]
    assert projtile.convert_mlp(root) == 12
    for module, (w, s, b) in zip(linears, before, strict=True):
        n, k = w.shape[0], s.shape[1] * 64
        info = slab.slab_info(module)
        assert info == slab.SlabInfo(n_real=n, k=k, tiles=-(-n // 256))
        units = info.tiles * info.groups * 256
        assert module.weight.ndim == 1 and module.weight.size == units * 8
        assert module.scales.ndim == 1 and module.scales.size == units
        assert module.biases.ndim == 1 and module.biases.size == units
        for got, want in zip(_arrays(module), slab.repack(w, s, b)[:3], strict=True):
            assert _same(got, want)
        # mlx's parameter tree holds the planes; the marker stays out of it.
        params = module.parameters()
        for name in ("weight", "scales", "biases"):
            assert _same(params[name], getattr(module, name))
        assert slab.SLAB_ATTR not in params and slab.SLAB_ATTR in vars(module)
    assert projtile.convert_mlp(root) == 0


def test_stock_quantized_matmul_raises_on_a_converted_module():
    root = pfx.tiny_quantized_model(seed=0)
    projtile.convert_mlp(root)
    module = root.model.layers[0].mlp.gate_proj
    x = pfx.activations(2, 256, seed=1)
    with pytest.raises(ValueError):
        mx.eval(mx.quantized_matmul(x, *_arrays(module), transpose=True, group_size=64, bits=4))
    with pytest.raises(ValueError):
        mx.eval(module(x))


def test_revert_mlp_restores_the_arrays_bitwise_and_clears_the_marker():
    root = pfx.tiny_quantized_model(seed=0)
    linears = _mlp_linears(root)
    before = [_arrays(m) for m in linears]
    xs = [pfx.activations(3, m.scales.shape[1] * 64, seed=i) for i, m in enumerate(linears)]
    stock = [m(x) for m, x in zip(linears, xs, strict=True)]
    mx.eval(stock)
    projtile.convert_mlp(root)
    assert projtile.revert_mlp(root) == 12
    for module, arrays, x, want in zip(linears, before, xs, stock, strict=True):
        assert slab.slab_info(module) is None and slab.SLAB_ATTR not in vars(module)
        for got, original in zip(_arrays(module), arrays, strict=True):
            assert _same(got, original)
        assert _same(module(x), want)
    assert projtile.revert_mlp(root) == 0


def test_a_padded_module_converts_and_reverts_bitwise():
    module = ptfx.padded_module(48, K, seed=5)
    before = _arrays(module)
    x = pfx.activations(2, K, seed=6)
    want = module(x)
    mx.eval(want)
    projtile._convert(module)
    info = slab.slab_info(module)
    assert info == slab.SlabInfo(n_real=48, k=K, tiles=1)
    projtile._revert(module, info)
    assert slab.slab_info(module) is None
    for got, original in zip(_arrays(module), before, strict=True):
        assert _same(got, original)
    assert _same(module(x), want)


# ---- the stacks ------------------------------------------------------------------


def _layer_groups(root) -> list:
    out = []
    for layer in root.model.layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            out += [
                (gdn.in_proj_qkv, gdn.in_proj_z),
                (gdn.in_proj_b, gdn.in_proj_a),
                (gdn.out_proj,),
            ]
        else:
            attention = layer.self_attn
            out += [
                (attention.q_proj, attention.k_proj, attention.v_proj),
                (attention.o_proj,),
            ]
    return out


# The tiny model's stack keys in call order: GDN layer 0, attention layer 1, again.
_TINY_KEYS = [
    (256, (512, 256)),
    (256, (4, 4)),
    (256, (256,)),
    (256, (2048, 512, 512)),
    (1024, (256,)),
] * 2


def test_stack_groups_follow_the_verifiers_calls_layer_by_layer():
    root = pfx.tiny_quantized_model(seed=0)
    got = projtile.stack_groups(root)
    want = _layer_groups(root)
    assert [tuple(map(id, g)) for g in got] == [tuple(map(id, g)) for g in want]


def _assert_unpacks_to(stack, members) -> None:
    """The stack's planes unpack to its members' arrays concatenated along N, and
    each member's slice to that member's own arrays."""
    planes = slab.unpack(stack.codes, stack.scales, stack.biases, stack.info)
    for i, got in enumerate(planes):
        want = mx.concatenate([_arrays(m)[i] for m in members], axis=0)
        assert _same(got, want)
        for member, (start, stop) in zip(members, stack.slices, strict=True):
            assert _same(got[start:stop], _arrays(member)[i])


def test_build_stacks_concatenates_before_padding_and_records_slices():
    root = pfx.tiny_quantized_model(seed=0)
    projtile.build_stacks(root, keys_with_plans=lambda key: True)

    attention = root.model.layers[1].self_attn
    qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
    stack = projtile.stack_of(attention.q_proj)
    assert stack is not None and all(projtile.stack_of(m) is stack for m in qkv)
    assert stack.slices == ((0, 2048), (2048, 2560), (2560, 3072))
    assert stack.key == (256, (2048, 512, 512))
    assert stack.info == slab.SlabInfo(n_real=3072, k=256, tiles=12)
    _assert_unpacks_to(stack, qkv)
    o = projtile.stack_of(attention.o_proj)
    assert o is not None and o.key == (1024, (256,)) and o.slices == ((0, 256),)

    gdn = root.model.layers[0].linear_attn
    ba = projtile.stack_of(gdn.in_proj_b)
    assert ba is not None and projtile.stack_of(gdn.in_proj_a) is ba
    assert ba.key == (256, (4, 4)) and ba.slices == ((0, 4), (4, 8))
    assert (ba.info.tiles, ba.info.n_pad) == (1, 256)
    _assert_unpacks_to(ba, (gdn.in_proj_b, gdn.in_proj_a))
    qkvz = projtile.stack_of(gdn.in_proj_qkv)
    assert qkvz is not None and projtile.stack_of(gdn.in_proj_z) is qkvz
    assert qkvz.key == (256, (512, 256)) and qkvz.slices == ((0, 512), (512, 768))
    out = projtile.stack_of(gdn.out_proj)
    assert out is not None and out.key == (256, (256,))

    # A copy: the members keep mlx's arrays, out of whose tree the stack stays.
    for group in projtile.stack_groups(root):
        for member in group:
            assert member.weight.ndim == 2 and slab.slab_info(member) is None
            assert projtile.STACK_ATTR not in member.parameters()
    assert all(projtile.stack_of(m) is None for m in _mlp_linears(root))


def test_build_stacks_skips_a_group_without_a_plan():
    root = pfx.tiny_quantized_model(seed=0)
    asked = []

    def keys_with_plans(key):
        asked.append(key)
        return key != (256, (4, 4))

    projtile.build_stacks(root, keys_with_plans=keys_with_plans)
    assert asked == _TINY_KEYS
    for layer in (root.model.layers[0], root.model.layers[2]):
        gdn = layer.linear_attn
        assert projtile.stack_of(gdn.in_proj_b) is None
        assert projtile.stack_of(gdn.in_proj_a) is None
        assert projtile.stack_of(gdn.in_proj_qkv) is not None
        assert projtile.stack_of(gdn.out_proj) is not None


@pytest.mark.parametrize("reject", [None, (256, (4, 4))], ids=["every-key", "no-b+a"])
def test_copy_bytes_is_the_stacks_padded_bytes(reject):
    root = pfx.tiny_quantized_model(seed=0)

    def keys_with_plans(key):
        return key != reject

    want = projtile.copy_bytes(root, keys_with_plans=keys_with_plans)
    # Per layer pair: qkv+z, b+a, out_proj (K 256, 3 + 1 + 1 tiles), q+k+v (K 256,
    # 12 tiles), o_proj (K 1024, 1 tile); 9/16 byte per padded weight.
    pair = (768 * 256 + 256 * 256 + 256 * 256 + 3072 * 256 + 256 * 1024) * 9 // 16
    assert want == 2 * pair - (0 if reject is None else 2 * 256 * 256 * 9 // 16)
    assert projtile.build_stacks(root, keys_with_plans=keys_with_plans) == want
    stacks = {
        id(stack): stack
        for group in projtile.stack_groups(root)
        for stack in map(projtile.stack_of, group)
        if stack is not None
    }
    assert sum(s.codes.nbytes + s.scales.nbytes + s.biases.nbytes for s in stacks.values()) == want


def test_drop_stacks_takes_the_stack_off_every_member():
    root = ptfx.tiny_slab_model(seed=0)
    members = [m for group in projtile.stack_groups(root) for m in group]
    assert all(projtile.stack_of(m) is not None for m in members)
    projtile.drop_stacks(root)
    assert all(projtile.STACK_ATTR not in vars(m) for m in members)
    assert projtile.fork_key_state(root) == "active"


def test_tiny_slab_model_stacks_only_at_scope_all():
    for scope, stacked in (("all", True), ("mlp", False)):
        root = ptfx.tiny_slab_model(seed=0, scope=scope)
        assert all(slab.slab_info(m) is not None for m in _mlp_linears(root))
        members = [m for group in projtile.stack_groups(root) for m in group]
        assert all((projtile.stack_of(m) is not None) == stacked for m in members)
    with pytest.raises(ValueError):
        ptfx.padded_module(512, K, seed=0)


# ---- the fork key ----------------------------------------------------------------


def test_fork_key_state_follows_the_markers():
    root = pfx.tiny_quantized_model(seed=0)
    assert projtile.fork_key_state(root) == "off"
    projtile.build_stacks(root, keys_with_plans=lambda key: True)
    assert projtile.fork_key_state(root) == "off"
    projtile.convert_mlp(root)
    assert projtile.fork_key_state(root) == "active"
    projtile.revert_mlp(root)
    assert projtile.fork_key_state(root) == "off"


def test_a_half_converted_model_keys_active():
    root = pfx.tiny_quantized_model(seed=0)
    projtile._convert(root.model.layers[0].mlp.gate_proj)
    assert projtile.fork_key_state(root) == "active"


def test_fork_key_state_is_off_for_a_model_without_dense_mlps():
    """The engine keys every load, so a layout with no gate/up/down (a MoE block,
    no layers at all) must read as off rather than raise."""
    moe = types.SimpleNamespace(gate=object(), switch_mlp=object())
    root = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=moe)])
    )
    assert projtile.fork_key_state(root) == "off"
    assert projtile.fork_key_state(types.SimpleNamespace()) == "off"


# ---- the plans -------------------------------------------------------------------

P = projtile.Plan

# The six fused keys at the plans tuned for them on the M5 Pro, then the alone forms
# decode's two-to-eight-row calls reach.
_TABLE = {
    (5120, (12288, 1024, 1024)): P(128, 8, 80, True),  # q+k+v
    (6144, (5120,)): P(256, 8, 20, True, 2),  # o_proj, out_proj
    (5120, (10240, 6144)): P(128, 8, 80, True),  # in_proj_qkv+in_proj_z
    (5120, (48, 48)): P(128, 8, 2, True, 8),  # in_proj_b+in_proj_a
    (5120, (17408, 17408)): P(128, 4, 100, True),  # gate+up, the dual
    (17408, (5120,)): P(256, 8, 20, True, 2),  # down
    (5120, (12288,)): P(128, 8, 80, True),  # q
    (5120, (1024,)): P(128, 8, 8, True),  # k, v
    (5120, (10240,)): P(128, 4, 80, True),  # in_proj_qkv
    (5120, (6144,)): P(128, 8, 48, True),  # in_proj_z
    (5120, (48,)): P(128, 8, 2, True, 8),  # in_proj_b, in_proj_a: the b+a stack's tile
    (5120, (17408,)): P(256, 4, 68, True),  # gate, up
}
_DUAL = (5120, (17408, 17408))


def test_the_six_fused_plans_match_the_spike_and_the_alone_forms_the_spec():
    assert projtile.PLANS == _TABLE
    for key, plan in _TABLE.items():
        assert projtile.plan_for(key) == plan
        k, ns = key
        n = ns[0] if key == _DUAL else sum(ns)
        info = slab.SlabInfo(n_real=n, k=k, tiles=-(-n // 256))
        assert projtile.check_plan(info, plan, dual=key == _DUAL) is None, key
    assert projtile.plan_for((5120, (248320,))) is None  # lm_head
    assert projtile.plan_for((4096, (256,))) is None
    assert projtile.PLANS[(5120, (48, 48))].label == "TN128/S8/G2/pipe x8"
    assert projtile.PLANS[(5120, (17408,))].label == "TN256/S4/G68/pipe"
    assert P(128, 8, 40, False).label == "TN128/S8/G40/seq"
    # availability() compiles every plan the table holds, and nothing else.
    assert {P(*shape) for shape in projtile.DISTINCT_SHAPES} == set(_TABLE.values())


def test_measured_cores_is_the_m5_pro():
    assert frozenset({20}) == projtile.MEASURED_CORES


def test_mlp_keys_name_the_dual_each_linear_alone_and_down():
    want = [(256, (512, 512)), (256, (512,)), (512, (256,))]
    assert projtile.MLP_KEYS(pfx.tiny_quantized_model(seed=0)) == want
    assert projtile.MLP_KEYS(ptfx.tiny_slab_model(seed=0)) == want
    assert projtile.MLP_KEYS(types.SimpleNamespace()) == []


def test_plan_for_call_needs_planes_and_a_plan(monkeypatch):
    root = ptfx.tiny_slab_model(seed=0)
    gdn = root.model.layers[0].linear_attn
    attention = root.model.layers[1].self_attn
    mlp = root.model.layers[0].mlp
    qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
    gdn_call = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)

    def call(linears):
        launches = projtile.plan_for_call(linears)
        return None if launches is None else tuple(launch.plan for launch in launches)

    # The production table holds none of the tiny model's keys.
    assert call(qkv) is None and call((mlp.gate_proj, mlp.up_proj)) is None

    # A distinct plan per key, so each lookup shows which key it took.
    plans = {key: P(128, 4, i + 1, True) for i, key in enumerate(ptfx.TINY_PLANS)}
    monkeypatch.setattr(projtile, "PLANS", plans)
    # One plan per launch: a stack the call covers whole, the GDN call's two.
    assert call(qkv) == (plans[(256, (2048, 512, 512))],)
    assert call(gdn_call) == (plans[(256, (512, 256))], plans[(256, (4, 4))])
    # Each launch carries its key and its linears, so verify() never works them out again.
    launches = projtile.plan_for_call(gdn_call)
    assert launches is not None
    assert [launch.key for launch in launches] == [(256, (512, 256)), (256, (4, 4))]
    assert [tuple(map(id, launch.members)) for launch in launches] == [
        tuple(map(id, gdn_call[:2])),
        tuple(map(id, gdn_call[2:])),
    ]
    # gate+up is the dual's key; every lone linear is keyed by its own width.
    assert call((mlp.gate_proj, mlp.up_proj)) == (plans[(256, (512, 512))],)
    assert call((mlp.down_proj,)) == (plans[(512, (256,))],)
    assert call((mlp.up_proj,)) == (plans[(256, (512,))],)
    assert call((attention.q_proj,)) == (plans[(256, (2048,))],)
    assert call((attention.v_proj,)) == (plans[(256, (512,))],)
    assert call((attention.o_proj,)) == (plans[(1024, (256,))],)
    assert call((gdn.in_proj_z,)) == (plans[(256, (256,))],)
    assert call((gdn.in_proj_a,)) == (plans[(256, (4,))],)
    # Part of a stack, a mix of forms, three slab-only linears: no launch takes them.
    assert call(gdn_call[:3]) is None
    assert call(qkv[:2]) is None
    assert call((mlp.gate_proj, attention.q_proj)) is None
    assert call((mlp.gate_proj, mlp.up_proj, mlp.down_proj)) is None
    # A linear without planes: the head, and the attention and GDN linears at scope
    # mlp, which never copies them.
    assert call((root.lm_head,)) is None
    narrow = ptfx.tiny_slab_model(seed=0, scope="mlp")
    narrow_gdn = narrow.model.layers[0].linear_attn
    assert call((narrow.model.layers[1].self_attn.q_proj,)) is None
    assert call((narrow_gdn.in_proj_qkv, narrow_gdn.in_proj_z)) is None
    assert call((narrow_gdn.in_proj_b,)) is None
    mlp_narrow = narrow.model.layers[2].mlp
    assert call((mlp_narrow.gate_proj, mlp_narrow.up_proj)) == (plans[(256, (512, 512))],)
    plain = pfx.tiny_quantized_model(seed=0).model.layers[0].mlp
    assert call((plain.gate_proj, plain.up_proj)) is None
    # A key the table lacks takes the whole call off the tile, the GDN pair's too.
    del plans[(256, (512, 512))], plans[(256, (4, 4))]
    assert call((mlp.gate_proj, mlp.up_proj)) is None
    assert call(gdn_call) is None
    assert call((mlp.gate_proj,)) == (plans[(256, (512,))],)


_INFO = slab.SlabInfo(n_real=4160, k=2048, tiles=17)


@pytest.mark.parametrize(
    ("info", "plan", "dual", "words"),
    [
        (slab.SlabInfo(512, 2112, 2), P(128, 8, 4, True), False, "K=2112"),
        (_INFO, P(512, 8, 4, True), False, "N=4352"),
        (_INFO, P(64, 8, 4, True), False, "TN=64"),
        (_INFO, P(128, 2, 4, True), False, "S=2"),
        (_INFO, P(256, 16, 4, True), False, "S=16"),
        (_INFO, P(128, 8, 0, True), False, "GROUPS=0"),
        (_INFO, P(128, 8, 4, True, 0), False, "SPLITS=0"),
        (_INFO, P(128, 8, 4, True, 9), False, "SPLITS=9"),
        (_INFO, P(128, 8, 4, True, 2), True, "dual"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_check_plan_refuses_what_the_kernel_cannot_take(info, plan, dual, words):
    reason = projtile.check_plan(info, plan, dual=dual)
    assert reason is not None and words in reason


def test_check_plan_takes_every_split_up_to_k_over_256():
    for splits in range(1, 9):
        assert projtile.check_plan(_INFO, P(128, 8, 4, True, splits)) is None
    assert projtile.check_plan(_INFO, P(256, 4, 1, False), dual=True) is None


# ---- the kernel ------------------------------------------------------------------


def _tile_sources() -> tuple[str, str, str]:
    body = int8prefill._kernel_text("projection_tile.metal")
    return body, int8prefill._kernel_text("projection_tile.h"), body.split("// REDUCE\n")[-1]


def test_the_kernel_sources_split_at_reduce_and_carry_three_barriers():
    """The barriers are the prologue fill and the closing barrier per tile in the
    body, and the refill in FINISH_GROUP; a missing prologue or closing barrier is
    invisible to every output test, so their presence is pinned here."""
    body, header, reduce_body = _tile_sources()
    assert body.count("// REDUCE") == 1
    tile_body = body.split("// REDUCE\n")[0]
    assert tile_body.count("threadgroup_barrier(") == 2
    assert reduce_body.count("threadgroup_barrier(") == 0
    assert header.count("threadgroup_barrier(") == 1
    assert header.index("#define FINISH_GROUP") < header.index("threadgroup_barrier(")
    # MSL 4.1 rejects a cooperative tensor's decltype that keeps its address space.
    for text in (tile_body, header):
        assert text.count("decltype(") == text.count("remove_addrspace_t<decltype(")
    assert tile_body.count("decltype(") == 3
    assert "matmul2d_descriptor::mode::multiply)" in tile_body
    assert reduce_body.lstrip().startswith("#pragma clang fp reassociate(off)")


def test_pad8_appends_zero_rows_and_keeps_eight():
    x = pfx.activations(8, 256, seed=1)
    assert projtile.pad8(x) is x
    zeros = mx.zeros((8, 256), dtype=mx.bfloat16)
    for t in range(1, 8):
        got = projtile.pad8(x[:t])
        assert got.shape == (8, 256) and got.dtype == mx.bfloat16
        assert _same(got[:t], x[:t]) and _same(got[t:], zeros[t:])
    for bad in (x[:0], mx.concatenate([x, x[:1]]), x.reshape(1, 8, 256)):
        with pytest.raises(ValueError):
            projtile.pad8(bad)


def _capture_tile_launches(monkeypatch) -> list[dict]:
    """Every tile launch as a dict of what it was handed; each returns zeros of its
    shapes and nothing compiles."""
    launches: list[dict] = []

    def kernel(name, input_names, output_names, source, header, **options):
        def launch(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
            launches.append(
                {
                    "name": name,
                    "input_names": input_names,
                    "output_names": output_names,
                    "source": source,
                    "header": header,
                    "options": options,
                    "inputs": inputs,
                    "template": dict(template),
                    "grid": grid,
                    "threadgroup": threadgroup,
                    "shapes": output_shapes,
                    "dtypes": output_dtypes,
                }
            )
            return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes, strict=True)]

        return launch

    monkeypatch.setattr(projtile, "_kernel", kernel)
    return launches


def test_run_launches_the_plans_grid_and_the_split_reduce(monkeypatch):
    """The kernel object per (DUAL, SPLITOUT) and its header, the template, the
    persistent grid and the split's fixed-order reduce; nothing compiles here."""
    n, k, n_pad = 4160, 2048, 4352
    codes, scales, biases, info = slab.repack(*pfx.rand_linear(n, k, seed=1))
    up = slab.repack(*pfx.rand_linear(n, k, seed=3))[:3]
    x8 = pfx.activations(8, k, seed=2)
    body, header, reduce_body = _tile_sources()
    tile_body = body.split("// REDUCE\n")[0]
    common, nax_h = int8prefill._kernel_text("common.h"), int8prefill._kernel_text("nax.h")
    safe = {"compile_options": {"math_mode": "safe"}}
    launches = _capture_tile_launches(monkeypatch)

    y = projtile.run(x8, codes, scales, biases, info, P(128, 8, 80, True))
    assert y.shape == (8, n_pad) and y.dtype == mx.bfloat16
    (one,) = launches
    assert one["name"] == "sous_proj_tile_d0_s0"
    assert one["header"] == common + "#define DUAL 0\n#define SPLITOUT 0\n" + nax_h + header
    assert one["source"] == tile_body and one["options"] == safe
    assert one["input_names"] == ["x", "w0", "sc0", "bi0"] and one["output_names"] == ["y0"]
    assert all(a is b for a, b in zip(one["inputs"], (x8, codes, scales, biases), strict=True))
    assert one["template"] == {
        "N": n_pad, "K": k, "TN": 128, "S": 8, "GROUPS": 80, "PIPE": True, "SPLITS": 1
    }  # fmt: skip
    assert (one["grid"], one["threadgroup"]) == ((80 * 8 * 32, 1, 1), (256, 1, 1))
    assert (one["shapes"], one["dtypes"]) == ([(8, n_pad)], [mx.bfloat16])

    launches.clear()
    y = projtile.run(x8, codes, scales, biases, info, P(256, 8, 20, True, 2))
    assert y.shape == (8, n_pad) and y.dtype == mx.bfloat16
    split, reduce = launches
    assert split["name"] == "sous_proj_tile_d0_s1"
    assert split["header"] == common + "#define DUAL 0\n#define SPLITOUT 1\n" + nax_h + header
    assert split["template"]["SPLITS"] == 2 and split["template"]["TN"] == 256
    assert (split["grid"], split["threadgroup"]) == ((20 * 8 * 32, 2, 1), (256, 1, 1))
    assert (split["shapes"], split["dtypes"]) == ([(2, 8, n_pad)], [mx.float32])
    assert reduce["name"] == "sous_proj_tile_reduce"
    # The reduce uses no tensor ops, so its header leaves nax.h out.
    assert reduce["header"] == common and reduce["source"] == reduce_body
    assert reduce["options"] == safe and reduce["inputs"][0].shape == (2, 8, n_pad)
    assert reduce["template"] == {"TOTAL": 8 * n_pad, "SPLITS": 2}
    assert (reduce["grid"], reduce["threadgroup"]) == ((8 * n_pad, 1, 1), (256, 1, 1))
    assert (reduce["shapes"], reduce["dtypes"]) == ([(8, n_pad)], [mx.bfloat16])

    launches.clear()
    y0, y1 = projtile.run_dual(x8, (codes, scales, biases), up, info, P(128, 4, 100, True))
    assert y0.shape == y1.shape == (8, n_pad)
    (dual,) = launches
    assert dual["name"] == "sous_proj_tile_d1_s0"
    assert dual["header"] == common + "#define DUAL 1\n#define SPLITOUT 0\n" + nax_h + header
    assert dual["input_names"] == ["x", "w0", "sc0", "bi0", "w1", "sc1", "bi1"]
    assert dual["output_names"] == ["y0", "y1"]
    assert all(a is b for a, b in zip(dual["inputs"][4:], up, strict=True))
    assert (dual["grid"], dual["threadgroup"]) == ((100 * 4 * 32, 1, 1), (128, 1, 1))
    assert dual["shapes"] == [(8, n_pad), (8, n_pad)]


def test_the_reduce_grid_rounds_up_to_whole_threadgroups(monkeypatch):
    codes, scales, biases, info = slab.repack(*pfx.rand_linear(96, 1024, seed=1))
    launches = _capture_tile_launches(monkeypatch)
    projtile.run(
        pfx.activations(8, 1024, seed=2), codes, scales, biases, info, P(128, 8, 2, True, 4)
    )
    # 8 rows of one 256-column tile: 2048 elements, eight threadgroups of 256.
    assert launches[-1]["grid"] == (2048, 1, 1)


def test_run_and_run_dual_refuse_what_the_kernel_cannot_take(monkeypatch):
    """Refused before anything launches, so a caller's bug never reaches the GPU."""
    codes, scales, biases, info = slab.repack(*pfx.rand_linear(512, 1024, seed=1))
    planes = (codes, scales, biases)
    x8 = pfx.activations(8, 1024, seed=2)
    launches = _capture_tile_launches(monkeypatch)
    with pytest.raises(ValueError, match="SPLITS=5"):
        projtile.run(x8, *planes, info, P(128, 8, 4, True, 5))
    with pytest.raises(ValueError, match=r"\[8, 1024\] bf16"):
        projtile.run(x8[:5], *planes, info, P(128, 8, 4, True))
    with pytest.raises(ValueError, match=r"\[8, 1024\] bf16"):
        projtile.run(x8.astype(mx.float16), *planes, info, P(128, 8, 4, True))
    with pytest.raises(ValueError, match="planes"):
        projtile.run(x8, codes[:-8], scales, biases, info, P(128, 8, 4, True))
    with pytest.raises(ValueError, match="planes"):
        projtile.run(x8, codes, scales, biases[:-1], info, P(128, 8, 4, True))
    with pytest.raises(ValueError, match="dual"):
        projtile.run_dual(x8, planes, planes, info, P(128, 8, 4, True, 2))
    with pytest.raises(ValueError, match="planes"):
        projtile.run_dual(x8, planes, (codes, scales[:-1], biases), info, P(128, 8, 4, True))
    assert launches == []


def test_the_seams_are_the_real_launches():
    assert projtile.tile is projtile.run and projtile.tile_dual is projtile.run_dual


def test_the_stand_ins_keep_the_kernels_contract():
    """[8, n_pad] bf16 with zero columns past N, and the kernel's refusals."""
    codes, scales, biases, info = slab.repack(*pfx.rand_linear(300, 1024, seed=1))
    x8 = pfx.activations(8, 1024, seed=2)
    y = ptfx.stand_in_tile(x8, codes, scales, biases, info, P(128, 8, 4, True))
    assert y.shape == (8, 512) and y.dtype == mx.bfloat16
    assert not mx.any(y[:, 300:]).item() and mx.any(y[:, :300]).item()
    with pytest.raises(ValueError):
        ptfx.stand_in_tile(x8[:4], codes, scales, biases, info, P(128, 8, 4, True))
    with pytest.raises(ValueError):
        ptfx.stand_in_tile_dual(
            x8, (codes, scales, biases), (codes, scales, biases), info, P(128, 8, 4, True, 2)
        )


# ---- routing: the flag, the window, the round flag and the launches ---------------


@pytest.mark.parametrize("active", [0, 1], ids=["off", "active"])
@pytest.mark.parametrize("opened", [False, True], ids=["closed", "open"])
@pytest.mark.parametrize("round_flag", [0, 1], ids=["round-0", "round-1"])
def test_serves_needs_the_flag_the_window_and_the_round_flag(
    monkeypatch, active, opened, round_flag
):
    """With all three set, hook 1 serves a run of any rows (a one-row remainder of
    a cut verify included) and hook 2 a call of two to eight; else neither."""
    ptfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(projtile, "_active", active)
    monkeypatch.setattr(projtile, "_window", projtile.Window(serial=0) if opened else None)
    monkeypatch.setattr(projtile, "_round_flag", round_flag)
    on = bool(active and opened and round_flag)
    for rows in (1, 2, 8, 9):
        assert projtile.serves(rows, hook=1) is on, rows
        assert projtile.serves(rows, hook=2) is (on and 2 <= rows <= 8), rows


def test_open_window_resets_the_round_flag_and_close_window_never_raises(monkeypatch):
    ptfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(projtile, "_active", 1)
    projtile.close_window()  # nothing open
    assert projtile._window is None
    projtile.set_round_flag(0)
    projtile.open_window()
    first = projtile._window
    assert first is not None and projtile._round_flag == 1 and projtile.serves(5, hook=1)
    # The per-round lever closes the tile and leaves the window open.
    projtile.set_round_flag(0)
    assert projtile._window is first and not projtile.serves(5, hook=1)
    projtile.close_window()
    projtile.close_window()
    assert projtile._window is None and not projtile.serves(5, hook=1)
    # The next decode's window is a new one, and its rounds serve again.
    projtile.open_window()
    second = projtile._window
    assert second is not None and second.serial > first.serial
    assert projtile._round_flag == 1 and projtile.serves(5, hook=1)


def test_clear_drops_the_flag_the_window_the_round_flag_and_the_scope(monkeypatch):
    ptfx.guard_tile_state(monkeypatch)
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(projtile, "_scope", "all")
    projtile.open_window()
    projtile.clear()
    state = (projtile._active, projtile._window, projtile._round_flag, projtile._scope)
    assert state == (0, None, 0, None)


def _plans(linears) -> tuple:
    """plan_for_call's launches for a call the test expects the tile to serve."""
    launches = projtile.plan_for_call(linears)
    assert launches is not None
    return launches


def _distinct_plans(monkeypatch) -> dict:
    """A distinct plan per tiny key, so each launch shows which key it took."""
    plans = {key: P(128, 4, i + 1, True) for i, key in enumerate(ptfx.TINY_PLANS)}
    monkeypatch.setattr(projtile, "PLANS", plans)
    return plans


def _recorded_launches(monkeypatch) -> list[tuple]:
    """The stand-ins behind both tile seams, each launch recorded as (seam, the
    SlabInfo it was handed, its plan, the codes it read)."""
    launches: list[tuple] = []

    def tile(x8, codes, scales, biases, info, plan):
        launches.append(("tile", info, plan, codes))
        return ptfx.stand_in_tile(x8, codes, scales, biases, info, plan)

    def tile_dual(x8, gate, up, info, plan):
        launches.append(("dual", info, plan, gate[0]))
        return ptfx.stand_in_tile_dual(x8, gate, up, info, plan)

    monkeypatch.setattr(projtile, "tile", tile)
    monkeypatch.setattr(projtile, "tile_dual", tile_dual)
    return launches


def _stack_columns(stack, x, plan) -> Any:
    """The stand-in's [8, n_pad] over a whole stack for x's rows: what each member's
    slice is cut from."""
    k = x.shape[-1]
    x8 = projtile.pad8(x.reshape(x.size // k, k))
    return ptfx.stand_in_tile(x8, stack.codes, stack.scales, stack.biases, stack.info, plan)


def _alone(module, x) -> Any:
    """The stand-in over the module's own arrays (or its own slab planes)."""
    return pfx.stand_in_mma(x, [projkernel._weights(module)])[0]


def _close(got, want) -> bool:
    """Within a few bf16 ulps: a stand-in over a wider stack is a matmul of another
    width, which need not sum a column in the same order."""
    diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item()
    return diff <= 1e-2 * mx.max(mx.abs(want.astype(mx.float32))).item()


def test_verify_slices_a_stack_back_to_its_members(monkeypatch):
    plans = _distinct_plans(monkeypatch)
    launches = _recorded_launches(monkeypatch)
    root = ptfx.tiny_slab_model(seed=0)
    attention = root.model.layers[1].self_attn
    qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
    stack = projtile.stack_of(attention.q_proj)
    assert stack is not None
    x = pfx.activations(5, 256, seed=1).reshape(1, 5, 256)
    outs = projtile.verify(qkv, x, _plans(qkv))
    plan = plans[(256, (2048, 512, 512))]
    assert [(seam, info, p) for seam, info, p, _ in launches] == [("tile", stack.info, plan)]
    assert launches[0][3] is stack.codes
    assert isinstance(outs, tuple)
    assert [o.shape for o in outs] == [(1, 5, 2048), (1, 5, 512), (1, 5, 512)]
    y8 = _stack_columns(stack, x, plan)
    for out, member, (start, stop) in zip(outs, qkv, stack.slices, strict=True):
        assert _same(out, y8[:5, start:stop].reshape(1, 5, stop - start))
        assert _close(out, _alone(member, x))


def test_the_gdn_call_runs_as_two_launches(monkeypatch):
    plans = _distinct_plans(monkeypatch)
    launches = _recorded_launches(monkeypatch)
    root = ptfx.tiny_slab_model(seed=0)
    gdn = root.model.layers[2].linear_attn
    call = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
    qkvz, ba = projtile.stack_of(gdn.in_proj_qkv), projtile.stack_of(gdn.in_proj_b)
    assert qkvz is not None and ba is not None
    x = pfx.activations(3, 256, seed=2).reshape(1, 3, 256)
    outs = projtile.verify(call, x, _plans(call))
    first, second = plans[(256, (512, 256))], plans[(256, (4, 4))]
    assert [(seam, info, p) for seam, info, p, _ in launches] == [
        ("tile", qkvz.info, first),
        ("tile", ba.info, second),
    ]
    assert [o.shape for o in outs] == [(1, 3, 512), (1, 3, 256), (1, 3, 4), (1, 3, 4)]
    pieces = [(qkvz, first, 0), (qkvz, first, 1), (ba, second, 0), (ba, second, 1)]
    for out, member, (stack, plan, index) in zip(outs, call, pieces, strict=True):
        start, stop = stack.slices[index]
        want = _stack_columns(stack, x, plan)[:3, start:stop].reshape(1, 3, stop - start)
        assert _same(out, want)
        assert _close(out, _alone(member, x))


def test_verify_runs_gate_and_up_as_one_dual_launch_and_keeps_a_batch(monkeypatch):
    """Two slab-only linears of one shape share one dual launch; rows are the
    product of the leading dimensions, so a batch of two runs is one launch too."""
    plans = _distinct_plans(monkeypatch)
    launches = _recorded_launches(monkeypatch)
    mlp = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp
    pair = (mlp.gate_proj, mlp.up_proj)
    x = pfx.activations(6, 256, seed=3).reshape(2, 3, 256)
    gate, up = projtile.verify(pair, x, _plans(pair))
    info = slab.slab_info(mlp.gate_proj)
    assert [(seam, i, p) for seam, i, p, _ in launches] == [
        ("dual", info, plans[(256, (512, 512))])
    ]
    for out, member in zip((gate, up), pair, strict=True):
        assert out.shape == (2, 3, 512)
        assert _same(out, _alone(member, x))


def test_a_cut_of_the_whole_output_is_the_launch_itself(monkeypatch):
    """Eight 2-D rows through a linear whose width fills its tiles take the
    launch's output as it is, adding no slice and no reshape to the graph; fewer
    rows, or a leading batch dimension, are still cut and shaped."""
    _distinct_plans(monkeypatch)
    returned = []

    def tile(x8, codes, scales, biases, info, plan):
        returned.append(ptfx.stand_in_tile(x8, codes, scales, biases, info, plan))
        return returned[-1]

    monkeypatch.setattr(projtile, "tile", tile)
    up = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp.up_proj
    x = pfx.activations(8, 256, seed=6)
    (out,) = projtile.verify((up,), x, _plans((up,)))
    assert out is returned[-1]
    (out,) = projtile.verify((up,), x[:5], _plans((up,)))
    assert out.shape == (5, 512) and _same(out, returned[-1][:5])
    (out,) = projtile.verify((up,), x.reshape(1, 8, 256), _plans((up,)))
    assert out.shape == (1, 8, 512) and _same(out[0], returned[-1])


def test_plain_on_a_stack_member_launches_its_own_tiles(monkeypatch):
    """q, k and v each read only their own tiles of the q+k+v planes at their own
    alone plan; in_proj_b and in_proj_a share one tile, so each runs the b+a stack
    and takes its slice."""
    plans = _distinct_plans(monkeypatch)
    launches = _recorded_launches(monkeypatch)
    root = ptfx.tiny_slab_model(seed=0)
    attention = root.model.layers[1].self_attn
    stack = projtile.stack_of(attention.q_proj)
    assert stack is not None
    groups = stack.info.groups
    x = pfx.activations(3, 256, seed=4).reshape(1, 3, 256)
    for member, (start, stop) in zip(
        (attention.q_proj, attention.k_proj, attention.v_proj), stack.slices, strict=True
    ):
        launches.clear()
        out = projtile.plain(member, x, _plans((member,)))
        t0, t1 = start // slab.TILE, stop // slab.TILE
        ((seam, info, plan, codes),) = launches
        assert seam == "tile" and plan == plans[(256, (stop - start,))]
        assert info == slab.SlabInfo(n_real=stop - start, k=256, tiles=t1 - t0)
        words = groups * slab.TILE * slab.UNIT_BYTES // 4
        assert _same(codes, stack.codes[t0 * words : t1 * words])
        assert out.shape == (1, 3, stop - start)
        assert _same(out, _alone(member, x))

    gdn = root.model.layers[0].linear_attn
    ba = projtile.stack_of(gdn.in_proj_b)
    assert ba is not None
    for member, (start, stop) in zip((gdn.in_proj_b, gdn.in_proj_a), ba.slices, strict=True):
        launches.clear()
        out = projtile.plain(member, x, _plans((member,)))
        ((seam, info, plan, codes),) = launches
        assert (seam, info, plan) == ("tile", ba.info, plans[(256, (4,))])
        assert codes is ba.codes
        assert _same(out, _stack_columns(ba, x, plan)[:3, start:stop].reshape(1, 3, 4))
        assert _close(out, _alone(member, x))

    # A slab-only linear reads its own planes whole.
    launches.clear()
    up = root.model.layers[0].mlp.up_proj
    out = projtile.plain(up, x, _plans((up,)))
    assert [(seam, info, p) for seam, info, p, _ in launches] == [
        ("tile", slab.slab_info(up), plans[(256, (512,))])
    ]
    assert launches[0][3] is up.weight
    assert _same(out, _alone(up, x))


def test_a_stack_run_must_be_its_members_in_order(monkeypatch):
    """Every member carries its index in its stack, and a run of a call is the
    stack's launch only when it is the stack's members in order: k and v share a
    width, so widths alone would take q+v+k for q+k+v and hand k v's columns."""
    _distinct_plans(monkeypatch)
    root = ptfx.tiny_slab_model(seed=0)
    attention = root.model.layers[1].self_attn
    q, k, v = attention.q_proj, attention.k_proj, attention.v_proj
    assert [getattr(m, projtile.INDEX_ATTR) for m in (q, k, v)] == [0, 1, 2]
    assert projtile.plan_for_call((q, k, v)) is not None
    assert projtile.plan_for_call((q, v, k)) is None
    assert projtile.plan_for_call((q, q, q)) is None
    later = root.model.layers[3].self_attn
    assert projtile.plan_for_call((q, later.k_proj, later.v_proj)) is None
    x = pfx.activations(2, 256, seed=5).reshape(1, 2, 256)
    with pytest.raises(ValueError):
        projtile.verify((q, v, k), x, _plans((q, k, v)))
    # Launches of another call, or of one linear more or fewer, are refused too.
    with pytest.raises(ValueError):
        projtile.verify((q, k, v), x, _plans((later.q_proj, later.k_proj, later.v_proj)))
    with pytest.raises(ValueError):
        projtile.verify((q, k), x, _plans((q, k, v)))
    with pytest.raises(ValueError):
        projtile.plain(k, x, _plans((v,)))
    projtile.drop_stacks(root)
    assert all(projtile.INDEX_ATTR not in vars(m) for m in (q, k, v))


# ---- the kernel on the tensor units ----------------------------------------------

TILE_K = 2048
# (codes, scales, biases), or mlx's (weight, scales, biases).
_Planes = tuple[Any, Any, Any]


def _tile_linear(n: int, seed: int) -> tuple[_Planes, slab.SlabInfo, _Planes]:
    """A random projection as slab planes, its SlabInfo, and mlx's arrays."""
    linear = pfx.rand_linear(n, TILE_K, seed)
    codes, scales, biases, info = slab.repack(*linear)
    return (codes, scales, biases), info, linear


def _shape_id(shape) -> str:
    return P(*shape).label.replace("/", "-").replace(" ", "-")


@tile_nax
@pytest.mark.parametrize("shape", projtile.DISTINCT_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("n", [512, 96, 4160], ids=lambda n: f"n{n}")
def test_every_row_is_independent_of_the_row_count(shape, n):
    """Every row at T = 1..8 equals the same row run alone; rows past T and columns
    past N come out zero; a NaN in one row reaches that row and no other."""
    plan = P(*shape)
    planes, info, _ = _tile_linear(n, seed=n)
    x = pfx.activations(8, TILE_K, seed=1)
    alone = [projtile.run(projtile.pad8(x[r : r + 1]), *planes, info, plan)[:1] for r in range(8)]
    mx.eval(alone)
    for t in range(1, 9):
        y = projtile.run(projtile.pad8(x[:t]), *planes, info, plan)
        for r in range(t):
            assert _same(y[r : r + 1], alone[r]), (t, r)
        assert not mx.any(y[t:]).item()
        assert not mx.any(y[:, n:]).item()
    nan_row = mx.full((1, TILE_K), float("nan"), dtype=mx.bfloat16)
    y = projtile.run(mx.concatenate([x[:3], nan_row, x[4:]]), *planes, info, plan)
    for r in range(8):
        if r != 3:
            assert _same(y[r : r + 1], alone[r]), r
    assert mx.all(mx.isnan(y[3, :n])).item()


@tile_nax
@pytest.mark.parametrize("shape", [s for s in projtile.DISTINCT_SHAPES if s[4] > 1], ids=_shape_id)
def test_the_split_reduce_adds_partials_in_order(shape):
    plan = P(*shape)
    planes, info, _ = _tile_linear(4160, seed=5)
    x8 = pfx.activations(8, TILE_K, seed=6)
    part = projtile._partials(x8, *planes, info, plan)
    assert part.shape == (plan.splits, 8, info.n_pad) and part.dtype == mx.float32
    total = mx.zeros((8, info.n_pad), dtype=mx.float32)
    for s in range(plan.splits):
        total = total + part[s]
    assert _same(projtile.run(x8, *planes, info, plan), total.astype(mx.bfloat16))


@tile_nax
@pytest.mark.parametrize("shape", [s for s in projtile.DISTINCT_SHAPES if s[4] == 1], ids=_shape_id)
@pytest.mark.parametrize("n", [512, 4160], ids=lambda n: f"n{n}")
def test_the_dual_launch_equals_two_single_launches(shape, n):
    plan = P(*shape)
    gate, info, _ = _tile_linear(n, seed=11)
    up, _, _ = _tile_linear(n, seed=12)
    x8 = pfx.activations(8, TILE_K, seed=13)
    y0, y1 = projtile.run_dual(x8, gate, up, info, plan)
    assert _same(y0, projtile.run(x8, *gate, info, plan))
    assert _same(y1, projtile.run(x8, *up, info, plan))


def _f64(a) -> np.ndarray:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _rel_rms(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want**2)))


@tile_nax
@pytest.mark.parametrize("shape", projtile.DISTINCT_SHAPES, ids=_shape_id)
def test_the_tile_is_at_the_bf16_floor_against_an_fp32_reference(shape):
    """Against the product of the activations and the weights dequantised with
    their scales and biases upcast to fp32 (taken in float64), the tile's relative
    RMS is within TILE_FLOOR_FACTOR of the floor, that reference rounded to bf16:
    one bf16 rounding, and an fp32 sum that adds nothing visible to it."""
    plan, n = P(*shape), 4160
    gate, info, gate_linear = _tile_linear(n, seed=21)
    up, _, up_linear = _tile_linear(n, seed=22)
    x8 = pfx.activations(8, TILE_K, seed=23)
    outs = [(projtile.run(x8, *gate, info, plan), gate_linear)]
    if plan.splits == 1:
        duals = projtile.run_dual(x8, gate, up, info, plan)
        outs += zip(duals, (gate_linear, up_linear), strict=True)
    for y, (w, s, b) in outs:
        dense = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=4)
        ref = _f64(x8) @ _f64(dense).T
        floor = _rel_rms(_f64(mx.array(ref.astype(np.float32)).astype(mx.bfloat16)), ref)
        assert _rel_rms(_f64(y[:, :n]), ref) <= projtile.TILE_FLOOR_FACTOR * floor


# ---- enable(): the guards, the memory rule, the warm-up and the probe --------------

G = 2**30
_BUILD_ERROR = "Unable to build metal library\nprogram_source:3:1: error: boom"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _tile_log():
    """Every record the tile's logger emits inside the block, INFO included; they
    still propagate, so a test's own caplog sees them too."""
    logger = logging.getLogger("sous.engine.projtile")
    handler, level = _Records(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _enable(root, ready, **overrides) -> dict:
    """enable() as an active load calls it, with what a test overrides: a dflash
    drafter, the automatic budget, 64 GiB of headroom and an 8 GiB reserve."""
    kwargs: dict[str, Any] = {
        "config_value": True,
        **ready,
        "drafter_kind": "dflash",
        "cache_budget": None,
        "headroom_bytes": 64 * G,
        "reserve_bytes": 8 * G,
        **overrides,
    }
    return projtile.enable(root, **kwargs)


def _gated(root, ready, **overrides) -> tuple[dict, list[str], list[logging.LogRecord]]:
    """enable() with every warning and the tile's log records caught."""
    with warnings.catch_warnings(record=True) as caught, _tile_log() as records:
        warnings.simplefilter("always")
        status = _enable(root, ready, **overrides)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return status, messages, records


def _infos(records) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == logging.INFO]


def _planned(key) -> bool:
    return projtile.plan_for(key) is not None


def _marked(root) -> tuple[int, int, int]:
    """(slab-only MLP linears, linears carrying a stack, linears carrying an index)."""
    linears = [m for _, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]
    return (
        sum(slab.SLAB_ATTR in vars(m) for m in linears),
        sum(projtile.STACK_ATTR in vars(m) for m in linears),
        sum(projtile.INDEX_ATTR in vars(m) for m in linears),
    )


# The tiny model: 12 MLP linears and 18 stacked attention and GDN linears.
_CONVERTED = (12, 18, 18)
_MLP_ONLY = (12, 0, 0)
_UNTOUCHED = (0, 0, 0)


def _refusal(root, ready, *, warned: bool, **overrides) -> dict:
    """enable() expected to refuse. A load the tile does not apply to is one INFO
    line and no warning; one where it should have run and did not is exactly one
    warning, on one line, in the tile's words. Either way the routing state is
    clear and nothing of the tile is left on the model."""
    status, messages, records = _gated(root, ready, **overrides)
    infos = _infos(records)
    assert status["state"] == "unavailable", status
    assert set(status) == {
        "state",
        "scope",
        "reason",
        "copy_gb",
        "probe_seconds",
        "budget_left_gib",
    }
    assert (status["scope"], status["copy_gb"], status["budget_left_gib"]) == (None, None, None)
    if warned:
        assert messages == [
            f"sous: projection tile unavailable ({status['reason']}); "
            "verify runs the projection kernel"
        ]
        assert not any(m.startswith("projection tile unavailable") for m in infos), infos
    else:
        assert messages == []
        assert infos == [f"projection tile unavailable: {status['reason']}"]
    assert (projtile._active, projtile._scope, projtile._window) == (0, None, None)
    assert _marked(root) == _UNTOUCHED
    assert projtile.fork_key_state(root) == "off"
    return status


def test_enable_off_returns_before_any_guard(monkeypatch):
    ready = ptfx.tile_ready(monkeypatch)
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(projtile, "_scope", "all")
    projtile.open_window()

    def untouchable(*args, **kwargs):
        raise AssertionError("a guard ran while the tile is off")

    for name in (
        "_kernel_reason",
        "_drafter_reason",
        "_int8_reason",
        "_cores_reason",
        "_model_reason",
        "_pins_reason",
        "availability",
        "_compile_real_shapes",
        "copy_decision",
        "convert_mlp",
        "build_stacks",
        "warm_up",
        "probe",
    ):
        monkeypatch.setattr(projtile, name, untouchable)
    root = ptfx.ready_model()
    status, messages, records = _gated(root, ready, config_value=False, drafter_kind=None)
    assert status == {
        "state": "off",
        "scope": None,
        "reason": None,
        "copy_gb": None,
        "probe_seconds": None,
        "budget_left_gib": None,
    }
    assert messages == [] and records == []
    state = (projtile._active, projtile._scope, projtile._window, projtile._round_flag)
    assert state == (0, None, None, 0)
    assert _marked(root) == _UNTOUCHED


_GUARD_REASONS = [
    "projection kernel off",
    "macOS 15.5 < 26.2",
    "no drafter",
    "int8 prefill off",
    "GPU core count 16 not measured (measured: 20)",
    "no tuned plan for (256, (512, 512))",
    "mlx 0.99.0 not validated",
    "the projection tile kernels failed: program_source:3:1: error: boom",
    "real shapes: program_source:3:1: error: boom",
    "floor: model.layers.1.mlp.down_proj is 0.01 from its fp32 reference",
]
# The kernel, the attention tile, the drafter, int8 prefill, the core count and the
# model only say the tile does not apply to this load. Pin drift, kernels this OS
# rejects (at the probe's shapes or the model's) and a failed probe say it should
# have run and did not.
_GUARD_WARNS = [False] * 6 + [True] * 4


@pytest.mark.parametrize("first", range(len(_GUARD_REASONS)))
def test_enable_names_the_first_failing_guard_in_order(monkeypatch, first):
    """Guard `first` and every guard after it fail; the reason is guard `first`'s."""
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    overrides: dict = {}
    device = str(mx.device_info()["device_name"])

    def fails(root):
        raise RuntimeError(_BUILD_ERROR)

    without_dual = {k: p for k, p in projtile.PLANS.items() if k != (256, (512, 512))}
    breakers = [
        lambda: overrides.update(kernel_status={"state": "off", "reason": None}),
        lambda: overrides.update(
            tile_status={"state": "unavailable", "reason": _GUARD_REASONS[1], "splits": None}
        ),
        lambda: overrides.update(drafter_kind=None),
        lambda: overrides.update(int8_status={"state": "off", "reason": None, "routed": 0}),
        lambda: monkeypatch.setattr(
            gpucores, "read", lambda: gpucores.GPUCores(16, device, None, "iokit")
        ),
        lambda: monkeypatch.setattr(projtile, "PLANS", without_dual),
        lambda: monkeypatch.setattr(projtile, "_pins_reason", lambda: _GUARD_REASONS[6]),
        lambda: monkeypatch.setattr(
            projtile, "availability", lambda: nax.Availability(False, _GUARD_REASONS[7])
        ),
        lambda: monkeypatch.setattr(projtile, "_compile_real_shapes", fails),
        lambda: monkeypatch.setattr(projtile, "probe", lambda root, scope: _GUARD_REASONS[9]),
    ]
    for brk in breakers[first:]:
        brk()
    status = _refusal(root, ready, warned=_GUARD_WARNS[first], **overrides)
    assert status["reason"] == _GUARD_REASONS[first]
    # The cost is reported once the kernels at the model's shapes have started.
    assert (status["probe_seconds"] is None) == (first < 8)


def _untag_int8(root):
    object.__setattr__(root.model.layers[2].mlp.down_proj, int8prefill._TAG, False)


def _untag_kernel(root):
    object.__setattr__(root.model.layers[2].mlp.up_proj, projkernel._TAG, False)


def _fp16_scales(root):
    module = root.model.layers[3].self_attn.k_proj
    module.scales = module.scales.astype(mx.float16)


@pytest.mark.parametrize(
    ("breaker", "reason"),
    [
        (_untag_int8, "projection model.layers.2.mlp.down_proj is not routed by int8 prefill"),
        (
            _untag_kernel,
            "projection model.layers.2.mlp.up_proj is not routed by the projection kernel",
        ),
        (
            _fp16_scales,
            "projection model.layers.3.self_attn.k_proj is not affine 4-bit gs64 with bf16 "
            "scales in mlx's layout",
        ),
    ],
    ids=["int8", "kernel", "eligible"],
)
def test_enable_refuses_a_model_whose_linears_its_paths_cannot_take(monkeypatch, breaker, reason):
    """A slab-only linear that no path routes would reach stock, which raises on
    flat planes; a linear it would copy must be one the kernel decodes. Either
    only says the tile does not apply to this model."""
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    breaker(root)
    assert _refusal(root, ready, warned=False)["reason"] == reason


def test_enable_refuses_a_model_without_a_dense_mlp(monkeypatch):
    ready = ptfx.tile_ready(monkeypatch)
    root = types.SimpleNamespace(model=types.SimpleNamespace(layers=[]))
    with _tile_log() as records:
        status = _enable(root, ready)
    assert (status["state"], status["reason"]) == ("unavailable", "no dense MLP")
    assert _infos(records) == ["projection tile unavailable: no dense MLP"]


def test_enable_refuses_an_exception_in_a_guard_with_its_error_line(monkeypatch, caplog):
    ready = ptfx.tile_ready(monkeypatch)
    error = RuntimeError(_BUILD_ERROR)

    def broken(root):
        raise error

    monkeypatch.setattr(projtile, "_model_reason", broken)
    status = _refusal(ptfx.ready_model(), ready, warned=True)
    assert status["reason"] == "program_source:3:1: error: boom"
    assert status["probe_seconds"] is None
    (record,) = [r for r in caplog.records if r.getMessage() == "projection tile: enabling failed"]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


# (config value, configured budget, headroom, copy, reserve, copies, text); the
# automatic rule keeps 2 GiB of slack beside the reserve.
COPY_CASES = [
    (True, None, 35 * G, 4 * G, 16 * G, False,
     "copy would leave 13.0 GiB of prompt-cache budget (< 14)"),
    (True, None, 44 * G, 4 * G, 16 * G, True, "budget left 22.0 GiB"),
    (True, 12 * G, 35 * G, 4 * G, 16 * G, True,
     "configured budget 12.0 GiB fits beside the copy; budget left 13.0 GiB"),
    (True, 14 * G, 35 * G, 4 * G, 16 * G, False,
     "configured budget 14.0 GiB does not fit beside the copy; budget left 13.0 GiB"),
    (True, 0, 20 * G, 4 * G, 16 * G, False,
     "configured budget 0.0 GiB does not fit beside the copy; budget left -2.0 GiB"),
    (True, None, None, 4 * G, 16 * G, False, "headroom unreadable"),
    (True, 12 * G, None, 4 * G, 16 * G, False, "headroom unreadable"),
    ("mlp", None, 44 * G, 4 * G, 16 * G, False, "scope mlp by config"),
    ("all", None, 20 * G, 4 * G, 16 * G, True, "scope all by config; budget left -2.0 GiB"),
    ("all", 12 * G, None, 4 * G, 16 * G, True, "scope all by config; budget left unknown"),
]  # fmt: skip


def _copy_id(case) -> str:
    config, budget, headroom, *_ = case
    gib = lambda v: "auto" if v is None else f"{v // G}G"  # noqa: E731
    return f"{config}-budget-{gib(budget)}-headroom-{gib(headroom)}"


@pytest.mark.parametrize("case", COPY_CASES, ids=[_copy_id(c) for c in COPY_CASES])
def test_copy_decision_table(case):
    """Bytes in, the copy and why out; never a warning, even where a configured
    budget no longer fits: the user chose it."""
    config, budget, headroom, copy, reserve, copies, text = case
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = projtile.copy_decision(config, budget, headroom, copy, reserve)
    assert got == (copies, text)
    assert caught == []


def test_enable_converts_the_mlp_and_builds_stacks_at_scope_all(monkeypatch):
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    copy = projtile.copy_bytes(root, keys_with_plans=_planned)
    counted = (dict(projkernel.calls), dict(int8prefill.calls))
    status, messages, records = _gated(root, ready, headroom_bytes=40 * G, reserve_bytes=8 * G)
    assert messages == []
    seconds = status["probe_seconds"]
    assert isinstance(seconds, float)
    left = round((40 * G - copy - 8 * G - 2 * G) / G, 2)
    assert status == {
        "state": "active",
        "scope": "all",
        "reason": None,
        "copy_gb": copy / 1e9,
        "probe_seconds": seconds,
        "budget_left_gib": left,
    }
    assert (projtile._active, projtile._scope, projtile._window) == (1, "all", None)
    assert _marked(root) == _CONVERTED
    assert projtile.fork_key_state(root) == "active"
    assert _infos(records) == [
        f"projection tile: scope=all copy={copy / 1e9:.1f} GB budget left {left:.1f} GiB "
        f"probe {seconds:.2f}s"
    ]
    # The counters are as the load found them: a load serves no call.
    assert (dict(projkernel.calls), dict(int8prefill.calls)) == counted


@pytest.mark.parametrize(
    ("config", "headroom", "why"),
    [
        (True, 19 * G, "copy would leave 13.0 GiB of prompt-cache budget (< 14)"),
        ("mlp", 64 * G, "scope mlp by config"),
        (True, None, "headroom unreadable"),
    ],
    ids=["shortfall", "config", "unreadable"],
)
def test_enable_serves_mlp_only_when_the_copy_is_skipped(monkeypatch, config, headroom, why):
    """No copy: the MLP is still converted and served, the attention and GDN
    linears keep the projection kernel, and the one INFO line names why."""
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    status, messages, records = _gated(
        root, ready, config_value=config, headroom_bytes=headroom, reserve_bytes=4 * G
    )
    assert messages == []
    assert (status["state"], status["scope"], status["reason"]) == ("active", "mlp", why)
    assert status["copy_gb"] is None
    left = None if headroom is None else round((headroom - 4 * G - 2 * G) / G, 2)
    assert status["budget_left_gib"] == left
    assert (projtile._active, projtile._scope) == (1, "mlp")
    assert _marked(root) == _MLP_ONLY
    budget = "unknown" if left is None else f"{left:.1f} GiB"
    assert _infos(records) == [
        f"projection tile: scope=mlp copy=0.0 GB budget left {budget} "
        f"probe {status['probe_seconds']:.2f}s ({why})"
    ]


def test_enable_reports_scope_mlp_when_no_stack_key_has_a_plan(monkeypatch):
    """A copy decided but never built (every attention and GDN key outside the
    plan table) serves exactly what scope "mlp" serves, so the status says so,
    with the reason, rather than "all" with nothing copied."""
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    untuned = {k: v for k, v in projtile.PLANS.items() if k not in ptfx.TINY_PLANS}
    mlp_plans = {k: ptfx.TINY_PLAN for k in projtile.MLP_KEYS(root)}
    monkeypatch.setattr(projtile, "PLANS", {**untuned, **mlp_plans})
    groups = projtile.stack_groups(root)
    assert groups and not any(_planned(projtile._key(group)) for group in groups)
    status, messages, records = _gated(
        root, ready, config_value="all", headroom_bytes=40 * G, reserve_bytes=8 * G
    )
    assert messages == []
    why = "no stack key has a plan"
    left = round((40 * G - 8 * G - 2 * G) / G, 2)
    assert status == {
        "state": "active",
        "scope": "mlp",
        "reason": why,
        "copy_gb": None,
        "probe_seconds": status["probe_seconds"],
        "budget_left_gib": left,
    }
    assert (projtile._active, projtile._scope) == (1, "mlp")
    assert _marked(root) == _MLP_ONLY
    assert _infos(records) == [
        f"projection tile: scope=mlp copy=0.0 GB budget left {left:.1f} GiB "
        f"probe {status['probe_seconds']:.2f}s ({why})"
    ]


def _arrays_of(root) -> list[tuple[str, tuple]]:
    return [
        (path, (m.weight, m.scales, m.biases))
        for path, m in root.named_modules()
        if isinstance(m, nn.QuantizedLinear)
    ]


@pytest.mark.parametrize("failure", ["returns", "raises"])
def test_a_failed_probe_reverts_the_conversion_bitwise_and_drops_the_stacks(
    monkeypatch, caplog, failure
):
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    before = _arrays_of(root)
    seen: list = []

    def failing(root, scope):
        seen.append((_marked(root), scope))
        if failure == "raises":
            raise RuntimeError("device lost")
        return "isolation: a NaN in row 3 changed the other rows of model.layers.1.mlp.down_proj"

    monkeypatch.setattr(projtile, "probe", failing)
    status = _refusal(root, ready, warned=True)
    assert seen == [(_CONVERTED, "all")]
    assert isinstance(status["probe_seconds"], float)
    if failure == "raises":
        assert status["reason"] == "device lost"
        (record,) = [
            r for r in caplog.records if r.getMessage() == "projection tile: enabling failed"
        ]
        assert record.exc_info is not None
    after = _arrays_of(root)
    assert [path for path, _ in after] == [path for path, _ in before]
    for (path, got), (_, want) in zip(after, before, strict=True):
        for g, w in zip(got, want, strict=True):
            assert _same(g, w), path


def test_a_revert_that_raises_leaves_the_markers_authoritative(monkeypatch, caplog):
    """The stacks still go; the linears the revert could not reach stay slab-only,
    marked, and keyed so: every reader is marker-driven."""
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    monkeypatch.setattr(projtile, "probe", lambda root, scope: "floor: boom")

    def revert_fails(root):
        raise RuntimeError("revert failed")

    monkeypatch.setattr(projtile, "revert_mlp", revert_fails)
    status, messages, _ = _gated(root, ready)
    assert status["state"] == "unavailable" and status["reason"] == "floor: boom"
    assert len(messages) == 1
    assert _marked(root) == _MLP_ONLY
    assert projtile.fork_key_state(root) == "active"
    assert any("revert" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_a_conversion_that_converts_nothing_refuses_the_load(monkeypatch):
    """A root the conversion finds no MLP under would otherwise go on to serve an
    unconverted model as converted."""
    ready = ptfx.tile_ready(monkeypatch)
    monkeypatch.setattr(projtile, "convert_mlp", lambda root: 0)
    status = _refusal(ptfx.ready_model(), ready, warned=True)
    assert status["reason"] == "the MLP conversion converted no linear"
    assert isinstance(status["probe_seconds"], float)


def test_enable_compiles_the_real_shapes_before_converting(monkeypatch):
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    order: list = []
    convert = projtile._convert

    def recording_convert(module):
        order.append("convert")
        convert(module)

    monkeypatch.setattr(projtile, "_convert", recording_convert)
    monkeypatch.setattr(
        projtile, "_compile_real_shapes", lambda r: order.append(("compile", _marked(r)))
    )
    assert _enable(root, ready)["state"] == "active"
    assert order[0] == ("compile", _UNTOUCHED) and order[1:] == ["convert"] * 12

    # A toolchain that rejects them refuses the load with the model never touched.
    order.clear()
    fresh = ptfx.ready_model(seed=1)

    def fails(root):
        raise RuntimeError(_BUILD_ERROR)

    monkeypatch.setattr(projtile, "_compile_real_shapes", fails)
    status = _refusal(fresh, ready, warned=True)
    assert status["reason"] == "real shapes: program_source:3:1: error: boom"
    assert order == []


def test_compile_real_shapes_runs_every_mlp_path_on_zero_planes(monkeypatch):
    """Before anything is converted: the slab MMA bodies at T = 1..8 over gate+up
    and each linear, the slab one-row body and int8's slab GEMM, at every distinct
    MLP shape, on planes of their own; the counters are left as found."""
    pfx.proj_ready(monkeypatch)
    root = ptfx.ready_model()
    seen: dict[str, list] = {"mma": [], "row": [], "qmm": []}

    def mma(x, weights):
        seen["mma"].append((x.shape[0], tuple(w.slab for w in weights)))
        return pfx.stand_in_mma(x, weights)

    def one_row(x2, w, scales, biases, slab=None):
        seen["row"].append(slab)
        return pfx.stand_in_one_row(x2, w, scales, biases, slab=slab)

    def qmm(qa, sa, ra, w, scales, biases, *, slab=None):
        seen["qmm"].append((qa.shape[0], slab))
        assert not mx.any(w).item() and not mx.any(scales).item()
        return ptfx.slab_reference_qmm(qa, sa, ra, w, scales, biases, slab=slab)

    monkeypatch.setattr(projkernel, "mma", mma)
    monkeypatch.setattr(projkernel, "one_row", one_row)
    monkeypatch.setattr(int8prefill, "qmm", qmm)
    counted = dict(projkernel.calls)
    projtile._compile_real_shapes(root)
    up = slab.SlabInfo(n_real=512, k=256, tiles=2)
    down = slab.SlabInfo(n_real=256, k=512, tiles=1)
    groups = [(up, up), (up,), (down,)]
    assert seen["mma"] == [(t, g) for g in groups for t in range(1, 9)]
    assert seen["row"] == [up, down]
    assert seen["qmm"] == [(int8prefill.SLAB_MIN_ROWS, up), (int8prefill.SLAB_MIN_ROWS, down)]
    assert projkernel.calls == counted
    assert _marked(root) == _UNTOUCHED


def test_no_plane_outlives_its_model(monkeypatch):
    ready = ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    assert _enable(root, ready, config_value="all")["state"] == "active"
    stack = projtile.stack_of(root.model.layers[1].self_attn.q_proj)
    assert stack is not None
    refs = [weakref.ref(stack.codes), weakref.ref(root.model.layers[0].mlp.gate_proj.weight)]
    del stack, root
    gc.collect()
    assert [ref() for ref in refs] == [None, None]


def _probe_ready(monkeypatch, scope: str = "all"):
    """The tiny model converted (and, at scope all, stacked) as enable() leaves it
    before the probe, behind ready hooks and stand-ins."""
    ptfx.tile_ready(monkeypatch)
    root = ptfx.ready_model()
    projtile.convert_mlp(root)
    if scope == "all":
        projtile.build_stacks(root, keys_with_plans=_planned)
    return root


def test_the_probe_passes_the_stand_ins_and_leaves_its_state(monkeypatch):
    for scope in ("all", "mlp"):
        root = _probe_ready(monkeypatch, scope)
        counted = (dict(projkernel.calls), dict(int8prefill.calls))
        projtile.set_round_flag(0)
        assert projtile.probe(root, scope) is None
        assert (dict(projkernel.calls), dict(int8prefill.calls)) == counted
        state = (projtile._active, projtile._window, projtile._round_flag)
        assert state == (0, None, 0)


def _nonzero_rows(x8) -> int:
    return int(mx.sum(mx.any(x8 != 0, axis=1)).item())


def _scaled(y, factor):
    return (y.astype(mx.float32) * factor).astype(mx.bfloat16)


def _by_rows(x8, codes, scales, biases, info, plan):
    """Rows that move with the number of rows in the launch."""
    y = ptfx.stand_in_tile(x8, codes, scales, biases, info, plan)
    return _scaled(y, 1 + _nonzero_rows(x8) / 64)


def _off_the_floor(x8, codes, scales, biases, info, plan):
    """T-invariant and isolated, but 1 % off everywhere: what a kernel that lost
    precision would look like to every check but the floor."""
    return _scaled(ptfx.stand_in_tile(x8, codes, scales, biases, info, plan), 1.01)


def _mixes_rows(x8, codes, scales, biases, info, plan):
    """Exact on finite rows, but a NaN in one row reaches every row."""
    y = ptfx.stand_in_tile(x8, codes, scales, biases, info, plan)
    return y + 0 * mx.sum(y, axis=0, keepdims=True)


_QKV = (
    "model.layers.1.self_attn.q_proj+model.layers.1.self_attn.k_proj"
    "+model.layers.1.self_attn.v_proj"
)


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        (_by_rows, f"T-invariance: {_QKV} at T=2 differs from its rows alone"),
        (_off_the_floor, "floor: model.layers.1.self_attn.q_proj is "),
        (_mixes_rows, "isolation: a NaN in row 3 changed the other rows of "),
    ],
    ids=["t-invariance", "floor", "isolation"],
)
def test_probe_fails_t_invariance_floor_and_isolation_for_broken_stand_ins(
    monkeypatch, broken, reason
):
    root = _probe_ready(monkeypatch)
    monkeypatch.setattr(projtile, "tile", broken)
    got = projtile.probe(root, "all")
    assert got is not None and got.startswith(reason), got
    assert projtile._active == 0


@pytest.mark.parametrize("dual", [False, True], ids=["single", "dual"])
def test_probe_covers_the_dual_launch_and_the_mlp_at_scope_mlp(monkeypatch, dual):
    """At scope mlp the probe still checks the MLP's launches: the dual's and each
    linear's own."""
    root = _probe_ready(monkeypatch, "mlp")
    if dual:

        def broken_dual(x8, gate, up, info, plan):
            return tuple(
                _scaled(y, 1.01) for y in ptfx.stand_in_tile_dual(x8, gate, up, info, plan)
            )

        monkeypatch.setattr(projtile, "tile_dual", broken_dual)
        want = "floor: model.layers.1.mlp.gate_proj is "
    else:
        monkeypatch.setattr(projtile, "tile", _off_the_floor)
        want = "floor: model.layers.1.mlp.down_proj is "
    got = projtile.probe(root, "mlp")
    assert got is not None and got.startswith(want), got


def _nudged(y):
    """One bf16 ulp off in the first column of the first row."""
    bits = y.view(mx.uint16)
    nudged = mx.where(
        (mx.arange(bits.shape[-2])[:, None] == 0) & (mx.arange(bits.shape[-1]) == 0),
        bits ^ 1,
        bits,
    )
    return nudged.view(mx.bfloat16)


@pytest.mark.parametrize("path", ["mma", "one_row", "int8"])
def test_probe_fails_slab_parity_for_a_slab_path_that_differs(monkeypatch, path):
    """The slab-only MLP's bodies must equal mlx's layout bit for bit: the
    projection kernel's at T = 1..8, its one-row kernel, int8's GEMM at 9, 64 and
    128 rows."""
    root = _probe_ready(monkeypatch)
    if path == "mma":
        mma = projkernel.mma

        def slab_off(x, weights):
            outs = mma(x, weights)
            if isinstance(weights[0], projkernel.Weights) and x.size // x.shape[-1] == 3:
                return tuple(_nudged(o.reshape(-1, o.shape[-1])).reshape(o.shape) for o in outs)
            return outs

        monkeypatch.setattr(projkernel, "mma", slab_off)
        want = (
            "slab: model.layers.1.mlp.gate_proj differs from mlx's layout on the "
            "projection kernel at T=3"
        )
    elif path == "one_row":
        row = projkernel.one_row

        def slab_off_row(x2, w, scales, biases, slab=None):
            if slab is None:
                return row(x2, w, scales, biases)
            return _nudged(row(x2, w, scales, biases, slab=slab))

        monkeypatch.setattr(projkernel, "one_row", slab_off_row)
        want = "slab: model.layers.1.mlp.gate_proj differs from mlx's layout on the one-row kernel"
    else:

        def slab_off_qmm(qa, sa, ra, w, scales, biases, *, slab=None):
            y = ptfx.slab_reference_qmm(qa, sa, ra, w, scales, biases, slab=slab)
            return _nudged(y) if slab is not None and qa.shape[0] == 64 else y

        monkeypatch.setattr(int8prefill, "qmm", slab_off_qmm)
        want = (
            "slab: model.layers.1.mlp.gate_proj differs from mlx's layout on int8's GEMM at 64 rows"
        )
    assert projtile.probe(root, "all") == want


def test_probe_fails_slab_parity_where_only_a_cancelling_layout_shows_it(monkeypatch):
    """The cancelling linears repacked as slab planes: a one-row slab body off only
    where large terms cancel passes every random row and fails here."""
    root = _probe_ready(monkeypatch)
    row = projkernel.one_row

    def off_where_cancelling(x2, w, scales, biases, slab=None):
        if slab is None:
            return row(x2, w, scales, biases)
        y = row(x2, w, scales, biases, slab=slab)
        big = mx.max(mx.abs(x2.astype(mx.float32))).item() > 50
        return _nudged(y) if big and slab.n_real == projtile._CANCEL_N else y

    monkeypatch.setattr(projkernel, "one_row", off_where_cancelling)
    got = projtile.probe(root, "all")
    assert got is not None and got.startswith("slab: a quarter-cancelling K=256 linear "), got


def test_probe_fails_routing_when_the_hooks_never_reach_the_tile(monkeypatch):
    root = _probe_ready(monkeypatch)
    monkeypatch.setattr(projtile, "serves", lambda rows, *, hook: False)
    assert projtile.probe(root, "all") == (
        "routing: the exact verifier's MLP did not reach the tile inside the window"
    )


def test_probe_fails_routing_when_the_tile_serves_outside_the_window(monkeypatch):
    root = _probe_ready(monkeypatch)
    monkeypatch.setattr(projtile, "serves", lambda rows, *, hook: bool(projtile._active))
    assert projtile.probe(root, "all") == "routing: the tile served the MLP outside the window"


def test_probe_restores_its_state_when_it_raises(monkeypatch):
    root = _probe_ready(monkeypatch)

    def fails(*args):
        raise RuntimeError("device lost")

    monkeypatch.setattr(projtile, "tile", fails)
    counted = dict(projkernel.calls)
    with pytest.raises(RuntimeError, match="device lost"):
        projtile.probe(root, "all")
    assert (projtile._active, projtile._window) == (0, None)
    assert projkernel.calls == counted


def test_warm_up_runs_the_int8_slab_gemm_once_per_mlp_shape(monkeypatch):
    """And every launch the model reaches on the tile, and the slab bodies per MLP
    shape, each on zero rows; the counters are left as found."""
    root = _probe_ready(monkeypatch)
    gemms: list = []
    launches: set = set()

    def qmm(qa, sa, ra, w, scales, biases, *, slab=None):
        gemms.append((qa.shape[0], slab))
        return ptfx.slab_reference_qmm(qa, sa, ra, w, scales, biases, slab=slab)

    def tile(x8, codes, scales, biases, info, plan):
        launches.add(("tile", info.k, info.n_pad))
        return ptfx.stand_in_tile(x8, codes, scales, biases, info, plan)

    def tile_dual(x8, gate, up, info, plan):
        launches.add(("dual", info.k, info.n_pad))
        return ptfx.stand_in_tile_dual(x8, gate, up, info, plan)

    monkeypatch.setattr(int8prefill, "qmm", qmm)
    monkeypatch.setattr(projtile, "tile", tile)
    monkeypatch.setattr(projtile, "tile_dual", tile_dual)
    counted = (dict(projkernel.calls), dict(int8prefill.calls))
    projtile.warm_up(root, "all")
    up = slab.slab_info(root.model.layers[0].mlp.up_proj)
    down = slab.slab_info(root.model.layers[0].mlp.down_proj)
    assert gemms == [(int8prefill.SLAB_MIN_ROWS, up), (int8prefill.SLAB_MIN_ROWS, down)]
    assert launches == {
        ("dual", 256, 512),  # gate+up
        ("tile", 256, 512),  # gate, up, k, v alone
        ("tile", 512, 256),  # down
        ("tile", 256, 3072),  # q+k+v
        ("tile", 256, 2048),  # q alone
        ("tile", 1024, 256),  # o_proj
        ("tile", 256, 768),  # in_proj_qkv+in_proj_z
        ("tile", 256, 256),  # in_proj_b+in_proj_a, in_proj_z alone, out_proj
    }
    assert (dict(projkernel.calls), dict(int8prefill.calls)) == counted
    # At scope mlp only the MLP's launches.
    launches.clear()
    gemms.clear()
    narrow = ptfx.tiny_slab_model(seed=1, scope="mlp")
    projtile.warm_up(narrow, "mlp")
    assert launches == {("dual", 256, 512), ("tile", 256, 512), ("tile", 512, 256)}
    assert len(gemms) == 2


# The 27B's config.json as sous tune reads it.
_CONFIG_27B: dict[str, Any] = {
    "model_type": "qwen3_5",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
    },
}
_SWITCHES = {
    "projection_tile": True,
    "projection_kernel": True,
    "attention_tile": True,
    "int8_prefill": True,
}


def _with_text(**changes) -> dict:
    return {**_CONFIG_27B, "text_config": {**_CONFIG_27B["text_config"], **changes}}


@pytest.mark.parametrize(
    ("switches", "model_config", "drafter_kind", "reason"),
    [
        ({}, _CONFIG_27B, "dflash", None),
        ({"projection_tile": "mlp"}, _CONFIG_27B, "dflash", None),
        ({"projection_tile": "all"}, _CONFIG_27B, "dflash", None),
        ({"projection_tile": False}, _CONFIG_27B, "dflash", "projection tile off"),
        ({"projection_kernel": False}, _CONFIG_27B, "dflash", _GUARD_REASONS[0]),
        ({}, _CONFIG_27B, None, _GUARD_REASONS[2]),
        ({"int8_prefill": False}, _CONFIG_27B, "dflash", _GUARD_REASONS[3]),
        (
            {},
            _with_text(intermediate_size=13824),
            "dflash",
            "no tuned plan for (5120, (13824, 13824))",
        ),
    ],
    ids=["active", "mlp", "all", "off", "kernel-off", "no-drafter", "int8-off", "no-plan"],
)
def test_static_reason_mirrors_enable(monkeypatch, switches, model_config, drafter_kind, reason):
    """Decided from the config, config.json and the machine alone, in enable()'s
    order and words: the switch, the kernel's whole static check (which carries
    the tile's), the drafter, int8 prefill, the core count, then the plans for
    the MLP's keys."""
    pfx.proj_ready(monkeypatch)
    config = types.SimpleNamespace(**{**_SWITCHES, **switches})
    assert projtile.static_reason(config, model_config, drafter_kind) == reason


def test_static_reason_reads_the_core_count_against_the_tiles_own_set(monkeypatch):
    pfx.proj_ready(monkeypatch)
    config = types.SimpleNamespace(**_SWITCHES)
    monkeypatch.setattr(projtile, "MEASURED_CORES", frozenset({24}))
    assert projtile.static_reason(config, _CONFIG_27B, "dflash") == (
        "GPU core count 20 not measured (measured: 24)"
    )


def test_static_reason_reads_nothing_while_the_tile_is_off(monkeypatch):
    pfx.proj_ready(monkeypatch)

    def untouchable(*args, **kwargs):
        raise AssertionError("read the machine for a switched-off tile")

    monkeypatch.setattr(projkernel, "static_reason", untouchable)
    monkeypatch.setattr(gpucores, "read", untouchable)
    config = types.SimpleNamespace(**{**_SWITCHES, "projection_tile": False})
    assert projtile.static_reason(config, {}, None) == "projection tile off"


_TINY_TEXT = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "full_attention_interval",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
)


def test_static_copy_bytes_matches_copy_bytes_on_the_tiny_model(monkeypatch):
    ptfx.tiny_plans(monkeypatch)
    root = pfx.tiny_quantized_model()
    text = {name: getattr(root.args, name) for name in _TINY_TEXT}
    want = projtile.copy_bytes(root, keys_with_plans=_planned)
    assert want > 0
    kinds = [
        "linear_attention" if layer.is_linear else "full_attention" for layer in root.model.layers
    ]
    by_types = {k: v for k, v in text.items() if k != "full_attention_interval"}
    for config in (
        {"text_config": text},
        text,
        {"text_config": {**by_types, "layer_types": kinds}},
    ):
        assert projtile.static_copy_bytes(config) == want
    # A key without a plan is left out of both.
    plans = dict(projtile.PLANS)
    del plans[(256, (4, 4))]
    monkeypatch.setattr(projtile, "PLANS", plans)
    fewer = projtile.copy_bytes(root, keys_with_plans=_planned)
    assert fewer < want and projtile.static_copy_bytes(text) == fewer


def test_static_copy_bytes_of_the_27b():
    """64 layers, every fourth full attention: per linear-attention layer qkv+z
    (16384 wide), b+a (96, one tile) and out_proj; per full-attention layer q+k+v
    (14336) and o_proj; 9/16 byte per padded weight."""
    linear = (16384 * 5120 + 256 * 5120 + 5120 * 6144) * 9 // 16
    full = (14336 * 5120 + 5120 * 6144) * 9 // 16
    assert projtile.static_copy_bytes(_CONFIG_27B) == 48 * linear + 16 * full == 4_093_378_560


@tile_nax
@pytest.mark.parametrize("scope", ["all", "mlp"])
def test_enable_goes_active_on_the_real_kernels(monkeypatch, scope):
    """The real tile, projection kernel, one-row kernel and int8 GEMM, through the
    real availability, real-shape compile, warm-up and probe, on the tiny model at
    the tiny plans."""
    ptfx.guard_tile_state(monkeypatch)
    tile_status = pfx.proj_ready(monkeypatch, real_pins=True)
    pfx.real_kernel(monkeypatch)
    projkernel.install_hooks()
    monkeypatch.setattr(projkernel, "_active", 1)
    ptfx.tiny_plans(monkeypatch)
    root = ptfx.ready_model()
    status = projtile.enable(
        root,
        config_value=scope,
        kernel_status={"state": "active", "reason": None, "probe_seconds": 0.5},
        tile_status=tile_status,
        int8_status={"state": "active", "reason": None, "routed": 24},
        drafter_kind="dflash",
        cache_budget=None,
        headroom_bytes=None,
        reserve_bytes=0,
    )
    assert (status["state"], status["scope"]) == ("active", scope), status
    assert _marked(root) == (_CONVERTED if scope == "all" else _MLP_ONLY)
