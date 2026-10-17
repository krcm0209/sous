"""The projection tile (sous.engine.projtile) and the slab layout it reads
(sous.engine.slab): the layout and its inverse, the slab-only MLP conversion and
its revert, the stacked copies of the attention and GDN linears, the fork key's
reading of the markers, and the tile kernel with its plans. Plain mlx except the
`tile_nax` tests, which run the kernel on the tensor units."""

import types
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")

from sous.engine import int8prefill, projtile, slab  # noqa: E402 — after the guards
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
    call = projtile.plan_for_call
    # The production table holds none of the tiny model's keys.
    assert call(qkv) is None and call((mlp.gate_proj, mlp.up_proj)) is None

    # A distinct plan per key, so each lookup shows which key it took.
    plans = {key: P(128, 4, i + 1, True) for i, key in enumerate(ptfx.TINY_PLANS)}
    monkeypatch.setattr(projtile, "PLANS", plans)
    # One plan per launch: a stack the call covers whole, the GDN call's two.
    assert call(qkv) == (plans[(256, (2048, 512, 512))],)
    assert call(gdn_call) == (plans[(256, (512, 256))], plans[(256, (4, 4))])
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
