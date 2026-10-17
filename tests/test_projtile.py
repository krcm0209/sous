"""The projection tile (sous.engine.projtile) and the slab layout it reads
(sous.engine.slab): the layout and its inverse, the slab-only MLP conversion and
its revert, the stacked copies of the attention and GDN linears, and the fork
key's reading of the markers. Plain mlx: nothing here needs tensor units."""

import types

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")

from sous.engine import projtile, slab  # noqa: E402 — after the guards
from tests import proj_fixtures as pfx  # noqa: E402
from tests import projtile_fixtures as ptfx  # noqa: E402

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
