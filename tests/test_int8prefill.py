"""INT8-activation prefill: Stage A, the K-order, eligibility, routing, enablement.

Everything here runs on any Metal GPU (CI is macos-15, no tensor units); the GEMM
itself is exercised by the `nax`-gated tests at the bottom, which skip where the
neural accelerators are absent and run on the M5 Pro.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sous.engine import int8prefill as i8  # noqa: E402 — after the importorskip guards


def _unreorder(qa):
    """Inverse of reorder_k: (c, a, b, j) -> (c, a, j, b) within each 64-code group."""
    m, k = qa.shape
    return qa.reshape(m, k // 64, 4, 2, 2, 4).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)


# ---- K-order --------------------------------------------------------------------


def test_reorder_k_slot_formula():
    """Slot 16c + 4t + j of a group holds code 16c + 8*(t>>1) + 2j + (t&1); the
    second group is the same pattern offset by 64."""
    ids = mx.arange(128, dtype=mx.int32).astype(mx.int8).reshape(1, 128)
    out = i8.reorder_k(ids).tolist()[0]
    for c in range(4):
        for t in range(4):
            for j in range(4):
                assert out[16 * c + 4 * t + j] == 16 * c + 8 * (t >> 1) + 2 * j + (t & 1)
    assert out[64:] == [64 + v for v in out[:64]]


def test_reorder_k_round_trips():
    x = mx.random.randint(-127, 128, (3, 256)).astype(mx.int8)
    out = i8.reorder_k(x)
    assert out.shape == x.shape and out.dtype == mx.int8
    assert mx.array_equal(_unreorder(out), x).item()


# ---- Stage A --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
def test_stage_a_matches_the_reference(dtype):
    m, k = 37, 1024
    x = (mx.random.normal((m, k)) * 0.5).astype(dtype)
    qa, sa, ra = i8.stage_a(x)
    mx.eval(qa, sa, ra)
    assert (qa.shape, qa.dtype) == ((m, k), mx.int8)
    assert (sa.shape, sa.dtype) == ((m,), mx.float32)
    assert (ra.shape, ra.dtype) == ((k // 64, m), mx.int16)

    xf = x.astype(mx.float32)
    amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    q_ref = mx.clip(mx.round(xf * (127.0 / amax)), -127, 127)
    q_got = _unreorder(qa).astype(mx.float32)
    diff = mx.abs(q_got - q_ref)
    # fp32 (kernel) vs fp32-from-the-same-inputs (mx ops) differ only at rounding
    # ties: never by more than one code, and rarely (the spike measured 0.03%).
    assert mx.max(diff).item() <= 1
    assert mx.mean(diff > 0).item() <= 1e-3
    assert mx.allclose(sa, (amax / 127.0).reshape(-1), rtol=1e-6, atol=0.0).item()
    # Ra is the sum of the kernel's OWN codes, exactly — the affine correction
    # relies on it — so it is checked against q_got, not q_ref.
    ra_ref = q_got.reshape(m, k // 64, 64).sum(axis=-1).astype(mx.int16).T
    assert mx.array_equal(ra, ra_ref).item()


def test_stage_a_zero_row_emits_zeros_not_nan():
    rest = (mx.random.normal((4, 128)) * 0.5).astype(mx.bfloat16)
    x = mx.concatenate([mx.zeros((1, 128), dtype=mx.bfloat16), rest])
    qa, sa, ra = i8.stage_a(x)
    assert mx.all(qa[0] == 0).item()
    assert sa[0].item() == 0.0
    assert mx.all(ra[:, 0] == 0).item()
    assert not mx.any(mx.isnan(sa)).item()


def test_stage_a_flattens_leading_dims():
    x = (mx.random.normal((1, 200, 128)) * 0.5).astype(mx.bfloat16)
    qa, sa, ra = i8.stage_a(x)
    assert qa.shape == (200, 128) and sa.shape == (200,) and ra.shape == (2, 200)


# ---- availability -----------------------------------------------------------------


def _fake_platform(monkeypatch, mac_ver: str, arch: str):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "mac_ver", lambda: (mac_ver, ("", "", ""), "arm64"))
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})


@pytest.mark.parametrize(
    ("mac_ver", "arch", "available", "needle"),
    [
        ("26.6.2", "applegpu_g17s", True, None),
        ("26.2", "applegpu_g17g", True, None),
        ("26.6.2", "applegpu_g18p", True, None),
        ("26.6.2", "applegpu_g17p", False, "gen 17 < 18"),
        ("26.6.2", "applegpu_g16s", False, "gen 16 < 17"),
        ("26.1", "applegpu_g17s", False, "macOS 26.1 < 26.2"),
        ("26.6.2", "something_else", False, "unrecognized"),
    ],
)
def test_availability_mirrors_mlx_nax_rule(monkeypatch, mac_ver, arch, available, needle):
    _fake_platform(monkeypatch, mac_ver, arch)
    a = i8.availability()
    assert a.available is available
    if needle is not None:
        assert a.reason is not None and needle in a.reason


def test_availability_on_this_machine_reports_a_reason_when_unavailable():
    a = i8.availability()
    assert a.available or (isinstance(a.reason, str) and a.reason)


# ---- GEMM (needs the tensor units; skipped where they are absent) ----------------

nax = pytest.mark.skipif(
    not i8.availability().available, reason=f"no tensor units: {i8.availability().reason}"
)


def _unpack_q4(w):
    """uint32 [N, K/8] -> codes int32 [N, K]; code i of a word sits at bits 4i."""
    shifts = (mx.arange(8, dtype=mx.uint32) * 4).reshape(1, 1, 8)
    codes = (w[:, :, None] >> shifts) & mx.array(0xF, dtype=mx.uint32)
    return codes.reshape(w.shape[0], -1).astype(mx.int32)


def _random_words(n, words):
    hi = mx.random.randint(0, 1 << 16, (n, words)).astype(mx.uint32)
    lo = mx.random.randint(0, 1 << 16, (n, words)).astype(mx.uint32)
    return (hi << mx.array(16, dtype=mx.uint32)) | lo


@nax
@pytest.mark.parametrize(
    ("k", "n", "m"),
    [(64, 64, 32), (128, 64, 32), (1024, 128, 37), (1024, 128, 64), (256, 192, 300)],
)
def test_gemm_is_bit_exact_with_power_of_two_metadata(k, n, m):
    """With power-of-two scales and biases every fp32 partial sum is exact, so the
    kernel's only rounding is the final bf16 cast: it must equal the fp64-exact
    reference rounded to bf16. This is the test that fails if the fragment layout
    or the K-order is wrong."""
    qa_nat = mx.random.randint(-127, 128, (m, k)).astype(mx.int8)
    ra = qa_nat.astype(mx.int32).reshape(m, k // 64, 64).sum(axis=-1)  # [M, G]
    sa = mx.ones((m,), dtype=mx.float32)
    w = _random_words(n, k // 8)
    s, b = 2.0**-8, -(2.0**-5)
    scales = mx.full((n, k // 64), s, dtype=mx.bfloat16)
    biases = mx.full((n, k // 64), b, dtype=mx.bfloat16)
    # Integer-exact in fp32 on the CPU stream: |products| <= 127*15, sums < 2^24.
    dots = mx.matmul(qa_nat.astype(mx.float32), _unpack_q4(w).astype(mx.float32).T, stream=mx.cpu)
    ref = dots * s + ra.sum(axis=-1, keepdims=True).astype(mx.float32) * b
    got = i8.qmm(i8.reorder_k(qa_nat), sa, mx.contiguous(ra.astype(mx.int16).T), w, scales, biases)
    assert got.shape == (m, n) and got.dtype == mx.bfloat16
    assert mx.array_equal(got, ref.astype(mx.bfloat16)).item()


@nax
def test_gemm_applies_the_row_scale():
    k, n, m = 128, 64, 40
    qa_nat = mx.random.randint(-127, 128, (m, k)).astype(mx.int8)
    ra = qa_nat.astype(mx.int32).reshape(m, 2, 64).sum(axis=-1)
    sa = mx.array([2.0 ** (i % 5) for i in range(m)], dtype=mx.float32)
    w = _random_words(n, k // 8)
    scales = mx.full((n, 2), 2.0**-8, dtype=mx.bfloat16)
    biases = mx.full((n, 2), -(2.0**-5), dtype=mx.bfloat16)
    dots = mx.matmul(qa_nat.astype(mx.float32), _unpack_q4(w).astype(mx.float32).T, stream=mx.cpu)
    ref = (dots * 2.0**-8 + ra.sum(axis=-1, keepdims=True).astype(mx.float32) * -(2.0**-5)) * sa[
        :, None
    ]
    got = i8.qmm(i8.reorder_k(qa_nat), sa, mx.contiguous(ra.astype(mx.int16).T), w, scales, biases)
    assert mx.array_equal(got, ref.astype(mx.bfloat16)).item()


@nax
def test_gemm_tracks_stock_qmm_within_the_int8_noise():
    k, n, m = 5120, 1024, 300
    w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4, mode="affine")
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    got = i8.qmm(*i8.stage_a(x), packed, scales, biases).astype(mx.float32)
    ref = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True, group_size=64, bits=4, mode="affine"
    ).astype(mx.float32)
    rel_rms = mx.sqrt(mx.mean((got - ref) ** 2)) / mx.sqrt(mx.mean(ref**2))
    # Per-row int8 activation quantization noise; the spike measured 9.2e-3.
    assert rel_rms.item() < 1.5e-2


@nax
def test_linear_int8_keeps_leading_dims_and_matches_qmm():
    parent = nn.Sequential(nn.Linear(256, 128, bias=False))
    parent.set_dtype(mx.bfloat16)
    nn.quantize(parent, group_size=64, bits=4)
    linear = parent.layers[0]
    x = (mx.random.normal((1, 200, 256)) * 0.5).astype(mx.bfloat16)
    out = i8.linear_int8(linear, x)
    assert out.shape == (1, 200, 128) and out.dtype == mx.bfloat16
    stage = i8.stage_a(x)
    direct = i8.qmm(*stage, linear.weight, linear.scales, linear.biases).reshape(1, 200, 128)
    assert mx.array_equal(out, direct).item()
    # A handed-down stage is used as-is.
    assert mx.array_equal(i8.linear_int8(linear, x, stage), direct).item()


def test_qmm_rejects_untiled_shapes():
    qa = mx.zeros((32, 64), dtype=mx.int8)
    sa = mx.ones((32,), dtype=mx.float32)
    ra = mx.zeros((1, 32), dtype=mx.int16)
    with pytest.raises(ValueError, match="N=48"):
        i8.qmm(qa, sa, ra, mx.zeros((48, 8), dtype=mx.uint32), mx.ones((48, 1)), mx.zeros((48, 1)))


# ---- eligibility --------------------------------------------------------------


def _qlinear(k, n, *, bias=False, dtype=mx.bfloat16, **quant):
    parent = nn.Sequential(nn.Linear(k, n, bias=bias))
    parent.set_dtype(dtype)
    nn.quantize(parent, **{"group_size": 64, "bits": 4, **quant})
    return parent.layers[0]


def test_eligible_linear_table():
    assert i8.eligible_linear(_qlinear(128, 256))
    assert i8.eligible_linear(_qlinear(128, 256, dtype=mx.float16))
    assert not i8.eligible_linear(_qlinear(128, 256, bits=8))
    assert not i8.eligible_linear(_qlinear(128, 256, group_size=128))
    assert not i8.eligible_linear(_qlinear(128, 256, group_size=32, mode="mxfp4"))
    assert not i8.eligible_linear(_qlinear(128, 48)), "N must tile by 64"
    assert not i8.eligible_linear(_qlinear(128, 256, bias=True))
    assert not i8.eligible_linear(_qlinear(128, 256, dtype=mx.float32)), "fp32 scales"
    assert not i8.eligible_linear(nn.Linear(128, 256)), "not quantized"


# ---- routing (any Metal GPU: qmm is replaced by an exact-enough reference) ------


def _reference_qmm(qa, sa, ra, w, scales, biases):
    x_hat = _unreorder(qa).astype(mx.float32) * sa[:, None]
    wd = mx.dequantize(w, scales, biases, group_size=64, bits=4, mode="affine")
    return (x_hat @ wd.astype(mx.float32).T).astype(scales.dtype)


class _TinyGDN(nn.Module):
    """Stand-in with the projection names the GDN wrapper looks for."""

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(128, 256, bias=False)
        self.in_proj_z = nn.Linear(128, 128, bias=False)
        self.in_proj_a = nn.Linear(128, 48, bias=False)
        self.out_proj = nn.Linear(128, 128, bias=False)

    def __call__(self, inputs, mask=None, cache=None):
        mixed = self.in_proj_qkv(inputs)[..., :128] + self.in_proj_z(inputs)
        return self.out_proj(mixed) + self.in_proj_a(inputs)[..., :1]


class _Counter:
    def __init__(self, monkeypatch):
        self.stage_calls = 0
        self.qmm_calls = 0
        real_stage_a = i8.stage_a

        def counting_stage_a(x):
            self.stage_calls += 1
            return real_stage_a(x)

        def counting_qmm(*args):
            self.qmm_calls += 1
            return _reference_qmm(*args)

        monkeypatch.setattr(i8, "stage_a", counting_stage_a)
        monkeypatch.setattr(i8, "qmm", counting_qmm)


def _mlp():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    mlp = Qwen3_5MLP(128, 256)
    mlp.set_dtype(mx.bfloat16)
    nn.quantize(mlp, group_size=64, bits=4)
    return mlp


def _tag_all(module):
    for _, m in module.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            object.__setattr__(m, i8._TAG, True)


@pytest.fixture
def wrappers():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    i8.install_wrappers([(Qwen3_5MLP, "mlp"), (_TinyGDN, "gdn")])


def test_mlp_routes_with_one_shared_stage_above_the_floor(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    mlp = _mlp()
    x = (mx.random.normal((1, 130, 128)) * 0.5).astype(mx.bfloat16)
    stock = mlp(x)
    _tag_all(mlp)
    routed = mlp(x)
    assert (counter.stage_calls, counter.qmm_calls) == (2, 3), "gate+up share; down has its own"
    rel = mx.sqrt(mx.mean((routed - stock).astype(mx.float32) ** 2)) / mx.sqrt(
        mx.mean(stock.astype(mx.float32) ** 2)
    )
    assert rel.item() < 3e-2


def test_rows_below_the_floor_take_the_stock_path(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    mlp = _mlp()
    _tag_all(mlp)
    mlp((mx.random.normal((1, 127, 128)) * 0.5).astype(mx.bfloat16))
    mlp((mx.random.normal((1, 1, 128)) * 0.5).astype(mx.bfloat16))
    assert counter.qmm_calls == 0


def test_untagged_modules_are_never_routed(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    other = _mlp()  # same class, never tagged: a second resident model
    other((mx.random.normal((1, 130, 128)) * 0.5).astype(mx.bfloat16))
    assert counter.qmm_calls == 0


def test_gdn_shares_one_stage_between_qkv_and_z_and_clears_it(monkeypatch, wrappers):
    counter = _Counter(monkeypatch)
    gdn = _TinyGDN()
    gdn.set_dtype(mx.bfloat16)
    nn.quantize(gdn, group_size=64, bits=4)
    for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
        object.__setattr__(getattr(gdn, name), i8._TAG, True)
    x = (mx.random.normal((1, 200, 128)) * 0.5).astype(mx.bfloat16)
    gdn(x)
    # one Stage A for qkv+z, one for out_proj; in_proj_a (N=48) untagged -> stock
    assert (counter.stage_calls, counter.qmm_calls) == (2, 3)
    assert getattr(gdn.in_proj_qkv, i8._STAGE) is None
    assert getattr(gdn.in_proj_z, i8._STAGE) is None


def test_install_wrappers_is_idempotent():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    i8.install_wrappers([(Qwen3_5MLP, "mlp")])
    first = Qwen3_5MLP.__call__
    i8.install_wrappers([(Qwen3_5MLP, "mlp")])
    assert Qwen3_5MLP.__call__ is first


# ---- enable() -----------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self, down_bits=4):
        super().__init__()
        self.mlp = _mlp()
        if down_bits != 4:
            self.mlp.down_proj = _qlinear(256, 128, bits=down_bits)
        self.linear_attn = _TinyGDN()
        self.linear_attn.set_dtype(mx.bfloat16)
        nn.quantize(self.linear_attn, group_size=64, bits=4)
        self.self_attn = nn.Sequential(nn.Linear(128, 256, bias=False))
        self.self_attn.set_dtype(mx.bfloat16)
        nn.quantize(self.self_attn, group_size=64, bits=4)


class _Inner(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class _LanguageModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _Inner(layers)
        self.lm_head = _qlinear(128, 256)


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.language_model = _LanguageModel(layers)


def _tags(model):
    return sorted(
        path
        for path, m in model.named_modules()
        if isinstance(m, nn.QuantizedLinear) and getattr(m, i8._TAG, False)
    )


def test_enable_off_touches_nothing():
    model = _Model([_Layer()])
    assert i8.enable(model, enabled=False) == {"state": "off", "reason": None, "routed": 0}
    assert _tags(model) == []


def test_enable_unavailable_tags_nothing(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(False, "no tensor units"))
    model = _Model([_Layer()])
    status = i8.enable(model, enabled=True)
    assert status == {"state": "unavailable", "reason": "no tensor units", "routed": 0}
    assert _tags(model) == []


def test_enable_tags_mlp_and_gdn_but_not_attention_or_lm_head(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(), _Layer()])
    status = i8.enable(model, enabled=True)
    assert status == {"state": "active", "reason": None, "routed": 12}
    tagged = _tags(model)
    assert len(tagged) == 12
    assert all(".layers." in p for p in tagged)
    assert not any("self_attn" in p or "lm_head" in p or "in_proj_a" in p for p in tagged)


def test_enable_tags_an_mlp_all_or_nothing(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(down_bits=8)])
    status = i8.enable(model, enabled=True)
    assert status["routed"] == 3, "only the GDN projections; the MLP with an 8-bit down stays stock"
    assert not any(".mlp." in p for p in _tags(model))


def test_enable_with_no_eligible_projection_is_unavailable(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    model = _Model([])
    status = i8.enable(model, enabled=True)
    assert status["state"] == "unavailable"
    assert status["reason"] is not None and "no eligible projections" in status["reason"]


def test_enable_degrades_when_warm_up_fails(monkeypatch):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))

    def boom(model):
        raise RuntimeError("compiler said no")

    monkeypatch.setattr(i8, "_warm_up", boom)
    model = _Model([_Layer()])
    with pytest.warns(UserWarning, match="compiler said no"):
        status = i8.enable(model, enabled=True)
    assert status == {"state": "unavailable", "reason": "compiler said no", "routed": 0}
    assert _tags(model) == []


def test_warm_up_runs_one_int8_linear_per_distinct_shape(monkeypatch):
    seen = []

    def fake_linear_int8(linear, x, stage=None):
        seen.append((x.shape, linear.weight.shape[0], x.dtype))
        return mx.zeros((x.shape[0], linear.weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(i8, "linear_int8", fake_linear_int8)
    model = _Model([_Layer(), _Layer()])
    i8._tag(model)
    i8._warm_up(model)
    # gate/up (128->256), down (256->128), qkv (128->256), z (128->128), out (128->128):
    # distinct (K, N, dtype) = {(128,256), (256,128), (128,128)}
    assert set(seen) == {
        ((128, 128), 256, mx.bfloat16),
        ((128, 256), 128, mx.bfloat16),
        ((128, 128), 128, mx.bfloat16),
    }
