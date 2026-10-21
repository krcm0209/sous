"""INT8-activation prefill: Stage A, the K-order, eligibility, routing, enablement.

Everything here runs on any Metal GPU (CI is macos-15, no tensor units); the GEMM
itself is exercised by the `nax`-gated tests at the bottom, which skip where the
neural accelerators are absent and run on the M5 Pro.
"""

import hashlib
import inspect
import types
import warnings

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sous.engine import int8prefill as i8  # noqa: E402 — after the importorskip guards
from sous.engine.slab import SLAB_ATTR, SlabInfo, repack, slab_info  # noqa: E402
from tests import proj_fixtures as pfx  # noqa: E402
from tests import projtile_fixtures as ptfx  # noqa: E402

_unreorder = ptfx.unreorder


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


# ---- kernel sources ---------------------------------------------------------------


def test_kernel_sources_ship_as_package_files():
    """The Metal lives in kernels/ as files, not Python strings, and only the GEMM's
    header pulls in the Metal-4 tensor-op include (Stage A must compile on any GPU)."""
    from importlib.resources import files

    root = files("sous.engine.kernels")
    texts = {
        n: root.joinpath(n).read_text()
        for n in ("common.h", "nax.h", "stage_a.metal", "q4_int8_gemm.metal")
    }
    assert all(texts.values())
    assert "MetalPerformancePrimitives" in texts["nax.h"]
    assert "MetalPerformancePrimitives" not in texts["common.h"] + texts["stage_a.metal"]
    assert "mpp::tensor_ops" in texts["q4_int8_gemm.metal"]


def test_the_gemm_source_defaults_slab_to_zero():
    """The slab read is switched by a header define, never a template argument
    the preprocessor cannot see: only the slab kernel's header sets it, so the
    mlx-layout kernel compiles from the source it compiled from before."""
    text = i8._kernel_text("q4_int8_gemm.metal")
    assert "#ifndef SLAB\n#define SLAB 0\n#endif\n" in text
    first = text.index("#if SLAB\n")
    assert text.index("#ifndef SLAB") < first
    # the weight base and strides, the per-group step, the scale and bias rows
    assert text.count("#if SLAB\n") == 3
    assert text.count("#else\n") == 3 and text.count("#endif\n") == 4


def test_kernel_hands_mlx_only_the_arguments_it_was_given(monkeypatch):
    """int8's kernels take mlx's defaults and pass neither of the keywords the
    attention tile needs: passing them would tie int8 prefill to mlx
    accepting them, and a TypeError there would turn it unavailable."""
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs
        return lambda **launch: []

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(i8, "_kernels", {})
    i8._kernel("sous_test_plain", ["x"], ["y"], "y[0] = x[0];", "")
    i8._kernel(
        "sous_test_strided",
        ["x"],
        ["y"],
        "y[0] = x[0];",
        "",
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )
    assert set(made["sous_test_plain"]) == {
        "name",
        "input_names",
        "output_names",
        "source",
        "header",
    }
    strided = made["sous_test_strided"]
    assert strided["ensure_row_contiguous"] is False
    assert strided["compile_options"] == {"math_mode": "safe"}


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
    monkeypatch.setattr(i8, "_gemm_probe", lambda: None)
    a = i8.availability()
    assert a.available is available
    if needle is not None:
        assert a.reason is not None and needle in a.reason


def test_availability_on_this_machine_reports_a_reason_when_unavailable():
    a = i8.availability()
    assert a.available or (isinstance(a.reason, str) and a.reason)


def test_availability_is_unavailable_when_the_gemm_does_not_compile(monkeypatch):
    """A toolchain that rejects the kernel reads as unavailable, so the load warns
    once and the `nax` tests below skip instead of hanging in mlx's exception path."""
    _fake_platform(monkeypatch, "27.0", "applegpu_g17s")
    monkeypatch.setattr(i8, "_gemm_probe", lambda: "Unable to build metal library from source")
    a = i8.availability()
    assert a.available is False
    assert a.reason is not None and "Unable to build metal library" in a.reason


_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "utils.h:544:27: error: no matching member function for call to "
    "'get_destination_cooperative_tensor'\n"
    "  auto acc0 = op.template get_destination_cooperative_tensor<...>();"
)


def test_gemm_probe_reports_the_compilers_error_line_and_logs_the_rest(monkeypatch, caplog):
    """mlx's first line is the same for every compile failure; the reason keeps the
    line that names this one, and the log keeps the whole report."""
    monkeypatch.setattr(i8, "_probe_ok", False)
    monkeypatch.setattr(i8, "stage_a", lambda x: (x, x, x))

    def rejected(*args):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(i8, "qmm", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.int8prefill"):
        reason = i8._gemm_probe()
    assert reason == (
        "utils.h:544:27: error: no matching member function for call to "
        "'get_destination_cooperative_tensor'"
    )
    assert "auto acc0" in caplog.text


def test_gemm_probe_remembers_success_but_not_failure(monkeypatch):
    """One bad moment must not switch int8 off for the rest of a daemon's life."""
    monkeypatch.setattr(i8, "_probe_ok", False)
    monkeypatch.setattr(i8, "stage_a", lambda x: (x, x, x))
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return mx.zeros((1,))

    monkeypatch.setattr(i8, "qmm", flaky)
    assert i8._gemm_probe() == "transient"
    assert i8._gemm_probe() is None
    assert i8._gemm_probe() is None
    assert len(calls) == 2


# ---- GEMM (needs the tensor units; skipped where they are absent) ----------------

# Read once: a failed probe is not remembered, so each call would compile again.
_AVAILABILITY = i8.availability()
nax = pytest.mark.skipif(
    not _AVAILABILITY.available, reason=f"int8 GEMM unavailable: {_AVAILABILITY.reason}"
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


def _exact_case(k, n, m):
    """Random int8 activations and Q4 words with power-of-two scales and biases:
    every fp32 partial sum the GEMM forms is exact, so its one rounding is the
    final bf16 cast. Returns (qa reordered, sa, ra, weight, scales, biases, the
    exact result rounded to bf16)."""
    qa_nat = mx.random.randint(-127, 128, (m, k)).astype(mx.int8)
    ra = qa_nat.astype(mx.int32).reshape(m, k // 64, 64).sum(axis=-1)  # [M, G]
    sa = mx.ones((m,), dtype=mx.float32)
    w = _random_words(n, k // 8)
    n_idx = mx.arange(n).reshape(n, 1)
    g_idx = mx.arange(k // 64).reshape(1, k // 64)
    scales = (2.0 ** (-(8 + ((n_idx + g_idx) % 3)))).astype(mx.bfloat16)
    biases = (-(2.0 ** (-(5 + ((n_idx + 2 * g_idx) % 3))))).astype(mx.bfloat16)
    # Integer-exact in fp32 on the CPU stream: |products| <= 127*15, sums < 2^24.
    qa_g = qa_nat.astype(mx.float32).reshape(m, k // 64, 64)
    w_g = _unpack_q4(w).astype(mx.float32).reshape(n, k // 64, 64)
    dots_g = mx.stack(
        [mx.matmul(qa_g[:, g], w_g[:, g].T, stream=mx.cpu) for g in range(k // 64)], axis=-1
    )  # [M, N, G], exact integers
    ref = mx.sum(dots_g * scales.astype(mx.float32)[None], axis=-1) + mx.sum(
        ra.astype(mx.float32)[:, None, :] * biases.astype(mx.float32)[None], axis=-1
    )
    qa, ra16 = i8.reorder_k(qa_nat), mx.contiguous(ra.astype(mx.int16).T)
    return qa, sa, ra16, w, scales, biases, ref.astype(mx.bfloat16)


@nax
@pytest.mark.parametrize(
    ("k", "n", "m"),
    [(64, 64, 32), (128, 64, 32), (1024, 128, 37), (1024, 128, 64), (256, 192, 300)],
)
def test_gemm_is_bit_exact_with_power_of_two_metadata(k, n, m):
    """With power-of-two scales and biases the kernel must equal the fp64-exact
    reference rounded to bf16. This is the test that fails if the fragment layout
    or the K-order is wrong."""
    qa, sa, ra, w, scales, biases, want = _exact_case(k, n, m)
    got = i8.qmm(qa, sa, ra, w, scales, biases)
    assert got.shape == (m, n) and got.dtype == mx.bfloat16
    assert mx.array_equal(got, want).item()


@nax
@pytest.mark.parametrize("n", [1024, 4160], ids=["n1024", "n4160"])
@pytest.mark.parametrize("rows", [9, 64, 128])
def test_the_slab_gemm_equals_the_mlx_layout_gemm(n, rows):
    """The slab read moves where a column's group and its scale and bias are
    read and nothing else, so on the same Stage A it equals the mlx-layout read
    bit for bit, at a width that fills its slab tiles and one (4160) whose last
    tile pads; and on power-of-two metadata both equal the exact result."""
    k = 1024
    w, scales, biases = pfx.rand_linear(n, k, seed=rows)
    stage = i8.stage_a(pfx.activations(rows, k, seed=n))
    codes, s, b, info = repack(w, scales, biases)
    want = i8.qmm(*stage, w, scales, biases)
    got = i8.qmm(*stage, codes, s, b, slab=info)
    assert got.shape == (rows, n) and got.dtype == mx.bfloat16
    assert mx.array_equal(got, want).item()

    qa, sa, ra, w, scales, biases, exact = _exact_case(k, n, rows)
    codes, s, b, info = repack(w, scales, biases)
    assert mx.array_equal(i8.qmm(qa, sa, ra, codes, s, b, slab=info), exact).item()
    assert mx.array_equal(i8.qmm(qa, sa, ra, w, scales, biases), exact).item()


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
    qa = mx.zeros((32, 96), dtype=mx.int8)
    with pytest.raises(ValueError, match="K=96"):
        i8.qmm(qa, sa, ra, mx.zeros((64, 12), dtype=mx.uint32), mx.ones((64, 2)), mx.zeros((64, 2)))


def _capture_launches(monkeypatch):
    """Every int8 kernel launch as (name, header, inputs, template, grid,
    output_shapes); each returns zeros of its shapes and nothing compiles."""
    launches = []

    def kernel(name, input_names, output_names, source, header, **options):
        def launch(*, inputs, template, grid, output_shapes, output_dtypes, **rest):
            launches.append((name, header, inputs, dict(template), grid, output_shapes))
            return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes, strict=True)]

        return launch

    monkeypatch.setattr(i8, "_kernel", kernel)
    return launches


def test_qmm_reads_slab_planes_in_place(monkeypatch):
    """A slab-only module's planes go to the slab kernel as they are, with no
    per-call transposed copy of its scales, and N is its real width, which the
    padded planes no longer show; mlx's layout keeps today's kernel and copy."""
    n, k, m = 320, 256, 9
    w, scales, biases = pfx.rand_linear(n, k, seed=1)
    codes, s, b, info = repack(w, scales, biases)
    assert info == SlabInfo(n_real=n, k=k, tiles=2)
    stage = i8.stage_a(pfx.activations(m, k, seed=2))
    launches = _capture_launches(monkeypatch)
    out = i8.qmm(*stage, codes, s, b, slab=info)
    assert out.shape == (m, n)
    name, header, inputs, template, grid, shapes = launches[-1]
    assert name == "sous_int8_q4_gemm_slab"
    common, nax_h = i8._kernel_text("common.h"), i8._kernel_text("nax.h")
    assert header == common + "#define SLAB 1\n" + nax_h
    assert inputs[3] is codes and inputs[4] is s and inputs[5] is b
    assert (template["K"], template["N"]) == (k, n)
    assert grid[0] == 32 * (n // i8.BN) and shapes == [(m, n)]

    i8.qmm(*stage, w, scales, biases)
    name, header, inputs, template, grid, shapes = launches[-1]
    assert name == "sous_int8_q4_gemm" and header == common + nax_h
    assert inputs[3] is w and inputs[4].shape == (k // 64, n) and inputs[5].shape == (k // 64, n)
    assert template["N"] == n and shapes == [(m, n)]


@pytest.mark.parametrize(("k_x", "k_planes"), [(1024, 256), (256, 1024)])
def test_qmm_refuses_slab_planes_of_another_k(monkeypatch, k_x, k_planes):
    """K comes from the activations: planes repacked at another K would be read
    with the wrong group count, past their end or short of it, so the call is
    refused before any launch."""
    w, scales, biases = pfx.rand_linear(256, k_planes, seed=3)
    codes, s, b, info = repack(w, scales, biases)
    launches = _capture_launches(monkeypatch)
    qa = mx.zeros((16, k_x), dtype=mx.int8)
    sa = ra = mx.zeros((16,), dtype=mx.float32)
    with pytest.raises(ValueError, match=f"planes hold K={k_planes}, not K={k_x}"):
        i8.qmm(qa, sa, ra, codes, s, b, slab=info)
    assert launches == []


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


_reference_qmm = ptfx.reference_qmm
_either_form_qmm = ptfx.slab_reference_qmm


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


def test_rows_ok_admits_nine_rows_on_a_slab_only_module_and_not_otherwise():
    """Stock quantized_matmul cannot read a slab-only module's flat planes, so
    int8 takes its calls from nine rows (the projection kernel serves up to
    eight); every other module keeps the 128-row floor."""
    info = SlabInfo(n_real=512, k=256, tiles=2)

    def rows(n, dtype=mx.bfloat16):
        return mx.zeros((1, n, 256), dtype=dtype)

    assert (i8.SLAB_MIN_ROWS, i8.MIN_ROWS) == (9, 128)
    assert i8._rows_ok(rows(9), info) and i8._rows_ok(rows(127), info)
    assert not i8._rows_ok(rows(8), info)
    assert not i8._rows_ok(rows(9)) and not i8._rows_ok(rows(127), None)
    assert i8._rows_ok(rows(128)) and i8._rows_ok(rows(128), info)
    assert i8._rows_ok(rows(9, mx.float16), info)
    assert not i8._rows_ok(rows(9, mx.float32), info)
    assert not i8._rows_ok(mx.zeros((256,), dtype=mx.bfloat16), info)
    # A slab-only module's rows are the product of the leading dimensions, as the
    # projection kernel counts them: the two gates meet at nine rows whatever the
    # batch. Every other module keeps its floor on the last of them.
    assert i8._rows_ok(mx.zeros((2, 5, 256), dtype=mx.bfloat16), info)
    assert not i8._rows_ok(mx.zeros((2, 4, 256), dtype=mx.bfloat16), info)
    assert not i8._rows_ok(mx.zeros((2, 100, 256), dtype=mx.bfloat16))


def test_a_nine_row_mlp_call_on_a_converted_module_routes_to_int8_with_one_stage(
    monkeypatch, wrappers
):
    """A slab-only MLP's 9-to-127-row call is int8's in one pass: gate and up
    share one Stage A as above the 128-row floor, down takes its own, and the
    call counts once as int8_small. Below nine rows it is not int8's: with no
    projection kernel hooked here it reaches stock, which refuses the planes."""
    stages, qas = [], []
    real_stage_a = i8.stage_a

    def counting_stage_a(x):
        stages.append(real_stage_a(x))
        return stages[-1]

    def either(qa, *args, slab=None):
        assert slab is not None, "every MLP linear here is slab-only"
        qas.append(qa)
        return _either_form_qmm(qa, *args, slab=slab)

    monkeypatch.setattr(i8, "stage_a", counting_stage_a)
    monkeypatch.setattr(i8, "qmm", either)
    monkeypatch.setitem(i8.calls, "int8_small", 0)
    stock = pfx.tiny_quantized_model(seed=0).model.layers[0].mlp
    mlp = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp
    _tag_all(mlp)
    x = pfx.activations(9, 256, seed=5).reshape(1, 9, 256)
    routed = mlp(x)
    assert len(stages) == 2 and len(qas) == 3
    assert qas[0] is stages[0][0] and qas[1] is stages[0][0] and qas[2] is stages[1][0]
    assert i8.calls["int8_small"] == 1
    want = stock(x)
    assert routed.shape == want.shape == (1, 9, 256)
    rel = mx.sqrt(mx.mean((routed - want).astype(mx.float32) ** 2)) / mx.sqrt(
        mx.mean(want.astype(mx.float32) ** 2)
    )
    assert rel.item() < 3e-2

    with pytest.raises(ValueError):
        mx.eval(mlp(x[:, :8]))
    assert len(qas) == 3 and i8.calls["int8_small"] == 1


def test_a_nine_row_call_of_a_slab_only_linear_routes_alone(monkeypatch, wrappers):
    """The same floor where a slab-only linear is called on its own, through the
    nn.QuantizedLinear wrapper, while a tagged linear in mlx's layout keeps 128."""
    served = []

    def either(*args, slab=None):
        served.append(slab)
        return _either_form_qmm(*args, slab=slab)

    monkeypatch.setattr(i8, "qmm", either)
    monkeypatch.setitem(i8.calls, "int8_small", 0)
    converted = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp.gate_proj
    plain = pfx.rand_module(512, 256, seed=7)
    for module in (converted, plain):
        object.__setattr__(module, i8._TAG, True)
    x = pfx.activations(9, 256, seed=6).reshape(1, 9, 256)
    assert converted(x).shape == (1, 9, 512)
    assert served == [slab_info(converted)] and i8.calls["int8_small"] == 1
    assert plain(x).shape == (1, 9, 512)
    assert len(served) == 1 and i8.calls["int8_small"] == 1


def test_a_batch_of_short_calls_on_a_slab_only_linear_routes_as_its_total_rows(
    monkeypatch, wrappers
):
    """[2, 5, K] is ten rows: past the projection kernel's eight, and stock cannot
    read the planes, so int8 takes it in one GEMM over the flattened rows."""
    served = []

    def either(*args, slab=None):
        served.append((args[0].shape[0], slab))
        return _either_form_qmm(*args, slab=slab)

    monkeypatch.setattr(i8, "qmm", either)
    monkeypatch.setitem(i8.calls, "int8_small", 0)
    stock = pfx.tiny_quantized_model(seed=0).model.layers[0].mlp.gate_proj
    converted = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp.gate_proj
    object.__setattr__(converted, i8._TAG, True)
    x = pfx.activations(10, 256, seed=8).reshape(2, 5, 256)
    got = converted(x)
    assert served == [(10, slab_info(converted))] and i8.calls["int8_small"] == 1
    want = stock(x)
    assert got.shape == want.shape == (2, 5, 512)
    rel = mx.sqrt(mx.mean((got - want).astype(mx.float32) ** 2)) / mx.sqrt(
        mx.mean(want.astype(mx.float32) ** 2)
    )
    assert rel.item() < 3e-2


def test_int8_small_counts_the_nine_to_127_row_calls_and_not_the_128_row_ones(
    monkeypatch, wrappers
):
    """The projection tile's gate reads int8_small as the evidence that a slab-only
    call went to int8 on its own floor: it moves from nine rows to 127, where stock
    served the call before the conversion, and not from 128 rows up, which int8
    takes with or without it. A linear and an MLP count the same."""
    monkeypatch.setattr(i8, "qmm", _either_form_qmm)
    monkeypatch.setitem(i8.calls, "int8_small", 0)
    mlp = ptfx.tiny_slab_model(seed=0, scope="mlp").model.layers[0].mlp
    _tag_all(mlp)
    for caller in (mlp.gate_proj, mlp):
        for rows in (9, 64, 127):
            before = i8.calls["int8_small"]
            caller(pfx.activations(rows, 256, seed=rows).reshape(1, rows, 256))
            assert i8.calls["int8_small"] == before + 1, (caller, rows)
        for rows in (128, 200):
            before = i8.calls["int8_small"]
            caller(pfx.activations(rows, 256, seed=rows).reshape(1, rows, 256))
            assert i8.calls["int8_small"] == before, (caller, rows)


def test_linear_int8_hands_qmm_a_slab_only_for_a_slab_only_module(monkeypatch):
    """Only a slab-only module's call names its form, so a six-argument qmm
    stand-in keeps serving an unconverted model, and the output is reshaped to
    the real width rather than the padded planes' length."""
    seen = []

    def six(qa, sa, ra, w, scales, biases):
        seen.append(None)
        return _reference_qmm(qa, sa, ra, w, scales, biases)

    def either(*args, slab=None):
        seen.append(slab)
        return _either_form_qmm(*args, slab=slab)

    module = ptfx.padded_module(320, 256, seed=3)
    x = pfx.activations(10, 256, seed=4).reshape(1, 10, 256)
    monkeypatch.setattr(i8, "qmm", six)
    want = i8.linear_int8(module, x)
    assert want.shape == (1, 10, 320)
    codes, s, b, info = repack(module.weight, module.scales, module.biases)
    module.weight, module.scales, module.biases = codes, s, b
    object.__setattr__(module, SLAB_ATTR, info)
    monkeypatch.setattr(i8, "qmm", either)
    got = i8.linear_int8(module, x)
    assert seen == [None, info]
    assert got.shape == (1, 10, 320)
    assert mx.array_equal(got, want).item()


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


def _digest(fn):
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def test_wrappers_leave_the_wrapped_source_readable(monkeypatch):
    """A source pin hashes inspect.getsource of a method, which follows __wrapped__:
    without it, an active int8 would read as mlx or mlx-vlm having changed the
    pinned method. The class's own source is the witness no wrapper can change."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet, Qwen3_5MLP

    classes = (nn.QuantizedLinear, Qwen3_5MLP, Qwen3_5GatedDeltaNet)
    # int8 installs once per process behind these two guards: reset both and the
    # three methods, so this test wraps afresh and teardown restores what it found.
    monkeypatch.setattr(i8, "_WRAPPED", set())
    monkeypatch.setattr(i8, "_QUANTIZED_LINEAR_WRAPPED", False)
    before = {cls: cls.__dict__["__call__"] for cls in classes}
    for cls, method in before.items():
        monkeypatch.setattr(cls, "__call__", method)
    i8.install_wrappers([(Qwen3_5MLP, "mlp"), (Qwen3_5GatedDeltaNet, "gdn")])
    for cls in classes:
        wrapper = cls.__dict__["__call__"]
        assert wrapper is not before[cls], cls
        assert wrapper.__wrapped__ is before[cls], cls
        source = inspect.getsource(wrapper)
        assert source.lstrip().startswith("def __call__("), cls
        assert source in inspect.getsource(cls), cls
        assert _digest(wrapper) == _digest(before[cls]), cls


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
    def __init__(self, layers, model_type="qwen3_5"):
        super().__init__()
        # mlx-vlm keeps the architecture name on `config`, as a plain attribute.
        self.config = types.SimpleNamespace(model_type=model_type)
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


@pytest.fixture
def m5(monkeypatch):
    """A machine the platform rule passes, whatever this one is."""
    monkeypatch.setattr(i8.nax, "platform_reason", lambda: None)


def _quiet_enable(model, caplog):
    """enable() asserted to warn nothing; returns its status and the INFO text."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level("INFO", logger="sous.engine.int8prefill"):
            status = i8.enable(model, enabled=True)
    return status, caplog.text


def test_a_mac_without_the_tensor_units_is_one_info_line_not_a_warning(monkeypatch, caplog):
    """On by default, int8 prefill does not apply to most Macs: saying so is
    INFO, as the attention tile's expected refusals are. A warning would fire on
    every load for a condition nobody can act on."""
    monkeypatch.setattr(i8.nax, "platform_reason", lambda: "no tensor units")
    model = _Model([_Layer()])
    status, logged = _quiet_enable(model, caplog)
    assert status == {"state": "unavailable", "reason": "no tensor units", "routed": 0}
    assert "int8 prefill unavailable: no tensor units" in logged
    assert _tags(model) == []


def test_a_gemm_probe_failure_on_the_m5_still_warns(monkeypatch, m5):
    """The kernel should have run here and did not (a macOS update can reject
    it, #123): a silently inert default would hide that."""
    monkeypatch.setattr(
        i8, "availability", lambda: i8.Availability(False, "the int8 GEMM probe failed: no")
    )
    monkeypatch.setattr(
        i8, "_warm_up", lambda model: pytest.fail("the warm-up ran after the probe refused")
    )
    model = _Model([_Layer()])
    with pytest.warns(UserWarning, match="probe failed"):
        status = i8.enable(model, enabled=True)
    assert status["state"] == "unavailable" and _tags(model) == []


def test_enable_tags_mlp_and_gdn_but_not_attention_or_lm_head(monkeypatch, m5):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(), _Layer()])
    status = i8.enable(model, enabled=True)
    assert status == {"state": "active", "reason": None, "routed": 12}
    tagged = _tags(model)
    assert len(tagged) == 12
    assert all(".layers." in p for p in tagged)
    assert not any("self_attn" in p or "lm_head" in p or "in_proj_a" in p for p in tagged)


def test_enable_tags_an_mlp_all_or_nothing(monkeypatch, m5):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer(down_bits=8)])
    status = i8.enable(model, enabled=True)
    assert status["routed"] == 3, "only the GDN projections; the MLP with an 8-bit down stays stock"
    assert not any(".mlp." in p for p in _tags(model))


def test_a_model_int8_cannot_route_is_refused_before_the_probe(monkeypatch, caplog, m5):
    """On an M5 whose macOS rejects the kernel, a model int8 could never route
    must not pay for the probe or warn about it: its refusal is the model's."""
    probed = []

    def failing_probe():
        probed.append(1)
        return i8.Availability(False, "the int8 GEMM probe failed: no")

    monkeypatch.setattr(i8, "availability", failing_probe)
    status, logged = _quiet_enable(_Model([_Layer()], model_type="gemma3"), caplog)
    assert status["state"] == "unavailable" and "gemma3" in status["reason"]
    assert "gemma3" in logged and probed == []


def test_enable_refuses_the_moe_variant_that_reuses_the_dense_classes(monkeypatch, caplog, m5):
    """qwen3_5_moe imports Qwen3_5GatedDeltaNet and Qwen3_5MLP (its shared expert)
    from the dense module, so a class-based walk would route parts of an untested
    architecture; the gate is the model type."""
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    monkeypatch.setattr(i8, "_warm_up", lambda model: None)
    model = _Model([_Layer()], model_type="qwen3_5_moe")
    status, logged = _quiet_enable(model, caplog)
    assert "qwen3_5_moe" in logged
    assert status["state"] == "unavailable"
    assert status["reason"] is not None and "qwen3_5_moe" in status["reason"]
    assert _tags(model) == []


def test_model_type_is_read_from_config_then_model_then_args():
    assert (
        i8._model_type(types.SimpleNamespace(config=types.SimpleNamespace(model_type="a"))) == "a"
    )
    assert i8._model_type(types.SimpleNamespace(model_type="b")) == "b"
    assert i8._model_type(types.SimpleNamespace(args=types.SimpleNamespace(model_type="c"))) == "c"
    assert i8._model_type(types.SimpleNamespace()) == ""


def test_enable_with_no_eligible_projection_is_unavailable(monkeypatch, caplog, m5):
    """A checkpoint int8 cannot route (8-bit, mxfp4) is a choice of weights,
    not a fault: INFO."""
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))
    model = _Model([])
    status, logged = _quiet_enable(model, caplog)
    assert "no eligible projections" in logged
    assert status["state"] == "unavailable"
    assert status["reason"] is not None and "no eligible projections" in status["reason"]


def test_enable_degrades_when_warm_up_fails(monkeypatch, m5):
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))

    def boom(model):
        raise RuntimeError("compiler said no")

    monkeypatch.setattr(i8, "_warm_up", boom)
    model = _Model([_Layer()])
    with pytest.warns(UserWarning, match="compiler said no"):
        status = i8.enable(model, enabled=True)
    assert status == {"state": "unavailable", "reason": "compiler said no", "routed": 0}
    assert _tags(model) == []


def test_enable_reports_one_line_of_a_compiler_error(monkeypatch, m5):
    """The whole compiler output would otherwise go into the load warning and the
    status document's int8_prefill.reason."""
    monkeypatch.setattr(i8, "availability", lambda: i8.Availability(True))

    def boom(model):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(i8, "_warm_up", boom)
    with pytest.warns(UserWarning) as caught:
        status = i8.enable(_Model([_Layer()]), enabled=True)
    assert status["reason"].startswith("utils.h:544:27: error: no matching member function")
    assert "\n" not in str(caught[0].message)


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
