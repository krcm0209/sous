"""The small-row projection kernel (sous.engine.projkernel): its Metal source, the
variant choice, eligibility, availability, and the arithmetic every routing
decision rests on. The arithmetic tests run wherever the kernel compiles and
probes right (CI's macos-15 GPU included) and skip elsewhere."""

from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
np = pytest.importorskip("numpy")

from sous.engine import int8prefill, nax, projkernel  # noqa: E402 — after the guards
from tests import proj_fixtures as pfx  # noqa: E402

# ---- kernel source ---------------------------------------------------------------


def _text(name: str) -> str:
    return files("sous.engine.kernels").joinpath(name).read_text()


def test_the_kernel_file_holds_three_bodies_split_at_marker_lines():
    text = _text("projection_mma.metal")
    assert text.count("\n// STAGED\n") == 1 and text.count("\n// GROUP_SUMS\n") == 1
    assert "#include" not in text
    plain, staged, sums = projkernel._bodies()
    assert plain.count("__SELECT__") == 1 and staged.count("__SELECT__") == 1
    assert "__SELECT__" not in sums
    assert "#if PRESUM" in staged and "#if PRESUM" not in plain + sums
    # Mode h only: half weights times fp32 activations, no other operand types.
    for body in (plain, staged):
        assert "mma8<half, float>(" in body and body.count("mma8<") == 1
        assert "as_type<half2>(nib | 0x64006400u) - half2(1024.0h)" in body
    assert "mpp::" not in text


def test_the_bodies_freeze_the_constants_the_launch_assumes():
    """F, NSG and KS (and GB when staged) fix the summation order; the Python side
    sizes the grid and the variants' K rules from the same numbers."""
    plain, staged, _ = projkernel._bodies()
    for body in (plain, staged):
        assert "constexpr int F = 4;" in body
        assert "constexpr int NSG = 1;" in body
        assert "constexpr int KS = 4;" in body
    assert "constexpr int GB = 4;" in staged and "GB" not in plain
    assert (projkernel._COLS, projkernel._THREADS) == (8 * 4 * 1, 32 * 1 * 4)
    assert (projkernel._KS, projkernel._STAGED_K) == (4, 64 * 4 * 4)


def test_the_header_holds_mma8_and_the_presum_switch():
    header = _text("projection_mma.h")
    assert "inline void mma8(" in header and "simdgroup_multiply_accumulate" in header
    assert "#ifndef PRESUM\n#define PRESUM 0\n#endif" in header
    assert "MetalPerformancePrimitives" not in header


def test_the_linear_select_is_generated_per_count():
    assert projkernel._select(1) == (
        "#define W_ w0\n#define S_ s0\n#define B_ b0\n#define Y_ y0\n  const uint NL_ = N0;"
    )
    three = projkernel._select(3)
    assert "const device uint32_t *W_ = (L == 0 ? w0 : (L == 1 ? w1 : w2));" in three
    assert "device bfloat *Y_ = (L == 0 ? y0 : (L == 1 ? y1 : y2));" in three
    assert "const uint NL_ = (L == 0 ? N0 : (L == 1 ? N1 : N2));" in three
    four = projkernel._select(4)
    assert "const device bfloat *S_ = (L == 0 ? s0 : (L == 1 ? s1 : (L == 2 ? s2 : s3)));" in four


def test_every_kernel_is_built_with_safe_math_under_its_own_name(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs
        return lambda **launch: []

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    for kind in projkernel.VARIANTS:
        for nl in range(1, 5):
            projkernel._projection_kernel(kind, nl)
    projkernel._group_sums_kernel()
    assert set(made) == {
        f"sous_proj_{kind}_n{nl}" for kind in projkernel.VARIANTS for nl in range(1, 5)
    } | {"sous_proj_group_sums"}
    for name, kwargs in made.items():
        assert kwargs["compile_options"] == {"math_mode": "safe"}, name
        assert kwargs.get("ensure_row_contiguous", True) is True, name
        assert "__SELECT__" not in kwargs["source"], name
        assert kwargs["header"].startswith(_text("common.h")), name
    ps = made["sous_proj_staged_ps_n2"]
    assert ps["input_names"] == ["x", "sums", "w0", "s0", "b0", "w1", "s1", "b1"]
    assert ps["output_names"] == ["y0", "y1"]
    assert ps["header"].index("#define PRESUM 1\n") < ps["header"].index("#ifndef PRESUM")
    assert "#define PRESUM 1" not in made["sous_proj_staged_n2"]["header"]
    assert made["sous_proj_plain_n1"]["input_names"] == ["x", "w0", "s0", "b0"]
    assert made["sous_proj_group_sums"]["output_names"] == ["sums"]


# ---- the variant choice ----------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n_sum", "kind"),
    [
        (5120, 248_320, "staged_ps"),  # lm_head
        (5120, 12_288 + 1024 + 1024, "staged"),  # q+k+v
        (5120, 2 * 17_408, "staged"),  # gate+up
        (5120, 10_240 + 6144 + 48 + 48, "staged"),  # in_proj_qkv+z+b+a
        (17_408, 5120, "staged"),  # down
        (5120, 12_288, "staged"),  # q alone
        (6144, 5120, "plain"),  # o_proj, out_proj
        (5120, 6144, "plain"),  # in_proj_z alone
        (5120, 1024, "plain"),  # k or v alone
        (5120, 48, "plain"),  # in_proj_b or a alone
        (1024, 39_062, "plain"),  # just under the staging work
        (1024, 39_063, "staged"),  # just over it
        (1024, 100_000, "staged_ps"),
        (5184, 248_320, "plain"),  # K % 1024 != 0: plain whatever the width
        (1088, 50_000, "plain"),
    ],
)
def test_the_variant_follows_k_and_the_summed_width(k, n_sum, kind):
    assert projkernel.variant(k, n_sum) == kind


# ---- eligibility -----------------------------------------------------------------


def _qlinear(k, n, *, bias=False, dtype=mx.bfloat16, **quant):
    parent = nn.Sequential(nn.Linear(k, n, bias=bias))
    parent.set_dtype(dtype)
    nn.quantize(parent, **{"group_size": 64, "bits": 4, **quant})
    return parent.layers[0]


def test_eligible_takes_only_what_the_kernel_decodes():
    assert projkernel.eligible(_qlinear(256, 48)), "N need not fill a tile"
    assert projkernel.eligible(_qlinear(1024, 40))
    assert not projkernel.eligible(_qlinear(256, 64, dtype=mx.float16)), "fp16 scales"
    assert not projkernel.eligible(_qlinear(256, 64, dtype=mx.float32)), "fp32 scales"
    assert not projkernel.eligible(_qlinear(256, 64, bits=8))
    assert not projkernel.eligible(_qlinear(256, 64, group_size=128))
    assert not projkernel.eligible(_qlinear(256, 64, group_size=32, mode="mxfp4"))
    assert not projkernel.eligible(_qlinear(256, 64, bias=True))
    assert not projkernel.eligible(_qlinear(128, 64)), "K / 64 below the four-way K split"
    assert not projkernel.eligible(nn.Linear(256, 64)), "not quantized"


# ---- what linears() refuses (before any launch) ----------------------------------


@pytest.fixture
def no_launch(monkeypatch):
    def launched(*args):
        raise AssertionError("a refused call reached the kernel")

    monkeypatch.setattr(projkernel, "_launch", launched)


@pytest.mark.parametrize(
    ("shape", "dtype", "count", "match"),
    [
        ((3, 256), mx.float16, 1, "bf16"),
        ((3, 256), mx.float32, 1, "bf16"),
        ((9, 256), mx.bfloat16, 1, "rows"),
        ((3, 3, 256), mx.bfloat16, 1, "rows"),  # nine rows over two leading dims
        ((0, 256), mx.bfloat16, 1, "rows"),
        ((3, 256), mx.bfloat16, 0, "linears"),
        ((3, 256), mx.bfloat16, 5, "linears"),
        ((3, 512), mx.bfloat16, 1, "K"),
    ],
)
def test_linears_refuses_what_the_kernel_cannot_take(no_launch, shape, dtype, count, match):
    weights = [pfx.rand_linear(64, 256, 0)] * count
    with pytest.raises(ValueError, match=match):
        projkernel.linears(mx.zeros(shape, dtype=dtype), weights)


def test_linears_refuses_a_k_its_variant_cannot_split(no_launch, monkeypatch):
    """The plain body needs K / 64 >= 4 for its four K simdgroups, the staged ones
    K % 1024 == 0 for whole staging stages: a K that fits neither is refused rather
    than read past the end of the weights."""
    with pytest.raises(ValueError, match="K=128 does not fit the plain variant"):
        projkernel.linears(mx.zeros((2, 128), dtype=mx.bfloat16), [pfx.rand_linear(64, 128, 0)])
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: "staged")
    with pytest.raises(ValueError, match="K=1088 does not fit the staged variant"):
        projkernel.linears(mx.zeros((2, 1088), dtype=mx.bfloat16), [pfx.rand_linear(64, 1088, 0)])


# ---- the stand-in ----------------------------------------------------------------


def test_the_stand_in_keeps_rows_and_linears_apart():
    """What the routing tests assume of it: rows do not depend on the row count,
    and a fused call equals its linears one at a time."""
    weights = [pfx.rand_linear(n, 256, 60 + i) for i, n in enumerate((72, 48, 40))]
    x = pfx.activations(8, 256, seed=6)
    full = pfx.stand_in_mma(x[None], weights)
    assert [o.shape for o in full] == [(1, 8, 72), (1, 8, 48), (1, 8, 40)]
    assert all(o.dtype == mx.bfloat16 for o in full)
    for t in range(1, 9):
        part = pfx.stand_in_mma(x[:t], weights[1:2])
        assert mx.array_equal(part[0], full[1][0, :t]).item(), t
    with pytest.raises(ValueError, match="rows"):
        pfx.stand_in_mma(mx.zeros((9, 256), dtype=mx.bfloat16), weights)


# ---- availability ----------------------------------------------------------------


def test_kernel_available_reports_the_probes_reason(monkeypatch):
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    monkeypatch.setattr(projkernel, "_kernel_probe", lambda: "linear 0 is off its reference")
    assert projkernel.kernel_available() == nax.Availability(
        False, "the projection kernel failed: linear 0 is off its reference"
    )


def test_kernel_available_remembers_success_but_not_failure(monkeypatch):
    """One bad moment must not switch the kernel off for a long-lived daemon."""
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    calls = []

    def flaky():
        calls.append(1)
        return "transient" if len(calls) == 1 else None

    monkeypatch.setattr(projkernel, "_kernel_probe", flaky)
    assert not projkernel.kernel_available().available
    assert projkernel.kernel_available() == nax.Availability(True)
    assert projkernel.kernel_available() == nax.Availability(True)
    assert len(calls) == 2


_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:40:7: error: no matching function for call to 'mma8'\n"
    "      mma8<half, float>(dot[f], a, bf);\n"
)


def test_the_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(projkernel, "_launch", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.projkernel"):
        reason = projkernel._kernel_probe()
    assert reason == "program_source:40:7: error: no matching function for call to 'mma8'"
    assert "mma8<half, float>(dot[f], a, bf);" in caplog.text


def test_the_probe_refuses_a_kernel_that_computes_wrong(monkeypatch):
    def zeros(kind, x2, weights):
        return [mx.zeros((x2.shape[0], w.shape[0]), dtype=mx.bfloat16) for w, _, _ in weights]

    monkeypatch.setattr(projkernel, "_launch", zeros)
    reason = projkernel._kernel_probe()
    assert reason is not None and reason.startswith("linear 0 is off its reference")


def test_the_probe_refuses_a_variant_that_disagrees(monkeypatch):
    """Bitwise, not within tolerance: a staged body one rounding away from the
    plain one would break verify against decode."""

    def nudged(kind, x2, weights):
        outs = pfx.stand_in_mma(x2, weights)
        return [o * 1.0078125 if kind == "staged" else o for o in outs]  # about one bf16 ulp up

    monkeypatch.setattr(projkernel, "_launch", nudged)
    assert projkernel._kernel_probe() == "staged differs from plain in linear 0 of a 1-linear call"


def test_the_probe_refuses_a_row_that_depends_on_the_row_count(monkeypatch):
    def row_count(kind, x2, weights):
        outs = pfx.stand_in_mma(x2, weights)
        return list(outs) if x2.shape[0] == projkernel.MAX_ROWS else [o * 2 for o in outs]

    monkeypatch.setattr(projkernel, "_launch", row_count)
    assert projkernel._kernel_probe() == "plain: a row alone differs from the same row among eight"


@pfx.kernel
def test_the_real_probe_passes(monkeypatch):
    monkeypatch.setattr(projkernel, "_kernel_ok", False)
    assert projkernel._kernel_probe() is None


# ---- arithmetic (the real kernel) ------------------------------------------------


@pytest.fixture(params=projkernel.VARIANTS)
def forced(request, monkeypatch):
    """Every variant on every shape below: they share one arithmetic, so the choice
    by shape may send any call to any of them. K is a multiple of 1024 wherever this
    is used, which the staged bodies need."""
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: request.param)
    return request.param


@pfx.kernel
@pytest.mark.parametrize("k", [1024, 2048])
def test_rows_do_not_depend_on_the_row_count(forced, k):
    """Each row of a T-row call, T = 1..8, equals the same row of the eight-row
    call and the same row computed alone."""
    weights = [pfx.rand_linear(n, k, i) for i, n in enumerate((72, 48))]
    x = pfx.activations(8, k, seed=1)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        part = projkernel.linears(x[:t], weights)
        for got, want in zip(part, full, strict=True):
            assert got.shape == (t, want.shape[1]) and got.dtype == mx.bfloat16
            assert mx.array_equal(got, want[:t]).item(), (forced, k, t)
    for r in range(8):
        alone = projkernel.linears(x[r : r + 1], weights)
        for got, want in zip(alone, full, strict=True):
            assert mx.array_equal(got, want[r : r + 1]).item(), (forced, k, r)


@pfx.kernel
def test_the_plain_kernel_splits_an_uneven_k_the_same_way_for_every_row_count():
    """K / 64 = 17 does not divide by the four K simdgroups: the split is uneven
    but depends on K alone."""
    weights = [pfx.rand_linear(n, 1088, 10 + i) for i, n in enumerate((40, 48))]
    x = pfx.activations(8, 1088, seed=7)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        for got, want in zip(projkernel.linears(x[:t], weights), full, strict=True):
            assert mx.array_equal(got, want[:t]).item(), t


@pfx.kernel
@pytest.mark.parametrize("ns", [(72, 48), (40, 48, 8), (100, 48, 48, 72)])
def test_a_fused_call_equals_its_linears_one_at_a_time(forced, ns):
    weights = [pfx.rand_linear(n, 1024, 20 + i) for i, n in enumerate(ns)]
    x = pfx.activations(5, 1024, seed=2)
    fused = projkernel.linears(x, weights)
    for i, w in enumerate(weights):
        assert fused[i].shape == (5, ns[i])
        assert mx.array_equal(fused[i], projkernel.linear(x, *w)).item(), (forced, ns, i)


@pfx.kernel
@pytest.mark.parametrize("t", [1, 3, 8])
def test_the_three_variants_agree_bitwise(monkeypatch, t):
    """K = 2048 gives each K simdgroup two staging stages."""
    weights = [pfx.rand_linear(n, 2048, 30 + i) for i, n in enumerate((136, 48, 40))]
    x = pfx.activations(t, 2048, seed=3)
    outs = {}
    for kind in projkernel.VARIANTS:
        monkeypatch.setattr(projkernel, "variant", lambda k, n_sum, kind=kind: kind)
        outs[kind] = projkernel.linears(x, weights)
    for kind in ("staged", "staged_ps"):
        for got, want in zip(outs[kind], outs["plain"], strict=True):
            assert mx.array_equal(got, want).item(), (kind, t)


@pfx.kernel
def test_leading_dimensions_are_kept():
    weights = [pfx.rand_linear(n, 1024, 40 + i) for i, n in enumerate((72, 48))]
    x = pfx.activations(6, 1024, seed=8)
    flat = projkernel.linears(x, weights)
    for shape in ((1, 6, 1024), (2, 3, 1024)):
        shaped = projkernel.linears(x.reshape(shape), weights)
        for got, want in zip(shaped, flat, strict=True):
            assert got.shape == (*shape[:-1], want.shape[-1])
            assert mx.array_equal(got.reshape(want.shape), want).item()


@pfx.kernel
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_poisoned_row_leaves_the_other_rows_alone(forced, bad):
    weights = [pfx.rand_linear(n, 1024, 50 + i) for i, n in enumerate((72, 48))]
    x = pfx.activations(6, 1024, seed=4)
    clean = projkernel.linears(x, weights)
    rows = mx.arange(6)[:, None]
    cols = mx.arange(1024)[None, :]
    poisoned = mx.where((rows == 2) & (cols == 100), mx.array(bad, dtype=mx.bfloat16), x)
    got = projkernel.linears(poisoned, weights)
    keep = mx.array([0, 1, 3, 4, 5])
    for g, c in zip(got, clean, strict=True):
        assert mx.array_equal(g[keep], c[keep]).item(), (forced, bad)
        assert not mx.all(mx.isfinite(g[2])).item()


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _dequantize_f64(w, scales, biases):
    words = np.array(w)
    shifts = 4 * np.arange(8, dtype=np.uint32)
    codes = ((words[:, :, None] >> shifts) & 0xF).reshape(words.shape[0], -1)
    scale = np.repeat(_f64(scales), 64, axis=1)
    bias = np.repeat(_f64(biases), 64, axis=1)
    return codes.astype(np.float64) * scale + bias


@pfx.kernel
def test_the_kernel_rounds_like_a_float64_reference(forced):
    """Exact products, fp32 accumulation, one rounding to bf16: within half a bf16
    ulp of the float64 result plus fp32 accumulation noise, and correctly rounded
    almost everywhere. The kernel's output is evaluated before the CPU-side
    reference starts."""
    k = 2048
    weights = [pfx.rand_linear(n, k, 70 + i) for i, n in enumerate((136, 48))]
    x = pfx.activations(8, k, seed=5)
    got = projkernel.linears(x, weights)
    mx.eval(got)
    xs = _f64(x)
    for (w, scales, biases), y in zip(weights, got, strict=True):
        dense = _dequantize_f64(w, scales, biases)
        ref = xs @ dense.T
        magnitude = np.abs(xs) @ np.abs(dense).T
        y64 = _f64(y)
        assert np.all(np.abs(y64 - ref) <= np.abs(ref) * 2.0**-8 + magnitude * 2.0**-16)
        rounded = _f64(mx.array(ref.astype(np.float32)).astype(mx.bfloat16))
        assert np.mean(rounded == y64) >= 0.995, forced
