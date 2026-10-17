"""The small-row projection kernel (sous.engine.projkernel): its Metal source, the
variant choice, eligibility, availability, and the arithmetic every routing
decision rests on. The arithmetic tests run wherever the kernel compiles and
probes right (CI's macos-15 GPU included) and skip elsewhere; where Metal is and
the probe refuses, test_the_kernel_is_available_wherever_metal_is fails instead."""

import contextlib
import copy
import importlib
import inspect
import logging
import os
import types
import warnings
from importlib.resources import files
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
np = pytest.importorskip("numpy")

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
from tests import tile_fixtures as tfx  # noqa: E402

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


def test_the_mlx_layout_one_row_body_freezes_its_tile():
    """One simdgroup per column, four per threadgroup: the tile only sets the
    speed, but the launch sizes the grid from it, and its threadgroup memory
    (NSG * G * (RC + 1) fp32 values) caps the K the kernel takes at Apple's
    32 KB. Mode h only, the quarters added in the MMA kernel's order."""
    assert "#include" not in _text("projection_row.metal")
    body = projkernel._row_source()
    assert "mpp::" not in body
    assert "constexpr uint RC = 1;" in body and "constexpr uint NSG = 4;" in body
    assert "as_type<half2>(" in body and "- half2(1024.0h)" in body
    assert body.count("((acc + a1) + a2) + a3") == 1
    assert (projkernel._ROW_COLS, projkernel._ROW_THREADS) == (1 * 4, 32 * 4)
    assert 4 * (projkernel._ROW_MAX_K // 64) * (1 + 1) * 4 == 32 * 1024
    assert projkernel._ROW_MAX_K >= 17_408, "the 27B's down projection"


def test_the_bodies_carry_a_slab_switch_and_the_row_kernel_a_slab_body():
    """Slab addressing is a header define, as PRESUM is: the preprocessor cannot
    see a template argument. The MMA bodies switch only their weight, scale and
    bias addresses on it; the one-row kernel's slab body is a body of its own,
    after a marker line, with one lane per column of a 256-column tile and the
    mlx-layout body's summation order."""
    plain, staged, sums = projkernel._bodies()
    unit = "(ulong(n >> 8) * G * 256 + (n & 255)) * 32"
    for body in (plain, staged):
        assert "#if SLAB" in body and unit in body
        assert "scol[f][ulong(g) * 256]" in body and "bcol[f][ulong(g) * 256]" in body
    assert "SLAB" not in sums
    assert "#ifndef SLAB\n#define SLAB 0\n#endif" in _text("projection_mma.h")
    for template in (
        projkernel._dispatch(3, 1024, (72, 48))["template"],
        projkernel._row_dispatch(1024, 544)["template"],
        projkernel._row_dispatch(1024, 544, 3)["template"],
    ):
        assert all(name != "SLAB" for name, _ in template)
    text = _text("projection_row.metal")
    assert text.count("// SLAB\n") == 1 and text.count("\n// SLAB\n") == 1
    mlx_body, slab_body = projkernel._row_source(), projkernel._row_slab_source()
    assert text == mlx_body + "// SLAB\n" + slab_body
    assert "#include" not in slab_body and "mpp::" not in slab_body
    assert "const uint n = tile * 256 + r;" in slab_body
    assert "threadgroup float gsum[G];" in slab_body and "constexpr uint KS = 4;" in slab_body
    assert "as_type<half2>(" in slab_body and "- half2(1024.0h)" in slab_body
    assert "ulong(g) * 8192" in slab_body
    assert slab_body.count("((part[0] + part[1]) + part[2]) + part[3]") == 1
    assert "if (n < ND)" in slab_body
    assert projkernel._ROW_SLAB_THREADS == slab.TILE


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
            projkernel._projection_kernel(kind, nl, True)
    projkernel._group_sums_kernel()
    projkernel._row_kernel()
    projkernel._row_kernel(True)
    assert set(made) == {
        f"sous_proj_{kind}{form}_n{nl}"
        for kind in projkernel.VARIANTS
        for form in ("", "_slab")
        for nl in range(1, 5)
    } | {"sous_proj_group_sums", "sous_proj_row", "sous_proj_row_slab"}
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
    # The slab bodies are the same source under a header define, never a template
    # argument: the preprocessor cannot see one.
    for kind in projkernel.VARIANTS:
        for nl in range(1, 5):
            mlx_form, slab_form = (made[f"sous_proj_{kind}{f}_n{nl}"] for f in ("", "_slab"))
            assert slab_form["source"] == mlx_form["source"]
            assert slab_form["input_names"] == mlx_form["input_names"]
            header = slab_form["header"]
            assert header.index("#define SLAB 1\n") < header.index("#ifndef SLAB")
            assert "#define SLAB 1" not in mlx_form["header"]
    assert "#define PRESUM 1\n" in made["sous_proj_staged_ps_slab_n2"]["header"]
    assert "#define SLAB 1" not in made["sous_proj_group_sums"]["header"]
    row, row_slab = made["sous_proj_row"], made["sous_proj_row_slab"]
    for kernel in (row, row_slab):
        names = (kernel["input_names"], kernel["output_names"])
        assert names == (["x", "w", "scales", "biases"], ["y"])
        assert kernel["header"] == _text("common.h")
    assert row["source"] == projkernel._row_source()
    assert row_slab["source"] == projkernel._row_slab_source()


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
    assert projkernel.eligible(_qlinear(256, 2)), "eight scales, the fewest mlx binds as a buffer"
    assert not projkernel.eligible(_qlinear(256, 1)), "four scales: mlx binds them constant"
    assert not projkernel.eligible(nn.Linear(256, 64)), "not quantized"


# ---- what linears() refuses (before any launch) ----------------------------------


@pytest.fixture
def no_launch(monkeypatch):
    def launched(*args):
        raise AssertionError("a refused call reached the kernel")

    monkeypatch.setattr(projkernel, "_launch", launched)
    monkeypatch.setattr(projkernel, "one_row", launched)


@pytest.mark.parametrize(
    ("shape", "dtype", "count", "match"),
    [
        ((3, 256), mx.float16, 1, "bf16"),
        ((3, 256), mx.float32, 1, "bf16"),
        ((1, 256), mx.float16, 1, "bf16"),
        ((1, 256), mx.bfloat16, 5, "linears"),
        ((1, 512), mx.bfloat16, 1, "K"),
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
    for rows in (1, 2):
        with pytest.raises(ValueError, match="K=128 does not fit the plain variant"):
            projkernel.linears(
                mx.zeros((rows, 128), dtype=mx.bfloat16), [pfx.rand_linear(64, 128, 0)]
            )
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: "staged")
    with pytest.raises(ValueError, match="K=1088 does not fit the staged variant"):
        projkernel.linears(mx.zeros((2, 1088), dtype=mx.bfloat16), [pfx.rand_linear(64, 1088, 0)])


def test_linears_refuses_scales_mlx_would_bind_constant(no_launch):
    """mlx hands a kernel any input of fewer than eight elements in the constant
    address space, which the bodies' device pointers cannot take: a one-column
    linear at K = 256 has four scales and is refused here, not by the compiler."""
    with pytest.raises(ValueError, match="4 scales are fewer than 8"):
        projkernel.linears(mx.zeros((2, 256), dtype=mx.bfloat16), [pfx.rand_linear(1, 256, 0)])


# ---- which kernel linears() launches ---------------------------------------------


@pytest.fixture
def launches(monkeypatch):
    """Both kernels replaced by recording stand-ins: ("mma", rows, linears) per
    MMA launch, ("row", x shape, N) per one-row launch."""
    seen: list = []

    def mma_launch(kind, x2, weights):
        seen.append(("mma", x2.shape[0], len(weights)))
        return list(pfx.stand_in_mma(x2, weights))

    def row(x2, w, scales, biases):
        seen.append(("row", x2.shape, w.shape[0]))
        return pfx.stand_in_one_row(x2, w, scales, biases)

    monkeypatch.setattr(projkernel, "_launch", mma_launch)
    monkeypatch.setattr(projkernel, "one_row", row)
    return seen


@pytest.mark.parametrize("shape", [(1, 256), (1, 1, 256)])
def test_linears_sends_one_row_to_the_one_row_kernel_once_per_linear(launches, shape):
    """A fused call at one row is one one-row dispatch per linear: a row's bits
    do not depend on the linears beside it, so nothing is lost by splitting."""
    ns = (72, 48, 40)
    weights = [pfx.rand_linear(n, 256, 80 + i) for i, n in enumerate(ns)]
    before = dict(projkernel.calls)
    outs = projkernel.linears(_acts(*shape), weights)
    assert launches == [("row", (1, 256), n) for n in ns]
    assert _proj_counts_since(before) == {"one_row": 3}
    assert [o.shape for o in outs] == [(*shape[:-1], n) for n in ns]


def test_linears_keeps_two_rows_and_more_on_the_mma_kernel(launches):
    weights = [pfx.rand_linear(n, 256, 90 + i) for i, n in enumerate((72, 48))]
    before = dict(projkernel.calls)
    for t in range(2, projkernel.MAX_ROWS + 1):
        projkernel.linears(_acts(1, t, 256), weights)
    assert launches == [("mma", t, 2) for t in range(2, projkernel.MAX_ROWS + 1)]
    assert _proj_counts_since(before) == {}


def test_linears_keeps_a_row_too_wide_for_the_one_row_kernel_on_the_mma_kernel(launches):
    """Past _ROW_MAX_K the one-row kernel's group dots would not fit Apple's
    32 KB of threadgroup memory; the MMA kernel computes the same bits."""
    k = projkernel._ROW_MAX_K + 64
    linear = (
        mx.zeros((8, k // 8), dtype=mx.uint32),
        mx.zeros((8, k // 64), dtype=mx.bfloat16),
        mx.zeros((8, k // 64), dtype=mx.bfloat16),
    )
    projkernel.linears(mx.zeros((1, k), dtype=mx.bfloat16), [linear])
    assert launches == [("mma", 1, 1)]


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


def _stand_in_launch(kind, x2, weights):
    return list(pfx.stand_in_mma(x2, weights))


def _nudged_row(*, when=None):
    """A one-row stand-in about one bf16 ulp off the MMA kernel's row, on every
    row or only where `when(x2)` holds."""

    def row(x2, w, scales, biases):
        out = pfx.stand_in_one_row(x2, w, scales, biases)
        if when is not None and not when(x2):
            return out
        return (out.astype(mx.float32) * 1.0078125).astype(mx.bfloat16)

    return row


def test_the_probe_refuses_a_one_row_kernel_that_differs_from_the_mma_kernel(monkeypatch):
    """Bitwise, at one row: decode's row must be the row the MMA kernel serves
    a verify, or greedy parity is gone."""
    monkeypatch.setattr(projkernel, "_launch", _stand_in_launch)
    monkeypatch.setattr(projkernel, "_launch_row", _nudged_row())
    assert projkernel._kernel_probe() == (
        "one-row: linear 0 differs from the MMA kernel at one row (random activations)"
    )


def _big_in(layout):
    """True for a row in `layout`: its large values all lie on the layout's
    source and mirror columns. No layout's columns contain another's, so a row
    in one layout is never taken for another, and a random row has none."""

    def big(x2):
        source, mirror, _ = projkernel._mirror(x2.shape[-1], layout)
        large = mx.abs(x2[0].astype(mx.float32)) > 64
        return mx.any(large).item() and not mx.any(large & ~(source | mirror)).item()

    return big


@pytest.mark.parametrize("layout", projkernel.CANCELLING)
def test_the_probe_refuses_a_one_row_kernel_off_only_where_a_layout_cancels(monkeypatch, layout):
    monkeypatch.setattr(projkernel, "_launch", _stand_in_launch)
    monkeypatch.setattr(projkernel, "_launch_row", _nudged_row(when=_big_in(layout)))
    assert projkernel._kernel_probe() == (
        "one-row: linear 0 differs from the MMA kernel at one row "
        f"({layout}-cancelling activations)"
    )


def test_the_kernel_is_available_wherever_metal_is():
    """The arithmetic tests below skip when kernel_available() is False, and its
    probe answers False for wrong numbers as well as for no Metal: a kernel that
    computes wrong must fail here rather than turn them into skips. On a GPU the
    kernel has not been probed on, SOUS_TEST_SKIP_KERNEL_CHECK=1 skips this test
    (the arithmetic tests then skip with the probe's reason)."""
    if not mx.metal.is_available():
        pytest.skip("no Metal")
    if os.environ.get("SOUS_TEST_SKIP_KERNEL_CHECK") == "1":
        pytest.skip("SOUS_TEST_SKIP_KERNEL_CHECK=1")
    assert pfx.KERNEL.available, pfx.KERNEL.reason


# ---- arithmetic (the real kernel) ------------------------------------------------


@pytest.fixture(params=projkernel.VARIANTS)
def forced(request, monkeypatch):
    """Every variant on every shape below: they share one arithmetic, so the choice
    by shape may send any call to any of them. K is a multiple of 1024 wherever this
    is used, which the staged bodies need."""
    monkeypatch.setattr(projkernel, "variant", lambda k, n_sum: request.param)
    return request.param


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize("k", [1024, 2048])
def test_rows_do_not_depend_on_the_row_count(forced, k, layout):
    """Each row of a T-row call, T = 1..8, equals the same row of the eight-row
    call and the same row computed alone."""
    x, weights = pfx.projection_inputs(layout, 8, k, (72, 48), x_seed=1, w_seed=0)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        part = projkernel.linears(x[:t], weights)
        for got, want in zip(part, full, strict=True):
            assert got.shape == (t, want.shape[1]) and got.dtype == mx.bfloat16
            assert mx.array_equal(got, want[:t]).item(), (forced, layout, k, t)
    for r in range(8):
        alone = projkernel.linears(x[r : r + 1], weights)
        for got, want in zip(alone, full, strict=True):
            assert mx.array_equal(got, want[r : r + 1]).item(), (forced, layout, k, r)


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
def test_the_plain_kernel_splits_an_uneven_k_the_same_way_for_every_row_count(layout):
    """K / 64 = 17 does not divide by the four K simdgroups: the split is uneven
    but depends on K alone."""
    x, weights = pfx.projection_inputs(layout, 8, 1088, (40, 48), x_seed=7, w_seed=10)
    full = projkernel.linears(x, weights)
    for t in range(1, 9):
        for got, want in zip(projkernel.linears(x[:t], weights), full, strict=True):
            assert mx.array_equal(got, want[:t]).item(), (layout, t)


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize("ns", [(72, 48), (40, 48, 8), (100, 48, 48, 72)])
def test_a_fused_call_equals_its_linears_one_at_a_time(forced, ns, layout):
    x, weights = pfx.projection_inputs(layout, 5, 1024, ns, x_seed=2, w_seed=20)
    fused = projkernel.linears(x, weights)
    for i, w in enumerate(weights):
        assert fused[i].shape == (5, ns[i])
        assert mx.array_equal(fused[i], projkernel.linear(x, *w)).item(), (forced, layout, ns, i)


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize("t", [1, 3, 8])
def test_the_three_variants_agree_bitwise(monkeypatch, t, layout):
    """K = 2048 gives each K simdgroup two staging stages. Decode and verify can
    serve one projection on different variants (the choice reads the summed N of
    the call), so a variant that adds its K partials or its group sums in another
    order breaks greedy parity: the cancelling layouts are what show it."""
    x, weights = pfx.projection_inputs(layout, t, 2048, (136, 48, 40), x_seed=3, w_seed=30)
    outs = {}
    for kind in projkernel.VARIANTS:
        monkeypatch.setattr(projkernel, "variant", lambda k, n_sum, kind=kind: kind)
        outs[kind] = projkernel.linears(x, weights)
    for kind in ("staged", "staged_ps"):
        for got, want in zip(outs[kind], outs["plain"], strict=True):
            assert mx.array_equal(got, want).item(), (kind, layout, t)


# (K, the linears' N) for the one-row kernel against the MMA kernel: K from the
# fewest groups the K split takes to the 27B's down projection, an uneven split
# (1088) among them; every K % 1024 == 0 is checked against all three MMA
# variants. N covers partial column tiles (of the one-row kernel's 4 and the MMA
# kernel's 32), in_proj_b/a's 32 and 48, and the fewest scales mlx binds as a
# buffer.
_ONE_ROW_SHAPES = [
    (256, (2, 32, 48)),
    (512, (1, 37)),
    (1088, (48, 40)),
    (1024, (8, 33, 72)),
    (5120, (48, 130)),
    (6144, (32, 36)),
    (17_408, (32, 45)),
]


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize(("k", "ns"), _ONE_ROW_SHAPES)
def test_the_one_row_kernel_equals_the_mma_kernel_at_one_row(k, ns, layout):
    x, weights = pfx.projection_inputs(layout, 3, k, ns, x_seed=k, w_seed=k + 1)
    kinds = projkernel.VARIANTS if k % 1024 == 0 else ("plain",)
    for r in range(3):
        row = x[r : r + 1]
        for (w, scales, biases), n in zip(weights, ns, strict=True):
            got = projkernel._launch_row(row, w, scales, biases)
            assert got.shape == (1, n) and got.dtype == mx.bfloat16
            for kind in kinds:
                (want,) = projkernel._launch(kind, row, [(w, scales, biases)])
                assert mx.array_equal(got, want).item(), (k, n, layout, r, kind)


@pfx.kernel
def test_the_one_row_kernel_equals_the_mma_kernel_on_a_real_width_linear():
    """The 27B's gate+up width on random rows, a backstop beside the cancelling
    layouts: across tens of thousands of outputs a reordered epilogue or quarter
    sum, which random rows at the narrow shapes round alike, lands a few of them
    on another bf16."""
    k, n, rows = 5120, 17_408, 4
    x = pfx.activations(rows, k, 50)
    weights = pfx.rand_linear(n, k, 51)
    got = mx.concatenate([projkernel._launch_row(x[r : r + 1], *weights) for r in range(rows)])
    (want,) = projkernel._launch("plain", x, [weights])
    assert mx.array_equal(got, want).item()


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
def test_a_fused_one_row_call_equals_the_mma_kernels_fused_call(forced, layout):
    """linears() at one row against the dispatch it made before it had a
    one-row kernel: the MMA kernel's fused call at T = 1, in each variant."""
    ns = (136, 48, 40, 8)
    x, weights = pfx.projection_inputs(layout, 1, 2048, ns, x_seed=13, w_seed=14)
    before = dict(projkernel.calls)
    got = projkernel.linears(x[None], weights)
    assert _proj_counts_since(before) == {"one_row": len(ns)}
    want = projkernel._launch(forced, x, weights)
    for g, w, n in zip(got, want, ns, strict=True):
        assert g.shape == (1, 1, n)
        assert mx.array_equal(g[0], w).item(), (forced, layout, n)


_DOT_ORDER = "for (uint h = 0; h < 2; ++h)\n#pragma unroll\n        for (uint s = 0; s < 4; ++s)"
_QUARTERS = "((acc + a1) + a2) + a3"
_EPILOGUE = (
    "acc = fma(dots[sg][g][r], float(sc[g]), acc);\n"
    "        acc = fma(dots[sg][g][RC], float(bi[g]), acc);"
)
_LANE_SUM = (
    "L[fm] = ((xf[o] + xf[o + 1]) + (xf[o + 2] + xf[o + 3])) +\n"
    "              ((xf[o + 8] + xf[o + 9]) + (xf[o + 10] + xf[o + 11]));"
)
# name -> (the kernel's text, a reordered copy, the cancelling layout that shows
# it). Every one of these changes bits on a real-width random linear, and random
# rows at the probe's widths show none of them.
_REORDERED = {
    # the first and last quarters, which cancel, added before the middle two
    "quarters": (_QUARTERS, "((acc + a3) + a1) + a2", "quarter"),
    "paired quarters": (_QUARTERS, "(acc + a1) + (a2 + a3)", "middle"),
    "reversed quarters": (_QUARTERS, "((a3 + a2) + a1) + acc", "middle"),
    # a group's products with s outer and h inner
    "products": (
        _DOT_ORDER,
        "for (uint s = 0; s < 4; ++s)\n#pragma unroll\n        for (uint h = 0; h < 2; ++h)",
        "group",
    ),
    "group sum": (
        "((L[0] + L[1]) + (L[2] + L[3])) + ((L[4] + L[5]) + (L[6] + L[7]))",
        "((L[0] + L[2]) + (L[1] + L[3])) + ((L[4] + L[6]) + (L[5] + L[7]))",
        "group",
    ),
    "lane sum": (
        _LANE_SUM,
        "L[fm] = ((((((xf[o] + xf[o + 1]) + xf[o + 2]) + xf[o + 3]) + xf[o + 8]) + "
        "xf[o + 9]) + xf[o + 10]) + xf[o + 11];",
        "lane",
    ),
    "bias first": (
        _EPILOGUE,
        "acc = fma(dots[sg][g][RC], float(bi[g]), acc);\n"
        "        acc = fma(dots[sg][g][r], float(sc[g]), acc);",
        "pair",
    ),
    # stage by stage this reads as the same sum, but the products round apart
    "unfused epilogue": (
        _EPILOGUE,
        "{ const float p = dots[sg][g][r] * float(sc[g]);\n"
        "        const float b = dots[sg][g][RC] * float(bi[g]);\n"
        "        acc = (acc + p) + b; }",
        "pair",
    ),
}


@pfx.kernel
@pytest.mark.parametrize("name", list(_REORDERED))
def test_a_one_row_kernel_that_sums_in_another_order_is_caught(monkeypatch, name):
    """The negative control: the same kernel with one sum reordered. Its
    cancelling layout tells it apart from the MMA kernel, and
    kernel_available()'s probe refuses it. Built under another name, since mlx
    keeps a compiled library per kernel name."""
    order, other, layout = _REORDERED[name]
    text = projkernel._row_source()
    assert text.count(order) == 1
    monkeypatch.setattr(int8prefill, "_kernels", {})
    slug = name.replace(" ", "_")
    monkeypatch.setattr(projkernel, "_ROW_KERNEL", f"sous_proj_row_reordered_{slug}")
    reordered = text.replace(order, other)
    monkeypatch.setattr(projkernel, "_row_source", lambda: reordered)
    x, ((w, scales, biases),) = pfx.projection_inputs(layout, 8, 1024, (256,), x_seed=11, w_seed=12)
    got = mx.concatenate(
        [projkernel._launch_row(x[r : r + 1], w, scales, biases) for r in range(8)]
    )
    (want,) = projkernel._launch("plain", x, [(w, scales, biases)])
    assert not mx.array_equal(got, want).item()
    reason = projkernel._kernel_probe()
    assert reason is not None and reason.startswith("one-row: linear "), reason
    assert reason.endswith(
        f"differs from the MMA kernel at one row ({layout}-cancelling activations)"
    )


@pfx.kernel
@pytest.mark.parametrize(("n", "k"), [(2, 256), (1, 512)])
def test_the_smallest_eligible_projections_compile_and_compute(n, k):
    """Eight scales is the floor eligible() sets: were mlx to bind an array of
    that size constant too, these would fail in the compiler."""
    x = pfx.activations(3, k, 5)
    weights = [pfx.rand_linear(n, k, 6)]
    (got,) = projkernel.linears(x, weights)
    (want,) = pfx.stand_in_mma(x, weights)
    assert projkernel._stock_distance(got, want) <= projkernel.REL_RMS_BOUND


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


@pytest.mark.parametrize("k", [256, 1088, 2048])
@pytest.mark.parametrize("layout", projkernel.CANCELLING)
def test_the_cancelling_layouts_cancel_term_by_term(layout, k):
    """What makes them sensitive to summation order: each mirror column's products
    are exactly the negated products of its source column (same codes, same
    scales, negated activations), and they dwarf the rest of the sum."""
    x, ((w, scales, biases),) = pfx.projection_inputs(layout, 4, k, (40,), x_seed=9, w_seed=90)
    xs, dense = _f64(x), _dequantize_f64(w, scales, biases)
    terms = xs[:, None, :] * dense[None, :, :]
    source, mirror, offset = (np.array(m) for m in projkernel._mirror(k, layout))
    src = np.flatnonzero(source)
    assert src.size and np.array_equal(np.flatnonzero(mirror), src + offset)
    assert not np.any(source & mirror)
    assert np.array_equal(terms[..., src + offset], -terms[..., src])
    big = np.abs(terms[..., source | mirror]).sum(axis=-1)
    rest = np.abs(terms[..., ~(source | mirror)]).sum(axis=-1)
    assert np.all(big > 100 * rest)


@pytest.mark.parametrize(
    ("layout", "first", "second"),
    [("quarter", 0, 3), ("middle", 1, 2)],
)
@pytest.mark.parametrize("k", [256, 1088, 2048])
def test_the_quarter_layouts_cancel_one_k_quarter_against_another(layout, first, second, k):
    """Split as the kernels split K: whole quant groups, g0 = q * G // 4."""
    groups = k // 64
    starts = [64 * (q * groups // 4) for q in range(5)]
    source, mirror, _ = (np.flatnonzero(np.array(m)) for m in projkernel._mirror(k, layout))
    assert source[0] == starts[first] and source[-1] < starts[first + 1]
    assert mirror[0] == starts[second] and mirror[-1] < starts[second + 1]


# ---- the slab form ----------------------------------------------------------------


def _slab_entry(w, scales, biases):
    """mlx's arrays repacked into slab planes, as the descriptor a slab-only module
    hands the kernel."""
    codes, s, b, info = slab.repack(w, scales, biases)
    return projkernel.Weights(codes, s, b, info)


def _identical(a, b) -> bool:
    """Shape, dtype and every bit: a bf16 -0.0 is not +0.0 here."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize(
    ("k", "n"),
    [(1024, 256), (1024, 544), (2048, 1024)],
    ids=["k1024-n256", "k1024-n544", "k2048-n1024"],
)
def test_slab_bodies_equal_mlx_bodies(forced, layout, k, n):
    """The slab bodies change where a (column, group) unit is read and nothing
    else, so every row equals the mlx-layout body's bit for bit at every row
    count, in every variant. N = 544 pads its last tile, and the second linear
    (72 columns) pads too and is reached through the multi-linear selection;
    linears() routes both forms alike, the one-row kernel's at T = 1 included."""
    ns = (n, 72)
    x, weights = pfx.projection_inputs(layout, 8, k, ns, x_seed=k + n, w_seed=n)
    slabs = [_slab_entry(*w) for w in weights]
    for t in range(1, projkernel.MAX_ROWS + 1):
        want = projkernel._launch(forced, x[:t], weights)
        got = projkernel._launch(forced, x[:t], slabs)
        for g, w, width in zip(got, want, ns, strict=True):
            assert g.shape == (t, width), (forced, t)
            assert _identical(g, w), (forced, layout, k, n, t)
        routed = projkernel.linears(x[:t], slabs)
        for g, w in zip(routed, projkernel.linears(x[:t], weights), strict=True):
            assert _identical(g, w), (forced, layout, k, n, t)


@pfx.kernel
@pytest.mark.parametrize("layout", pfx.LAYOUTS)
@pytest.mark.parametrize(
    ("k", "n"),
    [(1024, 256), (1024, 544), (1088, 48)],
    ids=["k1024-n256", "k1024-n544", "k1088-n48"],
)
def test_the_slab_one_row_kernel_equals_the_mlx_one_row_kernel(k, n, layout):
    """One lane per column over slab planes, in the mlx-layout body's order:
    the quarters, each group's dot chain from +0, the two-fma epilogue. K = 1088
    splits its 17 groups unevenly; N = 544 and 48 pad their last tile."""
    x, ((w, scales, biases),) = pfx.projection_inputs(layout, 3, k, (n,), x_seed=k, w_seed=n)
    codes, s, b, info = _slab_entry(w, scales, biases)
    for r in range(3):
        row = x[r : r + 1]
        got = projkernel._launch_row(row, codes, s, b, slab=info)
        assert got.shape == (1, n) and got.dtype == mx.bfloat16
        assert _identical(got, projkernel._launch_row(row, w, scales, biases)), (k, n, layout, r)


@pfx.kernel
def test_the_slab_one_row_kernel_equals_the_mlx_one_row_kernel_on_a_real_width_linear():
    """The 27B's gate width on random rows, the backstop the mlx-layout one-row
    kernel has: across tens of thousands of outputs a reordered sum lands a few
    on another bf16."""
    k, n, rows = 5120, 17_408, 4
    x = pfx.activations(rows, k, 52)
    weights = pfx.rand_linear(n, k, 53)
    codes, s, b, info = _slab_entry(*weights)
    got = mx.concatenate(
        [projkernel._launch_row(x[r : r + 1], codes, s, b, slab=info) for r in range(rows)]
    )
    want = mx.concatenate([projkernel._launch_row(x[r : r + 1], *weights) for r in range(rows)])
    assert _identical(got, want)


_SLAB_QUARTERS = "((part[0] + part[1]) + part[2]) + part[3]"
_SLAB_EPILOGUE = (
    "acc = fma(dot, float(sc[ulong(g) * 256]), acc);\n"
    "      acc = fma(gsum[g], float(bi[ulong(g) * 256]), acc);"
)
# The negative control's reorders as the slab body spells them: the products, the
# group sum and the lane sum read as they do in the mlx-layout body.
_SLAB_REORDERED = {
    "quarters": (_SLAB_QUARTERS, "((part[0] + part[3]) + part[1]) + part[2]", "quarter"),
    "paired quarters": (_SLAB_QUARTERS, "(part[0] + part[1]) + (part[2] + part[3])", "middle"),
    "reversed quarters": (_SLAB_QUARTERS, "((part[3] + part[2]) + part[1]) + part[0]", "middle"),
    "products": _REORDERED["products"],
    "group sum": _REORDERED["group sum"],
    "lane sum": _REORDERED["lane sum"],
    "bias first": (
        _SLAB_EPILOGUE,
        "acc = fma(gsum[g], float(bi[ulong(g) * 256]), acc);\n"
        "      acc = fma(dot, float(sc[ulong(g) * 256]), acc);",
        "pair",
    ),
    "unfused epilogue": (
        _SLAB_EPILOGUE,
        "{ const float p = dot * float(sc[ulong(g) * 256]);\n"
        "      const float pb = gsum[g] * float(bi[ulong(g) * 256]);\n"
        "      acc = (acc + p) + pb; }",
        "pair",
    ),
}


@pfx.kernel
@pytest.mark.parametrize("name", list(_SLAB_REORDERED))
def test_a_slab_one_row_kernel_that_sums_in_another_order_is_caught(monkeypatch, name):
    """The slab body's negative control: with one sum reordered, its layout tells
    it apart from the mlx-layout one-row kernel, so the equality above is not
    vacuous for the slab body either."""
    order, other, layout = _SLAB_REORDERED[name]
    text = projkernel._row_slab_source()
    assert text.count(order) == 1
    monkeypatch.setattr(int8prefill, "_kernels", {})
    slug = name.replace(" ", "_")
    monkeypatch.setattr(projkernel, "_ROW_SLAB_KERNEL", f"sous_proj_row_slab_reordered_{slug}")
    reordered = text.replace(order, other)
    monkeypatch.setattr(projkernel, "_row_slab_source", lambda: reordered)
    x, ((w, scales, biases),) = pfx.projection_inputs(layout, 8, 1024, (256,), x_seed=11, w_seed=12)
    codes, s, b, info = _slab_entry(w, scales, biases)
    got = mx.concatenate(
        [projkernel._launch_row(x[r : r + 1], codes, s, b, slab=info) for r in range(8)]
    )
    want = mx.concatenate(
        [projkernel._launch_row(x[r : r + 1], w, scales, biases) for r in range(8)]
    )
    assert not mx.array_equal(got, want).item()


def test_weights_is_a_bare_tuple_for_an_mlx_module_and_a_descriptor_for_a_slab_module():
    """Every stand-in, probe and test that unpacks three fields keeps working on
    an unconverted model; only a slab-only module's entry carries its SlabInfo."""
    lm = ptfx.tiny_slab_model(scope="mlp")
    q, gate = lm.model.layers[1].self_attn.q_proj, lm.model.layers[1].mlp.gate_proj
    mlx_entry = projkernel._weights(q)
    assert type(mlx_entry) is tuple and len(mlx_entry) == 3
    assert all(a is b for a, b in zip(mlx_entry, (q.weight, q.scales, q.biases), strict=True))
    slab_entry = projkernel._weights(gate)
    assert isinstance(slab_entry, projkernel.Weights)
    assert slab_entry.slab == slab.slab_info(gate) == slab.SlabInfo(n_real=512, k=256, tiles=2)
    assert slab_entry.w is gate.weight and slab_entry.w.ndim == 1
    assert projkernel._as_weights(mlx_entry) == projkernel.Weights(*mlx_entry, None)
    assert projkernel._as_weights(slab_entry) is slab_entry


def test_linears_refuses_a_call_mixing_slab_and_mlx_forms(no_launch):
    """A dispatch compiles one form: the slab define addresses every linear of it."""
    weights = [pfx.rand_linear(n, 256, 100 + i) for i, n in enumerate((256, 72))]
    mixed = [_slab_entry(*weights[0]), weights[1]]
    for rows in (1, 3):
        with pytest.raises(ValueError, match="mixes slab-ordered and mlx-layout"):
            projkernel.linears(mx.zeros((rows, 256), dtype=mx.bfloat16), mixed)


def test_linears_refuses_slab_planes_off_their_descriptor(no_launch):
    """The planes are flat: only the descriptor says what they hold, and planes of
    another K or width would be read out of bounds."""
    entry = _slab_entry(*pfx.rand_linear(256, 512, 110))
    with pytest.raises(ValueError, match="K=256"):
        projkernel.linears(mx.zeros((3, 256), dtype=mx.bfloat16), [entry])
    short = entry._replace(w=entry.w[: entry.w.size // 2])
    with pytest.raises(ValueError, match="planes"):
        projkernel.linears(mx.zeros((3, 512), dtype=mx.bfloat16), [short])


def test_the_descriptor_gives_k_and_n_for_a_converted_module():
    """A slab-only module's planes are flat, so K and N come from its SlabInfo; the
    dispatch key also names the form, since the slab bodies are pipelines of their
    own."""
    lm = ptfx.tiny_slab_model(scope="mlp")
    layer = lm.model.layers[1]
    mlp, attention = layer.mlp, layer.self_attn
    assert projkernel._in_features(mlp.gate_proj) == 256
    assert projkernel._in_features(mlp.down_proj) == 512
    assert projkernel._dispatch_key((mlp.gate_proj, mlp.up_proj)) == (256, (512, 512), True)
    assert projkernel._dispatch_key((mlp.down_proj,)) == (512, (256,), True)
    assert projkernel._dispatch_key((attention.q_proj, attention.k_proj, attention.v_proj)) == (
        256,
        (2048, 512, 512),
        False,
    )
    assert projkernel._dispatch_key((attention.o_proj,)) == (1024, (256,), False)


def test_the_stand_ins_accept_a_slab_entry():
    """The stand-ins read a slab entry through slab.unpack, so they compute what
    they compute over mlx's arrays, bit for bit, and refuse a mixed call as the
    kernel does."""
    weights = [pfx.rand_linear(n, 256, 120 + i) for i, n in enumerate((544, 72))]
    slabs = [_slab_entry(*w) for w in weights]
    x = pfx.activations(5, 256, seed=12)
    for got, want in zip(pfx.stand_in_mma(x, slabs), pfx.stand_in_mma(x, weights), strict=True):
        assert _identical(got, want)
    codes, s, b, info = slabs[0]
    got = pfx.stand_in_one_row(x[:1], codes, s, b, slab=info)
    assert _identical(got, pfx.stand_in_one_row(x[:1], *weights[0]))
    with pytest.raises(ValueError, match="mixes"):
        pfx.stand_in_mma(x, [slabs[0], weights[1]])


# ---- routing: the tags, the flag and the two hooks ---------------------------------


def _proj_counts_since(before):
    """The counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in projkernel.calls.items() if n != before[k]}


def _acts(*shape, dtype=None, seed=0):
    """Random activations of `shape`, bf16 unless `dtype` says otherwise."""
    x = mx.random.normal(shape, key=mx.random.key(seed))
    return x.astype(mx.bfloat16 if dtype is None else dtype)


def _tagged_linear(n, k, seed):
    linear = pfx.rand_module(n, k, seed)
    object.__setattr__(linear, projkernel._TAG, True)
    return linear


def _triples(*linears):
    return [(m.weight, m.scales, m.biases) for m in linears]


def _bitwise(a, b):
    return mx.array_equal(a, b).item()


def _no_kernel(x, weights):
    raise AssertionError("a call that should have been handed on reached the kernel")


@pytest.fixture(scope="module")
def tiny_target():
    return pfx.tiny_quantized_model()


@pytest.fixture
def fresh_target(tiny_target):
    """The tiny quantized model, untagged again after the test."""
    yield tiny_target
    projkernel.untag(tiny_target)


def test_the_counters_name_the_eight_paths():
    assert set(projkernel.calls) == {
        "draft_kernel",
        "verify_kernel",
        "verify_split",
        "verify_tile",
        "plain_kernel",
        "plain_tile",
        "declined",
        "one_row",
    }


@pfx.kernel
def test_a_one_row_call_through_hook_two_is_a_one_row_dispatch(monkeypatch):
    pfx.hooked(monkeypatch)
    pfx.real_kernel(monkeypatch)
    linear = _tagged_linear(256, 256, 0)
    x = _acts(1, 1, 256)
    before = dict(projkernel.calls)
    got = linear(x)
    assert _proj_counts_since(before) == {"plain_kernel": 1, "one_row": 1}
    (want,) = projkernel._launch("plain", x[0], _triples(linear))
    assert _bitwise(got[0], want)
    before = dict(projkernel.calls)
    linear(_acts(1, 3, 256))
    assert _proj_counts_since(before) == {"plain_kernel": 1}


def test_clear_drops_both_flags(monkeypatch):
    monkeypatch.setattr(projkernel, "_active", 1)
    monkeypatch.setattr(projkernel, "_draft_active", 1)
    projkernel.clear()
    assert projkernel._active == 0
    assert projkernel._draft_active == 0


def test_tag_marks_every_quantized_linear_of_the_language_model_and_nothing_else(fresh_target):
    from mlx.utils import tree_flatten

    # mlx-vlm's Model holds the text model as `language_model`, as enable() gets it.
    model = types.SimpleNamespace(language_model=fresh_target)
    assert projkernel.tag(model) == 31
    for path, module in fresh_target.named_modules():
        assert projkernel._tagged(module) == isinstance(module, nn.QuantizedLinear), path
    assert not any(projkernel._TAG in key for key, _ in tree_flatten(fresh_target.parameters()))
    projkernel.untag(model)
    assert not any(projkernel._tagged(module) for _, module in fresh_target.named_modules())


def test_the_drafters_own_linears_are_never_target_tagged(fresh_target):
    """DFlash binds the target's lm_head into the drafter: that head is the target's
    object and carries the target's tag; no linear the drafter owns does (they
    carry the drafter's own tag, which tag_drafter() sets)."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    vocab = fresh_target.args.vocab_size
    drafter = DFlashDraftModel(
        DFlashConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=vocab,
            mask_token_id=vocab - 1,
            target_layer_ids=[1],
            num_target_layers=4,
        )
    )
    drafter.set_dtype(mx.bfloat16)
    nn.quantize(drafter, group_size=64, bits=4)  # as the engine loads a drafter
    model = types.SimpleNamespace(language_model=fresh_target)
    drafter.bind(model)
    projkernel.tag(model)
    assert drafter.lm_head is fresh_target.lm_head and projkernel._tagged(drafter.lm_head)
    owned = [
        module
        for _, module in drafter.named_modules()
        if isinstance(module, nn.QuantizedLinear) and module is not fresh_target.lm_head
    ]
    assert len(owned) == 8
    assert not any(projkernel._tagged(module) for module in owned)


def test_install_hooks_wraps_each_method_once_over_its_original(monkeypatch):
    pfx.guard_proj_state(monkeypatch)
    verifier = projkernel._verifier_class()

    def methods():
        return (verifier._linear, verifier._linears, nn.QuantizedLinear.__call__)

    originals = [inspect.unwrap(method) for method in methods()]
    projkernel.install_hooks()
    first = methods()
    projkernel.install_hooks()
    assert all(now is then for now, then in zip(methods(), first, strict=True))
    for hook, original in zip(first, originals, strict=True):
        assert getattr(hook, projkernel._HOOK_MARK)
        assert inspect.unwrap(hook) is original
        # What the source pins hash on the next load in this process.
        assert inspect.getsource(hook) == inspect.getsource(original)


def test_the_verifier_hook_serves_tagged_verify_calls_through_the_class(monkeypatch):
    pfx.hooked(monkeypatch)
    verifier = projkernel._verifier_class()()
    q, k, v = (_tagged_linear(n, 256, seed) for seed, n in enumerate((512, 128, 128)))
    x = _acts(1, 5, 256)
    before = dict(projkernel.calls)
    single = verifier._linear(q, x)
    group = verifier._linears((q, k, v), x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}
    assert _bitwise(single, pfx.stand_in_mma(x, _triples(q))[0])
    assert isinstance(group, tuple)
    want = pfx.stand_in_mma(x, _triples(q, k, v))
    assert all(_bitwise(got, w) for got, w in zip(group, want, strict=True))


# case -> (flag, how many of the three linears are tagged, x, the counter it moves)
_VERIFY_HANDED_ON = {
    "flag clear": (0, 3, lambda: _acts(1, 3, 256), None),
    "untagged": (1, 0, lambda: _acts(1, 3, 256), None),
    "one linear untagged": (1, 2, lambda: _acts(1, 3, 256), None),
    "float16": (1, 3, lambda: _acts(1, 3, 256, dtype=mx.float16), "declined"),
    "float32": (1, 3, lambda: _acts(1, 3, 256, dtype=mx.float32), "declined"),
    "two dimensions": (1, 3, lambda: _acts(3, 256), "declined"),
    "a batch wider than eight rows": (1, 3, lambda: _acts(9, 1, 256), "declined"),
    "no rows": (1, 3, lambda: _acts(1, 0, 256), "declined"),
}


@pytest.mark.parametrize("method", ["_linear", "_linears"])
@pytest.mark.parametrize("case", list(_VERIFY_HANDED_ON))
def test_the_verifier_hook_hands_every_other_call_to_mlx_vlm(monkeypatch, method, case):
    """Checked against a recording original: the call's own arguments, and its
    answer returned."""
    flag, tagged, build, counter = _VERIFY_HANDED_ON[case]
    seen = []

    def original(self, linears, x):
        seen.append((self, linears, x))
        return "mlx-vlm's answer"

    wrap = projkernel._wrap_linear if method == "_linear" else projkernel._wrap_linears
    hook = wrap(original)
    assert getattr(hook, projkernel._HOOK_MARK) and inspect.unwrap(hook) is original
    monkeypatch.setattr(projkernel, "_active", flag)
    monkeypatch.setattr(projkernel, "mma", _no_kernel)
    group = [pfx.rand_module(128, 256, seed) for seed in range(3)]
    for linear in group[3 - tagged :]:  # the first is the one _linear gets
        object.__setattr__(linear, projkernel._TAG, True)
    linears = group[0] if method == "_linear" else tuple(group)
    verifier, x = object(), build()
    before = dict(projkernel.calls)
    assert hook(verifier, linears, x) == "mlx-vlm's answer"
    ((got_self, got_linears, got_x),) = seen
    assert got_self is verifier and got_linears is linears and got_x is x
    assert _proj_counts_since(before) == ({} if counter is None else {counter: 1})


@pytest.mark.parametrize(
    ("shape", "launches"),
    [((1, 9), [8, 1]), ((1, 12), [8, 4]), ((1, 16), [8, 8]), ((2, 6), [8, 4])],
)
def test_the_verifier_hook_cuts_a_long_verify_into_runs_equal_to_its_rows_alone(
    monkeypatch, shape, launches
):
    """Block 0 lets the drafter's own policy verify up to 16 rows. Each run holds
    at most eight rows, and every row equals the same row computed alone, as
    one-row decode computes it."""
    pfx.hooked(monkeypatch)
    seen = []

    def recording(x, weights):
        seen.append(x.size // x.shape[-1])
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", recording)
    verifier = projkernel._verifier_class()()
    group = [_tagged_linear(n, 256, seed) for seed, n in enumerate((256, 128))]
    x = _acts(*shape, 256)
    before = dict(projkernel.calls)
    outputs = verifier._linears(tuple(group), x)
    assert _proj_counts_since(before) == {"verify_kernel": 1, "verify_split": 1}
    assert seen == launches
    for out, linear in zip(outputs, group, strict=True):
        assert out.shape == (*shape, linear.weight.shape[0])
        for b in range(shape[0]):
            for t in range(shape[1]):
                (alone,) = pfx.stand_in_mma(x[b : b + 1, t : t + 1], _triples(linear))
                assert _bitwise(out[b : b + 1, t : t + 1], alone), (b, t)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (1, 8), (2, 4), (8,)])
def test_the_module_hook_serves_a_tagged_module_up_to_eight_rows(monkeypatch, shape):
    pfx.hooked(monkeypatch)
    linear = _tagged_linear(256, 256, 0)
    x = _acts(*shape, 256)
    before = dict(projkernel.calls)
    got = linear(x)
    assert _proj_counts_since(before) == {"plain_kernel": 1}
    assert _bitwise(got, pfx.stand_in_mma(x, _triples(linear))[0])


# case -> (flag, tagged, x, the counter it moves)
_PLAIN_HANDED_ON = {
    "flag clear": (0, True, lambda: _acts(1, 1, 256), None),
    "untagged": (1, False, lambda: _acts(1, 1, 256), None),
    "nine rows": (1, True, lambda: _acts(1, 9, 256), None),
    "nine rows across the batch": (1, True, lambda: _acts(3, 3, 256), None),
    "a prefill chunk": (1, True, lambda: _acts(1, 2048, 256), None),
    "no rows": (1, True, lambda: _acts(1, 0, 256), None),
    "float16": (1, True, lambda: _acts(1, 1, 256, dtype=mx.float16), "declined"),
    "float32": (1, True, lambda: _acts(1, 3, 256, dtype=mx.float32), "declined"),
    "one dimension": (1, True, lambda: _acts(256), "declined"),
}


@pytest.mark.parametrize("case", list(_PLAIN_HANDED_ON))
def test_the_module_hook_hands_every_other_call_to_what_it_wrapped(monkeypatch, case):
    flag, tagged, build, counter = _PLAIN_HANDED_ON[case]
    seen = []

    def original(self, x):
        seen.append((self, x))
        return "the wrapped method's answer"

    hook = projkernel._wrap_call(original)
    assert getattr(hook, projkernel._HOOK_MARK) and inspect.unwrap(hook) is original
    monkeypatch.setattr(projkernel, "_active", flag)
    monkeypatch.setattr(projkernel, "mma", _no_kernel)
    linear = pfx.rand_module(256, 256, 0)
    if tagged:
        object.__setattr__(linear, projkernel._TAG, True)
    x = build()
    before = dict(projkernel.calls)
    assert hook(linear, x) == "the wrapped method's answer"
    ((got_self, got_x),) = seen
    assert got_self is linear and got_x is x
    assert _proj_counts_since(before) == ({} if counter is None else {counter: 1})


def test_the_module_hook_leaves_untagged_modules_and_a_cleared_flag_on_mlx(monkeypatch):
    pfx.hooked(monkeypatch)
    stock = inspect.unwrap(nn.QuantizedLinear.__call__)
    other = pfx.rand_module(256, 256, 1)  # never tagged: a drafter's, another model's
    ours = _tagged_linear(256, 256, 2)
    x = _acts(1, 1, 256)
    before = dict(projkernel.calls)
    assert _bitwise(other(x), stock(other, x))
    projkernel.clear()
    assert _bitwise(ours(x), stock(ours, x))
    assert _proj_counts_since(before) == {}


def test_a_verify_row_equals_the_plain_row_through_the_two_hooks(monkeypatch, fresh_target):
    """The verifier's own _feed_forward at T = 5 against Qwen3_5MLP.__call__ one
    row at a time: gate+up fused and down alone on one side, three module calls
    on the other. A T- and group-invariant kernel gives both the same bits."""
    pfx.hooked(monkeypatch)
    projkernel.tag(fresh_target)
    mlp = fresh_target.model.layers[0].mlp
    x = _acts(1, 5, 256)
    before = dict(projkernel.calls)
    verify = projkernel._verifier_class()()._feed_forward(mlp, x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}
    before = dict(projkernel.calls)
    plain = [mlp(x[:, t : t + 1]) for t in range(5)]
    assert _proj_counts_since(before) == {"plain_kernel": 15}
    assert all(_bitwise(verify[:, t : t + 1], row) for t, row in enumerate(plain))


_OTHER_VERIFIERS = {
    "gemma4": ("mlx_vlm.models.gemma4.speculative_verifier", "Gemma4ExactSpeculativeVerifier"),
    "nemotron_h": (
        "mlx_vlm.models.nemotron_h.speculative_verifier",
        "NemotronHExactSpeculativeVerifier",
    ),
    "qwen4_exp": ("mlx_vlm.models.qwen4_exp.language", "Qwen4ExpBatchInvariantForward"),
}


@pytest.mark.parametrize("family", list(_OTHER_VERIFIERS))
def test_other_families_verifiers_inherit_the_hook_but_stay_stock_untagged(monkeypatch, family):
    """gemma4's and qwen4_exp's verifiers inherit the wrapped methods and
    nemotron_h's reaches them through super(): only the tags keep them stock."""
    from mlx_vlm.speculative.ops.linear import _target_verify_linear, _target_verify_linears

    module, name = _OTHER_VERIFIERS[family]
    pfx.hooked(monkeypatch)
    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, projkernel._verifier_class())
    verifier = cls()
    a, b = pfx.rand_module(256, 256, 1), pfx.rand_module(128, 256, 2)
    x = _acts(1, 3, 256)
    before = dict(projkernel.calls)
    single = verifier._linear(a, x)
    group = verifier._linears((a, b), x)
    assert _proj_counts_since(before) == {}
    assert _bitwise(single, _target_verify_linear(a, x))
    stock = _target_verify_linears((a, b), x)
    assert all(_bitwise(got, want) for got, want in zip(group, stock, strict=True))
    for linear in (a, b):
        object.__setattr__(linear, projkernel._TAG, True)
    before = dict(projkernel.calls)
    verifier._linear(a, x)
    verifier._linears((a, b), x)
    assert _proj_counts_since(before) == {"verify_kernel": 2}


@pytest.mark.parametrize("order", ["int8 first", "kernel first"])
def test_int8_and_the_kernel_share_quantized_linear_in_either_order(monkeypatch, order):
    """int8 takes a tagged module's calls of 128 rows and more, the kernel its
    calls of eight and fewer, and the rows between go to mlx, whichever wrapper
    sits on top. int8's routing seam, linear_int8, is replaced by a recorder, so
    no tensor unit is needed."""
    pfx.guard_proj_state(monkeypatch)
    stock = inspect.unwrap(nn.QuantizedLinear.__call__)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", stock)
    monkeypatch.setattr(int8prefill, "_QUANTIZED_LINEAR_WRAPPED", False)
    installs = [lambda: int8prefill.install_wrappers([]), projkernel.install_hooks]
    for install in installs if order == "int8 first" else installs[::-1]:
        install()
    top = nn.QuantizedLinear.__call__
    assert getattr(top, projkernel._HOOK_MARK) and inspect.unwrap(top) is stock
    projkernel.install_hooks()
    assert nn.QuantizedLinear.__call__ is top, "the mark is found through int8's wrapper"

    int8_rows = []

    def int8_path(linear, x, stage=None):
        int8_rows.append(x.shape[-2])
        return mx.zeros((*x.shape[:-1], linear.weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(int8prefill, "linear_int8", int8_path)
    monkeypatch.setattr(projkernel, "mma", pfx.stand_in_mma)
    monkeypatch.setattr(projkernel, "_active", 1)
    linear = _tagged_linear(128, 256, 0)
    object.__setattr__(linear, int8prefill._TAG, True)
    before = dict(projkernel.calls)
    long, short, between = _acts(1, 130, 256), _acts(1, 3, 256), _acts(1, 20, 256)
    assert _bitwise(linear(long), mx.zeros((1, 130, 128), dtype=mx.bfloat16))
    assert int8_rows == [130]
    assert _bitwise(linear(short), pfx.stand_in_mma(short, _triples(linear))[0])
    assert _bitwise(linear(between), stock(linear, between))
    assert int8_rows == [130]
    assert _proj_counts_since(before) == {"plain_kernel": 1}


# ---- routing to the projection tile ------------------------------------------------


def _tile_target(monkeypatch, *, scope="all"):
    """The tiny model slab-only (its attention and GDN linears stacked at scope
    "all") and tagged, behind both hooks, with the stand-ins as the projection
    kernel and as the tile. Serving on the tile still needs ptfx.window()."""
    ptfx.hooked_tile(monkeypatch)
    root = ptfx.tiny_slab_model(seed=0, scope=scope)
    projkernel.tag(root)
    return root


def _no_mma(x, weights):
    raise AssertionError("a run the tile serves reached the projection kernel")


def _recording_mma(monkeypatch) -> list[tuple[int, int]]:
    """The stand-in as the projection kernel, each launch recorded as (rows,
    linears)."""
    seen: list[tuple[int, int]] = []

    def recording(x, weights):
        seen.append((x.size // x.shape[-1], len(weights)))
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", recording)
    return seen


@pytest.mark.parametrize(("t", "runs"), [(5, 1), (12, 2), (9, 2)])
def test_the_verifier_hook_serves_every_run_on_the_tile_while_the_window_is_open(
    monkeypatch, t, runs
):
    """A call past eight rows is cut into runs first, and every run, a one-row
    remainder included, runs on the tile: none reaches the MMA kernel, and each
    row equals the same row verified alone."""
    root = _tile_target(monkeypatch)
    monkeypatch.setattr(projkernel, "mma", _no_mma)
    attention = root.model.layers[1].self_attn
    qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
    verifier = projkernel._verifier_class()()
    x = _acts(1, t, 256)
    with ptfx.window():
        before = dict(projkernel.calls)
        outs = verifier._linears(qkv, x)
        moved = _proj_counts_since(before)
        alone = [verifier._linears(qkv, x[:, r : r + 1]) for r in range(t)]
    split = {"verify_split": 1} if runs > 1 else {}
    assert moved == {"verify_kernel": 1, "verify_tile": runs, **split}
    for i, (out, width) in enumerate(zip(outs, (2048, 512, 512), strict=True)):
        assert out.shape == (1, t, width)
        for r in range(t):
            assert _bitwise(out[:, r : r + 1], alone[r][i]), (i, r)


def test_every_verifier_call_of_a_converted_load_runs_on_the_tile(monkeypatch):
    """q+k+v, the GDN four (two launches), gate+up (the dual), down, o_proj and
    out_proj: each call one served run."""
    root = _tile_target(monkeypatch)
    monkeypatch.setattr(projkernel, "mma", _no_mma)
    gdn = root.model.layers[0].linear_attn
    attention = root.model.layers[1].self_attn
    mlp = root.model.layers[0].mlp
    verifier = projkernel._verifier_class()()
    cases = [
        (verifier._linears, (attention.q_proj, attention.k_proj, attention.v_proj), 256),
        (verifier._linears, (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a), 256),
        (verifier._linears, (mlp.gate_proj, mlp.up_proj), 256),
        (verifier._linear, mlp.down_proj, 512),
        (verifier._linear, attention.o_proj, 1024),
        (verifier._linear, gdn.out_proj, 256),
    ]
    with ptfx.window():
        for method, linears, k in cases:
            before = dict(projkernel.calls)
            method(linears, _acts(1, 5, k))
            assert _proj_counts_since(before) == {"verify_kernel": 1, "verify_tile": 1}, k


def test_a_call_with_an_unservable_linear_stays_on_the_mma_kernel_whole(monkeypatch):
    """At scope mlp the attention and GDN linears hold no planes, and the head never
    does: inside the window their calls stay on the MMA kernel, every run of them."""
    root = _tile_target(monkeypatch, scope="mlp")
    seen = _recording_mma(monkeypatch)
    gdn = root.model.layers[0].linear_attn
    gdn_call = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
    verifier = projkernel._verifier_class()()
    with ptfx.window():
        before = dict(projkernel.calls)
        verifier._linears(gdn_call, _acts(1, 12, 256))
        verifier._linear(root.lm_head, _acts(1, 5, 256))
    assert _proj_counts_since(before) == {"verify_kernel": 2, "verify_split": 1}
    assert seen == [(8, 4), (4, 4), (5, 1)]


def test_a_key_outside_the_plan_table_is_served_by_the_mma_kernel(monkeypatch):
    """No default plan: without gate+up's key the MLP's fused call runs on the MMA
    kernel whole, while down, whose key the table holds, stays on the tile."""
    root = _tile_target(monkeypatch)
    plans = dict(projtile.PLANS)
    del plans[(256, (512, 512))]
    monkeypatch.setattr(projtile, "PLANS", plans)
    seen = _recording_mma(monkeypatch)
    mlp = root.model.layers[0].mlp
    with ptfx.window():
        before = dict(projkernel.calls)
        projkernel._verifier_class()()._feed_forward(mlp, _acts(1, 5, 256))
    assert _proj_counts_since(before) == {"verify_kernel": 2, "verify_tile": 1}
    assert seen == [(5, 2)]


def _close(got, want) -> bool:
    """Within a few bf16 ulps: the stand-in over a wider stack is a matmul of
    another width, which need not sum a column in the same order."""
    diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item()
    return diff <= 1e-2 * mx.max(mx.abs(want.astype(mx.float32))).item()


@pytest.mark.parametrize("rows", range(1, 9))
def test_the_module_hook_routes_two_to_eight_rows_to_the_tile_inside_the_window_only(
    monkeypatch, rows
):
    """Inside the window a call of two to eight rows runs on the tile and a call of
    one on the one-row kernel; outside it, every one of them on the projection
    kernel. A stack member reads its own tiles (k), a slab-only linear its own
    planes (gate), a member sharing a tile the whole stack (in_proj_a)."""
    root = _tile_target(monkeypatch)
    # The real dispatcher over the stand-ins, so a one-row call shows its dispatch.
    monkeypatch.setattr(projkernel, "mma", projkernel.linears)
    monkeypatch.setattr(projkernel, "_launch", _stand_in_launch)
    layer0 = root.model.layers[0]
    x = _acts(1, rows, 256)
    one = {"plain_kernel": 1, "one_row": 1}
    for module, exact in (
        (root.model.layers[1].self_attn.k_proj, True),
        (layer0.mlp.gate_proj, True),
        (layer0.linear_attn.in_proj_a, False),
    ):
        with ptfx.window():
            before = dict(projkernel.calls)
            inside = module(x)
            assert _proj_counts_since(before) == (one if rows == 1 else {"plain_tile": 1})
        before = dict(projkernel.calls)
        outside = module(x)
        assert _proj_counts_since(before) == (one if rows == 1 else {"plain_kernel": 1})
        assert inside.shape == outside.shape == (1, rows, projkernel._out_features(module))
        assert (_bitwise if exact else _close)(inside, outside)


@pytest.mark.parametrize("shape", [(1, 9), (1, 127), (2, 5)])
def test_nine_to_127_rows_on_a_slab_only_module_fall_through_to_int8(monkeypatch, shape):
    """The projection kernel hands a call past eight rows on, window or not, and on
    a slab-only module, whose planes stock cannot read, int8's wrapper takes it
    from nine rows: rows are the product of the leading dimensions on both sides,
    so a batch of two five-row calls is int8's too."""
    root = _tile_target(monkeypatch)
    stock = inspect.unwrap(nn.QuantizedLinear.__call__)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", stock)
    monkeypatch.setattr(int8prefill, "_QUANTIZED_LINEAR_WRAPPED", False)
    int8prefill.install_wrappers([])
    projkernel.install_hooks()
    served = []

    def qmm(qa, sa, ra, w, scales, biases, *, slab=None):
        assert slab is not None, "the module is slab-only"
        served.append((qa.shape[0], slab))
        return mx.zeros((qa.shape[0], slab.n_real), dtype=mx.bfloat16)

    monkeypatch.setattr(int8prefill, "qmm", qmm)
    gate = root.model.layers[0].mlp.gate_proj
    object.__setattr__(gate, int8prefill._TAG, True)
    x = _acts(*shape, 256)
    before, small = dict(projkernel.calls), int8prefill.calls["int8_small"]
    with ptfx.window():
        assert projkernel._serve_plain(gate, x) is None
        out = gate(x)
    assert out.shape == (*shape, 512)
    assert served == [(shape[0] * shape[1], slab.slab_info(gate))]
    assert int8prefill.calls["int8_small"] == small + 1
    assert _proj_counts_since(before) == {}


def test_the_round_flag_closes_the_tile_for_one_round(monkeypatch):
    """The gate's per-round lever: at 0 the window's calls take the projection
    kernel's paths, and the next window, as the next decode opens it, serves on
    the tile again."""
    root = _tile_target(monkeypatch)
    attention = root.model.layers[1].self_attn
    qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
    verifier = projkernel._verifier_class()()
    x = _acts(1, 5, 256)

    def one_round():
        before = dict(projkernel.calls)
        verifier._linears(qkv, x)
        attention.q_proj(x[:, :3])
        return _proj_counts_since(before)

    on_tile = {"verify_kernel": 1, "verify_tile": 1, "plain_tile": 1}
    with ptfx.window():
        projtile.set_round_flag(0)
        assert one_round() == {"verify_kernel": 1, "plain_kernel": 1}
        projtile.set_round_flag(1)
        assert one_round() == on_tile
        projtile.set_round_flag(0)
    with ptfx.window():
        assert one_round() == on_tile


def test_the_drafters_own_linears_never_reach_the_tile(monkeypatch):
    """A drafter-tagged module is the projection kernel's from six rows, inside the
    window too, even one that holds a stack."""
    _tile_target(monkeypatch)
    monkeypatch.setattr(projkernel, "_draft_active", 1)
    member = ptfx.tiny_slab_model(seed=1).model.layers[1].self_attn.k_proj
    assert projtile.stack_of(member) is not None
    object.__setattr__(member, projkernel._DRAFT_TAG, True)
    with ptfx.window():
        before = dict(projkernel.calls)
        member(_acts(1, 7, 256))
    assert _proj_counts_since(before) == {"draft_kernel": 1}


# ---- the gates, the pins, the warm-up, the probe and enable() -----------------------

# The tiny model's quantized projections: a one-row forward makes one hook-2 call
# for each. A verify forward makes 17 hook-1 calls: four per layer, then the head.
_PROJECTIONS = 31
_VERIFY_CALLS = 17
_BUILD_ERROR = "Unable to build metal library\nprogram_source:3:1: error: boom"


def _verifier_cls():
    return importlib.import_module(
        "mlx_vlm.models.qwen3_5.speculative_verifier"
    ).Qwen3_5BatchInvariantForward


def _projections(model) -> list:
    root = getattr(model, "language_model", model)
    return [m for _, m in root.named_modules() if isinstance(m, nn.QuantizedLinear)]


def _tagged_count(model) -> int:
    return sum(bool(getattr(m, projkernel._TAG, False)) for m in _projections(model))


def _hooks_in_place() -> bool:
    cls = _verifier_cls()
    return all(
        getattr(f, projkernel._HOOK_MARK, False)
        for f in (nn.QuantizedLinear.__call__, cls._linear, cls._linears)
    )


class _GateRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _gate_log():
    """Every record the kernel's logger emits inside the block, INFO included.
    Records still propagate, so a test's own caplog sees them too."""
    logger = logging.getLogger("sous.engine.projkernel")
    handler, level = _GateRecords(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _refusal(model, *, warned: bool, untagged: bool = True, **kwargs) -> dict:
    """enable() on `model`, expected to refuse, leaving the flag clear. A load
    the kernel does not apply to (the tile, the model, the drafter) is one INFO
    line and no warning, since most Macs load that way; one where it should
    have run and did not is exactly one warning, on one line, in the kernel's
    words. Unless the model kept its tags from an earlier activation, nothing
    on it is tagged afterwards."""
    kwargs = {"drafter_kind": None, "tile_status": dict(pfx.ACTIVE_TILE), **kwargs}
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = projkernel.enable(model, enabled=True, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert status["state"] == "unavailable"
    if warned:
        assert len(messages) == 1, messages
        assert messages[0].startswith("sous: projection kernel unavailable (")
        assert "\n" not in messages[0]
        assert not any(m.startswith("projection kernel unavailable") for m in infos), infos
    else:
        assert messages == []
        assert infos == [f"projection kernel unavailable: {status['reason']}"]
    assert projkernel._active == 0
    if untagged:
        assert _tagged_count(model) == 0
    return status


def _enabling_failures(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "sous.engine.projkernel"
        and r.getMessage() == "projection kernel: enabling failed"
    ]


def test_enable_off_returns_before_any_guard(monkeypatch):
    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 1)
    monkeypatch.setattr(projkernel, "_draft_active", 1)

    def untouchable(*args, **kwargs):
        raise AssertionError("a guard ran while the kernel is off")

    for name in (
        "_tile_reason",
        "_model_reason",
        "_greedy_verify_reason",
        "_pins_reason",
        "warm_up",
        "probe",
    ):
        monkeypatch.setattr(projkernel, name, untouchable)
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = projkernel.enable(
            types.SimpleNamespace(), enabled=False, drafter_kind="mtp", tile_status={}
        )
    assert status == {"state": "off", "reason": None, "probe_seconds": None}
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert records == []
    assert projkernel._active == 0
    assert projkernel._draft_active == 0


_GUARD_REASONS = [
    "macOS 15.5 < 26.2",
    "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
    "drafter kind 'mtp' (dflash only)",
    "the language model defines speculative_verify_dflash_hidden: "
    "greedy verify would take its tokens outside the kernel",
    "mlx 0.99.0 not validated",
    "warm-up: program_source:3:1: error: boom",
    "parity: verify row 2 of 5 differs from the one-row forward",
]
# Which of them warn: mlx-vlm drift (a greedy verify that leaves the hooks, a
# pinned dependency that changed), a kernel this OS rejects and a failed probe.
# The tile, the model and the drafter's kind only say the kernel does not apply
# to this load.
_GUARD_WARNS = [False, False, False, True, True, True, True]


@pytest.mark.parametrize("first", range(len(_GUARD_REASONS)))
def test_enable_names_the_first_failing_guard(monkeypatch, first):
    """Guard `first` and every guard after it fail; the reason is guard `first`'s."""
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    lm = pfx.tiny_quantized_model()
    kwargs: dict = {"drafter_kind": None, "tile_status": tile}

    def broken_warm_up(model):
        raise RuntimeError(_BUILD_ERROR)

    breakers = [
        lambda: kwargs.update(
            tile_status={"state": "unavailable", "reason": _GUARD_REASONS[0], "splits": None}
        ),
        lambda: monkeypatch.setattr(lm, "model_type", "qwen3_5_moe"),
        lambda: kwargs.update(drafter_kind="mtp"),
        lambda: object.__setattr__(lm, "speculative_verify_dflash_hidden", lambda *a, **k: None),
        lambda: monkeypatch.setattr(mx, "__version__", "0.99.0"),
        lambda: monkeypatch.setattr(projkernel, "warm_up", broken_warm_up),
        lambda: monkeypatch.setattr(projkernel, "probe", lambda model: _GUARD_REASONS[6]),
    ]
    for brk in breakers[first:]:
        brk()
    status = _refusal(lm, warned=_GUARD_WARNS[first], **kwargs)
    assert status["reason"] == _GUARD_REASONS[first]
    # The cost is reported once the warm-up has started.
    assert (status["probe_seconds"] is None) == (first < 5)


@pytest.mark.parametrize(
    ("tile_status", "reason"),
    [
        ({"state": "off", "reason": None, "splits": None}, "attention tile off"),
        (
            {"state": "unavailable", "reason": "split target 16 not measured (measured: 20)"},
            "split target 16 not measured (measured: 20)",
        ),
        ({"state": "unavailable", "reason": None}, "attention tile unavailable"),
    ],
)
def test_enable_rides_on_the_attention_tile(monkeypatch, tile_status, reason):
    pfx.proj_ready(monkeypatch)
    status = _refusal(pfx.tiny_quantized_model(), warned=False, tile_status=tile_status)
    assert status["reason"] == reason and status["probe_seconds"] is None


def test_enable_refuses_other_model_types(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    # The MoE variant reuses the verifier class and the dense MLP.
    monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
    status = _refusal(lm, warned=False)
    assert status["reason"] == "unsupported model type 'qwen3_5_moe' (qwen3_5 only)"


def test_enable_refuses_a_model_without_quantized_projections(monkeypatch):
    pfx.proj_ready(monkeypatch)
    status = _refusal(tfx.tiny_language_model(), warned=False)
    assert status["reason"] == "no quantized projections"


def _replace(lm, case: str) -> str:
    """Swap one projection for one the kernel cannot take; returns its path."""
    if case == "8-bit":
        head = nn.QuantizedLinear(256, tfx.VOCAB, bias=False, group_size=64, bits=8)
        head.set_dtype(mx.bfloat16)
        lm.lm_head = head
        return "lm_head"
    if case == "bias":
        out = nn.QuantizedLinear(256, 256, bias=True, group_size=64, bits=4)
        out.set_dtype(mx.bfloat16)
        lm.model.layers[0].linear_attn.out_proj = out
        return "model.layers.0.linear_attn.out_proj"
    down = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
    down.set_dtype(mx.float16)
    lm.model.layers[1].mlp.down_proj = down
    return "model.layers.1.mlp.down_proj"


@pytest.mark.parametrize("case", ["8-bit", "bias", "fp16 scales"])
def test_enable_refuses_a_projection_the_kernel_cannot_take(monkeypatch, case):
    """Decode and verify route every projection of the language model, so one
    the kernel cannot take refuses the load rather than leaving it stock."""
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    path = _replace(lm, case)
    status = _refusal(lm, warned=False)
    assert status["reason"] == (
        f"projection {path} is not affine 4-bit gs64 with bf16 scales and no bias"
    )


def test_enable_refuses_activations_that_are_not_bf16(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    lm.model.norm.set_dtype(mx.float16)
    status = _refusal(lm, warned=False)
    assert status["reason"] == f"activation dtype {mx.float16} is not bfloat16"


@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
def test_enable_refuses_drafters_other_than_dflash(monkeypatch, kind):
    pfx.proj_ready(monkeypatch)
    status = _refusal(pfx.tiny_quantized_model(), warned=False, drafter_kind=kind)
    assert status["reason"] == f"drafter kind {kind!r} (dflash only)"


def test_enable_refuses_a_target_whose_greedy_verify_skips_the_head(monkeypatch):
    """mlx-vlm's greedy verify takes its tokens from quantized_argmax, which
    neither hook sees, once the target defines this method. No qwen3_5 target
    defines it in the validated mlx-vlm, so it is drift and warns."""
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    object.__setattr__(lm, "speculative_verify_dflash_hidden", lambda *args, **kwargs: None)
    status = _refusal(lm, warned=True, drafter_kind="dflash")
    assert status["reason"] == (
        "the language model defines speculative_verify_dflash_hidden: "
        "greedy verify would take its tokens outside the kernel"
    )


def test_the_pins_hold_on_the_locked_mlx_and_mlx_vlm():
    assert set(projkernel.VALIDATED_PROJ_SOURCES) == {
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linear",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._feed_forward",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._gated_delta",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._model",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward.__call__",
        "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._layer",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5DecoderLayer.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5MLP.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet.__call__",
        "mlx_vlm.models.qwen3_5.language:Qwen3_5GatedDeltaNet._project_gates",
        "mlx_vlm.generate.ar:generate_step",
        "mlx.nn:QuantizedLinear.__call__",
        "mlx_vlm.speculative.dflash:_dflash_verify",
        "mlx_vlm.speculative.dflash:_dflash_verify_greedy",
        "mlx_vlm.speculative.dflash:_dflash_rounds",
    }
    assert projkernel._pins_reason() is None


def test_enable_refuses_an_unvalidated_mlx(monkeypatch):
    pfx.proj_ready(monkeypatch, real_pins=True)
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "mlx 0.99.0 not validated"


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        (
            "mlx_vlm.models.qwen3_5.speculative_verifier:Qwen3_5BatchInvariantForward._linears",
            "mlx-vlm Qwen3_5BatchInvariantForward._linears changed",
        ),
        (
            "mlx_vlm.speculative.dflash:_dflash_verify_greedy",
            "mlx-vlm _dflash_verify_greedy changed",
        ),
        ("mlx_vlm.speculative.dflash:_dflash_rounds", "mlx-vlm _dflash_rounds changed"),
        (
            "mlx_vlm.models.qwen3_5.language:Qwen3_5DecoderLayer.__call__",
            "mlx-vlm Qwen3_5DecoderLayer.__call__ changed",
        ),
        ("mlx.nn:QuantizedLinear.__call__", "mlx QuantizedLinear.__call__ changed"),
    ],
)
def test_enable_refuses_a_changed_source(monkeypatch, key, reason):
    pfx.proj_ready(monkeypatch, real_pins=True)
    monkeypatch.setitem(projkernel.VALIDATED_PROJ_SOURCES, key, "0" * 64)
    assert _refusal(pfx.tiny_quantized_model(), warned=True)["reason"] == reason


@pytest.mark.parametrize("order", ["int8 first", "int8 after"])
def test_the_pins_follow_int8s_wrappers_and_the_hooks_to_the_originals(monkeypatch, order):
    """int8 prefill wraps nn.QuantizedLinear.__call__, Qwen3_5MLP.__call__ and
    Qwen3_5GatedDeltaNet.__call__, all three pinned here, in either order with
    the hooks. With int8 after, its wrapper sits on hook 2 and carries the mark,
    so a later load neither re-wraps nor reads the pins as drifted."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet, Qwen3_5MLP

    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    for cls in (Qwen3_5MLP, Qwen3_5GatedDeltaNet):
        monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(int8prefill, "_WRAPPED", set())
    monkeypatch.setattr(int8prefill, "_QUANTIZED_LINEAR_WRAPPED", False)

    def install_int8() -> None:
        int8prefill.install_wrappers([(Qwen3_5MLP, "mlp"), (Qwen3_5GatedDeltaNet, "gdn")])

    if order == "int8 first":
        install_int8()
    status = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    if order == "int8 after":
        install_int8()
    installed = nn.QuantizedLinear.__call__
    projkernel.clear()
    status = projkernel.enable(
        pfx.tiny_quantized_model(seed=1), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    assert nn.QuantizedLinear.__call__ is installed
    assert projkernel._pins_reason() is None


def test_a_second_load_activates_again_on_the_hooks_the_first_installed(monkeypatch):
    """An idle unload and reload, or sous tune's arms, enable a fresh model in a
    process whose hooks are in place: the pins follow __wrapped__ through them
    to mlx-vlm's and mlx's own sources, and nothing is wrapped twice."""
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    first = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert first["state"] == "active", first
    cls = _verifier_cls()
    hooks = (nn.QuantizedLinear.__call__, cls.__dict__["_linear"], cls.__dict__["_linears"])
    projkernel.clear()
    second = projkernel.enable(
        pfx.tiny_quantized_model(seed=1), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert second["state"] == "active", second
    now = (nn.QuantizedLinear.__call__, cls.__dict__["_linear"], cls.__dict__["_linears"])
    assert all(a is b for a, b in zip(now, hooks, strict=True))


def test_enable_refuses_a_kernel_the_warm_up_cannot_build(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)
    error = RuntimeError(_BUILD_ERROR)

    def broken(model):
        raise error

    def untouchable(model):
        raise AssertionError("probed after a failed warm-up")

    monkeypatch.setattr(projkernel, "warm_up", broken)
    monkeypatch.setattr(projkernel, "probe", untouchable)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "warm-up: program_source:3:1: error: boom"
    assert isinstance(status["probe_seconds"], float)
    # The warning keeps one line; the log keeps the compiler's whole report.
    (record,) = [
        r
        for r in caplog.records
        if r.name == "sous.engine.projkernel"
        and r.getMessage() == "projection kernel: the warm-up failed"
    ]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


def test_warm_up_runs_every_dispatch_key_at_every_row_count(monkeypatch):
    pfx.guard_proj_state(monkeypatch)
    seen: list = []

    def spy(x, weights):
        seen.append((x.shape[-1], tuple(w.shape[0] for w, _, _ in weights), x.shape[1], x.dtype))
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", spy)
    projkernel.warm_up(pfx.tiny_quantized_model())
    keys = [
        (256, (2048, 512, 512)),  # q+k+v
        (256, (512, 512)),  # gate+up
        (256, (512, 256, 4, 4)),  # in_proj qkv+z+b+a
        (256, (512,)),  # lm_head, gate, up, k, v and in_proj_qkv alone
        (256, (2048,)),  # q alone
        (256, (256,)),  # in_proj_z and out_proj alone
        (256, (4,)),  # in_proj_b and in_proj_a alone
        (512, (256,)),  # down
        (1024, (256,)),  # o
    ]
    expected = sorted((k, n, t) for k, n in keys for t in range(1, projkernel.MAX_ROWS + 1))
    assert sorted((k, n, t) for k, n, t, _ in seen) == expected
    assert {dtype for *_, dtype in seen} == {mx.bfloat16}


# Every linear shape (K, N) of the tiny model, each a one-row kernel pipeline.
_TINY_LINEARS = {(256, 2048), (256, 512), (256, 256), (256, 4), (512, 256), (1024, 256)}


def test_warm_up_runs_the_one_row_kernel_once_per_linear_shape(monkeypatch):
    pfx.guard_proj_state(monkeypatch)
    pfx.stand_ins(monkeypatch)
    seen: list = []

    def spy(x2, w, scales, biases):
        seen.append((x2.shape, x2.dtype, w.shape[0]))
        return pfx.stand_in_one_row(x2, w, scales, biases)

    monkeypatch.setattr(projkernel, "one_row", spy)
    projkernel.warm_up(pfx.tiny_quantized_model())
    assert sorted((shape[1], n) for shape, _, n in seen) == sorted(_TINY_LINEARS)
    assert {(shape[0], dtype) for shape, dtype, _ in seen} == {(1, mx.bfloat16)}


@pfx.kernel
def test_warm_up_leaves_the_counters_as_it_found_them(monkeypatch):
    """Its one-row launches go through linears() on the real kernel, which
    counts what it serves; a load is not a served call."""
    pfx.guard_proj_state(monkeypatch)
    before = dict(projkernel.calls)
    projkernel.warm_up(pfx.tiny_quantized_model())
    assert projkernel.calls == before


@pytest.mark.parametrize(
    "real", [False, pytest.param(True, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
def test_warm_up_covers_a_converted_model(monkeypatch, real):
    """A slab-only module's dispatch keys and one-row shapes are pipelines of
    their own beside the mlx-layout ones, even where K and N coincide (gate and
    up alone against k, v, in_proj_qkv and lm_head): each runs once at every
    row count, and the counters are left as found."""
    pfx.guard_proj_state(monkeypatch)
    if not real:
        pfx.stand_ins(monkeypatch)
    mma, one_row = projkernel.mma, projkernel.one_row
    seen: list = []
    rows: list = []
    inside = [False]

    def spy(x, weights):
        entries = [projkernel._as_weights(e) for e in weights]
        ns = tuple(e.slab.n_real if e.slab else e.w.shape[0] for e in entries)
        seen.append((x.shape[-1], ns, entries[0].slab is not None, x.shape[1]))
        inside[0] = True
        try:
            return mma(x, weights)
        finally:
            inside[0] = False

    def row_spy(x2, w, scales, biases, slab=None):
        # the real `mma` reaches the one-row kernel too, at T = 1; these are the
        # one-row pipelines warm_up() states in their own right
        if not inside[0]:
            rows.append((x2.shape, w.shape[0] if slab is None else slab.n_real, slab is not None))
        if slab is None:
            return one_row(x2, w, scales, biases)
        return one_row(x2, w, scales, biases, slab=slab)

    monkeypatch.setattr(projkernel, "mma", spy)
    monkeypatch.setattr(projkernel, "one_row", row_spy)
    before = dict(projkernel.calls)
    projkernel.warm_up(ptfx.tiny_slab_model(scope="mlp"))
    assert projkernel.calls == before
    keys = [
        (256, (2048, 512, 512), False),  # q+k+v
        (256, (512, 256, 4, 4), False),  # in_proj qkv+z+b+a
        (256, (512,), False),  # lm_head, k, v and in_proj_qkv alone
        (256, (2048,), False),  # q alone
        (256, (256,), False),  # in_proj_z and out_proj alone
        (256, (4,), False),  # in_proj_b and in_proj_a alone
        (1024, (256,), False),  # o
        (256, (512, 512), True),  # gate+up
        (256, (512,), True),  # gate and up alone
        (512, (256,), True),  # down
    ]
    expected = sorted((*key, t) for key in keys for t in range(1, projkernel.MAX_ROWS + 1))
    assert sorted(seen) == expected
    shapes = {(256, 2048), (256, 512), (256, 256), (256, 4), (1024, 256)}
    want = sorted([(k, n, False) for k, n in shapes] + [(256, 512, True), (512, 256, True)])
    assert sorted((shape[1], n, form) for shape, n, form in rows) == want
    assert all(shape[0] == 1 for shape, _, _ in rows)


def _probe_model(monkeypatch, mma=None):
    """A tiny model as enable() hands it to the probe: tagged, both hooks in
    place, the flag clear, and `mma` (the stand-in by default) as the kernel."""
    lm = pfx.tiny_quantized_model()
    pfx.guard_proj_state(monkeypatch)
    pfx.hooked(monkeypatch)
    if mma is not None:
        monkeypatch.setattr(projkernel, "mma", mma)
    monkeypatch.setattr(projkernel, "_active", 0)
    projkernel.tag(lm)
    return lm


def _run_probe(lm) -> str | None:
    """probe(), asserting it left the flag clear and the counters as it found
    them, whatever it decided."""
    before = dict(projkernel.calls)
    reason = projkernel.probe(lm)
    assert projkernel._active == 0
    assert projkernel.calls == before
    return reason


def _rows(x) -> int:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return rows


def _scaled(outs, factor: float) -> tuple:
    return tuple((o.astype(mx.float32) * factor).astype(mx.bfloat16) for o in outs)


def test_probe_passes_the_stand_in(monkeypatch):
    assert _run_probe(_probe_model(monkeypatch)) is None


@pfx.kernel
def test_probe_passes_the_real_kernel(monkeypatch):
    lm = _probe_model(monkeypatch)
    pfx.real_kernel(monkeypatch)
    assert _run_probe(lm) is None


@pytest.mark.parametrize(
    "real", [False, pytest.param(True, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
def test_the_probe_passes_a_converted_model_on_its_own_forms(monkeypatch, real):
    """Every step reads the descriptor: the hooks serve the slab-only MLP, the
    stock reference of a slab-only linear is its unpacked arrays, and the
    one-row check holds the slab one-row kernel to the slab MMA bodies."""
    lm = ptfx.tiny_slab_model(scope="mlp")
    pfx.hooked(monkeypatch)
    if real:
        pfx.real_kernel(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 0)
    projkernel.tag(lm)
    one_row = projkernel.one_row
    forms: set = set()

    def row_spy(x2, w, scales, biases, slab=None):
        forms.add(slab is not None)
        if slab is None:
            return one_row(x2, w, scales, biases)
        return one_row(x2, w, scales, biases, slab=slab)

    monkeypatch.setattr(projkernel, "one_row", row_spy)
    assert _run_probe(lm) is None
    assert forms == {False, True}


def test_probe_fails_one_row_for_a_one_row_kernel_that_differs(monkeypatch):
    lm = _probe_model(monkeypatch)
    monkeypatch.setattr(projkernel, "one_row", _nudged_row())
    assert _run_probe(lm) == (
        "one-row: model.layers.0.linear_attn.in_proj_qkv differs from the MMA kernel "
        "(random activations)"
    )


@pytest.mark.parametrize("layout", projkernel.CANCELLING)
def test_probe_fails_one_row_where_a_layout_cancels_alone(monkeypatch, layout):
    lm = _probe_model(monkeypatch)
    monkeypatch.setattr(projkernel, "one_row", _nudged_row(when=_big_in(layout)))
    assert _run_probe(lm) == (
        "one-row: model.layers.0.linear_attn.in_proj_qkv differs from the MMA kernel "
        f"({layout}-cancelling activations)"
    )


def test_enable_refuses_a_one_row_kernel_that_differs_and_warns(monkeypatch):
    """Parity rests on the one-row kernel's bits: a mismatch switches the whole
    kernel off, MMA paths included, with a warning."""
    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(projkernel, "one_row", _nudged_row())
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"].startswith("one-row: "), status


def test_probe_refuses_projections_that_never_reach_the_kernel(monkeypatch):
    lm = _probe_model(monkeypatch)
    projkernel.untag(lm)
    assert _run_probe(lm) == "the exact verifier's projections did not reach the kernel"


def test_probe_refuses_when_the_models_own_calls_skip_hook_two(monkeypatch):
    lm = _probe_model(monkeypatch)
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", inspect.unwrap(nn.QuantizedLinear.__call__))
    assert _run_probe(lm) == "the model's own projections did not reach the kernel"


def test_probe_fails_parity_for_a_kernel_whose_rows_move_with_the_row_count(monkeypatch):
    def by_rows(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + _rows(x) / 64)

    reason = _run_probe(_probe_model(monkeypatch, by_rows))
    assert reason == "parity: verify row 0 of 5 differs from the one-row forward"


def test_probe_fails_t_invariance_past_the_rows_the_hooks_step_used(monkeypatch):
    def past_five(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + (_rows(x) > 5) / 16)

    reason = _run_probe(_probe_model(monkeypatch, past_five))
    assert reason == (
        "T-invariance: model.layers.1.self_attn.q_proj+model.layers.1.self_attn.k_proj"
        "+model.layers.1.self_attn.v_proj at T=6 differs from its rows alone"
    )


def test_probe_fails_grouping_for_a_kernel_whose_rows_move_with_the_group(monkeypatch):
    def by_group(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1 + (len(weights) > 2) / 16)

    reason = _run_probe(_probe_model(monkeypatch, by_group))
    assert reason is not None and reason.startswith(
        "grouping: model.layers.1.self_attn.q_proj fused in "
    ), reason


def test_probe_fails_correctness_for_a_kernel_wrong_the_same_way_everywhere(monkeypatch):
    def scaled(x, weights):
        return _scaled(pfx.stand_in_mma(x, weights), 1.1)

    reason = _run_probe(_probe_model(monkeypatch, scaled))
    assert reason is not None and reason.startswith(
        "correctness: model.layers.1.self_attn.q_proj is "
    ), reason


def test_probe_fails_isolation_for_a_kernel_that_mixes_rows(monkeypatch):
    def mixes_rows(x, weights):
        return tuple(
            o + 0 * mx.sum(o, axis=-2, keepdims=True) for o in pfx.stand_in_mma(x, weights)
        )

    reason = _run_probe(_probe_model(monkeypatch, mixes_rows))
    assert reason == (
        "isolation: a NaN in row 3 changed the other rows of model.layers.1.self_attn.q_proj"
    )


def test_the_stock_distance_fails_every_bound_on_a_nan():
    want = mx.ones((1, 8, 64), dtype=mx.bfloat16)
    got = mx.where(mx.arange(64) == 7, mx.array(float("nan"), dtype=mx.bfloat16), want)
    assert projkernel._stock_distance(got, want) == float("inf")
    assert not projkernel._stock_distance(want, got) <= projkernel.REL_RMS_BOUND
    assert projkernel._stock_distance(want, want) == 0.0


def test_probe_restores_the_flag_and_the_counters_when_it_raises(monkeypatch):
    def fails_through_the_hooks(x, weights):
        if projkernel._active:
            raise RuntimeError("device lost")
        return pfx.stand_in_mma(x, weights)

    lm = _probe_model(monkeypatch, fails_through_the_hooks)
    before = dict(projkernel.calls)
    with pytest.raises(RuntimeError, match="device lost"):
        projkernel.probe(lm)
    assert projkernel._active == 0
    assert projkernel.calls == before


def test_enable_refuses_a_failed_probe_with_its_cost_and_untags_the_model(monkeypatch):
    pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    tagged_when_probed: list = []

    def failing(model):
        tagged_when_probed.append(_tagged_count(model))
        return "correctness: lm_head is 0.101 from stock (bound 0.01)"

    monkeypatch.setattr(projkernel, "probe", failing)
    status = _refusal(lm, warned=True)
    assert status["reason"] == "correctness: lm_head is 0.101 from stock (bound 0.01)"
    assert isinstance(status["probe_seconds"], float)
    assert tagged_when_probed == [_PROJECTIONS]


def test_a_failed_probe_untags_only_the_model_it_was_given(monkeypatch):
    tile = pfx.proj_ready(monkeypatch)
    earlier = pfx.tiny_quantized_model()
    status = projkernel.enable(earlier, enabled=True, drafter_kind=None, tile_status=tile)
    assert status["state"] == "active", status
    monkeypatch.setattr(
        projkernel,
        "probe",
        lambda model: "parity: verify row 0 of 5 differs from the one-row forward",
    )
    _refusal(pfx.tiny_quantized_model(seed=1), warned=True)
    # An earlier load's tags are inert while the flag is clear, and never touched.
    assert _tagged_count(earlier) == _PROJECTIONS


def test_enable_refuses_an_exception_in_a_guard_with_its_error_line(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 1)
    error = RuntimeError(_BUILD_ERROR)

    def broken(model):
        raise error

    monkeypatch.setattr(projkernel, "_model_reason", broken)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "program_source:3:1: error: boom"
    assert status["probe_seconds"] is None
    # The warning keeps one line; the log keeps where it was raised.
    (record,) = _enabling_failures(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None and record.exc_info[1] is error


def test_enable_refuses_a_probe_that_raises_and_reports_what_it_cost(monkeypatch, caplog):
    pfx.proj_ready(monkeypatch)

    def fails_through_the_hooks(x, weights):
        if projkernel._active:
            raise RuntimeError("device lost")
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", fails_through_the_hooks)
    status = _refusal(pfx.tiny_quantized_model(), warned=True)
    assert status["reason"] == "device lost"
    assert isinstance(status["probe_seconds"], float)
    (record,) = _enabling_failures(caplog)
    assert record.exc_info is not None and "device lost" in str(record.exc_info[1])


@pytest.mark.parametrize("kind", [None, "dflash"])
def test_enable_warms_up_untagged_then_tags_hooks_and_probes(monkeypatch, caplog, kind):
    tile = pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    order: list = []
    monkeypatch.setattr(
        projkernel, "warm_up", lambda model: order.append(("warm-up", _tagged_count(model)))
    )
    monkeypatch.setattr(
        projkernel,
        "probe",
        lambda model: order.append(("probe", _tagged_count(model), _hooks_in_place())),
    )
    with caplog.at_level(logging.INFO, logger="sous.engine.projkernel"):
        status = projkernel.enable(lm, enabled=True, drafter_kind=kind, tile_status=tile)
    assert status == {"state": "active", "reason": None, "probe_seconds": status["probe_seconds"]}
    assert isinstance(status["probe_seconds"], float)
    assert order == [("warm-up", 0), ("probe", _PROJECTIONS, True)]
    assert projkernel._active == 1
    assert [r.getMessage() for r in caplog.records if r.name == "sous.engine.projkernel"] == [
        f"projection kernel: {_PROJECTIONS} projections, probe {status['probe_seconds']:.2f}s"
    ]


def test_enable_goes_active_through_the_real_warm_up_and_probe(monkeypatch):
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    status = projkernel.enable(
        pfx.tiny_quantized_model(), enabled=True, drafter_kind="dflash", tile_status=tile
    )
    assert status["state"] == "active", status
    assert isinstance(status["probe_seconds"], float)
    assert projkernel._active == 1


@pytest.mark.parametrize(
    "ending", ["disabled", "failed gate", "other model", "failed probe", "cleared"]
)
def test_enable_then_a_later_load_or_clear_leaves_verify_and_decode_stock(monkeypatch, ending):
    tile = pfx.proj_ready(monkeypatch)
    lm = pfx.tiny_quantized_model()
    ids = mx.random.randint(0, tfx.VOCAB, (43,), key=mx.random.key(5)).tolist()

    def verify() -> list:
        return tfx.verify_forward(lm, tfx.prefill_cache(lm, ids[:40]), ids[40:])

    def decode():
        cache = tfx.prefill_cache(lm, ids[:40])
        logits = lm(mx.array([ids[40:41]], dtype=mx.int32), cache=cache).logits
        mx.eval(logits)
        return logits

    verify_stock, decode_stock = verify(), decode()
    status = projkernel.enable(lm, enabled=True, drafter_kind=None, tile_status=tile)
    assert status["state"] == "active", status
    before = dict(projkernel.calls)
    verify()
    decode()
    # While active, both paths reach the kernel: what follows is not vacuous.
    assert projkernel.calls["verify_kernel"] - before["verify_kernel"] == _VERIFY_CALLS
    assert projkernel.calls["plain_kernel"] - before["plain_kernel"] == _PROJECTIONS

    if ending == "disabled":
        projkernel.enable(lm, enabled=False, drafter_kind=None, tile_status=tile)
    elif ending == "failed gate":
        _refusal(lm, warned=False, untagged=False, tile_status={"state": "off"})
    elif ending == "other model":
        monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
        _refusal(lm, warned=False, untagged=False)
    elif ending == "failed probe":
        monkeypatch.setattr(projkernel, "probe", lambda model: "parity: verify row 0 of 5 differs")
        _refusal(lm, warned=True)
    else:
        projkernel.clear()

    assert projkernel._active == 0
    before = dict(projkernel.calls)
    assert all(mx.array_equal(a, b).item() for a, b in zip(verify(), verify_stock, strict=True))
    assert mx.array_equal(decode(), decode_stock).item()
    for key in ("verify_kernel", "verify_split", "plain_kernel"):
        assert projkernel.calls[key] == before[key], key


_CONFIG_27B = {
    "model_type": "qwen3_5",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {"num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256},
}


def _model_config(**changes) -> dict:
    config = copy.deepcopy(_CONFIG_27B)
    for key, value in changes.items():
        config[key] = value
    return config


_STATIC_CASES = {
    "dflash": ({}, _CONFIG_27B, "dflash", None),
    "no drafter": ({}, _CONFIG_27B, None, None),
    "text-only layout": (
        {},
        {"model_type": "qwen3_5", "quantization": _CONFIG_27B["quantization"]}
        | _CONFIG_27B["text_config"],
        "dflash",
        None,
    ),
    "kernel off": ({"projection_kernel": False}, _CONFIG_27B, "dflash", "projection kernel off"),
    "tile off": ({"attention_tile": False}, _CONFIG_27B, "dflash", "attention tile off"),
    "another model type": (
        {},
        _model_config(model_type="qwen3_5_moe"),
        "dflash",
        "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
    ),
    "another attention shape": (
        {},
        _model_config(
            text_config={"num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256}
        ),
        "dflash",
        "unsupported attention shape 16/4/256 (24/4/256 only)",
    ),
    "unquantized": (
        {},
        {k: v for k, v in _CONFIG_27B.items() if k != "quantization"},
        "dflash",
        "unquantized weights (affine 4-bit gs64 only)",
    ),
    "mxfp4": (
        {},
        _model_config(quantization={"group_size": 32, "bits": 4, "mode": "mxfp4"}),
        "dflash",
        "quantization mxfp4 4-bit gs32 (affine 4-bit gs64 only)",
    ),
    "an 8-bit projection": (
        {},
        _model_config(
            quantization={
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "bits": 8,
                    "group_size": 64,
                },
            }
        ),
        "dflash",
        "projection language_model.model.layers.0.linear_attn.in_proj_qkv is affine 8-bit gs64"
        " (affine 4-bit gs64 only)",
    ),
    "an 8-bit embedding": (
        {},
        _model_config(
            quantization={
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
                "vision_tower.blocks.0.attn.qkv": {"bits": 8, "group_size": 64},
            }
        ),
        "dflash",
        None,
    ),
    "mtp drafter": ({}, _CONFIG_27B, "mtp", "drafter kind 'mtp' (dflash only)"),
}


@pytest.mark.parametrize("case", list(_STATIC_CASES))
def test_static_reason_reads_the_config_the_model_and_the_drafter(monkeypatch, case):
    pfx.proj_ready(monkeypatch)
    switches, model_config, drafter_kind, reason = _STATIC_CASES[case]
    config = types.SimpleNamespace(
        **{"attention_tile": True, "projection_kernel": True, **switches}
    )
    assert projkernel.static_reason(config, model_config, drafter_kind) == reason


@pytest.mark.parametrize("case", ["platform", "unreadable", "another GPU", "unmeasured"])
def test_static_reason_reads_the_machine_as_the_tile_does(monkeypatch, case):
    pfx.proj_ready(monkeypatch)
    device = str(mx.device_info()["device_name"])
    if case == "platform":
        monkeypatch.setattr(nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
        reason = "macOS 15.5 < 26.2"
    else:
        reading, reason = {
            "unreadable": (
                gpucores.GPUCores(None, None, "GPU core count unreadable: timed out", "none"),
                "GPU core count unreadable: timed out",
            ),
            "another GPU": (
                gpucores.GPUCores(20, "Apple M9 Ultra", None, "iokit"),
                f"IORegistry GPU 'Apple M9 Ultra' is not mlx's {device!r}",
            ),
            "unmeasured": (
                gpucores.GPUCores(16, device, None, "iokit"),
                "split target 16 not measured (measured: 20)",
            ),
        }[case]
        monkeypatch.setattr(gpucores, "read", lambda: reading)
    config = types.SimpleNamespace(attention_tile=True, projection_kernel=True)
    assert projkernel.static_reason(config, _CONFIG_27B, "dflash") == reason


def test_static_reason_reads_nothing_while_either_switch_is_off(monkeypatch):
    pfx.proj_ready(monkeypatch)

    def untouchable():
        raise AssertionError("read the machine for a switched-off kernel")

    monkeypatch.setattr(nax, "platform_reason", untouchable)
    monkeypatch.setattr(gpucores, "read", untouchable)
    for switches in ({"projection_kernel": False}, {"attention_tile": False}):
        config = types.SimpleNamespace(
            **{"attention_tile": True, "projection_kernel": True, **switches}
        )
        assert projkernel.static_reason(config, {}, "mtp") is not None


def test_static_reason_asks_the_tile_first_and_names_the_drafter_last(monkeypatch):
    """The kernel's check is the tile's plus its own: a refusal the tile
    makes comes first, then the quantization, then the drafter kind."""
    pfx.proj_ready(monkeypatch)
    config = types.SimpleNamespace(attention_tile=True, projection_kernel=True)
    mxfp4 = _model_config(quantization={"group_size": 32, "bits": 4, "mode": "mxfp4"})
    assert projkernel.static_reason(config, mxfp4, "mtp") == (
        "quantization mxfp4 4-bit gs32 (affine 4-bit gs64 only)"
    )
    moe = _model_config(model_type="qwen3_5_moe", quantization=mxfp4["quantization"])
    assert projkernel.static_reason(config, moe, "mtp") == (
        "unsupported model type 'qwen3_5_moe' (qwen3_5 only)"
    )


def test_static_reason_keeps_the_kernels_own_model_types(monkeypatch):
    """The tile's set and the kernel's are separate: a type only the tile
    takes must still be refused here, as enable() refuses it."""
    from sous.engine import tileattn

    pfx.proj_ready(monkeypatch)
    monkeypatch.setattr(
        tileattn, "SUPPORTED_MODEL_TYPES", tileattn.SUPPORTED_MODEL_TYPES | {"qwen3_5_moe"}
    )
    config = types.SimpleNamespace(attention_tile=True, projection_kernel=True)
    moe = _model_config(model_type="qwen3_5_moe")
    assert tileattn.static_reason(config, moe, "dflash") is None
    assert projkernel.static_reason(config, moe, "dflash") == (
        "unsupported model type 'qwen3_5_moe' (qwen3_5 only)"
    )


# --------------------------------------------------------------------------------
# The drafter's own linears: tagged apart, served from six rows
# --------------------------------------------------------------------------------


def _tiny_drafter(target):
    """A quantized DFlash drafter bound to `target`, as the engine loads one: eight
    linears of its own plus the target's lm_head."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.config import DFlashConfig
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    vocab = target.args.vocab_size
    drafter = DFlashDraftModel(
        DFlashConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=vocab,
            mask_token_id=vocab - 1,
            target_layer_ids=[1],
            num_target_layers=4,
        )
    )
    drafter.set_dtype(mx.bfloat16)
    nn.quantize(drafter, group_size=64, bits=4)
    drafter.bind(types.SimpleNamespace(language_model=target))
    return drafter


def test_the_drafter_tag_marks_its_own_linears_and_not_the_targets_head(fresh_target):
    drafter = _tiny_drafter(fresh_target)
    model = types.SimpleNamespace(language_model=fresh_target)
    projkernel.tag(model)
    assert projkernel.tag_drafter(drafter) == 8
    owned = [
        m
        for _, m in drafter.named_modules()
        if isinstance(m, nn.QuantizedLinear) and m is not fresh_target.lm_head
    ]
    assert all(projkernel._draft_tagged(m) and not projkernel._tagged(m) for m in owned)
    # The head DFlash binds in is the target's object: the target's tag, never the drafter's.
    assert projkernel._tagged(drafter.lm_head) and not projkernel._draft_tagged(drafter.lm_head)
    projkernel.untag_drafter(drafter)
    assert not any(projkernel._draft_tagged(m) for m in owned)
    projkernel.untag(model)


@pytest.mark.parametrize("rows", [6, 7, 8])
def test_the_module_hook_serves_a_drafter_tagged_module_from_six_rows(monkeypatch, rows):
    """Stock quantized_matmul falls off a cliff at eight rows (about 100 GB/s against
    the kernel's 175 on the drafter's shapes) and is as fast or faster below six, so
    the drafter's calls go to the kernel only from six rows up."""
    pfx.hooked(monkeypatch)
    monkeypatch.setattr(projkernel, "_draft_active", 1)
    linear = pfx.rand_module(256, 256, 0)
    object.__setattr__(linear, projkernel._DRAFT_TAG, True)
    x = _acts(1, rows, 256)
    before = dict(projkernel.calls)
    got = linear(x)
    assert _proj_counts_since(before) == {"draft_kernel": 1}
    assert _bitwise(got, pfx.stand_in_mma(x, _triples(linear))[0])


# case -> (draft flag, x, the counter it moves)
_DRAFT_HANDED_ON = {
    "flag clear": (0, lambda: _acts(1, 8, 256), None),
    "five rows": (1, lambda: _acts(1, 5, 256), None),
    "one row": (1, lambda: _acts(1, 1, 256), None),
    "nine rows": (1, lambda: _acts(1, 9, 256), None),
    "the context tail": (1, lambda: _acts(1, 2047, 256), None),
    "float16": (1, lambda: _acts(1, 8, 256, dtype=mx.float16), "declined"),
    # Only tagged modules route: with the drafter's flag set, any other
    # QuantizedLinear call of 6-8 bf16 rows in the process would otherwise
    # reach the kernel.
    "untagged module": (1, lambda: _acts(1, 8, 256), None),
}


@pytest.mark.parametrize("case", list(_DRAFT_HANDED_ON))
def test_the_module_hook_hands_the_drafters_other_calls_on(monkeypatch, case):
    flag, build, counter = _DRAFT_HANDED_ON[case]
    seen = []

    def original(self, x):
        seen.append((self, x))
        return "the wrapped method's answer"

    hook = projkernel._wrap_call(original)
    monkeypatch.setattr(projkernel, "_active", 1)
    monkeypatch.setattr(projkernel, "_draft_active", flag)
    monkeypatch.setattr(projkernel, "mma", _no_kernel)
    linear = pfx.rand_module(256, 256, 0)
    if case != "untagged module":
        object.__setattr__(linear, projkernel._DRAFT_TAG, True)
    x = build()
    before = dict(projkernel.calls)
    assert hook(linear, x) == "the wrapped method's answer"
    ((got_self, got_x),) = seen
    assert got_self is linear and got_x is x
    assert _proj_counts_since(before) == ({} if counter is None else {counter: 1})


def test_enable_drafter_tags_warms_probes_and_raises_its_own_flag(
    monkeypatch, caplog, fresh_target
):
    tile = pfx.proj_ready(monkeypatch)
    status = projkernel.enable(fresh_target, enabled=True, drafter_kind="dflash", tile_status=tile)
    assert status["state"] == "active", status
    drafter = _tiny_drafter(fresh_target)
    served = []
    serve_draft = projkernel._serve_draft

    def spy(module, x):
        out = serve_draft(module, x)
        served.append((module, x.shape, out is not None))
        return out

    monkeypatch.setattr(projkernel, "_serve_draft", spy)
    before = dict(projkernel.calls)
    with caplog.at_level(logging.INFO, logger="sous.engine.projkernel"):
        got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=8)
    assert got == {
        "state": "active",
        "reason": None,
        "linears": 8,
        "probe_seconds": got["probe_seconds"],
    }
    assert isinstance(got["probe_seconds"], float)
    assert projkernel._draft_active == 1 and projkernel._active == 1
    # The probe's last step is one call of a tagged drafter linear through the
    # hook, which the kernel served and counted, and which it then put back.
    ((module, shape, hit),) = served
    assert projkernel._draft_tagged(module) and hit
    assert shape == (1, projkernel.MAX_ROWS, module.weight.shape[1] * 8)
    assert projkernel.calls == before, "the warm-up and the probe leave the counters as found"
    assert [r.getMessage() for r in caplog.records if "drafter" in r.getMessage()] == [
        f"projection kernel drafter: 8 linears from {projkernel._DRAFT_MIN_ROWS} rows, "
        f"probe {got['probe_seconds']:.2f}s"
    ]
    # Served through the real hook at eight rows; handed on at five.
    linear = drafter.layers[0].mlp.gate_proj
    x8, x5 = _acts(1, 8, 256), _acts(1, 5, 256)
    before = dict(projkernel.calls)
    linear(x8)
    assert _proj_counts_since(before) == {"draft_kernel": 1}
    before = dict(projkernel.calls)
    linear(x5)
    assert _proj_counts_since(before) == {}
    projkernel.clear()
    assert projkernel._draft_active == 0 and projkernel._active == 0
    projkernel.untag_drafter(drafter)


def _drafter_untagged(drafter) -> bool:
    return not any(
        projkernel._draft_tagged(m)
        for _, m in drafter.named_modules()
        if isinstance(m, nn.QuantizedLinear)
    )


def _drafter_refusal(drafter, kernel_status, *, warned: bool, raised: str | None = None) -> dict:
    """enable_drafter() expected to refuse `unavailable`, reporting as the kernel's
    own refusals do: one where the route does not apply is one INFO line in the
    route's words and no warning; one where it should have run is exactly one
    warning, on one line, and no WARNING log line beside it but the traceback of
    whatever raised, which `raised` names. In the daemon a warning is itself a
    WARNING line, so a second one carrying the reason would print it twice. The
    flag is left clear and nothing on the drafter tagged."""
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        status = projkernel.enable_drafter(drafter, kernel_status=kernel_status, block_size=8)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    louder = [r for r in records if r.levelno >= logging.WARNING]
    assert status["state"] == "unavailable", status
    assert set(status) == {"state", "reason", "linears", "probe_seconds"}
    if warned:
        assert messages == [
            f"sous: projection kernel drafter unavailable ({status['reason']}); "
            "the drafter's own projections run stock"
        ]
        assert "\n" not in messages[0]
        assert infos == []
    else:
        assert messages == []
        assert infos == [f"projection kernel drafter unavailable: {status['reason']}"]
    assert [r.getMessage() for r in louder] == ([] if raised is None else [raised])
    assert all(r.exc_info is not None for r in louder), "only a traceback rides beside it"
    assert projkernel._draft_active == 0
    if hasattr(drafter, "layers"):
        assert _drafter_untagged(drafter)
    return status


@pytest.fixture
def drafted(monkeypatch, fresh_target):
    """(the tiny drafter bound to the tiny target, the kernel's active status):
    enable() has served the target, so the drafter route has a kernel to ride on."""
    tile = pfx.proj_ready(monkeypatch)
    status = projkernel.enable(fresh_target, enabled=True, drafter_kind="dflash", tile_status=tile)
    assert status["state"] == "active", status
    drafter = _tiny_drafter(fresh_target)
    yield drafter, status
    projkernel.untag_drafter(drafter)


def test_enable_drafter_is_off_while_the_kernel_is_switched_off(drafted):
    """The kernel's switch is the route's: switched off, it is `off`, which is
    never a refusal, and nothing is logged."""
    drafter, _ = drafted
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        got = projkernel.enable_drafter(
            drafter,
            kernel_status={"state": "off", "reason": None, "probe_seconds": None},
            block_size=8,
        )
    assert got == projkernel.drafter_off("projection kernel off")
    assert got == {
        "state": "off",
        "reason": "projection kernel off",
        "linears": 0,
        "probe_seconds": None,
    }
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert records == []
    assert projkernel._draft_active == 0 and _drafter_untagged(drafter)


@pytest.mark.parametrize(
    ("block", "kernel"),
    [(1, "active"), (3, "active"), (5, "active"), (3, "unavailable")],
)
def test_enable_drafter_is_off_below_its_floor(monkeypatch, drafted, block, kernel):
    """A verify block of 1-5 never feeds the drafter's calls six rows in steady
    state, so the route is `off` there, like a switch: nothing is warmed,
    probed, tagged or logged, so `active` means the route can serve. The block is
    the config's, so it answers before whether the kernel is active."""
    drafter, status = drafted
    if kernel != "active":
        status = {"state": kernel, "reason": "attention tile off", "probe_seconds": None}
    for step in ("_warm_up_drafter", "_probe_drafter", "_probe_drafter_hook", "tag_drafter"):
        ran = AssertionError(f"{step} ran below the floor")
        monkeypatch.setattr(projkernel, step, _raises(ran))
    with warnings.catch_warnings(record=True) as caught, _gate_log() as records:
        warnings.simplefilter("always")
        got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=block)
    assert got == projkernel.drafter_off(f"verify block {block} below the route's floor 6")
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    assert records == []
    assert projkernel._draft_active == 0 and _drafter_untagged(drafter)


@pytest.mark.parametrize("block", [0, 6, 7, 8, 12])
def test_enable_drafter_serves_from_its_floor_and_at_the_drafters_own_policy(drafted, block):
    """Block 0 lets the drafter's own policy pick the depth, which may exceed
    five rows (a v1 DFlash checkpoint's block of 16), so it proceeds like any
    block from six up; under the default DFlash2's runtime block of 5 it then
    serves nothing in steady state."""
    drafter, status = drafted
    got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=block)
    assert got["state"] == "active", got
    assert projkernel._draft_active == 1 and not _drafter_untagged(drafter)


def _requantized(drafter, path: str, **kwargs) -> None:
    """Swap the drafter's linear at `path` for a fresh one of the same shape,
    quantized as `kwargs` say: a checkpoint quantized otherwise survives
    nn.quantize, which leaves an existing QuantizedLinear alone."""
    parent, name = path.rsplit(".", 1)
    old = dict(drafter.named_modules())[path]
    n, k = old.weight.shape[0], old.weight.shape[1] * 8
    module = nn.QuantizedLinear(k, n, **{"bias": False, "group_size": 64, "bits": 4, **kwargs})
    setattr(dict(drafter.named_modules())[parent], name, module)


# case -> (how the drafter or the kernel's status is made to refuse, the reason)
_DRAFTER_DOES_NOT_APPLY: dict[str, tuple[Any, str]] = {
    "kernel unavailable": ("attention tile off", "attention tile off"),
    "kernel unavailable, no reason": (None, "projection kernel unavailable"),
    "no linears": ("no linears", "the drafter owns no quantized linear"),
    "group size 32": (
        {"group_size": 32},
        "drafter linear layers.0.mlp.gate_proj (K=256 N=512) has group size 32 (64 only)",
    ),
    "group size 128": (
        {"group_size": 128},
        "drafter linear layers.0.mlp.gate_proj (K=256 N=512) has group size 128 (64 only)",
    ),
    "8-bit": (
        {"bits": 8},
        "drafter linear layers.0.mlp.gate_proj (K=256 N=512) is 8-bit (4-bit only)",
    ),
    "a bias": (
        {"bias": True},
        "drafter linear layers.0.mlp.gate_proj (K=256 N=512) has a bias term",
    ),
    "float32 scales": (
        {},
        "drafter linear layers.0.mlp.gate_proj (K=256 N=512) has mlx.core.float32 scales "
        "and mlx.core.float32 biases (bfloat16 only)",
    ),
}


@pytest.mark.parametrize("case", list(_DRAFTER_DOES_NOT_APPLY))
def test_enable_drafter_refuses_a_load_it_does_not_apply_to_in_one_info_line(drafted, case):
    """Where the kernel is not active, or the drafter holds no linear the kernel
    takes, the route says so the way the kernel says it of a tile that is not
    active: `unavailable`, one INFO line, no warning. An ineligible linear is
    named with the K and N its packed weight holds and the requirement it
    misses; K off the scales would read 512 at group size 32 and 128 at 128."""
    drafter, status = drafted
    how, reason = _DRAFTER_DOES_NOT_APPLY[case]
    if case.startswith("kernel unavailable"):
        status = {"state": "unavailable", "reason": how, "probe_seconds": None}
    elif how == "no linears":
        drafter = types.SimpleNamespace(named_modules=lambda: [])
    else:
        _requantized(drafter, "layers.0.mlp.gate_proj", **how)
    got = _drafter_refusal(drafter, status, warned=False)
    assert got["reason"] == reason
    assert got["probe_seconds"] is None
    if case.startswith("kernel unavailable"):
        assert projkernel._active == 1, "the target's own route is untouched"


def test_enable_drafter_warns_once_for_a_failed_probe(monkeypatch, drafted):
    drafter, status = drafted
    monkeypatch.setattr(
        projkernel, "_probe_drafter", lambda linears: "correctness: K=256 N=512 at T=8 is off"
    )
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"] == "correctness: K=256 N=512 at T=8 is off"
    assert got["linears"] == 8 and isinstance(got["probe_seconds"], float)
    assert projkernel._active == 1, "the target's own route is untouched"


def _raises(error):
    def raising(*args, **kwargs):
        raise error

    return raising


# case -> (the step that raises, the reason, the traceback line beside the warning)
_DRAFTER_RAISES = {
    "warm-up": (
        "_warm_up_drafter",
        "warm-up: program_source:3:1: error: boom",
        "projection kernel: the drafter's warm-up failed",
    ),
    "probe": (
        "_probe_drafter",
        "probe: program_source:3:1: error: boom",
        "projection kernel: the drafter's probe failed",
    ),
    "a guard": (
        "_drafter_reason",
        "program_source:3:1: error: boom",
        "projection kernel: enabling the drafter route failed",
    ),
}


@pytest.mark.parametrize("case", list(_DRAFTER_RAISES))
def test_enable_drafter_warns_once_for_a_step_that_raises_and_logs_its_traceback(
    monkeypatch, drafted, case
):
    """A raise in the warm-up or the probe is reported as that step's, never the
    other's, and every reason is the error's one line; the traceback goes to the
    log as the kernel's own does."""
    drafter, status = drafted
    step, reason, line = _DRAFTER_RAISES[case]
    error = RuntimeError(_BUILD_ERROR)
    monkeypatch.setattr(projkernel, step, _raises(error))
    with warnings.catch_warnings(record=True), _gate_log() as records:
        warnings.simplefilter("always")
        got = _drafter_refusal(drafter, status, warned=True, raised=line)
    assert got["reason"] == reason
    (record,) = [r for r in records if r.levelno >= logging.WARNING]
    assert record.exc_info is not None and record.exc_info[1] is error
    if case != "a guard":
        assert isinstance(got["probe_seconds"], float)


def _recording(seen: set):
    """The stand-in kernel, recording (K, N, rows) for every linear it is asked for."""

    def record(x, weights):
        seen.update((x.shape[-1], w.shape[0], _rows(x)) for w, _, _ in weights)
        return pfx.stand_in_mma(x, weights)

    return record


def test_the_drafters_warm_up_and_probe_cover_every_row_count_the_hook_serves(monkeypatch, drafted):
    """Six, seven and eight rows on every (K, N) the drafter holds, in both: a
    shape that fails to compile, or that the kernel computes wrongly, at seven
    rows would otherwise go live."""
    drafter, _ = drafted
    linears = projkernel._drafter_linears(drafter)
    shapes = {(m.weight.shape[1] * 8, m.weight.shape[0]) for m in linears}
    want = {(k, n, t) for k, n in shapes for t in range(projkernel._DRAFT_MIN_ROWS, 9)}
    assert {t for *_, t in want} == {6, 7, 8}
    warmed: set = set()
    monkeypatch.setattr(projkernel, "mma", _recording(warmed))
    projkernel._warm_up_drafter(linears)
    assert warmed == want
    probed: set = set()
    monkeypatch.setattr(projkernel, "mma", _recording(probed))
    assert projkernel._probe_drafter(linears) is None
    assert probed == want


@pytest.mark.parametrize("rows", [6, 7, 8])
def test_enable_drafter_refuses_a_kernel_wrong_at_one_row_count(monkeypatch, drafted, rows):
    """The probe's negative control: a kernel wrong in one place, at one row
    count the hook serves, refuses the route with a warning and untags it."""
    drafter, status = drafted

    def wrong_at(x, weights):
        outs = pfx.stand_in_mma(x, weights)
        return _scaled(outs, 1.1) if _rows(x) == rows else outs

    monkeypatch.setattr(projkernel, "mma", wrong_at)
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"].startswith("correctness: K="), got
    assert f" at T={rows} is " in got["reason"], got
    assert projkernel._active == 1, "the target's own route is untouched"


def test_enable_drafter_refuses_a_kernel_that_writes_a_nan(monkeypatch, drafted):
    drafter, status = drafted
    nan = mx.array(float("nan"), dtype=mx.bfloat16)

    def nan_at_eight(x, weights):
        outs = pfx.stand_in_mma(x, weights)
        if _rows(x) != projkernel.MAX_ROWS:
            return outs
        return tuple(mx.where(mx.arange(o.shape[-1]) == 3, nan, o) for o in outs)

    monkeypatch.setattr(projkernel, "mma", nan_at_eight)
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"].startswith("correctness: K="), got
    assert got["reason"].endswith(f" at T=8 is inf from stock (bound {projkernel.REL_RMS_BOUND})")


def test_enable_drafter_refuses_a_distance_that_is_nan(monkeypatch, drafted):
    """A NaN distance fails `<=` against any bound; `>` would wave it through."""
    drafter, status = drafted
    monkeypatch.setattr(projkernel, "_stock_distance", lambda got, want: float("nan"))
    got = _drafter_refusal(drafter, status, warned=True)
    assert " is nan from stock " in got["reason"], got


class _Bypassing(nn.QuantizedLinear):
    """A QuantizedLinear subclass whose own __call__ goes straight to
    quantized_matmul, past nn.QuantizedLinear.__call__: the case the hook step
    covers. A forward that reads the weights without calling the module at all
    (a fused k/v) is the drafter pins' to catch, not this step's."""

    def __call__(self, x):
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


def test_enable_drafter_refuses_a_drafter_whose_calls_bypass_the_hook(drafted):
    """The warm-up and the probe call the kernel directly; only the probe's last
    step proves that a drafter linear's own __call__ reaches the hook."""
    drafter, status = drafted
    for module in projkernel._drafter_linears(drafter):
        module.__class__ = _Bypassing
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"] == "the drafter's linears do not reach the hook"
    assert isinstance(got["probe_seconds"], float)
    assert projkernel._active == 1, "the target's own route is untouched"


def test_the_drafters_hook_step_leaves_the_flag_clear_and_the_counters_as_found(
    monkeypatch, drafted
):
    drafter, _ = drafted
    linears = projkernel._drafter_linears(drafter)
    projkernel.tag_drafter(drafter)
    flags = []
    real = projkernel._serve_draft

    def spy(module, x):
        flags.append(projkernel._draft_active)
        return real(module, x)

    monkeypatch.setattr(projkernel, "_serve_draft", spy)
    before = dict(projkernel.calls)
    assert projkernel._probe_drafter_hook(linears) is None
    assert flags == [1], "the flag is raised around the one call"
    assert projkernel._draft_active == 0
    assert projkernel.calls == before


def test_enable_clears_a_drafter_route_left_from_an_earlier_load(monkeypatch, drafted):
    """A load with no DFlash drafter after one with it never reaches
    enable_drafter(), so only enable() itself can clear the drafter's flag: else
    the earlier drafter's tagged linears would keep riding a kernel that this
    load's checks never approved for them."""
    drafter, status = drafted
    got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=8)
    assert got["state"] == "active" and projkernel._draft_active == 1
    later = pfx.tiny_quantized_model(seed=1)
    tile = dict(pfx.ACTIVE_TILE)
    again = projkernel.enable(later, enabled=True, drafter_kind=None, tile_status=tile)
    assert again["state"] == "active", again
    assert projkernel._draft_active == 0
    # The earlier drafter's linears are still tagged, and now handed on.
    linear = drafter.layers[0].mlp.gate_proj
    assert projkernel._draft_tagged(linear)
    before = dict(projkernel.calls)
    linear(_acts(1, 8, 256))
    assert _proj_counts_since(before) == {}
    projkernel.untag(later)


@pytest.mark.parametrize("ending", ["kernel unavailable", "block below the floor"])
def test_enable_drafter_clears_its_flag_before_it_decides(drafted, ending):
    """A second decision that refuses, or is off, leaves the flag clear whatever
    the first one set."""
    drafter, status = drafted
    assert projkernel.enable_drafter(drafter, kernel_status=status, block_size=8)["state"] == (
        "active"
    )
    assert projkernel._draft_active == 1
    if ending == "kernel unavailable":
        unavailable = {"state": "unavailable", "reason": "attention tile off"}
        got = projkernel.enable_drafter(drafter, kernel_status=unavailable, block_size=8)
    else:
        got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=5)
    assert got["state"] in ("unavailable", "off"), got
    assert projkernel._draft_active == 0


_REAL_DRAFT_PINS = projkernel._draft_pins_reason


def _tiny_dflash2_drafter(cls=None):
    """An unbound DFlash2 drafter (or one of `cls`, a subclass), the default
    drafter's class, quantized as the engine quantizes one: its own linears are
    the kernel's to take, and its forward's classes are there to read."""
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    drafter = (cls or DFlash2DraftModel)(
        DFlash2Config(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=512,
            mask_token_id=511,
            target_layer_ids=[1],
            num_target_layers=4,
            block_size=8,
            conv_kernel_size=3,
            conv_group_size=64,
            selector_rank=64,
            selector_top_k=4,
        )
    )
    drafter.set_dtype(mx.bfloat16)
    nn.quantize(drafter, group_size=64, bits=4)
    return drafter


def test_the_drafter_pins_hold_on_the_locked_mlx_vlm(fresh_target):
    """Every function between the round loop's draft call and the drafter's
    linears, for the v1 DFlash and the DFlash2 drafters, read through each
    loaded drafter's own classes."""
    dflash = "mlx_vlm.speculative.drafters.qwen3_dflash.dflash"
    dflash2 = "mlx_vlm.speculative.drafters.dflash2.dflash2"
    assert set(projkernel.VALIDATED_DRAFTERS) == {
        f"{dflash}:DFlashDraftModel",
        f"{dflash2}:DFlash2DraftModel",
    }
    assert set(projkernel.VALIDATED_DRAFT_SOURCES) == {
        f"{dflash}:DFlashDraftModel.draft_block",
        f"{dflash}:DFlashDraftModel.draft_block_greedy",
        f"{dflash}:DFlashDraftModel._hidden",
        f"{dflash}:DFlashDraftModel._project_context_kv",
        f"{dflash}:DFlashDecoderLayer.__call__",
        f"{dflash}:DFlashAttention.__call__",
        f"{dflash}:DFlashAttention._project_kv",
        f"{dflash}:Qwen3MLP.__call__",
        f"{dflash2}:DFlash2DraftModel.draft_block",
        f"{dflash2}:DFlash2DecoderLayer.__call__",
        f"{dflash2}:GroupedDynamicCausalConv.prepare",
        f"{dflash2}:CandidateSelector.select",
    }
    from sous.engine import verifyattn

    for key, digest in projkernel.VALIDATED_DRAFT_SOURCES.items():
        module, qualname = key.split(":")
        assert verifyattn._source_digest(module, qualname) == digest, key
    assert _REAL_DRAFT_PINS(_tiny_drafter(fresh_target)) is None
    assert _REAL_DRAFT_PINS(_tiny_dflash2_drafter()) is None


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        (
            "mlx_vlm.speculative.drafters.qwen3_dflash.dflash:DFlashDraftModel._project_context_kv",
            "mlx-vlm DFlashDraftModel._project_context_kv changed",
        ),
        (
            "mlx_vlm.speculative.drafters.qwen3_dflash.dflash:DFlashAttention._project_kv",
            "mlx-vlm DFlashAttention._project_kv changed",
        ),
        (
            "mlx_vlm.speculative.drafters.qwen3_dflash.dflash:Qwen3MLP.__call__",
            "mlx-vlm Qwen3MLP.__call__ changed",
        ),
    ],
)
def test_enable_drafter_refuses_a_changed_drafter_forward(monkeypatch, drafted, key, reason):
    """A drafter forward that might no longer call its linears as modules, a
    fused k/v or a direct quantized_matmul, is a route that should have run:
    one warning, before anything is warmed."""
    drafter, status = drafted
    monkeypatch.setattr(projkernel, "_draft_pins_reason", _REAL_DRAFT_PINS)
    monkeypatch.setitem(projkernel.VALIDATED_DRAFT_SOURCES, key, "0" * 64)
    monkeypatch.setattr(
        projkernel, "_warm_up_drafter", _raises(AssertionError("warmed past drifted pins"))
    )
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"] == reason
    assert got["probe_seconds"] is None


def test_enable_drafter_warns_for_a_validated_drafter_whose_component_overrides_a_pin(
    monkeypatch, drafted
):
    """The pins are read through the loaded drafter's own classes: inside a
    validated drafter class, an attention that fuses its k and v reads as
    mlx-vlm drift, not as the pinned DFlashAttention."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashAttention

    class _FusedKV(DFlashAttention):
        def _project_kv(self, x):
            return super()._project_kv(x)

    drafter, status = drafted
    monkeypatch.setattr(projkernel, "_draft_pins_reason", _REAL_DRAFT_PINS)
    drafter.layers[0].self_attn.__class__ = _FusedKV
    got = _drafter_refusal(drafter, status, warned=True)
    assert got["reason"] == f"mlx-vlm {_FusedKV._project_kv.__qualname__} changed"


def test_enable_drafter_serves_the_default_drafters_class(monkeypatch, drafted):
    """DFlash2, the default drafter's class, through the real pins, eligibility,
    warm-up, probe and hook step."""
    _, status = drafted
    monkeypatch.setattr(projkernel, "_draft_pins_reason", _REAL_DRAFT_PINS)
    drafter = _tiny_dflash2_drafter()
    got = projkernel.enable_drafter(drafter, kernel_status=status, block_size=8)
    assert got["state"] == "active", got
    assert got["linears"] == len(projkernel._drafter_linears(drafter))
    projkernel.untag_drafter(drafter)


def test_enable_drafter_does_not_apply_to_a_drafter_class_it_was_not_validated_with(
    monkeypatch, drafted
):
    """A drafter of another class (here a DFlash2 subclass with its own _hidden)
    runs a forward nobody read against the hook: the route does not apply, one
    INFO line, no warning, before anything is warmed."""
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    class _OwnHidden(DFlash2DraftModel):
        def _hidden(self, inputs, target_hidden, cache):
            return super()._hidden(inputs, target_hidden, cache)

    _, status = drafted
    monkeypatch.setattr(projkernel, "_draft_pins_reason", _REAL_DRAFT_PINS)
    monkeypatch.setattr(
        projkernel, "_warm_up_drafter", _raises(AssertionError("warmed an unvalidated drafter"))
    )
    got = _drafter_refusal(_tiny_dflash2_drafter(_OwnHidden), status, warned=False)
    assert got["reason"] == f"drafter {_OwnHidden.__qualname__} not validated"
    assert got["probe_seconds"] is None
