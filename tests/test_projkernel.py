"""The small-row projection kernel (sous.engine.projkernel): its Metal source, the
variant choice, eligibility, availability, and the arithmetic every routing
decision rests on. The arithmetic tests run wherever the kernel compiles and
probes right (CI's macos-15 GPU included) and skip elsewhere."""

import importlib
import inspect
import types
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


def test_the_counters_name_the_four_paths():
    assert set(projkernel.calls) == {"verify_kernel", "verify_split", "plain_kernel", "declined"}


def test_clear_drops_the_flag(monkeypatch):
    monkeypatch.setattr(projkernel, "_active", 1)
    projkernel.clear()
    assert projkernel._active == 0


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


def test_the_drafters_own_linears_are_never_tagged(fresh_target):
    """DFlash binds the target's lm_head into the drafter: that head is the target's
    object and is tagged; every linear the drafter owns is not."""
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
