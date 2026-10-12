"""kv8 (sous.engine.kv8): int8 K/V and Q through the attention tile.

The quantiser needs no tensor units, so its tests run on any Metal GPU, CI's
macos-15 included. The int8 split kernel does: its entries run everywhere with a
plain-mlx stand-in, and the `nax`-marked tests run the real kernel where kv8's own
availability() passes.
"""

import contextlib
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, kv8, nax, tileattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402

N0 = tileattn.N0
HQ, HKV, D = tileattn.HQ, tileattn.HKV, tileattn.D


def _bf16(shape, seed, scale=1.0):
    x = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _assert_quantizes_like_the_reference(k, v, out_rows=None):
    rows = k.shape[2]
    codes, sc = kv8.quantize(k, v, out_rows or rows)
    want_codes, want_sc = kfx.reference_quantize(k, v)
    mx.eval(codes, sc, want_codes, want_sc)
    assert codes.dtype == mx.int8 and sc.dtype == mx.float16
    assert mx.array_equal(codes[:, :, :rows], want_codes).item()
    assert mx.array_equal(sc[:, :, :rows], want_sc).item()
    return codes[:, :, :rows], sc[:, :, :rows]


# ---- the quantiser ----------------------------------------------------------------


@pytest.mark.parametrize(("rows", "scale"), [(1, 1.0), (5, 0.03), (37, 4.0), (300, 1.0)])
def test_the_quantiser_matches_the_reference_on_random_rows(rows, scale):
    _assert_quantizes_like_the_reference(
        _bf16((1, HKV, rows, D), rows, scale), _bf16((1, HKV, rows, D), 100 + rows, 2 * scale)
    )


def test_the_quantiser_reads_views_of_a_padded_buffer_in_place():
    """The cache hands over [..., a:b, :] slices of a buffer padded in 256-row steps."""
    keys = _bf16((1, HKV, 768, D), 1)
    values = _bf16((1, HKV, 768, D), 2)
    view_k, view_v = keys[:, :, 300:437], values[:, :, 300:437]
    _assert_quantizes_like_the_reference(view_k, view_v, out_rows=512)


def test_tie_rows_round_half_to_even():
    """Every element but each row's first divides by the scale onto k + 1/2 exactly,
    so the codes are decided by the rounding rule alone."""
    x = kfx.tie_rows(8)
    scale = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True) / 127
    frac = (x.astype(mx.float32) / scale)[..., 1:] % 1
    assert mx.all(frac == 0.5).item()
    codes, _ = _assert_quantizes_like_the_reference(x, kfx.tie_rows(8, seed=1))
    ties = codes[..., 1:D].astype(mx.float32)
    want = mx.round((x.astype(mx.float32) / scale)[..., 1:])
    assert mx.array_equal(ties, want).item() and mx.all(ties % 2 == 0).item()


def test_mx_round_is_round_half_to_even_on_both_streams():
    """The reference quantiser's rounding is Metal's rint only while mx.round rounds
    half to even: a change in mlx fails here, not in a tie-row comparison."""
    ties = mx.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 126.5])
    want = [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 126.0]
    assert mx.round(ties, stream=mx.gpu).tolist() == want
    assert mx.round(ties, stream=mx.cpu).tolist() == want


_SPECIAL_ROWS = {
    # amax / 127 is 0: the row divides by 1 and every code is 0
    "zero": lambda shape, seed: mx.zeros(shape, dtype=mx.bfloat16),
    # amax / 127 rounds to 0 in fp16 (below 2**-25)
    "underflow": lambda shape, seed: _bf16(shape, seed, 1e-7),
    # amax / 127 is an fp16 subnormal
    "subnormal scale": lambda shape, seed: _bf16(shape, seed, 1e-4),
    # amax / 127 passes fp16's largest finite value: the scale saturates at 65504
    "saturating": lambda shape, seed: _bf16(shape, seed, 1e8),
}


@pytest.mark.parametrize("kind", list(_SPECIAL_ROWS))
def test_the_quantiser_matches_the_reference_on_special_rows(kind):
    shape = (1, HKV, 3, D)
    k = _SPECIAL_ROWS[kind](shape, 7)
    v = _bf16(shape, 8)
    codes, sc = _assert_quantizes_like_the_reference(k, v)
    if kind in ("zero", "underflow"):
        assert not mx.any(codes[..., :D]).item() and not mx.any(sc[..., 0]).item()
    if kind == "saturating":
        assert mx.all(sc[..., 0] == 65504).item()
        assert mx.all(mx.isfinite(sc)).item()


def test_bulk_equals_incremental():
    """A first build over a whole prefix and the same rows quantised in two later
    pieces give the same bits: each row depends on itself alone."""
    keys, values = _bf16((1, HKV, 256, D), 3), _bf16((1, HKV, 256, D), 4)
    bulk_codes, bulk_sc = kv8.quantize(keys[:, :, :50], values[:, :, :50], 256)
    first = kv8.quantize(keys[:, :, :17], values[:, :, :17], 17)
    second = kv8.quantize(keys[:, :, 17:50], values[:, :, 17:50], 33)
    mx.eval(bulk_codes, bulk_sc, first, second)
    pieces = mx.concatenate([first[0], second[0]], axis=2)
    assert mx.array_equal(bulk_codes[:, :, :50], pieces).item()
    assert mx.array_equal(bulk_sc[:, :, :50], mx.concatenate([first[1], second[1]], axis=2)).item()


@pytest.mark.parametrize(("rows", "out_rows"), [(5, 4), (0, 4)])
def test_quantize_refuses_rows_outside_one_to_out_rows(rows, out_rows):
    k = mx.zeros((1, HKV, rows, D), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="kv8 quantises"):
        kv8.quantize(k, k, out_rows)


def test_dequant_is_each_code_times_its_rows_scale():
    k, v = _bf16((1, HKV, 6, D), 5), _bf16((1, HKV, 6, D), 6, 3.0)
    codes, sc = kv8.quantize(k, v, 6)
    dk, dv = kv8.dequant(codes, sc)
    assert dk.shape == dv.shape == (1, HKV, 6, D) and dk.dtype == mx.float32
    want_k = codes[..., :D].astype(mx.float32) * sc[..., :1].astype(mx.float32)
    assert mx.array_equal(dk, want_k).item()
    # the format's own error on Gaussian rows: about 1 % relative RMS
    error = mx.sqrt(mx.mean((dv - v.astype(mx.float32)) ** 2) / mx.mean(v.astype(mx.float32) ** 2))
    assert error.item() < 0.02


# ---- kernel source and the kernel objects ----------------------------------------


def _text(name: str) -> str:
    return files("sous.engine.kernels").joinpath(name).read_text()


def test_the_quantiser_source_is_mpp_free():
    text = _text("kv8_quant.metal")
    assert "#include" not in text and "mpp::" not in text
    assert "rint(" in text and "65504.0f" in text and "k_strides" in text


def test_the_split_kernel_source_is_the_relay_over_int8_operands():
    text = _text("attention_tile_kv8.metal")
    assert "#include" not in text and "// REDUCE" not in text
    assert "nax_coord(lane)" in text and "codes_strides" in text and "sc_strides" in text
    # MSL 4.1 gives locals a generic address space: every local cooperative-tensor type
    # is stripped of it, or macOS 27's compiler rejects the operands.
    assert text.count("metal::remove_addrspace_t<decltype(") == 4
    assert text.count("<int8_t, int8_t, int32_t>()") == 2
    assert text.count("<PE, int8_t, float>()") == 2
    # One barrier for Q's row maxima, then the relay's two per block.
    assert text.count("threadgroup_barrier(") == 3
    # qcodes exists only in the debug kernel object: no code outside the QDBG block
    # may name it, or the production object would not compile.
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    before, rest = code.split("#ifdef QDBG")
    inside, after = rest.split("#endif", 1)
    assert "qcodes" in inside and "qcodes" not in before + after


@pytest.fixture
def fresh_kernels():
    """kv8's and the tile's kernel objects built afresh inside the test, and forgotten
    after it, so the recording kernels a test builds never serve a later one."""
    caches = (kv8._kernels, kv8._qdbg_split, tileattn._kernel_pair)
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def _recording(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs

        def launch(**call):
            return [
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(call["output_shapes"], call["output_dtypes"], strict=True)
            ]

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    return made


def test_the_kernels_are_built_with_safe_math_and_read_strides(monkeypatch, fresh_kernels):
    made = _recording(monkeypatch)
    quant, split, reduce = kv8._kernels()
    assert reduce is tileattn._kernel_pair()[1]
    q, s = made["sous_kv8_quant"], made["sous_kv8_tile_split"]
    assert (q["input_names"], q["output_names"]) == (["k", "v", "params"], ["codes", "sc"])
    assert (s["input_names"], s["output_names"]) == (
        ["q", "codes", "sc", "params"],
        ["part", "stats"],
    )
    for kwargs in (q, s):
        assert kwargs["ensure_row_contiguous"] is False
        assert kwargs["compile_options"] == {"math_mode": "safe"}
    assert q["header"] == _text("common.h")
    assert s["header"] == _text("common.h") + _text("nax.h")
    assert "#define QDBG" not in s["header"]


def test_the_debug_kernel_only_adds_the_qcodes_output(monkeypatch, fresh_kernels):
    made = _recording(monkeypatch)
    kv8._kernels()
    kv8._qdbg_split()
    split, debug = made["sous_kv8_tile_split"], made["sous_kv8_tile_split_qdbg"]
    assert debug["output_names"] == [*split["output_names"], "qcodes"]
    assert debug["input_names"] == split["input_names"] and debug["source"] == split["source"]
    assert debug["header"] == split["header"] + "#define QDBG 1\n"
    assert debug["compile_options"] == split["compile_options"]


def test_launches_after_the_first_do_no_kernel_text_work(monkeypatch, fresh_kernels):
    """tile8() runs once per full-attention layer per forward: the kernel sources are
    read once per process and the kernels built once, never per launch."""
    made = _recording(monkeypatch)
    read = []
    source = kv8._kernel_text

    def counted(name):
        read.append(name)
        return source(name)

    monkeypatch.setattr(kv8, "_kernel_text", counted)
    queries, codes, sc, _, _ = kfx.serving_codes(2000, 3)
    for _ in range(3):
        assert kv8.tile8(queries, codes, sc, 2000, 96).shape == (1, 3, HQ, D)
    assert set(made) == {
        "sous_kv8_quant",
        "sous_kv8_tile_split",
        "sous_attention_tile_split",
        "sous_attention_tile_reduce",
    }
    assert read == ["common.h", "kv8_quant.metal", "attention_tile_kv8.metal", "nax.h"]


# ---- the entries, with the stand-in (and the real kernel where it compiles) ------


@pytest.fixture(params=[pytest.param("stand-in"), pytest.param("real", marks=kfx.nax)])
def kernel8(request, monkeypatch):
    """The plain-mlx stand-in everywhere; the Metal kernel where kv8 is available."""
    if request.param == "stand-in":
        monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    return request.param


def test_the_entries_return_the_models_layouts(kernel8):
    q, codes, sc, _, _ = kfx.serving_codes(2000, 1)
    assert kv8.tile8(q, codes, sc, 2000, 96).shape == (1, 1, HQ, D)
    assert kv8.decode_codes(q, codes, sc, 2000, 20).shape == (1, HQ, 1, D)
    for n, t in ((2000, 5), (N0 + 7, 5), (2000, 12)):
        q, codes, sc, _, _ = kfx.serving_codes(n, t)
        out = kv8.verify_codes(q, codes, sc, n, 20)
        assert out.shape == (1, HQ, t, D) and out.dtype == mx.bfloat16


def test_decode_codes_refuses_fewer_than_n0_keys():
    q, codes, sc, _, _ = kfx.serving_codes(N0 - 1, 1)
    with pytest.raises(ValueError, match="kv8 serves from N0"):
        kv8.decode_codes(q, codes, sc, N0 - 1, 20)


def test_verify_codes_refuses_a_run_below_n0():
    """N0 + 2 keys over 5 rows: the first two rows see 1534 and 1535 keys."""
    q, codes, sc, _, _ = kfx.serving_codes(N0 + 2, 5)
    with pytest.raises(ValueError, match="got a row at 1534"):
        kv8.verify_codes(q, codes, sc, N0 + 2, 20)


def test_verify_codes_takes_the_callers_plan(monkeypatch):
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    n, t = 1930, 12
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    runs = tileattn.verify_plan(n - t, t, 20)
    assert len(runs) == 3  # cut at 8 rows and at the 1921 step boundary
    planned = kv8.verify_codes(q, codes, sc, n, 20, runs)
    assert mx.array_equal(planned, kv8.verify_codes(q, codes, sc, n, 20)).item()


@pytest.mark.parametrize(
    ("n", "t", "chunk"), [(2000, 3, 96), (2000, 8, 96), (4096, 8, 224), (1543, 5, 64)]
)
def test_a_stand_in_row_is_the_one_row_call_at_its_own_length(n, t, chunk):
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    out = kfx.stand_in_tile8(q, codes, sc, n, chunk)
    assert out.shape == (1, t, HQ, D) and out.dtype == mx.bfloat16
    for r in range(t):
        m = n - t + r + 1
        one = kfx.stand_in_tile8(q[:, :, r : r + 1], codes, sc, m, chunk)
        assert mx.array_equal(out[:, r], one[:, 0]).item(), r


@pytest.mark.parametrize(("n", "chunk"), [(2000, 96), (4096, 224), (1537, 64)])
def test_a_stand_in_row_moves_with_its_chunk(n, chunk):
    """So an equality between two paths proves they used the same chunk."""
    q, codes, sc, _, _ = kfx.serving_codes(n, 1)
    same_step = kfx.stand_in_tile8(q, codes, sc, n, chunk)
    next_step = kfx.stand_in_tile8(q, codes, sc, n, chunk + tileattn.GRAN)
    assert not mx.array_equal(same_step, next_step).item()


@pytest.mark.parametrize(("n", "t"), [(2000, 1), (2000, 5), (4096, 8)])
def test_the_stand_in_is_the_kernels_arithmetic(n, t):
    """The stand-in sits within the kernel's bound of the fp32 reference over the
    same codes, so the contract tests that run on it run on the kernel's numbers."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    got = kfx.stand_in_tile8(q, codes, sc, n, tileattn.chunk_for(n, 20))
    want = kfx.reference_attention(q, codes, sc, n)
    assert _relative_rms(got.transpose(0, 2, 1, 3), want) <= kv8.REL_RMS_BOUND


def _relative_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2))).item()


def _poisoned_past(codes, sc, n):
    """The mirror with -128 codes and NaN scales from row n to its capacity."""
    cap = codes.shape[2]
    bad_codes = mx.full((1, HKV, cap - n, 2 * D), -128, dtype=mx.int8)
    bad_sc = mx.full((1, HKV, cap - n, 2), float("nan"), dtype=mx.float16)
    poisoned = (
        mx.concatenate([codes[:, :, :n], bad_codes], axis=2),
        mx.concatenate([sc[:, :, :n], bad_sc], axis=2),
    )
    mx.eval(poisoned)
    return poisoned


def _assert_fenced(n, t):
    """-128 codes and NaN scales planted from n to the capacity, read through
    [:, :, :n] views of the capacity-sized buffers as the mirror hands them over (the
    scale plane's head stride stays a multiple of 4 rows), give the clean output."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    want = kv8.verify_codes(q, codes, sc, n, 20)
    bad_codes, bad_sc = _poisoned_past(codes, sc, n)
    assert bad_codes.shape[2] % 256 == 0 and mx.all(mx.isnan(bad_sc[:, :, n:])).item()
    got = kv8.verify_codes(q, bad_codes[:, :, :n], bad_sc[:, :, :n], n, 20)
    assert mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()


@pytest.mark.parametrize(("n", "t"), [(N0, 1), (2000, 3), (1600, 8), (1548, 12)])
def test_codes_and_scales_past_n_change_nothing(kernel8, n, t):
    _assert_fenced(n, t)


@pytest.mark.parametrize(("n", "t"), [(N0, 1), (2000, 3), (1600, 8), (1548, 12)])
def test_reads_stay_below_n(kernel8, n, t):
    """Buffers that end exactly at n (n a multiple of 4, so the scale plane's head
    stride stays 16-byte aligned) give the output of the capacity-sized ones: no read
    reaches past the last live row."""
    assert n % 4 == 0
    q, codes, sc, _, _ = kfx.serving_codes(n, t)
    want = kv8.verify_codes(q, codes, sc, n, 20)
    end_codes, end_sc = mx.contiguous(codes[:, :, :n]), mx.contiguous(sc[:, :, :n])
    mx.eval(end_codes, end_sc)
    got = kv8.verify_codes(q, end_codes, end_sc, n, 20)
    assert mx.array_equal(got, want).item()


def _parity_prefixes(mark, t):
    """The tile's three placements around a mark (rows below, straddling and from it),
    less those with a row below N0, which kv8 never serves."""
    return [p for p in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}) if p + 1 >= N0]


def _assert_rows_equal_decode(splits, mark, ts):
    """Every verify row around `mark` equals decode_codes() at its own key count. Pool
    row i is the query at position base + i, which sees base + i + 1 keys."""
    base = mark - 20
    q, codes, sc, _, _ = kfx.serving_codes(mark + 20, 40, seed=mark)
    decoded = {}

    def decode_at(pos):
        if pos not in decoded:
            i = pos - base
            decoded[pos] = kv8.decode_codes(q[:, :, i : i + 1], codes, sc, pos + 1, splits)
        return decoded[pos]

    for t in ts:
        for prefix in _parity_prefixes(mark, t):
            i = prefix - base
            out = kv8.verify_codes(q[:, :, i : i + t], codes, sc, prefix + t, splits)
            assert out.shape == (1, HQ, t, D) and out.dtype == mx.bfloat16
            for r in range(t):
                same = mx.array_equal(out[:, :, r : r + 1], decode_at(prefix + r)).item()
                assert same, (splits, mark, prefix, t, r)


@pytest.mark.slow
@pytest.mark.parametrize("splits", [10, 20])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_stand_in_verify_rows_equal_decode_rows(monkeypatch, splits, step):
    """At N0 and the first two step boundaries, T = 1..16: straddles and runs cut at
    8 rows."""
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    mark = tileattn.boundaries(4096, splits)[step]
    _assert_rows_equal_decode(splits, mark, range(1, 17))


# ---- availability and the compile probe ------------------------------------------


def test_availability_refuses_the_platform_before_compiling(monkeypatch):
    def compiled():
        raise AssertionError("the kernels compiled on a refused platform")

    monkeypatch.setattr(kv8.nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(kv8, "_compile_probe", compiled)
    assert kv8.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_availability_reports_the_compilers_error_line(monkeypatch):
    line = "program_source:12:3: error: use of undeclared identifier 'x'"
    monkeypatch.setattr(kv8.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(kv8, "_compile_probe", lambda: line)
    assert kv8.availability() == nax.Availability(False, f"the int8 K/V kernels failed: {line}")


_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
    "  x = sg / 2;\n"
)


def test_compile_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_COMPILER_ERROR)

    monkeypatch.setattr(kv8, "_compile_ok", False)
    monkeypatch.setattr(kv8, "tile8", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.kv8"):
        reason = kv8._compile_probe()
    assert reason == "program_source:12:3: error: use of undeclared identifier 'x'"
    assert "x = sg / 2;" in caplog.text


def test_compile_probe_runs_the_quantiser_and_every_row_variant(monkeypatch):
    """Only success is remembered: a failure is probed again on the next call."""
    rows, quantised = [], []
    quantize = kv8.quantize

    def counted(k, v, out_rows):
        quantised.append((k.shape[2], out_rows))
        return quantize(k, v, out_rows)

    def flaky(queries, codes, sc, n, chunk):
        rows.append(queries.shape[2])
        if len(rows) == 1:
            raise RuntimeError("transient")
        return mx.zeros((1, queries.shape[2], HQ, D), dtype=mx.bfloat16)

    monkeypatch.setattr(kv8, "_compile_ok", False)
    monkeypatch.setattr(kv8, "quantize", counted)
    monkeypatch.setattr(kv8, "tile8", flaky)
    assert kv8._compile_probe() == "transient"
    assert kv8._compile_probe() is None
    assert kv8._compile_probe() is None
    assert rows == [1, *range(1, tileattn.MAX_RUN + 1)]
    assert quantised == [(N0, N0 + 256)] * 2


# ---- the real kernel (needs the tensor units; skipped where they are absent) -----


@kfx.nax
def test_every_row_variant_compiles_and_runs(monkeypatch):
    monkeypatch.setattr(kv8, "_compile_ok", False)
    assert kv8._compile_probe() is None


@kfx.nax
@pytest.mark.parametrize("n", [N0 + 7, 2000, 57_601])
@pytest.mark.parametrize("t", [1, 5, 8])
@pytest.mark.parametrize("gain", [1, 8])
def test_real_kernel_is_within_its_bound_of_the_fp32_reference(n, t, gain):
    """The fp32 reference reads the same codes (K, V and the host's Q codes), so this
    judges the kernel, not the format; x8 queries sharpen the softmax."""
    q, codes, sc, _, _ = kfx.serving_codes(n, t, seed=n + t)
    q = q * gain  # exact in bf16
    got = kv8.verify_codes(q, codes, sc, n, 20)
    want = kfx.reference_attention(q, codes, sc, n)
    assert mx.all(mx.isfinite(got)).item()
    assert _relative_rms(got, want) <= kv8.REL_RMS_BOUND


def _debug_queries(t):
    """[1, 24, T, 256] bf16 laid out as serving lays queries out (a transposed view of
    [1, T, 24, 256]): a random row at a scale that alternates with T, tie rows after
    it, and one all-zero query head in the last row."""
    rows = [_bf16((1, HQ, 1, D), t, 8.0 if t % 2 else 0.01)]
    if t > 1:
        rows.append(kfx.tie_rows(t - 1, heads=HQ, seed=t))
    q = mx.concatenate(rows, axis=2)
    zero = (mx.arange(HQ)[:, None] == 5) & (mx.arange(t)[None, :] == t - 1)
    q = mx.where(zero[None, :, :, None], mx.zeros_like(q), q)
    q = mx.contiguous(q.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    mx.eval(q)
    return q


@kfx.nax
@pytest.mark.parametrize("t", range(1, tileattn.MAX_RUN + 1))
def test_real_kernel_quantises_q_as_the_host_does_bit_for_bit(t):
    """The debug kernel writes out the Q codes it staged: equal to the host's, ties
    and the zero row included, and the debug store changes no output bit."""
    q = _debug_queries(t)
    _, codes, sc, _, _ = kfx.serving_codes(N0 + 8, t, seed=t)
    n = N0 + 8
    chunk = tileattn.chunk_for(n, 20)
    out, staged = kv8._tile8_qcodes(q, codes, sc, n, chunk)
    want, _ = kfx.reference_quantize_q(q)
    assert staged.shape == (1, HQ, t, D) and staged.dtype == mx.int8
    assert mx.array_equal(staged, want).item()
    assert mx.array_equal(out, kv8.tile8(q, codes, sc, n, chunk)).item()


@kfx.nax
@pytest.mark.parametrize("mark", [1536, 1921, 2561, 12161, 57601, 131201])
def test_real_verify_rows_equal_decode_rows_at_step_boundaries(mark):
    """N0, the next two step boundaries, the far mark (the first boundary at or past
    57,000 keys) and two more, T = 1..8 at the tile's three placements."""
    assert mark in tileattn.boundaries(131_201, 20)
    assert tileattn.far_mark(20) == 57_601
    _assert_rows_equal_decode(20, mark, range(1, tileattn.MAX_RUN + 1))


@kfx.nax
@pytest.mark.parametrize("t", [1, 5, 8])
def test_real_kernel_reads_nothing_past_n_at_the_far_mark(t):
    _assert_fenced(tileattn.far_mark(20), t)


# ---- the serving window, the mirror and its trim wrap, the ceiling and the floor ----

# The M5 Pro's split target. The stand-in serves any S; the schedule is S's.
_MIRROR_S = 20


def _mirror_counts_since(before):
    """kv8's counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in kv8.calls.items() if n != before[k]}


def _mirror_append(cache, rows, *, seed):
    """`rows` random bf16 K/V rows appended through KVCache.update_and_fetch, as an
    attention layer appends them, and the cache evaluated."""
    k_key, v_key = mx.random.split(mx.random.key(seed))
    shape = (1, tileattn.HKV, rows, tileattn.D)
    keys = mx.random.normal(shape, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(shape, key=v_key).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)


def _mirror_cache(*segments, seed=0):
    """A real KVCache prefilled by one update_and_fetch per segment of rows."""
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    for i, rows in enumerate(segments):
        _mirror_append(cache, rows, seed=1000 * seed + i)
    return cache


def _mirror_of(cache):
    return getattr(cache, kv8.MIRROR_ATTR, None)


def _mirror_bulk(cache, n):
    """Rows [0, n) quantised in one call from contiguous copies: what every mirror
    must hold, since quantisation is a pure per-row function. The copies make a
    match also show the mirror's views were read through their strides."""
    codes, sc = kv8.quantize(
        mx.contiguous(cache.keys[:, :, :n]), mx.contiguous(cache.values[:, :, :n]), n
    )
    mx.eval(codes, sc)
    return codes, sc


def _mirror_matches(cache, n):
    """The cache's mirror is valid for exactly rows [0, n), and they equal a bulk
    quantisation of the cache's rows."""
    mirror = _mirror_of(cache)
    if mirror is None or mirror.n != n:
        return False
    codes, sc = _mirror_bulk(cache, n)
    return (
        mx.array_equal(mirror.codes[:, :, :n], codes).item()
        and mx.array_equal(mirror.sc[:, :, :n], sc).item()
    )


def _mirror_queries(t, *, seed):
    """[1, 24, T, 256] bf16 queries as served: a transposed [1, T, 24, 256] pool."""
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=mx.random.key(seed))
    pool = pool.astype(mx.bfloat16)
    mx.eval(pool)
    return pool.transpose(0, 2, 1, 3)


def _mirror_recording_quantize(monkeypatch):
    """kv8.quantize, recorded as (rows, out_rows) per dispatch."""
    seen = []
    real = kv8.quantize

    def recording(k, v, out_rows):
        seen.append((k.shape[2], out_rows))
        return real(k, v, out_rows)

    monkeypatch.setattr(kv8, "quantize", recording)
    return seen


def _mirror_fits(window, b):
    c = -(-b // 256) * 256
    return 1.504 * c + 0.504 * c / 16 + 272 <= window


@pytest.mark.parametrize(("window", "limit"), [(131072, 84481), (262144, 170241)])
def test_ceiling_is_the_last_step_boundary_whose_mirror_fits_beside_the_kv(window, limit):
    assert kv8.ceiling(window, _MIRROR_S) == limit
    bounds = tileattn.boundaries(window, _MIRROR_S)
    assert _mirror_fits(window, limit)
    assert not _mirror_fits(window, bounds[bounds.index(limit) + 1])


def test_ceiling_is_zero_below_the_first_fitting_window():
    """N0 rounds up to a capacity of 1,536 rows, which needs a 2,631-token window."""
    assert kv8.ceiling(0, _MIRROR_S) == 0
    assert kv8.ceiling(2630, _MIRROR_S) == 0
    assert kv8.ceiling(2631, _MIRROR_S) == tileattn.N0


def test_floor_must_be_a_step_boundary(monkeypatch):
    for floor in (tileattn.N0, 1921, 2561):
        monkeypatch.setattr(kv8, "N8", floor)
        assert kv8._floor_reason(_MIRROR_S) is None
    monkeypatch.setattr(kv8, "N8", 2000)
    reason = kv8._floor_reason(_MIRROR_S)
    assert reason == "floor 2000 keys is not a tile step boundary at 20 splits"
    # 1,921 is a boundary at 20 splits and not at 16, where steps are 512 keys.
    monkeypatch.setattr(kv8, "N8", 1921)
    assert kv8._floor_reason(16) is not None


def test_window_serials_increase_and_a_second_window_is_refused():
    before = dict(kv8.calls)
    with kfx.window() as first, pytest.raises(AssertionError):
        kv8.open_window()
    with kfx.window() as second:
        assert kv8._window is second
    assert kv8._window is None
    assert second.serial == first.serial + 1 == kv8._serial
    assert (first.state, first.rows, first.served, first.built) == ("idle", 0, 0, False)
    assert _mirror_counts_since(before) == {"windows": 2}


def test_window_absent_or_flag_clear_serves_and_builds_nothing(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1 << 20)
    monkeypatch.setattr(kv8, "_active", 1)
    cache = _mirror_cache(kv8.N8 + 8)
    assert kv8._window is None
    assert not kv8.serving_now() and not kv8.serves(kv8.N8)
    kv8.observe_first_row(cache, 1 << 21)  # no window: nothing to cap
    with pytest.raises(AssertionError):
        kv8.prepare(cache, 1)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)
    with kfx.window(force=False) as opened:
        monkeypatch.setattr(kv8, "_active", 0)
        assert not kv8.serving_now() and not kv8.serves(kv8.N8)
        assert opened.state == "idle"


def test_serves_rows_from_the_floor_to_below_the_ceiling(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 2561)
    with kfx.window(force=False):
        assert kv8.serving_now()
        assert [kv8.serves(n) for n in (kv8.N8 - 1, kv8.N8, 2560, 2561, 1 << 20)] == [
            False,
            True,
            True,
            False,
            False,
        ]


def test_window_of_zero_serves_nothing_unless_forced(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, kv8.ceiling(0, _MIRROR_S))
    rows = (kv8.N8 - 1, kv8.N8, kv8.N8 + 1, 1 << 20)
    with kfx.window(force=False):
        assert not any(kv8.serves(n) for n in rows)
    cache = _mirror_cache(kv8.N8 + 64)
    with kfx.window([cache]) as forced:
        assert [kv8.serves(n) for n in rows] == [False, True, True, True]
        kv8.observe_first_row(cache, 1 << 20)
        assert forced.state == "idle"
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        # No ceiling bounds a forced window's rows or capacity.
        assert (mirror.n, mirror.cap) == (cache.offset, 5888)
        assert forced.state == "served"


def test_window_close_drops_every_mirror_and_wrap_and_reports_the_window(monkeypatch):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    kfx.small_floor(monkeypatch)
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    kv = [_mirror_cache(kv8.N8 + 8, seed=1), _mirror_cache(kv8.N8 + 8, seed=2)]
    arrays = [ArraysCache(size=2), ArraysCache(size=2)]
    layers = [arrays[0], kv[0], arrays[1], kv[1]]  # qwen3_5's layout: linear, full, ...
    with kfx.window() as opened:
        for cache in kv:
            _mirror_append(cache, 1, seed=3)
            kv8.prepare(cache, 1)
            assert getattr(vars(cache)["trim"], kv8._TRIM_MARK)
        assert (opened.state, opened.served, opened.built) == ("served", 2, True)
        assert kv8.close_window(layers) == ("served", 2 * (kv8.N8 + 9))
    assert kv8._window is None and cleared == [True]
    before = dict(kv8.calls)
    for cache in kv:
        assert _mirror_of(cache) is None and "trim" not in vars(cache)
        assert cache.trim.__func__ is KVCache.trim
        cache.trim(1)
    assert _mirror_counts_since(before) == {}
    assert not any(kv8.MIRROR_ATTR in vars(c) for c in arrays)


def test_window_close_with_nothing_built_leaves_mlx_cache_alone(monkeypatch):
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(True))
    with kfx.window():
        assert kv8.close_window([]) == ("idle", 0)
    assert cleared == []


def test_mirror_wrap_calls_and_restores_a_trim_of_the_caches_own(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    seen = []

    def own_trim(n):
        seen.append(n)
        cache.offset -= n
        return n

    object.__setattr__(cache, "trim", own_trim)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert mirror.original_trim is own_trim
        assert cache.trim(2) == 2
        assert seen == [2] and mirror.n == cache.offset == kv8.N8 + 7
    assert vars(cache)["trim"] is own_trim


def test_window_close_after_clear_still_drops_the_mirrors(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        kv8.clear()
        assert kv8.close_window([cache]) == ("off", 0)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)


def test_window_clear_drops_the_flag_and_the_window_only(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        serial = kv8._serial
        kv8.clear()
        assert (kv8._active, kv8._window, kv8._serial) == (0, None, serial)
        assert not kv8.serving_now()
        # Mirrors live on caches, which go with the engine.
        assert _mirror_of(cache) is not None
    kv8.close_window([cache])


@pytest.mark.parametrize(("limit", "cap"), [(84481, 5888), (2561, 2816)])
def test_mirror_first_served_call_builds_the_prefix_in_one_dispatch(monkeypatch, limit, cap):
    """At min(round_up(n + GROWTH, 256), round_up(L, 256)) rows: GROWTH rows of
    headroom, never past what the ceiling priced."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, limit)
    cache = _mirror_cache(kv8.N8 + 63)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache], force=False) as opened:
        _mirror_append(cache, 1, seed=1)
        n = cache.offset
        mirror = kv8.prepare(cache, 1)
        assert seen == [(n, cap)]
        assert (mirror.cap, mirror.n, mirror.serial) == (cap, n, opened.serial)
        assert (mirror.codes.shape, mirror.sc.shape) == ((1, 4, cap, 512), (1, 4, cap, 2))
        assert (mirror.codes.dtype, mirror.sc.dtype) == (mx.int8, mx.float16)
        assert (opened.state, opened.rows, opened.served, opened.built) == ("served", n, 1, True)
        assert _mirror_counts_since(before) == {"windows": 1, "mirror_builds": 1, "mirror_rows": n}
        assert _mirror_matches(cache, n)


def test_mirror_each_served_call_quantises_its_own_rows_once(monkeypatch):
    """A DFlash-shaped sequence: verify 5, reject 3 (a trim through the wrap),
    verify 5, decode 1, verify 8. Each update is one dispatch of exactly the
    call's rows."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 16)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache]) as opened:
        _mirror_append(cache, 5, seed=1)
        kv8.prepare(cache, 5)
        built = cache.offset
        assert seen == [(built, kv8._round_up(built + kv8.GROWTH))]
        cache.trim(3)
        assert _mirror_of(cache).n == cache.offset
        for t, seed in ((5, 2), (1, 3), (8, 4)):
            _mirror_append(cache, t, seed=seed)
            seen.clear()
            kv8.prepare(cache, t)
            assert seen == [(t, t)]
            assert _mirror_matches(cache, cache.offset)
        assert opened.rows == built + 5 + 1 + 8 and opened.served == 4
    assert _mirror_counts_since(before) == {
        "windows": 1,
        "mirror_builds": 1,
        "mirror_rows": opened.rows,
        "mirror_trims": 1,
    }


def test_mirror_trim_then_unserved_append_requantises_exactly_the_replaced_rows(monkeypatch):
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32)
    seen = _mirror_recording_quantize(monkeypatch)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        cache.trim(6)
        assert _mirror_of(cache).n == cache.offset == kv8.N8 + 27
        _mirror_append(cache, 4, seed=2)  # rows N8 + 27 .. N8 + 30 replaced, not served
        _mirror_append(cache, 1, seed=3)  # the served call's own row
        seen.clear()
        kv8.prepare(cache, 1)
        assert seen == [(5, 5)]
        assert _mirror_matches(cache, kv8.N8 + 32)


def test_mirror_trim_that_bypasses_the_wrap_leaves_stale_rows(monkeypatch):
    """The negative control for the test above: the served call's clamp alone
    re-quantises only from offset - t, so the replaced rows stay stale. The wrap
    is the mechanism, the clamp a defence."""
    from mlx_vlm.models.cache import KVCache

    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        kv8.prepare(cache, 1)
        KVCache.trim(cache, 6)
        _mirror_append(cache, 4, seed=2)
        _mirror_append(cache, 1, seed=3)
        kv8.prepare(cache, 1)
        assert _mirror_of(cache).n == cache.offset
        assert not _mirror_matches(cache, cache.offset)


def test_mirror_shrinking_block_across_the_floor_ends_equal_to_a_bulk_quantisation(
    monkeypatch,
):
    """16 rows served from N8, a rollback to below N8, 8 rows below N8 that no
    call serves, then a served call: it re-quantises the 8 replaced rows with
    its own."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8)
    seen = _mirror_recording_quantize(monkeypatch)
    with kfx.window([cache]):
        _mirror_append(cache, 16, seed=1)
        assert kv8.serves(kv8.N8 + 1)
        kv8.prepare(cache, 16)
        cache.trim(24)
        assert _mirror_of(cache).n == cache.offset == kv8.N8 - 8
        assert not kv8.serves(kv8.N8 - 7)
        _mirror_append(cache, 8, seed=2)
        _mirror_append(cache, 5, seed=3)
        seen.clear()
        kv8.prepare(cache, 5)
        assert seen == [(13, 13)]
        assert _mirror_matches(cache, kv8.N8 + 5)


def test_mirror_from_another_window_is_rebuilt_from_row_zero(monkeypatch):
    """A window closed without this cache leaves its mirror behind; a fork
    restore then replaces every row without a trim, and a prefill appends. The
    next window's first served call sees the old serial and rebuilds from row 0,
    reusing the wrap already in place."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 32, seed=1)
    with kfx.window() as first:
        _mirror_append(cache, 1, seed=2)
        kv8.prepare(cache, 1)
    assert _mirror_of(cache).serial == first.serial
    wrap = vars(cache)["trim"]
    restored = _mirror_cache(kv8.N8 + 40, seed=3)
    cache.state = (restored.keys[:, :, : restored.offset], restored.values[:, :, : restored.offset])
    del restored
    _mirror_append(cache, 8, seed=4)
    before = dict(kv8.calls)
    with kfx.window([cache]) as second:
        _mirror_append(cache, 1, seed=5)
        mirror = kv8.prepare(cache, 1)
        assert second.serial > first.serial and mirror.serial == second.serial
        assert vars(cache)["trim"] is wrap and mirror.original_trim is None
        assert _mirror_counts_since(before) == {
            "windows": 1,
            "mirror_builds": 1,
            "mirror_rows": cache.offset,
        }
        assert _mirror_matches(cache, cache.offset)
    assert _mirror_of(cache) is None and "trim" not in vars(cache)


def test_mirror_of_a_cache_whose_capacity_is_not_a_multiple_of_four(monkeypatch):
    """KVCache cuts its buffer to the offset before it grows, so a second
    prefill segment leaves 1,537 + 512 = 2,049 rows of capacity; the mirror's
    capacity is its own."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 1, 300)
    assert cache.keys.shape[2] == 2049
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert cache.keys.shape[2] == 2049 and mirror.cap % 256 == 0
        assert _mirror_matches(cache, cache.offset)
        _mirror_append(cache, 5, seed=2)
        kv8.prepare(cache, 5)
        assert _mirror_matches(cache, cache.offset)


def test_mirror_served_decode_reads_no_row_past_its_keys(monkeypatch):
    """Rows past n hold -128 codes and NaN scales; a second call at the same
    offset re-quantises only its own row and must not see them."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 40)
    queries = _mirror_queries(1, seed=1)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=2)
        clean = kv8.decode(queries, cache, _MIRROR_S)
        mx.eval(clean)
        mirror = _mirror_of(cache)
        n = cache.offset
        mirror.codes[:, :, n:] = -128
        mirror.sc[:, :, n:] = float("nan")
        mx.eval(mirror.codes, mirror.sc)
        again = kv8.decode(queries, cache, _MIRROR_S)
        mx.eval(again)
    assert mx.all(mx.isfinite(again)).item()
    assert mx.array_equal(again, clean).item()


def test_mirror_served_calls_read_it_as_a_bulk_quantisation_reads(monkeypatch):
    """decode() equals decode_codes() over a bulk quantisation, and each row of a
    verify run equals the one-row call at its own length."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 24)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        n = cache.offset
        q1 = _mirror_queries(1, seed=2)
        before = dict(kv8.calls)
        got = kv8.decode(q1, cache, _MIRROR_S)
        assert _mirror_counts_since(before) == {
            "decode_kv8": 1,
            "mirror_builds": 1,
            "mirror_rows": n,
        }
        codes, sc = _mirror_bulk(cache, n)
        assert got.shape == (1, tileattn.HQ, 1, tileattn.D)
        assert mx.array_equal(got, kv8.decode_codes(q1, codes, sc, n, _MIRROR_S)).item()

        _mirror_append(cache, 5, seed=3)
        prefix = cache.offset - 5
        ((j, k, chunk),) = tileattn.verify_plan(prefix, 5, _MIRROR_S)
        q5 = _mirror_queries(5, seed=4)
        before = dict(kv8.calls)
        mirror = kv8.prepare(cache, 5)
        part = kv8.verify_run(q5, mirror, prefix, j, k, chunk)
        assert _mirror_counts_since(before) == {"mirror_rows": 5}
        assert part.shape == (1, 5, tileattn.HQ, tileattn.D)
        codes, sc = _mirror_bulk(cache, cache.offset)
        for r in range(5):
            alone = kv8.decode_codes(q5[:, :, r : r + 1], codes, sc, prefix + r + 1, _MIRROR_S)
            assert mx.array_equal(part[:, r : r + 1].transpose(0, 2, 1, 3), alone).item(), r


def test_mirror_grows_exactly_once_when_a_decode_passes_its_capacity(monkeypatch):
    """GROWTH cut to 256 rows so 8-row calls cross the capacity in seconds:
    built at 2,048 rows, grown once to 2,560 at offset 2,053."""
    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "GROWTH", 256)
    cache = _mirror_cache(kv8.N8 + 60)
    before = dict(kv8.calls)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        assert kv8.prepare(cache, 1).cap == 2048
        seed = 2
        while cache.offset < 2100:
            _mirror_append(cache, 8, seed=seed)
            mirror = kv8.prepare(cache, 8)
            mx.eval(mirror.codes, mirror.sc)
            seed += 1
        assert mirror.cap == 2560
        assert _mirror_counts_since(before) == {
            "windows": 1,
            "mirror_builds": 1,
            "mirror_rows": cache.offset,
            "mirror_grown": 1,
        }
        assert _mirror_matches(cache, cache.offset)


def test_mirror_capacity_stops_at_the_ceiling_and_no_row_past_it_is_written(monkeypatch):
    """L = 2,561 and a capacity of round_up(L, 256) = 2,816: a straddling verify
    grows the mirror to that and quantises rows up to L, not past."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 2561)
    monkeypatch.setattr(kv8, "GROWTH", 256)
    cache = _mirror_cache(kv8.N8 + 64)
    with kfx.window([cache], force=False):
        _mirror_append(cache, 1, seed=1)
        assert kv8.prepare(cache, 1).cap == 2048
        _mirror_append(cache, 700, seed=2)
        _mirror_append(cache, 1, seed=3)
        assert kv8.prepare(cache, 1).cap == 2560
        _mirror_append(cache, 2556 - cache.offset, seed=4)
        _mirror_append(cache, 8, seed=5)
        assert kv8.serves(2557) and not kv8.serves(2561)
        mirror = kv8.prepare(cache, 8)
        assert (mirror.cap, mirror.n, cache.offset) == (2816, 2561, 2564)
        assert _mirror_matches(cache, 2561)


def test_mirror_update_writes_in_place(monkeypatch):
    """The mirror's buffers are referenced from the Mirror alone, so mlx donates
    them to the two slice updates: an update's peak is its own rows, never a
    copy of the 12 MB mirror."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 63)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        mx.eval(mirror.codes, mirror.sc)
        size = mirror.codes.nbytes + mirror.sc.nbytes
        _mirror_append(cache, 5, seed=2)
        mx.synchronize()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mirror = kv8.prepare(cache, 5)
        mx.eval(mirror.codes, mirror.sc)
        peak = mx.get_peak_memory() - base
        assert _mirror_matches(cache, cache.offset)
    assert size > 12_000_000
    assert peak < size // 8, (peak, size)


def test_mirror_growth_holds_one_layers_old_buffer_at_a_time(monkeypatch):
    """Six layers grow in one forward, each from 5,888 to 10,240 rows. Each
    layer's new buffer is evaluated before its old one is dropped, so the peak
    passes the grown total by about one layer's buffer; grown lazily, all six
    old buffers would be alive at once."""
    kfx.small_floor(monkeypatch)
    caches = [_mirror_cache(kv8.N8 + 64, seed=s) for s in range(6)]
    row = kv8.BYTES_PER_ROW_PER_HEAD  # 512 codes and two fp16 scales
    before = dict(kv8.calls)
    with kfx.window(caches):
        for cache in caches:
            _mirror_append(cache, 1, seed=10)
            kv8.prepare(cache, 1)
            _mirror_append(cache, 4280, seed=11)
            mirror = kv8.prepare(cache, 4280)
            mx.eval(mirror.codes, mirror.sc)
            assert (mirror.cap, mirror.n) == (5888, cache.offset)
        for cache in caches:
            _mirror_append(cache, 16, seed=12)
        mx.synchronize()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        for cache in caches:
            kv8.prepare(cache, 16)
        mx.eval([(_mirror_of(c).codes, _mirror_of(c).sc) for c in caches])
        peak = mx.get_peak_memory() - base
        assert all(_mirror_of(c).cap == 10240 for c in caches)
    grown = 6 * (10240 - 5888) * tileattn.HKV * row
    layer = 10240 * tileattn.HKV * row
    assert _mirror_counts_since(before)["mirror_grown"] == 6
    assert peak - grown < 2 * layer, (peak, grown, layer)


def test_cap_a_call_at_the_ceiling_caps_once_and_drops_each_layers_mirror(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    first, second = _mirror_cache(kv8.N8 + 64, seed=1), _mirror_cache(kv8.N8 + 64, seed=2)
    before = dict(kv8.calls)
    with kfx.window([first, second], force=False) as opened:
        for cache in (first, second):
            _mirror_append(cache, 1, seed=3)
            kv8.prepare(cache, 1)
        kv8.observe_first_row(first, 1920)  # below the ceiling: nothing happens
        assert opened.state == "served" and _mirror_of(first) is not None
        kv8.observe_first_row(first, 1921)
        assert opened.state == "capped"
        assert _mirror_of(first) is None and "trim" not in vars(first)
        assert _mirror_of(second) is not None
        assert not kv8.serving_now() and not kv8.serves(kv8.N8 + 100)
        kv8.observe_first_row(second, 1925)
        assert _mirror_of(second) is None and "trim" not in vars(second)
        assert kv8.close_window([first, second]) == ("capped", 2 * (kv8.N8 + 65))
    counts = _mirror_counts_since(before)
    assert counts["windows_capped"] == 1 and counts["mirror_builds"] == 2


def test_cap_a_window_whose_first_call_is_past_the_ceiling_has_no_rows(monkeypatch):
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    cache = _mirror_cache(1930)
    with kfx.window([cache], force=False):
        _mirror_append(cache, 1, seed=1)
        assert not kv8.serves(cache.offset)
        kv8.observe_first_row(cache, cache.offset)
        assert kv8.close_window([cache]) == ("capped", 0)


def test_cap_a_straddling_verify_keeps_its_mirror_and_serves_only_rows_below_it(monkeypatch):
    """L = 1,921, a step boundary at 20 splits. An 8-row verify behind 1,915 keys
    is cut there: its first run (1,916 to 1,920 keys) reads the mirror, its
    second is the tile's, and the mirror holds no row at or past L. The commit
    trims through the wrap; the decode whose row is at L caps the window."""
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, 1921)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(1915)
    seen = _mirror_recording_quantize(monkeypatch)
    before = dict(kv8.calls)
    with kfx.window([cache], force=False) as opened:
        _mirror_append(cache, 8, seed=1)
        prefix = 1915
        kv8.observe_first_row(cache, prefix + 1)
        runs = tileattn.verify_plan(prefix, 8, _MIRROR_S)
        assert runs == [(0, 5, 96), (5, 8, 128)]
        assert [kv8.serves(prefix + j + 1) for j, _, _ in runs] == [True, False]
        mirror = kv8.prepare(cache, 8)
        assert seen == [(1921, 2048)]
        assert (mirror.n, mirror.cap) == (1921, 2048)
        queries = _mirror_queries(8, seed=2)
        part = kv8.verify_run(queries, mirror, prefix, 0, 5, 96)
        codes, sc = _mirror_bulk(cache, 1921)
        for r in range(5):
            n = prefix + r + 1
            alone = kv8.decode_codes(queries[:, :, r : r + 1], codes, sc, n, _MIRROR_S)
            assert mx.array_equal(part[:, r : r + 1].transpose(0, 2, 1, 3), alone).item(), r
        assert opened.state == "served" and _mirror_of(cache) is mirror
        cache.trim(3)  # 4 accepted and the bonus row: 1,920 keys committed
        assert mirror.n == cache.offset == 1920
        _mirror_append(cache, 1, seed=3)
        assert not kv8.serves(cache.offset)
        kv8.observe_first_row(cache, cache.offset)
        assert opened.state == "capped" and _mirror_of(cache) is None
        assert kv8.close_window([cache]) == ("capped", 1921)
    assert _mirror_counts_since(before) == {
        "windows": 1,
        "windows_capped": 1,
        "mirror_builds": 1,
        "mirror_rows": 1921,
        "mirror_trims": 1,
    }


def test_mirror_never_travels_with_a_fork(monkeypatch):
    """`fork_copy` builds a fresh cache from `state` and `meta_state`, so the
    mirror and its trim wrap stay with the cache that was served: a fork
    published to a slot or the store holds bf16 K/V and the class's `trim`,
    and trimming the fork leaves the served cache's mirror alone."""
    from mlx_vlm.models.cache import KVCache

    from sous.engine.promptcache import fork_copy

    kfx.small_floor(monkeypatch)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    cache = _mirror_cache(kv8.N8 + 32)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mx.eval(kv8.decode(_mirror_queries(1, seed=2), cache, _MIRROR_S))
        assert _mirror_matches(cache, kv8.N8 + 33)
        fork = KVCache()
        fork_copy([cache], [fork], mx.array)
        mx.eval(fork.keys, fork.values)
        assert fork.offset == cache.offset
        assert _mirror_of(fork) is None
        assert "trim" not in vars(fork), "the fork's trim is the class's"
        assert fork.trim(8) == 8
        assert _mirror_matches(cache, kv8.N8 + 33), "the served cache's mirror is untouched"
    assert _mirror_of(cache) is None


def test_mirror_trim_past_the_offset_lowers_n_to_the_new_offset(monkeypatch):
    """`KVCache.trim` clamps the count to the offset and returns what it
    trimmed, and the wrap reads the offset afterwards: a rollback asked for
    more rows than the cache holds (an abort after nothing was appended) and
    a trim of zero both leave n equal to the offset, never below it."""
    kfx.small_floor(monkeypatch)
    cache = _mirror_cache(kv8.N8 + 8)
    before = dict(kv8.calls)
    with kfx.window([cache]):
        _mirror_append(cache, 1, seed=1)
        mirror = kv8.prepare(cache, 1)
        assert mirror.n == kv8.N8 + 9
        assert cache.trim(0) == 0
        assert mirror.n == cache.offset == kv8.N8 + 9
        assert cache.trim(kv8.N8 + 100) == kv8.N8 + 9
        assert mirror.n == cache.offset == 0
    assert _mirror_counts_since(before)["mirror_trims"] == 2


# ---- dispatch: the two seams route each row to stock, the tile or kv8 ---------------

# The second and third tile step boundaries at S = 20: below the floor the tile
# serves (from N0), from the floor to the ceiling kv8, from the ceiling the tile.
DISPATCH_S = 20
DISPATCH_FLOOR = 1921
DISPATCH_CEIL = 2561
# The counters the seams move in kv8.calls; the mirror's own counters belong to
# the mirror's tests.
_DISPATCH_KV8_KEYS = (
    "decode_kv8",
    "verify_kv8",
    "verify_rows_kv8",
    "verify_launches_kv8",
    "windows_capped",
)


@pytest.fixture
def dispatch_lines(monkeypatch):
    """The tile's flag at S = 20 with the stand-in tile, kv8's stand-in kernel, the
    floor at 1921 and the ceiling at 2561, all for this test only."""
    assert tileattn.boundaries(DISPATCH_CEIL, DISPATCH_S) == [
        tileattn.N0,
        DISPATCH_FLOOR,
        DISPATCH_CEIL,
    ]
    tfx.hooked(monkeypatch, splits=DISPATCH_S)
    monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    monkeypatch.setattr(kv8, "N8", DISPATCH_FLOOR)
    kfx.ceiling(monkeypatch, DISPATCH_CEIL)


@contextlib.contextmanager
def _dispatch_open(monkeypatch, caches):
    """An unforced serving window over `caches`, as VLMEngine.decode() opens one
    around a generation, closed on the way out whatever the test asserted."""
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)
    window = kv8.open_window()
    try:
        yield window
    finally:
        kv8.close_window(caches)


def _dispatch_rows(t, *, seed):
    k_key, v_key = mx.random.split(mx.random.key(seed))
    shape = (1, tileattn.HKV, t, tileattn.D)
    keys = mx.random.normal(shape, key=k_key).astype(mx.bfloat16)
    values = mx.random.normal(shape, key=v_key).astype(mx.bfloat16)
    return keys, values


def _dispatch_cache(n, *, seed=0):
    """A KVCache holding n random bf16 rows, appended in one update as a prefill
    segment appends them."""
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    cache.update_and_fetch(*_dispatch_rows(n, seed=seed))
    mx.eval(cache.keys, cache.values)
    return cache


def _dispatch_append(cache, t, *, seed):
    """t more rows through update_and_fetch, as a forward's projections append
    them; returns the K/V views the attention call receives."""
    keys, values = cache.update_and_fetch(*_dispatch_rows(t, seed=seed))
    mx.eval(keys, values)
    return keys, values


def _dispatch_queries(t, *, seed):
    """[1, 24, T, 256] bf16 query rows, a transposed [1, T, 24, 256] pool as
    q_norm and rope hand them over."""
    pool = mx.random.normal((1, t, tileattn.HQ, tileattn.D), key=mx.random.key(seed))
    pool = pool.astype(mx.bfloat16)
    mx.eval(pool)
    return pool.transpose(0, 2, 1, 3)


def _dispatch_decode(cache, *, seed):
    """One decode step: a row appended, then the decode hook's decision on it. An
    output of None means the hook handed the call to mlx-vlm's own function."""
    keys, values = _dispatch_append(cache, 1, seed=seed)
    q = _dispatch_queries(1, seed=seed + 1)
    call = {"cache": cache, "scale": tileattn.SCALE, "mask": None}
    return q, keys, values, tileattn._serve_decode((q, keys, values), call)


def _dispatch_snapshot():
    return dict(tileattn.calls), dict(kv8.calls)


def _dispatch_moved(before):
    """(tileattn counters, kv8's seam counters) that moved since `before`, by how much."""
    tile_before, kv8_before = before
    tile = {k: n - tile_before[k] for k, n in tileattn.calls.items() if n != tile_before[k]}
    ours = {
        k: kv8.calls[k] - kv8_before[k] for k in _DISPATCH_KV8_KEYS if kv8.calls[k] != kv8_before[k]
    }
    return tile, ours


def _dispatch_mirror(cache):
    return getattr(cache, kv8.MIRROR_ATTR, None)


def _dispatch_kv8_row(q, keys, values, m):
    """The one-row kv8 call at key count m over the real quantiser's codes of keys
    [0, m): what a row kv8 serves must equal bit for bit."""
    codes, sc = kv8.quantize(keys[:, :, :m], values[:, :, :m], m)
    chunk = tileattn.chunk_for(m, DISPATCH_S)
    return kfx.stand_in_tile8(q, codes, sc, m, chunk).transpose(0, 2, 1, 3)


def _dispatch_tile_row(q, keys, values, m):
    """The one-row call at key count m without kv8: the tile from N0, stock below."""
    return tileattn.decode(q, keys[:, :, :m], values[:, :, :m], DISPATCH_S)


def _dispatch_assert_rows(out, q, keys, values, prefix, kv8_rows):
    """Every verify row r equals its one-row call at prefix + r + 1 keys: kv8's for
    the key counts in `kv8_rows`, the tile's or stock's for the rest."""
    assert out.shape == (1, tileattn.HQ, q.shape[2], tileattn.D) and out.dtype == mx.bfloat16
    for r in range(q.shape[2]):
        m = prefix + r + 1
        qr = q[:, :, r : r + 1]
        if m in kv8_rows:
            want = _dispatch_kv8_row(qr, keys, values, m)
        else:
            want = _dispatch_tile_row(qr, keys, values, m)
        assert mx.array_equal(out[:, :, r : r + 1], want).item(), (prefix, r, m)


@pytest.mark.parametrize(
    ("n", "path"),
    [
        pytest.param(tileattn.N0 - 1, "stock", id="below N0"),
        pytest.param(tileattn.N0, "tile", id="at N0"),
        pytest.param(DISPATCH_FLOOR - 1, "tile", id="below the floor"),
        pytest.param(DISPATCH_FLOOR, "kv8", id="at the floor"),
        pytest.param(DISPATCH_CEIL - 1, "kv8", id="below the ceiling"),
    ],
)
def test_dispatch_a_decode_takes_its_path_by_key_count(dispatch_lines, monkeypatch, n, path):
    """Stock below N0, the tile from N0 to the floor, kv8 from the floor up to the
    ceiling; only a kv8 call builds a mirror, and decode_tile counts only the
    tile's own calls."""
    cache = _dispatch_cache(n - 1)
    with _dispatch_open(monkeypatch, [cache]) as window:
        before = _dispatch_snapshot()
        q, keys, values, out = _dispatch_decode(cache, seed=n)
        assert keys.shape[2] == cache.offset == n
        moved = _dispatch_moved(before)
        if path == "stock":
            assert out is None
            assert moved == ({"decode_stock": 1}, {})
        elif path == "tile":
            assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, n)).item()
            assert moved == ({"decode_tile": 1}, {})
        else:
            assert mx.array_equal(out, _dispatch_kv8_row(q, keys, values, n)).item()
            assert moved == ({}, {"decode_kv8": 1})
            assert _dispatch_mirror(cache).n == n
        assert (_dispatch_mirror(cache) is not None) == (path == "kv8")
        assert window.state != "capped"


def test_dispatch_a_decode_from_the_ceiling_caps_the_window_and_the_tile_serves_it(
    dispatch_lines, monkeypatch
):
    """The first call at the ceiling drops the mirror and caps the window, counted
    once; that call and every later one run on the tile."""
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    with _dispatch_open(monkeypatch, [cache]) as window:
        _dispatch_decode(cache, seed=31)
        assert _dispatch_mirror(cache) is not None
        _dispatch_append(cache, DISPATCH_CEIL - 1 - cache.offset, seed=32)
        for n in (DISPATCH_CEIL, DISPATCH_CEIL + 1):
            before = _dispatch_snapshot()
            q, keys, values, out = _dispatch_decode(cache, seed=n)
            assert keys.shape[2] == n
            assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, n)).item()
            capped = {"windows_capped": 1} if n == DISPATCH_CEIL else {}
            assert _dispatch_moved(before) == ({"decode_tile": 1}, capped)
            assert window.state == "capped" and _dispatch_mirror(cache) is None


@pytest.mark.parametrize(
    ("prefix", "t", "want_tile", "want_kv8", "kv8_rows"),
    [
        pytest.param(
            DISPATCH_FLOOR - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_tile": 3,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {"verify_kv8": 1, "verify_rows_kv8": 5, "verify_launches_kv8": 1},
            range(DISPATCH_FLOOR, DISPATCH_FLOOR + 5),
            id="straddles the floor",
        ),
        pytest.param(
            DISPATCH_FLOOR + 8,
            12,
            {"verify_calls": 1, "verify_t9_up": 1, "verify_straddles": 1},
            {"verify_kv8": 1, "verify_rows_kv8": 12, "verify_launches_kv8": 2},
            range(DISPATCH_FLOOR + 9, DISPATCH_FLOOR + 21),
            id="two kv8 runs",
        ),
        pytest.param(
            DISPATCH_CEIL - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_tile": 5,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {"verify_kv8": 1, "verify_rows_kv8": 3, "verify_launches_kv8": 1},
            range(DISPATCH_CEIL - 3, DISPATCH_CEIL),
            id="straddles the ceiling",
        ),
        pytest.param(
            tileattn.N0 - 4,
            8,
            {
                "verify_calls": 1,
                "verify_t8": 1,
                "verify_rows_stock": 3,
                "verify_rows_tile": 5,
                "verify_launches": 1,
                "verify_straddles": 1,
            },
            {},
            range(0),
            id="stock and tile only",
        ),
        pytest.param(
            1700,
            5,
            {"verify_calls": 1, "verify_t5": 1, "verify_rows_tile": 5, "verify_launches": 1},
            {},
            range(0),
            id="tile only",
        ),
    ],
)
def test_dispatch_a_verify_serves_each_run_by_its_first_row(
    dispatch_lines, monkeypatch, prefix, t, want_tile, want_kv8, kv8_rows
):
    """Each run of verify_plan goes stock, to kv8 or to the tile by its first row's
    key count; a call with no run in [floor, ceiling) builds no mirror, and a call
    that straddles the ceiling serves its rows below it and keeps its mirror."""
    cache = _dispatch_cache(prefix)
    with _dispatch_open(monkeypatch, [cache]) as window:
        keys, values = _dispatch_append(cache, t, seed=prefix)
        q = _dispatch_queries(t, seed=prefix + 1)
        before = _dispatch_snapshot()
        out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert _dispatch_moved(before) == (want_tile, want_kv8)
        _dispatch_assert_rows(out, q, keys, values, prefix, kv8_rows)
        assert (_dispatch_mirror(cache) is not None) == bool(kv8_rows)
        assert window.state != "capped"


def test_dispatch_a_verify_whose_first_row_is_at_the_ceiling_caps_and_the_tile_serves_it(
    dispatch_lines, monkeypatch
):
    """A mirror built earlier in the window is dropped by the first call that
    starts at the ceiling, and that call runs on the tile."""
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    with _dispatch_open(monkeypatch, [cache]) as window:
        _dispatch_decode(cache, seed=41)
        assert _dispatch_mirror(cache) is not None
        _dispatch_append(cache, DISPATCH_CEIL - 1 - cache.offset, seed=42)
        keys, values = _dispatch_append(cache, 5, seed=43)
        q = _dispatch_queries(5, seed=44)
        before = _dispatch_snapshot()
        out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert _dispatch_moved(before) == (
            {"verify_calls": 1, "verify_t5": 1, "verify_rows_tile": 5, "verify_launches": 1},
            {"windows_capped": 1},
        )
        assert window.state == "capped" and _dispatch_mirror(cache) is None
        _dispatch_assert_rows(out, q, keys, values, DISPATCH_CEIL - 1, range(0))


@pytest.mark.parametrize(
    ("prefix", "t", "rows"),
    [
        pytest.param(DISPATCH_FLOOR + 8, 12, DISPATCH_FLOOR + 20, id="two kv8 runs"),
        pytest.param(DISPATCH_CEIL - 4, 8, DISPATCH_CEIL, id="straddles the ceiling"),
    ],
)
def test_dispatch_a_verify_quantises_its_rows_once(dispatch_lines, monkeypatch, prefix, t, rows):
    """One prepare per call however many runs read the mirror; the window's first
    build covers the committed prefix and the call's rows up to the ceiling, never
    past it."""
    prepared, quantised = [], []
    prepare, quantize = kv8.prepare, kv8.quantize

    def counted_prepare(cache, count):
        prepared.append(count)
        return prepare(cache, count)

    def counted_quantize(k, v, out_rows):
        quantised.append(k.shape[2])
        return quantize(k, v, out_rows)

    monkeypatch.setattr(kv8, "prepare", counted_prepare)
    monkeypatch.setattr(kv8, "quantize", counted_quantize)
    cache = _dispatch_cache(prefix)
    with _dispatch_open(monkeypatch, [cache]):
        keys, values = _dispatch_append(cache, t, seed=prefix)
        q = _dispatch_queries(t, seed=prefix + 1)
        tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
        assert prepared == [t]
        assert sum(quantised) == rows
        assert _dispatch_mirror(cache) is not None


def test_dispatch_serves_nothing_without_a_window(dispatch_lines, monkeypatch):
    """kv8 active but no decode running (a prefill, a load-time probe): both seams
    behave exactly as the tile does alone and no mirror is built."""
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)
    cache = _dispatch_cache(DISPATCH_FLOOR + 8)
    before = _dispatch_snapshot()
    q, keys, values, out = _dispatch_decode(cache, seed=51)
    assert mx.array_equal(out, _dispatch_tile_row(q, keys, values, DISPATCH_FLOOR + 9)).item()
    keys, values = _dispatch_append(cache, 5, seed=52)
    q = _dispatch_queries(5, seed=53)
    out = tileattn.serve_verify(q, keys, values, DISPATCH_S, cache)
    _dispatch_assert_rows(out, q, keys, values, DISPATCH_FLOOR + 9, range(0))
    assert _dispatch_moved(before) == (
        {
            "decode_tile": 1,
            "verify_calls": 1,
            "verify_t5": 1,
            "verify_rows_tile": 5,
            "verify_launches": 1,
        },
        {},
    )
    assert _dispatch_mirror(cache) is None


def test_dispatch_with_the_floor_at_n0_kv8_takes_the_first_tile_step(dispatch_lines, monkeypatch):
    """With the floor at N0, kv8 serves from N0 and the tile serves no decode below
    the ceiling."""
    kfx.small_floor(monkeypatch)
    assert kv8.N8 == tileattn.N0
    cache = _dispatch_cache(tileattn.N0 - 1)
    with _dispatch_open(monkeypatch, [cache]):
        before = _dispatch_snapshot()
        q, keys, values, out = _dispatch_decode(cache, seed=61)
        assert mx.array_equal(out, _dispatch_kv8_row(q, keys, values, tileattn.N0)).item()
        assert _dispatch_moved(before) == ({}, {"decode_kv8": 1})


def test_dispatch_the_verifier_hands_its_layer_cache_to_serve_verify(dispatch_lines, monkeypatch):
    """Through mlx-vlm's exact verifier on the tiny model: each full-attention
    layer's call reaches serve_verify with that layer's own KVCache, and kv8
    serves its rows from a mirror on that cache."""
    lm = tfx.tiny_language_model()
    prefix = DISPATCH_FLOOR + 8
    ids = mx.random.randint(0, tfx.VOCAB, (prefix + 3,), key=mx.random.key(7)).tolist()
    cache = tfx.prefill_cache(lm, ids[:prefix])
    full = [c for c, layer in zip(cache, lm.layers, strict=True) if not layer.is_linear]
    assert len(full) == 2
    tfx.verify_ready(monkeypatch, lm)
    seen = []
    serve_verify = tileattn.serve_verify

    def recording(queries, keys, values, splits, layer_cache):
        seen.append(layer_cache)
        return serve_verify(queries, keys, values, splits, layer_cache)

    monkeypatch.setattr(tileattn, "serve_verify", recording)
    with _dispatch_open(monkeypatch, full):
        before = _dispatch_snapshot()
        tfx.verify_forward(lm, cache, ids[prefix : prefix + 3])
        assert [id(c) for c in seen] == [id(c) for c in full]
        assert _dispatch_moved(before) == (
            {"verify_calls": 2, "verify_t3": 2},
            {"verify_kv8": 2, "verify_rows_kv8": 6, "verify_launches_kv8": 2},
        )
        assert all(_dispatch_mirror(c) is not None for c in full)
    assert all(_dispatch_mirror(c) is None for c in full)
