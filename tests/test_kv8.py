"""kv8 (sous.engine.kv8): int8 K/V and Q through the attention tile.

The quantiser needs no tensor units, so its tests run on any Metal GPU, CI's
macos-15 included. The int8 split kernel does: its entries run everywhere with a
plain-mlx stand-in, and the `nax`-marked tests run the real kernel where kv8's own
availability() passes.
"""

from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, kv8, nax, tileattn  # noqa: E402 — after the guard
from tests import kv8_fixtures as kfx  # noqa: E402

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
