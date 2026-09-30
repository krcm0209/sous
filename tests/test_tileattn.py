"""The attention tile (sous.engine.tileattn).

The schedule is pure Python. The kernel entries run on any Metal GPU with a
plain-mlx stand-in for the Metal kernel, which CI's GPU (macos-15) cannot
compile; the `nax`-marked tests run the real kernel where the tensor units are.
"""

import importlib
import inspect
import time
from importlib.resources import files

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import int8prefill, nax, tileattn, verifyattn  # noqa: E402 — after the guard
from tests import tile_fixtures as tfx  # noqa: E402

N0 = tileattn.N0
SPLIT_TARGETS = [10, 16, 20, 40]


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_below_n0_is_the_stock_path(splits):
    assert tileattn.chunk_for(N0 - 1, splits) == 0
    assert tileattn.chunk_for(N0, splits) > 0
    assert tileattn.step_of(N0 - 1, splits) == (1, N0 - 1, 0)


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_every_step_past_the_first_is_32_s_keys_wide(splits):
    marks = tileattn.boundaries(262_144, splits)
    for mark in marks[1:]:
        first, last, chunk = tileattn.step_of(mark, splits)
        assert (first, last - first + 1) == (mark, 32 * splits), mark
        assert chunk == tileattn.chunk_for(mark, splits) == tileattn.chunk_for(last, splits)
        assert tileattn.chunk_for(mark - 1, splits) == chunk - tileattn.GRAN


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_no_row_gets_more_than_s_splits(splits):
    for n in range(N0, 262_145):
        assert -(-n // tileattn.chunk_for(n, splits)) <= splits, n


def test_twenty_splits_from_the_tail_of_step_nineteen():
    assert -(-11_552 // tileattn.chunk_for(11_552, 20)) == 19
    assert all(-(-n // tileattn.chunk_for(n, 20)) == 20 for n in range(11_553, 262_145))


@pytest.mark.parametrize(
    ("splits", "first_four"),
    [
        (20, [1536, 1921, 2561, 3201]),
        (10, [1536, 1601, 1921, 2241]),
        (16, [1536, 1537, 2049, 2561]),  # N0 is a step of one key at 16 splits
        (40, [1536, 2561, 3841, 5121]),
    ],
)
def test_boundaries_start_at_n0(splits, first_four):
    assert tileattn.boundaries(262_144, splits)[:4] == first_four


def test_the_m5_pros_schedule_has_408_steps_to_262k():
    assert len(tileattn.boundaries(262_144, 20)) == 408


def test_step_of_either_side_of_the_57600_boundary():
    assert tileattn.step_of(57_600, 20) == (56_961, 57_600, 2_880)
    assert tileattn.step_of(57_601, 20) == (57_601, 58_240, 2_912)


@pytest.mark.parametrize(
    ("prefix", "t", "runs"),
    [
        # rows 1531..1535 are stock; 1536..1543 fill a run; 1544..1546 continue the step
        (1530, 16, [(0, 5, 0), (5, 13, 96), (13, 16, 96)]),
        # a run cut at 8 rows, then the step boundary at 57601 inside the call
        (57_590, 16, [(0, 8, 2880), (8, 10, 2880), (10, 16, 2912)]),
        (5000, 12, [(0, 8, 256), (8, 12, 256)]),
    ],
)
def test_verify_plan_cuts_at_n0_steps_and_eight_rows(prefix, t, runs):
    assert tileattn.verify_plan(prefix, t, 20) == runs


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_verify_plan_runs_cover_every_row_at_its_own_chunk(splits):
    for prefix in range(N0 - 20, N0 + 3 * 32 * splits + 1):
        for t in range(1, 17):
            runs = tileattn.verify_plan(prefix, t, splits)
            assert [j for j, _, _ in runs] == [0, *[k for _, k, _ in runs[:-1]]]
            assert runs[-1][1] == t
            for j, k, chunk in runs:
                assert 1 <= k - j <= tileattn.MAX_RUN
                assert all(tileattn.chunk_for(prefix + r + 1, splits) == chunk for r in range(j, k))
            for (j, k, chunk), (_, _, following) in zip(runs, runs[1:], strict=False):
                assert chunk != following or k - j == tileattn.MAX_RUN, (prefix, t, runs)


# ---- kernel source ---------------------------------------------------------------


def test_the_kernel_file_holds_a_split_body_and_a_reduce_body():
    text = files("sous.engine.kernels").joinpath("attention_tile.metal").read_text()
    assert text.count("\n// REDUCE\n") == 1 and "#include" not in text
    split, reduce = text.split(tileattn._REDUCE_MARK)
    assert "mpp::tensor_ops" in split and "nax_coord(lane)" in split
    # MSL 4.1 gives locals a generic address space: every local cooperative-tensor
    # type is stripped of it, or macOS 27's compiler rejects the operands.
    assert split.count("metal::remove_addrspace_t<decltype(") == 4
    assert "mpp::" not in reduce


def test_both_kernels_are_built_with_safe_math_and_the_split_reads_strides(monkeypatch):
    made = {}

    def metal_kernel(**kwargs):
        made[kwargs["name"]] = kwargs
        return lambda **launch: []

    monkeypatch.setattr(mx.fast, "metal_kernel", metal_kernel)
    # A fresh cache: the recording kernels must not outlive this test.
    monkeypatch.setattr(int8prefill, "_kernels", {})
    tileattn._kernel_pair()
    split = made["sous_attention_tile_split"]
    reduce = made["sous_attention_tile_reduce"]
    assert (split["input_names"], split["output_names"]) == (
        ["q", "k", "v", "params"],
        ["part", "stats"],
    )
    assert (reduce["input_names"], reduce["output_names"]) == (
        ["part", "stats", "params"],
        ["out"],
    )
    assert split["ensure_row_contiguous"] is False and "k_strides" in split["source"]
    assert reduce["ensure_row_contiguous"] is True
    assert split["compile_options"] == reduce["compile_options"] == {"math_mode": "safe"}
    assert "MetalPerformancePrimitives" in split["header"]
    assert "MetalPerformancePrimitives" not in reduce["header"]
    int8prefill._kernel("sous_test_plain", ["x"], ["y"], "y[0] = x[0];", "")
    plain = made["sous_test_plain"]
    assert plain["ensure_row_contiguous"] is True and plain["compile_options"] is None


# ---- entries, with the stand-in (and the real kernel where it compiles) ----------


@pytest.fixture(params=[pytest.param("stand-in"), pytest.param("real", marks=tfx.nax)])
def tile_kernel(request, monkeypatch):
    """The plain-mlx stand-in everywhere; the Metal kernel where the tensor units are."""
    if request.param == "stand-in":
        monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    return request.param


def test_in_scope_takes_the_serving_layout():
    assert tileattn.in_scope(*tfx.serving_qkv(2000, 3), tileattn.SCALE)


_OUT_OF_SCOPE = {
    "batch 2": lambda q, k, v, s: (
        mx.concatenate([q, q]),
        mx.concatenate([k, k]),
        mx.concatenate([v, v]),
        s,
    ),
    "16 query heads": lambda q, k, v, s: (q[:, :16], k, v, s),
    "2 KV heads": lambda q, k, v, s: (q, k[:, :2], v[:, :2], s),
    "head dim 128": lambda q, k, v, s: (q[..., :128], k[..., :128], v[..., :128], s),
    "float16": lambda q, k, v, s: (
        q.astype(mx.float16),
        k.astype(mx.float16),
        v.astype(mx.float16),
        s,
    ),
    "float16 queries": lambda q, k, v, s: (q.astype(mx.float16), k, v, s),
    "scale 0.1": lambda q, k, v, s: (q, k, v, 0.1),
    "fewer keys than rows": lambda q, k, v, s: (q, k[:, :, :2], v[:, :, :2], s),
    "unequal K and V": lambda q, k, v, s: (q, k, v[:, :, :-1], s),
    "quantized keys": lambda q, k, v, s: (q, (k, k), v, s),
    "rank 3": lambda q, k, v, s: (q[0], k, v, s),
}


@pytest.mark.parametrize("broken", list(_OUT_OF_SCOPE))
def test_in_scope_refuses_each_broken_property(broken):
    q, k, v = tfx.serving_qkv(2000, 3)
    assert not tileattn.in_scope(*_OUT_OF_SCOPE[broken](q, k, v, tileattn.SCALE))


@pytest.mark.parametrize(
    ("n", "t", "chunk"), [(2000, 3, 96), (2000, 8, 96), (4096, 8, 224), (1537, 5, 64)]
)
def test_a_stand_in_row_is_the_one_row_call_at_its_own_length(n, t, chunk):
    q, k, v = tfx.serving_qkv(n, t)
    out = tfx.stand_in_tile(q, k, v, n, chunk)
    assert out.shape == (1, t, tileattn.HQ, tileattn.D) and out.dtype == mx.bfloat16
    for r in range(t):
        m = n - t + r + 1
        one = tfx.stand_in_tile(q[:, :, r : r + 1], k[:, :, :m], v[:, :, :m], m, chunk)
        assert mx.array_equal(out[:, r], one[:, 0]).item(), r


@pytest.mark.parametrize(("n", "chunk"), [(2000, 96), (4096, 224), (1537, 64)])
def test_a_stand_in_row_moves_with_its_chunk(n, chunk):
    """So an equality between two paths proves they used the same chunk."""
    q, k, v = tfx.serving_qkv(n, 1)
    same_step = tfx.stand_in_tile(q, k, v, n, chunk)
    next_step = tfx.stand_in_tile(q, k, v, n, chunk + tileattn.GRAN)
    assert not mx.array_equal(same_step, next_step).item()


def _assert_rows_equal_decode(splits: int, mark: int, ts) -> None:
    """Every verify row around `mark` equals decode() at its own key count. Pool
    row i is the query at position base + i, which sees base + i + 1 keys; the
    prefixes put a call's rows all below, straddling and all from `mark`."""
    base = mark - 20
    q, k, v = tfx.serving_qkv(mark + 20, 40, seed=mark)
    decoded = {}

    def decode_at(pos):
        if pos not in decoded:
            i = pos - base
            decoded[pos] = tileattn.decode(
                q[:, :, i : i + 1], k[:, :, : pos + 1], v[:, :, : pos + 1], splits
            )
        return decoded[pos]

    for t in ts:
        for prefix in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1}):
            n = prefix + t
            i = prefix - base
            out = tileattn.verify(q[:, :, i : i + t], k[:, :, :n], v[:, :, :n], splits)
            assert out.shape == (1, tileattn.HQ, t, tileattn.D) and out.dtype == mx.bfloat16
            for r in range(t):
                same = mx.array_equal(out[:, :, r : r + 1], decode_at(prefix + r)).item()
                assert same, (splits, mark, prefix, t, r)


@pytest.mark.slow
@pytest.mark.parametrize("splits", [10, 20])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_stand_in_verify_rows_equal_decode_rows(monkeypatch, splits, step):
    """At N0 and the first two step boundaries, T = 1..16: straddles, runs cut at
    8 rows, and the rows below N0 against stock one-row attention."""
    monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    mark = tileattn.boundaries(4096, splits)[step]
    _assert_rows_equal_decode(splits, mark, range(1, 17))


@tfx.nax
@pytest.mark.parametrize("mark", [1536, 1921, 2561, 12161, 57601, 131201, 261761])
def test_real_verify_rows_equal_decode_rows_at_step_boundaries(mark):
    assert mark in tileattn.boundaries(262_144, 20)
    _assert_rows_equal_decode(20, mark, range(1, 17))


def test_the_entries_return_the_models_layouts(tile_kernel):
    q, k, v = tfx.serving_qkv(2000, 1)
    assert tileattn.tile(q, k, v, 2000, 96).shape == (1, 1, tileattn.HQ, tileattn.D)
    assert tileattn.decode(q, k, v, 20).shape == (1, tileattn.HQ, 1, tileattn.D)
    for n, t in ((2000, 5), (tileattn.N0 + 2, 5), (2000, 12)):
        q, k, v = tfx.serving_qkv(n, t)
        out = tileattn.verify(q, k, v, 20)
        assert out.shape == (1, tileattn.HQ, t, tileattn.D) and out.dtype == mx.bfloat16


@pytest.mark.parametrize(("n", "t"), [(tileattn.N0, 1), (2000, 3), (1600, 8), (1540, 12)])
def test_reads_stay_below_n(tile_kernel, n, t):
    """NaN planted past n in every KV head, and a buffer that ends exactly at n,
    both give the clean buffer's output."""
    q, k, v = tfx.serving_qkv(n, t)
    want = tileattn.verify(q, k, v, 20)
    nan = mx.full((1, tileattn.HKV, 256, tileattn.D), float("nan"), dtype=mx.bfloat16)
    k_nan = mx.concatenate([k, nan], axis=2)
    v_nan = mx.concatenate([v, nan], axis=2)
    k_end, v_end = mx.contiguous(k), mx.contiguous(v)
    mx.eval(k_nan, v_nan, k_end, v_end)
    assert mx.all(mx.isnan(k_nan[:, :, n:])).item() and not mx.any(mx.isnan(k_nan[:, :, :n])).item()
    for keys, values in ((k_nan[:, :, :n], v_nan[:, :, :n]), (k_end, v_end)):
        got = tileattn.verify(q, keys, values, 20)
        assert mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()


def test_a_wrong_partition_would_differ(monkeypatch):
    """The sensitivity half: a straddling run served as one call at a single chunk,
    and decode at the neighbouring step's chunk, both change the bits, so the
    equalities above are not vacuous."""
    monkeypatch.setattr(tileattn, "tile", tfx.stand_in_tile)
    splits = 10
    mark = tileattn.boundaries(4096, splits)[1]
    prefix, t = mark - 3, 4
    runs = tileattn.verify_plan(prefix, t, splits)
    assert [(j, k) for j, k, _ in runs] == [(0, 2), (2, 4)]
    q, k, v = tfx.serving_qkv(prefix + t, t)
    right = tileattn.verify(q, k, v, splits)
    one_call = tfx.stand_in_tile(q, k, v, prefix + t, runs[-1][2]).transpose(0, 2, 1, 3)
    assert not mx.array_equal(one_call[:, :, :2], right[:, :, :2]).item()
    first, _, chunk = tileattn.step_of(mark, splits)
    assert first == mark
    q1, k1, v1 = q[:, :, 1:2], k[:, :, :mark], v[:, :, :mark]
    previous_step = tfx.stand_in_tile(q1, k1, v1, mark, chunk - tileattn.GRAN)
    decoded = tileattn.decode(q1, k1, v1, splits)
    assert not mx.array_equal(decoded, previous_step.transpose(0, 2, 1, 3)).item()


def test_decode_below_n0_is_stock_one_row_attention(monkeypatch):
    def launched(*args):
        raise AssertionError("the tile ran below N0")

    monkeypatch.setattr(tileattn, "tile", launched)
    q, k, v = tfx.serving_qkv(tileattn.N0 - 1, 1)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=tileattn.SCALE, mask=None)
    assert mx.array_equal(tileattn.decode(q, k, v, 20), stock).item()


def test_verify_below_n0_plans_with_its_own_architecture(monkeypatch):
    """verifyattn._ARCH is set only by verifyattn's own probe; the tile reads mlx."""
    seen = []

    def grouped(queries, keys, values, scale, prefix, arch):
        seen.append((prefix, arch, scale))
        return mx.zeros(queries.shape, dtype=queries.dtype)

    monkeypatch.setattr(verifyattn, "grouped_attention", grouped)
    monkeypatch.setattr(verifyattn, "_ARCH", "applegpu_g13s")
    tileattn.verify(*tfx.serving_qkv(1000, 3), 20)
    assert seen == [(997, mx.device_info()["architecture"], tileattn.SCALE)]


def test_arch_is_read_from_mlx_once(monkeypatch):
    tileattn._arch.cache_clear()
    try:
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g17s"})
        assert tileattn._arch() == "applegpu_g17s"
        monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g14g"})
        assert tileattn._arch() == "applegpu_g17s"
    finally:
        tileattn._arch.cache_clear()


# ---- availability and the compile probe ------------------------------------------


def test_availability_refuses_the_platform_before_compiling(monkeypatch):
    def compiled():
        raise AssertionError("the kernel compiled on a refused platform")

    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: "macOS 15.5 < 26.2")
    monkeypatch.setattr(tileattn, "_compile_probe", compiled)
    assert tileattn.availability() == nax.Availability(False, "macOS 15.5 < 26.2")


def test_availability_reports_the_compilers_error_line(monkeypatch):
    line = "program_source:12:3: error: use of undeclared identifier 'x'"
    monkeypatch.setattr(tileattn.nax, "platform_reason", lambda: None)
    monkeypatch.setattr(tileattn, "_compile_probe", lambda: line)
    assert tileattn.availability() == nax.Availability(
        False, f"the attention tile kernel failed: {line}"
    )


_TILE_COMPILER_ERROR = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
    "  x = sg / 2;\n"
)


def test_compile_probe_keeps_the_error_line_and_logs_the_report(monkeypatch, caplog):
    def rejected(*args):
        raise RuntimeError(_TILE_COMPILER_ERROR)

    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "tile", rejected)
    with caplog.at_level("WARNING", logger="sous.engine.tileattn"):
        reason = tileattn._compile_probe()
    assert reason == "program_source:12:3: error: use of undeclared identifier 'x'"
    assert "x = sg / 2;" in caplog.text


def test_compile_probe_runs_every_row_variant_and_remembers_only_success(monkeypatch):
    rows = []

    def flaky(queries, keys, values, n, chunk):
        rows.append(queries.shape[2])
        if len(rows) == 1:
            raise RuntimeError("transient")
        return mx.zeros((1, queries.shape[2], tileattn.HQ, tileattn.D), dtype=mx.bfloat16)

    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "tile", flaky)
    assert tileattn._compile_probe() == "transient"
    assert tileattn._compile_probe() is None
    assert tileattn._compile_probe() is None
    assert rows == [1, *range(1, tileattn.MAX_RUN + 1)]


# ---- the real kernel (needs the tensor units; skipped where they are absent) -----


@tfx.nax
def test_every_row_variant_compiles_and_runs(monkeypatch):
    monkeypatch.setattr(tileattn, "_compile_ok", False)
    assert tileattn._compile_probe() is None


def _relative_rms(got, want):
    g, w = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2))).item()


def _fp32_reference(q, k, v):
    """softmax(q k^T * SCALE) v in float32 on the GPU, from the bf16 inputs; query
    head h reads KV head h // 6, as mlx's GQA does."""
    hkv, gqa, d = tileattn.HKV, tileattn.GQA, tileattn.D
    qf = q.astype(mx.float32).reshape(1, hkv, gqa, 1, d)
    kf = k.astype(mx.float32)[:, :, None]
    vf = v.astype(mx.float32)[:, :, None]
    p = mx.softmax((qf @ kf.swapaxes(-1, -2)) * tileattn.SCALE, axis=-1)
    return (p @ vf).reshape(1, tileattn.HQ, 1, d)


@tfx.nax
@pytest.mark.parametrize("n", [2048, 57_344, 196_608])
@pytest.mark.parametrize("gain", [1, 8])
def test_real_kernel_is_no_less_accurate_than_stock(n, gain):
    q, k, v = tfx.serving_qkv(n, 1, seed=n)
    q = q * gain  # exact in bf16: x8 sharpens the softmax
    want = _fp32_reference(q, k, v)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=tileattn.SCALE, mask=None)
    tile_error = _relative_rms(tileattn.decode(q, k, v, 20), want)
    stock_error = _relative_rms(stock, want)
    assert tile_error <= 1.05 * stock_error, (tile_error, stock_error)


# ---- the decode hook -------------------------------------------------------------


def _tile_counts_since(before):
    """The counters that moved since `before`, by how much."""
    return {k: n - before[k] for k, n in tileattn.calls.items() if n != before[k]}


def _language_module():
    return importlib.import_module("mlx_vlm.models.qwen3_5.language")


def _kv_cache_with(**attrs):
    from mlx_vlm.models.cache import KVCache

    cache = KVCache()
    for name, value in attrs.items():
        setattr(cache, name, value)
    return cache


def _subclass_cache():
    from mlx_vlm.models.cache import KVCache

    class Subclass(KVCache):
        pass

    return Subclass()


def _decode_call(n=2000, t=1, *, queries=None, dtype=None, cache=None, **kwargs):
    q, k, v = tfx.serving_qkv(n, t)
    if dtype is not None:
        q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
    cache = _kv_cache_with() if cache is None else cache
    call = {"cache": cache, "scale": tileattn.SCALE, "mask": None, **kwargs}
    return (q if queries is None else queries(q), k, v), call


# case -> (the call, the counter it moves; None moves none)
_PASSED_THROUGH = {
    "causal mask": (lambda: _decode_call(mask="causal"), "decode_out"),
    "array mask": (
        lambda: _decode_call(mask=mx.ones((1, 1, 1, 2000), dtype=mx.bool_)),
        "decode_out",
    ),
    "sinks": (lambda: _decode_call(sinks=mx.zeros((24,), dtype=mx.bfloat16)), "decode_out"),
    "float16": (lambda: _decode_call(dtype=mx.float16), "decode_out"),
    "scale 0.1": (lambda: _decode_call(scale=0.1), "decode_out"),
    "16 query heads": (lambda: _decode_call(queries=lambda q: q[:, :16]), "decode_out"),
    "KVCache subclass": (lambda: _decode_call(cache=_subclass_cache()), "decode_out"),
    "left padding": (
        lambda: _decode_call(cache=_kv_cache_with(left_padding=mx.array([1]))),
        "decode_out",
    ),
    "decode left padding": (
        lambda: _decode_call(cache=_kv_cache_with(_qwen3_5_decode_left_padding=[1])),
        "decode_out",
    ),
    "flag clear": (lambda: _decode_call(), None),
    "two rows": (lambda: _decode_call(t=2), None),
    "below N0": (lambda: _decode_call(n=tileattn.N0 - 1), "decode_stock"),
}


def test_the_decode_hook_serves_a_plain_one_row_call(monkeypatch):
    tfx.hooked(monkeypatch)
    q, k, v = tfx.serving_qkv(2000, 1)
    before = dict(tileattn.calls)
    got = _language_module().scaled_dot_product_attention(
        q, k, v, cache=_kv_cache_with(), scale=tileattn.SCALE, mask=None
    )
    assert _tile_counts_since(before) == {"decode_tile": 1}
    assert mx.array_equal(got, tileattn.decode(q, k, v, 20)).item()


@pytest.mark.parametrize("case", list(_PASSED_THROUGH))
def test_the_decode_hook_hands_every_other_call_to_mlx_vlm(monkeypatch, case):
    from mlx_vlm.models.base import scaled_dot_product_attention as original

    build, counter = _PASSED_THROUGH[case]
    tfx.hooked(monkeypatch, splits=0 if case == "flag clear" else 20)
    args, kwargs = build()
    before = dict(tileattn.calls)
    got = _language_module().scaled_dot_product_attention(*args, **kwargs)
    assert _tile_counts_since(before) == ({} if counter is None else {counter: 1})
    assert mx.array_equal(got, original(*args, **kwargs)).item()


def test_the_decode_hook_hands_a_quantized_cache_over_untouched(monkeypatch):
    """mlx-vlm's quantized path needs quantized K/V, so this one is checked against a
    recording original: the same arguments, and its answer returned."""
    seen = []

    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return "mlx-vlm's answer"

    wrapped = tileattn._decode_wrapper(original)
    assert inspect.unwrap(wrapped) is original and getattr(wrapped, tileattn._HOOK_MARK)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    q, k, v = tfx.serving_qkv(2000, 1)
    cache = _kv_cache_with(bits=4)
    before = dict(tileattn.calls)
    assert wrapped(q, k, v, cache=cache, scale=tileattn.SCALE, mask=None) == "mlx-vlm's answer"
    ((args, kwargs),) = seen
    assert all(a is b for a, b in zip(args, (q, k, v), strict=True))
    assert kwargs == {"cache": cache, "scale": tileattn.SCALE, "mask": None}
    assert _tile_counts_since(before) == {"decode_out": 1}


def test_install_decode_hook_is_idempotent(monkeypatch):
    from mlx_vlm.models.base import scaled_dot_product_attention as original

    language = _language_module()
    # Registers the restore: the replacement must not outlive this test.
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    tileattn.install_decode_hook()
    first = language.scaled_dot_product_attention
    tileattn.install_decode_hook()
    assert language.scaled_dot_product_attention is first
    assert getattr(first, tileattn._HOOK_MARK) and inspect.unwrap(first) is original


@pytest.fixture(scope="module")
def tiny_verifier():
    """The tiny model, its token ids, and prefilled caches by prefix. A verify
    forward rolls its cache back, so those caches are shared; a test that decodes
    builds its own."""
    lm = tfx.tiny_language_model()
    ids = mx.random.randint(0, tfx.VOCAB, (2000,), key=mx.random.key(7)).tolist()
    return lm, ids, {}


def test_a_real_attention_module_reaches_the_tile_once(tiny_verifier, monkeypatch):
    """One-row decode through Qwen3_5Attention.__call__ itself: its projections, rope
    and cache append, then the replaced global."""
    lm, ids, _ = tiny_verifier
    index = next(i for i, layer in enumerate(lm.layers) if not layer.is_linear)
    module = lm.layers[index].self_attn
    stock_cache = tfx.prefill_cache(lm, ids[:1700])[index]
    tile_cache = tfx.prefill_cache(lm, ids[:1700])[index]
    x = mx.random.normal((1, 1, 256), key=mx.random.key(3)).astype(mx.bfloat16)
    tfx.hooked(monkeypatch, splits=0)
    want = module(x, mask=None, cache=stock_cache)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    got = module(x, mask=None, cache=tile_cache)
    mx.eval(want, got)
    assert stock_cache.offset == tile_cache.offset == 1701
    assert _tile_counts_since(before) == {"decode_tile": 1}
    assert _relative_rms(got, want) <= 1e-2


# ---- the verify branch, through mlx-vlm's exact verifier -------------------------


def _prefilled_cache(verifier, prefix):
    lm, ids, caches = verifier
    if prefix not in caches:
        caches[prefix] = tfx.prefill_cache(lm, ids[:prefix])
    return caches[prefix]


@pytest.mark.slow
@pytest.mark.parametrize("prefix", [1000, 1530, 1700, 1915])
def test_verify_rows_follow_the_plan(tiny_verifier, monkeypatch, prefix):
    """Below N0, above it, and straddling N0 and the step at 1921, T = 1..16: each of
    the two full-attention layers sends one call, cut as verify_plan says."""
    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    tfx.hooked(monkeypatch)
    cache = _prefilled_cache(tiny_verifier, prefix)
    for t in range(1, 17):
        runs = tileattn.verify_plan(prefix, t, 20)
        tile_rows = sum(k - j for j, k, chunk in runs if chunk)
        expected = {
            "verify_calls": 2,
            f"verify_t{t}" if t <= tileattn.MAX_RUN else "verify_t9_up": 2,
            "verify_rows_tile": 2 * tile_rows,
            "verify_rows_stock": 2 * (t - tile_rows),
            "verify_launches": 2 * sum(1 for *_, chunk in runs if chunk),
            "verify_straddles": 2 * (len(runs) > 1),
        }
        before = dict(tileattn.calls)
        tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
        assert _tile_counts_since(before) == {k: n for k, n in expected.items() if n}, (prefix, t)
        assert all(c.offset == prefix for c in cache if hasattr(c, "offset"))


def test_an_untagged_verifier_never_reaches_the_tile(tiny_verifier, monkeypatch):
    lm, ids, _ = tiny_verifier
    for module in tfx.verify_ready(monkeypatch, lm):
        object.__setattr__(module, verifyattn._TAG, 0)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700:1703]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {}
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))


def test_a_tagged_verify_the_scope_refuses_is_counted_out_of_scope(tiny_verifier, monkeypatch):
    """A served turn should never take this path, so its count is the signal that a
    tagged module fell out of scope."""

    class OtherCache:
        pass

    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    monkeypatch.setattr(verifyattn, "_KVCACHE", OtherCache)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700:1703]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before, original = dict(tileattn.calls), verifyattn.calls["original"]
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {"verify_out": 2}
    assert verifyattn.calls["original"] - original == 2
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))


@pytest.mark.parametrize(("owner", "check"), [(verifyattn, "_post_ok"), (tileattn, "in_scope")])
@pytest.mark.parametrize("t", [1, 2, 3, 5])
def test_a_declined_verify_computes_todays_attention(tiny_verifier, monkeypatch, owner, check, t):
    """After the projections the rows are already appended, so a decline must
    reproduce the stock verifier (T = 1, 2) or the grouped path (T >= 3) itself."""
    lm, ids, _ = tiny_verifier
    tfx.verify_ready(monkeypatch, lm)
    cache = _prefilled_cache(tiny_verifier, 1700)
    tokens = ids[1700 : 1700 + t]
    tfx.hooked(monkeypatch, splits=0)
    stock = tfx.verify_forward(lm, cache, tokens)
    monkeypatch.setattr(owner, check, lambda *a: False)
    monkeypatch.setattr(tileattn, "_active_splits", 20)
    before = dict(tileattn.calls)
    ours = tfx.verify_forward(lm, cache, tokens)
    assert _tile_counts_since(before) == {"verify_declined": 2}
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))


# ---- the load-time probe -----------------------------------------------------------

_REJECTED_BUILD = (
    "[metal::Device] Unable to build metal library from source\n"
    "program_source:12:3: error: use of undeclared identifier 'x'\n"
)


def _guard_probe(monkeypatch) -> None:
    """Registers restores for every global a probe run touches, so a probe
    that fails a test cannot leak a hook, a flag or a compile verdict into
    the next one."""
    language = _language_module()
    monkeypatch.setattr(
        language, "scaled_dot_product_attention", language.scaled_dot_product_attention
    )
    monkeypatch.setattr(tileattn, "_active_splits", tileattn._active_splits)
    monkeypatch.setattr(tileattn, "_compile_ok", tileattn._compile_ok)


def _probe_setup(monkeypatch, tile=tfx.stand_in_tile) -> list:
    """`tile` as the kernel, a far mark of 3,201 keys instead of 57,601 (the
    stand-in is plain mlx, far slower than the kernel), a compile check that
    really runs, and the tiny model's full-attention modules."""
    _guard_probe(monkeypatch)
    monkeypatch.setattr(tileattn, "tile", tile)
    monkeypatch.setattr(tileattn, "_compile_ok", False)
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    return verifyattn._attention_modules(tfx.tiny_language_model())


def _run_probe(modules, arch: str | None = None) -> str | None:
    """probe() at S = 20, asserting it left the flag clear and the counters
    and mlx-vlm's global as it found them, whatever it decided."""
    installed = _language_module().scaled_dot_product_attention
    before = dict(tileattn.calls)
    reason = tileattn.probe(modules, 20, arch or tileattn._arch())
    assert tileattn._active_splits == 0
    assert tileattn.calls == before
    assert _language_module().scaled_dot_product_attention is installed
    return reason


def test_probe_far_mark_is_the_first_step_boundary_at_or_past_the_probe_keys(monkeypatch):
    assert tileattn.far_mark(20) == 57_601
    assert tileattn.far_mark(10) == 57_281
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_000)
    assert tileattn.far_mark(20) == 3_201
    monkeypatch.setattr(tileattn, "PROBE_FAR_KEYS", 3_201)
    assert tileattn.far_mark(20) == 3_201  # a boundary is its own far mark


def test_probe_parity_cases_cover_the_first_steps_the_far_mark_and_stock_transitions():
    marks, stock = tileattn.parity_cases(20, "applegpu_g17s")
    assert marks == [1536, 1921, 2561, 57601]
    # 's' parts change plan at 1024 and 1025 keys, both below N0: rows on both sides
    assert stock == [(1021, 2), (1022, 1), (1022, 2), (1023, 1), (1023, 2), (1024, 1), (1024, 2)]
    # 'g' parts change plan only at 4096 keys, where the tile serves
    assert tileattn.parity_cases(20, "applegpu_g14g") == ([1536, 1921, 2561, 57601], [])


@pytest.mark.slow
def test_probe_passes_the_stand_in(monkeypatch):
    assert _run_probe(_probe_setup(monkeypatch)) is None


@pytest.mark.slow
def test_probe_checks_the_stock_side_on_any_gpu_and_reuses_an_installed_hook(monkeypatch):
    modules = _probe_setup(monkeypatch)
    # The hook an earlier load installed, and the flag enable() cleared first.
    tfx.hooked(monkeypatch, splits=20)
    monkeypatch.setattr(tileattn, "_active_splits", 0)
    hook = _language_module().scaled_dot_product_attention
    assert tileattn.parity_cases(20, "applegpu_g17s")[1]  # the stock cases run on this GPU
    assert _run_probe(modules, "applegpu_g17s") is None
    assert _language_module().scaled_dot_product_attention is hook


def test_probe_fails_parity_for_a_kernel_that_ignores_the_stepped_chunk(monkeypatch):
    modules = _probe_setup(
        monkeypatch, lambda q, k, v, n, chunk: tfx.stand_in_tile(q, k, v, n, -(-n // 20))
    )
    reason = _run_probe(modules)
    assert reason is not None and reason.startswith("parity:"), reason


def test_probe_fails_correctness_for_a_kernel_wrong_the_same_way_on_both_paths(monkeypatch):
    def scaled(q, k, v, n, chunk):
        out = tfx.stand_in_tile(q, k, v, n, chunk)
        return (out.astype(mx.float32) * 1.1).astype(mx.bfloat16)

    modules = _probe_setup(monkeypatch, scaled)
    # Parity cannot see this kernel, so the sweep past N0 would only cost time.
    monkeypatch.setattr(tileattn, "parity_cases", lambda splits, arch: ([tileattn.N0], []))
    reason = _run_probe(modules)
    assert reason is not None and reason.startswith("correctness:"), reason


def test_probe_reports_a_kernel_that_fails_to_build_with_its_error_line(monkeypatch):
    def rejected(q, k, v, n, chunk):
        raise RuntimeError(_REJECTED_BUILD)

    modules = _probe_setup(monkeypatch, rejected)
    reason = _run_probe(modules)
    assert reason == "kernel: program_source:12:3: error: use of undeclared identifier 'x'"


def test_probe_refuses_a_module_that_never_reaches_the_tile(monkeypatch):
    modules = _probe_setup(monkeypatch)
    monkeypatch.setattr(tileattn, "parity_cases", lambda splits, arch: ([tileattn.N0], []))
    monkeypatch.setattr(tileattn, "_plain_kv_cache", lambda cache: False)
    assert _run_probe(modules) == "the model's attention module did not reach the tile"


def test_probe_restores_what_it_touched_when_it_raises(monkeypatch):
    def fails_past_the_first_step(q, k, v, n, chunk):
        if n >= 1921:
            raise RuntimeError("device lost")
        return tfx.stand_in_tile(q, k, v, n, chunk)

    modules = _probe_setup(monkeypatch, fails_past_the_first_step)
    installed = _language_module().scaled_dot_product_attention
    before = dict(tileattn.calls)
    with pytest.raises(RuntimeError, match="device lost"):
        tileattn.probe(modules, 20, tileattn._arch())
    assert tileattn._active_splits == 0
    assert tileattn.calls == before
    assert _language_module().scaled_dot_product_attention is installed


@tfx.nax
def test_probe_passes_the_real_kernel(monkeypatch):
    _guard_probe(monkeypatch)
    modules = verifyattn._attention_modules(tfx.tiny_language_model())
    start = time.perf_counter()
    reason = _run_probe(modules)
    assert reason is None, f"{reason} (after {time.perf_counter() - start:.2f}s)"
