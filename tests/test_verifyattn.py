"""Exact grouped verify attention (sous.engine.verifyattn).

The plan and grouping tables are pure Python. The GPU tests run on any Metal
GPU (CI is macos-15), and the slow tests repeat them in subprocesses with
MLX_METAL_GPU_ARCH set, so the 's' and 'd' dispatch tables the M5 Pro takes are
proven bit-exact on whatever GPU runs the suite.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import verifyattn  # noqa: E402 — after the importorskip guard

S, D, G = "applegpu_g13s", "applegpu_g13d", "applegpu_g14g"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("arch", "n_keys", "gqa", "q_len", "expected"),
    [
        # the vector path's limits
        (S, 5000, 6, 9, ("fallback", 0)),
        (S, 5000, 6, 6, ("fallback", 0)),  # 6 x 6 > 32
        (S, 2, 6, 3, ("fallback", 0)),  # more queries than keys
        # 's': two passes from 1024 keys; more blocks only past 1024 and n_simds > 4
        (S, 1023, 6, 1, ("1pass", 0)),
        (S, 1024, 6, 1, ("2pass", 64)),
        (S, 1025, 6, 1, ("2pass", 128)),
        (S, 8192, 6, 1, ("2pass", 128)),
        (S, 8193, 6, 1, ("2pass", 256)),
        (S, 32768, 6, 1, ("2pass", 256)),
        (S, 32769, 6, 1, ("2pass", 512)),
        (S, 65536, 6, 1, ("2pass", 512)),
        (S, 65537, 6, 1, ("2pass", 1024)),
        (S, 5000, 4, 1, ("2pass", 64)),  # n_simds 4 is not > 4
        (S, 5000, 4, 2, ("2pass", 128)),
        # 'd'
        (D, 1023, 6, 1, ("1pass", 0)),
        (D, 1024, 6, 1, ("2pass", 128)),
        (D, 16383, 6, 1, ("2pass", 128)),
        (D, 16384, 6, 1, ("2pass", 512)),
        (D, 65535, 6, 1, ("2pass", 512)),
        (D, 65536, 6, 1, ("2pass", 1024)),
        (D, 8193, 2, 1, ("2pass", 256)),  # n_simds <= 2 past 8192
        (D, 20000, 4, 1, ("2pass", 128)),  # n_simds 4: neither branch
        # every other suffix: two passes only with GQA from 4096 keys
        (G, 4095, 6, 1, ("1pass", 0)),
        (G, 4096, 6, 1, ("2pass", 64)),
        (G, 4096, 2, 1, ("2pass", 32)),
        (G, 100000, 1, 1, ("1pass", 0)),
    ],
)
def test_plan_mirrors_mlx_0_32_2_dispatch(arch, n_keys, gqa, q_len, expected):
    assert verifyattn.plan(arch, n_keys, gqa, q_len) == expected


@pytest.mark.parametrize(
    ("arch", "prefix", "t", "gqa", "expected"),
    [
        (G, 4093, 3, 6, [(0, 2), (2, 3)]),  # rows at 4094, 4095 | 4096
        (G, 4093, 8, 6, [(0, 2), (2, 7), (7, 8)]),  # past 5 rows the call falls back
        (S, 1021, 4, 6, [(0, 2), (2, 3), (3, 4)]),  # 1024 and 1025 are two transitions
        (S, 8190, 8, 6, [(0, 2), (2, 7), (7, 8)]),
        (S, 32766, 3, 6, [(0, 2), (2, 3)]),
        (S, 57000, 8, 6, [(0, 5), (5, 8)]),
        (S, 57000, 1, 6, [(0, 1)]),
    ],
)
def test_groups_split_at_every_plan_transition(arch, prefix, t, gqa, expected):
    assert verifyattn.groups(arch, prefix, t, gqa) == expected


@pytest.mark.parametrize(
    ("arch", "gqa", "expected"),
    [
        (S, 6, (1024, 1025, 8193, 32769, 65537)),
        (D, 6, (1024, 16384, 65536)),
        (G, 6, (4096,)),
        (G, 4, (4096,)),
    ],
)
def test_plan_transitions_are_the_dispatch_thresholds(arch, gqa, expected):
    assert verifyattn.plan_transitions(arch, gqa) == expected


def test_probe_cases_straddle_every_transition_and_stop_past_the_last():
    cases = verifyattn.probe_cases(S, 6)
    assert max(p + t for p, t in cases) == 65537 + 7
    for n in verifyattn.plan_transitions(S, 6):
        for t in range(verifyattn.MIN_ROWS, verifyattn.MAX_ROWS + 1):
            # some row sits at n - 1 and the next at n
            assert any(p + 1 <= n - 1 and n <= p + t for p, tt in cases if tt == t), (n, t)
    assert all(verifyattn.MIN_ROWS <= t <= verifyattn.MAX_ROWS for _, t in cases)
    assert max(p + t for p, t in verifyattn.probe_cases(G, 6)) == 4096 + 7


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(24, 4), (16, 4)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_probe_proves_grouping_exact_on_this_gpu(q_heads, kv_heads, dtype):
    arch = mx.device_info()["architecture"]
    assert verifyattn.probe(arch, q_heads, kv_heads, getattr(mx, dtype)) is None


def test_probe_catches_a_grouping_that_ignores_the_plan(monkeypatch):
    # One call over as many rows as the vector path takes, across transitions:
    # the probe must see the rows whose plan it changed.
    monkeypatch.setattr(
        verifyattn,
        "groups",
        lambda arch, prefix, t, gqa: (
            [(0, min(t, 32 // gqa))] + [(r, r + 1) for r in range(min(t, 32 // gqa), t)]
        ),
    )
    arch = mx.device_info()["architecture"]
    reason = verifyattn.probe(arch, 24, 4, mx.bfloat16)
    assert reason is not None and reason.startswith("grouped attention differs")


@pytest.mark.slow
@pytest.mark.parametrize("arch", [S, D])
def test_exactness_holds_under_every_dispatch_table(arch):
    """mlx reads MLX_METAL_GPU_ARCH once, when it builds the device, and routes
    SDPA by it; a child process with it set runs the other suffixes' tables on
    this GPU."""
    env = dict(os.environ, MLX_METAL_GPU_ARCH=arch)
    tests = [
        "tests/test_verifyattn.py::test_probe_proves_grouping_exact_on_this_gpu",
        "tests/test_verifyattn.py::test_probe_catches_a_grouping_that_ignores_the_plan",
        "tests/test_verifyattn.py::test_hooked_verify_forward_is_bit_equal_to_stock",
        "tests/test_verifyattn.py::test_one_call_over_the_straddle_would_differ",
    ]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "passed" in done.stdout and "skipped" not in done.stdout


@pytest.mark.slow
def test_the_arch_override_reaches_mlx():
    code = "import mlx.core as mx; print(mx.device_info()['architecture'])"
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, MLX_METAL_GPU_ARCH=S),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.stdout.strip() == S, done.stderr


def test_the_locked_mlx_is_validated():
    """A dependency bump fails here until someone re-reads mlx's SDPA dispatch
    against plan() and extends VALIDATED_MLX."""
    assert mx.__version__ in verifyattn.VALIDATED_MLX


def test_the_locked_mlx_vlm_sources_are_validated():
    """A dependency bump that changes any function the hook reads fails here
    until someone re-reads it and adds its new hash."""
    for qualname, module in verifyattn._SOURCES.items():
        digest = verifyattn._source_digest(module, qualname)
        assert digest in verifyattn.VALIDATED_MLX_VLM_SOURCES[qualname], qualname
    assert verifyattn._gate() is None


def test_gate_refuses_an_unvalidated_mlx(monkeypatch):
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    assert verifyattn._gate() == "mlx 0.99.0 not validated"


def test_gate_refuses_a_forced_sdpa_block_count(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "128")
    assert verifyattn._gate() == "MLX_SDPA_BLOCKS is set"


def test_gate_ignores_a_zero_sdpa_block_count(monkeypatch):
    monkeypatch.setenv("MLX_SDPA_BLOCKS", "0")
    assert verifyattn._gate() is None


def test_gate_refuses_a_changed_mlx_vlm_function(monkeypatch):
    monkeypatch.setitem(
        verifyattn.VALIDATED_MLX_VLM_SOURCES, "KVCache.make_mask", frozenset({"0" * 64})
    )
    assert verifyattn._gate() == "mlx-vlm KVCache.make_mask changed"


def test_an_unreadable_source_counts_as_changed():
    assert verifyattn._source_digest("mlx_vlm.models.cache", "KVCache.no_such_method") is None


GQA = 6
VOCAB = 512


def _tiny_language_model():
    """A random qwen3_5 language model with two full-attention layers at the
    27B's head dim and GQA ratio; mlx-vlm's verify path takes it unquantized."""
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    args = TextConfig(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=GQA,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=1,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,  # layers 1 and 3 are full attention
    )
    mx.random.seed(0)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    return lm


@pytest.fixture(scope="module")
def tiny():
    """The tiny model prefilled to just below this GPU's first plan transition,
    so verify rows of T = 3..5 straddle it."""
    arch = mx.device_info()["architecture"]
    lm = _tiny_language_model()
    prefix = verifyattn.plan_transitions(arch, GQA)[0] - 3
    ids = mx.random.randint(0, VOCAB, (prefix + 8,), key=mx.random.key(7)).tolist()
    cache = lm.make_cache()
    # Explicit positions and a zero delta, as the VLM engine hands them: the
    # bare language model cannot derive rope positions without a vision config.
    lm(
        mx.array([ids[:prefix]], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(prefix, dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    )
    mx.eval([c.state for c in cache])
    return lm, cache, ids, prefix


def _tag(lm, gqa):
    for module in verifyattn._attention_modules(lm):
        object.__setattr__(module, verifyattn._TAG, gqa)


def _ready(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    verifyattn.install_wrapper()
    monkeypatch.setattr(verifyattn, "_ARCH", mx.device_info()["architecture"])
    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)


def _verify(lm, cache, tokens):
    """One verify forward as DFlash runs it, then the speculative round aborted
    so the cache is back at the prefix."""
    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        capture_layer_ids=[0, 1, 2],
        speculative_verify=True,
    )
    arrays = [out.logits, *out.hidden_states]
    mx.eval(arrays)
    out.gdn_states.abort()
    return arrays


@pytest.mark.parametrize("t", [2, 3, 4, 5])
def test_hooked_verify_forward_is_bit_equal_to_stock(tiny, t, monkeypatch):
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + t]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    _tag(lm, GQA)
    before = dict(verifyattn.calls)
    try:
        ours = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert all(c.offset == prefix for c in cache if hasattr(c, "offset"))
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))
    grouped = verifyattn.calls["grouped"] - before["grouped"]
    original = verifyattn.calls["original"] - before["original"]
    # T = 2 is stock's own single call; both full-attention layers group from T = 3
    assert (grouped, original) == ((0, 2) if t == 2 else (2, 0))


def test_one_call_over_the_straddle_would_differ(tiny, monkeypatch):
    """The sensitivity half: ignoring the plan changes the logits, so the
    bit-equality above is not vacuous."""
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + 3]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    monkeypatch.setattr(verifyattn, "groups", lambda arch, p, t, gqa: [(0, t)])
    _tag(lm, GQA)
    try:
        bad = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert not mx.array_equal(stock[0], bad[0]).item()


@pytest.mark.parametrize("t", [3, 5])
def test_the_post_projection_fallback_is_the_stock_loop(tiny, t, monkeypatch):
    lm, cache, ids, prefix = tiny
    _ready(monkeypatch)
    tokens = ids[prefix : prefix + t]
    _tag(lm, 0)
    stock = _verify(lm, cache, tokens)
    monkeypatch.setattr(verifyattn, "_post_ok", lambda *a: False)
    _tag(lm, GQA)
    before = verifyattn.calls["loop"]
    try:
        ours = _verify(lm, cache, tokens)
    finally:
        _tag(lm, 0)
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, ours, strict=True))
    assert verifyattn.calls["loop"] - before == 2


def test_install_wrapper_is_idempotent():
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    verifyattn.install_wrapper()
    first = Qwen3_5BatchInvariantForward._attention
    verifyattn.install_wrapper()
    assert Qwen3_5BatchInvariantForward._attention is first


def _scope(
    monkeypatch,
    *,
    t=3,
    batch=1,
    mask="causal",
    cache=None,
    head_dim=256,
    tag=GQA,
    heads=24,
    kv_heads=4,
):
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    monkeypatch.setattr(verifyattn, "_KVCACHE", KVCache)
    attention = types.SimpleNamespace(
        head_dim=head_dim, num_attention_heads=heads, num_key_value_heads=kv_heads
    )
    setattr(attention, verifyattn._TAG, tag)
    return verifyattn._in_scope(
        Qwen3_5BatchInvariantForward(),
        attention,
        mx.zeros((batch, t, 8)),
        mask,
        KVCache() if cache is None else cache,
    )


def test_scope_takes_a_plain_causal_verify_of_three_to_eight_rows(monkeypatch):
    assert _scope(monkeypatch, t=3) and _scope(monkeypatch, t=8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"t": 2},  # stock's own single call
        {"t": 1},
        {"t": 9},
        {"batch": 2},
        {"mask": None},
        {"mask": "left_padded_decode"},
        {"head_dim": 128},
        {"tag": 0},  # untagged
        {"tag": 4},  # probed for another GQA ratio
    ],
)
def test_scope_refuses_everything_else(monkeypatch, overrides):
    assert not _scope(monkeypatch, **overrides)


def test_scope_refuses_an_array_mask(monkeypatch):
    assert not _scope(monkeypatch, mask=mx.ones((1, 1, 3, 3), dtype=mx.bool_))


def test_scope_refuses_kvcache_subclasses_quantized_and_left_padded_caches(monkeypatch):
    from mlx_vlm.models.cache import KVCache

    class Subclass(KVCache):
        pass

    quantized = KVCache()
    quantized.bits = 4  # ty: ignore[unresolved-attribute]
    padded = KVCache()
    padded._qwen3_5_decode_left_padding = [1]  # ty: ignore[unresolved-attribute]
    for cache in (Subclass(), quantized, padded):
        assert not _scope(monkeypatch, cache=cache), type(cache).__name__


def _fresh_model():
    return _tiny_language_model()


def test_enable_off_does_nothing():
    lm = _fresh_model()
    assert verifyattn.enable(lm, enabled=False) == {
        "state": "off",
        "reason": None,
        "probe_seconds": None,
    }
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_tags_every_full_attention_layer_with_the_proved_ratio():
    lm = _fresh_model()
    status = verifyattn.enable(lm, enabled=True)
    try:
        assert status["state"] == "active" and status["reason"] is None
        assert isinstance(status["probe_seconds"], float)
        modules = verifyattn._attention_modules(lm)
        assert len(modules) == 2 and all(getattr(m, verifyattn._TAG) == GQA for m in modules)
    finally:
        verifyattn._untag(lm)


def test_enable_refuses_other_model_types(monkeypatch):
    lm = _fresh_model()
    # The language model carries model_type itself, and _model_type reads it first.
    monkeypatch.setattr(lm, "model_type", "qwen3_5_moe")
    with pytest.warns(UserWarning, match="unsupported model type 'qwen3_5_moe'"):
        status = verifyattn.enable(lm, enabled=True)
    assert status == {
        "state": "unavailable",
        "reason": "unsupported model type 'qwen3_5_moe' (qwen3_5 only)",
        "probe_seconds": None,
    }


def test_enable_reports_a_failed_gate(monkeypatch):
    lm = _fresh_model()
    monkeypatch.setattr(verifyattn, "_gate", lambda: "mlx 0.99.0 not validated")
    with pytest.warns(UserWarning, match="mlx 0.99.0 not validated"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable" and status["reason"] == "mlx 0.99.0 not validated"
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_reports_a_failed_probe_with_its_cost(monkeypatch):
    lm = _fresh_model()
    monkeypatch.setattr(verifyattn, "probe", lambda *a: "grouped attention differs at prefix 1")
    with pytest.warns(UserWarning, match="differs at prefix 1"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable"
    assert isinstance(status["probe_seconds"], float)
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))


def test_enable_never_raises(monkeypatch):
    lm = _fresh_model()

    def boom(*a):
        raise RuntimeError("metal said no")

    monkeypatch.setattr(verifyattn, "probe", boom)
    with pytest.warns(UserWarning, match="metal said no"):
        status = verifyattn.enable(lm, enabled=True)
    assert status["state"] == "unavailable" and status["reason"] == "metal said no"
    assert all(not getattr(m, verifyattn._TAG, 0) for m in verifyattn._attention_modules(lm))
