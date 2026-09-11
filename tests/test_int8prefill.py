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
