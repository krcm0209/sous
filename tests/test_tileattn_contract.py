"""The attention tile through mlx-vlm's real qwen3_5 forward and exact verifier.

A tiny random language model at the 27B's attention shape (24 query and 4 KV heads, head dim 256,
bf16, two full-attention layers) is prefilled through the real forward. Verify rows, as DFlash
runs them, are compared with step-by-step one-row decode on a fresh prefill of the same prefix:
a hybrid cache cannot rewind past a decode step. Stock's own exactness is asserted first; the
tile (the plain-mlx stand-in on any GPU, the real kernel under ``nax``) must then keep every verify
row bitwise equal to decode, and with the flag cleared both paths must give the stock bits again.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import tileattn  # noqa: E402 — after the importorskip guard
from tests import tile_fixtures as tfx  # noqa: E402

SPLITS = 20
LAYERS = 2  # the tiny model's full-attention layers: every forward makes two attention calls
TILE_CASES = [
    (1000, 2),  # below N0: grouped stock rows, T = 2 included
    (1000, 3),
    (1533, 5),  # two stock rows, then three tile rows from N0
    (1700, 1),
    (1700, 2),  # the DFlash round loop's last round
    (1700, 3),
    (1700, 8),
    (1918, 5),  # straddles the first step boundary above N0 (1921 at S = 20)
]


@pytest.fixture(scope="module")
def tiny():
    lm = tfx.tiny_language_model()
    ids = mx.random.randint(0, tfx.VOCAB, (2600,), key=mx.random.key(132)).tolist()
    return lm, ids


def _verify_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    logits = tfx.verify_forward(lm, cache, ids[prefix : prefix + t])[0]
    return [logits[:, r] for r in range(t)]


def _decode_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    rows = []
    for tok in ids[prefix : prefix + t]:
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache).logits
        mx.eval(logits)
        rows.append(logits[:, -1])
    return rows


def _same(a, b):
    return [mx.array_equal(x, y).item() for x, y in zip(a, b, strict=True)]


def _delta(before, after):
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def _verify_counts(prefix, t):
    plan = tileattn.verify_plan(prefix, t, SPLITS)
    tile_rows = sum(k - j for j, k, chunk in plan if chunk)
    counts = {
        "verify_calls": LAYERS,
        f"verify_t{t}": LAYERS,
        "verify_rows_tile": LAYERS * tile_rows,
        "verify_rows_stock": LAYERS * (t - tile_rows),
        "verify_launches": LAYERS * sum(1 for *_, chunk in plan if chunk),
        "verify_straddles": LAYERS * (len(plan) > 1),
    }
    return {k: v for k, v in counts.items() if v}


def _decode_counts(prefix, t):
    tile = sum(1 for r in range(t) if prefix + r + 1 >= tileattn.N0)
    counts = {"decode_tile": LAYERS * tile, "decode_stock": LAYERS * (t - tile)}
    return {k: v for k, v in counts.items() if v}


@pytest.mark.parametrize(("prefix", "t"), [(1000, 2), (1000, 3), (1700, 3)])
def test_stock_verify_rows_equal_step_by_step_decode(tiny, prefix, t, monkeypatch):
    """Stock's own exactness on this GPU, which every tile assertion below
    rests on. Should it ever fail on CI, quantize the tiny model
    (nn.quantize(lm, group_size=64, bits=4)), as the 27B is, rather than
    weaken a tile assertion."""
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    assert tileattn._active_splits == 0
    assert all(_same(_verify_rows(lm, ids, prefix, t), _decode_rows(lm, ids, prefix, t)))


@pytest.mark.slow
@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=tfx.nax)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize(("prefix", "t"), TILE_CASES)
def test_the_tile_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, prefix, t, stand_in, monkeypatch
):
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    stock_verify = _verify_rows(lm, ids, prefix, t)
    stock_decode = _decode_rows(lm, ids, prefix, t)
    assert all(_same(stock_verify, stock_decode)), "stock's own exactness"

    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=stand_in)
    start = dict(tileattn.calls)
    verify = _verify_rows(lm, ids, prefix, t)
    verified = dict(tileattn.calls)
    decode = _decode_rows(lm, ids, prefix, t)
    assert all(_same(verify, decode)), (prefix, t)
    assert _delta(start, verified) == _verify_counts(prefix, t)
    assert _delta(verified, tileattn.calls) == _decode_counts(prefix, t)

    monkeypatch.setattr(tileattn, "_active_splits", 0)
    start = dict(tileattn.calls)
    assert all(_same(_verify_rows(lm, ids, prefix, t), stock_verify))
    assert all(_same(_decode_rows(lm, ids, prefix, t), stock_decode))
    assert tileattn.calls == start


def test_a_kernel_that_ignores_the_schedule_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: rows 1718..1722 share chunk 96, but a kernel
    that sizes its splits from the call's own key count (ceil(n / 20)) gives
    the verify call and the one-row call at 1,718 different partitions, so
    the equality above is not vacuous."""
    lm, ids = tiny
    tfx.verify_ready(monkeypatch, lm)
    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=True)
    monkeypatch.setattr(
        tileattn, "tile", lambda q, k, v, n, chunk: tfx.stand_in_tile(q, k, v, n, -(-n // 20))
    )
    assert not all(_same(_verify_rows(lm, ids, 1717, 5), _decode_rows(lm, ids, 1717, 5)))
