"""The projection kernel through mlx-vlm's real qwen3_5 forward, exact verifier and generate_step.

A tiny random qwen3_5 language model, quantized affine 4-bit gs64 as the 27B is, is prefilled
through the real forward. Verify rows, as DFlash runs them, go through the exact verifier's
``_linear``/``_linears`` (hook 1); step-by-step one-row decode on a fresh prefill of the same prefix
goes through ``nn.QuantizedLinear.__call__`` (hook 2): a hybrid cache cannot rewind past a decode
step. Stock's own exactness is asserted first. With the kernel (the plain-mlx stand-in on any GPU,
the real kernel wherever it compiles) every verify row must stay bitwise equal to decode, both
hooks must be reached by every projection, and with the flag cleared both paths give the stock bits
again. The real ``generate_step`` must reach hook 2 on a short prefill tail and on decode.
"""

import collections
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import projkernel  # noqa: E402 — after the importorskip guard
from tests import proj_fixtures as pfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402

PREFIX = 40  # above MAX_ROWS: the prefill itself stays on stock in every test
# Hook-1 calls per verify forward: four per layer (q+k+v and o, or qkv+z+b+a and
# out_proj, then gate+up and down) and the head.
VERIFY_CALLS = 4 * 4 + 1
# Hook-2 calls per one-row forward: seven projections per full-attention layer,
# eight per linear-attention layer, and the head.
PLAIN_CALLS = 2 * 7 + 2 * 8 + 1


@pytest.fixture(scope="module")
def tiny():
    lm = pfx.tiny_quantized_model()
    ids = mx.random.randint(0, tfx.VOCAB, (PREFIX + 12,), key=mx.random.key(148)).tolist()
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


def _kernel_on(monkeypatch, lm, *, stand_in: bool = True) -> None:
    """Both hooks in place, `lm` tagged and the flag set for this test only;
    the stand-in or the real kernel behind them."""
    pfx.guard_proj_state(monkeypatch)
    pfx.hooked(monkeypatch)
    if not stand_in:
        pfx.real_kernel(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 1)
    projkernel.tag(lm)


def _decoded(t: int, *, stand_in: bool) -> dict[str, int]:
    """The counters t one-row decode steps move: every projection through hook
    two, and on the real kernel each one a one-row dispatch."""
    moved = {"plain_kernel": PLAIN_CALLS * t}
    if not stand_in:
        moved["one_row"] = PLAIN_CALLS * t
    return moved


def _rows_of(x) -> int:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return rows


@pytest.mark.parametrize("t", [2, 5, 8])
def test_stock_verify_rows_equal_step_by_step_decode_on_the_quantized_model(tiny, t, monkeypatch):
    """Stock's own exactness on this GPU, which every assertion below rests on."""
    lm, ids = tiny
    pfx.guard_proj_state(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 0)
    assert all(_same(_verify_rows(lm, ids, PREFIX, t), _decode_rows(lm, ids, PREFIX, t)))


@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize("t", [2, 3, 5, 8])
def test_the_kernel_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, t, stand_in, monkeypatch
):
    lm, ids = tiny
    pfx.guard_proj_state(monkeypatch)
    monkeypatch.setattr(projkernel, "_active", 0)
    stock_verify = _verify_rows(lm, ids, PREFIX, t)
    stock_decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(stock_verify, stock_decode)), "stock's own exactness"

    _kernel_on(monkeypatch, lm, stand_in=stand_in)
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, t)
    verified = dict(projkernel.calls)
    decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(verify, decode)), t
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS}
    assert _delta(verified, projkernel.calls) == _decoded(t, stand_in=stand_in)

    monkeypatch.setattr(projkernel, "_active", 0)
    start = dict(projkernel.calls)
    assert all(_same(_verify_rows(lm, ids, PREFIX, t), stock_verify))
    assert all(_same(_decode_rows(lm, ids, PREFIX, t), stock_decode))
    assert projkernel.calls == start


def test_a_kernel_whose_rows_move_with_the_row_count_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: the equality above is not vacuous. A kernel whose
    rows depend on how many rows share the call gives a five-row verify and
    one-row decode different logits."""
    lm, ids = tiny
    _kernel_on(monkeypatch, lm)

    def by_rows(x, weights):
        factor = 1 + _rows_of(x) / 64
        return tuple(
            (o.astype(mx.float32) * factor).astype(mx.bfloat16)
            for o in pfx.stand_in_mma(x, weights)
        )

    monkeypatch.setattr(projkernel, "mma", by_rows)
    assert not all(_same(_verify_rows(lm, ids, PREFIX, 5), _decode_rows(lm, ids, PREFIX, 5)))


@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=pfx.kernel)], ids=["stand-in", "kernel"]
)
def test_a_verify_of_more_than_eight_rows_is_split_and_equals_its_rows_alone(
    tiny, stand_in, monkeypatch
):
    """Block 0 hands the drafter's own policy the depth, which can verify up to
    16 rows: every projection call is cut into runs of at most MAX_ROWS rows,
    and each of its twelve rows still equals one-row decode."""
    lm, ids = tiny
    t = 12
    _kernel_on(monkeypatch, lm, stand_in=stand_in)
    kernel = projkernel.mma
    rows: list[int] = []

    def counting(x, weights):
        rows.append(_rows_of(x))
        return kernel(x, weights)

    monkeypatch.setattr(projkernel, "mma", counting)
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, t)
    verified = dict(projkernel.calls)
    launches = list(rows)
    decode = _decode_rows(lm, ids, PREFIX, t)
    assert all(_same(verify, decode))
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS, "verify_split": VERIFY_CALLS}
    assert len(launches) == 2 * VERIFY_CALLS
    assert max(launches) <= projkernel.MAX_ROWS and sum(launches) == t * VERIFY_CALLS
    assert _delta(verified, projkernel.calls) == _decoded(t, stand_in=stand_in)


class _Wrapper:
    """The mlx-vlm model generate_step drives: a text-only embedding helper
    over the tiny language model, handing the explicit positions and zero
    delta the VLM engine hands mlx-vlm."""

    def __init__(self, lm):
        self.language_model = lm
        self.config = types.SimpleNamespace(model_type="qwen3_5", eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        n = input_ids.shape[1]
        return types.SimpleNamespace(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids),
            to_dict=lambda: {
                "inputs_embeds": None,
                "position_ids": mx.arange(n, dtype=mx.int32)[None],
                "rope_deltas": mx.zeros((1, 1), dtype=mx.int32),
            },
        )


def test_generate_step_reaches_hook_two_on_the_prefill_tail_and_on_decode(tiny, monkeypatch):
    """mlx-vlm's own generate loop: a 38-token prompt prefills as a 32-row
    chunk (stock), a 5-row tail and the one-row step that ends every prefill,
    then decodes two tokens. Every projection of the tail, of that step and of
    each decode step is served, which is why stored forks depend on the kernel."""
    from mlx_vlm.generate.ar import generate_step

    lm, _ = tiny
    _kernel_on(monkeypatch, lm)
    rows: list[int] = []

    def counting(x, weights):
        rows.append(_rows_of(x))
        return pfx.stand_in_mma(x, weights)

    monkeypatch.setattr(projkernel, "mma", counting)
    prompt = mx.random.randint(0, tfx.VOCAB, (1, 38), key=mx.random.key(7)).astype(mx.int32)
    start = dict(projkernel.calls)
    tokens = [
        token
        for token, _ in generate_step(
            prompt,
            _Wrapper(lm),  # ty: ignore[invalid-argument-type]
            None,
            None,
            max_tokens=2,
            prefill_step_size=32,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        )
    ]
    assert len(tokens) == 2
    assert collections.Counter(rows) == {5: PLAIN_CALLS, 1: 3 * PLAIN_CALLS}
    assert _delta(start, projkernel.calls) == {"plain_kernel": 4 * PLAIN_CALLS}


def test_enable_on_the_tiny_model_serves_both_real_paths(tiny, monkeypatch):
    """enable() itself, its real pins, warm-up and probe and the hooks it
    installs, over the real forward and exact verifier."""
    lm, ids = tiny
    tile = pfx.proj_ready(monkeypatch, real_pins=True)
    status = projkernel.enable(lm, enabled=True, drafter_kind="dflash", tile_status=tile)
    assert status["state"] == "active", status
    start = dict(projkernel.calls)
    verify = _verify_rows(lm, ids, PREFIX, 5)
    verified = dict(projkernel.calls)
    decode = _decode_rows(lm, ids, PREFIX, 5)
    assert all(_same(verify, decode))
    assert _delta(start, verified) == {"verify_kernel": VERIFY_CALLS}
    assert _delta(verified, projkernel.calls) == {"plain_kernel": 5 * PLAIN_CALLS}
