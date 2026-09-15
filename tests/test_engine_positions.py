"""The VLM backend hands mlx-vlm explicit positions on every prefill and
decode. mlx-vlm's text-only embedding helper returns positions local to the
ids it is given, generate_step merges them into the model's kwargs, and a
Qwen-style language model slices that array at its cache offset — past its
end for any continuation generate_step chunks — so the fused MRoPE kernel
reads a zero-length buffer out of bounds and every continued token lands at
position 0. The kwargs are the cure, for exactly the models whose helper
returns positions; exercised here without weights by stubbing mlx-vlm's two
entry points, the way tests/test_engine_unloaded.py builds engines without
them — and, for the one line of mlx-vlm the cure rests on, by driving the
real generate_step with a stub model."""

import sys
import threading
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine.vlm import VLMEngine  # noqa: E402 — after the importorskip guard


class _Layer:
    """An attention layer carries an int offset; a recurrent layer carries none."""

    def __init__(self, offset: int | None = None):
        if offset is not None:
            self.offset = offset


class _Embeddings:
    def __init__(self, position_ids):
        self.position_ids = position_ids


class _Model:
    """The wrapper model, reduced to its text-only embedding helper: it either
    returns rotary positions of its own (the Qwen lineage), returns none (a
    family that positions from the cache offset itself), or cannot run
    text-only at all. Records how it was called."""

    def __init__(self, positions: bool | None = True):
        self.positions = positions
        self.calls: list[tuple[tuple, dict]] = []
        self.language_model = object()
        self.config = types.SimpleNamespace(eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        self.calls.append(((input_ids, pixel_values), kwargs))
        if self.positions is None:
            raise RuntimeError("needs pixels")
        return _Embeddings(input_ids if self.positions else None)


def _engine(model: _Model, positional: bool = True) -> VLMEngine:
    from sous.engine.promptcache import PromptMemo

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = model  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()  # ty: ignore[invalid-assignment]
    engine._positional = positional
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    return engine


def test_positions_cover_the_cache_from_zero_through_the_new_tokens():
    # Recurrent layers first, as the default model lays them out: the offset
    # comes from the attention layers, whichever index they sit at.
    cache = [_Layer(), _Layer(), _Layer(offset=40), _Layer(offset=40)]
    kw = _engine(_Model())._positions(cache, 7)
    assert kw["position_ids"].shape == (1, 47)
    assert kw["position_ids"].dtype == mx.int32
    assert kw["position_ids"][0, 40:].tolist() == list(range(40, 47))
    assert kw["rope_deltas"].shape == (1, 1)
    assert kw["rope_deltas"].item() == 0


def test_a_cache_at_offset_zero_positions_from_zero():
    kw = _engine(_Model())._positions([_Layer(offset=0), _Layer()], 3)
    assert kw["position_ids"].tolist() == [[0, 1, 2]]
    assert _engine(_Model())._positions([], 3)["position_ids"].tolist() == [[0, 1, 2]]


def test_a_model_that_positions_itself_gets_no_kwargs():
    # Such a model derives cache_offset + rope_deltas itself and consumes a
    # caller's array verbatim in its own rank: leaving it alone is the fix.
    engine = _engine(_Model(positions=False), positional=False)
    assert engine._positions([_Layer(offset=40)], 7) == {}
    assert engine.positions == "model"


def test_the_probe_reads_the_helpers_positions_and_calls_it_as_mlx_vlm_does():
    model = _Model()
    engine = _engine(model)
    assert engine._helper_returns_positions() is True
    assert _engine(_Model(positions=False))._helper_returns_positions() is False
    # The shape generate_step calls the helper with for a text-only prompt:
    # one token of ids, pixel_values None in the second positional, mask=None.
    # A probe that asked any other way could fail — and answer "model" — for
    # a helper every real turn runs fine.
    ((ids, pixels), kwargs) = model.calls[0]
    assert ids.tolist() == [[0]] and pixels is None and kwargs == {"mask": None}


def test_a_helper_that_cannot_run_text_only_means_the_models_positions_and_a_warning():
    with pytest.warns(UserWarning, match="could not probe test/model"):
        assert _engine(_Model(positions=None))._helper_returns_positions() is False


def test_prefill_and_decode_hand_mlx_vlm_the_positions(monkeypatch):
    calls: dict[str, dict] = {}

    def generate(model, processor, prompt, **kwargs):
        calls["prefill"] = kwargs

    def stream_generate(model, processor, prompt, **kwargs):
        calls["decode"] = kwargs
        return iter(())

    stub = types.SimpleNamespace(generate=generate, stream_generate=stream_generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm", stub)
    engine = _engine(_Model())
    cache = [_Layer(), _Layer(offset=40)]
    engine.prefill(cache, [1, 2, 3])
    engine.decode(cache, [4, 5], 8)
    assert calls["prefill"]["position_ids"].tolist() == [list(range(43))]
    assert calls["prefill"]["rope_deltas"].tolist() == [[0]]
    assert calls["prefill"]["max_tokens"] == 0
    assert calls["decode"]["position_ids"].tolist() == [list(range(42))]
    assert calls["decode"]["rope_deltas"].tolist() == [[0]]
    assert calls["decode"]["max_tokens"] == 8


def test_prefill_of_nothing_asks_mlx_vlm_for_nothing(monkeypatch):
    # The existing early return keeps its meaning: no ids, no call.
    called = []
    stub = types.SimpleNamespace(generate=lambda *a, **k: called.append(k), stream_generate=None)
    monkeypatch.setitem(sys.modules, "mlx_vlm", stub)
    _engine(_Model()).prefill([_Layer(offset=40)], [])
    assert called == []


# ---- the mlx-vlm contract -------------------------------------------------


class _LocalPositionsHelper:
    """A wrapper model whose text-only helper returns positions local to the
    ids it is given — and distinguishable from anything the engine sends —
    over a language model that records the kwargs each forward receives."""

    def __init__(self):
        self.seen: list[dict] = []
        self.config = types.SimpleNamespace(model_type="fake", eos_token_id=0)
        self.language_model = self

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        n = input_ids.shape[1]
        return types.SimpleNamespace(
            inputs_embeds=mx.zeros((1, n, 4)),
            to_dict=lambda: {
                "inputs_embeds": None,
                "position_ids": mx.full((1, n), 999, dtype=mx.int32),
                "rope_deltas": mx.full((1, 1), 7, dtype=mx.int32),
            },
        )

    def __call__(self, inputs=None, inputs_embeds=None, cache=None, **kwargs):
        self.seen.append(kwargs)
        assert inputs_embeds is not None  # generate_step embeds the prompt before every forward
        n = inputs_embeds.shape[1]
        return types.SimpleNamespace(
            logits=mx.zeros((1, n, 8)), cross_attention_states=None, encoder_outputs=None
        )


def _forward_kwargs(**engine_kwargs) -> list[dict]:
    """Every kwargs dict the language model saw while the real generate_step
    processed a two-token prompt behind a 40-token cache."""
    from mlx_vlm.generate.ar import generate_step

    model = _LocalPositionsHelper()
    cache = types.SimpleNamespace(offset=40, state=None)
    list(
        generate_step(
            mx.array([[5, 6]]),
            model,  # ty: ignore[invalid-argument-type]
            None,
            None,
            prompt_cache=[cache],
            max_tokens=0,
            sampler=lambda logprobs: mx.zeros((logprobs.shape[0],), dtype=mx.int32),
            **engine_kwargs,
        )
    )
    assert model.seen, "generate_step never ran the language model"
    return model.seen


def test_mlx_vlm_hands_the_model_the_engines_positions_over_the_helpers():
    """The one line the cure rests on: generate_step re-applies a caller's
    position_ids/rope_deltas over the helper's own (from mlx-vlm 0.7.0; below
    it the helper's won and the engine's kwargs were inert). A bump that
    reorders that merge, or filters the two kwargs — GenerateKwargs does not
    declare them — would silently revert every continuation to the helper's
    local positions, and CI cannot load a model to notice. This test can,
    without weights: it only needs the merge."""
    engine = _engine(_Model())
    for kwargs in _forward_kwargs(**engine._positions([_Layer(offset=40)], 2)):
        assert kwargs["position_ids"].tolist() == [list(range(42))]
        assert kwargs["rope_deltas"].tolist() == [[0]]


def test_without_the_engines_kwargs_the_helpers_local_positions_reach_the_model():
    # The sensitivity half: the assertion above is only meaningful if the
    # helper's values would otherwise get through.
    for kwargs in _forward_kwargs():
        assert set(kwargs["position_ids"].flatten().tolist()) == {999}
        assert kwargs["rope_deltas"].tolist() == [[7]]
