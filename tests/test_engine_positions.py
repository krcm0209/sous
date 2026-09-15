"""The VLM backend hands mlx-vlm explicit positions on every prefill and
decode. mlx-vlm's text-only embedding helper returns positions local to the
ids it is given, generate_step merges them into the model's kwargs, and a
Qwen-style language model slices that array at its cache offset — past its
end for any continuation generate_step chunks — so the fused MRoPE kernel
reads a zero-length buffer out of bounds and every continued token lands at
position 0. The kwargs are the cure, for exactly the models whose helper
returns positions; exercised here without mlx-vlm by stubbing its two entry
points, the way tests/test_engine_unloaded.py builds engines without weights."""

import sys
import threading
import types
import warnings

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
    text-only at all."""

    def __init__(self, positions: bool | None = True):
        self.positions = positions
        self.probes = 0
        self.language_model = object()
        self.config = types.SimpleNamespace(eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        self.probes += 1
        if self.positions is None:
            raise RuntimeError("needs pixels")
        return _Embeddings(input_ids if self.positions else None)


def _engine(model: _Model) -> VLMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = model  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()  # ty: ignore[invalid-assignment]
    engine._positional = None
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._cache = PrefixCache(engine, enabled=True)
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


def test_an_empty_cache_positions_from_zero():
    kw = _engine(_Model())._positions([_Layer(offset=0), _Layer()], 3)
    assert kw["position_ids"].tolist() == [[0, 1, 2]]


def test_a_helper_that_returns_no_positions_means_no_kwargs():
    # Such a model derives cache_offset + rope_deltas itself and consumes a
    # caller's array verbatim in its own rank: leaving it alone is the fix.
    assert _engine(_Model(positions=False))._positions([_Layer(offset=40)], 7) == {}


def test_a_helper_that_cannot_run_text_only_means_no_kwargs():
    assert _engine(_Model(positions=None))._positions([_Layer(offset=40)], 7) == {}


def test_a_helper_that_cannot_run_text_only_warns_once():
    engine = _engine(_Model(positions=None))
    with pytest.warns(UserWarning, match="could not probe test/model"):
        assert engine._positions([_Layer(offset=40)], 7) == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            engine._positions([_Layer(offset=40)], 7) == {}
        )  # cached: no second probe, no second warning


def test_the_helper_is_probed_once_per_engine():
    model = _Model()
    engine = _engine(model)
    for _ in range(3):
        engine._positions([_Layer(offset=40)], 7)
    assert model.probes == 1
    assert engine._positional is True


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
    # The existing early return keeps its meaning: no ids, no call, no probe.
    called = []
    stub = types.SimpleNamespace(generate=lambda *a, **k: called.append(k), stream_generate=None)
    monkeypatch.setitem(sys.modules, "mlx_vlm", stub)
    model = _Model()
    _engine(model).prefill([_Layer(offset=40)], [])
    assert called == []
    assert model.probes == 0
