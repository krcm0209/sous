"""At temperature 0 the VLM engine hands mlx-vlm no sampler: its generate_step
takes `sampler is None and temperature == 0` as the greedy case, and only
that case runs the speculative walk's exact-match verify. An argmax sampler
built by make_sampler is a callable, so it measured the sampled code path
under a greedy pick."""

import sys
import threading
import types

from sous.engine.promptcache import PromptMemo
from sous.engine.vlm import VLMEngine


class _Layer:
    def __init__(self, offset: int = 0):
        self.offset = offset


def _engine(sampler) -> VLMEngine:
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = types.SimpleNamespace(config=types.SimpleNamespace(eos_token_id=0))  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = sampler
    engine._positional = False
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    return engine


def _stub(monkeypatch, calls: dict):
    def stream_generate(model, processor, prompt, **kwargs):
        calls.update(kwargs)
        return iter(())

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )


def test_temperature_zero_builds_no_sampler(monkeypatch):
    made = []
    # The import inside _make_sampler is `from mlx_vlm.sample_utils import
    # make_sampler`: a stub under that name in sys.modules is what it finds.
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **kw: made.append(kw) or object()),
    )
    assert VLMEngine._make_sampler(temperature=0, top_p=0.8, top_k=20) is None
    assert made == []
    assert VLMEngine._make_sampler(temperature=0.7, top_p=0.8, top_k=20) is not None
    assert made == [{"temp": 0.7, "top_p": 0.8, "top_k": 20}]


def test_decode_without_a_sampler_asks_for_greedy_generation(monkeypatch):
    calls: dict = {}
    _stub(monkeypatch, calls)
    _engine(None).decode([_Layer()], [4, 5], 8)
    assert calls["sampler"] is None
    assert calls["temperature"] == 0


def test_decode_with_a_sampler_passes_it_and_no_temperature(monkeypatch):
    calls: dict = {}
    _stub(monkeypatch, calls)
    sampler = object()
    _engine(sampler).decode([_Layer()], [4, 5], 8)
    assert calls["sampler"] is sampler
    assert "temperature" not in calls
