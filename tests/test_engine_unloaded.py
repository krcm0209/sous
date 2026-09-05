"""Using an engine after unload() must fail loudly with a named error.

These build engines via object.__new__ in the exact state unload() leaves
behind, so they exercise the guards without downloading a real MLX model —
the model-marked engine tests cannot run in CI.
"""

import threading

import pytest

from sous.engine.lm import LMEngine
from sous.engine.vlm import VLMEngine

MESSAGES = [{"role": "user", "content": "hi"}]


def _unloaded_lm() -> LMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(LMEngine)
    engine.model_id = "test/model"
    engine._model = None
    engine._tokenizer = None
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._cache = PrefixCache(engine, enabled=True)
    return engine


def _unloaded_vlm() -> VLMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = None
    engine._processor = None
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._cache = PrefixCache(engine, enabled=True)
    return engine


def test_lm_count_tokens_after_unload_raises_runtime_error():
    with pytest.raises(RuntimeError, match="has been unloaded"):
        _unloaded_lm().count_tokens(MESSAGES, [])


def test_lm_prompt_after_unload_raises_runtime_error():
    with pytest.raises(RuntimeError, match="has been unloaded"):
        _unloaded_lm()._prompt(MESSAGES, [])


def test_vlm_count_tokens_after_unload_raises_runtime_error():
    # Regression: _tokenizer used to read _processor directly, so this path
    # returned None and blew up with a bare AttributeError instead.
    with pytest.raises(RuntimeError, match="has been unloaded"):
        _unloaded_vlm().count_tokens(MESSAGES, [])


def test_vlm_prompt_after_unload_raises_runtime_error():
    with pytest.raises(RuntimeError, match="has been unloaded"):
        _unloaded_vlm()._prompt(MESSAGES, [])


def test_unloaded_error_names_the_model():
    with pytest.raises(RuntimeError, match="test/model"):
        _unloaded_lm().count_tokens(MESSAGES, [])


def test_lm_reset_prompt_cache_works_after_unload():
    """run_task resets in a finally, which can land after an idle unload. The
    helper leaves a live PrefixCache beside a dead model, which is exactly the
    state unload() produces — reset must not reach for the model."""
    engine = _unloaded_lm()
    engine.reset_prompt_cache()  # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0


def test_vlm_reset_prompt_cache_works_after_unload():
    engine = _unloaded_vlm()
    engine.reset_prompt_cache()  # must not raise
    assert engine.prompt_cache_stats()["hits"] == 0


def test_lm_headroom_never_raises_without_mlx(monkeypatch):
    import sous.engine.base as base

    monkeypatch.setattr(base, "live_headroom", lambda: None)
    assert _unloaded_lm().headroom() is None
