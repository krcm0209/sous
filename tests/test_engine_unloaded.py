"""Using an engine after unload() must fail loudly with a named error.

These build engines via object.__new__ in the exact state unload() leaves
behind, so they exercise the guards without downloading a real MLX model —
the model-marked engine tests cannot run in CI.
"""

import threading

import pytest

from sous.engine import draftctx
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
    engine._positional = True  # unload() leaves the load-time probe's answer alone
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_context = draftctx.OFF  # unload() closes the drafter's context
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


def test_vlm_unload_turns_the_attention_tile_off(monkeypatch):
    from sous.engine import tileattn

    monkeypatch.setattr(tileattn, "_active_splits", 20)
    _unloaded_vlm().unload()
    assert tileattn._active_splits == 0


def test_vlm_unload_turns_the_projection_kernel_off(monkeypatch):
    from sous.engine import projkernel

    monkeypatch.setattr(projkernel, "_active", 1)
    _unloaded_vlm().unload()
    assert projkernel._active == 0


class _ContextThatFailsToClose:
    def close(self) -> None:
        raise RuntimeError("close failed")


@pytest.mark.parametrize("failing", ["prompt cache", "draft context"])
def test_vlm_unload_clears_the_tile_and_projection_kernel_before_anything_can_raise(
    monkeypatch, failing
):
    from sous.engine import projkernel, tileattn

    def fails(*args, **kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(tileattn, "_active_splits", 20)
    monkeypatch.setattr(projkernel, "_active", 1)
    engine = _unloaded_vlm()
    if failing == "prompt cache":
        monkeypatch.setattr(engine, "reset_prompt_cache", fails)
    else:
        engine._draft_context = _ContextThatFailsToClose()  # ty: ignore[invalid-assignment]
    with pytest.raises(RuntimeError, match="failed"):
        engine.unload()
    assert tileattn._active_splits == 0
    assert projkernel._active == 0


def test_lm_headroom_never_raises_without_mlx(monkeypatch):
    import sous.engine.base as base

    monkeypatch.setattr(base, "live_headroom", lambda: None)
    assert _unloaded_lm().headroom() is None
