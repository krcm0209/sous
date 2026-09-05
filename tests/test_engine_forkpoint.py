"""The engines' fork boundary: what two probe renders of the system turn have
in common, verified as a token prefix of the whole render — exercised without
mlx by faking the tokenizer, the way tests/test_engine_unloaded.py builds
engines."""

import threading
from typing import cast

import pytest

from sous.engine.lm import LMEngine
from sous.engine.promptcache import FORK_MIN_TOKENS, CacheHooks, PrefixCache, PromptMemo
from sous.engine.vlm import VLMEngine


def _render(messages, add_generation_prompt, header_chars):
    out = []
    for m in messages:
        out.append("S" * header_chars if m["role"] == "system" else m["content"])
    if add_generation_prompt:
        out.append("G")
    return "|".join(out)


class FakeTokenizer:
    """Renders messages as a fixed-width token per character so a header that
    is a text prefix is also a token prefix, and vice versa.

    Refuses a message list with no user turn, exactly as the default model's
    template does (`raise_exception("No user query found in messages.")`) —
    the whole reason the boundary is probed rather than rendered directly."""

    bos_token = None

    def __init__(self, header_chars: int = FORK_MIN_TOKENS):
        self.header_chars = header_chars
        self.encoded: list[str] = []

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if not any(m["role"] == "user" for m in messages):
            raise Exception("No user query found in messages.")
        return _render(messages, add_generation_prompt, self.header_chars)

    def encode(self, text, add_special_tokens):
        self.encoded.append(text)
        return [ord(c) for c in text]


class Tolerant(FakeTokenizer):
    """A template that renders a system-only list instead of refusing it, the
    way Qwen3-0.6B's does. Both kinds are in the wild; only this one lets a
    system-only conversation reach the engine at all."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        return _render(messages, add_generation_prompt, self.header_chars)


# The header text the probe finds for a default-width fake: everything the
# two probe renders share, which is the system turn plus the separator the
# template puts before the first user turn's content.
HEADER = "S" * FORK_MIN_TOKENS + "|"


class Recording:
    """A PrefixCache stand-in recording the fork_at it was handed.

    A callable is resolved through the real `PrefixCache._fork_wanted` — with
    a budget and a cold miss, the only conditions under which it resolves one
    — so these tests see the shipped rule (including its warn-and-zero on a
    probe that raises) rather than a second copy of it here. That resolver's
    hooks are never reached: nothing below `generate` runs."""

    enabled = True

    def __init__(self):
        self.fork_ats: list[int] = []
        self._resolver = PrefixCache(cast("CacheHooks", None), max_bytes=1 << 40)

    def generate(self, stable_ids, full_ids, max_tokens, on_delta=None, fork_at=0):
        if callable(fork_at):
            fork_at = self._resolver._fork_wanted(
                threading.current_thread(), list(stable_ids), fork_at, 0
            )
        self.fork_ats.append(fork_at)
        return "text"

    def stats(self, owner=None):
        return {}

    def reset(self, owner=None):
        pass


class Spy(Recording):
    """Records the owner every passthrough was called with."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def stats(self, owner=None):
        self.seen.append(("stats", owner))
        return {}

    def reset(self, owner=None):
        self.seen.append(("reset", owner))


class Rewriting(FakeTokenizer):
    """A template that changes the system turn once the conversation has
    history. The two probe renders still agree with each other — a probe can
    only ever vary the first user turn — so what catches this is fork_point's
    per-turn check that the header is a token prefix of the real render."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        text = super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )
        return text.replace("S", "s", 1) if len(messages) > 2 else text


class RefusingProbes(FakeTokenizer):
    """A template that renders the real conversation but refuses the probe —
    whatever a probe render happens to contain, it raises."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if any(m["role"] == "user" and m["content"] in ("0", "1") for m in messages):
            raise Exception("no probing")
        return super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )


def _engine(tokenizer) -> tuple[LMEngine, Recording]:
    engine = object.__new__(LMEngine)
    engine.model_id = "test/model"
    engine._model = object()
    engine._tokenizer = tokenizer
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    rec = Recording()
    engine._cache = rec  # ty: ignore[invalid-assignment]
    return engine, rec


def _vlm_engine(tokenizer) -> tuple[VLMEngine, Recording]:
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    # _loaded() only checks for None; the fake never reaches the model.
    engine._model = object()  # ty: ignore[invalid-assignment]
    # VLMEngine._tokenizer reads `processor.tokenizer` when there is one, and
    # falls back to the processor itself — which is what the fake is.
    engine._processor = tokenizer
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    rec = Recording()
    engine._cache = rec  # ty: ignore[invalid-assignment]
    return engine, rec


SYSTEM = {"role": "system", "content": "ignored by the fake"}
USER = {"role": "user", "content": "hello"}
ASSISTANT = {"role": "assistant", "content": "hi"}


def test_a_leading_system_turn_long_enough_is_the_fork_point():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    # One past the system render: the probes share the separator the template
    # emits before the first user turn's content, and everything they share is
    # header — the boundary is no longer the system turn's own render length.
    assert rec.fork_ats == [FORK_MIN_TOKENS + 1]


def test_a_short_system_turn_does_not_fork():
    engine, rec = _engine(FakeTokenizer(header_chars=10))
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_no_system_turn_means_no_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([USER], [], 8)
    assert rec.fork_ats == [0]


def test_a_system_only_prompt_does_not_fork():
    # Nothing would follow the header, but _header_probe's `len(messages) < 2`
    # guard returns first: no probe is rendered and fork_point is never asked.
    # A tolerant template, because a system-only conversation cannot reach the
    # engine at all under one that refuses to render it.
    engine, rec = _engine(Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [0]


def test_a_template_whose_header_is_not_a_token_prefix_does_not_fork():
    engine, rec = _engine(Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [0]


def test_a_template_that_refuses_the_probe_warns_and_does_not_fork():
    engine, rec = _engine(RefusingProbes())
    with pytest.warns(UserWarning, match="header probe"):
        assert engine.generate([SYSTEM, USER], [], 8) == "text"
    assert rec.fork_ats == [0]


def test_the_header_is_encoded_once_across_turns():
    tokenizer = FakeTokenizer()
    engine, rec = _engine(tokenizer)
    engine.generate([SYSTEM, USER], [], 8)
    assert engine._memo.get("header", HEADER) == [ord(c) for c in HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], [], 8)
    # The renders repeat (they are cheap and text-keyed); the tokenize of the
    # ~50K-token header is what the memo slot has to save.
    assert tokenizer.encoded.count(HEADER) == 1
    assert rec.fork_ats == [FORK_MIN_TOKENS + 1, FORK_MIN_TOKENS + 1]


def test_the_header_render_is_skipped_when_the_cache_is_disabled():
    engine, rec = _engine(FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]
    assert engine._memo.get("header", HEADER) is None


def test_owner_scoped_stats_and_reset_pass_through():
    engine, _ = _engine(FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]


# ---- the same nine, for the VLM backend -----------------------------------
#
# _encode is monkeypatched away because its real body imports mlx_vlm; it is
# routed to the fake tokenizer's own encode() — the one the LM tests exercise
# — so the encode count is taken the same way on both backends, and the file
# stays runnable on a machine without mlx.


def _mlx_free_vlm(monkeypatch, tokenizer) -> tuple[VLMEngine, Recording]:
    monkeypatch.setattr(VLMEngine, "_encode", lambda self, text: self._tokenizer.encode(text, True))
    return _vlm_engine(tokenizer)


def test_vlm_a_leading_system_turn_long_enough_is_the_fork_point(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [FORK_MIN_TOKENS + 1]


def test_vlm_a_short_system_turn_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer(header_chars=10))
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_no_system_turn_means_no_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([USER], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_a_system_only_prompt_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_a_template_whose_header_is_not_a_token_prefix_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_a_template_that_refuses_the_probe_warns_and_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, RefusingProbes())
    with pytest.warns(UserWarning, match="header probe"):
        assert engine.generate([SYSTEM, USER], [], 8) == "text"
    assert rec.fork_ats == [0]


def test_vlm_the_header_is_encoded_once_across_turns(monkeypatch):
    tokenizer = FakeTokenizer()
    engine, rec = _mlx_free_vlm(monkeypatch, tokenizer)
    engine.generate([SYSTEM, USER], [], 8)
    assert engine._memo.get("header", HEADER) == [ord(c) for c in HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], [], 8)
    assert tokenizer.encoded.count(HEADER) == 1
    assert rec.fork_ats == [FORK_MIN_TOKENS + 1, FORK_MIN_TOKENS + 1]


def test_vlm_the_header_render_is_skipped_when_the_cache_is_disabled(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]
    assert engine._memo.get("header", HEADER) is None


def test_vlm_owner_scoped_stats_and_reset_pass_through(monkeypatch):
    engine, _ = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]
