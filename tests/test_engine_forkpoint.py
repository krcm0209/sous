"""The engines' fork boundaries: what two probe pairs' renders have in common
— the tool block, and the whole system block — each verified as a token
prefix of the whole render; exercised without mlx by faking the tokenizer,
the way tests/test_engine_unloaded.py builds engines."""

import threading
from typing import cast

import pytest

from sous.engine.lm import LMEngine
from sous.engine.promptcache import FORK_MIN_TOKENS, CacheHooks, PrefixCache, PromptMemo
from sous.engine.vlm import VLMEngine

# The fake's tool block, whenever a conversation has tools: long enough to be
# a fork boundary on its own.
TOOL_BLOCK = "T" * FORK_MIN_TOKENS


def _render(messages, tools, add_generation_prompt):
    """The default model's shape, one token per character: with tools, the
    system turn is the tool block, then a separator and the client's text —
    the separator only when there is text, as Qwen3.5/3.8's template does;
    without tools, just the text."""
    out = []
    for m in messages:
        if m["role"] == "system":
            block = TOOL_BLOCK if tools else ""
            sep = "\n\n" if tools and m["content"] else ""
            out.append(block + sep + m["content"])
        else:
            out.append(m["content"])
    if add_generation_prompt:
        out.append("G")
    return "|".join(out)


class FakeTokenizer:
    """Renders messages as a fixed-width token per character so a text prefix
    is also a token prefix, and vice versa.

    Refuses a message list with no user turn, exactly as the default model's
    template does (`raise_exception("No user query found in messages.")`) —
    the whole reason the boundaries are probed rather than rendered directly."""

    bos_token = None

    def __init__(self):
        self.encoded: list[str] = []

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if not any(m["role"] == "user" for m in messages):
            raise Exception("No user query found in messages.")
        return _render(messages, tools, add_generation_prompt)

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
        return _render(messages, tools, add_generation_prompt)


class SystemFirst(FakeTokenizer):
    """A template that renders the client's text BEFORE the tool block, the
    way Qwen3's does: the tools probe pair then shares nothing above the
    system text, and the header is the only boundary."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if not any(m["role"] == "user" for m in messages):
            raise Exception("No user query found in messages.")
        out = []
        for m in messages:
            if m["role"] == "system":
                out.append(m["content"] + ("\n\n" + TOOL_BLOCK if tools else ""))
            else:
                out.append(m["content"])
        if add_generation_prompt:
            out.append("G")
        return "|".join(out)


class Rewriting(FakeTokenizer):
    """A template that changes the system turn once the conversation has
    history. The probe renders still agree with each other — a probe can only
    ever vary the first user turn or the system text — so what catches this is
    fork_point's per-turn check that each boundary is a token prefix of the
    real render."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        text = super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )
        return text.replace("S", "s", 1) if len(messages) > 2 else text


class RefusingProbes(FakeTokenizer):
    """A template that renders the real conversation but refuses the probes —
    whatever a probe render happens to contain, it raises."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if any(m["role"] == "user" and m["content"] in ("0", "1") for m in messages):
            raise Exception("no probing")
        return super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )


class Recording:
    """A PrefixCache stand-in recording the fork_at it was handed.

    A callable is resolved through the real `PrefixCache._fork_boundaries` —
    with a budget and a cold miss — so these tests see the shipped rule
    (including its warn-and-empty on a probe that raises) rather than a second
    copy of it here. `reuse=0` is simply how this stand-in resolves it — not
    the only condition under which a real turn's probe would resolve — and
    that resolver's hooks are never reached: nothing below `generate` runs."""

    enabled = True

    def __init__(self):
        self.fork_ats: list[list[int]] = []
        self._resolver = PrefixCache(cast("CacheHooks", None), max_bytes=1 << 40)

    def generate(self, stable_ids, full_ids, max_tokens, on_delta=None, fork_at=()):
        if callable(fork_at):
            fork_at = self._resolver._fork_boundaries(
                threading.current_thread(), list(stable_ids), fork_at, 0
            )
        self.fork_ats.append(list(fork_at))
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


SYSTEM_TEXT = "S" * FORK_MIN_TOKENS
SYSTEM = {"role": "system", "content": SYSTEM_TEXT}
# Another session's system text: the same length, different at the first
# character — where a per-session scratchpad path would differ.
OTHER_SYSTEM = {"role": "system", "content": "s" + SYSTEM_TEXT[1:]}
SHORT_SYSTEM = {"role": "system", "content": "S" * 10}
EMPTY_SYSTEM = {"role": "system", "content": ""}
USER = {"role": "user", "content": "hello"}
ASSISTANT = {"role": "assistant", "content": "hi"}
TOOLS = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]

# What the probes find on the default-shaped fake. Without tools the header is
# the system text plus the separator before the first user turn's content.
# With tools the tool block plus its separator is the tools boundary, and the
# header runs on through the system text to that same user separator.
HEADER = SYSTEM_TEXT + "|"
HEADER_AT = len(HEADER)  # FORK_MIN_TOKENS + 1
TOOLS_TEXT = TOOL_BLOCK + "\n\n"
TOOLS_AT = len(TOOLS_TEXT)  # FORK_MIN_TOKENS + 2
TOOLED_HEADER = TOOLS_TEXT + SYSTEM_TEXT + "|"
TOOLED_HEADER_AT = len(TOOLED_HEADER)


def test_a_leading_system_turn_long_enough_is_a_header_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    # One past the system render: the probes share the separator the template
    # emits before the first user turn's content, and everything they share is
    # header. No tools, so no tools boundary.
    assert rec.fork_ats == [[HEADER_AT]]


def test_with_tools_rendered_first_the_tool_block_is_a_second_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]]


def test_two_sessions_with_different_system_text_share_the_tools_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_a_template_that_renders_the_system_text_first_has_no_tools_boundary():
    engine, rec = _engine(SystemFirst())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    # The two sessions' renders diverge inside the first line, so the tools
    # pair shares only what precedes the text: nothing here, a few tokens on
    # Qwen3. The header is the whole system turn, tool block included.
    assert rec.fork_ats == [[len(SYSTEM_TEXT) + 2 + len(TOOL_BLOCK) + 1]]


def test_a_system_turn_with_no_text_has_no_tools_boundary():
    # The template renders no separator before empty text, so the tools text
    # (block plus separator) is not a prefix of this render; the header — the
    # block plus the user separator — still is. No special case needed.
    engine, rec = _engine(FakeTokenizer())
    engine.generate([EMPTY_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(TOOL_BLOCK) + 1]]


def test_a_short_system_turn_does_not_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SHORT_SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[]]


def test_no_system_turn_means_no_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([USER], TOOLS, 8)
    assert rec.fork_ats == [[]]


def test_a_system_only_prompt_does_not_fork():
    # Nothing would follow the header, but _fork_probe's `len(messages) < 2`
    # guard returns first: no probe is rendered and fork_point is never asked.
    # A tolerant template, because a system-only conversation cannot reach the
    # engine at all under one that refuses to render it.
    engine, rec = _engine(Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [[]]


def test_a_template_whose_header_is_not_a_token_prefix_does_not_fork():
    engine, rec = _engine(Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [[]]


def test_a_template_that_refuses_the_probe_warns_and_does_not_fork():
    engine, rec = _engine(RefusingProbes())
    with pytest.warns(UserWarning, match="fork probe"):
        assert engine.generate([SYSTEM, USER], TOOLS, 8) == "text"
    assert rec.fork_ats == [[]]


def test_the_boundary_texts_are_encoded_once_across_turns():
    tokenizer = FakeTokenizer()
    engine, rec = _engine(tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert engine._memo.get("tools", TOOLS_TEXT) == [ord(c) for c in TOOLS_TEXT]
    assert engine._memo.get("header", TOOLED_HEADER) == [ord(c) for c in TOOLED_HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], TOOLS, 8)
    # The renders repeat (they are cheap and text-keyed); the two ~50K-token
    # encodes are what the memo slots have to save.
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_a_second_session_re_encodes_only_its_own_header():
    tokenizer = FakeTokenizer()
    engine, rec = _engine(tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1  # same tool set: memo hit
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert tokenizer.encoded.count(TOOLS_TEXT + OTHER_SYSTEM["content"] + "|") == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_the_probe_is_skipped_when_the_cache_is_disabled():
    engine, rec = _engine(FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[]]
    assert engine._memo.get("tools", TOOLS_TEXT) is None
    assert engine._memo.get("header", TOOLED_HEADER) is None


def test_owner_scoped_stats_and_reset_pass_through():
    engine, _ = _engine(FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]


# ---- the same, for the VLM backend -----------------------------------------
#
# _encode is monkeypatched away because its real body imports mlx_vlm; it is
# routed to the fake tokenizer's own encode() — the one the LM tests exercise
# — so the encode count is taken the same way on both backends, and the file
# stays runnable on a machine without mlx.


def _mlx_free_vlm(monkeypatch, tokenizer) -> tuple[VLMEngine, Recording]:
    monkeypatch.setattr(VLMEngine, "_encode", lambda self, text: self._tokenizer.encode(text, True))
    return _vlm_engine(tokenizer)


def test_vlm_a_leading_system_turn_long_enough_is_a_header_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[HEADER_AT]]


def test_vlm_with_tools_rendered_first_the_tool_block_is_a_second_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]]


def test_vlm_two_sessions_with_different_system_text_share_the_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_a_template_that_renders_the_system_text_first_has_no_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, SystemFirst())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(SYSTEM_TEXT) + 2 + len(TOOL_BLOCK) + 1]]


def test_vlm_a_system_turn_with_no_text_has_no_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([EMPTY_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(TOOL_BLOCK) + 1]]


def test_vlm_a_short_system_turn_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SHORT_SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_no_system_turn_means_no_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([USER], TOOLS, 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_system_only_prompt_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_template_whose_header_is_not_a_token_prefix_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_template_that_refuses_the_probe_warns_and_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, RefusingProbes())
    with pytest.warns(UserWarning, match="fork probe"):
        assert engine.generate([SYSTEM, USER], TOOLS, 8) == "text"
    assert rec.fork_ats == [[]]


def test_vlm_the_boundary_texts_are_encoded_once_across_turns(monkeypatch):
    tokenizer = FakeTokenizer()
    engine, rec = _mlx_free_vlm(monkeypatch, tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert engine._memo.get("tools", TOOLS_TEXT) == [ord(c) for c in TOOLS_TEXT]
    assert engine._memo.get("header", TOOLED_HEADER) == [ord(c) for c in TOOLED_HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_a_second_session_re_encodes_only_its_own_header(monkeypatch):
    tokenizer = FakeTokenizer()
    engine, rec = _mlx_free_vlm(monkeypatch, tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert tokenizer.encoded.count(TOOLS_TEXT + OTHER_SYSTEM["content"] + "|") == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_the_probe_is_skipped_when_the_cache_is_disabled(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[]]
    assert engine._memo.get("tools", TOOLS_TEXT) is None
    assert engine._memo.get("header", TOOLED_HEADER) is None


def test_vlm_owner_scoped_stats_and_reset_pass_through(monkeypatch):
    engine, _ = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]
