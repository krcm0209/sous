"""The engines' fork boundary: the system turn rendered alone, verified as a
token prefix of the whole render — exercised without mlx by faking the
tokenizer, the way tests/test_engine_unloaded.py builds engines."""

import threading

from sous.engine.lm import LMEngine
from sous.engine.promptcache import FORK_MIN_TOKENS, PromptMemo
from sous.engine.vlm import VLMEngine


class FakeTokenizer:
    """Renders messages as a fixed-width token per character so a header that
    is a text prefix is also a token prefix, and vice versa."""

    bos_token = None

    def __init__(self, header_chars: int = FORK_MIN_TOKENS):
        self.header_chars = header_chars

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        out = []
        for m in messages:
            if m["role"] == "system":
                out.append("S" * self.header_chars)
            else:
                out.append(m["content"])
        if add_generation_prompt:
            out.append("G")
        return "|".join(out)

    def encode(self, text, add_special_tokens):
        return [ord(c) for c in text]


class Recording:
    """A PrefixCache stand-in recording the fork_at it was handed."""

    enabled = True

    def __init__(self):
        self.fork_ats: list[int] = []

    def generate(self, stable_ids, full_ids, max_tokens, on_delta=None, fork_at=0):
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
    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        text = super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )
        # A template that changes the system turn once a user turn follows.
        return text.replace("S", "s", 1) if len(messages) > 1 else text


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


def test_a_leading_system_turn_long_enough_is_the_fork_point():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [FORK_MIN_TOKENS]  # the header render's length


def test_a_short_system_turn_does_not_fork():
    engine, rec = _engine(FakeTokenizer(header_chars=10))
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_no_system_turn_means_no_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([USER], [], 8)
    assert rec.fork_ats == [0]


def test_a_system_only_prompt_does_not_fork():
    # Nothing would follow the header, but _fork_at's `len(messages) < 2`
    # guard returns first: fork_point's strict-prefix rule is never asked.
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [0]


def test_a_template_whose_header_is_not_a_token_prefix_does_not_fork():
    engine, rec = _engine(Rewriting())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_the_header_render_is_skipped_when_the_cache_is_disabled():
    engine, rec = _engine(FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]
    assert engine._memo.get("header", "S" * FORK_MIN_TOKENS) is None


def test_owner_scoped_stats_and_reset_pass_through():
    engine, _ = _engine(FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]


# ---- the same seven, for the VLM backend ----------------------------------
#
# _encode is monkeypatched away because its real body imports mlx_vlm; the
# fake tokenizer's own encode() is what the LM tests exercise, and this keeps
# the file runnable on a machine without mlx.


def _mlx_free_vlm(monkeypatch, tokenizer) -> tuple[VLMEngine, Recording]:
    monkeypatch.setattr(VLMEngine, "_encode", lambda self, text: [ord(c) for c in text])
    return _vlm_engine(tokenizer)


def test_vlm_a_leading_system_turn_long_enough_is_the_fork_point(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [FORK_MIN_TOKENS]


def test_vlm_a_short_system_turn_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer(header_chars=10))
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_no_system_turn_means_no_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([USER], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_a_system_only_prompt_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_a_template_whose_header_is_not_a_token_prefix_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Rewriting())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]


def test_vlm_the_header_render_is_skipped_when_the_cache_is_disabled(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [0]
    assert engine._memo.get("header", "S" * FORK_MIN_TOKENS) is None


def test_vlm_owner_scoped_stats_and_reset_pass_through(monkeypatch):
    engine, _ = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]
