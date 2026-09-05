"""Text-only backend via mlx-lm."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import cast

from sous.engine.base import Delta, OnDelta
from sous.engine.promptcache import FORK_MIN_TOKENS, PrefixCache, PromptMemo, fork_point

# Two conversations that differ only in the first user turn's content, so that
# what their renders have in common is exactly the header — see _header_probe.
_PROBES = ({"role": "user", "content": "0"}, {"role": "user", "content": "1"})


class LMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = False,
        cache_budget: int | None = None,
        reserve_bytes: int = 0,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        from sous.engine.base import measure_cache_budget

        self.model_id = model_id
        # mlx-lm ships no type stubs, so the (model, tokenizer) arity of load()
        # is not visible to the type checker.
        self._model, self._tokenizer = load(model_id)  # ty: ignore[invalid-assignment]
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._tokenize_lock = threading.Lock()
        # Measured after load, so the weights are inside `active`.
        if cache_budget is None:
            cache_budget = measure_cache_budget(reserve_bytes)
        self._cache = PrefixCache(
            self, enabled=prompt_cache, max_bytes=cache_budget, reserve_bytes=reserve_bytes
        )

    def _loaded(self) -> tuple:
        """The (model, tokenizer) pair, or a clear error if already unloaded.

        EngineManager drops its reference right after unload(), so a call here
        on an unloaded engine should be unreachable — but raising beats the
        bare AttributeError on None that the attributes would otherwise give."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"engine for {self.model_id} has been unloaded")
        return self._model, self._tokenizer

    def _prompt(self, messages: list[dict], tools: list[dict], generation: bool = True) -> str:
        # enable_thinking=False: sous delegates mechanical prep, not reasoning —
        # a "thinking" model must not spend its turn budget on <think> chain-of-
        # thought instead of emitting the tool call. Inert on templates that
        # don't define the variable (e.g. plain non-thinking models).
        _, tokenizer = self._loaded()
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=generation,
            tokenize=False,
            enable_thinking=False,
        )

    def _encode(self, text: str) -> list[int]:
        # mlx_lm.generate's stream_generate (mlx_lm/generate.py:691-694) only
        # adds special tokens when the tokenizer has no bos_token, or the
        # prompt doesn't already start with it — because the chat template
        # usually emits BOS itself. Before this cache existed, sous handed
        # stream_generate a string and got this for free; encoding ids
        # ourselves has to replicate the same rule explicitly, mirroring
        # VLMEngine._encode's should_add_special_tokens for the same model.
        # A mismatch would not fail loudly — it would silently duplicate BOS
        # on every turn for any model whose template already emits it (e.g.
        # Llama-3, Gemma, Mistral MLX conversions).
        _, tokenizer = self._loaded()
        bos = getattr(tokenizer, "bos_token", None)
        add_special = bos is None or not text.startswith(bos)
        return list(tokenizer.encode(text, add_special_tokens=add_special))

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> list[int]:
        """Tokenize a render once per turn, for the stable and full slots;
        count_tokens and generate both read them. The memo's third slot, the
        header, is filled by _header_probe, which builds its text rather than
        rendering a message list."""
        # One lock for every tokenization: HF's fast tokenizer mutates shared
        # Rust state on each encode (set_truncation_and_padding), and since the
        # gateway there are two callers — a turn on a pool thread and Claude
        # Code's count_tokens, which it sends mid-turn. Not _gen_lock: that
        # would queue a token count behind a whole generation.
        with self._tokenize_lock:
            text = self._prompt(messages, tools, generation=slot == "full")
            cached = self._memo.get(slot, text)
            if cached is not None:
                return cached
            ids = self._encode(text)
            self._memo.put(slot, text, ids)
            return ids

    # ---- CacheHooks ------------------------------------------------------

    def new_cache(self) -> list:
        from mlx_lm.models.cache import make_prompt_cache

        model, _ = self._loaded()
        return make_prompt_cache(model)

    def prefill(self, cache: list, token_ids: list[int]) -> None:
        # mlx-lm has no prefill-only entry point: stream_generate raises on
        # max_tokens=0 because its `token` local is unbound when the loop never
        # runs. Calling the model directly is what generate_step does anyway,
        # and RoPE offsets come from the cache, so a warm suffix needs nothing
        # extra. The non-trimmable path and any turn that forks at the header
        # reach here; the trimmable path otherwise fuses prefill into decode.
        import mlx.core as mx

        model, _ = self._loaded()
        if not token_ids:
            return
        # Chunked at generate_step's own prefill_step_size (mlx_lm/generate.py:316
        # defaults to 2048). One unchunked call would materialise attention over
        # the whole delta at once, which is exactly the peak the spec promises not
        # to move.
        step = 2048
        for i in range(0, len(token_ids), step):
            model(mx.array(token_ids[i : i + step])[None], cache=cache)
            mx.eval([c.state for c in cache])

    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None = None
    ) -> str:
        from mlx_lm import stream_generate

        model, tokenizer = self._loaded()
        chunks: list[str] = []
        for r in stream_generate(
            model,
            tokenizer,
            token_ids,
            max_tokens=max_tokens,
            sampler=self._sampler,
            prompt_cache=cache,
        ):
            chunks.append(r.text)
            if on_delta is not None:
                on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
        return "".join(chunks)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx

        # promptcache is deliberately mlx-free and so types cache entries as
        # `object`; everything that reaches here is an mx.array.
        return mx.array(cast("mx.array", a))

    def headroom(self) -> int | None:
        # Imported inside the method so a test can monkeypatch
        # sous.engine.base.live_headroom and see it take effect here.
        from sous.engine.base import live_headroom

        return live_headroom()

    # ---- Engine ----------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        full_ids = self._ids("full", messages, tools)
        if not self._cache.enabled:
            # The stable render is only an anchor for reuse, and PrefixCache
            # discards it when disabled — so computing it (and the header)
            # would cost extra tokenizations per turn for nothing.
            return self._cache.generate([], full_ids, max_tokens, on_delta)
        stable_ids = self._ids("stable", messages, tools)
        return self._cache.generate(
            stable_ids,
            full_ids,
            max_tokens,
            on_delta,
            fork_at=self._header_probe(messages, tools, stable_ids),
        )

    def _header_probe(
        self, messages: list[dict], tools: list[dict], stable_ids: list[int]
    ) -> Callable[[], int] | int:
        """The header boundary — everything the template emits above the first
        user turn's content — as a closure PrefixCache resolves only when this
        turn could actually fork.

        Found from two probe conversations rather than from the system turn
        rendered alone: [model].id accepts any chat template, and a template
        may refuse a message list with no user turn outright (the default
        model's does: `raise_exception("No user query found in messages.")`).
        Two renders that differ only in the first user turn's content share
        exactly the header, whatever the template puts there. fork_point still
        verifies per turn that those ids really are a token prefix of this
        render, and that the header clears the fork floor.

        Reuse needs the rendered header to match token for token, which in
        practice means subagents of one type inside one Claude Code session:
        a new `claude` process puts its own session scratchpad path above the
        tool schemas, so its header diverges after ~1K tokens. The worker's
        short system prompt never qualifies, and a render below the floor
        cannot contain a header above it — so it never pays the probe."""
        if (
            len(messages) < 2
            or messages[0].get("role") != "system"
            or len(stable_ids) < FORK_MIN_TOKENS
        ):
            return 0

        def probe() -> int:
            with self._tokenize_lock:
                renders = [
                    self._prompt([messages[0], user], tools, generation=False) for user in _PROBES
                ]
                header = os.path.commonprefix(renders)
                # Memoized like the other renders: the header is the longest
                # thing in the prompt, and a conversation re-probes it on every
                # cold turn.
                header_ids = self._memo.get("header", header)
                if header_ids is None:
                    header_ids = self._encode(header)
                    self._memo.put("header", header, header_ids)
            return fork_point(header_ids, stable_ids)

        return probe

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._ids("full", messages, tools))

    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self._cache.reset(owner)
        # The memo is text-keyed and shared by every caller, so an owner-scoped
        # reset leaves it alone; only the drop-everything form (unload) clears it.
        if owner is None:
            self._memo.clear()

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._cache.stats(owner)

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self.reset_prompt_cache()
        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
