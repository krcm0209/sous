"""Text-only backend via mlx-lm."""

from __future__ import annotations

from sous.engine.promptcache import PrefixCache, PromptMemo


class LMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = False,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = model_id
        # mlx-lm ships no type stubs, so the (model, tokenizer) arity of load()
        # is not visible to the type checker.
        self._model, self._tokenizer = load(model_id)  # ty: ignore[invalid-assignment]
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._cache = PrefixCache(self, enabled=prompt_cache)

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

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> list[int]:
        """Tokenize a render once per turn. The two slots are the stable render
        and the full prompt; count_tokens and generate both read them."""
        _, tokenizer = self._loaded()
        text = self._prompt(messages, tools, generation=slot == "full")
        cached = self._memo.get(slot, text)
        if cached is not None:
            return cached
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
        bos = getattr(tokenizer, "bos_token", None)
        add_special = bos is None or not text.startswith(bos)
        ids = list(tokenizer.encode(text, add_special_tokens=add_special))
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
        # extra. Only the non-trimmable path reaches here.
        import mlx.core as mx  # ty: ignore[unresolved-import]

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

    def decode(self, cache: list, token_ids: list[int], max_tokens: int) -> str:
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
        return "".join(chunks)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return mx.array(a)

    # ---- Engine ----------------------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        stable_ids = self._ids("stable", messages, tools)
        full_ids = self._ids("full", messages, tools)
        return self._cache.generate(stable_ids, full_ids, max_tokens)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._ids("full", messages, tools))

    def reset_prompt_cache(self) -> None:
        self._cache.reset()
        self._memo.clear()

    def prompt_cache_stats(self) -> dict:
        return self._cache.stats()

    def unload(self) -> None:
        import gc

        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        self.reset_prompt_cache()
        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
