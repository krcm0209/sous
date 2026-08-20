"""Multimodal backend via mlx-vlm, used text-only in v1."""

from __future__ import annotations

from sous.engine.promptcache import PrefixCache, PromptMemo


class VLMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = True,
    ):
        from mlx_vlm import load
        from mlx_vlm.sample_utils import make_sampler

        self.model_id = model_id
        self._model, self._processor = load(model_id)
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._cache = PrefixCache(self, enabled=prompt_cache)

    def _loaded(self) -> tuple:
        """The (model, processor) pair, or a clear error if already unloaded.
        Same rationale as LMEngine._loaded."""
        if self._model is None or self._processor is None:
            raise RuntimeError(f"engine for {self.model_id} has been unloaded")
        return self._model, self._processor

    @property
    def _tokenizer(self):
        # Via _loaded() so the _prompt/count_tokens paths raise the same named
        # RuntimeError after unload() — reading _processor directly would hand
        # back None and fail later with a bare AttributeError.
        _, processor = self._loaded()
        return getattr(processor, "tokenizer", processor)

    def _prompt(self, messages: list[dict], tools: list[dict], generation: bool = True) -> str:
        # enable_thinking=False: same rationale as LMEngine. Confirmed inert
        # (no-op) for templates such as Qwen2-VL's that don't define the
        # variable — verified empirically, not assumed.
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=generation,
            tokenize=False,
            enable_thinking=False,
        )

    def _encode(self, text: str) -> list[int]:
        from mlx_vlm.utils import should_add_special_tokens

        model, processor = self._loaded()
        # Parity with prepare_inputs' text-only path, which is what mlx-vlm
        # itself would tokenize this prompt with. A mismatch would not fail —
        # it would silently miss on every turn.
        add_special = should_add_special_tokens(model.config.model_type, processor)
        return list(self._tokenizer.encode(text, add_special_tokens=add_special))

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> tuple[str, list[int]]:
        text = self._prompt(messages, tools, generation=slot == "full")
        cached = self._memo.get(slot, text)
        if cached is not None:
            return text, cached
        ids = self._encode(text)
        self._memo.put(slot, text, ids)
        return text, ids

    # ---- CacheHooks ------------------------------------------------------

    def new_cache(self) -> list:
        from mlx_vlm.models.cache import make_prompt_cache

        model, _ = self._loaded()
        # The nested language model, never the wrapper: many wrappers expose a
        # `layers` property but no `make_cache`, so passing the wrapper builds
        # all-plain KVCache and loses the model's real cache layout.
        return make_prompt_cache(model.language_model)

    def prefill(self, cache: list, token_ids: list[int]) -> None:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from mlx_vlm import generate

        model, processor = self._loaded()
        if not token_ids:
            return
        # max_tokens=0 is prefill-only: dispatch has an explicit
        # `if not generated_tokens:` branch that yields a result and returns
        # without touching any cache state.
        generate(
            model,
            processor,
            "",
            max_tokens=0,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
        )

    def decode(self, cache: list, token_ids: list[int], max_tokens: int) -> str:
        import mlx.core as mx  # ty: ignore[unresolved-import]
        from mlx_vlm import generate

        model, processor = self._loaded()
        # prompt_cache plus input_ids, not prompt_cache_state. mlx-vlm primes
        # Qwen mRoPE state before feeding a suffix, and that priming turns out
        # to be bit-identical to no priming for text-only prompts — so sous
        # owns the cache outright rather than driving mlx-vlm's reuse path.
        result = generate(
            model,
            processor,
            "",
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
        )
        # mlx-vlm returns GenerationResult in recent versions; older return str
        return result.text if hasattr(result, "text") else str(result)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return mx.array(a)

    # ---- Engine ----------------------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        _, stable_ids = self._ids("stable", messages, tools)
        _, full_ids = self._ids("full", messages, tools)
        return self._cache.generate(stable_ids, full_ids, max_tokens)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._ids("full", messages, tools)[1])

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
        self._processor = None
        gc.collect()
        mx.clear_cache()
