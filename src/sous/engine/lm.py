"""Text-only backend via mlx-lm."""

from __future__ import annotations


class LMEngine:
    def __init__(
        self, model_id: str, temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = model_id
        # mlx-lm ships no type stubs, so the (model, tokenizer) arity of load()
        # is not visible to the type checker.
        self._model, self._tokenizer = load(model_id)  # ty: ignore[invalid-assignment]
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

    def _loaded(self) -> tuple:
        """The (model, tokenizer) pair, or a clear error if already unloaded.

        EngineManager drops its reference right after unload(), so a call here
        on an unloaded engine should be unreachable — but raising beats the
        bare AttributeError on None that the attributes would otherwise give."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"engine for {self.model_id} has been unloaded")
        return self._model, self._tokenizer

    def _prompt(self, messages: list[dict], tools: list[dict]) -> str:
        # enable_thinking=False: sous delegates mechanical prep, not reasoning —
        # a "thinking" model must not spend its turn budget on <think> chain-of-
        # thought instead of emitting the tool call. Inert on templates that
        # don't define the variable (e.g. plain non-thinking models).
        _, tokenizer = self._loaded()
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate

        model, tokenizer = self._loaded()
        return generate(
            model,
            tokenizer,
            prompt=self._prompt(messages, tools),
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
        )

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        _, tokenizer = self._loaded()
        return len(tokenizer.encode(self._prompt(messages, tools)))

    def unload(self) -> None:
        import gc

        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
