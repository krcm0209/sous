"""Text-only backend via mlx-lm."""

from __future__ import annotations


class LMEngine:
    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model_id = model_id
        self._model, self._tokenizer = load(model_id)

    def _prompt(self, messages: list[dict], tools: list[dict]) -> str:
        return self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate

        return generate(
            self._model, self._tokenizer,
            prompt=self._prompt(messages, tools),
            max_tokens=max_tokens, verbose=False,
        )

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._tokenizer.encode(self._prompt(messages, tools)))

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
