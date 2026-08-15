"""Text-only backend via mlx-lm."""

from __future__ import annotations


class LMEngine:
    def __init__(self, model_id: str, temperature: float = 0.7,
                top_p: float = 0.8, top_k: int = 20):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = model_id
        self._model, self._tokenizer = load(model_id)
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

    def _prompt(self, messages: list[dict], tools: list[dict]) -> str:
        # enable_thinking=False: sous delegates mechanical prep, not reasoning —
        # a "thinking" model must not spend its turn budget on <think> chain-of-
        # thought instead of emitting the tool call. Inert on templates that
        # don't define the variable (e.g. plain non-thinking models).
        return self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate

        return generate(
            self._model, self._tokenizer,
            prompt=self._prompt(messages, tools),
            max_tokens=max_tokens, sampler=self._sampler, verbose=False,
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
