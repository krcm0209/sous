"""Multimodal backend via mlx-vlm, used text-only in v1."""

from __future__ import annotations


class VLMEngine:
    def __init__(self, model_id: str):
        from mlx_vlm import load

        self.model_id = model_id
        self._model, self._processor = load(model_id)

    @property
    def _tokenizer(self):
        return getattr(self._processor, "tokenizer", self._processor)

    def _prompt(self, messages: list[dict], tools: list[dict]) -> str:
        # enable_thinking=False: same rationale as LMEngine. Confirmed inert
        # (no-op) for templates such as Qwen2-VL's that don't define the
        # variable — verified empirically, not assumed.
        return self._tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        from mlx_vlm import generate

        result = generate(
            self._model, self._processor,
            self._prompt(messages, tools),
            max_tokens=max_tokens, verbose=False,
        )
        # mlx-vlm returns GenerationResult in recent versions; older return str
        return result.text if hasattr(result, "text") else str(result)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self._tokenizer.encode(self._prompt(messages, tools)))

    def unload(self) -> None:
        import gc

        import mlx.core as mx

        self._model = None
        self._processor = None
        gc.collect()
        mx.clear_cache()
