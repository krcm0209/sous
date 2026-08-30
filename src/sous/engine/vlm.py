"""Multimodal backend via mlx-vlm, used text-only in v1."""

from __future__ import annotations

import warnings
from typing import Any, cast

from sous.engine.promptcache import PrefixCache, PromptMemo


def _load_quantized_drafter(model: object, draft_id: str) -> tuple[Any, str]:
    """Download, quantize, and validate a speculative drafter for `model`.

    Raises on any problem — the caller degrades to running without one."""
    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import snapshot_download
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

    drafter, kind = cast("tuple[Any, str]", load_drafter(snapshot_download(draft_id)))
    # Published DFlash checkpoints ship bf16; left unquantized the drafter
    # costs more per round than speculation saves (measured in #58: a bf16
    # drafter regresses throughput below the no-drafter baseline).
    nn.quantize(drafter, group_size=64, bits=4)
    mx.eval(drafter.parameters())
    validate_drafter_compatibility(model, drafter, kind)
    return drafter, kind


class VLMEngine:
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        prompt_cache: bool = False,
        draft_id: str = "",
        draft_block_size: int = 0,
    ):
        from mlx_vlm import load
        from mlx_vlm.sample_utils import make_sampler

        self.model_id = model_id
        self._model, self._processor = load(model_id)
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._cache = PrefixCache(self, enabled=prompt_cache)
        self._draft = None
        self._draft_kind = ""
        self._draft_block_size = draft_block_size
        if draft_id:
            try:
                self._draft, self._draft_kind = _load_quantized_drafter(self._model, draft_id)
            except Exception as e:  # noqa: BLE001 — degrade, never block the model
                warnings.warn(
                    f"sous: speculative drafter {draft_id!r} unavailable for"
                    f" {model_id} ({e}); generating without it",
                    stacklevel=2,
                )

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

    def _ids(self, slot: str, messages: list[dict], tools: list[dict]) -> list[int]:
        text = self._prompt(messages, tools, generation=slot == "full")
        cached = self._memo.get(slot, text)
        if cached is not None:
            return cached
        ids = self._encode(text)
        self._memo.put(slot, text, ids)
        return ids

    # ---- CacheHooks ------------------------------------------------------

    def new_cache(self) -> list:
        from mlx_vlm.models.cache import make_prompt_cache

        model, _ = self._loaded()
        # The nested language model, never the wrapper: many wrappers expose a
        # `layers` property but no `make_cache`, so passing the wrapper builds
        # all-plain KVCache and loses the model's real cache layout.
        return make_prompt_cache(model.language_model)

    def prefill(self, cache: list, token_ids: list[int]) -> None:
        import mlx.core as mx
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
        import mlx.core as mx
        from mlx_vlm import generate

        model, processor = self._loaded()
        # Speculative decoding rides on the decode call only: prefill has no
        # tokens to draft, and generate_step captures the hidden states the
        # drafter needs during its own prefill of these input_ids. block size
        # 0 means None — let the drafter's own policy pick the depth.
        draft_kwargs = (
            {
                "draft_model": self._draft,
                "draft_kind": self._draft_kind,
                "draft_block_size": self._draft_block_size or None,
            }
            if self._draft is not None
            else {}
        )
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
            **draft_kwargs,
        )
        # mlx-vlm returns GenerationResult in recent versions; older return str
        return result.text if hasattr(result, "text") else str(result)

    def copy_array(self, a: object) -> object:
        import mlx.core as mx

        # promptcache is deliberately mlx-free and so types cache entries as
        # `object`; everything that reaches here is an mx.array.
        return mx.array(cast("mx.array", a))

    # ---- Engine ----------------------------------------------------------

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        full_ids = self._ids("full", messages, tools)
        # The stable render is only an anchor for reuse, and PrefixCache discards
        # it when disabled — so computing it would cost a whole extra tokenization
        # per turn for nothing.
        stable_ids = self._ids("stable", messages, tools) if self._cache.enabled else []
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

        import mlx.core as mx

        self.reset_prompt_cache()
        self._model = None
        self._processor = None
        self._draft = None
        gc.collect()
        mx.clear_cache()
