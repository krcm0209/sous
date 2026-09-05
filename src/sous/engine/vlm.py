"""Multimodal backend via mlx-vlm, used text-only in v1."""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Callable
from typing import Any, cast

from sous.engine.base import Delta, OnDelta
from sous.engine.promptcache import FORK_MIN_TOKENS, PrefixCache, PromptMemo, fork_point

# Two conversations that differ only in the first user turn's content, so that
# what their renders have in common is exactly the header — see _header_probe.
_PROBES = ({"role": "user", "content": "0"}, {"role": "user", "content": "1"})


def _load_quantized_drafter(model: object, draft_id: str) -> tuple[Any, str]:
    """Download, quantize, and validate a speculative drafter for `model`.

    Raises on any problem — the caller degrades to running without one."""
    import mlx.core as mx
    import mlx.nn as nn
    from huggingface_hub import snapshot_download
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

    # lazy=True: load_drafter's default evaluates every bf16 parameter on the
    # spot (~4 GB peak). Deferring keeps the weights unevaluated through
    # validation, and on the happy path quantize consumes the lazy loads so
    # the final eval materializes only the 4-bit copy.
    drafter, kind = cast("tuple[Any, str]", load_drafter(snapshot_download(draft_id), lazy=True))
    # Validate BEFORE quantize+eval: for an incompatible target (a swapped
    # [model].id) this fails while the weights are still lazy, so the mismatch
    # costs a config check instead of materializing ~4 GB of bf16 first.
    validate_drafter_compatibility(model, drafter, kind)
    # Published DFlash checkpoints ship bf16; left unquantized the drafter
    # costs more per round than speculation saves (measured in #58: a bf16
    # drafter regresses throughput below the no-drafter baseline).
    nn.quantize(drafter, group_size=64, bits=4)
    mx.eval(drafter.parameters())
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
        cache_budget: int | None = None,
        reserve_bytes: int = 0,
    ):
        from mlx_vlm import load
        from mlx_vlm.sample_utils import make_sampler

        from sous.engine.base import measure_cache_budget

        self.model_id = model_id
        self._model, self._processor = load(model_id)
        self._sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self._memo = PromptMemo()
        self._tokenize_lock = threading.Lock()
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
        # Measured after load AND after the drafter, so the weights of both are
        # inside `active` and the budget is what the machine actually has left.
        if cache_budget is None:
            cache_budget = measure_cache_budget(reserve_bytes)
        self._cache = PrefixCache(
            self, enabled=prompt_cache, max_bytes=cache_budget, reserve_bytes=reserve_bytes
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

    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None = None
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import stream_generate

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
        # generate() resets the tokenizer's shared stopping criteria before
        # every call and stream_generate does not; mirror it so a criteria
        # left mutated by another caller cannot change where this turn stops.
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)
        chunks: list[str] = []
        # prompt_cache plus input_ids, not prompt_cache_state. mlx-vlm primes
        # Qwen mRoPE state before feeding a suffix, and that priming turns out
        # to be bit-identical to no priming for text-only prompts — so sous
        # owns the cache outright rather than driving mlx-vlm's reuse path.
        for r in stream_generate(
            model,
            processor,
            "",
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
            **draft_kwargs,
        ):
            # Draft rows are the speculator's proposals, not accepted output;
            # generate() skips them the same way.
            if r.is_draft:
                continue
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
        self._processor = None
        self._draft = None
        gc.collect()
        mx.clear_cache()
