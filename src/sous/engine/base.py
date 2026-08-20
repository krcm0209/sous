"""Engine protocol, backend selection, and the lazy-loading manager."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Protocol

from sous.config import SousConfig


class Engine(Protocol):
    # Read-only: ManagedEngine exposes model_id as a property delegating to its
    # inner engine. A plain `model_id: str` attribute would demand a settable
    # one and exclude it from the protocol.
    @property
    def model_id(self) -> str: ...

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str: ...
    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int: ...
    def reset_prompt_cache(self) -> None: ...
    def prompt_cache_stats(self) -> dict: ...
    def unload(self) -> None: ...


def release_mlx_thread_state() -> None:
    """Destroy this thread's mlx streams before the thread exits.

    mlx >= 0.32.1 no longer cleans up per-thread state automatically (PR
    #4248 removed the GIL-acquiring thread_local guard), and the maintainers'
    contract — ml-explore/mlx#4327 — is that every thread that touched mlx
    calls mx.clear_streams() before exiting. A thread that skips it segfaults
    the ENTIRE process in the dyld TLS finalizer (CompileCache teardown
    reaching _Py_Dealloc without the GIL); observed killing the daemon
    mid-task. A no-op when mlx is absent or the thread never touched it —
    cleanup must never raise out of a dying thread.
    """
    try:
        # mlx.core is a compiled extension with no type stubs.
        import mlx.core as mx  # ty: ignore[unresolved-import]

        mx.clear_streams()
    except Exception:  # noqa: BLE001 — see docstring
        pass


def select_backend(model_config: dict) -> str:
    if "vision_config" in model_config:
        return "vlm"
    if "vl" in model_config.get("model_type", "").lower():
        return "vlm"
    return "lm"


def fetch_model_config(model_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "config.json")
    with open(path) as f:
        return json.load(f)


def _default_factory(
    model_id: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    prompt_cache: bool = True,
) -> Engine:
    backend = select_backend(fetch_model_config(model_id))
    if backend == "vlm":
        from sous.engine.vlm import VLMEngine

        return VLMEngine(
            model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt_cache=prompt_cache,
        )
    from sous.engine.lm import LMEngine

    return LMEngine(
        model_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prompt_cache=prompt_cache,
    )


class ManagedEngine:
    """Serializes generations on one engine instance. MLX generation is
    synchronous and uninterruptible: on a stall the worker abandons its
    generation thread, but that thread is still USING the engine, so a second
    concurrent generation (or an unload) on the same model would corrupt
    inference. The lock makes the next task wait for the stalled generation
    instead. Consequence: a truly wedged generation delays subsequent tasks
    until the daemon is restarted — process isolation is the future fix (see
    README limitations)."""

    def __init__(self, inner: Engine):
        self._inner = inner
        self._gen_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        with self._gen_lock:
            return self._inner.generate(messages, tools, max_tokens)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self._inner.count_tokens(messages, tools)

    def reset_prompt_cache(self) -> None:
        # No _gen_lock, on purpose. An abandoned stalled generation still holds
        # it, and run_task calls this in a finally — waiting there would wedge
        # the next task. PrefixCache's epoch guard is what makes a lock-free
        # reset safe against that thread's late write-back.
        self._inner.reset_prompt_cache()

    def prompt_cache_stats(self) -> dict:
        return self._inner.prompt_cache_stats()

    def generation_in_flight(self) -> bool:
        return self._gen_lock.locked()

    def unload(self) -> None:
        self._inner.unload()


class EngineManager:
    def __init__(self, config: SousConfig, engine_factory: Callable[[str], Engine] | None = None):
        self._config = config
        self._factory = engine_factory or (
            lambda model_id: _default_factory(
                model_id,
                config.temperature,
                config.top_p,
                config.top_k,
                config.prompt_cache,
            )
        )
        self._lock = threading.Lock()
        self._engine: ManagedEngine | None = None
        self._last_used: float | None = None

    def get(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = ManagedEngine(self._factory(self._config.model_id))
            self._last_used = time.monotonic()
            return self._engine

    def touch(self) -> None:
        with self._lock:
            self._last_used = time.monotonic()

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._engine is None or self._last_used is None:
                return False
            if self._engine.generation_in_flight():
                # Never free the model weights under an active (possibly
                # abandoned-as-stalled) generation.
                return False
            idle = time.monotonic() - self._last_used
            if idle > self._config.idle_unload_minutes * 60:
                self._engine.unload()
                self._engine = None
                return True
            return False

    def status(self) -> dict:
        with self._lock:
            idle = (time.monotonic() - self._last_used) if self._last_used else None
            return {
                "loaded": self._engine is not None,
                "model_id": self._config.model_id,
                "idle_seconds": idle,
            }
