"""Engine protocol, backend selection, and the lazy-loading manager."""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Protocol

from sous.config import SousConfig


class Engine(Protocol):
    model_id: str

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str: ...
    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int: ...
    def unload(self) -> None: ...


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


def _default_factory(model_id: str) -> Engine:
    backend = select_backend(fetch_model_config(model_id))
    if backend == "vlm":
        from sous.engine.vlm import VLMEngine
        return VLMEngine(model_id)
    from sous.engine.lm import LMEngine
    return LMEngine(model_id)


class EngineManager:
    def __init__(self, config: SousConfig,
                 engine_factory: Callable[[str], Engine] | None = None):
        self._config = config
        self._factory = engine_factory or _default_factory
        self._lock = threading.Lock()
        self._engine: Engine | None = None
        self._last_used: float | None = None

    def get(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = self._factory(self._config.model_id)
            self._last_used = time.monotonic()
            return self._engine

    def touch(self) -> None:
        with self._lock:
            self._last_used = time.monotonic()

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._engine is None or self._last_used is None:
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
