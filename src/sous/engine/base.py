"""Engine protocol, backend selection, and the lazy-loading manager."""

from __future__ import annotations

import contextlib
import json
import queue
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
        import mlx.core as mx

        mx.clear_streams()
    except Exception:  # noqa: BLE001 — see docstring
        pass


class GenerationStalled(Exception):
    """A generation produced no reply within its deadline."""


_CLOSE = object()  # session shutdown sentinel; ends the loop, carries no reset


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
        # the next task. What actually makes a lock-free reset safe against
        # that thread's late write-back is not the epoch guard by itself: the
        # cache is always published together with the exact token ids it
        # contains, and reuse_length demands a full strict-prefix match, so a
        # stale cache adopted by a later task is either rejected outright or
        # genuinely correct for it. The epoch is only a cheap early-out on
        # top of that — it skips the adoption, it doesn't guarantee it.
        self._inner.reset_prompt_cache()

    def prompt_cache_stats(self) -> dict:
        return self._inner.prompt_cache_stats()

    def generation_in_flight(self) -> bool:
        return self._gen_lock.locked()

    def session(self) -> GenerationSession:
        """One task's generation thread; run_task creates one per task."""
        return GenerationSession(self)

    def unload(self) -> None:
        self._inner.unload()


class GenerationSession:
    """One task's generations, all on one daemon thread (issue #34).

    mlx KV cache arrays are usable only from the thread whose streams created
    them (streams are thread-scoped for use — probed empirically, see the
    design spec), and every thread that touched mlx must call
    release_mlx_thread_state() before it exits (ml-explore/mlx#4327). A fresh
    thread per generation therefore killed the prompt cache every turn; one
    thread per task lets turn N+1 reuse turn N's cache.

    The loop re-checks `_abandoned` while it HOLDS _gen_lock: a request whose
    task gave up while still queued on the lock exits instead of running
    under the next task's identity (issue #34, consideration 7).

    close() sends _CLOSE and never joins. A healthy or abandoned-but-idle
    thread dequeues it, releases its mlx state, and exits — this is also what
    un-leaks a thread whose reply lost the timeout race, so nothing it pinned
    (worst case an ("err", e) traceback holding the KV cache) outlives the
    task. A wedged thread never dequeues it and is leaked deliberately, like
    the abandoned per-generation threads before this class: it never exits,
    so it never hits the TLS-teardown segfault, and _gen_lock keeps the next
    task off the engine meanwhile. _CLOSE deliberately carries no cache
    reset — a late reset from a stale session thread would race the next
    task's cache and stats, the same class of bug as consideration 7. Every
    reset belongs to the worker thread.
    """

    def __init__(self, managed: ManagedEngine):
        self._managed = managed
        # maxsize=1 plus put_nowait everywhere: at most one request is ever
        # outstanding, so Full in generate() means a protocol bug — failing
        # loudly beats deadlocking the worker inside run_task's finally.
        # close() alone tolerates Full: a stalled request the starved thread
        # never dequeued may still occupy the queue.
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._replies: queue.Queue = queue.Queue(maxsize=1)
        self._abandoned = threading.Event()
        self._closed = False
        # Kept as an attribute so tests can join it; production never joins —
        # a wedged generation must not block task teardown.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                req = self._requests.get()
                if req is _CLOSE:
                    return
                with self._managed._gen_lock:
                    if self._abandoned.is_set():
                        return
                    try:
                        reply = ("ok", self._managed._inner.generate(*req))
                    except BaseException as e:  # noqa: BLE001 — relayed to the caller
                        reply = ("err", e)
                if self._abandoned.is_set():
                    return
                self._replies.put_nowait(reply)
                # A parked thread must not pin the reply: an ("err", e) entry
                # holds the whole generation frame — KV cache included —
                # through the traceback.
                del reply
        finally:
            release_mlx_thread_state()

    def generate(
        self, messages: list[dict], tools: list[dict], max_tokens: int, timeout: float
    ) -> str:
        assert not self._closed and not self._abandoned.is_set(), (
            "session reused after close() or a stall"
        )
        self._requests.put_nowait((messages, tools, max_tokens))
        try:
            kind, value = self._replies.get(timeout=timeout)
        except queue.Empty:
            self._abandoned.set()
            raise GenerationStalled(f"generation stalled (> {round(timeout, 1)}s)") from None
        if kind == "err":
            raise value
        return value

    def close(self) -> None:
        """End the session thread; never joined (see the class docstring)."""
        if self._closed:
            return
        self._closed = True
        # Full means a timed-out request is still queued: the session thread
        # lost the scheduler race that stalled it and never dequeued it.
        # _CLOSE is unnecessary then — the thread will dequeue that request,
        # see _abandoned under the lock, and exit.
        with contextlib.suppress(queue.Full):
            self._requests.put_nowait(_CLOSE)


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

    def get(self) -> ManagedEngine:
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
