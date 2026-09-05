"""Engine protocol, backend selection, and the lazy-loading manager."""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from sous.config import SousConfig


@dataclass(frozen=True)
class Delta:
    """One streamed piece of a generation, delivered as the engine produces it.

    `output_tokens` counts everything generated so far, this piece included.
    `finish_reason` is None until the final piece, then "stop" (the model ended
    its turn) or "length" (max_tokens was reached). The final piece may carry
    empty text — the detokenizer's flush — so an empty delta is not "nothing
    happened".
    """

    text: str
    output_tokens: int
    finish_reason: str | None = None


# Called on the generating thread, inside the decode loop: it must return
# quickly and must never raise — an exception here fails the generation.
# Wrapping a callback in ReplaySafe below marks it replay-safe: a warm-cache
# failure partway through may still be retried cold.
OnDelta = Callable[[Delta], None]


class ReplaySafe:
    """An on_delta whose output never leaves the process (accounting only),
    so a failed warm attempt may still be retried cold: nothing was sent
    that a re-run would send twice."""

    __slots__ = ("fn",)

    def __init__(self, fn: OnDelta) -> None:
        self.fn = fn

    def __call__(self, delta: Delta) -> None:
        self.fn(delta)


class Engine(Protocol):
    # Read-only: ManagedEngine exposes model_id as a property delegating to its
    # inner engine. A plain `model_id: str` attribute would demand a settable
    # one and exclude it from the protocol.
    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str: ...
    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int: ...
    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None: ...
    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict: ...
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


def measure_cache_budget(reserve_bytes: int) -> int:
    """The automatic resident-slot budget, read once the weights are loaded so
    `active` is the weights (drafter included). Deliberately no
    release_mlx_thread_state() here: this runs on whichever thread loaded the
    engine, and that thread releases on its own schedule — a release from
    inside would destroy streams the caller still uses (#34).

    Runs inside both engine constructors, so it must never raise: an mlx API
    change here would otherwise brick delegation and the gateway together, on
    the shipped default. Every other reader of these numbers degrades instead
    (decide_context, live_headroom), and so does this one — to a single slot.
    """
    try:
        import mlx.core as mx

        from sous.engine.promptcache import auto_cache_budget

        info = mx.device_info()
        return auto_cache_budget(
            working_set=int(info["max_recommended_working_set_size"]),
            active=mx.get_active_memory(),
            reserve_bytes=reserve_bytes,
        )
    except Exception as e:  # noqa: BLE001 — an optimization, never a failure
        warnings.warn(
            f"sous: could not measure the prompt-cache budget ({type(e).__name__}); "
            "keeping a single slot",
            stacklevel=2,
        )
        return 0


def live_headroom() -> int | None:
    """Bytes a cache could still take without paging: the tighter of Metal's
    working set minus what mlx holds and available RAM plus mlx's reclaimable
    buffer cache (sous.context.auto_context_tokens' definition). Read on the
    owner thread mid-turn; same no-release rule as measure_cache_budget. None
    when the numbers are unavailable — the caller then skips its check."""
    try:
        import mlx.core as mx
        import psutil

        info = mx.device_info()
        metal = int(info["max_recommended_working_set_size"]) - mx.get_active_memory()
        system = psutil.virtual_memory().available + mx.get_cache_memory()
        return max(0, min(metal, system))
    except Exception:  # noqa: BLE001 — mlx absent or API moved; a check we skip, not a failure
        return None


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
    draft_id: str = "",
    draft_block_size: int = 0,
    cache_budget: int | None = None,
    reserve_tokens: int = 0,
) -> Engine:
    from sous.context import kv_bytes_per_token

    model_config = fetch_model_config(model_id)
    backend = select_backend(model_config)
    # One full window of KV, kept free for the in-flight turn before slots get
    # anything (the spec's "reserved out of the generation budget"). Unknown
    # per-token cost means the reserve cannot be sized, so the automatic
    # budget degrades to a single slot — never to an unbounded one.
    bytes_per_token = kv_bytes_per_token(model_config)
    reserve_bytes = reserve_tokens * bytes_per_token if bytes_per_token is not None else 0
    if bytes_per_token is None:
        # No reserve means the pressure check can never fire (headroom is
        # never below zero), so say so however the budget was set: with an
        # explicit one, the cap becomes the only thing bounding the map.
        auto = cache_budget is None
        warnings.warn(
            f"sous: KV cost per token unknown for {model_id}; the prompt cache's "
            "memory-pressure check is disabled"
            + (
                " and a single prompt-cache slot is kept (set [model].prompt_cache_gb to override)"
                if auto
                else ""
            ),
            stacklevel=2,
        )
        if auto:
            cache_budget = 0
    if backend == "vlm":
        # Import the module, not the class, so tests can monkeypatch the
        # engine class on its home module and be seen here.
        from sous.engine import vlm

        return vlm.VLMEngine(
            model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt_cache=prompt_cache,
            draft_id=draft_id,
            draft_block_size=draft_block_size,
            cache_budget=cache_budget,
            reserve_bytes=reserve_bytes,
        )
    # The drafter settings stop here: speculative decoding is an mlx-vlm
    # feature, and the mlx-lm backend has no parameter for it.
    from sous.engine import lm

    return lm.LMEngine(
        model_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prompt_cache=prompt_cache,
        cache_budget=cache_budget,
        reserve_bytes=reserve_bytes,
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

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        with self._gen_lock:
            return self._inner.generate(messages, tools, max_tokens, on_delta)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self._inner.count_tokens(messages, tools)

    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        # No _gen_lock, on purpose. An abandoned stalled generation still holds
        # it, and run_task calls this in a finally — waiting there would wedge
        # the next task. What actually makes a lock-free reset safe against
        # that thread's late write-back is not the epoch guard by itself: a
        # slot is always published together with the exact token ids it
        # contains, and reuse_length demands a full strict-prefix match, so a
        # stale slot adopted by a later task is either rejected outright or
        # genuinely correct for it. The epoch (and, per owner, retirement) is
        # only a cheap early-out on top of that — it skips the adoption, it
        # doesn't guarantee it.
        self._inner.reset_prompt_cache(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._inner.prompt_cache_stats(owner)

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
    thread per task lets turn N+1 reuse turn N's cache. The cache slot itself
    now records which thread built it, so a session on any other thread gets
    a cold miss instead of touching those arrays.

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
    reset belongs to the thread that owns the session: the worker thread for
    tasks, the gateway's turn thread after a stall (`sous.gateway.turn`).

    on_delta, when given, fires on this thread from inside the engine's decode
    loop — mid-generation, under _gen_lock. A stalled-and-abandoned generation
    keeps firing it until it ends, so a consumer must tolerate deltas that
    arrive after generate() has already raised GenerationStalled.
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

    @property
    def thread(self) -> threading.Thread:
        """The thread every generation of this session runs on — the owner
        of every prompt-cache slot those generations publish."""
        return self._thread

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
                # A parked thread must not pin either end of the exchange: an
                # ("err", e) reply holds the whole generation frame — KV cache
                # included — through the traceback, and a gateway request holds
                # its on_delta closure and through it the client's queue and
                # event loop, for as long as the thread waits for the next one.
                del reply, req
        finally:
            release_mlx_thread_state()

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        timeout: float,
        on_delta: OnDelta | None = None,
    ) -> str:
        assert not self._closed and not self._abandoned.is_set(), (
            "session reused after close() or a stall"
        )
        self._requests.put_nowait((messages, tools, max_tokens, on_delta))
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

    def join(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the session thread to exit after
        close(). Never blocks indefinitely: a wedged generation (see the
        class docstring) never dequeues _CLOSE, and the caller — an app
        shutdown hook — must not hang on that. Returns whether the thread
        actually exited."""
        self._thread.join(timeout)
        return not self._thread.is_alive()


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
                draft_id=config.speculative_draft_id,
                draft_block_size=config.speculative_block_size,
                cache_budget=(
                    None
                    if config.prompt_cache_gb is None
                    else int(config.prompt_cache_gb * (1 << 30))
                ),
                # The largest cache one turn can build on this daemon: the
                # gateway's window when it is on, else the worker's.
                reserve_tokens=max(
                    config.max_context_tokens,
                    config.gateway_max_context_tokens if config.gateway_enabled else 0,
                ),
            )
        )
        self._lock = threading.Lock()
        self._engine: ManagedEngine | None = None
        self._last_used: float | None = None
        self._leases = 0

    def get(self) -> ManagedEngine:
        with self._lock:
            if self._engine is None:
                self._engine = ManagedEngine(self._factory(self._config.model_id))
            self._last_used = time.monotonic()
            return self._engine

    def touch(self) -> None:
        with self._lock:
            self._last_used = time.monotonic()

    @contextlib.contextmanager
    def lease(self):
        """Pin the loaded engine for the caller's whole span of use.

        _gen_lock only covers generate(). A gateway turn holds the engine from
        get() through count_tokens() — seconds on a large prompt — before
        anything takes that lock, and the idle sweep runs on a different
        thread (the worker's loop), so it could free the weights in between.
        The worker never needed one: it sweeps on the same thread that runs
        its tasks, serially.
        """
        with self._lock:
            self._leases += 1
        try:
            yield
        finally:
            with self._lock:
                self._leases -= 1

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._engine is None or self._last_used is None:
                return False
            if self._engine.generation_in_flight() or self._leases:
                # Never free the model weights under an active (possibly
                # abandoned-as-stalled) generation, nor under a caller that is
                # holding this engine across calls that take no _gen_lock.
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
            out = {
                "loaded": self._engine is not None,
                "model_id": self._config.model_id,
                "idle_seconds": idle,
            }
            if self._engine is not None:
                # Counts and byte totals only; never a token id.
                out["prompt_cache"] = self._engine.prompt_cache_stats()
            return out
