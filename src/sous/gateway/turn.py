"""One gateway turn on the shared engine: serialized, thread-bridged, drained.

The daemon has one engine and one generation lock; the worker and the gateway
share both. A turn takes the gateway's own lock first (so gateway turns queue
in order and never find the one-slot GenerationSession busy), then the
engine's lock through the session, exactly as run_task does. Everything here
is synchronous and runs on whatever pool thread the route hands it; progress
crosses back to the event loop through the Sink.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from sous.config import SousConfig
from sous.engine.base import (
    Delta,
    EngineManager,
    GenerationSession,
    GenerationStalled,
    ManagedEngine,
    ReplaySafe,
    release_mlx_thread_state,
)


class Sink(Protocol):
    """Where a turn reports progress. Neither method runs on the event loop —
    `started` on the turn's thread, `delta` on the engine's session thread
    mid-decode — and neither may block or raise."""

    def started(self, input_tokens: int) -> None: ...
    def delta(self, delta: Delta) -> None: ...

    # True when `delta()` forwards nothing outside the process (accounting
    # only) — a non-streaming turn's sink, whose deltas the client never
    # sees, so a warm-cache failure may still be retried cold.
    replay_safe: bool


class PromptTooLong(Exception):
    def __init__(self, tokens: int, window: int):
        super().__init__(f"prompt is too long: {tokens} tokens > {window} maximum")
        self.tokens = tokens
        self.window = window


class GatewayBusy(Exception):
    """The gateway lock was not acquired within the turn timeout."""


class TurnAbandoned(Exception):
    """The client left while the turn was still queued for the gateway lock."""


@dataclass(frozen=True)
class TurnResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    cache_hit: bool
    reused_tokens: int
    seconds: float


class TurnRunner:
    def __init__(self, engines: EngineManager, config: SousConfig):
        self._engines = engines
        self._window = config.gateway_max_context_tokens
        self._timeout = float(config.gateway_generation_timeout_minutes * 60)
        self._lock = threading.Lock()
        # One long-lived session for every gateway turn: the prompt cache
        # lives on the session thread's mlx streams (#34), so a per-request
        # session would throw the cache away between a subagent's turns — and
        # the cache is what turns gate 2's ~200s cold prefill into seconds.
        self._session: GenerationSession | None = None
        self._session_engine: ManagedEngine | None = None

    def run(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sink: Sink,
        abandoned: threading.Event | None = None,
    ) -> TurnResult:
        if not self._lock.acquire(timeout=self._timeout):
            raise GatewayBusy(f"no generation slot within {self._timeout:.0f}s")
        started = time.monotonic()
        try:
            # The idle sweep runs on the worker's thread, and _gen_lock only
            # covers generate(): without a lease the model could be unloaded
            # under count_tokens(), which is seconds of work on a long prompt.
            with self._engines.lease():
                if abandoned is not None and abandoned.is_set():
                    # Queued behind another turn, and the client gave up meanwhile:
                    # a generation nobody reads would only delay the live requests
                    # behind it. (A turn that has started still drains — the GPU
                    # cannot be interrupted and the lock discipline depends on it.)
                    raise TurnAbandoned
                engine = self._engines.get()
                session = self._session_for(engine)
                input_tokens = engine.count_tokens(messages, tools)
                room = self._window - input_tokens
                if room <= 0:
                    raise PromptTooLong(input_tokens, self._window)
                # Hit/miss is for the log only: the counters are global, and a
                # worker task resetting the cache mid-turn zeroes them, so a hit
                # can read as a miss. Exact per-turn reuse comes with keyed slots.
                before = engine.prompt_cache_stats()
                sink.started(input_tokens)
                final: Delta | None = None

                def on_delta(delta: Delta) -> None:
                    nonlocal final
                    final = delta
                    sink.delta(delta)

                # ReplaySafe tells the prompt cache this callback's output
                # never reaches a client, so a warm-cache failure may still
                # be retried cold — true exactly when the sink itself is.
                callback = ReplaySafe(on_delta) if sink.replay_safe else on_delta

                try:
                    text = session.generate(
                        messages,
                        tools,
                        min(max_tokens, room),
                        timeout=self._timeout,
                        on_delta=callback,
                    )
                except GenerationStalled:
                    # The session is unusable after a stall (its thread may still
                    # be generating, holding the engine lock); the next turn gets a
                    # fresh one and waits on the lock like the worker would. Reset
                    # the cache too: when the abandoned thread finishes it publishes
                    # the KV cache it built on ITS streams, and a cache is usable
                    # only from the thread that built it (#34). The reset's epoch
                    # bump makes that late publish drop itself — the same guard
                    # run_task's finally relies on.
                    self._drop_session()
                    # Best-effort: a reset that raises would replace GenerationStalled
                    # with a generic 500 and lose the stall's classification.
                    with contextlib.suppress(Exception):
                        engine.reset_prompt_cache()
                    raise
                after = engine.prompt_cache_stats()
                return TurnResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=final.output_tokens if final else 0,
                    finish_reason=final.finish_reason if final else "stop",
                    cache_hit=after.get("hits", 0) > before.get("hits", 0),
                    reused_tokens=max(
                        0, after.get("reused_tokens", 0) - before.get("reused_tokens", 0)
                    ),
                    seconds=time.monotonic() - started,
                )
        finally:
            self._engines.touch()
            self._lock.release()
            # engines.get() may have loaded the model on this thread. Pool
            # threads outlive the call, but the invariant is per thread that
            # touched mlx (ml-explore/mlx#4327), and keeping it unconditional
            # is what makes it checkable.
            release_mlx_thread_state()

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        try:
            # Same race as run(): this whole call happens outside _gen_lock.
            with self._engines.lease():
                count = self._engines.get().count_tokens(messages, tools)
            self._engines.touch()
            return count
        finally:
            release_mlx_thread_state()

    def _session_for(self, engine: ManagedEngine) -> GenerationSession:
        if self._session is None or self._session_engine is not engine:
            # A different ManagedEngine means the model was idle-unloaded and
            # reloaded; the old session's thread would call into an engine
            # whose weights are gone.
            self._drop_session()
            self._session = engine.session()
            self._session_engine = engine
        return self._session

    def _drop_session(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._session_engine = None

    def close(self) -> None:
        with self._lock:
            self._drop_session()
