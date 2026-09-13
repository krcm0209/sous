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
    forked: bool = False  # the hit was served by copying a fork slot
    # On a miss: how many leading tokens the render shared with the closest
    # slot the session held (0 when it held none). Where two Claude Code
    # renders diverged — inside the tool block or after it — without a token
    # of either reaching the log. Always 0 on a hit.
    lcp: int = 0
    # Where the seconds went, and what the cache did — every one a count or a
    # duration; formatted by the gateway's turn line.
    prefilled_tokens: int = 0  # stable tokens the cache had to prefill
    took_len: int = 0  # length of the slot that served the turn; 0 on a miss
    forks: int = 0  # fork slots this turn published
    evictions: int = 0  # slots dropped while it ran, charged to this session
    pressure_evictions: int = 0  # of which by the pressure valve
    load_seconds: float = 0.0  # lease + engines.get(): a model load, ours or one we waited out
    queue_seconds: float = 0.0  # wait for the gateway lock, before `seconds` starts
    # Wait for the engine's own lock, which a delegated task's generation
    # holds; inside ttft_s and `seconds`, and in no phase below.
    engine_wait_seconds: float = 0.0
    tokenize_seconds: float = 0.0  # count_tokens plus the fork probe's renders
    ttft_seconds: float | None = None  # generate() entry → first delta; None if none came
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    bounds: tuple[int, int] = (0, 0)  # the probe's (tools, header) boundaries; 0 = absent


@dataclass(frozen=True)
class CountResult:
    count: int
    # A count can be the request that loads the model — seconds of work the
    # client sees as latency and the log must attribute.
    load_seconds: float
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
        # Set once close() has given up on acquiring _lock from a turn in
        # flight: that turn's own finally then drops the session for it
        # (see run()'s finally), and any turn still queued on the lock must
        # refuse to start rather than outlive a gateway that gave up on it.
        self._closing = False

    def run(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sink: Sink,
        abandoned: threading.Event | None = None,
    ) -> TurnResult:
        queued = time.monotonic()
        if not self._lock.acquire(timeout=self._timeout):
            raise GatewayBusy(f"no generation slot within {self._timeout:.0f}s")
        if self._closing:
            # Queued behind another turn while close() was giving up on it;
            # that other turn's finally already dropped (or will drop) the
            # session, so this one must not start a new generation on a
            # gateway that has already been told to shut down.
            self._lock.release()
            raise GatewayBusy("gateway is shutting down")
        started = time.monotonic()
        queue_seconds = started - queued
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
                # From `started`, not from here: entering the lease waits out
                # a load or unload another thread is in the middle of, which
                # costs this turn exactly what loading the model itself would.
                load_seconds = time.monotonic() - started
                session = self._session_for(engine)
                counting = time.monotonic()
                input_tokens = engine.count_tokens(messages, tools)
                tokenize_seconds = time.monotonic() - counting
                room = self._window - input_tokens
                if room <= 0:
                    raise PromptTooLong(input_tokens, self._window)
                # Owner-scoped, so the before/after delta is exact: only this
                # session's thread moves these counters, and the turn holds
                # the gateway lock, so nothing else's hit or reset can land
                # between the two reads.
                before = engine.prompt_cache_stats(owner=session.thread)
                sink.started(input_tokens)
                final: Delta | None = None
                first_delta_at: float | None = None

                def on_delta(delta: Delta) -> None:
                    nonlocal final, first_delta_at
                    if first_delta_at is None:
                        first_delta_at = time.monotonic()
                    final = delta
                    sink.delta(delta)

                # ReplaySafe tells the prompt cache this callback's output
                # never reaches a client, so a warm-cache failure may still
                # be retried cold — true exactly when the sink itself is.
                callback = ReplaySafe(on_delta) if sink.replay_safe else on_delta

                generating = time.monotonic()
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
                    # fresh one and waits on the lock like the worker would. Retire
                    # the stalled session's thread too: when the abandoned thread
                    # finishes it would publish the KV cache it built on ITS
                    # streams, and a cache is usable only from the thread that
                    # built it (#34). Retirement makes that late publish drop
                    # itself, and leaves the worker's slots alone.
                    stalled = session.thread
                    self._drop_session()
                    # Best-effort: a reset that raises would replace GenerationStalled
                    # with a generic 500 and lose the stall's classification.
                    with contextlib.suppress(Exception):
                        engine.reset_prompt_cache(owner=stalled)
                    raise
                after = engine.prompt_cache_stats(owner=session.thread)
                cache_hit = after.get("hits", 0) > before.get("hits", 0)

                def delta_of(key: str) -> int:
                    return max(0, after.get(key, 0) - before.get(key, 0))

                return TurnResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=final.output_tokens if final else 0,
                    finish_reason=final.finish_reason if final else "stop",
                    cache_hit=cache_hit,
                    forked=after.get("fork_hits", 0) > before.get("fork_hits", 0),
                    reused_tokens=delta_of("reused_tokens"),
                    # miss_lcp is a gauge the engine assigns per miss, so
                    # `after` holds this turn's value exactly when this turn
                    # missed — and a stale one from an earlier miss otherwise.
                    lcp=0 if cache_hit else after.get("miss_lcp", 0),
                    seconds=time.monotonic() - started,
                    # Gauges: assigned per turn by the cache, read directly like miss_lcp.
                    prefilled_tokens=after.get("prefilled_tokens", 0),
                    took_len=after.get("took_len", 0),
                    forks=delta_of("forks"),
                    evictions=delta_of("evictions"),
                    pressure_evictions=delta_of("pressure_evictions"),
                    load_seconds=load_seconds,
                    queue_seconds=queue_seconds,
                    engine_wait_seconds=session.lock_wait_seconds,
                    tokenize_seconds=tokenize_seconds + after.get("probe_seconds", 0.0),
                    ttft_seconds=None if first_delta_at is None else first_delta_at - generating,
                    prefill_seconds=after.get("prefill_seconds", 0.0),
                    decode_seconds=after.get("decode_seconds", 0.0),
                    bounds=(after.get("bound_lo", 0), after.get("bound_hi", 0)),
                )
        finally:
            self._engines.touch()
            if self._closing:
                # close() gave up on this turn's lock and returned False;
                # nothing else will ever retry the drop, so the turn that
                # held the lock does it here, while it still owns it. The
                # session's own thread dequeues _CLOSE and exits on its own
                # after this — joining it would make a turn thread wait on
                # shutdown work, which it must never do.
                self._drop_session()
            self._lock.release()
            # engines.get() may have loaded the model on this thread. Pool
            # threads outlive the call, but the invariant is per thread that
            # touched mlx (ml-explore/mlx#4327), and keeping it unconditional
            # is what makes it checkable.
            release_mlx_thread_state()

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> CountResult:
        started = time.monotonic()
        try:
            # Same race as run(): this whole call happens outside _gen_lock.
            with self._engines.lease():
                engine = self._engines.get()
                load_seconds = time.monotonic() - started  # the lease wait too, as in run()
                count = engine.count_tokens(messages, tools)
            self._engines.touch()
            return CountResult(count, load_seconds, time.monotonic() - started)
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

    def close(self, timeout: float = 2.0) -> bool:
        """Best-effort shutdown: drop the session and wait up to `timeout`
        seconds for its thread to actually exit. Returns False, touching
        nothing else, if the lock is not free within `timeout` — a turn in
        flight (or a wedged one still holding it from a stalled generation,
        see GenerationSession's docstring) must never hold up app shutdown.
        `_closing` is set first, unconditionally: when the lock is busy, the
        turn holding it sees the flag in its own finally and drops the
        session itself once it finishes, so shutdown mid-turn no longer
        leaks that session's thread. Never blocks longer than `timeout` in
        total."""
        self._closing = True
        deadline = time.monotonic() + timeout
        if not self._lock.acquire(timeout=timeout):
            return False
        try:
            session = self._session
            self._drop_session()
            if session is not None:
                session.join(max(0.0, deadline - time.monotonic()))
            return True
        finally:
            self._lock.release()
