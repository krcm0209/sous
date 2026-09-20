"""Engine protocol, backend selection, and the lazy-loading manager."""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import sys
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from sous.config import SousConfig

_logger = logging.getLogger("sous.engine")

# The idle sweep's cadence. A tick is a lock, a clock comparison and one psutil
# check per holder, so a slower one saves nothing worth having; all it has to be
# is well under idle_unload_minutes, which is minutes.
IDLE_SWEEP_SECONDS = 15.0


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
    mid-generation. A no-op when mlx is absent or the thread never touched it —
    cleanup must never raise out of a dying thread.
    """
    try:
        import mlx.core as mx

        mx.clear_streams()
    except Exception:  # noqa: BLE001 — see docstring
        pass


def _holder_alive(pid: int, create_time: float) -> bool:
    """Whether the process that registered a hold is still the one wearing
    `pid`: alive, not a zombie, and started within a second of the time it
    reported. A reused pid fails the start-time test. A zombie — exited,
    its parent yet to reap it — still has a start time psutil can read, so
    it is refused by status; a vanished process, a pid the kernel could
    never have issued or one psutil cannot read all raise, and count as gone
    rather than pin the model forever. The second of slack is defensive
    only: `sous claude` posts the same psutil reading the daemon takes."""
    import psutil

    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        return abs(process.create_time() - create_time) <= 1.0
    except psutil.Error, ValueError, TypeError:
        return False


def measure_cache_budget(reserve_bytes: int) -> int:
    """The automatic resident-slot budget, read once the weights are loaded so
    `active` is the weights (drafter included). Deliberately no
    release_mlx_thread_state() here: this runs on whichever thread loaded the
    engine, and that thread releases on its own schedule — a release from
    inside would destroy streams the caller still uses (#34).

    Runs inside both engine constructors, so it must never raise: an mlx API
    change here would otherwise brick every model load on the shipped
    default. Every other reader of these numbers degrades instead
    (live_headroom), and so does this one — to a single slot.
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
    """Bytes a cache could still take without Metal paging this process's own
    buffers: the device's recommended working set minus what mlx holds
    (`promptcache.metal_headroom`). Process-local by design — the machine-wide
    question goes to `kernel_memory_pressure`. Read on the owner thread
    mid-turn; same no-release rule as measure_cache_budget. None when the
    numbers are unavailable — the caller then skips its check."""
    try:
        import mlx.core as mx

        from sous.engine.promptcache import metal_headroom

        info = mx.device_info()
        return metal_headroom(
            working_set=int(info["max_recommended_working_set_size"]),
            active=mx.get_active_memory(),
        )
    except Exception:  # noqa: BLE001 — mlx absent or API moved; a check we skip, not a failure
        return None


def kernel_memory_pressure() -> int | None:
    """macOS's own memory-pressure classifier, `kern.memorystatus_vm_pressure_level`:
    1 normal, 2 warn, 4 critical. It already discounts purgeable file cache —
    the thing psutil's `available` counted as used right after a model load —
    and it is the signal the kernel itself acts on when it starts compressing
    and swapping. None anywhere it cannot be read (not macOS, sysctl missing),
    and the valve then stays out of the way. A libc call, no subprocess."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
        return value.value if rc == 0 else None
    except Exception:  # noqa: BLE001 — a reading we skip, never a failure
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
    int8_prefill: bool = False,
) -> Engine:
    from sous.engine.window import kv_bytes_per_token

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
            int8_prefill=int8_prefill,
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
        int8_prefill=int8_prefill,
    )


def default_engine_factory(config: SousConfig) -> Callable[[str], Engine]:
    """The factory EngineManager builds when given none: every [model] value
    mapped onto the backend. Public so a process that must wrap the real
    engine (the tune's suite runner counts its deltas) builds the same one
    rather than a second copy of this mapping."""
    return lambda model_id: _default_factory(
        model_id,
        config.temperature,
        config.top_p,
        config.top_k,
        config.prompt_cache,
        draft_id=config.speculative_draft_id,
        draft_block_size=config.speculative_block_size,
        cache_budget=(
            None if config.prompt_cache_gb is None else int(config.prompt_cache_gb * (1 << 30))
        ),
        reserve_tokens=config.max_context_tokens,
        int8_prefill=config.int8_prefill,
    )


class ManagedEngine:
    """Serializes generations on one engine instance. MLX generation is
    synchronous and uninterruptible: on a stall the caller (the endpoint's
    turn runner, a tune run) abandons its generation thread, but that thread
    is still USING the engine, so a second concurrent generation (or an
    unload) on the same model would corrupt inference. The lock makes the
    next turn wait for the stalled generation
    instead. Consequence: a truly wedged generation delays subsequent turns
    until the daemon is restarted — process isolation is the future fix (see
    README limitations)."""

    def __init__(self, inner: Engine):
        self._inner = inner
        self._gen_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def int8_prefill_status(self) -> dict | None:
        # Optional on purpose: fakes and older engines have no such attribute.
        return getattr(self._inner, "int8_prefill_status", None)

    @property
    def positions(self) -> str | None:
        # Optional on purpose: only the VLM backend sets this; the LM backend
        # has no such attribute.
        return getattr(self._inner, "positions", None)

    @property
    def drafter(self) -> str | None:
        """The speculative drafter the engine actually runs with: its id, ""
        for none, or None from a backend that has no notion of one (the LM
        backend, fakes). A VLM engine survives a drafter that fails to load
        by running without it, so a caller that must know the difference —
        the bench, whose row would otherwise carry a drafter's label over
        undrafted numbers — reads this rather than its own request."""
        return getattr(self._inner, "drafter", None)

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
        # it, and the turn runner calls this right after a stall — waiting
        # there would wedge the next turn. What makes a lock-free reset safe against
        # that thread's late write-back is not the epoch guard by itself: a
        # slot is always published together with the exact token ids it
        # contains, and reuse_length demands a full strict-prefix match, so a
        # stale slot adopted by a later turn is either rejected outright or
        # genuinely correct for it. The epoch (and, per owner, retirement) is
        # only a cheap early-out on top of that — it skips the adoption, it
        # doesn't guarantee it.
        self._inner.reset_prompt_cache(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._inner.prompt_cache_stats(owner)

    def generation_in_flight(self) -> bool:
        return self._gen_lock.locked()

    def session(self) -> GenerationSession:
        """A generation thread of the caller's own; the endpoint keeps one alive
        across turns so their caches survive."""
        return GenerationSession(self)

    def unload(self) -> None:
        self._inner.unload()


class GenerationSession:
    """One caller's generations, all on one daemon thread (issue #34).

    mlx KV cache arrays are usable only from the thread whose streams created
    them (streams are thread-scoped for use — probed empirically), and every
    thread that touched mlx must call release_mlx_thread_state() before it
    exits (ml-explore/mlx#4327). A fresh
    thread per generation therefore killed the prompt cache every turn; one
    thread per session lets turn N+1 reuse turn N's cache. The cache slot itself
    now records which thread built it, so a session on any other thread gets
    a cold miss instead of touching those arrays.

    The loop re-checks `_abandoned` while it HOLDS _gen_lock: a request whose
    turn gave up while still queued on the lock exits instead of running
    under the next turn's identity (issue #34, consideration 7).

    close() sends _CLOSE and never joins. A healthy or abandoned-but-idle
    thread dequeues it, releases its mlx state, and exits — this is also what
    un-leaks a thread whose reply lost the timeout race, so nothing it pinned
    (worst case an ("err", e) traceback holding the KV cache) outlives the
    turn. A wedged thread never dequeues it and is leaked deliberately, like
    the abandoned per-generation threads before this class: it never exits,
    so it never hits the TLS-teardown segfault, and _gen_lock keeps the next
    turn off the engine meanwhile. _CLOSE deliberately carries no cache
    reset — a late reset from a stale session thread would race the next
    turn's cache and stats, the same class of bug as consideration 7. Every
    reset belongs to the thread that owns the session: the endpoint's turn
    thread after a stall (`sous.api.turn`), the tune suite's thread at the
    end of every run (`sous.tune.suite.loop`).

    on_delta, when given, fires on this thread from inside the engine's decode
    loop — mid-generation, under _gen_lock. A stalled-and-abandoned generation
    keeps firing it until it ends, so a consumer must tolerate deltas that
    arrive after generate() has already raised GenerationStalled.
    """

    def __init__(self, managed: ManagedEngine):
        self._managed = managed
        # maxsize=1 plus put_nowait everywhere: at most one request is ever
        # outstanding, so Full in generate() means a protocol bug — failing
        # loudly beats deadlocking the session's owner (the endpoint's turn
        # runner, a tune run) on its way out.
        # close() alone tolerates Full: a stalled request the starved thread
        # never dequeued may still occupy the queue.
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._replies: queue.Queue = queue.Queue(maxsize=1)
        self._abandoned = threading.Event()
        self._closed = False
        # How long the latest request waited for _gen_lock — behind an
        # abandoned stalled generation, say. Written before the reply is queued, so the
        # caller reads it once generate() returns; nothing else can see it.
        self.lock_wait_seconds = 0.0
        # Kept as an attribute so tests can join it; production never joins —
        # a wedged generation must not block the turn's teardown.
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
                waiting = time.monotonic()
                with self._managed._gen_lock:
                    self.lock_wait_seconds = time.monotonic() - waiting
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
                # included — through the traceback, and an endpoint request holds
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
    def __init__(
        self,
        config: SousConfig,
        engine_factory: Callable[[str], Engine] | None = None,
        *,
        holder_alive: Callable[[int, float], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._factory = engine_factory or default_engine_factory(config)
        self._lock = threading.Lock()
        # A load or an unload is never done under _lock (minutes for a 27B):
        # get() and unload_if_idle() mark which is in progress and wait on
        # this for it to finish, so status() and every other short call on
        # the lock answer meanwhile.
        self._changed = threading.Condition(self._lock)
        self._engine: ManagedEngine | None = None
        self._loading = False
        self._unloading = False
        self._last_used: float | None = None
        self._leases = 0
        # Processes that pin the model: pid -> the start time it reported.
        # `sous claude` registers itself and then execs Claude Code, which
        # keeps the pid, so the holder IS that session; the sweep drops a
        # holder whose process is gone.
        self._holders: dict[int, float] = {}
        self._holder_alive: Callable[[int, float], bool] = holder_alive or _holder_alive
        self._preload: threading.Thread | None = None
        self._clock = clock
        # Bumped on the transitions the status document shows and the
        # in-flight registry cannot: a load starting or ending, an unload
        # starting or ending, a hold taken, a holder pruned, and every
        # reset of the idle clock — the terminal runs that clock itself
        # between documents, so a reset it never hears about is a wrong
        # number on screen until the next one.
        self._version = 0
        self._sweep: threading.Thread | None = None
        self._sweep_stop: threading.Event | None = None

    @property
    def version(self) -> int:
        """Read without the lock — one int; see Inflight.version for why a
        poller reads it before, not after, the snapshot it describes."""
        return self._version

    def _bump(self) -> None:
        """Lock held by the caller."""
        self._version += 1

    def get(self) -> ManagedEngine:
        with self._changed:
            while self._loading or self._unloading:
                self._changed.wait()
            if self._engine is not None:
                self._last_used = self._clock()
                self._bump()
                return self._engine
            self._loading = True
            self._bump()
        loading = time.monotonic()
        try:
            engine = ManagedEngine(self._load())
        except BaseException:
            with self._changed:
                self._loading = False
                self._bump()
                self._changed.notify_all()
            raise
        with self._changed:
            self._engine = engine
            self._loading = False
            self._last_used = self._clock()
            self._bump()
            self._changed.notify_all()
        # The one line that brackets a cold start in the daemon log —
        # before it, only huggingface_hub's own chatter said a load
        # happened, and a turn's `seconds` could not be split.
        line = f"model_load seconds={time.monotonic() - loading:.1f} model={engine.model_id}"
        if engine.positions is not None:
            line += f" positions={engine.positions}"
        _logger.info(line)
        return engine

    def _load(self) -> Engine:
        """Run the factory on a thread of its own and hand back what it built.

        A load touches mlx, and a thread that touched mlx must release its
        streams before it exits (ml-explore/mlx#4327) — a release after which
        that thread cannot run another op that needs a stream, a load
        included. The endpoint's turn and count pools
        call get() from threads that outlive the call and release
        unconditionally, so a load on one of them left it unable to load a
        second time: after an idle unload, the next cold start on that same
        pool thread failed with "There is no Stream(gpu, 0) in current
        thread", and the pool reuses its idle thread first, so every cold
        start after the first did. A thread that exists only for the load is
        the one shape the invariant fits: it releases on its way out, and the
        caller never touches mlx at all."""
        outcome: list = []

        def run() -> None:
            try:
                outcome.append(self._factory(self._config.model_id))
            except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
                outcome.append(exc)
            finally:
                release_mlx_thread_state()

        loader = threading.Thread(target=run, name="sous-model-load", daemon=True)
        loader.start()
        loader.join()
        if isinstance(outcome[0], BaseException):
            raise outcome[0]
        return outcome[0]

    def touch(self) -> None:
        with self._lock:
            self._last_used = self._clock()
            self._bump()

    @contextlib.contextmanager
    def lease(self):
        """Pin the loaded engine for the caller's whole span of use.

        _gen_lock only covers generate(). An endpoint turn holds the engine
        from get() through count_tokens() — seconds on a large prompt —
        before anything takes that lock, and the idle sweep runs on a thread
        of its own (`sous-idle-sweep`, start_idle_sweep()), so it could free
        the weights in between.
        """
        with self._lock:
            self._leases += 1
        try:
            yield
        finally:
            with self._lock:
                self._leases -= 1

    def hold(self, pid: int, create_time: float) -> dict:
        """Pin the model for as long as process `pid` (started at
        `create_time`) lives, loading it now if nothing is loaded and no load
        is under way. Never waits for the load and never raises because of
        it: a hold is an optimization, and the first turn's own get() reports
        a real failure."""
        refused = False
        with self._lock:
            self._prune_holders()
            self._holders[pid] = create_time
            self._bump()
            holders = len(self._holders)
            loaded = self._engine is not None
            preloading = not loaded and not self._loading and self._preload is None
            if preloading:
                self._preload = threading.Thread(
                    target=self._run_preload, name="sous-preload", daemon=True
                )
                try:
                    self._preload.start()
                except RuntimeError:
                    # The OS refused a thread. Forget it, or every later hold
                    # would see a preload that never ran and start none.
                    self._preload = None
                    preloading, refused = False, True
            loading = self._load_in_progress()
        _logger.info(f"hold pid={pid} (holders={holders})")
        if preloading:
            _logger.info(f"preloading {self._config.model_id}")
        if refused:
            _logger.warning("preload thread could not start (RuntimeError)")
        return {"loaded": loaded, "loading": loading, "holders": holders}

    def _load_in_progress(self) -> bool:
        """Lock held by the caller. The preload thread forgets itself only
        after its get() has returned, so once get() has published the
        engine `_preload` alone would report a load beside a loaded model."""
        return self._loading or (self._preload is not None and self._engine is None)

    def _run_preload(self) -> None:
        try:
            self.get()
        except Exception as exc:  # noqa: BLE001 — the load's own error surfaces on the first turn
            # The type only: a loader's message can name a path.
            _logger.warning(f"preload failed ({type(exc).__name__})")
        finally:
            # No mlx release here: get() loads on a thread of its own (see
            # _load), which releases its streams before it exits, and this
            # thread touched nothing.
            with self._lock:
                self._preload = None
                # Forgetting the thread is what ends `loading` when the load
                # failed — get()'s own bump left _preload set — so a client
                # that hears no version move keeps painting a dead load. A
                # load that succeeded published the engine and bumped in
                # get() itself, and forgetting the thread then changes
                # nothing a document shows: the engine is None here only
                # after a failure, or once an unload took it back first.
                if self._engine is None:
                    self._bump()

    def _prune_holders(self) -> None:
        """Drop every holder whose process is gone, and when that empties the
        registry restart the idle clock: the last session just left, so its
        idle time starts now and a relaunch inside idle_unload_minutes never
        pays a reload. Lock held by the caller — the psutil check is
        sub-millisecond per holder, and the registry it reads must not
        change under it. Idempotent, so status() and hold() prune too and
        never report a session that has ended; the clock restart lands on
        whichever caller sees the departure first."""
        if not self._holders:
            return
        gone = [
            pid for pid, started in self._holders.items() if not self._holder_alive(pid, started)
        ]
        for pid in gone:
            del self._holders[pid]
            _logger.info(f"hold released pid={pid} (holders={len(self._holders)})")
        if gone:
            self._bump()
        if gone and not self._holders and self._last_used is not None:
            self._last_used = self._clock()

    def _refusal(self) -> str | None:
        """Why the weights cannot be freed right now, or None. Lock held by
        the caller, holders already pruned. The one list the idle sweep and a
        requested unload both consult, so a new pin refuses on both paths.
        A load or an unload in progress is named as such: `_engine` is None
        during both, and "nothing loaded" would send a caller waiting for
        the memory back the wrong way."""
        if self._loading:
            return "a model load is in progress"
        if self._unloading:
            return "an unload is in progress"
        if self._engine is None:
            return "nothing loaded"
        if self._engine.generation_in_flight():
            # Never free the model weights under an active (possibly
            # abandoned-as-stalled) generation.
            return "a generation is in flight"
        if self._leases:
            # Nor under a caller holding this engine across calls that take
            # no _gen_lock.
            return "the engine is leased by a turn"
        if self._holders:
            return f"held by {len(self._holders)} session(s)"
        return None

    def _take(self) -> ManagedEngine:
        """Lock held by the caller, `_refusal()` just answered None: detach
        the engine and mark the unload under way."""
        engine, self._engine = self._engine, None
        assert engine is not None
        self._unloading = True
        self._bump()
        return engine

    def _free(self, engine: ManagedEngine) -> None:
        """Outside the lock — the weights come off the GPU over seconds."""
        try:
            engine.unload()
        finally:
            with self._changed:
                self._unloading = False
                self._bump()
                self._changed.notify_all()

    def unload_if_idle(self) -> bool:
        with self._changed:
            self._prune_holders()
            if self._refusal() is not None or self._last_used is None:
                return False
            idle = self._clock() - self._last_used
            if idle <= self._config.idle_unload_minutes * 60:
                return False
            engine = self._take()
        self._free(engine)
        return True

    def start_idle_sweep(self, interval_s: float = IDLE_SWEEP_SECONDS) -> None:
        """Retire an idle model without anyone asking: one daemon thread,
        `sous-idle-sweep`, calls unload_if_idle() every `interval_s` seconds
        until stop_idle_sweep(). A second start while one runs is a no-op."""
        with self._lock:
            if self._sweep is not None and self._sweep.is_alive():
                return
            stop = threading.Event()
            thread = threading.Thread(
                target=self._run_idle_sweep,
                args=(interval_s, stop),
                name="sous-idle-sweep",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError as e:
                # The OS refused a thread. Publishing it anyway would make
                # every later start a no-op and the stop a join on a thread
                # that never ran; the daemon just goes without the sweep.
                _logger.warning(f"idle sweep not started ({e}); idle models stay loaded")
                return
            self._sweep, self._sweep_stop = thread, stop

    def stop_idle_sweep(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread, stop = self._sweep, self._sweep_stop
        if thread is None or stop is None:
            return
        stop.set()
        thread.join(timeout)
        with self._lock:
            if thread.is_alive():
                # Mid-unload past the bound: left to finish on its own, and
                # kept on the books so a start meanwhile stays a no-op.
                _logger.warning(f"idle sweep still unloading after {timeout:.0f}s; not waited for")
            elif self._sweep is thread:
                self._sweep, self._sweep_stop = None, None

    def _run_idle_sweep(self, interval_s: float, stop: threading.Event) -> None:
        # An unload frees mlx arrays on this thread, and a thread that touched
        # mlx must release its state once before it exits (ml-explore/mlx#4327).
        # This thread never loads, so releasing on the way out is enough.
        try:
            while not stop.wait(interval_s):
                try:
                    self.unload_if_idle()
                except Exception as e:  # noqa: BLE001 — one bad tick must not end the sweep
                    _logger.warning(f"idle sweep failed ({type(e).__name__}: {e})")
        finally:
            release_mlx_thread_state()

    def unload_now(self) -> dict:
        """Free the weights on request — `sous tune` needs the memory — with
        every refusal the idle sweep makes and none of its clock: a session
        holding the model, a leased engine or an in-flight generation each
        keep it, and the caller learns which. Unlike the idle sweep (which
        leaves `_last_used` alone — the model was idle, not un-loaded), this
        path also clears it: the caller asking for the memory back is not a
        use of the model, so `idle_seconds` reads None afterwards rather than
        the clock the sweep would have kept running."""
        with self._changed:
            self._prune_holders()
            reason = self._refusal()
            if reason is not None:
                return {"unloaded": False, "reason": reason}
            engine = self._take()
            self._last_used = None
        self._free(engine)
        return {"unloaded": True, "reason": None}

    def status(self) -> dict:
        with self._lock:
            self._prune_holders()
            idle = (self._clock() - self._last_used) if self._last_used is not None else None
            out = {
                "loaded": self._engine is not None,
                "loading": self._load_in_progress(),
                # The weights come off the GPU over seconds, with memory_gb
                # the only sign of it and no version move until it ends: the
                # event stream keeps its heartbeat for this the way it does
                # for a load.
                "unloading": self._unloading,
                "model_id": self._config.model_id,
                "idle_seconds": idle,
                "holders": len(self._holders),
            }
            if self._engine is not None:
                # promptcache imports this module, so the import cannot be global.
                from sous.engine.promptcache import without_turn_gauges

                # Counts and byte totals only; never a token id.
                out["prompt_cache"] = without_turn_gauges(self._engine.prompt_cache_stats())
                int8 = self._engine.int8_prefill_status
                if int8 is not None:
                    out["int8_prefill"] = dict(int8)
                # Which side supplies the rotary positions behind a warm
                # cache: the load line says it once, this says it for as long
                # as the model is resident.
                if self._engine.positions is not None:
                    out["positions"] = self._engine.positions
            return out
