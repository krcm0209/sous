"""Throughput of one arm through sous's own engine: prefill and decode at a
short and a long context, warm-turn TTFT, load time and peak memory, read
from the prompt cache's per-turn gauges the way the endpoint's turn line is."""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sous.engine.base import Delta, EngineManager, ReplaySafe, release_mlx_thread_state
from sous.tune.arms import Arm
from sous.tune.payload import TOOLS

LONG_CONTEXT = 16384
SHORT_CONTEXT = 1024
PREFILL_CONTEXT = 2048
DECODE_TOKENS = 256
# A long-context measurement needs the prompt plus its answers to fit.
_LONG_HEADROOM = 1024
_UNLOAD_WAIT_SECONDS = 30.0
_UNLOAD_SLACK_BYTES = 1 << 30
# A budget-exhausted run abandons its generation, but MLX generation is
# synchronous and uninterruptible (ManagedEngine._gen_lock): the engine's
# session thread keeps decoding until it reaches its token cap, minutes on a
# slow model. The weights cannot be freed before that ends, and stopping the
# whole tune run over one arm's last, still-running generation would discard
# hours of already-measured results — so release() waits it out instead.
_INFLIGHT_WAIT_SECONDS = 900.0
_INFLIGHT_POLL_SECONDS = 2.0
_GIB = 1 << 30
_CONTINUE = "Continue with the next function."
# build_prompt's rough per-function token cost: each round adds enough
# functions to close the gap to target_tokens in one step, so the final
# overshoot is at most one function's worth rather than a doubling.
_TOKENS_PER_FUNCTION = 60

_SYSTEM = (
    "You are a coding assistant. Answer with code only, no prose, continuing the module "
    "in the same style."
)


@dataclass(frozen=True)
class BenchRow:
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    window: int
    ok: bool
    error: str | None
    load_seconds: float | None
    prefill_tps_2k: float | None
    prefill_tps_16k: float | None
    decode_tps_1k: float | None
    decode_tps_16k: float | None
    ttft_seconds: float | None
    peak_memory_bytes: int | None
    spread: float | None
    # False when the arm's teardown could not free the weights (refused or
    # raised): the row still reflects a finished measurement, but the run
    # must stop rather than measure a second arm's numbers with two models
    # resident. Defaults True so a results.jsonl written before this field
    # existed still reads as "released" on --resume.
    released: bool = True

    @property
    def key(self) -> tuple[str, str, int]:
        """The arm this row measured — `Arm.key`, never the label."""
        return (self.model_id, self.drafter_id, self.block_size)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchRow:
        kwargs = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name != "released"}
        kwargs["released"] = d.get("released", True)
        return cls(**kwargs)

    @classmethod
    def failed(cls, arm: Arm, error: str) -> BenchRow:
        return cls(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=False,
            error=error,
            load_seconds=None,
            prefill_tps_2k=None,
            prefill_tps_16k=None,
            decode_tps_1k=None,
            decode_tps_16k=None,
            ttft_seconds=None,
            peak_memory_bytes=None,
            spread=None,
        )

    def with_teardown_error(self, suffix: str) -> BenchRow:
        """The row of a finished measurement whose teardown left the weights
        resident: `ok` stays what the measurement made it, `released` is
        False so the run stops rather than measure a second arm beside
        them, and the report names why."""
        return dataclasses.replace(
            self, error=suffix if self.ok else f"{self.error}; {suffix}", released=False
        )


def _code_lines(rng: random.Random, n: int, start: int) -> list[str]:
    lines: list[str] = []
    for i in range(start, start + n):
        a, b = rng.randint(2, 97), rng.randint(2, 97)
        lines.append(f"def fn_{i}(x: int, y: int = {b}) -> int:")
        lines.append(f'    """Combine x with {a}; y offsets the result."""')
        lines.append(f"    total = x * {a} + y")
        lines.append(f"    return total % {a + b}")
        lines.append("")
    return lines


def build_prompt(
    count: Callable[[list[dict], list[dict]], int], target_tokens: int, *, seed: int = 0
) -> list[dict]:
    """A code-shaped conversation of at least `target_tokens` tokens by the
    engine's own count, grown by an estimate of the remaining deficit each
    round (`_TOKENS_PER_FUNCTION` per generated function) so a 16K prompt
    costs a handful of tokenizations and the final overshoot is at most one
    function, rather than doubling past the target. Deterministic per seed."""
    rng = random.Random(seed)
    lines: list[str] = []

    def render() -> list[dict]:
        body = "Review this module and continue it in the same style.\n\n" + "\n".join(lines)
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": body}]

    fn = 0
    current = count(render(), TOOLS)
    while current < target_tokens:
        deficit = target_tokens - current
        n = max(1, deficit // _TOKENS_PER_FUNCTION)
        lines.extend(_code_lines(rng, n, fn))
        fn += n
        current = count(render(), TOOLS)
    return render()


@dataclass(frozen=True)
class TurnResult:
    text: str
    gauges: dict
    produced: int
    first_delta_seconds: float | None
    last_delta_seconds: float | None


class _Turn:
    """One generate() through a session, with the gauges read on that
    session's thread right after — they are per-turn readings, never
    counter differences. Also times the first and last delta relative to
    generate()'s own entry, for _prefill_rate/_decode_rate's fallback."""

    def __init__(self, engine, session, timeout: float):
        self.engine, self.session, self.timeout = engine, session, timeout

    def run(self, messages: list[dict], max_tokens: int) -> TurnResult:
        first: list[float] = []
        last: list[float] = [0.0]
        tokens: list[int] = [0]
        retried = [False]
        started = time.monotonic()

        def on_delta(d: Delta) -> None:
            now = time.monotonic() - started
            # output_tokens counts from 1 and never falls within one attempt
            # (the detokenizer's final flush repeats the last count, so equal
            # is not a fall); a count that fell is the prompt cache's cold
            # retry starting over, whose start this clock never saw — its
            # timings are void, the gauges (reset per attempt) still stand.
            if d.output_tokens < tokens[0]:
                retried[0] = True
            if not first:
                first.append(now)
            last[0] = now
            tokens[0] = d.output_tokens

        # Accounting only — nothing this callback sees leaves the process —
        # so a warm attempt that fails may still be retried cold.
        text = self.session.generate(
            messages, TOOLS, max_tokens, self.timeout, ReplaySafe(on_delta)
        )
        gauges = self.engine.prompt_cache_stats(owner=self.session.thread)
        timed = bool(first) and not retried[0]
        return TurnResult(
            text=text,
            gauges=gauges,
            produced=tokens[0],
            first_delta_seconds=first[0] if timed else None,
            last_delta_seconds=last[0] if timed else None,
        )


def _rate(tokens: float | None, seconds: float | None) -> float | None:
    if not tokens or not seconds or seconds <= 0:
        return None
    return tokens / seconds


# On the text-only mlx-lm backend, once every layer's cache is trimmable,
# PrefixCache._run fuses the rest of prefill into its one decode call and
# never times the phases apart — prefill_seconds stays 0 (prefilled_tokens
# is still counted) and decode_seconds covers both. The VLM backend (the
# default model) always splits them, so the gauges are used whenever the
# phase timer actually ran; otherwise the deltas' own wall-clock timing is
# the only witness left. That clock starts before the request is handed to
# the session thread, so its prefill figure also carries the render and the
# tokenization the gauge excludes — a bench prompt's, at most a few percent
# of a multi-second prefill, but not the same basis.
def _prefill_rate(gauges: dict, prompt_tokens: int, first_delta: float | None) -> float | None:
    if gauges.get("prefilled_tokens") and gauges.get("prefill_seconds"):
        return _rate(gauges["prefilled_tokens"], gauges["prefill_seconds"])
    return _rate(prompt_tokens, first_delta)  # cold turn: the whole prompt was prefilled


def _decode_rate(
    produced: int, gauges: dict, first_delta: float | None, last_delta: float | None
) -> float | None:
    if gauges.get("decode_seconds") and gauges.get("prefill_seconds"):
        return _rate(produced, gauges["decode_seconds"])
    if first_delta is None or last_delta is None:
        return None
    if produced <= 1:
        # One token has no decode interval, but the turn did run: a
        # measured, degenerate rate that ranks last — not an absent one that
        # would drop the arm (and with it the baseline) from the decision.
        return 0.0
    return _rate(produced - 1, last_delta - first_delta)


def _continue(conversation: list[dict], answer: str) -> list[dict]:
    return [
        *conversation,
        {"role": "assistant", "content": answer},
        {"role": "user", "content": _CONTINUE},
    ]


def _check_drafter(arm: Arm, engine) -> None:
    """The engine survives a drafter that fails to load by running without
    it, which a measurement cannot: the row would carry the drafter's label
    over undrafted numbers, and the decision could write that drafter into
    the config. A backend that reports no drafter at all is not checked."""
    loaded = engine.drafter
    if loaded is not None and loaded != arm.drafter_id:
        raise RuntimeError(
            f"drafter {arm.drafter_id or 'none'} requested, engine runs with {loaded or 'none'}"
        )


def settle_resident(active_memory: Callable[[], int], baseline: int) -> int:
    """How much active memory stays above `baseline` once the allocator has
    had up to _UNLOAD_WAIT_SECONDS to give freed weights back: the figure a
    caller judges "released" on, after an unload or a failed load alike."""
    deadline = time.monotonic() + _UNLOAD_WAIT_SECONDS
    while active_memory() > baseline + _UNLOAD_SLACK_BYTES and time.monotonic() < deadline:
        time.sleep(0.2)
    return active_memory() - baseline


def release(
    manager: EngineManager,
    session,
    *,
    baseline: int,
    active_memory: Callable[[], int],
    label: str,
    out: Callable[..., None],
) -> str | None:
    """Free one arm's engine: the text of what went wrong when the weights
    stayed resident (a refused unload, memory that never came back, a
    teardown step that raised), None when they were freed. Never raises —
    the caller's measurement must survive its own teardown — and the run
    must stop on any text, because the next arm's peak-memory reading would
    be two models' worth."""
    try:
        if session is not None:
            session.close()
            # A wedged generation never dequeues _CLOSE, so this can't wait
            # for one; it only gives a healthy thread time to release its
            # mlx state.
            session.join(5.0)
        released = manager.unload_now()
        inflight_deadline = time.monotonic() + _INFLIGHT_WAIT_SECONDS
        waited = False
        while (
            not released["unloaded"]
            and released["reason"] == "a generation is in flight"
            and time.monotonic() < inflight_deadline
        ):
            if not waited:
                out(f"  {label}: waiting for an abandoned generation to end before the unload")
                waited = True
            time.sleep(_INFLIGHT_POLL_SECONDS)
            released = manager.unload_now()
        if not released["unloaded"]:
            # The weights are still resident — say so, and don't wait for
            # memory that cannot come back.
            out(f"  {label}: model not released ({released['reason']})")
            return f"unload refused: {released['reason']}"
        resident = settle_resident(active_memory, baseline)
        if resident > _UNLOAD_SLACK_BYTES:
            # The unload ran but the memory never came back: the same dirty
            # machine as a refused unload.
            out(f"  {label}: {resident / _GIB:.1f} GiB still resident after the unload")
            return f"memory not released: {resident / _GIB:.1f} GiB still resident"
        return None
    except Exception as e:  # noqa: BLE001 — the caller's row survives; only teardown failed
        out(f"  {label}: teardown failed ({type(e).__name__}: {e})")
        return f"teardown failed: {type(e).__name__}: {e}"


def _measure(
    arm: Arm, repeat: int, factory, timeout: float, peak_memory, reset_peak, active_memory, out
) -> BenchRow:
    baseline = active_memory()
    reset_peak()
    manager = EngineManager(arm.config, engine_factory=factory)
    loading = time.monotonic()
    engine = manager.get()
    load_seconds = time.monotonic() - loading
    # From here the weights are resident: everything that can fail runs
    # inside the try so the teardown below always gets to free them.
    session = None
    repeats = max(1, repeat)
    try:
        _check_drafter(arm, engine)
        session = engine.session()
        turn = _Turn(engine, session, timeout)
        count = engine.count_tokens
        short = build_prompt(count, SHORT_CONTEXT)
        decode_1k: list[float] = []
        ttfts: list[float] = []
        for _ in range(repeats):
            result = turn.run(short, DECODE_TOKENS)
            rate = _decode_rate(
                result.produced,
                result.gauges,
                result.first_delta_seconds,
                result.last_delta_seconds,
            )
            if rate is not None:
                decode_1k.append(rate)
            ttft = turn.run(_continue(short, result.text), 8).first_delta_seconds
            if ttft is not None:
                ttfts.append(ttft)
        prefill_prompt = build_prompt(count, PREFILL_CONTEXT, seed=1)
        prefill_tokens = count(prefill_prompt, TOOLS)
        prefill_2k_samples: list[float] = []
        for _ in range(repeats):
            result = turn.run(prefill_prompt, 1)
            rate = _prefill_rate(result.gauges, prefill_tokens, result.first_delta_seconds)
            if rate is not None:
                prefill_2k_samples.append(rate)
        long = build_prompt(count, LONG_CONTEXT, seed=2)
        long_tokens = count(long, TOOLS)
        prefill_16k = None
        decode_16k: list[float] = []
        # The 16K decode is what the decision ranks on, so it gets every
        # repeat too; the one 16K prefill is the prefix it continues from.
        if long_tokens + DECODE_TOKENS + _LONG_HEADROOM <= arm.window:
            result = turn.run(long, 1)
            prefill_16k = _prefill_rate(result.gauges, long_tokens, result.first_delta_seconds)
            warm = _continue(long, result.text)
            for _ in range(repeats):
                result = turn.run(warm, DECODE_TOKENS)
                rate = _decode_rate(
                    result.produced,
                    result.gauges,
                    result.first_delta_seconds,
                    result.last_delta_seconds,
                )
                if rate is not None:
                    decode_16k.append(rate)
        spread = None
        # The spread describes the measurement the decision ranks on. One
        # repeat has nothing to compare against — 0.0 would read as
        # "perfectly stable" rather than "unmeasured".
        deciding = decode_16k or decode_1k
        if len(deciding) >= 2 and max(deciding) > 0:
            spread = (max(deciding) - min(deciding)) / max(deciding)
        row = BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=True,
            error=None,
            load_seconds=load_seconds,
            prefill_tps_2k=max(prefill_2k_samples) if prefill_2k_samples else None,
            prefill_tps_16k=prefill_16k,
            decode_tps_1k=max(decode_1k) if decode_1k else None,
            decode_tps_16k=max(decode_16k) if decode_16k else None,
            ttft_seconds=min(ttfts) if ttfts else None,
            peak_memory_bytes=int(peak_memory()),
            spread=spread,
        )
    except BaseException as e:  # noqa: BLE001 — becomes the row's error
        row = BenchRow.failed(arm, f"{type(e).__name__}: {e}")

    # Teardown always runs, over whatever the try/except above produced: a
    # row already built in `row` can no longer be discarded by a teardown
    # problem the way a bare `finally` that returned from `try` could.
    error = release(
        manager, session, baseline=baseline, active_memory=active_memory, label=arm.label, out=out
    )
    if error is not None:
        row = row.with_teardown_error(error)

    out(f"  {arm.label}: {'done' if row.ok else 'failed'}")
    return row


def _peak_memory() -> int:
    import mlx.core as mx

    return int(mx.get_peak_memory())


def _reset_peak() -> None:
    import mlx.core as mx

    mx.reset_peak_memory()


def _active_memory() -> int:
    import mlx.core as mx

    mx.clear_cache()
    return int(mx.get_active_memory())


def bench_arm(
    arm: Arm,
    *,
    repeat: int = 2,
    factory: Callable[[str], object] | None = None,
    timeout: float = 600.0,
    peak_memory: Callable[[], int] | None = None,
    reset_peak: Callable[[], None] | None = None,
    active_memory: Callable[[], int] | None = None,
    out: Callable[..., None] = print,
) -> BenchRow:
    """The arm on a thread of its own, which releases its mlx state on the
    way out; a failure is a row that says so, and the next arm still runs."""
    outcome: list[BenchRow | BaseException] = []

    def run() -> None:
        try:
            outcome.append(
                _measure(
                    arm,
                    repeat,
                    factory,
                    timeout,
                    peak_memory or _peak_memory,
                    reset_peak or _reset_peak,
                    active_memory or _active_memory,
                    out,
                )
            )
        except BaseException as e:  # noqa: BLE001 — becomes the row's error
            outcome.append(e)
        finally:
            release_mlx_thread_state()

    worker = threading.Thread(target=run, name=f"sous-tune-{arm.label}", daemon=True)
    worker.start()
    worker.join()
    result = outcome[0]
    if isinstance(result, BaseException):
        # Only a failure before the load finished reaches here (_measure
        # turns everything after it into a row), so nothing is resident.
        return BenchRow.failed(arm, f"{type(result).__name__}: {result}")
    return result
