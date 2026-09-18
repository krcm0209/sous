"""Throughput of one arm through sous's own engine: prefill and decode at a
short and a long context, warm-turn TTFT, load time and peak memory, read
from the prompt cache's per-turn gauges the way the gateway's turn line is."""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sous.engine.base import Delta, EngineManager, release_mlx_thread_state
from sous.protocol import WORKER_TOOLS
from sous.tune.arms import Arm

LONG_CONTEXT = 16384
SHORT_CONTEXT = 1024
PREFILL_CONTEXT = 2048
DECODE_TOKENS = 256
# A long-context measurement needs the prompt plus its answers to fit.
_LONG_HEADROOM = 1024
_UNLOAD_WAIT_SECONDS = 30.0
_UNLOAD_SLACK_BYTES = 1 << 30

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

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchRow:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


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
    engine's own count, grown geometrically so a 16K prompt costs a handful
    of tokenizations rather than hundreds. Deterministic per seed."""
    rng = random.Random(seed)
    lines: list[str] = []

    def render() -> list[dict]:
        body = "Review this module and continue it in the same style.\n\n" + "\n".join(lines)
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": body}]

    fn = 0
    while count(render(), WORKER_TOOLS) < target_tokens:
        n = max(16, len(lines) // 5)
        lines.extend(_code_lines(rng, n, fn))
        fn += n
    return render()


class _Turn:
    """One generate() through a session, with the gauges read on that
    session's thread right after — they are per-turn readings, never
    counter differences."""

    def __init__(self, engine, session, timeout: float):
        self.engine, self.session, self.timeout = engine, session, timeout

    def run(self, messages: list[dict], max_tokens: int) -> tuple[str, dict, int, float | None]:
        first: list[float] = []
        tokens: list[int] = [0]
        started = time.monotonic()

        def on_delta(d: Delta) -> None:
            if not first:
                first.append(time.monotonic() - started)
            tokens[0] = d.output_tokens

        text = self.session.generate(messages, WORKER_TOOLS, max_tokens, self.timeout, on_delta)
        gauges = self.engine.prompt_cache_stats(owner=self.session.thread)
        return text, gauges, tokens[0], (first[0] if first else None)


def _rate(tokens: float | None, seconds: float | None) -> float | None:
    if not tokens or not seconds or seconds <= 0:
        return None
    return tokens / seconds


def _measure(
    arm: Arm, repeat: int, factory, timeout: float, peak_memory, reset_peak, active_memory, out
) -> BenchRow:
    baseline = active_memory()
    reset_peak()
    manager = EngineManager(arm.config, engine_factory=factory)
    loading = time.monotonic()
    engine = manager.get()
    load_seconds = time.monotonic() - loading
    session = engine.session()
    turn = _Turn(engine, session, timeout)
    count = engine.count_tokens
    try:
        short = build_prompt(count, SHORT_CONTEXT)
        decode_1k: list[float] = []
        ttfts: list[float] = []
        for _ in range(max(1, repeat)):
            text, gauges, produced, _ = turn.run(short, DECODE_TOKENS)
            rate = _rate(produced, gauges.get("decode_seconds"))
            if rate is not None:
                decode_1k.append(rate)
            warm = [
                *short,
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Continue with the next function."},
            ]
            _, _, _, ttft = turn.run(warm, 8)
            if ttft is not None:
                ttfts.append(ttft)
        _, gauges, _, _ = turn.run(build_prompt(count, PREFILL_CONTEXT, seed=1), 1)
        prefill_2k = _rate(gauges.get("prefilled_tokens"), gauges.get("prefill_seconds"))
        prefill_16k = decode_16k = None
        if arm.window >= LONG_CONTEXT + _LONG_HEADROOM:
            long = build_prompt(count, LONG_CONTEXT, seed=2)
            text, gauges, _, _ = turn.run(long, 1)
            prefill_16k = _rate(gauges.get("prefilled_tokens"), gauges.get("prefill_seconds"))
            warm = [
                *long,
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Continue with the next function."},
            ]
            _, gauges, produced, _ = turn.run(warm, DECODE_TOKENS)
            decode_16k = _rate(produced, gauges.get("decode_seconds"))
        spread = None
        # One repeat has nothing to compare against — 0.0 would read as
        # "perfectly stable" rather than "unmeasured".
        if len(decode_1k) >= 2:
            spread = (max(decode_1k) - min(decode_1k)) / max(decode_1k)
        row = BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=True,
            error=None,
            load_seconds=load_seconds,
            prefill_tps_2k=prefill_2k,
            prefill_tps_16k=prefill_16k,
            decode_tps_1k=max(decode_1k) if decode_1k else None,
            decode_tps_16k=decode_16k,
            ttft_seconds=min(ttfts) if ttfts else None,
            peak_memory_bytes=int(peak_memory()),
            spread=spread,
        )
    except BaseException as e:  # noqa: BLE001 — becomes the row's error
        row = BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            load_seconds=None,
            prefill_tps_2k=None,
            prefill_tps_16k=None,
            decode_tps_1k=None,
            decode_tps_16k=None,
            ttft_seconds=None,
            peak_memory_bytes=None,
            spread=None,
        )

    # Teardown always runs, over whatever the try/except above produced: a
    # row already built in `row` can no longer be discarded by a teardown
    # problem the way a bare `finally` that returned from `try` could.
    session.close()
    # A wedged generation never dequeues _CLOSE, so this can't wait for one;
    # it only gives a healthy thread time to release its mlx state.
    session.join(5.0)
    released = manager.unload_now()
    if released["unloaded"]:
        deadline = time.monotonic() + _UNLOAD_WAIT_SECONDS
        while active_memory() > baseline + _UNLOAD_SLACK_BYTES and time.monotonic() < deadline:
            time.sleep(0.2)
    else:
        # The weights are still resident — say so, and don't wait for memory
        # that cannot come back; the next arm's peak reading will be too.
        out(f"  {arm.label}: model not released ({released['reason']})")
        suffix = f"unload refused: {released['reason']}"
        row = dataclasses.replace(row, error=suffix if row.ok else f"{row.error}; {suffix}")

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
        return BenchRow(
            label=arm.label,
            model_id=arm.model_id,
            drafter_id=arm.drafter_id,
            block_size=arm.block_size,
            window=arm.window,
            ok=False,
            error=f"{type(result).__name__}: {result}",
            load_seconds=None,
            prefill_tps_2k=None,
            prefill_tps_16k=None,
            decode_tps_1k=None,
            decode_tps_16k=None,
            ttft_seconds=None,
            peak_memory_bytes=None,
            spread=None,
        )
    return result
