"""What the local model is doing right now, and what it just did.

A registry of in-flight gateway turns keyed by the `msg_` id the client
already holds, plus the last fifty completed turns' summaries: what
`/sous/status`, `/sous/events` and the MCP `server_status` tool read. Every
write is a dict update under one lock, microseconds long — `progress` runs
on the engine's session thread from inside the decode loop, where nothing
may block or raise — so no method raises for an unknown id and none takes
any lock but its own. Every value is a count, a duration, an identifier or
a phase name: never a token, a prompt or a tool name.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

PHASES = ("queued", "loading", "tokenizing", "prefill", "decode")
RECENT_TURNS = 50
# The rolling decode rate is tokens over the deltas of the last few seconds:
# long enough to smooth a drafter's bursts of one to four tokens, short
# enough that a stall reads as one.
RATE_WINDOW_SECONDS = 2.0
# A prefill below this size is mostly fixed cost (the probe, a fork copy) and
# would drag the rate that sizes the next turn's prefill ETA.
RATE_MIN_PREFILL_TOKENS = 512
# The newest turn's prefill rate carries a third of the estimate: one odd
# turn cannot swing the next ETA, a real change shows within three.
RATE_EMA = 0.33
# How many completed turns the typical output length is the median of, and
# how many it needs before it is offered at all.
TYPICAL_OUTPUT_TURNS = 20
TYPICAL_OUTPUT_MIN_TURNS = 5

# Answers (reused_tokens, to_prefill) once the prompt cache has decided them
# for a turn, None while it is still deciding. Called from a status reader's
# thread, off this module's lock — never from the turn's own thread, which
# is inside the generation for the whole prefill.
Probe = Callable[[], tuple[int, int] | None]


@dataclass
class _Turn:
    id: str
    model: str
    stream: bool
    max_tokens: int
    started_at: float
    phase_since: float
    phase: str = "queued"
    input_tokens: int = 0
    generated_tokens: int = 0
    reused_tokens: int | None = None
    to_prefill: int | None = None
    # (time, generated so far) per delta, enough for the rate window at a
    # hundred tokens a second.
    samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=256))
    probe: Probe | None = None


def _decode_rate(samples: deque[tuple[float, int]], now: float) -> float | None:
    """Tokens per second over the samples inside the rate window, or None
    with fewer than two of them — a stall, or a turn that just started."""
    recent = [(t, n) for t, n in samples if now - t <= RATE_WINDOW_SECONDS]
    if len(recent) < 2:
        return None
    (t0, n0), (t1, n1) = recent[0], recent[-1]
    if t1 <= t0 or n1 <= n0:
        return None
    return (n1 - n0) / (t1 - t0)


def _prefill_rate(summary: dict) -> float | None:
    tokens = summary.get("prefilled_tokens") or 0
    seconds = summary.get("prefill_s") or 0.0
    if tokens < RATE_MIN_PREFILL_TOKENS or seconds <= 0:
        return None
    return tokens / seconds


class Inflight:
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._turns: dict[str, _Turn] = {}
        self._recent: deque[dict] = deque(maxlen=RECENT_TURNS)
        self._prefill_tps: float | None = None
        self._version = 0

    @property
    def version(self) -> int:
        """Bumped by every change: a reader that saw the same number twice
        saw the same registry twice. Read without the lock — one int."""
        return self._version

    def begin(self, turn_id: str, *, model: str, stream: bool, max_tokens: int) -> None:
        now = self._clock()
        with self._lock:
            self._turns[turn_id] = _Turn(turn_id, model, stream, max_tokens, now, now)
            self._version += 1

    def phase(self, turn_id: str, phase: str, *, probe: Probe | None = None) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn.phase, turn.phase_since, turn.probe = phase, self._clock(), probe
            self._version += 1

    def sized(self, turn_id: str, input_tokens: int) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            turn.input_tokens = input_tokens
            self._version += 1

    def progress(self, turn_id: str, generated_tokens: int) -> None:
        """One delta. The first one is what ends the prefill, so it moves
        the turn to `decode` itself: nothing else can see that moment."""
        now = self._clock()
        with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return
            if turn.phase != "decode":
                turn.phase, turn.phase_since, turn.probe = "decode", now, None
            turn.generated_tokens = generated_tokens
            turn.samples.append((now, generated_tokens))
            self._version += 1

    def end(self, turn_id: str) -> None:
        with self._lock:
            if self._turns.pop(turn_id, None) is not None:
                self._version += 1

    def finished(self, summary: dict) -> None:
        """A completed or failed turn's summary — the turn line's fields —
        kept newest first. A prefill of real size also feeds the rolling
        prefill rate the next turn's ETA is sized with."""
        with self._lock:
            self._recent.appendleft(dict(summary))
            rate = _prefill_rate(summary)
            if rate is not None:
                self._prefill_tps = (
                    rate
                    if self._prefill_tps is None
                    else RATE_EMA * rate + (1 - RATE_EMA) * self._prefill_tps
                )
            self._version += 1

    def snapshot(self) -> dict:
        """`{"inflight": [...], "recent_turns": [...]}`, every value
        JSON-ready and every dict a copy. A turn still in its prefill is
        asked for its size through its probe — here, outside this lock,
        because the probe takes the prompt cache's — so a caller belongs on
        a worker thread, never on the event loop."""
        with self._lock:
            pending = [
                (turn.id, turn.probe)
                for turn in self._turns.values()
                if turn.phase == "prefill" and turn.probe is not None and turn.to_prefill is None
            ]
        decided: dict[str, tuple[int, int]] = {}
        for turn_id, probe in pending:
            try:
                answer = probe()
            except Exception:  # noqa: BLE001 — the turn may have ended under the probe
                answer = None
            if answer is not None:
                decided[turn_id] = answer
        now = self._clock()
        with self._lock:
            for turn_id, (reused, to_prefill) in decided.items():
                turn = self._turns.get(turn_id)
                if turn is not None and turn.phase == "prefill":
                    turn.reused_tokens, turn.to_prefill = reused, to_prefill
            typical = self._typical_output()
            return {
                "inflight": [self._entry(turn, now, typical) for turn in self._turns.values()],
                "recent_turns": [dict(s) for s in self._recent],
            }

    def _typical_output(self) -> int | None:
        """Lock held by the caller. The median output of the newest turns
        that produced one; None until enough have."""
        outputs = [s["output_tokens"] for s in self._recent if s.get("output_tokens")][
            :TYPICAL_OUTPUT_TURNS
        ]
        if len(outputs) < TYPICAL_OUTPUT_MIN_TURNS:
            return None
        return int(statistics.median(outputs))

    def _entry(self, turn: _Turn, now: float, typical: int | None) -> dict:
        """Lock held by the caller."""
        tps = _decode_rate(turn.samples, now)
        eta: float | None = None
        expected: int | None = None
        if turn.phase == "prefill" and turn.to_prefill and self._prefill_tps:
            eta = max(0.0, turn.to_prefill / self._prefill_tps - (now - turn.phase_since))
        elif turn.phase == "decode":
            # Capped by the client's max_tokens, which is the hard stop; the
            # median is the likely one.
            expected = min(typical, turn.max_tokens) if typical else None
            if tps and expected and turn.generated_tokens < expected:
                eta = (expected - turn.generated_tokens) / tps
        return {
            "id": turn.id,
            "model": turn.model,
            "stream": turn.stream,
            "started_at": turn.started_at,
            "phase": turn.phase,
            "phase_since": turn.phase_since,
            "input_tokens": turn.input_tokens,
            "max_tokens": turn.max_tokens,
            "reused_tokens": turn.reused_tokens,
            "to_prefill": turn.to_prefill,
            "generated_tokens": turn.generated_tokens,
            "expected_output_tokens": expected,
            "decode_tps": None if tps is None else round(tps, 1),
            "eta_seconds": None if eta is None else round(eta, 1),
        }
