"""The in-flight turn registry: phases, rates, the recent ring, and that
nothing in it blocks or raises on the threads that write to it."""

import threading

from sous.inflight import (
    RECENT_TURNS,
    TYPICAL_OUTPUT_MIN_TURNS,
    Inflight,
)


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _registry() -> tuple[Inflight, Clock]:
    clock = Clock()
    return Inflight(clock=clock), clock


def _summary(**overrides) -> dict:
    base = {
        "ts": 1_000.0,
        "id": "msg_a",
        "model": "sous-local",
        "stream": 1,
        "status": 200,
        "error": None,
        "stop_reason": "end_turn",
        "cache": "hit",
        "took": "turn@55175",
        "input_tokens": 58302,
        "output_tokens": 192,
        "reused_tokens": 55175,
        "prefilled_tokens": 3120,
        "prefill_s": 4.0,
        "decode_s": 12.0,
        "seconds": 16.5,
    }
    base.update(overrides)
    return base


def _begin(reg: Inflight, turn_id: str = "msg_a", max_tokens: int = 4096) -> None:
    reg.begin(turn_id, model="sous-local", stream=True, max_tokens=max_tokens)


def _entry(reg: Inflight, turn_id: str = "msg_a") -> dict:
    (entry,) = [e for e in reg.snapshot()["inflight"] if e["id"] == turn_id]
    return entry


def test_a_turn_walks_the_phases_and_leaves():
    reg, clock = _registry()
    _begin(reg)
    entry = _entry(reg)
    assert entry["phase"] == "queued" and entry["started_at"] == 1_000.0
    assert entry["model"] == "sous-local" and entry["stream"] is True
    assert entry["max_tokens"] == 4096 and entry["input_tokens"] == 0
    for phase in ("loading", "tokenizing", "prefill"):
        clock.tick(1.0)
        reg.phase("msg_a", phase)
        entry = _entry(reg)
        assert entry["phase"] == phase and entry["phase_since"] == clock.now
    reg.sized("msg_a", 58302)
    assert _entry(reg)["input_tokens"] == 58302
    clock.tick(1.0)
    reg.progress("msg_a", 1)
    entry = _entry(reg)
    # The first delta is what ends the prefill: nobody else can stamp it.
    assert entry["phase"] == "decode" and entry["phase_since"] == clock.now
    assert entry["generated_tokens"] == 1
    reg.end("msg_a")
    assert reg.snapshot()["inflight"] == []


def test_queued_turns_are_listed_in_arrival_order():
    reg, _ = _registry()
    _begin(reg, "msg_a")
    _begin(reg, "msg_b")
    assert [e["id"] for e in reg.snapshot()["inflight"]] == ["msg_a", "msg_b"]


def test_generated_tokens_and_the_decode_rate_follow_the_deltas():
    reg, clock = _registry()
    _begin(reg)
    reg.phase("msg_a", "prefill")
    for n in range(1, 31):  # 30 tokens over 2 s: 15 tok/s
        clock.tick(1 / 15)
        reg.progress("msg_a", n)
    entry = _entry(reg)
    assert entry["generated_tokens"] == 30
    assert entry["decode_tps"] == 15.0
    clock.tick(5.0)  # a stall: every sample has left the rate window
    assert _entry(reg)["decode_tps"] is None


def test_a_single_delta_has_no_rate_yet():
    reg, _ = _registry()
    _begin(reg)
    reg.progress("msg_a", 3)
    entry = _entry(reg)
    assert entry["decode_tps"] is None and entry["eta_seconds"] is None


def test_the_decode_eta_uses_the_typical_output_length_capped_by_max_tokens():
    reg, clock = _registry()
    assert _begin(reg, "msg_early") is None
    reg.progress("msg_early", 1)
    clock.tick(1.0)
    reg.progress("msg_early", 16)
    # Fewer than the minimum number of completed turns: no estimate, no bar.
    early = _entry(reg, "msg_early")
    assert early["expected_output_tokens"] is None and early["eta_seconds"] is None
    reg.end("msg_early")
    for n in range(TYPICAL_OUTPUT_MIN_TURNS):
        reg.finished(_summary(id=f"msg_{n}", output_tokens=600))
    _begin(reg)
    for n in range(1, 31):
        clock.tick(1 / 15)
        reg.progress("msg_a", n)
    entry = _entry(reg)
    assert entry["expected_output_tokens"] == 600
    assert entry["eta_seconds"] == 38.0  # (600 - 30) / 15 tok/s
    _begin(reg, "msg_b", max_tokens=100)
    for n in range(1, 31):
        clock.tick(1 / 15)
        reg.progress("msg_b", n)
    entry = _entry(reg, "msg_b")
    assert entry["expected_output_tokens"] == 100  # never past the client's cap
    for n in range(31, 101):
        reg.progress("msg_b", n)
    # Past the estimate the ETA is unknown again rather than negative.
    assert _entry(reg, "msg_b")["eta_seconds"] is None


def test_the_typical_output_is_a_median_over_recent_turns_only():
    reg, _ = _registry()
    for n in range(TYPICAL_OUTPUT_MIN_TURNS):
        reg.finished(_summary(id=f"old_{n}", output_tokens=5000))
    for n in range(20):
        reg.finished(_summary(id=f"new_{n}", output_tokens=200 + n))
    reg.finished(_summary(id="failed", status=500, error="api_error", output_tokens=0))
    _begin(reg)
    reg.progress("msg_a", 1)
    # The 20 newest with an output: 200..219, median 209 (the failure's 0
    # does not count as an output).
    assert _entry(reg)["expected_output_tokens"] == 209


def test_the_prefill_eta_comes_from_the_probe_and_the_rolling_rate():
    reg, clock = _registry()
    reg.finished(_summary(prefilled_tokens=4000, prefill_s=2.0))  # 2000 tok/s
    _begin(reg)
    answers: list[tuple[int, int] | None] = [None, (55175, 3000)]
    reg.phase("msg_a", "prefill", probe=lambda: answers.pop(0))
    entry = _entry(reg)
    # The cache has not decided yet: nothing to size the bar with.
    assert entry["reused_tokens"] is None and entry["to_prefill"] is None
    assert entry["eta_seconds"] is None
    clock.tick(0.5)
    entry = _entry(reg)
    assert entry["reused_tokens"] == 55175 and entry["to_prefill"] == 3000
    assert entry["eta_seconds"] == 1.0  # 3000 / 2000 tok/s, less the 0.5 s spent
    assert answers == []  # decided once; never asked again
    clock.tick(5.0)
    assert _entry(reg)["eta_seconds"] == 0.0
    reg.progress("msg_a", 1)
    entry = _entry(reg)
    assert entry["phase"] == "decode" and entry["to_prefill"] == 3000


def test_a_probe_that_raises_or_answers_late_is_not_an_error():
    reg, _ = _registry()
    _begin(reg)
    reg.phase("msg_a", "prefill", probe=lambda: 1 // 0)  # ty: ignore[invalid-argument-type]
    assert _entry(reg)["to_prefill"] is None
    reg.phase("msg_a", "prefill", probe=lambda: (0, 10))
    reg.end("msg_a")
    assert reg.snapshot()["inflight"] == []


def test_the_rolling_prefill_rate_ignores_small_prefills_and_smooths_the_rest():
    reg, _ = _registry()
    reg.finished(_summary(prefilled_tokens=100, prefill_s=1.0))  # below the floor
    _begin(reg)
    reg.phase("msg_a", "prefill", probe=lambda: (0, 2000))
    assert _entry(reg)["eta_seconds"] is None  # no rate yet
    reg.end("msg_a")
    reg.finished(_summary(prefilled_tokens=2000, prefill_s=1.0))  # 2000 tok/s
    reg.finished(_summary(prefilled_tokens=1000, prefill_s=1.0))  # 1000 tok/s
    reg.finished(_summary(prefilled_tokens=0, prefill_s=0.0))  # a hit with nothing to prefill
    _begin(reg, "msg_b")
    reg.phase("msg_b", "prefill", probe=lambda: (0, 1670))
    # 0.33 * 1000 + 0.67 * 2000 = 1670 tok/s; the zero-prefill hit changed nothing.
    assert _entry(reg, "msg_b")["eta_seconds"] == 1.0


def test_recent_turns_are_newest_first_and_capped():
    reg, _ = _registry()
    for n in range(RECENT_TURNS + 5):
        reg.finished(_summary(id=f"msg_{n}"))
    recent = reg.snapshot()["recent_turns"]
    assert len(recent) == RECENT_TURNS
    assert recent[0]["id"] == f"msg_{RECENT_TURNS + 4}"
    assert recent[-1]["id"] == "msg_5"
    recent[0]["id"] = "mutated"
    assert reg.snapshot()["recent_turns"][0]["id"] != "mutated"  # a copy, not the ring


def test_unknown_ids_are_ignored_never_raised():
    reg, _ = _registry()
    before = reg.version
    reg.phase("msg_x", "decode")
    reg.sized("msg_x", 5)
    reg.progress("msg_x", 5)
    reg.end("msg_x")
    assert reg.snapshot() == {"inflight": [], "recent_turns": []}
    assert reg.version == before


def test_every_change_bumps_the_version_and_a_snapshot_does_not():
    reg, _ = _registry()
    seen = [reg.version]

    def bumped() -> None:
        assert reg.version > seen[-1]
        seen.append(reg.version)

    _begin(reg)
    bumped()
    reg.phase("msg_a", "loading")
    bumped()
    reg.sized("msg_a", 10)
    bumped()
    reg.progress("msg_a", 1)
    bumped()
    reg.snapshot()
    assert reg.version == seen[-1]
    reg.end("msg_a")
    bumped()
    reg.finished(_summary())
    bumped()


def test_writes_from_many_threads_stay_consistent():
    reg, _ = _registry()
    ids = [f"msg_{n}" for n in range(16)]
    for turn_id in ids:
        _begin(reg, turn_id)

    def hammer(turn_id: str) -> None:
        for n in range(1, 201):
            reg.progress(turn_id, n)
        reg.snapshot()

    threads = [threading.Thread(target=hammer, args=(t,)) for t in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    assert {e["generated_tokens"] for e in reg.snapshot()["inflight"]} == {200}
    assert len(reg.snapshot()["inflight"]) == 16
