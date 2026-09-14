"""`sous top`: the words and fractions a document becomes, the chef's
frames, the feed reader, and the app under Textual's pilot against an
in-memory feed."""

import asyncio
import time
from pathlib import Path

import pytest
from rich.text import Text

from sous import tui
from sous.tui import Top, turn_view

BASE = 1_700_000_000.0  # 2023-11-14 22:13:20 UTC


class Clock:
    def __init__(self, now: float = BASE) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _doc(turn=None, *, loaded=True, loading=False, recent=None, tasks=None, behind=()):
    return {
        "engine": {
            "loaded": loaded,
            "loading": loading,
            "model_id": "mlx-community/Qwen3.8-27B-4bit",
            "idle_seconds": 252.0,
            "holders": 1,
            "memory_gb": 31.4,
            "prompt_cache": {
                "slots": 6,
                "resident_bytes": int(11.8 * (1 << 30)),
                "hits": 41,
                "fork_hits": 18,
                "misses": 7,
                "forks": 12,
                "evictions": 3,
                "pressure_evictions": 1,
                "reused_tokens": 1_820_000,
            },
        },
        "inflight": ([turn] if turn else []) + list(behind),
        "queue": {"queued": 2, "running": 1},
        "config": {
            "model_id": "mlx-community/Qwen3.8-27B-4bit",
            "idle_unload_minutes": 30,
            "gateway": {"enabled": True, "max_context_tokens": 262144},
        },
        "recent_turns": recent or [],
        "recent_tasks": tasks or [],
    }


def _turn(**overrides) -> dict:
    turn = {
        "id": "msg_f4cfe7d6880edc5168039e50",
        "model": "sous-local",
        "stream": True,
        "started_at": BASE - 41.3,
        "phase": "decode",
        "phase_since": BASE - 37.1,
        "input_tokens": 62141,
        "max_tokens": 1024,
        "reused_tokens": 53296,
        "to_prefill": 8845,
        "generated_tokens": 412,
        "expected_output_tokens": 1024,
        "decode_tps": 9.7,
        "eta_seconds": 62.0,
    }
    turn.update(overrides)
    return turn


def _summary(n: int, cache: str = "hit", **overrides) -> dict:
    summary = {
        "ts": BASE - 60 * n,
        "id": f"msg_{n:024x}",
        "model": "sous-local",
        "stream": 1,
        "status": 200,
        "error": None,
        "stop_reason": "end_turn",
        "cache": cache,
        "took": "turn@61904" if cache == "hit" else "none",
        "input_tokens": 62000,
        "output_tokens": 1024,
        "reused_tokens": 61904 if cache == "hit" else 0,
        "prefilled_tokens": 96 if cache == "hit" else 62000,
        "lcp": None,
        "lcp_region": None,
        "bounds": None,
        "forks": 0,
        "evicted": 0,
        "pressure": 0,
        "load_s": 0.0,
        "queue_s": 0.0,
        "engine_wait_s": 0.0,
        "tokenize_s": 0.2,
        "ttft_s": 0.4,
        "prefill_s": 0.4,
        "decode_s": 41.2,
        "prefill_tps": 240.0,
        "decode_tps": 9.0 + n * 0.3,
        "seconds": 44.1,
        "tools_hash": "1b21cd75",
        "system_hash": "9f8e7d6c",
    }
    summary.update(overrides)
    return summary


RECENT = [
    _summary(1, stop_reason="max_tokens"),
    _summary(2, "fork", took="fork@53296", reused_tokens=53296, evicted=2, pressure=2),
    _summary(3, "miss", lcp=0, lcp_region="tools", bounds=[1, 2], load_s=12.8),
    _summary(4, error="overloaded_error", status=529),
    _summary(5, error="abandoned", status=499),
]
TASKS = [
    {
        "id": "a",
        "state": "running",
        "title": "rename Delta callback across engine tests",
        "seconds": 41,
    },
    {
        "id": "b",
        "state": "done",
        "title": "add a regression test for the pressure valve",
        "seconds": 72,
    },
]


# --- words, strings and fractions -----------------------------------------------------


def test_clock_span_and_number_text():
    assert tui.clock_text(41.34) == "00:41" and tui.clock_text(41.3, tenths=True) == "00:41.3"
    assert tui.task_time(794) == "13:14" and tui.task_time(3725) == "62:05"
    assert tui.clock_text(-3) == "00:00" and tui.clock_text(3725) == "62:05"
    assert tui.span_text(252) == "4m 12s" and tui.span_text(48) == "48s"
    assert tui.span_text(3 * 3600 + 5 * 60) == "3h 05m"
    assert tui.kilo_text(61904) == "61.9k" and tui.kilo_text(96) == "96"
    assert tui.kilo_text(None) == "—"
    assert tui.seconds_text(104.94) == "104.9s" and tui.seconds_text(None) == "—"


def test_every_kitchen_word_is_glued_to_its_literal_or_its_number():
    """The legibility guard: a cache word beside the tokens it reused, a
    stop word beside how the ticket ended, a phase word beside its phase."""
    row = tui.rail_row(_summary(1, "fork", reused_tokens=53296, took="fork@53296"))
    assert row["cache"] == "DRAWER 53.3k" and row["took"] == "fork@53296"
    assert tui.rail_row(_summary(1, "miss"))["cache"] == "SCRATCH 0"
    assert tui.rail_row(_summary(1))["cache"] == "REHEAT 61.9k"
    failed = tui.rail_row(_summary(1, error="api_error", status=500))
    assert failed["cache"] == "DROPPED IT" and failed["in"] == "—" and failed["took"] == "none"
    assert failed["why"] == "DROPPED IT · 500 api_error"
    assert tui.stop_word(_summary(1, error="abandoned", status=499)) == "WALKED OUT"
    assert tui.stop_word(_summary(1, stop_reason="max_tokens")) == "PLATE FULL"
    assert tui.stop_word(_summary(1, stop_reason="tool_use")) == "ORDER UP"
    for line in tui.LEGEND.splitlines():
        assert line.strip(), "the legend has no blank line to lose a row to"
    for word in ("SEAR", "PLATE", "REHEAT", "DRAWER", "SCRATCH"):
        assert word in tui.LEGEND_STRIP and word in tui.LEGEND_STRIP_COMPACT


def test_why_column_names_the_stop_then_the_cost():
    assert tui.why_text(_summary(1)) == "ORDER UP"
    assert tui.why_text(_summary(1, stop_reason="max_tokens")) == "PLATE FULL"
    miss = _summary(1, "miss", lcp=62851, lcp_region="system", bounds=[62298, 62856])
    assert tui.why_text(miss) == "ORDER UP · lcp 62.9k · header"
    assert tui.why_text(_summary(1, "miss", lcp=0, lcp_region="tools")) == (
        "ORDER UP · lcp 0 · new tools"
    )
    assert tui.why_text(_summary(1, "miss", lcp=0, lcp_region=None)) == (
        "ORDER UP · lcp 0 · nothing to reheat"
    )
    assert tui.why_text(_summary(1, evicted=2)) == "ORDER UP · TOSSED ×2"
    assert tui.why_text(_summary(1, evicted=2, pressure=1)) == "ORDER UP · WALK-IN FULL ×1"
    assert tui.why_text(_summary(1, load_s=12.8, queue_s=2.5)) == (
        "ORDER UP · PREHEATING 12.8s · waited 2.5s"
    )
    assert tui.why_text(_summary(1, load_s=0.4, queue_s=0.9)) == "ORDER UP"
    assert tui.why_text(_summary(1, error="overloaded_error", status=529)) == (
        "DROPPED IT · 529 overloaded_error"
    )


def test_turn_view_fractions_and_words():
    now, received = BASE + 1.0, BASE
    view = turn_view(_turn(), now, received)
    assert view.phase == "decode" and view.word == "PLATING"
    assert view.elapsed == pytest.approx(42.3) and view.in_phase == pytest.approx(38.1)
    assert view.prefill_fraction == 1.0
    assert view.decode_fraction == pytest.approx(412 / 1024)
    assert view.eta == 62.0 and view.tps == 9.7
    assert turn_view(_turn(), now, received, stalled=True).word == "IN THE WINDOW"
    # Prefill with a size and an ETA: the bar is elapsed over (elapsed at
    # the event + eta), capped, and keeps moving after the event.
    prefill = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=6.0)
    assert turn_view(prefill, received, received).prefill_fraction == pytest.approx(37.1 / 43.1)
    later = turn_view(prefill, received + 2.0, received)
    assert later.prefill_fraction == pytest.approx(39.1 / 43.1)
    assert turn_view(prefill, received + 60.0, received).prefill_fraction == tui.PREFILL_CAP
    undecided = _turn(phase="prefill", to_prefill=None, reused_tokens=None, eta_seconds=None)
    assert turn_view(undecided, now, received).prefill_fraction is None
    assert turn_view(_turn(expected_output_tokens=None), now, received).decode_fraction is None
    assert turn_view(_turn(phase="tokenizing"), now, received).prefill_fraction == 0.0
    huge = _turn(generated_tokens=5000, expected_output_tokens=1024)
    assert turn_view(huge, now, received).decode_fraction == tui.PREFILL_CAP


def test_the_hi_score_is_the_best_rate_on_record():
    assert tui.hi_score(RECENT, []) == (10.5, BASE - 300)  # summary 5: 9.0 + 5 * 0.3
    assert tui.hi_score(RECENT, [9.0, 14.7]) == (14.7, None)
    assert tui.hi_score([], []) == (0.0, None)


def test_the_chef_has_a_face_for_every_state_and_holds_still_without_motion():
    for mood, (fps, frames) in tui.SPRITES.items():
        for frame in frames:
            assert len(frame) == 3 and all(len(row) == 7 for row in frame), mood
        # A pinned clock is a pinned frame; motion off is frame zero.
        assert tui.sprite_frame(mood, 12.34) == tui.sprite_frame(mood, 12.34)
        assert tui.sprite_frame(mood, 12.34, motion=False) == frames[0]
        if fps and len(frames) > 1:
            seen = {tui.sprite_frame(mood, n / fps) for n in range(len(frames) * 2)}
            assert seen == set(frames), mood
    quiet = tui.sprite_text("quiet", 0.0)
    assert "(-_-)" in quiet.plain and quiet.plain.count("\n") == 2
    assert "(x_x)" in tui.sprite_text("closed", 0.0).plain
    # The toque and the face are the terminal's own ink, bold; the apron teal.
    spans = [(quiet.plain[s.start : s.end], str(s.style)) for s in quiet.spans]
    assert ("▟███▙", "bold") in spans and ("▐███▌", tui.TEAL) in spans


def test_the_ticket_card_carries_every_field_of_the_turn():
    text = tui.ticket_text(_summary(3, "miss", lcp=0, lcp_region="tools", load_s=12.8))
    assert "ticket" in text and _summary(3)["id"] in text
    assert "SCRATCH miss" in text and "0.0s / 12.8s" in text
    assert "ORDER UP · lcp 0 · new tools · PREHEATING 12.8s" in text


def test_the_palette_reads_on_a_black_and_a_white_ground():
    """Every accent clears 3:1 against #1e1e1e and against white: the
    terminal's ground is never painted, so both are real."""

    def luminance(hex_colour: str) -> float:
        channels = []
        for n in (1, 3, 5):
            c = int(hex_colour[n : n + 2], 16) / 255
            channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        r, g, b = channels
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(a: float, b: float) -> float:
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)

    for colour in (tui.TEAL, tui.PINK, tui.PURPLE, tui.MUSTARD, tui.CHILLI, tui.SLATE):
        lum = luminance(colour)
        assert contrast(lum, luminance("#1e1e1e")) >= 3.0, colour
        assert contrast(lum, luminance("#ffffff")) >= 3.0, colour
    source = Path(tui.__file__).read_text()
    assert "background:" not in source.replace("background: ansi_default", "")


# --- the app --------------------------------------------------------------------------


class Feed:
    """A scripted feed: documents put on the queue are delivered in order;
    None makes the stream drop (FeedDown) and the next factory call
    reconnects."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.connections = 0

    def __call__(self):
        self.connections += 1
        return self._stream()

    async def _stream(self):
        while True:
            document = await self.queue.get()
            if document is None:
                raise tui.FeedDown("ConnectError")
            yield "status", document


def _run(test, *, size=(100, 30), clock=None):
    """Run `test(app, pilot, feed, clock)` inside the pilot."""
    clock = clock or Clock()
    feed = Feed()

    async def go():
        app = Top(feed, port=8383, clock=clock)
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            await test(app, pilot, feed, clock)
        return app

    return asyncio.run(go())


async def _deliver(feed: Feed, pilot, document: dict) -> None:
    await feed.queue.put(document)
    await pilot.pause(0.15)


def _plain(app, selector: str) -> str:
    return app.query_one(selector).render().plain


async def _until(pilot, predicate, *, ceiling: float = 1.5, step: float = 0.01) -> None:
    """Poll `predicate` every `step` seconds until it is true or `ceiling`
    passes, so a test that samples a tween, a highlight or a flash is not
    pinned to one exact instant. The caller repeats the same check in its
    own assertion right after this returns, so a tween, highlight or flash
    that never happens still fails the test — this only stops a slow run
    from missing a real one."""
    elapsed = 0.0
    while not predicate() and elapsed < ceiling:
        await pilot.pause(step)
        elapsed += step


def test_the_slip_shows_the_turn_from_the_document():
    async def test(app, pilot, feed, clock):
        behind = _turn(
            id="msg_64f2b36f3e4a528198cee4ea",
            phase="queued",
            started_at=BASE - 7,
            input_tokens=8912,
        )
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT, tasks=TASKS, behind=[behind]))
        slip = app.query_one(tui.Slip)
        assert slip.display and slip.border_title == "ORDER № msg_f4cfe7"
        assert slip.has_class("phase-decode") and slip.border_subtitle == "tear here"
        # The first document also looks like a phase change (there was no
        # prior phase), so the settling step's slate can still be showing.
        await _until(pilot, lambda: not slip.has_class("settling"))
        assert slip.styles.border_top[1].hex.upper() == tui.PHASE_COLOURS["decode"]
        headline = _plain(app, "#headline")
        assert headline.startswith(" PLATING  decode 412 tok") and "00:41.3" in headline
        assert headline.rstrip().endswith("ETA 01:02")
        assert app.query_one("#plate", tui.Gauge).fraction == pytest.approx(412 / 1024)
        assert app.query_one("#sear", tui.Gauge).fraction == 1.0
        assert _plain(app, "#shelf") == " REHEAT hit · reused 53,296 tok · 8,845 fresh"
        assert _plain(app, "#order").endswith("1st of 2")
        stub = app.query_one(tui.Stub)
        assert stub.display and stub.border_title == "ORDER № msg_64f2b3"
        assert "ON THE RAIL  queued 00:07   next up · 8,912 tok in" in stub.render().plain
        assert not app.query_one(tui.Card).display
        line = _plain(app, "#line-engine")
        assert line.startswith("  LINE IS OPEN  loaded") and "hold 1" in line
        assert "REHEAT  41 hit    DRAWER 18 fork" in _plain(app, "#line-body")
        body = _plain(app, "#line-body")
        assert " TASKS  ON RAIL 2 · COOKING 1" in body
        assert " ORDERS ORDER UP 3 · DROPPED IT 1" in body
        assert "        WALKED OUT 1" in body
        assert app.query_one("#line-chef", tui.Chef).mood == "decode"
        assert "HI-SCORE 10.5 tok/s" in app.query_one(tui.Strip).render().plain
        assert app.query_one(tui.Rail).row_count == 5
        assert _plain(app, "#tasks").startswith(" TASK  COOKING 0:41  rename Delta callback")
        assert _plain(app, "#footer") == tui.KEYS
        long_job = {
            "id": "c",
            "state": "running",
            "title": "the whole test suite",
            "seconds": 794,
        }
        await _deliver(feed, pilot, _doc(_turn(), tasks=[long_job]))
        assert _plain(app, "#tasks").startswith(" TASK  COOKING 13:14  the whole test suite")
        # Two-digit tallies (a plausible count on the book's 50-turn ring)
        # must still fit THE LINE's body at 100 columns without wrapping.
        big_recent = (
            [_summary(n) for n in range(10)]
            + [_summary(n, error="overloaded_error", status=529) for n in range(10, 22)]
            + [_summary(n, error="abandoned", status=499) for n in range(22, 24)]
        )
        await _deliver(feed, pilot, _doc(_turn(), recent=big_recent))
        wide_body = _plain(app, "#line-body")
        assert all(len(row) <= 35 for row in wide_body.splitlines())
        assert "WALKED OUT 2" in wide_body

    _run(test)


def test_prefill_shows_the_pulse_until_the_cache_decides():
    async def test(app, pilot, feed, clock):
        undecided = _turn(
            phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=None,
            to_prefill=None, reused_tokens=None, expected_output_tokens=None,
        )  # fmt: skip
        await _deliver(feed, pilot, _doc(undecided))
        assert app.query_one("#sear", tui.Gauge).fraction is None
        assert _plain(app, "#shelf") == " the walk-in is deciding"
        assert "ETA —" in _plain(app, "#headline")
        assert app.query_one(tui.Slip).has_class("phase-prefill")
        assert app.query_one("#line-chef", tui.Chef).mood == "prefill"
        decided = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=3.0)
        await _deliver(feed, pilot, _doc(decided))
        sear = app.query_one("#sear", tui.Gauge)
        assert sear.fraction == pytest.approx(37.1 / 40.1)
        assert sear.detail == "prefill 8,845 tok · ETA 00:03"

    _run(test)


def test_clocks_run_from_the_clock_and_a_silent_feed_reads_as_stalled():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        clock.now = BASE + 3.0
        await pilot.pause(0.25)  # the 10 fps ticker reads the clock
        slip = app.query_one(tui.Slip)
        assert "00:44.3" in _plain(app, "#headline") and not slip.has_class("stalled")
        assert app.query_one("#plate", tui.Gauge).fraction == pytest.approx(412 / 1024)
        clock.now = BASE + 6.0
        await pilot.pause(0.25)
        assert slip.has_class("stalled")
        assert _plain(app, "#headline").startswith(" IN THE WINDOW  stalled 00:06")
        assert slip.border_subtitle == "IN THE WINDOW · no event 6s"
        assert app.query_one("#line-chef", tui.Chef).mood == "stalled"
        clock.now = BASE + 12.0
        await pilot.pause(0.25)
        assert app.query_one("#line-chef", tui.Chef).mood == "stalled-long"

    _run(test)


def test_a_quiet_kitchen_shows_the_card_and_stops_the_ticker():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        await _deliver(feed, pilot, _doc(None, recent=RECENT))
        card = app.query_one(tui.Card)
        assert card.display and not app.query_one(tui.Slip).display
        assert card.border_title == "THE PASS IS CLEAR" and card.has_class("card-quiet")
        body = _plain(app, "#card-body")
        assert "KITCHEN QUIET  no orders on the rail for 4m 12s" in body
        assert "LIGHTS OUT in 25m 48s   idle unload at 30m" in body
        assert "HI-SCORE 10.5 tok/s  best plate rate today, set" in body
        assert "ORDER UP 3 · DROPPED IT 1 · WALKED OUT 1 · SCRATCH 1 miss" in body
        assert "s o u s" not in body and "▒▒▒▒▒" in _plain(app, "#card-art")
        assert app.query_one("#card-chef", tui.Chef).mood in ("done", "quiet")
        assert "KITCHEN QUIET 4m 12s" in app.query_one(tui.Strip).render().plain
        assert app.query_one("#firing", tui.LoadingIndicator).auto_refresh is None
        # Idle, the 10 fps clock is paused: a moved clock changes no headline.
        clock.now = BASE + 30.0
        await pilot.pause(0.25)
        assert "no orders on the rail for 4m 12s" in _plain(app, "#card-body")
        await _deliver(feed, pilot, _doc(None, loaded=False, loading=True))
        firing = app.query_one("#firing", tui.LoadingIndicator)
        assert card.border_title == "FIRING UP" and firing.display
        assert firing.auto_refresh == pytest.approx(1 / 16)
        assert app.query_one("#card-chef", tui.Chef).mood == "loading"
        clock.now = BASE + 40.0
        await pilot.pause(0.25)
        assert card.border_subtitle == "10s loading"

    _run(test)


def test_a_dropped_feed_redials_and_a_daemon_that_never_answered_exits_one():
    async def test(app, pilot, feed, clock):
        card = app.query_one(tui.Card)
        assert card.border_title == "KITCHEN CLOSED" and card.has_class("card-closed")
        assert app.query_one("#card-chef", tui.Chef).mood == "closed"
        await feed.queue.put(None)
        await pilot.pause(0.15)
        assert card.border_subtitle.startswith("attempt 1 · next try in")
        await pilot.pause(0.6)  # the first retry: 0.5 s
        assert feed.connections == 2
        await _deliver(feed, pilot, _doc(_turn()))
        assert app.query_one(tui.Slip).display and not app.query_one("#right").has_class("stale")
        await feed.queue.put(None)
        await pilot.pause(0.15)
        assert card.display and card.border_title == "VHS TRACKING"
        assert app.query_one("#right").has_class("stale")
        assert app.query_one(tui.LinePanel).border_subtitle == f"as of {tui.local_time(BASE)}"
        assert "redialing the daemon · try 1" in _plain(app, "#card-lead")
        bands = _plain(app, "#card-chef")
        clock.now = BASE + 0.25  # two VHS frames on: the card keeps its clock
        await pilot.pause(0.6)
        assert _plain(app, "#card-chef") != bands
        await pilot.press("r")  # retry now, ahead of the backoff
        await pilot.pause(0.1)
        assert feed.connections == 3
        await pilot.press("q")

    app = _run(test)
    assert app.return_code == 0

    async def never(app, pilot, feed, clock):
        await feed.queue.put(None)
        await pilot.pause(0.15)
        await pilot.press("q")

    assert _run(never).return_code == 1


def test_resize_reflows_to_every_breakpoint():
    async def test(app, pilot, feed, clock):
        behind = _turn(id="msg_64f2b36f3e4a528198cee4ea", phase="queued", started_at=BASE - 7)
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT, tasks=TASKS, behind=[behind]))
        rail = app.query_one(tui.Rail)
        assert tuple(rail._widths) == tui.RAIL_COLUMNS and rail._widths["why"] == 29
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.15)
        assert app.screen.has_class("compact")
        assert not app.query_one("#line-body").display and not app.query_one("#order").display
        assert app.query_one(tui.Stub).size.height == 1
        assert app.query_one("#tasks").display and app.query_one(tui.Rail).size.height == 4
        assert tuple(rail._widths) == tui.RAIL_COLUMNS_COMPACT
        assert _plain(app, "#legend") == tui.LEGEND_STRIP_COMPACT
        assert _plain(app, "#footer") == tui.KEYS_COMPACT
        await pilot.resize_terminal(50, 20)
        await pilot.pause(0.15)
        assert app.screen.has_class("narrow")
        for selector in ("#strip", "#rule", "#main", "#gingham", "#rail", "#legend"):
            assert not app.query_one(selector).display, selector
        assert _plain(app, "#footer") == " PLATING 412/1,024 9.7t/s 00:41  q"
        await pilot.resize_terminal(120, 36)
        await pilot.pause(0.15)
        assert app.screen.has_class("wide") and not app.screen.has_class("compact")
        assert app.query_one(tui.RatePanel).display
        assert app.query_one("#digits", tui.Digits).value == "9.7"
        assert tuple(rail._widths) == tui.RAIL_COLUMNS_WIDE
        assert rail.get_row(RECENT[1]["id"])[3] == "fork@53296"

    _run(test)


def test_keys_pause_the_rail_open_the_overlays_and_quit():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT[:2]))
        rail = app.query_one(tui.Rail)
        await pilot.press("p")
        await pilot.pause()
        assert rail.paused and "paused" in app.query_one("#gingham").render().plain
        # A resize while paused reflows the columns but must not blank the
        # book: the pause holds back new data, it never blanks what is
        # already on screen.
        before_rows, first_ticket = rail.row_count, rail.get_row_at(0)[1]
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.15)
        assert rail.row_count == before_rows and rail.get_row_at(0)[1] == first_ticket
        await pilot.resize_terminal(100, 30)
        await pilot.pause(0.15)
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT))
        assert rail.row_count == 2
        await pilot.press("p")
        await pilot.pause()
        assert rail.row_count == 5 and "paused" not in app.query_one("#gingham").render().plain
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, tui.Overlay)
        assert (
            app.screen.query_one("#overlay").render().plain.startswith(tui.LEGEND.splitlines()[0])
        )
        await pilot.resize_terminal(80, 24)  # reflows the pass, not the overlay
        await pilot.pause(0.15)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, tui.Overlay)
        assert app.screen.has_class("compact") and not app.query_one("#line-body").display
        await pilot.resize_terminal(100, 30)
        await pilot.pause(0.15)
        await pilot.press("question_mark")
        await pilot.pause()
        assert "brought to you by" in app.screen.query_one("#overlay").render().plain
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, tui.Overlay)
        assert app.screen.query_one("#overlay").border_title == "ORDER № msg_000000"
        # The DataTable's own enter binding and the App's must not both open
        # a ticket: exactly one Overlay reaches the stack, not two stacked.
        assert sum(isinstance(s, tui.Overlay) for s in app.screen_stack) == 1
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert app.motion is False
        await pilot.press("ctrl+c")

    assert _run(test).return_code == 0

    async def escape(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        await pilot.press("escape")

    assert _run(escape).return_code == 0


def test_the_app_keeps_the_terminals_own_ground():
    async def test(app, pilot, feed, clock):
        assert app.ansi_color is True and app.theme == "ansi-dark"
        assert app.current_theme.background == "ansi_default"

    _run(test)


# --- the feed reader --------------------------------------------------------------------


def test_a_closed_port_is_feed_down_at_once():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    async def go():
        started = time.monotonic()
        with pytest.raises(tui.FeedDown) as exc:
            async for _ in tui.sse_feed(port):
                pass
        assert "ConnectError" in str(exc.value)
        assert time.monotonic() - started < 3.0

    asyncio.run(go())


# --- motion ---------------------------------------------------------------------------


def test_the_plate_gauge_tweens_to_the_fed_value_and_snaps_without_motion():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(generated_tokens=100)))
        plate = app.query_one("#plate", tui.Gauge)
        await pilot.wait_for_animation()
        assert plate.fraction == pytest.approx(100 / 1024)
        await feed.queue.put(_doc(_turn(generated_tokens=900)))
        start, target = 100 / 1024, 900 / 1024
        await _until(pilot, lambda: start < plate.fraction < target)
        mid = plate.fraction
        assert start < mid < target, "the bar is between the two values mid-tween"
        await pilot.wait_for_animation()
        assert plate.fraction == pytest.approx(900 / 1024)
        app.motion = False
        await feed.queue.put(_doc(_turn(generated_tokens=200)))
        await _until(pilot, lambda: plate.fraction == 200 / 1024)
        assert plate.fraction == pytest.approx(200 / 1024)

    _run(test)


def test_a_phase_change_ramps_through_the_settling_step():
    async def test(app, pilot, feed, clock):
        prefill = _turn(phase="prefill", generated_tokens=0, decode_tps=None, eta_seconds=3.0)
        await _deliver(feed, pilot, _doc(prefill))
        slip = app.query_one(tui.Slip)
        # The very first document also looks like a phase change (there was
        # no prior phase), so it ramps through the same settling step.
        await _until(pilot, lambda: not slip.has_class("settling"))
        assert slip.has_class("phase-prefill") and not slip.has_class("settling")
        await feed.queue.put(_doc(_turn()))
        await _until(pilot, lambda: slip.has_class("phase-decode") and slip.has_class("settling"))
        assert slip.has_class("phase-decode") and slip.has_class("settling")
        assert slip.styles.border_top[1].hex.upper() == tui.SLATE
        await _until(pilot, lambda: not slip.has_class("settling"))
        assert not slip.has_class("settling")
        assert slip.styles.border_top[1].hex.upper() == tui.PHASE_COLOURS["decode"]
        await _deliver(feed, pilot, _doc(_turn(generated_tokens=500)))
        assert not slip.has_class("settling")
        app.motion = False
        await feed.queue.put(_doc(prefill))
        await pilot.pause(0.03)
        assert slip.has_class("phase-prefill") and not slip.has_class("settling")

    _run(test)


def test_a_new_ticket_on_the_rail_is_highlighted_then_settles():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn(), recent=RECENT[1:]))
        rail = app.query_one(tui.Rail)
        assert not isinstance(rail.get_row(RECENT[1]["id"])[0], Text)

        def cell_style() -> str | None:
            if RECENT[0]["id"] not in rail._shown:
                return None
            cell = rail.get_row(RECENT[0]["id"])[0]
            return str(cell.style) if isinstance(cell, Text) else None

        await feed.queue.put(_doc(_turn(), recent=RECENT))
        await _until(pilot, lambda: cell_style() == "reverse")
        assert cell_style() == "reverse"
        assert not isinstance(rail.get_row(RECENT[1]["id"])[0], Text)
        await _until(pilot, lambda: cell_style() == "bold")
        assert cell_style() == "bold"
        await _until(pilot, lambda: cell_style() is None)
        assert not isinstance(rail.get_row(RECENT[0]["id"])[0], Text)
        assert str(rail.get_row(RECENT[0]["id"])[2].style) == tui.TEAL
        app.motion = False
        await _deliver(feed, pilot, _doc(_turn(), recent=[_summary(9), *RECENT]))
        assert not isinstance(rail.get_row(_summary(9)["id"])[0], Text)

    _run(test)


def test_a_miss_or_a_pressure_eviction_flashes_the_line_once():
    async def test(app, pilot, feed, clock):
        await _deliver(feed, pilot, _doc(_turn()))
        line = app.query_one(tui.LinePanel)
        assert not line.has_class("flash")
        bumped = _doc(_turn())
        bumped["engine"]["prompt_cache"]["misses"] += 1
        await feed.queue.put(bumped)
        await _until(pilot, lambda: line.has_class("flash"))
        assert line.has_class("flash") and line.styles.border_top[1].hex.upper() == tui.PINK
        await _until(pilot, lambda: not line.has_class("flash"))
        assert not line.has_class("flash") and line.styles.border_top[1].hex.upper() == tui.TEAL
        app.motion = False
        quiet = _doc(_turn())
        quiet["engine"]["prompt_cache"]["pressure_evictions"] += 2
        await feed.queue.put(quiet)
        # Wait for the document itself to land (motion off never flashes, so
        # checking the class alone would pass even on a screen that never
        # saw this document) before asserting the flash stayed off.
        await _until(pilot, lambda: "WALK-IN FULL 3 pressure evict" in _plain(app, "#line-body"))
        assert not line.has_class("flash")

    _run(test)
