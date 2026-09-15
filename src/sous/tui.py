"""`sous top` — THE PASS: the daemon's live status as a terminal.

The one module in the tree that imports Textual, imported by the `sous top`
command function and nowhere else, so nothing the daemon runs loads it.
Everything on screen comes from the last status document `/sous/events`
delivered plus the clock: the functions in the first half turn a document
into words, strings and fractions and are tested without a terminal; the
widgets in the second half only display them.

The look: a restaurant's pass, drawn the way 1991 drew things. The turn in
flight is an order slip — a perforated border with ORDER № in its top edge
and "tear here" in its bottom — with the SEAR and PLATE gauges, a neon
graphic-EQ of tok/s and a kitchen-timer dial turning beside the clock. A
three-row pixel chef on THE LINE wears the engine's mood on its face.
Squiggles, triangles and confetti frame the title; gingham rules divide the
sections; the best rate today is the HI-SCORE. Two laws keep it legible:
whimsy decorates and numbers inform — every number is the terminal's own
foreground — and a kitchen word never appears without its literal term or
its number in the same cell. The terminal's own background is never
painted; every accent is a mid-luminance colour that reads on a black and a
white ground alike.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import localtime, strftime
from time import time as _wall_clock

import httpx
from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DataTable, Digits, LoadingIndicator, Sparkline, Static

Feed = Callable[[], AsyncIterator[tuple[str, dict]]]
Clock = Callable[[], float]

# --- vocabulary --------------------------------------------------------------------------
# The inline-gloss rule: a kitchen word shares its cell with the literal term
# or the number it stands for (`SEAR ▐███▌ prefill 4,296 tok`, `REHEAT 41 hit`).
# The legend strip above the footer and the `l` overlay carry the table.

PHASE_WORDS = {
    "queued": "ON THE RAIL",
    "loading": "PREHEATING",
    "tokenizing": "KNIFE WORK",
    "prefill": "SEARING",
    "decode": "PLATING",
}
DONE_WORD = "ORDER UP"
STALLED_WORD = "IN THE WINDOW"
CACHE_WORDS = {"hit": "REHEAT", "fork": "WARM DRAWER", "miss": "SCRATCH"}
RAIL_CACHE_WORDS = {**CACHE_WORDS, "fork": "DRAWER"}  # the rail's column is narrower
STOP_WORDS = {
    "end_turn": "ORDER UP",
    "tool_use": "ORDER UP",
    "max_tokens": "PLATE FULL",
}
FAILED_WORD = "DROPPED IT"
ABANDONED_WORD = "WALKED OUT"
ENGINE_WORDS = {
    "unloaded": "LIGHTS OUT",
    "loading": "FIRING UP",
    "loaded": "LINE IS OPEN",
}
QUIET_WORD = "KITCHEN QUIET"
CLOSED_WORD = "KITCHEN CLOSED"
TASK_WORDS = {
    "queued": "ON RAIL",
    "running": "COOKING",
    "awaiting_approval": "NEEDS CHEF",
    "done": "SERVED",
    "failed": "BURNT",
    "cancelled": "86'D",
}
REGION_WORDS = {
    "tools": "new tools",
    "system": "header",
    "tools-or-system": "tools or header",
    "conversation": "conversation",
}
LEGEND_STRIP = (
    " SEAR prefill  PLATE decode  REHEAT hit  DRAWER fork  SCRATCH miss"
    "  TOSSED evicted  86'D cancelled"
)
LEGEND_STRIP_COMPACT = " SEAR prefill · PLATE decode · REHEAT hit · DRAWER fork · SCRATCH miss"
KEYS = " q quit   l legend   ? about   m motion   p pause   ↑↓ orders   enter ticket   r reconnect"
KEYS_COMPACT = " q quit  l legend  ? about  m motion  p pause  ↑↓ orders  enter ticket"
LEGEND = """\
 ON THE RAIL    queued       admitted, waiting for a free slot
 PREHEATING     loading      model weights loading (10–15 s cold)
 KNIFE WORK     tokenizing   the prompt being counted
 SEARING        prefill      one fast burst of heat: reading the prompt
 PLATING        decode       slow and steady: writing tokens out
 ORDER UP       done         the turn finished
 IN THE WINDOW  stalled      no word from the daemon for 5 s; the clock runs
 PLATE FULL     max_tokens   the turn hit its output cap
 WALKED OUT     abandoned    the client left while the turn was queued
 DROPPED IT     failed       the turn failed or was refused
 REHEAT         hit          the prompt's prefix reused from a resident shelf
 WARM DRAWER    fork         a copy of another conversation's prefix, held hot
 SCRATCH        miss         nothing reusable; the prompt read cold
 TOSSED         evicted      a shelf dropped to make room
 WALK-IN FULL   pressure     shelves dropped under memory pressure
 LIGHTS OUT / FIRING UP / LINE IS OPEN   no model / loading / model resident
 hold N         holders      live `sous claude` sessions pinning the model
 ON RAIL / COOKING / NEEDS CHEF / SERVED / BURNT / 86'D
                             a delegated task queued / running / awaiting
                             your approval / done / failed / cancelled
"""
ABOUT = """\

   ∿▲● sous top ●▲∿

   brought to you by one Mac, one 27B chef,
   and no tokens billed to the dining room.

   any key
"""

# Seconds without a status event before the turn is shown as stalled, and
# before the chef grows a question mark.
STALL_SECONDS = 5.0
STALL_QUESTION_SECONDS = 10.0
# The reconnect backoff: first wait, then doubled up to the cap.
RETRY_SECONDS = (0.5, 8.0)
# The tok/s window, in seconds and in samples: one sample a second while a
# turn decodes, and nothing older than the window whatever it served before.
SPARK_SAMPLES = 60
# Breakpoints: below the first only the footer shows; below the second the
# columns stack; at the third the big rate readout appears.
NARROW_COLUMNS = 60
COMPACT_COLUMNS = 100
WIDE_COLUMNS = 120
# The prefill bar never claims more than this before the phase changes.
PREFILL_CAP = 0.95
DIAL = "◴◵◶◷"

# --- palette (relative luminance 0.19–0.29: ≥ 3:1 on #1e1e1e and on white) ------------
TEAL = "#17A2A2"  # chrome, rules, the apron, LINE IS OPEN, REHEAT, SERVED
PINK = "#D6488A"  # the title word, confetti, PLATING, the EQ's top, HI-SCORE, a flash
PURPLE = "#8E68DE"  # the order slip, ORDER №, the dial, the PLATE fill, KNIFE WORK
MUSTARD = "#A87A10"  # the SEAR fill, SEARING, PREHEATING, PLATE FULL, the stall ?
CHILLI = "#CE5450"  # BURNT, DROPPED IT, WALK-IN FULL, KITCHEN CLOSED
SLATE = "#79798E"  # legend, perforation, IN THE WINDOW, LIGHTS OUT, 86'D, steam, VHS
PHASE_COLOURS = {
    "queued": SLATE,
    "loading": MUSTARD,
    "tokenizing": PURPLE,
    "prefill": MUSTARD,
    "decode": PINK,
}
CACHE_COLOURS = {"hit": TEAL, "fork": PURPLE, "miss": MUSTARD}

# Decorative glyphs are East-Asian Ambiguous width in some terminals; a
# start-up measurement swaps the whole set for ASCII rather than let one
# two-cell squiggle shift a column. None of them ever sits in a table cell.
MOTIFS = {"squiggle": "∿", "triangle": "▲", "dot": "●", "dial": DIAL, "gingham": "▞▚"}
MOTIFS_ASCII = {
    "squiggle": "~",
    "triangle": "^",
    "dot": "o",
    "dial": "|/-\\",
    "gingham": "/\\",
}


def motifs() -> dict[str, str]:
    return MOTIFS if all(cell_len(g) == len(g) for g in MOTIFS.values()) else MOTIFS_ASCII


# --- words, strings and fractions -----------------------------------------------------


def clock_text(seconds: float, *, tenths: bool = False) -> str:
    """`mm:ss`, or `mm:ss.t`; never negative, hours roll into minutes."""
    seconds = max(0.0, seconds)
    if tenths:
        # Rounded to the tenth first: 41.3 is 41.299… in binary, and a
        # truncation would print it as 41.2.
        whole, tenth = divmod(round(seconds * 10), 10)
        return f"{whole // 60:02d}:{whole % 60:02d}.{tenth}"
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def task_time(seconds: float) -> str:
    """`m:ss` for a delegated task, minutes unbounded (a task can run an hour)."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def span_text(seconds: float) -> str:
    """`4m 12s`, `1h 03m`, `48s`: the idle and unload spans."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def kilo_text(count: int | None) -> str:
    """`26.5k` for a column; `—` for none."""
    if count is None:
        return "—"
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def seconds_text(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.1f}s"


def local_time(epoch: float) -> str:
    return strftime("%H:%M:%S", localtime(epoch))


def engine_word(engine: dict) -> str:
    if engine.get("loaded"):
        return ENGINE_WORDS["loaded"]
    return ENGINE_WORDS["loading" if engine.get("loading") else "unloaded"]


def stop_word(turn: dict) -> str:
    if turn.get("error"):
        return ABANDONED_WORD if turn.get("error") == "abandoned" else FAILED_WORD
    return STOP_WORDS.get(turn.get("stop_reason") or "", DONE_WORD)


def why_text(turn: dict) -> str:
    """The book's last column: how the ticket ended, then what made it cost
    what it cost — the miss and where it diverged, the walk-in's evictions,
    a cold load, a queue wait."""
    parts = [stop_word(turn)]
    if turn.get("error"):
        parts.append(f"{turn.get('status', '-')} {turn['error']}")
        return " · ".join(parts)
    if turn.get("cache") == "miss":
        region = REGION_WORDS.get(turn.get("lcp_region") or "", "nothing to reheat")
        parts.append(f"lcp {kilo_text(turn.get('lcp') or 0)} · {region}")
    if turn.get("pressure"):
        parts.append(f"WALK-IN FULL ×{turn['pressure']}")
    elif turn.get("evicted"):
        parts.append(f"TOSSED ×{turn['evicted']}")
    if (turn.get("load_s") or 0.0) >= 1.0:
        parts.append(f"PREHEATING {turn['load_s']:.1f}s")
    if (turn.get("queue_s") or 0.0) >= 1.0:
        parts.append(f"waited {turn['queue_s']:.1f}s")
    return " · ".join(parts)


# Column widths; `why` takes what is left. Each column costs its width plus
# two cells of the table's padding.
RAIL_WIDTHS = {
    "time": 8, "ticket": 10, "cache": 12, "took": 10, "in": 6, "out": 5,
    "sear": 6, "plate": 6, "total": 6, "why": 22,
}  # fmt: skip
RAIL_COLUMNS = ("time", "ticket", "cache", "in", "out", "plate", "total", "why")
RAIL_COLUMNS_WIDE = (
    "time", "ticket", "cache", "took", "in", "out", "sear", "plate", "total", "why",
)  # fmt: skip
RAIL_COLUMNS_COMPACT = ("time", "ticket", "cache", "plate", "total", "why")


def rail_row(turn: dict) -> dict[str, str]:
    """One completed turn as the book's cells, keyed by column name. The
    cache cell is the kitchen word beside the tokens it reused."""
    failed = bool(turn.get("error"))
    cache = turn.get("cache") or ""
    return {
        "time": local_time(turn.get("ts") or 0.0),
        "ticket": (turn.get("id") or "")[:10],
        "cache": ABANDONED_WORD
        if turn.get("error") == "abandoned"
        else FAILED_WORD
        if failed
        else f"{RAIL_CACHE_WORDS.get(cache, '—')} {kilo_text(turn.get('reused_tokens') or 0)}",
        "took": "none" if failed else turn.get("took") or "none",
        "in": "—" if failed else f"{turn.get('input_tokens', 0):,}",
        "out": "—" if failed else f"{turn.get('output_tokens', 0):,}",
        "sear": "—" if failed else seconds_text(turn.get("prefill_s")),
        "plate": "—" if failed else seconds_text(turn.get("decode_s")),
        "total": seconds_text(turn.get("seconds")),
        "why": why_text(turn),
    }


def turn_tallies(recent: list[dict]) -> tuple[int, int, int]:
    """How the book's turns ended, in the pass's own words: ORDER UP,
    DROPPED IT, WALKED OUT — never SERVED, BURNT or 86'D, which the legend
    reserves for delegated tasks, a different population."""
    up = sum(1 for t in recent if not t.get("error"))
    dropped = sum(1 for t in recent if t.get("error") and t.get("error") != "abandoned")
    walked = sum(1 for t in recent if t.get("error") == "abandoned")
    return up, dropped, walked


def hi_score(recent: list[dict], live: list[float]) -> tuple[float, float | None]:
    """The best decode rate on record — the completed turns' and the live
    samples' — and the time of the turn that set it."""
    best, when = 0.0, None
    for turn in recent:
        rate = turn.get("decode_tps") or 0.0
        if rate > best:
            best, when = rate, turn.get("ts")
    if live and max(live) > best:
        best, when = max(live), None
    return best, when


@dataclass(frozen=True)
class TurnView:
    """What the slip shows for the turn in flight, computed once per event or
    tick from the entry, the clock, and when the entry arrived."""

    id: str
    model: str
    phase: str
    word: str
    started_at: float
    elapsed: float
    in_phase: float
    input_tokens: int
    max_tokens: int
    reused: int | None
    to_prefill: int | None
    generated: int
    expected: int | None
    tps: float | None
    eta: float | None
    # 0–1 for a bar, None for the indeterminate pulse.
    prefill_fraction: float | None
    decode_fraction: float | None


def turn_view(turn: dict, now: float, received: float, *, stalled: bool = False) -> TurnView:
    """`received` is when the entry's document arrived — the moment its ETA
    was true — so the prefill bar keeps moving between events."""
    phase = turn.get("phase") or "queued"
    started = turn.get("started_at") or now
    since = turn.get("phase_since") or started
    eta = turn.get("eta_seconds")
    to_prefill = turn.get("to_prefill")
    prefill_fraction: float | None
    if phase == "decode":
        prefill_fraction = 1.0
    elif phase != "prefill":
        prefill_fraction = 0.0
    elif to_prefill and eta is not None:
        total = max(0.1, (received - since) + eta)
        prefill_fraction = min(PREFILL_CAP, (now - since) / total)
    else:
        prefill_fraction = None
    expected = turn.get("expected_output_tokens")
    generated = turn.get("generated_tokens") or 0
    decode_fraction = min(PREFILL_CAP, generated / expected) if expected else None
    return TurnView(
        id=turn.get("id") or "",
        model=turn.get("model") or "-",
        phase=phase,
        word=STALLED_WORD if stalled else PHASE_WORDS.get(phase, phase.upper()),
        started_at=started,
        elapsed=now - started,
        in_phase=now - since,
        input_tokens=turn.get("input_tokens") or 0,
        max_tokens=turn.get("max_tokens") or 0,
        reused=turn.get("reused_tokens"),
        to_prefill=to_prefill,
        generated=generated,
        expected=expected,
        tps=turn.get("decode_tps"),
        eta=eta,
        prefill_fraction=prefill_fraction,
        decode_fraction=decode_fraction,
    )


# --- the chef ----------------------------------------------------------------------------
# Every frame is 3 rows × 7 columns: columns 1–5 the chef (toque, face,
# apron), columns 0 and 6 a prop. Faces are ASCII so no ambiguous-width glyph
# can shift one by a cell; the frame is picked from the clock, so a pinned
# clock is a pinned picture.

SPRITES: dict[str, tuple[float, tuple[tuple[str, str, str], ...]]] = {
    "queued": (
        2,
        ((" ▟███▙ ", " (._.) ", " ▐███▌≡"), (" ▟███▙ ", " (._.) ", " ▐███▌=")),
    ),
    "loading": (
        3,
        ((" ▟███▙ ", " (o_o) ", " ▐███▌."), (" ▟███▙ ", " (o_o) ", " ▐███▌^")),
    ),
    "tokenizing": (
        6,
        (("╱▟███▙ ", " (>_<) ", " ▐███▌╴"), (" ▟███▙ ", "╱(>_<) ", " ▐███▌╴")),
    ),
    "prefill": (
        6,
        (
            (" ▟███▙ ", " (=_=)^", " ▐███▌▄"),
            (" ▟███▙ ", " (=_=)'", " ▐███▌▄"),
            (" ▟███▙ ", " (=_=) ", " ▐███▌▄"),
        ),
    ),
    "decode": (
        4,
        ((" ▟███▙ ", " (^_^) ", " ▐███▌═"), (" ▟███▙ ", " (^o^) ", " ▐███▌-")),
    ),
    "stalled": (
        1,
        ((" ▟███▙ ", " (-_o) ", " ▐███▌ "), (" ▟███▙ ", " (o_-) ", " ▐███▌ ")),
    ),
    "stalled-long": (1, ((" ▟███▙?", " (o_-) ", " ▐███▌ "),)),
    "done": (6, ((" ▟███▙*", " (^0^) ", " ▐███▌="), (" ▟███▙ ", " (^0^) ", " ▐███▌="))),
    "quiet": (
        0.5,
        ((" ▟███▙z", " (-_-) ", " ▐███▌ "), (" ▟███▙Z", " (-_-) ", " ▐▄█▄▌ ")),
    ),
    "closed": (0, ((" ▙███▟ ", " (x_x) ", " ▐░░░▌ "),)),
    "vhs": (
        8,
        (
            (" ▟███▙ ", " (=_=) ", "░▒▓░▒▓░"),
            (" ▟███▙ ", "▒▓░▒▓░▒", " ▐███▌ "),
            ("▓░▒▓░▒▓", " (=_=) ", " ▐███▌ "),
        ),
    ),
}
PROP_COLOURS = {
    "╱": MUSTARD, "╴": MUSTARD, "^": MUSTARD, "'": MUSTARD, "▄": MUSTARD, "?": MUSTARD,
    "═": PURPLE, "-": PURPLE, "≡": PURPLE, "=": PURPLE, ".": PURPLE,
    "*": PINK, "!": PINK,
    "z": SLATE, "Z": SLATE,
}  # fmt: skip
STEAM = ("    ∿ ∿    ", "   ∿   ∿   ")
POT = ("  ╭─────╮  ", " ═┫▒▒▒▒▒┣═ ", "  ╰─────╯  ")


def sprite_frame(mood: str, now: float, *, motion: bool = True) -> tuple[str, str, str]:
    fps, frames = SPRITES[mood]
    index = int(now * fps) % len(frames) if motion and fps else 0
    return frames[index]


def sprite_text(mood: str, now: float, *, motion: bool = True) -> Text:
    """The frame as styled text: toque and face in the terminal's own ink,
    bold; the apron teal; the props their own colours; the noise bands and
    a closed kitchen's apron in slate."""
    rows = []
    for n, row in enumerate(sprite_frame(mood, now, motion=motion)):
        left, chef, right = row[0], row[1:6], row[6]
        if set(row) <= set("░▒▓ "):
            rows.append(Text(row, style=SLATE))
            continue
        chef_style = (
            f"bold {SLATE}" if mood == "closed" and n == 2 else (TEAL if n == 2 else "bold")
        )
        rows.append(
            Text.assemble(
                (left, PROP_COLOURS.get(left, SLATE)),
                (chef, chef_style),
                (right, PROP_COLOURS.get(right, SLATE)),
            )
        )
    return Text("\n").join(rows)


# --- the feed ---------------------------------------------------------------------------


class FeedDown(Exception):
    """The daemon did not answer, refused, or hung up."""


async def sse_frames(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict]]:
    """(event, data) per frame of an SSE stream: the event's name (`message`
    when it carries none) and its data lines joined, as the JSON object they
    hold or `{}` when they hold anything else. Comment, id and retry lines
    are passed over, as the format says."""
    event: str | None = None
    data: list[str] = []
    async for line in lines:
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].removeprefix(" "))
        elif line == "" and (event is not None or data):
            try:
                payload = json.loads("\n".join(data))
            except ValueError:
                payload = {}
            yield event or "message", payload if isinstance(payload, dict) else {}
            event, data = None, []


async def sse_feed(port: int) -> AsyncIterator[tuple[str, dict]]:
    """`GET /sous/events` as (event, data) pairs until the daemon hangs up.
    The read timeout sits well past the daemon's 10 s ping, so a silent
    connection is one the daemon has left, not one between events."""
    url = f"http://127.0.0.1:{port}/sous/events"
    timeout = httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0)
    try:
        # trust_env=False: 127.0.0.1 is loopback, and a proxy variable must
        # never route or see this call — the launcher's rule, kept here.
        async with (
            httpx.AsyncClient(timeout=timeout, trust_env=False) as client,
            client.stream("GET", url) as reply,
        ):
            if reply.status_code != 200:
                raise FeedDown(f"status {reply.status_code}")
            async for frame in sse_frames(reply.aiter_lines()):
                yield frame
    except httpx.HTTPError as exc:
        # The type only: an httpx message can carry the URL it was building.
        raise FeedDown(type(exc).__name__) from None
    raise FeedDown("stream ended")


# --- widgets ------------------------------------------------------------------------------
# The feed sends a status document whenever something on it moved — while a
# turn is in flight, every second — and a document is applied to every widget
# at once. Textual's `Static.update` re-parses the text, drops the cached
# dimensions and lays the region out again whatever it is handed, and the
# border-title setter refreshes on every assignment: a heartbeat that changed
# only the idle clock cost fifteen repaints. The compare therefore lives at the
# widget boundary — one place, no caller to remember it, and no cached copy of
# the document, which would have to be invalidated for the clock and the one or
# two lines that genuinely tick.


class Line(Static):
    """A line (or block) of text that repaints only when the text changes."""

    def __init__(self, text: str = "", *, id: str) -> None:
        # markup=False throughout: the text carries the daemon's own model
        # ids and error strings, never console markup.
        super().__init__(text, id=id, markup=False)
        self._shown = text

    def show(self, text: str) -> None:
        if text != self._shown:
            self._shown = text
            self.update(text)


def _retitle(widget: Widget, *, title: str | None = None, subtitle: str | None = None) -> None:
    """Set a border title or subtitle only when it differs from the one drawn.
    The getter returns the markup the setter stored, so equal markup is the
    same picture."""
    if title is not None and widget.border_title != title:
        widget.border_title = title
    if subtitle is not None and widget.border_subtitle != subtitle:
        widget.border_subtitle = subtitle


class Strip(Widget):
    """The title strip: the name in its shapes on the left, a repeating run
    of shapes across, and the kitchen's headline — the HI-SCORE while the
    line is open, its state otherwise — on the right."""

    DEFAULT_CSS = "Strip { height: 1; }"

    def __init__(self) -> None:
        super().__init__(id="strip")
        self._headline = ""
        self._headline_colour = TEAL

    def show(self, headline: str, colour: str) -> None:
        if (headline, colour) == (self._headline, self._headline_colour):
            return
        self._headline, self._headline_colour = headline, colour
        self.refresh()

    def render(self) -> Text:
        m = motifs()
        s, t, d = m["squiggle"], m["triangle"], m["dot"]
        left = Text.assemble(
            (f"{s}{t}{d} ", TEAL),
            ("sous top", f"bold {PINK}"),
            (f" {d}{t}{s}", TEAL),
            ("  THE PASS", "bold"),
        )
        right = Text.assemble(
            (self._headline, f"bold {self._headline_colour}"), (f" {t}{d}{s}", TEAL)
        )
        room = self.size.width - left.cell_len - right.cell_len - 2
        pattern = Text()
        cycle = ((f"   {s}", TEAL), (f"   {t}", PINK), (f"    {d}", MUSTARD))
        while room > 0:
            for glyphs, colour in cycle:
                if pattern.cell_len + len(glyphs) > room:
                    pattern.append(" " * (room - pattern.cell_len))
                    room = 0
                    break
                pattern.append(glyphs, colour)
            else:
                continue
            break
        return Text.assemble(left, pattern, "  ", right)


class Rule(Widget):
    """A squiggle rule, or a gingham rule with a section name in it."""

    DEFAULT_CSS = "Rule { height: 1; }"

    def __init__(self, title: str = "", *, id: str) -> None:
        super().__init__(id=id)
        self._title = title

    def render(self) -> Text:
        m = motifs()
        width = self.size.width
        if not self._title:
            return Text(m["squiggle"] * width, style=TEAL)
        pair = m["gingham"]
        head = pair * 3
        tail_len = max(0, width - len(head) - len(self._title) - 2)
        out = Text.assemble((head, TEAL), (f" {self._title} ", f"bold {PINK}"))
        for n in range((tail_len + 1) // 2):
            out.append(pair[: max(0, min(2, tail_len - 2 * n))], TEAL if n % 2 == 0 else PINK)
        return out


class Gauge(Widget):
    """`SEAR  ▐████▁▁▁▌  prefill 4,296 tok · 6.2s` on one line: the kitchen
    word, a capped track, then the literal and whatever else the caller says.
    A fraction of None is the indeterminate pulse: a short bright band
    sliding along the track, positioned from the clock so a frozen clock is a
    frozen picture."""

    DEFAULT_CSS = "Gauge { height: 1; }"
    # The detail column's width, so the two gauges and the tok/s row share
    # one x-axis whatever their text says; the app sets it per breakpoint.
    detail_width = 30
    fraction: reactive[float | None] = reactive(0.0)

    def __init__(self, word: str, colour: str, *, id: str) -> None:
        super().__init__(id=id)
        self._word = word
        self._colour = colour
        self.detail = ""
        self.phase_time = 0.0
        self._target: tuple[float | None, str] | None = None

    def show(
        self,
        fraction: float | None,
        detail: str,
        phase_time: float,
        *,
        ease: float = 0.0,
    ) -> None:
        """`ease` is a tween's duration in seconds; 0 snaps. Only a number
        tweens to a number — the pulse (None) never tweens."""
        self.detail, self.phase_time = detail, phase_time
        if fraction is not None and (fraction, detail) == self._target:
            # The bar already drawn, or tweening to it: the 10 fps tick asks
            # for it again all through a decode, and each ask restarted the
            # tween and repainted. The pulse (None) moves with phase_time.
            return
        self._target = (fraction, detail)
        if ease and self.fraction is not None and fraction is not None:
            self.animate("fraction", fraction, duration=ease, easing="out_cubic")
        else:
            self.fraction = fraction
        self.refresh()

    def track_width(self) -> int:
        return max(8, self.size.width - 9 - self.detail_width)

    def render(self) -> Text:
        track = self.track_width()
        if self.fraction is None:
            band = 6
            span = max(1, track - band)
            step = int(self.phase_time * 12) % (2 * span)
            offset = step if step < span else 2 * span - step
            cells = Text.assemble(
                ("▁" * offset, SLATE),
                ("█" * band, self._colour),
                ("▁" * (track - offset - band), SLATE),
            )
        else:
            filled = int(round(self.fraction * track))
            cells = Text.assemble(("█" * filled, self._colour), ("▁" * (track - filled), SLATE))
        return Text.assemble(
            (f" {self._word:<6}", f"bold {self._colour}"),
            ("▐", SLATE),
            cells,
            ("▌", SLATE),
            f"  {self.detail[: self.detail_width - 2]}",
        )


class Chef(Widget):
    """The three-row pixel chef; its mood is the state it stands in."""

    DEFAULT_CSS = "Chef { width: 7; height: 3; }"

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.mood = "closed"
        self.now = 0.0
        self.motion = True
        self._shown = (self.mood, sprite_frame(self.mood, self.now, motion=self.motion))

    def show(self, mood: str, now: float, *, motion: bool) -> None:
        # The rendered picture is (mood, frame) alone — a closed kitchen's
        # frame never changes (fps 0) and a quiet one's only twice a second:
        # skip the repaint when it would be the one already on screen, so an
        # idle tick costs only what actually moved.
        shown = (mood, sprite_frame(mood, now, motion=motion))
        self.mood, self.now, self.motion = mood, now, motion
        if shown == self._shown:
            return
        self._shown = shown
        self.refresh()

    def render(self) -> Text:
        return sprite_text(self.mood, self.now, motion=self.motion)


class Slip(Vertical):
    """The order slip: the turn in flight on a perforated ticket."""

    def __init__(self) -> None:
        super().__init__(id="slip")
        self._phase = ""
        self._stats = ""

    def compose(self) -> ComposeResult:
        yield Line(id="headline")
        yield Static("", id="spacer1")
        yield Gauge("SEAR", MUSTARD, id="sear")
        yield Gauge("PLATE", PURPLE, id="plate")
        yield Static("", id="spacer2")
        with Horizontal(id="tps-row"):
            yield Static(" tok/s ", id="tps-label", markup=False)
            yield Sparkline([0.0], id="spark")
            yield Line(id="tps")
        yield Line(id="shelf")
        yield Static("", id="spacer3")
        yield Line(id="order")

    def _motion(self) -> bool:
        return getattr(self.app, "motion", True)

    def show(self, view: TurnView, behind: int, spark: list[float]) -> None:
        changed = view.phase != self._phase
        self._phase = view.phase
        for phase in PHASE_WORDS:
            self.set_class(phase == view.phase, f"phase-{phase}")
        if changed and self._motion():
            # Six accents cannot blend: the ramp is a slate step for a tenth
            # of a second, then the phase colour.
            self.add_class("settling")
            self.set_timer(0.1, lambda: self.remove_class("settling"))
        _retitle(self, title=f"ORDER № {view.id[:10]}")
        sep = " · " if not self.has_class("compact") else " "
        sear = f"prefill {view.to_prefill:,} tok" if view.to_prefill else "prefill sizing…"
        if view.phase == "prefill" and view.eta is not None:
            sear += f"{sep}ETA {clock_text(view.eta)}"
        elif view.phase == "decode":
            sear += f"{sep}done"
        self.query_one("#sear", Gauge).show(view.prefill_fraction, sear, view.in_phase)
        plate = f"{view.generated:,}"
        if view.expected:
            plate += f" / {view.expected:,} tok{sep}{int(100 * view.generated / view.expected)}%"
        else:
            plate += " tok"
        self.query_one("#plate", Gauge).show(
            view.decode_fraction if view.phase == "decode" else 0.0,
            plate,
            view.in_phase,
            ease=0.15 if self._motion() else 0.0,
        )
        padded = [0.0] * max(0, SPARK_SAMPLES - len(spark)) + spark[-SPARK_SAMPLES:]
        self.query_one("#spark", Sparkline).data = padded
        self._stats = f" {view.tps:.1f} now" if view.tps is not None else " — now"
        self.query_one("#tps", Line).show(self._stats)
        if view.reused is None:
            shelf = "the walk-in is deciding"
        elif view.reused:
            shelf = f"REHEAT hit · reused {view.reused:,} tok · {view.to_prefill or 0:,} fresh"
        else:
            shelf = f"SCRATCH miss · nothing reused · {view.to_prefill or 0:,} fresh"
        self.query_one("#shelf", Line).show(f" {shelf}")
        order = (
            f" in {view.input_tokens:,} tok   max {view.max_tokens:,}   "
            f"started {local_time(view.started_at)}"
        )
        if behind:
            order += f"   1st of {behind + 1}"
        self.query_one("#order", Line).show(order)
        self.tick(view, stalled=False, silence=0.0)

    def tick(self, view: TurnView, *, stalled: bool, silence: float) -> None:
        """The clocks, the dial and the phase word: run from the clock, not
        the feed."""
        self.set_class(stalled, "stalled")
        motion = self._motion()
        dial = motifs()["dial"][int(view.elapsed * 2) % 4 if motion else 0]
        clock = f"{dial} {clock_text(view.elapsed, tenths=True)}"
        literal = view.phase
        if stalled:
            head = f"{STALLED_WORD}  stalled {clock_text(silence)}"
            _retitle(self, subtitle=f"{STALLED_WORD} · no event {silence:.0f}s")
        else:
            head = f"{view.word}  {literal}"
            if view.phase == "decode":
                head += f" {view.generated:,} tok"
            _retitle(self, subtitle="tear here")
        eta = f"ETA {clock_text(view.eta)}" if view.eta is not None and not stalled else "ETA —"
        width = max(20, self.size.width - 2)
        gap = max(2, width - 1 - len(head) - len(clock) - len(eta) - 8)
        self.query_one("#headline", Line).show(
            f" {head}{' ' * (gap // 2)}{clock}{' ' * (gap - gap // 2 + 8)}{eta}"[:width]
        )
        sear = self.query_one("#sear", Gauge)
        sear.show(
            view.prefill_fraction,
            sear.detail,
            view.in_phase,
            ease=0.12 if motion else 0.0,
        )

    def reset(self) -> None:
        self._phase = ""
        for phase in PHASE_WORDS:
            self.remove_class(f"phase-{phase}")
        self.remove_class("stalled")


class Stub(Widget):
    """A queued turn behind the one on the slip: a slip of its own at full
    width, one perforated line when the columns stack."""

    def __init__(self) -> None:
        super().__init__(id="stub")
        self._turn: dict | None = None
        self._more = 0
        self._now = 0.0
        self._shown: tuple[str, bool] | None = None

    def show(self, turn: dict | None, more: int, now: float) -> None:
        self._turn, self._more, self._now = turn, more, now
        self.display = turn is not None
        if turn is not None:
            _retitle(self, title=f"ORDER № {turn.get('id', '')[:10]}", subtitle="tear here")
        # The words and the frame: a stack that crossed the compact
        # breakpoint draws the same line differently. An empty pass repaints
        # the empty stub once and then not again, whatever the clock says.
        shown = (self._line(), self.has_class("compact"))
        if shown != self._shown:
            self._shown = shown
            self.refresh()

    def _line(self) -> str:
        if self._turn is None:
            return ""
        t = self._turn
        waited = clock_text(self._now - (t.get("started_at") or self._now))
        tokens = t.get("input_tokens") or 0
        line = f"{PHASE_WORDS['queued']}  queued {waited}   next up · {tokens:,} tok in"
        if self._more:
            line += f" · +{self._more} more"
        return line

    def render(self) -> Text:
        line = self._line()
        if not line:
            return Text("")
        if self.has_class("compact"):
            pair = "┄┄"
            return Text.assemble((f" {pair} ", SLATE), (line, ""), (f" {pair}", SLATE))
        return Text.assemble(
            (" ", ""),
            (PHASE_WORDS["queued"], f"bold {SLATE}"),
            (line[len(PHASE_WORDS["queued"]) :], ""),
        )


class Card(Vertical):
    """The left column when there is no slip: the pass is clear, the stove
    is firing up, the feed is redialing, or the kitchen is closed."""

    def __init__(self) -> None:
        super().__init__(id="card")
        self._attempt = 0
        self._why = ""
        self._wait_text = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="card-top"):
            yield Chef(id="card-chef")
            yield Line(id="card-art")
            yield Line(id="card-lead")
        yield LoadingIndicator(id="firing")
        yield Line(id="card-body")

    def show_quiet(
        self, engine: dict, config: dict, recent: list[dict], now: float, motion: bool
    ) -> None:
        self._state("quiet")
        _retitle(self, title="THE PASS IS CLEAR")
        idle = engine.get("idle_seconds") or 0.0
        minutes = config.get("idle_unload_minutes")
        self.query_one("#card-chef", Chef).show("quiet", now, motion=motion)
        steam = STEAM[int(now * 2) % 2 if motion else 0]
        self.query_one("#card-art", Line).show("\n".join([steam, *POT]))
        self.query_one("#card-lead", Line).show("\n\n  stock on, nobody ordering")
        lines = ["", f" {QUIET_WORD}  no orders on the rail for {span_text(idle)}"]
        holders = engine.get("holders") or 0
        if engine.get("loaded") and holders:
            # The idle sweep never unloads under a hold, however long the
            # rail stays empty: a countdown here would run to a lights-out
            # that never comes.
            lines.append(f" {ENGINE_WORDS['loaded']} · hold {holders} · no idle unload while held")
        elif engine.get("loaded") and minutes and minutes * 60 > idle:
            lines.append(
                f" {ENGINE_WORDS['unloaded']} in {span_text(minutes * 60 - idle)}   "
                f"idle unload at {minutes}m"
            )
        else:
            lines.append(f" {ENGINE_WORDS['unloaded']} · the next order preheats the model")
        if recent:
            last = recent[0]
            lines += [
                "",
                f" LAST ORDER  {last.get('id', '')[:10]}  {stop_word(last)}"
                f"            {seconds_text(last.get('seconds'))} total",
            ]
            best, when = hi_score(recent, [])
            if best:
                set_at = f", set {local_time(when)}" if when else ""
                lines.append(f" HI-SCORE {best:.1f} tok/s  best plate rate today{set_at}")
            up, dropped, walked = turn_tallies(recent)
            scratch = sum(1 for t in recent if t.get("cache") == "miss")
            lines += [
                "",
                f" {DONE_WORD} {up} · {FAILED_WORD} {dropped} · {ABANDONED_WORD} {walked}"
                f" · SCRATCH {scratch} miss",
            ]
            oldest = min(t.get("ts") or now for t in recent)
            _retitle(self, subtitle=f"since {local_time(oldest)[:5]} · {span_text(now - oldest)}")
        else:
            _retitle(self, subtitle="")
        self.query_one("#card-body", Line).show("\n".join(lines))

    def show_firing(
        self,
        model_id: str,
        memory: float | None,
        elapsed: float,
        now: float,
        motion: bool,
    ) -> None:
        self._state("firing")
        _retitle(self, title=ENGINE_WORDS["loading"], subtitle=f"{elapsed:.0f}s loading")
        self.query_one("#card-chef", Chef).show("loading", now, motion=motion)
        self.query_one("#card-art", Line).show("")
        memory_text = f" · {memory:.1f} GB to read" if memory else ""
        self.query_one("#card-lead", Line).show(
            f"\n  loading {model_id.rsplit('/', 1)[-1]}{memory_text}"
        )
        self.query_one("#card-body", Line).show("")

    def show_redial(
        self, attempt: int, wait: float, why: str, as_of: str, now: float, motion: bool
    ) -> None:
        self._state("vhs")
        self._attempt, self._why, self._wait_text = attempt, why, f"{wait:.0f}"
        _retitle(self, title="VHS TRACKING", subtitle=f"as of {as_of}" if as_of else "")
        self.query_one("#card-chef", Chef).show("vhs", now, motion=motion)
        self.query_one("#card-art", Line).show("")
        self.query_one("#card-lead", Line).show(
            f"\n  redialing the daemon · try {attempt} · next in {wait:.0f}s\n  ({why})"
        )
        self.query_one("#card-body", Line).show("")

    def show_closed(self, port: int, attempt: int, wait: float, now: float) -> None:
        self._state("closed")
        self._attempt, self._wait_text = attempt, f"{wait:.0f}"
        _retitle(
            self,
            title=CLOSED_WORD,
            subtitle=f"attempt {attempt} · next try in {wait:.0f}s" if attempt else "",
        )
        self.query_one("#card-chef", Chef).show("closed", now, motion=False)
        self.query_one("#card-art", Line).show("")
        self.query_one("#card-lead", Line).show(
            f"\n  {CLOSED_WORD} — no daemon on 127.0.0.1:{port}\n  start it with:  sous serve"
        )
        self.query_one("#card-body", Line).show(
            "\n  (or: sous install-launchd)   q leaves; the screen keeps trying meanwhile"
        )

    def _state(self, state: str) -> None:
        for s in ("quiet", "firing", "vhs", "closed"):
            self.set_class(s == state, f"card-{s}")
        firing = self.query_one("#firing", LoadingIndicator)
        firing.display = state == "firing"
        # Its own mount starts a 16 Hz refresh; off the firing state that would
        # be the only timer running in a quiet kitchen.
        firing.auto_refresh = 1 / 16 if state == "firing" else None
        self.query_one("#card-art").display = state == "quiet"

    def tick(self, now: float, *, motion: bool, wait: float | None = None) -> None:
        """The idle tick's own work: the chef's motion, in a quiet kitchen
        the pot's steam, and — disconnected — the one number a real
        document or retry doesn't otherwise touch between events, the
        redial/closed countdown. Never a rebuild of the rest of the card's
        text."""
        chef = self.query_one("#card-chef", Chef)
        chef.show(chef.mood, now, motion=motion)
        if self.has_class("card-quiet"):
            steam = STEAM[int(now * 2) % 2 if motion else 0]
            self.query_one("#card-art", Line).show("\n".join([steam, *POT]))
        if wait is None:
            return
        text = f"{wait:.0f}"
        if text == self._wait_text:
            return
        self._wait_text = text
        if self.has_class("card-vhs"):
            self.query_one("#card-lead", Line).show(
                f"\n  redialing the daemon · try {self._attempt} · next in {text}s\n  ({self._why})"
            )
        elif self.has_class("card-closed") and self._attempt:
            _retitle(self, subtitle=f"attempt {self._attempt} · next try in {text}s")


class LinePanel(Vertical):
    """THE LINE: the chef, the engine, the walk-in's counters and the tickets."""

    def __init__(self) -> None:
        super().__init__(id="line")
        self.border_title = "THE LINE"

    def compose(self) -> ComposeResult:
        with Horizontal(id="line-top"):
            yield Chef(id="line-chef")
            yield Line(id="line-engine")
        yield Line(id="line-body")

    def show(
        self,
        mood: str,
        engine: dict,
        config: dict,
        queue: dict,
        recent: list[dict],
        now: float,
        motion: bool,
    ) -> None:
        self.query_one("#line-chef", Chef).show(mood, now, motion=motion)
        model = (config.get("model_id") or "-").rsplit("/", 1)[-1]
        memory = engine.get("memory_gb")
        holders = engine.get("holders") or 0
        c = engine.get("prompt_cache") or {}
        resident = (c.get("resident_bytes") or 0) / (1 << 30)
        up, dropped, walked = turn_tallies(recent)
        word = engine_word(engine)
        literal = (
            "loaded" if engine.get("loaded") else "loading" if engine.get("loading") else "unloaded"
        )
        gb = f"{memory:.1f} GB" if memory else "—"
        full = c.get("pressure_evictions", 0)
        compact = self.has_class("compact")
        if compact:
            self.query_one("#line-engine", Line).show(
                f"  {word}  {literal}   {model}   {gb}   hold {holders}\n"
                f"  REHEAT {c.get('hits', 0)} hit   DRAWER {c.get('fork_hits', 0)} fork   "
                f"SCRATCH {c.get('misses', 0)} miss\n"
                f"  TOSSED {c.get('evictions', 0)} evict  WALK-IN FULL {full}"
                f"   slots {c.get('slots', 0)} · {resident:.1f} GB"
            )
            self.query_one("#line-body", Line).show("")
        else:
            self.query_one("#line-engine", Line).show(
                f"  {word}  {literal}\n  {model}\n  {gb} · hold {holders}"
            )
            lines = [
                "",
                f" PROMPT CACHE   {c.get('slots', 0)} slots · {resident:.1f} GB",
                f" REHEAT {c.get('hits', 0):>3} hit    DRAWER {c.get('fork_hits', 0):>2} fork",
                f" SCRATCH {c.get('misses', 0):>2} miss  TOSSED {c.get('evictions', 0):>2} evict",
                f" WALK-IN FULL {c.get('pressure_evictions', 0)} pressure evict",
                f" reused {c.get('reused_tokens', 0):,} tok",
            ]
            # `size` is the content box: the sprite's three rows come off it.
            # The tickets block needs three more rows than the wide layout
            # leaves once the rate panel has taken its five. Tasks and turns
            # are two different populations, so each keeps its own row and
            # its own vocabulary rather than sharing one heading.
            if self.size.height - 3 >= len(lines) + 3:
                lines += [
                    f" {'TASKS':<7}ON RAIL {queue.get('queued', 0)}"
                    f" · COOKING {queue.get('running', 0)}",
                    f" {'ORDERS':<7}{DONE_WORD} {up} · {FAILED_WORD} {dropped}",
                    f" {'':<7}{ABANDONED_WORD} {walked}",
                ]
            self.query_one("#line-body", Line).show("\n".join(lines))
        idle = engine.get("idle_seconds")
        tail = f"idle {span_text(idle)}" if idle is not None and engine.get("loaded") else literal
        if compact:
            tail = f"ON RAIL {queue.get('queued', 0)} · COOKING {queue.get('running', 0)} · {tail}"
        _retitle(self, subtitle=tail)


class RatePanel(Vertical):
    """PLATE RATE: the live tok/s as three-row digits, at 120 columns."""

    def __init__(self) -> None:
        super().__init__(id="rate")
        self.border_title = "PLATE RATE  tok/s"

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Digits("", id="digits")
            yield Line(id="rate-text")

    def show(self, tps: float | None, floor: float | None, best: float, word: str) -> None:
        digits = self.query_one("#digits", Digits)
        value = f"{tps:.1f}" if tps is not None else ""
        # Digits.update refreshes whatever it is handed, and the panel is
        # blank at every width below 120 columns.
        if value != digits.value:
            digits.update(value)
        self.query_one("#rate-text", Line).show(
            f"tok/s now\nfloor {floor:.1f} · 60s\n{word}"
            if floor is not None
            else f"tok/s now\n\n{word}"
        )
        _retitle(self, subtitle=f"HI-SCORE {best:.1f} tok/s" if best else "")


class Rail(DataTable):
    """RECENT ORDERS: the last fifty completed tickets, newest on top."""

    def __init__(self) -> None:
        super().__init__(cursor_type="row", id="rail")
        self._columns: tuple[str, ...] = ()
        self._widths: dict[str, int] = {}
        self._shown: list[str] = []
        self.turns: list[dict] = []
        self.paused = False
        self._pending: list[dict] | None = None
        self._fading: list[Timer] = []

    def set_columns(self, columns: tuple[str, ...], screen_width: int) -> None:
        """The columns for a screen width; `why` takes what the others leave
        (the table's own padding costs two cells per column). A resize
        rebuilds the rows on the spot: a pause holds back new data, it
        never blanks what is already on screen."""
        widths = {name: RAIL_WIDTHS[name] for name in columns}
        used = sum(w + 2 for name, w in widths.items() if name != "why")
        widths["why"] = max(RAIL_WIDTHS["why"], screen_width - 2 - used - 2)
        if columns == self._columns and widths == self._widths:
            return
        self._columns, self._widths = columns, widths
        self._stop_fading()
        self.clear(columns=True)
        for name in columns:
            self.add_column(Text(name, style=SLATE), width=widths[name], key=name)
        self._shown = []
        self._rebuild(self.turns)

    def show(self, turns: list[dict]) -> list[str]:
        """Rebuild when the set of ids changed; returns the ids that are new."""
        if self.paused:
            self._pending = turns
            return []
        return self._rebuild(turns)

    def _rebuild(self, turns: list[dict]) -> list[str]:
        """Redraw every row from `turns`; returns the ids that are new."""
        ids = [t.get("id") or "" for t in turns]
        if ids == self._shown:
            return []
        new = [i for i in ids if i not in self._shown]
        had_rows = bool(self._shown)
        self._stop_fading()
        self.clear()
        self.turns = turns
        for turn in turns:
            cells = rail_row(turn)
            row = []
            for c in self._columns:
                value = cells[c]
                if len(value) > self._widths[c]:
                    # A cut with nothing to mark it reads as the whole value:
                    # `529 overloaded_e` looks like an error code of its own.
                    value = value[: self._widths[c] - 1] + "…"
                if c == "cache" and not turn.get("error"):
                    row.append(Text(value, style=CACHE_COLOURS.get(turn.get("cache") or "", "")))
                elif c == "cache":
                    row.append(Text(value, style=CHILLI))
                else:
                    row.append(value)
            self.add_row(*row, key=turn.get("id") or None)
        self._shown = ids
        if new and had_rows and getattr(self.app, "motion", True):
            self._highlight(new)
        return new

    def _stop_fading(self) -> None:
        """A highlight's last two steps are timers holding the cells of the
        table as it was. A rebuild that replaces those rows — or the columns
        they were measured against — stops the steps first: left running,
        they restyle rows the rebuild has already redrawn, and a second
        ticket landing inside the 200 ms re-flashed the first."""
        for timer in self._fading:
            timer.stop()
        self._fading = []

    def _highlight(self, ids: list[str]) -> None:
        """Three 100 ms steps — reverse, bold, then what the rebuild put
        there — on every cell of a row that just landed."""
        originals = {
            (row_key, name): self.get_cell(row_key, name)
            for row_key in ids
            if row_key in self._shown
            for name in self._columns
        }

        def restyle(style: str | None) -> None:
            for (row_key, name), original in originals.items():
                # A resize across a breakpoint inside the 200 ms window
                # rebuilds the table with another set of columns, so a step
                # still holding a row or a column that rebuild dropped would
                # raise out of the timer callback — and an exception there
                # takes the whole screen down.
                if row_key not in self._shown or name not in self._columns:
                    continue
                if style is None:
                    self.update_cell(row_key, name, original)
                else:
                    plain = original.plain if isinstance(original, Text) else original
                    self.update_cell(row_key, name, Text(plain, style=style))

        restyle("reverse")
        self._fading = [
            self.set_timer(0.1, lambda: restyle("bold")),
            self.set_timer(0.2, lambda: restyle(None)),
        ]

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        if not self.paused and self._pending is not None:
            pending, self._pending = self._pending, None
            self.show(pending)
        return self.paused

    def selected(self, row_key: str | None = None) -> dict | None:
        """The ticket under the cursor, or the one a row selection named."""
        if row_key is not None:
            return next((t for t in self.turns if t.get("id") == row_key), None)
        if not self.turns or self.cursor_row is None:
            return None
        return self.turns[self.cursor_row] if self.cursor_row < len(self.turns) else None


class Overlay(ModalScreen[None]):
    """The legend, the About card, or a ticket: a box in the middle, any key
    closes it."""

    BINDINGS = [Binding("escape", "dismiss", show=False)]

    def __init__(self, text: str | Text, title: str = "") -> None:
        super().__init__()
        self._text = text
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._text, id="overlay", markup=False)

    def on_mount(self) -> None:
        self.query_one("#overlay").border_title = self._title

    def on_key(self, event: events.Key) -> None:
        event.stop()
        self.dismiss(None)


def ticket_text(turn: dict) -> str:
    """One completed ticket — how it ended, what it reheated, and where its
    seconds went — for `enter`."""
    rows = [
        ("ticket", turn.get("id", "")),
        ("stop", stop_word(turn)),
        (
            "cache",
            f"{CACHE_WORDS.get(turn.get('cache') or '', '—')} {turn.get('cache') or ''}",
        ),
        ("took", turn.get("took") or "none"),
        (
            "in / out",
            f"{turn.get('input_tokens', 0):,} / {turn.get('output_tokens', 0):,} tok",
        ),
        (
            "reused / fresh",
            f"{turn.get('reused_tokens', 0):,} / {turn.get('prefilled_tokens', 0):,} tok",
        ),
        (
            "queue / load",
            f"{seconds_text(turn.get('queue_s'))} / {seconds_text(turn.get('load_s'))}",
        ),
        (
            "knife / sear / plate",
            " / ".join(seconds_text(turn.get(k)) for k in ("tokenize_s", "prefill_s", "decode_s")),
        ),
        ("first token", seconds_text(turn.get("ttft_s"))),
        ("total", seconds_text(turn.get("seconds"))),
        (
            "forks / tossed / pressure",
            f"{turn.get('forks', 0)} / {turn.get('evicted', 0)} / {turn.get('pressure', 0)}",
        ),
        ("why", why_text(turn)),
    ]
    return "\n".join(f" {k:<26} {v}" for k, v in rows)


def _slip_phase_css() -> str:
    """One `#slip.phase-<phase>` rule per phase, in that phase's own accent
    from `PHASE_COLOURS` — built outside the class's CSS f-string so the
    literal braces there stay readable."""
    return "\n    ".join(
        f"#slip.phase-{phase} {{ border: dashed {colour}; border-title-color: {colour}; "
        f"border-subtitle-color: {colour}; }}"
        for phase, colour in PHASE_COLOURS.items()
    )


class Top(App[int]):
    """`sous top`."""

    TITLE = "sous top"
    CSS = f"""
    Screen {{ background: ansi_default; color: ansi_default; layout: vertical; }}
    #main {{ height: 14; layout: horizontal; }}
    #left {{ width: 62; layout: vertical; margin-right: 1; }}
    #slip {{ height: 1fr; border: dashed {PURPLE}; border-title-color: {PURPLE};
             border-title-style: bold; border-subtitle-color: {PURPLE};
             border-subtitle-align: center; }}
    {_slip_phase_css()}
    #slip.stalled {{ border: dashed {SLATE}; border-title-color: {SLATE};
                     border-subtitle-color: {SLATE}; }}
    #slip.settling {{ border: dashed {SLATE}; }}
    #slip #spacer1, #slip #spacer2, #slip #spacer3 {{ height: 1; }}
    #tps-row {{ height: 1; }}
    #tps-label {{ width: auto; color: {SLATE}; }}
    #spark {{ width: 1fr; height: 1; }}
    #spark > .sparkline--max-color {{ color: {PINK}; }}
    #spark > .sparkline--min-color {{ color: {TEAL}; }}
    #tps {{ width: 30; }}
    #shelf {{ color: {TEAL}; }}
    #order {{ color: ansi_default; }}
    #stub {{ height: 3; border: dashed {SLATE}; border-title-color: {SLATE};
             border-subtitle-color: {SLATE}; border-subtitle-align: center; }}
    #card {{ height: 1fr; border: round {TEAL}; border-title-style: bold;
             border-subtitle-align: right; }}
    #card.card-quiet {{ border: round {TEAL}; border-title-color: {TEAL}; }}
    #card.card-firing {{ border: round {MUSTARD}; border-title-color: {MUSTARD}; }}
    #card.card-vhs {{ border: round {SLATE}; border-title-color: {SLATE}; }}
    #card.card-closed {{ border: round {CHILLI}; border-title-color: {CHILLI}; }}
    #card-top {{ height: 4; }}
    #card-chef {{ margin: 0 1 0 2; }}
    #card-art {{ width: 11; color: {SLATE}; margin: 0 2; }}
    #card-lead {{ width: 1fr; }}
    #firing {{ height: 1; color: {MUSTARD}; }}
    #line {{ width: 1fr; border: round {TEAL}; border-title-color: {TEAL}; border-title-style: bold;
             border-subtitle-align: right; border-subtitle-color: {SLATE}; }}
    #line.flash {{ border: round {PINK}; border-title-color: {PINK};
                   border-subtitle-color: {PINK}; }}
    #line-top {{ height: 3; }}
    #line-chef {{ margin: 0 0 0 2; }}
    #line-engine {{ width: 1fr; }}
    #rate {{ display: none; height: 5; border: round {PINK}; border-title-color: {PINK};
             border-subtitle-align: right; border-subtitle-color: {PINK}; }}
    #digits {{ width: auto; color: {PINK}; margin: 0 2 0 2; }}
    #rate-text {{ width: 1fr; }}
    #rail {{ height: 1fr; overflow-x: hidden; scrollbar-size: 0 0; }}
    #tasks {{ height: 1; color: ansi_default; }}
    #legend {{ height: 1; color: {SLATE}; }}
    #footer {{ height: 1; }}
    .stale {{ text-style: dim; }}
    Screen.compact #main {{ layout: vertical; height: 14; }}
    Screen.compact #left {{ width: 100%; height: 9; }}
    Screen.compact #slip {{ height: 8; }}
    Screen.compact #slip #spacer1, Screen.compact #slip #spacer3, Screen.compact #order {{
        display: none; }}
    Screen.compact #stub {{ height: 1; border: none; }}
    Screen.compact #line {{ height: 5; }}
    Screen.compact #line-body {{ display: none; }}
    Screen.compact #tps {{ width: 12; }}
    Screen.narrow #strip, Screen.narrow #rule, Screen.narrow #main, Screen.narrow #gingham,
    Screen.narrow #rail, Screen.narrow #tasks, Screen.narrow #legend {{ display: none; }}
    Screen.wide #main {{ height: 16; }}
    Screen.wide #left {{ width: 78; }}
    Screen.wide #rate {{ display: block; }}
    Screen.wide #tasks {{ height: 3; }}
    Overlay {{ align: center middle; }}
    #overlay {{ width: 84; border: round {PINK}; border-title-color: {PINK}; padding: 0 2;
                background: ansi_default; }}
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", show=False, priority=True),
        Binding("q", "quit", "quit"),
        Binding("escape", "quit", show=False),
        Binding("p", "pause", "pause"),
        Binding("l", "legend", "legend"),
        Binding("m", "motion", "motion"),
        Binding("question_mark", "about", "about"),
        Binding("enter", "ticket", "ticket"),
        Binding("r", "reconnect", "reconnect"),
    ]

    def __init__(self, feed: Feed, *, port: int, clock: Clock = _wall_clock) -> None:
        super().__init__(ansi_color=True)
        # ansi_color alone leaves the default theme in place; the ANSI theme
        # is what makes foreground and background the terminal's own, and
        # hex accents still render as they are.
        self.theme = "ansi-dark"
        self._feed = feed
        self._port = port
        self._clock = clock
        self.motion = True
        self._document: dict = {}
        self._turn: dict | None = None
        self._behind: list[dict] = []
        self._received_at = 0.0
        # The idle second last painted, so the tick repaints the clock's
        # lines once a second and never twice for the same words.
        self._idle_shown: int | None = None
        self._connected = False
        self._ever_connected = False
        self._attempt = 0
        self._retry_at: float | None = None
        self._retry_now = asyncio.Event()
        self._why = ""
        self._firing_since: float | None = None
        self._done_until = 0.0
        self._spark: deque[tuple[float, float]] = deque(maxlen=SPARK_SAMPLES)
        self._spark_at = float("-inf")
        self._counters: dict[str, int] = {}
        self._ticker: Timer | None = None
        self._idler: Timer | None = None
        self._base_screen: Screen | None = None
        self._width = 0

    # -- layout ---------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Strip()
        yield Rule(id="rule")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Slip()
                yield Stub()
                yield Card()
            with Vertical(id="right"):
                yield LinePanel()
                yield RatePanel()
        yield Rule("RECENT ORDERS", id="gingham")
        yield Rail()
        yield Line(id="tasks")
        yield Line(LEGEND_STRIP, id="legend")
        yield Line(KEYS, id="footer")

    def on_mount(self) -> None:
        self._base_screen = self.screen
        self._width = self.size.width
        self._breakpoints(self._width)
        self._show_left(None)
        self._ticker = self.set_interval(0.1, self._tick, pause=True)
        self._idler = self.set_interval(0.5, self._idle_tick, pause=True)
        self.query_one(Rail).focus()
        self.run_worker(self._pump(), exclusive=True)

    def on_resize(self, event: events.Resize) -> None:
        self._width = event.size.width
        self._breakpoints(self._width)
        if self._document:
            self._apply(self._document, resize=True)
        else:
            self._show_left(None)
            self._refresh_footer()

    def _breakpoints(self, width: int) -> None:
        screen = self._base_screen or self.screen  # not the overlay that may be up
        screen.set_class(width < NARROW_COLUMNS, "narrow")
        screen.set_class(width < COMPACT_COLUMNS, "compact")
        screen.set_class(width >= WIDE_COLUMNS, "wide")
        for widget in (
            self.query_one(Slip),
            self.query_one(Stub),
            self.query_one(LinePanel),
        ):
            widget.set_class(width < COMPACT_COLUMNS, "compact")
        for gauge in self.query(Gauge):
            gauge.detail_width = 30 if width >= COMPACT_COLUMNS else 26
        self.query_one("#legend", Line).show(
            LEGEND_STRIP if width >= COMPACT_COLUMNS else LEGEND_STRIP_COMPACT
        )
        self.query_one(Rail).set_columns(
            RAIL_COLUMNS_COMPACT
            if width < COMPACT_COLUMNS
            else RAIL_COLUMNS_WIDE
            if width >= WIDE_COLUMNS
            else RAIL_COLUMNS,
            width,
        )

    # -- the feed -------------------------------------------------------------------

    async def _pump(self) -> None:
        """Read the feed for as long as the app runs. A drop is shown and
        retried with backoff (or at once on `r`) — never a fallback to
        polling: same daemon, same port, and a daemon that is gone answers
        neither."""
        delay = RETRY_SECONDS[0]
        while True:
            try:
                async for event, payload in self._feed():
                    if event == "status":
                        self._apply(payload)
                        delay = RETRY_SECONDS[0]
            except FeedDown as exc:
                self._dropped(str(exc))
            except Exception as exc:  # noqa: BLE001 — a redial, never the end of the screen
                # Textual ends the app on a worker's uncaught exception, which
                # on the alternate screen is a terminal left mid-frame. The
                # type only: an exception's message can carry a URL.
                self._dropped(type(exc).__name__)
            self._attempt += 1
            self._retry_at = self._clock() + delay
            self._show_left(None)
            self._refresh_footer()
            self._retry_now.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._retry_now.wait(), delay)
            delay = min(delay * 2, RETRY_SECONDS[1])

    def _dropped(self, why: str) -> None:
        self._connected = False
        self._why = why
        self._turn = None
        if self._ticker is not None:
            self._ticker.pause()
        if self._idler is not None:
            self._idler.resume()  # the redial countdown and the tracking bands
        for selector in ("#right", "#rail", "#tasks"):
            self.query_one(selector).add_class("stale")
        _retitle(
            self.query_one(LinePanel),
            subtitle=f"as of {local_time(self._received_at)}" if self._ever_connected else "",
        )
        self._show_left(None)
        self._refresh_footer()

    # -- the document ---------------------------------------------------------------

    def _apply(self, document: dict, *, resize: bool = False) -> None:
        now = self._clock()
        if not resize:
            self._document = document
            self._received_at = now
            self._connected = self._ever_connected = True
            self._attempt = 0
            self._retry_at = None
            for selector in ("#right", "#rail", "#tasks"):
                self.query_one(selector).remove_class("stale")
        # On a fresh document the offset is zero; on a resize it is the
        # clock already on screen, never the document's older one.
        engine = self._engine_now(now)
        # The document paints its own second; the tick repaints only once
        # the clock has moved past it.
        idle_now = engine.get("idle_seconds")
        self._idle_shown = int(idle_now) if idle_now is not None else None
        config = document.get("config") or {}
        inflight = document.get("inflight") or []
        recent = document.get("recent_turns") or []
        queue = document.get("queue") or {}
        previous = self._turn
        # A resize re-applies the document that is already on screen, which
        # while the feed is down is a stale one: reading its turn back would
        # put a finished order on the pass and restart the 10 fps ticker
        # against a daemon that is gone. A live document sets _connected
        # above, so a turn in flight is unaffected.
        self._turn = inflight[0] if inflight and self._connected else None
        self._behind = inflight[1:]
        if not resize and previous is not None and self._turn is None and self.motion:
            # The ticket just left the pass: the chef's flourish, for a beat.
            self._done_until = now + 1.2
        if self._turn is not None:
            view = turn_view(self._turn, now, self._received_at)
            if not resize and view.phase == "decode" and now - self._spark_at >= 1.0:
                self._spark.append((now, view.tps or 0.0))
                self._spark_at = now
            self._show_left(view)
        else:
            self._show_left(None)
        if not engine.get("loading"):
            self._firing_since = None
        mood = self._mood(engine, now)
        self.query_one(LinePanel).show(mood, engine, config, queue, recent, now, self.motion)
        live = self._live_rates(now)
        best, _ = hi_score(recent, live)
        floor = [r for r in live if r]
        self.query_one(RatePanel).show(
            self._turn and turn_view(self._turn, now, self._received_at).tps,
            min(floor) if floor else None,
            best,
            f"{PHASE_WORDS.get(self._turn.get('phase', ''), '')}  {self._turn.get('phase', '')}"
            if self._turn
            else QUIET_WORD,
        )
        headline, colour = self._headline(engine, recent, now)
        self.query_one(Strip).show(headline, colour)
        self.query_one(Rail).show(recent)
        self._show_tasks(document.get("recent_tasks") or [])
        cache = engine.get("prompt_cache") or {}
        if not resize:
            rose = {
                key: cache.get(key, 0) > self._counters.get(key, cache.get(key, 0))
                for key in ("misses", "pressure_evictions")
            }
            self._counters = {k: cache.get(k, 0) for k in ("misses", "pressure_evictions")}
            if self.motion and (rose["misses"] or rose["pressure_evictions"]):
                self.query_one(LinePanel).add_class("flash")
                self.set_timer(0.25, lambda: self.query_one(LinePanel).remove_class("flash"))
        self._refresh_footer()
        busy = self._connected and (self._turn is not None or bool(engine.get("loading")))
        if self._ticker is not None:
            self._ticker.resume() if busy else self._ticker.pause()
        if self._idler is not None:
            # Idle against a live daemon the tick runs the idle clock — the
            # daemon sends no document while nothing moves — and
            # disconnected it is the only thing that moves the redial
            # countdown, so it runs whenever nothing is on the pass, with
            # the motion flag deciding only what it animates.
            self._idler.resume() if not busy else self._idler.pause()

    def _mood(self, engine: dict, now: float) -> str:
        if not self._connected:
            return "vhs" if self._ever_connected else "closed"
        if self._turn is not None:
            silence = now - self._received_at
            if silence > STALL_QUESTION_SECONDS:
                return "stalled-long"
            if silence > STALL_SECONDS:
                return "stalled"
            return self._turn.get("phase") or "queued"
        if now < self._done_until:
            return "done"
        if engine.get("loading"):
            return "loading"
        # An unloaded engine on a live daemon is still a quiet kitchen (the
        # headline says LIGHTS OUT); the closed face is for a daemon that is gone.
        return "quiet"

    def _headline(self, engine: dict, recent: list[dict], now: float) -> tuple[str, str]:
        if not self._connected:
            return (CLOSED_WORD, CHILLI) if not self._ever_connected else ("REDIALING...", SLATE)
        if self._turn is None and engine.get("loading"):
            return ENGINE_WORDS["loading"], MUSTARD
        if self._turn is None and engine.get("loaded"):
            return f"{QUIET_WORD} {span_text(engine.get('idle_seconds') or 0)}", TEAL
        if self._turn is None:
            return ENGINE_WORDS["unloaded"], SLATE
        best, _ = hi_score(recent, self._live_rates(now))
        return (f"HI-SCORE {best:.1f} tok/s", PINK) if best else ("first order", PINK)

    def _live_rates(self, now: float) -> list[float]:
        """The tok/s samples of the last SPARK_SAMPLES seconds, so the
        sparkline, the floor and the HI-SCORE's live half mean the window
        they are labelled with — not sixty samples of decode spread over
        whatever the pass served in the last hour."""
        return [tps for t, tps in self._spark if now - t <= SPARK_SAMPLES]

    def _show_left(self, view: TurnView | None) -> None:
        """The slip and the stub when a turn is on the pass; the card when
        not — quiet, firing up, redialing or closed."""
        now = self._clock()
        slip, stub, card = (
            self.query_one(Slip),
            self.query_one(Stub),
            self.query_one(Card),
        )
        engine = self._engine_now(now)
        config = self._document.get("config") or {}
        recent = self._document.get("recent_turns") or []
        if view is not None and self._connected:
            slip.display, card.display = True, False
            slip.show(view, len(self._behind), self._live_rates(now))
            stub.show(self._behind[0] if self._behind else None, len(self._behind) - 1, now)
            return
        slip.display, card.display = False, True
        stub.show(None, 0, now)
        slip.reset()
        wait = max(0.0, (self._retry_at or now) - now)
        if not self._connected and not self._ever_connected:
            card.show_closed(self._port, self._attempt, wait, now)
        elif not self._connected:
            card.show_redial(
                self._attempt,
                wait,
                self._why,
                local_time(self._received_at),
                now,
                self.motion,
            )
        elif engine.get("loading"):
            if self._firing_since is None:
                self._firing_since = now
            card.show_firing(
                config.get("model_id", "-"),
                engine.get("memory_gb"),
                now - self._firing_since,
                now,
                self.motion,
            )
        else:
            card.show_quiet(engine, config, recent, now, self.motion)

    def _show_tasks(self, tasks: list[dict]) -> None:
        rows = []
        for t in tasks[: 2 if self._width >= WIDE_COLUMNS else 1]:
            word = TASK_WORDS.get(t.get("state", ""), "?")
            seconds = t.get("seconds") or 0
            rows.append(
                f" TASK  {word} {task_time(seconds) if seconds else '—'}  "
                f"{(t.get('title') or '')[:70]}"
            )
        self.query_one("#tasks", Line).show("\n".join(rows) if rows else " TASK  none on the rail")

    def _refresh_footer(self) -> None:
        footer = self.query_one("#footer", Line)
        width = self._width
        now = self._clock()
        if width < NARROW_COLUMNS:
            if self._turn is not None:
                view = turn_view(self._turn, now, self._received_at)
                gen = f"{view.generated:,}" + (f"/{view.expected:,}" if view.expected else "")
                rate = f" {view.tps:.1f}t/s" if view.tps else ""
                footer.show(f" {view.word} {gen}{rate} {clock_text(view.elapsed)}  q")
            elif self._connected:
                engine = self._engine_now(now)
                footer.show(f" {QUIET_WORD} {span_text(engine.get('idle_seconds') or 0)}  q")
            else:
                footer.show(f" {CLOSED_WORD}  q")
            return
        footer.show(KEYS if width >= COMPACT_COLUMNS else KEYS_COMPACT)

    # -- timers ---------------------------------------------------------------------

    def _tick(self) -> None:
        now = self._clock()
        engine = self._document.get("engine") or {}
        config = self._document.get("config") or {}
        try:
            if self._turn is not None:
                silence = now - self._received_at
                stalled = silence > STALL_SECONDS
                view = turn_view(self._turn, now, self._received_at, stalled=stalled)
                self.query_one(Slip).tick(view, stalled=stalled, silence=silence)
                self.query_one("#line-chef", Chef).show(
                    self._mood(engine, now), now, motion=self.motion
                )
            elif engine.get("loading") and self._firing_since is not None:
                self.query_one(Card).show_firing(
                    config.get("model_id", "-"),
                    engine.get("memory_gb"),
                    now - self._firing_since,
                    now,
                    self.motion,
                )
                self.query_one("#line-chef", Chef).show("loading", now, motion=self.motion)
            if self._width < NARROW_COLUMNS:
                self._refresh_footer()
        except NoMatches:
            # A tick that lands while the screen is being torn down: nothing
            # left to draw on.
            return

    def _engine_now(self, now: float) -> dict:
        """The last document's engine block with its idle clock advanced by
        the time since that document arrived. The daemon sends no document
        while nothing moves, so between them the clock is the terminal's;
        anything that would reset it (a turn, a hold leaving) is a document."""
        engine = dict(self._document.get("engine") or {})
        idle = engine.get("idle_seconds")
        if idle is not None and self._connected:
            engine["idle_seconds"] = idle + max(0.0, now - self._received_at)
        return engine

    def _show_idle(self, engine: dict, now: float) -> None:
        """Repaint the lines that show the idle clock — the quiet card's
        span, its lights-out countdown, THE LINE's subtitle, the headline,
        the narrow footer — from the advanced clock. The card's `since …`
        subtitle rides along on the same repaint: it counts from the recent
        ring's oldest turn, not from the idle clock, but off the same `now`.
        Each widget skips text that did not change, so this is three or four
        repaints a second."""
        document = self._document
        config = document.get("config") or {}
        recent = document.get("recent_turns") or []
        queue = document.get("queue") or {}
        card = self.query_one(Card)
        if card.display and not engine.get("loading"):
            card.show_quiet(engine, config, recent, now, self.motion)
        mood = self._mood(engine, now)
        self.query_one(LinePanel).show(mood, engine, config, queue, recent, now, self.motion)
        headline, colour = self._headline(engine, recent, now)
        self.query_one(Strip).show(headline, colour)
        self._refresh_footer()

    def _idle_tick(self) -> None:
        now = self._clock()
        try:
            engine = self._engine_now(now)
            mood = self._mood(engine, now)
            self.query_one("#line-chef", Chef).show(mood, now, motion=self.motion)
            card = self.query_one(Card)
            if card.display:
                # Only the chef's motion, the pot's steam, and — disconnected
                # — the redial/closed countdown, never the rest of the
                # card's text: that only a real document, a retry or the
                # idle clock's next second changes.
                wait = max(0.0, (self._retry_at or now) - now) if not self._connected else None
                card.tick(now, motion=self.motion, wait=wait)
            if self._connected and self._turn is None and engine.get("idle_seconds") is not None:
                second = int(engine["idle_seconds"])
                if second != self._idle_shown:
                    self._idle_shown = second
                    # The whole second, so the span and the countdown it is
                    # subtracted from never disagree by one.
                    engine["idle_seconds"] = float(second)
                    self._show_idle(engine, now)
        except NoMatches:
            return

    # -- keys ------------------------------------------------------------------------

    async def action_quit(self) -> None:
        self.exit(return_code=0 if self._ever_connected else 1)

    def action_pause(self) -> None:
        paused = self.query_one(Rail).toggle_pause()
        self.query_one("#gingham", Rule)._title = (
            "RECENT ORDERS · paused" if paused else "RECENT ORDERS"
        )
        self.query_one("#gingham").refresh()

    def action_legend(self) -> None:
        self.push_screen(Overlay(LEGEND, "the legend"))

    def action_about(self) -> None:
        self.push_screen(Overlay(ABOUT, "about"))

    def action_ticket(self) -> None:
        self._open_ticket(self.query_one(Rail).selected())

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # The rail has focus and took the enter for itself: same card.
        key = event.row_key.value if event.row_key is not None else None
        self._open_ticket(self.query_one(Rail).selected(key))

    def _open_ticket(self, turn: dict | None) -> None:
        if turn is not None:
            self.push_screen(Overlay(ticket_text(turn), f"ORDER № {turn.get('id', '')[:10]}"))

    def action_reconnect(self) -> None:
        self._retry_now.set()

    def action_motion(self) -> None:
        self.motion = not self.motion
        if self._document:
            self._apply(self._document, resize=True)
        self.notify("motion off" if not self.motion else "motion on", timeout=1.5)


def run_top(port: int) -> int:
    """`sous top`: the app on the alternate screen until q, Esc or Ctrl-C;
    1 when no daemon ever answered, 0 otherwise."""
    app = Top(lambda: sse_feed(port), port=port)
    app.run()
    return app.return_code or 0
