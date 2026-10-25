"""kv8 on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 25 minutes, 1512 s measured, most of it four model
loads and seven cold prefills of the ~57.5K-token prompt with the prompt cache off):
- With the tile active and the default 131072 window, an engine reports kv8 active, with its probe
  time printed, its floor at N8 and its ceiling the largest step boundary whose mirror fits
  beside the bf16 KV; the test skips, naming the reason, where kv8 cannot be active.
- Greedy output with the DFlash2 drafter equals output without it, at block 5 and at block 8, on
  a prompt whose generation crosses the floor, on one whose generation straddles the step
  boundary at or past 57K keys, and on one whose generation crosses a ceiling lowered for it.
- Every decode's gauges carry the window's state, with the rows it quantised: `served` on the
  first two prompts and `capped` on the third, the same with the drafter on and off; the
  counters show decode and verify served from the mirror, trims followed, one window per
  decode, one cap per capped decode, and nothing declined or out of scope on the tile.
- With the drafter off, the far prompt's decode steps (peak read from the first token) peak
  between half the mirror at its capacity and that mirror plus one layer's and 0.5 GiB above
  the same steps with kv8 off (the mirror is resident, and written in place, not copied).
- An engine with kv8 off, built after the kv8 engines unloaded, reproduces the output of one built
  before any kv8 engine ran in the process, reports `off` on every decode and makes no kv8 call.
"""

import contextlib
import math
from collections.abc import Generator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores, kv8, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# The daemon's default [model].max_context_tokens: the window kv8's ceiling is sized against.
WINDOW = 131072
BLOCKS = (5, 8)
SLACK = 16
# The default model's full-attention layers, for the memory bound.
FULL_LAYERS, KV_HEADS, HEAD_DIM = 16, 4, 256
# The window state each prompt's decode must end in, drafter on or off.
EXPECTED = {"floor": "served", "far": "served", "cap": "capped"}


@contextlib.contextmanager
def _engine(*, kv_int8: bool, drafter: bool) -> Generator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=5 if drafter else 0,
        attention_tile=True,
        projection_kernel=True,
        kv_int8=kv_int8,
        kv_int8_window=WINDOW if kv_int8 else 0,
    )
    try:
        yield engine
    finally:
        engine.unload()


@contextlib.contextmanager
def _ceiling(line: int | None) -> Generator[None]:
    """kv8's ceiling set to `line` for the calls inside, then put back; None leaves it alone.
    The seams read the ceiling on every call, so a loaded engine takes the new line at once."""
    saved = kv8._ceiling
    if line is not None:
        kv8._ceiling = line
    try:
        yield
    finally:
        kv8._ceiling = saved


def _fits(boundary: int) -> bool:
    """Whether a mirror ending at this boundary fits beside the turn's bf16 KV in WINDOW:
    the bf16 KV up to the capacity, the mirror at it, one layer's old mirror during a growth,
    and the rows a straddling verify may hold past the line plus one capacity step."""
    cap = math.ceil(boundary / 256) * 256
    return 1.504 * cap + 0.504 * cap / 16 + 272 <= WINDOW


def _targets(splits: int) -> dict[str, tuple[int, tuple[int, ...], int | None]]:
    """name -> (prompt tokens, max_tokens values, ceiling for these calls or None), placed
    against the floor and the step boundaries so the test holds wherever N8 sits."""
    reach = kv8.N8 + 256 + tileattn.GRAN * splits  # no step is wider than GRAN * splits keys
    line = next(b for b in tileattn.boundaries(reach, splits) if b >= kv8.N8 + 256)
    return {
        "floor": (kv8.N8 - 86, (520, 521), None),
        "far": (tileattn.far_mark(splits) - 40, (160,), None),
        "cap": (line - 100, (400,), line),
    }


def _corpus(engine, n):
    files = sorted((ROOT / "src" / "sous").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ids = engine._encode("".join(f.read_text() for f in files))
    assert len(ids) >= n, len(ids)
    return ids[:n]


def _prompt(engine, ids, target):
    """(a one-message conversation whose rendered prompt is target tokens within SLACK, its
    rendered length)."""
    k, n, messages = target, 0, []
    for _ in range(8):
        messages = [{"role": "user", "content": HEAD + engine._tokenizer.decode(ids[:k])}]
        n = engine.count_tokens(messages, [])
        if abs(n - target) <= SLACK // 2:
            break
        k += target - n
    assert abs(n - target) <= SLACK, (target, n)
    return messages, n


def _generate(engine, prompts, peaks=None):
    """({(prompt, max_tokens): text}, {(prompt, max_tokens): (kv8 state, kv8 rows)},
    kv8 counter deltas, tile counter deltas, {(prompt, max_tokens): tokens generated}) at
    the engine's current block. With `peaks` a dict, each call's peak Metal memory above
    what was resident before it (the loaded weights) lands there under its
    (prompt, max_tokens)."""
    kv8_before, tile_before = dict(kv8.calls), dict(tileattn.calls)
    texts, gauges, generated = {}, {}, {}
    for name, (messages, _, caps, line) in prompts.items():
        for cap in caps:
            last: list[int] = [0]

            def on_delta(delta, last=last):  # on the generation thread: no blocking, no raise
                last[0] = delta.output_tokens

            resident = mx.get_active_memory()
            mx.reset_peak_memory()
            with _ceiling(line):
                texts[(name, cap)] = engine.generate(messages, [], cap, on_delta)
            if peaks is not None:
                peaks[(name, cap)] = (mx.get_peak_memory() - resident, resident)
            spec = engine.speculation()
            gauges[(name, cap)] = (spec.kv8, spec.kv8_rows)
            generated[(name, cap)] = last[0]
    kv8_counts = {k: v - kv8_before[k] for k, v in kv8.calls.items()}
    tile_counts = {k: v - tile_before[k] for k, v in tileattn.calls.items()}
    return texts, gauges, kv8_counts, tile_counts, generated


def _decode_peak(engine, messages, max_tokens):
    """Peak Metal memory, in bytes, of one generate() from its first token on. The peak is
    reset on the first delta, which follows the one-row forward that ends the prefill (the
    served forward that builds the mirror), so the prefill's scratch and the KVCache's growth
    steps are not in it and the mirror is already resident when it starts."""
    fired = []

    def on_delta(delta):  # on the generation thread, in the decode loop: no blocking, no raise
        if not fired:
            fired.append(True)
            mx.reset_peak_memory()

    engine.generate(messages, [], max_tokens, on_delta)
    assert fired, "the decode produced no token"
    return mx.get_peak_memory()


def _rows_range(key, prompts, generated, drafter):
    """(least, most or None) rows a decode's window has quantised per full-attention layer,
    from the prompt's rendered length n and the tokens the decode generated, g. mlx-vlm's
    generate_step runs each token's forward before it yields that token, so a decode,
    stopped by an end of turn or by max_tokens, ends with n + g keys. Where it serves
    throughout, the mirror is built over the prefix at the first served call and follows
    every key after it: exactly n + g rows with the drafter off. A drafter's rollbacks
    re-quantise rows and its last token need not be fed back, so with one n + g - 1 is only
    a floor. Where the ceiling cuts it, the mirror stops at the line: at most that many
    with the drafter off (a drafter's re-quantised rows below the line can add more)."""
    name, _ = key
    _, n, _, line = prompts[name]
    if line is None:
        end = n + generated[key]
        return (end - 1, None) if drafter else (end, end)
    return line - SLACK, None if drafter else line


def _check_crossing(prompts, generated, splits, label):
    """Each decode's last key reached its prompt's line by SLACK: an early end of turn is
    reported as such, before any kv8 gauge is read."""
    lines = {"floor": kv8.N8, "far": tileattn.far_mark(splits)}
    for key, g in generated.items():
        name, cap = key
        _, n, _, line = prompts[name]
        mark = line if line is not None else lines[name]
        assert n + g - 1 >= mark + SLACK, (
            f"{label}: generation ended before clearing the line: {key} prompt {n} tokens,"
            f" generated {g} of {cap}, last key {n + g - 1} < line {mark} + {SLACK}"
        )


def _check_status(status, splits):
    assert status["state"] == "active", status
    assert status["reason"] is None, status
    assert isinstance(status["probe_seconds"], float) and status["probe_seconds"] > 0, status
    assert status["floor"] == kv8.N8, status
    assert status["ceiling"] == kv8.ceiling(WINDOW, splits), status
    marks = tileattn.boundaries(WINDOW, splits)
    assert status["floor"] in marks and status["ceiling"] in marks, status
    assert status["ceiling"] > kv8.N8, status
    # The ceiling's memory rule on both sides: the ceiling fits, the next boundary does not.
    assert _fits(status["ceiling"]), status
    later = [b for b in marks if b > status["ceiling"]]
    assert not later or not _fits(later[0]), (status, later[:1])


def test_greedy_output_with_the_drafter_equals_output_without_it_on_kv8():
    avail = tileattn.availability()
    if not avail.available:
        pytest.skip(f"attention tile unavailable: {avail.reason}")
    cores, why = gpucores.matched_cores(str(mx.device_info()["device_name"]))
    if cores not in tileattn.MEASURED_SPLITS:
        pytest.skip(f"attention tile unavailable: {why or f'split target {cores} not measured'}")
    kernel = kv8.availability()
    if not kernel.available:
        pytest.skip(f"kv8 unavailable: {kernel.reason}")

    # First in the process, before any kv8 engine could open a window: the bf16 tile's paths.
    with _engine(kv_int8=False, drafter=True) as stock:
        assert stock.kv_int8_status["state"] == "off", stock.kv_int8_status
        assert stock.attention_tile_status["state"] == "active", stock.attention_tile_status
        splits = stock.attention_tile_status["splits"]
        targets = _targets(splits)
        ids = _corpus(stock, max(target for target, _, _ in targets.values()) + 64)
        prompts = {
            name: (*_prompt(stock, ids, target), caps, line)
            for name, (target, caps, line) in targets.items()
        }
        reference, reference_gauges, _, _, reference_generated = _generate(stock, prompts)

    with _engine(kv_int8=True, drafter=False) as plain:
        status = plain.kv_int8_status
        if status["state"] != "active":
            pytest.skip(f"kv8 {status['state']}: {status['reason']}")
        print(
            f"kv8 probe {status['probe_seconds']:.2f} s,"
            f" floor {status['floor']}, ceiling {status['ceiling']} at window {WINDOW}"
        )
        _check_status(status, splits)
        far_end = targets["far"][0] + max(targets["far"][1])
        assert far_end + SLACK < status["ceiling"], (far_end, status)
        peaks: dict = {}
        undrafted, plain_gauges, plain_counts, plain_tile, plain_generated = _generate(
            plain, prompts, peaks
        )
        print(
            "generated tokens, drafter off:"
            + "".join(
                f" {k[0]}/{k[1]} prompt {prompts[k[0]][1]} generated {g} kv8 {plain_gauges[k]};"
                for k, g in plain_generated.items()
            )
        )
        # After _generate, which has taken its counter deltas: the far prompt twice more, with
        # kv8 on and with the flag clear (decode() then opens no window and the tile serves
        # every row). Outputs and gauges are discarded; only the decode-only peaks are kept.
        far_messages, _, far_caps, _ = prompts["far"]
        peak_on = _decode_peak(plain, far_messages, max(far_caps))
        kv8._active = 0
        try:
            peak_off = _decode_peak(plain, far_messages, max(far_caps))
        finally:
            kv8._active = 1

    drafted, drafted_gauges, drafted_counts, drafted_tile, drafted_generated = {}, {}, {}, {}, {}
    with _engine(kv_int8=True, drafter=True) as engine:
        _check_status(engine.kv_int8_status, splits)
        assert engine._draft_block_size == 5
        for block in BLOCKS:
            engine._draft_block_size = block
            (
                drafted[block],
                drafted_gauges[block],
                drafted_counts[block],
                drafted_tile[block],
                drafted_generated[block],
            ) = _generate(engine, prompts)

    # unload() cleared the flag: the off engine below starts from none.
    assert kv8._active == 0

    with _engine(kv_int8=False, drafter=True) as off:
        assert off.kv_int8_status["state"] == "off", off.kv_int8_status
        again, again_gauges, off_counts, _, again_generated = _generate(off, prompts)

    # Parity: greedy output is the same with the drafter on and off, at every block.
    for block in BLOCKS:
        for key, text in undrafted.items():
            assert drafted[block][key] == text, (f"block {block}", *key)
    assert again == reference

    # Every decode reached its line: a short reply is named here, not read as a kv8 state.
    _check_crossing(prompts, reference_generated, splits, "stock")
    _check_crossing(prompts, plain_generated, splits, "kv8, drafter off")
    for block in BLOCKS:
        _check_crossing(prompts, drafted_generated[block], splits, f"kv8, block {block}")
    _check_crossing(prompts, again_generated, splits, "off again")

    # The gauges: each decode's window state and the rows it quantised. A served or capped
    # window first built its mirror over the whole prefix, then followed the generation.
    for key, (state, rows) in plain_gauges.items():
        assert state == EXPECTED[key[0]], (key, state, rows)
        least, most = _rows_range(key, prompts, plain_generated, drafter=False)
        assert rows >= least and (most is None or rows <= most), (key, state, rows, least, most)
    for block in BLOCKS:
        for key, (state, rows) in drafted_gauges[block].items():
            assert state == plain_gauges[key][0], (f"block {block}", key, state)
            least, _ = _rows_range(key, prompts, drafted_generated[block], drafter=True)
            assert rows >= least, (f"block {block}", key, state, rows, least)
    for gauges in (reference_gauges, again_gauges):
        assert set(gauges.values()) == {("off", 0)}, gauges

    # The counters. One decode per call (the prompt cache is off), so one window per call, and
    # one cap for the one capped prompt per pass.
    calls = sum(len(caps) for _, _, caps, _ in prompts.values())
    capped = sum(
        len(caps) for name, (_, _, caps, _) in prompts.items() if EXPECTED[name] == "capped"
    )
    assert plain_counts["windows"] == calls, plain_counts
    assert plain_counts["windows_capped"] == capped, plain_counts
    assert plain_counts["decode_kv8"] > 0, plain_counts
    assert plain_counts["mirror_builds"] > 0, plain_counts
    assert plain_counts["mirror_rows"] > 0, plain_counts
    assert plain_counts["verify_kv8"] == 0, plain_counts
    for block, counts in drafted_counts.items():
        assert counts["windows"] == calls, (block, counts)
        assert counts["windows_capped"] == capped, (block, counts)
        assert counts["decode_kv8"] > 0, (block, counts)
        assert counts["verify_kv8"] > 0, (block, counts)
        assert counts["verify_rows_kv8"] > 0, (block, counts)
        assert counts["verify_launches_kv8"] > 0, (block, counts)
        assert counts["mirror_trims"] > 0, (block, counts)
    assert drafted_tile[8]["verify_t8"] > 0, drafted_tile
    assert drafted_tile[8]["verify_straddles"] > 0, drafted_tile
    for counts in (plain_tile, *drafted_tile.values()):
        assert counts["decode_out"] == counts["verify_out"] == counts["verify_declined"] == 0
    assert not any(off_counts.values()), off_counts

    # Memory, drafter off: the far prompt's decode steps with kv8 on against with it off, each
    # peak read from the first token on. Both hold the weights and the bf16 KV identically, so
    # the difference is the mirror: its buffers at (within one 256-row step of) the capacity the
    # first build allocates, which rounds up from the prompt length plus GROWTH. The
    # lower bound proves the instrument sees the mirror at all (a peak set before the mirror
    # existed, as in the prefill, would read 0). The upper one catches a slice update that
    # copied the buffer instead of writing in place, which would add at least one more
    # mirror-sized buffer (about 1.9 GiB here) beyond the layer's old one and the slack for
    # scratch the two paths do not share.
    far = max(k for k in peaks if k[0] == "far")
    resident = peaks[far][1]
    keys = targets["far"][0] + far[1] + SLACK
    mirror_cap = math.ceil((keys + kv8.GROWTH) / 256) * 256
    mirror_bytes = kv8.BYTES_PER_ROW_PER_HEAD * KV_HEADS * FULL_LAYERS * mirror_cap
    layer_mirror = mirror_bytes // FULL_LAYERS
    low, high = mirror_bytes // 2, mirror_bytes + layer_mirror + (1 << 29)
    diff = peak_on - peak_off
    print(
        f"kv8 whole-decode peak above the {resident / 2**30:.2f} GiB resident weights, drafter off:"
        + "".join(f" {k[0]}/{k[1]} {v[0] / 2**30:.2f} GiB;" for k, v in peaks.items())
        + f" far decode-only peak, kv8 on {peak_on / 2**30:.2f} GiB, off {peak_off / 2**30:.2f}"
        f" GiB, on - off {diff / 2**30:.2f} GiB; bounds {low / 2**30:.2f} to"
        f" {high / 2**30:.2f} GiB = 0.5 x mirror {mirror_bytes / 2**30:.2f} to"
        f" mirror + layer {layer_mirror / 2**30:.2f} + 0.50"
    )
    assert low <= diff <= high, (peak_on, peak_off, diff, low, high, mirror_bytes, keys)
