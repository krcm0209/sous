"""The attention tile on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 30 minutes, most of it five cold prefills of the
~57.5K-token prompt with the prompt cache off):
- An engine with the tile reports it active; the test skips, naming the reason, where it cannot be.
- Greedy output with the DFlash2 drafter equals output without it, at block 3 and at block 8 (set
  on the engine directly: config caps the block at 5), on a ~1,450-token prompt whose generation
  crosses N0 (1,536 keys) and the first step boundary above it (1,921), and on a prompt ending just
  below the 57,600-key step boundary.
- Over the tile runs, the counters show verify rows below N0, a T = 2 round, straddling verify
  calls and, at block 8, T = 8 on the tile, with nothing declined or out of scope.
- An engine with the tile off, built after the tile engines unloaded, reproduces the output of
  one built before any tile engine ran in the process, which is stock's.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import gpucores, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# name -> (prompt tokens, max_tokens values). A response's last round verifies T = 2 only when
# the cap leaves exactly one token to emit; block-3 rounds emit one to three tokens, so one of
# three consecutive caps lands on it.
PROMPTS = {"short": (1450, (520, 521, 522)), "long": (57_560, (160,))}
SLACK = 16


@contextlib.contextmanager
def _engine(*, tile: bool, drafter: bool) -> Iterator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=3 if drafter else 0,
        attention_tile=tile,
    )
    try:
        yield engine
    finally:
        engine.unload()


def _corpus(engine, n):
    files = sorted((ROOT / "src" / "sous").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ids = engine._encode("".join(f.read_text() for f in files))
    assert len(ids) >= n, len(ids)
    return ids[:n]


def _prompt(engine, ids, target):
    """A one-message conversation whose rendered prompt is target tokens, within SLACK."""
    k, n, messages = target, 0, []
    for _ in range(8):
        messages = [{"role": "user", "content": HEAD + engine._tokenizer.decode(ids[:k])}]
        n = engine.count_tokens(messages, [])
        if abs(n - target) <= SLACK // 2:
            break
        k += target - n
    assert abs(n - target) <= SLACK, (target, n)
    return messages


def _generate(engine, prompts, blocks):
    """({(block, prompt, max_tokens): text}, {block: tile counter deltas}); block 0 keeps the
    engine's own block size."""
    texts, counts = {}, {}
    for block in blocks:
        if block:
            engine._draft_block_size = block
        before = dict(tileattn.calls)
        for name, (messages, caps) in prompts.items():
            for cap in caps:
                texts[(block, name, cap)] = engine.generate(messages, [], cap)
        counts[block] = {k: v - before[k] for k, v in tileattn.calls.items()}
    return texts, counts


def test_greedy_output_with_the_drafter_equals_output_without_it_on_the_tile():
    avail = tileattn.availability()
    if not avail.available:
        pytest.skip(f"attention tile unavailable: {avail.reason}")
    cores, why = gpucores.matched_cores(str(mx.device_info()["device_name"]))
    if cores not in tileattn.MEASURED_SPLITS:
        pytest.skip(f"attention tile unavailable: {why or f'split target {cores} not measured'}")

    # First in the process, before any tile engine could install the decode hook: stock's paths.
    with _engine(tile=False, drafter=True) as stock:
        assert stock.attention_tile_status["state"] == "off", stock.attention_tile_status
        ids = _corpus(stock, max(target for target, _ in PROMPTS.values()) + 64)
        prompts = {
            name: (_prompt(stock, ids, target), caps) for name, (target, caps) in PROMPTS.items()
        }
        reference, _ = _generate(stock, prompts, (3,))

    with _engine(tile=True, drafter=False) as plain:
        status = plain.attention_tile_status
        if status["state"] != "active":
            pytest.skip(f"attention tile {status['state']}: {status['reason']}")
        undrafted, plain_counts = _generate(plain, prompts, (0,))

    with _engine(tile=True, drafter=True) as drafted:
        assert drafted.attention_tile_status["state"] == "active", drafted.attention_tile_status
        drafted_texts, drafted_counts = _generate(drafted, prompts, (3, 8))

    with _engine(tile=False, drafter=True) as off:
        assert off.attention_tile_status["state"] == "off", off.attention_tile_status
        assert tileattn._active_splits == 0
        again, off_counts = _generate(off, prompts, (3,))

    for (_, name, cap), text in undrafted.items():
        assert drafted_texts[(3, name, cap)] == text, ("block 3", name, cap)
        assert drafted_texts[(8, name, cap)] == text, ("block 8", name, cap)
    assert again == reference
    tile = {
        k: plain_counts[0][k] + drafted_counts[3][k] + drafted_counts[8][k] for k in tileattn.calls
    }
    assert plain_counts[0]["decode_tile"] > 0, plain_counts
    assert tile["verify_rows_stock"] > 0, tile
    assert tile["verify_t2"] > 0, tile
    assert tile["verify_straddles"] > 0, tile
    assert drafted_counts[8]["verify_t8"] > 0, drafted_counts
    assert tile["decode_out"] == tile["verify_out"] == tile["verify_declined"] == 0, tile
    assert not any(off_counts[3].values()), off_counts
