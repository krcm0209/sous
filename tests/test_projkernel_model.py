"""The projection kernel on the real default model, greedy: drafter on against drafter off.

Acceptance bars (manual, on the M5 Pro; about 15 minutes, most of it five model loads and
drafter-off decode on the kernel):
- With the attention tile and the kernel on, an engine reports the kernel active; the test skips,
  naming the reason, where it cannot be.
- An unset block resolves to 5 with the kernel active and to 4 with it off.
- Greedy output with the DFlash2 drafter equals output without it at block 5, at blocks 8 and 12
  set on the engine directly (config caps the block at 5), and at block 0, the drafter's own
  policy. DFlash2's own policy verifies at most 5 rows (its runtime block is min(5, 8)), so block
  12 is the one whose verify rows exceed 8 and are split into runs.
- Over the kernel runs both hooks are reached, nothing is declined, and only block 12 splits.
- An engine with the kernel off, built after the kernel engines unloaded, reproduces the output
  of one built before any of them, and makes no kernel call.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from sous.config import SousConfig  # noqa: E402 — after the importorskip guard
from sous.engine import projkernel, tileattn  # noqa: E402
from sous.engine.base import fetch_model_config  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# name -> (prompt tokens, max_tokens values): one prompt whose generation stays below the
# attention tile's 1,536 keys and one above them, so the kernel is checked beside both attention
# paths; two caps on the short prompt end its last round at different row counts.
PROMPTS = {"short": (600, (256, 257)), "long": (3000, (256,))}
SLACK = 16
SPLIT_BLOCK = 12


@contextlib.contextmanager
def _engine(
    *, kernel: bool, drafter: bool, block: int = 4, explicit: bool = False
) -> Iterator[VLMEngine]:
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=0,
        draft_id=DRAFTER if drafter else "",
        draft_block_size=block if drafter else 0,
        draft_block_explicit=explicit,
        attention_tile=True,
        projection_kernel=kernel,
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


def _generate(engine, prompts):
    """({(prompt, max_tokens): text}, projkernel counter deltas) at the engine's current block."""
    before = dict(projkernel.calls)
    texts = {}
    for name, (messages, caps) in prompts.items():
        for cap in caps:
            texts[(name, cap)] = engine.generate(messages, [], cap)
    return texts, {k: v - before[k] for k, v in projkernel.calls.items()}


def test_greedy_output_with_the_drafter_equals_output_without_it_on_the_kernel():
    tile = tileattn.availability()
    if not tile.available:
        pytest.skip(f"attention tile unavailable: {tile.reason}")
    kernel = projkernel.kernel_available()
    if not kernel.available:
        pytest.skip(f"projection kernel unavailable: {kernel.reason}")
    why = projkernel.static_reason(SousConfig(), fetch_model_config(DEFAULT_27B), "dflash")
    if why is not None:
        pytest.skip(f"projection kernel unavailable: {why}")

    # Before any engine of this test activates the kernel: in a process that has run no kernel
    # engine, these are main's paths with no hook installed.
    with _engine(kernel=False, drafter=True) as stock:
        assert stock.projection_kernel_status["state"] == "off", stock.projection_kernel_status
        assert stock._draft_block_size == 4
        ids = _corpus(stock, max(target for target, _ in PROMPTS.values()) + 64)
        prompts = {
            name: (_prompt(stock, ids, target), caps) for name, (target, caps) in PROMPTS.items()
        }
        reference, _ = _generate(stock, prompts)

    with _engine(kernel=True, drafter=False) as plain:
        status = plain.projection_kernel_status
        if status["state"] != "active":
            pytest.skip(f"projection kernel {status['state']}: {status['reason']}")
        undrafted, plain_counts = _generate(plain, prompts)

    drafted, drafted_counts = {}, {}
    with _engine(kernel=True, drafter=True) as engine:
        assert engine.projection_kernel_status["state"] == "active", engine.projection_kernel_status
        assert engine._draft_block_size == 5
        for block in (5, 8, SPLIT_BLOCK):
            engine._draft_block_size = block
            drafted[block], drafted_counts[block] = _generate(engine, prompts)

    with _engine(kernel=True, drafter=True, block=0, explicit=True) as policy:
        assert policy.projection_kernel_status["state"] == "active", policy.projection_kernel_status
        assert policy._draft_block_size == 0
        assert not getattr(policy._draft, "prefer_requested_block_size", False)
        drafted[0], drafted_counts[0] = _generate(policy, prompts)

    with _engine(kernel=False, drafter=True) as off:
        assert off.projection_kernel_status["state"] == "off", off.projection_kernel_status
        assert off._draft_block_size == 4
        assert projkernel._active == 0
        again, off_counts = _generate(off, prompts)

    for block, texts in drafted.items():
        for key, text in undrafted.items():
            assert texts[key] == text, (f"block {block}", *key)
    assert again == reference
    assert plain_counts["plain_kernel"] > 0, plain_counts
    assert plain_counts["one_row"] > 0, plain_counts
    assert plain_counts["verify_kernel"] == 0, plain_counts
    for block, counts in drafted_counts.items():
        assert counts["verify_kernel"] > 0, (block, counts)
        assert counts["plain_kernel"] > 0, (block, counts)
        assert (counts["verify_split"] > 0) == (block == SPLIT_BLOCK), (block, counts)
    assert all(c["declined"] == 0 for c in (plain_counts, *drafted_counts.values()))
    assert not any(off_counts.values()), off_counts
