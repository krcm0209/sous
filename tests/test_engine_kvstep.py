"""The decode's KV growth step: VLMEngine.decode() grows its full-attention
caches by DECODE_KV_STEP rows and hands them back at mlx-vlm's own step, on
mlx-vlm's real KVCache (CPU mlx suffices), and the prompt cache prices the
capacity a decode leaves behind. Engines are built without weights, so these
run in CI."""

import sys
import threading
import types

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402 — after the guard

from sous.engine import draftctx, kv8, projtile  # noqa: E402
from sous.engine.promptcache import (  # noqa: E402
    PrefixCache,
    PromptMemo,
    Speculation,
    fork_copy,
    slot_bytes,
)
from sous.engine.vlm import VLMEngine  # noqa: E402
from sous.engine.window import DECODE_KV_STEP  # noqa: E402

# mlx-vlm's own step, which every cache outside a decode keeps.
DEFAULT_STEP = 256
HEADS, DIM = 1, 8


@pytest.fixture(autouse=True)
def _windows_shut(monkeypatch):
    """No kv8 or projection-tile window opens around these decodes."""
    monkeypatch.setattr(kv8, "_active", 0)
    monkeypatch.setattr(kv8, "_window", None)
    monkeypatch.setattr(projtile, "_active", 0)
    monkeypatch.setattr(projtile, "_window", None)


def _engine() -> VLMEngine:
    """A VLMEngine holding only what decode() and the prompt cache's hooks read:
    no weights, no drafter, positions left to the model, and no memory
    readings, so a publish never meets the pressure valve."""
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        config=types.SimpleNamespace(eos_token_id=0)
    )
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()
    engine._positional = False
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._draft_context = draftctx.OFF
    engine._speculation = Speculation()
    engine.headroom = lambda: None  # ty: ignore[invalid-assignment]
    engine.pressure = lambda: None  # ty: ignore[invalid-assignment]
    return engine


def _result(text: str):
    return types.SimpleNamespace(is_draft=False, text=text, generation_tokens=1, finish_reason=None)


def _stub_mlx_vlm(monkeypatch, stream_generate) -> None:
    monkeypatch.setitem(
        sys.modules, "mlx_vlm", types.SimpleNamespace(stream_generate=stream_generate)
    )


def _rows(n: int, seed: int):
    k = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)
    v = mx.random.normal((1, HEADS, n, DIM), key=mx.random.key(seed + 1)).astype(mx.bfloat16)
    mx.eval(k, v)
    return k, v


class _Subclass(KVCache):
    """A KVCache subclass: its layout and growth are its own, so the decode leaves
    its step alone."""


# ---- decode ----------------------------------------------------------------------


def test_decode_widens_every_kv_cache_and_hands_it_back_at_the_default_step(monkeypatch):
    kv = [KVCache(), KVCache()]
    recurrent = ArraysCache(2)
    sub = _Subclass()
    seen: list[object] = []

    def stream_generate(model, processor, prompt, **kwargs):
        assert kwargs["prompt_cache"] is cache
        seen.append(([c.step for c in kv], sub.step, "step" in vars(recurrent)))
        yield _result("ok")

    cache = [kv[0], recurrent, kv[1], sub]
    _stub_mlx_vlm(monkeypatch, stream_generate)
    assert _engine().decode(cache, [4, 5], 8) == "ok"
    assert seen == [([DECODE_KV_STEP, DECODE_KV_STEP], DEFAULT_STEP, False)]
    for c in (*kv, sub):
        assert "step" not in vars(c) and c.step == DEFAULT_STEP
    assert "step" not in vars(recurrent)
    assert KVCache.step == DEFAULT_STEP


def test_decode_restores_the_default_step_when_the_generation_raises(monkeypatch):
    kv = [KVCache(), KVCache()]
    during: list[list[int]] = []

    def failing(model, processor, prompt, **kwargs):
        during.append([c.step for c in kv])
        yield _result("partial")
        raise RuntimeError("boom")

    _stub_mlx_vlm(monkeypatch, failing)
    with pytest.raises(RuntimeError, match="boom"):
        _engine().decode(list(kv), [4, 5], 8)
    assert during == [[DECODE_KV_STEP, DECODE_KV_STEP]]
    for c in kv:
        assert "step" not in vars(c) and c.step == DEFAULT_STEP


# ---- growth on mlx-vlm's KVCache ---------------------------------------------------


def _restored(offset: int, step: int | None):
    """A KVCache at `offset` whose capacity is exactly its offset, as a fork copy
    leaves it, every row seeded the same whatever the step."""
    c = KVCache()
    c.keys, c.values = _rows(offset, seed=1)
    c.offset = offset
    if step is not None:
        c.step = step
    return c


def _append_one_at_a_time(c, keys, values) -> int:
    """Append every row through update_and_fetch, as decode steps do; return how
    many appends grew the buffers."""
    growths = 0
    for i in range(keys.shape[2]):
        capacity = c.keys.shape[2]
        c.update_and_fetch(keys[..., i : i + 1, :], values[..., i : i + 1, :])
        mx.eval(c.keys, c.values)
        growths += c.keys.shape[2] != capacity
    return growths


def test_a_kv_cache_at_the_decode_step_grows_once_where_the_default_grows_three_times(
    monkeypatch,
):
    """600 single-row appends from 56,403 keys: three whole-cache copies at 256,
    one at 4096, and the same rows either way, since a growth copies values and
    never changes one."""
    copies: list[int] = []
    real = mx.concatenate

    def counting(arrays, *args, **kwargs):
        copies.append(1)
        return real(arrays, *args, **kwargs)

    monkeypatch.setattr(mx, "concatenate", counting)
    start, n = 56_403, 600
    keys, values = _rows(n, seed=7)
    caches, growths = {}, {}
    for step in (DEFAULT_STEP, DECODE_KV_STEP):
        copies.clear()
        caches[step] = _restored(start, step)
        growths[step] = _append_one_at_a_time(caches[step], keys, values)
        # One concatenate for K and one for V per growth: no other copy.
        assert len(copies) == 2 * growths[step]
    assert growths == {DEFAULT_STEP: 3, DECODE_KV_STEP: 1}
    wide, narrow = caches[DECODE_KV_STEP], caches[DEFAULT_STEP]
    assert wide.offset == narrow.offset == start + n
    assert wide.keys.shape[2] == start + DECODE_KV_STEP
    assert mx.array_equal(wide.keys[..., : wide.offset, :], narrow.keys[..., : narrow.offset, :])
    assert mx.array_equal(
        wide.values[..., : wide.offset, :], narrow.values[..., : narrow.offset, :]
    )


# ---- the slot a decode leaves behind -----------------------------------------------


def test_a_published_turn_slot_is_priced_at_the_capacity_its_decode_grew(monkeypatch):
    """The decode grows the turn's cache by 4096 rows and the slot it publishes
    keeps that capacity past the anchor it is trimmed back to: slot_bytes reads
    the whole buffers, so the budget counts every padded row. A copy of the slot
    (a fork copy, a retained take) holds only its offset's rows."""
    prompt, generated = 300, 20

    def stream_generate(model, processor, prompt_text, **kwargs):
        layers = kwargs["prompt_cache"]
        rows = kwargs["input_ids"].shape[1]
        for c in layers:
            c.update_and_fetch(*_rows(rows, seed=3))
        for i in range(generated):
            for c in layers:
                c.update_and_fetch(*_rows(1, seed=10 + i))
            yield _result("x")

    _stub_mlx_vlm(monkeypatch, stream_generate)
    engine = _engine()
    engine.new_cache = lambda: [KVCache(), KVCache()]  # ty: ignore[invalid-assignment]
    pc = PrefixCache(engine, enabled=True)
    stable = list(range(1, prompt + 1))
    assert pc.generate(stable, [*stable, 9001, 9002, 9003], generated) == "x" * generated

    (slot,) = pc.slots()
    for c in slot.cache:
        assert c.offset == prompt
        # 303 prompt rows in one call grow an empty cache by one step; at 256
        # that capacity would be 512.
        assert c.keys.shape[2] == DECODE_KV_STEP
        assert "step" not in vars(c)
    row_bytes = HEADS * DIM * 2 * 2  # K and V, bf16
    assert slot.nbytes == slot_bytes(slot.cache) == 2 * DECODE_KV_STEP * row_bytes
    assert slot.nbytes > 2 * prompt * row_bytes
    assert pc.stats()["resident_bytes"] == slot.nbytes

    copy: list = [KVCache(), KVCache()]
    fork_copy(slot.cache, copy, engine.copy_array)
    assert [c.keys.shape[2] for c in copy] == [prompt, prompt]
    assert slot_bytes(copy) == 2 * prompt * row_bytes
