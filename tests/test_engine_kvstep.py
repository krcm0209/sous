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
        # Nothing the decode set is left on the instance.
        assert set(vars(c)) == {"keys", "values", "offset"}
        assert c.step == DEFAULT_STEP
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
        assert set(vars(c)) == {"keys", "values", "offset"}
        assert c.step == DEFAULT_STEP


def _padded(offset: int, capacity: int):
    c = KVCache()
    c.keys, c.values = _rows(capacity, seed=5)
    c.offset = offset
    return c


# The generation prompt the decode call prefills itself, and the DFlash block.
PROMPT_ROWS, BLOCK = 7, 8


def _decode_start(caches: list) -> list[tuple[bool, int]]:
    """What a decode does to its caches before the first round: generate_step
    appends the decode's own prompt rows, then the DFlash round loop reserves
    offset + block rows. Per cache, whether the reserve copied the buffers and
    the capacity it left."""
    from mlx_vlm.speculative.dflash import _reserve_dflash_target_cache

    for c in caches:
        c.update_and_fetch(*_rows(PROMPT_ROWS, seed=4))
    before = [c.keys for c in caches]
    _reserve_dflash_target_cache(caches, BLOCK)
    return [(c.keys is not b, c.keys.shape[2]) for c, b in zip(caches, before, strict=True)]


def _first_growth(c) -> int:
    """Rows the next growth adds, appending one row at a time as decode steps do."""
    capacity = c.keys.shape[2]
    while c.keys.shape[2] == capacity:
        c.update_and_fetch(*_rows(1, seed=6))
        mx.eval(c.keys, c.values)
    return c.keys.shape[2] - capacity


def test_a_decode_starts_without_a_copy_its_256_row_reserve_would_not_have_made(monkeypatch):
    """Three caches end a 256-step prefill at capacity 56,659: one with room past
    its prompt rows and the block, one whose prompt rows and block end exactly
    at the capacity, one 4 rows short. At mlx-vlm's step the reserve leaves the
    roomy one alone and copies the other two (56,659 is no multiple of 256, so
    an exact fit rounds up as a shortfall does), and the next growth adds 256
    rows; with the step at 4096 and mlx-vlm's own reserve the roomy one is
    copied too. Inside a decode the reserve copies only the short one, the
    exact fit included among those left alone, and the roomy one's first
    growth after it adds 4096 rows."""
    roomy, exact, short = 56_403, 56_644, 56_648
    capacity = 56_659
    assert exact + PROMPT_ROWS + BLOCK == capacity

    at_default = [_padded(n, capacity) for n in (roomy, exact, short)]
    assert _decode_start(at_default) == [(False, 56_659), (True, 56_832), (True, 56_832)]
    assert _first_growth(at_default[0]) == DEFAULT_STEP

    step_only = _padded(roomy, capacity)
    step_only.step = DECODE_KV_STEP
    assert _decode_start([step_only]) == [(True, 57_344)], "mlx-vlm's reserve stopped copying"

    seen: list[object] = []

    def stream_generate(model, processor, prompt, **kwargs):
        caches = kwargs["prompt_cache"]
        seen.append((_decode_start(caches), _first_growth(caches[0])))
        yield _result("ok")

    decoded = [_padded(n, capacity) for n in (roomy, exact, short)]
    _stub_mlx_vlm(monkeypatch, stream_generate)
    assert _engine().decode(decoded, [4, 5], 8) == "ok"
    assert seen == [([(False, 56_659), (False, 56_659), (True, 57_344)], DECODE_KV_STEP)]
    for c in decoded:
        assert set(vars(c)) == {"keys", "values", "offset"}


def test_a_layer_with_a_step_or_reserve_of_its_own_leaves_the_decode_with_them(monkeypatch):
    """The decode shadows both on the instance and puts back what it found,
    rather than dropping it: a layer that came with its own step or reserve
    keeps them, and one that came with neither leaves with neither."""
    own_step = KVCache()
    own_step.step = 512
    own_reserve = KVCache()

    def reserve(min_capacity_tokens):
        return ()

    own_reserve.prefix_cache_reserve = reserve  # ty: ignore[invalid-assignment]
    plain = KVCache()
    layers = [own_step, own_reserve, plain]
    seen: list[object] = []

    def stream_generate(model, processor, prompt, **kwargs):
        seen.append([(c.step, c.prefix_cache_reserve is reserve) for c in layers])
        yield _result("ok")

    _stub_mlx_vlm(monkeypatch, stream_generate)
    assert _engine().decode(list(layers), [4, 5], 8) == "ok"
    assert seen == [[(DECODE_KV_STEP, False)] * 3]
    assert vars(own_step)["step"] == 512 and "prefix_cache_reserve" not in vars(own_step)
    assert vars(own_reserve)["prefix_cache_reserve"] is reserve and "step" not in vars(own_reserve)
    assert set(vars(plain)) == {"keys", "values", "offset"}


def _raising(name: str):
    def raise_(*args, **kwargs):
        raise RuntimeError(name)

    return raise_


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (kv8, "open_window"),
        (projtile, "open_window"),
        (projtile, "close_window"),
        (kv8, "close_window"),
    ],
    ids=["kv8-open", "tile-open", "tile-close", "kv8-close"],
)
def test_the_step_goes_back_whatever_a_window_open_or_close_does(monkeypatch, module, name):
    """The step is set before either window opens and put back after both
    close, on every exit: an open that raises leaves no widened layer behind
    for the next prefill to grow by, and a close that raises does not skip the
    restore."""
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(module, name, _raising(name))
    kv = [KVCache(), KVCache()]
    seen: list[object] = []

    def stream_generate(model, processor, prompt, **kwargs):
        seen.append([c.step for c in kv])
        yield _result("ok")

    _stub_mlx_vlm(monkeypatch, stream_generate)
    with pytest.raises(RuntimeError, match=name):
        _engine().decode(list(kv), [4, 5], 8)
    assert seen == ([] if name == "open_window" else [[DECODE_KV_STEP, DECODE_KV_STEP]])
    for c in kv:
        assert set(vars(c)) == {"keys", "values", "offset"}
        assert c.step == DEFAULT_STEP


def test_decode_clears_mlx_buffer_cache_once_per_growth_and_never_otherwise(monkeypatch):
    """A growth frees every layer's old K and V into mlx's buffer cache, where
    nothing on the DFlash path releases them before the generation ends. The
    decode reads the capacity at every emitted token and clears once when it
    moved, and leaves the cache alone through a generation that never grew."""
    cleared: list[int] = []
    emitted: list[str] = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(len(emitted)))

    def generating(generated: int):
        def stream_generate(model, processor, prompt, **kwargs):
            caches = kwargs["prompt_cache"]
            for c in caches:
                c.update_and_fetch(*_rows(kwargs["input_ids"].shape[1], seed=3))
            for i in range(generated):
                for c in caches:
                    c.update_and_fetch(*_rows(1, seed=10 + i))
                emitted.append("x")
                yield _result("x")

        return stream_generate

    # Capacity 20 at offset 10: two prompt rows and eight tokens fill it, the
    # ninth token grows it by one step, and the last three fit there.
    grows = [_padded(10, 20), _padded(10, 20)]
    _stub_mlx_vlm(monkeypatch, generating(12))
    assert _engine().decode(grows, [4, 5], 12) == "x" * 12
    assert cleared == [9]
    assert [c.keys.shape[2] for c in grows] == [20 + DECODE_KV_STEP] * 2

    cleared.clear()
    emitted.clear()
    roomy = [_padded(10, 4096), _padded(10, 4096)]
    _stub_mlx_vlm(monkeypatch, generating(12))
    assert _engine().decode(roomy, [4, 5], 12) == "x" * 12
    assert cleared == []


def test_decode_kv_step_status_counts_the_layers_the_step_covers():
    """The load line's token: how many of the model's cache layers the decode
    grows by its step, read from a cache built as every turn's is, so a
    checkpoint whose attention layers are a KVCache subclass reads 0 rather
    than silently keeping mlx-vlm's step behind a line that names 4096."""
    engine = _engine()
    engine.new_cache = lambda: [KVCache(), ArraysCache(2), KVCache(), _Subclass()]  # ty: ignore[invalid-assignment]
    assert engine.decode_kv_step_status == {"step": DECODE_KV_STEP, "layers": 2}
    other = _engine()
    other.new_cache = lambda: [_Subclass(), ArraysCache(2)]  # ty: ignore[invalid-assignment]
    assert other.decode_kv_step_status == {"step": DECODE_KV_STEP, "layers": 0}


# ---- prefill ----------------------------------------------------------------------


def _kv_layers_of(layers: list) -> list:
    return [c for c in layers if type(c) is KVCache]


def _shadow(c) -> set[str]:
    """What decode() sets on a KVCache instance and must take off again."""
    return set(vars(c)) & {"step", "prefix_cache_reserve"}


def test_every_prefill_grows_at_the_default_step_and_only_the_decode_widens(monkeypatch):
    """A cold turn on a hybrid cache prefills to its fork stop, copies the fork,
    prefills to the anchor and decodes; a warm turn then extends the turn slot it
    left. Every prefill() call, the fork stop's included, runs with mlx-vlm's own
    step and nothing on the instance, since the fork store saves the padded
    buffers a prefill grew. Only the decode sets the step and the reserve (the
    rows the decode call prefills itself grow by its step), and no slot it
    leaves carries either."""
    boundary, prompt, generated = 4_100, 4_300, 60
    prefills: list[list[tuple[int, set[str]]]] = []
    decodes: list[list[tuple[int, set[str]]]] = []
    grown: list[list[int]] = []

    def append(layers, rows: int, seed: int) -> None:
        for c in _kv_layers_of(layers):
            c.update_and_fetch(*_rows(rows, seed=seed))
            mx.eval(c.keys, c.values)

    def generate(model, processor, prompt_text, **kwargs):
        layers = kwargs["prompt_cache"]
        assert kwargs["max_tokens"] == 0
        prefills.append([(c.step, _shadow(c)) for c in _kv_layers_of(layers)])
        append(layers, kwargs["input_ids"].shape[1], seed=20)
        grown.append([c.keys.shape[2] for c in _kv_layers_of(layers)])

    def stream_generate(model, processor, prompt_text, **kwargs):
        layers = kwargs["prompt_cache"]
        decodes.append([(c.step, _shadow(c)) for c in _kv_layers_of(layers)])
        append(layers, kwargs["input_ids"].shape[1], seed=21)
        for i in range(generated):
            append(layers, 1, seed=30 + i)
            yield _result("x")

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(generate=generate, stream_generate=stream_generate),
    )
    engine = _engine()
    engine.new_cache = lambda: [KVCache(), ArraysCache(2), KVCache()]  # ty: ignore[invalid-assignment]
    pc = PrefixCache(engine, enabled=True, max_bytes=1 << 30)
    stable = list(range(1, prompt + 1))
    assert pc.generate(stable, [*stable, 9001, 9002, 9003], generated, fork_at=[boundary]) == (
        "x" * generated
    )
    # The fork stop's prefill, then the anchor's, then the decode; 4,100 rows on an
    # empty cache grow it by 17 steps of 256, where 4096 would leave 8,192.
    assert len(prefills) == 2 and len(decodes) == 1
    assert [s.kind for s in pc.slots()].count("fork") == 1
    assert grown[0] == [17 * DEFAULT_STEP] * 2

    later = [*stable, 9001, 9002, 9003, *range(10_001, 10_301)]
    assert pc.generate(later, [*later, 9004, 9005, 9006], generated) == "x" * generated
    assert len(prefills) == 3 and len(decodes) == 2
    for seen in prefills:
        assert seen == [(DEFAULT_STEP, set()), (DEFAULT_STEP, set())]
    for seen in decodes:
        assert seen == [(DECODE_KV_STEP, {"step", "prefix_cache_reserve"})] * 2
    for slot in pc.slots():
        for c in _kv_layers_of(slot.cache):
            assert _shadow(c) == set() and c.step == DEFAULT_STEP


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
