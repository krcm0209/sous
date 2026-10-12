"""The int8 K/V mirror (kv8) through mlx-vlm's real qwen3_5 forward, exact
verifier and DFlash round loop, and through the engine's own prefill and decode.

A tiny random qwen3_5 language model at the 27B's attention shape (two
full-attention layers, from tile_fixtures) is prefilled past kv8's floor through
the real forward, with the attention tile's decode hook and verifyattn's wrapper
installed at S = 20, the floor N8 at N0 and the ceiling L at the step boundary
2,561. kv8.tile8 and tileattn.tile are the plain-mlx stand-ins on any GPU (the
real kernels under ``nax``); the quantiser is always the real one, which builds
on any Metal GPU. Every comparison with a one-row decode rests on stock's own
exactness on these ids, which test_tileattn_contract asserts.

Pinned here: both hooks reach kv8 only inside a window; every verify row equals
the one-row decode at its own key count, on the path the per-row rule gives it;
through mlx-vlm's own round loop the mirror follows partial commits and an abort
through the cache's instance trim and ends equal to quantising the committed
rows in one go; a round that straddles L keeps every row on its drafter-off path
and caps the window once; and the engine's prefill never builds a mirror while
its decode does, and closes its window whatever the generation raises.
"""

import threading
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import draftctx, kv8, tileattn  # noqa: E402 — after the importorskip guard
from sous.engine.promptcache import PromptMemo, Speculation  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402
from tests import kv8_fixtures as kfx  # noqa: E402
from tests import tile_fixtures as tfx  # noqa: E402

SPLITS = 20
LAYERS = 2  # the tiny model's full-attention layers: every forward makes two attention calls
FULL = (1, 3)  # their indices among the model's layers, and in its cache
# L for every test here: the step boundary after 1,921 at S = 20. Far below the
# ceiling of any real window, so the straddling cases stay a few rows long.
CEILING = 2561
BLOCK = 5  # the verify block the round-loop tests run, as the M5 Pro does
PROMPT = 1600  # the engine tests' prompt: its last row lies past N8 and below CEILING
# The kv8 counters a window and its forwards move; the mirror's own counters
# (builds, rows, trims, growths) are read only where a test is about them.
COUNTED = (
    "decode_kv8",
    "verify_kv8",
    "verify_rows_kv8",
    "verify_launches_kv8",
    "windows",
    "windows_capped",
)
KV8_CASES = [
    (1000, 3),  # below N0: stock rows, and a window that serves nothing
    (1533, 5),  # two stock rows, then three kv8 rows from N8 (N0 here)
    (1700, 1),
    (1700, 2),  # the DFlash round loop's last round
    (1700, 3),
    (1700, 8),
    (1717, 5),  # where a kernel that ignores the schedule breaks parity (below)
    (1918, 5),  # straddles the step boundary at 1,921 keys: two kv8 runs
    (2558, 5),  # straddles L: two kv8 rows, then three tile rows; decode caps at 2,561
]


@pytest.fixture(scope="module")
def tiny():
    lm = tfx.tiny_language_model()
    # test_tileattn_contract's ids, whose stock exactness it asserts.
    ids = mx.random.randint(0, tfx.VOCAB, (2600,), key=mx.random.key(132)).tolist()
    return lm, ids


def _serving(monkeypatch: pytest.MonkeyPatch, *, stand_in: bool = True) -> None:
    """kv8 as an active load leaves it, behind the tile's decode hook at S = 20:
    the flag raised, the small floor, CEILING as L, no window open, the stand-ins
    for both kernels (or the real ones) and the real quantiser. monkeypatch
    restores all of it, a window a failed test left open included."""
    assert CEILING in tileattn.boundaries(CEILING, SPLITS)
    tfx.hooked(monkeypatch, splits=SPLITS, stand_in=stand_in)
    kfx.small_floor(monkeypatch)
    kfx.ceiling(monkeypatch, CEILING)
    if stand_in:
        monkeypatch.setattr(kv8, "tile8", kfx.stand_in_tile8)
    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "_window", None)


def _prefill(lm, ids):
    """A fresh cache holding `ids`, prefilled in one call outside any window, and
    the greedy token after them."""
    cache = lm.make_cache()
    logits = lm(
        mx.array([ids], dtype=mx.int32),
        cache=cache,
        position_ids=mx.arange(len(ids), dtype=mx.int32)[None],
        rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
    ).logits
    mx.eval([c.state for c in cache])
    return cache, int(mx.argmax(logits[0, -1]).item())


def _step(lm, cache, token: int):
    """One one-row decode forward: the logits row after `token`, [1, vocab]."""
    logits = lm(mx.array([[token]], dtype=mx.int32), cache=cache).logits
    mx.eval(logits)
    return logits[:, -1]


def _in_window(cache, run):
    """`run()` inside a window over `cache`, opened and closed as decode() does;
    returns run's result and close_window's (state, rows)."""
    kv8.open_window()
    try:
        out = run()
    finally:
        closed = kv8.close_window(cache)
    return out, closed


def _verify_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    logits, closed = _in_window(
        cache, lambda: tfx.verify_forward(lm, cache, ids[prefix : prefix + t])[0]
    )
    return [logits[:, r] for r in range(t)], closed


def _decode_rows(lm, ids, prefix, t):
    cache = tfx.prefill_cache(lm, ids[:prefix])
    return _in_window(cache, lambda: [_step(lm, cache, tok) for tok in ids[prefix : prefix + t]])


def _greedy_off(lm, ids, prefix, count):
    """Drafter-off greedy decoding of `count` tokens after ids[:prefix]: the first
    from the prefill, each later one from a one-row forward inside one window, as
    a decode without a drafter runs. Returns the tokens and close_window's
    (state, rows)."""
    cache, first = _prefill(lm, ids[:prefix])
    tokens = [first]

    def run():
        for _ in range(count - 1):
            tokens.append(int(mx.argmax(_step(lm, cache, tokens[-1])).item()))

    _, closed = _in_window(cache, run)
    return tokens, closed


def _same(a, b):
    return [mx.array_equal(x, y).item() for x, y in zip(a, b, strict=True)]


def _delta(before, after):
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def _counted(before, after):
    return {k: v for k, v in _delta(before, after).items() if k in COUNTED}


def _nonzero(counts):
    return {k: v for k, v in counts.items() if v}


def _path(n_row: int) -> str:
    """The path the per-row rule gives a row that sees n_row keys."""
    if n_row < tileattn.N0:
        return "stock"
    if kv8.N8 <= n_row < CEILING:
        return "kv8"
    return "tile"


def _state(capped: bool, served: bool) -> str:
    """close_window's state for a window that capped or served (or neither)."""
    if capped:
        return "capped"
    return "served" if served else "idle"


def _verify_counts(prefix, t):
    """(tileattn, kv8) counter deltas for one T-row verify forward in a window of
    its own: the tile's per-call counts as before, each run's rows to the path of
    its first row (every row of a run shares it: N8 and L are step boundaries,
    and verify_plan cuts at every one)."""
    runs = [(k - j, _path(prefix + j + 1)) for j, k, _ in tileattn.verify_plan(prefix, t, SPLITS)]
    rows = {p: sum(n for n, q in runs if q == p) for p in ("stock", "tile", "kv8")}
    launches = {p: sum(1 for _, q in runs if q == p) for p in ("tile", "kv8")}
    tile = {
        "verify_calls": LAYERS,
        f"verify_t{t}": LAYERS,
        "verify_rows_stock": LAYERS * rows["stock"],
        "verify_rows_tile": LAYERS * rows["tile"],
        "verify_launches": LAYERS * launches["tile"],
        "verify_straddles": LAYERS * (len(runs) > 1),
    }
    eight = {
        "verify_kv8": LAYERS * (launches["kv8"] > 0),
        "verify_rows_kv8": LAYERS * rows["kv8"],
        "verify_launches_kv8": LAYERS * launches["kv8"],
        "windows": 1,
        "windows_capped": int(prefix + 1 >= CEILING),
    }
    return _nonzero(tile), _nonzero(eight)


def _decode_counts(prefix, t):
    """(tileattn, kv8) counter deltas for t one-row forwards in a window of their own."""
    paths = [_path(prefix + r + 1) for r in range(t)]
    tile = {
        "decode_stock": LAYERS * paths.count("stock"),
        "decode_tile": LAYERS * paths.count("tile"),
    }
    eight = {
        "decode_kv8": LAYERS * paths.count("kv8"),
        "windows": 1,
        "windows_capped": int(prefix + t >= CEILING),
    }
    return _nonzero(tile), _nonzero(eight)


def _assert_mirror_is_bulk(cache) -> None:
    """Every full-attention layer's mirror covers exactly the cache's committed
    rows and holds what quantising them in one go gives: no row left from a
    rejected draft, none missing."""
    for i in FULL:
        layer = cache[i]
        mirror = getattr(layer, kv8.MIRROR_ATTR)
        n = mirror.n
        assert n == layer.offset, (i, n, layer.offset)
        codes, sc = kv8.quantize(layer.keys[:, :, :n, :], layer.values[:, :, :n, :], n)
        assert mx.array_equal(mirror.codes[:, :, :n], codes).item(), i
        assert mx.array_equal(mirror.sc[:, :, :n], sc).item(), i


def _dropped(cache) -> bool:
    """What close_window leaves on every full-attention layer: no mirror, and the
    class's own trim again."""
    return all(
        getattr(cache[i], kv8.MIRROR_ATTR, None) is None and "trim" not in vars(cache[i])
        for i in FULL
    )


# ---- the hooks ----------------------------------------------------------------


def test_both_hooks_reach_kv8_through_the_real_forward_with_a_window_open(tiny, monkeypatch):
    """Inside a window, verify rows reach kv8 through verifyattn's wrapper and a
    one-row decode through the tile's decode hook. The abort that ends the
    verify forward trims the mirror through its wrap, and the decode step after
    it, with a token the verify did not write, re-quantises the row it replaced.
    Without a window the same forwards are the tile's and quantise nothing,
    though kv8's flag is up."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    prefix, t = 1700, 5
    other = (ids[prefix] + 1) % tfx.VOCAB

    cache = tfx.prefill_cache(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    window = kv8.open_window()
    try:
        tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
        tile_verified, kv8_verified = dict(tileattn.calls), dict(kv8.calls)
        for i in FULL:
            mirror = getattr(cache[i], kv8.MIRROR_ATTR)
            assert (mirror.serial, mirror.n) == (window.serial, prefix)
            assert mirror.cap % 256 == 0 and mirror.cap >= prefix + t
        _step(lm, cache, other)
        _assert_mirror_is_bulk(cache)
        assert window.state == "served"
    finally:
        state, rows = kv8.close_window(cache)
    assert state == "served" and rows >= prefix
    assert kv8._window is None and _dropped(cache)
    assert _delta(tile_start, tile_verified) == {"verify_calls": LAYERS, "verify_t5": LAYERS}
    assert _counted(kv8_start, kv8_verified) == {
        "windows": 1,
        "verify_kv8": LAYERS,
        "verify_rows_kv8": LAYERS * t,
        "verify_launches_kv8": LAYERS,
    }
    assert _delta(tile_verified, tileattn.calls) == {}
    assert _counted(kv8_verified, kv8.calls) == {"decode_kv8": LAYERS}

    cache = tfx.prefill_cache(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    tfx.verify_forward(lm, cache, ids[prefix : prefix + t])
    _step(lm, cache, other)
    assert kv8.calls == kv8_start
    assert _delta(tile_start, tileattn.calls) == {
        "verify_calls": LAYERS,
        "verify_t5": LAYERS,
        "verify_rows_tile": LAYERS * t,
        "verify_launches": LAYERS,
        "decode_tile": LAYERS,
    }
    assert all(getattr(cache[i], kv8.MIRROR_ATTR, None) is None for i in FULL)


@pytest.mark.slow
@pytest.mark.parametrize(
    "stand_in", [True, pytest.param(False, marks=kfx.nax)], ids=["stand-in", "kernel"]
)
@pytest.mark.parametrize(("prefix", "t"), KV8_CASES)
def test_kv8_keeps_verify_rows_equal_to_decode_through_the_real_forward(
    tiny, prefix, t, stand_in, monkeypatch
):
    """Every verify row equals the one-row decode at its own key count, on the
    path the per-row rule gives it: stock below N0, kv8 from N8 below L, the tile
    from L. A verify that straddles L serves its rows below L, keeps its mirror
    and leaves its window uncapped; the decode caps its window at the first row
    at L."""
    lm, ids = tiny
    _serving(monkeypatch, stand_in=stand_in)
    tfx.verify_ready(monkeypatch, lm)
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    verify, verify_closed = _verify_rows(lm, ids, prefix, t)
    tile_verified, kv8_verified = dict(tileattn.calls), dict(kv8.calls)
    decode, decode_closed = _decode_rows(lm, ids, prefix, t)
    assert all(_same(verify, decode)), (prefix, t)

    tile_want, kv8_want = _verify_counts(prefix, t)
    assert _delta(tile_start, tile_verified) == tile_want
    assert _counted(kv8_start, kv8_verified) == kv8_want
    served = "verify_rows_kv8" in kv8_want
    assert verify_closed[0] == _state(prefix + 1 >= CEILING, served)
    assert (verify_closed[1] > 0) == served

    tile_want, kv8_want = _decode_counts(prefix, t)
    assert _delta(tile_verified, tileattn.calls) == tile_want
    assert _counted(kv8_verified, kv8.calls) == kv8_want
    served = "decode_kv8" in kv8_want
    assert decode_closed[0] == _state(prefix + t >= CEILING, served)
    assert (decode_closed[1] > 0) == served


def test_a_kv8_kernel_that_ignores_the_schedule_breaks_parity_through_the_real_forward(
    tiny, monkeypatch
):
    """The sensitivity half: rows 1,718..1,722 share chunk 96, but a kernel that
    sizes its splits from the call's own key count (ceil(n / 20)) gives the
    verify call and the one-row call at 1,718 different partitions. The equality
    above is therefore not vacuous: the rows it compares are kv8's."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    monkeypatch.setattr(
        kv8,
        "tile8",
        lambda q, codes, sc, n, chunk: kfx.stand_in_tile8(q, codes, sc, n, -(-n // 20)),
    )
    verify, _ = _verify_rows(lm, ids, 1717, 5)
    decode, _ = _decode_rows(lm, ids, 1717, 5)
    assert not all(_same(verify, decode))


# ---- mlx-vlm's DFlash round loop ----------------------------------------------


class _Abort(Exception):
    """The walk failing part-way through a round."""


class _Scripted:
    """A DFlash drafter reduced to what the round loop touches, as
    test_draftctx's stub is, proposing the next block_size - 1 tokens of a
    precomputed greedy continuation, with a wrong token at the draft position
    `wrong` names for a round. Where it stands in the continuation is read off
    the acceptances the round loop records on it (each round commits its
    accepted drafts and a bonus), so after a rejection it proposes from the
    bonus the target chose."""

    prefer_requested_block_size = True

    def __init__(self, continuation: list[int], wrong: dict[int, int] | None = None):
        self.config = types.SimpleNamespace(target_layer_ids=[0, 2])
        self.continuation = continuation
        self.wrong = dict(wrong or {})
        self.round = -1
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def reset(self, target_model):
        return ["draft cache"]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        self.round += 1
        at = sum(int(a) + 1 for a in self.accept_lens)
        tokens = list(self.continuation[at + 1 : at + block_size])
        if self.round in self.wrong:
            q = self.wrong[self.round]
            tokens[q] = (tokens[q] + 1) % tfx.VOCAB
        return mx.array([tokens], dtype=token_dtype)

    draft_block_greedy = draft_block


def _walk(monkeypatch: pytest.MonkeyPatch, drafter: _Scripted, *, abort_round=None) -> list:
    """The round loop's own greedy walk, recording (round, drafted, accepted)
    for each round it finishes; in `abort_round` it raises _Abort instead, after
    the verify forward appended the round's rows and before the commit, so the
    loop aborts the round through SpeculativeCacheTransaction.abort."""
    from mlx_vlm.speculative import dflash

    real = dflash._speculative_walk
    record: list[tuple[int, int, int]] = []

    def walk(draft_tokens, target_tokens, budget):
        if drafter.round == abort_round:
            raise _Abort(f"round {drafter.round}")
        accepted, new_tokens = real(draft_tokens, target_tokens, budget)
        record.append((drafter.round, int(draft_tokens.shape[1]), accepted))
        return accepted, new_tokens

    monkeypatch.setattr(dflash, "_speculative_walk", walk)
    return record


def _rounds(lm, drafter: _Scripted, cache, first: int, max_tokens: int):
    """mlx-vlm's DFlash round loop over `cache`, greedy at block BLOCK, as
    generate_step hands it over after the prefill and the first bonus."""
    from mlx_vlm.speculative import dflash

    return dflash._dflash_rounds(
        types.SimpleNamespace(language_model=lm),  # ty: ignore[invalid-argument-type]
        drafter,  # ty: ignore[invalid-argument-type]
        cache,
        mx.zeros((1, 1, 2 * 256), dtype=mx.bfloat16),  # the prompt hidden: the stub never reads it
        first_bonus=first,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=BLOCK,
        greedy_sampling=True,
    )


class _Paths:
    """Which kernel served each key count, read off the two array-level calls
    every served row goes through: kv8.tile8 ("tile8") and tileattn.tile
    ("tile"). Row t of a T-row call sees n - T + t + 1 keys."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.seen: dict[int, set[str]] = {}
        for module, name in ((kv8, "tile8"), (tileattn, "tile")):
            monkeypatch.setattr(module, name, self._recording(name, getattr(module, name)))

    def _recording(self, kind, kernel):
        def run(queries, a, b, n, chunk):
            out = kernel(queries, a, b, n, chunk)
            for key_count in range(n - out.shape[1] + 1, n + 1):
                self.seen.setdefault(key_count, set()).add(kind)
            return out

        return run

    def take(self) -> dict[int, set[str]]:
        seen, self.seen = self.seen, {}
        return seen


@pytest.mark.slow
def test_the_mirror_follows_partial_commits_and_an_abort_through_the_real_round_loop(
    tiny, monkeypatch
):
    """mlx-vlm's own round loop, with kv8.serves wrapped to serve every other
    round only, as a per-round on/off harness wraps it. A served round's wrong
    draft is rejected and its rows trimmed off the cache at the commit
    (_trim_append_caches); the next round runs on the tile and writes those
    cache rows again with tokens the mirror never saw; the served round after
    it must quantise them anew. The
    clamp a served call runs first cannot see that, since by then the committed
    prefix has moved past the rejected rows: only the instance trim the commit
    reached can. The walk then raises in a served round, after its verify
    appended and quantised five rows, and the loop aborts the round through
    SpeculativeCacheTransaction.abort, which reaches the same trim. After all of
    it the mirror holds exactly what quantising the committed rows in one go
    gives."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    prefix = 1700
    cont, _ = _greedy_off(lm, ids, prefix, 40)
    drafter = _Scripted(cont, wrong={0: 1, 2: 1, 4: 1})
    serves = kv8.serves
    monkeypatch.setattr(kv8, "serves", lambda n_row: drafter.round % 2 == 0 and serves(n_row))
    record = _walk(monkeypatch, drafter, abort_round=6)
    cache, first = _prefill(lm, ids[:prefix])
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    kv8.open_window()
    try:
        with pytest.raises(_Abort):
            for _ in _rounds(lm, drafter, cache, first, max_tokens=64):
                pass
        assert [r for r, _, _ in record] == [0, 1, 2, 3, 4, 5]
        # Each served round was cut short by its wrong draft and followed by an
        # off round: the sequence the trim wrap alone keeps right.
        assert all(accepted < drafted for r, drafted, accepted in record if r % 2 == 0)
        assert cache[FULL[0]].offset == prefix + sum(a + 1 for _, _, a in record)
        _assert_mirror_is_bulk(cache)
    finally:
        kv8.close_window(cache)
    partial = sum(1 for _, drafted, accepted in record if accepted < drafted)
    moved = _delta(kv8_start, kv8.calls)
    assert moved.get("verify_kv8", 0) == LAYERS * 4  # rounds 0, 2, 4 and the aborted 6
    assert moved.get("verify_rows_kv8", 0) == LAYERS * BLOCK * 4
    # The wrap counts every trim it passes on: each partial commit, then the abort.
    assert moved.get("mirror_trims", 0) == LAYERS * (partial + 1)
    assert "decode_kv8" not in moved and "windows_capped" not in moved
    assert _delta(tile_start, tileattn.calls).get("verify_rows_tile", 0) == LAYERS * BLOCK * 3


@pytest.mark.slow
def test_a_round_that_straddles_the_ceiling_keeps_every_row_on_its_drafter_off_path(
    tiny, monkeypatch
):
    """L falls inside a verify round, and that round and the next reject drafts.
    Drafter off, one-row decode serves rows below L from the mirror and caps the
    window at the first row at L. Drafter on, the round loop commits the same
    tokens, every row it computes takes the path drafter off gave the row at
    that key count, and the window caps exactly once: at the first round whose
    first row is at or past L, never at a round that only straddles it."""
    lm, ids = tiny
    _serving(monkeypatch)
    tfx.verify_ready(monkeypatch, lm)
    paths = _Paths(monkeypatch)
    prefix, tokens = 2542, 32
    capped = kv8.calls["windows_capped"]

    cont, closed = _greedy_off(lm, ids, prefix, tokens + 1)
    off = paths.take()
    assert off == {
        n: {"tile8" if n < CEILING else "tile"} for n in range(prefix + 1, prefix + tokens + 1)
    }
    assert closed[0] == "capped" and kv8.calls["windows_capped"] == capped + 1

    # Rounds 0-2 accept every draft (rows 2,543-2,557); round 3's rows are
    # 2,558-2,562 and its second draft is wrong, so it commits 2,558-2,559;
    # round 4's rows are 2,560-2,564 and its third draft is wrong, so it commits
    # 2,560-2,562; round 5 starts at 2,563 and caps; round 7 has one draft left.
    drafter = _Scripted(cont, wrong={3: 1, 4: 2})
    record = _walk(monkeypatch, drafter)
    cache, first = _prefill(lm, ids[:prefix])
    assert first == cont[0]
    kv8.open_window()
    try:
        produced = [token for token, _ in _rounds(lm, drafter, cache, first, max_tokens=tokens)]
    finally:
        closed = kv8.close_window(cache)
    on = paths.take()
    assert produced == cont[1:tokens]
    assert [(drafted, accepted) for _, drafted, accepted in record] == [
        (4, 4),
        (4, 4),
        (4, 4),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 4),
        (1, 1),
    ]
    assert set(on) == set(range(prefix + 1, prefix + tokens + 1))
    assert on == off
    assert closed[0] == "capped" and kv8.calls["windows_capped"] == capped + 2


# ---- the engine's prefill and decode -------------------------------------------


class _Boom(Exception):
    """A generation failing after its own prefill segment was served."""


class _Wrapper:
    """The mlx-vlm model the engine holds: the tiny language model, and the
    config decode() reads the end-of-sequence id from."""

    def __init__(self, lm):
        self.language_model = lm
        self.config = types.SimpleNamespace(model_type="qwen3_5", eos_token_id=0)


def _engine(lm) -> VLMEngine:
    """A VLMEngine with no load behind it, as test_draftctx builds one: every
    attribute prefill() and decode() read, set by hand. No drafter, so the draft
    context is off and decode() runs no round loop."""
    model = _Wrapper(lm)
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/tiny-qwen3_5"
    engine._model = model  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = None
    # The engine's own positions, as the qwen3_5 lineage gets them: the bare
    # language model cannot derive rope positions without a vision config.
    engine._positional = True
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._draft_context = draftctx.enable(model, None, "")
    engine._speculation = Speculation()
    engine.kv_int8_status = {
        "state": "active",
        "reason": None,
        "probe_seconds": None,
        "floor": tileattn.N0,
        "ceiling": CEILING,
    }
    return engine


def _stub_mlx_vlm(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
    """generate() and stream_generate() reduced to the tiny model's real
    forwards, as generate_step makes them: the prompt but its last row in one
    chunk, then the one-row step that ends every prefill; stream_generate then
    yields greedy tokens, one one-row forward between each. With `fail`,
    stream_generate raises right after that last prefill step."""
    import mlx_vlm

    def forward_prompt(m, kwargs):
        lm, cache, ids = m.language_model, kwargs["prompt_cache"], kwargs["input_ids"]
        positions = {k: kwargs[k] for k in ("position_ids", "rope_deltas") if k in kwargs}
        if ids.shape[1] > 1:
            lm(ids[:, :-1], cache=cache, **positions)
        logits = lm(ids[:, -1:], cache=cache, **positions).logits
        mx.eval(logits, [c.state for c in cache])
        return logits

    def generate(m, processor, prompt, **kwargs):
        assert kwargs["max_tokens"] == 0
        forward_prompt(m, kwargs)

    def stream_generate(m, processor, prompt, **kwargs):
        logits = forward_prompt(m, kwargs)
        if fail:
            raise _Boom("after the prefill segment")
        count = kwargs["max_tokens"]
        for i in range(count):
            token = int(mx.argmax(logits[0, -1]).item())
            yield types.SimpleNamespace(
                is_draft=False, text=f"{token} ", generation_tokens=i + 1, finish_reason=None
            )
            if i + 1 < count:
                cache = kwargs["prompt_cache"]
                logits = m.language_model(mx.array([[token]], dtype=mx.int32), cache=cache).logits

    # On the real module, as test_draftctx stubs it: decode() imports more of
    # mlx_vlm than a stand-in package would carry.
    monkeypatch.setattr(mlx_vlm, "generate", generate)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream_generate)


def test_a_prefill_builds_no_mirror_and_the_cold_paths_decode_builds_one(tiny, monkeypatch):
    """prefill() runs with no window, so even the one-row step that ends it past
    N8 is the tile's. decode() opens one around the generation: on the cold path
    the one-row step that ends its own prefill segment builds the mirror over
    the whole prompt and every token after it reads it, and its finally drops
    the mirror and reports what the window served."""
    lm, ids = tiny
    _serving(monkeypatch)
    _stub_mlx_vlm(monkeypatch)
    engine = _engine(lm)

    cache = lm.make_cache()
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    assert engine.prefill(cache, ids[:PROMPT]) is None
    assert cache[FULL[0]].offset == PROMPT
    assert kv8.calls == kv8_start
    assert _delta(tile_start, tileattn.calls) == {"decode_tile": LAYERS}
    assert all(getattr(cache[i], kv8.MIRROR_ATTR, None) is None for i in FULL)

    cache = lm.make_cache()
    tile_start, kv8_start = dict(tileattn.calls), dict(kv8.calls)
    engine.decode(cache, ids[:PROMPT], 3)
    moved = _delta(kv8_start, kv8.calls)
    # The prefill segment's last step at 1,600 keys, then 1,601 and 1,602.
    assert _counted(kv8_start, kv8.calls) == {"windows": 1, "decode_kv8": 3 * LAYERS}
    assert moved.get("mirror_builds", 0) > 0
    assert _delta(tile_start, tileattn.calls) == {}
    spec = engine.speculation()
    assert spec.kv8 == "served" and spec.kv8_rows >= PROMPT
    assert kv8._window is None and _dropped(cache)


def test_an_exception_out_of_the_generation_closes_the_window(tiny, monkeypatch):
    """A generation that raises after kv8 built its mirror still leaves no
    window, no mirror and no trim wrap behind, and the turn's speculation as it
    was; the next decode opens a window of its own."""
    lm, ids = tiny
    _serving(monkeypatch)
    _stub_mlx_vlm(monkeypatch, fail=True)
    engine = _engine(lm)
    cache = lm.make_cache()
    builds = kv8.calls["mirror_builds"]
    with pytest.raises(_Boom):
        engine.decode(cache, ids[:PROMPT], 3)
    assert kv8.calls["mirror_builds"] > builds
    assert kv8._window is None and _dropped(cache)
    assert engine.speculation() == Speculation()

    _stub_mlx_vlm(monkeypatch)
    engine.decode(lm.make_cache(), ids[:PROMPT], 2)
    assert engine.speculation().kv8 == "served"
