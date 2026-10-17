"""The projection tile's engine wiring: where VLMEngine enables it and with what,
the decode window around the generation, prefill's refusal of an open window,
and unload. Engines are built without weights: object.__new__ for the decode,
prefill and unload seams, and the constructor over stubbed mlx-vlm for the load
order, so these run in CI."""

import contextlib
import sys
import threading
import types

import pytest

pytest.importorskip("mlx.core")

from sous.engine import draftctx, kv8, projtile  # noqa: E402 — after the guard
from sous.engine.promptcache import PrefixCache, PromptMemo, Speculation  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

_WINDOW_OPEN = "projection tile window open across a prefill"


@pytest.fixture(autouse=True)
def _tile_state(monkeypatch):
    """The tile's and kv8's routing state as a fresh load leaves them, restored
    after each test whatever it moved."""
    for name, value in (("_active", 0), ("_window", None), ("_round_flag", 0), ("_scope", None)):
        monkeypatch.setattr(projtile, name, value)
    monkeypatch.setattr(projtile, "_serial", projtile._serial)
    monkeypatch.setattr(kv8, "_active", 0)
    monkeypatch.setattr(kv8, "_window", None)
    monkeypatch.setattr(kv8, "_serial", kv8._serial)
    monkeypatch.setattr(kv8, "calls", dict.fromkeys(kv8.calls, 0))


def _engine() -> VLMEngine:
    """A VLMEngine holding only what decode(), prefill() and unload() read: no
    weights, no drafter, and positions left to the model so the stub needs no
    rotary kwargs."""
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
    engine._cache = PrefixCache(engine, enabled=True)
    return engine


def _result(text: str):
    """One stream_generate result: accepted text, never a draft row."""
    return types.SimpleNamespace(is_draft=False, text=text, generation_tokens=1, finish_reason=None)


def _stub_mlx_vlm(monkeypatch, **attrs) -> None:
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(**attrs))


# ---- prefill ---------------------------------------------------------------------


def test_prefill_refuses_an_open_tile_window(monkeypatch):
    """A window left open would put the tile's bits into KV a fork keeps, under a
    key that cannot tell; prefill raises before it reads the model at all."""

    def generate(*args, **kwargs):
        raise AssertionError("prefill ran past an open window")

    _stub_mlx_vlm(monkeypatch, generate=generate)
    engine = _engine()
    engine._model = None  # _loaded() would raise its own error
    projtile.open_window()
    with pytest.raises(RuntimeError, match=_WINDOW_OPEN):
        engine.prefill([], [1, 2, 3])
    projtile.close_window()
    with pytest.raises(RuntimeError, match="has been unloaded"):
        engine.prefill([], [1, 2, 3])


def test_prefill_runs_with_the_window_closed_while_the_tile_is_active(monkeypatch):
    seen: dict[str, object] = {}

    def generate(*args, **kwargs):
        seen["window"] = projtile._window

    _stub_mlx_vlm(monkeypatch, generate=generate)
    monkeypatch.setattr(projtile, "_active", 1)
    _engine().prefill([types.SimpleNamespace()], [1, 2, 3])
    assert "window" in seen and seen["window"] is None


# ---- decode ----------------------------------------------------------------------


def test_decode_serves_the_tile_inside_its_window_only_while_it_is_active(monkeypatch):
    """The window spans the generation, the decode's own generation-prompt chunk
    included, so the tile serves there and nowhere else; an inactive tile opens
    no window."""
    seen: list[tuple[object, bool]] = []

    def stream_generate(model, processor, prompt, **kwargs):
        seen.append((projtile._window, projtile.serves(5, hook=1)))
        yield _result("ok")

    _stub_mlx_vlm(monkeypatch, stream_generate=stream_generate)
    engine = _engine()
    serial = projtile._serial
    assert engine.decode([types.SimpleNamespace()], [4, 5], 8) == "ok"
    assert seen == [(None, False)] and projtile._serial == serial

    monkeypatch.setattr(projtile, "_active", 1)
    assert engine.decode([types.SimpleNamespace()], [4, 5], 8) == "ok"
    window, served = seen[1]
    assert isinstance(window, projtile.Window) and served is True
    assert projtile._window is None, "the window closes with the decode"
    assert not projtile.serves(5, hook=1)


def test_decode_opens_the_tile_window_after_kv8s_and_closes_it_first(monkeypatch):
    """Inside the drafter's seed, after kv8's open (an open that raised would
    otherwise leave the tile's window behind) and closed before kv8's."""
    order: list[object] = []
    cache = [types.SimpleNamespace()]

    @contextlib.contextmanager
    def seed(tail):
        order.append("seed")
        yield
        order.append("unseed")

    def stream_generate(model, processor, prompt, **kwargs):
        order.append("generate")
        yield _result("ok")

    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "open_window", lambda *, force=False: order.append("kv8 open"))
    monkeypatch.setattr(kv8, "close_window", lambda c: order.append("kv8 close") or ("served", 3))
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(projtile, "open_window", lambda: order.append("tile open"))
    monkeypatch.setattr(projtile, "close_window", lambda: order.append("tile close"))
    _stub_mlx_vlm(monkeypatch, stream_generate=stream_generate)
    engine = _engine()
    engine._draft_context = types.SimpleNamespace(seed=seed, fed=lambda: 0)  # ty: ignore[invalid-assignment]
    assert engine.decode(cache, [4, 5], 8) == "ok"
    assert order == [
        "seed",
        "kv8 open",
        "tile open",
        "generate",
        "tile close",
        "kv8 close",
        "unseed",
    ]
    spec = engine.speculation()
    assert (spec.kv8, spec.kv8_rows) == ("served", 3)


def test_decode_closes_the_tile_window_even_when_kv8s_close_raises(monkeypatch):
    # Recorded, not asserted, in the generator: kv8's close would replace the
    # assertion's error with its own, and the test would pass on it.
    during: list[object] = []

    def stream_generate(model, processor, prompt, **kwargs):
        during.append(projtile._window)
        yield _result("ok")

    def close_window(cache):
        raise RuntimeError("kv8 close failed")

    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(kv8, "open_window", lambda *, force=False: None)
    monkeypatch.setattr(kv8, "close_window", close_window)
    monkeypatch.setattr(projtile, "_active", 1)
    _stub_mlx_vlm(monkeypatch, stream_generate=stream_generate)
    engine = _engine()
    with pytest.raises(RuntimeError, match="kv8 close failed"):
        engine.decode([types.SimpleNamespace()], [4, 5], 8)
    assert len(during) == 1 and isinstance(during[0], projtile.Window)
    assert projtile._window is None
    # Nothing was left open for the next turn's prefill to refuse.
    _stub_mlx_vlm(monkeypatch, generate=lambda *args, **kwargs: None)
    engine.prefill([types.SimpleNamespace()], [1])


def test_decode_closes_both_windows_when_the_generation_raises(monkeypatch):
    during: list[tuple[object, object]] = []

    def failing(model, processor, prompt, **kwargs):
        during.append((projtile._window, kv8._window))
        yield _result("partial")
        raise RuntimeError("boom")

    monkeypatch.setattr(kv8, "_active", 1)
    monkeypatch.setattr(projtile, "_active", 1)
    _stub_mlx_vlm(monkeypatch, stream_generate=failing)
    with pytest.raises(RuntimeError, match="boom"):
        _engine().decode([types.SimpleNamespace()], [4, 5], 8)
    ((tile_window, kv8_window),) = during
    assert tile_window is not None and kv8_window is not None
    assert projtile._window is None and kv8._window is None


# ---- unload ----------------------------------------------------------------------


@pytest.mark.parametrize("failing", [None, "prompt cache"])
def test_unload_clears_the_tile(monkeypatch, failing):
    """First, beside the projection kernel, before anything that can raise: no
    call may reach the tile on the strength of this model's gates."""
    monkeypatch.setattr(projtile, "_active", 1)
    monkeypatch.setattr(projtile, "_scope", "all")
    monkeypatch.setattr(projtile, "_round_flag", 1)
    projtile.open_window()
    engine = _engine()
    if failing is None:
        engine.unload()
    else:

        def fails(*args, **kwargs):
            raise RuntimeError("reset failed")

        monkeypatch.setattr(engine, "reset_prompt_cache", fails)
        with pytest.raises(RuntimeError, match="reset failed"):
            engine.unload()
    state = (projtile._active, projtile._window, projtile._round_flag, projtile._scope)
    assert state == (0, None, 0, None)


# ---- load ------------------------------------------------------------------------

_GIB = 1 << 30


class _Tokenizer:
    bos_token = None
    chat_template = None


@pytest.mark.parametrize(
    ("drafter_loads", "cache_budget", "headroom"),
    [(True, None, 40 * _GIB), (False, 12 * _GIB, None)],
    ids=["automatic-budget", "explicit-budget"],
)
def test_vlm_engine_enables_the_tile_after_the_drafter_route_and_before_kv8(
    monkeypatch, tmp_path, drafter_loads, cache_budget, headroom
):
    """After the kernel and the drafter's route, whose statuses it reads, with
    int8's status and the drafter's kind; the headroom read just before it, the
    budget as configured (None when automatic) and the turn's reserve; the
    conversion read for the fork key right after; then kv8, the store and the
    budget."""
    from sous.engine import base, int8prefill, projkernel, tileattn, verifyattn, vlm

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="fake"),
        language_model=types.SimpleNamespace(),
        get_input_embeddings=lambda *a, **kw: types.SimpleNamespace(position_ids=None),
    )
    processor = types.SimpleNamespace(tokenizer=_Tokenizer())
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(__name__="mlx_vlm", load=lambda model_id: (model, processor)),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.sample_utils",
        types.SimpleNamespace(__name__="mlx_vlm.sample_utils", make_sampler=lambda **kw: None),
    )

    drafter = types.SimpleNamespace(prefer_requested_block_size=False)

    def load_drafter(m, draft_id):
        if not drafter_loads:
            raise RuntimeError("no such repo")
        return drafter, "dflash"

    order: list[str] = []
    seen: dict[str, object] = {}
    int8_status = {"state": "active", "reason": None, "routed": 24}
    tile_status = {"state": "active", "reason": None, "splits": 20, "probe_seconds": 0.43}
    proj_status = {"state": "active", "reason": None, "probe_seconds": 0.37}
    ptile_status = {
        "state": "active",
        "scope": "all",
        "reason": None,
        "copy_gb": 4.09,
        "probe_seconds": 1.2,
        "budget_left_gib": 13.1,
    }

    def enable_drafter(d, *, kernel_status):
        order.append("drafter_route")
        return {"state": "active", "reason": None, "linears": 47, "probe_seconds": 0.12}

    def live_headroom():
        order.append("headroom")
        return headroom

    def ptile_enable(m, **kwargs):
        order.append("projection_tile")
        seen.update(model=m, **kwargs)
        return ptile_status

    def fork_key_state(root):
        order.append("fork_key_state")
        seen["root"] = root
        return "active"

    def kv8_enable(m, **kwargs):
        order.append("kv_int8")
        return {"state": "off", "reason": None}

    class Store:
        def __init__(self, path, fields, budget):
            order.append("fork_store")
            seen["fields"] = fields

    def budget(reserve_bytes):
        order.append("budget")
        return 0

    monkeypatch.setattr(vlm, "_load_quantized_drafter", load_drafter)
    monkeypatch.setattr(int8prefill, "enable", lambda m, *, enabled: int8_status)
    monkeypatch.setattr(
        verifyattn,
        "enable",
        lambda m, *, enabled: {"state": "active", "reason": None, "probe_seconds": 0.21},
    )
    monkeypatch.setattr(
        tileattn, "enable", lambda m, *, enabled, drafter_kind, verify_status: tile_status
    )
    monkeypatch.setattr(
        projkernel, "enable", lambda m, *, enabled, drafter_kind, tile_status: proj_status
    )
    monkeypatch.setattr(projkernel, "enable_drafter", enable_drafter)
    monkeypatch.setattr(vlm, "_pin_block_size", lambda d, block_size: order.append("pin"))
    monkeypatch.setattr(base, "live_headroom", live_headroom)
    monkeypatch.setattr(projtile, "enable", ptile_enable)
    monkeypatch.setattr(projtile, "fork_key_state", fork_key_state)
    monkeypatch.setattr(kv8, "enable", kv8_enable)
    monkeypatch.setattr(draftctx, "enable", lambda m, d, kind: draftctx.OFF)
    monkeypatch.setattr(vlm, "ForkStore", Store)
    monkeypatch.setattr(base, "measure_cache_budget", budget)

    def build() -> VLMEngine:
        return vlm.VLMEngine(
            "test/model",
            draft_id="z-lab/drafter",
            prompt_cache=True,
            int8_prefill=True,
            attention_tile=True,
            projection_kernel=True,
            projection_tile="all",
            cache_budget=cache_budget,
            reserve_bytes=8 * _GIB,
            fork_dir=tmp_path / "forks",
        )

    if drafter_loads:
        engine = build()
    else:
        with pytest.warns(UserWarning, match="speculative drafter"):
            engine = build()
    route = ["drafter_route"] if drafter_loads else []
    tail = ["kv_int8", "fork_store"] + (["budget"] if cache_budget is None else [])
    assert order == [
        *route,
        "headroom",
        "projection_tile",
        "fork_key_state",
        "pin",
        *tail,
    ]
    assert seen["model"] is model
    assert seen["config_value"] == "all"
    assert seen["kernel_status"] is proj_status
    assert seen["tile_status"] is tile_status
    assert seen["int8_status"] is int8_status
    assert seen["drafter_kind"] == ("dflash" if drafter_loads else None)
    assert seen["cache_budget"] == cache_budget
    assert seen["headroom_bytes"] == headroom
    assert seen["reserve_bytes"] == 8 * _GIB
    assert seen["root"] is model.language_model
    assert engine.projection_tile_status is ptile_status
    assert engine._ptile == "active"
    fields = seen["fields"]
    assert isinstance(fields, dict) and fields["ptile"] == "active"


def test_vlm_engine_leaves_the_tile_off_by_default(monkeypatch):
    """Tests and scripts that build an engine themselves keep mlx's layout: the
    constructor's default is off, and the key reads the model unconverted."""
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="fake"),
        get_input_embeddings=lambda *a, **kw: types.SimpleNamespace(position_ids=None),
    )
    processor = types.SimpleNamespace(tokenizer=_Tokenizer())
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(__name__="mlx_vlm", load=lambda model_id: (model, processor)),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.sample_utils",
        types.SimpleNamespace(__name__="mlx_vlm.sample_utils", make_sampler=lambda **kw: None),
    )
    engine = VLMEngine("test/model", cache_budget=0)
    assert engine.projection_tile_status["state"] == "off"
    assert engine._ptile == "off"
