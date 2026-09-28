"""The DFlash drafter's context (sous.engine.draftctx).

The sink, the forward hook and the drafter wraps are exercised on stubs; the
two contracts they rest on — that mlx-vlm's generate_step passes an extra
`capture_layer_ids` into every prefill forward, chunked the way it is, and
that its DFlash round loop hands the drafter's first draft after a reset the
whole prompt hidden — are driven through the real functions on stub models,
the way tests/test_engine_positions.py drives generate_step. No weights.
"""

import sys
import threading
import types
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import draftctx  # noqa: E402 — after the importorskip guard
from sous.engine.promptcache import PromptMemo, Speculation  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402


def _states(layers: int, start: int, n: int, width: int = 2) -> list:
    """One forward's hidden states: per layer an [1, n, width] array whose
    values name the absolute position plus 100 x the layer, so a slice of the
    sink's tail can be checked exactly."""
    positions = mx.arange(start, start + n, dtype=mx.float32)[None, :, None]
    return [mx.broadcast_to(positions + 100 * layer, (1, n, width)) for layer in range(layers)]


# ---- the sink -----------------------------------------------------------------


def test_sink_keeps_the_newest_positions_across_forwards():
    sink = draftctx.Sink([0, 1], keep=5)
    sink.take(_states(2, 0, 4))
    sink.take(_states(2, 4, 4))
    sink.take(_states(2, 8, 1))
    tail = sink.finish()
    assert tail is not None and tail.shape == (1, 5, 4)  # layers along the last axis
    assert tail[0, :, 0].tolist() == [4, 5, 6, 7, 8]
    assert tail[0, :, 2].tolist() == [104, 105, 106, 107, 108]


def test_sink_drops_the_chunks_the_tail_can_no_longer_reach():
    sink = draftctx.Sink([0], keep=3)
    sink.take(_states(1, 0, 10))
    sink.take(_states(1, 10, 10))
    tail = sink.finish()
    assert tail is not None and tail[0, :, 0].tolist() == [17, 18, 19]


def test_a_sink_that_keeps_nothing_takes_nothing():
    sink = draftctx.Sink([0], keep=0)
    sink.take(_states(1, 0, 4))
    assert sink.finish() is None


def test_sink_hands_back_everything_when_the_prefill_is_shorter_than_the_window():
    sink = draftctx.Sink([0], keep=2047)
    sink.take(_states(1, 0, 6))
    tail = sink.finish()
    assert tail is not None and tail.shape == (1, 6, 2)


def test_sink_ignores_forwards_that_captured_nothing():
    sink = draftctx.Sink([0], keep=4)
    sink.take(None)
    sink.take([])
    sink.take(mx.zeros((1, 3, 2)))  # not the list a forward returns
    assert sink.finish() is None


def _active_bytes() -> int:
    """Active memory once every completed op's inputs are released: on Metal
    they go with the command buffer's completion handler, which mx.eval
    does not wait for."""
    import gc

    gc.collect()
    mx.synchronize()
    return mx.get_active_memory()


@pytest.mark.parametrize(
    ("forwards", "keep"),
    [
        ([2048, 1024], 2047),  # a chunk boundary inside the tail: a view would hold 3072 rows
        ([1200, 1200], 2047),  # ... 2400 rows
        ([2048, 100], 2047),
        ([2048], 2047),  # one forward longer than the tail: a view would hold it whole
        ([2047], 2047),  # exactly the tail
        ([1500, 500, 3, 1], 2047),  # the tail is everything
    ],
)
def test_the_tail_holds_no_more_memory_than_its_own_rows(forwards, keep):
    """A slice of an mlx array is a view on its parent: sliced off the
    assembled chunks, the tail would keep every row of them alive across the
    decode — half again to twice its size when a chunk boundary falls inside
    it. The assembled tail must own exactly its rows, to within the
    allocator's page."""
    width = 256
    sink = draftctx.Sink([0], keep)
    base = _active_bytes()
    for i, n in enumerate(forwards):
        sink.take([mx.zeros((1, n, width), dtype=mx.bfloat16) + i])
    tail = sink.finish()
    assert tail is not None
    rows = min(keep, sum(forwards))
    assert tail.shape == (1, rows, width) and tail.dtype == mx.bfloat16
    assert tail[0, -1, 0].item() == len(forwards) - 1  # the newest forward's rows last
    held = _active_bytes() - base
    assert held <= rows * width * 2 + 16 * 1024, (held, rows * width * 2)


def test_sink_finish_empties_it():
    sink = draftctx.Sink([0], keep=4)
    sink.take(_states(1, 0, 2))
    assert sink.finish() is not None
    assert sink.finish() is None


# ---- the window ---------------------------------------------------------------


def _config(**kw) -> types.SimpleNamespace:
    base = {"draft_window_size": None, "sliding_window": 2048, "layer_types": ["full_attention"]}
    return types.SimpleNamespace(**{**base, **kw})


def test_window_reads_the_drafters_cache_geometry():
    sliding = _config(layer_types=["sliding_attention"] * 5)
    assert draftctx.window(types.SimpleNamespace(config=sliding)) == 2047
    buffered = _config(draft_window_size=512, layer_types=["sliding_attention"])
    assert draftctx.window(types.SimpleNamespace(config=buffered)) == 512
    unbounded = _config()
    assert draftctx.window(types.SimpleNamespace(config=unbounded)) == draftctx.DEFAULT_WINDOW
    assert draftctx.window(object()) == draftctx.DEFAULT_WINDOW


# ---- stubs for the hooks -----------------------------------------------------


class _LM:
    """A language model whose forward returns hidden states for the layers
    asked for, numbered by the cache's offset — the shape the hook reads."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, inputs: Any = None, inputs_embeds=None, cache=None, **kwargs):
        self.calls.append(dict(kwargs, n=int(inputs.shape[1])))
        n = int(inputs.shape[1])
        offset = cache[0].offset if cache else 0
        if cache:
            cache[0].offset += n
        layers = kwargs.get("capture_layer_ids")
        return types.SimpleNamespace(
            logits=mx.zeros((1, n, 8)),
            cross_attention_states=None,
            encoder_outputs=None,
            hidden_states=None if layers is None else _states(len(layers), offset, n),
        )


class _Drafter:
    """A DFlash drafter reduced to what the round loop and the wraps touch."""

    prefer_requested_block_size = True

    def __init__(self, layer_ids=(5, 19)):
        self.config = _config(target_layer_ids=list(layer_ids), layer_types=["sliding_attention"])
        self.calls: list[tuple[str, Any]] = []
        self.resets = 0
        self.accept_lens: list[int] = []
        self.draft_lens: list[int] = []

    def reset(self, target_model):
        self.resets += 1
        return ["cache"]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        self.calls.append(("sampled", hidden))
        return self._propose(last_bonus, block_size, token_dtype)

    def draft_block_greedy(
        self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32
    ):
        self.calls.append(("greedy", hidden))
        return self._propose(last_bonus, block_size, token_dtype)

    @staticmethod
    def _propose(last_bonus, block_size, token_dtype):
        # The target below always says "next = current + 1", so proposing
        # exactly that is a full acceptance and anything else a rejection.
        bonus = last_bonus if isinstance(last_bonus, int) else int(last_bonus.item())
        return mx.array([[bonus + i for i in range(1, block_size)]], dtype=token_dtype)


def _model(lm=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(language_model=lm or _LM())


def _cache() -> types.SimpleNamespace:
    return types.SimpleNamespace(offset=0, state=None)


# ---- enable() -----------------------------------------------------------------


def test_enable_is_off_without_a_drafter():
    ctx = draftctx.enable(_model(), None, "")
    assert ctx is draftctx.OFF and ctx.status["state"] == "off" and not ctx.active
    assert ctx.sink(True) is None and ctx.fed() == 0
    with ctx.seed(mx.zeros((1, 1, 1))):
        pass  # nothing to seed, nothing raised


def _without(name: str) -> _Drafter:
    drafter = _Drafter()
    setattr(drafter, name, None)  # shadows the method: not callable
    return drafter


def _with_hook(name: str) -> _Drafter:
    drafter = _Drafter()
    setattr(drafter, name, lambda *a: a)
    return drafter


@pytest.mark.parametrize(
    ("drafter", "kind", "reason"),
    [
        (_Drafter(), "eagle3", "drafter kind 'eagle3'"),
        (_Drafter(layer_ids=()), "dflash", "names no target layers"),
        (types.SimpleNamespace(prefer_requested_block_size=False), "dflash", "no target layers"),
        (_without("reset"), "dflash", "has no reset"),
        (_without("draft_block"), "dflash", "has no draft_block"),
        (_with_hook("choose_block_ceiling"), "dflash", "defines choose_block_ceiling"),
        (_with_hook("choose_initial_block_size"), "dflash", "defines choose_initial_block_size"),
    ],
)
def test_enable_refuses_what_it_cannot_seed_with_a_warning(drafter, kind, reason):
    with pytest.warns(UserWarning, match=f"drafter context unavailable \\(.*{reason}"):
        ctx = draftctx.enable(_model(), drafter, kind)
    assert ctx.status["state"] == "unavailable" and reason in ctx.status["reason"]
    assert not ctx.active and ctx.sink(True) is None


def test_enable_refuses_a_drafter_the_round_loop_prepares_hidden_states_for():
    drafter = _Drafter()
    drafter.prepare_target_hidden = lambda hidden: hidden  # ty: ignore[unresolved-attribute]
    with pytest.warns(UserWarning, match="defines prepare_target_hidden"):
        assert draftctx.enable(_model(), drafter, "dflash").status["state"] == "unavailable"


def test_enable_refuses_a_model_without_a_language_model():
    with pytest.warns(UserWarning, match="no language_model"):
        ctx = draftctx.enable(types.SimpleNamespace(), _Drafter(), "dflash")
    assert ctx.status["state"] == "unavailable"


def test_enable_refuses_a_language_model_that_decodes_outside_its_forward():
    lm = _LM()
    lm._batch_invariant_decode = lambda *a, **kw: None  # ty: ignore[unresolved-attribute]
    with pytest.warns(UserWarning, match="outside its forward"):
        assert draftctx.enable(_model(lm), _Drafter(), "dflash").status["state"] == "unavailable"


def test_enable_never_raises():
    class Explosive:
        prefer_requested_block_size = True

        @property
        def config(self):
            raise RuntimeError("no config today")

    with pytest.warns(UserWarning, match="no config today"):
        assert draftctx.enable(_model(), Explosive(), "dflash").status["state"] == "unavailable"


def test_enable_arms_the_model_and_wraps_the_drafter():
    lm, drafter = _LM(), _Drafter()
    ctx = draftctx.enable(_model(lm), drafter, "dflash")
    assert ctx.status == {"state": "active", "reason": None, "window": 2047, "layers": 2}
    assert ctx.active and ctx.layer_ids == [5, 19] and ctx.keep == 2047
    assert type(lm) in draftctx._WRAPPED
    # Per-instance shadows, the way mlx-vlm's prefer_requested_block_size rides.
    assert {"reset", "draft_block", "draft_block_greedy"} <= set(vars(drafter))
    sink = ctx.sink(True)
    assert sink is not None and sink.layer_ids == [5, 19] and sink.keep == 2047
    assert draftctx.DraftContext.capture_kwargs(sink) == {"capture_layer_ids": [5, 19]}
    assert draftctx.DraftContext.capture_kwargs(None) == {}


# ---- the forward hook ---------------------------------------------------------


def test_the_hook_captures_only_while_a_sink_is_armed():
    lm = _LM()
    ctx = draftctx.enable(_model(lm), _Drafter(), "dflash")
    cache = [_cache()]
    lm(mx.zeros((1, 3), dtype=mx.int32), cache=cache, capture_layer_ids=[5, 19])  # unarmed
    sink = ctx.sink(True)
    with ctx.arm(_model(lm), sink) as armed:
        assert armed is sink and getattr(lm, draftctx._SINK) is sink
        lm(mx.zeros((1, 2), dtype=mx.int32), cache=cache, capture_layer_ids=[5, 19])
    assert getattr(lm, draftctx._SINK) is None
    lm(mx.zeros((1, 4), dtype=mx.int32), cache=cache, capture_layer_ids=[5, 19])  # disarmed again
    tail = sink.finish()
    assert tail is not None and tail[0, :, 0].tolist() == [3, 4]  # the armed forward alone


def test_arming_no_sink_touches_nothing():
    lm = _LM()
    ctx = draftctx.enable(_model(lm), _Drafter(), "dflash")
    with ctx.arm(_model(lm), None) as armed:
        assert armed is None and draftctx._SINK not in vars(lm)


def test_install_forward_hook_is_idempotent():
    class Once:
        calls = 0

        def __call__(self, inputs=None, **kwargs):
            Once.calls += 1
            return types.SimpleNamespace(hidden_states=None)

    try:
        draftctx.install_forward_hook(Once)
        draftctx.install_forward_hook(Once)
        Once()(None)
        assert Once.calls == 1
        # One layer of wrapping: the hook's __wrapped__ is the class's own forward.
        inner = Once.__call__.__wrapped__  # ty: ignore[unresolved-attribute]
        assert not hasattr(inner, "__wrapped__")
    finally:
        draftctx._WRAPPED.discard(Once)


def test_a_sink_that_fails_disarms_itself_and_the_forward_still_returns():
    lm = _LM()
    ctx = draftctx.enable(_model(lm), _Drafter(), "dflash")

    class Broken(draftctx.Sink):
        def take(self, hidden_states):
            raise ValueError("boom")

    sink = Broken([5, 19], 4)
    with ctx.arm(_model(lm), sink), pytest.warns(UserWarning, match="capture failed"):
        out = lm(mx.zeros((1, 2), dtype=mx.int32), cache=[_cache()], capture_layer_ids=[5, 19])
        assert out.logits.shape == (1, 2, 8)
        assert getattr(lm, draftctx._SINK) is None


# ---- the drafter wraps --------------------------------------------------------


def _draft(drafter, name: str, bonus: int, hidden):
    return getattr(drafter, name)(bonus, hidden, ["cache"], 3, lambda logits: logits)


@pytest.mark.parametrize("name", ["draft_block", "draft_block_greedy"])
def test_the_first_draft_after_a_reset_gets_the_seeded_context_in_front(name):
    drafter = _Drafter()
    ctx = draftctx.enable(_model(), drafter, "dflash")
    tail = mx.full((1, 3, 4), 7.0)
    own = mx.zeros((1, 2, 4))
    with ctx.seed(tail):
        drafter.reset(_model())
        _draft(drafter, name, 1, own)
        _draft(drafter, name, 2, own)
    kind = {"draft_block": "sampled", "draft_block_greedy": "greedy"}[name]
    assert [k for k, _ in drafter.calls] == [kind, kind]
    first, second = (hidden for _, hidden in drafter.calls)
    assert first.shape == (1, 5, 4) and first[0, :, 0].tolist() == [7, 7, 7, 0, 0]
    assert second.shape == (1, 2, 4)  # later rounds carry only their own rows
    assert ctx.fed() == 5 and drafter.resets == 1


def test_a_seed_the_decode_never_drew_on_does_not_leak_into_the_next():
    drafter = _Drafter()
    ctx = draftctx.enable(_model(), drafter, "dflash")
    with ctx.seed(mx.full((1, 3, 4), 7.0)):
        drafter.reset(_model())  # one token asked for: no round, no draft call
    assert ctx.fed() == 0
    drafter.reset(_model())
    _draft(drafter, "draft_block", 1, mx.zeros((1, 2, 4)))
    assert drafter.calls[-1][1].shape == (1, 2, 4) and ctx.fed() == 2


def test_a_decode_that_stops_before_the_round_loop_reads_as_no_context():
    """stream_generate can stop on the first token, before the round loop
    ever resets the drafter: the previous decode's context must not be
    reported as this one's."""
    drafter = _Drafter()
    ctx = draftctx.enable(_model(), drafter, "dflash")
    with ctx.seed(mx.full((1, 3, 4), 7.0)):
        drafter.reset(_model())
        _draft(drafter, "draft_block", 1, mx.zeros((1, 2, 4)))
    assert ctx.fed() == 5
    with ctx.seed(mx.full((1, 3, 4), 8.0)):
        pass  # no reset, no draft call
    assert ctx.fed() == 0


def test_enable_wraps_a_drafter_once():
    import inspect

    drafter = _Drafter()
    model = _model()
    draftctx.enable(model, drafter, "dflash")
    draftctx.enable(model, drafter, "dflash")
    for name in ("reset", "draft_block", "draft_block_greedy"):
        wrapper = vars(drafter)[name]
        assert inspect.ismethod(wrapper.__wrapped__)  # one layer over the class's own


@pytest.mark.parametrize(
    "tail",
    [
        mx.full((1, 3, 8), 7.0),  # another width: another model's layers
        mx.full((2, 3, 4), 7.0),  # another batch: rows the loop never handed out
        mx.full((3, 4), 7.0),  # not the round loop's rank at all
    ],
    ids=["width", "batch", "rank"],
)
def test_a_context_of_another_shape_is_left_out_rather_than_forced(tail):
    drafter = _Drafter()
    ctx = draftctx.enable(_model(), drafter, "dflash")
    with ctx.seed(tail):
        drafter.reset(_model())
        _draft(drafter, "draft_block", 1, mx.zeros((1, 2, 4)))
    assert drafter.calls[-1][1].shape == (1, 2, 4) and ctx.fed() == 2


def test_the_context_fed_is_reported_no_larger_than_the_window():
    """A full tail plus the generation prompt is handed over whole — the
    drafter drops the oldest rows past its window itself — but what it
    drafted from is one window, and that is the number logged."""
    drafter = _Drafter()
    drafter.config.sliding_window = 4  # a window of 3
    ctx = draftctx.enable(_model(), drafter, "dflash")
    assert ctx.keep == 3
    with ctx.seed(mx.full((1, 3, 4), 7.0)):
        drafter.reset(_model())
        _draft(drafter, "draft_block", 1, mx.zeros((1, 2, 4)))
    assert drafter.calls[-1][1].shape == (1, 5, 4)  # handed over whole
    assert ctx.fed() == 3


def test_seeding_nothing_leaves_the_drafters_own_context_alone():
    drafter = _Drafter()
    ctx = draftctx.enable(_model(), drafter, "dflash")
    with ctx.seed(None):
        drafter.reset(_model())
        _draft(drafter, "draft_block_greedy", 1, mx.zeros((1, 6, 4)))
    assert drafter.calls[-1][1].shape == (1, 6, 4) and ctx.fed() == 6


# ---- the contract with mlx-vlm's generate_step --------------------------------


class _Wrapper:
    """A wrapper model whose text-only helper embeds the ids; its language
    model is an _LM, so the class hook sees every forward generate_step makes."""

    def __init__(self):
        self.language_model = _LM()
        self.config = types.SimpleNamespace(model_type="fake", eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        n = input_ids.shape[1]
        return types.SimpleNamespace(
            inputs_embeds=mx.zeros((1, n, 4)),
            to_dict=lambda: {"inputs_embeds": None, "position_ids": None, "rope_deltas": None},
        )


def _prefill_through_generate_step(n_tokens: int, keep: int, layer_ids: list[int]):
    """Run the real generate_step over an n-token prompt with the capture
    kwarg, a sink armed; return the forwards it made and the sink's tail."""
    from mlx_vlm.generate.ar import generate_step

    model = _Wrapper()
    ctx = draftctx.enable(model, _Drafter(layer_ids), "dflash")
    sink = draftctx.Sink(layer_ids, keep)
    with ctx.arm(model, sink):
        list(
            generate_step(
                mx.arange(n_tokens, dtype=mx.int32)[None],
                model,  # ty: ignore[invalid-argument-type]
                None,
                None,
                prompt_cache=[_cache()],
                max_tokens=0,
                sampler=lambda logprobs: mx.zeros((logprobs.shape[0],), dtype=mx.int32),
                **draftctx.DraftContext.capture_kwargs(sink),
            )
        )
    return model.language_model.calls, sink.finish()


def test_generate_step_hands_the_capture_kwarg_to_every_prefill_forward():
    """The contract the prefill capture rests on: the extra kwarg reaches
    each chunk of the chunk loop and the final one-token step alike, so the
    forwards' hidden states cover every prompt token in order — and the
    chunking is generate_step's own (2048-token chunks, then the last token),
    which is why capturing through it leaves the KV cache exactly as an
    uncaptured prefill does."""
    calls, tail = _prefill_through_generate_step(4100, keep=3000, layer_ids=[5, 19])
    assert [c["n"] for c in calls] == [2048, 2048, 3, 1]
    assert all(c["capture_layer_ids"] == [5, 19] for c in calls)
    assert tail is not None and tail.shape == (1, 3000, 4)
    assert tail[0, :, 0].tolist() == list(range(1100, 4100))
    assert tail[0, :, 2].tolist() == [100 + p for p in range(1100, 4100)]


def test_a_prompt_shorter_than_the_window_is_captured_whole():
    calls, tail = _prefill_through_generate_step(9, keep=2047, layer_ids=[0])
    assert [c["n"] for c in calls] == [8, 1]
    assert tail is not None and tail[0, :, 0].tolist() == list(range(9))


class _Processor:
    """What stream_generate reads off a processor on a max_tokens=0 call:
    stopping criteria it resets, and a detokenizer it copies and resets but
    never feeds."""

    class _Stopping:
        def reset(self, eos):
            pass

        def add_eos_token_ids(self, ids):
            pass

        def __call__(self, token):
            return False

    class _Detokenizer:
        last_segment = ""

        def reset(self):
            pass

        def add_token(self, *args, **kwargs):
            raise AssertionError("a prefill has nothing to detokenize")

        def finalize(self):
            pass

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(stopping_criteria=self._Stopping())
        self.detokenizer = self._Detokenizer()


def test_mlx_vlm_generate_passes_the_capture_kwarg_through_to_every_forward():
    """The engine's entry is generate(), whose dispatch pops the kwargs it
    knows and hands the rest to generate_step: an undeclared one such as
    capture_layer_ids (GenerateKwargs lists it no more than position_ids)
    has to survive that, or every capture is silently empty."""
    from mlx_vlm import generate

    model = _Wrapper()
    ctx = draftctx.enable(model, _Drafter(), "dflash")
    sink = draftctx.Sink([5, 19], 2047)
    with ctx.arm(model, sink):
        generate(
            model,  # ty: ignore[invalid-argument-type]
            _Processor(),
            "",
            max_tokens=0,
            verbose=False,
            prompt_cache=[_cache()],
            input_ids=mx.arange(30, dtype=mx.int32)[None],
            **draftctx.DraftContext.capture_kwargs(sink),
        )
    calls = model.language_model.calls
    assert [c["n"] for c in calls] == [29, 1]
    assert all(c["capture_layer_ids"] == [5, 19] for c in calls)
    tail = sink.finish()
    assert tail is not None and tail[0, :, 0].tolist() == list(range(30))


# ---- the contract with mlx-vlm's DFlash round loop ----------------------------


class _Target:
    """A target whose greedy pick is always "the token after this one", so a
    drafter proposing that is fully accepted and the test can count rounds.
    Its hidden states carry the token id, so the rows a draft saw are legible;
    two captured layers of width 2 concatenate to the width 4 the seeded and
    prompt hidden states below have."""

    def __init__(self):
        self.verify_calls = 0

    def __call__(self, inputs, cache=None, **kwargs):
        self.verify_calls += 1
        n = int(inputs.shape[1])
        vocab = 64
        logits = (mx.arange(vocab)[None, None, :] == (inputs + 1)[:, :, None]).astype(mx.float32)
        tokens = inputs.astype(mx.float32)[:, :, None]
        layers = kwargs.get("capture_layer_ids") or []
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=[mx.broadcast_to(tokens, (1, n, 2)) for _ in layers],
            gdn_states=None,
            shared_kv_states=None,
        )


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_the_round_loop_hands_the_first_draft_the_prompt_hidden_with_the_seed_in_front(greedy):
    """The contract the seeding rests on: _dflash_rounds resets the drafter,
    then gives its first draft call the prompt hidden it was handed — whole
    — and every later call only the verify rows the target accepted. The
    wrap prepends the seeded tail there and nowhere else, and the lifetime
    counters count exactly this call's rounds."""
    from mlx_vlm.speculative.dflash import _dflash_rounds

    target = _Target()
    drafter = _Drafter()
    model = types.SimpleNamespace(language_model=target)
    ctx = draftctx.enable(model, drafter, "dflash")
    prompt_hidden = mx.full((1, 7, 4), -1.0)  # the decode call's own prefill: 7 tokens
    tail = mx.full((1, 5, 4), -2.0)  # what the prefill before it captured
    before = draftctx.rounds_snapshot(drafter)
    with ctx.seed(tail):
        produced = [
            tok
            for tok, _ in _dflash_rounds(
                model,  # ty: ignore[invalid-argument-type]
                drafter,  # ty: ignore[invalid-argument-type]
                [],
                prompt_hidden,
                first_bonus=10,
                max_tokens=6,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                greedy_sampling=greedy,
            )
        ]
    # Two rounds of a fully accepted 2-token draft plus its bonus: 11 12 13, 14 15 16 — capped at 6.
    assert produced == [11, 12, 13, 14, 15]
    assert drafter.resets == 1
    kinds = {kind for kind, _ in drafter.calls}
    assert kinds == ({"greedy"} if greedy else {"sampled"})
    first, second = (hidden for _, hidden in drafter.calls)
    assert first.shape == (1, 12, 4) and first[0, :, 0].tolist() == [-2] * 5 + [-1] * 7
    assert second.shape == (1, 3, 4) and second[0, :, 0].tolist() == [10, 11, 12]
    assert ctx.fed() == 12
    assert draftctx.rounds_since(drafter, before) == (2, 4, 4)
    assert draftctx.rounds_since(None, None) == (0, 0, 0)


# ---- the engine's wiring -------------------------------------------------------


def _engine(model, drafter=None) -> VLMEngine:
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = model
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()
    engine._positional = False
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = drafter
    engine._draft_kind = "dflash" if drafter is not None else ""
    engine._draft_block_size = 3
    engine._draft_context = draftctx.enable(model, drafter, engine._draft_kind)
    engine._speculation = Speculation()
    return engine


def _stub_mlx_vlm(monkeypatch, model, calls: dict, rounds: int = 0) -> None:
    """generate() runs two forwards of the model's language model with the
    kwargs it was given, the way generate_step would; stream_generate() runs
    a reset and one draft call on the drafter, the way the round loop would,
    and bumps the drafter's lifetime counters by `rounds` rounds."""

    def generate(m, processor, prompt, **kwargs):
        calls["prefill"] = kwargs
        extra = {k: v for k, v in kwargs.items() if k == "capture_layer_ids"}
        cache = kwargs["prompt_cache"]
        ids = kwargs["input_ids"]
        n = ids.shape[1]
        m.language_model(ids[:, : n - 1], inputs_embeds=None, cache=cache, **extra)
        m.language_model(ids[:, n - 1 :], inputs_embeds=None, cache=cache, **extra)

    def stream_generate(m, processor, prompt, **kwargs):
        calls["decode"] = kwargs
        drafter = kwargs.get("draft_model")
        if drafter is not None:
            drafter.reset(m)
            drafter.draft_block(1, mx.zeros((1, kwargs["input_ids"].shape[1], 4)), [], 3, None)
            # The lifetime counters _record_speculative_round keeps, which
            # survive the reset above: rounds, accepted drafts, drafted.
            for attr, step in (
                ("speculative_total_rounds", rounds),
                ("speculative_total_accepted", 1.5 * rounds),
                ("speculative_total_drafted", 2 * rounds),
            ):
                setattr(drafter, attr, getattr(drafter, attr, 0) + step)
        return iter(())

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(generate=generate, stream_generate=stream_generate),
    )


def test_a_captured_prefill_asks_for_the_target_layers_and_keeps_the_tail(monkeypatch):
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model, _Drafter())
    cache = [_cache()]
    capture = engine.prefill(cache, list(range(10)), capture=True)
    assert calls["prefill"]["capture_layer_ids"] == [5, 19]
    assert isinstance(capture, draftctx.Sink)
    assert getattr(model.language_model, draftctx._SINK) is None  # disarmed after
    assert cache[0].offset == 10
    tail = draftctx.DraftContext.tail(capture)
    assert tail is not None and tail[0, :, 0].tolist() == list(range(10))


def test_a_capture_continues_across_a_turns_prefills(monkeypatch):
    """Fork stops split a cold prefill into segments; the capture an earlier
    segment returned is handed back in, and the tail spans them."""
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model, _Drafter())
    cache = [_cache()]
    capture = engine.prefill(cache, [], capture=True)  # an empty stop still starts one
    assert isinstance(capture, draftctx.Sink)
    assert engine.prefill(cache, list(range(10)), capture=capture) is capture
    assert engine.prefill(cache, list(range(10, 15)), capture=capture) is capture
    tail = draftctx.DraftContext.tail(capture)
    assert tail is not None and tail[0, :, 0].tolist() == list(range(15))
    assert draftctx.DraftContext.tail(capture) is None  # assembled once


def test_a_plain_prefill_captures_nothing_and_asks_for_nothing(monkeypatch):
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model, _Drafter())
    assert engine.prefill([_cache()], [1, 2, 3]) is None
    assert "capture_layer_ids" not in calls["prefill"]
    assert draftctx.DraftContext.tail(None) is None


def test_a_captured_prefill_without_a_drafter_is_a_plain_prefill(monkeypatch):
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model)
    assert engine.prefill([_cache()], [1, 2, 3], capture=True) is None
    assert "capture_layer_ids" not in calls["prefill"]


def test_decode_seeds_the_drafter_and_reports_the_rounds(monkeypatch):
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls, rounds=4)
    drafter = _Drafter()
    engine = _engine(model, drafter)
    capture = draftctx.Sink([5, 19], 2047)
    capture.take([mx.full((1, 5, 2), 9.0), mx.full((1, 5, 2), 9.0)])
    engine.decode([_cache()], [1, 2], 8, context=capture)
    assert calls["decode"]["draft_model"] is drafter
    ((_, hidden),) = drafter.calls
    assert hidden.shape == (1, 7, 4) and hidden[0, :, 0].tolist() == [9] * 5 + [0] * 2
    spec = engine.speculation()
    assert (spec.rounds, spec.accepted, spec.drafted, spec.context) == (4, 6, 8, 7)
    assert getattr(drafter, draftctx._PENDING) is None  # nothing left for a later turn


def test_a_capture_that_failed_part_way_seeds_nothing(monkeypatch):
    """A tail missing a forward's rows would end short of the generation
    prompt: the wrong positions cost more acceptance than none, so a sink
    that fails voids what it held and refuses the rest of the turn."""
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model, _Drafter())

    class Flaky(draftctx.Sink):
        def take(self, hidden_states):
            if self._held >= 4:  # the forward after the first rows
                raise ValueError("boom")
            super().take(hidden_states)

    capture = Flaky([5, 19], 2047)
    with pytest.warns(UserWarning, match="capture failed"):
        assert engine.prefill([_cache()], list(range(10)), capture=capture) is capture
    assert draftctx.DraftContext.tail(capture) is None
    engine.prefill([_cache()], list(range(10, 14)), capture=capture)
    assert draftctx.DraftContext.tail(capture) is None


def test_close_lets_go_of_the_drafter_and_unload_calls_it():
    """The wraps close over the drafter and the drafter binds the target's
    embedding and head: an unload that kept either would keep the weights."""
    import gc
    import weakref

    from sous.engine.promptcache import PrefixCache

    model, drafter = _Wrapper(), _Drafter()
    engine = _engine(model, drafter)
    engine._cache = PrefixCache(engine, enabled=False)
    assert engine._draft_context.active
    gone = weakref.ref(drafter)
    lm = weakref.ref(model.language_model)
    del drafter
    engine.unload()
    del model
    gc.collect()
    assert gone() is None and lm() is None
    assert engine._draft_context is draftctx.OFF and engine.draft_context_status["state"] == "off"


def test_a_decode_that_never_drafted_reports_zero_rounds_and_its_context(monkeypatch):
    """mlx-vlm answers None for the counts of a call with no rounds (one
    token asked for); the line must read zeros, not raise."""
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls, rounds=0)
    engine = _engine(model, _Drafter())
    assert engine.speculation() == Speculation()  # before any decode
    engine.decode([_cache()], [1, 2], 1, context=None)
    assert engine.speculation() == Speculation(rounds=0, drafted=0, accepted=0, context=2)


def test_decode_without_a_drafter_reports_zeros(monkeypatch):
    model, calls = _Wrapper(), {}
    _stub_mlx_vlm(monkeypatch, model, calls)
    engine = _engine(model)
    engine.decode([_cache()], [1, 2], 8, context=None)
    assert "draft_model" not in calls["decode"]
    assert engine.speculation() == Speculation()


def test_the_engine_exposes_the_status():
    engine = _engine(_Wrapper(), _Drafter())
    assert engine.draft_context_status == {
        "state": "active",
        "reason": None,
        "window": 2047,
        "layers": 2,
    }
    assert _engine(_Wrapper()).draft_context_status["state"] == "off"
