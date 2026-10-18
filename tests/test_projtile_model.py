"""The projection tile on the real default model: block invariance, the bf16 floor, the
slab-only MLP paths, drafter on against drafter off, and the scope on the load line.

Every engine but one tile-off reference runs what a 64 GB M5 Pro serves: int8 prefill, the
attention tile, the projection kernel, kv8 at the default 131072 window and the DFlash2
drafter at its resolved block, with the tile forced to scope "all" or "mlp" and an explicit
12 GiB prompt-cache budget. Acceptance bars (manual, on the M5 Pro):
- Each such engine reports the tile active at its scope and kv8 active; the test skips,
  naming the reason, where the tile cannot be.
- Block invariance, the contract on the drafter's path: at each scope, greedy token ids at
  blocks 2, 12 (set on the engine; 12 is the one whose verify rows exceed 8 and are cut into
  runs) and 0 (the drafter's own policy) equal block 8's (the resolved block), on a
  ~1,450-token prompt and on one ending just below the 57,600-key step boundary, 256 tokens.
  Every block's verify runs reach the tile, only block 12 cuts a call into runs, and every
  run of a cut call whose linears have planes is served on the tile.
- The bf16 floor: every linear the tile serves, fused as the verifier calls it and alone, on
  eight varied rows, is within the element bound of a float64 reference everywhere, and its
  relative RMS is within TILE_FLOOR_FACTOR of the floor (that reference rounded to bf16).
- The slab-only MLP: every layer's gate, up and down called with 1, 5, 8 and 128 rows equal
  a tile-off engine's bit for bit; with 9, 64 and 127 rows, which int8's slab GEMM serves
  (stock on the tile-off engine), they are within REL_RMS_BOUND of it.
- Drafter on (block 8, scope "all") against drafter off: over every position before each
  prompt's first divergence D is the 99.9th percentile of the change in the top-two margin
  between the runs; at each first differing token the drafter-off margin between the two
  emitted tokens is at most max(D, 2 ulp of its top logit) and the drafter run's logits for
  those two tokens are within D of the drafter-off run's.
- The load line and the status document name the scope, and at "all" the copy's size.
"""

import contextlib
import functools
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")

from sous.config import SousConfig  # noqa: E402 — after the importorskip guard
from sous.engine import int8prefill, kv8, projkernel, projtile, slab, tileattn  # noqa: E402
from sous.engine.base import EngineManager, fetch_model_config  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

pytestmark = pytest.mark.model

DEFAULT_27B = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
ROOT = Path(__file__).resolve().parents[1]
HEAD = "Continue this Python source exactly where it stops, in the same style:\n\n"
# The attention tile's model-test prompts: generation from the short one crosses N0 (1,536
# keys) and the first step boundary above it; the long one ends just below the 57,600-key
# boundary, where kv8 serves.
PROMPTS = {"short": 1450, "long": 57_560}
MAX_TOKENS = 256
SLACK = 16
# The daemon's default [model].max_context_tokens: the window kv8's ceiling is sized against.
WINDOW = 131072
CACHE_BUDGET = 12 * 2**30
SCOPES = ("all", "mlp")
# 8 is what the block resolves to with the kernel active, so it goes first.
BLOCKS = (8, 2, 12, 0)
SPLIT_BLOCK = 12
# Rows a slab-only MLP call computes as a tile-off engine does (the one-row kernel, the
# projection kernel, int8 from 128), and rows that int8's slab GEMM takes where stock served.
SLAB_EQUAL_ROWS = (1, 5, 8, 128)
SLAB_INT8_ROWS = (9, 64, 127)
_MLP = ("gate_proj", "up_proj", "down_proj")


# ---- engines and prompts ----------------------------------------------------------


def _build(scope: bool | str, *, drafter: bool = True) -> VLMEngine:
    started = time.perf_counter()
    engine = VLMEngine(
        DEFAULT_27B,
        temperature=0.0,
        cache_budget=CACHE_BUDGET,
        draft_id=DRAFTER if drafter else "",
        draft_block_explicit=False,
        int8_prefill=True,
        attention_tile=True,
        projection_kernel=True,
        projection_tile=scope,
        kv_int8=True,
        kv_int8_window=WINDOW,
    )
    print(
        f"load {scope=} {drafter=}: {time.perf_counter() - started:.1f} s,"
        f" projection_tile {engine.projection_tile_status}"
    )
    return engine


@contextlib.contextmanager
def _engine(scope: bool | str, *, drafter: bool = True) -> Iterator[VLMEngine]:
    engine = _build(scope, drafter=drafter)
    try:
        yield engine
    finally:
        engine.unload()


def _require_the_tile() -> None:
    """Skip, naming the reason, where the tile cannot load: every kernel it rides on
    compiles here, and the static checks (the attention tile's split target, the core
    count, a plan for each of the 27B's MLP keys) read config.json alone."""
    for name, avail in (
        ("int8 prefill", int8prefill.availability()),
        ("attention tile", tileattn.availability()),
        ("projection kernel", projkernel.kernel_available()),
        ("kv8", kv8.availability()),
        ("projection tile", projtile.availability()),
    ):
        if not avail.available:
            pytest.skip(f"{name} unavailable: {avail.reason}")
    config = SousConfig(
        int8_prefill=True, attention_tile=True, projection_kernel=True, projection_tile="all"
    )
    why = projtile.static_reason(config, fetch_model_config(DEFAULT_27B), "dflash")
    if why is not None:
        pytest.skip(f"projection tile unavailable: {why}")


def _check_tile(engine: VLMEngine, scope: str) -> None:
    status = engine.projection_tile_status
    if status["state"] != "active":
        pytest.skip(f"projection tile {status['state']}: {status['reason']}")
    assert status["scope"] == scope, status
    assert engine.kv_int8_status["state"] == "active", engine.kv_int8_status
    assert engine.projection_kernel_status["state"] == "active", engine.projection_kernel_status
    assert engine.int8_prefill_status["state"] == "active", engine.int8_prefill_status


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


def _prompts(engine) -> dict[str, list[dict]]:
    ids = _corpus(engine, max(PROMPTS.values()) + 64)
    return {name: _prompt(engine, ids, target) for name, target in PROMPTS.items()}


def _root(engine: VLMEngine) -> Any:
    return int8prefill._root(engine._model)


# ---- what a decode hands back -----------------------------------------------------


class _Trace:
    """One decode: the token ids it emitted and, when recording, the raw bf16 logits
    behind them. `steps` holds what generate_step's own forward handed its logits
    processors ([1, V], before the logsumexp): every token of a drafter-off run (one
    more is computed than emitted) and only the first of a drafter run. `rounds` holds
    each greedy verify's logits ([1, T, V], before its sampler), the tokens the sampler
    took from them and the verify's input, the bonus and T - 1 drafts."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.steps: list[Any] = []
        self.rounds: list[tuple[Any, Any, Any]] = []

    def step(self, tokens: Any, logits: Any) -> Any:
        self.steps.append(logits)
        return logits


@contextlib.contextmanager
def _tap(*, logits: bool = False) -> Iterator[list[_Trace]]:
    """One _Trace per decode inside the block. VLMEngine.decode() imports
    stream_generate from the mlx_vlm package at call time, so a wrapper set there sees
    every decode's results (the last one carries the emitted ids) and can add a logits
    processor, which generate_step hands `outputs.logits[:, -1, :]` before it takes the
    logsumexp: that is every token of a drafter-off run and the first of a drafter
    run. The drafter run's later tokens come from greedy verify, whose logits reach
    their sampler inside _dflash_verify_greedy, a module global _dflash_rounds looks up
    per round. Both wrappers hand back what they wrap returned, so the decode is
    unchanged. Install after the engine loads: the kernels' load-time source pins read
    _dflash_verify_greedy (through __wrapped__, which functools.wraps sets)."""
    import mlx_vlm
    from mlx_vlm.speculative import dflash

    traces: list[_Trace] = []
    stream_generate = mlx_vlm.stream_generate
    verify_greedy = dflash._dflash_verify_greedy

    @functools.wraps(stream_generate)
    def tapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
        trace = _Trace()
        traces.append(trace)
        if logits:
            assert "logits_processors" not in kwargs, kwargs
            kwargs["logits_processors"] = [trace.step]
        for result in stream_generate(*args, **kwargs):
            if result.token_ids is not None:
                trace.ids = list(result.token_ids)
            yield result

    @functools.wraps(verify_greedy)
    def recording(lm: Any, verify_input: Any, cache: Any, layers: Any, sampler: Any) -> Any:
        trace = traces[-1]

        def sample(out_logits: Any) -> Any:
            tokens = sampler(out_logits)
            trace.rounds.append((out_logits, tokens, verify_input))
            return tokens

        return verify_greedy(lm, verify_input, cache, layers, sample)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mlx_vlm, "stream_generate", tapped)
        if logits:
            mp.setattr(dflash, "_dflash_verify_greedy", recording)
        yield traces


def _decode(engine: VLMEngine, messages: list[dict], traces: list[_Trace]) -> _Trace:
    """One greedy generate (one decode: the prompt cache is off) and its trace."""
    count = len(traces)
    engine.generate(messages, [], MAX_TOKENS)
    assert len(traces) == count + 1, "a generate ran other than one decode"
    trace = traces[-1]
    assert trace.ids, "the decode emitted no token"
    return trace


@contextlib.contextmanager
def _split_runs() -> Iterator[list[tuple[int, int]]]:
    """(runs the tile should serve, runs it served) for every verifier call the
    projection kernel cut into runs: every run when each linear of the call has a
    launch on the tile, none otherwise (the head; attention and GDN at scope "mlp")."""
    seen: list[tuple[int, int]] = []
    serve = projkernel._serve_verify

    @functools.wraps(serve)
    def counted(linears: Any, x: Any) -> Any:
        before = projkernel.calls["verify_tile"]
        out = serve(linears, x)
        rows = x.shape[-2] if getattr(x, "ndim", 0) == 3 else 0
        if out is not None and rows > projkernel.MAX_ROWS:
            servable = projtile.plan_for_call(linears) is not None
            want = -(-rows // projkernel.MAX_ROWS) if servable else 0
            seen.append((want, projkernel.calls["verify_tile"] - before))
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(projkernel, "_serve_verify", counted)
        yield seen


# ---- block invariance -------------------------------------------------------------


def test_greedy_output_is_block_invariant_on_the_tile():
    _require_the_tile()
    for scope in SCOPES:
        with _engine(scope) as engine:
            _check_tile(engine, scope)
            assert engine._draft_block_size == 8
            drafter = engine._draft
            assert drafter is not None
            prompts = _prompts(engine)
            ids: dict[tuple[int, str], list[int]] = {}
            counts: dict[int, dict[str, int]] = {}
            splits: dict[int, list[tuple[int, int]]] = {}
            spec: dict[tuple[int, str], Any] = {}
            with _tap() as traces, _split_runs() as seen:
                for block in BLOCKS:
                    engine._draft_block_size = block
                    if block == 0:
                        # What a load at block 0 leaves: _pin_block_size pins only a block
                        # above 0, so the drafter's own policy picks every round's depth.
                        drafter.prefer_requested_block_size = False
                    before, cut = dict(projkernel.calls), len(seen)
                    started = time.perf_counter()
                    for name, messages in prompts.items():
                        ids[(block, name)] = _decode(engine, messages, traces).ids
                        spec[(block, name)] = engine.speculation()
                    counts[block] = {k: v - before[k] for k, v in projkernel.calls.items()}
                    splits[block] = seen[cut:]
                    print(
                        f"scope {scope} block {block}: {time.perf_counter() - started:.1f} s,"
                        + "".join(
                            f" {name} {len(ids[(block, name)])} tokens"
                            f" rounds {spec[(block, name)].rounds}"
                            f" drafted {spec[(block, name)].drafted}"
                            f" accepted {spec[(block, name)].accepted}"
                            f" kv8 {spec[(block, name)].kv8};"
                            for name in prompts
                        )
                        + f" calls {counts[block]};"
                        f" split calls {len(splits[block])},"
                        f" runs on the tile {sum(got for _, got in splits[block])}"
                    )

        for block in BLOCKS[1:]:
            for name in prompts:
                got, want = ids[(block, name)], ids[(8, name)]
                first = next(
                    (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None
                )
                assert got == want, (scope, f"block {block}", name, "first difference", first)
        for block in BLOCKS:
            assert counts[block]["verify_tile"] > 0, (scope, block, counts[block])
            assert counts[block]["declined"] == 0, (scope, block, counts[block])
            assert (counts[block]["verify_split"] > 0) == (block == SPLIT_BLOCK), (
                scope,
                block,
                counts[block],
            )
        # Block 2 drafts one token a round, so its verify is two rows throughout.
        assert all(spec[(2, name)].drafted == spec[(2, name)].rounds for name in prompts)
        # The long prompt's generation sits between kv8's floor and its ceiling.
        assert all(spec[(block, "long")].kv8 == "served" for block in BLOCKS), spec
        cut = splits[SPLIT_BLOCK]
        assert cut, f"scope {scope}: block {SPLIT_BLOCK} cut no verifier call into runs"
        assert sum(want for want, _ in cut) > 0, (scope, "no cut call was the tile's")
        assert all(got == want for want, got in cut), (scope, [c for c in cut if c[0] != c[1]])


# ---- the bf16 floor against float64 -----------------------------------------------

# The float64 reference and its element bound, from the spike's accuracy checker. The
# reference dequantises exactly (a bf16 scale times a 4-bit code is a 12-bit product) and
# multiplies in float64; `mag` is the sum of the absolute terms the kernel's fp32
# arithmetic adds, |x| . (|scale| q + |bias|), since scale * (x . q) and bias * sum(x) can
# cancel. The bound, |y - ref| <= ulp_bf16(|ref|) + (64 + 2G + 4) 2^-24 mag + K 2^-126,
# covers the one bf16 rounding, any order of a group's 64-product dot and of its row sum,
# the epilogue's 2G fmas (a split plan's chain is shorter: G / SPLITS groups' fmas plus
# SPLITS - 1 adds in the reduce) and fp32 denormal flushing: any correct fp32 evaluation
# meets it, so a wrong scale, bias, code or column fails it.
_U = 2.0**-24
_CHUNK = 1 << 22  # weights dequantised per float64 chunk: 32 MB


def _f64(a: Any) -> Any:
    return np.array(a.astype(mx.float32), dtype=np.float64)


def _codes(w: Any) -> Any:
    """uint32 [C, K/8] -> uint8 [C, K]; nibble i of word j is input 8j + i, low first."""
    c = np.empty(w.shape + (8,), dtype=np.uint8)
    for i in range(8):
        c[..., i] = (w >> np.uint32(4 * i)) & np.uint32(15)
    return c.reshape(w.shape[0], -1)


def _reference(x: Any, w: Any, scales: Any, biases: Any) -> tuple[Any, Any]:
    """(ref, mag), float64 [R, N], from x float64 [R, K], w uint32 [N, K/8] and the
    scales and biases float64 [N, K/64]."""
    ax = np.abs(x)
    n, k = w.shape[0], w.shape[1] * 8
    groups = k // slab.GROUP
    ref = np.empty((x.shape[0], n))
    mag = np.empty_like(ref)
    step = max(64, _CHUNK // k)
    for a in range(0, n, step):
        z = min(n, a + step)
        q = _codes(w[a:z]).astype(np.float64).reshape(z - a, groups, slab.GROUP)
        s, b = scales[a:z, :, None], biases[a:z, :, None]
        ref[:, a:z] = x @ (q * s + b).reshape(z - a, k).T
        mag[:, a:z] = ax @ (q * np.abs(s) + np.abs(b)).reshape(z - a, k).T
    return ref, mag


def _ulp_bf16(a: Any) -> Any:
    a = np.maximum(np.abs(a), 2.0**-126)
    return np.exp2(np.floor(np.log2(a)) - 7)


def _allowed(ref: Any, mag: Any, k: int) -> Any:
    return _ulp_bf16(ref) + (64 + 2 * (k // slab.GROUP) + 4) * _U * mag + k * 2.0**-126


def _bf16_round(ref: Any) -> Any:
    """The nearest bf16 to each value, ties to even, in float64 (no double rounding
    through fp32): the least error any bf16 output can have."""
    u = _ulp_bf16(ref)
    return np.round(ref / u) * u


def _rel_rms(y: Any, ref: Any) -> float:
    d = y - ref
    return math.sqrt(float((d * d).sum()) / max(float((ref * ref).sum()), 1e-300))


def _fault_caught(y: Any, ref: Any, mag: Any, k: int) -> bool:
    """The bound's power: one output column replaced by another's must violate it."""
    y = y.copy()
    col = y.shape[1] - 1
    y[:, col] = y[:, (col + 4097) % y.shape[1]]
    return bool((~(np.abs(y - ref) <= _allowed(ref, mag, k))).any())


def _floor_rows(k: int) -> tuple[Any, Any]:
    """Eight bf16 rows of K (normal, a wide exponent spread, sparse outliers, small, sparse
    with -0s, large, half-scale, a narrower spread), as mlx and float64, fixed per K."""
    rng = np.random.default_rng(k)
    sparse = rng.standard_normal(k) * (rng.random(k) < 0.4)
    sparse[rng.random(k) < 0.1] = -0.0
    rows = np.stack(
        [
            rng.standard_normal(k),
            rng.standard_normal(k) * np.exp2(rng.integers(-12, 8, size=k)),
            rng.standard_normal(k) + (rng.random(k) < 0.01) * rng.standard_normal(k) * 200,
            rng.standard_normal(k) * 0.03,
            sparse,
            rng.standard_normal(k) * 64,
            rng.standard_normal(k) * 0.5,
            rng.standard_normal(k) * np.exp2(rng.integers(-6, 4, size=k)),
        ]
    )
    x = mx.array(rows.astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x)
    return x, _f64(x)


def _mlx_form(module: Any) -> tuple[Any, Any, Any]:
    """A linear's weights in mlx's layout as numpy: a slab-only linear's planes unpacked
    (the conversion is bitwise), a copied one's own arrays, which the stack was built
    from, so a stack that copied them wrong fails against them."""
    info = slab.slab_info(module)
    arrays = (module.weight, module.scales, module.biases)
    if info is not None:
        arrays = slab.unpack(*arrays, info)
    w, scales, biases = (mx.contiguous(a) for a in arrays)
    return np.array(w), _f64(scales), _f64(biases)


def _layer_calls(layer: Any) -> list[tuple[Any, ...]]:
    """A layer's verifier calls (gate+up, down, and q+k+v and o_proj or the four GDN
    input projections and out_proj), then every linear of a fused call alone, as the
    decode's own two-to-eight-row calls reach it."""
    mlp = layer.mlp
    fused: list[tuple[Any, ...]] = [(mlp.gate_proj, mlp.up_proj), (mlp.down_proj,)]
    if layer.is_linear:
        gdn = layer.linear_attn
        fused += [(gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a), (gdn.out_proj,)]
    else:
        attention = layer.self_attn
        fused += [(attention.q_proj, attention.k_proj, attention.v_proj), (attention.o_proj,)]
    return fused + [(m,) for call in fused if len(call) > 1 for m in call]


def test_the_tile_sits_at_the_bf16_floor_on_every_linear():
    _require_the_tile()
    # The real kernels, not a test's stand-ins.
    assert projtile.tile is projtile.run and projtile.tile_dual is projtile.run_dual
    with _engine("all") as engine:
        _check_tile(engine, "all")
        root = _root(engine)
        names = {id(m): path for path, m in root.named_modules()}
        rows: dict[int, tuple[Any, Any]] = {}
        failures: list[str] = []
        ratios: list[float] = []
        checked = linears = 0
        started = time.perf_counter()
        for layer in root.model.layers:
            refs: dict[int, tuple[Any, Any]] = {}
            for call in _layer_calls(layer):
                launches = projtile.plan_for_call(call)
                label = "+".join(names[id(m)] for m in call)
                assert launches is not None, f"no launch on the tile for {label}"
                k = projkernel._in_features(call[0])
                if k not in rows:
                    rows[k] = _floor_rows(k)
                x, x64 = rows[k]
                outs = projtile.verify(call, x, launches)
                mx.eval(outs)
                for module, y in zip(call, outs, strict=True):
                    if id(module) not in refs:
                        refs[id(module)] = _reference(x64, *_mlx_form(module))
                        linears += 1
                    ref, mag = refs[id(module)]
                    got = _f64(y)
                    where = f"{names[id(module)]} in {label}"
                    violations = int((~(np.abs(got - ref) <= _allowed(ref, mag, k))).sum())
                    error, floor = _rel_rms(got, ref), _rel_rms(_bf16_round(ref), ref)
                    ratios.append(error / floor)
                    checked += 1
                    if violations:
                        failures.append(f"{where}: {violations} elements past the bound")
                    if not error <= projtile.TILE_FLOOR_FACTOR * floor:
                        failures.append(f"{where}: rel RMS {error:.4e}, floor {floor:.4e}")
                    if not _fault_caught(got, ref, mag, k):
                        failures.append(f"{where}: a wrong column passes the bound")
        print(
            f"bf16 floor: {linears} linears, {checked} outputs (fused and alone) in"
            f" {time.perf_counter() - started:.1f} s; rel RMS / floor"
            f" {min(ratios):.6f}..{max(ratios):.6f} (bar {projtile.TILE_FLOOR_FACTOR})"
        )
        assert not failures, failures


# ---- the slab-only MLP against a tile-off engine ----------------------------------


def _mlp_inputs(root: Any) -> dict[tuple[int, int], Any]:
    """[1, t, K] rows per (K, t), uniform on [-1, 1]. Per-row int8 activation noise is
    what separates int8 from stock: on normal rows, whose amax runs to 4 sigma over these
    K, it is about 0.95e-2 relative, at the bound itself; on uniform rows about 0.4e-2,
    so the bound still parts that noise from a wrong slab read (relative error near 1)."""
    ks = sorted({projkernel._in_features(getattr(root.model.layers[0].mlp, n)) for n in _MLP})
    out: dict[tuple[int, int], Any] = {}
    for i, k in enumerate(ks):
        x = mx.random.uniform(-1, 1, (1, max(SLAB_EQUAL_ROWS), k), key=mx.random.key(500 + i))
        x = x.astype(mx.bfloat16)
        for t in SLAB_EQUAL_ROWS + SLAB_INT8_ROWS:
            out[(k, t)] = mx.contiguous(x[:, :t])
    mx.eval(list(out.values()))
    return out


def _mlp_outputs(
    root: Any, inputs: dict[tuple[int, int], Any]
) -> Iterator[tuple[tuple[int, str, int], Any]]:
    """Every layer's gate, up and down called as modules at every row count, a layer at
    a time, evaluated."""
    for i, layer in enumerate(root.model.layers):
        outs = {}
        for name in _MLP:
            module = getattr(layer.mlp, name)
            k = projkernel._in_features(module)
            for t in SLAB_EQUAL_ROWS + SLAB_INT8_ROWS:
                outs[(i, name, t)] = module(inputs[(k, t)])
        mx.eval(list(outs.values()))
        yield from outs.items()


def _counts() -> dict[str, int]:
    return {**projkernel.calls, **int8prefill.calls}


def test_the_slab_only_mlp_paths_equal_a_tile_off_engine():
    _require_the_tile()
    with _engine(False) as off:
        assert off.projection_tile_status["state"] == "off", off.projection_tile_status
        root = _root(off)
        layers = len(root.model.layers)
        assert all(
            slab.slab_info(getattr(layer.mlp, n)) is None
            for layer in root.model.layers
            for n in _MLP
        )
        inputs = _mlp_inputs(root)
        before = _counts()
        reference = dict(_mlp_outputs(root, inputs))
        off_counts = {k: v - before[k] for k, v in _counts().items()}

    with _engine("mlp") as engine:
        _check_tile(engine, "mlp")
        root = _root(engine)
        assert all(
            slab.slab_info(getattr(layer.mlp, n)) is not None
            for layer in root.model.layers
            for n in _MLP
        )
        before = _counts()
        unequal: list[tuple[int, str, int]] = []
        distance: dict[tuple[str, int], float] = {}
        for key, got in _mlp_outputs(root, inputs):
            want = reference.pop(key)
            _, name, t = key
            if t in SLAB_EQUAL_ROWS:
                if not mx.array_equal(got, want).item():
                    unequal.append(key)
            else:
                d = projkernel._stock_distance(got, want)
                distance[(name, t)] = max(distance.get((name, t), 0.0), d)
        tile_counts = {k: v - before[k] for k, v in _counts().items()}

    print(
        "slab-only MLP against stock, worst rel RMS over the layers:"
        + "".join(f" {name}@{t} {d:.3e};" for (name, t), d in sorted(distance.items()))
        + f" tile-off counts {off_counts}; tile counts {tile_counts}"
    )
    assert not reference
    assert not unequal, unequal[:20]
    assert all(d <= projkernel.REL_RMS_BOUND for d in distance.values()), distance
    # Outside a decode window nothing reaches the tile; the slab-only module's 9-127-row
    # calls are int8's, one each, and the tile-off engine's are stock's.
    for counts in (off_counts, tile_counts):
        assert counts["verify_tile"] == counts["plain_tile"] == counts["declined"] == 0, counts
    assert off_counts["int8_small"] == 0, off_counts
    assert tile_counts["int8_small"] == layers * len(_MLP) * len(SLAB_INT8_ROWS), tile_counts


# ---- drafter on against drafter off -----------------------------------------------


def _greedy(row: Any) -> int:
    """The token generate_step samples from one [1, V] row of raw logits at temperature
    0: the argmax of the bf16 log-probabilities, as it computes them."""
    return int(mx.argmax(row - mx.logsumexp(row, axis=-1, keepdims=True), axis=-1).item())


def _undrafted(trace: _Trace) -> tuple[list[int], Any]:
    """(ids, [P, V] raw logits) of a drafter-off decode, one row per emitted token."""
    ids = trace.ids
    assert len(trace.steps) >= len(ids), (len(trace.steps), len(ids))
    assert not trace.rounds
    steps = trace.steps[: len(ids)]
    assert [_greedy(row) for row in steps] == ids, "the logits do not give the emitted tokens"
    return ids, mx.concatenate(steps)


def _drafted(trace: _Trace) -> tuple[list[int], Any]:
    """(ids, [P, V] raw logits) of a drafter decode: the first token's row from the
    decode's own forward, then each round's verify rows as the greedy walk commits them,
    the accepted drafts and the target's token after them, cut at the token budget."""
    ids = trace.ids
    assert len(trace.steps) == 1, len(trace.steps)
    rows, tokens = [trace.steps[0]], [_greedy(trace.steps[0])]
    for logits, targets, verify_input in trace.rounds:
        drafts = verify_input[0, 1:].tolist()
        target = targets.reshape(-1).tolist()
        accepted = next(
            (i for i, (d, t) in enumerate(zip(drafts, target, strict=False)) if d != t), len(drafts)
        )
        take = min(accepted + 1, MAX_TOKENS - len(tokens))
        rows.append(logits[0, :take])
        tokens += target[:take]
    assert tokens[: len(ids)] == ids, "the verify rows do not give the emitted tokens"
    return ids, mx.concatenate(rows)[: len(ids)]


def _top2(rows: Any) -> tuple[Any, Any]:
    """The two largest raw logits of every row, float64: (first, second)."""
    x = rows.astype(mx.float32)
    first = mx.argmax(x, axis=-1)
    rest = mx.where(mx.arange(x.shape[-1])[None] == first[:, None], -mx.inf, x)
    l1 = mx.take_along_axis(x, first[:, None], axis=-1)[:, 0]
    l2 = mx.take_along_axis(x, mx.argmax(rest, axis=-1)[:, None], axis=-1)[:, 0]
    return np.array(l1, dtype=np.float64), np.array(l2, dtype=np.float64)


def test_drafter_on_against_drafter_off_within_the_measured_perturbation():
    _require_the_tile()
    runs: dict[str, dict[str, tuple[list[int], Any]]] = {}
    with _engine("all", drafter=False) as plain:
        status = plain.projection_tile_status
        assert (status["state"], status["reason"]) == ("unavailable", "no drafter"), status
        assert plain.kv_int8_status["state"] == "active", plain.kv_int8_status
        prompts = _prompts(plain)
        with _tap(logits=True) as traces:
            runs["off"] = {
                name: _undrafted(_decode(plain, m, traces)) for name, m in prompts.items()
            }
    with _engine("all") as engine:
        _check_tile(engine, "all")
        assert engine._draft_block_size == 8
        before = dict(projkernel.calls)
        with _tap(logits=True) as traces:
            runs["on"] = {name: _drafted(_decode(engine, m, traces)) for name, m in prompts.items()}
        assert projkernel.calls["verify_tile"] > before["verify_tile"]

    # D: the change in each run's top-two margin, over every position before the
    # prompt's first divergence, pooled over the prompts.
    deltas, firsts = [], {}
    for name in prompts:
        (off_ids, off_rows), (on_ids, on_rows) = runs["off"][name], runs["on"][name]
        n = min(len(off_ids), len(on_ids))
        first = next((i for i in range(n) if off_ids[i] != on_ids[i]), None)
        assert first is not None or len(off_ids) == len(on_ids), (name, len(off_ids), len(on_ids))
        firsts[name] = first
        stop = n if first is None else first
        if stop:
            off_1, off_2 = _top2(off_rows[:stop])
            on_1, on_2 = _top2(on_rows[:stop])
            deltas.append(np.abs((on_1 - on_2) - (off_1 - off_2)))
    assert deltas, "no position before a divergence to measure the perturbation on"
    pooled = np.concatenate(deltas)
    d = float(np.percentile(pooled, 99.9))
    assert math.isfinite(d), pooled
    print(
        f"drafter on against off: D {d:.4g} over {pooled.size} positions"
        f" (max {pooled.max():.4g}, nonzero {int((pooled > 0).sum())})"
    )

    failures = []
    for name, p in firsts.items():
        if p is None:
            print(f"  {name}: {len(runs['off'][name][0])} tokens identical")
            continue
        (off_ids, off_rows), (on_ids, on_rows) = runs["off"][name], runs["on"][name]
        a, b = off_ids[p], on_ids[p]
        off = off_rows[p].astype(mx.float32)
        on = on_rows[p].astype(mx.float32)
        top = float(mx.max(off).item())
        margin = abs(off[a].item() - off[b].item())
        bar = max(d, 2 * float(_ulp_bf16(abs(top))))
        shift = (abs(on[a].item() - off[a].item()), abs(on[b].item() - off[b].item()))
        print(
            f"  {name}: first divergence at token {p}, off emits {a} and on {b};"
            f" off margin {margin:.4g} (bar {bar:.4g}, top logit {top:.4g});"
            f" on - off for the two tokens {shift[0]:.4g}, {shift[1]:.4g} (bar D {d:.4g})"
        )
        if not (margin <= bar and shift[0] <= d and shift[1] <= d):
            failures.append((name, p, a, b, margin, bar, shift, d))
    assert not failures, failures


# ---- the status document and the load line ----------------------------------------


def _managed(config: SousConfig, scope: str) -> tuple[EngineManager, list[VLMEngine]]:
    """An EngineManager whose factory builds the scope's engine, and the engines it
    built. Loaded through it as the daemon loads: on a sous-model-load thread, whose end
    releases its mlx state, with the load line written once the load returns."""
    built: list[VLMEngine] = []

    def factory(model_id: str) -> VLMEngine:
        assert model_id == DEFAULT_27B
        built.append(_build(scope))
        return built[-1]

    return EngineManager(config, factory), built


def test_the_status_and_load_line_name_the_scope(tmp_path, caplog):
    _require_the_tile()
    caplog.set_level(logging.INFO, logger="sous.engine")
    copy = projtile.static_copy_bytes(fetch_model_config(DEFAULT_27B))
    config = SousConfig(data_dir=tmp_path / "data", config_path=tmp_path / "config.toml")
    for scope in SCOPES:
        manager, built = _managed(config, scope)
        caplog.clear()
        manager.get()
        try:
            (engine,) = built
            loads = [
                r.getMessage() for r in caplog.records if r.getMessage().startswith("model_load ")
            ]
            assert len(loads) == 1, loads
            line, status = loads[0], manager.status()
            print(line)
            if engine.projection_tile_status["state"] != "active":
                pytest.skip(f"projection tile: {engine.projection_tile_status}")
            tokens = line.split()
            block = status["projection_tile"]
            assert set(block) == {"state", "scope", "reason", "copy_gb", "probe_seconds"}, block
            assert (block["state"], block["scope"]) == ("active", scope), block
            assert isinstance(block["probe_seconds"], float) and block["probe_seconds"] > 0, block
            assert status["kv_int8"]["state"] == "active", status["kv_int8"]
            assert status["projection_kernel"]["state"] == "active", status["projection_kernel"]
            assert status["draft_block"] == 8, status
            for token in (
                f"projection_tile=active:{scope}",
                f"projection_tile_probe_s={block['probe_seconds']}",
                "projection_kernel=active",
                "projection_kernel_drafter=active",
                "draft_block=8",
                "kv_int8=active",
            ):
                assert token in tokens, (token, line)
            assert any(t.startswith("projection_tile_budget_gib=") for t in tokens), line
            if scope == "all":
                stacked = projtile.copy_bytes(_root(engine), keys_with_plans=projtile._has_plan)
                assert stacked == copy, (stacked, copy)
                assert block["copy_gb"] == copy / 1e9 and block["reason"] is None, block
                assert f"projection_tile_copy_gb={copy / 1e9:.1f}" in tokens, line
                assert "projection_tile_copy=skipped" not in tokens, line
            else:
                assert block["copy_gb"] is None, block
                assert block["reason"] == "scope mlp by config", block
                assert "projection_tile_copy=skipped" in tokens, line
                assert not any(t.startswith("projection_tile_copy_gb=") for t in tokens), line
        finally:
            assert manager.unload_now()["unloaded"]
