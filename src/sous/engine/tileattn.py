"""A tensor-unit attention tile for qwen3_5's one-row decode and exact verifier.

The six query heads that share a KV head are packed with a call's rows into MPP
``matmul2d`` fragments on the M5's tensor units: a split kernel per (key split,
KV head) with an fp32 online softmax in registers, then a reduce kernel that
combines the splits in a fixed order. The kernels (``kernels/attention_tile.metal``)
are specialised to bf16, 24 query / 4 KV heads, head dim 256 and scale 1/16.

Exactness rests on one property: a row's output depends on its key partition,
the chunk each split covers, and on nothing else about the call. Its 32-key
blocks align the same way in every call, keys past its limit get probability
exactly 0, and a (row, key) product does not depend on the fragment row it sits
in. So the chunk is fixed within each step of 32 * S keys (S the GPU's core
count, one wave of threadgroups); a verify call is cut into runs that never
cross N0, a step boundary or 8 rows; and below N0 decode and verify both take
stock paths that are exact against each other. Every verify row then equals a
one-row decode at its own length, so greedy output with a drafter equals greedy
output without one, whatever the block size.

Two hooks reach it. One-row decode goes through a replacement of the
``scaled_dot_product_attention`` global in mlx-vlm's qwen3_5 language module,
the call ``Qwen3_5Attention.__call__`` makes after its projections; verify rows
come from verifyattn's wrapper of the exact verifier. Both decide after the
projections and fall back to the very call the model would have made, and both
act only while ``_active_splits`` is set. Only an ``enable()`` that passed every
gate and the load-time probe sets it; every ``enable()`` and every unload clears
it. Nothing here raises into a model load.
"""

from __future__ import annotations

import functools
import importlib
import logging
import time
import traceback
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from sous.engine import gpucores, nax
from sous.engine.int8prefill import _error_line, _kernel, _kernel_text, _model_type

logger = logging.getLogger("sous.engine.tileattn")

HQ = 24
HKV = 4
GQA = 6
D = 256
# D ** -0.5, compiled into the kernel as SCALE_LOG2.
SCALE = 0.0625
# The split kernel is instantiated for T = 1..MAX_RUN rows; longer verify calls are cut.
MAX_RUN = 8
# The first key count the tile serves. Below it stock one-row SDPA is as fast (T = 1
# wall time against stock: 0.97x at 1024 keys, 1.02-1.04x at 1101-1281, 1.08x at
# 1536, 1.11x at 2048).
N0 = 1536
# The kernel's key block: a chunk is a multiple of it, and a step is GRAN * S keys.
GRAN = 32
# Split targets measured for speed: 20, the M5 Pro's GPU core count.
MEASURED_SPLITS = frozenset({20})
# attention_tile.metal holds the split body, this line, then the reduce body.
_REDUCE_MARK = "// REDUCE\n"
# mlx's default, stated rather than inherited: parity between the row variants needs
# each to round exactly as written, which relaxed and fast math do not promise.
_SAFE_MATH = {"math_mode": "safe"}

# The active split target S, or 0: set only by an enable() that reached active,
# cleared first by every enable() and by clear(). A plain int, so nothing it holds
# outlives an unload.
_active_splits = 0
# Set on the decode wrapper so a second install never wraps it again.
_HOOK_MARK = "_sous_attention_tile"
# Per attention call (one layer of one forward), not per turn: the tests read deltas.
# Nothing is logged per turn.
calls: dict[str, int] = {
    "decode_tile": 0,  # one-row call served by the kernel (n >= N0)
    "decode_stock": 0,  # one-row call in scope below N0: mlx-vlm's function answered
    "decode_out": 0,  # one-row call while active but out of scope: mlx-vlm's function
    "verify_calls": 0,
    "verify_rows_tile": 0,
    "verify_rows_stock": 0,  # rows below N0, through grouped_attention
    "verify_launches": 0,  # tile runs
    "verify_straddles": 0,  # calls cut into more than one run
    "verify_declined": 0,  # in scope before the projections, refused after them
    "verify_out": 0,  # tagged verifier calls while active but out of scope
    "verify_t1": 0,
    "verify_t2": 0,
    "verify_t3": 0,
    "verify_t4": 0,
    "verify_t5": 0,
    "verify_t6": 0,
    "verify_t7": 0,
    "verify_t8": 0,
    "verify_t9_up": 0,
}


def chunk_for(n: int, splits: int) -> int:
    """The split size for a row that sees n keys; 0 means the stock path. Step k
    covers n in (GRAN*S*(k-1), GRAN*S*k] and uses chunk GRAN*k, so no row gets more
    than S splits: S splits x HKV heads is one wave of threadgroups on S cores."""
    if n < N0:
        return 0
    return GRAN * -(-n // (GRAN * splits))


def step_of(n: int, splits: int) -> tuple[int, int, int]:
    """(first n, last n, chunk) of the step that holds key count n."""
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return (1, N0 - 1, 0)
    k = chunk // GRAN
    return (max(N0, GRAN * splits * (k - 1) + 1), GRAN * splits * k, chunk)


def boundaries(n_max: int, splits: int) -> list[int]:
    """Every key count B <= n_max whose step differs from B - 1's, N0 first."""
    out = [N0]
    n = N0
    while True:
        n = step_of(n, splits)[1] + 1
        if n > n_max:
            return out
        out.append(n)


def verify_plan(prefix: int, t: int, splits: int) -> list[tuple[int, int, int]]:
    """A T-row verify behind `prefix` keys as runs (j, k, chunk) of rows [j, k);
    row r sees prefix + r + 1 keys and chunk 0 means stock. A run never crosses N0
    or a step boundary and never holds more than MAX_RUN rows, so any T is served."""
    runs: list[tuple[int, int, int]] = []
    j = 0
    while j < t:
        chunk = chunk_for(prefix + j + 1, splits)
        k = j + 1
        while k < t and k - j < MAX_RUN and chunk_for(prefix + k + 1, splits) == chunk:
            k += 1
        runs.append((j, k, chunk))
        j = k
    return runs


def _kernel_pair() -> tuple[Callable[..., list[Any]], Callable[..., list[Any]]]:
    """The split and reduce kernels, built on first use. The split kernel reads K/V
    in place through their strides: a KVCache hands back a slice of its padded
    buffer, which ensure_row_contiguous would copy whole on every call. Only the
    split kernel uses the tensor ops, so only its header carries nax.h."""
    split_source, reduce_source = _kernel_text("attention_tile.metal").split(_REDUCE_MARK)
    common = _kernel_text("common.h")
    split = _kernel(
        "sous_attention_tile_split",
        ["q", "k", "v", "params"],
        ["part", "stats"],
        split_source,
        common + _kernel_text("nax.h"),
        ensure_row_contiguous=False,
        compile_options=_SAFE_MATH,
    )
    reduce = _kernel(
        "sous_attention_tile_reduce",
        ["part", "stats", "params"],
        ["out"],
        reduce_source,
        common,
        compile_options=_SAFE_MATH,
    )
    return split, reduce


def tile(queries: Any, keys: Any, values: Any, n: int, chunk: int) -> Any:
    """One kernel call over T <= MAX_RUN rows: row t sees keys [0, n - T + t], every
    K/V read is clamped to n - 1, and keys are split at multiples of `chunk`.
    Queries are [1, 24, T, 256], K/V [1, 4, >= n, 256], all bf16 with innermost
    stride 1. Returns the contiguous [1, T, 24, 256] bf16 output."""
    import mlx.core as mx

    split, reduce = _kernel_pair()
    t = queries.shape[2]
    mr = GQA * t
    rgt = -(-mr // 16)
    nsplit = -(-n // chunk)
    params = mx.array([n, chunk, nsplit], dtype=mx.int32)
    part, stats = split(
        inputs=[queries, keys, values, params],
        template=[("TR", t), ("MR", mr), ("RGT", rgt)],
        grid=(64 * rgt, nsplit, HKV),
        threadgroup=(64 * rgt, 1, 1),
        output_shapes=[(HKV * nsplit * mr * D,), (HKV * nsplit * mr * 2,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = reduce(
        inputs=[part, stats, params],
        template=[("MR", mr)],
        grid=(D, mr, HKV),
        threadgroup=(D, 1, 1),
        output_shapes=[(1, t, HQ, D)],
        output_dtypes=[mx.bfloat16],
    )
    return out


@functools.cache
def _arch() -> str:
    """This process's GPU architecture, for grouped_attention's plan. Read here, never
    from verifyattn._ARCH, which is set only when verifyattn's own probe passed."""
    import mlx.core as mx

    return str(mx.device_info().get("architecture", ""))


def in_scope(queries: Any, keys: Any, values: Any, scale: float) -> bool:
    """What decode() and verify() assume of the projected tensors; both hooks check
    it after the projections."""
    import mlx.core as mx

    return (
        isinstance(queries, mx.array)
        and isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[0] == keys.shape[0] == values.shape[0] == 1
        and queries.shape[1] == HQ
        and keys.shape[1] == values.shape[1] == HKV
        and queries.shape[3] == keys.shape[3] == values.shape[3] == D
        and queries.dtype == keys.dtype == values.dtype == mx.bfloat16
        and queries.shape[2] >= 1
        and keys.shape[2] == values.shape[2] >= queries.shape[2]
        and scale == SCALE
    )


def decode(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The one-row call over all n = keys.shape[2] keys, [1, 24, 1, 256]. Stock one-row
    SDPA below N0, where the decode hook never calls this; the probe and the tests
    compare against it there."""
    import mlx.core as mx

    n = keys.shape[2]
    chunk = chunk_for(n, splits)
    if chunk == 0:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=SCALE, mask=None)
    return tile(queries, keys, values, n, chunk).transpose(0, 2, 1, 3)


def verify(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """The verifier's T-row causal call, row r over keys [0, n - T + r], as
    [1, 24, T, 256]: one tile call per run of verify_plan(), grouped stock calls for
    the runs below N0, concatenated in row order."""
    import mlx.core as mx

    from sous.engine.verifyattn import grouped_attention

    t = queries.shape[2]
    n = keys.shape[2]
    prefix = n - t
    runs = verify_plan(prefix, t, splits)
    if len(runs) == 1 and runs[0][2]:
        return tile(queries, keys, values, n, runs[0][2]).transpose(0, 2, 1, 3)
    parts = []
    for j, k, chunk in runs:
        q = queries if (j, k) == (0, t) else queries[:, :, j:k, :]
        kk = keys if k == t else keys[:, :, : prefix + k, :]
        vv = values if k == t else values[:, :, : prefix + k, :]
        if chunk:
            parts.append(tile(q, kk, vv, prefix + k, chunk))
        else:
            stock = grouped_attention(q, kk, vv, SCALE, prefix + j, _arch())
            parts.append(stock.transpose(0, 2, 1, 3))
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.transpose(0, 2, 1, 3)


_compile_ok = False


def _compile_probe() -> str | None:
    """Compile and run every row variant, T = 1..MAX_RUN, on serving-layout inputs;
    None when they all work, else the compiler's error line. The kernel is compiled
    at runtime against the OS's own Metal toolchain, so a new macOS can reject a
    source that compiled before. The inputs are built with GPU ops and evaluated
    first: a compile failure with a CPU-stream op in flight deadlocks inside mlx's
    exception path (0.32.2) instead of raising. Only success is remembered, so one
    bad moment cannot switch the tile off for a long-lived daemon."""
    global _compile_ok
    if _compile_ok:
        return None
    import mlx.core as mx

    try:
        q_key, k_key, v_key = mx.random.split(mx.random.key(0), 3)
        rows = mx.random.normal((1, MAX_RUN, HQ, D), key=q_key).astype(mx.bfloat16)
        # One 256-token step past N0: the K/V are slices of a padded buffer, as served.
        keys = mx.random.normal((1, HKV, N0 + 256, D), key=k_key).astype(mx.bfloat16)
        values = mx.random.normal((1, HKV, N0 + 256, D), key=v_key).astype(mx.bfloat16)
        mx.eval(rows, keys, values)
        queries = rows.transpose(0, 2, 1, 3)
        chunk = chunk_for(N0, min(MEASURED_SPLITS))
        for t in range(1, MAX_RUN + 1):
            mx.eval(tile(queries[:, :, :t], keys[:, :, :N0], values[:, :, :N0], N0, chunk))
    except Exception as e:  # noqa: BLE001 — any failure means the kernel cannot serve
        # The one-line reason cannot carry a compiler's full report; the log can.
        logger.warning("attention tile: the kernel failed to build or run:\n%s", e)
        return _error_line(str(e))
    _compile_ok = True
    return None


def availability() -> nax.Availability:
    """The tensor-unit rule the tile shares with int8 prefill, then a compile and run
    of every row variant. The `nax` tests skip on this, never on int8's: the two
    kernels can fail to compile on different toolchains."""
    reason = nax.platform_reason()
    if reason is not None:
        return nax.Availability(False, reason)
    failed = _compile_probe()
    if failed is not None:
        return nax.Availability(False, f"the attention tile kernel failed: {failed}")
    return nax.Availability(True)


def clear() -> None:
    """Stop serving: both hooks pass every call through from here on."""
    global _active_splits
    _active_splits = 0


def _bind(queries, keys, values, cache, scale, mask, sinks=None) -> tuple:
    """mlx-vlm's base.scaled_dot_product_attention parameter list, which verifyattn's
    source gate pins."""
    return queries, keys, values, cache, scale, mask, sinks


def _plain_kv_cache(cache: Any) -> bool:
    """mlx-vlm's single-sequence KVCache itself (its subclasses keep other layouts),
    unquantized and without left padding. The left-padded helpers hand the global a
    None or batch cache, so they never pass."""
    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache
    return (
        type(cache) is kv_cache
        and not hasattr(cache, "bits")
        and getattr(cache, "_qwen3_5_decode_left_padding", None) is None
        and getattr(cache, "left_padding", None) is None
    )


def _serve_decode(args: tuple, kwargs: dict) -> Any | None:
    """The tile's output for a one-row call it serves, else None, and the wrapper
    then makes the very call Qwen3_5Attention.__call__ made: deciding after the
    projections never appends to the cache twice. Multi-row (prefill) calls and
    calls while the flag is clear are not counted."""
    splits = _active_splits
    if not splits:
        return None
    try:
        queries, keys, values, cache, scale, mask, sinks = _bind(*args, **kwargs)
    except TypeError:
        return None
    if getattr(queries, "ndim", 0) != 4 or queries.shape[2] != 1:
        return None
    if not (
        mask is None
        and sinks is None
        and _plain_kv_cache(cache)
        and in_scope(queries, keys, values, scale)
    ):
        calls["decode_out"] += 1
        return None
    if keys.shape[2] < N0:
        calls["decode_stock"] += 1
        return None
    calls["decode_tile"] += 1
    return decode(queries, keys, values, splits)


def _decode_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """mlx-vlm's attention function with the one-row calls the tile serves taken
    out; every other call reaches `original` with its own arguments."""

    @functools.wraps(original)
    def scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Any:
        out = _serve_decode(args, kwargs)
        return original(*args, **kwargs) if out is None else out

    setattr(scaled_dot_product_attention, _HOOK_MARK, True)
    return scaled_dot_product_attention


def install_decode_hook() -> None:
    """Replace the global Qwen3_5Attention.__call__ calls after its projections; the
    module has no other seam there. Idempotent. enable() installs it on the first
    activation only, so a process that never activates the tile runs mlx-vlm's
    function untouched, and the wrapper passes every call through while the flag
    is clear, so an unload leaves it inert."""
    language: Any = importlib.import_module(_DECODE_MODULE)
    current = language.scaled_dot_product_attention
    if not getattr(current, _HOOK_MARK, False):
        language.scaled_dot_product_attention = _decode_wrapper(current)


def serve_verify(queries: Any, keys: Any, values: Any, splits: int) -> Any:
    """verify() for the exact verifier's hook, counted by T, by the rows each path
    took and by the calls cut into more than one run."""
    t = queries.shape[2]
    runs = verify_plan(keys.shape[2] - t, t, splits)
    calls["verify_calls"] += 1
    calls[f"verify_t{t}" if t <= MAX_RUN else "verify_t9_up"] += 1
    for j, k, chunk in runs:
        if chunk:
            calls["verify_rows_tile"] += k - j
            calls["verify_launches"] += 1
        else:
            calls["verify_rows_stock"] += k - j
    if len(runs) > 1:
        calls["verify_straddles"] += 1
    return verify(queries, keys, values, splits)


# The dense Qwen3.5-family text model. The families that reuse its language
# module, qwen3_5_moe among them, would reach the same hooked global, so they
# are refused by model type, not by class.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
# DFlash drafts with mx.fast.scaled_dot_product_attention of its own, so its
# proposals never reach the hooked global. mlx-vlm's qwen3_5 MTP drafter builds
# its layers from the target's attention class over plain KVCaches: the decode
# hook would serve its proposals, and that path is unmeasured.
SUPPORTED_DRAFTER_KINDS = frozenset({"dflash"})
# Home of Qwen3_5Attention and of the global it calls after its projections.
_DECODE_MODULE = "mlx_vlm.models.qwen3_5.language"
# The one-row forward's path to that global with mask None, which decode parity
# rests on: sha256 of inspect.getsource, as verifyattn pins its own sources.
VALIDATED_DECODE_SOURCES: dict[str, frozenset[str]] = {
    "Qwen3_5Attention.__call__": frozenset(
        {"b051c66057acc90f4ed78691f90254685cfdb20d5814b0d58beec4d2bb34c184"}
    ),
    "Qwen3_5Model.__call__": frozenset(
        {"9dfc2048db0d818595bcdeb306a59e80cf1217331784fe7361c92a2dbf13ab65"}
    ),
    "LanguageModel.__call__": frozenset(
        {"af7f3f6427c2e59ab5266798b7518eaf026c3f7f749dd0962e38a6acece49cc9"}
    ),
}


# The probe's far parity mark is the first step boundary at or past this many
# keys: every row runs its full S splits there (from 11,553 keys at S = 20),
# and subagent turns sit at 44-77K keys.
PROBE_FAR_KEYS = 57_000
# How far the probe lets the tile land from stock one-row SDPA, as a relative
# RMS difference. The tile's error against a float64 reference is at or below
# stock's, and one bf16 ulp of the output is about 4e-3.
REL_RMS_BOUND = 1e-2


def far_mark(splits: int) -> int:
    """The first step boundary at or past PROBE_FAR_KEYS."""
    reach = PROBE_FAR_KEYS + GRAN * splits  # no step is wider than GRAN * splits keys
    return next(b for b in boundaries(reach, splits) if b >= PROBE_FAR_KEYS)


def parity_cases(splits: int, arch: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(tile marks, stock cases). Each tile mark (N0, the next two step
    boundaries and the far mark) is probed at T = 1..8 with rows below it,
    straddling it and from it. The stock cases put T = 1 and 2 rows on both
    sides of every verifyattn plan transition below N0, where verify runs
    grouped_attention and must still equal stock one-row SDPA."""
    from sous.engine.verifyattn import plan_transitions

    first = boundaries(N0 + 2 * GRAN * splits, splits)
    marks = sorted({first[0], first[1], first[2], far_mark(splits)})
    stock = sorted(
        {
            (p, t)
            for m in plan_transitions(arch, GQA)
            if m < N0
            for t in (1, 2)
            for p in (m - t - 1, m - 1 - t // 2, m - 1)
        }
    )
    return marks, stock


def _rel_rms(got: Any, want: Any) -> float:
    import mlx.core as mx

    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    return float(mx.sqrt(mx.mean((g - w) ** 2)) / mx.sqrt(mx.mean(w**2)))


def _close(got: Any, want: Any) -> bool:
    """Finite and within REL_RMS_BOUND of `want`: a NaN anywhere fails, since
    no comparison with NaN is true."""
    import mlx.core as mx

    return bool(mx.all(mx.isfinite(got)).item()) and _rel_rms(got, want) <= REL_RMS_BOUND


def probe(modules: list[Any], splits: int, arch: str) -> str | None:
    """None when the tile may serve `modules` on this GPU at `splits` splits,
    else why not. Runs on the loading thread with the flag clear, and leaves
    the flag clear and the counters and mlx-vlm's global as it found them.
    Its arrays are gone before the cache is cleared, so the budget measured
    next reads true."""
    import mlx.core as mx

    global _active_splits
    language = importlib.import_module(_DECODE_MODULE)
    installed = language.scaled_dot_product_attention
    counted = dict(calls)
    try:
        return _probe_steps(modules, splits, arch, language)
    except BaseException as e:
        # The traceback holds the steps' frame, and every probe array in it,
        # until the caller has handled the error: drop them now.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        _active_splits = 0
        language.scaled_dot_product_attention = installed
        calls.update(counted)
        mx.clear_cache()


def _probe_steps(modules: list[Any], splits: int, arch: str, language: Any) -> str | None:
    """Every input is made with GPU ops (no float64, which is CPU-only in mlx),
    and each step's K/V inputs are evaluated before that step's launches: a
    kernel failure with a CPU-stream op in flight deadlocks mlx's exception
    path. Inputs are laid out as serving lays them out: transposed query rows,
    K/V slices of 256-step padded buffers."""
    import mlx.core as mx

    global _active_splits
    # Every row variant compiles now, so no compile first happens mid-turn.
    failed = _compile_probe()
    if failed is not None:
        return f"kernel: {failed}"

    marks, stock_cases = parity_cases(splits, arch)
    far = far_mark(splits)
    first = boundaries(N0 + 2 * GRAN * splits, splits)
    mid_step = (first[1] + first[2]) // 2
    top = max(far + 16, max((p + t for p, t in stock_cases), default=0))
    cap = -(-(top + 1) // 256) * 256
    k_key, v_key, q_key, x_key, m_key = mx.random.split(mx.random.key(0), 5)
    kb = mx.random.normal((1, HKV, cap, D), key=k_key).astype(mx.bfloat16)
    vb = mx.random.normal((1, HKV, cap, D), key=v_key).astype(mx.bfloat16)
    qpool = mx.random.normal((1, 20, HQ, D), key=q_key).astype(mx.bfloat16)
    nan_tail = mx.full((1, HKV, 256, D), float("nan"), dtype=mx.bfloat16)
    mx.eval(kb, vb, qpool, nan_tail)

    def rows(start: int, t: int) -> Any:
        return qpool[:, start : start + t].transpose(0, 2, 1, 3)

    # Parity: every verify row equals decode() at its own key count.
    def parity(point: int, cases: list[tuple[int, int]]) -> str | None:
        base = point - 10  # query row i sits at position base + i and sees base + i + 1 keys
        dec = {
            pos + 1: decode(rows(pos - base, 1), kb[:, :, : pos + 1], vb[:, :, : pos + 1], splits)
            for pos in range(base, base + 20)
        }
        for prefix, t in cases:
            n = prefix + t
            out = verify(rows(prefix - base, t), kb[:, :, :n], vb[:, :, :n], splits)
            same = mx.stack(
                [mx.array_equal(out[:, :, r], dec[prefix + r + 1][:, :, 0]) for r in range(t)]
            )
            if not mx.all(same).item():
                return f"parity: verify rows at prefix {prefix}, T={t} differ from decode"
        return None

    for mark in marks:
        cases = [
            (p, t)
            for t in range(1, MAX_RUN + 1)
            for p in sorted({mark - t - 1, mark - 1 - t // 2, mark - 1})
        ]
        if (reason := parity(mark, cases)) is not None:
            return reason
    for prefix, t in stock_cases:
        if (reason := parity(prefix + 1, [(prefix, t)])) is not None:
            return reason

    # Reads stay in [0, n): NaN past n in every KV head, and a buffer that
    # ends exactly at n, give the clean result.
    for n, t in ((N0, 1), (mid_step, 3), (far, MAX_RUN)):
        want = verify(rows(0, t), kb[:, :, :n], vb[:, :, :n], splits)
        k_nan = mx.concatenate([kb[:, :, :n], nan_tail], axis=2)
        v_nan = mx.concatenate([vb[:, :, :n], nan_tail], axis=2)
        k_end, v_end = mx.contiguous(kb[:, :, :n]), mx.contiguous(vb[:, :, :n])
        mx.eval(want, k_nan, v_nan, k_end, v_end)
        for keys, values in ((k_nan[:, :, :n], v_nan[:, :, :n]), (k_end, v_end)):
            got = verify(rows(0, t), keys, values, splits)
            if not (mx.array_equal(got, want).item() and mx.all(mx.isfinite(got)).item()):
                return f"reads past n: T={t} at n={n} differs from the clean buffer"
    # A for-loop leaves its last iteration's names bound: at the far mark that
    # is about 470 MB of K/V copies, which would sit beside the correctness
    # section's own and raise the load's transient peak for nothing.
    del want, k_nan, v_nan, k_end, v_end, keys, values, got

    # Correctness, which parity cannot see in a kernel wrong the same way on
    # both paths: x8 queries, a dominant key at n - 1, and a second one at n,
    # just past the live range, that the tile must not read.
    q8 = (qpool[:, :1].astype(mx.float32) * 8).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    dominant = q8[:, :, 0].reshape(1, HKV, GQA, D)[:, :, :1]
    ones = mx.ones((1, HKV, 1, D), dtype=mx.bfloat16)
    for n in sorted({N0, mid_step, far}):
        kp = mx.concatenate([kb[:, :, : n - 1], dominant, dominant, kb[:, :, n + 1 :]], axis=2)
        vp = mx.concatenate([vb[:, :, : n - 1], ones, -ones, vb[:, :, n + 1 :]], axis=2)
        mx.eval(kp, vp)
        got = decode(q8, kp[:, :, :n], vp[:, :, :n], splits)
        want = mx.fast.scaled_dot_product_attention(
            q8, kp[:, :, :n], vp[:, :, :n], scale=SCALE, mask=None
        )
        if not _close(got, want):
            rms = _rel_rms(got, want)
            return f"correctness: n={n} is {rms:.3g} from stock (bound {REL_RMS_BOUND})"

    # The loaded model's own module, with its projections and rope, through
    # mlx-vlm's global, over a KVCache as a fork restore leaves it: keys,
    # values and offset assigned, the padded buffer NaN past the live keys.
    module = modules[0]
    n = mid_step
    x = mx.random.normal((1, 1, module.o_proj.weight.shape[0]), key=x_key).astype(mx.bfloat16)
    capacity = -(-n // 256) * 256
    kv_cache = importlib.import_module("mlx_vlm.models.cache").KVCache

    def restored() -> Any:
        cache = kv_cache()
        k_rows, v_rows = mx.random.split(m_key)
        tail = nan_tail[:, :, : capacity - n + 1]
        live = (1, HKV, n - 1, D)
        cache.keys = mx.concatenate(
            [mx.random.normal(live, key=k_rows).astype(mx.bfloat16), tail], axis=2
        )
        cache.values = mx.concatenate(
            [mx.random.normal(live, key=v_rows).astype(mx.bfloat16), tail], axis=2
        )
        cache.offset = n - 1
        mx.eval(cache.keys, cache.values)
        return cache

    if not getattr(language.scaled_dot_product_attention, _HOOK_MARK, False):
        language.scaled_dot_product_attention = _decode_wrapper(
            language.scaled_dot_product_attention
        )
    _active_splits = 0
    want = module(x, mask=None, cache=restored())
    mx.eval(want)
    before = calls["decode_tile"]
    _active_splits = splits
    got = module(x, mask=None, cache=restored())
    mx.eval(got)
    _active_splits = 0
    if calls["decode_tile"] != before + 1:
        return "the model's attention module did not reach the tile"
    if not _close(got, want):
        return f"correctness: the model's module is {_rel_rms(got, want):.3g} from stock"
    return None


def _refuse(
    reason: str, probe_seconds: float | None = None, *, expected: bool = False
) -> dict[str, Any]:
    """This load cannot use the tile: report why. An `expected` refusal says
    only that the tile does not apply here (the Mac, the model, the drafter's
    kind, the split target), which is how most Macs load a default-on tile:
    one INFO line, and the reason in the status. Any other refusal means the
    tile should have run and did not (a kernel or probe failure, a pinned
    dependency that drifted, the exact verifier missing beside DFlash):
    one warning, since a switch that is silently inert there hides a fault."""
    # A compiler error runs to dozens of lines; the status needs the one naming it.
    reason = _error_line(reason)
    if expected:
        logger.info("attention tile unavailable: %s", reason)
    else:
        warnings.warn(
            f"sous: attention tile unavailable ({reason}); "
            "decode and verify run the stock attention paths",
            stacklevel=3,
        )
    return {
        "state": "unavailable",
        "reason": reason,
        "splits": None,
        "probe_seconds": probe_seconds,
    }


def _model_reason(model: Any) -> str | None:
    """Why this model's attention is not the one the kernel is specialised
    for, or None."""
    import mlx.core as mx

    from sous.engine import verifyattn

    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)"
    modules = verifyattn._attention_modules(model)
    if not modules:
        return "no full-attention layers"
    for module in modules:
        q, kv, d = module.num_attention_heads, module.num_key_value_heads, module.head_dim
        if (q, kv, d) != (HQ, HKV, D):
            return f"unsupported attention shape {q}/{kv}/{d} (24/4/256 only)"
        if module.scale != SCALE:
            return f"attention scale {module.scale} is not 1/16"
        dtype = module.k_norm.weight.dtype
        if dtype != mx.bfloat16:
            return f"attention dtype {dtype} is not bfloat16"
    return None


def _drafter_reason(drafter_kind: str | None) -> str | None:
    """A drafter whose proposals could reach the hooked global, or None."""
    if drafter_kind is None or drafter_kind in SUPPORTED_DRAFTER_KINDS:
        return None
    return f"drafter kind {drafter_kind!r} (dflash only)"


def _verify_reason(drafter_kind: str | None, verify_status: Mapping[str, Any] | None) -> str | None:
    """With a drafter, the verifier must be on its exact grouped path too:
    otherwise decode runs the tile while verify runs stock SDPA, and output
    with the drafter no longer equals output without it."""
    if drafter_kind is None:
        return None
    state = (verify_status or {}).get("state")
    if state != "active":
        return (
            f"exact verify attention is {state}: "
            "the verifier would run stock attention beside the tile"
        )
    return None


def _decode_sources_reason() -> str | None:
    """Which function on the one-row forward's path changed, or None."""
    from sous.engine.verifyattn import _source_digest

    for qualname, digests in VALIDATED_DECODE_SOURCES.items():
        if _source_digest(_DECODE_MODULE, qualname) not in digests:
            return f"mlx-vlm {qualname} changed"
    return None


def _split_target(device_name: str) -> tuple[int | None, str | None]:
    """(S, None) when this GPU's core count is a measured split target, else
    (None, why). The guards prove exactness, not speed: an unmeasured core
    count may run the tile slower than stock."""
    cores, reason = gpucores.matched_cores(device_name)
    if cores is None:
        return None, reason
    if cores not in MEASURED_SPLITS:
        measured = ", ".join(str(s) for s in sorted(MEASURED_SPLITS))
        return None, f"split target {cores} not measured (measured: {measured})"
    return cores, None


def enable(
    model: Any,
    *,
    enabled: bool,
    drafter_kind: str | None,
    verify_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether this load's decode and verify attention run on the tile.
    Returns the status the engine exposes and never raises: a model load must
    not fail because an accelerator is missing."""
    global _active_splits
    # First, on every call: a flag left from an earlier load must never serve
    # this one, whatever happens below.
    _active_splits = 0
    if not enabled:
        return {"state": "off", "reason": None, "splits": None, "probe_seconds": None}
    seconds: float | None = None
    try:
        # Before anything imports mlx, so a GPU without tensor units reports
        # that whatever its model.
        reason = nax.platform_reason()
        if reason is not None:
            return _refuse(reason, expected=True)
        import mlx.core as mx

        from sous.engine import verifyattn

        reason = _model_reason(model) or _drafter_reason(drafter_kind)
        if reason is not None:
            return _refuse(reason, expected=True)
        reason = (
            _verify_reason(drafter_kind, verify_status)
            or verifyattn._gate()
            or _decode_sources_reason()
        )
        if reason is not None:
            return _refuse(reason)
        splits, reason = _split_target(str(mx.device_info().get("device_name", "")))
        if splits is None:
            return _refuse(reason or "no split target", expected=True)
        modules = verifyattn._attention_modules(model)
        start = time.perf_counter()
        try:
            reason = probe(modules, splits, _arch())
        finally:
            seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds)
        install_decode_hook()
        _active_splits = splits
        logger.info(
            "attention tile: %d layers, %d splits, probe %.2fs", len(modules), splits, seconds
        )
        return {"state": "active", "reason": None, "splits": splits, "probe_seconds": seconds}
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _active_splits = 0
        # The warning carries one line; the traceback goes to the log.
        logger.warning("attention tile: enabling failed", exc_info=True)
        return _refuse(str(e) or type(e).__name__, seconds)
