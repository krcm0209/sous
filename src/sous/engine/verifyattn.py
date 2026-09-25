"""Exact grouped attention for mlx-vlm's speculative verifier (qwen3_5).

The exact verifier (``Qwen3_5BatchInvariantForward._attention``) runs a T > 2
verify forward's attention as T single-query SDPA calls, so the whole KV prefix
is streamed once per row. Here rows are grouped into one stock
``mx.fast.scaled_dot_product_attention`` call wherever mlx gives the group the
kernel plan each row would get alone. Inside one plan, mlx assigns key i to
simdgroup ``i % 32`` (one pass) or block ``i % blocks`` (two passes) whatever
the key count, and the causal mask skips excluded keys. So every row stays
bit-identical to the loop and to the M=1 decode.

``plan()`` mirrors mlx 0.32.2's dispatch (backend/metal/
scaled_dot_product_attention.cpp), which sous cannot inspect at runtime. So four
guards stand between it and a served turn:
- the mlx version;
- the source of every mlx-vlm function the hook reads;
- a load-time probe on this GPU;
- a per-call scope check, decided before the projections run. The projections
  append the verify rows to the KV cache, so the original method cannot be
  re-entered after them.

T = 2 is left alone. The stock verifier already makes one call there, and
grouping it would change output where rows straddle a plan transition.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import logging
import os
import time
import warnings
from typing import Any

from sous.engine.int8prefill import _model_type, _root

logger = logging.getLogger("sous.engine.verifyattn")

HEAD_DIM = 256
MIN_ROWS = 3
MAX_ROWS = 8
# Past mlx 0.32.2's last SDPA dispatch threshold (65536 keys).
_SCAN_TO = 70_000
# mlx releases whose SDPA dispatch plan() has been read against; re-read
# backend/metal/scaled_dot_product_attention.cpp and extend this on a bump.
VALIDATED_MLX = frozenset({"0.32.2"})
# The MoE variant reuses the same verifier class with other attention shapes.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})
# Every mlx-vlm function the hook calls or re-implements around, by the module
# that defines it. The daemon's tool environment can resolve a newer mlx-vlm
# than the lock, so these are checked at load, not only in CI.
_SOURCES = {
    "Qwen3_5BatchInvariantForward._attention": "mlx_vlm.models.qwen3_5.speculative_verifier",
    "Qwen3_5Attention._prepare_projected_qkv": "mlx_vlm.models.qwen3_5.language",
    "_create_qwen3_5_attention_mask": "mlx_vlm.models.qwen3_5.language",
    "_qwen3_5_left_padded_attention": "mlx_vlm.models.qwen3_5.language",
    "_qwen3_5_left_padding_info": "mlx_vlm.models.qwen3_5.language",
    "KVCache.update_and_fetch": "mlx_vlm.models.cache",
    "KVCache.make_mask": "mlx_vlm.models.cache",
    # The stock M=1 decode call and _row_loop's reference both go through
    # these three, not straight through mx.fast — the source gate has to
    # cover what actually decides the stock output, not just the T > 2 loop.
    "scaled_dot_product_attention": "mlx_vlm.models.base",
    "slice_kv_sequence": "mlx_vlm.models.base",
    "kv_sequence_length": "mlx_vlm.models.base",
}
# sha256 of inspect.getsource() as validated (identical in mlx-vlm 0.7.1 and
# 0.7.2). Add a hash only after re-reading that function against the hook.
VALIDATED_MLX_VLM_SOURCES: dict[str, frozenset[str]] = {
    "Qwen3_5BatchInvariantForward._attention": frozenset(
        {"7790ae37e7a0d78d2287217a3050e03a4b9b7c10eb1e60839e4315226186cdf1"}
    ),
    "Qwen3_5Attention._prepare_projected_qkv": frozenset(
        {"da21fd76217c72aa7ef15d4097e1fe428ff0022259f9556eb8fd95b59217c109"}
    ),
    "_create_qwen3_5_attention_mask": frozenset(
        {"a74a1135eb644698e1beb6e43a5d51e27e6490d0b0f4f6d951d08dd5e908c681"}
    ),
    "_qwen3_5_left_padded_attention": frozenset(
        {"e5d9f69065da9885674a6c3e11a65807948c5e9f9d7a1fef6211628328e7b312"}
    ),
    "_qwen3_5_left_padding_info": frozenset(
        {"1cea363e3150f9262f7fb23d11d525b6be32625d3407cd324850d82d6a151fd5"}
    ),
    "KVCache.update_and_fetch": frozenset(
        {"9cf2722ab3a6f71dba0e66bfb4f290b9611c740fa7958e84f61a616056d3df5b"}
    ),
    "KVCache.make_mask": frozenset(
        {"d65ba0d83b52a93bb00faae6bcbac8add067724ebe9dd8ba6aaa716f81b6b620"}
    ),
    "scaled_dot_product_attention": frozenset(
        {"9b12ea1ee54be83003091a88dd8657b1448e1a68681dd791e69fbe971cd27cb5"}
    ),
    "slice_kv_sequence": frozenset(
        {"5d6042d27927663a2813587889f7617727704d2e383cac1f986fc905c3acc74a"}
    ),
    "kv_sequence_length": frozenset(
        {"1f7c089333afa6730f5ea298b326f456294623d6360a145c7d15980d88275040"}
    ),
}
# Set with object.__setattr__ so it stays out of mlx's parameter tree; the value
# is the GQA ratio the probe proved, 0 when untagged.
_TAG = "_sous_verify_attention"
_ARCH = ""
_KVCACHE: type | None = None
_WRAPPED: set[type] = set()
# Per-path call counts; the tests read them to prove which path ran.
calls = {"grouped": 0, "loop": 0, "original": 0}


def plan(arch: str, n_keys: int, gqa: int, q_len: int) -> tuple[str, int]:
    """The kernel mlx 0.32.2 picks for one head-dim-256 SDPA call:
    ("fallback", 0), ("1pass", 0) or ("2pass", blocks). Mirrors
    has_fused_kernel/use_fallback and sdpa_vector_2pass's block table."""
    if q_len > 8 or q_len > n_keys or q_len * gqa > 32:
        return ("fallback", 0)
    suffix = arch[-1:]
    if not ((suffix in ("d", "s") and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096)):
        return ("1pass", 0)
    n_simds = gqa * q_len
    if suffix == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
    elif suffix == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return ("2pass", blocks)


def groups(arch: str, prefix: int, t: int, gqa: int) -> list[tuple[int, int]]:
    """Contiguous row runs [j, k) whose one causal call gets every member row's
    singleton plan; a single row is always exact on its own."""
    single = [plan(arch, prefix + r + 1, gqa, 1) for r in range(t)]
    out: list[tuple[int, int]] = []
    j = 0
    while j < t:
        k = j + 1
        while k < t:
            call = plan(arch, prefix + k + 1, gqa, k + 1 - j)
            if call[0] == "fallback" or any(s != call for s in single[j : k + 1]):
                break
            k += 1
        out.append((j, k))
        j = k
    return out


@functools.cache
def plan_transitions(arch: str, gqa: int) -> tuple[int, ...]:
    """Every key count N at which some verify group's plan differs from N - 1's.
    The scan starts past MAX_ROWS keys: below that the q_len > n_keys fallback
    moves with N, and no verify reaches it."""
    qs = range(1, min(MAX_ROWS, 32 // gqa) + 1)
    prev = [plan(arch, MAX_ROWS + 1, gqa, q) for q in qs]
    out: list[int] = []
    for n in range(MAX_ROWS + 2, _SCAN_TO):
        cur = [plan(arch, n, gqa, q) for q in qs]
        if cur != prev:
            out.append(n)
        prev = cur
    return tuple(out)


@functools.cache
def probe_cases(arch: str, gqa: int) -> tuple[tuple[int, int], ...]:
    """(prefix, T) pairs that put verify rows on both sides of every plan
    transition, plus one interior prefix between neighbours. Past the last
    transition nothing changes, so nothing beyond it needs proving."""
    marks = plan_transitions(arch, gqa)
    cases: set[tuple[int, int]] = set()
    for n in marks:
        for t in range(MIN_ROWS, MAX_ROWS + 1):
            cases.update((p, t) for p in (n - t, n - t // 2 - 1, n - 1))
    edges = (0, *marks) if marks else (0, 1024)
    for lo, hi in zip(edges, edges[1:], strict=False):
        cases.update(((lo + hi) // 2, t) for t in range(MIN_ROWS, MAX_ROWS + 1))
    return tuple(sorted(cases))


def grouped_attention(
    queries: Any, keys: Any, values: Any, scale: float, prefix: int, arch: str
) -> Any:
    """One stock SDPA call per group of rows that share their singleton plan:
    causal for a group, unmasked for a lone row, which is the M=1 decode call."""
    import mlx.core as mx

    t = queries.shape[2]
    gqa = queries.shape[1] // keys.shape[1]
    parts = [
        mx.fast.scaled_dot_product_attention(
            queries[:, :, j:k, :],
            keys[:, :, : prefix + k, :],
            values[:, :, : prefix + k, :],
            scale=scale,
            mask="causal" if k - j > 1 else None,
        )
        for j, k in groups(arch, prefix, t, gqa)
    ]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def _row_loop(queries: Any, keys: Any, values: Any, cache: Any, scale: float) -> Any:
    """mlx-vlm's own T > 2 loop: one single-query call per row over its own
    prefix, through mlx-vlm's helpers so quantized KV tuples still work."""
    import mlx.core as mx
    from mlx_vlm.models.base import (
        kv_sequence_length,
        scaled_dot_product_attention,
        slice_kv_sequence,
    )

    t = queries.shape[2]
    prefix = kv_sequence_length(keys) - t
    return mx.concatenate(
        [
            scaled_dot_product_attention(
                queries[:, :, i : i + 1, :],
                slice_kv_sequence(keys, prefix + i + 1),
                slice_kv_sequence(values, prefix + i + 1),
                cache=cache,
                scale=scale,
                mask=None,
            )
            for i in range(t)
        ],
        axis=2,
    )


def probe(arch: str, q_heads: int, kv_heads: int, dtype: Any) -> str | None:
    """Prove grouped_attention bit-identical to the per-row loop on this GPU at
    every plan transition. Returns why not, or None. Inputs are laid out as
    serving lays them out: queries are a transposed [1, T, Hq, D] slice, K/V
    slices of a buffer grown in KVCache's 256-token steps."""
    import mlx.core as mx

    # Serving always hands _row_loop a real KVCache, and mlx-vlm's own
    # scaled_dot_product_attention branches on isinstance/hasattr checks
    # against it — a bare None takes the same branch today, but only because
    # nothing there tests "cache is None". Use the real type so the probe
    # keeps proving what serving does, not a lookalike.
    cache = importlib.import_module("mlx_vlm.models.cache").KVCache()
    cases = probe_cases(arch, q_heads // kv_heads)
    capacity = -(-max(p + t for p, t in cases) // 256) * 256
    k_key, v_key, q_key = mx.random.split(mx.random.key(0), 3)
    keys = mx.random.normal((1, kv_heads, capacity, HEAD_DIM), key=k_key).astype(dtype)
    values = mx.random.normal((1, kv_heads, capacity, HEAD_DIM), key=v_key).astype(dtype)
    rows = mx.random.normal((1, MAX_ROWS, q_heads, HEAD_DIM), key=q_key).astype(dtype)
    scale = HEAD_DIM**-0.5
    try:
        for prefix, t in cases:
            queries = rows[:, :t].transpose(0, 2, 1, 3)
            k = keys[..., : prefix + t, :]
            v = values[..., : prefix + t, :]
            got = grouped_attention(queries, k, v, scale, prefix, arch)
            want = _row_loop(queries, k, v, cache, scale)
            equal = mx.array_equal(got, want).item()
            # Drop this case's slices and results now: a for-loop leaves its
            # last iteration's names bound after it ends, so without this the
            # final got/want/queries/k/v would still reference the pool's
            # buffers when clear_cache() runs below and free nothing.
            del queries, k, v, got, want
            if not equal:
                return f"grouped attention differs from the per-row loop at prefix {prefix}, T={t}"
    finally:
        del keys, values, rows, cache
        mx.clear_cache()
    return None


def _source_digest(module: str, qualname: str) -> str | None:
    """sha256 of an mlx-vlm function's source, or None when it cannot be read."""
    try:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()
    except ImportError, AttributeError, OSError, TypeError:
        return None


def _gate() -> str | None:
    """Why plan() or the hook cannot be trusted in this process, or None."""
    import mlx.core as mx

    version = mx.__version__  # ty: ignore[unresolved-attribute]
    if version not in VALIDATED_MLX:
        return f"mlx {version} not validated"
    # mlx honours this override of its two-pass block count; plan() does not.
    if os.environ.get("MLX_SDPA_BLOCKS", "0") not in ("", "0"):
        return "MLX_SDPA_BLOCKS is set"
    for qualname, module in _SOURCES.items():
        if _source_digest(module, qualname) not in VALIDATED_MLX_VLM_SOURCES[qualname]:
            return f"mlx-vlm {qualname} changed"
    return None


def _in_scope(verifier: Any, attention: Any, x: Any, mask: Any, cache: Any) -> bool:
    """Whether this call takes the grouped path. Decided before any projection:
    _prepare_projected_qkv appends the verify rows to the KV cache, so after it
    the original method could only append them a second time."""
    gqa = getattr(attention, _TAG, 0)
    if not gqa or _KVCACHE is None:
        return False
    if x.ndim != 3 or x.shape[0] != 1 or not (MIN_ROWS <= x.shape[1] <= MAX_ROWS):
        return False
    if not (isinstance(mask, str) and mask == "causal"):
        return False
    # The exact type: mlx-vlm's KVCache subclasses keep other layouts.
    if type(cache) is not _KVCACHE or hasattr(cache, "bits"):
        return False
    if getattr(cache, "_qwen3_5_decode_left_padding", None) is not None:
        return False
    info = verifier._helpers()._qwen3_5_left_padding_info(cache)
    if info is not None and info[1] > 0:
        return False
    heads = getattr(attention, "num_attention_heads", 0)
    kv_heads = getattr(attention, "num_key_value_heads", 0)
    return (
        getattr(attention, "head_dim", None) == HEAD_DIM
        and kv_heads > 0
        and heads == gqa * kv_heads
    )


def _post_ok(queries: Any, keys: Any, values: Any, length: int) -> bool:
    """What the grouped call assumes of the projected tensors."""
    import mlx.core as mx

    return (
        isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.dtype == keys.dtype == values.dtype
        and queries.shape[2] == length
        and keys.shape[-2] == values.shape[-2] >= length
        and queries.shape[1] % keys.shape[1] == 0
    )


def _wrap(cls: Any) -> None:
    import mlx.core as mx

    orig = cls._attention

    # functools.wraps: the source gate follows __wrapped__ back to mlx-vlm's body.
    @functools.wraps(orig)
    def _attention(self, attention, x, mask, cache, position_ids, position_embeddings):
        if not _in_scope(self, attention, x, mask, cache):
            if getattr(attention, _TAG, 0):
                calls["original"] += 1
            return orig(self, attention, x, mask, cache, position_ids, position_embeddings)
        batch, length, _ = x.shape
        q_proj_output, keys, values = self._linears(
            (attention.q_proj, attention.k_proj, attention.v_proj), x
        )
        queries, keys, values, gate, mask = attention._prepare_projected_qkv(
            q_proj_output, keys, values, cache, position_ids, position_embeddings, mask
        )
        output = self._helpers()._qwen3_5_left_padded_attention(
            queries, keys, values, cache=cache, scale=attention.scale, mask=mask
        )
        if output is None:
            if _post_ok(queries, keys, values, length):
                calls["grouped"] += 1
                prefix = keys.shape[-2] - length
                output = grouped_attention(queries, keys, values, attention.scale, prefix, _ARCH)
            else:
                calls["loop"] += 1
                output = _row_loop(queries, keys, values, cache, attention.scale)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self._linear(attention.o_proj, output * mx.sigmoid(gate))

    cls._attention = _attention


def install_wrapper() -> None:
    """Wrap the exact verifier's attention once per process. The wrapper acts only
    on tagged modules, so installing it changes nothing for any other model."""
    cls = importlib.import_module(
        "mlx_vlm.models.qwen3_5.speculative_verifier"
    ).Qwen3_5BatchInvariantForward
    if cls not in _WRAPPED:
        _wrap(cls)
        _WRAPPED.add(cls)


def _attention_modules(model: Any) -> list[Any]:
    """The loaded target's full-attention modules; linear-attention layers have none."""
    layers = getattr(getattr(_root(model), "model", None), "layers", None) or []
    return [
        layer.self_attn
        for layer in layers
        if not getattr(layer, "is_linear", True) and hasattr(layer, "self_attn")
    ]


def _untag(model: Any) -> None:
    for module in _attention_modules(model):
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, 0)


def _refuse(reason: str, probe_seconds: float | None = None) -> dict[str, Any]:
    """The verifier keeps mlx-vlm's per-row attention: say so once, and why."""
    warnings.warn(
        f"sous: exact grouped verify attention unavailable ({reason}); "
        "verifying with mlx-vlm's per-row attention",
        stacklevel=3,
    )
    return {"state": "unavailable", "reason": reason, "probe_seconds": probe_seconds}


def enable(model: Any, *, enabled: bool) -> dict[str, Any]:
    """Opt one loaded model's verifier in. Returns the status the engine exposes;
    never raises — a model load must not fail because of an optimisation."""
    global _ARCH, _KVCACHE
    if not enabled:
        return {"state": "off", "reason": None, "probe_seconds": None}
    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return _refuse(f"unsupported model type {model_type or 'unknown'!r} (qwen3_5 only)")
    seconds: float | None = None
    try:
        import mlx.core as mx

        reason = _gate()
        if reason is not None:
            return _refuse(reason)
        modules = _attention_modules(model)
        shapes = {(m.num_attention_heads, m.num_key_value_heads, m.head_dim) for m in modules}
        if len(shapes) != 1:
            return _refuse(f"expected one attention shape, found {sorted(shapes)}")
        q_heads, kv_heads, head_dim = shapes.pop()
        if head_dim != HEAD_DIM or kv_heads <= 0 or q_heads % kv_heads:
            return _refuse(f"unsupported attention shape {q_heads}/{kv_heads}/{head_dim}")
        arch = str(mx.device_info().get("architecture", ""))
        start = time.perf_counter()
        reason = probe(arch, q_heads, kv_heads, modules[0].k_norm.weight.dtype)
        seconds = round(time.perf_counter() - start, 2)
        if reason is not None:
            return _refuse(reason, seconds)
        _ARCH = arch
        _KVCACHE = importlib.import_module("mlx_vlm.models.cache").KVCache
        install_wrapper()
        for module in modules:
            object.__setattr__(module, _TAG, q_heads // kv_heads)
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _untag(model)
        return _refuse(str(e) or type(e).__name__, seconds)
    logger.info("exact verify attention: %d layers, probe %.2fs", len(modules), seconds)
    return {"state": "active", "reason": None, "probe_seconds": seconds}
