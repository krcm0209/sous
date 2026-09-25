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
from typing import Any

HEAD_DIM = 256
MIN_ROWS = 3
MAX_ROWS = 8
# Past mlx 0.32.2's last SDPA dispatch threshold (65536 keys).
_SCAN_TO = 70_000


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
    out: list[tuple[int, int]] = []
    j = 0
    while j < t:
        k = j + 1
        while k < t:
            call = plan(arch, prefix + k + 1, gqa, k + 1 - j)
            if call[0] == "fallback" or any(
                plan(arch, prefix + r + 1, gqa, 1) != call for r in range(j, k + 1)
            ):
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
            want = _row_loop(queries, k, v, None, scale)
            if not mx.array_equal(got, want).item():
                return f"grouped attention differs from the per-row loop at prefix {prefix}, T={t}"
    finally:
        del keys, values, rows
        mx.clear_cache()
    return None
