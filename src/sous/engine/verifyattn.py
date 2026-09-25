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
