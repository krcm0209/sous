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

import logging

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
