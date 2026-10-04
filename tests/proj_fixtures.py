"""Shared helpers for the projection kernel's tests: synthetic quantized linears, a
plain-mlx stand-in for the Metal kernel, and the `kernel` skip marker.

mlx is imported inside each helper, so importing this module needs only what
`projkernel.kernel_available()` itself imports."""

from typing import Any

import pytest

from sous.engine import projkernel

# Read once: a failed probe is not remembered, so each call would compile again.
KERNEL = projkernel.kernel_available()
kernel = pytest.mark.skipif(
    not KERNEL.available, reason=f"projection kernel unavailable: {KERNEL.reason}"
)


def rand_linear(n: int, k: int, seed: int) -> tuple[Any, Any, Any]:
    """An affine-Q4 gs64 projection as mlx stores it, (weight uint32 [N, K/8],
    scales bf16 [N, K/64], biases bf16 [N, K/64]): a bf16 normal * 0.02 quantized
    on the GPU and evaluated."""
    import mlx.core as mx

    dense = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    w, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    mx.eval(w, scales, biases)
    return w, scales, biases


def activations(rows: int, k: int, seed: int) -> Any:
    """[rows, K] bf16 activations, built on the GPU and evaluated."""
    import mlx.core as mx

    x = (mx.random.normal((rows, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    return x


def stand_in_mma(x: Any, weights: list[tuple[Any, Any, Any]]) -> tuple[Any, ...]:
    """A plain-mlx model of projkernel.mma's contract: every linear dequantized in
    fp32 and applied to each row on its own (the same one-row matmul whatever the
    row count), rounded to bf16. A row's output depends on its own activations and
    its own linear and nothing else, the property the routing tests rely on. It
    refuses what the kernel refuses, so a hook that forgets to split a long verify
    call fails here too."""
    import mlx.core as mx

    k = x.shape[-1]
    n_rows = x.size // k
    if x.dtype != mx.bfloat16 or not 1 <= n_rows <= projkernel.MAX_ROWS:
        raise ValueError(f"stand-in takes 1..{projkernel.MAX_ROWS} bf16 rows, got {x.shape}")
    rows = x.reshape(n_rows, k).astype(mx.float32)
    outs = []
    for w, scales, biases in weights:
        dense = mx.dequantize(
            w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=64, bits=4
        )
        y = mx.concatenate([rows[r : r + 1] @ dense.T for r in range(n_rows)], axis=0)
        outs.append(y.astype(mx.bfloat16).reshape(*x.shape[:-1], dense.shape[0]))
    return tuple(outs)
