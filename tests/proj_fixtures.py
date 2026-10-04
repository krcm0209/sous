"""Shared helpers for the projection kernel's tests: synthetic quantized linears, a
plain-mlx stand-in for the Metal kernel, and the `kernel` skip marker.

mlx is imported inside each helper, so importing this module needs only what
`projkernel.kernel_available()` itself imports."""

from typing import Any

import pytest

from sous.engine import int8prefill, projkernel
from tests import tile_fixtures as tfx

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


# The bitwise tests' inputs. Random activations leave a reordered fp32 sum one bf16
# rounding away from the same bits almost everywhere; the two cancelling layouts
# make the order show. "quarters" cancels the first K simdgroup's partial against
# the last's (the order the K partials are added in), "group" cancels inside every
# quant group (the order of the group's activation sum, in-line or presummed).
LAYOUTS = ("random", "quarters", "group")
_CANCEL = 256.0
_SMALL = 1e-2


def k_quarters(k: int) -> list[tuple[int, int]]:
    """The column range each of the kernel's four K simdgroups sums, split by whole
    quant groups as the kernel splits them (g0 = sk * G // 4)."""
    g = k // 64
    return [(64 * (sk * g // 4), 64 * ((sk + 1) * g // 4)) for sk in range(4)]


def _mirrors(k: int, layout: str) -> list[tuple[int, int, int]]:
    """(source, mirror, width) column spans a cancelling layout pairs up."""
    if layout == "quarters":
        (a, b), _, _, (c, _) = k_quarters(k)
        return [(a, c, b - a)]  # the last quarter is never narrower than the first
    if layout == "group":
        return [(g, g + 48, 16) for g in range(0, k, 64)]
    raise ValueError(layout)


def cancelling_activations(rows: int, k: int, layout: str, seed: int) -> Any:
    """[rows, K] bf16 activations, +c·v over each source span and exactly -c·v over
    its mirror, small values elsewhere: the large terms cancel, so the result is
    small and a reordered fp32 sum lands on another bf16. Built on the GPU and
    evaluated."""
    import mlx.core as mx

    keys = mx.random.split(mx.random.key(seed), 2)
    x = (mx.random.normal((rows, k), key=keys[0]) * _SMALL).astype(mx.bfloat16)
    big = (mx.random.normal((rows, k), key=keys[1]) * _CANCEL).astype(mx.bfloat16)
    for src, dst, width in _mirrors(k, layout):
        x[:, src : src + width] = big[:, src : src + width]
        x[:, dst : dst + width] = -big[:, src : src + width]
    mx.eval(x)
    return x


def cancelling_linear(n: int, k: int, layout: str, seed: int) -> tuple[Any, Any, Any]:
    """rand_linear's projection with each mirror span's weight columns copied from
    its source span, so the two spans' products cancel term by term. The spans are
    whole quant groups or the same columns of one group, so both quantize to the
    same codes."""
    import mlx.core as mx

    dense = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    for src, dst, width in _mirrors(k, layout):
        dense[:, dst : dst + width] = dense[:, src : src + width]
    w, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    mx.eval(w, scales, biases)
    return w, scales, biases


def projection_inputs(
    layout: str, rows: int, k: int, ns: tuple[int, ...], *, x_seed: int, w_seed: int
) -> tuple[Any, list[tuple[Any, Any, Any]]]:
    """(activations, one linear per N in `ns`, linear i seeded `w_seed + i`) in one
    of LAYOUTS; "random" is activations() and rand_linear()."""
    if layout == "random":
        x = activations(rows, k, x_seed)
        return x, [rand_linear(n, k, w_seed + i) for i, n in enumerate(ns)]
    weights = [cancelling_linear(n, k, layout, w_seed + i) for i, n in enumerate(ns)]
    return cancelling_activations(rows, k, layout, x_seed), weights


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


def rand_module(n: int, k: int, seed: int) -> Any:
    """rand_linear's projection as the module a model holds: an nn.QuantizedLinear,
    affine 4-bit at group size 64, bf16 scales and biases, no bias term."""
    import mlx.nn as nn

    module = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    module.weight, module.scales, module.biases = rand_linear(n, k, seed)
    return module


def tiny_quantized_model(*, seed: int = 0) -> Any:
    """A random qwen3_5 language model quantized as the 27B is (affine 4-bit,
    group 64, bf16 scales, no bias, lm_head untied). Linear-attention layers 0
    and 2, full-attention layers 1 and 3, and 31 quantized projections, every
    one with K >= 256 so that each of the plain variant's four K splits gets a
    whole group. Its K/N shapes give nine dispatch keys."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    args = TextConfig(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=tfx.VOCAB,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        head_dim=256,
        full_attention_interval=2,
    )
    mx.random.seed(seed)
    lm = LanguageModel(args)
    lm.set_dtype(mx.bfloat16)
    nn.quantize(lm, group_size=64, bits=4)
    mx.eval(lm.parameters())
    return lm


def guard_proj_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers restores for what install_hooks(), enable() and the probe leave
    behind for the rest of the process: nn.QuantizedLinear.__call__ and the
    verifier's _linear and _linears (the hooks), the flag, int8's install flag (a
    test may install int8's wrapper in either order with ours) and the remembered
    compile verdict. A test that installs or activates the kernel then cannot
    leak any of them into the next."""
    import mlx.nn as nn

    verifier = projkernel._verifier_class()
    monkeypatch.setattr(nn.QuantizedLinear, "__call__", nn.QuantizedLinear.__call__)
    monkeypatch.setattr(verifier, "_linear", verifier._linear)
    monkeypatch.setattr(verifier, "_linears", verifier._linears)
    monkeypatch.setattr(projkernel, "_active", projkernel._active)
    monkeypatch.setattr(
        int8prefill, "_QUANTIZED_LINEAR_WRAPPED", int8prefill._QUANTIZED_LINEAR_WRAPPED
    )
    monkeypatch.setattr(projkernel, "_kernel_ok", projkernel._kernel_ok)


def hooked(monkeypatch: pytest.MonkeyPatch, *, active: int = 1) -> None:
    """Both hooks installed and the flag at `active` for this test only, with the
    stand-in as the kernel; guard_proj_state's restores undo all of it."""
    guard_proj_state(monkeypatch)
    projkernel.install_hooks()
    monkeypatch.setattr(projkernel, "_active", active)
    monkeypatch.setattr(projkernel, "mma", stand_in_mma)


# The attention tile's status as an active load reports it; enable() rides on it.
ACTIVE_TILE: dict[str, Any] = {
    "state": "active",
    "reason": None,
    "splits": 20,
    "probe_seconds": 0.5,
}


def proj_ready(monkeypatch: pytest.MonkeyPatch, *, real_pins: bool = False) -> dict[str, Any]:
    """Every enable() and static_reason() guard that reads the machine passes
    on any GPU: the platform rule, and a core count of 20 under this GPU's own
    name. The stand-in is the kernel, and the pins pass unless `real_pins` (for
    the tests about the pins themselves). Registers guard_proj_state's restores
    and returns the attention tile's status to hand enable()."""
    import mlx.core as mx

    from sous.engine import gpucores, nax

    guard_proj_state(monkeypatch)
    device = str(mx.device_info()["device_name"])
    monkeypatch.setattr(nax, "platform_reason", lambda: None)
    monkeypatch.setattr(gpucores, "read", lambda: gpucores.GPUCores(20, device, None, "iokit"))
    monkeypatch.setattr(projkernel, "mma", stand_in_mma)
    if not real_pins:
        monkeypatch.setattr(projkernel, "_pins_reason", lambda: None)
    return dict(ACTIVE_TILE)
