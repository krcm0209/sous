"""INT8-activation prefill for affine-Q4/gs64 projections on the M5 tensor units.

Prefill GEMMs (>= MIN_ROWS rows) of tagged ``nn.QuantizedLinear`` projections run as
int8 activations x the checkpoint's own packed Q4 weights -> int32 on the neural
accelerators, with the affine correction applied per 64-code group. The two Metal
kernels live as ``.metal`` files in ``kernels/`` and are compiled by mlx at model load
(``mx.fast.metal_kernel``): no native build, no ABI coupling to an mlx version.

Derived from oMLX (jundot/omlx#3548, Apache License 2.0): qwen35_oq_a8.metal,
qwen35_oq_a8_nax.metal and oq_a8_decode.h. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import functools
import logging
import platform
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, cast

logger = logging.getLogger("sous.engine.int8prefill")

GROUP = 64
# Below this many rows the Stage-A pass costs more than the faster GEMM saves
# (crossover measured between 4 and 8 rows; 1.6x by 128). Set well above the
# crossover so decode (1 row) and speculative verify (<= 6 rows) never route.
MIN_ROWS = 128
# The PR's best Q4 tile: BM = 2 * 16 * WM rows, BN = 32 * WN columns, 32 * WM * WN
# threads per threadgroup. The seven tile variants are within ~3% of each other.
TILE_WM = 1
TILE_WN = 2
BM = 2 * 16 * TILE_WM
BN = 32 * TILE_WN
# The dense Qwen3.5-family text model (Qwen3.6/3.8 checkpoints declare it too). The
# MoE variant reuses the same GatedDeltaNet and MLP classes for its shared expert,
# so the class wrappers would fire there as well; it is untested and refused by
# model type rather than by class.
SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5"})


@functools.cache
def _kernel_text(name: str) -> str:
    """Kernel sources live as .metal/.h files in ``kernels/`` so they are edited and
    diffed as Metal, not as Python strings; mlx still receives them as text."""
    return files("sous.engine.kernels").joinpath(name).read_text(encoding="utf-8")


_kernels: dict[str, Callable[..., list[Any]]] = {}


def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str,
) -> Callable[..., list[Any]]:
    """One metal_kernel object per name, built on first use. mlx compiles each
    (source, template) instantiation once per process behind it."""
    kernel = _kernels.get(name)
    if kernel is None:
        import mlx.core as mx

        kernel = cast(
            "Callable[..., list[Any]]",
            mx.fast.metal_kernel(
                name=name,
                input_names=input_names,
                output_names=output_names,
                source=source,
                header=header,
            ),
        )
        _kernels[name] = kernel
    return kernel


def reorder_k(qa: Any) -> Any:
    """Put each 64-code group into the GEMM's fragment K-order.

    Slot ``16c + 4t + j`` of a group holds code ``k = 16c + 8*(t>>1) + 2j + (t&1)``:
    the order in which the GEMM's nibble decode hands weights to the tensor-op
    fragment. Legal because a dot product is order-independent and the affine
    group is the same set of 64 either way, so scales, biases and Ra are untouched.
    Writing ``t = 2a + b`` the map is the transpose of the last two axes of
    ``(c, a, j, b)``; one contiguous copy of an int8 [M, K].
    """
    import mlx.core as mx

    m, k = qa.shape
    return mx.contiguous(
        qa.reshape(m, k // GROUP, 4, 2, 4, 2).transpose(0, 1, 2, 3, 5, 4).reshape(m, k)
    )


def stage_a(x: Any) -> tuple[Any, Any, Any]:
    """bf16/fp16 ``[..., K]`` -> ``(qa int8 [M, K] reordered, sa f32 [M], ra int16 [K/64, M])``.

    Ra comes back group-major so a group's row sums are contiguous across the lanes
    that need them in the GEMM.
    """
    import mlx.core as mx

    k = x.shape[-1]
    m = x.size // k
    x2 = x.reshape(m, k)
    qa, sa, ra = _kernel(
        "sous_int8_stage_a",
        ["x"],
        ["qa", "sa", "ra"],
        _kernel_text("stage_a.metal"),
        _kernel_text("common.h"),
    )(
        inputs=[x2],
        template=[("T", x.dtype), ("K", k)],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,), (m, k // GROUP)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    )
    return reorder_k(qa), sa, mx.contiguous(ra.T)


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


_ARCH = re.compile(r"applegpu_g(\d+)(\D?)$")


def availability() -> Availability:
    """Mirror of mlx's own is_nax_available() (device.cpp, 0.32.2): macOS >= 26.2
    and an Apple GPU of generation >= 17 (>= 18 for 'p'-suffix parts) — the M5
    family and later. Older GPUs run the Metal-4 tensor ops on the plain shader
    path, where int8 buys nothing, so they are `unavailable`, not merely slow."""
    if platform.system() != "Darwin":
        return Availability(False, "not macOS")
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover - the lint job has no mlx
        return Availability(False, f"mlx unavailable: {e}")
    if not mx.metal.is_available():
        return Availability(False, "Metal unavailable")
    release = platform.mac_ver()[0]
    parts = tuple(int(p) for p in release.split(".") if p.isdigit())
    if parts[:2] < (26, 2):
        return Availability(False, f"macOS {release} < 26.2")
    arch = str(mx.device_info().get("architecture", ""))
    match = _ARCH.match(arch)
    if match is None:
        return Availability(False, f"unrecognized GPU architecture {arch!r}")
    gen = int(match.group(1))
    need = 18 if match.group(2) == "p" else 17
    if gen < need:
        return Availability(
            False, f"GPU {arch} predates the neural accelerators (gen {gen} < {need})"
        )
    return Availability(True)


def qmm(qa: Any, sa: Any, ra: Any, weight: Any, scales: Any, biases: Any) -> Any:
    """``[M, N]`` in ``scales.dtype`` from Stage-A outputs and a projection's packed
    affine-Q4 weight with its scales/biases in their stored ``[N, K/64]`` layout."""
    import mlx.core as mx

    m, k = qa.shape
    n = weight.shape[0]
    if n % BN or k % GROUP:
        raise ValueError(
            f"int8 prefill GEMM needs N % {BN} == 0 and K % {GROUP} == 0; got N={n} K={k}"
        )
    (out,) = _kernel(
        "sous_int8_q4_gemm",
        ["qa", "sa", "ra", "w", "scales", "biases", "Mp"],
        ["out"],
        _kernel_text("q4_int8_gemm.metal"),
        _kernel_text("common.h") + _kernel_text("nax.h"),
    )(
        inputs=[
            qa,
            sa,
            ra,
            weight,
            mx.contiguous(scales.T),
            mx.contiguous(biases.T),
            mx.array([m], dtype=mx.int32),
        ],
        template=[
            ("T", scales.dtype),
            ("WM", TILE_WM),
            ("WN", TILE_WN),
            ("K", k),
            ("N", n),
        ],
        grid=(32 * (n // BN), TILE_WM * ((m + BM - 1) // BM), TILE_WN),
        threadgroup=(32, TILE_WM, TILE_WN),
        output_shapes=[(m, n)],
        output_dtypes=[scales.dtype],
    )
    return out


def linear_int8(linear: Any, x: Any, stage: tuple[Any, Any, Any] | None = None) -> Any:
    """One tagged projection on the INT8 path; ``stage`` is a Stage A already taken
    for this exact ``x`` (shared across gate/up or qkv/z), else it is taken here."""
    qa, sa, ra = stage if stage is not None else stage_a(x)
    out = qmm(qa, sa, ra, linear.weight, linear.scales, linear.biases)
    return out.reshape(*x.shape[:-1], linear.weight.shape[0])


# --------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------

# Set with object.__setattr__ so they live in the module's __dict__: nn.Module's own
# __setattr__ would put a bool or a tuple of arrays INTO the parameter tree, where
# parameters() and named_modules() would find it.
_TAG = "_sous_int8_prefill"
_STAGE = "_sous_int8_stage"


def eligible_linear(linear: Any) -> bool:
    """Affine Q4 gs64 ``nn.QuantizedLinear`` with fp16/bf16 scales, no bias, and a
    column count the GEMM tiles by 64 — the only shape the kernel decodes."""
    import mlx.core as mx
    import mlx.nn as nn

    if not isinstance(linear, nn.QuantizedLinear) or "bias" in linear:
        return False
    if getattr(linear, "mode", None) != "affine":
        return False
    if getattr(linear, "bits", None) != 4 or getattr(linear, "group_size", None) != GROUP:
        return False
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return False
    if weight.dtype != mx.uint32 or weight.ndim != 2 or scales.ndim != 2:
        return False
    if scales.dtype not in (mx.float16, mx.bfloat16) or biases.dtype != scales.dtype:
        return False
    if scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        return False
    n = weight.shape[0]
    k = scales.shape[1] * GROUP
    return weight.shape[1] * 32 == k * 4 and n % BN == 0 and k % GROUP == 0


def _tagged(module: Any) -> bool:
    return bool(getattr(module, _TAG, False))


def _rows_ok(x: Any) -> bool:
    import mlx.core as mx

    return x.ndim >= 2 and x.shape[-2] >= MIN_ROWS and x.dtype in (mx.bfloat16, mx.float16)


def _root(model: Any) -> Any:
    # mlx-vlm wraps the text model as `language_model`; mlx-lm's Model is the root.
    return getattr(model, "language_model", model)


def _model_type(model: Any) -> str:
    # mlx-vlm keeps it on `config`; mlx-lm on the model and its `args`.
    for holder in (getattr(model, "config", None), model, getattr(model, "args", None)):
        value = getattr(holder, "model_type", None) if holder is not None else None
        if isinstance(value, str) and value:
            return value
    return ""


def _is_mlp(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj"))


def _tag(model: Any) -> int:
    """Tag eligible projections under the decoder layers; return how many."""
    import mlx.nn as nn

    count = 0
    seen: set[int] = set()
    for path, module in _root(model).named_modules():
        if ".layers." not in f".{path}." or "self_attn" in path:
            continue
        if _is_mlp(module):
            trio = (module.gate_proj, module.up_proj, module.down_proj)
            # All or nothing: with only gate/up tagged the MLP wrapper would fall
            # through to the original call, whose gate/up would then route alone.
            if all(eligible_linear(m) for m in trio):
                for m in trio:
                    if id(m) not in seen:
                        object.__setattr__(m, _TAG, True)
                        seen.add(id(m))
                        count += 1
            continue
        if isinstance(module, nn.QuantizedLinear) and eligible_linear(module):
            if id(module) in seen or _in_mlp_path(path):
                continue
            object.__setattr__(module, _TAG, True)
            seen.add(id(module))
            count += 1
    return count


def _in_mlp_path(path: str) -> bool:
    return path.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")


def _untag(model: Any) -> None:
    for _, module in _root(model).named_modules():
        if _TAG in getattr(module, "__dict__", {}):
            object.__setattr__(module, _TAG, False)
        if _STAGE in getattr(module, "__dict__", {}):
            object.__setattr__(module, _STAGE, None)


_WRAPPED: set[type] = set()
_QUANTIZED_LINEAR_WRAPPED = False

_TARGETS = (
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5MLP", "mlp"),
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5GatedDeltaNet", "gdn"),
    ("mlx_lm.models.qwen3_5", "MLP", "mlp"),
    ("mlx_lm.models.qwen3_5", "GatedDeltaNet", "gdn"),
)


def _resolve_targets() -> list[tuple[type, str]]:
    import importlib

    found: list[tuple[type, str]] = []
    for module_name, class_name, kind in _TARGETS:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except ImportError, AttributeError:
            continue
        found.append((cls, kind))
    return found


def _wrap_mlp(cls: type) -> None:
    import mlx.nn as nn

    orig = cls.__call__

    def call(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            _rows_ok(x)
            and _tagged(self.gate_proj)
            and _tagged(self.up_proj)
            and _tagged(self.down_proj)
        ):
            shared = stage_a(x)
            gate = linear_int8(self.gate_proj, x, shared)
            up = linear_int8(self.up_proj, x, shared)
            return linear_int8(self.down_proj, nn.silu(gate) * up)
        return orig(self, x, *args, **kwargs)

    cls.__call__ = call  # ty: ignore[invalid-assignment]


def _wrap_gdn(cls: type) -> None:
    orig = cls.__call__

    def call(self: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        share = [
            proj
            for proj in (getattr(self, "in_proj_qkv", None), getattr(self, "in_proj_z", None))
            if proj is not None and _tagged(proj)
        ]
        if not share or not _rows_ok(inputs):
            return orig(self, inputs, *args, **kwargs)
        stage = stage_a(inputs)
        for proj in share:
            object.__setattr__(proj, _STAGE, (inputs, stage))
        try:
            return orig(self, inputs, *args, **kwargs)
        finally:
            for proj in share:
                object.__setattr__(proj, _STAGE, None)

    cls.__call__ = call  # ty: ignore[invalid-assignment]


def _wrap_quantized_linear() -> None:
    import mlx.nn as nn

    orig = nn.QuantizedLinear.__call__

    def call(self: Any, x: Any) -> Any:
        if _tagged(self) and _rows_ok(x):
            handed = getattr(self, _STAGE, None)
            # Identity, not equality: the GDN wrapper handed a stage for this exact
            # input object; any other array gets its own Stage A.
            stage = handed[1] if handed is not None and handed[0] is x else None
            return linear_int8(self, x, stage)
        return orig(self, x)

    nn.QuantizedLinear.__call__ = call


def install_wrappers(classes: Iterable[tuple[type, str]] | None = None) -> None:
    """Wrap the MLP / GDN classes (default: the Qwen3.5 classes of mlx-vlm and
    mlx-lm that import) and ``nn.QuantizedLinear``, once per process. Wrappers
    act only on tagged modules, so installing them is behaviour-neutral for every
    other model in the process."""
    global _QUANTIZED_LINEAR_WRAPPED
    for cls, kind in _resolve_targets() if classes is None else classes:
        if cls in _WRAPPED:
            continue
        (_wrap_mlp if kind == "mlp" else _wrap_gdn)(cls)
        _WRAPPED.add(cls)
    if not _QUANTIZED_LINEAR_WRAPPED:
        _wrap_quantized_linear()
        _QUANTIZED_LINEAR_WRAPPED = True


def _warm_up(model: Any) -> None:
    """Compile every kernel instantiation the tagged projections need, on a zero
    input of MIN_ROWS rows: first-turn latency stays flat, and a compiler that
    rejects the source or a changed metal_kernel signature fails here, inside
    enable()'s degradation path, rather than inside a user's turn."""
    import mlx.core as mx
    import mlx.nn as nn

    done: set[tuple[int, int, Any]] = set()
    for _, module in _root(model).named_modules():
        if not (isinstance(module, nn.QuantizedLinear) and _tagged(module)):
            continue
        k = module.scales.shape[1] * GROUP
        key = (k, module.weight.shape[0], module.scales.dtype)
        if key in done:
            continue
        done.add(key)
        mx.eval(linear_int8(module, mx.zeros((MIN_ROWS, k), dtype=module.scales.dtype)))


def _refuse(reason: str) -> dict[str, Any]:
    """The requested accelerator cannot serve this load: say so once and report why.
    A silently inert opt-in is worse than one warning line."""
    warnings.warn(
        f"sous: int8 prefill requested but unavailable ({reason}); prefilling with stock kernels",
        stacklevel=3,
    )
    return {"state": "unavailable", "reason": reason, "routed": 0}


def enable(model: Any, *, enabled: bool) -> dict[str, Any]:
    """Opt one loaded model in. Returns the status the engine exposes; never raises
    — a model load must not fail because a prefill accelerator is missing."""
    if not enabled:
        return {"state": "off", "reason": None, "routed": 0}
    avail = availability()
    if not avail.available:
        return _refuse(avail.reason or "unavailable")
    model_type = _model_type(model)
    if model_type not in SUPPORTED_MODEL_TYPES:
        return _refuse(f"unsupported model type {model_type or 'unknown'!r} (dense qwen3_5 only)")
    try:
        routed = _tag(model)
        if routed == 0:
            _untag(model)
            return _refuse("no eligible projections (affine Q4 gs64 required)")
        install_wrappers()
        _warm_up(model)
    except Exception as e:  # noqa: BLE001 — degrade, never block the model
        _untag(model)
        return _refuse(str(e))
    logger.info("int8 prefill: routed %d projections", routed)
    return {"state": "active", "reason": None, "routed": routed}
