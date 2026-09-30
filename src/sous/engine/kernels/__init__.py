"""Metal sources for sous's runtime-compiled kernels: int8 prefill and the
attention tile.

Read by ``sous.engine.int8prefill`` and ``sous.engine.tileattn`` through
``importlib.resources`` and handed to ``mx.fast.metal_kernel``, which compiles them
at model load. The ``.metal`` files are kernel *bodies* (mlx generates the
signature), so they do not compile standalone. ``common.h`` is prepended to every
body; ``nax.h``, which includes the Metal-4 tensor-op header, only to the tensor-op
kernels: the int8 GEMM and the tile's split kernel. ``attention_tile.metal`` holds
two bodies, the split kernel and the reduce kernel, divided at its ``// REDUCE``
line.

The int8 kernels are derived from oMLX (jundot/omlx#3548, Apache License 2.0); see
THIRD_PARTY_NOTICES.md. ``attention_tile.metal`` is sous's own.
"""
