"""Metal sources for the int8 prefill kernels.

Read by ``sous.engine.int8prefill`` through ``importlib.resources`` and handed to
``mx.fast.metal_kernel``, which compiles them at model load. The ``.metal`` files are
kernel *bodies* (mlx generates the signature), so they do not compile standalone;
``common.h`` is prepended to both, ``nax.h`` to the GEMM only.

Derived from oMLX (jundot/omlx#3548, Apache License 2.0); see THIRD_PARTY_NOTICES.md.
"""
