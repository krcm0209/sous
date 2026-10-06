"""Metal sources for sous's runtime-compiled kernels: int8 prefill, the
attention tile and the small-row projection kernel.

Read by ``sous.engine.int8prefill``, ``sous.engine.tileattn`` and
``sous.engine.projkernel`` through ``importlib.resources`` and handed to
``mx.fast.metal_kernel``, which compiles them at model load. The ``.metal`` files
are kernel *bodies* (mlx generates the signature), so they do not compile
standalone. ``common.h`` is prepended to every body; ``nax.h``, which includes the
Metal-4 tensor-op header, only to the tensor-op kernels: the int8 GEMM and the
tile's split kernel. ``attention_tile.metal`` holds two bodies, the split kernel
and the reduce kernel, divided at its ``// REDUCE`` line. ``projection_mma.metal``
holds three, the plain kernel, the staged kernel and the group-sums kernel,
divided at its ``// STAGED`` and ``// GROUP_SUMS`` lines; ``projection_mma.h``
holds the 8x8x8 simdgroup multiply-accumulate helper and the presum switch.
``projection_row.metal`` is the one-row kernel, which serves a single row in the
same summation order with no MMA. None of these uses tensor ops, so they compile
on any Metal GPU and CI runs their tests.

The int8 kernels are derived from oMLX (jundot/omlx#3548) and the projection
kernel in part from Splash (incoai/splash and its Apple7/8 port), both under the
Apache License 2.0; see THIRD_PARTY_NOTICES.md. ``attention_tile.metal`` is sous's
own, and so is ``projection_row.metal`` apart from the nibble decode and epilogue it
shares with the projection kernel.
"""
