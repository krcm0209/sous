"""Metal sources for sous's runtime-compiled kernels: int8 prefill, the
attention tile, kv8's int8 attention and the small-row projection kernel.

Read by ``sous.engine.int8prefill``, ``sous.engine.tileattn``,
``sous.engine.kv8`` and ``sous.engine.projkernel`` through
``importlib.resources`` and handed to ``mx.fast.metal_kernel``, which compiles
them at model load. The ``.metal`` files are kernel *bodies* (mlx generates the
signature), so they do not compile standalone. ``common.h`` is prepended to every
body; ``nax.h``, which includes the Metal-4 tensor-op header, only to the
tensor-op kernels: the int8 GEMM and the tile's and kv8's split kernels.
``attention_tile.metal`` holds two bodies, the split kernel and the reduce kernel,
divided at its ``// REDUCE`` line. ``attention_tile_kv8.metal`` is the tile's
split kernel over int8 K, V and Q, whose partials the tile's reduce kernel
combines, and ``kv8_quant.metal`` the row quantiser that fills its int8 mirror;
the quantiser uses no tensor ops, so CI runs its tests. ``projection_mma.metal``
holds three, the plain kernel, the staged kernel and the group-sums kernel,
divided at its ``// STAGED`` and ``// GROUP_SUMS`` lines; ``projection_mma.h``
holds the 8x8x8 simdgroup multiply-accumulate helper and the presum and slab
switches. ``projection_row.metal`` is the one-row kernel, which serves a single
row in the same summation order with no MMA: two bodies, over mlx's layout and
over slab-ordered weights, divided at its ``// SLAB`` line. None of the
projection kernels uses tensor ops, so they compile on any Metal GPU and CI runs
their tests.

The int8 prefill kernels are derived from oMLX (jundot/omlx#3548) and the
projection kernel in part from Splash (incoai/splash and its Apple7/8 port), both
under the Apache License 2.0; see THIRD_PARTY_NOTICES.md. ``attention_tile.metal``,
``attention_tile_kv8.metal`` and ``kv8_quant.metal`` are sous's own, and so is
``projection_row.metal`` apart from the nibble decode and epilogue it shares with
the projection kernel.
"""
