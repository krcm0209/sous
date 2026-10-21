"""Metal sources for sous's runtime-compiled kernels: int8 prefill, the
attention tile, kv8's int8 attention, the small-row projection kernel and the
projection tile.

Read by ``sous.engine.int8prefill``, ``sous.engine.tileattn``,
``sous.engine.kv8``, ``sous.engine.projkernel`` and ``sous.engine.projtile``
through ``importlib.resources`` and handed to ``mx.fast.metal_kernel``, which
compiles them at model load. The ``.metal`` files are kernel *bodies* (mlx
generates the signature), so they do not compile standalone. ``common.h`` is
prepended to every body; ``nax.h``, which includes the Metal-4 tensor-op header,
only to the tensor-op kernels: the int8 GEMM, the attention tile's and kv8's
split kernels and the projection tile. ``attention_tile.metal`` holds two
bodies, the split kernel and the reduce kernel, divided at its ``// REDUCE``
line. ``attention_tile_kv8.metal`` is the tile's split kernel over int8 K, V and
Q, whose partials the tile's reduce kernel combines, and ``kv8_quant.metal`` the
row quantiser that fills its int8 mirror; the quantiser uses no tensor ops, so
CI runs its tests. ``projection_mma.metal`` holds three, the plain kernel, the
staged kernel and the group-sums kernel, divided at its ``// STAGED`` and
``// GROUP_SUMS`` lines; ``projection_mma.h`` holds the 8x8x8 simdgroup
multiply-accumulate helper and the presum and slab switches.
``projection_row.metal`` is the one-row kernel, which serves a single row in the
same summation order with no MMA: two bodies, over mlx's layout and over
slab-ordered weights, divided at its ``// SLAB`` line. None of those projection
kernels uses tensor ops, so they compile on any Metal GPU and CI runs their
tests. ``projection_tile.metal`` is the projection tile, which runs one MPP
``matmul2d`` per quant group over slab-ordered weights on the tensor units: it
holds the tile body and the split's fixed-order reduce, divided at its
``// REDUCE`` line. ``projection_tile.h`` holds the tile's helpers (the row sums,
the traversal guard and the per-group matmul and epilogue) and is prepended to
the tile body alone, after the ``DUAL`` and ``SPLITOUT`` defines projtile.py
writes and ``nax.h``; the reduce body gets ``common.h`` only.

The int8 prefill kernels are derived from oMLX (jundot/omlx#3548), the
projection kernel in part from Splash (incoai/splash and its Apple7/8 port), and
the projection tile from Splash's ``q4_mpp_tiles.h`` and ``linear_q4.metal`` at
0d5a035, all under the Apache License 2.0; see THIRD_PARTY_NOTICES.md. The slab
weight layout read by ``projection_mma.metal``'s ``SLAB`` bodies,
``projection_row.metal``'s slab body and ``q4_int8_gemm.metal``'s ``SLAB`` read is
Splash's too.
``attention_tile.metal``, ``attention_tile_kv8.metal`` and ``kv8_quant.metal`` are
sous's own, and so is ``projection_row.metal`` apart from the nibble decode and
epilogue it shares with the projection kernel and its slab body's layout.
"""
