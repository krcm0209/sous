"""Writing a cache's buffers to disk and reading them back, on the real
mlx-vlm cache classes with tiny arrays (no model): bit-exact, the offset
from metadata rather than the padded shape, and a further extension that
matches a cache that was never serialised — capacity included."""

from __future__ import annotations

import json
import re
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

import pytest

from sous.engine import forkio
from sous.engine.forkstore import GIB, ForkFileError, ForkStore, digest, read_header
from sous.engine.promptcache import fork_file_bytes

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_vlm.models.cache")

IDS = list(range(1000))
META = {"format": "sous-fork-v1", "layout": "1", "n_tokens": "1000", "boundary": "tools"}


def _kv(chunks: tuple[int, ...], seed: int = 0):
    """A KVCache fed random bf16 keys/values in chunks, and the arrays it saw."""
    mx.random.seed(seed)
    c = cache_mod.KVCache()
    fed = []
    for t in chunks:
        k = mx.random.normal((1, 4, t, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 4, t, 256)).astype(mx.bfloat16)
        c.update_and_fetch(k, v)
        fed.append((k, v))
    mx.eval(c.keys, c.values)
    return c, fed


def _arrays(seed: int = 1):
    mx.random.seed(seed)
    c = cache_mod.ArraysCache(size=2)
    c[0] = mx.random.normal((1, 16, 128, 128))
    c[1] = mx.random.normal((1, 16, 128))
    mx.eval(c[0], c[1])
    return c


def _equal(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def test_layer_kinds_names_the_two_supported_classes_and_refuses_others():
    assert forkio.layer_kinds([cache_mod.KVCache(), cache_mod.ArraysCache(size=2)]) == [
        "kv",
        "arrays",
    ]
    with pytest.raises(forkio.ForkUnsupported):
        forkio.layer_kinds([cache_mod.RotatingKVCache(max_size=8)])


def test_round_trip_is_bit_exact_and_restores_the_offset_from_metadata(tmp_path: Path):
    kv, _ = _kv((700, 300))  # offset 1000 in a step-padded buffer
    ar = _arrays()
    path = tmp_path / "1000-x.safetensors"
    forkio.persist_cache([kv, ar], path, IDS, META)
    header = read_header(path)
    assert header.metadata["c0_kind"] == "kv" and header.metadata["c0_offset"] == "1000"
    assert header.metadata["c1_kind"] == "arrays" and header.metadata["c1_size"] == "2"
    assert header.metadata["kinds"] == "kv,arrays"
    assert header.tensors["c0_k"].shape == tuple(kv.keys.shape)  # the full padded buffer
    assert header.tensors["ids"].shape == (1000,)
    fresh = [cache_mod.KVCache(), cache_mod.ArraysCache(size=2)]
    forkio.restore_cache(path, header, fresh, IDS)
    assert fresh[0].offset == 1000 and fresh[0].keys.shape[2] == kv.keys.shape[2]
    assert _equal(fresh[0].keys, kv.keys) and _equal(fresh[0].values, kv.values)
    assert fresh[0].keys.dtype == mx.bfloat16
    assert _equal(fresh[1][0], ar[0]) and _equal(fresh[1][1], ar[1])


def test_a_restored_cache_extends_like_one_that_was_never_serialised(tmp_path: Path):
    kv, _ = _kv((700, 300))
    twin, _ = _kv((700, 300))
    path = tmp_path / "1000-x.safetensors"
    forkio.persist_cache([kv], path, IDS, META)
    fresh = [cache_mod.KVCache()]
    forkio.restore_cache(path, read_header(path), fresh, IDS)
    mx.random.seed(9)
    k = mx.random.normal((1, 4, 130, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 4, 130, 256)).astype(mx.bfloat16)
    fresh[0].update_and_fetch(k, v)
    twin.update_and_fetch(k, v)
    assert fresh[0].offset == twin.offset == 1130
    assert fresh[0].keys.shape == twin.keys.shape  # same capacity: no forced regrowth
    assert _equal(fresh[0].keys[..., :1130, :], twin.keys[..., :1130, :])
    assert _equal(fresh[0].values[..., :1130, :], twin.values[..., :1130, :])


def test_an_absent_arrays_state_is_recorded_and_restored_as_none(tmp_path: Path):
    ar = cache_mod.ArraysCache(size=2)
    ar[0] = mx.zeros((1, 2, 2))
    mx.eval(ar[0])
    path = tmp_path / "1-x.safetensors"
    forkio.persist_cache([ar], path, [1], {**META, "n_tokens": "1"})
    header = read_header(path)
    assert header.metadata["c0_s1_none"] == "1" and "c0_s1" not in header.tensors
    fresh = [cache_mod.ArraysCache(size=2)]
    forkio.restore_cache(path, header, fresh, [1])
    assert fresh[0][1] is None and _equal(fresh[0][0], ar[0])


def _rewrite_header(path: Path, mutate: Callable[[dict], None]) -> None:
    """Rewrite `path`'s safetensors header through `mutate`, keeping every
    data byte where it was: the corruptions a header-only check can miss."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    mutate(header)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + n :])


@pytest.mark.parametrize(
    ("tamper", "refusal"),
    [
        ("ids", "ids differ"),
        ("kinds", "layer layout"),
        ("count", "layer layout"),
        ("offset", "layer 0 offset 600"),
        ("n_tokens", "n_tokens '511'"),
        ("no_ids", "no ids tensor"),
        ("no_offset", "layer 0 incomplete"),
        ("no_size", "layer 1 incomplete"),
        ("size", "layer 1 has 3 states, want 2"),
        ("no_state", "layer 1 state 1 missing"),
    ],
)
def test_restore_refuses_a_file_that_does_not_match(tmp_path: Path, tamper: str, refusal: str):
    kv, _ = _kv((512,))
    ar = _arrays()
    path = tmp_path / "512-x.safetensors"
    forkio.persist_cache([kv, ar], path, IDS[:512], {**META, "n_tokens": "512"})
    # Renamed rather than dropped, so the bytes still add up and the file
    # passes the header check to reach the loader's own.
    if tamper == "no_ids":
        _rewrite_header(path, lambda h: h.update(idz=h.pop("ids")))
    elif tamper == "no_state":
        _rewrite_header(path, lambda h: h.update(c1_sx=h.pop("c1_s1")))
    header = read_header(path)
    fresh = [cache_mod.KVCache(), cache_mod.ArraysCache(size=2)]
    ids = IDS[:512]
    if tamper == "ids":
        ids = [*IDS[:511], 999]
    elif tamper == "kinds":
        fresh = [cache_mod.ArraysCache(size=2), cache_mod.KVCache()]
    elif tamper == "count":
        fresh = [cache_mod.KVCache()]
    elif tamper == "offset":
        header.metadata["c0_offset"] = "600"
    elif tamper == "n_tokens":
        header.metadata["n_tokens"] = "511"
    elif tamper == "no_offset":
        del header.metadata["c0_offset"]
    elif tamper == "no_size":
        del header.metadata["c1_size"]
    elif tamper == "size":
        header.metadata["c1_size"] = "3"
    with pytest.raises(ForkFileError, match=re.escape(refusal)):
        forkio.restore_cache(path, header, fresh, ids)


@pytest.mark.parametrize("tamper", ["shape", "dtype", "unknown_dtype", "json"])
def test_a_tensor_that_disagrees_with_its_bytes_is_a_verification_failure(
    tmp_path: Path, tamper: str
):
    """The header check sees only the aggregate size; mx.load checks each
    tensor's shape and dtype against its byte range and raises its own
    RuntimeError, which must reach the store as a ForkFileError — else the
    file is kept, and every turn with that prefix reads it again."""
    kv, _ = _kv((512,))
    path = tmp_path / "512-x.safetensors"
    forkio.persist_cache([kv], path, IDS[:512], {**META, "n_tokens": "512"})
    if tamper == "shape":
        _rewrite_header(path, lambda h: h["c0_k"].update(shape=[1, 4, 256, 256]))
    elif tamper == "dtype":
        _rewrite_header(path, lambda h: h["c0_k"].update(dtype="F32"))
    elif tamper == "unknown_dtype":
        _rewrite_header(path, lambda h: h["c0_k"].update(dtype="Q9"))
    else:
        # A bare NaN: Python's parser takes it, mlx's refuses the header.
        _rewrite_header(path, lambda h: h["__metadata__"].update(probe=float("nan")))
    header = read_header(path)  # the aggregate size still matches
    fresh = [cache_mod.KVCache()]
    prefix = r"\[(load_safetensors|safetensor|json\.exception)"
    with pytest.raises(ForkFileError, match=r"512-x\.safetensors: " + prefix):
        forkio.restore_cache(path, header, fresh, IDS[:512])
    assert fresh[0].keys is None


def test_a_file_mx_load_cannot_open_is_not_a_verification_failure(tmp_path: Path):
    """mlx's "Failed to open" is about the path, not the bytes (a file
    removed or made unreadable after the header read): the store keeps what
    may be a good file rather than count a strike against it."""
    kv, _ = _kv((8,))
    path = tmp_path / "8-x.safetensors"
    forkio.persist_cache([kv], path, IDS[:8], {**META, "n_tokens": "8"})
    header = read_header(path)
    path.unlink()
    with pytest.raises(RuntimeError, match="Failed to open") as info:
        forkio.restore_cache(path, header, [cache_mod.KVCache()], IDS[:8])
    assert not isinstance(info.value, ForkFileError)


def _store(tmp_path: Path, budget: int = 10 * GIB) -> ForkStore:
    return ForkStore(
        tmp_path / "forks",
        {"backend": "vlm"},
        budget,
        disk_usage=lambda p: type("u", (), {"total": 1000 * GIB, "free": 200 * GIB})(),
        pid_alive=lambda pid: False,
        pid=lambda: 4242,
    )


def test_the_store_deletes_and_counts_a_file_mx_load_refuses(tmp_path: Path):
    """End to end with the real reader: the corrupt file leaves the disk and
    the index, so the next cold turn writes a good one in its place."""
    kv, _ = _kv((512,))
    ids = IDS[:512]
    s = _store(tmp_path)
    assert s.persist(
        ids,
        "tools",
        lambda p, m: forkio.persist_cache([kv], p, ids, m),
        expected_bytes=fork_file_bytes([kv], len(ids)),
    )
    path = s._path_for(len(ids), digest(ids))
    _rewrite_header(path, lambda h: h["c0_v"].update(shape=[1, 4, 256, 256]))
    entry = s.longest_prefix([*ids, 7])
    assert entry is not None
    with pytest.warns(UserWarning, match="fork restore failed .*; deleted"):
        assert (
            s.restore(entry, lambda p, h: forkio.restore_cache(p, h, [cache_mod.KVCache()], ids))
            is False
        )
    assert not path.exists() and s._failures == 1 and s.forks == 0
    assert s.touch(ids) is False  # absent: the boundary loop writes it again


def test_the_real_writer_on_a_vanished_key_directory_is_one_failure(tmp_path: Path):
    """mx.save_safetensors reports a missing directory as a RuntimeError, not
    an OSError: one counted failure, the store stays on, and the next write
    recreates the layout."""
    kv, _ = _kv((8,))
    ids = IDS[:8]
    s = _store(tmp_path)

    def write(path: Path, metadata: dict[str, str]) -> None:
        shutil.rmtree(tmp_path / "forks")
        forkio.persist_cache([kv], path, ids, metadata)

    with pytest.warns(UserWarning, match=r"fork persist failed \(RuntimeError\)"):
        assert s.persist(ids, "tools", write, expected_bytes=fork_file_bytes([kv], 8)) is False
    assert s.state == "active" and s._write_failures == 1
    assert s.persist(
        ids,
        "tools",
        lambda p, m: forkio.persist_cache([kv], p, ids, m),
        expected_bytes=fork_file_bytes([kv], 8),
    )
    assert s.forks == 1


def test_the_estimate_bounds_a_real_file_of_the_default_models_shape(tmp_path: Path):
    """The store refuses and budgets from fork_file_bytes before any byte
    is written, so it must bound what persist_cache really writes — header
    included, which grows with the layer count (the default model has 64:
    three recurrent layers to each attention layer) — without overshooting
    by more than the header allowance."""
    mx.random.seed(3)
    cache = []
    for _ in range(16):
        for _ in range(3):
            ar = cache_mod.ArraysCache(size=2)
            ar[0], ar[1] = mx.random.normal((1, 2, 8, 8)), mx.random.normal((1, 2, 8))
            cache.append(ar)
        kv = cache_mod.KVCache()
        k = mx.random.normal((1, 2, 300, 16)).astype(mx.bfloat16)
        kv.update_and_fetch(k, k)
        cache.append(kv)
    forkio.eval_cache(cache)
    ids = IDS[:300]
    path = tmp_path / "300-x.safetensors"
    forkio.persist_cache(cache, path, ids, {**META, "n_tokens": "300"})
    size = path.stat().st_size
    estimate = fork_file_bytes(cache, len(ids))
    assert size <= estimate <= size + (64 << 10)


def test_persist_refuses_a_kv_cache_that_was_never_filled(tmp_path: Path):
    with pytest.raises(forkio.ForkUnsupported, match="empty KVCache"):
        forkio.persist_cache([cache_mod.KVCache()], tmp_path / "0-x", [], META)


def test_persist_refuses_an_unsupported_layer(tmp_path: Path):
    with pytest.raises(forkio.ForkUnsupported):
        forkio.persist_cache([cache_mod.RotatingKVCache(max_size=8)], tmp_path / "x", [1], META)


def test_persist_refuses_a_kv_cache_whose_offset_does_not_match_the_ids(tmp_path: Path):
    """An offset that disagrees with the ids being persisted would only
    surface later, as a restore-time verification strike against a file
    that was never a valid snapshot of those ids to begin with."""
    kv, _ = _kv((700, 300))  # offset 1000
    with pytest.raises(forkio.ForkUnsupported):
        forkio.persist_cache([kv], tmp_path / "x", IDS[:999], {**META, "n_tokens": "999"})


def test_a_path_without_the_suffix_gets_one_from_mlx(tmp_path: Path):
    """mx.save_safetensors appends .safetensors to a name that lacks it; the
    store's temp names carry it already, and this pins why they must."""
    kv, _ = _kv((8,))
    forkio.persist_cache([kv], tmp_path / "8-x", IDS[:8], {**META, "n_tokens": "8"})
    assert (tmp_path / "8-x.safetensors").exists()


def test_eval_cache_materialises_every_layer():
    kv, _ = _kv((8,))
    lazy = [cache_mod.KVCache(), cache_mod.ArraysCache(size=2)]
    lazy[0].keys, lazy[0].values, lazy[0].offset = mx.array(kv.keys), mx.array(kv.values), 8
    lazy[1][0] = mx.array(kv.keys)
    forkio.eval_cache(lazy)  # must not raise, and must leave the arrays usable
    assert _equal(lazy[0].keys, kv.keys) and _equal(lazy[1][0], kv.keys)


def test_restore_does_not_mutate_cache_on_error(tmp_path: Path):
    """On ForkFileError, the cache must remain untouched. Validating all
    layers before applying any assignment ensures a mismatch at layer i
    does not leave layers 0..i-1 holding unevaluated file arrays."""
    kv, _ = _kv((8,))
    ar = _arrays()
    path = tmp_path / "8-x.safetensors"
    forkio.persist_cache([kv, ar], path, IDS[:8], {**META, "n_tokens": "8"})
    header = read_header(path)
    fresh = [cache_mod.KVCache(), cache_mod.ArraysCache(size=3)]
    with pytest.raises(ForkFileError):
        forkio.restore_cache(path, header, fresh, IDS[:8])
    assert fresh[0].keys is None and fresh[0].offset == 0
    assert fresh[1][0] is None
