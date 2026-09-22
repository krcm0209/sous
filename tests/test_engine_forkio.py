"""Writing a cache's buffers to disk and reading them back, on the real
mlx-vlm cache classes with tiny arrays (no model): bit-exact, the offset
from metadata rather than the padded shape, and a further extension that
matches a cache that was never serialised — capacity included."""

from __future__ import annotations

from pathlib import Path

import pytest

from sous.engine import forkio
from sous.engine.forkstore import ForkFileError, read_header

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


@pytest.mark.parametrize(
    "tamper",
    ["ids", "kinds", "count", "offset"],
)
def test_restore_refuses_a_file_that_does_not_match(tmp_path: Path, tamper: str):
    kv, _ = _kv((512,))
    ar = _arrays()
    path = tmp_path / "512-x.safetensors"
    forkio.persist_cache([kv, ar], path, IDS[:512], {**META, "n_tokens": "512"})
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
    with pytest.raises(ForkFileError):
        forkio.restore_cache(path, header, fresh, ids)


def test_persist_refuses_an_unsupported_layer(tmp_path: Path):
    with pytest.raises(forkio.ForkUnsupported):
        forkio.persist_cache([cache_mod.RotatingKVCache(max_size=8)], tmp_path / "x", [1], META)


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
