"""The disk store's rules, exercised without mlx: digest, header parse,
identity key, naming and budgets."""

from __future__ import annotations

import json
import os
import shutil
import struct
import warnings
from pathlib import Path

import pytest

from sous.engine import forkstore
from sous.engine.forkstore import (
    FORK_LAYOUT,
    FORMAT,
    GIB,
    ForkFileError,
    ForkStore,
    Header,
    auto_budget,
    digest,
    engine_epoch,
    file_name,
    floor_bytes,
    key_name,
    parse_file_name,
    parse_temp_name,
    read_header,
    temp_name,
    weights_identity,
)


def write_safetensors(path: Path, tensors: dict[str, bytes], metadata: dict[str, str]) -> None:
    """A minimal safetensors writer: the header is JSON after an 8-byte
    little-endian length; every tensor here is int32 of the given bytes."""
    header: dict = {"__metadata__": metadata}
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "I32",
            "shape": [len(data) // 4],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for data in tensors.values():
            f.write(data)


# ---- digest ---------------------------------------------------------------


def test_digest_is_32_hex_over_little_endian_int32():
    d = digest([1, 2, 3])
    assert len(d) == 32 and int(d, 16) >= 0
    assert d == digest((1, 2, 3))  # any sequence, same bytes
    assert d != digest([1, 2, 4])
    assert d != digest([1, 2])


# ---- header ---------------------------------------------------------------


def test_read_header_parses_metadata_tensors_and_expected_size(tmp_path: Path):
    p = tmp_path / "f.safetensors"
    write_safetensors(p, {"ids": b"\x01\x00\x00\x00" * 3, "x": b"\x00" * 8}, {"format": FORMAT})
    h = read_header(p)
    assert h.metadata == {"format": FORMAT}
    assert h.tensors["ids"].shape == (3,) and h.tensors["ids"].dtype == "I32"
    assert (h.tensors["x"].start, h.tensors["x"].end) == (12, 20)
    assert h.expected_size == p.stat().st_size


def test_read_header_refuses_a_truncated_file(tmp_path: Path):
    p = tmp_path / "f.safetensors"
    write_safetensors(p, {"x": b"\x00" * 8}, {})
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ForkFileError):
        read_header(p)


def test_read_header_refuses_an_oversized_header(tmp_path: Path):
    p = tmp_path / "f.safetensors"
    p.write_bytes(struct.pack("<Q", 2 << 20) + b"{")
    with pytest.raises(ForkFileError):
        read_header(p)


def test_read_header_refuses_garbage(tmp_path: Path):
    p = tmp_path / "f.safetensors"
    p.write_bytes(b"not a safetensors file at all")
    with pytest.raises(ForkFileError):
        read_header(p)


# ---- identity key ---------------------------------------------------------


def test_key_name_is_32_hex_and_changes_with_any_field():
    base = {"backend": "vlm", "mlx": "0.32.2", "weights": "abc"}
    k = key_name(base)
    assert len(k) == 32
    assert key_name({**base, "weights": "abd"}) != k
    assert key_name({**base, "extra": "1"}) != k
    assert key_name(dict(reversed(list(base.items())))) == k  # order-independent


def test_engine_epoch_is_stable_and_16_hex(monkeypatch):
    a = engine_epoch()
    assert len(a) == 16 and engine_epoch() == a


def test_engine_epoch_changes_when_a_source_byte_changes(monkeypatch, tmp_path: Path):
    src = tmp_path / "engine"
    (src / "kernels").mkdir(parents=True)
    for name in ("vlm.py", "lm.py", "int8prefill.py"):
        (src / name).write_text(f"# {name}\n")
    (src / "kernels" / "a.metal").write_text("kernel\n")
    (src / "kernels" / "common.h").write_text("header\n")
    monkeypatch.setattr(forkstore, "_ENGINE_SOURCES", src)
    forkstore.engine_epoch.cache_clear()
    before = engine_epoch()
    (src / "vlm.py").write_text("# vlm.py changed\n")
    forkstore.engine_epoch.cache_clear()
    changed = engine_epoch()
    assert changed != before
    (src / "kernels" / "common.h").write_text("header changed\n")
    forkstore.engine_epoch.cache_clear()
    assert engine_epoch() != changed  # the headers are compiled into the kernels
    forkstore.engine_epoch.cache_clear()


def test_weights_identity_is_the_snapshot_sha_for_a_hub_layout(tmp_path: Path):
    snap = tmp_path / "models--x" / "snapshots" / "73e3e38d981303bc594367cd910ea6eb48349da8"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert weights_identity(snap / "config.json") == "73e3e38d981303bc594367cd910ea6eb48349da8"


def test_weights_identity_hashes_config_and_safetensors_for_a_local_dir(tmp_path: Path):
    d = tmp_path / "local-model"
    d.mkdir()
    (d / "config.json").write_text('{"quantization": {"bits": 4}}')
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    a = weights_identity(d / "config.json")
    assert len(a) == 32 and a != "local-model"
    (d / "model.safetensors").write_bytes(b"\x00" * 32)  # same config, new weights
    assert weights_identity(d / "config.json") != a


# ---- naming ---------------------------------------------------------------


def test_file_name_round_trips():
    d = digest([1, 2, 3])
    assert file_name(57344, d) == f"57344-{d}.safetensors"
    assert parse_file_name(f"57344-{d}.safetensors") == (57344, d)
    assert parse_file_name("57344-zz.safetensors") is None
    assert parse_file_name("key.json") is None
    assert parse_file_name(f"57344-{d}.4242.tmp.safetensors") is None


def test_temp_name_keeps_the_safetensors_suffix_and_carries_the_pid(tmp_path: Path):
    final = tmp_path / "1-abc.safetensors"
    t = temp_name(final, 4242)
    assert t.parent == final.parent
    assert t.name == "1-abc.4242.tmp.safetensors"
    assert parse_temp_name(t.name) == 4242
    assert parse_temp_name("1-abc.safetensors") is None


# ---- budgets --------------------------------------------------------------


def test_auto_budget_is_a_quarter_of_free_space_capped_at_16_gib():
    assert auto_budget(free=100 * GIB) == 16 * GIB
    assert auto_budget(free=40 * GIB) == 10 * GIB
    assert auto_budget(free=0) == 0


def test_floor_bytes_is_10_gib_or_five_percent_whichever_is_larger():
    assert floor_bytes(total=100 * GIB) == 10 * GIB
    assert floor_bytes(total=1000 * GIB) == 50 * GIB


KEY = {"backend": "vlm", "mlx": "0.32.2", "weights": "sha1"}


class Disk:
    """A fake shutil.disk_usage: total/free the test sets."""

    def __init__(self, total: int, free: int):
        self.total, self.free = total, free

    def __call__(self, path):
        return type("u", (), {"total": self.total, "free": self.free})()


def store(tmp_path: Path, budget=None, free=200 * GIB, total=1000 * GIB, **kw) -> ForkStore:
    return ForkStore(
        tmp_path / "forks",
        KEY,
        budget,
        disk_usage=Disk(total, free),
        pid_alive=kw.pop("pid_alive", lambda pid: False),
        pid=kw.pop("pid", lambda: 4242),
    )


def fake_write(size: int, ids: list[int] | None = None):
    """A writer the engine would supply: writes a minimal safetensors file
    of `size` payload bytes carrying the metadata and the ids tensor."""

    def write(path: Path, metadata: dict[str, str]) -> None:
        tensors = {"ids": array_bytes(ids or [1]), "x": b"\x00" * size}
        write_safetensors(path, tensors, metadata)

    return write


def array_bytes(ids: list[int]) -> bytes:
    import array

    return array.array("i", ids).tobytes()


IDS = list(range(1, 5001))


# ---- construction --------------------------------------------------------


def test_construction_creates_a_private_root_with_key_json_and_never_index(tmp_path: Path):
    s = store(tmp_path)
    assert s.state == "active" and s.reason is None
    root = tmp_path / "forks"
    assert (root / ".metadata_never_index").exists()
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    assert json.loads((root / s.key / "key.json").read_text()) == KEY
    assert s.status() == {
        "state": "active",
        "reason": None,
        "forks": 0,
        "bytes": 0,
        "budget_bytes": 16 * GIB,
        "evictions": 0,
    }


def test_auto_budget_is_derived_from_free_space(tmp_path: Path):
    assert store(tmp_path, free=40 * GIB).budget == 10 * GIB
    assert store(tmp_path, budget=3 * GIB).budget == 3 * GIB


def test_construction_disables_the_store_when_the_root_cannot_be_written(tmp_path: Path):
    blocker = tmp_path / "forks"
    blocker.write_text("a file where the directory should be")
    with pytest.warns(UserWarning, match="fork store"):
        s = store(tmp_path)
    assert s.state == "unavailable" and s.reason
    assert s.longest_prefix(IDS) is None
    assert s.persist(IDS[:100], "tools", fake_write(8)) is False


def test_construction_logs_one_line_naming_the_directory_budget_and_contents(tmp_path, caplog):
    with caplog.at_level("INFO", logger="sous.engine"):
        store(tmp_path)
    lines = [r.getMessage() for r in caplog.records if "forks on disk" in r.getMessage()]
    assert len(lines) == 1
    assert str(tmp_path / "forks") in lines[0] and "budget=16.0GiB" in lines[0]
    assert "found=0" in lines[0]


# ---- persist ---------------------------------------------------------------


def test_persist_writes_via_temp_then_replace_and_indexes_the_file(tmp_path: Path):
    s = store(tmp_path)
    seen: list[tuple[Path, dict]] = []

    def write(path: Path, metadata: dict[str, str]) -> None:
        seen.append((path, dict(metadata)))
        fake_write(64, IDS[:100])(path, metadata)

    assert s.persist(IDS[:100], "tools", write) is True
    temp, metadata = seen[0]
    assert temp.name.endswith(".4242.tmp.safetensors") and not temp.exists()
    final = tmp_path / "forks" / s.key / file_name(100, digest(IDS[:100]))
    assert final.exists() and oct(final.stat().st_mode & 0o777) == "0o600"
    assert metadata["format"] == FORMAT and metadata["layout"] == str(FORK_LAYOUT)
    assert metadata["n_tokens"] == "100" and metadata["boundary"] == "tools"
    assert all(metadata[k] == v for k, v in KEY.items())
    assert s.has(IDS[:100]) and not s.has(IDS[:99])
    assert (s.forks, s.bytes) == (1, final.stat().st_size)


def test_persist_skips_a_digest_already_on_disk_and_touches_it(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(64, IDS[:100]))
    final = tmp_path / "forks" / s.key / file_name(100, digest(IDS[:100]))
    os.utime(final, (1, 1))
    calls = []
    assert s.persist(IDS[:100], "tools", lambda p, m: calls.append(p)) is False
    assert calls == []
    assert final.stat().st_mtime > 1  # a resident fork's file must not be the oldest


def test_persist_refuses_when_the_write_would_breach_the_free_space_floor(tmp_path: Path):
    s = store(tmp_path, free=10 * GIB + 100, total=100 * GIB)  # floor is 10 GiB
    with pytest.warns(UserWarning, match="free space"):
        assert s.persist(IDS[:100], "tools", fake_write(200, IDS[:100])) is False
    assert s.state == "active"  # a skip, not a disable
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second refusal is silent
        assert s.persist(IDS[:200], "tools", fake_write(200, IDS[:200])) is False


def test_a_writer_that_raises_leaves_no_temp_and_warns(tmp_path: Path):
    s = store(tmp_path)

    def write(path: Path, metadata: dict[str, str]) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.warns(UserWarning, match="fork persist failed"):
        assert s.persist(IDS[:100], "tools", write) is False
    assert list((tmp_path / "forks" / s.key).glob("*.safetensors")) == []
    assert s.state == "active"


def test_a_repeated_writer_failure_warns_only_once(tmp_path: Path):
    """A layer class the format does not cover fails the same way on every
    cold turn; one line says so, not one per turn."""
    s = store(tmp_path)

    def write(path: Path, metadata: dict[str, str]) -> None:
        raise ValueError("layer class RotatingKVCache cannot be persisted")

    with pytest.warns(UserWarning, match="not persisting again"):
        assert s.persist(IDS[:100], "tools", write) is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert s.persist(IDS[:200], "tools", write) is False
    assert s.state == "active"


@pytest.mark.parametrize("errno_name", ["ENOSPC", "EACCES", "EROFS", "EDQUOT"])
def test_a_disk_error_disables_the_store(tmp_path: Path, errno_name: str):
    import errno

    s = store(tmp_path)

    def write(path: Path, metadata: dict[str, str]) -> None:
        raise OSError(getattr(errno, errno_name), "disk")

    with pytest.warns(UserWarning, match="fork persist failed"):
        assert s.persist(IDS[:100], "tools", write) is False
    assert s.state == "unavailable" and errno_name.lower() in (s.reason or "").lower()


def test_a_missing_directory_is_recreated_before_a_write(tmp_path: Path):
    s = store(tmp_path)
    shutil.rmtree(tmp_path / "forks")
    assert s.persist(IDS[:100], "tools", fake_write(64, IDS[:100])) is True
    assert s.state == "active" and s.forks == 1


def test_a_concurrent_writer_of_the_same_digest_skips(tmp_path: Path):
    s = store(tmp_path)
    inner: list[bool] = []

    def write(path: Path, metadata: dict[str, str]) -> None:
        # Re-enter for the same ids while this write is in flight: skipped.
        inner.append(s.persist(IDS[:100], "tools", fake_write(8, IDS[:100])))
        fake_write(64, IDS[:100])(path, metadata)

    assert s.persist(IDS[:100], "tools", write) is True
    assert inner == [False] and s.forks == 1


def test_persist_evicts_least_recently_used_across_keys_to_nine_tenths_of_the_budget(tmp_path):
    # Four prefixes of one length, so every file is the same size.
    a, b, c, d = IDS[:10], [*IDS[:9], 1001], [*IDS[:9], 1002], [*IDS[:9], 1003]
    other = ForkStore(
        tmp_path / "forks",
        {**KEY, "weights": "sha2"},
        10 * GIB,
        disk_usage=Disk(1000 * GIB, 200 * GIB),
        pid_alive=lambda p: False,
        pid=lambda: 1,
    )
    assert other.persist(a, "tools", fake_write(300, a))
    old = next((tmp_path / "forks" / other.key).glob("*.safetensors"))
    os.utime(old, (1, 1))
    size = old.stat().st_size
    # Three files fit under nine tenths of this budget; a fourth does not.
    s = store(tmp_path, budget=int(size * 3.6))
    assert s.persist(b, "tools", fake_write(300, b))
    assert s.persist(c, "tools", fake_write(300, c))
    assert old.exists() and s.evictions == 0
    assert s.persist(d, "tools", fake_write(300, d))
    assert not old.exists()  # the oldest across keys went first
    assert s.evictions == 1 and s.status()["evictions"] == 1 and s.forks == 3


def test_a_fork_larger_than_the_whole_budget_is_not_written(tmp_path: Path):
    s = store(tmp_path, budget=100)
    with pytest.warns(UserWarning, match="larger than the budget"):
        assert s.persist(IDS[:10], "tools", fake_write(300, IDS[:10])) is False
    assert s.forks == 0 and s.state == "active"
    assert list((tmp_path / "forks" / s.key).glob("*.safetensors")) == []


# ---- lookup ----------------------------------------------------------------


def test_longest_prefix_wants_a_strict_prefix_and_prefers_the_longest(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(8, IDS[:100]))
    assert s.persist(IDS[:200], "tools", fake_write(8, IDS[:200]))
    e = s.longest_prefix(IDS[:300])
    assert e is not None and e.n == 200 and e.digest == digest(IDS[:200])
    mid = s.longest_prefix(IDS[:200])
    assert mid is not None and mid.n == 100
    assert s.longest_prefix(IDS[:100]) is None  # an exact match leaves nothing to decode
    assert s.longest_prefix([9, *IDS[1:300]]) is None


def test_the_index_is_rebuilt_from_the_directory_at_construction(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(8, IDS[:100]))
    again = store(tmp_path)
    assert again.forks == 1 and again.has(IDS[:100])
    e = again.longest_prefix(IDS[:200])
    assert e is not None and e.size == s.bytes


def test_construction_skips_and_deletes_files_of_the_wrong_format_or_size(tmp_path: Path):
    s = store(tmp_path)
    d = tmp_path / "forks" / s.key
    write_safetensors(
        d / file_name(5, digest(IDS[:5])),
        {"x": b"\x00" * 8},
        {"format": "someone-elses", "n_tokens": "5"},
    )
    p = d / file_name(6, digest(IDS[:6]))
    fake_write(64, IDS[:6])(
        p, {"format": FORMAT, "layout": str(FORK_LAYOUT), "n_tokens": "6", **KEY}
    )
    p.write_bytes(p.read_bytes()[:-10])  # truncated
    again = store(tmp_path)
    assert again.forks == 0
    assert list(d.glob("*.safetensors")) == []


def test_construction_sweeps_dead_writers_temp_files_and_keeps_live_ones(tmp_path: Path):
    s = store(tmp_path)
    d = tmp_path / "forks" / s.key
    dead = d / "1-x.99.tmp.safetensors"
    live = d / "1-x.77.tmp.safetensors"
    dead.write_bytes(b"x")
    live.write_bytes(b"x")
    store(tmp_path, pid_alive=lambda pid: pid == 77)
    assert not dead.exists() and live.exists()


def test_construction_enforces_the_budget_over_every_key(tmp_path: Path):
    big = store(tmp_path, budget=10 * GIB)
    a, b = IDS[:10], [*IDS[:9], 1001]
    assert big.persist(a, "tools", fake_write(300, a))
    assert big.persist(b, "tools", fake_write(300, b))
    size = next((tmp_path / "forks" / big.key).glob("*.safetensors")).stat().st_size
    small = store(tmp_path, budget=size + size // 2)  # room for one
    assert small.forks == 1 and small.evictions == 1


# ---- restore ---------------------------------------------------------------


def test_restore_calls_the_loader_with_the_path_and_header_and_touches(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(64, IDS[:100]))
    e = s.longest_prefix(IDS[:200])
    assert e is not None
    os.utime(e.path, (1, 1))
    seen = []

    def load(path: Path, header: Header) -> None:
        seen.append((path, header.metadata["n_tokens"]))

    assert s.restore(e, load) is True
    assert seen == [(e.path, "100")]
    assert e.path.stat().st_mtime > 1


def test_a_verification_failure_deletes_the_file_and_three_in_a_row_disable(tmp_path: Path):
    s = store(tmp_path)
    for n in (100, 200, 300, 400):
        assert s.persist(IDS[:n], "tools", fake_write(8, IDS[:n]))

    def bad(path: Path, header: Header) -> None:
        raise ForkFileError("ids mismatch")

    for n in (100, 200):
        e = s.longest_prefix(IDS[: n + 1])
        assert e is not None and e.n == n
        with pytest.warns(UserWarning, match="fork restore failed"):
            assert s.restore(e, bad) is False
        assert not e.path.exists() and s.state == "active"
    e = s.longest_prefix(IDS[:301])
    assert e is not None
    with pytest.warns(UserWarning, match="fork restore failed"):
        assert s.restore(e, bad) is False
    assert s.state == "unavailable" and "3 consecutive" in (s.reason or "")
    assert s.longest_prefix(IDS[:401]) is None


def test_a_memory_or_io_failure_keeps_the_file(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(8, IDS[:100]))
    e = s.longest_prefix(IDS[:200])
    assert e is not None
    for exc in (MemoryError(), RuntimeError("no stream"), OSError("io")):

        def load(path: Path, header: Header, exc=exc) -> None:
            raise exc

        with pytest.warns(UserWarning, match="fork restore failed"):
            assert s.restore(e, load) is False
        assert e.path.exists() and s.state == "active"


def test_a_successful_restore_resets_the_failure_streak(tmp_path: Path):
    s = store(tmp_path)
    for n in (100, 200, 300):
        assert s.persist(IDS[:n], "tools", fake_write(8, IDS[:n]))

    def bad(path: Path, header: Header) -> None:
        raise ForkFileError("bad")

    for n in (100, 200):
        e = s.longest_prefix(IDS[: n + 1])
        assert e is not None
        with pytest.warns(UserWarning):
            s.restore(e, bad)
    e = s.longest_prefix(IDS[:301])
    assert e is not None
    assert s.restore(e, lambda p, h: None) is True
    assert s.persist(IDS[:400], "tools", fake_write(8, IDS[:400]))
    e = s.longest_prefix(IDS[:401])
    assert e is not None
    with pytest.warns(UserWarning):
        s.restore(e, bad)
    assert s.state == "active"


def test_a_file_deleted_under_the_store_is_a_plain_miss(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(8, IDS[:100]))
    e = s.longest_prefix(IDS[:200])
    assert e is not None
    e.path.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert s.restore(e, lambda p, h: None) is False
    assert s.forks == 0 and s.longest_prefix(IDS[:200]) is None and s.state == "active"


def test_a_file_being_restored_is_never_evicted(tmp_path: Path):
    a, b = IDS[:10], [*IDS[:9], 1001]
    s = store(tmp_path, budget=10 * GIB)
    assert s.persist(a, "tools", fake_write(300, a))
    e = s.longest_prefix(IDS[:11])
    assert e is not None
    os.utime(e.path, (1, 1))
    s.budget = e.size + e.size // 2  # room for one file

    def load(path: Path, header: Header) -> None:
        # A write that needs room while this file is being read takes it
        # from any other file, never from under the reader — and overshoots
        # rather than refuse when there is nothing else.
        assert s.persist(b, "tools", fake_write(300, b)) is True
        assert path.exists()

    assert s.restore(e, load) is True
    assert s.forks == 2


def test_touch_refreshes_the_mtime_of_a_stored_prefix(tmp_path: Path):
    s = store(tmp_path)
    assert s.persist(IDS[:100], "tools", fake_write(8, IDS[:100]))
    e = s.longest_prefix(IDS[:200])
    assert e is not None
    os.utime(e.path, (1, 1))
    s.touch(IDS[:100])
    assert e.path.stat().st_mtime > 1
    s.touch(IDS[:50])  # unknown: a no-op


def test_nested_restore_of_same_file_refcounts_eviction_protection(tmp_path: Path):
    """Concurrent restores of the same file must both complete; the first
    to finish does not remove the second's eviction protection."""
    a, b = IDS[:10], [*IDS[:9], 1001]
    s = store(tmp_path, budget=10 * GIB)
    assert s.persist(a, "tools", fake_write(300, a))
    e = s.longest_prefix(IDS[:11])
    assert e is not None
    os.utime(e.path, (1, 1))
    s.budget = e.size + e.size // 2  # room for one file

    def nested_restore(path: Path, header: Header) -> None:
        # While the outer restore is in flight, do another restore of the
        # same file, then persist to seek room. The inner restore should not
        # shield the file from eviction (it uses the same path), but the
        # outer restore is still in progress, so the refcount must be >0.
        inner_saw_file = None

        def inner_load(inner_path: Path, inner_header: Header) -> None:
            nonlocal inner_saw_file
            inner_saw_file = inner_path.exists()
            # A write that needs room while two restores are in flight: the
            # file under both readers has a refcount of 2, so it survives.
            assert s.persist(b, "tools", fake_write(300, b)) is True
            assert path.exists()

        inner_restore_ok = s.restore(e, inner_load)
        assert inner_restore_ok is True and inner_saw_file is True

    assert s.restore(e, nested_restore) is True
    assert s.forks == 2


def test_oversize_and_free_space_warnings_are_separate(tmp_path: Path):
    """A fork too large for the budget and one that would breach free space
    are different operator signals; one must not silence the other."""
    # Start with a tiny budget; an oversized fork is refused and warns.
    s = store(tmp_path, budget=100)
    with pytest.warns(UserWarning, match="larger than the budget"):
        assert s.persist(IDS[:10], "tools", fake_write(300, IDS[:10])) is False
    # Now set disk free below floor with a normal-sized fork; free-space
    # warning should fire independently.
    s.budget = 1000
    s_low_disk = ForkStore(
        tmp_path / "forks",
        KEY,
        1000,
        disk_usage=Disk(100 * GIB, 9 * GIB + 100),  # floor is 10 GiB
        pid_alive=lambda p: False,
        pid=lambda: 4242,
    )
    with pytest.warns(UserWarning, match="free space"):
        assert s_low_disk.persist(IDS[:100], "tools", fake_write(200, IDS[:100])) is False
