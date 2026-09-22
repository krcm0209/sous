"""The disk store's rules, exercised without mlx: digest, header parse,
identity key, naming and budgets."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from sous.engine import forkstore
from sous.engine.forkstore import (
    FORMAT,
    GIB,
    ForkFileError,
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
