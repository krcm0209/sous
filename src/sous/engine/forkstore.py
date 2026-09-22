"""Fork slots on disk: the rules, the index and the budget.

This module imports no mlx, deliberately, like promptcache: every decision —
which file serves a render, what the identity key contains, what gets
evicted, when a write is refused — is made here, where a fake writer and a
fake loader can exercise it in CI. The array work (writing a cache's
buffers, reading them back) lives in forkio and reaches this module only as
callables.
"""

from __future__ import annotations

import array
import functools
import hashlib
import json
import logging
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

_logger = logging.getLogger("sous.engine")

GIB = 1 << 30
# Bumped only when the on-disk layout changes; a change to the numerics is
# caught by engine_epoch, which hashes the sources that compute the KV.
FORK_LAYOUT = 1
FORMAT = "sous-fork-v1"
AUTO_BUDGET_CAP = 16 * GIB
AUTO_BUDGET_FRACTION = 0.25
FLOOR_MIN = 10 * GIB
FLOOR_FRACTION = 0.05
# Eviction runs down to this share of the budget, so a write near the cap
# does not sweep on every turn.
EVICT_TO = 0.9
# A 57K-token file's header measured 12.8 KB; anything past this is not ours.
MAX_HEADER_BYTES = 1 << 20
# Consecutive verification failures that retire the store for the daemon's
# life: a store whose files all fail would otherwise delete itself one cold
# turn at a time.
MAX_FAILURES = 3

# The modules whose bytes decide what KV a token id produces; hashed into
# the identity key so a change to them can never be forgotten.
_ENGINE_SOURCES = files("sous.engine")
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py")


class ForkFileError(Exception):
    """A file that failed verification: format, size, layer layout, offsets
    or ids. The store deletes it. Anything else raised around a file — an
    allocation failure, an OSError — keeps the file."""


def digest(ids: Sequence[int]) -> str:
    """blake2b-16 over the ids as little-endian int32: the index key for a
    prefix, and the file name's second half."""
    return hashlib.blake2b(array.array("i", ids).tobytes(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Header:
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]
    # 8 + header length + the last tensor's end: what st_size must equal.
    expected_size: int


def read_header(path: Path) -> Header:
    """The safetensors header, parsed in pure Python: an 8-byte little-endian
    length and a JSON object. mx.load would do the same and hand back lazy
    arrays bound to the calling thread's streams; this touches nothing."""
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ForkFileError(f"{path.name}: short header length")
            (n,) = struct.unpack("<Q", raw)
            if n > MAX_HEADER_BYTES:
                raise ForkFileError(f"{path.name}: header of {n} bytes is not ours")
            body = f.read(n)
        if len(body) != n:
            raise ForkFileError(f"{path.name}: truncated header")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ForkFileError(f"{path.name}: header is not an object")
        metadata = parsed.pop("__metadata__", {}) or {}
        tensors: dict[str, TensorInfo] = {}
        end = 0
        for name, entry in parsed.items():
            start, stop = entry["data_offsets"]
            tensors[name] = TensorInfo(str(entry["dtype"]), tuple(entry["shape"]), start, stop)
            end = max(end, stop)
        expected = 8 + n + end
        actual = path.stat().st_size
        if actual != expected:
            # The truncation guard: a crash mid-write, or a copy that lost
            # its tail, cannot pass as a whole file.
            raise ForkFileError(f"{path.name}: size {actual} != {expected}")
        return Header({str(k): str(v) for k, v in metadata.items()}, tensors, expected)
    except ForkFileError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ForkFileError(f"{path.name}: unreadable header ({type(e).__name__})") from e


def key_name(fields: Mapping[str, str]) -> str:
    """The identity key's directory name: blake2b-16 over the canonical JSON
    of its fields. The fields themselves go into key.json and every file's
    metadata, for forensics."""
    canonical = json.dumps(dict(sorted(fields.items())), separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


@functools.cache
def engine_epoch() -> str:
    """A hash of the engine sources that decide ids -> KV, read once.
    A formatter pass costs one cold start; a semantic change can never be
    missed, which a hand-bumped constant cannot promise."""
    h = hashlib.blake2b(digest_size=8)
    for name in _EPOCH_FILES:
        h.update(name.encode())
        h.update(_ENGINE_SOURCES.joinpath(name).read_bytes())
    # The .h files are compiled into both kernels (the nibble-decode and
    # K-order contract lives there), so they are part of the numerics too.
    kernels = _ENGINE_SOURCES.joinpath("kernels")
    for entry in sorted(kernels.iterdir(), key=lambda e: e.name):
        if entry.name.endswith((".metal", ".h")):
            h.update(entry.name.encode())
            h.update(entry.read_bytes())
    return h.hexdigest()


def weights_identity(config_path: Path) -> str:
    """What identifies the weights behind a config.json: the Hub snapshot's
    commit sha when the file sits under `snapshots/<sha>/` (never resolve the
    path — the file is a symlink into blobs/), else a hash of config.json's
    bytes and every sibling *.safetensors' name, size and mtime — a
    re-quantisation into the same directory leaves config.json unchanged."""
    directory = config_path.parent
    if directory.parent.name == "snapshots":
        return directory.name
    h = hashlib.blake2b(digest_size=16)
    h.update(config_path.read_bytes())
    for weights in sorted(directory.glob("*.safetensors")):
        st = weights.stat()
        h.update(f"{weights.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def file_name(n: int, dig: str) -> str:
    return f"{n}-{dig}.safetensors"


def parse_file_name(name: str) -> tuple[int, str] | None:
    """(n_tokens, digest) from a canonical name, else None — a temp file, a
    key.json, anything not ours."""
    stem, dot, suffix = name.rpartition(".")
    if not dot or suffix != "safetensors":
        return None
    n, dash, dig = stem.partition("-")
    if not dash or not n.isdecimal() or len(dig) != 32:
        return None
    try:
        int(dig, 16)
    except ValueError:
        return None
    return int(n), dig


def temp_name(final: Path, pid: int) -> Path:
    """`<final>.<pid>.tmp.safetensors`: mx.save_safetensors appends the
    suffix to a name that lacks it, so the temp name carries it already; the
    pid lets a startup sweep tell a crashed writer's leftover from a live
    one's."""
    stem = final.name[: -len(".safetensors")]
    return final.with_name(f"{stem}.{pid}.tmp.safetensors")


def parse_temp_name(name: str) -> int | None:
    if not name.endswith(".tmp.safetensors"):
        return None
    pid = name[: -len(".tmp.safetensors")].rpartition(".")[2]
    return int(pid) if pid.isdecimal() else None


def auto_budget(free: int) -> int:
    """`"auto"`: a quarter of the volume's free bytes, capped at 16 GiB —
    four to five tools forks of the default model."""
    return min(AUTO_BUDGET_CAP, int(free * AUTO_BUDGET_FRACTION))


def floor_bytes(total: int) -> int:
    """Free space a write must leave: 10 GiB or 5 % of the volume."""
    return max(FLOOR_MIN, int(total * FLOOR_FRACTION))
