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
import dataclasses
import errno as errno_mod
import functools
import hashlib
import json
import logging
import os
import shutil
import struct
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class Entry:
    """One file in the index: enough to pick it and to charge it."""

    path: Path
    key: str
    n: int
    digest: str
    size: int
    mtime: float


_DISABLING_ERRNOS = frozenset(
    {errno_mod.ENOSPC, errno_mod.EDQUOT, errno_mod.EACCES, errno_mod.EROFS}
)


class ForkStore:
    """One directory per identity key under `root`; the directory is the
    index. Files are shared across every owner thread — a file has no
    thread affinity; the arrays a restore allocates belong to the restoring
    thread — while residency in PrefixCache stays owner-scoped.

    Every method is safe to call from any thread and holds `_lock` only
    across dict work: never across a write, a read or a status build. A
    persist or a restore runs on the caller's thread (the owner's, by the
    caller's discipline) and reports through warnings, never exceptions."""

    def __init__(
        self,
        root: Path,
        key_fields: Mapping[str, str],
        budget: int | None,
        *,
        pid_alive: Callable[[int], bool] | None = None,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        pid: Callable[[], int] = os.getpid,
    ):
        self.root = root
        self.key_fields = dict(key_fields)
        self.key = key_name(self.key_fields)
        self.dir = root / self.key
        self._disk_usage = disk_usage
        self._pid = pid
        self._pid_alive = pid_alive or _pid_alive
        self._lock = threading.Lock()
        # Every key's files, for the budget; only this key's are candidates.
        self._entries: dict[Path, Entry] = {}
        self._writing: set[str] = set()
        self._restoring: set[Path] = set()
        self._failures = 0
        self._floor_warned = False
        # Writer failures already reported, by exception type: a cache the
        # format cannot cover fails identically on every cold turn.
        self._warned_writers: set[str] = set()
        self.state = "active"
        self.reason: str | None = None
        self.forks = 0
        self.bytes = 0
        self.evictions = 0
        try:
            self._ensure_dirs()
            usage = disk_usage(root)
            self.budget = auto_budget(usage.free) if budget is None else budget
            self._scan()
            with self._lock:
                self._evict_locked(0)
                self._recount_locked()
        except OSError as e:
            self.budget = 0
            self._disable(f"{e.strerror or type(e).__name__} at {root}")
            return
        _logger.info(
            f"prompt-cache forks on disk: {root} budget={self.budget / GIB:.1f}GiB "
            f"found={self.forks} ({self.bytes / GIB:.1f} GiB)"
        )

    # ---- construction ------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.dir.mkdir(mode=0o700, exist_ok=True)
        marker = self.root / ".metadata_never_index"
        if not marker.exists():
            marker.touch()
        key_json = self.dir / "key.json"
        if not key_json.exists():
            key_json.write_text(json.dumps(self.key_fields, sort_keys=True, indent=2))

    def _scan(self) -> None:
        """Sizes and mtimes across every key (for the budget), headers for
        this key only (for lookup), dead writers' temp files unlinked, and
        this key's files that fail the header check deleted."""
        entries: dict[Path, Entry] = {}
        for key_dir in self.root.iterdir():
            if not key_dir.is_dir():
                continue
            for item in key_dir.iterdir():
                pid = parse_temp_name(item.name)
                if pid is not None:
                    if not self._pid_alive(pid):
                        item.unlink(missing_ok=True)
                    continue
                parsed = parse_file_name(item.name)
                if parsed is None:
                    continue
                n, dig = parsed
                st = item.stat()
                if key_dir.name == self.key and not self._verify_header(item, n, st.st_size):
                    item.unlink(missing_ok=True)
                    continue
                entries[item] = Entry(item, key_dir.name, n, dig, st.st_size, st.st_mtime)
        with self._lock:
            self._entries = entries

    def _verify_header(self, path: Path, n: int, size: int) -> bool:
        try:
            header = read_header(path)
        except ForkFileError:
            return False
        meta = header.metadata
        if meta.get("format") != FORMAT or meta.get("layout") != str(FORK_LAYOUT):
            return False
        if meta.get("n_tokens") != str(n) or header.expected_size != size:
            return False
        return all(meta.get(k) == v for k, v in self.key_fields.items())

    # ---- bookkeeping (lock held by the caller) -------------------------------

    def _recount_locked(self) -> None:
        self.forks = sum(1 for e in self._entries.values() if e.key == self.key)
        self.bytes = sum(e.size for e in self._entries.values())

    def _evict_locked(self, incoming: int) -> None:
        """LRU by mtime across every key until `incoming` more bytes fit
        under EVICT_TO of the budget, never a file being restored."""
        target = int(self.budget * EVICT_TO) if incoming else self.budget
        while sum(e.size for e in self._entries.values()) + incoming > target:
            victims = [e for e in self._entries.values() if e.path not in self._restoring]
            if not victims:
                return
            oldest = min(victims, key=lambda e: e.mtime)
            oldest.path.unlink(missing_ok=True)
            del self._entries[oldest.path]
            self.evictions += 1

    def _disable(self, reason: str) -> None:
        self.state, self.reason = "unavailable", reason
        warnings.warn(f"sous fork store: disabled ({reason})", stacklevel=3)

    # ---- public ------------------------------------------------------------

    def status(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "forks": self.forks,
            "bytes": self.bytes,
            "budget_bytes": self.budget,
            "evictions": self.evictions,
        }

    def _path_for(self, n: int, dig: str) -> Path:
        return self.dir / file_name(n, dig)

    def has(self, ids: Sequence[int]) -> bool:
        with self._lock:
            return self._path_for(len(ids), digest(ids)) in self._entries

    def longest_prefix(self, stable_ids: Sequence[int]) -> Entry | None:
        """The longest stored prefix `stable_ids` strictly extends, by the
        reuse_length rule (an exact match leaves nothing to decode). One
        digest per distinct stored length, longest first, stop at the first
        hit: under 8 ms for eight lengths."""
        if self.state != "active":
            return None
        with self._lock:
            mine = [e for e in self._entries.values() if e.key == self.key]
        by_length: dict[int, list[Entry]] = {}
        for e in mine:
            if 0 < e.n < len(stable_ids):
                by_length.setdefault(e.n, []).append(e)
        for n in sorted(by_length, reverse=True):
            dig = digest(stable_ids[:n])
            for e in by_length[n]:
                if e.digest == dig:
                    return e
        return None

    def touch(self, ids: Sequence[int]) -> None:
        path = self._path_for(len(ids), digest(ids))
        with self._lock:
            entry = self._entries.get(path)
        if entry is None:
            return
        try:
            os.utime(path, None)
            with self._lock:
                self._entries[path] = dataclasses.replace(entry, mtime=time.time())
        except OSError:
            pass

    def persist(
        self,
        ids: Sequence[int],
        boundary: str,
        write: Callable[[Path, dict[str, str]], None],
    ) -> bool:
        """Write `ids`' cache through `write(temp_path, metadata)` unless a
        file for it exists (then touch it), a write of it is in flight, the
        store is off, or the write would breach the free-space floor. Room
        is made first — never from a file being restored — and the file is
        indexed only after os.replace. Never raises."""
        if self.state != "active":
            return False
        n, dig = len(ids), digest(ids)
        final = self._path_for(n, dig)
        with self._lock:
            if final in self._entries:
                known = True
            elif dig in self._writing:
                return False
            else:
                known = False
                self._writing.add(dig)
        if known:
            self.touch(ids)
            return False
        temp = temp_name(final, self._pid())
        from importlib.metadata import version

        try:
            self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = {
                "format": FORMAT,
                "layout": str(FORK_LAYOUT),
                **self.key_fields,
                "n_tokens": str(n),
                "boundary": boundary,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sous": version("sous-mcp"),
            }
            write(temp, metadata)
            os.chmod(temp, 0o600)
            size = temp.stat().st_size
            if size > self.budget:
                temp.unlink(missing_ok=True)
                if not self._floor_warned:
                    self._floor_warned = True
                    warnings.warn(
                        f"sous fork store: not persisting a {size / GIB:.1f} GiB fork "
                        f"larger than the budget ({self.budget / GIB:.1f} GiB)",
                        stacklevel=2,
                    )
                return False
            usage = self._disk_usage(self.root)
            if usage.free - size < floor_bytes(usage.total):
                temp.unlink(missing_ok=True)
                if not self._floor_warned:
                    self._floor_warned = True
                    warnings.warn(
                        "sous fork store: not persisting a fork — the write would leave "
                        f"less than {floor_bytes(usage.total) / GIB:.0f} GiB of free space",
                        stacklevel=2,
                    )
                return False
            with self._lock:
                self._evict_locked(size)
            os.replace(temp, final)
            st = final.stat()
            with self._lock:
                self._entries[final] = Entry(final, self.key, n, dig, st.st_size, st.st_mtime)
                self._recount_locked()
            return True
        except OSError as e:
            temp.unlink(missing_ok=True)
            warnings.warn(
                f"sous fork store: fork persist failed ({type(e).__name__}: {e.strerror})",
                stacklevel=2,
            )
            if e.errno in _DISABLING_ERRNOS:
                self._disable(f"{os.strerror(e.errno)} ({errno_mod.errorcode[e.errno]})")
            return False
        except Exception as e:  # noqa: BLE001 — an optimization never fails a turn
            temp.unlink(missing_ok=True)
            kind = type(e).__name__
            if kind not in self._warned_writers:
                self._warned_writers.add(kind)
                warnings.warn(
                    f"sous fork store: fork persist failed ({kind}); not persisting again",
                    stacklevel=2,
                )
            return False
        finally:
            with self._lock:
                self._writing.discard(dig)

    def restore(self, entry: Entry, load: Callable[[Path, Header], None]) -> bool:
        """Read `entry` through `load(path, header)`. A ForkFileError from
        the header check or the loader is a verification failure: the file
        is deleted and three in a row retire the store. Anything else keeps
        the file (an allocation failure is not corruption). A file that is
        gone is a plain miss. Never raises."""
        if self.state != "active":
            return False
        with self._lock:
            if entry.path not in self._entries:
                return False
            self._restoring.add(entry.path)
        try:
            try:
                st = entry.path.stat()
            except FileNotFoundError:
                self._forget(entry)
                return False
            try:
                header = read_header(entry.path)
                if header.expected_size != st.st_size:
                    raise ForkFileError(f"{entry.path.name}: size {st.st_size}")
                load(entry.path, header)
            except ForkFileError as e:
                warnings.warn(f"sous fork store: fork restore failed ({e}); deleted", stacklevel=2)
                entry.path.unlink(missing_ok=True)
                self._forget(entry)
                self._failures += 1
                if self._failures >= MAX_FAILURES:
                    self._disable(f"{MAX_FAILURES} consecutive restore failures")
                return False
            except Exception as e:  # noqa: BLE001 — keep the file: not corruption
                warnings.warn(
                    f"sous fork store: fork restore failed ({type(e).__name__}); kept",
                    stacklevel=2,
                )
                return False
            self._failures = 0
            self.touch_entry(entry)
            return True
        finally:
            with self._lock:
                self._restoring.discard(entry.path)

    def touch_entry(self, entry: Entry) -> None:
        try:
            os.utime(entry.path, None)
            with self._lock:
                if entry.path in self._entries:
                    self._entries[entry.path] = dataclasses.replace(entry, mtime=time.time())
        except OSError:
            pass

    def _forget(self, entry: Entry) -> None:
        with self._lock:
            self._entries.pop(entry.path, None)
            self._recount_locked()


def _pid_alive(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)
