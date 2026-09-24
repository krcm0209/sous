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
import contextlib
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
# Consecutive verification failures that retire the store until the next
# model load (the store is built per load): a store whose files all fail
# would otherwise delete itself one cold turn at a time.
MAX_FAILURES = 3

# The modules whose bytes decide what KV a token id produces; hashed into
# the identity key so a change to them can never be forgotten.
_ENGINE_SOURCES = files("sous.engine")
_EPOCH_FILES = ("vlm.py", "lm.py", "int8prefill.py")


class ForkFileError(Exception):
    """A file that failed verification: format, size, layer layout, offsets
    or ids. The store drops it from the index and deletes it when it can.
    Anything else raised around a file — an
    allocation failure, an OSError (a file that cannot be opened or read
    right now says nothing about its bytes) — keeps the file."""


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
    arrays bound to the calling thread's streams; this touches nothing.
    Bytes that are not a whole fork file raise ForkFileError; an OSError
    (permissions, no free descriptor, the file gone) propagates as itself,
    so no caller deletes a good file it merely could not open."""
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
        if not isinstance(metadata, dict):
            raise ForkFileError(f"{path.name}: metadata is not an object")
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
    except (ValueError, KeyError, TypeError) as e:
        raise ForkFileError(f"{path.name}: unreadable header ({type(e).__name__})") from e


def key_name(fields: Mapping[str, str]) -> str:
    """The identity key's directory name: blake2b-16 over the canonical JSON
    of its fields. The fields themselves go into key.json and every file's
    metadata, for forensics."""
    canonical = json.dumps(dict(sorted(fields.items())), separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


_NUMERIC_ENV = ("MLX_ENABLE_TF32", "MLX_SDPA_BLOCKS")


def fork_key_fields(
    *,
    backend: str,
    backend_version: str,
    weights: str,
    gpu: str,
    positions: str,
    int8_status: Mapping[str, Any],
) -> dict[str, str]:
    """Everything that changes the KV arrays behind identical token ids, and
    nothing else: the backend and its package, mlx, the GPU, the weights,
    the engine sources (epoch) and the file layout, which side supplies the
    rotary positions, whether int8 prefill actually ran (`off` and
    `unavailable` are the same numerics), and the two mlx-core environment
    switches that change matmul precision and attention accumulation. The
    template, tokenizer, sampling, drafter and window are deliberately
    absent: ids are compared exactly, and none of them touches a prefill."""
    from importlib.metadata import version

    fields = {
        "backend": backend,
        "backend_version": backend_version,
        "mlx": version("mlx"),
        "gpu": gpu,
        "weights": weights,
        "epoch": engine_epoch(),
        "layout": str(FORK_LAYOUT),
        "positions": positions,
        "int8": (
            f"active:{int8_status.get('routed', 0)}"
            if int8_status.get("state") == "active"
            else "off"
        ),
    }
    env = ";".join(f"{k}={os.environ[k]}" for k in _NUMERIC_ENV if k in os.environ)
    if env:
        fields["env"] = env
    return fields


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


@functools.cache
def _sous_version() -> str:
    from importlib.metadata import version

    return version("sous-mcp")


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


def _errno_name(e: OSError) -> str:
    return errno_mod.errorcode.get(e.errno or 0, type(e).__name__)


class ForkStore:
    """One directory per identity key under `root`; the directory is the
    index. Files are shared across every owner thread — a file has no
    thread affinity; the arrays a restore allocates belong to the restoring
    thread — while residency in PrefixCache stays owner-scoped.

    Every method is safe to call from any thread and holds `_lock` only
    across dict work: never across a write, a read, a delete or a status
    build. A persist or a restore runs on the caller's thread (the owner's,
    by the caller's discipline) and reports through warnings, never
    exceptions."""

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
        self._restoring: dict[Path, int] = {}
        # Files out of the index but not yet unlinked (evicted, or failed
        # verification): a write of the same ids in that window would be
        # deleted by the unlink after it.
        self._unlinking: set[Path] = set()
        # Files whose unlink failed: back in the index, and not picked again
        # while they stay there; leaving the index drops the exemption.
        self._undeletable: set[Path] = set()
        self._failures = 0
        # Consecutive persist failures short of a disabling errno: a cache the
        # format cannot cover, or a disk that fails the same way, fails
        # identically on every cold turn, and this is what retires the store
        # rather than letting it try forever.
        self._write_failures = 0
        self._oversize_warned = False
        self._floor_warned = False
        self._unlink_warned = False
        # Failures already reported, by exception type or errno name: a
        # writer's, and a restore's that kept its file. The store's own sets,
        # not the warnings registry, which a filter reset empties.
        self._warned_writers: set[str] = set()
        self._warned_restores: set[str] = set()
        self.state = "active"
        self.reason: str | None = None
        self.forks = 0
        self.bytes = 0
        self.evictions = 0
        try:
            self._ensure_dirs()
            self._scan()
            usage = disk_usage(root)
            with self._lock:
                found = sum(e.size for e in self._entries.values())
            # `usage.free` already excludes this store's own files (they are
            # real bytes on disk by the time this reads it); add them back so
            # a restart derives the same budget the first run did, rather
            # than a smaller one that evicts what the previous run wrote.
            self.budget = auto_budget(usage.free + found) if budget is None else budget
            with self._lock:
                victims = self._evict_locked(self.budget)
                self._recount_locked()
            self._unlink(victims)
        except Exception as e:  # noqa: BLE001 — a model load must never fail because of this
            self.budget = 0
            reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
            self._disable(f"{reason or type(e).__name__} at {root}")
            return
        # Eviction at construction is the one that can sweep every key's files
        # at once (a lowered budget), so the load line says so.
        evicted = f" evicted={self.evictions}" if self.evictions else ""
        _logger.info(
            f"prompt-cache forks on disk: {root} budget={self.budget / GIB:.1f}GiB "
            f"found={self.forks} ({self.bytes / GIB:.1f} GiB){evicted}"
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
        this key's files that fail the header check deleted — both where
        they can be, since neither is ever indexed.

        Tolerant per item, so one bad file never disables the whole store: a
        pid_alive check that cannot answer (a crafted or unreadable pid)
        leaves the temp file alone rather than guessing whether to delete
        someone else's in-flight write; a file stat()-ed out from under the
        scan (removed mid-scan, racing another thread's eviction), or whose
        header cannot be read right now, is skipped; a key directory that
        vanishes is skipped; a delete that fails leaves the file for the
        next load. Anything else — a header whose JSON recurses past Python's
        stack, an unreadable key directory — still propagates, since it
        means the scan itself cannot be trusted; the constructor disables
        the store for that."""
        entries: dict[Path, Entry] = {}
        for key_dir in self.root.iterdir():
            if not key_dir.is_dir():
                continue
            try:
                items = list(key_dir.iterdir())
            except FileNotFoundError, NotADirectoryError:
                # Removed since the root was listed (an rm -rf, which the
                # write path survives through _ensure_dirs): as gone as a
                # vanished file.
                continue
            for item in items:
                pid = parse_temp_name(item.name)
                if pid is not None:
                    try:
                        alive = self._pid_alive(pid)
                    except Exception:  # noqa: BLE001 — can't tell; leave it
                        continue
                    if not alive:
                        with contextlib.suppress(OSError):
                            item.unlink(missing_ok=True)
                    continue
                parsed = parse_file_name(item.name)
                if parsed is None:
                    continue
                n, dig = parsed
                try:
                    st = item.stat()
                except OSError:
                    continue
                size = st.st_size
                if key_dir.name == self.key:
                    try:
                        header = self._verify_header(item, n)
                    except OSError:
                        continue  # unreadable now, not bad bytes: next load looks again
                    if header is None:
                        # Never indexed either way; a delete that fails
                        # leaves it for the next load to try again.
                        with contextlib.suppress(OSError):
                            item.unlink(missing_ok=True)
                        continue
                    # read_header checked the size against a stat of its own;
                    # the one above may predate a same-digest replacement.
                    size = header.expected_size
                entries[item] = Entry(item, key_dir.name, n, dig, size, st.st_mtime)
        with self._lock:
            self._entries = entries

    def _verify_header(self, path: Path, n: int) -> Header | None:
        """The header when `path` is a fork of this key for `n` ids, None
        when its bytes say otherwise; an OSError reading it propagates."""
        try:
            header = read_header(path)
        except ForkFileError:
            return None
        meta = header.metadata
        if meta.get("format") != FORMAT or meta.get("layout") != str(FORK_LAYOUT):
            return None
        if meta.get("n_tokens") != str(n):
            return None
        if not all(meta.get(k) == v for k, v in self.key_fields.items()):
            return None
        return header

    # ---- bookkeeping (lock held by the caller) -------------------------------

    def _recount_locked(self) -> None:
        # forks counts only this identity key's files; bytes sums every
        # key's, because the budget (and the LRU that enforces it) spans
        # every key directory under root, not just the current model's.
        self.forks = sum(1 for e in self._entries.values() if e.key == self.key)
        self.bytes = sum(e.size for e in self._entries.values())

    def _evict_locked(self, target: int, keep: Path | None = None) -> list[Entry]:
        """LRU by mtime across every key until the index fits under `target`
        bytes, never a file being restored, one that would not delete, nor
        `keep`. The victims leave the index here; the caller unlinks them
        with `_unlink` once it has released the lock."""
        victims: list[Entry] = []
        total = sum(e.size for e in self._entries.values())
        candidates = sorted(
            (
                e
                for e in self._entries.values()
                if e.path != keep
                and e.path not in self._undeletable
                and self._restoring.get(e.path, 0) == 0
            ),
            key=lambda e: e.mtime,
        )
        for oldest in candidates:
            if total <= target:
                break
            del self._entries[oldest.path]
            self._unlinking.add(oldest.path)
            total -= oldest.size
            victims.append(oldest)
        return victims

    def _unlink(self, victims: list[Entry]) -> None:
        """Delete what `_evict_locked` took out of the index, outside the
        lock. A file that will not go goes back into the index, so the
        budget still counts its bytes, and is not picked again while it stays
        indexed: the next
        eviction takes the next oldest instead of counting it as freed."""
        if not victims:
            return
        failed: list[tuple[Entry, OSError]] = []
        for victim in victims:
            try:
                victim.path.unlink(missing_ok=True)
            except OSError as e:
                failed.append((victim, e))
        with self._lock:
            for victim in victims:
                self._unlinking.discard(victim.path)
            for victim, _ in failed:
                self._entries.setdefault(victim.path, victim)
                self._undeletable.add(victim.path)
            self.evictions += len(victims) - len(failed)
            self._recount_locked()
        if failed and not self._unlink_warned:
            self._unlink_warned = True
            warnings.warn(
                f"sous fork store: could not delete an evicted fork ({_errno_name(failed[0][1])})"
                "; kept",
                stacklevel=3,
            )

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

    def touch(self, ids: Sequence[int]) -> bool:
        """Refresh `ids`' file's mtime; whether the store still holds it. One
        digest for the lookup and the touch together."""
        return self._touch(self._path_for(len(ids), digest(ids)))

    def _touch(self, path: Path) -> bool:
        with self._lock:
            entry = self._entries.get(path)
        return entry is not None and self.touch_entry(entry)

    def persist(
        self,
        ids: Sequence[int],
        boundary: str,
        write: Callable[[Path, dict[str, str]], None],
        *,
        expected_bytes: int,
    ) -> bool:
        """Write `ids`' cache through `write(temp_path, metadata)` unless a
        file for it exists (then touch it), a write of it is in flight, the
        store is off, the file is still being deleted, or the caller's own
        `expected_bytes` estimate would breach the budget or the free-space
        floor — checked before `write` ever runs, so a refused write never
        touches disk. The file is indexed only after os.replace. The real size
        is checked against the budget once more after the write, belt and
        braces against an under-estimate; the floor is not re-checked there,
        since the pre-check already reserved it. Room is made only once the
        file is in place, by its real size — never from a file being restored
        — so a write that fails or is refused has cost the store nothing; the
        budget is exceeded by that one file for the moment, the floor never.
        Never raises."""
        if self.state != "active":
            return False
        n, dig = len(ids), digest(ids)
        final = self._path_for(n, dig)
        if self._touch(final):
            return False
        with self._lock:
            if final in self._entries or dig in self._writing or final in self._unlinking:
                return False
            self._writing.add(dig)
        temp: Path | None = None
        try:
            if expected_bytes > self.budget:
                if not self._oversize_warned:
                    self._oversize_warned = True
                    warnings.warn(
                        f"sous fork store: not persisting a {expected_bytes / GIB:.1f} GiB "
                        f"fork larger than the budget ({self.budget / GIB:.1f} GiB)",
                        stacklevel=2,
                    )
                return False
            # rm -rf ~/.sous/forks under a running daemon must not wedge the
            # next write: restore the promised layout before touching disk.
            self._ensure_dirs()
            usage = self._disk_usage(self.root)
            if usage.free - expected_bytes < floor_bytes(usage.total):
                if not self._floor_warned:
                    self._floor_warned = True
                    warnings.warn(
                        "sous fork store: not persisting a fork — the write would leave "
                        f"less than {floor_bytes(usage.total) / GIB:.0f} GiB of free space",
                        stacklevel=2,
                    )
                return False
            temp = temp_name(final, self._pid())
            metadata = {
                "format": FORMAT,
                "layout": str(FORK_LAYOUT),
                **self.key_fields,
                "n_tokens": str(n),
                "boundary": boundary,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sous": _sous_version(),
            }
            write(temp, metadata)
            os.chmod(temp, 0o600)
            size = temp.stat().st_size
            if size > self.budget:
                temp.unlink(missing_ok=True)
                if not self._oversize_warned:
                    self._oversize_warned = True
                    warnings.warn(
                        f"sous fork store: not persisting a {size / GIB:.1f} GiB fork "
                        f"larger than the budget ({self.budget / GIB:.1f} GiB)",
                        stacklevel=2,
                    )
                return False
            os.replace(temp, final)
            st = final.stat()
            with self._lock:
                self._entries[final] = Entry(final, self.key, n, dig, st.st_size, st.st_mtime)
                victims = self._evict_locked(int(self.budget * EVICT_TO), keep=final)
                self._recount_locked()
            self._write_failures = 0
            self._unlink(victims)
            return True
        except Exception as e:  # noqa: BLE001 — an optimization never fails a turn
            if temp is not None:
                temp.unlink(missing_ok=True)
            kind = detail = type(e).__name__
            if isinstance(e, OSError):
                kind = _errno_name(e)
                detail = f"{kind}: {e.strerror}" if e.strerror else kind
                if e.errno in _DISABLING_ERRNOS:
                    warnings.warn(f"sous fork store: fork persist failed ({detail})", stacklevel=2)
                    self._disable(f"{os.strerror(e.errno)} ({kind})")
                    return False
            self._write_failures += 1
            if self._write_failures >= MAX_FAILURES:
                self._disable(f"{MAX_FAILURES} consecutive persist failures ({detail})")
            elif kind not in self._warned_writers:
                self._warned_writers.add(kind)
                warnings.warn(
                    f"sous fork store: fork persist failed ({detail}); will retry on "
                    "the next cold turn",
                    stacklevel=2,
                )
            return False
        finally:
            with self._lock:
                self._writing.discard(dig)

    def restore(self, entry: Entry, load: Callable[[Path, Header], None]) -> bool:
        """Read `entry` through `load(path, header)`. A ForkFileError from
        the header check or the loader is a verification failure: the file
        leaves the index, is deleted when it can be, and three in a row
        retire the store. Anything else keeps the file — an allocation
        failure, or a file that cannot be opened right now, is not
        corruption — and warns once per kind of failure. A file already gone
        when the header is read is a plain miss; one removed after that
        reaches the loader as a failure to open and is kept until the next
        touch finds it gone. Never raises."""
        if self.state != "active":
            return False
        with self._lock:
            if entry.path not in self._entries:
                return False
            self._restoring[entry.path] = self._restoring.get(entry.path, 0) + 1
        try:
            try:
                header = read_header(entry.path)
                load(entry.path, header)
            except FileNotFoundError:
                self._forget(entry)
                return False
            except ForkFileError as e:
                failed = self._delete(entry)
                outcome = f"could not delete it ({_errno_name(failed)})" if failed else "deleted"
                warnings.warn(
                    f"sous fork store: fork restore failed ({e}); {outcome}", stacklevel=2
                )
                self._failures += 1
                if self._failures >= MAX_FAILURES:
                    self._disable(f"{MAX_FAILURES} consecutive restore failures")
                return False
            except Exception as e:  # noqa: BLE001 — keep the file: not corruption
                kind = type(e).__name__
                if kind not in self._warned_restores:
                    self._warned_restores.add(kind)
                    warnings.warn(
                        f"sous fork store: fork restore failed ({kind}); kept", stacklevel=2
                    )
                return False
            self._failures = 0
            self.touch_entry(entry)
            return True
        finally:
            with self._lock:
                count = self._restoring.get(entry.path, 0)
                if count > 1:
                    self._restoring[entry.path] = count - 1
                else:
                    self._restoring.pop(entry.path, None)

    def touch_entry(self, entry: Entry) -> bool:
        """Refresh `entry`'s mtime; False when its file is gone (removed under
        a live daemon), which drops it from the index — a stale entry would
        read as stored forever, and the fork would never be written again.
        Any other OSError keeps the entry: the file is there, only the touch
        failed."""
        try:
            os.utime(entry.path, None)
        except FileNotFoundError:
            self._forget(entry)
            return False
        except OSError:
            return True
        with self._lock:
            if entry.path in self._entries:
                self._entries[entry.path] = dataclasses.replace(entry, mtime=time.time())
        return True

    def _forget(self, entry: Entry) -> None:
        with self._lock:
            self._forget_locked(entry)

    def _forget_locked(self, entry: Entry) -> None:
        # A path that leaves the index is a new file if it is ever written
        # again, so an old failure to delete it no longer exempts it.
        self._entries.pop(entry.path, None)
        self._undeletable.discard(entry.path)
        self._recount_locked()

    def _delete(self, entry: Entry) -> OSError | None:
        """Take a file that failed verification out of the index and off
        the disk, guarded like an eviction's victim so a write of the same
        ids cannot land between the two; the error when it would not go,
        and then it stays out of the index — it is not served again."""
        with self._lock:
            self._forget_locked(entry)
            self._unlinking.add(entry.path)
        try:
            entry.path.unlink(missing_ok=True)
        except OSError as e:
            return e
        finally:
            with self._lock:
                self._unlinking.discard(entry.path)
        return None


def _pid_alive(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)
