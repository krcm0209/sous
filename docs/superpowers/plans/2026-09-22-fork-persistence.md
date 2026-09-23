# Fork Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tools fork survives the weights: the live cache is written to disk at the lowest fork boundary, and a new session's first turn after an idle unload, a restart or a reboot restores it in seconds instead of prefilling 113–272 s.

**Architecture:** Two new modules split by what touches mlx. `engine/forkstore.py` is mlx-free and owns every rule — digest, identity key, directory index, budget, LRU, free-space floor, temp/rename discipline, the pure-Python safetensors header parse, in-flight sets, the disable rule — so it is tested in CI with fakes like `promptcache.py`. `engine/forkio.py` touches mlx and owns every array: it writes a cache's full padded buffers to one safetensors file and reads them back by assigning `keys`/`values`/`offset` directly. `PrefixCache` gets an optional store and two new call points: a synchronous persist at the lowest boundary inside `_run` (before the resident copy, outside the prefill timer, with its own handler) and a restore that replaces the miss branch in `generate` (into the turn's own cache; residency comes from the existing charged path once `reuse <= b` is allowed). Both engines build the store after load from an identity key of everything that changes the KV behind identical ids; only `default_engine_factory` hands them a directory, so tests and `sous tune` never reach `~/.sous`.

**Tech Stack:** Python 3.14, `hashlib.blake2b`, `shutil.disk_usage`, `psutil` (already a dependency), `mx.save_safetensors`/`mx.load` (mlx 0.32.2, mlx-vlm 0.7.1, mlx-lm 0.31.3 — all already pinned). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-fork-persistence-design.md` — every section; the plan argues from it. "Decisions this plan makes beyond the spec's text" below lists the six places the plan's mechanism differs from the spec's wording and why.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`; `ty` flags an *unused* pragma, so add one only when `ty check` names the rule.
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local, and only in `engine/forkio.py`, `engine/vlm.py`, `engine/lm.py` and `engine/base.py`. `engine/forkstore.py` and `engine/promptcache.py` import no mlx, ever — that is what keeps their rules testable in CI (the test job runs on `macos-15` with mlx installed, so `tests/test_engine_forkio.py` may use real `mlx_vlm.models.cache` classes on tiny arrays; the model tests stay local-only).
- No new thread anywhere. The persist runs on the owner (session) thread inside `_run`; the restore on the turn's thread inside `generate`; the directory scan on the model-load thread touching no mlx. Every mlx-touching thread already calls `release_mlx_thread_state()` on its way out; this plan adds none.
- Save only evaluated, contiguous buffers (`c.keys`, `c.values`, never the `state` slice); restore `KVCache` by assigning `keys`, `values`, then `offset` from metadata, never through the `state` setter; `mx.eval` restored arrays before returning.
- Every disk operation has its own `try`/`except`/`warnings.warn` and can only warn: a cold turn's `generate` re-raises rather than retrying, so an escaping disk error would fail a viable 170 s prefill.
- The `PrefixCache` bookkeeping lock is held only across list and dict work; no persist, restore, digest scan, header parse or status build runs under it, and `stats()` reads the store's counters as plain ints.
- The store is OFF unless a directory is handed to the engine. The only place that builds one is `default_engine_factory` with `forks=True` (its default) from `config.data_dir / "forks"`; `sous tune` passes `forks=False`. Tests never touch the real `~/.sous`: every path comes from `tmp_path`.
- The API never logs a request body; the store writes token ids and KV to a 0o700 directory as 0o600 files and logs nothing about their content — no token, no text, only counts, bytes and seconds.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` once committed (this plan and its spec included).
- **No plan language in committed code.** Code and test comments must never cite a spec, a plan, a task or a step. Where a code block below carries such a reference, restate the fact. Before the PR: `git diff main -- src tests | grep -nE '^\+.*(Task [0-9]|Step [0-9]|[Dd]ecision [0-9]|[Ss]pec[: ]|docs/superpowers)'` must print nothing.
- Never loosen an existing assertion to make a test pass. Where this plan changes a behaviour an existing test pinned (named per task), restate the expected value and the docstring.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>` — model-less, whatever the harness suggests.
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` (`pyproject.toml` sets `-q`; never add another), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`. Line length 100, `E501` enforced: a long string is written as adjacent literals.
- The suite at `8cb3f34` (main `7873999` plus the spec commit) collects `1204 passed, 8 skipped, 33 deselected` under `-m "not model"` on this Mac (macOS 15.5; the eight skips are `tests/test_int8prefill.py` without tensor units). Each task's "Expected" line gives the delta; on a machine with tensor units `passed` is 8 higher and there are no skips.
- Branch: the worktree's `claude/sous-issues-perf-review-b8ed31` (spec commit `8cb3f34` on `main`); rename to `feat/fork-persistence` at PR time if desired. `main` is protected: PR with all four CI jobs green.
- The maintainer's daemon on the M5 Pro is not restarted by any task. The live gate (Task 12) is the orchestrator's, after the maintainer's explicit go-ahead.

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/engine/forkstore.py` (new) | mlx-free rules: `digest`, `read_header`, `weights_identity`, `engine_epoch`, `key_name`, file naming, `auto_budget`/`floor_bytes`, `ForkFileError`; `ForkStore` — construction scan, budget, LRU, floor, temp sweep, `has`/`longest_prefix`/`persist`/`restore`/`touch`/`status`, in-flight sets, disable rule. |
| `src/sous/engine/forkio.py` (new) | mlx array IO shared by both backends: `layer_kinds`, `eval_cache`, `persist_cache`, `restore_cache`, `ForkUnsupported`. |
| `src/sous/engine/promptcache.py` | `CacheHooks` gains `eval_cache`/`persist`/`restore`; `StoreHooks` protocol; stats gauges and counters; `_fork_boundaries` returns `(b, need_slot, need_file)`; `_persist`; `_restore_from_disk`; `stats()["disk"]`; docstring rule about files vs arrays. |
| `src/sous/engine/vlm.py`, `src/sous/engine/lm.py` | Constructor params `fork_dir`, `fork_budget`, `weights_identity`; `fork_store`; `_fork_key_fields`; the three hooks delegating to `forkio`. |
| `src/sous/engine/base.py` | `weights_identity_for(model_id)`; `_default_factory(..., fork_dir, fork_budget)` with the `reserve_bytes == 0` gate; `default_engine_factory(config, *, forks=True)`. |
| `src/sous/config.py` | `prompt_cache_disk_gb` (auto / GiB / 0), validator, `_KNOWN`, the `prompt_cache = false` warning. |
| `src/sous/tune/suite/runner.py`, `src/sous/tune/bench.py` | `forks=False` at both factory sites. |
| `src/sous/api/turn.py`, `src/sous/api/routes.py` | `TurnResult.from_disk`/`persist_seconds`/`restore_seconds`; `cache=disk`, `persist_s=`, `restore_s=`. |
| `src/sous/server.py`, `src/sous/cli.py`, `src/sous/tui.py` | `config.prompt_cache_disk_gb` in the document; `sous status` disk line; `sous top` wide-line suffix. |
| `tests/test_forkstore.py` (new), `tests/test_engine_forkio.py` (new), `tests/test_promptcache.py`, `tests/test_engine_base.py`, `tests/test_engine_vlm.py`, `tests/test_config.py`, `tests/test_api_turn.py`, `tests/test_api_routes.py`, `tests/test_cli.py`, `tests/test_tui.py`, `tests/test_server.py`, `tests/test_tune_runner.py`, `tests/test_tune_bench.py` | Tests per task. |
| `README.md`, `CLAUDE.md` | Task 10. |

## Decisions this plan makes beyond the spec's text

1. **Truncation guard from the header, not a `file_bytes` metadata field.** The spec records the expected file size in metadata; the size is not known until the file is written. The safetensors header already carries every tensor's data range, so `read_header` computes `expected_size = 8 + header_len + max(end)` and the store compares it with `st_size` at scan and before every restore. Same guard, no chicken-and-egg.
2. **One shared `forkio.py` rather than two copies of the array code.** The spec says "the engines own every array"; both backends' caches are the same two class shapes (`KVCache`, `ArraysCache` from mlx-lm or mlx-vlm, checked by class name so neither library is imported), so the hooks in `vlm.py` and `lm.py` are one-line delegations to a module that is still engine-land and still function-local about mlx.
3. **The store owns the policy around a write or a read; the engine owns the bytes.** `ForkStore.persist(ids, boundary, write)` handles the floor, the budget, the in-flight set, the temp name, `os.replace`, the index and the counters, and calls the engine's `write(temp_path, metadata)` in the middle; `ForkStore.restore(entry, load)` handles the restoring set, `utime`, the delete-on-verification-failure rule and the consecutive-failure disable, and calls `load(path, header)` in the middle. `PrefixCache` sees six methods and a status dict, all fakeable.
4. **`weights_identity_for` calls `hf_hub_download(..., local_files_only=True)`.** `fetch_model_config` has just downloaded `config.json`, so the cached path resolves with no network; the snapshot directory name is the commit sha (the path must NOT be `resolve()`d — that follows the symlink into `blobs/`). A path not under `snapshots/` (a local directory) hashes `config.json` plus each `*.safetensors` `(name, size, mtime_ns)`.
5. **`sous top` shows the disk fork count, not the bytes.** The spec extends the wide `PROMPT CACHE` line with count and bytes; that line is 32 columns and every row of the wide body must fit 35 (an existing test pins it), so the count alone rides on the `reused` line and the bytes stay on `sous status`.
6. **`disk_evictions` lives on the store, not in `PromptCacheStats`.** Evictions are the store's daemon-wide bookkeeping, not a per-owner count, so they surface as `prompt_cache.disk.evictions` in the status document (a sixth key beside the spec's five) rather than as a prompt-cache counter, and `PrefixCache` is never told a file was dropped. The spec's `sous version` metadata field is kept as `sous`.

---

### Task 1: forkstore rules — digest, header, identity key, naming, budgets

**Files:**
- Create: `src/sous/engine/forkstore.py`
- Test: `tests/test_forkstore.py`

**Interfaces:**
- Produces (used by Tasks 2–7): `GIB`, `FORMAT = "sous-fork-v1"`, `FORK_LAYOUT = 1`, `ForkFileError`, `digest(ids) -> str`, `TensorInfo`, `Header`, `read_header(path) -> Header`, `key_name(fields) -> str`, `engine_epoch() -> str`, `weights_identity(config_path) -> str`, `file_name(n, digest) -> str`, `parse_file_name(name) -> tuple[int, str] | None`, `temp_name(final, pid) -> Path`, `parse_temp_name(name) -> int | None`, `auto_budget(free) -> int`, `floor_bytes(total) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forkstore.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_forkstore.py -v 2>&1 | tail -5`
Expected: `ModuleNotFoundError: No module named 'sous.engine.forkstore'`.

- [ ] **Step 3: Write the module's rules**

```python
# src/sous/engine/forkstore.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forkstore.py -v 2>&1 | tail -5`
Expected: 14 passed.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check 2>&1 | tail -3`
Expected: no diagnostics.

```bash
git add src/sous/engine/forkstore.py tests/test_forkstore.py
git commit -m "feat(engine): fork store rules — digest, header parse, identity key, naming, budgets" -m "The mlx-free half of persisting fork slots: everything that decides which file serves a render and what makes a file stale, testable in CI with fakes. The epoch is hashed from the engine sources so a change to the numerics can never be forgotten." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `ForkStore` — index, budget, LRU, floor, persist/restore policy

**Files:**
- Modify: `src/sous/engine/forkstore.py` (append)
- Test: `tests/test_forkstore.py` (append)

**Interfaces:**
- Consumes: Task 1's helpers.
- Produces: `Entry(path, key, n, digest, size, mtime)`; `class ForkStore` with `__init__(root: Path, key_fields: Mapping[str, str], budget: int | None, *, pid_alive=None, disk_usage=shutil.disk_usage, pid=os.getpid)` (`pid_alive=None` means the module's psutil-backed `_pid_alive`), attributes `state: str` (`"active"` | `"unavailable"`), `reason: str | None`, `key: str`, `dir: Path`, `budget: int`, `forks: int`, `bytes: int`, `evictions: int`; methods `status() -> dict`, `has(ids) -> bool`, `longest_prefix(stable_ids) -> Entry | None`, `persist(ids, boundary: str, write: Callable[[Path, dict[str, str]], None]) -> bool`, `restore(entry, load: Callable[[Path, Header], None]) -> bool`, `touch(ids) -> None`.
- The `write` callable receives the temp path and the metadata dict the store composed (`format`, `layout`, every key field, `n_tokens`, `boundary`, `created`); the engine adds its per-layer fields and writes. The `load` callable raises `ForkFileError` for a verification failure and anything else for a keep-the-file failure.

- [ ] **Step 1: Write the failing tests**

First extend the file's header import block — never a mid-file import (E402): add `import os`, `import shutil` and `import warnings` to the stdlib group, and `FORK_LAYOUT`, `ForkStore` and `Header` to the `from sous.engine.forkstore import (...)` list. Then append:

```python
# tests/test_forkstore.py (append)
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
    write_safetensors(d / file_name(5, digest(IDS[:5])), {"x": b"\x00" * 8},
                      {"format": "someone-elses", "n_tokens": "5"})
    p = d / file_name(6, digest(IDS[:6]))
    fake_write(64, IDS[:6])(p, {"format": FORMAT, "layout": str(FORK_LAYOUT), "n_tokens": "6",
                                **KEY})
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_forkstore.py -v 2>&1 | tail -5`
Expected: `ImportError: cannot import name 'ForkStore'`.

- [ ] **Step 3: Write `Entry` and `ForkStore`**

Append to `src/sous/engine/forkstore.py`:

```python
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
```

Add to the module's imports, keeping each group sorted: `import dataclasses`, `import errno as errno_mod`, `import os`, `import shutil`, `import threading`, `import time`, `import warnings`; `Callable` beside `Mapping, Sequence` in the `collections.abc` import; and `from typing import Any`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forkstore.py -v 2>&1 | tail -5`
Expected: 43 passed (14 from before and 29 here; the disk-error test is four cases).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check 2>&1 | tail -3`
Expected: no diagnostics.

```bash
git add src/sous/engine/forkstore.py tests/test_forkstore.py
git commit -m "feat(engine): fork store index, budget, LRU and the persist/restore policy" -m "The store owns everything around a write or a read — the free-space floor, LRU across keys, temp-then-replace, the in-flight sets, delete-on-verification-failure and the consecutive-failure disable — and hands the bytes to a callable, so the array code stays in the engines and the policy is testable here." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: forkio — the arrays

**Files:**
- Create: `src/sous/engine/forkio.py`
- Test: `tests/test_engine_forkio.py`

**Interfaces:**
- Consumes: `forkstore.ForkFileError`, `forkstore.Header`.
- Produces: `ForkUnsupported(Exception)`, `layer_kinds(cache) -> list[str]` (`"kv"` | `"arrays"`), `eval_cache(cache) -> None`, `persist_cache(cache, path, ids, metadata) -> None`, `restore_cache(path, header, cache, ids) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_forkio.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine_forkio.py -v 2>&1 | tail -5`
Expected: `ModuleNotFoundError: No module named 'sous.engine.forkio'`.

- [ ] **Step 3: Write the module**

```python
# src/sous/engine/forkio.py
"""The array half of fork persistence: one safetensors file per fork,
written from a live cache and read back into a fresh one. Every mlx import
is function-local; the rules around a file (naming, budget, when to write,
what to delete) live in forkstore, which imports no mlx at all.

Two layer classes are supported, recognised by name so neither mlx-lm nor
mlx-vlm is imported here: KVCache (keys, values, offset — the attention
layers) and ArraysCache (a list of state arrays — the recurrent layers).
They are what make_prompt_cache builds for every model sous serves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sous.engine.forkstore import ForkFileError, Header


class ForkUnsupported(Exception):
    """A cache with a layer class the file format does not cover."""


def layer_kinds(cache: Sequence[Any]) -> list[str]:
    kinds: list[str] = []
    for c in cache:
        name = type(c).__name__
        if name == "KVCache":
            kinds.append("kv")
        elif name == "ArraysCache":
            kinds.append("arrays")
        else:
            raise ForkUnsupported(f"layer class {name} cannot be persisted")
    return kinds


def eval_cache(cache: Sequence[Any]) -> None:
    """Materialise every layer's arrays on the calling thread. A fork copy
    is lazy until something evaluates it, and mx.get_active_memory() — what
    the pressure valve reads — does not count a lazy array."""
    import mlx.core as mx

    arrays = [a for c in cache for a in _arrays_of(c) if a is not None]
    if arrays:
        mx.eval(*arrays)


def _arrays_of(c: Any) -> list[Any]:
    if type(c).__name__ == "KVCache":
        return [c.keys, c.values]
    return list(c.cache)


def persist_cache(
    cache: Sequence[Any], path: Path, ids: Sequence[int], metadata: Mapping[str, str]
) -> None:
    """Write `cache` at `path`. The FULL padded key/value buffers go in, with
    the offset in metadata: saving the offset slice would materialise every
    tensor at once (a transient the size of the whole fork), and restoring
    the padded buffer keeps the capacity a never-serialised cache has. The
    arrays are evaluated first — free when they already are — so the save
    itself allocates nothing."""
    import mlx.core as mx

    kinds = layer_kinds(cache)
    eval_cache(cache)
    arrays: dict[str, Any] = {"ids": mx.array(list(ids), dtype=mx.int32)}
    meta: dict[str, str] = {**metadata, "kinds": ",".join(kinds)}
    for i, (c, kind) in enumerate(zip(cache, kinds, strict=True)):
        if kind == "kv":
            if c.keys is None or c.values is None:
                raise ForkUnsupported(f"layer {i}: an empty KVCache cannot be persisted")
            meta[f"c{i}_kind"] = "kv"
            meta[f"c{i}_offset"] = str(int(c.offset))
            arrays[f"c{i}_k"] = c.keys
            arrays[f"c{i}_v"] = c.values
        else:
            meta[f"c{i}_kind"] = "arrays"
            meta[f"c{i}_size"] = str(len(c.cache))
            for j, state in enumerate(c.cache):
                if state is None:
                    meta[f"c{i}_s{j}_none"] = "1"
                else:
                    arrays[f"c{i}_s{j}"] = state
    mx.save_safetensors(str(path), arrays, metadata=meta)


def restore_cache(path: Path, header: Header, cache: Sequence[Any], ids: Sequence[int]) -> None:
    """Fill `cache` — fresh from new_cache() — from `path`, on the calling
    thread, and evaluate it: the arrays then belong to this thread and are
    visible to mx.get_active_memory() before the pressure valve next reads
    it. A KVCache gets keys, values and then offset from metadata — never
    the state setter, which derives the offset from the padded shape and
    would misposition every token appended afterwards. Raises ForkFileError
    on any mismatch: the caller deletes the file."""
    import mlx.core as mx

    meta = header.metadata
    kinds = layer_kinds(cache)
    if meta.get("kinds", "").split(",") != kinds:
        raise ForkFileError(f"{path.name}: layer layout {meta.get('kinds')!r} != {kinds}")
    n = len(ids)
    if meta.get("n_tokens") != str(n):
        raise ForkFileError(f"{path.name}: n_tokens {meta.get('n_tokens')!r} != {n}")
    loaded = mx.load(str(path))
    if "ids" not in loaded:
        raise ForkFileError(f"{path.name}: no ids tensor")
    if loaded["ids"].tolist() != list(ids):
        raise ForkFileError(f"{path.name}: ids differ")
    targets: list[Any] = []
    for i, (c, kind) in enumerate(zip(cache, kinds, strict=True)):
        if kind == "kv":
            try:
                offset = int(meta[f"c{i}_offset"])
                keys, values = loaded[f"c{i}_k"], loaded[f"c{i}_v"]
            except (KeyError, ValueError) as e:
                raise ForkFileError(f"{path.name}: layer {i} incomplete") from e
            if offset != n or keys.shape[2] < n or values.shape[2] < n:
                raise ForkFileError(f"{path.name}: layer {i} offset {offset} for {n} ids")
            c.keys, c.values, c.offset = keys, values, offset
            targets += [keys, values]
        else:
            try:
                size = int(meta[f"c{i}_size"])
            except (KeyError, ValueError) as e:
                raise ForkFileError(f"{path.name}: layer {i} incomplete") from e
            if size != len(c.cache):
                raise ForkFileError(f"{path.name}: layer {i} has {size} states, want {len(c.cache)}")
            states: list[Any] = []
            for j in range(size):
                if meta.get(f"c{i}_s{j}_none") == "1":
                    states.append(None)
                elif f"c{i}_s{j}" in loaded:
                    states.append(loaded[f"c{i}_s{j}"])
                    targets.append(states[-1])
                else:
                    raise ForkFileError(f"{path.name}: layer {i} state {j} missing")
            c.state = states
    mx.eval(*targets)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine_forkio.py -v 2>&1 | tail -5`
Expected: 11 passed (the mismatch test is four cases).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check 2>&1 | tail -3`
Expected: no diagnostics (a `# ty: ignore[...]` may be needed on attribute access of the untyped cache classes — add exactly the rule `ty` names).

```bash
git add src/sous/engine/forkio.py tests/test_engine_forkio.py
git commit -m "feat(engine): write a cache's buffers to safetensors and read them back" -m "Full padded buffers with the offset in metadata, restored by assigning keys, values and offset directly: the state slice would materialise the whole fork on write, and the state setter derives the offset from the padded shape and would misposition every appended token." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: promptcache — hooks, stats, the store protocol and the test fakes

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`CacheHooks` ~412-435, `TURN_GAUGES` ~283-294, `PromptCacheStats` ~314-352, `PrefixCache.__init__` ~478-514, `stats()` ~828-846, `_copy_of` ~848-870), `src/sous/engine/vlm.py` and `src/sous/engine/lm.py` (the three hook methods — both engines pass themselves as `CacheHooks`, so `ty` rejects the Protocol growing a member they lack)
- Test: `tests/test_promptcache.py` (`FakeHooks` ~314-378, `_empty_stats` ~154, `test_stats_as_dict_reports_every_counter` ~876)

**Interfaces:**
- Consumes: nothing new from Tasks 1–3 at runtime (promptcache stays mlx-free and imports nothing from `forkstore`; the store reaches it as a `StoreHooks` object).
- Produces: `CacheHooks.eval_cache(cache)`, `CacheHooks.persist(cache, path, ids, metadata)`, `CacheHooks.restore(path, header, cache, ids)`; `StoreHooks` protocol (`state`, `has`, `longest_prefix`, `persist`, `restore`, `touch`, `status`); `PrefixCache(..., store: StoreHooks | None = None)`; stats fields `persists`, `restores`, `disk_hits`, `restore_skips` (counters) and `persist_seconds`, `restore_seconds` (turn gauges); `stats()["disk"]`; `DISK_OFF`. Tests: `FakeHooks.log`, `FakeHooks.persist_calls`, `FakeHooks.restore_calls`, `FakeHooks.evaluated`, `FakeStore`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py`, after `FakeHooks` (before `STABLE_1`):

```python
class FakeEntry:
    """What the store hands back for a stored prefix: its length, its bytes
    and a path the fake loader is called with."""

    def __init__(self, n: int, ids: list[int], size: int):
        self.n, self.ids, self.size = n, list(ids), size
        self.path = f"/fake/{n}"


class FakeStore:
    """The disk store, in memory: files keyed by their ids. Records every
    call so a test can see the order and the arguments; `state` and the
    two failure switches script the store's refusals."""

    def __init__(self, size: int = 4096):
        self.files: dict[tuple[int, ...], FakeEntry] = {}
        self.size = size
        self.state = "active"
        self.touched: list[list[int]] = []
        self.persist_calls: list[tuple[list[int], str]] = []
        self.restore_calls: list[int] = []
        self.refuse_persist = False
        self.fail_restore = False
        self.evictions = 0

    def has(self, ids) -> bool:
        return tuple(ids) in self.files

    def longest_prefix(self, stable_ids):
        best = None
        for key, entry in self.files.items():
            if 0 < entry.n < len(stable_ids) and list(stable_ids[: entry.n]) == entry.ids:
                if best is None or entry.n > best.n:
                    best = entry
        return best

    def persist(self, ids, boundary, write) -> bool:
        self.persist_calls.append((list(ids), boundary))
        if self.state != "active" or self.refuse_persist or self.has(ids):
            return False
        write(f"/fake/{len(ids)}.tmp", {"n_tokens": str(len(ids)), "boundary": boundary})
        self.files[tuple(ids)] = FakeEntry(len(ids), list(ids), self.size)
        return True

    def restore(self, entry, load) -> bool:
        self.restore_calls.append(entry.n)
        if self.fail_restore:
            return False
        load(entry.path, {"n_tokens": str(entry.n)})
        return True

    def touch(self, ids) -> None:
        self.touched.append(list(ids))

    def status(self) -> dict:
        return {
            "state": self.state,
            "reason": None,
            "forks": len(self.files),
            "bytes": sum(e.size for e in self.files.values()),
            "budget_bytes": 1 << 34,
            "evictions": self.evictions,
        }


def stored(*prefixes: list[int], size: int = 4096) -> FakeStore:
    """A store already holding files for `prefixes`."""
    s = FakeStore(size=size)
    for ids in prefixes:
        s.files[tuple(ids)] = FakeEntry(len(ids), list(ids), size)
    return s
```

Add to `FakeHooks.__init__` (after `self.pressure_value`):

```python
        # The order of the calls a turn makes, so a test can pin that the
        # persist runs at the boundary before the resident copy is taken.
        self.log: list[str] = []
        self.persist_calls: list[tuple[list[int], list[int]]] = []  # (ids, offsets seen)
        self.restore_calls: list[list[int]] = []
        self.evaluated: list[list] = []
        self.persist_impl = None
```

Add to `FakeHooks` these methods, and one `self.log.append(...)` line at the top of the existing `new_cache`, `prefill` and `decode` (`"new_cache"`, `"prefill"`, `"decode"`):

```python
    def eval_cache(self, cache):
        self.evaluated.append(cache)

    def persist(self, cache, path, ids, metadata):
        self.log.append("persist")
        if self.persist_impl is not None:
            return self.persist_impl(self, cache, path, ids, metadata)
        offsets = [c.offset for c in cache if isinstance(c, FakeTrimmable)]
        self.persist_calls.append((list(ids), offsets))

    def restore(self, path, header, cache, ids):
        self.log.append("restore")
        self.restore_calls.append(list(ids))
        self._advance(cache, len(ids))
```

Replace `_empty_stats`:

```python
def _empty_stats() -> dict:
    from sous.engine.promptcache import DISK_OFF

    return {**PromptCacheStats().as_dict(), "slots": 0, "resident_bytes": 0, "disk": dict(DISK_OFF)}
```

In `test_stats_as_dict_reports_every_counter`, add to the constructor call `persists=1, restores=2, disk_hits=2, restore_skips=1,` and to the expected dict, after `"moved": 1,`:

```python
        "persists": 1,
        "restores": 2,
        "disk_hits": 2,
        "restore_skips": 1,
```

and after `"decode_seconds": 0.0,`:

```python
        "persist_seconds": 0.0,
        "restore_seconds": 0.0,
```

Append these tests at the end of the stats section (after `test_begin_turn_resets_exactly_the_turn_gauges`):

```python
def test_persist_and_restore_seconds_are_turn_gauges_and_the_counts_are_not():
    from sous.engine.promptcache import TURN_GAUGES

    assert {"persist_seconds", "restore_seconds"} <= TURN_GAUGES
    assert TURN_GAUGES.isdisjoint({"persists", "restores", "disk_hits", "restore_skips"})
    a = PromptCacheStats(persists=1, restores=2, persist_seconds=3.0)
    b = PromptCacheStats(persists=1, restores=1, persist_seconds=1.0)
    a.add(b)
    assert (a.persists, a.restores, a.persist_seconds) == (2, 3, 3.0)


def test_stats_report_the_store_status_or_off():
    from sous.engine.promptcache import DISK_OFF

    assert PrefixCache(FakeHooks(trimmable=True)).stats()["disk"] == DISK_OFF
    s = stored(HEADER)
    pc = PrefixCache(FakeHooks(trimmable=True), store=s)
    assert pc.stats()["disk"] == s.status()
    assert pc.stats()["disk"]["forks"] == 1


def test_a_fork_copy_is_evaluated_before_it_is_charged():
    """A copy is lazy until something evaluates it, and the pressure valve
    reads active memory: an unevaluated fork would hand it headroom that
    does not exist."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=[FORK])
    assert len(h.evaluated) == 1  # the fork copy
    pc.generate(C2, C2_FULL, 16, fork_at=[FORK])
    assert len(h.evaluated) == 2  # the copy the fork hit took
```

(`HEADER` is at ~line 783; `H`, `FORK`, `C1`, `C1_FULL`, `C2`, `C2_FULL` are the single-boundary constants at ~lines 1552-1559. Module-level order does not matter — the names resolve at call time.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "stats or evaluated or store_status" -v 2>&1 | tail -8`
Expected: `test_stats_as_dict_reports_every_counter`, `test_persist_and_restore_seconds_are_turn_gauges_and_the_counts_are_not`, `test_stats_report_the_store_status_or_off`, `test_a_fork_copy_is_evaluated_before_it_is_charged` fail (`TypeError: unexpected keyword 'persists'`, `ImportError: DISK_OFF`, `unexpected keyword 'store'`, `AttributeError`).

- [ ] **Step 3: Extend promptcache**

In `TURN_GAUGES` add `"persist_seconds"` and `"restore_seconds"`. After `without_turn_gauges`:

```python
# The status document's `prompt_cache.disk` block when no store was built:
# off by config, or an engine constructed without a directory (every test).
DISK_OFF: dict = {
    "state": "off",
    "reason": None,
    "forks": 0,
    "bytes": 0,
    "budget_bytes": 0,
    "evictions": 0,
}
```

In `PromptCacheStats`, after `moved: int = 0`:

```python
    # Disk forks: files written, files read back, turns served from one,
    # and restores the memory readings refused (the file stays).
    persists: int = 0
    restores: int = 0
    disk_hits: int = 0
    restore_skips: int = 0
```

and after `decode_seconds`:

```python
    persist_seconds: float = 0.0  # writing the live cache at a boundary; never prefill time
    restore_seconds: float = 0.0  # reading a fork off disk; never prefill time
```

Extend `CacheHooks` (add `from pathlib import Path` to the imports):

```python
    def eval_cache(self, cache: list) -> None: ...
    def persist(self, cache: list, path: Path, ids: list[int], metadata: dict[str, str]) -> None: ...
    def restore(self, path: Path, header: Any, cache: list, ids: list[int]) -> None: ...
```

replacing the class docstring's first sentence `The five things only an engine can do.` with `The things only an engine can do.` and adding: "`eval_cache` materialises a cache's arrays on the calling thread (a fork copy is lazy, and the pressure valve reads active memory). `persist` writes the live working cache at a boundary to `path` — called on the owner thread, holding no lock, never with a Slot; `restore` fills a fresh cache from `path` on the calling thread and evaluates it, raising the store's verification error on any mismatch."

Add the store protocol after `CacheHooks`:

```python
class StoreHooks(Protocol):
    """The disk store as the orchestrator sees it (engine/forkstore.ForkStore
    in production, a fake in tests). Files are shared across owner threads;
    the arrays a restore allocates belong to the restoring thread, so
    residency stays owner-scoped while the file is not."""

    state: str

    def has(self, ids: Sequence[int]) -> bool: ...
    def longest_prefix(self, stable_ids: Sequence[int]) -> Any: ...
    def persist(
        self, ids: Sequence[int], boundary: str, write: Callable[[Path, dict[str, str]], None]
    ) -> bool: ...
    def restore(self, entry: Any, load: Callable[[Path, Any], None]) -> bool: ...
    def touch(self, ids: Sequence[int]) -> None: ...
    def status(self) -> dict: ...
```

`PrefixCache.__init__` gains `store: StoreHooks | None = None` after `reserve_bytes` and sets `self._store = store`. Add to the class docstring: "Disk forks (`store`) are the one thing here shared across owners: a file has no thread affinity, and every restore allocates arrays of its own on the restoring thread. The owner rule is about arrays, not content."

In `stats()`, read `disk = self._store.status() if self._store is not None else dict(DISK_OFF)` *before* `with self._lock:` (no store call runs under the bookkeeping lock) and add `"disk": disk` to the returned dict after `"resident_bytes"`.

In `_copy_of`, after `fork_copy(cache, copy, hooks.copy_array)` add `hooks.eval_cache(copy)`; in `_run`'s loop likewise after its `fork_copy(...)` line. Update `Slot`'s docstring with one sentence: "A fork's file on disk, when the store keeps one, is shared across owners; only the arrays are owner-scoped."

`vlm.py`: add `from pathlib import Path` to the imports, `Any` is already imported, and add `from sous.engine import forkio` (forkio imports no mlx at module level). `lm.py`: add `from pathlib import Path`, change `from typing import cast` to `from typing import Any, cast`, and add `from sous.engine import forkio`. In both, in the CacheHooks section after `pressure`:

```python
    def eval_cache(self, cache: list) -> None:
        forkio.eval_cache(cache)

    def persist(self, cache: list, path: Path, ids: list[int], metadata: dict[str, str]) -> None:
        forkio.persist_cache(cache, path, ids, metadata)

    def restore(self, path: Path, header: Any, cache: list, ids: list[int]) -> None:
        forkio.restore_cache(path, header, cache, ids)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_promptcache.py 2>&1 | tail -3`
Expected: 3 more than before in this file, all green (the seven `_empty_stats()` comparisons pass because `stats()` now carries `disk`).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff format . && uv run ruff check . && uv run ty check 2>&1 | tail -3` — no diagnostics.

```bash
git add src/sous/engine/promptcache.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_promptcache.py
git commit -m "feat(engine): prompt-cache hooks and counters for forks on disk" -m "The store protocol, the three array hooks, the per-turn gauges (never folded into prefill time) and the counters, plus the eval of every fork copy: a lazy copy is invisible to the memory the pressure valve reads." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: promptcache — persist at the lowest boundary

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`_fork_boundaries` ~620-687, `_run` ~1117-1188, new `_persist`)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: Task 4's `StoreHooks`, `FakeStore`, `FakeHooks.log`.
- Produces: `_fork_boundaries(...) -> list[tuple[int, bool, bool]]` (`boundary, need_slot, need_file`); `_persist(stats, cache, ids, boundary_kind)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py` after `test_a_second_session_starts_from_the_tools_fork_and_its_next_subagent_from_its_header`:

```python
# ---- forks on disk: the write ----------------------------------------------


def test_a_cold_turn_persists_the_lowest_boundary_from_the_live_cache_before_the_copy():
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    assert list(s.files) == [tuple(T)]  # the tools boundary only, never the header
    assert s.persist_calls == [(T, "tools")]
    ids, offsets = h.persist_calls[0]
    assert ids == T and offsets == [TOOLS_AT] * 2  # the live cache, at the boundary
    # the working cache, the prefill to the boundary, the write, THEN the
    # resident copy's allocation
    assert h.log[:4] == ["new_cache", "prefill", "persist", "new_cache"]
    st = pc.stats()
    assert (st["persists"], st["forks"]) == (1, 2)  # both resident forks still made


def test_a_second_turn_skips_the_write_and_touches_the_file():
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    # A different session, same tools: warm from the resident fork, and the
    # boundary loop no longer enters for the tools boundary at all.
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert len(s.persist_calls) == 1
    assert s.touched == [T]
    assert pc.stats()["persists"] == 1


def test_a_turn_warm_at_its_header_fork_still_touches_the_tools_file():
    """The tools file belongs to a fork that is resident and never restored,
    so nothing else refreshes it; without the touch it is the oldest file in
    the store and the first the LRU takes."""
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    pc.generate(AX2, AX2_FULL, 16, fork_at=BOUNDS_A)  # warm at HA, above the tools boundary
    assert s.touched == [T]


def test_a_single_header_boundary_is_persisted_as_the_header():
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(C1, C1_FULL, 16, fork_at=[FORK])
    assert s.persist_calls == [(H, "header")]


@pytest.mark.parametrize("trimmable", [True, False])
def test_at_budget_zero_the_file_is_written_and_no_fork_is_resident(trimmable):
    """A machine that cannot hold a resident fork still gets the disk copy:
    the probe resolves, the turn stops at the boundary, the live cache is
    written, and no copy is charged."""
    h = FakeHooks(trimmable=trimmable)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=0, store=s)
    resolved = []

    def probe():
        resolved.append(True)
        return BOUNDS_A

    pc.generate(AX1, AX1_FULL, 16, fork_at=probe)
    assert resolved == [True]
    assert list(s.files) == [tuple(T)]
    assert [x.kind for x in pc.slots()] == ["turn"]
    st = pc.stats()
    assert (st["persists"], st["forks"], st["bound_lo"], st["bound_hi"]) == (1, 0, TOOLS_AT, len(HA))
    assert h.prefilled[0] == T  # stopped at the boundary to write


def test_without_a_store_budget_zero_still_resolves_no_probe():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=0)
    resolved = []

    def probe():
        resolved.append(True)
        return BOUNDS_A

    pc.generate(AX1, AX1_FULL, 16, fork_at=probe)
    assert resolved == []


def test_a_raising_persist_hook_does_not_fail_a_cold_turn():
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)

    def boom(hooks, cache, path, ids, metadata):
        raise OSError("disk")

    h.persist_impl = boom
    assert pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A) == "text"
    assert pc.stats()["persists"] == 0
    assert pc.stats()["forks"] == 2  # the resident forks were still taken


def test_a_store_that_refuses_counts_nothing():
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    s.refuse_persist = True
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    assert pc.stats()["persists"] == 0 and h.persist_calls == []


def test_persist_time_is_its_own_gauge_not_prefill_time(clock):
    class WritingHooks(TimedHooks):
        def persist(self, cache, path, ids, metadata):
            self.clock.advance(3.0)
            super().persist(cache, path, ids, metadata)

    h = WritingHooks(clock, trimmable=False)
    pc = PrefixCache(h, max_bytes=ROOMY, store=FakeStore())
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    st = pc.stats()
    assert st["persist_seconds"] == pytest.approx(3.0)
    # three prefill segments (to T, to HA, the rest) at 2 s each; the write is not in it
    assert st["prefill_seconds"] == pytest.approx(6.0)
    assert st["decode_seconds"] == pytest.approx(5.0)


def test_a_retired_owner_still_persists_though_its_fork_is_refused():
    """Retirement exists because the arrays live on a dead thread's streams;
    a file has no thread. The stall path's reset must not lose the write."""
    h = FakeHooks(trimmable=True)
    s = FakeStore()
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    original = h.prefill

    def prefill(cache, token_ids):
        original(cache, token_ids)
        pc.reset(threading.current_thread())  # the endpoint retiring this session

    h.prefill = prefill  # ty: ignore[invalid-assignment]
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    assert list(s.files) == [tuple(T)]
    assert pc.slots() == []  # every publish refused


def test_a_turn_starting_exactly_at_a_boundary_republishes_a_missing_fork():
    """A fork lost to a pressure eviction, or one a disk restore started at,
    is rebuilt through the existing charged path: the prefill to the
    boundary is empty and the copy is taken from the live cache."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    at_header = h.new_cache()
    h._advance(at_header, len(HA))
    pc._plant(at_header, HA)  # a turn slot standing exactly at the header boundary
    pc.generate(AX2, AX2_FULL, 16, fork_at=BOUNDS_A)  # takes it: reuse == len(HA)
    assert [x.held for x in pc.slots() if x.kind == "fork"] == [HA]
    assert h.prefilled[0] == []  # an empty stop at the boundary
    st = pc.stats()
    assert (st["forks"], st["retained"]) == (1, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "persist or budget_zero or republishes" -v 2>&1 | tail -12`
Expected: every new test fails (`s.files` empty, `persists == 0`, `resolved == []` at budget 0, `forks == 2` not 3).

- [ ] **Step 3: Rewrite `_fork_boundaries` and the boundary loop**

Replace `_fork_boundaries`'s signature and body from the early return down:

```python
    def _fork_boundaries(
        self,
        stats: PromptCacheStats,
        owner: threading.Thread,
        stable_ids: list[int],
        fork_at: Sequence[int] | Callable[[], Sequence[int]],
        reuse: int,
    ) -> list[tuple[int, bool, bool]]:
        """The boundaries this turn will stop at, ascending, each with what
        it wants there: a resident copy (`need_slot` — this owner holds no
        fork with exactly those ids, and there is a budget to charge one to)
        and/or a file (`need_file` — the store has none for those ids; only
        the lowest boundary is ever written, the one the next session shares).
        A boundary is returned when either is true. The candidates are those
        this turn prefills to (`reuse <= b`: a turn that starts exactly at a
        boundary — a disk restore, or a fork the valve took — republishes
        there through the same path, with an empty prefill), that leave
        something to prefill after them (`b < len(stable_ids)`) and that
        clear the fork floor. Nothing is resolved when there is neither a
        budget nor a store: at `max_bytes == 0` without a store nothing about
        the turn changes.

        <keep the existing paragraphs about resolving the probe on warm
        turns and about resolving it before the lock>
        """
        if self.max_bytes <= 0 and self._store is None:
            return []
        <unchanged: resolve `candidates` from fork_at, with the probe_seconds
        timer and the warning>
        inside = sorted({b for b in candidates if b >= FORK_MIN_TOKENS and b < len(stable_ids)})
        if len(inside) >= 2:
            stats.bound_lo, stats.bound_hi = inside[0], inside[-1]
        elif inside:
            stats.bound_lo, stats.bound_hi = 0, inside[0]
        wanted = sorted({b for b in inside if reuse <= b})
        if not inside:
            return []
        with self._lock:
            forks = [s.held for s in self._slots if s.owner is owner and s.kind == "fork"]
        # The lowest boundary's file, when it exists, is touched on every
        # turn that passes or starts at or above it: a resident fork's file
        # is otherwise the oldest in the store and the first the LRU takes.
        store = self._store if self._store is not None and self._store.state == "active" else None
        lowest = inside[0]
        on_disk = store is not None and store.has(stable_ids[:lowest])
        if on_disk:
            store.touch(stable_ids[:lowest])
        out: list[tuple[int, bool, bool]] = []
        for b in wanted:
            need_slot = self.max_bytes > 0 and stable_ids[:b] not in forks
            need_file = store is not None and b == lowest and not on_disk
            if need_slot or need_file:
                out.append((b, need_slot, need_file))
        return out
```

Add `_persist` after `_copy_of`:

```python
    def _persist(
        self, stats: PromptCacheStats, cache: list, ids: list[int], boundary: str
    ) -> None:
        """Write the live cache, which stands exactly at `ids`' end, to the
        store — before any resident copy, so the machine holds the live cache
        alone across the write, and outside the prefill timer, so the turn
        line's prefill rate stays true. Charged to nothing: the live cache
        is the turn's own. Its own handler, because a cold turn's generate
        re-raises rather than retrying and a disk error must never fail a
        viable prefill."""
        if self._store is None:
            return
        hooks = self._hooks
        started = _clock()
        try:
            if self._store.persist(
                ids, boundary, lambda path, metadata: hooks.persist(cache, path, ids, metadata)
            ):
                stats.persists += 1
        except Exception as e:  # noqa: BLE001 — an optimization must never fail a turn
            warnings.warn(
                f"sous prompt cache: fork persist failed ({type(e).__name__}); "
                "continuing without one",
                stacklevel=2,
            )
        finally:
            stats.persist_seconds += _clock() - started
```

Rewrite the head of `_run`'s boundary loop (up to `price = slot_bytes(cache)`):

```python
        for boundary, need_slot, need_file in self._fork_boundaries(
            stats, owner, stable_ids, fork_at, reuse
        ):
            # Stop at the boundary and continue from there: a fork is taken
            # while prefilling past it because no layer can be rewound to it
            # afterwards. Ascending order means no copy is ever wanted behind
            # a prefix already prefilled. An empty segment (a turn that
            # started exactly here) is a no-op on both backends.
            started = _clock()
            hooks.prefill(cache, list(stable_ids[reuse:boundary]))
            reuse = boundary
            stats.prefill_seconds += _clock() - started
            if need_file:
                # The lowest boundary is the tools boundary when the probe
                # found two, and the header when it found one.
                kind = "tools" if stats.bound_lo else "header"
                self._persist(stats, cache, list(stable_ids[:boundary]), kind)
            if not need_slot:
                continue
            started = _clock()
            # What the cache holds at the boundary is what the copy will hold,
            # so it is also the price — re-read at every stop, since the cache
            # has grown.
            price = slot_bytes(cache)
```

The rest of the loop body (`copy`/`slot`/`try`/`except`/`finally`) is unchanged except the `hooks.eval_cache(copy)` Task 4 added.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_promptcache.py 2>&1 | tail -3`
Expected: 12 more than after Task 4 (the budget-zero test is two cases), all green. `test_a_callable_fork_at_is_not_resolved_without_a_budget` still passes (no store); `test_a_turn_warm_at_the_header_fork_forks_nothing` still passes (the header boundary at `reuse == len(HA)` finds its fork resident and is not the lowest boundary).

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): write the live cache to the fork store at the lowest boundary" -m "Before the resident copy and outside the prefill timer, whether or not the RAM budget can hold a copy: a 48 GB machine at prompt_cache_gb = 0 gets a warm cold-start it never had. A turn that starts exactly at a boundary republishes a resident fork there through the existing charged path." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: promptcache — restore on a miss

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`generate` ~872-1115: the fork-clone branch and the miss branch; new `_restore_from_disk`)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: Task 4/5.
- Produces: `_restore_from_disk(stats, stable_ids) -> tuple[list, int] | None`; on a disk hit `stats.hits`, `disk_hits`, `reused_tokens`, `took_len`, `took_kind == "disk"`, `restores`, `restore_seconds` move and `misses` does not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py`:

```python
# ---- forks on disk: the read ------------------------------------------------


def test_a_miss_restores_the_stored_tools_fork_into_the_turns_own_cache():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert s.restore_calls == [TOOLS_AT] and h.restore_calls == [T]
    st = pc.stats()
    assert (st["hits"], st["disk_hits"], st["misses"], st["fork_hits"]) == (1, 1, 0, 0)
    assert (st["reused_tokens"], st["took_len"], st["took_kind"]) == (TOOLS_AT, TOOLS_AT, "disk")
    assert st["restores"] == 1
    # an empty stop at T (to republish it), then only what lies past the prefix
    assert h.prefilled[:2] == [[], HB[TOOLS_AT:]]
    assert s.touched == [T]
    assert h.decoded[-1] == BX1_FULL[len(HB) :]
    # Residency came back through the boundary loop: the tools fork is
    # resident again and B's own header fork was published as always.
    assert sorted(x.held for x in pc.slots() if x.kind == "fork") == sorted([T, HB])
    assert st["forks"] == 2
    assert len(s.persist_calls) == 0  # the file already existed: no rewrite


def test_the_restored_cache_is_the_working_cache_not_a_planted_slot():
    """The restored arrays are the turn's own — never charged, never in the
    map — and the resident copy is made room for through _make_room like
    every fork: the machine never holds restored + planted + copy."""
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    seen: list[int] = []
    h.on_new_cache = lambda: seen.append(len(pc.slots()))
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    # new_cache for the restore (map empty), then for the resident copy of
    # T (map still empty: nothing was planted), then for HB's copy.
    assert seen[:2] == [0, 0]


def test_a_restore_at_budget_zero_plants_nothing_and_serves_the_turn():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=0, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert (st["disk_hits"], st["took_kind"], st["forks"]) == (1, "disk", 0)
    assert [x.kind for x in pc.slots()] == ["turn"]


def test_the_longest_stored_prefix_wins():
    h = FakeHooks(trimmable=True)
    s = stored(T, HB)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert s.restore_calls == [len(HB)]
    assert pc.stats()["took_len"] == len(HB)


def test_a_resident_fork_is_preferred_over_the_file():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)  # cold: restores T, republishes it
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)  # warm from the resident T
    assert s.restore_calls == [TOOLS_AT]  # once
    st = pc.stats()
    assert (st["disk_hits"], st["fork_hits"]) == (1, 1)


def test_a_restore_is_skipped_at_critical_pressure_and_the_file_kept():
    h = FakeHooks(trimmable=True)
    h.pressure_value = 4
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert (st["misses"], st["restore_skips"], st["disk_hits"]) == (1, 1, 0)
    assert s.restore_calls == [] and tuple(T) in s.files


def test_a_restore_makes_metal_room_by_dropping_resident_slots():
    class RoomHooks(FakeHooks):
        pc: PrefixCache

        def headroom(self):
            # Headroom is short by one slot until the map is empty.
            return 10 if self.pc.slots() else 10_000

    h = RoomHooks(trimmable=True)
    s = stored(T, size=100)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    h.pc = pc
    pc._plant(h.new_cache(), [7, 7, 7])  # someone else's resident slot
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert st["disk_hits"] == 1
    assert st["pressure_evictions"] == 1 and st["evictions"] >= 1
    assert [7, 7, 7] not in [x.held for x in pc.slots()]


def test_a_restore_is_skipped_when_headroom_is_short_and_nothing_can_be_dropped():
    h = FakeHooks(trimmable=True)
    h.headroom_value = 10
    s = stored(T, size=100)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert (st["misses"], st["restore_skips"]) == (1, 1) and s.restore_calls == []


def test_no_restore_is_attempted_after_a_failed_fork_clone():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    _fail_next_copy(h)
    with pytest.warns(UserWarning, match="fork clone failed"):
        pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert s.restore_calls == [TOOLS_AT]  # the first turn's, not a second attempt
    assert pc.stats()["misses"] == 1


def test_a_failed_restore_is_a_cold_miss_with_the_counters_untouched():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    s.fail_restore = True
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert (st["misses"], st["disk_hits"], st["restores"], st["took_kind"]) == (1, 0, 0, "")
    assert h.prefilled[0] == T and st["reused_tokens"] == 0  # prefilled from zero


def test_a_raising_restore_hook_is_a_cold_miss():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)

    def boom(*a, **k):
        raise MemoryError()

    h.restore = boom  # ty: ignore[invalid-assignment]
    # The fake store lets the loader's exception through; the real one
    # catches it and returns False. Either way the turn goes cold.
    with pytest.warns(UserWarning, match="fork restore failed"):
        assert pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B) == "text"
    assert pc.stats()["misses"] == 1


def test_a_restored_turn_whose_generation_fails_retries_cold_and_keeps_the_file():
    h = FakeHooks(trimmable=True, fail_once=True)
    s = stored(T)
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    with pytest.warns(UserWarning, match="retrying cold"):
        assert pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B) == "text"
    st = pc.stats()
    assert st["cold_retries"] == 1 and st["restores"] == 1
    assert (st["took_kind"], st["restore_seconds"]) == ("", 0.0)  # the line describes the retry
    assert tuple(T) in s.files


def test_restore_time_is_its_own_gauge(clock):
    class ReadingHooks(TimedHooks):
        def restore(self, path, header, cache, ids):
            self.clock.advance(1.5)
            super().restore(path, header, cache, ids)

    h = ReadingHooks(clock, trimmable=False)
    pc = PrefixCache(h, max_bytes=ROOMY, store=stored(T))
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    st = pc.stats()
    assert st["restore_seconds"] == pytest.approx(1.5)
    # an empty stop at T (2 s on the fake clock), to HB, then the rest
    assert st["prefill_seconds"] == pytest.approx(6.0)


def test_a_disabled_store_is_never_consulted():
    h = FakeHooks(trimmable=True)
    s = stored(T)
    s.state = "unavailable"
    pc = PrefixCache(h, max_bytes=ROOMY, store=s)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert pc.stats()["misses"] == 1
```

In `FakeStore.longest_prefix`, honour `state`: add `if self.state != "active": return None` as its first line.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "restore or disk or longest_stored or preferred_over" -v 2>&1 | tail -16`
Expected: the new tests fail (`disk_hits == 0`, `misses == 1`, `restore_calls == []`).

- [ ] **Step 3: Add the restore**

In `generate`, in the fork-clone branch, record the failure: change

```python
            warm = self._copy_of(stats, slot.cache, "fork clone failed", "prefilling cold")
            if warm is None:
```

so that the `if warm is None:` body also sets `clone_failed = True`; initialise `clone_failed = False` beside `warm: list | None = None` and `kind = ""` above the branch.

Replace the miss branch (`else:` after `if slot is not None:`):

```python
        else:
            # A failed clone was an allocation failure of the size a restore
            # would ask for again; go straight to the cold path then.
            restored = None if clone_failed else self._restore_from_disk(stats, stable_ids)
            if restored is not None:
                warm, reuse = restored
                stats.took_len = reuse
                stats.took_kind = "disk"
                stats.hits += 1
                stats.disk_hits += 1
                stats.reused_tokens += reuse
            else:
                reuse = 0
                stats.misses += 1
                # Scanned here, after the clone, so a fork whose copy failed is
                # measured like any other miss instead of keeping an earlier one's.
                stats.miss_lcp = max(
                    (common_prefix_length(held, stable_ids) for held in candidates), default=0
                )
                warm = hooks.new_cache()
```

Add the method after `_persist`:

```python
    def _restore_from_disk(
        self, stats: PromptCacheStats, stable_ids: list[int]
    ) -> tuple[list, int] | None:
        """The longest stored prefix of `stable_ids`, read into a fresh cache
        on this thread — the turn's own working cache, uncharged, covered by
        the reserve — or None for a plain miss.

        The one place a whole fork is allocated in one step, so the two
        readings the valve uses are consulted first: while Metal's headroom
        is short of the file, drop least-recently-used resident slots (a miss
        took nothing, so nothing is protected); if it is still short, or the
        kernel is at critical, skip — the file stays, the turn prefills
        cold. Residency is not made here: the boundary loop republishes a
        resident fork at the restored length through its charged path.
        Runs before the probe is resolved, so a turn warm at a restored
        tools fork still publishes its own header fork."""
        store = self._store
        if store is None or store.state != "active":
            return None
        entry = store.longest_prefix(stable_ids)
        if entry is None:
            return None
        hooks = self._hooks
        started = _clock()
        try:
            while (headroom := hooks.headroom()) is not None and headroom < entry.size:
                with self._lock:
                    if not self._evictable(()):
                        break
                    self._drop_lru((), pressure=True)
            headroom = hooks.headroom()
            level = hooks.pressure()
            if (headroom is not None and headroom < entry.size) or (
                level is not None and level >= KERNEL_PRESSURE_CRITICAL
            ):
                stats.restore_skips += 1
                return None
            cache = hooks.new_cache()
            ids = list(stable_ids[: entry.n])
            if not store.restore(
                entry, lambda path, header: hooks.restore(path, header, cache, ids)
            ):
                return None
            stats.restores += 1
            return cache, entry.n
        except Exception as e:  # noqa: BLE001 — an optimization must never fail a turn
            warnings.warn(
                f"sous prompt cache: fork restore failed ({type(e).__name__}); prefilling cold",
                stacklevel=2,
            )
            return None
        finally:
            stats.restore_seconds += _clock() - started
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_promptcache.py 2>&1 | tail -3`
Expected: 14 more than after Task 5, all green.

- [ ] **Step 5: Whole suite, lint, type-check, commit**

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: `1204 + 14 (Task 1) + 29 (Task 2) + 11 (Task 3) + 3 + 12 + 14 = 1287 passed, 8 skipped, 33 deselected` (the forkio tests count as passed on this Mac, which has mlx).

```bash
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): serve a miss from a fork on disk" -m "The longest stored prefix is read into the turn's own cache on its own thread after the memory readings say there is room; residency comes back through the boundary loop, so the machine never holds a restored cache, a planted slot and a copy at once. A failed clone, critical pressure or short headroom leave the file and take the cold path." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: config — `[model].prompt_cache_disk_gb`

**Files:**
- Modify: `src/sous/config.py` (`_KNOWN` ~17-30, `SousConfig` ~113, new `_prompt_cache_disk_gb` beside `_prompt_cache_gb` ~322, `load_config` ~441-461), `src/sous/server.py` (`status_document` config block ~100-111)
- Test: `tests/test_config.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `SousConfig.prompt_cache_disk_gb: float | None` (None = auto, 0.0 = off, else GiB); the status document's `config.prompt_cache_disk_gb`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` after the `prompt_cache_gb` tests:

```python
# ---- [model].prompt_cache_disk_gb ---------------------------------------------


def test_prompt_cache_disk_gb_is_a_known_key_defaulting_to_auto(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprompt_cache_disk_gb = 'auto'\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert not [w for w in caught if "unknown" in str(w.message).lower()]
    assert cfg.prompt_cache_disk_gb is None
    assert load_config(tmp_path / "missing.toml").prompt_cache_disk_gb is None


def test_prompt_cache_disk_gb_accepts_a_number_and_zero(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprompt_cache_disk_gb = 24\n")
    assert load_config(p).prompt_cache_disk_gb == 24.0
    p.write_text("[model]\nprompt_cache_disk_gb = 0\n")
    assert load_config(p).prompt_cache_disk_gb == 0.0


@pytest.mark.parametrize("bad", ["-1", "true", "'lots'", "nan", "inf", "1e308"])
def test_prompt_cache_disk_gb_rejects_garbage_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nprompt_cache_disk_gb = {bad}\n")
    with pytest.warns(UserWarning, match=r"\[model\]\.prompt_cache_disk_gb"):
        cfg = load_config(p)
    assert cfg.prompt_cache_disk_gb is None


def test_prompt_cache_disk_gb_beside_a_disabled_cache_warns(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprompt_cache = false\nprompt_cache_disk_gb = 8\n")
    with pytest.warns(UserWarning, match="prompt_cache_disk_gb has no effect"):
        load_config(p)
```

In `tests/test_server.py`, two tests pin the `config` block: the one asserting `service.status_document()["config"] == {...}` (~line 353) gains `"prompt_cache_disk_gb": None,` after `"max_context_tokens": 131072,`, and the one asserting `set(doc["config"]) == {...}` (~line 393) gains `"prompt_cache_disk_gb",`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k disk -v 2>&1 | tail -8`
Expected: `unknown key 'prompt_cache_disk_gb'` warnings surface and `AttributeError: 'SousConfig' object has no attribute 'prompt_cache_disk_gb'`.

- [ ] **Step 3: Add the knob**

In `_KNOWN["model"]` add `"prompt_cache_disk_gb"`. In `SousConfig`, after `prompt_cache_gb`:

```python
    # Fork slots kept on disk under <data_dir>/forks, in GiB, so a tools fork
    # survives an idle unload, a restart and a reboot. None means automatic:
    # a quarter of the volume's free space, capped at 16 GiB (four to five
    # forks of the default model). 0 keeps nothing on disk.
    prompt_cache_disk_gb: float | None = None
```

After `_prompt_cache_gb`:

```python
def _prompt_cache_disk_gb(model: dict) -> float | None:
    """[model].prompt_cache_disk_gb: "auto" (None) or a non-negative number
    of GiB, the prompt_cache_gb shape. Anything else warns and means auto."""
    value = model.get("prompt_cache_disk_gb", "auto")
    if value == "auto":
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not math.isfinite(value * (1 << 30))
        or value < 0
    ):
        warnings.warn(
            f'sous config: [model].prompt_cache_disk_gb {value!r} must be "auto" or a '
            'non-negative number of GiB; using "auto"',
            stacklevel=3,
        )
        return None
    if model.get("prompt_cache") is False and value:
        warnings.warn(
            "sous config: [model].prompt_cache_disk_gb has no effect while "
            "[model].prompt_cache is false",
            stacklevel=3,
        )
    return float(value)
```

In `load_config`, after `prompt_cache_gb=_prompt_cache_gb(model),` add `prompt_cache_disk_gb=_prompt_cache_disk_gb(model),`. In `server.py`'s `status_document`, add `"prompt_cache_disk_gb": self.config.prompt_cache_disk_gb,` to the `config` block after `"max_context_tokens"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_config.py tests/test_server.py 2>&1 | tail -3`
Expected: 9 more than before across the two files (4 functions, one with 6 params), all green.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add src/sous/config.py src/sous/server.py tests/test_config.py tests/test_server.py
git commit -m "feat(config): [model].prompt_cache_disk_gb" -m "The prompt_cache_gb shape: auto, a number of GiB, or 0 for off. Auto is derived from the volume at engine construction so a default-on feature cannot claim a fixed slice of a disk sous knows nothing about." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: engines and the factory — build the store, keep `sous tune` out

**Files:**
- Modify: `src/sous/engine/forkstore.py` (add `fork_key_fields`), `src/sous/engine/vlm.py` (`__init__` ~50-104, hooks section ~184), `src/sous/engine/lm.py` (`__init__` ~14-46, hooks ~108), `src/sous/engine/base.py` (`fetch_model_config` ~220, `_default_factory` ~228-297, `default_engine_factory` ~300-318), `src/sous/tune/suite/runner.py:357`, `src/sous/tune/bench.py:332`
- Test: `tests/test_forkstore.py`, `tests/test_engine_base.py`, `tests/test_engine_forkhooks.py` (new), `tests/test_tune_runner.py`, `tests/test_tune_bench.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: `forkstore.fork_key_fields(*, backend, backend_version, weights, gpu, positions, int8_status) -> dict[str, str]`; `base.weights_identity_for(model_id) -> str`; `_default_factory(..., fork_dir: Path | None = None, fork_budget: int | None = None)`; `default_engine_factory(config, *, forks: bool = True)`; `VLMEngine(..., fork_dir=None, fork_budget=None, weights_identity="")` and the same three on `LMEngine`; `engine.fork_store: ForkStore | None`; `_fork_key_fields(weights)` (the three hooks arrived in Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forkstore.py`:

```python
def test_fork_key_fields_carry_everything_that_changes_the_kv(monkeypatch):
    from sous.engine.forkstore import FORK_LAYOUT, fork_key_fields

    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    f = fork_key_fields(
        backend="vlm",
        backend_version="0.7.1",
        weights="sha",
        gpu="applegpu_g16s",
        positions="engine",
        int8_status={"state": "active", "reason": None, "routed": 336},
    )
    assert f["backend"] == "vlm" and f["backend_version"] == "0.7.1"
    assert f["mlx"] and f["weights"] == "sha" and f["gpu"] == "applegpu_g16s"
    assert f["epoch"] == engine_epoch() and f["layout"] == str(FORK_LAYOUT)
    assert f["positions"] == "engine" and f["int8"] == "active:336"
    assert f["env"] == "MLX_ENABLE_TF32=1"
    off = fork_key_fields(
        backend="lm", backend_version="0.31.3", weights="sha", gpu="g", positions="model",
        int8_status={"state": "unavailable", "reason": "no tensor units", "routed": 0},
    )
    assert off["int8"] == "off"  # off and unavailable ran the same numerics
    monkeypatch.delenv("MLX_ENABLE_TF32")
    assert "env" not in fork_key_fields(
        backend="lm", backend_version="x", weights="w", gpu="g", positions="model",
        int8_status={"state": "off", "reason": None, "routed": 0},
    )
```

Create `tests/test_engine_forkhooks.py`:

```python
"""The engines' three fork hooks are one-line delegations to forkio, and
the identity key they build reads the engine's own facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sous.engine import forkio, lm, vlm


@pytest.mark.parametrize("cls", [vlm.VLMEngine, lm.LMEngine])
def test_hooks_delegate_to_forkio(monkeypatch, cls, tmp_path: Path):
    calls: list[tuple] = []
    monkeypatch.setattr(forkio, "eval_cache", lambda cache: calls.append(("eval", cache)))
    monkeypatch.setattr(
        forkio, "persist_cache", lambda cache, path, ids, meta: calls.append(("persist", path))
    )
    monkeypatch.setattr(
        forkio, "restore_cache", lambda path, header, cache, ids: calls.append(("restore", path))
    )
    engine = cls.__new__(cls)  # no model: only the hooks are exercised
    engine.eval_cache(["c"])
    engine.persist(["c"], tmp_path / "f", [1], {})
    engine.restore(tmp_path / "f", None, ["c"], [1])
    assert [c[0] for c in calls] == ["eval", "persist", "restore"]


def test_vlm_key_fields_read_positions_and_the_realised_int8_state(monkeypatch):
    engine = vlm.VLMEngine.__new__(vlm.VLMEngine)
    engine._positional = True
    engine.int8_prefill_status = {"state": "active", "reason": None, "routed": 12}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "vlm" and fields["positions"] == "engine"
    assert fields["int8"] == "active:12" and fields["weights"] == "sha"
    assert fields["gpu"]  # mx.device_info()["architecture"] on this Mac


def test_lm_key_fields_name_the_lm_backend(monkeypatch):
    engine = lm.LMEngine.__new__(lm.LMEngine)
    engine.int8_prefill_status = {"state": "off", "reason": None, "routed": 0}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off"
```

In `tests/test_engine_base.py`, `test_default_engine_factory_maps_every_model_value_onto_the_backend` (~line 1812) pins `_default_factory`'s kwargs exactly: add `"fork_dir": None,` and `"fork_budget": None,` to its expected dict (that config has `prompt_cache=False`, so no directory is handed over). Then append (beside `test_default_factory_threads_the_cache_budget_and_reserve`):

```python
def test_default_factory_threads_fork_settings_and_the_weights_identity(monkeypatch, tmp_path):
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(
        base,
        "fetch_model_config",
        lambda mid: {"model_type": "qwen3", "num_hidden_layers": 2, "num_key_value_heads": 1,
                     "head_dim": 8},
    )
    monkeypatch.setattr(base, "weights_identity_for", lambda mid: f"weights-of-{mid}")
    base._default_factory(
        "m", prompt_cache=True, cache_budget=7, reserve_tokens=1000,
        fork_dir=tmp_path / "forks", fork_budget=5,
    )
    assert seen["fork_dir"] == tmp_path / "forks" and seen["fork_budget"] == 5
    assert seen["weights_identity"] == "weights-of-m"
    seen.clear()
    base._default_factory("m", prompt_cache=True, cache_budget=7, reserve_tokens=1000)
    assert seen["fork_dir"] is None and seen["weights_identity"] == ""


def test_default_factory_keeps_forks_off_disk_when_the_kv_cost_is_unknown(monkeypatch, tmp_path):
    from sous.engine import base, lm

    seen = {}

    class FakeLM:
        def __init__(self, model_id, **kw):
            seen.update(kw)

    monkeypatch.setattr(lm, "LMEngine", FakeLM)
    monkeypatch.setattr(base, "fetch_model_config", lambda mid: {"model_type": "qwen3"})
    monkeypatch.setattr(base, "weights_identity_for", lambda mid: "w")
    with pytest.warns(UserWarning, match="forks are not persisted"):
        base._default_factory(
            "m", prompt_cache=True, cache_budget=0, reserve_tokens=1000, fork_dir=tmp_path
        )
    assert seen["fork_dir"] is None


def test_default_engine_factory_hands_the_engine_a_directory_only_for_the_daemon(
    monkeypatch, tmp_path
):
    from sous.engine import base

    seen = {}

    def fake_default_factory(model_id, *args, **kwargs):
        seen.update(kwargs)
        return FakeEngine([])

    monkeypatch.setattr(base, "_default_factory", fake_default_factory)
    cfg = _cfg(tmp_path)
    base.default_engine_factory(cfg)("m")
    assert seen["fork_dir"] == cfg.data_dir / "forks" and seen["fork_budget"] is None
    base.default_engine_factory(cfg, forks=False)("m")
    assert seen["fork_dir"] is None
    base.default_engine_factory(_cfg(tmp_path, prompt_cache_disk_gb=0.0))("m")
    assert seen["fork_dir"] is None
    base.default_engine_factory(_cfg(tmp_path, prompt_cache=False))("m")
    assert seen["fork_dir"] is None
    base.default_engine_factory(_cfg(tmp_path, prompt_cache_disk_gb=2.0))("m")
    assert seen["fork_dir"] == cfg.data_dir / "forks" and seen["fork_budget"] == 2 << 30


def test_weights_identity_for_reads_the_cached_config_without_the_network(monkeypatch, tmp_path):
    from sous.engine import base

    snap = tmp_path / "models--x" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    seen = {}

    def fake_download(repo_id, filename, **kw):
        seen.update(kw)
        return str(snap / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert base.weights_identity_for("x") == "abc123"
    assert seen["local_files_only"] is True
```

Append to `tests/test_tune_runner.py`:

```python
def test_a_suite_arm_gets_no_fork_store(monkeypatch, tmp_path):
    import sous.tune.suite.runner as runner_mod

    seen = {}

    def fake_default_engine_factory(config, **kw):
        seen.update(kw)
        return lambda mid: FakeEngine([FINISH])

    monkeypatch.setattr(runner_mod, "default_engine_factory", fake_default_engine_factory)
    run_suite(
        _arm(tmp_path),
        [_task()],
        runs=1,
        done=set(),
        record=lambda r: None,
        scratch=tmp_path / "scratch",
        out=lambda *a: None,
        factory=None,
        python=PYTHON,
        active_memory=lambda: 0,
    )
    assert seen == {"forks": False}
```

Append to `tests/test_tune_bench.py`:

```python
def test_a_bench_arm_gets_no_fork_store(monkeypatch, tmp_path):
    import sous.tune.bench as bench_mod

    factory, engines = _fake()
    seen = {}

    def fake_default_engine_factory(config, **kw):
        seen.update(kw)
        return factory

    monkeypatch.setattr(bench_mod, "default_engine_factory", fake_default_engine_factory)
    row = bench_arm(
        _arm(tmp_path),
        factory=None,
        repeat=1,
        peak_memory=lambda: 4096,
        reset_peak=lambda: None,
        active_memory=lambda: 100,
        out=lambda *a, **k: None,
    )
    assert row.ok and seen == {"forks": False}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_forkstore.py tests/test_engine_forkhooks.py tests/test_engine_base.py tests/test_tune_runner.py tests/test_tune_bench.py -k "fork or weights_identity" -v 2>&1 | tail -12`
Expected: `ImportError: fork_key_fields`, `AttributeError: 'VLMEngine' object has no attribute 'eval_cache'`, `AttributeError: weights_identity_for`, `TypeError: unexpected keyword 'forks'`.

- [ ] **Step 3: Implement**

`forkstore.py`, after `key_name`:

```python
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
```

`vlm.py`: add `from sous.engine.forkstore import ForkStore, fork_key_fields` to the imports (mlx-free at module level; `Path` and `forkio` arrived with the hooks). Constructor: add parameters `fork_dir: Path | None = None`, `fork_budget: int | None = None`, `weights_identity: str = ""` after `int8_prefill`. After the drafter block and before the budget is measured:

```python
        # Forks on disk, when the factory handed us a directory: built after
        # the drafter and int8 have settled, since the key reads both. Tests
        # and sous tune never pass one, so nothing they build touches ~/.sous.
        self.fork_store: ForkStore | None = None
        if fork_dir is not None and prompt_cache:
            self.fork_store = ForkStore(
                fork_dir, self._fork_key_fields(weights_identity), fork_budget
            )
```

and pass `store=self.fork_store` to `PrefixCache(...)`. Add the method after `drafter`:

```python
    def _fork_key_fields(self, weights: str) -> dict[str, str]:
        from importlib.metadata import version

        import mlx.core as mx

        return fork_key_fields(
            backend="vlm",
            backend_version=version("mlx-vlm"),
            weights=weights,
            gpu=str(mx.device_info().get("architecture", "")),
            positions=self.positions,
            int8_status=self.int8_prefill_status,
        )
```

`lm.py`: add `from sous.engine.forkstore import ForkStore, fork_key_fields` beside the other `sous.engine` imports; the same three parameters, the same store construction (before `PrefixCache`, after `int8_prefill_status`), and:

```python
    def _fork_key_fields(self, weights: str) -> dict[str, str]:
        from importlib.metadata import version

        import mlx.core as mx

        # mlx-lm's text models take their positions from the cache offset:
        # the model's, always.
        return fork_key_fields(
            backend="lm",
            backend_version=version("mlx-lm"),
            weights=weights,
            gpu=str(mx.device_info().get("architecture", "")),
            positions="model",
            int8_status=self.int8_prefill_status,
        )
```

`base.py`: add `from pathlib import Path` to the imports; after `fetch_model_config`:

```python
def weights_identity_for(model_id: str) -> str:
    """The identity of the weights behind `model_id`, from the config.json
    fetch_model_config has just cached: `local_files_only` makes this a
    cache lookup, never a second Hub round-trip, and the un-resolved path's
    parent is the snapshot commit."""
    from huggingface_hub import hf_hub_download

    from sous.engine.forkstore import weights_identity

    return weights_identity(Path(hf_hub_download(model_id, "config.json", local_files_only=True)))
```

`_default_factory` gains `fork_dir: Path | None = None, fork_budget: int | None = None` after `int8_prefill`. In the `bytes_per_token is None` branch, extend the warning: the message ends `"... (set [model].prompt_cache_gb to override)" if auto else ""` — append `+ (" and prompt-cache forks are not persisted" if fork_dir is not None else "")` and set `fork_dir = None` in that branch (the pressure valve is off; the store must not be the one thing allocating a fork at a stroke). Then `weights = weights_identity_for(model_id) if fork_dir is not None else ""`, and pass `fork_dir=fork_dir, fork_budget=fork_budget, weights_identity=weights` to both constructors.

`default_engine_factory(config: SousConfig, *, forks: bool = True)`:

```python
    disk = config.prompt_cache_disk_gb
    fork_dir = (
        config.data_dir / "forks" if forks and config.prompt_cache and disk != 0 else None
    )
    fork_budget = None if disk is None else int(disk * (1 << 30))
```

passed as `fork_dir=fork_dir, fork_budget=fork_budget`. Docstring: "`forks=False` is what `sous tune` passes: its arms would write synthetic forks into the live store's LRU, and a 2 s write inside a timed prefill moves the number the run reports."

`runner.py:357`: `base = factory or default_engine_factory(arm.config, forks=False)`. `bench.py`: add `default_engine_factory` to the `sous.engine.base` import and change `_measure`'s first lines to `factory = factory or default_engine_factory(arm.config, forks=False)` before `EngineManager(arm.config, engine_factory=factory)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: 11 more than after Task 7 (1 + 4 + 4 + 1 + 1; the forkhooks delegation test is two cases), all green; the existing `test_default_factory_*` tests still pass (new kwargs default off) and `test_default_engine_factory_maps_every_model_value_onto_the_backend` passes with its two new keys.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add src/sous/engine/forkstore.py src/sous/engine/vlm.py src/sous/engine/lm.py src/sous/engine/base.py src/sous/tune/suite/runner.py src/sous/tune/bench.py tests/test_forkstore.py tests/test_engine_forkhooks.py tests/test_engine_base.py tests/test_tune_runner.py tests/test_tune_bench.py
git commit -m "feat(engine): build the fork store from the engine's identity and wire the hooks" -m "Only the daemon's factory hands an engine a directory, so every direct construction — the test suite, sous tune's arms — stays out of ~/.sous. The key is read from the engine after load: positions owner, realised int8 state, versions, GPU and the weights' snapshot, with no second Hub round-trip." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: the surfaces — turn line, status document, `sous status`, `sous top`

**Files:**
- Modify: `src/sous/api/turn.py` (`TurnResult` ~61-100, `_turn` return ~296-330), `src/sous/api/routes.py` (`_turn_summary` ~188-233, `_turn_line` ~236-258), `src/sous/cli.py` (`status_lines` ~372-420), `src/sous/tui.py` (~1080-1086)
- Test: `tests/test_api_turn.py`, `tests/test_api_routes.py`, `tests/test_cli.py`, `tests/test_tui.py`

**Interfaces:**
- Produces: `TurnResult.from_disk: bool`, `TurnResult.persist_seconds: float`, `TurnResult.restore_seconds: float`; summary keys `persist_s`, `restore_s`; `cache=disk`; `sous status` line `forks on disk: …`; `sous top` wide `reused` line suffix ` · disk N`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_turn.py`:

```python
def test_a_disk_restore_is_reported_as_from_disk_with_its_seconds(tmp_path: Path):
    inner = FakeEngine(["one", "two"])
    inner.stats = {"hits": 0, "disk_hits": 0, "took_kind": ""}
    runner, _ = _runner(tmp_path, inner)
    first = runner.run(MSGS, [], 8, RecordingSink())
    assert (first.from_disk, first.persist_seconds, first.restore_seconds) == (False, 0.0, 0.0)

    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        inner.stats = {
            "hits": 1,
            "disk_hits": 1,
            "reused_tokens": 50312,
            "took_kind": "disk",
            "took_len": 50312,
            "restore_seconds": 1.4,
            "persist_seconds": 2.1,
        }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    second = runner.run(MSGS, [], 8, RecordingSink())
    assert (second.cache_hit, second.forked, second.from_disk) == (True, False, True)
    assert (second.took_kind, second.took_len) == ("disk", 50312)
    assert (second.persist_seconds, second.restore_seconds) == (2.1, 1.4)
```

In `tests/test_api_routes.py`, add `"persist_seconds": 0.0, "restore_seconds": 0.0, "disk_hits": 0,` to `_ALL_GAUGES`, assert `f["persist_s"] == "0.0" and f["restore_s"] == "0.0"` in `test_the_turn_line_attributes_a_hit`, and append:

```python
def test_the_turn_line_names_a_disk_restore(tmp_path: Path, capsys):
    inner = FakeEngine(["first", "reply"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {
                **_ALL_GAUGES,
                "hits": 1,
                "disk_hits": 1,
                "reused_tokens": 50312,
                "prefilled_tokens": 574,
                "took_len": 50312,
                "took_kind": "disk",
                "forks": 1,
                "restore_seconds": 1.4,
                "persist_seconds": 0.0,
                "prefill_seconds": 1.5,
                "decode_seconds": 2.0,
            }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    app = _app(tmp_path, inner)
    assert _post(app, _body(system="Be terse.", tools=[READ_TOOL])).status_code == 200
    r = _post(app, _body(system="Be terse.", tools=[READ_TOOL]))
    assert r.status_code == 200
    f = _fields(_turn_lines(capsys.readouterr().err)[1])
    assert f["cache"] == "disk" and f["took"] == "disk@50312"
    assert f["reused_tokens"] == "50312" and f["forks"] == "1"
    assert f["restore_s"] == "1.4" and f["persist_s"] == "0.0"
    assert f["prefill_tps"] == "382.7"  # the restore is not in the prefill rate
    assert "lcp" not in f
```

Append to `tests/test_engine_base.py` (beside `test_status_carries_the_int8_prefill_view_when_the_engine_reports_one`):

```python
@pytest.mark.parametrize("state", ["off", "active", "unavailable"])
def test_status_carries_the_disk_block_the_cache_reports(tmp_path, state):
    inner = FakeEngine([])
    inner.stats = {
        "hits": 0,
        "disk": {"state": state, "reason": None, "forks": 0, "bytes": 0, "budget_bytes": 0,
                 "evictions": 0},
    }
    manager = EngineManager(_cfg(tmp_path), engine_factory=lambda mid: inner)
    manager.get()
    assert manager.status()["prompt_cache"]["disk"]["state"] == state
```

In `tests/test_cli.py` beside the `status_lines` tests (grep `def test_status_lines`), append:

```python
def test_status_lines_report_forks_on_disk_when_the_store_is_active_or_unavailable():
    from sous.cli import status_lines

    doc = _document()
    doc["engine"]["prompt_cache"]["disk"] = {
        "state": "active", "reason": None, "forks": 3, "bytes": int(9.4 * (1 << 30)),
        "budget_bytes": 16 << 30, "evictions": 0,
    }
    lines = status_lines(doc, now=0.0)
    assert "  forks on disk: 3 · 9.4 GB" in lines
    doc["engine"]["prompt_cache"]["disk"] = {
        "state": "unavailable", "reason": "No space left on device (ENOSPC)", "forks": 0,
        "bytes": 0, "budget_bytes": 0, "evictions": 0,
    }
    assert "  forks on disk: unavailable (No space left on device (ENOSPC))" in status_lines(
        doc, now=0.0
    )
    doc["engine"]["prompt_cache"]["disk"] = {"state": "off", "reason": None, "forks": 0,
                                             "bytes": 0, "budget_bytes": 0, "evictions": 0}
    assert not [x for x in status_lines(doc, now=0.0) if "forks on disk" in x]
```

In `tests/test_tui.py`, add to `_doc`'s `prompt_cache` dict (~line 44): `"disk": {"state": "active", "reason": None, "forks": 3, "bytes": int(9.4 * (1 << 30)), "budget_bytes": 16 << 30, "evictions": 0},`. In the wide-layout test that asserts `"REHEAT  41 hit    DRAWER 18 fork" in _plain(app, "#line-body")` (~line 434), add on the next line:

```python
        assert " reused 1,820,000 tok · disk 3" in _plain(app, "#line-body")
```

The same test asserts every row of the wide body fits in 35 columns; the disk count rides on the `reused` line because the `PROMPT CACHE` line is 32 columns already.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_api_turn.py tests/test_api_routes.py tests/test_cli.py tests/test_tui.py -k "disk or attributes_a_hit" -v 2>&1 | tail -8`
Expected: `AttributeError: from_disk`, `KeyError: 'persist_s'`, the `forks on disk` line absent.

- [ ] **Step 3: Implement**

`turn.py`, `TurnResult`: after `forked`, `from_disk: bool = False  # the hit was served by a fork read off disk`; after `decode_seconds`, `persist_seconds: float = 0.0  # writing a fork at a boundary; in no phase` and `restore_seconds: float = 0.0  # reading a fork off disk; in no phase`. In `_turn`'s `return TurnResult(...)`: `from_disk=delta_of("disk_hits") > 0,`, `persist_seconds=after.get("persist_seconds", 0.0),`, `restore_seconds=after.get("restore_seconds", 0.0),`.

`routes.py`, `_turn_summary`: `cache = "disk" if result.from_disk else "fork" if result.forked else "hit" if result.cache_hit else "miss"`; add `"persist_s": result.persist_seconds,` and `"restore_s": result.restore_seconds,` after `"decode_s"`. `_turn_line`: after the `f"prefill_s={s['prefill_s']:.1f} decode_s={s['decode_s']:.1f} "` literal insert a new adjacent literal on its own line, `f"persist_s={s['persist_s']:.1f} restore_s={s['restore_s']:.1f} "` (the existing line is near 100 columns; `ruff format` never splits a string). Any other reader of the summary that lists its keys (the TUI's ticket text, `_recent_turn_text` in `cli.py`) needs nothing: they read named keys.

`cli.py`, `status_lines`: after the engine line is appended and before the in-flight block:

```python
    disk = (engine.get("prompt_cache") or {}).get("disk")
    if isinstance(disk, dict) and disk.get("state") == "active":
        gb = (disk.get("bytes") or 0) / (1 << 30)
        lines.append(f"  forks on disk: {disk.get('forks', 0)} · {gb:.1f} GB")
    elif isinstance(disk, dict) and disk.get("state") == "unavailable":
        lines.append(f"  forks on disk: unavailable ({disk.get('reason') or '?'})")
```

`tui.py`, the wide layout: replace the `reused` line with

```python
                f" reused {c.get('reused_tokens', 0):,} tok" + _disk_suffix(c.get("disk")),
```

give the new outcome a kitchen word — `CACHE_WORDS = {"hit": "REHEAT", "fork": "WARM DRAWER", "miss": "SCRATCH", "disk": "PANTRY"}` (`RAIL_CACHE_WORDS` inherits it; the two `LEGEND_STRIP` strings are width-bound and stay as they are) and a `LEGEND` row after `WARM DRAWER`: ` PANTRY         disk         a fork read off disk into the turn's own cache` — and add a module-level helper near `turn_tallies`:

```python
def _disk_suffix(disk: object) -> str:
    """` · disk 3` when forks are kept on disk; nothing otherwise. The wide
    panel's line count is fixed and every row must fit 35 columns, so the
    count rides on the shortest line and the bytes stay with `sous status`."""
    if not isinstance(disk, dict) or disk.get("state") != "active":
        return ""
    return f" · disk {disk.get('forks', 0)}"
```

Then regenerate the committed render, which the `_doc` change and the new suffix alter: `SOUS_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_tui.py -k renders_as_committed`, then run the same test without the variable and review the diff of `tests/snapshots/sous-top-100x30.svg` (the wide body's `reused` row gains ` · disk 3`; nothing else moves).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: 6 more than after Task 8 (the disk-block test is three cases), all green — 1313 passed on this Mac.

- [ ] **Step 5: Lint, type-check, commit**

```bash
git add src/sous/api/turn.py src/sous/api/routes.py src/sous/cli.py src/sous/tui.py tests/test_api_turn.py tests/test_api_routes.py tests/test_cli.py tests/test_tui.py tests/test_engine_base.py tests/snapshots/sous-top-100x30.svg
git commit -m "feat(api,cli): show forks on disk — cache=disk, persist_s/restore_s, status lines" -m "cache=fork keeps meaning a resident fork was copied so the log series stays comparable; the two new durations sit outside every phase, as the fork copy already does, so prefill_tps stays true on the one turn that shows the feature working." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: documentation, the plan-language sweep and the whole-branch check

**Files:**
- Modify: `README.md` ("Prompt cache and continuity" ~296-318, the turn-line paragraph ~396-420, the sample config ~493-506, the `prompt_cache_gb` paragraph ~543-596, "Security model" ~740-746), `CLAUDE.md` (the `#99` gotcha ending "only a daemon restart drops them" ~143-145, the prompt-cache-slots gotcha ~59-80, "Security boundary" ~261-277)

- [ ] **Step 1: README — "Prompt cache and continuity"**

Replace the two sentences `Forks live as long as the weights: \`[model].idle_unload_minutes\` drops them with the model.` (~315) with:

> The *tools* fork is also kept on disk: at the tools boundary a cold turn writes the live cache to `~/.sous/forks/` (~3.1 GiB at 50K tokens, about 2 s on the internal SSD) before taking its resident copy, and a miss whose render extends a stored file reads it back in seconds — across an idle unload, a daemon restart, an upgrade that leaves the engine's numerics alone, and a reboot. The header fork lives only as long as the weights (`[model].idle_unload_minutes` drops it with the model): measured on the maintainer's log, keeping it on disk too would have saved about eight seconds in nine days for a second 3.5 GiB write per cold turn. A file is used only when its ids are exactly a prefix of the render, whole or not at all, and only by a daemon whose backend, mlx and mlx-vlm versions, GPU, weights snapshot, engine sources, positions owner and int8 state match the ones that wrote it; anything else is a natural miss and ages out of the budget below.

- [ ] **Step 2: README — the turn line**

In the `took` sentence add, after `` `fork@N` for a shared boundary, ``: `` `disk@N` for a fork read off disk (`cache=disk`; the turn also republishes it as a resident fork, so its line reads `forks=1`), ``. In the `cache` sentence, after `or \`miss\``, insert `` — `disk` when the prefix came off disk ``. After the `decode_s` sentence (`... as far as the prompt cache measures it — not a strict partition of \`seconds\`:`) add to the list of things timed into neither phase: `` a fork written to disk (`persist_s`) or read from it (`restore_s`) is timed into neither phase either — both are their own fields, so `prefill_tps` stays a prefill rate on the turn that restores ``.

- [ ] **Step 3: README — configuration**

In the sample config, after `# prompt_cache_gb = 8` add `# prompt_cache_disk_gb = "auto"`. After the `prompt_cache_gb` paragraph (which ends `... so "every new session" means every new session inside that window.` — delete that last sentence, it is false now, and the one before it about forks living as long as the weights) add:

> `[model].prompt_cache_disk_gb` (default `"auto"` — a number sets the GiB, `0` keeps nothing on disk) bounds the tools forks kept under `~/.sous/forks/`, one file per tool array, least-recently-used first when the budget says so. `"auto"` is a quarter of the volume's free space at load, capped at 16 GiB — four to five forks of the default model; a write is skipped, with one warning, when it would leave the volume with less than 10 GiB (or 5 %) free. The daemon logs the directory, the budget and what it found at every load; `sous status` and `sous top` show the count and bytes, or the reason the store is off. `rm -rf ~/.sous/forks` is the eraser, safe under a running daemon. Exclude the directory from Time Machine: its files are worthless the moment the weights or a version change, and churn with every new tool array. Requires `prompt_cache = true`; a daemon that cannot size the model's KV keeps forks in memory only.

- [ ] **Step 4: README — "Security model"**

After `... and never logs a request body, header value or query string.` add:

> One thing derived from prompts is kept at rest: a persisted fork holds the KV and the token ids of the rendered tool block (and, for a header-only template, the system text) of the sessions that produced it, unencrypted, in `~/.sous/forks/` (mode 0700, files 0600). `prompt_cache_disk_gb = 0` turns it off; `rm -rf ~/.sous/forks` clears it.

- [ ] **Step 5: CLAUDE.md**

Replace `Slots built before this fix hold mispositioned keys: only a daemon restart drops them.` with: `A fork on disk written by a build with wrong KV would survive a restart, which is why the fork store's identity key hashes the engine sources (\`forkstore.engine_epoch\`: vlm.py, lm.py, int8prefill.py, kernels/*.metal and *.h) beside the versions, the GPU and the weights snapshot — any change to what a token id produces invalidates every file; \`rm -rf ~/.sous/forks\` is the manual eraser.`

In the prompt-cache-slots gotcha, after `... never rendering the system turn alone, which Qwen3.8's template refuses.` add: `The lowest boundary is also written to disk (\`engine/forkstore.py\`, \`engine/forkio.py\`): the live cache's full padded \`keys\`/\`values\` at the boundary, before the resident copy, outside the prefill timer, with its own handler; a miss reads the longest stored prefix into the turn's own cache on its own thread and the boundary loop republishes a resident fork there (\`reuse <= b\`). Restore a \`KVCache\` by assigning \`keys\`, \`values\`, then \`offset\` from metadata — never through \`state\`, whose setter derives the offset from the padded shape and mispositions every appended token — and \`mx.eval\` the arrays: a lazy array is invisible to the memory the pressure valve reads (fork copies are lazy too; \`eval_cache\` after every \`fork_copy\`). Save only evaluated, contiguous buffers: a slice or a transpose is materialised whole on save. Files are shared across owner threads; only arrays are owner-scoped. Only \`default_engine_factory\` hands an engine a store directory — tests and \`sous tune\` (\`forks=False\`) never reach \`~/.sous\`.`

In "Security boundary", after `... and stores none.` add: `The fork store (\`~/.sous/forks\`, 0700/0600) is the one place the daemon keeps prompt-derived data at rest: rendered tool-block and system-text token ids plus their KV, never a header or a credential; it logs counts and bytes, never a token.`

- [ ] **Step 6: Sweep, verify, commit**

Run: `git diff main -- src tests | grep -nE '^\+.*(Task [0-9]|Step [0-9]|[Dd]ecision [0-9]|[Ss]pec[: ]|docs/superpowers)'` — must print nothing.
Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3; uv run ty check 2>&1 | tail -2; uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: `1313 passed, 8 skipped, 33 deselected` on this Mac, no diagnostics, lockfile in sync.

```bash
git add README.md CLAUDE.md
git commit -m "docs: forks on disk — prompt cache, turn line, config, security model" -m "The two sentences that said forks live as long as the weights are now false for the tools fork; the #99 note that a restart clears bad KV is replaced by what the identity key guarantees." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: the model-marked test (local only)

**Files:**
- Test: `tests/test_engine_vlm.py` (append)

Runs on a Mac with the test models cached (`TINY_VLM`, `HYBRID_VLM` at the top of the file); never in CI. The witness is serialisation identity plus one extension compared on cached keys — never greedy text, and never against a cold one-pass prefill (a split prefill drifts on a hybrid's deeper layers).

- [ ] **Step 1: Write the test**

```python
@pytest.mark.model
@pytest.mark.parametrize(
    ("model_id", "is_hybrid"),
    [
        pytest.param(TINY_VLM, False, id="pure-attention"),
        pytest.param(HYBRID_VLM, True, id="linear-attention-hybrid"),
    ],
)
def test_vlm_a_fork_read_off_disk_matches_the_one_it_was_written_from(
    model_id, is_hybrid, tmp_path
):
    """Persist the live cache at a boundary, drop it, restore into a fresh
    cache: every layer's arrays equal, offsets equal, and one further chunk
    extends the restored cache exactly as it extends the source — compared
    on cached keys (every layer on a pure-attention model, the first
    attention layer on a hybrid, whose deeper layers drift between prefill
    shapes for reasons that have nothing to do with the file)."""
    import mlx.core as mx
    from mlx_vlm.models.cache import make_prompt_cache

    from sous.engine import forkio
    from sous.engine.forkstore import read_header
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(model_id, cache_budget=0)
    model, _ = e._loaded()
    ids = e._encode("def f(x):\n    return x + 1\n" * 40)
    header, tail = ids[: len(ids) // 2], ids[len(ids) // 2 :]

    src = make_prompt_cache(model.language_model)
    e.prefill(src, header)
    path = tmp_path / f"{len(header)}-x.safetensors"
    forkio.persist_cache(src, path, header, {"format": "sous-fork-v1", "layout": "1",
                                              "n_tokens": str(len(header)), "boundary": "tools"})
    restored = make_prompt_cache(model.language_model)
    forkio.restore_cache(path, read_header(path), restored, header)

    for a, b in zip(restored, src, strict=True):
        assert int(getattr(a, "offset", 0) or 0) == int(getattr(b, "offset", 0) or 0)
        for xa, xb in zip(a.state, b.state, strict=True):
            if xa is None or xb is None:
                assert xa is None and xb is None
                continue
            assert bool(mx.array_equal(xa, xb).item())

    e.prefill(restored, tail)
    e.prefill(src, tail)
    attention = [i for i, c in enumerate(src) if getattr(c, "keys", None) is not None]
    layers = attention[:1] if is_hybrid else attention
    for i in layers:
        got = restored[i].keys[..., : restored[i].offset, :].astype(mx.float32)
        want = src[i].keys[..., : src[i].offset, :].astype(mx.float32)
        d = mx.max(mx.abs(got - want))
        mx.eval(d)
        assert d.item() == 0.0
    e.unload()


@pytest.mark.model
def test_vlm_engine_serves_a_second_session_from_a_fork_on_disk(tmp_path):
    """End to end on the real tokenizer and template: a store with a fork
    written by one engine serves a fresh engine's first turn as cache=disk."""
    from sous.engine.promptcache import FORK_MIN_TOKENS
    from sous.engine.vlm import VLMEngine

    # Qwen2-VL's template renders no tools, so the system text carries the
    # boundary (the header, then — the only boundary, and the one written).
    tools: list[dict] = []
    system = {"role": "system", "content": "You are terse. " * (FORK_MIN_TOKENS // 3)}
    first = VLMEngine(TINY_VLM, prompt_cache=True, cache_budget=1 << 34,
                      fork_dir=tmp_path / "forks", weights_identity="test")
    first.generate([system, {"role": "user", "content": "Say A."}], tools, 4)
    assert first.prompt_cache_stats()["persists"] == 1
    first.unload()

    second = VLMEngine(TINY_VLM, prompt_cache=True, cache_budget=1 << 34,
                       fork_dir=tmp_path / "forks", weights_identity="test")
    second.generate([system, {"role": "user", "content": "Say B."}], tools, 4)
    s = second.prompt_cache_stats()
    assert (s["disk_hits"], s["restores"], s["misses"]) == (1, 1, 0)
    assert s["reused_tokens"] >= FORK_MIN_TOKENS and s["took_kind"] == "disk"
    assert s["disk"]["forks"] == 1 and s["forks"] >= 1  # republished as a resident fork
    second.unload()
```

- [ ] **Step 2: Run it locally**

Run: `uv run pytest -m model tests/test_engine_vlm.py -k "off_disk or fork_on_disk" -v 2>&1 | tail -6`
Expected: 3 passed (two parametrised cases and the end-to-end). Nothing under `~/.sous` is touched (`tmp_path` throughout).

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_vlm.py
git commit -m "test(engine): a fork read off disk equals the one it was written from" -m "Serialisation identity on every layer, then one extension compared on cached keys — the witness the positions work settled on; never greedy text." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: the live gate on the M5 Pro (orchestrator, after the maintainer's go-ahead)

Not a code task. Runs on the maintainer's M5 Pro over `ssh m5pro`, against a daemon built from this branch in the `~/code/personal/sous-remote` checkout — never the working copy — with the maintainer's own daemon stopped and restored afterwards (see the memory notes on restarting the unmanaged daemon: `sous stop`, wait for the pid in `daemon.lock` to go, start with a `fork`+`setsid` launcher, restore the installed `sous` at the end). Record every number in the PR.

Five arms plus a memory sample:

1. **Cold read.** Install the branch, start the daemon, run one `sous claude` session whose first subagent turn is cold (`cache=miss`, `persist_s` > 0, one file under `~/.sous/forks/<key>/`, the construction INFO line present). Stop the daemon. `sudo purge` (or reboot). Start the daemon, run a new session with the same subagent type: the first turn logs `cache=disk took=disk@~50K forks=1` with `restore_s` — the genuinely cold number the README's "in seconds" rests on. Pass: `restore_s` under 10 s and `prefilled_tokens` under 3,000.
2. **The write turn's rate.** On arm 1's cold turn, `prefill_tps` within 5 % of the log's cold-miss baselines (395 tok/s at 44.6K, 366 at 63.6K) and `persist_s` recorded (expect ~2 s). Sample `sysctl kern.memorystatus_vm_pressure_level` and `vm_stat | head -8` immediately before the turn, at the moment the file appears (`ls -l ~/.sous/forks/*/`) and ten seconds later, with the weights loaded and two resident forks. Pass: the level stays at 1 and `pressure=0` on the turn line. If it moves to 2, the fix is to skip the kernel half of `_evict_pressure` for the publish that follows a persist in the same turn (never the Metal half, never `F_NOCACHE`) — file it, do not improvise it in the gate.
3. **Dedupe across a restart.** Restart the daemon (no purge), same session type: `cache=disk`, `persist_s=0.0`, the file's inode and mtime unchanged apart from the `utime` touch (`stat -f '%i %m'` before and after), `forks=1`.
4. **`prompt_cache_gb = 0`.** Set it, restart, run a session: the first turn `cache=disk took=disk@N` with `sous status` showing `0 slots` after it, and `tokenize_s` on the line recorded (the probe now resolves at budget 0). Restore the config afterwards.
5. **`rm -rf` under the daemon.** With the daemon running, `rm -rf ~/.sous/forks`; run a cold session: the directory reappears, a file is written, `sous status` shows `forks on disk: 1 · …` and never `unavailable`.

Also confirm `sous tune --quick` on the branch leaves `~/.sous/forks` untouched (compare `ls -lR` before and after).

Afterwards: reinstall the maintainer's `sous` from `main`, restart the daemon, and leave `~/.sous/forks` in place unless the maintainer says otherwise.

---

## Self-review notes

- **Spec coverage.** §1 write path → Task 5; §2 format → Tasks 2–3 (decision 1 replaces `file_bytes` with the header-derived size); §3 key → Tasks 1 and 8 (`fork_key_fields`, `weights_identity_for`, `engine_epoch`); §4 restore → Task 6; §5 store rules → Task 2; §6 wiring → Tasks 7–8; §7 observability → Task 9 (the `/sous/events` wake is a stated non-goal); §8 `eval_cache` → Tasks 3–4; Invariants → the docstring edits in Task 4 and CLAUDE.md in Task 10; Testing → each task's tests, Task 11, Task 12; Documentation → Task 10.
- **Names used across tasks.** `ForkStore(root, key_fields, budget, *, pid_alive, disk_usage, pid)`; `store.has/longest_prefix/persist/restore/touch/status/state`; `Entry.n/size/path/digest`; `Header.metadata/tensors/expected_size`; hooks `eval_cache(cache)`, `persist(cache, path, ids, metadata)`, `restore(path, header, cache, ids)`; stats `persists/restores/disk_hits/restore_skips/persist_seconds/restore_seconds`; `stats()["disk"]`; `TurnResult.from_disk/persist_seconds/restore_seconds`; `_default_factory(..., fork_dir, fork_budget)`; `default_engine_factory(config, *, forks=True)`; engines `fork_dir/fork_budget/weights_identity`, `fork_store`, `_fork_key_fields(weights)`.
