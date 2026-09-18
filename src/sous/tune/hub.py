"""Hugging Face Hub reads the tune makes: sizes, cache presence, the
download plan, consent for each snapshot, and the downloads themselves."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from sous.tune.hardware import hub_cache_dir

_GB = 10**9
_GIB = 1 << 30
# What mlx-vlm's loader fetches from a repo (mlx_vlm.utils.get_model_path):
# the weights and the files that read them. The size a consent prompt quotes
# is the safetensors total, so the download must not reach past it to
# formats no loader here opens.
LOADER_PATTERNS = (
    "*.json",
    "*.jsonl",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
)


def _cached_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path | None:
    """The local snapshot directory, or None: a lookup that never touches the
    network, so an absent snapshot is a plain answer rather than a download.

    Answered from the cache's own files rather than the Hub's own
    completeness check (`snapshot_download(..., local_files_only=True)`):
    that check counts non-weight files an mlx-vlm download never fetches
    (a cached model reads as missing) and is satisfied by a snapshot that
    holds only a config.json fetched for metadata, with no weights on disk
    at all (an uncached model reads as ready to load, for free)."""
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir)
    if not isinstance(config, str):  # None (absent) or the _CACHED_NO_EXIST sentinel
        return None
    snapshot = Path(config).parent
    return snapshot if _weights_complete(snapshot) else None


def _weights_complete(snapshot: Path) -> bool:
    """Whether every weight shard the checkpoint names is present: the
    Hub's own completeness check counts README-style files a model loader
    never fetches, and a metadata-only entry (config.json fetched for the
    fit arithmetic) has a snapshot directory with no weights at all."""
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except ValueError, OSError:
            return False
        return bool(names) and all((snapshot / name).is_file() for name in names)
    return any(snapshot.glob("*.safetensors"))


def is_cached(repo_id: str, *, cached_path: Callable[[str], Path | None] | None = None) -> bool:
    return (cached_path or _cached_path)(repo_id) is not None


def _hub_api():
    from huggingface_hub import HfApi

    return HfApi()


def snapshot_bytes(
    repo_id: str, *, api=None, cached_path: Callable[[str], Path | None] | None = None
) -> int | None:
    """On disk when cached (every file, symlinks followed), else the Hub's
    safetensors sizes; None when neither can answer (offline, private)."""
    local = (cached_path or _cached_path)(repo_id)
    if local is not None:
        return sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, names in os.walk(local, followlinks=True)
            for f in names
            if f.endswith(".safetensors")
        )
    try:
        info = (api or _hub_api()).model_info(repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001 — offline or refused: unknown, not an error
        return None
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", "")
        if name.endswith(".safetensors"):
            total += int(getattr(sibling, "size", 0) or 0)
    return total or None


@dataclass(frozen=True)
class Download:
    repo_id: str
    bytes: int | None
    role: str
    removes: str


def plan_downloads(
    items: list[tuple[str, str, str]],
    *,
    cached: Callable[[str], bool] = is_cached,
    size: Callable[[str], int | None] = snapshot_bytes,
) -> list[Download]:
    """Every snapshot the arms need that is not on disk, once each, in first-
    mention order — so the user sees one block and answers each entry once."""
    plan: list[Download] = []
    seen: set[str] = set()
    for repo_id, role, removes in items:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        if cached(repo_id):
            continue
        plan.append(Download(repo_id, size(repo_id), role, removes))
    return plan


def _size(n: int | None) -> str:
    return "size unknown" if n is None else f"{n / _GB:.1f} GB"


def confirm(prompt: str, ask: Callable[[str], str] = input) -> bool:
    """One yes/no question; EOF and anything but a yes are a no."""
    try:
        answer = ask(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def ask_consent(
    plan: list[Download],
    free_bytes: int,
    *,
    hub_cache: str | None = None,
    ask: Callable[[str], str] = input,
    out: Callable[..., None] = print,
) -> set[str]:
    """The whole plan first, then one question per snapshot: a user decides
    each download knowing everything the run wants and what a no costs."""
    if not plan:
        return set()
    hub = hub_cache if hub_cache is not None else hub_cache_dir()
    out(f"Downloads needed (free disk {free_bytes / _GIB:.1f} GiB at {hub}):")
    for n, d in enumerate(plan, start=1):
        out(f"  {n}. {d.repo_id:45s} {_size(d.bytes):>13s}   {d.role}; refusing: {d.removes}")
    approved: set[str] = set()
    for n, d in enumerate(plan, start=1):
        if confirm(f"Download #{n} ({d.repo_id}, {_size(d.bytes)})? [y/N] ", ask):
            approved.add(d.repo_id)
    return approved


def fetch(
    repo_ids: Iterable[str],
    *,
    download: Callable[[str], object] | None = None,
    out: Callable[..., None] = print,
) -> None:
    """Every approved snapshot before any measurement, so a bench never
    stalls on the network mid-run."""
    if download is None:
        from huggingface_hub import snapshot_download

        download = functools.partial(snapshot_download, allow_patterns=list(LOADER_PATTERNS))
    done: set[str] = set()
    for repo_id in repo_ids:
        if repo_id in done:
            continue
        done.add(repo_id)
        out(f"downloading {repo_id} ...")
        download(repo_id)
