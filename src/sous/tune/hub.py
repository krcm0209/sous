"""Hugging Face Hub reads the tune makes: sizes, cache presence, the
download plan, consent for each snapshot, and the downloads themselves."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

_GB = 10**9
_GIB = 1 << 30


def _cached_path(repo_id: str) -> Path | None:
    """The local snapshot directory, or None: a lookup that never touches the
    network, so an absent snapshot is a plain answer rather than a download."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except LocalEntryNotFoundError, OSError, ValueError:
        return None


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
        if repo_id in seen or cached(repo_id):
            continue
        seen.add(repo_id)
        plan.append(Download(repo_id, size(repo_id), role, removes))
    return plan


def _size(n: int | None) -> str:
    return "size unknown" if n is None else f"{n / _GB:.1f} GB"


def ask_consent(
    plan: list[Download],
    free_bytes: int,
    *,
    ask: Callable[[str], str] = input,
    out: Callable[..., None] = print,
) -> set[str]:
    """The whole plan first, then one question per snapshot: a user decides
    each download knowing everything the run wants and what a no costs."""
    if not plan:
        return set()
    hub = os.path.join("~", ".cache", "huggingface", "hub")
    out(f"Downloads needed (free disk {free_bytes / _GIB:.1f} GiB at {hub}):")
    for n, d in enumerate(plan, start=1):
        out(f"  {n}. {d.repo_id:45s} {_size(d.bytes):>13s}   {d.role}; refusing: {d.removes}")
    approved: set[str] = set()
    for n, d in enumerate(plan, start=1):
        prompt = f"Download #{n} ({d.repo_id}, {_size(d.bytes)})? [y/N] "
        try:
            answer = ask(prompt)
        except EOFError:
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
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

        download = snapshot_download
    done: set[str] = set()
    for repo_id in repo_ids:
        if repo_id in done:
            continue
        done.add(repo_id)
        out(f"downloading {repo_id} ...")
        download(repo_id)
