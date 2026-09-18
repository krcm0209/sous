"""Where a tune run keeps what it measured: one directory per run under the
data dir, results appended as they land so an interrupted run can resume."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from pathlib import Path


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        self.run_id = path.name

    @classmethod
    def new(cls, base: Path, *, clock: Callable[[], float] = time.time) -> RunDir:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
        # The stamp has one-second resolution and nothing serialises tune
        # processes: two started in the same second must not share a
        # directory, or one reads the other's rows as its own.
        for n in itertools.count(1):
            path = base / (stamp if n == 1 else f"{stamp}-{n}")
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return cls(path)
        raise AssertionError("unreachable")

    @classmethod
    def existing(cls, base: Path, run_id: str) -> RunDir:
        path = base / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"no tune run {run_id!r} under {base}")
        return cls(path)

    @property
    def results(self) -> Path:
        return self.path / "results.jsonl"

    def append(self, kind: str, row: dict) -> None:
        # One line per result, flushed at once: a run killed mid-arm keeps
        # every arm that finished, and --resume reads exactly those.
        with self.results.open("a") as f:
            f.write(json.dumps({"kind": kind, **row}, allow_nan=False) + "\n")

    def rows(self, kind: str) -> list[dict]:
        if not self.results.exists():
            return []
        out: list[dict] = []
        for line in self.results.read_text().splitlines():
            if not line.strip():
                continue
            # A line a full disk or a kill cut short is a result that never
            # landed: --resume measures that arm again rather than failing
            # on the file it exists to read.
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.pop("kind", None) == kind:
                out.append(row)
        return out

    def write_json(self, name: str, obj: object) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.write_text(text)
        return p
