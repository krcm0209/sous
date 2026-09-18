"""What the user reads: the report, the config diff, and the apply."""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path

import tomlkit

from sous.tune.arms import Refusal
from sous.tune.bench import BenchRow
from sous.tune.candidates import Checkpoint
from sous.tune.decide import QuickChoice
from sous.tune.hardware import Hardware, gib

STALE_AFTER_DAYS = 90
_GB = 10**9
_KIB = 1024


def _fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def _quant(cp: Checkpoint) -> str:
    q = cp.quant
    if q.mode is None:
        return "unquantized"
    layout = f"{q.mode} {q.bits}-bit gs{q.group_size}"
    if q.overrides:
        layout += f", {q.overrides} layer overrides"
    verifier = "fast verifier" if q.verifier_fast else "slow verifier"
    if q.verifier_slow_layers:
        verifier += f" ({q.verifier_slow_layers} slow layers)"
    int8 = "int8-routable" if q.int8_routable else "no int8 prefill"
    return f"{layout}; {verifier}; {int8}"


def _candidate_line(cp: Checkpoint, refusal: Refusal | None) -> str:
    size = "size unknown" if cp.bytes is None else f"{cp.bytes / _GB:.1f} GB"
    kv = (
        "KV unknown"
        if cp.kv_bytes_per_token is None
        else f"{cp.kv_bytes_per_token // _KIB} KiB/token"
    )
    line = f"  {cp.id:45s} {size:>12s}  {kv:>13s}  {_quant(cp)}"
    if refusal is not None:
        line += f"\n      refused: {refusal.reason}"
    return line


def _row_line(row: BenchRow, current_model: str) -> str:
    tag = "" if row.model_id == current_model else "  [quality untested]"
    if not row.ok:
        return f"  {row.label:45s} failed: {row.error}{tag}"
    peak = "-" if row.peak_memory_bytes is None else gib(row.peak_memory_bytes)
    return (
        f"  {row.label:45s} prefill {_fmt(row.prefill_tps_2k)}/{_fmt(row.prefill_tps_16k)} tok/s"
        f"  decode {_fmt(row.decode_tps_1k)}/{_fmt(row.decode_tps_16k)} tok/s"
        f"  ttft {_fmt(row.ttft_seconds, ' s')}  peak {peak}"
        f"  spread {_fmt(None if row.spread is None else row.spread * 100, '%', 0)}"
        f"  load {_fmt(row.load_seconds, ' s')}{tag}"
    )


def render_report(
    *,
    hardware: Hardware,
    table_age_days: int,
    checkpoints: dict[str, Checkpoint],
    refusals: list[Refusal],
    rows: list[BenchRow],
    choice: QuickChoice | None,
    current_model: str,
) -> str:
    refused = {r.model_id: r for r in refusals if not r.drafter_id}
    out = ["# sous tune --quick", "", "## Hardware", ""]
    out.append(
        f"  {hardware.chip}, {gib(hardware.memory_bytes)} unified memory, Metal working set "
        f"{gib(hardware.working_set_bytes)}, macOS {hardware.macos}"
    )
    nax = (
        "NAX tensor units: yes" if hardware.nax else f"NAX tensor units: no ({hardware.nax_reason})"
    )
    out.append(f"  {nax}; " + ", ".join(f"{k} {v}" for k, v in hardware.versions.items()))
    out.append(f"  hub cache {hardware.hub_cache}: {gib(hardware.disk_free_bytes)} free")
    if table_age_days > STALE_AFTER_DAYS:
        out.append(
            f"  the candidate table is {table_age_days} days old and may be stale; "
            "`sous tune --discover` can look for newer checkpoints"
        )
    out += ["", "## Candidates", "  (prefill/decode at 2K/16K and 1K/16K tokens of context)", ""]
    for cp in checkpoints.values():
        out.append(_candidate_line(cp, refused.get(cp.id)))
    for r in refusals:
        if r.drafter_id:
            out.append(f"  {r.drafter_id:45s} dropped for {r.model_id}: {r.reason}")
        elif r.model_id not in checkpoints:
            out.append(f"  {r.model_id:45s} refused: {r.reason}")
    out += ["", "## Throughput", ""]
    out += [_row_line(r, current_model) for r in rows] or ["  (nothing measured)"]
    out += ["", "## Choice", ""]
    if choice is None:
        out.append(
            f"  cannot recommend a setting: no successful measurement of the configured model "
            f"({current_model}). A quick run never changes the model; if it does not fit this "
            "machine, run the full `sous tune` (a later release) or set [model].id by hand to "
            "one of the candidates above."
        )
    else:
        out.append(f"  {choice.label}")
        out += [f"    {r}" for r in choice.reasons]
        if not choice.changes:
            out.append("    no config change")
    return "\n".join(out) + "\n"


def config_diff(config_path: Path, changes: dict[str, dict[str, object]]) -> tuple[str, str]:
    """The new file text and its unified diff against the current one. tomlkit
    keeps comments, ordering and spacing, so the diff is the changed keys and
    nothing else."""
    old = config_path.read_text() if config_path.is_file() else ""
    doc = tomlkit.parse(old) if old else tomlkit.document()
    for section, keys in changes.items():
        table = doc.setdefault(section, tomlkit.table())
        for key, value in keys.items():
            table[key] = value
    new = tomlkit.dumps(doc)
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(config_path),
            tofile=f"{config_path} (tuned)",
        )
    )
    return new, diff


def apply_changes(config_path: Path, new_text: str, run_id: str) -> Path | None:
    backup = None
    if config_path.is_file():
        backup = config_path.with_name(f"{config_path.name}.bak-tune-{run_id}")
        shutil.copyfile(config_path, backup)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(new_text)
    return backup


def restart_note(changes: dict[str, dict[str, object]], *, managed: bool, label: str) -> str | None:
    """Keys the daemon reads once at load need a restart to take effect;
    everything a quick run writes is read per task or per turn."""
    model = changes.get("model", {})
    if "id" not in model and "int8_prefill" not in model:
        return None
    if managed:
        return f"restart the daemon to load it: launchctl kickstart -k gui/{os.getuid()}/{label}"
    return "restart the daemon to load it: sous stop, then sous serve"
