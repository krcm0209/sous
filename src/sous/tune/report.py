"""What the user reads: the report, the config diff, and the apply."""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path

import tomlkit

from sous.tune.arms import Refusal
from sous.tune.bench import BenchRow
from sous.tune.candidates import DRAFTER_QUANT_DIVISOR, Checkpoint
from sous.tune.decide import ArmSummary, FullChoice, QuickChoice
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
    if q.int8_excluded_layers:
        int8 += f", {q.int8_excluded_layers} layers excluded"
    return f"{layout}; {verifier}; {int8}"


def _candidate_line(cp: Checkpoint, refusal: Refusal | None) -> str:
    # A drafter's own KV cost and quant layout are not what a reader wants
    # here — it never runs unquantized and never on its own — just what it
    # costs resident, the way the candidate table's note field does.
    if cp.num_target_layers is not None:
        if cp.bytes is None:
            resident = "size unknown"
        else:
            gb = cp.bytes / DRAFTER_QUANT_DIVISOR / _GB
            resident = f"{gb:.1f} GB resident after 4-bit quantization"
        line = f"  {cp.id:45s} drafter  {resident}"
    else:
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


def _row_line(row: BenchRow, current_model: str, *, tag: bool) -> str:
    # In a quick run only the current model's quality is known; the full run
    # measured every model's, so nothing is untested there.
    untested = "  [quality untested]" if tag and row.model_id != current_model else ""
    if not row.ok:
        return f"  {row.label:45s} failed: {row.error}{untested}"
    peak = "-" if row.peak_memory_bytes is None else gib(row.peak_memory_bytes)
    line = (
        f"  {row.label:45s} prefill {_fmt(row.prefill_tps_2k)}/{_fmt(row.prefill_tps_16k)} tok/s"
        f"  decode {_fmt(row.decode_tps_1k)}/{_fmt(row.decode_tps_16k)} tok/s"
        f"  ttft {_fmt(row.ttft_seconds, ' s')}  peak {peak}"
        f"  spread {_fmt(None if row.spread is None else row.spread * 100, '%', 0)}"
        f"  load {_fmt(row.load_seconds, ' s')}{untested}"
    )
    if row.error:
        # A finished measurement whose teardown was refused or failed: the
        # weights stayed resident, and every arm measured after it says so
        # here, not only on the console of the run that hit it.
        line += f"\n      {row.error}"
    return line


def _suite_line(s: ArmSummary) -> str:
    return (
        f"  {s.label:45s} grade {s.mean_grade:.2f}  completed {s.completed}/{s.runs}"
        f"  wall {s.wall_seconds:.0f} s  repetitions {s.repetitions}  malformed {s.malformed}"
        f"  denied {s.approvals_denied}  output {s.output_tokens} tok"
        f"  peak {'-' if s.peak_memory_bytes is None else gib(s.peak_memory_bytes)}"
    )


def render_report(
    *,
    hardware: Hardware,
    table_age_days: int,
    checkpoints: dict[str, Checkpoint],
    refusals: list[Refusal],
    rows: list[BenchRow],
    choice: QuickChoice | FullChoice | None,
    current_model: str,
    quick: bool = True,
    suite: list[ArmSummary] | None = None,
    tasks: int = 0,
    runs: int = 0,
) -> str:
    refused = {r.model_id: r for r in refusals if not r.drafter_id}
    out = ["# sous tune --quick" if quick else "# sous tune", "", "## Hardware", ""]
    out.append(
        f"  {hardware.chip}, {gib(hardware.memory_bytes)} unified memory, Metal working set "
        f"{gib(hardware.working_set_bytes)}, macOS {hardware.macos}"
    )
    if hardware.nax:
        nax = "NAX tensor units: yes"
    elif hardware.nax_reason is None:
        nax = "NAX tensor units: no"
    else:
        nax = f"NAX tensor units: no ({hardware.nax_reason})"
    out.append(f"  {nax}; " + ", ".join(f"{k} {v}" for k, v in hardware.versions.items()))
    out.append(f"  hub cache {hardware.hub_cache}: {gib(hardware.disk_free_bytes)} free")
    if table_age_days > STALE_AFTER_DAYS:
        out.append(
            f"  the candidate table is {table_age_days} days old and may be stale; "
            "a newer sous release may list newer checkpoints, and --models measures ids of your own"
        )
    out += ["", "## Candidates", ""]
    for cp in checkpoints.values():
        out.append(_candidate_line(cp, refused.get(cp.id)))
    for r in refusals:
        if r.drafter_id:
            out.append(f"  {r.drafter_id:45s} dropped for {r.model_id}: {r.reason}")
        elif r.model_id not in checkpoints:
            out.append(f"  {r.model_id:45s} refused: {r.reason}")
    out += ["", "## Throughput", "  (prefill/decode at 2K/16K and 1K/16K tokens of context)", ""]
    out += [_row_line(r, current_model, tag=quick) for r in rows] or ["  (nothing measured)"]
    if not quick:
        out += [
            "",
            "## Suite",
            f"  ({tasks} tasks x {runs} runs per arm; grade = mean hidden-grader score in [0, 1]; "
            "wall = the sum of every run's seconds)",
            "",
        ]
        out += [_suite_line(s) for s in suite or []] or [
            "  (the suite did not run)" if not tasks else "  (no suite runs)"
        ]
    out += ["", "## Choice", ""]
    if choice is None and quick:
        out.append(
            f"  cannot recommend a setting: no successful measurement of the configured model "
            f"({current_model}). A quick run never changes the model; if it does not fit this "
            "machine, run the full `sous tune` or set [model].id by hand to one of the "
            "candidates above."
        )
    elif choice is None and not tasks:
        # The run stopped before the suite began (a bench arm's weights
        # stayed resident): the stop line above the report says why.
        out.append("  cannot recommend a setting: the suite did not run")
    elif choice is None:
        out.append("  cannot recommend a setting: no arm completed the suite")
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
    exists = config_path.is_file()
    if exists:
        backup = config_path.with_name(f"{config_path.name}.bak-tune-{run_id}")
        # copy2, not copyfile: a config the user made private stays private
        # in its backup.
        shutil.copy2(config_path, backup)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary file first and swapped in with os.replace, which
    # is atomic on the same filesystem: a crash mid-write leaves the temp
    # file half-written, never the real config truncated. The swap installs
    # a new inode, so the old file's mode is copied over first.
    tmp = config_path.with_name(f"{config_path.name}.tmp-{run_id}")
    tmp.write_text(new_text)
    if exists:
        shutil.copymode(config_path, tmp)
    os.replace(tmp, config_path)
    return backup


def restart_note(changes: dict[str, dict[str, object]], *, managed: bool) -> str | None:
    """`server.py` reads the config once at startup and builds one
    EngineManager closed over every [model] value; nothing is re-read at
    runtime. So every key `sous tune --quick` can write — not just the model
    id or int8_prefill — needs a restart before the daemon acts on it."""
    from sous.cli import restart_hint

    keys = [key for section in changes.values() for key in section]
    if not keys:
        return None
    return f"restart the daemon to apply {', '.join(keys)}: {restart_hint(managed=managed)}"
