"""sous configuration: TOML file with defaults, hot-reloaded allowlist."""

from __future__ import annotations

import shlex
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import tomlkit

DEFAULT_CONFIG_PATH = Path.home() / ".sous" / "config.toml"
DEFAULT_DATA_DIR = Path.home() / ".sous"

DEFAULT_ALLOWLIST: list[str] = [
    "pytest",
    "python -m pytest",
    "npm test",
    "npx eslint",
    "npx prettier",
    "ruff",
    "black",
    "mypy",
    "go test",
    "cargo test",
    "cargo check",
    "make test",
    # `uv run <tool>` variants of the Python tools above (plus ty): in uv
    # projects the bare tools usually aren't on the daemon's PATH at all, so
    # without these every worker self-check needs a human approval. Full
    # `uv run <tool>` entries, never a bare `uv run` prefix — that would
    # allowlist running arbitrary scripts.
    "uv run pytest",
    "uv run python -m pytest",
    "uv run ruff",
    "uv run black",
    "uv run mypy",
    "uv run ty",
]

_KNOWN = {
    "server": {"port"},
    "model": {
        "id",
        "idle_unload_minutes",
        "max_context_tokens",
        "temperature",
        "top_p",
        "top_k",
        "prompt_cache",
    },
    "budgets": {"max_turns", "max_minutes", "max_tokens_per_generation"},
    "commands": {"allowlist", "timeout_seconds", "approval_timeout_minutes"},
    "context": {"mode", "fraction", "min_tokens"},
    "tasks": {"retention"},
}


@dataclass(frozen=True)
class SousConfig:
    server_port: int = 8383
    model_id: str = "mlx-community/Qwen3.8-27B-mxfp8"
    idle_unload_minutes: int = 30
    max_context_tokens: int = 32768
    # Qwen's documented non-thinking-mode sampling settings — greedy (temp=0)
    # decoding cannot escape a bad completion once one happens (a nudge can't
    # change an argmax pick over a near-identical prompt), so some stochastic
    # sampling is required for the worker to have any chance of recovering.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Reuse one KV cache across the turns of a task, prefilling only what the
    # conversation gained, instead of re-prefilling from scratch every turn.
    # Off by default: worker.py's _generate_with_timeout runs each generation
    # on a fresh daemon thread, and that thread's exit releases the mlx
    # streams the cache arrays live on, so turn N+1 (a new thread) can't
    # touch them and falls back to a cold retry. Today, turning this on
    # costs one wasted warm attempt per turn instead of saving anything.
    prompt_cache: bool = True
    max_turns: int = 40
    max_minutes: int = 15
    max_tokens_per_generation: int = 4096
    command_timeout_seconds: int = 120
    approval_timeout_minutes: int = 10
    task_retention: int = 200
    # "fixed": serve max_context_tokens as-is. "auto": size the window per
    # task from live memory headroom (see sous.context), using `fraction` of
    # it and never dropping below `min_tokens`.
    context_mode: str = "fixed"
    context_fraction: float = 0.8
    context_min_tokens: int = 8192
    data_dir: Path = DEFAULT_DATA_DIR
    config_path: Path = DEFAULT_CONFIG_PATH


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            # A typo in the hand-edited config must not crash the daemon at
            # boot (launchd KeepAlive would restart-loop it) — warn and run
            # on defaults, matching the unknown-key stance.
            warnings.warn(f"sous config: cannot parse {path} ({e}); using defaults", stacklevel=3)
            return {}


def _warn_unknown(raw: dict) -> None:
    for section, values in raw.items():
        if section not in _KNOWN:
            warnings.warn(f"sous config: unknown section [{section}]", stacklevel=3)
            continue
        if isinstance(values, dict):
            for key in values:
                if key not in _KNOWN[section]:
                    warnings.warn(f"sous config: unknown key {key!r} in [{section}]", stacklevel=3)


def _section(raw: dict, name: str) -> dict:
    """A valid-TOML config with the wrong SHAPE (`server = 1`) must not crash
    the daemon at boot any more than a syntax error would — warn naming the
    section and fall back to defaults for it."""
    value = raw.get(name, {})
    if not isinstance(value, dict):
        warnings.warn(
            f"sous config: [{name}] is not a table "
            f"(got {type(value).__name__}); using defaults for it",
            stacklevel=4,
        )
        return {}
    return value


def _context_values(context: dict) -> tuple[str, float, int]:
    """Validated [context] policy values, each degrading to its default with
    a warning — same stance as the rest of the config. These are safety
    knobs: a fraction over 1 defeats the anti-thrashing headroom guarantee,
    and a typo'd mode would silently disable the auto sizing the user asked
    for."""
    mode = context.get("mode", "fixed")
    if mode not in ("fixed", "auto"):
        warnings.warn(
            f"sous config: [context].mode {mode!r} is neither 'fixed' nor 'auto'; using 'fixed'",
            stacklevel=3,
        )
        mode = "fixed"
    fraction = context.get("fraction", 0.8)
    if isinstance(fraction, bool) or not isinstance(fraction, int | float) or not 0 < fraction <= 1:
        warnings.warn(
            f"sous config: [context].fraction {fraction!r} must be in (0, 1]; using 0.8",
            stacklevel=3,
        )
        fraction = 0.8
    min_tokens = context.get("min_tokens", 8192)
    if isinstance(min_tokens, bool) or not isinstance(min_tokens, int) or min_tokens <= 0:
        warnings.warn(
            f"sous config: [context].min_tokens {min_tokens!r} must be a positive "
            f"integer; using 8192",
            stacklevel=3,
        )
        min_tokens = 8192
    return mode, float(fraction), min_tokens


def load_config(config_path: Path | None = None) -> SousConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_toml(path)
    _warn_unknown(raw)
    server = _section(raw, "server")
    model = _section(raw, "model")
    budgets = _section(raw, "budgets")
    commands = _section(raw, "commands")
    context = _section(raw, "context")
    context_mode, context_fraction, context_min_tokens = _context_values(context)
    tasks = _section(raw, "tasks")
    return SousConfig(
        server_port=server.get("port", 8383),
        model_id=model.get("id", "mlx-community/Qwen3.8-27B-mxfp8"),
        idle_unload_minutes=model.get("idle_unload_minutes", 30),
        max_context_tokens=model.get("max_context_tokens", 32768),
        temperature=model.get("temperature", 0.7),
        top_p=model.get("top_p", 0.8),
        top_k=model.get("top_k", 20),
        prompt_cache=model.get("prompt_cache", True),
        max_turns=budgets.get("max_turns", 40),
        max_minutes=budgets.get("max_minutes", 15),
        max_tokens_per_generation=budgets.get("max_tokens_per_generation", 4096),
        command_timeout_seconds=commands.get("timeout_seconds", 120),
        approval_timeout_minutes=commands.get("approval_timeout_minutes", 10),
        task_retention=tasks.get("retention", 200),
        context_mode=context_mode,
        context_fraction=context_fraction,
        context_min_tokens=context_min_tokens,
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
        config_path=path,
    )


def current_allowlist(config_path: Path) -> list[list[str]]:
    """Hot path: re-read the allowlist on every command execution."""
    raw = _read_toml(config_path)
    entries = _section(raw, "commands").get("allowlist", DEFAULT_ALLOWLIST)
    if not isinstance(entries, list):
        # This escapes into delegate_task/server_status/run_command — a wrong
        # shape must degrade to defaults, never raise out of the service API.
        warnings.warn(
            f"sous config: [commands].allowlist is not a list "
            f"(got {type(entries).__name__}); using defaults",
            stacklevel=2,
        )
        entries = DEFAULT_ALLOWLIST
    parsed: list[list[str]] = []
    for entry in entries:
        if not isinstance(entry, str):
            warnings.warn(
                f"sous config: skipping non-string allowlist entry {entry!r}", stacklevel=2
            )
            continue
        try:
            parsed.append(shlex.split(entry))
        except ValueError as e:
            # One unparseable entry (e.g. an unbalanced quote) must not
            # disable delegation — skip it, keep the valid ones.
            warnings.warn(
                f"sous config: skipping unparseable allowlist entry {entry!r} ({e})", stacklevel=2
            )
    return parsed


def persist_allowlist_entry(command: str, config_path: Path) -> None:
    """Append one command to the allowlist, preserving file formatting."""
    doc = tomlkit.parse(config_path.read_text()) if config_path.is_file() else tomlkit.document()
    commands = doc.setdefault("commands", tomlkit.table())
    # Seed the defaults ONLY when the key is absent — i.e. the array is being
    # created for the first time (new file, or a section that never had an
    # allowlist). An explicitly empty allowlist (`allowlist = []`) is a
    # deliberate fail-closed posture — deny everything, make a human approve
    # each command — and must gain only the command just approved, never be
    # silently repopulated with the defaults.
    seed_defaults = "allowlist" not in commands
    allow = commands.setdefault("allowlist", tomlkit.array())
    if seed_defaults:
        for entry in DEFAULT_ALLOWLIST:
            allow.append(entry)
    if command not in list(allow):
        allow.append(command)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(tomlkit.dumps(doc))
