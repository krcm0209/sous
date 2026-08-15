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
    "pytest", "python -m pytest", "npm test", "npx eslint", "npx prettier",
    "ruff", "black", "mypy", "go test", "cargo test", "cargo check", "make test",
]

_KNOWN = {
    "server": {"port"},
    "model": {"id", "idle_unload_minutes", "max_context_tokens",
              "temperature", "top_p", "top_k"},
    "budgets": {"max_turns", "max_minutes", "max_tokens_per_generation"},
    "commands": {"allowlist", "timeout_seconds", "approval_timeout_minutes"},
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
    max_turns: int = 40
    max_minutes: int = 15
    max_tokens_per_generation: int = 4096
    command_timeout_seconds: int = 120
    approval_timeout_minutes: int = 10
    task_retention: int = 200
    data_dir: Path = DEFAULT_DATA_DIR
    config_path: Path = DEFAULT_CONFIG_PATH


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _warn_unknown(raw: dict) -> None:
    for section, values in raw.items():
        if section not in _KNOWN:
            warnings.warn(f"sous config: unknown section [{section}]", stacklevel=3)
            continue
        if isinstance(values, dict):
            for key in values:
                if key not in _KNOWN[section]:
                    warnings.warn(
                        f"sous config: unknown key {key!r} in [{section}]", stacklevel=3
                    )


def load_config(config_path: Path | None = None) -> SousConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_toml(path)
    _warn_unknown(raw)
    server = raw.get("server", {})
    model = raw.get("model", {})
    budgets = raw.get("budgets", {})
    commands = raw.get("commands", {})
    tasks = raw.get("tasks", {})
    return SousConfig(
        server_port=server.get("port", 8383),
        model_id=model.get("id", "mlx-community/Qwen3.8-27B-mxfp8"),
        idle_unload_minutes=model.get("idle_unload_minutes", 30),
        max_context_tokens=model.get("max_context_tokens", 32768),
        temperature=model.get("temperature", 0.7),
        top_p=model.get("top_p", 0.8),
        top_k=model.get("top_k", 20),
        max_turns=budgets.get("max_turns", 40),
        max_minutes=budgets.get("max_minutes", 15),
        max_tokens_per_generation=budgets.get("max_tokens_per_generation", 4096),
        command_timeout_seconds=commands.get("timeout_seconds", 120),
        approval_timeout_minutes=commands.get("approval_timeout_minutes", 10),
        task_retention=tasks.get("retention", 200),
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
        config_path=path,
    )


def current_allowlist(config_path: Path) -> list[list[str]]:
    """Hot path: re-read the allowlist on every command execution."""
    raw = _read_toml(config_path)
    entries = raw.get("commands", {}).get("allowlist", DEFAULT_ALLOWLIST)
    return [shlex.split(e) for e in entries]


def persist_allowlist_entry(command: str, config_path: Path) -> None:
    """Append one command to the allowlist, preserving file formatting."""
    if config_path.is_file():
        doc = tomlkit.parse(config_path.read_text())
    else:
        doc = tomlkit.document()
    commands = doc.setdefault("commands", tomlkit.table())
    allow = commands.setdefault("allowlist", tomlkit.array())
    if not list(allow):
        for entry in DEFAULT_ALLOWLIST:
            allow.append(entry)
    if command not in list(allow):
        allow.append(command)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(tomlkit.dumps(doc))
