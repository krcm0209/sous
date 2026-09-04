"""sous configuration: TOML file with defaults, hot-reloaded allowlist."""

from __future__ import annotations

import shlex
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

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
        "speculative_draft_id",
        "speculative_block_size",
    },
    "budgets": {"max_turns", "max_minutes", "max_tokens_per_generation"},
    "commands": {"allowlist", "timeout_seconds", "approval_timeout_minutes"},
    "context": {"mode", "fraction", "min_tokens"},
    "tasks": {"retention"},
    "gateway": {
        "enabled",
        "local_models",
        "max_context_tokens",
        "generation_timeout_minutes",
        "upstream_url",
    },
}

# Claude Code refuses to run against a model advertising less than 48K of
# context (oMLX gates on the same 48 * 1024). A smaller gateway window would
# never be used, so the config clamps up to this instead of serving it.
GATEWAY_MIN_CONTEXT_TOKENS = 48 * 1024
# Where the gateway forwards every request it does not serve itself.
GATEWAY_DEFAULT_UPSTREAM = "https://api.anthropic.com"
# Plaintext is tolerated only this far: the forwarded requests carry the
# user's OAuth token, and an http:// upstream anywhere else would put it on
# the wire in the clear.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class SousConfig:
    server_port: int = 8383
    model_id: str = "mlx-community/Qwen3.8-27B-4bit"
    idle_unload_minutes: int = 30
    max_context_tokens: int = 32768
    # Qwen's documented non-thinking-mode sampling settings — greedy (temp=0)
    # decoding cannot escape a bad completion once one happens (a nudge can't
    # change an argmax pick over a near-identical prompt), so some stochastic
    # sampling is required for the worker to have any chance of recovering.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Speculative decoding (VLM backend only): a DFlash-style drafter predicts
    # blocks the target verifies in one forward — ~1.8x decode on the default
    # affine-4bit model with the shipped sampling, up to ~2.4x greedy
    # (krcm0209/sous#55, #58). Empty id disables it. The
    # drafter must match the target architecture; when it doesn't (or fails to
    # load), the engine logs and continues without it. Block size 0 lets the
    # drafter's own policy pick the depth — the best-measured setting.
    speculative_draft_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    speculative_block_size: int = 0
    # Reuse one KV cache across the turns of a task, prefilling only what the
    # conversation gained, instead of re-prefilling from scratch every turn.
    # Works because all of a task's generations share one GenerationSession
    # thread (#34) — mlx streams are thread-scoped, so the cache only
    # survives between turns that run on the thread that built it.
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
    # Anthropic-compatible endpoint on the daemon (issue #41), off by default
    # and experimental. It serves Claude Code with the local model and never
    # touches the toolexec sandbox: Claude Code executes the tools under its
    # own permission system. The window is the gateway's own — Claude Code's
    # prompts are far larger than the worker's, and the worker's cap stays put.
    gateway_enabled: bool = False
    gateway_local_models: tuple[str, ...] = ("sous-local",)
    gateway_max_context_tokens: int = 65536
    gateway_generation_timeout_minutes: int = 30
    gateway_upstream_url: str = GATEWAY_DEFAULT_UPSTREAM
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


def _speculative_block_size(model: dict) -> int:
    """Validated [model].speculative_block_size, degrading to 0 (auto) with a
    warning — same stance as [context]. This one is a silent-truncation knob:
    mlx-vlm treats the value as the total verify-block size and ends its round
    loop when it is <= 1, so a configured 1 (or a negative) would cap every
    response at a single token without any error."""
    value = model.get("speculative_block_size", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value == 1 or value < 0:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} must be 0 (auto) "
            "or an integer >= 2; using 0",
            stacklevel=3,
        )
        return 0
    return value


def _gateway_values(gateway: dict) -> tuple[bool, tuple[str, ...], int, int]:
    """Validated [gateway] values, each degrading to its default with a
    warning — except the window, which is clamped UP to the Claude Code floor:
    a smaller value can only be a misjudged floor, and the floor is the
    closest thing to what the user asked for that would actually work."""
    enabled = gateway.get("enabled", False)
    if not isinstance(enabled, bool):
        warnings.warn(
            f"sous config: [gateway].enabled {enabled!r} must be true or false; using false",
            stacklevel=3,
        )
        enabled = False
    models = gateway.get("local_models", ["sous-local"])
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) and m for m in models)
    ):
        warnings.warn(
            f"sous config: [gateway].local_models {models!r} must be a non-empty list of "
            "model ids; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    # Spec: honest ids are mandatory. Claude Code ignores
    # CLAUDE_CODE_MAX_CONTEXT_TOKENS for any id that canonicalizes to claude-*
    # and trusts its built-in window instead, so an impersonating id silently
    # forfeits the window control the gateway relies on (and, once routing
    # lands, would pull real Claude traffic onto the local model). Substring,
    # not prefix: canonicalization strips provider prefixes, and no honest
    # local id has any reason to contain the word at all.
    if any("claude" in m.lower() for m in models):
        warnings.warn(
            f"sous config: [gateway].local_models {models!r} impersonates a Claude model; "
            "Claude Code ignores its context-window env vars for claude-* ids, so use an "
            "honest id like 'sous-local'; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    window = gateway.get("max_context_tokens", 65536)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        warnings.warn(
            f"sous config: [gateway].max_context_tokens {window!r} must be a positive "
            "integer; using 65536",
            stacklevel=3,
        )
        window = 65536
    elif window < GATEWAY_MIN_CONTEXT_TOKENS:
        warnings.warn(
            f"sous config: [gateway].max_context_tokens {window} is below Claude Code's "
            f"{GATEWAY_MIN_CONTEXT_TOKENS}-token floor; using {GATEWAY_MIN_CONTEXT_TOKENS}",
            stacklevel=3,
        )
        window = GATEWAY_MIN_CONTEXT_TOKENS
    timeout = gateway.get("generation_timeout_minutes", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        warnings.warn(
            f"sous config: [gateway].generation_timeout_minutes {timeout!r} must be a "
            "positive integer; using 30",
            stacklevel=3,
        )
        timeout = 30
    return enabled, tuple(models), window, timeout


def _upstream_url(gateway: dict) -> str:
    """The forwarding target as an origin — scheme + host[:port] and nothing
    else. A path or query would silently change what is forwarded; userinfo
    would be a credential sous stored; http is allowed only to loopback."""
    value = gateway.get("upstream_url", GATEWAY_DEFAULT_UPSTREAM)
    if isinstance(value, str):
        try:
            parts = urlsplit(value)
        except ValueError:
            # An unbalanced IPv6 bracket makes urlsplit raise rather than return.
            parts = None
        if (
            parts is not None
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and parts.path in ("", "/")
            and not parts.query
            and not parts.fragment
            and (
                parts.scheme == "https"
                or (parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS)
            )
        ):
            return f"{parts.scheme}://{parts.netloc}"
    warnings.warn(
        f"sous config: [gateway].upstream_url {value!r} must be an https origin with no "
        f"path (plain http only for a loopback host); using {GATEWAY_DEFAULT_UPSTREAM}",
        stacklevel=3,
    )
    return GATEWAY_DEFAULT_UPSTREAM


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
    gateway = _section(raw, "gateway")
    gateway_enabled, gateway_models, gateway_window, gateway_timeout = _gateway_values(gateway)
    return SousConfig(
        server_port=server.get("port", 8383),
        model_id=model.get("id", "mlx-community/Qwen3.8-27B-4bit"),
        idle_unload_minutes=model.get("idle_unload_minutes", 30),
        max_context_tokens=model.get("max_context_tokens", 32768),
        temperature=model.get("temperature", 0.7),
        top_p=model.get("top_p", 0.8),
        top_k=model.get("top_k", 20),
        prompt_cache=model.get("prompt_cache", True),
        speculative_draft_id=model.get("speculative_draft_id", "z-lab/Qwen3.8-27B-DFlash2"),
        speculative_block_size=_speculative_block_size(model),
        max_turns=budgets.get("max_turns", 40),
        max_minutes=budgets.get("max_minutes", 15),
        max_tokens_per_generation=budgets.get("max_tokens_per_generation", 4096),
        command_timeout_seconds=commands.get("timeout_seconds", 120),
        approval_timeout_minutes=commands.get("approval_timeout_minutes", 10),
        task_retention=tasks.get("retention", 200),
        context_mode=context_mode,
        context_fraction=context_fraction,
        context_min_tokens=context_min_tokens,
        gateway_enabled=gateway_enabled,
        gateway_local_models=gateway_models,
        gateway_max_context_tokens=gateway_window,
        gateway_generation_timeout_minutes=gateway_timeout,
        gateway_upstream_url=_upstream_url(gateway),
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
