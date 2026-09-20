"""sous configuration: one TOML file, `[server]` for the endpoint and
`[model]` for the engine behind it, every key with a default."""

from __future__ import annotations

import math
import re
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_CONFIG_PATH = Path.home() / ".sous" / "config.toml"
DEFAULT_DATA_DIR = Path.home() / ".sous"

_KNOWN = {
    "server": {"port", "upstream_url", "local_models", "generation_timeout_minutes"},
    "model": {
        "id",
        "idle_unload_minutes",
        "max_context_tokens",
        "temperature",
        "top_p",
        "top_k",
        "prompt_cache",
        "prompt_cache_gb",
        "speculative_draft_id",
        "speculative_block_size",
        "int8_prefill",
    },
}

# Settings 0.7.0 removed with the worker path, and where the ones that moved
# now live: a config written for 0.6 is told so once at load, instead of
# steering nothing in silence.
_OBSOLETE: dict[str, dict[str, str | None]] = {
    "gateway": {
        "enabled": None,
        "local_models": "[server].local_models",
        "upstream_url": "[server].upstream_url",
        "generation_timeout_minutes": "[server].generation_timeout_minutes",
        "max_context_tokens": "[model].max_context_tokens",
    },
    "budgets": {},
    "commands": {},
    "context": {},
    "tasks": {},
}

# Claude Code refuses to run against a model advertising less than 48K of
# context (oMLX gates on the same 48 * 1024). A smaller window would never be
# used, so the config clamps up to this instead of serving it.
MIN_CONTEXT_TOKENS = 48 * 1024
# Where the daemon forwards every request it does not serve itself.
DEFAULT_UPSTREAM = "https://api.anthropic.com"
# Plaintext is tolerated only this far: the forwarded requests carry the
# user's OAuth token, and an http:// upstream anywhere else would put it on
# the wire in the clear.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class SousConfig:
    server_port: int = 8383
    upstream_url: str = DEFAULT_UPSTREAM
    local_models: tuple[str, ...] = ("sous-local",)
    generation_timeout_minutes: int = 30
    model_id: str = "mlx-community/Qwen3.8-27B-4bit"
    idle_unload_minutes: int = 30
    # The served window: what `sous claude` exports as CLAUDE_CODE_MAX_CONTEXT_TOKENS
    # and the render cap of a local turn. Below 49152 Claude Code compacts
    # constantly, so the floor is enforced at load.
    max_context_tokens: int = 131072
    # Qwen's documented non-thinking-mode sampling settings — greedy (temp=0)
    # decoding cannot escape a bad completion once one happens (a nudge can't
    # change an argmax pick over a near-identical prompt), so a retry of a
    # turn that went wrong would go wrong identically.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Speculative decoding (VLM backend only): a DFlash-style drafter predicts
    # blocks the target verifies in one forward — ~1.8x decode on the default
    # affine-4bit model with the shipped sampling, up to ~2.4x greedy
    # (krcm0209/sous#55, #58). Empty id disables it. The
    # drafter must match the target architecture; when it doesn't (or fails to
    # load), the engine logs and continues without it. Block size 3 measured
    # best on the M5 Pro against the drafter's adaptive policy (+3% on prose,
    # +13% on code re-emission); 0 hands the choice back to that policy. Above
    # 5 is clamped: mlx's fused attention kernel takes at most 5 verify rows
    # at the default model's GQA ratio, and 6–8 rows run 5–6x slower per layer.
    speculative_draft_id: str = "z-lab/Qwen3.8-27B-DFlash2"
    speculative_block_size: int = 3
    # Prefill matmuls of affine-Q4/gs64 projections on the M5 tensor units with
    # int8 activations: ~1.4x prefill measured on the M5 Pro (2026-09-11), but
    # int8 activations change numerics (KL 0.033 vs stock on the standard prompt,
    # inside the 4-bit weight envelope of 0.052), so it ships off until the
    # tool-loop A/B says otherwise. Ignored, with a status reason, on GPUs
    # without neural accelerators (pre-M5) or macOS < 26.2.
    int8_prefill: bool = False
    # Reuse one KV cache across the turns of a conversation, prefilling only
    # what it gained, instead of re-prefilling from scratch every turn. Works
    # because mlx streams are thread-scoped (#34): a slot only survives
    # between turns that run on the thread that built it.
    prompt_cache: bool = True
    # Memory resident prompt-cache slots may hold beyond the in-flight turn's
    # own cache, in GiB. None means automatic: what Metal's working set has
    # left once the weights, one full window of KV and 2 GiB of slack are paid
    # for. 0 keeps a single slot. Slots are what let two conversations
    # interleave on the local model without evicting each other, and what lets
    # a new subagent start from a copy of the ~50K-token header its
    # predecessor already prefilled.
    prompt_cache_gb: float | None = None
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
        if section in _OBSOLETE:
            continue
        if section not in _KNOWN:
            warnings.warn(f"sous config: unknown section [{section}]", stacklevel=3)
            continue
        if isinstance(values, dict):
            for key in values:
                if key not in _KNOWN[section]:
                    warnings.warn(f"sous config: unknown key {key!r} in [{section}]", stacklevel=3)


def _warn_obsolete(raw: dict, window: int) -> None:
    """One warning for a whole config written for 0.6, naming where each
    setting went. One and not one per key: the point is to send the reader to
    the file once, and a dozen warnings on a daemon's first line is noise
    nobody reads to the end of."""
    notes: list[str] = []
    for section, keys in _OBSOLETE.items():
        values = raw.get(section)
        if values is None:
            continue
        if not keys or not isinstance(values, dict):
            notes.append(f"[{section}] (removed with the worker path)")
            continue
        for key in values:
            home = keys.get(key)
            if key not in keys:
                notes.append(f"[{section}].{key} (unknown)")
            elif home is None:
                notes.append(f"[{section}].{key} (removed: the endpoint is always on)")
            elif key == "max_context_tokens":
                notes.append(f"[{section}].{key} (now {home}; the served window is {window})")
            else:
                notes.append(f"[{section}].{key} (now {home})")
    if notes:
        warnings.warn(
            "sous config: settings from 0.6 ignored: "
            + "; ".join(notes)
            + " — see README, Upgrading from 0.6",
            stacklevel=3,
        )


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


def _server_values(server: dict) -> tuple[tuple[str, ...], int]:
    """Validated [server].local_models and [server].generation_timeout_minutes,
    each degrading to its default with a warning — the stance the whole file
    takes: a typo must not stop the daemon from coming up."""
    models = server.get("local_models", ["sous-local"])
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) and m for m in models)
    ):
        warnings.warn(
            f"sous config: [server].local_models {models!r} must be a non-empty list of "
            "model ids; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    # Honest ids are mandatory. Claude Code ignores
    # CLAUDE_CODE_MAX_CONTEXT_TOKENS for any id that canonicalizes to claude-*
    # and trusts its built-in window instead, so an impersonating id silently
    # forfeits the window control the endpoint relies on, and pulls real
    # Claude traffic onto the local model. Substring, not prefix:
    # canonicalization strips provider prefixes, and no honest local id has
    # any reason to contain the word at all.
    if any("claude" in m.lower() for m in models):
        warnings.warn(
            f"sous config: [server].local_models {models!r} impersonates a Claude model; "
            "Claude Code ignores its context-window env vars for claude-* ids, so use an "
            "honest id like 'sous-local'; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    timeout = server.get("generation_timeout_minutes", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        warnings.warn(
            f"sous config: [server].generation_timeout_minutes {timeout!r} must be a "
            "positive integer; using 30",
            stacklevel=3,
        )
        timeout = 30
    return tuple(models), timeout


def _model_window(model: dict) -> int:
    """[model].max_context_tokens, clamped UP to the Claude Code floor rather
    than defaulted: a smaller value can only be a misjudged floor, and the
    floor is the closest thing to what the user asked for that would work."""
    window = model.get("max_context_tokens", 131072)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        warnings.warn(
            f"sous config: [model].max_context_tokens {window!r} must be a positive "
            "integer; using 131072",
            stacklevel=3,
        )
        return 131072
    if window < MIN_CONTEXT_TOKENS:
        warnings.warn(
            f"sous config: [model].max_context_tokens {window} is below Claude Code's "
            f"{MIN_CONTEXT_TOKENS}-token floor; using {MIN_CONTEXT_TOKENS}",
            stacklevel=3,
        )
        return MIN_CONTEXT_TOKENS
    return window


SPECULATIVE_BLOCK_DEFAULT = 3
# The largest verify block mlx's fused vector-attention kernel still takes at
# the default model's GQA ratio (q_len <= 8 and q_len x gqa <= 32, gqa 6 →
# 5 rows). 6–8 rows fall off the fused path and cost 5–6x per layer, which no
# acceptance rate pays back.
SPECULATIVE_BLOCK_MAX = 5


def _speculative_block_size(model: dict) -> int:
    """Validated [model].speculative_block_size: 0 (the drafter's own policy)
    or 2..5, degrading to the default with a warning — same stance as the rest
    of the file. This one is a silent-truncation knob: mlx-vlm treats the value
    as the total verify-block size and ends its round loop when it is <= 1,
    so a configured 1 (or a negative) would cap every response at a single
    token without any error. Above the maximum is clamped rather than
    defaulted: the intent ("as deep as pays") is clear, only the number is
    past where it pays."""
    value = model.get("speculative_block_size", SPECULATIVE_BLOCK_DEFAULT)
    if isinstance(value, bool) or not isinstance(value, int) or value == 1 or value < 0:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} must be 0 (auto) "
            f"or an integer from 2 to {SPECULATIVE_BLOCK_MAX}; using {SPECULATIVE_BLOCK_DEFAULT}",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_DEFAULT
    if value > SPECULATIVE_BLOCK_MAX:
        warnings.warn(
            f"sous config: [model].speculative_block_size {value!r} exceeds "
            f"{SPECULATIVE_BLOCK_MAX}, the most verify rows mlx's fused attention kernel "
            f"takes on this model; using {SPECULATIVE_BLOCK_MAX}",
            stacklevel=3,
        )
        return SPECULATIVE_BLOCK_MAX
    return value


def _prompt_cache_gb(model: dict) -> float | None:
    """[model].prompt_cache_gb: "auto" (None) or a non-negative number of GiB.
    Anything else warns and means auto."""
    value = model.get("prompt_cache_gb", "auto")
    if value == "auto":
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        # isfinite, not a NaN check: TOML spells inf and -inf too, and an
        # infinite budget reaches EngineManager as int(inf * (1 << 30)) —
        # an OverflowError out of engine load rather than a bad budget.
        or not math.isfinite(value)
        # The scaled value has to be finite too, not just the value: 1e308 is
        # a finite float whose product with 1 << 30 is not, so it would reach
        # EngineManager as the same int(inf) OverflowError. Test what the
        # engine will actually compute.
        or not math.isfinite(value * (1 << 30))
        or value < 0
    ):
        warnings.warn(
            f'sous config: [model].prompt_cache_gb {value!r} must be "auto" or a '
            'non-negative number of GiB; using "auto"',
            stacklevel=3,
        )
        return None
    return float(value)


def _int8_prefill(model: dict) -> bool:
    """[model].int8_prefill: true or false; anything else warns and means false,
    the same stance as the other [model] knobs (a typo must not turn on a path
    that changes numerics)."""
    value = model.get("int8_prefill", False)
    if isinstance(value, bool):
        return value
    warnings.warn(
        f"sous config: [model].int8_prefill {value!r} must be true or false; using false",
        stacklevel=3,
    )
    return False


# A registered name (letters, digits, dots, hyphens — RFC 3986's reg-name as
# the DNS world actually spells it) or an IPv6 literal with its brackets
# already stripped by urlsplit.
_HOSTNAME = re.compile(r"[A-Za-z0-9.-]+")
_IPV6_LITERAL = re.compile(r"[0-9A-Fa-f:.]+")


def _is_buildable_origin(hostname: str, origin: str) -> bool:
    """Whether httpx will accept this origin, checked here rather than
    discovered at boot: urlsplit's shape checks pass things httpx.URL refuses
    (a control character in the host is `httpx.InvalidURL`) or silently
    percent-encodes into a name that can never resolve (a space), and either
    way `Upstream.__init__` would raise out of daemon startup — a launchd
    KeepAlive restart loop — instead of the warn-and-default this validator
    exists to give."""
    if not (_HOSTNAME.fullmatch(hostname) or _IPV6_LITERAL.fullmatch(hostname)):
        return False
    # Function-local so `import sous.config` itself stays cheap; load_config()
    # does reach here for every valid upstream_url, so each CLI invocation
    # pays httpx's ~45 ms import once. Acceptable for a command-line tool.
    import httpx

    try:
        httpx.URL(origin)
    except httpx.InvalidURL:
        return False
    return True


def _upstream_url(server: dict) -> str:
    """The forwarding target as an origin — scheme + host[:port] and nothing
    else. A path or query would silently change what is forwarded; userinfo
    would be a credential sous stored; http is allowed only to loopback."""
    value = server.get("upstream_url", DEFAULT_UPSTREAM)
    if isinstance(value, str):
        try:
            parts = urlsplit(value)
            # urlsplit parses the port lazily, so ":abc" and ":99999" both
            # leave a valid-looking hostname behind and would be accepted here
            # only to raise httpx.InvalidURL out of Upstream.__init__ — i.e. to
            # kill the daemon at boot instead of warning and defaulting.
            _ = parts.port
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
            candidate = f"{parts.scheme}://{parts.netloc}"
            if _is_buildable_origin(parts.hostname, candidate):
                return candidate
    warnings.warn(
        f"sous config: [server].upstream_url {value!r} must be an https origin with no "
        f"path (plain http only for a loopback host); using {DEFAULT_UPSTREAM}",
        stacklevel=3,
    )
    return DEFAULT_UPSTREAM


def load_config(config_path: Path | None = None) -> SousConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_toml(path)
    _warn_unknown(raw)
    server = _section(raw, "server")
    model = _section(raw, "model")
    window = _model_window(model)
    _warn_obsolete(raw, window)
    local_models, generation_timeout = _server_values(server)
    return SousConfig(
        server_port=server.get("port", 8383),
        upstream_url=_upstream_url(server),
        local_models=local_models,
        generation_timeout_minutes=generation_timeout,
        model_id=model.get("id", "mlx-community/Qwen3.8-27B-4bit"),
        idle_unload_minutes=model.get("idle_unload_minutes", 30),
        max_context_tokens=window,
        temperature=model.get("temperature", 0.7),
        top_p=model.get("top_p", 0.8),
        top_k=model.get("top_k", 20),
        prompt_cache=model.get("prompt_cache", True),
        prompt_cache_gb=_prompt_cache_gb(model),
        speculative_draft_id=model.get("speculative_draft_id", "z-lab/Qwen3.8-27B-DFlash2"),
        speculative_block_size=_speculative_block_size(model),
        int8_prefill=_int8_prefill(model),
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
        config_path=path,
    )
