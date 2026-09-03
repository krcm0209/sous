import warnings
from pathlib import Path

from sous.config import (
    DEFAULT_ALLOWLIST,
    SousConfig,
    current_allowlist,
    load_config,
    persist_allowlist_entry,
)


def test_default_allowlist_covers_uv_run_wrappers(tmp_path: Path):
    """Python projects usually reach their tools through `uv run <tool>` — the
    bare tool is often not on the daemon's PATH at all. With only bare names
    allowlisted, every worker self-verification round-trips through a human
    approval, which is exactly the friction the allowlist exists to avoid."""
    import shlex

    from sous.toolexec import command_allowed

    allow = current_allowlist(tmp_path / "missing.toml")
    for cmd in ("uv run pytest -q", "uv run ruff check .", "uv run ty check"):
        assert command_allowed(shlex.split(cmd), allow), cmd


def test_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.server_port == 8383
    assert cfg.model_id == "mlx-community/Qwen3.8-27B-4bit"
    assert cfg.idle_unload_minutes == 30
    assert cfg.max_context_tokens == 32768
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.top_k == 20
    assert cfg.max_turns == 40
    assert cfg.max_minutes == 15
    assert cfg.max_tokens_per_generation == 4096
    assert cfg.command_timeout_seconds == 120
    assert cfg.approval_timeout_minutes == 10
    assert cfg.task_retention == 200


def test_partial_file_overrides_only_given_keys(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 9000\n")
    cfg = load_config(p)
    assert cfg.server_port == 9000
    assert cfg.max_turns == 40  # untouched default


def test_sampler_keys_overridable_from_file(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\ntemperature = 0.2\ntop_p = 0.9\ntop_k = 40\n")
    cfg = load_config(p)
    assert cfg.temperature == 0.2
    assert cfg.top_p == 0.9
    assert cfg.top_k == 40


def test_unknown_keys_warn_not_crash(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 9000\nbogus = 1\n[wat]\nx = 2\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.server_port == 9000
    assert any("bogus" in str(w.message) or "wat" in str(w.message) for w in caught)


def test_current_allowlist_reflects_live_edits(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[commands]\nallowlist = ["pytest"]\n')
    assert current_allowlist(p) == [["pytest"]]
    p.write_text('[commands]\nallowlist = ["pytest", "npm test"]\n')
    assert current_allowlist(p) == [["pytest"], ["npm", "test"]]


def test_current_allowlist_defaults_when_missing(tmp_path: Path):
    assert current_allowlist(tmp_path / "nope.toml") == [
        __import__("shlex").split(e) for e in DEFAULT_ALLOWLIST
    ]


def test_malformed_toml_returns_defaults_and_warns(tmp_path: Path):
    """I2: a syntax error in the hand-edited config must not crash the daemon
    at boot (launchd KeepAlive would restart-loop it)."""
    p = tmp_path / "config.toml"
    p.write_text("[server\nport = 9000\n")  # missing closing bracket
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.server_port == 8383  # defaults, not a raise
    assert any("config" in str(w.message).lower() for w in caught)


def test_malformed_toml_allowlist_falls_back_to_defaults(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[commands\nallowlist = ["pytest"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allow = current_allowlist(p)
    assert ["pytest"] in allow  # defaults still allow delegation
    assert any("config" in str(w.message).lower() for w in caught)


def test_unparseable_allowlist_entry_skipped_with_warning(tmp_path: Path):
    """I2: one unbalanced quote must not disable delegation entirely."""
    p = tmp_path / "config.toml"
    p.write_text('[commands]\nallowlist = ["pytest", "don\'t", "ruff"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allow = current_allowlist(p)
    assert ["pytest"] in allow and ["ruff"] in allow  # valid entries survive
    assert len(allow) == 2
    assert any("allowlist" in str(w.message).lower() for w in caught)


def test_non_table_section_falls_back_to_defaults(tmp_path: Path):
    """A2: a syntactically valid config whose section has the wrong SHAPE
    (server = 1) must not crash `sous serve` at boot — warn and use defaults
    for that section, same stance as a TOML syntax error."""
    p = tmp_path / "config.toml"
    p.write_text('server = 1\nmodel = "x"\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.server_port == 8383
    assert cfg.model_id == "mlx-community/Qwen3.8-27B-4bit"
    messages = [str(w.message) for w in caught]
    assert any("server" in m for m in messages)
    assert any("model" in m for m in messages)


def test_commands_not_a_table_current_allowlist_defaults(tmp_path: Path):
    """A2: [commands] with the wrong shape must not raise out of the hot
    allowlist read (which server_status/delegate_task/run_command all hit)."""
    p = tmp_path / "config.toml"
    p.write_text("commands = 5\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allow = current_allowlist(p)
    assert allow == [__import__("shlex").split(e) for e in DEFAULT_ALLOWLIST]
    assert any("commands" in str(w.message) for w in caught)


def test_allowlist_not_a_list_warns_and_uses_defaults(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[commands]\nallowlist = "pytest"\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allow = current_allowlist(p)
    assert allow == [__import__("shlex").split(e) for e in DEFAULT_ALLOWLIST]
    assert any("allowlist" in str(w.message) for w in caught)


def test_non_string_allowlist_entry_skipped_with_warning(tmp_path: Path):
    """A2: a non-string entry currently raises AttributeError out of
    shlex.split, escaping into delegate_task/server_status/run_command."""
    p = tmp_path / "config.toml"
    p.write_text('[commands]\nallowlist = [1, "pytest"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        allow = current_allowlist(p)
    assert allow == [["pytest"]]
    assert any("allowlist" in str(w.message) for w in caught)


def test_persist_allowlist_entry_appends_and_preserves(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('# my comment\n[commands]\nallowlist = ["pytest"]\n')
    persist_allowlist_entry("go vet", p)
    assert ["go", "vet"] in current_allowlist(p)
    assert "# my comment" in p.read_text()


def test_persist_allowlist_entry_creates_missing_file(tmp_path: Path):
    p = tmp_path / "config.toml"
    persist_allowlist_entry("go vet", p)
    assert ["go", "vet"] in current_allowlist(p)
    assert ["pytest"] in current_allowlist(p)  # defaults seeded too


def test_persist_respects_explicitly_empty_allowlist(tmp_path: Path):
    """An explicitly empty allowlist (`allowlist = []`) is a deliberate
    fail-closed posture: deny everything, make a human approve each command.
    Approving ONE command must append exactly that command — not silently
    seed all the defaults, which would reverse the user's security decision
    without telling them."""
    p = tmp_path / "config.toml"
    p.write_text("[commands]\nallowlist = []\n")
    persist_allowlist_entry("go vet", p)
    assert current_allowlist(p) == [["go", "vet"]]


def test_persist_seeds_defaults_when_commands_section_absent(tmp_path: Path):
    """First-run behavior: a config with no [commands] section at all gets
    the documented default allowlist plus the approved command."""
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 9000\n")
    persist_allowlist_entry("go vet", p)
    allow = current_allowlist(p)
    assert ["go", "vet"] in allow
    assert ["pytest"] in allow  # defaults seeded on first creation
    assert load_config(p).server_port == 9000  # rest of the file untouched


def test_persist_seeds_defaults_when_allowlist_key_absent(tmp_path: Path):
    """[commands] exists but has no allowlist key: the array is being
    created for the first time, so the defaults are seeded (same first-run
    rationale as a missing section)."""
    p = tmp_path / "config.toml"
    p.write_text("[commands]\ntimeout_seconds = 60\n")
    persist_allowlist_entry("go vet", p)
    allow = current_allowlist(p)
    assert ["go", "vet"] in allow
    assert ["pytest"] in allow
    assert load_config(p).command_timeout_seconds == 60


def test_persist_populated_allowlist_gains_only_the_new_entry(tmp_path: Path):
    """A populated allowlist gains the approved command and nothing else."""
    p = tmp_path / "config.toml"
    p.write_text('[commands]\nallowlist = ["pytest"]\n')
    persist_allowlist_entry("go vet", p)
    assert current_allowlist(p) == [["pytest"], ["go", "vet"]]


def test_context_section_parsed(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[context]\nmode = "auto"\nfraction = 0.6\nmin_tokens = 4096\n')
    cfg = load_config(p)
    assert cfg.context_mode == "auto"
    assert cfg.context_fraction == 0.6
    assert cfg.context_min_tokens == 4096


def test_context_defaults_to_fixed_mode(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.context_mode == "fixed"
    assert cfg.context_fraction == 0.8
    assert cfg.context_min_tokens == 8192


def test_context_invalid_values_warn_and_default(tmp_path: Path):
    """A fraction over 1 defeats the anti-thrashing headroom guarantee, an
    unknown mode silently disables auto sizing the user asked for, and a
    non-positive floor is nonsense — each must warn and fall back, matching
    how the rest of the config degrades."""
    p = tmp_path / "config.toml"
    p.write_text('[context]\nmode = "warp"\nfraction = 2.0\nmin_tokens = -5\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.context_mode == "fixed"
    assert cfg.context_fraction == 0.8
    assert cfg.context_min_tokens == 8192
    assert sum("context" in str(w.message) for w in caught) >= 3


def test_context_fraction_boundaries(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[context]\nfraction = 1.0\n")
    assert load_config(p).context_fraction == 1.0  # full headroom is coherent
    p.write_text("[context]\nfraction = 0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.context_fraction == 0.8  # zero would size every window to the floor
    assert caught


def test_prompt_cache_defaults_to_true():
    assert SousConfig().prompt_cache is True


def test_prompt_cache_can_be_disabled(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nprompt_cache = false\n")
    assert load_config(path).prompt_cache is False


def test_prompt_cache_is_a_known_model_key(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nprompt_cache = true\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_config(path)
    assert not [w for w in caught if "unknown" in str(w.message).lower()]


def test_speculative_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    assert cfg.speculative_block_size == 0


def test_speculative_keys_from_toml_without_unknown_key_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[model]\nspeculative_draft_id = ""\nspeculative_block_size = 3\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.speculative_draft_id == ""
    assert cfg.speculative_block_size == 3
    assert not [w for w in caught if "unknown" in str(w.message).lower()]


def test_speculative_block_size_one_or_negative_warns_and_defaults_to_auto(tmp_path: Path):
    """mlx-vlm treats the override as the total verify-block size and ends the
    round loop at <= 1 — a configured 1 would silently truncate every response
    to one token. Invalid values degrade to 0 (auto) with a warning, matching
    the [context] policy stance."""
    for bad in ("1", "-3", "true"):
        p = tmp_path / f"c{bad}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {bad}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == 0, bad
        assert any("speculative_block_size" in str(w.message) for w in caught), bad


def test_speculative_block_size_zero_and_ge_two_accepted(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nspeculative_block_size = 2\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.speculative_block_size == 2
    assert not [w for w in caught if "speculative_block_size" in str(w.message)]
    p.write_text("[model]\nspeculative_block_size = 0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.speculative_block_size == 0
    assert not [w for w in caught if "speculative_block_size" in str(w.message)]


# --- [gateway] -----------------------------------------------------------------


def test_gateway_defaults_are_off_and_above_the_claude_code_floor(tmp_path: Path):
    from sous.config import GATEWAY_MIN_CONTEXT_TOKENS

    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.gateway_enabled is False
    assert cfg.gateway_local_models == ("sous-local",)
    assert cfg.gateway_max_context_tokens == 65536
    assert cfg.gateway_max_context_tokens >= GATEWAY_MIN_CONTEXT_TOKENS == 49152
    assert cfg.gateway_generation_timeout_minutes == 30


def test_gateway_section_overrides_every_key_without_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[gateway]\n"
        "enabled = true\n"
        'local_models = ["sous-local", "sous-fast"]\n'
        "max_context_tokens = 131072\n"
        "generation_timeout_minutes = 5\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.gateway_enabled is True
    assert cfg.gateway_local_models == ("sous-local", "sous-fast")
    assert cfg.gateway_max_context_tokens == 131072
    assert cfg.gateway_generation_timeout_minutes == 5


def test_gateway_window_below_the_floor_clamps_up_with_a_warning(tmp_path: Path):
    """Claude Code refuses models under 48K of context, so a smaller window can
    only be a misjudged floor: serve the floor, not the default, and say so."""
    from sous.config import GATEWAY_MIN_CONTEXT_TOKENS

    p = tmp_path / "config.toml"
    p.write_text("[gateway]\nmax_context_tokens = 32768\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_max_context_tokens == GATEWAY_MIN_CONTEXT_TOKENS
    assert any("floor" in str(w.message) for w in caught)


def test_gateway_bad_values_degrade_to_defaults_with_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[gateway]\n"
        'enabled = "yes"\n'
        "local_models = []\n"
        "max_context_tokens = -1\n"
        "generation_timeout_minutes = 0\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_enabled is False
    assert cfg.gateway_local_models == ("sous-local",)
    assert cfg.gateway_max_context_tokens == 65536
    assert cfg.gateway_generation_timeout_minutes == 30
    messages = " ".join(str(w.message) for w in caught)
    for key in ("enabled", "local_models", "max_context_tokens", "generation_timeout_minutes"):
        assert key in messages, key


def test_gateway_local_models_rejects_non_string_entries(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[gateway]\nlocal_models = ["ok", 3]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_local_models == ("sous-local",)
    assert any("local_models" in str(w.message) for w in caught)


def test_gateway_local_models_rejects_claude_ids(tmp_path: Path):
    """Claude Code ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS for ids that
    canonicalize to claude-*, so an impersonating id silently forfeits the
    window control the gateway depends on — the spec makes honest ids
    mandatory, not preferable."""
    p = tmp_path / "config.toml"
    p.write_text('[gateway]\nlocal_models = ["sous-local", "Claude-haiku-4-5"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_local_models == ("sous-local",)
    assert any(
        "local_models" in str(w.message) and "claude" in str(w.message).lower() for w in caught
    )
