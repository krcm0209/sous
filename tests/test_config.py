import warnings
from pathlib import Path

import pytest

from sous.config import SousConfig, load_config


def test_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.server_port == 8383
    assert cfg.model_id == "mlx-community/Qwen3.8-27B-4bit"
    assert cfg.idle_unload_minutes == 30
    assert cfg.max_context_tokens == 131072
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.top_k == 20


def test_partial_file_overrides_only_given_keys(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[server]\nport = 9000\n")
    cfg = load_config(p)
    assert cfg.server_port == 9000
    assert cfg.idle_unload_minutes == 30  # untouched default


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


def test_prompt_cache_gb_defaults_to_auto(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x'\n")
    cfg = load_config(p)
    assert cfg.prompt_cache_gb is None


def test_prompt_cache_gb_accepts_a_number(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprompt_cache_gb = 12.5\n")
    cfg = load_config(p)
    assert cfg.prompt_cache_gb == 12.5


def test_prompt_cache_gb_zero_means_a_single_slot(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nprompt_cache_gb = 0\n")
    cfg = load_config(p)
    assert cfg.prompt_cache_gb == 0.0


@pytest.mark.parametrize("bad", ["-1", "true", "'lots'", "nan", "inf", "-inf", "1e308"])
def test_prompt_cache_gb_rejects_garbage_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nprompt_cache_gb = {bad}\n")
    with pytest.warns(UserWarning, match=r"\[model\]\.prompt_cache_gb"):
        cfg = load_config(p)
    assert cfg.prompt_cache_gb is None


# ---- [model].int8_prefill -----------------------------------------------------


def test_int8_prefill_defaults_off(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nid = 'x/y'\n")
    assert load_config(p).int8_prefill is False
    assert SousConfig().int8_prefill is False


def test_int8_prefill_reads_true_without_an_unknown_key_warning(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[model]\nint8_prefill = true\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.int8_prefill is True


@pytest.mark.parametrize("bad", ['"yes"', "1", "0.5"])
def test_int8_prefill_rejects_non_booleans_with_a_warning(tmp_path: Path, bad: str):
    p = tmp_path / "config.toml"
    p.write_text(f"[model]\nint8_prefill = {bad}\n")
    with pytest.warns(UserWarning, match=r"\[model\]\.int8_prefill"):
        cfg = load_config(p)
    assert cfg.int8_prefill is False


def test_speculative_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.speculative_draft_id == "z-lab/Qwen3.8-27B-DFlash2"
    # 3 measured best on the M5 Pro against the drafter's own adaptive policy
    # (+3% on prose, +13% on code re-emission); 0 would hand the choice back.
    assert cfg.speculative_block_size == 3


def test_speculative_keys_from_toml_without_unknown_key_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[model]\nspeculative_draft_id = ""\nspeculative_block_size = 3\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.speculative_draft_id == ""
    assert cfg.speculative_block_size == 3
    assert not [w for w in caught if "unknown" in str(w.message).lower()]


def test_speculative_block_size_one_or_negative_warns_and_uses_the_default(tmp_path: Path):
    """mlx-vlm treats the override as the total verify-block size and ends the
    round loop at <= 1 — a configured 1 would silently truncate every response
    to one token. Invalid values degrade to the default (3) with a warning,
    matching the [context] policy stance."""
    for bad in ("1", "-3", "true", '"3"'):
        p = tmp_path / f"c{len(bad)}{bad[0]}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {bad}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == 3, bad
        assert any("speculative_block_size" in str(w.message) for w in caught), bad


def test_speculative_block_size_above_five_warns_and_is_clamped_to_five(tmp_path: Path):
    """mlx's fused attention kernel takes at most 5 verify rows at the default
    model's GQA ratio; 6–8 rows fall off it and run 5–6x slower per layer, so
    a larger block is a net loss the user cannot see. Clamp, keeping the
    intent (as deep as pays), and say so."""
    for big in ("6", "9", "16"):
        p = tmp_path / f"c{big}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {big}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == 5, big
        assert any(
            "speculative_block_size" in str(w.message) and "5" in str(w.message) for w in caught
        ), big


def test_speculative_block_size_zero_and_two_to_five_accepted(tmp_path: Path):
    for ok in (2, 3, 4, 5):
        p = tmp_path / f"c{ok}.toml"
        p.write_text(f"[model]\nspeculative_block_size = {ok}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.speculative_block_size == ok
        assert not [w for w in caught if "speculative_block_size" in str(w.message)]
    p = tmp_path / "config.toml"
    p.write_text("[model]\nspeculative_block_size = 0\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.speculative_block_size == 0
    assert not [w for w in caught if "speculative_block_size" in str(w.message)]


# --- [server] and the served window --------------------------------------------


def test_server_and_model_sections_override_every_key_without_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[server]\n"
        "port = 9100\n"
        'local_models = ["sous-local", "sous-fast"]\n'
        "generation_timeout_minutes = 5\n"
        'upstream_url = "http://127.0.0.1:9000"\n'
        "\n[model]\n"
        "max_context_tokens = 262144\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.server_port == 9100
    assert cfg.local_models == ("sous-local", "sous-fast")
    assert cfg.max_context_tokens == 262144
    assert cfg.generation_timeout_minutes == 5
    assert cfg.upstream_url == "http://127.0.0.1:9000"


def test_bad_server_and_model_values_degrade_to_defaults_with_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[server]\n"
        "local_models = []\n"
        "generation_timeout_minutes = 0\n"
        "upstream_url = 7\n"
        "\n[model]\n"
        "max_context_tokens = -1\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.local_models == ("sous-local",)
    assert cfg.max_context_tokens == 131072
    assert cfg.generation_timeout_minutes == 30
    assert cfg.upstream_url == "https://api.anthropic.com"
    messages = " ".join(str(w.message) for w in caught)
    for key in (
        "local_models",
        "max_context_tokens",
        "generation_timeout_minutes",
        "upstream_url",
    ):
        assert key in messages, key


def test_local_models_rejects_non_string_entries(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[server]\nlocal_models = ["ok", 3]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.local_models == ("sous-local",)
    assert any("local_models" in str(w.message) for w in caught)


def test_local_models_rejects_claude_ids(tmp_path: Path):
    """Claude Code ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS for ids that
    canonicalize to claude-*, so an impersonating id silently forfeits the
    window control the endpoint depends on — honest ids are mandatory, not
    preferable."""
    p = tmp_path / "config.toml"
    p.write_text('[server]\nlocal_models = ["sous-local", "Claude-haiku-4-5"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.local_models == ("sous-local",)
    assert any(
        "local_models" in str(w.message) and "claude" in str(w.message).lower() for w in caught
    )


def test_upstream_defaults_to_the_anthropic_api(tmp_path: Path):
    from sous.config import DEFAULT_UPSTREAM

    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.upstream_url == DEFAULT_UPSTREAM == "https://api.anthropic.com"


def test_upstream_accepts_https_origins_and_loopback_http(tmp_path: Path):
    p = tmp_path / "config.toml"
    for raw, expected in (
        ("https://gateway.example.com", "https://gateway.example.com"),
        ("https://gateway.example.com:8443/", "https://gateway.example.com:8443"),
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("http://localhost:9000/", "http://localhost:9000"),
        ("http://[::1]:9000", "http://[::1]:9000"),
        ("https://[::1]:8443", "https://[::1]:8443"),
    ):
        p.write_text(f'[server]\nupstream_url = "{raw}"\n')
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert load_config(p).upstream_url == expected, raw


def test_upstream_rejects_anything_that_is_not_a_bare_origin(tmp_path: Path):
    """Forwarded requests carry the user's OAuth token: a plaintext upstream
    anywhere but loopback would put it on the wire in the clear, and a path,
    query or userinfo would silently change what gets forwarded."""
    p = tmp_path / "config.toml"
    for raw in (
        '"http://gateway.example.com"',
        '"https://api.anthropic.com/v1"',
        '"https://api.anthropic.com/?x=1"',
        '"https://user:pw@api.anthropic.com"',
        '"https://api.anthropic.com#frag"',
        '"ftp://api.anthropic.com"',
        '"api.anthropic.com"',
        '"http://[::1"',
        '"https://api.anthropic.com:abc"',
        '"https://api.anthropic.com:99999"',
        # Shape-valid, but not a host httpx can build a URL from (a control
        # character) or one that could ever resolve (a space, an underscore).
        # These used to pass validation and raise httpx.InvalidURL out of
        # Upstream.__init__ instead — killing the daemon at boot.
        '"https://api anthropic.com"',
        '"https://api.anthropic.com\\u0001"',
        '"https://ap_i.anthropic.com"',
        "42",
    ):
        p.write_text(f"[server]\nupstream_url = {raw}\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(p)
        assert cfg.upstream_url == "https://api.anthropic.com", raw
        assert any("upstream_url" in str(w.message) for w in caught), raw


@pytest.mark.parametrize(
    ("section", "key", "bad", "default"),
    [
        ("server", "port", "70000", 8383),
        ("server", "port", "0", 8383),
        ("server", "port", '"abc"', 8383),
        ("server", "port", "true", 8383),
        ("model", "idle_unload_minutes", "-5", 30),
        ("model", "idle_unload_minutes", "1.5", 30),
        ("model", "idle_unload_minutes", "false", 30),
    ],
)
def test_a_bad_port_or_idle_span_warns_and_uses_the_default(
    tmp_path: Path, section: str, key: str, bad: str, default: int
):
    """A bad port would take the daemon down at bind (launchd restart-loops
    it); a negative idle span would unload on every sweep tick."""
    path = tmp_path / "config.toml"
    path.write_text(f"[{section}]\n{key} = {bad}\n")
    with pytest.warns(UserWarning, match=rf"\[{section}\]\.{key}"):
        cfg = load_config(path)
    field = "server_port" if key == "port" else key
    assert getattr(cfg, field) == default


def test_an_idle_span_of_zero_is_a_setting_not_a_typo(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nidle_unload_minutes = 0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_config(path).idle_unload_minutes == 0


def test_settings_from_0_6_are_ignored_with_one_warning_naming_their_new_homes(tmp_path: Path):
    path = tmp_path / "config.toml"
    # The obsolete window differs from the served one, so the note can only
    # pass by naming the value in effect.
    path.write_text(
        '[gateway]\nenabled = true\nlocal_models = ["sous-local"]\nmax_context_tokens = 262144\n'
        'upstream_url = "https://api.anthropic.com"\ngeneration_timeout_minutes = 10\n\n'
        "[model]\nmax_context_tokens = 131072\n\n"
        '[budgets]\nmax_turns = 3\n\n[commands]\nallowlist = ["pytest"]\n\n'
        '[context]\nmode = "auto"\n\n[tasks]\nretention = 5\n'
    )
    with pytest.warns(UserWarning) as caught:
        cfg = load_config(path)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    m = messages[0]
    assert m.startswith("sous config: settings from 0.6 ignored: ")
    for note in (
        "[gateway].enabled (removed: the endpoint is always on)",
        "[gateway].local_models (now [server].local_models)",
        "[gateway].upstream_url (now [server].upstream_url)",
        "[gateway].generation_timeout_minutes (now [server].generation_timeout_minutes)",
        "[gateway].max_context_tokens "
        "(now [model].max_context_tokens; the served window is 131072)",
        "[budgets] (removed with the worker path)",
        "[commands] (removed with the worker path)",
        "[context] (removed with the worker path)",
        "[tasks] (removed with the worker path)",
    ):
        assert note in m
    assert cfg.max_context_tokens == 131072
    assert cfg.local_models == ("sous-local",)


def test_a_worker_era_window_is_clamped_to_the_floor(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nmax_context_tokens = 32768\n")
    with pytest.warns(
        UserWarning,
        match=r"\[model\]\.max_context_tokens 32768 is below Claude Code's 49152-token floor",
    ):
        cfg = load_config(path)
    assert cfg.max_context_tokens == 49152


def test_server_and_model_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.upstream_url == "https://api.anthropic.com"
    assert cfg.local_models == ("sous-local",)
    assert cfg.generation_timeout_minutes == 30
    assert cfg.max_context_tokens == 131072
    assert not hasattr(cfg, "gateway_enabled")
