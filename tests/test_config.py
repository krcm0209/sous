import warnings
from pathlib import Path

from sous.config import (
    DEFAULT_ALLOWLIST,
    current_allowlist,
    load_config,
    persist_allowlist_entry,
)


def test_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.server_port == 8383
    assert cfg.model_id == "mlx-community/Qwen3.8-27B-mxfp8"
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
    assert cfg.model_id == "mlx-community/Qwen3.8-27B-mxfp8"
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
