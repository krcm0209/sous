import warnings
from pathlib import Path

from sous.config import (
    DEFAULT_ALLOWLIST,
    SousConfig,
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
    p.write_text('[server]\nport = 9000\n')
    cfg = load_config(p)
    assert cfg.server_port == 9000
    assert cfg.max_turns == 40  # untouched default


def test_sampler_keys_overridable_from_file(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[model]\ntemperature = 0.2\ntop_p = 0.9\ntop_k = 40\n')
    cfg = load_config(p)
    assert cfg.temperature == 0.2
    assert cfg.top_p == 0.9
    assert cfg.top_k == 40


def test_unknown_keys_warn_not_crash(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[server]\nport = 9000\nbogus = 1\n[wat]\nx = 2\n')
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
