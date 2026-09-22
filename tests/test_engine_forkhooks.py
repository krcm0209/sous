"""The engines' three fork hooks are one-line delegations to forkio, and
the identity key they build reads the engine's own facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sous.engine import forkio, lm, vlm


@pytest.mark.parametrize("cls", [vlm.VLMEngine, lm.LMEngine])
def test_hooks_delegate_to_forkio(monkeypatch, cls, tmp_path: Path):
    calls: list[tuple] = []
    monkeypatch.setattr(forkio, "eval_cache", lambda cache: calls.append(("eval", cache)))
    monkeypatch.setattr(
        forkio, "persist_cache", lambda cache, path, ids, meta: calls.append(("persist", path))
    )
    monkeypatch.setattr(
        forkio, "restore_cache", lambda path, header, cache, ids: calls.append(("restore", path))
    )
    engine = cls.__new__(cls)  # no model: only the hooks are exercised
    engine.eval_cache(["c"])
    engine.persist(["c"], tmp_path / "f", [1], {})
    engine.restore(tmp_path / "f", None, ["c"], [1])
    assert [c[0] for c in calls] == ["eval", "persist", "restore"]


def test_vlm_key_fields_read_positions_and_the_realised_int8_state(monkeypatch):
    engine = vlm.VLMEngine.__new__(vlm.VLMEngine)
    engine._positional = True
    engine.int8_prefill_status = {"state": "active", "reason": None, "routed": 12}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "vlm" and fields["positions"] == "engine"
    assert fields["int8"] == "active:12" and fields["weights"] == "sha"
    assert fields["gpu"]  # mx.device_info()["architecture"] on this Mac


def test_lm_key_fields_name_the_lm_backend(monkeypatch):
    engine = lm.LMEngine.__new__(lm.LMEngine)
    engine.int8_prefill_status = {"state": "off", "reason": None, "routed": 0}
    fields = engine._fork_key_fields("sha")
    assert fields["backend"] == "lm" and fields["positions"] == "model"
    assert fields["int8"] == "off"
