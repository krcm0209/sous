"""The engines' three fork hooks are one-line delegations to forkio, and
the identity key they build reads the engine's own facts."""

from __future__ import annotations

import sys
import types
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


class _Tokenizer:
    """A stub tokenizer: enough for LMEngine.__init__ to accept it without
    the mlx_lm.sample_utils.make_sampler call needing anything real."""

    bos_token = None
    chat_template = None


def _stub_mlx_lm(monkeypatch) -> None:
    model = object()
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(__name__="mlx_lm", load=lambda model_id: (model, _Tokenizer())),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(__name__="mlx_lm.sample_utils", make_sampler=lambda **kw: None),
    )


def test_a_store_construction_failure_disables_forks_without_failing_the_load(
    monkeypatch, tmp_path: Path
):
    """A model load must never fail because the disk store could not be
    built: the engine warns and keeps going with forks off rather than
    propagating the exception."""
    _stub_mlx_lm(monkeypatch)

    def raise_store(*a, **kw):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(lm, "ForkStore", raise_store)
    with pytest.warns(UserWarning, match="fork store off"):
        engine = lm.LMEngine(
            "test/model",
            prompt_cache=True,
            cache_budget=0,
            fork_dir=tmp_path / "forks",
            weights_identity="w",
        )
    assert engine.fork_store is None


def test_a_working_store_construction_leaves_forks_on(monkeypatch, tmp_path: Path):
    _stub_mlx_lm(monkeypatch)
    engine = lm.LMEngine(
        "test/model",
        prompt_cache=True,
        cache_budget=0,
        fork_dir=tmp_path / "forks",
        weights_identity="w",
    )
    assert engine.fork_store is not None
