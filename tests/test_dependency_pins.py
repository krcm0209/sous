"""pyproject pins mlx and mlx-vlm to exactly what the exact paths validate."""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

from sous.engine import verifyattn

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pin(name: str) -> str:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    specs = [d[len(name) :] for d in deps if re.match(rf"{re.escape(name)}\s*[=<>!~]", d)]
    assert len(specs) == 1, f"{name}: {specs}"
    pin = re.fullmatch(r"\s*==\s*([\w.]+)", specs[0])
    assert pin, f"{name} must be pinned with ==, not {specs[0]!r}"
    return pin.group(1)


def test_mlx_is_pinned_to_a_validated_version():
    """Verify attention, the attention tile and the projection kernel refuse
    any mlx outside VALIDATED_MLX, so a fresh install must resolve one."""
    assert _pin("mlx") in verifyattn.VALIDATED_MLX


def test_the_pins_are_the_installed_versions():
    """The source-hash tests read the installed mlx-vlm, so they vouch for
    what a fresh install gets only while the pin is that version."""
    assert _pin("mlx") == version("mlx")
    assert _pin("mlx-vlm") == version("mlx-vlm")
