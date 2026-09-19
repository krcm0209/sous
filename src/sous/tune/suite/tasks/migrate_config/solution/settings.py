"""Named configurations under configs/: load(name) returns
{section: {key: value}} with every value a string."""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def load(name: str) -> dict[str, dict[str, str]]:
    with (CONFIG_DIR / f"{name}.json").open() as f:
        return json.load(f)
