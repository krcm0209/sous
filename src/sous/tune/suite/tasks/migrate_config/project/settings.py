"""Named configurations under configs/: load(name) returns
{section: {key: value}} with every value a string."""

import configparser
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def load(name: str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser()
    parser.read(CONFIG_DIR / f"{name}.ini")
    return {section: dict(parser[section]) for section in parser.sections()}
