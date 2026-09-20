import hashlib
import json
from pathlib import Path

from sous.tune import payload


def test_the_payload_names_the_eight_tools_in_order():
    names = [t["function"]["name"] for t in payload.TOOLS]
    assert names == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "run_command",
        "finish",
    ]
    assert payload.TOOLSET.names == frozenset(names)


def test_the_payload_is_pinned():
    # bench numbers and suite grades are only comparable across runs while
    # the model sees the same schema and prompt; a change here is deliberate
    # and re-baselines both.
    digest = hashlib.sha256(
        (json.dumps(payload.TOOLS, sort_keys=True) + payload.SYSTEM_TEMPLATE).encode()
    ).hexdigest()
    assert digest == "355642573e84df7d7c548c15cdda9c0828879d28c69accf19904f5cb26f439ae"


def test_the_system_prompt_lists_the_root_and_its_entries(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "README.md").write_text("x")
    text = payload.build_system_prompt(tmp_path)
    assert f"Project root: {tmp_path}" in text
    assert "README.md\npkg/" in text
