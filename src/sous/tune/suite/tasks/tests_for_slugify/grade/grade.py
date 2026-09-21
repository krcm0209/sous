"""Score = 1 if the model's tests pass on the pristine module, times the
share of five mutants — one per documented behaviour — those tests catch.
No tests, or tests that fail on the real module, score zero."""

import shutil
import tempfile
from pathlib import Path

_HEAD = 'import re\n\n_NON_WORD = re.compile(r"[^a-z0-9]+")\n\n\n'
MUTANTS = {
    "no lower-casing": _HEAD
    + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text).strip("-")\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "leading and trailing dashes kept": _HEAD
    + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text.lower())\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "a run becomes several dashes": (
        'import re\n\n_NON_WORD = re.compile(r"[^a-z0-9]")\n\n\n'
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text.lower()).strip("-")\n'
        '    return slug[:max_length].rstrip("-")\n'
    ),
    "max_length ignored": _HEAD
    + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    return _NON_WORD.sub("-", text.lower()).strip("-")\n'
    ),
    "a dash the cut leaves is kept": _HEAD
    + (
        "def slugify(text: str, max_length: int = 40) -> str:\n"
        '    slug = _NON_WORD.sub("-", text.lower()).strip("-")\n'
        "    return slug[:max_length]\n"
    ),
}


def _fails(tests, project: Path) -> bool:
    passed, total, _ = tests(project, project / "tests")
    return total == 0 or passed < total


def grade(project: Path, tests) -> tuple[float, str]:
    passed, total, detail = tests(project, project / "tests")
    if total == 0 or passed < total:
        return 0.0, f"the model's tests on the pristine module: {detail}"
    caught = []
    for name, source in MUTANTS.items():
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / "project"
            shutil.copytree(project, copy)
            (copy / "slugify.py").write_text(source)
            if _fails(tests, copy):
                caught.append(name)
    return (
        len(caught) / len(MUTANTS),
        f"{total} tests pass on the pristine module; caught {len(caught)}/{len(MUTANTS)} "
        f"mutants ({', '.join(caught) or 'none'})",
    )
