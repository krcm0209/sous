"""Score = the share of the originally undocumented public functions that
now carry a docstring, provided the hidden behaviour tests still all pass;
a behaviour change scores zero whatever was documented."""

import ast
from pathlib import Path

UNDOCUMENTED = ("perimeter_rect", "perimeter_circle", "area_triangle", "scale")


def grade(project: Path, tests) -> tuple[float, str]:
    try:
        tree = ast.parse((project / "shapes.py").read_text())
    except (OSError, SyntaxError) as e:
        return 0.0, f"shapes.py unreadable: {e}"
    documented = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node)
    }
    done = [name for name in UNDOCUMENTED if name in documented]
    passed, total, detail = tests(project)
    if total == 0 or passed < total:
        return 0.0, f"behaviour changed: {detail}"
    return len(done) / len(UNDOCUMENTED), f"{len(done)}/{len(UNDOCUMENTED)} documented; {detail}"
