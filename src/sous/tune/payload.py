"""What the local model sees during a tune: the eight-tool schema and the
system prompt the worker path used. Kept as fixtures so bench numbers and
suite grades stay comparable across releases — a change here re-baselines
both, and the pin test says so."""

from __future__ import annotations

from pathlib import Path

from sous.protocol import ToolSet


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS: list[dict] = [
    _tool(
        "read_file",
        "Read a file (line-numbered). Use offset/limit for large files.",
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "0-based start line"},
            "limit": {"type": "integer", "description": "max lines"},
        },
        ["path"],
    ),
    _tool(
        "write_file",
        "Create or overwrite a file with the given content.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _tool(
        "edit_file",
        "Replace one exact, unique occurrence of `old` with `new`.",
        {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        ["path", "old", "new"],
    ),
    _tool(
        "list_dir",
        "List one directory's entries.",
        {"path": {"type": "string", "description": "default: project root"}},
        [],
    ),
    _tool(
        "glob",
        "Find files by glob pattern, e.g. **/*.py",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    _tool(
        "grep",
        "Regex-search file contents. Returns path:line:text hits.",
        {
            "pattern": {"type": "string"},
            "glob_pattern": {"type": "string", "description": "default **/*"},
        },
        ["pattern"],
    ),
    _tool(
        "run_command",
        "Run a verification command (tests/linter/formatter). "
        "Only allowlisted commands run; others need human approval, which may "
        "take a while or be denied — continue without it if denied.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    _tool(
        "finish",
        "Declare the task complete and report what you did.",
        {
            "summary": {"type": "string", "description": "what was done and why"},
            "concerns": {"type": "string", "description": "doubts, TODOs, risks"},
        },
        ["summary"],
    ),
]

TOOLSET = ToolSet.from_tools(TOOLS)

SYSTEM_TEMPLATE = """You are sous, a focused coding subcontractor working alone \
inside one project. Complete the task below exactly as instructed.

Rules:
- Make minimal changes. Follow the existing code style.
- Read a file before editing it. Never invent file contents.
- Use run_command only for tests, linters, and formatters.
- run_command executes one single command from the project root, without a
  shell: never use 'cd', '&&', pipes, or redirection.
- When the task is complete (or you cannot proceed), call finish with an honest summary.

Project root: {root}
Top-level entries:
{listing}
"""


def build_system_prompt(project_root: Path) -> str:
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in project_root.iterdir())[:50]
    return SYSTEM_TEMPLATE.format(root=project_root, listing="\n".join(entries))
