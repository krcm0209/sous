"""Worker-facing tool schemas and the tool-call parser (Qwen hermes format)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


WORKER_TOOLS: list[dict] = [
    _tool("read_file", "Read a file (line-numbered). Use offset/limit for large files.",
          {"path": {"type": "string"},
           "offset": {"type": "integer", "description": "0-based start line"},
           "limit": {"type": "integer", "description": "max lines"}},
          ["path"]),
    _tool("write_file", "Create or overwrite a file with the given content.",
          {"path": {"type": "string"}, "content": {"type": "string"}},
          ["path", "content"]),
    _tool("edit_file", "Replace one exact, unique occurrence of `old` with `new`.",
          {"path": {"type": "string"}, "old": {"type": "string"},
           "new": {"type": "string"}},
          ["path", "old", "new"]),
    _tool("list_dir", "List one directory's entries.",
          {"path": {"type": "string", "description": "default: project root"}}, []),
    _tool("glob", "Find files by glob pattern, e.g. **/*.py",
          {"pattern": {"type": "string"}}, ["pattern"]),
    _tool("grep", "Regex-search file contents. Returns path:line:text hits.",
          {"pattern": {"type": "string"},
           "glob_pattern": {"type": "string", "description": "default **/*"}},
          ["pattern"]),
    _tool("run_command", "Run a verification command (tests/linter/formatter). "
          "Only allowlisted commands run; others need human approval, which may "
          "take a while or be denied — continue without it if denied.",
          {"command": {"type": "string"}}, ["command"]),
    _tool("finish", "Declare the task complete and report what you did.",
          {"summary": {"type": "string", "description": "what was done and why"},
           "concerns": {"type": "string", "description": "doubts, TODOs, risks"}},
          ["summary"]),
]

TOOL_NAMES = {t["function"]["name"] for t in WORKER_TOOLS}

_OPEN_RE = re.compile(r"<tool_call>\s*")
_DECODER = json.JSONDecoder()


class ParseError(Exception):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _OPEN_RE.finditer(text):
        try:
            payload, _end = _DECODER.raw_decode(text, match.end())
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON in tool_call: {e}") from e
        if not isinstance(payload, dict):
            raise ParseError("tool_call payload must be a JSON object")
        name = payload.get("name")
        if name not in TOOL_NAMES:
            raise ParseError(f"unknown tool: {name!r}")
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ParseError("tool_call arguments must be a JSON object")
        calls.append(ToolCall(name, arguments))
    return calls
