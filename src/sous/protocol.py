"""Worker-facing tool schemas and the tool-call parser.

Two wire formats are accepted, distinguished by the first non-space
character after ``<tool_call>``:

- ``{`` — hermes JSON: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
  (emitted by the smoke model and other MLX models).
- ``<`` — the XML-ish format Qwen3's chat_template.jinja instructs::

      <tool_call>
      <function=NAME>
      <parameter=KEY>
      value
      </parameter>
      </function>
      </tool_call>
"""

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

TOOL_NAMES = {t["function"]["name"] for t in WORKER_TOOLS}

# Declared parameter types per tool. The XML-ish format writes string
# arguments raw and non-string arguments via tojson, so the schema is the
# only way to know that e.g. read_file's offset must become an int.
_PARAM_TYPES: dict[str, dict[str, str]] = {
    t["function"]["name"]: {
        key: prop.get("type", "string")
        for key, prop in t["function"]["parameters"]["properties"].items()
    }
    for t in WORKER_TOOLS
}

_OPEN_RE = re.compile(r"<tool_call>\s*")
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>")
_WS_RE = re.compile(r"\s*")
_DECODER = json.JSONDecoder()


def _skip_ws(text: str, pos: int) -> int:
    """Index of the first non-whitespace char at or after `pos`.

    `\\s*` matches at every position (possibly zero-width), so the match can
    never be None — the check is here to make that invariant checkable rather
    than assumed."""
    match = _WS_RE.match(text, pos)
    assert match is not None
    return match.end()


class ParseError(Exception):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    pos = 0
    while (match := _OPEN_RE.search(text, pos)) is not None:
        start = match.end()
        first = text[start : start + 1]
        if first == "{":
            call, pos = _parse_json_call(text, start)
        elif first == "<":
            call, pos = _parse_xml_call(text, start)
        else:
            raise ParseError(
                "tool_call must contain a JSON object or a <function=...> "
                f"block, got {text[start : start + 20]!r}"
            )
        calls.append(call)
    return calls


def _parse_json_call(text: str, start: int) -> tuple[ToolCall, int]:
    """Hermes JSON: {"name": ..., "arguments": {...}}. Returns (call, end)."""
    try:
        payload, end = _DECODER.raw_decode(text, start)
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
    return ToolCall(name, arguments), end


def _parse_xml_call(text: str, start: int) -> tuple[ToolCall, int]:
    """Qwen3 XML-ish: <function=NAME><parameter=KEY>...</parameter></function>.

    Returns (call, end) where end is past the closing </tool_call> so tag
    text embedded in argument values is never re-scanned as a new call.
    """
    fn = _FUNCTION_RE.match(text, start)
    if fn is None:
        raise ParseError("expected <function=NAME> after <tool_call>")
    name = fn.group(1)
    if name not in TOOL_NAMES:
        raise ParseError(f"unknown tool: {name!r}")
    param_types = _PARAM_TYPES[name]

    arguments: dict = {}
    pos = fn.end()
    while True:
        pos = _skip_ws(text, pos)
        param = _PARAM_RE.match(text, pos)
        if param is not None:
            key = param.group(1)
            close = text.find("</parameter>", param.end())
            if close == -1:
                raise ParseError(f"unterminated <parameter={key}> block")
            raw = _strip_wrapping_newlines(text[param.end() : close])
            arguments[key] = _coerce(name, key, param_types.get(key, "string"), raw)
            pos = close + len("</parameter>")
        elif text.startswith("</function>", pos):
            pos += len("</function>")
            break
        else:
            raise ParseError(f"expected <parameter=KEY> or </function> in {name} tool_call")
    after = _skip_ws(text, pos)
    if text.startswith("</tool_call>", after):
        pos = after + len("</tool_call>")
    return ToolCall(name, arguments), pos


def _strip_wrapping_newlines(raw: str) -> str:
    """Remove exactly the one newline the chat template adds on each side
    of a parameter value ('<parameter=KEY>\\n' + value + '\\n</parameter>').

    Never .strip(): a write_file content that legitimately ends with a
    newline must keep it, or the worker writes a corrupted file.
    """
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _coerce(tool: str, key: str, typ: str, raw: str):
    """Coerce a raw XML-ish parameter value using its declared schema type.

    Strings stay raw (the template writes them unquoted); non-strings were
    written via tojson. Failing loudly here beats handing the executor
    offset="5".
    """
    if typ == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            raise ParseError(
                f"parameter {key!r} of {tool} must be an integer, got {raw!r}"
            ) from None
    if typ == "boolean":
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ParseError(f"parameter {key!r} of {tool} must be a boolean, got {raw!r}")
    if typ == "number":
        try:
            return float(raw.strip())
        except ValueError:
            raise ParseError(f"parameter {key!r} of {tool} must be a number, got {raw!r}") from None
    return raw
