"""Worker-facing tool schemas and the tool-call parser.

A parsed call's name is validated against a `ToolSet` — `WORKER_TOOLSET` by
default, or the request's own tools in the gateway, which parses calls
against whatever tool set Claude Code sent rather than the worker's eight.

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
import math
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


class ParseError(Exception):
    pass


def _reject_constant(name: str) -> None:
    """json's parse_constant hook for NaN/Infinity/-Infinity: without it,
    json.loads silently accepts these non-finite tokens (`float("nan")` and
    friends succeed), and json.dumps later re-emits the invalid `NaN` /
    `Infinity` literals that break every JSON consumer downstream, including
    Starlette's JSONResponse (a 500) and a streamed partial_json token."""
    raise ParseError(f"non-finite number {name!r} is not valid JSON")


def _schema_type(prop: object) -> str:
    """The JSON-schema type coercion should target. A union `type` list (e.g.
    `["integer", "string"]`, `["integer", "null"]`) becomes every member
    joined by `|` in schema order, so `_coerce` can try each member in turn
    and prefer the one the schema lists first; a property without a `type`
    (enum, anyOf, free-form) stays a raw string."""
    if not isinstance(prop, dict):
        return "string"
    typ = prop.get("type", "string")
    if isinstance(typ, list):
        members = [t for t in typ if isinstance(t, str)]
        return "|".join(members) if members else "string"
    return typ if isinstance(typ, str) else "string"


@dataclass(frozen=True)
class ToolSet:
    """The tools one generation may call, and how to type their arguments.

    The XML-ish format writes string arguments raw and everything else via
    tojson, so the schema is the only way to know that e.g. read_file's offset
    must become an int — hence the per-tool parameter types.

    `strict` is the worker's contract: a name outside the set is a malformed
    turn, handled by FORMAT_REMINDER. The gateway is not strict — Claude Code
    answers a hallucinated tool with its own tool-not-found result, which the
    model can recover from, whereas ending the turn here could not be undone.
    An unknown tool then coerces nothing: every parameter stays a string.
    """

    names: frozenset[str]
    param_types: dict[str, dict[str, str]]
    strict: bool = True

    @classmethod
    def from_tools(cls, tools: list[dict], *, strict: bool = True) -> ToolSet:
        param_types = {
            t["function"]["name"]: {
                key: _schema_type(prop)
                for key, prop in (
                    (t["function"].get("parameters") or {}).get("properties") or {}
                ).items()
            }
            for t in tools
        }
        return cls(frozenset(param_types), param_types, strict)

    def check(self, name: str) -> dict[str, str]:
        """Parameter types for `name`; ParseError when strict and unknown."""
        if name in self.param_types:
            return self.param_types[name]
        if self.strict:
            raise ParseError(f"unknown tool: {name!r}")
        return {}


WORKER_TOOLSET = ToolSet.from_tools(WORKER_TOOLS)

_OPEN_RE = re.compile(r"<tool_call>\s*")
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>")
_WS_RE = re.compile(r"\s*")


def _finite_float(text: str) -> float:
    """json's parse_float hook: an overflowing literal such as `1e999` is an
    ordinary number to the scanner (parse_constant never sees it) and becomes
    `inf`, which is exactly as unserialisable as `Infinity`."""
    value = float(text)
    if not math.isfinite(value):
        raise ParseError(f"non-finite number {text!r} is not valid JSON")
    return value


_DECODER = json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float)


def _skip_ws(text: str, pos: int) -> int:
    """Index of the first non-whitespace char at or after `pos`.

    `\\s*` matches at every position (possibly zero-width), so the match can
    never be None — the check is here to make that invariant checkable rather
    than assumed."""
    match = _WS_RE.match(text, pos)
    assert match is not None
    return match.end()


@dataclass
class ToolCall:
    name: str
    arguments: dict


def parse_tool_calls(text: str, toolset: ToolSet = WORKER_TOOLSET) -> list[ToolCall]:
    calls: list[ToolCall] = []
    pos = 0
    while (match := _OPEN_RE.search(text, pos)) is not None:
        start = match.end()
        first = text[start : start + 1]
        if first == "{":
            call, pos = _parse_json_call(text, start, toolset)
        elif first == "<":
            call, pos = _parse_xml_call(text, start, toolset)
        else:
            raise ParseError(
                "tool_call must contain a JSON object or a <function=...> "
                f"block, got {text[start : start + 20]!r}"
            )
        calls.append(call)
    return calls


def _parse_json_call(text: str, start: int, toolset: ToolSet) -> tuple[ToolCall, int]:
    """Hermes JSON: {"name": ..., "arguments": {...}}. Returns (call, end)."""
    try:
        payload, end = _DECODER.raw_decode(text, start)
    except (ValueError, RecursionError) as e:
        # ValueError covers json.JSONDecodeError (an ordinary malformed
        # tool_call); it also covers a 5000+ digit integer literal, which
        # exceeds Python's int-from-string digit cap. RecursionError is a
        # pathologically deep argument (~100k nested arrays) blowing the C
        # decoder's stack. None of these used to be ParseError, so each
        # escaped parse_tool_calls uncaught — a bare 500 non-streaming, a
        # truncated SSE stream — instead of the malformed-turn fallback
        # TurnAssembler.finish only catches ParseError for.
        raise ParseError(f"invalid JSON in tool_call: {e}") from e
    if not isinstance(payload, dict):
        raise ParseError("tool_call payload must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError(f"tool_call name must be a non-empty string, got {name!r}")
    toolset.check(name)
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ParseError("tool_call arguments must be a JSON object")
    return ToolCall(name, arguments), end


def _parse_xml_call(text: str, start: int, toolset: ToolSet) -> tuple[ToolCall, int]:
    """Qwen3 XML-ish: <function=NAME><parameter=KEY>...</parameter></function>.

    Returns (call, end) where end is past the closing </tool_call> so tag
    text embedded in argument values is never re-scanned as a new call.
    """
    fn = _FUNCTION_RE.match(text, start)
    if fn is None:
        raise ParseError("expected <function=NAME> after <tool_call>")
    name = fn.group(1)
    param_types = toolset.check(name)

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
    """Coerce a raw XML-ish parameter value against its declared schema type.

    `typ` may name a union (`"integer|string"`, `"integer|null"`) — each
    member is tried in schema order and the first that accepts `raw` wins,
    matching what a Hermes JSON call would have received untouched (a
    union's `"string"` member always accepts; its `"null"` member accepts
    only the literal `null` text). A single-member `typ` coerces directly,
    with the same error message the pre-union code raised, since a lone
    type has nothing to fall back to.
    """
    members = typ.split("|")
    if len(members) == 1:
        return _coerce_member(tool, key, members[0], raw)
    for member in members:
        try:
            return _coerce_member(tool, key, member, raw)
        except ParseError:
            continue
    raise ParseError(f"parameter {key!r} of {tool} must be one of {typ}, got {raw!r}")


def _coerce_member(tool: str, key: str, typ: str, raw: str):
    """Coerce `raw` against one union member. Strings stay raw (the template
    writes them unquoted); non-strings were written via tojson. Failing
    loudly here beats handing the executor offset="5"."""
    if typ == "null":
        if raw.strip() == "null":
            return None
        raise ParseError(f"parameter {key!r} of {tool} must be null, got {raw!r}")
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
            value = float(raw.strip())
        except ValueError:
            raise ParseError(f"parameter {key!r} of {tool} must be a number, got {raw!r}") from None
        if not math.isfinite(value):
            raise ParseError(f"parameter {key!r} of {tool} must be a finite number, got {raw!r}")
        return value
    if typ in ("array", "object"):
        try:
            value = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
        except ValueError, RecursionError:
            # Same escapes as _parse_json_call's raw_decode: ValueError
            # covers json.JSONDecodeError and an oversized integer literal;
            # RecursionError is a deeply nested value blowing the C
            # decoder's stack. Neither used to fail loudly as ParseError.
            raise ParseError(
                f"parameter {key!r} of {tool} must be a JSON {typ}, got {raw!r}"
            ) from None
        if (typ == "array" and not isinstance(value, list)) or (
            typ == "object" and not isinstance(value, dict)
        ):
            raise ParseError(f"parameter {key!r} of {tool} must be a JSON {typ}, got {raw!r}")
        return value
    return raw
