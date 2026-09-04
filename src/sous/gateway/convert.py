"""Anthropic Messages requests → the chat messages and tools the engine renders.

Pure: no engine, no mlx, no I/O. Everything Claude Code sends that the local
model cannot use is dropped here, deliberately and visibly (the spec's
accommodation checklist): tools that carry no schema, thinking blocks, images,
and the per-request volatile markers that would otherwise defeat the prefix
cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A client tool's `type` is absent, null, or "custom", and it comes with its own
# input_schema. Every other value names an Anthropic-defined tool: some genuinely
# server-side (web_search_*, web_fetch_*, code_execution_*, tool_search_tool_*),
# the rest client-executed built-ins whose schema is implicit (bash_*,
# text_editor_*, computer_*, memory_*, browser_toolset_*). Both kinds are dropped
# for one reason — no client-supplied schema, and the local chat template can
# only offer a tool it can render a schema for. The frontier model knows the
# built-ins from training; the local model does not. Translating them into
# explicit schemas is future work, and nothing is lost meanwhile: driving a
# non-claude model, Claude Code sends custom-typed equivalents instead
# (Read/Write/Bash with schemas, observed in a live session).
_CLIENT_TOOL_TYPES = (None, "custom")

# The first system block of a Claude Code request: per-request random values
# (checklist item 4), so it can never be part of a reusable prefix.
_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"

# Claude Code appends a freshly decremented copy of this marker on every
# request and keeps the stale ones (oMLX measured full ~60k-token re-prefills
# per turn until it was stripped). Nothing downstream reads it. The core
# pattern is oMLX's, verbatim; the MULTILINE line-anchoring (^...$) is ours —
# Claude Code always emits the marker on a line of its own (a standalone
# system-prompt line, a marker-only text block after tool results, or its own
# line inside a <system-reminder>), so anchoring keeps the prefix-stability
# benefit for every real shape while leaving a user's inline mention of the
# marker text alone. It still eats up to two preceding newlines with the
# marker, and tolerates trailing spaces or a CR before the line end so a
# marker with an unexpected line ending cannot stay behind and thrash the cache.
_TOTAL_TOKENS_RE = re.compile(r"(?m)\n{0,2}^<total_tokens>\d+ tokens left</total_tokens>[ \t]*\r?$")
# One of Claude Code's assembly paths wraps that marker in a system-reminder;
# stripping the marker leaves the hollow shell, which goes too.
_EMPTY_REMINDER_RE = re.compile(r"\n{0,2}<system-reminder>\s*</system-reminder>")

_ROLES = ("user", "assistant", "system")
_OMITTED = "[{kind} omitted: sous serves text only]"


class RequestError(Exception):
    """A request the gateway rejects, in Anthropic's error vocabulary."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message

    def body(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message}}


def _invalid(message: str) -> RequestError:
    return RequestError(400, "invalid_request_error", message)


@dataclass
class ChatRequest:
    """What the engine needs from a /v1/messages body, plus what the response
    has to echo (`model`) and the log has to mention (the `type` of every
    dropped tool — never its client-supplied name)."""

    model: str
    messages: list[dict]
    tools: list[dict]
    max_tokens: int
    stream: bool
    dropped_tool_types: list[str] = field(default_factory=list)


def strip_volatile(text: str) -> str:
    if "<total_tokens>" not in text:
        return text
    result = _EMPTY_REMINDER_RE.sub("", _TOTAL_TOKENS_RE.sub("", text))
    # MULTILINE `$` matches before the marker's own trailing newline rather
    # than consuming it, so a marker-only text still leaves that newline (or
    # CRLF) behind; collapse a wholly-whitespace residue so `_user_turns`
    # doesn't manufacture a blank turn from it.
    return "" if not result.strip() else result


def _text_parts(content: object, *, drop_billing: bool) -> list[str]:
    """The text of a string-or-blocks content value; non-text blocks skipped."""
    if isinstance(content, str):
        # Same drop as the block form below: Claude Code sends the header as a
        # block today, but a string system prompt is the same per-request noise.
        if drop_billing and content.startswith(_BILLING_HEADER_PREFIX):
            return []
        return [content] if content else []
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        if drop_billing and text.startswith(_BILLING_HEADER_PREFIX):
            continue
        parts.append(text)
    return parts


def _system_text(system: object, messages: list[dict]) -> str:
    """One system prompt from the canonical field plus every inline system
    message, canonical first. Qwen's template accepts a system message only
    at index 0 (it raises otherwise), so this is the only place they can go."""
    parts = _text_parts(system, drop_billing=True) if system is not None else []
    for msg in messages:
        if msg.get("role") == "system":
            parts.extend(_text_parts(msg.get("content"), drop_billing=True))
    return strip_volatile("\n\n".join(parts))


def _tool_result_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") in ("image", "document"):
                parts.append(_OMITTED.format(kind=item["type"]))
        return "\n".join(parts)
    return str(content)


def _user_turns(content: object) -> list[dict]:
    if isinstance(content, str):
        text = strip_volatile(content)
        # A message that was only the marker contributes nothing; a genuinely
        # empty message keeps its (empty) turn.
        return [{"role": "user", "content": text}] if text or not content else []
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    out: list[dict] = []
    texts: list[str] = []
    recognized = False

    def flush() -> None:
        # Claude Code appends the <total_tokens> marker as its own text block
        # after every tool-result batch. Once stripped there is nothing left,
        # and an empty user turn right after the tool responses would read to
        # the model as "the user said nothing".
        text = strip_volatile("\n".join(texts))
        texts.clear()
        if text:
            out.append({"role": "user", "content": text})

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            recognized = True
            texts.append(str(block.get("text", "")))
        elif kind == "tool_result":
            # The template groups consecutive tool messages into one user turn
            # and matches results to calls by order only, so text that came
            # before a result must be its own turn to keep the sequence.
            recognized = True
            flush()
            text = _tool_result_text(block.get("content"))
            if block.get("is_error") is True:
                # Without this, a failed and a successful tool execution render
                # identically to the local model; exactly `True` per the
                # Anthropic field (anything else, including absent, is not an
                # error).
                text = f"[tool error]\n{text}" if text else "[tool error]"
            out.append({"role": "tool", "content": text})
        elif kind in ("image", "document"):
            recognized = True
            texts.append(_OMITTED.format(kind=kind))
        # thinking, redacted_thinking, server_tool_use, *_tool_result and
        # anything newer: dropped (checklist item 8 — tolerate, never 4xx).
    flush()
    if not out and not recognized:
        out.append({"role": "user", "content": ""})
    return out


def _assistant_turn(content: object) -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    texts: list[str] = []
    calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            arguments = block.get("input")
            # The template iterates arguments as a mapping (a JSON string
            # raises inside Jinja), so anything else becomes an empty call.
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    },
                }
            )
        # thinking / redacted_thinking: the local model never produced them
        # and cannot verify them; dropped.
    turn: dict = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        turn["tool_calls"] = calls
    return turn


def chat_messages(system: object, messages: list[dict]) -> list[dict]:
    """The chat-template message list for a request. Raises RequestError."""
    out: list[dict] = []
    prompt = _system_text(system, messages)
    if prompt:
        out.append({"role": "system", "content": prompt})
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # folded into the system prompt above
        if role == "user":
            out.extend(_user_turns(msg.get("content")))
        elif role == "assistant":
            out.append(_assistant_turn(msg.get("content")))
        else:
            raise _invalid(f"messages: role must be one of {_ROLES}, got {role!r}")
    return out


def chat_tools(tools: object) -> tuple[list[dict], list[str]]:
    """OpenAI-style function schemas for the template, plus the `type` of each
    tool dropped on the way (for the log).

    The type is one of Anthropic's fixed identifiers; the name is free-form
    client text out of the request body, so it is not returned at all."""
    if tools is None:
        return [], []
    if not isinstance(tools, list):
        raise _invalid("tools must be a list")
    out: list[dict] = []
    dropped: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise _invalid("tools: each entry must be an object")
        tool_type = tool.get("type")
        if tool_type is not None and not isinstance(tool_type, str):
            # The API's `type` is a string; a non-string is a shape error, not
            # a value to log — str()-ing it here would put arbitrary body
            # content into the daemon log via dropped_tool_types.
            raise _invalid("tools: type must be a string")
        if tool_type not in _CLIENT_TOOL_TYPES:
            dropped.append(tool_type)
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name:
            raise _invalid("tools: each client tool needs a name")
        if not isinstance(schema, dict):
            raise _invalid(f"tools: {name} needs an input_schema object")
        props = schema.get("properties")
        if props is not None and not isinstance(props, dict):
            raise _invalid(f"tools: {name} input_schema.properties must be an object")
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": schema,
                },
            }
        )
    return out, dropped


def parse_messages_request(body: object) -> ChatRequest:
    """Validate a /v1/messages body and convert it. Unknown top-level fields
    (thinking, output_config, context_management, metadata, temperature,
    top_p, top_k, stop_sequences, tool_choice, ...) are ignored: sampling is
    the daemon's [model] configuration and thinking is off."""
    if not isinstance(body, dict):
        raise _invalid("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _invalid("model: field required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages: at least one message is required")
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in _ROLES:
            raise _invalid(f"messages: each message needs a role in {_ROLES}")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise _invalid("max_tokens: must be a positive integer")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise _invalid("stream: must be true or false")
    tools, dropped = chat_tools(body.get("tools"))
    return ChatRequest(
        model, chat_messages(body.get("system"), messages), tools, max_tokens, stream, dropped
    )


def parse_count_tokens_request(body: object) -> ChatRequest:
    """count_tokens has neither max_tokens nor stream; otherwise identical."""
    if not isinstance(body, dict):
        raise _invalid("request body must be a JSON object")
    return parse_messages_request({**body, "max_tokens": 1, "stream": False})
