"""Turn output → Anthropic content blocks and stream events.

Pure. `TurnAssembler` is fed the engine's deltas as they arrive and emits
Anthropic stream events (as dicts; SSE framing is the route's job); the same
object then yields the non-streaming Message, so both response shapes come
from one code path — Claude Code retries a failed stream non-streaming with a
byte-identical body, and the two must agree.
"""

from __future__ import annotations

import json
import secrets

from sous.engine.base import Delta
from sous.protocol import ParseError, ToolCall, ToolSet, parse_tool_calls

_OPEN = "<tool_call>"


def new_message_id() -> str:
    return "msg_" + secrets.token_hex(12)


def new_tool_use_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def stop_reason(finish_reason: str | None, has_calls: bool) -> str:
    if has_calls:
        return "tool_use"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def _tag_prefix_len(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of
    `<tool_call>` — text that must wait for the next delta to be classified."""
    for n in range(min(len(_OPEN) - 1, len(text)), 0, -1):
        if text.endswith(_OPEN[:n]):
            return n
    return 0


class TextSplitter:
    """Decides, delta by delta, which text is safe to send as `text_delta`.

    Three things wait until more text settles them: a suffix that could be
    the start of `<tool_call>` (the tag arrives split across deltas), trailing
    whitespace (the turn's text is `.strip()`ped, and the model puts newlines
    before its tool call), and everything from the first `<tool_call>` on
    (tool calls are parsed whole, at the end). Leading whitespace is dropped
    so a text block never opens with a blank — the phantom block of checklist
    item 6.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._started = False
        self._tail = ""  # from the first <tool_call> on; feed() never emits it

    def feed(self, text: str) -> str:
        if self._tail:
            self._tail += text
            return ""
        self._pending += text
        if not self._started:
            self._pending = self._pending.lstrip()
        idx = self._pending.find(_OPEN)
        if idx != -1:
            # The whitespace between the prose and the tag goes with the tail:
            # if the call turns out unparseable, the raw text is returned
            # verbatim, blank lines included.
            out = self._pending[:idx].rstrip()
            self._tail = self._pending[len(out) :]
            self._pending = ""
        else:
            tag = _tag_prefix_len(self._pending)
            body = self._pending[: len(self._pending) - tag]
            hold = tag + (len(body) - len(body.rstrip()))
            cut = len(self._pending) - hold
            out, self._pending = self._pending[:cut], self._pending[cut:]
        self._started = self._started or bool(out)
        return out

    def finish(self) -> str:
        """Everything not yet emitted: held text plus the raw tool-call tail."""
        out = self._pending + self._tail
        self._pending = self._tail = ""
        return out if self._started else out.lstrip()


class TurnAssembler:
    """Builds one assistant message from deltas (streaming) or from the final
    text alone (non-streaming), emitting Anthropic stream events either way.

    Block order: a text block if there is non-blank prose, then one tool_use
    block per parsed call, indices contiguous from 0. An empty reply gets one
    empty text block (what the API itself returns). Prose after a tool call is
    dropped — the template forbids it. A tool call the parser cannot read is
    returned as text with end_turn: the model's malformed turn is shown, not
    hidden.
    """

    def __init__(self, message_id: str, model: str, toolset: ToolSet):
        self.message_id = message_id
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self._toolset = toolset
        self._splitter = TextSplitter()
        self._fed = ""
        self._blocks: list[dict] = []
        self._text_index: int | None = None

    def start(self, input_tokens: int) -> list[dict]:
        self.input_tokens = input_tokens
        return [
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }
        ]

    def feed(self, delta: Delta) -> list[dict]:
        self._fed += delta.text
        self.output_tokens = max(self.output_tokens, delta.output_tokens)
        return self._emit_text(self._splitter.feed(delta.text))

    def finish(self, text: str, output_tokens: int, finish_reason: str | None) -> list[dict]:
        """Close the turn. `text` is the engine's whole reply; whatever the
        deltas did not carry (all of it, on the non-streaming path) is fed
        first so both paths see identical text."""
        events: list[dict] = []
        if len(text) > len(self._fed):
            events += self._emit_text(self._splitter.feed(text[len(self._fed) :]))
            self._fed = text
        self.output_tokens = max(self.output_tokens, output_tokens)
        remainder = self._splitter.finish()
        calls: list[ToolCall] | None
        try:
            calls = parse_tool_calls(text, self._toolset)
        except ParseError:
            calls = None
        if calls:
            events += self._close_text()
            for call in calls:
                index = len(self._blocks)
                block = {
                    "type": "tool_use",
                    "id": new_tool_use_id(),
                    "name": call.name,
                    "input": call.arguments,
                }
                self._blocks.append(block)
                events.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    }
                )
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(call.arguments),
                        },
                    }
                )
                events.append({"type": "content_block_stop", "index": index})
        else:
            events += self._emit_text(remainder.rstrip())
            events += self._close_text()
            if not self._blocks:
                self._blocks.append({"type": "text", "text": ""})
                events.append(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                events.append({"type": "content_block_stop", "index": 0})
        self.stop_reason = stop_reason(finish_reason, bool(calls))
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            }
        )
        events.append({"type": "message_stop"})
        return events

    def message(self) -> dict:
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [dict(block) for block in self._blocks],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            # No cache accounting fields (the SDKs read absent ones as null):
            # a truthful split needs the per-turn reuse count, which arrives
            # with keyed cache slots (Phase 3a). message_start says the same.
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        }

    def _emit_text(self, text: str) -> list[dict]:
        if not text:
            return []
        events: list[dict] = []
        if self._text_index is None:
            self._text_index = len(self._blocks)
            self._blocks.append({"type": "text", "text": ""})
            events.append(
                {
                    "type": "content_block_start",
                    "index": self._text_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        self._blocks[self._text_index]["text"] += text
        events.append(
            {
                "type": "content_block_delta",
                "index": self._text_index,
                "delta": {"type": "text_delta", "text": text},
            }
        )
        return events

    def _close_text(self) -> list[dict]:
        if self._text_index is None:
            return []
        index, self._text_index = self._text_index, None
        return [{"type": "content_block_stop", "index": index}]
