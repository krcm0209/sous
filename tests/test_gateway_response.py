"""Turn output → Anthropic content blocks and stream events. Pure."""

import json
import re

import pytest

from sous.engine.base import Delta
from sous.gateway.response import (
    TextSplitter,
    TurnAssembler,
    new_message_id,
    new_tool_use_id,
    stop_reason,
)
from sous.protocol import _MAX_ARGUMENT_DEPTH, ToolSet

TOOLS = ToolSet.from_tools(
    [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
        }
    ],
    strict=False,
)
XML_CALL = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _stream(
    text_pieces: list[str], toolset: ToolSet = TOOLS, finish: str = "stop"
) -> tuple[TurnAssembler, list[dict]]:
    """Feed pieces as deltas, then finish with the joined text."""
    a = TurnAssembler("msg_x", "sous-local", toolset)
    events = a.start(100)
    for n, piece in enumerate(text_pieces, start=1):
        events += a.feed(Delta(piece, n, None))
    events += a.finish("".join(text_pieces), len(text_pieces), finish)
    return a, events


# --- TextSplitter -------------------------------------------------------------------


def test_splitter_emits_plain_text_but_holds_trailing_whitespace_and_tag_prefixes():
    s = TextSplitter()
    assert s.feed("Hello") == "Hello"
    assert s.feed(" world\n\n") == " world"  # trailing newlines held
    assert s.feed("<tool") == ""  # could be <tool_call>
    assert (
        s.feed("s are fun") == "\n\n<tools are fun"
    )  # it wasn't: released, with the held newlines
    assert s.finish() == ""


def test_splitter_drops_leading_whitespace():
    s = TextSplitter()
    assert s.feed("\n\n") == ""
    assert s.feed("  Hi") == "Hi"


def test_splitter_stops_at_tool_call_and_finish_returns_the_tail():
    s = TextSplitter()
    assert s.feed("Reading now.\n\n<tool_") == "Reading now."
    assert s.feed("call>\n<function=Read>") == ""
    assert s.feed("\n</function>\n</tool_call>") == ""
    # The blank lines before the tag ride with the tail, so an unparseable
    # call can be returned verbatim.
    assert s.finish() == "\n\n<tool_call>\n<function=Read>\n</function>\n</tool_call>"


def test_splitter_finish_flushes_held_text_when_no_call_came():
    s = TextSplitter()
    assert s.feed("Done <") == "Done"
    assert s.finish() == " <"


# --- TurnAssembler: streaming ---------------------------------------------------------


def test_text_only_turn_streams_one_text_block():
    a, events = _stream(["Hel", "lo", "!"])
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"] == {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "sous-local",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    assert events[1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    assert [e["delta"]["text"] for e in events[2:5]] == ["Hel", "lo", "!"]
    assert events[5] == {"type": "content_block_stop", "index": 0}
    assert events[6] == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    }
    assert a.message()["content"] == [{"type": "text", "text": "Hello!"}]
    assert a.message()["stop_reason"] == "end_turn"


def test_tool_call_after_prose_streams_text_then_one_buffered_tool_use_block():
    """Decision 4: prose streams; the call is parsed whole and emitted as
    start(input: {}) → one input_json_delta → stop, indices contiguous."""
    pieces = [
        "I will read",
        " a.py.\n\n",
        "<tool_",
        "call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n</function>\n</tool_call>",
    ]
    a, events = _stream(pieces)
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [e["delta"]["text"] for e in events[2:4]] == ["I will read", " a.py."]
    start = events[5]
    assert start["index"] == 1
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "Read"
    assert start["content_block"]["input"] == {}
    assert re.fullmatch(r"toolu_[0-9a-f]{24}", start["content_block"]["id"])
    assert events[6] == {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": json.dumps({"file_path": "a.py"})},
    }
    assert events[8]["delta"]["stop_reason"] == "tool_use"
    assert a.message()["content"] == [
        {"type": "text", "text": "I will read a.py."},
        {
            "type": "tool_use",
            "id": start["content_block"]["id"],
            "name": "Read",
            "input": {"file_path": "a.py"},
        },
    ]


def test_tool_call_alone_produces_no_phantom_text_block():
    """Checklist item 6: leading newlines before the call never open a text
    block; the tool_use block takes index 0."""
    a, events = _stream(["\n", XML_CALL])
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["index"] == 0 and events[1]["content_block"]["type"] == "tool_use"
    assert a.message()["content"][0]["type"] == "tool_use"


def test_two_tool_calls_get_consecutive_indices():
    a, events = _stream([XML_CALL + "\n" + XML_CALL])
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert len({s["content_block"]["id"] for s in starts}) == 2


def test_trailing_prose_after_a_tool_call_is_dropped():
    a, _ = _stream([XML_CALL + "\nDone!"])
    assert [b["type"] for b in a.message()["content"]] == ["tool_use"]


def test_unparseable_tool_call_comes_back_as_text_with_end_turn():
    """The model's malformed turn is shown, not hidden."""
    raw = "Sure.\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed"
    a, events = _stream(
        ["Sure.\n\n", "<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed"]
    )
    assert a.message()["content"] == [{"type": "text", "text": raw}]
    assert a.stop_reason == "end_turn"
    assert [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"] == [
        "Sure.",
        "\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed",
    ]


def test_non_finite_number_argument_comes_back_as_text_with_end_turn():
    """A NaN argument must never reach json.dumps(allow_nan=False) — the
    property Starlette's JSONResponse enforces and the 500 in the bug report
    violated."""
    number_tools = ToolSet.from_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "parameters": {
                        "type": "object",
                        "properties": {"offset": {"type": "number"}},
                    },
                },
            }
        ],
        strict=False,
    )
    raw = (
        "<tool_call>\n<function=Read>\n<parameter=offset>\nNaN\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    a, _ = _stream([raw], toolset=number_tools)
    assert a.message()["content"] == [{"type": "text", "text": raw}]
    assert a.stop_reason == "end_turn"
    json.dumps(a.message(), allow_nan=False)


def test_tool_call_argument_past_the_depth_cap_comes_back_as_text_with_end_turn():
    """An argument nested past `_MAX_ARGUMENT_DEPTH` is a ParseError from
    `parse_tool_calls`, which TurnAssembler.finish falls back to plain text
    for — the same path a ParseError from ordinary malformed JSON takes.
    (This used to be the RecursionError a ~100k-deep argument raised, which
    is stack-size dependent rather than a real depth contract.)"""
    n = _MAX_ARGUMENT_DEPTH + 1
    nested = "[" * n + "]" * n
    raw = '<tool_call>{"name": "Read", "arguments": {"x": ' + nested + "}}</tool_call>"
    a, _ = _stream([raw])
    assert a.message()["content"] == [{"type": "text", "text": raw}]
    assert a.stop_reason == "end_turn"


def test_unknown_tool_name_passes_through_as_tool_use():
    text = (
        "<tool_call>\n<function=Imaginary>\n<parameter=x>\n1\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    a, _ = _stream([text])
    assert a.message()["content"] == [
        {
            "type": "tool_use",
            "id": a.message()["content"][0]["id"],
            "name": "Imaginary",
            "input": {"x": "1"},
        }
    ]


def test_empty_reply_yields_one_empty_text_block():
    a, events = _stream(["", ""])
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert a.message()["content"] == [{"type": "text", "text": ""}]


def test_length_finish_maps_to_max_tokens_unless_a_call_parsed():
    a, _ = _stream(["cut off mid"], finish="length")
    assert a.stop_reason == "max_tokens"
    b, _ = _stream([XML_CALL], finish="length")
    assert b.stop_reason == "tool_use"
    assert stop_reason(None, False) == "end_turn"


def test_final_empty_delta_still_updates_output_tokens():
    """Both mlx libraries flush the detokenizer with a possibly empty final
    piece that carries the real total and the finish reason."""
    a = TurnAssembler("msg_x", "sous-local", TOOLS)
    a.start(1)
    a.feed(Delta("hi", 1, None))
    a.feed(Delta("", 2, "stop"))
    a.finish("hi", 2, "stop")
    assert a.output_tokens == 2
    assert a.message()["usage"] == {"input_tokens": 1, "output_tokens": 2}


# --- non-streaming path is the same code path ----------------------------------------


@pytest.mark.parametrize("split", [1, 3, 7, 12])
def test_non_streaming_message_equals_the_streamed_one(split):
    text = "Let me look.\n\n" + XML_CALL
    a = TurnAssembler("msg_x", "sous-local", TOOLS)
    a.start(5)
    pieces = [text[i : i + split] for i in range(0, len(text), split)]
    for n, piece in enumerate(pieces, start=1):
        a.feed(Delta(piece, n, None))
    a.finish(text, len(pieces), "stop")
    b = TurnAssembler("msg_x", "sous-local", TOOLS)
    b.start(5)
    b.finish(text, len(pieces), "stop")

    def strip_ids(m: dict) -> dict:
        return {**m, "content": [{k: v for k, v in b.items() if k != "id"} for b in m["content"]]}

    assert strip_ids(a.message()) == strip_ids(b.message())
    assert b.message()["content"][0] == {"type": "text", "text": "Let me look."}


def test_message_shape():
    a, _ = _stream(["ok"])
    m = a.message()
    assert set(m) == {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "usage",
    }
    assert m["type"] == "message" and m["role"] == "assistant" and m["stop_sequence"] is None


def test_ids_have_anthropic_shapes():
    assert re.fullmatch(r"msg_[0-9a-f]{24}", new_message_id())
    assert re.fullmatch(r"toolu_[0-9a-f]{24}", new_tool_use_id())
    assert new_message_id() != new_message_id()
