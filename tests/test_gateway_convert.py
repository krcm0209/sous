"""Anthropic request → chat-template inputs. Pure; every Claude Code
accommodation from the gateway spec's checklist is pinned here."""

import pytest

from sous.gateway.convert import (
    ChatRequest,
    RequestError,
    chat_messages,
    chat_tools,
    parse_count_tokens_request,
    parse_messages_request,
    strip_volatile,
)

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}


def _body(**overrides) -> dict:
    body = {
        "model": "sous-local",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


# --- system prompt: canonical field, inline system messages, volatile markers --


def test_string_system_becomes_the_leading_system_message():
    out = chat_messages("Be terse.", [{"role": "user", "content": "hi"}])
    assert out == [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]


def test_system_blocks_are_joined_and_the_billing_header_block_is_dropped():
    """Checklist item 4: the x-anthropic-billing-header block carries
    per-request random values that would defeat the prefix cache."""
    system = [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cc_is_subagent=true"},
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Rules follow."},
    ]
    out = chat_messages(system, [{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": "You are Claude Code.\n\nRules follow."}


def test_a_string_system_carrying_the_billing_header_is_dropped_too():
    """Same checklist item, the other shape the field accepts: dropped only in
    the block form, the per-request random values would still land in the
    cached prefix, and a string system prompt is a documented shape."""
    header = "x-anthropic-billing-header: cc_version=2.1; cc_is_subagent=true"
    assert chat_messages(header, [{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}
    ]
    # Only the header itself: an ordinary string system prompt still stands.
    assert chat_messages("Be terse.", [])[0] == {"role": "system", "content": "Be terse."}


def test_inline_system_messages_fold_into_the_leading_system_message():
    """Checklist item 1: Claude Code >= 2.1.154 puts system content in
    messages[] (gate 2 saw a 7.8 KB string there); Qwen's template accepts a
    system message only at index 0, canonical field first."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "system", "content": "Inline instructions."},
        {"role": "system", "content": [{"type": "text", "text": "More."}]},
    ]
    out = chat_messages([{"type": "text", "text": "Canonical."}], messages)
    assert out == [
        {"role": "system", "content": "Canonical.\n\nInline instructions.\n\nMore."},
        {"role": "user", "content": "task"},
    ]


def test_inline_system_without_a_canonical_field_still_leads():
    out = chat_messages(
        None, [{"role": "user", "content": "q"}, {"role": "system", "content": "S"}]
    )
    assert out[0] == {"role": "system", "content": "S"}


def test_total_tokens_markers_are_stripped_from_system_and_user_text():
    """Checklist item 3: a freshly decremented copy is appended every request
    and the stale ones kept — without stripping, no prefix past it is reusable."""
    marker = "\n\n<total_tokens>15000000 tokens left</total_tokens>"
    assert strip_volatile("Rules." + marker) == "Rules."
    assert strip_volatile("a" + marker + marker) == "a"
    assert strip_volatile("no marker here") == "no marker here"
    wrapped = "<system-reminder>\n<total_tokens>5 tokens left</total_tokens>\n</system-reminder>"
    assert strip_volatile("Hi.\n\n" + wrapped) == "Hi."
    out = chat_messages("Sys." + marker, [{"role": "user", "content": "Hi." + marker}])
    assert out == [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hi."}]


def test_an_inline_marker_mention_survives_but_an_own_line_marker_still_strips():
    """The regex is line-anchored: Claude Code always emits the real marker on
    a line of its own, so anchoring keeps every real shape stripped while
    leaving a user's inline mention of the marker text (e.g. asking what it
    means) alone."""
    inline = "please explain what <total_tokens>5 tokens left</total_tokens> means here"
    assert strip_volatile(inline) == inline
    assert strip_volatile("A\n\n<total_tokens>9 tokens left</total_tokens>\n\nB") == "A\n\nB"


def test_a_marker_with_trailing_space_or_crlf_still_strips():
    # The line anchor must not let a marker with an odd line ending survive:
    # its ever-changing count would then sit in every turn's prefix.
    assert strip_volatile("Rules.\n\n<total_tokens>9 tokens left</total_tokens> \n") == "Rules.\n"
    crlf = strip_volatile("Rules.\r\n\r\n<total_tokens>9 tokens left</total_tokens>\r\n")
    assert "<total_tokens>" not in crlf
    assert crlf.startswith("Rules.\r\n")


def test_a_marker_only_text_strips_to_nothing_even_with_a_trailing_newline():
    """MULTILINE `$` matches before the marker's own trailing newline rather
    than consuming it, so a marker-only text still leaves that newline behind
    once the marker itself is gone. Left in place it reads to `_user_turns`
    as a truthy `"\\n"` message and manufactures the blank turn the
    marker-only suppression exists to prevent."""
    marker = "<total_tokens>123 tokens left</total_tokens>"
    assert strip_volatile(marker + "\n") == ""
    assert strip_volatile(marker + "\r\n") == ""
    assert strip_volatile(marker + "  \n") == ""
    assert strip_volatile(marker + "\n\n") == ""
    # Real text after the marker's line is untouched — only a wholly
    # whitespace residue collapses.
    assert strip_volatile(marker + "\nHello") == "\nHello"


def test_a_whitespace_only_text_with_no_marker_keeps_its_turn():
    """The new whitespace-only collapse is reachable only through the marker
    branch (the function early-returns when no marker is present at all), so
    a message that is merely blank — never touched the marker path — must
    come through unchanged."""
    assert strip_volatile("   ") == "   "


def test_marker_only_text_after_tool_results_adds_no_user_turn():
    """Claude Code emits the marker as an attachment after every tool-result
    batch (bare, or wrapped in a system-reminder). Stripped, nothing remains,
    and an empty user turn after the tool responses would tell the model the
    user said nothing. A trailing newline (or CRLF, or trailing spaces before
    one) is how Claude Code actually terminates the attachment, and MULTILINE
    `$` matches before that newline rather than consuming it — it must not
    survive as a lone-newline user turn either."""
    marker = "<total_tokens>123 tokens left</total_tokens>"
    wrapped = f"<system-reminder>\n{marker}\n</system-reminder>"
    trailers = (
        marker,
        wrapped,
        marker + "\n",
        marker + "\r\n",
        marker + "  \n",
        marker + "\n\n",
        wrapped + "\n",
    )
    for trailer in trailers:
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
                    {"type": "text", "text": trailer},
                ],
            },
            {"role": "user", "content": trailer},
        ]
        assert chat_messages(None, messages)[2:] == [{"role": "tool", "content": "A"}]


def test_tool_result_content_is_not_marker_stripped():
    """A file the model read may legitimately contain the marker text; only
    the volatile places Claude Code writes it are stripped."""
    text = "<total_tokens>5 tokens left</total_tokens>"
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}],
        },
    ]
    assert chat_messages(None, messages)[-1] == {"role": "tool", "content": text}


# --- user turns -------------------------------------------------------------------


def test_user_text_blocks_join_with_newlines():
    out = chat_messages(
        None,
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ],
    )
    assert out == [{"role": "user", "content": "a\nb"}]


def test_tool_results_become_tool_messages_in_order_with_text_flushed_first():
    """Qwen's template matches results to calls by ORDER (it never renders the
    id) and groups consecutive tool messages into one user turn; text that
    preceded a result must be its own turn to keep that order."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a"}},
                {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "note before"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": [{"type": "text", "text": "B1"}, {"type": "text", "text": "B2"}],
                    "is_error": True,
                },
                {"type": "text", "text": "note after"},
            ],
        },
    ]
    out = chat_messages(None, messages)
    assert out[1] == {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [
            {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "a"}}},
            {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "b"}}},
        ],
    }
    assert out[2:] == [
        {"role": "user", "content": "note before"},
        {"role": "tool", "content": "A"},
        {"role": "tool", "content": "[tool error]\nB1\nB2"},
        {"role": "user", "content": "note after"},
    ]


def test_is_error_absent_or_false_produces_no_marker():
    """Only `is_error` exactly `True` marks a failure; a successful result (the
    field absent, or explicitly False) renders unmarked."""
    for is_error in (None, False):
        content: dict = {"type": "tool_result", "tool_use_id": "t1", "content": "A"}
        if is_error is not None:
            content["is_error"] = is_error
        messages = [
            {"role": "user", "content": "go"},
            {"role": "user", "content": [content]},
        ]
        assert chat_messages(None, messages)[-1] == {"role": "tool", "content": "A"}


def test_is_error_with_empty_text_is_the_marker_alone():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True}],
        },
    ]
    assert chat_messages(None, messages)[-1] == {"role": "tool", "content": "[tool error]"}


def test_images_and_documents_become_placeholders_and_unknown_blocks_are_skipped():
    """Checklist item 8: tolerate, never 4xx. sous serves text only."""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "what is this?"},
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
        {"type": "something_new_20270101", "payload": 1},
    ]
    out = chat_messages(None, [{"role": "user", "content": content}])
    assert out == [
        {
            "role": "user",
            "content": (
                "[image omitted: sous serves text only]\nwhat is this?\n"
                "[document omitted: sous serves text only]"
            ),
        }
    ]


def test_empty_user_content_yields_an_empty_user_turn():
    assert chat_messages(None, [{"role": "user", "content": []}]) == [
        {"role": "user", "content": ""}
    ]


def test_tool_result_without_content_is_an_empty_tool_message():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    ]
    assert chat_messages(None, messages)[-1] == {"role": "tool", "content": ""}


# --- assistant turns --------------------------------------------------------------


def test_assistant_thinking_blocks_are_dropped_and_string_input_becomes_empty_args():
    content = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "redacted_thinking", "data": "..."},
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": "not-a-dict"},
    ]
    out = chat_messages(
        None, [{"role": "user", "content": "q"}, {"role": "assistant", "content": content}]
    )
    assert out[1] == {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": {}}}],
    }


def test_assistant_string_content_passes_through():
    out = chat_messages(
        None, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}]
    )
    assert out[1] == {"role": "assistant", "content": "A"}


def test_unknown_role_is_a_400():
    with pytest.raises(RequestError) as exc:
        chat_messages(None, [{"role": "tool", "content": "x"}])
    assert exc.value.status == 400 and exc.value.error_type == "invalid_request_error"


def test_non_list_non_string_content_is_a_400():
    with pytest.raises(RequestError):
        chat_messages(None, [{"role": "user", "content": {"type": "text", "text": "x"}}])


# --- tools ------------------------------------------------------------------------


def test_client_tools_convert_to_function_schemas():
    tools, dropped = chat_tools([READ_TOOL, {"name": "NoDesc", "input_schema": {"type": "object"}}])
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": READ_TOOL["input_schema"],
            },
        },
        {
            "type": "function",
            "function": {"name": "NoDesc", "description": "", "parameters": {"type": "object"}},
        },
    ]
    assert dropped == []


def test_schemaless_tools_are_dropped_and_only_their_type_reported():
    """Checklist item 7: anything with a non-custom `type` arrives without an
    input_schema — server-side tools and client-executed built-ins alike. The
    `name` is free-form client text, so it is never carried out of here."""
    tools, dropped = chat_tools(
        [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "browser_toolset_20260801", "configs": {}},
            {"type": "bash_20250124", "name": "bash"},
            {"type": "custom", **READ_TOOL},
            READ_TOOL,
        ]
    )
    assert [t["function"]["name"] for t in tools] == ["Read", "Read"]
    assert dropped == ["web_search_20250305", "browser_toolset_20260801", "bash_20250124"]


@pytest.mark.parametrize("bad_type", [{"token": "s3cr3t"}, 5, ["x"], 1.5])
def test_non_string_tool_type_is_a_400_not_logged(bad_type):
    """A non-string `type` must be rejected before the drop logic's str()
    could turn arbitrary body content into a log line."""
    with pytest.raises(RequestError) as exc:
        chat_tools([{"type": bad_type, "name": "X", "input_schema": {"type": "object"}}])
    assert exc.value.status == 400
    assert exc.value.error_type == "invalid_request_error"


@pytest.mark.parametrize("ok_type", [None, "custom"])
def test_none_or_custom_tool_type_still_converts(ok_type):
    tool = {"name": "X", "input_schema": {"type": "object"}}
    if ok_type is not None:
        tool = {"type": ok_type, **tool}
    tools, dropped = chat_tools([tool])
    assert [t["function"]["name"] for t in tools] == ["X"]
    assert dropped == []


def test_absent_tool_type_still_converts():
    tools, dropped = chat_tools([{"name": "X", "input_schema": {"type": "object"}}])
    assert [t["function"]["name"] for t in tools] == ["X"]
    assert dropped == []


def test_client_tool_without_schema_or_name_is_a_400():
    with pytest.raises(RequestError, match="input_schema"):
        chat_tools([{"name": "Broken"}])
    with pytest.raises(RequestError, match="name"):
        chat_tools([{"input_schema": {"type": "object"}}])


@pytest.mark.parametrize("properties", [["a"], "abc"])
def test_non_object_properties_is_a_400(properties):
    """`ToolSet.from_tools` does `(parameters.get("properties") or {}).items()`;
    a non-empty non-object reaches that `.items()` and raises AttributeError
    instead of the shaped 400 the client needs (routes.py calls from_tools
    after the RequestError handling, so an unrejected value there is a bare
    500)."""
    with pytest.raises(RequestError) as exc:
        chat_tools(
            [{"name": "Broken", "input_schema": {"type": "object", "properties": properties}}]
        )
    assert exc.value.status == 400
    assert exc.value.error_type == "invalid_request_error"


def test_empty_list_properties_is_also_rejected():
    """An empty list is falsy and would be harmless downstream, but Anthropic's
    API requires JSON Schema, where `properties` is an object — so this is a
    400 too, not a regression."""
    with pytest.raises(RequestError) as exc:
        chat_tools([{"name": "Broken", "input_schema": {"type": "object", "properties": []}}])
    assert exc.value.status == 400
    assert exc.value.error_type == "invalid_request_error"


def test_absent_or_object_properties_still_convert():
    tools, _ = chat_tools(
        [
            {"name": "NoProps", "input_schema": {"type": "object"}},
            {"name": "ObjProps", "input_schema": {"type": "object", "properties": {}}},
        ]
    )
    assert [t["function"]["name"] for t in tools] == ["NoProps", "ObjProps"]


def test_no_tools_is_fine():
    assert chat_tools(None) == ([], [])


# --- whole requests ---------------------------------------------------------------


def test_parse_messages_request_converts_and_ignores_unknown_fields():
    """Decision 9: thinking, output_config, context_management, metadata and
    the sampling knobs are accepted and ignored; gate 1 saw all of them."""
    body = _body(
        stream=True,
        system=[{"type": "text", "text": "S"}],
        tools=[READ_TOOL],
        thinking={"type": "adaptive", "display": "omitted"},
        output_config={"effort": "high"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        metadata={"user_id": "abc"},
        temperature=1,
        tool_choice={"type": "auto"},
        stop_sequences=["x"],
        totally_new_field=True,
    )
    chat = parse_messages_request(body)
    assert isinstance(chat, ChatRequest)
    assert chat.model == "sous-local" and chat.stream is True and chat.max_tokens == 32000
    assert chat.messages == [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert chat.tools[0]["function"]["name"] == "Read"
    assert chat.dropped_tool_types == []


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ([], "JSON object"),
        (_body(model=""), "model"),
        (_body(messages=[]), "messages"),
        (_body(messages=[{"role": "robot", "content": "x"}]), "role"),
        (_body(messages=["not a dict"]), "role"),
        (_body(max_tokens=0), "max_tokens"),
        (_body(max_tokens=True), "max_tokens"),
        ({k: v for k, v in _body().items() if k != "max_tokens"}, "max_tokens"),
        (_body(stream="yes"), "stream"),
        (_body(tools="nope"), "tools"),
    ],
)
def test_parse_messages_request_rejects_bad_shapes_with_400(body, fragment):
    with pytest.raises(RequestError) as exc:
        parse_messages_request(body)
    assert exc.value.status == 400
    assert exc.value.error_type == "invalid_request_error"
    assert fragment in exc.value.message
    assert exc.value.body() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": exc.value.message},
    }


def test_parse_count_tokens_request_needs_no_max_tokens_or_stream():
    body = {
        "model": "sous-local",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [READ_TOOL],
    }
    chat = parse_count_tokens_request(body)
    assert chat.stream is False and chat.tools[0]["function"]["name"] == "Read"
    with pytest.raises(RequestError):
        parse_count_tokens_request("not an object")


def test_gate_capture_shape_converts_cleanly():
    """The structure gate 1 captured (comment 5440572099 / dump inspection):
    billing header first in a 3-block system list, cache_control on some
    blocks, a two-block user turn, then a plain-string inline system message."""
    body = _body(
        system=[
            {
                "type": "text",
                "text": (
                    "x-anthropic-billing-header: cc_version=2.1.238; cc_entrypoint=cli; "
                    "cc_is_subagent=true"
                ),
            },
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "Long system prompt.", "cache_control": {"type": "ephemeral"}},
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Agent prompt."},
                    {
                        "type": "text",
                        "text": "Environment.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "system", "content": "Inline subagent system text."},
        ],
        tools=[READ_TOOL],
        thinking={"type": "adaptive", "display": "omitted"},
    )
    chat = parse_messages_request(body)
    assert chat.messages == [
        {
            "role": "system",
            "content": (
                "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
                "Long system prompt.\n\nInline subagent system text."
            ),
        },
        {"role": "user", "content": "Agent prompt.\nEnvironment."},
    ]
