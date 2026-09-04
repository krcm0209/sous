import pytest

from sous.protocol import WORKER_TOOLS, ParseError, ToolCall, parse_tool_calls

EXPECTED_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "grep",
    "run_command",
    "finish",
}


def test_worker_tools_names_and_shape():
    names = {t["function"]["name"] for t in WORKER_TOOLS}
    assert names == EXPECTED_TOOLS
    for t in WORKER_TOOLS:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


def test_parse_single_call():
    text = 'On it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    [call] = parse_tool_calls(text)
    assert call == ToolCall("read_file", {"path": "a.py"})


def test_parse_multiple_calls():
    text = (
        '<tool_call>{"name": "glob", "arguments": {"pattern": "**/*.py"}}</tool_call>'
        '<tool_call>{"name": "list_dir", "arguments": {}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["glob", "list_dir"]


def test_no_calls_returns_empty():
    assert parse_tool_calls("I think we should refactor.") == []


def test_malformed_json_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": "glob", "arguments":</tool_call>')


def test_unknown_tool_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>')


def test_non_dict_arguments_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": "glob", "arguments": "x"}</tool_call>')


def test_missing_name_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')


def test_write_file_with_closing_tag_in_content():
    """write_file whose content contains the literal </tool_call> substring."""
    text = (
        '<tool_call>{"name": "write_file", "arguments": '
        '{"path": "a.txt", "content": "abc</tool_call>def"}}</tool_call>'
    )
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "abc</tool_call>def"


def test_two_calls_first_with_closing_tag_in_arguments():
    """Two adjacent calls where the first has </tool_call> in its arguments."""
    text = (
        '<tool_call>{"name": "edit_file", "arguments": '
        '{"path": "x.py", "old": "foo</tool_call>", "new": "bar"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "y.py"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "edit_file"
    assert calls[0].arguments["old"] == "foo</tool_call>"
    assert calls[1].name == "read_file"


def test_write_file_with_opening_tag_in_content():
    """write_file whose content contains the literal <tool_call> substring."""
    text = (
        '<tool_call>{"name": "write_file", "arguments": '
        '{"path": "a.txt", "content": "prefix <tool_call> suffix"}}</tool_call>'
    )
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "prefix <tool_call> suffix"


def test_call_with_both_opening_and_closing_tags_in_content():
    """Call whose content contains both <tool_call> and </tool_call> substrings."""
    text = (
        '<tool_call>{"name": "write_file", "arguments": '
        '{"path": "test.txt", "content": "start <tool_call>data</tool_call> end"}}</tool_call>'
    )
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "start <tool_call>data</tool_call> end"


def test_two_calls_first_with_opening_tag_in_arguments():
    """Two adjacent calls where the first has <tool_call> in its arguments."""
    text = (
        '<tool_call>{"name": "edit_file", "arguments": '
        '{"path": "x.py", "old": "foo <tool_call> bar", "new": "baz"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "y.py"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "edit_file"
    assert calls[0].arguments["old"] == "foo <tool_call> bar"
    assert calls[1].name == "read_file"


# --- XML-ish function/parameter format (Qwen3 chat_template.jinja) ---

# Verbatim text emitted by Qwen3.8-27B in a real validation run.
QWEN_XML_REAL = '''<tool_call>
<function=write_file>
<parameter=path>
/tmp/x/shapes.py
</parameter>
<parameter=content>
import math


def area_circle(r: float) -> float:
    """Return the area of a circle with radius r."""
    return math.pi * r * r
</parameter>
</function>
</tool_call>'''

QWEN_XML_REAL_CONTENT = (
    "import math\n"
    "\n"
    "\n"
    "def area_circle(r: float) -> float:\n"
    '    """Return the area of a circle with radius r."""\n'
    "    return math.pi * r * r"
)


def test_xml_real_model_fixture_parses():
    """The exact text the real model emitted must parse to one ToolCall,
    with the multi-line content preserved byte-for-byte (internal blank
    lines intact, only the template's single wrapping newlines removed)."""
    calls = parse_tool_calls(QWEN_XML_REAL)
    assert calls == [
        ToolCall("write_file", {"path": "/tmp/x/shapes.py", "content": QWEN_XML_REAL_CONTENT})
    ]


def test_xml_trailing_blank_line_in_content_preserved():
    """A trailing newline inside content must survive: only the template's
    one wrapping newline is stripped, never .strip()/.rstrip()."""
    text = (
        "<tool_call>\n"
        "<function=write_file>\n"
        "<parameter=path>\n"
        "a.txt\n"
        "</parameter>\n"
        "<parameter=content>\n"
        "line1\n"
        "line2\n"
        "\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    [call] = parse_tool_calls(text)
    assert call.arguments["content"] == "line1\nline2\n"


def test_xml_integer_parameter_coerced_to_int():
    text = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\na.py\n</parameter>\n"
        "<parameter=offset>\n5\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    [call] = parse_tool_calls(text)
    assert call.arguments["offset"] == 5
    assert isinstance(call.arguments["offset"], int)
    assert not isinstance(call.arguments["offset"], bool)


def test_xml_uncoercible_integer_raises():
    text = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\na.py\n</parameter>\n"
        "<parameter=offset>\nnot-a-number\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    with pytest.raises(ParseError):
        parse_tool_calls(text)


def test_xml_two_adjacent_calls_in_order():
    text = (
        "<tool_call>\n<function=glob>\n"
        "<parameter=pattern>\n**/*.py\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=list_dir>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["glob", "list_dir"]
    assert calls[0].arguments == {"pattern": "**/*.py"}
    assert calls[1].arguments == {}


def test_mixed_json_and_xml_calls_both_parse():
    text = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>\n'
        "<tool_call>\n<function=list_dir>\n</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls == [ToolCall("read_file", {"path": "a.py"}), ToolCall("list_dir", {})]


def test_xml_unknown_tool_raises():
    with pytest.raises(ParseError):
        parse_tool_calls("<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>")


def test_xml_prose_before_call_ignored():
    """The template permits reasoning text before the tool call."""
    text = "I will write the shapes module now.\n\n" + QWEN_XML_REAL
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == QWEN_XML_REAL_CONTENT


def test_xml_embedded_tool_call_tags_in_value_not_rescanned():
    """Cursor advance: tag text inside an argument value must not be
    re-scanned as a new call (same guarantee as the JSON path)."""
    text = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=content>\nstart <tool_call>data</tool_call> end\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text)
    assert call.arguments["content"] == "start <tool_call>data</tool_call> end"


def test_xml_unterminated_parameter_raises():
    text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n"
    with pytest.raises(ParseError):
        parse_tool_calls(text)


def test_tool_call_with_unrecognized_payload_raises():
    with pytest.raises(ParseError):
        parse_tool_calls("<tool_call>garbage</tool_call>")


# --- request-scoped tool sets (gateway) ----------------------------------------

from sous.protocol import WORKER_TOOLSET, ToolSet  # noqa: E402 — grouped with its tests

CLIENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": ["integer", "null"]},
                    "mode": {"enum": ["a", "b"]},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"todos": {"type": "array", "items": {"type": "object"}}},
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                    "meta": {"type": "object"},
                },
                "required": ["command"],
            },
        },
    },
]


def test_default_toolset_is_the_strict_worker_set():
    assert WORKER_TOOLSET.names == EXPECTED_TOOLS
    assert WORKER_TOOLSET.strict is True
    assert WORKER_TOOLSET.param_types["read_file"]["offset"] == "integer"
    assert WORKER_TOOLSET.param_types["list_dir"] == {"path": "string"}


def test_request_scoped_toolset_accepts_client_tool_names():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = '<tool_call>{"name": "Read", "arguments": {"file_path": "a.py"}}</tool_call>'
    assert parse_tool_calls(text, ts) == [ToolCall("Read", {"file_path": "a.py"})]


def test_strict_request_scoped_toolset_rejects_worker_names():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=True)
    with pytest.raises(ParseError, match="unknown tool"):
        parse_tool_calls('<tool_call>{"name": "read_file", "arguments": {}}</tool_call>', ts)


def test_non_strict_toolset_passes_unknown_names_through_untyped():
    """Claude Code answers a hallucinated tool with its own tool-not-found
    result, which the model can recover from; ending the turn here could not
    be undone. With no schema to coerce against, every value stays a string."""
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Imaginary>\n<parameter=count>\n5\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert parse_tool_calls(text, ts) == [ToolCall("Imaginary", {"count": "5"})]


def test_xml_number_union_and_untyped_parameters_coerce_from_the_client_schema():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Read>\n"
        "<parameter=file_path>\na.py\n</parameter>\n"
        "<parameter=offset>\n10\n</parameter>\n"
        "<parameter=limit>\n20\n</parameter>\n"
        "<parameter=mode>\na\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments == {"file_path": "a.py", "offset": 10.0, "limit": 20, "mode": "a"}
    assert isinstance(call.arguments["limit"], int)


def test_xml_array_and_object_parameters_parse_as_json():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=TodoWrite>\n"
        '<parameter=todos>\n[{"content": "x", "status": "pending"}]\n</parameter>\n'
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=Bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        '<parameter=meta>\n{"a": [1, 2]}\n</parameter>\n'
        "<parameter=run_in_background>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text, ts)
    assert calls[0].arguments == {"todos": [{"content": "x", "status": "pending"}]}
    assert calls[1].arguments == {"command": "ls", "meta": {"a": [1, 2]}, "run_in_background": True}


def test_xml_malformed_array_parameter_raises():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=TodoWrite>\n<parameter=todos>\nnot json\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="todos"):
        parse_tool_calls(text, ts)


def test_xml_wrong_json_shape_for_array_parameter_raises():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        '<tool_call>\n<function=TodoWrite>\n<parameter=todos>\n{"not": "a list"}\n'
        "</parameter>\n</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="array"):
        parse_tool_calls(text, ts)


def test_json_call_with_non_string_name_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": 5, "arguments": {}}</tool_call>')


# --- non-finite numbers (NaN/Infinity are not valid JSON) ----------------------


@pytest.mark.parametrize("raw", ["NaN", "inf", "-Infinity"])
def test_xml_number_parameter_rejects_non_finite_values(raw):
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Read>\n"
        "<parameter=file_path>\na.py\n</parameter>\n"
        f"<parameter=offset>\n{raw}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="finite"):
        parse_tool_calls(text, ts)


def test_xml_number_parameter_still_parses_a_finite_float():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Read>\n"
        "<parameter=file_path>\na.py\n</parameter>\n"
        "<parameter=offset>\n1.5\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments["offset"] == 1.5


@pytest.mark.parametrize(
    "call",
    [
        '<tool_call>{"name": "finish", "arguments": {"x": 1e999}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"x": [1, -1e999]}}</tool_call>',
        '<tool_call>{"name": "finish", "arguments": {"x": {"y": 1e999}}}</tool_call>',
    ],
)
def test_json_call_with_an_overflowing_literal_raises(call):
    # 1e999 is a plain numeric literal to the scanner — parse_constant never
    # sees it — yet float("1e999") is inf, as unserialisable as Infinity.
    with pytest.raises(ParseError):
        parse_tool_calls(call)


def test_json_call_with_nan_argument_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": "finish", "arguments": {"x": NaN}}</tool_call>')


def test_json_call_with_infinity_nested_in_array_argument_raises():
    with pytest.raises(ParseError):
        parse_tool_calls(
            '<tool_call>{"name": "finish", "arguments": {"x": [1, Infinity]}}</tool_call>'
        )


def test_xml_object_parameter_rejects_non_finite_nested_number():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        '<parameter=meta>\n{"a": NaN}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError):
        parse_tool_calls(text, ts)


def test_toolset_tolerates_tools_without_properties():
    ts = ToolSet.from_tools(
        [{"type": "function", "function": {"name": "Ping", "parameters": {"type": "object"}}}],
        strict=True,
    )
    assert ts.param_types == {"Ping": {}}
    assert parse_tool_calls("<tool_call>\n<function=Ping>\n</function>\n</tool_call>", ts) == [
        ToolCall("Ping", {})
    ]


# --- nullable schema types ------------------------------------------------------

from sous.protocol import _schema_type  # noqa: E402 — grouped with its tests

NULLABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Search",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": ["integer", "null"]},
                    "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                    "opts": {"type": ["null", "object"]},
                    "depth": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }
]


def test_schema_type_keeps_every_union_member_in_schema_order():
    assert _schema_type({"type": ["null", "integer"]}) == "null|integer"
    assert _schema_type({"type": ["integer", "null"]}) == "integer|null"
    assert _schema_type({"type": ["integer", "string"]}) == "integer|string"
    assert _schema_type({"type": ["string", "integer", "null"]}) == "string|integer|null"
    assert _schema_type({"type": "integer"}) == "integer"
    assert _schema_type({"type": ["null"]}) == "null"
    assert _schema_type({"enum": ["a"]}) == "string"


def test_xml_null_for_a_nullable_parameter_becomes_none():
    """Claude Code sends `["integer", "null"]` schemas (Read.limit is one).
    The template writes a null argument as the text `null`, and coercing that
    to an int used to fail the parse and demote the whole call to prose."""
    ts = ToolSet.from_tools(NULLABLE_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Search>\n"
        "<parameter=query>\nx\n</parameter>\n"
        "<parameter=limit>\nnull\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments == {"query": "x", "limit": None}


def test_a_nullable_parameter_still_coerces_a_real_value():
    ts = ToolSet.from_tools(NULLABLE_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Search>\n"
        "<parameter=query>\nx\n</parameter>\n"
        "<parameter=limit>\n20\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments["limit"] == 20 and isinstance(call.arguments["limit"], int)


def test_null_for_a_non_nullable_parameter_still_raises():
    """The worker's fail-loudly contract: a schema that does not allow null
    must not silently hand the executor a None."""
    ts = ToolSet.from_tools(NULLABLE_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Search>\n"
        "<parameter=query>\nx\n</parameter>\n"
        "<parameter=depth>\nnull\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="depth"):
        parse_tool_calls(text, ts)


def test_nullable_array_and_object_parameters_accept_null():
    ts = ToolSet.from_tools(NULLABLE_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Search>\n"
        "<parameter=query>\nx\n</parameter>\n"
        "<parameter=tags>\nnull\n</parameter>\n"
        "<parameter=opts>\nnull\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments == {"query": "x", "tags": None, "opts": None}


def test_json_call_passes_a_real_null_through_untouched():
    """Regression pin for the path this fix does not touch: JSON arguments are
    native, so a null has always arrived as None."""
    ts = ToolSet.from_tools(NULLABLE_TOOLS, strict=False)
    text = '<tool_call>{"name": "Search", "arguments": {"query": "x", "limit": null}}</tool_call>'
    assert parse_tool_calls(text, ts) == [ToolCall("Search", {"query": "x", "limit": None})]


# --- multi-member (non-nullable) unions ----------------------------------------

MULTI_MEMBER_UNION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "int_or_string": {"type": ["integer", "string"]},
                    "string_or_int": {"type": ["string", "integer"]},
                    "int_or_bool": {"type": ["integer", "boolean"]},
                },
                "required": [],
            },
        },
    }
]


def _grep_call(param: str, raw: str):
    ts = ToolSet.from_tools(MULTI_MEMBER_UNION_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Grep>\n"
        f"<parameter={param}>\n{raw}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    return call.arguments[param]


def test_xml_union_prefers_a_leading_non_string_member_that_accepts():
    """`["integer", "string"]` demoted a valid XML call to prose before this
    fix: only the first union member (`"integer"`) was ever tried, so a
    genuinely string-shaped argument like `abc` raised instead of falling
    through to `"string"`."""
    assert _grep_call("int_or_string", "abc") == "abc"
    assert _grep_call("int_or_string", "42") == 42


def test_xml_union_schema_order_wins_over_a_later_stricter_member():
    """`["string", "integer"]` tries `"string"` first, so even a numeric-
    looking value stays a string — schema order, not type specificity,
    decides."""
    value = _grep_call("string_or_int", "42")
    assert value == "42" and isinstance(value, str)


def test_xml_union_with_no_accepting_member_raises_naming_the_union():
    with pytest.raises(ParseError, match=r"one of integer\|boolean"):
        _grep_call("int_or_bool", "maybe")


# --- decoder escapes: every raw_decode/json.loads failure must be a ParseError --

from sous.protocol import _MAX_ARGUMENT_DEPTH  # noqa: E402 — grouped with its tests


def test_json_call_with_an_argument_past_the_depth_cap_raises_parse_error():
    """`RecursionError` is stack-size dependent (a payload that decodes on a
    big-stack machine would still blow up re-encoding it elsewhere), so the
    real contract is `_MAX_ARGUMENT_DEPTH`, checked on the whole decoded
    payload — one level for the payload dict itself, one for its "arguments"
    value, then the argument's own nesting. `_MAX_ARGUMENT_DEPTH + 1` levels
    of array nesting clears that +2 envelope with room to spare."""
    n = _MAX_ARGUMENT_DEPTH + 1
    nested = "[" * n + "]" * n
    call = '<tool_call>{"name": "finish", "arguments": {"x": ' + nested + "}}</tool_call>"
    with pytest.raises(ParseError):
        parse_tool_calls(call)


def test_json_call_with_an_argument_exactly_at_the_depth_cap_parses():
    """Positive control: the payload dict (depth 1) and its "arguments"
    dict (depth 2) already spend 2 of the 64 levels, so the argument value
    itself can nest `_MAX_ARGUMENT_DEPTH - 2` levels deep and still parse —
    one level more (test above) is the first depth that raises."""
    n = _MAX_ARGUMENT_DEPTH - 2
    nested = "[" * n + "]" * n
    call = '<tool_call>{"name": "finish", "arguments": {"x": ' + nested + "}}</tool_call>"
    [call_obj] = parse_tool_calls(call)
    assert call_obj.name == "finish"


def test_json_call_with_nested_dicts_past_the_depth_cap_raises_parse_error():
    """Dicts count toward the depth cap exactly like arrays do."""
    n = _MAX_ARGUMENT_DEPTH + 1
    nested = '{"a":' * n + "1" + "}" * n
    call = '<tool_call>{"name": "finish", "arguments": {"x": ' + nested + "}}</tool_call>"
    with pytest.raises(ParseError):
        parse_tool_calls(call)


def test_json_call_with_a_5000_digit_integer_argument_raises_parse_error():
    """A 5000-digit integer literal exceeds Python's int-from-string digit
    limit and raises ValueError out of `_DECODER.raw_decode`, not
    json.JSONDecodeError."""
    call = '<tool_call>{"name": "finish", "arguments": {"x": ' + "1" * 5000 + "}}</tool_call>"
    with pytest.raises(ParseError):
        parse_tool_calls(call)


def test_xml_object_parameter_past_the_depth_cap_raises_parse_error():
    """The XML `object`/`array` coercion path's `json.loads` call is checked
    against the same `_MAX_ARGUMENT_DEPTH` cap. Here `_check_depth` sees the
    decoded value directly (no payload/arguments envelope), so
    `_MAX_ARGUMENT_DEPTH + 1` levels of nesting is the first depth past it."""
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    n = _MAX_ARGUMENT_DEPTH + 1
    nested = '{"a":' * n + "1" + "}" * n
    text = (
        "<tool_call>\n<function=Bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        f"<parameter=meta>\n{nested}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError):
        parse_tool_calls(text, ts)


def test_xml_integer_parameter_with_5000_digits_still_raises_parse_error():
    """Regression pin: the XML integer branch already catches ValueError
    from `int()`'s digit-limit check and must keep doing so."""
    text = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\na.py\n</parameter>\n"
        f"<parameter=offset>\n{'1' * 5000}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    with pytest.raises(ParseError):
        parse_tool_calls(text)
