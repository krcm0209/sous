import pytest

from sous.protocol import WORKER_TOOLS, ParseError, ToolCall, parse_tool_calls

EXPECTED_TOOLS = {
    "read_file", "write_file", "edit_file", "list_dir",
    "glob", "grep", "run_command", "finish",
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
    text = '<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "abc</tool_call>def"}}</tool_call>'
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "abc</tool_call>def"


def test_two_calls_first_with_closing_tag_in_arguments():
    """Two adjacent calls where the first has </tool_call> in its arguments."""
    text = (
        '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old": "foo</tool_call>", "new": "bar"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "y.py"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "edit_file"
    assert calls[0].arguments["old"] == "foo</tool_call>"
    assert calls[1].name == "read_file"


def test_write_file_with_opening_tag_in_content():
    """write_file whose content contains the literal <tool_call> substring."""
    text = '<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "prefix <tool_call> suffix"}}</tool_call>'
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "prefix <tool_call> suffix"


def test_call_with_both_opening_and_closing_tags_in_content():
    """Call whose content contains both <tool_call> and </tool_call> substrings."""
    text = '<tool_call>{"name": "write_file", "arguments": {"path": "test.txt", "content": "start <tool_call>data</tool_call> end"}}</tool_call>'
    [call] = parse_tool_calls(text)
    assert call.name == "write_file"
    assert call.arguments["content"] == "start <tool_call>data</tool_call> end"


def test_two_calls_first_with_opening_tag_in_arguments():
    """Two adjacent calls where the first has <tool_call> in its arguments."""
    text = (
        '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old": "foo <tool_call> bar", "new": "baz"}}</tool_call>'
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
        ToolCall("write_file", {"path": "/tmp/x/shapes.py",
                                "content": QWEN_XML_REAL_CONTENT})
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
    assert calls == [ToolCall("read_file", {"path": "a.py"}),
                     ToolCall("list_dir", {})]


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
