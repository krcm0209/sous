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
