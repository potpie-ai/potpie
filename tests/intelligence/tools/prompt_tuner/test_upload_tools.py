"""
Tests for the parse_uploaded_trace tool.
"""

import json

from app.modules.intelligence.tools.prompt_tuner.upload_tools import (
    ParseUploadedTraceTool,
)


def test_parse_json_tool_call_array():
    tool = ParseUploadedTraceTool(None, None)
    data = [
        {"tool_name": "search_colgrep", "arguments": {"query": "auth"}},
        {"tool_name": "fetch_file", "arguments": {"path": "auth.py"}},
    ]
    result = tool.run(content=json.dumps(data))
    assert "Tool Calls (2)" in result
    assert "search_colgrep" in result
    assert "fetch_file" in result


def test_parse_json_langfuse_export():
    tool = ParseUploadedTraceTool(None, None)
    data = {
        "input": "How does auth work?",
        "output": "Auth uses JWT tokens...",
        "observations": [
            {"name": "search_colgrep", "type": "SPAN", "input": "auth", "output": "results"},
        ],
    }
    result = tool.run(content=json.dumps(data))
    assert "Input" in result
    assert "Output" in result
    assert "search_colgrep" in result


def test_parse_plain_text():
    tool = ParseUploadedTraceTool(None, None)
    result = tool.run(content="User: How does auth work?\nAssistant: Auth uses JWT...")
    assert "plain text" in result
    assert "How does auth work?" in result
