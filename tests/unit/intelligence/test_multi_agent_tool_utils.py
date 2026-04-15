from types import SimpleNamespace

import pytest

from app.modules.intelligence.agents.chat_agents.multi_agent.utils import tool_utils


pytestmark = pytest.mark.unit


class _ToolCallPart:
    def __init__(self, tool_name: str, args: str, tool_call_id: str = "call-1"):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id

    def args_as_dict(self):
        raise ValueError("truncated json")


def test_repair_truncated_tool_args_json_closes_brackets_before_braces():
    repaired = tool_utils._repair_truncated_tool_args_json('{"paths":["a')

    assert repaired == {"paths": ["a"]}


def test_safe_parse_tool_args_repairs_and_sanitizes_event_part():
    event = SimpleNamespace(part=_ToolCallPart("search_files", '{"paths":["a'))

    parsed = tool_utils._safe_parse_tool_args(event, "search_files")

    assert parsed == {"paths": ["a"]}
    assert event.part.args == '{"paths": ["a"]}'


def test_create_tool_result_response_uses_full_regular_result_for_summary(monkeypatch):
    captured = {}
    full_raw = "x" * (tool_utils._MAX_TOOL_RESULT_STREAM_CHARS + 25)

    def fake_get_tool_response_message(tool_name, result):
        captured["response_result"] = result
        return "rendered response"

    def fake_get_tool_result_info_content(tool_name, result):
        captured["summary_result"] = result
        return "rendered summary"

    monkeypatch.setattr(
        tool_utils, "get_tool_response_message", fake_get_tool_response_message
    )
    monkeypatch.setattr(
        tool_utils, "get_tool_result_info_content", fake_get_tool_result_info_content
    )

    event = SimpleNamespace(
        result=SimpleNamespace(
            tool_name="search_files", content=full_raw, tool_call_id="call-2"
        )
    )

    response = tool_utils.create_tool_result_response(event)

    assert captured["response_result"] == full_raw
    assert captured["summary_result"] == full_raw
    assert response.tool_response == "rendered response"
    assert response.tool_call_details["summary"] == "rendered summary"
    assert response.tool_call_details["content"].startswith("x" * 50)
    assert "[truncated" in response.tool_call_details["content"]
    assert response.is_truncated is True
    assert response.original_length == len(full_raw)


def test_create_tool_result_response_uses_full_delegation_result_for_summary(
    monkeypatch,
):
    captured = {}
    full_raw = "delegated result " * 900

    monkeypatch.setattr(tool_utils, "is_delegation_tool", lambda tool_name: True)
    monkeypatch.setattr(
        tool_utils,
        "extract_agent_type_from_delegation_tool",
        lambda tool_name: "codebase",
    )
    monkeypatch.setattr(
        tool_utils, "get_delegation_response_message", lambda agent_type: "done"
    )

    def fake_get_delegation_result_content(agent_type, result):
        captured["delegation_result"] = result
        return "delegation summary"

    monkeypatch.setattr(
        tool_utils,
        "get_delegation_result_content",
        fake_get_delegation_result_content,
    )

    event = SimpleNamespace(
        result=SimpleNamespace(
            tool_name="delegate_to_codebase_agent",
            content=full_raw,
            tool_call_id="call-3",
        )
    )

    response = tool_utils.create_tool_result_response(event)

    assert captured["delegation_result"] == full_raw
    assert response.tool_response == "done"
    assert response.tool_call_details["summary"] == "delegation summary"
    assert response.tool_call_details["content"]

