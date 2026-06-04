"""Regression tests for local search tool exposure to agents."""

from pathlib import Path

import pytest

from app.modules.intelligence.tools.local_search_tools import tools as local_tools
from app.modules.intelligence.tools.local_search_tools.search_bash_tool import (
    SearchBashInput,
)
from app.modules.intelligence.tools.local_search_tools.search_text_tool import (
    SearchTextInput,
)

pytestmark = pytest.mark.unit


def test_create_local_search_tools_exposes_text_and_bash():
    tool_map = {tool.name: tool for tool in local_tools.create_local_search_tools()}

    assert "search_text" in tool_map
    assert "search_bash" in tool_map
    assert tool_map["search_text"].args_schema is SearchTextInput
    assert tool_map["search_bash"].args_schema is SearchBashInput


def test_search_text_structured_tool_invokes_model_wrapper(monkeypatch):
    captured: dict[str, SearchTextInput] = {}

    def fake_search_text(input_data: SearchTextInput) -> str:
        captured["input"] = input_data
        return "ok"

    monkeypatch.setattr(local_tools, "search_text_tool", fake_search_text)
    tool = next(t for t in local_tools.create_local_search_tools() if t.name == "search_text")

    assert tool.invoke({"query": "zipmapRawKeyLength", "use_bash": True}) == "ok"
    assert captured["input"].query == "zipmapRawKeyLength"
    assert captured["input"].use_bash is True


def test_search_bash_structured_tool_invokes_model_wrapper(monkeypatch):
    captured: dict[str, SearchBashInput] = {}

    def fake_search_bash(input_data: SearchBashInput) -> str:
        captured["input"] = input_data
        return "ok"

    monkeypatch.setattr(local_tools, "search_bash_tool", fake_search_bash)
    tool = next(t for t in local_tools.create_local_search_tools() if t.name == "search_bash")

    assert tool.invoke({"command": 'rg -n "zipmapEncodeLength" src tests'}) == "ok"
    assert captured["input"].command == 'rg -n "zipmapEncodeLength" src tests'


def test_tool_service_registers_local_search_tool_factory():
    source = (
        Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "tools"
        / "tool_service.py"
    ).read_text(encoding="utf-8")

    assert "create_local_search_tools" in source


def test_debug_agent_requests_search_bash_after_search_text():
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent import (
        DEBUG_AGENT_BASE_TOOLS,
        DEBUG_AGENT_DAP_TOOLS,
        DEBUG_AGENT_TERMINAL_TOOLS,
    )

    tool_list = (
        list(DEBUG_AGENT_BASE_TOOLS)
        + list(DEBUG_AGENT_DAP_TOOLS)
        + list(DEBUG_AGENT_TERMINAL_TOOLS)
    )
    assert "search_text" in tool_list
    assert "search_bash" in tool_list
    assert tool_list.index("search_text") < tool_list.index("search_bash")


def test_debug_agent_requests_terminal_tools_for_local_mode():
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent import (
        DEBUG_AGENT_TERMINAL_TOOLS,
    )

    assert "execute_terminal_command" in DEBUG_AGENT_TERMINAL_TOOLS
    assert "terminal_session_output" in DEBUG_AGENT_TERMINAL_TOOLS
    assert "terminal_session_signal" in DEBUG_AGENT_TERMINAL_TOOLS
