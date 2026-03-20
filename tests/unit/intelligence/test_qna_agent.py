from types import SimpleNamespace

import pytest

import app.modules.intelligence.agents.chat_agents.system_agents.qna_agent as qna_module
from app.modules.intelligence.agents.chat_agents.system_agents.qna_agent import QnAAgent


pytestmark = pytest.mark.unit


class DummyToolsProvider:
    def __init__(self):
        self.requested = None
        self.exclude_embedding_tools = None

    def get_tools(self, requested, exclude_embedding_tools=False):
        self.requested = requested
        self.exclude_embedding_tools = exclude_embedding_tools
        return []


class DummyRagAgent:
    def __init__(self, llm_provider, agent_config, tools):
        self.llm_provider = llm_provider
        self.agent_config = agent_config
        self.tools = tools


def _fake_llm_provider():
    return SimpleNamespace(
        supports_pydantic=lambda _: False,
        chat_config=SimpleNamespace(model="test-model", capabilities=[]),
    )


def test_qna_prompt_requires_rg_and_avoids_kg_tools():
    prompt = qna_module.qna_task_prompt

    assert "Use rg (ripgrep), never grep" in prompt
    assert "ask_knowledge_graph_queries" not in prompt
    assert "get_code_from_probable_node_name" not in prompt
    assert "get_node_neighbours_from_node_id" not in prompt
    assert "get_code_from_multiple_node_ids" not in prompt
    assert "get_nodes_from_tags" not in prompt


def test_qna_agent_tool_list_excludes_kg_and_includes_bash(monkeypatch):
    monkeypatch.setattr(qna_module, "PydanticRagAgent", DummyRagAgent)

    tools_provider = DummyToolsProvider()
    agent = QnAAgent(
        llm_provider=_fake_llm_provider(),
        tools_provider=tools_provider,
        prompt_provider=SimpleNamespace(),
    )

    built = agent._build_agent()

    assert isinstance(built, DummyRagAgent)
    assert "bash_command" in tools_provider.requested
    assert "get_code_file_structure" in tools_provider.requested
    assert "fetch_file" in tools_provider.requested
    assert "fetch_files_batch" in tools_provider.requested
    assert "analyze_code_structure" in tools_provider.requested

    assert "ask_knowledge_graph_queries" not in tools_provider.requested
    assert "get_code_from_probable_node_name" not in tools_provider.requested
    assert "get_node_neighbours_from_node_id" not in tools_provider.requested
    assert "get_code_from_multiple_node_ids" not in tools_provider.requested
    assert "get_nodes_from_tags" not in tools_provider.requested
