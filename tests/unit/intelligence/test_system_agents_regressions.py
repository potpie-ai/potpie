from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.intelligence.agents.chat_agent import ChatContext


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def postgres_server_env(monkeypatch):
    monkeypatch.setenv(
        "POSTGRES_SERVER", "postgresql://test_user:test_pass@localhost:5432/test_db"
    )


@pytest.mark.asyncio
async def test_qna_agent_reuses_existing_file_structure_context():
    from app.modules.intelligence.agents.chat_agents.system_agents.qna_agent import (
        FILE_STRUCTURE_CONTEXT_MARKER,
        FILE_STRUCTURE_HEADER,
        QnAAgent,
    )

    tools_provider = MagicMock()
    tools_provider.get_code_from_multiple_node_ids_tool.run_multiple = AsyncMock(
        return_value="code snippet"
    )
    tools_provider.file_structure_tool.fetch_repo_structure = AsyncMock(
        return_value="repo tree"
    )

    agent = QnAAgent(MagicMock(), tools_provider, MagicMock())
    ctx = ChatContext(
        project_id="project-1",
        project_name="Project",
        curr_agent_id="qna-agent",
        history=[],
        query="How does auth work?",
    )

    first = await agent._enriched_context(ctx)
    second = await agent._enriched_context(first)

    assert tools_provider.file_structure_tool.fetch_repo_structure.await_count == 1
    assert second.additional_context.count(FILE_STRUCTURE_CONTEXT_MARKER) == 1
    assert second.additional_context.count(FILE_STRUCTURE_HEADER.strip()) == 1


def test_specgen_agent_uses_registered_todo_tool_names():
    from app.modules.intelligence.agents.chat_agents.system_agents.specgen.spec_gen_agent import (
        SpecGenAgent,
    )

    tools_provider = MagicMock()
    tools_provider.get_tools.return_value = []
    llm_provider = MagicMock()
    prompt_provider = MagicMock()

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.specgen.spec_gen_agent.PydanticRagAgent",
        return_value=SimpleNamespace(),
    ):
        agent = SpecGenAgent(llm_provider, tools_provider, prompt_provider)
        agent._build_agent()

    requested_tools = tools_provider.get_tools.call_args.args[0]

    assert "add_todo" in requested_tools
    assert "read_todos" in requested_tools
    assert "write_todos" in requested_tools
    assert "remove_todo" in requested_tools
    assert "add_subtask" in requested_tools
    assert "set_dependency" in requested_tools
    assert "update_todo_status" in requested_tools
    assert "get_available_tasks" in requested_tools
    assert "create_todo" not in requested_tools
    assert "get_todo" not in requested_tools
    assert "list_todos" not in requested_tools
