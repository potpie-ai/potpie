import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.modules.intelligence.agents.chat_agent import ChatContext


os.environ.setdefault(
    "POSTGRES_SERVER", "postgresql://test_user:test_pass@localhost:5432/test_db"
)


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_qna_agent_reuses_existing_file_structure_context():
    from app.modules.intelligence.agents.chat_agents.system_agents.qna_agent import (
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
    assert second.additional_context.count("File Structure of the project:") == 1


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
    assert "get_available_tasks" in requested_tools
    assert "create_todo" not in requested_tools
    assert "get_todo" not in requested_tools
    assert "list_todos" not in requested_tools
