from unittest.mock import MagicMock

import pytest

from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.agents.chat_agent import (
    AgentWithInfo,
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
)
from app.modules.intelligence.agents.chat_agents.auto_router_agent import AutoRouterAgent

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Override global DB bootstrap for this isolated router integration test module."""
    yield


@pytest.fixture(scope="session", autouse=True)
def require_github_tokens():
    """Disable unrelated live GitHub gating for this module."""
    yield


class DummyAgent(ChatAgent):
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return ChatAgentResponse(response=f"handled:{ctx.curr_agent_id}", tool_calls=[], citations=[])

    async def run_stream(self, ctx: ChatContext):
        yield ChatAgentResponse(response="stream", tool_calls=[], citations=[])


async def test_impact_analysis_agent_is_registered():
    llm_provider = MagicMock()
    prompt_provider = MagicMock()
    tools_provider = MagicMock()

    service = AgentsService(None, llm_provider, prompt_provider, tools_provider)

    assert "impact_analysis_agent" in service.system_agents
    assert (
        await service.validate_agent_id("test-user", "impact_analysis_agent")
        == "SYSTEM_AGENT"
    )


async def test_auto_router_executes_current_impact_agent():
    impact_agent = DummyAgent()
    router = AutoRouterAgent(
        llm_provider=MagicMock(),
        agents={
            "impact_analysis_agent": AgentWithInfo(
                agent=impact_agent,
                id="impact_analysis_agent",
                name="Impact",
                description="Impact analysis",
            )
        },
    )

    ctx = ChatContext(
        project_id="project-id",
        project_name="owner/repo",
        curr_agent_id="impact_analysis_agent",
        history=[],
        query="I changed src/module/service.cs function ChangedFunction",
    )

    response = await router.run(ctx)
    assert response.response == "handled:impact_analysis_agent"
