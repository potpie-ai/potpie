"""FW003: agent list authorization uses server-defined modes, not client flags."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.intelligence.agents.agent_list_scope import AgentListMode
from app.modules.intelligence.agents.agents_service import AgentInfo, AgentsService
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    AgentVisibility,
)


pytestmark = pytest.mark.unit


def _fake_custom_agent(
    agent_id: str,
    role: str = "Helper",
    goal: str = "Help",
    visibility: AgentVisibility = AgentVisibility.PRIVATE,
):
    return SimpleNamespace(
        id=agent_id,
        role=role,
        goal=goal,
        deployment_status="STOPPED",
        visibility=visibility,
    )


def _build_service() -> AgentsService:
    db = MagicMock()
    llm = MagicMock()
    prompts = MagicMock()
    tools = MagicMock()
    service = AgentsService(db, llm, prompts, tools)
    # Keep system catalog small and deterministic for assertions.
    service.system_agents = {
        "code_gen_agent": SimpleNamespace(
            name="Code Gen",
            description="Generates code",
        ),
        "qna_agent": SimpleNamespace(
            name="QnA",
            description="Answers questions",
        ),
    }
    return service


@pytest.mark.asyncio
async def test_owned_mode_excludes_system_and_shared_agents():
    service = _build_service()
    owned = [_fake_custom_agent("owned-1")]

    with patch(
        "app.modules.intelligence.agents.agents_service.CustomAgentService"
    ) as mock_cls:
        mock_cls.return_value.list_agents = AsyncMock(return_value=owned)
        result = await service.list_available_agents(
            {"user_id": "user-a"}, mode=AgentListMode.OWNED
        )

        mock_cls.return_value.list_agents.assert_awaited_once_with(
            "user-a", include_public=False, include_shared=False
        )

    assert [agent.id for agent in result] == ["owned-1"]
    assert all(agent.status != "SYSTEM" for agent in result)


@pytest.mark.asyncio
async def test_runtime_mode_includes_system_and_shared_scope():
    service = _build_service()
    customs = [
        _fake_custom_agent("owned-1"),
        _fake_custom_agent("shared-1", visibility=AgentVisibility.SHARED),
    ]

    with patch(
        "app.modules.intelligence.agents.agents_service.CustomAgentService"
    ) as mock_cls:
        mock_cls.return_value.list_agents = AsyncMock(return_value=customs)
        result = await service.list_available_agents(
            {"user_id": "user-a"}, mode=AgentListMode.RUNTIME
        )

        mock_cls.return_value.list_agents.assert_awaited_once_with(
            "user-a", include_public=False, include_shared=True
        )

    ids = [agent.id for agent in result]
    assert "code_gen_agent" in ids
    assert "qna_agent" in ids
    assert "owned-1" in ids
    assert "shared-1" in ids


@pytest.mark.asyncio
async def test_legacy_privilege_flags_cannot_escalate_owned_mode_via_router():
    """Tampering list_system_agents/include_* must not change owned-mode results."""
    from app.modules.auth.auth_service import AuthService
    from app.modules.intelligence.agents import agents_router

    app = FastAPI()
    app.include_router(agents_router.router, prefix="/api/v1")

    async def fake_auth():
        return {"user_id": "user-a", "email": "a@example.com"}

    app.dependency_overrides[AuthService.check_auth] = fake_auth

    owned_agents = [
        AgentInfo(
            id="owned-1",
            name="Helper",
            description="Help",
            status="STOPPED",
            visibility=AgentVisibility.PRIVATE,
        )
    ]

    with (
        patch(
            "app.modules.intelligence.agents.agents_router.ProviderService",
            return_value=MagicMock(),
        ),
        patch(
            "app.modules.intelligence.agents.agents_router.ToolService",
            return_value=MagicMock(),
        ),
        patch(
            "app.modules.intelligence.agents.agents_router.PromptService",
            return_value=MagicMock(),
        ),
        patch(
            "app.modules.intelligence.agents.agents_router.AgentsController"
        ) as mock_controller_cls,
        patch(
            "app.core.database.get_db",
            return_value=MagicMock(),
        ),
    ):
        mock_controller = MagicMock()
        mock_controller.list_available_agents = AsyncMock(return_value=owned_agents)
        mock_controller_cls.return_value = mock_controller

        client = TestClient(app)
        response = client.get(
            "/api/v1/list-available-agents/",
            params={
                "mode": "owned",
                "list_system_agents": "true",
                "include_public": "true",
                "include_shared": "true",
            },
        )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "owned-1",
            "name": "Helper",
            "description": "Help",
            "status": "STOPPED",
            "visibility": "private",
        }
    ]
    # Controller must be called with owned mode regardless of legacy flags.
    args, kwargs = mock_controller.list_available_agents.await_args
    assert args[1] is AgentListMode.OWNED or kwargs.get("mode") is AgentListMode.OWNED


@pytest.mark.asyncio
async def test_custom_agent_controller_owned_ignores_would_be_public_flags():
    from app.modules.intelligence.agents.custom_agents.custom_agent_controller import (
        CustomAgentController,
    )

    controller = CustomAgentController.__new__(CustomAgentController)
    controller.db = MagicMock()
    controller.service = MagicMock()
    controller.service.list_agents = AsyncMock(return_value=[])
    controller.user_service = MagicMock()

    await controller.list_agents("user-a", mode=AgentListMode.OWNED)
    controller.service.list_agents.assert_awaited_once_with("user-a", False, False)

    await controller.list_agents("user-a", mode=AgentListMode.RUNTIME)
    assert controller.service.list_agents.await_args_list[-1].args == (
        "user-a",
        False,
        True,
    )
