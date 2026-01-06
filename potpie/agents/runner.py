"""AgentRunner - provides fluent access to agents via attribute access."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from potpie.agents.handle import AgentHandle, AgentInfo
from potpie.exceptions import AgentNotFoundError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.modules.intelligence.provider.provider_service import ProviderService
    from app.modules.intelligence.tools.tool_service import ToolService
    from app.modules.intelligence.prompts.prompt_service import PromptService
    from app.modules.intelligence.agents.agents_service import AgentsService

logger = logging.getLogger(__name__)


class AgentRunner:
    """Provides fluent access to agents via attribute access.

    This class enables the `runtime.agents.agent_name` pattern,
    dynamically creating AgentHandle instances for valid agents.

    The runner supports per-query user context: when ChatContext.user_id
    is provided, a fresh AgentsService is created with that user_id to
    ensure tools have correct permissions.

    Example:
        response = await runtime.agents.codebase_qna_agent.query(ctx)
        async for chunk in runtime.agents.debugging_agent.stream(ctx):
            print(chunk.response, end="")
    """

    def __init__(
        self,
        db_session: Session,
        user_id: str,
        provider_service: ProviderService,
        tool_service: ToolService,
        prompt_service: PromptService,
        provider_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the AgentRunner.

        Args:
            db_session: Database session
            user_id: Default user identifier for agent operations
            provider_service: LLM provider service
            tool_service: Tool service for agent tools
            prompt_service: Prompt service for agent prompts
            provider_config: Optional provider configuration for creating
                per-user provider services
        """
        self._db = db_session
        self._user_id = user_id
        self._provider_service = provider_service
        self._tool_service = tool_service
        self._prompt_service = prompt_service
        self._provider_config = provider_config or {}

        self._agents_service: Optional[AgentsService] = None
        self._agents_service_cache: Dict[str, AgentsService] = {}
        self._handles: Dict[str, AgentHandle] = {}

    def _get_agents_service(self, user_id: Optional[str] = None) -> AgentsService:
        """Get or create AgentsService for a specific user.

        Args:
            user_id: User ID to create service for. If None, uses default.

        Returns:
            AgentsService configured for the specified user
        """
        effective_user_id = user_id or self._user_id

        if effective_user_id in self._agents_service_cache:
            return self._agents_service_cache[effective_user_id]

        from app.modules.intelligence.agents.agents_service import AgentsService
        from app.modules.intelligence.tools.tool_service import ToolService
        from app.modules.intelligence.provider.provider_service import ProviderService

        if effective_user_id == self._user_id:
            tool_service = self._tool_service
            provider_service = self._provider_service
        else:
            tool_service = ToolService(self._db, effective_user_id)
            if self._provider_config:
                provider_service = ProviderService.create_from_config(
                    self._db,
                    effective_user_id,
                    **self._provider_config,
                )
            else:
                provider_service = ProviderService.create(self._db, effective_user_id)

        agents_service = AgentsService(
            self._db,
            provider_service,
            self._prompt_service,
            tool_service,
        )
        self._agents_service_cache[effective_user_id] = agents_service
        return agents_service

    def _is_valid_agent(self, agent_id: str) -> bool:
        """Check if agent_id is a valid system agent.

        Args:
            agent_id: The agent identifier to validate

        Returns:
            True if agent exists, False otherwise
        """
        agents_service = self._get_agents_service()
        return agent_id in agents_service.system_agents

    def __getattr__(self, agent_id: str) -> AgentHandle:
        """Dynamic attribute access for agents.

        Allows: runtime.agents.codebase_qna_agent
        Instead of: runtime.agents.get("codebase_qna_agent")

        Args:
            agent_id: The agent identifier (attribute name)

        Returns:
            AgentHandle for the requested agent

        Raises:
            AgentNotFoundError: If agent doesn't exist
            AttributeError: For private attributes (starting with _)
        """
        if agent_id.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{agent_id}'"
            )

        if agent_id not in self._handles:
            if not self._is_valid_agent(agent_id):
                available = ", ".join(self.list_agent_ids())
                raise AgentNotFoundError(
                    f"Unknown agent: '{agent_id}'. Available agents: {available}"
                )

            self._handles[agent_id] = AgentHandle(agent_id, self._get_agents_service)

        return self._handles[agent_id]

    def get(self, agent_id: str) -> AgentHandle:
        """Get an agent handle by ID (alternative to attribute access).

        Args:
            agent_id: The agent identifier

        Returns:
            AgentHandle for the requested agent

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        return self.__getattr__(agent_id)

    def list_agents(self) -> List[AgentInfo]:
        """List all available system agents.

        Returns:
            List of AgentInfo with id, name, description
        """
        agents_service = self._get_agents_service()
        return [
            AgentInfo(
                id=agent_id,
                name=agent.name,
                description=agent.description,
            )
            for agent_id, agent in agents_service.system_agents.items()
        ]

    def list_agent_ids(self) -> List[str]:
        """List all available agent IDs.

        Returns:
            List of agent ID strings
        """
        agents_service = self._get_agents_service()
        return list(agents_service.system_agents.keys())

    def __repr__(self) -> str:
        agent_count = len(self._get_agents_service().system_agents)
        return f"<AgentRunner agents={agent_count}>"

    def __dir__(self) -> List[str]:
        """Support tab-completion in interactive environments."""
        base = list(super().__dir__())
        try:
            base.extend(self.list_agent_ids())
        except Exception:
            pass
        return base
