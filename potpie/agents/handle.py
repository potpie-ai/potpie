"""AgentHandle - provides query/stream interface for a specific agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Optional

from potpie.agents.context import ChatContext, ChatAgentResponse
from potpie.exceptions import AgentExecutionError

if TYPE_CHECKING:
    from app.modules.intelligence.agents.agents_service import AgentsService


@dataclass
class AgentInfo:
    """Information about an available agent."""

    id: str
    name: str
    description: str


class AgentHandle:
    """Handle for a specific agent with query/stream methods.

    Provides a clean interface for interacting with a single agent.
    Supports per-query user context via ChatContext.user_id.

    Example:
        response = await handle.query(ctx)
        print(response.response)

        async for chunk in handle.stream(ctx):
            print(chunk.response, end="")
    """

    def __init__(
        self,
        agent_id: str,
        agents_service_factory: Callable[[Optional[str]], "AgentsService"],
    ):
        """Initialize agent handle.

        Args:
            agent_id: The agent identifier
            agents_service_factory: Factory function that returns AgentsService
                for a given user_id (or default if None)
        """
        self._agent_id = agent_id
        self._agents_service_factory = agents_service_factory

    @property
    def agent_id(self) -> str:
        """Get the agent ID for this handle."""
        return self._agent_id

    async def query(self, ctx: ChatContext) -> ChatAgentResponse:
        """Execute agent synchronously and return complete response.

        Args:
            ctx: ChatContext with project_id, query, history, etc.
                If ctx.user_id is set, tools will use that user's permissions.

        Returns:
            ChatAgentResponse with response text, citations, tool_calls

        Raises:
            AgentExecutionError: If agent execution fails

        Example:
            ctx = ChatContext(
                project_id="...",
                project_name="my-repo",
                curr_agent_id="codebase_qna_agent",
                query="How does authentication work?",
                history=[],
                user_id="user-123",  # Optional: specify user for permissions
            )
            response = await runtime.agents.codebase_qna_agent.query(ctx)
            print(response.response)
        """
        ctx.curr_agent_id = self._agent_id
        agents_service = self._agents_service_factory(ctx.user_id)
        try:
            return await agents_service.execute(ctx)
        except Exception as e:
            raise AgentExecutionError(
                f"Agent '{self._agent_id}' execution failed: {e}"
            ) from e

    async def stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        """Execute agent with streaming response.

        Yields partial responses as they are generated.

        Args:
            ctx: ChatContext with project_id, query, history, etc.
                If ctx.user_id is set, tools will use that user's permissions.

        Yields:
            ChatAgentResponse chunks as they are generated

        Raises:
            AgentExecutionError: If agent execution fails

        Example:
            async for chunk in runtime.agents.debugging_agent.stream(ctx):
                print(chunk.response, end="", flush=True)
        """
        ctx.curr_agent_id = self._agent_id
        agents_service = self._agents_service_factory(ctx.user_id)
        try:
            async for chunk in agents_service.execute_stream(ctx):
                yield chunk
        except Exception as e:
            raise AgentExecutionError(
                f"Agent '{self._agent_id}' streaming failed: {e}"
            ) from e

    def __repr__(self) -> str:
        return f"<AgentHandle agent_id='{self._agent_id}'>"
