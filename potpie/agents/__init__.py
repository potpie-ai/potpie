"""Agents module for PotpieRuntime library.

Provides fluent access to Potpie's AI agents for code intelligence.

Example:
    async with PotpieRuntime(config) as runtime:
        ctx = ChatContext(
            project_id="...",
            project_name="my-repo",
            curr_agent_id="codebase_qna_agent",
            query="How does authentication work?",
            history=[],
        )
        response = await runtime.agents.codebase_qna_agent.query(ctx)
        print(response.response)
"""

from potpie.agents.runner import AgentRunner
from potpie.agents.handle import AgentHandle, AgentInfo
from potpie.agents.context import (
    ChatContext,
    ChatAgentResponse,
    ToolCallResponse,
    ToolCallEventType,
)

__all__ = [
    "AgentRunner",
    "AgentHandle",
    "AgentInfo",
    "ChatContext",
    "ChatAgentResponse",
    "ToolCallResponse",
    "ToolCallEventType",
]
