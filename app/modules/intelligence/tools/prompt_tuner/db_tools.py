"""
Database lookup tools for the Prompt Tuner Agent.

Provides tools to fetch message traces and custom agent prompts
from Potpie's Postgres database.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# fetch_message_trace
# ---------------------------------------------------------------------------

class FetchMessageTraceInput(BaseModel):
    message_id: Optional[str] = Field(
        default=None,
        description="ID of a specific message to fetch. Provide either message_id or conversation_id.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID to fetch the most recent AI message with tool calls. Provide either message_id or conversation_id.",
    )


class FetchMessageTraceTool:
    name = "fetch_message_trace"
    description = """Fetch a message trace from Potpie's database.

    Given a message_id, returns that specific message's content, tool_calls, and thinking.
    Given a conversation_id, returns the most recent AI-generated message with tool calls.

    Use this when the user provides a Potpie message or conversation ID instead of a Langfuse trace.
    """
    args_schema = FetchMessageTraceInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = FetchMessageTraceInput(**kwargs)

        if not input_data.message_id and not input_data.conversation_id:
            return "Provide either message_id or conversation_id."

        if input_data.message_id:
            msg = (
                self.sql_db.query(Message)
                .filter(Message.id == input_data.message_id)
                .first()
            )
            if not msg:
                return f"Message not found: {input_data.message_id}"
            return _format_message(msg)

        # conversation_id: get most recent AI message with tool_calls
        msg = (
            self.sql_db.query(Message)
            .filter(
                Message.conversation_id == input_data.conversation_id,
                Message.type == MessageType.AI_GENERATED,
                Message.tool_calls.isnot(None),
            )
            .order_by(Message.created_at.desc())
            .first()
        )
        if not msg:
            return f"No AI message with tool calls found in conversation: {input_data.conversation_id}"

        # Also fetch the preceding human message for context
        human_msg = (
            self.sql_db.query(Message)
            .filter(
                Message.conversation_id == input_data.conversation_id,
                Message.type == MessageType.HUMAN,
                Message.created_at < msg.created_at,
            )
            .order_by(Message.created_at.desc())
            .first()
        )

        lines = []
        if human_msg:
            lines.append("## User Message")
            lines.append(f"```\n{human_msg.content}\n```")
            lines.append("")

        lines.append(_format_message(msg))
        return "\n".join(lines)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


def _format_message(msg: Message) -> str:
    lines = []
    lines.append(f"## Message: {msg.id}")
    lines.append(f"- Type: {msg.type}")
    lines.append(f"- Conversation: {msg.conversation_id}")
    lines.append(f"- Created: {msg.created_at}")

    lines.append(f"\n### Content\n```\n{msg.content[:3000]}\n```")

    if msg.tool_calls:
        tool_calls = msg.tool_calls if isinstance(msg.tool_calls, list) else []
        lines.append(f"\n### Tool Calls ({len(tool_calls)})")
        for i, tc in enumerate(tool_calls, 1):
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except (json.JSONDecodeError, TypeError):
                    lines.append(f"\n#### Tool Call {i}\n```\n{tc}\n```")
                    continue
            name = tc.get("tool_name", "unknown")
            event_type = tc.get("event_type", "?")
            details = tc.get("tool_call_details", {})
            response = tc.get("tool_response", "")

            lines.append(f"\n#### Tool Call {i}: {name} ({event_type})")
            if details:
                details_str = json.dumps(details, indent=2, default=str)[:1500]
                lines.append(f"- Details:\n```json\n{details_str}\n```")
            if response:
                lines.append(f"- Response: {response[:500]}")

    if msg.thinking:
        lines.append(f"\n### Thinking\n```\n{msg.thinking[:2000]}\n```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fetch_agent_prompt
# ---------------------------------------------------------------------------

class FetchAgentPromptInput(BaseModel):
    agent_id: str = Field(
        ..., description="The custom agent ID to fetch the prompt for."
    )


class FetchAgentPromptTool:
    name = "fetch_agent_prompt"
    description = """Fetch a custom agent's current prompt configuration from the database.

    Returns the agent's role, goal, backstory, system_prompt, and tasks.
    Use this to get the full prompt that needs to be tuned.
    """
    args_schema = FetchAgentPromptInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    def run(self, **kwargs: Any) -> str:
        input_data = FetchAgentPromptInput(**kwargs)

        agent = (
            self.sql_db.query(CustomAgent)
            .filter(
                CustomAgent.id == input_data.agent_id,
                (CustomAgent.user_id == self.user_id) | (CustomAgent.visibility == "public"),
            )
            .first()
        )
        if not agent:
            return f"Custom agent not found or not accessible: {input_data.agent_id}"

        lines = []
        lines.append(f"## Custom Agent: {agent.id}")
        lines.append(f"- Visibility: {agent.visibility}")
        lines.append(f"- Created: {agent.created_at}")
        lines.append(f"- Updated: {agent.updated_at}")

        if agent.role:
            lines.append(f"\n### Role\n```\n{agent.role}\n```")
        if agent.goal:
            lines.append(f"\n### Goal\n```\n{agent.goal}\n```")
        if agent.backstory:
            lines.append(f"\n### Backstory\n```\n{agent.backstory}\n```")
        if agent.system_prompt:
            lines.append(f"\n### System Prompt\n```\n{agent.system_prompt}\n```")
        if agent.tasks:
            tasks_str = json.dumps(agent.tasks, indent=2, default=str)
            lines.append(f"\n### Tasks\n```json\n{tasks_str}\n```")

        return "\n".join(lines)

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def fetch_message_trace_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = FetchMessageTraceTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=FetchMessageTraceTool.name,
        description=FetchMessageTraceTool.description,
        args_schema=FetchMessageTraceInput,
    )


def fetch_agent_prompt_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = FetchAgentPromptTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name=FetchAgentPromptTool.name,
        description=FetchAgentPromptTool.description,
        args_schema=FetchAgentPromptInput,
    )
