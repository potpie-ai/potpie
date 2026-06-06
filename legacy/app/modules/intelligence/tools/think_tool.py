import asyncio
from typing import Dict, Any, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.intelligence.provider.provider_service import ProviderService


class ThinkToolInput(BaseModel):
    thought: str = Field(description="Your thoughts to process")


class ThinkTool:
    """Tool for thinking and processing thoughts."""

    name = "think"
    description = """Use this tool to pause, recollect your thoughts, and figure out the problem at hand.

When calling this tool, include in your thought:
- What you've learned so far from tool results and conversation
- The current problem or question you're trying to solve
- What information you have and what's missing
- What you're considering doing next

This helps you step back, synthesize information, and plan your next actions. The tool uses the same LLM as your parent agent."""

    def __init__(
        self,
        sql_db: Session,
        user_id: str,
        provider_service: Optional[ProviderService] = None,
    ):
        self.sql_db = sql_db
        self.user_id = user_id
        # Use provided ProviderService if available, otherwise create a new one
        # This ensures the think tool uses the same LLM configuration as the parent agent
        self.provider_service = provider_service or ProviderService(sql_db, user_id)

    async def arun(self, thought: str) -> Dict[str, Any]:
        """Process the thought using structured thinking asynchronously."""
        prompt = """You are helping an AI agent pause and reflect on its current situation. The agent has shared its thoughts, context, and what it's trying to figure out.

Your role is to:
1. **Synthesize** - Summarize what the agent knows and what the core problem is
2. **Identify gaps** - What information is missing or unclear?
3. **Suggest next steps** - What should the agent do next? Be specific and actionable

Keep your response concise and focused. Structure it as:
- **Current understanding:** Brief summary of what's known
- **Key questions:** What needs to be clarified or investigated?
- **Recommended next steps:** Specific actions to take

Here's what the agent shared:

{thought}"""

        messages = [
            {
                "role": "system",
                "content": "You help AI agents pause, reflect, and plan next steps. Provide concise, actionable guidance.",
            },
            {"role": "user", "content": prompt.format(thought=thought)},
        ]

        try:
            response = await self.provider_service.call_llm(
                messages=messages, config_type="chat"
            )
            return {"success": True, "analysis": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run(self, thought: str) -> Dict[str, Any]:
        """Synchronous wrapper for arun."""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we need to use a different approach
            import concurrent.futures

            def run_in_thread():
                # Create a new event loop in a separate thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.arun(thought))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        except RuntimeError:
            # No event loop running, we can create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.arun(thought))
            finally:
                loop.close()


def think_tool(
    sql_db: Session, user_id: str, provider_service: Optional[ProviderService] = None
) -> StructuredTool:
    """Create and return the think tool.

    Args:
        sql_db: Database session
        user_id: User ID for the tool
        provider_service: Optional ProviderService instance to share with parent agent.
                         If not provided, a new instance will be created.
    """
    tool_instance = ThinkTool(sql_db, user_id, provider_service)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="think",
        description=tool_instance.description,
        args_schema=ThinkToolInput,
    )
