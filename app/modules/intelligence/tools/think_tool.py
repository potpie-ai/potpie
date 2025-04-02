import asyncio
from typing import Dict, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.intelligence.provider.provider_service import ProviderService


class ThinkToolInput(BaseModel):
    thought: str = Field(description="Your thoughts to process")


class ThinkTool:
    """Tool for thinking and processing thoughts."""

    name = "think"
    description = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests."""

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.provider_service = ProviderService(sql_db, user_id)

    async def arun(self, thought: str) -> Dict[str, Any]:
        """Process the thought using structured thinking asynchronously."""
        prompt = """
        ## Using the think tool

        Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
        - List the specific rules that apply to the current request
        - Check if all required information is collected
        - Verify that the planned action complies with all policies
        - Iterate over tool results for correctness

        Here are some examples of what to iterate over inside the think tool:
        <think_tool_example_1>
        User wants to cancel flight ABC123
        - Need to verify: user ID, reservation ID, reason
        - Check cancellation rules:
          * Is it within 24h of booking?
          * If not, check ticket class and insurance
        - Verify no segments flown or are in the past
        - Plan: collect missing info, verify rules, get confirmation
        </think_tool_example_1>

        <think_tool_example_2>
        User wants to book 3 tickets to NYC with 2 checked bags each
        - Need user ID to check:
          * Membership tier for baggage allowance
          * Which payments methods exist in profile
        - Baggage calculation:
          * Economy class × 3 passengers
          * If regular member: 1 free bag each → 3 extra bags = $150
          * If silver member: 2 free bags each → 0 extra bags = $0
          * If gold member: 3 free bags each → 0 extra bags = $0
        - Payment rules to verify:
          * Max 1 travel certificate, 1 credit card, 3 gift cards
          * All payment methods must be in profile
          * Travel certificate remainder goes to waste
        - Plan:
        1. Get user ID
        2. Verify membership level for bag fees
        3. Check which payment methods in profile and if their combination is allowed
        4. Calculate total: ticket price + any bag fees
        5. Get explicit confirmation for booking
        </think_tool_example_2>

        Given the following thought, process it using structured thinking and return a detailed analysis:

        {thought}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert at structured thinking and analysis.",
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
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.arun(thought))
        finally:
            # Clean up if we created a new loop
            if not loop.is_running():
                loop.close()


def think_tool(sql_db: Session, user_id: str) -> StructuredTool:
    """Create and return the think tool."""
    tool_instance = ThinkTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="think",
        description=tool_instance.description,
        args_schema=ThinkToolInput,
    )
