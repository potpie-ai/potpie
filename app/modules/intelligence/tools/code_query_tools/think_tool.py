import asyncio
from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.provider.provider_schema import ThinkResponse


class ThinkRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 3


class ThinkTool:
    name = "think"
    description = """"Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
        This tool allows for step-by-step thinking through problems with multiple iterations.
        
        :param query: string, the question or problem to think about
        :param max_iterations: optional int, maximum number of thinking iterations (default: 3)

        example:
        {
            "query": "What are the implications of using microservices vs monolithic architecture?",
            "max_iterations": 3
        }

        Returns a ThinkResponse containing:
        - thoughts: list of all thinking steps
        - conclusion: final conclusion
        - confidence: confidence score (0-1)
        - reasoning_path: list of key reasoning steps
        """

    def __init__(self, db: Session):
        self.provider_service = ProviderService(db, "dummy")

    async def think_about(self, query: str, max_iterations: Optional[int] = 3) -> ThinkResponse:
        return await self.provider_service.think(query, max_iterations or 3)

    async def arun(self, query: str, max_iterations: Optional[int] = 3) -> dict:
        """Async execution of the think tool"""
        response = await self.think_about(query, max_iterations)
        return {
            "thoughts": response.thoughts,
            "conclusion": response.conclusion,
            "confidence": response.confidence,
            "reasoning_path": response.reasoning_path
        }

    def run(self, query: str, max_iterations: Optional[int] = 3) -> dict:
        """Synchronous execution of the think tool"""
        return asyncio.run(self.think_about(query, max_iterations)).__dict__


def get_think_tool(db: Session) -> StructuredTool:
    """Create and return a structured think tool instance"""
    return StructuredTool(
        name="think",
        description=""""Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."Use this tool when you need to think through a complex problem step by step.
        The tool will help break down the problem, analyze it from multiple angles, and provide
        a structured response with thoughts, conclusions, and reasoning path.
        
        Useful for:
        - Complex architectural decisions
        - Trade-off analysis
        - Problem decomposition
        - Multi-step reasoning
        
        Input should be a clear question or problem statement.""",
        coroutine=ThinkTool(db).arun,
        func=ThinkTool(db).run,
        args_schema=ThinkRequest,
    ) 