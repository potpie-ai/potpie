from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
from pydantic import BaseModel, Field


class ChatAgentResponse(BaseModel):
    response: str = Field(
        ...,
        description="Full response to the query",
    )
    citations: List[str] = Field(
        ...,
        description="List of file names extracted from context and referenced in the response",
    )


class ChatAgent(ABC):
    """Interface for chat agents. Chat agents will be used in conversation APIs"""

    @abstractmethod
    async def run(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> ChatAgentResponse:
        """Run synchronously in a blocking manner, return entire response at once"""
        pass

    @abstractmethod
    async def run_stream(
        self,
        query: str,
        history: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, yield response piece by piece"""
        pass
