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


class ChatContext(BaseModel):
    project_id: str
    curr_agent_id: str
    history: List[str]
    node_ids: Optional[List[str]] = None
    additional_context: str = ""
    query: str


class ChatAgent(ABC):
    """Interface for chat agents. Chat agents will be used in conversation APIs"""

    @abstractmethod
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run synchronously in a blocking manner, return entire response at once"""
        pass

    @abstractmethod
    def run_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, yield response piece by piece"""
        pass


class AgentWithInfo:
    def __init__(self, agent: ChatAgent, id: str, name: str, description: str):
        self.id = id
        self.name = name
        self.description = description
        self.agent = agent
