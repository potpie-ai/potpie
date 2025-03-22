from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCallEventType(Enum):
    CALL = "call"
    RESULT = "result"


class ToolCallResponse(BaseModel):
    call_id: str = Field(
        ...,
        description="ID of the tool call",
    )
    event_type: ToolCallEventType = Field(..., description="Type of the event")
    tool_name: str = Field(
        ...,
        description="Name of the tool",
    )
    tool_response: str = Field(
        ...,
        description="Response from the tool",
    )
    tool_call_details: Dict[str, Any] = Field(
        ...,
        description="Details of the tool call",
    )


class ChatAgentResponse(BaseModel):
    response: str = Field(
        ...,
        description="Full response to the query",
    )
    tool_calls: List[ToolCallResponse] = Field([], description="List of tool calls")
    citations: List[str] = Field(
        ...,
        description="List of file names extracted from context and referenced in the response",
    )


class ChatContext(BaseModel):
    project_id: str
    project_name: str
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
