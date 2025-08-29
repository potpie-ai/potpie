from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
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
    # Multimodal support - images attached to the current message
    image_attachments: Optional[Dict[str, Dict[str, Union[str, int]]]] = (
        None  # attachment_id -> {base64, mime_type, file_size, etc}
    )
    # Context images from recent conversation history
    context_images: Optional[Dict[str, Dict[str, Union[str, int]]]] = None

    def has_images(self) -> bool:
        """Check if this context contains any images"""
        return bool(self.image_attachments) or bool(self.context_images)

    def get_all_images(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get all images (current message + context) combined with metadata"""
        all_images = {}
        if self.image_attachments:
            for img_id, img_data in self.image_attachments.items():
                img_data_with_context = img_data.copy()
                img_data_with_context["context_type"] = "current_message"
                img_data_with_context["relevance"] = "high"
                all_images[img_id] = img_data_with_context
        if self.context_images:
            for img_id, img_data in self.context_images.items():
                img_data_with_context = img_data.copy()
                img_data_with_context["context_type"] = "conversation_history"
                img_data_with_context["relevance"] = "medium"
                all_images[img_id] = img_data_with_context
        return all_images

    def get_current_images_only(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get only current message images without historical context"""
        return self.image_attachments if self.image_attachments else {}

    def get_context_images_only(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get only historical context images"""
        return self.context_images if self.context_images else {}


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
