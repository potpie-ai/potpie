from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# Define Enums
class PromptStatusType(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class PromptType(str, Enum):
    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"


# Request Schema for Creating a Prompt
class PromptCreate(BaseModel):
    text: str = Field(..., min_length=1, description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (System or Human)")
    status: Optional[PromptStatusType] = Field(
        PromptStatusType.ACTIVE, description="Status of the prompt (active or inactive)"
    )
    # Remove the version field from here


# Request Schema for Updating a Prompt
class PromptUpdate(BaseModel):
    text: Optional[str] = Field(
        None,
        min_length=1,
        description="The text content of the prompt",
    )
    type: Optional[PromptType] = Field(
        None, description="Type of the prompt (System or Human)"
    )
    status: Optional[PromptStatusType] = Field(
        None, description="Status of the prompt (active or inactive)"
    )
    # Remove the version field from here


# Response Schema for a Single Prompt
class PromptResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the prompt")
    text: str = Field(..., description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (System or Human)")
    version: int = Field(..., description="Version number of the prompt")
    status: PromptStatusType = Field(
        ..., description="Status of the prompt (active or inactive)"
    )
    created_by: Optional[str] = Field(
        None, description="ID of the user who created the prompt"
    )
    created_at: datetime = Field(
        ..., description="Timestamp of when the prompt was created"
    )
    updated_at: datetime = Field(
        ..., description="Timestamp of when the prompt was last updated"
    )

    class Config:
        from_attributes = True


# Response Schema for Listing Prompts
class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    total: int


# Schema for Agent Prompt Mapping
class AgentPromptMappingCreate(BaseModel):
    agent_id: str = Field(..., description="ID of the agent")
    prompt_id: str = Field(..., description="ID of the prompt")
    prompt_stage: int = Field(..., description="Stage of the prompt (1, 2, 3, etc.)")


class AgentPromptMappingResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the mapping")
    agent_id: str = Field(..., description="ID of the agent")
    prompt_id: str = Field(..., description="ID of the prompt")
    prompt_stage: int = Field(..., description="Stage of the prompt")

    class Config:
        from_attributes = True


class RequestModel(BaseModel):
    conversation_id: str
    prompt: str


class EnhancedPromptResponse(BaseModel):
    enhancedprompt: str
