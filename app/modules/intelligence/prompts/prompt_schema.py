from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Define Enums
class PromptType(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"

class VisibilityType(str, Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"

class StatusType(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

# Request Schema for Creating a Prompt
class PromptCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (SYSTEM or USER)")
    visibility: VisibilityType = Field(..., description="Visibility of the prompt (PUBLIC or PRIVATE)")
    version: Optional[int] = Field(1, description="Version number of the prompt")
    status: Optional[StatusType] = Field(StatusType.ACTIVE, description="Status of the prompt (ACTIVE or INACTIVE)")

# Request Schema for Updating a Prompt
class PromptUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=1000, description="The text content of the prompt")
    type: Optional[PromptType] = Field(None, description="Type of the prompt (SYSTEM or USER)")
    visibility: Optional[VisibilityType] = Field(None, description="Visibility of the prompt (PUBLIC or PRIVATE)")
    status: Optional[StatusType] = Field(None, description="Status of the prompt (ACTIVE or INACTIVE)")

# Response Schema for a Single Prompt
class PromptResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the prompt")
    text: str = Field(..., description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (SYSTEM or USER)")
    visibility: VisibilityType = Field(..., description="Visibility of the prompt (PUBLIC or PRIVATE)")
    version: int = Field(..., description="Version number of the prompt")
    status: StatusType = Field(..., description="Status of the prompt (ACTIVE or INACTIVE)")
    created_at: str = Field(..., description="Timestamp of when the prompt was created")
    updated_at: str = Field(..., description="Timestamp of when the prompt was last updated")

    class Config:
        orm_mode = True

# Response Schema for Listing Prompts
class PromptListResponse(BaseModel):
    prompts: List[PromptResponse] = Field(..., description="List of prompts")
    total: int = Field(..., description="Total number of prompts returned")
