from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Define Enums
class PromptType(str, Enum):
    SYSTEM = "System"
    USER = "User"

class PromptVisibilityType(str, Enum):
    PUBLIC = "Public"
    PRIVATE = "Private"

class PromptStatusType(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

# Request Schema for Creating a Prompt
class PromptCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (System or User)")
    visibility: PromptVisibilityType = Field(..., description="Visibility of the prompt (Public or Private)")
    version: Optional[int] = Field(1, description="Version number of the prompt")
    status: Optional[PromptStatusType] = Field(PromptStatusType.ACTIVE, description="Status of the prompt (active or inactive)")
    created_by: Optional[str] = Field(None, description="ID of the user who created the prompt")

# Request Schema for Updating a Prompt
class PromptUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=1000, description="The text content of the prompt")
    type: Optional[PromptType] = Field(None, description="Type of the prompt (System or User)")
    visibility: Optional[PromptVisibilityType] = Field(None, description="Visibility of the prompt (Public or Private)")
    status: Optional[PromptStatusType] = Field(None, description="Status of the prompt (active or inactive)")
    version: Optional[int] = Field(None, description="Version number of the prompt")

# Response Schema for a Single Prompt
class PromptResponse(BaseModel):
    id: str = Field(..., description="Unique identifier of the prompt")
    text: str = Field(..., description="The text content of the prompt")
    type: PromptType = Field(..., description="Type of the prompt (System or User)")
    visibility: PromptVisibilityType = Field(..., description="Visibility of the prompt (Public or Private)")
    version: int = Field(..., description="Version number of the prompt")
    status: PromptStatusType = Field(..., description="Status of the prompt (active or inactive)")
    created_by: Optional[str] = Field(None, description="ID of the user who created the prompt")
    created_at: str = Field(..., description="Timestamp of when the prompt was created")
    updated_at: str = Field(..., description="Timestamp of when the prompt was last updated")

    class Config:
        orm_mode = True

# Response Schema for Listing Prompts
class PromptListResponse(BaseModel):
    prompts: List[PromptResponse] = Field(..., description="List of prompts")
    total: int = Field(..., description="Total number of prompts returned")
