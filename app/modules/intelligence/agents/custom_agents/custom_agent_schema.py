from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskBase(BaseModel):
    description: str
    tools: List[str]
    expected_output: Any = Field(description="A JSON object")

    @field_validator("expected_output")
    def validate_json_object(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Expected a JSON object")
        return v


class TaskCreate(TaskBase):
    pass  # No additional fields for creation


class Task(TaskBase):
    id: int

    model_config = {"from_attributes": True}


class AgentBase(BaseModel):
    role: str
    goal: str
    backstory: str
    system_prompt: str


class AgentVisibility(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    SHARED = "shared"


class AgentCreate(AgentBase):
    tasks: List[TaskCreate]

    @field_validator("tasks")
    def validate_tasks(cls, tasks):
        if not tasks:
            raise ValueError("At least one task is required")
        if len(tasks) > 5:
            raise ValueError("Maximum of 5 tasks allowed")
        return tasks


class AgentUpdate(BaseModel):
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    system_prompt: Optional[str] = None
    tasks: Optional[List[TaskCreate]] = None
    visibility: Optional[AgentVisibility] = None


class Agent(AgentBase):
    id: str
    user_id: str
    tasks: List[Task]
    deployment_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    deployment_status: Optional[str]
    visibility: AgentVisibility = AgentVisibility.PRIVATE

    model_config = {"from_attributes": True}


class AgentShare(BaseModel):
    agent_id: str
    shared_with_user_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentSharingRequest(BaseModel):
    agent_id: str
    visibility: Optional[AgentVisibility] = None
    shared_with_email: Optional[str] = None

    # Use model_post_init instead of field validator to validate entire model
    def model_post_init(self, __context) -> None:
        # If neither visibility nor email is provided, that's an error
        if self.visibility is None and not self.shared_with_email:
            raise ValueError(
                "Must provide either visibility change or email to share with"
            )

        # Can't make public and share with specific user at the same time
        if self.visibility == AgentVisibility.PUBLIC and self.shared_with_email:
            raise ValueError(
                "Cannot both make agent public and share with specific user"
            )


class RevokeAgentAccessRequest(BaseModel):
    agent_id: str
    user_email: str


class AgentSharesResponse(BaseModel):
    agent_id: str
    shared_with: list[str]


class ListAgentsRequest(BaseModel):
    include_public: bool = True
    include_shared: bool = True


class NodeContext(BaseModel):
    node_id: str
    name: str


class NodeInfo(BaseModel):
    node_id: str
    name: Optional[str] = None


class QueryRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    query: str
    node_ids: Optional[List[NodeInfo]] = None
    project_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: Any
    conversation_id: str


class PromptBasedAgentRequest(BaseModel):
    prompt: str


class AgentPlan(BaseModel):
    role: str
    goal: str
    backstory: str
    system_prompt: str
    tasks: List[TaskCreate]
