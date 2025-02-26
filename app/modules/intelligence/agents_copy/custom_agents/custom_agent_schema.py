from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator


class TaskBase(BaseModel):
    description: str
    tools: List[str]
    expected_output: Any = Field(description="A JSON object")

    @validator("expected_output")
    def validate_json_object(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Expected a JSON object")
        return v


class TaskCreate(TaskBase):
    pass  # No additional fields for creation


class Task(TaskBase):
    id: int

    class Config:
        from_attributes = True


class AgentBase(BaseModel):
    role: str
    goal: str
    backstory: str
    system_prompt: str


class AgentCreate(AgentBase):
    tasks: List[TaskCreate]

    @validator("tasks")
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


class Agent(AgentBase):
    id: str
    user_id: str
    tasks: List[Task]
    deployment_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    deployment_status: Optional[str]

    class Config:
        from_attributes = True


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
