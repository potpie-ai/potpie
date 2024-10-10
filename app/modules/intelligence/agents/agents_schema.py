from typing import Any, List

from pydantic import BaseModel, Field, field_validator


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


class TaskCreate(BaseModel):
    description: str
    tools: List[str]
    expected_output: Any = Field(description="A JSON object")

    @field_validator("expected_output")
    @classmethod
    def validate_json_object(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Expected a JSON object")
        return v


class AgentCreate(BaseModel):
    role: str
    goal: str
    backstory: str
    tasks: List[TaskCreate]

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, tasks):
        if not 1 <= len(tasks) <= 5:
            raise ValueError("Number of tasks must be between 1 and 5")
        return tasks


class Agent(AgentCreate):
    user_id: str

    class Config:
        orm_mode = True
