from typing import List

from pydantic import BaseModel, field_validator


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


class TaskCreate(BaseModel):
    description: str
    expected_output: str
    tools: List[str]


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
