from typing import List

from pydantic import BaseModel, validator


class PromptInfo(BaseModel):
    type: str
    text: str


class ToolInfo(BaseModel):
    name: str
    description: str


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    prompts: List[PromptInfo]
    tools: List[ToolInfo]


class AgentDetailsAsACrewAIAgent(BaseModel):
    id: str
    name: str
    description: str
    backstory: str
    goals: List[str]
    tools: List[ToolInfo]


class TaskCreate(BaseModel):
    description: str
    expected_output: str


class AgentCreate(BaseModel):
    role: str
    goal: str
    backstory: str
    tool_ids: List[str]
    tasks: List[TaskCreate]

    @validator('tasks')
    def validate_tasks(cls, tasks):
        if not 1 <= len(tasks) <= 5:
            raise ValueError('Number of tasks must be between 1 and 5')
        return tasks


class Agent(AgentCreate):
    user_id: str

    class Config:
        orm_mode = True
