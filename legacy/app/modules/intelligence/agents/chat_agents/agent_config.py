from typing import List

from pydantic import BaseModel


class TaskConfig(BaseModel):
    """Configuration for an individual agent task."""

    description: str
    expected_output: str


class AgentConfig(BaseModel):
    """Top-level agent configuration used by Pydantic-based agents."""

    role: str
    goal: str
    backstory: str
    tasks: List[TaskConfig]
    max_iter: int = 15
