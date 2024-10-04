from typing import List, Dict
from pydantic import BaseModel


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