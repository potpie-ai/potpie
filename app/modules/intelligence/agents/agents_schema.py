from pydantic import BaseModel
from typing import Optional
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import AgentVisibility


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str
    visibility: Optional[AgentVisibility] = None
