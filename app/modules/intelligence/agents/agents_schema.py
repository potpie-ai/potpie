from pydantic import BaseModel


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str
