from pydantic import BaseModel

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    provider: str
    status: str

    class Config:
        from_attributes = True 