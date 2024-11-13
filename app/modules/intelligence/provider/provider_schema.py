from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class SetProviderRequest(BaseModel):
    provider: str
