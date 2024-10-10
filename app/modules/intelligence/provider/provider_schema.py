from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class SetProviderRequest(BaseModel):
    provider: str


class GetProviderResponse(BaseModel):
    preferred_llm: str
    model_type: str
