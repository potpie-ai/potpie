from pydantic import BaseModel


class LLMProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class SetLLMProviderRequest(BaseModel):
    provider: str


class GetLLMProviderResponse(BaseModel):
    preferred_llm: str
    model_type: str
