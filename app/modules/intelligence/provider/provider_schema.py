from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class SetProviderRequest(BaseModel):
    provider: str
    low_reasoning_model: str = None  # optional, user specified model name
    high_reasoning_model: str = None  # optional, user specified model name


class GetProviderResponse(BaseModel):
    preferred_llm: str
    model_type: str  # e.g., "global", can be extended for agent-specific in future
    low_reasoning_model: str
    high_reasoning_model: str
