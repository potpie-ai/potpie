from typing import List, Optional
from pydantic import BaseModel


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class AvailableModelOption(BaseModel):
    id: str  # Full model identifier (e.g., "anthropic/claude-3-5-haiku-20241022")
    name: str  # Display name
    description: str
    provider: str  # Provider identifier (e.g., "anthropic")
    is_chat_model: bool  # Whether this is a chat model
    is_inference_model: bool  # Whether this is an inference model


class AvailableModelsResponse(BaseModel):
    models: List[AvailableModelOption]


class SetProviderRequest(BaseModel):
    chat_model: Optional[str] = None  # Full model identifier for chat
    inference_model: Optional[str] = None  # Full model identifier for inference


class ModelInfo(BaseModel):
    provider: str
    id: str
    name: str


class GetProviderResponse(BaseModel):
    chat_model: Optional[ModelInfo] = None
    inference_model: Optional[ModelInfo] = None


class DualProviderConfig(BaseModel):
    chat_config: GetProviderResponse
    inference_config: GetProviderResponse
