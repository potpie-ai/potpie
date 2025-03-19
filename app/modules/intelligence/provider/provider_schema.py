from pydantic import BaseModel, validator, Field
from typing import Optional, List, Literal


class ProviderInfo(BaseModel):
    id: str
    name: str
    description: str


class SetProviderRequest(BaseModel):
    provider: str
    low_reasoning_model: Optional[str] = None  # optional, user specified model name
    high_reasoning_model: Optional[str] = None  # optional, user specified model name
    config_type: Literal["chat", "inference"] = "chat"  # must be either "chat" or "inference"
    selected_model: Optional[str] = Field(None, description="User-friendly model name from the mapping")
    
    @validator('selected_model')
    def validate_selected_model(cls, v, values):
        if 'provider' in values and values['provider']:
            # Only require selected_model in two cases:
            # 1. If config_type is explicitly set (meaning it's a config type change)
            # 2. If neither low_reasoning_model nor high_reasoning_model are provided (need selected_model for defaults)
            config_type_specified = 'config_type' in values and values['config_type'] != "chat"
            no_models_specified = (
                ('low_reasoning_model' not in values or values['low_reasoning_model'] is None) and
                ('high_reasoning_model' not in values or values['high_reasoning_model'] is None)
            )
            
            if (config_type_specified or no_models_specified) and not v:
                raise ValueError("selected_model is required when changing config type or when no models are specified")
        return v


class GetProviderResponse(BaseModel):
    preferred_llm: str
    model_type: str  # e.g., "global", can be extended for agent-specific in future
    low_reasoning_model: str
    high_reasoning_model: str
    config_type: Literal["chat", "inference"] = "chat"  # must be either "chat" or "inference"
    selected_model: Optional[str] = None  # user-friendly model name


class DualProviderConfig(BaseModel):
    chat_config: GetProviderResponse
    inference_config: GetProviderResponse


class AvailableModelOption(BaseModel):
    id: str
    name: str
    description: str
    provider: str


class AvailableModelsResponse(BaseModel):
    chat_models: List[AvailableModelOption]
    inference_models: List[AvailableModelOption]
