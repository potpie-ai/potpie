from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .provider_schema import GetProviderResponse, ProviderInfo, SetProviderRequest, DualProviderConfig, AvailableModelsResponse
from .provider_service import ProviderService, PLATFORM_PROVIDERS


class ProviderController:
    def __init__(self, db: Session, user_id: str):
        self.service = ProviderService.create(db, user_id)
        self.user_id = user_id

    async def list_available_llms(self) -> List[ProviderInfo]:
        try:
            providers = await self.service.list_available_llms()
            return providers
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing LLM providers: {str(e)}"
            )
            
    async def list_available_models(self) -> AvailableModelsResponse:
        try:
            models = await self.service.list_available_models()
            return models
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing available models: {str(e)}"
            )

    async def set_global_ai_provider(
        self, user_id: str, provider_request: SetProviderRequest
    ):
        provider = provider_request.provider.lower()
        low_reasoning_model = provider_request.low_reasoning_model
        high_reasoning_model = provider_request.high_reasoning_model
        config_type = provider_request.config_type
        selected_model = provider_request.selected_model

        # Validate the config_type
        if config_type not in ["chat", "inference"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid config_type: {config_type}. Must be 'chat' or 'inference'."
            )

        # if provider not in PLATFORM_PROVIDERS and provider not in [p.id for p in await self.list_available_llms()]: # check if provider is valid
        #     raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

        # add a supported provider list and check here
        if (
            provider not in PLATFORM_PROVIDERS
        ):  # for non-platform providers, model names are required
            if not low_reasoning_model or not high_reasoning_model:
                raise HTTPException(
                    status_code=400,
                    detail=f"For provider {provider}, both low_reasoning_model and high_reasoning_model must be specified.",
                )
        try:
            response = await self.service.set_global_ai_provider(
                user_id, provider, low_reasoning_model, high_reasoning_model, config_type, selected_model
            )
            return response
        except (
            ValueError
        ) as ve:  # Catch ValueError from service for API key not set error
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error setting AI provider: {str(e)}"
            )

    async def get_global_ai_provider(self, user_id: str, config_type: str = "chat") -> GetProviderResponse:
        try:
            provider_response = await self.service.get_global_ai_provider(user_id, config_type)
            return provider_response
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting AI provider: {str(e)}"
            )
            
    async def get_dual_provider_config(self, user_id: str) -> DualProviderConfig:
        try:
            # Get the configs, using defaults if not set
            chat_config = await self.service.get_global_ai_provider(user_id, "chat")
            inference_config = await self.service.get_global_ai_provider(user_id, "inference")
            
            # Make sure both configs have the correct config_type set
            if chat_config:
                chat_config.config_type = "chat"
            if inference_config:
                inference_config.config_type = "inference"
            
            return DualProviderConfig(
                chat_config=chat_config,
                inference_config=inference_config
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting dual provider configuration: {str(e)}"
            )
