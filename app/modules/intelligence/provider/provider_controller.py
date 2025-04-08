from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .provider_schema import (
    GetProviderResponse,
    ProviderInfo,
    SetProviderRequest,
    AvailableModelsResponse,
)
from .provider_service import ProviderService


class ProviderController:
    def __init__(self, db: Session, user_id: str):
        self.service = ProviderService.create(db, user_id)
        self.user_id = user_id

    async def list_available_llms(self) -> List[ProviderInfo]:
        """List available LLM providers."""
        try:
            providers = await self.service.list_available_llms()
            return providers
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing LLM providers: {str(e)}"
            )

    async def list_available_models(self) -> AvailableModelsResponse:
        """List available models for both chat and inference."""
        try:
            models = await self.service.list_available_models()
            return models
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing available models: {str(e)}"
            )

    async def set_global_ai_provider(self, user_id: str, request: SetProviderRequest):
        """Update the global AI provider configuration."""
        try:
            result = await self.service.set_global_ai_provider(user_id, request)
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error setting global AI provider: {str(e)}",
            )

    async def get_global_ai_provider(self, user_id: str) -> GetProviderResponse:
        """Get the current global AI provider configuration."""
        try:
            return await self.service.get_global_ai_provider(user_id)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting global AI provider: {str(e)}",
            )
