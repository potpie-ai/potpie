from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .provider_schema import GetProviderResponse, ProviderInfo, SetProviderRequest
from .provider_service import ProviderService

import os
litellm_provider = os.getenv("LITELLM_PROVIDER")
class ProviderController:
    def __init__(self, db: Session, user_id: str):
        self.service = ProviderService.create(db, user_id)
        self.user_id = user_id

    async def list_available_llms(self) -> List[ProviderInfo]:
        try:
            providers = await self.service.list_available_llms()
            return litellm_provider
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing LLM providers: {str(e)}"
            )

    async def set_global_ai_provider(
        self, user_id: str, provider_request: SetProviderRequest
    ):
        try:
            response = await self.service.set_global_ai_provider(
                user_id, litellm_provider
            )
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error setting AI provider: {str(e)}"
            )

    async def get_global_ai_provider(self, user_id: str):
        try:
            provider = await self.service.get_global_ai_provider(user_id)
            return {"provider": litellm_provider}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting AI provider: {str(e)}"
            )

    async def get_preferred_llm(self, user_id: str) -> GetProviderResponse:
        try:
            preferred_llm, model_type = await self.service.get_preferred_llm(user_id)
            return GetProviderResponse(
                preferred_llm=litellm_provider, model_type=model_type
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error getting preferred LLM: {str(e)}"
            )
