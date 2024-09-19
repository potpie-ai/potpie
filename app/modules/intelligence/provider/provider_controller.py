from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .provider_schema import ProviderInfo, SetProviderRequest
from .provider_service import ProviderService


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

    async def set_global_ai_provider(
        self, user_id: str, provider_request: SetProviderRequest
    ):
        try:
            response = await self.service.set_global_ai_provider(
                user_id, provider_request.provider
            )
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error setting AI provider: {str(e)}"
            )
