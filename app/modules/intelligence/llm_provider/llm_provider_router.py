from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.llm_provider.llm_provider_controller import (
    LLMProviderController,
)
from app.modules.intelligence.llm_provider.llm_provider_schema import (
    GetLLMProviderResponse,
    LLMProviderInfo,
    SetLLMProviderRequest,
)

router = APIRouter()


class ProviderAPI:
    @staticmethod
    @router.get("/list-available-llms/", response_model=List[LLMProviderInfo])
    async def list_available_llms(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = LLMProviderController(db, user_id)
        return await controller.list_available_llms()

    @staticmethod
    @router.post("/set-global-ai-provider/")
    async def set_global_ai_provider(
        provider_request: SetLLMProviderRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = LLMProviderController(db, user_id)
        return await controller.set_global_ai_provider(
            user["user_id"], provider_request
        )

    @staticmethod
    @router.get("/get-preferred-llm/", response_model=GetLLMProviderResponse)
    async def get_preferred_llm(
        user_id: str,
        db: Session = Depends(get_db),
        hmac_signature: str = Header(..., alias="X-HMAC-Signature"),
    ):
        if not AuthService.verify_hmac_signature(user_id, hmac_signature):
            raise HTTPException(status_code=401, detail="Unauthorized")
        controller = LLMProviderController(db, user_id)
        return await controller.get_preferred_llm(user_id)
