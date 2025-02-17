from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService

from .provider_controller import ProviderController
from .provider_schema import ProviderInfo, SetProviderRequest

router = APIRouter()


class ProviderAPI:
    @staticmethod
    @router.get("/list-available-llms/", response_model=List[ProviderInfo])
    async def list_available_llms(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ProviderController(db, user_id)
        return await controller.list_available_llms()

    @staticmethod
    @router.post("/set-global-ai-provider/")
    async def set_global_ai_provider(
        provider_request: SetProviderRequest,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ProviderController(db, user_id)
        return await controller.set_global_ai_provider(
            user["user_id"], provider_request
        )

    @staticmethod
    @router.get("/get-global-ai-provider/")
    async def get_global_ai_provider(
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        user_id = user["user_id"]
        controller = ProviderController(db, user_id)
        return await controller.get_global_ai_provider(user_id)
