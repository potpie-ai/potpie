from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_controller import UsageController
from app.modules.usage.usage_schema import UsageResponse

router = APIRouter()


class UsageAPI:
    @staticmethod
    @router.get("/usage", response_model=UsageResponse)
    async def get_usage(
        start_date: datetime,
        end_date: datetime,
        user=Depends(AuthService.check_auth),
        async_db: AsyncSession = Depends(get_async_db),
    ):
        user_id = user["user_id"]
        return await UsageController.get_user_usage(
            async_db, start_date, end_date, user_id
        )
