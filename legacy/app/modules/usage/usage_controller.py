from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.usage.usage_schema import UsageResponse
from app.modules.usage.usage_service import UsageService


class UsageController:
    @staticmethod
    async def get_user_usage(
        session: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        user_id: str,
    ) -> UsageResponse:
        usage_data = await UsageService.get_usage_data(
            session, start_date=start_date, end_date=end_date, user_id=user_id
        )
        return usage_data
