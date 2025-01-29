from datetime import datetime

from app.modules.usage.usage_schema import UsageResponse
from app.modules.usage.usage_service import UsageService


class UsageController:
    @staticmethod
    async def get_user_usage(
        start_date: datetime, end_date: datetime, user_id: str
    ) -> UsageResponse:
        usage_data = await UsageService.get_usage_data(start_date, end_date, user_id)
        return usage_data
