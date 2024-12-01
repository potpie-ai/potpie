from sqlalchemy.orm import Session
from app.modules.usage.usage_service import UsageService
from app.modules.usage.usage_schema import UsageResponse

class UsageController:
    def __init__(self, db: Session):
        self.db = db

    def get_usage_by_user_id(self, user_id: str) -> UsageResponse:
        return UsageService.get_usage_by_user_id(self.db, user_id)

    def create_usage(self, user_id: str) -> UsageResponse:
        return UsageService.create_usage(self.db, user_id)
