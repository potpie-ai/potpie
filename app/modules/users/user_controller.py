from sqlalchemy.orm import Session

from app.modules.users.user_schema import (
    UserProfileResponse,
)
from app.modules.users.user_service import UserService


class UserController:
    def __init__(self, db: Session):
        self.service = UserService(db)
        self.sql_db = db

    async def get_user_profile_pic(self, uid: str) -> UserProfileResponse:
        return await self.service.get_user_profile_pic(uid)
