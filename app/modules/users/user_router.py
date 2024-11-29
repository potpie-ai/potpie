from typing import List

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.users.user_controller import UserController
from app.modules.users.user_schema import (
    UserConversationListResponse,
    UserProfileResponse,
)
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


class UserAPI:
    @staticmethod
    @router.get(
        "/user/conversations/",
        response_model=List[UserConversationListResponse],
    )
    async def get_conversations_for_user(
        user=Depends(AuthService.check_auth),
        start: int = Query(0, ge=0),
        limit: int = Query(10, ge=1),
        db: Session = Depends(get_db),
    ):
        user_id = user["user_id"]
        controller = UserController(db)
        return await controller.get_conversations_for_user(user_id, start, limit)

    @router.get("/user/{user_id}/public-profile", response_model=UserProfileResponse)
    async def fetch_user_profile_pic(
        user_id: str,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        controller = UserController(db)
        return await controller.get_user_profile_pic(user_id)
