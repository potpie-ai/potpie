from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.users.user_controller import UserController
from app.modules.users.user_schema import UserProfileResponse
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


class UserAPI:


    @router.get("/user/{user_id}/public-profile", response_model=UserProfileResponse)
    async def fetch_user_profile_pic(
        user_id: str,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        controller = UserController(db)
        return await controller.get_user_profile_pic(user_id)
