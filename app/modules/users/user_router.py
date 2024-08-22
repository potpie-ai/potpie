from typing import List
from fastapi import Depends, Query
from app.modules.utils.APIRouter import APIRouter
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.users.user_controller import UserController
from app.modules.users.user_schema import UserConversationListResponse

router = APIRouter()

class UserAPI:
    @staticmethod
    @router.get("/users/{user_id}/conversations/", response_model=List[UserConversationListResponse])
    async def get_conversations_for_user(
        user_id: str,
        start: int = Query(0, ge=0),
        limit: int = Query(10, ge=1),
        db: Session = Depends(get_db)
    ):
        controller = UserController(db)
        return await controller.get_conversations_for_user(user_id, start, limit)
    


