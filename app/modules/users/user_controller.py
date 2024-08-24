from typing import List

from sqlalchemy.orm import Session

from app.modules.users.user_schema import UserConversationListResponse
from app.modules.users.user_service import UserService


class UserController:
    def __init__(self, db: Session):
        self.service = UserService(db)

    async def get_conversations_for_user(
        self, user_id: str, start: int, limit: int
    ) -> List[UserConversationListResponse]:
        conversations = self.service.get_conversations_for_user(user_id, start, limit)

        response = [
            UserConversationListResponse(
                id=conversation.id,
                user_id=conversation.user_id,
                title=conversation.title,
                status=conversation.status,
                project_ids=conversation.project_ids,
                created_at=conversation.created_at.isoformat(),
                updated_at=conversation.updated_at.isoformat(),
            )
            for conversation in conversations
        ]

        return response
