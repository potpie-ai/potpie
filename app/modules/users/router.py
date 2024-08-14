from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.conversations.conversation.schema import ConversationResponse

router = APIRouter()

class UserAPI:
    
    @staticmethod
    @router.get("/users/{user_id}/conversations/", response_model=List[ConversationResponse])
    async def get_conversations_for_user(
        user_id: str,
        start: int = Query(0, ge=0),  # Start index, default is 0
        limit: int = Query(10, ge=1),  # Number of items to return, default is 10
        db: Session = Depends(get_db)
    ):
        # Mocked data instead of actual logic
        conversations = [
            ConversationResponse(
                id=f"mock-conversation-id-{i}",
                user_id=user_id,
                title=f"Mock Conversation Title {i}",
                status="active",
                project_ids=["project1", "project2"],
                agent_ids=["agent1", "agent2"],
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-02T00:00:00Z",
            ) for i in range(start, start + limit)
        ]
        return conversations
