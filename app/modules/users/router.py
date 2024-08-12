from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db


from app.modules.conversations.conversation.schema import ConversationResponse


router = APIRouter()

@router.get("/users/{user_id}/conversations/", response_model=List[ConversationResponse], tags=["User"])
def get_conversations_for_user(user_id: str, db: Session = Depends(get_db)):
    # Boilerplate for fetching a user's conversation list, sorted by timestamp
    pass
