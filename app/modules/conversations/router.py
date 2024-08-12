from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db


from app.modules.conversations.conversation.schema import ConversationInfoResponse, CreateConversation, ConversationResponse
from app.modules.conversations.message.schema import MessageResponse


router = APIRouter()

@router.post("/conversations/", response_model=ConversationResponse, tags=["Conversations"])
def create_conversation(conversation_data: CreateConversation, db: Session = Depends(get_db)):
    # Boilerplate for creating a new conversation
    pass


@router.get("/conversations/{conversation_id}/messages/", response_model=List[MessageResponse], tags=["Conversations"])
def get_conversation_history(conversation_id: str, page: int = 1, db: Session = Depends(get_db)):
    # Boilerplate for fetching conversation history with pagination
    pass


@router.delete("/conversations/{conversation_id}/", status_code=status.HTTP_204_NO_CONTENT, tags=["Conversations"])
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    # Boilerplate for deleting a conversation
    pass


@router.get("/conversations/{conversation_id}/regenerate/", response_model=MessageResponse, tags=["Conversations"])
def regenerate_response(conversation_id: str, db: Session = Depends(get_db)):
    # Boilerplate for regenerating the last message of the conversation
    pass


@router.get("/conversations/{conversation_id}/info/", response_model=ConversationInfoResponse, tags=["Conversations"])
def get_conversation_info(conversation_id: str, db: Session = Depends(get_db)):
    # Boilerplate for retrieving limited conversation info
    pass