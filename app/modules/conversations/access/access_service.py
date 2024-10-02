from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError 
from typing import List
from uuid6 import uuid7 
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message
from app.modules.conversations.access.access_model import SharedChat

class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(self, conversation_id: str, recipient_emails: List[str]) -> str:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found.")
        

        shareable_link = f"coversations/shared/{conversation_id}"

        # Check if chat has already been shared with any of the recipient emails
        existing_shared_chat = self.db.query(SharedChat).filter_by(conversation_id=conversation_id).first()
        if existing_shared_chat:
      
            for email in recipient_emails:
                if email not in existing_shared_chat.shared_with_emails:
                    existing_shared_chat.shared_with_emails.append(email)
            self.db.commit()  # Commit the changes to the existing shared chat
            return shareable_link
        
        new_shared_chat = SharedChat(
            id=str(uuid7()),  # Generate a unique ID for the shared chat
            conversation_id=conversation_id,
            shared_with_emails=[recipient_emails],  # Store the recipient email
            share_link=shareable_link,
        )
        try:
            self.db.add(new_shared_chat)
            self.db.commit()
        except IntegrityError as e:
            self.db.rollback()
            raise ShareChatServiceError("Failed to save shared chat due to a database integrity error.") from e


        return shareable_link

    async def retrieve_shared_chat(self, conversation_id: str, user_email: str) -> dict:
        shared_chat = self.db.query(SharedChat).filter_by(conversation_id=conversation_id).first()
        if not shared_chat:
            raise ShareChatServiceError("Chat not found or access denied.")
        
        if user_email not in shared_chat.shared_with_emails:
            raise ShareChatServiceError("Access denied.")

        messages = (
            self.db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at)
            .all()
        )

        return {
            "id": conversation_id,
            "messages": [
                {
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                }
                for message in messages
            ],
        }
