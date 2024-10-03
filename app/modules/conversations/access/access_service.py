from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError 
from typing import List
from uuid6 import uuid7 
from app.modules.conversations.conversation.conversation_model import Conversation



class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(self, conversation_id: str, recipient_emails: List[str]) -> str:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found.")

        shareable_link = f"conversations/shared/{conversation_id}"

        # Check if chat has already been shared with any of the recipient emails
        existing_shared_chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if existing_shared_chat:
            # Initialize shared_with_emails if it is None
            if existing_shared_chat.shared_with_emails is None:
                existing_shared_chat.shared_with_emails = []  # Initialize as an empty list

            for email in recipient_emails:
                if email not in existing_shared_chat.shared_with_emails:
                    existing_shared_chat.shared_with_emails.append(email)

            self.db.commit()  # Commit the changes to the existing shared chat
            return shareable_link
        
        # If there's no existing shared chat, create a new one
        new_shared_chat = Conversation(
            id=str(uuid7()),  # Generate a new unique ID if necessary
            shared_with_emails=recipient_emails,  # Store the recipient emails directly
            # Include any other necessary fields for the new conversation
        )
        try:
            self.db.add(new_shared_chat)
            self.db.commit()
        except IntegrityError as e:
            self.db.rollback()
            raise ShareChatServiceError("Failed to save shared chat due to a database integrity error.") from e

        return shareable_link

