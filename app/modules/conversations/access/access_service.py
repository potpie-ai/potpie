from typing import List

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_model import Conversation


class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(
        self, conversation_id: str, recipient_emails: List[str]
    ) -> str:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found.")

        existing_emails = chat.shared_with_emails or []
        existing_emails_set = set(existing_emails)
        unique_new_emails_set = set(recipient_emails)

        if unique_new_emails_set.issubset(existing_emails_set):
            raise ShareChatServiceError("All provided emails have already been shared.")

        to_share = unique_new_emails_set - existing_emails_set
        if to_share:
            try:
                updated_emails = existing_emails + list(to_share)
                self.db.query(Conversation).filter_by(id=conversation_id).update(
                    {Conversation.shared_with_emails: updated_emails},
                    synchronize_session=False,
                )
                self.db.commit()
            except IntegrityError as e:
                self.db.rollback()
                raise ShareChatServiceError(
                    "Failed to update shared chat due to a database integrity error."
                ) from e

            return conversation_id
