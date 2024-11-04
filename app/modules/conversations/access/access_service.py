from typing import List

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.modules.conversations.conversation.conversation_model import (
    Conversation,
    Visibility,
)


class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(
        self,
        conversation_id: str,
        user_id: str,
        recipient_emails: List[str] = None,
        visibility: Visibility = None,
    ) -> str:
        chat = (
            self.db.query(Conversation)
            .filter_by(id=conversation_id, user_id=user_id)
            .first()
        )
        if not chat:
            raise HTTPException(
                404, "Chat does not exist or you are not authorized to access it."
            )

        # Default to PRIVATE if visibility is not specified
        visibility = visibility or Visibility.PRIVATE

        if visibility == Visibility.PUBLIC:
            chat.visibility = Visibility.PUBLIC
            self.db.commit()
            return conversation_id

        # Handle PRIVATE visibility (default case)
        chat.visibility = Visibility.PRIVATE
        if recipient_emails:
            existing_emails = chat.shared_with_emails or []
            existing_emails_set = set(existing_emails)
            unique_new_emails_set = set(recipient_emails)

            # if unique_new_emails_set.issubset(existing_emails_set):
            #     raise HTTPException(
            #         200, "All emails are already shared with this chat."
            #     )

            to_share = unique_new_emails_set - existing_emails_set
            if to_share:
                try:
                    updated_emails = existing_emails + list(to_share)
                    self.db.query(Conversation).filter_by(id=conversation_id).update(
                        {
                            Conversation.shared_with_emails: updated_emails,
                            Conversation.visibility: visibility,
                        },
                        synchronize_session=False,
                    )
                    self.db.commit()
                except IntegrityError as e:
                    self.db.rollback()
                    raise ShareChatServiceError(
                        "Failed to update shared chat due to a database integrity error."
                    ) from e
            self.db.commit()
            return conversation_id
        else:
            self.db.query(Conversation).filter_by(id=conversation_id).update(
                {Conversation.visibility: visibility}, synchronize_session=False
            )
            self.db.commit()

        return conversation_id

    async def get_shared_emails(self, conversation_id: str, user_id: str) -> List[str]:
        chat = (
            self.db.query(Conversation)
            .filter_by(id=conversation_id, user_id=user_id)
            .first()
        )
        if not chat:
            raise HTTPException(
                404, "Chat does not exist or you are not authorized to access it."
            )

        return chat.shared_with_emails or []

    async def remove_access(
        self, conversation_id: str, user_id: str, emails_to_remove: List[str]
    ) -> bool:
        """Remove access for specified emails from a conversation."""
        chat = (
            self.db.query(Conversation)
            .filter_by(id=conversation_id, user_id=user_id)
            .first()
        )
        if not chat:
            raise HTTPException(
                status_code=404,
                detail="Chat does not exist or you are not authorized to access it.",
            )

        if not chat.shared_with_emails:
            raise ShareChatServiceError("Chat has no shared access to remove.")

        existing_emails = set(chat.shared_with_emails)
        emails_to_remove_set = set(emails_to_remove)

        # Check if any of the emails to remove actually have access
        if not emails_to_remove_set.intersection(existing_emails):
            raise ShareChatServiceError(
                "None of the specified emails have access to this chat."
            )

        try:
            updated_emails = list(existing_emails - emails_to_remove_set)
            self.db.query(Conversation).filter_by(id=conversation_id).update(
                {Conversation.shared_with_emails: updated_emails},
                synchronize_session=False,
            )
            self.db.commit()
            return True
        except IntegrityError as e:
            self.db.rollback()
            raise ShareChatServiceError(
                "Failed to update shared chat due to a database integrity error."
            ) from e
