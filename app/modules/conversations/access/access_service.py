from sqlalchemy.orm import Session
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message
from app.modules.utils.posthog_helper import PostHogClient

class ShareChatServiceError(Exception):
    """Base exception class for ShareChatService errors."""


class ShareChatService:
    def __init__(self, db: Session):
        self.db = db

    async def share_chat(self, conversation_id: str, recipient_email: str) -> str:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found.")
        
        # You may add permission checks here
        # if not user_has_permission(chat, user_id):
        #     raise ShareChatServiceError("You do not have permission to share this chat.")

        # Generate shareable link (you might want to create a unique URL or token)
        shareable_link = f"coversations/shared/{conversation_id}"

        # Log sharing event
        PostHogClient().send_event(
            "share_chat_event",
            {"conversation_id": conversation_id, "recipient_email": recipient_email}
        )

        return shareable_link

    async def retrieve_shared_chat(self, conversation_id: str, user_email: str) -> dict:
        chat = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if not chat:
            raise ShareChatServiceError("Chat not found or access denied.")
        
        if chat.recipient_email != user_email:
            raise ShareChatServiceError("You do not have permission to access this chat.")

        messages = (
            self.db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at)
            .all()
        )

        return {
            "id": chat.id,
            "messages": [
                {
                    "sender": message.sender_id,
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                }
                for message in messages
            ],
        }
