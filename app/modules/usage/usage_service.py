from sqlalchemy.orm import Session
from sqlalchemy import func
from app.modules.usage.usage_models import Usage
from app.modules.usage.usage_schema import UsageResponse
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType

class UsageService:
    @staticmethod
    def get_usage_by_user_id(db: Session, user_id: str) -> UsageResponse:
        usage = db.query(Usage).filter(Usage.uid == user_id).first()
        if usage:
            return UsageResponse(
                conversation_count=usage.conversation_count,
                messages_count=usage.messages_count
            )
        return None

    @staticmethod
    def create_usage(db: Session, user_id: str) -> UsageResponse:
        usage = db.query(Usage).filter(Usage.uid == user_id).first()
        if not usage:
            conversation_count = db.query(func.count(Conversation.id)).filter(Conversation.user_id == user_id).scalar()
            messages_count = db.query(func.count(Message.id)).filter(
                Message.conversation_id.in_(
                    db.query(Conversation.id).filter(Conversation.user_id == user_id)
                ),
                Message.type == MessageType.HUMAN
            ).scalar()

            usage = Usage(
                uid=user_id,
                conversation_count=conversation_count,
                messages_count=messages_count
            )
            db.add(usage)
            db.commit()
            db.refresh(usage)

        return UsageResponse(
            conversation_count=usage.conversation_count,
            messages_count=usage.messages_count
        ) 