from sqlalchemy import TIMESTAMP, Column, String, Integer, func, ForeignKey, event, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.core.base_model import Base
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType
import logging

class Usage(Base):
    __tablename__ = "usage"

    uid = Column(String(255), ForeignKey('users.uid'), primary_key=True)
    usage_data = Column(JSONB, nullable=True)  # For future usage
    conversation_count = Column(Integer, default=0, nullable=False)
    messages_count = Column(Integer, default=0, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship with User
    user = relationship("User", back_populates="usage")

    @staticmethod
    @event.listens_for(Conversation, 'after_insert')
    def increment_conversation_count(mapper, connection, target):
        logging.info(f"Conversation insert event triggered for user_id: {target.user_id}")
        connection.execute(
            text("""
                INSERT INTO usage (uid, conversation_count, messages_count, created_at, updated_at)
                VALUES (:user_id, 1, 0, NOW(), NOW())
                ON CONFLICT (uid) 
                DO UPDATE SET 
                    conversation_count = usage.conversation_count + 1,
                    updated_at = NOW()
            """),
            {"user_id": target.user_id}
        )

    @staticmethod
    @event.listens_for(Message, 'after_insert')
    def increment_messages_count(mapper, connection, target):
        logging.info(f"Message insert event triggered for conversation_id: {target.conversation_id}, type: {target.type}")
        if target.type == MessageType.HUMAN:
            connection.execute(
                text("""
                    INSERT INTO usage (uid, conversation_count, messages_count, created_at, updated_at)
                    SELECT c.user_id, 0, 1, NOW(), NOW()
                    FROM conversations c
                    WHERE c.id = :conversation_id
                    ON CONFLICT (uid) 
                    DO UPDATE SET 
                        messages_count = usage.messages_count + 1,
                        updated_at = NOW()
                """),
                {"conversation_id": target.conversation_id}
            )


