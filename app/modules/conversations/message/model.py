from sqlalchemy import Column, String, ForeignKey, TIMESTAMP, func, Text, Enum as SQLAEnum, CheckConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

# Define the MessageType Enum
class MessageType(str, enum.Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(255), primary_key=True)
    conversation_id = Column(String(255), ForeignKey("conversations.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    sender_id = Column(String(255), nullable=True)  # Allow sender_id to be nullable
    type = Column(SQLAEnum(MessageType), nullable=False)  # Type of message (AI_GENERATED or HUMAN)
    created_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), nullable=False)  # Use UTC timestamp

    # Relationship to the Conversation model
    conversation = relationship("Conversation", back_populates="messages")

    # Add a CHECK constraint to enforce the sender_id logic
    __table_args__ = (
        CheckConstraint(
            "(type = 'HUMAN' AND sender_id IS NOT NULL) OR (type = 'AI_GENERATED' AND sender_id IS NULL)",
            name="check_sender_id_for_type"
        ),
    )
