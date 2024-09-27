import enum

from sqlalchemy import TIMESTAMP, CheckConstraint, Column
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class MessageStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"  # Possible Future extension


class MessageType(str, enum.Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"
    SYSTEM_GENERATED = "SYSTEM_GENERATED"


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(255), primary_key=True)
    conversation_id = Column(
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content = Column(Text, nullable=False)
    sender_id = Column(String(255), nullable=True)
    type = Column(SQLAEnum(MessageType), nullable=False)
    status = Column(
        SQLAEnum(MessageStatus), default=MessageStatus.ACTIVE, nullable=False
    )
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    citations = Column(Text, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        CheckConstraint(
            "(type = 'HUMAN' AND sender_id IS NOT NULL) OR "
            "(type IN ('AI_GENERATED', 'SYSTEM_GENERATED') AND sender_id IS NULL)",
            name="check_sender_id_for_type",
        ),
    )
