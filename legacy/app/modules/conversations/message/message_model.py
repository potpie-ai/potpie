import enum

from sqlalchemy import TIMESTAMP, Boolean, CheckConstraint, Column, JSON
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
    type = Column(
        SQLAEnum(
            MessageType, name="message_type_enum", create_type=False, native_enum=False
        ),
        nullable=False,
    )
    status = Column(
        SQLAEnum(
            MessageStatus,
            name="message_status_enum",
            create_type=False,
            native_enum=False,
        ),
        default=MessageStatus.ACTIVE,
        nullable=False,
    )
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    citations = Column(Text, nullable=True)
    has_attachments = Column(Boolean, default=False, nullable=False)
    tool_calls = Column(JSON, nullable=True)
    thinking = Column(Text, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    attachments = relationship("MessageAttachment", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(
            "(type = 'HUMAN' AND sender_id IS NOT NULL) OR "
            "(type IN ('AI_GENERATED', 'SYSTEM_GENERATED') AND sender_id IS NULL)",
            name="check_sender_id_for_type",
        ),
    )
