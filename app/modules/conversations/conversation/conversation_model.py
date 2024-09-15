import enum

from sqlalchemy import ARRAY, TIMESTAMP, Column
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base
from app.modules.conversations.message.message_model import Message #noqa

class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True, index=True)
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255), nullable=False)
    status = Column(
        SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False
    )
    project_ids = Column(ARRAY(String), nullable=False)
    agent_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
