from sqlalchemy import Column, String, TIMESTAMP, func, ForeignKey, Enum as SQLAEnum, ARRAY
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.uid", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    status = Column(SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)
    project_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# Conversation relationships
Conversation.user = relationship("User", back_populates="conversations")
Conversation.messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")