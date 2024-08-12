from sqlalchemy import Column, String, TIMESTAMP, func, ForeignKey, Enum as SQLAEnum, Index
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum

# Define the Status Enum
class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.uid"), nullable=False, index=True)  # ForeignKey to User model with index
    title = Column(String(255), nullable=False)  # Title of the conversation
    status = Column(SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)  # Status of the conversation
    project_ids = Column(ARRAY(String), nullable=False)
    agent_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), nullable=False)  # Use UTC timestamp
    updated_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), onupdate=func.utcnow(), nullable=False)  # Use UTC timestamp

    # Relationship to the Message model
    messages = relationship("Message", back_populates="conversation")

    # Optional: Relationship back to User model (if needed)
    user = relationship("User", back_populates="conversations")
