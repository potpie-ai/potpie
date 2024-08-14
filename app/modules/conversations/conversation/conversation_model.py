from sqlalchemy import Table, Column, String, TIMESTAMP, func, ForeignKey, Enum as SQLAEnum, MetaData, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
import enum
from app.core.database import Base

# Define the Status Enum
class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

# MetaData object needed for Table creation
metadata = MetaData()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.uid"), nullable=False, index=True)  # ForeignKey to User model with index
    title = Column(String(255), nullable=False)  # Title of the conversation
    status = Column(SQLAEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)  # Status of the conversation
    project_ids = Column(ARRAY(String), nullable=False)  # Array of associated project IDs
    created_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), nullable=False)  # Use UTC timestamp
    updated_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), onupdate=func.utcnow(), nullable=False)  # Use UTC timestamp

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    user = relationship("User", back_populates="conversations")
    agents = relationship("Agent", secondary="conversation_agents", back_populates="conversations")

# Junction table for many-to-many relationship between Conversation and Agent
conversation_agents = Table(
    "conversation_agents",
    Base.metadata,
    Column("conversation_id", String(255), ForeignKey("conversations.id", ondelete="CASCADE")),
    Column("agent_id", String(255), ForeignKey("agents.id", ondelete="CASCADE"))
)
