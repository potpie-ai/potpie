from sqlalchemy import TIMESTAMP, Column, String, ForeignKey, Text, ARRAY, func
from sqlalchemy.orm import relationship
from app.core.base_model import Base


class SharedChat(Base):
    __tablename__ = "shared_chats"

    id = Column(String(255), primary_key=True)  # Unique identifier for the shared chat
    conversation_id = Column(
        String(255),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    shared_with_emails = Column(ARRAY(String), nullable=False)  # List of emails to share with
    share_link = Column(String(255), unique=True, nullable=False)  # Unique link for accessing shared chat
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)

    # Relationship to the Conversation model
    conversation = relationship("Conversation", back_populates="shared_chats")
