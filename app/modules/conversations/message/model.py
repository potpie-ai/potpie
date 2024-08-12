from sqlalchemy import Column, String, ForeignKey, TIMESTAMP, func, Text
from sqlalchemy.orm import relationship
from app.core.database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(255), primary_key=True)
    conversation_id = Column(String(255), ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    sender_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp(), nullable=False)

    # Relationship to the Conversation model
    conversation = relationship("Conversation", back_populates="messages")
