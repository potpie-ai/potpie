from sqlalchemy import Column, String, TIMESTAMP, func, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from app.core.database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), ForeignKey("users.uid"), nullable=False)  # ForeignKey to User model
    project_ids = Column(ARRAY(String), nullable=False)
    agent_ids = Column(ARRAY(String), nullable=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp(), nullable=False)
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp(), nullable=False)

    # Relationship to the Message model
    messages = relationship("Message", back_populates="conversation")

    # Optional: Relationship back to User model (if needed)
    user = relationship("User", back_populates="conversations")
