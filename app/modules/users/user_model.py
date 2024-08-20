from sqlalchemy import TIMESTAMP, Boolean, Column, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    uid = Column(String(255), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(255))
    email_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.utcnow(), nullable=False)  # Use UTC timestamp
    last_login_at = Column(TIMESTAMP, default=func.current_timestamp())
    provider_info = Column(JSONB)
    provider_username = Column(String(255))

    
    # Relationships
    projects = relationship(
        "Project", back_populates="user"
    )  # Assumes a 'Project' class exists
        # Relationship to Conversation model
    conversations = relationship("Conversation", back_populates="user")
