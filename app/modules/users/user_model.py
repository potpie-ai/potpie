from sqlalchemy import TIMESTAMP, Boolean, Column, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.base_model import Base


class User(Base):
    __tablename__ = "users"

    uid = Column(String(255), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(255))
    email_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=func.now(), nullable=False)
    last_login_at = Column(TIMESTAMP(timezone=True), default=func.now())
    provider_info = Column(JSONB)
    provider_username = Column(String(255))

    # User relationships
    projects = relationship("Project", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    created_prompts = relationship("Prompt", back_populates="creator")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)
