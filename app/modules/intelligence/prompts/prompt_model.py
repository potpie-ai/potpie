import enum

from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey, func, UniqueConstraint, Integer, CheckConstraint
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.orm import relationship

from app.core.database import Base

# Define enums for the Prompt model
class PromptType(enum.Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"

class PromptVisibilityType(enum.Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"

class PromptStatusType(enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

class Prompt(Base):
    __tablename__ = 'prompts'
    
    id = Column(String, primary_key=True, nullable=False)  # UUID stored as a string, no length limit
    text = Column(Text, nullable=False)
    type = Column(SQLAEnum(PromptType), nullable=False)  # Using ENUM for type
    visibility = Column(SQLAEnum(PromptVisibilityType), nullable=False)  # Using ENUM for visibility
    version = Column(Integer, default=1, nullable=False)  # Version stored as an integer
    status = Column(SQLAEnum(PromptStatusType), default=PromptStatusType.ACTIVE, nullable=False)  # Using ENUM for status
    created_by = Column(String, ForeignKey('users.uid'), nullable=True)  # Nullable for system prompts
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Define constraints
    __table_args__ = (
        UniqueConstraint('text', 'version', 'created_by', name='unique_text_version_user'),
        CheckConstraint('version > 0', name='check_version_positive'),
        CheckConstraint('created_at <= updated_at', name='check_timestamps'),
        CheckConstraint(
            "(type = 'SYSTEM' AND created_by IS NULL) OR (type = 'USER' AND created_by IS NOT NULL)",
            name='check_system_user_prompts'
        ),
    )

class PromptAccess(Base):
    __tablename__ = 'prompt_access'
    
    prompt_id = Column(String, ForeignKey('prompts.id', ondelete='CASCADE'), primary_key=True)
    user_id = Column(String, ForeignKey('users.uid', ondelete='CASCADE'), primary_key=True)
    

Prompt.creator = relationship("User", back_populates="created_prompts")
Prompt.accesses = relationship("PromptAccess", back_populates="prompt")
PromptAccess.prompt = relationship("Prompt", back_populates="accesses")
PromptAccess.user = relationship("User", back_populates="accessible_prompts")
