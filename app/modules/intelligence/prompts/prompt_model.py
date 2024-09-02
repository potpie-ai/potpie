import enum

from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey, func, UniqueConstraint, Integer
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.orm import relationship

from app.core.database import Base

# Define enums for the Prompt model
class PromptType(enum.Enum):
    SYSTEM = "System"
    USER = "User"

class PromptVisibilityType(enum.Enum):
    PUBLIC = "Public"
    PRIVATE = "Private"

class PromptStatusType(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

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
        UniqueConstraint('text', 'version', name='unique_text_version'),
    )

    # Relationships
    creator = relationship("User", back_populates="prompts")
    accesses = relationship("PromptAccess", back_populates="prompt")

class PromptAccess(Base):
    __tablename__ = 'prompt_access'
    
    prompt_id = Column(String, ForeignKey('prompts.id', ondelete='CASCADE'), primary_key=True)
    user_id = Column(String, ForeignKey('users.uid', ondelete='CASCADE'), primary_key=True)
    
    # Relationships
    prompt = relationship("Prompt", back_populates="accesses")
    user = relationship("User", back_populates="accessible_prompts")
