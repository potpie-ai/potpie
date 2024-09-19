import enum

from sqlalchemy import TIMESTAMP, CheckConstraint, Column
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import relationship

from app.core.base_model import Base


# Define enums for the Prompt model
class PromptStatusType(enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class PromptType(enum.Enum):
    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, nullable=False)
    text = Column(Text, nullable=False)
    type = Column(SQLAEnum(PromptType), nullable=False)
    version = Column(Integer, default=1, nullable=False)
    status = Column(
        SQLAEnum(PromptStatusType), default=PromptStatusType.ACTIVE, nullable=False
    )
    created_by = Column(String, ForeignKey("users.uid"), nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Define constraints
    __table_args__ = (
        UniqueConstraint(
            "text", "version", "created_by", name="unique_text_version_user"
        ),
        CheckConstraint("version > 0", name="check_version_positive"),
        CheckConstraint("created_at <= updated_at", name="check_timestamps"),
    )

    # Define relationship to User
    creator = relationship("User", back_populates="created_prompts")


class AgentPromptMapping(Base):
    __tablename__ = "agent_prompt_mappings"

    id = Column(String, primary_key=True, nullable=False)
    agent_id = Column(String, nullable=False)
    prompt_id = Column(
        String, ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False
    )
    prompt_stage = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("agent_id", "prompt_stage", name="unique_agent_prompt_stage"),
    )
