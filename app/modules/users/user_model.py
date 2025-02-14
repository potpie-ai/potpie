from sqlalchemy import TIMESTAMP, Boolean, Column, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.core.base_model import Base
from app.modules.conversations.conversation.conversation_model import (  # noqa
    Conversation,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (  # noqa
    CustomAgent,
)
from app.modules.intelligence.prompts.prompt_model import Prompt  # noqa
from app.modules.projects.projects_model import Project  # noqa
from app.modules.users.user_preferences_model import UserPreferences  # noqa


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
    custom_agents = relationship("CustomAgent", back_populates="user")
