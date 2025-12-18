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

    # Legacy provider fields (kept for backward compatibility)
    provider_info = Column(JSONB)
    provider_username = Column(String(255))

    # Organization fields (for SSO)
    organization = Column(String(255))  # Email domain (e.g., "acme.com")
    organization_name = Column(String(255))  # Human-readable name (e.g., "Acme Corp")

    # User relationships
    projects = relationship("Project", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    created_prompts = relationship("Prompt", back_populates="creator")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)
    custom_agents = relationship("CustomAgent", back_populates="user")

    # SSO relationships
    auth_providers = relationship(
        "UserAuthProvider",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def get_primary_provider(self):
        """Get the user's primary authentication provider"""
        for provider in self.auth_providers:
            if provider.is_primary:
                return provider
        return None

    def has_provider(self, provider_type: str) -> bool:
        """Check if user has a specific provider linked"""
        return any(p.provider_type == provider_type for p in self.auth_providers)
