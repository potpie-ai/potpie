"""SQLAlchemy models re-export for potpie library.

This module aggregates all SQLAlchemy models needed by the potpie library.
Importing this module ensures all models are registered with SQLAlchemy's
Base metadata and that forward references are resolved.

When using potpie as a library in external projects, import from here
instead of from app.core.models.
"""

# Auth models
from app.modules.auth.auth_provider_model import (  # noqa: F401
    AuthAuditLog,
    OrganizationSSOConfig,
    PendingProviderLink,
    UserAuthProvider,
)

# Conversation models
from app.modules.conversations.conversation.conversation_model import (  # noqa: F401
    Conversation,
)
from app.modules.conversations.message.message_model import Message  # noqa: F401

# Intelligence models
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (  # noqa: F401
    CustomAgent,
    CustomAgentShare,
)
from app.modules.intelligence.prompts.prompt_model import (  # noqa: F401
    AgentPromptMapping,
    Prompt,
)

# Integration models
from app.modules.integrations.integration_model import Integration  # noqa: F401

# Media models
from app.modules.media.media_model import MessageAttachment  # noqa: F401

# Parsing models
from app.modules.parsing.models.inference_cache_model import InferenceCache  # noqa: F401

# Project models
from app.modules.projects.projects_model import Project  # noqa: F401

# Search models
from app.modules.search.search_models import SearchIndex  # noqa: F401

# Task models
from app.modules.tasks.task_model import Task  # noqa: F401

# User models
from app.modules.users.user_model import User  # noqa: F401
from app.modules.users.user_preferences_model import UserPreferences  # noqa: F401

__all__ = [
    # Auth
    "AuthAuditLog",
    "OrganizationSSOConfig",
    "PendingProviderLink",
    "UserAuthProvider",
    # Conversations
    "Conversation",
    "Message",
    # Intelligence
    "CustomAgent",
    "CustomAgentShare",
    "AgentPromptMapping",
    "Prompt",
    # Integrations
    "Integration",
    # Media
    "MessageAttachment",
    # Parsing
    "InferenceCache",
    # Projects
    "Project",
    # Search
    "SearchIndex",
    # Tasks
    "Task",
    # Users
    "User",
    "UserPreferences",
]
