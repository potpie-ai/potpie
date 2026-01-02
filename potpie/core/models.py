"""SQLAlchemy models re-export for potpie library.

This module aggregates all SQLAlchemy models needed by the potpie library.
Importing this module ensures all models are registered with SQLAlchemy's
Base metadata and that forward references are resolved.

When using potpie as a library in external projects, import from here
instead of from app.core.models.
"""

from app.modules.conversations.conversation.conversation_model import (  # noqa: F401
    Conversation,
)
from app.modules.conversations.message.message_model import Message  # noqa: F401
from app.modules.integrations.integration_model import Integration  # noqa: F401
from app.modules.media.media_model import MessageAttachment  # noqa: F401
from app.modules.intelligence.prompts.prompt_model import (  # noqa: F401
    AgentPromptMapping,
    Prompt,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (  # noqa: F401
    CustomAgent,
    CustomAgentShare,
)
from app.modules.projects.projects_model import Project  # noqa: F401
from app.modules.search.search_models import SearchIndex  # noqa: F401
from app.modules.tasks.task_model import Task  # noqa: F401
from app.modules.users.user_model import User  # noqa: F401
from app.modules.users.user_preferences_model import UserPreferences  # noqa: F401

__all__ = [
    "Conversation",
    "Message",
    "Integration",
    "MessageAttachment",
    "AgentPromptMapping",
    "Prompt",
    "CustomAgent",
    "CustomAgentShare",
    "Project",
    "SearchIndex",
    "Task",
    "User",
    "UserPreferences",
]
