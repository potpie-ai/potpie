from app.core.database import Base
from app.modules.users.user_model import User  #noqa
from app.modules.users.user_preferences_model import UserPreferences #noqa
from app.modules.conversations.conversation.conversation_model import Conversation #noqa
from app.modules.conversations.message.message_model import Message #noqa
from app.modules.intelligence.prompts.prompt_model import AgentPromptMapping, Prompt #noqa
from app.modules.projects.projects_model import Project #noqa
from app.modules.search.search_models import SearchIndex #noqa
from app.modules.tasks.task_model import Task #noqa