import os
import time
from logging.config import fileConfig

from alembic import context
from alembic.operations import ops
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from app.core.base_model import Base
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message
from app.modules.integrations.integration_model import Integration
from app.modules.media.media_model import MessageAttachment
from app.modules.intelligence.prompts.prompt_model import AgentPromptMapping, Prompt
from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent, CustomAgentShare
from app.modules.projects.projects_model import Project
from app.modules.search.search_models import SearchIndex
from app.modules.tasks.task_model import Task
from app.modules.users.user_model import User
from app.modules.users.user_preferences_model import UserPreferences
from app.modules.auth.auth_provider_model import UserAuthProvider, PendingProviderLink, OrganizationSSOConfig, AuthAuditLog

# Register all models with SQLAlchemy by referencing them
_MODELS = (Conversation, Message, Integration, MessageAttachment, AgentPromptMapping, Prompt,
           CustomAgent, CustomAgentShare, Project, SearchIndex, Task, User, UserPreferences,
           UserAuthProvider, PendingProviderLink, OrganizationSSOConfig, AuthAuditLog)

target_metadata = Base.metadata
# Load environment variables from .env
load_dotenv(override=True)

# Interpret the config file for Python logging.
fileConfig(context.config.config_file_name)
target_metadata = Base.metadata
# Construct the database URL from environment variables
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")

# Set the SQLAlchemy URL dynamically from the constructed DATABASE_URL
config = context.config
config.set_main_option("sqlalchemy.url", POSTGRES_SERVER)

# Add your models' metadata object for 'autogenerate' support


def process_revision_directives(context, revision, directives):
    """Automatically prepend timestamp to migration filenames."""
    for directive in directives:
        if isinstance(directive, ops.MigrationScript):
            # Get the current timestamp
            timestamp = time.strftime("%Y%m%d%H%M%S")
            # Modify the revision ID to include the timestamp
            directive.rev_id = f"{timestamp}_{directive.rev_id}"


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table="alembic_version",
            compare_type=True,
            process_revision_directives=process_revision_directives,  # Add the timestamp hook here
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise Exception("Offline migrations not supported")
else:
    run_migrations_online()
