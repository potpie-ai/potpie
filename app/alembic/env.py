import os
import time
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from alembic.operations import ops

from app.core.database import Base
from app.modules.users.model import User  # Import all your models
from app.modules.projects.model import Project
from app.modules.conversations.conversation.model import Conversation
from app.modules.conversations.message.model import Message


from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Interpret the config file for Python logging.
fileConfig(context.config.config_file_name)

# Construct the database URL from environment variables
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")

# Set the SQLAlchemy URL dynamically from the constructed DATABASE_URL
config = context.config
config.set_main_option("sqlalchemy.url", POSTGRES_SERVER)

# Add your models' metadata object for 'autogenerate' support
target_metadata = Base.metadata

def process_revision_directives(context, revision, directives):
    """Automatically prepend timestamp to migration filenames."""
    for directive in directives:
        if isinstance(directive, ops.MigrationScript):
            # Get the current timestamp
            timestamp = time.strftime('%Y%m%d%H%M%S')
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