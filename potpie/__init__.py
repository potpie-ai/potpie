"""
PotpieRuntime - Native Python library for Potpie's code intelligence capabilities.

This library provides direct access to Potpie's core features without requiring
REST API calls, enabling embedding of code intelligence into Python applications.

Example:
    >>> from potpie import PotpieRuntime, RuntimeConfig
    >>>
    >>> config = RuntimeConfig(
    ...     postgres_url="postgresql://user:pass@localhost:5432/potpie",
    ...     neo4j_uri="bolt://localhost:7687",
    ...     neo4j_username="neo4j",
    ...     neo4j_password="password",
    ... )
    >>>
    >>> async with PotpieRuntime(config) as runtime:
    ...     # User context is passed per-operation, not at runtime creation
    ...     project_id = await runtime.projects.register(
    ...         repo_name="owner/repo",
    ...         branch_name="main",
    ...         user_id="user-123",
    ...     )
    ...     print(f"Registered project: {project_id}")

Or from environment:
    >>> from potpie import PotpieRuntime
    >>> runtime = PotpieRuntime.from_env()
"""

from potpie.runtime import PotpieRuntime
from potpie.config import RuntimeConfig, RuntimeConfigBuilder
from potpie.exceptions import (
    PotpieError,
    ConfigurationError,
    DatabaseError,
    Neo4jError,
    RedisError,
    NotInitializedError,
    ProjectError,
    ProjectNotFoundError,
    AgentError,
    AgentNotFoundError,
    AgentExecutionError,
    ParsingError,
    UserError,
    UserNotFoundError,
    MediaError,
    MediaNotFoundError,
    ConversationError,
    ConversationNotFoundError,
    RepositoryError,
    RepositoryNotFoundError,
)
from potpie.types import (
    ProjectInfo,
    ProjectStatus,
    ParsingResult,
    UserInfo,
    RepositoryInfo,
    RepositoryStatus,
    VolumeInfo,
)
from potpie.agents import (
    AgentRunner,
    AgentHandle,
    AgentInfo,
    ChatContext,
    ChatAgentResponse,
    ToolCallResponse,
)

__version__ = "0.1.0"

__all__ = [
    # Main entry point
    "PotpieRuntime",
    # Configuration
    "RuntimeConfig",
    "RuntimeConfigBuilder",
    # Types
    "ProjectInfo",
    "ProjectStatus",
    "ParsingResult",
    "UserInfo",
    "RepositoryInfo",
    "RepositoryStatus",
    "VolumeInfo",
    # Agent types
    "AgentRunner",
    "AgentHandle",
    "AgentInfo",
    "ChatContext",
    "ChatAgentResponse",
    "ToolCallResponse",
    # Base exception
    "PotpieError",
    # Configuration errors
    "ConfigurationError",
    # Infrastructure errors
    "DatabaseError",
    "Neo4jError",
    "RedisError",
    "NotInitializedError",
    # Domain errors
    "ProjectError",
    "ProjectNotFoundError",
    "AgentError",
    "AgentNotFoundError",
    "AgentExecutionError",
    "ParsingError",
    "UserError",
    "UserNotFoundError",
    "MediaError",
    "MediaNotFoundError",
    "ConversationError",
    "ConversationNotFoundError",
    "RepositoryError",
    "RepositoryNotFoundError",
]
