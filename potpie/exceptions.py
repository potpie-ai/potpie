"""Exception hierarchy for PotpieRuntime library."""


class PotpieError(Exception):
    """Base exception for all Potpie library errors."""

    pass


class ConfigurationError(PotpieError):
    """Configuration is invalid or missing required values."""

    pass


class NotInitializedError(PotpieError):
    """Runtime or component not initialized."""

    pass


class DatabaseError(PotpieError):
    """Database operation failed."""

    pass


class Neo4jError(PotpieError):
    """Neo4j operation failed."""

    pass


class RedisError(PotpieError):
    """Redis operation failed."""

    pass


class ProjectError(PotpieError):
    """Project operation failed."""

    pass


class ProjectNotFoundError(ProjectError):
    """Project does not exist."""

    pass


class AgentError(PotpieError):
    """Agent operation failed."""

    pass


class AgentNotFoundError(AgentError):
    """Agent does not exist."""

    pass


class AgentExecutionError(AgentError):
    """Error during agent execution."""

    pass


class ParsingError(PotpieError):
    """Parsing operation failed."""

    pass


class UserError(PotpieError):
    """User operation failed."""

    pass


class UserNotFoundError(UserError):
    """User does not exist."""

    pass


class MediaError(PotpieError):
    """Media operation failed."""

    pass


class MediaNotFoundError(MediaError):
    """Media/attachment does not exist."""

    pass


class ConversationError(PotpieError):
    """Conversation operation failed."""

    pass


class ConversationNotFoundError(ConversationError):
    """Conversation does not exist."""

    pass


class RepositoryError(PotpieError):
    """Repository operation failed."""

    pass


class RepositoryNotFoundError(RepositoryError):
    """Repository does not exist."""

    pass
