"""Configuration for PotpieRuntime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from potpie.exceptions import ConfigurationError


@dataclass
class RuntimeConfig:
    """Configuration for PotpieRuntime.

    All required database connections and optional settings for the runtime.
    """

    # Required: PostgreSQL connection
    postgres_url: str

    # Required: Neo4j knowledge graph
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # Optional: Redis for caching/streaming
    redis_url: Optional[str] = None

    # Optional: LLM configuration
    llm_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None  # Deprecated: use llm_chat_model
    llm_chat_model: Optional[str] = None
    llm_inference_model: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Optional: Default user context for library operations
    default_user_id: str = "library-user"
    default_user_email: str = "library@potpie.local"

    # Optional: Development settings
    project_path: str = "./projects"
    development_mode: bool = False

    # Optional: Connection pool settings
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    def validate(self) -> None:
        """Validate configuration, raise ConfigurationError if invalid."""
        if not self.postgres_url:
            raise ConfigurationError("postgres_url is required")
        if not self.neo4j_uri:
            raise ConfigurationError("neo4j_uri is required")
        if not self.neo4j_username:
            raise ConfigurationError("neo4j_username is required")
        if not self.neo4j_password:
            raise ConfigurationError("neo4j_password is required")

        # Validate postgres URL format
        if not self.postgres_url.startswith(("postgresql://", "postgres://")):
            raise ConfigurationError(
                "postgres_url must start with postgresql:// or postgres://"
            )

        # Validate neo4j URI format
        if not self.neo4j_uri.startswith(("bolt://", "neo4j://", "neo4j+s://")):
            raise ConfigurationError(
                "neo4j_uri must start with bolt://, neo4j://, or neo4j+s://"
            )

        # Validate pool settings
        if self.db_pool_size < 1:
            raise ConfigurationError("db_pool_size must be at least 1")
        if self.db_max_overflow < 0:
            raise ConfigurationError("db_max_overflow must be non-negative")
        if self.db_pool_timeout < 1:
            raise ConfigurationError("db_pool_timeout must be at least 1")

    @classmethod
    def from_env(cls, env_prefix: str = "") -> RuntimeConfig:
        """Create configuration from environment variables.

        Args:
            env_prefix: Optional prefix for environment variables (e.g., "POTPIE_")

        Returns:
            RuntimeConfig populated from environment variables

        Raises:
            ConfigurationError: If required environment variables are missing
        """

        def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(f"{env_prefix}{key}", default)

        def get_env_required(key: str) -> str:
            value = get_env(key)
            if value is None:
                raise ConfigurationError(
                    f"Required environment variable {env_prefix}{key} is not set"
                )
            return value

        def get_env_bool(key: str, default: bool = False) -> bool:
            value = get_env(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "enabled")

        def get_env_int(key: str, default: int) -> int:
            value = get_env(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                raise ConfigurationError(
                    f"Environment variable {env_prefix}{key} must be an integer"
                )

        # Build Redis URL from components if individual vars are set
        redis_url = get_env("REDIS_URL")
        if redis_url is None:
            redis_host = get_env("REDISHOST", "localhost")
            redis_port = get_env_int("REDISPORT", 6379)
            redis_user = get_env("REDISUSER", "")
            redis_password = get_env("REDISPASSWORD", "")
            if redis_host:
                if redis_user and redis_password:
                    redis_url = f"redis://{redis_user}:{redis_password}@{redis_host}:{redis_port}/0"
                else:
                    redis_url = f"redis://{redis_host}:{redis_port}/0"

        config = cls(
            postgres_url=get_env_required("POSTGRES_SERVER"),
            neo4j_uri=get_env_required("NEO4J_URI"),
            neo4j_username=get_env_required("NEO4J_USERNAME"),
            neo4j_password=get_env_required("NEO4J_PASSWORD"),
            redis_url=redis_url,
            llm_provider=get_env("LLM_PROVIDER", "openai"),
            llm_api_key=get_env("OPENAI_API_KEY") or get_env("LLM_API_KEY"),
            llm_model=get_env("LLM_MODEL"),
            llm_chat_model=get_env("CHAT_MODEL") or get_env("LLM_MODEL"),
            llm_inference_model=get_env("INFERENCE_MODEL"),
            llm_base_url=get_env("LLM_API_BASE"),
            default_user_id=get_env("POTPIE_USER_ID", "library-user"),
            default_user_email=get_env("POTPIE_USER_EMAIL", "library@potpie.local"),
            project_path=get_env("PROJECT_PATH", "./projects"),
            development_mode=get_env_bool("isDevelopmentMode"),
            db_pool_size=get_env_int("DB_POOL_SIZE", 10),
            db_max_overflow=get_env_int("DB_MAX_OVERFLOW", 10),
            db_pool_timeout=get_env_int("DB_POOL_TIMEOUT", 30),
            db_pool_recycle=get_env_int("DB_POOL_RECYCLE", 1800),
        )

        config.validate()
        return config


class RuntimeConfigBuilder:
    """Builder pattern for RuntimeConfig."""

    def __init__(self):
        self._postgres_url: Optional[str] = None
        self._neo4j_uri: Optional[str] = None
        self._neo4j_username: Optional[str] = None
        self._neo4j_password: Optional[str] = None
        self._redis_url: Optional[str] = None
        self._llm_provider: str = "openai"
        self._llm_api_key: Optional[str] = None
        self._llm_model: Optional[str] = None
        self._llm_chat_model: Optional[str] = None
        self._llm_inference_model: Optional[str] = None
        self._llm_base_url: Optional[str] = None
        self._default_user_id: str = "library-user"
        self._default_user_email: str = "library@potpie.local"
        self._project_path: str = "./projects"
        self._development_mode: bool = False
        self._db_pool_size: int = 10
        self._db_max_overflow: int = 10
        self._db_pool_timeout: int = 30
        self._db_pool_recycle: int = 1800

    def postgres(self, url: str) -> RuntimeConfigBuilder:
        """Set PostgreSQL connection URL."""
        self._postgres_url = url
        return self

    def neo4j(self, uri: str, username: str, password: str) -> RuntimeConfigBuilder:
        """Set Neo4j connection details."""
        self._neo4j_uri = uri
        self._neo4j_username = username
        self._neo4j_password = password
        return self

    def redis(self, url: str) -> RuntimeConfigBuilder:
        """Set Redis connection URL."""
        self._redis_url = url
        return self

    def llm(
        self,
        provider: str,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        chat_model: Optional[str] = None,
        inference_model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> RuntimeConfigBuilder:
        """Set LLM configuration."""
        self._llm_provider = provider
        self._llm_api_key = api_key
        self._llm_model = model
        self._llm_chat_model = chat_model or model
        self._llm_inference_model = inference_model
        self._llm_base_url = base_url
        return self

    def user(
        self, user_id: str, email: str = "library@potpie.local"
    ) -> RuntimeConfigBuilder:
        """Set default user context for library operations."""
        self._default_user_id = user_id
        self._default_user_email = email
        return self

    def project_path(self, path: str) -> RuntimeConfigBuilder:
        """Set project storage path."""
        self._project_path = path
        return self

    def development_mode(self, enabled: bool = True) -> RuntimeConfigBuilder:
        """Enable or disable development mode."""
        self._development_mode = enabled
        return self

    def pool_settings(
        self,
        size: int = 10,
        max_overflow: int = 10,
        timeout: int = 30,
        recycle: int = 1800,
    ) -> RuntimeConfigBuilder:
        """Set database connection pool settings."""
        self._db_pool_size = size
        self._db_max_overflow = max_overflow
        self._db_pool_timeout = timeout
        self._db_pool_recycle = recycle
        return self

    def build(self) -> RuntimeConfig:
        """Build and validate the RuntimeConfig."""
        if self._postgres_url is None:
            raise ConfigurationError("postgres_url is required - call .postgres()")
        if self._neo4j_uri is None:
            raise ConfigurationError("neo4j connection is required - call .neo4j()")

        config = RuntimeConfig(
            postgres_url=self._postgres_url,
            neo4j_uri=self._neo4j_uri,
            neo4j_username=self._neo4j_username or "",
            neo4j_password=self._neo4j_password or "",
            redis_url=self._redis_url,
            llm_provider=self._llm_provider,
            llm_api_key=self._llm_api_key,
            llm_model=self._llm_model,
            llm_chat_model=self._llm_chat_model,
            llm_inference_model=self._llm_inference_model,
            llm_base_url=self._llm_base_url,
            default_user_id=self._default_user_id,
            default_user_email=self._default_user_email,
            project_path=self._project_path,
            development_mode=self._development_mode,
            db_pool_size=self._db_pool_size,
            db_max_overflow=self._db_max_overflow,
            db_pool_timeout=self._db_pool_timeout,
            db_pool_recycle=self._db_pool_recycle,
        )
        config.validate()
        return config
