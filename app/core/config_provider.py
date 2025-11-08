import os
from typing import Any

from dotenv import load_dotenv

from .storage_strategies import (
    S3StorageStrategy,
    GCSStorageStrategy,
    AzureStorageStrategy,
)

load_dotenv()


class MediaServiceConfigError(Exception):
    pass


class ConfigProvider:
    def __init__(self):
        self.neo4j_config = {
            "uri": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
        }
        self.github_key = os.getenv("GITHUB_PRIVATE_KEY")
        self.is_development_mode = os.getenv("isDevelopmentMode", "disabled")
        self.is_multimodal_enabled = os.getenv("isMultimodalEnabled", "auto")
        self.gcp_project_id = os.getenv("GCS_PROJECT_ID")
        self.gcp_bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.google_application_credentials = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.object_storage_provider = os.getenv(
            "OBJECT_STORAGE_PROVIDER", "auto"
        ).lower()

        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        self.aws_region = os.getenv("AWS_REGION")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        # Strategy registry
        self._storage_strategies = {
            "s3": S3StorageStrategy(),
            "gcs": GCSStorageStrategy(),
            "azure": AzureStorageStrategy(),
        }

    def get_neo4j_config(self):
        return self.neo4j_config

    def get_github_key(self):
        return self.github_key

    def is_github_configured(self):
        """Check if GitHub credentials are configured."""
        return bool(self.github_key and os.getenv("GITHUB_APP_ID"))

    def get_demo_repo_list(self):
        return [
            {
                "id": "demo8",
                "name": "langchain",
                "full_name": "langchain-ai/langchain",
                "private": False,
                "url": "https://github.com/langchain-ai/langchain",
                "owner": "langchain-ai",
            },
            {
                "id": "demo6",
                "name": "cal.com",
                "full_name": "calcom/cal.com",
                "private": False,
                "url": "https://github.com/calcom/cal.com",
                "owner": "calcom",
            },
            {
                "id": "demo5",
                "name": "formbricks",
                "full_name": "formbricks/formbricks",
                "private": False,
                "url": "https://github.com/formbricks/formbricks",
                "owner": "formbricks",
            },
            {
                "id": "demo3",
                "name": "gateway",
                "full_name": "Portkey-AI/gateway",
                "private": False,
                "url": "https://github.com/Portkey-AI/gateway",
                "owner": "Portkey-AI",
            },
            {
                "id": "demo2",
                "name": "crewAI",
                "full_name": "crewAIInc/crewAI",
                "private": False,
                "url": "https://github.com/crewAIInc/crewAI",
                "owner": "crewAIInc",
            },
            {
                "id": "demo1",
                "name": "agentops",
                "full_name": "AgentOps-AI/agentops",
                "private": False,
                "url": "https://github.com/AgentOps-AI/agentops",
                "owner": "AgentOps-AI",
            },
            {
                "id": "demo0",
                "name": "agentstack",
                "full_name": "AgentOps-AI/AgentStack",
                "private": False,
                "url": "https://github.com/AgentOps-AI/AgentStack",
                "owner": "AgentOps-AI",
            },
        ]

    def get_redis_url(self):
        redishost = os.getenv("REDISHOST", "localhost")
        redisport = int(os.getenv("REDISPORT", 6379))
        redisuser = os.getenv("REDISUSER", "")
        redispassword = os.getenv("REDISPASSWORD", "")
        # Construct the Redis URL
        if redisuser and redispassword:
            redis_url = f"redis://{redisuser}:{redispassword}@{redishost}:{redisport}/0"
        else:
            redis_url = f"redis://{redishost}:{redisport}/0"
        return redis_url

    def get_is_development_mode(self):
        return self.is_development_mode == "enabled"

    def get_is_multimodal_enabled(self) -> bool:
        """
        Determine if multimodal functionality is enabled.

        Logic:
        - "disabled": Always disabled regardless of GCP vars
        - "enabled": Force enabled (requires GCP vars, will fail if missing)
        - "auto": Automatic detection based on GCP variable presence (default)
        """

        if self.is_multimodal_enabled.lower() == "disabled":
            return False
        if self.is_multimodal_enabled.lower() == "enabled":
            return True
        else:  # "auto" mode
            return self._detect_object_storage_dependencies()[0]

    def get_media_storage_backend(self) -> str:
        _, backend = self._detect_object_storage_dependencies()
        return backend

    def get_object_storage_descriptor(self) -> dict[str, Any]:
        backend = self.get_media_storage_backend()
        strategy = self._storage_strategies.get(backend)

        if not strategy:
            raise MediaServiceConfigError(f"Unsupported storage provider: {backend}")

        try:
            return strategy.get_descriptor(self)
        except ValueError as e:
            raise MediaServiceConfigError(str(e)) from e

    def _detect_object_storage_dependencies(self) -> tuple[bool, str]:
        # Check explicit provider selection first
        if (
            self.object_storage_provider != "auto"
            and self.object_storage_provider in self._storage_strategies
        ):
            strategy = self._storage_strategies[self.object_storage_provider]
            is_ready = strategy.is_ready(self)
            return is_ready, self.object_storage_provider

        # Auto-detection: return first ready provider
        for provider, strategy in self._storage_strategies.items():
            if strategy.is_ready(self):
                return True, provider

        return False, "none"

    @staticmethod
    def get_stream_ttl_secs() -> int:
        return int(os.getenv("REDIS_STREAM_TTL_SECS", "900"))  # 15 minutes

    @staticmethod
    def get_stream_maxlen() -> int:
        return int(os.getenv("REDIS_STREAM_MAX_LEN", "1000"))

    @staticmethod
    def get_stream_prefix() -> str:
        return os.getenv("REDIS_STREAM_PREFIX", "chat:stream")


config_provider = ConfigProvider()
