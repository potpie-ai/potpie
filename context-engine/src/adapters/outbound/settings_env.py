"""Environment-backed ContextEngineSettingsPort."""

import os

from domain.ports.settings import ContextEngineSettingsPort


class EnvContextEngineSettings(ContextEngineSettingsPort):
    def is_enabled(self) -> bool:
        return os.getenv("CONTEXT_GRAPH_ENABLED", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def neo4j_uri(self) -> str | None:
        return os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")

    def neo4j_user(self) -> str | None:
        return os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")

    def neo4j_password(self) -> str | None:
        return os.getenv("NEO4J_PASSWORD")
