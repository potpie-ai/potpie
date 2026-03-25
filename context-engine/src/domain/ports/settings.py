"""Runtime configuration for context-engine (port)."""

from typing import Protocol


class ContextEngineSettingsPort(Protocol):
    def is_enabled(self) -> bool:
        """CONTEXT_GRAPH_ENABLED or equivalent."""

    def neo4j_uri(self) -> str | None:
        ...

    def neo4j_user(self) -> str | None:
        ...

    def neo4j_password(self) -> str | None:
        ...
