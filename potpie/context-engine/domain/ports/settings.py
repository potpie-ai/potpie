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

    def graph_db_backend(self) -> str:
        """Structural graph backend: ``neo4j`` (default) or ``falkordb``."""

    def falkordb_url(self) -> str | None:
        ...

    def falkordb_graph_name(self) -> str:
        ...

    def falkordb_mode(self) -> str:
        """``lite`` (embedded) or ``server``."""

    def falkordb_lite_path(self) -> str:
        ...

    def backfill_max_prs_per_run(self) -> int:
        """Max merged PRs to ingest per backfill run (deterministic cap)."""
