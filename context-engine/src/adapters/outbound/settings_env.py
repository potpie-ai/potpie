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

    def backfill_max_prs_per_run(self) -> int:
        raw = os.getenv("CONTEXT_GRAPH_BACKFILL_MAX_PRS_PER_RUN", "100").strip()
        try:
            n = int(raw)
        except ValueError:
            n = 100
        return max(1, min(n, 500))
