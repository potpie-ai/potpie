"""Environment-backed ContextEngineSettingsPort."""

import os

from domain.ports.settings import ContextEngineSettingsPort


def context_engine_neo4j_uri() -> str | None:
    """Neo4j for the context graph (optional dedicated cluster/DB)."""
    return (
        os.getenv("CONTEXT_ENGINE_NEO4J_URI")
        or os.getenv("CONTEXT_ENGINE_NEO4J_URL")
        or ""
    ).strip() or None


def context_engine_neo4j_user() -> str | None:
    v = (
        os.getenv("CONTEXT_ENGINE_NEO4J_USERNAME")
        or os.getenv("CONTEXT_ENGINE_NEO4J_USER")
        or ""
    ).strip()
    return v or None


def context_engine_neo4j_password() -> str | None:
    v = (os.getenv("CONTEXT_ENGINE_NEO4J_PASSWORD") or "").strip()
    return v or None


class EnvContextEngineSettings(ContextEngineSettingsPort):
    def is_enabled(self) -> bool:
        raw = os.getenv("CONTEXT_GRAPH_ENABLED")
        if raw is None:
            return True
        s = raw.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        return s in ("1", "true", "yes", "on")

    def neo4j_uri(self) -> str | None:
        return (
            context_engine_neo4j_uri()
            or os.getenv("NEO4J_URI")
            or os.getenv("NEO4J_URL")
        )

    def neo4j_user(self) -> str | None:
        return (
            context_engine_neo4j_user()
            or os.getenv("NEO4J_USERNAME")
            or os.getenv("NEO4J_USER")
        )

    def neo4j_password(self) -> str | None:
        pw = context_engine_neo4j_password()
        if pw is not None:
            return pw
        return os.getenv("NEO4J_PASSWORD")

    def graph_db_backend(self) -> str:
        v = (os.getenv("GRAPH_DB_BACKEND") or "neo4j").strip().lower()
        return v or "neo4j"

    def falkordb_url(self) -> str | None:
        v = (
            os.getenv("CONTEXT_ENGINE_FALKORDB_URL") or os.getenv("FALKORDB_URL") or ""
        ).strip()
        return v or None

    def falkordb_graph_name(self) -> str:
        v = (
            os.getenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME")
            or os.getenv("FALKORDB_GRAPH_NAME")
            or ""
        ).strip()
        return v or "context_graph"

    def falkordb_mode(self) -> str:
        v = (
            (
                os.getenv("CONTEXT_ENGINE_FALKORDB_MODE")
                or os.getenv("FALKORDB_MODE")
                or ""
            )
            .strip()
            .lower()
        )
        return v or "lite"

    def falkordb_lite_path(self) -> str:
        v = (
            os.getenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH")
            or os.getenv("FALKORDB_LITE_PATH")
            or ""
        ).strip()
        return v or ".potpie/context_graph/falkordb.db"

    def backfill_max_prs_per_run(self) -> int:
        raw = os.getenv("CONTEXT_GRAPH_BACKFILL_MAX_PRS_PER_RUN", "100").strip()
        try:
            n = int(raw)
        except ValueError:
            n = 100
        return max(1, min(n, 500))
