"""Environment-backed ContextEngineSettingsPort."""

import os
from pathlib import Path

from potpie_context_engine.domain.ports.settings import ContextEngineSettingsPort


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


def context_engine_graph_db_backend() -> str:
    v = (os.getenv("GRAPH_DB_BACKEND") or "neo4j").strip().lower()
    return v or "neo4j"


def context_engine_falkordb_url() -> str | None:
    v = (
        os.getenv("CONTEXT_ENGINE_FALKORDB_URL") or os.getenv("FALKORDB_URL") or ""
    ).strip()
    return v or None


def context_engine_falkordb_graph_name() -> str:
    v = (
        os.getenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME")
        or os.getenv("FALKORDB_GRAPH_NAME")
        or ""
    ).strip()
    return v or "context_graph"


def context_engine_falkordb_mode() -> str:
    v = (
        (os.getenv("CONTEXT_ENGINE_FALKORDB_MODE") or os.getenv("FALKORDB_MODE") or "")
        .strip()
        .lower()
    )
    return v or "lite"


def context_engine_falkordb_lite_path() -> str:
    v = (
        os.getenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH")
        or os.getenv("FALKORDB_LITE_PATH")
        or ""
    ).strip()
    if v:
        return v
    # Home-rooted absolute default (same resolution as the pot store): a
    # relative default lands wherever the daemon's cwd happens to be — e.g.
    # inside site-packages for an installed CLI, wiped on reinstall.
    home = (os.getenv("CONTEXT_ENGINE_HOME") or "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".potpie"
    return str(base / "context_graph" / "falkordb.db")


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
        return context_engine_graph_db_backend()

    def falkordb_url(self) -> str | None:
        return context_engine_falkordb_url()

    def falkordb_graph_name(self) -> str:
        return context_engine_falkordb_graph_name()

    def falkordb_mode(self) -> str:
        return context_engine_falkordb_mode()

    def falkordb_lite_path(self) -> str:
        return context_engine_falkordb_lite_path()

    def backfill_max_prs_per_run(self) -> int:
        raw = os.getenv("CONTEXT_GRAPH_BACKFILL_MAX_PRS_PER_RUN", "100").strip()
        try:
            n = int(raw)
        except ValueError:
            n = 100
        return max(1, min(n, 500))
