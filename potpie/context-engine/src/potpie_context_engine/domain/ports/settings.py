"""Runtime configuration for context-engine (port)."""

from typing import Protocol


class ContextEngineSettingsPort(Protocol):
    def is_enabled(self) -> bool:
        """CONTEXT_GRAPH_ENABLED or equivalent."""

    def neo4j_uri(self) -> str | None: ...

    def neo4j_user(self) -> str | None: ...

    def neo4j_password(self) -> str | None: ...

    # -- Graph backend selection -------------------------------------------
    # Concrete defaults (not ``...``) so existing implementers keep working
    # without change: anything that doesn't override these reports the
    # default ``neo4j`` backend, leaving production untouched.
    def graph_db_backend(self) -> str:
        """Which graph backend to use: ``neo4j`` (default), ``falkordb``, etc."""
        return "neo4j"

    def falkordb_url(self) -> str | None:
        """Redis-protocol URL for FalkorDB server/container mode."""
        return None

    def falkordb_graph_name(self) -> str:
        """FalkorDB graph (keyspace) name for the context graph."""
        return "context_graph"

    def falkordb_mode(self) -> str:
        """FalkorDB runtime mode: ``lite`` (default, embedded) or ``server``."""
        return "lite"

    def falkordb_lite_path(self) -> str:
        """Filesystem path for the embedded FalkorDBLite database (lite mode)."""
        return ".potpie/context_graph/falkordb.db"

    def ladybug_path(self) -> str:
        """Filesystem path for the embedded Ladybug database."""
        return ".potpie/context_graph/ladybug.lbdb"

    def ladybug_vector_efs(self) -> int:
        """HNSW search depth (``efs``) for ``QUERY_VECTOR_INDEX``."""
        return 200

    def ladybug_vector_efc(self) -> int:
        """HNSW construction effort (``efc``) for ``CREATE_VECTOR_INDEX``."""
        return 200

    def ladybug_vector_mu(self) -> int:
        """HNSW upper-layer max degree (``mu``)."""
        return 30

    def ladybug_vector_ml(self) -> int:
        """HNSW lower-layer max degree (``ml``)."""
        return 60

    def ladybug_vector_pu(self) -> float:
        """HNSW upper-layer sampling fraction (``pu``)."""
        return 0.05

    def ladybug_vector_metric(self) -> str:
        """Vector distance metric: ``cosine``, ``l2``, ``l2sq``, ``dotproduct``."""
        return "cosine"

    def ladybug_vector_cache_embeddings(self) -> bool:
        """Cache embedding column in RAM during index build."""
        return True

    def ladybug_vector_normalize_embeddings(self) -> bool:
        """L2-normalize embeddings at write/query time (recommended for cosine)."""
        return True

    def backfill_max_prs_per_run(self) -> int:
        """Max merged PRs to ingest per backfill run (deterministic cap)."""
