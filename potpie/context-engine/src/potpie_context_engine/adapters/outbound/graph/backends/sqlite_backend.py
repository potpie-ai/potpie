"""Strict local SQLite + sqlite-vec ``GraphBackend`` profile."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from potpie_context_core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_core.lifecycle import DONE, FAILED, SetupPlan, StepResult
from potpie_context_core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.adapters.outbound.graph.backends._unimplemented import (
    UnimplementedSnapshot,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_analytics import (
    ClaimQueryAnalytics,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_inspection import (
    ClaimQueryInspection,
)
from potpie_context_engine.adapters.outbound.graph.backends.claim_query_semantic import (
    ClaimQuerySemanticSearch,
)
from potpie_context_engine.adapters.outbound.graph.entity_summary_repair import (
    repaired_entity_properties,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.claim_query import (
    SQLiteClaimQuery,
    encode_vector,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.connection import (
    SQLiteConnectionFactory,
)
from potpie_context_engine.adapters.outbound.graph.sqlite.mutation import SQLiteMutation
from potpie_context_engine.adapters.outbound.graph.sqlite.schema import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    ensure_schema,
    verify_vector_projection,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    build_strict_sentence_transformer_embedder,
    validate_strict_minilm_embedder,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_engine.domain.embedding_modes import (
    SEMANTIC_EMBEDDER_ALIASES,
    normalize_embedding_mode,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort

_PROFILE = "sqlite"


def default_sqlite_graph_path(home: Path | None = None) -> Path:
    configured = (os.getenv("CONTEXT_ENGINE_SQLITE_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return (home or default_home()) / "context_graph" / "graph.sqlite3"


@dataclass(slots=True)
class SQLiteGraphBackend:
    """Canonical SQLite graph store with strict MiniLM/sqlite-vec semantics."""

    home: Path = field(default_factory=default_home)
    path: Path | None = None
    embedder: EmbedderPort | None = None
    settings: Any = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    connections: SQLiteConnectionFactory | None = field(default=None, repr=False)
    _claim_query: SQLiteClaimQuery = field(init=False)
    _mutation: SQLiteMutation = field(init=False)
    _semantic: ClaimQuerySemanticSearch = field(init=False)
    _inspection: ClaimQueryInspection = field(init=False)
    _analytics: ClaimQueryAnalytics = field(init=False)

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        self.path = Path(self.path) if self.path is not None else default_sqlite_graph_path(
            self.home
        )
        if self.embedder is None:
            self.embedder = build_strict_sentence_transformer_embedder()
        validate_strict_minilm_embedder(self.embedder)
        if self.connections is None:
            self.connections = SQLiteConnectionFactory(self.path)
        self._claim_query = SQLiteClaimQuery(self.connections, self.embedder)
        self._mutation = SQLiteMutation(
            self.connections,
            self.embedder,
            definition=self.definition,
        )
        self._semantic = ClaimQuerySemanticSearch(self._claim_query)
        self._inspection = ClaimQueryInspection(self._claim_query)
        self._analytics = ClaimQueryAnalytics(
            self._claim_query,
            entity_summary_repair=self._repair_entity_summaries,
        )

    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def match_mode(self) -> str:
        return "vector"

    @property
    def claim_query(self) -> SQLiteClaimQuery:
        return self._claim_query

    @property
    def mutation(self) -> SQLiteMutation:
        return self._mutation

    @property
    def semantic(self) -> ClaimQuerySemanticSearch:
        return self._semantic

    @property
    def inspection(self) -> ClaimQueryInspection:
        return self._inspection

    @property
    def analytics(self) -> ClaimQueryAnalytics:
        return self._analytics

    @property
    def snapshot(self) -> UnimplementedSnapshot:
        return UnimplementedSnapshot(_PROFILE)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            semantic=True,
            inspection=True,
            analytics=True,
            snapshot=False,
        )

    def bind_definition(self, definition: GraphDefinition) -> SQLiteGraphBackend:
        return SQLiteGraphBackend(
            home=self.home,
            path=self.path,
            embedder=self.embedder,
            settings=self.settings,
            definition=definition,
            connections=self.connections,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        mode = normalize_embedding_mode(plan.embeddings)
        model = plan.embedding_model.strip()
        normalized_model = model.lower()
        if normalized_model.startswith("sentence-transformers/"):
            normalized_model = normalized_model[len("sentence-transformers/") :]
        if (
            mode not in SEMANTIC_EMBEDDER_ALIASES
            or normalized_model != DEFAULT_SENTENCE_TRANSFORMER_MODEL.lower()
        ):
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=(
                    "sqlite requires sentence-transformers/"
                    f"{DEFAULT_SENTENCE_TRANSFORMER_MODEL}; hashing and lexical "
                    "fallbacks are unsupported"
                ),
                metadata={
                    "profile": _PROFILE,
                    "path": str(self.path),
                    "embedding_mode": plan.embeddings,
                    "embedding_model": plan.embedding_model,
                },
            )
        if self.settings is not None:
            is_enabled = getattr(self.settings, "is_enabled", None)
            if callable(is_enabled) and not bool(is_enabled()):
                return StepResult(
                    step="backend.provision",
                    state=FAILED,
                    detail="sqlite backend is disabled by CONTEXT_GRAPH_ENABLED",
                    metadata={"profile": _PROFILE, "path": str(self.path)},
                )
        try:
            self._probe_embedder()
            assert self.connections is not None
            with self.connections.connect() as connection:
                ensure_schema(connection)
                verify_vector_projection(connection)
        except Exception as exc:  # noqa: BLE001 - setup reports stable failure.
            return StepResult(
                step="backend.provision",
                state=FAILED,
                detail=str(exc),
                metadata={
                    "profile": _PROFILE,
                    "path": str(self.path),
                    "embedding_provider": EMBEDDING_PROVIDER,
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_dim": EMBEDDING_DIM,
                },
            )
        return StepResult(
            step="backend.provision",
            state=DONE,
            detail=(
                f"sqlite graph store ready at {self.path}; "
                f"{EMBEDDING_MODEL} + sqlite-vec v0.1.9"
            ),
            metadata={
                "profile": _PROFILE,
                "path": str(self.path),
                "embedding_provider": EMBEDDING_PROVIDER,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
                "sqlite_vec_version": "v0.1.9",
            },
        )

    def _probe_embedder(self) -> None:
        assert self.embedder is not None
        validate_strict_minilm_embedder(self.embedder)
        encode_vector(self.embedder.embed("potpie sqlite semantic readiness"))

    def _repair_entity_summaries(self, pot_id: str) -> int:
        assert self.connections is not None
        with self.connections.connect() as connection:
            ensure_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                repaired = 0
                rows = list(
                    connection.execute(
                        """
                        SELECT entity_key, properties_json
                        FROM entities
                        WHERE pot_id = ?
                        """,
                        (pot_id,),
                    )
                )
                for row in rows:
                    properties = _json_mapping(row["properties_json"])
                    fixed = repaired_entity_properties(
                        str(row["entity_key"]), properties
                    )
                    if fixed is None:
                        continue
                    connection.execute(
                        """
                        UPDATE entities SET properties_json = ?
                        WHERE pot_id = ? AND entity_key = ?
                        """,
                        (
                            _json_dump(fixed),
                            pot_id,
                            str(row["entity_key"]),
                        ),
                    )
                    repaired += 1
                connection.commit()
                return repaired
            except BaseException:
                connection.rollback()
                raise


def _json_mapping(raw: Any) -> dict[str, Any]:
    import json

    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(
        value,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = ["SQLiteGraphBackend", "default_sqlite_graph_path"]
