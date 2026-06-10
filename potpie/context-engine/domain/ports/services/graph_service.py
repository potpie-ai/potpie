"""``GraphService`` — the data-plane service.

The data plane behind three of the four tools: ``resolve``, ``search``, and
``record``. It owns the readers, ranking, record lowering, and envelope
assembly, and it talks to a ``GraphBackend`` (claim_query + mutation + semantic)
— never to a store directly.

``GraphService`` is *not* the agent contract. ``AgentContextPort`` is the public
4-tool surface; ``GraphService`` is the data-plane half it composes (the other
halves being ``PotManagementService`` and ``SkillManager``). The two share the
same request DTOs but sit at different altitudes: ``GraphService.resolve`` does
the read work; ``AgentContextPort.resolve`` is the public binding to it.

Graph Surface Lite (V1.5) adds four V2-shaped methods — ``catalog`` / ``read`` /
``search_entities`` / ``mutate`` — that discover, query, and directly mutate the
graph. They are CLI-only in V1.5 (the MCP surface stays at exactly four tools).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol

from domain.agent_envelope import AgentEnvelope
from domain.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from domain.semantic_mutations import (
    SemanticMutationRequest,
    SemanticMutationResult,
)


@dataclass(frozen=True, slots=True)
class DataPlaneStatus:
    """Data-plane half of ``context_status``: backend readiness + coverage."""

    pot_id: str
    backend_profile: str
    backend_ready: bool
    reader_backed_includes: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    freshness: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    match_mode: str = "lexical"
    """Active semantic-match mode (``vector`` | ``lexical``) so empty results
    are debuggable, not mysterious (Retrieval Hardening / R1)."""
    detail: str | None = None


# --- Graph Surface Lite DTOs ------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphCatalogRequest:
    """``graph catalog`` — discover the graph contract.

    ``task`` is accepted now but ignored in V1.5 (V2 turns it into a
    subgraph/view ranker); accepting it keeps that change additive.
    """

    pot_id: str
    task: str | None = None
    subgraph: str | None = None


@dataclass(frozen=True, slots=True)
class GraphCatalogResult:
    """The static contract a harness needs to use the graph without docs."""

    graph_contract_version: str
    ontology_version: str
    commands: tuple[str, ...]
    truth_classes: tuple[str, ...]
    mutation_operations: tuple[str, ...]
    review_required_operations: tuple[str, ...]
    deferred_operations: tuple[str, ...]
    views: tuple[Mapping[str, Any], ...]
    entity_types: tuple[Mapping[str, Any], ...]
    predicates: tuple[Mapping[str, Any], ...]
    match_mode: str = "lexical"
    source_authorities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
            "commands": list(self.commands),
            "truth_classes": list(self.truth_classes),
            "mutation_operations": list(self.mutation_operations),
            "review_required_operations": list(self.review_required_operations),
            "deferred_operations": list(self.deferred_operations),
            "source_authorities": list(self.source_authorities),
            "match_mode": self.match_mode,
            "views": [dict(v) for v in self.views],
            "entity_types": [dict(e) for e in self.entity_types],
            "predicates": [dict(p) for p in self.predicates],
        }


@dataclass(frozen=True, slots=True)
class GraphReadRequest:
    """``graph read`` — a V2-style read over a named view."""

    pot_id: str
    view: str
    query: str | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 12
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    freshness_preference: str = "balanced"
    depth: int | None = None
    direction: str | None = None
    environment: str | None = None


@dataclass(frozen=True, slots=True)
class GraphEntityCandidate:
    """One entity surfaced by ``search_entities``."""

    key: str
    labels: tuple[str, ...]
    name: str | None = None
    summary: str | None = None
    description: str | None = None
    score: float = 0.0
    supporting_claims: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEntitySearchRequest:
    """``graph search-entities`` — typed structured lookup for identity resolution."""

    pot_id: str
    query: str
    type: str | None = None
    predicate: str | None = None
    environment: str | None = None
    limit: int = 10


@dataclass(frozen=True, slots=True)
class GraphEntitySearchResult:
    entities: tuple[GraphEntityCandidate, ...]
    match_mode: str
    graph_contract_version: str
    ontology_version: str
    subgraph_versions: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
            "match_mode": self.match_mode,
            "subgraph_versions": dict(self.subgraph_versions),
            "entities": [
                {
                    "key": c.key,
                    "labels": list(c.labels),
                    "name": c.name,
                    "summary": c.summary,
                    "description": c.description,
                    "score": c.score,
                    "supporting_claims": [dict(s) for s in c.supporting_claims],
                }
                for c in self.entities
            ],
        }


class GraphService(Protocol):
    """Data plane for resolve/search/record + Graph Surface Lite."""

    def resolve(self, request: ResolveRequest) -> AgentEnvelope:
        """Expand intent → includes, run readers over the backend, rank, and
        assemble one ``AgentEnvelope``."""
        ...

    def search(self, request: SearchRequest) -> AgentEnvelope:
        """Narrow lookup; same envelope shape as ``resolve``."""
        ...

    def record(self, request: RecordRequest) -> RecordReceipt:
        """Lower a durable record into a semantic mutation and apply it."""
        ...

    def data_plane_status(self, pot_id: str) -> DataPlaneStatus:
        """Backend readiness + coverage for ``context_status``."""
        ...

    # --- Graph Surface Lite (V1.5) -----------------------------------------
    def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult:
        """Return the V1.5 graph contract (versions, views, ops, ontology)."""
        ...

    def read(self, request: GraphReadRequest) -> AgentEnvelope:
        """V2-style read over a named view, routed through the read trunk."""
        ...

    def search_entities(
        self, request: GraphEntitySearchRequest
    ) -> GraphEntitySearchResult:
        """Narrow entity/claim lookup for identity resolution before writes."""
        ...

    def mutate(self, request: SemanticMutationRequest) -> SemanticMutationResult:
        """Validate, risk-classify, lower, and dry-run or apply semantic mutations."""
        ...


__all__ = [
    "DataPlaneStatus",
    "GraphCatalogRequest",
    "GraphCatalogResult",
    "GraphEntityCandidate",
    "GraphEntitySearchRequest",
    "GraphEntitySearchResult",
    "GraphReadRequest",
    "GraphService",
]
