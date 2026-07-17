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

from potpie_context_core.domain.agent_envelope import AgentEnvelope
from potpie_context_core.domain.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_core.domain.semantic_mutations import (
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
class GraphDescribeRequest:
    """``graph describe`` — the executable contract for a subgraph or view.

    Routed through the service (not answered CLI-side) so the contract always
    reflects the daemon's ontology build, like every other graph command.
    """

    subgraph: str
    view: str | None = None
    include_examples: bool = False


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
    """``graph read`` — a V2-style read over a named subgraph/view."""

    pot_id: str
    subgraph: str
    view: str
    query: str | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 12
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    include_invalidated: bool = False
    freshness_preference: str = "balanced"
    depth: int | None = None
    direction: str | None = None
    environment: str | None = None
    source_refs: tuple[str, ...] = ()
    detail: str = "compact"
    relations: str = "summary"
    query_threshold: float = 0.70


@dataclass(frozen=True, slots=True)
class GraphReadResult:
    """Contract-shaped V2 read result.

    This is the graph workbench read body. It intentionally does not expose the
    internal ``AgentEnvelope`` used by lower-level readers.
    """

    view: str
    subgraph: str
    ok: bool = True
    status: str | None = None
    message: str | None = None
    items: tuple[Mapping[str, Any], ...] = ()
    coverage: tuple[Mapping[str, Any], ...] = ()
    freshness: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    match_mode: str = "lexical"
    backed: bool = True
    read_shape: str = "flat_claims"
    inline_relations: tuple[str, ...] = ()
    inline_relation_count: int = 0
    graph_contract_version: str = ""
    ontology_version: str = ""
    subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    unsupported: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    as_of: datetime | None = None
    detail: str = "compact"
    relations: str = "summary"

    def to_dict(self) -> dict[str, Any]:
        detail = normalize_read_detail(self.detail)
        relations = normalize_read_relations(self.relations)
        out = {
            "ok": self.ok,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
            "view": self.view,
            "subgraph": self.subgraph,
            "backed": self.backed,
            "match_mode": self.match_mode,
            "read_shape": self.read_shape,
            "inline_relations": list(self.inline_relations),
            "inline_relation_count": self.inline_relation_count,
            "detail": detail,
            "relations_detail": relations,
            "items": [
                read_item_for_detail(item, detail=detail, relations=relations)
                for item in self.items
            ],
            "coverage": [dict(report) for report in self.coverage],
            "freshness": dict(self.freshness),
            "quality": dict(self.quality),
            "source_refs": list(self.source_refs),
            "subgraph_versions": dict(self.subgraph_versions),
            "unsupported": [dict(item) for item in self.unsupported],
            "warnings": list(self.warnings),
            "as_of": self.as_of.isoformat() if self.as_of else None,
        }
        if self.status:
            out["status"] = self.status
        if self.message:
            out["message"] = self.message
        return out


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
    subgraph: str | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)
    truth: str | None = None
    source_system: str | None = None
    source_family: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    environment: str | None = None
    external_id: str | None = None
    source_refs: tuple[str, ...] = ()
    limit: int = 10
    supporting_claims: int = 0


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

    def describe(self, request: GraphDescribeRequest) -> dict[str, Any]:
        """Executable contract for one subgraph or view (``describe_contract``)."""
        ...

    def read(self, request: GraphReadRequest) -> GraphReadResult:
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


def normalize_read_detail(value: str | None) -> str:
    detail = (value or "compact").strip().lower()
    if detail not in {"compact", "full"}:
        raise ValueError("detail must be one of: compact, full")
    return detail


def normalize_read_relations(value: str | None) -> str:
    relations = (value or "summary").strip().lower()
    if relations not in {"summary", "full"}:
        raise ValueError("relations must be one of: summary, full")
    return relations


def read_item_for_detail(
    item: Mapping[str, Any], *, detail: str, relations: str
) -> dict[str, Any]:
    payload = dict(item)
    relation_items = _relation_items(payload.get("relations"))

    if detail == "full":
        out = dict(payload)
    else:
        out = {
            key: payload[key]
            for key in (
                "entity_key",
                "entity_type",
                "score",
                "summary",
                "status",
                "source_refs",
                "truth",
                "coverage_status",
            )
            if key in payload
        }
        claim = payload.get("claim")
        if isinstance(claim, Mapping):
            compact_claim = {
                key: claim.get(key)
                for key in (
                    "claim_key",
                    "predicate",
                    "subject_key",
                    "object_key",
                )
                if claim.get(key) is not None
            }
            if compact_claim:
                out["claim"] = compact_claim

    if relations == "full":
        out["relations"] = relation_items
        return out

    out.pop("relations", None)
    out["relation_count"] = len(relation_items)
    if relation_items:
        out["relation_predicates"] = sorted(
            {
                str(rel.get("predicate") or rel.get("type"))
                for rel in relation_items
                if rel.get("predicate") or rel.get("type")
            }
        )
        out["related_keys"] = _first_relation_keys(relation_items)
    return out


def _relation_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _first_relation_keys(relations: list[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for rel in relations:
        key = rel.get("related_key") or rel.get("to_key") or rel.get("from_key")
        if not isinstance(key, str) or not key or key in seen:
            continue
        keys.append(key)
        seen.add(key)
        if len(keys) >= 10:
            break
    return keys


__all__ = [
    "DataPlaneStatus",
    "GraphCatalogRequest",
    "GraphCatalogResult",
    "GraphDescribeRequest",
    "GraphEntityCandidate",
    "GraphEntitySearchRequest",
    "GraphEntitySearchResult",
    "GraphReadRequest",
    "GraphReadResult",
    "GraphService",
    "normalize_read_detail",
    "normalize_read_relations",
    "read_item_for_detail",
]
