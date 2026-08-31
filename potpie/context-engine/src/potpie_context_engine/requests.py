"""Typed, context-bound requests for the public Context Engine facade."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping, TypeVar

from potpie_context_engine.core.actor import Actor


class EngineRequest:
    """Base for one explicitly named request with no context selector."""

    def to_payload(self) -> dict[str, object]:
        """Return the operation payload used by the typed daemon codec."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


RequestT = TypeVar("RequestT", bound=EngineRequest)


@dataclass(frozen=True, slots=True)
class ResolveRequest(EngineRequest):
    task: str | None = None
    intent: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "fast"
    source_policy: str = "references_only"
    max_items: int = 12
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    include_invalidated: bool = False
    freshness_preference: str = "balanced"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchRequest(EngineRequest):
    query: str = ""
    include: tuple[str, ...] = ()
    scope: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "fast"
    source_policy: str = "references_only"
    max_items: int = 12
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecordRequest(EngineRequest):
    record_type: str = ""
    summary: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    idempotency_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataPlaneStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class CatalogRequest(EngineRequest):
    task: str | None = None
    subgraph: str | None = None


@dataclass(frozen=True, slots=True)
class DescribeRequest(EngineRequest):
    subgraph: str = ""
    view: str | None = None
    include_examples: bool = False


@dataclass(frozen=True, slots=True)
class ReadRequest(EngineRequest):
    subgraph: str = ""
    view: str = ""
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
class SearchEntitiesRequest(EngineRequest):
    query: str = ""
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
class MutateRequest(EngineRequest):
    mutation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NeighborhoodRequest(EngineRequest):
    entity_key: str = ""
    depth: int = 2
    direction: str = "both"
    predicates: tuple[str, ...] = ()
    limit: int = 50


@dataclass(frozen=True, slots=True)
class InspectRequest(NeighborhoodRequest):
    pass


@dataclass(frozen=True, slots=True)
class ExportSnapshotRequest(EngineRequest):
    destination: str = ""


@dataclass(frozen=True, slots=True)
class ImportSnapshotRequest(EngineRequest):
    source: str = ""


@dataclass(frozen=True, slots=True)
class RepairRequest(EngineRequest):
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResetContextRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ProposeRequest(EngineRequest):
    mutation: Mapping[str, Any] = field(default_factory=dict)
    ttl_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class CommitRequest(EngineRequest):
    plan_id: str = ""
    approved_by: str | None = None
    verify: bool = False


@dataclass(frozen=True, slots=True)
class HistoryRequest(EngineRequest):
    entity_key: str | None = None
    claim_key: str | None = None
    subgraph: str | None = None
    plan_id: str | None = None
    mutation_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50


@dataclass(frozen=True, slots=True)
class QualityRequest(EngineRequest):
    report: str = "summary"
    subgraph: str | None = None
    limit: int = 50
    confidence_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class InboxAddRequest(EngineRequest):
    summary: str = ""
    details: str | None = None
    evidence: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    suspected_subgraphs: tuple[str, ...] = ()
    created_by: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InboxListRequest(EngineRequest):
    status: tuple[str, ...] = ()
    claimed_by: str | None = None
    suspected_subgraph: str | None = None
    source_ref: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 50


@dataclass(frozen=True, slots=True)
class InboxShowRequest(EngineRequest):
    item_id: str = ""


@dataclass(frozen=True, slots=True)
class InboxClaimRequest(EngineRequest):
    item_id: str = ""
    claimed_by: str | None = None


@dataclass(frozen=True, slots=True)
class InboxMarkAppliedRequest(EngineRequest):
    item_id: str = ""
    closed_by: str | None = None
    linked_plan_id: str | None = None
    linked_mutation_id: str | None = None


@dataclass(frozen=True, slots=True)
class InboxMarkRejectedRequest(EngineRequest):
    item_id: str = ""
    closed_by: str | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class InboxCloseRequest(EngineRequest):
    item_id: str = ""
    closed_by: str | None = None
    linked_plan_id: str | None = None
    linked_mutation_id: str | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SubmitEventRequest(EngineRequest):
    source_system: str = ""
    event_type: str = ""
    action: str = ""
    source_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    ingestion_kind: str = "agent_reconciliation"
    source_channel: str = "engine"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    dedup_key: str | None = None
    event_id: str | None = None
    provider: str | None = None
    provider_host: str | None = None
    repo_name: str | None = None
    source_event_id: str | None = None
    artifact_refs: tuple[str, ...] = ()
    occurred_at: datetime | None = None
    actor: Actor | None = None
    wait: bool = False
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SubmitArtifactRequest(EngineRequest):
    source_system: str = ""
    artifact_type: str = ""
    artifact_id: str = ""
    artifact: Mapping[str, Any] = field(default_factory=dict)
    source_ref: str | None = None
    source_channel: str = "engine"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    provider: str | None = None
    provider_host: str | None = None
    repo_name: str | None = None
    occurred_at: datetime | None = None
    actor: Actor | None = None
    wait: bool = False
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProcessingStatusRequest(EngineRequest):
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class NudgeRequest(EngineRequest):
    event: str = ""
    session_id: str = ""
    scope: Mapping[str, Any] = field(default_factory=dict)
    path: str | None = None
    query: str | None = None
    limit: int = 5


def request_from_payload(
    request_type: type[RequestT], payload: Mapping[str, object]
) -> RequestT:
    """Decode a wire mapping into one exact operation request type."""

    from potpie_context_engine.typed_serialization import decode_dataclass

    return decode_dataclass(request_type, payload)


__all__ = [
    "EngineRequest",
    "request_from_payload",
    *[
        name
        for name in globals()
        if name.endswith("Request") and name != "EngineRequest"
    ],
]
