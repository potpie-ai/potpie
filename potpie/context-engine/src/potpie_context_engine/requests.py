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
    query_threshold: float | None = None


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
    # Return the durable receipt before the slow readback; the caller verifies
    # it separately so a failed check cannot hide a write that landed.
    defer_verification: bool = False


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
    include_claims: bool = True


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


# --- document resources -----------------------------------------------------
# Document payloads (the bytes the graph only points at) and the retrieval
# index over them. See ``docs/context-graph/resources.md``.


@dataclass(frozen=True, slots=True)
class ResourceImportRequest(EngineRequest):
    """Absorb one chunk directory as document ``doc``; re-import replaces it.

    ``files`` is the directory's *contents*, keyed by POSIX-relative path, read
    by the caller (``core.ports.resource_store.read_import_files``). There is
    deliberately no server-side path field: the operation never reads the
    executing host's filesystem on a caller's behalf, so it means the same thing
    in-process, behind the loopback daemon, and behind any future remote
    transport. A host that composes ``ResourceFacade`` itself may still pass
    ``source_dir`` to it for a job running beside the store; that path must
    never be accepted from a remote caller.
    """

    doc: str = ""
    files: Mapping[str, str] = field(default_factory=dict)
    source_ref: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceGetRequest(EngineRequest):
    resource_ids: tuple[str, ...] = ()
    with_neighbors: bool = False


@dataclass(frozen=True, slots=True)
class ResourceListRequest(EngineRequest):
    doc: str = ""
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceRmRequest(EngineRequest):
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ResourceStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ResourceIndexStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ResourceIndexBuildRequest(EngineRequest):
    """Embed pending windows now; ``doc`` first re-derives that document's rows."""

    doc: str | None = None
    wait: bool = False


@dataclass(frozen=True, slots=True)
class ResourceIndexRebuildRequest(EngineRequest):
    """Drop and re-derive the index from the stored files (one ``doc`` or all)."""

    doc: str | None = None


# -- graph commit history, journal and rollback -------------------------------
#
# Only identifiers cross this boundary. Restore plans, inverse records and
# preview bodies are rebuilt and checked server-side; no request carries them.


@dataclass(frozen=True, slots=True)
class CommitStatusRequest(EngineRequest):
    plan_id: str = ""


@dataclass(frozen=True, slots=True)
class VerifyCommitRequest(EngineRequest):
    plan_id: str = ""


@dataclass(frozen=True, slots=True)
class JournalStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class CommitsRequest(EngineRequest):
    cursor: str | None = None
    limit: int = 50
    actor: str | None = None
    origin: str | None = None
    logical_key: str | None = None


@dataclass(frozen=True, slots=True)
class CommitShowRequest(EngineRequest):
    commit_id: str = ""
    offset: int = 0
    limit: int = 100


@dataclass(frozen=True, slots=True)
class RevertPreviewRequest(EngineRequest):
    commit_id: str = ""
    expected_head: str = ""


@dataclass(frozen=True, slots=True)
class RollbackPreviewRequest(EngineRequest):
    target_commit_id: str = ""
    expected_head: str = ""


@dataclass(frozen=True, slots=True)
class ApplyPreviewRequest(EngineRequest):
    preview_id: str = ""


@dataclass(frozen=True, slots=True)
class DisableRollbackRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class RebuildCommitsRequest(EngineRequest):
    pass


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
