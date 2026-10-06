"""Engine-owned result types returned by the public Context Engine facade."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.core.graph_history import GraphHistoryResult
from potpie_context_engine.core.graph_inbox import GraphInboxResult
from potpie_context_engine.core.graph_plans import (
    GraphMutationCommitResult,
    GraphMutationProposal,
)
from potpie_context_engine.core.graph_quality import GraphQualityResult
from potpie_context_engine.core.ports.agent_context import RecordReceipt
from potpie_context_engine.core.ports.graph.analytics import RepairReport
from potpie_context_engine.core.ports.graph.inspection import GraphSlice
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.core.ports.graph_service import (
    DataPlaneStatus,
    GraphCatalogResult,
    GraphEntitySearchResult,
    GraphReadResult,
)
from potpie_context_engine.core.ports.resource_index import (
    DrainReport,
    IndexReport,
    ResourceIndexStatus,
)
from potpie_context_engine.core.ports.resource_store import (
    ResourceBatchResult,
    ResourceStoreStatus,
    SectionManifest,
)
from potpie_context_engine.core.resource_to_semantic import ResourceDeleteResult
from potpie_context_engine.core.resource_to_semantic import (
    ResourceImportResult as CoreResourceImportResult,
)
from potpie_context_engine.core.semantic_mutations import SemanticMutationResult
from potpie_context_engine.domain.ingestion_event_models import (
    EventReceipt,
    IngestionEvent,
)
from potpie_context_engine.domain.nudge import GraphNudgeResult
from potpie_context_engine.typed_serialization import decode_typed_value


ResolveResult: TypeAlias = AgentEnvelope
SearchResult: TypeAlias = AgentEnvelope
RecordResult: TypeAlias = RecordReceipt
DataPlaneStatusResult: TypeAlias = DataPlaneStatus
CatalogResult: TypeAlias = GraphCatalogResult
ReadResult: TypeAlias = GraphReadResult
SearchEntitiesResult: TypeAlias = GraphEntitySearchResult
MutateResult: TypeAlias = SemanticMutationResult
NeighborhoodResult: TypeAlias = GraphSlice
InspectResult: TypeAlias = GraphSlice
ExportSnapshotResult: TypeAlias = SnapshotManifest
ImportSnapshotResult: TypeAlias = SnapshotManifest
RepairResult: TypeAlias = RepairReport
ProposeResult: TypeAlias = GraphMutationProposal
CommitResult: TypeAlias = GraphMutationCommitResult
HistoryResult: TypeAlias = GraphHistoryResult
QualityResult: TypeAlias = GraphQualityResult
InboxAddResult: TypeAlias = GraphInboxResult
InboxListResult: TypeAlias = GraphInboxResult
InboxShowResult: TypeAlias = GraphInboxResult
InboxClaimResult: TypeAlias = GraphInboxResult
InboxMarkAppliedResult: TypeAlias = GraphInboxResult
InboxMarkRejectedResult: TypeAlias = GraphInboxResult
InboxCloseResult: TypeAlias = GraphInboxResult
SubmitEventResult: TypeAlias = EventReceipt
SubmitArtifactResult: TypeAlias = EventReceipt
ProcessingStatusResult: TypeAlias = IngestionEvent
NudgeResult: TypeAlias = GraphNudgeResult


ResourceImportResult: TypeAlias = CoreResourceImportResult
ResourceGetResult: TypeAlias = ResourceBatchResult
"""Every chunk that resolved plus one outcome per requested id.

``status`` is ``"success"`` with empty ``outcomes`` when every id resolved;
anything else names the ids that need follow-up."""
ResourceRmResult: TypeAlias = ResourceDeleteResult
ResourceStatusResult: TypeAlias = ResourceStoreStatus
ResourceIndexStatusResult: TypeAlias = ResourceIndexStatus
ResourceIndexBuildResult: TypeAlias = DrainReport


@dataclass(frozen=True, slots=True)
class ResourceListResult:
    """A document's sections, each with its chunk refs and stored revision."""

    doc: str
    sections: tuple[SectionManifest, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceIndexRebuildResult:
    """One report per document the rebuild re-derived."""

    reports: tuple[IndexReport, ...] = ()


@dataclass(frozen=True, slots=True)
class ResetContextResult:
    context_id: str
    reset: bool
    resources_purged: bool | None = None
    """The resource store's own answer, after the graph reset succeeded.

    ``None`` means no resource store is composed, which is neither a purge nor
    a failed one."""


@dataclass(frozen=True, slots=True)
class DescribeResult(Mapping[str, Any]):
    """Executable graph contract returned by ``describe``."""

    document: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


ResultT = TypeVar("ResultT")


def result_from_payload(result_type: type[ResultT], payload: object) -> ResultT:
    """Reconstruct a protocol result using the operation catalog's exact type."""

    return decode_typed_value(payload, result_type)  # type: ignore[return-value]


__all__ = [
    "CatalogResult",
    "CommitResult",
    "DataPlaneStatusResult",
    "DescribeResult",
    "ExportSnapshotResult",
    "HistoryResult",
    "ImportSnapshotResult",
    "InboxAddResult",
    "InboxClaimResult",
    "InboxCloseResult",
    "InboxListResult",
    "InboxMarkAppliedResult",
    "InboxMarkRejectedResult",
    "InboxShowResult",
    "InspectResult",
    "MutateResult",
    "NeighborhoodResult",
    "NudgeResult",
    "ProcessingStatusResult",
    "ProposeResult",
    "QualityResult",
    "ReadResult",
    "RecordResult",
    "RepairResult",
    "ResetContextResult",
    "ResolveResult",
    "ResourceGetResult",
    "ResourceImportResult",
    "ResourceIndexBuildResult",
    "ResourceIndexRebuildResult",
    "ResourceIndexStatusResult",
    "ResourceListResult",
    "ResourceRmResult",
    "ResourceStatusResult",
    "SearchEntitiesResult",
    "SearchResult",
    "SubmitArtifactResult",
    "SubmitEventResult",
    "result_from_payload",
]
