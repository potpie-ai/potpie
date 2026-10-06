"""Engine-owned result types returned by the public Context Engine facade."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, TypeAlias, TypeVar

from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.core.graph_history import GraphHistoryResult
from potpie_context_engine.core.graph_inbox import GraphInboxResult
from potpie_context_engine.core.graph_plans import (
    GraphIngestionVerificationResult,
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
CommitStatusResult: TypeAlias = GraphMutationCommitResult
VerifyCommitResult: TypeAlias = GraphIngestionVerificationResult


@dataclass(frozen=True, slots=True)
class ResetContextResult:
    context_id: str
    reset: bool


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


@dataclass(frozen=True, slots=True)
class GraphJournalResult(Mapping[str, Any]):
    """JSON document answered by commit-history, journal and rollback operations.

    The commit service speaks in plain documents: commit headers, recorded field
    changes, a server-held preview. Its refusals stay inside the document as
    ``ok: false`` with a ``status`` code and ``reasons``, the same way a refused
    plan commit does, so a caller branches on ``ok`` rather than on an error.
    """

    document: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> GraphJournalResult:
        """Normalize a commit-service answer into plain JSON values."""

        return cls(document=_plain(value))

    @property
    def ok(self) -> bool:
        return self.document.get("ok", True) is not False

    def __getitem__(self, key: str) -> Any:
        return self.document[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.document)

    def __len__(self) -> int:
        return len(self.document)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


JournalStatusResult: TypeAlias = GraphJournalResult
CommitsResult: TypeAlias = GraphJournalResult
CommitShowResult: TypeAlias = GraphJournalResult
RevertPreviewResult: TypeAlias = GraphJournalResult
RollbackPreviewResult: TypeAlias = GraphJournalResult
ApplyPreviewResult: TypeAlias = GraphJournalResult
DisableRollbackResult: TypeAlias = GraphJournalResult
RebuildCommitsResult: TypeAlias = GraphJournalResult


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _plain(value.value)
    return value


ResultT = TypeVar("ResultT")


def result_from_payload(result_type: type[ResultT], payload: object) -> ResultT:
    """Reconstruct a protocol result using the operation catalog's exact type."""

    return decode_typed_value(payload, result_type)  # type: ignore[return-value]


__all__ = [
    "ApplyPreviewResult",
    "CatalogResult",
    "CommitResult",
    "CommitShowResult",
    "CommitStatusResult",
    "CommitsResult",
    "DataPlaneStatusResult",
    "DescribeResult",
    "DisableRollbackResult",
    "ExportSnapshotResult",
    "GraphJournalResult",
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
    "JournalStatusResult",
    "MutateResult",
    "NeighborhoodResult",
    "NudgeResult",
    "ProcessingStatusResult",
    "ProposeResult",
    "QualityResult",
    "ReadResult",
    "RebuildCommitsResult",
    "RecordResult",
    "RepairResult",
    "ResetContextResult",
    "ResolveResult",
    "RevertPreviewResult",
    "RollbackPreviewResult",
    "SearchEntitiesResult",
    "SearchResult",
    "SubmitArtifactResult",
    "SubmitEventResult",
    "VerifyCommitResult",
    "result_from_payload",
]
