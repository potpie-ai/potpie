"""Context-bound request markers for the public Context Engine façade.

The migration intentionally owns these request identities before it moves the
legacy request payloads out of ``potpie_context_core``. Payload fields remain
operation-specific mappings during that mechanical move; callers cannot supply
or override an engine context selector here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class EngineRequest:
    """Base for one explicitly named, context-bound engine operation request."""

    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolveRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class SearchRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class RecordRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class DataPlaneStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class CatalogRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class DescribeRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ReadRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class SearchEntitiesRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class MutateRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class NeighborhoodRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InspectRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ExportSnapshotRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ImportSnapshotRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class RepairRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ProposeRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class CommitRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class HistoryRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class QualityRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxAddRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxListRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxShowRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxClaimRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxMarkAppliedRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxMarkRejectedRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class InboxCloseRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class SubmitEventRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class SubmitArtifactRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class ProcessingStatusRequest(EngineRequest):
    pass


@dataclass(frozen=True, slots=True)
class NudgeRequest(EngineRequest):
    pass


__all__ = [name for name in globals() if name.endswith("Request")]
