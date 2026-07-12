"""Stable request/result DTOs supported across the package boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from potpie_context_engine.domain.agent_envelope import AgentEnvelope
from potpie_context_engine.domain.graph_history import (
    GraphHistoryRequest,
    GraphHistoryResult,
)
from potpie_context_engine.domain.graph_plans import (
    GraphMutationCommitResult,
    GraphMutationProposal,
)
from potpie_context_engine.domain.graph_quality import GraphQualityResult
from potpie_context_engine.domain.ports.agent_context import (
    RecordReceipt,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)
from potpie_context_engine.domain.ports.graph.mutation import BackendReadiness
from potpie_context_engine.domain.ports.ledger.client import (
    LedgerHealth,
    LedgerPage,
    LedgerSource,
)
from potpie_context_engine.domain.ports.services.graph_service import (
    GraphCatalogRequest,
    GraphCatalogResult,
    GraphEntitySearchRequest,
    GraphEntitySearchResult,
    GraphReadRequest,
    GraphReadResult,
)
from potpie_context_engine.domain.ports.services.pot_management import (
    PotInfo,
    SourceInfo,
)


@dataclass(frozen=True, slots=True)
class EngineStatusRequest:
    pot_id: str | None = None


@dataclass(frozen=True, slots=True)
class EngineStatusReport:
    schema_version: str
    pot_id: str | None
    pot_name: str | None
    backend: str
    backend_ready: bool
    storage_ready: bool
    ingestion_ready: bool
    source_count: int
    last_ingestion_at: datetime | None = None
    degraded_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmptyRequest:
    pass


@dataclass(frozen=True, slots=True)
class PotListResult:
    items: tuple[PotInfo, ...]
    count: int


@dataclass(frozen=True, slots=True)
class PotInfoRequest:
    ref: str | None = None


@dataclass(frozen=True, slots=True)
class PotCreateRequest:
    name: str
    repo: str | None = None
    use: bool = False


@dataclass(frozen=True, slots=True)
class PotUseRequest:
    ref: str


@dataclass(frozen=True, slots=True)
class PotRenameRequest:
    ref: str
    new_name: str


@dataclass(frozen=True, slots=True)
class PotResetRequest:
    ref: str
    confirm: bool = False


@dataclass(frozen=True, slots=True)
class PotArchiveRequest:
    ref: str


@dataclass(frozen=True, slots=True)
class SourceAddRequest:
    pot_id: str
    kind: str
    location: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class SourceListRequest:
    pot_id: str


@dataclass(frozen=True, slots=True)
class SourceListResult:
    items: tuple[SourceInfo, ...]
    count: int


@dataclass(frozen=True, slots=True)
class SourceStatusRequest:
    pot_id: str
    source_id: str


@dataclass(frozen=True, slots=True)
class SourceRemoveRequest:
    pot_id: str
    source_id: str


@dataclass(frozen=True, slots=True)
class OperationResult:
    ok: bool = True


@dataclass(frozen=True, slots=True)
class GraphStatusRequest:
    pot_id: str


@dataclass(frozen=True, slots=True)
class GraphProposeRequest:
    pot_id: str
    payload: Mapping[str, Any]
    ttl_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class GraphCommitRequest:
    pot_id: str
    plan_id: str
    approved_by: str | None = None
    verify: bool = False


@dataclass(frozen=True, slots=True)
class GraphQualityRequest:
    pot_id: str
    report: str = "summary"
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LedgerStatusRequest:
    pass


@dataclass(frozen=True, slots=True)
class LedgerSourcesRequest:
    pot_id: str


@dataclass(frozen=True, slots=True)
class LedgerSourcesResult:
    items: tuple[LedgerSource, ...]
    count: int


@dataclass(frozen=True, slots=True)
class LedgerQueryRequest:
    pot_id: str
    source_id: str | None = None
    kind: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 100


@dataclass(frozen=True, slots=True)
class LedgerPullRequest:
    pot_id: str
    source_id: str
    limit: int = 100


@dataclass(frozen=True, slots=True)
class TimelineRecentRequest:
    pot_id: str
    query: str | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 20
    since: datetime | None = None
    until: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProvisionInspectRequest:
    pot_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProvisionStep:
    name: str
    required: bool
    state: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ProvisionPlan:
    backend: str
    data_dir: str | None
    steps: tuple[ProvisionStep, ...]


@dataclass(frozen=True, slots=True)
class ProvisionApplyRequest:
    pot_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProvisionReport:
    ok: bool
    backend: str
    steps: tuple[ProvisionStep, ...]


__all__ = [
    "AgentEnvelope",
    "BackendReadiness",
    "EmptyRequest",
    "EngineStatusReport",
    "EngineStatusRequest",
    "GraphCatalogRequest",
    "GraphCatalogResult",
    "GraphCommitRequest",
    "GraphEntitySearchRequest",
    "GraphEntitySearchResult",
    "GraphHistoryRequest",
    "GraphHistoryResult",
    "GraphMutationCommitResult",
    "GraphMutationProposal",
    "GraphProposeRequest",
    "GraphQualityRequest",
    "GraphQualityResult",
    "GraphReadRequest",
    "GraphReadResult",
    "GraphStatusRequest",
    "LedgerHealth",
    "LedgerPage",
    "LedgerPullRequest",
    "LedgerQueryRequest",
    "LedgerSource",
    "LedgerSourcesRequest",
    "LedgerSourcesResult",
    "LedgerStatusRequest",
    "OperationResult",
    "PotArchiveRequest",
    "PotCreateRequest",
    "PotInfo",
    "PotInfoRequest",
    "PotListResult",
    "PotRenameRequest",
    "PotResetRequest",
    "PotUseRequest",
    "ProvisionApplyRequest",
    "ProvisionInspectRequest",
    "ProvisionPlan",
    "ProvisionReport",
    "ProvisionStep",
    "RecordReceipt",
    "RecordRequest",
    "ResolveRequest",
    "SearchRequest",
    "SourceAddRequest",
    "SourceInfo",
    "SourceListRequest",
    "SourceListResult",
    "SourceRemoveRequest",
    "SourceStatusRequest",
    "TimelineRecentRequest",
]
