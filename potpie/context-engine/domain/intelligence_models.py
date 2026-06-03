"""Normalized request/response models for the context intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from domain.graph_quality import GraphQualityReport
from domain.source_references import (
    FreshnessReport,
    SourceFallback,
    SourceReferenceRecord,
)
from domain.source_resolution import SourceResolutionResult


@dataclass
class ArtifactRef:
    """Reference to a work artifact (PR, issue, task, etc.)."""

    kind: str  # e.g. "pr", "issue", "task"
    identifier: str  # e.g. "694", "LINEAR-123"


@dataclass
class ContextScope:
    """Optional narrowing scope for deterministic lookups."""

    repo_name: str | None = None
    branch: str | None = None
    file_path: str | None = None
    function_name: str | None = None
    symbol: str | None = None
    pr_number: int | None = None  # when PR-scoped discussions/artifact are needed
    services: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    environment: str | None = None
    ticket_ids: list[str] = field(default_factory=list)
    user: str | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass
class ContextBudget:
    """Latency and result-size budget for agent context wraps."""

    max_items: int = 12
    max_tokens: int | None = None
    timeout_ms: int = 4000
    freshness: str = "prefer_fresh"


@dataclass
class ContextResolutionRequest:
    pot_id: str
    query: str
    consumer_hint: str | None = None
    artifact_ref: ArtifactRef | None = None
    scope: ContextScope | None = None
    intent: str | None = None
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    mode: str = "fast"
    source_policy: str = "references_only"
    budget: ContextBudget = field(default_factory=ContextBudget)
    as_of: datetime | None = None
    timeout_ms: int = 4000

    def __post_init__(self) -> None:
        if self.timeout_ms != 4000 and self.budget.timeout_ms == 4000:
            self.budget.timeout_ms = self.timeout_ms

    @property
    def effective_timeout_ms(self) -> int:
        return self.budget.timeout_ms if self.budget else self.timeout_ms

    @property
    def effective_max_items(self) -> int:
        value = self.budget.max_items if self.budget else 12
        return max(1, min(value, 50))


@dataclass
class CapabilitySet:
    semantic_search: bool = False
    artifact_context: bool = True
    change_history: bool = True
    decision_context: bool = True
    discussion_context: bool = True
    ownership_context: bool = True
    project_map_context: bool = False
    debugging_memory_context: bool = False
    causal_chain_context: bool = False


@dataclass
class CoverageReport:
    status: str  # "complete", "partial", "empty"
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    missing_reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolutionError:
    source: str
    error: str
    recoverable: bool = True


@dataclass
class ResolutionMeta:
    provider: str
    total_latency_ms: int = 0
    per_call_latency_ms: dict[str, int] = field(default_factory=dict)
    capabilities_used: list[str] = field(default_factory=list)
    schema_version: str = "1"


@dataclass
class ArtifactContext:
    kind: str
    identifier: str
    title: str | None = None
    summary: str | None = None
    author: str | None = None
    created_at: str | None = None
    url: str | None = None
    extra: dict[str, Any] | None = None


@dataclass
class ChangeRecord:
    file_path: str | None = None
    function_name: str | None = None
    artifact_ref: str | None = None
    summary: str | None = None
    date: str | None = None
    pr_number: int | None = None
    title: str | None = None
    change_type: str | None = None
    feature_area: str | None = None
    decisions: list[str] | None = None


@dataclass
class DecisionRecord:
    decision: str
    rationale: str | None = None
    alternatives_rejected: str | None = None
    source_ref: str | None = None
    file_path: str | None = None
    pr_number: int | None = None


@dataclass
class DiscussionRecord:
    source_ref: str | None = None
    file_path: str | None = None
    line: int | None = None
    participants: list[str] | None = None
    summary: str | None = None
    full_text: str | None = None
    headline: str | None = None


@dataclass
class OwnershipRecord:
    file_path: str
    owner: str
    confidence_signal: str | None = None


@dataclass
class ProjectContextRecord:
    """Canonical project-map context for Phase 4 entities."""

    family: str
    kind: str
    entity_key: str | None = None
    name: str | None = None
    summary: str | None = None
    status: str | None = None
    source_ref: str | None = None
    source_uri: str | None = None
    relationships: list[dict[str, Any]] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChainItem:
    """One step in a backwards-walked causal chain (cause → effect order)."""

    node_uuid: str
    name: str | None = None
    summary: str | None = None
    reference_time: str | None = None
    source_refs: list[str] = field(default_factory=list)
    confidence: float = 0.7
    relation_from_previous: str | None = None


@dataclass
class DebuggingMemoryRecord:
    """Reusable debugging knowledge for incidents, symptoms, and fixes."""

    family: str
    kind: str
    entity_key: str | None = None
    title: str | None = None
    summary: str | None = None
    status: str | None = None
    severity: str | None = None
    root_cause: str | None = None
    fix_type: str | None = None
    source_ref: str | None = None
    source_uri: str | None = None
    affected_scope: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_signals: list[dict[str, Any]] = field(default_factory=list)
    related_changes: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceBundle:
    request: ContextResolutionRequest
    semantic_hits: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactContext] = field(default_factory=list)
    changes: list[ChangeRecord] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    discussions: list[DiscussionRecord] = field(default_factory=list)
    ownership: list[OwnershipRecord] = field(default_factory=list)
    project_map: list[ProjectContextRecord] = field(default_factory=list)
    debugging_memory: list[DebuggingMemoryRecord] = field(default_factory=list)
    causal_chain: list[CausalChainItem] = field(default_factory=list)
    source_refs: list[SourceReferenceRecord] = field(default_factory=list)
    coverage: CoverageReport = field(
        default_factory=lambda: CoverageReport(status="empty")
    )
    freshness: FreshnessReport = field(default_factory=FreshnessReport)
    quality: GraphQualityReport = field(default_factory=GraphQualityReport)
    fallbacks: list[SourceFallback] = field(default_factory=list)
    source_resolution: SourceResolutionResult = field(
        default_factory=SourceResolutionResult
    )
    open_conflicts: list[dict[str, Any]] = field(default_factory=list)
    recommended_next_actions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[ResolutionError] = field(default_factory=list)
    meta: ResolutionMeta = field(
        default_factory=lambda: ResolutionMeta(provider="unknown")
    )
