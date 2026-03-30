"""Normalized request/response models for the context intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactRef:
    """Reference to a work artifact (PR, issue, task, etc.)."""

    kind: str  # e.g. "pr", "issue", "task"
    identifier: str  # e.g. "694", "LINEAR-123"


@dataclass
class ContextScope:
    """Optional narrowing scope for deterministic lookups."""

    file_path: str | None = None
    function_name: str | None = None
    symbol: str | None = None
    pr_number: int | None = None  # when PR-scoped discussions/artifact are needed


@dataclass
class ContextResolutionRequest:
    project_id: str
    query: str
    consumer_hint: str | None = None
    artifact_ref: ArtifactRef | None = None
    scope: ContextScope | None = None
    timeout_ms: int = 4000


@dataclass
class CapabilitySet:
    semantic_search: bool = False
    artifact_context: bool = True
    change_history: bool = True
    decision_context: bool = True
    discussion_context: bool = True
    ownership_context: bool = True
    workflow_context: bool = False


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
class IntelligenceBundle:
    request: ContextResolutionRequest
    semantic_hits: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ArtifactContext] = field(default_factory=list)
    changes: list[ChangeRecord] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    discussions: list[DiscussionRecord] = field(default_factory=list)
    ownership: list[OwnershipRecord] = field(default_factory=list)
    coverage: CoverageReport = field(default_factory=lambda: CoverageReport(status="empty"))
    errors: list[ResolutionError] = field(default_factory=list)
    meta: ResolutionMeta = field(default_factory=lambda: ResolutionMeta(provider="unknown"))
