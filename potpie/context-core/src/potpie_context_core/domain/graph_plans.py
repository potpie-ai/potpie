"""Graph V2 mutation-plan DTOs and serializers.

The plan store persists a server-created, validated mutation plan so
``graph commit`` can apply exactly that lowered batch by id. The agent never
resends the mutation payload at commit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

from potpie_context_core.domain.context_events import EventRef
from potpie_context_core.domain.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION
from potpie_context_core.domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_core.domain.llm_reconciliation import EvidenceRef
from potpie_context_core.domain.reconciliation import MutationBatch


class GraphMutationPlanStatus(StrEnum):
    validated = "validated"
    invalid = "invalid"
    conflict = "conflict"
    review_required = "review_required"
    approved = "approved"
    committed = "committed"
    expired = "expired"
    abandoned = "abandoned"
    error = "error"


TERMINAL_PLAN_STATUSES: frozenset[str] = frozenset(
    {
        GraphMutationPlanStatus.invalid.value,
        GraphMutationPlanStatus.conflict.value,
        GraphMutationPlanStatus.committed.value,
        GraphMutationPlanStatus.expired.value,
        GraphMutationPlanStatus.abandoned.value,
    }
)


@dataclass(frozen=True, slots=True)
class GraphMutationDiff:
    entity_upserts: int = 0
    edge_upserts: int = 0
    edge_deletes: int = 0
    invalidations: int = 0
    claim_keys: tuple[str, ...] = ()

    @classmethod
    def from_batch(
        cls, batch: MutationBatch | None, *, claim_keys: tuple[str, ...] = ()
    ) -> "GraphMutationDiff":
        if batch is None:
            return cls(claim_keys=claim_keys)
        return cls(
            entity_upserts=len(batch.entity_upserts),
            edge_upserts=len(batch.edge_upserts),
            edge_deletes=len(batch.edge_deletes),
            invalidations=len(batch.invalidations),
            claim_keys=claim_keys,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_upserts": self.entity_upserts,
            "edge_upserts": self.edge_upserts,
            "edge_deletes": self.edge_deletes,
            "invalidations": self.invalidations,
            "claims_asserted": len(self.claim_keys),
            "claims_retracted": self.invalidations,
            "claim_keys": list(self.claim_keys),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "GraphMutationDiff":
        data = raw or {}
        return cls(
            entity_upserts=_int(data.get("entity_upserts")),
            edge_upserts=_int(data.get("edge_upserts")),
            edge_deletes=_int(data.get("edge_deletes")),
            invalidations=_int(data.get("invalidations")),
            claim_keys=tuple(str(k) for k in data.get("claim_keys") or ()),
        )


@dataclass(frozen=True, slots=True)
class GraphMutationApproval:
    approved_by: str
    approved_at: datetime
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat(),
        }
        if self.reason:
            out["reason"] = self.reason
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "GraphMutationApproval | None":
        if not raw:
            return None
        approved_by = str(raw.get("approved_by") or "").strip()
        approved_at = _parse_datetime(raw.get("approved_at"))
        if not approved_by or approved_at is None:
            return None
        reason = raw.get("reason")
        return cls(
            approved_by=approved_by,
            approved_at=approved_at,
            reason=str(reason) if reason else None,
        )


@dataclass(frozen=True, slots=True)
class GraphMutationPlanRecord:
    plan_id: str
    pot_id: str
    status: str
    risk: str
    created_at: datetime
    expires_at: datetime
    original_payload: Mapping[str, Any] = field(default_factory=dict)
    validation_issues: tuple[Mapping[str, Any], ...] = ()
    accepted_ops: tuple[Mapping[str, Any], ...] = ()
    review_required_ops: tuple[Mapping[str, Any], ...] = ()
    rejected_ops: tuple[Mapping[str, Any], ...] = ()
    lowered_batch: MutationBatch | None = None
    provenance: ProvenanceContext | None = None
    expected_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    current_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    diff: GraphMutationDiff = field(default_factory=GraphMutationDiff)
    warnings: tuple[str, ...] = ()
    approval: GraphMutationApproval | None = None
    mutation_id: str | None = None
    committed_at: datetime | None = None
    final_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    detail: str | None = None
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "plan_id": self.plan_id,
            "pot_id": self.pot_id,
            "status": self.status,
            "risk": self.risk,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "original_payload": _json_safe(self.original_payload),
            "validation_issues": [_json_safe(i) for i in self.validation_issues],
            "accepted_ops": [_json_safe(op) for op in self.accepted_ops],
            "review_required_ops": [_json_safe(op) for op in self.review_required_ops],
            "rejected_ops": [_json_safe(op) for op in self.rejected_ops],
            "lowered_batch": mutation_batch_to_dict(self.lowered_batch)
            if self.lowered_batch is not None
            else None,
            "provenance": provenance_context_to_dict(self.provenance)
            if self.provenance is not None
            else None,
            "expected_subgraph_versions": dict(self.expected_subgraph_versions),
            "current_subgraph_versions": dict(self.current_subgraph_versions),
            "subgraph_versions": dict(self.current_subgraph_versions),
            "diff": self.diff.to_dict(),
            "warnings": list(self.warnings),
            "approval": self.approval.to_dict() if self.approval else None,
            "mutation_id": self.mutation_id,
            "committed_at": self.committed_at.isoformat()
            if self.committed_at is not None
            else None,
            "final_subgraph_versions": dict(self.final_subgraph_versions),
            "detail": self.detail,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphMutationPlanRecord":
        return cls(
            plan_id=str(raw.get("plan_id") or ""),
            pot_id=str(raw.get("pot_id") or ""),
            status=str(raw.get("status") or GraphMutationPlanStatus.invalid.value),
            risk=str(raw.get("risk") or "low"),
            created_at=_parse_datetime(raw.get("created_at"))
            or datetime.now(timezone.utc),
            expires_at=_parse_datetime(raw.get("expires_at"))
            or datetime.now(timezone.utc),
            original_payload=dict(raw.get("original_payload") or {}),
            validation_issues=tuple(
                dict(i) for i in raw.get("validation_issues") or ()
            ),
            accepted_ops=tuple(dict(i) for i in raw.get("accepted_ops") or ()),
            review_required_ops=tuple(
                dict(i) for i in raw.get("review_required_ops") or ()
            ),
            rejected_ops=tuple(dict(i) for i in raw.get("rejected_ops") or ()),
            lowered_batch=mutation_batch_from_dict(raw.get("lowered_batch")),
            provenance=provenance_context_from_dict(raw.get("provenance")),
            expected_subgraph_versions=_int_mapping(
                raw.get("expected_subgraph_versions")
            ),
            current_subgraph_versions=_int_mapping(
                raw.get("current_subgraph_versions")
                # ``to_dict`` also emits the ``subgraph_versions`` alias; honor it
                # so alias-only persisted payloads don't deserialize to empty.
                or raw.get("subgraph_versions")
            ),
            diff=GraphMutationDiff.from_dict(raw.get("diff")),
            warnings=tuple(str(w) for w in raw.get("warnings") or ()),
            approval=GraphMutationApproval.from_dict(raw.get("approval")),
            mutation_id=str(raw.get("mutation_id") or "") or None,
            committed_at=_parse_datetime(raw.get("committed_at")),
            final_subgraph_versions=_int_mapping(raw.get("final_subgraph_versions")),
            detail=str(raw.get("detail") or "") or None,
            graph_contract_version=str(
                raw.get("graph_contract_version") or GRAPH_CONTRACT_VERSION
            ),
            ontology_version=str(raw.get("ontology_version") or ONTOLOGY_VERSION),
        )

    def is_expired(self, *, now: datetime | None = None) -> bool:
        probe = now or datetime.now(timezone.utc)
        return probe >= self.expires_at


@dataclass(frozen=True, slots=True)
class GraphMutationProposal:
    ok: bool
    plan_id: str
    status: str
    risk: str
    pot_id: str
    auto_applicable: bool
    expires_at: datetime
    expected_subgraph_versions: Mapping[str, int]
    current_subgraph_versions: Mapping[str, int]
    diff: GraphMutationDiff
    warnings: tuple[str, ...] = ()
    issues: tuple[Mapping[str, Any], ...] = ()
    rejected_operations: tuple[Mapping[str, Any], ...] = ()
    review_required_operations: tuple[Mapping[str, Any], ...] = ()
    claim_keys: tuple[str, ...] = ()
    recommended_next_action: str | None = None
    detail: str | None = None
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "plan_id": self.plan_id,
            "status": self.status,
            "risk": self.risk,
            "pot_id": self.pot_id,
            "auto_applicable": self.auto_applicable,
            "expires_at": self.expires_at.isoformat(),
            "expected_subgraph_versions": dict(self.expected_subgraph_versions),
            "current_subgraph_versions": dict(self.current_subgraph_versions),
            "diff": self.diff.to_dict(),
            "warnings": list(self.warnings),
            "issues": [_json_safe(i) for i in self.issues],
            "rejected_operations": [_json_safe(op) for op in self.rejected_operations],
            "review_required_operations": [
                _json_safe(op) for op in self.review_required_operations
            ],
            "claim_keys": list(self.claim_keys),
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }
        if self.recommended_next_action:
            out["recommended_next_action"] = self.recommended_next_action
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True, slots=True)
class GraphIngestionVerificationResult:
    """Post-commit verification for source-backed graph ingestion."""

    ok: bool
    status: str
    plan_id: str
    pot_id: str
    claim_keys: tuple[str, ...] = ()
    readback_claim_keys: tuple[str, ...] = ()
    missing_claim_keys: tuple[str, ...] = ()
    readback_count: int = 0
    quality_status: str | None = None
    quality_counts: Mapping[str, int] = field(default_factory=dict)
    quality_delta: Mapping[str, int] = field(default_factory=dict)
    quality_regressions: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    checked_reports: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported: tuple[Mapping[str, Any], ...] = ()
    detail: str | None = None
    recommended_next_action: str | None = None
    subgraph_versions: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "plan_id": self.plan_id,
            "pot_id": self.pot_id,
            "claim_keys": list(self.claim_keys),
            "readback_claim_keys": list(self.readback_claim_keys),
            "missing_claim_keys": list(self.missing_claim_keys),
            "readback_count": self.readback_count,
            "quality_status": self.quality_status,
            "quality_counts": dict(self.quality_counts),
            "quality_delta": dict(self.quality_delta),
            "quality_regressions": {
                key: dict(value) for key, value in self.quality_regressions.items()
            },
            "checked_reports": list(self.checked_reports),
            "warnings": list(self.warnings),
            "unsupported": [dict(item) for item in self.unsupported],
            "subgraph_versions": dict(self.subgraph_versions),
        }
        if self.detail:
            out["detail"] = self.detail
        if self.recommended_next_action:
            out["recommended_next_action"] = self.recommended_next_action
        return out


@dataclass(frozen=True, slots=True)
class GraphMutationCommitResult:
    ok: bool
    plan_id: str
    status: str
    risk: str
    pot_id: str
    mutation_id: str | None = None
    applied_at: datetime | None = None
    expected_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    current_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    new_subgraph_versions: Mapping[str, int] = field(default_factory=dict)
    diff: GraphMutationDiff = field(default_factory=GraphMutationDiff)
    claim_keys: tuple[str, ...] = ()
    approval: GraphMutationApproval | None = None
    verification: GraphIngestionVerificationResult | None = None
    detail: str | None = None
    recommended_next_action: str | None = None
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "plan_id": self.plan_id,
            "status": self.status,
            "risk": self.risk,
            "pot_id": self.pot_id,
            "mutation_id": self.mutation_id,
            "applied_at": self.applied_at.isoformat()
            if self.applied_at is not None
            else None,
            "expected_subgraph_versions": dict(self.expected_subgraph_versions),
            "current_subgraph_versions": dict(self.current_subgraph_versions),
            "new_subgraph_versions": dict(self.new_subgraph_versions),
            "subgraph_versions": dict(
                self.new_subgraph_versions or self.current_subgraph_versions
            ),
            "diff": self.diff.to_dict(),
            "claim_keys": list(self.claim_keys),
            "approval": self.approval.to_dict() if self.approval else None,
            "verification": self.verification.to_dict()
            if self.verification is not None
            else None,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
        }
        if self.mutation_id:
            out["history_pointer"] = {"mutation_id": self.mutation_id}
            out["audit_ref"] = f"audit:mutation:{self.mutation_id}"
        if self.detail:
            out["detail"] = self.detail
        if self.recommended_next_action:
            out["recommended_next_action"] = self.recommended_next_action
        return out


def mutation_batch_to_dict(batch: MutationBatch) -> dict[str, Any]:
    return {
        "event_ref": event_ref_to_dict(batch.event_ref),
        "summary": batch.summary,
        "entity_upserts": [
            {
                "entity_key": item.entity_key,
                "labels": list(item.labels),
                "properties": _json_safe(item.properties),
            }
            for item in batch.entity_upserts
        ],
        "edge_upserts": [
            {
                "edge_type": item.edge_type,
                "from_entity_key": item.from_entity_key,
                "to_entity_key": item.to_entity_key,
                "properties": _json_safe(item.properties),
            }
            for item in batch.edge_upserts
        ],
        "edge_deletes": [
            {
                "edge_type": item.edge_type,
                "from_entity_key": item.from_entity_key,
                "to_entity_key": item.to_entity_key,
            }
            for item in batch.edge_deletes
        ],
        "invalidations": [
            {
                "target_entity_key": item.target_entity_key,
                "target_edge": list(item.target_edge) if item.target_edge else None,
                "reason": item.reason,
                "superseded_by_key": item.superseded_by_key,
                "valid_to": item.valid_to,
            }
            for item in batch.invalidations
        ],
        "evidence": [
            {"kind": item.kind, "ref": item.ref, "metadata": _json_safe(item.metadata)}
            for item in batch.evidence
        ],
        "confidence": batch.confidence,
        "warnings": list(batch.warnings),
        "ontology_downgrades": [_json_safe(item) for item in batch.ontology_downgrades],
    }


def mutation_batch_from_dict(raw: Mapping[str, Any] | None) -> MutationBatch | None:
    if not raw:
        return None
    return MutationBatch(
        event_ref=event_ref_from_dict(raw.get("event_ref")),
        summary=str(raw.get("summary") or ""),
        entity_upserts=[
            EntityUpsert(
                entity_key=str(item.get("entity_key") or ""),
                labels=tuple(str(label) for label in item.get("labels") or ()),
                properties=dict(item.get("properties") or {}),
            )
            for item in raw.get("entity_upserts") or ()
            if isinstance(item, Mapping)
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type=str(item.get("edge_type") or ""),
                from_entity_key=str(item.get("from_entity_key") or ""),
                to_entity_key=str(item.get("to_entity_key") or ""),
                properties=dict(item.get("properties") or {}),
            )
            for item in raw.get("edge_upserts") or ()
            if isinstance(item, Mapping)
        ],
        edge_deletes=[
            EdgeDelete(
                edge_type=str(item.get("edge_type") or ""),
                from_entity_key=str(item.get("from_entity_key") or ""),
                to_entity_key=str(item.get("to_entity_key") or ""),
            )
            for item in raw.get("edge_deletes") or ()
            if isinstance(item, Mapping)
        ],
        invalidations=[
            InvalidationOp(
                target_entity_key=str(item.get("target_entity_key") or "") or None,
                target_edge=tuple(item.get("target_edge"))
                if item.get("target_edge")
                else None,
                reason=str(item.get("reason") or ""),
                superseded_by_key=str(item.get("superseded_by_key") or "") or None,
                valid_to=str(item.get("valid_to") or "") or None,
            )
            for item in raw.get("invalidations") or ()
            if isinstance(item, Mapping)
        ],
        evidence=[
            EvidenceRef(
                kind=str(item.get("kind") or ""),
                ref=str(item.get("ref") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw.get("evidence") or ()
            if isinstance(item, Mapping)
        ],
        confidence=_float_or_none(raw.get("confidence")),
        warnings=[str(item) for item in raw.get("warnings") or ()],
        ontology_downgrades=[
            dict(item)
            for item in raw.get("ontology_downgrades") or ()
            if isinstance(item, Mapping)
        ],
    )


def provenance_context_to_dict(ctx: ProvenanceContext) -> dict[str, Any]:
    return {
        "source_event_id": ctx.source_event_id,
        "source_system": ctx.source_system,
        "source_kind": ctx.source_kind,
        "source_ref": ctx.source_ref,
        "event_occurred_at": ctx.event_occurred_at.isoformat()
        if ctx.event_occurred_at
        else None,
        "event_received_at": ctx.event_received_at.isoformat()
        if ctx.event_received_at
        else None,
        "created_by_agent": ctx.created_by_agent,
        "reconciliation_run_id": ctx.reconciliation_run_id,
        "actor_user_id": ctx.actor_user_id,
        "actor_surface": ctx.actor_surface,
        "actor_client_name": ctx.actor_client_name,
        "actor_auth_method": ctx.actor_auth_method,
    }


def provenance_context_from_dict(
    raw: Mapping[str, Any] | None,
) -> ProvenanceContext | None:
    if not raw:
        return None
    return ProvenanceContext(
        source_event_id=str(raw.get("source_event_id") or "") or None,
        source_system=str(raw.get("source_system") or "") or None,
        source_kind=str(raw.get("source_kind") or "") or None,
        source_ref=str(raw.get("source_ref") or "") or None,
        event_occurred_at=_parse_datetime(raw.get("event_occurred_at")),
        event_received_at=_parse_datetime(raw.get("event_received_at")),
        created_by_agent=str(raw.get("created_by_agent") or "") or None,
        reconciliation_run_id=str(raw.get("reconciliation_run_id") or "") or None,
        actor_user_id=str(raw.get("actor_user_id") or "") or None,
        actor_surface=str(raw.get("actor_surface") or "") or None,
        actor_client_name=str(raw.get("actor_client_name") or "") or None,
        actor_auth_method=str(raw.get("actor_auth_method") or "") or None,
    )


def event_ref_to_dict(event_ref: EventRef | None) -> dict[str, Any] | None:
    if event_ref is None:
        return None
    return {
        "event_id": event_ref.event_id,
        "source_system": event_ref.source_system,
        "pot_id": event_ref.pot_id,
    }


def event_ref_from_dict(raw: Mapping[str, Any] | None) -> EventRef | None:
    if not raw:
        return None
    event_id = str(raw.get("event_id") or "")
    source_system = str(raw.get("source_system") or "")
    pot_id = str(raw.get("pot_id") or "")
    if not event_id or not source_system or not pot_id:
        return None
    return EventRef(event_id=event_id, source_system=source_system, pot_id=pot_id)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "GraphIngestionVerificationResult",
    "GraphMutationApproval",
    "GraphMutationCommitResult",
    "GraphMutationDiff",
    "GraphMutationPlanRecord",
    "GraphMutationPlanStatus",
    "GraphMutationProposal",
    "TERMINAL_PLAN_STATUSES",
    "event_ref_from_dict",
    "event_ref_to_dict",
    "mutation_batch_from_dict",
    "mutation_batch_to_dict",
    "provenance_context_from_dict",
    "provenance_context_to_dict",
]
