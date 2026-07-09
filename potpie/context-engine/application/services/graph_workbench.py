"""Shared Graph V2 workbench envelope assembly."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any

from application.services.semantic_mutation_lowering import lower_semantic_request
from application.services.semantic_mutation_validator import validate_semantic_request
from domain.errors import CapabilityNotImplemented
from domain.graph_contract import GRAPH_CONTRACT_VERSION as DATA_PLANE_CONTRACT_VERSION
from domain.graph_contract import EVIDENCE_REQUIRED_TRUTH_CLASSES, MutationRisk
from domain.graph_history import (
    GraphHistoryEntry,
    GraphHistoryRequest,
    GraphHistoryResult,
)
from domain.graph_inbox import (
    TERMINAL_INBOX_STATUSES,
    GraphInboxItem,
    GraphInboxResult,
    GraphInboxStatus,
)
from domain.graph_mutations import ProvenanceContext
from domain.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationApproval,
    GraphMutationCommitResult,
    GraphMutationDiff,
    GraphMutationPlanRecord,
    GraphMutationPlanStatus,
    GraphMutationProposal,
    TERMINAL_PLAN_STATUSES,
)
from domain.graph_quality import (
    EpisodicEdgeConflictInput,
    GraphQualityFinding,
    GraphQualityResult,
    detect_family_conflicts,
)
from domain.graph_workbench import (
    GRAPH_WORKBENCH_ADMIN_COMMANDS,
    GRAPH_WORKBENCH_COMMANDS,
    GRAPH_WORKBENCH_CONTRACT_VERSION,
    GRAPH_WORKBENCH_LEGACY_COMMANDS,
    GraphCommandEnvelope,
    GraphCommandError,
    GraphUnsupported,
    GraphUnsupportedResult,
)
from domain.graph_workbench_ontology import ranked_catalog_views
from domain.ontology import EDGE_TYPES
from domain.ports.graph.backend import GraphBackend
from domain.ports.graph.inbox_store import GraphInboxStorePort
from domain.ports.graph.plan_store import GraphPlanStorePort
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.semantic_mutations import (
    LoweredOperation,
    SemanticMutationParseError,
    SemanticMutationRequest,
    SemanticMutationValidationIssue,
)
from domain.singleton_predicates import is_singleton_predicate

_DEFAULT_PLAN_TTL_SECONDS = 3600
_QUALITY_SCAN_MAX = 5000
_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5
_QUALITY_SUMMARY_REPORTS = (
    "duplicate-candidates",
    "stale-facts",
    "conflicting-claims",
    "orphan-entities",
    "low-confidence",
    "projection-drift",
)

_CROSS_CUTTING_RESULT_KEYS = frozenset(
    {
        "ok",
        "graph_contract_version",
        "ontology_version",
        "subgraph_versions",
        "pot_id",
        "warnings",
        "unsupported",
        "recommended_next_action",
    }
)


class GraphWorkbenchService:
    """Graph V2 workbench workflow layer above the backend write door."""

    def __init__(
        self,
        *,
        backend: GraphBackend,
        plan_store: GraphPlanStorePort,
        inbox_store: GraphInboxStorePort | None = None,
        default_plan_ttl_seconds: int = _DEFAULT_PLAN_TTL_SECONDS,
    ) -> None:
        self.backend = backend
        self.plan_store = plan_store
        self.inbox_store = inbox_store
        self.default_plan_ttl_seconds = default_plan_ttl_seconds

    def propose(
        self,
        payload: Mapping[str, Any],
        *,
        pot_id: str,
        ttl_seconds: int | None = None,
    ) -> GraphMutationProposal:
        """Validate, lower, diff, and persist a mutation plan without writing."""
        now = datetime.now(timezone.utc)
        plan_id = f"mutation-plan:{uuid.uuid4().hex[:12]}"
        current_versions = _subgraph_versions(self.backend, pot_id)
        expected_versions = _expected_versions(payload, current_versions)
        conflict = _version_conflict(expected_versions, current_versions)
        expires_at = now + timedelta(
            seconds=max(1, int(ttl_seconds or self.default_plan_ttl_seconds))
        )
        payload_with_pot = dict(payload)
        payload_with_pot["pot_id"] = pot_id

        try:
            request = SemanticMutationRequest.parse(payload, pot_id=pot_id)
        except SemanticMutationParseError as exc:
            issue = {
                "code": "invalid_mutation_payload",
                "message": str(exc),
                "severity": "error",
                "op_index": None,
            }
            record = GraphMutationPlanRecord(
                plan_id=plan_id,
                pot_id=pot_id,
                status=GraphMutationPlanStatus.invalid.value,
                risk=MutationRisk.low.value,
                created_at=now,
                expires_at=expires_at,
                original_payload=payload_with_pot,
                validation_issues=(issue,),
                rejected_ops=(issue,),
                expected_subgraph_versions=expected_versions,
                current_subgraph_versions=current_versions,
                detail=str(exc),
            )
            self.plan_store.save(record)
            return _proposal_from_record(
                record,
                ok=False,
                recommended_next_action="Fix the mutation JSON and run graph propose again.",
            )

        semantic_plan = validate_semantic_request(request)
        status = _status_for_proposal(semantic_plan, conflict=bool(conflict))
        if status not in {
            GraphMutationPlanStatus.invalid.value,
            GraphMutationPlanStatus.conflict.value,
        }:
            lower_semantic_request(request, semantic_plan)

        claim_keys = tuple(
            key for op in semantic_plan.accepted_ops for key in op.claim_keys
        )
        diff = GraphMutationDiff.from_batch(
            semantic_plan.batch,
            claim_keys=claim_keys,
        )
        issues = tuple(issue.to_dict() for issue in semantic_plan.issues)
        rejected = _rejected_operation_summaries(semantic_plan.issues)
        warnings = tuple(
            issue.message for issue in semantic_plan.issues if not issue.is_error
        ) + tuple(semantic_plan.warnings)
        detail = None
        recommended = None
        if conflict:
            detail = _conflict_message(conflict)
            recommended = "Reread the affected graph views and propose a new plan."
        elif status == GraphMutationPlanStatus.invalid.value:
            detail = "; ".join(i.message for i in semantic_plan.errors) or None
            recommended = "Fix the validation errors and run graph propose again."
        elif status == GraphMutationPlanStatus.review_required.value:
            recommended = "Review the persisted plan, then commit with --approved-by when policy allows."
        elif status == GraphMutationPlanStatus.validated.value:
            recommended = f"Commit with `potpie graph commit {plan_id} --json`."

        record = GraphMutationPlanRecord(
            plan_id=plan_id,
            pot_id=pot_id,
            status=status,
            risk=semantic_plan.risk,
            created_at=now,
            expires_at=expires_at,
            original_payload=payload_with_pot,
            validation_issues=issues,
            accepted_ops=tuple(
                _lowered_operation_summary(op) for op in semantic_plan.accepted_ops
            ),
            review_required_ops=tuple(
                _lowered_operation_summary(op)
                for op in semantic_plan.review_required_ops
            ),
            rejected_ops=rejected,
            lowered_batch=semantic_plan.batch,
            provenance=semantic_plan.provenance,
            expected_subgraph_versions=expected_versions,
            current_subgraph_versions=current_versions,
            diff=diff,
            warnings=warnings,
            detail=detail,
        )
        self.plan_store.save(record)
        return _proposal_from_record(
            record,
            ok=status
            in {
                GraphMutationPlanStatus.validated.value,
                GraphMutationPlanStatus.review_required.value,
            },
            recommended_next_action=recommended,
        )

    def commit(
        self,
        plan_id: str,
        *,
        pot_id: str,
        approved_by: str | None = None,
        verify: bool = False,
    ) -> GraphMutationCommitResult:
        """Apply an unexpired server-created plan by id."""
        now = datetime.now(timezone.utc)
        record = self.plan_store.get(pot_id=pot_id, plan_id=plan_id)
        if record is None:
            return GraphMutationCommitResult(
                ok=False,
                plan_id=plan_id,
                status="not_found",
                risk=MutationRisk.low.value,
                pot_id=pot_id,
                detail=f"mutation plan {plan_id!r} was not found for this pot",
                recommended_next_action="Run graph propose and commit the returned plan_id.",
            )

        if record.status in TERMINAL_PLAN_STATUSES:
            return GraphMutationCommitResult(
                ok=False,
                plan_id=record.plan_id,
                status=record.status,
                risk=record.risk,
                pot_id=record.pot_id,
                expected_subgraph_versions=record.expected_subgraph_versions,
                current_subgraph_versions=record.current_subgraph_versions,
                diff=record.diff,
                claim_keys=_claim_keys_from_record(record),
                approval=record.approval,
                detail=f"plan is {record.status} and cannot be committed",
                recommended_next_action="Create a fresh proposal if a write is still needed.",
            )

        if record.is_expired(now=now):
            expired = replace(
                record,
                status=GraphMutationPlanStatus.expired.value,
                detail="plan expired before commit",
            )
            self.plan_store.save(expired)
            return GraphMutationCommitResult(
                ok=False,
                plan_id=expired.plan_id,
                status=expired.status,
                risk=expired.risk,
                pot_id=expired.pot_id,
                expected_subgraph_versions=expired.expected_subgraph_versions,
                current_subgraph_versions=_subgraph_versions(self.backend, pot_id),
                diff=expired.diff,
                claim_keys=_claim_keys_from_record(expired),
                detail="plan expired before commit",
                recommended_next_action="Reread current graph state and propose a new plan.",
            )

        current_versions = _subgraph_versions(self.backend, pot_id)
        conflict = _version_conflict(
            record.expected_subgraph_versions, current_versions
        )
        if conflict:
            conflicted = replace(
                record,
                status=GraphMutationPlanStatus.conflict.value,
                current_subgraph_versions=current_versions,
                detail=_conflict_message(conflict),
            )
            self.plan_store.save(conflicted)
            return GraphMutationCommitResult(
                ok=False,
                plan_id=conflicted.plan_id,
                status=conflicted.status,
                risk=conflicted.risk,
                pot_id=conflicted.pot_id,
                expected_subgraph_versions=conflicted.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                diff=conflicted.diff,
                claim_keys=_claim_keys_from_record(conflicted),
                detail=conflicted.detail,
                recommended_next_action="Reread current graph state and propose a new plan.",
            )

        approval = record.approval
        approval_error = _approval_error(record, approved_by=approved_by)
        if approval_error:
            return GraphMutationCommitResult(
                ok=False,
                plan_id=record.plan_id,
                status=record.status,
                risk=record.risk,
                pot_id=record.pot_id,
                expected_subgraph_versions=record.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                diff=record.diff,
                claim_keys=_claim_keys_from_record(record),
                approval=approval,
                detail=approval_error,
                recommended_next_action=(
                    f"Review the plan, then run `potpie graph commit {plan_id} "
                    "--approved-by <user-ref> --json` when policy allows."
                ),
            )
        if approved_by and approval is None:
            approval = GraphMutationApproval(
                approved_by=approved_by,
                approved_at=now,
            )
            record = replace(
                record,
                status=GraphMutationPlanStatus.approved.value,
                approval=approval,
            )
            self.plan_store.save(record)

        quality_before = (
            _verification_quality_snapshot(self.backend, pot_id=pot_id)
            if verify
            else None
        )

        if record.lowered_batch is None or not _batch_has_work(record.lowered_batch):
            committed = replace(
                record,
                status=GraphMutationPlanStatus.committed.value,
                committed_at=now,
                final_subgraph_versions=current_versions,
                approval=approval,
            )
            self.plan_store.save(committed)
            verification = (
                _verify_ingestion_commit(
                    self.backend,
                    pot_id=pot_id,
                    record=committed,
                    before_quality=quality_before,
                )
                if verify
                else None
            )
            return GraphMutationCommitResult(
                ok=True,
                plan_id=committed.plan_id,
                status=committed.status,
                risk=committed.risk,
                pot_id=committed.pot_id,
                applied_at=now,
                expected_subgraph_versions=committed.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                new_subgraph_versions=current_versions,
                diff=committed.diff,
                claim_keys=_claim_keys_from_record(committed),
                approval=approval,
                verification=verification,
                detail="plan had no structural mutations to apply",
                recommended_next_action=_verification_next_action(verification),
            )

        result = self.backend.mutation.apply(
            record.lowered_batch,
            expected_pot_id=pot_id,
            provenance_context=_provenance_for_commit(
                record.provenance,
                approved_by=approved_by,
            ),
        )
        if not result.ok:
            errored = replace(
                record,
                status=GraphMutationPlanStatus.error.value,
                current_subgraph_versions=current_versions,
                approval=approval,
                detail=result.error,
            )
            self.plan_store.save(errored)
            return GraphMutationCommitResult(
                ok=False,
                plan_id=errored.plan_id,
                status=errored.status,
                risk=errored.risk,
                pot_id=errored.pot_id,
                expected_subgraph_versions=errored.expected_subgraph_versions,
                current_subgraph_versions=current_versions,
                diff=errored.diff,
                claim_keys=_claim_keys_from_record(errored),
                approval=approval,
                detail=result.error,
                recommended_next_action="Inspect backend readiness with `potpie graph status --json`.",
            )

        final_versions = _subgraph_versions(self.backend, pot_id)
        committed = replace(
            record,
            status=GraphMutationPlanStatus.committed.value,
            mutation_id=result.mutation_id,
            committed_at=now,
            final_subgraph_versions=final_versions,
            approval=approval,
        )
        self.plan_store.save(committed)
        verification = (
            _verify_ingestion_commit(
                self.backend,
                pot_id=pot_id,
                record=committed,
                before_quality=quality_before,
            )
            if verify
            else None
        )
        return GraphMutationCommitResult(
            ok=True,
            plan_id=committed.plan_id,
            status=committed.status,
            risk=committed.risk,
            pot_id=committed.pot_id,
            mutation_id=result.mutation_id,
            applied_at=now,
            expected_subgraph_versions=committed.expected_subgraph_versions,
            current_subgraph_versions=current_versions,
            new_subgraph_versions=final_versions,
            diff=committed.diff,
            claim_keys=_claim_keys_from_record(committed),
            approval=approval,
            verification=verification,
            recommended_next_action=_verification_next_action(verification),
        )

    def history(
        self,
        *,
        pot_id: str,
        entity_key: str | None = None,
        claim_key: str | None = None,
        subgraph: str | None = None,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> GraphHistoryResult:
        """Return read-only plan and claim history for a bounded filter."""
        request = GraphHistoryRequest(
            pot_id=pot_id,
            entity_key=_clean_str(entity_key),
            claim_key=_clean_str(claim_key),
            subgraph=_clean_str(subgraph),
            plan_id=_clean_str(plan_id),
            mutation_id=_clean_str(mutation_id),
            since=since,
            until=until,
            limit=max(1, min(int(limit or 50), 500)),
        )
        records = self.plan_store.list(
            pot_id=pot_id,
            plan_id=request.plan_id,
            mutation_id=request.mutation_id,
            since=since,
            until=until,
            limit=request.limit,
        )
        records = tuple(
            record
            for record in records
            if _record_matches_history_filter(record, request)
        )
        entries = [_history_entry_from_plan(record) for record in records]

        rows, unsupported = _history_claim_rows(self.backend, request, records)
        entries.extend(_history_entry_from_claim(row) for row in rows)
        entries = _dedupe_history_entries(entries)
        entries.sort(key=_history_sort_key, reverse=True)
        entries = entries[: request.limit]

        detail = None
        recommended = None
        if not entries and unsupported:
            detail = "claim history is unavailable for the active backend"
            recommended = "Use graph status to inspect backend capabilities."
        elif not entries:
            detail = "no graph history matched the supplied filters"
            recommended = "Relax the history filters or inspect current graph reads."

        return GraphHistoryResult(
            ok=True,
            pot_id=pot_id,
            filters=request.filters(),
            entries=tuple(entries),
            unsupported=unsupported,
            detail=detail,
            recommended_next_action=recommended,
            subgraph_versions=_subgraph_versions(self.backend, pot_id),
        )

    def quality(
        self,
        *,
        pot_id: str,
        report: str,
        subgraph: str | None = None,
        limit: int = 50,
        confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    ) -> GraphQualityResult:
        """Return bounded read-only graph quality findings."""
        clean_report = _normalize_quality_report(report)
        clean_limit = max(1, min(int(limit or 50), 500))
        filters = _quality_filters(
            report=clean_report,
            subgraph=subgraph,
            limit=clean_limit,
            confidence_threshold=confidence_threshold,
        )
        if clean_report == "summary":
            return _quality_summary_result(
                self.backend,
                pot_id=pot_id,
                filters=filters,
            )

        findings, metrics, unsupported = _quality_deep_report(
            self.backend,
            pot_id=pot_id,
            report=clean_report,
            subgraph=subgraph,
            limit=clean_limit,
            confidence_threshold=confidence_threshold,
        )

        status = _quality_status(findings, unsupported)
        detail = None
        recommended = None
        if unsupported and not findings:
            detail = f"{clean_report} is unavailable for the active backend"
            recommended = "Use graph status to inspect backend capabilities."
        elif findings:
            recommended = (
                "Review findings, then use graph propose/commit for semantic "
                "fact corrections or graph inbox add for uncertain work."
            )
        return GraphQualityResult(
            ok=True,
            pot_id=pot_id,
            report=clean_report,
            status=status,
            findings=findings,
            metrics=metrics,
            filters=filters,
            unsupported=unsupported,
            detail=detail,
            recommended_next_action=recommended,
            subgraph_versions=_subgraph_versions(self.backend, pot_id),
        )

    def inbox_add(
        self,
        *,
        pot_id: str,
        summary: str,
        details: str | None = None,
        evidence: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        suspected_subgraphs: tuple[str, ...] = (),
        created_by: Mapping[str, Any] | None = None,
    ) -> GraphInboxResult:
        """Persist a pending graph-work item without writing graph facts."""
        store = self._inbox_store()
        now = datetime.now(timezone.utc)
        item = GraphInboxItem(
            item_id=f"graph-inbox:{uuid.uuid4().hex[:12]}",
            pot_id=pot_id,
            status=GraphInboxStatus.pending.value,
            summary=_required_clean(summary, "summary"),
            details=_clean_str(details),
            evidence=_clean_tuple(evidence),
            source_refs=_clean_tuple(source_refs),
            suspected_subgraphs=_clean_tuple(suspected_subgraphs),
            created_by=dict(created_by or {"surface": "cli"}),
            created_at=now,
        )
        store.save(item)
        return GraphInboxResult(
            ok=True,
            pot_id=pot_id,
            action="add",
            item=item,
            recommended_next_action=(
                "Process with graph catalog/describe/read/search-entities, then "
                "graph propose/commit, and close the inbox item."
            ),
        )

    def inbox_list(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> GraphInboxResult:
        """Return a bounded inbox worklist."""
        store = self._inbox_store()
        statuses = _normalize_inbox_status_filter(status)
        clean_limit = max(1, min(int(limit or 50), 500))
        items = store.list(
            pot_id=pot_id,
            status=statuses,
            claimed_by=_clean_str(claimed_by),
            suspected_subgraph=_clean_str(suspected_subgraph),
            source_ref=_clean_str(source_ref),
            since=since,
            until=until,
            limit=clean_limit,
        )
        filters: dict[str, Any] = {"limit": clean_limit}
        if statuses:
            filters["status"] = list(statuses)
        for key, value in (
            ("claimed_by", claimed_by),
            ("suspected_subgraph", suspected_subgraph),
            ("source_ref", source_ref),
        ):
            clean = _clean_str(value)
            if clean:
                filters[key] = clean
        if since:
            filters["since"] = since.isoformat()
        if until:
            filters["until"] = until.isoformat()
        return GraphInboxResult(
            ok=True,
            pot_id=pot_id,
            action="list",
            items=items,
            filters=filters,
        )

    def inbox_show(self, *, pot_id: str, item_id: str) -> GraphInboxResult:
        """Return one inbox item."""
        item = self._inbox_store().get(
            pot_id=pot_id,
            item_id=_required_clean(item_id, "item_id"),
        )
        if item is None:
            return _inbox_missing_result(pot_id=pot_id, action="show", item_id=item_id)
        return GraphInboxResult(ok=True, pot_id=pot_id, action="show", item=item)

    def inbox_claim(
        self,
        *,
        pot_id: str,
        item_id: str,
        claimed_by: str,
    ) -> GraphInboxResult:
        """Claim a pending inbox item for processing."""
        store = self._inbox_store()
        item = store.get(
            pot_id=pot_id,
            item_id=_required_clean(item_id, "item_id"),
        )
        if item is None:
            return _inbox_missing_result(pot_id=pot_id, action="claim", item_id=item_id)
        terminal = _terminal_inbox_result(item, action="claim")
        if terminal is not None:
            return terminal
        now = datetime.now(timezone.utc)
        claimed = replace(
            item,
            status=GraphInboxStatus.claimed.value,
            claimed_by=_required_clean(claimed_by, "claimed_by"),
            claimed_at=now,
        )
        store.save(claimed)
        return GraphInboxResult(
            ok=True,
            pot_id=pot_id,
            action="claim",
            item=claimed,
            recommended_next_action=(
                "Use graph catalog/describe/read/search-entities before proposing a write."
            ),
        )

    def inbox_mark_applied(
        self,
        *,
        pot_id: str,
        item_id: str,
        closed_by: str,
        linked_plan_id: str | None = None,
        linked_mutation_id: str | None = None,
    ) -> GraphInboxResult:
        """Close an inbox item as applied through propose/commit."""
        plan_id = _clean_str(linked_plan_id)
        mutation_id = _clean_str(linked_mutation_id)
        if not (plan_id or mutation_id):
            raise ValueError("--plan or --mutation is required")
        return self._close_inbox_item(
            pot_id=pot_id,
            item_id=item_id,
            status=GraphInboxStatus.applied.value,
            closed_by=closed_by,
            linked_plan_id=plan_id,
            linked_mutation_id=mutation_id,
            rejection_reason=None,
            action="mark-applied",
        )

    def inbox_mark_rejected(
        self,
        *,
        pot_id: str,
        item_id: str,
        closed_by: str,
        rejection_reason: str,
    ) -> GraphInboxResult:
        """Close an inbox item as rejected with the review reason."""
        return self._close_inbox_item(
            pot_id=pot_id,
            item_id=item_id,
            status=GraphInboxStatus.rejected.value,
            closed_by=closed_by,
            linked_plan_id=None,
            linked_mutation_id=None,
            rejection_reason=_required_clean(rejection_reason, "rejection_reason"),
            action="mark-rejected",
        )

    def inbox_close(
        self,
        *,
        pot_id: str,
        item_id: str,
        closed_by: str,
        linked_plan_id: str | None = None,
        linked_mutation_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> GraphInboxResult:
        """Close an inbox item with a plan, mutation, or rejection reason."""
        plan_id = _clean_str(linked_plan_id)
        mutation_id = _clean_str(linked_mutation_id)
        reason = _clean_str(rejection_reason)
        if not (plan_id or mutation_id or reason):
            raise ValueError("--plan, --mutation, or --reason is required")
        return self._close_inbox_item(
            pot_id=pot_id,
            item_id=item_id,
            status=GraphInboxStatus.closed.value,
            closed_by=closed_by,
            linked_plan_id=plan_id,
            linked_mutation_id=mutation_id,
            rejection_reason=reason,
            action="close",
        )

    def _close_inbox_item(
        self,
        *,
        pot_id: str,
        item_id: str,
        status: str,
        closed_by: str,
        linked_plan_id: str | None,
        linked_mutation_id: str | None,
        rejection_reason: str | None,
        action: str,
    ) -> GraphInboxResult:
        store = self._inbox_store()
        item = store.get(
            pot_id=pot_id,
            item_id=_required_clean(item_id, "item_id"),
        )
        if item is None:
            return _inbox_missing_result(pot_id=pot_id, action=action, item_id=item_id)
        terminal = _terminal_inbox_result(item, action=action)
        if terminal is not None:
            return terminal
        closed = replace(
            item,
            status=status,
            closed_by=_required_clean(closed_by, "closed_by"),
            closed_at=datetime.now(timezone.utc),
            linked_plan_id=linked_plan_id,
            linked_mutation_id=linked_mutation_id,
            rejection_reason=rejection_reason,
        )
        store.save(closed)
        return GraphInboxResult(ok=True, pot_id=pot_id, action=action, item=closed)

    def _inbox_store(self) -> GraphInboxStorePort:
        if self.inbox_store is None:
            raise CapabilityNotImplemented(
                "graph.inbox.store",
                detail="graph inbox persistence is not configured",
                recommended_next_action=(
                    "Use a host profile that wires GraphInboxStorePort."
                ),
            )
        return self.inbox_store


def new_graph_request_id() -> str:
    return f"req:{uuid.uuid4().hex}"


def graph_success_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    result: Mapping[str, Any] | None = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=True,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=dict(result or {}),
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
    )


def graph_error_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    code: str,
    message: str,
    detail: Any = None,
    subgraph_versions: Mapping[str, int] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    unsupported: tuple[GraphUnsupported, ...] | list[GraphUnsupported] = (),
    recommended_next_action: str | Mapping[str, Any] | None = None,
) -> GraphCommandEnvelope:
    return GraphCommandEnvelope(
        ok=False,
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        result=None,
        subgraph_versions=dict(subgraph_versions or {}),
        warnings=tuple(warnings),
        unsupported=tuple(unsupported),
        recommended_next_action=recommended_next_action,
        error=GraphCommandError(code=code, message=message, detail=detail),
    )


def graph_not_implemented_envelope(
    *,
    command: str,
    request_id: str,
    pot_id: str | None,
    detail: str | None = None,
    recommended_next_action: str | None = None,
) -> GraphCommandEnvelope:
    message = f"{command} is not implemented yet"
    return graph_error_envelope(
        command=command,
        request_id=request_id,
        pot_id=pot_id,
        code="not_implemented",
        message=message,
        detail=detail,
        unsupported=(
            GraphUnsupported(
                name=command,
                reason="not_implemented",
                detail=detail,
            ),
        ),
        recommended_next_action=recommended_next_action,
    )


def graph_not_implemented_result(command: str, *, detail: str | None = None) -> dict:
    return GraphUnsupportedResult(
        status="not_implemented",
        command=command,
        detail=detail,
    ).to_dict()


def normalize_workbench_result(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, int], tuple[str, ...], tuple[GraphUnsupported, ...]
]:
    """Move cross-cutting fields from a legacy result into envelope fields."""
    result = dict(payload)
    subgraph_versions = _mapping_of_ints(result.pop("subgraph_versions", {}))
    warnings = _string_tuple(result.pop("warnings", ()))
    unsupported = _unsupported_tuple(result.pop("unsupported", ()))
    unsupported += _unsupported_tuple(result.pop("unsupported_includes", ()))
    for key in _CROSS_CUTTING_RESULT_KEYS:
        result.pop(key, None)
    return result, subgraph_versions, warnings, unsupported


def normalize_catalog_result(
    payload: Mapping[str, Any],
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """Project the V1.5 data-plane catalog as a V2 workbench catalog body."""
    result = dict(payload)
    data_plane_version = result.pop(
        "graph_contract_version", DATA_PLANE_CONTRACT_VERSION
    )
    result.pop("ontology_version", None)
    result.pop("ok", None)
    result["data_plane_graph_contract_version"] = data_plane_version
    result["workbench_graph_contract_version"] = GRAPH_WORKBENCH_CONTRACT_VERSION
    result["commands"] = list(GRAPH_WORKBENCH_COMMANDS)
    result["admin_commands"] = list(GRAPH_WORKBENCH_ADMIN_COMMANDS)
    result["legacy_commands"] = list(GRAPH_WORKBENCH_LEGACY_COMMANDS)
    views, ranking = ranked_catalog_views(
        result.get("views", ()),
        task,
    )
    result["views"] = views
    if task:
        result["task"] = task
        result["task_ranking"] = ranking
    result["transition"] = {
        "legacy_commands_callable": True,
        "mutate_replacement": "propose + commit",
        "inspect_replacement": "neighborhood",
    }
    return result


def _mapping_of_ints(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return out


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return (str(value),)


def _unsupported_tuple(value: Any) -> tuple[GraphUnsupported, ...]:
    if value is None:
        return ()
    if isinstance(value, GraphUnsupported):
        return (value,)
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return (GraphUnsupported(name=str(value), reason="unsupported"),)
    out: list[GraphUnsupported] = []
    for item in value:
        if isinstance(item, GraphUnsupported):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            out.append(
                GraphUnsupported(
                    name=str(item.get("name") or item.get("include") or "unknown"),
                    reason=str(item.get("reason") or item.get("code") or "unsupported"),
                    detail=item.get("detail"),
                )
            )
            continue
        out.append(GraphUnsupported(name=str(item), reason="unsupported"))
    return tuple(out)


def _subgraph_versions(backend: GraphBackend, pot_id: str) -> dict[str, int]:
    try:
        counts = dict(backend.analytics.counts(pot_id))
    except Exception:
        counts = {}
    try:
        return {"_global": int(counts.get("claims", 0))}
    except (TypeError, ValueError):
        return {"_global": 0}


def _expected_versions(
    payload: Mapping[str, Any],
    current_versions: Mapping[str, int],
) -> dict[str, int]:
    provided = _mapping_of_ints(payload.get("expected_subgraph_versions"))
    if "_global" not in provided and "_global" in current_versions:
        provided["_global"] = int(current_versions["_global"])
    return provided


def _version_conflict(
    expected: Mapping[str, int],
    current: Mapping[str, int],
) -> dict[str, Any] | None:
    for key, expected_value in expected.items():
        if key not in current:
            continue
        actual = int(current[key])
        if int(expected_value) != actual:
            return {
                "subgraph": key,
                "expected_version": int(expected_value),
                "actual_version": actual,
            }
    return None


def _conflict_message(conflict: Mapping[str, Any]) -> str:
    return (
        f"{conflict.get('subgraph')} changed after the plan was proposed "
        f"(expected {conflict.get('expected_version')}, "
        f"actual {conflict.get('actual_version')})"
    )


def _status_for_proposal(semantic_plan, *, conflict: bool) -> str:
    if conflict:
        return GraphMutationPlanStatus.conflict.value
    if semantic_plan.errors or semantic_plan.decision == "rejected":
        return GraphMutationPlanStatus.invalid.value
    if semantic_plan.decision == "review_required":
        return GraphMutationPlanStatus.review_required.value
    return GraphMutationPlanStatus.validated.value


def _lowered_operation_summary(op: LoweredOperation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "op_index": op.op_index,
        "op": op.op,
        "risk": op.risk,
        "status": op.status,
        "claim_keys": list(op.claim_keys),
    }
    if op.subgraph:
        out["subgraph"] = op.subgraph
    if op.truth:
        out["truth"] = op.truth
    return out


def _rejected_operation_summaries(
    issues: tuple[SemanticMutationValidationIssue, ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "op_index": issue.op_index,
            "code": issue.code,
            "message": issue.message,
        }
        for issue in issues
        if issue.is_error
    )


def _proposal_from_record(
    record: GraphMutationPlanRecord,
    *,
    ok: bool,
    recommended_next_action: str | None = None,
) -> GraphMutationProposal:
    claim_keys = _claim_keys_from_record(record)
    return GraphMutationProposal(
        ok=ok,
        plan_id=record.plan_id,
        status=record.status,
        risk=record.risk,
        pot_id=record.pot_id,
        auto_applicable=(
            record.status == GraphMutationPlanStatus.validated.value
            and record.risk == MutationRisk.low.value
        ),
        expires_at=record.expires_at,
        expected_subgraph_versions=record.expected_subgraph_versions,
        current_subgraph_versions=record.current_subgraph_versions,
        diff=record.diff,
        warnings=record.warnings,
        issues=record.validation_issues,
        rejected_operations=record.rejected_ops,
        review_required_operations=record.review_required_ops,
        claim_keys=claim_keys,
        recommended_next_action=recommended_next_action,
        detail=record.detail,
    )


def _claim_keys_from_record(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    keys: list[str] = []
    for op in record.accepted_ops:
        raw = op.get("claim_keys") if isinstance(op, Mapping) else None
        if isinstance(raw, (list, tuple)):
            keys.extend(str(key) for key in raw if key)
    return tuple(dict.fromkeys(keys))


def _verify_ingestion_commit(
    backend: GraphBackend,
    *,
    pot_id: str,
    record: GraphMutationPlanRecord,
    before_quality: Mapping[str, Any] | None,
) -> GraphIngestionVerificationResult:
    claim_keys = _claim_keys_from_record(record)
    readback = _verification_readback(backend, pot_id=pot_id, claim_keys=claim_keys)
    after_quality = _verification_quality_snapshot(backend, pot_id=pot_id)
    before_counts = _quality_count_map(before_quality)
    after_counts = _quality_count_map(after_quality)
    deltas = _quality_count_delta(before_counts, after_counts)
    regressions = _quality_regressions(
        before_quality=before_quality,
        after_quality=after_quality,
        deltas=deltas,
    )
    unsupported = tuple(readback["unsupported"]) + tuple(after_quality["unsupported"])
    warnings: list[str] = []
    status = "ok"
    ok = True
    detail = None
    recommended = None

    if readback["missing_claim_keys"]:
        ok = False
        status = "degraded"
        detail = "committed plan did not read back all expected claim keys"
        recommended = (
            "Inspect graph history for the plan and rerun graph commit verification "
            "after backend repair if the mutation was applied."
        )
    elif readback["unsupported"]:
        ok = False
        status = "partial"
        detail = "claim readback is unavailable for the active backend"
        recommended = "Use graph status to inspect backend claim-query support."
    elif regressions:
        ok = False
        status = "degraded"
        detail = "quality findings increased after commit"
        recommended = (
            "Run the affected graph quality reports, then correct facts with "
            "graph propose/commit or add a graph inbox item for uncertain work."
        )
    elif unsupported:
        status = "partial"
        warnings.append("some verification checks are unavailable for this backend")
        recommended = "Use graph status to inspect backend capabilities."
    elif after_quality["status"] not in {"ok", "empty"}:
        status = "watch"
        warnings.append(f"graph quality status is {after_quality['status']}")
        recommended = (
            "Review graph quality summary before relying on newly committed memory."
        )

    unembedded_claim_keys = tuple(readback.get("unembedded_claim_keys", ()))
    if unembedded_claim_keys:
        # The claims read back (Part of the write landed) but carry no fact
        # embedding, so they fall out of semantic ranking. Not a data-loss
        # failure — but it must never be a silent one.
        warnings.append(
            f"{len(unembedded_claim_keys)} committed claim(s) read back without "
            "a fact embedding; semantic ranking is degraded for them"
        )
        if status == "ok":
            status = "watch"
            recommended = (
                "Run 'potpie graph repair semantic_index' to re-embed claims "
                "missing vectors."
            )

    return GraphIngestionVerificationResult(
        ok=ok,
        status=status,
        plan_id=record.plan_id,
        pot_id=pot_id,
        claim_keys=claim_keys,
        readback_claim_keys=tuple(readback["readback_claim_keys"]),
        missing_claim_keys=tuple(readback["missing_claim_keys"]),
        unembedded_claim_keys=unembedded_claim_keys,
        readback_count=int(readback["readback_count"]),
        quality_status=str(after_quality["status"]),
        quality_counts=after_counts,
        quality_delta=deltas,
        quality_regressions=regressions,
        checked_reports=tuple(after_quality["checked_reports"]),
        warnings=tuple(warnings),
        unsupported=unsupported,
        detail=detail,
        recommended_next_action=recommended,
        subgraph_versions=_subgraph_versions(backend, pot_id),
    )


def _verification_readback(
    backend: GraphBackend,
    *,
    pot_id: str,
    claim_keys: tuple[str, ...],
) -> dict[str, Any]:
    if not claim_keys:
        return {
            "readback_claim_keys": (),
            "missing_claim_keys": (),
            "unembedded_claim_keys": (),
            "readback_count": 0,
            "unsupported": (),
        }
    try:
        rows = backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                claim_key_in=claim_keys,
                include_invalidated=False,
                limit=max(len(claim_keys) * 2, 20),
            )
        )
    except CapabilityNotImplemented as exc:
        return {
            "readback_claim_keys": (),
            "missing_claim_keys": (),
            "unembedded_claim_keys": (),
            "readback_count": 0,
            "unsupported": (
                _unsupported_from_exception(exc, fallback="claim_query.find_claims"),
            ),
        }
    deduped_rows = _dedupe_claim_rows(list(rows))
    readback_keys = tuple(dict.fromkeys(_row_claim_key(row) for row in deduped_rows))
    readback_set = set(readback_keys)
    # When the backend runs in vector mode, every committed claim should carry
    # a fact embedding; a claim without one silently drops out of semantic
    # ranking, so the gap must surface here rather than in nobody's logs.
    vector_mode = (
        getattr(backend.claim_query, "match_mode", "lexical") == "vector"
    )
    unembedded = (
        tuple(
            key
            for row in deduped_rows
            if (key := _row_claim_key(row)) and row.fact_embedding is None
        )
        if vector_mode
        else ()
    )
    return {
        "readback_claim_keys": readback_keys,
        "missing_claim_keys": tuple(
            key for key in claim_keys if key not in readback_set
        ),
        "unembedded_claim_keys": unembedded,
        "readback_count": len(readback_keys),
        "unsupported": (),
    }


def _verification_quality_snapshot(
    backend: GraphBackend,
    *,
    pot_id: str,
) -> dict[str, Any]:
    try:
        result = _quality_summary_result(
            backend,
            pot_id=pot_id,
            filters={
                "report": "summary",
                "limit": 50,
                "confidence_threshold": _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            },
        )
    except CapabilityNotImplemented as exc:
        return {
            "status": "unavailable",
            "quality_counts": {},
            "total_findings": 0,
            "checked_reports": (),
            "unsupported": (
                _unsupported_from_exception(exc, fallback="graph.quality.summary"),
            ),
        }
    metrics = dict(result.metrics)
    reports = metrics.get("quality_reports")
    checked_reports = (
        tuple(str(key) for key in reports.keys())
        if isinstance(reports, Mapping)
        else _QUALITY_SUMMARY_REPORTS
    )
    return {
        "status": str(result.status or "unknown"),
        "quality_counts": _int_map(metrics.get("quality_counts")),
        "total_findings": _safe_int(metrics.get("total_findings")),
        "checked_reports": checked_reports,
        "unsupported": tuple(result.unsupported),
    }


def _quality_count_map(snapshot: Mapping[str, Any] | None) -> dict[str, int]:
    if not snapshot:
        return {}
    counts = _int_map(snapshot.get("quality_counts"))
    if "total_findings" not in counts:
        counts["total_findings"] = _safe_int(snapshot.get("total_findings"))
    return counts


def _quality_count_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def _quality_regressions(
    *,
    before_quality: Mapping[str, Any] | None,
    after_quality: Mapping[str, Any],
    deltas: Mapping[str, int],
) -> dict[str, Mapping[str, int]]:
    if not before_quality:
        return {}
    regressions = {
        key: {
            "before": _safe_int(_quality_count_map(before_quality).get(key)),
            "after": _safe_int(_quality_count_map(after_quality).get(key)),
            "delta": int(delta),
        }
        for key, delta in deltas.items()
        if delta > 0
    }
    before_rank = _quality_status_rank(str(before_quality.get("status") or "unknown"))
    after_rank = _quality_status_rank(str(after_quality.get("status") or "unknown"))
    if after_rank > before_rank:
        regressions["health_status"] = {
            "before": before_rank,
            "after": after_rank,
            "delta": after_rank - before_rank,
        }
    return regressions


def _quality_status_rank(status: str) -> int:
    return {
        "ok": 0,
        "empty": 0,
        "watch": 1,
        "partial": 1,
        "unavailable": 1,
        "unknown": 1,
        "degraded": 2,
    }.get(status, 1)


def _verification_next_action(
    verification: GraphIngestionVerificationResult | None,
) -> str | None:
    if verification is None:
        return None
    if not verification.ok or verification.status != "ok":
        return verification.recommended_next_action
    return None


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _safe_int(raw) for key, raw in value.items()}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _record_matches_history_filter(
    record: GraphMutationPlanRecord,
    request: GraphHistoryRequest,
) -> bool:
    if request.claim_key and request.claim_key not in _claim_keys_from_record(record):
        return False
    if request.subgraph and request.subgraph not in _record_subgraphs(record):
        return False
    if request.entity_key and request.entity_key not in _record_entity_keys(record):
        return False
    return True


def _history_entry_from_plan(record: GraphMutationPlanRecord) -> GraphHistoryEntry:
    source_refs = _source_refs_from_record(record)
    claim_keys = _claim_keys_from_record(record)
    payload = {
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat(),
        "validation_issues": [dict(i) for i in record.validation_issues],
        "accepted_operations": [dict(op) for op in record.accepted_ops],
        "review_required_operations": [dict(op) for op in record.review_required_ops],
        "rejected_operations": [dict(op) for op in record.rejected_ops],
        "approval": record.approval.to_dict() if record.approval else None,
        "committed_at": record.committed_at.isoformat()
        if record.committed_at
        else None,
        "expected_subgraph_versions": dict(record.expected_subgraph_versions),
        "current_subgraph_versions": dict(record.current_subgraph_versions),
        "final_subgraph_versions": dict(record.final_subgraph_versions),
        "diff": record.diff.to_dict(),
        "claim_keys": list(claim_keys),
    }
    summary = f"mutation plan {record.status}"
    if record.mutation_id:
        summary += f" committed as {record.mutation_id}"
    return GraphHistoryEntry(
        kind="plan",
        id=record.plan_id,
        status=record.status,
        occurred_at=record.committed_at or record.created_at,
        plan_id=record.plan_id,
        mutation_id=record.mutation_id,
        entity_keys=_record_entity_keys(record),
        subgraph=_single_or_none(_record_subgraphs(record)),
        source_refs=source_refs,
        summary=summary,
        detail=record.detail,
        payload=payload,
    )


def _history_claim_rows(
    backend: GraphBackend,
    request: GraphHistoryRequest,
    records: tuple[GraphMutationPlanRecord, ...],
) -> tuple[tuple[ClaimRow, ...], tuple[Mapping[str, Any], ...]]:
    mutation_ids = {
        item
        for item in (
            request.mutation_id,
            *tuple(record.mutation_id for record in records),
        )
        if item
    }
    claim_keys = set()
    if request.claim_key:
        claim_keys.add(request.claim_key)
    if request.plan_id and not mutation_ids:
        for record in records:
            claim_keys.update(_claim_keys_from_record(record))

    should_query = bool(
        request.entity_key
        or request.claim_key
        or request.subgraph
        or request.mutation_id
        or mutation_ids
        or claim_keys
    )
    if not should_query:
        return (), ()

    base = {
        "pot_id": request.pot_id,
        "claim_key_in": tuple(sorted(claim_keys)),
        "subgraph_in": (request.subgraph,) if request.subgraph else (),
        "mutation_id_in": tuple(sorted(mutation_ids)),
        "valid_at_after": request.since,
        "valid_at_before": request.until,
        "include_invalidated": True,
        "limit": max(request.limit * 4, request.limit),
    }
    filters: list[ClaimQueryFilter]
    if request.entity_key:
        filters = [
            ClaimQueryFilter(**base, subject_key_in=(request.entity_key,)),
            ClaimQueryFilter(**base, object_key_in=(request.entity_key,)),
        ]
    else:
        filters = [ClaimQueryFilter(**base)]

    try:
        claim_query = backend.claim_query
        rows = []
        for filter_ in filters:
            rows.extend(claim_query.find_claims(filter_))
    except CapabilityNotImplemented as exc:
        return (
            (),
            (
                {
                    "name": getattr(exc, "capability", "claim_history"),
                    "reason": "not_implemented",
                    "detail": getattr(exc, "detail", str(exc)),
                },
            ),
        )
    return _dedupe_claim_rows(rows), ()


def _history_entry_from_claim(row: ClaimRow) -> GraphHistoryEntry:
    source_refs = row.source_refs or ((row.source_ref,) if row.source_ref else ())
    payload = {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "valid_at": _dt_iso(row.valid_at),
        "invalid_at": _dt_iso(row.invalid_at),
        "observed_at": _dt_iso(row.observed_at),
        "valid_until": _dt_iso(row.valid_until),
        "confidence": row.confidence,
        "evidence_strength": row.evidence_strength,
        "environment": row.environment,
        "fact": row.fact,
        "description": row.description,
        "source_system": row.source_system,
        "source_ref": row.source_ref,
        "properties": dict(row.properties),
    }
    return GraphHistoryEntry(
        kind="claim",
        id=row.claim_key
        or f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.mutation_id or ''}",
        status="invalidated" if row.invalid_at else "current",
        occurred_at=row.invalid_at or row.valid_at or row.observed_at,
        mutation_id=row.mutation_id,
        claim_key=row.claim_key,
        entity_keys=tuple(key for key in (row.subject_key, row.object_key) if key),
        subgraph=row.subgraph,
        truth=row.truth,
        source_refs=source_refs,
        evidence=row.evidence,
        summary=row.fact or row.description,
        payload=payload,
    )


def _normalize_quality_report(report: str | None) -> str:
    clean = (report or "summary").strip().lower().replace("_", "-")
    aliases = {
        "duplicates": "duplicate-candidates",
        "duplicate": "duplicate-candidates",
        "stale": "stale-facts",
        "conflicts": "conflicting-claims",
        "conflicting": "conflicting-claims",
        "orphans": "orphan-entities",
        "low-confidence-claims": "low-confidence",
        "drift": "projection-drift",
    }
    clean = aliases.get(clean, clean)
    allowed = {
        "summary",
        "duplicate-candidates",
        "stale-facts",
        "conflicting-claims",
        "orphan-entities",
        "low-confidence",
        "projection-drift",
    }
    if clean not in allowed:
        raise ValueError(
            f"unknown quality report {clean!r}; expected one of: "
            f"{', '.join(sorted(allowed))}"
        )
    return clean


def _quality_filters(
    *,
    report: str,
    subgraph: str | None,
    limit: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"report": report, "limit": limit}
    clean_subgraph = _clean_str(subgraph)
    if clean_subgraph:
        out["subgraph"] = clean_subgraph
    if report == "low-confidence":
        out["confidence_threshold"] = confidence_threshold
    return out


def _quality_deep_report(
    backend: GraphBackend,
    *,
    pot_id: str,
    report: str,
    subgraph: str | None,
    limit: int,
    confidence_threshold: float,
) -> tuple[
    tuple[GraphQualityFinding, ...],
    dict[str, Any],
    tuple[Mapping[str, Any], ...],
]:
    if report == "duplicate-candidates":
        return _quality_duplicate_candidates(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
        )
    if report == "stale-facts":
        return _quality_stale_facts(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
        )
    if report == "conflicting-claims":
        return _quality_conflicting_claims(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
        )
    if report == "orphan-entities":
        return _quality_orphan_entities(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
        )
    if report == "low-confidence":
        return _quality_low_confidence(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
            confidence_threshold=confidence_threshold,
        )
    if report == "projection-drift":
        return _quality_projection_drift(
            backend,
            pot_id=pot_id,
            subgraph=subgraph,
            limit=limit,
        )
    raise ValueError(f"unknown quality report {report!r}")


def _quality_summary_result(
    backend: GraphBackend,
    *,
    pot_id: str,
    filters: Mapping[str, Any],
) -> GraphQualityResult:
    unsupported: list[Mapping[str, Any]] = []
    counts = _quality_analytics_call(
        backend,
        pot_id=pot_id,
        method="counts",
        unsupported=unsupported,
    )
    freshness = _quality_analytics_call(
        backend,
        pot_id=pot_id,
        method="freshness",
        unsupported=unsupported,
    )
    quality = _quality_analytics_call(
        backend,
        pot_id=pot_id,
        method="quality",
        unsupported=unsupported,
    )
    report_summaries: dict[str, dict[str, Any]] = {}
    quality_counts: dict[str, int] = {}
    summary_findings: list[GraphQualityFinding] = []
    summary_unsupported: list[Mapping[str, Any]] = []
    subgraph = _clean_str(filters.get("subgraph"))
    limit = max(1, min(int(filters.get("limit") or 50), 500))
    confidence_threshold = float(
        filters.get("confidence_threshold") or _DEFAULT_LOW_CONFIDENCE_THRESHOLD
    )
    for report in _QUALITY_SUMMARY_REPORTS:
        findings, report_metrics, report_unsupported = _quality_deep_report(
            backend,
            pot_id=pot_id,
            report=report,
            subgraph=subgraph,
            limit=limit,
            confidence_threshold=confidence_threshold,
        )
        report_status = _quality_status(findings, report_unsupported)
        finding_count = len(findings)
        quality_counts[report.replace("-", "_")] = finding_count
        report_summaries[report] = {
            "status": report_status,
            "finding_count": finding_count,
            "severity_counts": _quality_severity_counts(findings),
            "unsupported_count": len(report_unsupported),
            "metrics": report_metrics,
        }
        summary_findings.extend(findings)
        summary_unsupported.extend(report_unsupported)

    unsupported.extend(summary_unsupported)
    metrics: dict[str, Any] = {
        "counts": counts,
        "freshness": freshness,
        "backend_quality": quality,
        "quality_counts": quality_counts,
        "quality_reports": report_summaries,
        "total_findings": sum(quality_counts.values()),
    }
    status = _quality_status(tuple(summary_findings), tuple(unsupported))
    if status == "ok" and quality.get("status"):
        status = str(quality.get("status"))
    detail = None
    recommended = None
    if unsupported and not counts and not freshness and not quality:
        status = "unavailable"
        detail = "graph analytics are unavailable for the active backend"
        recommended = "Use graph status to inspect backend capabilities."
    elif status in {"degraded", "watch", "partial"}:
        recommended = (
            "Run the affected `graph quality <report>` command for details, then "
            "use graph propose/commit for semantic corrections or graph inbox add "
            "for uncertain work."
        )
    return GraphQualityResult(
        ok=True,
        pot_id=pot_id,
        report="summary",
        status=status,
        metrics=metrics,
        filters=filters,
        unsupported=tuple(unsupported),
        detail=detail,
        recommended_next_action=recommended,
        subgraph_versions=_subgraph_versions(backend, pot_id),
    )


def _quality_severity_counts(
    findings: tuple[GraphQualityFinding, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def _quality_analytics_call(
    backend: GraphBackend,
    *,
    pot_id: str,
    method: str,
    unsupported: list[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        fn = getattr(backend.analytics, method)
        return dict(fn(pot_id))
    except CapabilityNotImplemented as exc:
        unsupported.append(
            _unsupported_from_exception(exc, fallback=f"analytics.{method}")
        )
        return {}


def _quality_duplicate_candidates(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend, pot_id=pot_id, subgraph=subgraph, limit=limit
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    entity_keys = sorted(
        {key for row in rows for key in (row.subject_key, row.object_key)}
    )
    labels, props, prop_unsupported = _quality_entity_metadata(
        backend, pot_id=pot_id, entity_keys=entity_keys
    )
    unsupported += prop_unsupported
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in entity_keys:
        name = _entity_display_name(props.get(key, {}))
        if not name:
            continue
        label = _primary_label(labels.get(key, ()), key)
        buckets[(label, _normalize_identity_text(name))].append(key)

    findings: list[GraphQualityFinding] = []
    for (label, normalized_name), keys in sorted(buckets.items()):
        unique_keys = tuple(dict.fromkeys(keys))
        if len(unique_keys) < 2:
            continue
        source_refs = _source_refs_for_entities(rows, unique_keys)
        findings.append(
            _quality_finding(
                kind="duplicate-candidate",
                severity="warning",
                summary=(
                    f"{len(unique_keys)} {label} entities share display name "
                    f"{normalized_name!r}"
                ),
                entity_keys=unique_keys,
                source_refs=source_refs,
                suggested_action={
                    "type": "review",
                    "command": "graph inbox add",
                    "reason": (
                        "Duplicate entity candidates need identity review before "
                        "merge_duplicate_entities is proposed."
                    ),
                },
                payload={"label": label, "normalized_name": normalized_name},
            )
        )
        if len(findings) >= limit:
            break
    return (
        tuple(findings),
        {"scanned_claims": len(rows), "scanned_entities": len(entity_keys)},
        unsupported,
    )


def _quality_stale_facts(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend, pot_id=pot_id, subgraph=subgraph, limit=limit
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    now = datetime.now(timezone.utc)
    findings: list[GraphQualityFinding] = []
    for row in rows:
        stale_reasons = _stale_reasons(row, now=now)
        if not stale_reasons:
            continue
        findings.append(
            _quality_finding(
                kind="stale-fact",
                severity="warning",
                summary=f"{row.predicate} claim is stale",
                claim_keys=(_row_claim_key(row),),
                entity_keys=(row.subject_key, row.object_key),
                predicates=(row.predicate,),
                subgraph=row.subgraph,
                source_refs=_row_source_refs(row),
                evidence=row.evidence,
                detail="; ".join(stale_reasons),
                suggested_action={
                    "type": "semantic_mutation",
                    "command": "graph propose",
                    "reason": "Refresh, supersede, or end the stale fact.",
                },
                payload=_claim_payload(row),
            )
        )
        if len(findings) >= limit:
            break
    return tuple(findings), {"scanned_claims": len(rows)}, unsupported


def _quality_conflicting_claims(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend, pot_id=pot_id, subgraph=subgraph, limit=limit
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    entity_keys = sorted({row.object_key for row in rows})
    labels, _props, label_unsupported = _quality_entity_metadata(
        backend, pot_id=pot_id, entity_keys=entity_keys, properties=False
    )
    unsupported += label_unsupported
    findings = list(_singleton_conflict_findings(rows, limit=limit))
    if len(findings) < limit:
        findings.extend(
            _predicate_family_conflict_findings(
                rows,
                object_labels=labels,
                existing_ids={finding.finding_id for finding in findings},
                limit=limit - len(findings),
            )
        )
    return tuple(findings[:limit]), {"scanned_claims": len(rows)}, unsupported


def _quality_orphan_entities(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend,
        pot_id=pot_id,
        subgraph=subgraph,
        limit=limit,
        include_invalidated=True,
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    all_entities = {key for row in rows for key in (row.subject_key, row.object_key)}
    live_degree: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.invalid_at is not None:
            continue
        live_degree[row.subject_key] += 1
        live_degree[row.object_key] += 1
    orphan_keys = tuple(
        sorted(key for key in all_entities if live_degree.get(key, 0) == 0)
    )
    findings: list[GraphQualityFinding] = []
    for key in orphan_keys[:limit]:
        source_refs = _source_refs_for_entities(rows, (key,))
        findings.append(
            _quality_finding(
                kind="orphan-entity",
                severity="warning",
                summary=f"{key} has no live useful claims",
                entity_keys=(key,),
                source_refs=source_refs,
                suggested_action={
                    "type": "review",
                    "command": "graph inbox add",
                    "reason": "Verify whether the entity should be reconnected or cleaned up.",
                },
            )
        )
    return (
        tuple(findings),
        {"scanned_claims": len(rows), "scanned_entities": len(all_entities)},
        unsupported,
    )


def _quality_low_confidence(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
    confidence_threshold: float,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend, pot_id=pot_id, subgraph=subgraph, limit=limit
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    findings: list[GraphQualityFinding] = []
    for row in rows:
        reasons = _confidence_reasons(row, threshold=confidence_threshold)
        if not reasons:
            continue
        severity = "error" if "missing required evidence" in reasons else "warning"
        findings.append(
            _quality_finding(
                kind="low-confidence",
                severity=severity,
                summary=f"{row.predicate} claim needs stronger evidence",
                claim_keys=(_row_claim_key(row),),
                entity_keys=(row.subject_key, row.object_key),
                predicates=(row.predicate,),
                subgraph=row.subgraph,
                source_refs=_row_source_refs(row),
                evidence=row.evidence,
                detail="; ".join(reasons),
                suggested_action={
                    "type": "semantic_mutation",
                    "command": "graph propose",
                    "reason": "Attach stronger evidence or retract/supersede the claim.",
                },
                payload=_claim_payload(row),
            )
        )
        if len(findings) >= limit:
            break
    return tuple(findings), {"scanned_claims": len(rows)}, unsupported


def _quality_projection_drift(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
) -> tuple[
    tuple[GraphQualityFinding, ...], dict[str, Any], tuple[Mapping[str, Any], ...]
]:
    rows, unsupported = _quality_claim_rows(
        backend, pot_id=pot_id, subgraph=subgraph, limit=limit
    )
    if unsupported:
        return (), {"scanned_claims": 0}, unsupported
    findings = list(
        _invalid_endpoint_findings(backend, pot_id=pot_id, rows=rows, limit=limit)
    )
    inspection_unsupported: tuple[Mapping[str, Any], ...] = ()
    try:
        sl = backend.inspection.slice(
            pot_id=pot_id,
            filter_=ClaimQueryFilter(
                pot_id=pot_id,
                subgraph_in=(subgraph,) if subgraph else (),
                limit=_quality_scan_limit(limit),
            ),
        )
    except CapabilityNotImplemented as exc:
        inspection_unsupported = (
            _unsupported_from_exception(exc, fallback="inspection.slice"),
        )
        sl = None
    unsupported += inspection_unsupported
    if sl is not None:
        claim_edges = {
            (row.subject_key, row.predicate, row.object_key)
            for row in rows
            if row.invalid_at is None
        }
        projection_edges = {
            (edge.from_key, edge.predicate, edge.to_key) for edge in sl.edges
        }
        missing = sorted(claim_edges - projection_edges)
        extra = sorted(projection_edges - claim_edges)
        if missing or extra:
            findings.append(
                _quality_finding(
                    kind="projection-drift",
                    severity="error",
                    summary="Inspection projection does not match canonical claims",
                    predicates=tuple(sorted({item[1] for item in (*missing, *extra)})),
                    suggested_action={
                        "type": "operator_repair",
                        "command": "graph repair --all",
                        "reason": "Rebuild derived projections from canonical claims.",
                    },
                    payload={
                        "missing_edges": [
                            {"from": s, "predicate": p, "to": o}
                            for s, p, o in missing[:limit]
                        ],
                        "extra_edges": [
                            {"from": s, "predicate": p, "to": o}
                            for s, p, o in extra[:limit]
                        ],
                    },
                )
            )
    metrics = {
        "scanned_claims": len(rows),
        "projection_checked": sl is not None,
    }
    return tuple(findings[:limit]), metrics, unsupported


def _quality_claim_rows(
    backend: GraphBackend,
    *,
    pot_id: str,
    subgraph: str | None,
    limit: int,
    include_invalidated: bool = False,
) -> tuple[tuple[ClaimRow, ...], tuple[Mapping[str, Any], ...]]:
    try:
        rows = backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                subgraph_in=(subgraph,) if subgraph else (),
                include_invalidated=include_invalidated,
                limit=_quality_scan_limit(limit),
            )
        )
    except CapabilityNotImplemented as exc:
        return (), (
            _unsupported_from_exception(exc, fallback="claim_query.find_claims"),
        )
    return _dedupe_claim_rows(list(rows)), ()


def _quality_entity_metadata(
    backend: GraphBackend,
    *,
    pot_id: str,
    entity_keys: list[str],
    properties: bool = True,
) -> tuple[
    Mapping[str, tuple[str, ...]],
    Mapping[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
]:
    if not entity_keys:
        return {}, {}, ()
    unsupported: list[Mapping[str, Any]] = []
    try:
        labels = dict(
            backend.claim_query.entity_labels(pot_id=pot_id, entity_keys=entity_keys)
        )
    except CapabilityNotImplemented as exc:
        labels = {}
        unsupported.append(
            _unsupported_from_exception(exc, fallback="claim_query.entity_labels")
        )
    props: dict[str, Mapping[str, Any]] = {}
    if properties:
        for key in entity_keys:
            try:
                props[key] = dict(
                    backend.claim_query.entity_properties(pot_id=pot_id, entity_key=key)
                )
            except CapabilityNotImplemented as exc:
                unsupported.append(
                    _unsupported_from_exception(
                        exc, fallback="claim_query.entity_properties"
                    )
                )
                break
    return labels, props, tuple(unsupported)


def _quality_scan_limit(limit: int) -> int:
    return max(limit, min(_QUALITY_SCAN_MAX, max(limit * 20, 100)))


def _quality_status(
    findings: tuple[GraphQualityFinding, ...],
    unsupported: tuple[Mapping[str, Any], ...],
) -> str:
    if any(finding.severity in {"error", "blocking"} for finding in findings):
        return "degraded"
    if findings:
        return "watch"
    if unsupported:
        return "partial"
    return "ok"


def _unsupported_from_exception(
    exc: CapabilityNotImplemented,
    *,
    fallback: str,
) -> Mapping[str, Any]:
    return {
        "name": getattr(exc, "capability", fallback) or fallback,
        "reason": "not_implemented",
        "detail": getattr(exc, "detail", str(exc)),
    }


def _quality_finding(
    *,
    kind: str,
    severity: str,
    summary: str,
    entity_keys: tuple[str, ...] = (),
    claim_keys: tuple[str, ...] = (),
    predicates: tuple[str, ...] = (),
    subgraph: str | None = None,
    source_refs: tuple[str, ...] = (),
    evidence: tuple[Mapping[str, Any], ...] = (),
    detail: str | None = None,
    suggested_action: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> GraphQualityFinding:
    clean_entities = tuple(dict.fromkeys(entity_keys))
    clean_claims = tuple(dict.fromkeys(claim_keys))
    clean_predicates = tuple(dict.fromkeys(predicates))
    finding_id = _quality_finding_id(
        kind,
        *clean_entities,
        *clean_claims,
        *clean_predicates,
        subgraph or "",
        summary,
    )
    return GraphQualityFinding(
        finding_id=finding_id,
        kind=kind,
        severity=severity,
        summary=summary,
        entity_keys=clean_entities,
        claim_keys=clean_claims,
        predicates=clean_predicates,
        subgraph=subgraph,
        source_refs=tuple(dict.fromkeys(source_refs)),
        evidence=evidence,
        detail=detail,
        suggested_action=dict(suggested_action or {}),
        payload=dict(payload or {}),
    )


def _quality_finding_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()
    return f"quality:{kind}:{digest[:16]}"


def _stale_reasons(row: ClaimRow, *, now: datetime) -> list[str]:
    reasons: list[str] = []
    if row.valid_until is not None and row.valid_until < now:
        reasons.append(f"valid_until elapsed at {row.valid_until.isoformat()}")
    freshness = str(row.properties.get("freshness") or "").lower()
    sync_status = str(row.properties.get("sync_status") or "").lower()
    if freshness == "stale" or sync_status == "stale":
        reasons.append("source reference is marked stale")
    return reasons


def _confidence_reasons(row: ClaimRow, *, threshold: float) -> list[str]:
    reasons: list[str] = []
    if row.confidence is not None and row.confidence < threshold:
        reasons.append(f"confidence {row.confidence:.2f} is below {threshold:.2f}")
    has_evidence = bool(row.evidence or row.source_refs or row.source_ref)
    if row.truth in EVIDENCE_REQUIRED_TRUTH_CLASSES and not has_evidence:
        reasons.append("missing required evidence")
    return reasons


def _singleton_conflict_findings(
    rows: tuple[ClaimRow, ...],
    *,
    limit: int,
) -> tuple[GraphQualityFinding, ...]:
    buckets: dict[tuple[str, str], list[ClaimRow]] = defaultdict(list)
    for row in rows:
        if row.invalid_at is None and is_singleton_predicate(row.predicate):
            buckets[(row.subject_key, row.predicate)].append(row)
    findings: list[GraphQualityFinding] = []
    for (subject, predicate), group in sorted(buckets.items()):
        objects = {row.object_key for row in group}
        if len(objects) < 2:
            continue
        claim_keys = tuple(_row_claim_key(row) for row in group)
        findings.append(
            _quality_finding(
                kind="conflicting-claim",
                severity="warning",
                summary=f"{subject} has conflicting live {predicate} claims",
                entity_keys=tuple(dict.fromkeys([subject, *sorted(objects)])),
                claim_keys=claim_keys,
                predicates=(predicate,),
                subgraph=_single_or_none(
                    tuple(dict.fromkeys(row.subgraph for row in group if row.subgraph))
                ),
                source_refs=_source_refs_from_rows(group),
                evidence=tuple(item for row in group for item in row.evidence),
                suggested_action={
                    "type": "semantic_mutation",
                    "command": "graph propose",
                    "reason": "Supersede or retract the incorrect singleton claim.",
                },
                payload={"object_keys": sorted(objects)},
            )
        )
        if len(findings) >= limit:
            break
    return tuple(findings)


def _predicate_family_conflict_findings(
    rows: tuple[ClaimRow, ...],
    *,
    object_labels: Mapping[str, tuple[str, ...]],
    existing_ids: set[str],
    limit: int,
) -> tuple[GraphQualityFinding, ...]:
    row_by_id: dict[str, ClaimRow] = {}
    inputs: list[EpisodicEdgeConflictInput] = []
    for row in rows:
        if row.invalid_at is not None:
            continue
        row_id = _row_claim_key(row)
        row_by_id[row_id] = row
        inputs.append(
            EpisodicEdgeConflictInput(
                uuid=row_id,
                name=row.predicate,
                source_uuid=row.subject_key,
                target_uuid=row.object_key,
                valid_at=row.valid_at,
                created_at=row.observed_at,
                target_labels=object_labels.get(row.object_key, ()),
            )
        )
    findings: list[GraphQualityFinding] = []
    for conflict in detect_family_conflicts(inputs):
        a = row_by_id.get(str(conflict.get("edge_a_uuid")))
        b = row_by_id.get(str(conflict.get("edge_b_uuid")))
        if a is None or b is None:
            continue
        severity = "error" if conflict.get("severity") == "blocking" else "warning"
        finding = _quality_finding(
            kind="conflicting-claim",
            severity=severity,
            summary=(
                f"{conflict.get('family')} conflict between "
                f"{a.object_key} and {b.object_key}"
            ),
            entity_keys=(a.subject_key, a.object_key, b.object_key),
            claim_keys=(_row_claim_key(a), _row_claim_key(b)),
            predicates=(a.predicate, b.predicate),
            subgraph=a.subgraph if a.subgraph == b.subgraph else None,
            source_refs=_source_refs_from_rows((a, b)),
            evidence=tuple(item for row in (a, b) for item in row.evidence),
            suggested_action={
                "type": "semantic_mutation"
                if conflict.get("auto_resolvable")
                else "review",
                "command": "graph propose"
                if conflict.get("auto_resolvable")
                else "graph inbox add",
                "reason": str(conflict.get("suggested_action") or "human_review"),
            },
            payload=conflict,
        )
        if finding.finding_id in existing_ids:
            continue
        findings.append(finding)
        if len(findings) >= limit:
            break
    return tuple(findings)


def _invalid_endpoint_findings(
    backend: GraphBackend,
    *,
    pot_id: str,
    rows: tuple[ClaimRow, ...],
    limit: int,
) -> tuple[GraphQualityFinding, ...]:
    entity_keys = sorted(
        {key for row in rows for key in (row.subject_key, row.object_key)}
    )
    labels, _props, unsupported = _quality_entity_metadata(
        backend, pot_id=pot_id, entity_keys=entity_keys, properties=False
    )
    if unsupported:
        return ()
    findings: list[GraphQualityFinding] = []
    for row in rows:
        spec = EDGE_TYPES.get(row.predicate)
        if spec is None or not spec.allowed_pairs:
            continue
        subject_labels = labels.get(row.subject_key, ())
        object_labels = labels.get(row.object_key, ())
        if not subject_labels or not object_labels:
            continue
        if spec.allows(subject_labels, object_labels):
            continue
        allowed = set(spec.allowed_pairs)
        findings.append(
            _quality_finding(
                kind="invalid-endpoint-pair",
                severity="error",
                summary=f"{row.predicate} has invalid endpoint labels",
                entity_keys=(row.subject_key, row.object_key),
                claim_keys=(_row_claim_key(row),),
                predicates=(row.predicate,),
                subgraph=row.subgraph,
                source_refs=_row_source_refs(row),
                evidence=row.evidence,
                suggested_action={
                    "type": "semantic_mutation",
                    "command": "graph propose",
                    "reason": "Retract, supersede, or rewrite this invalid claim.",
                },
                payload={
                    "subject_labels": list(subject_labels),
                    "object_labels": list(object_labels),
                    "allowed_pairs": [
                        {"subject": subject, "object": object_}
                        for subject, object_ in sorted(allowed)
                    ],
                },
            )
        )
        if len(findings) >= limit:
            break
    return tuple(findings)


def _row_claim_key(row: ClaimRow) -> str:
    return row.claim_key or (
        f"{row.predicate}:{row.subject_key}:{row.object_key}:{row.mutation_id or ''}"
    )


def _row_source_refs(row: ClaimRow) -> tuple[str, ...]:
    refs = list(row.source_refs)
    if row.source_ref:
        refs.append(row.source_ref)
    return tuple(dict.fromkeys(refs))


def _source_refs_from_rows(
    rows: tuple[ClaimRow, ...] | list[ClaimRow],
) -> tuple[str, ...]:
    refs: list[str] = []
    for row in rows:
        refs.extend(_row_source_refs(row))
    return tuple(dict.fromkeys(refs))


def _source_refs_for_entities(
    rows: tuple[ClaimRow, ...],
    entity_keys: tuple[str, ...],
) -> tuple[str, ...]:
    wanted = set(entity_keys)
    refs: list[str] = []
    for row in rows:
        if row.subject_key in wanted or row.object_key in wanted:
            refs.extend(_row_source_refs(row))
    return tuple(dict.fromkeys(refs))


def _entity_display_name(props: Mapping[str, Any]) -> str | None:
    for key in ("name", "display_name", "title", "summary"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_identity_text(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _primary_label(labels: tuple[str, ...], entity_key: str) -> str:
    if labels:
        return labels[0]
    if ":" in entity_key:
        return entity_key.split(":", 1)[0]
    return "Entity"


def _claim_payload(row: ClaimRow) -> dict[str, Any]:
    return {
        "claim_key": _row_claim_key(row),
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "truth": row.truth,
        "confidence": row.confidence,
        "valid_at": _dt_iso(row.valid_at),
        "valid_until": _dt_iso(row.valid_until),
        "observed_at": _dt_iso(row.observed_at),
        "source_refs": list(_row_source_refs(row)),
    }


def _record_subgraphs(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    values: list[str] = []
    for op in (*record.accepted_ops, *record.review_required_ops):
        if isinstance(op, Mapping) and op.get("subgraph"):
            values.append(str(op["subgraph"]))
    if record.lowered_batch is not None:
        for edge in record.lowered_batch.edge_upserts:
            subgraph = edge.properties.get("subgraph")
            if isinstance(subgraph, str) and subgraph:
                values.append(subgraph)
    return tuple(dict.fromkeys(values))


def _record_entity_keys(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    keys: list[str] = []
    if record.lowered_batch is not None:
        for entity in record.lowered_batch.entity_upserts:
            keys.append(entity.entity_key)
        for edge in record.lowered_batch.edge_upserts:
            keys.extend((edge.from_entity_key, edge.to_entity_key))
        for invalidation in record.lowered_batch.invalidations:
            if invalidation.target_entity_key:
                keys.append(invalidation.target_entity_key)
            if invalidation.target_edge:
                _predicate, subject, object_ = invalidation.target_edge
                keys.extend((subject, object_))
    for value in _payload_entity_keys(record.original_payload):
        keys.append(value)
    return tuple(dict.fromkeys(key for key in keys if key))


def _payload_entity_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        key = value.get("key")
        if isinstance(key, str) and ":" in key:
            keys.append(key)
        for child in value.values():
            keys.extend(_payload_entity_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_payload_entity_keys(child))
    return tuple(keys)


def _source_refs_from_record(record: GraphMutationPlanRecord) -> tuple[str, ...]:
    refs = list(_payload_source_refs(record.original_payload))
    if record.lowered_batch is not None:
        refs.extend(item.ref for item in record.lowered_batch.evidence if item.ref)
    if record.provenance and record.provenance.source_ref:
        refs.append(record.provenance.source_ref)
    return tuple(dict.fromkeys(refs))


def _payload_source_refs(value: Any) -> tuple[str, ...]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key in ("source_ref", "source"):
            ref = value.get(key)
            if isinstance(ref, str) and ref:
                refs.append(ref)
        source_refs = value.get("source_refs")
        if isinstance(source_refs, str) and source_refs:
            refs.append(source_refs)
        elif isinstance(source_refs, (list, tuple)):
            refs.extend(str(ref) for ref in source_refs if ref)
        for child in value.values():
            refs.extend(_payload_source_refs(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            refs.extend(_payload_source_refs(child))
    return tuple(refs)


def _dedupe_claim_rows(rows: list[ClaimRow]) -> tuple[ClaimRow, ...]:
    out: list[ClaimRow] = []
    seen: set[tuple[str | None, str, str, str, str | None]] = set()
    for row in rows:
        key = (
            row.claim_key,
            row.predicate,
            row.subject_key,
            row.object_key,
            row.mutation_id,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return tuple(out)


def _dedupe_history_entries(
    entries: list[GraphHistoryEntry],
) -> list[GraphHistoryEntry]:
    out: list[GraphHistoryEntry] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.kind, entry.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _history_sort_key(entry: GraphHistoryEntry) -> datetime:
    return entry.occurred_at or datetime.min.replace(tzinfo=timezone.utc)


def _single_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if len(values) == 1 else None


def _clean_str(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    return clean or None


def _required_clean(value: str | None, field: str) -> str:
    clean = _clean_str(value)
    if not clean:
        raise ValueError(f"{field} is required")
    return clean


def _clean_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    for value in values:
        clean = _clean_str(str(value) if value is not None else None)
        if clean:
            out.append(clean)
    return tuple(dict.fromkeys(out))


def _normalize_inbox_status_filter(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return ()
    allowed = {status.value for status in GraphInboxStatus}
    out: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            status = item.strip().lower()
            if not status:
                continue
            if status not in allowed:
                raise ValueError(
                    f"unknown inbox status {status!r}; expected one of: "
                    f"{', '.join(sorted(allowed))}"
                )
            out.append(status)
    return tuple(dict.fromkeys(out))


def _inbox_missing_result(
    *,
    pot_id: str,
    action: str,
    item_id: str,
) -> GraphInboxResult:
    return GraphInboxResult(
        ok=False,
        pot_id=pot_id,
        action=action,
        detail=f"inbox item {item_id!r} was not found for this pot",
        recommended_next_action="Run graph inbox list --json and use an item_id from the result.",
    )


def _terminal_inbox_result(
    item: GraphInboxItem,
    *,
    action: str,
) -> GraphInboxResult | None:
    if item.status not in TERMINAL_INBOX_STATUSES:
        return None
    return GraphInboxResult(
        ok=False,
        pot_id=item.pot_id,
        action=action,
        item=item,
        detail=f"inbox item is {item.status} and cannot be changed",
        recommended_next_action="Create a new inbox item if more graph work remains.",
    )


def _dt_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _approval_error(
    record: GraphMutationPlanRecord,
    *,
    approved_by: str | None,
) -> str | None:
    if record.review_required_ops:
        return (
            "plan contains review-required operations that the local commit path "
            "does not apply yet"
        )
    if record.status == GraphMutationPlanStatus.review_required.value or (
        record.risk in {MutationRisk.medium.value, MutationRisk.high.value}
    ):
        if not (record.approval or (approved_by and approved_by.strip())):
            return "approval required for medium- or high-risk plan"
    return None


def _batch_has_work(batch) -> bool:
    return bool(
        batch.entity_upserts
        or batch.edge_upserts
        or batch.edge_deletes
        or batch.invalidations
    )


def _provenance_for_commit(
    provenance: ProvenanceContext | None,
    *,
    approved_by: str | None,
) -> ProvenanceContext | None:
    if not approved_by:
        return provenance
    if provenance is None:
        return ProvenanceContext(actor_user_id=approved_by)
    if provenance.actor_user_id:
        return provenance
    return replace(provenance, actor_user_id=approved_by)


__all__ = [
    "GraphWorkbenchService",
    "graph_error_envelope",
    "graph_not_implemented_envelope",
    "graph_not_implemented_result",
    "graph_success_envelope",
    "new_graph_request_id",
    "normalize_catalog_result",
    "normalize_workbench_result",
]
