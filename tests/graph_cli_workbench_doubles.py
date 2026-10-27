"""Focused recording ports for graph CLI argument and response contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from potpie_context_engine.core.graph_inbox import GraphInboxItem, GraphInboxResult
from potpie_context_engine.core.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationDiff,
    GraphMutationProposal,
)
from potpie_context_engine.core.graph_quality import (
    GraphQualityFinding,
    GraphQualityResult,
)


def _proposal(
    *,
    ok: bool = True,
    status: str = "validated",
    risk: str = "low",
    plan_id: str = "mutation-plan:test",
    issues: tuple[dict, ...] = (),
) -> GraphMutationProposal:
    return GraphMutationProposal(
        ok=ok,
        plan_id=plan_id,
        status=status,
        risk=risk,
        pot_id="p",
        auto_applicable=status == "validated" and risk == "low",
        expires_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
        expected_subgraph_versions={"_global": 0},
        current_subgraph_versions={"_global": 0},
        diff=GraphMutationDiff(edge_upserts=1, claim_keys=("claim:test",)),
        issues=issues,
        rejected_operations=issues,
        claim_keys=("claim:test",),
    )


def _inbox_result(
    *,
    action: str = "add",
    status: str = "pending",
    ok: bool = True,
) -> GraphInboxResult:
    item = GraphInboxItem(
        item_id="graph-inbox:test",
        pot_id="p",
        status=status,
        summary="Possible graph update",
        evidence=("github:pr:955",),
        source_refs=("github:pr:955",),
        suspected_subgraphs=("debugging",),
        created_by={"surface": "cli", "actor": "codex"},
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )
    return GraphInboxResult(
        ok=ok,
        pot_id="p",
        action=action,
        item=item if action != "list" else None,
        items=(item,) if action == "list" else (),
        filters={"limit": 50} if action == "list" else {},
        detail=None if ok else "inbox item could not be changed",
    )


def _quality_result(
    *,
    report: str = "summary",
    status: str = "ok",
) -> GraphQualityResult:
    finding = GraphQualityFinding(
        finding_id="quality:low-confidence:test",
        kind="low-confidence",
        severity="warning",
        summary="claim needs stronger evidence",
        claim_keys=("claim:test",),
        entity_keys=("service:web", "repo:github.com/acme/web"),
        predicates=("DEFINED_IN",),
        source_refs=("repo:manifest",),
    )
    return GraphQualityResult(
        ok=True,
        pot_id="p",
        report=report,
        status=status,
        findings=() if report == "summary" else (finding,),
        metrics=(
            {
                "counts": {"claims": 3},
                "quality_counts": {
                    "duplicate_candidates": 0,
                    "stale_facts": 0,
                    "conflicting_claims": 1,
                    "orphan_entities": 0,
                    "low_confidence": 0,
                    "projection_drift": 2,
                },
                "quality_reports": {
                    "conflicting-claims": {
                        "status": status,
                        "finding_count": 1,
                        "severity_counts": {"warning": 1},
                    },
                    "projection-drift": {
                        "status": status,
                        "finding_count": 2,
                        "severity_counts": {"error": 2},
                    },
                },
                "total_findings": 3,
            }
            if report == "summary"
            else {"scanned_claims": 3}
        ),
        filters={"report": report, "limit": 50},
        subgraph_versions={"_global": 3},
    )


class ProposalRecorder:
    def __init__(self, *, proposal):
        self.proposal = proposal
        self.propose_calls = []
        self.propose_approvals = []
        self.commit_calls = []

    def propose(self, payload, *, pot_id, ttl_seconds=None, approved_by=None):
        self.propose_calls.append((payload, pot_id, ttl_seconds))
        self.propose_approvals.append(approved_by)
        return self.proposal


class CommitRecorder:
    def __init__(self, *, commit_result):
        self.commit_result = commit_result
        self.commit_calls = []
        self.deferred_verification = []
        self.verify_calls = []

    def commit(
        self,
        plan_id,
        *,
        pot_id,
        approved_by=None,
        verify=False,
        defer_verification=False,
        recover_stale=True,
    ):
        self.commit_calls.append((plan_id, pot_id, approved_by, verify))
        self.deferred_verification.append(defer_verification)
        if verify and defer_verification:
            return replace(self.commit_result, verification=None)
        return self.commit_result

    def verify_commit(self, plan_id, *, pot_id):
        self.verify_calls.append((plan_id, pot_id))
        return self.commit_result.verification or GraphIngestionVerificationResult(
            ok=True, status="ok", plan_id=plan_id, pot_id=pot_id
        )


class InboxRecorder:
    def __init__(self, *, inbox_result):
        self.inbox_result = inbox_result
        self.inbox_calls = []

    def inbox_add(self, **kwargs):
        self.inbox_calls.append(("add", kwargs))
        return self.inbox_result

    def inbox_list(self, **kwargs):
        self.inbox_calls.append(("list", kwargs))
        return self.inbox_result

    def inbox_claim(self, **kwargs):
        self.inbox_calls.append(("claim", kwargs))
        return self.inbox_result

    def inbox_mark_applied(self, **kwargs):
        self.inbox_calls.append(("mark-applied", kwargs))
        return self.inbox_result

    def inbox_mark_rejected(self, **kwargs):
        self.inbox_calls.append(("mark-rejected", kwargs))
        return self.inbox_result

    def inbox_close(self, **kwargs):
        self.inbox_calls.append(("close", kwargs))
        return self.inbox_result


class QualityRecorder:
    def __init__(self, *, quality_result):
        self.quality_result = quality_result
        self.quality_calls = []

    def quality(self, **kwargs):
        self.quality_calls.append(kwargs)
        return self.quality_result
