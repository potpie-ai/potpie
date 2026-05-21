"""P9 use-case readers (rebuild plan P9).

These tests exercise the four UC readers (CodingPreferences, InfraTopology,
Timeline, PriorBugs) against the in-memory claim store. The goal is to
prove (a) the canonical edge-shape feeds through the readers cleanly,
(b) ranking + coverage propagate, (c) scope filters work, and (d) the
F1/F2/F4 fix paths produce the right answers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from adapters.outbound.claim_query.in_memory import InMemoryClaimQueryStore
from application.readers import (
    CodingPreferencesReader,
    InfraTopologyReader,
    PriorBugsReader,
    TimelineReader,
)
from application.readers._common import ReadRequest
from domain.ports.claim_query import ClaimRow
from domain.ranking import RankingService


_NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _row(
    *,
    pot_id: str = "pot-1",
    predicate: str,
    subject_key: str,
    object_key: str,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    evidence_strength: str = "attested",
    source_system: str = "agent",
    source_ref: str | None = None,
    fact: str | None = None,
    properties: dict | None = None,
) -> ClaimRow:
    return ClaimRow(
        pot_id=pot_id,
        predicate=predicate,
        subject_key=subject_key,
        object_key=object_key,
        valid_at=valid_at or _NOW - timedelta(days=1),
        invalid_at=invalid_at,
        evidence_strength=evidence_strength,
        source_system=source_system,
        source_ref=source_ref or f"src:{predicate}:{subject_key}",
        fact=fact,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# CodingPreferencesReader
# ---------------------------------------------------------------------------


class TestCodingPreferencesReader:
    def _setup_store(self) -> InMemoryClaimQueryStore:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="policy:use-httpx",
                object_key="scope:python",
                fact="use httpx not requests in python projects",
                evidence_strength="stated",
                properties={
                    "code_scope": {"language": "python"},
                    "policy_kind": "library_choice",
                    "strength": "strong",
                },
            )
        )
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="policy:no-eval",
                object_key="scope:any",
                fact="never use eval in javascript",
                evidence_strength="attested",
                properties={
                    "code_scope": {"language": "javascript"},
                    "policy_kind": "security",
                    "strength": "hard",
                },
            )
        )
        return store

    def test_scope_filters_off_irrelevant_policies(self) -> None:
        store = self._setup_store()
        reader = CodingPreferencesReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"language": "python"}, max_items=10)
        )
        keys = [r.candidate.payload["subject_key"] for r in response.items]
        assert "policy:use-httpx" in keys
        assert "policy:no-eval" not in keys

    def test_semantic_query_surfaces_matching_fact(self) -> None:
        store = self._setup_store()
        reader = CodingPreferencesReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"language": "python"},
                query="httpx requests",
                max_items=5,
            )
        )
        assert response.items
        assert "httpx" in (response.items[0].candidate.payload["fact"] or "")

    def test_empty_pool_reports_empty_coverage(self) -> None:
        reader = CodingPreferencesReader(
            claim_query=InMemoryClaimQueryStore(), ranker=RankingService()
        )
        response = reader.read(ReadRequest(pot_id="pot-1", scope={"language": "rust"}))
        assert response.items == ()
        assert response.coverage_status == "empty"


# ---------------------------------------------------------------------------
# InfraTopologyReader (F1)
# ---------------------------------------------------------------------------


class TestInfraTopologyReader:
    def _setup_store(self) -> InMemoryClaimQueryStore:
        store = InMemoryClaimQueryStore()
        # F1 fix: Deployment OF_SERVICE Service + Deployment DEPLOYED_TO Env
        store.add(
            _row(
                predicate="OF_SERVICE",
                subject_key="deployment:k8s:auth:auth-svc",
                object_key="service:auth-svc",
                fact="deployment auth-svc runs service auth-svc",
                evidence_strength="deterministic",
                properties={"environment": "prod"},
            )
        )
        store.add(
            _row(
                predicate="DEPLOYED_TO",
                subject_key="deployment:k8s:auth:auth-svc",
                object_key="environment:prod",
                fact="deployment auth-svc deployed to prod",
                evidence_strength="deterministic",
                properties={"environment": "prod"},
            )
        )
        # A staging deployment of the same service (env-filtered out)
        store.add(
            _row(
                predicate="OF_SERVICE",
                subject_key="deployment:k8s:auth:auth-svc-staging",
                object_key="service:auth-svc",
                fact="deployment auth-svc-staging runs service auth-svc",
                evidence_strength="deterministic",
                properties={"environment": "staging"},
            )
        )
        return store

    def test_f1_service_to_env_link_returned(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        preds = {r.candidate.payload["predicate"] for r in response.items}
        assert "OF_SERVICE" in preds
        assert "DEPLOYED_TO" in preds
        # And we found something — F1 fix proven.
        assert response.coverage_status != "empty"

    def test_environment_filter_excludes_staging(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"], "environment": "prod"},
            )
        )
        envs = {
            r.candidate.payload.get("environment") for r in response.items
        }
        # Only prod should appear; staging filtered.
        assert "staging" not in envs

    def test_no_anchor_returns_neutral_overlap(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(ReadRequest(pot_id="pot-1", scope={}))
        # Unscoped: returns all infra predicates, all neutral overlap
        assert len(response.items) >= 2


# ---------------------------------------------------------------------------
# TimelineReader (F4)
# ---------------------------------------------------------------------------


class TestTimelineReader:
    def _setup_store(self) -> InMemoryClaimQueryStore:
        store = InMemoryClaimQueryStore()
        # F4 fix: MENTIONS provenance — PR Activity mentions service
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:1042",
                object_key="service:auth-svc",
                fact="activity:github:pr:1042 mentions service:auth-svc",
                evidence_strength="attested",
                valid_at=_NOW - timedelta(hours=2),
            )
        )
        # Person authored the activity — non-mentions
        store.add(
            _row(
                predicate="PERFORMED_BY",
                subject_key="activity:github:pr:1042",
                object_key="person:alice",
                fact="alice authored PR 1042",
                evidence_strength="deterministic",
                valid_at=_NOW - timedelta(hours=2),
            )
        )
        # Old activity that mentioned a different service
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:900",
                object_key="service:users-svc",
                fact="activity:github:pr:900 mentions service:users-svc",
                valid_at=_NOW - timedelta(days=14),
            )
        )
        return store

    def test_f4_mentions_link_returns_activity_for_service(self) -> None:
        store = self._setup_store()
        reader = TimelineReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                since=_NOW - timedelta(days=7),
                until=_NOW,
            )
        )
        # The PR-1042 activity (via MENTIONS) is returned
        keys = [r.candidate.payload["subject_key"] for r in response.items]
        assert "activity:github:pr:1042" in keys

    def test_old_activities_outside_window_excluded(self) -> None:
        store = self._setup_store()
        reader = TimelineReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["users-svc"]},
                since=_NOW - timedelta(days=2),
                until=_NOW,
            )
        )
        # PR-900 mentioned users-svc but is 14d old → outside the window
        assert response.coverage_status == "empty"

    def test_freshness_pref_defaults_to_fresh_for_timeline(self) -> None:
        store = self._setup_store()
        # Add a stale + recent activity for the same service
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:1100",
                object_key="service:auth-svc",
                fact="x",
                valid_at=_NOW - timedelta(hours=1),
            )
        )
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:0500",
                object_key="service:auth-svc",
                fact="y",
                valid_at=_NOW - timedelta(days=4),
            )
        )
        reader = TimelineReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        # The freshest activity should top
        assert response.items[0].candidate.payload["subject_key"] in {
            "activity:github:pr:1042",
            "activity:github:pr:1100",
        }


# ---------------------------------------------------------------------------
# PriorBugsReader (UC4)
# ---------------------------------------------------------------------------


class TestPriorBugsReader:
    def _setup_store(self) -> InMemoryClaimQueryStore:
        store = InMemoryClaimQueryStore()
        # A worked fix for the symptom in scope
        store.add(
            _row(
                predicate="RESOLVED",
                subject_key="fix:pool-exhaustion-123",
                object_key="bug_pattern:queuepool",
                fact="connection pool exhausted: raise pool size",
                evidence_strength="attested",
                properties={"scope_keys": ["service:auth-svc"]},
            )
        )
        # A failed attempt — should be labeled
        store.add(
            _row(
                predicate="ATTEMPTED_FIX_FAILED",
                subject_key="fix:pool-restart-456",
                object_key="bug_pattern:queuepool",
                fact="connection pool exhausted: restart didn't help",
                evidence_strength="stated",
                properties={"scope_keys": ["service:auth-svc"]},
            )
        )
        # Two verifications boost the worked fix's corroboration
        for src in ("alice", "bob"):
            store.add(
                _row(
                    predicate="VERIFIED",
                    subject_key="fix:pool-exhaustion-123",
                    object_key=f"person:{src}",
                    fact=f"{src} verified the fix",
                    evidence_strength="attested",
                )
            )
        return store

    def test_worked_fix_ranked_above_failed_attempt(self) -> None:
        store = self._setup_store()
        reader = PriorBugsReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="connection pool exhausted",
            )
        )
        top = response.items[0].candidate.payload
        assert top["predicate"] == "RESOLVED"
        assert top["verification_count"] >= 1

    def test_failed_attempt_labeled(self) -> None:
        store = self._setup_store()
        reader = PriorBugsReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="connection pool exhausted",
            )
        )
        failed = [
            r for r in response.items
            if r.candidate.payload["is_attempted_failed_fix"]
        ]
        assert failed

    def test_narrower_scope_hidden_from_unrelated_scope(self) -> None:
        store = self._setup_store()
        reader = PriorBugsReader(
            claim_query=store, ranker=RankingService()
        )
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["billing-svc"]},
                query="connection pool",
            )
        )
        assert response.coverage_status == "empty"
