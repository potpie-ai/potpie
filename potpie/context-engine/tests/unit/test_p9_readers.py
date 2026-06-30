"""P9 use-case readers (rebuild plan P9).

These tests exercise the four UC readers (CodingPreferences, InfraTopology,
Timeline, PriorBugs) against the in-memory claim store. The goal is to
prove (a) the canonical edge-shape feeds through the readers cleanly,
(b) ranking + coverage propagate, (c) scope filters work, and (d) the
F1/F2/F4 fix paths produce the right answers.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from application.readers import (
    CodingPreferencesReader,
    DecisionsReader,
    DocsReader,
    FeaturesReader,
    InfraTopologyReader,
    OwnersReader,
    PriorBugsReader,
    TimelineReader,
)
from application.readers._common import ReadRequest, dedupe_claim_rows
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
    claim_key: str | None = None,
    subgraph: str | None = None,
    truth: str = "agent_claim",
    environment: str | None = None,
    properties: dict | None = None,
) -> ClaimRow:
    ref = source_ref or f"src:{predicate}:{subject_key}"
    return ClaimRow(
        pot_id=pot_id,
        predicate=predicate,
        subject_key=subject_key,
        object_key=object_key,
        valid_at=valid_at or _NOW - timedelta(days=1),
        invalid_at=invalid_at,
        evidence_strength=evidence_strength,
        source_system=source_system,
        source_ref=ref,
        fact=fact,
        properties=properties or {},
        claim_key=claim_key or f"claim:{predicate}:{subject_key}:{object_key}",
        subgraph=subgraph,
        truth=truth,
        environment=environment,
        source_refs=(ref,),
    )


def test_dedupe_claim_rows_uses_claim_key_then_triple_and_sources() -> None:
    first = replace(
        _row(
            predicate="RESOLVED",
            subject_key="fix:pool",
            object_key="bug_pattern:pool",
            claim_key="claim:resolved",
            source_ref="src:first",
        ),
        claim_key=None,
    )
    duplicate = replace(first, fact="same claim from duplicate backend row")
    distinct_source = replace(
        first,
        source_ref="src:second",
        source_refs=("src:second",),
        fact="same triple from another source",
    )

    assert dedupe_claim_rows([first, duplicate, distinct_source]) == [
        first,
        distinct_source,
    ]


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
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"language": "python"}, max_items=10)
        )
        keys = [r.candidate.payload["subject_key"] for r in response.items]
        assert "policy:use-httpx" in keys
        assert "policy:no-eval" not in keys

    def test_semantic_query_surfaces_matching_fact(self) -> None:
        store = self._setup_store()
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())
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
        # Topology core: Service DEPLOYED_TO Environment, env stamped on edge.
        store.add(
            _row(
                predicate="DEPLOYED_TO",
                subject_key="service:auth-svc",
                object_key="environment:prod",
                fact="service auth-svc deployed to prod",
                evidence_strength="deterministic",
                truth="source_observation",
                environment="prod",
            )
        )
        # Same service in staging (env-filtered out for prod queries).
        store.add(
            _row(
                predicate="DEPLOYED_TO",
                subject_key="service:auth-svc",
                object_key="environment:staging",
                fact="service auth-svc deployed to staging",
                evidence_strength="deterministic",
                truth="source_observation",
                environment="staging",
            )
        )
        # An extra topology edge: Service USES DataStore.
        store.add(
            _row(
                predicate="USES",
                subject_key="service:auth-svc",
                object_key="datastore:auth-pg",
                fact="service auth-svc uses datastore auth-pg",
                evidence_strength="deterministic",
                properties={},
            )
        )
        return store

    def test_f1_service_to_env_link_returned(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        preds = {r.candidate.payload["predicate"] for r in response.items}
        assert "DEPLOYED_TO" in preds
        # Service → Environment is a direct edge now (no Deployment node).
        assert response.coverage_status != "empty"

    def test_environment_filter_excludes_staging(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"], "environment": "prod"},
            )
        )
        envs = {r.candidate.payload.get("environment") for r in response.items}
        # Only prod should appear; staging and unqualified rows are filtered.
        assert envs == {"prod"}

    def test_environment_filter_can_include_unqualified_when_explicit(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={
                    "services": ["auth-svc"],
                    "environment": "prod",
                    "include_unqualified_environment": True,
                },
            )
        )
        envs = {r.candidate.payload.get("environment") for r in response.items}
        assert "prod" in envs
        assert None in envs
        assert "staging" not in envs

    def test_environment_filter_applies_during_traversal(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:ledger-api",
                object_key="service:cache",
                fact="ledger depends on cache in prod",
                environment="prod",
            )
        )
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:ledger-api",
                object_key="service:queue",
                fact="ledger depends on queue without an environment qualifier",
                properties={},
            )
        )
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:ledger-api",
                object_key="service:staging-worker",
                fact="ledger depends on staging worker",
                environment="staging",
            )
        )
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:queue",
                object_key="service:worker",
                fact="queue depends on worker in prod",
                environment="prod",
            )
        )
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"service": "ledger-api", "environment": "Prod"},
                depth=2,
                direction="out",
            )
        )

        endpoints = {
            (
                r.candidate.payload["subject_key"],
                r.candidate.payload["object_key"],
                r.candidate.payload.get("environment"),
            )
            for r in response.items
        }
        assert ("service:ledger-api", "service:cache", "prod") in endpoints
        assert all("service:queue" not in pair[:2] for pair in endpoints)
        assert all("service:staging-worker" not in pair[:2] for pair in endpoints)

    def test_no_anchor_returns_neutral_overlap(self) -> None:
        store = self._setup_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(ReadRequest(pot_id="pot-1", scope={}))
        # Unscoped: returns all infra predicates, all neutral overlap
        assert len(response.items) >= 2


# ---------------------------------------------------------------------------
# FeaturesReader
# ---------------------------------------------------------------------------


class TestFeaturesReader:
    def test_returns_only_feature_claims_for_anchor(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="PROVIDES",
                subject_key="repo:github.com/acme/widgets",
                object_key="feature:search",
                fact="widgets repo provides search",
                evidence_strength="deterministic",
            )
        )
        store.add(
            _row(
                predicate="DEFINED_IN",
                subject_key="service:search-api",
                object_key="repo:github.com/acme/widgets",
                fact="search api lives in widgets repo",
                evidence_strength="deterministic",
            )
        )
        reader = FeaturesReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"anchor_entity_key": "repo:github.com/acme/widgets"},
            )
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        assert predicates == {"PROVIDES"}


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
        reader = TimelineReader(claim_query=store, ranker=RankingService())
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
        reader = TimelineReader(claim_query=store, ranker=RankingService())
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

    def test_window_uses_source_occurred_at_when_valid_at_is_ingestion_time(
        self,
    ) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="TOUCHED",
                subject_key="activity:github:pr:1043",
                object_key="service:auth-svc",
                fact="PR 1043 touched auth on 2026-05-10",
                valid_at=_NOW,
                properties={"occurred_at": "2026-05-10T12:00:00+00:00"},
            )
        )
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"service": "auth-svc"},
                since=datetime(2026, 5, 10, tzinfo=timezone.utc),
                until=datetime(2026, 5, 11, tzinfo=timezone.utc),
            )
        )

        assert response.items
        payload = response.items[0].candidate.payload
        assert payload["occurred_at"] == "2026-05-10T12:00:00+00:00"

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
        reader = TimelineReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        # The freshest activity should top
        assert response.items[0].candidate.payload["subject_key"] in {
            "activity:github:pr:1042",
            "activity:github:pr:1100",
        }

    def test_query_mode_prioritizes_relevance_over_recency(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:new",
                object_key="service:auth-svc",
                fact="dependency maintenance and whitespace cleanup",
                valid_at=_NOW - timedelta(hours=1),
            )
        )
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:old",
                object_key="service:auth-svc",
                fact="token refresh race caused oauth callback failures",
                valid_at=_NOW - timedelta(days=10),
            )
        )
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"service": "auth-svc"},
                query="oauth callback token refresh race",
                max_items=2,
            )
        )

        assert (
            response.items[0].candidate.payload["subject_key"]
            == "activity:github:pr:old"
        )

    def test_timeline_dedupes_edges_per_activity(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="MENTIONS",
                subject_key="activity:github:pr:dupe",
                object_key="service:auth-svc",
                fact="PR mentions auth service",
                valid_at=_NOW - timedelta(hours=2),
            )
        )
        store.add(
            _row(
                predicate="TOUCHED",
                subject_key="activity:github:pr:dupe",
                object_key="service:auth-svc",
                fact="PR touched auth service",
                valid_at=_NOW - timedelta(hours=2),
            )
        )
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "auth-svc"}, max_items=10)
        )

        assert len(response.items) == 1
        payload = response.items[0].candidate.payload
        assert payload["activity_key"] == "activity:github:pr:dupe"
        assert payload["properties"]["activity_edge_count"] == 2


# ---------------------------------------------------------------------------
# Decisions / Owners / Docs readers
# ---------------------------------------------------------------------------


class TestNewUseCaseReaders:
    def test_decisions_reader_returns_scope_and_affected_claims(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="DECIDED",
                subject_key="decision:use-neo4j",
                object_key="service:context-engine",
                fact="Use Neo4j for the context graph backend",
            )
        )
        store.add(
            _row(
                predicate="AFFECTS",
                subject_key="decision:use-neo4j",
                object_key="code:context-engine:graph",
                fact="Neo4j decision affects graph adapter code",
            )
        )
        reader = DecisionsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1", scope={"service": "context-engine"}, max_items=10
            )
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        assert {"DECIDED", "AFFECTS"} <= predicates

    def test_owners_reader_returns_owner_and_team_context(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="OWNED_BY",
                subject_key="service:context-engine",
                object_key="person:alice",
                fact="context engine is owned by alice",
            )
        )
        store.add(
            _row(
                predicate="MEMBER_OF",
                subject_key="person:alice",
                object_key="team:platform",
                fact="alice is on the platform team",
            )
        )
        reader = OwnersReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1", scope={"service": "context-engine"}, max_items=10
            )
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        assert {"OWNED_BY", "MEMBER_OF"} <= predicates

    def test_docs_reader_returns_document_references_for_scope(self) -> None:
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1", entity_key="document:graph-runbook", labels=("Document",)
        )
        store.add(
            _row(
                predicate="RELATED_TO",
                subject_key="document:graph-runbook",
                object_key="service:context-engine",
                fact="Graph runbook documents context engine operations",
            )
        )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"service": "context-engine"},
                query="graph runbook",
            )
        )

        assert response.items
        assert (
            response.items[0].candidate.payload["subject_key"]
            == "document:graph-runbook"
        )


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
        reader = PriorBugsReader(claim_query=store, ranker=RankingService())
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
        reader = PriorBugsReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="connection pool exhausted",
            )
        )
        failed = [
            r for r in response.items if r.candidate.payload["is_attempted_failed_fix"]
        ]
        assert failed

    def test_narrower_scope_hidden_from_unrelated_scope(self) -> None:
        store = self._setup_store()
        reader = PriorBugsReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["billing-svc"]},
                query="connection pool",
            )
        )
        assert response.coverage_status == "empty"

    def test_matching_reproduction_expands_to_known_fix(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="REPRODUCES",
                subject_key="bug_pattern:ambiguous-pot",
                object_key="service:context-engine",
                fact="ambiguous pot scope causes graph read to fail",
                evidence_strength="attested",
            )
        )
        store.add(
            _row(
                predicate="RESOLVED",
                subject_key="fix:explicit-pot",
                object_key="bug_pattern:ambiguous-pot",
                fact="pass --pot or use active pot resolution to fix ambiguous scope",
                evidence_strength="attested",
            )
        )
        reader = PriorBugsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"service": "context-engine"},
                query="graph read fails ambiguous pot current repo",
                max_items=10,
            )
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        assert {"REPRODUCES", "RESOLVED"} <= predicates
