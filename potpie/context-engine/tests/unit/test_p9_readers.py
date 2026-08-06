"""P9 use-case readers (rebuild plan P9).

These tests exercise the four UC readers (CodingPreferences, InfraTopology,
Timeline, PriorBugs) against the in-memory claim store. The goal is to
prove (a) the canonical edge-shape feeds through the readers cleanly,
(b) ranking + coverage propagate, (c) scope filters work, and (d) the
F1/F2/F4 fix paths produce the right answers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.readers import (
    CodingPreferencesReader,
    DecisionsReader,
    DocsReader,
    FeaturesReader,
    InfraTopologyReader,
    OwnersReader,
    PriorBugsReader,
    TimelineReader,
)
from potpie_context_engine.application.readers._common import (
    ReadRequest,
    claim_semantic_similarity,
    dedupe_claim_rows,
    row_matches_query,
)
from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_engine.domain.ranking import RankingService


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


def test_claim_semantic_similarity_ignores_booleans() -> None:
    row = _row(
        predicate="DEPENDS_ON",
        subject_key="service:a",
        object_key="service:b",
        properties={"semantic_similarity": True},
    )
    assert claim_semantic_similarity(row) is None
    assert not row_matches_query(row, "unrelated query")


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

    def test_query_filters_out_non_matching_preferences(self) -> None:
        store = self._setup_store()
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"language": "python"},
                query="does-not-exist-12345",
                max_items=5,
            )
        )
        assert response.items == ()
        assert response.coverage_status == "empty"

    def test_query_threshold_can_be_relaxed(self) -> None:
        store = self._setup_store()
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())

        strict = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"language": "python"},
                query="httpx absent",
                max_items=5,
            )
        )
        relaxed = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"language": "python"},
                query="httpx absent",
                query_threshold=0.30,
                max_items=5,
            )
        )

        assert strict.items == ()
        assert relaxed.items
        assert relaxed.items[0].candidate.payload["subject_key"] == "policy:use-httpx"

    def test_empty_pool_reports_empty_coverage(self) -> None:
        reader = CodingPreferencesReader(
            claim_query=InMemoryClaimQueryStore(), ranker=RankingService()
        )
        response = reader.read(ReadRequest(pot_id="pot-1", scope={"language": "rust"}))
        assert response.items == ()
        assert response.coverage_status == "empty"

    def test_repo_path_scope_excludes_conflicting_preferences(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:alpha-pytest",
                object_key="code:github.com/mock/alpha-checkout:src/checkout",
                fact="add pytest tests for checkout changes",
                properties={
                    "code_scope": {
                        "repo": "github.com/mock/alpha-checkout",
                        "service": "checkout-api",
                        "file_path": "src/checkout",
                        "language": "python",
                    }
                },
            )
        )
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:beta-logging",
                object_key="code:github.com/mock/beta-billing:src/billing",
                fact="add structured logging for billing changes",
                properties={
                    "code_scope": {
                        "repo": "github.com/mock/beta-billing",
                        "service": "billing-worker",
                        "file_path": "src/billing",
                        "language": "python",
                    }
                },
            )
        )
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:global-python",
                object_key="scope:any",
                fact="prefer small focused python tests",
            )
        )
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={
                    "repo": "github.com/mock/alpha-checkout",
                    "path": "src/checkout/cache.py",
                    "language": "python",
                },
                max_items=10,
            )
        )

        keys = {r.candidate.payload["subject_key"] for r in response.items}
        assert "preference:alpha-pytest" in keys
        assert "preference:global-python" in keys
        assert "preference:beta-logging" not in keys

    def test_beta_scope_excludes_alpha_preference(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:alpha-pytest",
                object_key="code:github.com/mock/alpha-checkout:src/checkout",
                fact="add pytest tests for checkout changes",
                properties={
                    "code_scope": {
                        "repo": "github.com/mock/alpha-checkout",
                        "file_path": "src/checkout",
                    }
                },
            )
        )
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:beta-logging",
                object_key="code:github.com/mock/beta-billing:src/billing",
                fact="add structured logging for billing changes",
                properties={
                    "code_scope": {
                        "repo": "github.com/mock/beta-billing",
                        "file_path": "src/billing",
                    }
                },
            )
        )
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"repo": "github.com/mock/beta-billing", "path": "src/billing"},
                max_items=10,
            )
        )

        keys = {r.candidate.payload["subject_key"] for r in response.items}
        assert keys == {"preference:beta-logging"}

    def test_conflicting_language_framework_and_service_preferences_are_excluded(
        self,
    ) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:python-fastapi",
                object_key="service:checkout-api",
                fact="python fastapi checkout preference",
                properties={
                    "code_scope": {
                        "service": "checkout-api",
                        "language": "python",
                        "framework": "fastapi",
                    }
                },
            )
        )
        store.add(
            _row(
                predicate="POLICY_APPLIES_TO",
                subject_key="preference:go-chi",
                object_key="service:billing-worker",
                fact="go chi billing preference",
                properties={
                    "code_scope": {
                        "service": "billing-worker",
                        "language": "go",
                        "framework": "chi",
                    }
                },
            )
        )
        reader = CodingPreferencesReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={
                    "service": "checkout-api",
                    "language": "python",
                    "framework": "fastapi",
                },
                max_items=10,
            )
        )

        keys = {r.candidate.payload["subject_key"] for r in response.items}
        assert keys == {"preference:python-fastapi"}


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

    def _setup_semantic_store(self) -> InMemoryClaimQueryStore:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:auth-svc",
                object_key="datastore:redis-cache",
                fact="auth-svc depends on redis cache for session connection pooling",
                evidence_strength="deterministic",
            )
        )
        store.add(
            _row(
                predicate="DEPENDS_ON",
                subject_key="service:auth-svc",
                object_key="service:kafka-broker",
                fact="auth-svc publishes login events to the kafka broker",
                evidence_strength="deterministic",
            )
        )
        return store

    def test_query_makes_anchored_ranking_semantic(self) -> None:
        store = self._setup_semantic_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="redis connection pool for sessions",
            )
        )
        assert len(response.items) == 2
        top, second = response.items
        assert top.candidate.payload["object_key"] == "datastore:redis-cache"
        assert (
            top.breakdown["semantic_similarity"]
            > second.breakdown["semantic_similarity"]
        )

    def test_query_makes_unanchored_ranking_semantic(self) -> None:
        store = self._setup_semantic_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={},
                query="redis connection pool for sessions",
            )
        )
        assert response.items
        top = response.items[0]
        assert top.candidate.payload["object_key"] == "datastore:redis-cache"
        sims = {r.breakdown["semantic_similarity"] for r in response.items}
        assert len(sims) > 1  # not flat-scored

    def test_no_query_keeps_neutral_similarity(self) -> None:
        store = self._setup_semantic_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        assert response.items
        assert all(r.breakdown["semantic_similarity"] == 0.5 for r in response.items)

    def test_query_does_not_change_traversal_discovery(self) -> None:
        store = self._setup_semantic_store()
        reader = InfraTopologyReader(claim_query=store, ranker=RankingService())
        without_query = reader.read(
            ReadRequest(pot_id="pot-1", scope={"services": ["auth-svc"]})
        )
        with_query = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="redis connection pool for sessions",
            )
        )
        assert {r.candidate.candidate_key for r in without_query.items} == {
            r.candidate.candidate_key for r in with_query.items
        }


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

    def test_timeline_scope_filters_by_repo_service_and_path(self) -> None:
        store = _timeline_scope_store()
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        unscoped = reader.read(ReadRequest(pot_id="pot-1", max_items=10))
        assert _timeline_activity_keys(unscoped) == {
            "activity:github:pr:alpha",
            "activity:github:pr:beta",
        }

        by_service = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "checkout-api"}, max_items=10)
        )
        assert _timeline_activity_keys(by_service) == {"activity:github:pr:alpha"}

        by_repo = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"repo": "github.com/mock/alpha-checkout"},
                max_items=10,
            )
        )
        assert _timeline_activity_keys(by_repo) == {"activity:github:pr:alpha"}

        by_path = reader.read(
            ReadRequest(pot_id="pot-1", scope={"path": "src/checkout"}, max_items=10)
        )
        assert _timeline_activity_keys(by_path) == {"activity:github:pr:alpha"}

        by_repo_and_path = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={
                    "repo": "github.com/mock/alpha-checkout",
                    "path": "src/checkout",
                },
                max_items=10,
            )
        )
        assert _timeline_activity_keys(by_repo_and_path) == {"activity:github:pr:alpha"}

        beta = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"repo": "github.com/mock/beta-billing", "path": "src/billing"},
                max_items=10,
            )
        )
        assert _timeline_activity_keys(beta) == {"activity:github:pr:beta"}

    def test_timeline_anchor_scope_survives_unrelated_activity_cap(self) -> None:
        """Scoped repo reads must not drop matches when unrelated rows fill the cap."""
        store = InMemoryClaimQueryStore()
        for index in range(250):
            store.add(
                _row(
                    predicate="TOUCHED",
                    subject_key=f"activity:github:pr:filler-{index}",
                    object_key="repo:github.com/mock/unrelated-repo",
                    fact=f"filler activity {index}",
                    valid_at=_NOW - timedelta(hours=1),
                    properties={"occurred_at": "2026-07-03T12:00:00+00:00"},
                )
            )
        store.add(
            _row(
                predicate="TOUCHED",
                subject_key="activity:github:pr:alpha",
                object_key="repo:github.com/mock/alpha-checkout",
                fact="alpha checkout latency change",
                valid_at=_NOW - timedelta(hours=3),
                properties={"occurred_at": "2026-07-01T10:00:00+00:00"},
            )
        )
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"repo": "github.com/mock/alpha-checkout"},
                max_items=10,
            )
        )

        assert _timeline_activity_keys(response) == {"activity:github:pr:alpha"}

    def test_timeline_scope_includes_performed_edges(self) -> None:
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="PERFORMED",
                subject_key="repo:github.com/mock/alpha-checkout",
                object_key="activity:github:pr:alpha",
                fact="alpha checkout team performed the activity",
            )
        )
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"repo": "github.com/mock/alpha-checkout"},
                max_items=10,
            )
        )

        assert _timeline_activity_keys(response) == {"activity:github:pr:alpha"}

    def test_timeline_source_ref_filter_still_narrows_events(self) -> None:
        store = _timeline_scope_store()
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                source_refs=("mock:github:mock/alpha-checkout#pr/101",),
                max_items=10,
            )
        )

        assert _timeline_activity_keys(response) == {"activity:github:pr:alpha"}

    def test_timeline_query_filters_activity_groups(self) -> None:
        store = _timeline_scope_store()
        reader = TimelineReader(claim_query=store, ranker=RankingService())

        alpha = reader.read(
            ReadRequest(pot_id="pot-1", query="ALPHA_SENTINEL", max_items=10)
        )
        assert _timeline_activity_keys(alpha) == {"activity:github:pr:alpha"}

        nonsense = reader.read(
            ReadRequest(pot_id="pot-1", query="does-not-exist-12345", max_items=10)
        )
        assert nonsense.items == ()
        assert nonsense.coverage_status == "empty"


def _timeline_scope_store() -> InMemoryClaimQueryStore:
    store = InMemoryClaimQueryStore()
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:alpha",
            object_key="repo:github.com/mock/alpha-checkout",
            fact="ALPHA_SENTINEL checkout latency change",
            valid_at=_NOW - timedelta(hours=3),
            source_ref="mock:github:mock/alpha-checkout#pr/101",
            properties={"occurred_at": "2026-07-01T10:00:00+00:00"},
        )
    )
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:alpha",
            object_key="service:checkout-api",
            fact="ALPHA_SENTINEL touched checkout service",
            valid_at=_NOW - timedelta(hours=3),
            source_ref="mock:github:mock/alpha-checkout#pr/101",
            properties={"occurred_at": "2026-07-01T10:00:00+00:00"},
        )
    )
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:alpha",
            object_key="code:github.com/mock/alpha-checkout:src/checkout/cache.py",
            fact="ALPHA_SENTINEL touched checkout cache",
            valid_at=_NOW - timedelta(hours=3),
            source_ref="mock:github:mock/alpha-checkout#pr/101",
            properties={"occurred_at": "2026-07-01T10:00:00+00:00"},
        )
    )
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:beta",
            object_key="repo:github.com/mock/beta-billing",
            fact="BETA_SENTINEL invoice retry change",
            valid_at=_NOW - timedelta(hours=2),
            source_ref="mock:github:mock/beta-billing#pr/202",
            properties={"occurred_at": "2026-07-02T11:00:00+00:00"},
        )
    )
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:beta",
            object_key="service:billing-worker",
            fact="BETA_SENTINEL touched billing service",
            valid_at=_NOW - timedelta(hours=2),
            source_ref="mock:github:mock/beta-billing#pr/202",
            properties={"occurred_at": "2026-07-02T11:00:00+00:00"},
        )
    )
    store.add(
        _row(
            predicate="TOUCHED",
            subject_key="activity:github:pr:beta",
            object_key="code:github.com/mock/beta-billing:src/billing/retry.py",
            fact="BETA_SENTINEL touched billing retry",
            valid_at=_NOW - timedelta(hours=2),
            source_ref="mock:github:mock/beta-billing#pr/202",
            properties={"occurred_at": "2026-07-02T11:00:00+00:00"},
        )
    )
    return store


def _timeline_activity_keys(response) -> set[str]:
    return {item.candidate.payload["activity_key"] for item in response.items}


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

    def test_docs_reader_returns_document_sections_and_their_parent(self) -> None:
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1",
            entity_key="docsection:q3-review:capacity",
            labels=("DocumentSection",),
        )
        store.add(
            _row(
                predicate="DOCUMENTS",
                subject_key="docsection:q3-review:capacity",
                object_key="service:context-engine",
                fact="Q3 capacity planning for the context engine",
            )
        )
        store.add(
            _row(
                predicate="SECTION_OF",
                subject_key="docsection:q3-review:capacity",
                object_key="document:q3-review",
                fact="capacity section of the Q3 review",
            )
        )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "context-engine"})
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        # The section is found on its own label, then expanded one hop so the
        # result carries the document it came from.
        assert {"DOCUMENTS", "SECTION_OF"} <= predicates

    def test_docs_reader_returns_sections_with_their_chunk_ids(self) -> None:
        """An unscoped read is search-then-get in two calls: the section claim
        an import wrote comes back already holding the ids ``resource get``
        takes, so no ``resource list`` hop sits in between (R13)."""
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1",
            entity_key="docsection:q3-review:capacity",
            labels=("DocumentSection",),
        )
        store.add(
            _row(
                predicate="SECTION_OF",
                subject_key="docsection:q3-review:capacity",
                object_key="document:q3-review",
                fact="capacity planning for Q3, by team",
                source_ref="potpie://res/q3-review/capacity/0000",
            )
        )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(ReadRequest(pot_id="pot-1", query="capacity planning"))

        assert response.items
        payload = response.items[0].candidate.payload
        assert payload["subject_key"] == "docsection:q3-review:capacity"
        assert payload["chunk_ids"] == ["potpie://res/q3-review/capacity/0000"]

    def test_docs_reader_omits_chunk_ids_when_a_claim_has_none(self) -> None:
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1", entity_key="document:plain", labels=("Document",)
        )
        store.add(
            _row(
                predicate="DOCUMENTS",
                subject_key="document:plain",
                object_key="service:context-engine",
                fact="a document with no stored chunks",
            )
        )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "context-engine"})
        )

        assert "chunk_ids" not in response.items[0].candidate.payload

    def test_docs_reader_still_reads_legacy_related_to_doc_claims(self) -> None:
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1", entity_key="document:legacy-note", labels=("Document",)
        )
        store.add(
            _row(
                predicate="RELATED_TO",
                subject_key="document:legacy-note",
                object_key="service:context-engine",
                fact="note written before DOCUMENTS existed",
            )
        )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "context-engine"})
        )

        assert [r.candidate.payload["subject_key"] for r in response.items] == [
            "document:legacy-note"
        ]

    def test_docs_reader_returns_one_item_when_both_spellings_exist(self) -> None:
        # Nothing migrates the RELATED_TO fallback, so a document can carry
        # both predicates for one scope. That is one document, not two.
        store = InMemoryClaimQueryStore()
        store.set_entity_label(
            pot_id="pot-1", entity_key="document:runbook", labels=("Document",)
        )
        for predicate in ("RELATED_TO", "DOCUMENTS"):
            store.add(
                _row(
                    predicate=predicate,
                    subject_key="document:runbook",
                    object_key="service:payments-api",
                    fact="runbook for the payments api",
                )
            )
        reader = DocsReader(claim_query=store, ranker=RankingService())

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "payments-api"})
        )

        assert [r.candidate.payload["predicate"] for r in response.items] == [
            "DOCUMENTS"
        ]


# ---------------------------------------------------------------------------
# DocsReader relevance floor
#
# A KNN returns k rows whether or not any of them answers the query, so a docs
# read without a floor pads its reply to the limit with whatever the corpus
# holds. Measured on the resource corpus before this landed: a database
# incident question returned 12 hits spanning 0.021 of score, all from a cloud
# cost spreadsheet, with the runbook section holding the literal error string
# absent entirely.
#
# Similarities are set explicitly here rather than computed, because the whole
# point is behaviour across a *scale*: real scores on that corpus top out near
# 0.60 and clamp to exactly 0.0 at orthogonal.
# ---------------------------------------------------------------------------


def _scored(row: ClaimRow, similarity: float | None) -> ClaimRow:
    properties = dict(row.properties)
    if similarity is None:
        properties.pop("semantic_similarity", None)
    else:
        properties["semantic_similarity"] = similarity
    return replace(row, properties=properties)


def _doc_label(entity_key: str) -> str | None:
    if entity_key.startswith("document:"):
        return "Document"
    if entity_key.startswith("docsection:"):
        return "DocumentSection"
    return None


class _StubDocClaimQuery:
    """Claim-query stub with per-row control over the stamped similarity.

    Mirrors the one backend behaviour that matters here: a query carries a
    similarity onto every row it returns, and a lookup **by key** does not —
    which is why the reader's one-hop ``SECTION_OF`` expansion arrives
    unscored.
    """

    def __init__(self, rows: Sequence[tuple[ClaimRow, float | None]]) -> None:
        self._rows = list(rows)

    def find_claims(self, filter_) -> list[ClaimRow]:
        out: list[ClaimRow] = []
        for row, similarity in self._rows:
            if filter_.predicate_in and row.predicate not in filter_.predicate_in:
                continue
            if filter_.subject_key_in and row.subject_key not in filter_.subject_key_in:
                continue
            if filter_.object_key_in and row.object_key not in filter_.object_key_in:
                continue
            if (
                filter_.subject_label
                and _doc_label(row.subject_key) != filter_.subject_label
            ):
                continue
            out.append(_scored(row, similarity if filter_.fact_query else None))
        return out

    def entity_labels(self, *, pot_id: str, entity_keys):  # pragma: no cover - unused
        return {}


def _docs_reader(rows: Sequence[tuple[ClaimRow, float | None]]) -> DocsReader:
    return DocsReader(claim_query=_StubDocClaimQuery(rows), ranker=RankingService())


def _documents_row(subject_key: str, fact: str) -> ClaimRow:
    return _row(
        predicate="DOCUMENTS",
        subject_key=subject_key,
        object_key="service:payments-api",
        fact=fact,
    )


class TestDocsReaderRelevanceFloor:
    def test_drops_rows_far_below_the_best_match_in_the_pool(self) -> None:
        best = _documents_row("docsection:alpha:one", "quarterly capacity forecast")
        near = _documents_row("docsection:alpha:two", "capacity planning notes")
        filler = _documents_row("docsection:beta:one", "vendor invoice totals")
        reader = _docs_reader([(best, 0.60), (near, 0.31), (filler, 0.10)])

        response = reader.read(
            ReadRequest(pot_id="pot-1", query="quarterly capacity forecast")
        )

        # Floor is half of the pool's best (0.60) — 0.31 clears it, 0.10 does not.
        assert [r.candidate.payload["subject_key"] for r in response.items] == [
            "docsection:alpha:one",
            "docsection:alpha:two",
        ]

    def test_keeps_an_exact_lexical_match_below_the_floor(self) -> None:
        """The strongest relevance signal is the weakest embedding signal.

        A rare identifier is one token to a reader and noise to a sentence
        embedder, so a verbatim hit survives on its text even when the vector
        channel scored it near zero.
        """
        best = _documents_row("docsection:alpha:one", "connection pooling overview")
        verbatim = _documents_row(
            "docsection:beta:one",
            "raise pool_max_conns when DBConnectionPoolExhausted fires",
        )
        reader = _docs_reader([(best, 0.60), (verbatim, 0.02)])

        response = reader.read(
            ReadRequest(pot_id="pot-1", query="DBConnectionPoolExhausted")
        )

        assert "docsection:beta:one" in {
            r.candidate.payload["subject_key"] for r in response.items
        }

    def test_returns_nothing_when_the_whole_pool_scored_zero(self) -> None:
        """A tail of exact zeros is not a weak answer, it is no answer.

        Every row at or past orthogonal clamps to 0.0, so a purely relative
        floor would admit the entire tail on the strength of its own worst
        member.
        """
        rows = [
            (_documents_row(f"docsection:alpha:{i}", f"unrelated section {i}"), 0.0)
            for i in range(3)
        ]
        reader = _docs_reader(rows)

        response = reader.read(ReadRequest(pot_id="pot-1", query="how to make a cake"))

        assert response.items == ()
        assert response.coverage_status == "empty"

    def test_no_query_returns_the_whole_pool(self) -> None:
        rows = [
            (_documents_row(f"docsection:alpha:{i}", f"section {i}"), None)
            for i in range(3)
        ]
        reader = _docs_reader(rows)

        response = reader.read(
            ReadRequest(pot_id="pot-1", scope={"service": "payments-api"})
        )

        assert len(response.items) == 3


class TestDocsReaderStructuralRows:
    """The one-hop ``SECTION_OF`` expansion is fetched by key, never scored.

    The query here deliberately shares no token with either row, so these
    exercise the floor rather than the lexical rescue.
    """

    QUERY = "database connection pooling"

    def _rows(
        self, section_similarity: float
    ) -> Sequence[tuple[ClaimRow, float | None]]:
        section = _documents_row("docsection:alpha:one", "quarterly capacity forecast")
        structure = _row(
            predicate="SECTION_OF",
            subject_key="docsection:alpha:one",
            object_key="document:alpha",
            fact="capacity forecast section",
            source_ref="potpie://res/alpha/one/0000",
        )
        return [(section, section_similarity), (structure, None)]

    def test_kept_when_its_section_cleared_the_floor(self) -> None:
        """Dropping it would take the section's fetchable chunk ids with it."""
        reader = _docs_reader(self._rows(0.60))

        response = reader.read(ReadRequest(pot_id="pot-1", query=self.QUERY))

        predicates = [r.candidate.payload["predicate"] for r in response.items]
        assert "SECTION_OF" in predicates
        chunk_ids = [
            ref
            for item in response.items
            for ref in item.candidate.payload.get("chunk_ids", [])
        ]
        assert "potpie://res/alpha/one/0000" in chunk_ids

    def test_dropped_when_its_section_did_not(self) -> None:
        reader = _docs_reader(self._rows(0.0))

        response = reader.read(ReadRequest(pot_id="pot-1", query=self.QUERY))

        assert response.items == ()

    def test_is_scored_as_its_section_not_as_unknown(self) -> None:
        """Regression: an unscored row takes the ranker's neutral 0.5 default.

        On a corpus whose real scores top out near 0.60 that puts "never
        compared to the query" above every row that was compared and matched —
        measured, four such rows held the top of a scoped read. It was admitted
        because its section matched, so it is scored the same way.
        """
        reader = _docs_reader(self._rows(0.60))

        response = reader.read(ReadRequest(pot_id="pot-1", query=self.QUERY))

        structural = next(
            item
            for item in response.items
            if item.candidate.payload["predicate"] == "SECTION_OF"
        )
        assert structural.breakdown["semantic_similarity"] == 0.60


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

    def test_section_claims_do_not_crowd_or_leak_into_prior_bugs(self) -> None:
        # P9: a dense knowledge/SECTION_OF corpus that shares symptom text with
        # a real bug must neither appear as prior_bugs items nor empty the
        # reader when the bug claim exists.
        store = InMemoryClaimQueryStore()
        store.add(
            _row(
                predicate="REPRODUCES",
                subject_key="bug_pattern:pool-timeout",
                object_key="service:auth-svc",
                fact="connection pool exhausted under load timeout",
                evidence_strength="attested",
                subgraph="debugging",
                properties={"scope_keys": ["service:auth-svc"]},
            )
        )
        store.add(
            _row(
                predicate="RESOLVED",
                subject_key="fix:pool-timeout",
                object_key="bug_pattern:pool-timeout",
                fact="raise the pool size when connection pool exhausted",
                evidence_strength="attested",
                subgraph="debugging",
            )
        )
        for i in range(40):
            store.add(
                _row(
                    predicate="SECTION_OF",
                    subject_key=f"docsection:ops-runbook:pool-{i}",
                    object_key="document:ops-runbook",
                    fact="connection pool exhausted under load timeout",
                    evidence_strength="stated",
                    subgraph="knowledge",
                )
            )
            store.set_entity_label(
                pot_id="pot-1",
                entity_key=f"docsection:ops-runbook:pool-{i}",
                labels=("Entity", "DocumentSection"),
            )

        reader = PriorBugsReader(claim_query=store, ranker=RankingService())
        response = reader.read(
            ReadRequest(
                pot_id="pot-1",
                scope={"services": ["auth-svc"]},
                query="connection pool exhausted under load timeout",
                max_items=8,
            )
        )

        predicates = {r.candidate.payload["predicate"] for r in response.items}
        assert "SECTION_OF" not in predicates
        assert "DOCUMENTS" not in predicates
        assert {"REPRODUCES", "RESOLVED"} & predicates
        assert response.coverage_status != "empty"
