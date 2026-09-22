"""Native W5 acceptance across every locally runnable graph backend.

The assertions here deliberately cross the mutation, storage, query, and
timeline-reader boundaries.  Server profiles are opt-in and must point at an
isolated test database; the local profiles always receive fresh storage.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_core.semantic_mutation_lowering import lower_semantic_request
from potpie_context_core.semantic_mutation_validator import validate_semantic_request
from potpie_context_core.semantic_mutations import SemanticMutationRequest
from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.timeline_reader import TimelineReader
from potpie_context_engine.domain.ranking import RankingService


PROFILES = ("in_memory", "embedded", "falkordb_lite", "neo4j")


@pytest.fixture(params=PROFILES)
def temporal_backend(request, tmp_path, monkeypatch):
    profile = request.param
    pot_id = "w5-temporal-" + uuid.uuid4().hex
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv(
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "w5_temporal_" + uuid.uuid4().hex
    )
    if profile == "falkordb_lite":
        pytest.importorskip("redislite.falkordb_client")
    if profile == "neo4j":
        endpoint = os.environ.get("PROTOCOL_TEST_NEO4J_URI")
        if not endpoint:
            pytest.skip("PROTOCOL_TEST_NEO4J_URI must name an isolated test server")
        monkeypatch.setenv("CONTEXT_ENGINE_NEO4J_URI", endpoint)
        monkeypatch.setenv(
            "CONTEXT_ENGINE_NEO4J_USERNAME",
            os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
        )
        monkeypatch.setenv(
            "CONTEXT_ENGINE_NEO4J_PASSWORD",
            os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", "test"),
        )

    backend = (
        EmbeddedGraphBackend(home=tmp_path / "embedded")
        if profile == "embedded"
        else build_backend(profile)
    )
    if profile == "neo4j":
        assert backend.mutation.reset_pot(pot_id)["ok"]
    try:
        yield profile, pot_id, backend
    finally:
        if profile == "neo4j":
            assert backend.mutation.reset_pot(pot_id)["ok"]
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()


def _edge(
    claim_key: str,
    *,
    subject: str,
    object_: str,
    valid_at: datetime,
    valid_until: datetime | None = None,
    environment: str | None = None,
    source_ref: str | None = None,
    occurred_at: datetime | None = None,
) -> EdgeUpsert:
    properties = {
        "claim_key": claim_key,
        "subgraph": "recent_changes",
        "truth": "authoritative_fact",
        "fact": claim_key,
        "description": claim_key,
        "source_ref": source_ref or f"test:{claim_key}",
        "source_refs": [source_ref or f"test:{claim_key}"],
        "valid_at": valid_at.isoformat(),
        "valid_from": valid_at.isoformat(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    if valid_until is not None:
        properties["valid_until"] = valid_until.isoformat()
    if environment is not None:
        properties["environment"] = environment
    if occurred_at is not None:
        properties["occurred_at"] = occurred_at.isoformat()
    return EdgeUpsert(
        edge_type="TOUCHED",
        from_entity_key=subject,
        to_entity_key=object_,
        properties=properties,
    )


def _apply(backend, pot_id: str, *, edges=(), invalidations=()) -> None:
    entity_keys = {
        key
        for edge in edges
        for key in (edge.from_entity_key, edge.to_entity_key)
    }
    result = backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    entity_key=key,
                    labels=("Activity",)
                    if key.startswith("activity:")
                    else ("Repository",)
                    if key.startswith("repo:")
                    else ("Service",),
                )
                for key in sorted(entity_keys)
            ],
            edge_upserts=list(edges),
            invalidations=list(invalidations),
        ),
        expected_pot_id=pot_id,
    )
    assert result.ok, result.error


def _keys(backend, pot_id: str, *, as_of: datetime | None = None) -> set[str]:
    return {
        row.claim_key
        for row in backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=pot_id, as_of=as_of)
        )
    }


def test_current_claims_obey_half_open_validity_and_scoped_end(temporal_backend):
    _, pot_id, backend = temporal_backend
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=2)
    end = now + timedelta(days=2)
    common_subject = "activity:deploy:checkout"
    common_object = "repo:github.com/acme/widgets"
    edges = (
        _edge("claim:current", subject="activity:current", object_=common_object, valid_at=start),
        _edge("claim:future-start", subject="activity:future", object_=common_object, valid_at=end),
        _edge(
            "claim:valid-until-future",
            subject="activity:scheduled-end",
            object_=common_object,
            valid_at=start,
            valid_until=end,
        ),
        _edge(
            "claim:prod",
            subject=common_subject,
            object_=common_object,
            valid_at=start,
            environment="prod",
        ),
        _edge(
            "claim:staging",
            subject=common_subject,
            object_=common_object,
            valid_at=start,
            environment="staging",
        ),
        _edge("claim:expired", subject="activity:expired", object_=common_object, valid_at=start),
    )
    _apply(backend, pot_id, edges=edges)
    _apply(
        backend,
        pot_id,
        invalidations=(
            InvalidationOp(
                target_entity_key=None,
                target_edge=None,
                target_claim_keys=("claim:prod",),
                reason="production rollout ends on schedule",
                valid_to=end.isoformat(),
            ),
            InvalidationOp(
                target_entity_key=None,
                target_edge=None,
                target_claim_keys=("claim:expired",),
                reason="fact expired before the current read",
                valid_to=(now - timedelta(days=1)).isoformat(),
            ),
        ),
    )

    current = _keys(backend, pot_id)
    assert "claim:current" in current
    assert "claim:valid-until-future" in current
    assert "claim:prod" in current
    assert "claim:staging" in current
    assert "claim:future-start" not in current
    assert "claim:expired" not in current

    immediately_before_end = _keys(backend, pot_id, as_of=end - timedelta(microseconds=1))
    at_end = _keys(backend, pot_id, as_of=end)
    assert {"claim:valid-until-future", "claim:prod"} <= immediately_before_end
    assert "claim:valid-until-future" not in at_end
    assert "claim:prod" not in at_end
    assert "claim:staging" in at_end
    assert "claim:future-start" in at_end


def test_offset_scheduled_correction_is_normalized_before_native_query(
    temporal_backend,
):
    _, pot_id, backend = temporal_backend
    subject = "service:api"
    object_ = "service:ledger"
    source_ref = "fixture:offset-schedule"
    _apply(
        backend,
        pot_id,
        edges=(
            EdgeUpsert(
                "DEPENDS_ON",
                subject,
                object_,
                {
                    "claim_key": "claim:offset-schedule",
                    "subgraph": "infra_topology",
                    "truth": "source_observation",
                    "source_ref": source_ref,
                    "source_refs": [source_ref],
                    "fact": "API depends on ledger",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                    "valid_from": "2026-01-01T00:00:00+00:00",
                },
            ),
        ),
    )
    request = SemanticMutationRequest.parse(
        {
            "operations": [
                {
                    "op": "end_relation_validity",
                    "subgraph": "infra_topology",
                    "subject": {"key": subject, "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": object_, "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": source_ref}],
                    "reason": "scheduled in source timezone",
                    "valid_until": "2099-01-01T05:30:00+05:30",
                }
            ]
        },
        pot_id=pot_id,
        allow_review_required=True,
        approved_by="test:reviewer",
    )
    plan = validate_semantic_request(request, claim_query=backend.claim_query)
    assert plan.ok
    lower_semantic_request(request, plan, claim_query=backend.claim_query)
    assert plan.batch.invalidations[0].valid_to == "2099-01-01T00:00:00+00:00"
    result = backend.mutation.apply(plan.batch, expected_pot_id=pot_id)
    assert result.ok, result.error

    before = _keys(
        backend,
        pot_id,
        as_of=datetime(2098, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    at_end = _keys(
        backend, pot_id, as_of=datetime(2099, 1, 1, tzinfo=timezone.utc)
    )
    assert "claim:offset-schedule" in before
    assert "claim:offset-schedule" not in at_end


def test_timeline_event_date_is_identical_across_repo_service_and_source_routes(
    temporal_backend,
):
    _, pot_id, backend = temporal_backend
    ingested_at = datetime.now(timezone.utc) - timedelta(hours=1)
    occurred_at = ingested_at - timedelta(days=8)
    activity = "activity:github:pr:1074"
    repo = "repo:github.com/acme/widgets"
    service = "service:checkout"
    repo_ref = "github:acme/widgets:pull:1074"
    service_ref = "github:acme/widgets:pull:1074#service"
    _apply(
        backend,
        pot_id,
        edges=(
            _edge(
                "claim:pr-1074-repo",
                subject=activity,
                object_=repo,
                valid_at=ingested_at,
                source_ref=repo_ref,
                occurred_at=occurred_at,
            ),
            _edge(
                "claim:pr-1074-service",
                subject=activity,
                object_=service,
                valid_at=ingested_at + timedelta(minutes=5),
                source_ref=service_ref,
            ),
        ),
    )
    reader = TimelineReader(claim_query=backend.claim_query, ranker=RankingService())

    routes = (
        ReadRequest(pot_id=pot_id, scope={"repo": "github.com/acme/widgets"}),
        ReadRequest(pot_id=pot_id, scope={"service": "checkout"}),
        ReadRequest(pot_id=pot_id, scope={}, source_refs=(repo_ref,)),
    )
    event_dates = []
    for route in routes:
        response = reader.read(route)
        assert len(response.items) == 1
        event = response.items[0].candidate.payload
        assert event["activity_key"] == activity
        event_dates.append(event["occurred_at"])

    assert event_dates == [occurred_at.isoformat()] * 3
