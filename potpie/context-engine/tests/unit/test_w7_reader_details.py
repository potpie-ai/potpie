"""W7 acceptance tests for useful, honest, bounded reader details."""

from __future__ import annotations

from datetime import datetime, timezone

from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.readers import (
    DecisionsReader,
    InfraTopologyReader,
    PriorBugsReader,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.domain.ranking import RankingService
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_core.ports.agent_context import RecordRequest, ResolveRequest


def _row(predicate: str, subject: str, object_: str, **properties) -> ClaimRow:
    return ClaimRow(
        pot_id="p",
        predicate=predicate,
        subject_key=subject,
        object_key=object_,
        fact=properties.pop("fact", f"{subject} {predicate} {object_}"),
        valid_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
        evidence_strength="attested",
        claim_key=f"claim:{predicate}:{subject}:{object_}",
        source_ref=f"source:{predicate.lower()}",
        source_refs=(f"source:{predicate.lower()}",),
        properties=properties,
    )


def test_fix_returns_remedy_root_cause_and_mixed_verification_honestly() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("REPRODUCES", "bug_pattern:pool", "service:auth"))
    store.add(_row("RESOLVED", "fix:pool", "bug_pattern:pool"))
    store.add(_row("VERIFIED", "person:alice", "fix:pool", outcome="passed"))
    store.add(_row("VERIFIED", "person:bob", "fix:pool", outcome="failed"))
    store.set_entity_properties(
        pot_id="p",
        entity_key="fix:pool",
        properties={
            "root_cause": "connections leaked on cancellation",
            "fix_steps": ["close the connection in finally"],
            "verification_status": "mixed",
        },
    )

    response = PriorBugsReader(store, RankingService()).read(
        ReadRequest(pot_id="p", scope={"service": "auth"}, max_items=10)
    )
    fix = next(
        item.candidate.payload
        for item in response.items
        if item.candidate.payload["predicate"] == "RESOLVED"
    )

    assert fix["details"]["root_cause"] == "connections leaked on cancellation"
    assert fix["details"]["fix_steps"] == ["close the connection in finally"]
    assert [
        outcome["succeeded"] for outcome in fix["details"]["verification_outcomes"]
    ] == [True, False]
    assert fix["verification_count"] == 1


def test_failed_only_verification_does_not_add_positive_rank_signal() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("REPRODUCES", "bug_pattern:pool", "service:auth"))
    store.add(_row("RESOLVED", "fix:pool", "bug_pattern:pool"))
    for index in range(13):
        store.add(
            _row(
                "VERIFIED",
                f"activity:check:{index}",
                "fix:pool",
                status=1 if index == 0 else "failed",
                fact="x" * 3_000 if index == 0 else "failed check",
            )
        )

    response = PriorBugsReader(store, RankingService()).read(
        ReadRequest(pot_id="p", scope={"service": "auth"}, max_items=10)
    )
    fix = next(
        item.candidate.payload
        for item in response.items
        if item.candidate.payload["predicate"] == "RESOLVED"
    )
    assert fix["verification_count"] == 0
    assert all(
        outcome["succeeded"] is False
        for outcome in fix["details"]["verification_outcomes"]
    )
    assert len(fix["details"]["verification_outcomes"][0]["fact"]) == 2_000
    assert fix["details"]["omitted"]["verification_outcomes"] == {"items": 1}


def test_decision_returns_rationale_and_bounded_alternatives() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("DECIDED", "decision:storage", "service:context-engine"))
    store.set_entity_properties(
        pot_id="p",
        entity_key="decision:storage",
        properties={
            "rationale": "A single write contract keeps adapters coherent.",
            "alternatives_rejected": ["x" * 3_000]
            + [f"alternative {i}" for i in range(14)],
        },
    )

    response = DecisionsReader(store, RankingService()).read(
        ReadRequest(pot_id="p", scope={"service": "context-engine"}, max_items=10)
    )
    details = response.items[0].candidate.payload["details"]
    assert details["rationale"].startswith("A single write contract")
    assert len(details["alternatives_rejected"]) == 12
    assert len(details["alternatives_rejected"][0]) == 2_000
    assert details["omitted"]["alternatives_rejected"]["items"] == 3
    assert details["omitted"]["alternatives_rejected"]["entries"]["0"] == {
        "characters": 1_000
    }


def test_infra_returns_api_links_and_endpoint_details() -> None:
    store = InMemoryClaimQueryStore()
    store.add(_row("EXPOSES", "service:daemon", "api_endpoint:health"))
    store.set_entity_properties(
        pot_id="p",
        entity_key="api_endpoint:health",
        properties={"method": "GET", "path": "/health", "protocol": "HTTP"},
    )

    response = InfraTopologyReader(store, RankingService()).read(
        ReadRequest(pot_id="p", scope={"service": "daemon"}, max_items=10)
    )
    api = next(
        item.candidate.payload
        for item in response.items
        if item.candidate.payload["predicate"] == "EXPOSES"
    )
    assert api["details"]["object"] == {
        "method": "GET",
        "path": "/health",
        "protocol": "HTTP",
    }


def test_public_records_preserve_each_verification_occurrence_outcome() -> None:
    graph = DefaultGraphService(backend=InMemoryGraphBackend())
    fix = graph.record(
        RecordRequest(
            pot_id="p",
            record_type="fix",
            summary="Demo fix",
            details={"fix_id": "fix:demo", "fix_steps": ["apply remedy"]},
            scope={"service": "api"},
            source_refs=("test:fix",),
        )
    )
    assert fix.accepted
    for summary, outcome, source_ref in (
        ("check passed", "worked", "test:1"),
        ("check failed", "didnt_work", "test:2"),
    ):
        receipt = graph.record(
            RecordRequest(
                pot_id="p",
                record_type="verification",
                summary=summary,
                details={"target_ref": "fix:demo", "outcome": outcome},
                scope={"service": "api"},
                source_refs=(source_ref,),
            )
        )
        assert receipt.accepted

    envelope = graph.resolve(
        ResolveRequest(pot_id="p", include=("prior_bugs",), scope={"service": "api"})
    )
    resolved = next(
        item.payload
        for item in envelope.items
        if item.payload.get("predicate") == "RESOLVED"
    )
    outcomes = resolved["details"]["verification_outcomes"]
    assert [(item["fact"], item["outcome"]) for item in outcomes] == [
        ("check passed", "worked"),
        ("check failed", "didnt_work"),
    ]
    assert resolved["verification_count"] == 1
