"""W7/W8 acceptance through public write/read contracts on shipped backends."""

from __future__ import annotations

import dataclasses
import json
import os
import uuid

import pytest

from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.ports.agent_context import RecordRequest, ResolveRequest
from potpie_context_core.ports.graph_service import (
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService


@pytest.fixture(params=("in_memory", "embedded", "falkordb_lite", "neo4j"))
def answers(request, tmp_path, monkeypatch):
    profile = request.param
    pot_id = "w7-w8-" + uuid.uuid4().hex
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv(
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "answers_" + uuid.uuid4().hex
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
    try:
        yield pot_id, DefaultGraphService(backend=backend)
    finally:
        if profile == "neo4j":
            assert backend.mutation.reset_pot(pot_id)["ok"]
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()


def _record(graph, pot_id, kind, summary, details):
    receipt = graph.record(
        RecordRequest(
            pot_id=pot_id,
            record_type=kind,
            summary=summary,
            details=details,
            scope={"service": "checkout"},
            source_refs=(f"fixture:{kind}:{summary}",),
        )
    )
    assert receipt.accepted, receipt.detail


def _dump(value):
    return json.dumps(value, default=str)


def test_saved_fix_and_failed_verification_remain_answerable(answers):
    pot_id, graph = answers
    _record(
        graph,
        pot_id,
        "fix",
        "Checkout socket exhaustion",
        {
            "fix_id": "fix:checkout-socket",
            "root_cause": "Leaked retry sockets",
            "fix_steps": ["Close retry sockets in finally"],
            "verification_status": "unverified",
        },
    )
    _record(
        graph,
        pot_id,
        "verification",
        "Socket soak test failed",
        {
            "target_ref": "fix:checkout-socket",
            "outcome": "didnt_work",
        },
    )
    read = graph.read(
        GraphReadRequest(
            pot_id=pot_id,
            subgraph="debugging",
            view="prior_occurrences",
            scope={"service": "checkout"},
            detail="full",
            relations="full",
        )
    )
    envelope = graph.resolve(
        ResolveRequest(
            pot_id=pot_id,
            include=("prior_bugs",),
            scope={"service": "checkout"},
        )
    )
    for output in (read.to_dict(), dataclasses.asdict(envelope)):
        text = _dump(output)
        assert "Leaked retry sockets" in text
        assert "Close retry sockets in finally" in text
        assert "didnt_work" in text
    resolved = [
        item for item in envelope.items if item.payload.get("predicate") == "RESOLVED"
    ]
    assert resolved
    assert all(item.payload.get("verification_count", 0) == 0 for item in resolved)

    _record(
        graph,
        pot_id,
        "verification",
        "Socket soak test passed later",
        {
            "target_ref": "fix:checkout-socket",
            "outcome": "worked",
        },
    )
    checked = graph.resolve(
        ResolveRequest(
            pot_id=pot_id,
            include=("prior_bugs",),
            scope={"service": "checkout"},
        )
    )
    fix = next(
        item.payload
        for item in checked.items
        if item.payload.get("predicate") == "RESOLVED"
    )
    outcomes = {
        outcome["fact"]: outcome["outcome"]
        for outcome in fix["details"]["verification_outcomes"]
    }
    assert outcomes == {
        "Socket soak test failed": "didnt_work",
        "Socket soak test passed later": "worked",
    }
    assert fix["verification_count"] == 1


def test_saved_decision_reasoning_survives_named_read_and_resolve(answers):
    pot_id, graph = answers
    _record(
        graph,
        pot_id,
        "decision",
        "Choose bounded retry concurrency",
        {
            "rationale": "Prevent retry storms from exhausting shared sockets",
            "alternatives_rejected": ["Unbounded retry fanout"],
        },
    )
    read = graph.read(
        GraphReadRequest(
            pot_id=pot_id,
            subgraph="decisions",
            view="active_decisions",
            scope={"service": "checkout"},
            detail="compact",
        )
    )
    envelope = graph.resolve(
        ResolveRequest(
            pot_id=pot_id,
            include=("decisions",),
            scope={"service": "checkout"},
        )
    )
    for output in (read.to_dict(), dataclasses.asdict(envelope)):
        text = _dump(output)
        assert "Prevent retry storms from exhausting shared sockets" in text
        assert "Unbounded retry fanout" in text


def _seed_prs(graph, pot_id):
    entities, edges = [], []
    for org in ("acme", "other"):
        activity = f"activity:github:{org}/widgets:pr:1074"
        repo = f"repo:github.com/{org}/widgets"
        source = f"https://github.com/{org}/widgets/pull/1074"
        entities.extend(
            [
                EntityUpsert(
                    entity_key=activity,
                    labels=("Activity",),
                    properties={
                        "name": "Repair daemon socket lifecycle",
                        "external_id": "1074",
                        "source_ref": source,
                        "occurred_at": "2026-09-15T14:12:11+00:00",
                    },
                ),
                EntityUpsert(entity_key=repo, labels=("Repository",)),
            ]
        )
        edges.append(
            EdgeUpsert(
                "TOUCHED",
                activity,
                repo,
                {
                    "claim_key": f"claim:{org}:1074",
                    "subgraph": "recent_changes",
                    "truth": "source_observation",
                    "fact": "Repair daemon socket lifecycle",
                    "description": "Repair daemon socket lifecycle",
                    "source_ref": source,
                    "source_refs": [source],
                    "valid_at": "2026-09-15T14:12:11+00:00",
                },
            )
        )
    result = graph.backend.mutation.apply(
        MutationBatch(entity_upserts=entities, edge_upserts=edges),
        expected_pot_id=pot_id,
    )
    assert result.ok, result.error


def test_exact_pr_identity_is_repo_scoped_and_missing_ticket_is_not_found(answers):
    pot_id, graph = answers
    _seed_prs(graph, pot_id)
    found = graph.search_entities(
        GraphEntitySearchRequest(
            pot_id=pot_id,
            query="PR #1074",
            scope={"repo": "github.com/acme/widgets"},
        )
    )
    assert [entity.key for entity in found.entities] == [
        "activity:github:acme/widgets:pr:1074"
    ]
    assert found.match_status == "exact_match"
    duplicates = graph.search_entities(
        GraphEntitySearchRequest(pot_id=pot_id, query="PR #1074")
    )
    assert duplicates.match_status == "ambiguous_exact_match"
    assert set(duplicates.matching_repositories) == {
        "repo:github.com/acme/widgets",
        "repo:github.com/other/widgets",
    }
    timeline = graph.read(
        GraphReadRequest(
            pot_id=pot_id,
            subgraph="recent_changes",
            view="timeline",
            query="PR #1074",
            scope={"repo": "github.com/acme/widgets"},
            detail="full",
        )
    )
    assert "activity:github:acme/widgets:pr:1074" in _dump(timeline.to_dict())
    assert "activity:github:other/widgets:pr:1074" not in _dump(timeline.to_dict())
    absent = graph.search_entities(
        GraphEntitySearchRequest(pot_id=pot_id, query="PR #999999")
    )
    assert not absent.entities
    assert absent.match_status == "no_exact_match"
