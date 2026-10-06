"""Answer-bearing fix fields survive every human read renderer.

A fix's root cause, steps, verification and source status reach the agent
through ``graph read``, ``resolve`` and ``graph neighborhood``; each human
renderer has to show them rather than only the claim's evidence note.
"""

from __future__ import annotations

import pytest

from potpie.cli.commands import graph as graph_cli
from potpie.cli.commands import query as query_cli
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.core.ports.agent_context import ResolveRequest
from potpie_context_engine.core.ports.claim_query import ClaimRow
from potpie_context_engine.core.ports.graph_service import GraphReadRequest

pytestmark = pytest.mark.unit


def _service_with_a_verified_fix() -> DefaultGraphService:
    backend = InMemoryGraphBackend()
    store = backend.store
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="REPRODUCES",
            subject_key="bug_pattern:cookie",
            object_key="service:api",
            claim_key="claim:symptom",
            fact="Second daemon causes a 401",
            source_refs=("test:symptom",),
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="RESOLVED",
            subject_key="fix:cookie",
            object_key="bug_pattern:cookie",
            claim_key="claim:fix",
            fact="Use a daemon-specific cookie name",
            source_refs=("test:fix",),
        )
    )
    store.add(
        ClaimRow(
            pot_id="p",
            predicate="VERIFIED",
            subject_key="activity:check",
            object_key="fix:cookie",
            claim_key="claim:check",
            fact="Two-daemon test passed",
            source_refs=("test:check",),
            properties={"outcome": "passed"},
        )
    )
    store.set_entity_properties(
        pot_id="p",
        entity_key="fix:cookie",
        properties={
            "root_cause": "Cookie name was shared across local daemons",
            "fix_steps": ["Suffix the cookie name per daemon"],
            "verification_status": "passed",
            "source_status": "local only",
        },
    )
    return DefaultGraphService(backend=backend)


def test_graph_read_and_resolve_text_carry_the_fix_answer_fields() -> None:
    service = _service_with_a_verified_fix()
    read = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="debugging",
            view="prior_occurrences",
            scope={"service": "api"},
            detail="full",
            relations="full",
        )
    )
    resolved = service.resolve(
        ResolveRequest(pot_id="p", include=("prior_bugs",), scope={"service": "api"})
    )

    for text in (
        graph_cli._read_human(
            read, format_="raw", sort="auto", dedupe="auto", event_limit=None
        ),
        query_cli._envelope_human(resolved),
    ):
        assert "Cookie name was shared" in text
        assert "Suffix the cookie name" in text
        assert "passed" in text
        assert "local only" in text


def test_neighborhood_text_carries_the_fix_answer_fields() -> None:
    service = _service_with_a_verified_fix()
    sl = service.backend.inspection.neighborhood(
        pot_id="p", entity_key="fix:cookie", depth=1, limit=10
    )
    payload = {
        "entity_key": "fix:cookie",
        "identity_status": "exact",
        "detail": "full",
        "node_count": len(sl.nodes),
        "truncated": sl.truncated,
        "relations": [graph_cli._neighborhood_relation(edge) for edge in sl.edges],
        "nodes": [
            {
                "key": node.key,
                "labels": list(node.labels),
                "properties": dict(node.properties),
            }
            for node in sl.nodes
        ],
        "edges": [
            {
                "predicate": edge.predicate,
                "from": edge.from_key,
                "to": edge.to_key,
                "properties": dict(edge.properties),
            }
            for edge in sl.edges
        ],
    }

    text = graph_cli._neighborhood_human(payload)

    assert "Cookie name was shared" in text
    assert "Suffix the cookie name" in text
    assert "local only" in text
