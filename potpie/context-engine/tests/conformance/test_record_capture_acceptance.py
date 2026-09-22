"""W9 record retry acceptance across each shipped local/graph backend."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace

import pytest
from potpie_context_core.api import build_graph_runtime
from potpie_context_core.ports.agent_context import RecordRequest
from potpie_context_core.ports.graph_service import GraphReadRequest

from potpie_context_engine.adapters.outbound.graph.backends import build_backend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.testing import InMemoryGraphPlanStore


@pytest.fixture(params=("in_memory", "embedded", "falkordb_lite", "neo4j"))
def capture_runtime(request, tmp_path, monkeypatch):
    profile = request.param
    pot_id = "w9-record-capture-" + uuid.uuid4().hex
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv(
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "record_capture_" + uuid.uuid4().hex
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
    runtime = build_graph_runtime(backend, InMemoryGraphPlanStore())
    try:
        yield profile, pot_id, runtime
    finally:
        if profile == "neo4j":
            assert backend.mutation.reset_pot(pot_id)["ok"]
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()


def _request(pot_id: str) -> RecordRequest:
    return RecordRequest(
        pot_id=pot_id,
        record_type="fix",
        summary="Checkout retry sockets exhausted the pool",
        details={
            "fix_id": "fix:w9-checkout-retry-sockets",
            "root_cause": "Cancelled retries left sockets open",
            "fix_steps": [
                "Close sockets in finally",
                "Bound concurrent retry attempts",
            ],
        },
        scope={"service": "checkout"},
        source_refs=("fixture:w9:incident:socket-exhaustion",),
        idempotency_key="w9:record-capture:socket-exhaustion",
        metadata={"surface": "conformance"},
    )


def test_record_retry_identity_and_scoped_readback(capture_runtime):
    profile, pot_id, runtime = capture_runtime
    original = _request(pot_id)

    first = runtime.record(original)
    assert first.accepted, (profile, first.detail)
    assert first.mutations_applied > 0
    replay = runtime.record(original)
    assert replay.accepted and replay.status == "duplicate", (profile, replay.detail)
    assert replay.mutations_applied == 0
    assert replay.metadata["plan_id"] == first.metadata["plan_id"]
    assert replay.metadata["mutation_id"] == first.metadata["mutation_id"]

    changed_requests = (
        replace(
            original,
            details={
                **original.details,
                "root_cause": "A different cause",
                "fix_steps": ["Restart the worker"],
            },
        ),
        replace(original, source_refs=("fixture:w9:incident:different-evidence",)),
    )
    for changed in changed_requests:
        rejected = runtime.record(changed)
        assert not rejected.accepted, profile
        assert rejected.status == "rejected"
        assert "different content" in (rejected.detail or "")
        # A refused collision is audit history; it must not take ownership of
        # the retry key or disturb the original receipt.
        recovered = runtime.record(original)
        assert recovered.accepted and recovered.status == "duplicate"
        assert recovered.metadata["mutation_id"] == first.metadata["mutation_id"]

    read = runtime.read(
        GraphReadRequest(
            pot_id=pot_id,
            subgraph="debugging",
            view="prior_occurrences",
            scope={"service": "checkout"},
            detail="full",
            relations="full",
        )
    )
    assert read.ok, (profile, read.to_dict())
    rendered = json.dumps(read.to_dict(), default=str)
    assert "Cancelled retries left sockets open" in rendered
    assert "Close sockets in finally" in rendered
    assert "Bound concurrent retry attempts" in rendered
    assert "fixture:w9:incident:socket-exhaustion" in rendered
    assert "A different cause" not in rendered
    assert "Restart the worker" not in rendered
    assert "fixture:w9:incident:different-evidence" not in rendered
