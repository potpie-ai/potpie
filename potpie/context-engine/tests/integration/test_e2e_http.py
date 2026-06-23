"""Live end-to-end tests that drive the **FastAPI HTTP layer** with a real
(live-Neo4j) container bound via ``app.dependency_overrides`` (see e2e_surface.md).

Every other e2e test calls the use-case/adapter layer directly; this module
is the only one that traverses the real entrypoint — auth + hardening
middleware, request validation, the policy tenant boundary, deps wiring, and
unsupported legacy graph-query handling. Deterministic: no LLM, no Postgres
(DB-backed endpoints are covered at the service layer by the batching and
pipeline suites). Skips with the rest of the suite when Neo4j is down.
"""

from __future__ import annotations

import asyncio

import pytest

from context_engine.domain.context_events import EventRef
from context_engine.domain.graph_mutations import EdgeUpsert, EntityUpsert
from context_engine.domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.integration

API = "/api/v1/context"


def _seed_plan(pot_id: str) -> ReconciliationPlan:
    """A tiny topology so reads have something deterministic to return."""
    return ReconciliationPlan(
        event_ref=EventRef(event_id="http-seed", source_system="test", pot_id=pot_id),
        summary="http seed",
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
            EntityUpsert(
                "repo:acme-platform",
                ("Entity", "Repository"),
                {"name": "acme/platform"},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEFINED_IN", "service:web", "repo:acme-platform", {"path": "apps/web"}
            ),
        ],
        edge_deletes=[],
        invalidations=[],
    )


def _seed(container, pot_id: str) -> None:
    assert container.backend is not None
    asyncio.run(
        container.backend.mutation.apply_async(
            _seed_plan(pot_id),
            expected_pot_id=pot_id,
        )
    )


def _count_entities(settings, pot_id: str) -> int:
    """Direct Cypher count of canonical entity nodes (the structural read
    stack that exposed graph overviews was removed with the episodic tier)."""
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    )
    try:
        with drv.session() as session:
            rec = session.run(
                "MATCH (e:Entity {group_id:$g}) RETURN count(e) AS n", g=pot_id
            ).single()
            return int(rec["n"]) if rec is not None else 0
    finally:
        drv.close()


@pytest.fixture()
def make_client(container, monkeypatch):
    """Factory building the real app bound to the live container.

    Auth is controlled per-call: ``allow_no_auth`` toggles the dev escape
    hatch (which also satisfies the policy tenant boundary), ``api_key`` sets
    ``CONTEXT_ENGINE_API_KEY``. The DB deps are stubbed (None) since these
    tests only touch Neo4j-backed endpoints.
    """
    from fastapi.testclient import TestClient

    from context_engine.adapters.inbound.http import deps
    from context_engine.adapters.inbound.http.app import create_app

    def _make(*, allow_no_auth: bool = True, api_key: str | None = None) -> TestClient:
        if allow_no_auth:
            monkeypatch.setenv("CONTEXT_ENGINE_ALLOW_NO_AUTH", "1")
        else:
            monkeypatch.delenv("CONTEXT_ENGINE_ALLOW_NO_AUTH", raising=False)
        if api_key is not None:
            monkeypatch.setenv("CONTEXT_ENGINE_API_KEY", api_key)
        else:
            monkeypatch.delenv("CONTEXT_ENGINE_API_KEY", raising=False)

        app = create_app()
        app.dependency_overrides[deps.get_container_or_503] = lambda: container
        app.dependency_overrides[deps.get_db_optional] = lambda: None
        return TestClient(app, raise_server_exceptions=False)

    return _make


# ---------------------------------------------------------------------------
# Auth + policy tenant boundary (the hardening gate, fail-closed)
# ---------------------------------------------------------------------------


class TestHttpAuth:
    def test_missing_api_key_is_503(self, make_client, pot_id) -> None:
        # No key configured and no dev escape hatch → refuse to serve.
        client = make_client(allow_no_auth=False, api_key=None)
        r = client.post(f"{API}/status", json={"pot_id": pot_id})
        assert r.status_code == 503

    def test_wrong_api_key_is_401(self, make_client, pot_id) -> None:
        client = make_client(allow_no_auth=False, api_key="s3cret")
        r = client.post(
            f"{API}/status",
            json={"pot_id": pot_id},
            headers={"X-API-Key": "WRONG"},
        )
        assert r.status_code == 401

    def test_authenticated_without_tenant_scope_is_403(
        self, make_client, pot_id
    ) -> None:
        # Correct key, but the wired resolver is NOT actor-scoped and the dev
        # escape hatch is off → the policy tenant boundary fails closed.
        client = make_client(allow_no_auth=False, api_key="s3cret")
        r = client.post(
            f"{API}/status",
            json={"pot_id": pot_id},
            headers={"X-API-Key": "s3cret"},
        )
        assert r.status_code == 403

    def test_dev_no_auth_mode_allows(self, make_client, pot_id) -> None:
        client = make_client(allow_no_auth=True)
        r = client.post(f"{API}/status", json={"pot_id": pot_id})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Query surface end-to-end (serialization + shape, not just non-null)
# ---------------------------------------------------------------------------


class TestHttpQuery:
    def test_query_context_graph_is_not_supported(self, make_client, pot_id) -> None:
        client = make_client()
        r = client.post(f"{API}/query/context-graph", json={"pot_id": pot_id})
        assert r.status_code == 501, r.text
        body = r.json()
        assert body["detail"]["code"] == "http_context_graph_query_not_supported"

    def test_query_validation_is_bypassed_for_unsupported_route(self, make_client) -> None:
        client = make_client()
        r = client.post(f"{API}/query/context-graph", json={"goal": "neighborhood"})
        assert r.status_code == 501


# ---------------------------------------------------------------------------
# Operator reset through the HTTP surface
# ---------------------------------------------------------------------------


class TestHttpReset:
    def test_reset_clears_pot_via_http(
        self, make_client, container, settings, pot_id
    ) -> None:
        _seed(container, pot_id)
        assert _count_entities(settings, pot_id) >= 1

        client = make_client()
        r = client.post(f"{API}/reset", json={"pot_id": pot_id, "skip_ledger": True})
        assert r.status_code == 200, r.text
        assert r.json().get("ok") is True
        assert _count_entities(settings, pot_id) == 0


# ---------------------------------------------------------------------------
# Hardening middleware is actually installed on the live app
# ---------------------------------------------------------------------------


class TestHttpHardening:
    def test_security_headers_present(self, make_client) -> None:
        client = make_client()
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.headers.get("x-content-type-options") == "nosniff"
