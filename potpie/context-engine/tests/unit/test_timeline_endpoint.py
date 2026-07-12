"""Unit test for ``GET /pots/{pot_id}/timeline``.

Drives the real router factory + the real read trunk (orchestrator →
TimelineReader → envelope) over the in-memory claim store, with a fake
allow-all policy and auth. No Neo4j / Postgres. Proves the endpoint wiring:
window derivation, scope anchoring, verb_class filtering, and the flattened
activity shape the UI consumes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie_context_engine.adapters.inbound.http.api.v1.context.router import (
    _parse_window,
    create_context_router,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.domain.ports.claim_query import ClaimRow

API = "/api/v1/context"


def _touched(
    activity: str, service: str, *, verb_class: str, fact: str, age_days: float
) -> ClaimRow:
    return ClaimRow(
        pot_id="pot-1",
        predicate="TOUCHED",
        subject_key=activity,
        object_key=service,
        valid_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        evidence_strength="attested",
        source_system="github",
        source_ref=f"src:{activity}",
        fact=fact,
        properties={"verb_class": verb_class},
    )


class _AllowDecision:
    allowed = True
    reason = "ok"
    detail = None
    status_code = 200


class _AllowPolicy:
    def authorize(self, **_: Any) -> _AllowDecision:
        return _AllowDecision()


class _FakeContainer:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def policy(self) -> _AllowPolicy:
        return _AllowPolicy()


def _client() -> TestClient:
    store = InMemoryClaimQueryStore()
    store.add_many(
        [
            _touched(
                "activity:github:pr-1",
                "service:web",
                verb_class="code_change",
                fact="PR #1 idempotency merged",
                age_days=2,
            ),
            _touched(
                "activity:deploy:rev-1",
                "service:web",
                verb_class="deployment",
                fact="Deploy rev-1 to prod",
                age_days=1,
            ),
            _touched(
                "activity:github:pr-old",
                "service:web",
                verb_class="code_change",
                fact="ancient PR",
                age_days=120,
            ),
        ]
    )
    graph = DefaultGraphService(backend=InMemoryGraphBackend(store=store))
    container = _FakeContainer(graph)
    router = create_context_router(
        require_auth=lambda: {"user_id": "u"},
        get_container=lambda: container,  # type: ignore[arg-type]
        get_db=lambda: None,
        get_db_optional=lambda: None,
    )
    app = FastAPI()
    app.include_router(router, prefix=API)
    return TestClient(app)


def test_parse_window() -> None:
    assert _parse_window("24h") == timedelta(hours=24)
    assert _parse_window("7d") == timedelta(days=7)
    assert _parse_window("2w") == timedelta(weeks=2)
    assert _parse_window("30m") == timedelta(minutes=30)
    assert _parse_window("garbage") is None
    assert _parse_window(None) is None


def test_timeline_window_and_scope() -> None:
    client = _client()
    resp = client.get(
        f"{API}/pots/pot-1/timeline", params={"service": "web", "window": "14d"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    facts = {it["title"] for it in body["items"]}
    assert "PR #1 idempotency merged" in facts
    assert "Deploy rev-1 to prod" in facts
    assert "ancient PR" not in facts  # outside the 14d window
    # The UI-facing shape carries the event kind + identity.
    kinds = {it["verb_class"] for it in body["items"]}
    assert kinds == {"code_change", "deployment"}
    assert all(it["activity_key"].startswith("activity:") for it in body["items"])
    assert body["window"]["since"] is not None


def test_timeline_verb_class_filter() -> None:
    client = _client()
    resp = client.get(
        f"{API}/pots/pot-1/timeline",
        params={"service": "web", "window": "14d", "verb_class": "deployment"},
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["verb_class"] == "deployment"
