"""Live FalkorDB **server** round-trip against a running FalkorDB instance.

Designed for CI (GitHub Actions ``falkordb/falkordb`` service). Skips when
``FALKORDB_URL`` is unset or the server is unreachable.
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid
from urllib.parse import urlparse

import pytest

from adapters.outbound.graph.backends import build_backend
from adapters.outbound.settings_env import EnvContextEngineSettings
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ports.claim_query import ClaimQueryFilter
from domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.integration


def _redis_reachable(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _plan(
    pot_id: str, summary: str = "prefers structured logging"
) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="agent", pot_id=pot_id),
        summary=summary,
        entity_upserts=[
            EntityUpsert(entity_key="pref:logging", labels=("Preference",)),
            EntityUpsert(entity_key="svc:api", labels=("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPENDS_ON",
                from_entity_key="pref:logging",
                to_entity_key="svc:api",
                properties={"fact": summary},
            )
        ],
    )


@pytest.fixture()
def falkordb_live_env(monkeypatch: pytest.MonkeyPatch) -> EnvContextEngineSettings:
    url = (
        os.environ.get("CONTEXT_ENGINE_FALKORDB_URL")
        or os.environ.get("FALKORDB_URL")
        or ""
    ).strip()
    if not url:
        pytest.skip("FALKORDB_URL not configured")
    if not _redis_reachable(url):
        pytest.skip(f"FalkorDB not reachable at {url}")
    monkeypatch.setenv("FALKORDB_URL", url)
    monkeypatch.setenv("FALKORDB_MODE", "server")
    monkeypatch.setenv("GRAPH_DB_BACKEND", "falkordb")
    monkeypatch.setenv("CONTEXT_GRAPH_ENABLED", "1")
    graph_name = (
        os.environ.get("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME")
        or os.environ.get("FALKORDB_GRAPH_NAME")
        or "context_graph_ci"
    )
    monkeypatch.setenv("FALKORDB_GRAPH_NAME", graph_name)
    return EnvContextEngineSettings()


def test_falkordb_server_mutation_claim_query_roundtrip(
    falkordb_live_env: EnvContextEngineSettings,
) -> None:
    pot_id = f"ci_{uuid.uuid4().hex[:12]}"
    backend = build_backend("falkordb", settings=falkordb_live_env)
    assert backend.profile == "falkordb"

    result = asyncio.run(
        backend.mutation.apply_async(_plan(pot_id), expected_pot_id=pot_id)
    )
    assert result.ok

    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_id))
    assert len(rows) == 1
    assert rows[0].fact == "prefers structured logging"

    backend.mutation.reset_pot(pot_id)
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_id)) == []
