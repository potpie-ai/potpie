"""End-to-end ingest → ask pipeline on the FalkorDBLite backend.

Unlike ``test_falkordb_roundtrip.py`` (which drives the adapters directly), this
exercises the **full wired stack** the CLI/HTTP path uses: ``build_container``
with ``GRAPH_DB_BACKEND=falkordb`` selects the FalkorDB writer + reader behind
``ContextGraphService`` + ``ReadOrchestrator``, backed by an embedded
FalkorDBLite file (no server, no Docker, no LLM, no Postgres).

Pattern mirrors the deterministic Neo4j e2e (``test_e2e_topology.py``):
ingest via ``context_graph.apply_plan_async`` (the same write trunk the
reconciliation agent drives), then read it back — both at the claim-query
surface the P9 readers sit on and through the high-level resolve query.

Skips cleanly when FalkorDBLite isn't installed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration

# Distribution ``falkordblite`` exposes ``redislite.falkordb_client``; skip if absent.
pytest.importorskip("redislite.falkordb_client")

from adapters.outbound.graph.falkordb_reader import (  # noqa: E402
    FalkorDBClaimQueryStore,
)
from adapters.outbound.graph.falkordb_writer import FalkorDBGraphWriter  # noqa: E402
from adapters.outbound.settings_env import EnvContextEngineSettings  # noqa: E402
from bootstrap.container import build_container  # noqa: E402
from domain.context_events import EventRef  # noqa: E402
from domain.graph_mutations import EdgeUpsert, EntityUpsert  # noqa: E402
from domain.graph_query import (  # noqa: E402
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
)
from domain.ports.claim_query import ClaimQueryFilter  # noqa: E402
from domain.reconciliation import ReconciliationPlan  # noqa: E402

_POT = "potA"


def _topology_plan(pot_id: str) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="seed", source_system="test", pot_id=pot_id),
        summary="topology",
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
            EntityUpsert("service:auth", ("Entity", "Service"), {"name": "auth"}),
            EntityUpsert(
                "environment:prod", ("Entity", "Environment"), {"name": "prod"}
            ),
        ],
        edge_upserts=[
            EdgeUpsert("DEPENDS_ON", "service:web", "service:auth", {}),
            EdgeUpsert(
                "DEPLOYED_TO",
                "service:web",
                "environment:prod",
                {"environment": "prod", "evidence_strength": "deterministic"},
            ),
        ],
        edge_deletes=[],
        invalidations=[],
    )


@pytest.fixture()
def lite_container(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A real container wired to an embedded FalkorDBLite file in a temp dir."""
    monkeypatch.setenv("GRAPH_DB_BACKEND", "falkordb")
    monkeypatch.setenv("FALKORDB_MODE", "lite")
    monkeypatch.setenv("FALKORDB_LITE_PATH", str(tmp_path / "context_graph.db"))
    monkeypatch.delenv("CONTEXT_GRAPH_ENABLED", raising=False)  # defaults enabled
    return build_container(settings=EnvContextEngineSettings(), pots=MagicMock())


def test_container_selects_enabled_falkordb_lite(lite_container) -> None:
    # Lite needs no URL/server: the wired writer is enabled out of the box, and
    # both ports are the FalkorDB implementations sharing one embedded handle.
    assert isinstance(lite_container.graph_writer, FalkorDBGraphWriter)
    assert lite_container.graph_writer.enabled is True
    assert isinstance(
        lite_container.context_graph._orchestrator.claim_query, FalkorDBClaimQueryStore
    )


def test_ingest_then_read_back_claims(lite_container) -> None:
    # Ingest through the same write trunk the reconciliation agent drives.
    res = asyncio.run(
        lite_container.context_graph.apply_plan_async(
            _topology_plan(_POT), expected_pot_id=_POT
        )
    )
    assert res.ok is True
    assert res.mutation_summary.entity_upserts_applied == 3
    assert res.mutation_summary.edge_upserts_applied == 2

    # "Ask" at the claim-query surface the P9 readers query over.
    reader = lite_container.context_graph._orchestrator.claim_query
    depends = reader.find_claims(
        ClaimQueryFilter(pot_id=_POT, predicate_in=("DEPENDS_ON",))
    )
    assert len(depends) == 1
    assert depends[0].subject_key == "service:web"
    assert depends[0].object_key == "service:auth"

    labels = reader.entity_labels(pot_id=_POT, entity_keys=["service:web"])
    assert "Service" in labels["service:web"]


def test_resolve_query_runs_over_falkordb(lite_container) -> None:
    # The high-level read trunk (resolve envelope) runs end to end on FalkorDBLite
    # without an LLM — goal=RETRIEVE is deterministic.
    asyncio.run(
        lite_container.context_graph.apply_plan_async(
            _topology_plan(_POT), expected_pot_id=_POT
        )
    )
    query = ContextGraphQuery(
        pot_id=_POT,
        query="What does the web service depend on?",
        goal=ContextGraphGoal.RETRIEVE,
        scope=ContextGraphScope(services=["web"]),
    )
    result = asyncio.run(lite_container.context_graph.query_async(query))
    assert result.error is None
    assert isinstance(result.result, dict)


def test_reset_clears_partition(lite_container) -> None:
    asyncio.run(
        lite_container.context_graph.apply_plan_async(
            _topology_plan(_POT), expected_pot_id=_POT
        )
    )
    out = asyncio.run(lite_container.graph_writer.reset_pot(_POT))
    assert out["ok"] is True
    assert out["group_id_nodes_remaining"] == 0
    reader = lite_container.context_graph._orchestrator.claim_query
    assert reader.find_claims(ClaimQueryFilter(pot_id=_POT, predicate_in=("DEPENDS_ON",))) == []
