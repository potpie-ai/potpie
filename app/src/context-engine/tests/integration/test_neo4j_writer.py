"""Writer-shape invariants — live Neo4j, no LLM / no Postgres.

These tests exercise the Cypher writer directly (predicate allowlist,
``OWNED_BY`` singleton supersession, bitemporal ``valid_at``/``invalid_at``
stamping) rather than going through the ``GraphMutationPort`` /
``ReconciliationPlan`` path. They lived in ``test_e2e_topology.py`` until
the graph-backend cleanup (item 1 of the graph-backend TODO) removed
``container.graph_writer`` as an exposed slot; they now build a writer
from settings directly and assert against the canonical ``:RELATES_TO``
graph with direct Cypher reads.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_e2e_topology — kept narrow and self-contained
# so writer-shape tests don't drag the topology-test module along)
# ---------------------------------------------------------------------------


def _read_relates_to(settings, pot_id: str, predicate: str | None = None) -> list[dict]:
    """Direct Cypher read of canonical :RELATES_TO edges for exact assertions."""
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    )
    try:
        with drv.session() as session:
            where = "WHERE r.name = $p " if predicate else ""
            rows = session.run(
                "MATCH (a:Entity {group_id:$g})-[r:RELATES_TO]->(b:Entity {group_id:$g}) "
                + where
                + "RETURN a.entity_key AS subj, r.name AS pred, b.entity_key AS obj, "
                "r.environment AS environment, r.evidence_strength AS strength, "
                "r.valid_at AS valid_at, r.invalid_at AS invalid_at, "
                "r.expired_at AS expired_at, "
                "r.superseded_by_object AS superseded_by_object",
                g=pot_id,
                **({"p": predicate} if predicate else {}),
            )
            return [dict(rec) for rec in rows]
    finally:
        drv.close()


def _live_owner_as_of(settings, pot_id: str, as_of_iso: str) -> str | None:
    """Object of the OWNED_BY edge live at ``as_of`` (the bitemporal predicate).

    This is the exact point-in-time selection a future as_of-aware reader must
    implement over canonical ``:RELATES_TO`` edges: an edge is live at ``T`` when
    ``valid_at <= T`` and (``invalid_at`` is null or ``invalid_at > T``). Encoded
    here against live data so a regression in the writer's bitemporal stamping is
    caught even though no read surface exposes ``as_of`` for these edges yet.
    """
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    )
    try:
        with drv.session() as session:
            rec = session.run(
                "MATCH (a:Entity {group_id:$g})-[r:RELATES_TO {name:'OWNED_BY'}]->"
                "(b:Entity {group_id:$g}) "
                "WHERE r.valid_at <= $as_of "
                "  AND (r.invalid_at IS NULL OR r.invalid_at > $as_of) "
                "RETURN b.entity_key AS obj",
                g=pot_id,
                as_of=as_of_iso,
            ).single()
            return rec["obj"] if rec is not None else None
    finally:
        drv.close()


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


@pytest.fixture()
def neo4j_writer(settings):
    """A live Neo4j writer for tests that drive write-shape invariants."""
    from adapters.outbound.graph.backends.neo4j.writer import Neo4jGraphWriter

    return Neo4jGraphWriter(settings)


# ---------------------------------------------------------------------------
# Predicate allowlist — the Cypher writer drops off-catalog predicates
# ---------------------------------------------------------------------------


class TestPredicateAllowlist:
    def test_off_catalog_edge_is_rejected(
        self, container, neo4j_writer, settings, pot_id
    ) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-reject")
        asyncio.run(neo4j_writer.upsert_entities(
            pot_id,
            [
                EntityUpsert("service:a", ("Entity", "Service"), {"name": "a"}),
                EntityUpsert("service:b", ("Entity", "Service"), {"name": "b"}),
            ],
            prov,
        ))
        written = asyncio.run(neo4j_writer.upsert_edges(
            pot_id,
            [
                EdgeUpsert("DEPENDS_ON", "service:a", "service:b", {"source_ref": "ok"}),
                EdgeUpsert("TOTALLY_FAKE", "service:a", "service:b", {"source_ref": "bad"}),
            ],
            prov,
        ))
        assert written == 1  # fake predicate dropped by the writer
        preds = {row["pred"] for row in _read_relates_to(settings, pot_id)}
        assert preds == {"DEPENDS_ON"}


# ---------------------------------------------------------------------------
# OWNED_BY singleton — a new deterministic owner supersedes the prior one
# ---------------------------------------------------------------------------


class TestOwnedBySingleton:
    def test_owned_by_singleton_supersedes_prior_owner(
        self, container, neo4j_writer, settings, pot_id
    ) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-singleton")
        asyncio.run(neo4j_writer.upsert_entities(
            pot_id,
            [
                EntityUpsert("service:billing", ("Entity", "Service"), {"name": "billing"}),
                EntityUpsert("team:payments", ("Entity", "Team"), {"name": "payments"}),
                EntityUpsert("team:platform", ("Entity", "Team"), {"name": "platform"}),
            ],
            prov,
        ))
        det = {"evidence_strength": "deterministic", "source_ref": "owner:1"}
        asyncio.run(neo4j_writer.upsert_edges(
            pot_id,
            [EdgeUpsert("OWNED_BY", "service:billing", "team:payments", dict(det))],
            prov,
        ))
        # New deterministic owner supersedes the old (OWNED_BY is singleton).
        asyncio.run(neo4j_writer.upsert_edges(
            pot_id,
            [EdgeUpsert("OWNED_BY", "service:billing", "team:platform", {"evidence_strength": "deterministic", "source_ref": "owner:2"})],
            prov,
        ))

        owned = _read_relates_to(settings, pot_id, predicate="OWNED_BY")
        live = [r for r in owned if r["invalid_at"] is None]
        superseded = [r for r in owned if r["invalid_at"] is not None]
        assert len(live) == 1
        assert live[0]["obj"] == "team:platform"
        assert len(superseded) == 1
        assert superseded[0]["obj"] == "team:payments"


# ---------------------------------------------------------------------------
# Bitemporal contract — the canonical :RELATES_TO valid_at/invalid_at model
# ---------------------------------------------------------------------------


class TestBitemporalContract:
    """Lock the write-side bitemporal invariants that point-in-time reads rest on.

    The singleton test above asserts *which* edge is live after supersession;
    this asserts the *temporal* shape — explicit ``valid_at``, ``invalid_at``
    stamped at exactly the new claim's event-time, and a point-in-time selection
    that returns the right owner at any instant.
    """

    _T1 = "2025-01-01T00:00:00+00:00"
    _T2 = "2025-06-01T00:00:00+00:00"

    def _seed_supersession(self, writer, pot_id: str) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-bitemporal")
        asyncio.run(writer.upsert_entities(
            pot_id,
            [
                EntityUpsert("service:billing", ("Entity", "Service"), {"name": "billing"}),
                EntityUpsert("team:payments", ("Entity", "Team"), {"name": "payments"}),
                EntityUpsert("team:platform", ("Entity", "Team"), {"name": "platform"}),
            ],
            prov,
        ))
        asyncio.run(writer.upsert_edges(
            pot_id,
            [
                EdgeUpsert(
                    "OWNED_BY",
                    "service:billing",
                    "team:payments",
                    {"evidence_strength": "deterministic", "source_ref": "owner:1", "valid_at": self._T1},
                )
            ],
            prov,
        ))
        asyncio.run(writer.upsert_edges(
            pot_id,
            [
                EdgeUpsert(
                    "OWNED_BY",
                    "service:billing",
                    "team:platform",
                    {"evidence_strength": "deterministic", "source_ref": "owner:2", "valid_at": self._T2},
                )
            ],
            prov,
        ))

    def test_supersession_stamps_bitemporal_fields(
        self, container, neo4j_writer, settings, pot_id
    ) -> None:
        self._seed_supersession(neo4j_writer, pot_id)
        rows = {r["obj"]: r for r in _read_relates_to(settings, pot_id, predicate="OWNED_BY")}
        old, new = rows["team:payments"], rows["team:platform"]

        # Old claim: valid from T1, invalidated at exactly the new claim's
        # event-time (T2), with a system-time expiry stamp and a back-pointer.
        assert old["valid_at"] == self._T1
        assert old["invalid_at"] == new["valid_at"] == self._T2
        assert _dt(old["valid_at"]) < _dt(old["invalid_at"])
        assert old["expired_at"] is not None
        assert old["superseded_by_object"] == "team:platform"
        # New claim: live from T2 onward (no invalidation).
        assert new["invalid_at"] is None

    def test_point_in_time_selection_returns_owner_at_instant(
        self, container, neo4j_writer, settings, pot_id
    ) -> None:
        self._seed_supersession(neo4j_writer, pot_id)
        # Between T1 and T2 the old owner is live; after T2 the new owner is.
        assert _live_owner_as_of(settings, pot_id, "2025-03-01T00:00:00+00:00") == "team:payments"
        assert _live_owner_as_of(settings, pot_id, "2025-12-01T00:00:00+00:00") == "team:platform"
        # Before either claim, no owner is live.
        assert _live_owner_as_of(settings, pot_id, "2024-01-01T00:00:00+00:00") is None
