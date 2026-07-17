"""Live end-to-end integration tests against a real Neo4j (see e2e_surface.md).

Each test resolves a host-managed pot, applies explicit graph mutations, and
reads them back via direct Cypher against the canonical ``:RELATES_TO`` graph.
The module skips entirely if Neo4j is unreachable; each pot's partition is reset
on teardown (see conftest).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from potpie_context_core.domain.context_events import EventRef
from potpie_context_core.domain.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceRef
from potpie_context_core.domain.ports.pot_resolution import RepoRef
from potpie_context_core.domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
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


def _apply_plan(container, plan: ReconciliationPlan, *, expected_pot_id: str):
    assert container.backend is not None
    return asyncio.run(
        container.backend.mutation.apply_async(
            plan,
            expected_pot_id=expected_pot_id,
        )
    )


def _count_entities(settings, pot_id: str) -> int:
    """Direct Cypher count of canonical entity nodes in a pot partition.

    Replaces the removed ``structural.get_graph_overview(...)["totals"]
    ["entities"]`` read; the structural read stack was deleted along with the episodic tier.
    """
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


def _label_counts(settings, pot_id: str) -> dict[str, int]:
    """Direct Cypher per-label counts of canonical entity nodes in a pot."""
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    )
    try:
        with drv.session() as session:
            rows = session.run(
                "MATCH (e:Entity {group_id:$g}) UNWIND labels(e) AS lbl "
                "RETURN lbl AS label, count(*) AS n",
                g=pot_id,
            )
            return {rec["label"]: int(rec["n"]) for rec in rows}
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


def _topology_plan(pot_id: str) -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e2e-plan", source_system="test", pot_id=pot_id),
        summary="topology",
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"name": "web"}),
            EntityUpsert("service:auth", ("Entity", "Service"), {"name": "auth"}),
            EntityUpsert(
                "repo:acme-platform",
                ("Entity", "Repository"),
                {"name": "acme/platform"},
            ),
            EntityUpsert(
                "datastore:web-cache", ("Entity", "DataStore"), {"name": "web-cache"}
            ),
            EntityUpsert(
                "environment:prod", ("Entity", "Environment"), {"name": "prod"}
            ),
            EntityUpsert(
                "cluster:aws-use1", ("Entity", "Cluster"), {"name": "aws-use1"}
            ),
            EntityUpsert("person:alice", ("Entity", "Person"), {"name": "alice"}),
            EntityUpsert("team:platform", ("Entity", "Team"), {"name": "platform"}),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEFINED_IN", "service:web", "repo:acme-platform", {"path": "apps/web"}
            ),
            EdgeUpsert("DEPENDS_ON", "service:web", "service:auth", {}),
            EdgeUpsert("USES", "service:web", "datastore:web-cache", {}),
            EdgeUpsert("HOSTED_ON", "environment:prod", "cluster:aws-use1", {}),
            EdgeUpsert("MEMBER_OF", "person:alice", "team:platform", {}),
            EdgeUpsert("OWNED_BY", "service:web", "team:platform", {}),
        ],
        edge_deletes=[],
        invalidations=[],
    )


# ---------------------------------------------------------------------------
# Environment & container
# ---------------------------------------------------------------------------


class TestEnvironmentAndContainer:
    def test_neo4j_enabled(self, container, settings) -> None:
        assert settings.neo4j_uri()
        assert container.graph_writer.enabled is True
        assert container.context_graph is not None

    def test_container_wires_topology_ports(self, container) -> None:
        assert container.graph_writer is not None
        assert container.context_graph is not None
        assert container.context_graph.enabled is True


# ---------------------------------------------------------------------------
# Pot lifecycle (host-managed; engine resolves)
# ---------------------------------------------------------------------------


class TestPotLifecycle:
    def test_resolve_pot_returns_repo(self, container, pot_id, repo_name) -> None:
        resolved = container.pots.resolve_pot(pot_id)
        assert resolved is not None
        assert resolved.pot_id == pot_id
        assert resolved.primary_repo() is not None
        assert resolved.primary_repo().repo_name == repo_name

    def test_known_pot_ids_includes_pot(self, container, pot_id) -> None:
        assert pot_id in container.pots.known_pot_ids()

    def test_find_pots_for_repo(self, container, pot_id, repo_name) -> None:
        owner, repo = repo_name.split("/", 1)
        ref = RepoRef(
            provider="github", provider_host="github.com", owner=owner, repo=repo
        )
        assert pot_id in container.pots.find_pots_for_repo(ref)

    def test_unknown_pot_resolves_to_none(self, container) -> None:
        assert container.pots.resolve_pot("nonexistent-pot-xyz") is None


# ---------------------------------------------------------------------------
# Backend mutation path
# ---------------------------------------------------------------------------


class TestApplyPlanIngestion:
    def test_apply_plan_writes_full_topology(self, container, settings, pot_id) -> None:
        plan = _topology_plan(pot_id)
        res = _apply_plan(container, plan, expected_pot_id=pot_id)
        assert res.ok is True
        assert res.mutation_summary.entity_upserts_applied == 8
        assert res.mutation_summary.edge_upserts_applied == 6

        preds = {row["pred"] for row in _read_relates_to(settings, pot_id)}
        assert {
            "DEFINED_IN",
            "DEPENDS_ON",
            "USES",
            "HOSTED_ON",
            "MEMBER_OF",
            "OWNED_BY",
        } <= preds

    def test_off_catalog_edge_is_rejected(self, container, settings, pot_id) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-reject")
        asyncio.run(
            container.graph_writer.upsert_entities(
                pot_id,
                [
                    EntityUpsert("service:a", ("Entity", "Service"), {"name": "a"}),
                    EntityUpsert("service:b", ("Entity", "Service"), {"name": "b"}),
                ],
                prov,
            )
        )
        written = asyncio.run(
            container.graph_writer.upsert_edges(
                pot_id,
                [
                    EdgeUpsert(
                        "DEPENDS_ON", "service:a", "service:b", {"source_ref": "ok"}
                    ),
                    EdgeUpsert(
                        "TOTALLY_FAKE", "service:a", "service:b", {"source_ref": "bad"}
                    ),
                ],
                prov,
            )
        )
        assert written == 1  # fake predicate dropped by the writer
        preds = {row["pred"] for row in _read_relates_to(settings, pot_id)}
        assert preds == {"DEPENDS_ON"}

    def test_owned_by_singleton_supersedes_prior_owner(
        self, container, settings, pot_id
    ) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-singleton")
        asyncio.run(
            container.graph_writer.upsert_entities(
                pot_id,
                [
                    EntityUpsert(
                        "service:billing", ("Entity", "Service"), {"name": "billing"}
                    ),
                    EntityUpsert(
                        "team:payments", ("Entity", "Team"), {"name": "payments"}
                    ),
                    EntityUpsert(
                        "team:platform", ("Entity", "Team"), {"name": "platform"}
                    ),
                ],
                prov,
            )
        )
        det = {"evidence_strength": "deterministic", "source_ref": "owner:1"}
        asyncio.run(
            container.graph_writer.upsert_edges(
                pot_id,
                [EdgeUpsert("OWNED_BY", "service:billing", "team:payments", dict(det))],
                prov,
            )
        )
        # New deterministic owner supersedes the old (OWNED_BY is singleton).
        asyncio.run(
            container.graph_writer.upsert_edges(
                pot_id,
                [
                    EdgeUpsert(
                        "OWNED_BY",
                        "service:billing",
                        "team:platform",
                        {"evidence_strength": "deterministic", "source_ref": "owner:2"},
                    )
                ],
                prov,
            )
        )

        owned = _read_relates_to(settings, pot_id, predicate="OWNED_BY")
        live = [r for r in owned if r["invalid_at"] is None]
        superseded = [r for r in owned if r["invalid_at"] is not None]
        assert len(live) == 1
        assert live[0]["obj"] == "team:platform"
        assert len(superseded) == 1
        assert superseded[0]["obj"] == "team:payments"


# ---------------------------------------------------------------------------
# Read-back the canonical topology (direct Cypher)
# ---------------------------------------------------------------------------


class TestCanonicalReadback:
    def test_apply_plan_populates_canonical_entities(
        self, container, settings, pot_id
    ) -> None:
        _apply_plan(container, _topology_plan(pot_id), expected_pot_id=pot_id)
        assert _count_entities(settings, pot_id) == 8
        labels = _label_counts(settings, pot_id)
        assert labels.get("Service", 0) == 2
        for lbl in (
            "Repository",
            "Environment",
            "DataStore",
            "Cluster",
            "Team",
            "Person",
        ):
            assert labels.get(lbl, 0) >= 1
        # Topology edges land as canonical :RELATES_TO claims.
        preds = {row["pred"] for row in _read_relates_to(settings, pot_id)}
        assert "DEFINED_IN" in preds


# ---------------------------------------------------------------------------
# Pot reset
# ---------------------------------------------------------------------------


class TestPotReset:
    def test_reset_pot_clears_partition(self, container, settings, pot_id) -> None:
        _apply_plan(container, _topology_plan(pot_id), expected_pot_id=pot_id)
        assert _count_entities(settings, pot_id) == 8

        out = asyncio.run(container.graph_writer.reset_pot(pot_id))
        assert out["ok"] is True

        assert _count_entities(settings, pot_id) == 0
        assert _read_relates_to(settings, pot_id) == []


# ---------------------------------------------------------------------------
# Bitemporal contract — the canonical :RELATES_TO valid_at/invalid_at model
# ---------------------------------------------------------------------------


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TestBitemporalContract:
    """Lock the write-side bitemporal invariants that point-in-time reads rest on.

    The singleton test (above) asserts *which* edge is live after supersession;
    this asserts the *temporal* shape — explicit ``valid_at``, ``invalid_at``
    stamped at exactly the new claim's event-time, and a point-in-time selection
    that returns the right owner at any instant.
    """

    _T1 = "2025-01-01T00:00:00+00:00"
    _T2 = "2025-06-01T00:00:00+00:00"

    def _seed_supersession(self, container, pot_id: str) -> None:
        prov = ProvenanceRef(pot_id=pot_id, source_event_id="e2e-bitemporal")
        asyncio.run(
            container.graph_writer.upsert_entities(
                pot_id,
                [
                    EntityUpsert(
                        "service:billing", ("Entity", "Service"), {"name": "billing"}
                    ),
                    EntityUpsert(
                        "team:payments", ("Entity", "Team"), {"name": "payments"}
                    ),
                    EntityUpsert(
                        "team:platform", ("Entity", "Team"), {"name": "platform"}
                    ),
                ],
                prov,
            )
        )
        asyncio.run(
            container.graph_writer.upsert_edges(
                pot_id,
                [
                    EdgeUpsert(
                        "OWNED_BY",
                        "service:billing",
                        "team:payments",
                        {
                            "evidence_strength": "deterministic",
                            "source_ref": "owner:1",
                            "valid_at": self._T1,
                        },
                    )
                ],
                prov,
            )
        )
        asyncio.run(
            container.graph_writer.upsert_edges(
                pot_id,
                [
                    EdgeUpsert(
                        "OWNED_BY",
                        "service:billing",
                        "team:platform",
                        {
                            "evidence_strength": "deterministic",
                            "source_ref": "owner:2",
                            "valid_at": self._T2,
                        },
                    )
                ],
                prov,
            )
        )

    def test_supersession_stamps_bitemporal_fields(
        self, container, settings, pot_id
    ) -> None:
        self._seed_supersession(container, pot_id)
        rows = {
            r["obj"]: r
            for r in _read_relates_to(settings, pot_id, predicate="OWNED_BY")
        }
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
        self, container, settings, pot_id
    ) -> None:
        self._seed_supersession(container, pot_id)
        # Between T1 and T2 the old owner is live; after T2 the new owner is.
        assert (
            _live_owner_as_of(settings, pot_id, "2025-03-01T00:00:00+00:00")
            == "team:payments"
        )
        assert (
            _live_owner_as_of(settings, pot_id, "2025-12-01T00:00:00+00:00")
            == "team:platform"
        )
        # Before either claim, no owner is live.
        assert _live_owner_as_of(settings, pot_id, "2024-01-01T00:00:00+00:00") is None


# ---------------------------------------------------------------------------
# Idempotency — re-applying the same plan must not duplicate edges
# ---------------------------------------------------------------------------


class TestPlanIdempotency:
    def test_reapplying_plan_is_idempotent(self, container, settings, pot_id) -> None:
        plan = _topology_plan(pot_id)
        first = _apply_plan(container, plan, expected_pot_id=pot_id)
        assert first.ok is True

        entities_before = _count_entities(settings, pot_id)
        edges_before = _read_relates_to(settings, pot_id)

        # Re-apply the identical plan (same source_refs → same MERGE keys).
        second = _apply_plan(container, plan, expected_pot_id=pot_id)
        assert second.ok is True

        entities_after = _count_entities(settings, pot_id)
        edges_after = _read_relates_to(settings, pot_id)

        assert entities_after == entities_before
        assert len(edges_after) == len(edges_before)
