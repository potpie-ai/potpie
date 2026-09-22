"""Real private FalkorDBLite checks for atomic batch application."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from potpie_context_core.errors import GraphMutationVersionConflict
from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
    FalkorDBGraphBackend,
)

client = pytest.importorskip("redislite.falkordb_client")
pytestmark = pytest.mark.integration


class Settings:
    def is_enabled(self):
        return True

    def falkordb_graph_name(self):
        return "atomic"


@pytest.fixture()
def graph(tmp_path: Path):
    db = client.FalkorDB(str(tmp_path / "atomic.db"))
    try:
        yield db.select_graph("atomic")
    finally:
        db.close()


def _plan(value: str) -> MutationBatch:
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service"), {"description": value})
        ]
    )


def _context(mid: str) -> ProvenanceContext:
    return ProvenanceContext(mutation_id=mid, source_event_id=f"source:{mid}")


def test_stale_parallel_apply_has_one_winner(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                backend.mutation.compare_and_apply,
                _plan(value),
                expected_pot_id="p",
                expected_version=0,
                provenance_context=_context(value),
            )
            for value in ("first", "second")
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result().ok)
        except GraphMutationVersionConflict:
            outcomes.append(False)
    assert sorted(outcomes) == [False, True]
    assert backend.mutation.current_version("p") == 1


def test_entity_update_and_receipt_survive_backend_recreation(graph) -> None:
    first = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    result = first.mutation.compare_and_apply(
        _plan("updated"),
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("m1"),
    )
    second = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    replay = second.mutation.compare_and_apply(
        _plan("updated"),
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("m1"),
    )
    assert replay == result
    assert second.mutation.current_version("p") == 1
    props = second.claim_query.entity_properties(pot_id="p", entity_key="service:web")
    assert props["description"] == "updated"


def test_same_source_environment_claim_keys_remain_distinct(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service")),
            EntityUpsert("service:db", ("Entity", "Service")),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:web",
                "service:db",
                {
                    "source_ref": "config:shared",
                    "claim_key": f"claim:{env}",
                    "environment": env,
                    "truth": "source_observation",
                },
            )
            for env in ("prod", "staging")
        ],
    )
    result = backend.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("scoped"),
    )
    assert result.mutation_summary.edge_upserts_applied == 2
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", predicate_in=("DEPENDS_ON",))
    )
    assert {(row.environment, row.claim_key) for row in rows} == {
        ("prod", "claim:prod"),
        ("staging", "claim:staging"),
    }


def test_reset_advances_revision_preserves_receipt_and_blocks_stale_write(
    graph,
) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    plan = _plan("before reset")
    original = backend.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("original"),
    )
    report = backend.mutation.reset_pot("p")
    assert report["ok"] and report["version"] == 2
    assert not backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:web"
    )
    with pytest.raises(GraphMutationVersionConflict):
        backend.mutation.compare_and_apply(
            _plan("stale"),
            expected_pot_id="p",
            expected_version=1,
            provenance_context=_context("stale"),
        )
    replay = backend.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("original"),
    )
    assert replay == original
    assert not backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:web"
    )


def test_raw_invalidation_is_atomic_and_advances_revision(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service")),
            EntityUpsert("service:db", ("Entity", "Service")),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:web",
                "service:db",
                {"claim_key": "claim:dependency", "source_ref": "config:one"},
            )
        ],
    )
    backend.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=_context("seed"),
    )
    assert backend.mutation.invalidate(
        pot_id="p", claim_keys=["claim:dependency"], reason="retired"
    ) == 1
    assert backend.mutation.current_version("p") == 2
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="p", include_invalidated=True)
    )
    assert len(rows) == 1
    assert rows[0].invalid_at is not None


@pytest.mark.parametrize("expiry_days", [7, -1], ids=["future", "expired"])
def test_evidence_marker_refresh_preserves_invalidation_and_claim(
    graph, expiry_days
) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    invalid_at = (
        datetime.now(timezone.utc) + timedelta(days=expiry_days)
    ).isoformat()

    def batch(properties):
        return MutationBatch(
            entity_upserts=[
                EntityUpsert("document:guide", ("Entity", "Document")),
                EntityUpsert("service:api", ("Entity", "Service")),
            ],
            edge_upserts=[
                EdgeUpsert(
                    "DOCUMENTS",
                    "document:guide",
                    "service:api",
                    {
                        "claim_key": "claim:guide-api",
                        "source_ref": "resource:guide",
                        "fact": "Guide documents API behavior",
                        **properties,
                    },
                )
            ],
        )

    backend.mutation.apply(
        batch(
            {
                "invalid_at": invalid_at,
                "invalidation_reason": "scheduled source retirement",
                "invalidated_by": "refresh:scheduled",
            }
        ),
        expected_pot_id="marker",
        provenance_context=_context("marker-seed"),
    )
    before = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="marker",
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )[0]
    backend.mutation.apply(
        batch(
            {
                "evidence_review_required": True,
                "evidence_review_reason": "source version changed",
            }
        ),
        expected_pot_id="marker",
        provenance_context=_context("marker-refresh"),
    )

    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="marker",
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )
    assert len(rows) == 1
    assert rows[0].invalid_at.isoformat() == invalid_at
    assert rows[0].fact == "Guide documents API behavior"
    assert rows[0].properties["invalidation_reason"] == "scheduled source retirement"
    assert rows[0].properties["evidence_review_required"] is True
    assert rows[0].properties.get("revived_at") == before.properties.get("revived_at")

    backend.mutation.apply(
        batch({}),
        expected_pot_id="marker",
        provenance_context=_context("ordinary-reassert"),
    )
    revived = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id="marker",
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )
    assert len(revived) == 1
    assert revived[0].invalid_at is None


def test_single_query_failure_rolls_back_graph_revision_and_receipt(
    graph, monkeypatch
) -> None:
    from potpie_context_engine.adapters.outbound.graph import falkordb_atomic

    original = falkordb_atomic._compile

    def broken(*args, **kwargs):
        query, params, result = original(*args, **kwargs)
        return (
            query.replace("RETURN v.version", "WITH v RETURN keys(1)"),
            params,
            result,
        )

    monkeypatch.setattr(falkordb_atomic, "_compile", broken)
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    mixed = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:web", ("Entity", "Service")),
            EntityUpsert("service:db", ("Entity", "Service")),
        ],
        edge_upserts=[EdgeUpsert("DEPENDS_ON", "service:web", "service:db")],
    )
    with pytest.raises(Exception):
        backend.mutation.compare_and_apply(
            mixed,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=_context("failed"),
        )
    assert backend.mutation.current_version("p") == 0
    assert not backend.claim_query.entity_properties(
        pot_id="p", entity_key="service:web"
    )
    assert (
        backend.mutation.lookup_execution(
            mixed, expected_pot_id="p", mutation_id="failed"
        ).state
        == "absent"
    )
