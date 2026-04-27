"""Integration-style scenario for causal search + resolve (see docs 04-causal-multihop-context.md)."""

from __future__ import annotations

import pytest

from adapters.outbound.graphiti.query_helpers import merge_causal_expanded_search_rows


class _ScenarioStructural:
    """Simulates Neo4j: semantic top hits miss ``scaling-pain`` but one hop finds it."""

    def expand_causal_neighbours(
        self,
        pot_id: str,
        node_uuids: list[str],
        *,
        depth: int = 1,
    ) -> list[dict]:
        _ = depth
        _ = pot_id
        # Top semantic hit seeds include ``migration_decision``; causal edge to root cause.
        if "episode-migration-decision" in node_uuids:
            return [
                {
                    "neighbor_uuid": "episode-mongodb-scaling-pain",
                    "name": "MongoDB scaling pain (Mar 2025)",
                    "summary": "Write contention; 40+ minute aggregations on ledger paths.",
                    "edge_uuid": "rel-cause-1",
                    "edge_name": "CAUSED",
                    "seed_uuid": "episode-migration-decision",
                }
            ]
        return []


@pytest.fixture
def enable_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_CAUSAL_EXPAND", "1")


def test_ledger_migration_query_surfaces_scaling_pain_in_top_five(
    monkeypatch: pytest.MonkeyPatch, enable_causal: None
) -> None:
    """
    Flat semantic rank can miss the root-cause episode; one-hop CAUSED expansion
    pulls ``episode-mongodb-scaling-pain`` into the merged list.
    """
    stub = _ScenarioStructural()
    # Simulated semantic rows (edges): effects cluster / migration event — not the scaling episode.
    # Order matters: first rows seed ``episode-migration-decision`` within top-3 unique nodes.
    rows = [
        {
            "uuid": "edge-decision",
            "name": "DECIDES_FOR",
            "summary": "Decision to migrate ledger off MongoDB",
            "score": 0.48,
            "source_node_uuid": "episode-migration-decision",
            "target_node_uuid": "ledger-svc",
        },
        {
            "uuid": "edge-decom",
            "name": "DECOMMISSIONED",
            "summary": "Old cluster torn down after migration",
            "score": 0.55,
            "source_node_uuid": "cluster-old",
            "target_node_uuid": "episode-migration-completed",
        },
        {
            "uuid": "edge-mig",
            "name": "MIGRATED_TO",
            "summary": "Ledger store cut over to Postgres",
            "score": 0.52,
            "source_node_uuid": "ledger-svc",
            "target_node_uuid": "episode-migration-completed",
        },
    ]
    out = merge_causal_expanded_search_rows(
        rows, stub, "pot-ledger-demo", limit=5
    )
    uuids = [r["uuid"] for r in out]
    assert "episode-mongodb-scaling-pain" in uuids
    assert uuids.index("episode-mongodb-scaling-pain") < 5
