from __future__ import annotations

from datetime import datetime, timedelta, timezone

from potpie_context_engine.core.graph_contract import GRAPH_CONTRACT_VERSION, ONTOLOGY_VERSION
from potpie_context_engine.core.graph_plans import GraphMutationDiff, GraphMutationPlanRecord


def test_commit_attempt_fields_preserve_existing_positional_constructor_order() -> None:
    created_at = datetime.now(timezone.utc)
    committed_at = created_at + timedelta(seconds=1)
    diff = GraphMutationDiff(entity_upserts=1)

    record = GraphMutationPlanRecord(
        "plan:1",
        "pot:1",
        "committed",
        "low",
        created_at,
        created_at + timedelta(hours=1),
        {"operations": []},
        (),
        (),
        (),
        (),
        None,
        None,
        {"topology": 1},
        {"topology": 1},
        diff,
        (),
        None,
        "mutation:1",
        committed_at,
        {"topology": 2},
        "done",
        GRAPH_CONTRACT_VERSION,
        ONTOLOGY_VERSION,
    )

    assert record.committed_at == committed_at
    assert record.final_subgraph_versions == {"topology": 2}
    assert record.detail == "done"
    assert record.graph_contract_version == GRAPH_CONTRACT_VERSION
    assert record.ontology_version == ONTOLOGY_VERSION
    assert record.commit_attempt_id is None
    assert record.commit_attempt_started_at is None
