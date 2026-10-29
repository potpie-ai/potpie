"""Legacy JSON time defects remain visible without becoming stored graph fields."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    dump_store,
    load_store,
)
from potpie_context_engine.adapters.outbound.graph.memory_journal import (
    records_from_store,
)
from potpie_context_engine.core.mutation_plan_store import InMemoryGraphPlanStore
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.workbench_service import GraphWorkbenchService

pytestmark = pytest.mark.unit
POT = "diagnostic-pot"
START = "2025-01-01T00:00:00+00:00"

DEFECTS = [
    pytest.param(
        {"invalid_at": "malformed"}, {}, {"invalid_at"}, id="malformed-invalidation"
    ),
    pytest.param(
        {"valid_until": "malformed"}, {}, {"valid_until"}, id="malformed-scheduled-end"
    ),
    pytest.param(
        {"valid_at": "2025-01-01T00:00:00"}, {}, {"valid_at"}, id="naive-start"
    ),
    pytest.param(
        {"observed_at": "2025-01-01"}, {}, {"observed_at"}, id="date-only-observation"
    ),
    pytest.param(
        {},
        {"valid_from": "2025-02-01T00:00:00Z"},
        {"valid_at", "valid_from"},
        id="contradictory-start-alias",
    ),
    pytest.param(
        {},
        {"valid_at": "2025-02-01T00:00:00Z"},
        {"valid_at"},
        id="contradictory-same-field",
    ),
    pytest.param(
        {"invalid_at": "2025-03-01T00:00:00Z"},
        {"invalid_at": "malformed"},
        {"invalid_at"},
        id="masked-malformed-property",
    ),
    pytest.param(
        {"valid_until": "2024-01-01T00:00:00Z"},
        {},
        {"valid_until"},
        id="nonpositive-interval",
    ),
    pytest.param(
        {"invalid_at": "2025-02-01T00:00:00.000000001Z"},
        {},
        {"invalid_at"},
        id="submicrosecond-end",
    ),
]


def _state(canonical, properties):
    return {
        "format_version": "2",
        "rows": [
            {
                "pot_id": POT,
                "predicate": "RELATED_TO",
                "subject_key": "service:a",
                "object_key": "service:b",
                "claim_key": "claim:legacy",
                "record_id": "record:legacy",
                "valid_at": START,
                "properties": deepcopy(properties),
                **canonical,
            }
        ],
        "labels": [
            [POT, key, ["Entity", "Service"]] for key in ("service:a", "service:b")
        ],
        "entity_properties": [
            [POT, key, {"record_id": f"record:{key}"}]
            for key in ("service:a", "service:b")
        ],
    }


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize("canonical,properties,diagnostic_keys", DEFECTS)
def test_raw_time_diagnostics_survive_load_without_serializing(
    profile, canonical, properties, diagnostic_keys, tmp_path
):
    state = _state(canonical, properties)
    before = deepcopy(state)
    if profile == "json":
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(state))
        disk = path.read_bytes()
        backend = EmbeddedGraphBackend(home=tmp_path)
        store = backend._inner.store
    else:
        store = load_store(state)
        backend = InMemoryGraphBackend(store=store)
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, include_invalidated=True)
    )
    assert len(rows) == 1
    assert diagnostic_keys <= rows[0].temporal_diagnostics.keys()
    assert all(rows[0].temporal_diagnostics[key] for key in diagnostic_keys)
    if canonical.get("invalid_at") == "malformed":
        assert rows[0].invalid_at is None
    if canonical.get("valid_until") == "malformed":
        assert rows[0].valid_until is None
    assert state == before
    assert "temporal_diagnostics" not in json.dumps(dump_store(store))
    assert "temporal_diagnostics" not in json.dumps(
        backend.snapshot.export_data(pot_id=POT)
    )
    for record in records_from_store(store, POT).values():
        assert "temporal_diagnostics" not in record.fields
    if profile == "json":
        assert path.read_bytes() == disk


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_equivalent_offset_aliases_have_no_diagnostics(profile, tmp_path):
    state = _state({}, {"valid_from": "2025-01-01T05:30:00+05:30"})
    if profile == "json":
        (tmp_path / "graph.json").write_text(json.dumps(state))
        backend = EmbeddedGraphBackend(home=tmp_path)
    else:
        backend = InMemoryGraphBackend(store=load_store(state))
    row = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))[0]
    assert row.temporal_diagnostics == {}


@pytest.mark.parametrize("canonical,properties,diagnostic_keys", DEFECTS)
def test_verification_refuses_matching_for_legacy_json_time_defects(
    canonical, properties, diagnostic_keys, tmp_path
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=InMemoryGraphPlanStore()
    )
    proposal = workbench.propose(
        {
            "operations": [
                {
                    "op": "link_entities",
                    "subgraph": "infra_topology",
                    "subject": {"key": "service:a", "type": "Service"},
                    "predicate": "DEPENDS_ON",
                    "object": {"key": "service:b", "type": "Service"},
                    "truth": "source_observation",
                    "evidence": [{"source_ref": "repo:manifest"}],
                    "description": "service dependency",
                    "valid_from": START,
                }
            ]
        },
        pot_id=POT,
    )
    assert proposal.ok
    receipt = workbench.commit(proposal.plan_id, pot_id=POT)
    assert receipt.ok
    path = tmp_path / "graph.json"
    state = json.loads(path.read_text())
    state["rows"][0].update(canonical)
    state["rows"][0]["properties"].update(properties)
    path.write_text(json.dumps(state))
    disk = path.read_bytes()
    loaded = EmbeddedGraphBackend(home=tmp_path)
    restarted = GraphWorkbenchService(backend=loaded, plan_store=workbench.plan_store)
    report = restarted.verify_commit(proposal.plan_id, pot_id=POT)
    assert report.outcome == "inconclusive"
    assert not report.ok
    outcomes = report.content_readback["claim_outcomes"]
    assert outcomes[0]["outcome"] == "inconclusive"
    assert diagnostic_keys <= outcomes[0]["lifecycle"]["temporal_diagnostics"].keys()
    assert (
        restarted.commit_status(proposal.plan_id, pot_id=POT).execution
        == receipt.execution
    )
    assert loaded.mutation.current_version(POT) == 1
    assert path.read_bytes() == disk
