"""Real FalkorDBLite coverage for native snapshot transactions."""

from pathlib import Path

import pytest

from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceContext
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import FalkorDBGraphBackend
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend

client = pytest.importorskip("redislite.falkordb_client")
pytestmark = pytest.mark.integration


class Settings:
    def is_enabled(self):
        return True

    def falkordb_graph_name(self):
        return "snapshot"


@pytest.fixture()
def graph(tmp_path: Path):
    database = client.FalkorDB(str(tmp_path / "snapshot.db"))
    try:
        yield database.select_graph("snapshot")
    finally:
        connection = getattr(database, "connection", None)
        if connection is not None and getattr(connection, "cleanupregistry", False):
            connection.shutdown(save=True, now=True, force=True)
        database.close()


def test_native_snapshot_round_trip_is_idempotent_and_advances_revision(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:web", ("Entity", "Service"), {"custom": "value"}),
                EntityUpsert("service:db", ("Entity", "Service"), {"isolated": True}),
            ],
            edge_upserts=[
                EdgeUpsert(
                    "DEPENDS_ON", "service:web", "service:db",
                    {"claim_key": "claim:source:one", "source_ref": "config:1", "truth": "source_observation"},
                )
            ],
        ),
        expected_pot_id="source",
        provenance_context=ProvenanceContext(mutation_id="seed", source_event_id="source:seed"),
    )
    payload = backend.snapshot.export_data(pot_id="source")
    imported = backend.snapshot.import_data(pot_id="target", payload=payload)
    assert (imported.entity_count, imported.claim_count) == (2, 1)
    assert backend.snapshot.export_data(pot_id="target")["claims"][0]["claim_key"] == "claim:target:one"
    first_version = backend.mutation.current_version("target")
    backend.snapshot.import_data(pot_id="target", payload=payload)
    assert backend.mutation.current_version("target") == first_version
    assert len(backend.snapshot.export_data(pot_id="target")["claims"]) == 1


def test_empty_export_and_invalid_label_do_not_create_revision_state(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    assert backend.snapshot.export_data(pot_id="fresh") == {
        "format_version": "2", "pot_id": "fresh", "entities": [], "claims": []
    }
    with pytest.raises(ValueError, match="invalid snapshot entity label"):
        backend.snapshot.import_data(
            pot_id="fresh",
            payload={
                "format_version": "2", "pot_id": "source",
                "entities": [{"key": "bad", "labels": ["Bad Label"], "properties": {}}],
                "claims": [],
            },
        )
    assert backend.mutation.current_version("fresh") == 0
    assert backend.snapshot.export_data(pot_id="fresh")["entities"] == []


@pytest.mark.parametrize(
    "reserved",
    ["__potpie_snapshot_properties_v2", "__potpie_snapshot_claim_fields_v2"],
)
def test_reserved_snapshot_property_is_rejected_before_native_mutation(
    graph, reserved: str
) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    with pytest.raises(ValueError, match="reserved key"):
        backend.snapshot.import_data(
            pot_id="target",
            payload={
                "format_version": "2", "pot_id": "source",
                "entities": [
                    {
                        "key": "entity:a", "labels": ["Entity"],
                        "properties": {reserved: "user"},
                    }
                ],
                "claims": [],
            },
        )
    assert backend.mutation.current_version("target") == 0


def test_conflict_rejects_before_revision_or_data_changes(graph) -> None:
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    base = {
        "format_version": "2", "pot_id": "source",
        "entities": [{"key": "service:web", "labels": ["Entity", "Service"], "properties": {"name": "web"}}],
        "claims": [],
    }
    backend.snapshot.import_data(pot_id="target", payload=base)
    version = backend.mutation.current_version("target")
    changed = {**base, "entities": [{**base["entities"][0], "properties": {"name": "other"}}]}
    with pytest.raises(ValueError, match="conflicts with target"):
        backend.snapshot.import_data(pot_id="target", payload=changed)
    assert backend.mutation.current_version("target") == version
    assert backend.snapshot.export_data(pot_id="target")["entities"][0]["properties"]["name"] == "web"


def test_nested_and_json_looking_properties_survive_falkor_embedded_falkor(
    graph, tmp_path: Path
) -> None:
    original = {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {
                "key": "service:web",
                "labels": ["Entity", "Service"],
                "properties": {
                    "aliases": ["web", "frontend"],
                    "nested": {"owners": ["team:a"], "tier": 1},
                    "json_looking_string": '{"must":"stay a string"}',
                    "optional": None,
                },
            },
            {"key": "service:db", "labels": ["Entity", "Service"], "properties": {}},
            {
                "key": "doc:isolated",
                "labels": ["Document", "Entity"],
                "properties": {"meta": {"kind": "runbook"}},
            },
        ],
        "claims": [
            {
                "claim_key": "claim:source:rich",
                "predicate": "DEPENDS_ON",
                "subject_key": "service:web",
                "object_key": "service:db",
                "valid_at": "2026-01-01T00:00:00+00:00",
                "invalid_at": "2026-08-01T00:00:00+00:00",
                "evidence_strength": "deterministic",
                "source_system": "github",
                "source_ref": "github:pr:1",
                "fact": "web depends on db",
                "fact_embedding": [0.12345678901234567, -0.000000000123456789],
                "subgraph": "infra_topology",
                "truth": "authoritative_fact",
                "confidence": 0.9,
                "description": "Rich snapshot claim",
                "environment": "prod",
                "observed_at": "2026-01-02T00:00:00+00:00",
                "valid_until": None,
                "mutation_id": "mutation:rich",
                "source_refs": ["github:pr:1"],
                "evidence": [{"source_ref": "github:pr:1", "authority": "external_system"}],
                "graph_contract_version": "1.5",
                "ontology_version": "1",
                "properties": {
                    "nested": {"regions": ["us", "eu"]},
                    "json_looking_string": "[1,2,3]",
                    "optional": None,
                },
            }
        ],
    }
    native = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    native.snapshot.import_data(pot_id="first", payload=original)
    native.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    "service:web", ("Entity", "Service"),
                    {"description": "patched after snapshot import"},
                )
            ]
        ),
        expected_pot_id="first",
        provenance_context=ProvenanceContext(
            mutation_id="post-import-patch", source_event_id="source:patch"
        ),
    )
    first = native.snapshot.export_data(pot_id="first")
    first_web = next(row for row in first["entities"] if row["key"] == "service:web")
    assert first_web["properties"]["description"] == "patched after snapshot import"
    assert first_web["properties"]["nested"] == {"owners": ["team:a"], "tier": 1}
    assert first_web["properties"]["json_looking_string"] == '{"must":"stay a string"}'
    assert first_web["properties"]["optional"] is None

    embedded = EmbeddedGraphBackend(home=tmp_path / "embedded")
    embedded.snapshot.import_data(pot_id="middle", payload=first)
    middle = embedded.snapshot.export_data(pot_id="middle")
    native.snapshot.import_data(pot_id="last", payload=middle)
    native.snapshot.import_data(pot_id="last", payload=middle)
    last = native.snapshot.export_data(pot_id="last")

    web = next(row for row in last["entities"] if row["key"] == "service:web")
    assert web["properties"]["description"] == "patched after snapshot import"
    assert web["properties"]["nested"] == {"owners": ["team:a"], "tier": 1}
    assert web["properties"]["json_looking_string"] == '{"must":"stay a string"}'
    assert web["properties"]["optional"] is None
    isolated = next(row for row in last["entities"] if row["key"] == "doc:isolated")
    assert isolated["properties"] == {"meta": {"kind": "runbook"}}
    claim = last["claims"][0]
    assert claim["properties"] == original["claims"][0]["properties"]
    assert claim["invalid_at"] == "2026-08-01T00:00:00+00:00"
    assert claim["evidence"] == original["claims"][0]["evidence"]
    assert claim["fact_embedding"] == original["claims"][0]["fact_embedding"]
    vector = original["claims"][0]["fact_embedding"]
    result = graph.ro_query(
        "MATCH ()-[r:RELATES_TO {group_id:$pot,claim_key:$key}]->() "
        "RETURN vec.cosineDistance(r.fact_embedding,vecf32($embedding))",
        params={"pot": "last", "key": "claim:last:rich", "embedding": vector},
    )
    assert result.result_set[0][0] == pytest.approx(0.0, abs=1e-6)
