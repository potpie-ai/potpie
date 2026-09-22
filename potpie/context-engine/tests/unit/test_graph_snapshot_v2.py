from __future__ import annotations

from copy import deepcopy

import pytest

from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)


def _backend_with_snapshot_data() -> InMemoryGraphBackend:
    backend = InMemoryGraphBackend()
    backend.store.add(
        ClaimRow(
            pot_id="source",
            claim_key="claim:source:one",
            predicate="DEPENDS_ON",
            subject_key="service:a",
            object_key="service:b",
            truth="authoritative_fact",
            evidence_strength="deterministic",
            properties={"weight": 2},
            source_refs=("github:pr:1",),
        )
    )
    backend.store.set_entity_label(
        pot_id="source", entity_key="service:a", labels=("Service",)
    )
    backend.store.set_entity_properties(
        pot_id="source", entity_key="service:a", properties={"name": "A"}
    )
    backend.store.set_entity_label(
        pot_id="source", entity_key="isolated", labels=("Document",)
    )
    backend.store.set_entity_properties(
        pot_id="source", entity_key="isolated", properties={"title": "Only node"}
    )
    return backend


def test_v2_round_trip_preserves_entities_metadata_and_is_idempotent() -> None:
    payload = _backend_with_snapshot_data().snapshot.export_data(pot_id="source")
    assert payload["format_version"] == "2"
    assert [row["key"] for row in payload["entities"]] == [
        "isolated", "service:a", "service:b"
    ]
    assert payload["claims"][0]["evidence_strength"] == "deterministic"

    target = InMemoryGraphBackend()
    first = target.snapshot.import_data(pot_id="target", payload=payload)
    second = target.snapshot.import_data(pot_id="target", payload=payload)

    assert first.entity_count == second.entity_count == 3
    assert first.claim_count == second.claim_count == 1
    exported = target.snapshot.export_data(pot_id="target")
    assert len(exported["claims"]) == 1
    assert exported["claims"][0]["claim_key"] == "claim:target:one"
    isolated = next(row for row in exported["entities"] if row["key"] == "isolated")
    assert isolated == {
        "key": "isolated",
        "labels": ["Document"],
        "properties": {"title": "Only node"},
    }


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},
        {"format_version": "999", "pot_id": "source", "entities": [], "claims": []},
        {"format_version": "2", "pot_id": "source", "entities": [], "claims": "bad"},
    ],
)
def test_invalid_snapshot_is_rejected_before_mutation(bad_payload) -> None:
    backend = _backend_with_snapshot_data()
    before = backend.snapshot.export_data(pot_id="source")
    with pytest.raises(ValueError):
        backend.snapshot.import_data(pot_id="source", payload=bad_payload)
    assert backend.snapshot.export_data(pot_id="source") == before


def test_conflict_is_rejected_before_any_new_rows_are_applied() -> None:
    target = _backend_with_snapshot_data()
    payload = target.snapshot.export_data(pot_id="source")
    payload = deepcopy(payload)
    payload["entities"].append(
        {"key": "new", "labels": ["Entity"], "properties": {}}
    )
    payload["claims"][0]["description"] = "conflicting metadata"
    before = target.snapshot.export_data(pot_id="source")
    with pytest.raises(ValueError, match="conflicts with target"):
        target.snapshot.import_data(pot_id="source", payload=payload)
    assert target.snapshot.export_data(pot_id="source") == before


def test_legacy_snapshot_migrates_keyless_claim_and_is_retry_safe() -> None:
    payload = {
        "format_version": "1",
        "pot_id": "old",
        "labels": {"a": ["Service"]},
        "claims": [{"predicate": "USES", "subject_key": "a", "object_key": "b"}],
    }
    backend = InMemoryGraphBackend()
    backend.snapshot.import_data(pot_id="new", payload=payload)
    backend.snapshot.import_data(pot_id="new", payload=payload)
    exported = backend.snapshot.export_data(pot_id="new")
    assert len(exported["claims"]) == 1
    assert exported["claims"][0]["claim_key"].startswith("claim:new:")
    assert {row["key"] for row in exported["entities"]} == {"a", "b"}


def test_embedded_export_reloads_current_disk_state(tmp_path) -> None:
    stale = EmbeddedGraphBackend(home=tmp_path)
    writer = EmbeddedGraphBackend(home=tmp_path)
    writer.snapshot.import_data(
        pot_id="p",
        payload={
            "format_version": "2",
            "pot_id": "p",
            "entities": [{"key": "only", "labels": ["Entity"], "properties": {}}],
            "claims": [],
        },
    )
    assert stale.snapshot.export_data(pot_id="p")["entities"] == [
        {"key": "only", "labels": ["Entity"], "properties": {}}
    ]


@pytest.mark.parametrize("location", ("entity", "claim"))
@pytest.mark.parametrize("marker", (
    "__potpie_snapshot_properties_v2", "__potpie_snapshot_claim_fields_v2",
))
def test_reserved_native_properties_marker_is_rejected(location: str, marker: str) -> None:
    payload = {
        "format_version": "2",
        "pot_id": "p",
        "entities": [
            {"key": "a", "labels": ["Entity"], "properties": {}},
            {"key": "b", "labels": ["Entity"], "properties": {}},
        ],
        "claims": [
            {
                "claim_key": "claim:p:one",
                "predicate": "USES",
                "subject_key": "a",
                "object_key": "b",
                "properties": {},
            }
        ],
    }
    if location == "entity":
        payload["entities"][0]["properties"][marker] = "user value"
    else:
        payload["claims"][0]["properties"][marker] = "user value"

    with pytest.raises(ValueError, match="reserved key"):
        InMemoryGraphBackend().snapshot.import_data(pot_id="p", payload=payload)
