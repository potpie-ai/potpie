"""Explicit raw repairs share atomic receipts without erasing legacy evidence."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
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
from potpie_context_engine.core.graph_journal import (
    JournalError,
    JournalLimits,
    encode_journal,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.legacy_records import (
    REPAIR_ENVELOPE_BYTES,
    RecordRepair,
    repair_effect_bytes,
)
from potpie_context_engine.core.legacy_repair_policy import plan_legacy_repairs
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "legacy-repair-pot"
OFFSET = "2025-01-01T05:30:00.123456+05:30"
UTC = "2025-01-01T00:00:00.123456+00:00"


def _state(*, claim_id=True, entity_id=True):
    return {
        "format_version": "2",
        "rows": [
            {
                "pot_id": POT,
                "predicate": "RELATED_TO",
                "subject_key": "service:a",
                "object_key": "service:b",
                "claim_key": "claim:old",
                "valid_at": OFFSET,
                "uuid": "historical-claim-uuid",
                "legacy_extra": {"keep": [None, "raw"]},
                "retired": True,
                "properties": {"valid_from": OFFSET},
                **({"record_id": "claim-record"} if claim_id else {}),
            }
        ],
        "labels": [
            [POT, key, ["Entity", "Service"]] for key in ("service:a", "service:b")
        ],
        "entity_properties": [
            [
                POT,
                "service:a",
                {
                    "uuid": "historical-entity-uuid",
                    "valid_from": OFFSET,
                    **({"record_id": "entity-a-record"} if entity_id else {}),
                },
            ],
            [POT, "service:b", {"record_id": "entity-b-record"}],
        ],
    }


def _backend(profile, tmp_path, state):
    if profile == "json":
        (tmp_path / "graph.json").write_text(json.dumps(state))
        return EmbeddedGraphBackend(home=tmp_path)
    return InMemoryGraphBackend(store=load_store(state))


def _store(backend):
    return (
        backend._inner.store
        if isinstance(backend, EmbeddedGraphBackend)
        else backend.store
    )


def _apply(backend, batch, mid="repair"):
    return backend.mutation.apply(
        batch,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id=mid),
    )


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_inventory_is_full_raw_read_and_dry_run_is_unchanged(profile, tmp_path):
    state = _state(claim_id=False, entity_id=False)
    backend = _backend(profile, tmp_path, state)
    before = dump_store(_store(backend))
    disk = (tmp_path / "graph.json").read_bytes() if profile == "json" else None
    inventory = backend.mutation.legacy_inventory(POT)
    assert asyncio.run(backend.mutation.legacy_inventory_async(POT)) == inventory
    assert inventory.max_repair_records == 5000
    assert inventory.max_repair_bytes == 8_000_000
    claim = next(record for record in inventory.records if record.kind == "claim")
    assert claim.fields["valid_at"] == OFFSET
    assert claim.fields["uuid"] == "historical-claim-uuid"
    assert "record_id" not in claim.fields
    assert claim.fields["retired"] is True
    assert claim.fields["legacy_extra"] == {"keep": [None, "raw"]}
    assert claim.properties == {"valid_from": OFFSET}
    assert claim.relationship_type == "RELATES_TO"
    assert claim.subject_pot_id == claim.object_pot_id == POT
    effects, findings = plan_legacy_repairs(inventory)
    assert len(effects) == 2
    assert not any(finding.blocking for finding in findings)
    assert dump_store(_store(backend)) == before
    assert backend.mutation.current_version(POT) == 0
    if disk is not None:
        assert (tmp_path / "graph.json").read_bytes() == disk


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize("journal", [False, True])
def test_exact_normalization_is_revisioned_replayable_and_preserves_aliases(
    profile, journal, tmp_path
):
    backend = _backend(profile, tmp_path, _state())
    if journal:
        backend.journal.activate(
            pot_id=POT,
            rollback_enabled=True,
            limits=JournalLimits(max_records=17, max_bytes=123456),
        )
    inventory = backend.mutation.legacy_inventory(POT)
    if journal:
        assert inventory.journal_active
        assert (inventory.max_repair_records, inventory.max_repair_bytes) == (
            17,
            123456,
        )
    effects, findings = plan_legacy_repairs(inventory)
    assert not any(finding.blocking for finding in findings)
    batch = MutationBatch(record_repairs=effects)
    result = _apply(backend, batch)
    assert result.execution.version_before == 0
    assert result.execution.version_after == 1
    assert result.execution.affected_claim_record_ids == ("claim-record",)
    assert result.administrative_summary == {
        "repaired_records": 2,
        "repaired_entities": 1,
        "repaired_claims": 1,
    }
    assert _apply(backend, batch) == result
    assert backend.mutation.current_version(POT) == 1
    records = backend.mutation.legacy_inventory(POT).records
    claim = next(record for record in records if record.kind == "claim")
    entity = next(record for record in records if record.key == "service:a")
    assert claim.fields["valid_at"] == claim.properties["valid_from"] == UTC
    assert claim.fields["record_id"] == "claim-record"
    assert claim.fields["uuid"] == "historical-claim-uuid"
    assert entity.properties["record_id"] == "entity-a-record"
    assert entity.properties["uuid"] == "historical-entity-uuid"
    assert plan_legacy_repairs(backend.mutation.legacy_inventory(POT))[0] == ()
    if journal:
        receipt = backend.journal.get_receipt(pot_id=POT, commit_id="repair")
        assert receipt.origin == "repair"
        assert not receipt.rollback_supported
        assert receipt.unsupported_reason
    if profile == "json":
        reopened = EmbeddedGraphBackend(home=tmp_path)
        assert _apply(reopened, batch) == result
        assert reopened.mutation.legacy_inventory(POT).records == records


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_explicit_identity_backfill_keeps_valid_distinct_legacy_uuids(
    profile, tmp_path
):
    backend = _backend(profile, tmp_path, _state(claim_id=False, entity_id=False))
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    _apply(backend, MutationBatch(record_repairs=effects))
    records = backend.mutation.legacy_inventory(POT).records
    claim = next(record for record in records if record.kind == "claim")
    entity = next(record for record in records if record.key == "service:a")
    assert claim.fields["record_id"] == claim.fields["uuid"] == "historical-claim-uuid"
    assert (
        entity.properties["record_id"]
        == entity.properties["uuid"]
        == "historical-entity-uuid"
    )
    backend.journal.activate(pot_id=POT)


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_generated_backfill_ids_are_frozen_and_reused_once(profile, tmp_path):
    state = _state(claim_id=False, entity_id=False)
    state["rows"][0].pop("uuid")
    state["entity_properties"][0][2].pop("uuid")
    backend = _backend(profile, tmp_path, state)
    effects, findings = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    assert not any(finding.blocking for finding in findings)
    chosen = {
        effect.before.key: effect.after().value("record_id") for effect in effects
    }
    assert all(isinstance(identity, str) and identity for identity in chosen.values())
    batch = MutationBatch(record_repairs=effects)
    result = _apply(backend, batch)
    assert _apply(backend, batch) == result
    records = backend.mutation.legacy_inventory(POT).records
    for record in records:
        if record.key in chosen:
            assert record.value("record_id") == chosen[record.key]
            assert "uuid" not in record.fields
            assert "uuid" not in record.properties
    assert plan_legacy_repairs(backend.mutation.legacy_inventory(POT))[0] == ()


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize(
    "bad", ["not-a-timestamp", "2025-01-01", "2025-01-01T00:00:00", None]
)
def test_unrelated_write_preserves_raw_values_presence_and_unknown_fields(
    profile, bad, tmp_path
):
    state = _state()
    state["rows"][0]["invalid_at"] = bad
    state["rows"][0]["valid_until"] = "malformed-end"
    state["rows"][0]["properties"]["invalid_at"] = "masked-malformed-property"
    backend = _backend(profile, tmp_path, state)
    original = backend.mutation.legacy_inventory(POT).records
    _apply(
        backend,
        MutationBatch(
            entity_upserts=[EntityUpsert("service:new", ("Entity", "Service"))]
        ),
        "unrelated",
    )
    current = backend.mutation.legacy_inventory(POT).records
    assert all(record in current for record in original)
    serialized = dump_store(_store(backend))["rows"][0]
    assert serialized["invalid_at"] == bad
    assert serialized["valid_until"] == "malformed-end"
    assert serialized["uuid"] == "historical-claim-uuid"
    assert serialized["legacy_extra"] == {"keep": [None, "raw"]}
    assert "stored_source" not in serialized
    if profile == "json":
        raw = json.loads((tmp_path / "graph.json").read_text())["rows"][0]
        assert raw == serialized
        assert (
            EmbeddedGraphBackend(home=tmp_path).mutation.legacy_inventory(POT).records
            == current
        )


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_failed_beforeimage_match_rolls_back_every_repair(profile, tmp_path):
    backend = _backend(profile, tmp_path, _state())
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    changed = replace(
        effects[-1], before=replace(effects[-1].before, key="claim:changed")
    )
    before = dump_store(_store(backend))
    with pytest.raises(ValueError, match="missing, changed, or ambiguous"):
        _apply(backend, MutationBatch(record_repairs=(*effects[:-1], changed)))
    assert dump_store(_store(backend)) == before
    assert backend.mutation.current_version(POT) == 0


def test_failed_disk_publication_keeps_raw_state_and_has_no_receipt(
    tmp_path, monkeypatch
):
    backend = _backend("json", tmp_path, _state())
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    batch = MutationBatch(record_repairs=effects)
    before = (tmp_path / "graph.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected persistence failure")

    monkeypatch.setattr(EmbeddedGraphBackend, "_write_state", fail)
    with pytest.raises(OSError, match="injected persistence failure"):
        _apply(backend, batch)
    assert (tmp_path / "graph.json").read_bytes() == before
    assert backend.mutation.current_version(POT) == 0
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=POT, mutation_id="repair"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize("defect", ["entity-id", "claim-id", "endpoint", "alias"])
def test_journal_activation_never_migrates_or_fabricates_records(
    profile, defect, tmp_path
):
    state = _state(claim_id=defect != "claim-id", entity_id=defect != "entity-id")
    if defect == "endpoint":
        state["labels"].pop()
        state["entity_properties"].pop()
    if defect == "alias":
        state["rows"][0]["uuid"] = "historical-entity-uuid"
    backend = _backend(profile, tmp_path, state)
    before = dump_store(_store(backend))
    with pytest.raises(JournalError):
        backend.journal.activate(pot_id=POT)
    assert dump_store(_store(backend)) == before
    assert backend.journal.journal_state(POT) is None
    assert backend.mutation.current_version(POT) == 0


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize("journal", [False, True])
def test_entity_identity_is_immutable_and_aliases_have_global_owners(
    profile, journal, tmp_path
):
    backend = _backend(profile, tmp_path, _state())
    if journal:
        backend.journal.activate(pot_id=POT)
    before = backend.mutation.legacy_inventory(POT)
    for key, properties in [
        ("service:a", {"record_id": "replacement"}),
        ("service:a", {"uuid": "replacement"}),
        ("service:new", {"record_id": "historical-claim-uuid"}),
    ]:
        with pytest.raises(ValueError, match="immutable|shared by multiple"):
            _apply(
                backend,
                MutationBatch(
                    entity_upserts=[
                        EntityUpsert(key, ("Entity", "Service"), properties)
                    ]
                ),
                f"reject-{key}-{properties}",
            )
        assert backend.mutation.legacy_inventory(POT) == before
        assert backend.mutation.current_version(POT) == 0
    _apply(
        backend,
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    "service:a",
                    ("Entity", "Service"),
                    {
                        "record_id": "entity-a-record",
                        "uuid": "historical-entity-uuid",
                        "name": "renamed",
                    },
                )
            ]
        ),
        "same-identity",
    )
    entity = next(
        record
        for record in backend.mutation.legacy_inventory(POT).records
        if record.key == "service:a"
    )
    assert entity.properties["name"] == "renamed"
    assert entity.properties["uuid"] == "historical-entity-uuid"


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_alias_collision_blocks_the_entire_repair_chunk(profile, tmp_path):
    state = _state()
    state["entity_properties"][1][2]["uuid"] = "historical-claim-uuid"
    backend = _backend(profile, tmp_path, state)
    effects, findings = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    assert any(finding.blocking for finding in findings)
    before = dump_store(_store(backend))
    with pytest.raises(ValueError, match="shared by multiple"):
        _apply(backend, MutationBatch(record_repairs=effects))
    assert dump_store(_store(backend)) == before
    assert backend.mutation.current_version(POT) == 0


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_existing_identityless_entity_requires_explicit_repair(profile, tmp_path):
    backend = _backend(profile, tmp_path, _state(entity_id=False))
    before = backend.mutation.legacy_inventory(POT)
    with pytest.raises(ValueError, match="migration_required"):
        _apply(
            backend,
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:a",
                        ("Entity", "Service"),
                        {"record_id": "invented"},
                    )
                ]
            ),
        )
    assert backend.mutation.legacy_inventory(POT) == before
    assert backend.mutation.current_version(POT) == 0


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_uuid_only_claim_refresh_requires_explicit_repair(profile, tmp_path):
    state = _state(claim_id=False)
    state["rows"][0]["retired"] = False
    scope = {
        "subgraph": "code_topology",
        "source_system": "fixture",
        "source_ref": "fixture:legacy",
    }
    state["rows"][0].update(scope)
    backend = _backend(profile, tmp_path, state)
    before = backend.mutation.legacy_inventory(POT)
    with pytest.raises(ValueError, match="migration_required"):
        _apply(
            backend,
            MutationBatch(
                edge_upserts=[
                    EdgeUpsert(
                        "RELATED_TO",
                        "service:a",
                        "service:b",
                        {
                            "claim_key": "claim:old",
                            "valid_from": OFFSET,
                            **scope,
                        },
                    )
                ]
            ),
        )
    assert backend.mutation.legacy_inventory(POT) == before
    assert backend.mutation.current_version(POT) == 0


def test_native_memory_datetime_evidence_is_normalized_only_by_explicit_repair():
    state = _state()
    state["rows"][0]["valid_at"] = datetime.fromisoformat(OFFSET)
    state["rows"][0]["properties"]["valid_from"] = datetime.fromisoformat(OFFSET)
    state["entity_properties"][0][2]["valid_from"] = datetime.fromisoformat(OFFSET)
    backend = InMemoryGraphBackend(store=load_store(state))
    inventory = backend.mutation.legacy_inventory(POT)
    claim = next(record for record in inventory.records if record.kind == "claim")
    assert claim.field_types == {"valid_at": "datetime"}
    assert claim.property_types == {"valid_from": "datetime"}
    effects, _ = plan_legacy_repairs(inventory)
    _apply(backend, MutationBatch(record_repairs=effects))
    repaired = backend.mutation.legacy_inventory(POT)
    assert all(
        not record.property_types and not record.field_types
        for record in repaired.records
    )
    assert plan_legacy_repairs(repaired)[0] == ()


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_repair_identity_reuse_with_changed_body_is_rejected(profile, tmp_path):
    backend = _backend(profile, tmp_path, _state(claim_id=False, entity_id=False))
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    batch = MutationBatch(record_repairs=effects)
    result = _apply(backend, batch)
    changed = RecordRepair(
        effects[-1].before,
        {**effects[-1].field_updates, "record_id": "different-frozen-id"},
        effects[-1].property_updates,
    )
    with pytest.raises(ValueError, match="different|reused|fingerprint"):
        _apply(backend, MutationBatch(record_repairs=(*effects[:-1], changed)))
    assert _apply(backend, batch) == result
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_repair_preserves_old_journal_and_execution_receipts(profile, tmp_path):
    backend = _backend(profile, tmp_path, _state())
    backend.journal.activate(pot_id=POT)
    original_batch = MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "service:new",
                ("Entity", "Service"),
                {"name": "original"},
            )
        ]
    )
    original_result = _apply(backend, original_batch, "old")
    receipt_bytes = json.dumps(
        encode_journal(
            backend.journal.get_receipt(
                pot_id=POT,
                commit_id="old",
            )
        ),
        sort_keys=True,
    )
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    _apply(backend, MutationBatch(record_repairs=effects))
    assert _apply(backend, original_batch, "old") == original_result
    assert (
        json.dumps(
            encode_journal(
                backend.journal.get_receipt(
                    pot_id=POT,
                    commit_id="old",
                )
            ),
            sort_keys=True,
        )
        == receipt_bytes
    )
    assert backend.mutation.current_version(POT) == 2


@pytest.mark.parametrize("profile", ["memory", "json"])
def test_typed_undo_of_unrelated_change_keeps_raw_claim_evidence(profile, tmp_path):
    state = _state()
    state["rows"][0]["invalid_at"] = "malformed"
    backend = _backend(profile, tmp_path, state)
    backend.journal.activate(pot_id=POT, rollback_enabled=True)
    original = next(
        record
        for record in backend.mutation.legacy_inventory(POT).records
        if record.kind == "claim"
    )
    _apply(
        backend,
        MutationBatch(
            entity_upserts=[
                EntityUpsert(
                    "service:a",
                    ("Entity", "Service"),
                    {"name": "changed"},
                )
            ]
        ),
        "entity-update",
    )
    plan = backend.journal.plan_restore(
        pot_id=POT,
        target_commit_id="entity-update",
        expected_head="entity-update",
    )
    backend.journal.apply_restore(plan, mutation_id="undo", actor="fixture")
    current = next(
        record
        for record in backend.mutation.legacy_inventory(POT).records
        if record.kind == "claim"
    )
    assert current == original


@pytest.mark.parametrize("profile", ["memory", "json"])
@pytest.mark.parametrize("bound", ["records", "bytes"])
def test_storage_only_repair_cannot_evade_active_journal_limits(
    profile, bound, tmp_path
):
    backend = _backend(profile, tmp_path, _state())
    effects, _ = plan_legacy_repairs(backend.mutation.legacy_inventory(POT))
    limits = (
        JournalLimits(max_records=1)
        if bound == "records"
        else JournalLimits(
            max_bytes=REPAIR_ENVELOPE_BYTES
            + sum(repair_effect_bytes(effect) for effect in effects)
            - 1
        )
    )
    backend.journal.activate(pot_id=POT, limits=limits)
    before = backend.mutation.legacy_inventory(POT)
    with pytest.raises(ValueError, match="transaction limits"):
        _apply(backend, MutationBatch(record_repairs=effects))
    assert backend.mutation.legacy_inventory(POT) == before
    assert backend.mutation.current_version(POT) == 0
    assert backend.journal.journal_state(POT).sequence == 0
