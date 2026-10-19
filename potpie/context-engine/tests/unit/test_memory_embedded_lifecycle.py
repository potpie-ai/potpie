"""Administrative entry points share graph mutation revisions and receipts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    _Mutation,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "lifecycle-pot"


@pytest.fixture(params=["memory", "embedded"])
def backend(request, tmp_path):
    return (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )


def _plan():
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:a", ("Service",)),
            EntityUpsert("service:b", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                "RELATES_TO", "service:a", "service:b", {"claim_key": "claim:a"}
            ),
            EdgeUpsert(
                "RELATES_TO", "service:b", "service:a", {"claim_key": "claim:b"}
            ),
        ],
    )


def _seed_legacy_entity(backend):
    def seed(inner):
        inner.store.set_entity_label(
            pot_id=POT, entity_key="service:a", labels=("Decision",)
        )
        inner.store.set_entity_properties(
            pot_id=POT,
            entity_key="service:a",
            properties={
                "title": "Legacy service",
                "record_id": "entity:a",
                "uuid": "entity-uuid:a",
            },
        )

    if isinstance(backend, EmbeddedGraphBackend):
        backend._transact(seed)
    else:
        seed(backend)


@pytest.mark.parametrize("journal", [False, True])
def test_direct_invalidation_targets_claim_keys_only(backend, journal):
    if journal:
        backend.journal.activate(pot_id=POT)
    backend.mutation.apply(_plan(), expected_pot_id=POT)
    assert backend.mutation.invalidate(pot_id=POT, claim_keys=["service:a"]) == 0
    assert all(
        row.invalid_at is None
        for row in backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    )
    assert backend.mutation.invalidate(pot_id=POT, claim_keys=["claim:a"]) == 1
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, include_invalidated=True)
    )
    assert {row.claim_key for row in rows if row.invalid_at is not None} == {"claim:a"}
    before = backend.mutation.current_version(POT)
    assert backend.mutation.invalidate(pot_id=POT, claim_keys=[]) == 0
    assert backend.mutation.current_version(POT) == before


@pytest.mark.parametrize("journal", [False, True])
def test_repair_changes_advance_once_and_no_work_does_not_advance(backend, journal):
    _seed_legacy_entity(backend)
    if journal:
        backend.journal.activate(pot_id=POT)
    before = backend.mutation.current_version(POT)
    result = backend.analytics.repair(
        POT, targets=["entity_summaries", "entity_labels"]
    )
    assert result.repaired == {"entity_summaries": 1, "entity_labels": 1}
    assert backend.mutation.current_version(POT) == before + 1
    props = backend.claim_query.entity_properties(pot_id=POT, entity_key="service:a")
    assert props["summary"] == "Legacy service"
    assert props["record_id"] == "entity:a"
    assert props["uuid"] == "entity-uuid:a"
    repeated = backend.analytics.repair(
        POT, targets=["entity_summaries", "entity_labels"]
    )
    assert repeated.repaired == {"entity_summaries": 0, "entity_labels": 0}
    assert backend.mutation.current_version(POT) == before + 1


def test_repair_apply_failure_rolls_back_selected_effects(backend, monkeypatch):
    _seed_legacy_entity(backend)
    original = _Mutation._apply_uncached
    before = deepcopy(backend.snapshot.export_data(pot_id=POT))

    def fail_after_apply(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("repair apply failed")

    monkeypatch.setattr(_Mutation, "_apply_uncached", fail_after_apply)
    with pytest.raises(RuntimeError, match="repair apply failed"):
        backend.analytics.repair(POT)
    assert backend.snapshot.export_data(pot_id=POT) == before
    assert backend.mutation.current_version(POT) == 0


def test_failed_embedded_repair_persistence_preserves_graph_and_prior_receipt(
    tmp_path, monkeypatch
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    _seed_legacy_entity(backend)
    plan = MutationBatch(entity_upserts=[EntityUpsert("service:b", ("Service",))])
    receipt = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="before-repair"),
    )
    before = (tmp_path / "graph.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("repair persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail)
        with pytest.raises(OSError, match="repair persistence failed"):
            backend.analytics.repair(POT)
    assert (tmp_path / "graph.json").read_bytes() == before
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="before-repair"
        ).result
        == receipt
    )


def _snapshot():
    return {
        "format_version": "2",
        "pot_id": POT,
        "entities": [
            {
                "key": "service:a",
                "labels": ["Service"],
                "properties": {"record_id": "entity:a", "uuid": "uuid:a"},
            },
            {
                "key": "service:b",
                "labels": ["Service"],
                "properties": {"record_id": "entity:b", "uuid": "uuid:b"},
            },
        ],
        "claims": [
            {
                "predicate": "RELATED_TO",
                "subject_key": "service:a",
                "object_key": "service:b",
                "claim_key": "claim:imported",
                "record_id": "claim-record:imported",
                "properties": {"uuid": "uuid:claim"},
            }
        ],
    }


def test_reset_always_advances_and_retains_original_execution_receipt(backend):
    plan = _plan()
    original = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="before-reset"),
    )
    assert backend.mutation.reset_pot(POT)["removed_claims"] == 2
    assert backend.mutation.current_version(POT) == 2
    assert backend.mutation.reset_pot(POT)["removed_claims"] == 0
    assert backend.mutation.current_version(POT) == 3
    replay = backend.mutation.compare_and_apply(
        plan,
        expected_pot_id=POT,
        expected_version=0,
        provenance_context=ProvenanceContext(mutation_id="before-reset"),
    )
    assert replay == original
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []
    assert backend.mutation.current_version(POT) == 3


def test_empty_reset_advances_once(backend):
    assert backend.mutation.current_version(POT) == 0
    assert backend.mutation.reset_pot(POT)["removed_claims"] == 0
    assert backend.mutation.current_version(POT) == 1


@pytest.mark.parametrize("operation", ["reset", "invalidate"])
def test_embedded_lifecycle_persistence_failure_preserves_receipts_and_revision(
    tmp_path, monkeypatch, operation
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    plan = _plan()
    receipt = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="before-failure"),
    )
    before = (tmp_path / "graph.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("lifecycle persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail)
        with pytest.raises(OSError, match="lifecycle persistence failed"):
            if operation == "reset":
                backend.mutation.reset_pot(POT)
            else:
                backend.mutation.invalidate(pot_id=POT, claim_keys=["claim:a"])
    assert (tmp_path / "graph.json").read_bytes() == before
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert restarted.mutation.current_version(POT) == 1
    assert len(restarted.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 2
    assert (
        restarted.mutation.lookup_execution(
            plan, expected_pot_id=POT, mutation_id="before-failure"
        ).result
        == receipt
    )


def test_import_preserves_incarnation_and_entity_identity(backend):
    backend.snapshot.import_data(
        pot_id=POT, payload=_snapshot(), mutation_id="import:identity"
    )
    props = backend.claim_query.entity_properties(pot_id=POT, entity_key="service:a")
    assert props["record_id"] == "entity:a"
    assert props["uuid"] == "uuid:a"
    row = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))[0]
    assert row.record_id == "claim-record:imported"
    assert row.properties["uuid"] == "uuid:claim"
    assert backend.mutation.current_version(POT) == 1
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert (
            restarted.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))[0].record_id
            == row.record_id
        )


def test_default_import_identity_replays_after_reset_without_restoring_data(backend):
    payload = _snapshot()
    backend.snapshot.import_data(pot_id=POT, payload=payload)
    assert backend.mutation.current_version(POT) == 1
    backend.mutation.reset_pot(POT)
    backend.snapshot.import_data(pot_id=POT, payload=payload)
    assert backend.mutation.current_version(POT) == 2
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []


def test_explicit_import_identity_rejects_changed_payload(backend):
    from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
        MutationExecutionReuseError,
    )

    payload = _snapshot()
    backend.snapshot.import_data(
        pot_id=POT, payload=payload, mutation_id="import:bound"
    )
    payload["claims"][0]["fact"] = "changed content"
    with pytest.raises(MutationExecutionReuseError):
        backend.snapshot.import_data(
            pot_id=POT, payload=payload, mutation_id="import:bound"
        )
    assert backend.mutation.current_version(POT) == 1


def test_failed_import_persistence_does_not_publish_identity(tmp_path, monkeypatch):
    backend = EmbeddedGraphBackend(home=tmp_path)
    payload = _snapshot()

    def fail(*args, **kwargs):
        raise OSError("import persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail)
        with pytest.raises(OSError, match="import persistence failed"):
            backend.snapshot.import_data(
                pot_id=POT, payload=payload, mutation_id="import:retry"
            )
    assert backend.mutation.current_version(POT) == 0
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []
    backend.snapshot.import_data(
        pot_id=POT, payload=payload, mutation_id="import:retry"
    )
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))[0].record_id
        == "claim-record:imported"
    )


def test_journal_reset_and_import_remain_explicitly_unsupported(backend):
    from potpie_context_engine.core.graph_journal import JournalError

    backend.journal.activate(pot_id=POT)
    with pytest.raises(JournalError, match="reset is unsupported"):
        backend.mutation.reset_pot(POT)
    with pytest.raises(JournalError, match="snapshot import is unsupported"):
        backend.snapshot.import_data(pot_id=POT, payload=_snapshot())
    assert backend.mutation.current_version(POT) == 0


def test_empty_snapshot_gets_receipt_without_advancing_revision(backend):
    payload = {"format_version": "2", "pot_id": POT, "entities": [], "claims": []}
    first = backend.snapshot.import_data(pot_id=POT, payload=payload)
    second = backend.snapshot.import_data(pot_id=POT, payload=payload)
    assert first.metadata["execution"]["version_before"] == 0
    assert first.metadata["execution"]["version_after"] == 0
    assert first.metadata == second.metadata
    assert backend.mutation.current_version(POT) == 0


def test_explicit_new_import_identity_can_restore_after_reset(backend):
    payload = _snapshot()
    backend.snapshot.import_data(pot_id=POT, payload=payload)
    backend.mutation.reset_pot(POT)
    result = backend.snapshot.import_data(
        pot_id=POT, payload=payload, mutation_id="import:restore"
    )
    assert result.metadata["imported_claims"] == 1
    assert backend.mutation.current_version(POT) == 3
    assert (
        backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))[0].record_id
        == "claim-record:imported"
    )


def test_duplicate_import_incarnation_is_rejected_before_any_effect(backend):
    payload = _snapshot()
    payload["claims"][0]["record_id"] = payload["entities"][0]["properties"][
        "record_id"
    ]
    with pytest.raises(ValueError, match="duplicate snapshot incarnation"):
        backend.snapshot.import_data(pot_id=POT, payload=payload)
    assert backend.mutation.current_version(POT) == 0
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []


def test_import_incarnation_cannot_change_its_logical_record(backend):
    backend.snapshot.import_data(pot_id=POT, payload=_snapshot())
    other = _snapshot()
    other["entities"][0]["key"] = "service:c"
    other["claims"][0]["subject_key"] = "service:c"
    other["claims"][0]["claim_key"] = "claim:another"
    other["claims"][0]["record_id"] = "claim-record:another"
    other["claims"][0]["properties"]["uuid"] = "uuid:another"
    with pytest.raises(ValueError, match="incarnation"):
        backend.snapshot.import_data(
            pot_id=POT, payload=other, mutation_id="import:collision"
        )
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.claim_query.entity_labels(pot_id=POT, entity_keys=["service:c"]) == {}
    )
