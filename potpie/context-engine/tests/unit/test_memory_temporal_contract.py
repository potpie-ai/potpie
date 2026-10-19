"""Memory/JSON reads compare instants; new imports persist strict UTC times."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    CompletedMutationExecution,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    ExecutionMetadata,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)

pytestmark = pytest.mark.unit
POT = "temporal-pot"
POINT = datetime(2026, 1, 1, tzinfo=timezone.utc)
MICROSECOND = timedelta(microseconds=1)


@pytest.fixture(params=["memory", "json"])
def backend(request, tmp_path):
    return (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )


def _transact(backend, operation):
    if isinstance(backend, EmbeddedGraphBackend):
        return backend._transact(operation)
    return operation(backend)


def _row(key, **fields):
    return ClaimRow(
        pot_id=POT,
        predicate="RELATED_TO",
        subject_key=f"service:{key}",
        object_key="service:target",
        claim_key=f"claim:{key}",
        record_id=f"record:{key}",
        **fields,
    )


def _seed(backend, rows, *, journal=False):
    def seed(inner):
        inner.store.add_many(rows)
        for row in rows:
            for key in (row.subject_key, row.object_key):
                inner.store.set_entity_label(
                    pot_id=POT, entity_key=key, labels=("Entity", "Service")
                )
                inner.store.set_entity_properties(
                    pot_id=POT,
                    entity_key=key,
                    properties={"record_id": f"entity-record:{key}"},
                )

    _transact(backend, seed)
    if journal:
        backend.journal.activate(pot_id=POT)


def _claims(backend, **filters):
    return backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT, **filters))


@pytest.mark.parametrize("journal", [False, True])
@pytest.mark.parametrize("endpoint", ["invalid_at", "valid_until"])
def test_legacy_offsets_and_microseconds_obey_half_open_validity(
    backend, journal, endpoint
):
    start = datetime.fromisoformat("2026-01-01T05:30:00.000001+05:30")
    end = datetime.fromisoformat("2025-12-31T19:00:00.000003-05:00")
    _seed(
        backend,
        [
            _row(
                "offset",
                valid_at=start,
                observed_at=POINT + timedelta(days=300),
                **{endpoint: end},
            )
        ],
        journal=journal,
    )
    assert _claims(backend, as_of=POINT) == []
    assert len(_claims(backend, as_of=POINT + MICROSECOND)) == 1
    assert len(_claims(backend, as_of=POINT + 2 * MICROSECOND)) == 1
    assert _claims(backend, as_of=POINT + 3 * MICROSECOND) == []
    equivalent = datetime.fromisoformat("2025-12-31T20:00:00.000002-04:00")
    assert _claims(backend, as_of=equivalent) == _claims(
        backend, as_of=POINT + 2 * MICROSECOND
    )
    assert (
        len(
            _claims(
                backend,
                include_invalidated=True,
                valid_at_after=start,
                valid_at_before=start,
            )
        )
        == 1
    )
    assert _claims(backend, include_invalidated=True, valid_at_before=POINT) == []
    assert (
        _claims(
            backend, include_invalidated=True, valid_at_after=POINT + 2 * MICROSECOND
        )
        == []
    )


@pytest.mark.parametrize("journal", [False, True])
def test_query_captures_one_clock_instant_for_entire_scan(
    backend, journal, monkeypatch
):
    _seed(
        backend,
        [
            _row(key, valid_at=POINT, valid_until=POINT + MICROSECOND)
            for key in ("first", "second")
        ],
        journal=journal,
    )

    class AdvancingClock(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            point = POINT + cls.calls * MICROSECOND
            cls.calls += 1
            return point

    monkeypatch.setattr(
        "potpie_context_engine.adapters.outbound.graph.in_memory_reader.datetime",
        AdvancingClock,
    )
    assert {row.claim_key for row in _claims(backend)} == {
        "claim:first",
        "claim:second",
    }
    assert AdvancingClock.calls == 1
    assert len(_claims(backend, as_of=POINT)) == 2
    assert AdvancingClock.calls == 1


@pytest.mark.parametrize("journal", [False, True])
def test_future_and_observation_filters_run_before_ranking_and_limit(backend, journal):
    common = {
        "fact": "needle target",
        "environment": "Prod",
        "truth": "authoritative_fact",
    }
    _seed(
        backend,
        [
            _row("future", valid_at=POINT + MICROSECOND, **common),
            _row(
                "ended", valid_at=POINT - timedelta(days=1), invalid_at=POINT, **common
            ),
            _row("other-env", valid_at=POINT, **{**common, "environment": "Stage"}),
            _row(
                "other-truth",
                valid_at=POINT,
                **{**common, "truth": "source_observation"},
            ),
            _row(
                "current",
                valid_at=POINT,
                observed_at=POINT + timedelta(days=10),
                **common,
            ),
        ],
        journal=journal,
    )
    result = _claims(
        backend,
        as_of=POINT,
        environment_in=("prod",),
        truth_in=("AUTHORITATIVE_FACT",),
        exact_text_in=("NEEDLE",),
        exact_text_pattern=".*needle.*",
        fact_query="needle target",
        limit=1,
    )
    assert [row.claim_key for row in result] == ["claim:current"]
    assert result[0].observed_at > POINT
    assert result[0].properties["semantic_similarity"] == 1.0
    assert len(_claims(backend, include_invalidated=True)) == 5


@pytest.mark.parametrize("journal", [False, True])
def test_legacy_naive_read_times_remain_readable_as_utc(backend, journal):
    _seed(
        backend, [_row("legacy", valid_at=POINT.replace(tzinfo=None))], journal=journal
    )
    assert len(_claims(backend, as_of=POINT)) == 1
    assert _claims(backend, as_of=POINT - MICROSECOND) == []


def _snapshot(*, start="2026-01-01T05:30:00.000001+05:30", **fields):
    return {
        "format_version": "2",
        "pot_id": POT,
        "entities": [
            {
                "key": key,
                "labels": ["Service"],
                "properties": {"record_id": f"entity:{key}"},
            }
            for key in ("service:a", "service:b")
        ],
        "claims": [
            {
                "predicate": "RELATED_TO",
                "subject_key": "service:a",
                "object_key": "service:b",
                "claim_key": "claim:imported",
                "record_id": "record:imported",
                "valid_at": start,
                "invalid_at": "2025-12-31T19:00:00.000003-05:00",
                "observed_at": "2026-02-01T03:30:00+03:30",
                "properties": {},
                **fields,
            }
        ],
    }


@pytest.mark.parametrize("journal", [False, True])
def test_raw_write_persists_utc_instants_and_keeps_original_retry_identity(
    backend, journal
):
    if journal:
        backend.journal.activate(pot_id=POT)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert(key, ("Entity", "Service"))
            for key in ("service:a", "service:b")
        ],
        edge_upserts=[
            EdgeUpsert(
                "RELATED_TO",
                "service:a",
                "service:b",
                {
                    "claim_key": "claim:raw",
                    "valid_at": "2026-01-01T05:30:00.000001+05:30",
                    "valid_from": "2026-01-01T00:00:00.000001Z",
                    "valid_until": "2025-12-31T19:00:00.000003-05:00",
                    "observed_at": "2026-02-01T03:30:00+03:30",
                },
            )
        ],
    )
    before = deepcopy(plan)
    context = ProvenanceContext(mutation_id="raw-utc")
    receipt = backend.mutation.apply(
        plan, expected_pot_id=POT, provenance_context=context
    )
    assert plan == before
    row = _claims(backend, include_invalidated=True)[0]
    assert row.valid_at.isoformat() == "2026-01-01T00:00:00.000001+00:00"
    assert row.valid_until.isoformat() == "2026-01-01T00:00:00.000003+00:00"
    assert row.observed_at.isoformat() == "2026-02-01T00:00:00+00:00"
    assert receipt.execution.batch_fingerprint == batch_fingerprint(plan)
    assert (
        backend.mutation.apply(plan, expected_pot_id=POT, provenance_context=context)
        == receipt
    )
    assert backend.mutation.current_version(POT) == 1
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert _claims(restarted, include_invalidated=True) == [row]
        assert (
            restarted.mutation.apply(
                plan, expected_pot_id=POT, provenance_context=context
            )
            == receipt
        )


def test_snapshot_import_normalizes_physical_times_without_changing_identity(backend):
    payload = _snapshot(properties={"valid_from": "2026-01-01T00:00:00.000001Z"})
    before = deepcopy(payload)
    batch, _ = prepare_snapshot_import(pot_id=POT, payload=payload, expected_version=0)
    manifest = backend.snapshot.import_data(pot_id=POT, payload=payload)
    assert payload == before
    assert manifest.metadata["mutation_id"] == f"snapshot:{batch_fingerprint(batch)}"
    assert manifest.metadata["execution"]["batch_fingerprint"] == batch_fingerprint(
        batch
    )
    row = _claims(backend, include_invalidated=True)[0]
    assert row.valid_at.isoformat() == "2026-01-01T00:00:00.000001+00:00"
    assert row.invalid_at.isoformat() == "2026-01-01T00:00:00.000003+00:00"
    assert row.observed_at.isoformat() == "2026-02-01T00:00:00+00:00"
    assert "valid_from" not in row.properties
    assert (
        backend.snapshot.import_data(pot_id=POT, payload=payload).metadata
        == manifest.metadata
    )
    assert backend.mutation.current_version(POT) == 1
    if isinstance(backend, EmbeddedGraphBackend):
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert _claims(restarted, include_invalidated=True) == [row]
        assert (
            restarted.snapshot.import_data(pot_id=POT, payload=payload).metadata
            == manifest.metadata
        )


def test_snapshot_merge_accepts_equivalent_offset_spelling_under_new_identity(backend):
    first = _snapshot()
    backend.snapshot.import_data(pot_id=POT, payload=first, mutation_id="offset-form")
    second = _snapshot(
        start="2026-01-01T00:00:00.000001Z",
        invalid_at="2026-01-01T00:00:00.000003Z",
        observed_at="2026-02-01T00:00:00Z",
    )
    manifest = backend.snapshot.import_data(
        pot_id=POT, payload=second, mutation_id="utc-form"
    )
    assert manifest.metadata["imported_claims"] == 0
    assert manifest.metadata["imported_entities"] == 0
    assert len(_claims(backend, include_invalidated=True)) == 1
    assert backend.mutation.current_version(POT) == 2


def test_offset_import_persistence_failure_preserves_original_bytes_and_receipts(
    tmp_path, monkeypatch
):
    backend = EmbeddedGraphBackend(home=tmp_path)
    prior = _snapshot()
    original = backend.snapshot.import_data(
        pot_id=POT, payload=prior, mutation_id="prior"
    )
    payload = _snapshot(claim_key="claim:retry", record_id="record:retry")
    before = deepcopy(payload)
    batch, _ = prepare_snapshot_import(
        pot_id=POT, payload=payload, expected_version=1, mutation_id="retry"
    )
    disk = (backend.home / "graph.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected temporal import persistence failure")

    with monkeypatch.context() as patch:
        patch.setattr(EmbeddedGraphBackend, "_write_state", fail)
        with pytest.raises(OSError, match="temporal import persistence failure"):
            backend.snapshot.import_data(
                pot_id=POT, payload=payload, mutation_id="retry"
            )
    assert payload == before
    assert (backend.home / "graph.json").read_bytes() == disk
    restarted = EmbeddedGraphBackend(home=tmp_path)
    assert restarted.mutation.current_version(POT) == 1
    assert (
        restarted.mutation.lookup_execution(
            batch, expected_pot_id=POT, mutation_id="retry"
        ).state
        == "absent"
    )
    assert (
        restarted.snapshot.import_data(
            pot_id=POT, payload=prior, mutation_id="prior"
        ).metadata
        == original.metadata
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "naive-start",
        "naive-observation",
        "contradictory-alias",
        "empty-interval",
        "negative-interval",
    ],
)
def test_new_invalid_snapshot_import_rolls_back_without_receipt(backend, invalid):
    existing = _snapshot()
    manifest = backend.snapshot.import_data(
        pot_id=POT, payload=existing, mutation_id="prior"
    )
    payload = _snapshot()
    claim = payload["claims"][0]
    claim["record_id"] = "record:rejected"
    claim["claim_key"] = "claim:rejected"
    if invalid == "naive-start":
        claim["valid_at"] = "2026-01-01T00:00:00.000001"
    elif invalid == "naive-observation":
        claim["observed_at"] = "2026-02-01T00:00:00"
    elif invalid == "contradictory-alias":
        claim["properties"]["valid_from"] = "2026-01-01T00:00:00.000002Z"
    elif invalid == "empty-interval":
        claim["valid_until"] = "2026-01-01T00:00:00.000001Z"
    else:
        claim["invalid_at"] = "2026-01-01T00:00:00Z"
    batch, _ = prepare_snapshot_import(
        pot_id=POT, payload=payload, expected_version=1, mutation_id="rejected"
    )
    before = deepcopy(backend.snapshot.export_data(pot_id=POT))
    disk = (
        (backend.home / "graph.json").read_bytes()
        if isinstance(backend, EmbeddedGraphBackend)
        else None
    )
    with pytest.raises(ValueError):
        backend.snapshot.import_data(
            pot_id=POT, payload=payload, mutation_id="rejected"
        )
    assert backend.snapshot.export_data(pot_id=POT) == before
    assert backend.mutation.current_version(POT) == 1
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=POT, mutation_id="rejected"
        ).state
        == "absent"
    )
    assert (
        backend.snapshot.import_data(
            pot_id=POT, payload=existing, mutation_id="prior"
        ).metadata
        == manifest.metadata
    )
    if disk is not None:
        assert (backend.home / "graph.json").read_bytes() == disk


def test_completed_legacy_naive_snapshot_replays_original_receipt_before_new_validation(
    backend,
):
    payload = _snapshot(start="2026-01-01T00:00:00.000001")
    batch, context = prepare_snapshot_import(
        pot_id=POT, payload=payload, expected_version=0, mutation_id="legacy"
    )
    fingerprint = batch_fingerprint(batch)
    execution = ExecutionMetadata(
        pot_id=POT,
        applied_at=POINT,
        version_before=0,
        version_after=1,
        batch_fingerprint=fingerprint,
        request_fingerprint=context.prepared.request_fingerprint,
        affected_claim_record_ids=("record:imported",),
    )
    result = MutationResult(
        ok=True,
        mutation_id="legacy",
        mutation_summary=MutationSummary(),
        administrative_summary={"imported_entities": 2, "imported_claims": 1},
        execution=execution,
    )

    def seed_receipt(inner):
        inner.execution_registry.replace_completed(
            (CompletedMutationExecution(POT, "legacy", fingerprint, result),)
        )
        inner.revisions[POT] = 2

    _transact(backend, seed_receipt)
    disk = (
        (backend.home / "graph.json").read_bytes()
        if isinstance(backend, EmbeddedGraphBackend)
        else None
    )
    manifest = backend.snapshot.import_data(
        pot_id=POT, payload=payload, mutation_id="legacy"
    )
    assert manifest.metadata["execution"] == execution.to_dict()
    assert _claims(backend, include_invalidated=True) == []
    assert backend.mutation.current_version(POT) == 2
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=POT, mutation_id="legacy"
        ).result
        == result
    )
    if disk is not None:
        assert (backend.home / "graph.json").read_bytes() == disk
        restarted = EmbeddedGraphBackend(home=backend.home)
        assert (
            restarted.snapshot.import_data(
                pot_id=POT, payload=payload, mutation_id="legacy"
            ).metadata
            == manifest.metadata
        )
