"""Live end-to-end tests for the multi-event batching state machine against
**real Postgres** — see e2e_surface.md.

The pipeline suite only exercises a single (N=1) submit→process path. This
module covers the production reliability path that was previously untested
end-to-end: debounce/coalescing, idempotent re-submission, windowed flush
eligibility, and stale-batch reaping (crash/OOM recovery). Deterministic — no
LLM. Skips when Postgres (or Neo4j, via the shared container) is unavailable.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy import text

from potpie_context_engine.application.use_cases.flush_windowed_batches import (
    flush_ready_windowed_pots,
)
from potpie_context_engine.application.use_cases.reap_stale_batches import (
    reap_stale_batches,
)
from potpie_context_engine.domain.ingestion_event_models import (
    IngestionSubmissionRequest,
)
from potpie_context_engine.domain.ingestion_kinds import (
    INGESTION_KIND_AGENT_RECONCILIATION,
)
from potpie_context_engine.domain.reconciliation_batch import (
    BATCH_STATUS_CLAIMED,
    BATCH_STATUS_FAILED,
    BATCH_STATUS_PENDING,
)

pytestmark = pytest.mark.integration


def _req(pot_id: str, source_id: str) -> IngestionSubmissionRequest:
    return IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="http",
        source_system="test",
        event_type="deployment",
        action="succeeded",
        payload={"service": "auth-svc", "source_id": source_id},
        source_id=source_id,
        repo_name="acme/platform",
    )


def _submit(container, sm, request):
    """Submit one event (windowed/async) in its own committed session."""
    session = sm()
    try:
        receipt = container.ingestion_submission(session).submit(request, sync=False)
        session.commit()
        return receipt
    finally:
        session.close()


class _SpyJobs:
    """Captures enqueue_batch calls so we can assert the flush dispatched."""

    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue_batch(self, batch_id: str) -> None:
        self.enqueued.append(batch_id)


# ---------------------------------------------------------------------------
# Coalescing / debounce
# ---------------------------------------------------------------------------


class TestCoalescing:
    def test_three_events_coalesce_into_one_open_batch(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        receipts = [
            _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, f"evt-{i}"))
            for i in range(3)
        ]
        # All three land in the same open batch (the receipt's job_id).
        batch_ids = {r.job_id for r in receipts}
        assert len(batch_ids) == 1

        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            open_batch = batch_repo.get_open_batch_id_for_pot(pot_id)
            assert open_batch == receipts[0].job_id
            refs = batch_repo.list_events_for_batch(open_batch)
            assert {r.event_id for r in refs} == {r.event_id for r in receipts}
            assert len(refs) == 3
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Idempotent re-submission (A4 — webhook redelivery)
# ---------------------------------------------------------------------------


class TestEventIdempotency:
    def test_duplicate_source_id_is_deduped(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        first = _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-dup"))
        second = _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-dup"))

        assert first.duplicate is False
        assert second.duplicate is True
        assert second.event_id == first.event_id

        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            refs = batch_repo.list_events_for_batch(first.job_id)
            # Re-delivery must not add a second membership row.
            assert len(refs) == 1
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Windowed flush eligibility
# ---------------------------------------------------------------------------


class TestWindowedFlush:
    def test_batch_flushes_only_after_window_elapses(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        # Configure a 1-minute window for this pot, then submit one event.
        session = pg_test_db.sessionmaker()
        try:
            db_container.ingestion_config(session).set(
                pot_id=pot_id, mode="windowed", window_minutes=1, min_batch_size=None
            )
            session.commit()
        finally:
            session.close()

        receipt = _submit(
            db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-win")
        )
        batch_id = receipt.job_id

        session = pg_test_db.sessionmaker()
        try:
            config = db_container.ingestion_config(session)
            batches = db_container.batch_repository(session)
            spy = _SpyJobs()

            # Before the window elapses: not eligible.
            early = flush_ready_windowed_pots(
                config=config, batches=batches, jobs=spy, now_unix_seconds=time.time()
            )
            assert early.batches_enqueued == 0
            assert spy.enqueued == []

            # 2 minutes later (> 1-minute window): eligible and enqueued.
            late = flush_ready_windowed_pots(
                config=config,
                batches=batches,
                jobs=spy,
                now_unix_seconds=time.time() + 120,
            )
            assert late.batches_enqueued == 1
            assert batch_id in spy.enqueued
            # Enqueue does not transition the batch — the worker claims it later.
            assert batches.get_open_batch_id_for_pot(pot_id) == batch_id
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Stale-batch reaping (worker crash / OOM recovery)
# ---------------------------------------------------------------------------


class TestReapStaleBatch:
    def test_stale_in_flight_batch_is_failed_and_events_surfaced(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        r1 = _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-a"))
        _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-b"))
        batch_id = r1.job_id

        # Claim it (simulating a worker dequeue), then backdate claimed_at so
        # the batch looks abandoned past its lease.
        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            claimed = batch_repo.claim_batch_by_id(batch_id)
            assert claimed is not None and claimed.status == BATCH_STATUS_CLAIMED
            session.commit()
            session.execute(
                text(
                    "UPDATE context_reconciliation_batches "
                    "SET claimed_at = now() - interval '3 hours' WHERE id = :id"
                ),
                {"id": batch_id},
            )
            session.commit()
        finally:
            session.close()

        # Reap with a 2-hour lease: the 3-hour-old claim is definitively dead.
        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            ledger = db_container.reconciliation_ledger(session)
            outcome = reap_stale_batches(
                batches=batch_repo, reco_ledger=ledger, lease_seconds=7200.0
            )
            session.commit()

            assert outcome.batches_reaped == 1
            assert batch_id in outcome.reaped_batch_ids
            assert outcome.events_failed == 2  # both events surfaced for retry
            assert batch_repo.get_batch(batch_id).status == BATCH_STATUS_FAILED
        finally:
            session.close()


# ---------------------------------------------------------------------------
# A new event opens a fresh batch while the prior one is in-flight
# ---------------------------------------------------------------------------


class TestFreshBatchWhenInFlight:
    def test_new_event_opens_fresh_batch_after_claim(
        self, db_container, pg_test_db, pot_id
    ) -> None:
        first = _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-1"))

        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            batch_repo.claim_batch_by_id(first.job_id)  # → in-flight
            session.commit()
        finally:
            session.close()

        second = _submit(db_container, pg_test_db.sessionmaker, _req(pot_id, "evt-2"))
        assert second.job_id != first.job_id

        session = pg_test_db.sessionmaker()
        try:
            batch_repo = db_container.batch_repository(session)
            assert batch_repo.get_batch(first.job_id).status == BATCH_STATUS_CLAIMED
            assert batch_repo.get_batch(second.job_id).status == BATCH_STATUS_PENDING
            refs = batch_repo.list_events_for_batch(second.job_id)
            assert {r.event_id for r in refs} == {second.event_id}
        finally:
            session.close()
