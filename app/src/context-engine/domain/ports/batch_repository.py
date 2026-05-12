"""Reconciliation batch repository (port).

Producers call ``upsert_open_batch_for_pot`` to coalesce an event into the
pot's open ``pending`` batch (creating one if none exists). The host queue
adapter then enqueues a worker job for that batch id; the worker calls
``claim_batch_by_id`` to atomically transition the row to ``claimed`` and
invoke the agent.
"""

from __future__ import annotations

from typing import Protocol

from domain.reconciliation_batch import BatchEventRef, ReconciliationBatch


class BatchRepositoryPort(Protocol):
    def upsert_open_batch_for_pot(self, pot_id: str, event_id: str) -> str:
        """Add the event to the pot's open ``pending`` batch, creating one if needed.

        - Creates a new ``pending`` batch if no pending batch exists for the pot.
        - Returns the existing ``pending`` batch id when one is already open
          (so burst events coalesce into a single agent run).
        - Events arriving while the batch is ``claimed`` / ``running`` create
          a *new* ``pending`` batch — the in-flight run has already snapshotted
          its event list and would otherwise miss them.
        - Idempotent on ``event_id`` membership: re-adding the same event
          does not duplicate the membership row.

        Returns the batch id.
        """
        ...

    def claim_batch_by_id(self, batch_id: str) -> ReconciliationBatch | None:
        """Atomically transition ``pending`` → ``claimed`` for one batch.

        Returns the claimed batch, or ``None`` if the batch is missing or
        already past ``pending`` (a redundant enqueue races and loses).
        Implementations should use ``FOR UPDATE SKIP LOCKED`` so concurrent
        workers don't double-claim the same row.
        """
        ...

    def get_batch(self, batch_id: str) -> ReconciliationBatch | None: ...

    def list_events_for_batch(self, batch_id: str) -> list[BatchEventRef]: ...

    def mark_batch_running(self, batch_id: str) -> None:
        """Move ``claimed`` → ``running`` once the worker actually starts processing."""

    def mark_batch_done(
        self,
        batch_id: str,
        *,
        completed_event_ids: list[str],
    ) -> None:
        """Mark batch + named events processed. Other events in the batch (if any)
        remain unmarked so a follow-up retry can pick them up if needed."""

    def mark_batch_failed(self, batch_id: str, error: str) -> None: ...
