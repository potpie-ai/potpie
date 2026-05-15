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

    def add_events_to_open_batch_for_pot(
        self, pot_id: str, event_ids: list[str]
    ) -> str:
        """Bulk variant of ``upsert_open_batch_for_pot``.

        Adds *all* of ``event_ids`` to the pot's open ``pending`` batch in a
        single transaction: one advisory lock, one batch resolution, one
        bulk membership insert, one commit — regardless of how many ids are
        passed. Same coalescing / in-flight semantics as the single-event
        method (events that arrive while a batch is claimed/running open a
        fresh pending batch), and idempotent on ``(batch_id, event_id)``
        membership so re-adding an event is a no-op rather than a duplicate.

        ``event_ids`` must be non-empty. Returns the batch id; the whole set
        lands in that one batch, which the worker then hands to the agent
        in one chunked pass.
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

    def get_open_batch_id_for_pot(self, pot_id: str) -> str | None:
        """Return the id of the pot's currently-pending batch, or ``None``.

        Used by the windowed-batch flusher and the user-facing
        "force ingest now" endpoint to enqueue without admitting a new
        event. ``upsert_open_batch_for_pot`` creates a batch on demand;
        this method is the read-only sibling — it never creates one.
        """
        ...

    def get_latest_batch_id_for_event(self, event_id: str) -> str | None:
        """Most-recent batch an event belongs to, or ``None``.

        Backs the per-event activity stream: the durable execution log is
        batch-scoped, so the stream endpoint resolves event → newest batch
        (a retry re-adds the event to a fresh batch) and tails that.
        """
        ...
