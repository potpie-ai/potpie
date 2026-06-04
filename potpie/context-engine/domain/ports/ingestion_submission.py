"""Inbound submission port: single entry for producers (async + sync waiter)."""

from __future__ import annotations

from typing import Protocol

from domain.ingestion_event_models import EventReceipt, IngestionSubmissionRequest


class IngestionSubmissionService(Protocol):
    """
    Accept canonical submission requests, dedupe, persist, enqueue, optionally wait.

    Inbound adapters (HTTP, CLI, webhooks) should call only this service for ingestion.
    """

    def submit(
        self,
        request: IngestionSubmissionRequest,
        *,
        sync: bool = False,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> EventReceipt:
        """Persist event, enqueue processing.

        ``sync``: run inline reconcile/apply instead of enqueue (HTTP/CLI sync modes).

        When ``wait`` (and not ``sync``), block until the event store reports a terminal status.
        """
        ...
