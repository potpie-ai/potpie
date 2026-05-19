"""Hatchet implementation of ``ContextGraphJobQueuePort`` (self-hosted control plane)."""

from __future__ import annotations

from typing import Any

from adapters.outbound.hatchet.env_bootstrap import prepare_hatchet_client_env
from domain.hatchet_events import (
    EVENT_MAINTENANCE,
    EVENT_PROCESS_BATCH,
)


class HatchetContextGraphJobQueue:
    """
    Push events to Hatchet; workers must register ``on_events`` for the same keys.

    Configure the Hatchet client via ``HATCHET_CLIENT_TOKEN`` (JWT) and optional
    ``HATCHET_CLIENT_HOST_PORT`` / ``HATCHET_CLIENT_SERVER_URL`` for self-hosting
    (see https://docs.hatchet.run/self-hosting and ``docs/hatchet-local.md`` in this repo).
    """

    def __init__(self, hatchet: Any) -> None:
        self._hatchet = hatchet

    @classmethod
    def from_env(cls) -> HatchetContextGraphJobQueue:
        try:
            from hatchet_sdk import Hatchet
        except ImportError as e:
            raise RuntimeError(
                "Hatchet queue backend requires the hatchet-sdk package. "
                "Install with: pip install hatchet-sdk"
            ) from e
        prepare_hatchet_client_env()
        return cls(Hatchet())

    def enqueue_batch(self, batch_id: str) -> None:
        self._hatchet.event.push(EVENT_PROCESS_BATCH, {"batch_id": batch_id})

    def enqueue_maintenance_sweep(
        self,
        pot_id: str,
        job_family: str,
        *,
        reason: str | None = None,
        triggered_by: str = "quality_report",
    ) -> None:
        """Schedule a maintenance job for a pot.

        job_family must be one of MAINTENANCE_JOB_FAMILIES from graph_quality.
        Workers must register on_events for EVENT_MAINTENANCE.
        """
        self._hatchet.event.push(
            EVENT_MAINTENANCE,
            {
                "pot_id": pot_id,
                "job_family": job_family,
                "reason": reason or "",
                "triggered_by": triggered_by,
            },
        )

    def dispatch_quality_recommendations(
        self,
        pot_id: str,
        recommendations: list[dict[str, str]],
    ) -> int:
        """Dispatch all maintenance jobs recommended by a GraphQualityReport."""
        count = 0
        for rec in recommendations:
            job_family = rec.get("job", "")
            reason = rec.get("reason", "")
            if job_family:
                self.enqueue_maintenance_sweep(
                    pot_id,
                    job_family,
                    reason=reason,
                    triggered_by="quality_report",
                )
                count += 1
        return count
