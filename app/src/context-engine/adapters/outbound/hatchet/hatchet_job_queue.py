"""Hatchet implementation of ``ContextGraphJobQueuePort`` (self-hosted control plane)."""

from __future__ import annotations

from typing import Any

from adapters.outbound.hatchet.env_bootstrap import prepare_hatchet_client_env
from domain.hatchet_events import (
    EVENT_APPLY_EPISODE,
    EVENT_BACKFILL,
    EVENT_INGEST_PR,
    EVENT_INGESTION_AGENT,
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

    def enqueue_backfill(
        self, pot_id: str, *, target_repo_name: str | None = None
    ) -> None:
        payload: dict = {"pot_id": pot_id}
        if target_repo_name is not None:
            payload["target_repo_name"] = target_repo_name
        self._hatchet.event.push(EVENT_BACKFILL, payload)

    def enqueue_ingest_pr(
        self,
        pot_id: str,
        pr_number: int,
        *,
        is_live_bridge: bool = True,
        repo_name: str | None = None,
    ) -> None:
        payload: dict = {
            "pot_id": pot_id,
            "pr_number": pr_number,
            "is_live_bridge": is_live_bridge,
        }
        if repo_name is not None:
            payload["repo_name"] = repo_name
        self._hatchet.event.push(EVENT_INGEST_PR, payload)

    def enqueue_ingestion_event(self, event_id: str, *, pot_id: str, kind: str) -> None:
        self._hatchet.event.push(
            EVENT_INGESTION_AGENT,
            {"event_id": event_id, "pot_id": pot_id, "kind": kind},
        )

    def enqueue_episode_apply(self, pot_id: str, event_id: str, sequence: int) -> None:
        self._hatchet.event.push(
            EVENT_APPLY_EPISODE,
            {
                "pot_id": pot_id,
                "event_id": event_id,
                "sequence": sequence,
            },
        )
