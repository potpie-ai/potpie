"""Drive the engine's real reconciliation pipeline.

Submits each ``ReplayEvent`` via ``POST /api/v2/context/events/reconcile``
(the canonical agent-reconciliation entry point) and polls the event
ledger until the event reaches a terminal state. The reconciliation
agent runs for real on the engine side — no mocking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)

from benchmarks.core.replay import ReplayEvent

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 1.5
DEFAULT_TIMEOUT_S = 180.0
# Both canonical statuses (from get_event endpoint) and raw DB statuses
# (from list_events / some paths) need to be treated as terminal.
TERMINAL_STATUSES = frozenset(
    {
        # canonical (domain.ingestion_event_models.IngestionEventStatus)
        "done",
        "error",
        # raw DB statuses (domain.ingestion_db_status._DB_STATUSES_DONE / _ERROR)
        "reconciled",
        "failed",
        # additional engine paths
        "rejected",
        "reconciliation_rejected",
    }
)


@dataclass(frozen=True)
class IngestionOutcome:
    fixture_path: str
    event_id: str | None
    terminal_status: str
    duration_s: float
    error: str | None = None
    soft_downgrades: int = 0


def submit_event(
    client: PotpieContextApiClient,
    pot_id: str,
    event: ReplayEvent,
    *,
    fallback_repo_name: str | None = None,
) -> dict[str, Any]:
    """Submit a single replay event via /events/reconcile.

    ``repo_name`` is required by the engine. Linear events typically
    carry ``repo_name=None`` in the envelope (they're not repo-scoped),
    but the bench's ephemeral pot has a primary repo attached, so we
    fall back to that.
    """
    repo_name = event.repo_name or fallback_repo_name
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "ingestion_kind": "agent_reconciliation",
        "source_system": event.connector,
        "event_type": event.event_type,
        "action": event.action,
        "source_id": event.source_id,
        "occurred_at": event.occurred_at.isoformat(),
        "payload": event.payload,
        "repo_name": repo_name or "",
    }
    response = client.post_context("/events/reconcile", json_body=body)
    client._raise_for_status(response)
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def wait_for_event_terminal(
    client: PotpieContextApiClient,
    event_id: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Poll /events/{event_id} until terminal or timeout."""
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        try:
            last = client.get_event(event_id)
        except PotpieContextApiError as exc:
            # 404 right after submission is normal — the row may not be
            # readable yet on a slow DB. Treat any non-200 as transient.
            logger.debug("get_event %s transient: %s", event_id, exc)
            time.sleep(poll_interval_s)
            continue
        status = str(last.get("status") or "").lower()
        if status in TERMINAL_STATUSES:
            return last
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"event {event_id} did not reach terminal state in {timeout_s}s "
        f"(last status: {last.get('status')!r})"
    )


def submit_and_wait(
    client: PotpieContextApiClient,
    pot_id: str,
    event: ReplayEvent,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    fallback_repo_name: str | None = None,
) -> IngestionOutcome:
    started = time.monotonic()
    try:
        submission = submit_event(client, pot_id, event, fallback_repo_name=fallback_repo_name)
    except PotpieContextApiError as exc:
        return IngestionOutcome(
            fixture_path=event.fixture_path,
            event_id=None,
            terminal_status="submit_failed",
            duration_s=time.monotonic() - started,
            error=f"submit_failed: {exc}",
        )

    event_id = submission.get("event_id") or submission.get("id")
    if not event_id:
        return IngestionOutcome(
            fixture_path=event.fixture_path,
            event_id=None,
            terminal_status="submit_failed",
            duration_s=time.monotonic() - started,
            error=f"no event_id in response: {submission!r}",
        )

    try:
        terminal = wait_for_event_terminal(client, str(event_id), timeout_s=timeout_s)
    except TimeoutError as exc:
        return IngestionOutcome(
            fixture_path=event.fixture_path,
            event_id=str(event_id),
            terminal_status="timeout",
            duration_s=time.monotonic() - started,
            error=str(exc),
        )

    status = str(terminal.get("status") or "").lower()
    # Both the canonical "done" (from get_event) and the raw DB "reconciled"
    # mean the event was fully reconciled. Anything else carrying an
    # explicit error field is a failure.
    success_statuses = {"done", "reconciled"}
    error = (
        terminal.get("error") or terminal.get("error_message")
        if status not in success_statuses
        else None
    )
    soft_downgrades = len(terminal.get("downgrades") or [])
    return IngestionOutcome(
        fixture_path=event.fixture_path,
        event_id=str(event_id),
        terminal_status=status or "unknown",
        duration_s=time.monotonic() - started,
        error=str(error) if error else None,
        soft_downgrades=soft_downgrades,
    )


def replay_all(
    client: PotpieContextApiClient,
    pot_id: str,
    events: list[ReplayEvent],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    fallback_repo_name: str | None = None,
) -> list[IngestionOutcome]:
    """Submit events sequentially, blocking on reconciliation each time."""
    outcomes: list[IngestionOutcome] = []
    for event in events:
        outcome = submit_and_wait(
            client, pot_id, event, timeout_s=timeout_s, fallback_repo_name=fallback_repo_name
        )
        outcomes.append(outcome)
        if outcome.error:
            logger.warning(
                "ingest failed: %s status=%s error=%s",
                event.fixture_path, outcome.terminal_status, outcome.error,
            )
    return outcomes
