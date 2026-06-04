"""Assemble ``ReconciliationRequest`` from a normalized event."""

from __future__ import annotations

from typing import Any

from domain.context_events import ContextEvent
from domain.reconciliation import ReconciliationRequest


def build_reconciliation_request(
    event: ContextEvent,
    *,
    repo_name: str | None = None,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> ReconciliationRequest:
    return ReconciliationRequest(
        event=event,
        pot_id=event.pot_id,
        repo_name=repo_name or event.repo_name,
        prior_attempts=list(prior_attempts or []),
    )
