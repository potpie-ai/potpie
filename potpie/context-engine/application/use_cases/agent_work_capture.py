"""Capture agent-working events against a reconciliation run."""

from __future__ import annotations

from typing import Any

from domain.ports.reconciliation_agent import ReconciliationAgentPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort


def bind_agent_work_recorder(
    agent: ReconciliationAgentPort,
    ledger: ReconciliationLedgerPort,
    run_id: str,
) -> None:
    """Attach durable run-event capture when the agent implementation supports it."""
    setter = getattr(agent, "set_work_event_recorder", None)
    if callable(setter):
        setter(_LedgerAgentWorkRecorder(ledger, run_id))


def clear_agent_work_recorder(agent: ReconciliationAgentPort) -> None:
    setter = getattr(agent, "set_work_event_recorder", None)
    if callable(setter):
        setter(None)


class _LedgerAgentWorkRecorder:
    def __init__(self, ledger: ReconciliationLedgerPort, run_id: str) -> None:
        self._ledger = ledger
        self._run_id = run_id

    def record(
        self,
        event_kind: str,
        *,
        title: str | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._ledger.record_run_work_event(
            self._run_id,
            event_kind=event_kind,
            title=title,
            body=body,
            payload=payload,
        )
