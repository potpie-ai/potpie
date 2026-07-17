"""Drive the engine's real reconciliation pipeline.

Submits each ``ReplayEvent`` via ``POST /api/v2/context/events/reconcile``
(the canonical agent-reconciliation entry point) and polls the event
ledger until the event reaches a terminal state. The reconciliation
agent runs for real on the engine side — no mocking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from potpie_context_engine.adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)

from potpie_context_engine.benchmarks.core.replay import ReplayEvent

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 1.5
DEFAULT_TIMEOUT_S = 180.0
# Both canonical statuses (from get_event endpoint) and raw DB statuses
# (from list_events / some paths) need to be treated as terminal.
TERMINAL_STATUSES = frozenset(
    {
        # canonical (potpie_context_engine.domain.ingestion_event_models.IngestionEventStatus)
        "done",
        "error",
        # raw DB statuses (potpie_context_engine.domain.ingestion_db_status._DB_STATUSES_DONE / _ERROR)
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
    # Engine lifecycle snapshot at terminal-state time — populated on
    # failure / timeout so the ingestion evaluator can include
    # `stage`, `job_id`, `step_error`, and `lifecycle_status` in the
    # error message. Empty on success.
    lifecycle: dict[str, Any] = field(default_factory=dict)


def _event_body(
    pot_id: str, event: ReplayEvent, *, fallback_repo_name: str | None = None
) -> dict[str, Any]:
    """The canonical /events/reconcile body for a replay event.

    Shared by the HTTP submit path and the in-process fast path so both bind
    the event to the pot's repo identically.
    """
    repo_name = fallback_repo_name or event.repo_name
    return {
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


def submit_event(
    client: PotpieContextApiClient,
    pot_id: str,
    event: ReplayEvent,
    *,
    fallback_repo_name: str | None = None,
) -> dict[str, Any]:
    """Submit a single replay event via /events/reconcile.

    ``repo_name`` is required by the engine and validated against the
    pot's repo binding (``repo_not_in_pot`` if mismatched). Linear
    events typically carry ``repo_name=None`` in the envelope (not
    repo-scoped), and so do universe seed events — they describe the
    canonical org, not a specific repo of the bench's runtime pot. In
    both cases we substitute the pot's primary repo so the submission
    passes the binding check.

    Seed events with an explicit envelope-level repo (e.g.
    ``acme/platform``) are also rewritten to the pot's repo — the
    bench has no way to bind an arbitrary pot to the universe's
    intended monorepo, and the agent extracts org/repo references
    from the payload content regardless.
    """
    # Every synthetic fixture references the universe monorepo (e.g.
    # ``acme/platform``), but the ephemeral pot is bound to
    # ``POTPIE_BENCH_REPO`` (e.g. ``acme/sandbox``). Bind *all* events —
    # seeds, signals, and distractors — to the pot's repo so submission
    # passes the ``repo_not_in_pot`` check. The reconciliation agent still
    # extracts org/repo references from the payload content regardless, so
    # the bound repo only satisfies the partition check; it does not change
    # what gets reconciled. (Previously only ``role == "seed"`` was
    # rewritten, so every signal/distractor event failed submission.)
    body = _event_body(pot_id, event, fallback_repo_name=fallback_repo_name)
    response = client.post_context("/events/reconcile", json_body=body)
    client._raise_for_status(response)
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


_LIFECYCLE_FIELDS = (
    "status",
    "lifecycle_status",
    "stage",
    "step_done",
    "step_total",
    "step_error",
    "job_id",
    "reconciliation_runs",
    "error",
    "error_message",
)


def _lifecycle_snapshot(event_row: dict[str, Any]) -> dict[str, Any]:
    """Pull the diagnostic fields a failed ingest should surface."""
    out: dict[str, Any] = {}
    for k in _LIFECYCLE_FIELDS:
        v = event_row.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


def _format_lifecycle(snapshot: dict[str, Any]) -> str:
    """One-line, log-friendly representation. Empty if snapshot is empty."""
    if not snapshot:
        return ""
    parts = []
    for k in _LIFECYCLE_FIELDS:
        if k in snapshot:
            parts.append(f"{k}={snapshot[k]!r}")
    return " ".join(parts)


class _IngestTimeout(TimeoutError):
    """Carries the last poll response so the caller can record a snapshot."""

    def __init__(self, message: str, last: dict[str, Any]) -> None:
        super().__init__(message)
        self.last = last


def wait_for_event_terminal(
    client: PotpieContextApiClient,
    event_id: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Poll /events/{event_id} until terminal or timeout.

    On timeout, the raised ``_IngestTimeout`` carries the last poll
    response so callers can record the engine-side lifecycle snapshot.
    """
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        try:
            last = client.get_event(event_id)
        except PotpieContextApiError as exc:
            logger.debug("get_event %s transient: %s", event_id, exc)
            time.sleep(poll_interval_s)
            continue
        status = str(last.get("status") or "").lower()
        if status in TERMINAL_STATUSES:
            return last
        time.sleep(poll_interval_s)
    raise _IngestTimeout(
        f"event {event_id} did not reach terminal state in {timeout_s}s "
        f"(last status: {last.get('status')!r})",
        last=last,
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
        submission = submit_event(
            client, pot_id, event, fallback_repo_name=fallback_repo_name
        )
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
    except _IngestTimeout as exc:
        snapshot = _lifecycle_snapshot(exc.last)
        suffix = f" [{_format_lifecycle(snapshot)}]" if snapshot else ""
        return IngestionOutcome(
            fixture_path=event.fixture_path,
            event_id=str(event_id),
            terminal_status="timeout",
            duration_s=time.monotonic() - started,
            error=f"{exc}{suffix}",
            lifecycle=snapshot,
        )

    return _outcome_from_terminal(
        fixture_path=event.fixture_path,
        event_id=str(event_id),
        terminal=terminal,
        duration_s=time.monotonic() - started,
    )


def _outcome_from_terminal(
    *, fixture_path: str, event_id: str, terminal: dict[str, Any], duration_s: float
) -> IngestionOutcome:
    """Build an :class:`IngestionOutcome` from a terminal get_event row."""
    status = str(terminal.get("status") or "").lower()
    success_statuses = {"done", "reconciled"}
    error = (
        terminal.get("error") or terminal.get("error_message")
        if status not in success_statuses
        else None
    )
    soft_downgrades = len(terminal.get("downgrades") or [])
    snapshot: dict[str, Any] = {}
    if status not in success_statuses:
        snapshot = _lifecycle_snapshot(terminal)
        if snapshot:
            error = (
                error or "non-success terminal status"
            ) + f" [{_format_lifecycle(snapshot)}]"
    return IngestionOutcome(
        fixture_path=fixture_path,
        event_id=event_id,
        terminal_status=status or "unknown",
        duration_s=duration_s,
        error=str(error) if error else None,
        soft_downgrades=soft_downgrades,
        lifecycle=snapshot,
    )


def replay_all(
    client: PotpieContextApiClient,
    pot_id: str,
    events: list[ReplayEvent],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    fallback_repo_name: str | None = None,
    parallel_roles: tuple[str, ...] = ("seed", "distractor"),
    max_concurrency: int = 6,
) -> list[IngestionOutcome]:
    """Run an ingestion timeline against the engine.

    Signal events run sequentially (most scenarios encode order in their
    ``ingest:`` list, and out-of-order arrival is itself a graded
    behaviour on adversarial scenarios). Seed and distractor events run
    in parallel up to ``max_concurrency`` since they have no implied
    arrival order — this is the difference between an 8-seed warmup
    taking 8 × T seconds and ~T seconds.

    Returns outcomes in the same order as ``events`` so the evaluator
    can correlate by index.
    """
    if getattr(client, "inprocess", False):
        return _replay_all_inprocess(
            client, pot_id, events, fallback_repo_name=fallback_repo_name
        )

    from concurrent.futures import ThreadPoolExecutor

    # Phase 1: run parallel-eligible events (seeds + distractors) concurrently.
    parallel_indices = [i for i, e in enumerate(events) if e.role in parallel_roles]
    sequential_indices = [
        i for i, e in enumerate(events) if e.role not in parallel_roles
    ]
    outcomes: dict[int, IngestionOutcome] = {}

    def _one(idx_event: tuple[int, ReplayEvent]) -> tuple[int, IngestionOutcome]:
        idx, ev = idx_event
        return idx, submit_and_wait(
            client,
            pot_id,
            ev,
            timeout_s=timeout_s,
            fallback_repo_name=fallback_repo_name,
        )

    if parallel_indices:
        n_workers = max(1, min(max_concurrency, len(parallel_indices)))
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="bench-ingest"
        ) as pool:
            for idx, outcome in pool.map(
                _one, ((i, events[i]) for i in parallel_indices)
            ):
                outcomes[idx] = outcome
                if outcome.error:
                    logger.warning(
                        "ingest failed: %s status=%s error=%s",
                        events[idx].fixture_path,
                        outcome.terminal_status,
                        outcome.error,
                    )

    # Phase 2: signals in declared order. They may depend on prior signals
    # having reached the graph (e.g. an issue_state_change after an
    # issue_create), so we don't run these concurrently.
    for idx in sequential_indices:
        event = events[idx]
        outcome = submit_and_wait(
            client,
            pot_id,
            event,
            timeout_s=timeout_s,
            fallback_repo_name=fallback_repo_name,
        )
        outcomes[idx] = outcome
        if outcome.error:
            logger.warning(
                "ingest failed: %s status=%s error=%s",
                event.fixture_path,
                outcome.terminal_status,
                outcome.error,
            )

    return [outcomes[i] for i in range(len(events))]


def _replay_all_inprocess(
    client: Any,
    pot_id: str,
    events: list[ReplayEvent],
    *,
    fallback_repo_name: str | None = None,
) -> list[IngestionOutcome]:
    """In-process ingestion: submit all events, reconcile once, read back.

    The in-process driver has no worker/flush timer, so we admit every event
    into the pot's open batch (single-threaded — no asyncio-across-threads
    race), then reconcile the batch(es) inline in one pass, then read each
    event's terminal state. Order is preserved by index.
    """
    started = time.monotonic()
    logger.info("in-process: submitting %d events into the batch queue...", len(events))
    event_ids: list[str] = []
    for ev in events:
        body = _event_body(pot_id, ev, fallback_repo_name=fallback_repo_name)
        try:
            event_ids.append(str(client.submit_only(pot_id, body)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("in-process submit failed: %s: %s", ev.fixture_path, exc)
            event_ids.append("")

    logger.info(
        "in-process: reconciling %d events inline (this is the slow step)...",
        len(events),
    )
    try:
        n = client.process_pending(pot_id)
        logger.info(
            "in-process: reconciled %d batch(es) in %.1fs",
            n,
            time.monotonic() - started,
        )
    except Exception:  # noqa: BLE001 — per-event status still read below
        logger.exception("in-process process_pending failed for pot %s", pot_id)

    dur = time.monotonic() - started
    outcomes: list[IngestionOutcome] = []
    for ev, eid in zip(events, event_ids):
        if not eid:
            outcomes.append(
                IngestionOutcome(
                    fixture_path=ev.fixture_path,
                    event_id=None,
                    terminal_status="submit_failed",
                    duration_s=dur,
                    error="in-process submit failed",
                )
            )
            continue
        row = client.get_event(eid)
        if not row:
            outcomes.append(
                IngestionOutcome(
                    fixture_path=ev.fixture_path,
                    event_id=eid,
                    terminal_status="unknown",
                    duration_s=dur,
                    error="event row not found after processing",
                )
            )
            continue
        outcome = _outcome_from_terminal(
            fixture_path=ev.fixture_path, event_id=eid, terminal=row, duration_s=dur
        )
        if outcome.error:
            logger.warning(
                "in-process ingest issue: %s status=%s error=%s",
                ev.fixture_path,
                outcome.terminal_status,
                outcome.error,
            )
        outcomes.append(outcome)
    return outcomes
