"""Commit a plan without losing its durable receipt to a lost response.

A commit is a mutation. When its response is lost the write may or may not
have landed, and the daemon protocol makes no promise that resubmitting a
mutation is safe, so this module never resubmits one. What is always safe is
reading the plan's durable receipt, which writes nothing. A lost response, or
an overlapping commit still holding the plan, therefore becomes a short,
bounded series of ``commit_status`` reads.

``--verify`` runs as its own read after the receipt is durable
(``defer_verification``), so a slow or failed readback can never hide a write
that landed: the receipt is reported, and the verification carries its own
status and the command that repeats it.

When polling cannot settle the outcome, the original failure is reported with
its own code and exit status, and the next action names the plan's history
read instead of a retry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from potpie.cli.commands._common import EngineClientError
from potpie_context_engine.core.graph_plans import (
    TERMINAL_PLAN_STATUSES,
    GraphIngestionVerificationResult,
    GraphMutationCommitResult,
)
from potpie_context_engine.requests import (
    CommitRequest,
    CommitStatusRequest,
    VerifyCommitRequest,
)

POLL_ATTEMPTS = 3
POLL_INTERVAL_SECONDS = 0.5

Runner = Callable[[Any], Any]


def outcome_unknown(error: object) -> bool:
    """Whether a failed call may have run anyway (a dispatched mutation)."""

    details = getattr(error, "details", None)
    return (
        getattr(error, "category", None) == "protocol_transport"
        and isinstance(details, Mapping)
        and details.get("outcome_unknown") is True
    )


def commit_with_recovery(
    client: Any,
    *,
    plan_id: str,
    pot_id: str,
    approved_by: str | None,
    verify: bool,
    history_command: str,
    retry_command: str,
    run: Runner,
    sleep: Callable[[float], None] | None = None,
) -> GraphMutationCommitResult:
    """Commit ``plan_id`` once and settle its receipt without a second write.

    ``run`` executes one engine-client awaitable and returns its outcome
    (``ok`` with ``value`` or ``error``). Failures are raised as
    ``EngineClientError`` with their own code, so the command boundary maps
    them to the same exit status a failed commit always had.
    """

    outcome = run(
        client.commit(
            CommitRequest(
                plan_id=plan_id,
                approved_by=approved_by,
                verify=verify,
                defer_verification=verify,
            )
        )
    )
    if not getattr(outcome, "ok", False):
        if not outcome_unknown(outcome.error):
            raise EngineClientError(outcome.error)
        receipt = _poll_receipt(
            client, plan_id=plan_id, pot_id=pot_id, run=run, sleep=sleep
        )
        if receipt is None:
            raise EngineClientError(
                _with_history_action(outcome.error, history_command)
            )
        result = receipt
    else:
        result = outcome.value
    if result.status == "committing":
        # Another caller holds the plan's reservation. Wait for its receipt;
        # never commit the plan a second time from here.
        receipt = _poll_receipt(
            client, plan_id=plan_id, pot_id=pot_id, run=run, sleep=sleep
        )
        result = receipt or replace(
            result,
            detail=result.detail
            or "Another commit of this plan is still running; its outcome is not known yet.",
            recommended_next_action=history_command,
        )
    if result.ok and verify:
        result = replace(
            result,
            verification=_verify(
                client,
                plan_id=plan_id,
                pot_id=pot_id,
                retry_command=retry_command,
                run=run,
            ),
        )
    return result


def _poll_receipt(
    client: Any,
    *,
    plan_id: str,
    pot_id: str,
    run: Runner,
    sleep: Callable[[float], None] | None,
) -> GraphMutationCommitResult | None:
    for attempt in range(POLL_ATTEMPTS):
        if attempt:
            (sleep or time.sleep)(POLL_INTERVAL_SECONDS)
        outcome = run(client.commit_status(CommitStatusRequest(plan_id=plan_id)))
        if not getattr(outcome, "ok", False):
            # An unreachable or busy daemon proves nothing about the write.
            continue
        receipt = outcome.value
        if receipt.plan_id != plan_id or receipt.pot_id != pot_id:
            return None
        if receipt.status in TERMINAL_PLAN_STATUSES or receipt.status == "error":
            return receipt
    return None


def _verify(
    client: Any,
    *,
    plan_id: str,
    pot_id: str,
    retry_command: str,
    run: Runner,
) -> GraphIngestionVerificationResult:
    outcome = run(client.verify_commit(VerifyCommitRequest(plan_id=plan_id)))
    if getattr(outcome, "ok", False):
        return outcome.value
    error = outcome.error
    message = str(getattr(error, "message", "verification failed"))
    return GraphIngestionVerificationResult(
        ok=False,
        status="unknown_completion" if outcome_unknown(error) else "error",
        plan_id=plan_id,
        pot_id=pot_id,
        detail=f"The commit is durable; verification did not complete: {message}",
        recommended_next_action=retry_command,
    )


def _with_history_action(error: object, history_command: str) -> object:
    try:
        return replace(
            error,
            message=(
                "The commit's response was lost and its receipt could not be "
                "read back; it may or may not have been applied."
            ),
            recommended_next_action=(
                f"check whether it landed with '{history_command}' before "
                "proposing the change again"
            ),
        )
    except TypeError:
        return error


__all__ = [
    "POLL_ATTEMPTS",
    "POLL_INTERVAL_SECONDS",
    "commit_with_recovery",
    "outcome_unknown",
]
