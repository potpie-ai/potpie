"""Commit orchestration: retain durable receipts across transport failures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import time

from potpie.daemon.client import RemoteSurface, RpcDeadlineExceeded
from potpie_context_core.errors import CapabilityNotImplemented, ContextEngineDisabled
from potpie_context_core.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationCommitResult,
    TERMINAL_PLAN_STATUSES,
)

_POLL_ATTEMPTS = 3
_POLL_TIMEOUT_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.5


@contextmanager
def _deadline(workbench, timeout: float | None):
    # Control transport locally; never send timeout as a service keyword.
    if not isinstance(workbench, RemoteSurface) or timeout is None:
        yield
        return
    client = workbench._client
    previous = client.timeout_s
    client.timeout_s = timeout
    try:
        yield
    finally:
        client.timeout_s = previous


def commit_with_recovery(
    workbench,
    *,
    plan_id: str,
    pot_id: str,
    approved_by: str | None,
    verify: bool,
    timeout: float | None,
    history_command: str,
    retry_command: str,
) -> GraphMutationCommitResult:
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("--timeout must be a finite positive number of seconds")
    options = {"defer_verification": True} if verify else {}
    with _deadline(workbench, timeout):
        try:
            result = workbench.commit(
                plan_id,
                pot_id=pot_id,
                approved_by=approved_by,
                verify=verify,
                **options,
            )
        except RpcDeadlineExceeded:
            result = _recover_receipt(workbench, plan_id, pot_id, history_command)
        # An overlapping retry can receive the active reservation before the
        # first caller finishes. Poll it, but never resubmit a mutation here.
        if result.status == "committing":
            result = _recover_receipt(workbench, plan_id, pot_id, history_command)
        if result.ok and verify:
            try:
                verification = workbench.verify_commit(plan_id, pot_id=pot_id)
            except Exception as exc:  # A failed check cannot erase a durable receipt.
                verification = GraphIngestionVerificationResult(
                    ok=False,
                    status="unknown_completion"
                    if isinstance(exc, RpcDeadlineExceeded)
                    else "error",
                    plan_id=plan_id,
                    pot_id=pot_id,
                    detail=f"Commit succeeded; verification did not complete: {exc}",
                    recommended_next_action=retry_command,
                )
            result = replace(result, verification=verification)
        return result


def _recover_receipt(workbench, plan_id, pot_id, history_command):
    for attempt in range(_POLL_ATTEMPTS):
        if attempt:
            time.sleep(_POLL_INTERVAL_SECONDS)
        try:
            with _deadline(workbench, _POLL_TIMEOUT_SECONDS):
                receipt = workbench.commit_status(plan_id, pot_id=pot_id)
            if receipt.plan_id != plan_id or receipt.pot_id != pot_id:
                break
            if receipt.status in TERMINAL_PLAN_STATUSES or receipt.status == "error":
                return receipt
        except (ContextEngineDisabled, CapabilityNotImplemented, ValueError):
            # A peer may be unavailable or too old to expose commit_status.
            # Neither proves that the original write failed.
            continue
    return GraphMutationCommitResult(
        ok=False,
        status="unknown_completion",
        plan_id=plan_id,
        pot_id=pot_id,
        risk="unknown",
        detail="Commit completion remains unknown after bounded status polling.",
        recommended_next_action=history_command,
    )
