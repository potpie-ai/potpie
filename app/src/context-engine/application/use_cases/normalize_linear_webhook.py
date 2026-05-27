"""Turn a Linear webhook body into a canonical :class:`IngestionSubmissionRequest`.

Linear webhooks ship with a small envelope — ``{action, type, data, ...}`` —
and we care about three kinds today:

* ``type="Issue"``, ``action="create"``    → linear ``issue`` event with action ``create``
* ``type="Issue"``, ``action="update"``    → ``update`` or ``state_change`` when
  ``updatedFrom.stateId`` is set
* ``type="Comment"``, ``action="create"``  → ``comment_added``; the envelope
  carries the full comment under ``data`` plus an issue reference under
  ``data.issue``. The resolver and planner then hydrate the issue details.

The output request carries ``ingestion_kind="agent_reconciliation"`` and a
deterministic ``source_id`` so repeat deliveries dedupe at the ledger.
"""

from __future__ import annotations

from typing import Any

from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.linear_events import parse_linear_datetime


class LinearWebhookError(ValueError):
    """Raised when the webhook body cannot be mapped to a context event."""


def normalize_linear_webhook(
    body: dict[str, Any],
    *,
    pot_id: str,
    source_channel: str = "webhook",
) -> IngestionSubmissionRequest:
    """Map a Linear webhook body to an :class:`IngestionSubmissionRequest`.

    Raises :class:`LinearWebhookError` for unknown types/actions so the inbound
    adapter can decide whether to 202-accept-and-drop or 400-reject.
    """
    if not isinstance(body, dict):
        raise LinearWebhookError("linear webhook body must be an object")
    ev_type = str(body.get("type") or "").strip()
    raw_action = str(body.get("action") or "").strip().lower()
    data = body.get("data")
    if not isinstance(data, dict):
        raise LinearWebhookError("linear webhook body missing 'data' object")

    if ev_type == "Issue":
        return _issue_request(body, data, raw_action, pot_id=pot_id, source_channel=source_channel)
    if ev_type == "Comment":
        return _comment_request(body, data, raw_action, pot_id=pot_id, source_channel=source_channel)
    raise LinearWebhookError(f"unsupported linear webhook type: {ev_type!r}")


def _issue_request(
    body: dict[str, Any],
    data: dict[str, Any],
    raw_action: str,
    *,
    pot_id: str,
    source_channel: str,
) -> IngestionSubmissionRequest:
    identifier = str(data.get("identifier") or "").strip()
    issue_id = str(data.get("id") or "").strip()
    if not identifier and not issue_id:
        raise LinearWebhookError("linear issue webhook missing id/identifier")

    updated_from = body.get("updatedFrom")
    action = raw_action or "update"
    previous_state = None
    if action == "update" and isinstance(updated_from, dict) and updated_from.get("stateId"):
        action = "state_change"
        # Linear's webhook only gives the prior stateId; we store it so downstream
        # planners can render "unknown -> <new>". Name hydration happens via the
        # resolver when present.
        previous_state = {
            "name": str(updated_from.get("stateName") or "previous"),
            "type": updated_from.get("stateType"),
        }
    if action == "remove":
        # Keep terminology aligned with LINEAR_ISSUE_ACTIONS.
        action = "remove"

    team_id = _team_id(data)
    payload: dict[str, Any] = {
        "action": action,
        "issue": data,
    }
    if previous_state is not None:
        payload["previous_state"] = previous_state

    source_id = _issue_source_id(identifier or issue_id, action)
    occurred_at = parse_linear_datetime(body.get("createdAt") or data.get("updatedAt"))
    return IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel=source_channel,
        source_system="linear",
        event_type="linear_issue",
        action=action,
        payload=payload,
        source_id=source_id,
        provider="linear",
        provider_host="linear.app",
        repo_name=f"team:{team_id}" if team_id else None,
        source_event_id=str(body.get("webhookId") or body.get("id") or "") or None,
        occurred_at=occurred_at,
    )


def _comment_request(
    body: dict[str, Any],
    data: dict[str, Any],
    raw_action: str,
    *,
    pot_id: str,
    source_channel: str,
) -> IngestionSubmissionRequest:
    if raw_action != "create":
        raise LinearWebhookError(
            f"unsupported linear comment action: {raw_action!r} (only 'create')"
        )
    issue = data.get("issue")
    if not isinstance(issue, dict):
        raise LinearWebhookError("linear comment webhook missing 'data.issue'")
    identifier = str(issue.get("identifier") or "").strip()
    if not identifier and not issue.get("id"):
        raise LinearWebhookError("linear comment webhook missing issue id/identifier")
    team_id = _team_id(issue)
    comment_id = str(data.get("id") or "").strip()
    payload: dict[str, Any] = {
        "action": "comment_added",
        "issue": issue,
        "comment": data,
    }
    source_id = f"linear:comment:{comment_id or identifier}"
    occurred_at = parse_linear_datetime(body.get("createdAt") or data.get("createdAt"))
    return IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel=source_channel,
        source_system="linear",
        event_type="linear_comment",
        action="comment_added",
        payload=payload,
        source_id=source_id,
        provider="linear",
        provider_host="linear.app",
        repo_name=f"team:{team_id}" if team_id else None,
        source_event_id=str(body.get("webhookId") or body.get("id") or comment_id) or None,
        occurred_at=occurred_at,
    )


def _issue_source_id(identifier: str, action: str) -> str:
    return f"linear:issue:{identifier}:{action}"


def _team_id(data: dict[str, Any]) -> str | None:
    if data.get("teamId"):
        return str(data["teamId"])
    team = data.get("team")
    if isinstance(team, dict) and team.get("id"):
        return str(team["id"])
    return None
