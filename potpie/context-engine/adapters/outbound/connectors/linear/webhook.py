"""Linear webhook normalization.

Linear webhooks ship with a small envelope — ``{action, type, data, ...}`` —
and we care about three kinds today:

* ``type="Issue"``, ``action="create"``    → linear ``issue`` event with action ``create``
* ``type="Issue"``, ``action="update"``    → ``update`` or ``state_change`` when
  ``updatedFrom.stateId`` is set
* ``type="Comment"``, ``action="create"``  → ``comment_added``; the envelope
  carries the full comment under ``data`` plus an issue reference under
  ``data.issue``. The resolver and planner then hydrate the issue details.

The output is a :class:`ContextEvent` produced by
:func:`linear_payload_to_event`. The connector's
:meth:`SourceConnectorPort.normalize_webhook` wraps that — it takes raw
bytes/headers and returns the event (or ``None`` for events Linear sends
that we choose to ignore).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from adapters.outbound.connectors.linear.events import parse_linear_datetime
from domain.context_events import ContextEvent


class LinearWebhookError(ValueError):
    """Raised when the webhook body cannot be mapped to a context event."""


def linear_payload_to_event(body: dict[str, Any]) -> ContextEvent:
    """Map a parsed Linear webhook body to a :class:`ContextEvent`.

    The returned event carries ``pot_id=""``; the inbound dispatcher fills
    in the pot id from its source-to-pot mapping before submitting.
    """
    if not isinstance(body, dict):
        raise LinearWebhookError("linear webhook body must be an object")
    ev_type = str(body.get("type") or "").strip()
    raw_action = str(body.get("action") or "").strip().lower()
    data = body.get("data")
    if not isinstance(data, dict):
        raise LinearWebhookError("linear webhook body missing 'data' object")

    if ev_type == "Issue":
        return _issue_event(body, data, raw_action)
    if ev_type == "Comment":
        return _comment_event(body, data, raw_action)
    raise LinearWebhookError(f"unsupported linear webhook type: {ev_type!r}")


def _issue_event(
    body: dict[str, Any],
    data: dict[str, Any],
    raw_action: str,
) -> ContextEvent:
    identifier = str(data.get("identifier") or "").strip()
    issue_id = str(data.get("id") or "").strip()
    if not identifier and not issue_id:
        raise LinearWebhookError("linear issue webhook missing id/identifier")

    updated_from = body.get("updatedFrom")
    action = raw_action or "update"
    previous_state = None
    new_state_id: str | None = None
    if (
        action == "update"
        and isinstance(updated_from, dict)
        and updated_from.get("stateId")
    ):
        action = "state_change"
        previous_state = {
            "name": str(updated_from.get("stateName") or "previous"),
            "type": updated_from.get("stateType"),
        }
        current_state = (
            data.get("state") if isinstance(data.get("state"), dict) else None
        )
        new_state_id = (
            str(current_state.get("id"))
            if current_state and current_state.get("id")
            else None
        )

    team_id = _team_id(data)
    payload: dict[str, Any] = {
        "action": action,
        "issue": data,
    }
    if previous_state is not None:
        payload["previous_state"] = previous_state

    source_id = _issue_source_id(identifier or issue_id, action, new_state_id)
    occurred_at = parse_linear_datetime(body.get("createdAt") or data.get("updatedAt"))
    return ContextEvent(
        event_id=str(uuid4()),
        source_system="linear",
        event_type="linear_issue",
        action=action,
        pot_id="",
        provider="linear",
        provider_host="linear.app",
        repo_name=f"team:{team_id}" if team_id else "",
        source_id=source_id,
        source_event_id=str(body.get("webhookId") or body.get("id") or "") or None,
        occurred_at=occurred_at,
        payload=payload,
    )


def _comment_event(
    body: dict[str, Any],
    data: dict[str, Any],
    raw_action: str,
) -> ContextEvent:
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
    return ContextEvent(
        event_id=str(uuid4()),
        source_system="linear",
        event_type="linear_comment",
        action="comment_added",
        pot_id="",
        provider="linear",
        provider_host="linear.app",
        repo_name=f"team:{team_id}" if team_id else "",
        source_id=source_id,
        source_event_id=str(body.get("webhookId") or body.get("id") or comment_id)
        or None,
        occurred_at=occurred_at,
        payload=payload,
    )


def _issue_source_id(identifier: str, action: str, state_id: str | None = None) -> str:
    if action == "state_change" and state_id:
        return f"linear:issue:{identifier}:state_change:{state_id}"
    return f"linear:issue:{identifier}:{action}"


def _team_id(data: dict[str, Any]) -> str | None:
    if data.get("teamId"):
        return str(data["teamId"])
    team = data.get("team")
    if isinstance(team, dict) and team.get("id"):
        return str(team["id"])
    return None
