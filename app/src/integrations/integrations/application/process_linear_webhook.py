"""Linear webhook routing use case.

This is the pure orchestration that turns a Linear webhook payload into
per-pot ingestion submissions. It has no I/O of its own beyond the ports
it consumes (a SQLAlchemy session + the context-engine container).

Flow:

1. Parse envelope, pull ``organizationId``.
2. Canonicalize the payload via :class:`LinearConnector.normalize_webhook`
   so the resulting ``ContextEvent`` matches what other code expects.
3. Find every active Linear integration row keyed to that org, and the
   ``project_sources`` rows linking those integrations to pots.
4. Filter to rows whose ``scope_json.team_id`` matches the event's team
   (a pot only gets events for teams it explicitly subscribed to).
5. Submit one ingestion event per (pot, integration). Failures are
   logged per-pot but do not abort the rest of the fan-out — Linear
   retries the whole webhook on a non-2xx response, so partial success
   should still return 2xx with details in the response body.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_source_model import (
    SOURCE_KIND_ISSUE_TRACKER_TEAM,
    ContextGraphPotSource,
)
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.exceptions import (
    WebhookPayloadError,
    WebhookProcessingError,
)
from integrations.domain.integrations_schema import IntegrationType

logger = logging.getLogger(__name__)


def process_linear_webhook(
    db: Session,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Route a Linear webhook payload to every subscribed pot.

    Returns a result dict the caller can serialize back to the HTTP
    response. The function commits the SQLAlchemy session itself so the
    Celery worker doesn't have to manage transaction boundaries.

    Raises:
        WebhookPayloadError: ``organizationId`` missing or envelope
            shape is unrecognized.
        WebhookProcessingError: a registered connector / container
            invariant is broken (programming error, not user error).
    """
    if not isinstance(payload, dict):
        raise WebhookPayloadError("Linear webhook payload must be an object")
    org_id = payload.get("organizationId")
    if not org_id or not isinstance(org_id, str):
        raise WebhookPayloadError("Linear webhook missing 'organizationId'")

    container = _build_container(db)
    if not container.settings.is_enabled():
        logger.info("linear webhook ignored: context-engine disabled (org=%s)", org_id)
        return {
            "status": "ignored",
            "reason": "context_engine_disabled",
            "organization_id": org_id,
        }

    connector = container.connectors.find_for_webhook("linear")
    if connector is None:
        raise WebhookProcessingError("linear connector not registered in container")

    # Re-serialize so we can reuse the connector's canonicalization
    # (which parses bytes). Payload is small; double-parse cost is
    # negligible and the layering stays clean.
    body_bytes = json.dumps(payload).encode("utf-8")
    try:
        event = connector.normalize_webhook(body_bytes, headers or {})
    except PermissionError:
        # Connector enforces signature itself when given a secret. We
        # verify at the HTTP layer too, so this is belt-and-braces.
        raise
    except Exception as exc:
        raise WebhookProcessingError(
            f"linear connector failed to normalize webhook for org {org_id}"
        ) from exc

    if event is None:
        logger.info(
            "linear webhook event ignored by connector (org=%s, type=%s)",
            org_id,
            payload.get("type"),
        )
        return {
            "status": "ignored",
            "reason": "ignored_event_type",
            "organization_id": org_id,
        }

    team_id = _team_id_from_event_repo_name(event.repo_name)
    targets = _find_linear_routing_targets(db, org_id=org_id, team_id=team_id)
    if not targets:
        logger.info(
            "linear webhook: no subscribed pots (org=%s, team=%s)",
            org_id,
            team_id,
        )
        return {
            "status": "ignored",
            "reason": "no_subscribed_pots",
            "organization_id": org_id,
            "team_id": team_id,
        }

    submission = container.ingestion_submission(db)
    routed: list[dict[str, Any]] = []
    for pot_id, integration_id in targets:
        try:
            request = _build_ingestion_request(
                pot_id=pot_id,
                integration_id=integration_id,
                event=event,
            )
            receipt = submission.submit(request)
            routed.append(
                {
                    "pot_id": pot_id,
                    "integration_id": integration_id,
                    "event_id": receipt.event_id,
                    "status": receipt.status,
                    "duplicate": receipt.duplicate,
                }
            )
        except Exception as exc:
            logger.exception(
                "linear webhook routing failed for pot=%s integration=%s: %s",
                pot_id,
                integration_id,
                exc,
            )
            routed.append(
                {
                    "pot_id": pot_id,
                    "integration_id": integration_id,
                    "error": "submission_failed",
                }
            )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    successes = sum(1 for r in routed if "event_id" in r)
    failures = len(routed) - successes
    logger.info(
        "linear webhook routed org=%s team=%s success=%d failed=%d",
        org_id,
        team_id,
        successes,
        failures,
    )
    return {
        "status": "routed",
        "organization_id": org_id,
        "team_id": team_id,
        "event_id": event.event_id,
        "routed": routed,
        "successes": successes,
        "failures": failures,
    }


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


def _build_container(db: Session):
    """Local import so unit tests can fake the container without DB."""
    from app.modules.context_graph.wiring import build_container_for_session

    return build_container_for_session(db)


def _team_id_from_event_repo_name(repo_name: str | None) -> str | None:
    """``LinearConnector.normalize_webhook`` packs the team into
    ``repo_name='team:<id>'``. Pull it back out for routing."""
    if not repo_name:
        return None
    prefix = "team:"
    return repo_name[len(prefix) :] if repo_name.startswith(prefix) else None


def _find_linear_routing_targets(
    db: Session,
    *,
    org_id: str,
    team_id: str | None,
) -> list[tuple[str, str]]:
    """Return ``[(pot_id, integration_id)]`` for every subscribed pot.

    A pot is "subscribed" to an event when it has a
    ``context_graph_pot_sources`` row whose ``integration_id`` points to a
    Linear integration for ``org_id`` AND whose ``scope_json.team_id``
    matches the event's team. When ``team_id`` is ``None`` (e.g. comment
    without a team context — shouldn't happen but we handle it), all pots
    subscribed to any team of this integration get the event.
    """
    integrations = (
        db.query(Integration)
        .filter(
            Integration.integration_type == IntegrationType.LINEAR.value,
            Integration.active.is_(True),
            Integration.unique_identifier == org_id,
        )
        .all()
    )
    if not integrations:
        return []
    integration_ids = [i.integration_id for i in integrations]
    rows = (
        db.query(ContextGraphPotSource)
        .filter(
            ContextGraphPotSource.provider == "linear",
            ContextGraphPotSource.source_kind == SOURCE_KIND_ISSUE_TRACKER_TEAM,
            ContextGraphPotSource.integration_id.in_(integration_ids),
            ContextGraphPotSource.sync_enabled.is_(True),
        )
        .all()
    )
    seen: set[tuple[str, str]] = set()
    targets: list[tuple[str, str]] = []
    for row in rows:
        if team_id:
            scope = _scope_dict(row.scope_json)
            row_team = scope.get("team_id")
            if str(row_team) != str(team_id):
                continue
        key = (str(row.pot_id), str(row.integration_id))
        if key in seen:
            continue
        seen.add(key)
        targets.append(key)
    return targets


def _scope_dict(raw: Any) -> dict[str, Any]:
    """``ContextGraphPotSource.scope_json`` is TEXT; tolerate dict too for tests."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_ingestion_request(
    *,
    pot_id: str,
    integration_id: str,
    event,
):
    """Convert a canonicalized ``ContextEvent`` into a submission request.

    Linear events are not tied to a repo within the pot. The connector packs
    the team id into ``event.repo_name='team:<id>'`` purely for routing in
    :func:`_find_linear_routing_targets`; the ingestion service rejects that
    value as ``repo_not_in_pot`` because the pot's repos are GitHub repos.
    We therefore omit repo/provider on the request and let the submission
    service derive scope from the pot's primary repo — the team id remains
    in ``payload['issue']['team']`` for downstream consumers.
    """
    from domain.ingestion_event_models import IngestionSubmissionRequest
    from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION

    enriched = dict(event.payload)
    enriched.setdefault("integration_id", integration_id)
    return IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="webhook",
        source_system=event.source_system,
        event_type=event.event_type,
        action=event.action,
        payload=enriched,
        source_id=event.source_id,
        source_event_id=event.source_event_id,
    )
