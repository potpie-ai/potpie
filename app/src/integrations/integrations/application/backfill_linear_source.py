"""Linear team backfill — one-shot, idempotent, runs at pot-attach.

Replaces the older cron-style ``linear_sync`` that wrote directly into
the legacy ledger. The new flow enumerates Linear issues for the team,
synthesises a ``Issue / create`` webhook envelope per issue, and feeds
them through the same path live webhooks use:
``LinearConnector.normalize_webhook → IngestionSubmissionService.submit``.

The connector's source_id is stable per ``(identifier, action)``, so
re-running the backfill is a no-op — duplicate events are dropped at
the ingestion ledger.

Inputs come from the existing ``LinearIssueTrackerAdapter``, which
itself uses the integration's stored access token. Failures per issue
are logged but do not abort the rest; the result records
(``ingested``, ``failed``) for observability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_source_model import (
    SOURCE_KIND_ISSUE_TRACKER_TEAM,
    ContextGraphPotSource,
)
from app.modules.context_graph.pot_sources_service import touch_pot_source_sync
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
from integrations.adapters.outbound.linear.adapter import LinearIssueTrackerAdapter
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import AuthData, IntegrationType

logger = logging.getLogger(__name__)


# A safety bound — backfilling a 10k-issue team in one job would block
# the worker and flood the ledger. The webhook path keeps the graph
# fresh; backfill just needs to seed history.
DEFAULT_MAX_ISSUES = 200


@dataclass
class BackfillResult:
    pot_source_id: str
    status: str  # "success", "partial_success", "error", "skipped"
    pot_id: str | None = None
    organization_id: str | None = None
    team_id: str | None = None
    ingested: int = 0
    failed: int = 0
    reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pot_source_id": self.pot_source_id,
            "status": self.status,
            "pot_id": self.pot_id,
            "organization_id": self.organization_id,
            "team_id": self.team_id,
            "ingested": self.ingested,
            "failed": self.failed,
            "reason": self.reason,
            "errors": self.errors[:10],  # cap to keep response small
        }


def backfill_linear_source(
    db: Session,
    pot_source_id: str,
    *,
    max_issues: int = DEFAULT_MAX_ISSUES,
) -> dict[str, Any]:
    """One-shot backfill for a ``context_graph_pot_sources`` row of provider=linear.

    Idempotent: stable ``source_id`` per issue means re-running this
    against the same row deduplicates at the ingestion ledger.

    Returns a dict (not a dataclass) for Celery-serialisability.
    """
    result = BackfillResult(pot_source_id=pot_source_id, status="skipped")

    row = (
        db.query(ContextGraphPotSource)
        .filter(ContextGraphPotSource.id == pot_source_id)
        .first()
    )
    if (
        row is None
        or row.provider != "linear"
        or row.source_kind != SOURCE_KIND_ISSUE_TRACKER_TEAM
    ):
        result.reason = "not_found_or_not_linear"
        return result.to_dict()

    result.pot_id = str(row.pot_id)
    scope = _scope_dict(row.scope_json)
    team_id = scope.get("team_id")
    result.team_id = str(team_id) if team_id else None
    if not team_id:
        result.status = "error"
        result.reason = "missing_team_id"
        touch_pot_source_sync(db, pot_source_id, error="missing_team_id")
        return result.to_dict()

    if not row.integration_id:
        result.status = "error"
        result.reason = "missing_integration_id"
        touch_pot_source_sync(db, pot_source_id, error="missing_integration_id")
        return result.to_dict()

    integration = (
        db.query(Integration)
        .filter(
            Integration.integration_id == row.integration_id,
            Integration.integration_type == IntegrationType.LINEAR.value,
            Integration.active.is_(True),
        )
        .first()
    )
    if integration is None:
        result.status = "error"
        result.reason = "integration_not_found"
        touch_pot_source_sync(db, pot_source_id, error="integration_not_found")
        return result.to_dict()

    org_id = (integration.scope_data or {}).get("org_id") or integration.unique_identifier
    result.organization_id = str(org_id) if org_id else None

    access_token = _resolve_access_token(integration)
    if not access_token:
        result.status = "error"
        result.reason = "missing_access_token"
        touch_pot_source_sync(db, pot_source_id, error="missing_access_token")
        return result.to_dict()

    container = _build_container(db)
    if not container.settings.is_enabled():
        result.status = "skipped"
        result.reason = "context_engine_disabled"
        return result.to_dict()
    connector = container.connectors.find_for_webhook("linear")
    if connector is None:
        result.status = "error"
        result.reason = "linear_connector_not_registered"
        return result.to_dict()
    submission = container.ingestion_submission(db)

    adapter = LinearIssueTrackerAdapter(access_token)
    adapter_scope = {"team_id": str(team_id)}
    try:
        for ref in adapter.iter_issues(scope=adapter_scope, updated_after=None):
            if result.ingested >= max_issues:
                break
            try:
                issue = adapter.get_issue(scope=adapter_scope, issue_id=ref.id)
                _submit_issue_snapshot(
                    submission=submission,
                    connector=connector,
                    pot_id=str(row.pot_id),
                    integration_id=str(integration.integration_id),
                    organization_id=str(org_id) if org_id else "",
                    issue=issue,
                )
                result.ingested += 1
            except Exception as exc:
                result.failed += 1
                msg = f"issue {ref.identifier or ref.id}: {exc}"
                result.errors.append(msg)
                logger.exception("linear backfill issue failed: %s", msg)
    except Exception as exc:
        result.status = "error"
        result.reason = str(exc)[:300]
        logger.exception("linear backfill enumeration failed for %s", pot_source_id)
        touch_pot_source_sync(db, pot_source_id, error=result.reason)
        try:
            db.commit()
        except Exception:
            db.rollback()
        return result.to_dict()

    err_msg = (
        f"{result.failed} of {result.ingested + result.failed} issues failed"
        if result.failed
        else None
    )
    touch_pot_source_sync(db, pot_source_id, error=err_msg)
    try:
        db.commit()
    except Exception:
        db.rollback()
        result.status = "error"
        result.reason = "db_commit_failed"
        return result.to_dict()

    result.status = "success" if result.failed == 0 else "partial_success"
    return result.to_dict()


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


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


def _resolve_access_token(integration: Integration) -> str | None:
    try:
        auth = AuthData.model_validate(integration.auth_data or {})
    except Exception:
        logger.warning(
            "linear backfill: invalid auth_data on integration %s",
            integration.integration_id,
        )
        return None
    if not auth.access_token:
        return None
    try:
        return decrypt_token(auth.access_token)
    except Exception:
        logger.warning(
            "linear backfill: token decrypt failed for integration %s",
            integration.integration_id,
        )
        return None


def _submit_issue_snapshot(
    *,
    submission,
    connector,
    pot_id: str,
    integration_id: str,
    organization_id: str,
    issue: dict[str, Any],
) -> None:
    """Submit one Linear issue through the same path as live webhooks.

    We synthesise a ``Issue / create`` webhook envelope so the connector
    runs the exact normalization the live path uses. The synthetic
    source_id collides with a live ``create`` event if one ever fires
    for the same issue — which is the intent: the live event wins
    naturally, and the duplicate is a no-op.
    """
    from domain.ingestion_event_models import IngestionSubmissionRequest
    from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION

    envelope = {
        "type": "Issue",
        "action": "create",
        "organizationId": organization_id,
        "data": issue,
    }
    body = json.dumps(envelope).encode("utf-8")
    event = connector.normalize_webhook(body, {})
    if event is None:
        raise RuntimeError(
            f"connector did not normalize issue {issue.get('identifier')!r}"
        )
    enriched = dict(event.payload)
    enriched.setdefault("integration_id", integration_id)
    enriched.setdefault("is_backfill", True)
    # See note in process_linear_webhook._build_ingestion_request: Linear
    # events are not tied to a repo in the pot, so we omit repo/provider
    # and let the submission service derive scope from the pot's primary.
    request = IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="backfill",
        source_system=event.source_system,
        event_type=event.event_type,
        action=event.action,
        payload=enriched,
        source_id=event.source_id,
        source_event_id=event.source_event_id,
    )
    submission.submit(request)


def _build_container(db: Session):
    """Local import so unit tests can fake the container without DB."""
    from app.modules.context_graph.wiring import build_container_for_session

    return build_container_for_session(db)
