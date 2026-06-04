"""Backfill / sync Linear issues for a project_sources row (Celery worker)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from domain.ports.ingestion_ledger import LedgerScope

from app.modules.context_graph.wiring import build_container_for_session
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import AuthData, IntegrationType
from integrations.adapters.outbound.linear.adapter import LinearIssueTrackerAdapter
from integrations.adapters.outbound.linear.ingest_linear_issue import ingest_linear_issue
from integrations.adapters.outbound.postgres.project_source_model import ProjectSource
from integrations.application.project_sources_service import touch_source_sync
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token

logger = logging.getLogger(__name__)

MAX_ISSUES_PER_RUN = 80
SOURCE_TYPE_SYNC = "linear_issue"


def _linear_ledger_scope(pot_id: str, team_id: str) -> LedgerScope:
    return LedgerScope(
        pot_id=pot_id,
        provider="linear",
        provider_host="linear.app",
        repo_name=f"team:{team_id}",
    )


def sync_linear_project_source(db: Session, project_source_id: str) -> dict[str, Any]:
    row = db.query(ProjectSource).filter(ProjectSource.id == project_source_id).first()
    if not row or row.provider != "linear":
        return {
            "status": "skipped",
            "reason": "not_found_or_not_linear",
            "project_source_id": project_source_id,
        }
    if not row.integration_id:
        touch_source_sync(db, project_source_id, error="missing_integration_id")
        return {
            "status": "error",
            "reason": "missing_integration_id",
            "project_source_id": project_source_id,
        }
    integration = (
        db.query(Integration)
        .filter(Integration.integration_id == row.integration_id)
        .first()
    )
    if (
        not integration
        or integration.integration_type != IntegrationType.LINEAR.value
        or not integration.active
    ):
        touch_source_sync(db, project_source_id, error="integration_not_found")
        return {
            "status": "error",
            "reason": "integration_not_found",
            "project_source_id": project_source_id,
        }

    auth_raw = integration.auth_data or {}
    try:
        auth = AuthData.model_validate(auth_raw)
    except Exception:
        touch_source_sync(db, project_source_id, error="invalid_auth_data")
        return {
            "status": "error",
            "reason": "invalid_auth_data",
            "project_source_id": project_source_id,
        }
    enc = auth.access_token
    if not enc:
        touch_source_sync(db, project_source_id, error="missing_access_token")
        return {
            "status": "error",
            "reason": "missing_access_token",
            "project_source_id": project_source_id,
        }
    try:
        access_token = decrypt_token(enc)
    except Exception:
        logger.exception("Linear token decrypt failed")
        touch_source_sync(db, project_source_id, error="token_decrypt_failed")
        return {
            "status": "error",
            "reason": "token_decrypt_failed",
            "project_source_id": project_source_id,
        }

    scope_json = row.scope_json or {}
    team_id = scope_json.get("team_id")
    if not team_id:
        touch_source_sync(db, project_source_id, error="missing_team_id")
        return {
            "status": "error",
            "reason": "missing_team_id",
            "project_source_id": project_source_id,
        }

    container = build_container_for_session(db)
    if not container.settings.is_enabled():
        return {
            "status": "skipped",
            "reason": "context_graph_disabled",
            "project_source_id": project_source_id,
        }

    ledger = container.ledger(db)
    episodic = container.episodic
    lscope = _linear_ledger_scope(row.project_id, str(team_id))
    sync = ledger.get_or_create_sync_state(lscope, SOURCE_TYPE_SYNC)
    cursor = sync.last_synced_at
    if cursor and cursor.tzinfo is None:
        cursor = cursor.replace(tzinfo=timezone.utc)

    ledger.update_sync_state_running(lscope, SOURCE_TYPE_SYNC)

    adapter = LinearIssueTrackerAdapter(access_token)
    issue_scope = {"team_id": str(team_id)}
    ingested = 0
    failed = 0
    latest: datetime | None = cursor

    try:
        for ref in adapter.iter_issues(scope=issue_scope, updated_after=cursor):
            if ingested >= MAX_ISSUES_PER_RUN:
                break
            try:
                issue = adapter.get_issue(scope=issue_scope, issue_id=ref.id)
                cwrap = issue.get("comments") if isinstance(issue, dict) else None
                comments = (
                    cwrap.get("nodes", [])
                    if isinstance(cwrap, dict)
                    else []
                )
                comments = [c for c in comments if isinstance(c, dict)]
                ingest_linear_issue(
                    ledger=ledger,
                    episodic=episodic,
                    scope=lscope,
                    issue=issue,
                    comments=comments,
                )
                ingested += 1
                if ref.updated_at:
                    if latest is None or ref.updated_at > latest:
                        latest = ref.updated_at
            except Exception:
                failed += 1
                logger.exception(
                    "Linear issue ingest failed source=%s issue=%s",
                    project_source_id,
                    ref.id,
                )

        ledger.update_sync_state_success(lscope, SOURCE_TYPE_SYNC, latest)
        err_msg = f"{failed} issue(s) failed to ingest" if failed else None
        touch_source_sync(db, project_source_id, error=err_msg)
        status = "success" if failed == 0 else "partial_success"
        return {
            "status": status,
            "project_source_id": project_source_id,
            "pot_id": row.project_id,
            "ingested": ingested,
            "failed": failed,
            "last_synced_at": latest.isoformat() if latest else None,
        }
    except Exception as exc:
        ledger.update_sync_state_error(lscope, SOURCE_TYPE_SYNC, str(exc))
        touch_source_sync(db, project_source_id, error=str(exc))
        logger.exception("Linear sync failed for source %s", project_source_id)
        return {
            "status": "error",
            "project_source_id": project_source_id,
            "error": str(exc),
        }
