"""CRUD and helpers for project_sources."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import IntegrationType
from integrations.adapters.outbound.postgres.project_source_model import ProjectSource
from app.modules.projects.projects_model import Project

logger = logging.getLogger(__name__)


def compute_scope_hash(scope: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def ensure_github_repository_source(
    db: Session, project_id: str, repo_name: str
) -> ProjectSource | None:
    """Idempotent: ensure a github/repository row exists for this project."""
    if not repo_name or not str(repo_name).strip():
        return None
    scope = {"repo_name": str(repo_name).strip()}
    h = compute_scope_hash(scope)
    existing = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.scope_hash == h,
        )
        .first()
    )
    if existing:
        return existing
    row = ProjectSource(
        id=str(uuid.uuid4()),
        project_id=project_id,
        integration_id=None,
        provider="github",
        source_kind="repository",
        scope_json=scope,
        scope_hash=h,
        sync_enabled=True,
        sync_mode="hybrid",
        webhook_status="not_applicable",
        health_score=100,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return (
            db.query(ProjectSource)
            .filter(ProjectSource.project_id == project_id, ProjectSource.scope_hash == h)
            .first()
        )
    db.refresh(row)
    return row


def list_github_repository_sources(db: Session, project_id: str) -> list[ProjectSource]:
    return (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.provider == "github",
            ProjectSource.source_kind == "repository",
        )
        .all()
    )


def list_linear_team_sources(db: Session, project_id: str) -> list[ProjectSource]:
    return (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.provider == "linear",
            ProjectSource.source_kind == "issue_tracker_team",
        )
        .all()
    )


def attach_linear_team_source(
    db: Session,
    *,
    project_id: str,
    integration_id: str,
    team_id: str,
    team_name: str | None = None,
    user_id: str,
) -> ProjectSource:
    normalized_team_id = str(team_id).strip()
    normalized_team_name = str(team_name).strip() if team_name else None
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user_id)
        .first()
    )
    if not project:
        raise ValueError("project_not_found_or_forbidden")
    integ = (
        db.query(Integration)
        .filter(
            Integration.integration_id == integration_id,
            Integration.created_by == user_id,
            Integration.integration_type == IntegrationType.LINEAR.value,
            Integration.active.is_(True),
        )
        .first()
    )
    if not integ:
        raise ValueError("integration_not_found_or_forbidden")
    if not normalized_team_id:
        raise ValueError("team_id_is_empty")
    scope: dict[str, Any] = {"team_id": normalized_team_id}
    if normalized_team_name:
        scope["team_name"] = normalized_team_name
    # Dedupe on the actual provider scope, not mutable display metadata.
    h = compute_scope_hash({"team_id": normalized_team_id})
    existing = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.scope_hash == h,
        )
        .first()
    )
    if existing:
        return existing
    row = ProjectSource(
        id=str(uuid.uuid4()),
        project_id=project_id,
        integration_id=integration_id,
        provider="linear",
        source_kind="issue_tracker_team",
        scope_json=scope,
        scope_hash=h,
        sync_enabled=True,
        sync_mode="hybrid",
        webhook_status="pending_setup",
        health_score=100,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return (
            db.query(ProjectSource)
            .filter(ProjectSource.project_id == project_id, ProjectSource.scope_hash == h)
            .first()
        )
    db.refresh(row)
    return row


def get_project_source(
    db: Session, source_id: str, user_id: str
) -> ProjectSource | None:
    row = (
        db.query(ProjectSource)
        .join(Project, Project.id == ProjectSource.project_id)
        .filter(ProjectSource.id == source_id, Project.user_id == user_id)
        .first()
    )
    return row


def list_all_sources_for_project(
    db: Session, project_id: str, user_id: str
) -> list[ProjectSource]:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user_id)
        .first()
    )
    if not project:
        return []
    return (
        db.query(ProjectSource)
        .filter(ProjectSource.project_id == project_id)
        .order_by(ProjectSource.created_at)
        .all()
    )


def delete_project_source(db: Session, source_id: str, user_id: str) -> bool:
    row = get_project_source(db, source_id, user_id)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True


def touch_source_sync(
    db: Session,
    source_id: str,
    *,
    error: str | None = None,
) -> None:
    row = db.query(ProjectSource).filter(ProjectSource.id == source_id).first()
    if not row:
        return
    row.last_sync_at = datetime.now(timezone.utc)
    row.last_error = error
    if error:
        row.health_score = max(0, (row.health_score or 100) - 10)
    else:
        row.health_score = min(100, (row.health_score or 0) + 5)
    db.commit()
