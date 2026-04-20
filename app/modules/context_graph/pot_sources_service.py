"""Helpers to mirror pot repositories into ``context_graph_pot_sources``.

Kept deliberately small: the source table is new and we only need two flows
today — create/delete mirror for a GitHub repo. Other source kinds can be
added as dedicated helpers when the routes for them land.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.context_graph_pot_source_model import (
    ContextGraphPotSource,
    SOURCE_KIND_ISSUE_TRACKER_TEAM,
    SOURCE_KIND_REPOSITORY,
)


def github_repo_scope_hash(provider: str, provider_host: str, owner: str, repo: str) -> str:
    raw = f"{provider}|{provider_host}|{owner.lower()}|{repo.lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def github_repo_scope_json(row: ContextGraphPotRepository) -> str:
    scope: dict[str, Any] = {
        "owner": row.owner,
        "repo": row.repo,
        "repo_name": f"{row.owner}/{row.repo}",
        "provider_host": row.provider_host,
        "external_repo_id": row.external_repo_id,
        "remote_url": row.remote_url,
        "default_branch": row.default_branch,
    }
    return json.dumps(scope, sort_keys=True)


def mirror_repository_into_sources(
    db: Session,
    repository: ContextGraphPotRepository,
    *,
    added_by_user_id: str,
) -> ContextGraphPotSource:
    """Create the matching source row; returns the existing row if present."""
    scope_hash = github_repo_scope_hash(
        repository.provider, repository.provider_host, repository.owner, repository.repo
    )
    existing = (
        db.query(ContextGraphPotSource)
        .filter(
            ContextGraphPotSource.pot_id == repository.pot_id,
            ContextGraphPotSource.provider == repository.provider,
            ContextGraphPotSource.source_kind == SOURCE_KIND_REPOSITORY,
            ContextGraphPotSource.scope_hash == scope_hash,
        )
        .first()
    )
    if existing is not None:
        return existing

    row = ContextGraphPotSource(
        id=str(uuid.uuid4()),
        pot_id=repository.pot_id,
        integration_id=None,
        provider=repository.provider,
        source_kind=SOURCE_KIND_REPOSITORY,
        scope_json=github_repo_scope_json(repository),
        scope_hash=scope_hash,
        sync_enabled=True,
        added_by_user_id=added_by_user_id,
    )
    db.add(row)
    db.flush()
    return row


def unmirror_repository_from_sources(
    db: Session, repository: ContextGraphPotRepository
) -> None:
    scope_hash = github_repo_scope_hash(
        repository.provider, repository.provider_host, repository.owner, repository.repo
    )
    db.query(ContextGraphPotSource).filter(
        ContextGraphPotSource.pot_id == repository.pot_id,
        ContextGraphPotSource.provider == repository.provider,
        ContextGraphPotSource.source_kind == SOURCE_KIND_REPOSITORY,
        ContextGraphPotSource.scope_hash == scope_hash,
    ).delete(synchronize_session=False)


def linear_team_scope_hash(team_id: str) -> str:
    raw = f"linear|team|{team_id.strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def attach_linear_team_source(
    db: Session,
    *,
    pot_id: str,
    integration_id: str,
    team_id: str,
    team_name: str | None,
    added_by_user_id: str,
) -> tuple[ContextGraphPotSource, bool]:
    """Create a linear/issue_tracker_team source row. Returns (row, already_attached)."""
    team_id_clean = team_id.strip()
    if not team_id_clean:
        raise ValueError("team_id is required")
    scope_hash = linear_team_scope_hash(team_id_clean)
    existing = (
        db.query(ContextGraphPotSource)
        .filter(
            ContextGraphPotSource.pot_id == pot_id,
            ContextGraphPotSource.provider == "linear",
            ContextGraphPotSource.source_kind == SOURCE_KIND_ISSUE_TRACKER_TEAM,
            ContextGraphPotSource.scope_hash == scope_hash,
        )
        .first()
    )
    if existing is not None:
        return existing, True

    scope: dict[str, Any] = {"team_id": team_id_clean}
    if team_name:
        scope["team_name"] = team_name.strip()
    row = ContextGraphPotSource(
        id=str(uuid.uuid4()),
        pot_id=pot_id,
        integration_id=integration_id,
        provider="linear",
        source_kind=SOURCE_KIND_ISSUE_TRACKER_TEAM,
        scope_json=json.dumps(scope, sort_keys=True),
        scope_hash=scope_hash,
        sync_enabled=True,
        added_by_user_id=added_by_user_id,
    )
    db.add(row)
    db.flush()
    return row, False


def serialize_source(row: ContextGraphPotSource) -> dict[str, Any]:
    try:
        scope = json.loads(row.scope_json) if row.scope_json else {}
    except (TypeError, ValueError):
        scope = {}
    return {
        "id": row.id,
        "pot_id": row.pot_id,
        "integration_id": row.integration_id,
        "provider": row.provider,
        "source_kind": row.source_kind,
        "scope": scope,
        "sync_enabled": bool(row.sync_enabled),
        "sync_mode": row.sync_mode,
        "webhook_status": row.webhook_status,
        "last_sync_at": row.last_sync_at.isoformat() if row.last_sync_at else None,
        "last_error": row.last_error,
        "health_score": row.health_score,
        "added_by_user_id": row.added_by_user_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
