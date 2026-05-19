"""Helpers to mirror pot repositories into ``context_graph_pot_sources``.

Kept deliberately small: the source table is new and we only need two flows
today — create/delete mirror for a GitHub repo. Other source kinds can be
added as dedicated helpers when the routes for them land.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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


def repository_for_source(
    db: Session, source: ContextGraphPotSource
) -> ContextGraphPotRepository | None:
    """Resolve the repo row a repository source mirrors — inverse of mirror.

    Matches on the stable ``scope_hash`` join key (computed at mirror time
    from provider/host/owner/repo and stored on the source row), not by
    re-parsing ``scope_json``. A corrupt or legacy ``scope_json`` therefore
    can no longer orphan the repo row when the source is deleted. Returns
    ``None`` for non-repository sources or when no repo row matches.
    """
    if source.source_kind != SOURCE_KIND_REPOSITORY:
        return None
    candidates = (
        db.query(ContextGraphPotRepository)
        .filter(
            ContextGraphPotRepository.pot_id == source.pot_id,
            ContextGraphPotRepository.provider == source.provider,
        )
        .all()
    )
    for repo_row in candidates:
        if (
            github_repo_scope_hash(
                repo_row.provider,
                repo_row.provider_host,
                repo_row.owner,
                repo_row.repo,
            )
            == source.scope_hash
        ):
            return repo_row
    return None


def repository_source_exists(
    db: Session, repository: ContextGraphPotRepository
) -> bool:
    """True when the mirrored source row for ``repository`` is present.

    A live attachment always has its source mirror (``attach_repo_to_pot``
    creates both together). Its absence means the repo was deleted via the
    Sources tab — ``delete_pot_source`` always drops the source row but only
    best-effort drops the repo row, so the surviving repo row is then an
    orphan, not a live attachment.
    """
    scope_hash = github_repo_scope_hash(
        repository.provider, repository.provider_host, repository.owner, repository.repo
    )
    return (
        db.query(ContextGraphPotSource)
        .filter(
            ContextGraphPotSource.pot_id == repository.pot_id,
            ContextGraphPotSource.provider == repository.provider,
            ContextGraphPotSource.source_kind == SOURCE_KIND_REPOSITORY,
            ContextGraphPotSource.scope_hash == scope_hash,
        )
        .first()
        is not None
    )


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


def emit_linear_backfill_event(
    db: Session,
    *,
    row: ContextGraphPotSource,
    submitted_by_user_id: str,
) -> str | None:
    """Submit the ``linear_team.added`` backfill seed for a newly attached team.

    The Linear analogue of ``attach_repo_to_pot._emit_bootstrap_event``: one
    ``agent_reconciliation`` event flows the same admission path as GitHub's
    ``repository.added`` and every live webhook, so the reconciliation agent
    (planner on, via the ``linear/linear_team/added`` playbook) enumerates the
    team's issues and seeds them. Idempotent on ``source_id`` so re-attaching
    the same team is a no-op at the ingestion ledger.

    Best-effort: returns ``None`` (and logs) if the context-engine isn't
    importable, the container can't be built, or submit raises. The source row
    is already committed by the caller, so attach stays successful regardless.
    """
    try:
        from app.modules.context_graph.wiring import (
            build_container_for_user_session,
        )
        from domain.ingestion_event_models import IngestionSubmissionRequest
        from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
    except Exception:
        logger.exception(
            "emit_linear_backfill_event: context-engine import failed; "
            "skipping backfill seed"
        )
        return None

    try:
        scope = json.loads(row.scope_json) if row.scope_json else {}
    except (TypeError, ValueError):
        scope = {}
    team_id = str((scope or {}).get("team_id") or "").strip()
    if not team_id:
        logger.warning(
            "emit_linear_backfill_event: no team_id in scope for source=%s",
            row.id,
        )
        return None
    team_name = (scope or {}).get("team_name")

    try:
        container = build_container_for_user_session(db, submitted_by_user_id)
    except Exception:
        logger.exception(
            "emit_linear_backfill_event: container build failed for pot=%s",
            row.pot_id,
        )
        return None

    request = IngestionSubmissionRequest(
        pot_id=row.pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="source_attach",
        source_system="linear",
        event_type="linear_team",
        action="added",
        source_id=f"linear_team_added:{team_id}",
        provider="linear",
        provider_host="linear.app",
        payload={
            "team_id": team_id,
            "team_name": team_name,
            "integration_id": row.integration_id,
            "pot_source_id": row.id,
            "submitted_by_user_id": submitted_by_user_id,
        },
    )
    try:
        receipt = container.ingestion_submission(db).submit(
            request, sync=False, wait=False
        )
    except Exception:
        logger.exception(
            "emit_linear_backfill_event: submit failed for pot=%s team=%s",
            row.pot_id,
            team_id,
        )
        return None
    logger.info(
        "emit_linear_backfill_event: enqueued event=%s pot=%s team=%s",
        receipt.event_id,
        row.pot_id,
        team_id,
    )
    return receipt.event_id


def touch_pot_source_sync(
    db: Session,
    source_id: str,
    *,
    error: str | None = None,
) -> None:
    """Stamp ``last_sync_at`` / ``last_error`` / ``health_score`` after a sync run.

    ``health_score`` is stored as TEXT in this table for forward compat; we
    parse it as int, adjust, and write it back as a string.
    """
    row = (
        db.query(ContextGraphPotSource)
        .filter(ContextGraphPotSource.id == source_id)
        .first()
    )
    if row is None:
        return
    row.last_sync_at = datetime.now(timezone.utc)
    row.last_error = error
    try:
        current = int(row.health_score) if row.health_score is not None else 100
    except (TypeError, ValueError):
        current = 100
    if error:
        row.health_score = str(max(0, current - 10))
    else:
        row.health_score = str(min(100, current + 5))
    db.commit()


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
