"""Unified Sources API (providers catalog, connections, project attachments, sync)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.context_graph.tasks import (
    context_graph_backfill_pot,
    context_graph_sync_linear_project_source,
)
from integrations.application.bootstrap import load_providers
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import AuthData, IntegrationType
from integrations.adapters.outbound.linear.graphql_client import linear_graphql
from integrations.adapters.outbound.postgres.project_source_model import ProjectSource
from integrations.application.project_sources_service import (
    attach_linear_team_source,
    delete_project_source,
    list_all_sources_for_project,
)
from integrations.domain.provider_registry import get_provider_registry
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
from app.modules.projects.projects_model import Project
from app.modules.utils.APIRouter import APIRouter
from app.modules.auth.auth_provider_model import UserAuthProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


class LinearAttachBody(BaseModel):
    integration_id: str = Field(..., description="integrations.integration_id for Linear")
    team_id: str = Field(..., min_length=1, description="Linear team UUID")
    team_name: str | None = None


@router.get("/providers")
async def list_providers() -> dict[str, Any]:
    load_providers()
    reg = get_provider_registry()
    items = []
    for p in reg.list_all():
        items.append(
            {
                "id": p.id,
                "display_name": p.display_name,
                "capabilities": list(p.capabilities),
                "source_kinds": list(p.source_kinds),
                "port_kind": p.port_kind,
                "oss_available": p.oss_available,
            }
        )
    return {"providers": items}


@router.get("/connections")
async def list_connections(
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    connections: list[dict[str, Any]] = []

    gh = (
        db.query(UserAuthProvider)
        .filter(
            UserAuthProvider.user_id == uid,
            UserAuthProvider.provider_type == "firebase_github",
        )
        .first()
    )
    connections.append(
        {
            "provider": "github",
            "kind": "user_auth_provider",
            "connected": bool(gh and gh.access_token),
            "id": str(gh.id) if gh else None,
            "integration_id": None,
            "name": None,
        }
    )

    linear_rows = (
        db.query(Integration)
        .filter(
            Integration.created_by == uid,
            Integration.integration_type == IntegrationType.LINEAR.value,
            Integration.active.is_(True),
        )
        .all()
    )
    for row in linear_rows:
        connections.append(
            {
                "provider": "linear",
                "kind": "integration",
                "connected": True,
                "integration_id": row.integration_id,
                "name": row.name,
            }
        )

    return {"connections": connections}


@router.get("/linear/teams")
async def linear_teams(
    integration_id: str,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    row = (
        db.query(Integration)
        .filter(
            Integration.integration_id == integration_id,
            Integration.created_by == uid,
            Integration.integration_type == IntegrationType.LINEAR.value,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="integration_not_found")
    auth = AuthData.model_validate(row.auth_data or {})
    if not auth.access_token:
        raise HTTPException(status_code=400, detail="missing_token")
    try:
        token = decrypt_token(auth.access_token)
        q = """
        query Teams {
          viewer {
            organization {
              teams {
                nodes { id name key }
              }
            }
          }
        }
        """
        data = linear_graphql(token, q, {})
    except Exception as exc:
        logger.exception("Failed to fetch Linear teams for integration %s", integration_id)
        raise HTTPException(
            status_code=502, detail=f"Linear API error: {exc}"
        ) from exc
    org = ((data.get("viewer") or {}).get("organization")) or {}
    teams = ((org.get("teams") or {}).get("nodes")) or []
    return {"teams": teams}


@router.get("/projects/{project_id}/sources")
async def get_project_sources(
    project_id: str,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    rows = list_all_sources_for_project(db, project_id, uid)
    return {
        "sources": [
            {
                "id": s.id,
                "provider": s.provider,
                "source_kind": s.source_kind,
                "scope_json": s.scope_json,
                "sync_enabled": s.sync_enabled,
                "last_sync_at": s.last_sync_at.isoformat() if s.last_sync_at else None,
                "last_error": s.last_error,
                "health_score": s.health_score,
                "integration_id": s.integration_id,
            }
            for s in rows
        ]
    }


@router.post("/projects/{project_id}/sources/linear")
async def attach_linear_to_project(
    project_id: str,
    body: LinearAttachBody,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    try:
        src = attach_linear_team_source(
            db,
            project_id=project_id,
            integration_id=body.integration_id,
            team_id=body.team_id,
            team_name=body.team_name,
            user_id=uid,
        )
    except ValueError as e:
        msg = str(e)
        if msg == "project_not_found_or_forbidden":
            raise HTTPException(status_code=404, detail="project_not_found") from e
        if msg == "integration_not_found_or_forbidden":
            raise HTTPException(
                status_code=404, detail="integration_not_found"
            ) from e
        raise
    return {"id": src.id, "provider": src.provider, "scope_json": src.scope_json}


@router.delete("/projects/{project_id}/sources/{source_id}")
async def detach_project_source(
    project_id: str,
    source_id: str,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    proj = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == uid)
        .first()
    )
    if not proj:
        raise HTTPException(status_code=404, detail="project_not_found")
    row = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.id == source_id,
            ProjectSource.project_id == project_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="source_not_found")
    if not delete_project_source(db, source_id, uid):
        raise HTTPException(status_code=404, detail="source_not_found")
    return {"status": "deleted", "id": source_id}


@router.post("/projects/{project_id}/sources/sync")
async def sync_project_sources(
    project_id: str,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    proj = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == uid)
        .first()
    )
    if not proj:
        raise HTTPException(status_code=404, detail="project_not_found")
    has_github = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.provider == "github",
            ProjectSource.sync_enabled.is_(True),
        )
        .first()
    ) is not None
    if has_github:
        context_graph_backfill_pot.delay(project_id)
    linear_sources = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.project_id == project_id,
            ProjectSource.provider == "linear",
            ProjectSource.sync_enabled.is_(True),
        )
        .all()
    )
    for ls in linear_sources:
        try:
            context_graph_sync_linear_project_source.delay(ls.id)
        except Exception as e:
            logger.warning("Failed to enqueue linear sync %s: %s", ls.id, e)
    return {
        "status": "enqueued",
        "project_id": project_id,
        "github_backfill": has_github,
        "linear_sources_queued": len(linear_sources),
    }


@router.post("/projects/{project_id}/sources/{source_id}/sync")
async def sync_one_project_source(
    project_id: str,
    source_id: str,
    user: dict = Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    uid = user["user_id"]
    proj = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == uid)
        .first()
    )
    if not proj:
        raise HTTPException(status_code=404, detail="project_not_found")
    row = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.id == source_id,
            ProjectSource.project_id == project_id,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="source_not_found")
    if row.provider == "linear":
        context_graph_sync_linear_project_source.delay(row.id)
        return {"status": "enqueued", "source_id": source_id, "kind": "linear"}
    if row.provider == "github":
        context_graph_backfill_pot.delay(project_id)
        return {"status": "enqueued", "source_id": source_id, "kind": "github"}
    raise HTTPException(status_code=400, detail="unsupported_provider")


@router.post("/webhooks/linear")
async def linear_sources_webhook(
    request: Request,
    db: Session = Depends(get_db),
    linear_signature: str | None = Header(default=None, alias="Linear-Signature"),
) -> dict[str, Any]:
    raw = await request.body()
    secret = (os.getenv("LINEAR_WEBHOOK_SECRET") or "").strip()
    if secret and not linear_signature:
        raise HTTPException(status_code=401, detail="missing_signature")
    if secret and linear_signature:
        expected = hmac.new(
            secret.encode("utf-8"),
            raw,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, linear_signature.strip()):
            raise HTTPException(status_code=401, detail="invalid_signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid_json") from None

    action = payload.get("action")
    team_id = (payload.get("data") or {}).get("teamId") or payload.get("teamId")
    if not team_id and isinstance(payload.get("data"), dict):
        d = payload["data"]
        team_id = (
            (d.get("team") or {}).get("id")
            if isinstance(d.get("team"), dict)
            else None
        )

    if not team_id:
        return {
            "status": "accepted",
            "enqueued": 0,
            "action": action,
            "note": "No team_id in payload; nothing enqueued.",
        }

    rows = (
        db.query(ProjectSource)
        .filter(
            ProjectSource.provider == "linear",
            ProjectSource.sync_enabled.is_(True),
            ProjectSource.scope_json["team_id"].astext == str(team_id),
        )
        .all()
    )
    n = 0
    for s in rows:
        try:
            context_graph_sync_linear_project_source.delay(s.id)
            n += 1
        except Exception as e:
            logger.warning("Webhook enqueue failed for source %s: %s", s.id, e)
    return {
        "status": "accepted",
        "enqueued": n,
        "action": action,
        "note": "Signature verified only if LINEAR_WEBHOOK_SECRET is set.",
    }
