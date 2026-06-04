"""Sources catalog and provider connection endpoints.

Pot ↔ source attachments live on the context-engine routes
(``/api/v1/context/pots/{pot_id}/sources/...`` in
``app/modules/context_graph/context_pot_routes.py``); the legacy
``/sources/projects/...`` routes were removed when ``project_sources``
was retired. This module is what's left: provider catalog, OAuth
connection listing, and Linear team enumeration.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from integrations.application.bootstrap import load_providers
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import AuthData, IntegrationType
from integrations.adapters.outbound.linear.graphql_client import linear_graphql
from integrations.domain.provider_registry import get_provider_registry
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
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
        raise HTTPException(
            status_code=502, detail=f"Linear API error: {exc}"
        ) from exc
    org = ((data.get("viewer") or {}).get("organization")) or {}
    teams = ((org.get("teams") or {}).get("nodes")) or []
    return {"teams": teams}
