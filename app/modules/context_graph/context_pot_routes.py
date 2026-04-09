"""Pot tenancy HTTP API under /api/v2/context (X-API-Key)."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.api_key_deps import get_api_key_user
from app.modules.context_graph.context_graph_pot_integration_model import (
    ContextGraphPotIntegration,
)
from app.modules.context_graph.context_graph_pot_member_model import ContextGraphPotMember
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.pot_access import (
    api_key_user_id,
    parse_role,
    require_manage_integrations,
    require_manage_members,
    require_manage_repos,
    require_pot_member,
    user_role_on_context_graph_pot,
)
from app.modules.context_graph.pot_member_roles import (
    POT_ROLE_OWNER,
)
from app.modules.users.user_model import User
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


def _context_graph_pot_row_or_404(db: Session, pot_id: str) -> ContextGraphPot:
    row = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown pot_id.")
    return row


def _recompute_pot_primary_repo_name(db: Session, pot_id: str) -> None:
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot is None:
        return
    first = (
        db.query(ContextGraphPotRepository)
        .filter(ContextGraphPotRepository.pot_id == pot_id)
        .order_by(ContextGraphPotRepository.created_at.asc())
        .first()
    )
    pot.primary_repo_name = f"{first.owner}/{first.repo}" if first else None


def _pot_summary(row: ContextGraphPot, *, role: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": row.id,
        "display_name": row.display_name,
        "slug": row.slug,
        "primary_repo_name": row.primary_repo_name,
        "created_by_user_id": row.created_by_user_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "archived_at": row.archived_at.isoformat() if row.archived_at else None,
    }
    if role is not None:
        out["role"] = role
    return out


class CreateContextPotBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    display_name: Optional[str] = Field(
        default=None,
        description="Optional label (does not need to match a GitHub repo).",
    )
    id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied UUID string; server generates one if omitted.",
    )
    slug: Optional[str] = Field(
        default=None,
        description="Optional unique slug per creator (for stable references).",
    )
    primary_repo_name: Optional[str] = Field(
        default=None,
        description="Optional owner/repo for GitHub-backed features (also stored as a pot repository).",
    )


class PatchContextPotBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    display_name: Optional[str] = None
    slug: Optional[str] = None
    archived: Optional[bool] = Field(
        default=None,
        description="If true, set archived_at; if false, clear archived_at (admin or owner).",
    )


class AddMemberBody(BaseModel):
    user_id: str = Field(description="Potpie user uid to add.")
    role: str = Field(description="admin or read_only (not owner).")


class PatchMemberBody(BaseModel):
    role: str


class AddRepositoryBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str = "github"
    provider_host: str = "github.com"
    owner: str
    repo: str
    external_repo_id: Optional[str] = None
    remote_url: Optional[str] = None
    default_branch: Optional[str] = None


class AddIntegrationBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    integration_type: str
    provider: str
    provider_host: Optional[str] = None
    external_account_id: Optional[str] = None
    config_json: Optional[str] = None


@router.get("/pots")
def list_context_pots(
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> list[dict[str, Any]]:
    uid = api_key_user_id(user)
    rows = (
        db.query(ContextGraphPot, ContextGraphPotMember.role)
        .join(
            ContextGraphPotMember,
            ContextGraphPotMember.pot_id == ContextGraphPot.id,
        )
        .filter(
            ContextGraphPotMember.user_id == uid,
            ContextGraphPot.archived_at.is_(None),
        )
        .order_by(ContextGraphPot.created_at.desc())
        .all()
    )
    return [_pot_summary(r, role=role) for r, role in rows]


@router.post("/pots")
def create_context_pot(
    body: CreateContextPotBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    pot_id = (body.id or "").strip()
    if pot_id:
        try:
            pot_id = str(uuid.UUID(pot_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="id must be a UUID") from e
        existing = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
        if existing is not None:
            raise HTTPException(status_code=409, detail="A pot with this id already exists")
    else:
        pot_id = str(uuid.uuid4())

    slug = (body.slug or "").strip() or None
    if slug:
        q = db.query(ContextGraphPot).filter(
            ContextGraphPot.created_by_user_id == uid,
            ContextGraphPot.slug == slug,
        )
        if q.first() is not None:
            raise HTTPException(status_code=409, detail="slug already in use for your account")

    row = ContextGraphPot(
        id=pot_id,
        user_id=uid,
        created_by_user_id=uid,
        display_name=(body.display_name or "").strip() or None,
        slug=slug,
        primary_repo_name=(body.primary_repo_name or "").strip() or None,
    )
    db.add(row)
    db.add(
        ContextGraphPotMember(
            pot_id=pot_id,
            user_id=uid,
            role=POT_ROLE_OWNER,
        )
    )
    prn = (body.primary_repo_name or "").strip()
    if prn and "/" in prn:
        o, rn = prn.split("/", 1)
        o, rn = o.strip(), rn.strip()
        if o and rn:
            db.add(
                ContextGraphPotRepository(
                    id=str(uuid.uuid4()),
                    pot_id=pot_id,
                    provider="github",
                    provider_host="github.com",
                    owner=o,
                    repo=rn,
                    added_by_user_id=uid,
                )
            )
    db.commit()
    db.refresh(row)
    return _pot_summary(row, role=POT_ROLE_OWNER)


@router.get("/pots/{pot_id}")
def get_context_pot(
    pot_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    role = require_pot_member(db, uid, pot_id)
    row = _context_graph_pot_row_or_404(db, pot_id)
    return _pot_summary(row, role=role)


@router.patch("/pots/{pot_id}")
def patch_context_pot(
    pot_id: str,
    body: PatchContextPotBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    role = require_manage_repos(db, uid, pot_id)
    row = _context_graph_pot_row_or_404(db, pot_id)
    if body.display_name is not None:
        row.display_name = body.display_name.strip() or None
    if body.slug is not None:
        s = body.slug.strip() or None
        if s:
            creator = row.created_by_user_id or row.user_id
            clash = (
                db.query(ContextGraphPot)
                .filter(
                    ContextGraphPot.created_by_user_id == creator,
                    ContextGraphPot.slug == s,
                    ContextGraphPot.id != pot_id,
                )
                .first()
            )
            if clash is not None:
                raise HTTPException(status_code=409, detail="slug already in use")
        row.slug = s
    if body.archived is not None:
        from datetime import datetime, timezone

        if body.archived:
            row.archived_at = datetime.now(timezone.utc)
        else:
            row.archived_at = None
    db.commit()
    db.refresh(row)
    return _pot_summary(row, role=role)


@router.get("/pots/{pot_id}/members")
def list_pot_members(
    pot_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> list[dict[str, Any]]:
    uid = api_key_user_id(user)
    require_pot_member(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    rows = (
        db.query(ContextGraphPotMember)
        .filter(ContextGraphPotMember.pot_id == pot_id)
        .all()
    )
    return [
        {
            "user_id": m.user_id,
            "role": m.role,
            "invited_by_user_id": m.invited_by_user_id,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }
        for m in rows
    ]


@router.post("/pots/{pot_id}/members")
def add_pot_member(
    pot_id: str,
    body: AddMemberBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_members(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    new_role = parse_role(body.role)
    if new_role == POT_ROLE_OWNER:
        raise HTTPException(
            status_code=400,
            detail="owner role cannot be assigned via this endpoint",
        )
    target = body.user_id.strip()
    if not target:
        raise HTTPException(status_code=400, detail="user_id required")
    if db.query(User).filter(User.uid == target).first() is None:
        raise HTTPException(status_code=404, detail="User not found")
    exists = (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == target,
        )
        .first()
    )
    if exists is not None:
        raise HTTPException(status_code=409, detail="User is already a member")
    m = ContextGraphPotMember(
        pot_id=pot_id,
        user_id=target,
        role=new_role,
        invited_by_user_id=uid,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return {
        "user_id": m.user_id,
        "role": m.role,
        "invited_by_user_id": m.invited_by_user_id,
    }


@router.patch("/pots/{pot_id}/members/{member_user_id}")
def patch_pot_member(
    pot_id: str,
    member_user_id: str,
    body: PatchMemberBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_members(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    new_role = parse_role(body.role)
    m = (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == member_user_id,
        )
        .first()
    )
    if m is None:
        raise HTTPException(status_code=404, detail="Member not found")
    if m.role == POT_ROLE_OWNER and new_role != POT_ROLE_OWNER:
        raise HTTPException(
            status_code=400,
            detail="owner role cannot be changed via this endpoint",
        )
    if new_role == POT_ROLE_OWNER and m.role != POT_ROLE_OWNER:
        raise HTTPException(
            status_code=400,
            detail="owner role cannot be assigned via this endpoint",
        )
    m.role = new_role
    db.commit()
    db.refresh(m)
    return {"user_id": m.user_id, "role": m.role}


@router.delete("/pots/{pot_id}/members/{member_user_id}")
def remove_pot_member(
    pot_id: str,
    member_user_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_members(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    m = (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == member_user_id,
        )
        .first()
    )
    if m is None:
        raise HTTPException(status_code=404, detail="Member not found")
    if m.role == POT_ROLE_OWNER:
        raise HTTPException(
            status_code=400,
            detail="owner cannot be removed; ownership transfer is not implemented",
        )
    db.delete(m)
    db.commit()
    return {"ok": True, "removed_user_id": member_user_id}


@router.get("/pots/{pot_id}/repositories")
def list_pot_repositories(
    pot_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> list[dict[str, Any]]:
    uid = api_key_user_id(user)
    require_pot_member(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    rows = (
        db.query(ContextGraphPotRepository)
        .filter(ContextGraphPotRepository.pot_id == pot_id)
        .order_by(ContextGraphPotRepository.created_at.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "provider": r.provider,
            "provider_host": r.provider_host,
            "owner": r.owner,
            "repo": r.repo,
            "repo_name": f"{r.owner}/{r.repo}",
            "external_repo_id": r.external_repo_id,
            "remote_url": r.remote_url,
            "default_branch": r.default_branch,
            "added_by_user_id": r.added_by_user_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.post("/pots/{pot_id}/repositories")
def add_pot_repository(
    pot_id: str,
    body: AddRepositoryBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_repos(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    owner = body.owner.strip()
    repo = body.repo.strip()
    if not owner or not repo:
        raise HTTPException(status_code=400, detail="owner and repo required")
    exists = (
        db.query(ContextGraphPotRepository)
        .filter(
            ContextGraphPotRepository.pot_id == pot_id,
            ContextGraphPotRepository.provider == body.provider.strip(),
            ContextGraphPotRepository.provider_host == body.provider_host.strip(),
            ContextGraphPotRepository.owner == owner,
            ContextGraphPotRepository.repo == repo,
        )
        .first()
    )
    if exists is not None:
        raise HTTPException(status_code=409, detail="Repository already attached")
    rid = str(uuid.uuid4())
    row = ContextGraphPotRepository(
        id=rid,
        pot_id=pot_id,
        provider=body.provider.strip(),
        provider_host=body.provider_host.strip(),
        owner=owner,
        repo=repo,
        external_repo_id=(body.external_repo_id or "").strip() or None,
        remote_url=(body.remote_url or "").strip() or None,
        default_branch=(body.default_branch or "").strip() or None,
        added_by_user_id=uid,
    )
    db.add(row)
    pot_row = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot_row is not None and not (pot_row.primary_repo_name or "").strip():
        pot_row.primary_repo_name = f"{owner}/{repo}"
    db.commit()
    db.refresh(row)
    return {
        "id": row.id,
        "repo_name": f"{row.owner}/{row.repo}",
        "provider": row.provider,
        "provider_host": row.provider_host,
    }


@router.delete("/pots/{pot_id}/repositories/{repository_id}")
def delete_pot_repository(
    pot_id: str,
    repository_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_repos(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    row = (
        db.query(ContextGraphPotRepository)
        .filter(
            ContextGraphPotRepository.pot_id == pot_id,
            ContextGraphPotRepository.id == repository_id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    db.delete(row)
    db.flush()
    _recompute_pot_primary_repo_name(db, pot_id)
    db.commit()
    return {"ok": True, "removed_id": repository_id}


@router.get("/pots/{pot_id}/integrations")
def list_pot_integrations(
    pot_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> list[dict[str, Any]]:
    uid = api_key_user_id(user)
    require_pot_member(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    rows = (
        db.query(ContextGraphPotIntegration)
        .filter(ContextGraphPotIntegration.pot_id == pot_id)
        .order_by(ContextGraphPotIntegration.created_at.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "integration_type": r.integration_type,
            "provider": r.provider,
            "provider_host": r.provider_host,
            "external_account_id": r.external_account_id,
            "created_by_user_id": r.created_by_user_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@router.post("/pots/{pot_id}/integrations")
def add_pot_integration(
    pot_id: str,
    body: AddIntegrationBody,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_integrations(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    iid = str(uuid.uuid4())
    row = ContextGraphPotIntegration(
        id=iid,
        pot_id=pot_id,
        integration_type=body.integration_type.strip(),
        provider=body.provider.strip(),
        provider_host=(body.provider_host or "").strip() or None,
        external_account_id=(body.external_account_id or "").strip() or None,
        config_json=body.config_json,
        created_by_user_id=uid,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "integration_type": row.integration_type}


@router.delete("/pots/{pot_id}/integrations/{integration_id}")
def delete_pot_integration(
    pot_id: str,
    integration_id: str,
    db: Session = Depends(get_db),
    user: dict[str, Any] = Depends(get_api_key_user),
) -> dict[str, Any]:
    uid = api_key_user_id(user)
    require_manage_integrations(db, uid, pot_id)
    _context_graph_pot_row_or_404(db, pot_id)
    row = (
        db.query(ContextGraphPotIntegration)
        .filter(
            ContextGraphPotIntegration.pot_id == pot_id,
            ContextGraphPotIntegration.id == integration_id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Integration not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "removed_id": integration_id}
