"""Pot tenancy HTTP API — factory supports both Firebase (v1) and API-Key (v2) auth."""

from __future__ import annotations

import hashlib
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.api_key_deps import get_api_key_user
from app.modules.context_graph.context_graph_pot_integration_model import (
    ContextGraphPotIntegration,
)
from app.modules.context_graph.context_graph_pot_invitation_model import (
    INVITATION_STATUS_ACCEPTED,
    INVITATION_STATUS_EXPIRED,
    INVITATION_STATUS_PENDING,
    INVITATION_STATUS_REVOKED,
    ContextGraphPotInvitation,
)
from app.modules.context_graph.context_graph_pot_member_model import (
    ContextGraphPotMember,
)
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.context_graph_pot_source_model import (
    ContextGraphPotSource,
)
from app.modules.context_graph.pot_access import (
    normalize_role,
    parse_assignable_role,
    require_manage_integrations,
    require_manage_members,
    require_manage_repos,
    require_pot_member,
)
from app.modules.context_graph.pot_member_roles import POT_ROLE_OWNER
from app.modules.context_graph.pot_sources_service import (
    attach_linear_team_source,
    mirror_repository_into_sources,
    serialize_source,
    unmirror_repository_from_sources,
)
from app.modules.users.user_model import User
from app.modules.utils.APIRouter import APIRouter


INVITATION_DEFAULT_TTL_DAYS = 14
_POT_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _normalize_pot_slug(raw: str | None) -> str:
    slug = (raw or "").strip().lower()
    if not slug:
        raise HTTPException(status_code=400, detail="slug is required")
    if not _POT_SLUG_RE.fullmatch(slug):
        raise HTTPException(
            status_code=400,
            detail="slug must be 1-63 chars using lowercase letters, numbers, and hyphens",
        )
    return slug


def _pot_slug_exists(
    db: Session, slug: str, *, exclude_pot_id: str | None = None
) -> bool:
    q = db.query(ContextGraphPot).filter(ContextGraphPot.slug == slug)
    if exclude_pot_id:
        q = q.filter(ContextGraphPot.id != exclude_pot_id)
    return q.first() is not None


def _mint_invitation_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return token, token_hash


def _invitation_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _serialize_invitation(row: ContextGraphPotInvitation) -> dict[str, Any]:
    return {
        "id": row.id,
        "pot_id": row.pot_id,
        "email": row.email,
        "role": normalize_role(row.role),
        "status": row.status,
        "invited_by_user_id": row.invited_by_user_id,
        "accepted_by_user_id": row.accepted_by_user_id,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "accepted_at": row.accepted_at.isoformat() if row.accepted_at else None,
    }


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
        description="Optional legacy label. New create flows should use slug only.",
    )
    id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied UUID string; server generates one if omitted.",
    )
    slug: str = Field(
        ...,
        description="Globally unique stable pot slug.",
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
    role: str = Field(
        default="user",
        description="Assignable role ('user' only; owner cannot be assigned).",
    )


class PatchMemberBody(BaseModel):
    role: str = Field(description="Assignable role ('user' only).")


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


class InviteBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: EmailStr
    role: str = Field(
        default="user", description="Assignable role (only 'user' is accepted)."
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        description="Invite lifetime; defaults to the server-configured TTL.",
    )


class AddGithubRepositorySourceBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    owner: str
    repo: str
    external_repo_id: Optional[str] = None
    remote_url: Optional[str] = None
    default_branch: Optional[str] = None
    provider_host: str = "github.com"


class AddLinearTeamSourceBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    integration_id: str = Field(
        ..., description="integrations.integration_id for Linear"
    )
    team_id: str = Field(..., min_length=1, description="Linear team UUID")
    team_name: Optional[str] = None


class PatchPotSourceBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sync_enabled: Optional[bool] = Field(
        default=None,
        description="Pause/resume ingestion for this source.",
    )


class RawIngestionBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="Short human-readable label.")
    content: Optional[str] = Field(
        default=None,
        description="Free-form text/markdown to ingest as a raw episode.",
    )
    url: Optional[str] = Field(
        default=None,
        description="Link to a resource — if provided, submitted as agent-assisted ingestion.",
    )
    repo_name: Optional[str] = Field(
        default=None,
        description="Optional owner/repo; required when the pot has multiple repositories.",
    )
    source_description: Optional[str] = None
    wait_for_terminal: bool = Field(
        default=False,
        description="If true, block until ingestion reaches a terminal state.",
    )


def make_pot_router(auth_dep: Callable) -> APIRouter:
    """Return a pots CRUD router wired to the given FastAPI auth dependency."""

    r = APIRouter()

    def _uid(user: dict[str, Any]) -> str:
        try:
            return str(user["user_id"])
        except KeyError as exc:
            raise HTTPException(
                status_code=401, detail="Missing user_id on credentials"
            ) from exc

    @r.get("/pots")
    def list_context_pots(
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
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

    @r.post("/pots")
    def create_context_pot(
        body: CreateContextPotBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        pot_id = (body.id or "").strip()
        if pot_id:
            try:
                pot_id = str(uuid.UUID(pot_id))
            except ValueError as e:
                raise HTTPException(status_code=400, detail="id must be a UUID") from e
            existing = (
                db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
            )
            if existing is not None:
                raise HTTPException(
                    status_code=409, detail="A pot with this id already exists"
                )
        else:
            pot_id = str(uuid.uuid4())

        slug = _normalize_pot_slug(body.slug)
        if _pot_slug_exists(db, slug):
            raise HTTPException(status_code=409, detail="slug already in use")

        row = ContextGraphPot(
            id=pot_id,
            user_id=uid,
            created_by_user_id=uid,
            display_name=(body.display_name or "").strip() or None,
            slug=slug,
            primary_repo_name=(body.primary_repo_name or "").strip() or None,
        )
        db.add(row)
        db.add(ContextGraphPotMember(pot_id=pot_id, user_id=uid, role=POT_ROLE_OWNER))
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

    @r.get("/pots/slug-availability/{slug}")
    def get_pot_slug_availability(
        slug: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        _uid(user)
        normalized = _normalize_pot_slug(slug)
        return {"slug": normalized, "available": not _pot_slug_exists(db, normalized)}

    @r.get("/pots/{pot_id}")
    def get_context_pot(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        role = require_pot_member(db, uid, pot_id)
        row = _context_graph_pot_row_or_404(db, pot_id)
        return _pot_summary(row, role=role)

    @r.patch("/pots/{pot_id}")
    def patch_context_pot(
        pot_id: str,
        body: PatchContextPotBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        role = require_manage_repos(db, uid, pot_id)
        row = _context_graph_pot_row_or_404(db, pot_id)
        if body.display_name is not None:
            row.display_name = body.display_name.strip() or None
        if body.slug is not None:
            s = _normalize_pot_slug(body.slug)
            if _pot_slug_exists(db, s, exclude_pot_id=pot_id):
                raise HTTPException(status_code=409, detail="slug already in use")
            row.slug = s
        if body.archived is not None:
            from datetime import datetime, timezone

            row.archived_at = datetime.now(timezone.utc) if body.archived else None
        db.commit()
        db.refresh(row)
        return _pot_summary(row, role=role)

    @r.get("/pots/{pot_id}/members")
    def list_pot_members(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
        require_pot_member(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        rows = (
            db.query(ContextGraphPotMember, User)
            .outerjoin(User, User.uid == ContextGraphPotMember.user_id)
            .filter(ContextGraphPotMember.pot_id == pot_id)
            .order_by(ContextGraphPotMember.created_at.asc())
            .all()
        )
        return [
            {
                "user_id": m.user_id,
                "role": normalize_role(m.role),
                "email": (u.email if u is not None else None),
                "display_name": (u.display_name if u is not None else None),
                "invited_by_user_id": m.invited_by_user_id,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m, u in rows
        ]

    @r.post("/pots/{pot_id}/members")
    def add_pot_member(
        pot_id: str,
        body: AddMemberBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_members(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        new_role = parse_assignable_role(body.role)
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
            pot_id=pot_id, user_id=target, role=new_role, invited_by_user_id=uid
        )
        db.add(m)
        db.commit()
        db.refresh(m)
        return {
            "user_id": m.user_id,
            "role": m.role,
            "invited_by_user_id": m.invited_by_user_id,
        }

    @r.patch("/pots/{pot_id}/members/{member_user_id}")
    def patch_pot_member(
        pot_id: str,
        member_user_id: str,
        body: PatchMemberBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_members(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        new_role = parse_assignable_role(body.role)
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
                status_code=400, detail="owner role cannot be changed via this endpoint"
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

    @r.delete("/pots/{pot_id}/members/{member_user_id}")
    def remove_pot_member(
        pot_id: str,
        member_user_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
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

    @r.get("/pots/{pot_id}/repositories")
    def list_pot_repositories(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
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

    @r.post("/pots/{pot_id}/repositories")
    def add_pot_repository(
        pot_id: str,
        body: AddRepositoryBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
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
        db.flush()
        pot_row = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
        if pot_row is not None and not (pot_row.primary_repo_name or "").strip():
            pot_row.primary_repo_name = f"{owner}/{repo}"
        source = mirror_repository_into_sources(db, row, added_by_user_id=uid)
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "repo_name": f"{row.owner}/{row.repo}",
            "provider": row.provider,
            "provider_host": row.provider_host,
            "source_id": source.id,
        }

    @r.delete("/pots/{pot_id}/repositories/{repository_id}")
    def delete_pot_repository(
        pot_id: str,
        repository_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
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
        unmirror_repository_from_sources(db, row)
        db.delete(row)
        db.flush()
        _recompute_pot_primary_repo_name(db, pot_id)
        db.commit()
        return {"ok": True, "removed_id": repository_id}

    @r.get("/pots/{pot_id}/integrations")
    def list_pot_integrations(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
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

    @r.post("/pots/{pot_id}/integrations")
    def add_pot_integration(
        pot_id: str,
        body: AddIntegrationBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
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

    @r.delete("/pots/{pot_id}/integrations/{integration_id}")
    def delete_pot_integration(
        pot_id: str,
        integration_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
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

    # ---- Pot invitations (owner-managed; members can see pending) --------

    @r.get("/pots/{pot_id}/invitations")
    def list_pot_invitations(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
        require_pot_member(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        rows = (
            db.query(ContextGraphPotInvitation)
            .filter(ContextGraphPotInvitation.pot_id == pot_id)
            .order_by(ContextGraphPotInvitation.created_at.desc())
            .all()
        )
        return [_serialize_invitation(row) for row in rows]

    @r.post("/pots/{pot_id}/invitations")
    def create_pot_invitation(
        pot_id: str,
        body: InviteBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_members(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)

        role = parse_assignable_role(body.role)
        email = str(body.email).strip().lower()
        if not email:
            raise HTTPException(status_code=400, detail="email required")

        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user is not None:
            already_member = (
                db.query(ContextGraphPotMember)
                .filter(
                    ContextGraphPotMember.pot_id == pot_id,
                    ContextGraphPotMember.user_id == existing_user.uid,
                )
                .first()
            )
            if already_member is not None:
                raise HTTPException(
                    status_code=409, detail="User is already a member of this pot."
                )

        pending = (
            db.query(ContextGraphPotInvitation)
            .filter(
                ContextGraphPotInvitation.pot_id == pot_id,
                ContextGraphPotInvitation.email == email,
                ContextGraphPotInvitation.status == INVITATION_STATUS_PENDING,
            )
            .first()
        )
        if pending is not None:
            raise HTTPException(
                status_code=409,
                detail="An invitation for this email is already pending.",
            )

        ttl_days = body.expires_in_days or INVITATION_DEFAULT_TTL_DAYS
        token, token_hash = _mint_invitation_token()
        now = datetime.now(timezone.utc)
        row = ContextGraphPotInvitation(
            id=str(uuid.uuid4()),
            pot_id=pot_id,
            email=email,
            role=role,
            invited_by_user_id=uid,
            token_hash=token_hash,
            status=INVITATION_STATUS_PENDING,
            expires_at=now + timedelta(days=max(1, int(ttl_days))),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        out = _serialize_invitation(row)
        # Return token ONCE so the caller can share the invite link.
        out["token"] = token
        return out

    @r.delete("/pots/{pot_id}/invitations/{invitation_id}")
    def revoke_pot_invitation(
        pot_id: str,
        invitation_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_members(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        row = (
            db.query(ContextGraphPotInvitation)
            .filter(
                ContextGraphPotInvitation.pot_id == pot_id,
                ContextGraphPotInvitation.id == invitation_id,
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Invitation not found")
        if row.status != INVITATION_STATUS_PENDING:
            raise HTTPException(
                status_code=400, detail="Only pending invitations can be revoked"
            )
        row.status = INVITATION_STATUS_REVOKED
        db.commit()
        return {"ok": True, "invitation_id": invitation_id}

    @r.post("/pot-invitations/{token}/accept")
    def accept_pot_invitation(
        token: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        if not token or len(token) < 16:
            raise HTTPException(status_code=400, detail="Invalid invitation token")
        row = (
            db.query(ContextGraphPotInvitation)
            .filter(
                ContextGraphPotInvitation.token_hash == _invitation_token_hash(token)
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Invitation not found")
        now = datetime.now(timezone.utc)
        if row.status != INVITATION_STATUS_PENDING:
            raise HTTPException(status_code=400, detail=f"Invitation is {row.status}")
        if row.expires_at is not None and row.expires_at < now:
            row.status = INVITATION_STATUS_EXPIRED
            db.commit()
            raise HTTPException(status_code=400, detail="Invitation has expired")

        accepting_user = db.query(User).filter(User.uid == uid).first()
        if accepting_user is None:
            raise HTTPException(status_code=401, detail="Unknown user")
        if (accepting_user.email or "").lower() != row.email.lower():
            raise HTTPException(
                status_code=403,
                detail="Invitation email does not match the signed-in user.",
            )

        existing = (
            db.query(ContextGraphPotMember)
            .filter(
                ContextGraphPotMember.pot_id == row.pot_id,
                ContextGraphPotMember.user_id == uid,
            )
            .first()
        )
        if existing is None:
            db.add(
                ContextGraphPotMember(
                    pot_id=row.pot_id,
                    user_id=uid,
                    role=normalize_role(row.role),
                    invited_by_user_id=row.invited_by_user_id,
                )
            )
        row.status = INVITATION_STATUS_ACCEPTED
        row.accepted_at = now
        row.accepted_by_user_id = uid
        db.commit()
        return {"ok": True, "pot_id": row.pot_id, "role": normalize_role(row.role)}

    # ---- Pot sources (the data-scope layer, separate from integrations) --

    @r.get("/pots/{pot_id}/sources")
    def list_pot_sources(
        pot_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> list[dict[str, Any]]:
        uid = _uid(user)
        require_pot_member(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        rows = (
            db.query(ContextGraphPotSource)
            .filter(ContextGraphPotSource.pot_id == pot_id)
            .order_by(ContextGraphPotSource.created_at.asc())
            .all()
        )
        return [serialize_source(row) for row in rows]

    @r.post("/pots/{pot_id}/sources/github/repository")
    def add_pot_github_repository_source(
        pot_id: str,
        body: AddGithubRepositorySourceBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        """Attach a GitHub repo to the pot and mirror it into ``context_graph_pot_sources``.

        This is the preferred surface for the UI repository picker — it creates
        both the repository row (what context-engine queries for ingestion) and
        the source row (what the UI lists on the Sources tab).
        """
        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        owner = body.owner.strip()
        repo = body.repo.strip()
        if not owner or not repo:
            raise HTTPException(status_code=400, detail="owner and repo required")
        provider = "github"
        provider_host = body.provider_host.strip() or "github.com"

        existing_repo = (
            db.query(ContextGraphPotRepository)
            .filter(
                ContextGraphPotRepository.pot_id == pot_id,
                ContextGraphPotRepository.provider == provider,
                ContextGraphPotRepository.provider_host == provider_host,
                ContextGraphPotRepository.owner == owner,
                ContextGraphPotRepository.repo == repo,
            )
            .first()
        )
        if existing_repo is not None:
            source = mirror_repository_into_sources(
                db, existing_repo, added_by_user_id=uid
            )
            db.commit()
            return {
                "id": source.id,
                "repository_id": existing_repo.id,
                "source": serialize_source(source),
                "already_attached": True,
            }

        repository = ContextGraphPotRepository(
            id=str(uuid.uuid4()),
            pot_id=pot_id,
            provider=provider,
            provider_host=provider_host,
            owner=owner,
            repo=repo,
            external_repo_id=(body.external_repo_id or "").strip() or None,
            remote_url=(body.remote_url or "").strip() or None,
            default_branch=(body.default_branch or "").strip() or None,
            added_by_user_id=uid,
        )
        db.add(repository)
        db.flush()

        pot_row = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
        if pot_row is not None and not (pot_row.primary_repo_name or "").strip():
            pot_row.primary_repo_name = f"{owner}/{repo}"

        source = mirror_repository_into_sources(db, repository, added_by_user_id=uid)
        db.commit()
        db.refresh(source)
        return {
            "id": source.id,
            "repository_id": repository.id,
            "source": serialize_source(source),
            "already_attached": False,
        }

    @r.get("/pots/{pot_id}/sources/linear/teams")
    def list_pot_linear_teams(
        pot_id: str,
        integration_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        """List Linear teams for a connected Linear integration owned by the caller.

        Used by the pot Sources picker; the integration must belong to the caller
        (matches the existing /api/v1/sources/linear/teams contract).
        """
        from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
        from integrations.adapters.outbound.linear.graphql_client import linear_graphql
        from integrations.adapters.outbound.postgres.integration_model import (
            Integration,
        )
        from integrations.domain.integrations_schema import AuthData, IntegrationType

        uid = _uid(user)
        require_pot_member(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)

        row = (
            db.query(Integration)
            .filter(
                Integration.integration_id == integration_id,
                Integration.created_by == uid,
                Integration.integration_type == IntegrationType.LINEAR.value,
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="integration_not_found")
        auth = AuthData.model_validate(row.auth_data or {})
        if not auth.access_token:
            raise HTTPException(status_code=400, detail="missing_token")
        try:
            token = decrypt_token(auth.access_token)
            data = linear_graphql(
                token,
                "query Teams { viewer { organization { teams { nodes { id name key } } } } }",
                {},
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"Linear API error: {exc}"
            ) from exc
        org = ((data.get("viewer") or {}).get("organization")) or {}
        teams = ((org.get("teams") or {}).get("nodes")) or []
        return {"teams": teams}

    @r.post("/pots/{pot_id}/sources/linear/team")
    def add_pot_linear_team_source(
        pot_id: str,
        body: AddLinearTeamSourceBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        from integrations.adapters.outbound.postgres.integration_model import (
            Integration,
        )
        from integrations.domain.integrations_schema import IntegrationType

        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)

        integ = (
            db.query(Integration)
            .filter(
                Integration.integration_id == body.integration_id,
                Integration.created_by == uid,
                Integration.integration_type == IntegrationType.LINEAR.value,
                Integration.active.is_(True),
            )
            .first()
        )
        if integ is None:
            raise HTTPException(status_code=404, detail="integration_not_found")

        try:
            row, already_attached = attach_linear_team_source(
                db,
                pot_id=pot_id,
                integration_id=body.integration_id,
                team_id=body.team_id,
                team_name=body.team_name,
                added_by_user_id=uid,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "source": serialize_source(row),
            "already_attached": already_attached,
        }

    @r.patch("/pots/{pot_id}/sources/{source_id}")
    def patch_pot_source(
        pot_id: str,
        source_id: str,
        body: PatchPotSourceBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        row = (
            db.query(ContextGraphPotSource)
            .filter(
                ContextGraphPotSource.pot_id == pot_id,
                ContextGraphPotSource.id == source_id,
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Source not found")
        if body.sync_enabled is not None:
            row.sync_enabled = bool(body.sync_enabled)
        db.commit()
        db.refresh(row)
        return serialize_source(row)

    @r.delete("/pots/{pot_id}/sources/{source_id}")
    def delete_pot_source(
        pot_id: str,
        source_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)
        row = (
            db.query(ContextGraphPotSource)
            .filter(
                ContextGraphPotSource.pot_id == pot_id,
                ContextGraphPotSource.id == source_id,
            )
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Source not found")

        # For GitHub repository sources, also remove the matching repository
        # row so the pot no longer routes webhooks there.
        if row.source_kind == "repository" and row.provider == "github":
            import json as _json

            try:
                scope = _json.loads(row.scope_json) if row.scope_json else {}
            except (TypeError, ValueError):
                scope = {}
            owner = (scope.get("owner") or "").strip()
            repo = (scope.get("repo") or "").strip()
            provider_host = (
                scope.get("provider_host") or "github.com"
            ).strip() or "github.com"
            if owner and repo:
                repo_row = (
                    db.query(ContextGraphPotRepository)
                    .filter(
                        ContextGraphPotRepository.pot_id == pot_id,
                        ContextGraphPotRepository.provider == "github",
                        ContextGraphPotRepository.provider_host == provider_host,
                        ContextGraphPotRepository.owner == owner,
                        ContextGraphPotRepository.repo == repo,
                    )
                    .first()
                )
                if repo_row is not None:
                    db.delete(repo_row)
        db.delete(row)
        db.flush()
        _recompute_pot_primary_repo_name(db, pot_id)
        db.commit()
        return {"ok": True, "removed_id": source_id}

    # ---- Raw ingestion (owner + user) -----------------------------------

    @r.post("/pots/{pot_id}/ingest/raw")
    def ingest_pot_raw(
        pot_id: str,
        body: RawIngestionBody,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        """Accept a raw note/link/content from a pot member and enqueue ingestion.

        Both owners and users can submit. The context-engine submission service
        persists the canonical ``context_events`` row, so the Ingestion tab
        will see the event immediately.
        """
        from app.modules.context_graph.pot_access import require_pot_raw_ingest
        from app.modules.context_graph.wiring import build_container_for_user_session
        from domain.ingestion_kinds import INGESTION_KIND_RAW_EPISODE
        from domain.ingestion_event_models import IngestionSubmissionRequest

        uid = _uid(user)
        require_pot_raw_ingest(db, uid, pot_id)
        _context_graph_pot_row_or_404(db, pot_id)

        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="name required")

        content = (body.content or "").strip()
        url = (body.url or "").strip()
        if not content and not url:
            raise HTTPException(
                status_code=400, detail="Provide either content or a url to ingest."
            )

        container = build_container_for_user_session(db, uid)
        try:
            request = IngestionSubmissionRequest(
                pot_id=pot_id,
                ingestion_kind=INGESTION_KIND_RAW_EPISODE,
                source_channel="ui_raw_ingest",
                source_system="manual",
                event_type="manual_submission"
                if url and not content
                else "raw_episode",
                action="ingest" if url and not content else "submit",
                source_id=f"manual_{uuid.uuid4()}",
                repo_name=(body.repo_name or None),
                payload={
                    "name": name,
                    "episode_body": content or url,
                    "source_description": body.source_description
                    or f"manual raw note by {uid}",
                    "url": url or None,
                    "title": name,
                    "submitted_text": content or None,
                    "content_type_guess": "link" if url and not content else "text",
                    "submitted_by_user_id": uid,
                },
            )
            receipt = container.ingestion_submission(db).submit(
                request,
                sync=False,
                wait=bool(body.wait_for_terminal),
                timeout_seconds=45.0 if body.wait_for_terminal else None,
            )
        except ValueError as e:
            if str(e) == "no_reconciliation_agent":
                raise HTTPException(status_code=503, detail=str(e)) from e
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "event_id": receipt.event_id,
            "status": receipt.status,
            "pot_id": pot_id,
            "error": receipt.error,
        }

    return r


# v2 router (X-API-Key auth) — keeps backward compatibility for api/router.py
router = make_pot_router(get_api_key_user)
