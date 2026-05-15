"""Pot tenancy HTTP API — factory supports both Firebase (v1) and API-Key (v2) auth."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from urllib.parse import quote

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
    INVITATION_STATUS_DECLINED,
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
from app.modules.utils.email_helper import EmailHelper


INVITATION_DEFAULT_TTL_DAYS = 14
_POT_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

logger = logging.getLogger(__name__)


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


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """Normalize a stored timestamp to aware-UTC before comparing to ``now``.

    ``expires_at`` is written aware-UTC, but a naive value can come back
    depending on the driver/dialect. Comparing naive vs. aware raises a
    TypeError, so treat a naive DB value as UTC (which is what we store).
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _invite_url(token: str) -> str:
    """Public accept link the invitee opens. Mirrors the frontend's
    ``${origin}/pots/join?token=...`` path; the host comes from FRONTEND_URL.
    """
    base = os.environ.get("FRONTEND_URL", "https://app.potpie.ai").rstrip("/")
    return f"{base}/pots/join?token={quote(token, safe='')}"


def _dispatch_invitation_email(
    *,
    to_email: str,
    pot_name: str | None,
    inviter_name: str | None,
    token: str,
    expires_at: datetime | None,
) -> None:
    """Fire-and-forget the invitation email from a sync route.

    Runs in a daemon thread so a slow/failing Resend call never blocks or
    breaks invite creation/resend. EmailHelper itself no-ops unless
    TRANSACTION_EMAILS_ENABLED is set.
    """

    invite_url = _invite_url(token)
    expires_display = (
        expires_at.strftime("%B %d, %Y") if expires_at is not None else None
    )

    def _run() -> None:
        try:
            asyncio.run(
                EmailHelper().send_pot_invitation(
                    to_email,
                    pot_name,
                    inviter_name,
                    invite_url,
                    expires_display,
                )
            )
        except Exception:
            logger.exception(
                "Failed to send pot invitation email to %s", to_email
            )

    threading.Thread(
        target=_run, name="pot-invite-email", daemon=True
    ).start()


def _serialize_invitation(row: ContextGraphPotInvitation) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    # Only pending invites can still be shared. Surfacing the token lets the
    # owner re-copy the link / resend; it's not a bearer credential because
    # accept also checks the signed-in user's email matches.
    if row.status == INVITATION_STATUS_PENDING and row.token:
        out["token"] = row.token
    return out


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


def _pending_invitation_payload(
    row: ContextGraphPotInvitation,
) -> dict[str, Any]:
    """The not-yet-answered invite to attach to a pot the user can see.

    The token is surfaced so the UI can call accept/decline directly. This is
    the same exposure the model/migration already deemed safe (token is an
    identifier, not a bearer credential — accept/decline re-check the
    signed-in user's email), and here it is scoped strictly tighter: only the
    invite addressed to the requesting user's own email is ever attached.
    """
    return {
        "id": row.id,
        "role": normalize_role(row.role),
        "token": row.token,
        "status": row.status,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
    }


def _user_email_lower(db: Session, uid: str) -> str:
    me = db.query(User).filter(User.uid == uid).first()
    return (me.email or "").lower() if me and me.email else ""


def _pending_invitation_for(
    db: Session, *, pot_id: str, email_lower: str
) -> ContextGraphPotInvitation | None:
    """A single pending invite for ``email_lower`` on ``pot_id`` (or None).

    Invitation emails are always stored lowercased at creation, so matching
    against a lowered user email is correct.
    """
    if not email_lower:
        return None
    return (
        db.query(ContextGraphPotInvitation)
        .filter(
            ContextGraphPotInvitation.pot_id == pot_id,
            ContextGraphPotInvitation.email == email_lower,
            ContextGraphPotInvitation.status == INVITATION_STATUS_PENDING,
        )
        .first()
    )


def _remove_auto_added_member(
    db: Session, *, pot_id: str, user_id: str
) -> bool:
    """Drop a pot membership that an invite auto-created.

    Called when an invite is declined (by the invitee) or revoked (by the
    owner): the invitee was added on invite, so opting out / being revoked
    must also remove the membership and make the pot disappear for them.
    An owner row is never removed (defensive — owners are not invited).
    Returns True if a row was deleted.
    """
    m = (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == user_id,
        )
        .first()
    )
    if m is not None and normalize_role(m.role) != POT_ROLE_OWNER:
        db.delete(m)
        return True
    return False


# FE-actionable signal: the invitee authenticated but has no account yet.
# Account creation is owned by the frontend (`/api/v1/signup`); the join
# page detects this code and routes the user through signup, after which
# reopening the link resolves by uid and the email match below applies.
INVITE_SIGNUP_REQUIRED_DETAIL = (
    "No account found for this email. Please sign up first, then reopen "
    "the invitation link."
)


def _require_invited_user(
    db: Session,
    *,
    claims: dict[str, Any],
    uid: str,
    invite_email: str,
) -> None:
    """Match the signed-in account to the invite; never provision.

    Signup is a frontend responsibility (`/api/v1/signup`). Here we only
    require that an account already exists for the signed-in identity and
    that its email matches the invitation. A brand-new invitee who has not
    signed up yet gets a distinct 401 the join page can act on.

    The email check prefers the verified Firebase *token* email (the
    authoritative signed-in identity) and falls back to the stored row.
    """
    invite_email_l = (invite_email or "").lower()
    token_email = str(claims.get("email") or "").strip().lower()

    user = db.query(User).filter(User.uid == uid).first()
    if user is None:
        raise HTTPException(
            status_code=401, detail=INVITE_SIGNUP_REQUIRED_DETAIL
        )

    signed_in_email = token_email or (user.email or "").lower()
    if signed_in_email != invite_email_l:
        raise HTTPException(
            status_code=403,
            detail="Invitation email does not match the signed-in user.",
        )


def _pot_summary(
    row: ContextGraphPot,
    *,
    role: str | None = None,
    pending_invitation: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    # Present only while the user has not answered the invite, so the UI can
    # show an Accept / Decline banner on an otherwise-normal pot.
    out["pending_invitation"] = pending_invitation
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
        # One batched lookup for the user's own un-answered invites across all
        # visible pots (avoids an N+1 over _pending_invitation_for).
        pending_by_pot: dict[str, ContextGraphPotInvitation] = {}
        pot_ids = [pot.id for pot, _ in rows]
        email_lower = _user_email_lower(db, uid)
        if pot_ids and email_lower:
            for iv in (
                db.query(ContextGraphPotInvitation)
                .filter(
                    ContextGraphPotInvitation.pot_id.in_(pot_ids),
                    ContextGraphPotInvitation.email == email_lower,
                    ContextGraphPotInvitation.status
                    == INVITATION_STATUS_PENDING,
                )
                .all()
            ):
                pending_by_pot[iv.pot_id] = iv
        return [
            _pot_summary(
                pot,
                role=role,
                pending_invitation=(
                    _pending_invitation_payload(pending_by_pot[pot.id])
                    if pot.id in pending_by_pot
                    else None
                ),
            )
            for pot, role in rows
        ]

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
        iv = _pending_invitation_for(
            db, pot_id=pot_id, email_lower=_user_email_lower(db, uid)
        )
        return _pot_summary(
            row,
            role=role,
            pending_invitation=(
                _pending_invitation_payload(iv) if iv is not None else None
            ),
        )

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
        archive_transition = False
        if body.archived is not None:
            from datetime import datetime, timezone

            was_archived = row.archived_at is not None
            row.archived_at = datetime.now(timezone.utc) if body.archived else None
            archive_transition = bool(body.archived) and not was_archived
        db.commit()
        db.refresh(row)
        if archive_transition:
            from app.modules.context_graph.archive_pot_cleanup import (
                dispatch_pot_sandbox_cleanup,
            )

            dispatch_pot_sandbox_cleanup(db, pot_id=pot_id)
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
        from app.modules.context_graph.attach_repo_to_pot import (
            attach_repo_to_pot,
            UnknownPotError,
        )

        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        if not body.owner.strip() or not body.repo.strip():
            raise HTTPException(status_code=400, detail="owner and repo required")
        try:
            result = attach_repo_to_pot(
                db,
                pot_id=pot_id,
                provider=body.provider,
                provider_host=body.provider_host,
                owner=body.owner,
                repo=body.repo,
                external_repo_id=body.external_repo_id,
                remote_url=body.remote_url,
                default_branch=body.default_branch,
                submitted_by_user_id=uid,
            )
        except UnknownPotError:
            raise HTTPException(status_code=404, detail="Unknown pot_id.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if result.already_attached:
            raise HTTPException(status_code=409, detail="Repository already attached")
        row = result.repository
        return {
            "id": row.id,
            "repo_name": f"{row.owner}/{row.repo}",
            "provider": row.provider,
            "provider_host": row.provider_host,
            "source_id": result.source_id,
            "bootstrap_event_id": result.bootstrap_event_id,
        }

    @r.delete("/pots/{pot_id}/repositories/{repository_id}")
    def delete_pot_repository(
        pot_id: str,
        repository_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        from app.modules.context_graph.detach_repo_from_pot import (
            UnknownPotError as _DetachUnknownPot,
            UnknownRepositoryError,
            detach_repo_from_pot,
        )

        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        try:
            result = detach_repo_from_pot(
                db,
                pot_id=pot_id,
                repository_id=repository_id,
            )
        except _DetachUnknownPot:
            raise HTTPException(status_code=404, detail="Unknown pot_id.")
        except UnknownRepositoryError:
            raise HTTPException(status_code=404, detail="Repository not found")
        return {"ok": True, "removed_id": result.repository_id}

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
        pot = _context_graph_pot_row_or_404(db, pot_id)

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
            token=token,
            status=INVITATION_STATUS_PENDING,
            expires_at=now + timedelta(days=max(1, int(ttl_days))),
        )
        db.add(row)
        # Auto-add: if the invitee already has an account, make them a member
        # immediately so the pot is visible in their UI right away. They can
        # still accept (keep it) or decline (get removed). The 409 above
        # guarantees they are not already a member here. Brand-new invitees
        # have no uid yet, so for them membership is created on accept.
        if existing_user is not None:
            db.add(
                ContextGraphPotMember(
                    pot_id=pot_id,
                    user_id=existing_user.uid,
                    role=normalize_role(role),
                    invited_by_user_id=uid,
                )
            )
        db.commit()
        db.refresh(row)

        inviter = db.query(User).filter(User.uid == uid).first()
        _dispatch_invitation_email(
            to_email=email,
            pot_name=pot.display_name or pot.slug,
            inviter_name=(inviter.display_name or inviter.email)
            if inviter is not None
            else None,
            token=token,
            expires_at=row.expires_at,
        )

        out = _serialize_invitation(row)
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
        # The invitee was auto-added on invite — revoking must also eject them
        # so the pot disappears from their UI (mirrors decline).
        invited_user = (
            db.query(User).filter(User.email == row.email).first()
        )
        if invited_user is not None:
            _remove_auto_added_member(
                db, pot_id=pot_id, user_id=invited_user.uid
            )
        db.commit()
        return {"ok": True, "invitation_id": invitation_id}

    @r.post("/pots/{pot_id}/invitations/{invitation_id}/resend")
    def resend_pot_invitation(
        pot_id: str,
        invitation_id: str,
        db: Session = Depends(get_db),
        user: dict[str, Any] = Depends(auth_dep),
    ) -> dict[str, Any]:
        uid = _uid(user)
        require_manage_members(db, uid, pot_id)
        pot = _context_graph_pot_row_or_404(db, pot_id)
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
                status_code=400,
                detail="Only pending invitations can be resent",
            )
        if not row.token:
            # Legacy row created before tokens were persisted and missed the
            # backfill — re-mint so it becomes shareable again.
            token, token_hash = _mint_invitation_token()
            row.token = token
            row.token_hash = token_hash
            db.commit()
            db.refresh(row)

        inviter = db.query(User).filter(User.uid == uid).first()
        _dispatch_invitation_email(
            to_email=row.email,
            pot_name=pot.display_name or pot.slug,
            inviter_name=(inviter.display_name or inviter.email)
            if inviter is not None
            else None,
            token=row.token,
            expires_at=row.expires_at,
        )
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
        expires_at = _as_aware_utc(row.expires_at)
        if expires_at is not None and expires_at < now:
            row.status = INVITATION_STATUS_EXPIRED
            db.commit()
            raise HTTPException(status_code=400, detail="Invitation has expired")

        _require_invited_user(
            db, claims=user, uid=uid, invite_email=row.email
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

    @r.post("/pot-invitations/{token}/decline")
    def decline_pot_invitation(
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
        if row.status != INVITATION_STATUS_PENDING:
            raise HTTPException(status_code=400, detail=f"Invitation is {row.status}")

        _require_invited_user(
            db, claims=user, uid=uid, invite_email=row.email
        )

        # Opting out: remove the membership the invite auto-created so the pot
        # disappears from their UI. No-op for a brand-new invitee who was never
        # auto-added (no account at invite time) — declining still records the
        # choice and stops the invite from being accepted later.
        _remove_auto_added_member(db, pot_id=row.pot_id, user_id=uid)
        row.status = INVITATION_STATUS_DECLINED
        db.commit()
        return {"ok": True, "pot_id": row.pot_id, "declined": True}

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
        from app.modules.context_graph.attach_repo_to_pot import (
            attach_repo_to_pot,
            UnknownPotError,
        )

        uid = _uid(user)
        require_manage_repos(db, uid, pot_id)
        if not body.owner.strip() or not body.repo.strip():
            raise HTTPException(status_code=400, detail="owner and repo required")
        provider_host = body.provider_host.strip() or "github.com"
        try:
            result = attach_repo_to_pot(
                db,
                pot_id=pot_id,
                provider="github",
                provider_host=provider_host,
                owner=body.owner,
                repo=body.repo,
                external_repo_id=body.external_repo_id,
                remote_url=body.remote_url,
                default_branch=body.default_branch,
                submitted_by_user_id=uid,
            )
        except UnknownPotError:
            raise HTTPException(status_code=404, detail="Unknown pot_id.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        response: dict[str, Any] = {
            "id": result.source_id,
            "repository_id": result.repository_id,
            "source": serialize_source(result.source),
            "already_attached": result.already_attached,
        }
        if not result.already_attached:
            response["bootstrap_event_id"] = result.bootstrap_event_id
        return response

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
