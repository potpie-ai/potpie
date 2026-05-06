"""Resolve a user's role on a context-graph pot (membership or pot owner row)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_member_model import ContextGraphPotMember
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.pot_member_roles import (
    ASSIGNABLE_POT_ROLES,
    POT_ROLE_OWNER,
    can_ingest_or_reset,
    can_ingest_raw,
    can_manage_members,
    can_manage_repos_and_integrations,
    can_query_context,
    normalize_role,
)


def user_role_on_context_graph_pot(db: Session, user_id: str, pot_id: str) -> Optional[str]:
    row = (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == user_id,
        )
        .first()
    )
    if row is not None:
        return normalize_role(row.role)
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot is not None and pot.user_id == user_id:
        return POT_ROLE_OWNER
    return None


def require_pot_member(db: Session, user_id: str, pot_id: str) -> str:
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot is None or pot.archived_at is not None:
        raise HTTPException(status_code=404, detail="Unknown pot_id or not a member.")
    role = user_role_on_context_graph_pot(db, user_id, pot_id)
    if role is None or not can_query_context(role):
        raise HTTPException(status_code=404, detail="Unknown pot_id or not a member.")
    return role


def require_pot_ingest(db: Session, user_id: str, pot_id: str) -> str:
    """Owner-only gate for webhook / backfill driven graph mutations."""
    role = require_pot_member(db, user_id, pot_id)
    if not can_ingest_or_reset(role):
        raise HTTPException(
            status_code=403,
            detail="Only the pot owner can mutate pot context.",
        )
    return role


def require_pot_raw_ingest(db: Session, user_id: str, pot_id: str) -> str:
    """Members (owner or user) may manually submit raw notes / links / content."""
    role = require_pot_member(db, user_id, pot_id)
    if not can_ingest_raw(role):
        raise HTTPException(
            status_code=403,
            detail="This role cannot ingest context for this pot.",
        )
    return role


def require_pot_reset(db: Session, user_id: str, pot_id: str) -> str:
    return require_pot_ingest(db, user_id, pot_id)


def require_manage_members(db: Session, user_id: str, pot_id: str) -> str:
    role = require_pot_member(db, user_id, pot_id)
    if not can_manage_members(role):
        raise HTTPException(status_code=403, detail="Only the pot owner can manage members.")
    return role


def require_manage_repos(db: Session, user_id: str, pot_id: str) -> str:
    role = require_pot_member(db, user_id, pot_id)
    if not can_manage_repos_and_integrations(role):
        raise HTTPException(
            status_code=403,
            detail="This role cannot attach or detach repositories for this pot.",
        )
    return role


def require_manage_integrations(db: Session, user_id: str, pot_id: str) -> str:
    return require_manage_repos(db, user_id, pot_id)


def parse_assignable_role(value: str) -> str:
    """Return the canonical role string for API inputs; only ``user`` is assignable."""
    v = (value or "").strip().lower()
    if v == POT_ROLE_OWNER:
        raise HTTPException(
            status_code=400,
            detail="owner role cannot be assigned via this endpoint.",
        )
    if v not in ASSIGNABLE_POT_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of: {', '.join(ASSIGNABLE_POT_ROLES)}",
        )
    return v


# Backwards-compatible alias used elsewhere (parses any active role, not just
# the assignable set). Use :func:`parse_assignable_role` in member endpoints.
def parse_role(value: str) -> str:
    return parse_assignable_role(value)


__all__ = [
    "api_key_user_id",
    "normalize_role",
    "parse_assignable_role",
    "parse_role",
    "require_manage_integrations",
    "require_manage_members",
    "require_manage_repos",
    "require_pot_ingest",
    "require_pot_member",
    "require_pot_raw_ingest",
    "require_pot_reset",
    "user_role_on_context_graph_pot",
]


def api_key_user_id(user: dict[str, Any]) -> str:
    try:
        return str(user["user_id"])
    except KeyError as e:
        raise HTTPException(status_code=401, detail="Missing user_id on credentials") from e
