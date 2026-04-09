"""Resolve a user's role on a context-graph pot (membership or pot owner row)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_member_model import ContextGraphPotMember
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.pot_member_roles import (
    ALL_POT_ROLES,
    POT_ROLE_OWNER,
    can_ingest_or_reset,
    can_manage_members,
    can_manage_repos_and_integrations,
    can_query_context,
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
        return row.role
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
    role = require_pot_member(db, user_id, pot_id)
    if not can_ingest_or_reset(role):
        raise HTTPException(
            status_code=403,
            detail="This role cannot ingest or mutate context for this pot.",
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


def parse_role(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in ALL_POT_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of: {', '.join(ALL_POT_ROLES)}",
        )
    return v


def api_key_user_id(user: dict[str, Any]) -> str:
    try:
        return str(user["user_id"])
    except KeyError as e:
        raise HTTPException(status_code=401, detail="Missing user_id on credentials") from e
