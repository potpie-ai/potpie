"""Role strings for context-graph pot membership (Potpie-owned tenancy)."""

from __future__ import annotations

POT_ROLE_OWNER = "owner"
POT_ROLE_ADMIN = "admin"
POT_ROLE_READ_ONLY = "read_only"

ALL_POT_ROLES: tuple[str, ...] = (POT_ROLE_OWNER, POT_ROLE_ADMIN, POT_ROLE_READ_ONLY)


def can_query_context(role: str) -> bool:
    return role in ALL_POT_ROLES


def can_ingest_or_reset(role: str) -> bool:
    return role in (POT_ROLE_OWNER, POT_ROLE_ADMIN)


def can_manage_members(role: str) -> bool:
    return role == POT_ROLE_OWNER


def can_manage_repos_and_integrations(role: str) -> bool:
    return role in (POT_ROLE_OWNER, POT_ROLE_ADMIN)
