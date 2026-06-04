"""Role strings for context-graph pot membership (Potpie-owned tenancy).

Product model has exactly two active roles: ``owner`` and ``user``.
Legacy ``admin`` and ``read_only`` rows are migrated to ``user`` and must not
be returned by the API or exposed in the UI.
"""

from __future__ import annotations

POT_ROLE_OWNER = "owner"
POT_ROLE_USER = "user"

ALL_POT_ROLES: tuple[str, ...] = (POT_ROLE_OWNER, POT_ROLE_USER)
ASSIGNABLE_POT_ROLES: tuple[str, ...] = (POT_ROLE_USER,)


def normalize_role(role: str | None) -> str:
    """Map any persisted role value to the strict active set.

    Unknown or legacy values collapse to ``user`` so API responses never leak
    ``admin`` / ``read_only``. ``owner`` stays ``owner``.
    """
    v = (role or "").strip().lower()
    if v == POT_ROLE_OWNER:
        return POT_ROLE_OWNER
    return POT_ROLE_USER


def can_query_context(role: str) -> bool:
    return normalize_role(role) in ALL_POT_ROLES


def can_ingest_raw(role: str) -> bool:
    """Owners and users can manually submit raw notes, links, and content."""
    return normalize_role(role) in ALL_POT_ROLES


def can_reset(role: str) -> bool:
    return normalize_role(role) == POT_ROLE_OWNER


def can_manage_members(role: str) -> bool:
    return normalize_role(role) == POT_ROLE_OWNER


def can_manage_repos_and_integrations(role: str) -> bool:
    return normalize_role(role) == POT_ROLE_OWNER


# Backwards-compatible alias so callers that used the old name keep working
# during the migration; both point at the owner-only mutation rule.
def can_ingest_or_reset(role: str) -> bool:  # pragma: no cover - legacy alias
    return normalize_role(role) == POT_ROLE_OWNER
