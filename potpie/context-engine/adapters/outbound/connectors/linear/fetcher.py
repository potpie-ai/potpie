"""Port for Linear source reads used by the Linear source resolver.

The context-engine does not own a Linear HTTP/GraphQL client — the existing
``integrations/`` module owns that. This port is the narrow surface the
``LinearIssueResolver`` depends on, so tests can use a fake and so hosts can
plug any adapter (OAuth-backed, service account, fixture) without dragging in
Celery or the integrations DB models.

Implementations return raw payloads; normalization is done by the caller via
:func:`domain.linear_events.linear_issue_from_payload`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class LinearIssueFetcher(Protocol):
    """Fetch one Linear issue by its identifier.

    ``issue_id`` accepts either the canonical team-prefixed identifier
    (``ENG-123``) or the opaque Linear UUID — adapters normalize internally.
    ``None`` signals "not found"; raising is reserved for transport or auth
    errors so the resolver can translate them to fallback codes.

    ``pot_id`` lets multi-tenant adapters pick the right credentials per
    call. Single-tenant adapters (e.g. an env-keyed fetcher) ignore it.
    """

    def get_issue(
        self,
        issue_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None: ...

    def list_issues(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate compact issue refs for a pot's connected Linear team.

        OPTIONAL capability. Single-issue resolvers / minimal fakes may omit
        it; the ``linear_list_issues`` backfill tool is only surfaced when the
        wired fetcher actually implements this (capability-checked via
        ``getattr``), so the agent never sees a list tool that can't run.

        Plain enumerator: ``updated_after`` / ``limit`` are applied as given —
        the backfill window/cap policy is computed by the caller. Returns
        ``{id, identifier, updated_at}`` dicts.
        """
        ...

    # --- OPTIONAL: project / document enumeration (backfill) --------------
    # Same capability-guarded contract as ``list_issues``: each is surfaced
    # as an agent tool only when the wired fetcher implements it. List
    # methods are plain enumerators (caller computes window/cap); ``get_*``
    # mirror ``get_issue`` (``None`` = not found; raise on auth/transport).

    def list_projects(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact project refs ``{id, name, updated_at}`` for the team."""
        ...

    def get_project(
        self,
        project_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        """One project's detail (name/description/state/lead/dates/teams)."""
        ...

    def list_documents(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact document refs ``{id, title, updated_at}`` for the team."""
        ...

    def get_document(
        self,
        document_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        """One document's detail (title/content/url/project/creator)."""
        ...
