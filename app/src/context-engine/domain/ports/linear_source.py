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

from typing import Any, Protocol


class LinearIssueFetcher(Protocol):
    """Fetch one Linear issue by its identifier.

    ``issue_id`` accepts either the canonical team-prefixed identifier
    (``ENG-123``) or the opaque Linear UUID — adapters normalize internally.
    ``None`` signals "not found"; raising is reserved for transport or auth
    errors so the resolver can translate them to fallback codes.
    """

    def get_issue(self, issue_id: str) -> dict[str, Any] | None:
        ...
