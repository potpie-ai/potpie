"""Issue tracker read API (port) for Linear, Jira, etc."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, Protocol


class IssueTrackerPort(Protocol):
    """Provider-specific adapters implement this; orchestration uses it, not raw SDKs."""

    def iter_issues(
        self,
        *,
        scope: dict[str, Any],
        updated_after: datetime | None = None,
    ) -> Iterator[Any]:
        """Yield issues within the attachment scope (team/project/label filters in ``scope``)."""
        ...

    def get_issue(self, *, scope: dict[str, Any], issue_id: str) -> dict[str, Any]:
        """Fetch one issue by provider-native id (e.g. Linear internal id or key)."""
        ...

    def get_issue_comments(
        self,
        *,
        scope: dict[str, Any],
        issue_id: str,
    ) -> list[dict[str, Any]]:
        """Return comment thread for the issue (shape is provider-normalized by adapter)."""
        ...
