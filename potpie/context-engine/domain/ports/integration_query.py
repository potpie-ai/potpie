"""Provider-neutral read interfaces for issue/workflow systems (ports)."""

from __future__ import annotations

from typing import Any, Protocol


class ArtifactQueryPort(Protocol):
    """Fetch artifacts (documents, attachments) by reference."""

    def fetch_artifact(self, ref: str) -> dict[str, Any]:
        ...


class IssueTrackerPort(Protocol):
    """Read issues/tickets from an external tracker."""

    def get_issue(self, issue_key: str) -> dict[str, Any]:
        ...


class WorkTrackingPort(Protocol):
    """Read workflow items (Linear/Jira-style)."""

    def get_work_item(self, item_id: str) -> dict[str, Any]:
        ...
