"""Host resolves opaque project_id to repo + credentials (port)."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ResolvedProject:
    project_id: str
    repo_name: str
    """When True, backfill / ingest may proceed."""
    ready: bool = True


class ProjectResolutionPort(Protocol):
    def resolve(self, project_id: str) -> ResolvedProject | None:
        ...
