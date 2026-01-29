"""Parsing-related type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from potpie.types.project import ProjectStatus


@dataclass
class ParsingResult:
    """Result of a parsing operation."""

    success: bool
    project_id: str
    status: ProjectStatus
    node_count: Optional[int] = None
    error_message: Optional[str] = None

    @classmethod
    def success_result(
        cls,
        project_id: str,
        node_count: Optional[int] = None,
    ) -> ParsingResult:
        """Create a successful parsing result."""
        return cls(
            success=True,
            project_id=project_id,
            status=ProjectStatus.READY,
            node_count=node_count,
            error_message=None,
        )

    @classmethod
    def error_result(
        cls,
        project_id: str,
        error_message: str,
    ) -> ParsingResult:
        """Create an error parsing result."""
        return cls(
            success=False,
            project_id=project_id,
            status=ProjectStatus.ERROR,
            node_count=None,
            error_message=error_message,
        )
