"""Read-only integration tools for reconciliation (port)."""

from __future__ import annotations

from typing import Any, Protocol

from domain.reconciliation import ReconciliationRequest


class ToolDescriptor:
    """Minimal tool metadata (library-owned; not LangChain-shaped)."""

    __slots__ = ("name", "category", "description", "json_schema")

    def __init__(
        self,
        *,
        name: str,
        category: str,
        description: str,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.category = category
        self.description = description
        self.json_schema = json_schema or {}


class ReconciliationToolsPort(Protocol):
    def list_tools(self, request: ReconciliationRequest) -> list[ToolDescriptor]:
        """Tools available for this run (read-only in v1)."""

    def execute_read_tool(
        self,
        request: ReconciliationRequest,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a named read-only tool."""
