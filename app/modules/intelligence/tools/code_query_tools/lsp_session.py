"""
Core session primitives shared between LSP manager and client implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.modules.intelligence.tools.code_query_tools.lsp_types import LspMethod


@dataclass(frozen=True)
class SessionKey:
    """Composite key identifying an active LSP session."""

    project_id: str
    language: str


class BaseLspServerSession:
    """
    Abstract interface for a running language server session.

    Concrete implementations should encapsulate the actual language server,
    expose lifecycle control (start/shutdown), and provide request hooks.
    """

    def __init__(self, session_key: SessionKey, workspace_root: str) -> None:
        self.session_key = session_key
        self.workspace_root = workspace_root

    async def ensure_started(self) -> None:
        raise NotImplementedError

    async def send_request(self, method: LspMethod, payload: dict) -> Any:
        raise NotImplementedError

    async def send_notification(self, method: str, payload: dict) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError
