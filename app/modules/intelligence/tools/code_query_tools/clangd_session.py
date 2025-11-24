"""
Clangd-based C/C++ code analysis session.

This module provides a clangd-powered implementation for C/C++ code analysis
that uses clangd's native cache feature for optimal performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.modules.intelligence.tools.code_query_tools.lsp_session import (
    BaseLspServerSession,
    SessionKey,
)
from app.modules.intelligence.tools.code_query_tools.lsp_types import (
    LspMethod,
)
from app.modules.intelligence.tools.code_query_tools.pygls_client_session import (
    LanguageServerConfig,
    PyglsClientSession,
)

logger = logging.getLogger(__name__)


class ClangdSession(BaseLspServerSession):
    """
    Clangd-based LSP session for C/C++ code analysis.

    This session uses clangd's native cache feature (background indexing)
    for optimal performance. No custom caching layer is used - we rely on
    clangd's built-in caching and indexing capabilities.
    """

    def __init__(
        self,
        session_key: SessionKey,
        workspace_root: str,
        cache_dir: Path,
        config: LanguageServerConfig,
    ) -> None:
        super().__init__(session_key, workspace_root)
        # Note: cache_dir is kept for compatibility but not used for custom caching
        # Clangd uses its native cache in .cache/clangd/index/ via symlink
        self._pygls_session = PyglsClientSession(
            session_key=session_key,
            workspace_root=workspace_root,
            config=config,
        )
        self._initialized = False

    async def ensure_started(self) -> None:
        """Ensure the underlying clangd session is started."""
        await self._pygls_session.ensure_started()
        self._initialized = True

    async def send_request(self, method: LspMethod, payload: dict) -> Any:
        """
        Handle LSP requests using clangd.

        All requests are passed directly to clangd, which uses its native
        cache and indexing for optimal performance.
        """
        # Pass all requests directly to clangd - it handles caching internally
        return await self._pygls_session.send_request(method, payload)

    async def send_notification(
        self, method: str, payload: Optional[dict] = None
    ) -> None:
        """Forward notifications to the underlying clangd session."""
        await self._pygls_session.send_notification(method, payload)

    async def shutdown(self) -> None:
        """Shutdown the underlying clangd session."""
        await self._pygls_session.shutdown()
        self._initialized = False
