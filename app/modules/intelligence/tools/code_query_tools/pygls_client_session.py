"""
pygls-backed LSP client session implementation.

This module provides a concrete `BaseLspServerSession` that uses pygls'
`JsonRPCClient` to spawn and communicate with language servers over stdio.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from pygls.client import JsonRPCClient

from app.modules.intelligence.tools.code_query_tools.lsp_session import (
    BaseLspServerSession,
    SessionKey,
)
from app.modules.intelligence.tools.code_query_tools.lsp_types import LspMethod

logger = logging.getLogger(__name__)


@dataclass
class LanguageServerConfig:
    """Process-level configuration for launching a language server."""

    command: Sequence[str]
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int = 15
    index_wait_seconds: float = 0.0

    @classmethod
    def from_command_string(cls, command: str, **kwargs: Any) -> "LanguageServerConfig":
        return cls(command=shlex.split(command), **kwargs)


class PyglsClientSession(BaseLspServerSession):
    """Concrete LSP session powered by pygls' JsonRPCClient."""

    def __init__(
        self,
        session_key: SessionKey,
        workspace_root: str,
        config: LanguageServerConfig,
    ) -> None:
        super().__init__(session_key, workspace_root)
        self.config = config
        self._client = JsonRPCClient()
        self._startup_lock = asyncio.Lock()
        self._initialized = False
        # Track diagnostics to monitor when files are analyzed
        self._diagnostics_received: Dict[str, int] = {}  # uri -> count of diagnostics
        self._diagnostics_lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        if self._initialized:
            return

        async with self._startup_lock:
            if self._initialized:
                return

            if not self.config.command:
                raise RuntimeError("Language server command is not configured.")

            executable = self.config.command[0]
            if shutil.which(executable) is None:
                raise RuntimeError(
                    f"Language server executable '{executable}' is not available in PATH."
                )

            env = os.environ.copy()
            env.update(self.config.environment)

            # Set up diagnostics handler before starting the client
            # This allows us to track when Pyright finishes analyzing files
            async def handle_diagnostics(params: Dict[str, Any]) -> None:
                """Handle textDocument/publishDiagnostics notifications."""
                uri = params.get("uri", "")
                diagnostics = params.get("diagnostics", [])
                async with self._diagnostics_lock:
                    self._diagnostics_received[uri] = len(diagnostics)

            # Register diagnostics handler
            # Note: pygls client might not have a direct way to register notification handlers
            # We'll need to check the actual API

            await self._client.start_io(
                self.config.command[0],
                *self.config.command[1:],
                cwd=self.workspace_root,
                env=env,
            )

            # Try to register diagnostics handler after client is started
            # Note: pygls JsonRPCClient doesn't expose a direct way to register notification handlers
            # We'll use polling instead to detect when files are analyzed
            # The diagnostics tracking is set up but won't be used unless we find a way to register handlers

            workspace_uri = f"file://{self.workspace_root}"

            # Determine if this is clangd based on command
            is_clangd = "clangd" in self.config.command[0].lower()

            # Build capabilities - different for clangd vs other servers
            capabilities = {
                "textDocument": {
                    "publishDiagnostics": {
                        "categorySupport": True,
                        "codeActionsInline": True,
                    },
                    "completion": {"completionItem": {"snippetSupport": True}},
                },
            }

            if is_clangd:
                # For clangd: enable workDoneProgress for background indexing tracking
                # UTF-8 encoding provides 2-3x performance improvement over UTF-16
                capabilities["offsetEncoding"] = ["utf-8", "utf-16"]  # Prefer UTF-8
                capabilities["window"] = {
                    "workDoneProgress": True,  # Enable for clangd background indexing
                }
                capabilities["workspace"] = {
                    "workDoneProgress": True,
                }
            else:
                # For other servers (like Pyright): disable workDoneProgress
                # Pyright sends requests we can't handle properly
                capabilities["window"] = {
                    "workDoneProgress": False,
                }

            init_params = {
                "processId": os.getpid(),
                "clientInfo": {"name": "potpie-lsp-client", "version": "0.1"},
                "rootUri": workspace_uri,
                "rootPath": self.workspace_root,
                "workspaceFolders": [
                    {
                        "uri": workspace_uri,
                        "name": os.path.basename(self.workspace_root),
                    }
                ],
                "capabilities": capabilities,
                "initializationOptions": self.config.initialization_options,
            }

            future = self._client.protocol.send_request_async("initialize", init_params)
            try:
                init_response = await asyncio.wait_for(
                    future, timeout=self.config.timeout_seconds
                )

                # Log clangd-specific initialization response details
                # pygls returns a response object, access attributes directly
                if is_clangd and init_response:
                    try:
                        # Try to access as object attributes first
                        if hasattr(init_response, "result"):
                            result = init_response.result
                        elif hasattr(init_response, "get"):
                            # Fallback to dict-like access
                            result = init_response.get("result", {})
                        else:
                            result = init_response

                        # Access result data (could be dict or object)
                        if hasattr(result, "capabilities"):
                            capabilities = result.capabilities
                        elif isinstance(result, dict):
                            capabilities = result.get("capabilities", {})
                        else:
                            capabilities = {}

                        if hasattr(result, "offsetEncoding"):
                            offset_encoding = result.offsetEncoding
                        elif isinstance(result, dict):
                            offset_encoding = result.get("offsetEncoding", "unknown")
                        else:
                            offset_encoding = "unknown"

                        logger.info(
                            f"[LSP] Clangd initialized: offsetEncoding={offset_encoding}, "
                            f"capabilities={list(capabilities.keys()) if isinstance(capabilities, dict) else 'available'}"
                        )
                    except Exception as exc:
                        # Log but don't fail if we can't extract details
                        logger.debug(
                            f"[LSP] Could not extract clangd init details: {exc}"
                        )

            except asyncio.TimeoutError:
                # Detect language server from command
                command_name = (
                    self.config.command[0] if self.config.command else "unknown"
                )
                server_name = os.path.basename(command_name)

                # Log more details about the timeout
                logger.error(
                    f"[LSP] Initialize request timed out after {self.config.timeout_seconds}s. "
                    f"Command: {' '.join(self.config.command)}. "
                    f"Workspace: {self.workspace_root}. "
                    f"Check if {server_name} is installed and working correctly."
                )
                # Try to get stderr output if available (for debugging)
                if hasattr(self._client, "_transport") and hasattr(
                    self._client._transport, "stderr"
                ):
                    try:
                        # Note: pygls may not expose stderr directly, but we can log what we know
                        logger.error(
                            f"[LSP] {server_name} process may have failed to start or is hanging. "
                            f"Check {server_name} installation: which {server_name}"
                        )
                    except Exception:
                        pass
                raise

            self._client.protocol.notify("initialized", {})
            self._initialized = True

    async def wait_for_indexing_complete(
        self, timeout: float = 120.0, test_query_interval: float = 2.0
    ) -> bool:
        """
        Wait for indexing to complete by polling with test queries.

        This method sends periodic workspace/symbol queries and waits until
        we get consistent responses, indicating indexing is complete.

        Args:
            timeout: Maximum time to wait in seconds
            test_query_interval: Time between test queries in seconds

        Returns:
            True if indexing completed, False if timeout
        """
        from app.modules.intelligence.tools.code_query_tools.lsp_types import (
            LspMethod,
        )

        start_time = asyncio.get_event_loop().time()
        last_result_count = None
        stable_count = 0
        required_stable_checks = 2  # Need 2 consecutive same results
        min_symbols_for_completion = (
            1  # Require at least 1 symbol before considering complete
        )

        import logging

        logger = logging.getLogger(__name__)
        poll_count = 0

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                poll_count += 1
                elapsed = asyncio.get_event_loop().time() - start_time

                # Send a test query
                # Try with empty query first - Pyright should return all symbols
                # If that doesn't work, we might need to try with a specific query
                result = await self.send_request(
                    LspMethod.WORKSPACE_SYMBOL, {"query": ""}
                )

                # Count results
                result_count = len(result) if isinstance(result, list) else 0

                # Log progress every 10 polls (every 30 seconds with 3s interval)
                if poll_count % 10 == 0:
                    logger.info(
                        f"[LSP] Polling progress: {poll_count} polls, "
                        f"{elapsed:.1f}s elapsed, {result_count} symbols found"
                    )

                    # If we've been polling for a while with 0 results, try a different approach
                    if poll_count >= 20 and result_count == 0:
                        logger.warning(
                            f"[LSP] Still getting 0 symbols after {poll_count} polls. "
                            "Pyright may not be indexing properly. "
                            "This might indicate a configuration issue."
                        )

                # Only consider indexing complete if we have symbols AND the count is stable
                # This prevents false positives when Pyright returns 0 symbols initially
                if result_count >= min_symbols_for_completion:
                    if (
                        last_result_count is not None
                        and result_count == last_result_count
                    ):
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            logger.info(
                                f"[LSP] Indexing complete: {result_count} symbols found, "
                                f"stable for {required_stable_checks} checks, "
                                f"{poll_count} polls, {elapsed:.1f}s elapsed"
                            )
                            return (
                                True  # Indexing complete with stable non-zero results
                            )
                    else:
                        stable_count = 0
                        last_result_count = result_count
                else:
                    # Reset if we get 0 symbols - indexing hasn't started yet
                    stable_count = 0
                    last_result_count = None

                # Wait before next check
                await asyncio.sleep(test_query_interval)

            except Exception as exc:
                # If query fails, wait and retry
                import logging

                logger = logging.getLogger(__name__)
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.warning(
                    f"[LSP] Polling query failed (poll {poll_count}, {elapsed:.1f}s): {exc}"
                )
                await asyncio.sleep(test_query_interval)

        # Timeout reached
        import logging

        logger = logging.getLogger(__name__)
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.warning(
            f"[LSP] Indexing polling timeout after {poll_count} polls, {elapsed:.1f}s elapsed"
        )
        return False  # Timeout

    async def send_request(self, method: LspMethod, payload: dict) -> Any:
        if not self._initialized:
            await self.ensure_started()

        future = self._client.protocol.send_request_async(method.value, payload)
        return await asyncio.wait_for(future, timeout=self.config.timeout_seconds)

    async def send_notification(
        self, method: str, payload: Optional[dict] = None
    ) -> None:
        if not self._initialized:
            await self.ensure_started()
        self._client.protocol.notify(method, payload)

    async def shutdown(self) -> None:
        if not self._initialized:
            return

        try:
            future = self._client.protocol.send_request_async("shutdown")
            await asyncio.wait_for(future, timeout=self.config.timeout_seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            try:
                self._client.protocol.notify("exit")
            finally:
                await self._client.stop()
                self._initialized = False
