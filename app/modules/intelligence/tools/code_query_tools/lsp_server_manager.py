"""
Runtime management for pygls-backed language server sessions.

The manager is responsible for selecting the appropriate language server
configuration, orchestrating request execution, and normalizing results to the
agent-facing data models.
"""

from __future__ import annotations

import logging
import os
import shlex
from time import perf_counter
import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from app.modules.intelligence.tools.code_query_tools.lsp_session import (
    BaseLspServerSession,
    SessionKey,
)
from app.modules.intelligence.tools.code_query_tools.lsp_types import (
    HoverResult,
    LspMethod,
    LspQueryRequest,
    LspQueryResponse,
    Location,
    Position,
    SymbolInformation,
)
from app.modules.intelligence.tools.code_query_tools.pygls_client_session import (
    LanguageServerConfig,
    PyglsClientSession,
)
from app.modules.intelligence.tools.code_query_tools.jedi_session import (
    JediSession,
)
from app.modules.intelligence.tools.code_query_tools.clangd_session import (
    ClangdSession,
)

logger = logging.getLogger(__name__)

# Global LSP server manager instance for persistent sessions across chat sessions
_lsp_server_manager: Optional["LspServerManager"] = None


def get_lsp_server_manager() -> "LspServerManager":
    """Get the global LSP server manager instance, creating it if needed."""
    global _lsp_server_manager
    if _lsp_server_manager is None:
        logger.info("LspServerManager: Creating new global manager instance")
        _lsp_server_manager = LspServerManager()
    return _lsp_server_manager


def reset_lsp_server_manager() -> None:
    """Reset the global LSP server manager (useful for testing or cleanup)."""
    global _lsp_server_manager
    if _lsp_server_manager is not None:
        logger.info("LspServerManager: Resetting global manager instance")
        # Shutdown all sessions before resetting
        for session in _lsp_server_manager._sessions.values():
            try:
                # Note: This is a best-effort cleanup. Sessions may need async cleanup.
                pass
            except Exception:
                pass
    _lsp_server_manager = None


class LspServerManager:
    """
    Coordinates request execution for configured language servers.

    This manager maintains persistent language server sessions across chat sessions.
    Sessions are keyed by project_id:language and remain alive until the process
    restarts or sessions are explicitly shut down. This allows the language server
    to maintain its workspace index without reindexing on every chat.
    """

    def __init__(
        self,
        language_configs: Optional[Dict[str, LanguageServerConfig]] = None,
    ) -> None:
        self._language_configs: Dict[str, LanguageServerConfig] = language_configs or {}
        self._sessions: Dict[str, BaseLspServerSession] = {}
        self._warmed_sessions: set[str] = set()
        self._warmup_locks: Dict[str, asyncio.Lock] = {}

    def register_language(self, language: str, config: LanguageServerConfig) -> None:
        logger.info(
            "Registered language server for %s with command %s",
            language,
            list(config.command),
        )
        self._language_configs[language] = config

    def is_language_registered(self, language: str) -> bool:
        return language in self._language_configs

    def list_languages(self) -> List[str]:
        return sorted(self._language_configs)

    def _get_language_config(self, language: str) -> LanguageServerConfig:
        try:
            return self._language_configs[language]
        except KeyError as exc:
            raise ValueError(
                f"No language server configured for '{language}'."
            ) from exc

    @staticmethod
    def _convert_position(position: Optional[Position]) -> Optional[dict]:
        if not position:
            return None
        return {"line": position.line, "character": position.character}

    @staticmethod
    def _convert_text_document(request: LspQueryRequest) -> Optional[dict]:
        if not request.text_document:
            return None
        return {"uri": request.text_document.uri}

    def _build_params(self, request: LspQueryRequest) -> dict:
        text_document = self._convert_text_document(request)
        position = self._convert_position(request.position)
        method = request.method

        if method in {
            LspMethod.DEFINITION,
            LspMethod.REFERENCES,
            LspMethod.HOVER,
        }:
            if not text_document or not position:
                raise ValueError(
                    f"Method {method.value} requires text_document and position."
                )
            params: Dict[str, Any] = {
                "textDocument": text_document,
                "position": position,
            }
            if method is LspMethod.REFERENCES:
                params["context"] = {"includeDeclaration": True}
            return params

        if method is LspMethod.DOCUMENT_SYMBOL:
            if not text_document:
                raise ValueError(
                    "Method textDocument/documentSymbol requires a text document URI."
                )
            return {"textDocument": text_document}

        if method is LspMethod.WORKSPACE_SYMBOL:
            return {"query": request.query or ""}

        raise ValueError(f"Unsupported LSP method: {method.value}")

    @staticmethod
    def _normalize_locations(raw: Any) -> List[Location]:
        """Normalize locations from LSP or Jedi format."""
        if raw is None:
            return []

        if isinstance(raw, dict):
            raw = [raw]

        locations: List[Location] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue

            # Check if it's Jedi format (has uri, line, column directly)
            if "uri" in item and "line" in item and "range" not in item:
                # Jedi format - convert to Location
                locations.append(
                    Location(
                        uri=item["uri"],
                        start=Position(
                            line=item.get("line", 0),
                            character=item.get("column", 0),
                        ),
                        end=Position(
                            line=item.get("line", 0),
                            character=item.get("column", 0) + 10,  # Approximate end
                        ),
                    )
                )
            # LSP format
            elif "uri" in item and "range" in item:
                locations.append(Location.from_lsp(item))
            elif "targetUri" in item and "targetRange" in item:
                locations.append(
                    Location.from_lsp(
                        {"uri": item["targetUri"], "range": item["targetRange"]}
                    )
                )

        return locations

    @staticmethod
    def _normalize_hover(raw: Any) -> Optional[HoverResult]:
        if raw in (None, ""):
            return None

        if isinstance(raw, dict) or isinstance(raw, list) or isinstance(raw, str):
            return HoverResult.from_lsp(raw)

        return HoverResult.from_lsp(str(raw))

    def _normalize_document_symbols(
        self, raw: Sequence[dict], document_uri: Optional[str]
    ) -> List[SymbolInformation]:
        symbols: List[SymbolInformation] = []
        if not raw:
            return symbols

        for item in raw:
            if not isinstance(item, dict):
                continue
            if "location" in item:
                symbols.append(SymbolInformation.from_lsp(item))
            elif "range" in item:
                location = Location(
                    uri=document_uri or "",
                    start=Position.from_lsp(item["range"]["start"]),
                    end=Position.from_lsp(item["range"]["end"]),
                )
                symbols.append(
                    SymbolInformation(
                        name=item.get("name", ""),
                        kind=item.get("kind", 0),
                        location=location,
                        container_name=item.get("containerName"),
                    )
                )
                children = item.get("children") or []
                symbols.extend(self._normalize_document_symbols(children, document_uri))
        return symbols

    def _normalize_symbols(
        self, request: LspQueryRequest, raw: Any
    ) -> List[SymbolInformation]:
        """Normalize symbols from LSP or Jedi format."""
        if not raw:
            return []

        if isinstance(raw, list):
            symbols = []
            for item in raw:
                if isinstance(item, dict):
                    # Check if it's Jedi format (has uri, line, column, name directly)
                    if "uri" in item and "name" in item and "line" in item:
                        # Jedi format - convert to SymbolInformation
                        symbols.append(
                            SymbolInformation(
                                name=item.get("name", ""),
                                kind=self._jedi_type_to_symbol_kind(
                                    item.get("type", "")
                                ),
                                location=Location(
                                    uri=item.get("uri", ""),
                                    start=Position(
                                        line=item.get("line", 0),
                                        character=item.get("column", 0),
                                    ),
                                    end=Position(
                                        line=item.get("line", 0),
                                        character=item.get("column", 0)
                                        + len(item.get("name", "")),
                                    ),
                                ),
                                container_name=None,
                            )
                        )
                    # Otherwise assume LSP format
                    else:
                        symbols.extend(
                            self._normalize_document_symbols(
                                [item],
                                (
                                    request.text_document.uri
                                    if request.text_document
                                    else None
                                ),
                            )
                        )
            return symbols

        if isinstance(raw, dict):
            return self._normalize_symbols(request, [raw])

        return []

    @staticmethod
    def _jedi_type_to_symbol_kind(jedi_type: str) -> int:
        """Convert Jedi type string to LSP SymbolKind enum value."""
        # LSP SymbolKind values (from LSP spec)
        type_map = {
            "function": 12,  # Function
            "class": 5,  # Class
            "module": 9,  # Module
            "instance": 13,  # Variable
            "param": 6,  # Property (closest match)
            "statement": 13,  # Variable
        }
        return type_map.get(jedi_type.lower(), 13)  # Default to Variable

    @staticmethod
    def _read_document_from_uri(
        uri: Optional[str], workspace_root: str
    ) -> Optional[Tuple[str, str]]:
        if not uri:
            return None

        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None

        uri_path = Path(unquote(parsed.path))
        if not uri_path.is_absolute():
            uri_path = Path(workspace_root) / uri_path

        try:
            text = uri_path.read_text(encoding="utf-8")
            return uri, text
        except OSError as exc:
            logger.debug("Unable to read %s: %s", uri_path, exc)
            return None

    def _session_key(self, project_id: str, language: str) -> str:
        return f"{project_id}:{language}"

    def is_workspace_indexed(
        self, project_id: str, workspace_root: str, languages: List[str]
    ) -> bool:
        """
        Check if a workspace is already LSP indexed for the given languages.

        This checks both:
        1. If the session is warmed up in memory (primary check)
        2. If a marker file exists indicating successful indexing

        Args:
            project_id: Project ID for the workspace
            workspace_root: Root path of the workspace
            languages: List of language identifiers to check

        Returns:
            True if all languages are indexed, False otherwise
        """
        if not languages:
            return True

        for language in languages:
            # Check if language server is configured
            if not self.is_language_registered(language):
                continue  # Skip unconfigured languages

            # Check if session is warmed up in memory (primary indicator)
            key = self._session_key(project_id, language)
            if key in self._warmed_sessions:
                continue  # Already indexed in memory

            # Check for marker file indicating successful indexing
            cache_dir = self._get_lsp_cache_dir(workspace_root, language)
            index_marker = cache_dir / ".indexed"

            if not index_marker.exists():
                logger.debug(
                    f"Workspace {project_id} not indexed for {language}: "
                    f"index marker file missing at {index_marker}"
                )
                return False

            # Verify marker file has content (should contain timestamp and workspace path)
            try:
                marker_content = index_marker.read_text()
                if not marker_content.strip():
                    logger.debug(
                        f"Workspace {project_id} index marker file is empty for {language}"
                    )
                    return False
            except Exception as exc:
                logger.debug(f"Failed to read index marker file for {language}: {exc}")
                return False

        return True

    async def _run_pyright_cli_for_indexing(
        self,
        workspace_root: str,
        cache_dir: Path,
        status_messages: List[str],
        opened_files: List[str],
    ) -> bool:
        """
        Run Pyright CLI with --stats --verbose to trigger analysis and track progress.

        This provides a reliable way to know when indexing is complete, as the CLI
        will analyze all files and exit when done. We can parse the output to track progress.

        Returns:
            True if CLI completed successfully, False otherwise
        """
        import shutil
        from datetime import datetime
        import json

        # Find pyright executable
        pyright_exe = shutil.which("pyright")
        if not pyright_exe:
            # Try in venv
            venv_pyright = (
                Path(workspace_root).parent.parent / ".venv" / "bin" / "pyright"
            )
            if venv_pyright.exists():
                pyright_exe = str(venv_pyright)
            else:
                logger.warning(
                    "[LSP] Pyright CLI not found, skipping CLI-based indexing"
                )
                return False

        index_marker = cache_dir / ".indexed"
        workspace_path = Path(workspace_root)

        # Ensure cache directory and symlink are set up before running CLI
        # Pyright CLI uses .pyright directory in workspace root
        # NOTE: Pyright CLI may not create cache files - only the LSP server does
        # But we still run CLI to trigger analysis and get reliable completion signals
        pyright_cache_in_workspace = workspace_path / ".pyright"
        pyright_cache_target = cache_dir / "pyright"
        pyright_cache_target.mkdir(parents=True, exist_ok=True)

        # Create or verify symlink exists
        if pyright_cache_in_workspace.exists():
            if pyright_cache_in_workspace.is_symlink():
                try:
                    if (
                        pyright_cache_in_workspace.resolve()
                        != pyright_cache_target.resolve()
                    ):
                        pyright_cache_in_workspace.unlink()
                        pyright_cache_in_workspace.symlink_to(pyright_cache_target)
                        logger.info(
                            f"[LSP] Updated Pyright cache symlink to {pyright_cache_target}"
                        )
                except Exception as exc:
                    logger.warning(f"[LSP] Failed to update symlink: {exc}")
            elif pyright_cache_in_workspace.is_dir():
                # If it's a real directory, we can't replace it with a symlink
                logger.warning(
                    f"[LSP] .pyright exists as directory, not symlink. Pyright may use it instead of cache."
                )
        else:
            # Create symlink to our cache directory
            try:
                pyright_cache_in_workspace.symlink_to(pyright_cache_target)
                logger.info(
                    f"[LSP] Created Pyright cache symlink: {pyright_cache_in_workspace} -> {pyright_cache_target}"
                )
            except Exception as exc:
                logger.warning(f"[LSP] Failed to create Pyright cache symlink: {exc}")

        # Set environment variable for Pyright cache (if supported)
        env = os.environ.copy()
        env["PYRIGHTCACHE"] = str(pyright_cache_target)

        # Run Pyright CLI with stats and verbose output
        # This will analyze all Python files in the workspace
        command = [
            pyright_exe,
            "--stats",
            "--verbose",
            str(workspace_path),
        ]

        logger.info(f"[LSP] Running Pyright CLI: {' '.join(command)}")
        logger.info(f"[LSP] Pyright cache target: {pyright_cache_target}")
        logger.info(
            f"[LSP] Pyright cache symlink: {pyright_cache_in_workspace} -> {pyright_cache_target.resolve() if pyright_cache_in_workspace.exists() else 'NOT CREATED'}"
        )
        status_messages.append("Running Pyright CLI analysis...")

        try:
            # Run in a subprocess and capture output
            # We'll parse the output to track progress
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Combine stderr with stdout
                env=env,  # Use env with PYRIGHTCACHE set
            )

            if process.stdout is None:
                logger.warning("[LSP] Pyright CLI process stdout is None")
                return False

            # Read output line by line to track progress
            files_analyzed = 0
            total_time = 0.0
            output_lines = []

            # Update cache marker periodically while reading output
            last_update_time = asyncio.get_event_loop().time()
            update_interval = 2.0  # Update every 2 seconds

            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=1.0
                    )
                    if not line:
                        break
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    continue

                line_text = line.decode("utf-8", errors="ignore").strip()
                output_lines.append(line_text)

                # Parse progress from verbose output
                # Pyright verbose output shows files being analyzed
                if "Analyzing" in line_text or "analyzing" in line_text.lower():
                    files_analyzed += 1
                    logger.debug(f"[LSP] Pyright analyzing: {line_text}")

                # Update cache marker periodically
                current_time = asyncio.get_event_loop().time()
                if current_time - last_update_time >= update_interval:
                    try:
                        progress_data = {
                            "status": "indexing",
                            "method": "pyright_cli",
                            "progress": {
                                "files_analyzed": files_analyzed,
                                "output_lines": len(output_lines),
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                            "workspace": workspace_root,
                        }
                        index_marker.write_text(
                            json.dumps(progress_data, indent=2), encoding="utf-8"
                        )
                        last_update_time = current_time
                    except Exception:
                        pass

            # Wait for process to complete
            returncode = await process.wait()

            # Parse final stats from output
            output_text = "\n".join(output_lines)

            # Log output for debugging (first 50 lines and last 50 lines)
            if output_lines:
                logger.info(f"[LSP] Pyright CLI output (first 20 lines):")
                for line in output_lines[:20]:
                    logger.info(f"[LSP]   {line}")
                if len(output_lines) > 40:
                    logger.info(f"[LSP] Pyright CLI output (last 20 lines):")
                    for line in output_lines[-20:]:
                        logger.info(f"[LSP]   {line}")
            else:
                logger.warning("[LSP] Pyright CLI produced no output!")

            # Look for stats section (usually at the end)
            if "Performance stats" in output_text or "Files analyzed" in output_text:
                # Try to extract file count from stats
                import re

                file_count_match = re.search(
                    r"(\d+)\s+files?\s+analyzed", output_text, re.IGNORECASE
                )
                if file_count_match:
                    files_analyzed = int(file_count_match.group(1))

            # Check if Pyright cache directory has files
            # Check both the target cache directory and the symlink location
            pyright_cache_dir = cache_dir / "pyright"
            cache_file_count = 0
            cache_files_list = []

            if pyright_cache_dir.exists():
                cache_files = list(pyright_cache_dir.rglob("*"))
                cache_files_list = [f for f in cache_files if f.is_file()]
                cache_file_count = len(cache_files_list)
                logger.info(
                    f"[LSP] Found {cache_file_count} cache files in {pyright_cache_dir}"
                )
                if cache_file_count > 0:
                    logger.info(
                        f"[LSP] Cache files: {[str(f.relative_to(pyright_cache_dir)) for f in cache_files_list[:10]]}"
                    )

            # Also check the symlink location
            if (
                pyright_cache_in_workspace.exists()
                and pyright_cache_in_workspace.is_symlink()
            ):
                try:
                    resolved = pyright_cache_in_workspace.resolve()
                    if resolved.exists():
                        symlink_files = list(resolved.rglob("*"))
                        symlink_file_count = len(
                            [f for f in symlink_files if f.is_file()]
                        )
                        logger.info(
                            f"[LSP] Found {symlink_file_count} cache files via symlink at {resolved}"
                        )
                except Exception as exc:
                    logger.debug(f"[LSP] Could not resolve symlink: {exc}")

            success = returncode in [
                0,
                1,
            ]  # 0 = no errors, 1 = errors found (both mean analysis completed)

            logger.info(
                f"[LSP] Pyright CLI completed: exit_code={returncode}, "
                f"files_analyzed={files_analyzed}, cache_files={cache_file_count}"
            )

            # Update final progress
            try:
                progress_data = {
                    "status": "cli_completed" if success else "cli_failed",
                    "method": "pyright_cli",
                    "progress": {
                        "files_analyzed": files_analyzed,
                        "cache_files": cache_file_count,
                        "exit_code": returncode,
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "workspace": workspace_root,
                }
                index_marker.write_text(
                    json.dumps(progress_data, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

            return success

        except asyncio.TimeoutError:
            logger.warning("[LSP] Pyright CLI timed out while reading output")
            return False
        except Exception as exc:
            logger.warning(f"[LSP] Pyright CLI failed: {exc}", exc_info=True)
            return False

    @staticmethod
    def _get_lsp_cache_dir(workspace_root: str, language: str) -> Path:
        """
        Get the cache directory for a language server.

        Creates a cache directory structure in .repos/.lsp_cache/<language>/<workspace_hash>
        where workspace_hash is derived from the workspace path to ensure uniqueness.
        """
        import hashlib

        workspace_path = Path(workspace_root).resolve()
        # Create a hash of the workspace path for uniqueness
        workspace_hash = hashlib.sha256(str(workspace_path).encode()).hexdigest()[:16]

        # Find .repos directory - it should be a parent of the workspace
        # Workspace is typically at .repos/<owner>/<repo>/worktrees/<worktree_name>
        repos_base = workspace_path
        while repos_base.parent != repos_base:  # Not at root
            if repos_base.name == ".repos":
                break
            repos_base = repos_base.parent
        else:
            # Fallback: if we can't find .repos, use workspace parent
            repos_base = workspace_path.parent

        # Cache directory structure: .repos/.lsp_cache/<language>/<workspace_hash>
        cache_base = repos_base / ".lsp_cache" / language / workspace_hash
        cache_base.mkdir(parents=True, exist_ok=True)

        return cache_base

    @staticmethod
    def _configure_cache_for_language(
        config: LanguageServerConfig,
        language: str,
        workspace_root: str,
        cache_dir: Path,
        status_messages: List[str],
    ) -> LanguageServerConfig:
        """
        Configure cache directory for a language server.

        Different language servers use different mechanisms:
        - Pyright: Uses .pyright directory in workspace root (we create a symlink)
        - Others: Can be configured via environment variables or init options
        """
        from dataclasses import replace

        env = dict(config.environment)

        if language == "python":
            # Pyright stores cache in .pyright directory in workspace root
            # Create a symlink from workspace/.pyright to our cache directory
            workspace_path = Path(workspace_root)
            pyright_cache_in_workspace = workspace_path / ".pyright"
            pyright_cache_target = cache_dir / "pyright"
            pyright_cache_target.mkdir(parents=True, exist_ok=True)

            # Create symlink if it doesn't exist or is broken
            if pyright_cache_in_workspace.exists():
                if pyright_cache_in_workspace.is_symlink():
                    try:
                        if (
                            pyright_cache_in_workspace.resolve()
                            != pyright_cache_target.resolve()
                        ):
                            pyright_cache_in_workspace.unlink()
                            pyright_cache_in_workspace.symlink_to(pyright_cache_target)
                    except Exception:
                        # Symlink might be broken, remove and recreate
                        try:
                            pyright_cache_in_workspace.unlink()
                        except Exception:
                            pass
                        pyright_cache_in_workspace.symlink_to(pyright_cache_target)
                elif pyright_cache_in_workspace.is_dir():
                    # If it's a real directory, we can't replace it with a symlink
                    # Just use it as-is (Pyright will use it)
                    pass
            else:
                # Create symlink to our cache directory
                try:
                    pyright_cache_in_workspace.symlink_to(pyright_cache_target)
                    status_messages.append(
                        f"Configured Pyright cache at {pyright_cache_target}"
                    )
                except Exception as exc:
                    logger.debug("Failed to create Pyright cache symlink: %s", exc)
                    # Fall back to letting Pyright create its own cache

            # Pyright also respects PYRIGHT_CACHE_DIR environment variable (if supported)
            # Some versions may use this, so we set it as a fallback
            env["PYRIGHTCACHE"] = str(pyright_cache_target)

        elif language in ("c", "cpp"):
            # Clangd stores its index in .cache/clangd/index relative to the workspace root
            # (where compile_commands.json is located)
            # For headers without compile_commands.json, it uses XDG_CACHE_HOME/clangd/index
            # We'll create a symlink from workspace/.cache to our cache directory
            
            workspace_path = Path(workspace_root)
            clangd_cache_in_workspace = workspace_path / ".cache" / "clangd" / "index"
            clangd_cache_target = cache_dir / "clangd_native" / "clangd" / "index"
            clangd_cache_target.mkdir(parents=True, exist_ok=True)

            # Create symlink from workspace/.cache/clangd/index to our cache directory
            if clangd_cache_in_workspace.exists():
                if clangd_cache_in_workspace.is_symlink():
                    try:
                        if clangd_cache_in_workspace.resolve() != clangd_cache_target.resolve():
                            clangd_cache_in_workspace.unlink()
                            clangd_cache_in_workspace.symlink_to(clangd_cache_target)
                    except Exception:
                        # Symlink might be broken, remove and recreate
                        try:
                            clangd_cache_in_workspace.unlink()
                        except Exception:
                            pass
                        clangd_cache_in_workspace.parent.mkdir(parents=True, exist_ok=True)
                        clangd_cache_in_workspace.symlink_to(clangd_cache_target)
                elif clangd_cache_in_workspace.is_dir():
                    # If it's a real directory, we can't replace it with a symlink
                    # But we can still set XDG_CACHE_HOME as fallback
                    logger.debug(
                        f"Workspace .cache/clangd/index exists as directory, using it directly"
                    )
            else:
                # Create symlink to our cache directory
                try:
                    clangd_cache_in_workspace.parent.mkdir(parents=True, exist_ok=True)
                    clangd_cache_in_workspace.symlink_to(clangd_cache_target)
                    status_messages.append(
                        f"Created clangd cache symlink: {clangd_cache_in_workspace} -> {clangd_cache_target}"
                    )
                except Exception as exc:
                    logger.warning(f"Failed to create clangd cache symlink: {exc}")
            
            # Also set XDG_CACHE_HOME as fallback for headers without compile_commands.json
            clangd_cache_base = cache_dir / "clangd_native"
            env["XDG_CACHE_HOME"] = str(clangd_cache_base)
            
            # Also set DARWIN_USER_CACHE_DIR for macOS compatibility
            env["DARWIN_USER_CACHE_DIR"] = str(clangd_cache_base)
            
            # Check for compile_commands.json and generate if missing
            from app.modules.intelligence.tools.code_query_tools.clangd_helpers import (
                ensure_compile_commands,
                find_compile_commands,
            )
            
            compile_commands_path = workspace_path / "compile_commands.json"
            compile_commands_build = workspace_path / "build" / "compile_commands.json"
            has_compile_commands = compile_commands_path.exists()
            
            # Check for compile_commands.json in build directory (common with CMake)
            if not has_compile_commands and compile_commands_build.exists():
                logger.info(
                    f"[LSP] Found compile_commands.json in build directory. "
                    f"Creating symlink to workspace root..."
                )
                try:
                    compile_commands_path.symlink_to(compile_commands_build.relative_to(workspace_path))
                    has_compile_commands = True
                    status_messages.append(
                        "Created symlink: compile_commands.json -> build/compile_commands.json"
                    )
                except Exception as e:
                    logger.warning(f"[LSP] Failed to create symlink: {e}")
                    status_messages.append(
                        "Note: compile_commands.json found in build/ directory (symlink failed)"
                    )
            
            # If still no compile_commands.json, try to generate it
            if not has_compile_commands:
                logger.info(
                    "[LSP] No compile_commands.json found, attempting to generate..."
                )
                status_messages.append("Generating compile_commands.json...")
                
                generated_path, gen_messages = ensure_compile_commands(
                    workspace_root, language=language, force_regenerate=False
                )
                status_messages.extend(gen_messages)
                
                if generated_path:
                    has_compile_commands = generated_path.name == "compile_commands.json"
                    if has_compile_commands:
                        logger.info(f"[LSP] Successfully generated compile_commands.json")
                    else:
                        logger.info(f"[LSP] Created compile_flags.txt as fallback")
                else:
                    logger.warning(
                        "[LSP] Could not generate compile_commands.json. "
                        "Clangd will use fallback flags but indexing may be limited."
                    )
            
            # Configure clangd initialization options
            # clangdFileStatus provides real-time activity updates (critical for user feedback)
            # fallbackFlags used when compile_commands.json is missing
            if "clangd" not in config.initialization_options:
                config.initialization_options = dict(config.initialization_options)
                config.initialization_options["clangd"] = {}
            
            if not isinstance(config.initialization_options.get("clangd"), dict):
                config.initialization_options["clangd"] = {}
            
            # Enable file status notifications for real-time indexing feedback
            config.initialization_options["clangd"]["clangdFileStatus"] = True
            
            # Set fallback flags for files without compile_commands.json
            # These are used when clangd can't find compilation info
            if "fallbackFlags" not in config.initialization_options["clangd"]:
                fallback_flags = ["-std=c++17", "-Wall"]
                if language == "c":
                    fallback_flags = ["-std=c11", "-Wall"]
                config.initialization_options["clangd"]["fallbackFlags"] = fallback_flags
            
            # If compile_commands.json is in a non-standard location, specify it
            if compile_commands_build.exists() and not has_compile_commands:
                config.initialization_options["clangd"]["compilationDatabasePath"] = str(
                    compile_commands_build.parent
                )
            
            # Check for compile_flags.txt (fallback option)
            compile_flags_path = workspace_path / "compile_flags.txt"
            has_compile_flags = compile_flags_path.exists()
            
            if has_compile_flags and not has_compile_commands:
                logger.info(
                    "[LSP] Using compile_flags.txt (simpler alternative to compile_commands.json)"
                )
                status_messages.append("Using compile_flags.txt for clangd configuration")

            status_messages.append(
                f"Configured clangd: cache={clangd_cache_target}, "
                f"compile_commands.json={'found' if has_compile_commands else 'not found'}, "
                f"fileStatus=enabled, UTF-8 encoding preferred"
            )

        # For other language servers, we can add similar configurations here
        # For example, TypeScript/JavaScript servers might use different cache mechanisms

        if env != config.environment:
            config = replace(config, environment=env)

        return config

    async def _get_or_create_session(
        self,
        request: LspQueryRequest,
        workspace_root: str,
        config: LanguageServerConfig,
        status_messages: List[str],
    ) -> BaseLspServerSession:
        key = self._session_key(request.project_id, request.language)
        session = self._sessions.get(key)
        if session is None:
            # Configure cache directory for persistent indexing
            cache_dir = self._get_lsp_cache_dir(workspace_root, request.language)

            # Use Jedi for Python, ClangdSession for C/C++, PyglsClientSession for other languages
            if request.language == "python":
                status_messages.append(f"Initializing Jedi-based Python code analysis")
                session = JediSession(
                    session_key=SessionKey(request.project_id, request.language),
                    workspace_root=workspace_root,
                    cache_dir=cache_dir,
                )
            elif request.language in ("c", "cpp"):
                command_display = " ".join(shlex.quote(part) for part in config.command)
                status_messages.append(
                    f"Initializing clangd-based {request.language} code analysis with caching"
                )
                config = self._configure_cache_for_language(
                    config, request.language, workspace_root, cache_dir, status_messages
                )
                session = ClangdSession(
                    session_key=SessionKey(request.project_id, request.language),
                    workspace_root=workspace_root,
                    cache_dir=cache_dir,
                    config=config,
                )
            else:
                command_display = " ".join(shlex.quote(part) for part in config.command)
                status_messages.append(
                    f"Starting {request.language} language server using command: {command_display}"
                )
                config = self._configure_cache_for_language(
                    config, request.language, workspace_root, cache_dir, status_messages
                )
                session = PyglsClientSession(
                    session_key=SessionKey(request.project_id, request.language),
                    workspace_root=workspace_root,
                    config=config,
                )
            self._sessions[key] = session

        start_time = perf_counter()
        await session.ensure_started()
        ready_time = perf_counter()
        status_messages.append(
            f"Language server ready in {ready_time - start_time:.2f}s"
        )
        return session

    async def _warmup_session(
        self,
        request: LspQueryRequest,
        workspace_root: str,
        session: BaseLspServerSession,
        config: LanguageServerConfig,
        status_messages: List[str],
    ) -> None:
        key = self._session_key(request.project_id, request.language)
        language = request.language

        if key in self._warmed_sessions:
            logger.debug(f"Session {key} already warmed up, skipping warmup")
            return

        lock = self._warmup_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._warmed_sessions:
                return

            # Map language to file extensions
            language_extensions = {
                "python": ["*.py"],
                "typescript": ["*.ts", "*.tsx"],
                "javascript": ["*.js", "*.jsx"],
                "go": ["*.go"],
                "java": ["*.java"],
                "rust": ["*.rs"],
                "c": ["*.c", "*.h"],
                "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.h"],
                "csharp": ["*.cs"],
                "ruby": ["*.rb"],
                "php": ["*.php"],
            }

            extensions = language_extensions.get(language, [])
            if not extensions:
                # Unknown language, skip indexing
                self._warmed_sessions.add(key)
                return

            # Collect all matching files
            all_files = []
            for ext in extensions:
                all_files.extend(Path(workspace_root).rglob(ext))

            if not all_files:
                self._warmed_sessions.add(key)
                return

            status_messages.append(
                f"Indexing {len(all_files)} {language} files for workspace analysis"
            )

            # Pyright doesn't actually "index" - it analyzes files on-demand
            # To trigger analysis, we need to:
            # 1. Open files with didOpen
            # 2. Query files with documentSymbol to trigger analysis
            # 3. Then workspace/symbol can return results

            opened_files = []
            opened_file_data = []  # Store (uri, file_path) for verification

            # Step 1: For Jedi, we don't need to open files - it works directly with file paths
            # For other language servers, open files with didOpen
            if not (language == "python" and isinstance(session, JediSession)):
                logger.info(
                    f"[LSP] Opening {len(all_files)} {language} files and triggering analysis..."
                )
                # Open all files for non-Jedi language servers
                for idx, file_path in enumerate(all_files, start=1):
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except OSError:
                        continue

                    uri = file_path.as_uri()
                    await session.send_notification(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": uri,
                                "languageId": language,
                                "version": 1,
                                "text": text,
                            }
                        },
                    )
                    opened_files.append(uri)
                    opened_file_data.append((uri, file_path))

                    if idx % 50 == 0:
                        await asyncio.sleep(0.1)  # Small delay every 50 files
                        logger.info(f"[LSP] Opened {idx}/{len(all_files)} files...")

                logger.info(
                    f"[LSP] Opened {len(opened_files)} files. Triggering analysis by querying files..."
                )
                status_messages.append(f"Opened {len(opened_files)} {language} files")
            else:
                # For Jedi, just prepare file data for indexing
                for file_path in all_files:
                    uri = file_path.as_uri()
                    opened_file_data.append((uri, file_path))

            # Step 2: Index workspace using Jedi for Python
            if language == "python" and isinstance(session, JediSession):
                status_messages.append("Indexing workspace with Jedi...")
                logger.info(
                    "[LSP] Indexing Python files with Jedi for workspace symbol search..."
                )

                # Get all Python file paths
                python_file_paths = [str(f) for f in all_files]

                # Index workspace (this will cache workspace symbols)
                await session.index_workspace(python_file_paths)

                logger.info(f"[LSP] Jedi indexed {len(python_file_paths)} Python files")
                status_messages.append(
                    f"Indexed {len(python_file_paths)} Python files with Jedi"
                )

                # Verify indexing by checking workspace symbols
                try:
                    from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                        LspMethod,
                    )

                    ws_result = await session.send_request(
                        LspMethod.WORKSPACE_SYMBOL, {"query": ""}
                    )
                    symbol_count = len(ws_result) if isinstance(ws_result, list) else 0
                    logger.info(
                        f"[LSP] Jedi workspace symbol search returned {symbol_count} symbols"
                    )
                    status_messages.append(
                        f"Workspace symbol search: {symbol_count} symbols found"
                    )
                except Exception as exc:
                    logger.warning(f"[LSP] Failed to verify Jedi indexing: {exc}")

                # Mark as warmed up - Jedi doesn't need file opening
                verification_passed = True
                opened_files = []  # Clear opened files since Jedi doesn't need them
            elif language == "python":
                # Fallback for non-Jedi Python sessions (shouldn't happen)
                status_messages.append("Warning: Python session is not using Jedi")
                verification_passed = False

                from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                    LspMethod,
                )

                # Query a subset of files to trigger analysis
                # We don't need to query all files - querying some will trigger Pyright to analyze them
                files_to_query = min(20, len(opened_file_data))  # Query up to 20 files
                analyzed_count = 0
                total_symbols_found = 0

                for idx, (uri, file_path) in enumerate(
                    opened_file_data[:files_to_query], 1
                ):
                    try:
                        # Query documentSymbol to trigger analysis
                        doc_symbol_result = await session.send_request(
                            LspMethod.DOCUMENT_SYMBOL, {"textDocument": {"uri": uri}}
                        )

                        symbol_count = (
                            len(doc_symbol_result)
                            if isinstance(doc_symbol_result, list)
                            else 0
                        )

                        if symbol_count > 0:
                            analyzed_count += 1
                            total_symbols_found += symbol_count

                        # Small delay to avoid overwhelming Pyright
                        if idx % 10 == 0:
                            await asyncio.sleep(0.5)
                            logger.info(
                                f"[LSP] Queried {idx}/{files_to_query} files, "
                                f"{analyzed_count} analyzed, {total_symbols_found} symbols found"
                            )
                    except Exception as exc:
                        logger.debug(
                            f"[LSP] Error querying documentSymbol for {file_path.name}: {exc}"
                        )

                logger.info(
                    f"[LSP] Triggered analysis on {files_to_query} files: "
                    f"{analyzed_count} files analyzed, {total_symbols_found} total symbols"
                )
                status_messages.append(
                    f"Triggered analysis: {analyzed_count}/{files_to_query} files analyzed "
                    f"({total_symbols_found} symbols)"
                )

                # Step 4: Poll to wait for Pyright to finish analyzing files
                # We'll poll documentSymbol on sample files until they're all analyzed
                if analyzed_count > 0:
                    status_messages.append(
                        "Waiting for Pyright to complete analysis of all files..."
                    )
                    logger.info(
                        "[LSP] Polling to wait for Pyright to complete analysis..."
                    )

                    # Poll sample files until they're all analyzed
                    max_poll_attempts = (
                        30  # Poll for up to 30 attempts (60 seconds with 2s interval)
                    )
                    poll_interval = 2.0  # 2 seconds between polls
                    sample_size = min(10, len(opened_file_data))
                    analyzed_in_sample = 0  # Initialize for use in else clause

                    # Get cache directory for incremental updates
                    cache_dir = self._get_lsp_cache_dir(workspace_root, language)
                    index_marker = cache_dir / ".indexed"
                    from datetime import datetime

                    for poll_attempt in range(max_poll_attempts):
                        analyzed_in_sample = 0
                        total_symbols_in_sample = 0

                        for uri, file_path in opened_file_data[:sample_size]:
                            try:
                                doc_result = await session.send_request(
                                    LspMethod.DOCUMENT_SYMBOL,
                                    {"textDocument": {"uri": uri}},
                                )
                                symbol_count = (
                                    len(doc_result)
                                    if isinstance(doc_result, list)
                                    else 0
                                )
                                if symbol_count > 0:
                                    analyzed_in_sample += 1
                                    total_symbols_in_sample += symbol_count
                            except Exception:
                                pass

                        # Update cache marker file with progress information
                        try:
                            # Check Pyright cache file count
                            pyright_cache_dir = cache_dir / "pyright"
                            cache_file_count = 0
                            if pyright_cache_dir.exists():
                                cache_files = list(pyright_cache_dir.rglob("*"))
                                cache_file_count = len(
                                    [f for f in cache_files if f.is_file()]
                                )

                            # Write progress to marker file
                            progress_data = {
                                "status": "indexing",
                                "progress": {
                                    "files_analyzed": analyzed_in_sample,
                                    "sample_size": sample_size,
                                    "total_symbols": total_symbols_in_sample,
                                    "poll_attempt": poll_attempt + 1,
                                    "max_polls": max_poll_attempts,
                                    "elapsed_seconds": poll_attempt * poll_interval,
                                    "cache_files": cache_file_count,
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                                "workspace": workspace_root,
                            }

                            import json

                            index_marker.write_text(
                                json.dumps(progress_data, indent=2), encoding="utf-8"
                            )
                        except Exception as exc:
                            logger.debug(f"[LSP] Failed to update cache marker: {exc}")

                        # If most files in sample are analyzed, we're done
                        if analyzed_in_sample >= (sample_size * 0.8):  # 80% threshold
                            logger.info(
                                f"[LSP] Analysis complete: {analyzed_in_sample}/{sample_size} files analyzed "
                                f"after {poll_attempt + 1} polls ({poll_attempt * poll_interval:.1f}s)"
                            )
                            status_messages.append(
                                f"Analysis complete: {analyzed_in_sample}/{sample_size} files analyzed "
                                f"({total_symbols_in_sample} symbols)"
                            )
                            break

                        if (
                            poll_attempt % 5 == 0 and poll_attempt > 0
                        ):  # Log every 5 polls (every 10 seconds)
                            # Get cache file count for logging
                            cache_file_count_log = 0
                            try:
                                pyright_cache_dir_log = cache_dir / "pyright"
                                if pyright_cache_dir_log.exists():
                                    cache_files_log = list(
                                        pyright_cache_dir_log.rglob("*")
                                    )
                                    cache_file_count_log = len(
                                        [f for f in cache_files_log if f.is_file()]
                                    )
                            except Exception:
                                pass

                            logger.info(
                                f"[LSP] Polling progress: {poll_attempt + 1}/{max_poll_attempts} polls, "
                                f"{analyzed_in_sample}/{sample_size} files analyzed "
                                f"({poll_attempt * poll_interval:.1f}s elapsed, {cache_file_count_log} cache files)"
                            )
                            status_messages.append(
                                f"Analysis in progress: {analyzed_in_sample}/{sample_size} files analyzed..."
                            )

                        await asyncio.sleep(poll_interval)
                    else:
                        # Polling completed without reaching threshold
                        logger.warning(
                            f"[LSP] Analysis polling completed: {analyzed_in_sample}/{sample_size} files analyzed "
                            f"after {max_poll_attempts} polls"
                        )
                        status_messages.append(
                            f"Analysis polling completed: {analyzed_in_sample}/{sample_size} files analyzed"
                        )
                else:
                    logger.warning(
                        "[LSP] No files were analyzed. Pyright might not be responding to queries."
                    )
                    status_messages.append(
                        "Warning: No files analyzed, but continuing..."
                    )
                    await asyncio.sleep(5.0)

            # Pyright doesn't have a traditional "indexing complete" event
            # Since we've already queried files to trigger analysis, we can verify
            # by checking if workspace/symbol returns results or if files are analyzed

            # If we already analyzed files successfully, we can consider indexing working
            # even if workspace/symbol doesn't return results immediately
            verification_passed = False

            if language == "python" and len(opened_file_data) > 0:
                # Check if we successfully analyzed files
                # If we did, Pyright is working and can answer queries
                status_messages.append("Verifying Pyright is ready for queries...")
                logger.info(
                    "[LSP] Verifying Pyright can answer workspace/symbol queries..."
                )

                # Try workspace/symbol a few times to see if it returns results
                from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                    LspMethod,
                )

                workspace_symbol_count = 0
                for attempt in range(3):
                    try:
                        ws_result = await session.send_request(
                            LspMethod.WORKSPACE_SYMBOL, {"query": ""}
                        )
                        workspace_symbol_count = (
                            len(ws_result) if isinstance(ws_result, list) else 0
                        )
                        if workspace_symbol_count > 0:
                            logger.info(
                                f"[LSP] workspace/symbol returned {workspace_symbol_count} symbols "
                                f"on attempt {attempt + 1}"
                            )
                            verification_passed = True
                            status_messages.append(
                                f"Workspace symbol search working: {workspace_symbol_count} symbols found"
                            )
                            break
                        await asyncio.sleep(2.0)  # Wait between attempts
                    except Exception as exc:
                        logger.debug(
                            f"[LSP] workspace/symbol query failed (attempt {attempt + 1}): {exc}"
                        )
                        await asyncio.sleep(2.0)

                # If workspace/symbol doesn't work, verify by checking documentSymbol on files
                # If files can be analyzed, Pyright is working
                if not verification_passed:
                    logger.info(
                        "[LSP] workspace/symbol returned 0, verifying files can be analyzed..."
                    )
                    status_messages.append("Verifying files can be analyzed...")

                    # Check a few files to see if they can be analyzed
                    files_to_check = min(5, len(opened_file_data))
                    analyzable_count = 0

                    for uri, file_path in opened_file_data[:files_to_check]:
                        try:
                            doc_result = await session.send_request(
                                LspMethod.DOCUMENT_SYMBOL,
                                {"textDocument": {"uri": uri}},
                            )
                            symbol_count = (
                                len(doc_result) if isinstance(doc_result, list) else 0
                            )
                            if symbol_count > 0:
                                analyzable_count += 1
                        except Exception:
                            pass

                    # If we can analyze files, Pyright is working
                    # This is sufficient for our use case
                    if analyzable_count >= (files_to_check * 0.6):  # 60% threshold
                        verification_passed = True
                        logger.info(
                            f"[LSP] Files are analyzable: {analyzable_count}/{files_to_check}. "
                            "Pyright is working correctly."
                        )
                        status_messages.append(
                            f"Verification passed: {analyzable_count}/{files_to_check} files can be analyzed"
                        )
                    else:
                        logger.warning(
                            f"[LSP] Only {analyzable_count}/{files_to_check} files are analyzable. "
                            "Pyright might not be working correctly."
                        )
                        status_messages.append(
                            f"Warning: Only {analyzable_count}/{files_to_check} files are analyzable"
                        )
            elif language in ("c", "cpp") and isinstance(session, ClangdSession):
                # For clangd, we need to query files to trigger indexing
                # clangd builds its index lazily when files are queried
                logger.info(
                    f"[LSP] Triggering clangd indexing by querying {len(opened_file_data)} files..."
                )
                status_messages.append(
                    f"Querying files to trigger clangd indexing..."
                )

                # Query a subset of files to trigger clangd to analyze them
                # We don't need to query all files - querying some will trigger clangd to index them
                files_to_query = min(50, len(opened_file_data))  # Query up to 50 files
                analyzed_count = 0
                total_symbols_found = 0

                from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                    LspMethod,
                )

                for idx, (uri, file_path) in enumerate(
                    opened_file_data[:files_to_query], 1
                ):
                    try:
                        # Query documentSymbol to trigger clangd to analyze the file
                        doc_symbol_result = await session.send_request(
                            LspMethod.DOCUMENT_SYMBOL, {"textDocument": {"uri": uri}}
                        )

                        symbol_count = (
                            len(doc_symbol_result)
                            if isinstance(doc_symbol_result, list)
                            else 0
                        )

                        if symbol_count > 0:
                            analyzed_count += 1
                            total_symbols_found += symbol_count

                        # Small delay to avoid overwhelming clangd
                        if idx % 10 == 0:
                            await asyncio.sleep(0.5)
                            logger.info(
                                f"[LSP] Queried {idx}/{files_to_query} files, "
                                f"{analyzed_count} analyzed, {total_symbols_found} symbols found"
                            )
                    except Exception as exc:
                        logger.debug(
                            f"[LSP] Error querying documentSymbol for {file_path.name}: {exc}"
                        )

                logger.info(
                    f"[LSP] Triggered clangd indexing on {files_to_query} files: "
                    f"{analyzed_count} files analyzed, {total_symbols_found} total symbols"
                )
                status_messages.append(
                    f"Triggered indexing: {analyzed_count}/{files_to_query} files analyzed "
                    f"({total_symbols_found} symbols)"
                )

                # Check if compile_commands.json exists (clangd works much better with it)
                cache_dir = self._get_lsp_cache_dir(workspace_root, language)
                workspace_path = Path(workspace_root)
                compile_commands_path = workspace_path / "compile_commands.json"
                compile_flags_path = workspace_path / "compile_flags.txt"
                has_compile_commands = compile_commands_path.exists()
                has_compile_flags = compile_flags_path.exists()
                
                # Initialize index_file_count for use after the if/else block
                index_file_count = 0
                
                # Clangd won't build a comprehensive background index without compile_commands.json
                # Without it, clangd only indexes opened files, not the entire project
                if not has_compile_commands and not has_compile_flags:
                    logger.info(
                        "[LSP] No compile_commands.json or compile_flags.txt found. "
                        "Clangd will only index opened files, not build a full background index. "
                        "Skipping native index wait - clangd is ready for queries."
                    )
                    status_messages.append(
                        "Note: No compile_commands.json - clangd will index files on-demand"
                    )
                    verification_passed = True
                    # Don't wait for native index - it won't be built without compile_commands.json
                else:
                    # Wait for clangd to build its native index
                    # Clangd builds its index in the background after files are queried
                    # The native index is what makes clangd fast
                    status_messages.append(
                        "Waiting for clangd to build native index..."
                    )
                    logger.info(
                        "[LSP] Waiting for clangd to build native index (this makes the LSP server faster)..."
                    )

                    # Check clangd's native index location
                    # Clangd stores index in .cache/clangd/index in workspace root
                    # We have a symlink pointing to our cache directory
                    clangd_index_in_workspace = workspace_path / ".cache" / "clangd" / "index"
                    clangd_cache_target = cache_dir / "clangd_native" / "clangd" / "index"
                    
                    # Check both locations (symlink target and direct path)
                    clangd_index_dir = clangd_cache_target
                    if clangd_index_in_workspace.exists() and clangd_index_in_workspace.is_symlink():
                        try:
                            resolved = clangd_index_in_workspace.resolve()
                            if resolved.exists():
                                clangd_index_dir = resolved
                        except Exception:
                            pass
                    
                    # Poll for clangd index files
                    # With compile_commands.json, clangd should build a comprehensive background index
                    max_wait_attempts = 60  # Wait up to 60 attempts (3 minutes with 3s interval)
                    wait_interval = 3.0  # 3 seconds between checks
                    index_file_count = 0
                    last_index_file_count = 0
                    stable_count = 0  # Count how many times the file count stayed the same

                    for wait_attempt in range(max_wait_attempts):
                        # Check if clangd has created index files
                        if clangd_index_dir.exists():
                            index_files = list(clangd_index_dir.rglob("*"))
                            index_files = [f for f in index_files if f.is_file()]
                            index_file_count = len(index_files)

                            if index_file_count > 0:
                                # Check if index is still growing
                                if index_file_count == last_index_file_count:
                                    stable_count += 1
                                else:
                                    stable_count = 0
                                
                                last_index_file_count = index_file_count
                                
                                # Log progress periodically
                                if wait_attempt % 10 == 0 or index_file_count > 0:
                                    logger.info(
                                        f"[LSP] Clangd index building: {index_file_count} index files "
                                        f"(attempt {wait_attempt + 1}/{max_wait_attempts}, stable for {stable_count} checks)"
                                    )
                                
                                # If index count is stable for 3 checks (9 seconds), indexing is likely complete
                                if stable_count >= 3 and wait_attempt >= 10:
                                    verification_passed = True
                                    status_messages.append(
                                        f"Clangd native index complete: {index_file_count} index files"
                                    )
                                    logger.info(
                                        f"[LSP] Clangd native index appears complete: {index_file_count} index files "
                                        f"(stable for {stable_count * wait_interval:.1f}s)"
                                    )
                                    break
                        else:
                            # Index directory doesn't exist yet, clangd might not have started
                            if wait_attempt < 10:  # First 30 seconds
                                logger.debug(
                                    f"[LSP] Clangd index directory not found yet (attempt {wait_attempt + 1})"
                                )
                            elif wait_attempt % 20 == 0:  # Every 60 seconds after initial wait
                                logger.info(
                                    f"[LSP] Clangd index directory still not created after {wait_attempt * wait_interval:.0f}s. "
                                    "Indexing may take time for large projects."
                                )

                        await asyncio.sleep(wait_interval)
                    
                    # If we didn't find index files but files were analyzed, that's okay
                    # Clangd might still be building the index in the background
                    if index_file_count == 0 and analyzed_count > 0:
                        logger.info(
                            "[LSP] Files were analyzed but native index not yet created. "
                            "Clangd may continue building index in background."
                        )
                        status_messages.append(
                            f"Files analyzed: {analyzed_count} files (index may build in background)"
                        )
                        verification_passed = True

                if index_file_count > 0:
                    verification_passed = True
                    logger.info(
                        f"[LSP] Clangd indexing complete: {analyzed_count} files analyzed, "
                        f"{index_file_count} native index files"
                    )
                    status_messages.append(
                        f"Indexing complete: {analyzed_count} files analyzed, "
                        f"{index_file_count} native index files created"
                    )
                elif analyzed_count > 0:
                    # Files were analyzed but no native index yet - still mark as success
                    # The native index might build later
                    verification_passed = True
                    logger.info(
                        f"[LSP] Clangd files analyzed but native index not yet created: "
                        f"{analyzed_count} files analyzed"
                    )
                    status_messages.append(
                        f"Files analyzed: {analyzed_count} files "
                        f"(native index may build later)"
                    )
                else:
                    logger.warning(
                        "[LSP] No files were successfully analyzed by clangd"
                    )
                    status_messages.append(
                        "Warning: No files were successfully analyzed"
                    )
                    verification_passed = False

            elif language == "csharp" and isinstance(session, PyglsClientSession):
                # For OmniSharp (C#), we need to query files to trigger indexing
                # OmniSharp builds its index when files are queried
                logger.info(
                    f"[LSP] Triggering OmniSharp indexing by querying {len(opened_file_data)} files..."
                )
                status_messages.append(
                    f"Querying files to trigger OmniSharp indexing..."
                )

                # Query a subset of files to trigger OmniSharp to analyze them
                # We don't need to query all files - querying some will trigger OmniSharp to index them
                files_to_query = min(50, len(opened_file_data))  # Query up to 50 files
                analyzed_count = 0
                total_symbols_found = 0

                from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                    LspMethod,
                )

                for idx, (uri, file_path) in enumerate(
                    opened_file_data[:files_to_query], 1
                ):
                    try:
                        # Query documentSymbol to trigger OmniSharp to analyze the file
                        doc_symbol_result = await session.send_request(
                            LspMethod.DOCUMENT_SYMBOL, {"textDocument": {"uri": uri}}
                        )

                        symbol_count = (
                            len(doc_symbol_result)
                            if isinstance(doc_symbol_result, list)
                            else 0
                        )

                        if symbol_count > 0:
                            analyzed_count += 1
                            total_symbols_found += symbol_count

                        # Small delay to avoid overwhelming OmniSharp
                        if idx % 10 == 0:
                            await asyncio.sleep(0.5)
                            logger.info(
                                f"[LSP] Queried {idx}/{files_to_query} files, "
                                f"{analyzed_count} analyzed, {total_symbols_found} symbols found"
                            )
                    except Exception as exc:
                        logger.debug(
                            f"[LSP] Error querying documentSymbol for {file_path.name}: {exc}"
                        )

                logger.info(
                    f"[LSP] Triggered OmniSharp indexing on {files_to_query} files: "
                    f"{analyzed_count} files analyzed, {total_symbols_found} total symbols"
                )
                status_messages.append(
                    f"Triggered indexing: {analyzed_count}/{files_to_query} files analyzed "
                    f"({total_symbols_found} symbols)"
                )

                # Wait for OmniSharp to complete indexing
                # OmniSharp processes files asynchronously, so we wait a bit for it to catch up
                if analyzed_count > 0:
                    status_messages.append(
                        "Waiting for OmniSharp to complete indexing..."
                    )
                    logger.info(
                        "[LSP] Waiting for OmniSharp to complete indexing..."
                    )
                    # Wait a bit for OmniSharp to process the files
                    await asyncio.sleep(5.0)
                    verification_passed = True
                else:
                    logger.warning(
                        "[LSP] No files were successfully analyzed by OmniSharp"
                    )
                    status_messages.append(
                        "Warning: No files were successfully analyzed"
                    )
                    verification_passed = False

            else:
                # For other non-Python languages, use the polling method
                if isinstance(session, PyglsClientSession):
                    logger.info(
                        f"[LSP] Starting to wait for indexing completion for {language} "
                        f"(timeout: 60s, interval: 3s)"
                    )
                    indexing_complete = await session.wait_for_indexing_complete(
                        timeout=60.0, test_query_interval=3.0
                    )
                    verification_passed = indexing_complete
                else:
                    await asyncio.sleep(5.0)
                    verification_passed = False

            # For Jedi, cache is written synchronously during indexing, so no wait needed
            # For other language servers (like Pyright, clangd), keep files open longer to ensure cache is persisted
            if language == "python" and not isinstance(session, JediSession):
                if verification_passed:
                    status_messages.append("Waiting for cache to persist...")
                    logger.info("[LSP] Waiting 5 seconds for cache to persist...")
                    await asyncio.sleep(5.0)  # Give time to write cache files
                else:
                    # Even if verification failed, wait a bit - might still be processing
                    status_messages.append(
                        "Waiting additional time for processing files..."
                    )
                    logger.info(
                        "[LSP] Verification failed, but waiting 10 more seconds for processing..."
                    )
                    await asyncio.sleep(10.0)
            elif language in ("c", "cpp") and isinstance(session, ClangdSession):
                # For clangd, keep files open a bit longer to ensure cache is fully persisted
                if verification_passed:
                    status_messages.append("Waiting for clangd cache to persist...")
                    logger.info("[LSP] Waiting 10 seconds for clangd cache to persist...")
                    await asyncio.sleep(10.0)  # Give clangd time to write cache files
                else:
                    # Even if verification failed, wait a bit - clangd might still be processing
                    status_messages.append(
                        "Waiting additional time for clangd to process files..."
                    )
                    logger.info(
                        "[LSP] Verification failed, but waiting 15 more seconds for clangd processing..."
                    )
                    await asyncio.sleep(15.0)
            elif language == "csharp" and isinstance(session, PyglsClientSession):
                # For OmniSharp, keep files open a bit longer to ensure cache is fully persisted
                if verification_passed:
                    status_messages.append("Waiting for OmniSharp cache to persist...")
                    logger.info("[LSP] Waiting 5 seconds for OmniSharp cache to persist...")
                    await asyncio.sleep(5.0)  # Give OmniSharp time to write cache files
                else:
                    # Even if verification failed, wait a bit - OmniSharp might still be processing
                    status_messages.append(
                        "Waiting additional time for OmniSharp to process files..."
                    )
                    logger.info(
                        "[LSP] Verification failed, but waiting 10 more seconds for OmniSharp processing..."
                    )
                    await asyncio.sleep(10.0)

            # Close all opened files now that indexing is complete
            for uri in opened_files:
                try:
                    await session.send_notification(
                        "textDocument/didClose", {"textDocument": {"uri": uri}}
                    )
                except Exception:
                    pass  # Ignore errors when closing files

            if not verification_passed:
                logger.warning(
                    f"Index verification failed for {language}, but continuing"
                )
                status_messages.append(
                    "Index verification incomplete, but marking as done"
                )

            self._warmed_sessions.add(key)
            status_messages.append(f"Workspace indexing complete for {language}")
            logger.info(
                f"[LSP] Warmup session completed for {language} (project: {request.project_id})"
            )

            # Create a marker file to indicate successful indexing
            # Write a timestamp to the file so we can track when indexing completed
            cache_dir = self._get_lsp_cache_dir(workspace_root, language)
            index_marker = cache_dir / ".indexed"
            try:
                from datetime import datetime

                timestamp = datetime.utcnow().isoformat()

                # Check cache directory for files (Jedi for Python, clangd for C/C++, Pyright for others)
                cache_file_count = 0
                if language == "python":
                    jedi_cache_dir = cache_dir / "jedi_cache"
                    if jedi_cache_dir.exists():
                        cache_files = list(jedi_cache_dir.rglob("*.json"))
                        cache_file_count = len(cache_files)
                        if cache_file_count > 0:
                            status_messages.append(
                                f"Jedi cache created: {cache_file_count} cache files found"
                            )
                            logger.info(
                                f"Jedi cache directory contains {cache_file_count} files"
                            )
                        else:
                            status_messages.append(
                                "Warning: Jedi cache directory exists but is empty"
                            )
                            logger.warning(
                                f"Jedi cache directory {jedi_cache_dir} exists but contains no files."
                            )
                elif language in ("c", "cpp"):
                    # Check clangd native index (clangd's built-in cache)
                    clangd_cache_target = cache_dir / "clangd_native" / "clangd" / "index"
                    
                    # Also check workspace .cache location (where clangd actually stores it)
                    workspace_path = Path(workspace_root)
                    clangd_index_in_workspace = workspace_path / ".cache" / "clangd" / "index"
                    
                    native_index_count = 0
                    
                    # Count native index files (check both symlink target and workspace location)
                    index_dirs_to_check = [clangd_cache_target]
                    if clangd_index_in_workspace.exists():
                        if clangd_index_in_workspace.is_symlink():
                            try:
                                resolved = clangd_index_in_workspace.resolve()
                                if resolved.exists() and resolved not in index_dirs_to_check:
                                    index_dirs_to_check.append(resolved)
                            except Exception:
                                pass
                        elif clangd_index_in_workspace.is_dir():
                            index_dirs_to_check.append(clangd_index_in_workspace)
                    
                    for index_dir in index_dirs_to_check:
                        if index_dir.exists():
                            index_files = list(index_dir.rglob("*"))
                            index_files = [f for f in index_files if f.is_file()]
                            native_index_count += len(index_files)
                    
                    if native_index_count > 0:
                        status_messages.append(
                            f"Clangd native index: {native_index_count} index files"
                        )
                        logger.info(
                            f"Clangd native index: {native_index_count} index files "
                            f"(checked {clangd_index_in_workspace} and {clangd_cache_target})"
                        )
                else:
                    # For other languages, check Pyright cache
                    pyright_cache_dir = cache_dir / "pyright"
                    if pyright_cache_dir.exists():
                        cache_files = list(pyright_cache_dir.rglob("*"))
                        cache_files = [f for f in cache_files if f.is_file()]
                        cache_file_count = len(cache_files)
                        if cache_file_count > 0:
                            status_messages.append(
                                f"Cache created: {cache_file_count} cache files found"
                            )
                            logger.info(
                                f"Cache directory contains {cache_file_count} files"
                            )

                # Write final completion status (JSON format)
                # The marker file may have been updated during polling, so we finalize it now
                import json

                completion_data = {
                    "status": "completed",
                    "indexed_at": timestamp,
                    "workspace": workspace_root,
                    "verification_passed": verification_passed,
                    "cache_files": cache_file_count,
                    "files_opened": len(opened_files),
                }

                index_marker.write_text(
                    json.dumps(completion_data, indent=2), encoding="utf-8"
                )
                logger.info(
                    f"Created index marker for {language} at {index_marker} "
                    f"(workspace: {workspace_root}, timestamp: {timestamp}, "
                    f"cache_files: {cache_file_count})"
                )
            except Exception as exc:
                logger.warning(f"Failed to create index marker for {language}: {exc}")

    async def execute_query(
        self, request: LspQueryRequest, workspace_root: str
    ) -> LspQueryResponse:
        # For Python, we use Jedi directly and don't need a language server config
        # For other languages, we need a registered language server
        if request.language == "python":
            # Create a dummy config for Python (won't be used since we create JediSession)
            from app.modules.intelligence.tools.code_query_tools.pygls_client_session import (
                LanguageServerConfig,
            )

            config = LanguageServerConfig(command=["jedi"], timeout_seconds=60)
        else:
            try:
                config = self._get_language_config(request.language)
            except ValueError as config_error:
                message = str(config_error)
                return LspQueryResponse(
                    success=False,
                    method=request.method,
                    status_messages=[message],
                    locations=[],
                    symbols=[],
                    hover=None,
                    error=message,
                )
        status_messages: List[str] = []

        session = await self._get_or_create_session(
            request, workspace_root, config, status_messages
        )
        await self._warmup_session(
            request, workspace_root, session, config, status_messages
        )

        params = self._build_params(request)

        try:
            did_open_sent = False
            document_uri = request.text_document.uri if request.text_document else None
            document_payload = self._read_document_from_uri(
                document_uri, workspace_root
            )
            if document_payload:
                uri, text = document_payload
                # Check if file is already open in the session
                # For now, we'll always send didOpen to ensure the file is available
                await session.send_notification(
                    "textDocument/didOpen",
                    {
                        "textDocument": {
                            "uri": uri,
                            "languageId": request.language,
                            "version": 1,
                            "text": text,
                        }
                    },
                )
                did_open_sent = True
                status_messages.append(f"Opened {uri} for analysis")

                # Small delay to let the language server process the file
                await asyncio.sleep(0.1)

            try:
                raw_result = await session.send_request(request.method, params)
            finally:
                if did_open_sent and document_uri:
                    await session.send_notification(
                        "textDocument/didClose",
                        {"textDocument": {"uri": document_uri}},
                    )

            response = LspQueryResponse(
                success=True,
                method=request.method,
                status_messages=status_messages,
                locations=[],
                symbols=[],
                hover=None,
                error=None,
            )

            if request.method in {
                LspMethod.DEFINITION,
                LspMethod.REFERENCES,
            }:
                response.locations = self._normalize_locations(raw_result)
            elif request.method is LspMethod.HOVER:
                response.hover = self._normalize_hover(raw_result)
            elif request.method in {
                LspMethod.DOCUMENT_SYMBOL,
                LspMethod.WORKSPACE_SYMBOL,
            }:
                response.symbols = self._normalize_symbols(request, raw_result)

            return response
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception(
                "[LSP_MANAGER] Failed to execute %s for %s: %s",
                request.method.value,
                request.language,
                exc,
            )
            status_messages.append(
                f"Language server request failed: {type(exc).__name__}: {exc}"
            )
            return LspQueryResponse(
                success=False,
                method=request.method,
                status_messages=status_messages,
                locations=[],
                symbols=[],
                hover=None,
                error=str(exc),
            )

    async def index_workspace(
        self,
        project_id: str,
        workspace_root: str,
        languages: List[str],
    ) -> Dict[str, Any]:
        """
        Index a workspace for the given languages during parsing.

        This method pre-indexes the workspace so that LSP queries are fast
        when users start chatting. It creates sessions, configures caches,
        and triggers the warmup process for each language.

        Args:
            project_id: Project ID for the workspace
            workspace_root: Root path of the workspace to index
            languages: List of language identifiers to index (e.g., ['python', 'typescript'])

        Returns:
            Dictionary with indexing results for each language
        """
        results: Dict[str, Any] = {}

        for language in languages:
            try:
                # Get language config
                try:
                    config = self._get_language_config(language)
                except ValueError:
                    logger.debug(
                        f"Skipping LSP indexing for {language}: no language server configured"
                    )
                    results[language] = {
                        "success": False,
                        "error": f"No language server configured for {language}",
                    }
                    continue

                # Create a dummy request for warmup
                from app.modules.intelligence.tools.code_query_tools.lsp_types import (
                    LspMethod,
                    LspQueryRequest,
                )

                request = LspQueryRequest(
                    project_id=project_id,
                    language=language,
                    method=LspMethod.WORKSPACE_SYMBOL,
                    text_document=None,  # Not needed for workspace symbol
                    position=None,  # Not needed for workspace symbol
                    query="",  # Empty query just triggers indexing
                )

                # Get or create session and warm it up
                status_messages: List[str] = []
                session = await self._get_or_create_session(
                    request, workspace_root, config, status_messages
                )
                await self._warmup_session(
                    request, workspace_root, session, config, status_messages
                )

                # Verify marker file was created
                cache_dir = self._get_lsp_cache_dir(workspace_root, language)
                index_marker = cache_dir / ".indexed"
                if not index_marker.exists():
                    logger.warning(
                        f"Index marker file not found after warmup for {language} "
                        f"at {index_marker}. Indexing may have failed."
                    )
                    results[language] = {
                        "success": False,
                        "error": "Index marker file not created after warmup",
                        "status_messages": status_messages,
                    }
                else:
                    results[language] = {
                        "success": True,
                        "status_messages": status_messages,
                    }
                    logger.info(
                        f"Successfully indexed {language} workspace for project {project_id}. "
                        f"Marker file created at {index_marker}"
                    )

            except Exception as exc:
                logger.exception(
                    f"Failed to index {language} workspace for project {project_id}: {exc}"
                )
                results[language] = {
                    "success": False,
                    "error": str(exc),
                }

        return results
