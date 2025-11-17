"""
Structured tool scaffolding for querying language servers via LSP.

This module mirrors the design of `bash_command_tool.py`, but targets LSP
capabilities (definitions, references, hover, symbol search). The current
implementation focuses on request validation and outlines the future control
flow for invoking an `LspServerManager`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional, Sequence

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.code_query_tools.lsp_server_manager import (
    LspServerManager,
)
from app.modules.intelligence.tools.code_query_tools.lsp_types import (
    LspMethod,
    LspQueryRequest,
    Position,
    TextDocumentIdentifier,
)
from app.modules.intelligence.tools.code_query_tools.pygls_client_session import (
    LanguageServerConfig,
)
from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_SERVER_COMMANDS: Dict[str, Sequence[str]] = {
    "python": ["pyright-langserver", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
    "javascript": ["typescript-language-server", "--stdio"],
    "go": ["gopls"],
    "java": ["jdtls"],
    "rust": ["rust-analyzer"],
    "c": ["clangd", "--background-index"],
    "cpp": ["clangd", "--background-index"],
    "csharp": ["OmniSharp", "--lsp"],
    "ruby": ["solargraph", "stdio"],
    "php": ["intelephense", "--stdio"],
}

LSP_COMMAND_OVERRIDE_PREFIX = "LSP_COMMAND_"
PYTHON_LSP_INIT_OPTIONS: Dict[str, Any] = {
    "plugins": {
        "workspace_symbols": {"enabled": True},
    }
}


class LspQueryToolInput(BaseModel):
    """Structured input validated before invoking the language server."""

    project_id: str = Field(..., description="Project ID backing the repository.")
    language: str = Field(
        ..., description="Language identifier (e.g., 'python', 'typescript')."
    )
    method: LspMethod = Field(..., description="LSP method to invoke.")
    path: Optional[str] = Field(
        None,
        description="Optional path within the repository (relative to repo root).",
    )
    uri: Optional[str] = Field(
        None,
        description="Optional file:// URI for the target document. Overrides `path`.",
    )
    line: Optional[int] = Field(
        None,
        description="Zero-based line number used for position-based requests.",
    )
    character: Optional[int] = Field(
        None,
        description="Zero-based character index used for position-based requests.",
    )
    query: Optional[str] = Field(
        None,
        description="Search query for symbol requests.",
    )

    @model_validator(mode="after")
    def validate_position_requirements(self) -> "LspQueryToolInput":
        """Ensure inputs align with the selected LSP method."""

        method: LspMethod = self.method
        requires_position = method in {
            LspMethod.DEFINITION,
            LspMethod.REFERENCES,
            LspMethod.HOVER,
        }

        has_position = self.line is not None and self.character is not None
        if requires_position and not has_position:
            raise ValueError(
                f"Method {method.value} requires both `line` and `character` inputs."
            )

        return self


class LspQueryTool:
    """
    Skeleton tool responsible for invoking LSP queries on behalf of agents.

    The class mirrors the structure of `BashCommandTool`, including user/project
    validation, but delegates to `LspServerManager` for actual language server
    interactions. The `_run` method currently raises `NotImplementedError` to
    indicate the missing execution logic.
    """

    name: str = "lsp_query"
    description: str = (
        "Query language servers for definitions, references, hover, and symbols."
    )
    args_schema: type[BaseModel] = LspQueryToolInput

    def __init__(self, sql_db: Session, user_id: str) -> None:
        self.sql_db = sql_db
        self.user_id = user_id
        self.project_service = ProjectService(sql_db)
        self.repo_manager: Optional[RepoManager] = None
        from app.modules.intelligence.tools.code_query_tools.lsp_server_manager import (
            get_lsp_server_manager,
        )

        self.server_manager = get_lsp_server_manager()

        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                self.repo_manager = RepoManager()
                logger.info("LspQueryTool: RepoManager initialized")
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("LspQueryTool: Failed to initialize RepoManager: %s", exc)

        if self.repo_manager:
            self._configure_language_servers()

    def _get_project_details(self, project_id: str) -> Dict[str, Any]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _resolve_text_document(
        self, worktree_path: str, path: Optional[str], uri: Optional[str]
    ) -> Optional[TextDocumentIdentifier]:
        """
        Normalize the file reference for position-based queries.

        Prioritizes explicit URIs. When only a repo-relative path is provided,
        constructs a file:// URI rooted in the worktree.
        """
        if uri:
            return TextDocumentIdentifier(uri=uri)

        if path:
            normalized_path = os.path.normpath(path)
            if os.path.isabs(normalized_path) or normalized_path.startswith(".."):
                raise ValueError(
                    f"Invalid path '{path}'. Only repository-relative paths are allowed."
                )
            absolute_path = os.path.abspath(
                os.path.join(worktree_path, normalized_path)
            )
            return TextDocumentIdentifier(uri=f"file://{absolute_path}")

        return None

    def _configure_language_servers(self) -> None:
        legacy_python_command = os.getenv("PYTHON_LSP_COMMAND")
        if legacy_python_command and not os.getenv(
            f"{LSP_COMMAND_OVERRIDE_PREFIX}PYTHON"
        ):
            os.environ[f"{LSP_COMMAND_OVERRIDE_PREFIX}PYTHON"] = legacy_python_command

        # Environment overrides (LSP_COMMAND_<LANG>=command string)
        for env_key, command_str in os.environ.items():
            if not env_key.startswith(LSP_COMMAND_OVERRIDE_PREFIX):
                continue
            language = env_key[len(LSP_COMMAND_OVERRIDE_PREFIX) :].lower()
            if not command_str:
                continue
            try:
                config = LanguageServerConfig.from_command_string(command_str)
                if language == "python":
                    if not config.initialization_options:
                        config.initialization_options = dict(PYTHON_LSP_INIT_OPTIONS)
                    config.index_wait_seconds = 0.0
                    if config.timeout_seconds < 60:
                        config.timeout_seconds = 60
                self.server_manager.register_language(language, config)
            except Exception as exc:
                logger.warning(
                    "Failed to register LSP override for %s (%s): %s",
                    language,
                    command_str,
                    exc,
                )

        # Built-in defaults for popular languages
        for language, command in DEFAULT_LANGUAGE_SERVER_COMMANDS.items():
            if self.server_manager.is_language_registered(language):
                continue

            executable = command[0]
            resolved = shutil.which(executable)
            if not resolved:
                venv_candidate = os.path.join(
                    os.path.dirname(sys.executable), executable
                )
                if os.path.exists(venv_candidate):
                    resolved = venv_candidate
            # Check in .lsp_binaries directory (for binaries installed by install script)
            if not resolved:
                # Find project root by looking for common markers (alembic.ini, requirements.txt, etc.)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = current_dir
                for _ in range(10):  # Limit search depth
                    if os.path.exists(
                        os.path.join(project_root, "alembic.ini")
                    ) or os.path.exists(os.path.join(project_root, "requirements.txt")):
                        break
                    parent = os.path.dirname(project_root)
                    if parent == project_root:  # Reached filesystem root
                        project_root = None
                        break
                    project_root = parent

                if project_root:
                    lsp_binaries_dir = os.path.join(project_root, ".lsp_binaries")
                    if language == "csharp":
                        # Check for OmniSharp in .lsp_binaries/omnisharp/
                        # The executable is "OmniSharp" (capital O), not "omnisharp"
                        omnisharp_dir = os.path.join(lsp_binaries_dir, "omnisharp")
                        if os.path.exists(omnisharp_dir):
                            # Find OmniSharp executable (could be in subdirectory)
                            # Try capital O first (correct name)
                            for root, dirs, files in os.walk(omnisharp_dir):
                                # Look for "OmniSharp" (capital O) - the actual executable name
                                if "OmniSharp" in files:
                                    candidate = os.path.join(root, "OmniSharp")
                                    if os.path.exists(candidate) and os.access(
                                        candidate, os.X_OK
                                    ):
                                        resolved = candidate
                                        break
                                # Fallback to lowercase for compatibility
                                if executable in files:
                                    candidate = os.path.join(root, executable)
                                    if os.path.exists(candidate) and os.access(
                                        candidate, os.X_OK
                                    ):
                                        resolved = candidate
                                        break
                                # Windows fallback
                                if "OmniSharp.exe" in files:
                                    candidate = os.path.join(root, "OmniSharp.exe")
                                    if os.path.exists(candidate) and os.access(
                                        candidate, os.X_OK
                                    ):
                                        resolved = candidate
                                        break
            if not resolved:
                logger.debug(
                    "Skipping LSP registration for %s: executable '%s' not found in PATH or .lsp_binaries",
                    language,
                    executable,
                )
                continue

            resolved_command = [resolved, *command[1:]]
            if language == "python":
                config = LanguageServerConfig(
                    command=resolved_command,
                    initialization_options=dict(PYTHON_LSP_INIT_OPTIONS),
                    index_wait_seconds=0.0,
                    timeout_seconds=60,
                )
            elif language in ("c", "cpp"):
                # Clangd can take longer to respond, especially when indexing
                # Add production-ready flags for better performance and reliability
                # --malloc-trim: Release freed memory back to OS (prevents fragmentation)
                # --limit-results=50: Limit result sets for better performance
                # --limit-references=500: Limit reference counts
                # --pch-storage=disk: Store precompiled headers on disk (reduces memory)
                # -j=4: Limit parallelism to 4 workers (adjust based on system)
                clangd_flags = [
                    "--background-index",
                    "--limit-results=50",
                    "--limit-references=500",
                    "--pch-storage=disk",
                    "-j=4",
                ]
                # Note: --malloc-trim is not available in all clangd versions
                # Only add it if explicitly needed (requires clangd 14+)

                # Merge with any existing flags (avoid duplicates)
                # Extract flag names (without values) from existing flags
                existing_flag_names = set()
                for flag in resolved_command[1:]:
                    flag_name = flag.split("=")[0] if "=" in flag else flag
                    existing_flag_names.add(flag_name)

                # Add new flags that don't conflict
                final_flags = list(resolved_command[1:])  # Start with existing
                for flag in clangd_flags:
                    flag_name = flag.split("=")[0] if "=" in flag else flag
                    if flag_name not in existing_flag_names:
                        final_flags.append(flag)
                        existing_flag_names.add(flag_name)

                final_command = [resolved_command[0]] + final_flags

                config = LanguageServerConfig(
                    command=final_command,
                    timeout_seconds=120,  # 2 minutes timeout for clangd
                )
            elif language == "csharp":
                # OmniSharp can take longer to respond, especially when indexing
                # OmniSharp uses --lsp flag for LSP mode
                # OmniSharp may need more time to initialize, especially for large projects
                # or when restoring NuGet packages
                config = LanguageServerConfig(
                    command=resolved_command,
                    timeout_seconds=300,  # 5 minutes timeout for OmniSharp (can be slow to initialize)
                    initialization_options={
                        # Disable some features that might slow down initialization
                        "EnableMSBuildLoadProjectsOnDemand": True,
                        "EnableRoslynAnalyzers": True,
                        "EnableEditorConfigSupport": True,
                        "OrganizeImportsOnFormat": False,  # Disable to speed up
                    },
                )
            else:
                config = LanguageServerConfig(command=resolved_command)
            self.server_manager.register_language(language, config)

    async def _execute_async(
        self,
        project_id: str,
        language: str,
        method: LspMethod,
        path: Optional[str] = None,
        uri: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.repo_manager:
            return {
                "success": False,
                "error": "Repo manager is not enabled. LSP queries require a local worktree.",
            }

        details = self._get_project_details(project_id)
        repo_name = details["project_name"]
        branch = details.get("branch_name")
        commit_id = details.get("commit_id")

        worktree_path = self.repo_manager.get_repo_path(
            repo_name, branch=branch, commit_id=commit_id
        )
        if not worktree_path or not os.path.exists(worktree_path):
            return {
                "success": False,
                "error": f"Worktree not found for project {project_id}. Ensure the repository is available.",
            }

        try:
            self.repo_manager.update_last_accessed(
                repo_name, branch=branch, commit_id=commit_id
            )
        except Exception as local_exc:  # pragma: no cover - defensive
            logger.debug(
                "Failed to update last accessed time for %s: %s", repo_name, local_exc
            )

        text_document = self._resolve_text_document(worktree_path, path, uri)
        position = (
            Position(line=line, character=character)
            if line is not None and character is not None
            else None
        )

        request = LspQueryRequest(
            project_id=project_id,
            language=language,
            method=method,
            text_document=text_document,
            position=position,
            query=query,
        )

        logger.info(
            "[LSP_QUERY] Received request %s for project %s (repo=%s, branch=%s, commit=%s)",
            request.method.value,
            project_id,
            repo_name,
            branch,
            commit_id,
        )

        response = await self.server_manager.execute_query(request, worktree_path)
        return response.model_dump(by_alias=True)

    def _run(
        self,
        project_id: str,
        language: str,
        method: LspMethod,
        path: Optional[str] = None,
        uri: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self._execute_async(
                project_id,
                language,
                method,
                path,
                uri,
                line,
                character,
                query,
            )
        )

    async def _arun(
        self,
        project_id: str,
        language: str,
        method: LspMethod,
        path: Optional[str] = None,
        uri: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._execute_async(
            project_id,
            language,
            method,
            path,
            uri,
            line,
            character,
            query,
        )


def lsp_query_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    """
    Factory returning a structured tool when RepoManager is enabled.

    Mirrors the behavior of `bash_command_tool` by disabling the tool when
    repositories are not locally available.
    """

    repo_manager_enabled = os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    if not repo_manager_enabled:
        logger.debug("LspQueryTool not created: REPO_MANAGER_ENABLED is false")
        return None

    tool_instance = LspQueryTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug("LspQueryTool not created: RepoManager initialization failed")
        return None

    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=LspQueryToolInput,
    )
