"""
Jedi-based Python code analysis session with caching.

This module provides a Jedi-powered implementation for Python code analysis
that caches results to disk for fast subsequent lookups.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import jedi

from app.modules.intelligence.tools.code_query_tools.lsp_session import (
    BaseLspServerSession,
    SessionKey,
)
from app.modules.intelligence.tools.code_query_tools.lsp_types import (
    LspMethod,
)

logger = logging.getLogger(__name__)


class JediCache:
    """Cached Jedi-based code analysis."""

    def __init__(self, project_root: str, cache_dir: Path):
        """Initialize Jedi cache for a project."""
        self.project = jedi.Project(path=project_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self, filepath: str, line: int, column: int, operation: str
    ) -> str:
        """Generate cache key based on file content hash."""
        try:
            with open(filepath, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            return f"{operation}_{file_hash}_{line}_{column}"
        except Exception:
            # Fallback if file can't be read
            return f"{operation}_{Path(filepath).name}_{line}_{column}"

    def _uri_to_path(self, uri: str) -> str:
        """Convert file:// URI to filesystem path."""
        parsed = urlparse(uri)
        return unquote(parsed.path)

    def get_definitions(self, uri: str, line: int, column: int) -> List[Dict[str, Any]]:
        """Get definitions (goto) with caching."""
        filepath = self._uri_to_path(uri)
        cache_key = self._get_cache_key(filepath, line, column, "def")
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Failed to load cache for {cache_key}: {exc}")

        # Use Jedi (line is 1-based in Jedi)
        try:
            script = jedi.Script(path=filepath, project=self.project)
            definitions = script.goto(line + 1, column)

            # Convert to serializable format
            result = []
            for d in definitions:
                result.append(
                    {
                        "name": d.name,
                        "type": d.type,
                        "module_path": str(d.module_path) if d.module_path else None,
                        "line": d.line - 1 if d.line else 0,  # Convert to 0-based
                        "column": d.column,
                        "docstring": d.docstring(),
                        "full_name": d.full_name,
                        "uri": f"file://{d.module_path}" if d.module_path else uri,
                    }
                )

            # Cache it
            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as exc:
                logger.debug(f"Failed to cache {cache_key}: {exc}")

            return result
        except Exception as exc:
            logger.warning(
                f"Jedi get_definitions failed for {filepath}:{line}:{column}: {exc}"
            )
            return []

    def get_references(self, uri: str, line: int, column: int) -> List[Dict[str, Any]]:
        """Get all references to symbol with caching."""
        filepath = self._uri_to_path(uri)
        cache_key = self._get_cache_key(filepath, line, column, "ref")
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Failed to load cache for {cache_key}: {exc}")

        try:
            script = jedi.Script(path=filepath, project=self.project)
            references = script.get_references(line + 1, column)

            result = []
            for ref in references:
                result.append(
                    {
                        "name": ref.name,
                        "module_path": (
                            str(ref.module_path) if ref.module_path else None
                        ),
                        "line": ref.line - 1 if ref.line else 0,  # Convert to 0-based
                        "column": ref.column,
                        "is_definition": ref.is_definition(),
                        "uri": f"file://{ref.module_path}" if ref.module_path else uri,
                    }
                )

            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as exc:
                logger.debug(f"Failed to cache {cache_key}: {exc}")

            return result
        except Exception as exc:
            logger.warning(
                f"Jedi get_references failed for {filepath}:{line}:{column}: {exc}"
            )
            return []

    def get_hover(self, uri: str, line: int, column: int) -> Optional[Dict[str, Any]]:
        """Get hover information (type info + docstring) with caching."""
        filepath = self._uri_to_path(uri)
        cache_key = self._get_cache_key(filepath, line, column, "hover")
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Failed to load cache for {cache_key}: {exc}")

        try:
            script = jedi.Script(path=filepath, project=self.project)
            inferred = script.infer(line + 1, column)

            if not inferred:
                return None

            # Get first inference result
            inf = inferred[0]
            docstring = inf.docstring()
            type_name = inf.type or inf.name

            result = {
                "contents": [
                    {
                        "language": "python",
                        "value": (
                            f"```python\n{type_name}\n```\n\n{docstring}"
                            if docstring
                            else f"```python\n{type_name}\n```"
                        ),
                    }
                ],
                "full_name": inf.full_name,
                "type": inf.type,
                "name": inf.name,
            }

            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as exc:
                logger.debug(f"Failed to cache {cache_key}: {exc}")

            return result
        except Exception as exc:
            logger.warning(
                f"Jedi get_hover failed for {filepath}:{line}:{column}: {exc}"
            )
            return None

    def get_document_symbols(self, uri: str) -> List[Dict[str, Any]]:
        """Get all definitions in a file with caching."""
        filepath = self._uri_to_path(uri)
        cache_key = hashlib.sha256(str(filepath).encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"names_{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Failed to load cache for {cache_key}: {exc}")

        try:
            script = jedi.Script(path=filepath, project=self.project)
            names = script.get_names(
                all_scopes=True, definitions=True, references=False
            )

            result = []
            for n in names:
                result.append(
                    {
                        "name": n.name,
                        "type": n.type,
                        "line": n.line - 1 if n.line else 0,  # Convert to 0-based
                        "column": n.column,
                        "full_name": n.full_name,
                        "uri": uri,
                    }
                )

            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f)
            except Exception as exc:
                logger.debug(f"Failed to cache {cache_key}: {exc}")

            return result
        except Exception as exc:
            logger.warning(f"Jedi get_document_symbols failed for {filepath}: {exc}")
            return []

    def get_workspace_symbols(self, query: str = "") -> List[Dict[str, Any]]:
        """Get workspace symbols (searches all files in project)."""
        # For workspace symbols, we need to search across all files
        # This is expensive, so we cache the entire workspace symbol index
        cache_file = self.cache_dir / "workspace_symbols.json"

        # Load cached workspace symbols if available
        cached_symbols = []
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached_symbols = json.load(f)
            except Exception:
                pass

        # Filter by query if provided
        if query:
            query_lower = query.lower()
            return [
                s
                for s in cached_symbols
                if query_lower in (s.get("name") or "").lower()
                or query_lower in (s.get("full_name") or "").lower()
            ]

        return cached_symbols

    def index_workspace(self, python_files: List[str]) -> None:
        """Index all Python files in workspace for workspace symbol search."""
        cache_file = self.cache_dir / "workspace_symbols.json"

        # Check if already indexed recently
        if cache_file.exists():
            # Could add timestamp check here to re-index if files changed
            logger.debug(f"Workspace symbols already cached at {cache_file}")
            return

        logger.info(
            f"Indexing {len(python_files)} Python files for workspace symbols..."
        )
        all_symbols = []

        for filepath in python_files:
            try:
                script = jedi.Script(path=filepath, project=self.project)
                names = script.get_names(
                    all_scopes=True, definitions=True, references=False
                )

                for n in names:
                    all_symbols.append(
                        {
                            "name": n.name,
                            "type": n.type,
                            "line": n.line - 1 if n.line else 0,
                            "column": n.column,
                            "full_name": n.full_name,
                            "uri": f"file://{filepath}",
                        }
                    )
            except Exception as exc:
                logger.debug(f"Failed to index {filepath}: {exc}")

        # Cache workspace symbols
        try:
            with open(cache_file, "w") as f:
                json.dump(all_symbols, f)
            logger.info(f"Cached {len(all_symbols)} workspace symbols")
        except Exception as exc:
            logger.warning(f"Failed to cache workspace symbols: {exc}")


class JediSession(BaseLspServerSession):
    """Jedi-based LSP session for Python code analysis."""

    def __init__(
        self,
        session_key: SessionKey,
        workspace_root: str,
        cache_dir: Path,
    ) -> None:
        super().__init__(session_key, workspace_root)
        self.cache_dir = cache_dir / "jedi_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._jedi_cache = JediCache(workspace_root, self.cache_dir)
        self._initialized = True

    async def ensure_started(self) -> None:
        """Jedi doesn't need startup - it's ready immediately."""
        pass

    async def send_request(self, method: LspMethod, payload: dict) -> Any:
        """Handle LSP requests using Jedi."""
        if method == LspMethod.DEFINITION:
            uri = payload["textDocument"]["uri"]
            position = payload["position"]
            return self._jedi_cache.get_definitions(
                uri, position["line"], position["character"]
            )

        elif method == LspMethod.REFERENCES:
            uri = payload["textDocument"]["uri"]
            position = payload["position"]
            return self._jedi_cache.get_references(
                uri, position["line"], position["character"]
            )

        elif method == LspMethod.HOVER:
            uri = payload["textDocument"]["uri"]
            position = payload["position"]
            return self._jedi_cache.get_hover(
                uri, position["line"], position["character"]
            )

        elif method == LspMethod.DOCUMENT_SYMBOL:
            uri = payload["textDocument"]["uri"]
            return self._jedi_cache.get_document_symbols(uri)

        elif method == LspMethod.WORKSPACE_SYMBOL:
            query = payload.get("query", "")
            return self._jedi_cache.get_workspace_symbols(query)

        else:
            raise ValueError(f"Unsupported method: {method}")

    async def send_notification(
        self, method: str, payload: Optional[dict] = None
    ) -> None:
        """Jedi doesn't need notifications."""
        pass

    async def shutdown(self) -> None:
        """Jedi doesn't need shutdown."""
        pass

    async def index_workspace(self, python_files: List[str]) -> None:
        """Index workspace for symbol search."""
        # Run indexing in executor to avoid blocking
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._jedi_cache.index_workspace, python_files)
