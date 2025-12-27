"""Code parsing using tree-sitter."""

from pathlib import Path
from typing import Optional

from potpie.types import CodeEdge, CodeNode, ParseResult


class Parser:
    """Tree-sitter based code parser.

    Parses source files to extract code structure (classes, functions, etc.)
    and relationships (calls, imports, inheritance).
    """

    def __init__(self):
        """Initialize parser with tree-sitter languages."""
        self._languages = {}  # language -> tree_sitter.Language

    async def parse_file(
        self, file_path: Path
    ) -> tuple[list[CodeNode], list[CodeEdge]]:
        """Parse a single file.

        Args:
            file_path: Path to source file.

        Returns:
            Tuple of (nodes, edges) extracted from the file.
        """
        # TODO: Implement tree-sitter parsing
        raise NotImplementedError("Parser.parse_file() not yet implemented")

    async def parse_directory(
        self,
        directory: Path,
        patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ParseResult:
        """Parse all files in a directory.

        Args:
            directory: Root directory to parse.
            patterns: Glob patterns to include (e.g., ["*.py", "*.js"]).
            exclude_patterns: Glob patterns to exclude (e.g., ["**/test/**"]).

        Returns:
            ParseResult with all nodes and edges.
        """
        # TODO: Implement directory parsing
        raise NotImplementedError("Parser.parse_directory() not yet implemented")

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        return extension_map.get(file_path.suffix.lower())
