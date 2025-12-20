"""Shared types for Potpie."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class NodeType(str, Enum):
    """Types of nodes in the code graph."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"


@dataclass
class CodeNode:
    """Represents a node in the code knowledge graph."""

    id: str
    name: str
    node_type: NodeType
    file_path: str
    start_line: int
    end_line: int
    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class CodeEdge:
    """Represents an edge (relationship) in the code knowledge graph."""

    source_id: str
    target_id: str
    edge_type: str
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ParseResult:
    """Result of parsing a repository."""

    nodes: list[CodeNode]
    edges: list[CodeEdge]
    file_count: int
    error_count: int


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response from the chat engine."""

    content: str
    sources: list[str]
    metadata: Optional[dict[str, Any]] = None
