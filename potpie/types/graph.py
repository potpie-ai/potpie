from dataclasses import dataclass
from typing import List

from .base import EdgeType


@dataclass
class CodeNode:
    """Represents a node in the code knowledge graph."""

    id: str
    type: str  # class, function, variable, etc.
    name: str
    content: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None
    project_id: str | None = None
    metadata: dict | None = None
    embedding: List[float] | None = None


@dataclass
class CodeEdge:
    """Represents an edge in the code knowledge graph."""

    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict | None = None
