from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    node_id: str
    score: float
    metadata: dict
