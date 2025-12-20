from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class DocstringRequest(BaseModel):
    node_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocstringNode(BaseModel):
    node_id: str
    docstring: str
    tags: Optional[List[str]] = []


class DocstringResponse(BaseModel):
    docstrings: List[DocstringNode]


class QueryRequest(BaseModel):
    project_id: str
    query: str
    node_ids: Optional[List[str]] = None


class QueryResponse(BaseModel):
    node_id: str
    docstring: str
    file_path: str
    start_line: int
    end_line: int
    similarity: float
