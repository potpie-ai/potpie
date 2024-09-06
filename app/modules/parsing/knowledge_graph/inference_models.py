from typing import Dict, List, Optional

from pydantic import BaseModel, Field

class DocstringRequest(BaseModel):
    node_id: str
    text: str

class DocstringNode(BaseModel):
    node_id: str
    docstring: str 

class DocstringResponse(BaseModel):
    docstrings: List[DocstringNode]

class QueryRequest(BaseModel):
    project_id: str
    query: str
    node_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    node_id: str
    docstring: str
    similarity: float