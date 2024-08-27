from typing import List

from pydantic import BaseModel


class SearchRequest(BaseModel):
    project_id: str
    query: str


class SearchResult(BaseModel):
    node_id: str
    name: str
    file_path: str
    content: str
    match_type: str
    relevance: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
