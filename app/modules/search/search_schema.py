from typing import List

from pydantic import BaseModel, Field, validator


class SearchRequest(BaseModel):
    project_id: str
    query: str = Field(..., min_length=1, strip_whitespace=True)

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Search query cannot be empty or contain only whitespace")
        return v


class SearchResult(BaseModel):
    node_id: str
    name: str
    file_path: str
    content: str
    match_type: str
    relevance: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
