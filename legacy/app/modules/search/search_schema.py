from typing import List

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    project_id: str
    query: str = Field(..., min_length=1)

    @field_validator("query", mode="after")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Search query cannot be empty or contain only whitespace")
        return s


class SearchResult(BaseModel):
    node_id: str
    name: str
    file_path: str
    content: str
    match_type: str
    relevance: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
