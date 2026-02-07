from typing import List

from pydantic import BaseModel, Field, constr, validator


class SearchRequest(BaseModel):
    project_id: str
    # Use pydantic constr to ensure whitespace is stripped and minimum length enforced.
    # This enables model-level validation; a RequestValidationError will be converted to
    # a 400 for the search endpoint by the app-level exception handler.
    query: constr(min_length=1, strip_whitespace=True)

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
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
