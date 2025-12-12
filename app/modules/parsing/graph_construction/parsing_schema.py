from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ParsingRequest(BaseModel):
    """
    Request schema for parsing a repository.
    
    Uses a single unified 'repository' identifier that can be either:
    - A remote repository in 'owner/repo' format (e.g., 'octocat/Hello-World')
    - A local filesystem path (e.g., '/path/to/repo' or 'C:\\path\\to\\repo')
    
    The system automatically detects the type and resolves it appropriately.
    """
    repository: str = Field(
        ...,
        description="Repository identifier: 'owner/repo' for remote, or path for local"
    )
    branch_name: Optional[str] = Field(default=None, description="Branch name to parse")
    commit_id: Optional[str] = Field(default=None, description="Specific commit ID to parse")

    @field_validator('repository')
    @classmethod
    def validate_repository(cls, v: str) -> str:
        """Validate that repository identifier is not empty."""
        if not v or not v.strip():
            raise ValueError("Repository identifier cannot be empty")
        return v.strip()


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    """
    Resolved repository details after resolution.
    
    This model contains the normalized and validated repository information
    that downstream services can use without additional resolution logic.
    """
    repository: str = Field(
        ...,
        description="Original repository identifier provided by user"
    )
    repo_name: str = Field(
        ...,
        description="Normalized repository name extracted from identifier"
    )
    repo_path: Optional[str] = Field(
        default=None,
        description="Resolved local filesystem path (only for local repositories)"
    )
    branch_name: Optional[str] = Field(default=None, description="Branch name")
    commit_id: Optional[str] = Field(default=None, description="Commit ID")
    is_local: bool = Field(
        ...,
        description="True if this is a local repository, False if remote"
    )
