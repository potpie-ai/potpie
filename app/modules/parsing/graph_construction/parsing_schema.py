from typing import Optional

from pydantic import BaseModel, Field

from app.modules.parsing.utils.repository_resolver import RepositoryResolver
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ParsingRequest(BaseModel):
    repository_identifier: Optional[str] = Field(
        default=None,
        description=(
            "Unified repository identifier. Can be a remote GitHub repo "
            "(e.g. 'owner/repo') or a local filesystem path "
            "(e.g. '/path/to/repo'). Preferred over the deprecated "
            "repo_name / repo_path fields."
        ),
    )
    # Deprecated
    repo_name: Optional[str] = Field(default=None)
    repo_path: Optional[str] = Field(default=None)
    branch_name: Optional[str] = Field(default=None)
    commit_id: Optional[str] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        if self.repository_identifier:
            resolved = RepositoryResolver.classify(self.repository_identifier)
            if resolved.is_local:
                self.repo_path = resolved.repo_path
                self.repo_name = resolved.repo_name
            else:
                self.repo_name = resolved.repo_name
                self.repo_path = None
        elif self.repo_name or self.repo_path:
            # Backward-compatible
            logger.warning(
                "ParsingRequest: 'repo_name' and 'repo_path' are deprecated. "
                "Use 'repository_identifier' instead."
            )
            if self.repo_name and not self.repo_path:
                if RepositoryResolver.looks_like_path(self.repo_name):
                    resolved = RepositoryResolver.classify(self.repo_name)
                    self.repo_path = resolved.repo_path
                    self.repo_name = resolved.repo_name
        else:
            raise ValueError(
                "Either repository_identifier, repo_name, or repo_path "
                "must be provided."
            )


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repo_name: str
    branch_name: str
    repo_path: Optional[str] = None
    commit_id: Optional[str] = None
    is_local: bool = False


class ParsingStatusRequest(BaseModel):
    repo_name: str = Field(..., description="Repository name to check status for")
    commit_id: Optional[str] = Field(
        default=None, description="Commit ID to check status for"
    )
    branch_name: Optional[str] = Field(
        default=None, description="Branch name (used if commit_id is not provided)"
    )
