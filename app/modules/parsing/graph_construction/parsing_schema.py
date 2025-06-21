from typing import Optional

from pydantic import BaseModel, Field, validator


class ParsingRequest(BaseModel):
    repo_name: Optional[str] = Field(default=None)
    repo_path: Optional[str] = Field(default=None)
    branch_name: Optional[str] = Field(default=None)
    commit_id: Optional[str] = Field(default=None)

    @validator('repo_name', 'repo_path', pre=True)
    def clean_empty_strings(cls, v):
        if v == "":
            return None
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Only validate if both fields are None or empty
        if not self.repo_name and not self.repo_path:
            raise ValueError("Either repo_name or repo_path must be provided.")


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repo_name: str
    branch_name: str
