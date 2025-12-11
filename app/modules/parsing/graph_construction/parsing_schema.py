from pydantic import BaseModel, Field, model_validator


class ParsingRequest(BaseModel):
    repo_name: str | None = Field(
        default=None,
        description="The full name of the GitHub repository (e.g., 'owner/repo').",
        examples=["facebook/react", "torvalds/linux"],
        min_length=3,
    )
    repo_path: str | None = Field(
        default=None,
        description="The absolute local file path to the repository directory.",
        examples=[
            "/Users/dev/projects/my-repo",
            "./data/repos/linux.git/feat/add_nvidia_support",
        ],
    )
    branch_name: str | None = Field(
        default=None,
        description="The specific branch to parse. If omitted, defaults to the repository's HEAD.",
        examples=["main", "develop", "feature/auth-v2"],
    )
    commit_id: str | None = Field(
        default=None,
        description="The specific commit SHA to parse. Mutually exclusive with branch_name.",
        examples=["a1b2c3d4", "998877665544332211aabbccddeeff00998877"],
    )
    inference: bool = Field(
        default=True,
        description="Whether to run the knowledge graph inference engine after parsing.",
    )

    @model_validator(mode="after")
    def check_repo_source(self) -> "ParsingRequest":
        if bool(self.repo_name) == bool(self.repo_path):
            raise ValueError("Exactly one of repo_name or repo_path must be provided.")
        return self

    @model_validator(mode="after")
    def check_version_conflict(self) -> "ParsingRequest":
        if self.branch_name and self.commit_id:
            raise ValueError("Cannot provide both branch_name and commit_id.")
        return self


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repo_name: str
    branch_name: str
    repo_path: Optional[str] = None
    commit_id: Optional[str] = None
