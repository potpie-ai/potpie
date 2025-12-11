from typing import Optional, List
import re

from pydantic import BaseModel, Field, field_validator


class ParseFilters(BaseModel):
    """Filters for excluding files and directories during repository parsing."""

    excluded_directories: List[str] = Field(default_factory=list)
    excluded_files: List[str] = Field(default_factory=list)
    excluded_extensions: List[str] = Field(default_factory=list)

    @field_validator("excluded_extensions")
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Validate extension formats - must start with a dot."""
        if not v:
            return v

        validated = []
        for ext in v:
            ext = ext.strip()
            if not ext:
                continue

            # Ensure extension starts with a dot
            if not ext.startswith("."):
                ext = f".{ext}"

            # Validate extension format (alphanumeric, dash, underscore only)
            if not re.match(r"^\.[a-zA-Z0-9_-]+$", ext):
                raise ValueError(
                    f"Invalid extension format: {ext}. Extensions must contain only alphanumeric characters, dashes, or underscores."
                )

            validated.append(ext)

        return validated

    @field_validator("excluded_directories", "excluded_files")
    @classmethod
    def validate_paths(cls, v: List[str]) -> List[str]:
        """Validate directory and file patterns - block dangerous patterns."""
        if not v:
            return v

        validated = []
        dangerous_patterns = [
            r"\.\./",  # Parent directory traversal
            r"\./",  # Current directory reference
            r"^/",  # Absolute paths
            r"~",  # Home directory expansion
            r"\$",  # Variable expansion
            r"`",  # Command substitution
            r"\|",  # Pipe operator
            r";",  # Command separator
            r"&",  # Background operator
            r">",  # Redirection
            r"<",  # Redirection
        ]

        for path in v:
            path = path.strip()
            if not path:
                continue

            # Check for dangerous patterns
            for pattern in dangerous_patterns:
                if re.search(pattern, path):
                    raise ValueError(
                        f"Invalid path pattern: {path}. Path contains dangerous characters or patterns."
                    )

            # Check for null bytes
            if "\x00" in path:
                raise ValueError(
                    f"Invalid path pattern: {path}. Path contains null bytes."
                )

            # Validate reasonable length
            if len(path) > 255:
                raise ValueError(
                    f"Invalid path pattern: {path}. Path exceeds maximum length of 255 characters."
                )

            validated.append(path)

        return validated


class ParsingRequest(BaseModel):
    repo_name: Optional[str] = Field(default=None)
    repo_path: Optional[str] = Field(default=None)
    branch_name: Optional[str] = Field(default=None)
    commit_id: Optional[str] = Field(default=None)
    filters: Optional[ParseFilters] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.repo_name and not self.repo_path:
            raise ValueError("Either repo_name or repo_path must be provided.")


class ParsingResponse(BaseModel):
    message: str
    status: str
    project_id: str


class RepoDetails(BaseModel):
    repo_name: str
    branch_name: str
    repo_path: Optional[str] = None
    commit_id: Optional[str] = None
