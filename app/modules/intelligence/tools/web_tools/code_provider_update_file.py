import os
import secrets
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.code_provider.provider_factory import CodeProviderFactory


class CodeProviderUpdateFileInput(BaseModel):
    """Input for updating a file in a GitHub repository."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    file_path: str = Field(..., description="The path to the file in the repository")
    branch_name: str = Field(
        ..., description="The name of the branch where the file is located"
    )
    content: str = Field(..., description="The new content for the file")
    commit_message: str = Field(..., description="The commit message")


class CodeProviderUpdateFileTool:
    """Tool for updating files in a GitHub repository branch."""

    name: str = "Update a file in a branch"
    description: str = """
    Update a file in a GitHub repository branch.
    Useful for making changes to configuration files, code, documentation, or any other file in a repository.
    The tool will handle encoding the content and creating a commit on the specified branch.
    """
    args_schema: Type[BaseModel] = CodeProviderUpdateFileInput

    gh_token_list: List[str] = []

    @classmethod
    def initialize_tokens(cls):
        token_string = os.getenv("GH_TOKEN_LIST", "")
        cls.gh_token_list = [
            token.strip() for token in token_string.split(",") if token.strip()
        ]
        if not cls.gh_token_list:
            raise ValueError(
                "GitHub token list is empty or not set in environment variables"
            )
        logger.info(f"Initialized {len(cls.gh_token_list)} GitHub tokens")

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        if not CodeProviderUpdateFileTool.gh_token_list:
            CodeProviderUpdateFileTool.initialize_tokens()

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = secrets.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        """Get GitHub client using provider factory."""
        try:
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        except ValueError as e:
            logger.exception(
                f"Failed to create provider for repository '{repo_name}': {str(e)}"
            )
            raise ValueError(
                f"Repository {repo_name} not found or inaccessible on GitHub"
            ) from e

        if provider is None:
            message = (
                f"Provider factory returned None for repository '{repo_name}'. "
                "Unable to obtain client."
            )
            logger.error(message)
            raise ValueError(message)

        client = getattr(provider, "client", None)
        if client is None:
            message = (
                f"Provider '{type(provider).__name__}' does not expose a client for "
                f"repository '{repo_name}'."
            )
            logger.error(message)
            raise ValueError(message)

        if not hasattr(client, "get_repo"):
            message = (
                f"Client of type '{type(client).__name__}' for repository "
                f"'{repo_name}' does not support required operations."
            )
            logger.error(message)
            raise ValueError(message)

        return client

    def _run(
        self,
        repo_name: str,
        file_path: str,
        branch_name: str,
        content: str,
        commit_message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a file in a GitHub repository branch.

        Args:
            repo_name: The full name of the repository (e.g., 'username/repo_name')
            file_path: The path to the file in the repository
            branch_name: The name of the branch where the file is located
            content: The new content for the file
            commit_message: The commit message
            author_name: Optional author name for the commit
            author_email: Optional author email for the commit

        Returns:
            Dict containing the result of the update operation
        """
        logger.info(
            f"[UPDATE_FILE] Starting file update: repo={repo_name}, file={file_path}, branch={branch_name}"
        )
        try:
            # Initialize GitHub client
            logger.info(f"[UPDATE_FILE] Getting client for repo: {repo_name}")
            g = self._get_github_client(repo_name)

            # Normalize input repo_name if needed, then get actual name for API calls
            from app.modules.parsing.utils.repo_name_normalizer import (
                normalize_repo_name,
                get_actual_repo_name_for_lookup,
            )
            import os

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            # Normalize input repo_name first (in case it comes in as "root/repo")
            normalized_input = normalize_repo_name(repo_name, provider_type)
            # Then convert to actual format for API calls
            actual_repo_name = get_actual_repo_name_for_lookup(
                normalized_input, provider_type
            )
            logger.info(
                f"[UPDATE_FILE] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
            )

            repo = g.get_repo(actual_repo_name)
            logger.info(f"[UPDATE_FILE] Successfully got repo object: {repo.name}")

            # Try to get the file to check if it exists and get its SHA
            try:
                logger.info(
                    f"[UPDATE_FILE] Checking if file exists: {file_path} on branch: {branch_name}"
                )
                file = repo.get_contents(file_path, ref=branch_name)
                sha = file.sha
                file_exists = True
                logger.info(f"[UPDATE_FILE] File exists with sha: {sha}")
            except GithubException as e:
                if e.status == 404:
                    # File doesn't exist
                    file_exists = False
                    sha = None
                    logger.info(
                        "[UPDATE_FILE] File does not exist (404), will create new file"
                    )
                else:
                    logger.error(
                        f"[UPDATE_FILE] Error checking file existence: status={e.status}, data={e.data}"
                    )
                    raise e

            # Create commit with author info if provided
            commit_kwargs = {"message": commit_message}
            if author_name and author_email:
                commit_kwargs["author"] = {"name": "Potpie", "email": "hi@potpie.ai"}

            # Update or create the file
            if file_exists:
                logger.info(f"[UPDATE_FILE] Updating existing file: {file_path}")
                result = repo.update_file(
                    path=file_path,
                    content=content,
                    sha=sha,
                    branch=branch_name,
                    **commit_kwargs,
                )
                logger.info(
                    f"[UPDATE_FILE] Successfully updated file, commit sha: {result['commit'].sha}"
                )
                return {
                    "success": True,
                    "operation": "update",
                    "file_path": file_path,
                    "commit_sha": result["commit"].sha,
                    "branch": branch_name,
                    "url": result["commit"].html_url,
                }
            else:
                logger.info(f"[UPDATE_FILE] Creating new file: {file_path}")
                result = repo.create_file(
                    path=file_path,
                    content=content,
                    branch=branch_name,
                    **commit_kwargs,
                )
                logger.info(
                    f"[UPDATE_FILE] Successfully created file, commit sha: {result['commit'].sha}"
                )
                return {
                    "success": True,
                    "operation": "create",
                    "file_path": file_path,
                    "commit_sha": result["commit"].sha,
                    "branch": branch_name,
                    "url": result["commit"].html_url,
                }

        except GithubException as e:
            logger.error(
                f"[UPDATE_FILE] GithubException: status={e.status}, data={e.data}, message={str(e)}"
            )
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status,
                "data": e.data,
            }
        except Exception as e:
            logger.error(
                f"[UPDATE_FILE] Unexpected exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            return {"success": False, "error": f"Error updating file: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        file_path: str,
        branch_name: str,
        content: str,
        commit_message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async implementation of the tool."""
        return self._run(
            repo_name=repo_name,
            file_path=file_path,
            branch_name=branch_name,
            content=content,
            commit_message=commit_message,
            author_name=author_name,
            author_email=author_email,
        )


def code_provider_update_file_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    from app.modules.code_provider.provider_factory import has_code_provider_credentials

    if not has_code_provider_credentials():
        logger.warning(
            "No code provider credentials configured. Please set CODE_PROVIDER_TOKEN, "
            "GH_TOKEN_LIST, GITHUB_APP_ID, or CODE_PROVIDER_USERNAME/PASSWORD."
        )
        return None

    tool_instance = CodeProviderUpdateFileTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Update a file in a branch",
        description="""
        Update a file in a GitHub repository branch.
        Useful for making changes to configuration files, code, documentation, or any other file in a repository.
        The tool will handle encoding the content and creating a commit on the specified branch.
        """,
        args_schema=CodeProviderUpdateFileInput,
    )
