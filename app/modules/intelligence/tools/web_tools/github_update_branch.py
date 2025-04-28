import logging
import os
import random
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from github.Auth import AppAuth
import requests
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.core.config_provider import config_provider


class GitHubUpdateFileInput(BaseModel):
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


class GitHubUpdateFileTool:
    """Tool for updating files in a GitHub repository branch."""

    name: str = "Update a file in a branch in GitHub"
    description: str = """
    Update a file in a GitHub repository branch.
    Useful for making changes to configuration files, code, documentation, or any other file in a repository.
    The tool will handle encoding the content and creating a commit on the specified branch.
    """
    args_schema: Type[BaseModel] = GitHubUpdateFileInput

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
        logging.info(f"Initialized {len(cls.gh_token_list)} GitHub tokens")

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        if not GitHubUpdateFileTool.gh_token_list:
            GitHubUpdateFileTool.initialize_tokens()

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = random.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        try:
            # Try authenticated access first
            private_key = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + config_provider.get_github_key()
                + "\n-----END RSA PRIVATE KEY-----\n"
            )
            app_id = os.environ["GITHUB_APP_ID"]
            auth = AppAuth(app_id=app_id, private_key=private_key)
            jwt = auth.create_jwt()

            # Get installation ID
            url = f"https://api.github.com/repos/{repo_name}/installation"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to get installation ID for {repo_name}")

            app_auth = auth.get_installation_auth(response.json()["id"])
            return Github(auth=app_auth)
        except Exception as private_error:
            logging.info(f"Failed to access private repo: {str(private_error)}")
            # If authenticated access fails, try public access
            try:
                return self.get_public_github_instance()
            except Exception as public_error:
                logging.error(f"Failed to access public repo: {str(public_error)}")
                raise Exception(
                    f"Repository {repo_name} not found or inaccessible on GitHub"
                )

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
        try:
            # Initialize GitHub client
            g = self._get_github_client(repo_name)
            repo = g.get_repo(repo_name)

            # Try to get the file to check if it exists and get its SHA
            try:
                file = repo.get_contents(file_path, ref=branch_name)
                sha = file.sha
                file_exists = True
            except GithubException as e:
                if e.status == 404:
                    # File doesn't exist
                    file_exists = False
                    sha = None
                else:
                    raise e

            # Create commit with author info if provided
            commit_kwargs = {"message": commit_message}
            if author_name and author_email:
                commit_kwargs["author"] = {"name": "Potpie", "email": "hi@potpie.ai"}

            # Update or create the file
            if file_exists:
                result = repo.update_file(
                    path=file_path,
                    content=content,
                    sha=sha,
                    branch=branch_name,
                    **commit_kwargs,
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
                result = repo.create_file(
                    path=file_path,
                    content=content,
                    branch=branch_name,
                    **commit_kwargs,
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
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status,
                "data": e.data,
            }
        except Exception as e:
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


def github_update_branch_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    if not os.getenv("GITHUB_APP_ID") or not config_provider.get_github_key():
        logging.warning(
            "GitHub app credentials not set, GitHub tool will not be initialized"
        )
        return None

    tool_instance = GitHubUpdateFileTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Update a file in a branch in GitHub",
        description="""
        Update a file in a GitHub repository branch.
        Useful for making changes to configuration files, code, documentation, or any other file in a repository.
        The tool will handle encoding the content and creating a commit on the specified branch.
        """,
        args_schema=GitHubUpdateFileInput,
    )
