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


class GitHubCreateBranchInput(BaseModel):
    """Input for creating a new branch in a GitHub repository."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    base_branch: str = Field(
        ..., description="The name of the branch to branch from (e.g., 'main')"
    )
    new_branch_name: str = Field(
        ..., description="The name of the new branch to be created"
    )


class GitHubCreateBranchTool:
    """Tool for creating a new branch in a GitHub repository."""

    name: str = "Create a new branch in GitHub"
    description: str = """
    Create a new branch in a GitHub repository.
    Useful for starting a new feature, bugfix, or any work that requires a separate branch.
    The tool will create the branch from the specified base branch.
    """
    args_schema: Type[BaseModel] = GitHubCreateBranchInput
    gh_token_list: List[str] = []

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

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
        base_branch: str,
        new_branch_name: str,
    ) -> Dict[str, Any]:
        """
        Create a new branch in a GitHub repository.

        Args:
            repo_name: The full name of the repository (e.g., 'username/repo_name')
            base_branch: The name of the branch to branch from (e.g., 'main')
            new_branch_name: The name of the new branch to be created

        Returns:
            Dict containing the result of the branch creation operation
        """
        try:
            # Initialize GitHub client
            g = self._get_github_client(repo_name)
            repo = g.get_repo(repo_name)

            # Get the base branch reference
            try:
                base_ref = repo.get_git_ref(f"heads/{base_branch}")
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Base branch '{base_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            # Check if the new branch already exists
            try:
                repo.get_git_ref(f"heads/{new_branch_name}")
                return {
                    "success": False,
                    "error": f"Branch '{new_branch_name}' already exists",
                }
            except GithubException as e:
                if e.status != 404:
                    # If error is not "Not Found", it's an unexpected error
                    return {
                        "success": False,
                        "error": f"Error checking branch existence: {str(e)}",
                        "status_code": e.status,
                    }
                # 404 means the branch doesn't exist, which is what we want

            # Create the new branch
            new_ref = repo.create_git_ref(
                ref=f"refs/heads/{new_branch_name}", sha=base_ref.object.sha
            )

            return {
                "success": True,
                "operation": "create_branch",
                "base_branch": base_branch,
                "new_branch": new_branch_name,
                "sha": new_ref.object.sha,
                "url": f"https://github.com/{repo_name}/tree/{new_branch_name}",
            }

        except GithubException as e:
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            return {"success": False, "error": f"Error creating branch: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        base_branch: str,
        new_branch_name: str,
    ) -> Dict[str, Any]:
        """Async implementation of the tool."""
        return self._run(
            repo_name=repo_name,
            base_branch=base_branch,
            new_branch_name=new_branch_name,
        )


def github_create_branch_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:

    tool_instance = GitHubCreateBranchTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Create a new branch in GitHub",
        description="""
        Create a new branch in a GitHub repository.
        Useful for starting a new feature, bugfix, or any work that requires a separate branch.
        The tool will create the branch from the specified base branch.
        """,
        args_schema=GitHubCreateBranchInput,
    )
