import logging
import os
import random
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from github.Auth import AppAuth
import requests
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.core.config_provider import config_provider


class GitHubCreatePullRequestInput(BaseModel):
    """Input for creating a pull request in a GitHub repository."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    head_branch: str = Field(
        ..., description="The name of the branch where your changes are implemented"
    )
    base_branch: str = Field(
        ..., description="The branch you want the changes pulled into (e.g., 'main')"
    )
    title: str = Field(..., description="The title of the pull request")
    body: str = Field(..., description="The body/description of the pull request")
    reviewers: Optional[List[str]] = Field(
        default=None,
        description="Optional list of GitHub usernames to request as reviewers",
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Optional list of labels to apply to the pull request"
    )


class GitHubCreatePullRequestTool:
    """Tool for creating a pull request in a GitHub repository."""

    name: str = "Create a new pull request in GitHub"
    description: str = """
    Create a new pull request in a GitHub repository.
    Useful for proposing and collaborating on changes made in a branch.
    The tool will create a pull request from your specified head branch to the base branch.
    """
    args_schema: Type[BaseModel] = GitHubCreatePullRequestInput

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
        if not GitHubCreatePullRequestTool.gh_token_list:
            GitHubCreatePullRequestTool.initialize_tokens()

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
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a pull request in a GitHub repository.

        Args:
            repo_name: The full name of the repository (e.g., 'username/repo_name')
            head_branch: The name of the branch where your changes are implemented
            base_branch: The branch you want the changes pulled into (e.g., 'main')
            title: The title of the pull request
            body: The body/description of the pull request
            reviewers: Optional list of GitHub usernames to request as reviewers
            labels: Optional list of labels to apply to the pull request

        Returns:
            Dict containing the result of the pull request creation operation
        """
        try:
            # Initialize GitHub client
            g = self._get_github_client(repo_name)
            repo = g.get_repo(repo_name)

            # Check if the branches exist
            try:
                repo.get_git_ref(f"heads/{head_branch}")
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Head branch '{head_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            try:
                repo.get_git_ref(f"heads/{base_branch}")
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Base branch '{base_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            # Create the pull request
            pr = repo.create_pull(
                title=title, body=body, head=head_branch, base=base_branch
            )

            # Add reviewers if provided
            if reviewers:
                try:
                    pr.create_review_request(reviewers=reviewers)
                except GithubException as e:
                    logging.warning(f"Error adding reviewers: {str(e)}")

            # Add labels if provided
            if labels:
                try:
                    pr.add_to_labels(*labels)
                except GithubException as e:
                    logging.warning(f"Error adding labels: {str(e)}")

            return {
                "success": True,
                "operation": "create_pull_request",
                "pr_number": pr.number,
                "title": pr.title,
                "head_branch": head_branch,
                "base_branch": base_branch,
                "url": pr.html_url,
                "reviewers_added": reviewers is not None,
                "labels_added": labels is not None,
            }

        except GithubException as e:
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            return {"success": False, "error": f"Error creating pull request: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async implementation of the tool."""
        # For simplicity, we're using the sync version in async context
        # In a production environment, you'd want to use aiohttp or similar
        return self._run(
            repo_name=repo_name,
            head_branch=head_branch,
            base_branch=base_branch,
            title=title,
            body=body,
            reviewers=reviewers,
            labels=labels,
        )


def github_create_pull_request_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    if not os.getenv("GITHUB_APP_ID") or not config_provider.get_github_key():
        logging.warning(
            "GitHub app credentials not set, GitHub tool will not be initialized"
        )
        return None

    tool_instance = GitHubCreatePullRequestTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Create a new pull request in GitHub",
        description="""
        Create a new pull request in a GitHub repository.
        Useful for proposing and collaborating on changes made in a branch.
        The tool will create a pull request from your specified head branch to the base branch.
        """,
        args_schema=GitHubCreatePullRequestInput,
    )
