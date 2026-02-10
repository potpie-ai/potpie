import os
import secrets
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.code_provider.provider_factory import CodeProviderFactory


class CodeProviderCreatePullRequestInput(BaseModel):
    """Input for creating a pull request in a repository."""

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
        description="Optional list of usernames to request as reviewers",
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Optional list of labels to apply to the pull request"
    )


class CodeProviderCreatePullRequestTool:
    """Tool for creating a pull request in a repository."""

    name: str = "Create a new pull request"
    description: str = """
    Create a new pull request in a repository.
    Useful for proposing and collaborating on changes made in a branch.
    The tool will create a pull request from your specified head branch to the base branch.
    """
    args_schema: Type[BaseModel] = CodeProviderCreatePullRequestInput

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
        if not CodeProviderCreatePullRequestTool.gh_token_list:
            CodeProviderCreatePullRequestTool.initialize_tokens()

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = secrets.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        """Get code provider client using provider factory."""
        try:
            logger.info(f"[CREATE_PR] Creating provider for repo: {repo_name}")
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
            logger.info(
                f"[CREATE_PR] Provider created successfully, type: {type(provider).__name__}"
            )
            logger.info(f"[CREATE_PR] Client object: {type(provider.client).__name__}")
            return provider.client
        except Exception as e:
            logger.error(
                f"[CREATE_PR] Failed to get client: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise Exception(
                f"Repository {repo_name} not found or inaccessible: {str(e)}"
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
        logger.info(
            f"[CREATE_PR] Starting PR creation: repo={repo_name}, head={head_branch}, base={base_branch}, title={title}"
        )
        try:
            # Initialize GitHub client
            logger.info(f"[CREATE_PR] Getting client for repo: {repo_name}")
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
                f"[CREATE_PR] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
            )

            repo = g.get_repo(actual_repo_name)
            logger.info(f"[CREATE_PR] Successfully got repo object: {repo.name}")

            # Check if the branches exist
            try:
                logger.info(
                    f"[CREATE_PR] Checking if head branch exists: heads/{head_branch}"
                )
                head_ref = repo.get_git_ref(f"heads/{head_branch}")
                logger.info(
                    f"[CREATE_PR] Head branch exists: {head_ref.ref}, sha: {head_ref.object.sha}"
                )
            except GithubException as e:
                logger.error(
                    f"[CREATE_PR] Head branch '{head_branch}' not found: status={e.status}, data={e.data}"
                )
                return {
                    "success": False,
                    "error": f"Head branch '{head_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            try:
                logger.info(
                    f"[CREATE_PR] Checking if base branch exists: heads/{base_branch}"
                )
                base_ref = repo.get_git_ref(f"heads/{base_branch}")
                logger.info(
                    f"[CREATE_PR] Base branch exists: {base_ref.ref}, sha: {base_ref.object.sha}"
                )
            except GithubException as e:
                logger.error(
                    f"[CREATE_PR] Base branch '{base_branch}' not found: status={e.status}, data={e.data}"
                )
                return {
                    "success": False,
                    "error": f"Base branch '{base_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            # Create the pull request
            logger.info(
                f"[CREATE_PR] Creating pull request: head={head_branch}, base={base_branch}"
            )

            # For GitBucket, use raw API call to avoid PyGithub parsing issues
            if provider_type == "gitbucket":
                logger.info(
                    "[CREATE_PR] Using raw API call for GitBucket compatibility"
                )
                try:
                    import json

                    # Make raw API request
                    post_parameters = {
                        "title": title,
                        "body": body,
                        "head": head_branch,
                        "base": base_branch,
                    }
                    headers, data = repo._requester.requestJsonAndCheck(
                        "POST",
                        f"{repo.url}/pulls",
                        input=post_parameters,
                    )
                    logger.info(
                        f"[CREATE_PR] Raw API response received (type: {type(data)}): {data}"
                    )

                    # Parse JSON string if needed
                    if isinstance(data, str):
                        logger.info("[CREATE_PR] Parsing JSON string response")
                        data = json.loads(data)

                    # Extract PR details from raw response
                    pr_number = data.get("number")
                    pr_url = data.get("html_url")
                    logger.info(
                        f"[CREATE_PR] Successfully created PR #{pr_number}: {pr_url}"
                    )

                    result = {
                        "success": True,
                        "operation": "create_pull_request",
                        "pr_number": pr_number,
                        "title": title,
                        "head_branch": head_branch,
                        "base_branch": base_branch,
                        "url": pr_url,
                        "reviewers_added": False,  # Skip reviewers for GitBucket
                        "labels_added": False,  # Skip labels for GitBucket
                    }
                    logger.info(f"[CREATE_PR] Returning success result: {result}")
                    return result
                except Exception as e:
                    logger.error(
                        f"[CREATE_PR] Raw API call failed: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )
                    raise

            # For GitHub, use standard PyGithub method
            pr = repo.create_pull(
                title=title, body=body, head=head_branch, base=base_branch
            )
            logger.info(
                f"[CREATE_PR] Successfully created PR #{pr.number}: {pr.html_url}"
            )

            # Add reviewers if provided
            if reviewers:
                try:
                    logger.info(f"[CREATE_PR] Adding reviewers: {reviewers}")
                    pr.create_review_request(reviewers=reviewers)
                    logger.info("[CREATE_PR] Successfully added reviewers")
                except GithubException as e:
                    logger.warning(
                        f"[CREATE_PR] Error adding reviewers: status={e.status}, data={e.data}, message={str(e)}"
                    )

            # Add labels if provided
            if labels:
                try:
                    logger.info(f"[CREATE_PR] Adding labels: {labels}")
                    pr.add_to_labels(*labels)
                    logger.info("[CREATE_PR] Successfully added labels")
                except GithubException as e:
                    logger.warning(
                        f"[CREATE_PR] Error adding labels: status={e.status}, data={e.data}, message={str(e)}"
                    )

            result = {
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
            logger.info(f"[CREATE_PR] Returning success result: {result}")
            return result

        except GithubException as e:
            logger.error(
                f"[CREATE_PR] GithubException caught: status={e.status}, data={e.data}, message={str(e)}"
            )
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            logger.error(
                f"[CREATE_PR] Unexpected exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
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


def code_provider_create_pull_request_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    from app.modules.code_provider.provider_factory import has_code_provider_credentials

    if not has_code_provider_credentials():
        logger.warning(
            "No code provider credentials configured. Please set CODE_PROVIDER_TOKEN, "
            "GH_TOKEN_LIST, GITHUB_APP_ID, or CODE_PROVIDER_USERNAME/PASSWORD."
        )
        return None

    tool_instance = CodeProviderCreatePullRequestTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Create a new pull request",
        description="""
        Create a new pull request in a repository.
        Useful for proposing and collaborating on changes made in a branch.
        The tool will create a pull request from your specified head branch to the base branch.
        """,
        args_schema=CodeProviderCreatePullRequestInput,
    )
