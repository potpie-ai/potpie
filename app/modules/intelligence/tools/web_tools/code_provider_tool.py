import asyncio
import os
import secrets
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
from typing import Any, Dict, List, Optional

from github import Github
from github.GithubException import UnknownObjectException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.code_provider.provider_factory import CodeProviderFactory


class CodeProviderToolInput(BaseModel):
    repo_name: str = Field(
        description="The full repository name in format 'owner/repo' WITHOUT any quotes"
    )
    issue_number: Optional[int] = Field(
        description="The issue or pull request number to fetch", default=None
    )
    is_pull_request: bool = Field(
        description="Whether to fetch a pull request (True) or issue (False)",
        default=False,
    )


class CodeProviderTool:
    name = "Code Provider Tool"
    description = """Fetches GitHub issues and pull request information including diffs.
        :param repo_name: string, the full repository name (owner/repo)
        :param issue_number: optional int, the issue or PR number to fetch
        :param is_pull_request: optional bool, whether to fetch a PR (True) or issue (False)

            example:
            {
                "repo_name": 'owner/repo',
                "issue_number": 123,
                "is_pull_request": true
            }

        Returns dictionary containing the issue/PR content, metadata, and success status.
        """

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
        if not CodeProviderTool.gh_token_list:
            CodeProviderTool.initialize_tokens()

    async def arun(
        self,
        repo_name: str,
        issue_number: Optional[int] = None,
        is_pull_request: bool = False,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.run, repo_name, issue_number, is_pull_request
        )

    def run(
        self,
        repo_name: str,
        issue_number: Optional[int] = None,
        is_pull_request: bool = False,
    ) -> Dict[str, Any]:
        try:
            repo_name = repo_name.strip('"')
            content = self._fetch_github_content(
                repo_name, issue_number, is_pull_request
            )
            if not content:
                return {
                    "success": False,
                    "error": "Failed to fetch GitHub content",
                    "content": None,
                }
            return content
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "content": None,
            }

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = secrets.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        """Get GitHub client using provider factory."""
        try:
            # Use the standard provider factory instead of the GitHub-specific fallback
            provider = CodeProviderFactory.create_provider()
            return provider.client
        except Exception as e:
            logger.error(f"Failed to get GitHub client: {str(e)}")
            raise Exception(f"Repository {repo_name} not found or inaccessible")

    def _fetch_github_content(
        self, repo_name: str, issue_number: Optional[int], is_pull_request: bool
    ) -> Optional[Dict[str, Any]]:
        try:
            github = self._get_github_client(repo_name)

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
                f"[CODE_PROVIDER_TOOL] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
            )

            repo = github.get_repo(actual_repo_name)

            if issue_number is None:
                # Fetch all issues/PRs
                if is_pull_request:
                    items = list(
                        repo.get_pulls(state="all")[:10]
                    )  # Limit to 10 most recent
                else:
                    items = list(
                        repo.get_issues(state="all")[:10]
                    )  # Limit to 10 most recent

                return {
                    "success": True,
                    "content": [
                        {
                            "number": item.number,
                            "title": item.title,
                            "state": item.state,
                            "created_at": item.created_at.isoformat(),
                            "updated_at": item.updated_at.isoformat(),
                            "body": item.body,
                            "url": item.html_url,
                        }
                        for item in items
                    ],
                    "metadata": {
                        "repo": repo_name,
                        "type": "pull_requests" if is_pull_request else "issues",
                        "count": len(items),
                    },
                }
            else:
                try:
                    # Fetch specific issue/PR
                    if is_pull_request:
                        item = repo.get_pull(issue_number)
                        diff = item.get_files()
                        changes = [
                            {
                                "filename": file.filename,
                                "status": file.status,
                                "additions": file.additions,
                                "deletions": file.deletions,
                                "changes": file.changes,
                                "patch": file.patch if file.patch else None,
                            }
                            for file in diff
                        ]
                    else:
                        item = repo.get_issue(issue_number)
                        changes = None

                    return {
                        "success": True,
                        "content": {
                            "number": item.number,
                            "title": item.title,
                            "state": item.state,
                            "created_at": item.created_at.isoformat(),
                            "updated_at": item.updated_at.isoformat(),
                            "body": item.body,
                            "url": item.html_url,
                            "changes": changes,
                        },
                        "metadata": {
                            "repo": repo_name,
                            "type": "pull_request" if is_pull_request else "issue",
                            "number": issue_number,
                        },
                    }
                except UnknownObjectException:
                    return {
                        "success": False,
                        "error": f"{'Pull request' if is_pull_request else 'Issue'} #{issue_number} not found in {repo_name}",
                        "content": None,
                    }

        except Exception as e:
            logger.error(f"Error fetching GitHub content: {str(e)}")
            return None


def code_provider_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    from app.modules.code_provider.provider_factory import has_code_provider_credentials

    if not has_code_provider_credentials():
        logger.warning(
            "No code provider credentials configured. Please set CODE_PROVIDER_TOKEN, "
            "GH_TOKEN_LIST, GITHUB_APP_ID, or CODE_PROVIDER_USERNAME/PASSWORD."
        )
        return None

    tool_instance = CodeProviderTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Code Provider Content Fetcher",
        description="""Fetches repository issues and pull request information including diffs.
        :param repo_name: string, the full repository name (owner/repo)
        :param issue_number: optional int, the issue or PR number to fetch
        :param is_pull_request: optional bool, whether to fetch a PR (True) or issue (False)

            example:
            {
                "repo_name": "owner/repo",
                "issue_number": 123,
                "is_pull_request": true
            }

        Returns dictionary containing the issue/PR content, metadata, and success status.""",
        args_schema=CodeProviderToolInput,
    )
