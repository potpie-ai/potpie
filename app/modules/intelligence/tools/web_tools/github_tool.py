import asyncio
import os
from typing import Any, Dict, Optional

from github import Github
from github.GithubException import UnknownObjectException
from app.modules.intelligence.tools.tool_schema import OnyxTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.parsing.utils.repo_name_normalizer import (
    get_actual_repo_name_for_lookup,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RepositoryAccessError(Exception):
    """Raised when a repository cannot be accessed via the configured provider."""


class GithubToolInput(BaseModel):
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


class GithubTool:
    name = "GitHub Tool"
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

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

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
        except RepositoryAccessError as e:
            logger.exception(
                "Repository access error", repo_name=repo_name, user_id=self.user_id
            )
            return {
                "success": False,
                "error": str(e),
                "content": None,
            }
        except Exception:
            logger.exception(
                "An unexpected error occurred",
                repo_name=repo_name,
                user_id=self.user_id,
            )
            return {
                "success": False,
                "error": "An unexpected error occurred",
                "content": None,
            }

    def _get_github_client(self, repo_name: str) -> Github:
        """Get GitHub client using provider factory with PAT-first fallback logic."""
        try:
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        except ValueError as e:
            logger.exception(
                "Failed to create provider for repository '%s': %s", repo_name, str(e)
            )
            raise RepositoryAccessError(
                f"Repository {repo_name} not found or inaccessible"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error creating provider for repository '%s': %s",
                repo_name,
                str(e),
            )
            raise RepositoryAccessError(
                f"Repository {repo_name} not found or inaccessible"
            ) from e

        if provider is None:
            message = (
                f"Provider factory returned None for repository '{repo_name}'. "
                "Unable to obtain client."
            )
            logger.error(message)
            raise RepositoryAccessError(message)

        client = getattr(provider, "client", None)
        if client is None:
            message = (
                f"Provider '{type(provider).__name__}' does not expose a client for "
                f"repository '{repo_name}'."
            )
            logger.error(message)
            raise RepositoryAccessError(message)

        if not hasattr(client, "get_repo"):
            message = (
                f"Client of type '{type(client).__name__}' for repository "
                f"'{repo_name}' does not support required operations."
            )
            logger.error(message)
            raise RepositoryAccessError(message)

        return client

    def _fetch_github_content(
        self, repo_name: str, issue_number: Optional[int], is_pull_request: bool
    ) -> Optional[Dict[str, Any]]:
        try:
            github = self._get_github_client(repo_name)

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            actual_repo_name = get_actual_repo_name_for_lookup(repo_name, provider_type)
            logger.info(
                "[GITHUB_TOOL] Provider type: %s, Original repo: %s, Actual repo for API: %s",
                provider_type,
                repo_name,
                actual_repo_name,
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
                    missing_item = "Pull request" if is_pull_request else "Issue"
                    return {
                        "success": False,
                        "error": f"{missing_item} #{issue_number} not found in {repo_name}",
                        "content": None,
                    }

        except RepositoryAccessError:
            raise
        except Exception:
            logger.exception(
                "Error fetching GitHub content",
                repo_name=repo_name,
                issue_number=issue_number,
                is_pull_request=is_pull_request,
            )
            return None

    @staticmethod
    def _has_pat_credentials() -> bool:
        return bool(os.getenv("CODE_PROVIDER_TOKEN") or os.getenv("GH_TOKEN_LIST"))

    @staticmethod
    def _has_app_credentials() -> bool:
        return bool(os.getenv("GITHUB_APP_ID") and config_provider.get_github_key())


def github_tool(sql_db: Session, user_id: str) -> Optional[OnyxTool]:
    # Initialize when either PAT-based credentials or App credentials are present
    if not (GithubTool._has_pat_credentials() or GithubTool._has_app_credentials()):
        logger.warning(
            "GitHub credentials not set (PAT or App). GitHub tool will not be initialized"
        )
        return None

    tool_instance = GithubTool(sql_db, user_id)
    return OnyxTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="GitHub Content Fetcher",
        description="""Fetches GitHub issues and pull request information including diffs.
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
        args_schema=GithubToolInput,
    )
