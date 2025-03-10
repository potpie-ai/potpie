import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Optional

import requests
from github import Github
from github.Auth import AppAuth
from github.GithubException import UnknownObjectException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider


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
        if not GithubTool.gh_token_list:
            GithubTool.initialize_tokens()

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
            logging.exception(f"An unexpected error occurred: {str(e)}")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "content": None,
            }

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

    def _fetch_github_content(
        self, repo_name: str, issue_number: Optional[int], is_pull_request: bool
    ) -> Optional[Dict[str, Any]]:
        try:
            github = self._get_github_client(repo_name)
            repo = github.get_repo(repo_name)

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
            logging.error(f"Error fetching GitHub content: {str(e)}")
            return None


def github_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    if not os.getenv("GITHUB_APP_ID") or not config_provider.get_github_key():
        logging.warning(
            "GitHub app credentials not set, GitHub tool will not be initialized"
        )
        return None

    tool_instance = GithubTool(sql_db, user_id)
    return StructuredTool.from_function(
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
