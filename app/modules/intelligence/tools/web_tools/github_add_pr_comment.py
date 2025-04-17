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


class GitHubPRComment(BaseModel):
    """Model for a single GitHub PR comment."""

    file_path: str = Field(..., description="The path of the file to comment on")
    line_number: int = Field(..., description="The line number to comment on")
    comment_body: str = Field(..., description="The text content of the comment")
    code_snippet: Optional[str] = Field(
        default=None,
        description="Optional code snippet from the PR to reference in the comment",
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="Optional code suggestion to replace the referenced code",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="For multi-line comments, the starting line number (inclusive)",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="For multi-line comments, the ending line number (inclusive)",
    )


class GitAddPRCommentsInput(BaseModel):
    """Input for adding multiple comments to a GitHub pull request."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    pr_number: int = Field(..., description="The pull request number to comment on")
    comments: List[GitHubPRComment] = Field(
        ..., description="List of comments to add to the PR"
    )
    general_comment: Optional[str] = Field(
        default=None, description="Optional general comment for the entire PR"
    )
    review_action: str = Field(
        default="COMMENT",
        description="Review action to take: 'COMMENT', 'APPROVE', or 'REQUEST_CHANGES'",
    )


class GitAddPRCommentsTool:
    """Tool for adding multiple comments to GitHub pull requests with code snippet references."""

    name: str = "Add comments to a GitHub pull request"
    description: str = """
    Add multiple comments to a GitHub pull request.
    Can add general comments, specific file comments, reference code snippets, and suggest code changes.
    Supports full GitHub-style code review functionality.
    """
    args_schema: Type[BaseModel] = GitAddPRCommentsInput

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
        if not GitAddPRCommentsTool.gh_token_list:
            GitAddPRCommentsTool.initialize_tokens()

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

    def _format_comment_body(self, comment: GitHubPRComment) -> str:
        """Format a comment body with code snippet and suggestion if provided."""
        body = comment.comment_body

        # Add code snippet reference if provided
        if comment.code_snippet:
            body += f"\n\n```\n{comment.code_snippet}\n```"

        # Add suggestion if provided
        if comment.suggestion:
            body += f"\n\n```suggestion\n{comment.suggestion}\n```"

        return body

    def _run(
        self,
        repo_name: str,
        pr_number: int,
        comments: List[GitHubPRComment],
        general_comment: Optional[str] = None,
        review_action: str = "COMMENT",
    ) -> Dict[str, Any]:
        """
        Add multiple comments to a GitHub pull request.

        Args:
            repo_name: The full name of the repository (e.g., 'username/repo_name')
            pr_number: The number of the pull request to comment on
            comments: List of comments to add to the PR with file paths and line numbers
            general_comment: Optional general comment for the entire PR
            review_action: Review action to take: 'COMMENT', 'APPROVE', or 'REQUEST_CHANGES'

        Returns:
            Dict containing the result of the PR comment operation
        """
        # Validate review_action
        valid_actions = ["COMMENT", "APPROVE", "REQUEST_CHANGES"]
        if review_action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid review_action: {review_action}. Must be one of: {', '.join(valid_actions)}",
            }

        try:
            # Initialize GitHub client
            g = self._get_github_client(repo_name)
            repo = g.get_repo(repo_name)

            # Get the pull request
            try:
                pr = repo.get_pull(pr_number)
            except GithubException as e:
                return {
                    "success": False,
                    "error": f"Pull request #{pr_number} not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            # If no comments and no general comment, return error
            if not comments and not general_comment:
                return {
                    "success": False,
                    "error": "Must provide at least one comment or a general comment",
                }

            # If only general comment without file comments, add as issue comment
            if not comments and general_comment:
                comment = pr.create_issue_comment(general_comment)
                return {
                    "success": True,
                    "operation": "add_general_comment",
                    "pr_number": pr_number,
                    "comment_id": comment.id,
                    "url": comment.html_url,
                }

            # Get the latest commit in the PR for review comments
            commits = list(pr.get_commits())
            if not commits:
                return {
                    "success": False,
                    "error": "No commits found in this pull request",
                }
            latest_commit = commits[-1]

            # Prepare review comments
            review_comments = []
            errors = []

            for idx, comment in enumerate(comments):
                try:
                    # Format the comment body with code snippet and suggestion if provided
                    formatted_body = self._format_comment_body(comment)

                    # Prepare comment data
                    comment_data = {
                        "path": comment.file_path,
                        "position": comment.line_number,
                        "body": formatted_body,
                    }

                    # Handle multi-line comments if start_line and end_line are provided
                    if comment.start_line is not None and comment.end_line is not None:
                        comment_data["start_line"] = comment.start_line
                        comment_data["line"] = comment.end_line
                        # In multi-line mode, position refers to the end line
                        comment_data["position"] = comment.end_line

                    review_comments.append(comment_data)
                except Exception as e:
                    errors.append(f"Error with comment {idx+1}: {str(e)}")

            # If we have errors with any comments, return them
            if errors:
                return {
                    "success": False,
                    "error": "Errors occurred while preparing comments",
                    "details": errors,
                }

            # Create the review with all comments
            review_body = general_comment if general_comment else ""

            review = pr.create_review(
                commit=latest_commit,
                body=review_body,
                event=review_action,
                comments=review_comments,
            )

            return {
                "success": True,
                "operation": "add_pr_comments",
                "pr_number": pr_number,
                "review_id": review.id,
                "action": review_action,
                "url": pr.html_url,
                "comments_count": len(review_comments),
                "errors": errors if errors else None,
            }

        except GithubException as e:
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            return {"success": False, "error": f"Error adding PR comments: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        pr_number: int,
        comments: List[GitHubPRComment],
        general_comment: Optional[str] = None,
        review_action: str = "COMMENT",
    ) -> Dict[str, Any]:
        """Async implementation of the tool."""
        # For simplicity, we're using the sync version in async context
        # In a production environment, you'd want to use aiohttp or similar
        return self._run(
            repo_name=repo_name,
            pr_number=pr_number,
            comments=comments,
            general_comment=general_comment,
            review_action=review_action,
        )


def git_add_pr_comments_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    if not os.getenv("GITHUB_APP_ID") or not config_provider.get_github_key():
        logging.warning(
            "GitHub app credentials not set, GitHub tool will not be initialized"
        )
        return None

    tool_instance = GitAddPRCommentsTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Add comments to a GitHub pull request",
        description="""
        Add multiple comments to a GitHub pull request.
        Can add general comments, specific file comments, reference code snippets, and suggest code changes.
        Supports full GitHub-style code review functionality.
        """,
        args_schema=GitAddPRCommentsInput,
    )
