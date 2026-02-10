import os
import secrets
from app.modules.utils.logger import setup_logger

from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.code_provider.provider_factory import CodeProviderFactory

logger = setup_logger(__name__)


class CodeProviderPRComment(BaseModel):
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


class CodeProviderAddPRCommentsInput(BaseModel):
    """Input for adding multiple comments to a GitHub pull request."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    pr_number: int = Field(..., description="The pull request number to comment on")
    comments: List[CodeProviderPRComment] = Field(
        ..., description="List of comments to add to the PR"
    )
    general_comment: Optional[str] = Field(
        default=None, description="Optional general comment for the entire PR"
    )
    review_action: str = Field(
        default="COMMENT",
        description="Review action to take: 'COMMENT', 'APPROVE', or 'REQUEST_CHANGES'",
    )


class CodeProviderAddPRCommentsTool:
    """Tool for adding multiple comments to GitHub pull requests with code snippet references."""

    name: str = "Add comments to a pull request"
    description: str = """
    Add multiple comments to a GitHub pull request.
    Can add general comments, specific file comments, reference code snippets, and suggest code changes.
    Supports full GitHub-style code review functionality.
    """
    args_schema: Type[BaseModel] = CodeProviderAddPRCommentsInput

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
        if not CodeProviderAddPRCommentsTool.gh_token_list:
            CodeProviderAddPRCommentsTool.initialize_tokens()

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = secrets.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        """Get code provider client using provider factory."""
        try:
            logger.info(f"[ADD_PR_COMMENT] Creating provider for repo: {repo_name}")
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
            logger.info(
                f"[ADD_PR_COMMENT] Provider created successfully, type: {type(provider).__name__}"
            )
            logger.info(
                f"[ADD_PR_COMMENT] Client object: {type(provider.client).__name__}"
            )
            return provider.client
        except Exception as e:
            logger.exception("[ADD_PR_COMMENT] Failed to get client")
            raise Exception(
                f"Repository {repo_name} not found or inaccessible: {str(e)}"
            )

    def _format_comment_body(self, comment: CodeProviderPRComment) -> str:
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
        comments: List[CodeProviderPRComment],
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
        logger.info(
            f"[ADD_PR_COMMENT] Starting PR comment operation: repo={repo_name}, pr={pr_number}, action={review_action}, num_comments={len(comments) if comments else 0}"
        )

        # Validate review_action
        valid_actions = ["COMMENT", "APPROVE", "REQUEST_CHANGES"]
        if review_action not in valid_actions:
            logger.error(f"[ADD_PR_COMMENT] Invalid review_action: {review_action}")
            return {
                "success": False,
                "error": f"Invalid review_action: {review_action}. Must be one of: {', '.join(valid_actions)}",
            }

        try:
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

            # Get provider for URL construction later
            provider = CodeProviderFactory.create_provider_with_fallback(
                normalized_input
            )
            g = provider.client

            logger.info(
                f"[ADD_PR_COMMENT] Provider type: {provider_type}, Original repo: {repo_name}, Normalized: {normalized_input}, Actual repo for API: {actual_repo_name}"
            )

            repo = g.get_repo(actual_repo_name)
            logger.info(f"[ADD_PR_COMMENT] Successfully got repo object: {repo.name}")

            # Get the pull request
            try:
                logger.info(f"[ADD_PR_COMMENT] Getting PR #{pr_number}")
                pr = repo.get_pull(pr_number)
                logger.info(
                    f"[ADD_PR_COMMENT] Successfully got PR #{pr.number}: {pr.title}"
                )
            except GithubException as e:
                logger.error(
                    f"[ADD_PR_COMMENT] PR #{pr_number} not found: status={e.status}, data={e.data}"
                )
                return {
                    "success": False,
                    "error": f"Pull request #{pr_number} not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                }

            # If no comments and no general comment, return error
            if not comments and not general_comment:
                logger.error("[ADD_PR_COMMENT] No comments or general comment provided")
                return {
                    "success": False,
                    "error": "Must provide at least one comment or a general comment",
                }

            # If only general comment without file comments, add as issue comment
            if not comments and general_comment:
                logger.info("[ADD_PR_COMMENT] Adding general comment only")

                # For GitBucket, use raw API call to avoid URL validation issues
                if provider_type == "gitbucket":
                    logger.info(
                        "[ADD_PR_COMMENT] Using raw API call for GitBucket compatibility"
                    )
                    try:
                        import json

                        # Make raw API request for comment
                        post_parameters = {"body": general_comment}
                        headers, data = repo._requester.requestJsonAndCheck(
                            "POST",
                            f"{repo.url}/issues/{pr_number}/comments",
                            input=post_parameters,
                        )
                        logger.info(
                            f"[ADD_PR_COMMENT] Raw API response received (type: {type(data)}): {data}"
                        )

                        # Parse JSON string if needed
                        if isinstance(data, str):
                            logger.info("[ADD_PR_COMMENT] Parsing JSON string response")
                            data = json.loads(data)

                        comment_id = data.get("id")
                        comment_url = data.get("html_url")
                        logger.info(
                            f"[ADD_PR_COMMENT] Successfully added general comment: {comment_id}"
                        )

                        return {
                            "success": True,
                            "operation": "add_general_comment",
                            "pr_number": pr_number,
                            "comment_id": comment_id,
                            "url": comment_url,
                        }
                    except Exception as e:
                        logger.exception("[ADD_PR_COMMENT] Raw API call failed")
                        return {
                            "success": False,
                            "error": f"Failed to add comment via raw API: {str(e)}",
                        }

                # For GitHub, use standard PyGithub method
                comment = pr.create_issue_comment(general_comment)
                logger.info(
                    f"[ADD_PR_COMMENT] Successfully added general comment: {comment.id}"
                )
                return {
                    "success": True,
                    "operation": "add_general_comment",
                    "pr_number": pr_number,
                    "comment_id": comment.id,
                    "url": comment.html_url,
                }

            # Get the latest commit in the PR for review comments
            logger.info("[ADD_PR_COMMENT] Getting commits from PR")
            commits = list(pr.get_commits())
            if not commits:
                logger.error("[ADD_PR_COMMENT] No commits found in PR")
                return {
                    "success": False,
                    "error": "No commits found in this pull request",
                }
            latest_commit = commits[-1]
            logger.info(f"[ADD_PR_COMMENT] Latest commit: {latest_commit.sha}")

            # Prepare review comments
            review_comments = []
            errors = []

            for idx, comment in enumerate(comments):
                try:
                    logger.info(
                        f"[ADD_PR_COMMENT] Processing comment {idx + 1}/{len(comments)}: file={comment.file_path}, line={comment.line_number}"
                    )
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
                        logger.info(
                            f"[ADD_PR_COMMENT] Multi-line comment: start={comment.start_line}, end={comment.end_line}"
                        )
                        comment_data["start_line"] = comment.start_line
                        comment_data["line"] = comment.end_line
                        # In multi-line mode, position refers to the end line
                        comment_data["position"] = comment.end_line

                    review_comments.append(comment_data)
                    logger.info(
                        f"[ADD_PR_COMMENT] Successfully prepared comment {idx + 1}"
                    )
                except Exception as e:
                    error_msg = f"Error with comment {idx + 1}: {str(e)}"
                    logger.exception(f"[ADD_PR_COMMENT] Error with comment {idx + 1}")
                    errors.append(error_msg)

            # If we have errors with any comments, return them
            if errors:
                logger.error(f"[ADD_PR_COMMENT] Errors preparing comments: {errors}")
                return {
                    "success": False,
                    "error": "Errors occurred while preparing comments",
                    "details": errors,
                }

            # Create the review with all comments
            review_body = general_comment if general_comment else ""
            logger.info(
                f"[ADD_PR_COMMENT] Creating review with {len(review_comments)} comments"
            )

            # For GitBucket, use raw API call for reviews
            if provider_type == "gitbucket":
                logger.info(
                    "[ADD_PR_COMMENT] Using raw API call for GitBucket review compatibility"
                )
                try:
                    import json

                    # GitBucket may have limited review API support, fall back to individual comments
                    logger.info(
                        "[ADD_PR_COMMENT] Adding comments individually for GitBucket"
                    )
                    added_comments = []

                    for idx, comment in enumerate(review_comments):
                        try:
                            # Add each comment individually
                            post_params = {
                                "body": comment["body"],
                                "commit_id": latest_commit.sha,
                                "path": comment["path"],
                                "position": comment["position"],
                            }

                            headers, data = repo._requester.requestJsonAndCheck(
                                "POST",
                                f"{repo.url}/pulls/{pr_number}/comments",
                                input=post_params,
                            )

                            if isinstance(data, str):
                                data = json.loads(data)

                            added_comments.append(data.get("id"))
                            logger.info(
                                f"[ADD_PR_COMMENT] Added comment {idx + 1}/{len(review_comments)}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[ADD_PR_COMMENT] Failed to add comment {idx + 1}: {str(e)}"
                            )
                            errors.append(f"Comment {idx + 1} failed: {str(e)}")

                    # Add general comment if provided
                    if review_body:
                        try:
                            post_params = {"body": review_body}
                            headers, data = repo._requester.requestJsonAndCheck(
                                "POST",
                                f"{repo.url}/issues/{pr_number}/comments",
                                input=post_params,
                            )
                            logger.info("[ADD_PR_COMMENT] Added general review comment")
                        except Exception as e:
                            logger.error(
                                f"[ADD_PR_COMMENT] Failed to add general comment: {str(e)}"
                            )

                    # Use the normalized input we already computed
                    normalized_repo_name = normalized_input
                    # Construct GitBucket URL using normalized name
                    base_url = (
                        provider.get_api_base_url()
                        if hasattr(provider, "get_api_base_url")
                        else "http://localhost:8080"
                    )
                    if base_url.endswith("/api/v3"):
                        base_url = base_url[:-7]  # Remove '/api/v3'
                    pr_url = f"{base_url}/{normalized_repo_name}/pull/{pr_number}"

                    result = {
                        "success": True,
                        "operation": "add_pr_comments",
                        "pr_number": pr_number,
                        "review_id": None,  # GitBucket doesn't return review ID
                        "action": "COMMENT",  # GitBucket may not support review actions
                        "url": pr_url,
                        "comments_count": len(added_comments),
                        "errors": errors if errors else None,
                    }
                    logger.info(f"[ADD_PR_COMMENT] Returning success result: {result}")
                    return result
                except Exception as e:
                    logger.exception("[ADD_PR_COMMENT] Raw API call failed")
                    return {
                        "success": False,
                        "error": f"Failed to add comments via raw API: {str(e)}",
                    }

            # For GitHub, use standard PyGithub method
            review = pr.create_review(
                commit=latest_commit,
                body=review_body,
                event=review_action,
                comments=review_comments,
            )
            logger.info(f"[ADD_PR_COMMENT] Successfully created review: id={review.id}")

            result = {
                "success": True,
                "operation": "add_pr_comments",
                "pr_number": pr_number,
                "review_id": review.id,
                "action": review_action,
                "url": pr.html_url,
                "comments_count": len(review_comments),
                "errors": errors if errors else None,
            }
            logger.info(f"[ADD_PR_COMMENT] Returning success result: {result}")
            return result

        except GithubException as e:
            logger.error(
                f"[ADD_PR_COMMENT] GithubException caught: status={e.status}, data={e.data}, message={str(e)}"
            )
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            logger.exception("[ADD_PR_COMMENT] Unexpected exception")
            return {"success": False, "error": f"Error adding PR comments: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        pr_number: int,
        comments: List[CodeProviderPRComment],
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


def code_provider_add_pr_comments_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    from app.modules.code_provider.provider_factory import has_code_provider_credentials

    if not has_code_provider_credentials():
        logger.warning(
            "No code provider credentials configured. Please set CODE_PROVIDER_TOKEN, "
            "GH_TOKEN_LIST, GITHUB_APP_ID, or CODE_PROVIDER_USERNAME/PASSWORD."
        )
        return None

    tool_instance = CodeProviderAddPRCommentsTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Add comments to a pull request",
        description="""
        Add multiple comments to a GitHub pull request.
        Can add general comments, specific file comments, reference code snippets, and suggest code changes.
        Supports full GitHub-style code review functionality.
        """,
        args_schema=CodeProviderAddPRCommentsInput,
    )
