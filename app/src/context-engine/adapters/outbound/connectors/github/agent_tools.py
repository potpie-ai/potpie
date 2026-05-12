"""GitHub agent tools: fetch PR / commit / review / issue data via ``GitHubReadPort``.

Tools surface a single repo's read API to the agent. Each tool takes the
``repo_name`` explicitly so the agent can disambiguate when the pot has
multiple repos. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_github_tools(...)])``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from adapters.outbound.connectors.github.api_client import GitHubReadPort

logger = logging.getLogger(__name__)


def build_github_tools(
    source_for_repo: Callable[[str], GitHubReadPort],
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes GitHub read endpoints.

    Args:
        source_for_repo: Resolves ``repo_name`` ("owner/repo") to a
            :class:`GitHubReadPort`. Typically
            :attr:`ContextEngineContainer.source_for_repo`.

    Returns:
        A callable matching the agent's ``add_extra_tools`` contract.
    """

    def _builder(_state: Any) -> list[Any]:
        del _state
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping github tools"
                )
                return []

        def _resolve(repo_name: str) -> GitHubReadPort:
            return source_for_repo(repo_name)

        def github_get_pull_request(
            repo_name: str,
            pr_number: int,
            include_diff: bool = False,
        ) -> dict[str, Any]:
            """Fetch one pull request (title/body/state/branches/labels; diff optional)."""
            try:
                return _resolve(repo_name).get_pull_request(
                    repo_name, pr_number, include_diff=include_diff
                )
            except Exception as exc:
                logger.exception("github_get_pull_request %s#%s failed", repo_name, pr_number)
                return {"error": str(exc)}

        def github_get_pull_request_commits(
            repo_name: str,
            pr_number: int,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """List commits on a PR (sha, author, message, authored_at)."""
            try:
                return _resolve(repo_name).get_pull_request_commits(repo_name, pr_number)
            except Exception as exc:
                logger.exception(
                    "github_get_pull_request_commits %s#%s failed",
                    repo_name,
                    pr_number,
                )
                return {"error": str(exc)}

        def github_get_pull_request_review_comments(
            repo_name: str,
            pr_number: int,
            limit: int = 100,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Inline review comments on a PR (file/line/body/author)."""
            try:
                return _resolve(repo_name).get_pull_request_review_comments(
                    repo_name, pr_number, limit=int(limit)
                )
            except Exception as exc:
                logger.exception(
                    "github_get_pull_request_review_comments %s#%s failed",
                    repo_name,
                    pr_number,
                )
                return {"error": str(exc)}

        def github_get_pull_request_issue_comments(
            repo_name: str,
            pr_number: int,
            limit: int = 50,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Conversation-thread comments on a PR (body/author/created_at)."""
            try:
                return _resolve(repo_name).get_pull_request_issue_comments(
                    repo_name, pr_number, limit=int(limit)
                )
            except Exception as exc:
                logger.exception(
                    "github_get_pull_request_issue_comments %s#%s failed",
                    repo_name,
                    pr_number,
                )
                return {"error": str(exc)}

        def github_get_issue(repo_name: str, issue_number: int) -> dict[str, Any]:
            """Fetch one issue (title/body/state/labels/assignees)."""
            try:
                return _resolve(repo_name).get_issue(repo_name, int(issue_number))
            except Exception as exc:
                logger.exception("github_get_issue %s#%s failed", repo_name, issue_number)
                return {"error": str(exc)}

        return [
            Tool(
                github_get_pull_request,
                name="github_get_pull_request",
                description=(
                    "Fetch one pull request by number from a GitHub repo (owner/repo). "
                    "Returns title, body, state, head/base branches, labels, milestone. "
                    "Set include_diff=true to also return per-file patches."
                ),
            ),
            Tool(
                github_get_pull_request_commits,
                name="github_get_pull_request_commits",
                description=(
                    "List commits on a pull request, including SHA, author login, "
                    "and commit message."
                ),
            ),
            Tool(
                github_get_pull_request_review_comments,
                name="github_get_pull_request_review_comments",
                description=(
                    "List inline review comments on a pull request "
                    "(per file/line, with author and body)."
                ),
            ),
            Tool(
                github_get_pull_request_issue_comments,
                name="github_get_pull_request_issue_comments",
                description=(
                    "List conversation-thread comments on a pull request "
                    "(top-level body/author/created_at, not inline review comments)."
                ),
            ),
            Tool(
                github_get_issue,
                name="github_get_issue",
                description="Fetch one GitHub issue by number (title/body/state/labels).",
            ),
        ]

    return _builder
