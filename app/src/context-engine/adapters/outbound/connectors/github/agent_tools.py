"""GitHub agent tools: fetch PR / commit / review / issue data via ``GitHubReadPort``.

Tools surface a single repo's read API to the agent. Each tool takes the
``repo_name`` explicitly so the agent can disambiguate when the pot has
multiple repos. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_github_tools(...)])``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from adapters.outbound.connectors.github.api_client import GitHubReadPort
from domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


def build_github_tools(
    source_for_repo: Callable[[str], GitHubReadPort],
    allowed_repos_for_pot: Callable[[str], set[str]] | None = None,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes GitHub read endpoints.

    Args:
        source_for_repo: Resolves ``repo_name`` ("owner/repo") to a
            :class:`GitHubReadPort`. Typically
            :attr:`ContextEngineContainer.source_for_repo`.
        allowed_repos_for_pot: Resolves the pot id being reconciled to the
            set of ``owner/repo`` names attached to that pot. **Required for
            tenant isolation**: every tool rejects a model-supplied
            ``repo_name`` that is not in this set *before* touching
            ``source_for_repo`` (which authenticates with a shared org
            credential). When unwired or unresolvable the builder fails
            closed — all GitHub tool calls return ``unknown_repo`` — so a
            prompt-injected agent can never exfiltrate a repo the pot has
            no relationship to (security review C-5).

    Returns:
        A callable matching the agent's ``add_extra_tools`` contract.
    """

    def _builder(state: Any) -> list[Any]:
        pot_id = getattr(state, "pot_id", None)
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

        # Resolve the pot's attached-repo allowlist once per batch. ``None``
        # means we could not establish scope → fail closed (deny all).
        allowed: set[str] | None = None
        if allowed_repos_for_pot is not None and pot_id:
            try:
                allowed = {
                    str(r).strip().lower()
                    for r in allowed_repos_for_pot(str(pot_id))
                    if r
                }
            except Exception:
                logger.exception(
                    "github tools: failed to resolve repo allowlist for pot %s",
                    pot_id,
                )
                allowed = set()

        def _repo_allowed(repo_name: str) -> bool:
            if allowed is None:
                return False
            return bool(repo_name) and repo_name.strip().lower() in allowed

        def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
            """Reject any repo not attached to the pot before calling out."""

            @functools.wraps(fn)
            def _wrapped(repo_name: str, *args: Any, **kwargs: Any) -> Any:
                if not _repo_allowed(repo_name):
                    logger.warning(
                        "github tool %s blocked: repo %r not attached to "
                        "pot %s",
                        getattr(fn, "__name__", "?"),
                        repo_name,
                        pot_id,
                    )
                    return {"error": "unknown_repo", "repo_name": repo_name}
                return fn(repo_name, *args, **kwargs)

            return _wrapped

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
                return {"error": safe_error(exc)}

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
                return {"error": safe_error(exc)}

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
                return {"error": safe_error(exc)}

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
                return {"error": safe_error(exc)}

        def github_get_issue(repo_name: str, issue_number: int) -> dict[str, Any]:
            """Fetch one issue (title/body/state/labels/assignees)."""
            try:
                return _resolve(repo_name).get_issue(repo_name, int(issue_number))
            except Exception as exc:
                logger.exception("github_get_issue %s#%s failed", repo_name, issue_number)
                return {"error": safe_error(exc)}

        def github_list_pull_requests(
            repo_name: str,
            state: str = "all",
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate PRs (compact refs) for the backfill todo list."""
            try:
                items = _resolve(repo_name).list_pull_requests(
                    repo_name, state=state, limit=limit
                )
            except Exception as exc:
                logger.exception("github_list_pull_requests %s failed", repo_name)
                return {"error": safe_error(exc)}
            return {"repo_name": repo_name, "count": len(items), "pull_requests": items}

        def github_list_issues(
            repo_name: str,
            state: str = "all",
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate issues (compact refs, PRs excluded) for the backfill todo list."""
            try:
                items = _resolve(repo_name).list_issues(
                    repo_name, state=state, limit=limit
                )
            except Exception as exc:
                logger.exception("github_list_issues %s failed", repo_name)
                return {"error": safe_error(exc)}
            return {"repo_name": repo_name, "count": len(items), "issues": items}

        return [
            Tool(
                _guard(github_get_pull_request),
                name="github_get_pull_request",
                description=(
                    "Fetch one pull request by number from a GitHub repo (owner/repo). "
                    "Returns title, body, state, head/base branches, labels, milestone. "
                    "Set include_diff=true to also return per-file patches."
                ),
            ),
            Tool(
                _guard(github_get_pull_request_commits),
                name="github_get_pull_request_commits",
                description=(
                    "List commits on a pull request, including SHA, author login, "
                    "and commit message."
                ),
            ),
            Tool(
                _guard(github_get_pull_request_review_comments),
                name="github_get_pull_request_review_comments",
                description=(
                    "List inline review comments on a pull request "
                    "(per file/line, with author and body)."
                ),
            ),
            Tool(
                _guard(github_get_pull_request_issue_comments),
                name="github_get_pull_request_issue_comments",
                description=(
                    "List conversation-thread comments on a pull request "
                    "(top-level body/author/created_at, not inline review comments)."
                ),
            ),
            Tool(
                _guard(github_get_issue),
                name="github_get_issue",
                description="Fetch one GitHub issue by number (title/body/state/labels).",
            ),
            Tool(
                _guard(github_list_pull_requests),
                name="github_list_pull_requests",
                description=(
                    "Enumerate a repo's pull requests as compact refs "
                    "(number/title/state/merged/updated_at/author), newest "
                    "first. Bounded by the server-side backfill window and a "
                    "hard item cap — older/overflow PRs are intentionally "
                    "omitted. Use this to seed the backfill todo list, then "
                    "hydrate each PR with github_get_pull_request. state is "
                    "'open' | 'closed' | 'all' (default 'all')."
                ),
            ),
            Tool(
                _guard(github_list_issues),
                name="github_list_issues",
                description=(
                    "Enumerate a repo's issues as compact refs "
                    "(number/title/state/updated_at/author), newest first, "
                    "pull requests excluded. Bounded by the same backfill "
                    "window + item cap as github_list_pull_requests. Use it to "
                    "seed the backfill todo list, then hydrate each with "
                    "github_get_issue. state is 'open' | 'closed' | 'all'."
                ),
            ),
        ]

    return _builder
