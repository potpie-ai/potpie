"""GitLab agent tools: fetch MR / commit / review / issue data via ``GitLabReadPort``.

Tools surface a single project's read API to the agent. Each tool takes
the ``project`` path explicitly so the agent can disambiguate when the pot
has multiple projects. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_gitlab_tools(...)])``.

The tool surface is the GitLab counterpart of ``connectors/github/
agent_tools.py``, plus the review signals GitLab exposes and GitHub does
not (approvals, resolvable discussion threads, state events) and the
issue↔MR links that carry task tracking.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabReadPort,
)
from potpie_context_engine.domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


def build_gitlab_tools(
    source_for_project: Callable[[str], GitLabReadPort],
    allowed_projects_for_pot: Callable[[str], set[str]] | None = None,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes GitLab read endpoints.

    Args:
        source_for_project: Resolves a project path ("group/project", which
            may contain nested subgroups) to a :class:`GitLabReadPort`.
        allowed_projects_for_pot: Resolves the pot id being reconciled to
            the set of project paths attached to that pot. **Required for
            tenant isolation**: every tool rejects a model-supplied
            ``project`` that is not in this set *before* touching
            ``source_for_project`` (which authenticates with a shared
            instance credential). When unwired or unresolvable the builder
            fails closed — all GitLab tool calls return ``unknown_project``
            — so a prompt-injected agent can never exfiltrate a project the
            pot has no relationship to (security review C-5).

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
                    "pydantic-ai/pydantic-deep Tool not importable; skipping gitlab tools"
                )
                return []

        # Resolve the pot's attached-project allowlist once per batch.
        # ``None`` means we could not establish scope → fail closed.
        allowed: set[str] | None = None
        if allowed_projects_for_pot is not None and pot_id:
            try:
                allowed = {
                    str(p).strip().strip("/").lower()
                    for p in allowed_projects_for_pot(str(pot_id))
                    if p
                }
            except Exception:
                logger.exception(
                    "gitlab tools: failed to resolve project allowlist for pot %s",
                    pot_id,
                )
                allowed = set()

        def _project_allowed(project: str) -> bool:
            if allowed is None:
                return False
            return bool(project) and project.strip().strip("/").lower() in allowed

        def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
            """Reject any project not attached to the pot before calling out."""

            @functools.wraps(fn)
            def _wrapped(project: str, *args: Any, **kwargs: Any) -> Any:
                if not _project_allowed(project):
                    logger.warning(
                        "gitlab tool %s blocked: project %r not attached to pot %s",
                        getattr(fn, "__name__", "?"),
                        project,
                        pot_id,
                    )
                    return {"error": "unknown_project", "project": project}
                return fn(project, *args, **kwargs)

            return _wrapped

        def _resolve(project: str) -> GitLabReadPort:
            return source_for_project(project)

        def gitlab_get_merge_request(
            project: str,
            iid: int,
            include_diff: bool = False,
        ) -> dict[str, Any]:
            """Fetch one merge request (title/body/state/branches/labels; diff optional)."""
            try:
                return _resolve(project).get_merge_request(
                    project, int(iid), include_diff=include_diff
                )
            except Exception as exc:
                logger.exception("gitlab_get_merge_request %s!%s failed", project, iid)
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_commits(
            project: str,
            iid: int,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """List commits on a merge request (sha, author, message, committed_at)."""
            try:
                return _resolve(project).get_merge_request_commits(project, int(iid))
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_commits %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_discussions(
            project: str,
            iid: int,
            limit: int = 100,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Inline review comments on a merge request (file/line/body/author/resolved)."""
            try:
                return _resolve(project).get_merge_request_discussions(
                    project, int(iid), limit=int(limit)
                )
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_discussions %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_notes(
            project: str,
            iid: int,
            limit: int = 50,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Conversation-thread notes on a merge request, system notes included."""
            try:
                return _resolve(project).get_merge_request_notes(
                    project, int(iid), limit=int(limit)
                )
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_notes %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_approvals(
            project: str,
            iid: int,
        ) -> dict[str, Any]:
            """Approval state for a merge request (who approved, how many remain)."""
            try:
                return _resolve(project).get_merge_request_approvals(project, int(iid))
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_approvals %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_state_events(
            project: str,
            iid: int,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """State transitions on a merge request (opened/closed/merged/reopened)."""
            try:
                return _resolve(project).get_merge_request_state_events(
                    project, int(iid)
                )
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_state_events %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_merge_request_closes_issues(
            project: str,
            iid: int,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Issues a merge request closes on merge."""
            try:
                return _resolve(project).get_merge_request_closes_issues(
                    project, int(iid)
                )
            except Exception as exc:
                logger.exception(
                    "gitlab_get_merge_request_closes_issues %s!%s failed", project, iid
                )
                return {"error": safe_error(exc)}

        def gitlab_get_issue(project: str, iid: int) -> dict[str, Any]:
            """Fetch one issue (title/body/state/labels/assignees/milestone/due date)."""
            try:
                return _resolve(project).get_issue(project, int(iid))
            except Exception as exc:
                logger.exception("gitlab_get_issue %s#%s failed", project, iid)
                return {"error": safe_error(exc)}

        def gitlab_get_issue_notes(
            project: str,
            iid: int,
            limit: int = 50,
        ) -> list[dict[str, Any]] | dict[str, Any]:
            """Comments on an issue (body/author/created_at)."""
            try:
                return _resolve(project).get_issue_notes(
                    project, int(iid), limit=int(limit)
                )
            except Exception as exc:
                logger.exception("gitlab_get_issue_notes %s#%s failed", project, iid)
                return {"error": safe_error(exc)}

        def gitlab_get_issue_links(project: str, iid: int) -> dict[str, Any]:
            """Merge requests related to, or closing, an issue."""
            try:
                return _resolve(project).get_issue_links(project, int(iid))
            except Exception as exc:
                logger.exception("gitlab_get_issue_links %s#%s failed", project, iid)
                return {"error": safe_error(exc)}

        def gitlab_list_merge_requests(
            project: str,
            state: str = "all",
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate merge requests (compact refs) for the backfill todo list."""
            try:
                items = _resolve(project).list_merge_requests(
                    project, state=state, limit=limit
                )
            except Exception as exc:
                logger.exception("gitlab_list_merge_requests %s failed", project)
                return {"error": safe_error(exc)}
            return {
                "project": project,
                "count": len(items),
                "merge_requests": items,
            }

        def gitlab_list_issues(
            project: str,
            state: str = "all",
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate issues (compact refs) for the backfill todo list."""
            try:
                items = _resolve(project).list_issues(project, state=state, limit=limit)
            except Exception as exc:
                logger.exception("gitlab_list_issues %s failed", project)
                return {"error": safe_error(exc)}
            return {"project": project, "count": len(items), "issues": items}

        return [
            Tool(
                _guard(gitlab_get_merge_request),
                name="gitlab_get_merge_request",
                description=(
                    "Fetch one merge request by iid from a GitLab project "
                    "(group/project, subgroups allowed). Returns title, body, "
                    "state, source/target branches, labels, milestone, "
                    "reviewers, assignees, draft and merge status. Set "
                    "include_diff=true to also return per-file patches."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_commits),
                name="gitlab_get_merge_request_commits",
                description=(
                    "List commits on a merge request, including SHA, author, "
                    "and commit message."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_discussions),
                name="gitlab_get_merge_request_discussions",
                description=(
                    "List inline review comments on a merge request (per "
                    "file/line, with author, body, discussion thread id, and "
                    "whether the thread was resolved). This is GitLab's "
                    "equivalent of PR review comments."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_notes),
                name="gitlab_get_merge_request_notes",
                description=(
                    "List conversation-thread notes on a merge request "
                    "(body/author/created_at), oldest first. System notes are "
                    "included and are the review-action trail on GitLab CE — "
                    "'approved this merge request', 'requested review from', "
                    "'marked as draft'."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_approvals),
                name="gitlab_get_merge_request_approvals",
                description=(
                    "Approval state of a merge request: who approved, how many "
                    "approvals are required and left. Returns "
                    "available=false when the instance does not expose the "
                    "approvals endpoint — fall back to the system notes from "
                    "gitlab_get_merge_request_notes."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_state_events),
                name="gitlab_get_merge_request_state_events",
                description=(
                    "State transitions on a merge request (opened, closed, "
                    "merged, reopened) with the acting user and timestamp."
                ),
            ),
            Tool(
                _guard(gitlab_get_merge_request_closes_issues),
                name="gitlab_get_merge_request_closes_issues",
                description=(
                    "Issues that a merge request closes when it merges — the "
                    "merge-request-to-issue task-tracking link."
                ),
            ),
            Tool(
                _guard(gitlab_get_issue),
                name="gitlab_get_issue",
                description=(
                    "Fetch one GitLab issue by iid (title/body/state/labels/"
                    "assignees/milestone/due date/weight/time tracking)."
                ),
            ),
            Tool(
                _guard(gitlab_get_issue_notes),
                name="gitlab_get_issue_notes",
                description=(
                    "List comments on a GitLab issue (body/author/created_at), "
                    "oldest first."
                ),
            ),
            Tool(
                _guard(gitlab_get_issue_links),
                name="gitlab_get_issue_links",
                description=(
                    "Merge requests related to an issue and merge requests that "
                    "close it — use to connect a filed issue to the work that "
                    "resolved it."
                ),
            ),
            Tool(
                _guard(gitlab_list_merge_requests),
                name="gitlab_list_merge_requests",
                description=(
                    "Enumerate a project's merge requests as compact refs "
                    "(iid/title/state/merged/updated_at/author), newest first. "
                    "Bounded by the server-side backfill window and a hard item "
                    "cap — older/overflow merge requests are intentionally "
                    "omitted. Use this to seed the backfill todo list, then "
                    "hydrate each with gitlab_get_merge_request. state is "
                    "'opened' | 'closed' | 'merged' | 'all' (default 'all')."
                ),
            ),
            Tool(
                _guard(gitlab_list_issues),
                name="gitlab_list_issues",
                description=(
                    "Enumerate a project's issues as compact refs "
                    "(iid/title/state/updated_at/author/labels), newest first. "
                    "Bounded by the same backfill window + item cap as "
                    "gitlab_list_merge_requests. Use it to seed the backfill "
                    "todo list, then hydrate each with gitlab_get_issue. state "
                    "is 'opened' | 'closed' | 'all'."
                ),
            ),
        ]

    return _builder
