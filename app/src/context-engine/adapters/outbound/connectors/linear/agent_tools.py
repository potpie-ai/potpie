"""Linear agent tools: fetch issue detail via the host's ``LinearIssueFetcher``.

Mirrors :mod:`adapters.outbound.connectors.github.agent_tools`. The fetcher
resolves the Linear access token *per pot* (walking
``pot_id → context_graph_pot_sources → integrations`` and decrypting the
stored OAuth token), so reads run against whatever Linear account the pot's
owner connected — never a shared service key. The host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_linear_tools(...)])``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from adapters.outbound.connectors.linear.fetcher import LinearIssueFetcher
from domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


def build_linear_tools(
    fetcher: LinearIssueFetcher,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes Linear read endpoints.

    Args:
        fetcher: Resolves a Linear issue by identifier for a given pot.
            Typically ``ContextEngineLinearFetcher`` (per-pot OAuth token).

    Returns:
        A callable matching the agent's ``add_extra_tools`` contract. The
        builder closes over ``state.pot_id`` so token resolution stays scoped
        to the pot being reconciled.
    """

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping linear tools"
                )
                return []

        pot_id = getattr(state, "pot_id", None)

        def linear_list_issues(
            team_id: str | None = None,
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate Linear issues (compact refs) for the backfill todo list.

            Bounded by the same backfill window + item cap as the GitHub list
            tools. Hydrate each ref via ``linear_get_issue``.
            """
            from domain.backfill_window import (
                backfill_window_since,
                clamp_backfill_limit,
            )

            try:
                issues = fetcher.list_issues(
                    pot_id=pot_id,
                    team_id=team_id,
                    updated_after=backfill_window_since(),
                    limit=clamp_backfill_limit(limit),
                )
            except PermissionError as exc:
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_list_issues failed pot=%s", pot_id)
                return {"error": safe_error(exc)}
            return {"count": len(issues), "issues": issues}

        def linear_list_projects(
            team_id: str | None = None,
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate Linear projects (compact refs) for the backfill todo list."""
            from domain.backfill_window import (
                backfill_window_since,
                clamp_backfill_limit,
            )

            try:
                projects = fetcher.list_projects(
                    pot_id=pot_id,
                    team_id=team_id,
                    updated_after=backfill_window_since(),
                    limit=clamp_backfill_limit(limit),
                )
            except PermissionError as exc:
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_list_projects failed pot=%s", pot_id)
                return {"error": safe_error(exc)}
            return {"count": len(projects), "projects": projects}

        def linear_get_project(project_id: str) -> dict[str, Any]:
            """Fetch one Linear project's detail by id (name/description/state/lead/dates)."""
            try:
                project = fetcher.get_project(project_id, pot_id=pot_id)
            except PermissionError as exc:
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_get_project %s failed", project_id)
                return {"error": safe_error(exc)}
            if project is None:
                return {"found": False, "project_id": project_id}
            return {"found": True, "project": project}

        def linear_list_documents(
            team_id: str | None = None,
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Enumerate Linear documents (compact refs) for the backfill todo list."""
            from domain.backfill_window import (
                backfill_window_since,
                clamp_backfill_limit,
            )

            try:
                docs = fetcher.list_documents(
                    pot_id=pot_id,
                    team_id=team_id,
                    updated_after=backfill_window_since(),
                    limit=clamp_backfill_limit(limit),
                )
            except PermissionError as exc:
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_list_documents failed pot=%s", pot_id)
                return {"error": safe_error(exc)}
            return {"count": len(docs), "documents": docs}

        def linear_get_document(document_id: str) -> dict[str, Any]:
            """Fetch one Linear document's detail by id (title/content/project/creator)."""
            try:
                doc = fetcher.get_document(document_id, pot_id=pot_id)
            except PermissionError as exc:
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_get_document %s failed", document_id)
                return {"error": safe_error(exc)}
            if doc is None:
                return {"found": False, "document_id": document_id}
            return {"found": True, "document": doc}

        def linear_get_issue(issue_id: str) -> dict[str, Any]:
            """Fetch one Linear issue's detail by identifier (``ENG-123`` or UUID)."""
            try:
                issue = fetcher.get_issue(issue_id, pot_id=pot_id)
            except PermissionError as exc:
                # Auth/token problem for this pot's Linear connection — surface
                # it so the agent adds a warning instead of inventing facts.
                return {"error": "linear_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("linear_get_issue %s failed", issue_id)
                return {"error": safe_error(exc)}
            if issue is None:
                return {"found": False, "issue_id": issue_id}
            return {"found": True, "issue": issue}

        tools = [
            Tool(
                linear_get_issue,
                name="linear_get_issue",
                description=(
                    "Fetch one Linear issue by its identifier (team-prefixed "
                    "key like 'ENG-123' or the opaque Linear UUID). Returns the "
                    "issue detail (title, description, state, assignee, team, "
                    "labels, timestamps) for the Linear account connected to "
                    "this pot. Use it to ground issue references found in "
                    "commits, PR bodies, or webhook payloads."
                ),
            ),
        ]
        # Enumeration is an optional fetcher capability — only surface the
        # backfill list tool when the wired fetcher actually implements it,
        # so minimal fakes / single-issue resolvers stay valid.
        if callable(getattr(fetcher, "list_issues", None)):
            tools.append(
                Tool(
                    linear_list_issues,
                    name="linear_list_issues",
                    description=(
                        "Enumerate Linear issues for this pot's connected team "
                        "as compact refs (id/identifier/updated_at), newest "
                        "first. Bounded by the backfill window + item cap — "
                        "older/overflow issues are intentionally omitted and "
                        "resolve lazily via linear_get_issue. Use this to seed "
                        "the backfill todo list. Pass team_id only for a "
                        "multi-team pot; otherwise the pot's team is used."
                    ),
                )
            )
        if callable(getattr(fetcher, "list_projects", None)):
            tools.append(
                Tool(
                    linear_list_projects,
                    name="linear_list_projects",
                    description=(
                        "Enumerate Linear projects for this pot's connected "
                        "team as compact refs (id/name/updated_at), newest "
                        "first. Same backfill window + item cap as "
                        "linear_list_issues. Seed the todo list, then hydrate "
                        "each via linear_get_project."
                    ),
                )
            )
        if callable(getattr(fetcher, "get_project", None)):
            tools.append(
                Tool(
                    linear_get_project,
                    name="linear_get_project",
                    description=(
                        "Fetch one Linear project's detail by id "
                        "(name, description, state, lead, start/target dates, "
                        "teams). Use it to seed a project as durable context "
                        "(canonical label Feature or Document)."
                    ),
                )
            )
        if callable(getattr(fetcher, "list_documents", None)):
            tools.append(
                Tool(
                    linear_list_documents,
                    name="linear_list_documents",
                    description=(
                        "Enumerate Linear project documents for this pot's "
                        "connected team as compact refs (id/title/updated_at), "
                        "newest first. Same backfill window + item cap. Seed "
                        "the todo list, then hydrate each via "
                        "linear_get_document. Standalone (non-project) docs are "
                        "out of team scope by design."
                    ),
                )
            )
        if callable(getattr(fetcher, "get_document", None)):
            tools.append(
                Tool(
                    linear_get_document,
                    name="linear_get_document",
                    description=(
                        "Fetch one Linear document's detail by id (title, "
                        "markdown content, url, owning project, creator). Seed "
                        "lasting docs as the canonical label Document."
                    ),
                )
            )
        return tools

    return _builder


__all__ = ["build_linear_tools"]
