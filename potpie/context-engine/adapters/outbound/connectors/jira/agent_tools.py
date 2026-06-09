"""Jira agent tools: fetch issue / epic / changelog data via ``JiraIssueFetcher``.

Mirrors :mod:`adapters.outbound.connectors.linear.agent_tools`. The fetcher
resolves the Jira access token *per pot* (walking the pot's connected Jira
integration and decrypting the stored OAuth token), so reads run against
whatever Jira site the pot's owner connected — never a shared service key. The
host wires this in via
``PydanticDeepReconciliationAgent.add_extra_tools([build_jira_tools(...)])``.

Enumeration (``jira_search_issues``) and changelog tools are OPTIONAL fetcher
capabilities: each is surfaced only when the wired fetcher implements it
(``getattr`` capability check), so a minimal single-issue resolver never
advertises a tool that can't run.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from adapters.outbound.connectors.jira.fetcher import JiraIssueFetcher
from domain.error_redaction import safe_error

logger = logging.getLogger(__name__)


def build_jira_tools(
    fetcher: JiraIssueFetcher,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch tool builder that exposes Jira read endpoints.

    Args:
        fetcher: Resolves Jira reads for a given pot. Typically a per-pot
            OAuth-backed adapter over the integrations module's ``JiraOAuth``.

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
                    "pydantic-ai/pydantic-deep Tool not importable; skipping jira tools"
                )
                return []

        pot_id = getattr(state, "pot_id", None)

        def jira_get_issue(issue_key: str) -> dict[str, Any]:
            """Fetch one Jira issue or epic's detail by key (e.g. ``PROJ-123``)."""
            try:
                issue = fetcher.get_issue(issue_key, pot_id=pot_id)
            except PermissionError as exc:
                return {"error": "jira_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("jira_get_issue %s failed", issue_key)
                return {"error": safe_error(exc)}
            if issue is None:
                return {"found": False, "issue_key": issue_key}
            return {"found": True, "issue": issue}

        def jira_search_issues(
            jql: str,
            limit: int | None = None,
        ) -> dict[str, Any]:
            """Run a JQL search; return compact issue refs for the todo list.

            The diff-sync cursor rides inside ``jql`` (e.g. ``project = PROJ
            AND updated >= "2026-06-01" ORDER BY updated ASC``). Bounded by the
            shared backfill item cap. Hydrate each ref via ``jira_get_issue``.
            """
            from domain.backfill_window import clamp_backfill_limit

            try:
                issues = fetcher.search_issues(
                    jql,
                    pot_id=pot_id,
                    limit=clamp_backfill_limit(limit),
                )
            except PermissionError as exc:
                return {"error": "jira_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("jira_search_issues failed pot=%s", pot_id)
                return {"error": safe_error(exc)}
            return {"count": len(issues), "issues": issues}

        def jira_get_issue_changelog(issue_key: str) -> dict[str, Any]:
            """Changelog entries for one issue (fields changed, when, by whom)."""
            try:
                entries = fetcher.get_issue_changelog(issue_key, pot_id=pot_id)
            except PermissionError as exc:
                return {"error": "jira_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("jira_get_issue_changelog %s failed", issue_key)
                return {"error": safe_error(exc)}
            return {"issue_key": issue_key, "count": len(entries), "changelog": entries}

        def jira_bulk_fetch_changelogs(issue_keys: list[str]) -> dict[str, Any]:
            """Changelogs for several issues at once, keyed by issue key."""
            try:
                by_key = fetcher.bulk_fetch_changelogs(
                    list(issue_keys), pot_id=pot_id
                )
            except PermissionError as exc:
                return {"error": "jira_auth_failed", "message": safe_error(exc)}
            except Exception as exc:
                logger.exception("jira_bulk_fetch_changelogs failed pot=%s", pot_id)
                return {"error": safe_error(exc)}
            return {"count": len(by_key), "changelogs": by_key}

        tools = [
            Tool(
                jira_get_issue,
                name="jira_get_issue",
                description=(
                    "Fetch one Jira issue or epic by key (e.g. 'PROJ-123') for "
                    "the Jira site connected to this pot. Returns the full "
                    "payload (summary, description, issuetype, status, "
                    "assignee, parent/epic link, labels, timestamps). Use it to "
                    "hydrate refs from jira_search_issues, or to ground issue "
                    "keys found in commits / PR bodies."
                ),
            ),
        ]
        if callable(getattr(fetcher, "search_issues", None)):
            tools.append(
                Tool(
                    jira_search_issues,
                    name="jira_search_issues",
                    description=(
                        "Run a JQL search against this pot's connected Jira "
                        "project and return compact issue refs "
                        "(key/summary/issuetype/updated_at). You build the JQL: "
                        "for diff-sync use 'project = <KEY> AND updated >= "
                        "<cursor> ORDER BY updated ASC'. Bounded by a hard item "
                        "cap. Seed the todo list, then hydrate each via "
                        "jira_get_issue."
                    ),
                )
            )
        if callable(getattr(fetcher, "get_issue_changelog", None)):
            tools.append(
                Tool(
                    jira_get_issue_changelog,
                    name="jira_get_issue_changelog",
                    description=(
                        "Fetch the field-level changelog for one Jira issue "
                        "(what changed, when, by whom). Use ONLY when the "
                        "issue-level payload does not explain what changed "
                        "since the last cursor — it is verbose."
                    ),
                )
            )
        if callable(getattr(fetcher, "bulk_fetch_changelogs", None)):
            tools.append(
                Tool(
                    jira_bulk_fetch_changelogs,
                    name="jira_bulk_fetch_changelogs",
                    description=(
                        "Fetch changelogs for several Jira issues at once "
                        "(issue_keys=[...]), keyed by issue key. Cheaper than "
                        "calling jira_get_issue_changelog per issue when "
                        "auditing many stale refs."
                    ),
                )
            )
        return tools

    return _builder


__all__ = ["build_jira_tools"]
