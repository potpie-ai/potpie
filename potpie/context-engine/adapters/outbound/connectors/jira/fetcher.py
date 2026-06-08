"""Port for Jira source reads used by the Jira agent tools.

Mirrors :mod:`adapters.outbound.connectors.linear.fetcher`: the context-engine
does not own a Jira REST client — the ``integrations/`` module owns Atlassian
OAuth + HTTP (see ``integrations/adapters/outbound/oauth/jira_oauth.py``). This
Protocol is the narrow surface the ``build_jira_tools`` agent tools depend on,
so tests use a fake and hosts can plug an OAuth-backed adapter without dragging
the integrations DB/Celery into the reconciliation path.

``pot_id`` lets multi-tenant adapters resolve the right per-pot Jira
credentials. ``get_issue`` returns ``None`` for "not found"; raising is
reserved for transport/auth errors so the tool surfaces a warning instead of
inventing facts. Enumeration / changelog methods are OPTIONAL capabilities,
surfaced as agent tools only when the wired fetcher implements them.
"""

from __future__ import annotations

from typing import Any, Protocol


class JiraIssueFetcher(Protocol):
    """Fetch Jira issues / epics for a pot's connected Jira project."""

    def get_issue(
        self,
        issue_key: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        """One issue or epic's detail by key (e.g. ``PROJ-123``)."""
        ...

    # --- OPTIONAL: JQL enumeration (diff-sync candidate refs) -------------
    def search_issues(
        self,
        jql: str,
        *,
        pot_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a JQL search; return compact refs.

        The agent builds the JQL (e.g.
        ``project = PROJ AND updated >= "2026-06-01" ORDER BY updated ASC``)
        so the diff-sync cursor rides inside ``jql``. Returns
        ``{key, summary, issuetype, updated_at}`` dicts; ``limit`` is applied
        as given (the caller clamps it).
        """
        ...

    # --- OPTIONAL: field-level change history -----------------------------
    def get_issue_changelog(
        self,
        issue_key: str,
        *,
        pot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Changelog entries for one issue (what fields changed, when, by whom)."""
        ...

    def bulk_fetch_changelogs(
        self,
        issue_keys: list[str],
        *,
        pot_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Changelogs for several issues at once, keyed by issue key."""
        ...
