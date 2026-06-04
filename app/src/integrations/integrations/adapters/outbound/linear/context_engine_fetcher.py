"""LinearIssueFetcher adapter for the context-engine.

Multi-tenant: resolves the Linear access token per call by walking
``pot_id → context_graph_pot_sources → integrations`` and decrypting the
stored token. Falls back to ``LINEAR_API_KEY`` when no pot context is
available (dev / standalone container).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_source_model import (
    SOURCE_KIND_ISSUE_TRACKER_TEAM,
    ContextGraphPotSource,
)
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
from integrations.adapters.outbound.linear.adapter import (
    _DOCUMENT_DETAIL,
    _ISSUE_DETAIL,
    _PROJECT_DETAIL,
)
from integrations.adapters.outbound.linear.graphql_client import (
    LinearGraphQLError,
    linear_graphql,
)
from integrations.adapters.outbound.postgres.integration_model import Integration
from integrations.domain.integrations_schema import AuthData, IntegrationType

logger = logging.getLogger(__name__)


class ContextEngineLinearFetcher:
    """Resolve and call Linear GraphQL with the right token for a given pot."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get_issue(
        self,
        issue_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        token = self._resolve_token(pot_id)
        if not token:
            logger.info(
                "linear fetcher: no token available for pot=%s issue=%s",
                pot_id,
                issue_id,
            )
            return None
        try:
            data = linear_graphql(token, _ISSUE_DETAIL, {"id": issue_id})
        except LinearGraphQLError as exc:
            msg = str(exc).lower()
            if "401" in msg or "unauthorized" in msg or "forbidden" in msg or "403" in msg:
                raise PermissionError(str(exc)) from exc
            if "not found" in msg or "entitynotfound" in msg:
                return None
            raise
        issue = (data or {}).get("issue")
        return issue if isinstance(issue, dict) else None

    def list_issues(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate compact issue refs for a pot's Linear team.

        Plain enumerator: ``updated_after`` and ``limit`` are applied as
        given. The bounded-backfill window/cap policy is the *caller's*
        concern (the context-engine ``linear_list_issues`` tool computes
        them) so this integrations-side adapter stays free of context-engine
        domain imports. Returns ``{id, identifier, updated_at}`` dicts; the
        agent hydrates each via ``linear_get_issue``.
        """
        token = self._resolve_token(pot_id)
        if not token:
            logger.info("linear fetcher: no token for pot=%s (list_issues)", pot_id)
            return []
        resolved_team = team_id or self._team_id_for_pot(pot_id)
        if not resolved_team:
            logger.info("linear fetcher: no team scope for pot=%s", pot_id)
            return []
        from integrations.adapters.outbound.linear.adapter import (
            LinearIssueTrackerAdapter,
        )

        adapter = LinearIssueTrackerAdapter(token)
        out: list[dict[str, Any]] = []
        try:
            for ref in adapter.iter_issues(
                scope={"team_id": resolved_team},
                updated_after=updated_after,
            ):
                upd = getattr(ref, "updated_at", None)
                out.append(
                    {
                        "id": getattr(ref, "id", None),
                        "identifier": getattr(ref, "identifier", None),
                        "updated_at": upd.isoformat() if upd else None,
                    }
                )
                if limit is not None and len(out) >= limit:
                    break
        except Exception:
            logger.exception("linear fetcher: list_issues failed pot=%s", pot_id)
            return out
        return out

    def get_project(
        self,
        project_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self._detail(_PROJECT_DETAIL, project_id, "project", pot_id)

    def get_document(
        self,
        document_id: str,
        *,
        pot_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self._detail(_DOCUMENT_DETAIL, document_id, "document", pot_id)

    def _detail(
        self,
        query: str,
        node_id: str,
        result_key: str,
        pot_id: str | None,
    ) -> dict[str, Any] | None:
        """Shared single-node fetch (same auth/not-found contract as get_issue)."""
        token = self._resolve_token(pot_id)
        if not token:
            logger.info(
                "linear fetcher: no token for pot=%s %s=%s",
                pot_id,
                result_key,
                node_id,
            )
            return None
        try:
            data = linear_graphql(token, query, {"id": node_id})
        except LinearGraphQLError as exc:
            msg = str(exc).lower()
            if (
                "401" in msg
                or "unauthorized" in msg
                or "forbidden" in msg
                or "403" in msg
            ):
                raise PermissionError(str(exc)) from exc
            if "not found" in msg or "entitynotfound" in msg:
                return None
            raise
        node = (data or {}).get(result_key)
        return node if isinstance(node, dict) else None

    def list_projects(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate compact project refs for a pot's Linear team.

        Plain enumerator (window/cap applied by the caller). Returns
        ``{id, name, updated_at}``; hydrate via ``linear_get_project``.
        """
        def _to_dict(r: Any) -> dict[str, Any]:
            upd = getattr(r, "updated_at", None)
            return {
                "id": getattr(r, "id", None),
                "name": getattr(r, "name", None),
                "updated_at": upd.isoformat() if upd else None,
            }

        return self._list(
            "iter_projects",
            _to_dict,
            pot_id=pot_id,
            team_id=team_id,
            updated_after=updated_after,
            limit=limit,
        )

    def list_documents(
        self,
        *,
        pot_id: str | None = None,
        team_id: str | None = None,
        updated_after: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate compact document refs for a pot's Linear team.

        Plain enumerator (window/cap applied by the caller). Returns
        ``{id, title, updated_at}``; hydrate via ``linear_get_document``.
        """
        def _to_dict(r: Any) -> dict[str, Any]:
            upd = getattr(r, "updated_at", None)
            return {
                "id": getattr(r, "id", None),
                "title": getattr(r, "title", None),
                "updated_at": upd.isoformat() if upd else None,
            }

        return self._list(
            "iter_documents",
            _to_dict,
            pot_id=pot_id,
            team_id=team_id,
            updated_after=updated_after,
            limit=limit,
        )

    def _list(
        self,
        iter_method: str,
        mapper,
        *,
        pot_id: str | None,
        team_id: str | None,
        updated_after: datetime | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        """Shared enumerate-and-cap over an adapter ``iter_*`` method."""
        token = self._resolve_token(pot_id)
        if not token:
            logger.info(
                "linear fetcher: no token for pot=%s (%s)", pot_id, iter_method
            )
            return []
        resolved_team = team_id or self._team_id_for_pot(pot_id)
        if not resolved_team:
            logger.info("linear fetcher: no team scope for pot=%s", pot_id)
            return []
        from integrations.adapters.outbound.linear.adapter import (
            LinearIssueTrackerAdapter,
        )

        adapter = LinearIssueTrackerAdapter(token)
        out: list[dict[str, Any]] = []
        try:
            for ref in getattr(adapter, iter_method)(
                scope={"team_id": resolved_team},
                updated_after=updated_after,
            ):
                out.append(mapper(ref))
                if limit is not None and len(out) >= limit:
                    break
        except Exception:
            logger.exception(
                "linear fetcher: %s failed pot=%s", iter_method, pot_id
            )
            return out
        return out

    def _resolve_token(self, pot_id: str | None) -> str | None:
        if pot_id:
            token = self._token_for_pot(pot_id)
            if token:
                return token
        env_token = (os.getenv("LINEAR_API_KEY") or "").strip()
        return env_token or None

    def _linear_source_row(self, pot_id: str) -> ContextGraphPotSource | None:
        return (
            self._db.query(ContextGraphPotSource)
            .filter(
                ContextGraphPotSource.pot_id == pot_id,
                ContextGraphPotSource.provider == "linear",
                ContextGraphPotSource.source_kind == SOURCE_KIND_ISSUE_TRACKER_TEAM,
                ContextGraphPotSource.integration_id.isnot(None),
            )
            .order_by(ContextGraphPotSource.created_at.desc())
            .first()
        )

    def _team_id_for_pot(self, pot_id: str | None) -> str | None:
        if not pot_id:
            return None
        row = self._linear_source_row(pot_id)
        if row is None or not row.scope_json:
            return None
        try:
            scope = json.loads(row.scope_json)
        except (TypeError, ValueError):
            return None
        team_id = scope.get("team_id") if isinstance(scope, dict) else None
        return str(team_id) if team_id else None

    def _token_for_pot(self, pot_id: str) -> str | None:
        row = self._linear_source_row(pot_id)
        if row is None:
            return None
        integration = (
            self._db.query(Integration)
            .filter(
                Integration.integration_id == row.integration_id,
                Integration.integration_type == IntegrationType.LINEAR.value,
                Integration.active.is_(True),
            )
            .first()
        )
        if integration is None:
            return None
        try:
            auth = AuthData.model_validate(integration.auth_data or {})
        except Exception:
            logger.warning(
                "linear fetcher: invalid auth_data on integration %s",
                integration.integration_id,
            )
            return None
        if not auth.access_token:
            return None
        try:
            return decrypt_token(auth.access_token)
        except Exception:
            logger.warning(
                "linear fetcher: token decrypt failed for integration %s",
                integration.integration_id,
            )
            return None
