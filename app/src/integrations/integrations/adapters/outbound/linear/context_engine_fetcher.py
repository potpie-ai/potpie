"""LinearIssueFetcher adapter for the context-engine.

Multi-tenant: resolves the Linear access token per call by walking
``pot_id → context_graph_pot_sources → integrations`` and decrypting the
stored token. Falls back to ``LINEAR_API_KEY`` when no pot context is
available (dev / standalone container).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_source_model import (
    SOURCE_KIND_ISSUE_TRACKER_TEAM,
    ContextGraphPotSource,
)
from integrations.adapters.outbound.crypto.token_encryption import decrypt_token
from integrations.adapters.outbound.linear.adapter import _ISSUE_DETAIL
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

    def _resolve_token(self, pot_id: str | None) -> str | None:
        if pot_id:
            token = self._token_for_pot(pot_id)
            if token:
                return token
        env_token = (os.getenv("LINEAR_API_KEY") or "").strip()
        return env_token or None

    def _token_for_pot(self, pot_id: str) -> str | None:
        row = (
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
