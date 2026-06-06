"""Unit tests for ``IntegrationsService.save_linear_integration``.

Focus: the IntegrityError → typed-exception translation path that closes
the race between ``check_existing_linear_integration`` and the actual
``INSERT``. Before the per-user composite unique index landed, this race
caused raw ``psycopg2`` errors to leak through the OAuth callback into
the redirect URL surfaced to the browser.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from integrations.application.integrations_service import IntegrationsService
from integrations.domain.exceptions import LinearOrganizationAlreadyIntegratedError
from integrations.domain.integrations_schema import LinearSaveRequest


def _make_save_request() -> LinearSaveRequest:
    return LinearSaveRequest(
        code="abcdef1234567890",
        redirect_uri="http://localhost:8000/api/v1/integrations/linear/callback",
        instance_name="Linear Integration",
        integration_type="linear",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _stub_linear_oauth(service: IntegrationsService) -> None:
    """Pretend the OAuth token exchange + userinfo call succeed."""
    service.linear_oauth = MagicMock()
    service.linear_oauth.exchange_code_for_tokens = AsyncMock(
        return_value={
            "access_token": "live_tok",
            "token_type": "Bearer",
            "scope": "read",
            "expires_at": 9999999999,
        }
    )
    service.linear_oauth.get_user_info_from_api = AsyncMock(
        return_value={
            "id": "linear-user-1",
            "name": "Alice",
            "email": "alice@example.com",
            "organization": {
                "id": "org-xyz",
                "name": "Acme",
                "urlKey": "acme",
            },
        }
    )


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def service(mock_db):
    svc = IntegrationsService(mock_db)
    _stub_linear_oauth(svc)
    return svc


class TestSaveLinearIntegrationDedupRace:
    """Cover the per-user dedup paths around the commit."""

    @pytest.mark.asyncio
    async def test_explicit_check_returns_existing__raises_typed(self, service, mock_db):
        """Pre-commit check finds an existing row → typed exception, no commit."""
        service.check_existing_linear_integration = AsyncMock(
            return_value={"integration_id": "existing-int-1"}
        )

        with pytest.raises(LinearOrganizationAlreadyIntegratedError) as exc_info:
            await service.save_linear_integration(_make_save_request(), "user-A")
        assert exc_info.value.integration_id == "existing-int-1"
        # We bailed before reaching the INSERT.
        mock_db.commit.assert_not_called()
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_integrity_error_with_surviving_row__raises_typed(
        self, service, mock_db
    ):
        """Race: pre-check is clean, INSERT trips the composite unique, then
        re-querying finds the row the concurrent transaction wrote. We must
        rollback and re-raise as a typed exception so the router can show
        the friendly already-connected screen instead of a 500/raw SQL."""
        # First call (pre-commit) clean; second call (post-IntegrityError) finds it.
        service.check_existing_linear_integration = AsyncMock(
            side_effect=[None, {"integration_id": "racer-int-1"}]
        )
        mock_db.commit.side_effect = IntegrityError(
            "INSERT INTO integrations",
            params={},
            orig=Exception(
                "duplicate key value violates unique constraint "
                '"uq_integrations_unique_identifier_per_user"'
            ),
        )

        with pytest.raises(LinearOrganizationAlreadyIntegratedError) as exc_info:
            await service.save_linear_integration(_make_save_request(), "user-A")
        assert exc_info.value.integration_id == "racer-int-1"
        mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_integrity_error_with_no_surviving_row__generic_exception(
        self, service, mock_db
    ):
        """Pathological: IntegrityError fires but we can't find the conflicting
        row afterwards (e.g. it was rolled back in another tx). We must NOT
        leak the raw DB exception text — emit a generic message instead."""
        service.check_existing_linear_integration = AsyncMock(
            side_effect=[None, None]
        )
        mock_db.commit.side_effect = IntegrityError(
            "INSERT INTO integrations",
            params={},
            orig=Exception("constraint internals nobody should see"),
        )

        with pytest.raises(Exception) as exc_info:
            await service.save_linear_integration(_make_save_request(), "user-A")

        # Must not be the typed dedup (no row to point at), but must also
        # not be the raw IntegrityError.
        assert not isinstance(
            exc_info.value, LinearOrganizationAlreadyIntegratedError
        )
        assert not isinstance(exc_info.value, IntegrityError)
        # The catch-all wraps the message into the "Failed to save Linear
        # integration: …" form; verify our generic phrasing landed there.
        message = str(exc_info.value)
        assert "constraint internals" not in message
        assert "could not be saved" in message or "conflict" in message
        mock_db.rollback.assert_called_once()
