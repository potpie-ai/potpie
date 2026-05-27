"""Unit tests for IntegrationsService."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from integrations.application.integrations_service import IntegrationsService
from integrations.domain.integrations_schema import (
    IntegrationType,
    IntegrationStatus,
    AuthData,
    ScopeData,
    IntegrationMetadata,
)


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def service(mock_db):
    return IntegrationsService(mock_db)


class TestIntegrationsServiceDbToDict:
    def test_db_to_dict_basic(self, service):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-1"
        mock_integration.name = "Test"
        mock_integration.integration_type = "sentry"
        mock_integration.status = "active"
        mock_integration.active = True
        mock_integration.auth_data = {"access_token": "tok"}
        mock_integration.scope_data = {"org_slug": "my-org"}
        mock_integration.integration_metadata = {"instance_name": "inst"}
        mock_integration.unique_identifier = "uid-1"
        mock_integration.created_by = "user-1"
        mock_integration.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_integration.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)

        out = service._db_to_dict(mock_integration)
        assert out["integration_id"] == "int-1"
        assert out["name"] == "Test"
        assert out["integration_type"] == "sentry"
        assert out["status"] == "active"
        assert out["active"] is True
        assert out["auth_data"] == {"access_token": "tok"}
        assert out["scope_data"] == {"org_slug": "my-org"}
        assert out["metadata"] == {"instance_name": "inst"}
        assert out["unique_identifier"] == "uid-1"
        assert out["created_by"] == "user-1"
        assert "2025-01-01" in out["created_at"]
        assert "2025-01-02" in out["updated_at"]

    def test_db_to_dict_none_json_fields(self, service):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-2"
        mock_integration.name = "Test2"
        mock_integration.integration_type = "jira"
        mock_integration.status = "inactive"
        mock_integration.active = False
        mock_integration.auth_data = None
        mock_integration.scope_data = None
        mock_integration.integration_metadata = None
        mock_integration.unique_identifier = "uid-2"
        mock_integration.created_by = "user-2"
        mock_integration.created_at = datetime(2025, 2, 1, tzinfo=timezone.utc)
        mock_integration.updated_at = datetime(2025, 2, 2, tzinfo=timezone.utc)

        out = service._db_to_dict(mock_integration)
        assert out["auth_data"] == {}
        assert out["scope_data"] == {}
        assert out["metadata"] == {}

    def test_db_to_dict_none_dates(self, service):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-3"
        mock_integration.name = "Test3"
        mock_integration.integration_type = "linear"
        mock_integration.status = "active"
        mock_integration.active = True
        mock_integration.auth_data = {}
        mock_integration.scope_data = {}
        mock_integration.integration_metadata = {}
        mock_integration.unique_identifier = "uid-3"
        mock_integration.created_by = "user-3"
        mock_integration.created_at = None
        mock_integration.updated_at = None

        out = service._db_to_dict(mock_integration)
        assert out["created_at"] is None
        assert out["updated_at"] is None


class TestIntegrationsServiceDbToSchema:
    def test_db_to_schema_full(self, service):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-s1"
        mock_integration.name = "Sentry"
        mock_integration.integration_type = "sentry"
        mock_integration.status = "active"
        mock_integration.active = True
        mock_integration.auth_data = {"access_token": "at", "refresh_token": "rt"}
        mock_integration.scope_data = {"org_slug": "org1"}
        mock_integration.integration_metadata = {"instance_name": "my-sentry"}
        mock_integration.unique_identifier = "uid-s1"
        mock_integration.created_by = "user-s1"
        mock_integration.created_at = datetime(2025, 3, 1, tzinfo=timezone.utc)
        mock_integration.updated_at = datetime(2025, 3, 2, tzinfo=timezone.utc)

        schema = service._db_to_schema(mock_integration)
        assert schema.integration_id == "int-s1"
        assert schema.name == "Sentry"
        assert schema.integration_type == IntegrationType.SENTRY
        assert schema.status == IntegrationStatus.ACTIVE
        assert schema.active is True
        assert schema.auth_data.access_token == "at"
        assert schema.scope_data.org_slug == "org1"
        assert schema.metadata.instance_name == "my-sentry"
        assert schema.unique_identifier == "uid-s1"
        assert schema.created_by == "user-s1"

    def test_db_to_schema_empty_auth_scope_metadata(self, service):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-s2"
        mock_integration.name = "Empty"
        mock_integration.integration_type = "confluence"
        mock_integration.status = "pending"
        mock_integration.active = False
        mock_integration.auth_data = None
        mock_integration.scope_data = None
        mock_integration.integration_metadata = None
        mock_integration.unique_identifier = "uid-s2"
        mock_integration.created_by = "user-s2"
        mock_integration.created_at = None
        mock_integration.updated_at = None

        schema = service._db_to_schema(mock_integration)
        assert schema.auth_data.access_token is None
        assert schema.scope_data.org_slug is None
        assert schema.metadata.instance_name == ""


class TestIntegrationsServiceSentryStatus:
    @pytest.mark.asyncio
    async def test_get_sentry_integration_status_not_connected(self, service):
        service.sentry_oauth.get_user_info = MagicMock(return_value=None)
        result = await service.get_sentry_integration_status("user-1")
        assert result.user_id == "user-1"
        assert result.is_connected is False

    @pytest.mark.asyncio
    async def test_get_sentry_integration_status_connected(self, service):
        service.sentry_oauth.get_user_info = MagicMock(
            return_value={"scope": "read", "expires_at": "2025-12-31T00:00:00"}
        )
        result = await service.get_sentry_integration_status("user-2")
        assert result.user_id == "user-2"
        assert result.is_connected is True
        assert result.scope == "read"


class TestIntegrationsServiceGetIntegrationsByUser:
    @pytest.mark.asyncio
    async def test_get_integrations_by_user_empty(self, service, mock_db):
        mock_db.query.return_value.filter.return_value.all.return_value = []
        result = await service.get_integrations_by_user("user-1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_integrations_by_user_returns_dict(self, service, mock_db):
        mock_integration = MagicMock()
        mock_integration.integration_id = "int-1"
        mock_integration.name = "Test"
        mock_integration.integration_type = "sentry"
        mock_integration.status = "active"
        mock_integration.active = True
        mock_integration.auth_data = {}
        mock_integration.scope_data = {}
        mock_integration.integration_metadata = {}
        mock_integration.unique_identifier = "uid-1"
        mock_integration.created_by = "user-1"
        mock_integration.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        mock_integration.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_integration]
        result = await service.get_integrations_by_user("user-1")
        assert "int-1" in result
        assert result["int-1"]["name"] == "Test"
