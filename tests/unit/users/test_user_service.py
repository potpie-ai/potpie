"""Unit tests for UserService (sync methods)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.modules.users.user_service import UserService
from app.modules.users.user_model import User
from app.modules.users.user_schema import CreateUser


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def user_service(mock_db):
    return UserService(mock_db)


class TestUserServiceGetUserByUid:
    def test_get_user_by_uid_found(self, user_service, mock_db):
        mock_user = User(uid="u1", email="u1@test.com", display_name="User One")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        result = user_service.get_user_by_uid("u1")
        assert result is mock_user
        mock_db.query.assert_called_with(User)

    def test_get_user_by_uid_not_found(self, user_service, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        result = user_service.get_user_by_uid("nonexistent")
        assert result is None

    def test_get_user_by_uid_exception_returns_none(self, user_service, mock_db):
        mock_db.query.return_value.filter.return_value.first.side_effect = Exception(
            "db error"
        )
        result = user_service.get_user_by_uid("u1")
        assert result is None


class TestUserServiceCreateUser:
    def test_create_user_success(self, user_service, mock_db):
        now = datetime.now(timezone.utc)
        details = CreateUser(
            uid="new-user",
            email="new@test.com",
            display_name="New User",
            email_verified=True,
            created_at=now,
            last_login_at=now,
            provider_info={},
            provider_username=None,
        )
        uid, message, error = user_service.create_user(details)
        assert uid == "new-user"
        assert "created" in message.lower()
        assert error is False
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_create_user_db_error(self, user_service, mock_db):
        now = datetime.now(timezone.utc)
        details = CreateUser(
            uid="new-user",
            email="new@test.com",
            display_name="New User",
            email_verified=True,
            created_at=now,
            last_login_at=now,
            provider_info={},
            provider_username=None,
        )
        mock_db.commit.side_effect = Exception("constraint error")
        uid, message, error = user_service.create_user(details)
        assert uid == ""
        assert error is True


class TestUserServiceGetUserIdByEmail:
    def test_get_user_id_by_email_found(self, user_service, mock_db):
        mock_user = User(uid="u1", email="u1@test.com", display_name="User One")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        result = user_service.get_user_id_by_email("u1@test.com")
        assert result == "u1"

    def test_get_user_id_by_email_not_found(self, user_service, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        result = user_service.get_user_id_by_email("nonexistent@test.com")
        assert result is None


class TestUserServiceSetupDummyUser:
    def test_setup_dummy_user_creates_user_once(self, user_service, monkeypatch):
        monkeypatch.setenv("defaultUsername", "dummy-user")
        create_result = ("dummy-user", "created", False)

        with patch.object(UserService, "get_user_by_uid", return_value=None):
            with patch.object(
                UserService,
                "create_user",
                return_value=create_result,
            ) as mock_create_user:
                user_service.setup_dummy_user()

        mock_create_user.assert_called_once()
        created_user = mock_create_user.call_args.args[0]
        assert isinstance(created_user, CreateUser)
        assert created_user.uid == "dummy-user"
        assert created_user.email == "defaultuser@potpie.ai"
        assert created_user.display_name == "Dummy User"
        assert created_user.email_verified is True
        assert created_user.provider_info == {"access_token": "dummy_token"}
        assert created_user.provider_username == "self"
        assert created_user.created_at is not None
        assert created_user.last_login_at is not None
        assert created_user.created_at == created_user.last_login_at

    def test_setup_dummy_user_does_not_create_when_user_exists(
        self, user_service, monkeypatch
    ):
        monkeypatch.setenv("defaultUsername", "dummy-user")
        existing_user = User(
            uid="dummy-user",
            email="defaultuser@potpie.ai",
            display_name="Dummy User",
        )

        with patch.object(UserService, "get_user_by_uid", return_value=existing_user):
            with patch.object(UserService, "create_user") as mock_create_user:
                user_service.setup_dummy_user()

        mock_create_user.assert_not_called()


class TestUserServiceGetUserIdsByEmails:
    def test_get_user_ids_by_emails_found(self, user_service, mock_db):
        mock_users = [
            User(uid="u1", email="u1@test.com", display_name="User One"),
            User(uid="u2", email="u2@test.com", display_name="User Two"),
        ]
        mock_db.query.return_value.filter.return_value.all.return_value = mock_users

        result = user_service.get_user_ids_by_emails(["u1@test.com", "u2@test.com"])

        assert result == ["u1", "u2"]

    def test_get_user_ids_by_emails_not_found_returns_empty_list(
        self, user_service, mock_db
    ):
        mock_db.query.return_value.filter.return_value.all.return_value = []

        result = user_service.get_user_ids_by_emails(["missing@test.com"])

        assert result == []


class TestUserServiceUpdateLastLogin:
    def test_update_last_login_found(self, user_service, mock_db):
        mock_user = MagicMock()
        mock_user.provider_info = {}
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        message, error = user_service.update_last_login("u1", "oauth-token-xyz")
        assert error is False
        assert "Updated" in message
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert mock_user.provider_info.get("access_token") == "oauth-token-xyz"

    def test_update_last_login_not_found(self, user_service, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        message, error = user_service.update_last_login("nonexistent", "token")
        assert error is True
        assert "not found" in message.lower()
