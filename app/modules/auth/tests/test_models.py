"""
Unit tests for auth models
"""

import pytest
from datetime import datetime, timedelta

from app.modules.auth.auth_provider_model import (
    UserAuthProvider,
    PendingProviderLink,
    OrganizationSSOConfig,
    AuthAuditLog,
)
from app.modules.users.user_model import User


class TestUserModel:
    """Test User model enhancements"""

    def test_user_has_auth_providers_relationship(self, db_session, test_user):
        """Test User has auth_providers relationship"""
        assert hasattr(test_user, "auth_providers")
        assert isinstance(test_user.auth_providers, list)

    def test_get_primary_provider(self, db_session, test_user_with_github):
        """Test User.get_primary_provider()"""
        primary = test_user_with_github.get_primary_provider()
        
        assert primary is not None
        assert primary.provider_type == "firebase_github"
        assert primary.is_primary == True

    def test_get_primary_provider_none(self, db_session, test_user):
        """Test get_primary_provider when user has no providers"""
        primary = test_user.get_primary_provider()
        assert primary is None

    def test_has_provider(self, db_session, test_user_with_github):
        """Test User.has_provider()"""
        assert test_user_with_github.has_provider("firebase_github") == True
        assert test_user_with_github.has_provider("sso_google") == False

    def test_organization_fields(self, db_session):
        """Test User has organization fields"""
        user = User(
            uid="test-123",
            email="user@company.com",
            display_name="Test",
            email_verified=True,
            organization="company.com",
            organization_name="Test Company",
            created_at=datetime.utcnow(),
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.organization == "company.com"
        assert user.organization_name == "Test Company"


class TestUserAuthProvider:
    """Test UserAuthProvider model"""

    def test_create_provider(self, db_session, test_user):
        """Test creating a provider"""
        provider = UserAuthProvider(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-123",
            provider_data={"email": "test@example.com"},
            is_primary=True,
            linked_at=datetime.utcnow(),
        )
        db_session.add(provider)
        db_session.commit()
        
        assert provider.id is not None
        assert provider.user_id == test_user.uid
        assert provider.provider_type == "sso_google"

    def test_unique_user_provider_constraint(self, db_session, test_user_with_github):
        """Test unique constraint on (user_id, provider_type)"""
        # Try to add duplicate provider for same user
        duplicate = UserAuthProvider(
            user_id=test_user_with_github.uid,
            provider_type="firebase_github",  # Same as existing
            provider_uid="different-uid",
            linked_at=datetime.utcnow(),
        )
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_unique_provider_uid_constraint(self, db_session, test_user, test_user_with_github):
        """Test unique constraint on (provider_type, provider_uid)"""
        # Try to use same provider_uid for different user
        duplicate = UserAuthProvider(
            user_id=test_user.uid,  # Different user
            provider_type="firebase_github",
            provider_uid="github-123",  # Same as existing
            linked_at=datetime.utcnow(),
        )
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_cascade_delete(self, db_session, test_user_with_github):
        """Test providers are deleted when user is deleted"""
        user_id = test_user_with_github.uid
        
        # Verify provider exists
        provider = db_session.query(UserAuthProvider).filter_by(
            user_id=user_id
        ).first()
        assert provider is not None
        
        # Delete user
        db_session.delete(test_user_with_github)
        db_session.commit()
        
        # Verify provider is deleted
        provider = db_session.query(UserAuthProvider).filter_by(
            user_id=user_id
        ).first()
        assert provider is None

    def test_oauth_tokens(self, db_session, test_user):
        """Test storing OAuth tokens"""
        provider = UserAuthProvider(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-123",
            access_token="access-token-123",
            refresh_token="refresh-token-456",
            token_expires_at=datetime.utcnow() + timedelta(hours=1),
            linked_at=datetime.utcnow(),
        )
        db_session.add(provider)
        db_session.commit()
        
        assert provider.access_token == "access-token-123"
        assert provider.refresh_token == "refresh-token-456"
        assert provider.token_expires_at is not None


class TestPendingProviderLink:
    """Test PendingProviderLink model"""

    def test_create_pending_link(self, db_session, test_user):
        """Test creating a pending link"""
        link = PendingProviderLink(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-789",
            provider_data={"email": "test@example.com"},
            token="unique-token-123",
            expires_at=datetime.utcnow() + timedelta(minutes=15),
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0",
        )
        db_session.add(link)
        db_session.commit()
        
        assert link.id is not None
        assert link.token == "unique-token-123"

    def test_unique_token_constraint(self, db_session, test_user, pending_link):
        """Test unique constraint on token"""
        duplicate = PendingProviderLink(
            user_id=test_user.uid,
            provider_type="sso_azure",
            provider_uid="azure-123",
            provider_data={},
            token=pending_link.token,  # Same token
            expires_at=datetime.utcnow() + timedelta(minutes=15),
        )
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_expiration(self, db_session, test_user):
        """Test checking if link is expired"""
        link = PendingProviderLink(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-789",
            provider_data={},
            token="token-123",
            expires_at=datetime.utcnow() - timedelta(minutes=1),  # Expired
        )
        db_session.add(link)
        db_session.commit()
        
        assert link.expires_at < datetime.utcnow()

    def test_cascade_delete_with_user(self, db_session, test_user, pending_link):
        """Test pending links are deleted when user is deleted"""
        token = pending_link.token
        
        # Delete user
        db_session.delete(test_user)
        db_session.commit()
        
        # Verify link is deleted
        link = db_session.query(PendingProviderLink).filter_by(
            token=token
        ).first()
        assert link is None


class TestOrganizationSSOConfig:
    """Test OrganizationSSOConfig model"""

    def test_create_config(self, db_session):
        """Test creating SSO config"""
        config = OrganizationSSOConfig(
            domain="company.com",
            organization_name="Test Company",
            sso_provider="google",
            sso_config={
                "client_id": "test-client-id",
                "hosted_domain": "company.com",
            },
            enforce_sso=True,
            allow_other_providers=False,
            configured_at=datetime.utcnow(),
            is_active=True,
        )
        db_session.add(config)
        db_session.commit()
        
        assert config.id is not None
        assert config.domain == "company.com"
        assert config.enforce_sso == True

    def test_unique_domain_constraint(self, db_session, org_sso_config):
        """Test unique constraint on domain"""
        duplicate = OrganizationSSOConfig(
            domain=org_sso_config.domain,  # Same domain
            sso_provider="azure",
            sso_config={},
            configured_at=datetime.utcnow(),
        )
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_sso_config_jsonb(self, db_session, org_sso_config):
        """Test JSONB storage for sso_config"""
        assert isinstance(org_sso_config.sso_config, dict)
        assert org_sso_config.sso_config.get("client_id") == "test-client-id"

    def test_query_active_configs(self, db_session, org_sso_config):
        """Test querying only active configs"""
        # Create inactive config
        inactive = OrganizationSSOConfig(
            domain="inactive.com",
            sso_provider="google",
            sso_config={},
            configured_at=datetime.utcnow(),
            is_active=False,
        )
        db_session.add(inactive)
        db_session.commit()
        
        # Query active only
        active_configs = db_session.query(OrganizationSSOConfig).filter_by(
            is_active=True
        ).all()
        
        assert len(active_configs) == 1
        assert active_configs[0].domain == "company.com"


class TestAuthAuditLog:
    """Test AuthAuditLog model"""

    def test_create_audit_log(self, db_session, test_user):
        """Test creating an audit log entry"""
        log = AuthAuditLog(
            user_id=test_user.uid,
            event_type="login",
            provider_type="firebase_github",
            status="success",
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0",
            created_at=datetime.utcnow(),
        )
        db_session.add(log)
        db_session.commit()
        
        assert log.id is not None
        assert log.event_type == "login"
        assert log.status == "success"

    def test_failed_login_log(self, db_session):
        """Test logging failed login (no user_id)"""
        log = AuthAuditLog(
            user_id=None,  # Failed login, no user
            event_type="failed_login",
            provider_type="sso_google",
            status="failure",
            error_message="Invalid token",
            ip_address="192.168.1.1",
            created_at=datetime.utcnow(),
        )
        db_session.add(log)
        db_session.commit()
        
        assert log.id is not None
        assert log.user_id is None
        assert log.error_message == "Invalid token"

    def test_extra_data_jsonb(self, db_session, test_user):
        """Test JSONB storage for extra_data"""
        log = AuthAuditLog(
            user_id=test_user.uid,
            event_type="custom_event",
            status="success",
            extra_data={
                "custom_field": "value",
                "nested": {"key": "value"},
            },
            created_at=datetime.utcnow(),
        )
        db_session.add(log)
        db_session.commit()
        
        assert isinstance(log.extra_data, dict)
        assert log.extra_data["custom_field"] == "value"

    def test_query_by_event_type(self, db_session, test_user):
        """Test querying logs by event type"""
        # Create multiple logs
        for event in ["login", "login", "link_provider"]:
            log = AuthAuditLog(
                user_id=test_user.uid,
                event_type=event,
                status="success",
                created_at=datetime.utcnow(),
            )
            db_session.add(log)
        db_session.commit()
        
        # Query login events
        login_logs = db_session.query(AuthAuditLog).filter_by(
            event_type="login"
        ).all()
        
        assert len(login_logs) == 2

    def test_query_by_time_range(self, db_session, test_user):
        """Test querying logs by time range"""
        old_log = AuthAuditLog(
            user_id=test_user.uid,
            event_type="login",
            status="success",
            created_at=datetime.utcnow() - timedelta(days=2),
        )
        recent_log = AuthAuditLog(
            user_id=test_user.uid,
            event_type="login",
            status="success",
            created_at=datetime.utcnow(),
        )
        db_session.add_all([old_log, recent_log])
        db_session.commit()
        
        # Query last 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_logs = db_session.query(AuthAuditLog).filter(
            AuthAuditLog.created_at >= cutoff
        ).all()
        
        assert len(recent_logs) == 1

